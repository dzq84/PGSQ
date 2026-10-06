from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

import numpy as np


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def corpus_wer(records: Iterable[dict], prefix: str = "reconstruction") -> float | None:
    errors = 0
    words = 0
    for record in records:
        counts = record.get("asr", {}).get(prefix)
        if counts is None:
            continue
        errors += int(counts["substitutions"])
        errors += int(counts["deletions"])
        errors += int(counts["insertions"])
        words += int(counts["reference_words"])
    return errors / words if words else None


def corpus_delta_wer(records: Iterable[dict]) -> float | None:
    records = list(records)
    reference = corpus_wer(records, "reference")
    reconstruction = corpus_wer(records, "reconstruction")
    if reference is None or reconstruction is None:
        return None
    return reconstruction - reference


def corpus_rtf(records: Iterable[dict], prefix: str) -> float | None:
    key = f"{prefix}_seconds"
    total_seconds = 0.0
    total_duration = 0.0
    for record in records:
        value = record.get(key)
        duration = record.get("duration")
        if value is None or duration is None:
            continue
        total_seconds += float(value)
        total_duration += float(duration)
    return total_seconds / total_duration if total_duration > 0 else None


def _speaker_groups(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        speaker = record.get("speaker_id")
        if not speaker:
            raise ValueError("Every record requires speaker_id for cluster bootstrap")
        groups[str(speaker)].append(record)
    return dict(groups)


def _continuous_statistic(records: list[dict], metric: str) -> float:
    values = [
        float(record["metrics"][metric])
        for record in records
        if record.get("metrics", {}).get(metric) is not None
    ]
    return float(mean(values)) if values else math.nan


def _wer_aggregate(records: Iterable[dict], prefix: str) -> tuple[float, float]:
    errors = 0
    words = 0
    for record in records:
        counts = record.get("asr", {}).get(prefix)
        if counts is None:
            continue
        errors += int(counts["substitutions"])
        errors += int(counts["deletions"])
        errors += int(counts["insertions"])
        words += int(counts["reference_words"])
    return float(errors), float(words)


def _rtf_aggregate(records: Iterable[dict], prefix: str) -> tuple[float, float]:
    key = f"{prefix}_seconds"
    seconds = 0.0
    duration = 0.0
    for record in records:
        value = record.get(key)
        item_duration = record.get("duration")
        if value is None or item_duration is None:
            continue
        seconds += float(value)
        duration += float(item_duration)
    return seconds, duration


def _metric_aggregate(records: list[dict], metric: str) -> tuple[float, ...]:
    if metric == "wer_reconstruction":
        return _wer_aggregate(records, "reconstruction")
    if metric == "wer_reference":
        return _wer_aggregate(records, "reference")
    if metric == "wer_delta":
        return (
            *_wer_aggregate(records, "reconstruction"),
            *_wer_aggregate(records, "reference"),
        )
    if metric in {"encoder_rtf", "decoder_rtf"}:
        return _rtf_aggregate(records, metric.removesuffix("_rtf"))
    values = [
        float(record["metrics"][metric])
        for record in records
        if record.get("metrics", {}).get(metric) is not None
        and math.isfinite(float(record["metrics"][metric]))
    ]
    return math.fsum(values), float(len(values))


def _metric_from_aggregate(metric: str, aggregate: Iterable[float]) -> float:
    values = tuple(float(value) for value in aggregate)
    if metric == "wer_delta":
        reconstruction = values[0] / values[1] if values[1] else math.nan
        reference = values[2] / values[3] if values[3] else math.nan
        return reconstruction - reference
    return values[0] / values[1] if values[1] else math.nan


def _resample_aggregates(
    aggregates: dict[str, tuple[float, ...]],
    speakers: list[str],
    *,
    resamples: int,
    seed: int,
    estimate: Callable[[tuple[float, ...]], float],
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    width = len(next(iter(aggregates.values())))
    for index in range(resamples):
        chosen = generator.choice(speakers, size=len(speakers), replace=True)
        combined = [0.0] * width
        for speaker in chosen:
            for field, value in enumerate(aggregates[str(speaker)]):
                combined[field] += value
        estimates[index] = estimate(tuple(combined))
    return estimates


def speaker_cluster_bootstrap(
    records: list[dict],
    *,
    metric: str,
    resamples: int = 10_000,
    seed: int = 2027,
) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    groups = _speaker_groups(records)
    speakers = sorted(groups)
    if not speakers:
        raise ValueError("Cannot bootstrap an empty record set")
    if metric == "wer_reconstruction":
        statistic: Callable[[list[dict]], float | None] = partial(
            corpus_wer, prefix="reconstruction"
        )
        point = statistic(records)
    elif metric == "wer_reference":
        statistic = partial(corpus_wer, prefix="reference")
        point = statistic(records)
    elif metric == "wer_delta":
        statistic = corpus_delta_wer
        point = statistic(records)
    elif metric in {"encoder_rtf", "decoder_rtf"}:
        prefix = metric.removesuffix("_rtf")
        statistic = partial(corpus_rtf, prefix=prefix)
        point = statistic(records)
    else:
        statistic = partial(_continuous_statistic, metric=metric)
        point = statistic(records)
    if point is None or not math.isfinite(point):
        raise ValueError(f"Metric {metric} has no finite observations")

    aggregates = {
        speaker: _metric_aggregate(groups[speaker], metric) for speaker in speakers
    }
    estimates = _resample_aggregates(
        aggregates,
        speakers,
        resamples=resamples,
        seed=seed,
        estimate=partial(_metric_from_aggregate, metric),
    )
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        raise ValueError(f"Metric {metric} produced no finite bootstrap samples")
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        "resamples": resamples,
        "finite_resamples": int(len(finite)),
        "seed": seed,
        "cluster": "speaker",
        "speakers": len(speakers),
    }


def paired_speaker_cluster_bootstrap(
    left_records: list[dict],
    right_records: list[dict],
    *,
    metric: str,
    resamples: int = 10_000,
    seed: int = 2027,
) -> dict:
    left = {record["cut_id"]: record for record in left_records}
    right = {record["cut_id"]: record for record in right_records}
    if set(left) != set(right):
        raise ValueError("Paired bootstrap requires identical cut IDs")
    paired = []
    for cut_id in sorted(left):
        if left[cut_id].get("speaker_id") != right[cut_id].get("speaker_id"):
            raise ValueError(f"Speaker mismatch for paired cut {cut_id}")
        paired.append((left[cut_id], right[cut_id]))
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in paired:
        groups[str(pair[0]["speaker_id"])].append(pair)
    speakers = sorted(groups)
    if not speakers:
        raise ValueError("Cannot bootstrap an empty paired record set")

    def difference(pairs: list[tuple[dict, dict]]) -> float:
        left_rows = [pair[0] for pair in pairs]
        right_rows = [pair[1] for pair in pairs]
        if metric == "wer_reconstruction":
            left_value = corpus_wer(left_rows, "reconstruction")
            right_value = corpus_wer(right_rows, "reconstruction")
        elif metric == "wer_reference":
            left_value = corpus_wer(left_rows, "reference")
            right_value = corpus_wer(right_rows, "reference")
        elif metric == "wer_delta":
            left_value = corpus_delta_wer(left_rows)
            right_value = corpus_delta_wer(right_rows)
        elif metric in {"encoder_rtf", "decoder_rtf"}:
            prefix = metric.removesuffix("_rtf")
            left_value = corpus_rtf(left_rows, prefix)
            right_value = corpus_rtf(right_rows, prefix)
        else:
            values = [
                (
                    pair[0].get("metrics", {}).get(metric),
                    pair[1].get("metrics", {}).get(metric),
                )
                for pair in pairs
            ]
            differences = [
                float(left_value) - float(right_value)
                for left_value, right_value in values
                if left_value is not None and right_value is not None
                and math.isfinite(float(left_value)) and math.isfinite(float(right_value))
            ]
            return float(mean(differences)) if differences else math.nan
        if left_value is None or right_value is None:
            return math.nan
        return float(left_value - right_value)

    def paired_aggregate(pairs: list[tuple[dict, dict]]) -> tuple[float, ...]:
        left_rows = [pair[0] for pair in pairs]
        right_rows = [pair[1] for pair in pairs]
        if metric in {
            "wer_reconstruction",
            "wer_reference",
            "wer_delta",
            "encoder_rtf",
            "decoder_rtf",
        }:
            return (
                *_metric_aggregate(left_rows, metric),
                *_metric_aggregate(right_rows, metric),
            )
        values = [
            (
                pair[0].get("metrics", {}).get(metric),
                pair[1].get("metrics", {}).get(metric),
            )
            for pair in pairs
        ]
        differences = [
            float(left_value) - float(right_value)
            for left_value, right_value in values
            if left_value is not None
            and right_value is not None
            and math.isfinite(float(left_value))
            and math.isfinite(float(right_value))
        ]
        return math.fsum(differences), float(len(differences))

    def paired_estimate(aggregate: tuple[float, ...]) -> float:
        if metric in {
            "wer_reconstruction",
            "wer_reference",
            "wer_delta",
            "encoder_rtf",
            "decoder_rtf",
        }:
            midpoint = len(aggregate) // 2
            left_value = _metric_from_aggregate(metric, aggregate[:midpoint])
            right_value = _metric_from_aggregate(metric, aggregate[midpoint:])
            return left_value - right_value
        return _metric_from_aggregate(metric, aggregate)

    point = difference(paired)
    if not math.isfinite(point):
        raise ValueError(f"Metric {metric} has no finite paired observations")
    aggregates = {speaker: paired_aggregate(groups[speaker]) for speaker in speakers}
    estimates = _resample_aggregates(
        aggregates,
        speakers,
        resamples=resamples,
        seed=seed,
        estimate=paired_estimate,
    )
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        raise ValueError(f"Metric {metric} produced no finite paired samples")
    return {
        "metric": metric,
        "contrast": "left_minus_right",
        "point_estimate": point,
        "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        "resamples": resamples,
        "finite_resamples": int(len(finite)),
        "seed": seed,
        "cluster": "speaker",
        "speakers": len(speakers),
        "paired_cuts": len(paired),
    }


def atomic_write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute speaker-cluster bootstrap CIs")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--paired-records", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    records = load_jsonl(args.records)
    result = {
        metric: speaker_cluster_bootstrap(
            records, metric=metric, resamples=args.resamples, seed=args.seed
        )
        for metric in args.metrics
    }
    if args.paired_records is not None:
        paired_records = load_jsonl(args.paired_records)
        result["paired"] = {
            metric: paired_speaker_cluster_bootstrap(
                records,
                paired_records,
                metric=metric,
                resamples=args.resamples,
                seed=args.seed,
            )
            for metric in args.metrics
        }
    atomic_write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
