from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.evaluation.statistics import load_jsonl
from dmel_codec.evaluation.summarize import summarize_records
from dmel_codec.evaluation.validity import validate_primary_metric_domain


QUALITY_DIRECTIONS = {
    "wer_reconstruction": "lower",
    "wer_delta": "lower",
    "visqol_speech": "higher",
    "pesq_wb": "higher",
    "speaker_cosine": "higher",
}

def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cut_ids_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for cut_id in sorted(str(record["cut_id"]) for record in records):
        digest.update(cut_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_complete_run(
    directory: Path,
) -> tuple[dict[str, Any], str, dict[str, str], dict[str, list[str]]]:
    directory = directory.resolve()
    records_path = directory / "per_utterance.jsonl"
    summary_path = directory / "summary.json"
    if not records_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete evaluation directory: {directory}")
    records = load_jsonl(records_path)
    rebuilt = summarize_records(records)
    rebuilt["per_utterance_sha256"] = sha256_file(records_path)
    observed = json.loads(summary_path.read_text(encoding="utf-8"))
    if observed != rebuilt:
        raise ValueError(f"summary.json does not rebuild exactly: {directory}")
    if observed["failed_utterances"]:
        raise ValueError(f"Pilot gate requires zero reconstruction failures: {directory}")
    exclusions = validate_primary_metric_domain(records)
    utterances = int(observed["utterances"])
    for metric in ("visqol_speech", "pesq_wb", "speaker_cosine"):
        value = observed.get("metrics", {}).get(metric, {})
        expected_failures = len(exclusions.get(metric, []))
        expected_count = utterances - expected_failures
        if (
            value.get("mean") is None
            or int(value.get("count", -1)) != expected_count
            or int(value.get("failures", -1)) != expected_failures
        ):
            raise ValueError(f"Pilot gate lacks complete {metric}: {directory}")
    for metric in ("reference", "reconstruction", "delta"):
        value = observed.get("corpus_wer", {}).get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Pilot gate lacks finite WER {metric}: {directory}")
    return (
        observed,
        _cut_ids_hash(records),
        {
            "per_utterance.jsonl": sha256_file(records_path),
            "summary.json": sha256_file(summary_path),
        },
        exclusions,
    )


def _quality_values(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "wer_reconstruction": float(summary["corpus_wer"]["reconstruction"]),
        "wer_delta": float(summary["corpus_wer"]["delta"]),
        "visqol_speech": float(summary["metrics"]["visqol_speech"]["mean"]),
        "pesq_wb": float(summary["metrics"]["pesq_wb"]["mean"]),
        "speaker_cosine": float(summary["metrics"]["speaker_cosine"]["mean"]),
    }


def _comparison(ours: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    ours_values = _quality_values(ours)
    baseline_values = _quality_values(baseline)
    baseline_better = {
        metric: (
            baseline_values[metric] < ours_values[metric]
            if direction == "lower"
            else baseline_values[metric] > ours_values[metric]
        )
        for metric, direction in QUALITY_DIRECTIONS.items()
    }
    quality_dominates = all(baseline_better.values())
    ours_rate = float(ours["bitrate_kbps"]["nominal"])
    baseline_rate = float(baseline["bitrate_kbps"]["nominal"])
    return {
        "system": baseline["system"],
        "nominal_bitrate_kbps": baseline_rate,
        "quality": baseline_values,
        "strictly_better": baseline_better,
        "strict_quality_dominance": quality_dominates,
        "strict_rate_quality_dominance": quality_dominates
        and baseline_rate <= ours_rate,
    }


def evaluate_pilot_gate(
    ours_directory: str | Path,
    dac_q2_directory: str | Path,
    dac_q3_directory: str | Path,
    *,
    repair_rounds: int,
) -> dict[str, Any]:
    if repair_rounds not in {0, 1, 2}:
        raise ValueError("repair_rounds must be 0, 1, or 2")
    named_directories = {
        "ours": Path(ours_directory),
        "dac_q2": Path(dac_q2_directory),
        "dac_q3": Path(dac_q3_directory),
    }
    loaded = {
        name: _load_complete_run(directory)
        for name, directory in named_directories.items()
    }
    cut_hashes = {name: value[1] for name, value in loaded.items()}
    if len(set(cut_hashes.values())) != 1:
        raise ValueError(f"Pilot systems use different cut IDs: {cut_hashes}")
    metric_domain_exclusions = {name: value[3] for name, value in loaded.items()}
    if len({json.dumps(value, sort_keys=True) for value in metric_domain_exclusions.values()}) != 1:
        raise ValueError(
            "Pilot systems have different metric-domain exclusion sets: "
            f"{metric_domain_exclusions}"
        )
    summaries = {name: value[0] for name, value in loaded.items()}
    if len({summary["split"] for summary in summaries.values()}) != 1:
        raise ValueError("Pilot systems use different splits")

    expected_rates = {"ours": 1.7463776043357018, "dac_q2": 1.5, "dac_q3": 2.25}
    for name, expected in expected_rates.items():
        observed = float(summaries[name]["bitrate_kbps"]["nominal"])
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Unexpected {name} nominal bitrate: {observed}")

    comparisons = {
        name: _comparison(summaries["ours"], summaries[name])
        for name in ("dac_q2", "dac_q3")
    }
    dominated_by_both = all(
        comparison["strict_quality_dominance"]
        for comparison in comparisons.values()
    )
    if dominated_by_both and repair_rounds >= 2:
        disposition = "escalate_no_go"
    elif dominated_by_both:
        disposition = "controlled_repair"
    else:
        disposition = "manual_go_review"
    return {
        "schema_version": 1,
        "split": summaries["ours"]["split"],
        "repair_rounds_completed": repair_rounds,
        "quality_directions": QUALITY_DIRECTIONS,
        "ours": {
            "system": summaries["ours"]["system"],
            "nominal_bitrate_kbps": summaries["ours"]["bitrate_kbps"]["nominal"],
            "quality": _quality_values(summaries["ours"]),
        },
        "comparisons": comparisons,
        "strict_quality_dominated_by_both_dac_points": dominated_by_both,
        "execplan_no_go_condition_met": dominated_by_both and repair_rounds >= 2,
        "disposition": disposition,
        "cut_ids_sha256": next(iter(cut_hashes.values())),
        "metric_domain_exclusions": next(iter(metric_domain_exclusions.values())),
        "runs": {
            name: {
                "run_id": summaries[name]["run_id"],
                "directory": str(named_directories[name].resolve()),
                "sources": loaded[name][2],
            }
            for name in named_directories
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the preregistered M6 pilot gate")
    parser.add_argument("--ours-run", type=Path, required=True)
    parser.add_argument("--dac-q2-run", type=Path, required=True)
    parser.add_argument("--dac-q3-run", type=Path, required=True)
    parser.add_argument("--repair-rounds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_pilot_gate(
        args.ours_run,
        args.dac_q2_run,
        args.dac_q3_run,
        repair_rounds=args.repair_rounds,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
