from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.tools.pilot_gate import _load_complete_run, _quality_values


SPLITS = ("dev-clean", "dev-other")
ROUTES = ("codec", "vocoder", "joint")
METRIC_DIRECTIONS = {
    "wer_reconstruction": "lower",
    "wer_delta": "lower",
    "visqol_speech": "higher",
    "pesq_wb": "higher",
    "speaker_cosine": "higher",
}
RANKING_METRICS = {
    "wer_delta",
    "visqol_speech",
    "pesq_wb",
    "speaker_cosine",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _point_pass(metric: str, value: float) -> bool:
    if METRIC_DIRECTIONS[metric] == "lower":
        return value <= 0.0
    return value >= 0.0


def _significance(metric: str, ci95: tuple[float, float]) -> str:
    lower, upper = ci95
    if METRIC_DIRECTIONS[metric] == "lower":
        if upper < 0.0:
            return "beneficial"
        if lower > 0.0:
            return "harmful"
    else:
        if lower > 0.0:
            return "beneficial"
        if upper < 0.0:
            return "harmful"
    return "unresolved"


def assess_route(
    paired_by_split: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if set(paired_by_split) != set(SPLITS):
        raise ValueError(f"Expected paired statistics for {SPLITS}")
    assessments: dict[str, dict[str, Any]] = {}
    beneficial = 0
    harmful = 0
    for split in SPLITS:
        metrics = paired_by_split[split]
        if set(metrics) != set(METRIC_DIRECTIONS):
            raise ValueError(
                f"Unexpected paired metrics for {split}: {sorted(metrics)}"
            )
        split_result: dict[str, Any] = {}
        for metric in METRIC_DIRECTIONS:
            item = metrics[metric]
            point = float(item["point_estimate"])
            ci95 = tuple(float(value) for value in item["ci95"])
            significance = _significance(metric, ci95)
            if metric in RANKING_METRICS:
                beneficial += significance == "beneficial"
                harmful += significance == "harmful"
            split_result[metric] = {
                "point_estimate": point,
                "ci95": list(ci95),
                "point_direction_pass": _point_pass(metric, point),
                "significance": significance,
            }
        assessments[split] = split_result
    point_pass = all(
        metric["point_direction_pass"]
        for split in assessments.values()
        for metric in split.values()
    )
    return {
        "eligible_before_listening": point_pass,
        "beneficial_significant_metrics": beneficial,
        "harmful_significant_metrics": harmful,
        "splits": assessments,
    }


def _validate_paired_document(
    document: dict[str, Any],
    *,
    expected_differences: dict[str, float],
    expected_cuts: int,
) -> dict[str, dict[str, Any]]:
    paired = document.get("paired")
    if not isinstance(paired, dict) or set(paired) != set(METRIC_DIRECTIONS):
        raise ValueError("Paired statistics do not contain the frozen metric set")
    for metric, expected in expected_differences.items():
        item = paired[metric]
        if item.get("contrast") != "left_minus_right":
            raise ValueError(f"Unexpected contrast for {metric}")
        if int(item.get("resamples", 0)) != 10_000:
            raise ValueError(f"Expected 10,000 paired resamples for {metric}")
        if int(item.get("finite_resamples", 0)) != 10_000:
            raise ValueError(f"Non-finite paired resamples for {metric}")
        if int(item.get("paired_cuts", -1)) != expected_cuts:
            raise ValueError(f"Paired cut count mismatch for {metric}")
        point = float(item["point_estimate"])
        if not math.isclose(point, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"Paired point estimate disagrees with summaries for {metric}: "
                f"{point} != {expected}"
            )
        ci95 = item.get("ci95")
        if (
            not isinstance(ci95, list)
            or len(ci95) != 2
            or not all(math.isfinite(float(value)) for value in ci95)
        ):
            raise ValueError(f"Invalid paired interval for {metric}")
    return paired


def _evaluation_identity(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metadata_path = directory / "model_metadata.json"
    git_state_path = directory / "git_state.json"
    if not metadata_path.is_file() or not git_state_path.is_file():
        raise FileNotFoundError(
            f"M+ direction evaluation lacks model/git metadata: {directory}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    git_state = json.loads(git_state_path.read_text(encoding="utf-8"))
    revision = metadata.get("system", {}).get("checkpoint_revision_and_hash", {})
    raw_checkpoint_hash = revision.get("checkpoint_sha256")
    if not isinstance(raw_checkpoint_hash, str) or len(raw_checkpoint_hash) != 64:
        raise ValueError(
            f"M+ direction evaluation lacks the raw checkpoint SHA-256: {directory}"
        )
    commit = git_state.get("commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError(f"M+ direction evaluation lacks a git commit: {directory}")
    return {
        "system": str(summary["system"]),
        "variant": str(summary["variant"]),
        "seed": int(summary["seed"]),
        "config_hash": str(summary["config_hash"]),
        "checkpoint_hash": str(summary["checkpoint_hash"]),
        "raw_checkpoint_sha256": raw_checkpoint_hash,
        "git_commit": commit,
    }


def _load_comparison(
    candidate_directory: Path,
    source_directory: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    candidate = _load_complete_run(candidate_directory)
    source = _load_complete_run(source_directory)
    candidate_summary, candidate_cut_hash, candidate_hashes, candidate_exclusions = (
        candidate
    )
    source_summary, source_cut_hash, source_hashes, source_exclusions = source
    if candidate_cut_hash != source_cut_hash:
        raise ValueError(f"Cut IDs differ for {expected_split}")
    if candidate_exclusions != source_exclusions:
        raise ValueError(f"Metric-domain exclusions differ for {expected_split}")
    if {
        str(candidate_summary["split"]),
        str(source_summary["split"]),
    } != {expected_split}:
        raise ValueError(f"Unexpected split for {expected_split}")
    candidate_rate = float(candidate_summary["bitrate_kbps"]["nominal"])
    source_rate = float(source_summary["bitrate_kbps"]["nominal"])
    expected_rate = 1.7463776043357018
    if not math.isclose(candidate_rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Unexpected candidate bitrate: {candidate_rate}")
    if not math.isclose(source_rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Unexpected source bitrate: {source_rate}")
    if int(candidate_summary["seed"]) != 2026 or int(source_summary["seed"]) != 2026:
        raise ValueError("M+ direction gate requires the matched seed-2026 source")

    candidate_values = _quality_values(candidate_summary)
    source_values = _quality_values(source_summary)
    differences = {
        metric: candidate_values[metric] - source_values[metric]
        for metric in METRIC_DIRECTIONS
    }
    statistics_path = candidate_directory / "paired_vs_source.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(statistics_path)
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    paired = _validate_paired_document(
        statistics,
        expected_differences=differences,
        expected_cuts=int(candidate_summary["utterances"]),
    )
    return {
        "candidate_run_id": candidate_summary["run_id"],
        "source_run_id": source_summary["run_id"],
        "candidate_identity": _evaluation_identity(
            candidate_directory, candidate_summary
        ),
        "source_identity": _evaluation_identity(source_directory, source_summary),
        "cut_ids_sha256": candidate_cut_hash,
        "metric_domain_exclusions": candidate_exclusions,
        "paired": paired,
        "artifacts": {
            "candidate": candidate_hashes,
            "source": source_hashes,
            "paired_vs_source.json": sha256_file(statistics_path),
        },
    }


def evaluate_direction_gate(
    *,
    source_directories: dict[str, Path],
    candidate_directories: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    if set(source_directories) != set(SPLITS):
        raise ValueError(f"Expected source directories for {SPLITS}")
    if set(candidate_directories) != set(ROUTES):
        raise ValueError(f"Expected candidates for {ROUTES}")
    results: dict[str, Any] = {}
    for route in ROUTES:
        if set(candidate_directories[route]) != set(SPLITS):
            raise ValueError(f"Expected both dev splits for {route}")
        comparisons = {
            split: _load_comparison(
                candidate_directories[route][split],
                source_directories[split],
                expected_split=split,
            )
            for split in SPLITS
        }
        assessment = assess_route(
            {split: comparison["paired"] for split, comparison in comparisons.items()}
        )
        assessment["runs"] = {
            split: {key: value for key, value in comparison.items() if key != "paired"}
            for split, comparison in comparisons.items()
        }
        results[route] = assessment

    eligible = [
        route for route in ROUTES if results[route]["eligible_before_listening"]
    ]
    if eligible:
        maximum_benefits = max(
            results[route]["beneficial_significant_metrics"] for route in eligible
        )
        statistical_leaders = [
            route
            for route in eligible
            if results[route]["beneficial_significant_metrics"] == maximum_benefits
        ]
        disposition = "listening_required"
    else:
        statistical_leaders = []
        disposition = "retain_source"
    return {
        "schema_version": 1,
        "status": "MPLUS_DIRECTION_GATE",
        "quality_directions": METRIC_DIRECTIONS,
        "selection_rule": (
            "On both complete dev splits, candidate-minus-source point estimates "
            "must be <=0 for WER/DeltaWER and >=0 for ViSQOL/PESQ/speaker; "
            "10,000-resample paired intervals rank evidence, but matched listening "
            "is mandatory before promotion."
        ),
        "eligible_before_listening": eligible,
        "statistical_leaders": statistical_leaders,
        "automatic_promotion": False,
        "disposition": disposition,
        "routes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered M+ complete-dev direction gate"
    )
    for split in SPLITS:
        suffix = split.replace("-", "_")
        parser.add_argument(f"--source-{split}", dest=f"source_{suffix}", type=Path)
        for route in ROUTES:
            parser.add_argument(
                f"--{route}-{split}", dest=f"{route}_{suffix}", type=Path
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = [
        name for name, value in vars(args).items() if name != "output" and value is None
    ]
    if missing:
        parser.error(f"Missing required run directories: {', '.join(sorted(missing))}")
    source_directories = {
        split: getattr(args, f"source_{split.replace('-', '_')}") for split in SPLITS
    }
    candidate_directories = {
        route: {
            split: getattr(args, f"{route}_{split.replace('-', '_')}")
            for split in SPLITS
        }
        for route in ROUTES
    }
    result = evaluate_direction_gate(
        source_directories=source_directories,
        candidate_directories=candidate_directories,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
