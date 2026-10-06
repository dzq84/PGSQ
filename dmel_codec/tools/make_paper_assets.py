from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dmel_codec.evaluation.cache import canonical_hash, sha256_file
from dmel_codec.evaluation.statistics import load_jsonl
from dmel_codec.evaluation.summarize import summarize_records
from dmel_codec.evaluation.validity import validate_primary_metric_domain


PRIMARY_METRICS = ("visqol_speech", "pesq_wb", "speaker_cosine")
SEED_AGGREGATE_METRICS = (
    "nominal_bitrate_kbps",
    "entropy_bitrate_kbps",
    "wer_reconstruction",
    "wer_delta",
    "visqol_speech",
    "pesq_wb",
    "speaker_cosine",
    "encoder_rtf",
    "decoder_rtf",
)
REQUIRED_STATISTICS = (
    *PRIMARY_METRICS,
    "wer_reconstruction",
    "wer_delta",
    "encoder_rtf",
    "decoder_rtf",
)


@dataclass(frozen=True)
class PaperRun:
    directory: Path
    summary: dict[str, Any]
    statistics: dict[str, Any] | None
    dependency: dict[str, Any] | None
    cut_ids_sha256: str
    primary_metric_exclusions: dict[str, list[str]]
    source_hashes: dict[str, str]
    resolved_config: dict[str, Any] | None
    model_metadata: dict[str, Any] | None
    git_state: dict[str, Any] | None


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _cut_ids_sha256(records: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for cut_id in sorted(str(record["cut_id"]) for record in records):
        digest.update(cut_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return value


def _load_run(directory: Path, *, allow_incomplete: bool) -> PaperRun:
    directory = directory.resolve()
    records_path = directory / "per_utterance.jsonl"
    summary_path = directory / "summary.json"
    if not records_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Paper run needs per_utterance.jsonl and summary.json: {directory}"
        )
    records = load_jsonl(records_path)
    rebuilt = summarize_records(records)
    rebuilt["per_utterance_sha256"] = sha256_file(records_path)
    observed = _load_json(summary_path)
    if observed != rebuilt:
        raise ValueError(
            f"summary.json does not exactly rebuild from per_utterance.jsonl: {directory}"
        )

    statistics_path = directory / "statistics.json"
    statistics = _load_json(statistics_path) if statistics_path.is_file() else None
    if statistics is None and not allow_incomplete:
        raise FileNotFoundError(
            f"Missing speaker-bootstrap statistics: {statistics_path}"
        )
    if statistics is not None and not allow_incomplete:
        missing_statistics = [
            key for key in REQUIRED_STATISTICS if key not in statistics
        ]
        if missing_statistics:
            raise ValueError(
                f"Run {observed['run_id']} lacks bootstrap statistics: {missing_statistics}"
            )
        invalid_statistics = [
            key
            for key in REQUIRED_STATISTICS
            if int(statistics[key].get("resamples", 0)) < 10_000
            or statistics[key].get("cluster") != "speaker"
        ]
        if invalid_statistics:
            raise ValueError(
                "Formal paper statistics require >=10000 speaker-cluster resamples: "
                f"{observed['run_id']}:{invalid_statistics}"
            )

    dependency_path = directory / "code_dependency.json"
    dependency = _load_json(dependency_path) if dependency_path.is_file() else None
    needs_dependency = (
        observed["variant"] in {"ours", "dac"}
        and int(observed["num_parallel_streams"]) >= 2
    )
    if needs_dependency and dependency is None and not allow_incomplete:
        raise FileNotFoundError(f"Missing code dependency artifact: {dependency_path}")

    present_primary_metrics = [
        metric for metric in PRIMARY_METRICS if metric in observed.get("metrics", {})
    ]
    primary_metric_exclusions = (
        validate_primary_metric_domain(
            records, required_metrics=present_primary_metrics
        )
        if present_primary_metrics
        else {}
    )
    if not allow_incomplete:
        missing = [
            metric
            for metric in PRIMARY_METRICS
            if observed.get("metrics", {}).get(metric, {}).get("mean") is None
        ]
        if observed.get("corpus_wer", {}).get("reconstruction") is None:
            missing.append("wer_reconstruction")
        if missing:
            raise ValueError(
                f"Run {observed['run_id']} lacks primary metrics: {missing}"
            )

    source_hashes = {
        "per_utterance.jsonl": sha256_file(records_path),
        "summary.json": sha256_file(summary_path),
    }
    for name in (
        "statistics.json",
        "code_dependency.json",
        "resolved_config.yaml",
        "model_metadata.json",
        "data_manifest.json",
        "git_state.json",
    ):
        path = directory / name
        if path.is_file():
            source_hashes[name] = sha256_file(path)
    resolved_config_path = directory / "resolved_config.yaml"
    model_metadata_path = directory / "model_metadata.json"
    git_state_path = directory / "git_state.json"
    return PaperRun(
        directory=directory,
        summary=observed,
        statistics=statistics,
        dependency=dependency,
        cut_ids_sha256=_cut_ids_sha256(records),
        primary_metric_exclusions=primary_metric_exclusions,
        source_hashes=source_hashes,
        resolved_config=(
            _load_yaml(resolved_config_path) if resolved_config_path.is_file() else None
        ),
        model_metadata=(
            _load_json(model_metadata_path) if model_metadata_path.is_file() else None
        ),
        git_state=_load_json(git_state_path) if git_state_path.is_file() else None,
    )


def _validate_runs(runs: list[PaperRun]) -> None:
    if not runs:
        raise ValueError("At least one result directory is required")
    identities = [
        (
            run.summary["system"],
            run.summary["variant"],
            run.summary["seed"],
            run.summary["split"],
        )
        for run in runs
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Paper inputs contain a duplicate system/variant/seed/split")
    by_split: dict[str, set[str]] = {}
    for run in runs:
        by_split.setdefault(str(run.summary["split"]), set()).add(run.cut_ids_sha256)
    mismatches = {
        split: hashes for split, hashes in by_split.items() if len(hashes) > 1
    }
    if mismatches:
        raise ValueError(f"Systems use different cut IDs within a split: {mismatches}")
    exclusion_profiles: dict[str, set[str]] = {}
    for run in runs:
        profile = json.dumps(run.primary_metric_exclusions, sort_keys=True)
        exclusion_profiles.setdefault(str(run.summary["split"]), set()).add(profile)
    exclusion_mismatches = {
        split: profiles
        for split, profiles in exclusion_profiles.items()
        if len(profiles) > 1
    }
    if exclusion_mismatches:
        raise ValueError(
            "Systems use different primary-metric exclusion sets within a split: "
            f"{exclusion_mismatches}"
        )


def _paper_identity(run: PaperRun) -> tuple[Any, ...]:
    if run.resolved_config is None:
        raise ValueError(f"Paper run lacks resolved_config.yaml: {run.directory}")
    return (
        run.resolved_config.get("system"),
        run.summary["system"],
        run.summary["variant"],
        int(run.summary["seed"]),
        run.summary["config_hash"],
        run.summary["checkpoint_hash"],
        run.resolved_config.get("decode_seed_or_policy"),
        run.summary["split"],
    )


def _gate_system_identity(system: dict[str, Any]) -> tuple[Any, ...]:
    return (
        system["adapter"],
        system["system"],
        system["variant"],
        int(system["seed"]),
        system["config_hash"],
        system["checkpoint_hash"],
        system["decode_seed_or_policy"],
    )


def _paper_display(registry: dict[str, Any]) -> dict[str, Any]:
    display = registry.get("paper_display")
    known_roles = registry.get("roles")
    if not isinstance(display, dict) or display.get("schema_version") != 1:
        raise ValueError("Frozen test gate lacks paper-display schema v1")
    if not isinstance(known_roles, list):
        raise ValueError("Frozen test gate lacks final-system roles")
    result: dict[str, Any] = {"schema_version": 1}
    for field in (
        "main_table_roles",
        "ablation_table_roles",
        "rate_quality_roles",
    ):
        values = display.get(field)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(f"Frozen paper_display.{field} is invalid")
        if len(values) != len(set(values)):
            raise ValueError(f"Frozen paper_display.{field} contains duplicates")
        unknown = sorted(set(values) - set(known_roles))
        if unknown:
            raise ValueError(
                f"Frozen paper_display.{field} has unknown roles: {unknown}"
            )
        result[field] = list(values)
    return result


def _test_gate_role_lookup(test_gate: str | Path) -> dict[tuple[Any, ...], set[str]]:
    gate = _load_json(Path(test_gate).expanduser().resolve())
    result: dict[tuple[Any, ...], set[str]] = {}
    for system in gate.get("systems", []):
        identity = _gate_system_identity(system)
        if identity in result:
            raise ValueError("Frozen test gate repeats a paper system identity")
        result[identity] = set(system.get("roles", []))
    return result


def _validate_test_gate_coverage(
    runs: list[PaperRun], test_gate: str | Path
) -> dict[str, Any]:
    gate_path = Path(test_gate).expanduser().resolve()
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    gate = _load_json(gate_path)
    if gate.get("schema_version") != 3 or gate.get("status") != "TEST_GATE_FROZEN":
        raise ValueError("Formal paper assets require a schema-v3 frozen test gate")
    registry = gate.get("final_system_registry")
    if not isinstance(registry, dict) or not registry.get("roles"):
        raise ValueError("Frozen test gate lacks final-system registry coverage")
    system_roles = [
        role for system in gate.get("systems", []) for role in system.get("roles", [])
    ]
    if (
        len(system_roles) != len(set(system_roles))
        or sorted(system_roles) != registry["roles"]
        or registry.get("unique_system_identities") != len(gate.get("systems", []))
    ):
        raise ValueError("Frozen test gate has inconsistent final-system coverage")
    paper_display = _paper_display(registry)

    expected = {
        (
            system["adapter"],
            system["system"],
            system["variant"],
            int(system["seed"]),
            system["config_hash"],
            system["checkpoint_hash"],
            system["decode_seed_or_policy"],
            split,
        )
        for system in gate.get("systems", [])
        for split in ("test-clean", "test-other")
    }
    observed = set()
    gate_sha256 = sha256_file(gate_path)
    for run in runs:
        identity = _paper_identity(run)
        split = str(run.summary["split"])
        if split not in {"test-clean", "test-other"}:
            raise ValueError(f"Formal paper input is not a test run: {split}")
        if run.resolved_config["system_config_hash"] != run.summary["config_hash"]:
            raise ValueError(f"Paper run config hash mismatch: {run.summary['run_id']}")
        if run.resolved_config.get("test_gate_sha256") != gate_sha256:
            raise ValueError(
                f"Paper run does not bind the frozen test gate: {run.summary['run_id']}"
            )
        if run.resolved_config.get("manifest_sha256") != gate.get(
            "test_manifests", {}
        ).get(split):
            raise ValueError(
                f"Paper run manifest is not frozen for {split}: {run.summary['run_id']}"
            )
        if sorted(run.resolved_config.get("metrics", [])) != gate.get("metrics"):
            raise ValueError(
                f"Paper run metrics differ from the test gate: {run.summary['run_id']}"
            )
        if (
            run.model_metadata is None
            or "metrics" not in run.model_metadata
            or canonical_hash(run.model_metadata["metrics"])
            != gate.get("metric_metadata_hash")
        ):
            raise ValueError(
                "Paper run metric revisions differ from the test gate: "
                f"{run.summary['run_id']}"
            )
        if (
            run.git_state is None
            or run.git_state.get("commit") != gate.get("git_commit")
            or run.git_state.get("dirty")
        ):
            raise ValueError(
                f"Paper run git state differs from the test gate: "
                f"{run.summary['run_id']}"
            )
        observed.add(identity)
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise ValueError(
            "Formal paper inputs do not exactly cover the frozen test systems; "
            f"missing={missing}, extra={extra}"
        )
    return {
        "schema_version": 3,
        "status": "TEST_GATE_FROZEN",
        "sha256": gate_sha256,
        "git_commit": gate["git_commit"],
        "metric_metadata_hash": gate["metric_metadata_hash"],
        "final_system_registry_sha256": registry["sha256"],
        "requirements_sha256": registry["requirements_sha256"],
        "roles": registry["roles"],
        "unique_system_identities": registry["unique_system_identities"],
        "paper_display": paper_display,
        "test_manifests": gate["test_manifests"],
        "metrics": gate["metrics"],
    }


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _system_label(summary: dict[str, Any]) -> str:
    system = str(summary["system"])
    display_names = {
        "dMel-same-front-end (adapted)": "dMel adapted",
        "dMel-quantization-aware-decoder (adapted)": "dMel adapted + dec.",
        "Ours-M+ (full codec + frozen vocoder + waveform GAN)": "Ours-M+ codec",
        "Ours-M+ (frozen codec + full vocoder + waveform GAN)": "Ours-M+ vocoder",
        "Ours-M+ (full codec + full vocoder + waveform GAN)": "Ours-M+ joint",
    }
    if system.startswith("DAC 24 kHz q="):
        system = system.replace("DAC 24 kHz ", "DAC ", 1)
    escaped = _latex_escape(display_names.get(system, system))
    lowered = system.lower()
    if lowered.startswith("ours"):
        return escaped + r"$^{L,E}$"
    if lowered.startswith("dac"):
        return escaped + r"$^{E}$"
    if lowered.startswith("dmel"):
        complexity = summary.get("model_complexity") or {}
        trained_parameters = int(complexity.get("codec_parameters") or 0)
        if trained_parameters > 0:
            return escaped + r"$^{L,A,E}$"
        return escaped + r"$^{A,E}$"
    if lowered == "mel oracle":
        return escaped + r"$^{C,E}$"
    return escaped + r"$^{C}$"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _format(value: Any, digits: int = 3, scale: float = 1.0) -> str:
    finite = _finite(value)
    return "--" if finite is None else f"{finite * scale:.{digits}f}"


def _run_value(run: PaperRun, key: str) -> float | None:
    summary = run.summary
    if key == "nominal_bitrate_kbps":
        return _finite(summary.get("bitrate_kbps", {}).get("nominal"))
    if key == "entropy_bitrate_kbps":
        return _finite(summary.get("bitrate_kbps", {}).get("entropy"))
    if key == "token_frame_rate":
        return _finite(summary.get("token_frame_rate"))
    if key == "wer_reconstruction":
        return _finite(summary.get("corpus_wer", {}).get("reconstruction"))
    if key == "wer_delta":
        return _finite(summary.get("corpus_wer", {}).get("delta"))
    if key in {"encoder_rtf", "decoder_rtf"}:
        prefix = key.removesuffix("_rtf")
        return _finite(summary.get("rtf", {}).get(prefix))
    return _finite(summary.get("metrics", {}).get(key, {}).get("mean"))


def _seed_groups(runs: Iterable[PaperRun]) -> list[list[PaperRun]]:
    grouped: dict[tuple[str, str, str], list[PaperRun]] = {}
    for run in runs:
        key = (
            str(run.summary["system"]),
            str(run.summary["variant"]),
            str(run.summary["split"]),
        )
        grouped.setdefault(key, []).append(run)
    result = []
    for group in grouped.values():
        group.sort(key=lambda run: int(run.summary["seed"]))
        first = group[0].summary
        complexity = first.get("model_complexity") or {}
        fixed = (
            first["bitrate_kbps"]["nominal"],
            first["token_frame_rate"],
            first["codebook_cardinalities"],
            complexity.get("codec_parameters"),
            complexity.get("external_pretrained_parameters"),
        )
        for run in group[1:]:
            complexity = run.summary.get("model_complexity") or {}
            observed = (
                run.summary["bitrate_kbps"]["nominal"],
                run.summary["token_frame_rate"],
                run.summary["codebook_cardinalities"],
                complexity.get("codec_parameters"),
                complexity.get("external_pretrained_parameters"),
            )
            if observed != fixed:
                raise ValueError(
                    "Cannot aggregate seeds with different operating points: "
                    f"{first['system']}/{first['split']}"
                )
        result.append(group)
    return sorted(
        result,
        key=lambda group: (
            str(group[0].summary["split"]),
            str(group[0].summary["system"]),
            str(group[0].summary["variant"]),
        ),
    )


def _seed_estimate(
    group: list[PaperRun], key: str
) -> tuple[float | None, float | None]:
    values = [_run_value(run, key) for run in group]
    finite = [value for value in values if value is not None]
    if len(finite) != len(group):
        return None, None
    return mean(finite), stdev(finite) if len(finite) > 1 else None


def _seed_aggregate_payload(runs: Iterable[PaperRun]) -> dict[str, Any]:
    operating_points = []
    for group in _seed_groups(runs):
        first = group[0].summary
        metrics: dict[str, Any] = {}
        for key in SEED_AGGREGATE_METRICS:
            values = [_run_value(run, key) for run in group]
            finite = [value for value in values if value is not None]
            complete = len(finite) == len(group)
            metrics[key] = {
                "count": len(finite),
                "mean": mean(finite) if complete else None,
                "sample_std": stdev(finite) if complete and len(finite) > 1 else None,
                "values_by_seed": {
                    str(run.summary["seed"]): value for run, value in zip(group, values)
                },
            }
        operating_points.append(
            {
                "system": first["system"],
                "variant": first["variant"],
                "split": first["split"],
                "seed_count": len(group),
                "seeds": [int(run.summary["seed"]) for run in group],
                "run_ids": [str(run.summary["run_id"]) for run in group],
                "metrics": metrics,
            }
        )
    return {
        "schema_version": 1,
        "estimator": (
            "Unweighted mean and sample standard deviation across completed runs "
            "at an identical operating point; sample_std is null for one run."
        ),
        "operating_points": operating_points,
    }


def _compact_estimate_across_seeds(
    group: list[PaperRun],
    *,
    key: str,
    digits: int = 3,
    scale: float = 1.0,
) -> str:
    point, deviation = _seed_estimate(group, key)
    if point is None:
        return "--"
    if deviation is None:
        return _format(point, digits, scale)
    return (
        f"${point * scale:.{digits}f}"
        rf"\mathbin{{\pm}}{deviation * scale:.{digits}f}$"
    )


def _main_table_row(group: list[PaperRun]) -> list[str]:
    summary = group[0].summary
    return [
        _system_label(summary),
        _latex_escape(summary["split"]),
        str(len(group)),
        _format(summary["bitrate_kbps"]["nominal"]),
        _format(summary["token_frame_rate"], 2),
        _compact_estimate_across_seeds(
            group,
            key="wer_reconstruction",
            digits=2,
            scale=100.0,
        ),
        _compact_estimate_across_seeds(
            group,
            key="wer_delta",
            digits=2,
            scale=100.0,
        ),
        _compact_estimate_across_seeds(
            group,
            key="visqol_speech",
        ),
        _compact_estimate_across_seeds(
            group,
            key="pesq_wb",
        ),
        _compact_estimate_across_seeds(
            group,
            key="speaker_cosine",
        ),
    ]


def _dependency_mean(run: PaperRun, field: str) -> float | None:
    if run.dependency is None:
        return None
    if field == "adjusted_nmi":
        values = [
            _finite(
                pair.get(
                    "adjusted_normalized_mutual_information",
                    pair.get("normalized_mutual_information"),
                )
            )
            for pair in run.dependency.get("pairs", [])
        ]
    elif field == "probe_lift":
        values = [
            _finite(item.get("accuracy_lift"))
            for item in run.dependency.get("probe_results", [])
        ]
    else:
        raise ValueError(f"Unknown dependency field: {field}")
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else None


def _ablation_table_row(run: PaperRun) -> list[str]:
    summary = run.summary
    return [
        _system_label(summary),
        _latex_escape(summary["split"]),
        _format(summary["bitrate_kbps"]["nominal"]),
        _format(_run_value(run, "wer_reconstruction"), 2, 100.0),
        _format(_run_value(run, "visqol_speech")),
        _format(_run_value(run, "pesq_wb")),
        _format(_run_value(run, "speaker_cosine")),
        _format(_dependency_mean(run, "adjusted_nmi")),
        _format(_dependency_mean(run, "probe_lift")),
    ]


def _runs_for_roles(
    runs: list[PaperRun],
    role_lookup: dict[tuple[Any, ...], set[str]],
    roles: tuple[str, ...],
    split: str,
) -> list[PaperRun]:
    selected = []
    for role in roles:
        matches = [
            run
            for run in runs
            if run.summary["split"] == split
            and role in (_run_roles(run, role_lookup) or set())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Frozen result prose requires one {split} run for role {role}; "
                f"found {len(matches)}"
            )
        selected.append(matches[0])
    unique = {run.summary["run_id"]: run for run in selected}
    if len(unique) != len(roles):
        raise ValueError(f"Frozen result prose roles alias within {split}: {roles}")
    return sorted(unique.values(), key=lambda run: int(run.summary["seed"]))


def _result_estimate(
    group: list[PaperRun],
    *,
    key: str,
    digits: int = 3,
    scale: float = 1.0,
) -> str:
    value = _compact_estimate_across_seeds(
        group,
        key=key,
        digits=digits,
        scale=scale,
    )
    if value == "--":
        run_ids = [run.summary["run_id"] for run in group]
        raise ValueError(f"Frozen result prose lacks {key}: {run_ids}")
    return value


def _result_pair(
    clean: list[PaperRun],
    other: list[PaperRun],
    *,
    key: str,
    digits: int = 3,
    scale: float = 1.0,
) -> str:
    return "/".join(
        (
            _result_estimate(clean, key=key, digits=digits, scale=scale),
            _result_estimate(other, key=key, digits=digits, scale=scale),
        )
    )


def _dependency_pair(clean: PaperRun, other: PaperRun, field: str) -> str:
    values = (_dependency_mean(clean, field), _dependency_mean(other, field))
    if any(value is None for value in values):
        raise ValueError(
            f"Frozen result prose lacks {field}: "
            f"{clean.summary['run_id']}, {other.summary['run_id']}"
        )
    return "/".join(_format(value) for value in values)


def _result_summary_tex(
    runs: list[PaperRun],
    role_lookup: dict[tuple[Any, ...], set[str]] | None,
) -> str:
    if role_lookup is None:
        return "% Frozen result prose requires a schema-v3 test gate.\n"

    flagship_roles = (
        "flagship_m_seed2026",
        "flagship_m_seed2027",
        "flagship_m_seed2028",
    )
    singleton_roles = (
        "mel_oracle",
        "dmel_same_frontend",
        "dmel_quantization_aware_decoder",
        "dac_q1",
        "dac_q2",
        "dac_q3",
        "ours_075_seed2026",
        "base_m_seed2026",
        "joint_fullband_seed2026",
        "random_groups_seed2026",
    )
    groups: dict[str, dict[str, list[PaperRun]]] = {}
    for split in ("test-clean", "test-other"):
        groups[split] = {
            "flagship": _runs_for_roles(
                runs,
                role_lookup,
                flagship_roles,
                split,
            )
        }
        for role in singleton_roles:
            groups[split][role] = _runs_for_roles(
                runs,
                role_lookup,
                (role,),
                split,
            )

    clean = groups["test-clean"]
    other = groups["test-other"]
    flagship = clean["flagship"]
    flagship_other = other["flagship"]
    flagship_identity = {
        (
            run.summary["system"],
            run.summary["variant"],
            run.summary["bitrate_kbps"]["nominal"],
            run.summary["token_frame_rate"],
        )
        for run in flagship + flagship_other
    }
    if len(flagship_identity) != 1:
        raise ValueError("Frozen flagship result prose mixes operating points")

    def pair(role: str, key: str, digits: int = 3, scale: float = 1.0) -> str:
        return _result_pair(
            clean[role],
            other[role],
            key=key,
            digits=digits,
            scale=scale,
        )

    flagship_name = _system_label(flagship[0].summary)
    flagship_rate = _format(flagship[0].summary["bitrate_kbps"]["nominal"])
    flagship_hz = _format(flagship[0].summary["token_frame_rate"], 2)
    low_rate = _format(clean["ours_075_seed2026"][0].summary["bitrate_kbps"]["nominal"])
    base_clean, base_other = (
        clean["base_m_seed2026"][0],
        other["base_m_seed2026"][0],
    )
    fullband_clean, fullband_other = (
        clean["joint_fullband_seed2026"][0],
        other["joint_fullband_seed2026"][0],
    )
    random_clean, random_other = (
        clean["random_groups_seed2026"][0],
        other["random_groups_seed2026"][0],
    )

    return "\n".join(
        [
            "% Generated from frozen test runs. Do not edit values by hand.",
            r"\paragraph{Rate--quality.}",
            (
                f"Table~\\ref{{tab:main-results}} and Fig.~\\ref{{fig:rate-quality}} "
                f"summarize the frozen test results. The three-seed flagship "
                f"{flagship_name} operates at {flagship_rate} kbps and "
                f"{flagship_hz} frames/s. On test-clean/test-other, its WER "
                f"(\\%) is {_result_pair(flagship, flagship_other, key='wer_reconstruction', digits=2, scale=100.0)}, "
                f"$\\Delta$WER (percentage points) is {_result_pair(flagship, flagship_other, key='wer_delta', digits=2, scale=100.0)}, "
                f"and ViSQOL/PESQ/speaker cosine are "
                f"{_result_pair(flagship, flagship_other, key='visqol_speech')}, "
                f"{_result_pair(flagship, flagship_other, key='pesq_wb')}, and "
                f"{_result_pair(flagship, flagship_other, key='speaker_cosine')}, respectively."
            ),
            (
                f"The bracketing DAC q=2/q=3 WER values are "
                f"{pair('dac_q2', 'wer_reconstruction', 2, 100.0)} and "
                f"{pair('dac_q3', 'wer_reconstruction', 2, 100.0)}, while their "
                f"speaker cosine values are {pair('dac_q2', 'speaker_cosine')} and "
                f"{pair('dac_q3', 'speaker_cosine')}. At the {low_rate}-kbps "
                f"boundary, Ours-0.75 and DAC q=1 obtain WER values "
                f"{pair('ours_075_seed2026', 'wer_reconstruction', 2, 100.0)} "
                f"and {pair('dac_q1', 'wer_reconstruction', 2, 100.0)}."
            ),
            r"\paragraph{Spectral controls.}",
            (
                f"On clean/other, Mel Oracle reaches ViSQOL "
                f"{pair('mel_oracle', 'visqol_speech')}, PESQ "
                f"{pair('mel_oracle', 'pesq_wb')}, and speaker cosine "
                f"{pair('mel_oracle', 'speaker_cosine')}. At 37.5 kbps, direct "
                f"and decoder-adapted dMel obtain WER (\\%) "
                f"{pair('dmel_same_frontend', 'wer_reconstruction', 2, 100.0)} "
                f"and {pair('dmel_quantization_aware_decoder', 'wer_reconstruction', 2, 100.0)}, "
                f"with speaker cosine {pair('dmel_same_frontend', 'speaker_cosine')} "
                f"and {pair('dmel_quantization_aware_decoder', 'speaker_cosine')}. "
                f"These controls are not rate-matched to the low-rate codecs."
            ),
            r"\paragraph{Grouping and stream dependence.}",
            (
                f"At the matched 1.746-kbps point, base grouped Ours, joint "
                f"full-band, and random-group controls obtain WER (\\%) "
                f"{pair('base_m_seed2026', 'wer_reconstruction', 2, 100.0)}, "
                f"{pair('joint_fullband_seed2026', 'wer_reconstruction', 2, 100.0)}, "
                f"and {pair('random_groups_seed2026', 'wer_reconstruction', 2, 100.0)}; "
                f"their speaker cosine values are "
                f"{pair('base_m_seed2026', 'speaker_cosine')}, "
                f"{pair('joint_fullband_seed2026', 'speaker_cosine')}, and "
                f"{pair('random_groups_seed2026', 'speaker_cosine')}."
            ),
            (
                f"Their mean pairwise adjusted NMI values are "
                f"{_dependency_pair(base_clean, base_other, 'adjusted_nmi')}, "
                f"{_dependency_pair(fullband_clean, fullband_other, 'adjusted_nmi')}, "
                f"and {_dependency_pair(random_clean, random_other, 'adjusted_nmi')}, "
                f"with probe lifts {_dependency_pair(base_clean, base_other, 'probe_lift')}, "
                f"{_dependency_pair(fullband_clean, fullband_other, 'probe_lift')}, "
                f"and {_dependency_pair(random_clean, random_other, 'probe_lift')}. "
                f"These dependence estimates are interpreted jointly with reconstruction "
                f"quality; a lower value alone is not evidence of a better factorization."
            ),
            "",
        ]
    )


def _is_ablation(run: PaperRun) -> bool:
    system = str(run.summary["system"]).lower()
    return "joint full-band" in system or "random groups" in system


def _run_roles(
    run: PaperRun,
    role_lookup: dict[tuple[Any, ...], set[str]] | None,
) -> set[str] | None:
    if role_lookup is None:
        return None
    identity = _paper_identity(run)[:-1]
    if identity not in role_lookup:
        raise ValueError(f"Paper run lacks frozen roles: {run.summary['run_id']}")
    return role_lookup[identity]


def _include_in_main_table(
    run: PaperRun,
    *,
    run_roles: set[str] | None = None,
    display_roles: set[str] | None = None,
) -> bool:
    if run_roles is not None:
        if display_roles is None:
            raise ValueError("Formal main-table filtering needs frozen display roles")
        return bool(run_roles & display_roles)
    summary = run.summary
    if str(summary["variant"]).lower() == "dac":
        return int(summary["num_parallel_streams"]) in {2, 3}
    system = str(summary["system"]).lower()
    return system not in {"ground truth", "identity", "ours-h"}


def _include_in_rate_quality(
    run: PaperRun,
    *,
    run_roles: set[str] | None = None,
    display_roles: set[str] | None = None,
) -> bool:
    if run_roles is not None:
        if display_roles is None:
            raise ValueError("Formal rate-curve filtering needs frozen display roles")
        return bool(run_roles & display_roles)
    summary = run.summary
    variant = str(summary["variant"]).lower()
    system = str(summary["system"]).lower()
    if variant == "dac":
        return True
    return system in {"ours-0.75", "ours-l", "ours-m", "ours-h"} or system.startswith(
        "ours-m+"
    )


def _rate_quality_runs(
    runs: list[PaperRun],
    *,
    role_lookup: dict[tuple[Any, ...], set[str]] | None,
    display_roles: set[str] | None,
) -> list[PaperRun]:
    selected = [
        run
        for run in runs
        if _include_in_rate_quality(
            run,
            run_roles=_run_roles(run, role_lookup),
            display_roles=display_roles,
        )
    ]
    if role_lookup is not None:
        return selected
    mplus_splits = {
        str(run.summary["split"])
        for run in selected
        if str(run.summary["system"]).lower().startswith("ours-m+")
    }
    return [
        run
        for run in selected
        if not (
            str(run.summary["system"]).lower() == "ours-m"
            and str(run.summary["split"]) in mplus_splits
        )
    ]


def _latex_table(
    runs: list[PaperRun],
    *,
    ablation: bool,
    role_lookup: dict[tuple[Any, ...], set[str]] | None = None,
    display_roles: set[str] | None = None,
) -> str:
    if ablation:
        if role_lookup is not None:
            if display_roles is None:
                raise ValueError("Formal ablation table needs frozen display roles")
            selected = [
                run
                for run in runs
                if bool(_run_roles(run, role_lookup) & display_roles)
            ]
        else:
            selected = [run for run in runs if _is_ablation(run)]
            contexts = {(run.summary["split"], run.summary["seed"]) for run in selected}
            selected.extend(
                run
                for run in runs
                if not _is_ablation(run)
                and str(run.summary["system"]).lower() == "ours-m"
                and (run.summary["split"], run.summary["seed"]) in contexts
            )
    else:
        selected = [
            run
            for run in runs
            if not _is_ablation(run)
            and _include_in_main_table(
                run,
                run_roles=_run_roles(run, role_lookup),
                display_roles=display_roles,
            )
        ]
    selected = list({run.summary["run_id"]: run for run in selected}.values())
    groups = [[run] for run in selected] if ablation else _seed_groups(selected)
    groups.sort(
        key=lambda group: (
            str(group[0].summary["split"]),
            float(group[0].summary["bitrate_kbps"]["nominal"] or 0),
            str(group[0].summary["system"]),
            int(group[0].summary["seed"]),
        ),
    )
    if not groups:
        kind = "ablation" if ablation else "main-table"
        return f"% No matching {kind} runs were supplied.\n"
    rows = [
        " & ".join(
            _ablation_table_row(group[0]) if ablation else _main_table_row(group)
        )
        + r" \\"
        for group in groups
    ]
    caption = (
        (
            "Matched grouped and full-band controls. Cells show point estimates. "
            "Adj. NMI is mean pairwise permutation-adjusted NMI; probe lift is "
            "accuracy above the training-majority baseline."
        )
        if ablation
        else (
            "LibriTTS reconstruction results. Multi-seed rows report mean $\\pm$ "
            "sample standard deviation across training seeds; single-seed rows show "
            "point estimates. Ground Truth, DAC curve endpoints, and the existing "
            "Ours-H upper-rate reference remain in the traceable result artifacts "
            "rather than this compact table."
        )
    )
    label = "tab:ablations" if ablation else "tab:main-results"
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.6pt}",
            (
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrr@{}}"
                if ablation
                else r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrrr@{}}"
            ),
            r"\toprule",
            (
                r"System & Split & kbps & WER\% & ViSQOL & PESQ & Spk. & Adj. NMI & Probe lift \\"
                if ablation
                else r"System & Split & Seeds & kbps & Hz & WER\% & $\Delta$WER & ViSQOL & PESQ & Spk. \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular*}",
            rf"\caption{{{caption} Per-run confidence intervals, entropy bitrate, parameter counts, and RTF remain in the result manifest and CSV. $L$: LibriTTS-trained component; $E$: externally pretrained component; $A$: adapted dMel; $C$: control.}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )


def _csv_value(run: PaperRun, metric: str) -> Any:
    summary = run.summary
    if metric == "wer_reconstruction":
        return summary.get("corpus_wer", {}).get("reconstruction")
    if metric == "delta_wer":
        return summary.get("corpus_wer", {}).get("delta")
    return summary.get("metrics", {}).get(metric, {}).get("mean")


def _csv_table(
    runs: Iterable[PaperRun],
    *,
    role_lookup: dict[tuple[Any, ...], set[str]] | None = None,
    main_display_roles: set[str] | None = None,
    rate_display_roles: set[str] | None = None,
    rate_included_directories: set[Path] | None = None,
) -> str:
    stream = io.StringIO(newline="")
    fieldnames = [
        "run_id",
        "system",
        "variant",
        "seed",
        "split",
        "utterances",
        "failed_utterances",
        "nominal_bitrate_kbps",
        "entropy_bitrate_kbps",
        "token_frame_rate",
        "codec_parameters",
        "external_pretrained_parameters",
        "wer_reconstruction",
        "delta_wer",
        "visqol_speech",
        "pesq_wb",
        "speaker_cosine",
        "encoder_rtf",
        "decoder_rtf",
        "config_hash",
        "checkpoint_hash",
        "per_utterance_sha256",
        "cut_ids_sha256",
        "roles",
        "included_in_main_table",
        "included_in_rate_quality",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for run in runs:
        summary = run.summary
        complexity = summary.get("model_complexity") or {}
        roles = _run_roles(run, role_lookup)
        writer.writerow(
            {
                "run_id": summary["run_id"],
                "system": summary["system"],
                "variant": summary["variant"],
                "seed": summary["seed"],
                "split": summary["split"],
                "utterances": summary["utterances"],
                "failed_utterances": summary["failed_utterances"],
                "nominal_bitrate_kbps": summary["bitrate_kbps"]["nominal"],
                "entropy_bitrate_kbps": summary["bitrate_kbps"]["entropy"],
                "token_frame_rate": summary["token_frame_rate"],
                "codec_parameters": complexity.get("codec_parameters"),
                "external_pretrained_parameters": complexity.get(
                    "external_pretrained_parameters"
                ),
                "wer_reconstruction": _csv_value(run, "wer_reconstruction"),
                "delta_wer": _csv_value(run, "delta_wer"),
                "visqol_speech": _csv_value(run, "visqol_speech"),
                "pesq_wb": _csv_value(run, "pesq_wb"),
                "speaker_cosine": _csv_value(run, "speaker_cosine"),
                "encoder_rtf": summary.get("rtf", {}).get("encoder"),
                "decoder_rtf": summary.get("rtf", {}).get("decoder"),
                "config_hash": summary["config_hash"],
                "checkpoint_hash": summary["checkpoint_hash"],
                "per_utterance_sha256": summary["per_utterance_sha256"],
                "cut_ids_sha256": run.cut_ids_sha256,
                "roles": "|".join(sorted(roles)) if roles is not None else "",
                "included_in_main_table": (
                    not _is_ablation(run)
                    and _include_in_main_table(
                        run,
                        run_roles=roles,
                        display_roles=main_display_roles,
                    )
                ),
                "included_in_rate_quality": (
                    not _is_ablation(run)
                    and (
                        run.directory in rate_included_directories
                        if rate_included_directories is not None
                        else _include_in_rate_quality(
                            run,
                            run_roles=roles,
                            display_roles=rate_display_roles,
                        )
                    )
                ),
            }
        )
    return stream.getvalue()


def _atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    figure.savefig(
        temporary,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "dmel_codec.tools.make_paper_assets",
            "CreationDate": None,
        },
    )
    plt.close(figure)
    os.replace(temporary, path)


def _method_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.1, 2.15))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    stages = [
        (0.01, 0.10, "Waveform", "$x$", "#e8e8e8"),
        (0.14, 0.11, "Log-mel", "$100 \\times T$", "#c6dbef"),
        (0.28, 0.12, "Contiguous\nsplit", "10 bands", "#9ecae1"),
        (0.43, 0.11, "Shared\nencoder", "tied $E$", "#b8e0d2"),
        (0.57, 0.13, "Downsample\n+ FSQ", "$\\downarrow 4$, per band", "#f6d6a8"),
        (0.73, 0.12, "Mel\ndecoder", "zero input", "#d9d2e9"),
        (0.88, 0.11, "Fixed\nBigVGAN", "external", "#f2c6c2"),
    ]
    center_y = 0.57
    height = 0.31
    for x, width, title, subtitle, color in stages:
        axis.add_patch(
            FancyBboxPatch(
                (x, center_y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                linewidth=0.8,
                edgecolor="#444444",
                facecolor=color,
            )
        )
        axis.text(
            x + width / 2,
            center_y + 0.045,
            title,
            ha="center",
            va="center",
            fontsize=6.9,
            fontweight="semibold",
            linespacing=0.95,
        )
        axis.text(
            x + width / 2,
            center_y - 0.07,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#333333",
        )

    for left, right in zip(stages, stages[1:]):
        start = left[0] + left[1] + 0.003
        end = right[0] - 0.003
        axis.add_patch(
            FancyArrowPatch(
                (start, center_y),
                (end, center_y),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color="#333333",
            )
        )

    fsq_x = stages[4][0] + stages[4][1] / 2
    axis.text(
        fsq_x,
        0.20,
        "10 flat parallel streams at 23.4375 frames/s",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="semibold",
    )
    axis.text(
        fsq_x,
        0.075,
        "$K \\in \\{9,25,175,1000\\}$ per stream; no residual ordering",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#333333",
    )
    axis.add_patch(
        FancyArrowPatch(
            (fsq_x, center_y - height / 2 - 0.01),
            (fsq_x, 0.26),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.7,
            color="#555555",
        )
    )
    _atomic_figure(path, figure)


def _rate_family(system: str) -> tuple[str, str, str]:
    lowered = system.lower()
    if lowered.startswith("ours"):
        return "Ours", "#d55e00", "o"
    if lowered.startswith("dac"):
        return "DAC", "#0072b2", "s"
    if lowered.startswith("dmel-quantization-aware-decoder"):
        return "Adapted dMel + decoder", "#009e73", "P"
    if lowered.startswith("dmel"):
        return "Adapted dMel", "#009e73", "D"
    return system, "#666666", "^"


def _rate_quality_figure(runs: list[PaperRun], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    series: dict[tuple[str, str], list[list[PaperRun]]] = {}
    for group in _seed_groups(runs):
        rate, _ = _seed_estimate(group, "nominal_bitrate_kbps")
        if rate is None or rate <= 0:
            continue
        family, _, _ = _rate_family(str(group[0].summary["system"]))
        split = str(group[0].summary["split"])
        series.setdefault((family, split), []).append(group)

    for (family, split), groups in sorted(series.items()):
        groups.sort(
            key=lambda group: float(group[0].summary["bitrate_kbps"]["nominal"])
        )
        _, color, marker = _rate_family(str(groups[0][0].summary["system"]))
        linestyle = "--" if split.endswith("other") else "-"
        label = f"{family}, {split}"
        for axis, metric, scale in zip(
            axes,
            ("visqol_speech", "wer_delta"),
            (1.0, 100.0),
        ):
            points = []
            for group in groups:
                rate, _ = _seed_estimate(group, "nominal_bitrate_kbps")
                value, deviation = _seed_estimate(group, metric)
                if rate is not None and value is not None:
                    points.append((rate, value * scale, deviation))
            if not points:
                continue
            rates = [point[0] for point in points]
            values = [point[1] for point in points]
            deviations = [
                0.0 if point[2] is None else point[2] * scale for point in points
            ]
            axis.errorbar(
                rates,
                values,
                yerr=deviations if any(deviations) else None,
                color=color,
                marker=marker,
                markersize=4.5,
                markerfacecolor="white" if split.endswith("other") else color,
                linewidth=1.0,
                linestyle=linestyle if len(points) > 1 else "none",
                capsize=2.0,
                label=label,
            )
    axes[0].set_ylabel("ViSQOL MOS-LQO")
    axes[1].set_ylabel(r"$\Delta$WER (pp)")
    for axis in axes:
        axis.set_xlabel("Nominal bitrate (kbps)")
        axis.set_xscale("log")
        axis.grid(True, color="#d9d9d9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            fontsize=6,
        )
        figure.subplots_adjust(bottom=0.34, wspace=0.33)
    else:
        figure.subplots_adjust(wspace=0.33)
    _atomic_figure(path, figure)


def _dependency_figure(runs: list[PaperRun], path: Path) -> None:
    dependency_runs = [run for run in runs if run.dependency is not None]
    labels = [
        f"{run.summary['system']}\n{run.summary['split']}, s{run.summary['seed']}"
        for run in dependency_runs
    ]
    nmi = [
        mean(
            float(pair["normalized_mutual_information"])
            for pair in run.dependency["pairs"]
        )
        for run in dependency_runs
    ]
    lifts = [
        mean(float(item["accuracy_lift"]) for item in run.dependency["probe_results"])
        for run in dependency_runs
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    positions = list(range(len(labels)))
    axes[0].bar(positions, nmi, color="#4477aa")
    axes[1].bar(positions, lifts, color="#228833")
    axes[0].set_ylabel("Mean pairwise adjusted NMI")
    axes[1].set_ylabel("Mean probe accuracy lift")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right", fontsize=7)
        axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(bottom=0.31, wspace=0.34)
    _atomic_figure(path, figure)


def generate_paper_assets(
    run_directories: Iterable[str | Path],
    *,
    output_directory: str | Path,
    protocol: str | Path | None = None,
    allow_incomplete: bool = False,
    test_gate: str | Path | None = None,
) -> dict[str, Any]:
    protocol = Path(protocol) if protocol is not None else None
    if protocol is not None and not protocol.is_file():
        raise FileNotFoundError(protocol)
    runs = [
        _load_run(Path(directory), allow_incomplete=allow_incomplete)
        for directory in run_directories
    ]
    _validate_runs(runs)
    if test_gate is not None and allow_incomplete:
        raise ValueError("Frozen test-gate assets cannot use --allow-incomplete")
    test_gate_binding = None
    role_lookup = None
    main_display_roles = None
    ablation_display_roles = None
    rate_display_roles = None
    if test_gate is not None:
        test_gate_binding = _validate_test_gate_coverage(runs, test_gate)
        role_lookup = _test_gate_role_lookup(test_gate)
        if (
            sha256_file(Path(test_gate).expanduser().resolve())
            != test_gate_binding["sha256"]
        ):
            raise ValueError("Frozen test gate changed during paper generation")
        main_display_roles = set(test_gate_binding["paper_display"]["main_table_roles"])
        ablation_display_roles = set(
            test_gate_binding["paper_display"]["ablation_table_roles"]
        )
        rate_display_roles = set(
            test_gate_binding["paper_display"]["rate_quality_roles"]
        )
    if test_gate is None and any(
        str(run.summary["split"]).startswith("test") for run in runs
    ):
        raise ValueError("Test paper assets require a frozen --test-gate")
    runs.sort(
        key=lambda run: (
            str(run.summary["split"]),
            str(run.summary["system"]),
            int(run.summary["seed"]),
        )
    )

    output = Path(output_directory)
    tables = output / "tables"
    figures = output / "figures"
    artifacts = output / "artifacts"
    main_tex = tables / "main_results.tex"
    ablation_tex = tables / "ablations.tex"
    result_summary_tex = tables / "results_summary.tex"
    main_csv = artifacts / "main_results.csv"
    ablation_csv = artifacts / "ablations.csv"
    seed_aggregates = artifacts / "seed_aggregates.json"
    method_figure = figures / "method.pdf"
    rate_figure = figures / "rate_quality.pdf"
    dependency_figure = figures / "code_dependency.pdf"

    main_runs = [run for run in runs if not _is_ablation(run)]
    ablation_runs = [run for run in runs if _is_ablation(run)]
    rate_runs = _rate_quality_runs(
        main_runs,
        role_lookup=role_lookup,
        display_roles=rate_display_roles,
    )
    rate_run_directories = {run.directory for run in rate_runs}
    _atomic_text(
        main_tex,
        _latex_table(
            runs,
            ablation=False,
            role_lookup=role_lookup,
            display_roles=main_display_roles,
        ),
    )
    _atomic_text(
        ablation_tex,
        _latex_table(
            runs,
            ablation=True,
            role_lookup=role_lookup,
            display_roles=ablation_display_roles,
        ),
    )
    _atomic_text(result_summary_tex, _result_summary_tex(runs, role_lookup))
    _atomic_text(
        main_csv,
        _csv_table(
            main_runs,
            role_lookup=role_lookup,
            main_display_roles=main_display_roles,
            rate_display_roles=rate_display_roles,
            rate_included_directories=rate_run_directories,
        ),
    )
    _atomic_text(
        ablation_csv,
        _csv_table(
            ablation_runs,
            role_lookup=role_lookup,
            main_display_roles=main_display_roles,
            rate_display_roles=rate_display_roles,
            rate_included_directories=rate_run_directories,
        ),
    )
    _atomic_json(seed_aggregates, _seed_aggregate_payload(main_runs))
    _method_figure(method_figure)
    _rate_quality_figure(rate_runs, rate_figure)
    _dependency_figure(runs, dependency_figure)

    generated_paths = (
        main_tex,
        ablation_tex,
        result_summary_tex,
        main_csv,
        ablation_csv,
        seed_aggregates,
        method_figure,
        rate_figure,
        dependency_figure,
    )
    manifest = {
        "schema_version": 2,
        "status": (
            "PAPER_ASSETS_FROZEN"
            if test_gate_binding is not None
            else "PAPER_ASSETS_DIAGNOSTIC"
        ),
        "protocol": (
            {"name": protocol.name, "sha256": sha256_file(protocol)}
            if protocol is not None
            else None
        ),
        "allow_incomplete": allow_incomplete,
        "test_gate": test_gate_binding,
        "paper_display": (
            test_gate_binding["paper_display"]
            if test_gate_binding is not None
            else {
                "schema_version": 1,
                "status": "DIAGNOSTIC_NAME_FALLBACK",
            }
        ),
        "runs": [
            {
                "run_id": run.summary["run_id"],
                "system": run.summary["system"],
                "variant": run.summary["variant"],
                "seed": run.summary["seed"],
                "split": run.summary["split"],
                "utterances": run.summary["utterances"],
                "config_hash": run.summary["config_hash"],
                "checkpoint_hash": run.summary["checkpoint_hash"],
                "cut_ids_sha256": run.cut_ids_sha256,
                "primary_metric_exclusions": run.primary_metric_exclusions,
                "roles": sorted(_run_roles(run, role_lookup) or []),
                "included_in_main_table": (
                    not _is_ablation(run)
                    and _include_in_main_table(
                        run,
                        run_roles=_run_roles(run, role_lookup),
                        display_roles=main_display_roles,
                    )
                ),
                "included_in_rate_quality": (
                    not _is_ablation(run) and run.directory in rate_run_directories
                ),
                "sources": run.source_hashes,
            }
            for run in runs
        ],
        "generated": {
            str(path.relative_to(output)): sha256_file(path) for path in generated_paths
        },
    }
    manifest_path = artifacts / "result_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate traceable LaTeX, CSV, and PDF paper assets"
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--test-gate", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing primary metrics, bootstrap CIs, and dependencies for smoke only",
    )
    args = parser.parse_args()
    if not args.allow_incomplete and args.test_gate is None:
        parser.error("formal paper assets require --test-gate")
    manifest = generate_paper_assets(
        args.run_dir,
        output_directory=args.output_dir,
        protocol=args.protocol,
        allow_incomplete=args.allow_incomplete,
        test_gate=args.test_gate,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
