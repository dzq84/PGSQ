from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml

from dmel_codec.evaluation.cache import canonical_hash, sha256_file
from dmel_codec.evaluation.run_codec_eval import DEFAULT_METRICS
from dmel_codec.evaluation.statistics import load_jsonl
from dmel_codec.evaluation.summarize import summarize_records
from dmel_codec.evaluation.validity import (
    PRIMARY_METRICS,
    validate_primary_metric_domain,
)
from dmel_codec.tools import mplus_replication, ours_075_release


FROZEN_SOURCES = (
    "dmel_codec/evaluation/run_codec_eval.py",
    "dmel_codec/evaluation/metrics.py",
    "dmel_codec/evaluation/summarize.py",
    "dmel_codec/evaluation/statistics.py",
    "dmel_codec/evaluation/dependency.py",
    "dmel_codec/evaluation/validity.py",
    "dmel_codec/tools/freeze_test_gate.py",
    "dmel_codec/tools/mplus_replication.py",
    "dmel_codec/tools/ours_075_release.py",
    "dmel_codec/tools/make_paper_assets.py",
    "dmel_codec/tools/test_queue.py",
    "scripts/evaluate.sh",
    "scripts/run_test_queue.sh",
    "scripts/statistics.sh",
)
DEFAULT_SYSTEM_REQUIREMENTS = Path(
    "dmel_codec/config/evaluation/test_system_requirements.yaml"
)


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


def _verified_paper_display(requirements: dict[str, Any]) -> dict[str, Any]:
    display = requirements.get("paper_display")
    roles = requirements.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Invalid final-system requirements roles")
    if not isinstance(display, dict) or display.get("schema_version") != 1:
        raise ValueError("Final-system requirements need paper_display schema v1")
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
            raise ValueError(f"paper_display.{field} must be a non-empty role list")
        if len(values) != len(set(values)):
            raise ValueError(f"paper_display.{field} contains duplicate roles")
        unknown = sorted(set(values) - set(roles))
        if unknown:
            raise ValueError(f"paper_display.{field} has unknown roles: {unknown}")
        result[field] = list(values)
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _verified_run(directory: Path, expected_metrics: list[str]) -> dict[str, Any]:
    directory = directory.resolve()
    required = (
        "per_utterance.jsonl",
        "summary.json",
        "resolved_config.yaml",
        "model_metadata.json",
        "data_manifest.json",
        "git_state.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete dev result {directory}: {missing}")

    records_path = directory / "per_utterance.jsonl"
    records = load_jsonl(records_path)
    cut_ids = [str(record["cut_id"]) for record in records]
    if len(cut_ids) != len(set(cut_ids)):
        raise ValueError(f"Dev run contains duplicate cut IDs: {directory}")
    rebuilt = summarize_records(records)
    rebuilt["per_utterance_sha256"] = sha256_file(records_path)
    summary = _load_json(directory / "summary.json")
    if summary != rebuilt:
        raise ValueError(f"Summary does not rebuild exactly: {directory}")
    if not str(summary["split"]).startswith("dev"):
        raise ValueError(
            f"Test gate selections must come from dev, got {summary['split']}"
        )

    config = yaml.safe_load(
        (directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    if config.get("max_cuts") is not None:
        raise ValueError(f"Truncated dev run cannot enter the test gate: {directory}")
    if sorted(config.get("metrics", [])) != expected_metrics:
        raise ValueError(f"Dev metric set does not match frozen metrics: {directory}")
    if config.get("system_config_hash") != summary["config_hash"]:
        raise ValueError(f"Resolved config hash disagrees with summary: {directory}")

    data_manifest = _load_json(directory / "data_manifest.json")
    if int(data_manifest.get("cuts", -1)) != int(summary["utterances"]):
        raise ValueError(f"Dev run does not cover its complete manifest: {directory}")
    model_metadata = _load_json(directory / "model_metadata.json")
    revision = model_metadata.get("system", {}).get("checkpoint_revision_and_hash", {})
    raw_checkpoint_sha256 = revision.get("checkpoint_sha256")
    if raw_checkpoint_sha256 is None and config.get("system") == "dmel_decoder":
        decoder_revision = revision.get("quantization_aware_decoder", {})
        if isinstance(decoder_revision, dict):
            raw_checkpoint_sha256 = decoder_revision.get("checkpoint_sha256")
    if raw_checkpoint_sha256 is not None and (
        not isinstance(raw_checkpoint_sha256, str) or len(raw_checkpoint_sha256) != 64
    ):
        raise ValueError(f"Invalid raw checkpoint SHA-256 in {directory}")
    git_state = _load_json(directory / "git_state.json")
    primary_metrics = sorted(set(expected_metrics) & set(PRIMARY_METRICS))
    primary_metric_exclusions = validate_primary_metric_domain(
        records, required_metrics=primary_metrics
    )
    return {
        "adapter": config["system"],
        "system": summary["system"],
        "variant": summary["variant"],
        "seed": int(summary["seed"]),
        "config_hash": summary["config_hash"],
        "checkpoint_hash": summary["checkpoint_hash"],
        "raw_checkpoint_sha256": raw_checkpoint_sha256,
        "decode_seed_or_policy": config["decode_seed_or_policy"],
        "selection_split": summary["split"],
        "selection_run_id": summary["run_id"],
        "selection_summary_sha256": sha256_file(directory / "summary.json"),
        "selection_per_utterance_sha256": summary["per_utterance_sha256"],
        "cut_ids_sha256": canonical_hash(sorted(cut_ids)),
        "metric_metadata_hash": canonical_hash(model_metadata["metrics"]),
        "primary_metric_exclusions": primary_metric_exclusions,
        "nominal_bitrate_kbps": float(summary["bitrate_kbps"]["nominal"]),
        "token_frame_rate_hz": float(summary["token_frame_rate"]),
        "num_parallel_streams": int(summary["num_parallel_streams"]),
        "codebook_cardinalities": [
            int(value) for value in summary["codebook_cardinalities"]
        ],
        "git": git_state,
    }


def _system_identity(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["adapter"],
        entry["system"],
        entry["variant"],
        entry["seed"],
        entry["config_hash"],
        entry["checkpoint_hash"],
        entry["decode_seed_or_policy"],
    )


def _validate_role(
    role: str, entry: dict[str, Any], requirement: dict[str, Any]
) -> None:
    for field in ("adapter", "variant", "seed", "num_parallel_streams"):
        if entry[field] != requirement[field]:
            raise ValueError(
                f"Final-system role {role} has unexpected {field}: "
                f"{entry[field]!r} != {requirement[field]!r}"
            )
    systems = requirement.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"Final-system requirement {role} has no allowed systems")
    if entry["system"] not in systems:
        raise ValueError(
            f"Final-system role {role} has unexpected system: "
            f"{entry['system']!r} not in {systems!r}"
        )
    for actual_field, requirement_field in (
        ("nominal_bitrate_kbps", "nominal_bitrate_kbps"),
        ("token_frame_rate_hz", "token_frame_rate_hz"),
    ):
        if not math.isclose(
            float(entry[actual_field]),
            float(requirement[requirement_field]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                f"Final-system role {role} has unexpected {actual_field}: "
                f"{entry[actual_field]!r} != {requirement[requirement_field]!r}"
            )
    cardinality = requirement.get("codebook_cardinality")
    expected_cardinalities = (
        []
        if cardinality is None
        else [int(cardinality)] * int(requirement["num_parallel_streams"])
    )
    if entry["codebook_cardinalities"] != expected_cardinalities:
        raise ValueError(
            f"Final-system role {role} has unexpected codebook cardinalities"
        )


def _verified_registry(
    registry_path: Path,
    requirements_path: Path,
    expected_metrics: list[str],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[Any, ...], list[str]],
    dict[str, dict[str, Any]],
]:
    registry = _load_yaml(registry_path)
    requirements = _load_yaml(requirements_path)
    if registry.get("schema_version") != 1 or registry.get("status") != (
        "FINAL_SYSTEMS_PREREGISTERED"
    ):
        raise ValueError("Invalid or incomplete final-system registry")
    if requirements.get("schema_version") != 1 or not isinstance(
        requirements.get("roles"), dict
    ):
        raise ValueError("Invalid final-system requirements")
    registered = registry.get("systems")
    if not isinstance(registered, list):
        raise ValueError("Final-system registry systems must be a list")
    by_role: dict[str, dict[str, Any]] = {}
    for item in registered:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ValueError("Every final-system registry item needs a role")
        role = item["role"]
        if role in by_role:
            raise ValueError(f"Duplicate final-system role: {role}")
        by_role[role] = item
    required_roles = set(requirements["roles"])
    registered_roles = set(by_role)
    if registered_roles != required_roles:
        missing = sorted(required_roles - registered_roles)
        extra = sorted(registered_roles - required_roles)
        raise ValueError(
            f"Final-system registry role mismatch; missing={missing}, extra={extra}"
        )

    unique_runs: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    roles_by_identity: dict[tuple[Any, ...], list[str]] = {}
    consistency_groups: dict[str, list[dict[str, Any]]] = {}
    entry_by_role: dict[str, dict[str, Any]] = {}
    for role in sorted(required_roles):
        item = by_role[role]
        requirement = requirements["roles"][role]
        if not isinstance(requirement, dict):
            raise ValueError(f"Invalid final-system requirement: {role}")
        pair = []
        for split, key in (
            ("dev-clean", "dev_clean_run"),
            ("dev-other", "dev_other_run"),
        ):
            raw_path = item.get(key)
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                raise ValueError(f"Final-system role {role} requires an absolute {key}")
            entry = _verified_run(Path(raw_path), expected_metrics)
            if entry["selection_split"] != split:
                raise ValueError(
                    f"Final-system role {role} {key} must be {split}, "
                    f"got {entry['selection_split']}"
                )
            _validate_role(role, entry, requirement)
            pair.append(entry)
        clean, other = pair
        entry_by_role[role] = clean
        identity = _system_identity(clean)
        if _system_identity(other) != identity:
            raise ValueError(
                f"Final-system role {role} has different dev-clean/dev-other identities"
            )
        roles_by_identity.setdefault(identity, []).append(role)
        for entry in pair:
            key = (identity, entry["selection_split"])
            previous = unique_runs.get(key)
            if previous is not None and (
                previous["selection_run_id"] != entry["selection_run_id"]
                or previous["selection_summary_sha256"]
                != entry["selection_summary_sha256"]
                or previous["selection_per_utterance_sha256"]
                != entry["selection_per_utterance_sha256"]
            ):
                raise ValueError(
                    "Roles sharing one system identity must reference the same dev runs"
                )
            unique_runs[key] = entry
        group = requirement.get("consistency_group")
        if group is not None:
            consistency_groups.setdefault(str(group), []).append(clean)

    consistency_fields = (
        "adapter",
        "system",
        "variant",
        "nominal_bitrate_kbps",
        "token_frame_rate_hz",
        "num_parallel_streams",
        "codebook_cardinalities",
        "decode_seed_or_policy",
    )
    for group, entries in consistency_groups.items():
        first = entries[0]
        for entry in entries[1:]:
            if any(entry[field] != first[field] for field in consistency_fields):
                raise ValueError(
                    f"Final-system consistency group {group} mixes operating points"
                )
    for roles in roles_by_identity.values():
        roles.sort()
    return list(unique_runs.values()), roles_by_identity, entry_by_role


def _verified_artifact_record(record: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an artifact record")
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{label} requires path and SHA-256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, observed_hash


def _raw_checkpoint(entry_by_role: dict[str, dict[str, Any]], role: str) -> str:
    value = entry_by_role[role].get("raw_checkpoint_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Final-system role {role} lacks a raw checkpoint SHA-256")
    return value


def _validate_mplus_provenance(
    *,
    selection_path: Path,
    replication_completion_paths: list[Path],
    entry_by_role: dict[str, dict[str, Any]],
    dev_commit: str,
    repository_root: Path,
) -> dict[str, Any]:
    selection_path = selection_path.expanduser().resolve()
    selection = ours_075_release.validate_direction_selection(
        selection_path, expected_commit=dev_commit
    )
    gate_path, gate_hash = _verified_artifact_record(
        selection.get("direction_gate"), label="M+ direction gate"
    )
    gate = _load_json(gate_path)
    selected = str(selection["selected_direction"])
    selected_identity = mplus_replication.selected_seed2026_identity(
        gate,
        selected,
        expected_commit=str(
            selection.get("direction_gate_evaluation_commit", dev_commit)
        ),
    )
    if selection.get("selected_seed2026_evaluation_identity") != selected_identity:
        raise ValueError(
            "M+ direction selection does not bind its selected seed-2026 identity"
        )
    expected_system = mplus_replication.SYSTEM_NAME_BY_ROUTE[selected]
    for seed in mplus_replication.FORMAL_SEEDS:
        role = f"flagship_m_seed{seed}"
        entry = entry_by_role[role]
        if entry["system"] != expected_system:
            raise ValueError(
                f"Final flagship role {role} disagrees with selected route {selected}"
            )

    source_hashes: dict[int, str] = {}
    for seed in mplus_replication.FORMAL_SEEDS:
        source = selection.get("source_checkpoints", {}).get(str(seed))
        if not isinstance(source, dict):
            raise ValueError(f"M+ direction selection lacks source seed {seed}")
        _, source_hash = _verified_artifact_record(
            source.get("checkpoint"), label=f"M+ source seed {seed} checkpoint"
        )
        source_hashes[seed] = source_hash
    if _raw_checkpoint(entry_by_role, "base_m_seed2026") != source_hashes[2026]:
        raise ValueError("Base-M final role differs from its frozen source checkpoint")

    completion_records: dict[int, dict[str, Any]] = {}
    if selected == "source":
        if replication_completion_paths:
            raise ValueError("Source-retained M+ selection cannot have replications")
        for seed in mplus_replication.FORMAL_SEEDS:
            role = f"flagship_m_seed{seed}"
            if _raw_checkpoint(entry_by_role, role) != source_hashes[seed]:
                raise ValueError(
                    f"Source-retained flagship role {role} uses another checkpoint"
                )
    else:
        if (
            _raw_checkpoint(entry_by_role, "flagship_m_seed2026")
            != (selected_identity["raw_checkpoint_sha256"])
        ):
            raise ValueError("Selected M+ seed-2026 final role uses another checkpoint")
        if len(replication_completion_paths) != 2:
            raise ValueError(
                "Promoted M+ direction requires seed-2027/2028 completion receipts"
            )
        for completion_path in replication_completion_paths:
            completion_path = completion_path.expanduser().resolve()
            completion = _load_json(completion_path)
            seed = int(completion.get("seed", -1))
            if (
                completion.get("schema_version") != 1
                or completion.get("status") != "MPLUS_REPLICATION_COMPLETE"
                or completion.get("direction") != selected
                or int(completion.get("final_global_step", -1)) != 100_000
                or seed not in mplus_replication.REPLICATION_SEEDS
                or seed in completion_records
            ):
                raise ValueError(
                    f"Invalid M+ replication completion: {completion_path}"
                )
            best_path, _ = _verified_artifact_record(
                completion.get("best_checkpoint"),
                label=f"M+ replication seed {seed} best checkpoint",
            )
            run_dir = best_path.parent.parent
            preflight = run_dir / "replication_preflight.json"
            validated_preflight = mplus_replication.validate_existing_preflight(
                selection=selection_path,
                preflight=preflight,
                repository_root=repository_root,
            )
            if validated_preflight.get("seed") != seed:
                raise ValueError("M+ replication preflight seed mismatch")
            rebuilt = mplus_replication.verify_final_selection(
                run_dir=run_dir, preflight=preflight
            )
            if completion != rebuilt:
                raise ValueError(
                    f"M+ replication completion does not rebuild: {completion_path}"
                )
            role = f"flagship_m_seed{seed}"
            if (
                _raw_checkpoint(entry_by_role, role)
                != rebuilt["best_checkpoint"]["sha256"]
            ):
                raise ValueError(f"Final flagship role {role} uses another checkpoint")
            completion_records[seed] = {
                "path": str(completion_path),
                "sha256": sha256_file(completion_path),
                "best_checkpoint_sha256": rebuilt["best_checkpoint"]["sha256"],
            }
        if set(completion_records) != set(mplus_replication.REPLICATION_SEEDS):
            raise ValueError("M+ replication completions do not cover seeds 2027/2028")

    return {
        "direction_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "direction_gate_sha256": gate_hash,
        "selected_direction": selected,
        "selected_system": expected_system,
        "selected_seed2026_checkpoint_sha256": selected_identity[
            "raw_checkpoint_sha256"
        ],
        "source_checkpoint_sha256_by_seed": {
            str(seed): value for seed, value in sorted(source_hashes.items())
        },
        "replication_completions": {
            str(seed): value for seed, value in sorted(completion_records.items())
        },
    }


def _validate_ours_075_provenance(
    *,
    release_path: Path,
    completion_path: Path,
    mplus_selection_path: Path,
    entry_by_role: dict[str, dict[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    release_path = release_path.expanduser().resolve()
    completion_path = completion_path.expanduser().resolve()
    release = ours_075_release.verify_release(
        release=release_path, repository_root=repository_root
    )
    selection_record = release["mplus_direction_selection"]
    if (
        Path(selection_record["path"]).expanduser().resolve()
        != mplus_selection_path.expanduser().resolve()
        or selection_record["sha256"] != sha256_file(mplus_selection_path)
    ):
        raise ValueError("Ours-0.75 release binds another M+ direction selection")
    completion = _load_json(completion_path)
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "OURS_075_TRAINING_COMPLETE"
        or int(completion.get("seed", -1)) != 2026
        or int(completion.get("final_global_step", -1)) != 100_000
    ):
        raise ValueError("Invalid Ours-0.75 training-completion receipt")
    completion_release_path, completion_release_hash = _verified_artifact_record(
        completion.get("release"), label="Ours-0.75 completion release"
    )
    if (
        completion_release_path != release_path
        or completion_release_hash != sha256_file(release_path)
    ):
        raise ValueError("Ours-0.75 completion binds another release")
    run_dir = Path(str(completion.get("run_dir", ""))).expanduser().resolve()
    rebuilt = ours_075_release.verify_final_selection(
        run_dir=run_dir,
        release=release_path,
        repository_root=repository_root,
    )
    if completion != rebuilt:
        raise ValueError("Ours-0.75 training-completion receipt does not rebuild")
    if (
        _raw_checkpoint(entry_by_role, "ours_075_seed2026")
        != rebuilt["best_checkpoint"]["sha256"]
    ):
        raise ValueError("Ours-0.75 final role uses another checkpoint")
    return {
        "formal_release": {
            "path": str(release_path),
            "sha256": sha256_file(release_path),
        },
        "training_completion": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
        },
        "best_checkpoint_sha256": rebuilt["best_checkpoint"]["sha256"],
    }


def freeze_test_gate(
    *,
    system_registry: str | Path,
    manifest_summary: str | Path,
    protocol: str | Path | None = None,
    output: str | Path,
    mplus_direction_selection: str | Path | None = None,
    mplus_replication_completions: Iterable[str | Path] = (),
    ours_075_release_marker: str | Path | None = None,
    ours_075_training_completion: str | Path | None = None,
    metrics: Iterable[str] | None = None,
    repository_root: str | Path = ".",
    allow_dirty: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and not replace:
        raise FileExistsError(f"Refusing to replace an existing test gate: {output}")
    manifest_summary = Path(manifest_summary)
    protocol = Path(protocol) if protocol is not None else None
    if not manifest_summary.is_file():
        raise FileNotFoundError("Manifest summary is required")
    if protocol is not None and not protocol.is_file():
        raise FileNotFoundError(protocol)
    expected_metrics = sorted(
        metrics if metrics is not None else DEFAULT_METRICS.split(",")
    )
    root = Path(repository_root).resolve()
    requirements_path = root / DEFAULT_SYSTEM_REQUIREMENTS
    if not requirements_path.is_file():
        raise FileNotFoundError(requirements_path)
    paper_display = _verified_paper_display(_load_yaml(requirements_path))
    registry_path = Path(system_registry).expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    verified_runs, roles_by_identity, entry_by_role = _verified_registry(
        registry_path, requirements_path, expected_metrics
    )
    if not verified_runs:
        raise ValueError("Complete dev-clean and dev-other runs are required")
    run_identities = [
        (_system_identity(entry), entry["selection_split"]) for entry in verified_runs
    ]
    if len(run_identities) != len(set(run_identities)):
        raise ValueError("Duplicate system/split selection in test gate inputs")
    dev_splits = {entry["selection_split"] for entry in verified_runs}
    required_dev_splits = {"dev-clean", "dev-other"}
    if dev_splits != required_dev_splits:
        raise ValueError(
            "Formal test gate requires both dev-clean and dev-other runs; "
            f"got {sorted(dev_splits)}"
        )
    for split in sorted(required_dev_splits):
        cut_id_hashes = {
            entry["cut_ids_sha256"]
            for entry in verified_runs
            if entry["selection_split"] == split
        }
        if len(cut_id_hashes) != 1:
            raise ValueError(f"{split} runs use different cut IDs")
        exclusion_profiles = {
            json.dumps(entry["primary_metric_exclusions"], sort_keys=True)
            for entry in verified_runs
            if entry["selection_split"] == split
        }
        if len(exclusion_profiles) != 1:
            raise ValueError(
                f"{split} runs use different primary-metric exclusion sets"
            )

    metric_hashes = {entry["metric_metadata_hash"] for entry in verified_runs}
    if len(metric_hashes) != 1:
        raise ValueError("Dev runs disagree on metric revisions or asset manifests")
    git_commits = {entry["git"].get("commit") for entry in verified_runs}
    if len(git_commits) != 1 or None in git_commits:
        raise ValueError("Dev runs do not share one git commit")
    dev_commit = next(iter(git_commits))
    dirty_runs = [
        entry["selection_run_id"]
        for entry in verified_runs
        if entry["git"].get("dirty")
    ]
    if dirty_runs and not allow_dirty:
        raise ValueError(f"Formal test gate requires clean dev runs: {dirty_runs}")

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for entry in verified_runs:
        grouped.setdefault(_system_identity(entry), {})[entry["selection_split"]] = (
            entry
        )
    incomplete = [
        identity
        for identity, entries in grouped.items()
        if set(entries) != required_dev_splits
    ]
    if incomplete:
        raise ValueError(
            "Every frozen system must have matched dev-clean and dev-other runs: "
            f"{incomplete}"
        )

    systems = []
    for entries in grouped.values():
        selection = dict(entries["dev-clean"])
        robustness = entries["dev-other"]
        selection.pop("metric_metadata_hash")
        selection.pop("git")
        selection["robustness"] = {
            "split": robustness["selection_split"],
            "run_id": robustness["selection_run_id"],
            "summary_sha256": robustness["selection_summary_sha256"],
            "per_utterance_sha256": robustness["selection_per_utterance_sha256"],
            "cut_ids_sha256": robustness["cut_ids_sha256"],
            "primary_metric_exclusions": robustness["primary_metric_exclusions"],
        }
        selection["roles"] = roles_by_identity[_system_identity(selection)]
        systems.append(selection)

    manifest = _load_json(manifest_summary)
    test_manifests = {}
    for split in ("test-clean", "test-other"):
        key = f"{split}_eval.jsonl.gz"
        metadata = manifest.get("manifests", {}).get(key)
        if not isinstance(metadata, dict) or "sha256" not in metadata:
            raise ValueError(f"Manifest summary lacks {key}")
        test_manifests[split] = metadata["sha256"]

    if not allow_dirty:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if current_commit != dev_commit:
            raise ValueError(
                "Current repository commit does not match the selected dev runs"
            )
        current_status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).strip()
        if current_status:
            raise ValueError("Formal test gate requires a clean current worktree")
    provenance_roles = {
        "base_m_seed2026",
        "flagship_m_seed2026",
        "flagship_m_seed2027",
        "flagship_m_seed2028",
        "ours_075_seed2026",
    }
    training_provenance: dict[str, Any] = {}
    if provenance_roles <= set(entry_by_role):
        if mplus_direction_selection is None:
            raise ValueError("Final test gate requires MPLUS_DIRECTION_SELECTED")
        if ours_075_release_marker is None:
            raise ValueError("Final test gate requires OURS_075_FORMAL_RELEASE")
        if ours_075_training_completion is None:
            raise ValueError("Final test gate requires OURS_075_TRAINING_COMPLETE")
        selection_path = Path(mplus_direction_selection)
        training_provenance["mplus"] = _validate_mplus_provenance(
            selection_path=selection_path,
            replication_completion_paths=[
                Path(value) for value in mplus_replication_completions
            ],
            entry_by_role=entry_by_role,
            dev_commit=str(dev_commit),
            repository_root=root,
        )
        training_provenance["ours_075"] = _validate_ours_075_provenance(
            release_path=Path(ours_075_release_marker),
            completion_path=Path(ours_075_training_completion),
            mplus_selection_path=selection_path,
            entry_by_role=entry_by_role,
            repository_root=root,
        )
    source_hashes = {}
    for relative in FROZEN_SOURCES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = sha256_file(path)
    systems.sort(
        key=lambda entry: (
            entry["adapter"],
            entry["system"],
            entry["seed"],
            entry["config_hash"],
        )
    )
    gate = {
        "schema_version": 3,
        "status": ("TEST_GATE_DIAGNOSTIC" if allow_dirty else "TEST_GATE_FROZEN"),
        "allow_dirty_dev_runs": allow_dirty,
        "git_commit": dev_commit,
        "protocol_sha256": sha256_file(protocol) if protocol is not None else None,
        "manifest_summary_sha256": sha256_file(manifest_summary),
        "test_manifests": test_manifests,
        "metrics": expected_metrics,
        "metric_metadata_hash": next(iter(metric_hashes)),
        "dev_protocol": {
            "selection_split": "dev-clean",
            "robustness_split": "dev-other",
        },
        "source_hashes": source_hashes,
        "final_system_registry": {
            "schema_version": 1,
            "status": "FINAL_SYSTEMS_PREREGISTERED",
            "sha256": sha256_file(registry_path),
            "requirements_sha256": sha256_file(requirements_path),
            "roles": sorted(
                role for roles in roles_by_identity.values() for role in roles
            ),
            "unique_system_identities": len(roles_by_identity),
            "paper_display": paper_display,
        },
        "training_provenance": training_provenance,
        "systems": systems,
    }
    _atomic_json(output, gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze matched dev-clean/dev-other systems and evaluator hashes "
            "before test"
        )
    )
    parser.add_argument("--system-registry", type=Path, required=True)
    parser.add_argument("--mplus-direction-selection", type=Path)
    parser.add_argument(
        "--mplus-replication-complete", type=Path, action="append", default=[]
    )
    parser.add_argument("--ours-075-release", type=Path)
    parser.add_argument("--ours-075-complete", type=Path)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    gate = freeze_test_gate(
        system_registry=args.system_registry,
        manifest_summary=args.manifest_summary,
        protocol=args.protocol,
        output=args.output,
        mplus_direction_selection=args.mplus_direction_selection,
        mplus_replication_completions=args.mplus_replication_complete,
        ours_075_release_marker=args.ours_075_release,
        ours_075_training_completion=args.ours_075_complete,
        metrics=(value.strip() for value in args.metrics.split(",") if value.strip()),
        repository_root=args.repository_root,
        replace=args.replace,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
