from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from dmel_codec.evaluation.cache import canonical_hash, sha256_file
from dmel_codec.tools.resource_accounting import execution_metadata
from dmel_codec.train_codec import get_config


TRAINING_ROUTES = ("codec", "vocoder", "joint")
SELECTABLE_ROUTES = ("source", *TRAINING_ROUTES)
SYSTEM_NAME_BY_ROUTE = {
    "source": "Ours-M",
    "codec": "Ours-M+ (full codec + frozen vocoder + waveform GAN)",
    "vocoder": "Ours-M+ (frozen codec + full vocoder + waveform GAN)",
    "joint": "Ours-M+ (full codec + full vocoder + waveform GAN)",
}
EXPERIMENT_BY_ROUTE = {
    "codec": "refinement_waveform_codec_gan_100k",
    "vocoder": "refinement_waveform_vocoder_full_gan_100k",
    "joint": "refinement_waveform_full_codec_full_vocoder_gan_100k",
}
GENERATOR_CUDNN_BY_ROUTE = {
    "codec": False,
    "vocoder": True,
    "joint": False,
}
FORMAL_SEEDS = (2026, 2027, 2028)
REPLICATION_SEEDS = (2027, 2028)
SELECTION_MODES = ("matched_listening", "objective_only_user_authorized")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git_state(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=repository_root, text=True
    ).strip()
    return commit, bool(status)


def _expected_replication_config(
    *,
    route: str,
    seed: int,
    run_dir: Path,
    source_checkpoint: Path,
    git_commit: str,
) -> dict[str, Any]:
    if route not in TRAINING_ROUTES:
        raise ValueError(f"Invalid replication route: {route}")
    run_name = f"mplus_selected_{git_commit[:12]}_{route}"
    config = get_config(
        [
            f"experiment={EXPERIMENT_BY_ROUTE[route]}",
            "codec=ours_m",
            f"seed={seed}",
            f"experiment.name={run_name}",
            f"experiment.run_dir={run_dir}",
            f"experiment.init_from={source_checkpoint}",
            f"init_from={source_checkpoint}",
        ]
    )
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("Expected M+ replication config to resolve to a mapping")
    resolved["resume_from"] = None
    return resolved


def _verified_artifact(record: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an artifact record")
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{label} requires path and sha256")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return path, observed


def _verified_source(run_dir: Path, expected_seed: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    required = (
        "checkpoint_selection.json",
        "resolved_config.yaml",
        "model_metadata.json",
        "data_manifest.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source run {run_dir}: {missing}")

    selection_path = run_dir / "checkpoint_selection.json"
    selection = _load_json(selection_path)
    if selection.get("monitor") != "val_loss" or selection.get("mode") != "min":
        raise ValueError(f"Unexpected source selector in {selection_path}")
    best = selection.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"Source selector has no best checkpoint: {selection_path}")
    filename = best.get("filename")
    expected_hash = best.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_hash, str):
        raise ValueError(f"Invalid best checkpoint record: {selection_path}")
    checkpoint = run_dir / "checkpoints" / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_hash = sha256_file(checkpoint)
    if observed_hash != expected_hash:
        raise ValueError(f"Source checkpoint SHA-256 mismatch: {checkpoint}")

    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(resolved, dict) or int(resolved.get("seed", -1)) != expected_seed:
        raise ValueError(f"Source run seed mismatch: {run_dir}")
    metadata = _load_json(run_dir / "model_metadata.json")
    definition = metadata.get("codec_definition")
    if not isinstance(definition, dict):
        raise ValueError(f"Source run has no codec definition: {run_dir}")
    expected_rate = 1.746377604335702
    if abs(float(definition.get("nominal_bitrate_kbps", -1.0)) - expected_rate) > 1e-9:
        raise ValueError(f"Source is not the frozen Ours-M operating point: {run_dir}")
    if definition.get("decoder_input_policy") != "zeros":
        raise ValueError(f"Source decoder is not deterministic-zero policy: {run_dir}")

    return {
        "run_dir": str(run_dir),
        "seed": expected_seed,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "global_step": int(best["global_step"]),
            "sha256": observed_hash,
        },
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "model_metadata_sha256": sha256_file(run_dir / "model_metadata.json"),
        "data_manifest_sha256": sha256_file(run_dir / "data_manifest.json"),
    }


def selected_seed2026_identity(
    gate: dict[str, Any], selected: str, *, expected_commit: str
) -> dict[str, Any]:
    routes = gate.get("routes")
    if not isinstance(routes, dict) or set(routes) != set(TRAINING_ROUTES):
        raise ValueError("M+ direction gate does not contain every training route")
    identities: list[dict[str, Any]] = []
    route_names = TRAINING_ROUTES if selected == "source" else (selected,)
    identity_key = "source_identity" if selected == "source" else "candidate_identity"
    for route in route_names:
        runs = routes.get(route, {}).get("runs")
        if not isinstance(runs, dict) or set(runs) != {"dev-clean", "dev-other"}:
            raise ValueError(f"M+ direction gate lacks both dev runs for {route}")
        for split in ("dev-clean", "dev-other"):
            identity = runs[split].get(identity_key)
            if not isinstance(identity, dict):
                raise ValueError(
                    f"M+ direction gate lacks {identity_key} for {route}/{split}"
                )
            identities.append(identity)
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError("M+ selected seed-2026 identity differs across dev evidence")
    expected_system = SYSTEM_NAME_BY_ROUTE[selected]
    if (
        first.get("system") != expected_system
        or first.get("variant") != "ours"
        or int(first.get("seed", -1)) != 2026
        or first.get("git_commit") != expected_commit
    ):
        raise ValueError("M+ selected seed-2026 evaluation identity is inconsistent")
    raw_hash = first.get("raw_checkpoint_sha256")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        raise ValueError("M+ selected seed-2026 identity lacks a checkpoint SHA-256")
    return dict(first)


def _validate_listening_receipt(
    *, receipt_record: Any, gate_hash: str, selected: str
) -> dict[str, Any]:
    receipt_path, _ = _verified_artifact(
        receipt_record, label="matched-listening receipt"
    )
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "MPLUS_MATCHED_LISTENING_COMPLETE"
    ):
        raise ValueError("Invalid matched-listening receipt")
    if (
        receipt.get("direction_gate_sha256") != gate_hash
        or receipt.get("selected_direction") != selected
    ):
        raise ValueError("Matched-listening receipt disagrees with the selection")
    evaluated = receipt.get("evaluated_directions")
    if not isinstance(evaluated, list) or evaluated != list(TRAINING_ROUTES):
        raise ValueError("Matched listening must include all three training directions")
    _verified_artifact(receipt.get("listening_manifest"), label="listening manifest")
    return receipt


def _validate_objective_selection(
    *, evidence: Any, gate: dict[str, Any], gate_hash: str, selected: str
) -> None:
    if not isinstance(evidence, dict):
        raise ValueError("Objective-only selection requires structured evidence")
    note = evidence.get("user_authorization_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Objective-only selection requires a user authorization note")
    leaders = gate.get("statistical_leaders")
    if not isinstance(leaders, list) or len(leaders) != 1:
        raise ValueError("Objective-only selection requires one statistical leader")
    if selected != leaders[0] or selected not in TRAINING_ROUTES:
        raise ValueError(
            "Objective-only selection must use the unique statistical leader"
        )
    if evidence.get("direction_gate_sha256") != gate_hash:
        raise ValueError("Objective-only evidence does not bind the direction gate")
    if evidence.get("statistical_leaders") != leaders:
        raise ValueError(
            "Objective-only evidence changed the frozen statistical leader"
        )
    if evidence.get("matched_listening_waived") is not True:
        raise ValueError(
            "Objective-only evidence must explicitly waive matched listening"
        )


def validate_direction_selection(path: Path, *, expected_commit: str) -> dict[str, Any]:
    selection = _load_json(path)
    schema_version = selection.get("schema_version")
    if schema_version not in (1, 2) or selection.get("status") != (
        "MPLUS_DIRECTION_SELECTED"
    ):
        raise ValueError("Invalid M+ direction-selection marker")
    if selection.get("git_commit") != expected_commit:
        raise ValueError("M+ direction selection was created from another commit")
    selected = selection.get("selected_direction")
    if selected not in SELECTABLE_ROUTES:
        raise ValueError(f"Invalid selected M+ direction: {selected}")
    expected_disposition = (
        "retain_source" if selected == "source" else "replicate_selected_direction"
    )
    if selection.get("disposition") != expected_disposition:
        raise ValueError("M+ selection disposition disagrees with its direction")

    gate_path, gate_hash = _verified_artifact(
        selection.get("direction_gate"), label="direction gate"
    )
    gate = _load_json(gate_path)
    if gate.get("schema_version") != 1 or gate.get("status") != (
        "MPLUS_DIRECTION_GATE"
    ):
        raise ValueError("Invalid complete-development direction gate")
    if gate.get("automatic_promotion") is not False:
        raise ValueError("Direction gate must prohibit automatic promotion")
    eligible = gate.get("eligible_before_listening")
    if not isinstance(eligible, list):
        raise ValueError("Direction gate has no eligibility list")
    if selected != "source" and selected not in eligible:
        raise ValueError("Selected M+ direction failed the complete-development gate")

    if schema_version == 1:
        _validate_listening_receipt(
            receipt_record=selection.get("listening_receipt"),
            gate_hash=gate_hash,
            selected=selected,
        )
        return selection

    mode = selection.get("selection_mode")
    if mode not in SELECTION_MODES:
        raise ValueError(f"Invalid M+ selection mode: {mode}")
    if mode == "matched_listening":
        _validate_listening_receipt(
            receipt_record=selection.get("selection_evidence"),
            gate_hash=gate_hash,
            selected=selected,
        )
    else:
        _validate_objective_selection(
            evidence=selection.get("selection_evidence"),
            gate=gate,
            gate_hash=gate_hash,
            selected=selected,
        )
    gate_commit = selection.get("direction_gate_evaluation_commit")
    if not isinstance(gate_commit, str) or len(gate_commit) != 40:
        raise ValueError("Direction selection lacks its dev-evidence commit")
    selected_identity = selected_seed2026_identity(
        gate, selected, expected_commit=gate_commit
    )
    if selection.get("selected_seed2026_evaluation_identity") != selected_identity:
        raise ValueError(
            "Direction selection changed its seed-2026 evaluation identity"
        )
    return selection


def freeze_direction_selection(
    *,
    direction_gate: Path,
    listening_receipt: Path | None,
    objective_only_user_authorized: bool = False,
    authorization_note: str | None = None,
    source_run_directories: dict[int, Path],
    output: Path,
    repository_root: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to replace direction selection: {output}")
    if set(source_run_directories) != set(FORMAL_SEEDS):
        raise ValueError(f"Source runs must cover exactly {FORMAL_SEEDS}")

    direction_gate = direction_gate.expanduser().resolve()
    gate = _load_json(direction_gate)
    if gate.get("schema_version") != 1 or gate.get("status") != "MPLUS_DIRECTION_GATE":
        raise ValueError("Invalid M+ direction gate")
    if gate.get("automatic_promotion") is not False:
        raise ValueError("M+ direction gate must prohibit automatic promotion")
    gate_hash = sha256_file(direction_gate)
    if objective_only_user_authorized == (listening_receipt is not None):
        raise ValueError(
            "Choose exactly one selection mode: matched listening or objective-only"
        )
    if objective_only_user_authorized:
        leaders = gate.get("statistical_leaders")
        if not isinstance(leaders, list) or len(leaders) != 1:
            raise ValueError("Objective-only selection requires one statistical leader")
        selected = leaders[0]
        selection_mode = "objective_only_user_authorized"
        selection_evidence: dict[str, Any] = {
            "direction_gate_sha256": gate_hash,
            "matched_listening_waived": True,
            "statistical_leaders": leaders,
            "user_authorization_note": authorization_note,
        }
        _validate_objective_selection(
            evidence=selection_evidence,
            gate=gate,
            gate_hash=gate_hash,
            selected=selected,
        )
    else:
        if authorization_note is not None:
            raise ValueError(
                "Authorization note is only valid for objective-only selection"
            )
        assert listening_receipt is not None
        listening_receipt = listening_receipt.expanduser().resolve()
        receipt = _load_json(listening_receipt)
        selected = receipt.get("selected_direction")
        if selected not in SELECTABLE_ROUTES:
            raise ValueError(f"Invalid selected direction: {selected}")
        _validate_listening_receipt(
            receipt_record={
                "path": str(listening_receipt),
                "sha256": sha256_file(listening_receipt),
            },
            gate_hash=gate_hash,
            selected=selected,
        )
        selection_mode = "matched_listening"
        selection_evidence = {
            "path": str(listening_receipt),
            "sha256": sha256_file(listening_receipt),
        }

    eligible = gate.get("eligible_before_listening")
    if not isinstance(eligible, list):
        raise ValueError("Direction gate has no eligibility list")
    if selected != "source" and selected not in eligible:
        raise ValueError(f"Selected direction failed the complete-dev gate: {selected}")
    if selected == "source":
        disposition = "retain_source"
    else:
        if gate.get("disposition") != "listening_required":
            raise ValueError("A promoted route requires listening_required disposition")
        disposition = "replicate_selected_direction"

    repository_root = repository_root.expanduser().resolve()
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise ValueError("Formal direction selection requires a clean worktree")
    identity_commit = (
        gate.get("routes", {})
        .get(selected if selected != "source" else TRAINING_ROUTES[0], {})
        .get("runs", {})
        .get("dev-clean", {})
        .get("candidate_identity" if selected != "source" else "source_identity", {})
        .get("git_commit")
    )
    if not isinstance(identity_commit, str) or len(identity_commit) != 40:
        raise ValueError("Direction gate lacks a valid evaluation commit")
    if selection_mode == "matched_listening" and identity_commit != commit:
        raise ValueError(
            "Matched-listening selection requires same-commit dev evidence"
        )
    selected_identity = selected_seed2026_identity(
        gate, selected, expected_commit=identity_commit
    )
    sources = {
        str(seed): _verified_source(source_run_directories[seed], seed)
        for seed in FORMAL_SEEDS
    }
    if len({value["data_manifest_sha256"] for value in sources.values()}) != 1:
        raise ValueError("Source seeds do not share one data manifest")
    if len({value["model_metadata_sha256"] for value in sources.values()}) != 1:
        raise ValueError("Source seeds do not share one codec definition")
    if selected == "source" and (
        selected_identity["raw_checkpoint_sha256"]
        != sources["2026"]["checkpoint"]["sha256"]
    ):
        raise ValueError(
            "Direction-gate source checkpoint differs from frozen seed-2026 source"
        )
    marker = {
        "schema_version": 2,
        "status": "MPLUS_DIRECTION_SELECTED",
        "selected_direction": selected,
        "selection_mode": selection_mode,
        "disposition": disposition,
        "git_commit": commit,
        "direction_gate_evaluation_commit": identity_commit,
        "direction_gate": {
            "path": str(direction_gate),
            "sha256": gate_hash,
        },
        "selection_evidence": selection_evidence,
        "selected_seed2026_evaluation_identity": selected_identity,
        "source_checkpoints": sources,
    }
    _atomic_json(output, marker)
    validate_direction_selection(output, expected_commit=commit)
    return marker


def verify_replication_preflight(
    *,
    selection: Path,
    seed: int,
    run_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    selection = selection.expanduser().resolve()
    repository_root = repository_root.expanduser().resolve()
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise ValueError("Formal replication requires a clean worktree")
    marker = validate_direction_selection(selection, expected_commit=commit)
    route = marker.get("selected_direction")
    if route == "source":
        raise ValueError(
            "The frozen decision retains source; no replication is allowed"
        )
    if route not in TRAINING_ROUTES:
        raise ValueError(f"Invalid selected training direction: {route}")
    if seed not in REPLICATION_SEEDS:
        raise ValueError(f"Replication seed must be one of {REPLICATION_SEEDS}")

    gate_path, gate_hash = _verified_artifact(
        marker.get("direction_gate"), label="direction gate"
    )
    gate = _load_json(gate_path)
    if route not in gate.get("eligible_before_listening", []):
        raise ValueError("Selected route is not eligible in the bound direction gate")
    if marker.get("direction_gate", {}).get("sha256") != gate_hash:
        raise ValueError("Direction selection disagrees with its gate")

    source = marker.get("source_checkpoints", {}).get(str(seed))
    if not isinstance(source, dict):
        raise ValueError(f"Direction selection has no source for seed {seed}")
    checkpoint, checkpoint_hash = _verified_artifact(
        source.get("checkpoint"), label=f"seed-{seed} source checkpoint"
    )
    selection_path, _ = _verified_artifact(
        source.get("selection"), label=f"seed-{seed} source selection"
    )
    source_selection = _load_json(selection_path)
    if source_selection.get("best", {}).get("sha256") != checkpoint_hash:
        raise ValueError("Source selector no longer points to the frozen checkpoint")
    run_dir = run_dir.expanduser().resolve()
    resolved_config = _expected_replication_config(
        route=route,
        seed=seed,
        run_dir=run_dir,
        source_checkpoint=checkpoint,
        git_commit=commit,
    )

    return {
        "schema_version": 1,
        "status": "MPLUS_REPLICATION_PREFLIGHT",
        "seed": seed,
        "direction": route,
        "experiment": EXPERIMENT_BY_ROUTE[route],
        "run_dir": str(run_dir),
        "git_commit": commit,
        "direction_selection": {
            "path": str(selection),
            "sha256": sha256_file(selection),
        },
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "generator_backward_cudnn_enabled": GENERATOR_CUDNN_BY_ROUTE[route],
        "resolved_config_sha256": canonical_hash(resolved_config),
    }


def validate_existing_preflight(
    *, selection: Path, preflight: Path, repository_root: Path
) -> dict[str, Any]:
    selection = selection.expanduser().resolve()
    preflight = preflight.expanduser().resolve()
    document = _load_json(preflight)
    if document.get("schema_version") != 1 or document.get("status") != (
        "MPLUS_REPLICATION_PREFLIGHT"
    ):
        raise ValueError("Invalid existing replication preflight")
    seed = int(document.get("seed", -1))
    repository_root = repository_root.expanduser().resolve()
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise ValueError("Existing preflight requires a clean worktree")
    marker = validate_direction_selection(selection, expected_commit=commit)
    route = marker.get("selected_direction")
    if route not in TRAINING_ROUTES or document.get("direction") != route:
        raise ValueError("Existing preflight direction differs from selection marker")
    if document.get("experiment") != EXPERIMENT_BY_ROUTE[route]:
        raise ValueError("Existing preflight experiment differs from selected route")
    if (
        document.get("generator_backward_cudnn_enabled")
        is not (GENERATOR_CUDNN_BY_ROUTE[route])
    ):
        raise ValueError("Existing preflight runtime differs from selected route")
    selection_record = document.get("direction_selection")
    if not isinstance(selection_record, dict) or (
        selection_record.get("path") != str(selection)
        or selection_record.get("sha256") != sha256_file(selection)
    ):
        raise ValueError(
            "Existing preflight does not bind the current selection marker"
        )
    source = marker.get("source_checkpoints", {}).get(str(seed), {}).get("checkpoint")
    source_record = document.get("source_checkpoint")
    if not isinstance(source, dict) or not isinstance(source_record, dict):
        raise ValueError("Existing preflight has no frozen source checkpoint")
    if {
        "path": source.get("path"),
        "sha256": source.get("sha256"),
    } != source_record:
        raise ValueError("Existing preflight source differs from selection marker")
    _verified_artifact(source_record, label=f"seed-{seed} source checkpoint")
    run_dir_value = document.get("run_dir")
    if not isinstance(run_dir_value, str) or not Path(run_dir_value).is_absolute():
        raise ValueError("Existing preflight has no absolute replication run directory")

    if commit != marker.get("git_commit") or commit != document.get("git_commit"):
        raise ValueError("Existing preflight requires the frozen clean git commit")
    expected_config = _expected_replication_config(
        route=route,
        seed=seed,
        run_dir=Path(run_dir_value),
        source_checkpoint=Path(str(source_record["path"])),
        git_commit=commit,
    )
    if document.get("resolved_config_sha256") != canonical_hash(expected_config):
        raise ValueError("Existing preflight resolved config differs from frozen route")
    gate_path, gate_hash = _verified_artifact(
        marker.get("direction_gate"), label="direction gate"
    )
    gate = _load_json(gate_path)
    if route not in gate.get("eligible_before_listening", []):
        raise ValueError("Selected route is no longer eligible in the bound gate")
    if marker.get("direction_gate", {}).get("sha256") != gate_hash:
        raise ValueError("Direction selection disagrees with the existing preflight")
    return document


def _load_verified_checkpoint(
    checkpoint: Path,
    *,
    preflight_document: dict[str, Any],
    expected_step: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    global_step = int(payload["global_step"])
    if global_step != expected_step:
        raise ValueError(f"Checkpoint step {global_step} != {expected_step}")
    metadata = payload["codec_metadata"]
    expected_config_hash = preflight_document.get("resolved_config_sha256")
    if (
        not isinstance(expected_config_hash, str)
        or metadata.get("config_sha256") != expected_config_hash
        or canonical_hash(metadata.get("resolved_config")) != expected_config_hash
    ):
        raise ValueError("Checkpoint config differs from frozen preflight")
    initialization = metadata.get("model", {}).get("initialization", {})
    if initialization.get("sha256") != preflight_document.get(
        "source_checkpoint", {}
    ).get("sha256"):
        raise ValueError("Checkpoint initialization differs from frozen source")
    git = metadata.get("git", {})
    if (
        git.get("commit") != preflight_document.get("git_commit")
        or git.get("dirty") is not False
    ):
        raise ValueError("Checkpoint git identity differs from frozen preflight")
    observed_cudnn = metadata.get("training_runtime", {}).get(
        "generator_backward_cudnn_enabled"
    )
    if observed_cudnn is not preflight_document.get("generator_backward_cudnn_enabled"):
        raise ValueError("Generator backward runtime differs from frozen route")
    return payload


def verify_segment_checkpoint(
    *,
    checkpoint: Path,
    preflight: Path,
    target_step: int,
    start_time: str | None = None,
    end_time: str | None = None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    preflight_document = _load_json(preflight)
    if preflight_document.get("status") != "MPLUS_REPLICATION_PREFLIGHT":
        raise ValueError("Invalid replication preflight")
    if target_step < 2_000 or target_step > 100_000 or target_step % 2_000:
        raise ValueError("Target step must be a multiple of 2000 in [2000, 100000]")
    checkpoint = checkpoint.expanduser().resolve()
    expected_checkpoint = (
        Path(str(preflight_document.get("run_dir"))).expanduser().resolve()
        / "checkpoints"
        / "last.ckpt"
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("Segment checkpoint is outside the frozen replication run")
    payload = _load_verified_checkpoint(
        checkpoint,
        preflight_document=preflight_document,
        expected_step=target_step,
    )
    global_step = int(payload["global_step"])
    if len(payload["optimizer_states"]) != 2 or len(payload["lr_schedulers"]) != 2:
        raise ValueError("M+ checkpoint must contain two optimizers and schedulers")
    state = payload.get("LhotseDataModule")
    if not isinstance(state, dict):
        raise ValueError("M+ checkpoint has no LhotseDataModule state")
    sampler = state["train_sampler"]
    diagnostics = sampler["diagnostics"]
    epoch = diagnostics["current_epoch"]
    kept = int(diagnostics["stats_per_epoch"][epoch]["kept_batches"])
    if kept != target_step:
        raise ValueError(f"Sampler kept {kept} batches at step {target_step}")
    runtime = payload["codec_metadata"]["training_runtime"]
    observed_cudnn = runtime["generator_backward_cudnn_enabled"]
    supplied_resource_fields = (
        start_time,
        end_time,
        cuda_visible_devices,
    )
    if any(value is not None for value in supplied_resource_fields) and not all(
        value is not None for value in supplied_resource_fields
    ):
        raise ValueError("Segment resource metadata requires start, end, and GPUs")
    resource = {}
    if all(value is not None for value in supplied_resource_fields):
        resource = execution_metadata(
            start_time=str(start_time),
            end_time=str(end_time),
            cuda_visible_devices=str(cuda_visible_devices),
        )
    return {
        "schema_version": 1,
        "status": "MPLUS_REPLICATION_SEGMENT_VERIFIED",
        "direction": preflight_document["direction"],
        "seed": int(preflight_document["seed"]),
        "target_step": target_step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        **resource,
        "verified": {
            "global_step": global_step,
            "optimizer_states": 2,
            "scheduler_states": 2,
            "sampler_kept_batches": kept,
            "discarded_prefetched_batches": int(state["discarded_prefetched_batches"]),
            "generator_backward_cudnn_enabled": observed_cudnn,
        },
    }


def verify_final_selection(*, run_dir: Path, preflight: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    preflight_document = _load_json(preflight)
    if preflight_document.get("status") != "MPLUS_REPLICATION_PREFLIGHT":
        raise ValueError("Invalid replication preflight")
    if str(run_dir) != preflight_document.get("run_dir"):
        raise ValueError("Final replication run directory differs from preflight")
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    expected_config_hash = preflight_document.get("resolved_config_sha256")
    if (
        not isinstance(resolved, dict)
        or not isinstance(expected_config_hash, str)
        or canonical_hash(resolved) != expected_config_hash
    ):
        raise ValueError("Final replication resolved config differs from preflight")
    selection_path = run_dir / "checkpoint_selection.json"
    selection = _load_json(selection_path)
    candidates = selection.get("candidates")
    if (
        selection.get("monitor") != "val_loss"
        or selection.get("mode") != "min"
        or not isinstance(candidates, list)
        or len(candidates) != 2
    ):
        raise ValueError("Final replication selector is not frozen top-2 dev loss")
    last = selection.get("last_checkpoint")
    best = selection.get("best")
    if not isinstance(last, dict) or int(last.get("global_step", -1)) != 100_000:
        raise ValueError("Final replication last checkpoint is not step 100000")
    if not isinstance(best, dict) or best not in candidates:
        raise ValueError("Final replication best checkpoint is not a candidate")

    def verified_checkpoint(record: dict[str, Any], label: str) -> tuple[Path, str]:
        filename = record.get("filename")
        expected = record.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid {label} checkpoint filename")
        if not isinstance(expected, str):
            raise ValueError(f"Invalid {label} checkpoint SHA-256")
        path = run_dir / "checkpoints" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} checkpoint SHA-256 mismatch")
        return path, observed

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Invalid checkpoint candidate {index}")
        candidate_path, _ = verified_checkpoint(candidate, f"candidate {index}")
        _load_verified_checkpoint(
            candidate_path,
            preflight_document=preflight_document,
            expected_step=int(candidate.get("global_step", -1)),
        )
    last_path, last_hash = verified_checkpoint(last, "last")
    best_path, best_hash = verified_checkpoint(best, "best")
    final_segment = verify_segment_checkpoint(
        checkpoint=last_path, preflight=preflight, target_step=100_000
    )
    return {
        "schema_version": 1,
        "status": "MPLUS_REPLICATION_COMPLETE",
        "direction": preflight_document["direction"],
        "seed": int(preflight_document["seed"]),
        "final_global_step": 100_000,
        "last_checkpoint": {
            "path": str(last_path),
            "sha256": last_hash,
        },
        "best_checkpoint": {
            "path": str(best_path),
            "global_step": int(best["global_step"]),
            "score": float(best["score"]),
            "sha256": best_hash,
        },
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "replication_preflight_sha256": sha256_file(preflight),
        "resolved_config_sha256": sha256_file(resolved_path),
        "verified": final_segment["verified"],
    }


def _parse_source_runs(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError("--source-run values must use SEED=PATH")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate source seed: {seed}")
        result[seed] = Path(path_text)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and verify the M+ selected-direction replication contract"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--direction-gate", type=Path, required=True)
    freeze.add_argument("--listening-receipt", type=Path)
    freeze.add_argument("--objective-only-user-authorized", action="store_true")
    freeze.add_argument("--authorization-note")
    freeze.add_argument("--source-run", action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--repository-root", type=Path, default=Path.cwd())

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--selection", type=Path, required=True)
    preflight.add_argument("--seed", type=int, required=True)
    preflight.add_argument("--run-dir", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--repository-root", type=Path, default=Path.cwd())

    validate = subparsers.add_parser("validate-preflight")
    validate.add_argument("--selection", type=Path, required=True)
    validate.add_argument("--preflight", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path, default=Path.cwd())

    segment = subparsers.add_parser("verify-segment")
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--preflight", type=Path, required=True)
    segment.add_argument("--target-step", type=int, required=True)
    segment.add_argument("--start-time", required=True)
    segment.add_argument("--end-time", required=True)
    segment.add_argument("--cuda-visible-devices", required=True)
    segment.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("verify-final")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--preflight", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_direction_selection(
            direction_gate=args.direction_gate,
            listening_receipt=args.listening_receipt,
            objective_only_user_authorized=args.objective_only_user_authorized,
            authorization_note=args.authorization_note,
            source_run_directories=_parse_source_runs(args.source_run),
            output=args.output,
            repository_root=args.repository_root,
        )
    elif args.command == "preflight":
        result = verify_replication_preflight(
            selection=args.selection,
            seed=args.seed,
            run_dir=args.run_dir,
            repository_root=args.repository_root,
        )
        _atomic_json(args.output, result)
    elif args.command == "validate-preflight":
        result = validate_existing_preflight(
            selection=args.selection,
            preflight=args.preflight,
            repository_root=args.repository_root,
        )
    elif args.command == "verify-segment":
        result = verify_segment_checkpoint(
            checkpoint=args.checkpoint,
            preflight=args.preflight,
            target_step=args.target_step,
            start_time=args.start_time,
            end_time=args.end_time,
            cuda_visible_devices=args.cuda_visible_devices,
        )
        _atomic_json(args.output, result)
    else:
        result = verify_final_selection(
            run_dir=args.run_dir,
            preflight=args.preflight,
        )
        if args.output.is_file():
            if _load_json(args.output) != result:
                raise ValueError(
                    f"Existing completion artifact disagrees: {args.output}"
                )
        else:
            _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
