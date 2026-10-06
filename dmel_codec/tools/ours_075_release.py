from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from dmel_codec.evaluation.bitrate import nominal_bitrate_kbps
from dmel_codec.evaluation.cache import canonical_hash, sha256_file
from dmel_codec.tools import mplus_replication
from dmel_codec.train_codec import get_config


TRAINING_ROUTES = ("codec", "vocoder", "joint")
SELECTABLE_ROUTES = ("source", *TRAINING_ROUTES)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace Ours-0.75 release: {path}")
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


def _verified_artifact(record: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an artifact record")
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{label} requires path and sha256")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, observed_hash


def _contract_artifact(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{label} is not configured")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _formal_resolved_config() -> dict[str, Any]:
    config = get_config(
        ["experiment=pilot_clean100_repair2_100k", "codec=ours_075", "seed=2026"]
    )
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("Formal Ours-0.75 config must resolve to a mapping")
    resolved["resume_from"] = None
    return resolved


def _formal_contract() -> dict[str, Any]:
    config = get_config(
        ["experiment=pilot_clean100_repair2_100k", "codec=ours_075", "seed=2026"]
    )
    resolved_config = _formal_resolved_config()
    levels = [int(value) for value in config.codec.model.quantizer.levels]
    streams = int(config.codec.num_groups)
    cardinality = math.prod(levels)
    bitrate = nominal_bitrate_kbps(
        float(config.codec.token_frame_rate), [cardinality] * streams
    )
    contract = {
        "system": str(config.codec.system_name),
        "codec": str(config.codec.name),
        "experiment": str(config.experiment.name),
        "seed": int(config.seed),
        "run_dir": str(Path(config.experiment.run_dir).expanduser().resolve()),
        "train_manifest": _contract_artifact(
            config.data.module.train_cuts_path, label="train-clean-100 manifest"
        ),
        "validation_manifest": _contract_artifact(
            config.data.module.val_cuts_path, label="dev-clean manifest"
        ),
        "bigvgan_checkpoint": _contract_artifact(
            config.codec.model.vocoder.ckpt_path, label="BigVGAN checkpoint"
        ),
        "bigvgan_config": _contract_artifact(
            config.codec.model.vocoder.h_path, label="BigVGAN config"
        ),
        "sample_rate": int(config.codec.sample_rate),
        "token_frame_rate": float(config.codec.token_frame_rate),
        "num_parallel_streams": streams,
        "fsq_levels": levels,
        "codebook_cardinality": cardinality,
        "nominal_bitrate_kbps": bitrate,
        "max_steps": int(config.experiment.trainer.max_steps),
        "train_crop_seconds": float(config.data.train_crop_seconds),
        "weight_adv": float(config.codec.model.weight_adv),
        "optimizer_lr": float(config.codec.model.optimizer.lr),
        "decoder_input_policy": str(config.codec.model.decoder_input_policy),
        "resolved_config_sha256": canonical_hash(resolved_config),
    }
    expected = {
        "system": "Ours-0.75",
        "codec": "ours_075",
        "experiment": "pilot_clean100_repair2_100k",
        "seed": 2026,
        "sample_rate": 24000,
        "token_frame_rate": 23.4375,
        "num_parallel_streams": 10,
        "fsq_levels": [3, 3],
        "codebook_cardinality": 9,
        "max_steps": 100000,
        "train_crop_seconds": 3.0,
        "weight_adv": 0.2,
        "optimizer_lr": 0.0001,
        "decoder_input_policy": "zeros",
    }
    for key, value in expected.items():
        if contract[key] != value:
            raise ValueError(f"Unexpected formal Ours-0.75 {key}: {contract[key]}")
    if abs(bitrate - 0.7429511719) > 1e-9:
        raise ValueError(f"Unexpected formal Ours-0.75 bitrate: {bitrate}")
    if (
        Path(contract["train_manifest"]["path"]).name
        != "train-clean-100_codec.jsonl.gz"
    ):
        raise ValueError("Formal Ours-0.75 must train on train-clean-100")
    if Path(contract["validation_manifest"]["path"]).name != "dev-clean_eval.jsonl.gz":
        raise ValueError("Formal Ours-0.75 must select checkpoints on dev-clean")
    if config.get("init_from") is not None or config.get("resume_from") is not None:
        raise ValueError("Formal Ours-0.75 release must describe a fresh run")
    return contract


def validate_direction_selection(path: Path, *, expected_commit: str) -> dict[str, Any]:
    return mplus_replication.validate_direction_selection(
        path, expected_commit=expected_commit
    )


def create_release(
    *, selection: Path, repository_root: Path, output: Path
) -> dict[str, Any]:
    repository_root = repository_root.expanduser().resolve()
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise ValueError("Formal Ours-0.75 release requires a clean worktree")
    selection = selection.expanduser().resolve()
    selected = validate_direction_selection(selection, expected_commit=commit)
    contract = _formal_contract()
    release = {
        "schema_version": 1,
        "status": "OURS_075_FORMAL_RELEASE",
        "git_commit": commit,
        "mplus_direction_selection": {
            "path": str(selection),
            "sha256": sha256_file(selection),
            "selected_direction": selected["selected_direction"],
            "selection_mode": selected.get("selection_mode", "matched_listening"),
        },
        "formal_contract": contract,
        "formal_contract_sha256": canonical_hash(contract),
        "test_access": False,
    }
    _atomic_json(output.expanduser().resolve(), release)
    return release


def verify_release(*, release: Path, repository_root: Path) -> dict[str, Any]:
    release = release.expanduser().resolve()
    document = _load_json(release)
    if document.get("schema_version") != 1 or document.get("status") != (
        "OURS_075_FORMAL_RELEASE"
    ):
        raise ValueError("Invalid Ours-0.75 formal release")
    if document.get("test_access") is not False:
        raise ValueError("Ours-0.75 release must not authorize test access")
    repository_root = repository_root.expanduser().resolve()
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise ValueError("Formal Ours-0.75 training requires a clean worktree")
    if document.get("git_commit") != commit:
        raise ValueError("Ours-0.75 release was created from another commit")
    selection_path, selection_hash = _verified_artifact(
        document.get("mplus_direction_selection"), label="M+ direction selection"
    )
    selected = validate_direction_selection(selection_path, expected_commit=commit)
    selection_record = document["mplus_direction_selection"]
    if selection_record.get("sha256") != selection_hash or selection_record.get(
        "selected_direction"
    ) != selected.get("selected_direction"):
        raise ValueError("Ours-0.75 release selection binding is inconsistent")
    contract = _formal_contract()
    if document.get("formal_contract") != contract or document.get(
        "formal_contract_sha256"
    ) != canonical_hash(contract):
        raise ValueError("Ours-0.75 formal contract changed after release")
    return document


def _verified_checkpoint(
    *, run_dir: Path, record: dict[str, Any], label: str
) -> tuple[Path, str]:
    filename = record.get("filename")
    expected_hash = record.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid Ours-0.75 {label} checkpoint filename")
    if not isinstance(expected_hash, str):
        raise ValueError(f"Invalid Ours-0.75 {label} checkpoint SHA-256")
    checkpoint = run_dir / "checkpoints" / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_hash = sha256_file(checkpoint)
    if observed_hash != expected_hash:
        raise ValueError(f"Ours-0.75 {label} checkpoint SHA-256 mismatch")
    return checkpoint, observed_hash


def verify_final_selection(
    *, run_dir: Path, release: Path, repository_root: Path
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    release = release.expanduser().resolve()
    release_document = verify_release(release=release, repository_root=repository_root)
    contract = release_document["formal_contract"]
    if str(run_dir) != contract["run_dir"]:
        raise ValueError("Formal Ours-0.75 run directory disagrees with its release")
    selection_path = run_dir / "checkpoint_selection.json"
    resolved_path = run_dir / "resolved_config.yaml"
    if not selection_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(f"Incomplete formal Ours-0.75 run directory: {run_dir}")
    selection = _load_json(selection_path)
    candidates = selection.get("candidates")
    best = selection.get("best")
    last = selection.get("last_checkpoint")
    if (
        selection.get("monitor") != "val_loss"
        or selection.get("mode") != "min"
        or not isinstance(candidates, list)
        or len(candidates) != 2
        or not isinstance(best, dict)
        or best not in candidates
        or not isinstance(last, dict)
        or int(last.get("global_step", -1)) != 100_000
    ):
        raise ValueError(
            "Formal Ours-0.75 selection must be top-2 dev loss with a 100k last"
        )
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Invalid Ours-0.75 checkpoint candidate {index}")
        _verified_checkpoint(
            run_dir=run_dir, record=candidate, label=f"candidate {index}"
        )
    best_path, best_hash = _verified_checkpoint(
        run_dir=run_dir, record=best, label="best"
    )
    last_path, last_hash = _verified_checkpoint(
        run_dir=run_dir, record=last, label="last"
    )

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if (
        not isinstance(resolved, dict)
        or canonical_hash(resolved) != contract["resolved_config_sha256"]
        or int(resolved.get("seed", -1)) != 2026
        or resolved.get("codec", {}).get("name") != contract["codec"]
        or resolved.get("codec", {}).get("system_name") != contract["system"]
        or int(resolved.get("experiment", {}).get("trainer", {}).get("max_steps", -1))
        != 100_000
    ):
        raise ValueError(
            "Formal Ours-0.75 complete resolved config disagrees with its release"
        )

    payload = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    if int(payload.get("global_step", -1)) != 100_000:
        raise ValueError("Formal Ours-0.75 last checkpoint is not global step 100000")
    if (
        len(payload.get("optimizer_states", [])) != 2
        or len(payload.get("lr_schedulers", [])) != 2
    ):
        raise ValueError(
            "Formal Ours-0.75 checkpoint lacks two optimizer/scheduler states"
        )
    datamodule = payload.get("LhotseDataModule")
    if not isinstance(datamodule, dict):
        raise ValueError("Formal Ours-0.75 checkpoint lacks datamodule state")
    diagnostics = datamodule["train_sampler"]["diagnostics"]
    epoch = diagnostics["current_epoch"]
    kept = int(diagnostics["stats_per_epoch"][epoch]["kept_batches"])
    if kept != 100_000:
        raise ValueError(f"Formal Ours-0.75 sampler stopped at {kept}, not 100000")
    definition = payload.get("codec_metadata", {}).get("codec_definition", {})
    if definition.get("decoder_input_policy") != "zeros" or not math.isclose(
        float(definition.get("nominal_bitrate_kbps", -1.0)),
        float(contract["nominal_bitrate_kbps"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Formal Ours-0.75 checkpoint violates the released codec definition"
        )
    del payload
    return {
        "schema_version": 1,
        "status": "OURS_075_TRAINING_COMPLETE",
        "seed": 2026,
        "final_global_step": 100_000,
        "release": {
            "path": str(release),
            "sha256": sha256_file(release),
        },
        "run_dir": str(run_dir),
        "last_checkpoint": {"path": str(last_path), "sha256": last_hash},
        "best_checkpoint": {
            "path": str(best_path),
            "global_step": int(best["global_step"]),
            "score": float(best["score"]),
            "sha256": best_hash,
        },
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "verified": {
            "optimizer_states": 2,
            "scheduler_states": 2,
            "sampler_kept_batches": kept,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Release Ours-0.75 only after the complete M+ direction decision"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--selection", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify = subparsers.add_parser("verify")
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--repository-root", type=Path, default=Path.cwd())
    final = subparsers.add_parser("verify-final")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--release", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "create":
        result = create_release(
            selection=args.selection,
            repository_root=args.repository_root,
            output=args.output,
        )
    elif args.command == "verify":
        result = verify_release(
            release=args.release, repository_root=args.repository_root
        )
    else:
        result = verify_final_selection(
            run_dir=args.run_dir,
            release=args.release,
            repository_root=args.repository_root,
        )
        _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
