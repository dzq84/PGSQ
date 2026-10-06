from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dmel_codec.evaluation.adapters.ours import OursAdapter
from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.evaluation.run_codec_eval import DEFAULT_METRICS


ROUTES = ("source", "codec", "vocoder", "joint")
TRAINING_ROUTES = ROUTES[1:]
_TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


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
        raise FileExistsError(f"Refusing to replace final-eval marker: {path}")
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


def _selected_checkpoint(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    selection_path = run_dir / "checkpoint_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = _load_json(selection_path)
    if selection.get("monitor") != "val_loss" or selection.get("mode") != "min":
        raise ValueError(f"Unexpected selector definition: {selection_path}")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError(
            f"Selector must contain two ranked candidates: {selection_path}"
        )
    best = selection.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"Selector has no best checkpoint: {selection_path}")
    filename = best.get("filename")
    expected_hash = best.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_hash, str):
        raise ValueError(f"Invalid best checkpoint record: {selection_path}")
    checkpoint = (run_dir / "checkpoints" / filename).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_hash = sha256_file(checkpoint)
    if observed_hash != expected_hash:
        raise ValueError(f"Selected checkpoint SHA-256 mismatch: {checkpoint}")
    last = selection.get("last_checkpoint")
    if not isinstance(last, dict) or last.get("global_step") != 100000:
        raise ValueError(
            f"Selector has no step100000 final checkpoint: {selection_path}"
        )
    last_filename = last.get("filename")
    last_expected_hash = last.get("sha256")
    if not isinstance(last_filename, str) or not isinstance(last_expected_hash, str):
        raise ValueError(f"Invalid final checkpoint record: {selection_path}")
    last_checkpoint = (run_dir / "checkpoints" / last_filename).resolve()
    if not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)
    last_observed_hash = sha256_file(last_checkpoint)
    if last_observed_hash != last_expected_hash:
        raise ValueError(f"Final checkpoint SHA-256 mismatch: {last_checkpoint}")
    return {
        "path": str(checkpoint),
        "sha256": observed_hash,
        "bytes": checkpoint.stat().st_size,
        "global_step": int(best["global_step"]),
        "last_checkpoint": {
            "path": str(last_checkpoint),
            "sha256": last_observed_hash,
            "bytes": last_checkpoint.stat().st_size,
            "global_step": 100000,
        },
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
    }


def _verify_training_receipt(
    *, route: str, run_dir: Path, selected: dict[str, Any], receipt_path: Path
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = _load_json(receipt_path)
    if receipt.get("direction") != route or receipt.get("final_global_step") != 100000:
        raise ValueError(f"Invalid final-selection receipt for {route}: {receipt_path}")
    if receipt.get("selection_sha256") != selected["selection"]["sha256"]:
        raise ValueError(f"{route} receipt does not bind its selector")
    if (
        receipt.get("best_filename") != Path(selected["path"]).name
        or receipt.get("best_sha256") != selected["sha256"]
    ):
        raise ValueError(f"{route} receipt does not bind its selected checkpoint")

    if selected["last_checkpoint"]["sha256"] != receipt.get("final_checkpoint_sha256"):
        raise ValueError(f"{route} final checkpoint SHA-256 mismatch")
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path)}


def prepare_final_eval_marker(
    *,
    repository_root: Path,
    expected_git_commit: str,
    evaluation_tag: str,
    run_directories: dict[str, Path],
    selection_receipts: dict[str, Path],
    route_script: Path,
    postprocess_script: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace final-eval marker: {output}")
    if set(run_directories) != set(ROUTES):
        raise ValueError(f"Run directories must cover exactly {ROUTES}")
    if set(selection_receipts) != set(TRAINING_ROUTES):
        raise ValueError(f"Selection receipts must cover exactly {TRAINING_ROUTES}")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("Expected git commit must be a full lowercase SHA-1")
    if not _TAG_PATTERN.fullmatch(evaluation_tag):
        raise ValueError(f"Unsafe evaluation tag: {evaluation_tag}")

    repository_root = repository_root.expanduser().resolve()
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise ValueError("Final evaluation requires a clean worktree")
    if commit != expected_git_commit:
        raise ValueError(f"Git commit mismatch: {commit} != {expected_git_commit}")

    checkpoints = {
        route: _selected_checkpoint(run_directories[route]) for route in ROUTES
    }
    for route in TRAINING_ROUTES:
        checkpoints[route]["selection_receipt"] = _verify_training_receipt(
            route=route,
            run_dir=run_directories[route],
            selected=checkpoints[route],
            receipt_path=selection_receipts[route],
        )

    scripts: dict[str, dict[str, Any]] = {}
    for label, value in (
        ("route", route_script),
        ("postprocess", postprocess_script),
    ):
        path = value.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        scripts[label] = {"path": str(path), "sha256": sha256_file(path)}

    marker = {
        "schema_version": 1,
        "status": "FINAL_EVAL_READY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "evaluation_tag": evaluation_tag,
        "encoding_policy": dict(OursAdapter.ENCODING_POLICY),
        "metrics": sorted(DEFAULT_METRICS.split(",")),
        "checkpoints": checkpoints,
        "queue_scripts": scripts,
        "test_access": False,
    }
    _atomic_json(output, marker)
    return marker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the hash-bound marker that releases M+ final dev evaluation."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--evaluation-tag", required=True)
    for route in ROUTES:
        parser.add_argument(f"--{route}-run-dir", type=Path, required=True)
    for route in TRAINING_ROUTES:
        parser.add_argument(f"--{route}-selection-receipt", type=Path, required=True)
    parser.add_argument("--route-script", type=Path, required=True)
    parser.add_argument("--postprocess-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    marker = prepare_final_eval_marker(
        repository_root=args.repository_root,
        expected_git_commit=args.expected_git_commit,
        evaluation_tag=args.evaluation_tag,
        run_directories={route: getattr(args, f"{route}_run_dir") for route in ROUTES},
        selection_receipts={
            route: getattr(args, f"{route}_selection_receipt")
            for route in TRAINING_ROUTES
        },
        route_script=args.route_script,
        postprocess_script=args.postprocess_script,
        output=args.output,
    )
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
