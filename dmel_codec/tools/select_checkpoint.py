from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dmel_codec.evaluation.cache import sha256_file


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_state(payload: dict[str, Any], monitor: str) -> dict[str, Any]:
    matches = []
    for key, state in payload.get("callbacks", {}).items():
        if not isinstance(state, dict):
            continue
        if state.get("monitor") == monitor and "best_k_models" in state:
            matches.append((str(key), state))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ModelCheckpoint callback monitoring {monitor!r}, "
            f"found {len(matches)}"
        )
    return matches[0][1]


def _score(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Checkpoint score is not finite: {result}")
    return result


def _event_scores(log_directory: Path, monitor: str) -> tuple[dict[int, float], list[dict[str, Any]]]:
    event_files = sorted(log_directory.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files under {log_directory}")
    observations: dict[int, tuple[float, float]] = {}
    metadata = []
    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if monitor in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(monitor):
                previous = observations.get(int(event.step))
                if previous is None or event.wall_time >= previous[0]:
                    observations[int(event.step)] = (event.wall_time, float(event.value))
        metadata.append(
            {
                "relative_path": str(path.relative_to(log_directory)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not observations:
        raise ValueError(f"TensorBoard logs do not contain scalar {monitor!r}")
    return {step: value for step, (_, value) in observations.items()}, metadata


def select_checkpoint(
    run_directory: str | Path,
    *,
    monitor: str = "val_loss",
    mode: str = "min",
    output: str | Path | None = None,
) -> dict[str, Any]:
    run_directory = Path(run_directory).resolve()
    checkpoint_directory = run_directory / "checkpoints"
    last_path = checkpoint_directory / "last.ckpt"
    if not last_path.is_file():
        raise FileNotFoundError(last_path)
    payload = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    state = _checkpoint_state(payload, monitor)
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode!r}")
    serialized_mode = state.get("mode")
    if serialized_mode is not None and str(serialized_mode) != mode:
        raise ValueError(
            f"Serialized checkpoint mode {serialized_mode!r} disagrees with {mode!r}"
        )

    candidates = []
    for recorded_path, recorded_score in state["best_k_models"].items():
        path = checkpoint_directory / Path(str(recorded_path)).name
        if not path.is_file():
            raise FileNotFoundError(f"Recorded checkpoint is missing: {path}")
        candidates.append((path, _score(recorded_score)))
    if not candidates:
        raise ValueError("ModelCheckpoint state has no best-k candidates")

    chooser = min if mode == "min" else max
    chosen_path, chosen_score = chooser(candidates, key=lambda item: item[1])
    recorded_best = checkpoint_directory / Path(str(state["best_model_path"])).name
    recorded_score = _score(state["best_model_score"])
    if recorded_best != chosen_path or not math.isclose(
        recorded_score, chosen_score, rel_tol=1e-7, abs_tol=1e-8
    ):
        raise ValueError("Recorded best checkpoint disagrees with recomputed best-k choice")

    event_scores, event_files = _event_scores(run_directory / "logs", monitor)
    candidate_records = []
    for path, score in sorted(candidates, key=lambda item: item[0].name):
        candidate_payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        global_step = int(candidate_payload["global_step"])
        event_step = global_step - 1
        event_score = event_scores.get(event_step)
        if event_score is None:
            raise ValueError(
                f"No {monitor} TensorBoard scalar at event step {event_step} "
                f"for checkpoint {path.name}"
            )
        if not math.isclose(score, event_score, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(
                f"Checkpoint/TensorBoard score mismatch for {path.name}: "
                f"{score} != {event_score}"
            )
        candidate_records.append(
            {
                "filename": path.name,
                "global_step": global_step,
                "event_step": event_step,
                "score": score,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    chosen_record = next(
        record for record in candidate_records if record["filename"] == chosen_path.name
    )
    result = {
        "schema_version": 1,
        "monitor": monitor,
        "mode": mode,
        "selection_rule": f"{mode} {monitor} on dev-clean; checkpoint callback and TensorBoard must agree",
        "best": chosen_record,
        "candidates": candidate_records,
        "last_checkpoint": {
            "filename": last_path.name,
            "global_step": int(payload["global_step"]),
            "bytes": last_path.stat().st_size,
            "sha256": sha256_file(last_path),
        },
        "tensorboard_events": event_files,
    }
    output_path = Path(output) if output is not None else run_directory / "checkpoint_selection.json"
    _atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute and verify dev checkpoint selection"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--monitor", default="val_loss")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = select_checkpoint(
        args.run_dir,
        monitor=args.monitor,
        mode=args.mode,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
