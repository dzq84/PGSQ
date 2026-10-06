from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import yaml
from lhotse import CutSet

from dmel_codec.dataset.lhotse_tts_dataset import LhotseDataModule
from dmel_codec.dataset.prepare_libritts import sha256_file


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return float(ordered[index])


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "min": float(min(values)),
        "p05": _percentile(values, 0.05),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _load_configuration(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def replay_training_exposure(
    resolved_config: str | Path,
    *,
    max_steps: int | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Replay the configured Lhotse sampler without loading any audio."""

    config_path = Path(resolved_config).resolve()
    config = _load_configuration(config_path)
    try:
        module_config = dict(config["data"]["module"])
        configured_steps = int(config["experiment"]["trainer"]["max_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "resolved config must define data.module and experiment.trainer.max_steps"
        ) from error

    target = module_config.pop("_target_", None)
    expected_target = "dmel_codec.dataset.lhotse_tts_dataset.LhotseDataModule"
    if target not in (None, expected_target):
        raise ValueError(f"Unsupported data module target: {target!r}")
    if not bool(module_config.get("train_random_crop")):
        raise ValueError("Exposure replay currently requires train_random_crop=true")

    steps_requested = configured_steps if max_steps is None else int(max_steps)
    if steps_requested <= 0:
        raise ValueError("max_steps must be positive")
    crop_seconds = float(module_config["train_max_audio_seconds"])
    max_batch_seconds = float(module_config["train_max_durations"])
    if crop_seconds <= 0 or max_batch_seconds <= 0:
        raise ValueError("crop and batch durations must be positive")

    manifest_path = Path(module_config["train_cuts_path"]).resolve()
    manifest_cuts = list(CutSet.from_file(manifest_path))
    if not manifest_cuts:
        raise ValueError(f"Training manifest is empty: {manifest_path}")
    manifest_raw_seconds = sum(float(cut.duration) for cut in manifest_cuts)
    manifest_capped_seconds = sum(
        min(float(cut.duration), crop_seconds) for cut in manifest_cuts
    )

    # Keep construction identical to training while avoiding DataLoader workers and
    # forcing the single-rank schedule whose exposure is being reported.
    module_config.update(
        {
            "train_num_workers": 0,
            "val_num_workers": 0,
            "test_num_workers": 0,
            "world_size": 1,
            "rank": 0,
        }
    )
    datamodule = LhotseDataModule(**module_config)
    datamodule.setup("fit")
    sampler = datamodule.train_sampler
    if sampler is None:
        raise RuntimeError("Training sampler was not initialized")

    batch_seconds: list[float] = []
    cuts_per_batch: list[float] = []
    sampled_cut_ids: set[str] = set()
    sampled_cuts = 0
    epochs_entered = 0
    while len(batch_seconds) < steps_requested:
        sampler.set_epoch(epochs_entered)
        steps_before_epoch = len(batch_seconds)
        for batch in sampler:
            duration = sum(
                min(float(cut.duration), crop_seconds) for cut in batch
            )
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Sampler produced a batch with invalid duration")
            batch_seconds.append(duration)
            cuts_per_batch.append(float(len(batch)))
            sampled_cuts += len(batch)
            sampled_cut_ids.update(cut.id for cut in batch)
            if len(batch_seconds) == steps_requested:
                break
        if len(batch_seconds) == steps_before_epoch:
            raise RuntimeError(f"Sampler epoch {epochs_entered} produced no batches")
        epochs_entered += 1

    exposure_seconds = float(sum(batch_seconds))
    result = {
        "schema_version": 1,
        "resolved_config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "cuts": len(manifest_cuts),
            "raw_duration_seconds": manifest_raw_seconds,
            "raw_duration_hours": manifest_raw_seconds / 3600.0,
            "crop_capped_duration_seconds": manifest_capped_seconds,
            "crop_capped_duration_hours": manifest_capped_seconds / 3600.0,
        },
        "sampler": {
            "seed": int(module_config["seed"]),
            "world_size": 1,
            "rank": 0,
            "shuffle": True,
            "crop_seconds": crop_seconds,
            "configured_batch_duration_seconds": max_batch_seconds,
            "duration_constraint_is_strict": False,
            "waveforms_concatenated": False,
        },
        "replay": {
            "steps": len(batch_seconds),
            "epochs_entered": epochs_entered,
            "sampled_cuts": sampled_cuts,
            "unique_cut_ids": len(sampled_cut_ids),
            "effective_duration_seconds": exposure_seconds,
            "effective_duration_hours": exposure_seconds / 3600.0,
            "fraction_of_one_crop_capped_manifest_pass": (
                exposure_seconds / manifest_capped_seconds
            ),
            "batch_duration_seconds": _distribution(batch_seconds),
            "cuts_per_batch": _distribution(cuts_per_batch),
            "batches_below_99_percent_of_duration_limit": sum(
                duration < 0.99 * max_batch_seconds for duration in batch_seconds
            ),
            "batches_above_configured_duration": sum(
                duration > max_batch_seconds for duration in batch_seconds
            ),
        },
    }
    if output is not None:
        _atomic_json(Path(output).resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a resolved training sampler and report audio exposure"
    )
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay_training_exposure(
        args.resolved_config,
        max_steps=args.max_steps,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
