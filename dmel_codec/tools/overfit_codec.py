from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lhotse import CutSet
from torch.utils.data import DataLoader

from dmel_codec.dataset.lhotse_tts_dataset import LhotseTTSDataset
from dmel_codec.train_codec import get_config


class LossRecorder(Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if isinstance(outputs, dict):
            outputs = outputs["loss"]
        self.losses.append(float(outputs.detach().cpu()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_cuts(
    manifest: Path,
    *,
    num_cuts: int,
    min_duration: float,
    max_duration: float,
) -> CutSet:
    selected = []
    for cut in CutSet.from_file(manifest):
        if min_duration <= cut.duration <= max_duration:
            selected.append(cut)
            if len(selected) == num_cuts:
                break
    if len(selected) != num_cuts:
        raise ValueError(
            f"Found only {len(selected)} cuts in duration range "
            f"[{min_duration}, {max_duration}] in {manifest}"
        )
    return CutSet.from_cuts(selected)


def run_policy(
    *,
    batch: dict[str, Any],
    codec: str,
    experiment: str,
    policy: str,
    steps: int,
    seed: int,
    accelerator: str,
    overrides: list[str],
) -> dict[str, Any]:
    pl.seed_everything(seed, workers=True)
    config = get_config(
        [
            f"codec={codec}",
            f"experiment={experiment}",
            f"codec.model.decoder_input_policy={policy}",
            f"codec.model.lr_scheduler.lr_lambda.num_training_steps={steps}",
            *overrides,
        ]
    )
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    recorder = LossRecorder()
    loader = DataLoader([batch], batch_size=None)
    if accelerator == "gpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        max_steps=steps,
        max_epochs=steps,
        limit_train_batches=1,
        callbacks=[recorder],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=loader)
    elapsed = time.perf_counter() - started
    if len(recorder.losses) != steps:
        raise RuntimeError(
            f"Expected {steps} recorded losses, got {len(recorder.losses)}"
        )
    window = min(10, max(1, steps // 4))
    initial = mean(recorder.losses[:window])
    final = mean(recorder.losses[-window:])
    return {
        "policy": policy,
        "steps": steps,
        "global_step": trainer.global_step,
        "initial_loss_mean": initial,
        "final_loss_mean": final,
        "final_to_initial_ratio": final / initial,
        "best_loss": min(recorder.losses),
        "losses": recorder.losses,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if accelerator == "gpu" else None
        ),
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "codec_definition": model.codec_definition(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit fixed LibriTTS cuts and compare decoder input policies"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", default="ours_m")
    parser.add_argument("--experiment", default="smoke")
    parser.add_argument("--policies", nargs="+", default=["zeros", "cut_id_noise"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-cuts", type=int, default=8)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "gpu"),
        default="gpu" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; may be passed more than once",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator=gpu requested but CUDA is unavailable")
    cuts = select_cuts(
        manifest,
        num_cuts=args.num_cuts,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    batch = LhotseTTSDataset()[cuts]
    results = [
        run_policy(
            batch=batch,
            codec=args.codec,
            experiment=args.experiment,
            policy=policy,
            steps=args.steps,
            seed=args.seed,
            accelerator=args.accelerator,
            overrides=args.override,
        )
        for policy in args.policies
    ]
    payload = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selected_cut_ids": list(batch["cut_ids"]),
        "selected_durations_seconds": [
            length / 24_000 for length in batch["audio_lengths"].tolist()
        ],
        "seed": args.seed,
        "codec": args.codec,
        "experiment": args.experiment,
        "overrides": args.override,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(0) if args.accelerator == "gpu" else "cpu"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for result in results:
        print(
            f"{result['policy']}: {result['initial_loss_mean']:.6f} -> "
            f"{result['final_loss_mean']:.6f} "
            f"(ratio={result['final_to_initial_ratio']:.3f}, "
            f"{result['seconds_per_step']:.3f}s/step)"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
