from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from dmel_codec.dataset.lhotse_tts_dataset import LhotseDataModule
from dmel_codec.models.dmel_decoder_lit_module import DmelDecoderLitModule
from dmel_codec.utils.callbacks import FinalizedModelCheckpoint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository, text=True
        ).strip()

    status = run("status", "--short").splitlines()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def _validate_path(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _validate_output(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"output_dir must be absolute: {path}")
    repository = Path(__file__).resolve().parents[1]
    resolved = path.resolve()
    if resolved == repository or repository in resolved.parents:
        raise ValueError("output_dir must be outside the git checkout")
    return resolved


def _scientific_configuration(args: argparse.Namespace) -> dict[str, Any]:
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    # The recovery location is execution provenance, not a scientific
    # hyperparameter. Normalizing it keeps a resumed run's identity stable.
    configuration["resume_from"] = None
    return configuration


def _build_datamodule(
    args: argparse.Namespace,
    *,
    train_manifest: Path,
    val_manifest: Path,
) -> LhotseDataModule:
    return LhotseDataModule(
        stage="fit",
        train_cuts_path=str(train_manifest),
        val_cuts_path=str(val_manifest),
        train_max_durations=args.train_batch_seconds,
        val_max_durations=args.val_batch_seconds,
        train_num_workers=args.train_workers,
        val_num_workers=args.val_workers,
        pin_memory=True,
        seed=args.seed,
        train_max_audio_seconds=3.0,
        train_random_crop=True,
        train_crop_seed=args.seed,
    )


def train(args: argparse.Namespace) -> Path:
    train_manifest = _validate_path(args.train_manifest, "train_manifest")
    val_manifest = _validate_path(args.val_manifest, "val_manifest")
    range_metadata = _validate_path(args.range_metadata, "range_metadata")
    resume_from = (
        _validate_path(args.resume_from, "resume_from")
        if args.resume_from is not None
        else None
    )
    output_dir = _validate_output(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume_from is None:
        raise FileExistsError(
            f"Refusing to mix a fresh run into non-empty output_dir: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)

    model = DmelDecoderLitModule(
        range_metadata=range_metadata,
        hidden_channels=args.hidden_channels,
        residual_layers=args.residual_layers,
        dilation_cycle=args.dilation_cycle,
        static_loss_weight=args.static_loss_weight,
        delta_loss_weight=args.delta_loss_weight,
        delta2_loss_weight=args.delta2_loss_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        training_steps=args.max_steps,
        final_lr_ratio=args.final_lr_ratio,
    )
    resolved = _scientific_configuration(args)
    config_sha256 = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    resolved_path = output_dir / "resolved_config.json"
    if resume_from is not None:
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Resume output_dir lacks resolved_config.json: {resolved_path}"
            )
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        previous["resume_from"] = None
        if previous != resolved:
            raise ValueError("Resume configuration differs from the original run")
    data_metadata = {
        "train_manifest": {
            "path": str(train_manifest),
            "sha256": _sha256_file(train_manifest),
        },
        "val_manifest": {
            "path": str(val_manifest),
            "sha256": _sha256_file(val_manifest),
        },
        "range_metadata": {
            "path": str(range_metadata),
            "sha256": _sha256_file(range_metadata),
        },
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "configuration": resolved,
        "git": _git_state(),
        "data": data_metadata,
        "parameters": sum(parameter.numel() for parameter in model.decoder.parameters()),
        "resume": (
            {
                "checkpoint": str(resume_from),
                "checkpoint_sha256": _sha256_file(resume_from),
            }
            if resume_from is not None
            else None
        ),
    }
    model.set_checkpoint_metadata(metadata)
    if resume_from is None:
        _atomic_json(resolved_path, resolved)
    _atomic_json(output_dir / "data_manifest.json", data_metadata)
    _atomic_json(output_dir / "git_state.json", metadata["git"])
    _atomic_json(
        output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    _atomic_json(
        output_dir / "model_metadata.json",
        {
            "decoder_definition": model.decoder.definition(),
            "parameters": metadata["parameters"],
        },
    )

    datamodule = _build_datamodule(
        args,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
    )
    checkpoint = FinalizedModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="step-{step:08d}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        save_top_k=2,
        save_last=True,
    )
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs", name="tensorboard", log_graph=False
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_steps=args.max_steps,
        max_epochs=1000,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=20,
        callbacks=[checkpoint],
        logger=logger,
        use_distributed_sampler=False,
        enable_progress_bar=args.enable_progress_bar,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LibriTTS-only quantization-aware decoder for adapted dMel"
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--range-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--residual-layers", type=int, default=12)
    parser.add_argument("--dilation-cycle", type=int, default=4)
    parser.add_argument("--static-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--delta2-loss-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--val-check-interval", type=int, default=500)
    parser.add_argument("--train-batch-seconds", type=float, default=60.0)
    parser.add_argument("--val-batch-seconds", type=float, default=30.0)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--enable-progress-bar", action="store_true")
    args = parser.parse_args(argv)
    if min(args.hidden_channels, args.residual_layers, args.dilation_cycle) <= 0:
        parser.error("decoder dimensions must be positive")
    if args.max_steps <= args.warmup_steps or args.val_check_interval <= 0:
        parser.error("max_steps must exceed warmup_steps; validation interval positive")
    if min(args.train_batch_seconds, args.val_batch_seconds) <= 0:
        parser.error("batch seconds must be positive")
    return args


def main() -> int:
    print(train(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
