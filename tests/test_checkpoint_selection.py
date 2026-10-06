from __future__ import annotations

import json

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from dmel_codec.tools.select_checkpoint import select_checkpoint


CALLBACK_KEY = (
    "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', "
    "'every_n_train_steps': 0, 'every_n_epochs': 1, "
    "'train_time_interval': None}"
)


def _checkpoint_payload(
    run_dir, global_step: int, score: float, *, serialized_mode: str | None = "min"
) -> dict:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = {
        str(checkpoint_dir / "step-00000010.ckpt"): torch.tensor(0.8),
        str(checkpoint_dir / "step-00000020.ckpt"): torch.tensor(0.6),
    }
    return {
        "global_step": global_step,
        "state_dict": {"weight": torch.tensor([global_step], dtype=torch.float32)},
        "callbacks": {
            CALLBACK_KEY: {
                "monitor": "val_loss",
                "mode": serialized_mode,
                "best_model_score": torch.tensor(0.6),
                "best_model_path": str(checkpoint_dir / "step-00000020.ckpt"),
                "current_score": torch.tensor(score),
                "best_k_models": candidates,
            }
        },
    }


def test_checkpoint_selection_cross_checks_tensorboard_and_hashes(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        _checkpoint_payload(run_dir, 10, 0.8),
        checkpoint_dir / "step-00000010.ckpt",
    )
    torch.save(
        _checkpoint_payload(run_dir, 20, 0.6),
        checkpoint_dir / "step-00000020.ckpt",
    )
    torch.save(_checkpoint_payload(run_dir, 25, 0.7), checkpoint_dir / "last.ckpt")
    writer = SummaryWriter(str(run_dir / "logs"))
    writer.add_scalar("val_loss", 0.8, 9)
    writer.add_scalar("val_loss", 0.6, 19)
    writer.close()

    result = select_checkpoint(run_dir)
    assert result["best"]["filename"] == "step-00000020.ckpt"
    assert result["best"]["global_step"] == 20
    assert len(result["best"]["sha256"]) == 64
    assert json.loads(
        (run_dir / "checkpoint_selection.json").read_text(encoding="utf-8")
    ) == result


def test_checkpoint_selection_accepts_lightning_state_without_serialized_mode(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step, score in ((10, 0.8), (20, 0.6)):
        torch.save(
            _checkpoint_payload(run_dir, step, score, serialized_mode=None),
            checkpoint_dir / f"step-{step:08d}.ckpt",
        )
    torch.save(
        _checkpoint_payload(run_dir, 25, 0.7, serialized_mode=None),
        checkpoint_dir / "last.ckpt",
    )
    writer = SummaryWriter(str(run_dir / "logs"))
    writer.add_scalar("val_loss", 0.8, 9)
    writer.add_scalar("val_loss", 0.6, 19)
    writer.close()

    result = select_checkpoint(run_dir, mode="min")
    assert result["best"]["filename"] == "step-00000020.ckpt"


def test_checkpoint_selection_rejects_tensorboard_score_mismatch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step, score in ((10, 0.8), (20, 0.6)):
        torch.save(
            _checkpoint_payload(run_dir, step, score),
            checkpoint_dir / f"step-{step:08d}.ckpt",
        )
    torch.save(_checkpoint_payload(run_dir, 25, 0.7), checkpoint_dir / "last.ckpt")
    writer = SummaryWriter(str(run_dir / "logs"))
    writer.add_scalar("val_loss", 0.8, 9)
    writer.add_scalar("val_loss", 0.7, 19)
    writer.close()

    with pytest.raises(ValueError, match="Checkpoint/TensorBoard score mismatch"):
        select_checkpoint(run_dir)
