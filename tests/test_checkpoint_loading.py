from __future__ import annotations

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dmel_codec.train_codec import get_config, load_initial_weights
from dmel_codec.utils.callbacks import (
    FinalizedModelCheckpoint,
    StopAfterValidationCheckpoint,
)


def test_init_from_loads_only_model_weights(tmp_path) -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "optimizer_states": [{"sentinel": "must not be restored"}],
            "global_step": 999,
        },
        checkpoint,
    )

    result = load_initial_weights(target, checkpoint, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters()
    ):
        torch.testing.assert_close(source_parameter, target_parameter)


def _smoke_model():
    config = get_config(
        [
            "experiment=smoke",
            "codec.model.lr_scheduler.lr_lambda.num_training_steps=4",
        ]
    )
    return hydra.utils.instantiate(config.codec.model, _convert_="partial")


def _batch() -> dict:
    audio = torch.randn(1, 1, 4096, generator=torch.Generator().manual_seed(8))
    return {
        "audios": audio,
        "audio_lengths": torch.tensor([audio.shape[-1]]),
        "cut_ids": ["checkpoint-cut"],
    }


def _trainer(*, max_steps: int, max_epochs: int, callbacks=None) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        max_epochs=max_epochs,
        limit_train_batches=1,
        logger=False,
        callbacks=callbacks,
        enable_checkpointing=callbacks is not None,
        enable_model_summary=False,
    )


def test_resume_restores_training_state_while_init_starts_fresh(tmp_path) -> None:
    loader = DataLoader([_batch()], batch_size=None)
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        every_n_train_steps=1,
        save_top_k=-1,
        save_last=True,
    )
    source = _smoke_model()
    source.set_checkpoint_metadata({"sentinel": "embedded"})
    first_trainer = _trainer(
        max_steps=2,
        max_epochs=2,
        callbacks=[checkpoint_callback],
    )
    first_trainer.fit(source, train_dataloaders=loader)
    checkpoint_path = checkpoint_callback.last_model_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert first_trainer.global_step == 2
    assert checkpoint["global_step"] == 2
    assert checkpoint["codec_metadata"]["sentinel"] == "embedded"
    assert checkpoint["optimizer_states"]
    assert checkpoint["lr_schedulers"]

    resumed = _smoke_model()
    resumed_trainer = _trainer(max_steps=4, max_epochs=4)
    resumed_trainer.fit(
        resumed,
        train_dataloaders=loader,
        ckpt_path=checkpoint_path,
    )
    assert resumed_trainer.global_step == 4

    initialized = _smoke_model()
    result = load_initial_weights(initialized, checkpoint_path, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    for saved, loaded in zip(source.parameters(), initialized.parameters()):
        torch.testing.assert_close(saved, loaded)

    fresh_trainer = _trainer(max_steps=1, max_epochs=1)
    fresh_trainer.fit(initialized, train_dataloaders=loader)
    assert fresh_trainer.global_step == 1


class _TinyValidationModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return (self.weight * batch).square().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", (self.weight * batch).square().mean())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_finalized_checkpoint_saves_true_final_step_between_validations(tmp_path) -> None:
    callback = FinalizedModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="step-{step:08d}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        max_epochs=10,
        val_check_interval=2,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=[callback],
        enable_model_summary=False,
    )
    train_loader = DataLoader([torch.tensor([1.0])] * 4, batch_size=1)
    val_loader = DataLoader([torch.tensor([1.0])], batch_size=1)
    trainer.fit(
        _TinyValidationModel(),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    last = torch.load(
        tmp_path / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert trainer.global_step == 3
    assert last["global_step"] == 3
    assert callback.best_model_path.endswith("step-00000002.ckpt")


def test_runtime_segment_stops_after_a_complete_validation_checkpoint(tmp_path) -> None:
    checkpoint = FinalizedModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="step-{step:08d}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=10,
        max_epochs=10,
        val_check_interval=2,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=[checkpoint, StopAfterValidationCheckpoint(2)],
        enable_model_summary=False,
    )
    train_loader = DataLoader([torch.tensor([1.0])] * 12, batch_size=1)
    val_loader = DataLoader([torch.tensor([1.0])], batch_size=1)
    trainer.fit(
        _TinyValidationModel(),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    saved = torch.load(
        tmp_path / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert trainer.global_step == 2
    assert saved["global_step"] == 2
    assert checkpoint.best_model_path.endswith("step-00000002.ckpt")
