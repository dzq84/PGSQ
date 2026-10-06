from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class FinalizedModelCheckpoint(ModelCheckpoint):
    """Keep validation-ranked checkpoints and refresh ``last`` at train end."""

    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        super().on_train_end(trainer, pl_module)
        if getattr(trainer, "_dmel_stopped_after_validation_checkpoint", False):
            # Keep the validation-time ``last`` state. Refreshing it after the
            # controlled stop would mark the partial epoch as completed.
            return
        # Lightning 2.3 does not save `last` when max_steps stops between validations.
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


class StopAfterValidationCheckpoint(Callback):
    """Stop only after the configured validation boundary has checkpointed."""

    def __init__(self, step: int) -> None:
        if step <= 0:
            raise ValueError("stop-after-validation step must be positive")
        self.step = int(step)

    def on_validation_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        del pl_module
        if not trainer.sanity_checking and trainer.global_step >= self.step:
            trainer._dmel_stopped_after_validation_checkpoint = True
            trainer.should_stop = True
