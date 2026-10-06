from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

from dmel_codec.baselines.dmel import (
    UniformMelQuantizer,
    load_range_metadata,
    make_frontend,
    sha256_file,
)
from dmel_codec.models.modules.wavenet import WaveNet
from dmel_codec.utils.spectrogram import length_aware_spectrogram
from dmel_codec.utils.utils import sequence_mask


class QuantizedMelDecoder(torch.nn.Module):
    """Deterministic residual decoder from dequantized dMel to continuous mel."""

    def __init__(
        self,
        *,
        num_mels: int = 100,
        hidden_channels: int = 256,
        residual_layers: int = 12,
        dilation_cycle: int = 4,
    ) -> None:
        super().__init__()
        self.num_mels = int(num_mels)
        self.hidden_channels = int(hidden_channels)
        self.residual_layers = int(residual_layers)
        self.dilation_cycle = int(dilation_cycle)
        self.refiner = WaveNet(
            input_channels=self.num_mels,
            output_channels=self.num_mels,
            residual_channels=self.hidden_channels,
            residual_layers=self.residual_layers,
            dilation_cycle=self.dilation_cycle,
        )
        # Start as the direct-dequantization control. This makes a pilot's gain
        # attributable to learned refinement rather than random initialization.
        torch.nn.init.zeros_(self.refiner.output_projection.conv.weight)
        torch.nn.init.zeros_(self.refiner.output_projection.conv.bias)

    def definition(self) -> dict[str, int | str]:
        return {
            "architecture": "residual_noncausal_wavenet",
            "num_mels": self.num_mels,
            "hidden_channels": self.hidden_channels,
            "residual_layers": self.residual_layers,
            "dilation_cycle": self.dilation_cycle,
        }

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[1] != self.num_mels:
            raise ValueError(
                f"Expected mel [B,{self.num_mels},F], got {tuple(mel.shape)}"
            )
        if mask is None:
            mask = torch.ones(
                mel.shape[0], 1, mel.shape[-1], device=mel.device, dtype=mel.dtype
            )
        if mask.shape != (mel.shape[0], 1, mel.shape[-1]):
            raise ValueError(f"Expected mask [B,1,F], got {tuple(mask.shape)}")
        masked = mel * mask
        return (masked + self.refiner(masked) * mask) * mask


class DmelDecoderLitModule(pl.LightningModule):
    """Train a mel-domain decoder while keeping the scalar tokenizer fixed."""

    def __init__(
        self,
        *,
        range_metadata: str | Path,
        hidden_channels: int = 256,
        residual_layers: int = 12,
        dilation_cycle: int = 4,
        static_loss_weight: float = 1.0,
        delta_loss_weight: float = 0.0,
        delta2_loss_weight: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_steps: int = 100,
        training_steps: int = 10_000,
        final_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        if static_loss_weight <= 0:
            raise ValueError("static_loss_weight must be positive")
        if min(delta_loss_weight, delta2_loss_weight) < 0:
            raise ValueError("temporal loss weights must be non-negative")
        if training_steps <= warmup_steps or warmup_steps < 0:
            raise ValueError("training_steps must exceed non-negative warmup_steps")
        self.save_hyperparameters()
        self.range_path = Path(range_metadata)
        metadata = load_range_metadata(self.range_path)
        quantizer = metadata["quantizer"]
        self.frontend = make_frontend()
        self.quantizer = UniformMelQuantizer(
            quantizer["minimum"], quantizer["maximum"], quantizer["num_bins"]
        )
        self.decoder = QuantizedMelDecoder(
            num_mels=100,
            hidden_channels=hidden_channels,
            residual_layers=residual_layers,
            dilation_cycle=dilation_cycle,
        )
        self.checkpoint_metadata: dict[str, Any] = {}

    def set_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        self.checkpoint_metadata = metadata

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["dmel_decoder_definition"] = self.decoder.definition()
        checkpoint["dmel_range"] = {
            "path": str(self.range_path.resolve()),
            "sha256": sha256_file(self.range_path),
        }
        checkpoint["dmel_decoder_metadata"] = self.checkpoint_metadata

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.expand(value.shape[0], value.shape[1], value.shape[2])
        return (value * expanded).sum() / expanded.sum().clamp_min(1.0)

    def reconstruction_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        static = self._masked_mean((prediction - target).abs(), mask)
        delta_mask = mask[..., 1:] * mask[..., :-1]
        prediction_delta = prediction[..., 1:] - prediction[..., :-1]
        target_delta = target[..., 1:] - target[..., :-1]
        delta = self._masked_mean(
            (prediction_delta - target_delta).abs(), delta_mask
        )
        delta2_mask = delta_mask[..., 1:] * delta_mask[..., :-1]
        prediction_delta2 = prediction_delta[..., 1:] - prediction_delta[..., :-1]
        target_delta2 = target_delta[..., 1:] - target_delta[..., :-1]
        delta2 = self._masked_mean(
            (prediction_delta2 - target_delta2).abs(), delta2_mask
        )
        total = (
            self.hparams.static_loss_weight * static
            + self.hparams.delta_loss_weight * delta
            + self.hparams.delta2_loss_weight * delta2
        )
        return total, {"static": static, "delta": delta, "delta2": delta2}

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        audios = batch["audios"].float()
        lengths = batch["audio_lengths"]
        with torch.no_grad():
            target = length_aware_spectrogram(self.frontend, audios, lengths).float()
            codes = self.quantizer.encode(target)
            dequantized = self.quantizer.decode(codes)
        frame_lengths = torch.div(lengths, 256, rounding_mode="floor")
        mask = sequence_mask(frame_lengths, target.shape[-1])[:, None].to(target.dtype)
        prediction = self.decoder(dequantized, mask)
        loss, components = self.reconstruction_loss(prediction, target * mask, mask)
        batch_size = audios.shape[0]
        self.log(
            f"{stage}_loss" if stage == "val" else f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in components.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.99),
            eps=1e-5,
        )

        def schedule(step: int) -> float:
            if step < self.hparams.warmup_steps:
                return float(step + 1) / max(1, self.hparams.warmup_steps)
            progress = (step - self.hparams.warmup_steps) / max(
                1, self.hparams.training_steps - self.hparams.warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return self.hparams.final_lr_ratio + (
                1.0 - self.hparams.final_lr_ratio
            ) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def load_quantized_mel_decoder(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[QuantizedMelDecoder, dict[str, Any]]:
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    definition = payload.get("dmel_decoder_definition")
    if not isinstance(definition, dict):
        raise ValueError("Checkpoint lacks dmel_decoder_definition")
    decoder = QuantizedMelDecoder(
        num_mels=int(definition["num_mels"]),
        hidden_channels=int(definition["hidden_channels"]),
        residual_layers=int(definition["residual_layers"]),
        dilation_cycle=int(definition["dilation_cycle"]),
    )
    prefix = "decoder."
    state = {
        key[len(prefix) :]: value
        for key, value in payload.get("state_dict", {}).items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError("Checkpoint lacks decoder parameters")
    decoder.load_state_dict(state, strict=True)
    decoder = decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder, payload
