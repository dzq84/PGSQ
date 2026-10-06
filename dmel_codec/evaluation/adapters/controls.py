from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dmel_codec.baselines.dmel import make_frontend
from dmel_codec.evaluation.adapters.base import Reconstruction, sha256_file
from dmel_codec.evaluation.adapters.dmel import (
    BIGVGAN_REVISION,
    BIGVGAN_SOURCE,
    load_bigvgan,
)
from dmel_codec.utils.spectrogram import length_aware_spectrogram


class GroundTruthControl:
    system_name = "Ground Truth"
    sample_rate = 24_000
    token_frame_rate = 0.0
    num_parallel_streams = 0
    codebook_cardinalities: tuple[int, ...] = ()
    nominal_bitrate = 0.0

    @property
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        return {"control": "identity", "checkpoint": None}

    def metadata(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "sample_rate": self.sample_rate,
            "token_frame_rate": self.token_frame_rate,
            "num_parallel_streams": 0,
            "codebook_cardinalities": [],
            "nominal_bitrate_kbps": 0.0,
            "checkpoint_revision_and_hash": self.checkpoint_revision_and_hash,
        }

    def reconstruct(
        self, waveform: torch.Tensor, lengths: torch.Tensor, cut_ids: list[str]
    ) -> Reconstruction:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("Ground Truth expects waveform [B,1,T]")
        if lengths.shape != (waveform.shape[0],) or len(cut_ids) != waveform.shape[0]:
            raise ValueError("Ground Truth batch metadata shape mismatch")
        output = waveform.clone()
        time = torch.arange(output.shape[-1], device=output.device)[None, :]
        output[:, 0].masked_fill_(time >= lengths.to(output.device)[:, None], 0)
        codes = torch.empty(waveform.shape[0], 0, 0, dtype=torch.long, device=output.device)
        return Reconstruction(output, codes, torch.zeros_like(lengths))


class MelOracleControl(GroundTruthControl):
    system_name = "Mel Oracle"
    token_frame_rate = 93.75

    def __init__(
        self,
        *,
        bigvgan_checkpoint: str | Path,
        bigvgan_config: str | Path,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.bigvgan_checkpoint = Path(bigvgan_checkpoint)
        self.bigvgan_config = Path(bigvgan_config)
        self.frontend = make_frontend().to(self.device).eval()
        self.vocoder = load_bigvgan(
            self.bigvgan_checkpoint, self.bigvgan_config, self.device
        )

    @property
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        return {
            "control": "unquantized ground-truth log-mel",
            "bigvgan": {
                "source": BIGVGAN_SOURCE,
                "revision": BIGVGAN_REVISION,
                "checkpoint_sha256": sha256_file(self.bigvgan_checkpoint),
                "config_sha256": sha256_file(self.bigvgan_config),
                "pretrained_external": True,
            },
        }

    @torch.inference_mode()
    def reconstruct(
        self, waveform: torch.Tensor, lengths: torch.Tensor, cut_ids: list[str]
    ) -> Reconstruction:
        if lengths.shape != (waveform.shape[0],) or len(cut_ids) != waveform.shape[0]:
            raise ValueError("Mel Oracle batch metadata shape mismatch")
        waveform = waveform.to(self.device).float()
        lengths = lengths.to(self.device)
        mel = length_aware_spectrogram(self.frontend, waveform, lengths)
        mel_lengths = torch.div(lengths, 256, rounding_mode="floor")
        outputs = []
        for index, (mel_length, audio_length) in enumerate(
            zip(mel_lengths.tolist(), lengths.tolist())
        ):
            decoded = self.vocoder(mel[index : index + 1, :, :mel_length])
            outputs.append(self.center_resize(decoded, audio_length))
        reconstructed = self.pad_waveforms(outputs, lengths)
        codes = torch.empty(waveform.shape[0], 0, 0, dtype=torch.long, device=self.device)
        return Reconstruction(reconstructed, codes, torch.zeros_like(lengths))

    @staticmethod
    def center_resize(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        difference = target_length - waveform.shape[-1]
        if difference > 0:
            left = difference // 2
            return torch.nn.functional.pad(waveform, (left, difference - left))
        if difference < 0:
            left = (-difference) // 2
            return waveform[..., left : left + target_length]
        return waveform

    @staticmethod
    def pad_waveforms(
        waveforms: list[torch.Tensor], lengths: torch.Tensor
    ) -> torch.Tensor:
        target = int(lengths.max().item())
        return torch.cat(
            [
                torch.nn.functional.pad(waveform, (0, target - length))
                for waveform, length in zip(waveforms, lengths.tolist())
            ],
            dim=0,
        )
