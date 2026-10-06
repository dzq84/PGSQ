from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from dmel_codec.baselines.dac import (
    DAC_LICENSE,
    DAC_PACKAGE_COMMIT,
    DAC_PACKAGE_VERSION,
    DAC_SOURCE,
    DAC_WEIGHT_TAG,
    DAC_WEIGHT_URL,
    load_dac_model,
)
from dmel_codec.evaluation.adapters.base import CodecAdapter, sha256_file


class DACAdapter(CodecAdapter):
    sample_rate = 24_000
    token_frame_rate = 75.0

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        n_quantizers: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 1 <= n_quantizers <= 32:
            raise ValueError("DAC n_quantizers must be within [1,32]")
        self.n_quantizers = int(n_quantizers)
        self.system_name = f"DAC 24 kHz q={self.n_quantizers}"
        self.num_parallel_streams = self.n_quantizers
        self.codebook_cardinalities = (1024,) * self.n_quantizers
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.model = load_dac_model(self.checkpoint, self.device)

    @property
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        return {
            "source": DAC_SOURCE,
            "package_version": DAC_PACKAGE_VERSION,
            "package_commit": DAC_PACKAGE_COMMIT,
            "license": DAC_LICENSE,
            "weight_tag": DAC_WEIGHT_TAG,
            "weight_url": DAC_WEIGHT_URL,
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "download_label": "8kbps",
            "all_32_codebooks_nominal_bitrate_kbps": 24.0,
            "pretrained_external": True,
        }

    def code_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(audio_lengths + 319, 320, rounding_mode="floor")

    @torch.inference_mode()
    def encode(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_waveforms(waveform, lengths)
        waveform = waveform.to(self.device).float()
        lengths = lengths.to(self.device)
        encoded = []
        frame_lengths = self.code_lengths(lengths)
        for index, audio_length in enumerate(lengths.tolist()):
            item = waveform[index : index + 1, :, :audio_length]
            item = self.model.preprocess(item, sample_rate=self.sample_rate)
            _, codes, _, _, _ = self.model.encode(
                item, n_quantizers=self.n_quantizers
            )
            encoded.append(codes.long())
        max_frames = max(item.shape[-1] for item in encoded)
        codes = torch.cat(
            [F.pad(item, (0, max_frames - item.shape[-1])) for item in encoded], dim=0
        )
        self.validate_codes(
            codes,
            batch_size=waveform.shape[0],
            num_streams=self.num_parallel_streams,
            cardinalities=self.codebook_cardinalities,
        )
        if not torch.equal(
            frame_lengths,
            torch.tensor(
                [item.shape[-1] for item in encoded],
                device=self.device,
                dtype=torch.long,
            ),
        ):
            raise RuntimeError("DAC observed code length disagrees with 320-sample hop")
        return codes, frame_lengths

    @torch.inference_mode()
    def decode(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        cut_ids: list[str],
    ) -> torch.Tensor:
        if len(cut_ids) != lengths.shape[0]:
            raise ValueError("cut_ids must contain one ID per code sequence")
        lengths = lengths.to(self.device)
        codes = codes.to(self.device)
        self.validate_codes(
            codes,
            batch_size=lengths.shape[0],
            num_streams=self.num_parallel_streams,
            cardinalities=self.codebook_cardinalities,
        )
        frame_lengths = self.code_lengths(lengths)
        waveforms = []
        for index, (frame_length, audio_length) in enumerate(
            zip(frame_lengths.tolist(), lengths.tolist())
        ):
            quantized = self.model.quantizer.from_codes(
                codes[index : index + 1, :, :frame_length]
            )[0]
            waveform = self.model.decode(quantized)[..., :audio_length]
            waveform = F.pad(waveform, (0, audio_length - waveform.shape[-1]))
            waveforms.append(waveform)
        return self.pad_waveforms(waveforms, lengths)
