from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dmel_codec.baselines.dmel import (
    APPLE_DMEL_LICENSE,
    APPLE_DMEL_REVISION,
    APPLE_DMEL_SOURCE,
    UniformMelQuantizer,
    load_range_metadata,
    make_frontend,
)
from dmel_codec.evaluation.adapters.base import CodecAdapter, sha256_file
from dmel_codec.models.dmel_decoder_lit_module import load_quantized_mel_decoder
from dmel_codec.models.modules.bigvgan.bigvgan import BigVGAN
from dmel_codec.utils.spectrogram import length_aware_spectrogram


BIGVGAN_SOURCE = "https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x"
BIGVGAN_REVISION = "c329ede9e9bbc100ddf5c91e2330a61921262370"


def load_bigvgan(
    checkpoint: str | Path,
    config: str | Path,
    device: torch.device,
) -> BigVGAN:
    checkpoint = Path(checkpoint)
    config = Path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)
    model = BigVGAN(
        h_path=str(config),
        ckpt_path=str(checkpoint),
        use_cuda_kernel=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["generator"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class DmelSameFrontendAdapter(CodecAdapter):
    system_name = "dMel-same-front-end (adapted)"
    sample_rate = 24_000
    token_frame_rate = 93.75
    num_parallel_streams = 100
    codebook_cardinalities = (16,) * 100

    def __init__(
        self,
        *,
        range_metadata: str | Path,
        bigvgan_checkpoint: str | Path,
        bigvgan_config: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.range_path = Path(range_metadata)
        self.range_metadata = load_range_metadata(self.range_path)
        quantizer = self.range_metadata["quantizer"]
        self.frontend = make_frontend().to(self.device).eval()
        self.quantizer = UniformMelQuantizer(
            quantizer["minimum"], quantizer["maximum"], quantizer["num_bins"]
        ).to(self.device)
        self.bigvgan_checkpoint = Path(bigvgan_checkpoint)
        self.bigvgan_config = Path(bigvgan_config)
        self.vocoder = load_bigvgan(
            self.bigvgan_checkpoint, self.bigvgan_config, self.device
        )

    @property
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        return {
            "apple_dmel": {
                "source": APPLE_DMEL_SOURCE,
                "revision": APPLE_DMEL_REVISION,
                "license": APPLE_DMEL_LICENSE,
                "role": "scalar quantizer reference; adapted frontend/range",
            },
            "range_metadata": {
                "sha256": sha256_file(self.range_path),
                "source_manifest_sha256": self.range_metadata["source"][
                    "manifest_sha256"
                ],
            },
            "bigvgan": {
                "source": BIGVGAN_SOURCE,
                "revision": BIGVGAN_REVISION,
                "checkpoint_sha256": sha256_file(self.bigvgan_checkpoint),
                "config_sha256": sha256_file(self.bigvgan_config),
                "pretrained_external": True,
            },
        }

    def code_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(audio_lengths, 256, rounding_mode="floor")

    def decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return mel

    @torch.inference_mode()
    def encode(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_waveforms(waveform, lengths)
        waveform = waveform.to(self.device).float()
        lengths = lengths.to(self.device)
        features = length_aware_spectrogram(self.frontend, waveform, lengths)
        codes = self.quantizer.encode(features)
        frame_lengths = self.code_lengths(lengths)
        mask = (
            torch.arange(codes.shape[-1], device=self.device)[None, :]
            < frame_lengths[:, None]
        )
        codes = codes.masked_fill(~mask[:, None, :], 0)
        self.validate_codes(
            codes,
            batch_size=waveform.shape[0],
            num_streams=self.num_parallel_streams,
            cardinalities=self.codebook_cardinalities,
        )
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
            mel = self.quantizer.decode(codes[index : index + 1, :, :frame_length])
            mel = self.decode_mel(mel)
            waveform = self.vocoder(mel)
            waveforms.append(self.center_resize(waveform, audio_length))
        return self.pad_waveforms(waveforms, lengths)


class DmelAdaptedDecoderAdapter(DmelSameFrontendAdapter):
    system_name = "dMel-quantization-aware-decoder (adapted)"

    def __init__(
        self,
        *,
        decoder_checkpoint: str | Path,
        range_metadata: str | Path,
        bigvgan_checkpoint: str | Path,
        bigvgan_config: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            range_metadata=range_metadata,
            bigvgan_checkpoint=bigvgan_checkpoint,
            bigvgan_config=bigvgan_config,
            device=device,
        )
        self.decoder_checkpoint = Path(decoder_checkpoint)
        self.mel_decoder, payload = load_quantized_mel_decoder(
            self.decoder_checkpoint, self.device
        )
        embedded_range = payload.get("dmel_range", {})
        expected_range_hash = sha256_file(self.range_path)
        if embedded_range.get("sha256") != expected_range_hash:
            raise ValueError(
                "dMel decoder checkpoint was trained with a different quantizer range"
            )
        self.decoder_metadata = payload.get("dmel_decoder_metadata", {})

    def _public_decoder_metadata(self) -> dict[str, Any]:
        metadata = self.decoder_metadata
        data = metadata.get("data", {})
        configuration = metadata.get("configuration", {})
        return {
            "config_sha256": metadata.get("config_sha256"),
            "git_commit": metadata.get("git", {}).get("commit"),
            "git_dirty": metadata.get("git", {}).get("dirty"),
            "train_manifest_sha256": data.get("train_manifest", {}).get("sha256"),
            "val_manifest_sha256": data.get("val_manifest", {}).get("sha256"),
            "range_sha256": data.get("range_metadata", {}).get("sha256"),
            "parameters": metadata.get("parameters"),
            "objective": {
                "static_loss_weight": configuration.get("static_loss_weight"),
                "delta_loss_weight": configuration.get("delta_loss_weight"),
                "delta2_loss_weight": configuration.get("delta2_loss_weight"),
            },
        }

    @property
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        metadata = dict(super().checkpoint_revision_and_hash)
        metadata["quantization_aware_decoder"] = {
            "checkpoint_sha256": sha256_file(self.decoder_checkpoint),
            "training_metadata": self._public_decoder_metadata(),
            "libritts_trained": True,
            "role": "deterministic mel-domain residual decoder",
        }
        return metadata

    def decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(
            mel.shape[0], 1, mel.shape[-1], device=mel.device, dtype=mel.dtype
        )
        return self.mel_decoder(mel, mask)
