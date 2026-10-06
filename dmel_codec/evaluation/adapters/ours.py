from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F

from dmel_codec.evaluation.adapters.base import CodecAdapter, sha256_file


class OursAdapter(CodecAdapter):
    ENCODING_POLICY = {
        "name": "utterance-independent-deterministic-v1",
        "batching": "one-valid-waveform-at-a-time",
        "cuda_deterministic_algorithms": True,
        "cuda_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        bigvgan_checkpoint: str | Path | None = None,
        bigvgan_config: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("codec_metadata", {})
        resolved = metadata.get("resolved_config")
        if not resolved:
            raise ValueError("Ours checkpoint is missing embedded resolved_config")
        config = OmegaConf.create(resolved)
        if config.codec.model.get("vocoder"):
            with open_dict(config):
                if bigvgan_checkpoint is not None:
                    config.codec.model.vocoder.ckpt_path = str(bigvgan_checkpoint)
                if bigvgan_config is not None:
                    config.codec.model.vocoder.h_path = str(bigvgan_config)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace is None:
                if torch.cuda.is_initialized():
                    raise ValueError(
                        "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before "
                        "CUDA initialization for deterministic Ours encoding"
                    )
                workspace = self.ENCODING_POLICY["cublas_workspace_config"]
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
            if workspace != self.ENCODING_POLICY["cublas_workspace_config"]:
                raise ValueError(
                    "Deterministic Ours encoding requires "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8"
                )
        self.model = hydra.utils.instantiate(
            config.codec.model, _convert_="partial"
        )
        result = self.model.load_state_dict(payload["state_dict"], strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected Ours checkpoint keys: {result.unexpected_keys}")
        if any(not key.startswith("vocoder.") for key in result.missing_keys):
            raise ValueError(
                "Ours checkpoint has missing non-vocoder keys: "
                f"{[key for key in result.missing_keys if not key.startswith('vocoder.')] }"
            )
        self.model.on_load_checkpoint(payload)
        del payload
        self.model = self.model.to(self.device).eval()
        self.config = config
        self.embedded_metadata = metadata
        codec_name = str(config.codec.name)
        suffix = codec_name.removeprefix("ours_").upper()
        self.system_name = str(config.codec.get("system_name", f"Ours-{suffix}"))
        self.sample_rate = int(self.model.sampling_rate)
        self.token_frame_rate = float(self.model.token_frame_rate)
        self.num_parallel_streams = int(self.model.quantizer.num_streams)
        self.codebook_cardinalities = (
            int(self.model.quantizer.codebook_cardinality),
        ) * self.num_parallel_streams
        expected = metadata.get("codec_definition", {})
        observed = self.model.codec_definition()
        mismatch = {
            key: (value, observed.get(key))
            for key, value in expected.items()
            if observed.get(key) != value
        }
        if mismatch:
            raise ValueError(
                f"Configured Ours model does not match checkpoint definition: {mismatch}"
            )

    @property
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        vocoder = self.config.codec.model.get("vocoder")
        result = {
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "git_commit": self.embedded_metadata.get("git", {}).get("commit"),
            "config_sha256": self.embedded_metadata.get("config_sha256"),
            "manifest": self.embedded_metadata.get("data"),
            "trained_on_libritts": True,
            "encoding_policy": dict(self.ENCODING_POLICY),
        }
        if vocoder:
            checkpoint = Path(str(vocoder.ckpt_path))
            config = Path(str(vocoder.h_path))
            result["bigvgan"] = {
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": sha256_file(config),
                "pretrained_external": True,
            }
            refinement = self.model.codec_definition().get("refinement")
            if refinement:
                result["bigvgan"].update(
                    {
                        "adapted_on_libritts": bool(
                            refinement["vocoder_trainable_parameters"]
                        ),
                        "adapted_parameters": int(
                            refinement["vocoder_trainable_parameters"]
                        ),
                        "adaptation_mode": refinement["mode"],
                    }
                )
        return result

    def code_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        mel_lengths = torch.div(
            audio_lengths,
            self.model.encode_mel_transform.hop_length,
            rounding_mode="floor",
        )
        return torch.div(
            mel_lengths,
            math.prod(self.model.quantizer.downsample_factor),
            rounding_mode="floor",
        )

    @contextmanager
    def _deterministic_encoding(self):
        if self.device.type != "cuda":
            yield
            return

        algorithms_enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=False,
            ):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.use_deterministic_algorithms(
                algorithms_enabled, warn_only=warn_only
            )

    @torch.inference_mode()
    def encode(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_waveforms(waveform, lengths)
        length_values = [int(value) for value in lengths.detach().cpu().tolist()]
        waveform = waveform.to(self.device)
        lengths = lengths.to(self.device)
        sequences = []
        item_lengths = []
        with self._deterministic_encoding():
            for index, audio_length in enumerate(length_values):
                item_codes, observed = self.model.encode(
                    waveform[index : index + 1, :, :audio_length],
                    lengths[index : index + 1],
                )
                frame_length = int(observed[0].item())
                sequences.append(item_codes.long()[..., :frame_length])
                item_lengths.append(observed)
        observed_lengths = torch.cat(item_lengths, dim=0)
        maximum = int(observed_lengths.max().item())
        codes = torch.cat(
            [F.pad(sequence, (0, maximum - sequence.shape[-1])) for sequence in sequences],
            dim=0,
        )
        expected_lengths = self.code_lengths(lengths)
        if not torch.equal(observed_lengths, expected_lengths):
            raise RuntimeError("Ours observed code lengths disagree with config")
        self.validate_codes(
            codes,
            batch_size=waveform.shape[0],
            num_streams=self.num_parallel_streams,
            cardinalities=self.codebook_cardinalities,
        )
        return codes, observed_lengths

    @torch.inference_mode()
    def decode(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        cut_ids: list[str],
    ) -> torch.Tensor:
        lengths = lengths.to(self.device)
        codes = codes.to(self.device)
        if len(cut_ids) != lengths.shape[0]:
            raise ValueError("cut_ids must contain one ID per code sequence")
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
            waveform, _ = self.model.decode(
                codes[index : index + 1, :, :frame_length],
                frame_lengths[index : index + 1],
                return_audios=True,
                audio_lengths=lengths[index : index + 1],
                cut_ids=[cut_ids[index]],
            )
            if waveform.shape[-1] != audio_length:
                raise RuntimeError("Ours decoder did not restore source length")
            waveforms.append(waveform)
        return self.pad_waveforms(waveforms, lengths)
