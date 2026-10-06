from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Reconstruction:
    waveform: torch.Tensor
    codes: torch.Tensor
    code_lengths: torch.Tensor


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CodecAdapter(ABC):
    """Common, length-aware interface for all reconstruction systems."""

    system_name: str
    sample_rate: int
    token_frame_rate: float
    num_parallel_streams: int
    codebook_cardinalities: Sequence[int]

    @property
    def nominal_bitrate(self) -> float:
        return self.token_frame_rate * sum(
            math.log2(cardinality) for cardinality in self.codebook_cardinalities
        ) / 1000.0

    @property
    @abstractmethod
    def checkpoint_revision_and_hash(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def code_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def decode(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        cut_ids: list[str],
    ) -> torch.Tensor:
        pass

    def reconstruct(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
        cut_ids: list[str],
    ) -> Reconstruction:
        self.validate_waveforms(waveform, lengths, cut_ids)
        codes, code_lengths = self.encode(waveform, lengths)
        reconstructed = self.decode(codes, lengths, cut_ids)
        self.validate_reconstruction(reconstructed, waveform, lengths)
        return Reconstruction(reconstructed, codes, code_lengths)

    def metadata(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "sample_rate": self.sample_rate,
            "token_frame_rate": self.token_frame_rate,
            "num_parallel_streams": self.num_parallel_streams,
            "codebook_cardinalities": list(self.codebook_cardinalities),
            "nominal_bitrate_kbps": self.nominal_bitrate,
            "checkpoint_revision_and_hash": self.checkpoint_revision_and_hash,
        }

    @staticmethod
    def validate_waveforms(
        waveform: torch.Tensor,
        lengths: torch.Tensor,
        cut_ids: list[str] | None = None,
    ) -> None:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected waveform [B,1,T], got {tuple(waveform.shape)}")
        if lengths.ndim != 1 or lengths.shape[0] != waveform.shape[0]:
            raise ValueError(f"Expected lengths [B], got {tuple(lengths.shape)}")
        if torch.any(lengths <= 0) or torch.any(lengths > waveform.shape[-1]):
            raise ValueError("Lengths must be positive and within the padded waveform")
        if cut_ids is not None and len(cut_ids) != waveform.shape[0]:
            raise ValueError("cut_ids must contain one ID per waveform")
        if not torch.isfinite(waveform).all():
            raise ValueError("Waveform contains NaN or Inf")

    @staticmethod
    def validate_codes(
        codes: torch.Tensor,
        *,
        batch_size: int,
        num_streams: int,
        cardinalities: Sequence[int],
    ) -> None:
        if codes.ndim != 3 or codes.shape[:2] != (batch_size, num_streams):
            raise ValueError(
                f"Expected codes [{batch_size},{num_streams},F], got {tuple(codes.shape)}"
            )
        if codes.dtype != torch.long:
            raise ValueError(f"Codes must be torch.long, got {codes.dtype}")
        for stream, cardinality in enumerate(cardinalities):
            values = codes[:, stream]
            if torch.any(values < 0) or torch.any(values >= cardinality):
                raise ValueError(
                    f"Stream {stream} contains an index outside [0,{cardinality})"
                )

    @staticmethod
    def validate_reconstruction(
        reconstructed: torch.Tensor,
        reference: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if reconstructed.shape != reference.shape:
            raise RuntimeError(
                f"Reconstruction shape {tuple(reconstructed.shape)} does not match "
                f"reference {tuple(reference.shape)}"
            )
        if not torch.isfinite(reconstructed).all():
            raise RuntimeError("Reconstruction contains NaN or Inf")
        lengths = lengths.to(reconstructed.device)
        time = torch.arange(reference.shape[-1], device=reconstructed.device)[None, :]
        padding = time >= lengths[:, None]
        if torch.count_nonzero(reconstructed[:, 0].masked_select(padding)):
            raise RuntimeError("Reconstruction padding must be exactly zero")

    @staticmethod
    def center_resize(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        difference = target_length - waveform.shape[-1]
        if difference > 0:
            left = difference // 2
            return F.pad(waveform, (left, difference - left))
        if difference < 0:
            crop = -difference
            left = crop // 2
            return waveform[..., left : left + target_length]
        return waveform

    @staticmethod
    def pad_waveforms(
        waveforms: list[torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        target = int(lengths.max().item())
        padded = []
        for waveform, length in zip(waveforms, lengths.tolist()):
            if waveform.shape != (1, 1, length):
                raise RuntimeError(
                    f"Decoded item has shape {tuple(waveform.shape)}, expected (1,1,{length})"
                )
            padded.append(F.pad(waveform, (0, target - length)))
        return torch.cat(padded, dim=0)
