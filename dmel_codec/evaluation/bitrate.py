from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


def nominal_bitrate_kbps(
    token_frame_rate: float, cardinalities: Sequence[int]
) -> float:
    if token_frame_rate < 0:
        raise ValueError("token_frame_rate must be non-negative")
    if any(cardinality < 2 for cardinality in cardinalities):
        raise ValueError("Every discrete stream must have cardinality >= 2")
    return token_frame_rate * sum(math.log2(value) for value in cardinalities) / 1000


@dataclass(frozen=True)
class StreamStatistics:
    cardinality: int
    observations: int
    symbols_used: int
    utilization: float
    entropy_bits: float
    perplexity: float
    dead_symbol_ratio: float


class CodeStatisticsAccumulator:
    """Accumulate exact valid-token histograms without retaining utterance codes."""

    def __init__(self, cardinalities: Sequence[int]) -> None:
        if any(cardinality < 2 for cardinality in cardinalities):
            raise ValueError("Every discrete stream must have cardinality >= 2")
        self.cardinalities = tuple(int(value) for value in cardinalities)
        self.counts = [torch.zeros(value, dtype=torch.long) for value in cardinalities]
        self.total_frames = 0

    def update(self, codes: torch.Tensor, code_lengths: torch.Tensor) -> None:
        if codes.ndim != 3 or codes.shape[1] != len(self.cardinalities):
            raise ValueError("codes must have shape [B,S,F] matching cardinalities")
        if code_lengths.shape != (codes.shape[0],):
            raise ValueError("code_lengths must have shape [B]")
        if torch.any(code_lengths < 0) or torch.any(code_lengths > codes.shape[-1]):
            raise ValueError("code_lengths lie outside the padded code tensor")
        codes = codes.detach().cpu().long()
        code_lengths = code_lengths.detach().cpu().long()
        mask = torch.arange(codes.shape[-1])[None, :] < code_lengths[:, None]
        observations = int(code_lengths.sum().item())
        for stream, cardinality in enumerate(self.cardinalities):
            values = codes[:, stream].masked_select(mask)
            if torch.any(values < 0) or torch.any(values >= cardinality):
                raise ValueError(f"Stream {stream} has an out-of-range symbol")
            self.counts[stream] += torch.bincount(values, minlength=cardinality)
        self.total_frames += observations

    def update_histograms(self, histograms: Sequence[Sequence[int]]) -> None:
        if len(histograms) != len(self.cardinalities):
            raise ValueError("Histogram stream count does not match cardinalities")
        observations: int | None = None
        for stream, (histogram, cardinality) in enumerate(
            zip(histograms, self.cardinalities)
        ):
            values = torch.as_tensor(histogram, dtype=torch.long)
            if values.shape != (cardinality,) or torch.any(values < 0):
                raise ValueError(f"Invalid histogram for stream {stream}")
            stream_observations = int(values.sum().item())
            if observations is None:
                observations = stream_observations
            elif stream_observations != observations:
                raise ValueError("Parallel stream histograms have different token counts")
            self.counts[stream] += values
        self.total_frames += observations or 0

    def stream_statistics(self) -> list[StreamStatistics]:
        result = []
        for cardinality, counts in zip(self.cardinalities, self.counts):
            observations = int(counts.sum().item())
            used = int(torch.count_nonzero(counts).item())
            if observations:
                probabilities = counts[counts > 0].double() / observations
                entropy = max(
                    0.0,
                    float(-(probabilities * probabilities.log2()).sum().item()),
                )
            else:
                entropy = 0.0
            result.append(
                StreamStatistics(
                    cardinality=cardinality,
                    observations=observations,
                    symbols_used=used,
                    utilization=used / cardinality,
                    entropy_bits=entropy,
                    perplexity=2**entropy,
                    dead_symbol_ratio=1 - used / cardinality,
                )
            )
        return result

    def empirical_bitrate_kbps(self, token_frame_rate: float) -> float:
        return token_frame_rate * sum(
            item.entropy_bits for item in self.stream_statistics()
        ) / 1000

    def as_dict(self, token_frame_rate: float) -> dict:
        streams = self.stream_statistics()
        return {
            "valid_frames": self.total_frames,
            "empirical_bitrate_kbps": self.empirical_bitrate_kbps(token_frame_rate),
            "streams": [item.__dict__ for item in streams],
        }
