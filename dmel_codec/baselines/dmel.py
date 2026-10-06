from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from lhotse import CutSet
from torch.nn import functional as F

from dmel_codec.utils.spectrogram import (
    LogMelSpectrogram,
    length_aware_spectrogram,
)


APPLE_DMEL_SOURCE = "https://github.com/apple/dmel"
APPLE_DMEL_REVISION = "134d9dea26099eca712a8cfd1e36329915df7442"
APPLE_DMEL_LICENSE = "Apple Sample Code License"
RANGE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frontend_definition() -> dict[str, Any]:
    return {
        "sample_rate": 24_000,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 100,
        "f_min": 0,
        "f_max": 12_000,
        "log_base": "natural",
        "center": False,
        "pre_stft_reflect_padding": 384,
    }


def make_frontend() -> LogMelSpectrogram:
    definition = frontend_definition()
    return LogMelSpectrogram(
        sample_rate=definition["sample_rate"],
        n_fft=definition["n_fft"],
        win_length=definition["win_length"],
        hop_length=definition["hop_length"],
        n_mels=definition["n_mels"],
        f_min=definition["f_min"],
        f_max=definition["f_max"],
        center=definition["center"],
    )


class UniformMelQuantizer(torch.nn.Module):
    """Apple dMel's nearest linear scalar quantizer without special LM tokens."""

    def __init__(self, minimum: float, maximum: float, num_bins: int = 16) -> None:
        super().__init__()
        if not minimum < maximum:
            raise ValueError("minimum must be strictly less than maximum")
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        self.quantize_minimum = float(minimum)
        self.quantize_maximum = float(maximum)
        step = round((self.quantize_maximum - self.quantize_minimum) / (num_bins - 1), 4)
        levels = torch.tensor(
            [self.quantize_minimum + index * step for index in range(num_bins)],
            dtype=torch.float32,
        )
        self.register_buffer("levels", levels)

    @property
    def minimum(self) -> float:
        return self.quantize_minimum

    @property
    def maximum(self) -> float:
        return self.quantize_maximum

    @property
    def num_bins(self) -> int:
        return int(self.levels.numel())

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        clamped = features.clamp(self.minimum, self.maximum)
        return (clamped[..., None] - self.levels).abs().argmin(-1).long()

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            raise ValueError("dMel indices must use torch.long")
        if torch.any(indices < 0) or torch.any(indices >= self.num_bins):
            raise ValueError("dMel index outside quantizer range")
        return self.levels[indices]


def load_range_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != RANGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dMel range schema in {path}")
    if metadata.get("frontend") != frontend_definition():
        raise ValueError("dMel range frontend does not match Ours frontend")
    quantizer = metadata.get("quantizer", {})
    if quantizer.get("num_bins") != 16 or quantizer.get("method") != "nearest_linear":
        raise ValueError("Expected a 16-bin nearest-linear dMel range")
    if not quantizer.get("minimum") < quantizer.get("maximum"):
        raise ValueError("Invalid dMel quantization range")
    return metadata


def _distributed_context(device_argument: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed range estimation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size
    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA range estimation requested but CUDA is unavailable")
    return device, rank, world_size


def _update_range(
    frontend: LogMelSpectrogram,
    audios: list[torch.Tensor],
    lengths: list[int],
    device: torch.device,
) -> tuple[float, float, int]:
    max_length = max(lengths)
    batch = torch.stack(
        [F.pad(audio, (0, max_length - audio.shape[-1])) for audio in audios]
    ).to(device)
    length_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
    features = length_aware_spectrogram(frontend, batch, length_tensor)
    frame_lengths = torch.div(length_tensor, 256, rounding_mode="floor")
    mask = (
        torch.arange(features.shape[-1], device=device)[None, :]
        < frame_lengths[:, None]
    )
    values = features.transpose(1, 2)[mask]
    return float(values.min()), float(values.max()), int(values.numel())


@torch.inference_mode()
def estimate_range(
    manifest: Path,
    output: Path,
    *,
    manifest_summary: Path | None,
    batch_seconds: float,
    device_argument: str,
) -> dict[str, Any] | None:
    device, rank, world_size = _distributed_context(device_argument)
    frontend = make_frontend().to(device).eval()
    local_minimum = float("inf")
    local_maximum = float("-inf")
    local_values = 0
    local_cuts = 0
    local_samples = 0
    audios: list[torch.Tensor] = []
    lengths: list[int] = []
    duration = 0.0

    def flush() -> None:
        nonlocal local_minimum, local_maximum, local_values, audios, lengths, duration
        if not audios:
            return
        minimum, maximum, count = _update_range(frontend, audios, lengths, device)
        local_minimum = min(local_minimum, minimum)
        local_maximum = max(local_maximum, maximum)
        local_values += count
        audios = []
        lengths = []
        duration = 0.0

    for index, cut in enumerate(CutSet.from_file(manifest)):
        if index % world_size != rank:
            continue
        audio = torch.from_numpy(cut.load_audio()).float()
        if audio.shape[0] != 1:
            raise ValueError(f"Expected mono audio for {cut.id}")
        if audios and duration + cut.duration > batch_seconds:
            flush()
        audios.append(audio)
        lengths.append(audio.shape[-1])
        duration += cut.duration
        local_cuts += 1
        local_samples += audio.shape[-1]
    flush()

    minimum = torch.tensor(local_minimum, device=device, dtype=torch.float64)
    maximum = torch.tensor(local_maximum, device=device, dtype=torch.float64)
    counts = torch.tensor(
        [local_values, local_cuts, local_samples], device=device, dtype=torch.long
    )
    if world_size > 1:
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return None

    summary = (
        {
            "path": str(manifest_summary.resolve()),
            "sha256": sha256_file(manifest_summary),
        }
        if manifest_summary is not None
        else None
    )
    metadata = {
        "schema_version": RANGE_SCHEMA_VERSION,
        "system": "dMel-same-front-end (adapted)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "manifest_summary": summary,
            "train_splits_only": [
                "train-clean-100",
                "train-clean-360",
                "train-other-500",
            ],
        },
        "frontend": frontend_definition(),
        "quantizer": {
            "method": "nearest_linear",
            "num_bins": 16,
            "minimum": float(minimum),
            "maximum": float(maximum),
            "range_estimator": "exact_global_min_max",
        },
        "statistics": {
            "cuts": int(counts[1]),
            "waveform_samples": int(counts[2]),
            "mel_values": int(counts[0]),
            "world_size": world_size,
            "batch_seconds_per_rank": batch_seconds,
        },
        "apple_tokenizer_reference": {
            "source": APPLE_DMEL_SOURCE,
            "revision": APPLE_DMEL_REVISION,
            "license": APPLE_DMEL_LICENSE,
            "compatibility": "same 16-bin nearest-linear scalar discretization; adapted frontend/range",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    if world_size > 1:
        dist.destroy_process_group()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate adapted dMel's exact range from training manifests"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path)
    parser.add_argument("--batch-seconds", type=float, default=60.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.manifest_summary is not None and not args.manifest_summary.is_file():
        raise FileNotFoundError(args.manifest_summary)
    if args.batch_seconds <= 0:
        raise ValueError("--batch-seconds must be positive")
    metadata = estimate_range(
        args.manifest,
        args.output,
        manifest_summary=args.manifest_summary,
        batch_seconds=args.batch_seconds,
        device_argument=args.device,
    )
    if metadata is not None:
        print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
