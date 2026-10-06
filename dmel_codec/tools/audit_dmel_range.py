from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from lhotse import CutSet
from lhotse.cut import Cut

from dmel_codec.baselines.dmel import (
    load_range_metadata,
    make_frontend,
    sha256_file,
)


def _cut_rank(cut_id: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"dmel-range-audit-v1:{seed}:{cut_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def select_cuts(cuts: Iterable[Cut], sample_cuts: int, seed: int) -> list[Cut]:
    if sample_cuts <= 0:
        raise ValueError("sample_cuts must be positive")
    selected = heapq.nsmallest(
        sample_cuts,
        cuts,
        key=lambda cut: _cut_rank(cut.id, seed),
    )
    if len(selected) != sample_cuts:
        raise ValueError(
            f"Requested {sample_cuts} cuts but manifest contains only {len(selected)}"
        )
    return sorted(selected, key=lambda cut: cut.id)


def histogram_quantiles(
    histogram: torch.Tensor,
    minimum: float,
    maximum: float,
    probabilities: Sequence[float],
) -> dict[str, float]:
    if histogram.ndim != 1 or histogram.numel() < 2:
        raise ValueError("histogram must contain at least two bins")
    if torch.any(histogram < 0) or int(histogram.sum()) <= 0:
        raise ValueError("histogram counts must be non-negative and non-empty")
    if not minimum < maximum:
        raise ValueError("minimum must be less than maximum")
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("quantile probabilities must be in [0, 1]")

    cumulative = histogram.to(torch.long).cumsum(0)
    observations = int(cumulative[-1])
    levels = torch.linspace(minimum, maximum, histogram.numel(), dtype=torch.float64)
    result = {}
    for probability in probabilities:
        target = min(
            observations,
            max(1, int(round(float(probability) * observations))),
        )
        index = int(torch.searchsorted(cumulative, torch.tensor(target)).item())
        result[f"{probability:g}"] = float(levels[min(index, levels.numel() - 1)])
    return result


@torch.inference_mode()
def audit_range(
    manifest: Path,
    range_metadata: Path,
    output: Path,
    *,
    sample_cuts: int,
    histogram_bins: int,
    seed: int,
    device: torch.device,
    probabilities: Sequence[float],
) -> dict:
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least two")
    metadata = load_range_metadata(range_metadata)
    quantizer = metadata["quantizer"]
    minimum = float(quantizer["minimum"])
    maximum = float(quantizer["maximum"])
    selected = select_cuts(CutSet.from_file(manifest), sample_cuts, seed)
    cut_id_sha256 = hashlib.sha256(
        "\n".join(cut.id for cut in selected).encode("utf-8")
    ).hexdigest()

    frontend = make_frontend().to(device).eval()
    histogram = torch.zeros(histogram_bins, dtype=torch.long)
    mel_values = 0
    below_minimum = 0
    above_maximum = 0
    for cut in selected:
        waveform = torch.from_numpy(cut.load_audio()).float().to(device)
        features = frontend(waveform.unsqueeze(0)).flatten()
        below_minimum += int((features < minimum).sum())
        above_maximum += int((features > maximum).sum())
        histogram += (
            torch.histc(
                features.clamp(minimum, maximum),
                bins=histogram_bins,
                min=minimum,
                max=maximum,
            )
            .round()
            .long()
            .cpu()
        )
        mel_values += features.numel()

    result = {
        "schema_version": 1,
        "selection": {
            "method": "bottom sha256 rank over cut ids",
            "namespace": "dmel-range-audit-v1",
            "seed": seed,
            "sample_cuts": sample_cuts,
            "cut_id_sha256": cut_id_sha256,
        },
        "source": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "range_metadata": str(range_metadata.resolve()),
            "range_metadata_sha256": sha256_file(range_metadata),
        },
        "frontend": metadata["frontend"],
        "exact_range": [minimum, maximum],
        "histogram_bins": histogram_bins,
        "mel_values": mel_values,
        "below_minimum_fraction": below_minimum / mel_values,
        "above_maximum_fraction": above_maximum / mel_values,
        "quantiles": histogram_quantiles(
            histogram, minimum, maximum, probabilities
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit adapted dMel range on a deterministic train-only cut sample"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--range-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-cuts", type=int, default=4096)
    parser.add_argument("--histogram-bins", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=(0.0001, 0.001, 0.005, 0.01, 0.99, 0.995, 0.999, 0.9999),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    for path in (args.manifest, args.range_metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = audit_range(
        args.manifest,
        args.range_metadata,
        args.output,
        sample_cuts=args.sample_cuts,
        histogram_bins=args.histogram_bins,
        seed=args.seed,
        device=torch.device(args.device),
        probabilities=args.quantiles,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
