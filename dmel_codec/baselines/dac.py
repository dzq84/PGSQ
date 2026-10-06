from __future__ import annotations

from importlib import metadata
from pathlib import Path

import torch


DAC_SOURCE = "https://github.com/descriptinc/descript-audio-codec"
DAC_PACKAGE_VERSION = "1.0.0"
DAC_PACKAGE_COMMIT = "c7cfc5d2647e26471dc394f95846a0830e7bec34"
DAC_WEIGHT_TAG = "0.0.4"
DAC_WEIGHT_URL = (
    "https://github.com/descriptinc/descript-audio-codec/releases/download/"
    "0.0.4/weights_24khz.pth"
)
DAC_LICENSE = "MIT"


def load_dac_model(checkpoint: str | Path, device: torch.device):
    try:
        installed = metadata.version("descript-audio-codec")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "DAC support is not installed; install the project 'dac' extra"
        ) from exc
    if installed != DAC_PACKAGE_VERSION:
        raise RuntimeError(
            f"Expected descript-audio-codec {DAC_PACKAGE_VERSION}, got {installed}"
        )
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import dac

    model = dac.DAC.load(checkpoint).to(device).eval()
    observed = {
        "sample_rate": int(model.sample_rate),
        "hop_length": int(model.hop_length),
        "n_codebooks": int(model.n_codebooks),
        "codebook_size": int(model.codebook_size),
    }
    expected = {
        "sample_rate": 24_000,
        "hop_length": 320,
        "n_codebooks": 32,
        "codebook_size": 1024,
    }
    if observed != expected:
        raise ValueError(f"Unexpected DAC 24 kHz checkpoint definition: {observed}")
    return model
