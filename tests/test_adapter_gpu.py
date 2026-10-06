from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

from dmel_codec.evaluation.adapters.dac import DACAdapter
from dmel_codec.evaluation.adapters.dmel import DmelSameFrontendAdapter
from dmel_codec.evaluation.adapters.ours import OursAdapter


pytestmark = pytest.mark.gpu


def _required_environment(*names: str) -> list[Path]:
    values = [os.environ.get(name) for name in names]
    if not all(values):
        pytest.skip(f"Set {', '.join(names)} to run external-checkpoint adapter test")
    paths = [Path(value) for value in values]
    if not all(path.is_file() for path in paths):
        pytest.skip(f"External adapter assets are absent: {paths}")
    return paths


def _batch() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    generator = torch.Generator().manual_seed(2027)
    waveform = torch.randn(2, 1, 50_417, generator=generator) * 0.03
    lengths = torch.tensor([50_417, 48_013])
    waveform[1, :, lengths[1] :] = 0
    return waveform, lengths, ["adapter-cut-a", "adapter-cut-b"]


def _assert_roundtrip(adapter) -> None:
    waveform, lengths, cut_ids = _batch()
    result = adapter.reconstruct(waveform, lengths, cut_ids)
    assert result.waveform.shape == waveform.shape
    assert result.codes.shape[:2] == (
        waveform.shape[0],
        adapter.num_parallel_streams,
    )
    assert torch.isfinite(result.waveform).all()
    assert torch.count_nonzero(result.waveform[1, :, lengths[1] :]) == 0


def test_dmel_same_frontend_gpu_roundtrip() -> None:
    range_path, bigvgan_checkpoint, bigvgan_config = _required_environment(
        "DMEL_RANGE", "BIGVGAN_CKPT", "BIGVGAN_CONFIG"
    )
    adapter = DmelSameFrontendAdapter(
        range_metadata=range_path,
        bigvgan_checkpoint=bigvgan_checkpoint,
        bigvgan_config=bigvgan_config,
        device="cuda",
    )
    _assert_roundtrip(adapter)
    assert adapter.nominal_bitrate == 37.5
    assert adapter.token_frame_rate == 93.75


def test_dac_gpu_roundtrip_and_operating_points() -> None:
    (checkpoint,) = _required_environment("DAC_CKPT")
    expected = {1: 0.75, 2: 1.50, 3: 2.25, 4: 3.00}
    for quantizers, bitrate in expected.items():
        adapter = DACAdapter(
            checkpoint=checkpoint,
            n_quantizers=quantizers,
            device="cuda",
        )
        _assert_roundtrip(adapter)
        assert adapter.nominal_bitrate == bitrate
        assert adapter.token_frame_rate == 75.0
        assert adapter.model.hop_length == 320
        assert adapter.model.n_codebooks == 32
        del adapter
        gc.collect()
        torch.cuda.empty_cache()


def test_ours_gpu_adapter_roundtrip() -> None:
    checkpoint, bigvgan_checkpoint, bigvgan_config = _required_environment(
        "OURS_CKPT", "BIGVGAN_CKPT", "BIGVGAN_CONFIG"
    )
    adapter = OursAdapter(
        checkpoint=checkpoint,
        bigvgan_checkpoint=bigvgan_checkpoint,
        bigvgan_config=bigvgan_config,
        device="cuda",
    )
    _assert_roundtrip(adapter)
    assert adapter.nominal_bitrate == pytest.approx(1.7463776043)
    assert adapter.token_frame_rate == 23.4375

    waveform, lengths, _ = _batch()
    batched_codes, batched_lengths = adapter.encode(waveform, lengths)
    for index, length in enumerate(lengths.tolist()):
        single_codes, single_lengths = adapter.encode(
            waveform[index : index + 1, :, :length], lengths[index : index + 1]
        )
        valid = int(single_lengths[0])
        assert batched_lengths[index] == single_lengths[0]
        assert torch.equal(
            batched_codes[index, :, :valid], single_codes[0, :, :valid]
        )
