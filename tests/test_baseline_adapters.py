from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dmel_codec.baselines.dmel import (
    UniformMelQuantizer,
    frontend_definition,
    load_range_metadata,
)
from dmel_codec.evaluation.adapters.base import CodecAdapter
from dmel_codec.evaluation.adapters.dmel import DmelAdaptedDecoderAdapter
from dmel_codec.evaluation.adapters.ours import OursAdapter
from dmel_codec.models.dmel_decoder_lit_module import (
    DmelDecoderLitModule,
    QuantizedMelDecoder,
    load_quantized_mel_decoder,
)
from dmel_codec.train_dmel_decoder import (
    _build_datamodule,
    _scientific_configuration,
    _validate_output,
)
from dmel_codec.train_codec import get_config


class _RateOnlyAdapter(CodecAdapter):
    system_name = "rate-test"
    sample_rate = 24_000
    token_frame_rate = 75.0
    num_parallel_streams = 3
    codebook_cardinalities = (1024, 1024, 1024)
    checkpoint_revision_and_hash = {}

    def code_lengths(self, audio_lengths):
        raise NotImplementedError

    def encode(self, waveform, lengths):
        raise NotImplementedError

    def decode(self, codes, lengths, cut_ids):
        raise NotImplementedError


def test_nominal_bitrate_is_derived_from_all_parallel_streams() -> None:
    assert _RateOnlyAdapter().nominal_bitrate == 2.25


def test_uniform_quantizer_matches_apple_nearest_linear_reference() -> None:
    quantizer = UniformMelQuantizer(-7.0, 2.0, 16)
    values = torch.tensor(
        [[[-8.0, -7.0, -6.7, -6.4, -0.1, 1.9, 2.0, 3.0]]]
    )
    official_step = np.round((2.0 - (-7.0)) / (2**4 - 1), 4)
    official_levels = torch.tensor(
        [-7.0 + index * official_step for index in range(16)]
    ).float()
    clamped = values.clamp(-7.0, 2.0)
    expected = (clamped[..., None] - official_levels).abs().argmin(-1)

    observed = quantizer.encode(values)

    assert torch.equal(observed, expected)
    torch.testing.assert_close(quantizer.decode(observed), official_levels[expected])


def test_range_metadata_rejects_frontend_or_quantizer_drift(tmp_path) -> None:
    path = tmp_path / "range.json"
    metadata = {
        "schema_version": 1,
        "frontend": frontend_definition(),
        "quantizer": {
            "method": "nearest_linear",
            "num_bins": 16,
            "minimum": -11.0,
            "maximum": 2.0,
        },
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert load_range_metadata(path) == metadata

    metadata["frontend"]["hop_length"] = 240
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="frontend"):
        load_range_metadata(path)


def test_quantized_mel_decoder_starts_as_identity() -> None:
    decoder = QuantizedMelDecoder(
        num_mels=4, hidden_channels=8, residual_layers=2, dilation_cycle=2
    )
    mel = torch.randn(2, 4, 7)
    mask = torch.tensor(
        [[[1, 1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 0, 0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoder(mel, mask), mel * mask)


def test_quantized_mel_decoder_checkpoint_roundtrip(tmp_path) -> None:
    range_path = tmp_path / "range.json"
    metadata = {
        "schema_version": 1,
        "frontend": frontend_definition(),
        "quantizer": {
            "method": "nearest_linear",
            "num_bins": 16,
            "minimum": -11.0,
            "maximum": -0.5,
        },
    }
    range_path.write_text(json.dumps(metadata), encoding="utf-8")
    module = DmelDecoderLitModule(
        range_metadata=range_path,
        hidden_channels=8,
        residual_layers=2,
        dilation_cycle=2,
        training_steps=20,
        warmup_steps=2,
    )
    payload = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(payload)
    checkpoint = tmp_path / "decoder.ckpt"
    torch.save(payload, checkpoint)

    loaded, loaded_payload = load_quantized_mel_decoder(checkpoint)
    mel = torch.randn(1, 100, 9)
    mask = torch.ones(1, 1, 9)
    torch.testing.assert_close(loaded(mel, mask), module.decoder(mel, mask))
    assert loaded_payload["dmel_range"]["sha256"]


def test_dmel_decoder_run_dir_must_be_absolute_and_outside_git(tmp_path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        _validate_output(tmp_path.relative_to(tmp_path))
    repository = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside"):
        _validate_output(repository / "runs" / "dmel-decoder")
    assert _validate_output(tmp_path / "run") == (tmp_path / "run").resolve()


def test_dmel_decoder_resume_path_does_not_change_scientific_config(tmp_path) -> None:
    fresh = argparse.Namespace(
        output_dir=tmp_path / "run",
        range_metadata=tmp_path / "range.json",
        learning_rate=1e-3,
        resume_from=None,
    )
    resumed = argparse.Namespace(
        output_dir=tmp_path / "run",
        range_metadata=tmp_path / "range.json",
        learning_rate=1e-3,
        resume_from=tmp_path / "checkpoints" / "last.ckpt",
    )

    assert _scientific_configuration(fresh) == _scientific_configuration(resumed)


def test_dmel_decoder_uses_cut_id_deterministic_training_crops(tmp_path) -> None:
    args = argparse.Namespace(
        seed=2026,
        train_batch_seconds=60.0,
        val_batch_seconds=30.0,
        train_workers=4,
        val_workers=2,
    )
    datamodule = _build_datamodule(
        args,
        train_manifest=tmp_path / "train.jsonl.gz",
        val_manifest=tmp_path / "dev.jsonl.gz",
    )

    assert datamodule.hparams.seed == 2026
    assert datamodule.hparams.train_crop_seed == 2026
    assert datamodule.hparams.train_max_audio_seconds == 3.0
    assert datamodule.hparams.train_random_crop is True


def test_dmel_decoder_public_metadata_does_not_expose_paths() -> None:
    adapter = object.__new__(DmelAdaptedDecoderAdapter)
    adapter.decoder_metadata = {
        "config_sha256": "config-hash",
        "git": {"commit": "commit", "dirty": False},
        "data": {
            "train_manifest": {"path": "/private/train", "sha256": "train-hash"},
            "val_manifest": {"path": "/private/dev", "sha256": "dev-hash"},
            "range_metadata": {"path": "/private/range", "sha256": "range-hash"},
        },
        "configuration": {
            "output_dir": "/private/run",
            "static_loss_weight": 1.0,
            "delta_loss_weight": 0.5,
            "delta2_loss_weight": 0.25,
        },
        "parameters": 123,
    }
    public = adapter._public_decoder_metadata()
    assert "/private" not in json.dumps(public)
    assert public["train_manifest_sha256"] == "train-hash"


def test_ours_adapter_uses_checkpoint_definition_and_lengths(
    tmp_path, monkeypatch
) -> None:
    config = get_config(["experiment=smoke", "codec=ours_m"])
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    model.set_checkpoint_metadata(
        {
            "resolved_config": OmegaConf.to_container(config, resolve=True),
            "config_sha256": "config-hash",
            "git": {"commit": "git-sha"},
            "data": {"manifests": {}},
        }
    )
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "smoke.ckpt"
    torch.save(checkpoint, checkpoint_path)

    adapter = OursAdapter(checkpoint=checkpoint_path)
    encode_calls = []
    original_encode = adapter.model.encode

    def recorded_encode(waveform, lengths):
        encode_calls.append((tuple(waveform.shape), lengths.tolist()))
        return original_encode(waveform, lengths)

    monkeypatch.setattr(adapter.model, "encode", recorded_encode)
    waveform = torch.randn(2, 1, 6123) * 0.01
    lengths = torch.tensor([6123, 5000])
    waveform[1, :, 5000:] = 0
    codes, code_lengths = adapter.encode(waveform, lengths)

    assert codes.shape[:2] == (2, 2)
    assert torch.equal(code_lengths, torch.div(lengths // 256, 4, rounding_mode="floor"))
    assert adapter.token_frame_rate == 23.4375
    assert adapter.nominal_bitrate == pytest.approx(
        23.4375 * 2 * math.log2(9) / 1000
    )
    assert adapter.checkpoint_revision_and_hash["git_commit"] == "git-sha"
    assert adapter.checkpoint_revision_and_hash["encoding_policy"] == {
        "name": "utterance-independent-deterministic-v1",
        "batching": "one-valid-waveform-at-a-time",
        "cuda_deterministic_algorithms": True,
        "cuda_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    assert encode_calls == [((1, 1, 6123), [6123]), ((1, 1, 5000), [5000])]


def test_ours_adapter_rejects_unpinned_cublas_workspace(
    tmp_path, monkeypatch
) -> None:
    config = get_config(["experiment=smoke", "codec=ours_m"])
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    model.set_checkpoint_metadata(
        {"resolved_config": OmegaConf.to_container(config, resolve=True)}
    )
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "ours.ckpt"
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(
        ValueError,
        match="CUBLAS_WORKSPACE_CONFIG=:4096:8",
    ):
        OursAdapter(checkpoint=checkpoint_path, device="cuda")


def test_ours_adapter_rejects_late_cublas_workspace_setup(
    tmp_path, monkeypatch
) -> None:
    config = get_config(["experiment=smoke", "codec=ours_m"])
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    model.set_checkpoint_metadata(
        {"resolved_config": OmegaConf.to_container(config, resolve=True)}
    )
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "ours.ckpt"
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(
        ValueError,
        match="must be set before CUDA initialization",
    ):
        OursAdapter(checkpoint=checkpoint_path, device="cuda")
