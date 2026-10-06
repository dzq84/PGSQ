from __future__ import annotations

import math

import hydra
import lightning.pytorch as pl
import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dmel_codec.models.codec_lit_modules import VQGAN
from dmel_codec.train_codec import get_config
from dmel_codec.utils.schedule import get_cosine_schedule_with_warmup_lr_lambda


def _smoke_model(*overrides: str) -> VQGAN:
    config = get_config(["experiment=smoke", *overrides])
    return hydra.utils.instantiate(config.codec.model, _convert_="partial")


def _audio_batch(batch_size: int = 2, length: int = 6123) -> dict:
    generator = torch.Generator().manual_seed(17)
    audios = torch.randn(batch_size, 1, length, generator=generator) * 0.05
    lengths = torch.tensor(
        [length - 1122 * index for index in range(batch_size)], dtype=torch.long
    )
    for index, item_length in enumerate(lengths.tolist()):
        audios[index, :, item_length:] = 0
    return {
        "audios": audios,
        "audio_lengths": lengths,
        "cut_ids": [f"cut-{index}" for index in range(batch_size)],
    }


@pytest.mark.parametrize(
    ("variant", "cardinality", "expected_kbps"),
    (
        ("ours_075", 9, 0.742951),
        ("ours_l", 25, 1.088404),
        ("ours_m", 175, 1.746378),
        ("ours_h", 1000, 2.335731),
    ),
)
def test_formal_bitrate_from_resolved_quantizer(
    variant: str, cardinality: int, expected_kbps: float
) -> None:
    config = get_config([f"codec={variant}", "experiment=full_libritts"])
    quantizer = hydra.utils.instantiate(config.codec.model.quantizer)
    frame_rate = config.codec.sample_rate / (
        config.codec.hop_length * math.prod(quantizer.downsample_factor)
    )
    bitrate = frame_rate * quantizer.num_streams * math.log2(
        quantizer.codebook_cardinality
    ) / 1000

    assert frame_rate == 23.4375
    assert quantizer.num_streams == 10
    assert quantizer.codebook_cardinality == cardinality
    assert bitrate == pytest.approx(expected_kbps, abs=1e-6)


def test_encode_decode_is_deterministic_and_restores_nonmultiple_length() -> None:
    model = _smoke_model().eval()
    batch = _audio_batch()

    with torch.inference_mode():
        codes_a, code_lengths_a = model.encode(
            batch["audios"], batch["audio_lengths"]
        )
        codes_b, code_lengths_b = model.encode(
            batch["audios"], batch["audio_lengths"]
        )
        mel_a = model.decode(
            codes_a,
            code_lengths_a,
            audio_lengths=batch["audio_lengths"],
            cut_ids=batch["cut_ids"],
        )
        mel_b = model.decode(
            codes_b,
            code_lengths_b,
            audio_lengths=batch["audio_lengths"],
            cut_ids=batch["cut_ids"],
        )

    assert torch.equal(codes_a, codes_b)
    assert torch.equal(code_lengths_a, code_lengths_b)
    assert codes_a.shape[1] == 2
    assert mel_a.shape == (2, 8, int(batch["audio_lengths"].max() // 256))
    assert torch.equal(mel_a, mel_b)
    assert torch.isfinite(mel_a).all()
    second_length = int(batch["audio_lengths"][1] // 256)
    assert torch.count_nonzero(mel_a[1, :, second_length:]) == 0


def test_quantizer_forward_codes_use_adapter_shape() -> None:
    model = _smoke_model().eval()
    batch = _audio_batch()
    with torch.inference_mode():
        features, _ = model.encode_unquantized(
            batch["audios"], batch["audio_lengths"]
        )
        forward_codes = model.quantizer(features).codes
        encode_codes = model.quantizer.encode(features)

    assert forward_codes.shape == encode_codes.shape
    assert forward_codes.ndim == 3
    assert torch.equal(forward_codes, encode_codes)


def test_code_usage_statistics_exclude_padding() -> None:
    codes = torch.tensor(
        [
            [[0, 1, 3, 3], [2, 2, 3, 3]],
            [[1, 1, 0, 3], [0, 2, 1, 3]],
        ]
    )
    metrics = VQGAN._code_usage_statistics(
        codes, torch.tensor([2, 3]), cardinality=4
    )

    assert metrics["train/code/stream_00/utilization"] == pytest.approx(0.5)
    assert metrics["train/code/stream_01/utilization"] == pytest.approx(0.75)
    assert metrics["train/code/min_utilization"] == pytest.approx(0.5)
    expected_entropy = -(2 / 5 * math.log2(2 / 5) + 3 / 5 * math.log2(3 / 5))
    assert metrics["train/code/stream_00/entropy_bits"] == pytest.approx(
        expected_entropy
    )


def test_codes_do_not_depend_on_other_batch_items_padding() -> None:
    model = _smoke_model().eval()
    short = _audio_batch(batch_size=1, length=6123)
    mixed = _audio_batch(batch_size=2, length=9217)
    mixed["audios"][0, :, :6123] = short["audios"][0]
    mixed["audios"][0, :, 6123:] = 0
    mixed["audio_lengths"][0] = 6123
    mixed["cut_ids"][0] = short["cut_ids"][0]

    with torch.inference_mode():
        codes_alone, lengths_alone = model.encode(
            short["audios"], short["audio_lengths"]
        )
        codes_mixed, lengths_mixed = model.encode(
            mixed["audios"], mixed["audio_lengths"]
        )

    code_length = int(lengths_alone[0])
    assert lengths_mixed[0] == lengths_alone[0]
    assert torch.equal(codes_alone[0, :, :code_length], codes_mixed[0, :, :code_length])


def test_cut_id_noise_is_repeatable_and_id_dependent() -> None:
    model = _smoke_model("codec.model.decoder_input_policy=cut_id_noise").eval()
    batch = _audio_batch()
    with torch.inference_mode():
        codes, lengths = model.encode(batch["audios"], batch["audio_lengths"])
        first = model.decode(
            codes,
            lengths,
            audio_lengths=batch["audio_lengths"],
            cut_ids=batch["cut_ids"],
        )
        repeated = model.decode(
            codes,
            lengths,
            audio_lengths=batch["audio_lengths"],
            cut_ids=batch["cut_ids"],
        )
        changed = model.decode(
            codes,
            lengths,
            audio_lengths=batch["audio_lengths"],
            cut_ids=["different-0", "different-1"],
        )
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)


def test_fixed_random_mel_grouping_changes_codes_deterministically() -> None:
    natural = _smoke_model().eval()
    shuffled = _smoke_model(
        "codec.model.mel_bin_permutation=[3,0,7,4,1,6,2,5]"
    ).eval()
    shuffled.load_state_dict(natural.state_dict())
    batch = _audio_batch()

    with torch.inference_mode():
        natural_features, _ = natural.encode_unquantized(
            batch["audios"], batch["audio_lengths"]
        )
        shuffled_a, _ = shuffled.encode_unquantized(
            batch["audios"], batch["audio_lengths"]
        )
        shuffled_b, _ = shuffled.encode_unquantized(
            batch["audios"], batch["audio_lengths"]
        )

    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.equal(natural_features, shuffled_a)


def test_mel_grouping_rejects_non_permutation() -> None:
    with pytest.raises(InstantiationException, match="each index exactly once"):
        _smoke_model("codec.model.mel_bin_permutation=[0,0]")


def test_joint_fullband_ablation_emits_ten_residual_streams() -> None:
    config = get_config(["experiment=ablation_joint_fullband", "codec=ours_m"])
    encoder = hydra.utils.instantiate(config.codec.model.encoder).eval()
    quantizer = hydra.utils.instantiate(config.codec.model.quantizer).eval()
    pre_projection = hydra.utils.instantiate(
        config.codec.model.pre_quantized_projection
    ).eval()
    projection = hydra.utils.instantiate(
        config.codec.model.quantized_projection
    ).eval()
    mels = torch.randn(2, 100, 20)

    with torch.inference_mode():
        encoded = encoder(mels)
        joint = pre_projection(encoded)
        result = quantizer(joint)
        reconstructed = quantizer.decode(result.codes, target_length=joint.shape[-1])
        projected = projection(result.z)

    assert encoded.shape == (2, 700, 20)
    assert joint.shape == (2, 70, 20)
    assert result.codes.shape == (2, 10, 5)
    assert reconstructed.shape == joint.shape
    torch.testing.assert_close(reconstructed, result.z)
    assert projected.shape == (2, 700, 20)


def test_checkpoint_records_codec_definition_without_vocoder_weights() -> None:
    model = _smoke_model()
    model.set_checkpoint_metadata(
        {"resolved_config": {"seed": 2026}, "git": {"commit": "abc"}}
    )
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)

    assert not any(key.startswith("vocoder.") for key in checkpoint["state_dict"])
    metadata = checkpoint["codec_metadata"]
    assert metadata["resolved_config"]["seed"] == 2026
    assert metadata["git"]["commit"] == "abc"
    assert metadata["codec_definition"]["quality_side_information"] == "none"
    assert metadata["codec_definition"]["commitment_vq_loss"] is False
    assert not hasattr(model, "quality_projection")


def test_scheduler_does_not_restart_after_training_budget() -> None:
    values = [
        get_cosine_schedule_with_warmup_lr_lambda(
            step,
            num_warmup_steps=100,
            num_training_steps=1000,
            final_lr_ratio=0.05,
        )
        for step in (1000, 1001, 2000)
    ]
    assert values == pytest.approx([0.05, 0.05, 0.05])


def test_gan_step_counts_one_codec_update_and_clears_discriminator_gradients() -> None:
    config = get_config(["experiment=smoke"])
    config.codec.model.discriminator = OmegaConf.create(
        {"_target_": "dmel_codec.models.modules.discriminator.Discriminator"}
    )
    config.codec.model.weight_adv = 0.2
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    loader = DataLoader([_audio_batch(batch_size=1)], batch_size=None)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    assert trainer.global_step == 1
    assert len(trainer.optimizers) == 2
    discriminator_steps = {
        int(state["step"].item())
        for state in trainer.optimizers[1].state.values()
        if "step" in state
    }
    assert discriminator_steps == {1}
    assert all(parameter.grad is None for parameter in model.discriminator.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_gradient_accumulation_flushes_short_epoch_tail() -> None:
    model = _smoke_model("codec.model.accumulate_grad=3")
    loader = DataLoader([_audio_batch(batch_size=1)] * 2, batch_size=None)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=10,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    assert trainer.global_step == 1
    assert model._accumulated_batches == 0
