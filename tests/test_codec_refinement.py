from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import hydra
import lightning.pytorch as pl
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dmel_codec.models.codec_refinement_lit_module import (
    CodecRefinement,
    MultiResolutionSTFTLoss,
)
from dmel_codec.models.modules.waveform_discriminator import WaveformDiscriminator
from dmel_codec.train_codec import get_config


class TinyVocoder(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.ckpt_path = str(checkpoint)
        self.num_kernels = 1
        self.conv_pre = torch.nn.Conv1d(8, 8, 3, padding=1)
        self.ups = torch.nn.ModuleList(
            [torch.nn.Conv1d(8, 8, 3, padding=1) for _ in range(3)]
        )
        self.resblocks = torch.nn.ModuleList(
            [torch.nn.Conv1d(8, 8, 3, padding=1) for _ in range(3)]
        )
        self.activation_post = torch.nn.PReLU(8)
        self.conv_post = torch.nn.Conv1d(8, 1, 3, padding=1)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_pre(mels)
        for upsample, residual in zip(self.ups, self.resblocks):
            hidden = torch.tanh(upsample(hidden) + residual(hidden))
        hidden = self.activation_post(hidden)
        return torch.tanh(
            self.conv_post(F.interpolate(hidden, scale_factor=256, mode="linear"))
        )


def _batch(length: int = 4096) -> dict:
    generator = torch.Generator().manual_seed(31)
    return {
        "audios": torch.randn(1, 1, length, generator=generator) * 0.03,
        "audio_lengths": torch.tensor([length]),
        "cut_ids": ["refinement-cut"],
    }


def _model(tmp_path: Path, mode: str) -> CodecRefinement:
    config = get_config(["experiment=smoke", "codec=ours_m"])
    components = config.codec.model
    checkpoint = tmp_path / "tiny-vocoder.pt"
    vocoder = TinyVocoder(checkpoint)
    torch.save({"generator": vocoder.state_dict()}, checkpoint)
    waveform_discriminator = None
    if mode in {
        "waveform_joint_vocoder_tail_gan",
        "waveform_full_codec_frozen_vocoder_gan",
        "waveform_vocoder_tail_only_gan",
        "waveform_joint_full_codec_gan",
        "waveform_vocoder_full_only_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    }:
        waveform_discriminator = WaveformDiscriminator(
            periods=(2,),
            resolutions=((256, 64, 256),),
            channel_multiplier=0.0625,
        )
    return CodecRefinement(
        encoder=hydra.utils.instantiate(components.encoder),
        quantizer=hydra.utils.instantiate(components.quantizer),
        decoder=hydra.utils.instantiate(components.decoder),
        discriminator=None,
        vocoder=TinyVocoder(checkpoint),
        encode_mel_transform=hydra.utils.instantiate(components.encode_mel_transform),
        gt_mel_transform=hydra.utils.instantiate(components.gt_mel_transform),
        optimizer=hydra.utils.instantiate(components.optimizer),
        lr_scheduler=hydra.utils.instantiate(components.lr_scheduler),
        sampling_rate=24_000,
        freeze_encoder=mode
        not in {
            "waveform_full_codec_frozen_vocoder_gan",
            "waveform_joint_full_codec_gan",
            "waveform_joint_full_codec_full_vocoder_gan",
        },
        dmel_groups=2,
        weight_adv=0.0,
        weight_mel=1.0,
        refinement_mode=mode,
        waveform_loss_weight=0.1,
        validation_waveform_loss_weight=0.1,
        stft_resolutions=((256, 64, 256), (512, 128, 512)),
        vocoder_tail_stages=2,
        vocoder_lr_scale=0.1,
        waveform_discriminator=waveform_discriminator,
        discriminator_lr_scale=1.0,
        gan_adversarial_weight=0.1,
        gan_feature_weight=0.1,
    )


def test_multi_resolution_stft_loss_is_finite_and_differentiable() -> None:
    target = torch.randn(2, 1, 1024)
    generated = (target * 0.8).detach().requires_grad_(True)
    lengths = torch.tensor([1024, 731])
    loss, components = MultiResolutionSTFTLoss(
        ((256, 64, 256), (512, 128, 512))
    )(generated, target, lengths)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert generated.grad is not None
    assert torch.count_nonzero(generated.grad) > 0


def test_validation_cache_release_is_cuda_only(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append(("synchronize", device))
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty", None)))

    CodecRefinement._release_validation_cuda_cache(torch.device("cpu"))
    assert calls == []

    device = torch.device("cuda", 0)
    CodecRefinement._release_validation_cuda_cache(device)
    assert calls == [("synchronize", device), ("empty", None)]


def test_refinement_freezes_bitstream_and_selects_exact_vocoder_tail(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, "waveform_joint_vocoder_tail").eval()
    batch = _batch()
    with torch.inference_mode():
        before, lengths_before = model.encode(
            batch["audios"], batch["audio_lengths"]
        )

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.quantizer.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert model.trainable_vocoder_parameter_names
    assert all(
        name.startswith(
            (
                "ups.1.",
                "ups.2.",
                "resblocks.1.",
                "resblocks.2.",
                "activation_post.",
                "conv_post.",
            )
        )
        for name in model.trainable_vocoder_parameter_names
    )
    assert all(
        parameter.requires_grad
        == (name in model.trainable_vocoder_parameter_names)
        for name, parameter in model.vocoder.named_parameters()
    )

    with torch.inference_mode():
        after, lengths_after = model.encode(batch["audios"], batch["audio_lengths"])
    assert torch.equal(before, after)
    assert torch.equal(lengths_before, lengths_after)
    assert "waveform_gan" not in model.codec_definition()["refinement"]
    assert "vocoder_scope" not in model.codec_definition()["refinement"]


def test_refinement_training_gradients_and_checkpoint_scope(tmp_path: Path) -> None:
    model = _model(tmp_path, "waveform_joint_vocoder_tail")
    batch = _batch()
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
    trainer.fit(model, train_dataloaders=DataLoader([batch], batch_size=None))

    assert trainer.global_step == 1
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.quantizer.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    vocoder_keys = {
        key for key in checkpoint["state_dict"] if key.startswith("vocoder.")
    }
    assert vocoder_keys
    assert vocoder_keys == set(
        checkpoint["codec_metadata"]["adapted_vocoder_state_keys"]
    )
    assert not any(key.startswith("vocoder.conv_pre.") for key in vocoder_keys)


def test_gan_refinement_steps_both_optimizers_without_discriminator_leak(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, "waveform_joint_vocoder_tail_gan")
    assert model.codec_definition()["refinement"]["waveform_gan"] is True
    discriminator_before = {
        name: parameter.detach().clone()
        for name, parameter in model.waveform_discriminator.named_parameters()
    }
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader([_batch(), _batch()], batch_size=None),
    )

    assert trainer.global_step == 2
    assert any(
        not torch.equal(discriminator_before[name], parameter)
        for name, parameter in model.waveform_discriminator.named_parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.waveform_discriminator.parameters()
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_generator_backward_uses_recorded_cudnn_policy(
    tmp_path: Path, monkeypatch
) -> None:
    model = _model(tmp_path, "waveform_full_codec_frozen_vocoder_gan")
    model.generator_backward_cudnn_enabled = False
    calls: list[tuple[str, bool]] = []

    @contextmanager
    def flags(*, enabled: bool):
        calls.append(("enter", enabled))
        try:
            yield
        finally:
            calls.append(("exit", enabled))

    monkeypatch.setattr(torch.backends.cudnn, "flags", flags)
    monkeypatch.setattr(
        model,
        "manual_backward",
        lambda loss: calls.append(("backward", bool(loss.item()))),
    )

    model._backward_generator(torch.ones(()))

    assert calls == [
        ("enter", False),
        ("backward", True),
        ("exit", False),
    ]
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    assert (
        checkpoint["codec_metadata"]["training_runtime"][
            "generator_backward_cudnn_enabled"
        ]
        is False
    )
    assert (
        "generator_backward_cudnn_enabled"
        not in model.codec_definition()["refinement"]
    )


@pytest.mark.parametrize(
    "mode",
    (
        "waveform_full_codec_frozen_vocoder_gan",
        "waveform_joint_full_codec_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    ),
)
def test_full_codec_gan_refinement_updates_encoder_and_marks_bitstream_mutable(
    tmp_path: Path,
    mode: str,
) -> None:
    model = _model(tmp_path, mode)
    encoder_before = {
        name: parameter.detach().clone()
        for name, parameter in model.encoder.named_parameters()
    }
    assert model.codec_definition()["refinement"]["bitstream_frozen"] is False
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader([_batch(), _batch()], batch_size=None),
    )

    assert any(
        not torch.equal(encoder_before[name], parameter)
        for name, parameter in model.encoder.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_codec_only_gan_freezes_vocoder_and_uses_two_generator_groups(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, "waveform_full_codec_frozen_vocoder_gan")
    definition = model.codec_definition()["refinement"]
    generator_groups = model.configure_optimizers()[0]["optimizer"].param_groups

    assert definition["codec_encoder_trainable"] is True
    assert definition["codec_decoder_trainable"] is True
    assert definition["vocoder_trainable_parameters"] == 0
    assert not model.trainable_vocoder_parameter_names
    assert all(not parameter.requires_grad for parameter in model.vocoder.parameters())
    assert [group["name"] for group in generator_groups] == [
        "decoder",
        "encoder_and_quantizer",
    ]
    assert [group["initial_lr"] for group in generator_groups] == pytest.approx(
        [1e-4, 5e-5]
    )


def test_vocoder_only_gan_freezes_codec_and_updates_only_vocoder_tail(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, "waveform_vocoder_tail_only_gan")
    batch = _batch()
    frozen_before = {
        name: parameter.detach().clone()
        for module_name, module in (
            ("encoder", model.encoder),
            ("pre_quantized_projection", model.pre_quantized_projection),
            ("quantizer", model.quantizer),
            ("quantized_projection", model.quantized_projection),
            ("decoder", model.decoder),
        )
        for name, parameter in module.named_parameters(prefix=module_name)
    }
    vocoder_before = {
        name: parameter.detach().clone()
        for name, parameter in model.vocoder.named_parameters()
        if parameter.requires_grad
    }
    definition = model.codec_definition()["refinement"]
    generator_groups = model.configure_optimizers()[0]["optimizer"].param_groups

    assert definition["bitstream_frozen"] is True
    assert definition["codec_encoder_trainable"] is False
    assert definition["codec_decoder_trainable"] is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
    assert vocoder_before
    assert [group["name"] for group in generator_groups] == ["vocoder"]
    assert [group["initial_lr"] for group in generator_groups] == pytest.approx([1e-5])

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader([batch, batch], batch_size=None),
    )

    frozen_after = {
        name: parameter.detach()
        for module_name, module in (
            ("encoder", model.encoder),
            ("pre_quantized_projection", model.pre_quantized_projection),
            ("quantizer", model.quantizer),
            ("quantized_projection", model.quantized_projection),
            ("decoder", model.decoder),
        )
        for name, parameter in module.named_parameters(prefix=module_name)
    }
    assert all(
        torch.equal(parameter, frozen_after[name])
        for name, parameter in frozen_before.items()
    )
    assert any(
        not torch.equal(parameter, dict(model.vocoder.named_parameters())[name])
        for name, parameter in vocoder_before.items()
    )


def test_full_vocoder_only_gan_freezes_codec_and_updates_complete_vocoder(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, "waveform_vocoder_full_only_gan")
    definition = model.codec_definition()["refinement"]
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.vocoder.named_parameters()
    }

    assert definition["bitstream_frozen"] is True
    assert definition["codec_decoder_trainable"] is False
    assert definition["vocoder_tail_stages"] == 0
    assert set(model.trainable_vocoder_parameter_names) == set(before)
    assert all(parameter.requires_grad for parameter in model.vocoder.parameters())

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader([_batch(), _batch()], batch_size=None),
    )

    assert any(
        not torch.equal(parameter, dict(model.vocoder.named_parameters())[name])
        for name, parameter in before.items()
    )
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    retained = checkpoint["codec_metadata"]["adapted_vocoder_state_keys"]
    assert "vocoder.conv_pre.weight" in retained


@pytest.mark.parametrize(
    ("mode", "expected_groups"),
    (
        ("mel_control", 1),
        ("waveform_frozen_vocoder", 1),
        ("waveform_joint_vocoder_tail", 2),
    ),
)
def test_refinement_optimizer_groups(
    tmp_path: Path, mode: str, expected_groups: int
) -> None:
    optimizer_config = _model(tmp_path, mode).configure_optimizers()
    assert len(optimizer_config["optimizer"].param_groups) == expected_groups


def test_gan_refinement_has_separate_generator_and_discriminator_optimizers(
    tmp_path: Path,
) -> None:
    optimizer_configs = _model(
        tmp_path, "waveform_joint_vocoder_tail_gan"
    ).configure_optimizers()
    assert len(optimizer_configs) == 2
    assert len(optimizer_configs[0]["optimizer"].param_groups) == 2
    assert len(optimizer_configs[1]["optimizer"].param_groups) == 1


@pytest.mark.parametrize(
    "mode",
    (
        "waveform_joint_full_codec_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    ),
)
def test_full_codec_gan_has_decoder_encoder_vocoder_optimizer_groups(
    tmp_path: Path, mode: str
) -> None:
    optimizer_configs = _model(tmp_path, mode).configure_optimizers()
    generator_groups = optimizer_configs[0]["optimizer"].param_groups

    assert len(generator_groups) == 3
    assert [group["name"] for group in generator_groups] == [
        "decoder",
        "encoder_and_quantizer",
        "vocoder",
    ]
    assert [group["initial_lr"] for group in generator_groups] == pytest.approx(
        [1e-4, 5e-5, 1e-5]
    )


def test_refinement_initialization_rejects_non_vocoder_mismatch() -> None:
    valid = torch.nn.modules.module._IncompatibleKeys(
        ["vocoder.conv_pre.bias", "waveform_discriminator.blocks.0.bias"],
        ["discriminator.blocks.0.bias"],
    )
    CodecRefinement.validate_initial_weights(valid)

    invalid = torch.nn.modules.module._IncompatibleKeys(
        ["decoder.input_projection.bias"], []
    )
    with pytest.raises(ValueError, match="initialization mismatch"):
        CodecRefinement.validate_initial_weights(invalid)
