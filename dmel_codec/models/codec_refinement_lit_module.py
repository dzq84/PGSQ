from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from dmel_codec.models.codec_lit_modules import VQGAN
from dmel_codec.models.modules.waveform_discriminator import (
    discriminator_lsgan_loss,
    feature_matching_loss,
    generator_lsgan_loss,
)
from dmel_codec.utils.utils import sequence_mask


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Length-aware spectral convergence and log-magnitude loss."""

    def __init__(
        self,
        resolutions: Sequence[Sequence[int]] = (
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ),
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        parsed = tuple(tuple(int(value) for value in item) for item in resolutions)
        if not parsed or any(len(item) != 3 for item in parsed):
            raise ValueError("Each STFT resolution must be [n_fft, hop, win_length]")
        for n_fft, hop_length, win_length in parsed:
            if not (0 < hop_length <= win_length <= n_fft):
                raise ValueError(f"Invalid STFT resolution: {(n_fft, hop_length, win_length)}")
        if spectral_convergence_weight < 0 or log_magnitude_weight < 0:
            raise ValueError("STFT loss weights must be non-negative")
        if spectral_convergence_weight == 0 and log_magnitude_weight == 0:
            raise ValueError("At least one STFT loss component must be enabled")
        self.resolutions = parsed
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.epsilon = float(epsilon)
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(
                f"window_{index}", torch.hann_window(win_length), persistent=False
            )

    @staticmethod
    def _validate(
        generated: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
    ) -> None:
        if generated.ndim != 3 or generated.shape[1] != 1:
            raise ValueError(f"Expected generated [B,1,T], got {generated.shape}")
        if generated.shape != target.shape:
            raise ValueError("Generated and target waveforms must have identical shape")
        if lengths.ndim != 1 or lengths.shape[0] != generated.shape[0]:
            raise ValueError("Waveform lengths must have shape [B]")
        if torch.any(lengths <= 0) or torch.any(lengths > generated.shape[-1]):
            raise ValueError("Waveform lengths must lie within the padded tensor")

    def _resolution_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        resolution_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_fft, hop_length, win_length = self.resolutions[resolution_index]
        window = getattr(self, f"window_{resolution_index}").to(
            device=generated.device, dtype=generated.dtype
        )
        convergence_values = []
        log_magnitude_values = []
        for sample_length in torch.unique(lengths, sorted=True).tolist():
            indices = torch.nonzero(
                lengths == sample_length, as_tuple=False
            ).flatten()
            current_generated = generated[indices, 0, :sample_length]
            current_target = target[indices, 0, :sample_length]
            if sample_length < n_fft:
                padding = n_fft - sample_length
                current_generated = F.pad(current_generated, (0, padding))
                current_target = F.pad(current_target, (0, padding))
            generated_stft = torch.stft(
                current_generated,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            ).abs()
            target_stft = torch.stft(
                current_target,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            ).abs()
            difference = generated_stft - target_stft
            convergence_values.append(
                torch.linalg.vector_norm(difference, dim=(-2, -1))
                / torch.linalg.vector_norm(target_stft, dim=(-2, -1)).clamp_min(
                    self.epsilon
                )
            )
            log_magnitude_values.append(
                (
                    generated_stft.clamp_min(self.epsilon).log()
                    - target_stft.clamp_min(self.epsilon).log()
                )
                .abs()
                .mean(dim=(-2, -1))
            )
        return (
            torch.cat(convergence_values).mean(),
            torch.cat(log_magnitude_values).mean(),
        )

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate(generated, target, lengths)
        generated = generated.float()
        target = target.float()
        convergence = []
        log_magnitude = []
        for index in range(len(self.resolutions)):
            current_convergence, current_log_magnitude = self._resolution_loss(
                generated, target, lengths, index
            )
            convergence.append(current_convergence)
            log_magnitude.append(current_log_magnitude)
        mean_convergence = torch.stack(convergence).mean()
        mean_log_magnitude = torch.stack(log_magnitude).mean()
        total = (
            self.spectral_convergence_weight * mean_convergence
            + self.log_magnitude_weight * mean_log_magnitude
        )
        return total, {
            "spectral_convergence": mean_convergence,
            "log_magnitude": mean_log_magnitude,
        }


class CodecRefinement(VQGAN):
    """Refine reconstruction with an explicit codec/vocoder training boundary."""

    MODES = {
        "mel_control",
        "waveform_frozen_vocoder",
        "waveform_joint_vocoder_tail",
        "waveform_joint_vocoder_tail_gan",
        "waveform_full_codec_frozen_vocoder_gan",
        "waveform_vocoder_tail_only_gan",
        "waveform_joint_full_codec_gan",
        "waveform_vocoder_full_only_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    }
    FULL_CODEC_MODES = {
        "waveform_full_codec_frozen_vocoder_gan",
        "waveform_joint_full_codec_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    }
    VOCODER_TRAINING_MODES = {
        "waveform_joint_vocoder_tail",
        "waveform_joint_vocoder_tail_gan",
        "waveform_vocoder_tail_only_gan",
        "waveform_joint_full_codec_gan",
        "waveform_vocoder_full_only_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    }
    FULL_VOCODER_MODES = {
        "waveform_vocoder_full_only_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    }
    TAIL_VOCODER_MODES = VOCODER_TRAINING_MODES - FULL_VOCODER_MODES
    VOCODER_ONLY_MODES = {
        "waveform_vocoder_tail_only_gan",
        "waveform_vocoder_full_only_gan",
    }
    GAN_MODES = {
        "waveform_joint_vocoder_tail_gan",
        "waveform_full_codec_frozen_vocoder_gan",
        "waveform_vocoder_tail_only_gan",
        "waveform_joint_full_codec_gan",
        "waveform_vocoder_full_only_gan",
        "waveform_joint_full_codec_full_vocoder_gan",
    }

    def __init__(
        self,
        *args,
        refinement_mode: str,
        waveform_loss_weight: float = 1.0,
        validation_waveform_loss_weight: float = 1.0,
        stft_resolutions: Sequence[Sequence[int]] = (
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ),
        vocoder_tail_stages: int = 2,
        encoder_lr_scale: float = 0.5,
        vocoder_lr_scale: float = 0.1,
        waveform_discriminator: torch.nn.Module | None = None,
        discriminator_lr_scale: float = 1.0,
        gan_adversarial_weight: float = 0.1,
        gan_feature_weight: float = 10.0,
        generator_backward_cudnn_enabled: bool = True,
        gradient_clip_val: float = 500.0,
        **kwargs,
    ) -> None:
        if refinement_mode not in self.MODES:
            raise ValueError(
                f"Unknown refinement_mode={refinement_mode!r}; expected {sorted(self.MODES)}"
            )
        if kwargs.get("discriminator") is not None:
            raise ValueError(
                "Codec mel discriminator must be null during waveform refinement"
            )
        full_codec_mode = refinement_mode in self.FULL_CODEC_MODES
        if full_codec_mode == bool(kwargs.get("freeze_encoder", False)):
            required = "false" if full_codec_mode else "true"
            raise ValueError(
                f"{refinement_mode} requires freeze_encoder={required}"
            )
        if int(kwargs.get("accumulate_grad", 1)) != 1:
            raise ValueError("Use Trainer accumulation for refinement, not accumulate_grad")
        if waveform_loss_weight < 0 or validation_waveform_loss_weight < 0:
            raise ValueError("Waveform loss weights must be non-negative")
        if min(encoder_lr_scale, vocoder_lr_scale, discriminator_lr_scale) <= 0:
            raise ValueError("optimizer learning-rate scales must be positive")
        if gan_adversarial_weight < 0 or gan_feature_weight < 0:
            raise ValueError("GAN loss weights must be non-negative")
        if not isinstance(generator_backward_cudnn_enabled, bool):
            raise TypeError("generator_backward_cudnn_enabled must be a bool")
        if gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")
        gan_mode = refinement_mode in self.GAN_MODES
        if gan_mode != (waveform_discriminator is not None):
            raise ValueError(
                "waveform_discriminator must be configured exactly for the GAN mode"
            )
        super().__init__(*args, **kwargs)
        if self.vocoder is None:
            raise ValueError("Decoder refinement requires a BigVGAN vocoder")

        self.refinement_mode = refinement_mode
        self.waveform_loss_weight = float(waveform_loss_weight)
        self.validation_waveform_loss_weight = float(validation_waveform_loss_weight)
        self.vocoder_tail_stages = int(vocoder_tail_stages)
        self.encoder_lr_scale = float(encoder_lr_scale)
        self.vocoder_lr_scale = float(vocoder_lr_scale)
        self.waveform_discriminator = waveform_discriminator
        self.discriminator_lr_scale = float(discriminator_lr_scale)
        self.gan_adversarial_weight = float(gan_adversarial_weight)
        self.gan_feature_weight = float(gan_feature_weight)
        self.generator_backward_cudnn_enabled = generator_backward_cudnn_enabled
        self.gradient_clip_val = float(gradient_clip_val)
        self.waveform_loss = MultiResolutionSTFTLoss(stft_resolutions)
        self.bitstream_frozen = not full_codec_mode
        self.decoder_trainable = refinement_mode not in self.VOCODER_ONLY_MODES
        self.automatic_optimization = not gan_mode

        for parameter in self.decoder.parameters():
            parameter.requires_grad_(self.decoder_trainable)
        for parameter in self.vocoder.parameters():
            parameter.requires_grad_(False)
        self._trainable_vocoder_names: set[str] = set()
        if self.refinement_mode in self.FULL_VOCODER_MODES:
            self._trainable_vocoder_names = self._unfreeze_vocoder_all(self.vocoder)
        elif self.refinement_mode in self.TAIL_VOCODER_MODES:
            self._trainable_vocoder_names = self._unfreeze_vocoder_tail(
                self.vocoder, self.vocoder_tail_stages
            )
        elif self.refinement_mode == "mel_control":
            self.waveform_loss_weight = 0.0

    def train(self, mode: bool = True):
        super().train(mode)
        if self.bitstream_frozen:
            self.encoder.eval()
            self.pre_quantized_projection.eval()
            self.quantizer.eval()
            self.quantized_projection.eval()
        if not self.decoder_trainable:
            self.decoder.eval()
        self.vocoder.eval()
        if self.waveform_discriminator is not None:
            self.waveform_discriminator.train(mode)
        return self

    def on_train_epoch_start(self) -> None:
        if self.waveform_discriminator is None:
            return
        optimizer_generator, optimizer_discriminator = self.optimizers()
        optimizer_generator.zero_grad()
        optimizer_discriminator.zero_grad()

    @staticmethod
    def _unfreeze_vocoder_tail(
        vocoder: torch.nn.Module, stages: int
    ) -> set[str]:
        required = ("ups", "resblocks", "activation_post", "conv_post", "num_kernels")
        missing = [name for name in required if not hasattr(vocoder, name)]
        if missing:
            raise ValueError(f"BigVGAN is missing tail modules: {missing}")
        if stages < 1 or stages > len(vocoder.ups):
            raise ValueError(
                f"vocoder_tail_stages must be in [1,{len(vocoder.ups)}], got {stages}"
            )
        modules: list[torch.nn.Module] = list(vocoder.ups[-stages:])
        resblock_count = stages * int(vocoder.num_kernels)
        modules.extend(vocoder.resblocks[-resblock_count:])
        modules.extend((vocoder.activation_post, vocoder.conv_post))
        trainable_ids = set()
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                trainable_ids.add(id(parameter))
        return {
            name
            for name, parameter in vocoder.named_parameters()
            if id(parameter) in trainable_ids
        }

    @staticmethod
    def _unfreeze_vocoder_all(vocoder: torch.nn.Module) -> set[str]:
        names = set()
        for name, parameter in vocoder.named_parameters():
            parameter.requires_grad_(True)
            names.add(name)
        if not names:
            raise ValueError("Full-vocoder refinement has no trainable parameters")
        return names

    @property
    def trainable_vocoder_parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._trainable_vocoder_names))

    @staticmethod
    def validate_initial_weights(load_result) -> None:
        invalid_missing = [
            key
            for key in load_result.missing_keys
            if not key.startswith(("vocoder.", "waveform_discriminator."))
        ]
        invalid_unexpected = [
            key
            for key in load_result.unexpected_keys
            if not key.startswith("discriminator.")
        ]
        if invalid_missing or invalid_unexpected:
            raise ValueError(
                "Refinement initialization mismatch: "
                f"missing={invalid_missing}, unexpected={invalid_unexpected}"
            )

    def codec_definition(self) -> dict[str, Any]:
        definition = super().codec_definition()
        definition["refinement"] = {
            "mode": self.refinement_mode,
            "bitstream_frozen": self.bitstream_frozen,
            "codec_encoder_trainable": not self.bitstream_frozen,
            "codec_decoder_trainable": self.decoder_trainable,
            "waveform_loss_weight": self.waveform_loss_weight,
            "validation_waveform_loss_weight": self.validation_waveform_loss_weight,
            "stft_resolutions": [list(item) for item in self.waveform_loss.resolutions],
            "vocoder_tail_stages": (
                self.vocoder_tail_stages
                if self.refinement_mode in self.TAIL_VOCODER_MODES
                else 0
            ),
            "vocoder_trainable_parameters": sum(
                parameter.numel()
                for parameter in self.vocoder.parameters()
                if parameter.requires_grad
            ),
        }
        if self.waveform_discriminator is not None:
            definition["refinement"].update(
                {
                    "waveform_gan": True,
                    "encoder_lr_scale": self.encoder_lr_scale,
                    "gan_adversarial_weight": self.gan_adversarial_weight,
                    "gan_feature_weight": self.gan_feature_weight,
                }
            )
        return definition

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get("state_dict", {})
        retained_names = (
            set(self.vocoder.state_dict())
            if self.refinement_mode in self.FULL_VOCODER_MODES
            else self._trainable_vocoder_names
        )
        retained_vocoder_keys = {f"vocoder.{name}" for name in retained_names}
        for name in list(state_dict):
            if name.startswith("vocoder.") and name not in retained_vocoder_keys:
                state_dict.pop(name)
        checkpoint["codec_metadata"] = {
            **self.checkpoint_metadata,
            "codec_definition": self.codec_definition(),
            "adapted_vocoder_state_keys": sorted(retained_vocoder_keys),
            "training_runtime": {
                "generator_backward_cudnn_enabled": (
                    self.generator_backward_cudnn_enabled
                ),
            },
        }

    def configure_optimizers(self):
        if self.optimizer_builder is None or self.lr_scheduler_builder is None:
            raise ValueError("optimizer and lr_scheduler must be configured for training")
        parameter_groups: list[tuple[list[torch.nn.Parameter], float, str]] = []
        decoder_parameters = [
            parameter for parameter in self.decoder.parameters() if parameter.requires_grad
        ]
        if decoder_parameters:
            parameter_groups.append((decoder_parameters, 1.0, "decoder"))
        if not self.bitstream_frozen:
            encoder_parameters = [
                parameter
                for module in (
                    self.encoder,
                    self.pre_quantized_projection,
                    self.quantizer,
                    self.quantized_projection,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            if not encoder_parameters:
                raise ValueError("Full-codec refinement has no trainable encoder parameters")
            parameter_groups.append(
                (encoder_parameters, self.encoder_lr_scale, "encoder_and_quantizer")
            )
        vocoder_parameters = [
            parameter for parameter in self.vocoder.parameters() if parameter.requires_grad
        ]
        if vocoder_parameters:
            parameter_groups.append(
                (vocoder_parameters, self.vocoder_lr_scale, "vocoder")
            )
        if not parameter_groups:
            raise ValueError("Refinement has no trainable generator parameters")

        first_parameters, first_scale, first_name = parameter_groups[0]
        optimizer = self.optimizer_builder(first_parameters)
        base_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.param_groups[0].update(
            {
                "lr": base_lr * first_scale,
                "initial_lr": base_lr * first_scale,
                "name": first_name,
            }
        )
        for parameters, scale, name in parameter_groups[1:]:
            optimizer.add_param_group(
                {
                    "params": parameters,
                    "lr": base_lr * scale,
                    "initial_lr": base_lr * scale,
                    "name": name,
                }
            )
        scheduler = self.lr_scheduler_builder(optimizer)
        generator_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "optimizer/refinement",
            },
        }
        if self.waveform_discriminator is None:
            return generator_config
        discriminator_optimizer = self.optimizer_builder(
            self.waveform_discriminator.parameters()
        )
        for group in discriminator_optimizer.param_groups:
            group["lr"] *= self.discriminator_lr_scale
            group["initial_lr"] = group["lr"]
        discriminator_scheduler = self.lr_scheduler_builder(discriminator_optimizer)
        return (
            generator_config,
            {
                "optimizer": discriminator_optimizer,
                "lr_scheduler": {
                    "scheduler": discriminator_scheduler,
                    "interval": "step",
                    "name": "optimizer/waveform_discriminator",
                },
            },
        )

    def _length_aware_vocoder(
        self,
        mels: torch.Tensor,
        mel_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if mels.shape[0] != mel_lengths.shape[0] or mels.shape[0] != audio_lengths.shape[0]:
            raise ValueError("Mel/audio lengths must match the batch size")
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (mel_length, audio_length) in enumerate(
            zip(mel_lengths.tolist(), audio_lengths.tolist())
        ):
            groups[(int(mel_length), int(audio_length))].append(index)

        waveforms: list[torch.Tensor | None] = [None] * mels.shape[0]
        for (mel_length, audio_length), indices in groups.items():
            index_tensor = torch.tensor(indices, device=mels.device, dtype=torch.long)
            current = self.vocoder(
                mels.index_select(0, index_tensor)[..., :mel_length]
            )
            current = self._resize_time(current, audio_length)
            for local_index, original_index in enumerate(indices):
                waveforms[original_index] = current[local_index]
        if any(waveform is None for waveform in waveforms):
            raise RuntimeError("Vocoder failed to reconstruct every batch item")
        max_length = int(audio_lengths.max().item())
        return torch.stack(
            [
                F.pad(waveform, (0, max_length - waveform.shape[-1]))
                for waveform in waveforms
            ]
        )

    def _reconstruct_training_batch(
        self, batch: dict[str, Any]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        audios = batch["audios"].float()
        audio_lengths = batch["audio_lengths"]
        cut_ids = batch.get("cut_ids")
        encode_mels, gt_mels, mel_lengths, mel_mask = self._mel_inputs(
            audios, audio_lengths
        )
        if self.bitstream_frozen:
            with torch.no_grad():
                encoded = self._encode_mels(encode_mels, mel_mask)
                quantized = self.quantizer(self.pre_quantized_projection(encoded))
                reconstructed = self.quantized_projection(quantized.z) * mel_mask
        else:
            encoded = self._encode_mels(encode_mels, mel_mask)
            quantized = self.quantizer(self.pre_quantized_projection(encoded))
            reconstructed = self.quantized_projection(quantized.z) * mel_mask
        generated_mels = self._decode_features(reconstructed, mel_mask, cut_ids)
        return audios, gt_mels, generated_mels, mel_lengths, mel_mask, quantized.codes

    def _losses(
        self, batch: dict[str, Any], *, waveform_weight: float
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        (
            audios,
            gt_mels,
            generated_mels,
            mel_lengths,
            mel_mask,
            codes,
        ) = self._reconstruct_training_batch(batch)
        mel_loss = self._weighted_mel_loss(generated_mels, gt_mels, mel_mask)
        waveform_loss = mel_loss.new_zeros(())
        spectral = mel_loss.new_zeros(())
        log_magnitude = mel_loss.new_zeros(())
        generated_audios = None
        if waveform_weight > 0:
            generated_audios = self._length_aware_vocoder(
                generated_mels, mel_lengths, batch["audio_lengths"]
            )
            waveform_mask = sequence_mask(
                batch["audio_lengths"], generated_audios.shape[-1]
            )[:, None].to(generated_audios.dtype)
            generated_audios = generated_audios * waveform_mask
            waveform_loss, components = self.waveform_loss(
                generated_audios, audios, batch["audio_lengths"]
            )
            spectral = components["spectral_convergence"]
            log_magnitude = components["log_magnitude"]
        total = self.weight_mel * mel_loss + waveform_weight * waveform_loss
        return total, {
            "loss_mel": mel_loss,
            "loss_waveform": waveform_loss,
            "spectral_convergence": spectral,
            "log_magnitude": log_magnitude,
        }, codes, generated_audios, audios

    def _gan_training_step(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.waveform_discriminator is None:
            raise RuntimeError("GAN training requires a waveform discriminator")
        total_regression, losses, codes, generated_audios, audios = self._losses(
            batch, waveform_weight=self.waveform_loss_weight
        )
        if generated_audios is None:
            raise RuntimeError("GAN training requires waveform_loss_weight > 0")

        optimizer_generator, optimizer_discriminator = self.optimizers()
        scheduler_generator, scheduler_discriminator = self.lr_schedulers()
        optimizer_generator.zero_grad()
        optimizer_discriminator.zero_grad()

        batch_size = audios.shape[0]
        discriminator_loss = total_regression.new_zeros(())
        for sample_length in torch.unique(batch["audio_lengths"], sorted=True).tolist():
            indices = torch.nonzero(
                batch["audio_lengths"] == sample_length, as_tuple=False
            ).flatten()
            real = audios.index_select(0, indices)[..., :sample_length]
            fake = generated_audios.index_select(0, indices)[..., :sample_length]
            real_logits, _ = self.waveform_discriminator(real)
            fake_logits, _ = self.waveform_discriminator(fake.detach())
            discriminator_loss = discriminator_loss + (
                indices.numel()
                * discriminator_lsgan_loss(real_logits, fake_logits)
                / batch_size
            )
        self.manual_backward(discriminator_loss)
        self.clip_gradients(
            optimizer_discriminator,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        raw_discriminator_optimizer = getattr(
            optimizer_discriminator, "optimizer", optimizer_discriminator
        )
        raw_discriminator_optimizer.step()
        scheduler_discriminator.step()
        optimizer_discriminator.zero_grad()

        adversarial_loss = total_regression.new_zeros(())
        matching_loss = total_regression.new_zeros(())
        previous = self._set_requires_grad(self.waveform_discriminator, False)
        try:
            for sample_length in torch.unique(
                batch["audio_lengths"], sorted=True
            ).tolist():
                indices = torch.nonzero(
                    batch["audio_lengths"] == sample_length, as_tuple=False
                ).flatten()
                real = audios.index_select(0, indices)[..., :sample_length]
                fake = generated_audios.index_select(0, indices)[..., :sample_length]
                with torch.no_grad():
                    _, real_features = self.waveform_discriminator(real)
                fake_logits, fake_features = self.waveform_discriminator(fake)
                weight = indices.numel() / batch_size
                adversarial_loss = adversarial_loss + weight * generator_lsgan_loss(
                    fake_logits
                )
                matching_loss = matching_loss + weight * feature_matching_loss(
                    real_features, fake_features
                )
        finally:
            self._restore_requires_grad(self.waveform_discriminator, previous)

        total_generator = (
            total_regression
            + self.gan_adversarial_weight * adversarial_loss
            + self.gan_feature_weight * matching_loss
        )
        self._backward_generator(total_generator)
        self.clip_gradients(
            optimizer_generator,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        optimizer_generator.step()
        scheduler_generator.step()
        optimizer_generator.zero_grad()

        self._log_training_metrics(
            batch,
            codes,
            total_generator,
            {
                **losses,
                "loss_adversarial": adversarial_loss,
                "loss_feature_matching": matching_loss,
                "loss_discriminator": discriminator_loss,
            },
        )
        return total_generator.detach()

    def _backward_generator(self, loss: torch.Tensor) -> None:
        # cuDNN 9.1 on the pinned Torch 2.6/CUDA 12.4 environment repeatedly
        # loses a convolution-backward engine on long full-codec GAN runs. The
        # native CUDA path preserves the graph and objective while avoiding that
        # backend failure; keep discriminator backward on its configured backend.
        with torch.backends.cudnn.flags(
            enabled=self.generator_backward_cudnn_enabled
        ):
            self.manual_backward(loss)

    def _log_training_metrics(
        self,
        batch: dict[str, Any],
        codes: torch.Tensor,
        total: torch.Tensor,
        losses: dict[str, torch.Tensor],
    ) -> None:
        batch_size = batch["audios"].shape[0]
        self.log("train/refinement/loss", total, on_step=True, prog_bar=True)
        for name, value in losses.items():
            self.log(f"train/refinement/{name}", value, on_step=True)
        self.log(
            "train/batch_audio_seconds",
            batch["audio_lengths"].sum().float() / self.sampling_rate,
            on_step=True,
            batch_size=batch_size,
        )
        if batch["audios"].is_cuda:
            self.log(
                "train/peak_cuda_memory_gib",
                torch.tensor(
                    torch.cuda.max_memory_allocated(batch["audios"].device) / (1024**3),
                    device=batch["audios"].device,
                ),
                on_step=True,
                batch_size=batch_size,
            )
        code_lengths = torch.div(
            batch["audio_lengths"],
            self.encode_mel_transform.hop_length
            * math.prod(self.quantizer.downsample_factor),
            rounding_mode="floor",
        )
        self.log_dict(
            self._code_usage_statistics(
                codes.detach(), code_lengths, self.quantizer.codebook_cardinality
            ),
            on_step=True,
            batch_size=batch_size,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        if self.waveform_discriminator is not None:
            return self._gan_training_step(batch)
        total, losses, codes, _, _ = self._losses(
            batch, waveform_weight=self.waveform_loss_weight
        )
        self._log_training_metrics(batch, codes, total, losses)
        return total

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        total, losses, _, _, _ = self._losses(
            batch, waveform_weight=self.validation_waveform_loss_weight
        )
        batch_size = batch["audios"].shape[0]
        self.log(
            "val_loss",
            total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in losses.items():
            self.log(
                f"val/refinement/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return total

    @staticmethod
    def _release_validation_cuda_cache(device: torch.device) -> None:
        if device.type != "cuda":
            return
        # Full-vocoder validation leaves large cuDNN workspaces fragmented before
        # the next GAN interval; both long refinement paths failed at that boundary.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        self._release_validation_cuda_cache(self.device)
