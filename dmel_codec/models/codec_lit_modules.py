from __future__ import annotations

import hashlib
import itertools
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt

from dmel_codec.models.modules.bigvgan.bigvgan import BigVGAN
from dmel_codec.models.modules.discriminator import Discriminator
from dmel_codec.models.modules.downsample_fsq import DownsampleFiniteScalarQuantize
from dmel_codec.models.modules.wavenet import WaveNet
from dmel_codec.utils.logger import RankedLogger
from dmel_codec.utils.spectrogram import LogMelSpectrogram, length_aware_spectrogram
from dmel_codec.utils.utils import avg_with_mask, plot_mel, sequence_mask


log = RankedLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class VQGAN(L.LightningModule):
    """Grouped-FSQ spectral codec with deterministic reconstruction semantics."""

    def __init__(
        self,
        encoder: WaveNet,
        quantizer: DownsampleFiniteScalarQuantize,
        vocoder: BigVGAN | None,
        encode_mel_transform: LogMelSpectrogram,
        gt_mel_transform: LogMelSpectrogram,
        optimizer: Callable | None = None,
        lr_scheduler: Callable | None = None,
        discriminator: Discriminator | None = None,
        decoder: WaveNet | None = None,
        pre_quantized_projection: torch.nn.Module | None = None,
        quantized_projection: torch.nn.Module | None = None,
        weight_adv: float = 1.0,
        weight_mel: float = 1.0,
        sampling_rate: int = 24_000,
        freeze_encoder: bool = False,
        dmel_groups: int = 0,
        dtype: torch.dtype | str = "float32",
        accumulate_grad: int = 1,
        decoder_input_policy: str = "zeros",
        decoder_noise_seed: int = 2026,
        mel_bin_permutation: list[int] | tuple[int, ...] | None = None,
        # Accepted only so old LM configs still instantiate. Neither affects this codec.
        weight_vq: float | None = None,
        quanlity_linear: int | None = None,
    ) -> None:
        super().__init__()
        self.encode_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        log.info(f"VQGAN using dtype: {self.encode_dtype}")
        if decoder is None:
            raise ValueError("decoder must be configured")
        if accumulate_grad < 1:
            raise ValueError("accumulate_grad must be at least 1")
        if decoder_input_policy not in {"zeros", "cut_id_noise"}:
            raise ValueError(
                "decoder_input_policy must be either 'zeros' or 'cut_id_noise'"
            )
        if weight_vq not in {None, 0, 0.0}:
            raise ValueError("FSQ has no commitment/VQ loss; weight_vq must be zero")
        if quanlity_linear is not None:
            warnings.warn(
                "quanlity_linear is a legacy compatibility argument and is ignored; "
                "quality side information has been removed",
                stacklevel=2,
            )

        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.pre_quantized_projection = (
            pre_quantized_projection or torch.nn.Identity()
        )
        self.has_pre_quantized_projection = pre_quantized_projection is not None
        self.quantized_projection = quantized_projection or torch.nn.Identity()
        self.has_quantized_projection = quantized_projection is not None
        self.discriminator = discriminator
        self.encode_mel_transform = encode_mel_transform
        self.gt_mel_transform = gt_mel_transform
        self.weight_adv = float(weight_adv)
        self.weight_mel = float(weight_mel)
        self.sampling_rate = int(sampling_rate)
        self.dmel_groups = int(dmel_groups)
        self.accumulate_grad = int(accumulate_grad)
        self.decoder_input_policy = decoder_input_policy
        self.decoder_noise_seed = int(decoder_noise_seed)
        self.mel_bin_permutation = (
            tuple(int(index) for index in mel_bin_permutation)
            if mel_bin_permutation is not None
            else None
        )
        if self.mel_bin_permutation is not None and sorted(
            self.mel_bin_permutation
        ) != list(range(len(self.mel_bin_permutation))):
            raise ValueError("mel_bin_permutation must contain each index exactly once")
        self.checkpoint_metadata: dict[str, Any] = {}
        self._accumulated_batches = 0

        if self.weight_adv != 0.0 and self.discriminator is None:
            raise ValueError("weight_adv is non-zero but discriminator is not configured")
        if freeze_encoder:
            for parameter in itertools.chain(
                self.encoder.parameters(),
                self.pre_quantized_projection.parameters(),
                self.quantizer.parameters(),
                self.quantized_projection.parameters(),
            ):
                parameter.requires_grad = False

        vocoder_ckpt = getattr(vocoder, "ckpt_path", None)
        log.info(f"Vocoder checkpoint: {vocoder_ckpt}")
        if vocoder is None:
            self.vocoder = None
        elif vocoder_ckpt and Path(vocoder_ckpt).is_file():
            payload = torch.load(vocoder_ckpt, map_location="cpu", weights_only=True)
            vocoder.load_state_dict(payload["generator"], strict=True)
            self.vocoder = vocoder.eval()
            for parameter in self.vocoder.parameters():
                parameter.requires_grad = False
        else:
            raise FileNotFoundError(
                f"BigVGAN checkpoint does not exist: {vocoder_ckpt!r}"
            )

        self.automatic_optimization = False
        self.strict_loading = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vocoder is not None:
            self.vocoder.eval()
        return self

    @property
    def token_frame_rate(self) -> float:
        return self.sampling_rate / (
            self.encode_mel_transform.hop_length
            * math.prod(self.quantizer.downsample_factor)
        )

    @property
    def nominal_bitrate(self) -> float:
        return (
            self.token_frame_rate
            * self.quantizer.num_streams
            * math.log2(self.quantizer.codebook_cardinality)
            / 1000.0
        )

    def codec_definition(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sampling_rate,
            "mel_hop_length": self.encode_mel_transform.hop_length,
            "downsample_factor": list(self.quantizer.downsample_factor),
            "token_frame_rate": self.token_frame_rate,
            "num_parallel_streams": self.quantizer.num_streams,
            "mel_group_count": self.dmel_groups,
            "quantizer_group_count": self.quantizer.groups,
            "residual_codebooks_per_group": self.quantizer.n_codebooks,
            "codebook_cardinalities": [
                self.quantizer.codebook_cardinality
            ]
            * self.quantizer.num_streams,
            "nominal_bitrate_kbps": self.nominal_bitrate,
            "decoder_input_policy": self.decoder_input_policy,
            "decoder_noise_seed": (
                self.decoder_noise_seed
                if self.decoder_input_policy == "cut_id_noise"
                else None
            ),
            "quality_side_information": "none",
            "commitment_vq_loss": False,
            "mel_bin_permutation": (
                list(self.mel_bin_permutation)
                if self.mel_bin_permutation is not None
                else None
            ),
            "pre_quantized_projection": self.has_pre_quantized_projection,
            "quantized_projection": self.has_quantized_projection,
        }

    def set_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        self.checkpoint_metadata = dict(metadata)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get("state_dict", {})
        for name in list(state_dict):
            if name.startswith("vocoder."):
                state_dict.pop(name)
        checkpoint["codec_metadata"] = {
            **self.checkpoint_metadata,
            "codec_definition": self.codec_definition(),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        saved_definition = checkpoint.get("codec_metadata", {}).get(
            "codec_definition", {}
        )
        saved_policy = saved_definition.get("decoder_input_policy")
        if saved_policy and saved_policy != self.decoder_input_policy:
            raise ValueError(
                "Checkpoint decoder policy mismatch: "
                f"saved={saved_policy}, configured={self.decoder_input_policy}"
            )

    def configure_optimizers(self):
        if self.optimizer_builder is None or self.lr_scheduler_builder is None:
            raise ValueError("optimizer and lr_scheduler must be configured for training")
        generator_parameters: Iterable[torch.nn.Parameter] = itertools.chain(
            self.encoder.parameters(),
            self.pre_quantized_projection.parameters(),
            self.quantizer.parameters(),
            self.quantized_projection.parameters(),
            self.decoder.parameters(),
        )
        optimizer_generator = self.optimizer_builder(generator_parameters)
        scheduler_generator = self.lr_scheduler_builder(optimizer_generator)
        generator_config = {
            "optimizer": optimizer_generator,
            "lr_scheduler": {
                "scheduler": scheduler_generator,
                "interval": "step",
                "name": "optimizer/generator",
            },
        }
        if self.discriminator is None:
            return generator_config

        optimizer_discriminator = self.optimizer_builder(
            self.discriminator.parameters()
        )
        scheduler_discriminator = self.lr_scheduler_builder(optimizer_discriminator)
        return (
            generator_config,
            {
                "optimizer": optimizer_discriminator,
                "lr_scheduler": {
                    "scheduler": scheduler_discriminator,
                    "interval": "step",
                    "name": "optimizer/discriminator",
                },
            },
        )

    def expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.repeat_interleave(self.dmel_groups, dim=0)

    def _encode_mels(self, mels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mel_bin_permutation is not None:
            if len(self.mel_bin_permutation) != mels.shape[1]:
                raise ValueError(
                    "mel_bin_permutation length does not match frontend mel bins"
                )
            permutation = torch.tensor(
                self.mel_bin_permutation, device=mels.device, dtype=torch.long
            )
            mels = mels.index_select(1, permutation)
        if self.dmel_groups > 0:
            batch_size, num_mels, time_size = mels.shape
            if num_mels % self.dmel_groups:
                raise ValueError(
                    f"{num_mels} mel bins are not divisible by {self.dmel_groups} groups"
                )
            grouped_mask = self.expand_mask(mask)
            grouped_mels = mels.contiguous().view(
                batch_size * self.dmel_groups,
                num_mels // self.dmel_groups,
                time_size,
            )
            return self.encoder(grouped_mels * grouped_mask) * grouped_mask
        return self.encoder(mels * mask) * mask

    def _decoder_input(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        cut_ids: list[str] | None,
    ) -> torch.Tensor:
        if self.decoder_input_policy == "zeros":
            return torch.zeros_like(condition)
        if cut_ids is None or len(cut_ids) != condition.shape[0]:
            raise ValueError("cut_id_noise requires one cut ID per batch item")
        noise = []
        for cut_id in cut_ids:
            digest = hashlib.sha256(
                f"{self.decoder_noise_seed}:{cut_id}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device=condition.device).manual_seed(seed)
            noise.append(
                torch.randn(
                    condition.shape[1:],
                    device=condition.device,
                    dtype=condition.dtype,
                    generator=generator,
                )
            )
        return torch.stack(noise) * mask

    def _decode_features(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        cut_ids: list[str] | None,
    ) -> torch.Tensor:
        decoder_input = self._decoder_input(features, mask, cut_ids)
        return self.decoder(decoder_input, condition=features * mask) * mask

    def _mel_inputs(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if audios.ndim != 3 or audios.shape[1] != 1:
            raise ValueError(f"Expected audios [B,1,T], got {tuple(audios.shape)}")
        if audio_lengths.ndim != 1 or audio_lengths.shape[0] != audios.shape[0]:
            raise ValueError(
                f"Expected audio_lengths [B], got {tuple(audio_lengths.shape)}"
            )
        audios = audios.float()
        with torch.no_grad():
            encode_mels = length_aware_spectrogram(
                self.encode_mel_transform, audios, audio_lengths
            ).to(self.encode_dtype)
            gt_mels = length_aware_spectrogram(
                self.gt_mel_transform, audios, audio_lengths
            ).float()
        mel_lengths = torch.div(
            audio_lengths,
            self.gt_mel_transform.hop_length,
            rounding_mode="floor",
        )
        mask = sequence_mask(mel_lengths, gt_mels.shape[-1])[:, None].float()
        return encode_mels, gt_mels * mask, mel_lengths, mask

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> list[bool]:
        previous = [parameter.requires_grad for parameter in module.parameters()]
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
        return previous

    @staticmethod
    def _restore_requires_grad(module: torch.nn.Module, previous: list[bool]) -> None:
        for parameter, enabled in zip(module.parameters(), previous):
            parameter.requires_grad_(enabled)

    @staticmethod
    def _rescale_gradients(optimizer, scale: float) -> None:
        if scale == 1.0:
            return
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        for group in raw_optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)

    def _optimizer_handles(self):
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        if self.discriminator is None:
            return optimizers, schedulers, None, None
        return optimizers[0], schedulers[0], optimizers[1], schedulers[1]

    def _step_accumulated_optimizers(self, *, tail_batches: int | None = None) -> None:
        optimizer_g, scheduler_g, optimizer_d, scheduler_d = self._optimizer_handles()
        batches = tail_batches or self.accumulate_grad
        gradient_scale = self.accumulate_grad / batches
        self._rescale_gradients(optimizer_g, gradient_scale)
        self.clip_gradients(
            optimizer_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        if optimizer_d is not None:
            self._rescale_gradients(optimizer_d, gradient_scale)
            self.clip_gradients(
                optimizer_d,
                gradient_clip_val=1000.0,
                gradient_clip_algorithm="norm",
            )
            # Only generator updates advance Lightning global_step. This makes
            # max_steps and both schedulers count actual codec update iterations.
            optimizer_d.optimizer.step()
            scheduler_d.step()
        optimizer_g.step()
        scheduler_g.step()
        optimizer_g.zero_grad()
        if optimizer_d is not None:
            optimizer_d.zero_grad()
        self._accumulated_batches = 0

    def on_train_epoch_start(self) -> None:
        if self._accumulated_batches == 0:
            optimizer_g, _, optimizer_d, _ = self._optimizer_handles()
            optimizer_g.zero_grad()
            if optimizer_d is not None:
                optimizer_d.zero_grad()

    def on_train_epoch_end(self) -> None:
        if self._accumulated_batches:
            raise RuntimeError(
                "Gradient accumulation tail was not flushed in training_step; "
                "the dataloader must expose its final batch to Lightning"
            )

    def _weighted_mel_loss(
        self, generated: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        distance = (generated - target).abs()
        num_mels = distance.shape[1]
        low_end = max(1, round(num_mels * 0.4))
        mid_end = min(num_mels - 1, max(low_end + 1, round(num_mels * 0.7)))
        band_loss = (
            0.6 * avg_with_mask(distance[:, :low_end], mask)
            + 0.3 * avg_with_mask(distance[:, low_end:mid_end], mask)
            + 0.1 * avg_with_mask(distance[:, mid_end:], mask)
        )
        return 0.5 * band_loss + 0.5 * avg_with_mask(distance, mask)

    @staticmethod
    def _code_usage_statistics(
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        cardinality: int,
    ) -> dict[str, torch.Tensor]:
        """Compute per-stream batch diagnostics while excluding padded frames."""
        if codes.ndim != 3:
            raise ValueError(f"Expected codes [B,S,F], got {tuple(codes.shape)}")
        if code_lengths.ndim != 1 or code_lengths.shape[0] != codes.shape[0]:
            raise ValueError("code_lengths must have shape [B]")
        if cardinality < 1:
            raise ValueError("cardinality must be positive")

        code_lengths = code_lengths.to(device=codes.device)
        valid = sequence_mask(code_lengths, codes.shape[-1])
        metrics: dict[str, torch.Tensor] = {}
        utilizations = []
        entropies = []
        for stream in range(codes.shape[1]):
            values = codes[:, stream][valid].long()
            counts = torch.bincount(values, minlength=cardinality).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            nonzero = probabilities > 0
            utilization = nonzero.sum().float() / cardinality
            entropy = -(probabilities[nonzero] * probabilities[nonzero].log2()).sum()
            metrics[f"train/code/stream_{stream:02d}/utilization"] = utilization
            metrics[f"train/code/stream_{stream:02d}/entropy_bits"] = entropy
            utilizations.append(utilization)
            entropies.append(entropy)
        metrics["train/code/min_utilization"] = torch.stack(utilizations).min()
        metrics["train/code/mean_entropy_bits"] = torch.stack(entropies).mean()
        return metrics

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        audios = batch["audios"]
        audio_lengths = batch["audio_lengths"]
        cut_ids = batch.get("cut_ids")
        encode_mels, gt_mels, mel_lengths, mel_mask = self._mel_inputs(
            audios, audio_lengths
        )
        batch_size = audios.shape[0]
        encoded_features = self._encode_mels(encode_mels, mel_mask)
        quantized = self.quantizer(self.pre_quantized_projection(encoded_features))
        code_lengths = torch.div(
            mel_lengths,
            math.prod(self.quantizer.downsample_factor),
            rounding_mode="floor",
        )
        reconstructed_features = self.quantized_projection(quantized.z) * mel_mask
        generated_mels = self._decode_features(
            reconstructed_features, mel_mask, cut_ids
        )

        loss_d = generated_mels.new_zeros(())
        loss_adv = generated_mels.new_zeros(())
        if self.discriminator is not None:
            real_logits = self.discriminator(gt_mels)
            fake_logits = self.discriminator(generated_mels.detach())
            discriminator_mask = F.interpolate(
                mel_mask, size=real_logits.shape[-1], mode="nearest"
            )
            loss_d = 0.5 * (
                avg_with_mask((real_logits - 1) ** 2, discriminator_mask)
                + avg_with_mask(fake_logits**2, discriminator_mask)
            )
            self.manual_backward(loss_d / self.accumulate_grad)

            previous = self._set_requires_grad(self.discriminator, False)
            try:
                generator_logits = self.discriminator(generated_mels)
                loss_adv = avg_with_mask(
                    (generator_logits - 1) ** 2, discriminator_mask
                )
            finally:
                self._restore_requires_grad(self.discriminator, previous)

        loss_mel = self._weighted_mel_loss(generated_mels, gt_mels, mel_mask)
        loss_generator = self.weight_mel * loss_mel + self.weight_adv * loss_adv
        self.manual_backward(loss_generator / self.accumulate_grad)
        self._accumulated_batches += 1

        self.log(
            "train/generator/loss",
            loss_generator,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/generator/loss_mel",
            loss_mel,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/batch_audio_seconds",
            audio_lengths.sum().float() / self.sampling_rate,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        if audios.is_cuda:
            self.log(
                "train/peak_cuda_memory_gib",
                torch.tensor(
                    torch.cuda.max_memory_allocated(audios.device) / (1024**3),
                    device=audios.device,
                ),
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        self.log_dict(
            self._code_usage_statistics(
                quantized.codes.detach(),
                code_lengths,
                self.quantizer.codebook_cardinality,
            ),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        if self.discriminator is not None:
            self.log(
                "train/generator/loss_adv",
                loss_adv,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "train/discriminator/loss",
                loss_d,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        if self._accumulated_batches == self.accumulate_grad or self.trainer.is_last_batch:
            self._step_accumulated_optimizers(
                tail_batches=self._accumulated_batches
            )
        return loss_generator.detach()

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        audios = batch["audios"].float()
        audio_lengths = batch["audio_lengths"]
        cut_ids = batch.get("cut_ids")
        encode_mels, gt_mels, mel_lengths, mel_mask = self._mel_inputs(
            audios, audio_lengths
        )
        encoded_features = self._encode_mels(encode_mels, mel_mask)
        reconstructed_features = self.quantized_projection(
            self.quantizer(self.pre_quantized_projection(encoded_features)).z
        ) * mel_mask
        generated_mels = self._decode_features(
            reconstructed_features, mel_mask, cut_ids
        )
        loss_mel = avg_with_mask((generated_mels - gt_mels).abs(), mel_mask)
        self.log(
            "val_loss",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=audios.shape[0],
        )

        if (
            batch_idx != 0
            or self.vocoder is None
            or not isinstance(self.logger, TensorBoardLogger)
        ):
            return loss_mel
        oracle_audios = self.vocoder(gt_mels)
        generated_audios = self.vocoder(generated_mels)
        audio_len = int(audio_lengths[0].item())
        mel_len = int(mel_lengths[0].item())
        image_mels = plot_mel(
            [gt_mels[0, :, :mel_len], generated_mels[0, :, :mel_len]],
            ["Ground-Truth", "Generated"],
        )
        self.logger.experiment.add_figure(
            f"sample-{batch_idx}/mels", image_mels, global_step=self.global_step
        )
        for name, audio in (
            ("gt", audios[0]),
            ("gen", generated_audios[0]),
            ("oracle", oracle_audios[0]),
        ):
            self.logger.experiment.add_audio(
                f"sample-{batch_idx}/wavs/{name}",
                audio[0, :audio_len].detach().cpu().float(),
                self.global_step,
                sample_rate=self.sampling_rate,
            )
        plt.close(image_mels)
        return loss_mel

    def encode(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_features, mel_lengths = self.encode_unquantized(
            audios, audio_lengths
        )
        return self.get_indices_from_unquantized_features(
            encoded_features, mel_lengths
        )

    def decode(
        self,
        indices: torch.Tensor,
        feature_lengths: torch.Tensor,
        return_audios: bool = False,
        *,
        audio_lengths: torch.Tensor | None = None,
        cut_ids: list[str] | None = None,
    ):
        target_mel_lengths = (
            torch.div(
                audio_lengths,
                self.encode_mel_transform.hop_length,
                rounding_mode="floor",
            )
            if audio_lengths is not None
            else None
        )
        features, mel_mask = self.get_quantized_features_from_indices(
            indices, feature_lengths, target_mel_lengths=target_mel_lengths
        )
        generated_mels = self._decode_features(features, mel_mask, cut_ids)
        if not return_audios:
            return generated_mels
        if self.vocoder is None:
            raise ValueError("Vocoder is not loaded")
        generated_audios = self.vocoder(generated_mels)
        if audio_lengths is not None:
            target_length = int(audio_lengths.max().item())
            generated_audios = self._resize_time(generated_audios, target_length)
            waveform_mask = sequence_mask(
                audio_lengths, target_length
            )[:, None].to(generated_audios.dtype)
            generated_audios = generated_audios * waveform_mask
        return generated_audios, generated_mels

    def encode_unquantized(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encode_mels, _, mel_lengths, mel_mask = self._mel_inputs(
            audios, audio_lengths
        )
        return self._encode_mels(encode_mels, mel_mask), mel_lengths

    @staticmethod
    def _resize_time(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        difference = target_length - tensor.shape[-1]
        if difference > 0:
            left = difference // 2
            return F.pad(tensor, (left, difference - left))
        if difference < 0:
            crop = -difference
            left = crop // 2
            return tensor[..., left : left + target_length]
        return tensor

    def get_quantized_features_from_indices(
        self,
        indices: torch.Tensor,
        feature_lengths: torch.Tensor,
        *,
        target_mel_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indices.ndim != 3:
            raise ValueError(f"Expected indices [B,S,F], got {tuple(indices.shape)}")
        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != indices.shape[0]:
            raise ValueError("feature_lengths must have shape [B]")
        factor = math.prod(self.quantizer.downsample_factor)
        mel_lengths = (
            target_mel_lengths
            if target_mel_lengths is not None
            else feature_lengths * factor
        )
        target_length = int(mel_lengths.max().item())
        features = self.quantized_projection(
            self.quantizer.decode(indices, target_length=target_length)
        )
        mel_mask = sequence_mask(mel_lengths, target_length)[:, None].to(
            device=indices.device, dtype=features.dtype
        )
        return features * mel_mask, mel_mask

    def get_indices_from_unquantized_features(
        self, unquantized_features: torch.Tensor, mel_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor = math.prod(self.quantizer.downsample_factor)
        indices_lengths = torch.div(mel_lengths, factor, rounding_mode="floor")
        indices = self.quantizer.encode(
            self.pre_quantized_projection(unquantized_features)
        )
        if indices.shape[1] != self.quantizer.num_streams:
            raise RuntimeError(
                f"Expected {self.quantizer.num_streams} streams, got {indices.shape[1]}"
            )
        return indices, indices_lengths
