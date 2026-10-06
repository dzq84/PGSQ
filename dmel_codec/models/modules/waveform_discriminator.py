"""Waveform GAN discriminators used for decoder/vocoder refinement.

The MPD/MRD architectures and LSGAN objectives follow NVIDIA BigVGAN v2.4
(MIT license), adapted here to remove its configuration-object dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        *,
        channel_multiplier: float = 1.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if period < 2:
            raise ValueError("period must be at least 2")
        self.period = int(period)
        norm = spectral_norm if use_spectral_norm else weight_norm
        channels = [
            1,
            int(32 * channel_multiplier),
            int(128 * channel_multiplier),
            int(512 * channel_multiplier),
            int(1024 * channel_multiplier),
            int(1024 * channel_multiplier),
        ]
        if min(channels) < 1:
            raise ValueError("channel_multiplier produces an empty layer")
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        (5, 1),
                        (3, 1) if index < 4 else (1, 1),
                        padding=(_padding(5), 0),
                    )
                )
                for index in range(5)
            ]
        )
        self.conv_post = norm(
            nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))
        )

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch, channels, samples = waveform.shape
        remainder = samples % self.period
        if remainder:
            waveform = F.pad(waveform, (0, self.period - remainder), mode="reflect")
            samples = waveform.shape[-1]
        hidden = waveform.view(batch, channels, samples // self.period, self.period)
        features = []
        for convolution in self.convs:
            hidden = F.leaky_relu(convolution(hidden), 0.1)
            features.append(hidden)
        hidden = self.conv_post(hidden)
        features.append(hidden)
        return torch.flatten(hidden, 1), features


class ResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolution: Sequence[int],
        *,
        channel_multiplier: float = 1.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(resolution) != 3:
            raise ValueError("resolution must contain n_fft, hop_length, win_length")
        self.n_fft, self.hop_length, self.win_length = map(int, resolution)
        if not 0 < self.hop_length <= self.win_length <= self.n_fft:
            raise ValueError(f"invalid resolution: {tuple(resolution)}")
        channels = int(32 * channel_multiplier)
        if channels < 1:
            raise ValueError("channel_multiplier produces an empty layer")
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        padding = (self.n_fft - self.hop_length) // 2
        mode = "reflect" if waveform.shape[-1] > padding else "replicate"
        waveform = F.pad(waveform, (padding, padding), mode=mode).squeeze(1)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=False,
            return_complex=True,
        )
        return spectrum.abs()

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self._spectrogram(waveform).unsqueeze(1)
        features = []
        for convolution in self.convs:
            hidden = F.leaky_relu(convolution(hidden), 0.1)
            features.append(hidden)
        hidden = self.conv_post(hidden)
        features.append(hidden)
        return torch.flatten(hidden, 1), features


class WaveformDiscriminator(nn.Module):
    """Combined BigVGAN-style multi-period and multi-resolution discriminator."""

    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        resolutions: Sequence[Sequence[int]] = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        ),
        channel_multiplier: float = 1.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if not periods or not resolutions:
            raise ValueError("periods and resolutions must both be non-empty")
        self.period_discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(
                    period,
                    channel_multiplier=channel_multiplier,
                    use_spectral_norm=use_spectral_norm,
                )
                for period in periods
            ]
        )
        self.resolution_discriminators = nn.ModuleList(
            [
                ResolutionDiscriminator(
                    resolution,
                    channel_multiplier=channel_multiplier,
                    use_spectral_norm=use_spectral_norm,
                )
                for resolution in resolutions
            ]
        )

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        logits = []
        features = []
        for discriminator in (
            *self.period_discriminators,
            *self.resolution_discriminators,
        ):
            current_logits, current_features = discriminator(waveform)
            logits.append(current_logits)
            features.append(current_features)
        return logits, features


def discriminator_lsgan_loss(
    real_logits: Sequence[torch.Tensor], fake_logits: Sequence[torch.Tensor]
) -> torch.Tensor:
    if len(real_logits) != len(fake_logits) or not real_logits:
        raise ValueError("real and fake discriminator outputs must align")
    losses = [
        (real - 1.0).square().mean() + fake.square().mean()
        for real, fake in zip(real_logits, fake_logits)
    ]
    return torch.stack(losses).mean()


def generator_lsgan_loss(fake_logits: Sequence[torch.Tensor]) -> torch.Tensor:
    if not fake_logits:
        raise ValueError("fake discriminator outputs must be non-empty")
    return torch.stack([(fake - 1.0).square().mean() for fake in fake_logits]).mean()


def feature_matching_loss(
    real_features: Sequence[Sequence[torch.Tensor]],
    fake_features: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("real and fake feature maps must align")
    losses = []
    for real_stack, fake_stack in zip(real_features, fake_features):
        if len(real_stack) != len(fake_stack):
            raise ValueError("real and fake feature-map stacks must align")
        losses.extend(
            (real.detach() - fake).abs().mean()
            for real, fake in zip(real_stack, fake_stack)
        )
    return torch.stack(losses).mean()
