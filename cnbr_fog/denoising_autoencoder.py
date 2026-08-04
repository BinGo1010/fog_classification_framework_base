"""TCN denoising autoencoder components for normal-IMU reconstruction.

The implementation follows ``NonFoG_Denoising_Autoencoder_65Hz_Training_Framework.md``
while keeping the sequence length configurable.  The Daphnet experiment uses
128 samples (two seconds at the dataset's actual 64 Hz sampling rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ChannelZScoreScaler:
    """Training-fold channel z-score parameters without value clipping."""

    mean: np.ndarray
    std: np.ndarray
    epsilon: float = 1e-8

    @classmethod
    def fit_channel_time(
        cls,
        windows: np.ndarray,
        *,
        epsilon: float = 1e-8,
    ) -> "ChannelZScoreScaler":
        """Fit over ``[window, channel, time]`` clean-normal windows."""

        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("windows must be non-empty [window, channel, time]")
        if not np.isfinite(values).all():
            raise ValueError("windows contain NaN or Inf")
        if not np.isfinite(epsilon) or float(epsilon) <= 0:
            raise ValueError("epsilon must be finite and positive")
        mean = np.mean(values, axis=(0, 2))
        std = np.std(values, axis=(0, 2), ddof=0)
        if np.any(std <= 0) or not np.isfinite(std).all():
            raise ValueError("channel standard deviation must be finite and positive")
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            epsilon=float(epsilon),
        )

    def transform_channel_time(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim < 2 or values.shape[-2] != self.mean.size:
            raise ValueError("values must end in [channel, time]")
        shape = (1,) * (values.ndim - 2) + (self.mean.size, 1)
        mean = self.mean.reshape(shape)
        denominator = (self.std + self.epsilon).reshape(shape)
        return ((values - mean) / denominator).astype(np.float32, copy=False)

    def inverse_channel_time(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim < 2 or values.shape[-2] != self.mean.size:
            raise ValueError("values must end in [channel, time]")
        shape = (1,) * (values.ndim - 2) + (self.mean.size, 1)
        mean = self.mean.reshape(shape)
        denominator = (self.std + self.epsilon).reshape(shape)
        return (values * denominator + mean).astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": float(self.epsilon),
            "definition": (
                "population mean/std over all cells of clean-normal training "
                "target windows, separately for each channel"
            ),
            "value_clipping": None,
        }


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TCNResidualBlock(nn.Module):
    """Two same-length dilated convolutions with a short residual skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        dropout: float,
        kernel_size: int = 3,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric same padding")
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(
            _group_count(out_channels, group_norm_groups), out_channels
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(
            _group_count(out_channels, group_norm_groups), out_channels
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.dropout(self.activation(self.norm1(self.conv1(x))))
        y = self.activation(self.norm2(self.conv2(y)))
        return residual + y


class TCNDenoisingAutoencoder(nn.Module):
    """Bottlenecked, skip-free TCN denoising autoencoder."""

    def __init__(
        self,
        *,
        in_channels: int = 9,
        input_samples: int = 128,
        latent_dim: int = 128,
        dropout: float = 0.1,
        residual_kernel_size: int = 3,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if input_samples < 16:
            raise ValueError("input_samples must be at least 16")
        if in_channels <= 0 or latent_dim <= 0:
            raise ValueError("in_channels and latent_dim must be positive")
        self.in_channels = int(in_channels)
        self.input_samples = int(input_samples)
        self.latent_dim = int(latent_dim)
        self.dropout_probability = float(dropout)
        self.residual_kernel_size = int(residual_kernel_size)
        self.group_norm_groups = int(group_norm_groups)

        self.input_conv = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.input_norm = nn.GroupNorm(_group_count(32, group_norm_groups), 32)
        self.activation = nn.GELU()

        self.encoder_blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    32,
                    32,
                    dilation=1,
                    dropout=dropout,
                    kernel_size=residual_kernel_size,
                    group_norm_groups=group_norm_groups,
                ),
                TCNResidualBlock(
                    32,
                    64,
                    dilation=2,
                    dropout=dropout,
                    kernel_size=residual_kernel_size,
                    group_norm_groups=group_norm_groups,
                ),
                TCNResidualBlock(
                    64,
                    128,
                    dilation=4,
                    dropout=dropout,
                    kernel_size=residual_kernel_size,
                    group_norm_groups=group_norm_groups,
                ),
            ]
        )
        self.downsample = nn.ModuleList(
            [
                nn.Conv1d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1),
            ]
        )
        lengths = [self.input_samples]
        for _ in range(3):
            lengths.append((lengths[-1] + 2 - (4 - 1) - 1) // 2 + 1)
        self.encoder_lengths = tuple(lengths)
        encoded_samples = self.encoder_lengths[-1]
        self.to_latent = nn.Linear(128 * encoded_samples, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * encoded_samples)

        self.decoder_blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    128,
                    64,
                    dilation=4,
                    dropout=dropout,
                    kernel_size=residual_kernel_size,
                    group_norm_groups=group_norm_groups,
                ),
                TCNResidualBlock(
                    64,
                    32,
                    dilation=2,
                    dropout=dropout,
                    kernel_size=residual_kernel_size,
                    group_norm_groups=group_norm_groups,
                ),
                TCNResidualBlock(
                    32,
                    32,
                    dilation=1,
                    dropout=dropout,
                    kernel_size=residual_kernel_size,
                    group_norm_groups=group_norm_groups,
                ),
            ]
        )
        self.output_conv = nn.Conv1d(32, in_channels, kernel_size=7, padding=3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (
            self.in_channels,
            self.input_samples,
        ):
            raise ValueError(
                f"expected [batch,{self.in_channels},{self.input_samples}], "
                f"got {tuple(x.shape)}"
            )
        y = self.activation(self.input_norm(self.input_conv(x)))
        for block, downsample in zip(self.encoder_blocks, self.downsample):
            y = downsample(block(y))
        return self.to_latent(y.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        y = self.from_latent(latent).reshape(
            latent.shape[0], 128, self.encoder_lengths[-1]
        )
        for block, target_length in zip(
            self.decoder_blocks,
            reversed(self.encoder_lengths[:-1]),
        ):
            y = F.interpolate(
                y,
                size=int(target_length),
                mode="linear",
                align_corners=False,
            )
            y = block(y)
        return self.output_conv(y)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        return self.decode(latent), latent

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "tcn_denoising_autoencoder",
            "input_shape": ["batch", self.in_channels, self.input_samples],
            "output_shape": ["batch", self.in_channels, self.input_samples],
            "input_conv": {"out_channels": 32, "kernel_size": 7, "padding": 3},
            "encoder_channels": [32, 64, 128],
            "encoder_dilations": [1, 2, 4],
            "encoder_lengths": list(self.encoder_lengths),
            "downsample": {"kernel_size": 4, "stride": 2, "padding": 1},
            "latent_dim": self.latent_dim,
            "decoder_channels": [64, 32, 32],
            "decoder_dilations": [4, 2, 1],
            "decoder_lengths": list(reversed(self.encoder_lengths[:-1])),
            "residual_kernel_size": self.residual_kernel_size,
            "normalization": "GroupNorm",
            "maximum_group_norm_groups": self.group_norm_groups,
            "activation": "GELU",
            "dropout": self.dropout_probability,
            "encoder_decoder_skip_connections": False,
            "output_activation": None,
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }


@dataclass(frozen=True)
class CorruptionConfig:
    time_mask_probability: float = 0.50
    channel_mask_probability: float = 0.20
    gaussian_noise_probability: float = 0.20
    clean_probability: float = 0.10
    time_mask_min_samples: int = 7
    time_mask_max_samples: int = 26
    gaussian_std_min: float = 0.01
    gaussian_std_max: float = 0.05

    def validate(self, *, channels: int, samples: int) -> None:
        probabilities = (
            self.time_mask_probability,
            self.channel_mask_probability,
            self.gaussian_noise_probability,
            self.clean_probability,
        )
        if any(value < 0 for value in probabilities) or not np.isclose(
            sum(probabilities), 1.0
        ):
            raise ValueError("corruption probabilities must be non-negative and sum to 1")
        if channels <= 0 or samples <= 0:
            raise ValueError("channels and samples must be positive")
        if not 1 <= self.time_mask_min_samples <= self.time_mask_max_samples <= samples:
            raise ValueError("invalid time-mask length range")
        if not 0 < self.gaussian_std_min <= self.gaussian_std_max:
            raise ValueError("invalid Gaussian noise standard-deviation range")

    def as_dict(self) -> dict[str, Any]:
        return {
            "exclusive_per_window": True,
            "time_mask_probability": self.time_mask_probability,
            "time_mask_min_samples": self.time_mask_min_samples,
            "time_mask_max_samples": self.time_mask_max_samples,
            "channel_mask_probability": self.channel_mask_probability,
            "channel_mask_count": 1,
            "gaussian_noise_probability": self.gaussian_noise_probability,
            "gaussian_std_min": self.gaussian_std_min,
            "gaussian_std_max": self.gaussian_std_max,
            "clean_probability": self.clean_probability,
            "training_only": True,
        }


def corrupt_batch(
    clean: torch.Tensor,
    config: CorruptionConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exactly one corruption mode per window.

    Mode IDs are 0=time mask, 1=channel mask, 2=Gaussian noise, 3=clean.
    """

    if clean.ndim != 3:
        raise ValueError("clean must have shape [batch, channel, time]")
    batch, channels, samples = clean.shape
    config.validate(channels=channels, samples=samples)
    output = clean.clone()
    draw = torch.rand(batch, device=clean.device, generator=generator)
    boundaries = torch.tensor(
        [
            config.time_mask_probability,
            config.time_mask_probability + config.channel_mask_probability,
            config.time_mask_probability
            + config.channel_mask_probability
            + config.gaussian_noise_probability,
        ],
        device=clean.device,
        dtype=draw.dtype,
    )
    modes = torch.bucketize(draw, boundaries)

    for row in torch.nonzero(modes == 0, as_tuple=False).flatten().tolist():
        length = int(
            torch.randint(
                config.time_mask_min_samples,
                config.time_mask_max_samples + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        start = int(
            torch.randint(
                0,
                samples - length + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        output[row, :, start : start + length] = 0.0

    for row in torch.nonzero(modes == 1, as_tuple=False).flatten().tolist():
        channel = int(
            torch.randint(
                0,
                channels,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        output[row, channel, :] = 0.0

    noise_rows = torch.nonzero(modes == 2, as_tuple=False).flatten()
    if noise_rows.numel():
        count = int(noise_rows.numel())
        std = torch.empty(count, 1, 1, device=clean.device).uniform_(
            config.gaussian_std_min,
            config.gaussian_std_max,
            generator=generator,
        )
        noise = torch.randn(
            count,
            channels,
            samples,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        output[noise_rows] = output[noise_rows] + std.to(clean.dtype) * noise
    return output, modes


def dae_combined_loss(
    reconstruction: torch.Tensor,
    clean: torch.Tensor,
    *,
    difference_weight: float = 0.2,
    frequency_weight: float = 0.1,
    huber_beta: float = 1.0,
    n_fft: int = 64,
    win_length: int = 64,
    hop_length: int = 16,
) -> dict[str, torch.Tensor]:
    """Time Huber + first-difference Huber + log-magnitude STFT L1."""

    if reconstruction.shape != clean.shape or clean.ndim != 3:
        raise ValueError("reconstruction and clean must share [batch,channel,time]")
    if clean.shape[-1] < win_length or n_fft < win_length:
        raise ValueError("STFT window does not fit the sequence")
    # Losses are evaluated in float32 even when the model forward uses AMP.
    predicted = reconstruction.float()
    target = clean.float()
    time_loss = F.smooth_l1_loss(predicted, target, beta=huber_beta)
    predicted_diff = predicted[..., 1:] - predicted[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    difference_loss = F.smooth_l1_loss(
        predicted_diff, target_diff, beta=huber_beta
    )

    window = torch.hann_window(win_length, device=clean.device, dtype=torch.float32)
    flat_predicted = predicted.flatten(0, 1)
    flat_target = target.flatten(0, 1)
    predicted_stft = torch.stft(
        flat_predicted,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    target_stft = torch.stft(
        flat_target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    frequency_loss = F.l1_loss(
        torch.log1p(predicted_stft.abs()),
        torch.log1p(target_stft.abs()),
    )
    total = time_loss + difference_weight * difference_loss + frequency_weight * frequency_loss
    return {
        "total": total,
        "time": time_loss,
        "difference": difference_loss,
        "frequency": frequency_loss,
    }


__all__ = [
    "ChannelZScoreScaler",
    "CorruptionConfig",
    "TCNDenoisingAutoencoder",
    "TCNResidualBlock",
    "corrupt_batch",
    "dae_combined_loss",
]
