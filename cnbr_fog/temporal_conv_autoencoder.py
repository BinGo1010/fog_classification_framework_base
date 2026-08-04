"""Time-preserving convolutional autoencoders for Daphnet NBM diagnostics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Two kernel-5 convolutions with GroupNorm and a short residual path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.activation(x + y)


class ConvNormGELU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int,
    ) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class TemporalConvAutoencoder(nn.Module):
    """Skip-free TC-DAE retaining a compressed temporal axis.

    Variants match the three-round experiment plan:

    ``base``: ``[B,9,128] -> [B,32,16]``
    ``wide``: ``[B,9,128] -> [B,64,16]``
    ``long``: ``[B,9,128] -> [B,48,32]``
    """

    VARIANTS = ("base", "wide", "long")

    def __init__(
        self,
        *,
        in_channels: int = 9,
        input_samples: int = 128,
        variant: str = "base",
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        if input_samples != 128:
            raise ValueError("The preregistered TC-DAE requires 128 input samples")
        self.in_channels = int(in_channels)
        self.input_samples = int(input_samples)
        self.variant = str(variant)

        self.encoder_stage1 = nn.Sequential(
            ConvNormGELU(
                in_channels, 32, kernel_size=7, stride=2, padding=3
            ),
            ResidualConvBlock(32),
        )
        self.encoder_stage2 = nn.Sequential(
            ConvNormGELU(32, 48, kernel_size=5, stride=2, padding=2),
            ResidualConvBlock(48),
        )
        if self.variant == "long":
            self.encoder_stage3: nn.Module = nn.Identity()
            latent_channels = 48
            latent_samples = 32
        else:
            latent_channels = 64 if self.variant == "wide" else 32
            self.encoder_stage3 = ConvNormGELU(
                48,
                latent_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            )
            latent_samples = 16
        self.latent_channels = latent_channels
        self.latent_samples = latent_samples

        if self.variant == "long":
            self.decoder_stage1 = nn.Sequential(
                ConvNormGELU(48, 32, kernel_size=5, padding=2),
                ResidualConvBlock(32),
            )
            self.decoder_stage2: nn.Module = nn.Identity()
            final_input_channels = 32
            upsample_count = 2
        else:
            self.decoder_stage1 = nn.Sequential(
                ConvNormGELU(latent_channels, 48, kernel_size=5, padding=2),
                ResidualConvBlock(48),
            )
            self.decoder_stage2 = nn.Sequential(
                ConvNormGELU(48, 32, kernel_size=5, padding=2),
                ResidualConvBlock(32),
            )
            final_input_channels = 32
            upsample_count = 3
        self.upsample_count = upsample_count
        self.decoder_final = nn.Sequential(
            nn.Conv1d(final_input_channels, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, in_channels, kernel_size=1),
        )

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.in_channels,
            self.input_samples,
        ):
            raise ValueError(
                f"expected [B,{self.in_channels},{self.input_samples}], got {tuple(x.shape)}"
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.encoder_stage3(self.encoder_stage2(self.encoder_stage1(x)))

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=False)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_channels, self.latent_samples)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"expected latent [B,{expected[0]},{expected[1]}]")
        y = self.decoder_stage1(self._upsample(latent))
        if self.variant != "long":
            y = self.decoder_stage2(self._upsample(y))
        y = self.decoder_final(self._upsample(y))
        if y.shape[-1] != self.input_samples:
            raise RuntimeError(f"decoder produced {y.shape[-1]} samples")
        return y

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        return self.decode(latent), latent

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "temporal_convolutional_denoising_autoencoder",
            "variant": self.variant,
            "input_shape": ["batch", self.in_channels, self.input_samples],
            "latent_shape": ["batch", self.latent_channels, self.latent_samples],
            "latent_elements": self.latent_channels * self.latent_samples,
            "encoder_decoder_long_skip": False,
            "short_residual_blocks": True,
            "normalization": "GroupNorm",
            "activation": "GELU",
            "output_activation": None,
            "upsampling": "linear interpolation followed by convolution",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


__all__ = ["ResidualConvBlock", "TemporalConvAutoencoder"]
