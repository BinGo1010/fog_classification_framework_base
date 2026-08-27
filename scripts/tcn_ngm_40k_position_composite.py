#!/usr/bin/env python
"""Position-conditioned 40k TCN normal-behavior model for 30-channel IMU data.

This NBM retains the skip-free temporal bottleneck used by ``tcn_ngm_40k`` but
avoids a single-vector decoder.  The encoder produces a sequence of 32 local
latent tokens.  A fixed low-resolution sinusoidal position code is concatenated
with those tokens before decoding, so every reconstructed time region is
conditioned on both its local latent state and its temporal position.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    TCNResidualStack,
    group_count,
)


WINDOW_SAMPLES = 128
RAW_CHANNELS = 30
STEM_CHANNELS = 30
MIDDLE_CHANNELS = 14
LATENT_CHANNELS = 16
LATENT_SAMPLES = 32
POSITION_CHANNELS = 4
POSITION_FREQUENCIES = (1.0, 2.0)
TCN_NGM_POSITION_COMPOSITE_30_PARAMETER_COUNT = 40_218


def sinusoidal_position(length: int) -> torch.Tensor:
    """Return fixed [1, 4, length] sin/cos position channels on [0, 1]."""

    if length <= 1:
        raise ValueError("position encoding requires at least two time samples")
    position = torch.linspace(0.0, 1.0, steps=length, dtype=torch.float32)
    channels: list[torch.Tensor] = []
    for frequency in POSITION_FREQUENCIES:
        phase = 2.0 * math.pi * float(frequency) * position
        channels.extend((torch.sin(phase), torch.cos(phase)))
    encoding = torch.stack(channels, dim=0).unsqueeze(0)
    if tuple(encoding.shape) != (1, POSITION_CHANNELS, length):
        raise AssertionError(f"unexpected position shape: {tuple(encoding.shape)}")
    return encoding


class PositionConditionedTCNNGM30(nn.Module):
    """Skip-free 30x128 -> 16x32 -> 30x128 temporal denoising autoencoder."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(
                RAW_CHANNELS,
                STEM_CHANNELS,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(group_count(STEM_CHANNELS), STEM_CHANNELS),
            nn.GELU(),
        )
        self.encoder_tcn = TCNResidualStack(
            STEM_CHANNELS, (1, 2), self.dropout
        )
        self.encoder_down = nn.Sequential(
            nn.Conv1d(
                STEM_CHANNELS,
                MIDDLE_CHANNELS,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(group_count(MIDDLE_CHANNELS), MIDDLE_CHANNELS),
            nn.GELU(),
        )
        self.encoder_middle_tcn = TCNResidualStack(
            MIDDLE_CHANNELS, (1, 2), self.dropout
        )
        self.to_bottleneck = nn.Sequential(
            nn.Conv1d(
                MIDDLE_CHANNELS, LATENT_CHANNELS, kernel_size=1, bias=False
            ),
            nn.GroupNorm(group_count(LATENT_CHANNELS), LATENT_CHANNELS),
            nn.GELU(),
        )

        # The only architecture change from the capacity-matched TCN NGM is
        # 16+4 rather than 16 decoder input channels.  This keeps the capacity
        # close to 40k while making local latent tokens position-aware.
        self.register_buffer(
            "decoder_position", sinusoidal_position(LATENT_SAMPLES), persistent=True
        )
        self.decoder_from_condition = nn.Sequential(
            nn.Conv1d(
                LATENT_CHANNELS + POSITION_CHANNELS,
                MIDDLE_CHANNELS,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(MIDDLE_CHANNELS), MIDDLE_CHANNELS),
            nn.GELU(),
        )
        self.decoder_to_stem = nn.Sequential(
            nn.Conv1d(
                MIDDLE_CHANNELS,
                STEM_CHANNELS,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(group_count(STEM_CHANNELS), STEM_CHANNELS),
            nn.GELU(),
        )
        self.decoder_tcn = TCNResidualStack(
            STEM_CHANNELS, (1, 2), self.dropout
        )
        self.decoder_to_output_width = nn.Sequential(
            nn.Conv1d(
                STEM_CHANNELS,
                LATENT_CHANNELS,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(group_count(LATENT_CHANNELS), LATENT_CHANNELS),
            nn.GELU(),
        )
        self.output_head = nn.Conv1d(
            LATENT_CHANNELS, RAW_CHANNELS, kernel_size=1
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_tcn(self.encoder_stem(x))
        x = self.encoder_middle_tcn(self.encoder_down(x))
        return self.to_bottleneck(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if tuple(z.shape[1:]) != (LATENT_CHANNELS, LATENT_SAMPLES):
            raise ValueError(f"expected [B,16,32] latent, got {tuple(z.shape)}")
        position = self.decoder_position.to(dtype=z.dtype).expand(z.shape[0], -1, -1)
        x = self.decoder_from_condition(torch.cat((z, position), dim=1))
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder_tcn(self.decoder_to_stem(x))
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.output_head(self.decoder_to_output_width(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (RAW_CHANNELS, WINDOW_SAMPLES)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected [B,{RAW_CHANNELS},{WINDOW_SAMPLES}], got {tuple(x.shape)}"
            )
        z = self.encode(x)
        reconstruction = self.decode(z)
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction shape {tuple(reconstruction.shape)} != {tuple(x.shape)}"
            )
        return reconstruction


def architecture_config() -> dict[str, Any]:
    """Return the serialization-stable architecture contract."""

    model = PositionConditionedTCNNGM30()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != TCN_NGM_POSITION_COMPOSITE_30_PARAMETER_COUNT:
        raise RuntimeError(
            "position-conditioned TCN NGM parameter contract changed: "
            f"{parameter_count} != {TCN_NGM_POSITION_COMPOSITE_30_PARAMETER_COUNT}"
        )
    return {
        "name": "position_conditioned_tcn_ngm40k_v1_30channel",
        "input_shape": ["B", RAW_CHANNELS, WINDOW_SAMPLES],
        "encoder": [
            "Conv1d(30,30,k=7,s=2,p=3)+GroupNorm+GELU",
            "TCNResidualBlock(30,d=1) then TCNResidualBlock(30,d=2)",
            "Conv1d(30,14,k=5,s=2,p=2)+GroupNorm+GELU",
            "TCNResidualBlock(14,d=1) then TCNResidualBlock(14,d=2)",
            "Conv1d(14,16,k=1)+GroupNorm+GELU",
        ],
        "bottleneck_shape": ["B", LATENT_CHANNELS, LATENT_SAMPLES],
        "position_conditioning": {
            "location": "concatenated with each local latent token before decoder",
            "encoding": "fixed sinusoidal sin/cos on normalized time",
            "frequencies": list(POSITION_FREQUENCIES),
            "shape": ["B", POSITION_CHANNELS, LATENT_SAMPLES],
        },
        "decoder": [
            "Concat(local latent 16, position 4) -> Conv1d(20,14,k=3,p=1)+GroupNorm+GELU",
            "linear interpolation x2",
            "Conv1d(14,30,k=5,p=2)+GroupNorm+GELU",
            "TCNResidualBlock(30,d=1) then TCNResidualBlock(30,d=2)",
            "linear interpolation x2",
            "Conv1d(30,16,k=7,p=3)+GroupNorm+GELU",
            "Conv1d(16,30,k=1), no output activation",
        ],
        "residual_block": (
            "two non-causal same-length k=3 convolutions, GroupNorm, "
            "GELU, dropout, identity residual"
        ),
        "dilations": [1, 2],
        "dropout": 0.10,
        "causal": False,
        "encoder_decoder_skip_connections": False,
        "input_output_global_residual": False,
        "teacher_forcing": False,
        "output_activation": None,
        "output_shape": ["B", RAW_CHANNELS, WINDOW_SAMPLES],
        "parameter_count": parameter_count,
    }

