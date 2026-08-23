#!/usr/bin/env python
"""Capacity-matched TCN denoising normal-gait models near 40k parameters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
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
STEM_CHANNELS = 30
LATENT_CHANNELS = 16
LATENT_SAMPLES = 32
TCN_NGM_9_PARAMETER_COUNT = 39_987
TCN_NGM_30_PARAMETER_COUNT = 40_050


class CapacityMatchedTCNNGM(nn.Module):
    """Skip-free Cx128 -> 16x32 -> Cx128 TCN denoising autoencoder."""

    def __init__(
        self,
        channels: int,
        middle_channels: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if channels <= 0 or middle_channels <= 0:
            raise ValueError("channels and middle_channels must be positive")
        self.channels = int(channels)
        self.middle_channels = int(middle_channels)
        self.dropout = float(dropout)

        self.encoder_stem = nn.Sequential(
            nn.Conv1d(
                self.channels,
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
                self.middle_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(
                group_count(self.middle_channels), self.middle_channels
            ),
            nn.GELU(),
        )
        self.encoder_middle_tcn = TCNResidualStack(
            self.middle_channels, (1, 2), self.dropout
        )
        self.to_bottleneck = nn.Sequential(
            nn.Conv1d(
                self.middle_channels, LATENT_CHANNELS, kernel_size=1, bias=False
            ),
            nn.GroupNorm(group_count(LATENT_CHANNELS), LATENT_CHANNELS),
            nn.GELU(),
        )

        self.decoder_from_bottleneck = nn.Sequential(
            nn.Conv1d(
                LATENT_CHANNELS,
                self.middle_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                group_count(self.middle_channels), self.middle_channels
            ),
            nn.GELU(),
        )
        self.decoder_to_stem = nn.Sequential(
            nn.Conv1d(
                self.middle_channels,
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
            LATENT_CHANNELS, self.channels, kernel_size=1
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_tcn(self.encoder_stem(x))  # [B,30,64]
        x = self.encoder_middle_tcn(self.encoder_down(x))
        return self.to_bottleneck(x)  # [B,16,32]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_from_bottleneck(z)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder_tcn(self.decoder_to_stem(x))
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.output_head(self.decoder_to_output_width(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.channels, WINDOW_SAMPLES)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected [B,{self.channels},{WINDOW_SAMPLES}], got {tuple(x.shape)}"
            )
        z = self.encode(x)
        if tuple(z.shape[1:]) != (LATENT_CHANNELS, LATENT_SAMPLES):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        reconstruction = self.decode(z)
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction shape {tuple(reconstruction.shape)} "
                f"!= {tuple(x.shape)}"
            )
        return reconstruction


class CapacityMatchedTCNNGM9(CapacityMatchedTCNNGM):
    """Daphnet configuration: 9 channels, 39,987 parameters."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(channels=9, middle_channels=20, dropout=dropout)


class CapacityMatchedTCNNGM30(CapacityMatchedTCNNGM):
    """processed_NBM_Exp configuration: 30 channels, 40,050 parameters."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(channels=30, middle_channels=14, dropout=dropout)


def model_spec(channels: int) -> tuple[type[CapacityMatchedTCNNGM], int, int]:
    if channels == 9:
        return CapacityMatchedTCNNGM9, 20, TCN_NGM_9_PARAMETER_COUNT
    if channels == 30:
        return CapacityMatchedTCNNGM30, 14, TCN_NGM_30_PARAMETER_COUNT
    raise ValueError(f"unsupported capacity-matched channel count: {channels}")


def architecture_config(channels: int) -> dict[str, Any]:
    model_class, middle_channels, expected = model_spec(channels)
    model = model_class()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected:
        raise RuntimeError(
            f"TCN-NGM parameter contract changed: {parameter_count} != {expected}"
        )
    return {
        "name": f"capacity_matched_tcn_ngm_v1_{channels}channel",
        "input_shape": ["B", channels, WINDOW_SAMPLES],
        "encoder": [
            f"Conv1d({channels},30,k=7,s=2,p=3)+GroupNorm+GELU",
            "TCNResidualBlock(30,d=1) then TCNResidualBlock(30,d=2)",
            f"Conv1d(30,{middle_channels},k=5,s=2,p=2)+GroupNorm+GELU",
            (
                f"TCNResidualBlock({middle_channels},d=1) then "
                f"TCNResidualBlock({middle_channels},d=2)"
            ),
            f"Conv1d({middle_channels},16,k=1)+GroupNorm+GELU",
        ],
        "bottleneck_shape": ["B", LATENT_CHANNELS, LATENT_SAMPLES],
        "decoder": [
            f"Conv1d(16,{middle_channels},k=3,p=1)+GroupNorm+GELU",
            "linear interpolation x2",
            f"Conv1d({middle_channels},30,k=5,p=2)+GroupNorm+GELU",
            "TCNResidualBlock(30,d=1) then TCNResidualBlock(30,d=2)",
            "linear interpolation x2",
            "Conv1d(30,16,k=7,p=3)+GroupNorm+GELU",
            f"Conv1d(16,{channels},k=1), no output activation",
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
        "output_shape": ["B", channels, WINDOW_SAMPLES],
        "parameter_count": parameter_count,
    }


@torch.no_grad()
def reconstruct_bct(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != WINDOW_SAMPLES:
        raise ValueError(f"expected [N,C,{WINDOW_SAMPLES}], got {values.shape}")
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(values[start : start + batch_size])
        ).to(device, non_blocking=True)
        outputs.append(model(batch).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(outputs, axis=0))


if __name__ == "__main__":
    for channels, model_class in (
        (9, CapacityMatchedTCNNGM9),
        (30, CapacityMatchedTCNNGM30),
    ):
        model = model_class()
        probe = torch.zeros(2, channels, WINDOW_SAMPLES)
        print(architecture_config(channels))
        print(
            f"channels={channels} latent={tuple(model.encode(probe).shape)} "
            f"output={tuple(model(probe).shape)}"
        )
