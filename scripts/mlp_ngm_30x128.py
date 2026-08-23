#!/usr/bin/env python
"""Factorized MLP denoising normal-gait models for 128-sample IMU windows."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


RAW_CHANNELS = 30
WINDOW_SAMPLES = 128
LATENT_CHANNELS = 16
LATENT_SAMPLES = 32
MLP_NGM_30_PARAMETER_COUNT = 39_018
MLP_NGM_9_PARAMETER_COUNT = 38_283


class ResidualTemporalMLP(nn.Module):
    """Pre-norm residual MLP operating on the last (temporal) axis."""

    def __init__(
        self,
        width: int = LATENT_SAMPLES,
        expansion: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.dropout(self.activation(self.fc1(self.norm(x)))))
        return x + residual


class FactorizedMLPNGM(nn.Module):
    """Skip-free Cx128 -> 16x32 -> Cx128 denoising MLP autoencoder.

    Linear layers are shared across sensor axes or temporal positions. This
    preserves the channel-time structure without the parameter cost of a
    fully connected 3840-dimensional autoencoder.
    """

    def __init__(self, channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)

        # Temporal compression, shared by all 30 sensor channels.
        self.encoder_time_1 = nn.Sequential(
            nn.Linear(WINDOW_SAMPLES, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Channel compression, shared by all 64 temporal positions.
        self.encoder_channel = nn.Sequential(
            nn.Linear(self.channels, LATENT_CHANNELS),
            nn.LayerNorm(LATENT_CHANNELS),
            nn.GELU(),
        )

        self.encoder_time_2 = nn.Sequential(
            nn.Linear(64, LATENT_SAMPLES),
            nn.LayerNorm(LATENT_SAMPLES),
            nn.GELU(),
        )

        self.bottleneck = nn.Sequential(
            ResidualTemporalMLP(dropout=dropout),
            ResidualTemporalMLP(dropout=dropout),
        )

        self.decoder_time_1 = nn.Sequential(
            nn.Linear(LATENT_SAMPLES, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Channel expansion, shared by all 64 temporal positions.
        self.decoder_channel = nn.Sequential(
            nn.Linear(LATENT_CHANNELS, self.channels),
            nn.LayerNorm(self.channels),
            nn.GELU(),
        )

        # Linear reconstruction head: no output activation.
        self.output_head = nn.Linear(64, WINDOW_SAMPLES)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_time_1(x)  # [B, 30, 64]
        x = self.encoder_channel(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder_time_2(x)  # [B, 16, 32]
        return self.bottleneck(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_time_1(z)  # [B, 16, 64]
        x = self.decoder_channel(x.transpose(1, 2)).transpose(1, 2)
        return self.output_head(x)  # [B, 30, 128]

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


class FactorizedMLPNGM30(FactorizedMLPNGM):
    """Thirty-channel model used by processed_NBM_Exp."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(channels=30, dropout=dropout)


class FactorizedMLPNGM9(FactorizedMLPNGM):
    """Nine-channel model used by Daphnet processed_NBM."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__(channels=9, dropout=dropout)


def expected_parameter_count(channels: int) -> int:
    return 37_968 + 35 * int(channels)


def architecture_config(channels: int = RAW_CHANNELS) -> dict[str, Any]:
    model = FactorizedMLPNGM(channels=channels)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(channels)
    if parameter_count != expected:
        raise RuntimeError(
            f"MLP-NGM parameter contract changed: {parameter_count} "
            f"!= {expected}"
        )
    return {
        "name": f"factorized_mlp_ngm_v1_{channels}channel",
        "input_shape": ["B", channels, WINDOW_SAMPLES],
        "bottleneck_shape": ["B", LATENT_CHANNELS, LATENT_SAMPLES],
        "output_shape": ["B", channels, WINDOW_SAMPLES],
        "temporal_mlp": "128->64->32; two residual 32->128->32 blocks; 32->64->128",
        "channel_mlp": f"{channels}->16->{channels}",
        "normalization": "LayerNorm",
        "activation": "GELU",
        "dropout": 0.10,
        "encoder_decoder_skip_connections": False,
        "output_activation": None,
        "parameter_count": parameter_count,
    }


@torch.no_grad()
def reconstruct_bct(
    model: nn.Module,
    x: torch.Tensor | Any,
    device: torch.device,
    batch_size: int = 128,
) -> Any:
    """Reconstruct a NumPy-like [N,C,128] array and return float32 NumPy data."""
    import numpy as np

    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != WINDOW_SAMPLES:
        raise ValueError(f"expected [N,C,{WINDOW_SAMPLES}], got {values.shape}")
    model.eval()
    output: list[Any] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(values[start : start + batch_size])
        ).to(device, non_blocking=True)
        output.append(model(batch).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(output, axis=0))


if __name__ == "__main__":
    model = FactorizedMLPNGM30()
    probe = torch.zeros(2, RAW_CHANNELS, WINDOW_SAMPLES)
    latent = model.encode(probe)
    output = model(probe)
    print(architecture_config(30))
    print(f"latent={tuple(latent.shape)} output={tuple(output.shape)}")
