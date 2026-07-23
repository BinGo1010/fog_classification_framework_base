"""Neural modules for conditional normal forecasting and residual diagnosis."""

from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    """Conv1d with explicit left padding and no future leakage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding:
            x = nn.functional.pad(x, (self.left_padding, 0))
        return super().forward(x)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConditionalNormalPredictor(nn.Module):
    """Forecast a Gaussian non-FOG future block from a historical IMU context."""

    def __init__(
        self,
        in_channels: int = 3,
        horizon: int = 32,
        hidden_channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
        min_logvar: float = -6.0,
        max_logvar: float = 3.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.horizon = int(horizon)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                CausalResidualBlock(hidden_channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.summary = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_decoder = nn.Linear(hidden_channels, in_channels * horizon)
        self.logvar_decoder = nn.Linear(hidden_channels, in_channels * horizon)
        nn.init.zeros_(self.mean_decoder.bias)
        nn.init.constant_(self.logvar_decoder.bias, -1.5)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.blocks(self.input_projection(context))
        state = self.summary(features[:, :, -1])
        delta = self.mean_decoder(state).reshape(-1, self.in_channels, self.horizon)
        # Residual parameterisation anchors the forecast to the latest observation.
        mean = context[:, :, -1:] + delta
        logvar = self.logvar_decoder(state).reshape(-1, self.in_channels, self.horizon)
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        return mean, logvar


class SamePadResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualTCNClassifier(nn.Module):
    """Window-level binary classifier used for residual and raw baselines."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                SamePadResidualBlock(hidden_channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.projection(x))
        pooled = torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1)
        return self.head(pooled).squeeze(1)


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Elementwise heteroscedastic Gaussian NLL, excluding the constant term."""

    return 0.5 * (torch.exp(-logvar) * (target - mean).square() + logvar).mean()
