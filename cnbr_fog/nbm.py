"""Interchangeable normal-behaviour predictors with a common Gaussian output.

Every predictor implements the same contract:

``mu, sigma = model(context)``

where ``context`` is ``[batch, channel, history]`` and both outputs are
``[batch, channel, forecast_horizon]``.  ``sigma`` is a strictly positive
standard deviation, not a variance or log-variance.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .models import CausalResidualBlock


NBM_NAMES = ("persistence", "linear_ar", "gru", "tcn", "transformer")


class NormalBehaviourModel(nn.Module):
    """Base class enforcing the common ``(mu, sigma)`` forecast interface."""

    model_name = "base"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        min_log_sigma: float = -3.0,
        max_log_sigma: float = 1.5,
    ):
        super().__init__()
        if int(in_channels) <= 0 or int(horizon) <= 0:
            raise ValueError("in_channels and horizon must be positive")
        if float(min_log_sigma) >= float(max_log_sigma):
            raise ValueError("min_log_sigma must be smaller than max_log_sigma")
        self.in_channels = int(in_channels)
        self.horizon = int(horizon)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

    def _check_context(self, context: torch.Tensor) -> None:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, channel, time]")
        if int(context.shape[1]) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {context.shape[1]}"
            )
        if int(context.shape[2]) < 1:
            raise ValueError("context must contain at least one time sample")

    def _distribution(
        self,
        mean: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (mean.shape[0], self.in_channels, self.horizon)
        if tuple(mean.shape) != expected:
            raise RuntimeError(f"mean shape {tuple(mean.shape)} != {expected}")
        if tuple(log_sigma.shape) not in {
            expected,
            (1, self.in_channels, self.horizon),
        }:
            raise RuntimeError(
                "log_sigma must be batch-shaped or broadcastable from "
                f"[1,{self.in_channels},{self.horizon}], got {tuple(log_sigma.shape)}"
            )
        sigma = torch.exp(
            log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        ).expand_as(mean)
        return mean, sigma

    def model_config(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "in_channels": self.in_channels,
            "horizon": self.horizon,
            "min_log_sigma": self.min_log_sigma,
            "max_log_sigma": self.max_log_sigma,
        }


class PersistenceNBM(NormalBehaviourModel):
    """Repeat the latest observation and learn only forecast uncertainty."""

    model_name = "persistence"

    def __init__(self, in_channels: int, horizon: int, **kwargs: Any):
        super().__init__(in_channels, horizon, **kwargs)
        self.log_sigma = nn.Parameter(
            torch.full((1, self.in_channels, self.horizon), -0.75)
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        mean = context[:, :, -1:].expand(-1, -1, self.horizon)
        return self._distribution(mean, self.log_sigma)


class LinearARNBM(NormalBehaviourModel):
    """Multivariate direct linear autoregression over the recent context."""

    model_name = "linear_ar"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        ar_order: int = 32,
        **kwargs: Any,
    ):
        super().__init__(in_channels, horizon, **kwargs)
        if int(ar_order) <= 0:
            raise ValueError("ar_order must be positive")
        self.ar_order = int(ar_order)
        input_width = self.in_channels * self.ar_order
        output_width = self.in_channels * self.horizon
        self.mean_head = nn.Linear(input_width, output_width)
        self.log_sigma_head = nn.Linear(input_width, output_width)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, -0.75)
        # Start from persistence while still allowing cross-channel VAR-like
        # relationships to be learned.
        with torch.no_grad():
            for channel in range(self.in_channels):
                source = channel * self.ar_order + self.ar_order - 1
                for step in range(self.horizon):
                    target = channel * self.horizon + step
                    self.mean_head.weight[target, source] = 1.0

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        if int(context.shape[-1]) < self.ar_order:
            raise ValueError(
                f"Linear-AR needs {self.ar_order} samples, got {context.shape[-1]}"
            )
        lags = context[:, :, -self.ar_order :]
        flattened = lags.reshape(lags.shape[0], -1)
        mean = self.mean_head(flattened).reshape(
            -1, self.in_channels, self.horizon
        )
        log_sigma = self.log_sigma_head(flattened).reshape(
            -1, self.in_channels, self.horizon
        )
        return self._distribution(mean, log_sigma)

    def model_config(self) -> dict[str, Any]:
        return {**super().model_config(), "ar_order": self.ar_order}


class GRUNBM(NormalBehaviourModel):
    """GRU context encoder with direct Gaussian multi-horizon decoding."""

    model_name = "gru"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(in_channels, horizon, **kwargs)
        if int(hidden_channels) <= 0 or int(num_layers) <= 0:
            raise ValueError("hidden_channels and num_layers must be positive")
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.encoder = nn.GRU(
            input_size=self.in_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.summary = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.decoder = nn.Linear(
            self.hidden_channels, 2 * self.in_channels * self.horizon
        )
        nn.init.zeros_(self.decoder.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        _, hidden = self.encoder(context.transpose(1, 2))
        state = self.summary(hidden[-1])
        decoded = self.decoder(state).reshape(
            -1, 2, self.in_channels, self.horizon
        )
        mean = context[:, :, -1:] + decoded[:, 0]
        return self._distribution(mean, decoded[:, 1])

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class TCNNBM(NormalBehaviourModel):
    """Causal temporal convolutional normal-behaviour predictor."""

    model_name = "tcn"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(in_channels, horizon, **kwargs)
        self.hidden_channels = int(hidden_channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.input_projection = nn.Sequential(
            nn.Conv1d(
                self.in_channels, self.hidden_channels, kernel_size=1, bias=False
            ),
            nn.GroupNorm(1, self.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                CausalResidualBlock(
                    self.hidden_channels,
                    self.kernel_size,
                    dilation,
                    self.dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.summary = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.decoder = nn.Linear(
            self.hidden_channels, 2 * self.in_channels * self.horizon
        )
        nn.init.zeros_(self.decoder.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        features = self.blocks(self.input_projection(context))
        state = self.summary(features[:, :, -1])
        decoded = self.decoder(state).reshape(
            -1, 2, self.in_channels, self.horizon
        )
        mean = context[:, :, -1:] + decoded[:, 0]
        return self._distribution(mean, decoded[:, 1])

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "hidden_channels": self.hidden_channels,
            "dilations": list(self.dilations),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
        }


class TransformerNBM(NormalBehaviourModel):
    """Transformer context encoder with direct Gaussian decoding."""

    model_name = "transformer"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_context_samples: int = 2048,
        **kwargs: Any,
    ):
        super().__init__(in_channels, horizon, **kwargs)
        if int(d_model) % int(nhead):
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.max_context_samples = int(max_context_samples)
        self.input_projection = nn.Linear(self.in_channels, self.d_model)
        self.register_buffer(
            "positional_encoding",
            self._sinusoidal_encoding(self.max_context_samples, self.d_model),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.decoder = nn.Linear(
            self.d_model, 2 * self.in_channels * self.horizon
        )
        nn.init.zeros_(self.decoder.bias)

    @staticmethod
    def _sinusoidal_encoding(length: int, width: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(length, width, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        length = int(context.shape[-1])
        if length > self.max_context_samples:
            raise ValueError(
                f"context length {length} exceeds {self.max_context_samples}"
            )
        sequence = self.input_projection(context.transpose(1, 2))
        sequence = sequence + self.positional_encoding[:, :length].to(
            dtype=sequence.dtype
        )
        state = self.final_norm(self.encoder(sequence)[:, -1])
        decoded = self.decoder(state).reshape(
            -1, 2, self.in_channels, self.horizon
        )
        mean = context[:, :, -1:] + decoded[:, 0]
        return self._distribution(mean, decoded[:, 1])

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_context_samples": self.max_context_samples,
        }


def canonical_nbm_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "persistence": "persistence",
        "persist": "persistence",
        "linear": "linear_ar",
        "linear_ar": "linear_ar",
        "ar": "linear_ar",
        "gru": "gru",
        "gru_nbm": "gru",
        "tcn": "tcn",
        "tcn_nbm": "tcn",
        "transformer": "transformer",
        "transformer_nbm": "transformer",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown NBM {name!r}; choose from {NBM_NAMES}")
    return aliases[normalized]


def build_nbm(
    name: str,
    in_channels: int,
    horizon: int,
    *,
    hidden_channels: int = 48,
    dropout: float = 0.1,
    linear_ar_order: int = 32,
    gru_layers: int = 1,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    transformer_ffn: int = 128,
    max_context_samples: int = 2048,
) -> NormalBehaviourModel:
    """Construct one predictor from the shared experiment configuration."""

    canonical = canonical_nbm_name(name)
    common = {"in_channels": in_channels, "horizon": horizon}
    if canonical == "persistence":
        return PersistenceNBM(**common)
    if canonical == "linear_ar":
        return LinearARNBM(**common, ar_order=linear_ar_order)
    if canonical == "gru":
        return GRUNBM(
            **common,
            hidden_channels=hidden_channels,
            num_layers=gru_layers,
            dropout=dropout,
        )
    if canonical == "tcn":
        return TCNNBM(
            **common,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    if canonical == "transformer":
        return TransformerNBM(
            **common,
            d_model=hidden_channels,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ffn,
            dropout=dropout,
            max_context_samples=max_context_samples,
        )
    raise AssertionError(canonical)


def gaussian_nll_sigma(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL (without the constant) for a positive standard deviation."""

    if target.shape != mean.shape or target.shape != sigma.shape:
        raise ValueError(
            f"target/mean/sigma shapes differ: {target.shape}, {mean.shape}, "
            f"{sigma.shape}"
        )
    if not torch.all(sigma > 0):
        raise ValueError("sigma must be strictly positive")
    return (torch.log(sigma) + 0.5 * ((target - mean) / sigma).square()).mean()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
