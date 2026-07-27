"""RF-matched TCN-M and plain dilated 1D-CNN classifiers.

The two classifiers in this module intentionally have identical trainable
parameters, parameter names, initialization order, local receptive field, and
readout.  Their only computational-graph difference is the identity addition
inside each temporal block:

* ``tcn_m``: ``x + F(x)``;
* ``cnn_rf125``: ``F(x)``.

Both consume a four-second residual history with shape ``[batch, 9, 256]`` and
use six two-convolution blocks with dilations ``(1, 2, 4, 8, 8, 8)``.  The
local convolutional-feature receptive field is therefore 125 samples
(1.953125 seconds at 64 Hz).  The final mean/max pooling still aggregates the
full 256-sample input window.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn


CANONICAL_RF125_CLASSIFIER_NAMES: tuple[str, ...] = (
    "tcn_m",
    "cnn_rf125",
)

RF125_CLASSIFIER_DISPLAY_NAMES: Mapping[str, str] = {
    "tcn_m": "TCN-M (RF125)",
    "cnn_rf125": "Dilated 1D-CNN (RF125)",
}

DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 8, 8)
DEFAULT_KERNEL_SIZE = 3
CONVOLUTIONS_PER_BLOCK = 2
DEFAULT_SAMPLING_RATE_HZ = 64


def convolutional_receptive_field(
    dilations: Sequence[int] = DEFAULT_DILATIONS,
    *,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    convolutions_per_block: int = CONVOLUTIONS_PER_BLOCK,
) -> int:
    """Return the local receptive field of the stacked temporal convolutions."""

    values = tuple(int(value) for value in dilations)
    if not values or any(value <= 0 for value in values):
        raise ValueError("dilations must be a non-empty sequence of positive values")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if convolutions_per_block <= 0:
        raise ValueError("convolutions_per_block must be positive")
    return 1 + convolutions_per_block * (kernel_size - 1) * sum(values)


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count all model parameters, optionally restricting to trainable ones."""

    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def parameter_schema_sha256(model: nn.Module) -> str:
    """Hash parameter/buffer names, shapes, and dtypes without hashing values."""

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def conv_linear_macs_per_window(
    *,
    in_channels: int,
    input_samples: int,
    hidden_channels: int,
    dilations: Sequence[int],
    kernel_size: int,
) -> int:
    """Analytic Conv1d/Linear MAC count for one input window.

    Normalization, activation, pooling, dropout, and elementwise residual
    additions are deliberately excluded.  Dilation changes sampling locations
    but not the number of convolution MACs.
    """

    projection = input_samples * in_channels * hidden_channels
    temporal = (
        len(tuple(dilations))
        * CONVOLUTIONS_PER_BLOCK
        * input_samples
        * hidden_channels
        * hidden_channels
        * kernel_size
    )
    head = (2 * hidden_channels) * hidden_channels + hidden_channels
    return int(projection + temporal + head)


class _MatchedSamePadBlock(nn.Module):
    """Two-convolution block whose optional identity path has no parameters."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        residual_skip: bool,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.residual_skip = bool(residual_skip)
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
        transformed = self.net(x)
        return x + transformed if self.residual_skip else transformed


class RF125MatchedClassifier(nn.Module):
    """Shared implementation for the strictly RF-matched classifier pair."""

    def __init__(
        self,
        *,
        canonical_name: str,
        residual_skip: bool,
        in_channels: int = 9,
        input_samples: int = 256,
        dropout: float = 0.15,
        hidden_channels: int = 48,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        canonical_name = str(canonical_name).strip().lower()
        if canonical_name not in CANONICAL_RF125_CLASSIFIER_NAMES:
            raise ValueError(f"unknown RF125 classifier {canonical_name!r}")
        if canonical_name == "tcn_m" and not residual_skip:
            raise ValueError("tcn_m requires residual_skip=True")
        if canonical_name == "cnn_rf125" and residual_skip:
            raise ValueError("cnn_rf125 requires residual_skip=False")
        if int(in_channels) <= 0 or int(input_samples) <= 0:
            raise ValueError("in_channels and input_samples must be positive")
        if int(hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        dilation_values = tuple(int(value) for value in dilations)
        receptive_field = convolutional_receptive_field(
            dilation_values,
            kernel_size=int(kernel_size),
        )
        if len(dilation_values) != 6:
            raise ValueError("the RF125 comparison requires exactly six blocks")
        if receptive_field != 125:
            raise ValueError(
                "the RF125 comparison requires a 125-sample receptive field, "
                f"got {receptive_field}"
            )

        self.canonical_name = canonical_name
        self.residual_skip = bool(residual_skip)
        self.in_channels = int(in_channels)
        self.input_samples = int(input_samples)
        self.dropout = float(dropout)
        self.hidden_channels = int(hidden_channels)
        self.dilations = dilation_values
        self.kernel_size = int(kernel_size)
        self.receptive_field_samples = receptive_field

        # Attribute names and module construction order intentionally match
        # ResidualTCNClassifier so both arms have an identical state schema.
        self.projection = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                _MatchedSamePadBlock(
                    self.hidden_channels,
                    self.kernel_size,
                    dilation,
                    self.dropout,
                    residual_skip=self.residual_skip,
                )
                for dilation in self.dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 1),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError(
                "expected input with shape [batch, channels, samples], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("input must contain at least one window")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[2] != self.input_samples:
            raise ValueError(
                f"expected {self.input_samples} input samples, got {x.shape[2]}"
            )
        if not x.is_floating_point() or x.is_complex():
            raise TypeError("input must have a real floating-point dtype")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        features = self.blocks(self.projection(x))
        pooled = torch.cat(
            [features.mean(dim=-1), features.amax(dim=-1)],
            dim=1,
        )
        return self.head(pooled).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        macs = conv_linear_macs_per_window(
            in_channels=self.in_channels,
            input_samples=self.input_samples,
            hidden_channels=self.hidden_channels,
            dilations=self.dilations,
            kernel_size=self.kernel_size,
        )
        residual_additions = (
            len(self.dilations)
            * self.hidden_channels
            * self.input_samples
            if self.residual_skip
            else 0
        )
        total_parameters = parameter_count(self)
        trainable_parameters = parameter_count(self, trainable_only=True)
        return {
            "canonical_name": self.canonical_name,
            "display_name": RF125_CLASSIFIER_DISPLAY_NAMES[self.canonical_name],
            "class_name": type(self).__name__,
            "family": "rf125_dilated_1d_cnn",
            "in_channels": self.in_channels,
            "input_samples": self.input_samples,
            "dropout": self.dropout,
            "hidden_channels": self.hidden_channels,
            "dilations": list(self.dilations),
            "n_blocks": len(self.dilations),
            "kernel_size": self.kernel_size,
            "convolutions_per_block": CONVOLUTIONS_PER_BLOCK,
            "local_receptive_field_samples": self.receptive_field_samples,
            "local_receptive_field_seconds": (
                self.receptive_field_samples / DEFAULT_SAMPLING_RATE_HZ
            ),
            "padding": "symmetric_same_zero",
            "causal": False,
            "normalization": "BatchNorm1d",
            "activation": "GELU",
            "global_pooling": "mean_and_max_over_full_input",
            "residual_skip": self.residual_skip,
            "block_equation": "x_plus_Fx" if self.residual_skip else "Fx",
            "conv_linear_macs_per_window": macs,
            "residual_elementwise_additions_per_window": residual_additions,
            "parameter_schema_sha256": parameter_schema_sha256(self),
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "output": "binary_logit",
        }


def canonical_rf125_classifier_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"classifier name must be a string, got {type(name).__name__}")
    canonical = name.strip().lower()
    if canonical not in CANONICAL_RF125_CLASSIFIER_NAMES:
        expected = ", ".join(CANONICAL_RF125_CLASSIFIER_NAMES)
        raise ValueError(
            f"unknown RF125 classifier {name!r}; expected one of: {expected}"
        )
    return canonical


def build_rf125_classifier(
    name: str,
    *,
    in_channels: int = 9,
    input_samples: int = 256,
    dropout: float = 0.15,
    hidden_channels: int = 48,
    dilations: Sequence[int] = DEFAULT_DILATIONS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
) -> RF125MatchedClassifier:
    """Construct one arm of the strictly matched RF125 comparison."""

    canonical = canonical_rf125_classifier_name(name)
    return RF125MatchedClassifier(
        canonical_name=canonical,
        residual_skip=canonical == "tcn_m",
        in_channels=in_channels,
        input_samples=input_samples,
        dropout=dropout,
        hidden_channels=hidden_channels,
        dilations=dilations,
        kernel_size=kernel_size,
    )


def rf125_classifier_config(
    name: str,
    *,
    in_channels: int = 9,
    input_samples: int = 256,
    dropout: float = 0.15,
    **architecture_kwargs: Any,
) -> dict[str, Any]:
    """Return JSON-ready architecture metadata without advancing CPU RNG."""

    with torch.random.fork_rng(devices=[]):
        model = build_rf125_classifier(
            name,
            in_channels=in_channels,
            input_samples=input_samples,
            dropout=dropout,
            **architecture_kwargs,
        )
    config = model.architecture_config()
    config["parameter_count"] = parameter_count(model)
    config["trainable_parameter_count"] = parameter_count(
        model,
        trainable_only=True,
    )
    return config


__all__ = [
    "CANONICAL_RF125_CLASSIFIER_NAMES",
    "RF125_CLASSIFIER_DISPLAY_NAMES",
    "DEFAULT_DILATIONS",
    "DEFAULT_KERNEL_SIZE",
    "CONVOLUTIONS_PER_BLOCK",
    "RF125MatchedClassifier",
    "build_rf125_classifier",
    "canonical_rf125_classifier_name",
    "conv_linear_macs_per_window",
    "convolutional_receptive_field",
    "parameter_count",
    "parameter_schema_sha256",
    "rf125_classifier_config",
]
