"""Alternative window-level classifiers for residual FoG diagnosis.

All classifiers in this module consume a fixed residual history with shape
``[batch, channels, samples]`` and return one binary logit per window with
shape ``[batch]``.  The canonical registry is intentionally small and stable so
that experiment manifests can use the model names as directory identifiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


CANONICAL_CLASSIFIER_NAMES: tuple[str, ...] = (
    "mlp",
    "cnn1d",
    "gru",
    "transformer",
)

CLASSIFIER_DISPLAY_NAMES: Mapping[str, str] = {
    "mlp": "MLP",
    "cnn1d": "Multi-scale 1D-CNN",
    "gru": "GRU",
    "transformer": "Lightweight Transformer",
}


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _dropout_probability(value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {value}")
    return value


class ResidualClassifier(nn.Module, ABC):
    """Base class providing the common input contract and model metadata."""

    canonical_name: str

    def __init__(
        self,
        *,
        in_channels: int,
        input_samples: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.input_samples = _positive_int(input_samples, "input_samples")
        self.dropout = _dropout_probability(dropout)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError(
                "expected residual input with shape [batch, channels, samples], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("residual input must contain at least one window")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[2] != self.input_samples:
            raise ValueError(
                f"expected {self.input_samples} input samples, got {x.shape[2]}"
            )
        if not (x.is_floating_point() or x.is_complex()):
            raise TypeError(
                "residual input must have a floating-point dtype, "
                f"got {x.dtype}"
            )
        if x.is_complex():
            raise TypeError(f"complex residual input is not supported, got {x.dtype}")

    def _common_config(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "display_name": CLASSIFIER_DISPLAY_NAMES[self.canonical_name],
            "class_name": type(self).__name__,
            "in_channels": self.in_channels,
            "input_samples": self.input_samples,
            "dropout": self.dropout,
            "output": "binary_logit",
        }

    @abstractmethod
    def architecture_config(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the architecture."""


class ResidualMLPClassifier(ResidualClassifier):
    """Shallow global mapping over the flattened residual history."""

    canonical_name = "mlp"

    def __init__(
        self,
        in_channels: int = 9,
        input_samples: int = 256,
        dropout: float = 0.15,
        hidden_features: int = 40,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_samples=input_samples,
            dropout=dropout,
        )
        self.hidden_features = _positive_int(hidden_features, "hidden_features")
        flattened_features = self.in_channels * self.input_samples
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_features, self.hidden_features),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.network(x).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        config = self._common_config()
        config.update(
            {
                "family": "multilayer_perceptron",
                "flattened_features": self.in_channels * self.input_samples,
                "hidden_features": self.hidden_features,
                "activation": "GELU",
                "pooling": "flatten",
            }
        )
        return config


class MultiScale1DCNNClassifier(ResidualClassifier):
    """Parallel local convolutions for short and medium FoG patterns."""

    canonical_name = "cnn1d"

    def __init__(
        self,
        in_channels: int = 9,
        input_samples: int = 256,
        dropout: float = 0.15,
        branch_channels: int = 32,
        hidden_channels: int = 128,
        head_features: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 7, 15),
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_samples=input_samples,
            dropout=dropout,
        )
        self.branch_channels = _positive_int(branch_channels, "branch_channels")
        self.hidden_channels = _positive_int(hidden_channels, "hidden_channels")
        self.head_features = _positive_int(head_features, "head_features")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one kernel")
        self.kernel_sizes = tuple(
            _positive_int(kernel_size, "kernel_size")
            for kernel_size in kernel_sizes
        )
        if any(kernel_size % 2 == 0 for kernel_size in self.kernel_sizes):
            raise ValueError(
                f"kernel_sizes must be odd for same padding, got {self.kernel_sizes}"
            )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.in_channels,
                        self.branch_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.branch_channels),
                    nn.GELU(),
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        concatenated_channels = self.branch_channels * len(self.kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(
                concatenated_channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.head_features),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        features = self.fusion(features)
        pooled = torch.cat(
            [features.mean(dim=-1), features.amax(dim=-1)],
            dim=1,
        )
        return self.head(pooled).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        config = self._common_config()
        config.update(
            {
                "family": "multi_scale_1d_cnn",
                "kernel_sizes": list(self.kernel_sizes),
                "branch_channels": self.branch_channels,
                "hidden_channels": self.hidden_channels,
                "head_features": self.head_features,
                "activation": "GELU",
                "normalization": "BatchNorm1d",
                "pooling": ["temporal_mean", "temporal_max"],
            }
        )
        return config


class ResidualGRUClassifier(ResidualClassifier):
    """Unidirectional GRU encoding residual state evolution over time."""

    canonical_name = "gru"

    def __init__(
        self,
        in_channels: int = 9,
        input_samples: int = 256,
        dropout: float = 0.15,
        hidden_size: int = 96,
        num_layers: int = 2,
        head_features: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_samples=input_samples,
            dropout=dropout,
        )
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.head_features = _positive_int(head_features, "head_features")

        self.input_norm = nn.LayerNorm(self.in_channels)
        self.gru = nn.GRU(
            input_size=self.in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.state_norm = nn.LayerNorm(self.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_features),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        sequence = self.input_norm(x.transpose(1, 2))
        _, final_state = self.gru(sequence)
        state = self.state_norm(final_state[-1])
        return self.head(state).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        config = self._common_config()
        config.update(
            {
                "family": "gru",
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "bidirectional": False,
                "head_features": self.head_features,
                "activation": "GELU",
                "normalization": ["input_LayerNorm", "state_LayerNorm"],
                "pooling": "final_hidden_state",
            }
        )
        return config


class LightweightTransformerClassifier(ResidualClassifier):
    """Small Transformer encoder with a learned window-level CLS token."""

    canonical_name = "transformer"

    def __init__(
        self,
        in_channels: int = 9,
        input_samples: int = 256,
        dropout: float = 0.15,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        head_features: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_samples=input_samples,
            dropout=dropout,
        )
        self.model_dim = _positive_int(model_dim, "model_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.feedforward_dim = _positive_int(feedforward_dim, "feedforward_dim")
        self.head_features = _positive_int(head_features, "head_features")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads, got "
                f"{self.model_dim} and {self.num_heads}"
            )

        self.input_norm = nn.LayerNorm(self.in_channels)
        self.input_projection = nn.Linear(self.in_channels, self.model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.input_samples + 1, self.model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.head = nn.Sequential(
            nn.Linear(self.model_dim, self.head_features),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_features, 1),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        sequence = self.input_projection(self.input_norm(x.transpose(1, 2)))
        cls = self.cls_token.expand(sequence.shape[0], -1, -1)
        sequence = torch.cat([cls, sequence], dim=1)
        sequence = sequence + self.position_embedding
        encoded = self.encoder(sequence)
        state = self.output_norm(encoded[:, 0])
        return self.head(state).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        config = self._common_config()
        config.update(
            {
                "family": "transformer_encoder",
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "feedforward_dim": self.feedforward_dim,
                "head_features": self.head_features,
                "activation": "GELU",
                "normalization": "pre_norm",
                "position_encoding": "learned_absolute",
                "pooling": "learned_CLS_token",
            }
        )
        return config


CLASSIFIER_REGISTRY: Mapping[str, type[ResidualClassifier]] = {
    "mlp": ResidualMLPClassifier,
    "cnn1d": MultiScale1DCNNClassifier,
    "gru": ResidualGRUClassifier,
    "transformer": LightweightTransformerClassifier,
}


def canonical_classifier_name(name: str) -> str:
    """Validate and normalize a model name used in experiment directories."""

    if not isinstance(name, str):
        raise TypeError(f"classifier name must be a string, got {type(name).__name__}")
    canonical = name.strip().lower()
    if canonical not in CLASSIFIER_REGISTRY:
        expected = ", ".join(CANONICAL_CLASSIFIER_NAMES)
        raise ValueError(
            f"unknown residual classifier {name!r}; expected one of: {expected}"
        )
    return canonical


def build_residual_classifier(
    name: str,
    *,
    in_channels: int = 9,
    input_samples: int = 256,
    dropout: float = 0.15,
    **architecture_kwargs: Any,
) -> ResidualClassifier:
    """Build a registered residual classifier from its canonical name."""

    canonical = canonical_classifier_name(name)
    classifier_type = CLASSIFIER_REGISTRY[canonical]
    return classifier_type(
        in_channels=in_channels,
        input_samples=input_samples,
        dropout=dropout,
        **architecture_kwargs,
    )


def classifier_config(
    name: str,
    *,
    in_channels: int = 9,
    input_samples: int = 256,
    dropout: float = 0.15,
    **architecture_kwargs: Any,
) -> dict[str, Any]:
    """Return model metadata without advancing the caller's CPU RNG state."""

    with torch.random.fork_rng(devices=[]):
        model = build_residual_classifier(
            name,
            in_channels=in_channels,
            input_samples=input_samples,
            dropout=dropout,
            **architecture_kwargs,
        )
    config = model.architecture_config()
    config["parameter_count"] = parameter_count(model)
    return config


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count all parameters, or only trainable parameters when requested."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "CANONICAL_CLASSIFIER_NAMES",
    "CLASSIFIER_DISPLAY_NAMES",
    "CLASSIFIER_REGISTRY",
    "ResidualClassifier",
    "ResidualMLPClassifier",
    "MultiScale1DCNNClassifier",
    "ResidualGRUClassifier",
    "LightweightTransformerClassifier",
    "build_residual_classifier",
    "canonical_classifier_name",
    "classifier_config",
    "parameter_count",
]
