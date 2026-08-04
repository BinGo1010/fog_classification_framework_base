"""Safe loading for validation-selected S01 mean forecaster artifacts.

Mean-only convergence models expose a unit-sigma compatibility adapter during
training.  That value is intentionally not a calibrated forecast scale.  This
module reconstructs the selected mean architecture and wraps it with the
train-residual fixed sigma stored in ``final_predictor.pt`` so downstream code
cannot accidentally standardize residuals with the placeholder unit scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .gru_convergence_models import (
    ClusterConditionedGRUMeanForecaster,
    GRUMeanForecaster,
    MoEGRUMeanForecaster,
)


ARTIFACT_SCHEMA_VERSION = "s01_fixed_sigma_mean_forecaster.v1"


def build_mean_model(
    model_class: str,
    constructor_kwargs: Mapping[str, Any],
) -> nn.Module:
    """Construct one supported mean model from executable artifact metadata."""

    kwargs = dict(constructor_kwargs)
    factories = {
        "GRUMeanForecaster": GRUMeanForecaster,
        "ClusterConditionedGRUMeanForecaster": (
            ClusterConditionedGRUMeanForecaster
        ),
        "MoEGRUMeanForecaster": MoEGRUMeanForecaster,
    }
    try:
        factory = factories[str(model_class)]
    except KeyError as error:
        raise ValueError(f"Unsupported artifact model_class: {model_class!r}") from error
    return factory(**kwargs)


class FixedSigmaGRUPredictor(nn.Module):
    """Combine a trained future-mean model with train-only fixed sigma.

    Inputs and targets are expected to have already been transformed by the
    artifact's Robust Scaler.  ``forward`` returns broadcast-compatible
    ``(mean, sigma)`` tensors; ``standardized_residual`` implements the exact
    clipped residual representation consumed by the downstream classifier.
    """

    def __init__(self, mean_model: nn.Module, fixed_sigma: torch.Tensor) -> None:
        super().__init__()
        if not hasattr(mean_model, "forward_mean"):
            raise TypeError("mean_model must expose forward_mean(context)")
        sigma = torch.as_tensor(fixed_sigma, dtype=torch.float32)
        if sigma.ndim != 3 or int(sigma.shape[0]) != 1:
            raise ValueError("fixed_sigma must have shape [1,channel,horizon]")
        if not torch.isfinite(sigma).all() or torch.any(sigma <= 0):
            raise ValueError("fixed_sigma must be finite and strictly positive")
        self.mean_model = mean_model
        self.register_buffer("fixed_sigma", sigma.contiguous())

    def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
        return getattr(self.mean_model, "forward_mean")(context)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward_mean(context)
        expected = (1, int(mean.shape[1]), int(mean.shape[2]))
        if tuple(self.fixed_sigma.shape) != expected:
            raise RuntimeError(
                f"fixed sigma shape {tuple(self.fixed_sigma.shape)} != {expected}"
            )
        return mean, self.fixed_sigma.expand(mean.shape[0], -1, -1)

    def standardized_residual(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        *,
        clip: float = 12.0,
    ) -> torch.Tensor:
        if float(clip) <= 0:
            raise ValueError("clip must be positive")
        mean, sigma = self(context)
        if target.shape != mean.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} != mean {tuple(mean.shape)}"
            )
        return torch.clamp((target - mean) / sigma, -float(clip), float(clip))


def load_gru_predictor_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FixedSigmaGRUPredictor:
    """Load and strictly reconstruct a ``final_predictor.pt`` artifact."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing GRU predictor artifact schema")
    model = build_mean_model(
        payload["model_class"], payload["constructor_kwargs"]
    )
    model.load_state_dict(payload["model_state"], strict=True)
    predictor = FixedSigmaGRUPredictor(model, payload["fixed_sigma"])
    predictor.to(torch.device(map_location))
    predictor.eval()
    return predictor


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "FixedSigmaGRUPredictor",
    "build_mean_model",
    "load_gru_predictor_artifact",
]
