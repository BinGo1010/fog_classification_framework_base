"""Core utilities for the Daphnet GRU-H200 feasibility experiment.

This module deliberately does not depend on any of the completed, audited
experiment runners.  It provides the small, testable building blocks needed by
the four-phase H200 experiment while leaving the source GRU horizon suite and
its recorded implementation hashes untouched.

Array conventions
-----------------
Forecast arrays use ``[window, channel, lead]``.  A model ensemble adds a
leading ``model`` dimension.  Classifier inputs use
``[batch, channel, sample]``.  ``sigma`` always denotes a positive standard
deviation, never a variance or log-variance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


H200_CHANNELS = 9
H200_CONTEXT_SAMPLES = 128
H200_HORIZON_SAMPLES = 128
H200_HISTORY_SAMPLES = 256
RAW6_SAMPLES = 384
STANDARDIZED_ERROR_CLIP = 12.0
RF125_DILATIONS = (1, 2, 4, 8, 8, 8)
RF125_KERNEL_SIZE = 3


@dataclass(frozen=True)
class H200ArmSpec:
    """Immutable input and classifier contract for one feasibility arm."""

    name: str
    display_name: str
    classifier_kind: str
    raw_channels: int
    normality_channels: int
    input_samples: int
    normality_source: str

    @property
    def is_dual_branch(self) -> bool:
        return self.classifier_kind == "dual"


H200_ARM_REGISTRY: Mapping[str, H200ArmSpec] = MappingProxyType(
    {
        "raw4": H200ArmSpec(
            name="raw4",
            display_name="Raw4",
            classifier_kind="single",
            raw_channels=H200_CHANNELS,
            normality_channels=0,
            input_samples=H200_HISTORY_SAMPLES,
            normality_source="none",
        ),
        "raw6": H200ArmSpec(
            name="raw6",
            display_name="Raw6",
            classifier_kind="single",
            raw_channels=H200_CHANNELS,
            normality_channels=0,
            input_samples=RAW6_SAMPLES,
            normality_source="none",
        ),
        "normality": H200ArmSpec(
            name="normality",
            display_name="z4 + log-sigma4",
            classifier_kind="single",
            raw_channels=0,
            normality_channels=2 * H200_CHANNELS,
            input_samples=H200_HISTORY_SAMPLES,
            normality_source="observed",
        ),
        "raw4_zero": H200ArmSpec(
            name="raw4_zero",
            display_name="Raw4 + zero normality branch",
            classifier_kind="dual",
            raw_channels=H200_CHANNELS,
            normality_channels=2 * H200_CHANNELS,
            input_samples=H200_HISTORY_SAMPLES,
            normality_source="zero",
        ),
        "raw4_normality": H200ArmSpec(
            name="raw4_normality",
            display_name="Raw4 + z4 + log-sigma4",
            classifier_kind="dual",
            raw_channels=H200_CHANNELS,
            normality_channels=2 * H200_CHANNELS,
            input_samples=H200_HISTORY_SAMPLES,
            normality_source="observed",
        ),
    }
)
H200_ARM_NAMES = tuple(H200_ARM_REGISTRY)
# Short compatibility name used by the experiment runner.
ARM_SPECS = H200_ARM_REGISTRY


def get_h200_arm(name: str) -> H200ArmSpec:
    """Resolve one arm by its stable, lower-case identifier."""

    key = str(name).strip().lower()
    if key not in H200_ARM_REGISTRY:
        raise ValueError(
            f"unknown H200 arm {name!r}; choose from {H200_ARM_NAMES}"
        )
    return H200_ARM_REGISTRY[key]


def rf125_receptive_field(
    dilations: Sequence[int] = RF125_DILATIONS,
    kernel_size: int = RF125_KERNEL_SIZE,
) -> int:
    """Return the receptive field of two same-pad convolutions per block."""

    values = tuple(int(value) for value in dilations)
    if not values or any(value <= 0 for value in values):
        raise ValueError("dilations must contain positive integers")
    if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return 1 + 2 * (int(kernel_size) - 1) * sum(values)


class _RF125ResidualBlock(nn.Module):
    """The parameterised temporal block shared by every experiment arm."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((int(kernel_size) - 1) * int(dilation)) // 2
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


class RF125TCNFeatureEncoder(nn.Module):
    """TCN-M feature encoder with a fixed 125-sample local receptive field."""

    def __init__(
        self,
        in_channels: int,
        input_samples: int,
        hidden_channels: int = 48,
        dropout: float = 0.15,
        dilations: Sequence[int] = RF125_DILATIONS,
        kernel_size: int = RF125_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        if min(int(in_channels), int(input_samples), int(hidden_channels)) <= 0:
            raise ValueError(
                "in_channels, input_samples, and hidden_channels must be positive"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        dilation_values = tuple(int(value) for value in dilations)
        receptive_field = rf125_receptive_field(
            dilation_values, kernel_size=int(kernel_size)
        )
        if len(dilation_values) != 6 or receptive_field != 125:
            raise ValueError(
                "RF125 encoder requires six blocks and a 125-sample "
                f"receptive field, got {len(dilation_values)} and "
                f"{receptive_field}"
            )

        self.in_channels = int(in_channels)
        self.input_samples = int(input_samples)
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.dilations = dilation_values
        self.kernel_size = int(kernel_size)
        self.receptive_field_samples = receptive_field
        self.output_features = 2 * self.hidden_channels

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
                _RF125ResidualBlock(
                    self.hidden_channels,
                    dilation,
                    self.kernel_size,
                    self.dropout,
                )
                for dilation in self.dilations
            ]
        )

    def _validate_input(self, x: torch.Tensor, *, name: str = "x") -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                f"{name} must have shape [batch, channel, sample], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError(f"{name} must contain at least one window")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"{name} has {x.shape[1]} channels; expected {self.in_channels}"
            )
        if int(x.shape[2]) != self.input_samples:
            raise ValueError(
                f"{name} has {x.shape[2]} samples; expected {self.input_samples}"
            )
        if not x.is_floating_point() or x.is_complex():
            raise TypeError(f"{name} must have a real floating-point dtype")
        if not torch.isfinite(x).all().item():
            raise ValueError(f"{name} contains NaN or Inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        features = self.blocks(self.projection(x))
        return torch.cat(
            [features.mean(dim=-1), features.amax(dim=-1)], dim=1
        )

    def architecture_config(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "input_samples": self.input_samples,
            "hidden_channels": self.hidden_channels,
            "dropout": self.dropout,
            "dilations": list(self.dilations),
            "kernel_size": self.kernel_size,
            "receptive_field_samples": self.receptive_field_samples,
            "output_features": self.output_features,
        }


class SingleBranchRF125Classifier(nn.Module):
    """Single RF125 encoder followed by the common binary classification head."""

    def __init__(
        self,
        in_channels: int,
        input_samples: int,
        hidden_channels: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = RF125TCNFeatureEncoder(
            in_channels=in_channels,
            input_samples=input_samples,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_features, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "kind": "single",
            "encoder": self.encoder.architecture_config(),
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }


class DualBranchRF125Classifier(nn.Module):
    """Matched Raw/normality RF125 encoders with late feature fusion.

    Both ``raw4_zero`` and ``raw4_normality`` are instances of this exact
    class with identical constructor arguments.  The zero control therefore
    changes only the values entering ``normality_encoder`` and not capacity.
    """

    def __init__(
        self,
        raw_channels: int = H200_CHANNELS,
        normality_channels: int = 2 * H200_CHANNELS,
        input_samples: int = H200_HISTORY_SAMPLES,
        hidden_channels: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.raw_encoder = RF125TCNFeatureEncoder(
            in_channels=raw_channels,
            input_samples=input_samples,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.normality_encoder = RF125TCNFeatureEncoder(
            in_channels=normality_channels,
            input_samples=input_samples,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        fused_features = (
            self.raw_encoder.output_features
            + self.normality_encoder.output_features
        )
        self.head = nn.Sequential(
            nn.Linear(fused_features, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        raw: torch.Tensor,
        normality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The runner stores every arm as one ndarray/tensor.  Accept its packed
        # [raw, normality] form as well as explicit branch tensors.
        if normality is None:
            if not isinstance(raw, torch.Tensor) or raw.ndim != 3:
                raise ValueError(
                    "packed dual input must have shape [batch, channel, sample]"
                )
            raw_channels = self.raw_encoder.in_channels
            expected_channels = (
                raw_channels + self.normality_encoder.in_channels
            )
            if int(raw.shape[1]) != expected_channels:
                raise ValueError(
                    f"packed dual input has {raw.shape[1]} channels; "
                    f"expected {expected_channels}"
                )
            raw, normality = (
                raw[:, :raw_channels, :],
                raw[:, raw_channels:, :],
            )
        self.raw_encoder._validate_input(raw, name="raw")
        assert normality is not None
        self.normality_encoder._validate_input(
            normality, name="normality"
        )
        if int(raw.shape[0]) != int(normality.shape[0]):
            raise ValueError("raw and normality batch sizes differ")
        raw_features = self.raw_encoder(raw)
        normality_features = self.normality_encoder(normality)
        fused = torch.cat([raw_features, normality_features], dim=1)
        return self.head(fused).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "kind": "dual",
            "raw_encoder": self.raw_encoder.architecture_config(),
            "normality_encoder": self.normality_encoder.architecture_config(),
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }


def build_h200_arm_classifier(
    arm: str,
    *,
    hidden_channels: int = 48,
    dropout: float = 0.15,
) -> nn.Module:
    """Build the classifier prescribed by one H200 arm."""

    spec = get_h200_arm(arm)
    if spec.is_dual_branch:
        return DualBranchRF125Classifier(
            raw_channels=spec.raw_channels,
            normality_channels=spec.normality_channels,
            input_samples=spec.input_samples,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    in_channels = spec.raw_channels or spec.normality_channels
    return SingleBranchRF125Classifier(
        in_channels=in_channels,
        input_samples=spec.input_samples,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )


def build_classifier(
    arm: str,
    hidden_channels: int = 48,
    dropout: float = 0.15,
) -> nn.Module:
    """Runner-facing alias for :func:`build_h200_arm_classifier`."""

    return build_h200_arm_classifier(
        arm, hidden_channels=hidden_channels, dropout=dropout
    )


def _validate_arm_array(
    name: str,
    values: np.ndarray,
    *,
    channels: int,
    samples: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if array.ndim != 3 or array.shape[1:] != (channels, samples):
        raise ValueError(
            f"{name} must have shape [N,{channels},{samples}], "
            f"got {array.shape}"
        )
    if array.shape[0] <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def build_arm_inputs(
    raw4: np.ndarray,
    raw6: np.ndarray,
    z4: np.ndarray,
    log_sigma4: np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialise all five classifier arms on aligned endpoints.

    Dual arms are packed along channels as ``[raw9, normality18]``.  The
    dual-branch classifier splits this tensor internally, which keeps the
    runner's dataset/loader interface identical for all arms.
    """

    raw4_array = _validate_arm_array(
        "raw4",
        raw4,
        channels=H200_CHANNELS,
        samples=H200_HISTORY_SAMPLES,
    )
    raw6_array = _validate_arm_array(
        "raw6", raw6, channels=H200_CHANNELS, samples=RAW6_SAMPLES
    )
    z_array = _validate_arm_array(
        "z4",
        z4,
        channels=H200_CHANNELS,
        samples=H200_HISTORY_SAMPLES,
    )
    log_sigma_array = _validate_arm_array(
        "log_sigma4",
        log_sigma4,
        channels=H200_CHANNELS,
        samples=H200_HISTORY_SAMPLES,
    )
    window_counts = {
        raw4_array.shape[0],
        raw6_array.shape[0],
        z_array.shape[0],
        log_sigma_array.shape[0],
    }
    if len(window_counts) != 1:
        raise ValueError("all arm inputs must use the same endpoint count")
    if np.any(np.abs(z_array) > STANDARDIZED_ERROR_CLIP + 1e-6):
        raise ValueError("z4 exceeds the preregistered clipping range")
    normality = np.concatenate(
        [z_array, log_sigma_array], axis=1, dtype=np.float32
    )
    zero_normality = np.zeros_like(normality)
    return {
        "raw4": np.ascontiguousarray(raw4_array),
        "raw6": np.ascontiguousarray(raw6_array),
        "normality": np.ascontiguousarray(normality),
        "raw4_zero": np.ascontiguousarray(
            np.concatenate(
                [raw4_array, zero_normality], axis=1, dtype=np.float32
            )
        ),
        "raw4_normality": np.ascontiguousarray(
            np.concatenate(
                [raw4_array, normality], axis=1, dtype=np.float32
            )
        ),
    }


def _require_float32_3d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [window, channel, lead]")
    if min(array.shape) <= 0:
        raise ValueError(f"{name} must have non-empty dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def validate_forecast_primitives(
    raw: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    *,
    expected_channels: int = H200_CHANNELS,
    expected_horizon: int = H200_HORIZON_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate aligned target, Gaussian mean, and positive sigma arrays.

    A sigma with one window is accepted and broadcast across windows.  This is
    useful for the persistence baseline's train-calibrated
    ``[1, channel, lead]`` uncertainty.
    """

    raw_array = _require_float32_3d("raw", raw)
    mean_array = _require_float32_3d("mean", mean)
    sigma_array = _require_float32_3d("sigma", sigma)
    expected_tail = (int(expected_channels), int(expected_horizon))
    if raw_array.shape[1:] != expected_tail:
        raise ValueError(
            f"raw tail shape {raw_array.shape[1:]} != {expected_tail}"
        )
    if mean_array.shape != raw_array.shape:
        raise ValueError("mean must have exactly the same shape as raw")
    if sigma_array.shape[1:] != expected_tail or sigma_array.shape[0] not in {
        1,
        raw_array.shape[0],
    }:
        raise ValueError(
            "sigma must have shape [1,C,H] or the same shape as raw"
        )
    if np.any(sigma_array <= 0):
        raise ValueError("sigma must be strictly positive")
    if sigma_array.shape[0] == 1 and raw_array.shape[0] != 1:
        sigma_array = np.broadcast_to(sigma_array, raw_array.shape)
    return raw_array, mean_array, sigma_array


def derive_forecast_primitives(
    raw: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    *,
    z_clip: float = STANDARDIZED_ERROR_CLIP,
    expected_channels: int = H200_CHANNELS,
    expected_horizon: int = H200_HORIZON_SAMPLES,
) -> dict[str, np.ndarray]:
    """Derive signed error, clipped z, log-sigma, and Gaussian NLL maps."""

    if not np.isfinite(z_clip) or float(z_clip) <= 0:
        raise ValueError("z_clip must be finite and positive")
    raw_array, mean_array, sigma_array = validate_forecast_primitives(
        raw,
        mean,
        sigma,
        expected_channels=expected_channels,
        expected_horizon=expected_horizon,
    )
    raw64 = raw_array.astype(np.float64, copy=False)
    mean64 = mean_array.astype(np.float64, copy=False)
    sigma64 = sigma_array.astype(np.float64, copy=False)
    error64 = raw64 - mean64
    z_unclipped64 = error64 / sigma64
    log_sigma64 = np.log(sigma64)
    arrays = {
        "raw": raw_array,
        "mean": mean_array,
        "mu": mean_array,
        "sigma": sigma_array,
        "error": error64,
        "z": np.clip(z_unclipped64, -z_clip, z_clip),
        "log_sigma": log_sigma64,
        "gaussian_nll": log_sigma64 + 0.5 * np.square(z_unclipped64),
    }
    return {
        key: np.ascontiguousarray(value, dtype=np.float32)
        for key, value in arrays.items()
    }


def gaussian_moment_match(
    means: np.ndarray,
    sigmas: np.ndarray,
    *,
    min_sigma: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match an equally weighted ensemble of Gaussian predictions.

    ``means`` and ``sigmas`` must have shape ``[model, ...]``.  The returned
    arrays have the remaining dimensions.  The variance includes both each
    model's aleatoric variance and between-model mean disagreement.
    """

    mean_array = np.asarray(means)
    sigma_array = np.asarray(sigmas)
    if mean_array.shape != sigma_array.shape or mean_array.ndim < 2:
        raise ValueError(
            "means and sigmas must have the same [model, ...] shape"
        )
    if mean_array.shape[0] <= 0:
        raise ValueError("the ensemble must contain at least one model")
    if not np.issubdtype(mean_array.dtype, np.floating) or not np.issubdtype(
        sigma_array.dtype, np.floating
    ):
        raise TypeError("means and sigmas must be floating-point arrays")
    if (
        not np.isfinite(mean_array).all()
        or not np.isfinite(sigma_array).all()
        or np.any(sigma_array <= 0)
    ):
        raise ValueError("ensemble parameters must be finite and sigma > 0")
    if not np.isfinite(min_sigma) or float(min_sigma) <= 0:
        raise ValueError("min_sigma must be finite and positive")

    mean64 = mean_array.astype(np.float64, copy=False)
    sigma64 = sigma_array.astype(np.float64, copy=False)
    matched_mean = mean64.mean(axis=0)
    second_moment = np.mean(
        np.square(sigma64) + np.square(mean64), axis=0
    )
    variance = np.maximum(
        second_moment - np.square(matched_mean), float(min_sigma) ** 2
    )
    return (
        np.ascontiguousarray(matched_mean, dtype=np.float32),
        np.ascontiguousarray(np.sqrt(variance), dtype=np.float32),
    )


def _validate_gaussian_arrays(
    mean: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean_array = np.asarray(mean)
    sigma_array = np.asarray(sigma)
    if mean_array.shape != sigma_array.shape or mean_array.ndim < 2:
        raise ValueError("mean and sigma must have the same shape with ndim >= 2")
    if not np.issubdtype(mean_array.dtype, np.floating) or not np.issubdtype(
        sigma_array.dtype, np.floating
    ):
        raise TypeError("mean and sigma must be floating-point arrays")
    if (
        not np.isfinite(mean_array).all()
        or not np.isfinite(sigma_array).all()
        or np.any(sigma_array <= 0)
    ):
        raise ValueError("mean/sigma must be finite and sigma > 0")
    return mean_array, sigma_array


def _channel_parameters(
    center: np.ndarray,
    scale: np.ndarray,
    channels: int,
    ndim: int,
) -> tuple[np.ndarray, np.ndarray]:
    center_array = np.asarray(center, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if center_array.shape != (channels,) or scale_array.shape != (channels,):
        raise ValueError(
            f"center and scale must each have shape ({channels},)"
        )
    if (
        not np.isfinite(center_array).all()
        or not np.isfinite(scale_array).all()
        or np.any(scale_array <= 0)
    ):
        raise ValueError("center/scale must be finite and scale > 0")
    shape = [1] * ndim
    shape[-2] = channels
    return center_array.reshape(shape), scale_array.reshape(shape)


def scaled_gaussian_to_physical(
    mean_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo one channel-wise scaler for a Gaussian forecast.

    The channel axis is the penultimate axis.  No clipping is applied to the
    mean because clipping a Gaussian parameter is not an invertible transform.
    """

    mean_array, sigma_array = _validate_gaussian_arrays(
        mean_scaled, sigma_scaled
    )
    channels = int(mean_array.shape[-2])
    center_view, scale_view = _channel_parameters(
        center, scale, channels, mean_array.ndim
    )
    physical_mean = mean_array.astype(np.float64) * scale_view + center_view
    physical_sigma = sigma_array.astype(np.float64) * scale_view
    return (
        np.ascontiguousarray(physical_mean, dtype=np.float32),
        np.ascontiguousarray(physical_sigma, dtype=np.float32),
    )


def physical_gaussian_to_scaled(
    mean_physical: np.ndarray,
    sigma_physical: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a physical-unit Gaussian in a channel-wise scaler space."""

    mean_array, sigma_array = _validate_gaussian_arrays(
        mean_physical, sigma_physical
    )
    channels = int(mean_array.shape[-2])
    center_view, scale_view = _channel_parameters(
        center, scale, channels, mean_array.ndim
    )
    scaled_mean = (mean_array.astype(np.float64) - center_view) / scale_view
    scaled_sigma = sigma_array.astype(np.float64) / scale_view
    return (
        np.ascontiguousarray(scaled_mean, dtype=np.float32),
        np.ascontiguousarray(scaled_sigma, dtype=np.float32),
    )


def ensemble_scaled_gaussians_to_outer(
    means_scaled: np.ndarray,
    sigmas_scaled: np.ndarray,
    inner_centers: np.ndarray,
    inner_scales: np.ndarray,
    outer_center: np.ndarray,
    outer_scale: np.ndarray,
    *,
    min_sigma: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match inner-model forecasts and convert to the outer scale."""

    mean_array = np.asarray(means_scaled)
    sigma_array = np.asarray(sigmas_scaled)
    if mean_array.shape != sigma_array.shape or mean_array.ndim < 3:
        raise ValueError(
            "scaled ensemble arrays must share [model,...,channel,lead] shape"
        )
    model_count = int(mean_array.shape[0])
    channels = int(mean_array.shape[-2])
    centers = np.asarray(inner_centers)
    scales = np.asarray(inner_scales)
    if centers.shape != (model_count, channels) or scales.shape != (
        model_count,
        channels,
    ):
        raise ValueError(
            "inner_centers and inner_scales must have shape [model, channel]"
        )

    physical_means: list[np.ndarray] = []
    physical_sigmas: list[np.ndarray] = []
    for model_index in range(model_count):
        physical_mean, physical_sigma = scaled_gaussian_to_physical(
            mean_array[model_index],
            sigma_array[model_index],
            centers[model_index],
            scales[model_index],
        )
        physical_means.append(physical_mean)
        physical_sigmas.append(physical_sigma)
    matched_mean, matched_sigma = gaussian_moment_match(
        np.stack(physical_means),
        np.stack(physical_sigmas),
        min_sigma=min_sigma,
    )
    return physical_gaussian_to_scaled(
        matched_mean,
        matched_sigma,
        outer_center,
        outer_scale,
    )


@dataclass(frozen=True)
class SubjectCrossFitFold:
    fold_index: int
    train_subjects: tuple[str, ...]
    heldout_subjects: tuple[str, ...]


@dataclass(frozen=True)
class SubjectCrossFitPlan:
    scheme: str
    subjects: tuple[str, ...]
    folds: tuple[SubjectCrossFitFold, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "subjects": list(self.subjects),
            "folds": [asdict(fold) for fold in self.folds],
        }


def validate_subject_crossfit_plan(plan: SubjectCrossFitPlan) -> None:
    """Reject subject overlap, omissions, duplicates, and wrong split sizes."""

    subjects = tuple(plan.subjects)
    if len(subjects) != 6 or len(set(subjects)) != 6:
        raise ValueError("the outer fold must contain six unique train subjects")
    expected = set(subjects)
    heldout_counts = {subject: 0 for subject in subjects}
    expected_folds = 3 if plan.scheme == "3fold" else 6
    expected_heldout = 2 if plan.scheme == "3fold" else 1
    if plan.scheme not in {"3fold", "loto"}:
        raise ValueError("scheme must be '3fold' or 'loto'")
    if len(plan.folds) != expected_folds:
        raise ValueError(f"{plan.scheme} requires {expected_folds} folds")
    for expected_index, fold in enumerate(plan.folds):
        train = tuple(fold.train_subjects)
        heldout = tuple(fold.heldout_subjects)
        if int(fold.fold_index) != expected_index:
            raise ValueError("cross-fit fold indices must be consecutive")
        if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
            raise ValueError("a cross-fit split contains duplicate subjects")
        if set(train) & set(heldout):
            raise ValueError("a held-out subject appears in the inner train set")
        if set(train) | set(heldout) != expected:
            raise ValueError("inner train and held-out subjects do not partition six")
        if len(heldout) != expected_heldout:
            raise ValueError("cross-fit held-out split has the wrong size")
        if len(train) != 6 - expected_heldout:
            raise ValueError("cross-fit train split has the wrong size")
        for subject in heldout:
            heldout_counts[subject] += 1
    if set(heldout_counts.values()) != {1}:
        raise ValueError("each outer-train subject must be held out exactly once")


def build_subject_crossfit_plan(
    subjects: Sequence[str], scheme: str = "3fold"
) -> SubjectCrossFitPlan:
    """Build deterministic 3-fold or leave-one-training-subject-out splits."""

    ordered = tuple(str(subject).strip() for subject in subjects)
    if len(ordered) != 6 or any(not subject for subject in ordered):
        raise ValueError("exactly six non-empty outer-train subjects are required")
    if len(set(ordered)) != 6:
        raise ValueError("outer-train subjects must be unique")
    normalized = str(scheme).strip().lower().replace("-", "")
    aliases = {
        "3fold": "3fold",
        "threefold": "3fold",
        "loto": "loto",
        "leaveonetrainingsubjectout": "loto",
    }
    if normalized not in aliases:
        raise ValueError("scheme must be '3fold' or 'loto'")
    canonical = aliases[normalized]
    if canonical == "3fold":
        heldout_groups = (
            ordered[0:2],
            ordered[2:4],
            ordered[4:6],
        )
    else:
        heldout_groups = tuple((subject,) for subject in ordered)
    folds = tuple(
        SubjectCrossFitFold(
            fold_index=index,
            train_subjects=tuple(
                subject for subject in ordered if subject not in set(heldout)
            ),
            heldout_subjects=tuple(heldout),
        )
        for index, heldout in enumerate(heldout_groups)
    )
    plan = SubjectCrossFitPlan(
        scheme=canonical, subjects=ordered, folds=folds
    )
    validate_subject_crossfit_plan(plan)
    return plan


def _paired_metric_values(
    candidate: Mapping[str, float] | Sequence[float] | np.ndarray,
    reference: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if isinstance(candidate, Mapping) or isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping) or not isinstance(
            reference, Mapping
        ):
            raise TypeError("candidate and reference must both be mappings")
        subjects = tuple(str(key) for key in candidate)
        if set(subjects) != {str(key) for key in reference}:
            raise ValueError("paired subject mappings have different keys")
        candidate_array = np.asarray(
            [candidate[key] for key in candidate], dtype=np.float64
        )
        reference_by_string = {str(key): value for key, value in reference.items()}
        reference_array = np.asarray(
            [reference_by_string[subject] for subject in subjects],
            dtype=np.float64,
        )
    else:
        candidate_array = np.asarray(candidate, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
        subjects = tuple(f"subject_{index}" for index in range(len(candidate_array)))
    if (
        candidate_array.ndim != 1
        or reference_array.ndim != 1
        or candidate_array.shape != reference_array.shape
        or candidate_array.size == 0
    ):
        raise ValueError("paired metrics must be non-empty equal-length vectors")
    if not np.isfinite(candidate_array).all() or not np.isfinite(
        reference_array
    ).all():
        raise ValueError("paired metrics contain NaN or Inf")
    return candidate_array, reference_array, subjects


def paired_subject_bootstrap(
    candidate: Mapping[str, float] | Sequence[float] | np.ndarray,
    reference: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    samples: int = 100_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of the subject-mean candidate-minus-reference delta."""

    candidate_array, reference_array, subjects = _paired_metric_values(
        candidate, reference
    )
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    delta = candidate_array - reference_array
    rng = np.random.default_rng(int(seed))
    draw_indices = rng.integers(
        0, len(delta), size=(int(samples), len(delta)), endpoint=False
    )
    bootstrap_means = delta[draw_indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(
        bootstrap_means, [alpha, 1.0 - alpha]
    )
    return {
        "subjects": list(subjects),
        "n_subjects": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
        "positive_subjects": int(np.count_nonzero(delta > 0)),
        "zero_subjects": int(np.count_nonzero(delta == 0)),
        "per_subject_delta": delta.tolist(),
    }


@dataclass(frozen=True)
class GoNoGoThresholds:
    minimum_pr_auc_delta: float = 0.02
    minimum_direction_subjects: int = 5
    conditional_minimum_direction_subjects: int = 5
    conditional_minimum_secondary_delta: float = 0.0
    maximum_recall_decline: float = 0.05
    maximum_false_alarm_ratio: float = 1.2
    stop_ci_upper: float = 0.01
    stop_maximum_positive_subjects: int = 2

    def __post_init__(self) -> None:
        unit_interval = {
            "minimum_pr_auc_delta": self.minimum_pr_auc_delta,
            "conditional_minimum_secondary_delta": (
                self.conditional_minimum_secondary_delta
            ),
            "maximum_recall_decline": self.maximum_recall_decline,
            "stop_ci_upper": self.stop_ci_upper,
        }
        for name, value in unit_interval.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        for name, value in (
            ("minimum_direction_subjects", self.minimum_direction_subjects),
            (
                "conditional_minimum_direction_subjects",
                self.conditional_minimum_direction_subjects,
            ),
        ):
            if int(value) < 1 or int(value) > 8:
                raise ValueError(f"{name} must lie in [1, 8]")
        if (
            int(self.stop_maximum_positive_subjects) < 0
            or int(self.stop_maximum_positive_subjects) >= 8
        ):
            raise ValueError("stop_maximum_positive_subjects must lie in [0, 7]")
        if (
            not math.isfinite(float(self.maximum_false_alarm_ratio))
            or float(self.maximum_false_alarm_ratio) < 1.0
        ):
            raise ValueError("maximum_false_alarm_ratio must be finite and >= 1")


def _metric_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def evaluate_go_no_go(
    *,
    subject_ids: Sequence[str],
    fusion_pr_auc: Sequence[float] | np.ndarray,
    raw6_pr_auc: Sequence[float] | np.ndarray,
    zero_pr_auc: Sequence[float] | np.ndarray,
    normality_pr_auc: Sequence[float] | np.ndarray,
    prevalence: Sequence[float] | np.ndarray,
    fusion_recall: Sequence[float] | np.ndarray,
    raw6_recall: Sequence[float] | np.ndarray,
    fusion_false_alarms_per_hour: Sequence[float] | np.ndarray,
    raw6_false_alarms_per_hour: Sequence[float] | np.ndarray,
    fusion_event_sensitivity: Sequence[float] | np.ndarray | None = None,
    raw6_event_sensitivity: Sequence[float] | np.ndarray | None = None,
    thresholds: GoNoGoThresholds = GoNoGoThresholds(),
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Apply the preregistered Phase-2 engineering Go/No-Go gates.

    The function expects one value per held-out subject and intentionally
    requires all eight main Daphnet subjects.  Seeds are not treated as
    independent observations.
    """

    subjects = tuple(str(subject).strip() for subject in subject_ids)
    if len(subjects) != 8 or any(not subject for subject in subjects):
        raise ValueError("Go/No-Go requires eight non-empty subject IDs")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Go/No-Go subject IDs must be unique and aligned")

    named = {
        "fusion_pr_auc": fusion_pr_auc,
        "raw6_pr_auc": raw6_pr_auc,
        "zero_pr_auc": zero_pr_auc,
        "normality_pr_auc": normality_pr_auc,
        "prevalence": prevalence,
        "fusion_recall": fusion_recall,
        "raw6_recall": raw6_recall,
        "fusion_false_alarms_per_hour": fusion_false_alarms_per_hour,
        "raw6_false_alarms_per_hour": raw6_false_alarms_per_hour,
    }
    vectors = {name: _metric_vector(name, value) for name, value in named.items()}
    lengths = {len(value) for value in vectors.values()}
    if lengths != {8}:
        raise ValueError("Go/No-Go requires exactly eight aligned subjects")
    for name in (
        "fusion_pr_auc",
        "raw6_pr_auc",
        "zero_pr_auc",
        "normality_pr_auc",
        "prevalence",
        "fusion_recall",
        "raw6_recall",
    ):
        if np.any(vectors[name] < 0) or np.any(vectors[name] > 1):
            raise ValueError(f"{name} must lie in [0, 1]")
    for name in ("fusion_false_alarms_per_hour", "raw6_false_alarms_per_hour"):
        if np.any(vectors[name] < 0):
            raise ValueError(f"{name} must be non-negative")
    if (fusion_event_sensitivity is None) != (raw6_event_sensitivity is None):
        raise ValueError(
            "fusion and raw6 event sensitivity must be supplied together"
        )

    fusion = vectors["fusion_pr_auc"]
    raw6 = vectors["raw6_pr_auc"]
    zero = vectors["zero_pr_auc"]
    normality = vectors["normality_pr_auc"]
    prevalence_values = vectors["prevalence"]
    practical = paired_subject_bootstrap(
        fusion,
        raw6,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    mechanism_delta = float(np.mean(fusion - zero))
    normality_direction = int(np.count_nonzero(normality > prevalence_values))
    recall_decline = float(
        np.mean(vectors["raw6_recall"] - vectors["fusion_recall"])
    )
    reference_fa = float(vectors["raw6_false_alarms_per_hour"].mean())
    candidate_fa = float(
        vectors["fusion_false_alarms_per_hour"].mean()
    )
    fa_ratio = (
        candidate_fa / reference_fa
        if reference_fa > 0
        else (1.0 if candidate_fa == 0 else float("inf"))
    )

    event_decline: float | None = None
    event_improvement: np.ndarray | None = None
    if fusion_event_sensitivity is not None:
        fusion_event = _metric_vector(
            "fusion_event_sensitivity", fusion_event_sensitivity
        )
        raw6_event = _metric_vector(
            "raw6_event_sensitivity", raw6_event_sensitivity
        )
        if len(fusion_event) != 8 or len(raw6_event) != 8:
            raise ValueError("event sensitivity requires eight subjects")
        if (
            np.any(fusion_event < 0)
            or np.any(fusion_event > 1)
            or np.any(raw6_event < 0)
            or np.any(raw6_event > 1)
        ):
            raise ValueError("event sensitivity must lie in [0, 1]")
        event_decline = float(np.mean(raw6_event - fusion_event))
        event_improvement = fusion_event - raw6_event

    recall_improvement = (
        vectors["fusion_recall"] - vectors["raw6_recall"]
    )
    false_alarm_improvement = (
        vectors["raw6_false_alarms_per_hour"]
        - vectors["fusion_false_alarms_per_hour"]
    )
    secondary_improvements: dict[str, dict[str, Any]] = {}
    for name, delta in (
        ("fog_recall", recall_improvement),
        ("false_alarms_per_hour", false_alarm_improvement),
        ("event_sensitivity", event_improvement),
    ):
        if delta is None:
            continue
        mean_delta = float(np.mean(delta))
        positive_subjects = int(np.count_nonzero(delta > 0.0))
        secondary_improvements[name] = {
            "mean_improvement": mean_delta,
            "positive_subjects": positive_subjects,
            "passes_conditional_direction": bool(
                mean_delta
                > thresholds.conditional_minimum_secondary_delta
                and positive_subjects
                >= thresholds.conditional_minimum_direction_subjects
            ),
        }
    has_consistent_secondary_improvement = any(
        value["passes_conditional_direction"]
        for value in secondary_improvements.values()
    )

    strong_checks = {
        "normality_above_prevalence_macro": float(normality.mean())
        > float(prevalence_values.mean()),
        "normality_direction": normality_direction
        >= thresholds.minimum_direction_subjects,
        "pr_auc_delta": practical["mean_delta"]
        >= thresholds.minimum_pr_auc_delta,
        "pr_auc_direction": practical["positive_subjects"]
        >= thresholds.minimum_direction_subjects,
        "better_than_zero_control": mechanism_delta > 0.0,
        "recall_safe": recall_decline <= thresholds.maximum_recall_decline,
        "false_alarm_safe": fa_ratio
        <= thresholds.maximum_false_alarm_ratio,
    }
    if event_decline is not None:
        strong_checks["event_sensitivity_safe"] = (
            event_decline <= thresholds.maximum_recall_decline
        )

    conditional_checks = {
        "positive_subthreshold_pr_auc_delta": (
            0.0 < practical["mean_delta"] < thresholds.minimum_pr_auc_delta
        ),
        "better_than_zero_control": mechanism_delta > 0.0,
        "improvement_not_single_subject": practical["positive_subjects"]
        > thresholds.stop_maximum_positive_subjects,
        "consistent_secondary_improvement": (
            has_consistent_secondary_improvement
        ),
        "recall_safe": recall_decline <= thresholds.maximum_recall_decline,
        "false_alarm_safe": fa_ratio <= thresholds.maximum_false_alarm_ratio,
    }
    if event_decline is not None:
        conditional_checks["event_sensitivity_safe"] = (
            event_decline <= thresholds.maximum_recall_decline
        )

    stop_reasons: list[str] = []
    if practical["ci_high"] < thresholds.stop_ci_upper:
        stop_reasons.append("paired_ci_upper_below_minimum_useful_delta")
    if practical["positive_subjects"] <= thresholds.stop_maximum_positive_subjects:
        stop_reasons.append("improvement_in_at_most_two_subjects")
    if practical["mean_delta"] <= 0.0:
        stop_reasons.append("no_positive_pr_auc_delta")
    if mechanism_delta <= 0.0:
        stop_reasons.append("not_better_than_zero_control")
    if recall_decline > thresholds.maximum_recall_decline:
        stop_reasons.append("recall_decline_exceeds_limit")
    if fa_ratio > thresholds.maximum_false_alarm_ratio:
        stop_reasons.append("false_alarm_ratio_exceeds_limit")
    if event_decline is not None and event_decline > thresholds.maximum_recall_decline:
        stop_reasons.append("event_sensitivity_decline_exceeds_limit")

    if all(strong_checks.values()):
        decision = "strong_go"
    elif all(conditional_checks.values()) and not stop_reasons:
        decision = "conditional_go"
    else:
        decision = "stop"
        if not conditional_checks["positive_subthreshold_pr_auc_delta"]:
            stop_reasons.append("outside_conditional_pr_auc_delta_range")
        if not conditional_checks["consistent_secondary_improvement"]:
            stop_reasons.append("no_consistent_secondary_metric_improvement")
    stop_reasons = list(dict.fromkeys(stop_reasons))
    return {
        "decision": decision,
        "subject_ids": list(subjects),
        "strong_checks": strong_checks,
        "conditional_checks": conditional_checks,
        "secondary_improvements": secondary_improvements,
        "stop_reasons": stop_reasons,
        "practical_pr_auc": practical,
        "mechanism_pr_auc_delta": mechanism_delta,
        "normality_above_prevalence_subjects": normality_direction,
        "normality_macro_pr_auc": float(normality.mean()),
        "macro_prevalence": float(prevalence_values.mean()),
        "recall_decline": recall_decline,
        "event_sensitivity_decline": event_decline,
        "false_alarm_ratio": float(fa_ratio),
        "thresholds": asdict(thresholds),
    }


def _forecast_metric_block(
    target: np.ndarray, mean: np.ndarray, sigma: np.ndarray
) -> dict[str, float]:
    target64 = target.astype(np.float64, copy=False)
    mean64 = mean.astype(np.float64, copy=False)
    sigma64 = sigma.astype(np.float64, copy=False)
    error = target64 - mean64
    z = error / sigma64
    return {
        "nll": float(np.mean(np.log(sigma64) + 0.5 * np.square(z))),
        "nll_with_gaussian_constant": float(
            np.mean(
                np.log(sigma64)
                + 0.5 * np.square(z)
                + 0.5 * np.log(2.0 * np.pi)
            )
        ),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "coverage_1sigma": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(z) <= 2.0)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "z_q025": float(np.quantile(z, 0.025)),
        "z_median": float(np.quantile(z, 0.5)),
        "z_q975": float(np.quantile(z, 0.975)),
        "z_abs_gt_12_fraction": float(
            np.mean(np.abs(z) > STANDARDIZED_ERROR_CLIP)
        ),
    }


def optimal_cross_correlation_lag(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    max_lag: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best Pearson-correlation lag for every window/channel.

    Inputs use ``[window, channel, sample]``.  A positive lag compares
    ``target[..., lag:]`` with ``prediction[..., :-lag]``; equivalently, the
    prediction leads the target and must be shifted right by ``lag`` samples.
    Ties prefer zero and then the smallest absolute lag.
    """

    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    if (
        target_array.shape != prediction_array.shape
        or target_array.ndim != 3
        or min(target_array.shape) <= 0
    ):
        raise ValueError(
            "target and prediction must share non-empty [window,channel,time] shape"
        )
    if not np.issubdtype(target_array.dtype, np.floating) or not np.issubdtype(
        prediction_array.dtype, np.floating
    ):
        raise TypeError("target and prediction must be floating-point arrays")
    if not np.isfinite(target_array).all() or not np.isfinite(
        prediction_array
    ).all():
        raise ValueError("target or prediction contains NaN/Inf")
    max_lag = int(max_lag)
    if max_lag < 0 or 2 * max_lag >= target_array.shape[-1]:
        raise ValueError("max_lag must leave a non-empty common central support")

    target64 = target_array.astype(np.float64, copy=False)
    prediction64 = prediction_array.astype(np.float64, copy=False)
    best_lag = np.zeros(target64.shape[:2], dtype=np.int16)
    best_correlation = np.full(target64.shape[:2], -np.inf, dtype=np.float64)
    # This order provides deterministic, minimum-absolute-lag tie breaking.
    ordered_lags = [0]
    for magnitude in range(1, max_lag + 1):
        ordered_lags.extend((-magnitude, magnitude))
    # Every lag is scored on the same central target samples.  Using the full
    # overlap for each lag gives large absolute lags fewer samples and hence a
    # higher-variance correlation estimate, which biases noisy windows toward
    # the search boundary.
    central_start = max_lag
    central_end = target64.shape[-1] - max_lag
    actual = target64[..., central_start:central_end]
    for lag in ordered_lags:
        predicted = prediction64[
            ..., central_start - lag : central_end - lag
        ]
        actual_centered = actual - actual.mean(axis=-1, keepdims=True)
        predicted_centered = predicted - predicted.mean(
            axis=-1, keepdims=True
        )
        numerator = np.sum(actual_centered * predicted_centered, axis=-1)
        denominator = np.sqrt(
            np.sum(np.square(actual_centered), axis=-1)
            * np.sum(np.square(predicted_centered), axis=-1)
        )
        correlation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        both_constant_equal = (denominator == 0) & np.all(
            np.isclose(actual, predicted, rtol=0.0, atol=1e-12), axis=-1
        )
        correlation = np.where(both_constant_equal, 1.0, correlation)
        improved = correlation > best_correlation + 1e-12
        best_correlation[improved] = correlation[improved]
        best_lag[improved] = lag
    return best_lag, np.ascontiguousarray(best_correlation, dtype=np.float32)


def _periodogram_band_power(
    signal: np.ndarray,
    *,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Integrate a Hann-window one-sided periodogram over ``[low, high)``."""

    values = np.asarray(signal)
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("signal must have shape [window,channel,time]")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("signal must be floating point")
    if not np.isfinite(values).all():
        raise ValueError("signal contains NaN or Inf")
    sampling_rate = float(sampling_rate)
    low_hz = float(low_hz)
    high_hz = float(high_hz)
    if (
        not np.isfinite(sampling_rate)
        or sampling_rate <= 0
        or not 0 <= low_hz < high_hz <= sampling_rate / 2.0
    ):
        raise ValueError("invalid sampling rate or frequency band")
    samples = int(values.shape[-1])
    window = np.hanning(samples).astype(np.float64)
    window_energy = float(np.sum(np.square(window)))
    if window_energy <= 0:
        raise ValueError("at least two time samples are required")
    demeaned = values.astype(np.float64) - values.astype(np.float64).mean(
        axis=-1, keepdims=True
    )
    spectrum = np.fft.rfft(demeaned * window, axis=-1)
    psd = np.square(np.abs(spectrum)) / (sampling_rate * window_energy)
    if samples % 2 == 0:
        if psd.shape[-1] > 2:
            psd[..., 1:-1] *= 2.0
    elif psd.shape[-1] > 1:
        psd[..., 1:] *= 2.0
    frequencies = np.fft.rfftfreq(samples, d=1.0 / sampling_rate)
    selected = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(selected):
        raise ValueError(
            f"band [{low_hz},{high_hz}) contains no Fourier bins"
        )
    frequency_resolution = sampling_rate / samples
    return np.sum(psd[..., selected], axis=-1) * frequency_resolution


def band_power_error(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    sampling_rate: float = 64.0,
    bands: Sequence[tuple[float, float]] = ((0.5, 3.0), (3.0, 8.0)),
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compare target/prediction power in preregistered locomotor bands."""

    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction shapes differ")
    if not np.isfinite(epsilon) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")
    result: dict[str, Any] = {}
    for low_hz, high_hz in bands:
        actual_power = _periodogram_band_power(
            target_array,
            sampling_rate=sampling_rate,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        predicted_power = _periodogram_band_power(
            prediction_array,
            sampling_rate=sampling_rate,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        error = predicted_power - actual_power
        log_ratio = np.log(
            (predicted_power + float(epsilon))
            / (actual_power + float(epsilon))
        )
        key = f"{float(low_hz):g}-{float(high_hz):g}Hz"
        result[key] = {
            "low_hz": float(low_hz),
            "high_hz": float(high_hz),
            "target_mean_power": float(actual_power.mean()),
            "prediction_mean_power": float(predicted_power.mean()),
            "mean_signed_error": float(error.mean()),
            "mean_absolute_error": float(np.abs(error).mean()),
            "mean_absolute_log_ratio": float(np.abs(log_ratio).mean()),
            "per_channel_target_mean_power": actual_power.mean(axis=0).tolist(),
            "per_channel_prediction_mean_power": predicted_power.mean(
                axis=0
            ).tolist(),
        }
    return result


def forecast_diagnostics(
    target: np.ndarray | Mapping[str, np.ndarray],
    mean: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    *,
    mask: np.ndarray | Sequence[int] | None = None,
    expected_channels: int = H200_CHANNELS,
    expected_horizon: int = H200_HORIZON_SAMPLES,
    sampling_rate: float = 64.0,
    max_lag: int = 16,
    diagnostic_max_windows: int | None = None,
) -> dict[str, Any]:
    """Compute Phase-0 accuracy/calibration diagnostics by lead quartile.

    The runner may pass the mapping returned by
    :func:`derive_forecast_primitives`; direct ``target, mean, sigma`` arrays
    remain supported.  ``mask`` is applied only along the window dimension.
    """

    if isinstance(target, Mapping):
        if mean is not None or sigma is not None:
            raise ValueError(
                "mean/sigma must be omitted when target is a primitives mapping"
            )
        payload = target
        target = np.asarray(payload.get("raw", payload.get("target")))
        mean_value = payload.get("mean", payload.get("mu"))
        sigma_value = payload.get("sigma")
        if mean_value is None or sigma_value is None:
            raise KeyError("primitives mapping requires raw, mean/mu, and sigma")
        mean = np.asarray(mean_value)
        sigma = np.asarray(sigma_value)
    if mean is None or sigma is None:
        raise ValueError("mean and sigma are required")
    target = np.asarray(target)
    mean = np.asarray(mean)
    sigma = np.asarray(sigma)
    if mask is not None:
        selector = np.asarray(mask)
        if selector.ndim != 1:
            raise ValueError("mask must be one-dimensional")
        if selector.dtype == bool and len(selector) != len(target):
            raise ValueError("boolean mask length differs from window count")
        target = target[selector]
        mean = mean[selector]
        if sigma.shape[0] != 1:
            sigma = sigma[selector]

    target_array, mean_array, sigma_array = validate_forecast_primitives(
        target,
        mean,
        sigma,
        expected_channels=expected_channels,
        expected_horizon=expected_horizon,
    )
    horizon = int(target_array.shape[-1])
    if horizon % 4:
        raise ValueError("forecast horizon must divide into four equal quartiles")
    width = horizon // 4
    quartiles: list[dict[str, Any]] = []
    for index in range(4):
        start = index * width
        end = (index + 1) * width
        quartiles.append(
            {
                "quartile": index + 1,
                "start_lead_sample": start,
                "end_lead_sample_exclusive": end,
                **_forecast_metric_block(
                    target_array[:, :, start:end],
                    mean_array[:, :, start:end],
                    sigma_array[:, :, start:end],
                ),
            }
        )
    per_channel = [
        {
            "channel": channel,
            **_forecast_metric_block(
                target_array[:, channel : channel + 1, :],
                mean_array[:, channel : channel + 1, :],
                sigma_array[:, channel : channel + 1, :],
            ),
        }
        for channel in range(int(expected_channels))
    ]
    if diagnostic_max_windows is not None and int(diagnostic_max_windows) <= 0:
        raise ValueError("diagnostic_max_windows must be positive or None")
    diagnostic_windows = (
        len(target_array)
        if diagnostic_max_windows is None
        else min(len(target_array), int(diagnostic_max_windows))
    )
    diagnostic_target = target_array[:diagnostic_windows]
    diagnostic_mean = mean_array[:diagnostic_windows]
    best_lag, peak_correlation = optimal_cross_correlation_lag(
        diagnostic_target, diagnostic_mean, max_lag=max_lag
    )
    lag_values, lag_counts = np.unique(best_lag, return_counts=True)
    mode_position = int(np.argmax(lag_counts))
    mode_lag = int(lag_values[mode_position])
    mode_fraction = float(lag_counts[mode_position] / best_lag.size)
    lag_by_channel = [
        {
            "channel": channel,
            "mean_lag_samples": float(best_lag[:, channel].mean()),
            "median_lag_samples": float(np.median(best_lag[:, channel])),
            "mean_absolute_lag_samples": float(
                np.abs(best_lag[:, channel]).mean()
            ),
            "mean_peak_correlation": float(
                peak_correlation[:, channel].mean()
            ),
        }
        for channel in range(int(expected_channels))
    ]
    return {
        "windows": int(target_array.shape[0]),
        "channels": int(target_array.shape[1]),
        "horizon_samples": horizon,
        "nll_definition": "log(sigma) + 0.5 * ((target-mean)/sigma)^2",
        "overall": _forecast_metric_block(
            target_array, mean_array, sigma_array
        ),
        "lead_quartiles": quartiles,
        "per_channel": per_channel,
        "signal_diagnostic_windows": diagnostic_windows,
        "cross_correlation": {
            "sign_convention": (
                "positive means prediction leads target and is shifted right"
            ),
            "max_lag_samples": int(max_lag),
            "mean_lag_samples": float(best_lag.mean()),
            "median_lag_samples": float(np.median(best_lag)),
            "mean_absolute_lag_samples": float(np.abs(best_lag).mean()),
            "mean_peak_correlation": float(peak_correlation.mean()),
            "mode_lag_samples": mode_lag,
            "mode_lag_fraction": mode_fraction,
            "boundary_lag_fraction": float(
                np.mean(np.abs(best_lag) == int(max_lag))
            ),
            "per_channel": lag_by_channel,
        },
        "band_power_error": band_power_error(
            diagnostic_target,
            diagnostic_mean,
            sampling_rate=sampling_rate,
        ),
    }


def persistence_mean(context: np.ndarray, horizon_samples: int) -> np.ndarray:
    """Repeat the latest context observation over every future lead."""

    context_array = np.asarray(context)
    if context_array.dtype != np.float32:
        raise TypeError("context must have dtype float32")
    if context_array.ndim != 3 or min(context_array.shape) <= 0:
        raise ValueError("context must have shape [window, channel, sample]")
    if not np.isfinite(context_array).all():
        raise ValueError("context contains NaN or Inf")
    if int(horizon_samples) <= 0:
        raise ValueError("horizon_samples must be positive")
    return np.ascontiguousarray(
        np.repeat(context_array[:, :, -1:], int(horizon_samples), axis=2),
        dtype=np.float32,
    )


def calibrate_persistence_sigma(
    clean_normal_context: np.ndarray,
    clean_normal_target: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Fit persistence sigma by channel/lead using clean training data only.

    The MLE-style estimator is
    ``sqrt(mean((target-last_context)^2, axis=window) + epsilon)`` and returns
    ``[1, channel, lead]`` for later broadcasting.  The caller is responsible
    for passing only the preregistered clean-normal training split.
    """

    context = np.asarray(clean_normal_context)
    target = np.asarray(clean_normal_target)
    if context.dtype != np.float32 or target.dtype != np.float32:
        raise TypeError("persistence calibration arrays must have dtype float32")
    if context.ndim != 3 or target.ndim != 3:
        raise ValueError("calibration arrays must be three-dimensional")
    if context.shape[:2] != target.shape[:2] or context.shape[0] <= 0:
        raise ValueError("context and target window/channel shapes differ")
    if (
        not np.isfinite(context).all()
        or not np.isfinite(target).all()
        or not np.isfinite(epsilon)
        or float(epsilon) <= 0
    ):
        raise ValueError("calibration values and epsilon must be finite")
    mean = persistence_mean(context, target.shape[-1])
    error64 = target.astype(np.float64) - mean.astype(np.float64)
    variance = np.mean(np.square(error64), axis=0, keepdims=True)
    sigma = np.sqrt(variance + float(epsilon))
    return np.ascontiguousarray(sigma, dtype=np.float32)


def persistence_forecast_diagnostics(
    context: np.ndarray,
    target: np.ndarray,
    calibrated_sigma: np.ndarray,
) -> dict[str, Any]:
    """Evaluate persistence with sigma fitted on a separate clean split."""

    target_array = np.asarray(target)
    if target_array.ndim != 3:
        raise ValueError("target must have shape [window, channel, lead]")
    mean = persistence_mean(context, target_array.shape[-1])
    return forecast_diagnostics(
        target_array,
        mean,
        calibrated_sigma,
        expected_channels=int(target_array.shape[1]),
        expected_horizon=int(target_array.shape[2]),
    )


def paired_bootstrap(
    candidate: Mapping[str, float] | Sequence[float] | np.ndarray,
    reference: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    samples: int = 100_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Runner-facing alias for the subject-paired bootstrap."""

    return paired_subject_bootstrap(
        candidate,
        reference,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def evaluate_phase2_gate(**kwargs: Any) -> dict[str, Any]:
    """Runner-facing alias for :func:`evaluate_go_no_go`."""

    return evaluate_go_no_go(**kwargs)


__all__ = [
    "ARM_SPECS",
    "DualBranchRF125Classifier",
    "GoNoGoThresholds",
    "H200_ARM_NAMES",
    "H200_ARM_REGISTRY",
    "H200ArmSpec",
    "RF125TCNFeatureEncoder",
    "SingleBranchRF125Classifier",
    "SubjectCrossFitFold",
    "SubjectCrossFitPlan",
    "build_h200_arm_classifier",
    "build_arm_inputs",
    "build_classifier",
    "build_subject_crossfit_plan",
    "calibrate_persistence_sigma",
    "band_power_error",
    "derive_forecast_primitives",
    "ensemble_scaled_gaussians_to_outer",
    "evaluate_go_no_go",
    "evaluate_phase2_gate",
    "forecast_diagnostics",
    "gaussian_moment_match",
    "get_h200_arm",
    "paired_subject_bootstrap",
    "paired_bootstrap",
    "persistence_forecast_diagnostics",
    "persistence_mean",
    "physical_gaussian_to_scaled",
    "optimal_cross_correlation_lag",
    "rf125_receptive_field",
    "scaled_gaussian_to_physical",
    "validate_forecast_primitives",
    "validate_subject_crossfit_plan",
]
