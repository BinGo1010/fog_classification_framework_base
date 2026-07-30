"""Phase-1 raw/NBM fusion representations for Daphnet IMU windows.

This module has no model-training dependencies.  It accepts aligned raw IMU
targets, NBM forecast errors, and conditional standard deviations using
``[window, channel, time]`` ordering.  The channel dimension is fixed at the
three-IMU, nine-axis Daphnet configuration.

F4 clips only its standardized-error channel group.  F5 deliberately uses the
*unclipped* standardized error in the Gaussian negative-log-likelihood map.
The additive ``0.5 * log(2*pi)`` constant is omitted because it carries no
diagnostic information.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


IMU_CHANNELS = 9
STANDARDIZED_ERROR_CLIP = 12.0


@dataclass(frozen=True)
class FusionRepresentationSpec:
    """Immutable metadata for one preregistered phase-1 representation."""

    display_name: str
    shape: str
    output_channels: int
    formula: str


FUSION_REPRESENTATION_REGISTRY: Mapping[str, FusionRepresentationSpec] = (
    MappingProxyType(
        {
            "F0": FusionRepresentationSpec(
                display_name="Raw",
                shape="[N, 9, T]",
                output_channels=9,
                formula="raw",
            ),
            "F1": FusionRepresentationSpec(
                display_name="Error",
                shape="[N, 9, T]",
                output_channels=9,
                formula="error",
            ),
            "F2": FusionRepresentationSpec(
                display_name="Raw + Error",
                shape="[N, 18, T]",
                output_channels=18,
                formula="cat(raw, error)",
            ),
            "F3": FusionRepresentationSpec(
                display_name="Raw + Zero",
                shape="[N, 18, T]",
                output_channels=18,
                formula="cat(raw, zeros_like(error))",
            ),
            "F4": FusionRepresentationSpec(
                display_name="Raw + z + log sigma",
                shape="[N, 27, T]",
                output_channels=27,
                formula=(
                    "cat(raw, clip(error / sigma, -12, 12), log(sigma))"
                ),
            ),
            "F5": FusionRepresentationSpec(
                display_name="Raw + Gaussian NLL",
                shape="[N, 18, T]",
                output_channels=18,
                formula=(
                    "cat(raw, log(sigma) + 0.5 * (error / sigma)^2)"
                ),
            ),
        }
    )
)
FUSION_REPRESENTATION_NAMES = tuple(FUSION_REPRESENTATION_REGISTRY)


def validate_fusion_inputs(
    raw: np.ndarray,
    error: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return aligned float32 fusion inputs without copying.

    All three inputs must have exactly the same ``[N, 9, T]`` shape.  Requiring
    float32 here prevents silent protocol drift caused by implicit conversions.
    """

    arrays = {
        "raw": np.asarray(raw),
        "error": np.asarray(error),
        "sigma": np.asarray(sigma),
    }
    for name, values in arrays.items():
        if values.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32")
        if values.ndim != 3:
            raise ValueError(f"{name} must have shape [N, 9, T]")
        if values.shape[0] <= 0 or values.shape[2] <= 0:
            raise ValueError(f"{name} must have non-empty N and T dimensions")
        if values.shape[1] != IMU_CHANNELS:
            raise ValueError(
                f"{name} must have {IMU_CHANNELS} IMU channels, "
                f"got {values.shape[1]}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf")

    raw_values = arrays["raw"]
    error_values = arrays["error"]
    sigma_values = arrays["sigma"]
    if error_values.shape != raw_values.shape:
        raise ValueError("error must have the same shape as raw")
    if sigma_values.shape != raw_values.shape:
        raise ValueError("sigma must have the same shape as raw")
    if np.any(sigma_values <= 0):
        raise ValueError("sigma must be strictly positive")
    return raw_values, error_values, sigma_values


def _as_finite_float32(name: str, values: np.ndarray) -> np.ndarray:
    """Convert a derived array to contiguous float32 without hiding overflow."""

    float32_limit = np.finfo(np.float32).max
    if not np.isfinite(values).all():
        raise ValueError(f"{name} produced NaN or Inf")
    if np.any(np.abs(values) > float32_limit):
        raise ValueError(f"{name} exceeds the finite float32 range")
    result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} produced non-finite float32 values")
    return result


def _derive_maps(
    error: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return clipped z, log-sigma, and full (unclipped-z) NLL maps."""

    error64 = error.astype(np.float64, copy=False)
    sigma64 = sigma.astype(np.float64, copy=False)
    z_unclipped = error64 / sigma64
    log_sigma = np.log(sigma64)
    z_clipped = np.clip(
        z_unclipped,
        -STANDARDIZED_ERROR_CLIP,
        STANDARDIZED_ERROR_CLIP,
    )
    gaussian_nll = log_sigma + 0.5 * np.square(z_unclipped)
    return (
        _as_finite_float32("clipped standardized error", z_clipped),
        _as_finite_float32("log sigma", log_sigma),
        _as_finite_float32("Gaussian NLL", gaussian_nll),
    )


def build_fusion_representations(
    raw: np.ndarray,
    error: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build all six preregistered F0--F5 representations."""

    raw_values, error_values, sigma_values = validate_fusion_inputs(
        raw,
        error,
        sigma,
    )
    raw_contiguous = np.ascontiguousarray(raw_values)
    error_contiguous = np.ascontiguousarray(error_values)
    zeros = np.zeros_like(error_contiguous)
    z_clipped, log_sigma, gaussian_nll = _derive_maps(
        error_values,
        sigma_values,
    )

    representations = {
        "F0": raw_contiguous,
        "F1": error_contiguous,
        "F2": np.concatenate(
            (raw_contiguous, error_contiguous),
            axis=1,
            dtype=np.float32,
        ),
        "F3": np.concatenate(
            (raw_contiguous, zeros),
            axis=1,
            dtype=np.float32,
        ),
        "F4": np.concatenate(
            (raw_contiguous, z_clipped, log_sigma),
            axis=1,
            dtype=np.float32,
        ),
        "F5": np.concatenate(
            (raw_contiguous, gaussian_nll),
            axis=1,
            dtype=np.float32,
        ),
    }
    for name, values in representations.items():
        spec = FUSION_REPRESENTATION_REGISTRY[name]
        if values.shape[1] != spec.output_channels:
            raise RuntimeError(
                f"{name} produced {values.shape[1]} channels; "
                f"expected {spec.output_channels}"
            )
        if values.dtype != np.float32 or not values.flags.c_contiguous:
            raise RuntimeError(f"{name} violated the contiguous float32 contract")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf")
    return representations


def build_fusion_representation(
    name: str,
    raw: np.ndarray,
    error: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Build one F0--F5 representation by its stable registry key."""

    if name not in FUSION_REPRESENTATION_REGISTRY:
        valid = ", ".join(FUSION_REPRESENTATION_NAMES)
        raise ValueError(f"unknown fusion representation {name!r}; use {valid}")
    raw_values, error_values, sigma_values = validate_fusion_inputs(
        raw,
        error,
        sigma,
    )
    raw_contiguous = np.ascontiguousarray(raw_values)
    error_contiguous = np.ascontiguousarray(error_values)
    if name == "F0":
        result = raw_contiguous
    elif name == "F1":
        result = error_contiguous
    elif name == "F2":
        result = np.concatenate(
            (raw_contiguous, error_contiguous),
            axis=1,
            dtype=np.float32,
        )
    elif name == "F3":
        result = np.concatenate(
            (raw_contiguous, np.zeros_like(error_contiguous)),
            axis=1,
            dtype=np.float32,
        )
    elif name == "F4":
        error64 = error_values.astype(np.float64, copy=False)
        sigma64 = sigma_values.astype(np.float64, copy=False)
        z_clipped = _as_finite_float32(
            "clipped standardized error",
            np.clip(
                error64 / sigma64,
                -STANDARDIZED_ERROR_CLIP,
                STANDARDIZED_ERROR_CLIP,
            ),
        )
        log_sigma = _as_finite_float32(
            "log sigma",
            np.log(sigma64),
        )
        result = np.concatenate(
            (raw_contiguous, z_clipped, log_sigma),
            axis=1,
            dtype=np.float32,
        )
    else:
        error64 = error_values.astype(np.float64, copy=False)
        sigma64 = sigma_values.astype(np.float64, copy=False)
        gaussian_nll = _as_finite_float32(
            "Gaussian NLL",
            np.log(sigma64)
            + 0.5 * np.square(error64 / sigma64),
        )
        result = np.concatenate(
            (raw_contiguous, gaussian_nll),
            axis=1,
            dtype=np.float32,
        )
    result = np.ascontiguousarray(result, dtype=np.float32)
    expected_channels = FUSION_REPRESENTATION_REGISTRY[
        name
    ].output_channels
    if result.shape[1] != expected_channels:
        raise RuntimeError(
            f"{name} produced {result.shape[1]} channels; "
            f"expected {expected_channels}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


__all__ = [
    "FUSION_REPRESENTATION_NAMES",
    "FUSION_REPRESENTATION_REGISTRY",
    "FusionRepresentationSpec",
    "IMU_CHANNELS",
    "STANDARDIZED_ERROR_CLIP",
    "build_fusion_representation",
    "build_fusion_representations",
    "validate_fusion_inputs",
]
