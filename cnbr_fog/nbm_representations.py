"""Diagnostic representations derived from probabilistic NBM forecasts.

The functions in this module are intentionally independent of the experiment
runner.  They make the scientific distinction between:

* the signed forecast error ``x - mu``;
* that error divided by a static, clean-normal calibration scale; and
* that error divided by the NBM's input-conditional scale.

All arrays use ``[window, channel, horizon]`` ordering.
"""

from __future__ import annotations

import numpy as np


REPRESENTATION_NAMES = (
    "error_x_minus_mu",
    "fixed_standardized_error",
    "dynamic_standardized_error",
)


def calibrate_fixed_sigma(
    calibration_errors: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Estimate a static channel-by-horizon Gaussian scale.

    ``calibration_errors`` must contain only clean-normal calibration windows.
    The returned shape is ``[1, channel, horizon]`` so it broadcasts over
    arbitrary train/validation/test window counts.
    """

    errors = np.asarray(calibration_errors, dtype=np.float64)
    if errors.ndim != 3 or errors.shape[0] == 0:
        raise ValueError(
            "calibration_errors must be non-empty [window, channel, horizon]"
        )
    if not np.isfinite(errors).all():
        raise ValueError("calibration_errors contain NaN or Inf")
    if not np.isfinite(epsilon) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")
    sigma = np.sqrt(np.mean(np.square(errors), axis=0, keepdims=True) + epsilon)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("fixed sigma calibration produced invalid values")
    return sigma.astype(np.float32, copy=False)


def build_nbm_representations(
    error: np.ndarray,
    dynamic_sigma: np.ndarray,
    fixed_sigma: np.ndarray,
    *,
    standardized_clip: float = 12.0,
) -> dict[str, np.ndarray]:
    """Build the three preregistered NBM representations.

    The signed error remains un-clipped.  Both standardized variants share the
    same symmetric clipping operator, so their only scientific difference is
    whether uncertainty is static or input conditional.
    """

    signed_error = np.asarray(error, dtype=np.float32)
    conditional_sigma = np.asarray(dynamic_sigma, dtype=np.float32)
    static_sigma = np.asarray(fixed_sigma, dtype=np.float32)
    if signed_error.ndim != 3:
        raise ValueError("error must have shape [window, channel, horizon]")
    if conditional_sigma.shape != signed_error.shape:
        raise ValueError("dynamic_sigma must have the same shape as error")
    if static_sigma.shape != (1, *signed_error.shape[1:]):
        raise ValueError(
            "fixed_sigma must have shape [1, channel, horizon]"
        )
    for name, values in (
        ("error", signed_error),
        ("dynamic_sigma", conditional_sigma),
        ("fixed_sigma", static_sigma),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if np.any(conditional_sigma <= 0) or np.any(static_sigma <= 0):
        raise ValueError("all sigma values must be strictly positive")
    if (
        not np.isfinite(standardized_clip)
        or float(standardized_clip) <= 0
    ):
        raise ValueError("standardized_clip must be finite and positive")

    clip = float(standardized_clip)
    fixed = np.clip(signed_error / static_sigma, -clip, clip)
    dynamic = np.clip(signed_error / conditional_sigma, -clip, clip)
    return {
        "error_x_minus_mu": np.ascontiguousarray(signed_error),
        "fixed_standardized_error": np.ascontiguousarray(
            fixed.astype(np.float32, copy=False)
        ),
        "dynamic_standardized_error": np.ascontiguousarray(
            dynamic.astype(np.float32, copy=False)
        ),
    }


__all__ = [
    "REPRESENTATION_NAMES",
    "build_nbm_representations",
    "calibrate_fixed_sigma",
]
