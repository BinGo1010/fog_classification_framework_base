from __future__ import annotations

import numpy as np
import pytest

from cnbr_fog.nbm_representations import (
    build_nbm_representations,
    calibrate_fixed_sigma,
)


def test_fixed_sigma_is_channel_and_horizon_rms() -> None:
    errors = np.asarray(
        [
            [[3.0, 4.0], [0.0, 8.0]],
            [[3.0, 0.0], [6.0, 0.0]],
        ],
        dtype=np.float32,
    )
    sigma = calibrate_fixed_sigma(errors, epsilon=1e-12)
    expected = np.sqrt(np.mean(errors.astype(np.float64) ** 2, axis=0))
    assert sigma.shape == (1, 2, 2)
    np.testing.assert_allclose(sigma[0], expected, rtol=1e-6, atol=1e-6)


def test_representations_only_standardized_variants_are_clipped() -> None:
    error = np.asarray([[[24.0, -6.0]]], dtype=np.float32)
    dynamic_sigma = np.asarray([[[1.0, 3.0]]], dtype=np.float32)
    fixed_sigma = np.asarray([[[2.0, 2.0]]], dtype=np.float32)
    result = build_nbm_representations(
        error,
        dynamic_sigma,
        fixed_sigma,
        standardized_clip=5.0,
    )
    np.testing.assert_array_equal(result["error_x_minus_mu"], error)
    np.testing.assert_array_equal(
        result["fixed_standardized_error"],
        np.asarray([[[5.0, -3.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result["dynamic_standardized_error"],
        np.asarray([[[5.0, -2.0]]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("error", "dynamic", "fixed", "message"),
    [
        (
            np.zeros((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            "error must have shape",
        ),
        (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 5), dtype=np.float32),
            np.ones((1, 3, 4), dtype=np.float32),
            "dynamic_sigma",
        ),
        (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
            "fixed_sigma",
        ),
    ],
)
def test_invalid_representation_shapes_are_rejected(
    error: np.ndarray,
    dynamic: np.ndarray,
    fixed: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_nbm_representations(error, dynamic, fixed)
