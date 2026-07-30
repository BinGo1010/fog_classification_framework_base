from __future__ import annotations

import numpy as np
import pytest

from cnbr_fog.fusion_representations import (
    FUSION_REPRESENTATION_NAMES,
    FUSION_REPRESENTATION_REGISTRY,
    STANDARDIZED_ERROR_CLIP,
    build_fusion_representation,
    build_fusion_representations,
    validate_fusion_inputs,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.arange(2 * 9 * 4, dtype=np.float32).reshape(2, 9, 4)
    error = np.full_like(raw, 2.0)
    error[0, 0] = np.asarray([24.0, -24.0, 1.0, -1.0], dtype=np.float32)
    sigma = np.full_like(raw, 2.0)
    sigma[0, 0] = 1.0
    return raw, error, sigma


def test_registry_fixes_ids_display_shapes_and_formulas() -> None:
    assert FUSION_REPRESENTATION_NAMES == (
        "F0",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    )
    expected = {
        "F0": ("Raw", "[N, 9, T]", 9, "raw"),
        "F1": ("Error", "[N, 9, T]", 9, "error"),
        "F2": ("Raw + Error", "[N, 18, T]", 18, "cat(raw, error)"),
        "F3": (
            "Raw + Zero",
            "[N, 18, T]",
            18,
            "cat(raw, zeros_like(error))",
        ),
        "F4": (
            "Raw + z + log sigma",
            "[N, 27, T]",
            27,
            "cat(raw, clip(error / sigma, -12, 12), log(sigma))",
        ),
        "F5": (
            "Raw + Gaussian NLL",
            "[N, 18, T]",
            18,
            "cat(raw, log(sigma) + 0.5 * (error / sigma)^2)",
        ),
    }
    for name, fields in expected.items():
        spec = FUSION_REPRESENTATION_REGISTRY[name]
        assert (
            spec.display_name,
            spec.shape,
            spec.output_channels,
            spec.formula,
        ) == fields
    with pytest.raises(TypeError):
        FUSION_REPRESENTATION_REGISTRY["F6"] = (  # type: ignore[index]
            FUSION_REPRESENTATION_REGISTRY["F0"]
        )


def test_f0_to_f5_values_shapes_and_output_contract() -> None:
    raw, error, sigma = _inputs()
    result = build_fusion_representations(raw, error, sigma)

    assert tuple(result) == FUSION_REPRESENTATION_NAMES
    for name, values in result.items():
        spec = FUSION_REPRESENTATION_REGISTRY[name]
        assert values.shape == (2, spec.output_channels, 4)
        assert values.dtype == np.float32
        assert values.flags.c_contiguous
        assert np.isfinite(values).all()

    zeros = np.zeros_like(raw)
    z = np.clip(
        error / sigma,
        -STANDARDIZED_ERROR_CLIP,
        STANDARDIZED_ERROR_CLIP,
    )
    log_sigma = np.log(sigma)
    nll = log_sigma + 0.5 * np.square(error / sigma)
    np.testing.assert_array_equal(result["F0"], raw)
    np.testing.assert_array_equal(result["F1"], error)
    np.testing.assert_array_equal(result["F2"], np.concatenate((raw, error), 1))
    np.testing.assert_array_equal(result["F3"], np.concatenate((raw, zeros), 1))
    np.testing.assert_allclose(
        result["F4"],
        np.concatenate((raw, z, log_sigma), 1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["F5"],
        np.concatenate((raw, nll), 1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_f4_clips_z_but_f5_uses_unclipped_z() -> None:
    raw, error, sigma = _inputs()
    result = build_fusion_representations(raw, error, sigma)
    np.testing.assert_array_equal(
        result["F4"][0, 9, :2],
        np.asarray([12.0, -12.0], dtype=np.float32),
    )
    # If F5 incorrectly reused clipped z, these values would be 72 instead.
    np.testing.assert_array_equal(
        result["F5"][0, 9, :2],
        np.asarray([288.0, 288.0], dtype=np.float32),
    )


def test_noncontiguous_inputs_produce_contiguous_outputs() -> None:
    raw, error, sigma = _inputs()
    raw_nc = raw[:, :, ::-1]
    error_nc = error[:, :, ::-1]
    sigma_nc = sigma[:, :, ::-1]
    assert not raw_nc.flags.c_contiguous
    result = build_fusion_representations(raw_nc, error_nc, sigma_nc)
    assert all(values.flags.c_contiguous for values in result.values())


def test_build_one_matches_build_all_and_rejects_unknown_name() -> None:
    raw, error, sigma = _inputs()
    expected = build_fusion_representations(raw, error, sigma)["F4"]
    actual = build_fusion_representation("F4", raw, error, sigma)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="unknown fusion representation"):
        build_fusion_representation("F9", raw, error, sigma)


@pytest.mark.parametrize(
    ("mutator", "error_type", "message"),
    [
        (
            lambda r, e, s: (r.astype(np.float64), e, s),
            TypeError,
            "raw must have dtype float32",
        ),
        (
            lambda r, e, s: (r[:, 0], e[:, 0], s[:, 0]),
            ValueError,
            "shape",
        ),
        (
            lambda r, e, s: (r[:, :8], e[:, :8], s[:, :8]),
            ValueError,
            "9 IMU channels",
        ),
        (
            lambda r, e, s: (r, e[:, :, :3], s),
            ValueError,
            "error must have the same shape",
        ),
        (
            lambda r, e, s: (r, e, s[:, :, :3]),
            ValueError,
            "sigma must have the same shape",
        ),
        (
            lambda r, e, s: (r, e, np.where(s == 1, 0, s).astype(np.float32)),
            ValueError,
            "strictly positive",
        ),
        (
            lambda r, e, s: (
                r,
                np.where(e == 24, np.nan, e).astype(np.float32),
                s,
            ),
            ValueError,
            "error contains NaN or Inf",
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    mutator,
    error_type: type[Exception],
    message: str,
) -> None:
    raw, error, sigma = _inputs()
    invalid = mutator(raw, error, sigma)
    with pytest.raises(error_type, match=message):
        validate_fusion_inputs(*invalid)


def test_gaussian_nll_that_cannot_fit_float32_is_rejected() -> None:
    raw = np.zeros((1, 9, 1), dtype=np.float32)
    error = np.full_like(raw, np.finfo(np.float32).max)
    sigma = np.full_like(raw, np.finfo(np.float32).tiny)
    with pytest.raises(ValueError, match="Gaussian NLL exceeds"):
        build_fusion_representations(raw, error, sigma)
