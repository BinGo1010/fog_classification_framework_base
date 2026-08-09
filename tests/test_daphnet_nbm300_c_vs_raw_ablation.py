import numpy as np
import torch

from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    build_scheme_c_features,
    paired_initialization,
    raw_features,
)
from scripts.run_daphnet_residual_calibration_abcd import build_abcd_features
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


def test_raw_features_are_role4_scaled_then_window_axis_centered() -> None:
    raw = np.arange(2 * 128 * 9, dtype=np.float32).reshape(2, 128, 9)
    scaler = RobustScaler(
        median=np.linspace(-2, 2, 9, dtype=np.float32),
        iqr=np.linspace(1, 3, 9, dtype=np.float32),
    )
    actual = raw_features(scaler, raw)
    expected = scaler.transform(raw)
    expected -= expected.mean(axis=1, keepdims=True)
    assert actual.shape == (2, 128, 9)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    maximum_mean = float(np.max(np.abs(np.mean(actual, axis=1, dtype=np.float64))))
    maximum_signal = float(np.max(np.abs(actual)))
    tolerance = max(
        1e-5,
        64.0 * float(np.finfo(np.float32).eps) * max(1.0, maximum_signal),
    )
    assert maximum_mean <= tolerance


def test_raw_features_accept_float32_centering_roundoff_at_large_scale() -> None:
    rng = np.random.default_rng(7)
    raw = (rng.normal(size=(4, 128, 9)) * 1e4).astype(np.float32)
    scaler = RobustScaler(
        median=np.zeros(9, dtype=np.float32),
        iqr=np.full(9, 0.1, dtype=np.float32),
    )
    actual = raw_features(scaler, raw)
    assert actual.shape == (4, 128, 9)
    assert np.all(np.isfinite(actual))


def test_paired_initialization_shares_all_compatible_weights() -> None:
    raw_state, raw_meta = paired_initialization(52, "RAW")
    full_state, full_meta = paired_initialization(52, "FULL_C")
    assert raw_meta["pair_id"] == full_meta["pair_id"]
    for name, raw_tensor in raw_state.items():
        full_tensor = full_state[name]
        if raw_tensor.shape == full_tensor.shape:
            assert torch.equal(raw_tensor, full_tensor)
        else:
            assert raw_tensor.ndim == full_tensor.ndim == 3
            assert full_tensor.shape[1] == 27 and raw_tensor.shape[1] == 9
            assert torch.equal(raw_tensor, full_tensor[:, :9, :])
            assert torch.count_nonzero(full_tensor[:, 9:, :]) == 0


def test_32hz_raw_and_scheme_c_shapes() -> None:
    rng = np.random.default_rng(32)
    raw = rng.normal(size=(5, 64, 9)).astype(np.float32)
    scaler = RobustScaler(
        median=np.linspace(-1, 1, 9, dtype=np.float32),
        iqr=np.linspace(0.5, 2.5, 9, dtype=np.float32),
    )
    raw_input = raw_features(scaler, raw, window_samples=64)
    assert raw_input.shape == (5, 64, 9)

    error = rng.normal(size=(5, 9, 64)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 1], dtype=np.int8)
    sigma = np.linspace(0.1, 1.0, 9, dtype=np.float32)
    scheme_c, clip_stats = build_scheme_c_features(
        error, labels, sigma, window_samples=64
    )
    assert scheme_c.shape == (5, 64, 27)
    residual_bct = scheme_c[:, :, :9].transpose(0, 2, 1)
    delta_bct = scheme_c[:, :, 18:].transpose(0, 2, 1)
    np.testing.assert_allclose(
        residual_bct.mean(axis=2, dtype=np.float64), 0.0, atol=2e-6
    )
    np.testing.assert_array_equal(delta_bct[:, :, 0], 0.0)
    assert clip_stats["overall"]["points"] == 5 * 9 * 64


def test_dynamic_scheme_c_matches_original_128_sample_implementation() -> None:
    rng = np.random.default_rng(128)
    error = rng.normal(size=(4, 9, 128)).astype(np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
    bias = rng.normal(size=9).astype(np.float32)
    sigma = np.linspace(0.05, 1.5, 9, dtype=np.float32)
    expected, _ = build_abcd_features(error, labels, "C", bias, sigma)
    actual, _ = build_scheme_c_features(error, labels, sigma, window_samples=128)
    np.testing.assert_array_equal(actual, expected)
