import numpy as np
import torch

from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    paired_initialization,
    raw_features,
)
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
    np.testing.assert_allclose(actual.mean(axis=1), 0.0, atol=1e-5)


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
