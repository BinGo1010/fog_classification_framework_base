from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.launch_daphnet_gru_residual_g123_7gpu import validate_contract
from scripts.run_daphnet_gru_residual_g123 import (
    GROUPS,
    build_g123_features,
)


def test_launcher_accepts_only_frozen_tcn_training_contracts() -> None:
    for maximum_epochs, patience in ((10, 2), (50, 6)):
        args = argparse.Namespace(
            tcn_seeds="0,52,161",
            tcn_max_epochs=maximum_epochs,
            tcn_patience=patience,
        )
        assert validate_contract(args) == (0, 52, 161)

    invalid = argparse.Namespace(
        tcn_seeds="0,52,161",
        tcn_max_epochs=50,
        tcn_patience=2,
    )
    with pytest.raises(ValueError, match="requires one of the TCN"):
        validate_contract(invalid)


def test_g123_exact_formulas_and_shared_feature_shape() -> None:
    rng = np.random.default_rng(123)
    error = rng.normal(0.0, 3.0, size=(6, 9, 128)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    bias = np.linspace(-0.2, 0.2, 9, dtype=np.float32)
    sigma = np.linspace(0.05, 0.25, 9, dtype=np.float32)
    standardized = (error - bias[None, :, None]) / (sigma[None, :, None] + 1e-6)
    clipped = np.clip(standardized, -12.0, 12.0).astype(np.float32)

    outputs = {}
    diagnostics = {}
    for group in GROUPS:
        outputs[group], diagnostics[group] = build_g123_features(
            error, labels, group, bias, sigma
        )
        assert outputs[group].shape == (6, 128, 27)
        assert np.all(np.isfinite(outputs[group]))
        assert np.all(outputs[group][:, 0, 18:] == 0)

    g1 = outputs["G1"].transpose(0, 2, 1)[:, :9]
    g2 = outputs["G2"].transpose(0, 2, 1)[:, :9]
    g3 = outputs["G3"].transpose(0, 2, 1)[:, :9]
    np.testing.assert_allclose(g1, clipped - clipped.mean(axis=2, keepdims=True), atol=2e-6)
    np.testing.assert_allclose(g2, clipped, atol=0, rtol=0)
    np.testing.assert_allclose(g3, np.arcsinh(standardized), atol=2e-6)
    assert np.max(np.abs(g1.mean(axis=2))) < 2e-6
    assert diagnostics["G1"] == diagnostics["G2"]
    assert diagnostics["G3"]["applicable"] is False


def test_abs_and_delta_blocks_match_signed_residual() -> None:
    rng = np.random.default_rng(456)
    error = rng.normal(size=(2, 9, 128)).astype(np.float32)
    labels = np.asarray([0, 1], dtype=np.int8)
    bias = np.zeros(9, dtype=np.float32)
    sigma = np.ones(9, dtype=np.float32)
    for group in GROUPS:
        features, _ = build_g123_features(error, labels, group, bias, sigma)
        bct = features.transpose(0, 2, 1)
        signed, absolute, delta = bct[:, :9], bct[:, 9:18], bct[:, 18:]
        np.testing.assert_allclose(absolute, np.abs(signed), atol=0, rtol=0)
        expected_delta = np.diff(signed, axis=2, prepend=signed[:, :, :1])
        np.testing.assert_allclose(delta, expected_delta, atol=0, rtol=0)
