from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_tcndae_nbm_three_sets as audit


def test_identity_reconstruction_metrics_are_ideal() -> None:
    rng = np.random.default_rng(10)
    actual = rng.normal(size=(8, 128, 9)).astype(np.float32)
    metrics, channels = audit.summarize_set(actual, actual.copy(), actual.copy())
    assert metrics["improvement_pct"] == 100.0
    assert metrics["median_corr"] > 0.999999
    assert metrics["median_nrmse"] == 0.0
    assert abs(metrics["median_amplitude_ratio"] - 1.0) < 1e-6
    assert metrics["spectral_cosine_distance"] < 1e-10
    assert metrics["residual_rms"] == 0.0
    assert metrics["raw_latent_distance_corr"] > 0.999999
    assert metrics["safety_pass"] is True
    assert len(channels) == 9


def test_zero_reconstruction_fails_safety_gate() -> None:
    rng = np.random.default_rng(11)
    actual = rng.normal(size=(8, 128, 9)).astype(np.float32)
    predicted = np.zeros_like(actual)
    latent = np.zeros((8, 32, 32), dtype=np.float32)
    metrics, _ = audit.summarize_set(actual, predicted, latent)
    assert metrics["improvement_pct"] == 0.0
    assert metrics["median_corr"] == 0.0
    assert metrics["median_nrmse"] > 0.99
    assert metrics["safety_pass"] is False


def test_manifest_bool_handles_csv_strings() -> None:
    assert audit.manifest_bool("True") is True
    assert audit.manifest_bool("false") is False
    assert audit.manifest_bool(1) is True
