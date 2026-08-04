from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_routeA_A2_A4", SCRIPTS / "run_daphnet_nbm_routeA_A2_A4.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_corruptions_are_deterministic_and_shape_safe() -> None:
    clean = np.ones((20, 128, 9), dtype=np.float32)
    for scheme in runner.DENOISING:
        first = runner.corrupt_mixture(clean, scheme, 17)
        second = runner.corrupt_mixture(clean, scheme, 17)
        assert first.shape == clean.shape
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(runner.corrupt_mixture(clean, "D0", 17), clean)


def test_residual_calibration_is_finite_and_centered() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(loc=2.0, scale=0.2, size=(30, 128, 9)).astype(np.float32)
    values = rng.normal(loc=2.0, scale=0.2, size=(8, 128, 9)).astype(np.float32)
    stats = runner.fit_residual_calibration(train, sigma_min=0.05)
    calibrated, saturation = runner.apply_residual_calibration(values, stats, "C2", 6.0)
    assert np.isfinite(calibrated).all()
    assert 0.0 <= saturation <= 1.0
    np.testing.assert_allclose(calibrated.mean(axis=1), 0.0, atol=1e-5)


def test_all_representations_have_expected_dimensions() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 128, 9)).astype(np.float32)
    xhat = rng.normal(size=(4, 128, 9)).astype(np.float32)
    residual = x - xhat
    expected = {"R0": (4, 128, 9), "R1": (4, 128, 9), "R2": (4, 128, 9),
                "R3": (4, 36), "R4": (4, 45), "R5": (4, 128, 27),
                "R6": (4, 128, 27)}
    for name, shape in expected.items():
        value = runner.build_representation(name, x, xhat, residual)
        assert value.shape == shape
        assert np.isfinite(value).all()


def test_binary_metrics_are_directionally_correct() -> None:
    normal = np.asarray([0.0, 0.1, 0.2, 0.3])
    fog = np.asarray([1.0, 1.1, 1.2, 1.3])
    metrics = runner.separation_metrics(normal, fog, normal)
    assert metrics["auroc"] == 1.0
    assert metrics["cliffs_delta"] == 1.0
    assert metrics["fog_to_nonfog_median_ratio"] > 3.0
