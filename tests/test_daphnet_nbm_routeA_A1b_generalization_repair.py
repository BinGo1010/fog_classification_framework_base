from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_routeA_A1b_generalization_repair",
    SCRIPTS / "run_daphnet_nbm_routeA_A1b_generalization_repair.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_context_m3_preserves_parameters_and_central_output_shape() -> None:
    counts = []
    for samples, latent_samples in ((128, 32), (256, 64), (384, 96)):
        model = runner.ContextM3(samples)
        output, latent = model(torch.randn(2, 9, samples))
        assert output.shape == (2, 9, 128)
        assert latent.shape == (2, 48, latent_samples)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts == [64633, 64633, 64633]


def test_lagged_metrics_recovers_known_shift() -> None:
    rng = np.random.default_rng(42)
    actual = rng.normal(size=(3, 128, 2)).astype(np.float32)
    predicted = np.zeros_like(actual)
    predicted[:, 5:, :] = actual[:, :-5, :]
    correlation, lag = runner.lagged_metrics(actual, predicted)
    np.testing.assert_array_equal(lag, np.full((3, 2), 5))
    assert np.min(correlation) > 0.999


def test_all_structural_losses_are_finite_and_differentiable() -> None:
    target = torch.randn(4, 9, 128)
    for name in runner.LOSSES:
        predicted = torch.randn(4, 9, 128, requires_grad=True)
        loss = runner.structural_loss(name, predicted, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert predicted.grad is not None
        assert torch.isfinite(predicted.grad).all()


def test_offset_calibration_is_cancelled_by_frozen_window_centering() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(8, 128, 9)).astype(np.float32)
    scale = np.linspace(0.5, 2.0, 9, dtype=np.float32)
    first = runner.legacy.window_axis_center((raw - np.arange(9)) / scale)
    second = runner.legacy.window_axis_center((raw - np.arange(9) * 5.0) / scale)
    np.testing.assert_allclose(first, second, atol=2e-5)


def test_zero_duration_plan_uses_full_test_without_calibration() -> None:
    item = SimpleNamespace(
        subject="S01",
        test_indices=np.asarray([3, 5, 8], dtype=np.int64),
        scaler=SimpleNamespace(
            median=np.zeros(9, dtype=np.float32),
            iqr=np.ones(9, dtype=np.float32),
        ),
    )
    plan = runner.calibration_plan(item, 0)
    assert plan.actual_seconds == 0.0
    assert len(plan.calibration_indices) == 0
    np.testing.assert_array_equal(plan.evaluation_indices, item.test_indices)


def test_repair_rank_obeys_pearson_first_priority() -> None:
    lower_pearson = [{
        "median_corr": 0.60,
        "median_nrmse": 0.20,
        "nrmse_p90": 0.30,
        "median_amplitude_ratio": 1.0,
        "strict_pass": True,
    }]
    higher_pearson = [{
        "median_corr": 0.61,
        "median_nrmse": 0.80,
        "nrmse_p90": 1.20,
        "median_amplitude_ratio": 0.8,
        "strict_pass": False,
    }]
    assert runner.repair_rank(higher_pearson) > runner.repair_rank(lower_pearson)
