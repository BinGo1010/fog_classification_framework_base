from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_routeA_final_residual_validation",
    SCRIPTS / "run_daphnet_nbm_routeA_final_residual_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def passing_metrics() -> dict:
    return {
        "zero_improvement_pct": 80.0,
        "template_improvement_pct": 20.0,
        "median_corr": 0.8,
        "median_nrmse": 0.4,
        "nrmse_p90": 0.8,
        "negative_improvement_window_fraction": 0.05,
        "median_amplitude_ratio": 0.95,
    }


def test_a1_gate_rejects_every_preregistered_threshold_violation() -> None:
    assert runner.a1_pass(passing_metrics())
    failures = (
        ("zero_improvement_pct", 20.0),
        ("template_improvement_pct", 10.0),
        ("median_corr", 0.499),
        ("median_nrmse", 0.851),
        ("nrmse_p90", 1.301),
        ("negative_improvement_window_fraction", 0.201),
    )
    for key, value in failures:
        assert not runner.a1_pass(dict(passing_metrics(), **{key: value}))


def test_identity_reconstruction_has_zero_error_and_no_negative_windows() -> None:
    rng = np.random.default_rng(20260802)
    actual = rng.normal(size=(8, 128, 9)).astype(np.float32)
    template = np.repeat(actual.mean(axis=0, keepdims=True), len(actual), axis=0)
    metrics, arrays = runner.reconstruction_metrics(actual, actual.copy(), template)
    assert metrics["mse"] == 0.0
    assert metrics["zero_improvement_pct"] == 100.0
    assert metrics["template_improvement_pct"] == 100.0
    assert metrics["median_corr"] > 0.999999
    assert metrics["median_nrmse"] == 0.0
    assert metrics["negative_improvement_window_fraction"] == 0.0
    assert arrays["window_nrmse"].shape == (8,)


def test_nearest_training_window_recovers_exact_matches() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(6, 128, 9)).astype(np.float32)
    test = train[[4, 1, 5]].copy()
    nearest, indices = runner.nearest_training_windows(train, test)
    np.testing.assert_array_equal(indices, [4, 1, 5])
    np.testing.assert_allclose(nearest, test)


def result_row(subject: str, seed: int, *, passed: bool, collapse: bool = False) -> dict:
    return {
        "subject_id": subject,
        "seed": seed,
        "split_type": "fixed_unseen_record_holdout",
        "test_record_or_block": "test",
        "strict_pass": passed,
        "waveform_collapse": collapse,
        "zero_improvement_pct": 70.0,
        "template_improvement_pct": 20.0,
        "median_corr": 0.75,
        "median_nrmse": 0.5,
        "nrmse_p90": 0.9,
        "negative_improvement_window_fraction": 0.05,
    }


def test_overall_gate_requires_five_subjects_and_limits_collapse() -> None:
    rows = [
        result_row(subject, seed, passed=subject in runner.SUBJECTS[:5])
        for subject in runner.SUBJECTS
        for seed in runner.SEEDS
    ]
    gate = runner.evaluate_a1_gate(rows)
    assert gate["status"] == "PASS"
    for row in rows:
        if row["subject_id"] in runner.SUBJECTS[:3]:
            row["waveform_collapse"] = True
    gate = runner.evaluate_a1_gate(rows)
    assert gate["status"] == "FAIL"
    assert gate["waveform_collapse_subject_count"] == 3


def test_waveform_collapse_is_independent_of_strict_template_gate() -> None:
    metrics = passing_metrics()
    metrics["template_improvement_pct"] = 0.0
    assert not runner.a1_pass(metrics)
    assert not runner.waveform_collapse(metrics)
    metrics["median_amplitude_ratio"] = 0.49
    assert runner.waveform_collapse(metrics)


def test_baseline_rows_flattens_all_four_frozen_comparators() -> None:
    metrics = {"mse": 1.0, "median_corr": 0.5}
    result = {
        "subject_id": "S01",
        "fold_id": "fold_01",
        "seed": 20260802,
        "baselines": {
            "B0_zero": metrics,
            "B1_training_mean_template": metrics,
            "B2_nearest_training_window": metrics,
            "B3_M3_TC_DAE": metrics,
        },
    }
    rows = runner.baseline_rows([result])
    assert len(rows) == 4
    assert {row["baseline"] for row in rows} == set(result["baselines"])
    assert all(row["subject_id"] == "S01" for row in rows)
