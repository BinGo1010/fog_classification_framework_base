from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_round3_n128_revision",
    SCRIPTS / "run_daphnet_nbm_round3_n128_revision.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def passing_row(subject: str, seed: int) -> dict:
    return {
        "subject_id": subject,
        "seed": seed,
        "sample_count": 128,
        "improvement_pct": 95.0,
        "median_corr": 0.94,
        "median_nrmse": 0.24,
        "median_amplitude_ratio": 0.98,
        "raw_latent_distance_corr": 0.82,
        "strict_pass": True,
        "tail_risk": False,
    }


def test_n128_gate_accepts_boundary_and_rejects_each_violation() -> None:
    metrics = passing_row("S01", runner.SEEDS[0])
    assert runner.n128_pass(metrics)
    for key, value in (
        ("improvement_pct", 39.9),
        ("median_corr", 0.599),
        ("median_nrmse", 0.751),
        ("median_amplitude_ratio", 0.649),
        ("raw_latent_distance_corr", 0.399),
    ):
        failed = dict(metrics, **{key: value})
        assert not runner.n128_pass(failed)


def test_identity_tail_metrics_have_no_tail_risk() -> None:
    rng = np.random.default_rng(20260802)
    actual = rng.normal(size=(16, 128, 9)).astype(np.float32)
    metrics = runner.tail_metrics(actual, actual.copy())
    assert metrics["nrmse_p95"] == 0.0
    assert metrics["pearson_p10"] > 0.999999
    assert metrics["negative_improvement_window_fraction"] == 0.0
    assert metrics["nrmse_gt_1_window_fraction"] == 0.0
    assert metrics["pearson_lt_0_2_window_fraction"] == 0.0
    assert runner.tail_risk_reasons(metrics) == []


def test_stable_cohort_can_pass_while_all_subject_gate_fails() -> None:
    rows = [
        passing_row(subject, seed)
        for subject in runner.SUBJECTS
        for seed in runner.SEEDS
    ]
    for row in rows:
        if row["subject_id"] == "S03":
            row.update(
                strict_pass=False,
                median_corr=0.45,
                median_nrmse=0.90,
            )
    result = runner.evaluate_gates(rows)
    assert result["all_subject_strict_gate"] == "FAIL"
    assert result["stable_cohort_gate"] == "PASS"
    assert result["strict_pass_count"] == 21
    assert result["stable_pass_count"] == 21
    assert result["final_status"] == "Stable-cohort PASS"
    assert result["formal_denoising_progression_eligible"]


def test_stable_gate_rejects_a_new_zero_of_three_subject() -> None:
    rows = [
        passing_row(subject, seed)
        for subject in runner.SUBJECTS
        for seed in runner.SEEDS
    ]
    for row in rows:
        if row["subject_id"] == "S02":
            row["strict_pass"] = False
    result = runner.evaluate_gates(rows)
    assert result["stable_cohort_gate"] == "FAIL"
    assert result["new_stable_subjects_0_of_3"] == ["S02"]


def test_worst_channel_uses_channel_median_nrmse() -> None:
    rows = [
        {"channel": "ankle", "nrmse": value} for value in (0.1, 0.2, 2.0)
    ] + [{"channel": "thigh", "nrmse": value} for value in (0.7, 0.8, 0.9)]
    name, value = runner.worst_channel(rows)
    assert name == "thigh"
    assert value == 0.8
