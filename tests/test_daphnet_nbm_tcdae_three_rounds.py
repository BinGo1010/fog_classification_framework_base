from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_tcdae_three_rounds",
    ROOT / "scripts" / "run_daphnet_nbm_tcdae_three_rounds.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_identity_metrics_pass_every_round() -> None:
    rng = np.random.default_rng(20260802)
    actual = rng.normal(size=(8, 128, 9)).astype(np.float32)
    latent = rng.normal(size=(8, 32, 16)).astype(np.float32)
    metrics, _ = runner.summarize(actual, actual.copy(), latent)
    assert metrics["nbm_mse"] == 0.0
    assert metrics["improvement_pct"] == 100.0
    assert metrics["median_corr"] > 0.999999
    assert metrics["median_nrmse"] == 0.0
    assert metrics["median_amplitude_ratio"] > 0.999999
    assert runner.round1_pass(metrics)
    assert runner.round2_pass(1, metrics)
    assert runner.round3_pass(32, metrics)


def test_zero_output_fails_and_has_zero_amplitude() -> None:
    time = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    actual = np.stack([np.sin((channel + 1) * time) for channel in range(9)], axis=1)
    actual = np.repeat(actual[None].astype(np.float32), 8, axis=0)
    latent = np.zeros((8, 32, 16), dtype=np.float32)
    metrics, _ = runner.summarize(actual, np.zeros_like(actual), latent)
    assert abs(metrics["improvement_pct"]) < 1e-8
    assert metrics["median_corr"] == 0.0
    assert metrics["median_amplitude_ratio"] == 0.0
    assert not runner.round1_pass(metrics)
    assert not runner.round2_pass(8, metrics)
    assert not runner.round3_pass(1, metrics)


def test_pairwise_distance_correlation_detects_preserved_geometry() -> None:
    rng = np.random.default_rng(7)
    actual = rng.normal(size=(8, 128, 9)).astype(np.float32)
    latent = actual.reshape(8, -1).copy()
    metrics, arrays = runner.summarize(actual, actual, latent)
    assert metrics["raw_latent_distance_corr"] > 0.999999
    assert arrays["latent_distance_matrix"].shape == (8, 8)
    assert runner.round2_pass(8, metrics)


def revised_gate_row(subject: str, seed: int, *, nrmse: float = 0.20) -> dict:
    return {
        "subject_id": subject,
        "seed": seed,
        "improvement_pct": 99.5,
        "median_corr": 0.98,
        "median_nrmse": nrmse,
        "median_amplitude_ratio": 0.99,
        "raw_latent_distance_corr": 0.85,
        "nbm_mse": 0.01,
        "zero_mse": 2.0,
        "latent_variance": 0.4,
        "latent_between_window_variance": 0.3,
        "reconstruction_variance_retention": 0.98,
        "inference_ms_per_batch": 1.2,
        "parameter_count": 100,
    }


def test_revised_gate_preserves_strict_fail_but_allows_boundary_progression() -> None:
    rows = [
        revised_gate_row(subject, seed)
        for subject in runner.REPRESENTATIVES
        for seed in runner.SEEDS
    ]
    rows[-1]["median_nrmse"] = 0.516
    result = runner.evaluate_revised_round2_gate(rows)
    assert result["strict_stability_gate"] == "FAIL"
    assert result["engineering_progression_gate"] == "CONDITIONAL PASS"
    assert result["strict_pass_count"] == 11
    assert result["strict_failures"][0]["boundary_failure"]
    assert all(result["conditions"].values())


def test_revised_gate_rejects_catastrophic_failure() -> None:
    rows = [
        revised_gate_row(subject, seed)
        for subject in runner.REPRESENTATIVES
        for seed in runner.SEEDS
    ]
    rows[-1].update(
        improvement_pct=0.0,
        median_corr=0.0,
        median_nrmse=1.2,
        median_amplitude_ratio=0.0,
        reconstruction_variance_retention=0.0,
    )
    result = runner.evaluate_revised_round2_gate(rows)
    assert result["status"] == "FAIL"
    assert result["catastrophic_failures"]
    assert not result["conditions"]["B_all_runs_meet_safety_floor"]


def test_m3_strict_pass_has_selection_priority() -> None:
    conditional = runner.evaluate_revised_round2_gate(
        [
            revised_gate_row(subject, seed, nrmse=0.516 if (subject, seed) == ("S07", runner.SEEDS[-1]) else 0.20)
            for subject in runner.REPRESENTATIVES
            for seed in runner.SEEDS
        ]
    )
    strict = runner.evaluate_revised_round2_gate(
        [
            revised_gate_row(subject, seed)
            for subject in runner.REPRESENTATIVES
            for seed in runner.SEEDS
        ]
    )
    selected, _ = runner.select_revised_round2_architecture(
        {"M2_tcdae_wide": conditional, "M3_tcdae_long": strict}
    )
    assert selected == "M3_tcdae_long"
