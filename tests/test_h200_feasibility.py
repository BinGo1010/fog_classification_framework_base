from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from cnbr_fog.h200_feasibility import (
    ARM_SPECS,
    H200_ARM_NAMES,
    DualBranchRF125Classifier,
    SubjectCrossFitPlan,
    build_arm_inputs,
    build_classifier,
    build_subject_crossfit_plan,
    band_power_error,
    calibrate_persistence_sigma,
    derive_forecast_primitives,
    ensemble_scaled_gaussians_to_outer,
    evaluate_phase2_gate,
    forecast_diagnostics,
    gaussian_moment_match,
    optimal_cross_correlation_lag,
    paired_bootstrap,
    persistence_forecast_diagnostics,
    physical_gaussian_to_scaled,
    rf125_receptive_field,
    scaled_gaussian_to_physical,
    validate_subject_crossfit_plan,
)


def _forecast_arrays(windows: int = 2):
    raw = np.full((windows, 9, 128), 3.0, dtype=np.float32)
    mean = np.full_like(raw, 1.0)
    sigma = np.full_like(raw, 0.5)
    return raw, mean, sigma


def test_arm_registry_is_the_preregistered_five_arm_matrix() -> None:
    assert H200_ARM_NAMES == (
        "raw4",
        "raw6",
        "normality",
        "raw4_zero",
        "raw4_normality",
    )
    assert ARM_SPECS["raw4"].input_samples == 256
    assert ARM_SPECS["raw6"].input_samples == 384
    assert ARM_SPECS["normality"].normality_channels == 18
    zero = ARM_SPECS["raw4_zero"]
    observed = ARM_SPECS["raw4_normality"]
    assert zero.classifier_kind == observed.classifier_kind == "dual"
    assert zero.raw_channels == observed.raw_channels == 9
    assert zero.normality_channels == observed.normality_channels == 18
    assert zero.input_samples == observed.input_samples == 256


def test_build_arm_inputs_preserves_alignment_and_zero_control() -> None:
    rng = np.random.default_rng(4)
    raw4 = rng.normal(size=(3, 9, 256)).astype(np.float32)
    raw6 = rng.normal(size=(3, 9, 384)).astype(np.float32)
    z4 = np.clip(
        rng.normal(size=(3, 9, 256)), -12, 12
    ).astype(np.float32)
    log_sigma4 = rng.normal(size=(3, 9, 256)).astype(np.float32)
    arms = build_arm_inputs(raw4, raw6, z4, log_sigma4)

    assert set(arms) == set(H200_ARM_NAMES)
    assert arms["raw4"].shape == (3, 9, 256)
    assert arms["raw6"].shape == (3, 9, 384)
    assert arms["normality"].shape == (3, 18, 256)
    assert arms["raw4_zero"].shape == (3, 27, 256)
    assert arms["raw4_normality"].shape == (3, 27, 256)
    np.testing.assert_array_equal(arms["normality"][:, :9], z4)
    np.testing.assert_array_equal(arms["normality"][:, 9:], log_sigma4)
    np.testing.assert_array_equal(arms["raw4_zero"][:, :9], raw4)
    assert np.count_nonzero(arms["raw4_zero"][:, 9:]) == 0
    np.testing.assert_array_equal(
        arms["raw4_normality"][:, 9:], arms["normality"]
    )

    with pytest.raises(ValueError, match="clipping"):
        build_arm_inputs(raw4, raw6, np.full_like(z4, 13), log_sigma4)


def test_rf125_single_and_dual_classifiers_validate_and_forward() -> None:
    assert rf125_receptive_field() == 125
    torch.manual_seed(8)
    raw4_model = build_classifier("raw4", hidden_channels=4, dropout=0.0)
    raw6_model = build_classifier("raw6", hidden_channels=4, dropout=0.0)
    normal_model = build_classifier(
        "normality", hidden_channels=4, dropout=0.0
    )
    zero_model = build_classifier(
        "raw4_zero", hidden_channels=4, dropout=0.0
    )
    fusion_model = build_classifier(
        "raw4_normality", hidden_channels=4, dropout=0.0
    )
    assert isinstance(zero_model, DualBranchRF125Classifier)
    assert isinstance(fusion_model, DualBranchRF125Classifier)
    assert zero_model.architecture_config() == fusion_model.architecture_config()
    assert {
        key: tuple(value.shape) for key, value in zero_model.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in fusion_model.state_dict().items()
    }

    raw4_model.eval()
    raw6_model.eval()
    normal_model.eval()
    zero_model.eval()
    with torch.no_grad():
        assert raw4_model(torch.zeros(2, 9, 256)).shape == (2,)
        assert raw6_model(torch.zeros(2, 9, 384)).shape == (2,)
        assert normal_model(torch.zeros(2, 18, 256)).shape == (2,)
        packed = torch.zeros(2, 27, 256)
        packed_logits = zero_model(packed)
        split_logits = zero_model(packed[:, :9], packed[:, 9:])
    assert packed_logits.shape == (2,)
    torch.testing.assert_close(packed_logits, split_logits)

    with pytest.raises(ValueError, match="channels"):
        raw4_model(torch.zeros(2, 8, 256))
    bad = torch.zeros(2, 9, 256)
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        raw4_model(bad)
    with pytest.raises(ValueError, match="batch sizes"):
        zero_model(torch.zeros(2, 9, 256), torch.zeros(3, 18, 256))


def test_forecast_primitives_are_signed_clipped_and_uncertainty_aware() -> None:
    raw, mean, sigma = _forecast_arrays()
    sigma[:, :, 0] = 0.1
    primitives = derive_forecast_primitives(raw, mean, sigma)
    assert set(primitives) == {
        "raw",
        "mean",
        "mu",
        "sigma",
        "error",
        "z",
        "log_sigma",
        "gaussian_nll",
    }
    np.testing.assert_allclose(primitives["error"], 2.0)
    assert np.all(primitives["z"][:, :, 0] == 12.0)
    np.testing.assert_allclose(
        primitives["z"][:, :, 1:], 4.0, rtol=1e-6
    )
    np.testing.assert_allclose(primitives["log_sigma"], np.log(sigma))
    assert np.all(primitives["gaussian_nll"][:, :, 0] > 100)

    broadcast = derive_forecast_primitives(raw, mean, sigma[:1])
    assert broadcast["sigma"].shape == raw.shape
    with pytest.raises(ValueError, match="positive"):
        derive_forecast_primitives(raw, mean, np.zeros_like(sigma))
    with pytest.raises(TypeError, match="float32"):
        derive_forecast_primitives(raw.astype(np.float64), mean, sigma)


def test_gaussian_moment_matching_includes_between_model_variance() -> None:
    means = np.asarray([[[0.0]], [[2.0]]], dtype=np.float32)
    sigmas = np.ones_like(means)
    mean, sigma = gaussian_moment_match(means, sigmas)
    np.testing.assert_allclose(mean, [[1.0]])
    np.testing.assert_allclose(sigma, [[np.sqrt(2.0)]], rtol=1e-6)

    with pytest.raises(ValueError, match="sigma > 0"):
        gaussian_moment_match(means, np.zeros_like(sigmas))


def test_scaled_physical_round_trip_and_outer_ensemble() -> None:
    mean_scaled = np.arange(18, dtype=np.float32).reshape(1, 9, 2)
    sigma_scaled = np.full_like(mean_scaled, 0.5)
    center = np.linspace(-1, 1, 9, dtype=np.float32)
    scale = np.linspace(0.5, 2.0, 9, dtype=np.float32)
    physical_mean, physical_sigma = scaled_gaussian_to_physical(
        mean_scaled, sigma_scaled, center, scale
    )
    restored_mean, restored_sigma = physical_gaussian_to_scaled(
        physical_mean, physical_sigma, center, scale
    )
    np.testing.assert_allclose(restored_mean, mean_scaled, rtol=1e-6)
    np.testing.assert_allclose(restored_sigma, sigma_scaled, rtol=1e-6)

    physical_target_mean = np.full((1, 9, 2), 10.0, dtype=np.float32)
    physical_target_sigma = np.full_like(physical_target_mean, 2.0)
    inner_centers = np.stack(
        [np.zeros(9, dtype=np.float32), np.full(9, 5, dtype=np.float32)]
    )
    inner_scales = np.stack(
        [np.full(9, 2, dtype=np.float32), np.ones(9, dtype=np.float32)]
    )
    scaled_means = []
    scaled_sigmas = []
    for index in range(2):
        inner_mean, inner_sigma = physical_gaussian_to_scaled(
            physical_target_mean,
            physical_target_sigma,
            inner_centers[index],
            inner_scales[index],
        )
        scaled_means.append(inner_mean)
        scaled_sigmas.append(inner_sigma)
    outer_mean, outer_sigma = ensemble_scaled_gaussians_to_outer(
        np.stack(scaled_means),
        np.stack(scaled_sigmas),
        inner_centers,
        inner_scales,
        np.full(9, 1, dtype=np.float32),
        np.full(9, 3, dtype=np.float32),
    )
    np.testing.assert_allclose(outer_mean, 3.0, rtol=1e-6)
    np.testing.assert_allclose(outer_sigma, 2.0 / 3.0, rtol=1e-6)


def test_crossfit_plans_hold_every_subject_out_exactly_once() -> None:
    subjects = ("S03", "S05", "S06", "S07", "S08", "S09")
    three_fold = build_subject_crossfit_plan(subjects, "3fold")
    assert len(three_fold.folds) == 3
    assert all(len(fold.train_subjects) == 4 for fold in three_fold.folds)
    assert all(len(fold.heldout_subjects) == 2 for fold in three_fold.folds)
    for fold in three_fold.folds:
        assert not set(fold.train_subjects) & set(fold.heldout_subjects)
    assert sorted(
        subject
        for fold in three_fold.folds
        for subject in fold.heldout_subjects
    ) == sorted(subjects)

    loto = build_subject_crossfit_plan(subjects, "loto")
    assert len(loto.folds) == 6
    assert all(len(fold.train_subjects) == 5 for fold in loto.folds)
    assert all(len(fold.heldout_subjects) == 1 for fold in loto.folds)

    broken_fold = replace(
        three_fold.folds[0],
        train_subjects=three_fold.folds[0].train_subjects
        + (three_fold.folds[0].heldout_subjects[0],),
    )
    broken = SubjectCrossFitPlan(
        scheme="3fold",
        subjects=three_fold.subjects,
        folds=(broken_fold,) + three_fold.folds[1:],
    )
    with pytest.raises(ValueError, match="held-out"):
        validate_subject_crossfit_plan(broken)


def test_paired_subject_bootstrap_is_paired_and_deterministic() -> None:
    reference = {f"S{i}": 0.4 + i * 0.01 for i in range(8)}
    candidate = {subject: value + 0.1 for subject, value in reference.items()}
    first = paired_bootstrap(candidate, reference, samples=1000, seed=9)
    second = paired_bootstrap(candidate, reference, samples=1000, seed=9)
    assert first == second
    assert first["n_subjects"] == 8
    assert first["positive_subjects"] == 8
    assert first["mean_delta"] == pytest.approx(0.1)
    assert first["ci_low"] == pytest.approx(0.1)
    assert first["ci_high"] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="different keys"):
        paired_bootstrap({"S01": 1.0}, {"S02": 1.0}, samples=10)


def _gate_inputs(fusion_delta: float) -> dict:
    return {
        "subject_ids": tuple(f"S{i + 1:02d}" for i in range(8)),
        "fusion_pr_auc": np.full(8, 0.40 + fusion_delta),
        "raw6_pr_auc": np.full(8, 0.40),
        "zero_pr_auc": np.full(8, 0.41),
        "normality_pr_auc": np.full(8, 0.30),
        "prevalence": np.full(8, 0.10),
        "fusion_recall": np.full(8, 0.81),
        "raw6_recall": np.full(8, 0.80),
        "fusion_false_alarms_per_hour": np.full(8, 1.0),
        "raw6_false_alarms_per_hour": np.full(8, 1.0),
        "bootstrap_samples": 1000,
    }


def test_phase2_gate_returns_strong_conditional_and_stop() -> None:
    strong = evaluate_phase2_gate(**_gate_inputs(0.04))
    assert strong["decision"] == "strong_go"
    assert strong["practical_pr_auc"]["positive_subjects"] == 8
    assert strong["mechanism_pr_auc_delta"] == pytest.approx(0.03)

    conditional = evaluate_phase2_gate(**_gate_inputs(0.015))
    assert conditional["decision"] == "conditional_go"
    assert not conditional["stop_reasons"]

    no_secondary = _gate_inputs(0.015)
    no_secondary["fusion_recall"] = np.full(8, 0.80)
    stopped_secondary = evaluate_phase2_gate(**no_secondary)
    assert stopped_secondary["decision"] == "stop"
    assert "no_consistent_secondary_metric_improvement" in stopped_secondary[
        "stop_reasons"
    ]

    stopped_inputs = _gate_inputs(-0.02)
    stopped = evaluate_phase2_gate(**stopped_inputs)
    assert stopped["decision"] == "stop"
    assert "no_positive_pr_auc_delta" in stopped["stop_reasons"]


def test_phase2_gate_rejects_bad_subject_identity_and_metric_ranges() -> None:
    duplicate_subjects = _gate_inputs(0.03)
    duplicate_subjects["subject_ids"] = ("S01",) * 8
    with pytest.raises(ValueError, match="unique"):
        evaluate_phase2_gate(**duplicate_subjects)

    invalid_metric = _gate_inputs(0.03)
    invalid_metric["fusion_recall"] = np.full(8, 1.1)
    with pytest.raises(ValueError, match="fusion_recall"):
        evaluate_phase2_gate(**invalid_metric)


def test_forecast_diagnostics_has_four_quartiles_coverage_and_mask() -> None:
    target = np.zeros((3, 9, 128), dtype=np.float32)
    mean = np.zeros_like(target)
    sigma = np.ones_like(target)
    primitives = derive_forecast_primitives(target, mean, sigma)
    diagnostics = forecast_diagnostics(
        primitives, mask=np.asarray([True, False, True])
    )
    assert diagnostics["windows"] == 2
    assert len(diagnostics["lead_quartiles"]) == 4
    assert [item["start_lead_sample"] for item in diagnostics["lead_quartiles"]] == [
        0,
        32,
        64,
        96,
    ]
    overall = diagnostics["overall"]
    assert overall["nll"] == pytest.approx(0.0)
    assert overall["rmse"] == pytest.approx(0.0)
    assert overall["mae"] == pytest.approx(0.0)
    assert overall["coverage_1sigma"] == pytest.approx(1.0)
    assert overall["coverage_2sigma"] == pytest.approx(1.0)
    assert len(diagnostics["per_channel"]) == 9
    assert diagnostics["signal_diagnostic_windows"] == 2
    assert diagnostics["cross_correlation"]["median_lag_samples"] == 0
    assert set(diagnostics["band_power_error"]) == {"0.5-3Hz", "3-8Hz"}


def test_phase0_lag_and_band_power_toy_signals() -> None:
    rng = np.random.default_rng(21)
    target = rng.normal(size=(2, 9, 128)).astype(np.float32)
    prediction = np.zeros_like(target)
    prediction[..., :-3] = target[..., 3:]
    lags, correlations = optimal_cross_correlation_lag(
        target, prediction, max_lag=8
    )
    np.testing.assert_array_equal(lags, np.full((2, 9), 3))
    np.testing.assert_allclose(correlations, 1.0, atol=1e-6)

    time = np.arange(128, dtype=np.float32) / 64.0
    low = np.sin(2 * np.pi * 2.0 * time)[None, None, :]
    high = np.sin(2 * np.pi * 5.0 * time)[None, None, :]
    low = np.broadcast_to(low, (2, 9, 128)).astype(np.float32).copy()
    high = np.broadcast_to(high, (2, 9, 128)).astype(np.float32).copy()
    identical = band_power_error(low, low)
    assert identical["0.5-3Hz"]["mean_absolute_error"] == pytest.approx(0.0)
    assert identical["3-8Hz"]["mean_absolute_error"] == pytest.approx(0.0)
    separated = band_power_error(low, high)
    assert (
        separated["0.5-3Hz"]["target_mean_power"]
        > separated["0.5-3Hz"]["prediction_mean_power"] * 100
    )
    assert (
        separated["3-8Hz"]["prediction_mean_power"]
        > separated["3-8Hz"]["target_mean_power"] * 100
    )

    diagnostics = forecast_diagnostics(
        target,
        prediction,
        np.ones_like(target),
        max_lag=8,
        diagnostic_max_windows=1,
    )
    assert diagnostics["signal_diagnostic_windows"] == 1
    assert diagnostics["cross_correlation"]["median_lag_samples"] == 3


def test_persistence_sigma_uses_window_mle_and_broadcasts_for_diagnostics() -> None:
    context = np.ones((4, 9, 128), dtype=np.float32)
    target = np.full((4, 9, 128), 3.0, dtype=np.float32)
    sigma = calibrate_persistence_sigma(context, target, epsilon=1e-8)
    assert sigma.shape == (1, 9, 128)
    np.testing.assert_allclose(sigma, 2.0, rtol=1e-6)

    diagnostics = persistence_forecast_diagnostics(context, target, sigma)
    assert diagnostics["overall"]["rmse"] == pytest.approx(2.0)
    assert diagnostics["overall"]["mae"] == pytest.approx(2.0)
    assert diagnostics["overall"]["coverage_1sigma"] == pytest.approx(1.0)
    assert diagnostics["overall"]["nll"] == pytest.approx(
        np.log(2.0) + 0.5, rel=1e-6
    )
