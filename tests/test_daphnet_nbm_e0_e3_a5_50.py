from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.nbm_e0_e3 import (  # noqa: E402
    HistoryPredictor,
    TrueBottleneckAE,
    apply_c1,
    benjamini_hochberg,
    build_e3b_input,
    chronological_calibration_split,
    fit_c1_mad,
    fixed_quarter_masks,
    reconstruction_rows,
    score_shift_metrics,
    threshold_metrics,
)
import run_daphnet_nbm_e0_e3_a5_50 as runner  # noqa: E402


def test_e2_models_have_true_bottlenecks_and_exact_output_shape() -> None:
    expected = {"P24": (24, 32, 768), "P16": (32, 16, 512)}
    for variant, (channels, samples, elements) in expected.items():
        model = TrueBottleneckAE(variant)
        output, latent = model(torch.randn(2, 9, 128))
        assert output.shape == (2, 9, 128)
        assert latent.shape == (2, channels, samples)
        config = model.architecture_config()
        assert config["latent_elements"] == elements
        assert config["latent_elements"] < config["input_elements"]
        output.square().mean().backward()


def test_e3_models_are_causal_shape_preserving_predictors() -> None:
    e3a = HistoryPredictor(24, 9)
    predicted, latent = e3a(torch.randn(2, 9, 256))
    assert predicted.shape == (2, 9, 128)
    assert latent.shape == (2, 24, 32)
    assert not e3a.architecture_config()["target_visible_in_input"]

    history = torch.randn(2, 9, 128)
    target = torch.randn(2, 9, 128)
    masks = fixed_quarter_masks(2, torch.device("cpu"))
    assert torch.stack([mask.sum() for mask in masks]).tolist() == [64, 64, 64, 64]
    e3b_input = build_e3b_input(history, target, masks[0])
    assert e3b_input.shape == (2, 10, 256)
    assert torch.all(e3b_input[:, :9, 128:160] == 0)
    assert torch.all(e3b_input[:, 9, 128:160] == 0)
    assert torch.all(e3b_input[:, 9, 160:] == 1)
    assert HistoryPredictor(24, 10)(e3b_input)[0].shape == (2, 9, 128)


def test_saved_run_round_trip_supports_cross_shard_aggregation(tmp_path: Path) -> None:
    role_rows = {
        role: [{"window_id": f"{role}_w0"}] for role in runner.ROLES
    }
    arrays = {
        runner.ARRAY_NAMES[role]: np.full((1, 128, 9), index, dtype=np.float32)
        for index, role in enumerate(runner.ROLES)
    }
    bundle = runner.SubjectBundle(
        subject="S01",
        scope="formal",
        role_rows=role_rows,
        raw=arrays,
        processed=arrays,
        scaler={},
        records={},
        split_runs={},
    )
    run_dir = tmp_path / "E2" / "training" / "S01" / "seed20260802"
    predicted = {name: values * 0.5 for name, values in arrays.items()}
    original = runner.RunOutputs(
        stage="E2",
        subject="S01",
        seed=20260802,
        model_name="E2_P24",
        model_config={"latent_elements": 768},
        training={"best_epoch": 4},
        rows=role_rows,
        actual=arrays,
        predicted=predicted,
        residual={name: arrays[name] - predicted[name] for name in arrays},
        run_dir=run_dir,
    )
    runner.save_run_outputs(original)
    loaded = runner.load_saved_runs(
        tmp_path / "E2",
        {"S01": bundle},
        (20260802,),
        expected_stage="E2",
    )[("S01", 20260802)]
    assert loaded.model_name == "E2_P24"
    assert loaded.training["best_epoch"] == 4
    assert loaded.rows["external_test_fog"][0]["window_id"] == "external_test_fog_w0"
    assert np.array_equal(loaded.predicted["test_fog"], predicted["test_fog"])


def test_e2_gate_excludes_diagnostic_and_clean_controls_from_fog_gain() -> None:
    e1_test = []
    e2_test = []
    reconstruction = []
    e1_shift = []
    e2_shift = []
    for subject in runner.FORMAL_SUBJECTS:
        for seed in runner.SEEDS:
            e1_test.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "auroc": 0.60,
                    "cliffs_delta": 0.20,
                    "fog_p50": 2.0,
                    "nonfog_p50": 1.0,
                }
            )
            e2_test.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "auroc": 0.65,
                    "cliffs_delta": 0.40,
                    "fog_p50": 2.2,
                    "nonfog_p50": 1.0,
                }
            )
            reconstruction.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "pearson_median": 0.80,
                    "nrmse_median": 0.30,
                    "nrmse_p90": 0.70,
                }
            )
            e1_shift.append({"subject_id": subject, "seed": seed, "shift_robust": 1.0})
            e2_shift.append({"subject_id": subject, "seed": seed, "shift_robust": 1.0})
    for subject in (*runner.DIAGNOSTIC_SUBJECTS, *runner.CLEAN_CONTROLS):
        for seed in runner.SEEDS:
            e1_test.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "auroc": np.nan,
                    "cliffs_delta": np.nan,
                    "fog_p50": np.nan,
                    "nonfog_p50": np.nan,
                }
            )
            e2_test.append(dict(e1_test[-1]))
    gate = runner.e2_gate(
        {"test_rows": e1_test, "shift_rows": e1_shift},
        {
            "test_rows": e2_test,
            "shift_rows": e2_shift,
            "reconstruction_rows": reconstruction,
        },
    )
    assert gate["status"] == "PASS"
    assert np.isclose(gate["median_differential_fog_minus_nonfog_error_gain"], 0.2)
    assert gate["differential_gain_subjects"] == 7
    assert gate["differential_gain_positive_subjects"] == 7
    assert set(gate["differential_gain_by_formal_subject"]) == set(runner.FORMAL_SUBJECTS)


def test_c1_mad_removes_channel_bias_and_uses_robust_scale() -> None:
    generator = np.random.default_rng(4)
    residual = generator.laplace(size=(20, 128, 9)).astype(np.float32)
    residual += np.arange(9, dtype=np.float32).reshape(1, 1, -1)
    parameters = fit_c1_mad(residual)
    calibrated = apply_c1(residual, parameters)
    assert np.allclose(np.median(calibrated, axis=(0, 1)), 0.0, atol=2e-5)
    robust_sigma = 1.4826 * np.median(np.abs(calibrated), axis=(0, 1))
    assert np.allclose(robust_sigma, 1.0, atol=2e-4)


def test_chronological_calibration_split_has_no_window_overlap() -> None:
    rows = [
        {
            "window_id": f"w{index}",
            "record_id": "R1",
            "start_index": index * 64,
            "end_index_exclusive": index * 64 + 128,
        }
        for index in range(12)
    ]
    calibration, threshold, audit = chronological_calibration_split(rows)
    assert len(calibration) > 0 and len(threshold) > 0
    assert max(int(rows[index]["end_index_exclusive"]) for index in calibration) <= min(
        int(rows[index]["start_index"]) for index in threshold
    )
    assert {row["c1_role"] for row in audit} == {
        "calibration",
        "score_threshold",
        "embargo_dropped",
    }


def test_v1_shift_threshold_and_bh_metrics_are_finite() -> None:
    generator = np.random.default_rng(7)
    actual = generator.normal(size=(8, 128, 9)).astype(np.float32)
    summary, channels, windows = reconstruction_rows(actual, actual.copy())
    assert summary["nrmse_median"] == 0.0
    assert np.isclose(summary["pearson_median"], 1.0)
    assert len(channels) == 9 and len(windows) == 8
    shift = score_shift_metrics(np.arange(20), np.arange(20) + 2)
    assert shift["shift_median"] == 2.0
    assert shift["wasserstein_distance"] == 2.0

    rows = []
    scores = []
    for index in range(20):
        fog = 8 <= index <= 11
        rows.append(
            {
                "record_id": "R1",
                "start_index": index * 64,
                "end_index_exclusive": index * 64 + 128,
                "y_binary": int(fog),
                "event_id": "F1" if fog else "",
            }
        )
        scores.append(10.0 if fog else 0.0)
    threshold = threshold_metrics(rows, np.asarray(scores), np.zeros(100), quantile=99.2)
    assert threshold["window_recall"] == 1.0
    assert threshold["event_fog_recall"] == 1.0
    assert threshold["false_alarm_event_count"] == 0
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_real_a5_50_context_manifest_never_overlaps_target_or_split_boundary() -> None:
    data_dir = (
        ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_A5_50"
    )
    if not data_dir.exists():
        return
    bundles, _, _ = runner.load_data(data_dir, ("S01",), max_windows_per_role=64)
    item = bundles["S01"]
    context, target, rows, audit = runner.context_target_arrays(
        item, "nbm_internal_train_nonfog", mode="E3A"
    )
    assert context.shape[1:] == (256, 9)
    assert target.shape[1:] == (128, 9)
    assert len(context) == len(rows) > 0
    kept = [row for row in audit if row["kept"]]
    assert all(row["context_end_index_exclusive"] == row["target_start_index"] for row in kept)
    assert all(row["history_target_overlap_samples"] == 0 for row in kept)
    assert all(row["same_manifest_split"] for row in kept)
