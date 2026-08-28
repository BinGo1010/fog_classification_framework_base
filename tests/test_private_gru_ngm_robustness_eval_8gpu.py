from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_private_gru_ngm_robustness as worker
from scripts import launch_private_gru_ngm_robustness_eval_8gpu as launcher


def test_evaluation_grids_and_mask_rounding_are_frozen() -> None:
    assert worker.GAUSSIAN_SIGMAS == (0.0, 0.02, 0.04, 0.08, 0.12)
    assert worker.MASK_RHOS == (0.0, 0.025, 0.05, 0.10, 0.15)
    assert [worker.mask_sample_count(value) for value in worker.MASK_RHOS] == [
        0,
        3,
        6,
        13,
        19,
    ]
    contract = worker.evaluation_contract()
    assert contract["test_roles"] == [0, 1]
    assert contract["test_time_training"] is False


def test_gaussian_corruption_is_deterministic_and_paired() -> None:
    clean = np.zeros((3, 128, 30), dtype=np.float32)
    condition_seed = worker.paired_condition_seed("P01", 2, "gaussian", 0.08)
    first = worker.apply_gaussian_noise(clean, 0.08, condition_seed)
    second = worker.apply_gaussian_noise(clean, 0.08, condition_seed)
    np.testing.assert_array_equal(first, second)
    assert first.shape == clean.shape
    assert not np.array_equal(first, clean)
    assert worker.paired_condition_seed("P01", 2, "gaussian", 0.08) == (
        condition_seed
    )
    np.testing.assert_array_equal(
        worker.apply_gaussian_noise(clean, 0.0, condition_seed), clean
    )


def test_temporal_mask_is_one_contiguous_all_channel_interval_per_window() -> None:
    clean = np.ones((5, 128, 30), dtype=np.float32)
    condition_seed = worker.paired_condition_seed(
        "P03", 1, "temporal_mask", 0.10
    )
    masked, length = worker.apply_contiguous_time_mask(
        clean, 0.10, condition_seed
    )
    repeated, repeated_length = worker.apply_contiguous_time_mask(
        clean, 0.10, condition_seed
    )
    assert length == repeated_length == 13
    np.testing.assert_array_equal(masked, repeated)
    for window in masked:
        zero_times = np.flatnonzero(np.all(window == 0.0, axis=1))
        assert len(zero_times) == length
        np.testing.assert_array_equal(
            zero_times, np.arange(zero_times[0], zero_times[0] + length)
        )
        assert np.all(window[zero_times] == 0.0)
        assert np.all(window[np.setdiff1d(np.arange(128), zero_times)] == 1.0)


def test_observed_scheme_c_has_90_channels() -> None:
    rng = np.random.default_rng(7)
    observed = rng.normal(size=(3, 128, 30)).astype(np.float32)
    model = worker.base.GRUReconstructionNBM(
        channels=30,
        hidden=worker.base.HIDDEN,
        bottleneck=worker.base.BOTTLENECK,
    )
    features, diagnostics = worker.scheme_c_from_observed(
        model,
        np.ones(30, dtype=np.float32),
        observed,
        torch.device("cpu"),
        2,
    )
    assert features.shape == (3, 90, 128)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:, :30, :].mean(axis=2), 0.0, atol=2e-6)
    assert 0.0 <= diagnostics["clipped_fraction"] <= 1.0


def test_launcher_builds_one_evaluation_job_per_frozen_pipeline() -> None:
    args = Namespace(
        python="python",
        data_dir=Path("/data/private/processed_NBM_Exp"),
        trained_root=Path("/runs/private_robustness_tcn"),
        batch_size=128,
        overwrite=False,
    )
    plan = {
        "subjects": [f"P{index:02d}" for index in range(1, 9)],
        "folds": [0, 1, 2],
        "seeds": [0, 52, 161, 5216, 52161],
        "job_count": 240,
    }
    jobs = launcher.jobs(args, plan)
    assert len(jobs) == 240
    assert "P01_fold0_seed0_none" in jobs[0]["id"]
    assert "P08_fold2_seed52161_gaussian_mask" in jobs[-1]["id"]
    assert jobs[0]["command"][jobs[0]["command"].index("--device") + 1] == (
        "cuda:0"
    )


def synthetic_per_fold_rows(plan: dict) -> list[dict]:
    rows = []
    grids = {
        "gaussian": worker.GAUSSIAN_SIGMAS,
        "temporal_mask": worker.MASK_RHOS,
    }
    for arm in worker.training.ARMS:
        for subject_index, subject in enumerate(plan["subjects"]):
            for seed_index, seed in enumerate(plan["seeds"]):
                for fold in plan["folds"]:
                    for corruption_type, levels in grids.items():
                        for level_index, level in enumerate(levels):
                            if arm == "none":
                                ap = 0.80 - 0.05 * level_index
                            else:
                                ap = 0.78 - 0.02 * level_index
                            ap += 0.01 * subject_index + 0.001 * seed_index
                            ap += 0.003 * fold
                            mask_samples = (
                                worker.mask_sample_count(level)
                                if corruption_type == "temporal_mask"
                                else 0
                            )
                            rows.append(
                                {
                                    "arm": arm,
                                    "arm_display_name": worker.training.ARM_DISPLAY_NAMES[
                                        arm
                                    ],
                                    "subject": subject,
                                    "fold": fold,
                                    "seed": seed,
                                    "corruption_type": corruption_type,
                                    "x_name": (
                                        "sigma_test"
                                        if corruption_type == "gaussian"
                                        else "rho_mask"
                                    ),
                                    "x_value": level,
                                    "x_percent": (
                                        level * 100.0
                                        if corruption_type == "temporal_mask"
                                        else ""
                                    ),
                                    "mask_samples": mask_samples,
                                    "realized_mask_fraction": mask_samples / 128,
                                    "ap": ap,
                                }
                            )
    return rows


def test_hierarchical_aggregation_and_paired_arm_difference() -> None:
    plan = {
        "subjects": ["P01", "P02"],
        "folds": [0, 1, 2],
        "seeds": [0, 52],
    }
    per_fold = synthetic_per_fold_rows(plan)
    subject_seed = launcher.build_subject_seed_rows(per_fold, plan)
    subject_summary = launcher.build_subject_summary_rows(subject_seed, plan)
    overall_seed = launcher.build_overall_seed_rows(subject_seed, plan)
    curve = launcher.build_curve_rows(overall_seed, plan)
    fig1 = launcher.build_figure_wide_rows(
        curve, overall_seed, "gaussian", plan
    )
    fig2 = launcher.build_figure_wide_rows(
        curve, overall_seed, "temporal_mask", plan
    )

    assert len(subject_seed) == 2 * 2 * 2 * 10
    assert len(subject_summary) == 2 * 2 * 10
    assert len(overall_seed) == 2 * 2 * 10
    assert len(curve) == 2 * 10
    assert len(fig1) == len(fig2) == 5

    target = next(
        row
        for row in subject_seed
        if row["arm"] == "none"
        and row["subject"] == "P01"
        and row["seed"] == 0
        and row["corruption_type"] == "gaussian"
        and row["x_value"] == 0.12
    )
    # Fold values are 0.60, 0.603, and 0.606.
    assert abs(target["ap"] - 0.603) < 1e-12
    highest_noise = fig1[-1]
    # At level index 4: Gaussian+Mask is 0.70 and None is 0.60 before
    # common subject/seed/fold offsets, so the paired difference is 0.10.
    assert abs(highest_noise["gaussian_mask_minus_none_ap_mean"] - 0.10) < 1e-12
    assert highest_noise["no_perturbation_ap_drop_from_clean_mean"] > (
        highest_noise["gaussian_mask_ap_drop_from_clean_mean"]
    )
