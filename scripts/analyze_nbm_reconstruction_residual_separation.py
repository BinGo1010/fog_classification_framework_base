#!/usr/bin/env python
"""Objective reconstruction and residual-separation diagnostics for final NBM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
DEFAULT_SUITE = (
    ROOT
    / "outputs"
    / "nonfog_gru_nbm_inceptiontime_within_subject_distributed_calibration_bottleneck32_finalnbm200_pat10_seed20260802"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("inf")


def split_separation(
    residual_unclipped: np.ndarray,
    residual_classifier: np.ndarray,
    reconstruction_mu: np.ndarray,
    bias: np.ndarray,
    sigma: np.ndarray,
    labels: np.ndarray,
    prefix: str,
) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    non_fog = labels == 0
    fog = labels == 1
    # Robust scalar magnitude per window before clipping and post-clip centering.
    pre_score = np.median(np.abs(residual_unclipped), axis=(1, 2))
    classifier_score = np.mean(np.abs(residual_classifier), axis=(1, 2))
    reconstruction_error = (
        residual_unclipped * sigma[None, None, :] + bias[None, None, :]
    )
    scaled_input = reconstruction_mu + reconstruction_error

    def smooth_l1(values: np.ndarray) -> np.ndarray:
        absolute = np.abs(values)
        return np.where(absolute < 1.0, 0.5 * values**2, absolute - 0.5)

    model_window_huber = smooth_l1(reconstruction_error).mean(axis=(1, 2))
    zero_window_huber = smooth_l1(scaled_input).mean(axis=(1, 2))
    nf_model_huber = float(np.mean(model_window_huber[non_fog]))
    nf_zero_huber = float(np.mean(zero_window_huber[non_fog]))
    fog_model_huber = float(np.mean(model_window_huber[fog]))
    nf_median = float(np.median(pre_score[non_fog]))
    fog_median = float(np.median(pre_score[fog]))
    nf_q95 = float(np.quantile(pre_score[non_fog], 0.95))
    output = {
        f"{prefix}_non_fog_windows": int(non_fog.sum()),
        f"{prefix}_fog_windows": int(fog.sum()),
        f"{prefix}_non_fog_nbm_huber": nf_model_huber,
        f"{prefix}_non_fog_zero_baseline_huber": nf_zero_huber,
        f"{prefix}_non_fog_huber_relative_improvement_vs_zero": safe_ratio(
            nf_zero_huber - nf_model_huber, nf_zero_huber
        ),
        f"{prefix}_fog_nbm_huber": fog_model_huber,
        f"{prefix}_fog_to_non_fog_nbm_huber_ratio": safe_ratio(
            fog_model_huber, nf_model_huber
        ),
        f"{prefix}_preclip_window_abs_median_non_fog": nf_median,
        f"{prefix}_preclip_window_abs_median_fog": fog_median,
        f"{prefix}_preclip_fog_to_non_fog_median_ratio": safe_ratio(
            fog_median, nf_median
        ),
        f"{prefix}_preclip_magnitude_auroc": float(
            roc_auc_score(labels, pre_score)
        ),
        f"{prefix}_preclip_magnitude_pr_auc": float(
            average_precision_score(labels, pre_score)
        ),
        f"{prefix}_preclip_non_fog_q95": nf_q95,
        f"{prefix}_fog_recall_at_non_fog_q95": float(
            np.mean(pre_score[fog] > nf_q95)
        ),
        f"{prefix}_classifier_input_magnitude_auroc": float(
            roc_auc_score(labels, classifier_score)
        ),
        f"{prefix}_classifier_input_magnitude_pr_auc": float(
            average_precision_score(labels, classifier_score)
        ),
    }
    return output


def main() -> None:
    suite = parse_args().suite_dir.resolve()
    rows = []
    for subject in SUBJECTS:
        directory = suite / subject
        final_nbm = read_json(directory / "artifacts" / "final_nbm.json")
        diagnostics = read_json(directory / "residual_diagnostics.json")
        calibration = final_nbm["residual_calibration"]
        bias = np.asarray(calibration["bias"], dtype=np.float32)
        sigma = np.asarray(calibration["sigma"], dtype=np.float32)
        with np.load(directory / "artifacts" / "residuals.npz") as arrays:
            validation = split_separation(
                arrays["validation_unclipped"],
                arrays["validation_residual"],
                arrays["validation_mu"],
                bias,
                sigma,
                arrays["validation_y"],
                "validation",
            )
            test = split_separation(
                arrays["test_unclipped"],
                arrays["test_residual"],
                arrays["test_mu"],
                bias,
                sigma,
                arrays["test_y"],
                "test",
            )
        rows.append(
            {
                "subject": subject,
                "nbm_fit_windows": final_nbm["fit_windows"],
                "nbm_calibration_windows": final_nbm[
                    "calibration_validation_windows"
                ],
                "nbm_best_epoch": final_nbm["best_epoch"],
                "nbm_epochs_completed": final_nbm["epochs_completed"],
                "nbm_validation_huber": final_nbm["best_validation_huber"],
                "nbm_parameter_count": final_nbm["parameter_count"],
                "sigma_min": float(np.min(calibration["sigma"])),
                "sigma_max": float(np.max(calibration["sigma"])),
                "sigma_floor_channel_count": len(
                    calibration["floor_applied_channels"]
                ),
                "oof_vs_validation_non_fog_median_abs_ratio": diagnostics[
                    "oof_vs_validation_nonfog_median_abs_ratio"
                ],
                "test_vs_validation_non_fog_median_abs_ratio": diagnostics[
                    "test_vs_validation_nonfog_median_abs_ratio"
                ],
                "test_non_fog_clip_fraction": diagnostics["test_final_nbm"][
                    "non_fog"
                ]["clip_fraction"],
                "test_fog_clip_fraction": diagnostics["test_final_nbm"]["fog"][
                    "clip_fraction"
                ],
                **validation,
                **test,
            }
        )

    with (suite / "nbm_reconstruction_residual_diagnostics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "residual_score": (
            "per-window median absolute standardized residual before clipping "
            "and post-clip window-axis centering"
        ),
        "interpretation": {
            "fog_to_non_fog_ratio": ">1 means larger FoG residual magnitude",
            "auroc": (
                "probability that a random FoG window has a higher residual "
                "magnitude than a random non-FoG window; 0.5 is chance"
            ),
            "fog_recall_at_non_fog_q95": (
                "FoG fraction above a threshold exceeded by approximately 5% "
                "of non-FoG windows"
            ),
        },
        "per_subject": rows,
        "means": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "nbm_validation_huber",
                "validation_preclip_fog_to_non_fog_median_ratio",
                "validation_preclip_magnitude_auroc",
                "validation_preclip_magnitude_pr_auc",
                "validation_fog_recall_at_non_fog_q95",
                "test_preclip_fog_to_non_fog_median_ratio",
                "test_preclip_magnitude_auroc",
                "test_preclip_magnitude_pr_auc",
                "test_fog_recall_at_non_fog_q95",
            )
        },
    }
    with (suite / "nbm_reconstruction_residual_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
