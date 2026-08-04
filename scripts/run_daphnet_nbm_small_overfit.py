"""Compatibility helpers for the completed GRU-NBM small-overfit experiment.

The completed results remain under ``outputs/daphnet_nbm_small_overfit_v1_*``.
Selection primitives live in ``daphnet_small_sample_selection`` so subsequent
diagnostics can reuse the exact frozen pool and deterministic window choices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
for location in (SCRIPT_ROOT, REPO_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from daphnet_small_sample_selection import *  # noqa: F403
from daphnet_small_sample_selection import DaphnetDataset, current


THRESHOLDS = {
    1: {"improvement_pct": 99.0, "median_corr": 0.99, "median_nrmse": 0.10},
    8: {"improvement_pct": 95.0, "median_corr": 0.95, "median_nrmse": 0.20},
    32: {"improvement_pct": 80.0, "median_corr": 0.90, "median_nrmse": 0.35},
    128: {"improvement_pct": 50.0, "median_corr": 0.75, "median_nrmse": 0.60},
}


def prepare_run_data(records, windows, selected):
    scaler = current.fit_scaler_unique_points(records, windows, selected)
    raw = current.raw_windows(records, windows, selected)
    values = current.prepare_nbm_windows(scaler, raw, center=True)
    return np.ascontiguousarray(values), scaler


def smooth_l1_np(
    actual: np.ndarray, predicted: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    difference = np.abs(np.asarray(actual, dtype=np.float64) - predicted)
    return np.where(
        difference < beta,
        0.5 * np.square(difference) / beta,
        difference - 0.5 * beta,
    )


def summarize_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    centered_actual = actual - actual.mean(axis=1, keepdims=True)
    centered_prediction = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_actual * centered_prediction, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(centered_actual), axis=1)
        * np.sum(np.square(centered_prediction), axis=1)
    )
    corr = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > eps,
    )
    rmse = np.sqrt(np.mean(np.square(actual - predicted), axis=1))
    actual_rms = np.sqrt(np.mean(np.square(actual), axis=1))
    predicted_rms = np.sqrt(np.mean(np.square(predicted), axis=1))
    nbm = float(smooth_l1_np(actual, predicted).mean())
    zero = float(smooth_l1_np(actual, np.zeros_like(actual)).mean())
    return {
        "nbm_huber": nbm,
        "zero_huber": zero,
        "improvement_pct": 100.0 * (zero - nbm) / max(zero, 1e-12),
        "median_corr": float(np.median(corr)),
        "p10_corr": float(np.percentile(corr, 10.0)),
        "worst_channel_corr": float(np.min(np.median(corr, axis=0))),
        "median_nrmse": float(np.median(rmse / (actual_rms + eps))),
        "median_amplitude_ratio": float(
            np.median(predicted_rms / (actual_rms + eps))
        ),
        "diff_huber": float(
            smooth_l1_np(np.diff(actual, axis=1), np.diff(predicted, axis=1)).mean()
        ),
    }


def pass_status(sample_count: int, metrics: dict[str, float]) -> str:
    threshold = THRESHOLDS[sample_count]
    checks = (
        metrics["improvement_pct"] >= threshold["improvement_pct"],
        metrics["median_corr"] >= threshold["median_corr"],
        metrics["median_nrmse"] <= threshold["median_nrmse"],
    )
    return "PASS" if all(checks) else "FAIL"


if __name__ == "__main__":
    raise SystemExit(
        "The completed experiment is preserved in outputs; use "
        "run_daphnet_nbm_tcdae_three_rounds.py for the follow-up diagnostic."
    )
