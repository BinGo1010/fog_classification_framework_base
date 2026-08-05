from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_b0_inceptiontime_confusion_matrices import (  # noqa: E402
    metrics_from_counts,
    restore_counts,
    seed_median_fold_counts,
)


def test_metrics_from_counts_binary_macro_metrics() -> None:
    metrics = metrics_from_counts(tn=80, fp=20, fn=10, tp=40)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["macro_recall"] == pytest.approx((0.8 + 0.8) / 2)
    assert metrics["macro_precision"] == pytest.approx((80 / 90 + 40 / 60) / 2)


def test_restore_counts_from_saved_metrics() -> None:
    row = pd.Series(
        {
            "n_windows": 150,
            "positive_windows": 50,
            "recall": 0.8,
            "specificity": 0.8,
        }
    )
    assert restore_counts(row) == {"tn": 80, "fp": 20, "fn": 10, "tp": 40}


def test_seed_median_is_one_typical_seed_not_seed_sum() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["S01"] * 3,
            "fold_id": ["f0"] * 3,
            "split": ["test"] * 3,
            "seed": [1, 2, 3],
            "tn": [80, 82, 78],
            "fp": [20, 18, 22],
            "fn": [10, 12, 8],
            "tp": [40, 38, 42],
        }
    )
    result = seed_median_fold_counts(frame).iloc[0]
    assert (result[["tn", "fp", "fn", "tp"]].to_numpy() == [80, 20, 10, 40]).all()
    assert result["n_evaluations"] == 150

