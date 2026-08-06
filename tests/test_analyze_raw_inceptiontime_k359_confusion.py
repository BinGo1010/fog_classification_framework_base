from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_raw_inceptiontime_k359_confusion_matrices import confusion_counts  # noqa: E402


def test_confusion_counts_layout() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    assert confusion_counts(y_true, y_pred) == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}

