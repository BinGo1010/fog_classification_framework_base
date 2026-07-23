"""Metrics, threshold selection, and result serialization helpers."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(function, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        return float(function(y_true, y_prob))
    except ValueError:
        return None


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= float(threshold)).astype(np.int8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    specificity = tn / (tn + fp) if tn + fp else None
    sensitivity = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "threshold": float(threshold),
            "n": int(y_true.size),
            "n_normal": int((y_true == 0).sum()),
            "n_fog": int((y_true == 1).sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_true, y_pred))
                if np.unique(y_true).size == 2
                else None
            ),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": (
                float(f1_score(y_true, y_pred, zero_division=0))
                if np.any(y_true == 1)
                else None
            ),
            "mcc": (
                float(matthews_corrcoef(y_true, y_pred))
                if np.unique(y_true).size == 2 and np.unique(y_pred).size == 2
                else None
            ),
            "auroc": _safe_auc(roc_auc_score, y_true, y_prob),
            "auprc": _safe_auc(average_precision_score, y_true, y_prob),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "confusion_matrix": cm.tolist(),
        }
    return metrics


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    """Select a validation-only threshold by balanced accuracy.

    FOG prevalence varies strongly by subject, so sensitivity/specificity balance
    is the primary objective.  F1 and then a higher threshold break ties.
    """

    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if np.unique(y_true).size < 2:
        return 0.5, binary_metrics(y_true, y_prob, 0.5)
    candidates = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_prob, best_threshold)
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        key = (
            float(metrics["balanced_accuracy"] or 0.0),
            float(metrics["f1"] or 0.0),
            float(threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def aggregate_fold_metrics(rows: list[dict], metric_keys: list[str]) -> dict:
    aggregate: dict[str, dict] = {}
    for key in metric_keys:
        values = [row.get(key) for row in rows]
        values = [float(value) for value in values if value is not None and math.isfinite(value)]
        if not values:
            aggregate[key] = {"mean": None, "std": None, "n_folds": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        aggregate[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "min": float(array.min()),
            "max": float(array.max()),
            "n_folds": int(array.size),
        }
    return aggregate


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
