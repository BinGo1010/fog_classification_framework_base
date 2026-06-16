from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, log_loss
)
from sklearn.preprocessing import label_binarize


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def top_k_accuracy(y_true, y_prob, k=1):
    y_true = np.asarray(y_true)
    top = np.argsort(y_prob, axis=1)[:, -k:]
    return float(np.mean([yt in top_i for yt, top_i in zip(y_true, top)]))


def compute_metrics(y_true, y_prob, num_classes: int, top_k=(1,), loss=None) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(num_classes))
    out: Dict[str, Any] = {}
    if loss is not None:
        out["loss"] = float(loss)
    else:
        out["loss"] = _safe(lambda: float(log_loss(y_true, y_prob, labels=labels)))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    for avg in ["macro", "micro", "weighted"]:
        out[f"precision_{avg}"] = float(precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0))
        out[f"recall_{avg}"] = float(recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0))
        out[f"f1_{avg}"] = float(f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0))
    out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    for k in top_k:
        if k <= num_classes:
            out[f"top_{k}_accuracy"] = top_k_accuracy(y_true, y_prob, k)
    if num_classes == 2:
        if len(np.unique(y_true)) < 2:
            out["roc_auc"] = None
            out["pr_auc"] = None
        else:
            out["roc_auc"] = _safe(lambda: float(roc_auc_score(y_true, y_prob[:, 1])))
            out["pr_auc"] = _safe(lambda: float(average_precision_score(y_true, y_prob[:, 1])))
    else:
        y_bin = label_binarize(y_true, classes=labels)
        out["roc_auc_ovr_macro"] = _safe(lambda: float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")))
        out["roc_auc_ovr_weighted"] = _safe(lambda: float(roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")))
        out["pr_auc_macro"] = _safe(lambda: float(average_precision_score(y_bin, y_prob, average="macro")))
        out["pr_auc_weighted"] = _safe(lambda: float(average_precision_score(y_bin, y_prob, average="weighted")))
    return out


def confusion_and_per_class(y_true, y_prob, num_classes: int):
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    total = cm.sum()
    for c in labels:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        specificity = tn / (tn + fp + 1e-12)
        rows.append({
            "class": c,
            "precision": float(precision[c]),
            "recall_sensitivity": float(recall[c]),
            "specificity": float(specificity),
            "f1": float(f1[c]),
            "support": int(support[c]),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    return cm, pd.DataFrame(rows)


def save_metric_artifacts(out_dir, split, y_true, y_prob, indices, num_classes, metrics, options=None):
    options = options or {}
    save_predictions = bool(options.get("save_predictions", True))
    save_confusion_matrix = bool(options.get("save_confusion_matrix", True))
    save_per_class = bool(options.get("save_per_class", True))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)
    pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_csv(out_dir / f"metrics_{split}.csv", index=False)
    if save_confusion_matrix or save_per_class:
        cm, per_class = confusion_and_per_class(y_true, y_prob, num_classes)
        if save_confusion_matrix:
            pd.DataFrame(cm).to_csv(out_dir / f"confusion_matrix_{split}.csv", index=True)
        if save_per_class:
            per_class.to_csv(out_dir / f"per_class_metrics_{split}.csv", index=False)
    if save_predictions:
        pred_df = pd.DataFrame({"index": indices, "y_true": y_true, "y_pred": y_pred})
        for c in range(num_classes):
            pred_df[f"prob_class_{c}"] = y_prob[:, c]
        pred_df.to_csv(out_dir / f"predictions_{split}.csv", index=False)
