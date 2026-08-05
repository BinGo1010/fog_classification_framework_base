#!/usr/bin/env python
"""Evaluate saved raw+InceptionTime models on train, validation, and test splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as core
from cnbr_fog.evaluation import binary_metrics


DEFAULT_SUITE = (
    REPO_ROOT
    / "outputs"
    / "raw_inceptiontime_within_subject_distributed_scaler_protocol_seed20260802"
)
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confusion_metrics(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = (int(value) for value in np.asarray(cm).ravel())

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    p0 = div(tn, tn + fn)
    p1 = div(tp, tp + fp)
    r0 = div(tn, tn + fp)
    r1 = div(tp, tp + fn)
    f0 = div(2 * tn, 2 * tn + fp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {
        "acc": div(tn + tp, tn + fp + fn + tp),
        "macro_precision": 0.5 * (p0 + p1),
        "macro_recall": 0.5 * (r0 + r1),
        "macro_f1": 0.5 * (f0 + f1),
    }


def add_confusion_axis(
    ax: plt.Axes,
    cm: np.ndarray,
    title: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
) -> Any:
    cm = np.asarray(cm, dtype=np.int64)
    row_total = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm,
        row_total,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_total != 0,
    )
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.tick_params(length=0, labelsize=6)
    if show_xlabel:
        ax.set_xlabel("Predicted", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("True", fontsize=7)
    for i in range(2):
        for j in range(2):
            color = "white" if normalized[i, j] >= 0.55 else "#272727"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=6,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_figure(fig: plt.Figure, base: Path, *, include_tiff: bool = False) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if include_tiff:
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite = args.suite.resolve()
    subjects = sorted(path.name for path in suite.iterdir() if path.is_dir() and path.name.startswith("S"))
    if not subjects:
        raise FileNotFoundError(f"no subject directories found in {suite}")
    device = core.resolve_device(args.device)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    cms: dict[tuple[str, str], np.ndarray] = {}
    probabilities_by_split: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    truths_by_split: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    predictions_by_split: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}

    for subject in subjects:
        subject_dir = suite / subject
        with (subject_dir / "metrics.json").open(encoding="utf-8") as handle:
            saved_metrics = json.load(handle)
        with (subject_dir / "training.json").open(encoding="utf-8") as handle:
            training = json.load(handle)
        threshold = float(saved_metrics["selected_threshold"])
        features = np.load(subject_dir / "artifacts" / "features.npz")
        payload = torch.load(
            subject_dir / "checkpoints" / "inception_time.pt",
            map_location=device,
            weights_only=False,
        )
        model = core.InceptionTimeClassifier(in_channels=9).to(device)
        model.load_state_dict(payload["model_state"])

        split_arrays = {
            "train": (features["train_oof_raw_centered"], features["train_y"]),
            "validation": (features["validation_raw_centered"], features["validation_y"]),
            "test": (features["test_raw_centered"], features["test_y"]),
        }
        for split, (x, y) in split_arrays.items():
            y_true, y_prob = core.classifier_predict(model, x, y, device)
            metrics = core.add_requested_macro_metrics(binary_metrics(y_true, y_prob, threshold))
            cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            y_pred = (y_prob >= threshold).astype(np.int8)
            cms[(subject, split)] = cm
            probabilities_by_split[split].append(y_prob)
            truths_by_split[split].append(y_true)
            predictions_by_split[split].append(y_pred)

            if split in ("validation", "test"):
                expected = saved_metrics[split]
                if cm.tolist() != expected["confusion_matrix"]:
                    raise AssertionError(f"{subject} {split}: confusion matrix differs from saved result")
                for key in ("acc", "macro_precision", "macro_recall", "macro_f1"):
                    if not np.isclose(float(metrics[key]), float(expected[key]), atol=1e-12):
                        raise AssertionError(f"{subject} {split}: {key} differs from saved result")

            metric_rows.append(
                {
                    "subject": subject,
                    "split": split,
                    "threshold": threshold,
                    "n": int(metrics["n"]),
                    "n_non_fog": int(metrics["n_normal"]),
                    "n_fog": int(metrics["n_fog"]),
                    "tn": int(metrics["tn"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "tp": int(metrics["tp"]),
                    "acc": float(metrics["acc"]),
                    "macro_precision": float(metrics["macro_precision"]),
                    "macro_recall": float(metrics["macro_recall"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "best_epoch": int(training["best_epoch"]),
                    "epochs_completed": int(training["epochs_completed"]),
                }
            )
            for index, (truth, probability, prediction) in enumerate(zip(y_true, y_prob, y_pred)):
                prediction_rows.append(
                    {
                        "subject": subject,
                        "split": split,
                        "split_row": index,
                        "y_true": int(truth),
                        "fog_probability": float(probability),
                        "threshold": threshold,
                        "y_pred": int(prediction),
                    }
                )

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
        image = None
        for col, split in enumerate(SPLITS):
            row = next(r for r in metric_rows if r["subject"] == subject and r["split"] == split)
            image = add_confusion_axis(
                axes[col],
                cms[(subject, split)],
                f"{split.capitalize()}\nACC {row['acc']:.3f} | Macro-F1 {row['macro_f1']:.3f}",
                show_ylabel=col == 0,
                show_xlabel=True,
            )
        fig.suptitle(f"{subject} raw + InceptionTime (threshold={threshold:.2f})", fontsize=9)
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row proportion")
        save_figure(fig, subject_dir / "train_validation_test_confusion_matrices")

    write_csv(suite / "train_validation_test_metrics.csv", metric_rows)
    write_csv(suite / "train_validation_test_predictions.csv", prediction_rows)

    aggregate_rows: list[dict[str, Any]] = []
    pooled_cms: dict[str, np.ndarray] = {}
    for split in SPLITS:
        split_rows = [row for row in metric_rows if row["split"] == split]
        pooled_cm = sum((cms[(subject, split)] for subject in subjects), np.zeros((2, 2), dtype=np.int64))
        pooled_cms[split] = pooled_cm
        pooled = confusion_metrics(pooled_cm)
        aggregate_rows.append(
            {
                "aggregation": "pooled_windows",
                "split": split,
                "subjects": len(subjects),
                "n": int(pooled_cm.sum()),
                "tn": int(pooled_cm[0, 0]),
                "fp": int(pooled_cm[0, 1]),
                "fn": int(pooled_cm[1, 0]),
                "tp": int(pooled_cm[1, 1]),
                **pooled,
            }
        )
        for statistic, function in (("subject_equal_mean", np.mean), ("subject_equal_std", np.std)):
            aggregate_rows.append(
                {
                    "aggregation": statistic,
                    "split": split,
                    "subjects": len(subjects),
                    "n": "",
                    "tn": "",
                    "fp": "",
                    "fn": "",
                    "tp": "",
                    **{
                        key: float(function([float(row[key]) for row in split_rows]))
                        for key in ("acc", "macro_precision", "macro_recall", "macro_f1")
                    },
                }
            )
    write_csv(suite / "train_validation_test_aggregate_metrics.csv", aggregate_rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    image = None
    for col, split in enumerate(SPLITS):
        row = next(
            r for r in aggregate_rows if r["aggregation"] == "pooled_windows" and r["split"] == split
        )
        image = add_confusion_axis(
            axes[col],
            pooled_cms[split],
            f"{split.capitalize()}\nACC {row['acc']:.3f} | Macro-F1 {row['macro_f1']:.3f}",
            show_ylabel=col == 0,
            show_xlabel=True,
        )
    fig.suptitle("Raw + InceptionTime: pooled windows across 8 subject-specific models", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row proportion")
    save_figure(fig, suite / "pooled_train_validation_test_confusion_matrices", include_tiff=True)

    fig, axes = plt.subplots(len(subjects), 3, figsize=(7.2, 15.0), constrained_layout=True)
    image = None
    for row_index, subject in enumerate(subjects):
        for col, split in enumerate(SPLITS):
            row = next(r for r in metric_rows if r["subject"] == subject and r["split"] == split)
            title = f"{subject} | {split}\nACC {row['acc']:.3f}, M-F1 {row['macro_f1']:.3f}"
            image = add_confusion_axis(
                axes[row_index, col],
                cms[(subject, split)],
                title,
                show_ylabel=col == 0,
                show_xlabel=row_index == len(subjects) - 1,
            )
    fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01, label="Row proportion")
    save_figure(fig, suite / "all_subjects_train_validation_test_confusion_matrices")

    payload = {
        "experiment": "raw + InceptionTime within-subject ablation",
        "subjects": subjects,
        "threshold_rule": "subject-specific threshold selected on validation balanced accuracy; same threshold applied to train, validation, and test",
        "train_interpretation": "resubstitution evaluation of the saved best model on the OOF-scaled features used for classifier fitting",
        "confusion_matrix_order": [["true_non_fog_pred_non_fog", "true_non_fog_pred_fog"], ["true_fog_pred_non_fog", "true_fog_pred_fog"]],
        "macro_metrics": "unweighted mean over non-FoG and FoG classes; zero_division=0",
        "per_subject_split": metric_rows,
        "aggregate": aggregate_rows,
    }
    (suite / "train_validation_test_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"suite": str(suite), "subjects": subjects, "aggregate": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
