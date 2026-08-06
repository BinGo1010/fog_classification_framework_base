#!/usr/bin/env python
"""Evaluate saved NBM -> TCN-M models on train, validation, and test splits."""

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

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as core
from cnbr_fog.evaluation import binary_metrics


DEFAULT_SUITE = (
    REPO_ROOT
    / "outputs"
    / "nonfog_gru_nbm_tcnm_within_subject_v1_window_axis_centered_"
      "distributed_calibration_bottleneck32_finalnbm200_pat10_seed20260802"
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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confusion_metrics(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = (int(value) for value in np.asarray(cm).ravel())

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision_non_fog = div(tn, tn + fn)
    precision_fog = div(tp, tp + fp)
    recall_non_fog = div(tn, tn + fp)
    recall_fog = div(tp, tp + fn)
    f1_non_fog = div(2 * tn, 2 * tn + fp + fn)
    f1_fog = div(2 * tp, 2 * tp + fp + fn)
    return {
        "acc": div(tn + tp, tn + fp + fn + tp),
        "macro_precision": 0.5 * (precision_non_fog + precision_fog),
        "macro_recall": 0.5 * (recall_non_fog + recall_fog),
        "macro_f1": 0.5 * (f1_non_fog + f1_fog),
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
    subjects = sorted(
        path.name
        for path in suite.iterdir()
        if path.is_dir() and path.name in core.SUBJECT_SPLITS
    )
    if not subjects:
        raise FileNotFoundError(f"no subject directories found in {suite}")
    device = core.resolve_device(args.device)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    cms: dict[tuple[str, str], np.ndarray] = {}

    for subject in subjects:
        subject_dir = suite / subject
        with (subject_dir / "metrics.json").open(encoding="utf-8") as handle:
            saved_metrics = json.load(handle)
        with (subject_dir / "training.json").open(encoding="utf-8") as handle:
            training = json.load(handle)
        threshold = float(saved_metrics["selected_threshold"])
        classifier_training = training["tcn_m"]
        with np.load(subject_dir / "artifacts" / "residuals.npz") as arrays:
            split_arrays = {
                "train": (arrays["train_oof_residual"].copy(), arrays["train_y"].copy()),
                "validation": (arrays["validation_residual"].copy(), arrays["validation_y"].copy()),
                "test": (arrays["test_residual"].copy(), arrays["test_y"].copy()),
            }
        payload = torch.load(
            subject_dir / "checkpoints" / "tcn_m.pt",
            map_location=device,
            weights_only=False,
        )
        model = core.ResidualTCNM().to(device)
        model.load_state_dict(payload["model_state"])

        for split, (x, y) in split_arrays.items():
            y_true, y_prob = core.classifier_predict(model, x, y, device)
            metrics = core.add_requested_macro_metrics(
                binary_metrics(y_true, y_prob, threshold)
            )
            cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            y_pred = (y_prob >= threshold).astype(np.int8)
            cms[(subject, split)] = cm

            if split in ("validation", "test"):
                expected = saved_metrics[split]
                if cm.tolist() != expected["confusion_matrix"]:
                    raise AssertionError(
                        f"{subject} {split}: confusion matrix differs from saved result"
                    )
                if not np.isclose(
                    float(metrics["acc"]), float(expected["accuracy"]), atol=1e-12
                ):
                    raise AssertionError(f"{subject} {split}: ACC differs")

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
                    "tcn_best_epoch": int(classifier_training["best_epoch"]),
                    "tcn_epochs_completed": int(classifier_training["epochs_completed"]),
                }
            )
            for index, (truth, probability, prediction) in enumerate(
                zip(y_true, y_prob, y_pred)
            ):
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
            row = next(
                item
                for item in metric_rows
                if item["subject"] == subject and item["split"] == split
            )
            image = add_confusion_axis(
                axes[col],
                cms[(subject, split)],
                f"{split.capitalize()}\nACC {row['acc']:.3f} | Macro-F1 {row['macro_f1']:.3f}",
                show_ylabel=col == 0,
                show_xlabel=True,
            )
        fig.suptitle(f"{subject} NBM -> TCN-M (threshold={threshold:.2f})", fontsize=9)
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row proportion")
        save_figure(fig, subject_dir / "train_validation_test_confusion_matrices")

    write_csv(suite / "train_validation_test_metrics.csv", metric_rows)
    write_csv(suite / "train_validation_test_predictions.csv", prediction_rows)

    aggregate_rows: list[dict[str, Any]] = []
    pooled_cms: dict[str, np.ndarray] = {}
    for split in SPLITS:
        split_rows = [row for row in metric_rows if row["split"] == split]
        pooled_cm = sum(
            (cms[(subject, split)] for subject in subjects),
            np.zeros((2, 2), dtype=np.int64),
        )
        pooled_cms[split] = pooled_cm
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
                **confusion_metrics(pooled_cm),
            }
        )
        for statistic, function in (
            ("subject_equal_mean", np.mean),
            ("subject_equal_std", np.std),
        ):
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
            item
            for item in aggregate_rows
            if item["aggregation"] == "pooled_windows" and item["split"] == split
        )
        image = add_confusion_axis(
            axes[col],
            pooled_cms[split],
            f"{split.capitalize()}\nACC {row['acc']:.3f} | Macro-F1 {row['macro_f1']:.3f}",
            show_ylabel=col == 0,
            show_xlabel=True,
        )
    fig.suptitle("NBM -> TCN-M: pooled windows across 8 subject-specific models", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row proportion")
    save_figure(
        fig,
        suite / "pooled_train_validation_test_confusion_matrices",
        include_tiff=True,
    )

    fig, axes = plt.subplots(len(subjects), 3, figsize=(7.2, 15.0), constrained_layout=True)
    image = None
    for row_index, subject in enumerate(subjects):
        for col, split in enumerate(SPLITS):
            row = next(
                item
                for item in metric_rows
                if item["subject"] == subject and item["split"] == split
            )
            image = add_confusion_axis(
                axes[row_index, col],
                cms[(subject, split)],
                f"{subject} | {split}\nACC {row['acc']:.3f}, M-F1 {row['macro_f1']:.3f}",
                show_ylabel=col == 0,
                show_xlabel=row_index == len(subjects) - 1,
            )
    fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01, label="Row proportion")
    save_figure(
        fig,
        suite / "all_subjects_train_validation_test_confusion_matrices",
        include_tiff=True,
    )

    report = {
        "experiment": "NBM -> TCN-M within-subject",
        "subjects": subjects,
        "threshold_rule": (
            "subject-specific threshold selected on validation balanced accuracy; "
            "the identical frozen threshold is applied to train, validation, and test"
        ),
        "train_interpretation": (
            "resubstitution evaluation of the saved best TCN-M on the OOF NBM "
            "residuals used to fit the classifier"
        ),
        "confusion_matrix_order": [
            ["true_non_fog_pred_non_fog", "true_non_fog_pred_fog"],
            ["true_fog_pred_non_fog", "true_fog_pred_fog"],
        ],
        "macro_metrics": (
            "unweighted mean over non-FoG and FoG classes; zero_division=0"
        ),
        "figure_contract": {
            "core_conclusion": (
                "The frozen validation thresholds preserve high overall accuracy, "
                "but class-balanced generalization and subject heterogeneity are "
                "visible when moving from train/validation to test."
            ),
            "evidence_chain": (
                "Per-subject train/validation/test confusion matrices plus pooled "
                "window counts and subject-equal metric summaries."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "exports": ["SVG", "PDF", "PNG", "600 dpi TIFF for overview figures"],
        },
        "per_subject_split": metric_rows,
        "aggregate": aggregate_rows,
    }
    (suite / "train_validation_test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"suite": str(suite), "aggregate": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
