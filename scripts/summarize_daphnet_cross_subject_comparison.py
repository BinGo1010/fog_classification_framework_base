#!/usr/bin/env python
"""Audit and compare the two eight-fold cross-subject TCN-M experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score


SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
ARCHITECTURES = ("gru_residual", "raw_target")
DISPLAY = {"gru_residual": "GRU + residual", "raw_target": "Raw target"}
METRICS = ("accuracy", "fog_recall", "specificity", "pr_auc")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold(folder: Path, architecture: str, subject: str) -> dict[str, Any]:
    done = load_json(folder / "DONE.json")
    config = load_json(folder / "config.json")
    classifier = load_json(folder / "classifier_training.json")
    test = load_json(folder / "metrics.json")["test"]
    if done.get("status") != "complete":
        raise ValueError(f"Incomplete run: {folder}")
    if config.get("architecture_variant") != architecture:
        raise ValueError(f"Architecture mismatch: {folder}")
    if config["split"]["test_subject"] != subject:
        raise ValueError(f"Test-subject mismatch: {folder}")
    with (folder / "test_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    y = np.asarray([int(row["y_true"]) for row in predictions], dtype=np.int8)
    probability = np.asarray([float(row["fog_probability"]) for row in predictions])
    prediction = np.asarray([int(row["y_pred"]) for row in predictions], dtype=np.int8)
    matrix = np.asarray(
        [
            [np.sum((y == 0) & (prediction == 0)), np.sum((y == 0) & (prediction == 1))],
            [np.sum((y == 1) & (prediction == 0)), np.sum((y == 1) & (prediction == 1))],
        ],
        dtype=np.int64,
    )
    if not np.array_equal(matrix, np.asarray(test["confusion_matrix"], dtype=np.int64)):
        raise AssertionError(f"Confusion matrix mismatch: {folder}")
    if abs(float(average_precision_score(y, probability)) - float(test["pr_auc"])) > 1e-12:
        raise AssertionError(f"PR-AUC mismatch: {folder}")
    nbm_path = folder / "nbm_training.json"
    nbm = load_json(nbm_path) if nbm_path.exists() else None
    return {
        "architecture": architecture,
        "test_subject": subject,
        "validation_subject": config["split"]["validation_subject"],
        "train_subjects": ",".join(config["split"]["train_subjects"]),
        "test_windows": int(test["n"]),
        "test_non_fog_windows": int(test["n_normal"]),
        "test_fog_windows": int(test["n_fog"]),
        "gru_best_epoch": nbm["best_epoch"] if nbm is not None else None,
        "gru_epochs_completed": nbm["epochs_completed"] if nbm is not None else None,
        "tcnm_best_epoch": classifier["best_epoch"],
        "tcnm_epochs_completed": classifier["epochs_completed"],
        "threshold": classifier["selected_threshold"],
        **{metric: test[metric] for metric in METRICS},
        "tn": int(test["tn"]),
        "fp": int(test["fp"]),
        "fn": int(test["fn"]),
        "tp": int(test["tp"]),
    }


def plot_montage(root: Path, architecture: str, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.int64)
        maximum = max(int(matrix.max()), 1)
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_xticks([0, 1], labels=["non-FoG", "FoG"])
        axis.set_yticks([0, 1], labels=["non-FoG", "FoG"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{row['test_subject']} · val={row['validation_subject']} · threshold={row['threshold']:.2f}\n"
            f"Acc={row['accuracy']:.3f} · Recall={row['fog_recall']:.3f}\n"
            f"Specificity={row['specificity']:.3f} · AP={row['pr_auc']:.3f}",
            fontsize=10,
        )
        cutoff = maximum / 2
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", fontsize=13,
                          color="white" if matrix[i,j] > cutoff else "black")
    figure.suptitle(DISPLAY[architecture] + " cross-subject test folds", fontsize=15)
    figure.savefig(root / f"confusion_matrices_{architecture}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    all_rows: list[dict[str, Any]] = []
    split_reference: dict[str, dict[str, np.ndarray]] = {}
    for architecture in ARCHITECTURES:
        architecture_rows = []
        for subject in SUBJECTS:
            folder = root / architecture / subject
            row = load_fold(folder, architecture, subject)
            architecture_rows.append(row)
            indices = np.load(folder / "split_indices.npz")
            current = {key: indices[key] for key in indices.files}
            if subject not in split_reference:
                split_reference[subject] = current
            elif any(
                not np.array_equal(current[key], split_reference[subject][key])
                for key in current
            ):
                raise AssertionError(f"Model split mismatch for {subject}")
        all_rows.extend(architecture_rows)
        plot_montage(root, architecture, architecture_rows)

    macro = {
        architecture: {
            metric: float(np.mean([row[metric] for row in all_rows if row["architecture"] == architecture]))
            for metric in METRICS
        }
        for architecture in ARCHITECTURES
    }
    deltas = {
        metric: macro["raw_target"][metric] - macro["gru_residual"][metric]
        for metric in METRICS
    }
    by_key = {(row["architecture"], row["test_subject"]): row for row in all_rows}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    positions = np.arange(len(SUBJECTS), dtype=np.float64)
    width = 0.36
    metric_titles = {
        "accuracy": "Accuracy",
        "fog_recall": "FoG Recall",
        "specificity": "Specificity",
        "pr_auc": "PR-AUC",
    }
    for axis, metric in zip(axes.flat, METRICS):
        gru_values = [by_key[("gru_residual", subject)][metric] for subject in SUBJECTS]
        raw_values = [by_key[("raw_target", subject)][metric] for subject in SUBJECTS]
        axis.bar(positions - width / 2, gru_values, width, label="GRU + residual", color="#2166ac")
        axis.bar(positions + width / 2, raw_values, width, label="Raw target", color="#b2182b")
        axis.axhline(macro["gru_residual"][metric], color="#2166ac", linestyle="--", linewidth=1)
        axis.axhline(macro["raw_target"][metric], color="#b2182b", linestyle="--", linewidth=1)
        axis.set_xticks(positions, labels=SUBJECTS)
        axis.set_ylim(0, 1.03)
        axis.set_ylabel(metric_titles[metric])
        axis.set_title(
            f"{metric_titles[metric]} · macro delta raw−GRU = {deltas[metric]:+.3f}"
        )
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False, ncol=2)
    figure.suptitle("Cross-subject comparison by held-out test subject", fontsize=15)
    figure.savefig(root / "metric_comparison.png", dpi=180)
    plt.close(figure)
    aggregate = {
        "subjects": list(SUBJECTS),
        "fold_protocol": "outer test subject plus cyclic subject-disjoint validation",
        "subject_macro": macro,
        "raw_target_minus_gru_residual_macro": deltas,
        "runs": all_rows,
    }
    (root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [
        "# Daphnet 两种模型跨被试实验",
        "",
        "- 每折：1 个测试被试、循环中的下一个 FoG 被试作验证，其余 8 人训练。",
        "- 训练、验证、测试被试完全不重叠；S04 和 S10 只作为训练被试。",
        "- Context/target/stride = 2/1/0.5 秒；seed = 42。",
        "- GRU 最大 epoch 50、patience 6；TCN-M 最大 epoch 12、patience 4。",
        "",
        "| Test | Val | 模型 | Accuracy | FoG Recall | Specificity | PR-AUC | TN/FP/FN/TP |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for subject in SUBJECTS:
        for architecture in ARCHITECTURES:
            row = by_key[(architecture, subject)]
            lines.append(
                f"| {subject} | {row['validation_subject']} | {DISPLAY[architecture]} "
                f"| {row['accuracy']:.4f} | {row['fog_recall']:.4f} "
                f"| {row['specificity']:.4f} | {row['pr_auc']:.4f} "
                f"| {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
            )
    lines.extend(["", "## 被试宏平均", ""])
    for architecture in ARCHITECTURES:
        values = macro[architecture]
        lines.append(
            f"- {DISPLAY[architecture]}: Accuracy {values['accuracy']:.6f}, "
            f"FoG Recall {values['fog_recall']:.6f}, Specificity {values['specificity']:.6f}, "
            f"PR-AUC {values['pr_auc']:.6f}."
        )
    lines.extend(
        [
            "",
            "Raw target − GRU + residual："
            f"Accuracy {deltas['accuracy']:+.6f}, FoG Recall {deltas['fog_recall']:+.6f}, "
            f"Specificity {deltas['specificity']:+.6f}, PR-AUC {deltas['pr_auc']:+.6f}。",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
