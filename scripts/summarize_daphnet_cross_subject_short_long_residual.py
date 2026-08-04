#!/usr/bin/env python
"""Audit and summarize the 2 s versus 2 s + 6 s residual LOSO suite."""

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
ARMS = ("short_2s", "short_2s_long_6s")
DISPLAY = {"short_2s": "Residual 2 s", "short_2s_long_6s": "Residual 2 s + 6 s"}
METRICS = ("accuracy", "fog_recall", "specificity", "pr_auc")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_arm(folder: Path, arm: str, subject: str, validation: str) -> dict[str, Any]:
    training = load_json(folder / "classifier_training.json")
    test = load_json(folder / "metrics.json")["test"]
    rows = list(csv.DictReader((folder / "test_predictions.csv").open("r", encoding="utf-8", newline="")))
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int8)
    probability = np.asarray([float(row["fog_probability"]) for row in rows])
    prediction = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int8)
    window_index = np.asarray([int(row["window_index"]) for row in rows], dtype=np.int64)
    matrix = np.asarray(
        [
            [np.sum((y == 0) & (prediction == 0)), np.sum((y == 0) & (prediction == 1))],
            [np.sum((y == 1) & (prediction == 0)), np.sum((y == 1) & (prediction == 1))],
        ], dtype=np.int64,
    )
    if not np.array_equal(matrix, np.asarray(test["confusion_matrix"], dtype=np.int64)):
        raise AssertionError(f"{subject}/{arm}: confusion matrix mismatch")
    if abs(float(average_precision_score(y, probability)) - float(test["pr_auc"])) > 1e-12:
        raise AssertionError(f"{subject}/{arm}: PR-AUC mismatch")
    return {
        "test_subject": subject,
        "validation_subject": validation,
        "arm": arm,
        "test_windows": int(len(y)),
        "test_non_fog_windows": int(np.sum(y == 0)),
        "test_fog_windows": int(np.sum(y == 1)),
        "parameter_count": int(training["architecture"]["parameter_count"]),
        "tcnm_best_epoch": int(training["best_epoch"]),
        "tcnm_epochs_completed": int(training["epochs_completed"]),
        "threshold": float(training["selected_threshold"]),
        **{metric: float(test[metric]) for metric in METRICS},
        "tn": int(test["tn"]), "fp": int(test["fp"]),
        "fn": int(test["fn"]), "tp": int(test["tp"]),
        "window_index": window_index,
        "y_true": y,
    }


def plot_confusions(root: Path, arm: str, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        maximum = max(int(matrix.max()), 1)
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_xticks([0, 1], labels=["non-FoG", "FoG"])
        axis.set_yticks([0, 1], labels=["non-FoG", "FoG"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{row['test_subject']} · val={row['validation_subject']} · T={row['threshold']:.2f}\n"
            f"Acc={row['accuracy']:.3f} · Recall={row['fog_recall']:.3f}\n"
            f"Specificity={row['specificity']:.3f} · AP={row['pr_auc']:.3f}",
            fontsize=10,
        )
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", fontsize=13,
                          color="white" if matrix[i,j] > maximum / 2 else "black")
    figure.suptitle(DISPLAY[arm] + " cross-subject test folds", fontsize=15)
    figure.savefig(root / f"confusion_matrices_{arm}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    all_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        fold = root / subject
        done = load_json(fold / "DONE.json")
        config = load_json(fold / "config.json")
        support = load_json(fold / "history_support.json")
        nbm = load_json(fold / "nbm_training.json")
        if done.get("status") != "complete" or config["split"]["test_subject"] != subject:
            raise ValueError(f"Incomplete or mismatched fold: {fold}")
        validation = config["split"]["validation_subject"]
        support_npz = np.load(fold / "history_support.npz")
        blocks = support_npz["test_block_window_index"]
        starts = support_npz["test_block_target_start"]
        ends = support_npz["test_block_target_end"]
        if blocks.shape[1] != 3 or np.any(np.diff(starts, axis=1) != 128):
            raise AssertionError(f"{subject}: not three 2-second blocks")
        if np.any(ends[:, :-1] != starts[:, 1:]) or np.any(ends[:, -1] - starts[:, 0] != 384):
            raise AssertionError(f"{subject}: history is not contiguous 6 seconds")
        arm_rows = [load_arm(fold / arm, arm, subject, validation) for arm in ARMS]
        if not np.array_equal(arm_rows[0]["window_index"], arm_rows[1]["window_index"]):
            raise AssertionError(f"{subject}: arm test anchors differ")
        if not np.array_equal(arm_rows[0]["y_true"], arm_rows[1]["y_true"]):
            raise AssertionError(f"{subject}: arm labels differ")
        for row in arm_rows:
            row["gru_best_epoch"] = int(nbm["best_epoch"])
            row["gru_epochs_completed"] = int(nbm["epochs_completed"])
            row["windows_before_history"] = int(support["test"]["windows_before_history"])
            row.pop("window_index")
            row.pop("y_true")
            all_rows.append(row)

    macro = {
        arm: {
            metric: float(np.mean([row[metric] for row in all_rows if row["arm"] == arm]))
            for metric in METRICS
        }
        for arm in ARMS
    }
    delta = {
        metric: macro["short_2s_long_6s"][metric] - macro["short_2s"][metric]
        for metric in METRICS
    }
    aggregate = {
        "subjects": list(SUBJECTS),
        "arms": list(ARMS),
        "subject_macro": macro,
        "short_long_minus_short": delta,
        "runs": all_rows,
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    by_key = {(row["arm"], row["test_subject"]): row for row in all_rows}
    for arm in ARMS:
        plot_confusions(root, arm, [by_key[(arm, subject)] for subject in SUBJECTS])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    positions = np.arange(len(SUBJECTS))
    width = 0.36
    titles = {"accuracy": "Accuracy", "fog_recall": "FoG Recall", "specificity": "Specificity", "pr_auc": "PR-AUC"}
    for axis, metric in zip(axes.flat, METRICS):
        short = [by_key[("short_2s", subject)][metric] for subject in SUBJECTS]
        dual = [by_key[("short_2s_long_6s", subject)][metric] for subject in SUBJECTS]
        axis.bar(positions - width / 2, short, width, label="Residual 2 s", color="#2166ac")
        axis.bar(positions + width / 2, dual, width, label="Residual 2 s + 6 s", color="#b2182b")
        axis.axhline(macro["short_2s"][metric], color="#2166ac", linestyle="--", linewidth=1)
        axis.axhline(macro["short_2s_long_6s"][metric], color="#b2182b", linestyle="--", linewidth=1)
        axis.set_xticks(positions, labels=SUBJECTS)
        axis.set_ylim(0, 1.03)
        axis.set_ylabel(titles[metric])
        axis.set_title(f"{titles[metric]} · macro delta = {delta[metric]:+.3f}")
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False, ncol=2)
    figure.suptitle("Cross-subject residual short/long comparison", fontsize=15)
    figure.savefig(root / "metric_comparison.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Daphnet GRU-NBM residual 2 s versus 2 s + 6 s LOSO",
        "",
        "- GRU context/forecast/stride = 2/2/1 seconds.",
        "- Long input = three contiguous non-overlapping 2-second residual blocks.",
        "- Both arms use the exact same complete-6-second anchors and labels.",
        "- The dual arm has a second TCN encoder and is not parameter matched.",
        "",
        "| Test | Val | Arm | Parameters | Test N/F | Threshold | Accuracy | FoG Recall | Specificity | PR-AUC | TN/FP/FN/TP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subject in SUBJECTS:
        for arm in ARMS:
            row = by_key[(arm, subject)]
            lines.append(
                f"| {subject} | {row['validation_subject']} | {DISPLAY[arm]} "
                f"| {row['parameter_count']:,} | {row['test_non_fog_windows']}/{row['test_fog_windows']} "
                f"| {row['threshold']:.2f} | {row['accuracy']:.4f} | {row['fog_recall']:.4f} "
                f"| {row['specificity']:.4f} | {row['pr_auc']:.4f} "
                f"| {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
            )
    lines.extend(["", "## Subject macro", ""])
    for arm in ARMS:
        values = macro[arm]
        lines.append(
            f"- {DISPLAY[arm]}: Accuracy {values['accuracy']:.6f}, FoG Recall {values['fog_recall']:.6f}, "
            f"Specificity {values['specificity']:.6f}, PR-AUC {values['pr_auc']:.6f}."
        )
    lines.extend(
        [
            "",
            f"Dual − short: Accuracy {delta['accuracy']:+.6f}, FoG Recall {delta['fog_recall']:+.6f}, "
            f"Specificity {delta['specificity']:+.6f}, PR-AUC {delta['pr_auc']:+.6f}。",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
