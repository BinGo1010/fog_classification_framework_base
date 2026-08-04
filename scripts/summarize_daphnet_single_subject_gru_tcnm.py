#!/usr/bin/env python
"""Audit and summarize the eight Daphnet within-subject GRU/TCN-M runs."""

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


SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    rows: list[dict[str, Any]] = []
    geometry: dict[str, Any] | None = None
    for subject in SUBJECTS:
        folder = root / subject
        done = load_json(folder / "DONE.json")
        config = load_json(folder / "config.json")
        nbm = load_json(folder / "nbm_training.json")
        classifier = load_json(folder / "classifier_training.json")
        test = load_json(folder / "metrics.json")["test"]
        if done.get("status") != "complete" or config.get("subject") != subject:
            raise ValueError(f"Incomplete or mismatched run: {folder}")
        current_geometry = config["windowing"]
        if geometry is None:
            geometry = current_geometry
        elif any(
            current_geometry[key] != geometry[key]
            for key in ("context_samples", "target_samples", "stride_samples")
        ):
            raise ValueError("Window geometry differs across subjects")
        matrix = np.zeros((2, 2), dtype=np.int64)
        with (folder / "test_predictions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            prediction_rows = list(csv.DictReader(handle))
        for prediction in prediction_rows:
            matrix[int(prediction["y_true"]), int(prediction["y_pred"])] += 1
        stored = np.asarray(test["confusion_matrix"], dtype=np.int64)
        if not np.array_equal(matrix, stored):
            raise AssertionError(f"Confusion matrix mismatch for {subject}")
        rows.append(
            {
                "subject": subject,
                "test_record": config["split"]["test"].split(";")[0].removeprefix("all "),
                "ignored_post_test_records": ",".join(
                    config["split"].get("ignored_post_test_records", [])
                ),
                "train_windows": config["window_statistics"]["train"]["windows"],
                "validation_windows": config["window_statistics"]["validation"]["windows"],
                "test_windows": test["n"],
                "test_non_fog_windows": test["n_normal"],
                "test_fog_windows": test["n_fog"],
                "gru_max_epochs": nbm["maximum_epochs"],
                "gru_patience": nbm["patience"],
                "gru_epochs_completed": nbm["epochs_completed"],
                "gru_best_epoch": nbm["best_epoch"],
                "gru_best_validation_nll": nbm["best_validation_gaussian_nll"],
                "tcnm_epochs_completed": classifier["epochs_completed"],
                "tcnm_best_epoch": classifier["best_epoch"],
                "validation_threshold": classifier["selected_threshold"],
                "accuracy": test["accuracy"],
                "fog_recall": test["fog_recall"],
                "specificity": test["specificity"],
                "pr_auc": test["pr_auc"],
                "tn": test["tn"],
                "fp": test["fp"],
                "fn": test["fn"],
                "tp": test["tp"],
            }
        )

    metric_names = ("accuracy", "fog_recall", "specificity", "pr_auc")
    aggregate = {
        "subjects": list(SUBJECTS),
        "windowing": geometry,
        "subject_macro": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in metric_names
        },
        "runs": rows,
    }
    (root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Daphnet 单被试 GRU-NBM + TCN-M 汇总",
        "",
        "- Subjects: S01, S02, S03, S05, S06, S07, S08, S09",
        "- GRU: maximum epochs 50, patience 6",
        "- TCN-M: maximum epochs 12, patience 4",
        (
            f"- Context/target/stride: {geometry['context_seconds']:g}/"
            f"{geometry['target_seconds']:g}/{geometry['stride_seconds']:g} seconds"
            if geometry is not None
            else ""
        ),
        "",
        "| Subject | Test N/F | GRU best/completed | TCN best/completed | Threshold | Accuracy | FoG Recall | Specificity | PR-AUC | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subject']} | {row['test_non_fog_windows']}/{row['test_fog_windows']} "
            f"| {row['gru_best_epoch']}/{row['gru_epochs_completed']} "
            f"| {row['tcnm_best_epoch']}/{row['tcnm_epochs_completed']} "
            f"| {row['validation_threshold']:.2f} | {row['accuracy']:.4f} "
            f"| {row['fog_recall']:.4f} | {row['specificity']:.4f} "
            f"| {row['pr_auc']:.4f} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    macro = aggregate["subject_macro"]
    sparse_fog = [
        f"{row['subject']} ({row['test_fog_windows']})"
        for row in rows
        if int(row["test_fog_windows"]) <= 10
    ]
    lines.extend(
        [
            "",
            "## Subject-macro",
            "",
            f"- Accuracy: {macro['accuracy']:.6f}",
            f"- FoG Recall: {macro['fog_recall']:.6f}",
            f"- Specificity: {macro['specificity']:.6f}",
            f"- PR-AUC: {macro['pr_auc']:.6f}",
            "",
            (
                "Test sets with at most 10 FoG windows: "
                + ", ".join(sparse_fog)
                + ". Their recall and PR-AUC have high sampling uncertainty."
                if sparse_fog
                else ""
            ),
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.5), constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray(
            [[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.int64
        )
        local_maximum = int(matrix.max())
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(local_maximum, 1))
        axis.set_xticks([0, 1], labels=["non-FoG", "FoG"])
        axis.set_yticks([0, 1], labels=["non-FoG", "FoG"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{row['subject']} · threshold={row['validation_threshold']:.2f}\n"
            f"Acc={row['accuracy']:.3f} · Recall={row['fog_recall']:.3f}\n"
            f"Specificity={row['specificity']:.3f} · AP={row['pr_auc']:.3f}",
            fontsize=10.5,
        )
        cutoff = float(local_maximum) / 2.0
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="white" if matrix[i, j] > cutoff else "black",
                )
    figure.savefig(root / "confusion_matrices.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
