#!/usr/bin/env python
"""Audit and summarize the eight within-subject raw-target TCN-M runs."""

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
METRICS = ("accuracy", "fog_recall", "specificity", "pr_auc")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    baseline_by_subject: dict[str, dict[str, Any]] = {}
    if args.baseline_dir is not None:
        baseline = load_json(args.baseline_dir.resolve() / "aggregate.json")
        baseline_by_subject = {row["subject"]: row for row in baseline["runs"]}

    rows: list[dict[str, Any]] = []
    geometry: dict[str, Any] | None = None
    for subject in SUBJECTS:
        folder = root / subject
        done = load_json(folder / "DONE.json")
        config = load_json(folder / "config.json")
        training = load_json(folder / "classifier_training.json")
        test = load_json(folder / "metrics.json")["test"]
        if done.get("status") != "complete" or config.get("subject") != subject:
            raise ValueError(f"Incomplete or mismatched run: {folder}")
        current = config["windowing"]
        if geometry is None:
            geometry = current
        elif any(
            current[key] != geometry[key]
            for key in ("context_samples", "target_samples", "stride_samples")
        ):
            raise ValueError("Window geometry differs across subjects")
        matrix = np.zeros((2, 2), dtype=np.int64)
        with (folder / "test_predictions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            predictions = list(csv.DictReader(handle))
        for prediction in predictions:
            matrix[int(prediction["y_true"]), int(prediction["y_pred"])] += 1
        if not np.array_equal(matrix, np.asarray(test["confusion_matrix"], dtype=np.int64)):
            raise AssertionError(f"Confusion matrix mismatch for {subject}")
        row = {
            "subject": subject,
            "test_record": config["split"]["test"].split(";")[0].removeprefix("all "),
            "train_windows": config["window_statistics"]["train"]["windows"],
            "validation_windows": config["window_statistics"]["validation"]["windows"],
            "test_windows": test["n"],
            "test_non_fog_windows": test["n_normal"],
            "test_fog_windows": test["n_fog"],
            "tcnm_epochs_completed": training["epochs_completed"],
            "tcnm_best_epoch": training["best_epoch"],
            "validation_threshold": training["selected_threshold"],
            **{metric: test[metric] for metric in METRICS},
            "tn": test["tn"],
            "fp": test["fp"],
            "fn": test["fn"],
            "tp": test["tp"],
        }
        if subject in baseline_by_subject:
            for metric in METRICS:
                row[f"delta_vs_gru_residual_{metric}"] = (
                    float(row[metric]) - float(baseline_by_subject[subject][metric])
                )
        rows.append(row)

    macro = {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }
    macro_delta = None
    if baseline_by_subject:
        macro_delta = {
            metric: macro[metric]
            - float(np.mean([float(baseline_by_subject[s][metric]) for s in SUBJECTS]))
            for metric in METRICS
        }
    aggregate = {
        "subjects": list(SUBJECTS),
        "ablation": "Robust-scaled raw target -> TCN-M; GRU and residual removed",
        "windowing": geometry,
        "subject_macro": macro,
        "subject_macro_delta_vs_gru_residual": macro_delta,
        "runs": rows,
    }
    (root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert geometry is not None
    lines = [
        "# Daphnet 单被试 raw-target TCN-M 消融汇总",
        "",
        "- 移除：GRU 正常行为预测器、标准化预测残差。",
        "- 输入：训练集 Robust Scaler 标准化后的 1 秒 target 原始信号，9 通道。",
        "- 2 秒 context 只用于保持窗口端点和数据切分一致，不输入 TCN-M。",
        f"- Context/target/stride: {geometry['context_seconds']:g}/{geometry['target_seconds']:g}/{geometry['stride_seconds']:g} 秒。",
        "- TCN-M 最大 epoch 12，patience 4；验证集选 epoch 和分类阈值。",
        "",
        "| 被试 | Test N/F | TCN best/completed | 阈值 | Accuracy | FoG Recall | Specificity | PR-AUC | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subject']} | {row['test_non_fog_windows']}/{row['test_fog_windows']} "
            f"| {row['tcnm_best_epoch']}/{row['tcnm_epochs_completed']} "
            f"| {row['validation_threshold']:.2f} | {row['accuracy']:.4f} "
            f"| {row['fog_recall']:.4f} | {row['specificity']:.4f} "
            f"| {row['pr_auc']:.4f} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    lines.extend(
        [
            "",
            "## 被试宏平均",
            "",
            f"- Accuracy: {macro['accuracy']:.6f}",
            f"- FoG Recall: {macro['fog_recall']:.6f}",
            f"- Specificity: {macro['specificity']:.6f}",
            f"- PR-AUC: {macro['pr_auc']:.6f}",
        ]
    )
    if macro_delta is not None:
        lines.extend(
            [
                "",
                "## 相对 GRU + residual 基线的宏平均差值",
                "",
                f"- Accuracy: {macro_delta['accuracy']:+.6f}",
                f"- FoG Recall: {macro_delta['fog_recall']:+.6f}",
                f"- Specificity: {macro_delta['specificity']:+.6f}",
                f"- PR-AUC: {macro_delta['pr_auc']:+.6f}",
            ]
        )
    sparse = [f"{row['subject']} ({row['test_fog_windows']})" for row in rows if row["test_fog_windows"] <= 10]
    if sparse:
        lines.extend(
            [
                "",
                "注意：测试 FoG 窗口不超过 10 个的被试为 " + ", ".join(sparse) + "，其 Recall 与 PR-AUC 方差很高。",
            ]
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.5), constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.int64)
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
        cutoff = local_maximum / 2.0
        for i in range(2):
            for j in range(2):
                axis.text(
                    j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if matrix[i, j] > cutoff else "black",
                )
    figure.savefig(root / "confusion_matrices.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
