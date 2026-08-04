#!/usr/bin/env python
"""S01 ablation: window-axis-centered raw 2 s IMU directly into the same TCN-M."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    FS,
    SUBJECT,
    ResidualTCNM,
    build_intervals,
    build_windows,
    choose_document_threshold,
    classifier_predict,
    event_results,
    raw_windows,
    resolve_device,
    set_seed,
    split_statistics,
    train_classifier,
    window_axis_center,
    write_csv,
    write_json,
)


EXPERIMENT = "window_axis_centered_raw_2s_tcnm_within_subject_S01_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"{EXPERIMENT}_seed20260802",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def plot_results(
    output_dir: Path,
    training: dict[str, Any],
    confusion: list[list[int]],
) -> None:
    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_weighted_bce"] for row in history])
    axes[0].set_title("Raw-centered TCN-M training BCE")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["validation_pr_auc"] for row in history])
    axes[1].axvline(training["best_epoch"], color="black", linestyle="--")
    axes[1].set_title("Raw-centered TCN-M validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / "tcn_m_training_validation.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("S01 raw-centered TCN-M test confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset = DaphnetDataset.load(args.data_dir)
    records = [record for record in dataset.records if record.subject_id == SUBJECT]
    intervals = build_intervals(records)
    windows = build_windows(records, intervals)
    indices = {
        name: windows.indices(name) for name in ("train", "validation", "test")
    }
    for name, split_indices in indices.items():
        if np.unique(windows.label[split_indices]).size != 2:
            raise ValueError(f"{name} lacks one class")

    features = {
        name: window_axis_center(raw_windows(records, windows, split_indices))
        for name, split_indices in indices.items()
    }
    maximum_center_error = {
        name: float(np.max(np.abs(values.mean(axis=1))))
        for name, values in features.items()
    }
    if max(maximum_center_error.values()) >= 5e-6:
        raise AssertionError(f"raw centering invariant failed: {maximum_center_error}")

    protocol = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": SUBJECT,
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "device": str(device),
        "sampling_rate_hz": FS,
        "input": {
            "shape": ["batch", 9, 128],
            "source": "raw 2 s IMU window",
            "robust_scaler": False,
            "nbm": False,
            "residual_calibration": False,
            "transform": "subtract each window/channel mean over 128 time samples",
            "maximum_observed_center_error": maximum_center_error,
        },
        "windowing_and_split": {
            "identical_to": "nonfog_gru_nbm_tcnm_within_subject_v1_S01_window_axis_centered_seed20260802",
            "window_seconds": 2.0,
            "stride_seconds": 1.0,
            "label_rule": "FoG iff >=50% of final 32 samples are FoG",
            "intervals": [item.__dict__ for item in intervals],
            "statistics": {
                name: split_statistics(windows, split_indices)
                for name, split_indices in indices.items()
            },
        },
        "classifier": {
            "class": ResidualTCNM.__name__,
            "architecture": "same causal TCN-M: 9-32-64-64-128, dilations 1-2-4-8",
            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
            "loss": "BCEWithLogitsLoss(pos_weight=N_nonFoG/N_FoG)",
            "batch_size": 128,
            "maximum_epochs": 30,
            "patience": 6,
            "monitor": "validation PR-AUC",
            "threshold": "validation balanced accuracy over 0.05..0.95 step 0.01",
        },
        "test_policy": "one frozen evaluation after validation-selected epoch and threshold",
    }
    write_json(output_dir / "config.json", protocol)
    print(
        f"PREFLIGHT device={device} stats={protocol['windowing_and_split']['statistics']} "
        f"center_error={maximum_center_error}",
        flush=True,
    )
    if args.dry_run:
        write_json(output_dir / "DRY_RUN.json", {"status": "complete"})
        return

    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    labels = {name: windows.label[split_indices] for name, split_indices in indices.items()}
    model, training = train_classifier(
        features["train"],
        labels["train"],
        features["validation"],
        labels["validation"],
        output_dir,
        device,
        args.seed,
        args.num_workers,
    )
    validation_true, validation_probability = classifier_predict(
        model, features["validation"], labels["validation"], device
    )
    threshold, validation_metrics = choose_document_threshold(
        validation_true, validation_probability
    )
    # The only test inference in this experiment occurs here.
    test_true, test_probability = classifier_predict(
        model, features["test"], labels["test"], device
    )
    test_metrics = binary_metrics(test_true, test_probability, threshold)
    validation_prediction = (validation_probability >= threshold).astype(np.int8)
    test_prediction = (test_probability >= threshold).astype(np.int8)
    events = event_results(records, windows, indices["test"], test_prediction)
    event_metrics = {
        "events": len(events),
        "detected": int(sum(bool(row["detected"]) for row in events)),
        "event_recall": float(np.mean([row["detected"] for row in events])),
    }
    metrics = {
        "selected_threshold": threshold,
        "threshold_source": "validation only",
        "validation": validation_metrics,
        "test": test_metrics,
        "test_event_detection": event_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "training.json",
        {key: value for key, value in training.items() if key != "history"},
    )
    save_npz(
        output_dir / "centered_raw_features.npz",
        train_x=features["train"],
        train_y=labels["train"],
        validation_x=features["validation"],
        validation_y=labels["validation"],
        test_x=features["test"],
        test_y=labels["test"],
    )

    prediction_rows: list[dict[str, Any]] = []
    for name, probability, prediction in (
        ("validation", validation_probability, validation_prediction),
        ("test", test_probability, test_prediction),
    ):
        for local, index in enumerate(indices[name]):
            record = records[int(windows.record_index[index])]
            prediction_rows.append(
                {
                    "split": name,
                    "record_id": record.record_id,
                    "window_start": int(windows.start[index]),
                    "window_end": int(windows.end[index]),
                    "y_true": int(windows.label[index]),
                    "probability": float(probability[local]),
                    "threshold": threshold,
                    "y_pred": int(prediction[local]),
                }
            )
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "test_event_detection.csv", events)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true\\pred": "non-FoG", "non-FoG": test_metrics["tn"], "FoG": test_metrics["fp"]},
            {"true\\pred": "FoG", "non-FoG": test_metrics["fn"], "FoG": test_metrics["tp"]},
        ],
    )
    plot_results(output_dir, training, test_metrics["confusion_matrix"])
    summary = f"""# S01逐窗口逐轴中心化原始信号 + TCN-M

- 完全删除NBM、Scaler、b/sigma和残差生成。
- 输入：原始9通道2秒窗口，每轴沿128点时间维减去该窗口均值。
- TCN-M最佳epoch：{training['best_epoch']}/{training['epochs_completed']}。
- 验证PR-AUC：{training['best_validation_pr_auc']:.6f}；阈值：{threshold:.2f}。

## 测试主指标

- Accuracy：{test_metrics['accuracy']:.6f}
- Balanced Accuracy：{test_metrics['balanced_accuracy']:.6f}
- FoG Precision：{test_metrics['precision']:.6f}
- FoG Recall：{test_metrics['sensitivity']:.6f}
- FoG F1：{test_metrics['f1']:.6f}
- Specificity：{test_metrics['specificity']:.6f}
- PR-AUC：{test_metrics['auprc']:.6f}
- ROC-AUC：{test_metrics['auroc']:.6f}
- 混淆矩阵：{test_metrics['confusion_matrix']}
- FoG事件检出：{event_metrics['detected']}/{event_metrics['events']}
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_json(
        output_dir / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "test_inference_count": 1,
            "metrics": metrics,
        },
    )
    print(
        f"COMPLETE threshold={threshold:.2f} pr_auc={test_metrics['auprc']:.6f} "
        f"accuracy={test_metrics['accuracy']:.6f} recall={test_metrics['sensitivity']:.6f} "
        f"specificity={test_metrics['specificity']:.6f} cm={test_metrics['confusion_matrix']} "
        f"events={event_metrics['detected']}/{event_metrics['events']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
