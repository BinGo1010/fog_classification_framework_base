#!/usr/bin/env python
"""Within-subject raw-centered classifier ablation with NBM fully removed."""

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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as core
from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics


EXPERIMENT = "raw_window_axis_centered_tcnm_within_subject_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=sorted(core.SUBJECT_SPLITS), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--classifier",
        choices=("tcn_m", "inception_time"),
        default="tcn_m",
    )
    parser.add_argument(
        "--distributed-scaler-protocol",
        action="store_true",
        help=(
            "Mirror the distributed NBM experiment: in every non-holdout block, "
            "use the chronological 80%% fit portion for RobustScaler fitting."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scaler_points(records, windows, indices: np.ndarray) -> int:
    masks: dict[int, np.ndarray] = {}
    for index in np.asarray(indices, dtype=np.int64):
        rec_idx = int(windows.record_index[index])
        masks.setdefault(rec_idx, np.zeros(len(records[rec_idx].y), dtype=bool))
        masks[rec_idx][int(windows.start[index]) : int(windows.end[index])] = True
    return int(sum(mask.sum() for mask in masks.values()))


def build_oof_raw(
    records,
    windows,
    train_indices: np.ndarray,
    distributed_scaler_protocol: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    features = np.empty((len(train_indices), core.WINDOW, 9), dtype=np.float32)
    assigned = np.zeros(len(train_indices), dtype=bool)
    local_lookup = {int(index): row for row, index in enumerate(train_indices)}
    fold_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for holdout in range(5):
        if distributed_scaler_protocol:
            fit_blocks = [b for b in range(5) if b != holdout]
            source_idx = train_indices[
                np.isin(windows.block[train_indices], fit_blocks)
            ]
            fit_idx, _, split_details = core.distributed_nbm_fit_cal_indices(
                windows, source_idx
            )
            calibration_block = None
        else:
            calibration_block = (holdout + 1) % 5
            fit_blocks = [b for b in range(5) if b not in {holdout, calibration_block}]
            fit_idx = train_indices[
                windows.clean_normal[train_indices]
                & np.isin(windows.block[train_indices], fit_blocks)
            ]
            split_details = []
        hold_idx = train_indices[windows.block[train_indices] == holdout]
        if min(len(fit_idx), len(hold_idx)) == 0:
            raise ValueError(f"empty scaler fold {holdout}")
        if np.intersect1d(fit_idx, hold_idx).size:
            raise AssertionError("held-out raw window entered scaler fit")
        scaler = core.fit_scaler_unique_points(records, windows, fit_idx)
        hold_raw = core.raw_windows(records, windows, hold_idx)
        transformed = core.prepare_nbm_windows(scaler, hold_raw, center=True)
        for source_row, index in enumerate(hold_idx):
            local_row = local_lookup[int(index)]
            if assigned[local_row]:
                raise AssertionError("duplicate OOF raw assignment")
            features[local_row] = transformed[source_row]
            assigned[local_row] = True
            record = records[int(windows.record_index[index])]
            manifest.append(
                {
                    "sample_id": f"{record.record_id}:{int(windows.start[index])}:{int(windows.end[index])}",
                    "subject_id": core.SUBJECT,
                    "record_id": record.record_id,
                    "window_start": int(windows.start[index]),
                    "window_end": int(windows.end[index]),
                    "label": int(windows.label[index]),
                    "outer_split": "train",
                    "inner_fold": holdout + 1,
                    "scaler_seen_this_window": False,
                }
            )
        fold_rows.append(
            {
                "inner_fold": holdout + 1,
                "holdout_block": holdout,
                "excluded_calibration_block": calibration_block,
                "distributed_scaler_protocol": distributed_scaler_protocol,
                "distributed_split_details": split_details,
                "scaler_fit_blocks": fit_blocks,
                "scaler_fit_clean_windows": int(len(fit_idx)),
                "scaler_fit_unique_points": scaler_points(records, windows, fit_idx),
                "holdout_windows": int(len(hold_idx)),
                "scaler": scaler.as_dict(),
            }
        )
    if not np.all(assigned):
        raise AssertionError("not every TCN training window received OOF raw features")
    return features, fold_rows, manifest


def fit_final_scaler(
    records,
    windows,
    train_indices: np.ndarray,
    distributed_scaler_protocol: bool = False,
):
    if distributed_scaler_protocol:
        fit_idx, _, split_details = core.distributed_nbm_fit_cal_indices(
            windows, train_indices
        )
        fit_blocks = [0, 1, 2, 3, 4]
    else:
        fit_idx = train_indices[
            windows.clean_normal[train_indices]
            & np.isin(windows.block[train_indices], [0, 1, 2, 3])
        ]
        fit_blocks = [0, 1, 2, 3]
        split_details = []
    scaler = core.fit_scaler_unique_points(records, windows, fit_idx)
    return scaler, {
        "fit_blocks": fit_blocks,
        "distributed_scaler_protocol": distributed_scaler_protocol,
        "distributed_split_details": split_details,
        "fit_clean_windows": int(len(fit_idx)),
        "fit_unique_points": scaler_points(records, windows, fit_idx),
        "scaler": scaler.as_dict(),
    }


def save_plots(output_dir: Path, training: dict[str, Any], confusion: list[list[int]]) -> None:
    classifier_name = str(training.get("model_name", "tcn_m"))
    display_name = "InceptionTime" if classifier_name == "inception_time" else "TCN-M"
    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_weighted_bce"] for row in history])
    axes[0].set_title(f"Raw centered {display_name} training BCE")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["validation_pr_auc"] for row in history])
    axes[1].axvline(training["best_epoch"], color="black", linestyle="--")
    axes[1].set_title("Validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / f"{classifier_name}_training_validation.png", dpi=180)
    plt.close(fig)

    cm = np.asarray(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{core.SUBJECT} raw-centered {display_name} test confusion")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    core.SUBJECT = str(args.subject)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    core.set_seed(args.seed)
    device = core.resolve_device(args.device)
    dataset = DaphnetDataset.load(args.data_dir)
    records = [record for record in dataset.records if record.subject_id == core.SUBJECT]
    intervals = core.build_intervals(records)
    windows = core.build_windows(records, intervals)
    train_indices = windows.indices("train")
    validation_indices = windows.indices("validation")
    test_indices = windows.indices("test")
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        if np.unique(windows.label[indices]).size != 2:
            raise ValueError(f"{name} lacks one class")
    config = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": core.SUBJECT,
        "seed": args.seed,
        "device": str(device),
        "data_dir": str(args.data_dir.resolve()),
        "input": "RobustScaler(raw 2 s window), then subtract each window/channel 128-point mean",
        "nbm_removed": True,
        "residual_removed": True,
        "residual_clipping_removed": True,
        "classifier": args.classifier,
        "split_intervals": [vars(item) for item in intervals],
        "split_statistics": {
            "train": core.split_statistics(windows, train_indices),
            "validation": core.split_statistics(windows, validation_indices),
            "test": core.split_statistics(windows, test_indices),
        },
        "scaler_crossfit": {
            "enabled_for_tcn_training_features": True,
            "distributed_protocol": bool(args.distributed_scaler_protocol),
            "fit_rule": (
                "chronological 80% fit portions from all four non-holdout blocks"
                if args.distributed_scaler_protocol
                else "clean non-FoG windows from three source blocks; held-out and next calibration block excluded"
            ),
        },
        "final_scaler": (
            "chronological 80% fit portions from all five train blocks; shared by validation and test"
            if args.distributed_scaler_protocol
            else "clean non-FoG train blocks 0..3 only; shared by validation and test"
        ),
        "classifier_config": {
            "architecture": (
                "six InceptionTime modules, kernels 39/19/9, 32 channels per branch, residual every three modules, GAP"
                if args.classifier == "inception_time"
                else "causal blocks 9-32-64-64-128, dilations 1-2-4-8, global average pooling"
            ),
            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
            "batch_size": 128,
            "max_epochs": 30,
            "patience": 6,
            "monitor": "validation PR-AUC",
            "threshold": "validation balanced accuracy over 0.05..0.95 step 0.01",
        },
    }
    core.write_json(output_dir / "config.json", config)
    print(
        f"PREFLIGHT subject={core.SUBJECT} device={device} "
        f"stats={config['split_statistics']}",
        flush=True,
    )
    if args.dry_run:
        core.write_json(output_dir / "DRY_RUN.json", {"status": "complete"})
        return

    train_x, fold_scalers, manifest = build_oof_raw(
        records, windows, train_indices, args.distributed_scaler_protocol
    )
    final_scaler, final_scaler_info = fit_final_scaler(
        records, windows, train_indices, args.distributed_scaler_protocol
    )
    validation_x = core.prepare_nbm_windows(
        final_scaler, core.raw_windows(records, windows, validation_indices), center=True
    )
    test_x = core.prepare_nbm_windows(
        final_scaler, core.raw_windows(records, windows, test_indices), center=True
    )
    train_y = windows.label[train_indices]
    validation_y = windows.label[validation_indices]
    test_y = windows.label[test_indices]
    center_errors = {
        "train_oof": float(np.max(np.abs(train_x.mean(axis=1)))),
        "validation": float(np.max(np.abs(validation_x.mean(axis=1)))),
        "test": float(np.max(np.abs(test_x.mean(axis=1)))),
    }
    # Float32 reduction can leave a few 1e-5 of numerical residue when a
    # RobustScaler produces unusually large values (notably S06 test data).
    if max(center_errors.values()) > 1e-4:
        raise AssertionError(f"raw window centering failed: {center_errors}")
    core.write_csv(output_dir / "artifacts" / "oof_manifest.csv", manifest)
    core.save_npz(
        output_dir / "artifacts" / "features.npz",
        train_oof_raw_centered=train_x,
        train_y=train_y,
        train_window_index=train_indices,
        validation_raw_centered=validation_x,
        validation_y=validation_y,
        validation_window_index=validation_indices,
        test_raw_centered=test_x,
        test_y=test_y,
        test_window_index=test_indices,
    )
    core.write_json(
        output_dir / "scalers.json",
        {
            "inner_folds": fold_scalers,
            "final": final_scaler_info,
            "maximum_absolute_window_axis_mean": center_errors,
        },
    )

    classifier, training = core.train_classifier(
        train_x,
        train_y,
        validation_x,
        validation_y,
        output_dir,
        device,
        args.seed,
        args.num_workers,
        args.classifier,
    )
    val_true, val_prob = core.classifier_predict(
        classifier, validation_x, validation_y, device
    )
    threshold, validation_metrics = core.choose_document_threshold(val_true, val_prob)
    test_true, test_prob = core.classifier_predict(classifier, test_x, test_y, device)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    validation_metrics = core.add_requested_macro_metrics(validation_metrics)
    test_metrics = core.add_requested_macro_metrics(test_metrics)
    val_pred = (val_prob >= threshold).astype(np.int8)
    test_pred = (test_prob >= threshold).astype(np.int8)
    events = core.event_results(records, windows, test_indices, test_pred)
    event_summary = {
        "events": len(events),
        "detected": int(sum(bool(row["detected"]) for row in events)),
        "event_recall": float(np.mean([row["detected"] for row in events])),
    }
    metrics = {
        "selected_threshold": threshold,
        "threshold_source": "validation only",
        "validation": validation_metrics,
        "test": test_metrics,
        "test_event_detection": event_summary,
    }
    core.write_json(output_dir / "metrics.json", metrics)
    core.write_json(
        output_dir / "training.json",
        {key: value for key, value in training.items() if key != "history"},
    )
    rows: list[dict[str, Any]] = []
    for split, indices, probability, prediction in (
        ("validation", validation_indices, val_prob, val_pred),
        ("test", test_indices, test_prob, test_pred),
    ):
        for local, index in enumerate(indices):
            record = records[int(windows.record_index[index])]
            rows.append(
                {
                    "split": split,
                    "subject_id": core.SUBJECT,
                    "record_id": record.record_id,
                    "window_start": int(windows.start[index]),
                    "window_end": int(windows.end[index]),
                    "y_true": int(windows.label[index]),
                    "fog_probability": float(probability[local]),
                    "threshold": threshold,
                    "y_pred": int(prediction[local]),
                }
            )
    core.write_csv(output_dir / "predictions.csv", rows)
    core.write_csv(output_dir / "test_event_detection.csv", events)
    save_plots(output_dir, training, test_metrics["confusion_matrix"])
    warnings: list[str] = []
    if int(test_metrics["n_fog"]) < 10:
        warnings.append(f"Only {test_metrics['n_fog']} FoG test window(s); metrics are unstable.")
    summary = f"""# {core.SUBJECT} raw centered {args.classifier} ablation

- NBM and residual calibration are completely removed.
- Input is cross-fitted RobustScaler raw 2-second IMU followed by per-window/per-axis centering.
- Classifier best/completed epoch: {training['best_epoch']}/{training['epochs_completed']}.
- Validation threshold: {threshold:.2f}.
- ACC: {test_metrics['acc']:.6f}.
- Macro-Precision: {test_metrics['macro_precision']:.6f}.
- Macro-Recall: {test_metrics['macro_recall']:.6f}.
- Macro-F1: {test_metrics['macro_f1']:.6f}.
- Confusion matrix: {test_metrics['confusion_matrix']}.
- Event detection: {event_summary['detected']}/{event_summary['events']}.
- Warning: {'; '.join(warnings) if warnings else 'none'}
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    core.write_json(
        output_dir / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "warnings": warnings,
        },
    )
    print(
        f"COMPLETE subject={core.SUBJECT} threshold={threshold:.2f} "
        f"acc={test_metrics['acc']:.6f} macro_precision={test_metrics['macro_precision']:.6f} "
        f"macro_recall={test_metrics['macro_recall']:.6f} "
        f"macro_f1={test_metrics['macro_f1']:.6f} cm={test_metrics['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
