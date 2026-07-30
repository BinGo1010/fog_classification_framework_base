#!/usr/bin/env python
"""Recompute and audit the four-method FoG reference-baseline suite."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.histories import make_common_history_plan
from cnbr_fog.resume import atomic_json_dump, validate_done
from daphnet_baselines import load_dataset
from run_daphnet_baseline_suite import (
    EXPECTED_CHANNEL_NAMES,
    EXPECTED_LOSO_SUBJECTS,
    SUITE_VERSION,
    add_requested_metrics,
    eligible_indices_for_subjects,
    metrics_from_predictions,
)
from run_cnbr_fog_loso import event_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit FI / TF-SVM / TF-RF / CNN-GRU outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--tolerance", type=float, default=2e-6)
    return parser.parse_args()


def close_enough(first: Any, second: Any, tolerance: float) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return math.isclose(
        float(first),
        float(second),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    config_path = args.output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else Path(config["data_dir"]).resolve()
    )
    failures: list[str] = []
    warnings: list[str] = []
    checked_cells = 0
    missing_cells = 0

    if config.get("suite_version") != SUITE_VERSION:
        failures.append(
            f"suite_version={config.get('suite_version')!r}, expected={SUITE_VERSION}"
        )
    adapter_name = str(config.get("dataset_adapter", "daphnet"))
    if adapter_name == "daphnet":
        if tuple(config.get("channel_names", [])) != EXPECTED_CHANNEL_NAMES:
            failures.append(
                "configured channel order is not the canonical 9-channel order"
            )
        if set(config.get("excluded_subjects", [])) != {"S04", "S10"}:
            failures.append("configured exclusions are not exactly S04/S10")
        if tuple(config.get("subjects", [])) != EXPECTED_LOSO_SUBJECTS:
            failures.append(
                "configured post-exclusion subject list is not canonical"
            )

    loaded = load_dataset(
        adapter_name,
        data_dir,
        excluded_subjects=config["excluded_subjects"],
        flatline_seconds=float(config["flatline_seconds"]),
        zero_tolerance=float(config["zero_tolerance"]),
    )
    dataset = loaded.dataset
    if tuple(dataset.subjects) != tuple(config.get("subjects", [])):
        failures.append(
            "loaded subject list differs from the configured subject list"
        )
    windows = dataset.make_windows(
        warmup_samples=int(config["context_samples"]),
        target_samples=int(config["horizon_samples"]),
        stride_samples=int(config["stride_samples"]),
        fog_fraction_threshold=float(config["fog_fraction_threshold"]),
        normal_guard_samples=int(config["normal_guard_samples"]),
    )
    plan = make_common_history_plan(
        windows,
        np.arange(len(windows), dtype=np.int64),
        int(config["horizon_samples"]),
        int(config["stride_samples"]),
        int(config["history_samples"]),
    )
    eligible = plan.anchor_window_indices

    for subject in config["folds_resolved"]:
        expected_indices = eligible_indices_for_subjects(
            dataset,
            windows,
            eligible,
            [subject],
        )
        reference_indices: np.ndarray | None = None
        reference_truth: np.ndarray | None = None
        for method in config["methods_resolved"]:
            root = args.output_dir / f"loso_{subject}" / method
            done_path = root / "DONE.json"
            if not done_path.exists():
                missing_cells += 1
                message = f"missing {subject}/{method}"
                if args.allow_partial:
                    warnings.append(message)
                else:
                    failures.append(message)
                continue
            try:
                validate_done(
                    done_path,
                    stage="baseline_method",
                    protocol_fingerprint=config["protocol_fingerprint"],
                    task_id=f"{subject}/{method}",
                )
            except Exception as error:
                failures.append(f"{subject}/{method} DONE: {error}")
                continue
            try:
                with (root / "metrics.json").open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    metrics = json.load(handle)
                with np.load(
                    root / "predictions.npz",
                    allow_pickle=False,
                ) as payload:
                    if set(payload.files) != {
                        "window_index",
                        "y_true",
                        "y_prob",
                        "y_pred",
                    }:
                        raise ValueError(
                            f"unexpected prediction arrays {payload.files}"
                        )
                    indices = np.asarray(payload["window_index"], dtype=np.int64)
                    truth = np.asarray(payload["y_true"], dtype=np.int8)
                    probability = np.asarray(payload["y_prob"], dtype=np.float64)
                    prediction = np.asarray(payload["y_pred"], dtype=np.int8)
                if not (
                    indices.shape
                    == truth.shape
                    == probability.shape
                    == prediction.shape
                ):
                    raise ValueError("prediction arrays have unequal shapes")
                if len(np.unique(indices)) != len(indices):
                    raise ValueError("window_index contains duplicates")
                if not np.array_equal(indices, expected_indices):
                    raise ValueError("test anchors differ from common LOSO support")
                if not np.array_equal(truth, windows.label[indices]):
                    raise ValueError("y_true differs from anchor labels")
                if not np.isfinite(probability).all():
                    raise ValueError("y_prob contains non-finite values")
                if np.any((probability < 0.0) | (probability > 1.0)):
                    raise ValueError("y_prob is outside [0,1]")
                expected_prediction = (
                    probability >= float(metrics["threshold"])
                ).astype(np.int8)
                if not np.array_equal(prediction, expected_prediction):
                    raise ValueError("y_pred does not match thresholded y_prob")
                if reference_indices is None:
                    reference_indices = indices
                    reference_truth = truth
                elif not (
                    np.array_equal(indices, reference_indices)
                    and np.array_equal(truth, reference_truth)
                ):
                    raise ValueError("method anchors/labels differ within fold")

                recomputed = add_requested_metrics(
                    binary_metrics(
                        truth,
                        probability,
                        float(metrics["threshold"]),
                    )
                )
                recomputed.update(
                    event_metrics(
                        dataset,
                        windows,
                        indices,
                        prediction,
                    )
                )
                metric_keys = (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "roc_auc",
                    "pr_auc",
                    "fog_recall",
                    "fog_f1",
                    "specificity",
                    "precision",
                    "mcc",
                    "event_sensitivity",
                    "false_alarm_events_per_hour",
                    "median_detection_delay_sec",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                )
                for key in metric_keys:
                    if not close_enough(
                        metrics.get(key),
                        recomputed.get(key),
                        args.tolerance,
                    ):
                        raise ValueError(
                            f"metric {key} differs: "
                            f"{metrics.get(key)!r} vs {recomputed.get(key)!r}"
                        )
                with np.load(
                    root / "validation_predictions.npz",
                    allow_pickle=False,
                ) as payload:
                    val_prob = np.asarray(payload["y_prob"], dtype=np.float64)
                    val_pred = np.asarray(payload["y_pred"], dtype=np.int8)
                    val_true = np.asarray(payload["y_true"], dtype=np.int8)
                    val_indices = np.asarray(
                        payload["window_index"],
                        dtype=np.int64,
                    )
                if not np.array_equal(val_true, windows.label[val_indices]):
                    raise ValueError("validation labels differ from WindowTable")
                if not np.isfinite(val_prob).all() or np.any(
                    (val_prob < 0.0) | (val_prob > 1.0)
                ):
                    raise ValueError("validation y_prob is invalid")
                if not np.array_equal(
                    val_pred,
                    (val_prob >= float(metrics["threshold"])).astype(np.int8),
                ):
                    raise ValueError("validation y_pred threshold mismatch")
                checked_cells += 1
            except Exception as error:
                failures.append(f"{subject}/{method}: {error}")

    aggregate_path = args.output_dir / "aggregate_metrics.json"
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)
        for method, payload in aggregate.items():
            truths: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            predictions: list[np.ndarray] = []
            for subject in payload.get("completed_folds", []):
                with np.load(
                    args.output_dir
                    / f"loso_{subject}"
                    / method
                    / "predictions.npz",
                    allow_pickle=False,
                ) as fold:
                    truths.append(np.asarray(fold["y_true"], dtype=np.int8))
                    probabilities.append(
                        np.asarray(fold["y_prob"], dtype=np.float64)
                    )
                    predictions.append(
                        np.asarray(fold["y_pred"], dtype=np.int8)
                    )
            if truths:
                pooled = metrics_from_predictions(
                    np.concatenate(truths),
                    np.concatenate(probabilities),
                    np.concatenate(predictions),
                )
                for key, value in pooled.items():
                    if not close_enough(
                        payload["pooled"].get(key),
                        value,
                        args.tolerance,
                    ):
                        failures.append(
                            f"aggregate {method}/{key} differs"
                        )

    expected_cells = (
        len(config["folds_resolved"]) * len(config["methods_resolved"])
    )
    report = {
        "audit_version": "fog_reference_baselines_audit.v2",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "data_dir": str(data_dir),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_cells": expected_cells,
        "checked_cells": checked_cells,
        "missing_cells": missing_cells,
        "allow_partial": bool(args.allow_partial),
        "failures": failures,
        "warnings": warnings,
        "status": "pass" if not failures else "fail",
    }
    atomic_json_dump(report, args.output_dir / "audit_report.json")
    print(
        f"[audit] status={report['status']} checked={checked_cells}/"
        f"{expected_cells} missing={missing_cells}",
        flush=True,
    )
    for failure in failures:
        print(f"[audit] ERROR {failure}", flush=True)
    for warning in warnings:
        print(f"[audit] WARNING {warning}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
