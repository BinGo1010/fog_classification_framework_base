#!/usr/bin/env python
"""Run a strict LOSO raw-IMU TCN baseline matched to CNBR residual histories.

This experiment deliberately contains no normal-behaviour predictor.  Each
classifier receives robust-scaled trunk accelerometer samples from the same
0.5/1/2/4-second target-history support used by the residual experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetTrunkDataset, WindowTable
from cnbr_fog.evaluation import aggregate_fold_metrics, save_json
from cnbr_fog.histories import (
    HistoryPlan,
    history_block_count,
    make_block_history_input,
    make_common_history_plan,
)

from run_cnbr_fog_loso import (
    deterministic_subsample,
    event_metrics,
    make_sequence_loader,
    parse_folds,
    parse_history_variants,
    parse_subject_list,
    pooled_metrics,
    resolve_device,
    select_validation_subject,
    set_seed,
    train_classifier,
    write_predictions_csv,
)


METRIC_KEYS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw trunk-IMU TCN history baseline with strict LOSO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"E:\fog-merged\dataset\1.Daphnet Freezing of Gait Dataset\processed_trunk"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/raw_tcn_daphnet_trunk_history_loso"),
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-subjects",
        default="",
        help="Subjects removed before scaling, windowing, and LOSO",
    )
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.25)
    parser.add_argument("--normal-guard-seconds", type=float, default=0.5)
    parser.add_argument("--fog-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--flatline-seconds", type=float, default=1.0)
    parser.add_argument("--robust-clip", type=float, default=12.0)
    parser.add_argument("--history-seconds", default="0.5,1,2,4")
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-classifier-windows",
        type=int,
        default=0,
        help="Stratified train-anchor cap for smoke tests; 0 uses all anchors",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.no_grad()
def extract_raw_blocks(
    args: argparse.Namespace,
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler_stats,
    context_samples: int,
    horizon_samples: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extract robust-scaled observed target blocks without invoking an NBM."""

    loader = make_sequence_loader(
        dataset,
        windows,
        indices,
        scaler_stats,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    raw_blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        target = sequence[:, :, context_samples:]
        if target.shape[-1] != horizon_samples:
            raise AssertionError("Raw target block does not match horizon")
        raw_blocks.append(target.numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())
    return {
        "raw": np.ascontiguousarray(
            np.concatenate(raw_blocks).astype(np.float32, copy=False)
        ),
        "y": np.concatenate(labels).astype(np.int8, copy=False),
        "window_index": np.concatenate(window_indices).astype(np.int64, copy=False),
    }


def support_summary(features: dict[str, np.ndarray], plan: HistoryPlan) -> dict:
    labels = np.asarray(features["y"], dtype=np.int8)[plan.anchor_rows]
    return {
        "windows": int(len(labels)),
        "class_counts": np.bincount(labels, minlength=2).astype(int).tolist(),
    }


def add_requested_metrics(metrics: dict) -> dict:
    """Add true two-class Macro-F1 and explicit FoG metric names."""

    tn, fp, fn, tp = [int(metrics[key]) for key in ("tn", "fp", "fn", "tp")]
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    metrics["macro_f1"] = 0.5 * (f1_nonfog + f1_fog)
    metrics["roc_auc"] = metrics.get("auroc")
    metrics["pr_auc"] = metrics.get("auprc")
    metrics["fog_recall"] = metrics.get("sensitivity")
    metrics["fog_f1"] = f1_fog
    return metrics


def write_fold_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = [
        "input",
        "history_seconds",
        "history_blocks",
        "input_samples",
        "test_subject",
        "val_subject",
        "classifier_seed",
        "threshold",
        "n",
        "n_normal",
        "n_fog",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_fold(
    args: argparse.Namespace,
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    test_subject: str,
    context_samples: int,
    horizon_samples: int,
    stride_samples: int,
    variants: list[tuple[str, float, int]],
    device: torch.device,
) -> list[dict]:
    fold_index = dataset.subjects.index(test_subject)
    set_seed(args.seed + fold_index)
    fold_dir = args.output_dir / f"loso_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    val_subject = select_validation_subject(
        test_subject, dataset.subjects, dataset, windows
    )
    train_subjects = [
        subject
        for subject in dataset.subjects
        if subject not in {test_subject, val_subject}
    ]
    scaler_stats = dataset.fit_scaler(train_subjects, clip=args.robust_clip)
    train_indices = dataset.window_indices_for_subjects(windows, train_subjects)
    val_indices = dataset.window_indices_for_subjects(windows, [val_subject])
    test_indices = dataset.window_indices_for_subjects(windows, [test_subject])

    train_features = extract_raw_blocks(
        args,
        dataset,
        windows,
        train_indices,
        scaler_stats,
        context_samples,
        horizon_samples,
        device,
    )
    val_features = extract_raw_blocks(
        args,
        dataset,
        windows,
        val_indices,
        scaler_stats,
        context_samples,
        horizon_samples,
        device,
    )
    test_features = extract_raw_blocks(
        args,
        dataset,
        windows,
        test_indices,
        scaler_stats,
        context_samples,
        horizon_samples,
        device,
    )

    max_history_samples = max(samples for _, _, samples in variants)
    train_plan = make_common_history_plan(
        windows,
        train_features["window_index"],
        horizon_samples,
        stride_samples,
        max_history_samples,
    )
    val_plan = make_common_history_plan(
        windows,
        val_features["window_index"],
        horizon_samples,
        stride_samples,
        max_history_samples,
    )
    test_plan = make_common_history_plan(
        windows,
        test_features["window_index"],
        horizon_samples,
        stride_samples,
        max_history_samples,
    )
    if min(
        len(train_plan.anchor_rows),
        len(val_plan.anchor_rows),
        len(test_plan.anchor_rows),
    ) == 0:
        raise RuntimeError(f"Empty common raw-history support in fold {test_subject}")
    if args.max_classifier_windows > 0:
        plan_rows = np.arange(len(train_plan.anchor_rows), dtype=np.int64)
        plan_labels = train_features["y"][train_plan.anchor_rows]
        selected = deterministic_subsample(
            plan_rows,
            args.max_classifier_windows,
            args.seed + 100 + fold_index,
            plan_labels,
        )
        train_plan = train_plan.take(selected)

    support = {
        "policy": "maximum_history_common_anchors",
        "train": support_summary(train_features, train_plan),
        "validation": support_summary(val_features, val_plan),
        "test": support_summary(test_features, test_plan),
    }
    np.savez_compressed(
        fold_dir / "history_support.npz",
        train_anchor_window_index=train_plan.anchor_window_indices,
        validation_anchor_window_index=val_plan.anchor_window_indices,
        test_anchor_window_index=test_plan.anchor_window_indices,
        train_history_window_index=train_features["window_index"][
            train_plan.max_chain_rows
        ],
        validation_history_window_index=val_features["window_index"][
            val_plan.max_chain_rows
        ],
        test_history_window_index=test_features["window_index"][
            test_plan.max_chain_rows
        ],
    )
    save_json(
        fold_dir / "fold_config.json",
        {
            "experiment": "raw_tcn_history_loso",
            "uses_nbm": False,
            "test_subject": test_subject,
            "val_subject": val_subject,
            "train_subjects": train_subjects,
            "scaler": scaler_stats.as_dict(),
            "history_support": support,
        },
    )
    print(
        f"[fold {test_subject}] train={train_subjects} val={val_subject} "
        f"raw anchors train/val/test={len(train_plan.anchor_rows)}/"
        f"{len(val_plan.anchor_rows)}/{len(test_plan.anchor_rows)}",
        flush=True,
    )

    rows: list[dict] = []
    for input_name, history_seconds, history_samples in variants:
        classifier_seed = args.seed + 10000 + fold_index
        set_seed(classifier_seed)
        classifier_train = make_block_history_input(
            train_features,
            train_plan,
            "raw",
            input_name,
            history_samples,
            horizon_samples,
            stride_samples,
        )
        classifier_val = make_block_history_input(
            val_features,
            val_plan,
            "raw",
            input_name,
            history_samples,
            horizon_samples,
            stride_samples,
        )
        classifier_test = make_block_history_input(
            test_features,
            test_plan,
            "raw",
            input_name,
            history_samples,
            horizon_samples,
            stride_samples,
        )
        metrics, test_prob, test_pred = train_classifier(
            fold_dir,
            input_name,
            args,
            classifier_train,
            classifier_val,
            classifier_test,
            device,
        )
        metrics.update(
            event_metrics(
                dataset,
                windows,
                classifier_test["window_index"],
                test_pred,
            )
        )
        metrics.update(
            {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "history_seconds": history_seconds,
                "input_samples": int(classifier_train[input_name].shape[-1]),
                "history_blocks": history_block_count(
                    history_samples, horizon_samples, stride_samples
                ),
                "classifier_seed": classifier_seed,
            }
        )
        add_requested_metrics(metrics)
        save_json(fold_dir / input_name / "metrics.json", metrics)
        write_predictions_csv(
            fold_dir / input_name / "predictions.csv",
            dataset,
            windows,
            classifier_test["window_index"],
            test_prob,
            test_pred,
        )
        rows.append(metrics)
        print(
            f"[fold {test_subject}] {input_name} "
            f"AUPRC={metrics['pr_auc']:.4f} BA={metrics['balanced_accuracy']:.4f} "
            f"Macro-F1={metrics['macro_f1']:.4f} FoG-F1={metrics['fog_f1']:.4f}",
            flush=True,
        )
        del classifier_train, classifier_val, classifier_test
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del train_features, val_features, test_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = DaphnetTrunkDataset.load(
        args.data_dir, flatline_seconds=args.flatline_seconds
    )
    source_subjects = list(dataset.subjects)
    excluded_subjects = parse_subject_list(args.exclude_subjects)
    unknown = sorted(set(excluded_subjects) - set(source_subjects))
    if unknown:
        raise ValueError(f"Unknown excluded subjects: {unknown}")
    if excluded_subjects:
        excluded = set(excluded_subjects)
        dataset = DaphnetTrunkDataset(
            root=dataset.root,
            records=[
                record
                for record in dataset.records
                if record.subject_id not in excluded
            ],
            sampling_rate_hz=dataset.sampling_rate_hz,
        )

    fs = dataset.sampling_rate_hz
    context_samples = int(round(args.context_seconds * fs))
    horizon_samples = int(round(args.horizon_seconds * fs))
    stride_samples = int(round(args.stride_seconds * fs))
    guard_samples = int(round(args.normal_guard_seconds * fs))
    windows = dataset.make_windows(
        warmup_samples=context_samples,
        target_samples=horizon_samples,
        stride_samples=stride_samples,
        fog_fraction_threshold=args.fog_fraction_threshold,
        normal_guard_samples=guard_samples,
    )
    residual_named = parse_history_variants(
        args.history_seconds,
        fs,
        horizon_samples,
        stride_samples,
    )
    if not residual_named:
        raise ValueError("--history-seconds must contain at least one duration")
    variants = [
        (name.replace("residual_", "raw_", 1), seconds, samples)
        for name, seconds, samples in residual_named
    ]
    input_names = [name for name, _, _ in variants]
    max_history_samples = max(samples for _, _, samples in variants)
    global_support = make_common_history_plan(
        windows,
        np.arange(len(windows), dtype=np.int64),
        horizon_samples,
        stride_samples,
        max_history_samples,
    )
    folds = parse_folds(args.folds, dataset.subjects)
    fold_record_indices = set(
        dataset.subject_record_indices(folds).astype(int).tolist()
    )
    evaluation_window_indices = global_support.anchor_window_indices[
        np.fromiter(
            (
                int(record_index) in fold_record_indices
                for record_index in windows.record_index[
                    global_support.anchor_window_indices
                ]
            ),
            dtype=bool,
            count=len(global_support.anchor_window_indices),
        )
    ]
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "experiment": "raw_tcn_history_loso",
        "uses_nbm": False,
        "input_representation": "robust_scaled_raw_trunk_acceleration",
        "scaler_fit": "valid_non_fog_samples_from_outer_training_subjects_only",
        "sampling_rate_hz": fs,
        "context_samples": context_samples,
        "horizon_samples": horizon_samples,
        "stride_samples": stride_samples,
        "normal_guard_samples": guard_samples,
        "source_subjects": source_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects": dataset.subjects,
        "folds_resolved": folds,
        "inputs_resolved": input_names,
        "history_variants": [
            {
                "input": name,
                "history_seconds": seconds,
                "history_samples": samples,
                "history_blocks": history_block_count(
                    samples, horizon_samples, stride_samples
                ),
            }
            for name, seconds, samples in variants
        ],
        "history_construction": (
            "non_overlapping_horizon_spaced_raw_blocks_common_maximum_support"
        ),
        "records": len(dataset.records),
        "windows": len(windows),
        "window_class_counts": np.bincount(
            windows.label, minlength=2
        ).astype(int).tolist(),
        "evaluation_windows": int(len(evaluation_window_indices)),
        "evaluation_window_class_counts": np.bincount(
            windows.label[evaluation_window_indices], minlength=2
        ).astype(int).tolist(),
        "invalid_samples": int(
            sum((~record.valid).sum() for record in dataset.records)
        ),
    }
    save_json(args.output_dir / "config.json", config)
    print(
        f"[INFO] raw-only device={device} records={len(dataset.records)} "
        f"subjects={dataset.subjects} windows={len(windows)} "
        f"common={config['evaluation_windows']} "
        f"counts={config['evaluation_window_class_counts']} "
        f"folds={folds} inputs={input_names}",
        flush=True,
    )

    all_rows: list[dict] = []
    for test_subject in folds:
        all_rows.extend(
            run_fold(
                args,
                dataset,
                windows,
                test_subject,
                context_samples,
                horizon_samples,
                stride_samples,
                variants,
                device,
            )
        )
        write_fold_summary(args.output_dir / "fold_summary.csv", all_rows)

    aggregate: dict[str, dict] = {}
    for input_name in input_names:
        rows = [row for row in all_rows if row["input"] == input_name]
        pooled = add_requested_metrics(
            pooled_metrics(rows, args.output_dir, input_name)
        )
        aggregate[input_name] = {
            "subject_macro": aggregate_fold_metrics(rows, METRIC_KEYS),
            "pooled": pooled,
            "completed_folds": [row["test_subject"] for row in rows],
        }
    save_json(args.output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
