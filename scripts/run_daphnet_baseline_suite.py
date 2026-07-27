#!/usr/bin/env python
"""Run three reference FoG baselines under one strict Daphnet LOSO protocol.

The suite contains:

* Freeze Index: a validation-thresholded domain rule;
* time-frequency handcrafted features with an RBF-SVM;
* a raw-IMU CNN-GRU classifier.

All methods use the same causal raw-history support and the same label attached
to the final 0.5-second target block. S04 and S10 are removed before windowing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnbr_fog.data import DaphnetDataset, RobustChannelScaler, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.histories import make_common_history_plan
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    capture_rng_state,
    dataset_fingerprint,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_checkpoint,
    validate_done,
)
from daphnet_baselines import (
    CNNGRUClassifier,
    HistoryWindowDataset,
    TimeFrequencyFeatureExtractor,
    freeze_index_features,
    materialize_history_windows,
    parameter_count,
)
from run_cnbr_fog_loso import (
    deterministic_subsample,
    event_metrics,
    parse_folds,
    parse_subject_list,
    select_validation_subject,
    write_predictions_csv,
)


SUITE_VERSION = "daphnet_reference_baselines.v1"
METHODS = ("cnn_gru", "freeze_index", "tf_svm")
EXPECTED_CHANNEL_NAMES = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)
EXPECTED_LOSO_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
)
SENSOR_SETS = {
    "all": tuple(range(9)),
    "ankle": (0, 1, 2),
    "thigh": (3, 4, 5),
    "trunk": (6, 7, 8),
}
IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_baseline_suite.py",
    "scripts/run_cnbr_fog_loso.py",
    "daphnet_baselines/__init__.py",
    "daphnet_baselines/data.py",
    "daphnet_baselines/features.py",
    "daphnet_baselines/models.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/resume.py",
)
METRIC_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daphnet FI / time-frequency SVM / CNN-GRU LOSO suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "daphnet_reference_baselines_seed42",
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument("--worker-fold", default="")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--exclude-subjects", default="S04,S10")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.25)
    parser.add_argument("--input-seconds", type=float, default=4.0)
    parser.add_argument("--normal-guard-seconds", type=float, default=0.5)
    parser.add_argument("--fog-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--flatline-seconds", type=float, default=1.0)
    parser.add_argument("--zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--robust-clip", type=float, default=12.0)
    parser.add_argument(
        "--sensor-set",
        choices=tuple(SENSOR_SETS),
        default="all",
        help="Channels supplied to SVM and CNN-GRU",
    )

    parser.add_argument(
        "--fi-channels",
        default="ankle_acc_vertical",
        help="Comma-separated raw channels; multiple scores use --fi-aggregation",
    )
    parser.add_argument(
        "--fi-aggregation",
        choices=("power_pool", "max", "mean"),
        default="power_pool",
    )
    parser.add_argument(
        "--fi-squared-ratio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--fi-power-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tune a low-motion total-power gate using train/validation only",
    )
    parser.add_argument(
        "--fi-power-quantiles",
        default="0,0.01,0.05,0.1,0.2",
        help="Training-power quantiles searched when --fi-power-gate is enabled",
    )

    parser.add_argument("--svm-c-grid", default="0.1,1,10")
    parser.add_argument("--svm-kernel", choices=("rbf", "linear"), default="rbf")
    parser.add_argument("--svm-max-iter", type=int, default=-1)
    parser.add_argument("--svm-cache-mb", type=float, default=2048.0)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument(
        "--feature-magnitudes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--cnn-channels", default="32,64")
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument(
        "--gru-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--classifier-patience", type=int, default=5)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=0,
        help="Shared stratified train cap for smoke tests; 0 uses all anchors",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def parse_methods(specification: str) -> list[str]:
    requested = [
        value.strip().lower()
        for value in str(specification).split(",")
        if value.strip()
    ]
    if not requested:
        raise ValueError("At least one baseline method is required")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate methods in {specification!r}")
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={METHODS}")
    # A fixed execution order makes resume and output comparison stable. The
    # neural method runs first so multi-GPU jobs do not sit idle behind SVM.
    return [method for method in METHODS if method in requested]


def parse_positive_floats(specification: str, label: str) -> list[float]:
    values: list[float] = []
    for raw in str(specification).split(","):
        if not raw.strip():
            continue
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} values must be finite and positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def parse_positive_ints(specification: str, label: str) -> tuple[int, ...]:
    values = tuple(
        int(raw.strip())
        for raw in str(specification).split(",")
        if raw.strip()
    )
    if not values or min(values) <= 0:
        raise ValueError(f"{label} must contain positive integers")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    positive = {
        "context_seconds": args.context_seconds,
        "horizon_seconds": args.horizon_seconds,
        "stride_seconds": args.stride_seconds,
        "input_seconds": args.input_seconds,
        "normal_guard_seconds": args.normal_guard_seconds,
        "flatline_seconds": args.flatline_seconds,
        "robust_clip": args.robust_clip,
        "svm_cache_mb": args.svm_cache_mb,
        "feature_batch_size": args.feature_batch_size,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "classifier_lr": args.classifier_lr,
        "batch_size": args.batch_size,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.num_workers < 0 or args.max_train_windows < 0:
        raise ValueError("worker counts and train cap must be non-negative")
    if not 0.0 < args.fog_fraction_threshold <= 1.0:
        raise ValueError("--fog-fraction-threshold must be in (0,1]")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0,1)")
    parse_methods(args.methods)
    parse_positive_floats(args.svm_c_grid, "--svm-c-grid")
    parse_positive_ints(args.cnn_channels, "--cnn-channels")
    quantiles = [
        float(value)
        for value in str(args.fi_power_quantiles).split(",")
        if value.strip()
    ]
    if not quantiles or any(value < 0.0 or value >= 1.0 for value in quantiles):
        raise ValueError("--fi-power-quantiles must contain values in [0,1)")
    if args.svm_max_iter == 0 or args.svm_max_iter < -1:
        raise ValueError("--svm-max-iter must be -1 or a positive integer")


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {specification}")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {
        "files": files,
        "sha256": canonical_fingerprint(files),
    }


def environment_payload(device: torch.device) -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "sklearn_version": __import__("sklearn").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        ),
        "selected_device": str(device),
        "command": [sys.executable, *sys.argv],
    }


def atomic_csv_write(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_joblib_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.joblib")
    try:
        joblib.dump(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def save_or_validate_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(f"Saved JSON is incompatible: {path}")
        return
    atomic_json_dump(payload, path)


def add_requested_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = (
        int(metrics[key]) for key in ("tn", "fp", "fn", "tp")
    )
    f1_nonfog = (
        2.0 * tn / (2.0 * tn + fp + fn)
        if 2 * tn + fp + fn
        else 0.0
    )
    f1_fog = (
        2.0 * tp / (2.0 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 0.0
    )
    metrics["macro_f1"] = 0.5 * (f1_nonfog + f1_fog)
    metrics["roc_auc"] = metrics.get("auroc")
    metrics["pr_auc"] = metrics.get("auprc")
    metrics["fog_recall"] = metrics.get("sensitivity")
    metrics["fog_f1"] = f1_fog
    return metrics


def metrics_from_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(y_score, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.int8)
    if not (truth.shape == score.shape == prediction.shape):
        raise ValueError("Prediction arrays must have the same shape")
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    recall_fog = tp / (tp + fn) if tp + fn else 0.0
    recall_nonfog = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1_fog = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_nonfog = 2.0 * tn / (2.0 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    denominator = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)
    )
    return {
        "n": int(len(truth)),
        "n_normal": int(np.sum(truth == 0)),
        "n_fog": int(np.sum(truth == 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(truth), 1),
        "balanced_accuracy": 0.5 * (recall_fog + recall_nonfog),
        "macro_f1": 0.5 * (f1_fog + f1_nonfog),
        "roc_auc": (
            float(roc_auc_score(truth, score))
            if np.unique(truth).size == 2
            else None
        ),
        "pr_auc": (
            float(average_precision_score(truth, score))
            if np.unique(truth).size == 2
            else None
        ),
        "fog_recall": recall_fog,
        "fog_f1": f1_fog,
        "specificity": recall_nonfog,
        "precision": precision,
        "mcc": (
            (tp * tn - fp * fn) / denominator if denominator else 0.0
        ),
    }


def parse_fi_channel_indices(
    specification: str,
    channel_names: Sequence[str],
) -> list[int]:
    names = [
        value.strip()
        for value in str(specification).split(",")
        if value.strip()
    ]
    if not names:
        raise ValueError("--fi-channels must not be empty")
    if len(names) != len(set(names)):
        raise ValueError("--fi-channels contains duplicates")
    unknown = [name for name in names if name not in channel_names]
    if unknown:
        raise ValueError(
            f"Unknown FI channels {unknown}; available={tuple(channel_names)}"
        )
    return [list(channel_names).index(name) for name in names]


def filter_dataset(
    dataset: DaphnetDataset,
    excluded_subjects: Sequence[str],
) -> DaphnetDataset:
    excluded = set(excluded_subjects)
    return DaphnetDataset(
        root=dataset.root,
        records=[
            record
            for record in dataset.records
            if record.subject_id not in excluded
        ],
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
    )


def eligible_indices_for_subjects(
    dataset: DaphnetDataset,
    windows: WindowTable,
    eligible_indices: np.ndarray,
    subjects: Sequence[str],
) -> np.ndarray:
    records = set(
        dataset.subject_record_indices(subjects).astype(int).tolist()
    )
    mask = np.fromiter(
        (
            int(windows.record_index[int(window_index)]) in records
            for window_index in eligible_indices
        ),
        dtype=bool,
        count=len(eligible_indices),
    )
    return np.asarray(eligible_indices[mask], dtype=np.int64)


def class_support(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    return {
        "windows": int(len(labels)),
        "class_counts": np.bincount(labels, minlength=2).astype(int).tolist(),
    }


def build_protocol(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    source_subjects: list[str],
    excluded_subjects: list[str],
    folds: list[str],
    methods: list[str],
    data_sha256: str,
    windows: WindowTable,
    eligible_indices: np.ndarray,
    samples: dict[str, int],
    fi_channel_indices: list[int],
    cnn_channels: tuple[int, ...],
    svm_c_grid: list[float],
    device: torch.device,
) -> dict[str, Any]:
    protocol = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channel_names": list(dataset.channel_names),
        "source_subjects": source_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects": list(dataset.subjects),
        "folds_resolved": folds,
        "methods_resolved": methods,
        "context_samples": samples["context"],
        "horizon_samples": samples["horizon"],
        "stride_samples": samples["stride"],
        "history_samples": samples["history"],
        "history_seconds": args.input_seconds,
        "normal_guard_samples": samples["guard"],
        "fog_fraction_threshold": args.fog_fraction_threshold,
        "flatline_seconds": args.flatline_seconds,
        "zero_tolerance": args.zero_tolerance,
        "robust_clip": args.robust_clip,
        "sensor_set": args.sensor_set,
        "sensor_channel_indices": list(SENSOR_SETS[args.sensor_set]),
        "fi_channel_indices": fi_channel_indices,
        "fi_channel_names": [
            dataset.channel_names[index] for index in fi_channel_indices
        ],
        "fi_aggregation": args.fi_aggregation,
        "fi_squared_ratio": args.fi_squared_ratio,
        "fi_power_gate": args.fi_power_gate,
        "fi_power_quantiles": [
            float(value)
            for value in str(args.fi_power_quantiles).split(",")
            if value.strip()
        ],
        "fi_locomotor_band_hz": [0.5, 3.0],
        "fi_freeze_band_hz": [3.0, 8.0],
        "svm_kernel": args.svm_kernel,
        "svm_c_grid": svm_c_grid,
        "svm_gamma": "scale",
        "svm_class_weight": "balanced",
        "svm_max_iter": args.svm_max_iter,
        "svm_cache_mb": args.svm_cache_mb,
        "feature_batch_size": args.feature_batch_size,
        "feature_magnitudes": args.feature_magnitudes,
        "cnn_channels": list(cnn_channels),
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "gru_bidirectional": args.gru_bidirectional,
        "dropout": args.dropout,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_train_windows": args.max_train_windows,
        "seed": args.seed,
        "amp": args.amp,
        "deterministic": args.deterministic,
        "window_count": int(len(windows)),
        "window_class_counts": np.bincount(
            windows.label,
            minlength=2,
        ).astype(int).tolist(),
        "evaluation_windows": int(len(eligible_indices)),
        "evaluation_window_class_counts": np.bincount(
            windows.label[eligible_indices],
            minlength=2,
        ).astype(int).tolist(),
        "anchor_policy": (
            f"maximum_{args.input_seconds:g}s_history_common_anchors"
        ),
        "label_policy": (
            f"final_{args.horizon_seconds:g}s_fog_fraction_at_least_"
            f"{args.fog_fraction_threshold:g}"
        ),
    }
    fingerprint = canonical_fingerprint(protocol)
    return {
        **protocol,
        "protocol_fingerprint": fingerprint,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "resume": bool(args.resume),
        "num_workers": int(args.num_workers),
    }


def prepare_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    eligible_indices: np.ndarray,
    test_subject: str,
) -> tuple[
    Path,
    str,
    list[str],
    RobustChannelScaler,
    dict[str, np.ndarray],
]:
    fold_index = dataset.subjects.index(test_subject)
    val_subject = select_validation_subject(
        test_subject,
        dataset.subjects,
        dataset,
        windows,
    )
    train_subjects = [
        subject
        for subject in dataset.subjects
        if subject not in {test_subject, val_subject}
    ]
    scaler = dataset.fit_scaler(train_subjects, clip=args.robust_clip)
    split_indices = {
        "train": eligible_indices_for_subjects(
            dataset,
            windows,
            eligible_indices,
            train_subjects,
        ),
        "validation": eligible_indices_for_subjects(
            dataset,
            windows,
            eligible_indices,
            [val_subject],
        ),
        "test": eligible_indices_for_subjects(
            dataset,
            windows,
            eligible_indices,
            [test_subject],
        ),
    }
    if args.max_train_windows > 0:
        candidates = np.arange(
            len(split_indices["train"]),
            dtype=np.int64,
        )
        selected_rows = deterministic_subsample(
            candidates,
            args.max_train_windows,
            args.seed + 100 + fold_index,
            windows.label[split_indices["train"]],
        )
        split_indices["train"] = split_indices["train"][selected_rows]
    for split, indices in split_indices.items():
        if len(indices) == 0 or np.unique(windows.label[indices]).size < 2:
            raise RuntimeError(
                f"Fold {test_subject} {split} split lacks two-class support"
            )

    fold_root = args.output_dir / f"loso_{test_subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "fold_index": fold_index,
        "method_seed": args.seed + 10000 + fold_index,
        "scaler": scaler.as_dict(),
        "support": {
            split: class_support(windows.label[indices])
            for split, indices in split_indices.items()
        },
    }
    save_or_validate_json(fold_root / "fold_config.json", fold_config)
    support_path = fold_root / "input_support.npz"
    if support_path.exists():
        with np.load(support_path, allow_pickle=False) as payload:
            for split, indices in split_indices.items():
                if not np.array_equal(payload[split], indices):
                    raise ValueError(
                        f"Saved input support differs for {test_subject}/{split}"
                    )
    else:
        atomic_npz_save(support_path, **split_indices)
    return fold_root, val_subject, train_subjects, scaler, split_indices


def checkpoint_base(
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
    }


def method_is_complete(
    root: Path,
    protocol_fingerprint: str,
    task_id: str,
) -> bool:
    complete = validate_done(
        root / "DONE.json",
        stage="baseline_method",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
    )
    return complete is not None


def write_predictions_csv_atomic(
    path: Path,
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        write_predictions_csv(
            temporary,
            dataset,
            windows,
            window_indices,
            y_prob,
            y_pred,
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def finalise_method_artifacts(
    *,
    root: Path,
    protocol_fingerprint: str,
    task_id: str,
    metrics: dict[str, Any],
    validation_indices: np.ndarray,
    validation_true: np.ndarray,
    validation_prob: np.ndarray,
    validation_pred: np.ndarray,
    test_indices: np.ndarray,
    test_true: np.ndarray,
    test_prob: np.ndarray,
    test_pred: np.ndarray,
    dataset: DaphnetDataset,
    windows: WindowTable,
    additional_artifacts: dict[str, Path],
) -> dict[str, Any]:
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions.npz"
    validation_path = root / "validation_predictions.npz"
    csv_path = root / "predictions.csv"
    atomic_json_dump(metrics, metrics_path)
    atomic_npz_save(
        predictions_path,
        window_index=np.asarray(test_indices, dtype=np.int64),
        y_true=np.asarray(test_true, dtype=np.int8),
        y_prob=np.asarray(test_prob, dtype=np.float64),
        y_pred=np.asarray(test_pred, dtype=np.int8),
    )
    atomic_npz_save(
        validation_path,
        window_index=np.asarray(validation_indices, dtype=np.int64),
        y_true=np.asarray(validation_true, dtype=np.int8),
        y_prob=np.asarray(validation_prob, dtype=np.float64),
        y_pred=np.asarray(validation_pred, dtype=np.int8),
    )
    write_predictions_csv_atomic(
        csv_path,
        dataset,
        windows,
        test_indices,
        test_prob,
        test_pred,
    )
    artifacts = {
        "metrics": metrics_path.resolve(),
        "predictions": predictions_path.resolve(),
        "validation_predictions": validation_path.resolve(),
        "predictions_csv": csv_path.resolve(),
        **{
            name: path.resolve()
            for name, path in additional_artifacts.items()
        },
    }
    atomic_json_dump(
        done_payload(
            stage="baseline_method",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            relative_to=root,
            artifacts=artifacts,
        ),
        root / "DONE.json",
    )
    return metrics


def common_metrics(
    *,
    method: str,
    test_subject: str,
    val_subject: str,
    args: argparse.Namespace,
    samples: dict[str, int],
    validation_true: np.ndarray,
    validation_prob: np.ndarray,
    test_true: np.ndarray,
    test_prob: np.ndarray,
    test_indices: np.ndarray,
    dataset: DaphnetDataset,
    windows: WindowTable,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    threshold, validation_metrics = choose_threshold(
        validation_true,
        validation_prob,
    )
    validation_pred = (
        np.asarray(validation_prob, dtype=np.float64) >= threshold
    ).astype(np.int8)
    test_pred = (
        np.asarray(test_prob, dtype=np.float64) >= threshold
    ).astype(np.int8)
    metrics = add_requested_metrics(
        binary_metrics(test_true, test_prob, threshold)
    )
    metrics.update(
        event_metrics(
            dataset,
            windows,
            test_indices,
            test_pred,
        )
    )
    metrics.update(
        {
            "experiment_id": method,
            "method": method,
            "input": f"raw_h{args.input_seconds:g}s",
            "history_seconds": float(args.input_seconds),
            "history_samples": int(samples["history"]),
            "test_subject": test_subject,
            "val_subject": val_subject,
            "seed": int(args.seed),
            "validation": add_requested_metrics(validation_metrics),
        }
    )
    return metrics, validation_pred, test_pred, float(threshold)


def _parse_fi_quantiles(specification: str) -> list[float]:
    values: list[float] = []
    for raw in str(specification).split(","):
        if not raw.strip():
            continue
        value = float(raw)
        if value not in values:
            values.append(value)
    return values


def _fi_split(
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    history_samples: int,
    fi_channel_indices: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    histories, labels, indices = materialize_history_windows(
        dataset.records,
        windows,
        window_indices,
        history_samples,
        scaler=None,
    )
    features = freeze_index_features(
        histories,
        dataset.sampling_rate_hz,
        fi_channel_indices,
        aggregation=args.fi_aggregation,
        squared_ratio=args.fi_squared_ratio,
    )
    return labels, {**features, "window_index": indices}


def run_freeze_index(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    fold_root: Path,
    test_subject: str,
    val_subject: str,
    dataset: DaphnetDataset,
    windows: WindowTable,
    split_indices: dict[str, np.ndarray],
    samples: dict[str, int],
    fi_channel_indices: list[int],
) -> dict[str, Any]:
    method = "freeze_index"
    task_id = f"{test_subject}/{method}"
    root = fold_root / method
    root.mkdir(parents=True, exist_ok=True)
    if args.resume and method_is_complete(
        root,
        config["protocol_fingerprint"],
        task_id,
    ):
        with (root / "metrics.json").open("r", encoding="utf-8") as handle:
            print(f"  [{test_subject}] {method}: complete", flush=True)
            return json.load(handle)

    started = time.perf_counter()
    val_true, validation = _fi_split(
        dataset,
        windows,
        split_indices["validation"],
        samples["history"],
        fi_channel_indices,
        args,
    )
    test_true, test = _fi_split(
        dataset,
        windows,
        split_indices["test"],
        samples["history"],
        fi_channel_indices,
        args,
    )
    val_score = np.asarray(validation["score"], dtype=np.float64)
    test_score = np.asarray(test["score"], dtype=np.float64)
    power_threshold: float | None = None
    power_quantile: float | None = None

    if args.fi_power_gate:
        _, train = _fi_split(
            dataset,
            windows,
            split_indices["train"],
            samples["history"],
            fi_channel_indices,
            args,
        )
        candidates = [
            (
                quantile,
                float(
                    np.quantile(
                        train["total_power"],
                        quantile,
                        method="linear",
                    )
                ),
            )
            for quantile in _parse_fi_quantiles(args.fi_power_quantiles)
        ]
        best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
        for quantile, candidate in candidates:
            candidate_score = np.where(
                validation["total_power"] >= candidate,
                validation["score"],
                0.0,
            )
            threshold, candidate_metrics = choose_threshold(
                val_true,
                candidate_score,
            )
            key = (
                float(candidate_metrics["balanced_accuracy"] or 0.0),
                float(candidate_metrics["f1"] or 0.0),
                float(threshold),
                -float(quantile),
            )
            if key > best_key:
                best_key = key
                power_threshold = candidate
                power_quantile = quantile
        if power_threshold is None:
            raise RuntimeError("Freeze Index power-gate search produced no candidate")
        val_score = np.where(
            validation["total_power"] >= power_threshold,
            validation["score"],
            0.0,
        )
        test_score = np.where(
            test["total_power"] >= power_threshold,
            test["score"],
            0.0,
        )

    metrics, val_pred, test_pred, threshold = common_metrics(
        method=method,
        test_subject=test_subject,
        val_subject=val_subject,
        args=args,
        samples=samples,
        validation_true=val_true,
        validation_prob=val_score,
        test_true=test_true,
        test_prob=test_score,
        test_indices=test["window_index"],
        dataset=dataset,
        windows=windows,
    )
    raw_threshold = (
        float("inf")
        if threshold >= 1.0
        else float(threshold / max(1.0 - threshold, 1e-12))
    )
    metrics.update(
        {
            "fi_channel_names": [
                dataset.channel_names[index] for index in fi_channel_indices
            ],
            "fi_aggregation": args.fi_aggregation,
            "fi_squared_ratio": bool(args.fi_squared_ratio),
            "fi_score_threshold": threshold,
            "fi_raw_threshold": raw_threshold,
            "power_gate_enabled": bool(args.fi_power_gate),
            "power_threshold": power_threshold,
            "power_train_quantile": power_quantile,
            "elapsed_sec": time.perf_counter() - started,
        }
    )
    rule_path = root / "rule.json"
    feature_path = root / "fi_features.npz"
    atomic_json_dump(
        {
            "method": method,
            "channel_indices": fi_channel_indices,
            "channel_names": metrics["fi_channel_names"],
            "locomotor_band_hz": [0.5, 3.0],
            "locomotor_high_exclusive": True,
            "freeze_band_hz": [3.0, 8.0],
            "freeze_high_inclusive": True,
            "aggregation": args.fi_aggregation,
            "squared_ratio": bool(args.fi_squared_ratio),
            "score_threshold": threshold,
            "raw_fi_threshold": raw_threshold,
            "power_gate_enabled": bool(args.fi_power_gate),
            "power_threshold": power_threshold,
            "power_train_quantile": power_quantile,
            "threshold_selection": "validation_balanced_accuracy",
        },
        rule_path,
    )
    atomic_npz_save(
        feature_path,
        window_index=test["window_index"],
        channel_locomotor_power=test["channel_locomotor_power"],
        channel_freeze_power=test["channel_freeze_power"],
        locomotor_power=test["locomotor_power"],
        freeze_power=test["freeze_power"],
        total_power=test["total_power"],
        fi_raw=test["freeze_index"],
        fi_score=test["score"],
        gated_score=test_score,
    )
    print(
        f"  [{test_subject}] {method}: "
        f"BA={metrics['balanced_accuracy']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f}",
        flush=True,
    )
    return finalise_method_artifacts(
        root=root,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        metrics=metrics,
        validation_indices=validation["window_index"],
        validation_true=val_true,
        validation_prob=val_score,
        validation_pred=val_pred,
        test_indices=test["window_index"],
        test_true=test_true,
        test_prob=test_score,
        test_pred=test_pred,
        dataset=dataset,
        windows=windows,
        additional_artifacts={"rule": rule_path, "features": feature_path},
    )


def _extract_tf_features(
    *,
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    history_samples: int,
    scaler: RobustChannelScaler,
    channel_indices: tuple[int, ...],
    extractor: TimeFrequencyFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories, labels, indices = materialize_history_windows(
        dataset.records,
        windows,
        window_indices,
        history_samples,
        scaler=scaler,
    )
    histories = np.ascontiguousarray(histories[:, channel_indices])
    features = extractor.transform(histories)
    return features, labels, indices


def _svm_pipeline(
    args: argparse.Namespace,
    c_value: float,
    *,
    probability: bool,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "svm",
                SVC(
                    C=float(c_value),
                    kernel=args.svm_kernel,
                    gamma="scale",
                    class_weight="balanced",
                    probability=probability,
                    cache_size=float(args.svm_cache_mb),
                    max_iter=int(args.svm_max_iter),
                    random_state=int(seed),
                ),
            ),
        ]
    )


def run_tf_svm(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    fold_root: Path,
    test_subject: str,
    val_subject: str,
    dataset: DaphnetDataset,
    windows: WindowTable,
    split_indices: dict[str, np.ndarray],
    scaler: RobustChannelScaler,
    samples: dict[str, int],
    svm_c_grid: list[float],
) -> dict[str, Any]:
    method = "tf_svm"
    task_id = f"{test_subject}/{method}"
    root = fold_root / method
    root.mkdir(parents=True, exist_ok=True)
    if args.resume and method_is_complete(
        root,
        config["protocol_fingerprint"],
        task_id,
    ):
        with (root / "metrics.json").open("r", encoding="utf-8") as handle:
            print(f"  [{test_subject}] {method}: complete", flush=True)
            return json.load(handle)

    started = time.perf_counter()
    channel_indices = SENSOR_SETS[args.sensor_set]
    channel_names = tuple(
        dataset.channel_names[index] for index in channel_indices
    )
    extractor = TimeFrequencyFeatureExtractor(
        dataset.sampling_rate_hz,
        channel_names,
        include_triad_magnitudes=args.feature_magnitudes,
        batch_size=args.feature_batch_size,
    )
    x_train, y_train, _ = _extract_tf_features(
        dataset=dataset,
        windows=windows,
        window_indices=split_indices["train"],
        history_samples=samples["history"],
        scaler=scaler,
        channel_indices=channel_indices,
        extractor=extractor,
    )
    x_val, y_val, val_indices = _extract_tf_features(
        dataset=dataset,
        windows=windows,
        window_indices=split_indices["validation"],
        history_samples=samples["history"],
        scaler=scaler,
        channel_indices=channel_indices,
        extractor=extractor,
    )
    x_test, y_test, test_indices = _extract_tf_features(
        dataset=dataset,
        windows=windows,
        window_indices=split_indices["test"],
        history_samples=samples["history"],
        scaler=scaler,
        channel_indices=channel_indices,
        extractor=extractor,
    )
    if not (
        np.isfinite(x_train).all()
        and np.isfinite(x_val).all()
        and np.isfinite(x_test).all()
    ):
        raise ValueError("Time-frequency features contain non-finite values")

    search_path = root / "search_results.json"
    search_payload: dict[str, Any] = {
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": task_id,
        "selection_metric": "validation_pr_auc",
        "candidates": [],
    }
    if args.resume and search_path.exists():
        with search_path.open("r", encoding="utf-8") as handle:
            search_payload = json.load(handle)
        if (
            search_payload.get("protocol_fingerprint")
            != config["protocol_fingerprint"]
            or search_payload.get("task_id") != task_id
        ):
            raise ValueError(f"Incompatible SVM search state: {search_path}")
    completed = {
        float(candidate["C"]): candidate
        for candidate in search_payload.get("candidates", [])
    }
    method_seed = args.seed + 10000 + dataset.subjects.index(test_subject)
    for c_value in svm_c_grid:
        if float(c_value) in completed:
            continue
        candidate = _svm_pipeline(
            args,
            c_value,
            probability=False,
            seed=method_seed,
        )
        candidate.fit(x_train, y_train)
        decision = np.asarray(candidate.decision_function(x_val), dtype=np.float64)
        row = {
            "C": float(c_value),
            "validation_pr_auc": float(average_precision_score(y_val, decision)),
            "validation_roc_auc": float(roc_auc_score(y_val, decision)),
            "fit_status": int(candidate.named_steps["svm"].fit_status_),
            "n_support": candidate.named_steps["svm"].n_support_.astype(int).tolist(),
        }
        search_payload.setdefault("candidates", []).append(row)
        atomic_json_dump(search_payload, search_path)
        completed[float(c_value)] = row

    best = max(
        completed.values(),
        key=lambda row: (
            float(row["validation_pr_auc"]),
            float(row["validation_roc_auc"]),
            -float(row["C"]),
        ),
    )
    selected_c = float(best["C"])
    model = _svm_pipeline(
        args,
        selected_c,
        probability=True,
        seed=method_seed,
    )
    model.fit(x_train, y_train)
    if int(model.named_steps["svm"].fit_status_) != 0:
        raise RuntimeError(
            f"SVM failed to converge for {test_subject}; "
            "increase --svm-max-iter"
        )
    val_prob = np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float64)
    test_prob = np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)
    metrics, val_pred, test_pred, threshold = common_metrics(
        method=method,
        test_subject=test_subject,
        val_subject=val_subject,
        args=args,
        samples=samples,
        validation_true=y_val,
        validation_prob=val_prob,
        test_true=y_test,
        test_prob=test_prob,
        test_indices=test_indices,
        dataset=dataset,
        windows=windows,
    )
    metrics.update(
        {
            "svm_kernel": args.svm_kernel,
            "selected_c": selected_c,
            "selection_validation_pr_auc": float(
                best["validation_pr_auc"]
            ),
            "selection_validation_roc_auc": float(
                best["validation_roc_auc"]
            ),
            "n_features": int(x_train.shape[1]),
            "train_counts": np.bincount(y_train, minlength=2).astype(int).tolist(),
            "method_seed": method_seed,
            "elapsed_sec": time.perf_counter() - started,
        }
    )
    model_path = root / "model.joblib"
    schema_path = root / "feature_schema.json"
    atomic_joblib_dump(model, model_path)
    atomic_json_dump(
        {
            "version": "daphnet_time_frequency.v1",
            "sampling_rate_hz": dataset.sampling_rate_hz,
            "history_samples": samples["history"],
            "history_seconds": args.input_seconds,
            "raw_scaler": scaler.as_dict(),
            "sensor_set": args.sensor_set,
            "channel_indices": list(channel_indices),
            "channel_names": list(channel_names),
            "include_triad_magnitudes": bool(args.feature_magnitudes),
            "feature_names": list(extractor.feature_names()),
            "feature_count": len(extractor.feature_names()),
            "spectral_estimator": "mean_removed_hann_rfft_one_sided_power",
            "locomotor_band_hz": [0.5, 3.0],
            "locomotor_high_exclusive": True,
            "freeze_band_hz": [3.0, 8.0],
            "freeze_high_inclusive": True,
        },
        schema_path,
    )
    print(
        f"  [{test_subject}] {method}: C={selected_c:g} "
        f"BA={metrics['balanced_accuracy']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f}",
        flush=True,
    )
    return finalise_method_artifacts(
        root=root,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        metrics=metrics,
        validation_indices=val_indices,
        validation_true=y_val,
        validation_prob=val_prob,
        validation_pred=val_pred,
        test_indices=test_indices,
        test_true=y_test,
        test_prob=test_prob,
        test_pred=test_pred,
        dataset=dataset,
        windows=windows,
        additional_artifacts={
            "model": model_path,
            "feature_schema": schema_path,
            "search_results": search_path,
        },
    )


def make_history_loader(
    *,
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    history_samples: int,
    scaler: RobustChannelScaler,
    channel_indices: tuple[int, ...],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    source = HistoryWindowDataset(
        dataset.records,
        windows,
        window_indices,
        history_samples,
        scaler,
        channel_indices,
    )
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Recreate workers at every epoch. Their base seeds are then derived
        # from the restored main-process RNG, preserving epoch-boundary resume.
        persistent_workers=False,
    )


def cnn_epoch(
    model: CNNGRUClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for x, y, window_index in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                if grad_scaler is None:
                    raise RuntimeError("Training requires a gradient scaler")
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch_n = int(y.numel())
        total_loss += float(loss.detach()) * batch_n
        total_n += batch_n
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probabilities.append(
            torch.sigmoid(logits.detach()).float().cpu().numpy()
        )
        indices.append(window_index.numpy().astype(np.int64))
    return (
        total_loss / max(total_n, 1),
        np.concatenate(truths),
        np.concatenate(probabilities),
        np.concatenate(indices),
    )


def run_cnn_gru(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    fold_root: Path,
    test_subject: str,
    val_subject: str,
    dataset: DaphnetDataset,
    windows: WindowTable,
    split_indices: dict[str, np.ndarray],
    scaler: RobustChannelScaler,
    samples: dict[str, int],
    cnn_channels: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    method = "cnn_gru"
    task_id = f"{test_subject}/{method}"
    root = fold_root / method
    root.mkdir(parents=True, exist_ok=True)
    if args.resume and method_is_complete(
        root,
        config["protocol_fingerprint"],
        task_id,
    ):
        with (root / "metrics.json").open("r", encoding="utf-8") as handle:
            print(f"  [{test_subject}] {method}: complete", flush=True)
            return json.load(handle)

    started = time.perf_counter()
    method_seed = args.seed + 10000 + dataset.subjects.index(test_subject)
    set_seed(method_seed, args.deterministic)
    channel_indices = SENSOR_SETS[args.sensor_set]
    pin_memory = device.type == "cuda"
    train_loader = make_history_loader(
        dataset=dataset,
        windows=windows,
        window_indices=split_indices["train"],
        history_samples=samples["history"],
        scaler=scaler,
        channel_indices=channel_indices,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_history_loader(
        dataset=dataset,
        windows=windows,
        window_indices=split_indices["validation"],
        history_samples=samples["history"],
        scaler=scaler,
        channel_indices=channel_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_history_loader(
        dataset=dataset,
        windows=windows,
        window_indices=split_indices["test"],
        history_samples=samples["history"],
        scaler=scaler,
        channel_indices=channel_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model_config = {
        "in_channels": len(channel_indices),
        "cnn_channels": cnn_channels,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
        "bidirectional": args.gru_bidirectional,
    }
    model = CNNGRUClassifier(**model_config).to(device)
    counts = np.bincount(
        windows.label[split_indices["train"]],
        minlength=2,
    ).astype(np.float64)
    pos_weight_value = min(
        math.sqrt(counts[0] / max(counts[1], 1.0)),
        6.0,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    start_epoch = 0
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        validate_checkpoint(
            payload,
            stage="baseline_cnn_gru",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
        )
        if payload.get("model_config") != model_config:
            raise ValueError(f"Incompatible CNN-GRU model config in {last_path}")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        grad_scaler.load_state_dict(payload["grad_scaler_state"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        restore_rng_state(payload["rng_state"])
        print(
            f"  [{test_subject}] {method}: resume epoch {start_epoch + 1}",
            flush=True,
        )

    epoch_started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
        train_loss, train_true, train_prob, _ = cnn_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            val_loss, val_true, val_prob, _ = cnn_epoch(
                model,
                val_loader,
                criterion,
                device,
                args.amp,
            )
        validation_score = float(average_precision_score(val_true, val_prob))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pr_auc": float(
                    average_precision_score(train_true, train_prob)
                ),
                "validation_loss": val_loss,
                "validation_pr_auc": validation_score,
            }
        )
        improved = validation_score > best_score + 1e-5
        if improved:
            best_epoch = epoch
            best_score = validation_score
            bad_epochs = 0
            atomic_torch_save(
                {
                    **checkpoint_base(
                        stage="baseline_cnn_gru",
                        protocol_fingerprint=config["protocol_fingerprint"],
                        task_id=task_id,
                    ),
                    "model_config": model_config,
                    "model_state": model.state_dict(),
                    "method_seed": method_seed,
                    "best_epoch": best_epoch,
                    "best_validation_pr_auc": best_score,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - epoch_started
        atomic_torch_save(
            {
                **checkpoint_base(
                    stage="baseline_cnn_gru",
                    protocol_fingerprint=config["protocol_fingerprint"],
                    task_id=task_id,
                ),
                "model_config": model_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
                "method_seed": method_seed,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_sec": elapsed,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
        print(
            f"  [{test_subject}] {method}: epoch={epoch:02d} "
            f"loss={train_loss:.5f} val_PR-AUC={validation_score:.5f}"
            f"{' *' if improved else ''}",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError(f"CNN-GRU produced no best checkpoint: {task_id}")
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    validate_checkpoint(
        best_payload,
        stage="baseline_cnn_gru",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    model.load_state_dict(best_payload["model_state"])
    with torch.no_grad():
        _, val_true, val_prob, val_indices = cnn_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.amp,
        )
        _, test_true, test_prob, test_indices = cnn_epoch(
            model,
            test_loader,
            criterion,
            device,
            args.amp,
        )
    if not np.array_equal(val_indices, split_indices["validation"]):
        raise AssertionError("CNN-GRU validation loader changed anchor order")
    if not np.array_equal(test_indices, split_indices["test"]):
        raise AssertionError("CNN-GRU test loader changed anchor order")
    metrics, val_pred, test_pred, threshold = common_metrics(
        method=method,
        test_subject=test_subject,
        val_subject=val_subject,
        args=args,
        samples=samples,
        validation_true=val_true,
        validation_prob=val_prob,
        test_true=test_true,
        test_prob=test_prob,
        test_indices=test_indices,
        dataset=dataset,
        windows=windows,
    )
    metrics.update(
        {
            "method_seed": method_seed,
            "model_parameters": parameter_count(model),
            "model_config": {
                **model_config,
                "cnn_channels": list(cnn_channels),
            },
            "best_epoch": int(best_payload["best_epoch"]),
            "best_validation_pr_auc": float(
                best_payload["best_validation_pr_auc"]
            ),
            "train_counts": counts.astype(int).tolist(),
            "pos_weight": float(pos_weight_value),
            "history": history,
            "elapsed_sec": (
                elapsed_before + time.perf_counter() - epoch_started
            ),
        }
    )
    print(
        f"  [{test_subject}] {method}: "
        f"BA={metrics['balanced_accuracy']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f}",
        flush=True,
    )
    return finalise_method_artifacts(
        root=root,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        metrics=metrics,
        validation_indices=val_indices,
        validation_true=val_true,
        validation_prob=val_prob,
        validation_pred=val_pred,
        test_indices=test_indices,
        test_true=test_true,
        test_prob=test_prob,
        test_pred=test_pred,
        dataset=dataset,
        windows=windows,
        additional_artifacts={"best": best_path, "last": last_path},
    )


def refresh_summaries(output_dir: Path, config: dict[str, Any]) -> None:
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    expected_folds = list(config["folds_resolved"])
    for method in config["methods_resolved"]:
        rows: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed: list[str] = []
        for subject in expected_folds:
            root = output_dir / f"loso_{subject}" / method
            metrics_path = root / "metrics.json"
            prediction_path = root / "predictions.npz"
            done_path = root / "DONE.json"
            if not (
                metrics_path.exists()
                and prediction_path.exists()
                and done_path.exists()
            ):
                continue
            validate_done(
                done_path,
                stage="baseline_method",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=f"{subject}/{method}",
            )
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            with np.load(prediction_path, allow_pickle=False) as payload:
                truths.append(np.asarray(payload["y_true"], dtype=np.int8))
                probabilities.append(
                    np.asarray(payload["y_prob"], dtype=np.float64)
                )
                predictions.append(
                    np.asarray(payload["y_pred"], dtype=np.int8)
                )
            rows.append(metrics)
            fold_rows.append(metrics)
            completed.append(subject)
        if rows:
            aggregate[method] = {
                "method": method,
                "completed_folds": completed,
                "subject_macro": aggregate_fold_metrics(
                    rows,
                    list(METRIC_KEYS),
                ),
                "pooled": metrics_from_predictions(
                    np.concatenate(truths),
                    np.concatenate(probabilities),
                    np.concatenate(predictions),
                ),
            }
        manifest_rows.append(
            {
                "experiment_id": method,
                "method": method,
                "input": f"raw_h{config['history_seconds']:g}s",
                "expected_folds": len(expected_folds),
                "completed_folds": len(completed),
                "status": (
                    "complete"
                    if completed == expected_folds
                    else ("partial" if completed else "pending")
                ),
                "completed_subjects": ",".join(completed),
            }
        )

    fold_columns = [
        "experiment_id",
        "method",
        "input",
        "history_seconds",
        "history_samples",
        "test_subject",
        "val_subject",
        "seed",
        "method_seed",
        "threshold",
        "n",
        "n_normal",
        "n_fog",
        *METRIC_KEYS,
        "tn",
        "fp",
        "fn",
        "tp",
        "best_epoch",
        "best_validation_pr_auc",
        "selected_c",
        "selection_validation_pr_auc",
        "n_features",
        "fi_score_threshold",
        "fi_raw_threshold",
        "power_gate_enabled",
        "power_threshold",
        "elapsed_sec",
    ]
    atomic_csv_write(
        output_dir / "fold_summary.csv",
        fold_rows,
        fold_columns,
    )
    atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        [
            "experiment_id",
            "method",
            "input",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    atomic_json_dump(aggregate, output_dir / "aggregate_metrics.json")
    expected_cells = len(expected_folds) * len(config["methods_resolved"])
    completed_cells = sum(
        int(row["completed_folds"]) for row in manifest_rows
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_methods": len(config["methods_resolved"]),
            "expected_folds": len(expected_folds),
            "expected_fold_cells": expected_cells,
            "completed_fold_cells": completed_cells,
            "status": (
                "complete" if completed_cells == expected_cells else "partial"
            ),
        },
        output_dir / "status.json",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    worker_mode = bool(str(args.worker_fold).strip())
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    excluded_subjects = parse_subject_list(args.exclude_subjects)
    if set(excluded_subjects) != {"S04", "S10"}:
        raise ValueError(
            "The core baseline suite requires exactly "
            "--exclude-subjects S04,S10"
        )
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed, args.deterministic)

    data_sha256 = dataset_fingerprint(args.data_dir)
    dataset = DaphnetDataset.load(
        args.data_dir,
        flatline_seconds=args.flatline_seconds,
        zero_tolerance=args.zero_tolerance,
    )
    if dataset.n_channels != 9:
        raise ValueError(
            f"Core three-IMU suite requires 9 channels, got {dataset.n_channels}"
        )
    if tuple(dataset.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError(
            "Expected ordered ankle/thigh/trunk channels, got "
            f"{dataset.channel_names}"
        )
    source_subjects = list(dataset.subjects)
    dataset = filter_dataset(dataset, excluded_subjects)
    if tuple(dataset.subjects) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            "Core suite requires the eight post-exclusion subjects "
            f"{EXPECTED_LOSO_SUBJECTS}, got {tuple(dataset.subjects)}"
        )
    fs = dataset.sampling_rate_hz
    if fs != 64:
        raise ValueError(f"Core Daphnet suite requires 64 Hz, got {fs} Hz")

    sample_values = {
        "context": args.context_seconds * fs,
        "horizon": args.horizon_seconds * fs,
        "stride": args.stride_seconds * fs,
        "history": args.input_seconds * fs,
        "guard": args.normal_guard_seconds * fs,
    }
    for name, value in sample_values.items():
        if not math.isclose(value, round(value), abs_tol=1e-9):
            raise ValueError(f"{name} duration is not an integer sample count")
    samples = {name: int(round(value)) for name, value in sample_values.items()}
    windows = dataset.make_windows(
        warmup_samples=samples["context"],
        target_samples=samples["horizon"],
        stride_samples=samples["stride"],
        fog_fraction_threshold=args.fog_fraction_threshold,
        normal_guard_samples=samples["guard"],
    )
    methods = parse_methods(args.methods)
    folds = parse_folds(args.folds, dataset.subjects)
    if not folds or len(folds) != len(set(folds)):
        raise ValueError("--folds must resolve to unique subjects")
    if worker_mode and tuple(folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            "Parallel worker mode requires --folds all so every worker "
            "shares one canonical protocol"
        )
    execution_folds = list(folds)
    if worker_mode:
        worker_folds = parse_folds(args.worker_fold, dataset.subjects)
        if len(worker_folds) != 1:
            raise ValueError("--worker-fold must resolve to exactly one subject")
        if worker_folds[0] not in folds:
            raise ValueError(
                f"Worker fold {worker_folds[0]} is outside {folds}"
            )
        execution_folds = worker_folds

    global_plan = make_common_history_plan(
        windows,
        np.arange(len(windows), dtype=np.int64),
        samples["horizon"],
        samples["stride"],
        samples["history"],
    )
    # The anchor support is global even when only a subset of outer folds is
    # requested.  Otherwise a one-fold smoke run would accidentally discard
    # the other subjects from its train and validation partitions.
    eligible_indices = global_plan.anchor_window_indices
    fi_channel_indices = parse_fi_channel_indices(
        args.fi_channels,
        dataset.channel_names,
    )
    cnn_channels = parse_positive_ints(
        args.cnn_channels,
        "--cnn-channels",
    )
    svm_c_grid = parse_positive_floats(
        args.svm_c_grid,
        "--svm-c-grid",
    )
    config = build_protocol(
        args,
        dataset,
        source_subjects,
        excluded_subjects,
        folds,
        methods,
        data_sha256,
        windows,
        eligible_indices,
        samples,
        fi_channel_indices,
        cnn_channels,
        svm_c_grid,
        device,
    )
    config_path = args.output_dir / "config.json"
    if worker_mode:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Initialize the suite before workers: missing {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            initialized = json.load(handle)
        if (
            initialized.get("protocol_fingerprint")
            != config["protocol_fingerprint"]
        ):
            raise ValueError(
                "Worker scientific configuration differs from initialized suite"
            )
    else:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                initialized = json.load(handle)
            if (
                initialized.get("protocol_fingerprint")
                != config["protocol_fingerprint"]
            ):
                raise ValueError(
                    "Saved suite belongs to a different scientific protocol"
                )
            # Runtime paths, device and worker count are intentionally outside
            # the scientific fingerprint. Refresh them so a run can be moved
            # or finalized on CPU without invalidating compatible artifacts.
            if initialized != config:
                atomic_json_dump(config, config_path)
        else:
            atomic_json_dump(config, config_path)
        environment_path = args.output_dir / "environment.json"
        if not environment_path.exists():
            atomic_json_dump(environment_payload(device), environment_path)
        save_or_validate_json(
            args.output_dir / "run_manifest.json",
            {
                "suite_version": SUITE_VERSION,
                "protocol_fingerprint": config["protocol_fingerprint"],
                "data_sha256": data_sha256,
                "implementation_sha256": config["implementation"]["sha256"],
                "expected_folds": folds,
                "expected_methods": methods,
                "expected_fold_cells": len(folds) * len(methods),
            },
        )

    if args.finalize_only:
        refresh_summaries(args.output_dir, config)
        print(
            f"[baseline-suite] summaries refreshed: {args.output_dir}",
            flush=True,
        )
        return

    for test_subject in execution_folds:
        print(
            f"[baseline-suite] fold={test_subject} device={device}",
            flush=True,
        )
        (
            fold_root,
            val_subject,
            _,
            scaler,
            split_indices,
        ) = prepare_fold(
            args,
            config,
            dataset,
            windows,
            eligible_indices,
            test_subject,
        )
        if "cnn_gru" in methods:
            run_cnn_gru(
                args=args,
                config=config,
                fold_root=fold_root,
                test_subject=test_subject,
                val_subject=val_subject,
                dataset=dataset,
                windows=windows,
                split_indices=split_indices,
                scaler=scaler,
                samples=samples,
                cnn_channels=cnn_channels,
                device=device,
            )
        if "freeze_index" in methods:
            run_freeze_index(
                args=args,
                config=config,
                fold_root=fold_root,
                test_subject=test_subject,
                val_subject=val_subject,
                dataset=dataset,
                windows=windows,
                split_indices=split_indices,
                samples=samples,
                fi_channel_indices=fi_channel_indices,
            )
        if "tf_svm" in methods:
            run_tf_svm(
                args=args,
                config=config,
                fold_root=fold_root,
                test_subject=test_subject,
                val_subject=val_subject,
                dataset=dataset,
                windows=windows,
                split_indices=split_indices,
                scaler=scaler,
                samples=samples,
                svm_c_grid=svm_c_grid,
            )
        if not worker_mode:
            refresh_summaries(args.output_dir, config)

    if worker_mode:
        print(
            f"[baseline-suite] worker fold complete: {execution_folds[0]}",
            flush=True,
        )
        return
    refresh_summaries(args.output_dir, config)
    print(f"[baseline-suite] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
