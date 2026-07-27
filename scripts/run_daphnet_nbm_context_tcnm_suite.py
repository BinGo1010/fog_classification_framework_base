#!/usr/bin/env python
"""Strict 4-NBM x 4-context Daphnet LOSO suite with a fixed TCN-M readout.

The experiment varies only the normal-behaviour model and the amount of
right-aligned context supplied to it.  Every arm shares a WindowTable built
with the maximum four-second context, the same clean-normal NBM training
windows, the same four-second residual histories, and the same TCN-M
classifier protocol.

Default matrix
--------------

* NBMs: Linear-AR, GRU, TCN, Transformer;
* context: C1=1 s, C2=2 s, C3=3 s, C4=4 s;
* forecast horizon: 0.5 s;
* residual history: 4 s, materialised as ``[batch, 9, 256]``;
* classifier: TCN-M, dilations ``(1, 2, 4, 8, 8, 8)``, RF=125 samples;
* evaluation: eight-fold LOSO after excluding S04 and S10.

The implementation deliberately imports the already-tested NBM and TCN-M
training primitives without modifying either completed historical suite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# Required by deterministic CUDA matrix multiplication.  It must be set before
# torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_daphnet_3imu_nbm_suite as core
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.histories import HistoryPlan, make_history_input
from cnbr_fog.nbm import NormalBehaviourModel
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)


SUITE_VERSION = "daphnet_nbm4_context4_h4_tcnm_loso.v1"
EXPECTED_CHANNEL_NAMES = core.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = core.EXPECTED_LOSO_SUBJECTS
DEFAULT_NBMS = ("linear_ar", "gru", "tcn", "transformer")
CONTEXT_DEFINITIONS = (
    {"context_id": "C1", "context_seconds": 1.0},
    {"context_id": "C2", "context_seconds": 2.0},
    {"context_id": "C3", "context_seconds": 3.0},
    {"context_id": "C4", "context_seconds": 4.0},
)
HISTORY_NAME = "residual_h4s"
HISTORY_SECONDS = 4.0
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
KERNEL_SIZE = 3
CONVOLUTIONS_PER_BLOCK = 2

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "daphnet_nbm4_context4_h4_tcnm_loso_seed42"
)

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_nbm_context_tcnm_suite.py",
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/__init__.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/resume.py",
)

CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "specificity",
    "precision",
    "fog_f1",
    "mcc",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
)

PUBLICATION_METRICS = (
    ("pr_auc", "PR-AUC"),
    ("balanced_accuracy", "BA"),
    ("macro_f1", "Macro-F1"),
    ("roc_auc", "AUROC"),
    ("fog_recall", "FoG Sensitivity/Recall"),
    ("specificity", "Specificity"),
    ("precision", "FoG Precision"),
    ("fog_f1", "FoG F1"),
    ("event_sensitivity", "Event Sensitivity"),
    ("false_alarm_events_per_hour", "FA/h"),
    ("median_detection_delay_sec", "Median Detection Delay"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daphnet 4-NBM x 4-context residual_h4s LOSO suite with a fixed "
            "TCN-M classifier"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help="Execute exactly one fold while retaining the shared 8-fold protocol",
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--exclude-subjects", default="S04,S10")
    parser.add_argument("--nbms", default=",".join(DEFAULT_NBMS))
    parser.add_argument("--context-seconds", default="1,2,3,4")
    parser.add_argument("--support-context-seconds", type=float, default=4.0)
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.25)
    parser.add_argument("--history-seconds", type=float, default=HISTORY_SECONDS)
    parser.add_argument("--normal-guard-seconds", type=float, default=0.5)
    parser.add_argument("--fog-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--flatline-seconds", type=float, default=1.0)
    parser.add_argument("--zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--robust-clip", type=float, default=12.0)
    parser.add_argument("--residual-clip", type=float, default=12.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nbm-hidden", type=int, default=48)
    parser.add_argument("--nbm-dropout", type=float, default=0.1)
    parser.add_argument("--linear-ar-seconds", type=float, default=0.5)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ffn", type=int, default=128)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)

    parser.add_argument("--normal-epochs", type=int, default=8)
    parser.add_argument("--normal-patience", type=int, default=3)
    parser.add_argument("--normal-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-normal-windows", type=int, default=30000)
    parser.add_argument("--max-classifier-windows", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-residuals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-protocol-subset",
        action="store_true",
        help=(
            "Development-only: allow a local subset of NBMs, contexts, or "
            "folds. Parallel workers and the publication auditor always "
            "require the complete preregistered protocol."
        ),
    )

    # Inert in normal runs; retained for smoke/resume testing.
    parser.add_argument("--stop-after-completed-tasks", type=int, default=0)
    parser.add_argument("--debug-interrupt-nbm-after-epoch", type=int, default=0)
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
    )
    return parser.parse_args()


def parse_contexts(specification: str, sampling_rate_hz: int) -> list[dict[str, Any]]:
    requested: list[float] = []
    for raw in str(specification).split(","):
        if not raw.strip():
            continue
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid context duration: {raw!r}")
        if value not in requested:
            requested.append(value)
    canonical = {
        float(item["context_seconds"]): str(item["context_id"])
        for item in CONTEXT_DEFINITIONS
    }
    unknown = sorted(set(requested) - set(canonical))
    if unknown:
        raise ValueError(
            f"Only the preregistered contexts 1,2,3,4 s are supported: {unknown}"
        )
    result = [
        {
            "context_id": canonical[seconds],
            "context_seconds": seconds,
            "context_samples": int(round(seconds * sampling_rate_hz)),
            "directory": (
                f"context_{canonical[seconds].lower()}_"
                f"{seconds:g}s".replace(".", "p")
            ),
        }
        for seconds in sorted(requested)
    ]
    if not result:
        raise ValueError("At least one context duration is required")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if not args.cache_residuals:
        raise ValueError("This suite requires residual caches for recovery/audit")
    positive_integers = {
        "normal_epochs": args.normal_epochs,
        "normal_patience": args.normal_patience,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "nbm_hidden": args.nbm_hidden,
        "classifier_hidden": args.classifier_hidden,
        "gru_layers": args.gru_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ffn": args.transformer_ffn,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [key for key, value in positive_integers.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These integer options must be positive: {invalid}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_normal_windows < 0 or args.max_classifier_windows < 0:
        raise ValueError("Window caps must be non-negative")
    positive_floats = {
        "support_context_seconds": args.support_context_seconds,
        "horizon_seconds": args.horizon_seconds,
        "stride_seconds": args.stride_seconds,
        "history_seconds": args.history_seconds,
        "flatline_seconds": args.flatline_seconds,
        "robust_clip": args.robust_clip,
        "residual_clip": args.residual_clip,
        "linear_ar_seconds": args.linear_ar_seconds,
        "normal_lr": args.normal_lr,
        "classifier_lr": args.classifier_lr,
    }
    invalid = [
        key
        for key, value in positive_floats.items()
        if not math.isfinite(float(value)) or float(value) <= 0
    ]
    if invalid:
        raise ValueError(f"These numeric options must be positive: {invalid}")
    if not math.isclose(args.support_context_seconds, 4.0):
        raise ValueError("Strict common support requires --support-context-seconds 4")
    if not math.isclose(args.horizon_seconds, 0.5):
        raise ValueError("This protocol fixes --horizon-seconds at 0.5")
    if not math.isclose(args.history_seconds, 4.0):
        raise ValueError("This protocol fixes --history-seconds at 4")
    if args.normal_guard_seconds < 0:
        raise ValueError("--normal-guard-seconds must be non-negative")
    if args.weight_decay < 0 or args.zero_tolerance < 0:
        raise ValueError("--weight-decay and --zero-tolerance must be non-negative")
    if not 0.0 < args.fog_fraction_threshold <= 1.0:
        raise ValueError("--fog-fraction-threshold must be in (0,1]")
    if not 0.0 <= args.nbm_dropout < 1.0:
        raise ValueError("--nbm-dropout must be in [0,1)")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0,1)")


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def environment_payload(device: torch.device) -> dict[str, Any]:
    cuda_devices: list[str] = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "selected_device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": [sys.executable, *sys.argv],
    }


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def context_target_split(
    sequence: torch.Tensor,
    context_samples: int,
    horizon_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the context immediately before the fixed target at sequence end."""

    context_samples = int(context_samples)
    horizon_samples = int(horizon_samples)
    required = context_samples + horizon_samples
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [batch, channel, time]")
    if context_samples <= 0 or horizon_samples <= 0:
        raise ValueError("context_samples and horizon_samples must be positive")
    if int(sequence.shape[-1]) < required:
        raise ValueError(
            f"sequence has {sequence.shape[-1]} samples; {required} required"
        )
    target = sequence[:, :, -horizon_samples:]
    context = sequence[:, :, -required:-horizon_samples]
    if context.shape[-1] != context_samples or target.shape[-1] != horizon_samples:
        raise AssertionError("right-aligned context/target split is malformed")
    return context, target


def right_aligned_normal_epoch(
    model: NormalBehaviourModel,
    loader: DataLoader,
    context_samples: int,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Core NBM epoch adapted to a common maximum-context WindowTable."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context, target = context_target_split(
            sequence,
            context_samples,
            model.horizon,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type,
                enabled=amp and device.type == "cuda",
            ):
                mean, sigma = model(context)
                loss = core.gaussian_nll_sigma(target, mean, sigma)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(sequence.shape[0])
        total_loss += float(loss.detach()) * batch
        total_n += batch
    if total_n == 0:
        raise RuntimeError("NBM DataLoader is empty")
    return total_loss / total_n


@torch.no_grad()
def extract_right_aligned_residual_blocks(
    args: argparse.Namespace,
    model: NormalBehaviourModel,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: Any,
    context_samples: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Forecast the common target and retain standardised residual blocks."""

    loader = core.make_sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    model.eval()
    residuals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    squared_error = 0.0
    absolute_error = 0.0
    sigma_sum = 0.0
    n_values = 0
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context, target = context_target_split(
            sequence,
            context_samples,
            model.horizon,
        )
        with torch.amp.autocast(
            device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            mean, sigma = model(context)
            residual = (target - mean) / sigma
        residual = residual.clamp(-args.residual_clip, args.residual_clip)
        error = (target - mean).float()
        squared_error += float(error.square().sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        sigma_sum += float(sigma.float().sum().cpu())
        n_values += int(error.numel())
        residuals.append(residual.float().cpu().numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())
    if not residuals:
        raise RuntimeError("Residual extraction DataLoader is empty")
    features = {
        "residual": np.ascontiguousarray(
            np.concatenate(residuals).astype(np.float32, copy=False)
        ),
        "y": np.concatenate(labels).astype(np.int8, copy=False),
        "window_index": np.concatenate(window_indices).astype(
            np.int64,
            copy=False,
        ),
    }
    diagnostics = {
        "windows": int(len(features["y"])),
        "class_counts": np.bincount(
            features["y"],
            minlength=2,
        ).astype(int).tolist(),
        "context_samples": int(context_samples),
        "horizon_samples": int(model.horizon),
        "forecast_rmse": math.sqrt(squared_error / max(n_values, 1)),
        "forecast_mae": absolute_error / max(n_values, 1),
        "mean_sigma": sigma_sum / max(n_values, 1),
        "residual_abs_mean": float(
            np.abs(features["residual"].astype(np.float64)).mean()
        ),
        "residual_rms": float(
            np.sqrt(np.mean(features["residual"].astype(np.float64) ** 2))
        ),
    }
    return features, diagnostics


def install_core_context_adapter() -> None:
    """Patch only this process's imported core helpers.

    The historical core runner file and its protocol hash remain untouched.
    Python resolves these helper names at call time, so the tested resumable
    NBM/cache machinery can use the right-aligned common-window semantics.
    """

    core.normal_epoch = right_aligned_normal_epoch
    core.extract_residual_blocks = extract_right_aligned_residual_blocks


def experiment_id(nbm: str, context_id: str, context_seconds: float) -> str:
    duration = f"{context_seconds:g}".replace(".", "p")
    return (
        f"{nbm}__{context_id.lower()}_context{duration}s__"
        f"{HISTORY_NAME}__tcn_m"
    )


def experiment_grid(
    nbms: list[str],
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for nbm in nbms:
        for context in contexts:
            result.append(
                {
                    **context,
                    "nbm": nbm,
                    "experiment_id": experiment_id(
                        nbm,
                        context["context_id"],
                        context["context_seconds"],
                    ),
                }
            )
    return result


def validate_protocol_selection(
    nbms: list[str],
    contexts: list[dict[str, Any]],
    folds: list[str],
    sampling_rate_hz: int,
    allow_subset: bool,
) -> None:
    """Reject accidental non-publication grids before any training starts."""

    if allow_subset:
        return
    expected_contexts = [
        (
            str(item["context_id"]),
            float(item["context_seconds"]),
            int(round(float(item["context_seconds"]) * sampling_rate_hz)),
        )
        for item in CONTEXT_DEFINITIONS
    ]
    actual_contexts = [
        (
            str(item["context_id"]),
            float(item["context_seconds"]),
            int(item["context_samples"]),
        )
        for item in contexts
    ]
    if tuple(nbms) != DEFAULT_NBMS:
        raise ValueError(
            "Strict protocol requires --nbms " + ",".join(DEFAULT_NBMS)
        )
    if actual_contexts != expected_contexts:
        raise ValueError(
            "Strict protocol requires --context-seconds 1,2,3,4"
        )
    if tuple(folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("Strict protocol requires --folds all")


def classifier_architecture(
    args: argparse.Namespace,
    in_channels: int,
    sampling_rate_hz: int,
) -> dict[str, Any]:
    receptive_field = rf.convolutional_receptive_field(
        TCN_M_DILATIONS,
        kernel_size=KERNEL_SIZE,
        convolutions_per_block=CONVOLUTIONS_PER_BLOCK,
    )
    if receptive_field != TCN_M_RF_SAMPLES:
        raise AssertionError("Canonical TCN-M receptive field changed")
    core.set_seed(args.seed, args.deterministic)
    model = rf.build_model(
        in_channels=in_channels,
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=TCN_M_DILATIONS,
    )
    payload = {
        "name": "tcn_m",
        "display_name": "TCN-M",
        "in_channels": int(in_channels),
        "hidden_channels": int(args.classifier_hidden),
        "dropout": float(args.classifier_dropout),
        "kernel_size": KERNEL_SIZE,
        "dilations": list(TCN_M_DILATIONS),
        "n_blocks": len(TCN_M_DILATIONS),
        "convolutions_per_block": CONVOLUTIONS_PER_BLOCK,
        "receptive_field_samples": receptive_field,
        "receptive_field_seconds": receptive_field / float(sampling_rate_hz),
        "parameter_count": rf.parameter_count(model),
        "global_pooling": "mean_and_max_over_full_input",
    }
    del model
    return payload


def build_protocol(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    source_subjects: list[str],
    excluded_subjects: list[str],
    folds: list[str],
    nbms: list[str],
    contexts: list[dict[str, Any]],
    data_sha256: str,
    windows: WindowTable,
    evaluation_indices: np.ndarray,
    support_context_samples: int,
    horizon_samples: int,
    stride_samples: int,
    guard_samples: int,
    history_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    classifier = classifier_architecture(
        args,
        dataset.n_channels,
        dataset.sampling_rate_hz,
    )
    experiments = experiment_grid(nbms, contexts)
    scientific = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channel_names": list(dataset.channel_names),
        "n_channels": dataset.n_channels,
        "source_subjects": source_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects": list(dataset.subjects),
        "folds_resolved": folds,
        "nbms_resolved": nbms,
        "context_variants": contexts,
        "experiments": experiments,
        "support_context_samples": int(support_context_samples),
        "support_context_seconds": (
            support_context_samples / float(dataset.sampling_rate_hz)
        ),
        "context_alignment": "right_aligned_immediately_before_target",
        "horizon_samples": int(horizon_samples),
        "horizon_seconds": horizon_samples / float(dataset.sampling_rate_hz),
        "stride_samples": int(stride_samples),
        "stride_seconds": stride_samples / float(dataset.sampling_rate_hz),
        "history_name": HISTORY_NAME,
        "history_samples": int(history_samples),
        "history_seconds": history_samples / float(dataset.sampling_rate_hz),
        "history_blocks": history_samples // horizon_samples,
        "history_construction": (
            "Eight chronological horizon-spaced 32-sample residual blocks; "
            "blocks do not overlap and labels come from the final block."
        ),
        "normal_support_policy": (
            "All contexts share clean-normal eligibility defined on the "
            "maximum 4 s context plus 0.5 s target and normal guard."
        ),
        "delta_pr_auc_reference": "same_nbm_C2_2s_context",
        "bootstrap_unit": "held_out_subject",
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "normal_guard_samples": int(guard_samples),
        "fog_fraction_threshold": float(args.fog_fraction_threshold),
        "flatline_seconds": float(args.flatline_seconds),
        "zero_tolerance": float(args.zero_tolerance),
        "robust_clip": float(args.robust_clip),
        "residual_clip": float(args.residual_clip),
        "seed": int(args.seed),
        "nbm_hidden": int(args.nbm_hidden),
        "nbm_dropout": float(args.nbm_dropout),
        "linear_ar_seconds": float(args.linear_ar_seconds),
        "linear_ar_effective_context_note": (
            "Linear-AR always consumes its final fixed AR order (default "
            "0.5 s), so C1-C4 form a context-invariant negative control."
        ),
        "gru_layers": int(args.gru_layers),
        "transformer_heads": int(args.transformer_heads),
        "transformer_layers": int(args.transformer_layers),
        "transformer_ffn": int(args.transformer_ffn),
        "classifier": classifier,
        "normal_epochs": int(args.normal_epochs),
        "normal_patience": int(args.normal_patience),
        "normal_lr": float(args.normal_lr),
        "classifier_epochs": int(args.classifier_epochs),
        "classifier_patience": int(args.classifier_patience),
        "classifier_lr": float(args.classifier_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_normal_windows": int(args.max_normal_windows),
        "max_classifier_windows": int(args.max_classifier_windows),
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "cache_residuals": bool(args.cache_residuals),
        "window_count": int(len(windows)),
        "window_class_counts": np.bincount(
            windows.label,
            minlength=2,
        ).astype(int).tolist(),
        "evaluation_windows": int(len(evaluation_indices)),
        "evaluation_window_class_counts": np.bincount(
            windows.label[evaluation_indices],
            minlength=2,
        ).astype(int).tolist(),
        "expected_experiments": len(experiments),
        "expected_fold_cells": len(experiments) * len(folds),
        "expected_nbm_tasks": len(experiments) * len(folds),
        "protocol_scope": (
            "development_subset"
            if args.allow_protocol_subset
            else "strict_4_nbm_x_4_context_x_8_fold"
        ),
    }
    fingerprint = canonical_fingerprint(scientific)
    return {
        **scientific,
        "protocol_fingerprint": fingerprint,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "resume": bool(args.resume),
        "num_workers": int(args.num_workers),
    }


def load_dataset_and_protocol(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    DaphnetDataset,
    WindowTable,
    list[dict[str, Any]],
    list[str],
]:
    data_sha256 = dataset_fingerprint(args.data_dir)
    dataset = DaphnetDataset.load(
        args.data_dir,
        flatline_seconds=args.flatline_seconds,
        zero_tolerance=args.zero_tolerance,
    )
    if dataset.n_channels != 9:
        raise ValueError(f"Expected nine IMU channels, got {dataset.n_channels}")
    if tuple(dataset.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError(
            "Expected ordered ankle/thigh/trunk channels, got "
            f"{dataset.channel_names}"
        )
    source_subjects = list(dataset.subjects)
    excluded_subjects = core.parse_subject_list(args.exclude_subjects)
    if set(excluded_subjects) != {"S04", "S10"}:
        raise ValueError("This suite requires exactly --exclude-subjects S04,S10")
    excluded = set(excluded_subjects)
    dataset = DaphnetDataset(
        root=dataset.root,
        records=[
            record
            for record in dataset.records
            if record.subject_id not in excluded
        ],
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
    )
    if tuple(dataset.subjects) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            f"Expected LOSO subjects {EXPECTED_LOSO_SUBJECTS}, "
            f"got {tuple(dataset.subjects)}"
        )
    fs = int(dataset.sampling_rate_hz)
    if fs != 64:
        raise ValueError(f"Daphnet protocol requires 64 Hz, got {fs}")
    contexts = parse_contexts(args.context_seconds, fs)
    support_context_samples = int(round(args.support_context_seconds * fs))
    if max(item["context_samples"] for item in contexts) > support_context_samples:
        raise ValueError("A context exceeds the maximum common support")
    horizon_samples = int(round(args.horizon_seconds * fs))
    stride_samples = int(round(args.stride_seconds * fs))
    history_samples = int(round(args.history_seconds * fs))
    guard_samples = int(round(args.normal_guard_seconds * fs))
    if history_samples % horizon_samples:
        raise ValueError("Residual history must be divisible by the horizon")
    windows = dataset.make_windows(
        warmup_samples=support_context_samples,
        target_samples=horizon_samples,
        stride_samples=stride_samples,
        fog_fraction_threshold=args.fog_fraction_threshold,
        normal_guard_samples=guard_samples,
    )
    nbms = core.parse_nbms(args.nbms)
    unsupported = sorted(set(nbms) - set(DEFAULT_NBMS))
    if unsupported:
        raise ValueError(
            "This experiment excludes Persistence and supports only "
            f"{DEFAULT_NBMS}; got {unsupported}"
        )
    folds = core.parse_folds(args.folds, dataset.subjects)
    validate_protocol_selection(
        nbms,
        contexts,
        folds,
        fs,
        bool(args.allow_protocol_subset),
    )
    global_plan = core.make_common_history_plan(
        windows,
        np.arange(len(windows), dtype=np.int64),
        horizon_samples,
        stride_samples,
        history_samples,
    )
    fold_records = set(dataset.subject_record_indices(folds).astype(int).tolist())
    evaluation_mask = np.fromiter(
        (
            int(record_index) in fold_records
            for record_index in windows.record_index[
                global_plan.anchor_window_indices
            ]
        ),
        dtype=bool,
        count=len(global_plan.anchor_window_indices),
    )
    evaluation_indices = global_plan.anchor_window_indices[evaluation_mask]
    config = build_protocol(
        args,
        dataset,
        source_subjects,
        excluded_subjects,
        folds,
        nbms,
        contexts,
        data_sha256,
        windows,
        evaluation_indices,
        support_context_samples,
        horizon_samples,
        stride_samples,
        guard_samples,
        history_samples,
        device,
    )
    return config, dataset, windows, contexts, folds


def initialise_or_validate_run(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    worker_mode: bool,
    execution_folds: list[str],
) -> None:
    config_path = args.output_dir / "config.json"
    if worker_mode and not config_path.exists():
        raise RuntimeError(
            "Missing config.json; initialize once with --finalize-only"
        )
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_fingerprint") != config["protocol_fingerprint"]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new output directory"
            )
    if not worker_mode:
        atomic_json_dump(config, config_path)

    runtime_fields = {
        "data_dir",
        "output_dir",
        "device",
        "resume",
        "num_workers",
    }
    run_manifest = {
        key: value for key, value in config.items() if key not in runtime_fields
    }
    manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if not manifest_path.exists():
            raise RuntimeError("Missing run_manifest.json for worker")
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != run_manifest:
            raise ValueError(f"Saved JSON is incompatible: {manifest_path}")
    else:
        core.save_or_validate_json(manifest_path, run_manifest)

    environment = environment_payload(device)
    environment["protocol_fingerprint"] = config["protocol_fingerprint"]
    if worker_mode:
        environment["worker_fold"] = execution_folds[0]
        atomic_json_dump(
            environment,
            args.output_dir
            / "worker_environments"
            / f"loso_{execution_folds[0]}.json",
        )
    else:
        atomic_json_dump(environment, args.output_dir / "environment.json")


def stable_bootstrap_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return int((int(base_seed) + offset) % (2**32))


def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "mean_delta": None,
            "ci_low": None,
            "ci_high": None,
            "n_paired_subjects": 0,
            "bootstrap_samples": int(samples),
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(values),
        size=(int(samples), len(values)),
        endpoint=False,
    )
    bootstrap = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_paired_subjects": int(len(values)),
        "bootstrap_samples": int(samples),
    }


def prediction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    fog_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    nonfog_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "macro_f1": 0.5 * (fog_f1 + nonfog_f1),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else None
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else None
        ),
        "fog_recall": recall,
        "specificity": specificity,
        "precision": precision,
        "fog_f1": fog_f1,
    }


def task_root_for(
    output_dir: Path,
    subject: str,
    experiment: Mapping[str, Any],
) -> Path:
    return (
        output_dir
        / f"loso_{subject}"
        / context_task_directory(experiment, subject)
        / str(experiment["nbm"])
        / HISTORY_NAME
        / "tcn_m"
    )


def nbm_root_for(
    output_dir: Path,
    subject: str,
    experiment: Mapping[str, Any],
) -> Path:
    return (
        output_dir
        / f"loso_{subject}"
        / context_task_directory(experiment, subject)
        / str(experiment["nbm"])
    )


def context_task_directory(
    experiment: Mapping[str, Any],
    subject: str,
) -> str:
    """Return a fold-bound context directory used in NBM/cache task IDs."""

    subject = str(subject).strip().upper()
    if subject not in EXPECTED_LOSO_SUBJECTS:
        raise ValueError(f"Unknown LOSO subject: {subject!r}")
    return f"{experiment['directory']}__loso_{subject.lower()}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _metric_summary_columns() -> list[str]:
    return [
        f"{metric}_{statistic}"
        for metric in CLASSIFICATION_METRICS
        for statistic in ("mean", "std")
    ]


def _format_mean_sd(summary: Mapping[str, Any], metric: str) -> str:
    payload = summary.get(metric, {})
    mean = payload.get("mean") if isinstance(payload, Mapping) else None
    std = payload.get("std") if isinstance(payload, Mapping) else None
    if mean is None or std is None:
        return ""
    precision = 3 if metric in {
        "false_alarm_events_per_hour",
        "median_detection_delay_sec",
    } else 4
    return f"{float(mean):.{precision}f} ± {float(std):.{precision}f}"


def _format_delta(delta: Mapping[str, Any]) -> str:
    if delta.get("mean_delta") is None:
        return ""
    return (
        f"{float(delta['mean_delta']):+.4f} "
        f"[{float(delta['ci_low']):+.4f}, "
        f"{float(delta['ci_high']):+.4f}]"
    )


def refresh_summaries(output_dir: Path, config: dict[str, Any]) -> None:
    """Rebuild root summaries from immutable per-cell artifacts."""

    expected_folds = list(config["folds_resolved"])
    experiments = list(config["experiments"])
    rows_by_experiment: dict[str, dict[str, dict[str, Any]]] = {
        item["experiment_id"]: {} for item in experiments
    }
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []

    loaded: dict[str, dict[str, Any]] = {}
    for item in experiments:
        experiment = str(item["experiment_id"])
        group_rows: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed: list[str] = []
        for subject in expected_folds:
            root = task_root_for(output_dir, subject, item)
            source_root = nbm_root_for(output_dir, subject, item)
            metrics_path = root / "metrics.json"
            predictions_path = root / "predictions.npz"
            done_path = root / "DONE.json"
            complete = core.validate_done(
                done_path,
                stage="rf_classifier",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=f"{subject}/{experiment}",
            )
            if complete is None:
                continue
            expected_classifier_artifacts = {
                "best",
                "last",
                "metrics",
                "predictions",
                "validation_predictions",
                "predictions_csv",
            }
            classifier_artifacts = complete.get("artifacts")
            if (
                not isinstance(classifier_artifacts, Mapping)
                or set(classifier_artifacts) != expected_classifier_artifacts
            ):
                raise ValueError(
                    f"Classifier DONE artifact set mismatch at {root}"
                )
            context_directory = context_task_directory(item, subject)
            nbm_done = core.validate_done(
                source_root / "nbm" / "DONE.json",
                stage="nbm",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=(
                    f"{context_directory}/{item['nbm']}/nbm"
                ),
            )
            if nbm_done is None:
                raise FileNotFoundError(
                    source_root / "nbm" / "DONE.json"
                )
            try:
                nbm_sha256 = str(
                    nbm_done["artifacts"]["best"]["sha256"]
                )
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Malformed NBM DONE manifest at {source_root}"
                ) from error
            residual_done = core.validate_done(
                source_root / "RESIDUAL_CACHE_DONE.json",
                stage="residual_cache",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=(
                    f"{context_directory}/{item['nbm']}/residual_cache"
                ),
                upstream_sha256=nbm_sha256,
            )
            if residual_done is None:
                raise FileNotFoundError(
                    source_root / "RESIDUAL_CACHE_DONE.json"
                )
            try:
                residual_sha256 = str(
                    residual_done["artifacts"]["cache"]["sha256"]
                )
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Malformed residual DONE manifest at {source_root}"
                ) from error
            support_path = (
                output_dir / f"loso_{subject}" / "history_support.npz"
            )
            if not support_path.exists():
                raise FileNotFoundError(support_path)
            support_sha256 = sha256_file(support_path)
            upstream_identity = {
                "source_residual_sha256": residual_sha256,
                "input_support_sha256": support_sha256,
            }
            for key, expected in upstream_identity.items():
                if complete.get(key) != expected:
                    raise ValueError(
                        f"Completed cell upstream mismatch at {root}: "
                        f"{key}={complete.get(key)!r}, expected={expected!r}"
                    )
            metrics = _load_json(metrics_path)
            identity = {
                "experiment_id": experiment,
                "test_subject": subject,
                "nbm": item["nbm"],
                "input": HISTORY_NAME,
            }
            for key, expected in identity.items():
                if metrics.get(key) != expected:
                    raise ValueError(
                        f"Completed cell identity mismatch at {root}: "
                        f"{key}={metrics.get(key)!r}, expected={expected!r}"
                    )
            for key in (
                "source_residual_sha256",
                "input_support_sha256",
            ):
                if metrics.get(key) != upstream_identity[key]:
                    raise ValueError(
                        f"Completed cell provenance mismatch at {root}: {key}"
                    )
            with np.load(predictions_path, allow_pickle=False) as payload:
                required = {"window_index", "y_true", "y_prob", "y_pred"}
                if set(payload.files) != required:
                    raise ValueError(
                        f"Unexpected prediction arrays at {predictions_path}"
                    )
                y_true = np.asarray(payload["y_true"], dtype=np.int8)
                y_prob = np.asarray(payload["y_prob"], dtype=np.float64)
                y_pred = np.asarray(payload["y_pred"], dtype=np.int8)
                window_index = np.asarray(
                    payload["window_index"],
                    dtype=np.int64,
                )
            if not (
                y_true.ndim
                == y_prob.ndim
                == y_pred.ndim
                == window_index.ndim
                == 1
                and len(y_true)
                == len(y_prob)
                == len(y_pred)
                == len(window_index)
            ):
                raise ValueError(
                    f"Misaligned prediction arrays at {predictions_path}"
                )
            if not np.isfinite(y_prob).all():
                raise ValueError(
                    f"Non-finite probabilities at {predictions_path}"
                )
            truths.append(y_true)
            probabilities.append(y_prob)
            predictions.append(y_pred)
            enriched = {
                **metrics,
                "context_id": item["context_id"],
                "context_seconds": item["context_seconds"],
                "context_samples": item["context_samples"],
                "classifier": "tcn_m",
            }
            group_rows.append(enriched)
            fold_rows.append(enriched)
            rows_by_experiment[experiment][subject] = enriched
            completed.append(subject)
        subject_macro = (
            aggregate_fold_metrics(group_rows, list(CLASSIFICATION_METRICS))
            if group_rows
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in CLASSIFICATION_METRICS
            }
        )
        loaded[experiment] = {
            "item": item,
            "completed": completed,
            "subject_macro": subject_macro,
            "pooled": (
                prediction_metrics(
                    np.concatenate(truths),
                    np.concatenate(probabilities),
                    np.concatenate(predictions),
                )
                if truths
                else None
            ),
        }
        manifest_rows.append(
            {
                "experiment_id": experiment,
                "nbm": item["nbm"],
                "context_id": item["context_id"],
                "context_seconds": item["context_seconds"],
                "context_samples": item["context_samples"],
                "horizon_seconds": config["horizon_seconds"],
                "history_seconds": config["history_seconds"],
                "classifier": "tcn_m",
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

    delta_rows: list[dict[str, Any]] = []
    for item in experiments:
        experiment = str(item["experiment_id"])
        reference_item = next(
            (
                candidate
                for candidate in experiments
                if candidate["nbm"] == item["nbm"]
                and candidate["context_id"] == "C2"
            ),
            None,
        )
        reference_id = (
            str(reference_item["experiment_id"])
            if reference_item is not None
            else ""
        )
        common_subjects: list[str] = []
        differences: list[float] = []
        if reference_id:
            for subject in expected_folds:
                current = rows_by_experiment[experiment].get(subject)
                reference = rows_by_experiment[reference_id].get(subject)
                if current is None or reference is None:
                    continue
                current_value = current.get("pr_auc")
                reference_value = reference.get("pr_auc")
                if current_value is None or reference_value is None:
                    continue
                common_subjects.append(subject)
                differences.append(float(current_value) - float(reference_value))
        delta = paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                f"{experiment}__vs__{reference_id}",
            ),
        )
        delta_payload = {
            "experiment_id": experiment,
            "reference_experiment_id": reference_id,
            "reference_definition": "same NBM at C2 (2 s context)",
            "common_subjects": ",".join(common_subjects),
            **delta,
        }
        delta_rows.append(delta_payload)

        content = loaded[experiment]
        subject_macro = content["subject_macro"]
        aggregate[experiment] = {
            **item,
            "classifier": config["classifier"],
            "completed_folds": content["completed"],
            "subject_macro": subject_macro,
            "pooled": content["pooled"],
            "delta_pr_auc_vs_same_nbm_c2": delta_payload,
        }
        numeric = {
            "experiment_id": experiment,
            "nbm": item["nbm"],
            "context_id": item["context_id"],
            "context_seconds": item["context_seconds"],
            "context_samples": item["context_samples"],
            "horizon_seconds": config["horizon_seconds"],
            "history_seconds": config["history_seconds"],
            "history_samples": config["history_samples"],
            "classifier": "tcn_m",
            "classifier_receptive_field_samples": config["classifier"][
                "receptive_field_samples"
            ],
            "classifier_receptive_field_seconds": config["classifier"][
                "receptive_field_seconds"
            ],
            "classifier_parameter_count": config["classifier"][
                "parameter_count"
            ],
            "completed_folds": len(content["completed"]),
            "delta_pr_auc_reference": reference_id,
            "delta_pr_auc_mean": delta["mean_delta"],
            "delta_pr_auc_ci_low": delta["ci_low"],
            "delta_pr_auc_ci_high": delta["ci_high"],
            "delta_pr_auc_n_paired_subjects": delta["n_paired_subjects"],
        }
        for metric in CLASSIFICATION_METRICS:
            numeric[f"{metric}_mean"] = subject_macro[metric]["mean"]
            numeric[f"{metric}_std"] = subject_macro[metric]["std"]
        summary_rows.append(numeric)

        publication = {
            "NBM": item["nbm"],
            "Context": (
                f"{item['context_id']} ({float(item['context_seconds']):g} s)"
            ),
            "Horizon": f"{float(config['horizon_seconds']):g} s",
            "PR-AUC": _format_mean_sd(subject_macro, "pr_auc"),
            "ΔPR-AUC [95% CI]": _format_delta(delta),
            "BA": _format_mean_sd(subject_macro, "balanced_accuracy"),
            "Macro-F1": _format_mean_sd(subject_macro, "macro_f1"),
            "AUROC": _format_mean_sd(subject_macro, "roc_auc"),
            "FoG Sensitivity/Recall": _format_mean_sd(
                subject_macro,
                "fog_recall",
            ),
            "Specificity": _format_mean_sd(subject_macro, "specificity"),
            "FoG Precision": _format_mean_sd(subject_macro, "precision"),
            "FoG F1": _format_mean_sd(subject_macro, "fog_f1"),
            "Event Sensitivity": _format_mean_sd(
                subject_macro,
                "event_sensitivity",
            ),
            "FA/h": _format_mean_sd(
                subject_macro,
                "false_alarm_events_per_hour",
            ),
            "Median Detection Delay": _format_mean_sd(
                subject_macro,
                "median_detection_delay_sec",
            ),
            "Completed folds": len(content["completed"]),
        }
        publication_rows.append(publication)

    summary_rows.sort(
        key=lambda row: (
            -float(row["pr_auc_mean"])
            if row["pr_auc_mean"] is not None
            else float("inf"),
            row["experiment_id"],
        )
    )
    ranking_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(summary_rows, start=1)
    ]

    fold_columns = [
        "experiment_id",
        "nbm",
        "context_id",
        "context_seconds",
        "context_samples",
        "classifier",
        "input",
        "history_seconds",
        "history_samples",
        "history_blocks",
        "test_subject",
        "val_subject",
        "classifier_seed",
        "threshold",
        "n",
        "n_normal",
        "n_fog",
        *CLASSIFICATION_METRICS,
        "tn",
        "fp",
        "fn",
        "tp",
        "best_epoch",
        "best_validation_auprc",
        "source_residual_sha256",
        "input_support_sha256",
    ]
    core.atomic_csv_write(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    core.atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        [
            "experiment_id",
            "nbm",
            "context_id",
            "context_seconds",
            "context_samples",
            "horizon_seconds",
            "history_seconds",
            "classifier",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    summary_columns = [
        "experiment_id",
        "nbm",
        "context_id",
        "context_seconds",
        "context_samples",
        "horizon_seconds",
        "history_seconds",
        "history_samples",
        "classifier",
        "classifier_receptive_field_samples",
        "classifier_receptive_field_seconds",
        "classifier_parameter_count",
        "completed_folds",
        "delta_pr_auc_reference",
        "delta_pr_auc_mean",
        "delta_pr_auc_ci_low",
        "delta_pr_auc_ci_high",
        "delta_pr_auc_n_paired_subjects",
        *_metric_summary_columns(),
    ]
    core.atomic_csv_write(
        output_dir / "aggregate_summary.csv",
        ranking_rows,
        ["rank", *summary_columns],
    )
    core.atomic_csv_write(
        output_dir / "paired_pr_auc_deltas.csv",
        delta_rows,
        [
            "experiment_id",
            "reference_experiment_id",
            "reference_definition",
            "common_subjects",
            "mean_delta",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "bootstrap_samples",
        ],
    )
    publication_columns = [
        "NBM",
        "Context",
        "Horizon",
        "PR-AUC",
        "ΔPR-AUC [95% CI]",
        "BA",
        "Macro-F1",
        "AUROC",
        "FoG Sensitivity/Recall",
        "Specificity",
        "FoG Precision",
        "FoG F1",
        "Event Sensitivity",
        "FA/h",
        "Median Detection Delay",
        "Completed folds",
    ]
    core.atomic_csv_write(
        output_dir / "publication_table.csv",
        publication_rows,
        publication_columns,
    )
    aggregate_payload = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "aggregation_unit": "held_out_subject",
        "metric_dispersion": "population standard deviation across LOSO folds",
        "delta_pr_auc": {
            "reference": "same NBM at C2 (2 s context)",
            "method": (
                "paired nonparametric bootstrap over held-out subjects"
            ),
            "confidence_level": 0.95,
            "samples": config["bootstrap_samples"],
            "seed": config["bootstrap_seed"],
        },
        "experiments": aggregate,
        "ranking_metric": "subject_macro_pr_auc_mean",
        "best_experiment": (
            ranking_rows[0]["experiment_id"]
            if ranking_rows and ranking_rows[0]["pr_auc_mean"] is not None
            else None
        ),
    }
    atomic_json_dump(aggregate_payload, output_dir / "aggregate_metrics.json")
    completed_cells = sum(int(row["completed_folds"]) for row in manifest_rows)
    expected_cells = len(experiments) * len(expected_folds)
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_experiments": len(experiments),
            "expected_nbm_tasks": expected_cells,
            "expected_classifier_cells": expected_cells,
            "completed_classifier_cells": completed_cells,
            "status": (
                "complete" if completed_cells == expected_cells else "partial"
            ),
            "best_experiment": aggregate_payload["best_experiment"],
        },
        output_dir / "status.json",
    )


def fold_classifier_config(
    args: argparse.Namespace,
    config: dict[str, Any],
    fold_index: int,
    support_sha256: str,
    test_subject: str,
    val_subject: str,
) -> dict[str, Any]:
    classifier_seed = int(args.seed + 10000 + fold_index)
    core.set_seed(classifier_seed, args.deterministic)
    reference_model = rf.build_model(
        in_channels=int(config["n_channels"]),
        hidden_channels=int(args.classifier_hidden),
        dropout=float(args.classifier_dropout),
        dilations=TCN_M_DILATIONS,
    )
    initial_hash = rf.state_dict_sha256(reference_model.state_dict())
    parameter_count = rf.parameter_count(reference_model)
    del reference_model
    if parameter_count != int(config["classifier"]["parameter_count"]):
        raise AssertionError("Fold TCN-M parameter count changed")
    return {
        "test_subject": test_subject,
        "val_subject": val_subject,
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256": initial_hash,
        "input_support_sha256": support_sha256,
    }


def run_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    contexts: list[dict[str, Any]],
    test_subject: str,
    device: torch.device,
) -> int:
    histories = [
        (
            HISTORY_NAME,
            float(config["history_seconds"]),
            int(config["history_samples"]),
        )
    ]
    (
        fold_root,
        val_subject,
        train_subjects,
        scaler,
        split_indices,
        plans,
        normal_train_indices,
        normal_val_indices,
    ) = core.prepare_fold(
        args,
        config,
        dataset,
        windows,
        test_subject,
        histories,
    )
    fold_index = list(dataset.subjects).index(test_subject)
    support_path = fold_root / "history_support.npz"
    support_sha256 = sha256_file(support_path)
    classifier_fold = fold_classifier_config(
        args,
        config,
        fold_index,
        support_sha256,
        test_subject,
        val_subject,
    )
    suite_fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "common_support_context_samples": config["support_context_samples"],
        "target_boundary_samples": config["support_context_samples"],
        "horizon_samples": config["horizon_samples"],
        "history_support_sha256": support_sha256,
        "classifier_seed": classifier_fold["classifier_seed"],
        "classifier_initial_state_sha256": classifier_fold[
            "reference_initial_state_sha256"
        ],
        "history_anchor_counts": {
            split: int(len(plan.anchor_rows))
            for split, plan in plans.items()
        },
        "split_window_counts": {
            split: int(len(indices))
            for split, indices in split_indices.items()
        },
        "normal_train_windows": int(len(normal_train_indices)),
        "normal_validation_windows": int(len(normal_val_indices)),
    }
    core.save_or_validate_json(
        fold_root / "context_suite_fold_config.json",
        suite_fold_config,
    )
    print(
        f"[fold {test_subject}] train={train_subjects} val={val_subject} "
        f"normal={len(normal_train_indices)}/{len(normal_val_indices)} "
        f"anchors={{{', '.join(f'{k}:{len(v.anchor_rows)}' for k, v in plans.items())}}}",
        flush=True,
    )

    nbms = list(config["nbms_resolved"])
    completed = 0
    for context in contexts:
        context_samples = int(context["context_samples"])
        for nbm_name in nbms:
            item = {
                **context,
                "nbm": nbm_name,
                "experiment_id": experiment_id(
                    nbm_name,
                    context["context_id"],
                    context["context_seconds"],
                ),
            }
            nbm_root = nbm_root_for(args.output_dir, test_subject, item)
            nbm_root.mkdir(parents=True, exist_ok=True)
            model, normal_training, nbm_sha256 = core.train_nbm_resumable(
                args,
                nbm_name,
                nbm_root,
                config["protocol_fingerprint"],
                args.seed + fold_index,
                dataset,
                windows,
                normal_train_indices,
                normal_val_indices,
                scaler,
                context_samples,
                int(config["horizon_samples"]),
                device,
            )
            features, residual_diagnostics = (
                core.load_or_extract_residual_cache(
                    args,
                    nbm_root,
                    config["protocol_fingerprint"],
                    nbm_name,
                    nbm_sha256,
                    model,
                    dataset,
                    windows,
                    split_indices,
                    scaler,
                    context_samples,
                    device,
                )
            )
            atomic_json_dump(
                {
                    "protocol_fingerprint": config["protocol_fingerprint"],
                    "experiment_id": item["experiment_id"],
                    "nbm": nbm_name,
                    "context_id": context["context_id"],
                    "context_seconds": context["context_seconds"],
                    "context_samples": context_samples,
                    "context_start_rule": (
                        "target_start - context_samples; right aligned"
                    ),
                    "nbm_sha256": nbm_sha256,
                    "normal_training": normal_training,
                    "residual_diagnostics": residual_diagnostics,
                },
                nbm_root / "nbm_summary.json",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            inputs = {
                split: make_history_input(
                    features[split],
                    plans[split],
                    HISTORY_NAME,
                    int(config["history_samples"]),
                    int(config["horizon_samples"]),
                    int(config["stride_samples"]),
                )
                for split in ("train", "validation", "test")
            }
            for split, payload in inputs.items():
                expected_shape = (
                    len(plans[split].anchor_rows),
                    dataset.n_channels,
                    int(config["history_samples"]),
                )
                if tuple(payload[HISTORY_NAME].shape) != expected_shape:
                    raise AssertionError(
                        f"Unexpected {split} input shape "
                        f"{payload[HISTORY_NAME].shape} != {expected_shape}"
                    )
            residual_cache_path = nbm_root / "residual_cache.npz"
            residual_sha256 = sha256_file(residual_cache_path)
            variant = {
                "variant": item["experiment_id"],
                "display_name": (
                    f"{nbm_name} {context['context_id']} + TCN-M"
                ),
                "experiment_id": item["experiment_id"],
                "dilations": list(TCN_M_DILATIONS),
                "receptive_field_samples": TCN_M_RF_SAMPLES,
                "receptive_field_seconds": (
                    TCN_M_RF_SAMPLES / float(dataset.sampling_rate_hz)
                ),
            }
            rf_fold_config = {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "classifier_seed": classifier_fold["classifier_seed"],
                "reference_initial_state_sha256": classifier_fold[
                    "reference_initial_state_sha256"
                ],
                "source": {
                    "source_residual_cache_sha256": residual_sha256,
                    "input_support_sha256": support_sha256,
                },
            }
            rf_config = {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "shared_parameter_count": config["classifier"][
                    "parameter_count"
                ],
            }
            # The RF trainer records SOURCE_NBM in metrics.  Set it explicitly
            # for this sequential cell before invoking the shared primitive.
            rf.SOURCE_NBM = nbm_name
            metrics = rf.train_classifier_resumable(
                args,
                rf_config,
                variant,
                task_root_for(args.output_dir, test_subject, item),
                rf_fold_config,
                inputs,
                dataset,
                windows,
                device,
            )
            print(
                f"[fold {test_subject}] {item['experiment_id']} "
                f"PR-AUC={metrics['pr_auc']:.4f} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"Recall={metrics['fog_recall']:.4f} "
                f"Specificity={metrics['specificity']:.4f}",
                flush=True,
            )
            completed += 1
            del inputs, features
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                args.stop_after_completed_tasks > 0
                and completed >= args.stop_after_completed_tasks
            ):
                raise RuntimeError(
                    "Intentional stop after completed classifier tasks"
                )
    return completed


def main() -> None:
    args = parse_args()
    validate_args(args)
    worker_mode = bool(str(args.worker_fold).strip())
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if paths_overlap(args.output_dir, args.data_dir):
        raise ValueError("Output directory must not overlap the processed data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = core.resolve_device(args.device)
    core.set_seed(args.seed, args.deterministic)
    install_core_context_adapter()

    config, dataset, windows, contexts, folds = load_dataset_and_protocol(
        args,
        device,
    )
    if worker_mode:
        canonical_contexts = [
            (item["context_id"], item["context_seconds"])
            for item in CONTEXT_DEFINITIONS
        ]
        actual_contexts = [
            (item["context_id"], item["context_seconds"])
            for item in contexts
        ]
        if (
            tuple(folds) != EXPECTED_LOSO_SUBJECTS
            or tuple(config["nbms_resolved"]) != DEFAULT_NBMS
            or actual_contexts != canonical_contexts
            or config["protocol_scope"]
            != "strict_4_nbm_x_4_context_x_8_fold"
        ):
            raise ValueError(
                "Parallel workers require the complete strict 4x4x8 protocol"
            )
    execution_folds = list(folds)
    if worker_mode:
        selected = core.parse_folds(args.worker_fold, dataset.subjects)
        if len(selected) != 1:
            raise ValueError("--worker-fold must resolve to one subject")
        if selected[0] not in folds:
            raise ValueError(f"Worker fold {selected[0]} is outside {folds}")
        execution_folds = selected

    initialise_or_validate_run(
        args,
        config,
        device,
        worker_mode,
        execution_folds,
    )
    if not worker_mode:
        refresh_summaries(args.output_dir, config)
    print(
        f"[INFO] suite={SUITE_VERSION} device={device} "
        f"subjects={dataset.subjects} windows={len(windows)} "
        f"common={config['evaluation_windows']} "
        f"contexts={[item['context_seconds'] for item in contexts]} "
        f"nbms={config['nbms_resolved']} "
        f"classifier=TCN-M/RF{TCN_M_RF_SAMPLES} "
        f"configured_folds={folds} execution_folds={execution_folds}",
        flush=True,
    )

    if args.finalize_only:
        refresh_summaries(args.output_dir, config)
        status = _load_json(args.output_dir / "status.json")
        print("[INFO] finalize-only: root summaries refreshed", flush=True)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        return

    completed = 0
    for subject in execution_folds:
        completed += run_fold(
            args,
            config,
            dataset,
            windows,
            contexts,
            subject,
            device,
        )
        if not worker_mode:
            refresh_summaries(args.output_dir, config)
    if worker_mode:
        print(
            json.dumps(
                {
                    "suite_version": SUITE_VERSION,
                    "protocol_fingerprint": config["protocol_fingerprint"],
                    "worker_fold": execution_folds[0],
                    "nbm_tasks_visited": completed,
                    "classifier_cells_visited": completed,
                    "status": "worker_complete",
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    refresh_summaries(args.output_dir, config)
    print(
        json.dumps(
            _load_json(args.output_dir / "status.json"),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
