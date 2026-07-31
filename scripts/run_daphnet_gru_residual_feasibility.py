#!/usr/bin/env python
"""Four-phase GRU-H200 residual-fusion feasibility experiment for Daphnet.

This runner is deliberately separate from the completed GRU horizon suite.  It
replays the immutable fold-specific H200 checkpoints, materialises the richer
``raw/mu/sigma/error/z/log_sigma`` primitive cache, and drives the exploratory
Phase 0--2 protocol.  Phase 3A/3B are exposed through the same CLI and delegate
to the cross-fitting hooks in :mod:`cnbr_fog.h200_feasibility`.

The source suite is never modified.  Every generated artifact is bound to a
protocol fingerprint and is written under a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_3imu_nbm_suite as nbm_runner
import run_daphnet_gru_horizon_ablation as horizon_source
import run_daphnet_tcn_rf_ablation as rf_runner
from cnbr_fog.data import DaphnetDataset, RobustChannelScaler, WindowTable
from cnbr_fog.evaluation import binary_metrics, choose_threshold
from cnbr_fog.h200_feasibility import (
    ARM_SPECS,
    H200_ARM_REGISTRY,
    build_arm_inputs,
    build_classifier,
    calibrate_persistence_sigma,
    derive_forecast_primitives,
    evaluate_phase2_gate,
    forecast_diagnostics,
    paired_bootstrap,
    persistence_forecast_diagnostics,
)
from cnbr_fog.h200_phase0_visuals import render_phase0_visualizations
from cnbr_fog.nbm import build_nbm
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    capture_rng_state,
    canonical_fingerprint,
    dataset_fingerprint,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_done,
)


SUITE_VERSION = "daphnet_gru_h200_residual_feasibility.v1"
SOURCE_SUITE_VERSION = "daphnet_gru_horizon4_h4_tcnm_loso.v1"
SOURCE_HORIZON_ID = "H200"
SOURCE_HORIZON_DIR = "horizon_h200_2s"
SOURCE_NBM = "gru"

SAMPLING_RATE_HZ = 64
CONTEXT_SAMPLES = 128
HORIZON_SAMPLES = 128
HISTORY_SAMPLES = 256
RAW6_SAMPLES = 384
STRIDE_SAMPLES = 16
LABEL_SAMPLES = 32
Z_CLIP = 12.0
EXPECTED_SUBJECTS = tuple(horizon_source.EXPECTED_LOSO_SUBJECTS)
EXPECTED_CHANNEL_NAMES = tuple(horizon_source.EXPECTED_CHANNEL_NAMES)

PHASES = ("0", "1", "2", "3a", "3b")
PHASE1_ARMS = (
    "raw4",
    "normality",
    "raw4_zero",
    "raw4_normality",
)
PHASE2_ARMS = tuple(ARM_SPECS)
PHASE3_ARMS = ("raw6", "raw4_zero", "raw4_normality")

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DEFAULT_SOURCE_SUITE_DIR = (
    REPO_ROOT / "outputs" / "daphnet_gru_horizon4_h4_tcnm_loso_seed42"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "daphnet_gru_h200_residual_feasibility_seed42"
)

METRIC_NAMES = (
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


@dataclass(frozen=True)
class ProtocolContext:
    config: dict[str, Any]
    dataset: DaphnetDataset
    master_windows: WindowTable
    windows_by_horizon: dict[str, WindowTable]
    classification_windows: WindowTable
    folds: tuple[str, ...]


@dataclass(frozen=True)
class FoldContext:
    subject: str
    val_subject: str
    train_subjects: tuple[str, ...]
    scaler: RobustChannelScaler
    split_indices: dict[str, np.ndarray]
    support: dict[str, np.ndarray]
    source_fold_config: dict[str, Any]
    source_best_path: Path
    source_best_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daphnet GRU-H200 normal forecast, residual diagnostics, "
            "Raw/normality fusion, and subject-cross-fitted confirmation"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--phase",
        choices=(*PHASES, "all"),
        default="0",
        help="Run one phase or the complete staged protocol",
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument("--source-suite-dir", type=Path, default=DEFAULT_SOURCE_SUITE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--force-next-phase",
        action="store_true",
        help=(
            "Continue an all-phase run after a failed/non-reportable stage gate; "
            "the override is recorded in status artifacts"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--phase1-train-windows", type=int, default=10_000)
    parser.add_argument("--phase1-eval-windows", type=int, default=5_000)
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument(
        "--phase0-diagnostic-max-windows",
        type=int,
        default=512,
        help="Fixed first-N windows used for lag and band-power diagnostics",
    )
    parser.add_argument(
        "--phase0-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render deterministic waist-sensor Phase-0 diagnostic figures",
    )
    parser.add_argument("--phase0-plot-windows", type=int, default=5)
    parser.add_argument("--phase0-plot-dpi", type=int, default=160)
    parser.add_argument(
        "--max-classifier-windows",
        type=int,
        default=0,
        help="Phase-2 training-only cap; zero uses every common anchor",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--phase3-nbm-seeds", default="42")
    parser.add_argument(
        "--phase3-classifier-seeds",
        default=None,
        help="Comma-separated; default is 42 for 3A and 42,43,44 for 3B",
    )
    parser.add_argument("--phase3-normal-epochs", type=int, default=None)
    parser.add_argument("--phase3-normal-patience", type=int, default=None)
    parser.add_argument("--phase3-normal-lr", type=float, default=None)
    parser.add_argument("--phase3-max-normal-windows", type=int, default=30_000)
    parser.add_argument(
        "--phase3-normal-validation-fraction", type=float, default=0.2
    )
    parser.add_argument("--phase3-min-z-std-ratio", type=float, default=0.5)
    parser.add_argument("--phase3-max-z-std-ratio", type=float, default=2.0)
    parser.add_argument(
        "--phase3-max-log-sigma-shift", type=float, default=math.log(2.0)
    )
    parser.add_argument(
        "--phase3-max-z-clip-rate-difference", type=float, default=0.05
    )
    parser.add_argument(
        "--force-phase3-representation-gate",
        action="store_true",
        help="Continue after the OOF/ensemble representation hard gate fails",
    )
    parser.add_argument(
        "--phase3-external-negative-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate frozen Phase-3B models on negative-only S04/S10",
    )
    parser.add_argument("--phase3-external-batch-size", type=int, default=None)
    parser.add_argument(
        "--phase3-external-minimum-positive-windows", type=int, default=2
    )
    parser.add_argument(
        "--phase3-external-merge-gap-seconds", type=float, default=0.5
    )
    parser.add_argument("--debug-interrupt-nbm-after-epoch", type=int, default=0)
    parser.add_argument(
        "--cache-compressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress primitive NPZ files; disable for faster high-I/O runs",
    )
    parser.add_argument(
        "--stop-after-tasks",
        type=int,
        default=0,
        help="Debug/resume hook; stop after this many newly completed tasks",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "classifier_hidden": args.classifier_hidden,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "phase1_epochs": args.phase1_epochs,
        "phase0_diagnostic_max_windows": args.phase0_diagnostic_max_windows,
        "phase0_plot_windows": args.phase0_plot_windows,
        "phase0_plot_dpi": args.phase0_plot_dpi,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "phase3_max_normal_windows": args.phase3_max_normal_windows,
        "phase3_external_minimum_positive_windows": (
            args.phase3_external_minimum_positive_windows
        ),
    }
    invalid = [name for name, value in positive.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {invalid}")
    nonnegative = {
        "num_workers": args.num_workers,
        "phase1_train_windows": args.phase1_train_windows,
        "phase1_eval_windows": args.phase1_eval_windows,
        "max_classifier_windows": args.max_classifier_windows,
        "stop_after_tasks": args.stop_after_tasks,
        "debug_interrupt_nbm_after_epoch": args.debug_interrupt_nbm_after_epoch,
        "weight_decay": args.weight_decay,
    }
    invalid = [name for name, value in nonnegative.items() if float(value) < 0]
    if invalid:
        raise ValueError(f"These options must be non-negative: {invalid}")
    if not 0.0 <= float(args.classifier_dropout) < 1.0:
        raise ValueError("--classifier-dropout must be in [0,1)")
    if not math.isfinite(args.classifier_lr) or args.classifier_lr <= 0:
        raise ValueError("--classifier-lr must be finite and positive")
    for name in ("phase3_nbm_seeds", "phase3_classifier_seeds"):
        value = getattr(args, name)
        if value is None:
            continue
        try:
            seeds = tuple(
                int(item.strip()) for item in str(value).split(",") if item.strip()
            )
        except ValueError as error:
            raise ValueError(f"--{name.replace('_', '-')} must contain integers") from error
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"--{name.replace('_', '-')} must be non-empty and unique")
    for name in ("phase3_normal_epochs", "phase3_normal_patience"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.phase3_normal_lr is not None and (
        not math.isfinite(args.phase3_normal_lr) or args.phase3_normal_lr <= 0
    ):
        raise ValueError("--phase3-normal-lr must be finite and positive")
    if args.phase3_external_batch_size is not None and (
        args.phase3_external_batch_size <= 0
    ):
        raise ValueError("--phase3-external-batch-size must be positive")
    if (
        not math.isfinite(args.phase3_external_merge_gap_seconds)
        or args.phase3_external_merge_gap_seconds < 0
    ):
        raise ValueError(
            "--phase3-external-merge-gap-seconds must be finite and non-negative"
        )
    if not 0.0 < args.phase3_normal_validation_fraction < 1.0:
        raise ValueError("--phase3-normal-validation-fraction must be in (0,1)")
    if not (
        0.0 < args.phase3_min_z_std_ratio <= 1.0
        <= args.phase3_max_z_std_ratio
    ):
        raise ValueError("Phase-3 z-std ratio bounds must straddle one")
    if args.phase3_max_log_sigma_shift < 0:
        raise ValueError("--phase3-max-log-sigma-shift must be non-negative")
    if not 0.0 <= args.phase3_max_z_clip_rate_difference <= 1.0:
        raise ValueError(
            "--phase3-max-z-clip-rate-difference must be in [0,1]"
        )
    if args.finalize_only and not args.resume:
        raise ValueError("--finalize-only requires --resume")
    if args.output_dir.resolve() == args.source_suite_dir.resolve():
        raise ValueError("Output directory must differ from the immutable source suite")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _resolve_artifact(done_path: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = done_path.parent / path
    return path.resolve()


def _resolve_device(specification: str) -> torch.device:
    specification = str(specification).strip().lower()
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _parse_folds(specification: str) -> tuple[str, ...]:
    folds = tuple(nbm_runner.parse_folds(specification, EXPECTED_SUBJECTS))
    if not folds:
        raise ValueError("At least one fold is required")
    return folds


def _phase_sequence(phase: str) -> tuple[str, ...]:
    return PHASES if phase == "all" else (phase,)


def _csv_write(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_h200_root(source_suite_dir: Path, subject: str) -> Path:
    return (
        source_suite_dir
        / f"loso_{subject}"
        / SOURCE_HORIZON_DIR
        / SOURCE_NBM
    )


def _validate_source_checkpoint(
    source_suite_dir: Path,
    source_fingerprint: str,
    subject: str,
) -> tuple[Path, str]:
    root = _source_h200_root(source_suite_dir, subject)
    done_path = root / "nbm" / "DONE.json"
    completed = validate_done(
        done_path,
        stage="nbm",
        protocol_fingerprint=source_fingerprint,
        task_id=f"{SOURCE_HORIZON_DIR}/{SOURCE_NBM}/nbm",
    )
    if completed is None:
        raise FileNotFoundError(done_path)
    best_entry = completed["artifacts"]["best"]
    best_path = _resolve_artifact(done_path, best_entry)
    observed = sha256_file(best_path)
    expected = str(best_entry["sha256"])
    if observed != expected:
        raise ValueError(f"Source H200 checkpoint hash changed: {subject}")
    return best_path, expected


def build_protocol(args: argparse.Namespace, device: torch.device) -> ProtocolContext:
    source_config = _load_json(args.source_suite_dir / "config.json")
    source_status = _load_json(args.source_suite_dir / "status.json")
    if source_config.get("suite_version") != SOURCE_SUITE_VERSION:
        raise ValueError("Unexpected source suite version")
    if source_status.get("status") != "complete":
        raise ValueError("Source GRU horizon suite is not complete")
    required = {
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "context_samples": CONTEXT_SAMPLES,
        "support_horizon_samples": HORIZON_SAMPLES,
        "fixed_label_samples": LABEL_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "residual_clip": Z_CLIP,
        "seed": 42,
    }
    for key, expected in required.items():
        if source_config.get(key) != expected:
            raise ValueError(
                f"Source {key}={source_config.get(key)!r}; expected {expected!r}"
            )
    if tuple(source_config.get("folds_resolved", [])) != EXPECTED_SUBJECTS:
        raise ValueError("Source LOSO subject order changed")
    h200 = [
        item
        for item in source_config.get("horizon_variants", [])
        if item.get("horizon_id") == SOURCE_HORIZON_ID
    ]
    if len(h200) != 1 or int(h200[0].get("horizon_samples", -1)) != HORIZON_SAMPLES:
        raise ValueError("Source suite lacks the required H200 definition")

    data_sha = dataset_fingerprint(args.data_dir)
    if data_sha != source_config.get("data_sha256"):
        raise ValueError("Current processed dataset differs from the source suite")
    loaded = DaphnetDataset.load(
        args.data_dir,
        flatline_seconds=float(source_config["flatline_seconds"]),
        zero_tolerance=float(source_config["zero_tolerance"]),
    )
    dataset = DaphnetDataset(
        root=loaded.root,
        records=[
            record
            for record in loaded.records
            if record.subject_id not in {"S04", "S10"}
        ],
        sampling_rate_hz=loaded.sampling_rate_hz,
        channel_names=loaded.channel_names,
    )
    if tuple(dataset.subjects) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected subjects {EXPECTED_SUBJECTS}, got {dataset.subjects}")
    if tuple(dataset.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError("Daphnet channel ordering differs from the source suite")

    raw_master = dataset.make_windows(
        warmup_samples=CONTEXT_SAMPLES,
        target_samples=HORIZON_SAMPLES,
        stride_samples=STRIDE_SAMPLES,
        fog_fraction_threshold=float(source_config["fog_fraction_threshold"]),
        normal_guard_samples=int(source_config["normal_guard_samples"]),
    )
    master = horizon_source.relabel_master_windows(
        dataset,
        raw_master,
        LABEL_SAMPLES,
        float(source_config["fog_fraction_threshold"]),
    )
    horizons = [dict(item) for item in horizon_source.HORIZON_DEFINITIONS]
    windows_by_horizon = {
        str(item["horizon_id"]): horizon_source.derive_horizon_windows(
            master,
            int(item["horizon_samples"]),
            LABEL_SAMPLES,
        )
        for item in horizons
    }
    classification_windows = horizon_source.derive_classification_windows(master)
    if horizon_source.window_table_sha256(master) != source_config["master_window_sha256"]:
        raise ValueError("Rebuilt master WindowTable differs from source")
    if (
        horizon_source.window_table_sha256(classification_windows)
        != source_config["classification_window_sha256"]
    ):
        raise ValueError("Rebuilt classification WindowTable differs from source")
    for horizon_id, windows in windows_by_horizon.items():
        if (
            horizon_source.window_table_sha256(windows)
            != source_config["derived_window_sha256"][horizon_id]
        ):
            raise ValueError(f"Rebuilt {horizon_id} WindowTable differs from source")

    source_checkpoints: dict[str, dict[str, Any]] = {}
    for subject in EXPECTED_SUBJECTS:
        path, digest = _validate_source_checkpoint(
            args.source_suite_dir,
            str(source_config["protocol_fingerprint"]),
            subject,
        )
        source_checkpoints[subject] = {
            "best_path": str(path),
            "best_sha256": digest,
            "fold_config_sha256": sha256_file(
                args.source_suite_dir / f"loso_{subject}" / "fold_config.json"
            ),
            "split_indices_sha256": sha256_file(
                args.source_suite_dir / f"loso_{subject}" / "split_indices.npz"
            ),
            "support_sha256": sha256_file(
                args.source_suite_dir
                / f"loso_{subject}"
                / "common_history_support.npz"
            ),
        }

    folds = _parse_folds(args.folds)
    scientific = {
        "suite_version": SUITE_VERSION,
        "source_suite_version": SOURCE_SUITE_VERSION,
        "source_protocol_fingerprint": source_config["protocol_fingerprint"],
        "source_checkpoints": source_checkpoints,
        "data_sha256": data_sha,
        "subjects": list(EXPECTED_SUBJECTS),
        "folds_resolved": list(folds),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "channel_names": list(EXPECTED_CHANNEL_NAMES),
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "history_samples": HISTORY_SAMPLES,
        "raw6_samples": RAW6_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "label_samples": LABEL_SAMPLES,
        "z_clip": Z_CLIP,
        "arms": {
            name: {
                "display_name": spec.display_name,
                "classifier_kind": spec.classifier_kind,
                "raw_channels": spec.raw_channels,
                "normality_channels": spec.normality_channels,
                "input_samples": spec.input_samples,
                "normality_source": spec.normality_source,
            }
            for name, spec in H200_ARM_REGISTRY.items()
        },
        "phase1_arms": list(PHASE1_ARMS),
        "phase2_arms": list(PHASE2_ARMS),
        "phase3_arms": list(PHASE3_ARMS),
        "seed": int(args.seed),
        "classifier_hidden": int(args.classifier_hidden),
        "classifier_dropout": float(args.classifier_dropout),
        "classifier_epochs": int(args.classifier_epochs),
        "classifier_patience": int(args.classifier_patience),
        "classifier_lr": float(args.classifier_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "phase1_train_windows": int(args.phase1_train_windows),
        "phase1_eval_windows": int(args.phase1_eval_windows),
        "phase1_epochs": int(args.phase1_epochs),
        "phase0_diagnostic_max_windows": int(
            args.phase0_diagnostic_max_windows
        ),
        "phase0_plots": bool(args.phase0_plots),
        "phase0_plot_windows": int(args.phase0_plot_windows),
        "phase0_plot_dpi": int(args.phase0_plot_dpi),
        "max_classifier_windows": int(args.max_classifier_windows),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "phase3_nbm_seeds": str(args.phase3_nbm_seeds),
        "phase3_classifier_seed_policy": (
            {"3a": [42], "3b": [42, 43, 44]}
            if args.phase3_classifier_seeds is None
            else {
                "3a": [
                    int(value.strip())
                    for value in args.phase3_classifier_seeds.split(",")
                    if value.strip()
                ],
                "3b": [
                    int(value.strip())
                    for value in args.phase3_classifier_seeds.split(",")
                    if value.strip()
                ],
            }
        ),
        "phase3_normal_epochs": int(
            args.phase3_normal_epochs
            if args.phase3_normal_epochs is not None
            else source_config.get("normal_epochs", 8)
        ),
        "phase3_normal_patience": int(
            args.phase3_normal_patience
            if args.phase3_normal_patience is not None
            else source_config.get("normal_patience", 3)
        ),
        "phase3_normal_lr": float(
            args.phase3_normal_lr
            if args.phase3_normal_lr is not None
            else source_config.get("normal_lr", 1e-3)
        ),
        "phase3_max_normal_windows": int(args.phase3_max_normal_windows),
        "phase3_normal_validation_fraction": float(
            args.phase3_normal_validation_fraction
        ),
        "phase3_representation_gate": {
            "min_z_std_ratio": float(args.phase3_min_z_std_ratio),
            "max_z_std_ratio": float(args.phase3_max_z_std_ratio),
            "max_log_sigma_shift": float(args.phase3_max_log_sigma_shift),
            "max_z_clip_rate_difference": float(
                args.phase3_max_z_clip_rate_difference
            ),
        },
        "phase3_external_negative_only": bool(
            args.phase3_external_negative_only
        ),
        "phase3_external_subjects": ["S04", "S10"],
        "phase3_external_event_definition": {
            "minimum_positive_windows": int(
                args.phase3_external_minimum_positive_windows
            ),
            "merge_gap_seconds": float(
                args.phase3_external_merge_gap_seconds
            ),
        },
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "phase2_is_exploratory": True,
        "phase2_train_residual_policy": "in_sample_outer_train",
        "phase3_train_residual_policy": "subject_level_cross_fitted",
    }
    fingerprint = canonical_fingerprint(scientific)
    config = {
        **scientific,
        "protocol_fingerprint": fingerprint,
        "data_dir": str(args.data_dir.resolve()),
        "source_suite_dir": str(args.source_suite_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "resume": bool(args.resume),
        "force_next_phase": bool(args.force_next_phase),
        "force_phase3_representation_gate": bool(
            args.force_phase3_representation_gate
        ),
        "phase3_external_batch_size": args.phase3_external_batch_size,
    }
    return ProtocolContext(
        config=config,
        dataset=dataset,
        master_windows=master,
        windows_by_horizon=windows_by_horizon,
        classification_windows=classification_windows,
        folds=folds,
    )


def initialise_run(args: argparse.Namespace, protocol: ProtocolContext) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        existing = _load_json(config_path)
        if existing.get("protocol_fingerprint") != protocol.config["protocol_fingerprint"]:
            raise ValueError(
                "Output directory belongs to another protocol; use a new directory"
            )
    else:
        if any(args.output_dir.iterdir()) and not args.resume:
            raise FileExistsError(
                f"{args.output_dir} is non-empty; use --resume or another directory"
            )
        atomic_json_dump(protocol.config, config_path)
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": protocol.config["protocol_fingerprint"],
            "phases": list(PHASES),
            "folds": list(protocol.folds),
            "status": "running",
            "force_next_phase": bool(args.force_next_phase),
        },
        args.output_dir / "status.json",
    )


def _load_scaler(path: Path) -> RobustChannelScaler:
    payload = _load_json(path)
    return RobustChannelScaler(
        center=np.asarray(payload["center"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
        clip=float(payload["clip"]),
    )


def load_fold_context(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    subject: str,
) -> FoldContext:
    source_fold_root = args.source_suite_dir / f"loso_{subject}"
    source_fold_config = _load_json(source_fold_root / "fold_config.json")
    if source_fold_config.get("test_subject") != subject:
        raise ValueError(f"Source fold identity mismatch: {subject}")
    val_subject = str(source_fold_config["val_subject"])
    train_subjects = tuple(str(value) for value in source_fold_config["train_subjects"])
    if (
        len(train_subjects) != 6
        or set(train_subjects) & {subject, val_subject}
        or set(train_subjects) | {subject, val_subject} != set(EXPECTED_SUBJECTS)
    ):
        raise ValueError(f"Invalid 6/1/1 source split for {subject}")

    scaler = _load_scaler(source_fold_root / "scaler.json")
    recomputed = protocol.dataset.fit_scaler(train_subjects, clip=scaler.clip)
    np.testing.assert_array_equal(recomputed.center, scaler.center)
    np.testing.assert_array_equal(recomputed.scale, scaler.scale)

    with np.load(source_fold_root / "split_indices.npz", allow_pickle=False) as payload:
        split_indices = {
            split: np.asarray(payload[f"{split}_window_index"], dtype=np.int64)
            for split in ("train", "validation", "test")
        }
    expected_splits = {
        "train": protocol.dataset.window_indices_for_subjects(
            protocol.classification_windows, train_subjects
        ),
        "validation": protocol.dataset.window_indices_for_subjects(
            protocol.classification_windows, [val_subject]
        ),
        "test": protocol.dataset.window_indices_for_subjects(
            protocol.classification_windows, [subject]
        ),
    }
    for split in expected_splits:
        if not np.array_equal(split_indices[split], expected_splits[split]):
            raise ValueError(f"Rebuilt {subject}/{split} indices differ from source")

    plans = horizon_source.build_common_history_support(
        protocol.windows_by_horizon,
        split_indices,
        HISTORY_SAMPLES,
        STRIDE_SAMPLES,
    )
    with np.load(
        source_fold_root / "common_history_support.npz", allow_pickle=False
    ) as payload:
        support = {key: np.asarray(payload[key]) for key in payload.files}
    for split in ("train", "validation", "test"):
        plan = plans[SOURCE_HORIZON_ID][split]
        expected_anchor = plan.anchor_window_indices
        expected_history = split_indices[split][plan.max_chain_rows]
        if not np.array_equal(support[f"{split}_anchor_window_index"], expected_anchor):
            raise ValueError(f"Rebuilt {subject}/{split} anchors differ from source")
        if not np.array_equal(
            support[f"{split}_h200_history_window_index"], expected_history
        ):
            raise ValueError(f"Rebuilt {subject}/{split} H200 history differs")
        expected_y = protocol.classification_windows.label[expected_anchor]
        if not np.array_equal(support[f"{split}_y"], expected_y):
            raise ValueError(f"Rebuilt {subject}/{split} labels differ from source")

    best_path, best_sha = _validate_source_checkpoint(
        args.source_suite_dir,
        str(protocol.config["source_protocol_fingerprint"]),
        subject,
    )
    return FoldContext(
        subject=subject,
        val_subject=val_subject,
        train_subjects=train_subjects,
        scaler=scaler,
        split_indices=split_indices,
        support=support,
        source_fold_config=source_fold_config,
        source_best_path=best_path,
        source_best_sha256=best_sha,
    )


def _reconstruct_source_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(payload["model_config"])
    expected = {
        "name": SOURCE_NBM,
        "in_channels": len(EXPECTED_CHANNEL_NAMES),
        "horizon": HORIZON_SAMPLES,
    }
    for key, value in expected.items():
        if model_config.get(key) != value:
            raise ValueError(f"Unexpected source model config: {model_config}")
    model = build_nbm(
        SOURCE_NBM,
        in_channels=int(model_config["in_channels"]),
        horizon=int(model_config["horizon"]),
        hidden_channels=int(model_config["hidden_channels"]),
        dropout=float(model_config["dropout"]),
        gru_layers=int(model_config["num_layers"]),
    )
    if model.model_config() != model_config:
        raise ValueError("Reconstructed GRU-H200 architecture differs from checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model.to(device)


def _primitive_root(output_dir: Path, subject: str) -> Path:
    return output_dir / f"loso_{subject}" / "h200_primitives"


def _primitive_path(output_dir: Path, subject: str, split: str) -> Path:
    return _primitive_root(output_dir, subject) / f"{split}_primitives.npz"


@torch.no_grad()
def _extract_split_primitives(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    fold: FoldContext,
    split: str,
    model: nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    indices = fold.split_indices[split]
    loader = nbm_runner.make_sequence_loader(
        protocol.dataset,
        protocol.master_windows,
        indices,
        fold.scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in ("raw", "mu", "sigma", "error", "z", "log_sigma")
    }
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    for sequence, y, window_index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        target = sequence[:, :, CONTEXT_SAMPLES:].float()
        with torch.amp.autocast(
            device.type, enabled=args.amp and device.type == "cuda"
        ):
            mean, sigma = model(context)
        derived = derive_forecast_primitives(
            target.cpu().numpy().astype(np.float32, copy=False),
            mean.float().cpu().numpy().astype(np.float32, copy=False),
            sigma.float().cpu().numpy().astype(np.float32, copy=False),
            z_clip=Z_CLIP,
        )
        chunks["raw"].append(derived["raw"])
        chunks["mu"].append(derived["mean"])
        for key in ("sigma", "error", "z", "log_sigma"):
            chunks[key].append(derived[key])
        labels.append(y.numpy().astype(np.int8, copy=False))
        window_indices.append(window_index.numpy().astype(np.int64, copy=False))
    if not labels:
        raise RuntimeError(f"Empty primitive loader: {fold.subject}/{split}")
    result = {
        key: np.ascontiguousarray(np.concatenate(values), dtype=np.float32)
        for key, values in chunks.items()
    }
    result["y"] = np.concatenate(labels).astype(np.int8, copy=False)
    result["window_index"] = np.concatenate(window_indices).astype(
        np.int64, copy=False
    )
    if not np.array_equal(result["window_index"], indices):
        raise ValueError(f"Primitive replay changed {fold.subject}/{split} order")
    expected_y = protocol.master_windows.label[indices]
    if not np.array_equal(result["y"], expected_y):
        raise ValueError(f"Primitive replay changed {fold.subject}/{split} labels")
    np.testing.assert_allclose(
        result["error"], result["raw"] - result["mu"], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        result["z"],
        np.clip(result["error"] / result["sigma"], -Z_CLIP, Z_CLIP),
        rtol=2e-4,
        atol=2e-4,
    )
    return result


def ensure_primitive_cache(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    fold: FoldContext,
    device: torch.device,
) -> dict[str, Path]:
    root = _primitive_root(args.output_dir, fold.subject)
    root.mkdir(parents=True, exist_ok=True)
    done_path = root / "DONE.json"
    task_id = f"loso_{fold.subject}/h200_primitives"
    complete = validate_done(
        done_path,
        stage="h200_primitive_cache",
        protocol_fingerprint=protocol.config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=fold.source_best_sha256,
    )
    paths = {
        split: _primitive_path(args.output_dir, fold.subject, split)
        for split in ("train", "validation", "test")
    }
    if complete is not None:
        return paths
    if args.finalize_only:
        raise FileNotFoundError(f"Primitive task is incomplete: {task_id}")

    model = _reconstruct_source_model(fold.source_best_path, device)
    diagnostics: dict[str, Any] = {}
    for split, path in paths.items():
        values = _extract_split_primitives(
            args, protocol, fold, split, model, device
        )
        atomic_npz_save(path, compressed=args.cache_compressed, **values)
        diagnostics[split] = forecast_diagnostics(
            values["raw"],
            values["mu"],
            values["sigma"],
            diagnostic_max_windows=args.phase0_diagnostic_max_windows,
        )
        diagnostics[split]["class_counts"] = np.bincount(
            values["y"], minlength=2
        ).astype(int).tolist()
        del values
    diagnostics_path = root / "diagnostics.json"
    atomic_json_dump(diagnostics, diagnostics_path)
    atomic_json_dump(
        done_payload(
            stage="h200_primitive_cache",
            protocol_fingerprint=protocol.config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=fold.source_best_sha256,
            relative_to=root,
            artifacts={
                **{f"{split}_cache": path for split, path in paths.items()},
                "diagnostics": diagnostics_path,
            },
        ),
        done_path,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def _load_primitives(path: Path) -> dict[str, np.ndarray]:
    expected = {"raw", "mu", "sigma", "error", "z", "log_sigma", "y", "window_index"}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError(f"Unexpected primitive keys: {path}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _extract_context_target(
    protocol: ProtocolContext,
    fold: FoldContext,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context = np.empty((len(indices), 9, CONTEXT_SAMPLES), dtype=np.float32)
    target = np.empty((len(indices), 9, HORIZON_SAMPLES), dtype=np.float32)
    for row, window_index in enumerate(np.asarray(indices, dtype=np.int64)):
        record_index = int(protocol.master_windows.record_index[window_index])
        record = protocol.dataset.records[record_index]
        start = int(protocol.master_windows.start[window_index])
        target_start = int(protocol.master_windows.target_start[window_index])
        target_end = int(protocol.master_windows.target_end[window_index])
        transformed = fold.scaler.transform(record.x[start:target_end])
        context[row] = transformed[:CONTEXT_SAMPLES].T
        target[row] = transformed[CONTEXT_SAMPLES:].T
    return context, target


def _phase0_task_root(output_dir: Path, subject: str) -> Path:
    return output_dir / "phase0" / f"loso_{subject}"


def run_phase0_fold(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    fold: FoldContext,
    primitive_paths: Mapping[str, Path],
) -> dict[str, Any]:
    root = _phase0_task_root(args.output_dir, fold.subject)
    root.mkdir(parents=True, exist_ok=True)
    task_id = f"phase0/loso_{fold.subject}"
    done_path = root / "DONE.json"
    complete = validate_done(
        done_path,
        stage="phase0_diagnostics",
        protocol_fingerprint=protocol.config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=fold.source_best_sha256,
    )
    metrics_path = root / "metrics.json"
    if complete is not None:
        return _load_json(metrics_path)
    if args.finalize_only:
        raise FileNotFoundError(metrics_path)

    train = _load_primitives(primitive_paths["train"])
    validation = _load_primitives(primitive_paths["validation"])
    train_clean = protocol.master_windows.clean_normal[train["window_index"]]
    validation_clean = protocol.master_windows.clean_normal[
        validation["window_index"]
    ]
    if not train_clean.any() or not validation_clean.any():
        raise RuntimeError(f"Clean-normal Phase-0 split is empty: {fold.subject}")

    train_context, train_target = _extract_context_target(
        protocol, fold, train["window_index"][train_clean]
    )
    validation_context, validation_target = _extract_context_target(
        protocol, fold, validation["window_index"][validation_clean]
    )
    persistence_sigma = calibrate_persistence_sigma(train_context, train_target)
    persistence = persistence_forecast_diagnostics(
        validation_context, validation_target, persistence_sigma
    )
    gru = forecast_diagnostics(
        validation["raw"][validation_clean],
        validation["mu"][validation_clean],
        validation["sigma"][validation_clean],
        diagnostic_max_windows=args.phase0_diagnostic_max_windows,
    )
    validation_indices = validation["window_index"]
    expected_labels, _ = horizon_source.fixed_endpoint_labels(
        protocol.dataset,
        protocol.master_windows,
        LABEL_SAMPLES,
        0.5,
    )
    identity_checks = {
        "context_end_equals_target_start": bool(
            np.all(
                protocol.master_windows.start[validation_indices]
                + CONTEXT_SAMPLES
                == protocol.master_windows.target_start[validation_indices]
            )
        ),
        "target_has_128_samples": bool(
            np.all(
                protocol.master_windows.target_end[validation_indices]
                - protocol.master_windows.target_start[validation_indices]
                == HORIZON_SAMPLES
            )
        ),
        "endpoint_label_uses_last_32_samples": bool(
            np.array_equal(validation["y"], expected_labels[validation_indices])
        ),
        "clean_mask_has_no_fog_in_guarded_support": bool(
            np.all(protocol.master_windows.clean_normal[validation_indices[validation_clean]])
        ),
        "all_primitives_finite": bool(
            all(
                np.isfinite(validation[key]).all()
                for key in ("raw", "mu", "sigma", "error", "z", "log_sigma")
            )
        ),
        "primitive_shapes_match": bool(
            len(
                {
                    tuple(validation[key].shape)
                    for key in ("raw", "mu", "sigma", "error", "z", "log_sigma")
                }
            )
            == 1
        ),
        "error_identity": bool(
            np.allclose(
                validation["error"],
                validation["raw"] - validation["mu"],
                rtol=1e-5,
                atol=1e-5,
            )
        ),
        "z_identity": bool(
            np.allclose(
                validation["z"],
                np.clip(
                    validation["error"] / validation["sigma"],
                    -Z_CLIP,
                    Z_CLIP,
                ),
                rtol=2e-4,
                atol=2e-4,
            )
        ),
        "window_ids_preserved": bool(
            np.array_equal(validation["window_index"], fold.split_indices["validation"])
        ),
    }
    if not all(identity_checks.values()):
        failed = [name for name, passed in identity_checks.items() if not passed]
        raise AssertionError(f"Phase-0 identity checks failed: {fold.subject}/{failed}")
    clean_error = validation["error"][validation_clean]
    clean_sigma = validation["sigma"][validation_clean]
    clean_unclipped_z = clean_error / clean_sigma
    z_clip_rate = float(np.mean(np.abs(clean_unclipped_z) > Z_CLIP))
    near_rmse = float(gru["lead_quartiles"][0]["rmse"])
    far_rmse = float(gru["lead_quartiles"][-1]["rmse"])
    far_to_near_rmse_ratio = far_rmse / max(near_rmse, 1e-6)
    channel_rmse = np.asarray(
        [row["rmse"] for row in gru["per_channel"]], dtype=np.float64
    )
    max_to_median_channel_rmse_ratio = float(
        channel_rmse.max() / max(float(np.median(channel_rmse)), 1e-6)
    )
    hard_checks = {
        "identities": bool(all(identity_checks.values())),
        "clean_nonfog_z_clip_rate_below_5pct": z_clip_rate < 0.05,
        "last_quartile_no_numeric_explosion": far_to_near_rmse_ratio <= 5.0,
    }
    diagnostic_warnings = {
        "large_mean_absolute_lag": bool(
            gru["cross_correlation"]["mean_absolute_lag_samples"] > 8.0
            and gru["cross_correlation"]["mean_peak_correlation"] >= 0.5
        ),
        "single_channel_error_concentration": bool(
            max_to_median_channel_rmse_ratio > 5.0
        ),
        "lag_search_boundary_concentration": bool(
            gru["cross_correlation"].get("boundary_lag_fraction", 0.0) > 0.25
        ),
    }
    metrics = {
        "subject": fold.subject,
        "val_subject": fold.val_subject,
        "train_subjects": list(fold.train_subjects),
        "clean_train_windows": int(train_clean.sum()),
        "clean_validation_windows": int(validation_clean.sum()),
        "persistence_sigma_fit_subjects": list(fold.train_subjects),
        "persistence_sigma_sha256": _array_sha256(persistence_sigma),
        "gru": gru,
        "persistence": persistence,
        "gru_minus_persistence_validation": {
            "nll": float(gru["overall"]["nll"] - persistence["overall"]["nll"]),
            "rmse": float(gru["overall"]["rmse"] - persistence["overall"]["rmse"]),
            "mae": float(gru["overall"]["mae"] - persistence["overall"]["mae"]),
        },
        "gru_better_rmse": bool(
            gru["overall"]["rmse"] < persistence["overall"]["rmse"]
        ),
        "identity_checks": identity_checks,
        "hard_checks": hard_checks,
        "diagnostic_warnings": diagnostic_warnings,
        "clean_nonfog_z_clip_rate": z_clip_rate,
        "far_to_near_rmse_ratio": float(far_to_near_rmse_ratio),
        "max_to_median_channel_rmse_ratio": (
            max_to_median_channel_rmse_ratio
        ),
    }
    del train, train_context, train_target, validation_context, validation_target
    visualization_artifacts: dict[str, Path] = {}
    if args.phase0_plots:
        test = _load_primitives(primitive_paths["test"])
        visualization_root = root / "figures"
        visualization_manifest = render_phase0_visualizations(
            protocol.dataset,
            protocol.master_windows,
            test,
            visualization_root,
            per_group=args.phase0_plot_windows,
            z_clip=Z_CLIP,
            dpi=args.phase0_plot_dpi,
        )
        insufficient = {
            group: count
            for group, count in visualization_manifest["selected_counts"].items()
            if int(count) < int(args.phase0_plot_windows)
        }
        if insufficient:
            raise RuntimeError(
                f"Phase-0 deterministic figure groups lack windows: {insufficient}"
            )
        manifest_path = visualization_root / "selection_manifest.json"
        visualization_artifacts["figure_manifest"] = manifest_path
        for group, selections in visualization_manifest["selections"].items():
            for selection in selections:
                key = f"figure_{group}_{int(selection['selection_rank']):02d}"
                visualization_artifacts[key] = (
                    visualization_root / str(selection["figure_path"])
                )
        metrics["visualizations"] = {
            "enabled": True,
            "manifest": manifest_path.relative_to(root).as_posix(),
            "selected_counts": visualization_manifest["selected_counts"],
            "trunk_channel_names": visualization_manifest[
                "trunk_channel_names"
            ],
        }
        del test
    else:
        metrics["visualizations"] = {"enabled": False}
    sigma_path = root / "persistence_sigma.npz"
    atomic_npz_save(sigma_path, sigma=persistence_sigma)
    atomic_json_dump(metrics, metrics_path)
    atomic_json_dump(
        done_payload(
            stage="phase0_diagnostics",
            protocol_fingerprint=protocol.config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=fold.source_best_sha256,
            relative_to=root,
            artifacts={
                "metrics": metrics_path,
                "persistence_sigma": sigma_path,
                **visualization_artifacts,
            },
        ),
        done_path,
    )
    return metrics


def _history_rows(
    primitive_window_index: np.ndarray,
    history_window_index: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(window_index): row
        for row, window_index in enumerate(
            np.asarray(primitive_window_index, dtype=np.int64)
        )
    }
    try:
        return np.asarray(
            [[lookup[int(value)] for value in chain] for chain in history_window_index],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("History support references a missing primitive row") from error


def _block_history(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[rows]
    if selected.shape[1:] != (2, 9, HORIZON_SAMPLES):
        raise ValueError(f"Unexpected H200 block history shape: {selected.shape}")
    return np.ascontiguousarray(
        selected.transpose(0, 2, 1, 3).reshape(len(rows), 9, HISTORY_SAMPLES),
        dtype=np.float32,
    )


def _raw6_history(
    protocol: ProtocolContext,
    fold: FoldContext,
    anchor_indices: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(anchor_indices), 9, RAW6_SAMPLES), dtype=np.float32)
    for row, window_index in enumerate(np.asarray(anchor_indices, dtype=np.int64)):
        record_index = int(protocol.classification_windows.record_index[window_index])
        end = int(protocol.classification_windows.target_end[window_index])
        start = end - RAW6_SAMPLES
        if start < 0:
            raise ValueError("Raw6 starts before its record")
        record = protocol.dataset.records[record_index]
        if not bool(record.valid[start:end].all()):
            raise ValueError("Raw6 contains invalid samples")
        result[row] = fold.scaler.transform(record.x[start:end]).T
    return result


def materialize_split_inputs(
    protocol: ProtocolContext,
    fold: FoldContext,
    primitive_path: Path,
    split: str,
) -> dict[str, Any]:
    primitives = _load_primitives(primitive_path)
    history_indices = np.asarray(
        fold.support[f"{split}_h200_history_window_index"], dtype=np.int64
    )
    rows = _history_rows(primitives["window_index"], history_indices)
    raw4 = _block_history(primitives["raw"], rows)
    z4 = _block_history(primitives["z"], rows)
    log_sigma4 = _block_history(primitives["log_sigma"], rows)
    anchors = np.asarray(fold.support[f"{split}_anchor_window_index"], dtype=np.int64)
    y = np.asarray(fold.support[f"{split}_y"], dtype=np.int8)
    raw6 = _raw6_history(protocol, fold, anchors)
    if not (
        len(raw4) == len(z4) == len(log_sigma4) == len(raw6) == len(y) == len(anchors)
    ):
        raise AssertionError("Arm inputs are not endpoint aligned")
    return {
        "raw4": raw4,
        "raw6": raw6,
        "z4": z4,
        "log_sigma4": log_sigma4,
        "y": y,
        "window_index": anchors,
    }


def _subsample_rows(
    rows: np.ndarray,
    labels: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if maximum <= 0 or len(rows) <= maximum:
        return np.asarray(rows, dtype=np.int64)
    return np.asarray(
        nbm_runner.deterministic_subsample(
            np.asarray(rows, dtype=np.int64), maximum, seed, labels
        ),
        dtype=np.int64,
    )


def prepare_arm_inputs(
    base: Mapping[str, Any],
    arm: str,
    rows: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    if arm not in H200_ARM_REGISTRY:
        raise ValueError(f"Unknown arm: {arm}")
    rows = np.asarray(rows, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for start in range(0, len(rows), int(chunk_size)):
        selected = rows[start : start + int(chunk_size)]
        built = build_arm_inputs(
            np.asarray(base["raw4"], dtype=np.float32)[selected],
            np.asarray(base["raw6"], dtype=np.float32)[selected],
            np.asarray(base["z4"], dtype=np.float32)[selected],
            np.asarray(base["log_sigma4"], dtype=np.float32)[selected],
        )
        chunks.append(built[arm])
    if not chunks:
        raise RuntimeError(f"Empty classifier input for arm {arm}")
    arrays = (np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32),)
    return (
        tuple(np.ascontiguousarray(value, dtype=np.float32) for value in arrays),
        np.asarray(base["y"], dtype=np.int8)[rows],
        np.asarray(base["window_index"], dtype=np.int64)[rows],
    )


def _array_loader(
    arrays: tuple[np.ndarray, ...],
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    shuffle_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    tensors = [torch.from_numpy(value).float() for value in arrays]
    tensors.append(torch.from_numpy(np.asarray(y, dtype=np.int64)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        *inputs, y = batch
        inputs = [value.to(device, non_blocking=True) for value in inputs]
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type, enabled=amp and device.type == "cuda"
            ):
                logits = model(*inputs)
                loss = criterion(logits, y)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        total_loss += float(loss.detach()) * int(y.numel())
        total_n += int(y.numel())
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
    if not truths:
        raise RuntimeError("Classifier loader is empty")
    return total_loss / total_n, np.concatenate(truths), np.concatenate(probabilities)


def _state_sha256(model: nn.Module) -> str:
    return rf_runner.state_dict_sha256(model.state_dict())


def _enrich_binary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = (int(metrics[key]) for key in ("tn", "fp", "fn", "tp"))
    nonfog_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    fog_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        **metrics,
        "macro_f1": 0.5 * (nonfog_f1 + fog_f1),
        "roc_auc": metrics.get("auroc"),
        "pr_auc": metrics.get("auprc"),
        "fog_recall": metrics.get("sensitivity"),
        "fog_f1": fog_f1,
    }


def _classifier_task_root(
    output_dir: Path, phase: str, subject: str, arm: str
) -> Path:
    return output_dir / f"phase{phase}" / f"loso_{subject}" / arm


def train_classifier_cell(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    fold: FoldContext,
    phase: str,
    arm: str,
    split_inputs: Mapping[str, tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, Any]:
    root = _classifier_task_root(args.output_dir, phase, fold.subject, arm)
    root.mkdir(parents=True, exist_ok=True)
    task_id = f"phase{phase}/loso_{fold.subject}/{arm}"
    done_path = root / "DONE.json"
    metrics_path = root / "metrics.json"
    complete = validate_done(
        done_path,
        stage="h200_classifier",
        protocol_fingerprint=protocol.config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=fold.source_best_sha256,
    )
    if complete is not None:
        return _load_json(metrics_path)
    if args.finalize_only:
        raise FileNotFoundError(metrics_path)

    classifier_seed = int(args.seed) + 10_000 + EXPECTED_SUBJECTS.index(fold.subject)
    rf_runner.set_seed(classifier_seed, args.deterministic)
    model = build_classifier(
        arm,
        args.classifier_hidden,
        args.classifier_dropout,
    ).to(device)
    initial_hash = _state_sha256(model)
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    architecture = (
        model.architecture_config()
        if hasattr(model, "architecture_config")
        else {"parameter_count": parameter_count}
    )

    train_arrays, y_train, _ = split_inputs["train"]
    val_arrays, y_val, val_index = split_inputs["validation"]
    test_arrays, y_test, test_index = split_inputs["test"]
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if min(counts) <= 0:
        raise RuntimeError(f"Training split lacks a class: {task_id}")
    pos_weight = min(math.sqrt(counts[0] / counts[1]), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    pin = device.type == "cuda"
    val_loader = _array_loader(
        val_arrays,
        y_val,
        args.batch_size,
        shuffle=False,
        shuffle_seed=classifier_seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = _array_loader(
        test_arrays,
        y_test,
        args.batch_size,
        shuffle=False,
        shuffle_seed=classifier_seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    epochs = args.phase1_epochs if phase == "1" else args.classifier_epochs
    patience = min(args.classifier_patience, epochs)
    best_path = root / "classifier_best.pt"
    last_path = root / "classifier_last.pt"
    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    elapsed_before = 0.0
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        expected_identity = {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": protocol.config["protocol_fingerprint"],
            "task_id": task_id,
            "arm": arm,
            "initial_state_sha256": initial_hash,
        }
        for key, value in expected_identity.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Classifier resume identity changed: {task_id}/{key}"
                )
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        grad_scaler.load_state_dict(payload["grad_scaler_state"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        restore_rng_state(payload["rng_state"])
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, epochs + 1):
        if bad_epochs >= patience:
            break
        train_loader = _array_loader(
            train_arrays,
            y_train,
            args.batch_size,
            shuffle=True,
            shuffle_seed=classifier_seed + epoch,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        train_loss, train_true, train_prob = _classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            val_loss, val_true, val_prob = _classifier_epoch(
                model, val_loader, criterion, device, args.amp
            )
        val_pr = float(average_precision_score(val_true, val_prob))
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": classifier_seed + epoch,
                "train_loss": float(train_loss),
                "train_pr_auc": float(
                    average_precision_score(train_true, train_prob)
                ),
                "validation_loss": float(val_loss),
                "validation_pr_auc": val_pr,
            }
        )
        if val_pr > best_score + 1e-5:
            best_score = val_pr
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "suite_version": SUITE_VERSION,
                    "protocol_fingerprint": protocol.config["protocol_fingerprint"],
                    "task_id": task_id,
                    "arm": arm,
                    "classifier_seed": classifier_seed,
                    "initial_state_sha256": initial_hash,
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_pr_auc": best_score,
                    "architecture": architecture,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        atomic_torch_save(
            {
                "suite_version": SUITE_VERSION,
                "protocol_fingerprint": protocol.config["protocol_fingerprint"],
                "task_id": task_id,
                "arm": arm,
                "initial_state_sha256": initial_hash,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_sec": elapsed_before + time.perf_counter() - started,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
    if not best_path.exists():
        raise RuntimeError(f"No classifier checkpoint produced: {task_id}")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        _, validation_true, validation_prob = _classifier_epoch(
            model, val_loader, criterion, device, args.amp
        )
        _, test_true, test_prob = _classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, validation_metrics = choose_threshold(
        validation_true, validation_prob
    )
    metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (np.asarray(test_prob) >= float(threshold)).astype(np.int8)
    metrics.update(
        rf_runner.event_metrics(
            protocol.dataset,
            protocol.classification_windows,
            test_index,
            test_pred,
        )
    )
    metrics = _enrich_binary_metrics(metrics)
    metrics.update(
        {
            "phase": phase,
            "arm": arm,
            "display_name": H200_ARM_REGISTRY[arm].display_name,
            "test_subject": fold.subject,
            "val_subject": fold.val_subject,
            "train_subjects": list(fold.train_subjects),
            "classifier_seed": classifier_seed,
            "parameter_count": parameter_count,
            "architecture": architecture,
            "initial_state_sha256": initial_hash,
            "best_epoch": int(best["best_epoch"]),
            "best_validation_pr_auc": float(best["best_validation_pr_auc"]),
            "validation": validation_metrics,
            "train_counts": counts.astype(int).tolist(),
            "pos_weight": float(pos_weight),
            "history": history,
            "elapsed_sec": float(
                elapsed_before + time.perf_counter() - started
            ),
            "endpoint_sha256": _array_sha256(test_index),
            "label_sha256": _array_sha256(test_true),
            "source_h200_best_sha256": fold.source_best_sha256,
        }
    )
    predictions_path = root / "predictions.npz"
    validation_path = root / "validation_predictions.npz"
    predictions_csv_path = root / "predictions.csv"
    atomic_npz_save(
        predictions_path,
        window_index=test_index,
        y_true=np.asarray(test_true, dtype=np.int8),
        y_prob=np.asarray(test_prob, dtype=np.float64),
        y_pred=test_pred,
    )
    validation_pred = (
        np.asarray(validation_prob) >= float(threshold)
    ).astype(np.int8)
    atomic_npz_save(
        validation_path,
        window_index=val_index,
        y_true=np.asarray(validation_true, dtype=np.int8),
        y_prob=np.asarray(validation_prob, dtype=np.float64),
        y_pred=validation_pred,
    )
    rf_runner.write_predictions_csv(
        predictions_csv_path,
        protocol.dataset,
        protocol.classification_windows,
        test_index,
        test_prob,
        test_pred,
    )
    atomic_json_dump(metrics, metrics_path)
    atomic_json_dump(
        done_payload(
            stage="h200_classifier",
            protocol_fingerprint=protocol.config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=fold.source_best_sha256,
            relative_to=root,
            artifacts={
                "best": best_path,
                "last": last_path,
                "metrics": metrics_path,
                "predictions": predictions_path,
                "validation_predictions": validation_path,
                "predictions_csv": predictions_csv_path,
            },
        ),
        done_path,
    )
    return metrics


def run_classifier_phase_fold(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    fold: FoldContext,
    primitive_paths: Mapping[str, Path],
    phase: str,
    arms: Sequence[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    fold_index = EXPECTED_SUBJECTS.index(fold.subject)
    selected_rows: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        labels = np.asarray(fold.support[f"{split}_y"], dtype=np.int8)
        rows = np.arange(len(labels), dtype=np.int64)
        if phase == "1":
            maximum = (
                args.phase1_train_windows
                if split == "train"
                else args.phase1_eval_windows
            )
        elif split == "train":
            maximum = args.max_classifier_windows
        else:
            maximum = 0
        selected_rows[split] = _subsample_rows(
            rows,
            labels,
            maximum,
            args.seed + 100 + fold_index + {"train": 0, "validation": 1, "test": 2}[split],
        )
    endpoint_reference = {
        split: np.asarray(
            fold.support[f"{split}_anchor_window_index"], dtype=np.int64
        )[rows]
        for split, rows in selected_rows.items()
    }
    results: list[dict[str, Any]] = []
    initial_hash_by_arm: dict[str, str] = {}
    parameter_count_by_arm: dict[str, int] = {}
    for arm in arms:
        prepared: dict[
            str, tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]
        ] = {}
        for split in ("train", "validation", "test"):
            base = materialize_split_inputs(
                protocol, fold, primitive_paths[split], split
            )
            prepared[split] = prepare_arm_inputs(
                base, arm, selected_rows[split]
            )
            del base
        for split in prepared:
            if not np.array_equal(prepared[split][2], endpoint_reference[split]):
                raise AssertionError(f"{phase}/{fold.subject}/{arm} endpoints differ")
        metrics = train_classifier_cell(
            args, protocol, fold, phase, arm, prepared, device
        )
        initial_hash_by_arm[arm] = str(metrics["initial_state_sha256"])
        parameter_count_by_arm[arm] = int(metrics["parameter_count"])
        results.append(metrics)
        del prepared
    if "raw4_zero" in arms and "raw4_normality" in arms:
        if parameter_count_by_arm["raw4_zero"] != parameter_count_by_arm["raw4_normality"]:
            raise AssertionError("Zero and normality dual branches differ in capacity")
        if initial_hash_by_arm["raw4_zero"] != initial_hash_by_arm["raw4_normality"]:
            raise AssertionError("Zero and normality dual branches differ at initialization")
    return results


def _aggregate_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [row.get(key) for row in rows]
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "mean": float(array.mean()) if len(array) else None,
        "std": float(array.std(ddof=0)) if len(array) else None,
        "n": int(len(array)),
    }


def aggregate_phase0(args: argparse.Namespace, protocol: ProtocolContext) -> dict[str, Any]:
    rows = []
    for subject in protocol.folds:
        path = _phase0_task_root(args.output_dir, subject) / "metrics.json"
        if path.exists():
            rows.append(_load_json(path))
    better = sum(bool(row["gru_better_rmse"]) for row in rows)
    completed = len(rows) == len(protocol.folds)
    reportable = completed and tuple(protocol.folds) == EXPECTED_SUBJECTS
    all_hard_checks_pass = bool(
        rows
        and all(
            all(bool(value) for value in row.get("hard_checks", {}).values())
            for row in rows
        )
    )
    persistence_gate_pass = bool(reportable and better >= 5)
    failed_hard_checks = {
        row["subject"]: [
            name
            for name, passed in row.get("hard_checks", {}).items()
            if not bool(passed)
        ]
        for row in rows
        if not all(bool(value) for value in row.get("hard_checks", {}).values())
    }
    warning_folds = {
        row["subject"]: [
            name
            for name, active in row.get("diagnostic_warnings", {}).items()
            if bool(active)
        ]
        for row in rows
        if any(bool(value) for value in row.get("diagnostic_warnings", {}).values())
    }
    if not completed:
        decision = "incomplete"
    elif not reportable:
        decision = "subset_only"
    elif persistence_gate_pass and all_hard_checks_pass:
        decision = "pass"
    else:
        decision = "fail"
    payload = {
        "completed_folds": [row["subject"] for row in rows],
        "expected_folds": list(protocol.folds),
        "completed": completed,
        "reportable_eight_fold_protocol": reportable,
        "decision": decision,
        "gru_better_rmse_subjects": int(better),
        "persistence_gate_pass": persistence_gate_pass,
        "all_hard_checks_pass": all_hard_checks_pass,
        "failed_hard_checks": failed_hard_checks,
        "diagnostic_warnings": warning_folds,
        "z_clip_rate": _aggregate_numeric(
            [{"value": row.get("clean_nonfog_z_clip_rate")} for row in rows],
            "value",
        ),
        "gru_rmse": _aggregate_numeric(
            [{"value": row["gru"]["overall"]["rmse"]} for row in rows], "value"
        ),
        "persistence_rmse": _aggregate_numeric(
            [{"value": row["persistence"]["overall"]["rmse"]} for row in rows], "value"
        ),
    }
    root = args.output_dir / "phase0"
    atomic_json_dump(payload, root / "aggregate.json")
    if completed:
        atomic_json_dump(
            done_payload(
                stage="phase0_aggregate",
                protocol_fingerprint=protocol.config["protocol_fingerprint"],
                task_id="phase0/aggregate",
                relative_to=root,
                artifacts={"aggregate": root / "aggregate.json"},
            ),
            root / "DONE.json",
        )
    return payload


def aggregate_classifier_phase(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    phase: str,
    arms: Sequence[str],
) -> dict[str, Any]:
    expected_subjects = ("S01",) if phase == "1" else protocol.folds
    rows: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in arms}
    for subject in expected_subjects:
        for arm in arms:
            path = _classifier_task_root(args.output_dir, phase, subject, arm) / "metrics.json"
            if not path.exists():
                continue
            row = _load_json(path)
            rows.append(row)
            by_arm[arm][subject] = row
    aggregates = {
        arm: {
            "completed_subjects": list(by_arm[arm]),
            "metrics": {
                metric: _aggregate_numeric(list(by_arm[arm].values()), metric)
                for metric in METRIC_NAMES
            },
        }
        for arm in arms
    }
    comparisons: dict[str, Any] = {}
    for candidate, reference in (
        ("raw4_normality", "raw6"),
        ("raw4_normality", "raw4_zero"),
        ("normality", "raw4"),
    ):
        if candidate not in by_arm or reference not in by_arm:
            continue
        subjects = [
            subject
            for subject in expected_subjects
            if subject in by_arm[candidate] and subject in by_arm[reference]
        ]
        if not subjects:
            continue
        comparisons[f"{candidate}_minus_{reference}"] = paired_bootstrap(
            {
                subject: by_arm[candidate][subject]["pr_auc"]
                for subject in subjects
            },
            {
                subject: by_arm[reference][subject]["pr_auc"]
                for subject in subjects
            },
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    complete = all(
        len(by_arm[arm]) == len(expected_subjects) for arm in arms
    )
    gate: dict[str, Any] | None = None
    smoke_gate: dict[str, Any] | None = None
    if phase == "1" and complete:
        finite_decreasing_loss: dict[str, bool] = {}
        for arm in arms:
            history = list(by_arm[arm]["S01"].get("history", []))
            losses = np.asarray(
                [row.get("train_loss") for row in history], dtype=np.float64
            )
            finite_decreasing_loss[arm] = bool(
                len(losses) >= 2
                and np.isfinite(losses).all()
                and float(losses.min()) < float(losses[0])
            )
        zero = by_arm.get("raw4_zero", {}).get("S01")
        fusion = by_arm.get("raw4_normality", {}).get("S01")
        capacity_match = bool(
            zero is not None
            and fusion is not None
            and int(zero["parameter_count"]) == int(fusion["parameter_count"])
            and zero["initial_state_sha256"] == fusion["initial_state_sha256"]
        )
        smoke_checks = {
            "all_arms_completed": complete,
            "finite_decreasing_train_loss_all_arms": all(
                finite_decreasing_loss.values()
            ),
            "zero_and_fusion_capacity_initialization_match": capacity_match,
        }
        smoke_gate = {
            "status": "pass" if all(smoke_checks.values()) else "fail",
            "checks": smoke_checks,
            "finite_decreasing_loss_by_arm": finite_decreasing_loss,
            "metrics_are_engineering_only": True,
        }
    if phase == "2" and complete and tuple(protocol.folds) == EXPECTED_SUBJECTS:
        ordered = EXPECTED_SUBJECTS
        gate = evaluate_phase2_gate(
            subject_ids=ordered,
            fusion_pr_auc=[by_arm["raw4_normality"][s]["pr_auc"] for s in ordered],
            raw6_pr_auc=[by_arm["raw6"][s]["pr_auc"] for s in ordered],
            zero_pr_auc=[by_arm["raw4_zero"][s]["pr_auc"] for s in ordered],
            normality_pr_auc=[by_arm["normality"][s]["pr_auc"] for s in ordered],
            prevalence=[
                by_arm["normality"][s]["n_fog"] / by_arm["normality"][s]["n"]
                for s in ordered
            ],
            fusion_recall=[by_arm["raw4_normality"][s]["fog_recall"] for s in ordered],
            raw6_recall=[by_arm["raw6"][s]["fog_recall"] for s in ordered],
            fusion_false_alarms_per_hour=[
                by_arm["raw4_normality"][s]["false_alarm_events_per_hour"]
                for s in ordered
            ],
            raw6_false_alarms_per_hour=[
                by_arm["raw6"][s]["false_alarm_events_per_hour"] for s in ordered
            ],
            fusion_event_sensitivity=[
                by_arm["raw4_normality"][s]["event_sensitivity"] for s in ordered
            ],
            raw6_event_sensitivity=[
                by_arm["raw6"][s]["event_sensitivity"] for s in ordered
            ],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    payload = {
        "phase": phase,
        "exploratory": phase == "2",
        "engineering_smoke": phase == "1",
        "eligible_for_final_scientific_table": phase != "1",
        "completed": complete,
        "folds": list(expected_subjects),
        "arms": aggregates,
        "comparisons": comparisons,
        "smoke_gate": smoke_gate,
        "gate": gate,
    }
    root = args.output_dir / f"phase{phase}"
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(payload, root / "aggregate.json")
    if gate is not None:
        atomic_json_dump(gate, root / "gate.json")
    columns = [
        "phase",
        "arm",
        "display_name",
        "test_subject",
        "val_subject",
        "threshold",
        "n",
        "n_normal",
        "n_fog",
        *METRIC_NAMES,
        "parameter_count",
        "initial_state_sha256",
        "best_epoch",
        "best_validation_pr_auc",
    ]
    _csv_write(root / "fold_metrics.csv", rows, columns)
    if complete:
        artifacts = {
            "aggregate": root / "aggregate.json",
            "fold_metrics": root / "fold_metrics.csv",
        }
        if gate is not None:
            artifacts["gate"] = root / "gate.json"
        atomic_json_dump(
            done_payload(
                stage=f"phase{phase}_aggregate",
                protocol_fingerprint=protocol.config["protocol_fingerprint"],
                task_id=f"phase{phase}/aggregate",
                relative_to=root,
                artifacts=artifacts,
            ),
            root / "DONE.json",
        )
    return payload


def run_crossfit_phase(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    phase: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run the leakage-safe Phase-3 subject cross-fitting implementation."""

    from cnbr_fog import h200_phase3 as h200

    hook_name = "run_phase3a" if phase == "3a" else "run_phase3b"
    hook = getattr(h200, hook_name, None)
    if hook is None:
        raise RuntimeError(f"Missing Phase-3 implementation hook: {hook_name}")
    return hook(
        args=args,
        protocol=protocol,
        device=device,
        arms=PHASE3_ARMS,
    )


def _check_stop(args: argparse.Namespace, completed: int) -> None:
    if args.stop_after_tasks > 0 and completed >= args.stop_after_tasks:
        raise RuntimeError("Intentional stop after requested completed tasks")


def _block_staged_run(
    args: argparse.Namespace,
    protocol: ProtocolContext,
    *,
    phase: str,
    reason: str,
) -> None:
    payload = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": protocol.config["protocol_fingerprint"],
        "requested_phase": args.phase,
        "folds": list(protocol.folds),
        "status": "blocked_by_stage_gate",
        "blocked_after_phase": phase,
        "reason": reason,
        "force_next_phase": bool(args.force_next_phase),
    }
    atomic_json_dump(payload, args.output_dir / "status.json")
    raise RuntimeError(
        f"Staged run stopped after Phase {phase}: {reason}. "
        "Inspect the gate artifact or use --force-next-phase for an "
        "explicit, recorded engineering override."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.source_suite_dir = args.source_suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    device = _resolve_device(args.device)
    rf_runner.set_seed(args.seed, args.deterministic)
    protocol = build_protocol(args, device)
    initialise_run(args, protocol)
    print(
        f"[INFO] suite={SUITE_VERSION} phase={args.phase} device={device} "
        f"folds={protocol.folds} source={args.source_suite_dir}",
        flush=True,
    )

    completed = 0
    phases = _phase_sequence(args.phase)
    for phase_position, phase in enumerate(phases):
        if phase in {"3a", "3b"}:
            aggregate = run_crossfit_phase(args, protocol, phase, device)
            if (
                phase == "3a"
                and phase_position + 1 < len(phases)
                and aggregate.get("decision", {}).get("status") != "pass"
                and not args.force_next_phase
            ):
                _block_staged_run(
                    args,
                    protocol,
                    phase=phase,
                    reason=(
                        "Phase-3A confirmation decision="
                        f"{aggregate.get('decision', {}).get('status')!r}"
                    ),
                )
            continue
        execution_folds = ("S01",) if phase == "1" else protocol.folds
        if phase == "1" and "S01" not in protocol.folds:
            raise ValueError("Phase 1 is preregistered to S01; include S01 in --folds")
        for subject in execution_folds:
            fold = load_fold_context(args, protocol, subject)
            primitive_paths = ensure_primitive_cache(
                args, protocol, fold, device
            )
            if phase == "0":
                run_phase0_fold(args, protocol, fold, primitive_paths)
                completed += 1
            elif phase == "1":
                results = run_classifier_phase_fold(
                    args,
                    protocol,
                    fold,
                    primitive_paths,
                    phase,
                    PHASE1_ARMS,
                    device,
                )
                completed += len(results)
            elif phase == "2":
                results = run_classifier_phase_fold(
                    args,
                    protocol,
                    fold,
                    primitive_paths,
                    phase,
                    PHASE2_ARMS,
                    device,
                )
                completed += len(results)
            else:
                raise AssertionError(phase)
            _check_stop(args, completed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if phase == "0":
            aggregate = aggregate_phase0(args, protocol)
            if (
                phase_position + 1 < len(phases)
                and aggregate["decision"] != "pass"
                and not args.force_next_phase
            ):
                _block_staged_run(
                    args,
                    protocol,
                    phase=phase,
                    reason=f"Phase-0 decision={aggregate['decision']}",
                )
        elif phase == "1":
            aggregate = aggregate_classifier_phase(
                args, protocol, phase, PHASE1_ARMS
            )
            smoke_status = (aggregate.get("smoke_gate") or {}).get("status")
            if (
                phase_position + 1 < len(phases)
                and smoke_status != "pass"
                and not args.force_next_phase
            ):
                _block_staged_run(
                    args,
                    protocol,
                    phase=phase,
                    reason=f"Phase-1 smoke gate status={smoke_status!r}",
                )
        elif phase == "2":
            aggregate = aggregate_classifier_phase(
                args, protocol, phase, PHASE2_ARMS
            )
            gate = aggregate.get("gate")
            gate_decision = gate.get("decision") if gate is not None else None
            if (
                phase_position + 1 < len(phases)
                and gate_decision not in {"strong_go", "conditional_go"}
                and not args.force_next_phase
            ):
                _block_staged_run(
                    args,
                    protocol,
                    phase=phase,
                    reason=f"Phase-2 gate decision={gate_decision!r}",
                )

    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": protocol.config["protocol_fingerprint"],
            "requested_phase": args.phase,
            "folds": list(protocol.folds),
            "status": "completed_requested_phases",
            "force_next_phase": bool(args.force_next_phase),
        },
        args.output_dir / "status.json",
    )


if __name__ == "__main__":
    main()
