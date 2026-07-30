#!/usr/bin/env python
"""Strict GRU-NBM forecast-horizon ablation for Daphnet FoG detection.

Only the NBM forecast horizon changes:

* H025: 0.25 s / 16 samples / 16 blocks in the four-second history;
* H050: 0.50 s / 32 samples /  8 blocks;
* H100: 1.00 s / 64 samples /  4 blocks;
* H200: 2.00 s / 128 samples / 2 blocks.

All arms use a two-second context, a four-second residual history with shape
``[batch, 9, 256]``, the fixed RF125 TCN-M classifier, identical LOSO splits,
and labels computed from the final 0.5 seconds at a common decision endpoint.
A master WindowTable with the maximum two-second horizon defines validity and
clean-normal eligibility.  Per-horizon WindowTables merely reframe that same
set of endpoints, so no arm gains extra records or normal training examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_daphnet_3imu_nbm_suite as core
import run_daphnet_nbm_context_tcnm_suite as context_suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.histories import HistoryPlan, make_common_history_plan, make_history_input
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    done_payload,
    sha256_file,
)


SUITE_VERSION = "daphnet_gru_horizon4_h4_tcnm_loso.v1"
NBM_NAME = "gru"
INPUT_NAME = "residual_h4s"
SAMPLING_RATE_HZ = 64
CONTEXT_SECONDS = 2.0
CONTEXT_SAMPLES = 128
HISTORY_SECONDS = 4.0
HISTORY_SAMPLES = 256
FIXED_LABEL_SECONDS = 0.5
FIXED_LABEL_SAMPLES = 32
STRIDE_SECONDS = 0.25
STRIDE_SAMPLES = 16
SUPPORT_HORIZON_SECONDS = 2.0
SUPPORT_HORIZON_SAMPLES = 128
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
EXPECTED_CHANNEL_NAMES = core.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = core.EXPECTED_LOSO_SUBJECTS

HORIZON_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "horizon_id": "H025",
        "horizon_seconds": 0.25,
        "horizon_samples": 16,
        "history_blocks": 16,
    },
    {
        "horizon_id": "H050",
        "horizon_seconds": 0.5,
        "horizon_samples": 32,
        "history_blocks": 8,
    },
    {
        "horizon_id": "H100",
        "horizon_seconds": 1.0,
        "horizon_samples": 64,
        "history_blocks": 4,
    },
    {
        "horizon_id": "H200",
        "horizon_seconds": 2.0,
        "horizon_samples": 128,
        "history_blocks": 2,
    },
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

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_gru_horizon_ablation.py",
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_daphnet_nbm_context_tcnm_suite.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/resume.py",
)

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "daphnet_gru_horizon4_h4_tcnm_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict GRU-NBM H025/H050/H100/H200 horizon ablation with "
            "common four-second residual support and TCN-M"
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
        help="Execute exactly one fold while retaining the full protocol",
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--exclude-subjects", default="S04,S10")
    parser.add_argument("--horizons", default="H025,H050,H100,H200")

    # These are exposed so launchers record the fixed scientific contract.
    parser.add_argument("--context-seconds", type=float, default=CONTEXT_SECONDS)
    parser.add_argument(
        "--support-horizon-seconds",
        type=float,
        default=SUPPORT_HORIZON_SECONDS,
    )
    parser.add_argument(
        "--fixed-label-seconds",
        type=float,
        default=FIXED_LABEL_SECONDS,
    )
    parser.add_argument("--stride-seconds", type=float, default=STRIDE_SECONDS)
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
    parser.add_argument("--gru-layers", type=int, default=1)
    # Required by the shared NBM factory although unused by this GRU-only suite.
    parser.add_argument("--linear-ar-seconds", type=float, default=0.5)
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
        help="Development-only: allow a horizon/fold subset",
    )
    parser.add_argument("--stop-after-completed-tasks", type=int, default=0)
    parser.add_argument("--debug-interrupt-nbm-after-epoch", type=int, default=0)
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
    )
    return parser.parse_args()


def parse_horizons(
    specification: str,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
) -> list[dict[str, Any]]:
    """Parse horizon IDs/durations and return canonical ordered definitions."""

    if int(sampling_rate_hz) != SAMPLING_RATE_HZ:
        raise ValueError("This preregistered suite requires 64 Hz data")
    by_id = {
        str(item["horizon_id"]).upper(): dict(item)
        for item in HORIZON_DEFINITIONS
    }
    aliases: dict[str, str] = {}
    for item in HORIZON_DEFINITIONS:
        horizon_id = str(item["horizon_id"])
        seconds = float(item["horizon_seconds"])
        aliases[horizon_id.upper()] = horizon_id
        aliases[f"{seconds:g}"] = horizon_id
        aliases[f"{seconds:.2f}"] = horizon_id
    requested: list[str] = []
    for raw in str(specification).split(","):
        token = raw.strip()
        if not token:
            continue
        horizon_id = aliases.get(token.upper()) or aliases.get(token)
        if horizon_id is None:
            raise ValueError(
                f"Unknown horizon {token!r}; use H025,H050,H100,H200"
            )
        if horizon_id not in requested:
            requested.append(horizon_id)
    if not requested:
        raise ValueError("At least one horizon is required")
    canonical_order = [str(item["horizon_id"]) for item in HORIZON_DEFINITIONS]
    selected = set(requested)
    result = [by_id[horizon_id] for horizon_id in canonical_order if horizon_id in selected]
    for item in result:
        expected = int(round(float(item["horizon_seconds"]) * sampling_rate_hz))
        if expected != int(item["horizon_samples"]):
            raise AssertionError(f"Invalid horizon definition: {item}")
        if HISTORY_SAMPLES % expected:
            raise AssertionError(f"Horizon does not divide history: {item}")
        if HISTORY_SAMPLES // expected != int(item["history_blocks"]):
            raise AssertionError(f"Invalid history block count: {item}")
    return result


def horizon_directory(horizon: Mapping[str, Any]) -> str:
    duration = f"{float(horizon['horizon_seconds']):g}".replace(".", "p")
    return f"horizon_{str(horizon['horizon_id']).lower()}_{duration}s"


def experiment_id(horizon: Mapping[str, Any]) -> str:
    duration = f"{float(horizon['horizon_seconds']):g}".replace(".", "p")
    return (
        f"gru__{str(horizon['horizon_id']).lower()}_horizon{duration}s__"
        f"{INPUT_NAME}__tcn_m"
    )


def horizon_grid(horizons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "directory": horizon_directory(item),
            "nbm": NBM_NAME,
            "input": INPUT_NAME,
            "classifier": "tcn_m",
            "experiment_id": experiment_id(item),
        }
        for item in horizons
    ]


experiment_grid = horizon_grid


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if not args.cache_residuals:
        raise ValueError("Residual caches are required for resume and audit")
    fixed = {
        "context_seconds": (args.context_seconds, CONTEXT_SECONDS),
        "support_horizon_seconds": (
            args.support_horizon_seconds,
            SUPPORT_HORIZON_SECONDS,
        ),
        "fixed_label_seconds": (args.fixed_label_seconds, FIXED_LABEL_SECONDS),
        "stride_seconds": (args.stride_seconds, STRIDE_SECONDS),
        "history_seconds": (args.history_seconds, HISTORY_SECONDS),
    }
    changed = [
        name
        for name, (actual, expected) in fixed.items()
        if not math.isclose(float(actual), float(expected))
    ]
    if changed:
        raise ValueError(f"Strict protocol fixes these options: {changed}")
    positive_ints = {
        "normal_epochs": args.normal_epochs,
        "normal_patience": args.normal_patience,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "nbm_hidden": args.nbm_hidden,
        "gru_layers": args.gru_layers,
        "classifier_hidden": args.classifier_hidden,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These integer options must be positive: {invalid}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_normal_windows < 0 or args.max_classifier_windows < 0:
        raise ValueError("Window caps must be non-negative")
    if args.normal_guard_seconds < 0:
        raise ValueError("--normal-guard-seconds must be non-negative")
    if args.weight_decay < 0 or args.zero_tolerance < 0:
        raise ValueError("--weight-decay and --zero-tolerance must be non-negative")
    if not 0.0 < args.fog_fraction_threshold <= 1.0:
        raise ValueError("--fog-fraction-threshold must be in (0,1]")
    for name in ("nbm_dropout", "classifier_dropout"):
        if not 0.0 <= float(getattr(args, name)) < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1)")
    for name in (
        "normal_lr",
        "classifier_lr",
        "robust_clip",
        "residual_clip",
        "flatline_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def window_table_sha256(windows: WindowTable) -> str:
    fields = (
        "record_index",
        "start",
        "target_start",
        "target_end",
        "label",
        "fog_fraction",
        "clean_normal",
    )
    return canonical_fingerprint(
        {name: array_sha256(np.asarray(getattr(windows, name))) for name in fields}
    )


def fixed_endpoint_labels(
    dataset: DaphnetDataset,
    windows: WindowTable,
    fixed_label_samples: int = FIXED_LABEL_SAMPLES,
    fog_fraction_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each endpoint using only its final fixed 0.5-second interval."""

    fixed_label_samples = int(fixed_label_samples)
    labels = np.empty(len(windows), dtype=np.int8)
    fractions = np.empty(len(windows), dtype=np.float32)
    for record_index, record in enumerate(dataset.records):
        rows = np.flatnonzero(windows.record_index == record_index)
        if not len(rows):
            continue
        ends = windows.target_end[rows].astype(np.int64)
        starts = ends - fixed_label_samples
        if np.any(starts < 0):
            raise ValueError("A fixed label interval starts before the record")
        prefix = np.r_[0, np.cumsum(record.y == 1, dtype=np.int64)]
        counts = prefix[ends] - prefix[starts]
        fraction = counts.astype(np.float64) / float(fixed_label_samples)
        fractions[rows] = fraction.astype(np.float32)
        labels[rows] = (fraction >= float(fog_fraction_threshold)).astype(np.int8)
    return labels, fractions


def relabel_master_windows(
    dataset: DaphnetDataset,
    master: WindowTable,
    fixed_label_samples: int = FIXED_LABEL_SAMPLES,
    fog_fraction_threshold: float = 0.5,
) -> WindowTable:
    labels, fractions = fixed_endpoint_labels(
        dataset,
        master,
        fixed_label_samples,
        fog_fraction_threshold,
    )
    return WindowTable(
        record_index=master.record_index.copy(),
        start=master.start.copy(),
        target_start=master.target_start.copy(),
        target_end=master.target_end.copy(),
        label=labels,
        fog_fraction=fractions,
        clean_normal=master.clean_normal.copy(),
    )


def derive_horizon_windows(
    master: WindowTable,
    horizon_samples: int,
    fixed_label_samples: int = FIXED_LABEL_SAMPLES,
) -> WindowTable:
    """Reframe master endpoints for one NBM horizon without changing support."""

    horizon_samples = int(horizon_samples)
    if horizon_samples <= 0 or horizon_samples > SUPPORT_HORIZON_SAMPLES:
        raise ValueError("horizon_samples must be in [1,128]")
    if int(fixed_label_samples) != FIXED_LABEL_SAMPLES:
        raise ValueError("This suite fixes labels at 32 samples")
    target_end = master.target_end.copy()
    target_start = target_end - horizon_samples
    start = target_start - CONTEXT_SAMPLES
    if np.any(start < 0):
        raise ValueError("Derived context starts before its record")
    return WindowTable(
        record_index=master.record_index.copy(),
        start=start.astype(np.int32, copy=False),
        target_start=target_start.astype(np.int32, copy=False),
        target_end=target_end,
        label=master.label.copy(),
        fog_fraction=master.fog_fraction.copy(),
        clean_normal=master.clean_normal.copy(),
    )


def derive_classification_windows(master: WindowTable) -> WindowTable:
    """Create the fixed final-0.5-second target used for labels/events."""

    return WindowTable(
        record_index=master.record_index.copy(),
        # The master start is also the start of the four-second residual support.
        start=master.start.copy(),
        target_start=(master.target_end - FIXED_LABEL_SAMPLES).astype(
            np.int32,
            copy=False,
        ),
        target_end=master.target_end.copy(),
        label=master.label.copy(),
        fog_fraction=master.fog_fraction.copy(),
        clean_normal=master.clean_normal.copy(),
    )


def build_common_history_support(
    windows_by_horizon: Mapping[str, WindowTable],
    split_indices: Mapping[str, np.ndarray],
    history_samples: int = HISTORY_SAMPLES,
    stride_samples: int = STRIDE_SAMPLES,
) -> dict[str, dict[str, HistoryPlan]]:
    """Build per-horizon plans and restrict all of them to common endpoints."""

    raw: dict[str, dict[str, HistoryPlan]] = {}
    for horizon_id, windows in windows_by_horizon.items():
        horizon_samples = int(windows.target_end[0] - windows.target_start[0])
        raw[horizon_id] = {
            split: make_common_history_plan(
                windows,
                indices,
                horizon_samples,
                stride_samples,
                history_samples,
            )
            for split, indices in split_indices.items()
        }

    result: dict[str, dict[str, HistoryPlan]] = {
        horizon_id: {} for horizon_id in windows_by_horizon
    }
    horizon_ids = list(windows_by_horizon)
    if not horizon_ids:
        raise ValueError("No horizons supplied")
    for split in split_indices:
        common = set(raw[horizon_ids[0]][split].anchor_window_indices.tolist())
        for horizon_id in horizon_ids[1:]:
            common &= set(raw[horizon_id][split].anchor_window_indices.tolist())
        ordered = np.asarray(
            [
                value
                for value in raw[horizon_ids[0]][split].anchor_window_indices
                if int(value) in common
            ],
            dtype=np.int64,
        )
        if not len(ordered):
            raise RuntimeError(f"Empty common history support for split {split}")
        for horizon_id in horizon_ids:
            plan = raw[horizon_id][split]
            lookup = {
                int(window_index): row
                for row, window_index in enumerate(plan.anchor_window_indices)
            }
            rows = np.asarray([lookup[int(value)] for value in ordered], dtype=np.int64)
            result[horizon_id][split] = plan.take(rows)
            if not np.array_equal(
                result[horizon_id][split].anchor_window_indices,
                ordered,
            ):
                raise AssertionError("Common history endpoints are misaligned")
    return result


def _model_shared_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the horizon-invariant GRU encoder/summary parameter subset."""

    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("encoder.") or name.startswith("summary.")
    }


def gru_architectures(
    args: argparse.Namespace,
    horizons: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Describe each direct decoder and verify shared initial encoder weights."""

    architectures: dict[str, dict[str, Any]] = {}
    encoder_hashes: list[str] = []
    for horizon in horizons:
        core.set_seed(seed, args.deterministic)
        model = core.build_model(
            args,
            NBM_NAME,
            len(EXPECTED_CHANNEL_NAMES),
            int(horizon["horizon_samples"]),
            CONTEXT_SAMPLES,
            SAMPLING_RATE_HZ,
        )
        shared_hash = rf.state_dict_sha256(_model_shared_state(model))
        encoder_hashes.append(shared_hash)
        decoder_parameters = sum(
            int(parameter.numel())
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.")
        )
        architectures[str(horizon["horizon_id"])] = {
            "model_config": model.model_config(),
            "parameter_count": core.parameter_count(model),
            "decoder_parameter_count": decoder_parameters,
            "shared_encoder_summary_parameter_count": (
                core.parameter_count(model) - decoder_parameters
            ),
            "initial_shared_encoder_summary_sha256": shared_hash,
        }
        del model
    if len(set(encoder_hashes)) != 1:
        raise AssertionError("GRU encoder/summary initialization differs by horizon")
    return architectures, encoder_hashes[0]


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def validate_protocol_selection(
    horizons: list[dict[str, Any]],
    folds: list[str],
    allow_subset: bool,
) -> None:
    if allow_subset:
        return
    if [item["horizon_id"] for item in horizons] != [
        item["horizon_id"] for item in HORIZON_DEFINITIONS
    ]:
        raise ValueError("Strict protocol requires H025,H050,H100,H200")
    if tuple(folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("Strict protocol requires --folds all")


def protocol_payload(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    source_subjects: list[str],
    excluded_subjects: list[str],
    folds: list[str],
    horizons: list[dict[str, Any]],
    master_windows: WindowTable,
    windows_by_horizon: Mapping[str, WindowTable],
    classification_windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Build the immutable scientific protocol and its fingerprint."""

    experiments = horizon_grid(horizons)
    architectures, shared_encoder_hash = gru_architectures(
        args,
        horizons,
        int(args.seed),
    )
    classifier = context_suite.classifier_architecture(
        args,
        dataset.n_channels,
        dataset.sampling_rate_hz,
    )
    if int(classifier["receptive_field_samples"]) != TCN_M_RF_SAMPLES:
        raise AssertionError("TCN-M receptive field changed")
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
        "nbm": NBM_NAME,
        "horizon_variants": horizons,
        "experiments": experiments,
        "context_seconds": CONTEXT_SECONDS,
        "context_samples": CONTEXT_SAMPLES,
        "support_horizon_seconds": SUPPORT_HORIZON_SECONDS,
        "support_horizon_samples": SUPPORT_HORIZON_SAMPLES,
        "fixed_label_seconds": FIXED_LABEL_SECONDS,
        "fixed_label_samples": FIXED_LABEL_SAMPLES,
        "stride_seconds": STRIDE_SECONDS,
        "stride_samples": STRIDE_SAMPLES,
        "history_name": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_shape": [dataset.n_channels, HISTORY_SAMPLES],
        "master_window_count": int(len(master_windows)),
        "master_window_sha256": window_table_sha256(master_windows),
        "derived_window_sha256": {
            horizon_id: window_table_sha256(windows)
            for horizon_id, windows in windows_by_horizon.items()
        },
        "classification_window_sha256": window_table_sha256(
            classification_windows
        ),
        "fixed_label_class_counts": np.bincount(
            classification_windows.label,
            minlength=2,
        ).astype(int).tolist(),
        "master_clean_normal_windows": int(master_windows.clean_normal.sum()),
        "master_support_policy": (
            "A 2 s context + maximum 2 s horizon + 0.5 s normal guard "
            "defines validity and clean-normal eligibility for every arm."
        ),
        "horizon_alignment": (
            "All forecast blocks are right-aligned to shared endpoints; each "
            "derived NBM context immediately precedes its own horizon."
        ),
        "classification_label_policy": (
            "All arms use the same label from the final 32 samples (0.5 s) "
            "at the common classification endpoint."
        ),
        "common_anchor_policy": (
            "Per-horizon complete 4 s history plans are intersected by global "
            "WindowTable row ID within each split before classifier training."
        ),
        "history_construction": (
            "Chronological horizon-spaced, non-overlapping residual blocks "
            "cover exactly 256 samples for every arm."
        ),
        "normal_guard_samples": int(
            round(float(args.normal_guard_seconds) * SAMPLING_RATE_HZ)
        ),
        "fog_fraction_threshold": float(args.fog_fraction_threshold),
        "flatline_seconds": float(args.flatline_seconds),
        "zero_tolerance": float(args.zero_tolerance),
        "robust_clip": float(args.robust_clip),
        "residual_clip": float(args.residual_clip),
        "seed": int(args.seed),
        "nbm_hidden": int(args.nbm_hidden),
        "nbm_dropout": float(args.nbm_dropout),
        "gru_layers": int(args.gru_layers),
        "gru_architectures": architectures,
        "shared_initial_gru_encoder_summary_sha256": shared_encoder_hash,
        "gru_decoder_disclosure": (
            "The direct Gaussian decoder output width and parameter count "
            "increase with horizon; encoder and summary architecture and "
            "initial values are shared."
        ),
        "classifier_hidden": int(args.classifier_hidden),
        "classifier_dropout": float(args.classifier_dropout),
        "classifier": classifier,
        "shared_parameter_count": int(classifier["parameter_count"]),
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
        "delta_pr_auc_reference": "H050",
        "bootstrap_unit": "held_out_subject",
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "expected_experiments": len(experiments),
        "expected_nbm_tasks": len(experiments) * len(folds),
        "expected_classifier_cells": len(experiments) * len(folds),
        "protocol_scope": (
            "development_subset"
            if args.allow_protocol_subset
            else "strict_gru_horizon4_x_8_fold"
        ),
        "fairness_contract": {
            "ablation_axis": "gru_nbm_forecast_horizon",
            "shared_fields": [
                "LOSO split",
                "robust scaler",
                "maximum-horizon clean-normal NBM rows",
                "two-second NBM context",
                "four-second classifier input length",
                "common classifier endpoint IDs and labels",
                "TCN-M architecture and initial values",
                "classifier epoch shuffle order",
                "loss and class weighting",
                "validation-only early stopping and threshold",
            ],
            "necessarily_horizon_dependent": [
                "GRU Gaussian decoder width and parameter count",
                "residual block length",
                "number of blocks concatenated into four seconds",
            ],
        },
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


build_protocol = protocol_payload


def load_dataset_and_protocol(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    DaphnetDataset,
    WindowTable,
    dict[str, WindowTable],
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
    if int(dataset.sampling_rate_hz) != SAMPLING_RATE_HZ:
        raise ValueError("Daphnet horizon suite requires 64 Hz")
    horizons = parse_horizons(args.horizons, dataset.sampling_rate_hz)
    folds = core.parse_folds(args.folds, dataset.subjects)
    validate_protocol_selection(horizons, folds, args.allow_protocol_subset)
    guard_samples = int(
        round(float(args.normal_guard_seconds) * dataset.sampling_rate_hz)
    )
    raw_master = dataset.make_windows(
        warmup_samples=CONTEXT_SAMPLES,
        target_samples=SUPPORT_HORIZON_SAMPLES,
        stride_samples=STRIDE_SAMPLES,
        fog_fraction_threshold=args.fog_fraction_threshold,
        normal_guard_samples=guard_samples,
    )
    master = relabel_master_windows(
        dataset,
        raw_master,
        FIXED_LABEL_SAMPLES,
        args.fog_fraction_threshold,
    )
    windows_by_horizon = {
        str(item["horizon_id"]): derive_horizon_windows(
            master,
            int(item["horizon_samples"]),
            FIXED_LABEL_SAMPLES,
        )
        for item in horizons
    }
    classification_windows = derive_classification_windows(master)
    config = protocol_payload(
        args,
        dataset,
        source_subjects,
        excluded_subjects,
        folds,
        horizons,
        master,
        windows_by_horizon,
        classification_windows,
        data_sha256,
        device,
    )
    return (
        config,
        dataset,
        master,
        windows_by_horizon,
        classification_windows,
        horizons,
        folds,
    )


def task_root_for(
    output_dir: Path,
    subject: str,
    horizon: Mapping[str, Any],
) -> Path:
    return (
        output_dir
        / f"loso_{subject}"
        / horizon_directory(horizon)
        / NBM_NAME
        / INPUT_NAME
        / "tcn_m"
    )


def nbm_root_for(
    output_dir: Path,
    subject: str,
    horizon: Mapping[str, Any],
) -> Path:
    return (
        output_dir
        / f"loso_{subject}"
        / horizon_directory(horizon)
        / NBM_NAME
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def prepare_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    master_windows: WindowTable,
    windows_by_horizon: Mapping[str, WindowTable],
    classification_windows: WindowTable,
    horizons: list[dict[str, Any]],
    test_subject: str,
) -> tuple[
    Path,
    str,
    list[str],
    Any,
    dict[str, np.ndarray],
    dict[str, dict[str, HistoryPlan]],
    np.ndarray,
    np.ndarray,
    str,
]:
    """Prepare one leakage-safe fold with exact common classification anchors."""

    fold_root = args.output_dir / f"loso_{test_subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    subjects = list(dataset.subjects)
    start = subjects.index(test_subject)
    val_subject = ""
    for offset in range(1, len(subjects)):
        candidate = subjects[(start + offset) % len(subjects)]
        candidate_indices = dataset.window_indices_for_subjects(
            classification_windows,
            [candidate],
        )
        if np.unique(classification_windows.label[candidate_indices]).size == 2:
            val_subject = candidate
            break
    if not val_subject:
        raise RuntimeError("No validation subject with both fixed-label classes")
    train_subjects = [
        subject
        for subject in subjects
        if subject not in {test_subject, val_subject}
    ]
    scaler = dataset.fit_scaler(train_subjects, clip=args.robust_clip)
    split_indices = {
        "train": dataset.window_indices_for_subjects(
            classification_windows,
            train_subjects,
        ),
        "validation": dataset.window_indices_for_subjects(
            classification_windows,
            [val_subject],
        ),
        "test": dataset.window_indices_for_subjects(
            classification_windows,
            [test_subject],
        ),
    }
    normal_train = dataset.window_indices_for_subjects(
        master_windows,
        train_subjects,
        clean_normal_only=True,
    )
    normal_validation = dataset.window_indices_for_subjects(
        master_windows,
        [val_subject],
        clean_normal_only=True,
    )
    fold_index = subjects.index(test_subject)
    normal_train = core.deterministic_subsample(
        normal_train,
        args.max_normal_windows,
        args.seed + fold_index,
    )
    plans = build_common_history_support(
        windows_by_horizon,
        split_indices,
        HISTORY_SAMPLES,
        STRIDE_SAMPLES,
    )
    horizon_ids = [str(item["horizon_id"]) for item in horizons]
    reference = horizon_ids[0]
    if args.max_classifier_windows > 0:
        reference_plan = plans[reference]["train"]
        plan_rows = np.arange(len(reference_plan.anchor_rows), dtype=np.int64)
        labels = classification_windows.label[
            reference_plan.anchor_window_indices
        ]
        selected = core.deterministic_subsample(
            plan_rows,
            args.max_classifier_windows,
            args.seed + 100 + fold_index,
            labels,
        )
        for horizon_id in horizon_ids:
            plans[horizon_id]["train"] = plans[horizon_id]["train"].take(
                selected
            )

    for split in ("train", "validation", "test"):
        expected_anchor = plans[reference][split].anchor_window_indices
        expected_labels = classification_windows.label[expected_anchor]
        if np.unique(expected_labels).size < 2:
            raise RuntimeError(
                f"Common {split} support lacks a class in fold {test_subject}"
            )
        for horizon_id in horizon_ids:
            plan = plans[horizon_id][split]
            if not np.array_equal(plan.anchor_window_indices, expected_anchor):
                raise AssertionError("Horizon plans do not share endpoints")
            if not np.array_equal(
                windows_by_horizon[horizon_id].label[plan.anchor_window_indices],
                expected_labels,
            ):
                raise AssertionError("Horizon plans do not share fixed labels")

    core.save_or_validate_json(fold_root / "scaler.json", scaler.as_dict())
    core.save_or_validate_npz(
        fold_root / "split_indices.npz",
        train_window_index=split_indices["train"],
        validation_window_index=split_indices["validation"],
        test_window_index=split_indices["test"],
        normal_train_window_index=normal_train,
        normal_validation_window_index=normal_validation,
    )
    support_arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        anchor = plans[reference][split].anchor_window_indices
        support_arrays[f"{split}_anchor_window_index"] = anchor
        support_arrays[f"{split}_y"] = classification_windows.label[anchor]
        for horizon_id in horizon_ids:
            plan = plans[horizon_id][split]
            support_arrays[
                f"{split}_{horizon_id.lower()}_history_window_index"
            ] = split_indices[split][plan.max_chain_rows]
    support_path = fold_root / "common_history_support.npz"
    core.save_or_validate_npz(support_path, **support_arrays)
    support_sha256 = sha256_file(support_path)

    scaler_sha256 = sha256_file(fold_root / "scaler.json")
    split_sha256 = sha256_file(fold_root / "split_indices.npz")
    fold_architectures, fold_shared_encoder_hash = gru_architectures(
        args,
        horizons,
        args.seed + fold_index,
    )
    support_hashes = {
        horizon_id: {
            split: array_sha256(
                support_arrays[
                    f"{split}_{horizon_id.lower()}_history_window_index"
                ]
            )
            for split in ("train", "validation", "test")
        }
        for horizon_id in horizon_ids
    }
    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "excluded_subjects": config["excluded_subjects"],
        "scaler": scaler.as_dict(),
        "scaler_sha256": scaler_sha256,
        "split_indices_sha256": split_sha256,
        "common_history_support_sha256": support_sha256,
        "normal_train_window_indices_sha256": array_sha256(normal_train),
        "normal_validation_window_indices_sha256": array_sha256(
            normal_validation
        ),
        "normal_train_windows": int(len(normal_train)),
        "normal_validation_windows": int(len(normal_validation)),
        "source_window_counts": {
            split: int(len(indices)) for split, indices in split_indices.items()
        },
        "common_anchor_counts": {
            split: int(len(plans[reference][split].anchor_rows))
            for split in ("train", "validation", "test")
        },
        "common_anchor_sha256": {
            split: array_sha256(plans[reference][split].anchor_window_indices)
            for split in ("train", "validation", "test")
        },
        "per_horizon_history_support_sha256": support_hashes,
        "per_horizon_gru_architecture": fold_architectures,
        "initial_shared_gru_encoder_summary_sha256": (
            fold_shared_encoder_hash
        ),
        "label_window_samples": FIXED_LABEL_SAMPLES,
        "classification_window_sha256": config[
            "classification_window_sha256"
        ],
    }
    core.save_or_validate_json(fold_root / "fold_config.json", fold_config)
    return (
        fold_root,
        val_subject,
        train_subjects,
        scaler,
        split_indices,
        plans,
        normal_train,
        normal_validation,
        support_sha256,
    )


def fold_classifier_config(
    args: argparse.Namespace,
    config: dict[str, Any],
    fold_index: int,
    support_sha256: str,
    test_subject: str,
    val_subject: str,
) -> dict[str, Any]:
    """Create the one shared TCN-M initialization identity for this fold."""

    return context_suite.fold_classifier_config(
        args,
        config,
        fold_index,
        support_sha256,
        test_subject,
        val_subject,
    )


def _nbm_task_id(horizon: Mapping[str, Any]) -> str:
    return f"{horizon_directory(horizon)}/{NBM_NAME}/nbm"


def _residual_task_id(horizon: Mapping[str, Any]) -> str:
    return f"{horizon_directory(horizon)}/{NBM_NAME}/residual_cache"


def run_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    master_windows: WindowTable,
    windows_by_horizon: Mapping[str, WindowTable],
    classification_windows: WindowTable,
    horizons: list[dict[str, Any]],
    test_subject: str,
    device: torch.device,
) -> int:
    (
        fold_root,
        val_subject,
        train_subjects,
        scaler,
        split_indices,
        plans,
        normal_train_indices,
        normal_val_indices,
        support_sha256,
    ) = prepare_fold(
        args,
        config,
        dataset,
        master_windows,
        windows_by_horizon,
        classification_windows,
        horizons,
        test_subject,
    )
    fold_index = list(dataset.subjects).index(test_subject)
    classifier_fold = fold_classifier_config(
        args,
        config,
        fold_index,
        support_sha256,
        test_subject,
        val_subject,
    )
    print(
        f"[fold {test_subject}] train={train_subjects} val={val_subject} "
        f"normal={len(normal_train_indices)}/{len(normal_val_indices)} "
        f"common_anchors="
        f"{ {split: len(plans[horizons[0]['horizon_id']][split].anchor_rows) for split in ('train', 'validation', 'test')} }",
        flush=True,
    )

    completed = 0
    fold_artifacts: dict[str, Path] = {
        "fold_config": fold_root / "fold_config.json",
        "scaler": fold_root / "scaler.json",
        "split_indices": fold_root / "split_indices.npz",
        "common_history_support": fold_root / "common_history_support.npz",
    }
    for horizon in horizons:
        horizon_id = str(horizon["horizon_id"])
        horizon_samples = int(horizon["horizon_samples"])
        history_blocks = int(horizon["history_blocks"])
        arm_windows = windows_by_horizon[horizon_id]
        nbm_root = nbm_root_for(args.output_dir, test_subject, horizon)
        nbm_root.mkdir(parents=True, exist_ok=True)

        model, normal_training, nbm_sha256 = core.train_nbm_resumable(
            args,
            NBM_NAME,
            nbm_root,
            config["protocol_fingerprint"],
            args.seed + fold_index,
            dataset,
            arm_windows,
            normal_train_indices,
            normal_val_indices,
            scaler,
            CONTEXT_SAMPLES,
            horizon_samples,
            device,
        )
        features, residual_diagnostics = core.load_or_extract_residual_cache(
            args,
            nbm_root,
            config["protocol_fingerprint"],
            NBM_NAME,
            nbm_sha256,
            model,
            dataset,
            arm_windows,
            split_indices,
            scaler,
            CONTEXT_SAMPLES,
            device,
        )
        architecture = config["gru_architectures"][horizon_id]
        atomic_json_dump(
            {
                "suite_version": SUITE_VERSION,
                "protocol_fingerprint": config["protocol_fingerprint"],
                "experiment_id": experiment_id(horizon),
                "horizon_id": horizon_id,
                "horizon_seconds": horizon["horizon_seconds"],
                "horizon_samples": horizon_samples,
                "context_seconds": CONTEXT_SECONDS,
                "context_samples": CONTEXT_SAMPLES,
                "history_seconds": HISTORY_SECONDS,
                "history_samples": HISTORY_SAMPLES,
                "history_blocks": history_blocks,
                "fixed_label_seconds": FIXED_LABEL_SECONDS,
                "fixed_label_samples": FIXED_LABEL_SAMPLES,
                "master_clean_normal_support": True,
                "derived_window_sha256": config["derived_window_sha256"][
                    horizon_id
                ],
                "nbm_sha256": nbm_sha256,
                "gru_architecture": architecture,
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
                plans[horizon_id][split],
                INPUT_NAME,
                HISTORY_SAMPLES,
                horizon_samples,
                STRIDE_SAMPLES,
            )
            for split in ("train", "validation", "test")
        }
        for split, payload in inputs.items():
            expected_shape = (
                len(plans[horizon_id][split].anchor_rows),
                dataset.n_channels,
                HISTORY_SAMPLES,
            )
            if tuple(payload[INPUT_NAME].shape) != expected_shape:
                raise AssertionError(
                    f"Unexpected {split}/{horizon_id} input shape "
                    f"{payload[INPUT_NAME].shape} != {expected_shape}"
                )
            expected_y = classification_windows.label[payload["window_index"]]
            if not np.array_equal(payload["y"], expected_y):
                raise AssertionError(
                    f"{split}/{horizon_id} does not use fixed 0.5 s labels"
                )
            reference_ids = plans[str(horizons[0]["horizon_id"])][
                split
            ].anchor_window_indices
            if not np.array_equal(payload["window_index"], reference_ids):
                raise AssertionError(
                    f"{split}/{horizon_id} does not use common endpoints"
                )

        residual_path = nbm_root / "residual_cache.npz"
        residual_sha256 = sha256_file(residual_path)
        cell_experiment = experiment_id(horizon)
        variant = {
            "variant": cell_experiment,
            "display_name": f"GRU-{horizon_id} + TCN-M",
            "experiment_id": cell_experiment,
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
            "shared_parameter_count": config["shared_parameter_count"],
        }
        # The shared classifier primitive reads these module constants at call
        # time.  Worker processes are fold-local, and cells run sequentially.
        rf.SOURCE_NBM = NBM_NAME
        rf.INPUT_NAME = INPUT_NAME
        rf.HISTORY_SECONDS = HISTORY_SECONDS
        rf.HISTORY_SAMPLES = HISTORY_SAMPLES
        rf.HISTORY_BLOCKS = history_blocks
        metrics = rf.train_classifier_resumable(
            args,
            rf_config,
            variant,
            task_root_for(args.output_dir, test_subject, horizon),
            rf_fold_config,
            inputs,
            dataset,
            # Event coverage and delay must always use final 0.5 s label windows.
            classification_windows,
            device,
        )
        print(
            f"[fold {test_subject}] {cell_experiment} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"BA={metrics['balanced_accuracy']:.4f} "
            f"Recall={metrics['fog_recall']:.4f} "
            f"Specificity={metrics['specificity']:.4f}",
            flush=True,
        )
        cell_root = task_root_for(args.output_dir, test_subject, horizon)
        fold_artifacts[f"{horizon_id}_classifier_done"] = (
            cell_root / "DONE.json"
        )
        fold_artifacts[f"{horizon_id}_nbm_done"] = (
            nbm_root / "nbm" / "DONE.json"
        )
        fold_artifacts[f"{horizon_id}_residual_done"] = (
            nbm_root / "RESIDUAL_CACHE_DONE.json"
        )
        completed += 1
        del inputs, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (
            args.stop_after_completed_tasks > 0
            and completed >= args.stop_after_completed_tasks
        ):
            raise RuntimeError("Intentional stop after completed classifier tasks")

    fold_done = done_payload(
        stage="horizon_fold",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=f"{test_subject}/fold",
        relative_to=fold_root,
        artifacts={
            name: path.resolve() for name, path in fold_artifacts.items()
        },
    )
    fold_done.update(
        {
            "test_subject": test_subject,
            "completed_horizons": [
                str(item["horizon_id"]) for item in horizons
            ],
            "common_history_support_sha256": support_sha256,
        }
    )
    atomic_json_dump(fold_done, fold_root / "FOLD_DONE.json")
    return completed


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


def _prediction_metrics(
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


def refresh_summaries(output_dir: Path, config: dict[str, Any]) -> None:
    """Rebuild root tables solely from immutable per-cell artifacts."""

    expected_folds = list(config["folds_resolved"])
    experiments = list(config["experiments"])
    rows_by_experiment: dict[str, dict[str, dict[str, Any]]] = {
        str(item["experiment_id"]): {} for item in experiments
    }
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}

    for horizon in experiments:
        experiment = str(horizon["experiment_id"])
        group_rows: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed: list[str] = []
        for subject in expected_folds:
            root = task_root_for(output_dir, subject, horizon)
            done = core.validate_done(
                root / "DONE.json",
                stage="rf_classifier",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=f"{subject}/{experiment}",
            )
            if done is None:
                continue
            source_root = nbm_root_for(output_dir, subject, horizon)
            nbm_done = core.validate_done(
                source_root / "nbm" / "DONE.json",
                stage="nbm",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=_nbm_task_id(horizon),
            )
            if nbm_done is None:
                raise FileNotFoundError(source_root / "nbm" / "DONE.json")
            nbm_sha = str(nbm_done["artifacts"]["best"]["sha256"])
            residual_done = core.validate_done(
                source_root / "RESIDUAL_CACHE_DONE.json",
                stage="residual_cache",
                protocol_fingerprint=config["protocol_fingerprint"],
                task_id=_residual_task_id(horizon),
                upstream_sha256=nbm_sha,
            )
            if residual_done is None:
                raise FileNotFoundError(
                    source_root / "RESIDUAL_CACHE_DONE.json"
                )
            residual_sha = str(residual_done["artifacts"]["cache"]["sha256"])
            support_path = (
                output_dir / f"loso_{subject}" / "common_history_support.npz"
            )
            if not support_path.exists():
                raise FileNotFoundError(support_path)
            support_sha = sha256_file(support_path)
            if done.get("source_residual_sha256") != residual_sha:
                raise ValueError(f"Classifier source cache changed at {root}")
            if done.get("input_support_sha256") != support_sha:
                raise ValueError(f"Classifier common support changed at {root}")

            metrics = _load_json(root / "metrics.json")
            identity = {
                "experiment_id": experiment,
                "variant": experiment,
                "nbm": NBM_NAME,
                "input": INPUT_NAME,
                "test_subject": subject,
                "history_samples": HISTORY_SAMPLES,
                "history_blocks": int(horizon["history_blocks"]),
            }
            for key, expected in identity.items():
                if metrics.get(key) != expected:
                    raise ValueError(
                        f"Completed cell identity mismatch at {root}: "
                        f"{key}={metrics.get(key)!r}, expected={expected!r}"
                    )
            if metrics.get("source_residual_sha256") != residual_sha:
                raise ValueError(f"Metrics source cache changed at {root}")
            if metrics.get("input_support_sha256") != support_sha:
                raise ValueError(f"Metrics common support changed at {root}")

            with np.load(root / "predictions.npz", allow_pickle=False) as payload:
                required = {"window_index", "y_true", "y_prob", "y_pred"}
                if set(payload.files) != required:
                    raise ValueError(f"Unexpected prediction arrays at {root}")
                window_index = np.asarray(payload["window_index"], dtype=np.int64)
                y_true = np.asarray(payload["y_true"], dtype=np.int8)
                y_prob = np.asarray(payload["y_prob"], dtype=np.float64)
                y_pred = np.asarray(payload["y_pred"], dtype=np.int8)
            if not (
                window_index.ndim
                == y_true.ndim
                == y_prob.ndim
                == y_pred.ndim
                == 1
                and len(window_index)
                == len(y_true)
                == len(y_prob)
                == len(y_pred)
            ):
                raise ValueError(f"Misaligned prediction arrays at {root}")
            if not np.isfinite(y_prob).all():
                raise ValueError(f"Non-finite probabilities at {root}")

            enriched = {
                **metrics,
                "horizon_id": horizon["horizon_id"],
                "horizon_seconds": horizon["horizon_seconds"],
                "horizon_samples": horizon["horizon_samples"],
                "history_blocks": horizon["history_blocks"],
                "classifier": "tcn_m",
            }
            group_rows.append(enriched)
            fold_rows.append(enriched)
            rows_by_experiment[experiment][subject] = enriched
            truths.append(y_true)
            probabilities.append(y_prob)
            predictions.append(y_pred)
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
            "item": horizon,
            "completed": completed,
            "subject_macro": subject_macro,
            "pooled": (
                _prediction_metrics(
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
                "horizon_id": horizon["horizon_id"],
                "horizon_seconds": horizon["horizon_seconds"],
                "horizon_samples": horizon["horizon_samples"],
                "history_blocks": horizon["history_blocks"],
                "history_samples": HISTORY_SAMPLES,
                "input_shape": f"9x{HISTORY_SAMPLES}",
                "nbm": NBM_NAME,
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

    reference_item = next(
        (
            item
            for item in experiments
            if str(item["horizon_id"]) == "H050"
        ),
        None,
    )
    reference_id = (
        str(reference_item["experiment_id"]) if reference_item is not None else ""
    )
    delta_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []

    for horizon in experiments:
        experiment = str(horizon["experiment_id"])
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
                differences.append(
                    float(current_value) - float(reference_value)
                )
        delta = context_suite.paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            context_suite.stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                f"{experiment}__vs__{reference_id}",
            ),
        )
        delta_payload = {
            "experiment_id": experiment,
            "reference_experiment_id": reference_id,
            "reference_definition": "GRU-NBM H050 (0.5 s horizon)",
            "common_subjects": ",".join(common_subjects),
            **delta,
        }
        delta_rows.append(delta_payload)

        content = loaded[experiment]
        subject_macro = content["subject_macro"]
        aggregate[experiment] = {
            **horizon,
            "nbm": NBM_NAME,
            "input": INPUT_NAME,
            "classifier": config["classifier"],
            "completed_folds": content["completed"],
            "subject_macro": subject_macro,
            "pooled": content["pooled"],
            "delta_pr_auc_vs_h050": delta_payload,
        }
        numeric = {
            "experiment_id": experiment,
            "horizon_id": horizon["horizon_id"],
            "horizon_seconds": horizon["horizon_seconds"],
            "horizon_samples": horizon["horizon_samples"],
            "history_blocks": horizon["history_blocks"],
            "history_seconds": HISTORY_SECONDS,
            "history_samples": HISTORY_SAMPLES,
            "input_shape": f"9x{HISTORY_SAMPLES}",
            "nbm": NBM_NAME,
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
            "gru_parameter_count": config["gru_architectures"][
                str(horizon["horizon_id"])
            ]["parameter_count"],
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
            "Horizon": (
                f"{horizon['horizon_id']} "
                f"({float(horizon['horizon_seconds']):g} s)"
            ),
            "Residual blocks": int(horizon["history_blocks"]),
            "Classifier input": f"[9,{HISTORY_SAMPLES}]",
            "PR-AUC": _format_mean_sd(subject_macro, "pr_auc"),
            "ΔPR-AUC [95% CI]": _format_delta(delta_payload),
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
        "horizon_id",
        "horizon_seconds",
        "horizon_samples",
        "history_blocks",
        "classifier",
        "nbm",
        "input",
        "history_seconds",
        "history_samples",
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
            "horizon_id",
            "horizon_seconds",
            "horizon_samples",
            "history_blocks",
            "history_samples",
            "input_shape",
            "nbm",
            "classifier",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    summary_columns = [
        "experiment_id",
        "horizon_id",
        "horizon_seconds",
        "horizon_samples",
        "history_blocks",
        "history_seconds",
        "history_samples",
        "input_shape",
        "nbm",
        "classifier",
        "classifier_receptive_field_samples",
        "classifier_receptive_field_seconds",
        "classifier_parameter_count",
        "gru_parameter_count",
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
        "Horizon",
        "Residual blocks",
        "Classifier input",
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
            "reference": "H050 (0.5 s GRU-NBM horizon)",
            "method": "paired nonparametric bootstrap over held-out subjects",
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
    completed_fold_manifests = sum(
        int((output_dir / f"loso_{subject}" / "FOLD_DONE.json").exists())
        for subject in expected_folds
    )
    status = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_experiments": len(experiments),
        "expected_nbm_tasks": expected_cells,
        "expected_classifier_cells": expected_cells,
        "completed_classifier_cells": completed_cells,
        "expected_fold_manifests": len(expected_folds),
        "completed_fold_manifests": completed_fold_manifests,
        "status": "complete" if completed_cells == expected_cells else "partial",
        "best_experiment": aggregate_payload["best_experiment"],
    }
    atomic_json_dump(status, output_dir / "status.json")
    if completed_cells == expected_cells:
        results_done = done_payload(
            stage="horizon_suite_results",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id="root/results",
            relative_to=output_dir,
            artifacts={
                "config": (output_dir / "config.json").resolve(),
                "run_manifest": (output_dir / "run_manifest.json").resolve(),
                "fold_summary": (output_dir / "fold_summary.csv").resolve(),
                "experiment_manifest": (
                    output_dir / "experiment_manifest.csv"
                ).resolve(),
                "aggregate_summary": (
                    output_dir / "aggregate_summary.csv"
                ).resolve(),
                "paired_pr_auc_deltas": (
                    output_dir / "paired_pr_auc_deltas.csv"
                ).resolve(),
                "publication_table": (
                    output_dir / "publication_table.csv"
                ).resolve(),
                "aggregate_metrics": (
                    output_dir / "aggregate_metrics.json"
                ).resolve(),
                "status": (output_dir / "status.json").resolve(),
            },
        )
        atomic_json_dump(results_done, output_dir / "RESULTS_DONE.json")


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

    (
        config,
        dataset,
        master_windows,
        windows_by_horizon,
        classification_windows,
        horizons,
        folds,
    ) = load_dataset_and_protocol(args, device)
    if worker_mode:
        if (
            tuple(folds) != EXPECTED_LOSO_SUBJECTS
            or [item["horizon_id"] for item in horizons]
            != [item["horizon_id"] for item in HORIZON_DEFINITIONS]
            or config["protocol_scope"] != "strict_gru_horizon4_x_8_fold"
        ):
            raise ValueError(
                "Parallel workers require the complete strict 4x8 protocol"
            )
    execution_folds = list(folds)
    if worker_mode:
        selected = core.parse_folds(args.worker_fold, dataset.subjects)
        if len(selected) != 1:
            raise ValueError("--worker-fold must resolve to one subject")
        if selected[0] not in folds:
            raise ValueError(f"Worker fold {selected[0]} is outside {folds}")
        execution_folds = selected

    context_suite.initialise_or_validate_run(
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
        f"subjects={dataset.subjects} master_windows={len(master_windows)} "
        f"horizons={[item['horizon_id'] for item in horizons]} "
        f"context={CONTEXT_SECONDS:g}s history={HISTORY_SECONDS:g}s "
        f"label={FIXED_LABEL_SECONDS:g}s "
        f"classifier=TCN-M/RF{TCN_M_RF_SAMPLES} "
        f"configured_folds={folds} execution_folds={execution_folds}",
        flush=True,
    )

    if args.finalize_only:
        refresh_summaries(args.output_dir, config)
        print("[INFO] finalize-only: root summaries refreshed", flush=True)
        print(
            json.dumps(
                _load_json(args.output_dir / "status.json"),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    completed = 0
    for subject in execution_folds:
        completed += run_fold(
            args,
            config,
            dataset,
            master_windows,
            windows_by_horizon,
            classification_windows,
            horizons,
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
