#!/usr/bin/env python
"""Transformer-NBM horizon-by-fusion ablation for Daphnet FoG detection.

The strict suite contains nine TCN-M inputs:

* three shared controls: Raw4, Raw6, and Raw4 + Zero;
* signed forecast error for H050, H100, and H200;
* Raw4 + signed forecast error for H050, H100, and H200.

Every arm uses the same LOSO split, fold-only robust scaler, common decision
endpoints, and labels from the terminal 0.5 seconds.  A two-second-horizon
master WindowTable defines validity and clean-normal NBM eligibility.  The
four-second diagnostic histories contain 8/4/2 non-overlapping forecast blocks
for H050/H100/H200 respectively, and therefore always have shape [N, 9, 256].
"""

from __future__ import annotations

import argparse
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
import run_daphnet_gru_horizon_ablation as horizon_base
import run_daphnet_nbm_context_tcnm_suite as context_suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, RobustChannelScaler, WindowTable
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.histories import (
    HistoryPlan,
    make_block_history_input,
    materialize_nonoverlap_residual_history,
)
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    dataset_fingerprint,
    done_payload,
    sha256_file,
    validate_done,
)


SUITE_VERSION = "daphnet_transformer_horizon_fusion_h4_tcnm_loso.v1"
NBM_NAME = "transformer"
SAMPLING_RATE_HZ = 64
CONTEXT_SECONDS = 2.0
CONTEXT_SAMPLES = 128
HISTORY_SECONDS = 4.0
HISTORY_SAMPLES = 256
RAW6_SECONDS = 6.0
RAW6_SAMPLES = 384
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

INPUT_VARIANTS = (
    "raw4",
    "raw6",
    "raw4_zero",
    "error_h050",
    "raw4_error_h050",
    "error_h100",
    "raw4_error_h100",
    "error_h200",
    "raw4_error_h200",
)

INPUT_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "cell_id": "raw4",
        "display_name": "Raw4",
        "input_variant": "raw4",
        "kind": "control",
        "horizon_id": "shared",
        "in_channels": 9,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 8,
        "formula": "terminal 4 s robust-scaled raw IMU",
    },
    {
        "cell_id": "raw6",
        "display_name": "Raw6",
        "input_variant": "raw6",
        "kind": "control",
        "horizon_id": "shared",
        "in_channels": 9,
        "history_seconds": RAW6_SECONDS,
        "history_samples": RAW6_SAMPLES,
        "history_blocks": 1,
        "formula": "terminal 6 s robust-scaled raw IMU",
    },
    {
        "cell_id": "raw4_zero",
        "display_name": "Raw4 + Zero",
        "input_variant": "raw4_zero",
        "kind": "control",
        "horizon_id": "shared",
        "in_channels": 18,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 8,
        "formula": "cat(raw4, zeros_like(raw4))",
    },
    {
        "cell_id": "error_h050",
        "display_name": "Error H050",
        "input_variant": "error_h050",
        "kind": "error",
        "horizon_id": "H050",
        "in_channels": 9,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 8,
        "formula": "x - mu_H050",
    },
    {
        "cell_id": "raw4_error_h050",
        "display_name": "Raw4 + Error H050",
        "input_variant": "raw4_error_h050",
        "kind": "fusion",
        "horizon_id": "H050",
        "in_channels": 18,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 8,
        "formula": "cat(raw4, x - mu_H050)",
    },
    {
        "cell_id": "error_h100",
        "display_name": "Error H100",
        "input_variant": "error_h100",
        "kind": "error",
        "horizon_id": "H100",
        "in_channels": 9,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 4,
        "formula": "x - mu_H100",
    },
    {
        "cell_id": "raw4_error_h100",
        "display_name": "Raw4 + Error H100",
        "input_variant": "raw4_error_h100",
        "kind": "fusion",
        "horizon_id": "H100",
        "in_channels": 18,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 4,
        "formula": "cat(raw4, x - mu_H100)",
    },
    {
        "cell_id": "error_h200",
        "display_name": "Error H200",
        "input_variant": "error_h200",
        "kind": "error",
        "horizon_id": "H200",
        "in_channels": 9,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 2,
        "formula": "x - mu_H200",
    },
    {
        "cell_id": "raw4_error_h200",
        "display_name": "Raw4 + Error H200",
        "input_variant": "raw4_error_h200",
        "kind": "fusion",
        "horizon_id": "H200",
        "in_channels": 18,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": 2,
        "formula": "cat(raw4, x - mu_H200)",
    },
)


def _comparison(
    comparison_id: str,
    new: str,
    reference: str,
    interpretation: str,
) -> dict[str, str]:
    return {
        "comparison_id": comparison_id,
        "new": new,
        "reference": reference,
        "interpretation": interpretation,
    }


COMPARISONS: tuple[dict[str, str], ...] = tuple(
    comparison
    for horizon_id in ("h050", "h100", "h200")
    for comparison in (
        _comparison(
            f"raw4_error_{horizon_id}_minus_raw4_zero",
            f"raw4_error_{horizon_id}",
            "raw4_zero",
            "error information beyond matched 18-channel capacity",
        ),
        _comparison(
            f"raw4_error_{horizon_id}_minus_raw4",
            f"raw4_error_{horizon_id}",
            "raw4",
            "practical fusion gain over four-second raw input",
        ),
        _comparison(
            f"raw4_error_{horizon_id}_minus_raw6",
            f"raw4_error_{horizon_id}",
            "raw6",
            "fusion gain beyond exposing the complete six-second raw support",
        ),
        _comparison(
            f"error_{horizon_id}_minus_raw4",
            f"error_{horizon_id}",
            "raw4",
            "whether signed forecast error can replace raw input",
        ),
    )
) + (
    _comparison(
        "error_h100_minus_error_h050",
        "error_h100",
        "error_h050",
        "one-second versus half-second signed error",
    ),
    _comparison(
        "error_h200_minus_error_h050",
        "error_h200",
        "error_h050",
        "two-second versus half-second signed error",
    ),
    _comparison(
        "raw4_error_h100_minus_raw4_error_h050",
        "raw4_error_h100",
        "raw4_error_h050",
        "one-second versus half-second fusion",
    ),
    _comparison(
        "raw4_error_h200_minus_raw4_error_h050",
        "raw4_error_h200",
        "raw4_error_h050",
        "two-second versus half-second fusion",
    ),
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
    "scripts/run_daphnet_transformer_horizon_fusion_ablation.py",
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
    REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daphnet_transformer_horizon_fusion_h4_tcnm_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict Transformer H050/H100/H200 error/fusion ablation with "
            "Raw4, Raw6, and Raw4+Zero controls"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--worker-fold", default="")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--exclude-subjects", default="S04,S10")
    parser.add_argument("--horizons", default="H050,H100,H200")
    parser.add_argument("--inputs", default=",".join(INPUT_VARIANTS))

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
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ffn", type=int, default=128)
    # Shared NBM factory arguments unused by this Transformer-only suite.
    parser.add_argument("--linear-ar-seconds", type=float, default=0.5)
    parser.add_argument("--gru-layers", type=int, default=1)
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
    parser.add_argument("--allow-protocol-subset", action="store_true")
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
    if int(sampling_rate_hz) != SAMPLING_RATE_HZ:
        raise ValueError("This suite requires 64 Hz Daphnet data")
    definitions = {
        str(item["horizon_id"]): dict(item) for item in HORIZON_DEFINITIONS
    }
    aliases: dict[str, str] = {}
    for item in HORIZON_DEFINITIONS:
        horizon_id = str(item["horizon_id"])
        seconds = float(item["horizon_seconds"])
        aliases[horizon_id] = horizon_id
        aliases[f"{seconds:g}"] = horizon_id
        aliases[f"{seconds:.2f}"] = horizon_id
    requested: set[str] = set()
    for raw in str(specification).split(","):
        token = raw.strip()
        if not token:
            continue
        horizon_id = aliases.get(token.upper()) or aliases.get(token)
        if horizon_id is None:
            raise ValueError(f"Unknown horizon {token!r}; use H050,H100,H200")
        requested.add(horizon_id)
    if not requested:
        raise ValueError("At least one horizon is required")
    result = [
        definitions[str(item["horizon_id"])]
        for item in HORIZON_DEFINITIONS
        if str(item["horizon_id"]) in requested
    ]
    for item in result:
        samples = int(item["horizon_samples"])
        if samples != round(float(item["horizon_seconds"]) * sampling_rate_hz):
            raise AssertionError(f"Invalid horizon definition: {item}")
        if HISTORY_SAMPLES % samples:
            raise AssertionError(f"Horizon does not divide four seconds: {item}")
        if HISTORY_SAMPLES // samples != int(item["history_blocks"]):
            raise AssertionError(f"Invalid history block count: {item}")
    return result


def parse_inputs(specification: str) -> list[str]:
    requested: list[str] = []
    for raw in str(specification).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token not in INPUT_VARIANTS:
            raise ValueError(
                f"Unknown input {token!r}; use {','.join(INPUT_VARIANTS)}"
            )
        if token not in requested:
            requested.append(token)
    if not requested:
        raise ValueError("At least one input is required")
    return [name for name in INPUT_VARIANTS if name in set(requested)]


def horizon_directory(horizon: Mapping[str, Any]) -> str:
    duration = f"{float(horizon['horizon_seconds']):g}".replace(".", "p")
    return f"horizon_{str(horizon['horizon_id']).lower()}_{duration}s"


def experiment_id(cell_id: str) -> str:
    return f"transformer_horizon_fusion__{cell_id}"


def _aligned_classifier_states(
    seed: int,
    hidden_channels: int,
    dropout: float,
    deterministic: bool,
) -> tuple[
    dict[int, dict[str, torch.Tensor]],
    dict[int, int],
    dict[int, str],
    str,
]:
    """Create 9/18-channel TCN states from one 18-channel template."""

    rf.set_seed(seed, deterministic)
    template = rf.build_model(
        in_channels=18,
        hidden_channels=hidden_channels,
        dropout=dropout,
        dilations=TCN_M_DILATIONS,
    )
    template_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    del template
    states: dict[int, dict[str, torch.Tensor]] = {}
    counts: dict[int, int] = {}
    hashes: dict[int, str] = {}
    for in_channels in (9, 18):
        model = rf.build_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            dilations=TCN_M_DILATIONS,
        )
        state: dict[str, torch.Tensor] = {}
        for name, target in model.state_dict().items():
            source = template_state[name]
            if name == "projection.0.weight":
                source = source[:, :in_channels, :]
            if source.shape != target.shape:
                raise AssertionError(
                    f"Cannot share TCN-M state {name}: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            state[name] = source.clone()
        model.load_state_dict(state, strict=True)
        states[in_channels] = state
        counts[in_channels] = rf.parameter_count(model)
        hashes[in_channels] = rf.state_dict_sha256(model.state_dict())
        del model
    shared_backbone = {
        name: value
        for name, value in template_state.items()
        if name != "projection.0.weight"
    }
    return states, counts, hashes, rf.state_dict_sha256(shared_backbone)


aligned_classifier_states = _aligned_classifier_states


def experiment_grid(
    args: argparse.Namespace,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
    selected_inputs: list[str] | None = None,
) -> list[dict[str, Any]]:
    if rf.convolutional_receptive_field(TCN_M_DILATIONS) != TCN_M_RF_SAMPLES:
        raise AssertionError("TCN-M receptive field changed")
    selected = set(INPUT_VARIANTS if selected_inputs is None else selected_inputs)
    _, counts, hashes, _ = _aligned_classifier_states(
        int(args.seed),
        int(args.classifier_hidden),
        float(args.classifier_dropout),
        bool(args.deterministic),
    )
    result: list[dict[str, Any]] = []
    for definition in INPUT_DEFINITIONS:
        cell_id = str(definition["cell_id"])
        if cell_id not in selected:
            continue
        channels = int(definition["in_channels"])
        result.append(
            {
                **definition,
                "variant": cell_id,
                "input": cell_id,
                "experiment_id": experiment_id(cell_id),
                "nbm": NBM_NAME,
                "classifier": "tcn_m",
                "dilations": list(TCN_M_DILATIONS),
                "kernel_size": 3,
                "n_blocks": len(TCN_M_DILATIONS),
                "convolutions_per_block": 2,
                "receptive_field_samples": TCN_M_RF_SAMPLES,
                "receptive_field_seconds": (
                    TCN_M_RF_SAMPLES / float(sampling_rate_hz)
                ),
                "parameter_count": counts[channels],
                "template_initial_state_sha256": hashes[channels],
                "input_shape": (
                    f"{channels}x{int(definition['history_samples'])}"
                ),
            }
        )
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if not args.cache_residuals:
        raise ValueError("Primitive caches are required for resume and audit")
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
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ffn": args.transformer_ffn,
        "classifier_hidden": args.classifier_hidden,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These integer options must be positive: {invalid}")
    if args.nbm_hidden % args.transformer_heads:
        raise ValueError("--nbm-hidden must be divisible by --transformer-heads")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_normal_windows < 0 or args.max_classifier_windows < 0:
        raise ValueError("Window caps must be non-negative")
    if args.stop_after_completed_tasks < 0:
        raise ValueError("--stop-after-completed-tasks must be non-negative")
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


# Stable public helpers used by tests and the independent auditor.
array_sha256 = horizon_base.array_sha256
window_table_sha256 = horizon_base.window_table_sha256
fixed_endpoint_labels = horizon_base.fixed_endpoint_labels
relabel_master_windows = horizon_base.relabel_master_windows
derive_horizon_windows = horizon_base.derive_horizon_windows
derive_classification_windows = horizon_base.derive_classification_windows
build_common_history_support = horizon_base.build_common_history_support


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def _transformer_architectures(
    args: argparse.Namespace,
    horizons: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], str]:
    architectures: dict[str, dict[str, Any]] = {}
    shared_hashes: list[str] = []
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
        shared_state = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if not name.startswith("decoder.")
        }
        shared_hash = rf.state_dict_sha256(shared_state)
        shared_hashes.append(shared_hash)
        decoder_parameters = sum(
            int(parameter.numel())
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.")
        )
        architectures[str(horizon["horizon_id"])] = {
            "model_config": model.model_config(),
            "parameter_count": core.parameter_count(model),
            "decoder_parameter_count": decoder_parameters,
            "shared_encoder_parameter_count": (
                core.parameter_count(model) - decoder_parameters
            ),
            "initial_shared_encoder_sha256": shared_hash,
        }
        del model
    if len(set(shared_hashes)) != 1:
        raise AssertionError("Transformer shared initialization differs by horizon")
    return architectures, shared_hashes[0]


def _validate_protocol_selection(
    horizons: list[dict[str, Any]],
    inputs: list[str],
    folds: list[str],
    allow_subset: bool,
) -> None:
    if allow_subset:
        return
    if [str(item["horizon_id"]) for item in horizons] != [
        str(item["horizon_id"]) for item in HORIZON_DEFINITIONS
    ]:
        raise ValueError("Strict protocol requires H050,H100,H200")
    if tuple(inputs) != INPUT_VARIANTS:
        raise ValueError("Strict protocol requires all nine input variants")
    if tuple(folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("Strict protocol requires --folds all")


def build_protocol(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    source_subjects: list[str],
    excluded_subjects: list[str],
    folds: list[str],
    horizons: list[dict[str, Any]],
    selected_inputs: list[str],
    master_windows: WindowTable,
    windows_by_horizon: Mapping[str, WindowTable],
    classification_windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    cells = experiment_grid(args, dataset.sampling_rate_hz, selected_inputs)
    architectures, shared_transformer_hash = _transformer_architectures(
        args,
        horizons,
        int(args.seed),
    )
    _, counts, hashes, backbone_hash = _aligned_classifier_states(
        int(args.seed),
        int(args.classifier_hidden),
        float(args.classifier_dropout),
        bool(args.deterministic),
    )
    scientific = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "channel_names": list(dataset.channel_names),
        "n_channels": int(dataset.n_channels),
        "source_subjects": source_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects": list(dataset.subjects),
        "folds_resolved": folds,
        "nbm": NBM_NAME,
        "horizon_variants": horizons,
        "input_variants": selected_inputs,
        "experiments": cells,
        "comparisons": list(COMPARISONS),
        "context_seconds": CONTEXT_SECONDS,
        "context_samples": CONTEXT_SAMPLES,
        "support_horizon_seconds": SUPPORT_HORIZON_SECONDS,
        "support_horizon_samples": SUPPORT_HORIZON_SAMPLES,
        "fixed_label_seconds": FIXED_LABEL_SECONDS,
        "fixed_label_samples": FIXED_LABEL_SAMPLES,
        "stride_seconds": STRIDE_SECONDS,
        "stride_samples": STRIDE_SAMPLES,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "raw6_seconds": RAW6_SECONDS,
        "raw6_samples": RAW6_SAMPLES,
        "master_window_count": int(len(master_windows)),
        "master_window_sha256": window_table_sha256(master_windows),
        "derived_window_sha256": {
            key: window_table_sha256(value)
            for key, value in windows_by_horizon.items()
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
            "A 2 s context + maximum 2 s horizon + normal guard defines "
            "validity and clean-normal eligibility for every horizon."
        ),
        "classification_label_policy": (
            "Every arm uses the same label from the final 32 samples at the "
            "common decision endpoint."
        ),
        "common_anchor_policy": (
            "Complete per-horizon four-second plans are intersected by global "
            "WindowTable row ID inside each split."
        ),
        "history_construction": (
            "Horizon-spaced non-overlapping blocks cover the terminal four "
            "seconds; Raw6 covers the earliest NBM context plus that history."
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
        "transformer_heads": int(args.transformer_heads),
        "transformer_layers": int(args.transformer_layers),
        "transformer_ffn": int(args.transformer_ffn),
        "transformer_architectures": architectures,
        "shared_initial_transformer_encoder_sha256": shared_transformer_hash,
        "classifier": {
            "name": "TCN-M",
            "hidden_channels": int(args.classifier_hidden),
            "dropout": float(args.classifier_dropout),
            "kernel_size": 3,
            "convolutions_per_block": 2,
            "dilations": list(TCN_M_DILATIONS),
            "receptive_field_samples": TCN_M_RF_SAMPLES,
            "receptive_field_seconds": (
                TCN_M_RF_SAMPLES / float(dataset.sampling_rate_hz)
            ),
            "parameter_count_by_in_channels": {
                str(key): value for key, value in counts.items()
            },
            "template_initial_state_sha256_by_in_channels": {
                str(key): value for key, value in hashes.items()
            },
            "shared_backbone_initial_state_sha256": backbone_hash,
            "fold_initialization_seed_rule": "seed + 10000 + fold_index",
            "global_pooling": "mean_and_max_over_full_input",
        },
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
        "bootstrap_unit": "held_out_subject",
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "expected_experiments": len(cells),
        "expected_nbm_tasks": len(horizons) * len(folds),
        "expected_classifier_cells": len(cells) * len(folds),
        "protocol_scope": (
            "development_subset"
            if args.allow_protocol_subset
            else "strict_transformer_horizon3_fusion9_x_8_fold"
        ),
        "fairness_contract": {
            "ablation_axis": "transformer_forecast_horizon_and_input_fusion",
            "same_master_clean_normal_support": True,
            "same_common_anchor_ids_and_labels": True,
            "same_fold_scaler": True,
            "same_tcn_backbone_initialization": True,
            "projection_prefix_slice_for_9_channels": True,
            "same_epoch_shuffle_seed_rule": True,
            "validation_only_early_stopping_and_threshold": True,
            "test_subject_excluded_from_all_model_selection": True,
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
            f"Expected subjects {EXPECTED_LOSO_SUBJECTS}, got {dataset.subjects}"
        )
    if int(dataset.sampling_rate_hz) != SAMPLING_RATE_HZ:
        raise ValueError("This suite requires Daphnet at 64 Hz")
    horizons = parse_horizons(args.horizons, dataset.sampling_rate_hz)
    selected_inputs = parse_inputs(args.inputs)
    selected_horizon_ids = {
        str(item["horizon_id"]) for item in horizons
    }
    required_horizon_ids = {
        str(item["horizon_id"])
        for item in INPUT_DEFINITIONS
        if item["cell_id"] in selected_inputs
        and item["horizon_id"] != "shared"
    }
    missing_horizons = sorted(required_horizon_ids - selected_horizon_ids)
    if missing_horizons:
        raise ValueError(
            "Selected inputs require missing horizons: "
            f"{missing_horizons}"
        )
    folds = core.parse_folds(args.folds, dataset.subjects)
    _validate_protocol_selection(
        horizons,
        selected_inputs,
        folds,
        args.allow_protocol_subset,
    )
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
    master_windows = relabel_master_windows(
        dataset,
        raw_master,
        FIXED_LABEL_SAMPLES,
        args.fog_fraction_threshold,
    )
    windows_by_horizon = {
        str(item["horizon_id"]): derive_horizon_windows(
            master_windows,
            int(item["horizon_samples"]),
            FIXED_LABEL_SAMPLES,
        )
        for item in horizons
    }
    classification_windows = derive_classification_windows(master_windows)
    config = build_protocol(
        args,
        dataset,
        source_subjects,
        excluded_subjects,
        folds,
        horizons,
        selected_inputs,
        master_windows,
        windows_by_horizon,
        classification_windows,
        data_sha256,
        device,
    )
    return (
        config,
        dataset,
        master_windows,
        windows_by_horizon,
        classification_windows,
        horizons,
        selected_inputs,
        folds,
    )


def task_root_for(
    output_dir: Path,
    subject: str,
    cell: str | Mapping[str, Any],
) -> Path:
    cell_id = str(cell if isinstance(cell, str) else cell["cell_id"])
    return output_dir / f"loso_{subject}" / cell_id


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
    RobustChannelScaler,
    dict[str, np.ndarray],
    dict[str, dict[str, HistoryPlan]],
    np.ndarray,
    np.ndarray,
    str,
    dict[str, Any],
]:
    """Create one leakage-safe fold and common horizon history support."""

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
        if np.unique(
            classification_windows.label[candidate_indices]
        ).size == 2:
            val_subject = candidate
            break
    if not val_subject:
        raise RuntimeError("No validation subject contains both classes")
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
        rows = np.arange(len(reference_plan.anchor_rows), dtype=np.int64)
        labels = classification_windows.label[
            reference_plan.anchor_window_indices
        ]
        selected = core.deterministic_subsample(
            rows,
            args.max_classifier_windows,
            args.seed + 100 + fold_index,
            labels,
        )
        for horizon_id in horizon_ids:
            plans[horizon_id]["train"] = plans[horizon_id]["train"].take(
                selected
            )

    for split in ("train", "validation", "test"):
        reference_anchor = plans[reference][split].anchor_window_indices
        reference_labels = classification_windows.label[reference_anchor]
        if np.unique(reference_labels).size != 2:
            raise RuntimeError(
                f"Common {split} support lacks a class in {test_subject}"
            )
        for horizon_id in horizon_ids:
            plan = plans[horizon_id][split]
            if not np.array_equal(
                plan.anchor_window_indices,
                reference_anchor,
            ):
                raise AssertionError("Horizon plans do not share endpoints")
            if not np.array_equal(
                windows_by_horizon[horizon_id].label[
                    plan.anchor_window_indices
                ],
                reference_labels,
            ):
                raise AssertionError("Horizon labels differ")

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

    classifier_seed = int(args.seed) + 10000 + fold_index
    _, counts, hashes, backbone_hash = _aligned_classifier_states(
        classifier_seed,
        int(args.classifier_hidden),
        float(args.classifier_dropout),
        bool(args.deterministic),
    )
    partial_fold_config: dict[str, Any] = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "excluded_subjects": config["excluded_subjects"],
        "scaler": scaler.as_dict(),
        "scaler_sha256": sha256_file(fold_root / "scaler.json"),
        "split_indices_sha256": sha256_file(
            fold_root / "split_indices.npz"
        ),
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
            split: array_sha256(
                plans[reference][split].anchor_window_indices
            )
            for split in ("train", "validation", "test")
        },
        "per_horizon_history_support_sha256": {
            horizon_id: {
                split: array_sha256(
                    support_arrays[
                        f"{split}_{horizon_id.lower()}_history_window_index"
                    ]
                )
                for split in ("train", "validation", "test")
            }
            for horizon_id in horizon_ids
        },
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256_by_in_channels": {
            str(key): value for key, value in hashes.items()
        },
        "parameter_count_by_in_channels": {
            str(key): value for key, value in counts.items()
        },
        "shared_backbone_initial_state_sha256": backbone_hash,
        "label_window_samples": FIXED_LABEL_SAMPLES,
        "classification_window_sha256": config[
            "classification_window_sha256"
        ],
    }
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
        partial_fold_config,
    )


@torch.no_grad()
def extract_transformer_primitives(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract robust-scaled target, signed error, and conditional sigma."""

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
    raw_chunks: list[np.ndarray] = []
    error_chunks: list[np.ndarray] = []
    sigma_chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        target = sequence[:, :, CONTEXT_SAMPLES:]
        with torch.amp.autocast(
            device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            mean, sigma = model(context)
        raw_chunks.append(target.float().cpu().numpy())
        error_chunks.append((target - mean.float()).cpu().numpy())
        sigma_chunks.append(sigma.float().cpu().numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())
    features = {
        "raw": np.ascontiguousarray(
            np.concatenate(raw_chunks).astype(np.float32, copy=False)
        ),
        "error": np.ascontiguousarray(
            np.concatenate(error_chunks).astype(np.float32, copy=False)
        ),
        "sigma": np.ascontiguousarray(
            np.concatenate(sigma_chunks).astype(np.float32, copy=False)
        ),
        "y": np.concatenate(labels).astype(np.int8, copy=False),
        "window_index": np.concatenate(window_indices).astype(
            np.int64,
            copy=False,
        ),
    }
    for key in ("raw", "error", "sigma"):
        if not np.isfinite(features[key]).all():
            raise ValueError(f"Non-finite Transformer primitive {key}")
    if np.any(features["sigma"] <= 0):
        raise ValueError("Transformer sigma must be positive")
    error64 = features["error"].astype(np.float64)
    diagnostics = {
        "windows": int(len(features["y"])),
        "class_counts": np.bincount(
            features["y"],
            minlength=2,
        ).astype(int).tolist(),
        "forecast_rmse": float(np.sqrt(np.mean(np.square(error64)))),
        "forecast_mae": float(np.mean(np.abs(error64))),
        "mean_sigma": float(
            features["sigma"].astype(np.float64).mean()
        ),
        "error_abs_mean": float(np.abs(error64).mean()),
        "error_rms": float(np.sqrt(np.mean(np.square(error64)))),
    }
    return features, diagnostics


def _primitive_cache_task_id(horizon: Mapping[str, Any]) -> str:
    return f"{horizon_directory(horizon)}/{NBM_NAME}/primitive_cache"


def load_or_extract_primitive_cache(
    args: argparse.Namespace,
    config: dict[str, Any],
    horizon: Mapping[str, Any],
    nbm_root: Path,
    nbm_sha256: str,
    model: torch.nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    split_indices: Mapping[str, np.ndarray],
    scaler: RobustChannelScaler,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], str]:
    cache_path = nbm_root / "transformer_primitives.npz"
    diagnostics_path = nbm_root / "primitive_diagnostics.json"
    done_path = nbm_root / "PRIMITIVE_CACHE_DONE.json"
    task_id = _primitive_cache_task_id(horizon)
    completed = validate_done(
        done_path,
        stage="transformer_primitive_cache",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    expected_keys = {
        f"{split}_{key}"
        for split in ("train", "validation", "test")
        for key in ("raw", "error", "sigma", "y", "window_index")
    }
    if completed is not None:
        with np.load(cache_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"Unexpected primitive arrays: {cache_path}")
            features = {
                split: {
                    key: np.asarray(payload[f"{split}_{key}"])
                    for key in ("raw", "error", "sigma", "y", "window_index")
                }
                for split in ("train", "validation", "test")
            }
        return features, _load_json(diagnostics_path), sha256_file(cache_path)

    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        features[split], diagnostics[split] = extract_transformer_primitives(
            args,
            model,
            dataset,
            windows,
            np.asarray(split_indices[split], dtype=np.int64),
            scaler,
            device,
        )
    atomic_npz_save(
        cache_path,
        **{
            f"{split}_{key}": features[split][key]
            for split in ("train", "validation", "test")
            for key in ("raw", "error", "sigma", "y", "window_index")
        },
    )
    atomic_json_dump(diagnostics, diagnostics_path)
    atomic_json_dump(
        done_payload(
            stage="transformer_primitive_cache",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=nbm_sha256,
            relative_to=nbm_root,
            artifacts={
                "cache": cache_path.resolve(),
                "diagnostics": diagnostics_path.resolve(),
            },
        ),
        done_path,
    )
    return features, diagnostics, sha256_file(cache_path)


def materialize_error_history(
    features: Mapping[str, np.ndarray],
    plan: HistoryPlan,
    horizon_samples: int,
    name: str,
) -> dict[str, np.ndarray]:
    payload = make_block_history_input(
        extracted=features,
        plan=plan,
        source_key="error",
        name=name,
        history_samples=HISTORY_SAMPLES,
        horizon_samples=int(horizon_samples),
        stride_samples=STRIDE_SAMPLES,
    )
    expected = (len(plan.anchor_rows), 9, HISTORY_SAMPLES)
    if tuple(payload[name].shape) != expected:
        raise AssertionError(
            f"Unexpected error history {payload[name].shape} != {expected}"
        )
    return payload


def materialize_raw4_history(
    features: Mapping[str, np.ndarray],
    plan: HistoryPlan,
    horizon_samples: int,
    name: str = "raw4",
) -> dict[str, np.ndarray]:
    payload = make_block_history_input(
        extracted=features,
        plan=plan,
        source_key="raw",
        name=name,
        history_samples=HISTORY_SAMPLES,
        horizon_samples=int(horizon_samples),
        stride_samples=STRIDE_SAMPLES,
    )
    expected = (len(plan.anchor_rows), 9, HISTORY_SAMPLES)
    if tuple(payload[name].shape) != expected:
        raise AssertionError(
            f"Unexpected Raw4 history {payload[name].shape} != {expected}"
        )
    return payload


def materialize_raw6_history(
    dataset: DaphnetDataset,
    classification_windows: WindowTable,
    anchor_indices: np.ndarray,
    scaler: RobustChannelScaler,
    name: str = "raw6",
) -> dict[str, np.ndarray]:
    """Extract the exact six-second causal raw union ending at each anchor."""

    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    values = np.empty(
        (len(anchor_indices), dataset.n_channels, RAW6_SAMPLES),
        dtype=np.float32,
    )
    for row, window_index in enumerate(anchor_indices):
        record_index = int(classification_windows.record_index[window_index])
        end = int(classification_windows.target_end[window_index])
        start = end - RAW6_SAMPLES
        if start < 0:
            raise ValueError("Raw6 starts before its record")
        record = dataset.records[record_index]
        if not bool(record.valid[start:end].all()):
            raise ValueError("Raw6 includes invalid samples")
        scaled = scaler.transform(record.x[start:end])
        if scaled.shape != (RAW6_SAMPLES, dataset.n_channels):
            raise AssertionError("Unexpected Raw6 source shape")
        values[row] = scaled.T
    if not np.isfinite(values).all():
        raise ValueError("Raw6 contains NaN or Inf")
    return {
        name: np.ascontiguousarray(values),
        "y": classification_windows.label[anchor_indices].astype(
            np.int8,
            copy=False,
        ),
        "window_index": anchor_indices,
    }


def materialize_fold_inputs(
    config: Mapping[str, Any],
    dataset: DaphnetDataset,
    classification_windows: WindowTable,
    horizons: list[dict[str, Any]],
    plans: Mapping[str, Mapping[str, HistoryPlan]],
    primitives: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    scaler: RobustChannelScaler,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, str],
]:
    """Build all requested controls/errors/fusions on common endpoints."""

    horizon_by_id = {
        str(item["horizon_id"]): item for item in horizons
    }
    reference_id = str(horizons[0]["horizon_id"])
    inputs_by_cell: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    raw4_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        raw4_by_split[split] = materialize_raw4_history(
            primitives[reference_id][split],
            plans[reference_id][split],
            int(horizon_by_id[reference_id]["horizon_samples"]),
            "raw4",
        )
        for horizon_id, horizon in horizon_by_id.items():
            comparison = materialize_raw4_history(
                primitives[horizon_id][split],
                plans[horizon_id][split],
                int(horizon["horizon_samples"]),
                "raw4",
            )
            if not np.array_equal(
                comparison["window_index"],
                raw4_by_split[split]["window_index"],
            ):
                raise AssertionError("Raw4 anchors differ across horizons")
            if not np.array_equal(
                comparison["y"],
                raw4_by_split[split]["y"],
            ):
                raise AssertionError("Raw4 labels differ across horizons")
            if not np.array_equal(
                comparison["raw4"],
                raw4_by_split[split]["raw4"],
            ):
                maximum = float(
                    np.max(
                        np.abs(
                            comparison["raw4"].astype(np.float64)
                            - raw4_by_split[split]["raw4"].astype(np.float64)
                        )
                    )
                )
                raise AssertionError(
                    f"Raw4 values differ across horizons: {maximum}"
                )

    selected = set(config["input_variants"])
    if "raw4" in selected:
        inputs_by_cell["raw4"] = {
            split: {
                "raw4": raw4_by_split[split]["raw4"],
                "y": raw4_by_split[split]["y"],
                "window_index": raw4_by_split[split]["window_index"],
            }
            for split in ("train", "validation", "test")
        }
    if "raw6" in selected:
        inputs_by_cell["raw6"] = {
            split: materialize_raw6_history(
                dataset,
                classification_windows,
                plans[reference_id][split].anchor_window_indices,
                scaler,
                "raw6",
            )
            for split in ("train", "validation", "test")
        }
    if "raw4_zero" in selected:
        inputs_by_cell["raw4_zero"] = {
            split: {
                "raw4_zero": np.ascontiguousarray(
                    np.concatenate(
                        (
                            raw4_by_split[split]["raw4"],
                            np.zeros_like(raw4_by_split[split]["raw4"]),
                        ),
                        axis=1,
                        dtype=np.float32,
                    )
                ),
                "y": raw4_by_split[split]["y"],
                "window_index": raw4_by_split[split]["window_index"],
            }
            for split in ("train", "validation", "test")
        }

    for horizon_id, horizon in horizon_by_id.items():
        suffix = horizon_id.lower()
        error_name = f"error_{suffix}"
        fusion_name = f"raw4_error_{suffix}"
        error_by_split = {
            split: materialize_error_history(
                primitives[horizon_id][split],
                plans[horizon_id][split],
                int(horizon["horizon_samples"]),
                error_name,
            )
            for split in ("train", "validation", "test")
        }
        if error_name in selected:
            inputs_by_cell[error_name] = error_by_split
        if fusion_name in selected:
            inputs_by_cell[fusion_name] = {}
            for split in ("train", "validation", "test"):
                if not np.array_equal(
                    error_by_split[split]["window_index"],
                    raw4_by_split[split]["window_index"],
                ):
                    raise AssertionError("Raw/error fusion anchors differ")
                inputs_by_cell[fusion_name][split] = {
                    fusion_name: np.ascontiguousarray(
                        np.concatenate(
                            (
                                raw4_by_split[split]["raw4"],
                                error_by_split[split][error_name],
                            ),
                            axis=1,
                            dtype=np.float32,
                        )
                    ),
                    "y": error_by_split[split]["y"],
                    "window_index": error_by_split[split]["window_index"],
                }

    fingerprints: dict[str, str] = {}
    definition_by_id = {
        str(item["cell_id"]): item for item in INPUT_DEFINITIONS
    }
    for cell_id, split_payloads in inputs_by_cell.items():
        fingerprints[cell_id] = canonical_fingerprint(
            {
                "cell_id": cell_id,
                "definition": definition_by_id[cell_id],
                "common_history_support_sha256": config.get(
                    "_fold_common_support_sha256",
                    "",
                ),
                "splits": {
                    split: {
                        "input_sha256": array_sha256(payload[cell_id]),
                        "y_sha256": array_sha256(payload["y"]),
                        "window_index_sha256": array_sha256(
                            payload["window_index"]
                        ),
                    }
                    for split, payload in split_payloads.items()
                },
            }
        )
    if set(inputs_by_cell) != selected:
        raise AssertionError(
            f"Materialized inputs {sorted(inputs_by_cell)} != {sorted(selected)}"
        )
    return inputs_by_cell, fingerprints


def build_input_fingerprints(
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    primitive_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Bind each classifier input to immutable upstream artifacts and formula."""

    definitions = {
        str(item["cell_id"]): item for item in INPUT_DEFINITIONS
    }
    support_sha = str(fold_config["common_history_support_sha256"])
    raw4_source_horizon = str(config["horizon_variants"][0]["horizon_id"])
    raw4_binding = {
        "source_primitive_cache_sha256": primitive_hashes[
            raw4_source_horizon
        ],
        "source_horizon_id": raw4_source_horizon,
        "common_history_support_sha256": support_sha,
        "history_samples": HISTORY_SAMPLES,
        "representation": "raw",
    }
    fingerprints: dict[str, str] = {}
    for cell_id in config["input_variants"]:
        definition = definitions[cell_id]
        kind = str(definition["kind"])
        if cell_id == "raw4":
            source = raw4_binding
        elif cell_id == "raw6":
            source = {
                "data_sha256": config["data_sha256"],
                "scaler_sha256": fold_config["scaler_sha256"],
                "common_anchor_sha256": fold_config[
                    "common_anchor_sha256"
                ],
                "common_history_support_sha256": support_sha,
                "history_samples": RAW6_SAMPLES,
                "representation": "raw",
            }
        elif cell_id == "raw4_zero":
            source = {
                "raw4": raw4_binding,
                "zero_map": {
                    "dtype": "float32",
                    "shape": f"9x{HISTORY_SAMPLES}",
                },
            }
        else:
            horizon_id = str(definition["horizon_id"])
            error_binding = {
                "source_primitive_cache_sha256": primitive_hashes[
                    horizon_id
                ],
                "source_horizon_id": horizon_id,
                "common_history_support_sha256": support_sha,
                "history_samples": HISTORY_SAMPLES,
                "representation": "signed_error_x_minus_mu",
            }
            source = (
                error_binding
                if kind == "error"
                else {"raw4": raw4_binding, "error": error_binding}
            )
        fingerprints[cell_id] = canonical_fingerprint(
            {
                "cell_id": cell_id,
                "definition": definition,
                "source": source,
            }
        )
    return fingerprints


def materialize_cell_input(
    cell: Mapping[str, Any],
    dataset: DaphnetDataset,
    classification_windows: WindowTable,
    horizons: list[dict[str, Any]],
    plans: Mapping[str, Mapping[str, HistoryPlan]],
    primitives: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    scaler: RobustChannelScaler,
) -> dict[str, dict[str, np.ndarray]]:
    """Materialize one cell at a time to bound seven-worker host memory."""

    cell_id = str(cell["cell_id"])
    horizon_by_id = {
        str(item["horizon_id"]): item for item in horizons
    }
    reference_id = str(horizons[0]["horizon_id"])
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        reference = primitives[reference_id][split]
        raw4 = materialize_raw4_history(
            reference,
            plans[reference_id][split],
            int(horizon_by_id[reference_id]["horizon_samples"]),
            "raw4",
        )
        if cell_id == "raw4":
            payload = {
                cell_id: raw4["raw4"],
                "y": raw4["y"],
                "window_index": raw4["window_index"],
            }
        elif cell_id == "raw6":
            payload = materialize_raw6_history(
                dataset,
                classification_windows,
                plans[reference_id][split].anchor_window_indices,
                scaler,
                cell_id,
            )
        elif cell_id == "raw4_zero":
            payload = {
                cell_id: np.ascontiguousarray(
                    np.concatenate(
                        (raw4["raw4"], np.zeros_like(raw4["raw4"])),
                        axis=1,
                        dtype=np.float32,
                    )
                ),
                "y": raw4["y"],
                "window_index": raw4["window_index"],
            }
        else:
            horizon_id = str(cell["horizon_id"])
            suffix = horizon_id.lower()
            error_name = f"error_{suffix}"
            error = materialize_error_history(
                primitives[horizon_id][split],
                plans[horizon_id][split],
                int(horizon_by_id[horizon_id]["horizon_samples"]),
                error_name,
            )
            if cell["kind"] == "error":
                payload = {
                    cell_id: error[error_name],
                    "y": error["y"],
                    "window_index": error["window_index"],
                }
            else:
                if not np.array_equal(
                    raw4["window_index"],
                    error["window_index"],
                ):
                    raise AssertionError("Raw4/error endpoints differ")
                payload = {
                    cell_id: np.ascontiguousarray(
                        np.concatenate(
                            (raw4["raw4"], error[error_name]),
                            axis=1,
                            dtype=np.float32,
                        )
                    ),
                    "y": error["y"],
                    "window_index": error["window_index"],
                }
        expected_shape = (
            len(plans[reference_id][split].anchor_rows),
            int(cell["in_channels"]),
            int(cell["history_samples"]),
        )
        if tuple(payload[cell_id].shape) != expected_shape:
            raise AssertionError(
                f"Unexpected {cell_id}/{split} shape "
                f"{payload[cell_id].shape} != {expected_shape}"
            )
        expected_indices = plans[reference_id][
            split
        ].anchor_window_indices
        if not np.array_equal(payload["window_index"], expected_indices):
            raise AssertionError(f"{cell_id}/{split} endpoints changed")
        expected_y = classification_windows.label[expected_indices]
        if not np.array_equal(payload["y"], expected_y):
            raise AssertionError(f"{cell_id}/{split} labels changed")
        if not np.isfinite(payload[cell_id]).all():
            raise ValueError(f"{cell_id}/{split} contains NaN or Inf")
        result[split] = payload
    return result


def _nbm_task_id(horizon: Mapping[str, Any]) -> str:
    return f"{horizon_directory(horizon)}/{NBM_NAME}/nbm"


def _cell_by_id(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(cell["cell_id"]): dict(cell)
        for cell in config["experiments"]
    }


def _rewrite_classifier_done(
    task_root: Path,
    config: Mapping[str, Any],
    subject: str,
    cell: Mapping[str, Any],
    metrics: Mapping[str, Any],
    input_fingerprint: str,
    support_sha256: str,
    initial_state_sha256: str,
) -> None:
    """Seal the enriched classifier identity and refresh artifact hashes."""

    artifacts = {
        "best": task_root / "classifier_best.pt",
        "last": task_root / "classifier_last.pt",
        "metrics": task_root / "metrics.json",
        "predictions": task_root / "predictions.npz",
        "validation_predictions": task_root / "validation_predictions.npz",
        "predictions_csv": task_root / "predictions.csv",
    }
    completed = done_payload(
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{cell['cell_id']}",
        relative_to=task_root,
        artifacts={name: path.resolve() for name, path in artifacts.items()},
    )
    completed.update(
        {
            "cell_id": cell["cell_id"],
            "input_variant": cell["input_variant"],
            "horizon_id": cell["horizon_id"],
            "in_channels": int(cell["in_channels"]),
            "source_residual_sha256": input_fingerprint,
            "input_fingerprint": input_fingerprint,
            "materialized_input_sha256": metrics[
                "materialized_input_sha256"
            ],
            "materialized_input_sha256_by_split": metrics[
                "materialized_input_sha256_by_split"
            ],
            "input_support_sha256": support_sha256,
            "common_history_support_sha256": support_sha256,
            "initial_state_sha256": initial_state_sha256,
            "metrics_identity_sha256": canonical_fingerprint(
                {
                    "cell_id": metrics["cell_id"],
                    "input_variant": metrics["input_variant"],
                    "horizon_id": metrics["horizon_id"],
                    "in_channels": metrics["in_channels"],
                    "input_fingerprint": metrics["input_fingerprint"],
                    "materialized_input_sha256": metrics[
                        "materialized_input_sha256"
                    ],
                    "common_history_support_sha256": metrics[
                        "common_history_support_sha256"
                    ],
                }
            ),
        }
    )
    atomic_json_dump(completed, task_root / "DONE.json")


def train_cell(
    args: argparse.Namespace,
    config: dict[str, Any],
    cell: dict[str, Any],
    fold_config: dict[str, Any],
    reference_states: Mapping[int, Mapping[str, torch.Tensor]],
    inputs: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    classification_windows: WindowTable,
    device: torch.device,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    subject = str(fold_config["test_subject"])
    in_channels = int(cell["in_channels"])
    input_fingerprint = str(fold_config["input_fingerprints"][cell_id])
    support_sha256 = str(fold_config["common_history_support_sha256"])
    materialized_by_split = {
        split: {
            "input_sha256": array_sha256(payload[cell_id]),
            "y_sha256": array_sha256(payload["y"]),
            "window_index_sha256": array_sha256(payload["window_index"]),
        }
        for split, payload in inputs.items()
    }
    materialized_input_sha256 = canonical_fingerprint(
        {
            "cell_id": cell_id,
            "input_fingerprint": input_fingerprint,
            "splits": materialized_by_split,
        }
    )
    initial_hash = str(
        fold_config["reference_initial_state_sha256_by_in_channels"][
            str(in_channels)
        ]
    )
    classifier_fold = {
        "test_subject": subject,
        "val_subject": fold_config["val_subject"],
        "classifier_seed": fold_config["classifier_seed"],
        "reference_initial_state_sha256": initial_hash,
        "_reference_initial_state": reference_states[in_channels],
        "source": {
            "source_residual_cache_sha256": input_fingerprint,
            "input_support_sha256": support_sha256,
        },
    }
    task_root = task_root_for(args.output_dir, subject, cell_id)
    original = {
        "INPUT_NAME": rf.INPUT_NAME,
        "SOURCE_NBM": rf.SOURCE_NBM,
        "HISTORY_SECONDS": rf.HISTORY_SECONDS,
        "HISTORY_SAMPLES": rf.HISTORY_SAMPLES,
        "HISTORY_BLOCKS": rf.HISTORY_BLOCKS,
    }
    rf.INPUT_NAME = cell_id
    rf.SOURCE_NBM = NBM_NAME
    rf.HISTORY_SECONDS = float(cell["history_seconds"])
    rf.HISTORY_SAMPLES = int(cell["history_samples"])
    rf.HISTORY_BLOCKS = int(cell["history_blocks"])
    try:
        metrics = rf.train_classifier_resumable(
            args,
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "shared_parameter_count": int(cell["parameter_count"]),
            },
            cell,
            task_root,
            classifier_fold,
            inputs,
            dataset,
            classification_windows,
            device,
        )
    finally:
        for key, value in original.items():
            setattr(rf, key, value)

    enriched = {
        **metrics,
        "cell_id": cell_id,
        "input_variant": cell["input_variant"],
        "input_kind": cell["kind"],
        "horizon_id": cell["horizon_id"],
        "horizon_seconds": (
            None
            if cell["horizon_id"] == "shared"
            else next(
                item["horizon_seconds"]
                for item in config["horizon_variants"]
                if item["horizon_id"] == cell["horizon_id"]
            )
        ),
        "in_channels": in_channels,
        "formula": cell["formula"],
        "input_fingerprint": input_fingerprint,
        "materialized_input_sha256": materialized_input_sha256,
        "materialized_input_sha256_by_split": materialized_by_split,
        "source_residual_sha256": input_fingerprint,
        "common_history_support_sha256": support_sha256,
        "input_support_sha256": support_sha256,
    }
    atomic_json_dump(enriched, task_root / "metrics.json")
    _rewrite_classifier_done(
        task_root,
        config,
        subject,
        cell,
        enriched,
        input_fingerprint,
        support_sha256,
        initial_hash,
    )
    return enriched


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
        fold_config,
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
    print(
        f"[fold {test_subject}] train={train_subjects} val={val_subject} "
        f"normal={len(normal_train_indices)}/{len(normal_val_indices)} "
        f"anchors="
        f"{ {split: len(plans[horizons[0]['horizon_id']][split].anchor_rows) for split in ('train', 'validation', 'test')} }",
        flush=True,
    )
    fold_index = list(dataset.subjects).index(test_subject)
    primitives: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ] = {}
    primitive_hashes: dict[str, str] = {}
    nbm_hashes: dict[str, str] = {}
    fold_artifacts: dict[str, Path] = {
        "fold_config": fold_root / "fold_config.json",
        "scaler": fold_root / "scaler.json",
        "split_indices": fold_root / "split_indices.npz",
        "common_history_support": fold_root / "common_history_support.npz",
        "input_fingerprints": fold_root / "input_fingerprints.json",
    }

    for horizon in horizons:
        horizon_id = str(horizon["horizon_id"])
        horizon_samples = int(horizon["horizon_samples"])
        nbm_root = nbm_root_for(args.output_dir, test_subject, horizon)
        nbm_root.mkdir(parents=True, exist_ok=True)
        model, normal_training, nbm_sha256 = core.train_nbm_resumable(
            args,
            NBM_NAME,
            nbm_root,
            config["protocol_fingerprint"],
            args.seed + fold_index,
            dataset,
            windows_by_horizon[horizon_id],
            normal_train_indices,
            normal_val_indices,
            scaler,
            CONTEXT_SAMPLES,
            horizon_samples,
            device,
        )
        features, diagnostics, cache_sha256 = (
            load_or_extract_primitive_cache(
                args,
                config,
                horizon,
                nbm_root,
                nbm_sha256,
                model,
                dataset,
                windows_by_horizon[horizon_id],
                split_indices,
                scaler,
                device,
            )
        )
        reference_horizon_id = str(horizons[0]["horizon_id"])
        primitives[horizon_id] = {
            split: {
                **(
                    {"raw": payload["raw"]}
                    if horizon_id == reference_horizon_id
                    else {}
                ),
                "error": payload["error"],
                "y": payload["y"],
                "window_index": payload["window_index"],
            }
            for split, payload in features.items()
        }
        primitive_hashes[horizon_id] = cache_sha256
        nbm_hashes[horizon_id] = nbm_sha256
        atomic_json_dump(
            {
                "suite_version": SUITE_VERSION,
                "protocol_fingerprint": config["protocol_fingerprint"],
                "horizon_id": horizon_id,
                "horizon_seconds": float(horizon["horizon_seconds"]),
                "horizon_samples": horizon_samples,
                "context_seconds": CONTEXT_SECONDS,
                "context_samples": CONTEXT_SAMPLES,
                "history_seconds": HISTORY_SECONDS,
                "history_samples": HISTORY_SAMPLES,
                "history_blocks": int(horizon["history_blocks"]),
                "fixed_label_seconds": FIXED_LABEL_SECONDS,
                "fixed_label_samples": FIXED_LABEL_SAMPLES,
                "master_clean_normal_support": True,
                "derived_window_sha256": config["derived_window_sha256"][
                    horizon_id
                ],
                "nbm_sha256": nbm_sha256,
                "primitive_cache_sha256": cache_sha256,
                "transformer_architecture": config[
                    "transformer_architectures"
                ][horizon_id],
                "normal_training": normal_training,
                "primitive_diagnostics": diagnostics,
            },
            nbm_root / "nbm_summary.json",
        )
        fold_artifacts[f"{horizon_id}_nbm_done"] = (
            nbm_root / "nbm" / "DONE.json"
        )
        fold_artifacts[f"{horizon_id}_primitive_done"] = (
            nbm_root / "PRIMITIVE_CACHE_DONE.json"
        )
        del model
        del features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    input_fingerprints = build_input_fingerprints(
        config,
        fold_config,
        primitive_hashes,
    )
    fold_config.update(
        {
            "nbm_best_sha256_by_horizon": nbm_hashes,
            "primitive_cache_sha256_by_horizon": primitive_hashes,
            "input_fingerprints": input_fingerprints,
        }
    )
    core.save_or_validate_json(
        fold_root / "input_fingerprints.json",
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "test_subject": test_subject,
            "common_history_support_sha256": support_sha256,
            "input_fingerprints": input_fingerprints,
        },
    )
    core.save_or_validate_json(fold_root / "fold_config.json", fold_config)

    reference_states, counts, hashes, backbone_hash = (
        _aligned_classifier_states(
            int(fold_config["classifier_seed"]),
            int(args.classifier_hidden),
            float(args.classifier_dropout),
            bool(args.deterministic),
        )
    )
    if {
        str(key): value for key, value in counts.items()
    } != fold_config["parameter_count_by_in_channels"]:
        raise AssertionError("Fold classifier parameter counts changed")
    if {
        str(key): value for key, value in hashes.items()
    } != fold_config["reference_initial_state_sha256_by_in_channels"]:
        raise AssertionError("Fold classifier initialization changed")
    if backbone_hash != fold_config[
        "shared_backbone_initial_state_sha256"
    ]:
        raise AssertionError("Fold shared TCN backbone changed")

    completed = 0
    cells = _cell_by_id(config)
    for cell_id in config["input_variants"]:
        cell = cells[cell_id]
        cell_inputs = materialize_cell_input(
            cell,
            dataset,
            classification_windows,
            horizons,
            plans,
            primitives,
            scaler,
        )
        metrics = train_cell(
            args,
            config,
            cell,
            fold_config,
            reference_states,
            cell_inputs,
            dataset,
            classification_windows,
            device,
        )
        print(
            f"[fold {test_subject}] {cell_id} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"BA={metrics['balanced_accuracy']:.4f} "
            f"Recall={metrics['fog_recall']:.4f} "
            f"Specificity={metrics['specificity']:.4f}",
            flush=True,
        )
        fold_artifacts[f"{cell_id}_classifier_done"] = (
            task_root_for(args.output_dir, test_subject, cell_id)
            / "DONE.json"
        )
        completed += 1
        del cell_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (
            args.stop_after_completed_tasks > 0
            and completed >= args.stop_after_completed_tasks
        ):
            raise RuntimeError("Intentional stop after classifier tasks")

    fold_done = done_payload(
        stage="transformer_horizon_fusion_fold",
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
            "completed_inputs": list(config["input_variants"]),
            "common_history_support_sha256": support_sha256,
            "nbm_best_sha256_by_horizon": nbm_hashes,
            "primitive_cache_sha256_by_horizon": primitive_hashes,
            "input_fingerprints": input_fingerprints,
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
    if not isinstance(payload, Mapping):
        return ""
    mean, std = payload.get("mean"), payload.get("std")
    if mean is None or std is None:
        return ""
    precision = (
        3
        if metric
        in {"false_alarm_events_per_hour", "median_detection_delay_sec"}
        else 4
    )
    return f"{float(mean):.{precision}f} +/- {float(std):.{precision}f}"


def _format_delta(delta: Mapping[str, Any] | None) -> str:
    if not delta or delta.get("mean_delta") is None:
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
    fog_f1 = (
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    )
    nonfog_f1 = (
        2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    )
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


def _load_completed_cell(
    output_dir: Path,
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    cell_id = str(cell["cell_id"])
    root = task_root_for(output_dir, subject, cell_id)
    completed = validate_done(
        root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{cell_id}",
    )
    if completed is None:
        return None
    fold_root = output_dir / f"loso_{subject}"
    fold_config = _load_json(fold_root / "fold_config.json")
    support_sha256 = sha256_file(
        fold_root / "common_history_support.npz"
    )
    input_fingerprint = str(
        fold_config["input_fingerprints"][cell_id]
    )
    initial_hash = str(
        fold_config["reference_initial_state_sha256_by_in_channels"][
            str(cell["in_channels"])
        ]
    )
    done_identity = {
        "cell_id": cell_id,
        "input_variant": cell["input_variant"],
        "horizon_id": cell["horizon_id"],
        "in_channels": int(cell["in_channels"]),
        "source_residual_sha256": input_fingerprint,
        "input_fingerprint": input_fingerprint,
        "input_support_sha256": support_sha256,
        "common_history_support_sha256": support_sha256,
        "initial_state_sha256": initial_hash,
    }
    for key, expected in done_identity.items():
        if completed.get(key) != expected:
            raise ValueError(
                f"Classifier DONE identity mismatch at {root}: {key}"
            )
    metrics = _load_json(root / "metrics.json")
    metrics_identity = {
        "experiment_id": cell["experiment_id"],
        "variant": cell_id,
        "cell_id": cell_id,
        "input": cell_id,
        "input_variant": cell["input_variant"],
        "horizon_id": cell["horizon_id"],
        "in_channels": int(cell["in_channels"]),
        "nbm": NBM_NAME,
        "test_subject": subject,
        "history_seconds": float(cell["history_seconds"]),
        "history_samples": int(cell["history_samples"]),
        "history_blocks": int(cell["history_blocks"]),
        "source_residual_sha256": input_fingerprint,
        "input_fingerprint": input_fingerprint,
        "input_support_sha256": support_sha256,
        "common_history_support_sha256": support_sha256,
        "initial_state_sha256": initial_hash,
    }
    for key, expected in metrics_identity.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Metrics identity mismatch at {root}: "
                f"{key}={metrics.get(key)!r}, expected={expected!r}"
            )
    classifier_config = metrics.get("classifier_config", {})
    if (
        classifier_config.get("in_channels") != int(cell["in_channels"])
        or classifier_config.get("dilations")
        != list(TCN_M_DILATIONS)
        or classifier_config.get("receptive_field_samples")
        != TCN_M_RF_SAMPLES
        or classifier_config.get("parameter_count")
        != int(cell["parameter_count"])
    ):
        raise ValueError(f"Classifier architecture mismatch at {root}")
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        required = {"window_index", "y_true", "y_prob", "y_pred"}
        if set(payload.files) != required:
            raise ValueError(f"Unexpected prediction arrays at {root}")
        arrays = {
            key: np.asarray(payload[key]) for key in required
        }
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError(f"Prediction lengths differ at {root}")
    if not np.isfinite(arrays["y_prob"]).all():
        raise ValueError(f"Non-finite prediction probabilities at {root}")
    with np.load(
        fold_root / "common_history_support.npz",
        allow_pickle=False,
    ) as support:
        if not np.array_equal(
            arrays["window_index"],
            support["test_anchor_window_index"],
        ):
            raise ValueError(f"Prediction endpoints changed at {root}")
        if not np.array_equal(arrays["y_true"], support["test_y"]):
            raise ValueError(f"Prediction labels changed at {root}")
    return metrics, arrays


def _paired_effect(
    rows_by_cell: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparison: Mapping[str, str],
    subjects: list[str],
    samples: int,
    base_seed: int,
) -> dict[str, Any]:
    differences: list[float] = []
    common_subjects: list[str] = []
    for subject in subjects:
        new = rows_by_cell[comparison["new"]].get(subject)
        reference = rows_by_cell[comparison["reference"]].get(subject)
        if new is None or reference is None:
            continue
        new_value = new.get("pr_auc")
        reference_value = reference.get("pr_auc")
        if new_value is None or reference_value is None:
            continue
        differences.append(float(new_value) - float(reference_value))
        common_subjects.append(subject)
    values = np.asarray(differences, dtype=np.float64)
    effect = context_suite.paired_bootstrap_mean_ci(
        values,
        int(samples),
        context_suite.stable_bootstrap_seed(
            int(base_seed),
            str(comparison["comparison_id"]),
        ),
    )
    tolerance = 1e-12
    effect.update(
        {
            "common_subjects": ",".join(common_subjects),
            "wins": int((values > tolerance).sum()),
            "ties": int((np.abs(values) <= tolerance).sum()),
            "losses": int((values < -tolerance).sum()),
            "bootstrap_seed": int(base_seed),
        }
    )
    return effect


def refresh_summaries(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Rebuild every root result solely from sealed fold artifacts."""

    cells = list(config["experiments"])
    subjects = list(config["folds_resolved"])
    rows_by_cell: dict[str, dict[str, dict[str, Any]]] = {
        str(cell["cell_id"]): {} for cell in cells
    }
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []
    aggregate_experiments: dict[str, Any] = {}

    for cell in cells:
        cell_id = str(cell["cell_id"])
        group: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed_subjects: list[str] = []
        for subject in subjects:
            completed = _load_completed_cell(
                output_dir,
                config,
                cell,
                subject,
            )
            if completed is None:
                continue
            metrics, arrays = completed
            enriched = {
                **metrics,
                "parameter_count": int(cell["parameter_count"]),
            }
            group.append(enriched)
            fold_rows.append(enriched)
            rows_by_cell[cell_id][subject] = enriched
            truths.append(np.asarray(arrays["y_true"], dtype=np.int8))
            probabilities.append(
                np.asarray(arrays["y_prob"], dtype=np.float64)
            )
            predictions.append(
                np.asarray(arrays["y_pred"], dtype=np.int8)
            )
            completed_subjects.append(subject)
        macro = (
            aggregate_fold_metrics(group, list(CLASSIFICATION_METRICS))
            if group
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in CLASSIFICATION_METRICS
            }
        )
        pooled = (
            _prediction_metrics(
                np.concatenate(truths),
                np.concatenate(probabilities),
                np.concatenate(predictions),
            )
            if truths
            else None
        )
        aggregate_experiments[str(cell["experiment_id"])] = {
            **cell,
            "completed_folds": completed_subjects,
            "subject_macro": macro,
            "pooled": pooled,
        }
        manifest_rows.append(
            {
                "experiment_id": cell["experiment_id"],
                "cell_id": cell_id,
                "display_name": cell["display_name"],
                "input_kind": cell["kind"],
                "horizon_id": cell["horizon_id"],
                "formula": cell["formula"],
                "in_channels": cell["in_channels"],
                "history_seconds": cell["history_seconds"],
                "history_samples": cell["history_samples"],
                "history_blocks": cell["history_blocks"],
                "input_shape": cell["input_shape"],
                "parameter_count": cell["parameter_count"],
                "expected_folds": len(subjects),
                "completed_folds": len(completed_subjects),
                "status": (
                    "complete"
                    if completed_subjects == subjects
                    else ("partial" if completed_subjects else "pending")
                ),
                "completed_subjects": ",".join(completed_subjects),
            }
        )
        aggregate_row: dict[str, Any] = {
            "experiment_id": cell["experiment_id"],
            "cell_id": cell_id,
            "display_name": cell["display_name"],
            "input_kind": cell["kind"],
            "horizon_id": cell["horizon_id"],
            "in_channels": cell["in_channels"],
            "history_seconds": cell["history_seconds"],
            "history_samples": cell["history_samples"],
            "history_blocks": cell["history_blocks"],
            "input_shape": cell["input_shape"],
            "parameter_count": cell["parameter_count"],
            "completed_folds": len(completed_subjects),
        }
        for metric in CLASSIFICATION_METRICS:
            aggregate_row[f"{metric}_mean"] = macro[metric]["mean"]
            aggregate_row[f"{metric}_std"] = macro[metric]["std"]
        aggregate_rows.append(aggregate_row)

    active_cell_ids = set(rows_by_cell)
    active_comparisons = [
        comparison
        for comparison in config["comparisons"]
        if comparison["new"] in active_cell_ids
        and comparison["reference"] in active_cell_ids
    ]
    comparison_rows: list[dict[str, Any]] = []
    effects_by_id: dict[str, dict[str, Any]] = {}
    for comparison in active_comparisons:
        effect = _paired_effect(
            rows_by_cell,
            comparison,
            subjects,
            int(config["bootstrap_samples"]),
            int(config["bootstrap_seed"]),
        )
        row = {**comparison, **effect}
        comparison_rows.append(row)
        effects_by_id[str(comparison["comparison_id"])] = row

    primary_effect_by_cell: dict[str, dict[str, Any]] = {}
    for horizon_id in ("h050", "h100", "h200"):
        fusion = f"raw4_error_{horizon_id}"
        error = f"error_{horizon_id}"
        fusion_comparison = (
            f"raw4_error_{horizon_id}_minus_raw4_zero"
        )
        error_comparison = f"error_{horizon_id}_minus_raw4"
        if fusion_comparison in effects_by_id:
            primary_effect_by_cell[fusion] = effects_by_id[
                fusion_comparison
            ]
        if error_comparison in effects_by_id:
            primary_effect_by_cell[error] = effects_by_id[
                error_comparison
            ]

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["pr_auc_mean"])
            if row["pr_auc_mean"] is not None
            else float("inf"),
            row["cell_id"],
        )
    )
    ranking_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(aggregate_rows, start=1)
    ]
    for cell in cells:
        cell_id = str(cell["cell_id"])
        aggregate = next(
            row for row in aggregate_rows if row["cell_id"] == cell_id
        )
        macro = aggregate_experiments[
            str(cell["experiment_id"])
        ]["subject_macro"]
        publication_rows.append(
            {
                "Input": cell["display_name"],
                "Horizon": cell["horizon_id"],
                "Channels": cell["in_channels"],
                "Shape": cell["input_shape"],
                "PR-AUC": _format_mean_sd(macro, "pr_auc"),
                "Primary delta PR-AUC [95% CI]": _format_delta(
                    primary_effect_by_cell.get(cell_id)
                ),
                "BA": _format_mean_sd(macro, "balanced_accuracy"),
                "Macro-F1": _format_mean_sd(macro, "macro_f1"),
                "AUROC": _format_mean_sd(macro, "roc_auc"),
                "FoG Sensitivity/Recall": _format_mean_sd(
                    macro,
                    "fog_recall",
                ),
                "Specificity": _format_mean_sd(macro, "specificity"),
                "FoG Precision": _format_mean_sd(macro, "precision"),
                "FoG F1": _format_mean_sd(macro, "fog_f1"),
                "Event Sensitivity": _format_mean_sd(
                    macro,
                    "event_sensitivity",
                ),
                "FA/h": _format_mean_sd(
                    macro,
                    "false_alarm_events_per_hour",
                ),
                "Median Detection Delay": _format_mean_sd(
                    macro,
                    "median_detection_delay_sec",
                ),
                "Completed folds": aggregate["completed_folds"],
            }
        )

    fold_columns = [
        "experiment_id",
        "variant",
        "cell_id",
        "input",
        "input_variant",
        "input_kind",
        "display_name",
        "formula",
        "horizon_id",
        "horizon_seconds",
        "in_channels",
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
        "pos_weight",
        "elapsed_sec",
        "source_residual_sha256",
        "input_fingerprint",
        "input_support_sha256",
        "common_history_support_sha256",
        "initial_state_sha256",
        "parameter_count",
    ]
    manifest_columns = [
        "experiment_id",
        "cell_id",
        "display_name",
        "input_kind",
        "horizon_id",
        "formula",
        "in_channels",
        "history_seconds",
        "history_samples",
        "history_blocks",
        "input_shape",
        "parameter_count",
        "expected_folds",
        "completed_folds",
        "status",
        "completed_subjects",
    ]
    aggregate_columns = [
        "rank",
        "experiment_id",
        "cell_id",
        "display_name",
        "input_kind",
        "horizon_id",
        "in_channels",
        "history_seconds",
        "history_samples",
        "history_blocks",
        "input_shape",
        "parameter_count",
        "completed_folds",
        *_metric_summary_columns(),
    ]
    comparison_columns = [
        "comparison_id",
        "new",
        "reference",
        "interpretation",
        "common_subjects",
        "mean_delta",
        "ci_low",
        "ci_high",
        "n_paired_subjects",
        "wins",
        "ties",
        "losses",
        "bootstrap_samples",
        "bootstrap_seed",
    ]
    publication_columns = [
        "Input",
        "Horizon",
        "Channels",
        "Shape",
        "PR-AUC",
        "Primary delta PR-AUC [95% CI]",
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
        output_dir / "fold_summary.csv",
        fold_rows,
        fold_columns,
    )
    core.atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        manifest_columns,
    )
    core.atomic_csv_write(
        output_dir / "aggregate_summary.csv",
        ranking_rows,
        aggregate_columns,
    )
    core.atomic_csv_write(
        output_dir / "paired_pr_auc_deltas.csv",
        comparison_rows,
        comparison_columns,
    )
    core.atomic_csv_write(
        output_dir / "publication_table.csv",
        publication_rows,
        publication_columns,
    )

    completed_classifier_cells = len(fold_rows)
    expected_classifier_cells = int(config["expected_classifier_cells"])
    completed_nbm_tasks = 0
    completed_primitive_tasks = 0
    for subject in subjects:
        for horizon in config["horizon_variants"]:
            root = nbm_root_for(output_dir, subject, horizon)
            if (root / "nbm" / "DONE.json").exists():
                completed_nbm_tasks += 1
            if (root / "PRIMITIVE_CACHE_DONE.json").exists():
                completed_primitive_tasks += 1
    expected_nbm_tasks = int(config["expected_nbm_tasks"])
    completed_fold_manifests = sum(
        int((output_dir / f"loso_{subject}" / "FOLD_DONE.json").exists())
        for subject in subjects
    )
    integrity_complete = (
        completed_classifier_cells == expected_classifier_cells
        and completed_nbm_tasks == expected_nbm_tasks
        and completed_primitive_tasks == expected_nbm_tasks
        and completed_fold_manifests == len(subjects)
    )
    best_experiment = (
        ranking_rows[0]["experiment_id"]
        if integrity_complete
        and ranking_rows
        and ranking_rows[0]["pr_auc_mean"] is not None
        else None
    )
    aggregate_payload = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "aggregation_unit": "held_out_subject",
        "metric_dispersion": "population standard deviation across LOSO folds",
        "ranking_metric": "subject_macro_pr_auc_mean",
        "best_experiment": best_experiment,
        "delta_pr_auc": {
            "method": "paired nonparametric bootstrap over held-out subjects",
            "confidence_level": 0.95,
            "samples": int(config["bootstrap_samples"]),
            "seed": int(config["bootstrap_seed"]),
        },
        "experiments": aggregate_experiments,
        "paired_pr_auc_comparisons": comparison_rows,
    }
    atomic_json_dump(
        aggregate_payload,
        output_dir / "aggregate_metrics.json",
    )
    status = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_experiments": len(cells),
        "expected_nbm_tasks": expected_nbm_tasks,
        "completed_nbm_tasks": completed_nbm_tasks,
        "expected_primitive_cache_tasks": expected_nbm_tasks,
        "completed_primitive_cache_tasks": completed_primitive_tasks,
        "expected_classifier_cells": expected_classifier_cells,
        "completed_classifier_cells": completed_classifier_cells,
        "expected_fold_manifests": len(subjects),
        "completed_fold_manifests": completed_fold_manifests,
        "status": "complete" if integrity_complete else "partial",
        "best_experiment": best_experiment,
    }
    atomic_json_dump(status, output_dir / "status.json")
    results_path = output_dir / "RESULTS_DONE.json"
    if integrity_complete:
        results_done = done_payload(
            stage="transformer_horizon_fusion_results",
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
        atomic_json_dump(results_done, results_path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    worker_mode = bool(str(args.worker_fold).strip())
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if paths_overlap(args.output_dir, args.data_dir):
        raise ValueError("Output directory must not overlap processed data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or another directory"
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
        selected_inputs,
        folds,
    ) = load_dataset_and_protocol(args, device)
    if worker_mode:
        if (
            tuple(folds) != EXPECTED_LOSO_SUBJECTS
            or [item["horizon_id"] for item in horizons]
            != [item["horizon_id"] for item in HORIZON_DEFINITIONS]
            or tuple(selected_inputs) != INPUT_VARIANTS
            or config["protocol_scope"]
            != "strict_transformer_horizon3_fusion9_x_8_fold"
        ):
            raise ValueError(
                "Parallel workers require the complete strict 3-horizon, "
                "9-input, 8-fold protocol"
            )
    execution_folds = list(folds)
    if worker_mode:
        selected = core.parse_folds(args.worker_fold, dataset.subjects)
        if len(selected) != 1:
            raise ValueError("--worker-fold must resolve to exactly one subject")
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
        f"inputs={selected_inputs} context={CONTEXT_SECONDS:g}s "
        f"history={HISTORY_SECONDS:g}s label={FIXED_LABEL_SECONDS:g}s "
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
                    "nbm_tasks_visited": len(horizons),
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
