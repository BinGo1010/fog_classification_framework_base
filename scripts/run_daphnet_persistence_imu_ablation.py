#!/usr/bin/env python
"""Strict seven-way IMU-input ablation on frozen Persistence residuals.

The canonical nine-channel Persistence residual cache and its four-second
history support are immutable inputs.  Each experiment selects one ordered
subset of ankle, thigh, and trunk channels before training the same TCN-M
readout.  The NBM, scaler, sigma, anchors, labels, and LOSO splits are never
retrained or rebuilt for a sensor subset.

To avoid allowing a different input projection shape to perturb all later
random weights, every classifier is derived from one deterministic nine-channel
reference state for its fold.  Common tensors are copied exactly and the
``projection.0.weight`` tensor is sliced by the canonical channel indices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    capture_rng_state,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_done,
)


SUITE_VERSION = "daphnet_persistence_h4_tcnm_imu7_loso.v1"
SOURCE_SUITE_VERSION = rf.SOURCE_SUITE_VERSION
SOURCE_NBM = rf.SOURCE_NBM
INPUT_NAME = rf.INPUT_NAME
HISTORY_SECONDS = rf.HISTORY_SECONDS
HISTORY_SAMPLES = rf.HISTORY_SAMPLES
HISTORY_BLOCKS = rf.HISTORY_BLOCKS
HORIZON_SAMPLES = 32
STRIDE_SAMPLES = 16
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
PROJECTION_WEIGHT_KEY = "projection.0.weight"
EXPECTED_CHANNEL_NAMES = rf.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = rf.EXPECTED_LOSO_SUBJECTS
CLASSIFICATION_METRICS = tuple(rf.CLASSIFICATION_METRICS)
_RF_BUILD_MODEL = rf.build_model

IMU_VARIANTS: dict[str, dict[str, Any]] = {
    "ankle": {
        "display_name": "Ankle",
        "sensor_count": 1,
        "channel_indices": (0, 1, 2),
        "channel_names": EXPECTED_CHANNEL_NAMES[0:3],
    },
    "thigh": {
        "display_name": "Thigh",
        "sensor_count": 1,
        "channel_indices": (3, 4, 5),
        "channel_names": EXPECTED_CHANNEL_NAMES[3:6],
    },
    "trunk": {
        "display_name": "Trunk",
        "sensor_count": 1,
        "channel_indices": (6, 7, 8),
        "channel_names": EXPECTED_CHANNEL_NAMES[6:9],
    },
    "ankle_thigh": {
        "display_name": "Ankle+Thigh",
        "sensor_count": 2,
        "channel_indices": (0, 1, 2, 3, 4, 5),
        "channel_names": EXPECTED_CHANNEL_NAMES[0:6],
    },
    "ankle_trunk": {
        "display_name": "Ankle+Trunk",
        "sensor_count": 2,
        "channel_indices": (0, 1, 2, 6, 7, 8),
        "channel_names": (
            *EXPECTED_CHANNEL_NAMES[0:3],
            *EXPECTED_CHANNEL_NAMES[6:9],
        ),
    },
    "thigh_trunk": {
        "display_name": "Thigh+Trunk",
        "sensor_count": 2,
        "channel_indices": (3, 4, 5, 6, 7, 8),
        "channel_names": EXPECTED_CHANNEL_NAMES[3:9],
    },
    "all_three": {
        "display_name": "All three",
        "sensor_count": 3,
        "channel_indices": tuple(range(9)),
        "channel_names": EXPECTED_CHANNEL_NAMES,
    },
}

COMPARISONS: tuple[dict[str, str], ...] = tuple(
    {
        "comparison_id": f"{name}_minus_all_three",
        "new": name,
        "reference": "all_three",
        "interpretation": (
            f"PR-AUC change when retaining {definition['display_name']} "
            "instead of all three IMUs"
        ),
    }
    for name, definition in IMU_VARIANTS.items()
    if name != "all_three"
)

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_persistence_imu_ablation.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/resume.py",
)

DEFAULT_DATA_DIR = rf.DEFAULT_DATA_DIR
DEFAULT_SOURCE_SUITE_DIR = rf.DEFAULT_SOURCE_SUITE_DIR
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daphnet_persistence_h4_tcnm_imu7_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seven-way Daphnet IMU-input ablation with frozen Persistence "
            "residual_h4s and a fixed TCN-M"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        default=DEFAULT_SOURCE_SUITE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help="Run exactly one fold; used by the multi-GPU scheduler",
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Permit reduced epochs/windows for pipeline validation; smoke "
            "outputs are explicitly non-reportable"
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
    parser.add_argument("--max-classifier-windows", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
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
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--stop-after-completed-tasks",
        type=int,
        default=0,
        help="Development-only stop hook after visiting classifier cells",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if int(args.seed) != 42:
        raise ValueError("This preregistered suite fixes --seed 42")
    positive = {
        "classifier_hidden": args.classifier_hidden,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [key for key, value in positive.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {invalid}")
    if (
        args.max_classifier_windows < 0
        or args.num_workers < 0
        or args.stop_after_completed_tasks < 0
    ):
        raise ValueError("Window/worker/stop counts must be non-negative")
    if 0 < args.max_classifier_windows < 2:
        raise ValueError("--max-classifier-windows must be zero or >= 2")
    if not math.isfinite(args.classifier_lr) or args.classifier_lr <= 0:
        raise ValueError("--classifier-lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0,1)")
    if not args.smoke:
        formal = {
            "classifier_hidden": (int(args.classifier_hidden), 48),
            "classifier_dropout": (
                float(args.classifier_dropout),
                0.15,
            ),
            "classifier_epochs": (int(args.classifier_epochs), 12),
            "classifier_patience": (int(args.classifier_patience), 4),
            "classifier_lr": (float(args.classifier_lr), 1e-3),
            "weight_decay": (float(args.weight_decay), 1e-4),
            "batch_size": (int(args.batch_size), 256),
            "max_classifier_windows": (
                int(args.max_classifier_windows),
                0,
            ),
            "bootstrap_samples": (
                int(args.bootstrap_samples),
                100000,
            ),
            "bootstrap_seed": (int(args.bootstrap_seed), 42),
            "deterministic": (bool(args.deterministic), True),
            "amp": (bool(args.amp), True),
        }
        changed = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in formal.items()
            if actual != expected
        }
        if changed:
            raise ValueError(
                "Formal protocol options changed; use canonical values or "
                f"add --smoke for a non-reportable run: {changed}"
            )


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def _common_state_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    common = {
        key: value
        for key, value in state_dict.items()
        if key != PROJECTION_WEIGHT_KEY
    }
    return rf.state_dict_sha256(common)


def build_shared_initialised_model(
    *,
    channel_indices: tuple[int, ...] | list[int],
    hidden_channels: int,
    dropout: float,
    seed: int,
    deterministic: bool,
    in_channels: int | None = None,
) -> torch.nn.Module:
    """Build a subset TCN-M by slicing one canonical nine-channel state.

    The RNG state is restored to the point immediately after constructing the
    reference model.  Consequently common parameter values and the stochastic
    training stream are independent of the selected input width.
    """

    indices = tuple(int(value) for value in channel_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("channel_indices must be non-empty and unique")
    if tuple(sorted(indices)) != indices:
        raise ValueError("channel_indices must retain canonical order")
    if min(indices) < 0 or max(indices) >= len(EXPECTED_CHANNEL_NAMES):
        raise ValueError("channel_indices are outside the canonical 9 channels")
    if in_channels is not None and int(in_channels) != len(indices):
        raise ValueError("in_channels does not match channel_indices")

    rf.set_seed(int(seed), bool(deterministic))
    reference = _RF_BUILD_MODEL(
        in_channels=len(EXPECTED_CHANNEL_NAMES),
        hidden_channels=int(hidden_channels),
        dropout=float(dropout),
        dilations=TCN_M_DILATIONS,
    )
    reference_state = {
        key: value.detach().clone()
        for key, value in reference.state_dict().items()
    }
    post_reference_rng = capture_rng_state()
    subset = _RF_BUILD_MODEL(
        in_channels=len(indices),
        hidden_channels=int(hidden_channels),
        dropout=float(dropout),
        dilations=TCN_M_DILATIONS,
    )
    subset_state = subset.state_dict()
    for key, target in subset_state.items():
        if key == PROJECTION_WEIGHT_KEY:
            source = reference_state[key][:, indices, :]
        else:
            source = reference_state[key]
        if target.shape != source.shape:
            raise AssertionError(
                f"Shared initialization shape mismatch: "
                f"{key}/{tuple(target.shape)}/{tuple(source.shape)}"
            )
        target.copy_(source)
    subset.load_state_dict(subset_state, strict=True)
    restore_rng_state(post_reference_rng)
    del reference
    return subset


def variant_protocol(
    args: argparse.Namespace,
    sampling_rate_hz: int,
) -> list[dict[str, Any]]:
    receptive_field = rf.convolutional_receptive_field(TCN_M_DILATIONS)
    if receptive_field != TCN_M_RF_SAMPLES:
        raise AssertionError("Canonical TCN-M receptive field changed")
    variants: list[dict[str, Any]] = []
    common_hashes: set[str] = set()
    for name, definition in IMU_VARIANTS.items():
        indices = tuple(definition["channel_indices"])
        model = build_shared_initialised_model(
            channel_indices=indices,
            hidden_channels=args.classifier_hidden,
            dropout=args.classifier_dropout,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        state = model.state_dict()
        parameter_count = rf.parameter_count(model)
        common_hash = _common_state_sha256(state)
        common_hashes.add(common_hash)
        variants.append(
            {
                "variant": name,
                "display_name": definition["display_name"],
                "experiment_id": f"persistence_h4s_tcnm__imu_{name}",
                "sensor_count": int(definition["sensor_count"]),
                "n_channels": len(indices),
                "channel_indices": list(indices),
                "channel_names": list(definition["channel_names"]),
                "dilations": list(TCN_M_DILATIONS),
                "n_blocks": len(TCN_M_DILATIONS),
                "convolutions_per_block": rf.CONVS_PER_BLOCK,
                "kernel_size": rf.KERNEL_SIZE,
                "receptive_field_samples": receptive_field,
                "receptive_field_seconds": (
                    receptive_field / float(sampling_rate_hz)
                ),
                "parameter_count": parameter_count,
                "input_projection_parameters": (
                    int(args.classifier_hidden) * len(indices)
                ),
                "reference_initial_state_sha256": (
                    rf.state_dict_sha256(state)
                ),
                "reference_common_state_sha256": common_hash,
                "shared_reference_state_sha256": common_hash,
                "input_bandwidth_ratio_vs_all": len(indices) / 9.0,
            }
        )
        del model
    if len(common_hashes) != 1:
        raise AssertionError("TCN-M common initial tensors are not identical")
    return variants


def build_protocol(
    args: argparse.Namespace,
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    required_source = {
        "context_samples": 128,
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "seed": 42,
        "residual_clip": 12.0,
    }
    for key, expected in required_source.items():
        actual = source_config.get(key)
        if isinstance(expected, float):
            matches = actual is not None and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Canonical source {key} changed: "
                f"expected={expected!r}, observed={actual!r}"
            )
    variants = variant_protocol(args, dataset.sampling_rate_hz)
    protocol = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "channel_names": list(dataset.channel_names),
        "n_source_channels": int(dataset.n_channels),
        "subjects": list(dataset.subjects),
        "excluded_subjects": list(source_config["excluded_subjects"]),
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbm": SOURCE_NBM,
        "nbm_policy": (
            "Freeze the canonical nine-channel Persistence checkpoint, "
            "learned sigma, scaler, and clipped standardized residual cache."
        ),
        "ablation_scope": (
            "Sensor information available to the TCN-M readout after a "
            "frozen canonical Persistence representation."
        ),
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "context_samples": int(source_config["context_samples"]),
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "window_count": int(len(windows)),
        "variants": variants,
        "expected_experiments": len(variants),
        "expected_classifier_cells": (
            len(variants) * len(EXPECTED_LOSO_SUBJECTS)
        ),
        "comparisons": list(COMPARISONS),
        "classifier": {
            "name": "tcn_m",
            "hidden_channels": int(args.classifier_hidden),
            "dropout": float(args.classifier_dropout),
            "kernel_size": rf.KERNEL_SIZE,
            "convolutions_per_block": rf.CONVS_PER_BLOCK,
            "dilations": list(TCN_M_DILATIONS),
            "receptive_field_samples": TCN_M_RF_SAMPLES,
            "receptive_field_seconds": (
                TCN_M_RF_SAMPLES / float(dataset.sampling_rate_hz)
            ),
            "global_pooling": "mean_and_max_over_full_4s_input",
            "variable_component": (
                "input projection width only; all common tensors are copied "
                "from one nine-channel reference state"
            ),
        },
        "classifier_epochs": int(args.classifier_epochs),
        "classifier_patience": int(args.classifier_patience),
        "classifier_lr": float(args.classifier_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_classifier_windows": int(args.max_classifier_windows),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "run_kind": "smoke" if args.smoke else "formal",
        "reportable": not bool(args.smoke),
        "fairness_contract": {
            "ablation_axis": "ordered IMU channel subset",
            "same_fold_scaler": True,
            "same_persistence_checkpoint_and_sigma": True,
            "same_anchor_history_and_labels": True,
            "same_tcn_m_common_initial_parameters": True,
            "variable_input_projection_only": True,
            "same_source_scaler_nbm_sigma_and_residual_cache": True,
            "same_fold_history_anchors_and_labels": True,
            "same_tcn_m_common_parameter_values": True,
            "projection_initialization": (
                "slice canonical 9-channel projection weight by ordered "
                "physical channel indices"
            ),
            "same_post_initialization_rng_state": True,
            "same_epoch_shuffle_rule": "classifier_seed + epoch",
            "same_optimizer_loss_and_class_weight_rule": True,
            "independent_validation_early_stopping": True,
            "independent_validation_threshold": True,
            "test_subject_never_selects_model_or_threshold": True,
            "support_warning": (
                "Do not rebuild sensor-specific valid masks; source support "
                "is retained to prevent missing-sensor records from changing "
                "the evaluated samples."
            ),
        },
    }
    fingerprint = canonical_fingerprint(protocol)
    return {
        **protocol,
        "protocol_fingerprint": fingerprint,
        "data_dir": str(args.data_dir),
        "source_suite_dir": str(args.source_suite_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "num_workers": int(args.num_workers),
        "resume": bool(args.resume),
        "smoke": bool(args.smoke),
    }


def subset_history_inputs(
    full_inputs: Mapping[str, Mapping[str, np.ndarray]],
    variant: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    indices = tuple(int(value) for value in variant["channel_indices"])
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        payload = full_inputs[split]
        full = np.asarray(payload[INPUT_NAME], dtype=np.float32)
        if full.ndim != 3 or full.shape[1:] != (9, HISTORY_SAMPLES):
            raise ValueError(
                f"Expected full [N,9,{HISTORY_SAMPLES}] history: "
                f"{split}/{full.shape}"
            )
        selected = np.ascontiguousarray(full[:, indices, :])
        if selected.shape[1:] != (len(indices), HISTORY_SAMPLES):
            raise AssertionError("IMU channel slicing changed history shape")
        if not np.isfinite(selected).all():
            raise ValueError(f"Non-finite subset history: {split}")
        result[split] = {
            INPUT_NAME: selected,
            "y": np.asarray(payload["y"], dtype=np.int8),
            "window_index": np.asarray(
                payload["window_index"],
                dtype=np.int64,
            ),
        }
    return result


@contextmanager
def _patched_fold_preparation() -> Iterator[None]:
    original_suite = rf.SUITE_VERSION
    original_local = dict(rf.TCN_VARIANTS["local"])
    try:
        rf.SUITE_VERSION = SUITE_VERSION
        rf.TCN_VARIANTS["local"] = {
            **rf.TCN_VARIANTS["local"],
            "dilations": TCN_M_DILATIONS,
            "receptive_field_samples": TCN_M_RF_SAMPLES,
        }
        yield
    finally:
        rf.SUITE_VERSION = original_suite
        rf.TCN_VARIANTS["local"] = original_local


@contextmanager
def _patched_shared_model_builder(
    args: argparse.Namespace,
    variant: Mapping[str, Any],
    classifier_seed: int,
) -> Iterator[None]:
    original_builder = rf.build_model

    def builder(
        *,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
        dilations: tuple[int, ...],
    ) -> torch.nn.Module:
        if tuple(int(value) for value in dilations) != TCN_M_DILATIONS:
            raise AssertionError("IMU suite attempted a non-TCN-M schedule")
        if int(in_channels) != int(variant["n_channels"]):
            raise AssertionError("Classifier input width differs from variant")
        return build_shared_initialised_model(
            channel_indices=variant["channel_indices"],
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            seed=classifier_seed,
            deterministic=args.deterministic,
        )

    try:
        rf.build_model = builder
        yield
    finally:
        rf.build_model = original_builder


def _fold_model_identities(
    args: argparse.Namespace,
    variants: list[dict[str, Any]],
    classifier_seed: int,
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    common_hashes: set[str] = set()
    for variant in variants:
        model = build_shared_initialised_model(
            channel_indices=variant["channel_indices"],
            hidden_channels=args.classifier_hidden,
            dropout=args.classifier_dropout,
            seed=classifier_seed,
            deterministic=args.deterministic,
        )
        state = model.state_dict()
        identity = {
            "variant": variant["variant"],
            "classifier_seed": int(classifier_seed),
            "n_channels": int(variant["n_channels"]),
            "channel_indices": list(variant["channel_indices"]),
            "parameter_count": rf.parameter_count(model),
            "initial_state_sha256": rf.state_dict_sha256(state),
            "common_state_sha256": _common_state_sha256(state),
            "projection_weight_sha256": rf.state_dict_sha256(
                {PROJECTION_WEIGHT_KEY: state[PROJECTION_WEIGHT_KEY]}
            ),
        }
        if identity["parameter_count"] != int(variant["parameter_count"]):
            raise AssertionError("Fold parameter count changed")
        identities[str(variant["variant"])] = identity
        common_hashes.add(str(identity["common_state_sha256"]))
        del model
    if len(common_hashes) != 1:
        raise AssertionError("Fold common initial state differs by IMU subset")
    return identities


def prepare_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
) -> tuple[
    Path,
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    with _patched_fold_preparation():
        fold_root, full_inputs, fold_config = rf.prepare_fold_inputs(
            args,
            config,
            dataset,
            windows,
            subject,
        )
    identities = _fold_model_identities(
        args,
        list(config["variants"]),
        int(fold_config["classifier_seed"]),
    )
    all_identity = identities["all_three"]
    if (
        all_identity["initial_state_sha256"]
        != fold_config["reference_initial_state_sha256"]
    ):
        raise AssertionError("All-three initialization differs from fold reference")

    source_cache_sha = str(
        fold_config["source"]["source_residual_cache_sha256"]
    )
    support_sha = str(fold_config["source"]["input_support_sha256"])
    input_fingerprints = {
        str(variant["variant"]): canonical_fingerprint(
            {
                "source_residual_cache_sha256": source_cache_sha,
                "input_support_sha256": support_sha,
                "representation": (
                    "canonical_clipped_standardized_persistence_residual"
                ),
                "channel_indices": list(variant["channel_indices"]),
                "channel_names": list(variant["channel_names"]),
                "n_channels": int(variant["n_channels"]),
                "history_samples": HISTORY_SAMPLES,
                "history_blocks": HISTORY_BLOCKS,
                "horizon_samples": HORIZON_SAMPLES,
                "source_stride_samples": STRIDE_SAMPLES,
            }
        )
        for variant in config["variants"]
    }
    fingerprint_path = fold_root / "sensor_input_fingerprints.json"
    rf.save_or_validate_json(
        fingerprint_path,
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "source_residual_cache_sha256": source_cache_sha,
            "input_support_sha256": support_sha,
            "variants": input_fingerprints,
        },
    )
    initialization_path = fold_root / "sensor_model_initialization.json"
    rf.save_or_validate_json(
        initialization_path,
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "strategy": (
                "canonical_9ch_reference_common_copy_and_projection_slice"
            ),
            "projection_weight_key": PROJECTION_WEIGHT_KEY,
            "common_state_shared": True,
            "variants": identities,
        },
    )
    imu_fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "classifier_seed": int(fold_config["classifier_seed"]),
        "source_fold_config_sha256": sha256_file(
            fold_root / "fold_config.json"
        ),
        "sensor_input_fingerprints_sha256": sha256_file(fingerprint_path),
        "sensor_model_initialization_sha256": sha256_file(
            initialization_path
        ),
        "source_residual_cache_sha256": source_cache_sha,
        "input_support_sha256": support_sha,
    }
    rf.save_or_validate_json(
        fold_root / "imu_fold_config.json",
        imu_fold_config,
    )
    fold_config = {
        **fold_config,
        "sensor_input_fingerprints": input_fingerprints,
        "sensor_model_initialization": identities,
        "imu_fold_config_sha256": sha256_file(
            fold_root / "imu_fold_config.json"
        ),
    }
    return fold_root, full_inputs, fold_config, identities


def task_root_for(
    output_dir: Path,
    subject: str,
    variant_name: str,
) -> Path:
    return output_dir / f"loso_{subject}" / variant_name


def validate_fold_binding_files(
    fold_root: Path,
    config: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Validate the fold-level hash chain used by every sensor variant."""

    required = (
        fold_root / "fold_config.json",
        fold_root / "input_support.npz",
        fold_root / "sensor_input_fingerprints.json",
        fold_root / "sensor_model_initialization.json",
        fold_root / "imu_fold_config.json",
    )
    if not all(path.exists() for path in required):
        return None
    fold_config = rf._load_json(fold_root / "fold_config.json")
    fingerprints_path = fold_root / "sensor_input_fingerprints.json"
    initialization_path = fold_root / "sensor_model_initialization.json"
    fingerprints = rf._load_json(fingerprints_path)
    initializations = rf._load_json(initialization_path)
    source_cache_sha = str(
        fold_config["source"]["source_residual_cache_sha256"]
    )
    support_sha = sha256_file(fold_root / "input_support.npz")
    if support_sha != str(fold_config["source"]["input_support_sha256"]):
        raise ValueError(f"Fold support hash changed: {subject}")
    expected_imu_fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "classifier_seed": int(fold_config["classifier_seed"]),
        "source_fold_config_sha256": sha256_file(
            fold_root / "fold_config.json"
        ),
        "sensor_input_fingerprints_sha256": sha256_file(
            fingerprints_path
        ),
        "sensor_model_initialization_sha256": sha256_file(
            initialization_path
        ),
        "source_residual_cache_sha256": source_cache_sha,
        "input_support_sha256": support_sha,
    }
    if rf._load_json(
        fold_root / "imu_fold_config.json"
    ) != expected_imu_fold_config:
        raise ValueError(f"IMU fold binding changed: {subject}")
    if fingerprints.get("protocol_fingerprint") != config[
        "protocol_fingerprint"
    ]:
        raise ValueError(f"Sensor fingerprints protocol changed: {subject}")
    if initializations.get("protocol_fingerprint") != config[
        "protocol_fingerprint"
    ]:
        raise ValueError(f"Initialization protocol changed: {subject}")
    if fingerprints.get("source_residual_cache_sha256") != source_cache_sha:
        raise ValueError(f"Sensor fingerprints source changed: {subject}")
    if fingerprints.get("input_support_sha256") != support_sha:
        raise ValueError(f"Sensor fingerprints support changed: {subject}")
    return fold_config, fingerprints, initializations


def imu_metadata_payload(
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(variant["variant"])
    identity = fold_config["sensor_model_initialization"][name]
    return {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": f"{fold_config['test_subject']}/{name}",
        "experiment_id": variant["experiment_id"],
        "variant": name,
        "display_name": variant["display_name"],
        "sensor_count": int(variant["sensor_count"]),
        "n_channels": int(variant["n_channels"]),
        "channel_indices": list(variant["channel_indices"]),
        "channel_names": list(variant["channel_names"]),
        "input_shape": [
            "batch",
            int(variant["n_channels"]),
            HISTORY_SAMPLES,
        ],
        "sensor_input_sha256": fold_config[
            "sensor_input_fingerprints"
        ][name],
        "source_residual_cache_sha256": fold_config["source"][
            "source_residual_cache_sha256"
        ],
        "input_support_sha256": fold_config["source"][
            "input_support_sha256"
        ],
        "source_nbm_best_sha256": fold_config["source"][
            "source_nbm_best_sha256"
        ],
        "parameter_count": int(identity["parameter_count"]),
        "initial_state_sha256": identity["initial_state_sha256"],
        "common_state_sha256": identity["common_state_sha256"],
        "projection_weight_sha256": identity[
            "projection_weight_sha256"
        ],
        "initialization_strategy": (
            "canonical_9ch_reference_common_copy_and_projection_slice"
        ),
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
    }


def save_imu_metadata_completion(
    task_root: Path,
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    subject = str(fold_config["test_subject"])
    name = str(variant["variant"])
    classifier_done_path = task_root / "DONE.json"
    classifier_done_sha = sha256_file(classifier_done_path)
    metadata = imu_metadata_payload(config, fold_config, variant)
    metadata_path = task_root / "imu_metadata.json"
    metadata_done_path = task_root / "IMU_METADATA_DONE.json"
    task_id = f"{subject}/{name}/imu_metadata"
    completed = validate_done(
        metadata_done_path,
        stage="imu_variant_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        upstream_sha256=classifier_done_sha,
    )
    if completed is not None:
        if rf._load_json(metadata_path) != metadata:
            raise ValueError(f"IMU metadata changed: {subject}/{name}")
        return metadata
    rf.save_or_validate_json(metadata_path, metadata)
    atomic_json_dump(
        done_payload(
            stage="imu_variant_metadata",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id=task_id,
            upstream_sha256=classifier_done_sha,
            relative_to=task_root,
            artifacts={"metadata": metadata_path},
        ),
        metadata_done_path,
    )
    return metadata


def train_variant_classifier(
    args: argparse.Namespace,
    config: dict[str, Any],
    variant: dict[str, Any],
    task_root: Path,
    fold_config: dict[str, Any],
    inputs: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    device: torch.device,
) -> dict[str, Any]:
    name = str(variant["variant"])
    identity = fold_config["sensor_model_initialization"][name]
    classifier_fold_config = {
        **fold_config,
        "reference_initial_state_sha256": identity[
            "initial_state_sha256"
        ],
    }
    classifier_config = {
        "protocol_fingerprint": config["protocol_fingerprint"],
        "shared_parameter_count": int(identity["parameter_count"]),
    }
    with _patched_shared_model_builder(
        args,
        variant,
        int(fold_config["classifier_seed"]),
    ):
        metrics = rf.train_classifier_resumable(
            args,
            classifier_config,
            variant,
            task_root,
            classifier_fold_config,
            inputs,
            dataset,
            windows,
            device,
        )
    expected = {
        "experiment_id": variant["experiment_id"],
        "variant": name,
        "input": INPUT_NAME,
        "initial_state_sha256": identity["initial_state_sha256"],
        "source_residual_sha256": fold_config["source"][
            "source_residual_cache_sha256"
        ],
        "input_support_sha256": fold_config["source"][
            "input_support_sha256"
        ],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"Classifier identity mismatch: "
                f"{fold_config['test_subject']}/{name}/{key}"
            )
    metadata = save_imu_metadata_completion(
        task_root,
        config,
        fold_config,
        variant,
    )
    return {**metrics, **metadata}


def _load_completed_cell(
    output_dir: Path,
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    name = str(variant["variant"])
    root = task_root_for(output_dir, subject, name)
    done = validate_done(
        root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{name}",
    )
    if done is None:
        return None
    fold_root = output_dir / f"loso_{subject}"
    binding = validate_fold_binding_files(
        fold_root,
        config,
        subject,
    )
    if binding is None:
        raise FileNotFoundError(f"Incomplete IMU fold binding: {subject}")
    fold_config, fingerprints, initializations = binding
    source_cache_sha = str(
        fold_config["source"]["source_residual_cache_sha256"]
    )
    support_sha = str(fold_config["source"]["input_support_sha256"])
    identity = initializations["variants"][name]
    if done.get("source_residual_sha256") != source_cache_sha:
        raise ValueError(f"Classifier source cache mismatch: {root}")
    if done.get("input_support_sha256") != support_sha:
        raise ValueError(f"Classifier support mismatch: {root}")
    if done.get("initial_state_sha256") != identity[
        "initial_state_sha256"
    ]:
        raise ValueError(f"Classifier initialization mismatch: {root}")

    metadata_done = validate_done(
        root / "IMU_METADATA_DONE.json",
        stage="imu_variant_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{name}/imu_metadata",
        upstream_sha256=sha256_file(root / "DONE.json"),
    )
    if metadata_done is None:
        return None
    in_memory_fold = {
        **fold_config,
        "sensor_input_fingerprints": fingerprints["variants"],
        "sensor_model_initialization": initializations["variants"],
    }
    expected_metadata = imu_metadata_payload(
        config,
        in_memory_fold,
        variant,
    )
    metadata = rf._load_json(root / "imu_metadata.json")
    if metadata != expected_metadata:
        raise ValueError(f"IMU metadata mismatch: {root}")
    metrics = rf._load_json(root / "metrics.json")
    expected_metrics = {
        "experiment_id": variant["experiment_id"],
        "variant": name,
        "test_subject": subject,
        "nbm": SOURCE_NBM,
        "input": INPUT_NAME,
        "initial_state_sha256": identity["initial_state_sha256"],
        "source_residual_sha256": source_cache_sha,
        "input_support_sha256": support_sha,
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise ValueError(f"Metrics identity mismatch: {root}/{key}")
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        expected_keys = {"window_index", "y_true", "y_prob", "y_pred"}
        if set(payload.files) != expected_keys:
            raise ValueError(f"Prediction array set mismatch: {root}")
        arrays = {
            key: np.asarray(payload[key])
            for key in expected_keys
        }
    with np.load(fold_root / "input_support.npz", allow_pickle=False) as support:
        if not np.array_equal(
            arrays["window_index"],
            support["test_anchor_window_index"],
        ):
            raise ValueError(f"Prediction support changed: {root}")
        if not np.array_equal(arrays["y_true"], support["test_y"]):
            raise ValueError(f"Prediction labels changed: {root}")
    return {**metrics, **metadata}, arrays


def stable_bootstrap_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big", signed=False)
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
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    rng = np.random.default_rng(int(seed))
    rows = rng.integers(
        0,
        len(values),
        size=(int(samples), len(values)),
        endpoint=False,
    )
    bootstrap = values[rows].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    tolerance = 1e-12
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_paired_subjects": int(len(values)),
        "bootstrap_samples": int(samples),
        "wins": int((values > tolerance).sum()),
        "ties": int((np.abs(values) <= tolerance).sum()),
        "losses": int((values < -tolerance).sum()),
    }


def _format_mean_sd(
    summary: Mapping[str, Any],
    metric: str,
) -> str:
    payload = summary.get(metric, {})
    if not isinstance(payload, Mapping):
        return ""
    mean, std = payload.get("mean"), payload.get("std")
    if mean is None or std is None:
        return ""
    digits = 3 if metric in {
        "false_alarm_events_per_hour",
        "median_detection_delay_sec",
    } else 4
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def refresh_summaries(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    variants = list(config["variants"])
    rows_by_variant: dict[str, dict[str, dict[str, Any]]] = {
        str(variant["variant"]): {} for variant in variants
    }
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    aggregate_payload: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []

    for variant in variants:
        name = str(variant["variant"])
        group: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed_subjects: list[str] = []
        for subject in EXPECTED_LOSO_SUBJECTS:
            cell = _load_completed_cell(
                output_dir,
                config,
                variant,
                subject,
            )
            if cell is None:
                continue
            metrics, arrays = cell
            group.append(metrics)
            fold_rows.append(metrics)
            rows_by_variant[name][subject] = metrics
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
            rf.prediction_metrics(
                np.concatenate(truths),
                np.concatenate(probabilities),
                np.concatenate(predictions),
            )
            if truths
            else None
        )
        aggregate_payload[variant["experiment_id"]] = {
            **variant,
            "completed_folds": completed_subjects,
            "subject_macro": macro,
            "pooled": pooled,
        }
        status = (
            "complete"
            if completed_subjects == list(EXPECTED_LOSO_SUBJECTS)
            else ("partial" if completed_subjects else "pending")
        )
        manifest_rows.append(
            {
                "experiment_id": variant["experiment_id"],
                "variant": name,
                "display_name": variant["display_name"],
                "sensor_count": variant["sensor_count"],
                "n_channels": variant["n_channels"],
                "channel_indices": ",".join(
                    map(str, variant["channel_indices"])
                ),
                "channel_names": ",".join(variant["channel_names"]),
                "parameter_count": variant["parameter_count"],
                "expected_folds": len(EXPECTED_LOSO_SUBJECTS),
                "completed_folds": len(completed_subjects),
                "status": status,
                "completed_subjects": ",".join(completed_subjects),
            }
        )
        aggregate_row: dict[str, Any] = {
            "experiment_id": variant["experiment_id"],
            "variant": name,
            "display_name": variant["display_name"],
            "sensor_count": variant["sensor_count"],
            "n_channels": variant["n_channels"],
            "parameter_count": variant["parameter_count"],
            "completed_folds": len(completed_subjects),
        }
        for metric in CLASSIFICATION_METRICS:
            aggregate_row[f"{metric}_mean"] = macro[metric]["mean"]
            aggregate_row[f"{metric}_std"] = macro[metric]["std"]
        aggregate_rows.append(aggregate_row)
        publication_rows.append(
            {
                "IMU combination": variant["display_name"],
                "Sensors": variant["sensor_count"],
                "Channels": variant["n_channels"],
                "PR-AUC": _format_mean_sd(macro, "pr_auc"),
                "BA": _format_mean_sd(macro, "balanced_accuracy"),
                "Macro-F1": _format_mean_sd(macro, "macro_f1"),
                "AUROC": _format_mean_sd(macro, "roc_auc"),
                "FoG Recall": _format_mean_sd(macro, "fog_recall"),
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
                "Delay (s)": _format_mean_sd(
                    macro,
                    "median_detection_delay_sec",
                ),
                "Completed folds": len(completed_subjects),
            }
        )
        efficiency_rows.append(
            {
                "variant": name,
                "display_name": variant["display_name"],
                "sensor_count": variant["sensor_count"],
                "n_channels": variant["n_channels"],
                "input_values_per_window": (
                    int(variant["n_channels"]) * HISTORY_SAMPLES
                ),
                "float32_input_bytes_per_window": (
                    int(variant["n_channels"]) * HISTORY_SAMPLES * 4
                ),
                "input_bandwidth_ratio_vs_all": variant[
                    "input_bandwidth_ratio_vs_all"
                ],
                "input_projection_parameters": variant[
                    "input_projection_parameters"
                ],
                "total_parameter_count": variant["parameter_count"],
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        differences: list[float] = []
        common_subjects: list[str] = []
        new_name = comparison["new"]
        reference_name = comparison["reference"]
        for subject in EXPECTED_LOSO_SUBJECTS:
            new = rows_by_variant[new_name].get(subject)
            reference = rows_by_variant[reference_name].get(subject)
            if new is None or reference is None:
                continue
            if new.get("pr_auc") is None or reference.get("pr_auc") is None:
                continue
            common_subjects.append(subject)
            differences.append(
                float(new["pr_auc"]) - float(reference["pr_auc"])
            )
        effect = paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                comparison["comparison_id"],
            ),
        )
        comparison_rows.append(
            {
                **comparison,
                "common_subjects": ",".join(common_subjects),
                **effect,
                "bootstrap_seed": int(config["bootstrap_seed"]),
            }
        )

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["pr_auc_mean"])
            if row["pr_auc_mean"] is not None
            else float("inf"),
            row["variant"],
        )
    )
    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(aggregate_rows, start=1)
    ]
    completed_cells = len(fold_rows)
    expected_cells = int(config["expected_classifier_cells"])
    complete = completed_cells == expected_cells
    formal_complete = complete and bool(config["reportable"])
    best_experiment = (
        ranked_rows[0]["experiment_id"]
        if formal_complete and ranked_rows
        else None
    )
    support_subjects: list[str] = []
    for subject in EXPECTED_LOSO_SUBJECTS:
        binding = validate_fold_binding_files(
            output_dir / f"loso_{subject}",
            config,
            subject,
        )
        if binding is not None:
            support_subjects.append(subject)
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "variants": list(IMU_VARIANTS),
            "shared_support_contract": (
                "All seven variants use the same fold-level anchors, "
                "eight-block history chains, labels, and source residual "
                "cache; only ordered channel slicing differs."
            ),
            "completed_support_subjects": support_subjects,
            "expected_subjects": list(EXPECTED_LOSO_SUBJECTS),
            "complete": support_subjects == list(EXPECTED_LOSO_SUBJECTS),
        },
        output_dir / "support_equivalence.json",
    )

    fold_columns = [
        "experiment_id",
        "variant",
        "display_name",
        "sensor_count",
        "n_channels",
        "channel_indices",
        "channel_names",
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
        "parameter_count",
        "initial_state_sha256",
        "common_state_sha256",
        "projection_weight_sha256",
        "sensor_input_sha256",
        "source_residual_cache_sha256",
        "input_support_sha256",
        "source_nbm_best_sha256",
    ]
    manifest_columns = [
        "experiment_id",
        "variant",
        "display_name",
        "sensor_count",
        "n_channels",
        "channel_indices",
        "channel_names",
        "parameter_count",
        "expected_folds",
        "completed_folds",
        "status",
        "completed_subjects",
    ]
    aggregate_columns = [
        "rank",
        "experiment_id",
        "variant",
        "display_name",
        "sensor_count",
        "n_channels",
        "parameter_count",
        "completed_folds",
        *[
            field
            for metric in CLASSIFICATION_METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ],
    ]
    _write_csv(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    _write_csv(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        manifest_columns,
    )
    _write_csv(
        output_dir / "aggregate_summary.csv",
        ranked_rows,
        aggregate_columns,
    )
    _write_csv(
        output_dir / "publication_table.csv",
        publication_rows,
        [
            "IMU combination",
            "Sensors",
            "Channels",
            "PR-AUC",
            "BA",
            "Macro-F1",
            "AUROC",
            "FoG Recall",
            "Specificity",
            "FoG Precision",
            "FoG F1",
            "Event Sensitivity",
            "FA/h",
            "Delay (s)",
            "Completed folds",
        ],
    )
    _write_csv(
        output_dir / "paired_pr_auc_deltas.csv",
        comparison_rows,
        [
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
        ],
    )
    _write_csv(
        output_dir / "sensor_efficiency.csv",
        efficiency_rows,
        [
            "variant",
            "display_name",
            "sensor_count",
            "n_channels",
            "input_values_per_window",
            "float32_input_bytes_per_window",
            "input_bandwidth_ratio_vs_all",
            "input_projection_parameters",
            "total_parameter_count",
        ],
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "aggregation_unit": "held_out_subject",
            "ranking_metric": "subject_macro_pr_auc_mean",
            "best_experiment": best_experiment,
            "experiments": aggregate_payload,
            "paired_pr_auc_comparisons": comparison_rows,
        },
        output_dir / "aggregate_metrics.json",
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_classifier_cells": expected_cells,
            "completed_classifier_cells": completed_cells,
            "status": (
                "complete"
                if formal_complete
                else ("smoke_complete" if complete else "partial")
            ),
            "reportable": bool(config["reportable"]),
            "best_experiment": best_experiment,
        },
        output_dir / "status.json",
    )


def initialize_protocol(
    args: argparse.Namespace,
    device: torch.device,
    worker_mode: bool,
) -> tuple[dict[str, Any], DaphnetDataset, WindowTable]:
    source_manifest, source_config = rf.build_source_manifest(
        args.source_suite_dir,
        verify_artifacts=not worker_mode,
    )
    if source_config.get("suite_version") != SOURCE_SUITE_VERSION:
        raise ValueError("Unexpected source suite version")
    dataset, windows, data_sha256 = rf.load_dataset_and_windows(
        args.data_dir,
        source_config,
    )
    config = build_protocol(
        args,
        source_manifest,
        source_config,
        dataset,
        windows,
        data_sha256,
        device,
    )
    config_path = args.output_dir / "config.json"
    if worker_mode and not config_path.exists():
        raise RuntimeError(
            "Missing config.json; initialize with --finalize-only first"
        )
    if config_path.exists():
        existing = rf._load_json(config_path)
        if existing.get("protocol_fingerprint") != config[
            "protocol_fingerprint"
        ]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new output "
                "directory"
            )
    if not worker_mode:
        atomic_json_dump(config, config_path)
    runtime_fields = {
        "data_dir",
        "source_suite_dir",
        "output_dir",
        "device",
        "num_workers",
        "resume",
        "smoke",
    }
    run_manifest = {
        key: value
        for key, value in config.items()
        if key not in runtime_fields
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if not run_manifest_path.exists():
            raise RuntimeError("Missing run_manifest.json for worker")
        if rf._load_json(run_manifest_path) != run_manifest:
            raise ValueError(
                f"Saved JSON is incompatible: {run_manifest_path}"
            )
    else:
        rf.save_or_validate_json(run_manifest_path, run_manifest)
    return config, dataset, windows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.source_suite_dir = args.source_suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    rf.validate_output_path(
        args.output_dir,
        args.source_suite_dir,
        args.data_dir,
    )
    worker_mode = bool(str(args.worker_fold).strip())
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = rf.resolve_device(args.device)
    rf.set_seed(args.seed, args.deterministic)
    configured_folds = rf.parse_folds(
        args.folds,
        list(EXPECTED_LOSO_SUBJECTS),
    )
    if tuple(configured_folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("This strict suite requires --folds all")
    execution_folds = list(configured_folds)
    if worker_mode:
        selected = rf.parse_folds(
            str(args.worker_fold),
            list(EXPECTED_LOSO_SUBJECTS),
        )
        if len(selected) != 1:
            raise ValueError("--worker-fold must resolve to one subject")
        execution_folds = selected

    config, dataset, windows = initialize_protocol(
        args,
        device,
        worker_mode,
    )
    environment = rf.environment_payload(device)
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
        atomic_json_dump(
            environment,
            args.output_dir / "environment.json",
        )
        refresh_summaries(args.output_dir, config)

    print(
        f"[INFO] suite={SUITE_VERSION} device={device} "
        f"source={args.source_suite_dir} folds={execution_folds} "
        f"imu_variants={list(IMU_VARIANTS)} "
        f"classifier=TCN-M/RF{TCN_M_RF_SAMPLES}",
        flush=True,
    )
    if args.finalize_only:
        refresh_summaries(args.output_dir, config)
        print("[INFO] finalize-only: root summaries refreshed", flush=True)
        print(
            json.dumps(
                rf._load_json(args.output_dir / "status.json"),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    completed_this_run = 0
    for subject in execution_folds:
        fold_root, full_inputs, fold_config, identities = prepare_fold(
            args,
            config,
            dataset,
            windows,
            subject,
        )
        print(
            f"[fold {subject}] train={fold_config['train_subjects']} "
            f"val={fold_config['val_subject']} "
            f"anchors={fold_config['history_anchor_counts']}",
            flush=True,
        )
        common_hashes: set[str] = set()
        for variant in config["variants"]:
            name = str(variant["variant"])
            completed = _load_completed_cell(
                args.output_dir,
                config,
                variant,
                subject,
            )
            if completed is not None:
                metrics = completed[0]
                print(
                    f"[fold {subject}] {variant['display_name']} "
                    "validated complete; skip",
                    flush=True,
                )
            else:
                inputs = subset_history_inputs(full_inputs, variant)
                metrics = train_variant_classifier(
                    args,
                    config,
                    variant,
                    task_root_for(args.output_dir, subject, name),
                    fold_config,
                    inputs,
                    dataset,
                    windows,
                    device,
                )
                del inputs
            common_hashes.add(str(metrics["common_state_sha256"]))
            completed_this_run += 1
            print(
                f"[fold {subject}] {variant['display_name']} "
                f"C={variant['n_channels']} "
                f"PR-AUC={metrics['pr_auc']:.4f} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"FoG-F1={metrics['fog_f1']:.4f}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                args.stop_after_completed_tasks > 0
                and completed_this_run >= args.stop_after_completed_tasks
            ):
                raise RuntimeError(
                    "Intentional stop after completed classifier tasks"
                )
        if len(common_hashes) != 1:
            raise AssertionError(
                f"Common initial tensors differ in fold {subject}"
            )
        del full_inputs, identities
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if worker_mode:
        print(
            json.dumps(
                {
                    "suite_version": SUITE_VERSION,
                    "protocol_fingerprint": config[
                        "protocol_fingerprint"
                    ],
                    "worker_fold": execution_folds[0],
                    "classifier_cells_visited": completed_this_run,
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
            rf._load_json(args.output_dir / "status.json"),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
