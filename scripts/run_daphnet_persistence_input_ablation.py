#!/usr/bin/env python
"""Strict Persistence input-representation ablation with a fixed TCN-M.

The suite freezes the canonical Daphnet Persistence NBM and compares four
representations on exactly the same LOSO support:

* robust-scaled raw target blocks;
* raw minus the Persistence mean;
* the un-clipped uncertainty-standardised error;
* the canonical standardised error clipped to [-12, 12].

Each classifier receives eight chronological, non-overlapping 0.5-second
blocks as a [batch, 9, 256] four-second history.  Prediction and classifier
anchors remain on the canonical 0.25-second grid.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_3imu_nbm_suite as core
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import (
    DaphnetDataset,
    RobustChannelScaler,
    WindowTable,
)
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.histories import (
    HistoryPlan,
    make_block_history_input,
    make_common_history_plan,
)
from cnbr_fog.nbm import PersistenceNBM
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    done_payload,
    sha256_file,
    validate_checkpoint,
    validate_done,
)


SUITE_VERSION = (
    "daphnet_persistence_input_ablation_h4_tcnm_stride025_loso.v1"
)
SOURCE_SUITE_VERSION = rf.SOURCE_SUITE_VERSION
SOURCE_NBM = rf.SOURCE_NBM
HISTORY_SECONDS = 4.0
HISTORY_SAMPLES = 256
HISTORY_BLOCKS = 8
CONTEXT_SAMPLES = 128
HORIZON_SAMPLES = 32
STRIDE_SAMPLES = 16
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
EXPECTED_CHANNEL_NAMES = rf.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = rf.EXPECTED_LOSO_SUBJECTS
CLASSIFICATION_METRICS = tuple(rf.CLASSIFICATION_METRICS)
FORMULA_ATOL = 5e-6
FORMULA_RTOL = 1e-6

REPRESENTATIONS: dict[str, dict[str, Any]] = {
    "raw_support_matched": {
        "display_name": "Raw-support-matched",
        "source_key": "raw",
        "uses_mu": False,
        "uses_sigma": False,
        "uses_residual_clip": False,
        "definition": "robust-scaled raw target",
        "scientific_question": "matched-support raw IMU reference",
    },
    "error_x_minus_mu": {
        "display_name": "x - mu",
        "source_key": "error",
        "uses_mu": True,
        "uses_sigma": False,
        "uses_residual_clip": False,
        "definition": "raw target minus Persistence mean",
        "scientific_question": "incremental contribution of centering",
    },
    "standardized_error": {
        "display_name": "(x - mu) / sigma",
        "source_key": "standardized_error",
        "uses_mu": True,
        "uses_sigma": True,
        "uses_residual_clip": False,
        "definition": "un-clipped uncertainty-standardised error",
        "scientific_question": "incremental contribution of uncertainty scaling",
    },
    "standardized_error_clip12": {
        "display_name": "clip((x - mu) / sigma, -12, 12)",
        "source_key": "standardized_error_clip12",
        "uses_mu": True,
        "uses_sigma": True,
        "uses_residual_clip": True,
        "definition": "canonical clipped standardised error",
        "scientific_question": "incremental contribution of residual clipping",
    },
}

COMPARISONS: tuple[dict[str, str], ...] = (
    {
        "comparison_id": "B_minus_A",
        "new": "error_x_minus_mu",
        "reference": "raw_support_matched",
        "interpretation": "Persistence centering contribution",
    },
    {
        "comparison_id": "C_minus_B",
        "new": "standardized_error",
        "reference": "error_x_minus_mu",
        "interpretation": "uncertainty standardisation contribution",
    },
    {
        "comparison_id": "D_minus_C",
        "new": "standardized_error_clip12",
        "reference": "standardized_error",
        "interpretation": "residual-space clipping contribution",
    },
    {
        "comparison_id": "D_minus_A",
        "new": "standardized_error_clip12",
        "reference": "raw_support_matched",
        "interpretation": "complete representation contribution",
    },
)

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_persistence_input_ablation.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/resume.py",
)

DEFAULT_DATA_DIR = rf.DEFAULT_DATA_DIR
DEFAULT_SOURCE_SUITE_DIR = rf.DEFAULT_SOURCE_SUITE_DIR
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daphnet_persistence_input_ablation_h4_tcnm_stride025_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict Daphnet Persistence input-representation ablation "
            "with residual_h4s support and TCN-M"
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
            "Allow reduced classifier epochs/windows for pipeline testing; "
            "smoke outputs can never receive a formal completion marker"
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
        help="Development-only smoke-test stop hook",
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
    if args.max_classifier_windows < 0 or args.num_workers < 0:
        raise ValueError("Window cap and num-workers must be non-negative")
    if 0 < args.max_classifier_windows < 2:
        raise ValueError("--max-classifier-windows must be zero or at least two")
    if args.stop_after_completed_tasks < 0:
        raise ValueError("--stop-after-completed-tasks must be non-negative")
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
                "Formal protocol options changed; use the canonical values "
                f"or add --smoke for a non-reportable run: {changed}"
            )


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def experiment_id(name: str) -> str:
    return f"persistence_h4s_tcnm__{name}"


def representation_variant(name: str) -> dict[str, Any]:
    definition = REPRESENTATIONS[name]
    return {
        "variant": name,
        "experiment_id": experiment_id(name),
        **definition,
        "dilations": list(TCN_M_DILATIONS),
        "receptive_field_samples": TCN_M_RF_SAMPLES,
        "receptive_field_seconds": TCN_M_RF_SAMPLES / 64.0,
    }


def build_protocol(
    args: argparse.Namespace,
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    required_source_values: dict[str, Any] = {
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "seed": 42,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
    }
    for key, expected in required_source_values.items():
        observed = source_config.get(key)
        if isinstance(expected, float):
            compatible = (
                observed is not None
                and math.isclose(
                    float(observed),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        else:
            compatible = observed == expected
        if not compatible:
            raise ValueError(
                "Input ablation must retain the canonical source protocol: "
                f"{key} expected={expected!r}, source={observed!r}"
            )

    if rf.convolutional_receptive_field(TCN_M_DILATIONS) != TCN_M_RF_SAMPLES:
        raise AssertionError("Canonical TCN-M receptive field changed")
    variants = [
        representation_variant(name) for name in REPRESENTATIONS
    ]
    counts: set[int] = set()
    initial_hashes: set[str] = set()
    for variant in variants:
        rf.set_seed(args.seed, args.deterministic)
        model = rf.build_model(
            in_channels=dataset.n_channels,
            hidden_channels=args.classifier_hidden,
            dropout=args.classifier_dropout,
            dilations=TCN_M_DILATIONS,
        )
        count = rf.parameter_count(model)
        state_hash = rf.state_dict_sha256(model.state_dict())
        counts.add(count)
        initial_hashes.add(state_hash)
        variant["parameter_count"] = count
        variant["reference_initial_state_sha256"] = state_hash
        del model
    if len(counts) != 1 or len(initial_hashes) != 1:
        raise AssertionError("Representations do not share architecture/init")

    scientific = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "channel_names": list(dataset.channel_names),
        "n_channels": int(dataset.n_channels),
        "subjects": list(dataset.subjects),
        "excluded_subjects": list(source_config["excluded_subjects"]),
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbm": SOURCE_NBM,
        "nbm_policy": (
            "Freeze canonical Persistence mean and learned sigma; do not "
            "retrain the normal-behaviour model."
        ),
        "context_samples": CONTEXT_SAMPLES,
        "persistence_effective_context_samples": 1,
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "history_block_spacing_samples": HORIZON_SAMPLES,
        "classification_target_definition": (
            "label of the final 32-sample target block"
        ),
        "robust_clip": float(source_config["robust_clip"]),
        "residual_clip": float(source_config["residual_clip"]),
        "representations": variants,
        "expected_experiments": len(variants),
        "expected_representation_cache_tasks": len(EXPECTED_LOSO_SUBJECTS),
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
            "parameter_count": counts.pop(),
            "global_pooling": "mean_and_max_over_full_4s_input",
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
            "same_fold_scaler": True,
            "same_persistence_checkpoint_and_sigma": True,
            "same_anchor_history_and_labels": True,
            "same_tcn_m_architecture": True,
            "same_classifier_seed_and_initial_state": True,
            "same_epoch_shuffle_rule": "classifier_seed + epoch",
            "independent_validation_early_stopping": True,
            "independent_validation_threshold": True,
            "test_subject_never_selects_model_or_threshold": True,
        },
        "window_count": int(len(windows)),
        "window_class_counts": np.bincount(
            windows.label,
            minlength=2,
        ).astype(int).tolist(),
    }
    fingerprint = canonical_fingerprint(scientific)
    return {
        **scientific,
        "protocol_fingerprint": fingerprint,
        "data_dir": str(args.data_dir),
        "source_suite_dir": str(args.source_suite_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "num_workers": int(args.num_workers),
        "resume": bool(args.resume),
        "smoke": bool(args.smoke),
    }


def _load_scaler(path: Path) -> RobustChannelScaler:
    payload = rf._load_json(path)
    center = np.asarray(payload.get("center"), dtype=np.float32)
    scale = np.asarray(payload.get("scale"), dtype=np.float32)
    clip = float(payload.get("clip"))
    if center.shape != (9,) or scale.shape != (9,):
        raise ValueError(f"Invalid source scaler shape: {path}")
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError(f"Non-finite source scaler: {path}")
    if np.any(scale <= 0) or not math.isfinite(clip) or clip <= 0:
        raise ValueError(f"Invalid source scaler values: {path}")
    return RobustChannelScaler(center=center, scale=scale, clip=clip)


def _source_bundle(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    subject: str,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
    RobustChannelScaler,
    PersistenceNBM,
    dict[str, Any],
]:
    source_features, cache_provenance = rf._load_source_cache(
        args,
        dict(config),
        subject,
    )
    fold_root = args.source_suite_dir / f"loso_{subject}"
    fold_config_path = fold_root / "fold_config.json"
    scaler_path = fold_root / "scaler.json"
    split_path = fold_root / "split_indices.npz"
    support_path = fold_root / "history_support.npz"
    best_path = fold_root / SOURCE_NBM / "nbm" / "best.pt"

    source_fold_config = rf._load_json(fold_config_path)
    if source_fold_config.get("protocol_fingerprint") != config["source"][
        "source_protocol_fingerprint"
    ]:
        raise ValueError(f"Source fold protocol mismatch: {subject}")
    if source_fold_config.get("test_subject") != subject:
        raise ValueError(f"Source fold subject mismatch: {subject}")
    expected_fold = config["source"]["folds"][subject]
    if sha256_file(fold_config_path) != expected_fold[
        "source_fold_config_sha256"
    ]:
        raise ValueError(f"Source fold config hash changed: {subject}")
    if sha256_file(support_path) != expected_fold[
        "source_history_support_sha256"
    ]:
        raise ValueError(f"Source history support hash changed: {subject}")
    if sha256_file(best_path) != expected_fold["source_nbm_best_sha256"]:
        raise ValueError(f"Source Persistence checkpoint changed: {subject}")

    scaler_payload = rf._load_json(scaler_path)
    if scaler_payload != source_fold_config.get("scaler"):
        raise ValueError(f"Source scaler/fold config mismatch: {subject}")
    scaler = _load_scaler(scaler_path)
    with np.load(split_path, allow_pickle=False) as payload:
        expected_split_keys = {
            "train_window_index",
            "validation_window_index",
            "test_window_index",
            "normal_train_window_index",
            "normal_validation_window_index",
        }
        if set(payload.files) != expected_split_keys:
            raise ValueError(f"Unexpected source split arrays: {subject}")
        for split in ("train", "validation", "test"):
            indices = np.asarray(
                payload[f"{split}_window_index"],
                dtype=np.int64,
            )
            if not np.array_equal(
                indices,
                source_features[split]["window_index"],
            ):
                raise ValueError(
                    f"Source split/cache order changed: {subject}/{split}"
                )

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint(
        checkpoint,
        stage="nbm",
        protocol_fingerprint=config["source"][
            "source_protocol_fingerprint"
        ],
        task_id=f"loso_{subject}/{SOURCE_NBM}/nbm",
    )
    model_config = checkpoint.get("model_config", {})
    expected_model_config = {
        "name": SOURCE_NBM,
        "in_channels": 9,
        "horizon": HORIZON_SAMPLES,
        "min_log_sigma": -3.0,
        "max_log_sigma": 1.5,
    }
    if model_config != expected_model_config:
        raise ValueError(
            f"Unexpected Persistence model config: {subject}/{model_config}"
        )
    model = PersistenceNBM(
        in_channels=9,
        horizon=HORIZON_SAMPLES,
        min_log_sigma=float(model_config["min_log_sigma"]),
        max_log_sigma=float(model_config["max_log_sigma"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    source_files = {
        "source_fold_config_sha256": sha256_file(fold_config_path),
        "source_scaler_sha256": sha256_file(scaler_path),
        "source_split_indices_sha256": sha256_file(split_path),
        "source_history_support_sha256": sha256_file(support_path),
        "source_nbm_best_sha256": sha256_file(best_path),
        **cache_provenance,
    }
    source_binding_sha256 = canonical_fingerprint(source_files)
    provenance = {
        **source_files,
        "source_binding_sha256": source_binding_sha256,
        "source_fold_config": source_fold_config,
        "source_scaler": scaler_payload,
    }
    return (
        source_features,
        provenance,
        source_fold_config,
        scaler,
        model,
        checkpoint,
    )


def _array_diagnostics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "n_values": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "abs_p50": float(np.quantile(absolute, 0.50)),
        "abs_p90": float(np.quantile(absolute, 0.90)),
        "abs_p95": float(np.quantile(absolute, 0.95)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_p999": float(np.quantile(absolute, 0.999)),
        "max_abs": float(absolute.max()),
        "nonfinite_values": int((~np.isfinite(array)).sum()),
    }


@torch.no_grad()
def extract_representation_split(
    args: argparse.Namespace,
    model: PersistenceNBM,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    labels: np.ndarray,
    canonical_clipped: np.ndarray,
    scaler: RobustChannelScaler,
    residual_clip: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    loader = core.make_sequence_loader(
        dataset,
        windows,
        np.asarray(indices, dtype=np.int64),
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    model = model.to(device)
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "raw": [],
        "mu": [],
        "error": [],
        "standardized_error": [],
    }
    observed_labels: list[np.ndarray] = []
    observed_indices: list[np.ndarray] = []
    sigma_reference: np.ndarray | None = None
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        raw = sequence[:, :, CONTEXT_SAMPLES:].float()
        with torch.amp.autocast(
            device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            mu, sigma = model(context)
            standardized = (raw - mu) / sigma
        mu = mu.float()
        sigma = sigma.float()
        error = raw - mu
        if sigma_reference is None:
            sigma_reference = sigma[:1].cpu().numpy().astype(
                np.float32,
                copy=True,
            )
        if not torch.equal(
            sigma,
            sigma_reference_to_tensor(
                sigma_reference,
                sigma.shape[0],
                sigma.device,
            ),
        ):
            raise AssertionError("Persistence sigma changed across windows")
        chunks["raw"].append(raw.cpu().numpy())
        chunks["mu"].append(mu.cpu().numpy())
        chunks["error"].append(error.cpu().numpy())
        chunks["standardized_error"].append(
            standardized.float().cpu().numpy()
        )
        observed_labels.append(y.numpy())
        observed_indices.append(index.numpy())

    if sigma_reference is None:
        raise RuntimeError("Representation extraction received no windows")
    features = {
        key: np.ascontiguousarray(
            np.concatenate(values).astype(np.float32, copy=False)
        )
        for key, values in chunks.items()
    }
    features["y"] = np.concatenate(observed_labels).astype(
        np.int8,
        copy=False,
    )
    features["window_index"] = np.concatenate(observed_indices).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(features["window_index"], indices):
        raise ValueError("Representation extraction changed window order")
    if not np.array_equal(features["y"], labels):
        raise ValueError("Representation extraction changed labels")
    expected_shape = (len(indices), 9, HORIZON_SAMPLES)
    for key in ("raw", "mu", "error", "standardized_error"):
        if features[key].shape != expected_shape:
            raise ValueError(f"Unexpected representation shape: {key}")
        if not np.isfinite(features[key]).all():
            raise ValueError(f"Non-finite representation: {key}")

    formula_error = features["raw"] - features["mu"]
    if not np.allclose(
        features["error"],
        formula_error,
        rtol=FORMULA_RTOL,
        atol=FORMULA_ATOL,
    ):
        raise AssertionError("error != raw - mu")
    formula_standardized = features["error"] / sigma_reference
    if not np.allclose(
        features["standardized_error"],
        formula_standardized,
        rtol=FORMULA_RTOL,
        atol=FORMULA_ATOL,
    ):
        raise AssertionError("standardized_error != error / sigma")
    computed_clipped = np.clip(
        features["standardized_error"],
        -float(residual_clip),
        float(residual_clip),
    ).astype(np.float32, copy=False)
    canonical = np.asarray(canonical_clipped, dtype=np.float32)
    canonical_max_abs_diff = float(
        np.max(np.abs(computed_clipped.astype(np.float64) - canonical))
    )
    if not np.allclose(
        computed_clipped,
        canonical,
        rtol=FORMULA_RTOL,
        atol=FORMULA_ATOL,
    ):
        raise AssertionError(
            "Replayed clipped error differs from canonical residual cache: "
            f"max_abs_diff={canonical_max_abs_diff}"
        )
    # Preserve the exact completed source method as group D.  CPU/GPU exp and
    # division can differ by a few ulps, so formula equality is audited with a
    # strict tolerance while D remains byte-identical to the source cache.
    features["standardized_error_clip12"] = np.ascontiguousarray(canonical)

    diagnostics: dict[str, Any] = {
        key: _array_diagnostics(features[key])
        for key in (
            "raw",
            "mu",
            "error",
            "standardized_error",
            "standardized_error_clip12",
        )
    }
    over = np.abs(features["standardized_error"]) > float(
        residual_clip
    )
    diagnostics["standardized_error"]["clip_fraction"] = float(over.mean())
    diagnostics["standardized_error"][
        "clip_fraction_by_channel"
    ] = over.mean(axis=(0, 2)).astype(float).tolist()
    diagnostics["canonical_clip_max_abs_diff"] = canonical_max_abs_diff
    return features, diagnostics, sigma_reference


def sigma_reference_to_tensor(
    sigma: np.ndarray,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(np.asarray(sigma, dtype=np.float32)).to(
        device
    ).expand(int(batch), -1, -1)


def representation_cache_keys() -> set[str]:
    keys = {"sigma"}
    for split in ("train", "validation", "test"):
        for key in (
            "raw",
            "mu",
            "error",
            "standardized_error",
            "standardized_error_clip12",
            "y",
            "window_index",
        ):
            keys.add(f"{split}_{key}")
    return keys


def load_or_create_representation_cache(
    args: argparse.Namespace,
    config: dict[str, Any],
    subject: str,
    fold_root: Path,
    dataset: DaphnetDataset,
    windows: WindowTable,
    source_features: Mapping[str, Mapping[str, np.ndarray]],
    source_provenance: Mapping[str, Any],
    scaler: RobustChannelScaler,
    model: PersistenceNBM,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    np.ndarray,
    dict[str, Any],
    str,
]:
    cache_path = fold_root / "representation_cache.npz"
    diagnostics_path = fold_root / "representation_diagnostics.json"
    done_path = fold_root / "REPRESENTATION_CACHE_DONE.json"
    task_id = f"{subject}/input_representation_cache"
    upstream = str(source_provenance["source_binding_sha256"])
    completed = validate_done(
        done_path,
        stage="input_representation_cache",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=upstream,
    )
    if completed is not None:
        if set(completed.get("artifacts", {})) != {
            "cache",
            "diagnostics",
        }:
            raise ValueError(f"Representation cache artifacts changed: {subject}")
        with np.load(cache_path, allow_pickle=False) as payload:
            if set(payload.files) != representation_cache_keys():
                raise ValueError(
                    f"Unexpected representation cache arrays: {subject}"
                )
            features = {
                split: {
                    key: np.asarray(payload[f"{split}_{key}"])
                    for key in (
                        "raw",
                        "mu",
                        "error",
                        "standardized_error",
                        "standardized_error_clip12",
                        "y",
                        "window_index",
                    )
                }
                for split in ("train", "validation", "test")
            }
            sigma = np.asarray(payload["sigma"], dtype=np.float32)
        diagnostics = rf._load_json(diagnostics_path)
        return features, sigma, diagnostics, sha256_file(cache_path)

    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    sigma: np.ndarray | None = None
    for split in ("train", "validation", "test"):
        source = source_features[split]
        current, split_diagnostics, current_sigma = (
            extract_representation_split(
                args,
                model,
                dataset,
                windows,
                np.asarray(source["window_index"], dtype=np.int64),
                np.asarray(source["y"], dtype=np.int8),
                np.asarray(source["residual"], dtype=np.float32),
                scaler,
                float(config["residual_clip"]),
                device,
            )
        )
        if sigma is None:
            sigma = current_sigma
        elif not np.array_equal(sigma, current_sigma):
            raise AssertionError("Persistence sigma differs between splits")
        features[split] = current
        diagnostics[split] = split_diagnostics
    assert sigma is not None
    if sigma.shape != (1, 9, HORIZON_SAMPLES):
        raise ValueError(f"Unexpected sigma shape: {sigma.shape}")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Persistence sigma must be finite and positive")
    diagnostics["sigma"] = {
        **_array_diagnostics(sigma),
        "shape": list(sigma.shape),
        "channel_min": sigma.min(axis=(0, 2)).astype(float).tolist(),
        "channel_median": np.median(
            sigma,
            axis=(0, 2),
        ).astype(float).tolist(),
        "channel_max": sigma.max(axis=(0, 2)).astype(float).tolist(),
    }
    atomic_npz_save(
        cache_path,
        sigma=sigma,
        **{
            f"{split}_{key}": features[split][key]
            for split in ("train", "validation", "test")
            for key in (
                "raw",
                "mu",
                "error",
                "standardized_error",
                "standardized_error_clip12",
                "y",
                "window_index",
            )
        },
    )
    atomic_json_dump(diagnostics, diagnostics_path)
    atomic_json_dump(
        done_payload(
            stage="input_representation_cache",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=upstream,
            relative_to=fold_root,
            artifacts={
                "cache": cache_path,
                "diagnostics": diagnostics_path,
            },
        ),
        done_path,
    )
    return features, sigma, diagnostics, sha256_file(cache_path)


def prepare_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
    device: torch.device,
) -> tuple[
    Path,
    dict[str, dict[str, np.ndarray]],
    dict[str, HistoryPlan],
    dict[str, Any],
]:
    fold_root = args.output_dir / f"loso_{subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    (
        source_features,
        source_provenance,
        source_fold_config,
        scaler,
        model,
        checkpoint,
    ) = _source_bundle(args, config, subject)
    features, sigma, diagnostics, representation_cache_sha256 = (
        load_or_create_representation_cache(
            args,
            config,
            subject,
            fold_root,
            dataset,
            windows,
            source_features,
            source_provenance,
            scaler,
            model,
            device,
        )
    )
    del model

    plans: dict[str, HistoryPlan] = {}
    support_arrays: dict[str, np.ndarray] = {}
    source_support_path = (
        args.source_suite_dir / f"loso_{subject}" / "history_support.npz"
    )
    with np.load(source_support_path, allow_pickle=False) as source_support:
        expected_source_keys = {
            f"{split}_{suffix}"
            for split in ("train", "validation", "test")
            for suffix in ("anchor_window_index", "history_window_index")
        }
        if set(source_support.files) != expected_source_keys:
            raise ValueError(f"Unexpected source support arrays: {subject}")
        for split in ("train", "validation", "test"):
            indices = np.asarray(
                features[split]["window_index"],
                dtype=np.int64,
            )
            plan = make_common_history_plan(
                windows,
                indices,
                HORIZON_SAMPLES,
                STRIDE_SAMPLES,
                HISTORY_SAMPLES,
            )
            if len(plan.anchor_rows) == 0:
                raise RuntimeError(f"Empty history support: {subject}/{split}")
            if not np.array_equal(
                plan.anchor_window_indices,
                source_support[f"{split}_anchor_window_index"],
            ):
                raise ValueError(
                    f"Anchor support differs from source: {subject}/{split}"
                )
            chain_indices = indices[plan.max_chain_rows]
            if not np.array_equal(
                chain_indices,
                source_support[f"{split}_history_window_index"],
            ):
                raise ValueError(
                    f"History support differs from source: {subject}/{split}"
                )
            cap_this_split = (
                args.max_classifier_windows > 0
                and (split == "train" or args.smoke)
            )
            if cap_this_split:
                rows = np.arange(len(plan.anchor_rows), dtype=np.int64)
                anchor_labels = windows.label[
                    plan.anchor_window_indices
                ]
                split_offset = {
                    "train": 0,
                    "validation": 1,
                    "test": 2,
                }[split]
                selected = rf.deterministic_subsample(
                    rows,
                    args.max_classifier_windows,
                    args.seed
                    + 100
                    + EXPECTED_LOSO_SUBJECTS.index(subject)
                    + split_offset,
                    anchor_labels,
                )
                plan = plan.take(selected)
                chain_indices = indices[plan.max_chain_rows]
            plans[split] = plan
            support_arrays[f"{split}_anchor_window_index"] = (
                plan.anchor_window_indices
            )
            support_arrays[f"{split}_history_window_index"] = chain_indices
            support_arrays[f"{split}_y"] = np.asarray(
                features[split]["y"][plan.anchor_rows],
                dtype=np.int8,
            )

    support_path = fold_root / "input_support.npz"
    core.save_or_validate_npz(support_path, **support_arrays)
    support_sha256 = sha256_file(support_path)
    input_fingerprints = {
        name: canonical_fingerprint(
            {
                "representation_cache_sha256": representation_cache_sha256,
                "source_key": definition["source_key"],
                "input_support_sha256": support_sha256,
                "history_samples": HISTORY_SAMPLES,
                "history_blocks": HISTORY_BLOCKS,
            }
        )
        for name, definition in REPRESENTATIONS.items()
    }
    fingerprint_path = fold_root / "representation_input_fingerprints.json"
    core.save_or_validate_json(
        fingerprint_path,
        {
            "protocol_fingerprint": config["protocol_fingerprint"],
            "representations": input_fingerprints,
        },
    )
    source_provenance_payload = {
        key: value
        for key, value in source_provenance.items()
        if key not in {"source_fold_config", "source_scaler"}
    }
    source_provenance_payload.update(
        {
            "representation_cache_sha256": representation_cache_sha256,
            "representation_cache_done_sha256": sha256_file(
                fold_root / "REPRESENTATION_CACHE_DONE.json"
            ),
            "input_support_sha256": support_sha256,
            "input_fingerprints_sha256": sha256_file(fingerprint_path),
        }
    )

    fold_index = EXPECTED_LOSO_SUBJECTS.index(subject)
    classifier_seed = args.seed + 10000 + fold_index
    rf.set_seed(classifier_seed, args.deterministic)
    reference_model = rf.build_model(
        in_channels=dataset.n_channels,
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=TCN_M_DILATIONS,
    )
    initial_state_sha256 = rf.state_dict_sha256(
        reference_model.state_dict()
    )
    del reference_model

    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "val_subject": source_fold_config["val_subject"],
        "train_subjects": source_fold_config["train_subjects"],
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256": initial_state_sha256,
        "source": source_provenance_payload,
        "source_scaler": source_provenance["source_scaler"],
        "source_persistence_model_config": checkpoint["model_config"],
        "source_persistence_parameter_count": int(
            sum(value.numel() for value in checkpoint["model_state"].values())
        ),
        "sigma_shape": list(sigma.shape),
        "representation_cache_sha256": representation_cache_sha256,
        "representation_input_fingerprints": input_fingerprints,
        "input_support_sha256": support_sha256,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "history_anchor_counts": {
            split: int(len(plans[split].anchor_rows))
            for split in ("train", "validation", "test")
        },
        "history_support_shared_by_all_representations": True,
        "label_support_shared_by_all_representations": True,
        "representation_diagnostics_present": bool(diagnostics),
    }
    core.save_or_validate_json(fold_root / "fold_config.json", fold_config)
    core.save_or_validate_json(
        fold_root / "source_provenance.json",
        source_provenance_payload,
    )
    return fold_root, features, plans, fold_config


def materialize_representation_inputs(
    features: Mapping[str, Mapping[str, np.ndarray]],
    plans: Mapping[str, HistoryPlan],
    representation: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    name = str(representation["variant"])
    source_key = str(representation["source_key"])
    inputs = {
        split: make_block_history_input(
            extracted=features[split],
            plan=plans[split],
            source_key=source_key,
            name=name,
            history_samples=HISTORY_SAMPLES,
            horizon_samples=HORIZON_SAMPLES,
            stride_samples=STRIDE_SAMPLES,
        )
        for split in ("train", "validation", "test")
    }
    for split, payload in inputs.items():
        tensor = np.asarray(payload[name])
        if tensor.shape[1:] != (9, HISTORY_SAMPLES):
            raise AssertionError(
                f"Input is not [B,9,256]: {name}/{split}/{tensor.shape}"
            )
        if not np.isfinite(tensor).all():
            raise ValueError(f"Non-finite history input: {name}/{split}")
    return inputs


def task_root_for(
    output_dir: Path,
    subject: str,
    representation_name: str,
) -> Path:
    return (
        output_dir
        / f"loso_{subject}"
        / representation_name
    )


def representation_metadata_payload(
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    representation: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(representation["variant"])
    return {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": f"{fold_config['test_subject']}/{name}",
        "experiment_id": representation["experiment_id"],
        "representation": name,
        "display_name": representation["display_name"],
        "definition": representation["definition"],
        "source_key": representation["source_key"],
        "uses_mu": bool(representation["uses_mu"]),
        "uses_sigma": bool(representation["uses_sigma"]),
        "uses_residual_clip": bool(
            representation["uses_residual_clip"]
        ),
        "robust_input_clip": float(config["robust_clip"]),
        "residual_clip": (
            float(config["residual_clip"])
            if representation["uses_residual_clip"]
            else None
        ),
        "representation_cache_sha256": fold_config[
            "representation_cache_sha256"
        ],
        "representation_input_sha256": fold_config[
            "representation_input_fingerprints"
        ][name],
        "input_support_sha256": fold_config["input_support_sha256"],
        "source_nbm_best_sha256": fold_config["source"][
            "source_nbm_best_sha256"
        ],
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "classifier_output_stride_seconds": (
            STRIDE_SAMPLES / float(config["sampling_rate_hz"])
        ),
    }


def save_representation_metadata_completion(
    task_root: Path,
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    representation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    subject = str(fold_config["test_subject"])
    name = str(representation["variant"])
    classifier_done_path = task_root / "DONE.json"
    classifier_done_sha256 = sha256_file(classifier_done_path)
    metadata_path = task_root / "representation_metadata.json"
    metadata_done_path = task_root / "REPRESENTATION_METADATA_DONE.json"
    task_id = f"{subject}/{name}/representation_metadata"
    completed = validate_done(
        metadata_done_path,
        stage="representation_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        upstream_sha256=classifier_done_sha256,
    )
    if completed is not None:
        if rf._load_json(metadata_path) != dict(metadata):
            raise ValueError(
                f"Representation metadata changed: {subject}/{name}"
            )
        return
    core.save_or_validate_json(metadata_path, dict(metadata))
    atomic_json_dump(
        done_payload(
            stage="representation_metadata",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id=task_id,
            upstream_sha256=classifier_done_sha256,
            relative_to=task_root,
            artifacts={"metadata": metadata_path},
        ),
        metadata_done_path,
    )


def train_representation_classifier(
    args: argparse.Namespace,
    config: dict[str, Any],
    representation: dict[str, Any],
    task_root: Path,
    fold_config: dict[str, Any],
    inputs: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    device: torch.device,
) -> dict[str, Any]:
    name = str(representation["variant"])
    classifier_fold_config = {
        **fold_config,
        "source": {
            "source_residual_cache_sha256": fold_config[
                "representation_cache_sha256"
            ],
            "input_support_sha256": fold_config[
                "input_support_sha256"
            ],
        },
    }
    original_input_name = rf.INPUT_NAME
    rf.INPUT_NAME = name
    try:
        metrics = rf.train_classifier_resumable(
            args,
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "shared_parameter_count": config["classifier"][
                    "parameter_count"
                ],
            },
            representation,
            task_root,
            classifier_fold_config,
            inputs,
            dataset,
            windows,
            device,
        )
    finally:
        rf.INPUT_NAME = original_input_name

    classifier_done = validate_done(
        task_root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=f"{fold_config['test_subject']}/{name}",
    )
    if classifier_done is None:
        raise RuntimeError(
            f"Classifier completion vanished: "
            f"{fold_config['test_subject']}/{name}"
        )
    metadata = representation_metadata_payload(
        config,
        fold_config,
        representation,
    )
    expected_identity = {
        "experiment_id": representation["experiment_id"],
        "variant": name,
        "input": name,
        "source_residual_sha256": fold_config[
            "representation_cache_sha256"
        ],
        "input_support_sha256": fold_config["input_support_sha256"],
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256"
        ],
    }
    for key, expected in expected_identity.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Classifier identity mismatch: "
                f"{fold_config['test_subject']}/{name}/{key}"
            )
    save_representation_metadata_completion(
        task_root,
        config,
        fold_config,
        representation,
        metadata,
    )
    return {**metrics, **metadata}


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


def _load_completed_cell(
    output_dir: Path,
    config: Mapping[str, Any],
    representation: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    name = str(representation["variant"])
    root = task_root_for(output_dir, subject, name)
    done = validate_done(
        root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{name}",
    )
    if done is None:
        return None
    expected_artifacts = {
        "best",
        "last",
        "metrics",
        "predictions",
        "validation_predictions",
        "predictions_csv",
    }
    if set(done.get("artifacts", {})) != expected_artifacts:
        raise ValueError(f"Classifier DONE artifact mismatch: {root}")
    fold_root = output_dir / f"loso_{subject}"
    fold_config = rf._load_json(fold_root / "fold_config.json")
    if fold_config.get("protocol_fingerprint") != config[
        "protocol_fingerprint"
    ]:
        raise ValueError(f"Fold protocol mismatch: {subject}")
    cache_sha = sha256_file(fold_root / "representation_cache.npz")
    support_sha = sha256_file(fold_root / "input_support.npz")
    if cache_sha != fold_config["representation_cache_sha256"]:
        raise ValueError(f"Representation cache changed: {subject}")
    if support_sha != fold_config["input_support_sha256"]:
        raise ValueError(f"Input support changed: {subject}")
    if done.get("source_residual_sha256") != cache_sha:
        raise ValueError(f"Classifier representation cache mismatch: {root}")
    if done.get("input_support_sha256") != support_sha:
        raise ValueError(f"Classifier support mismatch: {root}")
    if done.get("initial_state_sha256") != fold_config[
        "reference_initial_state_sha256"
    ]:
        raise ValueError(f"Classifier initialization mismatch: {root}")

    metadata_done = validate_done(
        root / "REPRESENTATION_METADATA_DONE.json",
        stage="representation_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{name}/representation_metadata",
        upstream_sha256=sha256_file(root / "DONE.json"),
    )
    if metadata_done is None:
        # The base classifier may have finished immediately before an
        # interruption.  Re-entering the worker repairs this tiny second stage.
        return None
    if set(metadata_done.get("artifacts", {})) != {"metadata"}:
        raise ValueError(f"Metadata DONE artifact mismatch: {root}")
    expected_metadata = representation_metadata_payload(
        config,
        fold_config,
        representation,
    )
    metadata = rf._load_json(root / "representation_metadata.json")
    if metadata != expected_metadata:
        raise ValueError(f"Representation metadata mismatch: {root}")

    metrics = rf._load_json(root / "metrics.json")
    expected_identity = {
        "experiment_id": representation["experiment_id"],
        "variant": name,
        "test_subject": subject,
        "nbm": SOURCE_NBM,
        "input": name,
        "source_residual_sha256": cache_sha,
        "input_support_sha256": support_sha,
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256"
        ],
    }
    for key, expected in expected_identity.items():
        if metrics.get(key) != expected:
            raise ValueError(f"Metrics identity mismatch: {root}/{key}")
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        expected_keys = {"window_index", "y_true", "y_prob", "y_pred"}
        if set(payload.files) != expected_keys:
            raise ValueError(f"Prediction array set mismatch: {root}")
        arrays = {
            key: np.asarray(payload[key])
            for key in expected_keys
        }
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1 or not np.isfinite(arrays["y_prob"]).all():
        raise ValueError(f"Invalid prediction arrays: {root}")
    with np.load(
        fold_root / "input_support.npz",
        allow_pickle=False,
    ) as support:
        if not np.array_equal(
            arrays["window_index"],
            support["test_anchor_window_index"],
        ):
            raise ValueError(f"Prediction support changed: {root}")
        if not np.array_equal(arrays["y_true"], support["test_y"]):
            raise ValueError(f"Prediction labels changed: {root}")
    return {**metrics, **metadata}, arrays


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
    precision = 3 if metric in {
        "false_alarm_events_per_hour",
        "median_detection_delay_sec",
    } else 4
    return f"{float(mean):.{precision}f} ± {float(std):.{precision}f}"


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


def _diagnostic_rows(output_dir: Path) -> tuple[list[dict], list[dict]]:
    input_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    for subject in EXPECTED_LOSO_SUBJECTS:
        path = output_dir / f"loso_{subject}" / "representation_diagnostics.json"
        if not path.exists():
            continue
        payload = rf._load_json(path)
        for split in ("train", "validation", "test"):
            split_payload = payload.get(split)
            if not isinstance(split_payload, Mapping):
                continue
            for representation, definition in REPRESENTATIONS.items():
                stats = split_payload.get(definition["source_key"])
                if not isinstance(stats, Mapping):
                    continue
                row = {
                    "test_subject": subject,
                    "split": split,
                    "representation": representation,
                    "n_windows": int(
                        int(stats["n_values"])
                        // (len(EXPECTED_CHANNEL_NAMES) * HORIZON_SAMPLES)
                    ),
                    **{
                        key: value
                        for key, value in stats.items()
                        if key != "clip_fraction_by_channel"
                    },
                    "clip_fraction_by_channel": (
                        json.dumps(
                            stats["clip_fraction_by_channel"],
                            separators=(",", ":"),
                        )
                        if "clip_fraction_by_channel" in stats
                        else ""
                    ),
                }
                input_rows.append(row)
        sigma = payload.get("sigma")
        if isinstance(sigma, Mapping):
            for channel_index, channel_name in enumerate(
                EXPECTED_CHANNEL_NAMES
            ):
                sigma_rows.append(
                    {
                        "test_subject": subject,
                        "channel_index": channel_index,
                        "channel_name": channel_name,
                        "sigma_min": sigma["channel_min"][channel_index],
                        "sigma_median": sigma[
                            "channel_median"
                        ][channel_index],
                        "sigma_max": sigma["channel_max"][channel_index],
                    }
                )
    return input_rows, sigma_rows


def refresh_summaries(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    variants = list(config["representations"])
    rows_by_representation: dict[str, dict[str, dict[str, Any]]] = {
        str(variant["variant"]): {} for variant in variants
    }
    fold_rows: list[dict[str, Any]] = []
    experiment_manifest: list[dict[str, Any]] = []
    aggregate_payload: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []

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
            rows_by_representation[name][subject] = metrics
            truths.append(np.asarray(arrays["y_true"], dtype=np.int8))
            probabilities.append(
                np.asarray(arrays["y_prob"], dtype=np.float64)
            )
            predictions.append(
                np.asarray(arrays["y_pred"], dtype=np.int8)
            )
            completed_subjects.append(subject)
        macro = (
            aggregate_fold_metrics(
                group,
                list(CLASSIFICATION_METRICS),
            )
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
        experiment_manifest.append(
            {
                "experiment_id": variant["experiment_id"],
                "representation": name,
                "display_name": variant["display_name"],
                "uses_mu": variant["uses_mu"],
                "uses_sigma": variant["uses_sigma"],
                "uses_residual_clip": variant["uses_residual_clip"],
                "expected_folds": len(EXPECTED_LOSO_SUBJECTS),
                "completed_folds": len(completed_subjects),
                "status": (
                    "complete"
                    if completed_subjects == list(EXPECTED_LOSO_SUBJECTS)
                    else ("partial" if completed_subjects else "pending")
                ),
                "completed_subjects": ",".join(completed_subjects),
            }
        )
        aggregate_row = {
            "experiment_id": variant["experiment_id"],
            "representation": name,
            "display_name": variant["display_name"],
            "completed_folds": len(completed_subjects),
        }
        for metric in CLASSIFICATION_METRICS:
            aggregate_row[f"{metric}_mean"] = macro[metric]["mean"]
            aggregate_row[f"{metric}_std"] = macro[metric]["std"]
        aggregate_rows.append(aggregate_row)
        publication_rows.append(
            {
                "Input representation": variant["display_name"],
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

    comparison_rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        differences: list[float] = []
        common_subjects: list[str] = []
        new_name = comparison["new"]
        reference_name = comparison["reference"]
        for subject in EXPECTED_LOSO_SUBJECTS:
            new = rows_by_representation[new_name].get(subject)
            reference = rows_by_representation[reference_name].get(subject)
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
            row["representation"],
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
    completed_cache_subjects = [
        subject
        for subject in EXPECTED_LOSO_SUBJECTS
        if validate_done(
            output_dir
            / f"loso_{subject}"
            / "REPRESENTATION_CACHE_DONE.json",
            stage="input_representation_cache",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=f"{subject}/input_representation_cache",
        )
        is not None
    ]
    support_equivalence = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "representations": list(REPRESENTATIONS),
        "shared_support_contract": (
            "All representations use one fold-level HistoryPlan, anchor set, "
            "eight-block history index matrix, and label vector."
        ),
        "completed_support_subjects": completed_cache_subjects,
        "expected_subjects": list(EXPECTED_LOSO_SUBJECTS),
        "complete": completed_cache_subjects
        == list(EXPECTED_LOSO_SUBJECTS),
    }
    atomic_json_dump(
        support_equivalence,
        output_dir / "support_equivalence.json",
    )

    fold_columns = [
        "experiment_id",
        "variant",
        "representation",
        "display_name",
        "definition",
        "uses_mu",
        "uses_sigma",
        "uses_residual_clip",
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
        "representation_cache_sha256",
        "representation_input_sha256",
        "input_support_sha256",
        "source_nbm_best_sha256",
    ]
    aggregate_columns = [
        "rank",
        "experiment_id",
        "representation",
        "display_name",
        "completed_folds",
        *[
            field
            for metric in CLASSIFICATION_METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ],
    ]
    _write_csv(
        output_dir / "experiment_manifest.csv",
        experiment_manifest,
        list(experiment_manifest[0])
        if experiment_manifest
        else [
            "experiment_id",
            "representation",
            "display_name",
            "uses_mu",
            "uses_sigma",
            "uses_residual_clip",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    _write_csv(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    _write_csv(
        output_dir / "aggregate_summary.csv",
        ranked_rows,
        aggregate_columns,
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
        output_dir / "publication_table.csv",
        publication_rows,
        [
            "Input representation",
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
    input_rows, sigma_rows = _diagnostic_rows(output_dir)
    _write_csv(
        output_dir / "input_diagnostics.csv",
        input_rows,
        [
            "test_subject",
            "split",
            "representation",
            "n_windows",
            "n_values",
            "mean",
            "std",
            "rms",
            "abs_p50",
            "abs_p90",
            "abs_p95",
            "abs_p99",
            "abs_p999",
            "max_abs",
            "nonfinite_values",
            "clip_fraction",
            "clip_fraction_by_channel",
        ],
    )
    _write_csv(
        output_dir / "sigma_diagnostics.csv",
        sigma_rows,
        [
            "test_subject",
            "channel_index",
            "channel_name",
            "sigma_min",
            "sigma_median",
            "sigma_max",
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
            "expected_representation_cache_tasks": len(
                EXPECTED_LOSO_SUBJECTS
            ),
            "completed_representation_cache_tasks": len(
                completed_cache_subjects
            ),
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
        core.save_or_validate_json(run_manifest_path, run_manifest)
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
        f"representations={list(REPRESENTATIONS)} "
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
        fold_root, features, plans, fold_config = prepare_fold(
            args,
            config,
            dataset,
            windows,
            subject,
            device,
        )
        print(
            f"[fold {subject}] train={fold_config['train_subjects']} "
            f"val={fold_config['val_subject']} "
            f"anchors={fold_config['history_anchor_counts']}",
            flush=True,
        )
        initial_hashes: set[str] = set()
        for representation in config["representations"]:
            name = str(representation["variant"])
            completed_cell = _load_completed_cell(
                args.output_dir,
                config,
                representation,
                subject,
            )
            if completed_cell is not None:
                metrics = completed_cell[0]
                print(
                    f"[fold {subject}] {representation['display_name']} "
                    "validated complete; skip",
                    flush=True,
                )
            else:
                inputs = materialize_representation_inputs(
                    features,
                    plans,
                    representation,
                )
                metrics = train_representation_classifier(
                    args,
                    config,
                    representation,
                    task_root_for(args.output_dir, subject, name),
                    fold_config,
                    inputs,
                    dataset,
                    windows,
                    device,
                )
                del inputs
            initial_hashes.add(str(metrics["initial_state_sha256"]))
            completed_this_run += 1
            print(
                f"[fold {subject}] {representation['display_name']} "
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
        if len(initial_hashes) != 1:
            raise AssertionError(
                f"Representations did not share initialization: {subject}"
            )
        del features
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
