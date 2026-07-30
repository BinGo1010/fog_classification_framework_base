#!/usr/bin/env python
"""Run the strict 4-NBM x 3-representation Daphnet LOSO ablation.

The suite reuses the completed NBM checkpoints from the canonical 5x4 source
suite and retrains only a fixed TCN-M diagnostic classifier.  For every NBM it
compares:

* signed forecast error ``x - mu``;
* error divided by a static clean-normal calibration scale; and
* error divided by the NBM's input-conditional scale.

Both standardized variants are clipped to [-12, 12].  The fixed scale is
estimated independently in every LOSO fold, for every NBM and every
channel-by-horizon element, using only the source fold's clean-normal
validation/calibration windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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
import run_daphnet_persistence_input_ablation as input_ablation
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
from cnbr_fog.nbm import NormalBehaviourModel, build_nbm, parameter_count
from cnbr_fog.nbm_representations import (
    REPRESENTATION_NAMES,
    build_nbm_representations,
    calibrate_fixed_sigma,
)
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    done_payload,
    sha256_file,
    validate_checkpoint,
    validate_done,
)


SUITE_VERSION = "daphnet_nbm4_representation3_h4_tcnm_loso.v1"
SOURCE_SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
NBMS = ("persistence", "gru", "tcn", "transformer")
REPRESENTATIONS: dict[str, dict[str, Any]] = {
    "error_x_minus_mu": {
        "display_name": "x - mu",
        "definition": "signed forecast error",
        "sigma_mode": "none",
        "standardized_clip": None,
    },
    "fixed_standardized_error": {
        "display_name": "(x - mu) / sigma_fixed",
        "definition": (
            "forecast error divided by clean-normal static "
            "channel-by-horizon RMS"
        ),
        "sigma_mode": "fixed",
        "standardized_clip": 12.0,
    },
    "dynamic_standardized_error": {
        "display_name": "(x - mu) / sigma_dynamic",
        "definition": (
            "forecast error divided by input-conditional NBM sigma"
        ),
        "sigma_mode": "dynamic",
        "standardized_clip": 12.0,
    },
}
if tuple(REPRESENTATIONS) != REPRESENTATION_NAMES:
    raise RuntimeError("Representation registry order changed")

EXPECTED_CHANNEL_NAMES = rf.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = rf.EXPECTED_LOSO_SUBJECTS
HISTORY_SECONDS = 4.0
HISTORY_SAMPLES = 256
HISTORY_BLOCKS = 8
CONTEXT_SAMPLES = 128
HORIZON_SAMPLES = 32
STRIDE_SAMPLES = 16
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
FIXED_SIGMA_EPSILON = 1e-6
SOURCE_REPLAY_TOLERANCE_AMP = 2e-2
SOURCE_REPLAY_TOLERANCE_NO_AMP = 2e-3
CLASSIFICATION_METRICS = tuple(rf.CLASSIFICATION_METRICS)

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_nbm_representation_ablation.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/nbm_representations.py",
    "cnbr_fog/resume.py",
)

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DEFAULT_SOURCE_SUITE_DIR = (
    REPO_ROOT / "outputs" / "daphnet_3imu_nbm_5x4_loso_seed42"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daphnet_nbm4_representation3_h4_tcnm_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daphnet 4-NBM x 3-representation, residual_h4s, TCN-M LOSO"
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
        help="Run exactly one fold; intended for an external GPU scheduler",
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--representations-only",
        action="store_true",
        help=(
            "Replay/validate all selected NBM checkpoints and materialize "
            "the three representation caches without training classifiers"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow reduced epochs/windows; smoke runs are not reportable",
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
    parser.add_argument("--stop-after-completed-tasks", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if int(args.seed) != 42:
        raise ValueError("The formal protocol fixes --seed 42")
    positive = {
        "classifier_hidden": args.classifier_hidden,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [key for key, value in positive.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {invalid}")
    if args.max_classifier_windows < 0 or args.num_workers < 0:
        raise ValueError("Window cap and num-workers must be non-negative")
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
            "classifier_dropout": (float(args.classifier_dropout), 0.15),
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
                "Formal protocol options changed; add --smoke for a "
                f"non-reportable run: {changed}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {
        "sha256": canonical_fingerprint(files),
        "files": files,
    }


def _load_scaler(path: Path) -> RobustChannelScaler:
    payload = _load_json(path)
    center = np.asarray(payload["center"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    clip = float(payload["clip"])
    if center.shape != (9,) or scale.shape != (9,):
        raise ValueError(f"Invalid scaler shape: {path}")
    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not math.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError(f"Invalid scaler values: {path}")
    return RobustChannelScaler(center=center, scale=scale, clip=clip)


def _artifact_path(done_path: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    return path if path.is_absolute() else done_path.parent / path


def build_source_manifest(
    source_suite_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_config = _load_json(source_suite_dir / "config.json")
    source_run_manifest = source_suite_dir / "run_manifest.json"
    source_status = _load_json(source_suite_dir / "status.json")
    if source_config.get("suite_version") != SOURCE_SUITE_VERSION:
        raise ValueError("Unexpected source suite version")
    if source_status.get("status") != "complete":
        raise ValueError("Source suite is not complete")
    required = {
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "seed": 42,
        "residual_clip": 12.0,
    }
    for key, expected in required.items():
        observed = source_config.get(key)
        if observed != expected:
            raise ValueError(
                f"Source protocol {key}={observed!r}, expected {expected!r}"
            )
    if tuple(source_config.get("nbms_resolved", [])) != (
        "persistence",
        "linear_ar",
        "gru",
        "tcn",
        "transformer",
    ):
        raise ValueError("Source NBM registry changed")

    protocol = str(source_config["protocol_fingerprint"])
    folds: dict[str, dict[str, Any]] = {}
    for subject in EXPECTED_LOSO_SUBJECTS:
        fold_root = source_suite_dir / f"loso_{subject}"
        shared = {
            "source_fold_config_sha256": sha256_file(
                fold_root / "fold_config.json"
            ),
            "source_scaler_sha256": sha256_file(
                fold_root / "scaler.json"
            ),
            "source_split_indices_sha256": sha256_file(
                fold_root / "split_indices.npz"
            ),
            "source_history_support_sha256": sha256_file(
                fold_root / "history_support.npz"
            ),
        }
        models: dict[str, dict[str, Any]] = {}
        for nbm in NBMS:
            nbm_root = fold_root / nbm
            nbm_done_path = nbm_root / "nbm" / "DONE.json"
            nbm_done = validate_done(
                nbm_done_path,
                stage="nbm",
                protocol_fingerprint=protocol,
                task_id=f"loso_{subject}/{nbm}/nbm",
            )
            if nbm_done is None:
                raise FileNotFoundError(nbm_done_path)
            best_entry = nbm_done["artifacts"]["best"]
            best_path = _artifact_path(nbm_done_path, best_entry)
            residual_done_path = nbm_root / "RESIDUAL_CACHE_DONE.json"
            residual_done = validate_done(
                residual_done_path,
                stage="residual_cache",
                protocol_fingerprint=protocol,
                task_id=f"loso_{subject}/{nbm}/residual_cache",
                upstream_sha256=str(best_entry["sha256"]),
            )
            if residual_done is None:
                raise FileNotFoundError(residual_done_path)
            cache_entry = residual_done["artifacts"]["cache"]
            cache_path = _artifact_path(residual_done_path, cache_entry)
            if sha256_file(best_path) != str(best_entry["sha256"]):
                raise ValueError(f"Source best checkpoint hash changed: {subject}/{nbm}")
            if sha256_file(cache_path) != str(cache_entry["sha256"]):
                raise ValueError(f"Source residual cache hash changed: {subject}/{nbm}")
            models[nbm] = {
                "source_nbm_best_sha256": str(best_entry["sha256"]),
                "source_nbm_best_bytes": int(best_entry["bytes"]),
                "source_nbm_done_sha256": sha256_file(nbm_done_path),
                "source_residual_cache_sha256": str(cache_entry["sha256"]),
                "source_residual_cache_bytes": int(cache_entry["bytes"]),
                "source_residual_done_sha256": sha256_file(
                    residual_done_path
                ),
            }
        folds[subject] = {**shared, "models": models}
    return (
        {
            "source_suite_version": SOURCE_SUITE_VERSION,
            "source_protocol_fingerprint": protocol,
            "source_run_manifest_sha256": sha256_file(source_run_manifest),
            "source_data_sha256": str(source_config["data_sha256"]),
            "source_seed": int(source_config["seed"]),
            "folds": folds,
        },
        source_config,
    )


def cell_id(nbm: str, representation: str) -> str:
    return f"{nbm}__{representation}"


def experiment_grid(
    args: argparse.Namespace,
    sampling_rate_hz: int,
) -> tuple[list[dict[str, Any]], int, str]:
    if rf.convolutional_receptive_field(TCN_M_DILATIONS) != TCN_M_RF_SAMPLES:
        raise AssertionError("TCN-M receptive field changed")
    rf.set_seed(args.seed, args.deterministic)
    reference = rf.build_model(
        in_channels=9,
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=TCN_M_DILATIONS,
    )
    count = rf.parameter_count(reference)
    state_hash = rf.state_dict_sha256(reference.state_dict())
    del reference
    cells: list[dict[str, Any]] = []
    for nbm in NBMS:
        for representation, definition in REPRESENTATIONS.items():
            identifier = cell_id(nbm, representation)
            cells.append(
                {
                    "variant": identifier,
                    "experiment_id": identifier,
                    "display_name": (
                        f"{nbm} | {definition['display_name']}"
                    ),
                    "nbm": nbm,
                    "representation": representation,
                    **definition,
                    "dilations": list(TCN_M_DILATIONS),
                    "kernel_size": 3,
                    "n_blocks": len(TCN_M_DILATIONS),
                    "convolutions_per_block": 2,
                    "receptive_field_samples": TCN_M_RF_SAMPLES,
                    "receptive_field_seconds": (
                        TCN_M_RF_SAMPLES / float(sampling_rate_hz)
                    ),
                    "parameter_count": count,
                }
            )
    return cells, count, state_hash


def build_protocol(
    args: argparse.Namespace,
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    cells, count, initial_hash = experiment_grid(
        args,
        dataset.sampling_rate_hz,
    )
    scientific = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channel_names": list(dataset.channel_names),
        "n_channels": dataset.n_channels,
        "subjects": list(dataset.subjects),
        "excluded_subjects": list(source_config["excluded_subjects"]),
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbms": list(NBMS),
        "representations": [
            {"name": name, **definition}
            for name, definition in REPRESENTATIONS.items()
        ],
        "cells": cells,
        "expected_classifier_cells": (
            len(EXPECTED_LOSO_SUBJECTS) * len(cells)
        ),
        "context_seconds": 2.0,
        "context_samples": CONTEXT_SAMPLES,
        "horizon_seconds": 0.5,
        "horizon_samples": HORIZON_SAMPLES,
        "predictor_stride_seconds": 0.25,
        "stride_samples": STRIDE_SAMPLES,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "standardized_clip": 12.0,
        "fixed_sigma": {
            "formula": (
                "sqrt(mean(clean_normal_validation_error^2, axis=windows) "
                "+ epsilon)"
            ),
            "shape": [1, 9, HORIZON_SAMPLES],
            "calibration_split": "source normal_validation_window_index",
            "epsilon": FIXED_SIGMA_EPSILON,
            "test_subject_used": False,
        },
        "classifier": {
            "name": "TCN-M",
            "hidden_channels": int(args.classifier_hidden),
            "dropout": float(args.classifier_dropout),
            "dilations": list(TCN_M_DILATIONS),
            "receptive_field_samples": TCN_M_RF_SAMPLES,
            "parameter_count": count,
            "seed42_template_initial_state_sha256": initial_hash,
            "fold_initialization_seed_rule": "42 + 10000 + fold_index",
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
            "same_source_nbm_checkpoint_within_nbm": True,
            "same_anchors_labels_and_history_blocks": True,
            "same_tcn_m_architecture": True,
            "same_classifier_seed_within_fold": True,
            "same_initial_state_within_fold": True,
            "same_epoch_shuffle_seed_rule": "classifier_seed + epoch",
            "validation_selects_epoch_and_threshold": True,
            "test_subject_never_calibrates_fixed_sigma": True,
            "standardized_variants_share_clip": True,
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


def _build_source_model(
    nbm: str,
    checkpoint: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> NormalBehaviourModel:
    model_config = dict(checkpoint["model_config"])
    if model_config.get("name") != nbm:
        raise ValueError(f"Checkpoint NBM mismatch: {model_config}")
    model = build_nbm(
        nbm,
        in_channels=9,
        horizon=HORIZON_SAMPLES,
        hidden_channels=int(source_config["nbm_hidden"]),
        dropout=float(source_config["nbm_dropout"]),
        linear_ar_order=int(
            round(
                float(source_config["linear_ar_seconds"])
                * float(source_config["sampling_rate_hz"])
            )
        ),
        gru_layers=int(source_config["gru_layers"]),
        transformer_heads=int(source_config["transformer_heads"]),
        transformer_layers=int(source_config["transformer_layers"]),
        transformer_ffn=int(source_config["transformer_ffn"]),
        max_context_samples=CONTEXT_SAMPLES,
    )
    if model.model_config() != model_config:
        raise ValueError(
            f"Reconstructed NBM config differs: {nbm}/{model.model_config()}"
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _load_source_cache(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    subject: str,
    nbm: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    root = args.source_suite_dir / f"loso_{subject}" / nbm
    done_path = root / "RESIDUAL_CACHE_DONE.json"
    expected = config["source"]["folds"][subject]["models"][nbm]
    completed = validate_done(
        done_path,
        stage="residual_cache",
        protocol_fingerprint=config["source"][
            "source_protocol_fingerprint"
        ],
        task_id=f"loso_{subject}/{nbm}/residual_cache",
        upstream_sha256=expected["source_nbm_best_sha256"],
    )
    if completed is None:
        raise FileNotFoundError(done_path)
    entry = completed["artifacts"]["cache"]
    if str(entry["sha256"]) != expected["source_residual_cache_sha256"]:
        raise ValueError(f"Source cache binding changed: {subject}/{nbm}")
    cache_path = _artifact_path(done_path, entry)
    expected_keys = {
        f"{split}_{key}"
        for split in ("train", "validation", "test")
        for key in ("residual", "y", "window_index")
    }
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError(f"Unexpected source arrays: {subject}/{nbm}")
        features = {
            split: {
                "dynamic_standardized_error": np.asarray(
                    payload[f"{split}_residual"],
                    dtype=np.float32,
                ),
                "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
                "window_index": np.asarray(
                    payload[f"{split}_window_index"],
                    dtype=np.int64,
                ),
            }
            for split in ("train", "validation", "test")
        }
    return (
        features,
        {
            "source_nbm_best_sha256": expected[
                "source_nbm_best_sha256"
            ],
            "source_residual_cache_sha256": expected[
                "source_residual_cache_sha256"
            ],
            "source_residual_done_sha256": expected[
                "source_residual_done_sha256"
            ],
        },
    )


@torch.no_grad()
def _extract_error_and_sigma(
    args: argparse.Namespace,
    model: NormalBehaviourModel,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    labels: np.ndarray,
    canonical_dynamic: np.ndarray,
    scaler: RobustChannelScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    error_chunks: list[np.ndarray] = []
    sigma_chunks: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    observed_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        target = sequence[:, :, CONTEXT_SAMPLES:].float()
        with torch.amp.autocast(
            device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            mean, sigma = model(context)
        error_chunks.append(
            (target - mean.float()).cpu().numpy().astype(np.float32)
        )
        sigma_chunks.append(sigma.float().cpu().numpy().astype(np.float32))
        observed_labels.append(y.numpy())
        observed_indices.append(index.numpy())
    error = np.ascontiguousarray(np.concatenate(error_chunks))
    dynamic_sigma = np.ascontiguousarray(np.concatenate(sigma_chunks))
    seen_y = np.concatenate(observed_labels).astype(np.int8, copy=False)
    seen_index = np.concatenate(observed_indices).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(seen_y, labels):
        raise ValueError("Replay changed labels")
    if not np.array_equal(seen_index, indices):
        raise ValueError("Replay changed window order")
    replay = build_nbm_representations(
        error,
        dynamic_sigma,
        np.ones((1, 9, HORIZON_SAMPLES), dtype=np.float32),
        standardized_clip=12.0,
    )["dynamic_standardized_error"]
    canonical = np.asarray(canonical_dynamic, dtype=np.float32)
    max_abs_diff = float(
        np.max(np.abs(replay.astype(np.float64) - canonical))
    )
    replay_tolerance = (
        SOURCE_REPLAY_TOLERANCE_AMP
        if args.amp
        else SOURCE_REPLAY_TOLERANCE_NO_AMP
    )
    if not np.allclose(
        replay,
        canonical,
        rtol=replay_tolerance,
        atol=replay_tolerance,
    ):
        raise AssertionError(
            "Replayed dynamic residual differs from source cache: "
            f"max_abs_diff={max_abs_diff}"
        )
    diagnostics = {
        "error": input_ablation._array_diagnostics(error),
        "dynamic_sigma": input_ablation._array_diagnostics(dynamic_sigma),
        "dynamic_replay_max_abs_diff": max_abs_diff,
        "dynamic_clip_fraction": float(
            (np.abs(error / dynamic_sigma) > 12.0).mean()
        ),
    }
    # The source cache is an immutable cross-device provenance reference.  All
    # three classifier representations in this suite use the replay above so
    # they share exactly the same newly generated numerator and model forward.
    return error, replay, diagnostics


def _representation_cache_keys() -> set[str]:
    keys = {"fixed_sigma", "normal_calibration_window_index"}
    for split in ("train", "validation", "test"):
        for key in ("error", "dynamic", "y", "window_index"):
            keys.add(f"{split}_{key}")
    return keys


def load_or_create_model_representation_cache(
    args: argparse.Namespace,
    config: dict[str, Any],
    subject: str,
    nbm: str,
    fold_root: Path,
    dataset: DaphnetDataset,
    windows: WindowTable,
    source_features: Mapping[str, Mapping[str, np.ndarray]],
    source_provenance: Mapping[str, Any],
    scaler: RobustChannelScaler,
    normal_validation_indices: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    np.ndarray,
    dict[str, Any],
    str,
]:
    model_root = fold_root / nbm
    model_root.mkdir(parents=True, exist_ok=True)
    cache_path = model_root / "representation_cache.npz"
    diagnostics_path = model_root / "representation_diagnostics.json"
    done_path = model_root / "REPRESENTATION_CACHE_DONE.json"
    task_id = f"{subject}/{nbm}/representation_cache"
    upstream = canonical_fingerprint(
        {
            **source_provenance,
            "source_scaler_sha256": config["source"]["folds"][subject][
                "source_scaler_sha256"
            ],
            "source_split_indices_sha256": config["source"]["folds"][
                subject
            ]["source_split_indices_sha256"],
        }
    )
    completed = validate_done(
        done_path,
        stage="nbm_representation_cache",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=upstream,
    )
    if completed is not None:
        with np.load(cache_path, allow_pickle=False) as payload:
            if set(payload.files) != _representation_cache_keys():
                raise ValueError(
                    f"Unexpected representation arrays: {subject}/{nbm}"
                )
            features = {
                split: {
                    "error_x_minus_mu": np.asarray(
                        payload[f"{split}_error"],
                        dtype=np.float32,
                    ),
                    "dynamic_standardized_error": np.asarray(
                        payload[f"{split}_dynamic"],
                        dtype=np.float32,
                    ),
                    "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
                    "window_index": np.asarray(
                        payload[f"{split}_window_index"],
                        dtype=np.int64,
                    ),
                }
                for split in ("train", "validation", "test")
            }
            fixed_sigma = np.asarray(
                payload["fixed_sigma"],
                dtype=np.float32,
            )
        for split in ("train", "validation", "test"):
            features[split]["fixed_standardized_error"] = (
                build_nbm_representations(
                    features[split]["error_x_minus_mu"],
                    np.ones_like(
                        features[split]["error_x_minus_mu"],
                        dtype=np.float32,
                    ),
                    fixed_sigma,
                    standardized_clip=12.0,
                )["fixed_standardized_error"]
            )
        return (
            features,
            fixed_sigma,
            _load_json(diagnostics_path),
            sha256_file(cache_path),
        )

    checkpoint_path = (
        args.source_suite_dir
        / f"loso_{subject}"
        / nbm
        / "nbm"
        / "best.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint(
        checkpoint,
        stage="nbm",
        protocol_fingerprint=config["source"][
            "source_protocol_fingerprint"
        ],
        task_id=f"loso_{subject}/{nbm}/nbm",
    )
    model = _build_source_model(nbm, checkpoint, config["source_config"])
    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source = source_features[split]
        error, dynamic, split_diagnostics = _extract_error_and_sigma(
            args,
            model,
            dataset,
            windows,
            np.asarray(source["window_index"], dtype=np.int64),
            np.asarray(source["y"], dtype=np.int8),
            np.asarray(
                source["dynamic_standardized_error"],
                dtype=np.float32,
            ),
            scaler,
            device,
        )
        features[split] = {
            "error_x_minus_mu": error,
            "dynamic_standardized_error": dynamic,
            "y": np.asarray(source["y"], dtype=np.int8),
            "window_index": np.asarray(
                source["window_index"],
                dtype=np.int64,
            ),
        }
        diagnostics[split] = split_diagnostics
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_indices = features["validation"]["window_index"]
    row_lookup = {
        int(window_index): row
        for row, window_index in enumerate(validation_indices)
    }
    missing = [
        int(window_index)
        for window_index in normal_validation_indices
        if int(window_index) not in row_lookup
    ]
    if missing:
        raise ValueError(
            f"Normal calibration windows missing from validation: {missing[:5]}"
        )
    calibration_rows = np.asarray(
        [row_lookup[int(index)] for index in normal_validation_indices],
        dtype=np.int64,
    )
    calibration_error = features["validation"]["error_x_minus_mu"][
        calibration_rows
    ]
    if not np.all(windows.clean_normal[normal_validation_indices]):
        raise ValueError("Fixed-sigma calibration contains non-clean windows")
    fixed_sigma = calibrate_fixed_sigma(
        calibration_error,
        epsilon=FIXED_SIGMA_EPSILON,
    )
    diagnostics["fixed_sigma"] = {
        **input_ablation._array_diagnostics(fixed_sigma),
        "shape": list(fixed_sigma.shape),
        "calibration_windows": int(len(calibration_rows)),
        "calibration_window_sha256": canonical_fingerprint(
            normal_validation_indices.astype(int).tolist()
        ),
    }
    for split in ("train", "validation", "test"):
        features[split]["fixed_standardized_error"] = (
            build_nbm_representations(
                features[split]["error_x_minus_mu"],
                np.ones_like(
                    features[split]["error_x_minus_mu"],
                    dtype=np.float32,
                ),
                fixed_sigma,
                standardized_clip=12.0,
            )["fixed_standardized_error"]
        )
        for name in REPRESENTATIONS:
            diagnostics[split][name] = input_ablation._array_diagnostics(
                features[split][name]
            )

    atomic_npz_save(
        cache_path,
        fixed_sigma=fixed_sigma,
        normal_calibration_window_index=np.asarray(
            normal_validation_indices,
            dtype=np.int64,
        ),
        **{
            f"{split}_error": features[split]["error_x_minus_mu"]
            for split in ("train", "validation", "test")
        },
        **{
            f"{split}_dynamic": features[split][
                "dynamic_standardized_error"
            ]
            for split in ("train", "validation", "test")
        },
        **{
            f"{split}_y": features[split]["y"]
            for split in ("train", "validation", "test")
        },
        **{
            f"{split}_window_index": features[split]["window_index"]
            for split in ("train", "validation", "test")
        },
    )
    atomic_json_dump(diagnostics, diagnostics_path)
    atomic_json_dump(
        done_payload(
            stage="nbm_representation_cache",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=upstream,
            relative_to=model_root,
            artifacts={
                "cache": cache_path,
                "diagnostics": diagnostics_path,
            },
        ),
        done_path,
    )
    return features, fixed_sigma, diagnostics, sha256_file(cache_path)


def prepare_fold_support(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
) -> tuple[
    Path,
    RobustChannelScaler,
    dict[str, np.ndarray],
    dict[str, HistoryPlan],
    dict[str, Any],
]:
    fold_root = args.output_dir / f"loso_{subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    source_fold_root = args.source_suite_dir / f"loso_{subject}"
    source_fold_config = _load_json(source_fold_root / "fold_config.json")
    if source_fold_config["test_subject"] != subject:
        raise ValueError(f"Source fold subject mismatch: {subject}")
    scaler = _load_scaler(source_fold_root / "scaler.json")
    with np.load(
        source_fold_root / "split_indices.npz",
        allow_pickle=False,
    ) as payload:
        split_indices = {
            split: np.asarray(
                payload[f"{split}_window_index"],
                dtype=np.int64,
            )
            for split in ("train", "validation", "test")
        }
        normal_validation = np.asarray(
            payload["normal_validation_window_index"],
            dtype=np.int64,
        )

    plans = {
        split: make_common_history_plan(
            windows,
            indices,
            HORIZON_SAMPLES,
            STRIDE_SAMPLES,
            HISTORY_SAMPLES,
        )
        for split, indices in split_indices.items()
    }
    source_support_path = source_fold_root / "history_support.npz"
    with np.load(source_support_path, allow_pickle=False) as source_support:
        for split, plan in plans.items():
            if not np.array_equal(
                plan.anchor_window_indices,
                source_support[f"{split}_anchor_window_index"],
            ):
                raise ValueError(
                    f"Anchor support changed: {subject}/{split}"
                )
            if not np.array_equal(
                split_indices[split][plan.max_chain_rows],
                source_support[f"{split}_history_window_index"],
            ):
                raise ValueError(
                    f"History support changed: {subject}/{split}"
                )
    if args.max_classifier_windows > 0:
        for split in ("train", "validation", "test"):
            if split != "train" and not args.smoke:
                continue
            rows = np.arange(len(plans[split].anchor_rows), dtype=np.int64)
            labels = windows.label[plans[split].anchor_window_indices]
            offset = {"train": 0, "validation": 1, "test": 2}[split]
            selected = rf.deterministic_subsample(
                rows,
                args.max_classifier_windows,
                args.seed
                + 100
                + EXPECTED_LOSO_SUBJECTS.index(subject)
                + offset,
                labels,
            )
            plans[split] = plans[split].take(selected)

    support_arrays: dict[str, np.ndarray] = {}
    for split, plan in plans.items():
        support_arrays[f"{split}_anchor_window_index"] = (
            plan.anchor_window_indices
        )
        support_arrays[f"{split}_history_window_index"] = (
            split_indices[split][plan.max_chain_rows]
        )
        support_arrays[f"{split}_y"] = windows.label[
            plan.anchor_window_indices
        ].astype(np.int8, copy=False)
    support_path = fold_root / "input_support.npz"
    core.save_or_validate_npz(support_path, **support_arrays)

    fold_index = EXPECTED_LOSO_SUBJECTS.index(subject)
    classifier_seed = args.seed + 10000 + fold_index
    rf.set_seed(classifier_seed, args.deterministic)
    reference_model = rf.build_model(
        in_channels=9,
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=TCN_M_DILATIONS,
    )
    initial_hash = rf.state_dict_sha256(reference_model.state_dict())
    del reference_model
    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "val_subject": source_fold_config["val_subject"],
        "train_subjects": source_fold_config["train_subjects"],
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256": initial_hash,
        "input_support_sha256": sha256_file(support_path),
        "history_anchor_counts": {
            split: int(len(plan.anchor_rows))
            for split, plan in plans.items()
        },
        "normal_validation_windows": int(len(normal_validation)),
        "source_fold_config_sha256": sha256_file(
            source_fold_root / "fold_config.json"
        ),
        "source_scaler_sha256": sha256_file(
            source_fold_root / "scaler.json"
        ),
        "source_split_indices_sha256": sha256_file(
            source_fold_root / "split_indices.npz"
        ),
        "source_history_support_sha256": sha256_file(source_support_path),
    }
    core.save_or_validate_json(fold_root / "fold_config.json", fold_config)
    return (
        fold_root,
        scaler,
        {**split_indices, "normal_validation": normal_validation},
        plans,
        fold_config,
    )


def materialize_inputs(
    features: Mapping[str, Mapping[str, np.ndarray]],
    plans: Mapping[str, HistoryPlan],
    representation: str,
) -> dict[str, dict[str, np.ndarray]]:
    inputs = {
        split: make_block_history_input(
            extracted=features[split],
            plan=plans[split],
            source_key=representation,
            name=representation,
            history_samples=HISTORY_SAMPLES,
            horizon_samples=HORIZON_SAMPLES,
            stride_samples=STRIDE_SAMPLES,
        )
        for split in ("train", "validation", "test")
    }
    for split, payload in inputs.items():
        array = np.asarray(payload[representation])
        if array.shape[1:] != (9, HISTORY_SAMPLES):
            raise ValueError(
                f"Unexpected classifier input: {split}/{array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Non-finite classifier input: {split}/{representation}"
            )
    return inputs


def task_root_for(
    output_dir: Path,
    subject: str,
    nbm: str,
    representation: str,
) -> Path:
    return (
        output_dir
        / f"loso_{subject}"
        / nbm
        / representation
    )


def train_cell(
    args: argparse.Namespace,
    config: dict[str, Any],
    cell: dict[str, Any],
    task_root: Path,
    fold_config: dict[str, Any],
    inputs: dict[str, dict[str, np.ndarray]],
    representation_binding: str,
    dataset: DaphnetDataset,
    windows: WindowTable,
    device: torch.device,
) -> dict[str, Any]:
    representation = str(cell["representation"])
    nbm = str(cell["nbm"])
    classifier_fold_config = {
        **fold_config,
        "source": {
            "source_residual_cache_sha256": representation_binding,
            "input_support_sha256": fold_config["input_support_sha256"],
        },
    }
    original_input = rf.INPUT_NAME
    original_nbm = rf.SOURCE_NBM
    rf.INPUT_NAME = representation
    rf.SOURCE_NBM = nbm
    try:
        metrics = rf.train_classifier_resumable(
            args,
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "shared_parameter_count": config["classifier"][
                    "parameter_count"
                ],
            },
            cell,
            task_root,
            classifier_fold_config,
            inputs,
            dataset,
            windows,
            device,
        )
    finally:
        rf.INPUT_NAME = original_input
        rf.SOURCE_NBM = original_nbm
    expected = {
        "experiment_id": cell["experiment_id"],
        "variant": cell["variant"],
        "nbm": nbm,
        "input": representation,
        "test_subject": fold_config["test_subject"],
        "source_residual_sha256": representation_binding,
        "input_support_sha256": fold_config["input_support_sha256"],
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256"
        ],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"Classifier identity mismatch: {cell['variant']}/{key}"
            )
    return metrics


def _load_completed_cell(
    output_dir: Path,
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    root = task_root_for(
        output_dir,
        subject,
        str(cell["nbm"]),
        str(cell["representation"]),
    )
    done = validate_done(
        root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{cell['variant']}",
    )
    if done is None:
        return None
    metrics = _load_json(root / "metrics.json")
    fold_root = output_dir / f"loso_{subject}"
    fold_config = _load_json(fold_root / "fold_config.json")
    provenance = _load_json(
        fold_root / str(cell["nbm"]) / "source_provenance.json"
    )
    expected_binding = canonical_fingerprint(
        {
            **provenance,
            "representation": cell["representation"],
        }
    )
    expected = {
        "experiment_id": cell["experiment_id"],
        "variant": cell["variant"],
        "nbm": cell["nbm"],
        "input": cell["representation"],
        "test_subject": subject,
        "source_residual_sha256": expected_binding,
        "input_support_sha256": fold_config["input_support_sha256"],
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256"
        ],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"Metrics identity mismatch: {root}/{key}")
    if done.get("source_residual_sha256") != expected_binding:
        raise ValueError(f"DONE source binding mismatch: {root}")
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in ("window_index", "y_true", "y_prob", "y_pred")
        }
    with np.load(
        output_dir / f"loso_{subject}" / "input_support.npz",
        allow_pickle=False,
    ) as support:
        if not np.array_equal(
            arrays["window_index"],
            support["test_anchor_window_index"],
        ):
            raise ValueError(f"Prediction support mismatch: {root}")
        if not np.array_equal(arrays["y_true"], support["test_y"]):
            raise ValueError(f"Prediction labels mismatch: {root}")
    return metrics, arrays


def _format_mean_sd(
    summary: Mapping[str, Any],
    metric: str,
) -> str:
    payload = summary.get(metric, {})
    mean, std = payload.get("mean"), payload.get("std")
    if mean is None or std is None:
        return ""
    precision = 3 if metric in {
        "false_alarm_events_per_hour",
        "median_detection_delay_sec",
    } else 4
    return f"{float(mean):.{precision}f} ± {float(std):.{precision}f}"


def refresh_summaries(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    cells = list(config["cells"])
    rows_by_cell: dict[str, dict[str, dict[str, Any]]] = {
        str(cell["variant"]): {} for cell in cells
    }
    fold_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []
    for cell in cells:
        group: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed_subjects: list[str] = []
        for subject in EXPECTED_LOSO_SUBJECTS:
            loaded = _load_completed_cell(
                output_dir,
                config,
                cell,
                subject,
            )
            if loaded is None:
                continue
            metrics, arrays = loaded
            group.append(metrics)
            rows_by_cell[str(cell["variant"])][subject] = metrics
            fold_rows.append(
                {
                    **metrics,
                    "representation": cell["representation"],
                    "sigma_mode": cell["sigma_mode"],
                }
            )
            truths.append(np.asarray(arrays["y_true"], dtype=np.int8))
            probabilities.append(
                np.asarray(arrays["y_prob"], dtype=np.float64)
            )
            predictions.append(np.asarray(arrays["y_pred"], dtype=np.int8))
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
        aggregate[str(cell["experiment_id"])] = {
            **cell,
            "completed_folds": completed_subjects,
            "subject_macro": macro,
            "pooled": pooled,
        }
        row = {
            "experiment_id": cell["experiment_id"],
            "nbm": cell["nbm"],
            "representation": cell["representation"],
            "display_name": cell["display_name"],
            "completed_folds": len(completed_subjects),
        }
        for metric in CLASSIFICATION_METRICS:
            row[f"{metric}_mean"] = macro[metric]["mean"]
            row[f"{metric}_std"] = macro[metric]["std"]
        aggregate_rows.append(row)
        publication_rows.append(
            {
                "NBM": cell["nbm"],
                "Representation": REPRESENTATIONS[
                    str(cell["representation"])
                ]["display_name"],
                "PR-AUC": _format_mean_sd(macro, "pr_auc"),
                "BA": _format_mean_sd(macro, "balanced_accuracy"),
                "Macro-F1": _format_mean_sd(macro, "macro_f1"),
                "AUROC": _format_mean_sd(macro, "roc_auc"),
                "FoG Recall": _format_mean_sd(macro, "fog_recall"),
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
    comparisons = (
        ("fixed_minus_error", "fixed_standardized_error", "error_x_minus_mu"),
        (
            "dynamic_minus_fixed",
            "dynamic_standardized_error",
            "fixed_standardized_error",
        ),
        (
            "dynamic_minus_error",
            "dynamic_standardized_error",
            "error_x_minus_mu",
        ),
    )
    for nbm in NBMS:
        for label, new, reference in comparisons:
            differences: list[float] = []
            subjects: list[str] = []
            new_rows = rows_by_cell[cell_id(nbm, new)]
            reference_rows = rows_by_cell[cell_id(nbm, reference)]
            for subject in EXPECTED_LOSO_SUBJECTS:
                if subject not in new_rows or subject not in reference_rows:
                    continue
                differences.append(
                    float(new_rows[subject]["pr_auc"])
                    - float(reference_rows[subject]["pr_auc"])
                )
                subjects.append(subject)
            effect = input_ablation.paired_bootstrap_mean_ci(
                np.asarray(differences, dtype=np.float64),
                int(config["bootstrap_samples"]),
                input_ablation.stable_bootstrap_seed(
                    int(config["bootstrap_seed"]),
                    f"{nbm}/{label}",
                ),
            )
            comparison_rows.append(
                {
                    "comparison_id": f"{nbm}__{label}",
                    "nbm": nbm,
                    "new": new,
                    "reference": reference,
                    "common_subjects": ",".join(subjects),
                    **effect,
                }
            )

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["pr_auc_mean"])
            if row["pr_auc_mean"] is not None
            else float("inf"),
            row["experiment_id"],
        )
    )
    ranked_rows = [
        {"rank": index, **row}
        for index, row in enumerate(aggregate_rows, start=1)
    ]
    completed_cells = len(fold_rows)
    expected_cells = int(config["expected_classifier_cells"])
    classifier_complete = completed_cells == expected_cells
    expected_caches = len(EXPECTED_LOSO_SUBJECTS) * len(NBMS)
    completed_caches = 0
    for subject in EXPECTED_LOSO_SUBJECTS:
        for nbm in NBMS:
            if (
                validate_done(
                    output_dir
                    / f"loso_{subject}"
                    / nbm
                    / "REPRESENTATION_CACHE_DONE.json",
                    stage="nbm_representation_cache",
                    protocol_fingerprint=config["protocol_fingerprint"],
                    task_id=f"{subject}/{nbm}/representation_cache",
                )
                is not None
            ):
                completed_caches += 1
    integrity_complete = (
        classifier_complete and completed_caches == expected_caches
    )
    reportable_complete = integrity_complete and bool(config["reportable"])
    best = (
        ranked_rows[0]["experiment_id"]
        if reportable_complete and ranked_rows
        else None
    )

    metric_columns = [
        field
        for metric in CLASSIFICATION_METRICS
        for field in (f"{metric}_mean", f"{metric}_std")
    ]
    _write_csv(
        output_dir / "fold_summary.csv",
        fold_rows,
        [
            "experiment_id",
            "variant",
            "nbm",
            "representation",
            "sigma_mode",
            "input",
            "test_subject",
            "val_subject",
            "threshold",
            "n",
            "n_normal",
            "n_fog",
            *CLASSIFICATION_METRICS,
            "tn",
            "fp",
            "fn",
            "tp",
            "classifier_seed",
            "best_epoch",
            "best_validation_auprc",
            "pos_weight",
            "elapsed_sec",
            "source_residual_sha256",
            "input_support_sha256",
            "initial_state_sha256",
        ],
    )
    _write_csv(
        output_dir / "aggregate_summary.csv",
        ranked_rows,
        [
            "rank",
            "experiment_id",
            "nbm",
            "representation",
            "display_name",
            "completed_folds",
            *metric_columns,
        ],
    )
    _write_csv(
        output_dir / "publication_table.csv",
        publication_rows,
        [
            "NBM",
            "Representation",
            "PR-AUC",
            "BA",
            "Macro-F1",
            "AUROC",
            "FoG Recall",
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
            "nbm",
            "new",
            "reference",
            "common_subjects",
            "mean_delta",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "wins",
            "ties",
            "losses",
            "bootstrap_samples",
        ],
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "aggregation_unit": "held_out_subject",
            "ranking_metric": "subject_macro_pr_auc_mean",
            "best_experiment": best,
            "experiments": aggregate,
            "paired_pr_auc_comparisons": comparison_rows,
        },
        output_dir / "aggregate_metrics.json",
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_representation_cache_tasks": (
                expected_caches
            ),
            "completed_representation_cache_tasks": completed_caches,
            "expected_classifier_cells": expected_cells,
            "completed_classifier_cells": completed_cells,
            "status": (
                "complete"
                if reportable_complete
                else (
                    "smoke_complete"
                    if integrity_complete
                    else "partial"
                )
            ),
            "reportable": bool(config["reportable"]),
            "best_experiment": best,
        },
        output_dir / "status.json",
    )


def initialize_protocol(
    args: argparse.Namespace,
    device: torch.device,
    worker_mode: bool,
) -> tuple[dict[str, Any], DaphnetDataset, WindowTable]:
    source_manifest, source_config = build_source_manifest(
        args.source_suite_dir
    )
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
    # Runtime-only source hyperparameters are needed to reconstruct checkpoints
    # but remain outside the immutable protocol fingerprint.
    config["source_config"] = {
        key: source_config[key]
        for key in (
            "sampling_rate_hz",
            "nbm_hidden",
            "nbm_dropout",
            "linear_ar_seconds",
            "gru_layers",
            "transformer_heads",
            "transformer_layers",
            "transformer_ffn",
        )
    }
    config_path = args.output_dir / "config.json"
    if worker_mode and not config_path.exists():
        raise RuntimeError(
            "Missing config.json; initialize once with --finalize-only"
        )
    if config_path.exists():
        existing = _load_json(config_path)
        if existing.get("protocol_fingerprint") != config[
            "protocol_fingerprint"
        ]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new "
                "output directory"
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
        "source_config",
    }
    run_manifest = {
        key: value
        for key, value in config.items()
        if key not in runtime_fields
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if _load_json(run_manifest_path) != run_manifest:
            raise ValueError("Saved run manifest is incompatible")
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
        raise ValueError("The formal suite requires --folds all")
    execution_folds = list(configured_folds)
    if worker_mode:
        execution_folds = rf.parse_folds(
            str(args.worker_fold),
            list(EXPECTED_LOSO_SUBJECTS),
        )
        if len(execution_folds) != 1:
            raise ValueError("--worker-fold must select exactly one subject")

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
        f"folds={execution_folds} nbms={list(NBMS)} "
        f"representations={list(REPRESENTATIONS)} classifier=TCN-M",
        flush=True,
    )
    if args.finalize_only:
        refresh_summaries(args.output_dir, config)
        print("[INFO] finalize-only: summaries refreshed", flush=True)
        print(
            json.dumps(
                _load_json(args.output_dir / "status.json"),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    completed_this_run = 0
    for subject in execution_folds:
        (
            fold_root,
            scaler,
            split_indices,
            plans,
            fold_config,
        ) = prepare_fold_support(
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
        initial_hashes: set[str] = set()
        for nbm in NBMS:
            source_features, source_provenance = _load_source_cache(
                args,
                config,
                subject,
                nbm,
            )
            for split in ("train", "validation", "test"):
                if not np.array_equal(
                    source_features[split]["window_index"],
                    split_indices[split],
                ):
                    raise ValueError(
                        f"NBM source split differs: {subject}/{nbm}/{split}"
                    )
            (
                features,
                fixed_sigma,
                diagnostics,
                representation_cache_sha,
            ) = load_or_create_model_representation_cache(
                args,
                config,
                subject,
                nbm,
                fold_root,
                dataset,
                windows,
                source_features,
                source_provenance,
                scaler,
                split_indices["normal_validation"],
                device,
            )
            provenance = {
                "source": source_provenance,
                "representation_cache_sha256": representation_cache_sha,
                "input_support_sha256": fold_config[
                    "input_support_sha256"
                ],
                "fixed_sigma_sha256": canonical_fingerprint(
                    fixed_sigma.astype(float).tolist()
                ),
                "fixed_sigma_calibration_windows": int(
                    len(split_indices["normal_validation"])
                ),
                "diagnostics_present": bool(diagnostics),
            }
            core.save_or_validate_json(
                fold_root / nbm / "source_provenance.json",
                provenance,
            )
            if args.representations_only:
                print(
                    f"[fold {subject}] {nbm} representation cache ready",
                    flush=True,
                )
                del features, source_features
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            for representation in REPRESENTATIONS:
                cell = next(
                    item
                    for item in config["cells"]
                    if item["nbm"] == nbm
                    and item["representation"] == representation
                )
                binding = canonical_fingerprint(
                    {
                        **provenance,
                        "representation": representation,
                    }
                )
                task_root = task_root_for(
                    args.output_dir,
                    subject,
                    nbm,
                    representation,
                )
                completed = _load_completed_cell(
                    args.output_dir,
                    config,
                    cell,
                    subject,
                )
                if completed is not None:
                    metrics = completed[0]
                    print(
                        f"[fold {subject}] {cell['display_name']} "
                        "validated complete; skip",
                        flush=True,
                    )
                else:
                    inputs = materialize_inputs(
                        features,
                        plans,
                        representation,
                    )
                    metrics = train_cell(
                        args,
                        config,
                        cell,
                        task_root,
                        fold_config,
                        inputs,
                        binding,
                        dataset,
                        windows,
                        device,
                    )
                    del inputs
                initial_hashes.add(str(metrics["initial_state_sha256"]))
                completed_this_run += 1
                print(
                    f"[fold {subject}] {cell['display_name']} "
                    f"PR-AUC={metrics['pr_auc']:.4f} "
                    f"BA={metrics['balanced_accuracy']:.4f} "
                    f"FoG-F1={metrics['fog_f1']:.4f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if (
                    args.stop_after_completed_tasks > 0
                    and completed_this_run
                    >= args.stop_after_completed_tasks
                ):
                    raise RuntimeError(
                        "Intentional stop after completed classifier tasks"
                    )
            del features, source_features
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not args.representations_only and len(initial_hashes) != 1:
            raise AssertionError(
                f"Classifier initial states differ in fold {subject}"
            )
        if not worker_mode:
            refresh_summaries(args.output_dir, config)

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
            _load_json(args.output_dir / "status.json"),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
