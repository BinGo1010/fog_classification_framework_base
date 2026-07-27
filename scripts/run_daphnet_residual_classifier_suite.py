#!/usr/bin/env python
"""Compare four downstream classifiers on Persistence residual_h4s.

The upstream representation and LOSO protocol are immutable:

* canonical Daphnet three-IMU / nine-channel data at 64 Hz;
* S04 and S10 excluded before windowing;
* completed Persistence NBM residual caches are reused, never retrained;
* every input is the same eight-block, four-second residual history;
* MLP, multi-scale 1D-CNN, GRU, and lightweight Transformer are compared.

Classifier architecture is the only intended experimental axis.  Training
support, fold seed, epoch shuffle, loss, optimizer, class weighting, early
stopping, and validation-only threshold selection are shared.
"""

from __future__ import annotations

import argparse
import csv
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.histories import (
    HistoryPlan,
    make_common_history_plan,
    make_history_input,
)
from cnbr_fog.residual_classifiers import (
    CANONICAL_CLASSIFIER_NAMES,
    CLASSIFIER_DISPLAY_NAMES,
    build_residual_classifier,
    classifier_config,
    parameter_count,
)
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    capture_rng_state,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_done,
)
from run_cnbr_fog_loso import (
    deterministic_subsample,
    event_metrics,
    parse_folds,
    write_predictions_csv,
)
from run_daphnet_tcn_rf_ablation import (
    CLASSIFICATION_METRICS,
    EXPECTED_CHANNEL_NAMES,
    EXPECTED_LOSO_SUBJECTS,
    HISTORY_BLOCKS,
    HISTORY_SAMPLES,
    HISTORY_SECONDS,
    INPUT_NAME,
    SOURCE_NBM,
    _load_json,
    _load_source_cache,
    add_requested_metrics,
    array_loader,
    atomic_csv_write,
    build_source_manifest,
    classifier_epoch,
    environment_payload,
    load_dataset_and_windows,
    prediction_metrics,
    resolve_device,
    save_or_validate_json,
    save_or_validate_npz,
    set_seed,
    state_dict_sha256,
    validate_output_path,
)


SUITE_VERSION = "daphnet_persistence_h4_residual_classifier_suite.v1"
CLASSIFIER_STAGE = "residual_classifier"
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
    / "daphnet_persistence_h4_classifier4_loso_seed42"
)
IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_residual_classifier_suite.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/residual_classifiers.py",
    "cnbr_fog/__init__.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/resume.py",
)
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
)
DEBUG_SMALL_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "mlp": {"hidden_features": 8},
    "cnn1d": {
        "branch_channels": 4,
        "hidden_channels": 8,
        "head_features": 8,
    },
    "gru": {"hidden_size": 8, "num_layers": 1, "head_features": 8},
    "transformer": {
        "model_dim": 8,
        "num_heads": 2,
        "num_layers": 1,
        "feedforward_dim": 16,
        "head_features": 8,
    },
}


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daphnet Persistence residual_h4s four-classifier LOSO suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-suite-dir", type=Path, default=DEFAULT_SOURCE_SUITE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help="Run exactly one fold; used by the multi-GPU scheduler.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Initialize/validate the protocol and rebuild root summaries only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-classifier-windows",
        type=int,
        default=0,
        help="Training-only deterministic cap; zero uses every common anchor.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
        help="Testing hook for exact epoch-boundary recovery.",
    )
    parser.add_argument(
        "--debug-small-models",
        action="store_true",
        help=(
            "Use tiny architecture widths for interface smoke tests only; "
            "never use this option for reportable experiments."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    for name in (
        "classifier_epochs",
        "classifier_patience",
        "batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.max_classifier_windows < 0:
        raise ValueError("num-workers and window cap must be non-negative")
    if 0 < args.max_classifier_windows < 2:
        raise ValueError("--max-classifier-windows must be zero or at least two")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0, 1)")
    if not math.isfinite(args.classifier_lr) or args.classifier_lr <= 0:
        raise ValueError("--classifier-lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")


def build_classifier(
    name: str,
    *,
    in_channels: int,
    input_samples: int,
    dropout: float,
    architecture_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    return build_residual_classifier(
        name,
        in_channels=in_channels,
        input_samples=input_samples,
        dropout=dropout,
        **dict(architecture_kwargs or {}),
    )


def architecture_kwargs(
    args: argparse.Namespace,
    name: str,
) -> dict[str, Any]:
    return (
        dict(DEBUG_SMALL_ARCHITECTURES[name])
        if args.debug_small_models
        else {}
    )


def classifier_protocol(
    args: argparse.Namespace,
    in_channels: int,
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for name in CANONICAL_CLASSIFIER_NAMES:
        architecture = classifier_config(
            name,
            in_channels=in_channels,
            input_samples=HISTORY_SAMPLES,
            dropout=args.classifier_dropout,
            **architecture_kwargs(args, name),
        )
        set_seed(args.seed, args.deterministic)
        model = build_classifier(
            name,
            in_channels=in_channels,
            input_samples=HISTORY_SAMPLES,
            dropout=args.classifier_dropout,
            architecture_kwargs=architecture_kwargs(args, name),
        )
        if parameter_count(model) != int(architecture["parameter_count"]):
            raise AssertionError(f"Parameter count mismatch for {name}")
        definitions.append(
            {
                "classifier": name,
                "display_name": CLASSIFIER_DISPLAY_NAMES[name],
                "experiment_id": f"persistence_h4s__{name}",
                "architecture": architecture,
                "parameter_count": int(architecture["parameter_count"]),
                "protocol_initial_state_sha256": state_dict_sha256(
                    model.state_dict()
                ),
            }
        )
        del model
    return definitions


def build_protocol(
    args: argparse.Namespace,
    source_manifest: dict,
    source_config: dict,
    dataset: DaphnetDataset,
    windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict:
    classifiers = classifier_protocol(args, dataset.n_channels)
    protocol = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channel_names": list(dataset.channel_names),
        "n_channels": dataset.n_channels,
        "excluded_subjects": list(source_config["excluded_subjects"]),
        "subjects": list(dataset.subjects),
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbm": SOURCE_NBM,
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "context_samples": int(source_config["context_samples"]),
        "horizon_samples": int(source_config["horizon_samples"]),
        "stride_samples": int(source_config["stride_samples"]),
        "window_count": len(windows),
        "classifiers": classifiers,
        "classifier_names": list(CANONICAL_CLASSIFIER_NAMES),
        "classifier_dropout": args.classifier_dropout,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_classifier_windows": args.max_classifier_windows,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "debug_small_models": bool(args.debug_small_models),
        "fairness_contract": {
            "ablation_axis": "downstream_classifier_architecture",
            "shared_fields": [
                "source_persistence_residual_cache",
                "residual_h4s_window_ids_and_labels",
                "training_validation_test_support",
                "training_subsample",
                "classifier_seed",
                "epoch_shuffle_order",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "class_weight",
                "maximum_epochs",
                "early_stopping",
                "validation_threshold_rule",
            ],
            "same_classifier_seed_within_fold": True,
            "same_epoch_shuffle_within_fold": True,
            "epoch_shuffle_seed_rule": "classifier_seed + epoch",
            "threshold_source": "validation_only_balanced_accuracy",
            "different_parameter_shapes_expected": True,
        },
        "interpretation": {
            "mlp": (
                "Shallow nonlinear global readout without an explicit temporal "
                "inductive bias; it is not a strict linear probe."
            ),
            "cnn1d": "Multi-scale local temporal FoG pattern extraction.",
            "gru": "Sequential residual-state evolution.",
            "transformer": "Long-range dependencies and attention-based readout.",
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
        "num_workers": args.num_workers,
        "resume": args.resume,
    }


def checkpoint_base(
    *,
    protocol_fingerprint: str,
    task_id: str,
    source_residual_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": CLASSIFIER_STAGE,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
        "source_residual_sha256": source_residual_sha256,
    }


def validate_classifier_checkpoint(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    task_id: str,
    source_residual_sha256: str,
    classifier_name: str,
) -> None:
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": CLASSIFIER_STAGE,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
        "source_residual_sha256": source_residual_sha256,
        "classifier": classifier_name,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Incompatible checkpoint {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )


def prepare_fold_inputs(
    args: argparse.Namespace,
    config: dict,
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
) -> tuple[Path, dict[str, dict[str, np.ndarray]], dict]:
    fold_root = args.output_dir / f"loso_{subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    extracted, provenance = _load_source_cache(args, config, subject)
    source_fold_config_path = (
        args.source_suite_dir / f"loso_{subject}" / "fold_config.json"
    )
    source_fold_config = _load_json(source_fold_config_path)
    if source_fold_config.get("protocol_fingerprint") != config["source"][
        "source_protocol_fingerprint"
    ]:
        raise ValueError(f"Source fold config protocol mismatch for {subject}")
    if source_fold_config.get("test_subject") != subject:
        raise ValueError(f"Source fold test subject mismatch for {subject}")

    plans: dict[str, HistoryPlan] = {}
    for split in ("train", "validation", "test"):
        residual = np.asarray(extracted[split]["residual"])
        labels = np.asarray(extracted[split]["y"], dtype=np.int8)
        indices = np.asarray(extracted[split]["window_index"], dtype=np.int64)
        if residual.ndim != 3 or residual.shape[1:] != (
            dataset.n_channels,
            int(config["horizon_samples"]),
        ):
            raise ValueError(f"Unexpected residual shape for {subject}/{split}")
        if not np.isfinite(residual).all():
            raise ValueError(f"Non-finite residual for {subject}/{split}")
        if len(residual) != len(labels) or len(labels) != len(indices):
            raise ValueError(f"Misaligned residual cache for {subject}/{split}")
        if not np.array_equal(labels, windows.label[indices]):
            raise ValueError(f"Source labels differ for {subject}/{split}")
        if len(indices) != int(
            source_fold_config["source_window_counts"][split]
        ):
            raise ValueError(f"Source window count changed for {subject}/{split}")
        plans[split] = make_common_history_plan(
            windows,
            indices,
            int(config["horizon_samples"]),
            int(config["stride_samples"]),
            HISTORY_SAMPLES,
        )
    if min(len(plan.anchor_rows) for plan in plans.values()) == 0:
        raise RuntimeError(f"Empty h4s support in fold {subject}")

    source_history_support_path = (
        args.source_suite_dir / f"loso_{subject}" / "history_support.npz"
    )
    expected_history_keys = {
        f"{split}_{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("anchor_window_index", "history_window_index")
    }
    with np.load(source_history_support_path, allow_pickle=False) as source_support:
        if set(source_support.files) != expected_history_keys:
            raise ValueError(f"Unexpected source history support for {subject}")
        for split, plan in plans.items():
            source_indices = np.asarray(
                extracted[split]["window_index"], dtype=np.int64
            )
            if not np.array_equal(
                source_support[f"{split}_anchor_window_index"],
                plan.anchor_window_indices,
            ):
                raise ValueError(
                    f"History anchors differ from source: {subject}/{split}"
                )
            if not np.array_equal(
                source_support[f"{split}_history_window_index"],
                source_indices[plan.max_chain_rows],
            ):
                raise ValueError(
                    f"History chains differ from source: {subject}/{split}"
                )
            if len(plan.anchor_rows) != int(
                source_fold_config["history_anchor_counts"][split]
            ):
                raise ValueError(
                    f"History count differs from source: {subject}/{split}"
                )

    if args.max_classifier_windows > 0:
        rows = np.arange(len(plans["train"].anchor_rows), dtype=np.int64)
        labels = windows.label[plans["train"].anchor_window_indices]
        selected = deterministic_subsample(
            rows,
            args.max_classifier_windows,
            args.seed + 100 + EXPECTED_LOSO_SUBJECTS.index(subject),
            labels,
        )
        plans["train"] = plans["train"].take(selected)

    inputs = {
        split: make_history_input(
            extracted[split],
            plans[split],
            INPUT_NAME,
            HISTORY_SAMPLES,
            int(config["horizon_samples"]),
            int(config["stride_samples"]),
        )
        for split in ("train", "validation", "test")
    }
    for split in inputs:
        if inputs[split][INPUT_NAME].shape[1:] != (
            dataset.n_channels,
            HISTORY_SAMPLES,
        ):
            raise AssertionError(f"Invalid h4s input shape: {subject}/{split}")

    source_prediction_files = {
        "validation": (
            args.source_suite_dir
            / f"loso_{subject}"
            / SOURCE_NBM
            / INPUT_NAME
            / "validation_predictions.npz"
        ),
        "test": (
            args.source_suite_dir
            / f"loso_{subject}"
            / SOURCE_NBM
            / INPUT_NAME
            / "predictions.npz"
        ),
    }
    for split, path in source_prediction_files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(
                payload["window_index"], inputs[split]["window_index"]
            ) or not np.array_equal(payload["y_true"], inputs[split]["y"]):
                raise ValueError(
                    f"Input support differs from source h4s: {subject}/{split}"
                )

    support_arrays: dict[str, np.ndarray] = {}
    for split, plan in plans.items():
        source_indices = np.asarray(
            extracted[split]["window_index"], dtype=np.int64
        )
        support_arrays[f"{split}_anchor_window_index"] = (
            plan.anchor_window_indices
        )
        support_arrays[f"{split}_history_window_index"] = source_indices[
            plan.max_chain_rows
        ]
        support_arrays[f"{split}_y"] = np.asarray(
            inputs[split]["y"], dtype=np.int8
        )
    support_path = fold_root / "input_support.npz"
    save_or_validate_npz(support_path, **support_arrays)
    provenance = {
        **provenance,
        "source_fold_config_sha256": sha256_file(source_fold_config_path),
        "source_history_support_sha256": sha256_file(
            source_history_support_path
        ),
        "source_history_support_bytes": int(
            source_history_support_path.stat().st_size
        ),
        "source_validation_predictions_sha256": sha256_file(
            source_prediction_files["validation"]
        ),
        "source_test_predictions_sha256": sha256_file(
            source_prediction_files["test"]
        ),
        "input_support_sha256": sha256_file(support_path),
    }
    fold_index = EXPECTED_LOSO_SUBJECTS.index(subject)
    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "val_subject": source_fold_config["val_subject"],
        "train_subjects": source_fold_config["train_subjects"],
        "classifier_seed": args.seed + 10000 + fold_index,
        "source": provenance,
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "history_anchor_counts": {
            split: int(len(plans[split].anchor_rows))
            for split in ("train", "validation", "test")
        },
        "history_construction": (
            "Eight chronological horizon-spaced 32-sample residual blocks; "
            "no overlap between blocks."
        ),
    }
    save_or_validate_json(fold_root / "fold_config.json", fold_config)
    save_or_validate_json(fold_root / "source_provenance.json", provenance)
    return fold_root, inputs, fold_config


def train_classifier_resumable(
    args: argparse.Namespace,
    config: dict,
    definition: dict,
    task_root: Path,
    fold_config: dict,
    inputs: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    device: torch.device,
) -> dict:
    task_root.mkdir(parents=True, exist_ok=True)
    name = str(definition["classifier"])
    subject = str(fold_config["test_subject"])
    task_id = f"{subject}/{name}"
    source_residual_sha = str(
        fold_config["source"]["source_residual_cache_sha256"]
    )
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    metrics_path = task_root / "metrics.json"
    predictions_path = task_root / "predictions.npz"
    validation_predictions_path = task_root / "validation_predictions.npz"
    predictions_csv_path = task_root / "predictions.csv"
    done_path = task_root / "DONE.json"
    complete = validate_done(
        done_path,
        stage=CLASSIFIER_STAGE,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    if complete is not None:
        if complete.get("classifier") != name:
            raise ValueError(f"Completed cell classifier changed: {task_id}")
        if complete.get("source_residual_sha256") != source_residual_sha:
            raise ValueError(f"Completed cell uses another source: {task_id}")
        expected_support_sha = str(
            fold_config["source"]["input_support_sha256"]
        )
        if complete.get("input_support_sha256") != expected_support_sha:
            raise ValueError(
                f"Completed cell uses another input support: {task_id}"
            )
        metrics = _load_json(metrics_path)
        if metrics.get("classifier") != name:
            raise ValueError(f"Completed metrics classifier changed: {task_id}")
        if metrics.get("input_support_sha256") != expected_support_sha:
            raise ValueError(
                f"Completed metrics use another input support: {task_id}"
            )
        if metrics.get("source_residual_sha256") != source_residual_sha:
            raise ValueError(
                f"Completed metrics use another source: {task_id}"
            )
        if (
            complete.get("initial_state_sha256")
            != metrics.get("initial_state_sha256")
        ):
            raise ValueError(
                f"Completed initialization hash changed: {task_id}"
            )
        return metrics

    classifier_seed = int(fold_config["classifier_seed"])
    set_seed(classifier_seed, args.deterministic)
    x_train = np.asarray(inputs["train"][INPUT_NAME], dtype=np.float32)
    y_train = np.asarray(inputs["train"]["y"], dtype=np.int8)
    x_validation = np.asarray(
        inputs["validation"][INPUT_NAME], dtype=np.float32
    )
    y_validation = np.asarray(inputs["validation"]["y"], dtype=np.int8)
    x_test = np.asarray(inputs["test"][INPUT_NAME], dtype=np.float32)
    y_test = np.asarray(inputs["test"]["y"], dtype=np.int8)
    model = build_classifier(
        name,
        in_channels=x_train.shape[1],
        input_samples=x_train.shape[2],
        dropout=args.classifier_dropout,
        architecture_kwargs=architecture_kwargs(args, name),
    ).to(device)
    architecture = {
        **model.architecture_config(),
        "parameter_count": int(parameter_count(model)),
    }
    if architecture != definition["architecture"]:
        raise AssertionError(f"Architecture config changed for {name}")
    initial_state_sha256 = state_dict_sha256(model.state_dict())

    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if min(counts) <= 0:
        raise RuntimeError(f"Training split lacks a class in fold {subject}")
    pos_weight_value = min(math.sqrt(counts[0] / counts[1]), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    pin_memory = device.type == "cuda"
    validation_loader = array_loader(
        x_validation,
        y_validation,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = array_loader(
        x_test,
        y_test,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    start_epoch = 0
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        validate_classifier_checkpoint(
            payload,
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
            source_residual_sha256=source_residual_sha,
            classifier_name=name,
        )
        if payload.get("architecture") != architecture:
            raise ValueError(f"Checkpoint architecture changed: {task_id}")
        if payload.get("initial_state_sha256") != initial_state_sha256:
            raise ValueError(f"Checkpoint initialization changed: {task_id}")
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
            f"      [{definition['display_name']}] resume at "
            f"epoch {start_epoch + 1}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
        train_loader = array_loader(
            x_train,
            y_train,
            args.batch_size,
            shuffle=True,
            shuffle_seed=classifier_seed + epoch,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        train_loss, train_true, train_probability = classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            (
                validation_loss,
                validation_true,
                validation_probability,
            ) = classifier_epoch(
                model,
                validation_loader,
                criterion,
                device,
                args.amp,
            )
        validation_auprc = float(
            average_precision_score(
                validation_true, validation_probability
            )
        )
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": classifier_seed + epoch,
                "train_loss": train_loss,
                "train_auprc": float(
                    average_precision_score(
                        train_true, train_probability
                    )
                ),
                "validation_loss": validation_loss,
                "validation_auprc": validation_auprc,
            }
        )
        improved = validation_auprc > best_score + 1e-5
        if improved:
            best_epoch = epoch
            best_score = validation_auprc
            bad_epochs = 0
            atomic_torch_save(
                {
                    **checkpoint_base(
                        protocol_fingerprint=config["protocol_fingerprint"],
                        task_id=task_id,
                        source_residual_sha256=source_residual_sha,
                    ),
                    "classifier": name,
                    "classifier_seed": classifier_seed,
                    "architecture": architecture,
                    "initial_state_sha256": initial_state_sha256,
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_auprc": best_score,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch_save(
            {
                **checkpoint_base(
                    protocol_fingerprint=config["protocol_fingerprint"],
                    task_id=task_id,
                    source_residual_sha256=source_residual_sha,
                ),
                "classifier": name,
                "classifier_seed": classifier_seed,
                "architecture": architecture,
                "initial_state_sha256": initial_state_sha256,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
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
            f"      [{definition['display_name']}] epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} val_auprc={validation_auprc:.5f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        interrupt_marker = task_root / ".debug_interrupted_once"
        if (
            args.debug_interrupt_classifier_after_epoch > 0
            and epoch >= args.debug_interrupt_classifier_after_epoch
            and not interrupt_marker.exists()
        ):
            atomic_json_dump({"interrupted_after_epoch": epoch}, interrupt_marker)
            raise RuntimeError(
                "Intentional classifier interruption after checkpoint"
            )

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint produced: {task_id}")
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    validate_classifier_checkpoint(
        best_payload,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        source_residual_sha256=source_residual_sha,
        classifier_name=name,
    )
    if best_payload.get("architecture") != architecture:
        raise ValueError(f"Best checkpoint architecture changed: {task_id}")
    model.load_state_dict(best_payload["model_state"])
    with torch.no_grad():
        _, validation_true, validation_probability = classifier_epoch(
            model, validation_loader, criterion, device, args.amp
        )
        _, test_true, test_probability = classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, validation_metrics = choose_threshold(
        validation_true, validation_probability
    )
    test_metrics = binary_metrics(test_true, test_probability, threshold)
    test_prediction = (
        np.asarray(test_probability, dtype=np.float64) >= float(threshold)
    ).astype(np.int8)
    test_metrics.update(
        event_metrics(
            dataset,
            windows,
            inputs["test"]["window_index"],
            test_prediction,
        )
    )
    test_metrics.update(
        {
            "experiment_id": definition["experiment_id"],
            "classifier": name,
            "display_name": definition["display_name"],
            "nbm": SOURCE_NBM,
            "input": INPUT_NAME,
            "history_seconds": HISTORY_SECONDS,
            "history_samples": HISTORY_SAMPLES,
            "history_blocks": HISTORY_BLOCKS,
            "test_subject": subject,
            "val_subject": fold_config["val_subject"],
            "classifier_seed": classifier_seed,
            "architecture": architecture,
            "parameter_count": architecture["parameter_count"],
            "initial_state_sha256": initial_state_sha256,
            "best_epoch": int(best_payload["best_epoch"]),
            "best_validation_auprc": float(
                best_payload["best_validation_auprc"]
            ),
            "validation": validation_metrics,
            "train_counts": counts.astype(int).tolist(),
            "pos_weight": float(pos_weight_value),
            "elapsed_sec": elapsed_before + time.perf_counter() - started,
            "history": history,
            "source_residual_sha256": source_residual_sha,
            "input_support_sha256": fold_config["source"][
                "input_support_sha256"
            ],
        }
    )
    add_requested_metrics(test_metrics)
    atomic_json_dump(test_metrics, metrics_path)
    atomic_npz_save(
        predictions_path,
        window_index=np.asarray(
            inputs["test"]["window_index"], dtype=np.int64
        ),
        y_true=np.asarray(test_true, dtype=np.int8),
        y_prob=np.asarray(test_probability, dtype=np.float64),
        y_pred=test_prediction,
    )
    validation_prediction = (
        np.asarray(validation_probability, dtype=np.float64)
        >= float(threshold)
    ).astype(np.int8)
    atomic_npz_save(
        validation_predictions_path,
        window_index=np.asarray(
            inputs["validation"]["window_index"], dtype=np.int64
        ),
        y_true=np.asarray(validation_true, dtype=np.int8),
        y_prob=np.asarray(validation_probability, dtype=np.float64),
        y_pred=validation_prediction,
    )
    write_predictions_csv(
        predictions_csv_path,
        dataset,
        windows,
        inputs["test"]["window_index"],
        np.asarray(test_probability, dtype=np.float64),
        test_prediction,
    )
    completed = done_payload(
        stage=CLASSIFIER_STAGE,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        relative_to=task_root,
        artifacts={
            "best": best_path.resolve(),
            "last": last_path.resolve(),
            "metrics": metrics_path.resolve(),
            "predictions": predictions_path.resolve(),
            "validation_predictions": validation_predictions_path.resolve(),
            "predictions_csv": predictions_csv_path.resolve(),
        },
    )
    completed.update(
        {
            "classifier": name,
            "source_residual_sha256": source_residual_sha,
            "input_support_sha256": fold_config["source"][
                "input_support_sha256"
            ],
            "initial_state_sha256": initial_state_sha256,
        }
    )
    atomic_json_dump(completed, done_path)
    return test_metrics


def paired_delta_summary(
    rows_by_classifier: dict[str, dict[str, dict]],
) -> dict[str, Any]:
    reference = rows_by_classifier.get("mlp", {})
    result: dict[str, Any] = {}
    for name in CANONICAL_CLASSIFIER_NAMES:
        if name == "mlp":
            continue
        comparison = rows_by_classifier.get(name, {})
        common_subjects = [
            subject
            for subject in EXPECTED_LOSO_SUBJECTS
            if subject in reference and subject in comparison
        ]
        metric_payload: dict[str, dict[str, Any]] = {}
        for metric in CLASSIFICATION_METRICS:
            deltas = [
                float(comparison[subject][metric])
                - float(reference[subject][metric])
                for subject in common_subjects
                if comparison[subject].get(metric) is not None
                and reference[subject].get(metric) is not None
            ]
            values = np.asarray(deltas, dtype=np.float64)
            metric_payload[metric] = {
                "mean_delta_vs_mlp": (
                    float(values.mean()) if len(values) else None
                ),
                "std_delta_vs_mlp": (
                    float(values.std(ddof=0)) if len(values) else None
                ),
                "n_paired_folds": int(len(values)),
            }
        result[name] = {
            "reference": "mlp",
            "common_subjects": common_subjects,
            "metrics": metric_payload,
        }
    return result


def refresh_summaries(output_dir: Path, config: dict) -> None:
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    rows_by_classifier: dict[str, dict[str, dict]] = {
        name: {} for name in CANONICAL_CLASSIFIER_NAMES
    }
    expected_folds = list(config["folds_resolved"])
    for definition in config["classifiers"]:
        name = definition["classifier"]
        group_rows: list[dict] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed: list[str] = []
        for subject in expected_folds:
            task_root = output_dir / f"loso_{subject}" / name
            metrics_path = task_root / "metrics.json"
            predictions_path = task_root / "predictions.npz"
            done_path = task_root / "DONE.json"
            if not (
                metrics_path.exists()
                and predictions_path.exists()
                and done_path.exists()
            ):
                continue
            metrics = _load_json(metrics_path)
            with np.load(predictions_path, allow_pickle=False) as payload:
                truths.append(np.asarray(payload["y_true"], dtype=np.int8))
                probabilities.append(
                    np.asarray(payload["y_prob"], dtype=np.float64)
                )
                predictions.append(
                    np.asarray(payload["y_pred"], dtype=np.int8)
                )
            group_rows.append(metrics)
            fold_rows.append(metrics)
            rows_by_classifier[name][subject] = metrics
            completed.append(subject)
        if group_rows:
            subject_macro = aggregate_fold_metrics(
                group_rows, CLASSIFICATION_METRICS
            )
            aggregate[definition["experiment_id"]] = {
                "classifier": name,
                "display_name": definition["display_name"],
                "architecture": definition["architecture"],
                "parameter_count": definition["parameter_count"],
                "completed_folds": completed,
                "subject_macro": subject_macro,
                "pooled": prediction_metrics(
                    np.concatenate(truths),
                    np.concatenate(probabilities),
                    np.concatenate(predictions),
                ),
            }
            row = {
                "classifier": name,
                "display_name": definition["display_name"],
                "parameter_count": definition["parameter_count"],
                "completed_folds": len(completed),
            }
            for metric in SUMMARY_METRICS:
                row[f"{metric}_mean"] = subject_macro[metric]["mean"]
                row[f"{metric}_std"] = subject_macro[metric]["std"]
            summary_rows.append(row)
        manifest_rows.append(
            {
                "experiment_id": definition["experiment_id"],
                "classifier": name,
                "display_name": definition["display_name"],
                "family": definition["architecture"]["family"],
                "parameter_count": definition["parameter_count"],
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
    aggregate["paired_deltas_vs_mlp"] = paired_delta_summary(
        rows_by_classifier
    )
    fold_columns = [
        "experiment_id",
        "classifier",
        "display_name",
        "nbm",
        "input",
        "history_seconds",
        "history_samples",
        "history_blocks",
        "test_subject",
        "val_subject",
        "classifier_seed",
        "parameter_count",
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
        "initial_state_sha256",
        "source_residual_sha256",
        "input_support_sha256",
    ]
    atomic_csv_write(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    manifest_columns = [
        "experiment_id",
        "classifier",
        "display_name",
        "family",
        "parameter_count",
        "expected_folds",
        "completed_folds",
        "status",
        "completed_subjects",
    ]
    atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        manifest_columns,
    )
    summary_columns = [
        "classifier",
        "display_name",
        "parameter_count",
        "completed_folds",
        *[
            f"{metric}_{statistic}"
            for metric in SUMMARY_METRICS
            for statistic in ("mean", "std")
        ],
    ]
    atomic_csv_write(
        output_dir / "aggregate_summary.csv",
        summary_rows,
        summary_columns,
    )
    atomic_json_dump(aggregate, output_dir / "aggregate_metrics.json")
    completed_cells = sum(int(row["completed_folds"]) for row in manifest_rows)
    expected_cells = len(expected_folds) * len(config["classifiers"])
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_experiments": len(config["classifiers"]),
            "expected_fold_cells": expected_cells,
            "completed_fold_cells": completed_cells,
            "status": "complete" if completed_cells == expected_cells else "partial",
        },
        output_dir / "status.json",
    )


def initialize_protocol(
    args: argparse.Namespace,
    device: torch.device,
    worker_mode: bool,
) -> tuple[dict, DaphnetDataset, WindowTable]:
    source_manifest, source_config = build_source_manifest(
        args.source_suite_dir,
        verify_artifacts=not worker_mode,
    )
    dataset, windows, data_sha256 = load_dataset_and_windows(
        args.data_dir, source_config
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
        existing = _load_json(config_path)
        if existing.get("protocol_fingerprint") != config["protocol_fingerprint"]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new output directory"
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
    }
    run_manifest = {
        key: value for key, value in config.items() if key not in runtime_fields
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if not run_manifest_path.exists():
            raise RuntimeError("Missing run_manifest.json for worker")
        if _load_json(run_manifest_path) != run_manifest:
            raise ValueError(f"Saved JSON is incompatible: {run_manifest_path}")
    else:
        save_or_validate_json(run_manifest_path, run_manifest)
    return config, dataset, windows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.source_suite_dir = args.source_suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_output_path(
        args.output_dir,
        args.source_suite_dir,
        args.data_dir,
    )
    worker_mode = bool(str(args.worker_fold).strip())
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed, args.deterministic)
    configured_folds = parse_folds(args.folds, list(EXPECTED_LOSO_SUBJECTS))
    if tuple(configured_folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            "This strict experiment requires --folds all (canonical 8 folds)"
        )
    execution_folds = list(configured_folds)
    if worker_mode:
        worker_folds = parse_folds(
            str(args.worker_fold), list(EXPECTED_LOSO_SUBJECTS)
        )
        if len(worker_folds) != 1:
            raise ValueError("--worker-fold must resolve to one subject")
        execution_folds = worker_folds

    config, dataset, windows = initialize_protocol(args, device, worker_mode)
    current_environment = environment_payload(device)
    current_environment["suite_version"] = SUITE_VERSION
    if worker_mode:
        current_environment.update(
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "worker_fold": execution_folds[0],
            }
        )
        atomic_json_dump(
            current_environment,
            args.output_dir
            / "worker_environments"
            / f"loso_{execution_folds[0]}.json",
        )
    else:
        atomic_json_dump(current_environment, args.output_dir / "environment.json")
        refresh_summaries(args.output_dir, config)
    print(
        f"[INFO] suite={SUITE_VERSION} device={device} "
        f"source={args.source_suite_dir} folds={execution_folds} "
        f"classifiers={list(CANONICAL_CLASSIFIER_NAMES)} input={INPUT_NAME}",
        flush=True,
    )
    if args.finalize_only:
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

    for subject in execution_folds:
        fold_root, inputs, fold_config = prepare_fold_inputs(
            args, config, dataset, windows, subject
        )
        print(
            f"[fold {subject}] train={fold_config['train_subjects']} "
            f"val={fold_config['val_subject']} "
            f"anchors={fold_config['history_anchor_counts']}",
            flush=True,
        )
        for definition in config["classifiers"]:
            metrics = train_classifier_resumable(
                args,
                config,
                definition,
                fold_root / definition["classifier"],
                fold_config,
                inputs,
                dataset,
                windows,
                device,
            )
            print(
                f"[fold {subject}] {definition['display_name']} "
                f"params={definition['parameter_count']} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"PR-AUC={metrics['pr_auc']:.4f} "
                f"FoG-Recall={metrics['fog_recall']:.4f}",
                flush=True,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not worker_mode:
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
