#!/usr/bin/env python
"""Audit the strict Persistence residual-h4s TCN-M stride ablation.

The auditor independently validates the immutable three-arm protocol, frozen
source-suite provenance, all fold-local support arrays, every completed
classifier cell, the deterministic S1/S3 equivalence contract, recomputed
window and event metrics, and all root summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_persistence_tcnm_stride_ablation as suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
    validate_done,
)


AUDIT_VERSION = "daphnet_persistence_h4_tcnm_stride3_audit.v1"
EXPECTED_SUBJECTS = tuple(suite.EXPECTED_LOSO_SUBJECTS)
EXPECTED_VARIANTS = ("s1", "s2", "s3")
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_VARIANT_STRIDES = {
    "s1": (16, 32, 0.25, 0.5),
    "s2": (16, 64, 0.25, 1.0),
    "s3": (32, 32, 0.5, 0.5),
}
CLASSIFIER_ARTIFACTS = {
    "best",
    "last",
    "metrics",
    "predictions",
    "validation_predictions",
    "predictions_csv",
}
PREDICTION_KEYS = {"window_index", "y_true", "y_prob", "y_pred"}
EQUIVALENCE_METRIC_KEYS = (
    "threshold",
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
    "tn",
    "fp",
    "fn",
    "tp",
    "best_epoch",
    "best_validation_auprc",
    "validation",
    "train_counts",
    "pos_weight",
    "history",
)
RESIDUAL_KEYS = {
    f"{split}_{key}"
    for split in EXPECTED_SPLITS
    for key in ("residual", "y", "window_index")
}
SUPPORT_KEYS = {
    f"{variant}_{split}_{suffix}"
    for variant in EXPECTED_VARIANTS
    for split in EXPECTED_SPLITS
    for suffix in (
        "predictor_window_index",
        "anchor_window_index",
        "history_window_index",
        "y",
    )
}
RUNTIME_FIELDS = {
    "data_dir",
    "source_suite_dir",
    "output_dir",
    "device",
    "num_workers",
    "resume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the strict Daphnet Persistence residual_h4s TCN-M "
            "three-stride LOSO suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        help=(
            "Fallback completed canonical NBM suite when the path recorded "
            "in config.json is unavailable."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Fallback processed Daphnet directory when the path recorded in "
            "config.json is unavailable."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing tasks while still rejecting corrupt completed tasks.",
    )
    parser.add_argument("--tolerance", type=float, default=2e-6)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_issue(
    report: dict[str, Any],
    level: str,
    message: str,
    **context: Any,
) -> None:
    report[level].append({"message": message, **context})


def require(
    report: dict[str, Any],
    condition: bool,
    message: str,
    **context: Any,
) -> bool:
    if not condition:
        add_issue(report, "failures", message, **context)
        return False
    return True


def equal_value(
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-9,
) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        try:
            return np.array_equal(np.asarray(actual), np.asarray(expected))
        except (TypeError, ValueError):
            return actual == expected
    if isinstance(actual, (int, float, np.integer, np.floating)) and isinstance(
        expected,
        (int, float, np.integer, np.floating),
    ):
        left, right = float(actual), float(expected)
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return math.isclose(
            left,
            right,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    return actual == expected


def validate_values(
    report: dict[str, Any],
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    keys: tuple[str, ...] | list[str] | set[str] | None = None,
    tolerance: float = 1e-9,
) -> None:
    selected = list(expected) if keys is None else list(keys)
    for key in selected:
        if key not in actual:
            add_issue(
                report,
                "failures",
                f"{label}: missing key",
                key=key,
            )
            continue
        if key not in expected:
            add_issue(
                report,
                "failures",
                f"{label}: no independently recomputed value",
                key=key,
            )
            continue
        if not equal_value(actual[key], expected[key], tolerance=tolerance):
            add_issue(
                report,
                "failures",
                f"{label}: value mismatch",
                key=key,
                actual=actual[key],
                expected=expected[key],
            )


def artifact_path(
    done_path: Path,
    done: Mapping[str, Any],
    name: str,
) -> Path:
    entry = done.get("artifacts", {}).get(name)
    if not isinstance(entry, Mapping):
        raise KeyError(f"Missing DONE artifact {name!r}: {done_path}")
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = done_path.parent / path
    return path.resolve()


def choose_existing_path(
    configured: Any,
    fallback: Path | None,
    required: tuple[str, ...],
) -> Path | None:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(str(configured)).expanduser())
    if fallback is not None:
        candidates.append(fallback.expanduser())
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and all((resolved / name).exists() for name in required):
            return resolved
    return None


def protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RUNTIME_FIELDS and key != "protocol_fingerprint"
    }


def validate_protocol(
    report: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    config_path = root / "config.json"
    if not require(
        report,
        config_path.exists(),
        "Missing config.json",
        path=str(config_path),
    ):
        return None, []
    try:
        config = load_json(config_path)
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load config.json",
            error=f"{type(error).__name__}: {error}",
        )
        return None, []

    protocol = str(config.get("protocol_fingerprint", ""))
    require(
        report,
        config.get("suite_version") == suite.SUITE_VERSION,
        "Unexpected suite version",
        actual=config.get("suite_version"),
        expected=suite.SUITE_VERSION,
    )
    require(
        report,
        bool(protocol)
        and canonical_fingerprint(protocol_payload(config)) == protocol,
        "Scientific protocol fingerprint mismatch",
        actual=protocol,
        recomputed=canonical_fingerprint(protocol_payload(config)),
    )

    manifest_path = root / "run_manifest.json"
    if require(
        report,
        manifest_path.exists(),
        "Missing run_manifest.json",
        path=str(manifest_path),
    ):
        try:
            manifest = load_json(manifest_path)
            expected_manifest = {
                key: value
                for key, value in config.items()
                if key not in RUNTIME_FIELDS
            }
            require(
                report,
                manifest == expected_manifest,
                "run_manifest.json differs from config scientific protocol",
            )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot validate run_manifest.json",
                error=f"{type(error).__name__}: {error}",
            )

    validate_values(
        report,
        config,
        {
            "sampling_rate_hz": 64,
            "channel_names": list(suite.EXPECTED_CHANNEL_NAMES),
            "n_channels": 9,
            "excluded_subjects": ["S04", "S10"],
            "subjects": list(EXPECTED_SUBJECTS),
            "folds_resolved": list(EXPECTED_SUBJECTS),
            "nbm": "persistence",
            "input": "residual_h4s",
            "history_seconds": 4.0,
            "history_samples": 256,
            "history_blocks": 8,
            "history_block_spacing_samples": 32,
            "context_samples": 128,
            "horizon_samples": 32,
            "source_stride_samples": 16,
            "source_stride_seconds": 0.25,
            "grid_origin_samples": 128,
            "expected_experiments": 3,
            "expected_fold_cells": 24,
            "seed": 42,
            "deterministic": True,
            "delta_pr_auc_reference": "S1",
        },
        label="config",
    )
    require(
        report,
        int(config.get("max_classifier_windows", -1)) == 0,
        "Strict reportable suite must use every classifier training anchor",
        actual=config.get("max_classifier_windows"),
    )

    fairness = config.get("fairness_contract")
    if isinstance(fairness, Mapping):
        validate_values(
            report,
            fairness,
            {
                "all_variants_share_frozen_source_nbm_and_cache": True,
                "shared_history_seconds": True,
                "shared_tcn_m_architecture": True,
                "same_classifier_seed_within_fold": True,
                "same_epoch_shuffle_seed_within_fold": True,
                "threshold_source": "validation_only_balanced_accuracy",
                "s1_s3_anchor_and_label_support_expected_identical": True,
                "s1_s3_classifier_tensors_expected_identical": True,
                "s1_s3_deterministic_results_expected_identical": True,
                "s1_extra_phase_predictions_are_not_consumed": True,
                "s2_anchor_support_is_phase_fixed_subset_of_s1": True,
            },
            label="config/fairness_contract",
        )
    else:
        add_issue(report, "failures", "Missing fairness contract")
    require(
        report,
        "s3_nbm" not in config,
        "Pure deployment-stride protocol must not define an S3 NBM refit",
    )

    variants = list(config.get("variants", []))
    require(
        report,
        len(variants) == 3
        and [str(item.get("variant")) for item in variants]
        == list(EXPECTED_VARIANTS),
        "Variant list/order is not exactly S1, S2, S3",
        actual=[item.get("variant") for item in variants],
    )
    for variant in variants:
        name = str(variant.get("variant"))
        if name not in EXPECTED_VARIANT_STRIDES:
            continue
        predictor_samples, classifier_samples, predictor_sec, classifier_sec = (
            EXPECTED_VARIANT_STRIDES[name]
        )
        validate_values(
            report,
            variant,
            {
                "variant": name,
                "experiment_id": suite.experiment_id(name),
                **suite.STRIDE_VARIANTS[name],
                "predictor_hz": 1.0 / predictor_sec,
                "classifier_hz": 1.0 / classifier_sec,
                "dilations": list(suite.TCN_M_DILATIONS),
                "receptive_field_samples": 125,
                "receptive_field_seconds": 125 / 64.0,
            },
            label=f"config/variants/{name}",
        )
        require(
            report,
            int(variant.get("predictor_stride_samples", -1))
            == predictor_samples
            and int(variant.get("classifier_stride_samples", -1))
            == classifier_samples,
            "Variant stride sample count mismatch",
            variant=name,
        )

    classifier = config.get("classifier")
    if isinstance(classifier, Mapping):
        validate_values(
            report,
            classifier,
            {
                "name": "tcn_m",
                "kernel_size": 3,
                "convolutions_per_block": 2,
                "dilations": list(suite.TCN_M_DILATIONS),
                "receptive_field_samples": 125,
                "receptive_field_seconds": 125 / 64.0,
                "global_pooling": "mean_and_max_over_full_4s_input",
            },
            label="config/classifier",
        )
        try:
            rf.set_seed(int(config["seed"]), bool(config["deterministic"]))
            model = rf.build_model(
                in_channels=9,
                hidden_channels=int(classifier["hidden_channels"]),
                dropout=float(classifier["dropout"]),
                dilations=tuple(suite.TCN_M_DILATIONS),
            )
            expected_count = rf.parameter_count(model)
            expected_hash = rf.state_dict_sha256(model.state_dict())
            require(
                report,
                int(classifier.get("parameter_count", -1)) == expected_count,
                "Classifier parameter count cannot be reproduced",
                actual=classifier.get("parameter_count"),
                expected=expected_count,
            )
            require(
                report,
                classifier.get("reference_initial_state_sha256")
                == expected_hash,
                "Classifier initial state hash cannot be reproduced",
            )
            for variant in variants:
                require(
                    report,
                    int(variant.get("parameter_count", -1))
                    == expected_count
                    and variant.get("reference_initial_state_sha256")
                    == expected_hash,
                    "Variant does not share the fixed TCN-M initialization",
                    variant=variant.get("variant"),
                )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot reconstruct protocol TCN-M",
                error=f"{type(error).__name__}: {error}",
            )
    else:
        add_issue(report, "failures", "Missing classifier protocol")

    implementation = config.get("implementation")
    if isinstance(implementation, Mapping):
        files = implementation.get("files")
        if isinstance(files, Mapping):
            current: dict[str, str] = {}
            for relative, saved_hash in files.items():
                path = REPO_ROOT / str(relative)
                if not require(
                    report,
                    path.exists(),
                    "Protocol implementation file is missing",
                    path=str(path),
                ):
                    continue
                current[str(relative)] = sha256_file(path)
                require(
                    report,
                    current[str(relative)] == saved_hash,
                    "Protocol implementation file changed",
                    path=str(path),
                    actual=current[str(relative)],
                    expected=saved_hash,
                )
            if len(current) == len(files):
                require(
                    report,
                    canonical_fingerprint(current)
                    == implementation.get("sha256"),
                    "Implementation manifest fingerprint mismatch",
                )
        else:
            add_issue(report, "failures", "Implementation file map is missing")
    else:
        add_issue(report, "failures", "Implementation manifest is missing")

    return config, variants


def resolve_external_inputs(
    report: dict[str, Any],
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[Path | None, Path | None]:
    source_root = choose_existing_path(
        config.get("source_suite_dir"),
        args.source_suite_dir,
        ("config.json", "run_manifest.json"),
    )
    data_root = choose_existing_path(
        config.get("data_dir"),
        args.data_dir,
        ("manifest.csv", "schema.json"),
    )
    require(
        report,
        source_root is not None,
        "Canonical source suite is unavailable; pass --source-suite-dir",
        configured=config.get("source_suite_dir"),
        fallback=str(args.source_suite_dir) if args.source_suite_dir else None,
    )
    require(
        report,
        data_root is not None,
        "Processed Daphnet data are unavailable; pass --data-dir",
        configured=config.get("data_dir"),
        fallback=str(args.data_dir) if args.data_dir else None,
    )
    return source_root, data_root


def validate_source_and_dataset(
    report: dict[str, Any],
    config: Mapping[str, Any],
    source_root: Path | None,
    data_root: Path | None,
) -> tuple[
    DaphnetDataset | None,
    WindowTable | None,
    dict[str, Any] | None,
]:
    if source_root is None or data_root is None:
        return None, None, None
    try:
        source_manifest, source_config = rf.build_source_manifest(
            source_root,
            verify_artifacts=True,
        )
        require(
            report,
            source_manifest == config.get("source"),
            "Canonical source manifest differs from the stride protocol",
        )
        require(
            report,
            source_config.get("suite_version") == suite.SOURCE_SUITE_VERSION,
            "Unexpected canonical source suite version",
            actual=source_config.get("suite_version"),
        )
        data_sha = dataset_fingerprint(data_root)
        require(
            report,
            data_sha == config.get("data_sha256"),
            "Processed-data fingerprint differs from protocol",
            actual=data_sha,
            expected=config.get("data_sha256"),
        )
        dataset, windows, reconstructed_sha = rf.load_dataset_and_windows(
            data_root,
            source_config,
        )
        require(
            report,
            reconstructed_sha == config.get("data_sha256"),
            "Dataset loader fingerprint differs from protocol",
        )
        require(
            report,
            len(windows) == int(config.get("window_count", -1)),
            "Reconstructed dense WindowTable count mismatch",
            actual=len(windows),
            expected=config.get("window_count"),
        )
        require(
            report,
            np.array_equal(
                np.bincount(windows.label, minlength=2),
                np.asarray(config.get("window_class_counts", [])),
            ),
            "Reconstructed dense WindowTable class counts mismatch",
        )
        return dataset, windows, source_config
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot validate source suite and processed dataset",
            error=f"{type(error).__name__}: {error}",
        )
        return None, None, None


def load_residual_cache(
    path: Path,
    *,
    expected_keys: set[str] = RESIDUAL_KEYS,
) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError(
                f"Residual cache keys differ: {path}: {sorted(payload.files)}"
            )
        return {
            split: {
                "residual": np.asarray(
                    payload[f"{split}_residual"],
                    dtype=np.float32,
                ),
                "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
                "window_index": np.asarray(
                    payload[f"{split}_window_index"],
                    dtype=np.int64,
                ),
            }
            for split in EXPECTED_SPLITS
        }


def validate_residual_arrays(
    report: dict[str, Any],
    residual: Mapping[str, Mapping[str, np.ndarray]],
    windows: WindowTable,
    expected_indices: Mapping[str, np.ndarray],
    *,
    label: str,
) -> None:
    for split in EXPECTED_SPLITS:
        values = residual[split]
        indices = np.asarray(values["window_index"], dtype=np.int64)
        expected = np.asarray(expected_indices[split], dtype=np.int64)
        require(
            report,
            np.array_equal(indices, expected),
            f"{label}: residual window support mismatch",
            split=split,
        )
        require(
            report,
            values["residual"].shape == (len(expected), 9, 32),
            f"{label}: residual shape mismatch",
            split=split,
            actual=list(values["residual"].shape),
            expected=[len(expected), 9, 32],
        )
        require(
            report,
            np.isfinite(values["residual"]).all(),
            f"{label}: residual contains non-finite values",
            split=split,
        )
        require(
            report,
            values["y"].shape == (len(expected),)
            and np.array_equal(values["y"], windows.label[expected]),
            f"{label}: residual labels differ from WindowTable",
            split=split,
        )


def validate_history_support(
    report: dict[str, Any],
    windows: WindowTable,
    predictor_indices: np.ndarray,
    anchors: np.ndarray,
    history: np.ndarray,
    labels: np.ndarray,
    *,
    subject: str,
    variant: str,
    split: str,
    predictor_stride: int,
    classifier_stride: int,
    origin: int,
) -> None:
    label = f"{subject}/{variant}/{split}"
    require(
        report,
        predictor_indices.ndim == anchors.ndim == labels.ndim == 1
        and history.shape == (len(anchors), 8)
        and len(labels) == len(anchors),
        "Input-support array shape mismatch",
        cell=label,
    )
    require(
        report,
        len(np.unique(predictor_indices)) == len(predictor_indices)
        and len(np.unique(anchors)) == len(anchors),
        "Input support contains duplicate indices",
        cell=label,
    )
    require(
        report,
        np.array_equal(history[:, -1], anchors),
        "Final history block is not the classifier anchor",
        cell=label,
    )
    require(
        report,
        np.array_equal(labels, windows.label[anchors]),
        "Support labels differ from WindowTable",
        cell=label,
    )
    require(
        report,
        np.isin(history, predictor_indices).all(),
        "History uses a residual outside the predictor grid",
        cell=label,
    )
    if len(predictor_indices):
        require(
            report,
            suite.grid_mask(
                windows,
                predictor_indices,
                stride_samples=predictor_stride,
                origin_samples=origin,
            ).all(),
            "Predictor grid contains an off-phase window",
            cell=label,
        )
    if len(anchors):
        require(
            report,
            suite.grid_mask(
                windows,
                anchors,
                stride_samples=classifier_stride,
                origin_samples=origin,
            ).all(),
            "Classifier anchors contain an off-phase window",
            cell=label,
        )
        records = windows.record_index[history]
        starts = windows.target_start[history].astype(np.int64)
        require(
            report,
            np.all(records == records[:, :1]),
            "History crosses a record boundary",
            cell=label,
        )
        require(
            report,
            np.all(np.diff(starts, axis=1) == 32),
            "History blocks are not exactly 32 samples apart",
            cell=label,
        )


def reconstruct_history_tensor(
    cache_indices: np.ndarray,
    residual: np.ndarray,
    history_indices: np.ndarray,
) -> np.ndarray:
    """Reconstruct the exact ``[B, 9, 256]`` classifier tensor."""

    indices = np.asarray(cache_indices, dtype=np.int64)
    history = np.asarray(history_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(np.diff(indices) <= 0):
        raise ValueError("Residual-cache window indices are not strictly sorted")
    if history.ndim != 2 or history.shape[1] != 8:
        raise ValueError("History support is not [B,8]")
    rows = np.searchsorted(indices, history)
    if np.any(rows >= len(indices)) or not np.array_equal(
        indices[rows],
        history,
    ):
        raise ValueError("History references a window outside the source cache")
    blocks = np.asarray(residual, dtype=np.float32)[rows]
    if blocks.shape != (len(history), 8, 9, 32):
        raise ValueError(f"Unexpected residual history shape: {blocks.shape}")
    return blocks.transpose(0, 2, 1, 3).reshape(len(history), 9, 256)


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != PREDICTION_KEYS:
            raise ValueError(
                f"Prediction key set mismatch: {path}: {sorted(payload.files)}"
            )
        arrays = {
            "window_index": np.asarray(
                payload["window_index"],
                dtype=np.int64,
            ),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Prediction array lengths differ: {path}")
    if not np.isfinite(arrays["y_prob"]).all():
        raise ValueError(f"Prediction probabilities are non-finite: {path}")
    if not np.isin(arrays["y_true"], (0, 1)).all() or not np.isin(
        arrays["y_pred"],
        (0, 1),
    ).all():
        raise ValueError(f"Prediction labels are not binary: {path}")
    return arrays


def recompute_requested_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return rf.add_requested_metrics(
        binary_metrics(y_true, y_prob, threshold)
    )


def validate_metric_map(
    report: dict[str, Any],
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    tolerance: float,
) -> None:
    validate_values(
        report,
        actual,
        expected,
        label=label,
        keys=set(expected),
        tolerance=tolerance,
    )


def recompute_deltas(
    config: Mapping[str, Any],
    by_variant: Mapping[str, Mapping[str, Mapping[str, Any]]],
    variants: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    reference = by_variant.get("s1", {})
    for variant in variants:
        name = str(variant["variant"])
        current = by_variant.get(name, {})
        common_subjects: list[str] = []
        differences: list[float] = []
        for subject in EXPECTED_SUBJECTS:
            current_cell = current.get(subject)
            reference_cell = reference.get(subject)
            if current_cell is None or reference_cell is None:
                continue
            current_value = current_cell["recomputed_test"].get("pr_auc")
            reference_value = reference_cell["recomputed_test"].get("pr_auc")
            if current_value is None or reference_value is None:
                continue
            common_subjects.append(subject)
            differences.append(
                float(current_value) - float(reference_value)
            )
        delta = suite.paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            suite.stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                f"{name}__vs__s1",
            ),
        )
        result[name] = {
            "experiment_id": variant["experiment_id"],
            "variant": name,
            "reference_variant": "s1",
            "common_subjects": ",".join(common_subjects),
            **delta,
        }
    return result


def csv_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def validate_csv_number(
    report: dict[str, Any],
    row: Mapping[str, str],
    key: str,
    expected: Any,
    *,
    label: str,
    tolerance: float,
) -> None:
    try:
        actual = csv_number(row.get(key))
    except ValueError:
        add_issue(
            report,
            "failures",
            f"{label}: invalid numeric CSV value",
            key=key,
            actual=row.get(key),
        )
        return
    require(
        report,
        equal_value(actual, expected, tolerance=tolerance),
        f"{label}: CSV value mismatch",
        key=key,
        actual=actual,
        expected=expected,
    )


def audit_root_summaries(
    report: dict[str, Any],
    root: Path,
    config: Mapping[str, Any],
    variants: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    tolerance: float,
) -> None:
    by_variant: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in EXPECTED_VARIANTS
    }
    for cell in completed:
        by_variant[cell["variant"]][cell["subject"]] = cell
    expected_deltas = recompute_deltas(config, by_variant, variants)
    variants_by_name = {
        str(variant["variant"]): variant for variant in variants
    }

    recomputed_groups: dict[str, dict[str, Any]] = {}
    ranked: list[tuple[float, str]] = []
    for name in EXPECTED_VARIANTS:
        cells = [
            by_variant[name][subject]
            for subject in EXPECTED_SUBJECTS
            if subject in by_variant[name]
        ]
        rows = [cell["recomputed_test"] for cell in cells]
        macro = (
            aggregate_fold_metrics(rows, list(suite.CLASSIFICATION_METRICS))
            if rows
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in suite.CLASSIFICATION_METRICS
            }
        )
        pooled = (
            rf.prediction_metrics(
                np.concatenate(
                    [cell["test"]["y_true"] for cell in cells]
                ),
                np.concatenate(
                    [cell["test"]["y_prob"] for cell in cells]
                ),
                np.concatenate(
                    [cell["test"]["y_pred"] for cell in cells]
                ),
            )
            if cells
            else None
        )
        completed_subjects = [cell["subject"] for cell in cells]
        recomputed_groups[name] = {
            "completed_subjects": completed_subjects,
            "subject_macro": macro,
            "pooled": pooled,
        }
        pr_auc = macro["pr_auc"]["mean"]
        if pr_auc is not None:
            ranked.append((-float(pr_auc), variants_by_name[name]["experiment_id"]))
    ranked.sort()
    best_experiment = ranked[0][1] if ranked else None

    equivalent_subjects: list[str] = []
    for subject in EXPECTED_SUBJECTS:
        s1 = by_variant["s1"].get(subject)
        s3 = by_variant["s3"].get(subject)
        if s1 is None or s3 is None:
            continue
        predictions_equal = all(
            np.array_equal(s1["test"][key], s3["test"][key])
            for key in PREDICTION_KEYS
        )
        metrics_equal = all(
            s1["metrics"].get(key) == s3["metrics"].get(key)
            for key in EQUIVALENCE_METRIC_KEYS
        )
        if predictions_equal and metrics_equal:
            equivalent_subjects.append(subject)
    equivalence_path = root / "stride_equivalence.json"
    if require(
        report,
        equivalence_path.exists(),
        "Missing stride_equivalence.json",
        path=str(equivalence_path),
    ):
        try:
            equivalence = load_json(equivalence_path)
            expected_equivalence = {
                "suite_version": suite.SUITE_VERSION,
                "protocol_fingerprint": config["protocol_fingerprint"],
                "scientific_expectation": (
                    "S1 and S3 are exactly equivalent because both classifiers "
                    "consume the same frozen Persistence residual blocks; S1's "
                    "extra phase-16 predictor calls are unused."
                ),
                "exact_comparison_fields": [
                    "test_window_index",
                    "test_y_true",
                    "test_y_prob",
                    "test_y_pred",
                    *EQUIVALENCE_METRIC_KEYS,
                ],
                "completed_equivalent_subjects": equivalent_subjects,
                "expected_subjects": list(EXPECTED_SUBJECTS),
                "complete_exact_equivalence": (
                    equivalent_subjects == list(EXPECTED_SUBJECTS)
                ),
                "s3_predictor_call_ratio_vs_s1": 0.5,
                "s3_classifier_call_ratio_vs_s1": 1.0,
            }
            require(
                report,
                equivalence == expected_equivalence,
                "stride_equivalence.json differs from independently recomputed equivalence",
            )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot validate stride_equivalence.json",
                error=f"{type(error).__name__}: {error}",
            )

    aggregate_path = root / "aggregate_metrics.json"
    if require(
        report,
        aggregate_path.exists(),
        "Missing aggregate_metrics.json",
        path=str(aggregate_path),
    ):
        try:
            aggregate = load_json(aggregate_path)
            validate_values(
                report,
                aggregate,
                {
                    "suite_version": suite.SUITE_VERSION,
                    "protocol_fingerprint": config["protocol_fingerprint"],
                    "aggregation_unit": "held_out_subject",
                    "ranking_metric": "subject_macro_pr_auc_mean",
                    "delta_pr_auc": {
                        "reference": "S1",
                        "method": "paired bootstrap over held-out subjects",
                        "samples": config["bootstrap_samples"],
                        "seed": config["bootstrap_seed"],
                        "confidence_level": 0.95,
                    },
                    "best_experiment": best_experiment,
                },
                label="aggregate_metrics.json",
            )
            saved_experiments = aggregate.get("experiments")
            expected_experiment_ids = {
                str(variant["experiment_id"]) for variant in variants
            }
            require(
                report,
                isinstance(saved_experiments, Mapping)
                and set(saved_experiments) == expected_experiment_ids,
                "Aggregate experiment key set mismatch",
                actual=(
                    sorted(saved_experiments)
                    if isinstance(saved_experiments, Mapping)
                    else None
                ),
                expected=sorted(expected_experiment_ids),
            )
            if isinstance(saved_experiments, Mapping):
                for name in EXPECTED_VARIANTS:
                    variant = variants_by_name[name]
                    saved = saved_experiments.get(variant["experiment_id"])
                    if not isinstance(saved, Mapping):
                        continue
                    group = recomputed_groups[name]
                    validate_values(
                        report,
                        saved,
                        {
                            **variant,
                            "completed_folds": group[
                                "completed_subjects"
                            ],
                            "delta_pr_auc_vs_s1": expected_deltas[name],
                        },
                        label=f"aggregate/{name}",
                    )
                    saved_macro = saved.get("subject_macro")
                    if isinstance(saved_macro, Mapping):
                        for metric, expected in group[
                            "subject_macro"
                        ].items():
                            actual = saved_macro.get(metric)
                            if isinstance(actual, Mapping):
                                validate_values(
                                    report,
                                    actual,
                                    expected,
                                    label=(
                                        f"aggregate/{name}/"
                                        f"subject_macro/{metric}"
                                    ),
                                    tolerance=tolerance,
                                )
                            else:
                                add_issue(
                                    report,
                                    "failures",
                                    "Aggregate subject metric is missing",
                                    variant=name,
                                    metric=metric,
                                )
                    else:
                        add_issue(
                            report,
                            "failures",
                            "Aggregate subject_macro is missing",
                            variant=name,
                        )
                    saved_pooled = saved.get("pooled")
                    expected_pooled = group["pooled"]
                    if expected_pooled is None:
                        require(
                            report,
                            saved_pooled is None,
                            "Pending aggregate has non-null pooled metrics",
                            variant=name,
                        )
                    elif isinstance(saved_pooled, Mapping):
                        validate_values(
                            report,
                            saved_pooled,
                            expected_pooled,
                            label=f"aggregate/{name}/pooled",
                            tolerance=tolerance,
                        )
                    else:
                        add_issue(
                            report,
                            "failures",
                            "Aggregate pooled metrics are missing",
                            variant=name,
                        )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot validate aggregate_metrics.json",
                error=f"{type(error).__name__}: {error}",
            )

    status_path = root / "status.json"
    if require(
        report,
        status_path.exists(),
        "Missing status.json",
        path=str(status_path),
    ):
        try:
            status = load_json(status_path)
            completed_count = len(completed)
            validate_values(
                report,
                status,
                {
                    "suite_version": suite.SUITE_VERSION,
                    "protocol_fingerprint": config["protocol_fingerprint"],
                    "expected_experiments": 3,
                    "expected_fold_cells": 24,
                    "completed_fold_cells": completed_count,
                    "status": (
                        "complete" if completed_count == 24 else "partial"
                    ),
                    "best_experiment": best_experiment,
                },
                label="status.json",
            )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot validate status.json",
                error=f"{type(error).__name__}: {error}",
            )

    expected_csv_rows = {
        "fold_summary.csv": len(completed),
        "experiment_manifest.csv": 3,
        "aggregate_summary.csv": 3,
        "paired_pr_auc_deltas.csv": 3,
        "publication_table.csv": 3,
        "efficiency_summary.csv": 3,
    }
    csv_payloads: dict[str, list[dict[str, str]]] = {}
    for filename, expected_rows in expected_csv_rows.items():
        path = root / filename
        if not require(
            report,
            path.exists(),
            "Root summary CSV is missing",
            path=str(path),
        ):
            continue
        try:
            rows = load_csv(path)
            csv_payloads[filename] = rows
            require(
                report,
                len(rows) == expected_rows,
                "Root summary CSV row count mismatch",
                path=str(path),
                actual=len(rows),
                expected=expected_rows,
            )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot load root summary CSV",
                path=str(path),
                error=f"{type(error).__name__}: {error}",
            )

    fold_rows = csv_payloads.get("fold_summary.csv", [])
    fold_by_id = {
        (row.get("test_subject", ""), row.get("variant", "")): row
        for row in fold_rows
    }
    require(
        report,
        len(fold_by_id) == len(fold_rows),
        "fold_summary.csv contains duplicate cell identities",
    )
    for cell in completed:
        key = (cell["subject"], cell["variant"])
        row = fold_by_id.get(key)
        if row is None:
            add_issue(
                report,
                "failures",
                "fold_summary.csv is missing an audited cell",
                subject=key[0],
                variant=key[1],
            )
            continue
        expected = {
            **cell["metrics"],
            "purpose": variants_by_name[cell["variant"]]["purpose"],
        }
        for text_key in (
            "experiment_id",
            "variant",
            "display_name",
            "purpose",
            "nbm",
            "input",
            "test_subject",
            "val_subject",
            "source_residual_sha256",
            "input_support_sha256",
        ):
            require(
                report,
                str(row.get(text_key, "")) == str(expected.get(text_key, "")),
                "fold_summary.csv text field mismatch",
                subject=key[0],
                variant=key[1],
                field=text_key,
            )
        for number_key in (
            "predictor_stride_seconds",
            "classifier_stride_seconds",
            "predictor_hz",
            "classifier_hz",
            "history_seconds",
            "history_samples",
            "history_blocks",
            "classifier_seed",
            "threshold",
            "n",
            "n_normal",
            "n_fog",
            *suite.CLASSIFICATION_METRICS,
            "tn",
            "fp",
            "fn",
            "tp",
            "best_epoch",
            "best_validation_auprc",
            "predictor_test_windows",
            "classifier_test_windows",
            "target_time_coverage_fraction",
            "minimum_two_positive_confirmation_seconds",
        ):
            validate_csv_number(
                report,
                row,
                number_key,
                expected.get(number_key),
                label=f"fold_summary/{key[0]}/{key[1]}",
                tolerance=tolerance,
            )

    manifest_rows = csv_payloads.get("experiment_manifest.csv", [])
    manifest_by_variant = {
        row.get("variant", ""): row for row in manifest_rows
    }
    require(
        report,
        set(manifest_by_variant) == set(EXPECTED_VARIANTS),
        "experiment_manifest.csv variant set mismatch",
    )
    for name, row in manifest_by_variant.items():
        variant = variants_by_name[name]
        subjects = recomputed_groups[name]["completed_subjects"]
        expected_status = (
            "complete"
            if len(subjects) == 8
            else ("partial" if subjects else "pending")
        )
        expected_text = {
            "experiment_id": variant["experiment_id"],
            "variant": name,
            "display_name": variant["display_name"],
            "purpose": variant["purpose"],
            "status": expected_status,
            "completed_subjects": ",".join(subjects),
        }
        for key, expected in expected_text.items():
            require(
                report,
                row.get(key) == str(expected),
                "experiment_manifest.csv field mismatch",
                variant=name,
                field=key,
                actual=row.get(key),
                expected=expected,
            )
        for key, expected in (
            ("predictor_stride_seconds", variant["predictor_stride_seconds"]),
            ("classifier_stride_seconds", variant["classifier_stride_seconds"]),
            ("expected_folds", 8),
            ("completed_folds", len(subjects)),
        ):
            validate_csv_number(
                report,
                row,
                key,
                expected,
                label=f"experiment_manifest/{name}",
                tolerance=tolerance,
            )

    delta_rows = csv_payloads.get("paired_pr_auc_deltas.csv", [])
    delta_by_variant = {
        row.get("variant", ""): row for row in delta_rows
    }
    require(
        report,
        set(delta_by_variant) == set(EXPECTED_VARIANTS),
        "paired_pr_auc_deltas.csv variant set mismatch",
    )
    for name, expected in expected_deltas.items():
        row = delta_by_variant.get(name)
        if row is None:
            continue
        for key in (
            "experiment_id",
            "variant",
            "reference_variant",
            "common_subjects",
        ):
            require(
                report,
                row.get(key) == str(expected[key]),
                "paired_pr_auc_deltas.csv identity mismatch",
                variant=name,
                field=key,
            )
        for key in (
            "mean_delta",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "bootstrap_samples",
        ):
            validate_csv_number(
                report,
                row,
                key,
                expected[key],
                label=f"paired_pr_auc_deltas/{name}",
                tolerance=1e-12,
            )

    aggregate_rows = csv_payloads.get("aggregate_summary.csv", [])
    aggregate_by_variant = {
        row.get("variant", ""): row for row in aggregate_rows
    }
    require(
        report,
        set(aggregate_by_variant) == set(EXPECTED_VARIANTS),
        "aggregate_summary.csv variant set mismatch",
    )
    for name, row in aggregate_by_variant.items():
        variant = variants_by_name[name]
        group = recomputed_groups[name]
        delta = expected_deltas[name]
        for key, expected in (
            ("completed_folds", len(group["completed_subjects"])),
            ("delta_pr_auc_mean", delta["mean_delta"]),
            ("delta_pr_auc_ci_low", delta["ci_low"]),
            ("delta_pr_auc_ci_high", delta["ci_high"]),
            (
                "delta_pr_auc_n_paired_subjects",
                delta["n_paired_subjects"],
            ),
        ):
            validate_csv_number(
                report,
                row,
                key,
                expected,
                label=f"aggregate_summary/{name}",
                tolerance=tolerance,
            )
        for metric, payload in group["subject_macro"].items():
            for statistic in ("mean", "std"):
                validate_csv_number(
                    report,
                    row,
                    f"{metric}_{statistic}",
                    payload[statistic],
                    label=f"aggregate_summary/{name}",
                    tolerance=tolerance,
                )
        require(
            report,
            row.get("experiment_id") == variant["experiment_id"],
            "aggregate_summary.csv experiment ID mismatch",
            variant=name,
        )

    efficiency_rows = csv_payloads.get("efficiency_summary.csv", [])
    efficiency_by_variant = {
        row.get("variant", ""): row for row in efficiency_rows
    }
    require(
        report,
        set(efficiency_by_variant) == set(EXPECTED_VARIANTS),
        "efficiency_summary.csv variant set mismatch",
    )
    for name, row in efficiency_by_variant.items():
        variant = variants_by_name[name]
        cells = list(by_variant[name].values())
        expected_efficiency = {
            "predictor_stride_seconds": variant["predictor_stride_seconds"],
            "classifier_stride_seconds": variant[
                "classifier_stride_seconds"
            ],
            "predictor_calls_per_hour": 3600.0
            / float(variant["predictor_stride_seconds"]),
            "classifier_calls_per_hour": 3600.0
            / float(variant["classifier_stride_seconds"]),
            "predictor_call_ratio_vs_0p25": 0.25
            / float(variant["predictor_stride_seconds"]),
            "classifier_call_ratio_vs_0p25": 0.25
            / float(variant["classifier_stride_seconds"]),
            "target_time_coverage_fraction": min(
                1.0,
                32.0 / float(variant["classifier_stride_samples"]),
            ),
            "minimum_two_positive_confirmation_seconds": (
                32 + int(variant["classifier_stride_samples"])
            )
            / 64.0,
            "test_predictor_windows_total": sum(
                int(cell["metrics"]["predictor_test_windows"])
                for cell in cells
            ),
            "test_classifier_windows_total": sum(
                int(cell["metrics"]["classifier_test_windows"])
                for cell in cells
            ),
            "completed_folds": len(cells),
        }
        for key, expected in expected_efficiency.items():
            validate_csv_number(
                report,
                row,
                key,
                expected,
                label=f"efficiency_summary/{name}",
                tolerance=tolerance,
            )

    publication_rows = csv_payloads.get("publication_table.csv", [])
    publication_by_name = {
        row.get("Experiment", "").lower(): row
        for row in publication_rows
    }
    require(
        report,
        set(publication_by_name) == set(EXPECTED_VARIANTS),
        "publication_table.csv experiment set mismatch",
    )
    for name, row in publication_by_name.items():
        variant = variants_by_name[name]
        require(
            report,
            row.get("Predictor stride")
            == f"{float(variant['predictor_stride_seconds']):g} s"
            and row.get("Classifier stride")
            == f"{float(variant['classifier_stride_seconds']):g} s",
            "publication_table.csv stride labels mismatch",
            variant=name,
        )
        validate_csv_number(
            report,
            row,
            "Completed folds",
            len(recomputed_groups[name]["completed_subjects"]),
            label=f"publication_table/{name}",
            tolerance=tolerance,
        )


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    counts = report["counts"]
    lines = [
        f"Audit version: {report['audit_version']}",
        f"Result directory: {report['result_dir']}",
        f"Status: {report['status']}",
        f"Allow partial: {report['allow_partial']}",
        f"Valid classifier cells: {counts['valid_classifier_cells']}/24",
        f"Missing classifier cells: {counts['missing_classifier_cells']}",
        f"Failures: {len(report['failures'])}",
        f"Warnings: {len(report['warnings'])}",
        "",
    ]
    if report["failures"]:
        lines.append("FAILURES")
        for index, failure in enumerate(report["failures"], start=1):
            lines.append(
                f"{index}. {json.dumps(failure, ensure_ascii=False)}"
            )
        lines.append("")
    if report["warnings"]:
        lines.append("WARNINGS")
        for index, warning in enumerate(report["warnings"], start=1):
            lines.append(
                f"{index}. {json.dumps(warning, ensure_ascii=False)}"
            )
        lines.append("")
    if report["missing_folds"]:
        lines.append("MISSING FOLDS")
        for item in report["missing_folds"]:
            lines.append(json.dumps(item, ensure_ascii=False))
        lines.append("")
    if report["missing_classifier_cells"]:
        lines.append("MISSING CLASSIFIER CELLS")
        for item in report["missing_classifier_cells"]:
            lines.append(json.dumps(item, ensure_ascii=False))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise SystemExit("--tolerance must be finite and positive")
    root = args.result_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Result directory does not exist: {root}")
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "allow_partial": bool(args.allow_partial),
        "status": "running",
        "counts": {
            "expected_subjects": 8,
            "expected_variants": 3,
            "expected_classifier_cells": 24,
            "valid_classifier_cells": 0,
            "missing_classifier_cells": 0,
            "missing_folds": 0,
        },
        "failures": [],
        "warnings": [],
        "missing_folds": [],
        "missing_classifier_cells": [],
    }

    config, variants = validate_protocol(report, root)
    completed: list[dict[str, Any]] = []
    valid_folds: dict[str, dict[str, Any] | None] = {
        subject: None for subject in EXPECTED_SUBJECTS
    }
    if config is not None and len(variants) == 3:
        report["protocol_fingerprint"] = config["protocol_fingerprint"]
        source_root, data_root = resolve_external_inputs(
            report,
            args,
            config,
        )
        report["source_suite_dir"] = (
            None if source_root is None else str(source_root)
        )
        report["data_dir"] = None if data_root is None else str(data_root)
        dataset, windows, source_config = validate_source_and_dataset(
            report,
            config,
            source_root,
            data_root,
        )
        if (
            source_root is not None
            and dataset is not None
            and windows is not None
            and source_config is not None
        ):
            for subject in EXPECTED_SUBJECTS:
                try:
                    valid_folds[subject] = audit_fold_support(
                        report,
                        root,
                        config,
                        subject,
                        source_root,
                        source_config,
                        dataset,
                        windows,
                    )
                except Exception as error:  # noqa: BLE001
                    add_issue(
                        report,
                        "failures",
                        "Unexpected fold-audit exception",
                        subject=subject,
                        error=f"{type(error).__name__}: {error}",
                    )
            for subject in EXPECTED_SUBJECTS:
                fold = valid_folds[subject]
                for variant in variants:
                    audited = audit_classifier_cell(
                        report,
                        root,
                        config,
                        variant,
                        subject,
                        fold,
                        dataset,
                        windows,
                        float(args.tolerance),
                    )
                    if audited is not None:
                        completed.append(audited)

            for subject, fold in valid_folds.items():
                if fold is None:
                    continue
                cells = [
                    cell
                    for cell in completed
                    if cell["subject"] == subject
                ]
                if len(cells) == 3:
                    require(
                        report,
                        len(
                            {
                                cell["initial_state_sha256"]
                                for cell in cells
                            }
                        )
                        == 1,
                        "Stride classifiers do not share one fold initialization",
                        subject=subject,
                    )
                    require(
                        report,
                        len(
                            {
                                int(cell["metrics"]["classifier_seed"])
                                for cell in cells
                            }
                        )
                        == 1,
                        "Stride classifiers do not share one fold seed",
                        subject=subject,
                    )
                    cells_by_variant = {
                        str(cell["variant"]): cell for cell in cells
                    }
                    audit_s1_s3_equivalence(
                        report,
                        subject,
                        cells_by_variant["s1"],
                        cells_by_variant["s3"],
                    )
            audit_root_summaries(
                report,
                root,
                config,
                variants,
                completed,
                float(args.tolerance),
            )

    report["counts"]["valid_classifier_cells"] = len(completed)
    report["counts"]["missing_classifier_cells"] = len(
        report["missing_classifier_cells"]
    )
    report["counts"]["missing_folds"] = len(report["missing_folds"])
    missing_total = (
        len(report["missing_folds"])
        + len(report["missing_classifier_cells"])
    )
    if missing_total and not args.allow_partial:
        add_issue(
            report,
            "failures",
            "Suite is incomplete; use --allow-partial only for an interim audit",
            missing_folds=len(report["missing_folds"]),
            missing_classifier_cells=len(
                report["missing_classifier_cells"]
            ),
        )
    full_complete = (
        len(completed) == 24
        and missing_total == 0
        and not report["failures"]
    )
    report["full_complete"] = bool(full_complete)
    complete_path = root / "SUITE_COMPLETE.json"
    if not full_complete and complete_path.exists():
        try:
            complete_path.unlink()
            add_issue(
                report,
                "failures",
                "Removed stale SUITE_COMPLETE.json because the suite is not fully valid",
                path=str(complete_path),
            )
        except OSError as error:
            add_issue(
                report,
                "failures",
                "Stale SUITE_COMPLETE.json exists and could not be removed",
                path=str(complete_path),
                error=f"{type(error).__name__}: {error}",
            )
    if report["failures"]:
        report["status"] = "fail"
        exit_code = 1
    elif missing_total:
        report["status"] = "partial_pass"
        exit_code = 0
    else:
        report["status"] = "pass"
        exit_code = 0

    report_path = root / "AUDIT_REPORT.json"
    text_path = root / "AUDIT_REPORT.txt"
    atomic_json_dump(report, report_path)
    write_text_report(text_path, report)
    if report["status"] == "pass" and report["full_complete"]:
        atomic_json_dump(
            {
                "format_version": 1,
                "suite_version": suite.SUITE_VERSION,
                "audit_version": AUDIT_VERSION,
                "status": "complete",
                "protocol_fingerprint": report[
                    "protocol_fingerprint"
                ],
                "expected_classifier_cells": 24,
                "checked_classifier_cells": len(completed),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_report_sha256": sha256_file(report_path),
            },
            complete_path,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                **report["counts"],
                "failures": len(report["failures"]),
                "warnings": len(report["warnings"]),
                "json_report": str(report_path),
                "text_report": str(text_path),
                "complete_marker": (
                    str(complete_path)
                    if report["status"] == "pass"
                    and report["full_complete"]
                    else None
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    raise SystemExit(exit_code)


def audit_s1_s3_equivalence(
    report: dict[str, Any],
    subject: str,
    s1: Mapping[str, Any],
    s3: Mapping[str, Any],
) -> None:
    """Enforce the deterministic equivalence implied by identical tensors."""

    for split in ("validation", "test"):
        s1_arrays = s1[split]
        s3_arrays = s3[split]
        for key in PREDICTION_KEYS:
            require(
                report,
                np.array_equal(s1_arrays[key], s3_arrays[key]),
                "S1/S3 deterministic prediction arrays differ",
                subject=subject,
                split=split,
                array=key,
            )

    allowed_arm_metadata = {
        "experiment_id",
        "variant",
        "task_id",
        "display_name",
        "predictor_stride_seconds",
        "predictor_stride_samples",
        "predictor_hz",
        "predictor_train_windows",
        "predictor_validation_windows",
        "predictor_test_windows",
        "elapsed_sec",
    }
    s1_metrics = {
        key: value
        for key, value in s1["metrics"].items()
        if key not in allowed_arm_metadata
    }
    s3_metrics = {
        key: value
        for key, value in s3["metrics"].items()
        if key not in allowed_arm_metadata
    }
    require(
        report,
        s1_metrics == s3_metrics,
        "S1/S3 deterministic metric payloads differ",
        subject=subject,
        differing_keys=sorted(
            key
            for key in set(s1_metrics) | set(s3_metrics)
            if s1_metrics.get(key) != s3_metrics.get(key)
        ),
    )
    require(
        report,
        s1["best_model_state_sha256"] == s3["best_model_state_sha256"],
        "S1/S3 best model states differ despite identical training tensors",
        subject=subject,
    )
    require(
        report,
        s1["last_model_state_sha256"] == s3["last_model_state_sha256"],
        "S1/S3 last model states differ despite identical training tensors",
        subject=subject,
    )


def audit_classifier_cell(
    report: dict[str, Any],
    root: Path,
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
    subject: str,
    fold: Mapping[str, Any] | None,
    dataset: DaphnetDataset,
    windows: WindowTable,
    tolerance: float,
) -> dict[str, Any] | None:
    name = str(variant["variant"])
    task_id = f"{subject}/{name}"
    task_root = suite.task_root_for(root, subject, name)
    done_path = task_root / "DONE.json"
    metadata_done_path = task_root / "STRIDE_METADATA_DONE.json"
    if not done_path.exists() or not metadata_done_path.exists():
        report["missing_classifier_cells"].append(
            {
                "subject": subject,
                "variant": name,
                "task_id": task_id,
                "missing": [
                    str(path)
                    for path in (done_path, metadata_done_path)
                    if not path.exists()
                ],
            }
        )
        return None
    if fold is None:
        add_issue(
            report,
            "failures",
            "Classifier DONE exists without valid fold support",
            task_id=task_id,
        )
        return None
    try:
        done = validate_done(
            done_path,
            stage="rf_classifier",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id=task_id,
        )
        if done is None:
            raise FileNotFoundError(done_path)
        require(
            report,
            set(done.get("artifacts", {})) == CLASSIFIER_ARTIFACTS,
            "Classifier DONE artifact map mismatch",
            task_id=task_id,
        )
        expected_residual = fold["fold_config"][
            "variant_source_residual_sha256"
        ][name]
        expected_support_sha = fold["support_sha256"]
        expected_initial = fold["reference_initial_state_sha256"]
        validate_values(
            report,
            done,
            {
                "source_residual_sha256": expected_residual,
                "input_support_sha256": expected_support_sha,
                "initial_state_sha256": expected_initial,
            },
            label=f"{task_id}/DONE",
        )
        metadata_done = validate_done(
            metadata_done_path,
            stage="stride_metadata",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id=f"{task_id}/stride_metadata",
            upstream_sha256=sha256_file(done_path),
        )
        if metadata_done is None:
            raise FileNotFoundError(metadata_done_path)
        require(
            report,
            set(metadata_done.get("artifacts", {})) == {"metadata"},
            "Stride-metadata DONE artifact map mismatch",
            task_id=task_id,
        )
        metadata_path = artifact_path(
            metadata_done_path,
            metadata_done,
            "metadata",
        )
        metadata = load_json(metadata_path)
        expected_metadata = suite.stride_metadata_payload(
            config,
            fold["fold_config"],
            variant,
        )
        require(
            report,
            metadata == expected_metadata,
            "Stride metadata cannot be independently reconstructed",
            task_id=task_id,
        )

        metrics_path = artifact_path(done_path, done, "metrics")
        test_path = artifact_path(done_path, done, "predictions")
        validation_path = artifact_path(
            done_path,
            done,
            "validation_predictions",
        )
        best_path = artifact_path(done_path, done, "best")
        last_path = artifact_path(done_path, done, "last")
        base_metrics = load_json(metrics_path)
        stride_only_keys = set(metadata) - {
            "experiment_id",
            "variant",
            "source_residual_sha256",
            "input_support_sha256",
            "initial_state_sha256",
        }
        require(
            report,
            not (set(base_metrics) & stride_only_keys),
            "Base metrics.json was rewritten with stride-only metadata",
            task_id=task_id,
            unexpected=sorted(set(base_metrics) & stride_only_keys),
        )
        metrics = {**base_metrics, **metadata}
        test = load_predictions(test_path)
        validation = load_predictions(validation_path)
        expected_test_support = fold["support"][name]["test"]
        expected_validation_support = fold["support"][name]["validation"]
        require(
            report,
            np.array_equal(
                test["window_index"],
                expected_test_support["anchor_window_index"],
            )
            and np.array_equal(
                test["y_true"],
                expected_test_support["y"],
            ),
            "Test predictions differ from audited support",
            task_id=task_id,
        )
        require(
            report,
            np.array_equal(
                validation["window_index"],
                expected_validation_support["anchor_window_index"],
            )
            and np.array_equal(
                validation["y_true"],
                expected_validation_support["y"],
            ),
            "Validation predictions differ from audited support",
            task_id=task_id,
        )

        classifier_seed = int(
            fold["fold_config"]["classifier_seed"]
        )
        predictor_counts = fold["fold_config"][
            "predictor_window_counts"
        ][name]
        classifier_counts = fold["fold_config"][
            "classifier_actual_anchor_counts"
        ][name]
        expected_identity = {
            "experiment_id": variant["experiment_id"],
            "variant": name,
            "display_name": variant["display_name"],
            "nbm": "persistence",
            "input": "residual_h4s",
            "history_seconds": 4.0,
            "history_samples": 256,
            "history_blocks": 8,
            "test_subject": subject,
            "val_subject": fold["fold_config"]["val_subject"],
            "classifier_seed": classifier_seed,
            "initial_state_sha256": expected_initial,
            "source_residual_sha256": expected_residual,
            "input_support_sha256": expected_support_sha,
            "predictor_stride_seconds": variant[
                "predictor_stride_seconds"
            ],
            "predictor_stride_samples": variant[
                "predictor_stride_samples"
            ],
            "classifier_stride_seconds": variant[
                "classifier_stride_seconds"
            ],
            "classifier_stride_samples": variant[
                "classifier_stride_samples"
            ],
            "predictor_hz": variant["predictor_hz"],
            "classifier_hz": variant["classifier_hz"],
            "predictor_train_windows": predictor_counts["train"],
            "predictor_validation_windows": predictor_counts[
                "validation"
            ],
            "predictor_test_windows": predictor_counts["test"],
            "classifier_train_windows": classifier_counts["train"],
            "classifier_validation_windows": classifier_counts[
                "validation"
            ],
            "classifier_test_windows": classifier_counts["test"],
            "target_time_coverage_fraction": min(
                1.0,
                32.0 / float(variant["classifier_stride_samples"]),
            ),
            "minimum_two_positive_confirmation_seconds": (
                32 + int(variant["classifier_stride_samples"])
            )
            / 64.0,
        }
        validate_values(
            report,
            base_metrics,
            expected_identity,
            keys={
                "experiment_id",
                "variant",
                "display_name",
                "nbm",
                "input",
                "history_seconds",
                "history_samples",
                "history_blocks",
                "test_subject",
                "val_subject",
                "classifier_seed",
                "initial_state_sha256",
                "source_residual_sha256",
                "input_support_sha256",
            },
            label=f"{task_id}/base metrics identity",
        )
        validate_values(
            report,
            metrics,
            expected_identity,
            label=f"{task_id}/metrics identity",
        )
        require(
            report,
            int(metrics.get("n", -1)) == len(test["y_true"])
            == int(classifier_counts["test"]),
            "Classifier test-window count mismatch",
            task_id=task_id,
        )

        threshold = float(metrics.get("threshold", float("nan")))
        require(
            report,
            math.isfinite(threshold) and 0.0 <= threshold <= 1.0,
            "Classifier threshold is invalid",
            task_id=task_id,
            threshold=threshold,
        )
        require(
            report,
            np.array_equal(
                test["y_pred"],
                (test["y_prob"] >= threshold).astype(np.int8),
            )
            and np.array_equal(
                validation["y_pred"],
                (validation["y_prob"] >= threshold).astype(np.int8),
            ),
            "Saved predictions do not equal probability >= threshold",
            task_id=task_id,
        )
        selected_threshold, selected_validation = choose_threshold(
            validation["y_true"],
            validation["y_prob"],
        )
        require(
            report,
            equal_value(
                threshold,
                selected_threshold,
                tolerance=1e-12,
            ),
            "Threshold was not selected exclusively from validation data",
            task_id=task_id,
            actual=threshold,
            expected=selected_threshold,
        )
        saved_validation = metrics.get("validation")
        if isinstance(saved_validation, Mapping):
            validate_metric_map(
                report,
                saved_validation,
                selected_validation,
                label=f"{task_id}/validation metrics",
                tolerance=tolerance,
            )
        else:
            add_issue(
                report,
                "failures",
                "Classifier validation metrics are missing",
                task_id=task_id,
            )
        recomputed_test = recompute_requested_metrics(
            test["y_true"],
            test["y_prob"],
            threshold,
        )
        validate_metric_map(
            report,
            metrics,
            recomputed_test,
            label=f"{task_id}/test metrics",
            tolerance=tolerance,
        )
        event_policy = config.get("event_policy", {})
        recomputed_events = suite.stride_aware_event_metrics(
            dataset,
            windows,
            test["window_index"],
            test["y_pred"],
            classifier_stride_samples=int(
                variant["classifier_stride_samples"]
            ),
            minimum_positive_windows=int(
                event_policy.get("minimum_positive_windows", 2)
            ),
            merge_gap_seconds=float(
                event_policy.get("merge_gap_seconds", 0.5)
            ),
        )
        validate_metric_map(
            report,
            metrics,
            recomputed_events,
            label=f"{task_id}/event metrics",
            tolerance=tolerance,
        )

        classifier_config = metrics.get("classifier_config")
        if not isinstance(classifier_config, Mapping):
            add_issue(
                report,
                "failures",
                "Classifier model config is missing",
                task_id=task_id,
            )
            classifier_config = {}
        validate_values(
            report,
            classifier_config,
            {
                "in_channels": 9,
                "hidden_channels": int(
                    config["classifier"]["hidden_channels"]
                ),
                "dropout": float(config["classifier"]["dropout"]),
                "kernel_size": 3,
                "dilations": list(suite.TCN_M_DILATIONS),
                "n_blocks": 6,
                "convolutions_per_block": 2,
                "receptive_field_samples": 125,
                "receptive_field_seconds": 125 / 64.0,
                "parameter_count": int(
                    config["classifier"]["parameter_count"]
                ),
                "initial_state_sha256": expected_initial,
                "global_pooling": "mean_and_max_over_full_input",
            },
            label=f"{task_id}/classifier_config",
        )

        best = torch.load(best_path, map_location="cpu", weights_only=False)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        checkpoint_base = {
            "format_version": 1,
            "stage": "rf_classifier",
            "protocol_fingerprint": config["protocol_fingerprint"],
            "task_id": task_id,
            "source_residual_sha256": expected_residual,
            "variant": name,
            "classifier_seed": classifier_seed,
            "classifier_config": dict(classifier_config),
        }
        validate_values(
            report,
            best,
            checkpoint_base,
            label=f"{task_id}/best checkpoint",
        )
        validate_values(
            report,
            last,
            checkpoint_base,
            label=f"{task_id}/last checkpoint",
        )
        require(
            report,
            int(best.get("best_epoch", -1))
            == int(metrics.get("best_epoch", -2)),
            "Best checkpoint epoch mismatch",
            task_id=task_id,
        )
        require(
            report,
            equal_value(
                best.get("best_validation_auprc"),
                metrics.get("best_validation_auprc"),
                tolerance=tolerance,
            ),
            "Best checkpoint validation AUPRC mismatch",
            task_id=task_id,
        )
        model = rf.build_model(
            in_channels=9,
            hidden_channels=int(config["classifier"]["hidden_channels"]),
            dropout=float(config["classifier"]["dropout"]),
            dilations=tuple(suite.TCN_M_DILATIONS),
        )
        model.load_state_dict(best["model_state"], strict=True)
        require(
            report,
            rf.parameter_count(model)
            == int(config["classifier"]["parameter_count"]),
            "Loaded classifier parameter count mismatch",
            task_id=task_id,
        )
        with torch.no_grad():
            logits = model.eval()(torch.zeros(2, 9, 256))
        require(
            report,
            tuple(logits.shape) == (2,)
            and torch.isfinite(logits).all().item(),
            "Classifier checkpoint fails [B,9,256] forward validation",
            task_id=task_id,
        )

        train_labels = fold["support"][name]["train"]["y"]
        train_counts = np.bincount(train_labels, minlength=2).astype(int)
        require(
            report,
            metrics.get("train_counts") == train_counts.tolist(),
            "Training class counts mismatch",
            task_id=task_id,
        )
        expected_weight = min(
            math.sqrt(train_counts[0] / train_counts[1]),
            6.0,
        )
        require(
            report,
            equal_value(
                metrics.get("pos_weight"),
                expected_weight,
                tolerance=tolerance,
            ),
            "Classifier positive-class weight mismatch",
            task_id=task_id,
        )
        history = metrics.get("history")
        if isinstance(history, list) and history:
            require(
                report,
                [int(row.get("epoch", -1)) for row in history]
                == list(range(1, len(history) + 1)),
                "Classifier training epochs are not contiguous",
                task_id=task_id,
            )
            require(
                report,
                len(history) <= int(config["classifier_epochs"]),
                "Classifier exceeds configured maximum epochs",
                task_id=task_id,
            )
            require(
                report,
                all(
                    int(row.get("shuffle_seed", -1))
                    == classifier_seed + int(row["epoch"])
                    for row in history
                ),
                "Classifier epoch shuffle seed rule mismatch",
                task_id=task_id,
            )
            require(
                report,
                last.get("history") == history
                and int(last.get("epoch", -1)) == len(history),
                "Last checkpoint history mismatch",
                task_id=task_id,
            )
        else:
            add_issue(
                report,
                "failures",
                "Classifier training history is missing",
                task_id=task_id,
            )

        return {
            "subject": subject,
            "variant": name,
            "experiment_id": variant["experiment_id"],
            "metrics": metrics,
            "recomputed_test": {
                **recomputed_test,
                **recomputed_events,
            },
            "test": test,
            "validation": validation,
            "initial_state_sha256": expected_initial,
            "best_model_state_sha256": rf.state_dict_sha256(
                best["model_state"]
            ),
            "last_model_state_sha256": rf.state_dict_sha256(
                last["model_state"]
            ),
        }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot validate classifier cell",
            task_id=task_id,
            error=f"{type(error).__name__}: {error}",
        )
        return None


def audit_fold_support(
    report: dict[str, Any],
    root: Path,
    config: Mapping[str, Any],
    subject: str,
    source_root: Path,
    source_config: Mapping[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
) -> dict[str, Any] | None:
    fold_root = root / f"loso_{subject}"
    fold_config_path = fold_root / "fold_config.json"
    provenance_path = fold_root / "source_provenance.json"
    support_path = fold_root / "input_support.npz"
    if not all(
        path.exists()
        for path in (fold_config_path, provenance_path, support_path)
    ):
        report["missing_folds"].append(
            {
                "subject": subject,
                "missing": [
                    str(path)
                    for path in (
                        fold_config_path,
                        provenance_path,
                        support_path,
                    )
                    if not path.exists()
                ],
            }
        )
        return None
    try:
        fold_config = load_json(fold_config_path)
        provenance_file = load_json(provenance_path)
        source_fold_config = load_json(
            source_root / f"loso_{subject}" / "fold_config.json"
        )
        source_cache_path = (
            source_root
            / f"loso_{subject}"
            / "persistence"
            / "residual_cache.npz"
        )
        source_cache = load_residual_cache(source_cache_path)
        source_expected = {
            split: np.asarray(
                source_cache[split]["window_index"],
                dtype=np.int64,
            )
            for split in EXPECTED_SPLITS
        }
        validate_residual_arrays(
            report,
            source_cache,
            windows,
            source_expected,
            label=f"{subject}/canonical source cache",
        )
        origin = int(config["grid_origin_samples"])
        source_sha = sha256_file(source_cache_path)
        support_sha = sha256_file(support_path)
        expected_seed = int(config["seed"]) + 10000 + EXPECTED_SUBJECTS.index(
            subject
        )
        rf.set_seed(expected_seed, bool(config["deterministic"]))
        reference_model = rf.build_model(
            in_channels=dataset.n_channels,
            hidden_channels=int(config["classifier"]["hidden_channels"]),
            dropout=float(config["classifier"]["dropout"]),
            dilations=tuple(suite.TCN_M_DILATIONS),
        )
        expected_fold_initial_sha = rf.state_dict_sha256(
            reference_model.state_dict()
        )
        del reference_model
        validate_values(
            report,
            fold_config,
            {
                "suite_version": suite.SUITE_VERSION,
                "protocol_fingerprint": config["protocol_fingerprint"],
                "test_subject": subject,
                "val_subject": source_fold_config["val_subject"],
                "train_subjects": source_fold_config["train_subjects"],
                "classifier_seed": expected_seed,
                "reference_initial_state_sha256": expected_fold_initial_sha,
                "input": "residual_h4s",
                "history_seconds": 4.0,
                "history_samples": 256,
                "history_blocks": 8,
                "s1_s3_anchor_label_support_identical": True,
                "s1_s3_classifier_tensors_identical": True,
                "s2_is_fixed_phase_subset_of_s1": True,
            },
            label=f"{subject}/fold_config",
        )
        require(
            report,
            "s3_predictor" not in fold_config
            and "s1_s3_residual_values_identical" not in fold_config,
            "Pure deployment fold must not contain legacy S3-refit fields",
            subject=subject,
        )
        canonical_provenance = fold_config.get("source")
        require(
            report,
            isinstance(canonical_provenance, Mapping),
            "Canonical source provenance is missing",
            subject=subject,
        )
        if isinstance(canonical_provenance, Mapping):
            source_fold_root = source_root / f"loso_{subject}"
            source_history_support = source_fold_root / "history_support.npz"
            source_validation_predictions = (
                source_fold_root
                / "persistence"
                / "residual_h4s"
                / "validation_predictions.npz"
            )
            source_test_predictions = (
                source_fold_root
                / "persistence"
                / "residual_h4s"
                / "predictions.npz"
            )
            source_manifest_fold = config["source"]["folds"][subject]
            expected_canonical_provenance = {
                "source_protocol_fingerprint": config["source"][
                    "source_protocol_fingerprint"
                ],
                "source_nbm": "persistence",
                "source_nbm_best_sha256": source_manifest_fold[
                    "source_nbm_best_sha256"
                ],
                "source_residual_cache_sha256": source_sha,
                "source_residual_cache_bytes": int(
                    source_cache_path.stat().st_size
                ),
                "source_residual_done_sha256": source_manifest_fold[
                    "source_residual_done_sha256"
                ],
                "source_fold_config_sha256": sha256_file(
                    source_fold_root / "fold_config.json"
                ),
                "source_history_support_sha256": sha256_file(
                    source_history_support
                ),
                "source_history_support_bytes": int(
                    source_history_support.stat().st_size
                ),
                "source_validation_predictions_sha256": sha256_file(
                    source_validation_predictions
                ),
                "source_test_predictions_sha256": sha256_file(
                    source_test_predictions
                ),
                "input_support_sha256": support_sha,
            }
            require(
                report,
                canonical_provenance == expected_canonical_provenance,
                "Fold canonical source provenance mismatch",
                subject=subject,
            )
        expected_variant_sources = {
            "s1": {
                "kind": "frozen_canonical_cache_dense_grid",
                "predictor_stride_samples": 16,
                "residual_cache_sha256": source_sha,
            },
            "s2": {
                "kind": "frozen_canonical_cache_dense_grid",
                "predictor_stride_samples": 16,
                "residual_cache_sha256": source_sha,
            },
            "s3": {
                "kind": "frozen_canonical_cache_phase32_subset",
                "predictor_stride_samples": 32,
                "residual_cache_sha256": source_sha,
            },
        }
        require(
            report,
            fold_config.get("variant_sources") == expected_variant_sources,
            "Variant source descriptions do not encode one frozen cache",
            subject=subject,
        )
        require(
            report,
            provenance_file
            == {
                "canonical_source": canonical_provenance,
                "variant_sources": fold_config.get("variant_sources"),
            },
            "source_provenance.json differs from fold_config",
            subject=subject,
        )
        expected_sources = {name: source_sha for name in EXPECTED_VARIANTS}
        require(
            report,
            fold_config.get("variant_source_residual_sha256")
            == expected_sources,
            "Variant residual-cache hash mapping mismatch",
            subject=subject,
            expected=expected_sources,
            actual=fold_config.get("variant_source_residual_sha256"),
        )
        with np.load(support_path, allow_pickle=False) as payload:
            require(
                report,
                set(payload.files) == SUPPORT_KEYS,
                "input_support.npz key set mismatch",
                subject=subject,
                actual=sorted(payload.files),
            )
            support = {
                variant: {
                    split: {
                        suffix: np.asarray(
                            payload[f"{variant}_{split}_{suffix}"],
                            dtype=(
                                np.int8
                                if suffix == "y"
                                else np.int64
                            ),
                        )
                        for suffix in (
                            "predictor_window_index",
                            "anchor_window_index",
                            "history_window_index",
                            "y",
                        )
                    }
                    for split in EXPECTED_SPLITS
                }
                for variant in EXPECTED_VARIANTS
            }

        variants_by_name = {
            str(item["variant"]): item for item in config["variants"]
        }
        for name in EXPECTED_VARIANTS:
            variant = variants_by_name[name]
            for split in EXPECTED_SPLITS:
                values = support[name][split]
                dense_indices = np.asarray(
                    source_cache[split]["window_index"],
                    dtype=np.int64,
                )
                cache_indices = (
                    dense_indices
                    if name != "s3"
                    else dense_indices[
                        suite.grid_mask(
                            windows,
                            dense_indices,
                            stride_samples=32,
                            origin_samples=origin,
                        )
                    ]
                )
                validate_history_support(
                    report,
                    windows,
                    values["predictor_window_index"],
                    values["anchor_window_index"],
                    values["history_window_index"],
                    values["y"],
                    subject=subject,
                    variant=name,
                    split=split,
                    predictor_stride=int(
                        variant["predictor_stride_samples"]
                    ),
                    classifier_stride=int(
                        variant["classifier_stride_samples"]
                    ),
                    origin=origin,
                )
                require(
                    report,
                    np.array_equal(
                        values["predictor_window_index"],
                        cache_indices,
                    ),
                    "Predictor support is not the exact frozen-cache deployment grid",
                    subject=subject,
                    variant=name,
                    split=split,
                )
                plan = suite.make_common_history_plan(
                    windows,
                    cache_indices,
                    32,
                    int(variant["predictor_stride_samples"]),
                    256,
                )
                keep = np.flatnonzero(
                    suite.grid_mask(
                        windows,
                        plan.anchor_window_indices,
                        stride_samples=int(
                            variant["classifier_stride_samples"]
                        ),
                        origin_samples=origin,
                    )
                )
                plan = plan.take(keep)
                require(
                    report,
                    np.array_equal(
                        values["anchor_window_index"],
                        plan.anchor_window_indices,
                    )
                    and np.array_equal(
                        values["history_window_index"],
                        cache_indices[plan.max_chain_rows],
                    )
                    and np.array_equal(
                        values["y"],
                        windows.label[plan.anchor_window_indices],
                    ),
                    "Saved support cannot be independently reconstructed",
                    subject=subject,
                    variant=name,
                    split=split,
                )
                for count_key, expected_count in (
                    ("predictor_window_counts", len(cache_indices)),
                    (
                        "classifier_candidate_anchor_counts",
                        len(plan.anchor_rows),
                    ),
                    (
                        "classifier_actual_anchor_counts",
                        len(plan.anchor_rows),
                    ),
                ):
                    actual = (
                        fold_config.get(count_key, {})
                        .get(name, {})
                        .get(split)
                    )
                    require(
                        report,
                        actual == expected_count,
                        "Fold support count mismatch",
                        subject=subject,
                        variant=name,
                        split=split,
                        count=count_key,
                        actual=actual,
                        expected=expected_count,
                    )

        for split in EXPECTED_SPLITS:
            s1 = support["s1"][split]
            s2 = support["s2"][split]
            s3_support = support["s3"][split]
            dense_indices = np.asarray(
                source_cache[split]["window_index"],
                dtype=np.int64,
            )
            dense_residual = np.asarray(
                source_cache[split]["residual"],
                dtype=np.float32,
            )
            s3_predictor_expected = dense_indices[
                suite.grid_mask(
                    windows,
                    dense_indices,
                    stride_samples=32,
                    origin_samples=origin,
                )
            ]
            require(
                report,
                np.array_equal(
                    s1["predictor_window_index"],
                    dense_indices,
                )
                and np.array_equal(
                    s2["predictor_window_index"],
                    dense_indices,
                )
                and np.array_equal(
                    s3_support["predictor_window_index"],
                    s3_predictor_expected,
                )
                and np.array_equal(
                    s1["anchor_window_index"],
                    s3_support["anchor_window_index"],
                )
                and np.array_equal(
                    s1["history_window_index"],
                    s3_support["history_window_index"],
                )
                and np.array_equal(s1["y"], s3_support["y"]),
                "S1 and S3 anchor/history/label support differ",
                subject=subject,
                split=split,
            )
            s1_tensor = reconstruct_history_tensor(
                dense_indices,
                dense_residual,
                s1["history_window_index"],
            )
            s3_tensor = reconstruct_history_tensor(
                dense_indices,
                dense_residual,
                s3_support["history_window_index"],
            )
            require(
                report,
                s1_tensor.shape == (len(s1["y"]), 9, 256)
                and np.array_equal(s1_tensor, s3_tensor),
                "S1/S3 reconstructed classifier tensors differ",
                subject=subject,
                split=split,
            )
            expected_unconsumed = len(dense_indices) - len(
                s3_predictor_expected
            )
            require(
                report,
                fold_config.get(
                    "s1_extra_predictor_windows_unconsumed",
                    {},
                ).get(split)
                == expected_unconsumed,
                "Unconsumed S1 extra-phase predictor count mismatch",
                subject=subject,
                split=split,
                expected=expected_unconsumed,
            )
            s2_mask = suite.grid_mask(
                windows,
                s1["anchor_window_index"],
                stride_samples=64,
                origin_samples=origin,
            )
            expected_s2 = s1["anchor_window_index"][
                s2_mask
            ]
            require(
                report,
                np.array_equal(
                    s2["anchor_window_index"],
                    expected_s2,
                )
                and np.array_equal(
                    s2["history_window_index"],
                    s1["history_window_index"][s2_mask],
                )
                and np.array_equal(s2["y"], s1["y"][s2_mask]),
                "S2 is not the fixed-phase subset of S1",
                subject=subject,
                split=split,
            )

        return {
            "fold_config": fold_config,
            "support": support,
            "support_sha256": support_sha,
            "source_cache_sha256": source_sha,
            "reference_initial_state_sha256": expected_fold_initial_sha,
        }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot validate fold support/provenance",
            subject=subject,
            error=f"{type(error).__name__}: {error}",
        )
        return None


if __name__ == "__main__":
    main()
