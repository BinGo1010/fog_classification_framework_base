#!/usr/bin/env python
"""Strict Persistence-NBM + residual_h4s + TCN-M stride ablation.

The frozen Persistence checkpoint/residual cache from the canonical 3-IMU NBM
suite is used by every arm. Three deployment schedules are compared under
otherwise identical eight-fold LOSO training:

* S1: predictor stride 0.25 s, classifier stride 0.5 s;
* S2: predictor stride 0.25 s, classifier stride 1.0 s;
* S3: predictor stride 0.5 s, classifier stride 0.5 s.

All classifiers receive four seconds of uncertainty-standardised residuals
with shape ``[batch, 9, 256]`` and use the fixed TCN-M dilation schedule
``(1, 2, 4, 8, 8, 8)``. Because the representation uses horizon-spaced
non-overlapping blocks, the extra phase of S1's 0.25-second predictor grid is
not consumed at 0.5-second classifier anchors. S1 and S3 must therefore have
identical classifier tensors and deterministic results; S3 tests whether those
unused predictor calls can be removed without changing diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import (
    DaphnetDataset,
    WindowTable,
)
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.histories import HistoryPlan, make_common_history_plan, make_history_input
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    done_payload,
    sha256_file,
    validate_done,
)


SUITE_VERSION = "daphnet_persistence_h4_tcnm_stride3_loso.v1"
SOURCE_SUITE_VERSION = rf.SOURCE_SUITE_VERSION
SOURCE_NBM = rf.SOURCE_NBM
INPUT_NAME = rf.INPUT_NAME
HISTORY_SECONDS = rf.HISTORY_SECONDS
HISTORY_SAMPLES = rf.HISTORY_SAMPLES
HISTORY_BLOCKS = rf.HISTORY_BLOCKS
HORIZON_SAMPLES = 32
SOURCE_STRIDE_SAMPLES = 16
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
EXPECTED_CHANNEL_NAMES = rf.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = rf.EXPECTED_LOSO_SUBJECTS
CLASSIFICATION_METRICS = tuple(rf.CLASSIFICATION_METRICS)

STRIDE_VARIANTS: dict[str, dict[str, Any]] = {
    "s1": {
        "display_name": "S1: pred0.25s-cls0.5s",
        "predictor_stride_seconds": 0.25,
        "predictor_stride_samples": 16,
        "classifier_stride_seconds": 0.5,
        "classifier_stride_samples": 32,
        "predictor_calls_consumed_fraction": 0.5,
        "purpose": "Only reduce classifier decision frequency",
    },
    "s2": {
        "display_name": "S2: pred0.25s-cls1.0s",
        "predictor_stride_seconds": 0.25,
        "predictor_stride_samples": 16,
        "classifier_stride_seconds": 1.0,
        "classifier_stride_samples": 64,
        "predictor_calls_consumed_fraction": 0.5,
        "purpose": "Test very-low-frequency classifier decisions",
    },
    "s3": {
        "display_name": "S3: pred0.5s-cls0.5s",
        "predictor_stride_seconds": 0.5,
        "predictor_stride_samples": 32,
        "classifier_stride_seconds": 0.5,
        "classifier_stride_samples": 32,
        "predictor_calls_consumed_fraction": 1.0,
        "purpose": "Fully non-overlapping prediction and decision schedule",
    },
}

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_persistence_tcnm_stride_ablation.py",
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
    / "daphnet_persistence_h4_tcnm_stride3_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict Persistence residual_h4s TCN-M predictor/classifier "
            "stride ablation"
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
    positive_integers = {
        "classifier_hidden": args.classifier_hidden,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [
        name for name, value in positive_integers.items() if int(value) <= 0
    ]
    if invalid:
        raise ValueError(f"These options must be positive integers: {invalid}")
    if (
        args.max_classifier_windows < 0
        or args.num_workers < 0
    ):
        raise ValueError("Window cap and num-workers must be non-negative")
    if args.stop_after_completed_tasks < 0:
        raise ValueError("--stop-after-completed-tasks must be non-negative")
    if 0 < args.max_classifier_windows < 2:
        raise ValueError("--max-classifier-windows must be zero or at least two")
    if not math.isfinite(args.classifier_lr) or args.classifier_lr <= 0:
        raise ValueError("--classifier-lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0,1)")


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def experiment_id(name: str) -> str:
    definition = STRIDE_VARIANTS[name]
    predictor = str(definition["predictor_stride_seconds"]).replace(".", "p")
    classifier = str(definition["classifier_stride_seconds"]).replace(".", "p")
    return (
        f"persistence_h4s_tcnm__{name}__"
        f"pred{predictor}s_cls{classifier}s"
    )


def build_variant_protocol(
    args: argparse.Namespace,
    sampling_rate_hz: int,
) -> tuple[list[dict[str, Any]], int, str]:
    receptive_field = rf.convolutional_receptive_field(TCN_M_DILATIONS)
    if receptive_field != TCN_M_RF_SAMPLES:
        raise AssertionError("Canonical TCN-M receptive field changed")
    variants: list[dict[str, Any]] = []
    counts: set[int] = set()
    hashes: set[str] = set()
    for name, definition in STRIDE_VARIANTS.items():
        rf.set_seed(args.seed, args.deterministic)
        model = rf.build_model(
            in_channels=len(EXPECTED_CHANNEL_NAMES),
            hidden_channels=args.classifier_hidden,
            dropout=args.classifier_dropout,
            dilations=TCN_M_DILATIONS,
        )
        count = rf.parameter_count(model)
        initial_hash = rf.state_dict_sha256(model.state_dict())
        counts.add(count)
        hashes.add(initial_hash)
        variants.append(
            {
                "variant": name,
                "experiment_id": experiment_id(name),
                **definition,
                "predictor_hz": 1.0
                / float(definition["predictor_stride_seconds"]),
                "classifier_hz": 1.0
                / float(definition["classifier_stride_seconds"]),
                "dilations": list(TCN_M_DILATIONS),
                "receptive_field_samples": receptive_field,
                "receptive_field_seconds": receptive_field
                / float(sampling_rate_hz),
                "parameter_count": count,
                "reference_initial_state_sha256": initial_hash,
            }
        )
        del model
    if len(counts) != 1 or len(hashes) != 1:
        raise AssertionError("TCN-M variants do not share architecture/init")
    return variants, counts.pop(), hashes.pop()


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
        "context_samples": 128,
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": SOURCE_STRIDE_SAMPLES,
        "seed": 42,
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "classifier_hidden": int(args.classifier_hidden),
        "classifier_dropout": float(args.classifier_dropout),
        "classifier_epochs": int(args.classifier_epochs),
        "classifier_patience": int(args.classifier_patience),
        "classifier_lr": float(args.classifier_lr),
        "max_classifier_windows": int(args.max_classifier_windows),
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
                "Stride ablation must retain the canonical source protocol: "
                f"{key} expected={expected!r}, source={observed!r}"
            )

    variants, parameter_count, initial_hash = build_variant_protocol(
        args,
        dataset.sampling_rate_hz,
    )
    scientific = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "channel_names": list(dataset.channel_names),
        "n_channels": int(dataset.n_channels),
        "excluded_subjects": list(source_config["excluded_subjects"]),
        "subjects": list(dataset.subjects),
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbm": SOURCE_NBM,
        "nbm_training_policy": (
            "All variants freeze and reuse the canonical stride-16 "
            "Persistence checkpoint, conditional uncertainty, and residual "
            "cache. Predictor stride is applied only as a deployment-time "
            "fixed-phase support selection."
        ),
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "history_block_spacing_samples": HORIZON_SAMPLES,
        "history_construction": (
            "Eight chronological non-overlapping 32-sample residual blocks"
        ),
        "classification_target_definition": (
            "The final 32-sample (0.5-second) residual target label is fixed "
            "for every arm; classifier stride changes decision frequency, "
            "not the label horizon."
        ),
        "context_samples": int(source_config["context_samples"]),
        "horizon_samples": int(source_config["horizon_samples"]),
        "source_stride_samples": int(source_config["stride_samples"]),
        "source_stride_seconds": (
            int(source_config["stride_samples"])
            / float(dataset.sampling_rate_hz)
        ),
        "grid_origin_samples": int(source_config["context_samples"]),
        "window_count": int(len(windows)),
        "window_class_counts": np.bincount(
            windows.label,
            minlength=2,
        ).astype(int).tolist(),
        "variants": variants,
        "expected_experiments": len(variants),
        "expected_fold_cells": len(variants) * len(EXPECTED_LOSO_SUBJECTS),
        "classifier": {
            "name": "tcn_m",
            "hidden_channels": int(args.classifier_hidden),
            "dropout": float(args.classifier_dropout),
            "kernel_size": rf.KERNEL_SIZE,
            "convolutions_per_block": rf.CONVS_PER_BLOCK,
            "dilations": list(TCN_M_DILATIONS),
            "receptive_field_samples": TCN_M_RF_SAMPLES,
            "receptive_field_seconds": TCN_M_RF_SAMPLES
            / float(dataset.sampling_rate_hz),
            "parameter_count": parameter_count,
            "reference_initial_state_sha256": initial_hash,
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
        "delta_pr_auc_reference": "S1",
        "event_policy": {
            "minimum_positive_windows": 2,
            "merge_gap_seconds": 0.5,
            "true_event_coverage": (
                "All true events overlapping each contiguous scheduled-output "
                "span. Intentional spacing between S2 outputs remains "
                "evaluable and may be missed; missing-anchor gaps split "
                "coverage."
            ),
        },
        "fairness_contract": {
            "ablation_axis": (
                "deployment predictor-call stride and classifier output stride"
            ),
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
    }


def grid_mask(
    windows: WindowTable,
    window_indices: np.ndarray,
    *,
    stride_samples: int,
    origin_samples: int,
) -> np.ndarray:
    """Select one record-local canonical phase from the dense source grid."""

    indices = np.asarray(window_indices, dtype=np.int64)
    target_start = windows.target_start[indices].astype(np.int64)
    relative = target_start - int(origin_samples)
    if np.any(relative < 0):
        raise ValueError("A target begins before the canonical grid origin")
    return np.remainder(relative, int(stride_samples)) == 0


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def stride_aware_event_metrics(
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    y_pred: np.ndarray,
    *,
    classifier_stride_samples: int,
    minimum_positive_windows: int = 2,
    merge_gap_seconds: float = 0.5,
) -> dict[str, Any]:
    """Evaluate events without treating missing classifier anchors as coverage."""

    fs = int(dataset.sampling_rate_hz)
    expected_step = int(classifier_stride_samples)
    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for global_index, prediction in zip(window_indices, y_pred):
        record_index = int(windows.record_index[global_index])
        by_record[record_index].append(
            (
                int(windows.target_start[global_index]),
                int(windows.target_end[global_index]),
                int(prediction),
            )
        )

    true_total = 0
    true_detected = 0
    predicted_total = 0
    matched_predictions = 0
    delays: list[float] = []
    evaluated_seconds = 0.0
    merge_gap = int(round(float(merge_gap_seconds) * fs))

    for record_index, rows in by_record.items():
        record = dataset.records[record_index]
        rows.sort(key=lambda item: item[0])
        if not rows:
            continue
        # A segment break represents genuinely missing scheduled output. It is
        # never bridged by debounce or merge logic.
        segments: list[list[tuple[int, int, int]]] = []
        segment: list[tuple[int, int, int]] = []
        for row in rows:
            if segment and row[0] - segment[-1][0] != expected_step:
                segments.append(segment)
                segment = []
            segment.append(row)
        if segment:
            segments.append(segment)

        # Intentional spacing between scheduled decisions (S2) remains part of
        # monitored time, so a FoG event occurring wholly between two outputs
        # is counted and may be missed. A genuinely missing scheduled anchor
        # splits the coverage and is not scored.
        coverage_mask = np.zeros(len(record.y), dtype=bool)
        for segment_rows in segments:
            coverage_mask[segment_rows[0][0] : segment_rows[-1][1]] = True
        evaluated_seconds += float(
            np.logical_and(coverage_mask, record.valid).sum()
        ) / fs

        # Each event retains its actual evaluated target intervals. Using only
        # the enclosing [start,end) envelope would create false overlap in the
        # intentional 0.5-second gaps of S2.
        predicted_events: list[dict[str, Any]] = []
        for segment_index, segment_rows in enumerate(segments):
            positive_runs: list[list[tuple[int, int, int]]] = []
            current: list[tuple[int, int, int]] = []
            for row in segment_rows:
                if row[2] == 1:
                    current.append(row)
                elif current:
                    positive_runs.append(current)
                    current = []
            if current:
                positive_runs.append(current)

            for run in positive_runs:
                if len(run) < int(minimum_positive_windows):
                    continue
                event = {
                    "intervals": [(row[0], row[1]) for row in run],
                    "start": run[0][0],
                    "end": run[-1][1],
                    "decision": run[
                        int(minimum_positive_windows) - 1
                    ][1],
                }
                if (
                    predicted_events
                    and predicted_events[-1]["segment"] == segment_index
                    and event["start"] - predicted_events[-1]["end"]
                    <= merge_gap
                ):
                    predicted_events[-1]["intervals"].extend(
                        event["intervals"]
                    )
                    predicted_events[-1]["end"] = event["end"]
                else:
                    event["segment"] = segment_index
                    predicted_events.append(event)

        true_intervals = [
            (start, end)
            for start, end in _boolean_runs(record.y == 1)
            if coverage_mask[start:end].any()
        ]
        true_total += len(true_intervals)
        predicted_total += len(predicted_events)
        used_predictions: set[int] = set()
        for true_start, true_end in true_intervals:
            matches = [
                index
                for index, event in enumerate(predicted_events)
                if index not in used_predictions
                and any(
                    max(true_start, pred_start) < min(true_end, pred_end)
                    for pred_start, pred_end in event["intervals"]
                )
            ]
            if not matches:
                continue
            match = min(
                matches,
                key=lambda index: predicted_events[index]["start"],
            )
            used_predictions.add(match)
            true_detected += 1
            decision_sample = predicted_events[match]["decision"]
            delays.append(max(0.0, (decision_sample - true_start) / fs))
        matched_predictions += len(used_predictions)

    false_events = predicted_total - matched_predictions
    return {
        "evaluable_true_events": int(true_total),
        "detected_true_events": int(true_detected),
        "predicted_events": int(predicted_total),
        "false_alarm_events": int(false_events),
        "event_sensitivity": (
            true_detected / true_total if true_total else None
        ),
        "false_alarm_events_per_hour": (
            false_events / (evaluated_seconds / 3600.0)
            if evaluated_seconds
            else None
        ),
        "median_detection_delay_sec": (
            float(np.median(delays)) if delays else None
        ),
        "mean_detection_delay_sec": (
            float(np.mean(delays)) if delays else None
        ),
        "evaluated_hours": evaluated_seconds / 3600.0,
    }


def _load_source_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    subject: str,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    extracted, provenance = rf._load_source_cache(args, config, subject)
    fold_config_path = (
        args.source_suite_dir / f"loso_{subject}" / "fold_config.json"
    )
    source_fold_config = rf._load_json(fold_config_path)
    if source_fold_config.get("protocol_fingerprint") != config["source"][
        "source_protocol_fingerprint"
    ]:
        raise ValueError(f"Source fold protocol mismatch: {subject}")
    if source_fold_config.get("test_subject") != subject:
        raise ValueError(f"Source fold subject mismatch: {subject}")
    expected_fold_config_sha256 = config["source"]["folds"][subject][
        "source_fold_config_sha256"
    ]
    if sha256_file(fold_config_path) != expected_fold_config_sha256:
        raise ValueError(f"Source fold config hash changed: {subject}")
    return extracted, provenance, source_fold_config, fold_config_path


def _validate_dense_source_support(
    args: argparse.Namespace,
    config: dict[str, Any],
    subject: str,
    extracted: Mapping[str, Mapping[str, np.ndarray]],
    plans: Mapping[str, HistoryPlan],
    source_fold_config: Mapping[str, Any],
) -> tuple[Path, dict[str, Path]]:
    source_support_path = (
        args.source_suite_dir / f"loso_{subject}" / "history_support.npz"
    )
    expected_support_sha256 = config["source"]["folds"][subject][
        "source_history_support_sha256"
    ]
    if sha256_file(source_support_path) != expected_support_sha256:
        raise ValueError(f"Source history support hash changed: {subject}")
    expected_keys = {
        f"{split}_{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("anchor_window_index", "history_window_index")
    }
    with np.load(source_support_path, allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError(f"Unexpected source support arrays: {subject}")
        for split, plan in plans.items():
            indices = np.asarray(
                extracted[split]["window_index"],
                dtype=np.int64,
            )
            if not np.array_equal(
                payload[f"{split}_anchor_window_index"],
                plan.anchor_window_indices,
            ):
                raise ValueError(f"Dense anchors changed: {subject}/{split}")
            if not np.array_equal(
                payload[f"{split}_history_window_index"],
                indices[plan.max_chain_rows],
            ):
                raise ValueError(f"Dense chains changed: {subject}/{split}")
            if len(plan.anchor_rows) != int(
                source_fold_config["history_anchor_counts"][split]
            ):
                raise ValueError(
                    f"Dense anchor count changed: {subject}/{split}"
                )

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
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(
                payload["window_index"],
                plans[split].anchor_window_indices,
            ):
                raise ValueError(
                    f"Source classifier anchors changed: {subject}/{split}"
                )
            if not np.array_equal(
                payload["y_true"],
                np.asarray(extracted[split]["y"])[plans[split].anchor_rows],
            ):
                raise ValueError(
                    f"Source classifier labels changed: {subject}/{split}"
                )
    return source_support_path, source_prediction_files


def prepare_fold_inputs(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
) -> tuple[
    Path,
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, Any],
]:
    fold_root = args.output_dir / f"loso_{subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    (
        source_extracted,
        source_provenance,
        source_fold_config,
        source_fold_config_path,
    ) = _load_source_fold(args, config, subject)

    dense_plans: dict[str, HistoryPlan] = {}
    for split in ("train", "validation", "test"):
        residual = np.asarray(source_extracted[split]["residual"])
        labels = np.asarray(source_extracted[split]["y"], dtype=np.int8)
        indices = np.asarray(
            source_extracted[split]["window_index"],
            dtype=np.int64,
        )
        if residual.shape[1:] != (
            dataset.n_channels,
            HORIZON_SAMPLES,
        ):
            raise ValueError(f"Unexpected residual shape: {subject}/{split}")
        if not np.isfinite(residual).all():
            raise ValueError(f"Non-finite residual: {subject}/{split}")
        if not (
            len(residual) == len(labels) == len(indices)
            and np.array_equal(labels, windows.label[indices])
        ):
            raise ValueError(f"Source cache misalignment: {subject}/{split}")
        if len(indices) != int(
            source_fold_config["source_window_counts"][split]
        ):
            raise ValueError(f"Source window count changed: {subject}/{split}")
        dense_plans[split] = make_common_history_plan(
            windows,
            indices,
            HORIZON_SAMPLES,
            SOURCE_STRIDE_SAMPLES,
            HISTORY_SAMPLES,
        )

    source_support_path, source_prediction_files = (
        _validate_dense_source_support(
            args,
            config,
            subject,
            source_extracted,
            dense_plans,
            source_fold_config,
        )
    )
    origin = int(config["grid_origin_samples"])
    s3_extracted: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        dense_indices = np.asarray(
            source_extracted[split]["window_index"],
            dtype=np.int64,
        )
        selected = grid_mask(
            windows,
            dense_indices,
            stride_samples=32,
            origin_samples=origin,
        )
        s3_extracted[split] = {
            key: np.asarray(value)[selected]
            for key, value in source_extracted[split].items()
        }
    extracted_by_variant = {
        "s1": source_extracted,
        "s2": source_extracted,
        "s3": s3_extracted,
    }
    base_plans_by_variant: dict[str, dict[str, HistoryPlan]] = {
        "s1": dense_plans,
        "s2": dense_plans,
        "s3": {
            split: make_common_history_plan(
                windows,
                np.asarray(
                    s3_extracted[split]["window_index"],
                    dtype=np.int64,
                ),
                HORIZON_SAMPLES,
                32,
                HISTORY_SAMPLES,
            )
            for split in ("train", "validation", "test")
        },
    }

    inputs_by_variant: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ] = {}
    plans_by_variant: dict[str, dict[str, HistoryPlan]] = {}
    support_arrays: dict[str, np.ndarray] = {}
    predictor_counts: dict[str, dict[str, int]] = {}
    candidate_anchor_counts: dict[str, dict[str, int]] = {}
    actual_anchor_counts: dict[str, dict[str, int]] = {}
    fold_index = EXPECTED_LOSO_SUBJECTS.index(subject)

    for variant in config["variants"]:
        name = str(variant["variant"])
        extracted = extracted_by_variant[name]
        predictor_stride = int(variant["predictor_stride_samples"])
        classifier_stride = int(variant["classifier_stride_samples"])
        inputs_by_variant[name] = {}
        plans_by_variant[name] = {}
        predictor_counts[name] = {}
        candidate_anchor_counts[name] = {}
        actual_anchor_counts[name] = {}

        for split in ("train", "validation", "test"):
            indices = np.asarray(
                extracted[split]["window_index"],
                dtype=np.int64,
            )
            predictor_mask = grid_mask(
                windows,
                indices,
                stride_samples=predictor_stride,
                origin_samples=origin,
            )
            predictor_indices = indices[predictor_mask]
            if not np.array_equal(predictor_indices, indices):
                raise ValueError(
                    f"Variant cache contains off-grid residuals: "
                    f"{subject}/{name}/{split}"
                )
            predictor_counts[name][split] = int(len(predictor_indices))

            base_plan = base_plans_by_variant[name][split]
            classifier_mask = grid_mask(
                windows,
                base_plan.anchor_window_indices,
                stride_samples=classifier_stride,
                origin_samples=origin,
            )
            plan = base_plan.take(np.flatnonzero(classifier_mask))
            candidate_anchor_counts[name][split] = int(
                len(plan.anchor_rows)
            )
            if len(plan.anchor_rows) == 0:
                raise RuntimeError(
                    f"Empty stride support: {subject}/{name}/{split}"
                )

            chain_indices = indices[plan.max_chain_rows]
            if not np.isin(chain_indices, predictor_indices).all():
                raise ValueError(
                    f"History uses a residual outside the predictor grid: "
                    f"{subject}/{name}/{split}"
                )

            if split == "train" and args.max_classifier_windows > 0:
                rows = np.arange(len(plan.anchor_rows), dtype=np.int64)
                anchor_labels = windows.label[
                    plan.anchor_window_indices
                ]
                selected = rf.deterministic_subsample(
                    rows,
                    args.max_classifier_windows,
                    args.seed + 100 + fold_index,
                    anchor_labels,
                )
                plan = plan.take(selected)
            plans_by_variant[name][split] = plan
            actual_anchor_counts[name][split] = int(len(plan.anchor_rows))
            inputs = make_history_input(
                extracted[split],
                plan,
                INPUT_NAME,
                HISTORY_SAMPLES,
                HORIZON_SAMPLES,
                predictor_stride,
            )
            if inputs[INPUT_NAME].shape[1:] != (
                dataset.n_channels,
                HISTORY_SAMPLES,
            ):
                raise AssertionError("Classifier input is not [B,9,256]")
            inputs_by_variant[name][split] = inputs

            prefix = f"{name}_{split}"
            support_arrays[f"{prefix}_predictor_window_index"] = (
                predictor_indices
            )
            support_arrays[f"{prefix}_anchor_window_index"] = (
                plan.anchor_window_indices
            )
            support_arrays[f"{prefix}_history_window_index"] = indices[
                plan.max_chain_rows
            ]
            support_arrays[f"{prefix}_y"] = np.asarray(
                inputs["y"],
                dtype=np.int8,
            )

    for split in ("train", "validation", "test"):
        for key in (INPUT_NAME, "y", "window_index"):
            if not np.array_equal(
                inputs_by_variant["s1"][split][key],
                inputs_by_variant["s3"][split][key],
            ):
                raise AssertionError(
                    f"S1/S3 classifier support differs: {subject}/{split}/{key}"
                )
        if not np.array_equal(
            support_arrays[f"s1_{split}_history_window_index"],
            support_arrays[f"s3_{split}_history_window_index"],
        ):
            raise AssertionError(
                f"S1/S3 history support differs: {subject}/{split}"
            )
        s1_anchor = support_arrays[f"s1_{split}_anchor_window_index"]
        s2_anchor = support_arrays[f"s2_{split}_anchor_window_index"]
        if not np.isin(s2_anchor, s1_anchor).all():
            raise AssertionError(
                f"S2 anchors are not nested in S1: {subject}/{split}"
            )
        if args.max_classifier_windows == 0:
            expected_s2 = s1_anchor[
                grid_mask(
                    windows,
                    s1_anchor,
                    stride_samples=64,
                    origin_samples=origin,
                )
            ]
            if not np.array_equal(s2_anchor, expected_s2):
                raise AssertionError(
                    f"S2 is not the fixed-phase S1 subset: {subject}/{split}"
                )

    support_path = fold_root / "input_support.npz"
    rf.save_or_validate_npz(support_path, **support_arrays)
    support_sha256 = sha256_file(support_path)
    source_provenance = {
        **source_provenance,
        "source_fold_config_sha256": sha256_file(source_fold_config_path),
        "source_history_support_sha256": sha256_file(source_support_path),
        "source_history_support_bytes": int(source_support_path.stat().st_size),
        "source_validation_predictions_sha256": sha256_file(
            source_prediction_files["validation"]
        ),
        "source_test_predictions_sha256": sha256_file(
            source_prediction_files["test"]
        ),
        "input_support_sha256": support_sha256,
    }
    variant_sources: dict[str, dict[str, Any]] = {
        "s1": {
            "kind": "frozen_canonical_cache_dense_grid",
            "predictor_stride_samples": 16,
            "residual_cache_sha256": source_provenance[
                "source_residual_cache_sha256"
            ],
        },
        "s2": {
            "kind": "frozen_canonical_cache_dense_grid",
            "predictor_stride_samples": 16,
            "residual_cache_sha256": source_provenance[
                "source_residual_cache_sha256"
            ],
        },
        "s3": {
            "kind": "frozen_canonical_cache_phase32_subset",
            "predictor_stride_samples": 32,
            "residual_cache_sha256": source_provenance[
                "source_residual_cache_sha256"
            ],
        },
    }
    variant_source_residual_sha256 = {
        name: str(payload["residual_cache_sha256"])
        for name, payload in variant_sources.items()
    }
    if len(set(variant_source_residual_sha256.values())) != 1:
        raise AssertionError("All variants must share one frozen cache")

    classifier_seed = args.seed + 10000 + fold_index
    rf.set_seed(classifier_seed, args.deterministic)
    reference_model = rf.build_model(
        in_channels=dataset.n_channels,
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
        "source": source_provenance,
        "variant_sources": variant_sources,
        "variant_source_residual_sha256": (
            variant_source_residual_sha256
        ),
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "predictor_window_counts": predictor_counts,
        "classifier_candidate_anchor_counts": candidate_anchor_counts,
        "classifier_actual_anchor_counts": actual_anchor_counts,
        "s1_s3_anchor_label_support_identical": True,
        "s1_s3_classifier_tensors_identical": True,
        "s1_extra_predictor_windows_unconsumed": {
            split: (
                predictor_counts["s1"][split]
                - predictor_counts["s3"][split]
            )
            for split in ("train", "validation", "test")
        },
        "s2_is_fixed_phase_subset_of_s1": (
            args.max_classifier_windows == 0
        ),
    }
    rf.save_or_validate_json(fold_root / "fold_config.json", fold_config)
    rf.save_or_validate_json(
        fold_root / "source_provenance.json",
        {
            "canonical_source": source_provenance,
            "variant_sources": variant_sources,
        },
    )
    return fold_root, inputs_by_variant, fold_config


def task_root_for(
    output_dir: Path,
    subject: str,
    variant_name: str,
) -> Path:
    return output_dir / f"loso_{subject}" / variant_name


def stride_metadata_payload(
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(variant["variant"])
    predictor_counts = fold_config["predictor_window_counts"][name]
    classifier_counts = fold_config[
        "classifier_actual_anchor_counts"
    ][name]
    return {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": f"{fold_config['test_subject']}/{name}",
        "experiment_id": variant["experiment_id"],
        "variant": name,
        "source_residual_sha256": fold_config[
            "variant_source_residual_sha256"
        ][name],
        "input_support_sha256": fold_config["source"][
            "input_support_sha256"
        ],
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256"
        ],
        "predictor_stride_seconds": float(
            variant["predictor_stride_seconds"]
        ),
        "predictor_stride_samples": int(
            variant["predictor_stride_samples"]
        ),
        "classifier_stride_seconds": float(
            variant["classifier_stride_seconds"]
        ),
        "classifier_stride_samples": int(
            variant["classifier_stride_samples"]
        ),
        "predictor_hz": float(variant["predictor_hz"]),
        "classifier_hz": float(variant["classifier_hz"]),
        "predictor_train_windows": int(predictor_counts["train"]),
        "predictor_validation_windows": int(
            predictor_counts["validation"]
        ),
        "predictor_test_windows": int(predictor_counts["test"]),
        "classifier_train_windows": int(classifier_counts["train"]),
        "classifier_validation_windows": int(
            classifier_counts["validation"]
        ),
        "classifier_test_windows": int(classifier_counts["test"]),
        "target_time_coverage_fraction": min(
            1.0,
            HORIZON_SAMPLES
            / float(variant["classifier_stride_samples"]),
        ),
        "minimum_two_positive_confirmation_seconds": (
            HORIZON_SAMPLES
            + int(variant["classifier_stride_samples"])
        )
        / float(config["sampling_rate_hz"]),
    }


def save_stride_metadata_completion(
    task_root: Path,
    config: Mapping[str, Any],
    fold_config: Mapping[str, Any],
    variant: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    subject = str(fold_config["test_subject"])
    name = str(variant["variant"])
    classifier_done_path = task_root / "DONE.json"
    classifier_done_sha256 = sha256_file(classifier_done_path)
    metadata_path = task_root / "stride_metadata.json"
    metadata_done_path = task_root / "STRIDE_METADATA_DONE.json"
    task_id = f"{subject}/{name}/stride_metadata"
    completed = validate_done(
        metadata_done_path,
        stage="stride_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        upstream_sha256=classifier_done_sha256,
    )
    if completed is not None:
        if rf._load_json(metadata_path) != dict(metadata):
            raise ValueError(f"Stride metadata changed: {subject}/{name}")
        return
    rf.save_or_validate_json(metadata_path, dict(metadata))
    atomic_json_dump(
        done_payload(
            stage="stride_metadata",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id=task_id,
            upstream_sha256=classifier_done_sha256,
            relative_to=task_root,
            artifacts={"metadata": metadata_path},
        ),
        metadata_done_path,
    )


def train_stride_classifier(
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
    classifier_fold_config = dict(fold_config)
    classifier_fold_config["source"] = dict(fold_config["source"])
    classifier_fold_config["source"]["source_residual_cache_sha256"] = (
        fold_config["variant_source_residual_sha256"][name]
    )
    original_event_metrics = rf.event_metrics

    def configured_event_metrics(
        dataset_arg: DaphnetDataset,
        windows_arg: WindowTable,
        window_indices: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        return stride_aware_event_metrics(
            dataset_arg,
            windows_arg,
            window_indices,
            y_pred,
            classifier_stride_samples=int(
                variant["classifier_stride_samples"]
            ),
        )

    rf.event_metrics = configured_event_metrics
    try:
        metrics = rf.train_classifier_resumable(
            args,
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "shared_parameter_count": config["classifier"][
                    "parameter_count"
                ],
            },
            variant,
            task_root,
            classifier_fold_config,
            inputs,
            dataset,
            windows,
            device,
        )
    finally:
        rf.event_metrics = original_event_metrics

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
    metadata = stride_metadata_payload(config, fold_config, variant)
    base_identity = {
        "experiment_id": metadata["experiment_id"],
        "variant": metadata["variant"],
        "source_residual_sha256": metadata[
            "source_residual_sha256"
        ],
        "input_support_sha256": metadata["input_support_sha256"],
        "initial_state_sha256": metadata["initial_state_sha256"],
    }
    for key, expected in base_identity.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Completed classifier identity mismatch: "
                f"{fold_config['test_subject']}/{name}/{key}"
            )
    save_stride_metadata_completion(
        task_root,
        config,
        fold_config,
        variant,
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
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_paired_subjects": int(len(values)),
        "bootstrap_samples": int(samples),
    }


def _load_completed_cell(
    output_dir: Path,
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    root = task_root_for(output_dir, subject, str(variant["variant"]))
    done = validate_done(
        root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{variant['variant']}",
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
    support_path = fold_root / "input_support.npz"
    support_sha = sha256_file(support_path)
    name = str(variant["variant"])
    expected_residual = fold_config[
        "variant_source_residual_sha256"
    ][name]
    if expected_residual != config["source"]["folds"][subject][
        "source_residual_cache_sha256"
    ]:
        raise ValueError(f"Canonical source cache changed: {subject}/{name}")
    if done.get("source_residual_sha256") != expected_residual:
        raise ValueError(f"Classifier source residual mismatch: {root}")
    if done.get("input_support_sha256") != support_sha:
        raise ValueError(f"Classifier input support mismatch: {root}")
    if done.get("initial_state_sha256") != fold_config[
        "reference_initial_state_sha256"
    ]:
        raise ValueError(f"Classifier initial state mismatch: {root}")
    metadata_done = validate_done(
        root / "STRIDE_METADATA_DONE.json",
        stage="stride_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{name}/stride_metadata",
        upstream_sha256=sha256_file(root / "DONE.json"),
    )
    if metadata_done is None:
        # A worker may have completed the base classifier immediately before
        # interruption. Treat the small metadata stage as resumable pending
        # work so scheduler initialization cannot deadlock.
        return None
    if set(metadata_done.get("artifacts", {})) != {"metadata"}:
        raise ValueError(f"Stride metadata DONE artifact mismatch: {root}")
    expected_metadata = stride_metadata_payload(
        config,
        fold_config,
        variant,
    )
    saved_metadata = rf._load_json(root / "stride_metadata.json")
    if saved_metadata != expected_metadata:
        raise ValueError(f"Stride metadata mismatch: {root}")
    metrics = rf._load_json(root / "metrics.json")
    expected_identity = {
        "experiment_id": variant["experiment_id"],
        "variant": variant["variant"],
        "test_subject": subject,
        "nbm": SOURCE_NBM,
        "input": INPUT_NAME,
        "source_residual_sha256": expected_residual,
        "input_support_sha256": support_sha,
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256"
        ],
    }
    for key, expected in expected_identity.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Completed metrics identity mismatch: {root}/{key}"
            )
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        expected_keys = {"window_index", "y_true", "y_prob", "y_pred"}
        if set(payload.files) != expected_keys:
            raise ValueError(f"Prediction array set mismatch: {root}")
        arrays = {
            key: np.asarray(payload[key])
            for key in expected_keys
        }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not np.isfinite(arrays["y_prob"]).all():
        raise ValueError(f"Prediction arrays are invalid: {root}")
    with np.load(support_path, allow_pickle=False) as support:
        if not np.array_equal(
            arrays["window_index"],
            support[f"{name}_test_anchor_window_index"],
        ):
            raise ValueError(f"Prediction support changed: {root}")
        if not np.array_equal(
            arrays["y_true"],
            support[f"{name}_test_y"],
        ):
            raise ValueError(f"Prediction labels changed: {root}")
    return {**metrics, **saved_metadata}, arrays


def _format_mean_sd(summary: Mapping[str, Any], metric: str) -> str:
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


def _format_delta(delta: Mapping[str, Any]) -> str:
    if delta.get("mean_delta") is None:
        return ""
    return (
        f"{float(delta['mean_delta']):+.4f} "
        f"[{float(delta['ci_low']):+.4f}, "
        f"{float(delta['ci_high']):+.4f}]"
    )


def refresh_summaries(output_dir: Path, config: dict[str, Any]) -> None:
    expected_folds = list(config["folds_resolved"])
    variants = list(config["variants"])
    rows_by_variant: dict[str, dict[str, dict[str, Any]]] = {
        str(item["variant"]): {} for item in variants
    }
    arrays_by_variant: dict[str, dict[str, dict[str, np.ndarray]]] = {
        str(item["variant"]): {} for item in variants
    }
    loaded: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for variant in variants:
        name = str(variant["variant"])
        group_rows: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed: list[str] = []
        for subject in expected_folds:
            cell = _load_completed_cell(
                output_dir,
                config,
                variant,
                subject,
            )
            if cell is None:
                continue
            metrics, arrays = cell
            enriched = {
                **metrics,
                "purpose": variant["purpose"],
            }
            group_rows.append(enriched)
            fold_rows.append(enriched)
            rows_by_variant[name][subject] = enriched
            arrays_by_variant[name][subject] = arrays
            truths.append(np.asarray(arrays["y_true"], dtype=np.int8))
            probabilities.append(
                np.asarray(arrays["y_prob"], dtype=np.float64)
            )
            predictions.append(np.asarray(arrays["y_pred"], dtype=np.int8))
            completed.append(subject)

        subject_macro = (
            aggregate_fold_metrics(
                group_rows,
                list(CLASSIFICATION_METRICS),
            )
            if group_rows
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
        loaded[name] = {
            "variant": variant,
            "completed": completed,
            "subject_macro": subject_macro,
            "pooled": pooled,
        }
        manifest_rows.append(
            {
                "experiment_id": variant["experiment_id"],
                "variant": name,
                "display_name": variant["display_name"],
                "predictor_stride_seconds": variant[
                    "predictor_stride_seconds"
                ],
                "classifier_stride_seconds": variant[
                    "classifier_stride_seconds"
                ],
                "purpose": variant["purpose"],
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

    equivalence_metric_keys = (
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
    equivalent_subjects: list[str] = []
    for subject in expected_folds:
        s1_metrics = rows_by_variant["s1"].get(subject)
        s3_metrics = rows_by_variant["s3"].get(subject)
        if s1_metrics is None or s3_metrics is None:
            continue
        s1_arrays = arrays_by_variant["s1"][subject]
        s3_arrays = arrays_by_variant["s3"][subject]
        if not all(
            np.array_equal(s1_arrays[key], s3_arrays[key])
            for key in ("window_index", "y_true", "y_prob", "y_pred")
        ):
            raise ValueError(
                f"S1/S3 deterministic predictions differ: {subject}"
            )
        for key in equivalence_metric_keys:
            if s1_metrics.get(key) != s3_metrics.get(key):
                raise ValueError(
                    f"S1/S3 deterministic metric differs: {subject}/{key}"
                )
        equivalent_subjects.append(subject)
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
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
                *equivalence_metric_keys,
            ],
            "completed_equivalent_subjects": equivalent_subjects,
            "expected_subjects": list(expected_folds),
            "complete_exact_equivalence": (
                equivalent_subjects == expected_folds
            ),
            "s3_predictor_call_ratio_vs_s1": 0.5,
            "s3_classifier_call_ratio_vs_s1": 1.0,
        },
        output_dir / "stride_equivalence.json",
    )

    reference_name = "s1"
    delta_rows: list[dict[str, Any]] = []
    aggregate_experiments: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []

    for variant in variants:
        name = str(variant["variant"])
        common_subjects: list[str] = []
        differences: list[float] = []
        for subject in expected_folds:
            current = rows_by_variant[name].get(subject)
            reference = rows_by_variant[reference_name].get(subject)
            if current is None or reference is None:
                continue
            if current.get("pr_auc") is None or reference.get("pr_auc") is None:
                continue
            common_subjects.append(subject)
            differences.append(
                float(current["pr_auc"]) - float(reference["pr_auc"])
            )
        delta = paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                f"{name}__vs__{reference_name}",
            ),
        )
        delta_payload = {
            "experiment_id": variant["experiment_id"],
            "variant": name,
            "reference_variant": reference_name,
            "common_subjects": ",".join(common_subjects),
            **delta,
        }
        delta_rows.append(delta_payload)

        content = loaded[name]
        subject_macro = content["subject_macro"]
        aggregate_experiments[variant["experiment_id"]] = {
            **variant,
            "completed_folds": content["completed"],
            "subject_macro": subject_macro,
            "pooled": content["pooled"],
            "delta_pr_auc_vs_s1": delta_payload,
        }
        row = {
            "experiment_id": variant["experiment_id"],
            "variant": name,
            "display_name": variant["display_name"],
            "predictor_stride_seconds": variant[
                "predictor_stride_seconds"
            ],
            "classifier_stride_seconds": variant[
                "classifier_stride_seconds"
            ],
            "predictor_hz": variant["predictor_hz"],
            "classifier_hz": variant["classifier_hz"],
            "target_time_coverage_fraction": min(
                1.0,
                HORIZON_SAMPLES
                / float(variant["classifier_stride_samples"]),
            ),
            "completed_folds": len(content["completed"]),
            "delta_pr_auc_mean": delta["mean_delta"],
            "delta_pr_auc_ci_low": delta["ci_low"],
            "delta_pr_auc_ci_high": delta["ci_high"],
            "delta_pr_auc_n_paired_subjects": delta[
                "n_paired_subjects"
            ],
        }
        for metric in CLASSIFICATION_METRICS:
            row[f"{metric}_mean"] = subject_macro[metric]["mean"]
            row[f"{metric}_std"] = subject_macro[metric]["std"]
        summary_rows.append(row)

        publication_rows.append(
            {
                "Experiment": name.upper(),
                "Predictor stride": (
                    f"{variant['predictor_stride_seconds']:g} s"
                ),
                "Classifier stride": (
                    f"{variant['classifier_stride_seconds']:g} s"
                ),
                "PR-AUC": _format_mean_sd(subject_macro, "pr_auc"),
                "ΔPR-AUC vs S1 [95% CI]": _format_delta(delta),
                "BA": _format_mean_sd(
                    subject_macro,
                    "balanced_accuracy",
                ),
                "Macro-F1": _format_mean_sd(subject_macro, "macro_f1"),
                "AUROC": _format_mean_sd(subject_macro, "roc_auc"),
                "FoG Sensitivity/Recall": _format_mean_sd(
                    subject_macro,
                    "fog_recall",
                ),
                "Specificity": _format_mean_sd(
                    subject_macro,
                    "specificity",
                ),
                "FoG Precision": _format_mean_sd(
                    subject_macro,
                    "precision",
                ),
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
        )

        completed_rows = list(rows_by_variant[name].values())
        efficiency_rows.append(
            {
                "experiment_id": variant["experiment_id"],
                "variant": name,
                "predictor_stride_seconds": variant[
                    "predictor_stride_seconds"
                ],
                "classifier_stride_seconds": variant[
                    "classifier_stride_seconds"
                ],
                "predictor_calls_per_hour": (
                    3600.0
                    / float(variant["predictor_stride_seconds"])
                ),
                "classifier_calls_per_hour": (
                    3600.0
                    / float(variant["classifier_stride_seconds"])
                ),
                "predictor_call_ratio_vs_0p25": (
                    0.25
                    / float(variant["predictor_stride_seconds"])
                ),
                "predictor_calls_consumed_fraction": variant[
                    "predictor_calls_consumed_fraction"
                ],
                "redundant_predictor_call_fraction": (
                    1.0
                    - float(
                        variant[
                            "predictor_calls_consumed_fraction"
                        ]
                    )
                ),
                "classifier_call_ratio_vs_0p25": (
                    0.25
                    / float(variant["classifier_stride_seconds"])
                ),
                "target_time_coverage_fraction": row[
                    "target_time_coverage_fraction"
                ],
                "minimum_two_positive_confirmation_seconds": (
                    HORIZON_SAMPLES
                    + int(variant["classifier_stride_samples"])
                )
                / float(config["sampling_rate_hz"]),
                "test_predictor_windows_total": sum(
                    int(item["predictor_test_windows"])
                    for item in completed_rows
                ),
                "test_classifier_windows_total": sum(
                    int(item["classifier_test_windows"])
                    for item in completed_rows
                ),
                "completed_folds": len(completed_rows),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -float(row["pr_auc_mean"])
            if row["pr_auc_mean"] is not None
            else float("inf"),
            row["variant"],
        )
    )
    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(summary_rows, start=1)
    ]

    fold_columns = [
        "experiment_id",
        "variant",
        "display_name",
        "purpose",
        "predictor_stride_seconds",
        "classifier_stride_seconds",
        "predictor_hz",
        "classifier_hz",
        "nbm",
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
        "predictor_test_windows",
        "classifier_test_windows",
        "target_time_coverage_fraction",
        "minimum_two_positive_confirmation_seconds",
        "source_residual_sha256",
        "input_support_sha256",
    ]
    rf.atomic_csv_write(
        output_dir / "fold_summary.csv",
        fold_rows,
        fold_columns,
    )
    rf.atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        [
            "experiment_id",
            "variant",
            "display_name",
            "predictor_stride_seconds",
            "classifier_stride_seconds",
            "purpose",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    metric_columns = [
        f"{metric}_{statistic}"
        for metric in CLASSIFICATION_METRICS
        for statistic in ("mean", "std")
    ]
    rf.atomic_csv_write(
        output_dir / "aggregate_summary.csv",
        ranked_rows,
        [
            "rank",
            "experiment_id",
            "variant",
            "display_name",
            "predictor_stride_seconds",
            "classifier_stride_seconds",
            "predictor_hz",
            "classifier_hz",
            "target_time_coverage_fraction",
            "completed_folds",
            "delta_pr_auc_mean",
            "delta_pr_auc_ci_low",
            "delta_pr_auc_ci_high",
            "delta_pr_auc_n_paired_subjects",
            *metric_columns,
        ],
    )
    rf.atomic_csv_write(
        output_dir / "paired_pr_auc_deltas.csv",
        delta_rows,
        [
            "experiment_id",
            "variant",
            "reference_variant",
            "common_subjects",
            "mean_delta",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "bootstrap_samples",
        ],
    )
    rf.atomic_csv_write(
        output_dir / "publication_table.csv",
        publication_rows,
        [
            "Experiment",
            "Predictor stride",
            "Classifier stride",
            "PR-AUC",
            "ΔPR-AUC vs S1 [95% CI]",
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
        ],
    )
    rf.atomic_csv_write(
        output_dir / "efficiency_summary.csv",
        efficiency_rows,
        list(efficiency_rows[0].keys()),
    )

    best_experiment = (
        ranked_rows[0]["experiment_id"]
        if ranked_rows and ranked_rows[0]["pr_auc_mean"] is not None
        else None
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
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
            "experiments": aggregate_experiments,
            "best_experiment": best_experiment,
        },
        output_dir / "aggregate_metrics.json",
    )
    completed_cells = sum(
        int(row["completed_folds"]) for row in manifest_rows
    )
    expected_cells = len(variants) * len(expected_folds)
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_experiments": len(variants),
            "expected_fold_cells": expected_cells,
            "completed_fold_cells": completed_cells,
            "s1_s3_exact_equivalence_folds": len(
                equivalent_subjects
            ),
            "status": (
                "complete"
                if completed_cells == expected_cells
                else "partial"
            ),
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
    if source_config["suite_version"] != SOURCE_SUITE_VERSION:
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
    }
    run_manifest = {
        key: value
        for key, value in config.items()
        if key not in runtime_fields
    }
    manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if not manifest_path.exists():
            raise RuntimeError("Missing run_manifest.json for worker")
        if rf._load_json(manifest_path) != run_manifest:
            raise ValueError(f"Saved JSON is incompatible: {manifest_path}")
    else:
        rf.save_or_validate_json(manifest_path, run_manifest)
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
        atomic_json_dump(environment, args.output_dir / "environment.json")
        refresh_summaries(args.output_dir, config)

    print(
        f"[INFO] suite={SUITE_VERSION} device={device} "
        f"source={args.source_suite_dir} folds={execution_folds} "
        f"variants={list(STRIDE_VARIANTS)} input={INPUT_NAME} "
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

    completed = 0
    for subject in execution_folds:
        fold_root, inputs_by_variant, fold_config = prepare_fold_inputs(
            args,
            config,
            dataset,
            windows,
            subject,
        )
        print(
            f"[fold {subject}] train={fold_config['train_subjects']} "
            f"val={fold_config['val_subject']} "
            f"anchors={fold_config['classifier_actual_anchor_counts']}",
            flush=True,
        )
        initial_hashes: set[str] = set()
        for variant in config["variants"]:
            name = str(variant["variant"])
            metrics = train_stride_classifier(
                args,
                config,
                variant,
                task_root_for(args.output_dir, subject, name),
                fold_config,
                inputs_by_variant[name],
                dataset,
                windows,
                device,
            )
            initial_hashes.add(str(metrics["initial_state_sha256"]))
            completed += 1
            print(
                f"[fold {subject}] {variant['display_name']} "
                f"PR-AUC={metrics['pr_auc']:.4f} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"Event-Sens={metrics['event_sensitivity']} "
                f"FA/h={metrics['false_alarm_events_per_hour']}",
                flush=True,
            )
            if (
                args.stop_after_completed_tasks > 0
                and completed >= args.stop_after_completed_tasks
            ):
                raise RuntimeError(
                    "Intentional stop after completed classifier tasks"
                )
        if len(initial_hashes) != 1:
            raise AssertionError(
                f"Stride variants did not share initialization: {subject}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not worker_mode:
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
