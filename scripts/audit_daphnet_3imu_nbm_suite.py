#!/usr/bin/env python
"""Independently audit a Daphnet three-IMU 5-NBM x 4-history suite.

The audit is deliberately stricter than checking that files exist.  It verifies
the immutable protocol, source data fingerprint, LOSO split isolation, DONE
artifact hashes, checkpoint reconstruction, prediction/metric consistency, and
the common held-out support used by every classifier cell.

By default the canonical 8-fold x 5-NBM x 4-history result is required.  Use
``--allow-partial`` for smoke runs or an interrupted suite; every completed
task is still checked with the same strict rules.  ``SUITE_COMPLETE.json`` is
written only after the canonical 160 classifier cells pass the full audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import binary_metrics, choose_threshold
from cnbr_fog.histories import make_common_history_plan
from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.nbm import NBM_NAMES, NormalBehaviourModel, build_nbm
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
    validate_checkpoint,
    validate_done,
)
from run_cnbr_fog_loso import deterministic_subsample, event_metrics


SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
EXPECTED_SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
EXPECTED_EXCLUDED = {"S04", "S10"}
EXPECTED_CHANNELS = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)
EXPECTED_IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/__init__.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/resume.py",
)
EXPECTED_HISTORIES = {
    "residual_h0p5s": {
        "history_seconds": 0.5,
        "history_samples": 32,
        "history_blocks": 1,
    },
    "residual_h1s": {
        "history_seconds": 1.0,
        "history_samples": 64,
        "history_blocks": 2,
    },
    "residual_h2s": {
        "history_seconds": 2.0,
        "history_samples": 128,
        "history_blocks": 4,
    },
    "residual_h4s": {
        "history_seconds": 4.0,
        "history_samples": 256,
        "history_blocks": 8,
    },
}

CORE_BINARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "auroc",
    "auprc",
)
EVENT_METRICS = (
    "evaluable_true_events",
    "detected_true_events",
    "predicted_events",
    "false_alarm_events",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
    "mean_detection_delay_sec",
    "evaluated_hours",
)
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
    "specificity",
    "precision",
    "mcc",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Daphnet three-IMU 5x4 NBM LOSO suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("outputs/daphnet_3imu_nbm_5x4_loso_seed42"),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Audit completed cells in a smoke or interrupted result without "
            "requiring the canonical 160 cells. Never marks a partial result complete."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(
    actual: Any,
    expected: Any,
    label: str,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(
                f"{label}: None mismatch, actual={actual!r}, expected={expected!r}"
            )
        return
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        if float(actual) != float(expected):
            raise AssertionError(
                f"{label}: non-finite mismatch, actual={actual}, expected={expected}"
            )
        return
    if not np.isclose(float(actual), float(expected), rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}: actual={float(actual):.12g}, "
            f"expected={float(expected):.12g}"
        )


def assert_metric_dict(
    saved: dict[str, Any],
    recomputed: dict[str, Any],
    keys: Iterable[str],
    label: str,
) -> None:
    for key in keys:
        require(key in saved, f"{label}: missing saved metric {key!r}")
        require(key in recomputed, f"{label}: missing recomputed metric {key!r}")
        assert_close(saved[key], recomputed[key], f"{label}/{key}")
    for key in ("tn", "fp", "fn", "tp", "n", "n_normal", "n_fog"):
        require(
            int(saved[key]) == int(recomputed[key]),
            f"{label}/{key}: {saved[key]} != {recomputed[key]}",
        )
    require(
        saved["confusion_matrix"] == recomputed["confusion_matrix"],
        f"{label}: confusion matrix mismatch",
    )


def resolved_equal(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def assert_done_artifacts(
    payload: dict[str, Any],
    expected: dict[str, Path],
    label: str,
    done_path: Path,
) -> None:
    artifacts = payload.get("artifacts")
    require(isinstance(artifacts, dict), f"{label}: DONE has no artifact map")
    require(
        set(artifacts) == set(expected),
        f"{label}: DONE artifact keys {set(artifacts)} != {set(expected)}",
    )
    for name, expected_path in expected.items():
        saved_path = Path(artifacts[name]["path"])
        if not saved_path.is_absolute():
            saved_path = done_path.parent / saved_path
        require(
            resolved_equal(saved_path, expected_path),
            f"{label}/{name}: DONE points to {saved_path}, expected {expected_path}",
        )


def protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    runtime_keys = {
        "protocol_fingerprint",
        "data_dir",
        "output_dir",
        "device",
        "resume",
        "num_workers",
    }
    return {key: value for key, value in config.items() if key not in runtime_keys}


def validate_protocol_matrix(
    config: dict[str, Any],
    allow_partial: bool,
) -> tuple[list[str], list[str], list[dict[str, Any]], bool]:
    require(config.get("suite_version") == SUITE_VERSION, "Unexpected suite version")
    require(int(config.get("n_channels", -1)) == 9, "Protocol is not 9-channel")
    require(
        tuple(config.get("channel_names", [])) == EXPECTED_CHANNELS,
        "Protocol channel names/order do not match ankle/thigh/trunk 3-axis data",
    )
    require(
        set(config.get("excluded_subjects", [])) == EXPECTED_EXCLUDED,
        "Protocol must exclude exactly S04 and S10",
    )
    require(
        tuple(config.get("subjects", [])) == EXPECTED_SUBJECTS,
        "Post-exclusion subject list is not the canonical eight subjects",
    )
    require(
        bool(config.get("cache_residuals", False)),
        "Strict prediction replay requires the suite's saved residual cache",
    )
    implementation = config.get("implementation")
    require(isinstance(implementation, dict), "Missing implementation fingerprint")
    implementation_files = implementation.get("files")
    require(
        isinstance(implementation_files, dict) and implementation_files,
        "Missing implementation file hashes",
    )
    require(
        tuple(implementation_files) == EXPECTED_IMPLEMENTATION_FILES,
        "Implementation provenance does not cover the runner's exact source set",
    )
    require(
        canonical_fingerprint(implementation_files)
        == implementation.get("sha256"),
        "Implementation aggregate fingerprint is invalid",
    )
    for relative, expected_sha in implementation_files.items():
        source_path = REPO_ROOT / relative
        if source_path.exists():
            require(
                sha256_file(source_path) == expected_sha,
                f"Implementation source drift detected: {relative}",
            )

    fingerprint = canonical_fingerprint(protocol_payload(config))
    require(
        fingerprint == config.get("protocol_fingerprint"),
        "Protocol fingerprint does not match config.json contents",
    )

    folds = list(config.get("folds_resolved", []))
    require(folds and len(folds) == len(set(folds)), "Invalid/duplicate resolved folds")
    require(
        set(folds).issubset(EXPECTED_SUBJECTS),
        f"Unknown or excluded fold in {folds}",
    )

    nbms = list(config.get("nbms_resolved", []))
    require(nbms and len(nbms) == len(set(nbms)), "Invalid/duplicate NBM list")
    require(
        set(nbms).issubset(set(NBM_NAMES)),
        f"Unknown NBM in {nbms}",
    )

    variants = list(config.get("history_variants", []))
    names = [str(item.get("input")) for item in variants]
    require(variants and len(names) == len(set(names)), "Invalid/duplicate histories")
    require(
        set(names).issubset(EXPECTED_HISTORIES),
        f"Unknown residual history in {names}",
    )
    for item in variants:
        name = str(item["input"])
        expected = EXPECTED_HISTORIES[name]
        assert_close(
            item["history_seconds"],
            expected["history_seconds"],
            f"config/{name}/history_seconds",
        )
        require(
            int(item["history_samples"]) == expected["history_samples"],
            f"config/{name}: wrong history sample count",
        )
        require(
            int(item["history_blocks"]) == expected["history_blocks"],
            f"config/{name}: wrong history block count",
        )

    full_protocol = (
        set(folds) == set(EXPECTED_SUBJECTS)
        and len(folds) == len(EXPECTED_SUBJECTS)
        and set(nbms) == set(NBM_NAMES)
        and len(nbms) == len(NBM_NAMES)
        and set(names) == set(EXPECTED_HISTORIES)
        and len(names) == len(EXPECTED_HISTORIES)
    )
    if not allow_partial:
        require(
            full_protocol,
            "Canonical 8-fold x 5-NBM x 4-history config required; "
            "use --allow-partial for a smoke result",
        )
    return folds, nbms, variants, full_protocol


def load_and_validate_dataset(
    config: dict[str, Any],
) -> tuple[DaphnetDataset, WindowTable]:
    data_root = Path(config["data_dir"]).resolve()
    require(data_root.exists(), f"Configured data directory does not exist: {data_root}")
    actual_data_sha = dataset_fingerprint(data_root)
    require(
        actual_data_sha == config.get("data_sha256"),
        "Daphnet source data fingerprint differs from config.json",
    )

    source = DaphnetDataset.load(
        data_root,
        flatline_seconds=float(config["flatline_seconds"]),
        zero_tolerance=float(config["zero_tolerance"]),
    )
    require(source.n_channels == 9, "Loaded source dataset is not 9-channel")
    require(
        tuple(source.channel_names) == EXPECTED_CHANNELS,
        "Loaded source channel names/order differ from the protocol",
    )
    require(
        list(source.subjects) == list(config["source_subjects"]),
        "Source subject list differs from config.json",
    )
    require(
        EXPECTED_EXCLUDED.issubset(source.subjects),
        "The source dataset does not contain both excluded subjects",
    )

    filtered = DaphnetDataset(
        root=source.root,
        records=[
            record
            for record in source.records
            if record.subject_id not in EXPECTED_EXCLUDED
        ],
        sampling_rate_hz=source.sampling_rate_hz,
        channel_names=source.channel_names,
    )
    require(
        tuple(filtered.subjects) == EXPECTED_SUBJECTS,
        "Filtered dataset does not contain exactly the canonical eight subjects",
    )
    require(
        not any(
            record.subject_id in EXPECTED_EXCLUDED for record in filtered.records
        ),
        "S04 or S10 survived dataset filtering",
    )
    require(
        int(config["sampling_rate_hz"]) == filtered.sampling_rate_hz,
        "Sampling-rate mismatch",
    )

    windows = filtered.make_windows(
        warmup_samples=int(config["context_samples"]),
        target_samples=int(config["horizon_samples"]),
        stride_samples=int(config["stride_samples"]),
        fog_fraction_threshold=float(config["fog_fraction_threshold"]),
        normal_guard_samples=int(config["normal_guard_samples"]),
    )
    require(
        len(windows) == int(config["window_count"]),
        f"Window count mismatch: {len(windows)} != {config['window_count']}",
    )
    counts = np.bincount(windows.label, minlength=2).astype(int).tolist()
    require(
        counts == list(config["window_class_counts"]),
        f"Window class-count mismatch: {counts} != {config['window_class_counts']}",
    )
    return filtered, windows


def check_indices(indices: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64)
    require(result.ndim == 1, f"{label}: indices must be one-dimensional")
    require(
        len(result) == len(np.unique(result)),
        f"{label}: duplicate window indices",
    )
    if len(result):
        require(int(result.min()) >= 0, f"{label}: negative window index")
        require(int(result.max()) < size, f"{label}: window index out of range")
    return result


def subjects_for_windows(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
) -> set[str]:
    return {
        dataset.records[int(record_index)].subject_id
        for record_index in windows.record_index[indices]
    }


def validate_fold_files(
    root: Path,
    subject: str,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
) -> dict[str, Any]:
    fold_root = root / f"loso_{subject}"
    fold_config = load_json(fold_root / "fold_config.json")
    require(
        fold_config.get("protocol_fingerprint") == config["protocol_fingerprint"],
        f"{subject}: fold protocol mismatch",
    )
    require(fold_config.get("test_subject") == subject, f"{subject}: wrong test subject")
    require(
        set(fold_config.get("excluded_subjects", [])) == EXPECTED_EXCLUDED,
        f"{subject}: wrong fold exclusions",
    )
    require(
        tuple(fold_config.get("channel_names", [])) == EXPECTED_CHANNELS,
        f"{subject}: wrong fold channels",
    )
    validation_subject = str(fold_config["val_subject"])
    training_subjects = list(fold_config["train_subjects"])
    require(
        validation_subject != subject,
        f"{subject}: validation subject equals the test subject",
    )
    require(
        len(training_subjects) == 6
        and len(set(training_subjects)) == len(training_subjects),
        f"{subject}: training subjects must be six unique subjects",
    )
    subject_order = list(dataset.subjects)
    fold_index = subject_order.index(subject)
    expected_validation_subject = ""
    for offset in range(1, len(subject_order)):
        candidate = subject_order[(fold_index + offset) % len(subject_order)]
        candidate_indices = dataset.window_indices_for_subjects(
            windows, [candidate]
        )
        if np.unique(windows.label[candidate_indices]).size == 2:
            expected_validation_subject = candidate
            break
    require(
        validation_subject == expected_validation_subject,
        f"{subject}: validation subject {validation_subject} differs from "
        f"deterministic selection {expected_validation_subject}",
    )
    expected_training_subjects = [
        candidate
        for candidate in subject_order
        if candidate not in {subject, expected_validation_subject}
    ]
    require(
        training_subjects == expected_training_subjects,
        f"{subject}: training-subject order/content differs from protocol",
    )
    participants = set(training_subjects) | {validation_subject, subject}
    require(
        not participants.intersection(EXPECTED_EXCLUDED),
        f"{subject}: excluded subject appears in fold participants",
    )
    require(
        participants == set(EXPECTED_SUBJECTS),
        f"{subject}: fold participants do not cover the eight subjects",
    )
    require(
        subject not in training_subjects and validation_subject not in training_subjects,
        f"{subject}: train/validation/test are not disjoint",
    )

    scaler = load_json(fold_root / "scaler.json")
    require(scaler == fold_config["scaler"], f"{subject}: scaler files differ")
    center = np.asarray(scaler["center"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    require(center.shape == (9,), f"{subject}: scaler center is not 9-channel")
    require(scale.shape == (9,), f"{subject}: scaler scale is not 9-channel")
    require(np.isfinite(center).all(), f"{subject}: non-finite scaler center")
    require(
        np.isfinite(scale).all() and np.all(scale > 0),
        f"{subject}: invalid scaler scale",
    )
    recomputed_scaler = dataset.fit_scaler(
        training_subjects, clip=float(config["robust_clip"])
    )
    require(
        np.array_equal(center, recomputed_scaler.center.astype(np.float64)),
        f"{subject}: scaler center differs from training-only recomputation",
    )
    require(
        np.array_equal(scale, recomputed_scaler.scale.astype(np.float64)),
        f"{subject}: scaler scale differs from training-only recomputation",
    )
    assert_close(
        scaler["clip"],
        recomputed_scaler.clip,
        f"{subject}/scaler_clip",
        rtol=0.0,
        atol=0.0,
    )

    split_path = fold_root / "split_indices.npz"
    support_path = fold_root / "history_support.npz"
    require(split_path.exists(), f"{subject}: missing split_indices.npz")
    require(support_path.exists(), f"{subject}: missing history_support.npz")
    with np.load(split_path, allow_pickle=False) as payload:
        required = {
            "train_window_index",
            "validation_window_index",
            "test_window_index",
            "normal_train_window_index",
            "normal_validation_window_index",
        }
        require(set(payload.files) == required, f"{subject}: unexpected split keys")
        train = check_indices(payload["train_window_index"], len(windows), f"{subject}/train")
        validation = check_indices(
            payload["validation_window_index"], len(windows), f"{subject}/validation"
        )
        test = check_indices(payload["test_window_index"], len(windows), f"{subject}/test")
        normal_train = check_indices(
            payload["normal_train_window_index"],
            len(windows),
            f"{subject}/normal_train",
        )
        normal_validation = check_indices(
            payload["normal_validation_window_index"],
            len(windows),
            f"{subject}/normal_validation",
        )

    expected_splits = {
        "train": dataset.window_indices_for_subjects(
            windows, expected_training_subjects
        ),
        "validation": dataset.window_indices_for_subjects(
            windows, [expected_validation_subject]
        ),
        "test": dataset.window_indices_for_subjects(windows, [subject]),
    }
    for split_name, saved_indices in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        require(
            np.array_equal(saved_indices, expected_splits[split_name]),
            f"{subject}: {split_name} indices differ from exact LOSO rebuild",
        )

    expected_normal_train = dataset.window_indices_for_subjects(
        windows, expected_training_subjects, clean_normal_only=True
    )
    expected_normal_train = deterministic_subsample(
        expected_normal_train,
        int(config["max_normal_windows"]),
        int(config["seed"]) + fold_index,
    )
    expected_normal_validation = dataset.window_indices_for_subjects(
        windows, [expected_validation_subject], clean_normal_only=True
    )
    require(
        np.array_equal(normal_train, expected_normal_train),
        f"{subject}: normal-train indices differ from deterministic clean "
        "non-FOG rebuild",
    )
    require(
        np.array_equal(normal_validation, expected_normal_validation),
        f"{subject}: normal-validation indices differ from clean non-FOG rebuild",
    )

    require(
        subjects_for_windows(dataset, windows, train) == set(training_subjects),
        f"{subject}: train indices belong to the wrong subjects",
    )
    require(
        subjects_for_windows(dataset, windows, validation) == {validation_subject},
        f"{subject}: validation indices belong to the wrong subject",
    )
    require(
        subjects_for_windows(dataset, windows, test) == {subject},
        f"{subject}: test indices belong to the wrong subject",
    )
    require(
        np.isin(normal_train, train).all(),
        f"{subject}: normal-train indices are not a train subset",
    )
    require(
        np.isin(normal_validation, validation).all(),
        f"{subject}: normal-validation indices are not a validation subset",
    )
    for label, indices in (
        ("normal_train", normal_train),
        ("normal_validation", normal_validation),
    ):
        require(np.all(windows.clean_normal[indices]), f"{subject}: {label} is not clean")
        require(np.all(windows.label[indices] == 0), f"{subject}: {label} contains FOG")

    source_counts = fold_config.get("source_window_counts")
    require(isinstance(source_counts, dict), f"{subject}: missing source counts")
    for split_name, expected_indices in expected_splits.items():
        require(
            int(source_counts[split_name]) == len(expected_indices),
            f"{subject}/{split_name}: source-window count mismatch",
        )

    maximum_history_samples = max(
        int(item["history_samples"]) for item in config["history_variants"]
    )
    maximum_blocks = max(
        int(item["history_blocks"]) for item in config["history_variants"]
    )
    expected_plans = {
        split_name: make_common_history_plan(
            windows,
            split_indices,
            int(config["horizon_samples"]),
            int(config["stride_samples"]),
            maximum_history_samples,
        )
        for split_name, split_indices in expected_splits.items()
    }
    if int(config["max_classifier_windows"]) > 0:
        train_plan = expected_plans["train"]
        plan_rows = np.arange(len(train_plan.anchor_rows), dtype=np.int64)
        plan_labels = windows.label[train_plan.anchor_window_indices]
        selected_rows = deterministic_subsample(
            plan_rows,
            int(config["max_classifier_windows"]),
            int(config["seed"]) + 100 + fold_index,
            plan_labels,
        )
        expected_plans["train"] = train_plan.take(selected_rows)

    supports: dict[str, dict[str, np.ndarray]] = {}
    with np.load(support_path, allow_pickle=False) as payload:
        expected_keys = {
            f"{split}_{kind}_window_index"
            for split in ("train", "validation", "test")
            for kind in ("anchor", "history")
        }
        require(set(payload.files) == expected_keys, f"{subject}: bad support keys")
        for split in ("train", "validation", "test"):
            anchor = np.asarray(
                payload[f"{split}_anchor_window_index"], dtype=np.int64
            )
            chain = np.asarray(
                payload[f"{split}_history_window_index"], dtype=np.int64
            )
            expected_plan = expected_plans[split]
            expected_anchor = expected_plan.anchor_window_indices
            expected_chain = expected_splits[split][
                expected_plan.max_chain_rows
            ]
            require(
                np.array_equal(anchor, expected_anchor),
                f"{subject}/{split}: history anchors differ from exact common plan",
            )
            require(
                np.array_equal(chain, expected_chain),
                f"{subject}/{split}: history chains differ from exact common plan",
            )
            require(
                chain.shape == (len(anchor), maximum_blocks),
                f"{subject}/{split}: wrong history support shape {chain.shape}",
            )
            require(
                np.array_equal(chain[:, -1], anchor),
                f"{subject}/{split}: final history block is not the anchor",
            )
            check_indices(anchor, len(windows), f"{subject}/{split}/anchors")
            require(
                np.all((chain >= 0) & (chain < len(windows))),
                f"{subject}/{split}: history index out of range",
            )
            record_ids = windows.record_index[chain]
            require(
                np.all(record_ids == record_ids[:, :1]),
                f"{subject}/{split}: history crosses a record boundary",
            )
            starts = windows.target_start[chain]
            if maximum_blocks > 1:
                require(
                    np.all(
                        np.diff(starts, axis=1)
                        == int(config["horizon_samples"])
                    ),
                    f"{subject}/{split}: history blocks are not horizon-spaced",
                )
            ends = windows.target_end[chain]
            require(
                np.all(
                    ends - starts == int(config["horizon_samples"])
                ),
                f"{subject}/{split}: history block has the wrong length",
            )
            require(
                np.all(ends <= windows.target_end[anchor, None]),
                f"{subject}/{split}: history uses a future block",
            )
            supports[split] = {"anchor": anchor, "chain": chain}

    counts = fold_config["history_anchor_counts"]
    for split in ("train", "validation", "test"):
        require(
            int(counts[split]) == len(supports[split]["anchor"]),
            f"{subject}/{split}: saved anchor count mismatch",
        )

    return {
        "root": fold_root,
        "validation_subject": validation_subject,
        "training_subjects": training_subjects,
        "scaler": recomputed_scaler,
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "support": supports,
    }


def build_nbm_from_protocol(
    config: dict[str, Any],
    name: str,
) -> NormalBehaviourModel:
    return build_nbm(
        name,
        int(config["n_channels"]),
        int(config["horizon_samples"]),
        hidden_channels=int(config["nbm_hidden"]),
        dropout=float(config["nbm_dropout"]),
        linear_ar_order=int(
            round(
                float(config["linear_ar_seconds"])
                * int(config["sampling_rate_hz"])
            )
        ),
        gru_layers=int(config["gru_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_ffn=int(config["transformer_ffn"]),
        max_context_samples=int(config["context_samples"]),
    )


def validate_resume_checkpoint(
    payload: dict[str, Any],
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str | None = None,
) -> None:
    validate_checkpoint(
        payload,
        stage=stage,
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
        upstream_sha256=upstream_sha256,
    )
    required = {
        "model_state",
        "optimizer_state",
        "grad_scaler_state",
        "epoch",
        "best_epoch",
        "bad_epochs",
        "history",
        "rng_state",
    }
    missing = sorted(required - set(payload))
    require(not missing, f"{task_id}: resume checkpoint missing {missing}")
    epoch = int(payload["epoch"])
    best_epoch = int(payload["best_epoch"])
    require(epoch >= 1, f"{task_id}: invalid completed epoch")
    require(0 <= best_epoch <= epoch, f"{task_id}: invalid best epoch")
    history = list(payload["history"])
    require(history, f"{task_id}: empty training history")
    require(
        int(history[-1]["epoch"]) == epoch,
        f"{task_id}: history does not end at checkpoint epoch",
    )


def validate_nbm_task(
    fold_root: Path,
    fold_subject: str,
    nbm_name: str,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any], NormalBehaviourModel]:
    nbm_root = fold_root / nbm_name
    stage_root = nbm_root / "nbm"
    best_path = stage_root / "best.pt"
    last_path = stage_root / "last.pt"
    training_path = stage_root / "training.json"
    done_path = stage_root / "DONE.json"
    task_id = f"loso_{fold_subject}/{nbm_name}/nbm"
    done = validate_done(
        done_path,
        stage="nbm",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    require(done is not None, f"{task_id}: missing NBM DONE")
    assert_done_artifacts(
        done,
        {"best": best_path, "last": last_path, "training": training_path},
        task_id,
        done_path,
    )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        best,
        stage="nbm",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    require(best.get("model_name") == nbm_name, f"{task_id}: wrong model name")
    expected_seed = int(config["seed"]) + EXPECTED_SUBJECTS.index(fold_subject)
    require(
        int(best["seed"]) == expected_seed,
        f"{task_id}: best-checkpoint seed mismatch",
    )
    model = build_nbm_from_protocol(config, nbm_name).cpu().eval()
    require(
        best.get("model_config") == model.model_config(),
        f"{task_id}: checkpoint model config differs from protocol",
    )
    model.load_state_dict(best["model_state"], strict=True)
    context = torch.linspace(
        -1.0,
        1.0,
        steps=2 * int(config["n_channels"]) * int(config["context_samples"]),
        dtype=torch.float32,
    ).reshape(2, int(config["n_channels"]), int(config["context_samples"]))
    with torch.no_grad():
        mean, sigma = model(context)
    expected_shape = (
        2,
        int(config["n_channels"]),
        int(config["horizon_samples"]),
    )
    require(tuple(mean.shape) == expected_shape, f"{task_id}: wrong mu shape")
    require(tuple(sigma.shape) == expected_shape, f"{task_id}: wrong sigma shape")
    require(torch.isfinite(mean).all().item(), f"{task_id}: non-finite mu")
    require(torch.isfinite(sigma).all().item(), f"{task_id}: non-finite sigma")
    require(torch.all(sigma > 0).item(), f"{task_id}: sigma is not positive")

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        last,
        stage="nbm",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    require(last.get("model_name") == nbm_name, f"{task_id}: wrong last model")
    require(
        int(last["seed"]) == expected_seed,
        f"{task_id}: last-checkpoint seed mismatch",
    )
    model.load_state_dict(last["model_state"], strict=True)
    training = load_json(training_path)
    require(training.get("model_name") == nbm_name, f"{task_id}: bad training JSON")
    require(
        int(training["seed"]) == expected_seed,
        f"{task_id}: training seed mismatch",
    )
    require(
        training.get("model_config") == best.get("model_config"),
        f"{task_id}: training/checkpoint model configs differ",
    )
    require(
        int(training["best_epoch"]) == int(best["best_epoch"]),
        f"{task_id}: best epoch mismatch",
    )
    assert_close(
        training["best_val_nll"],
        best["best_val_nll"],
        f"{task_id}/best_val_nll",
    )
    # Residual caches and classifier inputs are defined by the validation-best
    # normal-behaviour checkpoint, not by the final optimization epoch.
    model.load_state_dict(best["model_state"], strict=True)
    model.eval()
    return sha256_file(best_path), best["model_config"], model


def validate_residual_cache(
    fold_root: Path,
    fold_subject: str,
    nbm_name: str,
    config: dict[str, Any],
    nbm_sha256: str,
    fold_info: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    best_model: NormalBehaviourModel,
) -> dict[str, dict[str, np.ndarray]]:
    nbm_root = fold_root / nbm_name
    cache_path = nbm_root / "residual_cache.npz"
    diagnostics_path = nbm_root / "residual_diagnostics.json"
    done_path = nbm_root / "RESIDUAL_CACHE_DONE.json"
    task_id = f"loso_{fold_subject}/{nbm_name}/residual_cache"
    done = validate_done(
        done_path,
        stage="residual_cache",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    require(done is not None, f"{task_id}: missing residual-cache DONE")
    assert_done_artifacts(
        done,
        {"cache": cache_path, "diagnostics": diagnostics_path},
        task_id,
        done_path,
    )
    all_features: dict[str, dict[str, np.ndarray]] = {}
    with np.load(cache_path, allow_pickle=False) as payload:
        expected = {
            f"{split}_{key}"
            for split in ("train", "validation", "test")
            for key in ("residual", "y", "window_index")
        }
        require(set(payload.files) == expected, f"{task_id}: bad residual cache keys")
        for split in ("train", "validation", "test"):
            window_index = np.asarray(
                payload[f"{split}_window_index"], dtype=np.int64
            )
            labels = np.asarray(payload[f"{split}_y"], dtype=np.int8)
            residual = np.asarray(
                payload[f"{split}_residual"], dtype=np.float32
            )
            require(
                np.array_equal(window_index, fold_info["splits"][split]),
                f"{task_id}: {split} cache windows differ from the saved split",
            )
            require(
                np.array_equal(labels, windows.label[window_index]),
                f"{task_id}: {split} cache labels differ from the window table",
            )
            require(
                residual.shape
                == (
                    len(window_index),
                    int(config["n_channels"]),
                    int(config["horizon_samples"]),
                ),
                f"{task_id}: {split} residual tensor has shape {residual.shape}",
            )
            for start in range(0, len(residual), 2048):
                chunk = residual[start : start + 2048]
                require(
                    np.isfinite(chunk).all(),
                    f"{task_id}: {split} residual tensor is non-finite",
                )
                require(
                    np.all(np.abs(chunk) <= float(config["residual_clip"]) + 1e-6),
                    f"{task_id}: {split} residual exceeds residual_clip",
                )
            all_features[split] = {
                "residual": residual,
                "y": labels,
                "window_index": window_index,
            }

    context_samples = int(config["context_samples"])
    horizon_samples = int(config["horizon_samples"])
    residual_clip = float(config["residual_clip"])
    # The cache may have been emitted by CUDA kernels and is deliberately
    # replayed on CPU for portability.  A small cross-device float32 tolerance
    # is therefore required even without AMP (notably for recurrent models).
    replay_tolerance = 2e-2 if bool(config.get("amp", False)) else 2e-3
    best_model = best_model.cpu().eval()
    for split in ("train", "validation", "test"):
        features = all_features[split]
        require(
            len(features["window_index"]) > 0,
            f"{task_id}: {split} residual cache is empty",
        )
        sample_count = min(8, len(features["window_index"]))
        sample_rows = np.unique(
            np.linspace(
                0,
                len(features["window_index"]) - 1,
                num=sample_count,
                dtype=np.int64,
            )
        )
        sequences: list[np.ndarray] = []
        for row in sample_rows:
            window_index = int(features["window_index"][row])
            record_index = int(windows.record_index[window_index])
            start = int(windows.start[window_index])
            end = int(windows.target_end[window_index])
            scaled = fold_info["scaler"].transform(
                dataset.records[record_index].x[start:end]
            )
            sequences.append(np.ascontiguousarray(scaled.T, dtype=np.float32))
        sequence_tensor = torch.from_numpy(np.stack(sequences)).float()
        context = sequence_tensor[:, :, :context_samples]
        target = sequence_tensor[:, :, context_samples:]
        require(
            tuple(target.shape[1:])
            == (int(config["n_channels"]), horizon_samples),
            f"{task_id}/{split}: rebuilt target has the wrong shape",
        )
        with torch.no_grad():
            mean, sigma = best_model(context)
            require(
                torch.isfinite(mean).all().item(),
                f"{task_id}/{split}: replayed mu is non-finite",
            )
            require(
                torch.isfinite(sigma).all().item()
                and torch.all(sigma > 0).item(),
                f"{task_id}/{split}: replayed sigma is invalid",
            )
            replayed = ((target - mean) / sigma).clamp(
                -residual_clip, residual_clip
            )
        replayed_array = replayed.float().cpu().numpy()
        saved_array = features["residual"][sample_rows]
        require(
            np.allclose(
                replayed_array,
                saved_array,
                rtol=replay_tolerance,
                atol=replay_tolerance,
            ),
            f"{task_id}/{split}: cached residuals do not match "
            f"clip((target-mu)/sigma) from the best NBM "
            f"(sampled={len(sample_rows)}, "
            f"max_abs={float(np.max(np.abs(replayed_array - saved_array))):.6g})",
        )

    diagnostics = load_json(diagnostics_path)
    require(
        set(diagnostics) == {"train", "validation", "test"},
        f"{task_id}: incomplete residual diagnostics",
    )
    for split in ("train", "validation", "test"):
        saved = diagnostics[split]
        require(
            isinstance(saved, dict),
            f"{task_id}/{split}: residual diagnostics are not an object",
        )
        indices = fold_info["splits"][split]
        require(
            int(saved["windows"]) == len(indices),
            f"{task_id}/{split}: diagnostic window count mismatch",
        )
        require(
            list(saved["class_counts"])
            == np.bincount(windows.label[indices], minlength=2).astype(int).tolist(),
            f"{task_id}/{split}: diagnostic class counts mismatch",
        )
        for key in (
            "forecast_rmse",
            "forecast_mae",
            "mean_sigma",
            "residual_abs_mean",
            "residual_rms",
        ):
            require(
                math.isfinite(float(saved[key])),
                f"{task_id}/{split}: non-finite diagnostic {key}",
            )
        require(
            float(saved["mean_sigma"]) > 0.0,
            f"{task_id}/{split}: diagnostic sigma is not positive",
        )
    return all_features


def load_predictions(path: Path, label: str) -> dict[str, np.ndarray]:
    require(path.exists(), f"{label}: missing {path.name}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"window_index", "y_true", "y_prob", "y_pred"}
        require(set(payload.files) == required, f"{label}: bad prediction keys")
        result = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    lengths = {len(value) for value in result.values()}
    require(len(lengths) == 1, f"{label}: prediction arrays have different lengths")
    require(result["window_index"].ndim == 1, f"{label}: predictions are not 1D")
    require(
        len(np.unique(result["window_index"])) == len(result["window_index"]),
        f"{label}: duplicate prediction windows",
    )
    require(np.isin(result["y_true"], [0, 1]).all(), f"{label}: invalid y_true")
    require(np.isin(result["y_pred"], [0, 1]).all(), f"{label}: invalid y_pred")
    require(np.isfinite(result["y_prob"]).all(), f"{label}: non-finite probability")
    require(
        np.all((result["y_prob"] >= 0.0) & (result["y_prob"] <= 1.0)),
        f"{label}: probability outside [0,1]",
    )
    return result


def classifier_architecture_signature(
    payload: dict[str, Any],
) -> str:
    state = payload["model_state"]
    structure = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in state.items()
    ]
    return canonical_fingerprint(
        {
            "class": "ResidualTCNClassifier",
            "config": payload["classifier_config"],
            "state_structure": structure,
        }
    )


def materialize_history_from_cache(
    features: dict[str, np.ndarray],
    history_chain: np.ndarray,
    history_blocks: int,
    history_samples: int,
    label: str,
) -> np.ndarray:
    source_indices = np.asarray(features["window_index"], dtype=np.int64)
    require(
        len(source_indices) == len(np.unique(source_indices)),
        f"{label}: duplicate residual-cache indices",
    )
    order = np.argsort(source_indices)
    sorted_indices = source_indices[order]
    chain = np.asarray(history_chain, dtype=np.int64)[:, -int(history_blocks) :]
    positions = np.searchsorted(sorted_indices, chain)
    require(
        np.all(positions < len(sorted_indices)),
        f"{label}: history references a window absent from residual cache",
    )
    require(
        np.array_equal(sorted_indices[positions], chain),
        f"{label}: history/cache window mapping mismatch",
    )
    rows = order[positions]
    blocks = np.asarray(features["residual"], dtype=np.float32)[rows]
    # [anchor,block,channel,horizon] -> [anchor,channel,history]
    result = blocks.transpose(0, 2, 1, 3).reshape(
        len(chain), blocks.shape[2], -1
    )
    require(
        result.shape[-1] == int(history_samples),
        f"{label}: reconstructed history has {result.shape[-1]} samples",
    )
    return np.ascontiguousarray(result, dtype=np.float32)


@torch.no_grad()
def classifier_probabilities(
    model: ResidualTCNClassifier,
    inputs: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(inputs), int(batch_size)):
        batch = torch.from_numpy(inputs[start : start + batch_size]).float()
        probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    if not probabilities:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(probabilities).astype(np.float64, copy=False)


def requested_metrics(recomputed: dict[str, Any]) -> dict[str, float | None]:
    tn, fp, fn, tp = (
        int(recomputed[key]) for key in ("tn", "fp", "fn", "tp")
    )
    f1_nonfog = (
        2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    )
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "macro_f1": 0.5 * (f1_nonfog + f1_fog),
        "roc_auc": recomputed["auroc"],
        "pr_auc": recomputed["auprc"],
        "fog_recall": recomputed["sensitivity"],
        "fog_f1": f1_fog,
    }


def validate_classifier_task(
    fold_info: dict[str, Any],
    fold_subject: str,
    nbm_name: str,
    variant: dict[str, Any],
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    nbm_sha256: str,
    residual_features: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    input_name = str(variant["input"])
    task_root = fold_info["root"] / nbm_name / input_name
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    metrics_path = task_root / "metrics.json"
    predictions_path = task_root / "predictions.npz"
    validation_predictions_path = task_root / "validation_predictions.npz"
    predictions_csv_path = task_root / "predictions.csv"
    done_path = task_root / "DONE.json"
    task_id = f"{fold_subject}/{nbm_name}/{input_name}"

    done = validate_done(
        done_path,
        stage="classifier",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    require(done is not None, f"{task_id}: missing classifier DONE")
    assert_done_artifacts(
        done,
        {
            "best": best_path,
            "last": last_path,
            "metrics": metrics_path,
            "predictions": predictions_path,
            "validation_predictions": validation_predictions_path,
            "predictions_csv": predictions_csv_path,
        },
        task_id,
        done_path,
    )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        best,
        stage="classifier",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    expected_classifier_seed = (
        int(config["seed"])
        + 10000
        + EXPECTED_SUBJECTS.index(fold_subject)
    )
    require(
        int(best["classifier_seed"]) == expected_classifier_seed,
        f"{task_id}: best-checkpoint classifier seed mismatch",
    )
    expected_classifier_config = {
        "in_channels": 9,
        "hidden_channels": int(config["classifier_hidden"]),
        "dropout": float(config["classifier_dropout"]),
    }
    require(
        best.get("classifier_config") == expected_classifier_config,
        f"{task_id}: classifier config differs from protocol",
    )
    require(best.get("input_name") == input_name, f"{task_id}: wrong input name")
    classifier = ResidualTCNClassifier(
        in_channels=9,
        hidden_channels=int(config["classifier_hidden"]),
        dropout=float(config["classifier_dropout"]),
    ).cpu().eval()
    classifier.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        logits = classifier(
            torch.zeros(
                2, 9, int(variant["history_samples"]), dtype=torch.float32
            )
        )
    require(tuple(logits.shape) == (2,), f"{task_id}: bad classifier output shape")
    require(torch.isfinite(logits).all().item(), f"{task_id}: non-finite logits")
    architecture = classifier_architecture_signature(best)

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        last,
        stage="classifier",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    require(last.get("input_name") == input_name, f"{task_id}: wrong last input")
    require(
        int(last["classifier_seed"]) == expected_classifier_seed,
        f"{task_id}: last-checkpoint classifier seed mismatch",
    )
    classifier.load_state_dict(last["model_state"], strict=True)
    # Saved predictions are always emitted from the validation-selected best
    # model, not from the final training epoch.
    classifier.load_state_dict(best["model_state"], strict=True)

    metrics = load_json(metrics_path)
    require(metrics.get("input") == input_name, f"{task_id}: metrics input mismatch")
    require(metrics.get("nbm") == nbm_name, f"{task_id}: metrics NBM mismatch")
    require(
        metrics.get("test_subject") == fold_subject,
        f"{task_id}: metrics test subject mismatch",
    )
    require(
        metrics.get("val_subject") == fold_info["validation_subject"],
        f"{task_id}: metrics validation subject mismatch",
    )
    require(
        metrics.get("experiment_id")
        == f"{nbm_name}__{input_name.removeprefix('residual_')}",
        f"{task_id}: experiment id mismatch",
    )
    require(
        metrics.get("upstream_nbm_sha256") == nbm_sha256,
        f"{task_id}: metrics upstream NBM hash mismatch",
    )
    for key in ("history_samples", "history_blocks"):
        require(
            int(metrics[key]) == int(variant[key]),
            f"{task_id}: metrics {key} mismatch",
        )
    assert_close(
        metrics["history_seconds"],
        variant["history_seconds"],
        f"{task_id}/history_seconds",
    )
    require(
        int(metrics["classifier_seed"])
        == expected_classifier_seed,
        f"{task_id}: classifier seed mismatch",
    )

    test = load_predictions(predictions_path, f"{task_id}/test")
    validation = load_predictions(
        validation_predictions_path, f"{task_id}/validation"
    )
    expected_test_index = fold_info["support"]["test"]["anchor"]
    expected_validation_index = fold_info["support"]["validation"]["anchor"]
    require(
        np.array_equal(test["window_index"], expected_test_index),
        f"{task_id}: test windows differ from common support",
    )
    require(
        np.array_equal(validation["window_index"], expected_validation_index),
        f"{task_id}: validation windows differ from common support",
    )
    require(
        np.array_equal(test["y_true"], windows.label[expected_test_index]),
        f"{task_id}: test labels differ from window table",
    )
    require(
        np.array_equal(
            validation["y_true"], windows.label[expected_validation_index]
        ),
        f"{task_id}: validation labels differ from window table",
    )

    inference_tolerance = 5e-3 if bool(config.get("amp", False)) else 3e-5
    for split, saved_predictions in (
        ("validation", validation),
        ("test", test),
    ):
        reconstructed_input = materialize_history_from_cache(
            residual_features[split],
            fold_info["support"][split]["chain"],
            int(variant["history_blocks"]),
            int(variant["history_samples"]),
            f"{task_id}/{split}",
        )
        recomputed_probability = classifier_probabilities(
            classifier, reconstructed_input
        )
        require(
            np.allclose(
                recomputed_probability,
                saved_predictions["y_prob"],
                rtol=inference_tolerance,
                atol=inference_tolerance,
            ),
            f"{task_id}/{split}: checkpoint probabilities differ from saved "
            f"predictions (max_abs={float(np.max(np.abs(recomputed_probability - saved_predictions['y_prob']))):.6g})",
        )

    threshold = float(metrics["threshold"])
    require(0.0 <= threshold <= 1.0, f"{task_id}: invalid threshold")
    require(
        np.array_equal(
            test["y_pred"], (test["y_prob"] >= threshold).astype(np.int8)
        ),
        f"{task_id}: test threshold decisions mismatch",
    )
    require(
        np.array_equal(
            validation["y_pred"],
            (validation["y_prob"] >= threshold).astype(np.int8),
        ),
        f"{task_id}: validation threshold decisions mismatch",
    )

    selected_threshold, selected_validation_metrics = choose_threshold(
        validation["y_true"], validation["y_prob"]
    )
    assert_close(
        threshold,
        selected_threshold,
        f"{task_id}/selected_threshold",
        rtol=0.0,
        atol=1e-12,
    )
    require(
        isinstance(metrics.get("validation"), dict),
        f"{task_id}: missing validation metric object",
    )
    assert_metric_dict(
        metrics["validation"],
        selected_validation_metrics,
        CORE_BINARY_METRICS,
        f"{task_id}/validation",
    )

    recomputed = binary_metrics(test["y_true"], test["y_prob"], threshold)
    assert_metric_dict(metrics, recomputed, CORE_BINARY_METRICS, f"{task_id}/test")
    for key, value in requested_metrics(recomputed).items():
        require(key in metrics, f"{task_id}: missing requested metric {key}")
        assert_close(metrics[key], value, f"{task_id}/{key}")
    assert_close(
        metrics["best_validation_auprc"],
        best["best_validation_auprc"],
        f"{task_id}/best_validation_auprc",
    )

    recomputed_events = event_metrics(
        dataset,
        windows,
        test["window_index"],
        test["y_pred"],
    )
    for key in EVENT_METRICS:
        require(key in metrics, f"{task_id}: missing event metric {key}")
        if isinstance(recomputed_events[key], (int, np.integer)):
            require(
                int(metrics[key]) == int(recomputed_events[key]),
                f"{task_id}/{key}: event-count mismatch",
            )
        else:
            assert_close(metrics[key], recomputed_events[key], f"{task_id}/{key}")

    return {
        "window_index": test["window_index"],
        "y_true": test["y_true"],
        "architecture": architecture,
        "metrics": metrics,
        "test_predictions": test,
    }


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.exists(), f"Missing root summary: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path.name}: missing CSV header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def csv_optional_number(value: str) -> float | None:
    value = str(value).strip()
    return None if value == "" else float(value)


def assert_csv_number(
    value: str,
    expected: Any,
    label: str,
    *,
    integer: bool = False,
) -> None:
    actual = csv_optional_number(value)
    if integer:
        require(actual is not None, f"{label}: missing integer")
        require(
            int(actual) == int(expected) and float(actual).is_integer(),
            f"{label}: {actual} != {expected}",
        )
    else:
        assert_close(actual, expected, label)


def subject_macro_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in SUMMARY_METRICS:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        if not values:
            result[key] = {"mean": None, "std": None, "n_folds": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "min": float(array.min()),
            "max": float(array.max()),
            "n_folds": int(array.size),
        }
    return result


def pooled_prediction_metrics(
    truths: list[np.ndarray],
    probabilities: list[np.ndarray],
    predictions: list[np.ndarray],
) -> dict[str, Any]:
    y_true = np.concatenate(truths).astype(np.int8, copy=False)
    y_prob = np.concatenate(probabilities).astype(np.float64, copy=False)
    y_pred = np.concatenate(predictions).astype(np.int8, copy=False)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall_fog = tp / (tp + fn) if tp + fn else 0.0
    recall_nonfog = tn / (tn + fp) if tn + fp else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (recall_fog + recall_nonfog),
        "macro_f1": 0.5 * (f1_fog + f1_nonfog),
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
        "fog_recall": recall_fog,
        "fog_f1": f1_fog,
        "specificity": recall_nonfog,
    }


def validate_root_summaries(
    root: Path,
    config: dict[str, Any],
    folds: list[str],
    nbms: list[str],
    variants: list[dict[str, Any]],
    cells: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, int]:
    """Prove that root summaries are a current projection of audited cells."""

    expected_experiments = len(nbms) * len(variants)
    expected_fold_cells = len(folds) * expected_experiments
    completed_fold_cells = len(cells)

    status = load_json(root / "status.json")
    require(status.get("suite_version") == SUITE_VERSION, "status: suite mismatch")
    require(
        status.get("protocol_fingerprint") == config["protocol_fingerprint"],
        "status: protocol mismatch",
    )
    for key, expected in (
        ("expected_experiments", expected_experiments),
        ("expected_fold_cells", expected_fold_cells),
        ("completed_fold_cells", completed_fold_cells),
    ):
        require(int(status[key]) == expected, f"status/{key}: stale summary")
    expected_status = (
        "complete" if completed_fold_cells == expected_fold_cells else "partial"
    )
    require(status.get("status") == expected_status, "status: stale completion state")

    manifest_fields, manifest_rows = read_csv_rows(
        root / "experiment_manifest.csv"
    )
    expected_manifest_fields = [
        "experiment_id",
        "nbm",
        "history_seconds",
        "history_samples",
        "expected_folds",
        "completed_folds",
        "status",
        "completed_subjects",
    ]
    require(
        manifest_fields == expected_manifest_fields,
        "experiment_manifest.csv: unexpected columns/order",
    )
    require(
        len(manifest_rows) == expected_experiments,
        "experiment_manifest.csv: wrong experiment count",
    )
    manifest_by_id = {
        row["experiment_id"]: row for row in manifest_rows
    }
    require(
        len(manifest_by_id) == len(manifest_rows),
        "experiment_manifest.csv: duplicate experiment rows",
    )

    for nbm_name in nbms:
        for variant in variants:
            input_name = str(variant["input"])
            experiment_id = (
                f"{nbm_name}__{input_name.removeprefix('residual_')}"
            )
            require(
                experiment_id in manifest_by_id,
                f"experiment_manifest.csv: missing {experiment_id}",
            )
            row = manifest_by_id[experiment_id]
            require(row["nbm"] == nbm_name, f"{experiment_id}: manifest NBM")
            assert_csv_number(
                row["history_seconds"],
                variant["history_seconds"],
                f"{experiment_id}/manifest_history_seconds",
            )
            assert_csv_number(
                row["history_samples"],
                variant["history_samples"],
                f"{experiment_id}/manifest_history_samples",
                integer=True,
            )
            completed_subjects = [
                subject
                for subject in folds
                if (subject, nbm_name, input_name) in cells
            ]
            assert_csv_number(
                row["expected_folds"],
                len(folds),
                f"{experiment_id}/manifest_expected_folds",
                integer=True,
            )
            assert_csv_number(
                row["completed_folds"],
                len(completed_subjects),
                f"{experiment_id}/manifest_completed_folds",
                integer=True,
            )
            group_status = (
                "complete"
                if completed_subjects == folds
                else ("partial" if completed_subjects else "pending")
            )
            require(
                row["status"] == group_status,
                f"{experiment_id}: manifest status is stale",
            )
            require(
                row["completed_subjects"] == ",".join(completed_subjects),
                f"{experiment_id}: manifest completed subjects are stale",
            )

    fold_fields, fold_rows = read_csv_rows(root / "fold_summary.csv")
    expected_fold_fields = [
        "experiment_id",
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
        *SUMMARY_METRICS,
        "tn",
        "fp",
        "fn",
        "tp",
        "best_epoch",
        "best_validation_auprc",
        "upstream_nbm_sha256",
    ]
    require(
        fold_fields == expected_fold_fields,
        "fold_summary.csv: unexpected columns/order",
    )
    require(
        len(fold_rows) == completed_fold_cells,
        "fold_summary.csv: completed-cell count is stale",
    )
    fold_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in fold_rows:
        key = (row["test_subject"], row["nbm"], row["input"])
        require(key not in fold_by_key, f"fold_summary.csv: duplicate {key}")
        fold_by_key[key] = row
    require(
        set(fold_by_key) == set(cells),
        "fold_summary.csv: cell identities differ from audited DONE cells",
    )
    integer_fields = (
        "history_samples",
        "history_blocks",
        "classifier_seed",
        "n",
        "n_normal",
        "n_fog",
        "tn",
        "fp",
        "fn",
        "tp",
        "best_epoch",
    )
    float_fields = (
        "history_seconds",
        "threshold",
        *SUMMARY_METRICS,
        "best_validation_auprc",
    )
    for key, evidence in cells.items():
        row = fold_by_key[key]
        metrics = evidence["metrics"]
        for field in (
            "experiment_id",
            "nbm",
            "input",
            "test_subject",
            "val_subject",
            "upstream_nbm_sha256",
        ):
            require(
                row[field] == str(metrics[field]),
                f"fold_summary/{key}/{field}: stale value",
            )
        for field in integer_fields:
            assert_csv_number(
                row[field],
                metrics[field],
                f"fold_summary/{key}/{field}",
                integer=True,
            )
        for field in float_fields:
            assert_csv_number(
                row[field], metrics[field], f"fold_summary/{key}/{field}"
            )

    aggregate = load_json(root / "aggregate_metrics.json")
    expected_aggregate_ids: set[str] = set()
    for nbm_name in nbms:
        for variant in variants:
            input_name = str(variant["input"])
            experiment_id = (
                f"{nbm_name}__{input_name.removeprefix('residual_')}"
            )
            completed_subjects = [
                subject
                for subject in folds
                if (subject, nbm_name, input_name) in cells
            ]
            if not completed_subjects:
                continue
            expected_aggregate_ids.add(experiment_id)
            require(
                experiment_id in aggregate,
                f"aggregate_metrics.json: missing {experiment_id}",
            )
            saved_group = aggregate[experiment_id]
            require(saved_group["nbm"] == nbm_name, f"{experiment_id}: aggregate NBM")
            require(
                saved_group["input"] == input_name,
                f"{experiment_id}: aggregate input",
            )
            assert_close(
                saved_group["history_seconds"],
                variant["history_seconds"],
                f"{experiment_id}/aggregate_history_seconds",
            )
            require(
                saved_group["completed_folds"] == completed_subjects,
                f"{experiment_id}: aggregate completed folds are stale",
            )
            evidence_rows = [
                cells[(subject, nbm_name, input_name)]
                for subject in completed_subjects
            ]
            expected_macro = subject_macro_metrics(
                [evidence["metrics"] for evidence in evidence_rows]
            )
            saved_macro = saved_group["subject_macro"]
            require(
                set(saved_macro) == set(SUMMARY_METRICS),
                f"{experiment_id}: aggregate macro keys differ",
            )
            for metric_name, expected_values in expected_macro.items():
                saved_values = saved_macro[metric_name]
                require(
                    int(saved_values["n_folds"])
                    == int(expected_values["n_folds"]),
                    f"{experiment_id}/{metric_name}: aggregate fold count",
                )
                for statistic in ("mean", "std"):
                    assert_close(
                        saved_values[statistic],
                        expected_values[statistic],
                        f"{experiment_id}/{metric_name}/{statistic}",
                    )
                if expected_values["n_folds"]:
                    for statistic in ("min", "max"):
                        assert_close(
                            saved_values[statistic],
                            expected_values[statistic],
                            f"{experiment_id}/{metric_name}/{statistic}",
                        )
            expected_pooled = pooled_prediction_metrics(
                [
                    evidence["test_predictions"]["y_true"]
                    for evidence in evidence_rows
                ],
                [
                    evidence["test_predictions"]["y_prob"]
                    for evidence in evidence_rows
                ],
                [
                    evidence["test_predictions"]["y_pred"]
                    for evidence in evidence_rows
                ],
            )
            saved_pooled = saved_group["pooled"]
            require(
                set(saved_pooled) == set(expected_pooled),
                f"{experiment_id}: pooled metric keys differ",
            )
            for metric_name, expected_value in expected_pooled.items():
                if metric_name in {"n", "tn", "fp", "fn", "tp"}:
                    require(
                        int(saved_pooled[metric_name]) == int(expected_value),
                        f"{experiment_id}/pooled/{metric_name}",
                    )
                else:
                    assert_close(
                        saved_pooled[metric_name],
                        expected_value,
                        f"{experiment_id}/pooled/{metric_name}",
                    )
    require(
        set(aggregate) == expected_aggregate_ids,
        "aggregate_metrics.json: stale or unexpected experiment groups",
    )
    return {
        "expected_experiments": expected_experiments,
        "expected_fold_cells": expected_fold_cells,
        "completed_fold_cells": completed_fold_cells,
    }


def audit() -> dict[str, Any]:
    args = parse_args()
    root = args.result_dir.resolve()
    require(root.is_dir(), f"Result directory does not exist: {root}")
    config = load_json(root / "config.json")
    folds, nbms, variants, full_protocol = validate_protocol_matrix(
        config, args.allow_partial
    )
    configured_output = Path(config["output_dir"])
    require(
        resolved_equal(configured_output, root),
        f"config output_dir {configured_output} does not match {root}",
    )
    run_manifest_path = root / "run_manifest.json"
    provenance_files: list[str] = []
    if run_manifest_path.exists():
        runtime_only = {
            "data_dir",
            "output_dir",
            "device",
            "resume",
            "num_workers",
        }
        expected_run_manifest = {
            key: value for key, value in config.items() if key not in runtime_only
        }
        require(
            load_json(run_manifest_path) == expected_run_manifest,
            "run_manifest.json differs from the immutable portion of config.json",
        )
        provenance_files.append("run_manifest.json")
    environment_path = root / "environment.json"
    if environment_path.exists():
        load_json(environment_path)
        provenance_files.append("environment.json")
    provenance_ready = len(provenance_files) == 2
    if not args.allow_partial:
        require(
            provenance_ready,
            "Full audit requires run_manifest.json and environment.json",
        )
    allowed_done_paths: set[Path] = set()
    for subject in folds:
        fold_root = root / f"loso_{subject}"
        for nbm_name in nbms:
            nbm_root = fold_root / nbm_name
            allowed_done_paths.add((nbm_root / "nbm" / "DONE.json").resolve())
            allowed_done_paths.add(
                (nbm_root / "RESIDUAL_CACHE_DONE.json").resolve()
            )
            for variant in variants:
                allowed_done_paths.add(
                    (nbm_root / str(variant["input"]) / "DONE.json").resolve()
                )
    actual_done_paths = {
        path.resolve()
        for fold_root in root.glob("loso_*")
        if fold_root.is_dir()
        for path in fold_root.rglob("*DONE.json")
    }
    unexpected_done_paths = sorted(
        str(path.relative_to(root))
        for path in actual_done_paths - allowed_done_paths
    )
    require(
        not unexpected_done_paths,
        "Result contains DONE tasks outside the configured/excluded-safe matrix: "
        f"{unexpected_done_paths}",
    )
    dataset, windows = load_and_validate_dataset(config)

    verified_nbms = 0
    verified_residual_caches = 0
    verified_cells = 0
    missing_folds: list[str] = []
    missing_nbms: list[str] = []
    missing_residual_caches: list[str] = []
    missing_cells: list[str] = []
    fold_references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    architecture_signatures: set[str] = set()
    nbm_model_configs: dict[str, dict[str, Any]] = {}
    audited_cells: dict[tuple[str, str, str], dict[str, Any]] = {}

    for subject in folds:
        fold_root = root / f"loso_{subject}"
        if not fold_root.is_dir():
            missing_folds.append(subject)
            continue
        required_fold_files = (
            fold_root / "fold_config.json",
            fold_root / "scaler.json",
            fold_root / "split_indices.npz",
            fold_root / "history_support.npz",
        )
        missing_fold_files = [
            path.name for path in required_fold_files if not path.exists()
        ]
        if missing_fold_files:
            has_completed_task = any(fold_root.rglob("*DONE.json"))
            require(
                args.allow_partial and not has_completed_task,
                f"{subject}: incomplete fold metadata {missing_fold_files} "
                "coexists with a completed task",
            )
            missing_folds.append(subject)
            continue
        fold_info = validate_fold_files(root, subject, config, dataset, windows)
        support_index = fold_info["support"]["test"]["anchor"]
        support_truth = windows.label[support_index]
        fold_references[subject] = (support_index, support_truth)

        for nbm_name in nbms:
            nbm_done_path = fold_root / nbm_name / "nbm" / "DONE.json"
            if not nbm_done_path.exists():
                missing_nbms.append(f"{subject}/{nbm_name}")
                for variant in variants:
                    cell_done = fold_root / nbm_name / variant["input"] / "DONE.json"
                    require(
                        not cell_done.exists(),
                        f"{subject}/{nbm_name}: classifier DONE exists without NBM DONE",
                    )
                continue

            nbm_sha256, model_config, best_nbm_model = validate_nbm_task(
                fold_root, subject, nbm_name, config
            )
            verified_nbms += 1
            previous_model_config = nbm_model_configs.get(nbm_name)
            if previous_model_config is None:
                nbm_model_configs[nbm_name] = model_config
            else:
                require(
                    previous_model_config == model_config,
                    f"{nbm_name}: model config differs across folds",
                )

            residual_done = fold_root / nbm_name / "RESIDUAL_CACHE_DONE.json"
            residual_verified = False
            residual_features: dict[str, dict[str, np.ndarray]] = {}
            if bool(config.get("cache_residuals", True)):
                if residual_done.exists():
                    residual_features = validate_residual_cache(
                        fold_root,
                        subject,
                        nbm_name,
                        config,
                        nbm_sha256,
                        fold_info,
                        dataset,
                        windows,
                        best_nbm_model,
                    )
                    verified_residual_caches += 1
                    residual_verified = True
                else:
                    missing_residual_caches.append(f"{subject}/{nbm_name}")

            for variant in variants:
                input_name = str(variant["input"])
                task_id = f"{subject}/{nbm_name}/{input_name}"
                done_path = fold_root / nbm_name / input_name / "DONE.json"
                if not done_path.exists():
                    missing_cells.append(task_id)
                    continue
                if bool(config.get("cache_residuals", True)):
                    require(
                        residual_verified,
                        f"{task_id}: classifier DONE exists without a valid residual cache",
                    )
                evidence = validate_classifier_task(
                    fold_info,
                    subject,
                    nbm_name,
                    variant,
                    config,
                    dataset,
                    windows,
                    nbm_sha256,
                    residual_features,
                )
                require(
                    np.array_equal(evidence["window_index"], support_index),
                    f"{task_id}: cell window_index differs within fold",
                )
                require(
                    np.array_equal(evidence["y_true"], support_truth),
                    f"{task_id}: cell y_true differs within fold",
                )
                architecture_signatures.add(evidence["architecture"])
                cell_key = (subject, nbm_name, input_name)
                require(cell_key not in audited_cells, f"Duplicate cell {cell_key}")
                audited_cells[cell_key] = evidence
                verified_cells += 1

    expected_cells = len(folds) * len(nbms) * len(variants)
    expected_nbms = len(folds) * len(nbms)
    require(
        len(architecture_signatures) <= 1,
        f"Multiple downstream TCN architectures found: {architecture_signatures}",
    )

    verified_evaluation_windows = sum(
        len(reference[0]) for reference in fold_references.values()
    )
    verified_class_counts = np.zeros(2, dtype=np.int64)
    for _, truth in fold_references.values():
        verified_class_counts += np.bincount(truth, minlength=2)
    if len(fold_references) == len(folds):
        require(
            verified_evaluation_windows == int(config["evaluation_windows"]),
            "Configured evaluation-window count differs from fold supports",
        )
        require(
            verified_class_counts.astype(int).tolist()
            == list(config["evaluation_window_class_counts"]),
            "Configured evaluation class counts differ from fold supports",
        )

    summary_counts = validate_root_summaries(
        root,
        config,
        folds,
        nbms,
        variants,
        audited_cells,
    )
    require(
        summary_counts["completed_fold_cells"] == verified_cells,
        "Root summaries do not cover exactly the audited classifier cells",
    )

    complete = (
        full_protocol
        and provenance_ready
        and not missing_folds
        and not missing_nbms
        and not missing_cells
        and (
            not bool(config.get("cache_residuals", True))
            or not missing_residual_caches
        )
        and verified_nbms == 40
        and verified_cells == 160
        and len(architecture_signatures) == 1
    )
    if not args.allow_partial:
        require(not missing_folds, f"Missing folds: {missing_folds}")
        require(not missing_nbms, f"Missing NBM tasks: {missing_nbms}")
        require(
            not missing_residual_caches,
            f"Missing residual caches: {missing_residual_caches}",
        )
        require(not missing_cells, f"Missing classifier cells: {missing_cells}")
        require(complete, "Full suite did not satisfy the 160-cell completion gate")
    else:
        require(
            verified_nbms > 0 or verified_cells > 0,
            "No completed NBM or classifier task was available to audit",
        )

    marker_path = root / "SUITE_COMPLETE.json"
    marker: dict[str, Any] | None = None
    if complete:
        marker = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "data_sha256": config["data_sha256"],
            "implementation_sha256": config["implementation"]["sha256"],
            "folds": list(EXPECTED_SUBJECTS),
            "nbms": list(NBM_NAMES),
            "history_inputs": list(EXPECTED_HISTORIES),
            "verified_nbm_tasks": verified_nbms,
            "verified_residual_caches": verified_residual_caches,
            "verified_classifier_cells": verified_cells,
            "classifier_architecture_sha256": next(
                iter(architecture_signatures)
            ),
            "audit": "scripts/audit_daphnet_3imu_nbm_suite.py",
            "audit_sha256": sha256_file(Path(__file__)),
            "summary_sha256": {
                name: sha256_file(root / name)
                for name in (
                    "fold_summary.csv",
                    "experiment_manifest.csv",
                    "aggregate_metrics.json",
                    "status.json",
                )
            },
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "verified_complete",
        }
    else:
        require(
            not marker_path.exists(),
            "A stale SUITE_COMPLETE.json exists although the current result is partial",
        )

    report = {
        "status": "verified_complete" if complete else "verified_partial",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "full_protocol": full_protocol,
        "configured_folds": folds,
        "configured_nbms": nbms,
        "configured_histories": [item["input"] for item in variants],
        "expected_nbm_tasks": expected_nbms,
        "verified_nbm_tasks": verified_nbms,
        "expected_classifier_cells": expected_cells,
        "verified_classifier_cells": verified_cells,
        "verified_residual_caches": verified_residual_caches,
        "classifier_architecture_count": len(architecture_signatures),
        "classifier_architecture_sha256": (
            next(iter(architecture_signatures))
            if architecture_signatures
            else None
        ),
        "missing_folds": missing_folds,
        "missing_nbm_tasks": missing_nbms[:20],
        "missing_residual_caches": missing_residual_caches[:20],
        "missing_classifier_cells": missing_cells[:20],
        "suite_complete_path": str(marker_path) if complete else None,
        "root_summary": summary_counts,
        "verified_provenance_files": provenance_files,
    }
    report_path = root / "AUDIT_REPORT.json"
    report["audit_report_path"] = str(report_path)
    atomic_json_dump(report, report_path)
    if marker is not None:
        marker["audit_report_sha256"] = sha256_file(report_path)
        atomic_json_dump(marker, marker_path)
    return report


def main() -> None:
    report = audit()
    print("AUDIT_OK" if report["status"] == "verified_complete" else "AUDIT_PARTIAL_OK")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
