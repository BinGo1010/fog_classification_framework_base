#!/usr/bin/env python
"""Strict auditor for the Daphnet GRU-NBM forecast-horizon ablation.

This audit is intentionally more than an artifact-existence check.  It
independently rebuilds the maximum-horizon WindowTable and every fold, verifies
that H025/H050/H100/H200 use the same decision endpoints and final-0.5-second
labels, replays sampled GRU residuals from the validation-selected checkpoint,
reconstructs each ``[B, 9, 256]`` classifier input, replays TCN-M predictions,
recomputes metrics, and validates the root summaries.

``SUITE_COMPLETE.json`` is written only when all 32 classifier cells pass the
canonical audit.  A partial audit never creates a completion marker.
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
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_gru_horizon_ablation as suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.histories import make_common_history_plan
from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.nbm import NormalBehaviourModel, build_nbm
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
    validate_checkpoint,
    validate_done,
)
from run_cnbr_fog_loso import deterministic_subsample, event_metrics


AUDIT_VERSION = "daphnet_gru_horizon4_h4_tcnm_audit.v1"
EXPECTED_SUBJECTS = tuple(suite.EXPECTED_LOSO_SUBJECTS)
EXPECTED_EXCLUDED = {"S04", "S10"}
EXPECTED_CHANNELS = tuple(suite.EXPECTED_CHANNEL_NAMES)
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_HORIZONS = {
    "H025": {"seconds": 0.25, "samples": 16, "blocks": 16},
    "H050": {"seconds": 0.50, "samples": 32, "blocks": 8},
    "H100": {"seconds": 1.00, "samples": 64, "blocks": 4},
    "H200": {"seconds": 2.00, "samples": 128, "blocks": 2},
}
EXPECTED_CLASSIFIER_CELLS = len(EXPECTED_SUBJECTS) * len(EXPECTED_HORIZONS)
EXPECTED_DILATIONS = (1, 2, 4, 8, 8, 8)
EXPECTED_RESIDUAL_KEYS = {
    f"{split}_{name}"
    for split in EXPECTED_SPLITS
    for name in ("residual", "y", "window_index")
}
EXPECTED_SPLIT_KEYS = {
    "train_window_index",
    "validation_window_index",
    "test_window_index",
    "normal_train_window_index",
    "normal_validation_window_index",
}
EXPECTED_SUPPORT_KEYS = {
    *(f"{split}_anchor_window_index" for split in EXPECTED_SPLITS),
    *(f"{split}_y" for split in EXPECTED_SPLITS),
    *(
        f"{split}_{horizon_id.lower()}_history_window_index"
        for split in EXPECTED_SPLITS
        for horizon_id in EXPECTED_HORIZONS
    ),
}
EXPECTED_PREDICTION_KEYS = {"window_index", "y_true", "y_prob", "y_pred"}
NBM_ARTIFACTS = {"best", "last", "training"}
RESIDUAL_ARTIFACTS = {"cache", "diagnostics"}
CLASSIFIER_ARTIFACTS = {
    "best",
    "last",
    "metrics",
    "predictions",
    "validation_predictions",
    "predictions_csv",
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
EVENT_METRIC_KEYS = (
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
RUNTIME_CONFIG_FIELDS = {
    "protocol_fingerprint",
    "data_dir",
    "output_dir",
    "device",
    "resume",
    "num_workers",
}


class AuditError(AssertionError):
    """A protocol or artifact violation discovered by the auditor."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the strict Daphnet GRU H025/H050/H100/H200 "
            "residual_h4s TCN-M LOSO suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Fallback processed Daphnet directory if config.json moved hosts.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Audit all completed cells without requiring 32/32 completion. "
            "Partial results can never receive SUITE_COMPLETE.json."
        ),
    )
    parser.add_argument(
        "--residual-tolerance",
        type=float,
        default=None,
        help=(
            "Optional sampled residual replay tolerance. By default 2e-2 "
            "with AMP and 2e-3 without AMP."
        ),
    )
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=None,
        help=(
            "Optional CPU/CUDA classifier replay tolerance. By default 5e-3 "
            "with AMP and 3e-5 without AMP."
        ),
    )
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.exists(), f"Missing JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"Missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable scientific portion of a saved config."""

    return {
        key: value
        for key, value in config.items()
        if key not in RUNTIME_CONFIG_FIELDS
    }


def resolved_equal(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def artifact_path(done_path: Path, entry: Mapping[str, Any]) -> Path:
    value = Path(str(entry["path"]))
    if not value.is_absolute():
        value = done_path.parent / value
    return value.resolve()


def assert_done_artifacts(
    payload: Mapping[str, Any],
    expected: Mapping[str, Path],
    done_path: Path,
    label: str,
) -> None:
    artifacts = payload.get("artifacts")
    require(isinstance(artifacts, Mapping), f"{label}: DONE artifact map missing")
    require(
        set(artifacts) == set(expected),
        f"{label}: DONE artifact keys {sorted(artifacts)} != "
        f"{sorted(expected)}",
    )
    for name, expected_path in expected.items():
        entry = artifacts[name]
        require(
            isinstance(entry, Mapping),
            f"{label}/{name}: malformed DONE artifact entry",
        )
        actual_path = artifact_path(done_path, entry)
        require(
            resolved_equal(actual_path, expected_path),
            f"{label}/{name}: DONE path {actual_path} != {expected_path}",
        )
        require(actual_path.exists(), f"{label}/{name}: artifact is missing")
        require(
            str(entry.get("sha256")) == sha256_file(actual_path),
            f"{label}/{name}: artifact SHA-256 mismatch",
        )
        require(
            int(entry.get("bytes", -1)) == int(actual_path.stat().st_size),
            f"{label}/{name}: artifact byte count mismatch",
        )


def assert_close(
    actual: Any,
    expected: Any,
    label: str,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    if actual is None or expected is None:
        require(
            actual is None and expected is None,
            f"{label}: {actual!r} != {expected!r}",
        )
        return
    left = float(actual)
    right = float(expected)
    if not (math.isfinite(left) and math.isfinite(right)):
        require(left == right, f"{label}: {left!r} != {right!r}")
        return
    require(
        bool(np.isclose(left, right, rtol=rtol, atol=atol)),
        f"{label}: {left:.12g} != {right:.12g}",
    )


def assert_metric_dict(
    saved: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    keys: Iterable[str],
    label: str,
) -> None:
    for key in keys:
        require(key in saved, f"{label}: missing saved metric {key}")
        require(key in recomputed, f"{label}: no recomputed metric {key}")
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


def _window_table_equal(left: WindowTable, right: WindowTable) -> bool:
    return all(
        np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))
        for name in (
            "record_index",
            "start",
            "target_start",
            "target_end",
            "label",
            "fog_fraction",
            "clean_normal",
        )
    )


def validate_protocol(
    root: Path,
    *,
    allow_partial: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(root / "config.json")
    require(
        config.get("suite_version") == suite.SUITE_VERSION,
        "Unexpected suite_version",
    )
    require(
        canonical_fingerprint(protocol_payload(config))
        == config.get("protocol_fingerprint"),
        "config.json protocol fingerprint mismatch",
    )
    manifest = load_json(root / "run_manifest.json")
    require(
        manifest == {
            key: value
            for key, value in config.items()
            if key not in {
                "data_dir",
                "output_dir",
                "device",
                "resume",
                "num_workers",
            }
        },
        "run_manifest.json differs from the immutable config protocol",
    )

    require(
        tuple(config.get("folds_resolved", [])) == EXPECTED_SUBJECTS,
        "LOSO fold order is not the canonical eight subjects",
    )
    require(
        tuple(config.get("subjects", [])) == EXPECTED_SUBJECTS,
        "Post-exclusion subject list is not canonical",
    )
    require(
        set(config.get("excluded_subjects", [])) == EXPECTED_EXCLUDED,
        "Excluded subjects must be exactly S04 and S10",
    )
    require(
        int(config.get("sampling_rate_hz", -1)) == 64,
        "Sampling rate must be 64 Hz",
    )
    require(int(config.get("n_channels", -1)) == 9, "Expected nine IMU channels")
    require(
        tuple(config.get("channel_names", [])) == EXPECTED_CHANNELS,
        "Channel names/order differ from the canonical three-IMU order",
    )
    fixed_values = {
        "nbm": suite.NBM_NAME,
        "context_samples": 128,
        "support_horizon_samples": 128,
        "fixed_label_samples": 32,
        "stride_samples": 16,
        "history_name": suite.INPUT_NAME,
        "history_samples": 256,
        "history_shape": [9, 256],
        "protocol_scope": "strict_gru_horizon4_x_8_fold",
    }
    for key, expected in fixed_values.items():
        require(
            config.get(key) == expected,
            f"config/{key}: {config.get(key)!r} != {expected!r}",
        )
    require(bool(config.get("cache_residuals")), "Residual caching must be enabled")
    require(bool(config.get("deterministic")), "Deterministic training must be enabled")
    require(
        config.get("delta_pr_auc_reference") == "H050",
        "Paired PR-AUC reference must be H050",
    )
    require(
        int(config.get("expected_experiments", -1)) == 4,
        "Expected experiment count must be four",
    )
    require(
        int(config.get("expected_nbm_tasks", -1)) == EXPECTED_CLASSIFIER_CELLS,
        "Expected NBM task count must be 32",
    )
    require(
        int(config.get("expected_classifier_cells", -1))
        == EXPECTED_CLASSIFIER_CELLS,
        "Expected classifier cell count must be 32",
    )

    horizons = list(config.get("horizon_variants", []))
    require(len(horizons) == 4, "Exactly four horizon variants are required")
    for item, (horizon_id, expected) in zip(
        horizons,
        EXPECTED_HORIZONS.items(),
    ):
        require(item.get("horizon_id") == horizon_id, "Horizon order/ID mismatch")
        assert_close(
            item.get("horizon_seconds"),
            expected["seconds"],
            f"{horizon_id}/seconds",
            rtol=0.0,
            atol=1e-12,
        )
        require(
            int(item.get("horizon_samples", -1)) == expected["samples"],
            f"{horizon_id}: horizon sample count mismatch",
        )
        require(
            int(item.get("history_blocks", -1)) == expected["blocks"],
            f"{horizon_id}: history block count mismatch",
        )
        require(
            int(item["horizon_samples"]) * int(item["history_blocks"]) == 256,
            f"{horizon_id}: blocks do not cover exactly four seconds",
        )
    expected_experiments = suite.horizon_grid(
        [dict(item) for item in suite.HORIZON_DEFINITIONS]
    )
    require(
        list(config.get("experiments", [])) == expected_experiments,
        "Experiment grid differs from the preregistered four horizons",
    )

    classifier = config.get("classifier", {})
    require(isinstance(classifier, Mapping), "Missing classifier architecture")
    expected_classifier = {
        "name": "tcn_m",
        "in_channels": 9,
        "hidden_channels": int(config.get("classifier_hidden", 48)),
        "kernel_size": 3,
        "dilations": list(EXPECTED_DILATIONS),
        "n_blocks": 6,
        "convolutions_per_block": 2,
        "receptive_field_samples": 125,
        "global_pooling": "mean_and_max_over_full_input",
    }
    for key, expected in expected_classifier.items():
        require(
            classifier.get(key) == expected,
            f"classifier/{key}: {classifier.get(key)!r} != {expected!r}",
        )

    architectures = config.get("gru_architectures", {})
    require(
        isinstance(architectures, Mapping)
        and set(architectures) == set(EXPECTED_HORIZONS),
        "GRU architecture map does not cover all horizons",
    )
    shared_hashes: set[str] = set()
    shared_parameter_counts: set[int] = set()
    total_parameter_counts: list[int] = []
    for horizon_id, expected in EXPECTED_HORIZONS.items():
        architecture = architectures[horizon_id]
        model_config = architecture.get("model_config", {})
        require(model_config.get("name") == "gru", f"{horizon_id}: NBM is not GRU")
        require(
            int(model_config.get("in_channels", -1)) == 9,
            f"{horizon_id}: GRU channel count mismatch",
        )
        require(
            int(model_config.get("horizon", -1)) == expected["samples"],
            f"{horizon_id}: GRU direct-decoder horizon mismatch",
        )
        shared_hashes.add(
            str(architecture.get("initial_shared_encoder_summary_sha256"))
        )
        shared_parameter_counts.add(
            int(architecture.get("shared_encoder_summary_parameter_count", -1))
        )
        total_parameter_counts.append(int(architecture.get("parameter_count", -1)))
        require(
            int(architecture.get("decoder_parameter_count", -1)) > 0,
            f"{horizon_id}: decoder parameter count is invalid",
        )
    require(
        len(shared_hashes) == 1
        and next(iter(shared_hashes))
        == config.get("shared_initial_gru_encoder_summary_sha256"),
        "GRU encoder/summary initial state is not shared by all horizons",
    )
    require(
        len(shared_parameter_counts) == 1,
        "GRU encoder/summary parameter count varies by horizon",
    )
    require(
        total_parameter_counts == sorted(total_parameter_counts),
        "GRU direct-decoder parameter count must increase with horizon",
    )

    implementation = config.get("implementation", {})
    files = implementation.get("files", {}) if isinstance(implementation, Mapping) else {}
    require(
        isinstance(files, Mapping) and tuple(files) == tuple(suite.IMPLEMENTATION_FILES),
        "Implementation provenance file set mismatch",
    )
    require(
        canonical_fingerprint(files) == implementation.get("sha256"),
        "Implementation aggregate fingerprint mismatch",
    )
    for relative, expected_sha in files.items():
        source = REPO_ROOT / str(relative)
        if source.exists():
            require(
                sha256_file(source) == expected_sha,
                f"Implementation source drift detected: {relative}",
            )

    if not allow_partial:
        require(
            config.get("protocol_scope") == "strict_gru_horizon4_x_8_fold",
            "Canonical protocol required without --allow-partial",
        )
    return config, expected_experiments


def resolve_data_dir(config: Mapping[str, Any], fallback: Path | None) -> Path:
    candidates = [fallback, Path(str(config.get("data_dir", "")))]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    raise AuditError(
        "Processed Daphnet data cannot be found; pass --data-dir after moving results"
    )


def rebuild_dataset_and_windows(
    config: Mapping[str, Any],
    data_dir: Path,
) -> tuple[
    DaphnetDataset,
    WindowTable,
    dict[str, WindowTable],
    WindowTable,
]:
    require(
        dataset_fingerprint(data_dir) == config.get("data_sha256"),
        "Processed Daphnet data fingerprint differs from config.json",
    )
    source = DaphnetDataset.load(
        data_dir,
        flatline_seconds=float(config["flatline_seconds"]),
        zero_tolerance=float(config["zero_tolerance"]),
    )
    require(source.n_channels == 9, "Loaded dataset is not nine-channel")
    require(
        tuple(source.channel_names) == EXPECTED_CHANNELS,
        "Loaded channel names/order differ from protocol",
    )
    require(
        list(source.subjects) == list(config["source_subjects"]),
        "Source subject list differs from config.json",
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
        "Filtered dataset does not contain the canonical eight subjects",
    )
    raw_master = filtered.make_windows(
        warmup_samples=128,
        target_samples=128,
        stride_samples=16,
        fog_fraction_threshold=float(config["fog_fraction_threshold"]),
        normal_guard_samples=int(config["normal_guard_samples"]),
    )
    master = suite.relabel_master_windows(
        filtered,
        raw_master,
        fixed_label_samples=32,
        fog_fraction_threshold=float(config["fog_fraction_threshold"]),
    )
    windows_by_horizon = {
        horizon_id: suite.derive_horizon_windows(master, definition["samples"])
        for horizon_id, definition in EXPECTED_HORIZONS.items()
    }
    classification = suite.derive_classification_windows(master)
    require(
        len(master) == int(config["master_window_count"]),
        "Rebuilt master WindowTable count mismatch",
    )
    require(
        suite.window_table_sha256(master) == config["master_window_sha256"],
        "Rebuilt master WindowTable hash mismatch",
    )
    for horizon_id, windows in windows_by_horizon.items():
        require(
            suite.window_table_sha256(windows)
            == config["derived_window_sha256"][horizon_id],
            f"{horizon_id}: derived WindowTable hash mismatch",
        )
        require(
            np.array_equal(windows.label, classification.label),
            f"{horizon_id}: labels are not fixed endpoint labels",
        )
        require(
            np.array_equal(windows.clean_normal, master.clean_normal),
            f"{horizon_id}: clean-normal support differs from max horizon",
        )
    require(
        suite.window_table_sha256(classification)
        == config["classification_window_sha256"],
        "Classification WindowTable hash mismatch",
    )
    require(
        np.array_equal(
            np.bincount(classification.label, minlength=2),
            np.asarray(config["fixed_label_class_counts"], dtype=np.int64),
        ),
        "Fixed-label class counts mismatch",
    )
    require(
        int(master.clean_normal.sum())
        == int(config["master_clean_normal_windows"]),
        "Master clean-normal count mismatch",
    )
    return filtered, master, windows_by_horizon, classification


def recompute_common_support(
    windows_by_horizon: Mapping[str, WindowTable],
    split_indices: Mapping[str, np.ndarray],
    *,
    max_classifier_windows: int,
    seed: int,
    fold_index: int,
    labels: np.ndarray,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Independently rebuild common anchors and horizon-specific chains."""

    raw: dict[str, dict[str, Any]] = {}
    for horizon_id, definition in EXPECTED_HORIZONS.items():
        raw[horizon_id] = {
            split: make_common_history_plan(
                windows_by_horizon[horizon_id],
                indices,
                int(definition["samples"]),
                16,
                256,
            )
            for split, indices in split_indices.items()
        }

    result: dict[str, dict[str, dict[str, np.ndarray]]] = {
        horizon_id: {} for horizon_id in EXPECTED_HORIZONS
    }
    horizon_ids = list(EXPECTED_HORIZONS)
    for split in EXPECTED_SPLITS:
        common = set(raw[horizon_ids[0]][split].anchor_window_indices.tolist())
        for horizon_id in horizon_ids[1:]:
            common &= set(raw[horizon_id][split].anchor_window_indices.tolist())
        ordered = np.asarray(
            [
                int(value)
                for value in raw[horizon_ids[0]][split].anchor_window_indices
                if int(value) in common
            ],
            dtype=np.int64,
        )
        require(len(ordered) > 0, f"{split}: empty common horizon support")
        selected_rows: np.ndarray | None = None
        if split == "train" and int(max_classifier_windows) > 0:
            candidate_rows = np.arange(len(ordered), dtype=np.int64)
            selected_rows = deterministic_subsample(
                candidate_rows,
                int(max_classifier_windows),
                int(seed) + 100 + int(fold_index),
                np.asarray(labels, dtype=np.int8)[ordered],
            )
            ordered = ordered[selected_rows]
        for horizon_id in horizon_ids:
            plan = raw[horizon_id][split]
            lookup = {
                int(window_id): int(row)
                for row, window_id in enumerate(plan.anchor_window_indices)
            }
            rows = np.asarray(
                [lookup[int(window_id)] for window_id in ordered],
                dtype=np.int64,
            )
            chain = np.asarray(split_indices[split], dtype=np.int64)[
                plan.max_chain_rows[rows]
            ]
            result[horizon_id][split] = {
                "anchor": ordered.copy(),
                "chain": chain,
            }
    return result


def validate_history_geometry(
    windows: WindowTable,
    *,
    anchor: np.ndarray,
    chain: np.ndarray,
    horizon_samples: int,
    history_blocks: int,
    label: str,
) -> None:
    """Validate right alignment and exactly four seconds of non-overlap."""

    anchor = np.asarray(anchor, dtype=np.int64)
    chain = np.asarray(chain, dtype=np.int64)
    require(
        chain.shape == (len(anchor), int(history_blocks)),
        f"{label}: history support shape {chain.shape} != "
        f"({len(anchor)}, {history_blocks})",
    )
    require(
        np.array_equal(chain[:, -1], anchor),
        f"{label}: final residual block is not the classifier anchor",
    )
    if not len(anchor):
        return
    records = windows.record_index[chain]
    require(
        np.all(records == records[:, :1]),
        f"{label}: a history crosses a recording boundary",
    )
    starts = windows.target_start[chain].astype(np.int64)
    ends = windows.target_end[chain].astype(np.int64)
    require(
        np.all(ends - starts == int(horizon_samples)),
        f"{label}: at least one residual block has the wrong horizon",
    )
    if int(history_blocks) > 1:
        require(
            np.all(np.diff(starts, axis=1) == int(horizon_samples)),
            f"{label}: residual blocks are not horizon-spaced",
        )
        require(
            np.all(starts[:, 1:] == ends[:, :-1]),
            f"{label}: residual blocks overlap or leave a temporal gap",
        )
    require(
        np.all(ends[:, -1] == windows.target_end[anchor]),
        f"{label}: history is not right-aligned to the decision endpoint",
    )
    require(
        np.all(ends[:, -1] - starts[:, 0] == 256),
        f"{label}: residual history does not cover exactly 256 samples",
    )


def _expected_validation_subject(
    dataset: DaphnetDataset,
    classification_windows: WindowTable,
    test_subject: str,
) -> str:
    subjects = list(dataset.subjects)
    start = subjects.index(test_subject)
    for offset in range(1, len(subjects)):
        candidate = subjects[(start + offset) % len(subjects)]
        indices = dataset.window_indices_for_subjects(
            classification_windows,
            [candidate],
        )
        if np.unique(classification_windows.label[indices]).size == 2:
            return candidate
    raise AuditError(f"{test_subject}: no eligible validation subject")


def validate_fold(
    root: Path,
    subject: str,
    config: Mapping[str, Any],
    dataset: DaphnetDataset,
    master: WindowTable,
    windows_by_horizon: Mapping[str, WindowTable],
    classification_windows: WindowTable,
) -> dict[str, Any]:
    fold_root = root / f"loso_{subject}"
    require(fold_root.is_dir(), f"{subject}: fold directory is missing")
    fold_config_path = fold_root / "fold_config.json"
    scaler_path = fold_root / "scaler.json"
    split_path = fold_root / "split_indices.npz"
    support_path = fold_root / "common_history_support.npz"
    fold_config = load_json(fold_config_path)
    require(
        fold_config.get("suite_version") == suite.SUITE_VERSION,
        f"{subject}: fold suite version mismatch",
    )
    require(
        fold_config.get("protocol_fingerprint")
        == config["protocol_fingerprint"],
        f"{subject}: fold protocol fingerprint mismatch",
    )
    require(
        fold_config.get("test_subject") == subject,
        f"{subject}: fold test subject mismatch",
    )
    expected_val = _expected_validation_subject(
        dataset,
        classification_windows,
        subject,
    )
    expected_train = [
        candidate
        for candidate in dataset.subjects
        if candidate not in {subject, expected_val}
    ]
    require(
        fold_config.get("val_subject") == expected_val,
        f"{subject}: validation subject differs from deterministic selection",
    )
    require(
        fold_config.get("train_subjects") == expected_train,
        f"{subject}: training subjects differ from exact LOSO split",
    )
    require(
        set(fold_config.get("excluded_subjects", [])) == EXPECTED_EXCLUDED,
        f"{subject}: excluded-subject list mismatch",
    )
    require(
        len(expected_train) == 6
        and set(expected_train).isdisjoint({subject, expected_val}),
        f"{subject}: train/validation/test leakage",
    )

    scaler_payload = load_json(scaler_path)
    require(
        scaler_payload == fold_config.get("scaler"),
        f"{subject}: scaler.json differs from fold_config",
    )
    require(
        sha256_file(scaler_path) == fold_config.get("scaler_sha256"),
        f"{subject}: scaler hash mismatch",
    )
    recomputed_scaler = dataset.fit_scaler(
        expected_train,
        clip=float(config["robust_clip"]),
    )
    saved_center = np.asarray(scaler_payload["center"], dtype=np.float32)
    saved_scale = np.asarray(scaler_payload["scale"], dtype=np.float32)
    require(saved_center.shape == (9,), f"{subject}: scaler center is not 9D")
    require(saved_scale.shape == (9,), f"{subject}: scaler scale is not 9D")
    require(
        np.array_equal(saved_center, recomputed_scaler.center),
        f"{subject}: scaler center differs from train-only recomputation",
    )
    require(
        np.array_equal(saved_scale, recomputed_scaler.scale),
        f"{subject}: scaler scale differs from train-only recomputation",
    )
    assert_close(
        scaler_payload["clip"],
        recomputed_scaler.clip,
        f"{subject}/scaler_clip",
        rtol=0.0,
        atol=0.0,
    )

    require(split_path.exists(), f"{subject}: split_indices.npz is missing")
    require(
        sha256_file(split_path) == fold_config.get("split_indices_sha256"),
        f"{subject}: split index hash mismatch",
    )
    with np.load(split_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == EXPECTED_SPLIT_KEYS,
            f"{subject}: split_indices.npz key set mismatch",
        )
        split_indices = {
            "train": check_indices(
                payload["train_window_index"],
                len(master),
                f"{subject}/train",
            ),
            "validation": check_indices(
                payload["validation_window_index"],
                len(master),
                f"{subject}/validation",
            ),
            "test": check_indices(
                payload["test_window_index"],
                len(master),
                f"{subject}/test",
            ),
        }
        normal_train = check_indices(
            payload["normal_train_window_index"],
            len(master),
            f"{subject}/normal_train",
        )
        normal_validation = check_indices(
            payload["normal_validation_window_index"],
            len(master),
            f"{subject}/normal_validation",
        )

    expected_splits = {
        "train": dataset.window_indices_for_subjects(
            classification_windows,
            expected_train,
        ),
        "validation": dataset.window_indices_for_subjects(
            classification_windows,
            [expected_val],
        ),
        "test": dataset.window_indices_for_subjects(
            classification_windows,
            [subject],
        ),
    }
    for split, expected_indices in expected_splits.items():
        require(
            np.array_equal(split_indices[split], expected_indices),
            f"{subject}/{split}: split indices differ from exact LOSO rebuild",
        )
        require(
            int(fold_config["source_window_counts"][split])
            == len(expected_indices),
            f"{subject}/{split}: source window count mismatch",
        )
    require(
        subjects_for_windows(dataset, master, split_indices["train"])
        == set(expected_train),
        f"{subject}: training indices belong to the wrong subjects",
    )
    require(
        subjects_for_windows(dataset, master, split_indices["validation"])
        == {expected_val},
        f"{subject}: validation indices belong to the wrong subject",
    )
    require(
        subjects_for_windows(dataset, master, split_indices["test"]) == {subject},
        f"{subject}: test indices belong to the wrong subject",
    )

    fold_index = list(dataset.subjects).index(subject)
    expected_normal_train = dataset.window_indices_for_subjects(
        master,
        expected_train,
        clean_normal_only=True,
    )
    expected_normal_train = deterministic_subsample(
        expected_normal_train,
        int(config["max_normal_windows"]),
        int(config["seed"]) + fold_index,
    )
    expected_normal_validation = dataset.window_indices_for_subjects(
        master,
        [expected_val],
        clean_normal_only=True,
    )
    require(
        np.array_equal(normal_train, expected_normal_train),
        f"{subject}: normal training support differs from max-horizon rebuild",
    )
    require(
        np.array_equal(normal_validation, expected_normal_validation),
        f"{subject}: normal validation support differs from max-horizon rebuild",
    )
    require(
        np.all(master.clean_normal[normal_train])
        and np.all(master.label[normal_train] == 0),
        f"{subject}: normal training support contains FoG/guard violations",
    )
    require(
        np.all(master.clean_normal[normal_validation])
        and np.all(master.label[normal_validation] == 0),
        f"{subject}: normal validation support contains FoG/guard violations",
    )
    require(
        suite.array_sha256(normal_train)
        == fold_config["normal_train_window_indices_sha256"],
        f"{subject}: normal training support hash mismatch",
    )
    require(
        suite.array_sha256(normal_validation)
        == fold_config["normal_validation_window_indices_sha256"],
        f"{subject}: normal validation support hash mismatch",
    )
    require(
        int(fold_config["normal_train_windows"]) == len(normal_train)
        and int(fold_config["normal_validation_windows"])
        == len(normal_validation),
        f"{subject}: normal support count mismatch",
    )

    expected_support = recompute_common_support(
        windows_by_horizon,
        split_indices,
        max_classifier_windows=int(config["max_classifier_windows"]),
        seed=int(config["seed"]),
        fold_index=fold_index,
        labels=classification_windows.label,
    )
    require(
        support_path.exists(),
        f"{subject}: common_history_support.npz is missing",
    )
    support_sha = sha256_file(support_path)
    require(
        support_sha == fold_config["common_history_support_sha256"],
        f"{subject}: common support hash mismatch",
    )
    saved_support: dict[str, dict[str, dict[str, np.ndarray]]] = {
        horizon_id: {} for horizon_id in EXPECTED_HORIZONS
    }
    with np.load(support_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == EXPECTED_SUPPORT_KEYS,
            f"{subject}: common support key set mismatch",
        )
        reference_anchor: dict[str, np.ndarray] = {}
        for split in EXPECTED_SPLITS:
            anchor = check_indices(
                payload[f"{split}_anchor_window_index"],
                len(master),
                f"{subject}/{split}/common_anchor",
            )
            y = np.asarray(payload[f"{split}_y"], dtype=np.int8)
            require(
                y.shape == (len(anchor),),
                f"{subject}/{split}: common label shape mismatch",
            )
            require(
                np.array_equal(y, classification_windows.label[anchor]),
                f"{subject}/{split}: support labels are not final-0.5-s labels",
            )
            require(
                np.unique(y).size == 2,
                f"{subject}/{split}: common support lacks one class",
            )
            require(
                np.array_equal(
                    anchor,
                    expected_support["H025"][split]["anchor"],
                ),
                f"{subject}/{split}: common anchors differ from exact rebuild",
            )
            require(
                int(fold_config["common_anchor_counts"][split]) == len(anchor),
                f"{subject}/{split}: common anchor count mismatch",
            )
            require(
                suite.array_sha256(anchor)
                == fold_config["common_anchor_sha256"][split],
                f"{subject}/{split}: common anchor hash mismatch",
            )
            reference_anchor[split] = anchor
            for horizon_id, definition in EXPECTED_HORIZONS.items():
                chain = np.asarray(
                    payload[
                        f"{split}_{horizon_id.lower()}_history_window_index"
                    ],
                    dtype=np.int64,
                )
                require(
                    np.array_equal(
                        chain,
                        expected_support[horizon_id][split]["chain"],
                    ),
                    f"{subject}/{split}/{horizon_id}: chain differs from "
                    "independent common-support rebuild",
                )
                require(
                    suite.array_sha256(chain)
                    == fold_config["per_horizon_history_support_sha256"][
                        horizon_id
                    ][split],
                    f"{subject}/{split}/{horizon_id}: chain hash mismatch",
                )
                validate_history_geometry(
                    windows_by_horizon[horizon_id],
                    anchor=anchor,
                    chain=chain,
                    horizon_samples=int(definition["samples"]),
                    history_blocks=int(definition["blocks"]),
                    label=f"{subject}/{split}/{horizon_id}",
                )
                saved_support[horizon_id][split] = {
                    "anchor": anchor.copy(),
                    "chain": chain,
                    "y": y.copy(),
                }

    require(
        fold_config.get("classification_window_sha256")
        == config["classification_window_sha256"],
        f"{subject}: classification WindowTable provenance mismatch",
    )
    require(
        int(fold_config.get("label_window_samples", -1)) == 32,
        f"{subject}: fold label window is not fixed at 0.5 seconds",
    )
    fold_architectures = fold_config.get("per_horizon_gru_architecture", {})
    require(
        isinstance(fold_architectures, Mapping)
        and set(fold_architectures) == set(EXPECTED_HORIZONS),
        f"{subject}: fold GRU architecture map is incomplete",
    )
    fold_encoder_hashes = {
        str(payload["initial_shared_encoder_summary_sha256"])
        for payload in fold_architectures.values()
    }
    require(
        fold_encoder_hashes
        == {str(fold_config["initial_shared_gru_encoder_summary_sha256"])},
        f"{subject}: initial GRU encoder/summary differs by horizon",
    )

    # Recompute the shared TCN-M initialization rather than trusting metadata.
    classifier_seed = int(config["seed"]) + 10000 + fold_index
    suite.core.set_seed(classifier_seed, bool(config["deterministic"]))
    reference_model = rf.build_model(
        in_channels=9,
        hidden_channels=int(config["classifier"]["hidden_channels"]),
        dropout=float(config["classifier"]["dropout"]),
        dilations=EXPECTED_DILATIONS,
    )
    classifier_initial_sha = rf.state_dict_sha256(reference_model.state_dict())
    del reference_model

    return {
        "root": fold_root,
        "config": fold_config,
        "validation_subject": expected_val,
        "training_subjects": expected_train,
        "scaler": recomputed_scaler,
        "splits": split_indices,
        "normal_train": normal_train,
        "normal_validation": normal_validation,
        "support": saved_support,
        "support_path": support_path,
        "support_sha256": support_sha,
        "classifier_seed": classifier_seed,
        "classifier_initial_sha256": classifier_initial_sha,
    }


def build_gru_from_protocol(
    config: Mapping[str, Any],
    horizon_samples: int,
) -> NormalBehaviourModel:
    return build_nbm(
        "gru",
        in_channels=9,
        horizon=int(horizon_samples),
        hidden_channels=int(config["nbm_hidden"]),
        dropout=float(config["nbm_dropout"]),
        gru_layers=int(config["gru_layers"]),
    )


def _shared_gru_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("encoder.") or name.startswith("summary.")
    }


def validate_resume_checkpoint(
    payload: Mapping[str, Any],
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    source_residual_sha256: str | None = None,
) -> None:
    if stage == "rf_classifier":
        require(
            source_residual_sha256 is not None,
            f"{task_id}: classifier source hash is missing",
        )
        rf.validate_rf_checkpoint(
            payload,
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            source_residual_sha256=str(source_residual_sha256),
        )
    else:
        validate_checkpoint(
            payload,
            stage=stage,
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
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
    require(epoch >= 1, f"{task_id}: invalid last epoch")
    require(0 <= best_epoch <= epoch, f"{task_id}: invalid best epoch")
    history = list(payload["history"])
    require(history, f"{task_id}: empty training history")
    require(
        int(history[-1]["epoch"]) == epoch,
        f"{task_id}: history does not end at last checkpoint epoch",
    )


def validate_nbm_task(
    root: Path,
    subject: str,
    horizon: Mapping[str, Any],
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
) -> tuple[str, NormalBehaviourModel, dict[str, Any]]:
    horizon_id = str(horizon["horizon_id"])
    horizon_samples = int(horizon["horizon_samples"])
    nbm_root = suite.nbm_root_for(root, subject, horizon)
    stage_root = nbm_root / "nbm"
    best_path = stage_root / "best.pt"
    last_path = stage_root / "last.pt"
    training_path = stage_root / "training.json"
    done_path = stage_root / "DONE.json"
    task_id = suite._nbm_task_id(horizon)
    done = validate_done(
        done_path,
        stage="nbm",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(done is not None, f"{subject}/{horizon_id}: missing NBM DONE")
    assert_done_artifacts(
        done,
        {
            "best": best_path,
            "last": last_path,
            "training": training_path,
        },
        done_path,
        f"{subject}/{horizon_id}/nbm",
    )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        best,
        stage="nbm",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    expected_seed = int(config["seed"]) + EXPECTED_SUBJECTS.index(subject)
    require(best.get("model_name") == "gru", f"{task_id}: model is not GRU")
    require(int(best.get("seed", -1)) == expected_seed, f"{task_id}: seed mismatch")
    model = build_gru_from_protocol(config, horizon_samples).cpu().eval()
    require(
        best.get("model_config") == model.model_config(),
        f"{task_id}: best checkpoint model config differs from protocol",
    )
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        mean, sigma = model(torch.zeros(2, 9, 128, dtype=torch.float32))
    require(
        tuple(mean.shape) == (2, 9, horizon_samples)
        and tuple(sigma.shape) == (2, 9, horizon_samples),
        f"{task_id}: GRU output shape mismatch",
    )
    require(
        torch.isfinite(mean).all().item()
        and torch.isfinite(sigma).all().item()
        and torch.all(sigma > 0).item(),
        f"{task_id}: GRU output distribution is invalid",
    )

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        last,
        stage="nbm",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(last.get("model_name") == "gru", f"{task_id}: last model is not GRU")
    require(int(last.get("seed", -1)) == expected_seed, f"{task_id}: last seed mismatch")
    require(
        last.get("model_config") == best.get("model_config"),
        f"{task_id}: best/last model configs differ",
    )
    model.load_state_dict(last["model_state"], strict=True)

    training = load_json(training_path)
    require(training.get("model_name") == "gru", f"{task_id}: training model mismatch")
    require(int(training.get("seed", -1)) == expected_seed, f"{task_id}: training seed mismatch")
    require(
        training.get("model_config") == best.get("model_config"),
        f"{task_id}: training/checkpoint model configs differ",
    )
    require(
        int(training.get("parameter_count", -1))
        == int(config["gru_architectures"][horizon_id]["parameter_count"]),
        f"{task_id}: GRU parameter count mismatch",
    )
    require(
        int(training.get("train_windows", -1)) == len(fold["normal_train"])
        and int(training.get("validation_windows", -1))
        == len(fold["normal_validation"]),
        f"{task_id}: NBM clean-normal support count mismatch",
    )
    require(
        int(training.get("best_epoch", -1)) == int(best["best_epoch"]),
        f"{task_id}: best epoch mismatch",
    )
    assert_close(
        training["best_val_nll"],
        best["best_val_nll"],
        f"{task_id}/best_val_nll",
    )

    # Recreate the horizon-shared initial encoder/summary identity.
    suite.core.set_seed(expected_seed, bool(config["deterministic"]))
    initial_model = build_gru_from_protocol(config, horizon_samples)
    initial_shared_hash = rf.state_dict_sha256(_shared_gru_state(initial_model))
    del initial_model
    require(
        initial_shared_hash
        == fold["config"]["initial_shared_gru_encoder_summary_sha256"],
        f"{task_id}: initial shared GRU encoder hash mismatch",
    )
    require(
        initial_shared_hash
        == fold["config"]["per_horizon_gru_architecture"][horizon_id][
            "initial_shared_encoder_summary_sha256"
        ],
        f"{task_id}: per-horizon initial encoder provenance mismatch",
    )

    summary_path = nbm_root / "nbm_summary.json"
    summary = load_json(summary_path)
    expected_summary = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "experiment_id": suite.experiment_id(horizon),
        "horizon_id": horizon_id,
        "horizon_samples": horizon_samples,
        "context_samples": 128,
        "history_samples": 256,
        "history_blocks": int(horizon["history_blocks"]),
        "fixed_label_samples": 32,
        "master_clean_normal_support": True,
        "derived_window_sha256": config["derived_window_sha256"][horizon_id],
        "nbm_sha256": sha256_file(best_path),
        "gru_architecture": config["gru_architectures"][horizon_id],
        "normal_training": training,
    }
    for key, expected in expected_summary.items():
        require(
            summary.get(key) == expected,
            f"{task_id}/nbm_summary/{key}: value mismatch",
        )

    # Residuals must be generated by the validation-selected best checkpoint.
    model.load_state_dict(best["model_state"], strict=True)
    model.eval()
    return sha256_file(best_path), model, summary


def validate_residual_cache(
    root: Path,
    subject: str,
    horizon: Mapping[str, Any],
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    nbm_sha256: str,
    model: NormalBehaviourModel,
    *,
    tolerance: float,
) -> tuple[str, dict[str, dict[str, np.ndarray]]]:
    horizon_id = str(horizon["horizon_id"])
    horizon_samples = int(horizon["horizon_samples"])
    nbm_root = suite.nbm_root_for(root, subject, horizon)
    cache_path = nbm_root / "residual_cache.npz"
    diagnostics_path = nbm_root / "residual_diagnostics.json"
    done_path = nbm_root / "RESIDUAL_CACHE_DONE.json"
    task_id = suite._residual_task_id(horizon)
    done = validate_done(
        done_path,
        stage="residual_cache",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    require(done is not None, f"{subject}/{horizon_id}: missing residual DONE")
    assert_done_artifacts(
        done,
        {"cache": cache_path, "diagnostics": diagnostics_path},
        done_path,
        f"{subject}/{horizon_id}/residual",
    )

    features: dict[str, dict[str, np.ndarray]] = {}
    with np.load(cache_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == EXPECTED_RESIDUAL_KEYS,
            f"{task_id}: residual cache key set mismatch",
        )
        for split in EXPECTED_SPLITS:
            values = {
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
            expected_indices = fold["splits"][split]
            require(
                np.array_equal(values["window_index"], expected_indices),
                f"{task_id}/{split}: cache indices differ from LOSO split",
            )
            require(
                np.array_equal(values["y"], windows.label[expected_indices]),
                f"{task_id}/{split}: cache labels differ from fixed endpoint labels",
            )
            require(
                values["residual"].shape
                == (len(expected_indices), 9, horizon_samples),
                f"{task_id}/{split}: residual shape "
                f"{values['residual'].shape} != "
                f"({len(expected_indices)}, 9, {horizon_samples})",
            )
            require(
                np.isfinite(values["residual"]).all(),
                f"{task_id}/{split}: non-finite residual values",
            )
            require(
                np.all(
                    np.abs(values["residual"])
                    <= float(config["residual_clip"]) + 1e-6
                ),
                f"{task_id}/{split}: residual exceeds clipping range",
            )
            features[split] = values

    model = model.cpu().eval()
    residual_clip = float(config["residual_clip"])
    for split in EXPECTED_SPLITS:
        values = features[split]
        require(len(values["window_index"]) > 0, f"{task_id}/{split}: empty cache")
        rows = np.unique(
            np.linspace(
                0,
                len(values["window_index"]) - 1,
                num=min(8, len(values["window_index"])),
                dtype=np.int64,
            )
        )
        sequences: list[np.ndarray] = []
        for row in rows:
            window_index = int(values["window_index"][row])
            record_index = int(windows.record_index[window_index])
            start = int(windows.start[window_index])
            end = int(windows.target_end[window_index])
            scaled = fold["scaler"].transform(
                dataset.records[record_index].x[start:end]
            )
            sequences.append(np.ascontiguousarray(scaled.T, dtype=np.float32))
        sequence = torch.from_numpy(np.stack(sequences)).float()
        context = sequence[:, :, :128]
        target = sequence[:, :, 128:]
        require(
            tuple(target.shape[1:]) == (9, horizon_samples),
            f"{task_id}/{split}: rebuilt target shape mismatch",
        )
        with torch.no_grad():
            mean, sigma = model(context)
            replayed = ((target - mean) / sigma).clamp(
                -residual_clip,
                residual_clip,
            )
        saved = values["residual"][rows]
        replayed_array = replayed.float().cpu().numpy()
        require(
            np.allclose(
                saved,
                replayed_array,
                rtol=float(tolerance),
                atol=float(tolerance),
            ),
            f"{task_id}/{split}: sampled residual replay mismatch "
            f"(max_abs={float(np.max(np.abs(saved - replayed_array))):.6g})",
        )

    diagnostics = load_json(diagnostics_path)
    require(
        set(diagnostics) == set(EXPECTED_SPLITS),
        f"{task_id}: residual diagnostics split set mismatch",
    )
    for split in EXPECTED_SPLITS:
        payload = diagnostics[split]
        require(isinstance(payload, Mapping), f"{task_id}/{split}: bad diagnostics")
        require(
            int(payload["windows"]) == len(fold["splits"][split]),
            f"{task_id}/{split}: diagnostic window count mismatch",
        )
        require(
            list(payload["class_counts"])
            == np.bincount(
                windows.label[fold["splits"][split]],
                minlength=2,
            ).astype(int).tolist(),
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
                math.isfinite(float(payload[key])),
                f"{task_id}/{split}: non-finite diagnostic {key}",
            )
        require(
            float(payload["mean_sigma"]) > 0.0,
            f"{task_id}/{split}: mean sigma is not positive",
        )
    return sha256_file(cache_path), features


def materialize_history(
    features: Mapping[str, np.ndarray],
    chain: np.ndarray,
    *,
    horizon_samples: int,
    history_blocks: int,
    label: str,
) -> np.ndarray:
    """Rebuild one arm's exact TCN input from residual-cache rows."""

    source_indices = np.asarray(features["window_index"], dtype=np.int64)
    require(
        source_indices.ndim == 1
        and len(source_indices) == len(np.unique(source_indices)),
        f"{label}: residual cache contains duplicate/non-1D IDs",
    )
    history = np.asarray(chain, dtype=np.int64)
    require(
        history.ndim == 2 and history.shape[1] == int(history_blocks),
        f"{label}: history index shape mismatch",
    )
    order = np.argsort(source_indices)
    sorted_indices = source_indices[order]
    positions = np.searchsorted(sorted_indices, history)
    require(
        np.all(positions < len(sorted_indices)),
        f"{label}: support references an absent residual-cache row",
    )
    require(
        np.array_equal(sorted_indices[positions], history),
        f"{label}: history/cache row mapping mismatch",
    )
    blocks = np.asarray(features["residual"], dtype=np.float32)[order[positions]]
    require(
        blocks.shape[1:] == (int(history_blocks), 9, int(horizon_samples)),
        f"{label}: gathered residual block tensor shape mismatch",
    )
    result = blocks.transpose(0, 2, 1, 3).reshape(len(history), 9, -1)
    require(
        result.shape == (len(history), 9, 256),
        f"{label}: classifier input is {result.shape}, expected [B,9,256]",
    )
    require(np.isfinite(result).all(), f"{label}: classifier input is non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def load_predictions(path: Path, label: str) -> dict[str, np.ndarray]:
    require(path.exists(), f"{label}: prediction artifact is missing")
    with np.load(path, allow_pickle=False) as payload:
        require(
            set(payload.files) == EXPECTED_PREDICTION_KEYS,
            f"{label}: prediction array key set mismatch",
        )
        result = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    require(
        len({len(value) for value in result.values()}) == 1,
        f"{label}: prediction array lengths differ",
    )
    require(
        all(value.ndim == 1 for value in result.values()),
        f"{label}: prediction arrays are not one-dimensional",
    )
    require(
        len(result["window_index"]) == len(np.unique(result["window_index"])),
        f"{label}: duplicate prediction window IDs",
    )
    require(
        np.isin(result["y_true"], [0, 1]).all()
        and np.isin(result["y_pred"], [0, 1]).all(),
        f"{label}: predictions contain a non-binary label",
    )
    require(
        np.isfinite(result["y_prob"]).all()
        and np.all((result["y_prob"] >= 0.0) & (result["y_prob"] <= 1.0)),
        f"{label}: invalid predicted probabilities",
    )
    return result


@torch.no_grad()
def classifier_probabilities(
    model: ResidualTCNClassifier,
    inputs: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(inputs), int(batch_size)):
        x = torch.from_numpy(inputs[start : start + batch_size]).float()
        chunks.append(torch.sigmoid(model(x)).float().cpu().numpy())
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64, copy=False)


def requested_metrics(recomputed: Mapping[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = (
        int(recomputed[key]) for key in ("tn", "fp", "fn", "tp")
    )
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "macro_f1": 0.5 * (f1_nonfog + f1_fog),
        "roc_auc": recomputed["auroc"],
        "pr_auc": recomputed["auprc"],
        "fog_recall": recomputed["sensitivity"],
        "fog_f1": f1_fog,
    }


def expected_classifier_config(
    config: Mapping[str, Any],
    initial_state_sha256: str,
) -> dict[str, Any]:
    classifier = config["classifier"]
    return {
        "in_channels": 9,
        "hidden_channels": int(classifier["hidden_channels"]),
        "dropout": float(classifier["dropout"]),
        "kernel_size": 3,
        "dilations": list(EXPECTED_DILATIONS),
        "n_blocks": 6,
        "convolutions_per_block": 2,
        "receptive_field_samples": 125,
        "receptive_field_seconds": 125 / 64.0,
        "parameter_count": int(classifier["parameter_count"]),
        "initial_state_sha256": initial_state_sha256,
        "global_pooling": "mean_and_max_over_full_input",
    }


def audit_classifier_task(
    root: Path,
    subject: str,
    horizon: Mapping[str, Any],
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
    dataset: DaphnetDataset,
    classification_windows: WindowTable,
    residual_sha256: str,
    features: Mapping[str, Mapping[str, np.ndarray]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    horizon_id = str(horizon["horizon_id"])
    horizon_samples = int(horizon["horizon_samples"])
    history_blocks = int(horizon["history_blocks"])
    experiment = suite.experiment_id(horizon)
    task_root = suite.task_root_for(root, subject, horizon)
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    metrics_path = task_root / "metrics.json"
    prediction_path = task_root / "predictions.npz"
    validation_prediction_path = task_root / "validation_predictions.npz"
    predictions_csv_path = task_root / "predictions.csv"
    done_path = task_root / "DONE.json"
    task_id = f"{subject}/{experiment}"

    done = validate_done(
        done_path,
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(done is not None, f"{task_id}: missing classifier DONE")
    assert_done_artifacts(
        done,
        {
            "best": best_path,
            "last": last_path,
            "metrics": metrics_path,
            "predictions": prediction_path,
            "validation_predictions": validation_prediction_path,
            "predictions_csv": predictions_csv_path,
        },
        done_path,
        task_id,
    )
    require(
        done.get("source_residual_sha256") == residual_sha256,
        f"{task_id}: classifier DONE residual hash mismatch",
    )
    require(
        done.get("input_support_sha256") == fold["support_sha256"],
        f"{task_id}: classifier DONE common-support hash mismatch",
    )
    require(
        done.get("initial_state_sha256")
        == fold["classifier_initial_sha256"],
        f"{task_id}: classifier initial-state hash mismatch",
    )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    rf.validate_rf_checkpoint(
        best,
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        source_residual_sha256=residual_sha256,
    )
    expected_seed = int(fold["classifier_seed"])
    require(
        int(best.get("classifier_seed", -1)) == expected_seed,
        f"{task_id}: classifier seed mismatch",
    )
    require(best.get("variant") == experiment, f"{task_id}: checkpoint variant mismatch")
    expected_config = expected_classifier_config(
        config,
        fold["classifier_initial_sha256"],
    )
    require(
        best.get("classifier_config") == expected_config,
        f"{task_id}: best classifier config mismatch",
    )
    model = rf.build_model(
        in_channels=9,
        hidden_channels=int(config["classifier"]["hidden_channels"]),
        dropout=float(config["classifier"]["dropout"]),
        dilations=EXPECTED_DILATIONS,
    ).cpu().eval()
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        logits = model(torch.zeros(2, 9, 256, dtype=torch.float32))
    require(tuple(logits.shape) == (2,), f"{task_id}: classifier output shape mismatch")
    require(torch.isfinite(logits).all().item(), f"{task_id}: non-finite logits")

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        last,
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        source_residual_sha256=residual_sha256,
    )
    require(last.get("variant") == experiment, f"{task_id}: last variant mismatch")
    require(
        int(last.get("classifier_seed", -1)) == expected_seed,
        f"{task_id}: last classifier seed mismatch",
    )
    require(
        last.get("classifier_config") == expected_config,
        f"{task_id}: last classifier config mismatch",
    )
    for epoch_record in last["history"]:
        require(
            int(epoch_record["shuffle_seed"])
            == expected_seed + int(epoch_record["epoch"]),
            f"{task_id}: classifier epoch shuffle order differs from protocol",
        )

    metrics = load_json(metrics_path)
    expected_metric_identity = {
        "experiment_id": experiment,
        "variant": experiment,
        "nbm": "gru",
        "input": "residual_h4s",
        "history_seconds": 4.0,
        "history_samples": 256,
        "history_blocks": history_blocks,
        "test_subject": subject,
        "val_subject": fold["validation_subject"],
        "classifier_seed": expected_seed,
        "classifier_config": expected_config,
        "initial_state_sha256": fold["classifier_initial_sha256"],
        "source_residual_sha256": residual_sha256,
        "input_support_sha256": fold["support_sha256"],
    }
    for key, expected in expected_metric_identity.items():
        require(
            metrics.get(key) == expected,
            f"{task_id}/metrics/{key}: value mismatch",
        )
    train_y = fold["support"][horizon_id]["train"]["y"]
    counts = np.bincount(train_y, minlength=2).astype(int)
    expected_pos_weight = min(math.sqrt(counts[0] / counts[1]), 6.0)
    require(
        np.array_equal(
            np.asarray(metrics.get("train_counts"), dtype=np.int64),
            counts,
        ),
        f"{task_id}: classifier train class counts mismatch",
    )
    assert_close(
        metrics.get("pos_weight"),
        expected_pos_weight,
        f"{task_id}/pos_weight",
    )
    require(
        metrics.get("history") == last.get("history"),
        f"{task_id}: metrics and last-checkpoint histories differ",
    )
    require(
        int(metrics.get("best_epoch", -1)) == int(best["best_epoch"]),
        f"{task_id}: best epoch mismatch",
    )
    assert_close(
        metrics.get("best_validation_auprc"),
        best["best_validation_auprc"],
        f"{task_id}/best_validation_auprc",
    )

    predictions = {
        "validation": load_predictions(
            validation_prediction_path,
            f"{task_id}/validation",
        ),
        "test": load_predictions(prediction_path, f"{task_id}/test"),
    }
    reconstructed_inputs: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        support = fold["support"][horizon_id][split]
        prediction = predictions[split]
        require(
            np.array_equal(prediction["window_index"], support["anchor"]),
            f"{task_id}/{split}: prediction IDs differ from common anchors",
        )
        require(
            np.array_equal(prediction["y_true"], support["y"]),
            f"{task_id}/{split}: predictions do not use fixed 0.5-s labels",
        )
        reconstructed = materialize_history(
            features[split],
            support["chain"],
            horizon_samples=horizon_samples,
            history_blocks=history_blocks,
            label=f"{task_id}/{split}",
        )
        reconstructed_inputs[split] = reconstructed
        replayed = classifier_probabilities(model, reconstructed)
        require(
            np.allclose(
                replayed,
                prediction["y_prob"],
                rtol=float(tolerance),
                atol=float(tolerance),
            ),
            f"{task_id}/{split}: classifier probability replay mismatch "
            f"(max_abs={float(np.max(np.abs(replayed - prediction['y_prob']))):.6g})",
        )

    threshold = float(metrics["threshold"])
    require(0.0 <= threshold <= 1.0, f"{task_id}: invalid threshold")
    for split, prediction in predictions.items():
        require(
            np.array_equal(
                prediction["y_pred"],
                (prediction["y_prob"] >= threshold).astype(np.int8),
            ),
            f"{task_id}/{split}: threshold decisions mismatch",
        )
    selected_threshold, selected_validation = choose_threshold(
        predictions["validation"]["y_true"],
        predictions["validation"]["y_prob"],
    )
    assert_close(
        threshold,
        selected_threshold,
        f"{task_id}/selected_threshold",
        rtol=0.0,
        atol=1e-12,
    )
    require(
        isinstance(metrics.get("validation"), Mapping),
        f"{task_id}: validation metrics are missing",
    )
    assert_metric_dict(
        metrics["validation"],
        selected_validation,
        CORE_BINARY_METRICS,
        f"{task_id}/validation",
    )
    recomputed = binary_metrics(
        predictions["test"]["y_true"],
        predictions["test"]["y_prob"],
        threshold,
    )
    assert_metric_dict(metrics, recomputed, CORE_BINARY_METRICS, f"{task_id}/test")
    for key, expected in requested_metrics(recomputed).items():
        assert_close(metrics.get(key), expected, f"{task_id}/{key}")
    recomputed_events = event_metrics(
        dataset,
        classification_windows,
        predictions["test"]["window_index"],
        predictions["test"]["y_pred"],
    )
    for key in EVENT_METRIC_KEYS:
        require(key in metrics, f"{task_id}: missing event metric {key}")
        expected = recomputed_events[key]
        if isinstance(expected, (int, np.integer)):
            require(
                int(metrics[key]) == int(expected),
                f"{task_id}/{key}: event count mismatch",
            )
        else:
            assert_close(metrics[key], expected, f"{task_id}/{key}")

    # The CSV is already bound by its DONE hash; check that it represents the
    # same fixed endpoint rows rather than another horizon-dependent interval.
    prediction_rows = read_csv(predictions_csv_path)
    require(
        len(prediction_rows) == len(predictions["test"]["window_index"]),
        f"{task_id}: predictions.csv row count mismatch",
    )
    if prediction_rows:
        ids = predictions["test"]["window_index"]
        csv_y = np.asarray(
            [int(row["y_true"]) for row in prediction_rows],
            dtype=np.int8,
        )
        csv_prob = np.asarray(
            [float(row["y_prob"]) for row in prediction_rows],
            dtype=np.float64,
        )
        csv_pred = np.asarray(
            [int(row["y_pred"]) for row in prediction_rows],
            dtype=np.int8,
        )
        require(
            np.array_equal(csv_y, predictions["test"]["y_true"])
            and np.array_equal(csv_prob, predictions["test"]["y_prob"])
            and np.array_equal(csv_pred, predictions["test"]["y_pred"]),
            f"{task_id}: predictions.csv values differ from predictions.npz",
        )
        csv_target_start = np.asarray(
            [int(row["target_start"]) for row in prediction_rows],
            dtype=np.int64,
        )
        csv_target_end = np.asarray(
            [int(row["target_end_exclusive"]) for row in prediction_rows],
            dtype=np.int64,
        )
        require(
            np.array_equal(
                csv_target_start,
                classification_windows.target_start[ids],
            )
            and np.array_equal(
                csv_target_end,
                classification_windows.target_end[ids],
            ),
            f"{task_id}: predictions.csv does not use fixed 0.5-s intervals",
        )

    return {
        "subject": subject,
        "horizon_id": horizon_id,
        "experiment_id": experiment,
        "metrics": metrics,
        "predictions": predictions["test"],
        "validation_predictions": predictions["validation"],
        "input_sha256": {
            split: suite.array_sha256(reconstructed_inputs[split])
            for split in reconstructed_inputs
        },
        "initial_state_sha256": fold["classifier_initial_sha256"],
        "shuffle_seeds": [
            int(item["shuffle_seed"]) for item in metrics["history"]
        ],
    }


def _mapping_close(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key, value in expected.items():
        require(key in actual, f"{label}: missing key {key}")
        if isinstance(value, Mapping):
            require(
                isinstance(actual[key], Mapping),
                f"{label}/{key}: expected an object",
            )
            _mapping_close(actual[key], value, f"{label}/{key}")
        elif isinstance(value, (int, float, np.integer, np.floating)) or value is None:
            assert_close(actual[key], value, f"{label}/{key}")
        else:
            require(
                actual[key] == value,
                f"{label}/{key}: {actual[key]!r} != {value!r}",
            )


def recompute_paired_deltas(
    config: Mapping[str, Any],
    experiments: list[dict[str, Any]],
    by_experiment: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    reference = next(
        item for item in experiments if str(item["horizon_id"]) == "H050"
    )
    reference_id = str(reference["experiment_id"])
    reference_by_subject = {
        str(cell["subject"]): cell for cell in by_experiment[reference_id]
    }
    result: dict[str, dict[str, Any]] = {}
    for horizon in experiments:
        experiment_id = str(horizon["experiment_id"])
        current_by_subject = {
            str(cell["subject"]): cell for cell in by_experiment[experiment_id]
        }
        common_subjects = [
            subject
            for subject in EXPECTED_SUBJECTS
            if subject in current_by_subject and subject in reference_by_subject
        ]
        differences = np.asarray(
            [
                float(current_by_subject[subject]["metrics"]["pr_auc"])
                - float(reference_by_subject[subject]["metrics"]["pr_auc"])
                for subject in common_subjects
            ],
            dtype=np.float64,
        )
        delta = suite.context_suite.paired_bootstrap_mean_ci(
            differences,
            int(config["bootstrap_samples"]),
            suite.context_suite.stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                f"{experiment_id}__vs__{reference_id}",
            ),
        )
        result[experiment_id] = {
            "experiment_id": experiment_id,
            "reference_experiment_id": reference_id,
            "reference_definition": "GRU-NBM H050 (0.5 s horizon)",
            "common_subjects": ",".join(common_subjects),
            **delta,
        }
    return result


def validate_fold_done(
    root: Path,
    subject: str,
    config: Mapping[str, Any],
    experiments: list[dict[str, Any]],
    *,
    complete_cells: set[str],
) -> bool:
    fold_root = root / f"loso_{subject}"
    done_path = fold_root / "FOLD_DONE.json"
    expected_cell_ids = {
        f"{subject}/{item['experiment_id']}" for item in experiments
    }
    fold_complete = expected_cell_ids.issubset(complete_cells)
    if not fold_complete:
        require(
            not done_path.exists(),
            f"{subject}: FOLD_DONE exists before all four cells complete",
        )
        return False
    done = validate_done(
        done_path,
        stage="horizon_fold",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/fold",
    )
    require(done is not None, f"{subject}: completed fold lacks FOLD_DONE")
    artifacts: dict[str, Path] = {
        "fold_config": fold_root / "fold_config.json",
        "scaler": fold_root / "scaler.json",
        "split_indices": fold_root / "split_indices.npz",
        "common_history_support": fold_root / "common_history_support.npz",
    }
    for item in experiments:
        horizon_id = str(item["horizon_id"])
        nbm_root = suite.nbm_root_for(root, subject, item)
        task_root = suite.task_root_for(root, subject, item)
        artifacts[f"{horizon_id}_classifier_done"] = task_root / "DONE.json"
        artifacts[f"{horizon_id}_nbm_done"] = nbm_root / "nbm" / "DONE.json"
        artifacts[f"{horizon_id}_residual_done"] = (
            nbm_root / "RESIDUAL_CACHE_DONE.json"
        )
    assert_done_artifacts(done, artifacts, done_path, f"{subject}/fold")
    require(
        done.get("test_subject") == subject,
        f"{subject}: FOLD_DONE test subject mismatch",
    )
    require(
        done.get("completed_horizons") == list(EXPECTED_HORIZONS),
        f"{subject}: FOLD_DONE horizon list mismatch",
    )
    require(
        done.get("common_history_support_sha256")
        == sha256_file(fold_root / "common_history_support.npz"),
        f"{subject}: FOLD_DONE support hash mismatch",
    )
    return True


def validate_root_summaries(
    root: Path,
    config: Mapping[str, Any],
    experiments: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    completed_fold_manifests: int,
) -> None:
    by_experiment: dict[str, list[dict[str, Any]]] = {
        str(item["experiment_id"]): [] for item in experiments
    }
    for cell in completed:
        by_experiment[str(cell["experiment_id"])].append(cell)
    for cells in by_experiment.values():
        cells.sort(key=lambda item: EXPECTED_SUBJECTS.index(item["subject"]))

    expected_deltas = recompute_paired_deltas(
        config,
        experiments,
        by_experiment,
    )
    aggregate = load_json(root / "aggregate_metrics.json")
    status = load_json(root / "status.json")
    completed_count = len(completed)
    expected_state = (
        "complete"
        if completed_count == EXPECTED_CLASSIFIER_CELLS
        else "partial"
    )
    expected_status = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_experiments": 4,
        "expected_nbm_tasks": EXPECTED_CLASSIFIER_CELLS,
        "expected_classifier_cells": EXPECTED_CLASSIFIER_CELLS,
        "completed_classifier_cells": completed_count,
        "expected_fold_manifests": 8,
        "completed_fold_manifests": completed_fold_manifests,
        "status": expected_state,
    }
    for key, expected in expected_status.items():
        require(
            status.get(key) == expected,
            f"status.json/{key}: {status.get(key)!r} != {expected!r}",
        )
    expected_aggregate_identity = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "aggregation_unit": "held_out_subject",
        "ranking_metric": "subject_macro_pr_auc_mean",
    }
    for key, expected in expected_aggregate_identity.items():
        require(
            aggregate.get(key) == expected,
            f"aggregate_metrics.json/{key}: value mismatch",
        )
    saved_experiments = aggregate.get("experiments")
    require(
        isinstance(saved_experiments, Mapping)
        and set(saved_experiments) == set(by_experiment),
        "aggregate_metrics.json experiment key set mismatch",
    )

    ranked: list[tuple[float, str]] = []
    recomputed_macros: dict[str, dict[str, Any]] = {}
    for item in experiments:
        experiment_id = str(item["experiment_id"])
        cells = by_experiment[experiment_id]
        rows = [cell["metrics"] for cell in cells]
        macro = (
            aggregate_fold_metrics(rows, list(suite.CLASSIFICATION_METRICS))
            if rows
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in suite.CLASSIFICATION_METRICS
            }
        )
        recomputed_macros[experiment_id] = macro
        saved = saved_experiments[experiment_id]
        require(
            saved.get("completed_folds")
            == [cell["subject"] for cell in cells],
            f"{experiment_id}: aggregate completed-fold list mismatch",
        )
        _mapping_close(
            saved.get("subject_macro", {}),
            macro,
            f"{experiment_id}/subject_macro",
        )
        _mapping_close(
            saved.get("delta_pr_auc_vs_h050", {}),
            expected_deltas[experiment_id],
            f"{experiment_id}/delta_pr_auc_vs_h050",
        )
        if cells:
            pooled = suite._prediction_metrics(
                np.concatenate(
                    [cell["predictions"]["y_true"] for cell in cells]
                ),
                np.concatenate(
                    [cell["predictions"]["y_prob"] for cell in cells]
                ),
                np.concatenate(
                    [cell["predictions"]["y_pred"] for cell in cells]
                ),
            )
            _mapping_close(
                saved.get("pooled", {}),
                pooled,
                f"{experiment_id}/pooled",
            )
            pr_mean = macro["pr_auc"]["mean"]
            if pr_mean is not None:
                ranked.append((-float(pr_mean), experiment_id))
        else:
            require(
                saved.get("pooled") is None,
                f"{experiment_id}: pending experiment has pooled metrics",
            )
    ranked.sort()
    expected_best = ranked[0][1] if ranked else None
    require(
        aggregate.get("best_experiment") == expected_best,
        "aggregate_metrics.json best_experiment mismatch",
    )
    require(
        status.get("best_experiment") == expected_best,
        "status.json best_experiment mismatch",
    )

    manifest_rows = read_csv(root / "experiment_manifest.csv")
    aggregate_rows = read_csv(root / "aggregate_summary.csv")
    paired_rows = read_csv(root / "paired_pr_auc_deltas.csv")
    publication_rows = read_csv(root / "publication_table.csv")
    fold_rows = read_csv(root / "fold_summary.csv")
    require(len(manifest_rows) == 4, "experiment_manifest.csv must have four rows")
    require(len(aggregate_rows) == 4, "aggregate_summary.csv must have four rows")
    require(len(paired_rows) == 4, "paired_pr_auc_deltas.csv must have four rows")
    require(len(publication_rows) == 4, "publication_table.csv must have four rows")
    require(
        len(fold_rows) == completed_count,
        "fold_summary.csv completed-cell row count mismatch",
    )

    manifest_by_id = {
        row["experiment_id"]: row for row in manifest_rows
    }
    aggregate_by_id = {
        row["experiment_id"]: row for row in aggregate_rows
    }
    paired_by_id = {row["experiment_id"]: row for row in paired_rows}
    require(
        set(manifest_by_id)
        == set(aggregate_by_id)
        == set(paired_by_id)
        == set(by_experiment),
        "Root summary experiment IDs differ",
    )
    for item in experiments:
        experiment_id = str(item["experiment_id"])
        cells = by_experiment[experiment_id]
        completed_subjects = ",".join(cell["subject"] for cell in cells)
        manifest = manifest_by_id[experiment_id]
        require(
            int(manifest["completed_folds"]) == len(cells)
            and manifest["completed_subjects"] == completed_subjects,
            f"{experiment_id}: manifest completion mismatch",
        )
        require(
            int(manifest["history_blocks"]) == int(item["history_blocks"])
            and int(manifest["history_samples"]) == 256
            and manifest["input_shape"] == "9x256",
            f"{experiment_id}: manifest history geometry mismatch",
        )
        aggregate_row = aggregate_by_id[experiment_id]
        require(
            int(aggregate_row["completed_folds"]) == len(cells),
            f"{experiment_id}: aggregate CSV completion mismatch",
        )
        for metric in suite.CLASSIFICATION_METRICS:
            expected_metric = recomputed_macros[experiment_id][metric]
            for statistic in ("mean", "std"):
                text = aggregate_row[f"{metric}_{statistic}"]
                expected = expected_metric[statistic]
                if expected is None:
                    require(
                        text == "",
                        f"{experiment_id}/{metric}_{statistic}: expected blank",
                    )
                else:
                    assert_close(
                        float(text),
                        expected,
                        f"{experiment_id}/{metric}_{statistic}",
                    )
        paired = paired_by_id[experiment_id]
        expected_delta = expected_deltas[experiment_id]
        for key in (
            "experiment_id",
            "reference_experiment_id",
            "reference_definition",
            "common_subjects",
        ):
            require(
                paired[key] == str(expected_delta[key]),
                f"{experiment_id}/paired/{key}: value mismatch",
            )
        for key in (
            "mean_delta",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "bootstrap_samples",
        ):
            expected = expected_delta[key]
            if expected is None:
                require(
                    paired[key] == "",
                    f"{experiment_id}/paired/{key}: expected blank",
                )
            else:
                assert_close(
                    float(paired[key]),
                    expected,
                    f"{experiment_id}/paired/{key}",
                    rtol=1e-12,
                    atol=1e-12,
                )

    expected_fold_keys = {
        (cell["experiment_id"], cell["subject"]) for cell in completed
    }
    actual_fold_keys = {
        (row["experiment_id"], row["test_subject"]) for row in fold_rows
    }
    require(
        actual_fold_keys == expected_fold_keys,
        "fold_summary.csv cell identities mismatch",
    )

    results_done_path = root / "RESULTS_DONE.json"
    if completed_count == EXPECTED_CLASSIFIER_CELLS:
        done = validate_done(
            results_done_path,
            stage="horizon_suite_results",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id="root/results",
        )
        require(done is not None, "Complete suite lacks RESULTS_DONE.json")
        assert_done_artifacts(
            done,
            {
                "config": root / "config.json",
                "run_manifest": root / "run_manifest.json",
                "fold_summary": root / "fold_summary.csv",
                "experiment_manifest": root / "experiment_manifest.csv",
                "aggregate_summary": root / "aggregate_summary.csv",
                "paired_pr_auc_deltas": root / "paired_pr_auc_deltas.csv",
                "publication_table": root / "publication_table.csv",
                "aggregate_metrics": root / "aggregate_metrics.json",
                "status": root / "status.json",
            },
            results_done_path,
            "root/results",
        )
    else:
        require(
            not results_done_path.exists(),
            "Partial suite must not contain RESULTS_DONE.json",
        )


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"Audit version: {report.get('audit_version')}",
        f"Status: {report.get('status')}",
        f"Checked cells: {report.get('checked_cells')}/"
        f"{report.get('expected_cells')}",
        f"Checked fold manifests: {report.get('checked_fold_manifests')}/8",
        f"Full complete: {report.get('full_complete')}",
        f"Allow partial: {report.get('allow_partial')}",
        "",
        "Missing cells:",
    ]
    lines.extend(f"- {value}" for value in report.get("missing_cells", []))
    lines.extend(("", "Failures:"))
    lines.extend(f"- {value}" for value in report.get("failures", []))
    lines.extend(("", "Warnings:"))
    lines.extend(f"- {value}" for value in report.get("warnings", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.result_dir.resolve()
    require(root.is_dir(), f"Result directory does not exist: {root}")
    config, experiments = validate_protocol(
        root,
        allow_partial=bool(args.allow_partial),
    )
    data_dir = resolve_data_dir(config, args.data_dir)
    dataset, master, windows_by_horizon, classification_windows = (
        rebuild_dataset_and_windows(config, data_dir)
    )
    residual_tolerance = (
        float(args.residual_tolerance)
        if args.residual_tolerance is not None
        else (2e-2 if bool(config.get("amp")) else 2e-3)
    )
    prediction_tolerance = (
        float(args.prediction_tolerance)
        if args.prediction_tolerance is not None
        else (5e-3 if bool(config.get("amp")) else 3e-5)
    )
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "data_dir": str(data_dir),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_cells": EXPECTED_CLASSIFIER_CELLS,
        "checked_cells": 0,
        "checked_fold_manifests": 0,
        "missing_cells": [],
        "allow_partial": bool(args.allow_partial),
        "reportable": False,
        "full_complete": False,
        "failures": [],
        "warnings": [],
        "tolerances": {
            "sampled_residual_replay": residual_tolerance,
            "classifier_probability_replay": prediction_tolerance,
        },
    }

    completed: list[dict[str, Any]] = []
    complete_cell_ids: set[str] = set()
    folds: dict[str, dict[str, Any]] = {}
    for subject in EXPECTED_SUBJECTS:
        fold_root = root / f"loso_{subject}"
        if not fold_root.exists():
            report["missing_cells"].extend(
                f"{subject}/{item['experiment_id']}" for item in experiments
            )
            continue
        try:
            fold = validate_fold(
                root,
                subject,
                config,
                dataset,
                master,
                windows_by_horizon,
                classification_windows,
            )
            folds[subject] = fold
        except Exception as error:  # noqa: BLE001 - collect all fold failures.
            report["failures"].append(
                f"{subject}/fold: {type(error).__name__}: {error}"
            )
            report["missing_cells"].extend(
                f"{subject}/{item['experiment_id']}" for item in experiments
            )
            continue

        fold_cells: list[dict[str, Any]] = []
        for horizon in experiments:
            horizon_id = str(horizon["horizon_id"])
            cell_id = f"{subject}/{horizon['experiment_id']}"
            nbm_root = suite.nbm_root_for(root, subject, horizon)
            task_root = suite.task_root_for(root, subject, horizon)
            stage_presence = {
                "nbm": (nbm_root / "nbm" / "DONE.json").exists(),
                "residual": (nbm_root / "RESIDUAL_CACHE_DONE.json").exists(),
                "classifier": (task_root / "DONE.json").exists(),
            }
            if not all(stage_presence.values()):
                report["missing_cells"].append(cell_id)
                if stage_presence["residual"] and not stage_presence["nbm"]:
                    report["failures"].append(
                        f"{cell_id}: residual DONE exists without NBM DONE"
                    )
                if stage_presence["classifier"] and not stage_presence["residual"]:
                    report["failures"].append(
                        f"{cell_id}: classifier DONE exists without residual DONE"
                    )
                continue
            try:
                nbm_sha, model, _ = validate_nbm_task(
                    root,
                    subject,
                    horizon,
                    config,
                    fold,
                )
                residual_sha, features = validate_residual_cache(
                    root,
                    subject,
                    horizon,
                    config,
                    fold,
                    dataset,
                    windows_by_horizon[horizon_id],
                    nbm_sha,
                    model,
                    tolerance=residual_tolerance,
                )
                audited = audit_classifier_task(
                    root,
                    subject,
                    horizon,
                    config,
                    fold,
                    dataset,
                    classification_windows,
                    residual_sha,
                    features,
                    tolerance=prediction_tolerance,
                )
                completed.append(audited)
                fold_cells.append(audited)
                complete_cell_ids.add(cell_id)
            except Exception as error:  # noqa: BLE001 - collect all cell failures.
                report["failures"].append(
                    f"{cell_id}: {type(error).__name__}: {error}"
                )

        if len(fold_cells) > 1:
            reference = fold_cells[0]
            for cell in fold_cells[1:]:
                require(
                    np.array_equal(
                        cell["predictions"]["window_index"],
                        reference["predictions"]["window_index"],
                    )
                    and np.array_equal(
                        cell["predictions"]["y_true"],
                        reference["predictions"]["y_true"],
                    ),
                    f"{subject}: completed horizons do not share test IDs/labels",
                )
                require(
                    cell["initial_state_sha256"]
                    == reference["initial_state_sha256"],
                    f"{subject}: TCN-M initialization differs by horizon",
                )
                common_epochs = min(
                    len(cell["shuffle_seeds"]),
                    len(reference["shuffle_seeds"]),
                )
                require(
                    cell["shuffle_seeds"][:common_epochs]
                    == reference["shuffle_seeds"][:common_epochs],
                    f"{subject}: classifier epoch shuffle order differs by horizon",
                )

    checked_fold_manifests = 0
    for subject in EXPECTED_SUBJECTS:
        fold_root = root / f"loso_{subject}"
        if not fold_root.exists():
            continue
        try:
            checked_fold_manifests += int(
                validate_fold_done(
                    root,
                    subject,
                    config,
                    experiments,
                    complete_cells=complete_cell_ids,
                )
            )
        except Exception as error:  # noqa: BLE001
            report["failures"].append(
                f"{subject}/FOLD_DONE: {type(error).__name__}: {error}"
            )

    try:
        validate_root_summaries(
            root,
            config,
            experiments,
            completed,
            checked_fold_manifests,
        )
    except Exception as error:  # noqa: BLE001
        report["failures"].append(
            f"root summaries: {type(error).__name__}: {error}"
        )

    report["checked_cells"] = len(completed)
    report["checked_fold_manifests"] = checked_fold_manifests
    report["missing_cells"] = sorted(set(report["missing_cells"]))
    full_complete = (
        len(completed) == EXPECTED_CLASSIFIER_CELLS
        and checked_fold_manifests == len(EXPECTED_SUBJECTS)
        and not report["missing_cells"]
        and not report["failures"]
    )
    report["full_complete"] = full_complete
    report["reportable"] = full_complete
    if report["failures"]:
        report["status"] = "fail"
    elif report["missing_cells"]:
        report["status"] = "partial_pass" if args.allow_partial else "fail"
        if not args.allow_partial:
            report["failures"].append(
                "Suite is incomplete; --allow-partial is required for an "
                "interim audit"
            )
    else:
        report["status"] = "pass"
    return report


def finalize_audit_artifacts(
    root: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "AUDIT_REPORT.json"
    text_path = root / "AUDIT_REPORT.txt"
    complete_path = root / "SUITE_COMPLETE.json"
    atomic_json_dump(dict(report), report_path)
    write_text_report(text_path, report)
    if report.get("status") == "pass" and bool(report.get("full_complete")):
        atomic_json_dump(
            {
                "format_version": 1,
                "suite_version": suite.SUITE_VERSION,
                "audit_version": AUDIT_VERSION,
                "status": "complete",
                "protocol_fingerprint": report["protocol_fingerprint"],
                "expected_cells": EXPECTED_CLASSIFIER_CELLS,
                "checked_cells": EXPECTED_CLASSIFIER_CELLS,
                "checked_fold_manifests": len(EXPECTED_SUBJECTS),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_report_sha256": sha256_file(report_path),
            },
            complete_path,
        )
    elif complete_path.exists():
        complete_path.unlink()
    return report_path, text_path, complete_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.result_dir.resolve()
    try:
        report = audit(args)
    except Exception as error:  # noqa: BLE001 - always leave an audit artifact.
        report = {
            "audit_version": AUDIT_VERSION,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_dir": str(root),
            "expected_cells": EXPECTED_CLASSIFIER_CELLS,
            "checked_cells": 0,
            "checked_fold_manifests": 0,
            "missing_cells": [],
            "allow_partial": bool(args.allow_partial),
            "reportable": False,
            "full_complete": False,
            "failures": [f"fatal: {type(error).__name__}: {error}"],
            "warnings": [],
            "status": "fail",
        }
    finalize_audit_artifacts(root, report)
    print(
        f"[horizon-audit] status={report['status']} "
        f"checked={report.get('checked_cells', 0)}/"
        f"{EXPECTED_CLASSIFIER_CELLS} "
        f"folds={report.get('checked_fold_manifests', 0)}/8 "
        f"missing={len(report.get('missing_cells', []))} "
        f"failures={len(report.get('failures', []))}",
        flush=True,
    )
    if report.get("status") not in {"pass", "partial_pass"}:
        for failure in report.get("failures", []):
            print(f"[horizon-audit] FAIL {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
