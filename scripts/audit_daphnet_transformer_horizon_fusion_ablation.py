#!/usr/bin/env python
"""Strict audit for the Transformer horizon-by-fusion Daphnet suite.

The audit independently rebuilds LOSO support, clean-normal training sets,
three horizon-specific four-second histories, all nine classifier inputs,
classifier predictions, metrics, summary tables, and every DONE hash chain.
Partial audits never create ``SUITE_COMPLETE.json``.
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

import run_daphnet_tcn_rf_ablation as rf
import run_daphnet_transformer_horizon_fusion_ablation as suite
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


AUDIT_VERSION = "daphnet_transformer_horizon_fusion_audit.v1"
EXPECTED_SUBJECTS = tuple(suite.EXPECTED_LOSO_SUBJECTS)
EXPECTED_EXCLUDED = {"S04", "S10"}
EXPECTED_CHANNELS = tuple(suite.EXPECTED_CHANNEL_NAMES)
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_HORIZONS: dict[str, dict[str, Any]] = {
    "H050": {"seconds": 0.5, "samples": 32, "blocks": 8},
    "H100": {"seconds": 1.0, "samples": 64, "blocks": 4},
    "H200": {"seconds": 2.0, "samples": 128, "blocks": 2},
}
EXPECTED_INPUTS = tuple(suite.INPUT_VARIANTS)
EXPECTED_CELLS = len(EXPECTED_SUBJECTS) * len(EXPECTED_INPUTS)
EXPECTED_DILATIONS = (1, 2, 4, 8, 8, 8)
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
        f"{split}_{horizon.lower()}_history_window_index"
        for split in EXPECTED_SPLITS
        for horizon in EXPECTED_HORIZONS
    ),
}
EXPECTED_PRIMITIVE_KEYS = {
    f"{split}_{key}"
    for split in EXPECTED_SPLITS
    for key in ("raw", "error", "sigma", "y", "window_index")
}
PREDICTION_KEYS = {"window_index", "y_true", "y_prob", "y_pred"}
NBM_ARTIFACTS = {"best", "last", "training"}
PRIMITIVE_ARTIFACTS = {"cache", "diagnostics"}
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
    """A scientific-protocol or artifact-integrity violation."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the strict Daphnet Transformer H050/H100/H200 "
            "Error/Raw+Error TCN-M LOSO suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Fallback processed Daphnet directory after moving results.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Audit completed cells without requiring 72/72 completion. "
            "A partial audit can never create SUITE_COMPLETE.json."
        ),
    )
    parser.add_argument(
        "--primitive-tolerance",
        type=float,
        default=None,
        help="Tolerance for sampled Transformer primitive replay.",
    )
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=None,
        help="Tolerance for CPU replay of saved classifier probabilities.",
    )
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
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
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = done_path.parent / path
    return path.resolve()


def assert_done_artifacts(
    done: Mapping[str, Any],
    expected: Mapping[str, Path],
    done_path: Path,
    label: str,
) -> None:
    artifacts = done.get("artifacts")
    require(isinstance(artifacts, Mapping), f"{label}: artifact map missing")
    require(
        set(artifacts) == set(expected),
        f"{label}: DONE artifact keys differ: {sorted(artifacts)}",
    )
    for key, expected_path in expected.items():
        entry = artifacts[key]
        require(isinstance(entry, Mapping), f"{label}/{key}: bad entry")
        actual = artifact_path(done_path, entry)
        require(
            resolved_equal(actual, expected_path),
            f"{label}/{key}: path {actual} != {expected_path}",
        )
        require(actual.is_file(), f"{label}/{key}: artifact missing")
        require(
            str(entry.get("sha256")) == sha256_file(actual),
            f"{label}/{key}: SHA-256 mismatch",
        )
        require(
            int(entry.get("bytes", -1)) == int(actual.stat().st_size),
            f"{label}/{key}: byte-count mismatch",
        )


def assert_close(
    actual: Any,
    expected: Any,
    label: str,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    if actual in (None, "") or expected in (None, ""):
        require(
            actual in (None, "") and expected in (None, ""),
            f"{label}: {actual!r} != {expected!r}",
        )
        return
    left, right = float(actual), float(expected)
    if not (math.isfinite(left) and math.isfinite(right)):
        require(
            (math.isnan(left) and math.isnan(right)) or left == right,
            f"{label}: {left!r} != {right!r}",
        )
        return
    require(
        bool(np.isclose(left, right, rtol=rtol, atol=atol)),
        f"{label}: {left:.12g} != {right:.12g}",
    )


def mapping_close(
    observed: Any,
    expected: Any,
    label: str,
) -> None:
    if isinstance(expected, Mapping):
        require(isinstance(observed, Mapping), f"{label}: expected mapping")
        require(
            set(observed) == set(expected),
            f"{label}: key set differs",
        )
        for key, value in expected.items():
            mapping_close(observed[key], value, f"{label}/{key}")
    elif isinstance(expected, list):
        require(isinstance(observed, list), f"{label}: expected list")
        require(len(observed) == len(expected), f"{label}: list length differs")
        for index, value in enumerate(expected):
            mapping_close(observed[index], value, f"{label}/{index}")
    elif isinstance(expected, (int, float, np.integer, np.floating)) or expected is None:
        assert_close(observed, expected, label)
    else:
        require(observed == expected, f"{label}: {observed!r} != {expected!r}")


def check_indices(value: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    require(result.ndim == 1, f"{label}: indices must be 1-D")
    require(
        len(result) == len(np.unique(result)),
        f"{label}: duplicate indices",
    )
    if len(result):
        require(int(result.min()) >= 0, f"{label}: negative index")
        require(int(result.max()) < size, f"{label}: out-of-range index")
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


def validate_protocol(
    root: Path,
    *,
    allow_partial: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(root / "config.json")
    require(config.get("suite_version") == suite.SUITE_VERSION, "suite version differs")
    require(
        canonical_fingerprint(protocol_payload(config))
        == config.get("protocol_fingerprint"),
        "config protocol fingerprint cannot be reproduced",
    )
    run_manifest = load_json(root / "run_manifest.json")
    require(
        run_manifest
        == {
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
        "run_manifest differs from immutable config protocol",
    )
    require(
        tuple(config.get("folds_resolved", [])) == EXPECTED_SUBJECTS,
        "LOSO fold order is not the canonical eight subjects",
    )
    require(
        tuple(config.get("subjects", [])) == EXPECTED_SUBJECTS,
        "post-exclusion subject list differs",
    )
    require(
        set(config.get("excluded_subjects", [])) == EXPECTED_EXCLUDED,
        "excluded subjects must be exactly S04/S10",
    )
    fixed = {
        "nbm": "transformer",
        "sampling_rate_hz": 64,
        "n_channels": 9,
        "context_samples": 128,
        "support_horizon_samples": 128,
        "fixed_label_samples": 32,
        "stride_samples": 16,
        "history_samples": 256,
        "raw6_samples": 384,
        "expected_experiments": 9,
        "expected_nbm_tasks": 24,
        "expected_classifier_cells": EXPECTED_CELLS,
        "protocol_scope": "strict_transformer_horizon3_fusion9_x_8_fold",
    }
    for key, expected in fixed.items():
        require(
            config.get(key) == expected,
            f"config/{key}: {config.get(key)!r} != {expected!r}",
        )
    require(tuple(config.get("channel_names", [])) == EXPECTED_CHANNELS, "channel order differs")
    require(bool(config.get("cache_residuals")), "primitive caching is required")
    require(bool(config.get("deterministic")), "deterministic mode is required")

    horizons = list(config.get("horizon_variants", []))
    require(len(horizons) == 3, "exactly three horizons are required")
    for item, (horizon_id, expected) in zip(horizons, EXPECTED_HORIZONS.items()):
        require(item.get("horizon_id") == horizon_id, "horizon order/ID differs")
        assert_close(item.get("horizon_seconds"), expected["seconds"], f"{horizon_id}/seconds")
        require(int(item.get("horizon_samples", -1)) == expected["samples"], f"{horizon_id}: samples differ")
        require(int(item.get("history_blocks", -1)) == expected["blocks"], f"{horizon_id}: blocks differ")
        require(
            int(item["horizon_samples"]) * int(item["history_blocks"]) == 256,
            f"{horizon_id}: history is not four seconds",
        )

    require(
        tuple(config.get("input_variants", [])) == EXPECTED_INPUTS,
        "nine input variants/order differ",
    )
    experiments = list(config.get("experiments", []))
    require(
        [item.get("cell_id") for item in experiments] == list(EXPECTED_INPUTS),
        "experiment registry differs",
    )
    definitions = {
        str(item["cell_id"]): item for item in suite.INPUT_DEFINITIONS
    }
    for cell in experiments:
        cell_id = str(cell["cell_id"])
        definition = definitions[cell_id]
        for key in (
            "input_variant",
            "kind",
            "horizon_id",
            "in_channels",
            "history_samples",
            "history_blocks",
            "formula",
        ):
            require(
                cell.get(key) == definition.get(key),
                f"{cell_id}/{key}: experiment definition differs",
            )
        require(cell.get("experiment_id") == suite.experiment_id(cell_id), f"{cell_id}: experiment ID differs")
        require(cell.get("input_shape") == f"{cell['in_channels']}x{cell['history_samples']}", f"{cell_id}: input shape differs")
        require(tuple(cell.get("dilations", [])) == EXPECTED_DILATIONS, f"{cell_id}: TCN dilations differ")
        require(int(cell.get("receptive_field_samples", -1)) == 125, f"{cell_id}: RF differs")
    require(config.get("comparisons") == list(suite.COMPARISONS), "paired comparison registry differs")

    classifier = config.get("classifier", {})
    require(isinstance(classifier, Mapping), "classifier config missing")
    require(classifier.get("name") == "TCN-M", "classifier is not TCN-M")
    require(tuple(classifier.get("dilations", [])) == EXPECTED_DILATIONS, "TCN-M dilations differ")
    require(int(classifier.get("receptive_field_samples", -1)) == 125, "TCN-M RF is not 125")
    require(classifier.get("global_pooling") == "mean_and_max_over_full_input", "pooling rule differs")
    states, counts, hashes, backbone_hash = suite._aligned_classifier_states(
        int(config["seed"]),
        int(classifier["hidden_channels"]),
        float(classifier["dropout"]),
        bool(config["deterministic"]),
    )
    del states
    require(
        classifier.get("parameter_count_by_in_channels")
        == {str(key): value for key, value in counts.items()},
        "classifier parameter counts differ from reconstruction",
    )
    require(
        classifier.get("template_initial_state_sha256_by_in_channels")
        == {str(key): value for key, value in hashes.items()},
        "classifier template initial hashes differ",
    )
    require(
        classifier.get("shared_backbone_initial_state_sha256") == backbone_hash,
        "classifier shared-backbone hash differs",
    )

    architectures = config.get("transformer_architectures", {})
    require(
        isinstance(architectures, Mapping)
        and set(architectures) == set(EXPECTED_HORIZONS),
        "Transformer architecture registry is incomplete",
    )
    shared_hashes: set[str] = set()
    parameters: list[int] = []
    for horizon_id, expected in EXPECTED_HORIZONS.items():
        architecture = architectures[horizon_id]
        model_config = architecture.get("model_config", {})
        require(model_config.get("name") == "transformer", f"{horizon_id}: NBM is not Transformer")
        require(int(model_config.get("in_channels", -1)) == 9, f"{horizon_id}: channel count differs")
        require(int(model_config.get("horizon", -1)) == expected["samples"], f"{horizon_id}: decoder horizon differs")
        shared_hashes.add(str(architecture.get("initial_shared_encoder_sha256")))
        parameters.append(int(architecture.get("parameter_count", -1)))
        require(int(architecture.get("decoder_parameter_count", -1)) > 0, f"{horizon_id}: decoder count invalid")
        suite.core.set_seed(int(config["seed"]), bool(config["deterministic"]))
        reconstructed = build_nbm(
            "transformer",
            in_channels=9,
            horizon=int(expected["samples"]),
            hidden_channels=int(config["nbm_hidden"]),
            dropout=float(config["nbm_dropout"]),
            transformer_heads=int(config["transformer_heads"]),
            transformer_layers=int(config["transformer_layers"]),
            transformer_ffn=int(config["transformer_ffn"]),
            max_context_samples=128,
        )
        decoder_count = sum(
            int(parameter.numel())
            for name, parameter in reconstructed.named_parameters()
            if name.startswith("decoder.")
        )
        reconstructed_shared = rf.state_dict_sha256(
            {
                name: value.detach().cpu()
                for name, value in reconstructed.state_dict().items()
                if not name.startswith("decoder.")
            }
        )
        require(reconstructed.model_config() == model_config, f"{horizon_id}: architecture cannot be reconstructed")
        require(
            sum(int(parameter.numel()) for parameter in reconstructed.parameters())
            == int(architecture["parameter_count"]),
            f"{horizon_id}: parameter count cannot be reconstructed",
        )
        require(decoder_count == int(architecture["decoder_parameter_count"]), f"{horizon_id}: decoder count cannot be reconstructed")
        require(reconstructed_shared == architecture["initial_shared_encoder_sha256"], f"{horizon_id}: initial shared encoder hash differs")
        del reconstructed
    require(
        shared_hashes == {str(config.get("shared_initial_transformer_encoder_sha256"))},
        "Transformer shared initial encoder differs by horizon",
    )
    require(parameters == sorted(parameters), "Transformer parameter count must grow with horizon")

    implementation = config.get("implementation", {})
    files = implementation.get("files", {}) if isinstance(implementation, Mapping) else {}
    require(tuple(files) == tuple(suite.IMPLEMENTATION_FILES), "implementation file registry differs")
    require(canonical_fingerprint(files) == implementation.get("sha256"), "implementation aggregate hash differs")
    for relative, expected_sha in files.items():
        path = REPO_ROOT / str(relative)
        if path.is_file():
            require(sha256_file(path) == expected_sha, f"implementation source drift: {relative}")
    if not allow_partial:
        require(config.get("protocol_scope") == fixed["protocol_scope"], "full audit requires strict protocol")
    return config, experiments


def resolve_data_dir(config: Mapping[str, Any], fallback: Path | None) -> Path:
    for candidate in (fallback, Path(str(config.get("data_dir", "")))):
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    raise AuditError(
        "Processed Daphnet data not found; pass --data-dir after moving results"
    )


def rebuild_dataset_and_windows(
    config: Mapping[str, Any],
    data_dir: Path,
) -> tuple[DaphnetDataset, WindowTable, dict[str, WindowTable], WindowTable]:
    require(dataset_fingerprint(data_dir) == config.get("data_sha256"), "dataset fingerprint differs")
    source = DaphnetDataset.load(
        data_dir,
        flatline_seconds=float(config["flatline_seconds"]),
        zero_tolerance=float(config["zero_tolerance"]),
    )
    require(tuple(source.channel_names) == EXPECTED_CHANNELS, "loaded channel order differs")
    require(list(source.subjects) == list(config["source_subjects"]), "source subject registry differs")
    dataset = DaphnetDataset(
        root=source.root,
        records=[
            record
            for record in source.records
            if record.subject_id not in EXPECTED_EXCLUDED
        ],
        sampling_rate_hz=source.sampling_rate_hz,
        channel_names=source.channel_names,
    )
    require(tuple(dataset.subjects) == EXPECTED_SUBJECTS, "filtered subjects differ")
    raw_master = dataset.make_windows(
        warmup_samples=128,
        target_samples=128,
        stride_samples=16,
        fog_fraction_threshold=float(config["fog_fraction_threshold"]),
        normal_guard_samples=int(config["normal_guard_samples"]),
    )
    master = suite.relabel_master_windows(
        dataset,
        raw_master,
        fixed_label_samples=32,
        fog_fraction_threshold=float(config["fog_fraction_threshold"]),
    )
    windows = {
        horizon_id: suite.derive_horizon_windows(master, definition["samples"])
        for horizon_id, definition in EXPECTED_HORIZONS.items()
    }
    classification = suite.derive_classification_windows(master)
    require(len(master) == int(config["master_window_count"]), "master window count differs")
    require(suite.window_table_sha256(master) == config["master_window_sha256"], "master WindowTable hash differs")
    for horizon_id, value in windows.items():
        require(
            suite.window_table_sha256(value)
            == config["derived_window_sha256"][horizon_id],
            f"{horizon_id}: derived WindowTable hash differs",
        )
        require(np.array_equal(value.label, classification.label), f"{horizon_id}: labels differ")
        require(np.array_equal(value.clean_normal, master.clean_normal), f"{horizon_id}: clean-normal support differs")
    require(
        suite.window_table_sha256(classification)
        == config["classification_window_sha256"],
        "classification WindowTable hash differs",
    )
    require(
        np.array_equal(
            np.bincount(classification.label, minlength=2),
            np.asarray(config["fixed_label_class_counts"], dtype=np.int64),
        ),
        "fixed-label class counts differ",
    )
    require(int(master.clean_normal.sum()) == int(config["master_clean_normal_windows"]), "clean-normal count differs")
    return dataset, master, windows, classification


def recompute_common_support(
    windows_by_horizon: Mapping[str, WindowTable],
    split_indices: Mapping[str, np.ndarray],
    *,
    max_classifier_windows: int,
    seed: int,
    fold_index: int,
    labels: np.ndarray,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
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
    horizon_ids = list(EXPECTED_HORIZONS)
    result = {horizon_id: {} for horizon_id in horizon_ids}
    for split in EXPECTED_SPLITS:
        common_ids = set(raw[horizon_ids[0]][split].anchor_window_indices.tolist())
        for horizon_id in horizon_ids[1:]:
            common_ids &= set(raw[horizon_id][split].anchor_window_indices.tolist())
        ordered = np.asarray(
            [
                int(value)
                for value in raw[horizon_ids[0]][split].anchor_window_indices
                if int(value) in common_ids
            ],
            dtype=np.int64,
        )
        require(len(ordered) > 0, f"{split}: empty common support")
        if split == "train" and int(max_classifier_windows) > 0:
            rows = deterministic_subsample(
                np.arange(len(ordered), dtype=np.int64),
                int(max_classifier_windows),
                int(seed) + 100 + int(fold_index),
                np.asarray(labels, dtype=np.int8)[ordered],
            )
            ordered = ordered[rows]
        for horizon_id in horizon_ids:
            plan = raw[horizon_id][split]
            lookup = {
                int(window_id): int(row)
                for row, window_id in enumerate(plan.anchor_window_indices)
            }
            rows = np.asarray([lookup[int(value)] for value in ordered], dtype=np.int64)
            result[horizon_id][split] = {
                "anchor": ordered.copy(),
                "chain": np.asarray(split_indices[split], dtype=np.int64)[
                    plan.max_chain_rows[rows]
                ],
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
    anchor = np.asarray(anchor, dtype=np.int64)
    chain = np.asarray(chain, dtype=np.int64)
    require(chain.shape == (len(anchor), history_blocks), f"{label}: chain shape differs")
    require(np.array_equal(chain[:, -1], anchor), f"{label}: final block is not anchor")
    if not len(anchor):
        return
    records = windows.record_index[chain]
    require(np.all(records == records[:, :1]), f"{label}: history crosses records")
    starts = windows.target_start[chain].astype(np.int64)
    ends = windows.target_end[chain].astype(np.int64)
    require(np.all(ends - starts == horizon_samples), f"{label}: block horizon differs")
    if history_blocks > 1:
        require(np.all(np.diff(starts, axis=1) == horizon_samples), f"{label}: block spacing differs")
        require(np.all(starts[:, 1:] == ends[:, :-1]), f"{label}: blocks overlap/gap")
    require(np.all(ends[:, -1] == windows.target_end[anchor]), f"{label}: endpoint differs")
    require(np.all(ends[:, -1] - starts[:, 0] == 256), f"{label}: history is not 256 samples")


def expected_validation_subject(
    dataset: DaphnetDataset,
    windows: WindowTable,
    test_subject: str,
) -> str:
    subjects = list(dataset.subjects)
    start = subjects.index(test_subject)
    for offset in range(1, len(subjects)):
        candidate = subjects[(start + offset) % len(subjects)]
        indices = dataset.window_indices_for_subjects(windows, [candidate])
        if np.unique(windows.label[indices]).size == 2:
            return candidate
    raise AuditError(f"{test_subject}: no eligible validation subject")


def validate_fold(
    root: Path,
    subject: str,
    config: Mapping[str, Any],
    dataset: DaphnetDataset,
    master: WindowTable,
    windows_by_horizon: Mapping[str, WindowTable],
    classification: WindowTable,
) -> dict[str, Any]:
    fold_root = root / f"loso_{subject}"
    require(fold_root.is_dir(), f"{subject}: fold directory missing")
    fold_config_path = fold_root / "fold_config.json"
    scaler_path = fold_root / "scaler.json"
    split_path = fold_root / "split_indices.npz"
    support_path = fold_root / "common_history_support.npz"
    fingerprint_path = fold_root / "input_fingerprints.json"
    fold_config = load_json(fold_config_path)
    require(fold_config.get("suite_version") == suite.SUITE_VERSION, f"{subject}: fold suite differs")
    require(fold_config.get("protocol_fingerprint") == config["protocol_fingerprint"], f"{subject}: fold protocol differs")
    require(fold_config.get("test_subject") == subject, f"{subject}: test identity differs")
    validation_subject = expected_validation_subject(dataset, classification, subject)
    training_subjects = [
        value
        for value in dataset.subjects
        if value not in {subject, validation_subject}
    ]
    require(fold_config.get("val_subject") == validation_subject, f"{subject}: validation subject differs")
    require(fold_config.get("train_subjects") == training_subjects, f"{subject}: train subjects differ")
    require(set(fold_config.get("excluded_subjects", [])) == EXPECTED_EXCLUDED, f"{subject}: exclusions differ")
    require(
        len(training_subjects) == 6
        and set(training_subjects).isdisjoint({subject, validation_subject}),
        f"{subject}: LOSO leakage",
    )

    scaler_payload = load_json(scaler_path)
    require(scaler_payload == fold_config.get("scaler"), f"{subject}: scaler payload differs")
    require(sha256_file(scaler_path) == fold_config.get("scaler_sha256"), f"{subject}: scaler hash differs")
    scaler = dataset.fit_scaler(training_subjects, clip=float(config["robust_clip"]))
    require(
        np.array_equal(
            np.asarray(scaler_payload["center"], dtype=np.float32),
            scaler.center,
        ),
        f"{subject}: scaler center is not train-only",
    )
    require(
        np.array_equal(
            np.asarray(scaler_payload["scale"], dtype=np.float32),
            scaler.scale,
        ),
        f"{subject}: scaler scale is not train-only",
    )
    assert_close(scaler_payload["clip"], scaler.clip, f"{subject}/scaler_clip", rtol=0.0, atol=0.0)

    require(sha256_file(split_path) == fold_config.get("split_indices_sha256"), f"{subject}: split hash differs")
    with np.load(split_path, allow_pickle=False) as payload:
        require(set(payload.files) == EXPECTED_SPLIT_KEYS, f"{subject}: split keys differ")
        splits = {
            "train": check_indices(payload["train_window_index"], len(master), f"{subject}/train"),
            "validation": check_indices(payload["validation_window_index"], len(master), f"{subject}/validation"),
            "test": check_indices(payload["test_window_index"], len(master), f"{subject}/test"),
        }
        normal_train = check_indices(payload["normal_train_window_index"], len(master), f"{subject}/normal_train")
        normal_validation = check_indices(payload["normal_validation_window_index"], len(master), f"{subject}/normal_validation")

    expected_splits = {
        "train": dataset.window_indices_for_subjects(classification, training_subjects),
        "validation": dataset.window_indices_for_subjects(classification, [validation_subject]),
        "test": dataset.window_indices_for_subjects(classification, [subject]),
    }
    for split, expected in expected_splits.items():
        require(np.array_equal(splits[split], expected), f"{subject}/{split}: split differs from LOSO rebuild")
        require(int(fold_config["source_window_counts"][split]) == len(expected), f"{subject}/{split}: source count differs")
    require(subjects_for_windows(dataset, master, splits["train"]) == set(training_subjects), f"{subject}: train indices belong to wrong subjects")
    require(subjects_for_windows(dataset, master, splits["validation"]) == {validation_subject}, f"{subject}: validation indices belong to wrong subject")
    require(subjects_for_windows(dataset, master, splits["test"]) == {subject}, f"{subject}: test indices belong to wrong subject")

    fold_index = list(dataset.subjects).index(subject)
    expected_normal_train = dataset.window_indices_for_subjects(
        master,
        training_subjects,
        clean_normal_only=True,
    )
    expected_normal_train = deterministic_subsample(
        expected_normal_train,
        int(config["max_normal_windows"]),
        int(config["seed"]) + fold_index,
    )
    expected_normal_validation = dataset.window_indices_for_subjects(
        master,
        [validation_subject],
        clean_normal_only=True,
    )
    require(np.array_equal(normal_train, expected_normal_train), f"{subject}: normal-train support differs")
    require(np.array_equal(normal_validation, expected_normal_validation), f"{subject}: normal-validation support differs")
    require(
        np.all(master.clean_normal[normal_train])
        and np.all(master.label[normal_train] == 0),
        f"{subject}: normal training includes FoG/non-clean windows",
    )
    require(
        np.all(master.clean_normal[normal_validation])
        and np.all(master.label[normal_validation] == 0),
        f"{subject}: normal validation includes FoG/non-clean windows",
    )
    require(suite.array_sha256(normal_train) == fold_config["normal_train_window_indices_sha256"], f"{subject}: normal-train hash differs")
    require(suite.array_sha256(normal_validation) == fold_config["normal_validation_window_indices_sha256"], f"{subject}: normal-validation hash differs")
    require(int(fold_config["normal_train_windows"]) == len(normal_train), f"{subject}: normal-train count differs")
    require(int(fold_config["normal_validation_windows"]) == len(normal_validation), f"{subject}: normal-validation count differs")

    expected_support = recompute_common_support(
        windows_by_horizon,
        splits,
        max_classifier_windows=int(config["max_classifier_windows"]),
        seed=int(config["seed"]),
        fold_index=fold_index,
        labels=classification.label,
    )
    support_sha = sha256_file(support_path)
    require(support_sha == fold_config["common_history_support_sha256"], f"{subject}: common-support hash differs")
    support: dict[str, dict[str, dict[str, np.ndarray]]] = {
        horizon_id: {} for horizon_id in EXPECTED_HORIZONS
    }
    with np.load(support_path, allow_pickle=False) as payload:
        require(set(payload.files) == EXPECTED_SUPPORT_KEYS, f"{subject}: support keys differ")
        for split in EXPECTED_SPLITS:
            anchor = check_indices(
                payload[f"{split}_anchor_window_index"],
                len(master),
                f"{subject}/{split}/anchor",
            )
            labels = np.asarray(payload[f"{split}_y"], dtype=np.int8)
            require(labels.shape == (len(anchor),), f"{subject}/{split}: label shape differs")
            require(np.array_equal(labels, classification.label[anchor]), f"{subject}/{split}: labels are not final 0.5 s")
            require(np.unique(labels).size == 2, f"{subject}/{split}: common support lacks a class")
            require(np.array_equal(anchor, expected_support["H050"][split]["anchor"]), f"{subject}/{split}: anchor support differs")
            require(int(fold_config["common_anchor_counts"][split]) == len(anchor), f"{subject}/{split}: anchor count differs")
            require(suite.array_sha256(anchor) == fold_config["common_anchor_sha256"][split], f"{subject}/{split}: anchor hash differs")
            for horizon_id, definition in EXPECTED_HORIZONS.items():
                chain = np.asarray(
                    payload[f"{split}_{horizon_id.lower()}_history_window_index"],
                    dtype=np.int64,
                )
                require(np.array_equal(chain, expected_support[horizon_id][split]["chain"]), f"{subject}/{split}/{horizon_id}: chain differs")
                require(
                    suite.array_sha256(chain)
                    == fold_config["per_horizon_history_support_sha256"][horizon_id][split],
                    f"{subject}/{split}/{horizon_id}: chain hash differs",
                )
                validate_history_geometry(
                    windows_by_horizon[horizon_id],
                    anchor=anchor,
                    chain=chain,
                    horizon_samples=int(definition["samples"]),
                    history_blocks=int(definition["blocks"]),
                    label=f"{subject}/{split}/{horizon_id}",
                )
                support[horizon_id][split] = {
                    "anchor": anchor.copy(),
                    "chain": chain,
                    "y": labels.copy(),
                }

    require(fold_config.get("classification_window_sha256") == config["classification_window_sha256"], f"{subject}: classification provenance differs")
    require(int(fold_config.get("label_window_samples", -1)) == 32, f"{subject}: label duration differs")
    classifier_seed = int(config["seed"]) + 10000 + fold_index
    require(int(fold_config.get("classifier_seed", -1)) == classifier_seed, f"{subject}: classifier seed differs")
    states, counts, hashes, backbone_hash = suite._aligned_classifier_states(
        classifier_seed,
        int(config["classifier"]["hidden_channels"]),
        float(config["classifier"]["dropout"]),
        bool(config["deterministic"]),
    )
    require(
        fold_config.get("parameter_count_by_in_channels")
        == {str(key): value for key, value in counts.items()},
        f"{subject}: fold parameter counts differ",
    )
    require(
        fold_config.get("reference_initial_state_sha256_by_in_channels")
        == {str(key): value for key, value in hashes.items()},
        f"{subject}: fold initial-state hashes differ",
    )
    require(
        fold_config.get("shared_backbone_initial_state_sha256") == backbone_hash,
        f"{subject}: shared TCN backbone differs",
    )

    primitive_hashes = fold_config.get("primitive_cache_sha256_by_horizon")
    input_fingerprints = fold_config.get("input_fingerprints")
    require(
        isinstance(primitive_hashes, Mapping)
        and set(primitive_hashes) == set(EXPECTED_HORIZONS),
        f"{subject}: primitive hash registry incomplete",
    )
    require(
        isinstance(input_fingerprints, Mapping)
        and set(input_fingerprints) == set(EXPECTED_INPUTS),
        f"{subject}: input fingerprint registry incomplete",
    )
    expected_fingerprints = suite.build_input_fingerprints(
        config,
        fold_config,
        primitive_hashes,
    )
    require(input_fingerprints == expected_fingerprints, f"{subject}: input fingerprints cannot be reproduced")
    fingerprint_payload = load_json(fingerprint_path)
    require(
        fingerprint_payload
        == {
            "suite_version": suite.SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "test_subject": subject,
            "common_history_support_sha256": support_sha,
            "input_fingerprints": input_fingerprints,
        },
        f"{subject}: input_fingerprints.json differs",
    )
    return {
        "root": fold_root,
        "config": fold_config,
        "validation_subject": validation_subject,
        "training_subjects": training_subjects,
        "scaler": scaler,
        "splits": splits,
        "normal_train": normal_train,
        "normal_validation": normal_validation,
        "support": support,
        "support_sha256": support_sha,
        "classifier_seed": classifier_seed,
        "classifier_states": states,
        "classifier_hashes": hashes,
        "input_fingerprints": dict(input_fingerprints),
    }


def build_transformer(
    config: Mapping[str, Any],
    horizon_samples: int,
) -> NormalBehaviourModel:
    return build_nbm(
        "transformer",
        in_channels=9,
        horizon=int(horizon_samples),
        hidden_channels=int(config["nbm_hidden"]),
        dropout=float(config["nbm_dropout"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_ffn=int(config["transformer_ffn"]),
        max_context_samples=128,
    )


def shared_transformer_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("decoder.")
    }


def validate_resume_checkpoint(
    payload: Mapping[str, Any],
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    input_fingerprint: str | None = None,
) -> None:
    if stage == "rf_classifier":
        require(input_fingerprint is not None, f"{task_id}: input fingerprint missing")
        rf.validate_rf_checkpoint(
            payload,
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            source_residual_sha256=str(input_fingerprint),
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
    require(epoch >= 1 and 0 <= best_epoch <= epoch, f"{task_id}: invalid epoch state")
    history = list(payload["history"])
    require(history and int(history[-1]["epoch"]) == epoch, f"{task_id}: history does not end at saved epoch")


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
    require(done is not None, f"{subject}/{horizon_id}: NBM DONE missing")
    assert_done_artifacts(
        done,
        {"best": best_path, "last": last_path, "training": training_path},
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
    require(best.get("model_name") == "transformer", f"{task_id}: best model is not Transformer")
    require(int(best.get("seed", -1)) == expected_seed, f"{task_id}: best seed differs")
    model = build_transformer(config, horizon_samples).cpu().eval()
    require(best.get("model_config") == model.model_config(), f"{task_id}: model config differs")
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        mean, sigma = model(torch.zeros(2, 9, 128, dtype=torch.float32))
    require(
        tuple(mean.shape) == (2, 9, horizon_samples)
        and tuple(sigma.shape) == (2, 9, horizon_samples),
        f"{task_id}: output shape differs",
    )
    require(
        bool(torch.isfinite(mean).all())
        and bool(torch.isfinite(sigma).all())
        and bool(torch.all(sigma > 0)),
        f"{task_id}: invalid output distribution",
    )

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        last,
        stage="nbm",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(last.get("model_name") == "transformer", f"{task_id}: last model is not Transformer")
    require(int(last.get("seed", -1)) == expected_seed, f"{task_id}: last seed differs")
    require(last.get("model_config") == best.get("model_config"), f"{task_id}: best/last configs differ")
    training = load_json(training_path)
    require(training.get("model_name") == "transformer", f"{task_id}: training model differs")
    require(int(training.get("seed", -1)) == expected_seed, f"{task_id}: training seed differs")
    require(training.get("model_config") == best.get("model_config"), f"{task_id}: training model config differs")
    architecture = config["transformer_architectures"][horizon_id]
    require(int(training.get("parameter_count", -1)) == int(architecture["parameter_count"]), f"{task_id}: parameter count differs")
    require(
        int(training.get("train_windows", -1)) == len(fold["normal_train"])
        and int(training.get("validation_windows", -1))
        == len(fold["normal_validation"]),
        f"{task_id}: NBM support count differs",
    )
    require(int(training.get("best_epoch", -1)) == int(best["best_epoch"]), f"{task_id}: best epoch differs")
    assert_close(training["best_val_nll"], best["best_val_nll"], f"{task_id}/best_val_nll")

    suite.core.set_seed(expected_seed, bool(config["deterministic"]))
    initial_model = build_transformer(config, horizon_samples)
    initial_shared_hash = rf.state_dict_sha256(
        shared_transformer_state(initial_model)
    )
    del initial_model
    summary = load_json(nbm_root / "nbm_summary.json")
    expected_summary = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "horizon_id": horizon_id,
        "horizon_seconds": float(horizon["horizon_seconds"]),
        "horizon_samples": horizon_samples,
        "context_seconds": 2.0,
        "context_samples": 128,
        "history_seconds": 4.0,
        "history_samples": 256,
        "history_blocks": int(horizon["history_blocks"]),
        "fixed_label_seconds": 0.5,
        "fixed_label_samples": 32,
        "master_clean_normal_support": True,
        "derived_window_sha256": config["derived_window_sha256"][horizon_id],
        "nbm_sha256": sha256_file(best_path),
        "transformer_architecture": architecture,
        "normal_training": training,
    }
    for key, expected in expected_summary.items():
        require(summary.get(key) == expected, f"{task_id}/summary/{key}: differs")
    require(
        fold["config"]["nbm_best_sha256_by_horizon"][horizon_id]
        == sha256_file(best_path),
        f"{task_id}: fold NBM provenance hash differs",
    )
    model.load_state_dict(best["model_state"], strict=True)
    model.eval()
    return sha256_file(best_path), model, {
        "summary": summary,
        "initial_shared_encoder_sha256": initial_shared_hash,
    }


def validate_primitive_cache(
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
    cache_path = nbm_root / "transformer_primitives.npz"
    diagnostics_path = nbm_root / "primitive_diagnostics.json"
    done_path = nbm_root / "PRIMITIVE_CACHE_DONE.json"
    task_id = suite._primitive_cache_task_id(horizon)
    done = validate_done(
        done_path,
        stage="transformer_primitive_cache",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        upstream_sha256=nbm_sha256,
    )
    require(done is not None, f"{subject}/{horizon_id}: primitive DONE missing")
    assert_done_artifacts(
        done,
        {"cache": cache_path, "diagnostics": diagnostics_path},
        done_path,
        f"{subject}/{horizon_id}/primitive",
    )
    features: dict[str, dict[str, np.ndarray]] = {}
    with np.load(cache_path, allow_pickle=False) as payload:
        require(set(payload.files) == EXPECTED_PRIMITIVE_KEYS, f"{task_id}: primitive keys differ")
        for split in EXPECTED_SPLITS:
            values = {
                "raw": np.asarray(payload[f"{split}_raw"], dtype=np.float32),
                "error": np.asarray(payload[f"{split}_error"], dtype=np.float32),
                "sigma": np.asarray(payload[f"{split}_sigma"], dtype=np.float32),
                "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
                "window_index": np.asarray(payload[f"{split}_window_index"], dtype=np.int64),
            }
            expected_indices = fold["splits"][split]
            require(np.array_equal(values["window_index"], expected_indices), f"{task_id}/{split}: indices differ")
            require(np.array_equal(values["y"], windows.label[expected_indices]), f"{task_id}/{split}: labels differ")
            expected_shape = (len(expected_indices), 9, horizon_samples)
            for key in ("raw", "error", "sigma"):
                require(values[key].shape == expected_shape, f"{task_id}/{split}/{key}: shape differs")
                require(np.isfinite(values[key]).all(), f"{task_id}/{split}/{key}: non-finite")
            require(np.all(values["sigma"] > 0), f"{task_id}/{split}: sigma is non-positive")
            features[split] = values
    require(
        sha256_file(cache_path)
        == fold["config"]["primitive_cache_sha256_by_horizon"][horizon_id],
        f"{task_id}: fold primitive hash differs",
    )

    model = model.cpu().eval()
    for split in EXPECTED_SPLITS:
        values = features[split]
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
        require(tuple(target.shape[1:]) == (9, horizon_samples), f"{task_id}/{split}: target shape differs")
        with torch.no_grad():
            mean, sigma = model(context)
        replay = {
            "raw": target.float().numpy(),
            "error": (target - mean.float()).numpy(),
            "sigma": sigma.float().numpy(),
        }
        for key, expected in replay.items():
            observed = values[key][rows]
            local_tolerance = 0.0 if key == "raw" else float(tolerance)
            require(
                np.allclose(
                    observed,
                    expected,
                    rtol=local_tolerance,
                    atol=local_tolerance,
                ),
                f"{task_id}/{split}: sampled {key} replay differs",
            )

    diagnostics = load_json(diagnostics_path)
    require(set(diagnostics) == set(EXPECTED_SPLITS), f"{task_id}: diagnostic splits differ")
    for split in EXPECTED_SPLITS:
        payload = diagnostics[split]
        require(int(payload["windows"]) == len(fold["splits"][split]), f"{task_id}/{split}: diagnostic count differs")
        require(
            list(payload["class_counts"])
            == np.bincount(
                windows.label[fold["splits"][split]],
                minlength=2,
            ).astype(int).tolist(),
            f"{task_id}/{split}: diagnostic class counts differ",
        )
        for key in (
            "forecast_rmse",
            "forecast_mae",
            "mean_sigma",
            "error_abs_mean",
            "error_rms",
        ):
            require(math.isfinite(float(payload[key])), f"{task_id}/{split}/{key}: non-finite")
        require(float(payload["mean_sigma"]) > 0, f"{task_id}/{split}: mean sigma invalid")
    summary = load_json(nbm_root / "nbm_summary.json")
    require(summary.get("primitive_cache_sha256") == sha256_file(cache_path), f"{task_id}: summary primitive hash differs")
    require(summary.get("primitive_diagnostics") == diagnostics, f"{task_id}: summary diagnostics differ")
    return sha256_file(cache_path), features


def gather_history(
    features: Mapping[str, np.ndarray],
    chain: np.ndarray,
    source_key: str,
    *,
    horizon_samples: int,
    history_blocks: int,
    label: str,
) -> np.ndarray:
    source_indices = np.asarray(features["window_index"], dtype=np.int64)
    require(
        source_indices.ndim == 1
        and len(source_indices) == len(np.unique(source_indices)),
        f"{label}: primitive IDs invalid",
    )
    history = np.asarray(chain, dtype=np.int64)
    require(
        history.shape == (len(history), int(history_blocks)),
        f"{label}: chain block count differs",
    )
    order = np.argsort(source_indices)
    sorted_indices = source_indices[order]
    positions = np.searchsorted(sorted_indices, history)
    require(np.all(positions < len(sorted_indices)), f"{label}: chain references absent primitive")
    require(np.array_equal(sorted_indices[positions], history), f"{label}: chain/primitive mapping differs")
    blocks = np.asarray(features[source_key], dtype=np.float32)[order[positions]]
    require(
        blocks.shape[1:]
        == (int(history_blocks), 9, int(horizon_samples)),
        f"{label}: gathered block shape differs",
    )
    result = blocks.transpose(0, 2, 1, 3).reshape(len(history), 9, -1)
    require(result.shape == (len(history), 9, 256), f"{label}: result is not [B,9,256]")
    require(np.isfinite(result).all(), f"{label}: result is non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def materialize_raw6(
    dataset: DaphnetDataset,
    classification: WindowTable,
    anchor: np.ndarray,
    scaler: Any,
    *,
    label: str,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.int64)
    result = np.empty((len(anchor), 9, 384), dtype=np.float32)
    for row, window_index in enumerate(anchor):
        record_index = int(classification.record_index[window_index])
        end = int(classification.target_end[window_index])
        start = end - 384
        require(start >= 0, f"{label}: Raw6 starts before record")
        record = dataset.records[record_index]
        require(bool(record.valid[start:end].all()), f"{label}: Raw6 includes invalid samples")
        scaled = scaler.transform(record.x[start:end])
        require(scaled.shape == (384, 9), f"{label}: Raw6 source shape differs")
        result[row] = scaled.T
    require(np.isfinite(result).all(), f"{label}: Raw6 is non-finite")
    return np.ascontiguousarray(result)


def materialize_cell_inputs(
    cell: Mapping[str, Any],
    fold: Mapping[str, Any],
    dataset: DaphnetDataset,
    classification: WindowTable,
    primitives: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    cell_id = str(cell["cell_id"])
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in EXPECTED_SPLITS:
        reference_support = fold["support"]["H050"][split]
        raw4 = gather_history(
            primitives["H050"][split],
            reference_support["chain"],
            "raw",
            horizon_samples=32,
            history_blocks=8,
            label=f"{cell_id}/{split}/raw4",
        )
        # Raw4 must be byte-identical no matter which horizon partitions it.
        for horizon_id, definition in EXPECTED_HORIZONS.items():
            comparison = gather_history(
                primitives[horizon_id][split],
                fold["support"][horizon_id][split]["chain"],
                "raw",
                horizon_samples=int(definition["samples"]),
                history_blocks=int(definition["blocks"]),
                label=f"{cell_id}/{split}/{horizon_id}/raw4",
            )
            require(np.array_equal(comparison, raw4), f"{cell_id}/{split}: Raw4 differs by horizon")
        if cell_id == "raw4":
            values = raw4
        elif cell_id == "raw6":
            values = materialize_raw6(
                dataset,
                classification,
                reference_support["anchor"],
                fold["scaler"],
                label=f"{cell_id}/{split}",
            )
        elif cell_id == "raw4_zero":
            values = np.ascontiguousarray(
                np.concatenate((raw4, np.zeros_like(raw4)), axis=1),
                dtype=np.float32,
            )
            require(np.count_nonzero(values[:, 9:, :]) == 0, f"{cell_id}/{split}: zero channels are nonzero")
        else:
            horizon_id = str(cell["horizon_id"])
            definition = EXPECTED_HORIZONS[horizon_id]
            error = gather_history(
                primitives[horizon_id][split],
                fold["support"][horizon_id][split]["chain"],
                "error",
                horizon_samples=int(definition["samples"]),
                history_blocks=int(definition["blocks"]),
                label=f"{cell_id}/{split}/error",
            )
            values = (
                error
                if str(cell["kind"]) == "error"
                else np.ascontiguousarray(
                    np.concatenate((raw4, error), axis=1),
                    dtype=np.float32,
                )
            )
            if str(cell["kind"]) == "fusion":
                require(np.array_equal(values[:, :9], raw4), f"{cell_id}/{split}: fusion raw half differs")
                require(np.array_equal(values[:, 9:], error), f"{cell_id}/{split}: fusion error half differs")
        expected_shape = (
            len(reference_support["anchor"]),
            int(cell["in_channels"]),
            int(cell["history_samples"]),
        )
        require(values.shape == expected_shape, f"{cell_id}/{split}: shape {values.shape} != {expected_shape}")
        result[split] = {
            cell_id: values,
            "y": reference_support["y"],
            "window_index": reference_support["anchor"],
        }
    return result


def load_predictions(path: Path, label: str) -> dict[str, np.ndarray]:
    require(path.is_file(), f"{label}: predictions missing")
    with np.load(path, allow_pickle=False) as payload:
        require(set(payload.files) == PREDICTION_KEYS, f"{label}: prediction keys differ")
        result = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    require(len({len(value) for value in result.values()}) == 1, f"{label}: prediction lengths differ")
    require(all(value.ndim == 1 for value in result.values()), f"{label}: predictions are not 1-D")
    require(len(result["window_index"]) == len(np.unique(result["window_index"])), f"{label}: duplicate prediction IDs")
    require(np.isin(result["y_true"], [0, 1]).all(), f"{label}: labels are not binary")
    require(np.isin(result["y_pred"], [0, 1]).all(), f"{label}: decisions are not binary")
    require(
        np.isfinite(result["y_prob"]).all()
        and np.all((result["y_prob"] >= 0) & (result["y_prob"] <= 1)),
        f"{label}: probabilities invalid",
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
        chunks.append(torch.sigmoid(model(x)).float().numpy())
    return (
        np.concatenate(chunks).astype(np.float64, copy=False)
        if chunks
        else np.empty(0, dtype=np.float64)
    )


def requested_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = (int(metrics[key]) for key in ("tn", "fp", "fn", "tp"))
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "macro_f1": 0.5 * (f1_nonfog + f1_fog),
        "roc_auc": metrics["auroc"],
        "pr_auc": metrics["auprc"],
        "fog_recall": metrics["sensitivity"],
        "fog_f1": f1_fog,
    }


def assert_metric_dict(
    saved: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    label: str,
) -> None:
    for key in CORE_BINARY_METRICS:
        assert_close(saved.get(key), recomputed.get(key), f"{label}/{key}")
    for key in ("tn", "fp", "fn", "tp", "n", "n_normal", "n_fog"):
        require(int(saved[key]) == int(recomputed[key]), f"{label}/{key}: differs")
    require(saved["confusion_matrix"] == recomputed["confusion_matrix"], f"{label}: confusion matrix differs")


def expected_classifier_config(
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    initial_hash: str,
) -> dict[str, Any]:
    classifier = config["classifier"]
    return {
        "in_channels": int(cell["in_channels"]),
        "hidden_channels": int(classifier["hidden_channels"]),
        "dropout": float(classifier["dropout"]),
        "kernel_size": 3,
        "dilations": list(EXPECTED_DILATIONS),
        "n_blocks": 6,
        "convolutions_per_block": 2,
        "receptive_field_samples": 125,
        "receptive_field_seconds": 125 / 64.0,
        "parameter_count": int(cell["parameter_count"]),
        "initial_state_sha256": initial_hash,
        "global_pooling": "mean_and_max_over_full_input",
    }


def audit_classifier_task(
    root: Path,
    subject: str,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
    dataset: DaphnetDataset,
    classification: WindowTable,
    primitives: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    task_root = suite.task_root_for(root, subject, cell_id)
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    metrics_path = task_root / "metrics.json"
    prediction_path = task_root / "predictions.npz"
    validation_path = task_root / "validation_predictions.npz"
    prediction_csv_path = task_root / "predictions.csv"
    done_path = task_root / "DONE.json"
    task_id = f"{subject}/{cell_id}"
    input_fingerprint = str(fold["input_fingerprints"][cell_id])
    in_channels = int(cell["in_channels"])
    initial_hash = str(fold["classifier_hashes"][in_channels])

    done = validate_done(
        done_path,
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(done is not None, f"{task_id}: classifier DONE missing")
    assert_done_artifacts(
        done,
        {
            "best": best_path,
            "last": last_path,
            "metrics": metrics_path,
            "predictions": prediction_path,
            "validation_predictions": validation_path,
            "predictions_csv": prediction_csv_path,
        },
        done_path,
        task_id,
    )
    done_identity = {
        "cell_id": cell_id,
        "input_variant": cell["input_variant"],
        "horizon_id": cell["horizon_id"],
        "in_channels": in_channels,
        "source_residual_sha256": input_fingerprint,
        "input_fingerprint": input_fingerprint,
        "input_support_sha256": fold["support_sha256"],
        "common_history_support_sha256": fold["support_sha256"],
        "initial_state_sha256": initial_hash,
    }
    for key, expected in done_identity.items():
        require(done.get(key) == expected, f"{task_id}/DONE/{key}: differs")

    expected_config = expected_classifier_config(config, cell, initial_hash)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    rf.validate_rf_checkpoint(
        best,
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        source_residual_sha256=input_fingerprint,
    )
    require(int(best.get("classifier_seed", -1)) == fold["classifier_seed"], f"{task_id}: best seed differs")
    require(best.get("variant") == cell_id, f"{task_id}: best variant differs")
    require(best.get("classifier_config") == expected_config, f"{task_id}: best classifier config differs")
    model = rf.build_model(
        in_channels=in_channels,
        hidden_channels=int(config["classifier"]["hidden_channels"]),
        dropout=float(config["classifier"]["dropout"]),
        dilations=EXPECTED_DILATIONS,
    ).cpu().eval()
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        logits = model(
            torch.zeros(
                2,
                in_channels,
                int(cell["history_samples"]),
                dtype=torch.float32,
            )
        )
    require(tuple(logits.shape) == (2,), f"{task_id}: classifier output shape differs")
    require(bool(torch.isfinite(logits).all()), f"{task_id}: non-finite logits")

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        last,
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
        input_fingerprint=input_fingerprint,
    )
    require(last.get("variant") == cell_id, f"{task_id}: last variant differs")
    require(int(last.get("classifier_seed", -1)) == fold["classifier_seed"], f"{task_id}: last seed differs")
    require(last.get("classifier_config") == expected_config, f"{task_id}: last classifier config differs")
    shuffle_seeds: list[int] = []
    for row in last["history"]:
        epoch = int(row["epoch"])
        shuffle_seed = int(row["shuffle_seed"])
        require(
            shuffle_seed == fold["classifier_seed"] + epoch,
            f"{task_id}: epoch {epoch} shuffle seed differs from rule",
        )
        shuffle_seeds.append(shuffle_seed)

    metrics = load_json(metrics_path)
    horizon_seconds = (
        None
        if cell["horizon_id"] == "shared"
        else EXPECTED_HORIZONS[str(cell["horizon_id"])]["seconds"]
    )
    metric_identity = {
        "experiment_id": cell["experiment_id"],
        "variant": cell_id,
        "cell_id": cell_id,
        "input": cell_id,
        "input_variant": cell["input_variant"],
        "input_kind": cell["kind"],
        "display_name": cell["display_name"],
        "formula": cell["formula"],
        "horizon_id": cell["horizon_id"],
        "horizon_seconds": horizon_seconds,
        "in_channels": in_channels,
        "nbm": "transformer",
        "history_seconds": float(cell["history_seconds"]),
        "history_samples": int(cell["history_samples"]),
        "history_blocks": int(cell["history_blocks"]),
        "test_subject": subject,
        "val_subject": fold["validation_subject"],
        "classifier_seed": fold["classifier_seed"],
        "classifier_config": expected_config,
        "initial_state_sha256": initial_hash,
        "source_residual_sha256": input_fingerprint,
        "input_fingerprint": input_fingerprint,
        "input_support_sha256": fold["support_sha256"],
        "common_history_support_sha256": fold["support_sha256"],
    }
    for key, expected in metric_identity.items():
        require(metrics.get(key) == expected, f"{task_id}/metrics/{key}: differs")

    inputs = materialize_cell_inputs(
        cell,
        fold,
        dataset,
        classification,
        primitives,
    )
    materialized_by_split = {
        split: {
            "input_sha256": suite.array_sha256(payload[cell_id]),
            "y_sha256": suite.array_sha256(payload["y"]),
            "window_index_sha256": suite.array_sha256(payload["window_index"]),
        }
        for split, payload in inputs.items()
    }
    materialized_hash = canonical_fingerprint(
        {
            "cell_id": cell_id,
            "input_fingerprint": input_fingerprint,
            "splits": materialized_by_split,
        }
    )
    require(metrics.get("materialized_input_sha256_by_split") == materialized_by_split, f"{task_id}: materialized split hashes differ")
    require(metrics.get("materialized_input_sha256") == materialized_hash, f"{task_id}: materialized aggregate hash differs")
    require(done.get("materialized_input_sha256_by_split") == materialized_by_split, f"{task_id}: DONE split hashes differ")
    require(done.get("materialized_input_sha256") == materialized_hash, f"{task_id}: DONE materialized hash differs")
    require(
        done.get("metrics_identity_sha256")
        == canonical_fingerprint(
            {
                "cell_id": metrics["cell_id"],
                "input_variant": metrics["input_variant"],
                "horizon_id": metrics["horizon_id"],
                "in_channels": metrics["in_channels"],
                "input_fingerprint": metrics["input_fingerprint"],
                "materialized_input_sha256": metrics["materialized_input_sha256"],
                "common_history_support_sha256": metrics[
                    "common_history_support_sha256"
                ],
            }
        ),
        f"{task_id}: DONE metric identity hash differs",
    )

    train_y = inputs["train"]["y"]
    counts = np.bincount(train_y, minlength=2).astype(int)
    require(np.array_equal(np.asarray(metrics["train_counts"], dtype=np.int64), counts), f"{task_id}: train class counts differ")
    assert_close(
        metrics["pos_weight"],
        min(math.sqrt(counts[0] / counts[1]), 6.0),
        f"{task_id}/pos_weight",
    )
    require(metrics.get("history") == last.get("history"), f"{task_id}: metrics/checkpoint histories differ")
    require(int(metrics.get("best_epoch", -1)) == int(best["best_epoch"]), f"{task_id}: best epoch differs")
    assert_close(metrics["best_validation_auprc"], best["best_validation_auprc"], f"{task_id}/best_validation_auprc")
    history = list(metrics["history"])
    if history:
        selected = history[0]
        selected_score = float(selected["validation_auprc"])
        for row in history[1:]:
            score = float(row["validation_auprc"])
            if score > selected_score + 1e-5:
                selected, selected_score = row, score
        require(int(metrics["best_epoch"]) == int(selected["epoch"]), f"{task_id}: best epoch is not validation-selected")

    predictions = {
        "validation": load_predictions(validation_path, f"{task_id}/validation"),
        "test": load_predictions(prediction_path, f"{task_id}/test"),
    }
    for split in ("validation", "test"):
        prediction = predictions[split]
        payload = inputs[split]
        require(np.array_equal(prediction["window_index"], payload["window_index"]), f"{task_id}/{split}: IDs differ")
        require(np.array_equal(prediction["y_true"], payload["y"]), f"{task_id}/{split}: labels differ")
        replayed = classifier_probabilities(model, payload[cell_id])
        require(
            np.allclose(
                replayed,
                prediction["y_prob"],
                rtol=float(tolerance),
                atol=float(tolerance),
            ),
            f"{task_id}/{split}: probability replay differs",
        )
    threshold = float(metrics["threshold"])
    require(0 <= threshold <= 1, f"{task_id}: threshold invalid")
    for split, prediction in predictions.items():
        require(
            np.array_equal(
                prediction["y_pred"],
                (prediction["y_prob"] >= threshold).astype(np.int8),
            ),
            f"{task_id}/{split}: threshold decisions differ",
        )
    selected_threshold, validation_metrics = choose_threshold(
        predictions["validation"]["y_true"],
        predictions["validation"]["y_prob"],
    )
    assert_close(threshold, selected_threshold, f"{task_id}/selected_threshold", rtol=0.0, atol=1e-12)
    assert_metric_dict(metrics["validation"], validation_metrics, f"{task_id}/validation")
    test_metrics = binary_metrics(
        predictions["test"]["y_true"],
        predictions["test"]["y_prob"],
        threshold,
    )
    assert_metric_dict(metrics, test_metrics, f"{task_id}/test")
    for key, expected in requested_metrics(test_metrics).items():
        assert_close(metrics.get(key), expected, f"{task_id}/{key}")
    events = event_metrics(
        dataset,
        classification,
        predictions["test"]["window_index"],
        predictions["test"]["y_pred"],
    )
    for key in EVENT_METRIC_KEYS:
        require(key in metrics, f"{task_id}: event metric {key} missing")
        if isinstance(events[key], (int, np.integer)):
            require(int(metrics[key]) == int(events[key]), f"{task_id}/{key}: differs")
        else:
            assert_close(metrics[key], events[key], f"{task_id}/{key}")

    prediction_rows = read_csv(prediction_csv_path)
    require(len(prediction_rows) == len(predictions["test"]["window_index"]), f"{task_id}: prediction CSV count differs")
    if prediction_rows:
        ids = predictions["test"]["window_index"]
        require(
            np.array_equal(
                np.asarray([int(row["y_true"]) for row in prediction_rows], dtype=np.int8),
                predictions["test"]["y_true"],
            )
            and np.array_equal(
                np.asarray([float(row["y_prob"]) for row in prediction_rows], dtype=np.float64),
                predictions["test"]["y_prob"],
            )
            and np.array_equal(
                np.asarray([int(row["y_pred"]) for row in prediction_rows], dtype=np.int8),
                predictions["test"]["y_pred"],
            ),
            f"{task_id}: prediction CSV values differ",
        )
        require(
            np.array_equal(
                np.asarray([int(row["target_start"]) for row in prediction_rows], dtype=np.int64),
                classification.target_start[ids],
            )
            and np.array_equal(
                np.asarray([int(row["target_end_exclusive"]) for row in prediction_rows], dtype=np.int64),
                classification.target_end[ids],
            ),
            f"{task_id}: prediction CSV intervals are not final 0.5 s",
        )
    return {
        "subject": subject,
        "cell_id": cell_id,
        "experiment_id": str(cell["experiment_id"]),
        "metrics": metrics,
        "predictions": predictions["test"],
        "validation_predictions": predictions["validation"],
        "initial_state_sha256": initial_hash,
        "in_channels": in_channels,
        "shuffle_seeds": shuffle_seeds,
    }


def validate_fold_done(
    root: Path,
    subject: str,
    config: Mapping[str, Any],
    *,
    completed_cell_ids: set[str],
) -> bool:
    fold_root = root / f"loso_{subject}"
    done_path = fold_root / "FOLD_DONE.json"
    expected_subject_cells = {
        f"{subject}/{cell_id}" for cell_id in EXPECTED_INPUTS
    }
    full_fold = expected_subject_cells <= completed_cell_ids
    if not full_fold:
        require(not done_path.exists(), f"{subject}: FOLD_DONE exists before all nine cells")
        return False
    done = validate_done(
        done_path,
        stage="transformer_horizon_fusion_fold",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/fold",
    )
    require(done is not None, f"{subject}: completed fold lacks FOLD_DONE")
    artifacts: dict[str, Path] = {
        "fold_config": fold_root / "fold_config.json",
        "scaler": fold_root / "scaler.json",
        "split_indices": fold_root / "split_indices.npz",
        "common_history_support": fold_root / "common_history_support.npz",
        "input_fingerprints": fold_root / "input_fingerprints.json",
    }
    for horizon in config["horizon_variants"]:
        horizon_id = str(horizon["horizon_id"])
        nbm_root = suite.nbm_root_for(root, subject, horizon)
        artifacts[f"{horizon_id}_nbm_done"] = nbm_root / "nbm" / "DONE.json"
        artifacts[f"{horizon_id}_primitive_done"] = nbm_root / "PRIMITIVE_CACHE_DONE.json"
    for cell_id in EXPECTED_INPUTS:
        artifacts[f"{cell_id}_classifier_done"] = (
            suite.task_root_for(root, subject, cell_id) / "DONE.json"
        )
    assert_done_artifacts(done, artifacts, done_path, f"{subject}/fold")
    fold_config = load_json(fold_root / "fold_config.json")
    require(done.get("test_subject") == subject, f"{subject}: FOLD_DONE subject differs")
    require(done.get("completed_horizons") == list(EXPECTED_HORIZONS), f"{subject}: FOLD_DONE horizons differ")
    require(done.get("completed_inputs") == list(EXPECTED_INPUTS), f"{subject}: FOLD_DONE inputs differ")
    require(
        done.get("common_history_support_sha256")
        == fold_config["common_history_support_sha256"],
        f"{subject}: FOLD_DONE support hash differs",
    )
    require(
        done.get("nbm_best_sha256_by_horizon")
        == fold_config["nbm_best_sha256_by_horizon"],
        f"{subject}: FOLD_DONE NBM hashes differ",
    )
    require(
        done.get("primitive_cache_sha256_by_horizon")
        == fold_config["primitive_cache_sha256_by_horizon"],
        f"{subject}: FOLD_DONE primitive hashes differ",
    )
    require(
        done.get("input_fingerprints") == fold_config["input_fingerprints"],
        f"{subject}: FOLD_DONE input fingerprints differ",
    )
    return True


def expected_paired_effects(
    config: Mapping[str, Any],
    completed: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_cell: dict[str, dict[str, Mapping[str, Any]]] = {
        cell_id: {} for cell_id in EXPECTED_INPUTS
    }
    for item in completed:
        rows_by_cell[str(item["cell_id"])][str(item["subject"])] = item["metrics"]
    result: list[dict[str, Any]] = []
    for comparison in config["comparisons"]:
        result.append(
            {
                **comparison,
                **suite._paired_effect(
                    rows_by_cell,
                    comparison,
                    list(EXPECTED_SUBJECTS),
                    int(config["bootstrap_samples"]),
                    int(config["bootstrap_seed"]),
                ),
            }
        )
    return result


def validate_root_summaries(
    root: Path,
    config: Mapping[str, Any],
    experiments: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    checked_fold_manifests: int,
) -> None:
    paths = {
        "fold_summary": root / "fold_summary.csv",
        "experiment_manifest": root / "experiment_manifest.csv",
        "aggregate_summary": root / "aggregate_summary.csv",
        "paired": root / "paired_pr_auc_deltas.csv",
        "publication": root / "publication_table.csv",
        "aggregate_metrics": root / "aggregate_metrics.json",
        "status": root / "status.json",
    }
    for label, path in paths.items():
        require(path.is_file(), f"root summary missing: {label}")
    by_cell: dict[str, list[dict[str, Any]]] = {
        cell_id: [] for cell_id in EXPECTED_INPUTS
    }
    for item in completed:
        by_cell[str(item["cell_id"])].append(item)
    for values in by_cell.values():
        values.sort(key=lambda item: EXPECTED_SUBJECTS.index(str(item["subject"])))
    completed_count = len(completed)
    completed_nbm = sum(
        int(
            (
                suite.nbm_root_for(root, subject, horizon)
                / "nbm"
                / "DONE.json"
            ).exists()
        )
        for subject in EXPECTED_SUBJECTS
        for horizon in config["horizon_variants"]
    )
    completed_primitives = sum(
        int(
            (
                suite.nbm_root_for(root, subject, horizon)
                / "PRIMITIVE_CACHE_DONE.json"
            ).exists()
        )
        for subject in EXPECTED_SUBJECTS
        for horizon in config["horizon_variants"]
    )
    full_complete = (
        completed_count == EXPECTED_CELLS
        and completed_nbm == 24
        and completed_primitives == 24
        and checked_fold_manifests == 8
    )

    status = load_json(paths["status"])
    expected_status = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_experiments": 9,
        "expected_nbm_tasks": 24,
        "completed_nbm_tasks": completed_nbm,
        "expected_primitive_cache_tasks": 24,
        "completed_primitive_cache_tasks": completed_primitives,
        "expected_classifier_cells": EXPECTED_CELLS,
        "completed_classifier_cells": completed_count,
        "expected_fold_manifests": 8,
        "completed_fold_manifests": checked_fold_manifests,
        "status": "complete" if full_complete else "partial",
    }
    for key, expected in expected_status.items():
        require(status.get(key) == expected, f"status/{key}: differs")

    aggregate = load_json(paths["aggregate_metrics"])
    aggregate_identity = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "aggregation_unit": "held_out_subject",
        "metric_dispersion": "population standard deviation across LOSO folds",
        "ranking_metric": "subject_macro_pr_auc_mean",
        "delta_pr_auc": {
            "method": "paired nonparametric bootstrap over held-out subjects",
            "confidence_level": 0.95,
            "samples": int(config["bootstrap_samples"]),
            "seed": int(config["bootstrap_seed"]),
        },
    }
    for key, expected in aggregate_identity.items():
        require(aggregate.get(key) == expected, f"aggregate_metrics/{key}: differs")
    saved_experiments = aggregate.get("experiments")
    require(
        isinstance(saved_experiments, Mapping)
        and set(saved_experiments)
        == {str(cell["experiment_id"]) for cell in experiments},
        "aggregate experiment registry differs",
    )

    macro_by_cell: dict[str, dict[str, Any]] = {}
    pooled_by_cell: dict[str, dict[str, Any] | None] = {}
    ranked: list[tuple[float, str]] = []
    for cell in experiments:
        cell_id = str(cell["cell_id"])
        rows = [item["metrics"] for item in by_cell[cell_id]]
        macro = (
            aggregate_fold_metrics(rows, list(suite.CLASSIFICATION_METRICS))
            if rows
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in suite.CLASSIFICATION_METRICS
            }
        )
        macro_by_cell[cell_id] = macro
        pooled = (
            suite._prediction_metrics(
                np.concatenate([item["predictions"]["y_true"] for item in by_cell[cell_id]]),
                np.concatenate([item["predictions"]["y_prob"] for item in by_cell[cell_id]]),
                np.concatenate([item["predictions"]["y_pred"] for item in by_cell[cell_id]]),
            )
            if rows
            else None
        )
        pooled_by_cell[cell_id] = pooled
        saved = saved_experiments[str(cell["experiment_id"])]
        for key, expected in cell.items():
            require(saved.get(key) == expected, f"{cell_id}/aggregate identity/{key}: differs")
        require(
            saved.get("completed_folds")
            == [str(item["subject"]) for item in by_cell[cell_id]],
            f"{cell_id}: completed folds differ",
        )
        mapping_close(saved.get("subject_macro"), macro, f"{cell_id}/subject_macro")
        mapping_close(saved.get("pooled"), pooled, f"{cell_id}/pooled")
        if macro["pr_auc"]["mean"] is not None:
            ranked.append((-float(macro["pr_auc"]["mean"]), cell_id))
    ranked.sort()
    expected_best = (
        suite.experiment_id(ranked[0][1])
        if full_complete and ranked
        else None
    )
    require(aggregate.get("best_experiment") == expected_best, "aggregate best experiment differs")
    require(status.get("best_experiment") == expected_best, "status best experiment differs")

    expected_effects = expected_paired_effects(config, completed)
    expected_effect_by_id = {
        str(item["comparison_id"]): item for item in expected_effects
    }
    saved_effects = {
        str(item["comparison_id"]): item
        for item in aggregate.get("paired_pr_auc_comparisons", [])
    }
    require(set(saved_effects) == set(expected_effect_by_id), "aggregate paired registry differs")
    for comparison_id, expected in expected_effect_by_id.items():
        mapping_close(saved_effects[comparison_id], expected, f"aggregate paired/{comparison_id}")

    fold_rows = read_csv(paths["fold_summary"])
    require(len(fold_rows) == completed_count, "fold_summary row count differs")
    fold_map = {
        (row["test_subject"], row["cell_id"]): row for row in fold_rows
    }
    require(len(fold_map) == len(fold_rows), "fold_summary has duplicate cells")
    require(
        set(fold_map)
        == {(str(item["subject"]), str(item["cell_id"])) for item in completed},
        "fold_summary cell registry differs",
    )
    cell_by_id = {str(cell["cell_id"]): cell for cell in experiments}
    for item in completed:
        identity = (str(item["subject"]), str(item["cell_id"]))
        row = fold_map[identity]
        metrics = item["metrics"]
        cell = cell_by_id[identity[1]]
        expected_strings = {
            "experiment_id": str(cell["experiment_id"]),
            "variant": identity[1],
            "cell_id": identity[1],
            "input": identity[1],
            "input_variant": str(cell["input_variant"]),
            "input_kind": str(cell["kind"]),
            "display_name": str(cell["display_name"]),
            "formula": str(cell["formula"]),
            "horizon_id": str(cell["horizon_id"]),
            "in_channels": str(cell["in_channels"]),
            "history_samples": str(cell["history_samples"]),
            "history_blocks": str(cell["history_blocks"]),
            "test_subject": identity[0],
            "input_fingerprint": str(metrics["input_fingerprint"]),
            "common_history_support_sha256": str(
                metrics["common_history_support_sha256"]
            ),
            "initial_state_sha256": str(metrics["initial_state_sha256"]),
        }
        for key, expected in expected_strings.items():
            require(row.get(key) == expected, f"fold_summary/{identity}/{key}: differs")
        for key in (
            "threshold",
            "n",
            "n_normal",
            "n_fog",
            "tn",
            "fp",
            "fn",
            "tp",
            *suite.CLASSIFICATION_METRICS,
        ):
            assert_close(row.get(key), metrics.get(key), f"fold_summary/{identity}/{key}")

    manifest_rows = read_csv(paths["experiment_manifest"])
    require(len(manifest_rows) == 9, "experiment_manifest must have nine rows")
    manifest = {row["cell_id"]: row for row in manifest_rows}
    require(set(manifest) == set(EXPECTED_INPUTS), "experiment_manifest registry differs")
    for cell in experiments:
        cell_id = str(cell["cell_id"])
        subjects = [str(item["subject"]) for item in by_cell[cell_id]]
        expected = {
            "experiment_id": str(cell["experiment_id"]),
            "cell_id": cell_id,
            "display_name": str(cell["display_name"]),
            "input_kind": str(cell["kind"]),
            "horizon_id": str(cell["horizon_id"]),
            "formula": str(cell["formula"]),
            "in_channels": str(cell["in_channels"]),
            "history_samples": str(cell["history_samples"]),
            "history_blocks": str(cell["history_blocks"]),
            "input_shape": str(cell["input_shape"]),
            "parameter_count": str(cell["parameter_count"]),
            "expected_folds": "8",
            "completed_folds": str(len(subjects)),
            "status": (
                "complete"
                if subjects == list(EXPECTED_SUBJECTS)
                else ("partial" if subjects else "pending")
            ),
            "completed_subjects": ",".join(subjects),
        }
        for key, value in expected.items():
            require(manifest[cell_id].get(key) == value, f"manifest/{cell_id}/{key}: differs")

    aggregate_rows = read_csv(paths["aggregate_summary"])
    require(len(aggregate_rows) == 9, "aggregate_summary must have nine rows")
    aggregate_map = {row["cell_id"]: row for row in aggregate_rows}
    require(set(aggregate_map) == set(EXPECTED_INPUTS), "aggregate_summary registry differs")
    expected_rank = [cell_id for _, cell_id in ranked]
    expected_rank.extend(
        sorted(set(EXPECTED_INPUTS) - set(expected_rank))
    )
    for rank, cell_id in enumerate(expected_rank, start=1):
        row = aggregate_map[cell_id]
        require(int(row["rank"]) == rank, f"aggregate_summary/{cell_id}: rank differs")
        require(int(row["completed_folds"]) == len(by_cell[cell_id]), f"aggregate_summary/{cell_id}: fold count differs")
        for metric in suite.CLASSIFICATION_METRICS:
            for statistic in ("mean", "std"):
                assert_close(
                    row.get(f"{metric}_{statistic}"),
                    macro_by_cell[cell_id][metric][statistic],
                    f"aggregate_summary/{cell_id}/{metric}_{statistic}",
                )

    paired_rows = read_csv(paths["paired"])
    paired_map = {row["comparison_id"]: row for row in paired_rows}
    require(len(paired_rows) == len(config["comparisons"]), "paired CSV row count differs")
    require(set(paired_map) == set(expected_effect_by_id), "paired CSV registry differs")
    for comparison_id, expected in expected_effect_by_id.items():
        observed = paired_map[comparison_id]
        for key, value in expected.items():
            if key in {
                "mean_delta",
                "ci_low",
                "ci_high",
                "n_paired_subjects",
                "wins",
                "ties",
                "losses",
                "bootstrap_samples",
                "bootstrap_seed",
            }:
                assert_close(observed.get(key), value, f"paired/{comparison_id}/{key}", rtol=1e-12, atol=1e-12)
            else:
                require(observed.get(key) == str(value), f"paired/{comparison_id}/{key}: differs")

    publication_rows = read_csv(paths["publication"])
    require(len(publication_rows) == 9, "publication_table must have nine rows")
    publication_map = {row["Input"]: row for row in publication_rows}
    require(
        set(publication_map)
        == {str(cell["display_name"]) for cell in experiments},
        "publication registry differs",
    )
    primary_effect: dict[str, Mapping[str, Any]] = {}
    for suffix in ("h050", "h100", "h200"):
        primary_effect[f"raw4_error_{suffix}"] = expected_effect_by_id[
            f"raw4_error_{suffix}_minus_raw4_zero"
        ]
        primary_effect[f"error_{suffix}"] = expected_effect_by_id[
            f"error_{suffix}_minus_raw4"
        ]
    publication_metrics = {
        "PR-AUC": "pr_auc",
        "BA": "balanced_accuracy",
        "Macro-F1": "macro_f1",
        "AUROC": "roc_auc",
        "FoG Sensitivity/Recall": "fog_recall",
        "Specificity": "specificity",
        "FoG Precision": "precision",
        "FoG F1": "fog_f1",
        "Event Sensitivity": "event_sensitivity",
        "FA/h": "false_alarm_events_per_hour",
        "Median Detection Delay": "median_detection_delay_sec",
    }
    for cell in experiments:
        cell_id = str(cell["cell_id"])
        row = publication_map[str(cell["display_name"])]
        require(row["Horizon"] == str(cell["horizon_id"]), f"publication/{cell_id}: horizon differs")
        require(row["Channels"] == str(cell["in_channels"]), f"publication/{cell_id}: channels differ")
        require(row["Shape"] == str(cell["input_shape"]), f"publication/{cell_id}: shape differs")
        require(row["Completed folds"] == str(len(by_cell[cell_id])), f"publication/{cell_id}: fold count differs")
        for column, metric in publication_metrics.items():
            require(
                row[column] == suite._format_mean_sd(macro_by_cell[cell_id], metric),
                f"publication/{cell_id}/{column}: differs",
            )
        require(
            row["Primary delta PR-AUC [95% CI]"]
            == suite._format_delta(primary_effect.get(cell_id)),
            f"publication/{cell_id}: primary delta differs",
        )

    results_path = root / "RESULTS_DONE.json"
    if full_complete:
        done = validate_done(
            results_path,
            stage="transformer_horizon_fusion_results",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id="root/results",
        )
        require(done is not None, "complete suite lacks RESULTS_DONE")
        assert_done_artifacts(
            done,
            {
                "config": root / "config.json",
                "run_manifest": root / "run_manifest.json",
                "fold_summary": paths["fold_summary"],
                "experiment_manifest": paths["experiment_manifest"],
                "aggregate_summary": paths["aggregate_summary"],
                "paired_pr_auc_deltas": paths["paired"],
                "publication_table": paths["publication"],
                "aggregate_metrics": paths["aggregate_metrics"],
                "status": paths["status"],
            },
            results_path,
            "root/results",
        )
    else:
        require(not results_path.exists(), "partial suite contains RESULTS_DONE")


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"Audit version: {report.get('audit_version')}",
        f"Status: {report.get('status')}",
        f"Checked cells: {report.get('checked_cells')}/{report.get('expected_cells')}",
        f"Checked NBM tasks: {report.get('checked_nbm_tasks')}/24",
        f"Checked primitive tasks: {report.get('checked_primitive_tasks')}/24",
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
    dataset, master, windows_by_horizon, classification = (
        rebuild_dataset_and_windows(config, data_dir)
    )
    primitive_tolerance = (
        float(args.primitive_tolerance)
        if args.primitive_tolerance is not None
        else (3e-2 if bool(config.get("amp")) else 2e-3)
    )
    prediction_tolerance = (
        float(args.prediction_tolerance)
        if args.prediction_tolerance is not None
        else (5e-3 if bool(config.get("amp")) else 3e-5)
    )
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "suite_version": suite.SUITE_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "data_dir": str(data_dir),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_cells": EXPECTED_CELLS,
        "checked_cells": 0,
        "checked_nbm_tasks": 0,
        "checked_primitive_tasks": 0,
        "checked_fold_manifests": 0,
        "missing_cells": [],
        "allow_partial": bool(args.allow_partial),
        "reportable": False,
        "full_complete": False,
        "failures": [],
        "warnings": [],
        "tolerances": {
            "sampled_transformer_primitive_replay": primitive_tolerance,
            "classifier_probability_replay": prediction_tolerance,
        },
    }
    completed: list[dict[str, Any]] = []
    complete_cell_ids: set[str] = set()
    validated_folds: dict[str, dict[str, Any]] = {}

    for subject in EXPECTED_SUBJECTS:
        fold_root = root / f"loso_{subject}"
        if not fold_root.is_dir() or not (fold_root / "fold_config.json").is_file():
            report["missing_cells"].extend(
                f"{subject}/{cell_id}" for cell_id in EXPECTED_INPUTS
            )
            if fold_root.exists() and not args.allow_partial:
                report["failures"].append(f"{subject}: fold_config.json missing")
            continue
        try:
            fold = validate_fold(
                root,
                subject,
                config,
                dataset,
                master,
                windows_by_horizon,
                classification,
            )
            validated_folds[subject] = fold
        except Exception as error:  # noqa: BLE001 - collect independent failures.
            report["failures"].append(
                f"{subject}/fold: {type(error).__name__}: {error}"
            )
            report["missing_cells"].extend(
                f"{subject}/{cell_id}" for cell_id in EXPECTED_INPUTS
            )
            continue

        primitives: dict[
            str,
            dict[str, dict[str, np.ndarray]],
        ] = {}
        shared_encoder_hashes: set[str] = set()
        horizon_stages_ok = True
        for horizon in config["horizon_variants"]:
            horizon_id = str(horizon["horizon_id"])
            nbm_root = suite.nbm_root_for(root, subject, horizon)
            nbm_done = nbm_root / "nbm" / "DONE.json"
            primitive_done = nbm_root / "PRIMITIVE_CACHE_DONE.json"
            if not nbm_done.exists() or not primitive_done.exists():
                horizon_stages_ok = False
                if primitive_done.exists() and not nbm_done.exists():
                    report["failures"].append(
                        f"{subject}/{horizon_id}: primitive DONE exists without NBM DONE"
                    )
                continue
            try:
                nbm_sha, model, nbm_metadata = validate_nbm_task(
                    root,
                    subject,
                    horizon,
                    config,
                    fold,
                )
                report["checked_nbm_tasks"] += 1
                shared_encoder_hashes.add(
                    str(nbm_metadata["initial_shared_encoder_sha256"])
                )
                _, features = validate_primitive_cache(
                    root,
                    subject,
                    horizon,
                    config,
                    fold,
                    dataset,
                    windows_by_horizon[horizon_id],
                    nbm_sha,
                    model,
                    tolerance=primitive_tolerance,
                )
                report["checked_primitive_tasks"] += 1
                primitives[horizon_id] = features
            except Exception as error:  # noqa: BLE001
                horizon_stages_ok = False
                report["failures"].append(
                    f"{subject}/{horizon_id}: {type(error).__name__}: {error}"
                )
        if horizon_stages_ok:
            require(
                len(shared_encoder_hashes) == 1,
                f"{subject}: Transformer shared initialization differs by horizon",
            )

        fold_cells: list[dict[str, Any]] = []
        for cell in experiments:
            cell_id = str(cell["cell_id"])
            identity = f"{subject}/{cell_id}"
            task_done = suite.task_root_for(root, subject, cell_id) / "DONE.json"
            if not task_done.exists():
                report["missing_cells"].append(identity)
                continue
            if not horizon_stages_ok or set(primitives) != set(EXPECTED_HORIZONS):
                report["failures"].append(
                    f"{identity}: classifier DONE exists without all audited Transformer primitives"
                )
                continue
            try:
                audited = audit_classifier_task(
                    root,
                    subject,
                    cell,
                    config,
                    fold,
                    dataset,
                    classification,
                    primitives,
                    tolerance=prediction_tolerance,
                )
                completed.append(audited)
                fold_cells.append(audited)
                complete_cell_ids.add(identity)
            except Exception as error:  # noqa: BLE001
                report["failures"].append(
                    f"{identity}: {type(error).__name__}: {error}"
                )

        # All cells share endpoints, labels, one seed rule, and a full initial
        # state within each equal-channel group. Different early-stopping
        # lengths are legitimate, so only the common epoch prefix is compared.
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
                    f"{subject}: classifier endpoints/labels differ",
                )
                common_epochs = min(
                    len(cell["shuffle_seeds"]),
                    len(reference["shuffle_seeds"]),
                )
                require(
                    cell["shuffle_seeds"][:common_epochs]
                    == reference["shuffle_seeds"][:common_epochs],
                    f"{subject}: epoch shuffle common prefix differs",
                )
            for in_channels in (9, 18):
                group = [
                    item
                    for item in fold_cells
                    if int(item["in_channels"]) == in_channels
                ]
                require(
                    len({item["initial_state_sha256"] for item in group}) <= 1,
                    f"{subject}: {in_channels}-channel initial states differ",
                )

    checked_fold_manifests = 0
    for subject in EXPECTED_SUBJECTS:
        if not (root / f"loso_{subject}").exists():
            continue
        try:
            checked_fold_manifests += int(
                validate_fold_done(
                    root,
                    subject,
                    config,
                    completed_cell_ids=complete_cell_ids,
                )
            )
        except Exception as error:  # noqa: BLE001
            report["failures"].append(
                f"{subject}/FOLD_DONE: {type(error).__name__}: {error}"
            )
    report["checked_fold_manifests"] = checked_fold_manifests

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
    report["missing_cells"] = sorted(set(report["missing_cells"]))
    full_complete = (
        len(completed) == EXPECTED_CELLS
        and report["checked_nbm_tasks"] == 24
        and report["checked_primitive_tasks"] == 24
        and checked_fold_manifests == 8
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
                "Suite is incomplete; --allow-partial is required for an interim audit"
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
                "expected_cells": EXPECTED_CELLS,
                "checked_cells": EXPECTED_CELLS,
                "checked_nbm_tasks": 24,
                "checked_primitive_tasks": 24,
                "checked_fold_manifests": 8,
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
    except Exception as error:  # noqa: BLE001 - always write an audit result.
        report = {
            "audit_version": AUDIT_VERSION,
            "suite_version": suite.SUITE_VERSION,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_dir": str(root),
            "expected_cells": EXPECTED_CELLS,
            "checked_cells": 0,
            "checked_nbm_tasks": 0,
            "checked_primitive_tasks": 0,
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
        f"[transformer-horizon-fusion-audit] status={report['status']} "
        f"cells={report.get('checked_cells', 0)}/{EXPECTED_CELLS} "
        f"nbm={report.get('checked_nbm_tasks', 0)}/24 "
        f"primitives={report.get('checked_primitive_tasks', 0)}/24 "
        f"folds={report.get('checked_fold_manifests', 0)}/8 "
        f"missing={len(report.get('missing_cells', []))} "
        f"failures={len(report.get('failures', []))}",
        flush=True,
    )
    if report.get("status") not in {"pass", "partial_pass"}:
        for failure in report.get("failures", []):
            print(
                f"[transformer-horizon-fusion-audit] FAIL {failure}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
