#!/usr/bin/env python
"""Audit the strict Daphnet 4-NBM x 4-context TCN-M LOSO suite.

The audit is intentionally independent of the training scheduler.  It verifies
the immutable protocol, every completed NBM/residual/classifier artifact chain,
the common fold-local history support, saved prediction arrays, recomputed
window-level metrics, and the root aggregate/status counts.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_nbm_context_tcnm_suite as suite
from cnbr_fog.evaluation import aggregate_fold_metrics, binary_metrics
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file


AUDIT_VERSION = "daphnet_nbm4_context4_h4_tcnm_audit.v1"
EXPECTED_NBMS = ("linear_ar", "gru", "tcn", "transformer")
EXPECTED_CONTEXTS = (
    ("C1", 1.0, 64, "context_c1_1s"),
    ("C2", 2.0, 128, "context_c2_2s"),
    ("C3", 3.0, 192, "context_c3_3s"),
    ("C4", 4.0, 256, "context_c4_4s"),
)
EXPECTED_FOLDS = tuple(suite.EXPECTED_LOSO_SUBJECTS)
EXPECTED_DILATIONS = (1, 2, 4, 8, 8, 8)
EXPECTED_RESIDUAL_KEYS = {
    f"{split}_{key}"
    for split in ("train", "validation", "test")
    for key in ("residual", "y", "window_index")
}
EXPECTED_SUPPORT_KEYS = {
    f"{split}_{kind}_window_index"
    for split in ("train", "validation", "test")
    for kind in ("anchor", "history")
}
EXPECTED_SPLIT_KEYS = {
    "train_window_index",
    "validation_window_index",
    "test_window_index",
    "normal_train_window_index",
    "normal_validation_window_index",
}
EXPECTED_PREDICTION_KEYS = {"window_index", "y_true", "y_prob", "y_pred"}
WINDOW_METRIC_KEYS = (
    "threshold",
    "n",
    "n_normal",
    "n_fog",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "auroc",
    "auprc",
    "tn",
    "fp",
    "fn",
    "tp",
    "confusion_matrix",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
)
RUNTIME_CONFIG_FIELDS = {
    "data_dir",
    "output_dir",
    "device",
    "resume",
    "num_workers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the strict Daphnet 4-NBM x 4-context residual_h4s "
            "TCN-M LOSO suite."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Completed or partially completed suite output directory.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing cells while still rejecting corrupt completed cells.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
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


def values_equal(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except (TypeError, ValueError):
            return left == right
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        left_value = float(left)
        right_value = float(right)
        if math.isnan(left_value) or math.isnan(right_value):
            return math.isnan(left_value) and math.isnan(right_value)
        return math.isclose(left_value, right_value, rel_tol=1e-8, abs_tol=atol)
    return left == right


def delta_values_equal(left: Any, right: Any) -> bool:
    """Compare independently recomputed paired-delta values strictly."""

    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        left_value = float(left)
        right_value = float(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            return left_value == right_value
        return math.isclose(
            left_value,
            right_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return left == right


def validate_mapping_values(
    report: dict[str, Any],
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    keys: tuple[str, ...] | list[str] | set[str] | None = None,
) -> None:
    selected = list(keys) if keys is not None else list(expected)
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
                f"{label}: no recomputed value",
                key=key,
            )
            continue
        if not values_equal(actual[key], expected[key]):
            add_issue(
                report,
                "failures",
                f"{label}: value mismatch",
                key=key,
                actual=actual[key],
                expected=expected[key],
            )


def validate_paired_delta_payload(
    report: dict[str, Any],
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    require(
        report,
        set(actual) == set(expected),
        f"{label}: key set mismatch",
        actual=sorted(str(key) for key in actual),
        expected=sorted(str(key) for key in expected),
    )
    for key, expected_value in expected.items():
        if key not in actual:
            continue
        if not delta_values_equal(actual[key], expected_value):
            add_issue(
                report,
                "failures",
                f"{label}: value mismatch",
                key=key,
                actual=actual[key],
                expected=expected_value,
            )


def artifact_path(done_path: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = done_path.parent / path
    return path.resolve()


def validate_done(
    report: dict[str, Any],
    done_path: Path,
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    artifact_names: set[str],
    upstream_sha256: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = suite.core.validate_done(
            done_path,
            stage=stage,
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=upstream_sha256,
        )
    except Exception as error:  # noqa: BLE001 - audit must collect all failures.
        add_issue(
            report,
            "failures",
            "DONE validation failed",
            path=str(done_path),
            error=f"{type(error).__name__}: {error}",
        )
        return None
    if payload is None:
        return None
    actual_names = set(payload.get("artifacts", {}))
    require(
        report,
        actual_names == artifact_names,
        "DONE artifact set mismatch",
        path=str(done_path),
        actual=sorted(actual_names),
        expected=sorted(artifact_names),
    )
    return payload


def done_artifact(
    report: dict[str, Any],
    done_path: Path,
    payload: Mapping[str, Any],
    name: str,
) -> Path | None:
    entry = payload.get("artifacts", {}).get(name)
    if not isinstance(entry, Mapping):
        add_issue(
            report,
            "failures",
            "DONE artifact is missing",
            path=str(done_path),
            artifact=name,
        )
        return None
    try:
        return artifact_path(done_path, entry)
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "DONE artifact path is invalid",
            path=str(done_path),
            artifact=name,
            error=f"{type(error).__name__}: {error}",
        )
        return None


def strict_protocol_audit(
    report: dict[str, Any],
    result_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    config_path = result_dir / "config.json"
    if not config_path.exists():
        add_issue(report, "failures", "Missing config.json", path=str(config_path))
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
    scientific = {
        key: value
        for key, value in config.items()
        if key not in RUNTIME_CONFIG_FIELDS and key != "protocol_fingerprint"
    }
    require(
        report,
        bool(protocol) and canonical_fingerprint(scientific) == protocol,
        "Config protocol fingerprint mismatch",
        actual=protocol,
        recomputed=canonical_fingerprint(scientific),
    )

    run_manifest_path = result_dir / "run_manifest.json"
    if require(
        report,
        run_manifest_path.exists(),
        "Missing run_manifest.json",
        path=str(run_manifest_path),
    ):
        try:
            manifest = load_json(run_manifest_path)
            expected_manifest = {
                key: value
                for key, value in config.items()
                if key not in RUNTIME_CONFIG_FIELDS
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

    require(
        report,
        tuple(config.get("folds_resolved", [])) == EXPECTED_FOLDS,
        "LOSO folds are not the canonical eight subjects",
        actual=config.get("folds_resolved"),
        expected=list(EXPECTED_FOLDS),
    )
    require(
        report,
        tuple(config.get("nbms_resolved", [])) == EXPECTED_NBMS,
        "NBM set/order is not the fixed four-model protocol",
        actual=config.get("nbms_resolved"),
        expected=list(EXPECTED_NBMS),
    )
    require(
        report,
        set(config.get("excluded_subjects", [])) == {"S04", "S10"},
        "Excluded subjects must be exactly S04 and S10",
        actual=config.get("excluded_subjects"),
    )
    require(
        report,
        config.get("protocol_scope")
        == "strict_4_nbm_x_4_context_x_8_fold",
        "Protocol scope is not the strict 4-NBM x 4-context x 8-fold suite",
        actual=config.get("protocol_scope"),
        expected="strict_4_nbm_x_4_context_x_8_fold",
    )
    require(report, int(config.get("sampling_rate_hz", -1)) == 64, "Sampling rate is not 64 Hz")
    require(report, int(config.get("n_channels", -1)) == 9, "Input channel count is not nine")
    require(report, int(config.get("support_context_samples", -1)) == 256, "Common support is not four seconds")
    require(report, int(config.get("horizon_samples", -1)) == 32, "Forecast horizon is not 32 samples")
    require(report, int(config.get("history_samples", -1)) == 256, "Residual history is not 256 samples")
    require(report, int(config.get("history_blocks", -1)) == 8, "Residual history is not eight blocks")
    require(report, config.get("history_name") == suite.HISTORY_NAME, "Unexpected history name")

    actual_contexts = list(config.get("context_variants", []))
    require(
        report,
        len(actual_contexts) == len(EXPECTED_CONTEXTS),
        "Context variant count is not four",
        actual=len(actual_contexts),
    )
    for position, expected in enumerate(EXPECTED_CONTEXTS):
        if position >= len(actual_contexts):
            break
        context_id, seconds, samples, directory = expected
        actual = actual_contexts[position]
        validate_mapping_values(
            report,
            actual,
            {
                "context_id": context_id,
                "context_seconds": seconds,
                "context_samples": samples,
                "directory": directory,
            },
            label=f"context_variants[{position}]",
        )

    classifier = config.get("classifier", {})
    if not isinstance(classifier, Mapping):
        add_issue(report, "failures", "Classifier config is not an object")
        classifier = {}
    expected_classifier = {
        "name": "tcn_m",
        "in_channels": 9,
        "hidden_channels": 48,
        "kernel_size": 3,
        "dilations": list(EXPECTED_DILATIONS),
        "n_blocks": 6,
        "convolutions_per_block": 2,
        "receptive_field_samples": 125,
        "parameter_count": 89329,
        "global_pooling": "mean_and_max_over_full_input",
    }
    validate_mapping_values(
        report,
        classifier,
        expected_classifier,
        label="classifier",
    )
    recomputed_rf = 1 + 2 * (3 - 1) * sum(EXPECTED_DILATIONS)
    require(report, recomputed_rf == 125, "Internal RF125 formula changed")

    configured_experiments = list(config.get("experiments", []))
    expected_experiments = suite.experiment_grid(
        list(EXPECTED_NBMS),
        [
            {
                "context_id": context_id,
                "context_seconds": seconds,
                "context_samples": samples,
                "directory": directory,
            }
            for context_id, seconds, samples, directory in EXPECTED_CONTEXTS
        ],
    )
    require(
        report,
        configured_experiments == expected_experiments,
        "Experiment grid is not the fixed 4 NBM x 4 context Cartesian product",
    )
    require(
        report,
        int(config.get("expected_experiments", -1)) == 16,
        "Config expected_experiments is not 16",
    )
    require(
        report,
        int(config.get("expected_fold_cells", -1)) == 128,
        "Config expected_fold_cells is not 128",
    )
    require(
        report,
        int(config.get("expected_nbm_tasks", -1)) == 128,
        "Config expected_nbm_tasks is not 128",
    )
    return config, expected_experiments


def load_fold_support(
    report: dict[str, Any],
    result_dir: Path,
    subject: str,
    protocol: str,
) -> dict[str, Any] | None:
    fold_root = result_dir / f"loso_{subject}"
    if not fold_root.exists():
        return None
    required_paths = {
        "fold_config": fold_root / "fold_config.json",
        "suite_fold_config": fold_root / "context_suite_fold_config.json",
        "split_indices": fold_root / "split_indices.npz",
        "history_support": fold_root / "history_support.npz",
    }
    if not all(path.exists() for path in required_paths.values()):
        for name, path in required_paths.items():
            require(
                report,
                path.exists(),
                "Fold support artifact is missing",
                subject=subject,
                artifact=name,
                path=str(path),
            )
        return None
    try:
        fold_config = load_json(required_paths["fold_config"])
        suite_fold_config = load_json(required_paths["suite_fold_config"])
        with np.load(required_paths["split_indices"], allow_pickle=False) as payload:
            require(
                report,
                set(payload.files) == EXPECTED_SPLIT_KEYS,
                "split_indices.npz key set mismatch",
                subject=subject,
                actual=sorted(payload.files),
            )
            split_indices = {
                split: np.asarray(payload[f"{split}_window_index"], dtype=np.int64)
                for split in ("train", "validation", "test")
            }
            normal_indices = {
                "train": np.asarray(
                    payload["normal_train_window_index"],
                    dtype=np.int64,
                ),
                "validation": np.asarray(
                    payload["normal_validation_window_index"],
                    dtype=np.int64,
                ),
            }
        with np.load(required_paths["history_support"], allow_pickle=False) as payload:
            require(
                report,
                set(payload.files) == EXPECTED_SUPPORT_KEYS,
                "history_support.npz key set mismatch",
                subject=subject,
                actual=sorted(payload.files),
            )
            support = {
                split: {
                    "anchor": np.asarray(
                        payload[f"{split}_anchor_window_index"],
                        dtype=np.int64,
                    ),
                    "history": np.asarray(
                        payload[f"{split}_history_window_index"],
                        dtype=np.int64,
                    ),
                }
                for split in ("train", "validation", "test")
            }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load fold support",
            subject=subject,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    require(
        report,
        fold_config.get("protocol_fingerprint") == protocol,
        "fold_config protocol mismatch",
        subject=subject,
    )
    validate_mapping_values(
        report,
        suite_fold_config,
        {
            "suite_version": suite.SUITE_VERSION,
            "protocol_fingerprint": protocol,
            "test_subject": subject,
            "horizon_samples": 32,
            "common_support_context_samples": 256,
        },
        label=f"{subject}/context_suite_fold_config",
    )
    require(
        report,
        fold_config.get("test_subject") == subject,
        "fold_config test subject mismatch",
        subject=subject,
    )
    require(
        report,
        suite_fold_config.get("val_subject") == fold_config.get("val_subject"),
        "Fold validation subject mismatch",
        subject=subject,
    )
    require(
        report,
        suite_fold_config.get("train_subjects") == fold_config.get("train_subjects"),
        "Fold training subject list mismatch",
        subject=subject,
    )
    require(
        report,
        int(suite_fold_config.get("normal_train_windows", -1))
        == len(normal_indices["train"]),
        "normal_train_windows mismatch",
        subject=subject,
    )
    require(
        report,
        int(suite_fold_config.get("normal_validation_windows", -1))
        == len(normal_indices["validation"]),
        "normal_validation_windows mismatch",
        subject=subject,
    )
    support_hash = sha256_file(required_paths["history_support"])
    require(
        report,
        suite_fold_config.get("history_support_sha256") == support_hash,
        "Fold history support hash mismatch",
        subject=subject,
    )

    for split in ("train", "validation", "test"):
        indices = split_indices[split]
        anchor = support[split]["anchor"]
        history = support[split]["history"]
        require(
            report,
            indices.ndim == 1 and len(np.unique(indices)) == len(indices),
            "Split window indices are not unique one-dimensional IDs",
            subject=subject,
            split=split,
        )
        require(
            report,
            anchor.ndim == 1 and len(np.unique(anchor)) == len(anchor),
            "History anchors are not unique one-dimensional IDs",
            subject=subject,
            split=split,
        )
        require(
            report,
            history.shape == (len(anchor), 8),
            "History support shape is not [anchors,8]",
            subject=subject,
            split=split,
            actual=list(history.shape),
        )
        if history.shape == (len(anchor), 8):
            require(
                report,
                np.array_equal(history[:, -1], anchor),
                "Final history block is not the anchor window",
                subject=subject,
                split=split,
            )
            require(
                report,
                np.isin(history, indices).all(),
                "History support references a window outside its split",
                subject=subject,
                split=split,
            )
        require(
            report,
            int(suite_fold_config.get("history_anchor_counts", {}).get(split, -1))
            == len(anchor),
            "history_anchor_counts mismatch",
            subject=subject,
            split=split,
        )
        require(
            report,
            int(suite_fold_config.get("split_window_counts", {}).get(split, -1))
            == len(indices),
            "split_window_counts mismatch",
            subject=subject,
            split=split,
        )

    return {
        "fold_root": fold_root,
        "fold_config": fold_config,
        "suite_fold_config": suite_fold_config,
        "split_indices": split_indices,
        "normal_indices": normal_indices,
        "support": support,
        "support_hash": support_hash,
    }


def labels_for_anchors(
    report: dict[str, Any],
    *,
    subject: str,
    cell: str,
    split: str,
    window_index: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray | None:
    lookup = {int(index): int(label) for index, label in zip(window_index, labels)}
    missing = [int(index) for index in anchors if int(index) not in lookup]
    if missing:
        add_issue(
            report,
            "failures",
            "Prediction anchors are absent from residual cache",
            subject=subject,
            cell=cell,
            split=split,
            missing_count=len(missing),
            first_missing=missing[:5],
        )
        return None
    return np.asarray([lookup[int(index)] for index in anchors], dtype=np.int8)


def validate_prediction_file(
    report: dict[str, Any],
    path: Path,
    *,
    subject: str,
    cell: str,
    split: str,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            require(
                report,
                set(payload.files) == EXPECTED_PREDICTION_KEYS,
                "Prediction key set mismatch",
                subject=subject,
                cell=cell,
                split=split,
                actual=sorted(payload.files),
            )
            arrays = {
                "window_index": np.asarray(payload["window_index"], dtype=np.int64),
                "y_true": np.asarray(payload["y_true"], dtype=np.int8),
                "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
                "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
            }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load prediction artifact",
            subject=subject,
            cell=cell,
            split=split,
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )
        return None
    lengths = {len(value) for value in arrays.values()}
    require(
        report,
        lengths == {len(expected_indices)},
        "Prediction arrays have inconsistent lengths",
        subject=subject,
        cell=cell,
        split=split,
        lengths=sorted(lengths),
        expected=len(expected_indices),
    )
    require(
        report,
        np.array_equal(arrays["window_index"], expected_indices),
        "Prediction window IDs differ from common history anchors",
        subject=subject,
        cell=cell,
        split=split,
    )
    require(
        report,
        np.array_equal(arrays["y_true"], expected_labels),
        "Prediction labels differ from residual-cache/common support labels",
        subject=subject,
        cell=cell,
        split=split,
    )
    require(
        report,
        np.isfinite(arrays["y_prob"]).all()
        and ((arrays["y_prob"] >= 0.0) & (arrays["y_prob"] <= 1.0)).all(),
        "Prediction probabilities are non-finite or outside [0,1]",
        subject=subject,
        cell=cell,
        split=split,
    )
    require(
        report,
        np.isin(arrays["y_true"], [0, 1]).all()
        and np.isin(arrays["y_pred"], [0, 1]).all(),
        "Prediction labels are not binary",
        subject=subject,
        cell=cell,
        split=split,
    )
    expected_pred = (arrays["y_prob"] >= float(threshold)).astype(np.int8)
    require(
        report,
        np.array_equal(arrays["y_pred"], expected_pred),
        "Saved predictions do not equal probability >= saved threshold",
        subject=subject,
        cell=cell,
        split=split,
    )
    return arrays


def recompute_window_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    metrics = binary_metrics(y_true, y_prob, threshold)
    return suite.rf.add_requested_metrics(metrics)


def audit_cell(
    report: dict[str, Any],
    result_dir: Path,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    subject: str,
    fold: Mapping[str, Any] | None,
    fold_baseline: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any] | None:
    protocol = str(config["protocol_fingerprint"])
    experiment_name = str(experiment["experiment_id"])
    cell = f"{subject}/{experiment_name}"
    nbm_root = suite.nbm_root_for(result_dir, subject, experiment)
    task_root = suite.task_root_for(result_dir, subject, experiment)
    nbm_done_path = nbm_root / "nbm" / "DONE.json"
    residual_done_path = nbm_root / "RESIDUAL_CACHE_DONE.json"
    classifier_done_path = task_root / "DONE.json"
    done_presence = {
        "nbm": nbm_done_path.exists(),
        "residual_cache": residual_done_path.exists(),
        "classifier": classifier_done_path.exists(),
    }
    if not all(done_presence.values()):
        report["missing_cells"].append(
            {
                "cell": cell,
                "subject": subject,
                "experiment_id": experiment_name,
                "missing_stages": [
                    name for name, present in done_presence.items() if not present
                ],
            }
        )
        if done_presence["residual_cache"] and not done_presence["nbm"]:
            add_issue(
                report,
                "failures",
                "Residual DONE exists without NBM DONE",
                cell=cell,
            )
        if done_presence["classifier"] and not done_presence["residual_cache"]:
            add_issue(
                report,
                "failures",
                "Classifier DONE exists without residual-cache DONE",
                cell=cell,
            )
    if not done_presence["nbm"]:
        return None
    if fold is None:
        add_issue(
            report,
            "failures",
            "Cell artifact exists without valid fold support",
            cell=cell,
        )
        return None

    context_directory = suite.context_task_directory(experiment, subject)
    nbm_name = str(experiment["nbm"])
    nbm_task_id = f"{context_directory}/{nbm_name}/nbm"
    nbm_done = validate_done(
        report,
        nbm_done_path,
        stage="nbm",
        protocol_fingerprint=protocol,
        task_id=nbm_task_id,
        artifact_names={"best", "last", "training"},
    )
    if nbm_done is None:
        return None
    best_path = done_artifact(report, nbm_done_path, nbm_done, "best")
    training_path = done_artifact(report, nbm_done_path, nbm_done, "training")
    if best_path is None or training_path is None:
        return None
    nbm_sha256 = sha256_file(best_path)
    try:
        training = load_json(training_path)
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load NBM training artifact",
            cell=cell,
            error=f"{type(error).__name__}: {error}",
        )
        return None
    model_config = training.get("model_config", {})
    validate_mapping_values(
        report,
        training,
        {
            "model_name": nbm_name,
            "train_windows": fold["suite_fold_config"]["normal_train_windows"],
            "validation_windows": fold["suite_fold_config"][
                "normal_validation_windows"
            ],
        },
        label=f"{cell}/NBM training",
    )
    if isinstance(model_config, Mapping):
        validate_mapping_values(
            report,
            model_config,
            {
                "name": nbm_name,
                "in_channels": 9,
                "horizon": 32,
            },
            label=f"{cell}/NBM model_config",
        )
        if nbm_name == "transformer":
            require(
                report,
                int(model_config.get("max_context_samples", -1))
                == int(experiment["context_samples"]),
                "Transformer maximum context differs from experiment context",
                cell=cell,
            )
        if nbm_name == "linear_ar":
            expected_order = int(
                round(float(config["linear_ar_seconds"]) * 64.0)
            )
            require(
                report,
                int(model_config.get("ar_order", -1)) == expected_order,
                "Linear-AR order differs from protocol",
                cell=cell,
            )
    else:
        add_issue(report, "failures", "NBM model_config is not an object", cell=cell)

    summary_path = nbm_root / "nbm_summary.json"
    if require(
        report,
        summary_path.exists(),
        "Missing nbm_summary.json",
        cell=cell,
        path=str(summary_path),
    ):
        try:
            summary = load_json(summary_path)
            validate_mapping_values(
                report,
                summary,
                {
                    "protocol_fingerprint": protocol,
                    "experiment_id": experiment_name,
                    "nbm": nbm_name,
                    "context_id": experiment["context_id"],
                    "context_seconds": experiment["context_seconds"],
                    "context_samples": experiment["context_samples"],
                    "nbm_sha256": nbm_sha256,
                    "normal_training": training,
                },
                label=f"{cell}/nbm_summary",
            )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot validate nbm_summary.json",
                cell=cell,
                error=f"{type(error).__name__}: {error}",
            )

    residual_task_id = f"{context_directory}/{nbm_name}/residual_cache"
    if not done_presence["residual_cache"]:
        return None
    residual_done = validate_done(
        report,
        residual_done_path,
        stage="residual_cache",
        protocol_fingerprint=protocol,
        task_id=residual_task_id,
        upstream_sha256=nbm_sha256,
        artifact_names={"cache", "diagnostics"},
    )
    if residual_done is None:
        return None
    cache_path = done_artifact(report, residual_done_path, residual_done, "cache")
    diagnostics_path = done_artifact(
        report,
        residual_done_path,
        residual_done,
        "diagnostics",
    )
    if cache_path is None or diagnostics_path is None:
        return None
    residual_sha256 = sha256_file(cache_path)
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            require(
                report,
                set(payload.files) == EXPECTED_RESIDUAL_KEYS,
                "Residual cache key set mismatch",
                cell=cell,
                actual=sorted(payload.files),
            )
            residual = {
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
                for split in ("train", "validation", "test")
            }
        diagnostics = load_json(diagnostics_path)
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load residual-cache artifacts",
            cell=cell,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    expected_labels_by_split: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        values = residual[split]
        expected_ids = fold["split_indices"][split]
        require(
            report,
            values["residual"].shape == (len(expected_ids), 9, 32),
            "Residual array shape mismatch",
            cell=cell,
            split=split,
            actual=list(values["residual"].shape),
            expected=[len(expected_ids), 9, 32],
        )
        require(
            report,
            np.isfinite(values["residual"]).all(),
            "Residual cache contains non-finite values",
            cell=cell,
            split=split,
        )
        require(
            report,
            values["y"].shape == (len(expected_ids),)
            and values["window_index"].shape == (len(expected_ids),),
            "Residual cache IDs/labels shape mismatch",
            cell=cell,
            split=split,
        )
        require(
            report,
            np.array_equal(values["window_index"], expected_ids),
            "Residual window IDs differ from fold split",
            cell=cell,
            split=split,
        )
        require(
            report,
            np.isin(values["y"], [0, 1]).all(),
            "Residual labels are not binary",
            cell=cell,
            split=split,
        )
        baseline = fold_baseline.get(split)
        if baseline is None:
            fold_baseline[split] = {
                "window_index": values["window_index"].copy(),
                "y": values["y"].copy(),
            }
        else:
            require(
                report,
                np.array_equal(values["window_index"], baseline["window_index"])
                and np.array_equal(values["y"], baseline["y"]),
                "Residual IDs/labels differ across the 16 cells in this fold",
                cell=cell,
                split=split,
            )
        expected_labels = labels_for_anchors(
            report,
            subject=subject,
            cell=cell,
            split=split,
            window_index=values["window_index"],
            labels=values["y"],
            anchors=fold["support"][split]["anchor"],
        )
        if expected_labels is None:
            return None
        expected_labels_by_split[split] = expected_labels
        split_diagnostics = diagnostics.get(split, {})
        if isinstance(split_diagnostics, Mapping):
            validate_mapping_values(
                report,
                split_diagnostics,
                {
                    "windows": len(expected_ids),
                    "class_counts": np.bincount(
                        values["y"],
                        minlength=2,
                    ).astype(int).tolist(),
                    "context_samples": experiment["context_samples"],
                    "horizon_samples": 32,
                },
                label=f"{cell}/{split} residual diagnostics",
            )
        else:
            add_issue(
                report,
                "failures",
                "Residual diagnostics split is not an object",
                cell=cell,
                split=split,
            )

    classifier_task_id = f"{subject}/{experiment_name}"
    if not done_presence["classifier"]:
        return None
    classifier_done = validate_done(
        report,
        classifier_done_path,
        stage="rf_classifier",
        protocol_fingerprint=protocol,
        task_id=classifier_task_id,
        artifact_names={
            "best",
            "last",
            "metrics",
            "predictions",
            "validation_predictions",
            "predictions_csv",
        },
    )
    if classifier_done is None:
        return None
    require(
        report,
        classifier_done.get("source_residual_sha256") == residual_sha256,
        "Classifier DONE source residual hash mismatch",
        cell=cell,
    )
    require(
        report,
        classifier_done.get("input_support_sha256") == fold["support_hash"],
        "Classifier DONE input support hash mismatch",
        cell=cell,
    )
    require(
        report,
        classifier_done.get("initial_state_sha256")
        == fold["suite_fold_config"]["classifier_initial_state_sha256"],
        "Classifier initial-state hash differs within fold",
        cell=cell,
    )
    metrics_path = done_artifact(
        report,
        classifier_done_path,
        classifier_done,
        "metrics",
    )
    predictions_path = done_artifact(
        report,
        classifier_done_path,
        classifier_done,
        "predictions",
    )
    validation_predictions_path = done_artifact(
        report,
        classifier_done_path,
        classifier_done,
        "validation_predictions",
    )
    if (
        metrics_path is None
        or predictions_path is None
        or validation_predictions_path is None
    ):
        return None
    try:
        metrics = load_json(metrics_path)
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load classifier metrics",
            cell=cell,
            error=f"{type(error).__name__}: {error}",
        )
        return None
    validate_mapping_values(
        report,
        metrics,
        {
            "experiment_id": experiment_name,
            "variant": experiment_name,
            "nbm": nbm_name,
            "input": suite.HISTORY_NAME,
            "history_seconds": 4.0,
            "history_samples": 256,
            "history_blocks": 8,
            "test_subject": subject,
            "val_subject": fold["fold_config"]["val_subject"],
            "classifier_seed": fold["suite_fold_config"]["classifier_seed"],
            "source_residual_sha256": residual_sha256,
            "input_support_sha256": fold["support_hash"],
            "initial_state_sha256": fold["suite_fold_config"][
                "classifier_initial_state_sha256"
            ],
        },
        label=f"{cell}/metrics",
    )
    classifier_config = metrics.get("classifier_config", {})
    if isinstance(classifier_config, Mapping):
        validate_mapping_values(
            report,
            classifier_config,
            {
                "in_channels": 9,
                "hidden_channels": 48,
                "kernel_size": 3,
                "dilations": list(EXPECTED_DILATIONS),
                "n_blocks": 6,
                "convolutions_per_block": 2,
                "receptive_field_samples": 125,
                "parameter_count": 89329,
            },
            label=f"{cell}/classifier_config",
        )
    else:
        add_issue(
            report,
            "failures",
            "classifier_config is not an object",
            cell=cell,
        )
    threshold = float(metrics.get("threshold", float("nan")))
    require(
        report,
        math.isfinite(threshold) and 0.0 <= threshold <= 1.0,
        "Classifier threshold is invalid",
        cell=cell,
        threshold=threshold,
    )
    test_arrays = validate_prediction_file(
        report,
        predictions_path,
        subject=subject,
        cell=cell,
        split="test",
        expected_indices=fold["support"]["test"]["anchor"],
        expected_labels=expected_labels_by_split["test"],
        threshold=threshold,
    )
    validation_arrays = validate_prediction_file(
        report,
        validation_predictions_path,
        subject=subject,
        cell=cell,
        split="validation",
        expected_indices=fold["support"]["validation"]["anchor"],
        expected_labels=expected_labels_by_split["validation"],
        threshold=threshold,
    )
    if test_arrays is None or validation_arrays is None:
        return None
    recomputed_test = recompute_window_metrics(
        test_arrays["y_true"],
        test_arrays["y_prob"],
        threshold,
    )
    validate_mapping_values(
        report,
        metrics,
        recomputed_test,
        label=f"{cell}/test window metrics",
        keys=WINDOW_METRIC_KEYS,
    )
    saved_validation = metrics.get("validation", {})
    if isinstance(saved_validation, Mapping):
        recomputed_validation = binary_metrics(
            validation_arrays["y_true"],
            validation_arrays["y_prob"],
            threshold,
        )
        validate_mapping_values(
            report,
            saved_validation,
            recomputed_validation,
            label=f"{cell}/validation window metrics",
            keys=set(recomputed_validation),
        )
    else:
        add_issue(
            report,
            "failures",
            "Saved validation metrics are not an object",
            cell=cell,
        )
    return {
        "subject": subject,
        "experiment_id": experiment_name,
        "metrics": metrics,
        "recomputed_test": recomputed_test,
        "test": test_arrays,
        "validation": validation_arrays,
    }


def recompute_paired_pr_auc_deltas(
    config: Mapping[str, Any],
    experiments: list[dict[str, Any]],
    by_experiment: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Reproduce paired subject-level PR-AUC deltas from audited predictions."""

    rows_by_experiment = {
        experiment_id: {
            str(cell["subject"]): cell
            for cell in cells
        }
        for experiment_id, cells in by_experiment.items()
    }
    expected: dict[str, dict[str, Any]] = {}
    for item in experiments:
        experiment_id = str(item["experiment_id"])
        reference_item = next(
            (
                candidate
                for candidate in experiments
                if candidate["nbm"] == item["nbm"]
                and candidate["context_id"] == "C2"
            ),
            None,
        )
        reference_id = (
            str(reference_item["experiment_id"])
            if reference_item is not None
            else ""
        )
        common_subjects: list[str] = []
        differences: list[float] = []
        if reference_id:
            for subject in EXPECTED_FOLDS:
                current = rows_by_experiment[experiment_id].get(subject)
                reference = rows_by_experiment[reference_id].get(subject)
                if current is None or reference is None:
                    continue
                current_value = current["recomputed_test"].get("pr_auc")
                reference_value = reference["recomputed_test"].get("pr_auc")
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
                f"{experiment_id}__vs__{reference_id}",
            ),
        )
        expected[experiment_id] = {
            "experiment_id": experiment_id,
            "reference_experiment_id": reference_id,
            "reference_definition": "same NBM at C2 (2 s context)",
            "common_subjects": ",".join(common_subjects),
            **delta,
        }
    return expected


def audit_paired_delta_csv(
    report: dict[str, Any],
    path: Path,
    rows: list[dict[str, str]],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_columns = {
        "experiment_id",
        "reference_experiment_id",
        "reference_definition",
        "common_subjects",
        "mean_delta",
        "ci_low",
        "ci_high",
        "n_paired_subjects",
        "bootstrap_samples",
    }
    for row_index, row in enumerate(rows, start=2):
        require(
            report,
            set(row) == expected_columns,
            "paired_pr_auc_deltas.csv column set mismatch",
            path=str(path),
            row=row_index,
            actual=sorted(str(key) for key in row),
            expected=sorted(expected_columns),
        )

    experiment_ids = [str(row.get("experiment_id", "")) for row in rows]
    require(
        report,
        len(experiment_ids) == len(set(experiment_ids)),
        "paired_pr_auc_deltas.csv has duplicate experiment IDs",
        path=str(path),
        experiment_ids=experiment_ids,
    )
    require(
        report,
        set(experiment_ids) == set(expected),
        "paired_pr_auc_deltas.csv experiment ID set mismatch",
        path=str(path),
        actual=sorted(set(experiment_ids)),
        expected=sorted(expected),
    )
    rows_by_id = {
        str(row.get("experiment_id", "")): row
        for row in rows
    }
    for experiment_id, expected_payload in expected.items():
        row = rows_by_id.get(experiment_id)
        if row is None:
            continue
        parsed: dict[str, Any] = {
            "experiment_id": str(row.get("experiment_id", "")),
            "reference_experiment_id": str(
                row.get("reference_experiment_id", "")
            ),
            "reference_definition": str(
                row.get("reference_definition", "")
            ),
            "common_subjects": str(row.get("common_subjects", "")),
        }
        try:
            for key in ("mean_delta", "ci_low", "ci_high"):
                raw_value = row.get(key, "")
                parsed[key] = (
                    None
                    if raw_value is None or raw_value == ""
                    else float(raw_value)
                )
            parsed["n_paired_subjects"] = int(
                str(row.get("n_paired_subjects", ""))
            )
            parsed["bootstrap_samples"] = int(
                str(row.get("bootstrap_samples", ""))
            )
        except (TypeError, ValueError) as error:
            add_issue(
                report,
                "failures",
                "paired_pr_auc_deltas.csv contains an invalid numeric value",
                path=str(path),
                experiment_id=experiment_id,
                error=f"{type(error).__name__}: {error}",
            )
            continue
        validate_paired_delta_payload(
            report,
            parsed,
            expected_payload,
            label=f"paired_pr_auc_deltas.csv/{experiment_id}",
        )


def audit_root_summaries(
    report: dict[str, Any],
    result_dir: Path,
    config: Mapping[str, Any],
    experiments: list[dict[str, Any]],
    completed: list[dict[str, Any]],
) -> None:
    protocol = str(config["protocol_fingerprint"])
    by_experiment: dict[str, list[dict[str, Any]]] = {
        str(item["experiment_id"]): [] for item in experiments
    }
    for cell in completed:
        by_experiment[cell["experiment_id"]].append(cell)
    for cells in by_experiment.values():
        cells.sort(key=lambda item: EXPECTED_FOLDS.index(item["subject"]))
    expected_paired_deltas = recompute_paired_pr_auc_deltas(
        config,
        experiments,
        by_experiment,
    )

    aggregate_path = result_dir / "aggregate_metrics.json"
    status_path = result_dir / "status.json"
    if not require(
        report,
        aggregate_path.exists(),
        "Missing aggregate_metrics.json",
        path=str(aggregate_path),
    ):
        aggregate = None
    else:
        try:
            aggregate = load_json(aggregate_path)
        except Exception as error:  # noqa: BLE001
            aggregate = None
            add_issue(
                report,
                "failures",
                "Cannot load aggregate_metrics.json",
                error=f"{type(error).__name__}: {error}",
            )
    if not require(
        report,
        status_path.exists(),
        "Missing status.json",
        path=str(status_path),
    ):
        status = None
    else:
        try:
            status = load_json(status_path)
        except Exception as error:  # noqa: BLE001
            status = None
            add_issue(
                report,
                "failures",
                "Cannot load status.json",
                error=f"{type(error).__name__}: {error}",
            )

    completed_count = len(completed)
    expected_state = "complete" if completed_count == 128 else "partial"
    if status is not None:
        validate_mapping_values(
            report,
            status,
            {
                "suite_version": suite.SUITE_VERSION,
                "protocol_fingerprint": protocol,
                "expected_experiments": 16,
                "expected_nbm_tasks": 128,
                "expected_classifier_cells": 128,
                "completed_classifier_cells": completed_count,
                "status": expected_state,
            },
            label="status.json",
        )

    recomputed_best: str | None = None
    ranked: list[tuple[float, str]] = []
    if aggregate is not None:
        validate_mapping_values(
            report,
            aggregate,
            {
                "suite_version": suite.SUITE_VERSION,
                "protocol_fingerprint": protocol,
                "aggregation_unit": "held_out_subject",
                "ranking_metric": "subject_macro_pr_auc_mean",
            },
            label="aggregate_metrics.json",
        )
        saved_experiments = aggregate.get("experiments", {})
        require(
            report,
            isinstance(saved_experiments, Mapping)
            and set(saved_experiments) == set(by_experiment),
            "Aggregate experiment key set mismatch",
            actual=(
                sorted(saved_experiments)
                if isinstance(saved_experiments, Mapping)
                else None
            ),
        )
        if isinstance(saved_experiments, Mapping):
            for experiment in experiments:
                experiment_id = str(experiment["experiment_id"])
                saved = saved_experiments.get(experiment_id)
                if not isinstance(saved, Mapping):
                    continue
                cells = by_experiment[experiment_id]
                completed_subjects = [cell["subject"] for cell in cells]
                require(
                    report,
                    saved.get("completed_folds") == completed_subjects,
                    "Aggregate completed-fold list mismatch",
                    experiment_id=experiment_id,
                    actual=saved.get("completed_folds"),
                    expected=completed_subjects,
                )
                rows = [cell["metrics"] for cell in cells]
                recomputed_macro = (
                    aggregate_fold_metrics(
                        rows,
                        list(suite.CLASSIFICATION_METRICS),
                    )
                    if rows
                    else {
                        metric: {"mean": None, "std": None, "n_folds": 0}
                        for metric in suite.CLASSIFICATION_METRICS
                    }
                )
                saved_macro = saved.get("subject_macro", {})
                if isinstance(saved_macro, Mapping):
                    for metric, expected in recomputed_macro.items():
                        actual = saved_macro.get(metric)
                        if not isinstance(actual, Mapping):
                            add_issue(
                                report,
                                "failures",
                                "Aggregate subject-macro metric is missing",
                                experiment_id=experiment_id,
                                metric=metric,
                            )
                            continue
                        validate_mapping_values(
                            report,
                            actual,
                            expected,
                            label=f"{experiment_id}/subject_macro/{metric}",
                            keys=set(expected),
                        )
                else:
                    add_issue(
                        report,
                        "failures",
                        "Aggregate subject_macro is not an object",
                        experiment_id=experiment_id,
                    )
                saved_delta = saved.get("delta_pr_auc_vs_same_nbm_c2")
                if isinstance(saved_delta, Mapping):
                    validate_paired_delta_payload(
                        report,
                        saved_delta,
                        expected_paired_deltas[experiment_id],
                        label=(
                            "aggregate_metrics.json/"
                            f"{experiment_id}/delta_pr_auc_vs_same_nbm_c2"
                        ),
                    )
                else:
                    add_issue(
                        report,
                        "failures",
                        "Aggregate paired PR-AUC delta is missing",
                        experiment_id=experiment_id,
                    )
                if rows:
                    pr_mean = recomputed_macro["pr_auc"]["mean"]
                    if pr_mean is not None:
                        ranked.append((-float(pr_mean), experiment_id))
                    pooled = suite.prediction_metrics(
                        np.concatenate([cell["test"]["y_true"] for cell in cells]),
                        np.concatenate([cell["test"]["y_prob"] for cell in cells]),
                        np.concatenate([cell["test"]["y_pred"] for cell in cells]),
                    )
                    saved_pooled = saved.get("pooled")
                    if isinstance(saved_pooled, Mapping):
                        validate_mapping_values(
                            report,
                            saved_pooled,
                            pooled,
                            label=f"{experiment_id}/pooled",
                            keys=set(pooled),
                        )
                    else:
                        add_issue(
                            report,
                            "failures",
                            "Aggregate pooled metrics are missing",
                            experiment_id=experiment_id,
                        )
                else:
                    require(
                        report,
                        saved.get("pooled") is None,
                        "Pending experiment has non-null pooled metrics",
                        experiment_id=experiment_id,
                    )
        if ranked:
            ranked.sort()
            recomputed_best = ranked[0][1]
        require(
            report,
            aggregate.get("best_experiment") == recomputed_best,
            "Aggregate best_experiment mismatch",
            actual=aggregate.get("best_experiment"),
            expected=recomputed_best,
        )
    if status is not None:
        require(
            report,
            status.get("best_experiment") == recomputed_best,
            "status.json best_experiment mismatch",
            actual=status.get("best_experiment"),
            expected=recomputed_best,
        )

    csv_expectations = {
        "experiment_manifest.csv": 16,
        "aggregate_summary.csv": 16,
        "paired_pr_auc_deltas.csv": 16,
        "publication_table.csv": 16,
        "fold_summary.csv": completed_count,
    }
    for name, expected_rows in csv_expectations.items():
        path = result_dir / name
        if not require(report, path.exists(), "Root summary CSV is missing", path=str(path)):
            continue
        try:
            rows = load_csv(path)
            require(
                report,
                len(rows) == expected_rows,
                "Root summary CSV row count mismatch",
                path=str(path),
                actual=len(rows),
                expected=expected_rows,
            )
            if name == "experiment_manifest.csv":
                manifest_completed = sum(
                    int(row.get("completed_folds", "0") or 0) for row in rows
                )
                require(
                    report,
                    manifest_completed == completed_count,
                    "Manifest completed-fold total mismatch",
                    actual=manifest_completed,
                    expected=completed_count,
                )
            elif name == "paired_pr_auc_deltas.csv":
                audit_paired_delta_csv(
                    report,
                    path,
                    rows,
                    expected_paired_deltas,
                )
        except Exception as error:  # noqa: BLE001
            add_issue(
                report,
                "failures",
                "Cannot validate root summary CSV",
                path=str(path),
                error=f"{type(error).__name__}: {error}",
            )


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"Audit version: {report['audit_version']}",
        f"Result directory: {report['result_dir']}",
        f"Status: {report['status']}",
        f"Allow partial: {report['allow_partial']}",
        f"Expected cells: {report['counts']['expected_cells']}",
        f"Completed valid cells: {report['counts']['completed_valid_cells']}",
        f"Missing cells: {report['counts']['missing_cells']}",
        f"Failures: {len(report['failures'])}",
        f"Warnings: {len(report['warnings'])}",
        "",
    ]
    if report["failures"]:
        lines.append("FAILURES")
        for index, failure in enumerate(report["failures"], start=1):
            lines.append(f"{index}. {json.dumps(failure, ensure_ascii=False)}")
        lines.append("")
    if report["warnings"]:
        lines.append("WARNINGS")
        for index, warning in enumerate(report["warnings"], start=1):
            lines.append(f"{index}. {json.dumps(warning, ensure_ascii=False)}")
        lines.append("")
    if report["missing_cells"]:
        lines.append("MISSING CELLS")
        for item in report["missing_cells"]:
            lines.append(json.dumps(item, ensure_ascii=False))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Result directory does not exist: {result_dir}")
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(result_dir),
        "allow_partial": bool(args.allow_partial),
        "status": "running",
        "counts": {
            "expected_nbms": 4,
            "expected_contexts": 4,
            "expected_experiments": 16,
            "expected_folds": 8,
            "expected_cells": 128,
            "completed_valid_cells": 0,
            "missing_cells": 0,
        },
        "failures": [],
        "warnings": [],
        "missing_cells": [],
    }

    config, experiments = strict_protocol_audit(report, result_dir)
    completed: list[dict[str, Any]] = []
    if config is not None and len(experiments) == 16:
        protocol = str(config["protocol_fingerprint"])
        fold_data = {
            subject: load_fold_support(report, result_dir, subject, protocol)
            for subject in EXPECTED_FOLDS
        }
        fold_baselines: dict[str, dict[str, dict[str, np.ndarray]]] = {
            subject: {} for subject in EXPECTED_FOLDS
        }
        for subject in EXPECTED_FOLDS:
            for experiment in experiments:
                try:
                    audited = audit_cell(
                        report,
                        result_dir,
                        config,
                        experiment,
                        subject,
                        fold_data[subject],
                        fold_baselines[subject],
                    )
                    if audited is not None:
                        completed.append(audited)
                except Exception as error:  # noqa: BLE001
                    add_issue(
                        report,
                        "failures",
                        "Unexpected cell-audit exception",
                        subject=subject,
                        experiment_id=experiment.get("experiment_id"),
                        error=f"{type(error).__name__}: {error}",
                    )
        audit_root_summaries(
            report,
            result_dir,
            config,
            experiments,
            completed,
        )

    report["counts"]["completed_valid_cells"] = len(completed)
    report["counts"]["missing_cells"] = len(report["missing_cells"])
    if report["missing_cells"] and not args.allow_partial:
        add_issue(
            report,
            "failures",
            "Suite is incomplete; rerun with --allow-partial only for an interim audit",
            missing_cells=len(report["missing_cells"]),
        )
    if report["failures"]:
        report["status"] = "fail"
        exit_code = 1
    elif report["missing_cells"]:
        report["status"] = "partial_pass"
        exit_code = 0
    else:
        report["status"] = "pass"
        exit_code = 0

    atomic_json_dump(report, result_dir / "audit_report.json")
    write_text_report(result_dir / "audit_report.txt", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                **report["counts"],
                "failures": len(report["failures"]),
                "warnings": len(report["warnings"]),
                "json_report": str(result_dir / "audit_report.json"),
                "text_report": str(result_dir / "audit_report.txt"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
