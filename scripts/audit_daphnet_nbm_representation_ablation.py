#!/usr/bin/env python
"""Independent audit for the 4-NBM x 3-representation Daphnet suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_nbm_representation_ablation as suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.nbm_representations import calibrate_fixed_sigma
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    sha256_file,
    validate_done,
)


AUDIT_VERSION = "daphnet_nbm4_representation3_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Daphnet 4-NBM x 3-representation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--source-suite-dir", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _values_equal(
    left: Any,
    right: Any,
    *,
    tolerance: float = 1e-8,
) -> bool:
    """Compare nested metric payloads while tolerating float serialization."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(
            _values_equal(left[key], right[key], tolerance=tolerance)
            for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right,
            (list, tuple),
        ):
            return False
        return len(left) == len(right) and all(
            _values_equal(a, b, tolerance=tolerance)
            for a, b in zip(left, right)
        )
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        return _close(left, right, tolerance)
    return left == right


def _source_cache_upstream(
    config: Mapping[str, Any],
    subject: str,
    nbm: str,
) -> str:
    fold = config["source"]["folds"][subject]
    model = fold["models"][nbm]
    return canonical_fingerprint(
        {
            "source_nbm_best_sha256": model["source_nbm_best_sha256"],
            "source_residual_cache_sha256": model[
                "source_residual_cache_sha256"
            ],
            "source_residual_done_sha256": model[
                "source_residual_done_sha256"
            ],
            "source_scaler_sha256": fold["source_scaler_sha256"],
            "source_split_indices_sha256": fold[
                "source_split_indices_sha256"
            ],
        }
    )


def _expected_comparisons(
    fold_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    definitions = (
        (
            "fixed_minus_error",
            "fixed_standardized_error",
            "error_x_minus_mu",
        ),
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
    for nbm in suite.NBMS:
        for label, new, reference in definitions:
            new_rows = fold_metrics[suite.cell_id(nbm, new)]
            reference_rows = fold_metrics[suite.cell_id(nbm, reference)]
            subjects = [
                subject
                for subject in suite.EXPECTED_LOSO_SUBJECTS
                if subject in new_rows and subject in reference_rows
            ]
            differences = np.asarray(
                [
                    float(new_rows[subject]["pr_auc"])
                    - float(reference_rows[subject]["pr_auc"])
                    for subject in subjects
                ],
                dtype=np.float64,
            )
            effect = suite.input_ablation.paired_bootstrap_mean_ci(
                differences,
                int(config["bootstrap_samples"]),
                suite.input_ablation.stable_bootstrap_seed(
                    int(config["bootstrap_seed"]),
                    f"{nbm}/{label}",
                ),
            )
            result.append(
                {
                    "comparison_id": f"{nbm}__{label}",
                    "nbm": nbm,
                    "new": new,
                    "reference": reference,
                    "common_subjects": ",".join(subjects),
                    **effect,
                }
            )
    return result


def _prediction_threshold_failures(
    metrics: Mapping[str, Any],
    test_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    expected_validation_index: np.ndarray,
    expected_validation_y: np.ndarray,
) -> list[str]:
    """Validate that the saved classifier threshold came only from validation."""

    issues: list[str] = []
    validation_index = np.asarray(
        validation_arrays["window_index"],
        dtype=np.int64,
    )
    validation_true = np.asarray(
        validation_arrays["y_true"],
        dtype=np.int8,
    )
    validation_prob = np.asarray(
        validation_arrays["y_prob"],
        dtype=np.float64,
    )
    validation_pred = np.asarray(
        validation_arrays["y_pred"],
        dtype=np.int8,
    )
    if not np.array_equal(validation_index, expected_validation_index):
        issues.append("validation prediction support mismatch")
    if not np.array_equal(validation_true, expected_validation_y):
        issues.append("validation prediction labels mismatch")
    selected_threshold, validation_metrics = rf.choose_threshold(
        validation_true,
        validation_prob,
    )
    if not _close(metrics.get("threshold"), selected_threshold):
        issues.append("threshold was not selected from validation")
    expected_validation_pred = (
        validation_prob >= float(selected_threshold)
    ).astype(np.int8)
    if not np.array_equal(validation_pred, expected_validation_pred):
        issues.append("validation y_pred differs from threshold")
    if not _values_equal(
        validation_metrics,
        metrics.get("validation"),
    ):
        issues.append("saved validation metrics differ")
    if not _close(
        metrics.get("best_validation_auprc"),
        validation_metrics.get("auprc"),
    ):
        issues.append(
            "best validation PR-AUC differs from saved predictions"
        )

    test_true = np.asarray(test_arrays["y_true"], dtype=np.int8)
    test_prob = np.asarray(test_arrays["y_prob"], dtype=np.float64)
    test_pred = np.asarray(test_arrays["y_pred"], dtype=np.int8)
    expected_test_pred = (
        test_prob >= float(selected_threshold)
    ).astype(np.int8)
    if not np.array_equal(test_pred, expected_test_pred):
        issues.append("test y_pred differs from validation threshold")
    binary = rf.binary_metrics(
        test_true,
        test_prob,
        selected_threshold,
    )
    for key, expected_value in binary.items():
        if not _values_equal(expected_value, metrics.get(key)):
            issues.append(f"binary metric mismatch: {key}")
    return issues


def _calibration_partition_failures(
    calibration_ids: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
    *,
    windows: Any = None,
    dataset: Any = None,
    val_subject: str | None = None,
) -> list[str]:
    """Check that static-sigma calibration is clean validation-only data."""

    issues: list[str] = []
    calibration = np.asarray(calibration_ids, dtype=np.int64)
    validation = np.asarray(validation_ids, dtype=np.int64)
    test = np.asarray(test_ids, dtype=np.int64)
    if np.any(np.isin(calibration, test)):
        issues.append("calibration overlaps full test split")
    if not np.all(np.isin(calibration, validation)):
        issues.append("calibration is not contained in validation")
    if windows is None:
        return issues
    if (
        calibration.ndim != 1
        or np.any(calibration < 0)
        or np.any(calibration >= len(windows))
    ):
        issues.append("calibration contains out-of-range window IDs")
        return issues
    if not np.all(windows.clean_normal[calibration]):
        issues.append("calibration includes non-clean windows")
    if dataset is not None and val_subject is not None:
        subjects = {
            dataset.records[
                int(windows.record_index[int(index)])
            ].subject_id
            for index in calibration
        }
        if subjects != {str(val_subject)}:
            issues.append("calibration does not belong only to val subject")
    return issues


def audit(
    result_dir: Path,
    *,
    data_dir: Path | None = None,
    source_suite_dir: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    config = _load_json(result_dir / "config.json")
    failures: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(
        config.get("suite_version") == suite.SUITE_VERSION,
        "suite version mismatch",
    )
    require(
        tuple(config.get("nbms", [])) == suite.NBMS,
        "NBM registry mismatch",
    )
    require(
        tuple(
            item.get("name")
            for item in config.get("representations", [])
        )
        == tuple(suite.REPRESENTATIONS),
        "representation registry mismatch",
    )
    require(
        len(config.get("cells", [])) == 12,
        "experiment grid is not 4 x 3",
    )
    require(
        int(config.get("expected_classifier_cells", -1)) == 96,
        "expected classifier cell count is not 96",
    )
    require(
        config.get("fixed_sigma", {}).get("test_subject_used") is False,
        "fixed sigma protocol permits test-subject calibration",
    )

    resolved_source = (
        source_suite_dir.resolve()
        if source_suite_dir is not None
        else Path(config["source_suite_dir"]).resolve()
    )
    resolved_data = (
        data_dir.resolve()
        if data_dir is not None
        else Path(config["data_dir"]).resolve()
    )
    source_config = _load_json(resolved_source / "config.json")
    try:
        current_source_manifest, rebuilt_source_config = (
            suite.build_source_manifest(resolved_source)
        )
        require(
            current_source_manifest == config.get("source"),
            "immutable source manifest differs from result protocol",
        )
        require(
            rebuilt_source_config == source_config,
            "source config changed while rebuilding source manifest",
        )
    except Exception as error:
        failures.append(f"source manifest validation failed: {error}")
    warnings.append(
        "NBM checkpoints are not independently replayed by this auditor; "
        "source checkpoint/cache hashes and the runner-recorded replay "
        "diagnostics are verified instead."
    )
    try:
        dataset, windows, data_sha = rf.load_dataset_and_windows(
            resolved_data,
            source_config,
        )
        require(
            data_sha == config.get("data_sha256"),
            "dataset hash differs from result protocol",
        )
    except Exception as error:  # pragma: no cover - reported to caller
        failures.append(f"dataset reconstruction failed: {error}")
        dataset = None
        windows = None

    fold_metrics: dict[str, dict[str, dict[str, Any]]] = {
        str(cell["variant"]): {}
        for cell in config.get("cells", [])
    }
    fold_predictions: dict[
        str,
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = {
        str(cell["variant"]): {}
        for cell in config.get("cells", [])
    }
    completed_cells = 0
    completed_caches = 0
    for subject in suite.EXPECTED_LOSO_SUBJECTS:
        fold_root = result_dir / f"loso_{subject}"
        support_path = fold_root / "input_support.npz"
        fold_config_path = fold_root / "fold_config.json"
        if not support_path.exists():
            if not allow_partial:
                failures.append(f"missing fold support: {subject}")
            continue
        if not fold_config_path.exists():
            failures.append(f"missing fold config: {subject}")
            continue
        fold_config = _load_json(fold_config_path)
        source_fold = config["source"]["folds"][subject]
        require(
            fold_config.get("protocol_fingerprint")
            == config["protocol_fingerprint"],
            f"fold protocol differs: {subject}",
        )
        require(
            fold_config.get("test_subject") == subject,
            f"fold test subject differs: {subject}",
        )
        require(
            fold_config.get("input_support_sha256")
            == sha256_file(support_path),
            f"fold input-support hash differs: {subject}",
        )
        for field in (
            "source_fold_config_sha256",
            "source_scaler_sha256",
            "source_split_indices_sha256",
            "source_history_support_sha256",
        ):
            require(
                fold_config.get(field) == source_fold.get(field),
                f"fold source binding differs: {subject}/{field}",
            )
        with np.load(support_path, allow_pickle=False) as support:
            expected_support_keys = {
                f"{split}_{suffix}"
                for split in ("train", "validation", "test")
                for suffix in (
                    "anchor_window_index",
                    "history_window_index",
                    "y",
                )
            }
            require(
                set(support.files) == expected_support_keys,
                f"support arrays changed: {subject}",
            )
            validation_anchor_window_index = np.asarray(
                support["validation_anchor_window_index"],
                dtype=np.int64,
            )
            validation_y = np.asarray(
                support["validation_y"],
                dtype=np.int8,
            )
            if windows is not None:
                for split in ("train", "validation", "test"):
                    anchors = np.asarray(
                        support[f"{split}_anchor_window_index"],
                        dtype=np.int64,
                    )
                    labels = np.asarray(
                        support[f"{split}_y"],
                        dtype=np.int8,
                    )
                    history = np.asarray(
                        support[f"{split}_history_window_index"],
                        dtype=np.int64,
                    )
                    require(
                        np.array_equal(labels, windows.label[anchors]),
                        f"support labels differ: {subject}/{split}",
                    )
                    require(
                        history.ndim == 2
                        and history.shape[1] == suite.HISTORY_BLOCKS,
                        f"history does not contain 8 blocks: {subject}/{split}",
                    )
                    starts = windows.target_start[history]
                    require(
                        np.all(np.diff(starts, axis=1) == suite.HORIZON_SAMPLES),
                        f"history blocks are not chronological: {subject}/{split}",
                    )

        source_split_path = (
            resolved_source / f"loso_{subject}" / "split_indices.npz"
        )
        with np.load(source_split_path, allow_pickle=False) as source_split:
            source_split_indices = {
                split: np.asarray(
                    source_split[f"{split}_window_index"],
                    dtype=np.int64,
                )
                for split in ("train", "validation", "test")
            }
            expected_calibration_ids = np.asarray(
                source_split["normal_validation_window_index"],
                dtype=np.int64,
            )
            source_test_window_index = source_split_indices["test"]
            source_validation_window_index = source_split_indices[
                "validation"
            ]
        failures.extend(
            f"{subject}/source calibration: {issue}"
            for issue in _calibration_partition_failures(
                expected_calibration_ids,
                source_validation_window_index,
                source_test_window_index,
                windows=windows,
                dataset=dataset,
                val_subject=str(fold_config["val_subject"]),
            )
        )
        fold_initial_hashes: set[str] = set()
        for nbm in suite.NBMS:
            model_root = fold_root / nbm
            cache_path = model_root / "representation_cache.npz"
            expected_upstream = _source_cache_upstream(
                config,
                subject,
                nbm,
            )
            try:
                done = validate_done(
                    model_root / "REPRESENTATION_CACHE_DONE.json",
                    stage="nbm_representation_cache",
                    protocol_fingerprint=config["protocol_fingerprint"],
                    task_id=f"{subject}/{nbm}/representation_cache",
                    upstream_sha256=expected_upstream,
                )
            except Exception as error:
                failures.append(
                    f"invalid representation cache completion: "
                    f"{subject}/{nbm}: {error}"
                )
                continue
            if done is None:
                if not allow_partial:
                    failures.append(
                        f"missing representation cache: {subject}/{nbm}"
                    )
                continue
            require(
                set(done.get("artifacts", {})) == {
                    "cache",
                    "diagnostics",
                },
                f"representation cache artifact set changed: {subject}/{nbm}",
            )
            completed_caches += 1
            with np.load(cache_path, allow_pickle=False) as cache:
                require(
                    set(cache.files) == suite._representation_cache_keys(),
                    f"representation cache keys changed: {subject}/{nbm}",
                )
                fixed_sigma = np.asarray(
                    cache["fixed_sigma"],
                    dtype=np.float32,
                )
                calibration_ids = np.asarray(
                    cache["normal_calibration_window_index"],
                    dtype=np.int64,
                )
                require(
                    np.array_equal(
                        calibration_ids,
                        expected_calibration_ids,
                    ),
                    f"calibration IDs differ from source: {subject}/{nbm}",
                )
                failures.extend(
                    f"{subject}/{nbm}: {issue}"
                    for issue in _calibration_partition_failures(
                        calibration_ids,
                        source_validation_window_index,
                        source_test_window_index,
                        windows=windows,
                        dataset=dataset,
                        val_subject=str(fold_config["val_subject"]),
                    )
                )
                validation_ids = np.asarray(
                    cache["validation_window_index"],
                    dtype=np.int64,
                )
                validation_error = np.asarray(
                    cache["validation_error"],
                    dtype=np.float32,
                )
                lookup = {
                    int(index): row
                    for row, index in enumerate(validation_ids)
                }
                try:
                    calibration_rows = np.asarray(
                        [lookup[int(index)] for index in calibration_ids],
                        dtype=np.int64,
                    )
                    recomputed_sigma = calibrate_fixed_sigma(
                        validation_error[calibration_rows],
                        epsilon=suite.FIXED_SIGMA_EPSILON,
                    )
                    require(
                        np.allclose(
                            recomputed_sigma,
                            fixed_sigma,
                            rtol=1e-6,
                            atol=1e-6,
                        ),
                        f"fixed sigma formula mismatch: {subject}/{nbm}",
                    )
                except KeyError:
                    failures.append(
                        f"calibration IDs missing from validation: {subject}/{nbm}"
                    )
                for split in ("train", "validation", "test"):
                    error = np.asarray(
                        cache[f"{split}_error"],
                        dtype=np.float32,
                    )
                    dynamic = np.asarray(
                        cache[f"{split}_dynamic"],
                        dtype=np.float32,
                    )
                    labels = np.asarray(
                        cache[f"{split}_y"],
                        dtype=np.int8,
                    )
                    indices = np.asarray(
                        cache[f"{split}_window_index"],
                        dtype=np.int64,
                    )
                    require(
                        error.shape == dynamic.shape
                        and error.shape[1:]
                        == (9, suite.HORIZON_SAMPLES),
                        f"representation shape mismatch: {subject}/{nbm}/{split}",
                    )
                    require(
                        np.isfinite(error).all()
                        and np.isfinite(dynamic).all(),
                        f"non-finite representation: {subject}/{nbm}/{split}",
                    )
                    require(
                        np.all(np.abs(dynamic) <= 12.0 + 1e-6),
                        f"dynamic residual exceeds clip: {subject}/{nbm}/{split}",
                    )
                    if windows is not None:
                        require(
                            np.array_equal(labels, windows.label[indices]),
                            f"cache labels differ: {subject}/{nbm}/{split}",
                        )
                    require(
                        np.array_equal(
                            indices,
                            source_split_indices[split],
                        ),
                        f"cache split IDs differ from source: "
                        f"{subject}/{nbm}/{split}",
                    )

            initial_hashes: set[str] = set()
            for representation in suite.REPRESENTATIONS:
                cell = next(
                    item
                    for item in config["cells"]
                    if item["nbm"] == nbm
                    and item["representation"] == representation
                )
                loaded = suite._load_completed_cell(
                    result_dir,
                    config,
                    cell,
                    subject,
                )
                if loaded is None:
                    if not allow_partial:
                        failures.append(
                            f"missing classifier: {subject}/{nbm}/{representation}"
                        )
                    continue
                completed_cells += 1
                metrics, arrays = loaded
                fold_metrics[str(cell["variant"])][subject] = metrics
                fold_predictions[str(cell["variant"])][subject] = (
                    np.asarray(arrays["y_true"], dtype=np.int8),
                    np.asarray(arrays["y_prob"], dtype=np.float64),
                    np.asarray(arrays["y_pred"], dtype=np.int8),
                )
                initial_hashes.add(str(metrics["initial_state_sha256"]))
                fold_initial_hashes.add(str(metrics["initial_state_sha256"]))
                cell_root = suite.task_root_for(
                    result_dir,
                    subject,
                    nbm,
                    representation,
                )
                validation_path = (
                    cell_root / "validation_predictions.npz"
                )
                try:
                    with np.load(
                        validation_path,
                        allow_pickle=False,
                    ) as validation_payload:
                        require(
                            set(validation_payload.files)
                            == {
                                "window_index",
                                "y_true",
                                "y_prob",
                                "y_pred",
                            },
                            f"validation prediction arrays changed: "
                            f"{subject}/{nbm}/{representation}",
                        )
                        validation_index = np.asarray(
                            validation_payload["window_index"],
                            dtype=np.int64,
                        )
                        validation_true = np.asarray(
                            validation_payload["y_true"],
                            dtype=np.int8,
                        )
                        validation_prob = np.asarray(
                            validation_payload["y_prob"],
                            dtype=np.float64,
                        )
                        validation_pred = np.asarray(
                            validation_payload["y_pred"],
                            dtype=np.int8,
                        )
                except Exception as error:
                    failures.append(
                        f"invalid validation predictions: "
                        f"{subject}/{nbm}/{representation}: {error}"
                    )
                    continue
                threshold_issues = _prediction_threshold_failures(
                    metrics,
                    arrays,
                    {
                        "window_index": validation_index,
                        "y_true": validation_true,
                        "y_prob": validation_prob,
                        "y_pred": validation_pred,
                    },
                    validation_anchor_window_index,
                    validation_y,
                )
                failures.extend(
                    f"{subject}/{nbm}/{representation}: {issue}"
                    for issue in threshold_issues
                )
                recomputed = rf.prediction_metrics(
                    np.asarray(arrays["y_true"], dtype=np.int8),
                    np.asarray(arrays["y_prob"], dtype=np.float64),
                    np.asarray(arrays["y_pred"], dtype=np.int8),
                )
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "roc_auc",
                    "pr_auc",
                    "fog_recall",
                    "fog_f1",
                    "specificity",
                ):
                    require(
                        _close(recomputed.get(key), metrics.get(key)),
                        f"prediction metric mismatch: "
                        f"{subject}/{nbm}/{representation}/{key}",
                    )
                if dataset is not None and windows is not None:
                    event = rf.event_metrics(
                        dataset,
                        windows,
                        np.asarray(arrays["window_index"], dtype=np.int64),
                        np.asarray(arrays["y_pred"], dtype=np.int8),
                    )
                    for key, expected_value in event.items():
                        require(
                            _values_equal(
                                expected_value,
                                metrics.get(key),
                            ),
                            f"event metric mismatch: "
                            f"{subject}/{nbm}/{representation}/{key}",
                        )
            require(
                len(initial_hashes) <= 1,
                f"classifier initial states differ: {subject}/{nbm}",
            )
        require(
            len(fold_initial_hashes) <= 1,
            f"classifier initial states differ across NBM: {subject}",
        )

    aggregate_path = result_dir / "aggregate_metrics.json"
    if aggregate_path.exists():
        aggregate = _load_json(aggregate_path)
        experiments = aggregate.get("experiments", {})
        require(
            set(experiments)
            == {
                str(cell["experiment_id"])
                for cell in config.get("cells", [])
            },
            "aggregate experiment registry is incomplete or changed",
        )
        expected_best_candidates: list[tuple[float, str]] = []
        for cell in config.get("cells", []):
            identifier = str(cell["variant"])
            group = list(fold_metrics[identifier].values())
            saved = experiments.get(identifier)
            if saved is None:
                failures.append(
                    f"aggregate experiment is missing: {identifier}"
                )
                continue
            completed_subjects = [
                subject
                for subject in suite.EXPECTED_LOSO_SUBJECTS
                if subject in fold_metrics[identifier]
            ]
            require(
                saved.get("completed_folds") == completed_subjects,
                f"aggregate completed folds mismatch: {identifier}",
            )
            for key in (
                "experiment_id",
                "variant",
                "nbm",
                "representation",
                "display_name",
            ):
                require(
                    saved.get(key) == cell.get(key),
                    f"aggregate identity mismatch: {identifier}/{key}",
                )
            recomputed = (
                aggregate_fold_metrics(
                    group,
                    list(suite.CLASSIFICATION_METRICS),
                )
                if group
                else {
                    metric: {
                        "mean": None,
                        "std": None,
                        "n_folds": 0,
                    }
                    for metric in suite.CLASSIFICATION_METRICS
                }
            )
            require(
                _values_equal(
                    recomputed,
                    saved.get("subject_macro"),
                ),
                f"aggregate subject-macro payload mismatch: {identifier}",
            )
            prediction_rows = fold_predictions[identifier]
            if prediction_rows:
                truths = np.concatenate(
                    [
                        prediction_rows[subject][0]
                        for subject in completed_subjects
                    ]
                )
                probabilities = np.concatenate(
                    [
                        prediction_rows[subject][1]
                        for subject in completed_subjects
                    ]
                )
                predictions = np.concatenate(
                    [
                        prediction_rows[subject][2]
                        for subject in completed_subjects
                    ]
                )
                pooled = rf.prediction_metrics(
                    truths,
                    probabilities,
                    predictions,
                )
            else:
                pooled = None
            require(
                _values_equal(pooled, saved.get("pooled")),
                f"aggregate pooled payload mismatch: {identifier}",
            )
            pr_auc_mean = recomputed["pr_auc"]["mean"]
            if pr_auc_mean is not None:
                expected_best_candidates.append(
                    (float(pr_auc_mean), identifier)
                )

        expected_comparisons = _expected_comparisons(
            fold_metrics,
            config,
        )
        require(
            _values_equal(
                expected_comparisons,
                aggregate.get("paired_pr_auc_comparisons"),
            ),
            "aggregate paired PR-AUC comparisons mismatch",
        )
        fully_complete = (
            completed_cells == int(config["expected_classifier_cells"])
        )
        expected_best = None
        if (
            fully_complete
            and bool(config.get("reportable"))
            and expected_best_candidates
        ):
            expected_best = sorted(
                expected_best_candidates,
                key=lambda item: (-item[0], item[1]),
            )[0][1]
        require(
            aggregate.get("best_experiment") == expected_best,
            "aggregate best experiment mismatch",
        )
    elif not allow_partial:
        failures.append("aggregate_metrics.json is missing")

    expected_cells = int(config["expected_classifier_cells"])
    expected_caches = (
        len(suite.EXPECTED_LOSO_SUBJECTS) * len(suite.NBMS)
    )
    if not allow_partial:
        require(
            completed_cells == expected_cells,
            f"completed classifiers {completed_cells}/{expected_cells}",
        )
        require(
            completed_caches == expected_caches,
            f"completed caches {completed_caches}/{expected_caches}",
        )
    expected_best = None
    if aggregate_path.exists():
        expected_best = _load_json(aggregate_path).get("best_experiment")

    status_path = result_dir / "status.json"
    if status_path.exists():
        status = _load_json(status_path)
        expected_status = (
            "complete"
            if completed_cells == expected_cells
            and bool(config.get("reportable"))
            else (
                "smoke_complete"
                if completed_cells == expected_cells
                else "partial"
            )
        )
        expected_status_values = {
            "suite_version": suite.SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_representation_cache_tasks": expected_caches,
            "completed_representation_cache_tasks": completed_caches,
            "expected_classifier_cells": expected_cells,
            "completed_classifier_cells": completed_cells,
            "status": expected_status,
            "reportable": bool(config.get("reportable")),
            "best_experiment": expected_best,
        }
        require(
            status == expected_status_values,
            "root status payload mismatch",
        )
    elif not allow_partial:
        failures.append("status.json is missing")

    csv_contracts = (
        (
            "fold_summary.csv",
            completed_cells,
            lambda row: (
                row.get("experiment_id"),
                row.get("test_subject"),
            ),
            {
                (identifier, subject)
                for identifier, rows in fold_metrics.items()
                for subject in rows
            },
        ),
        (
            "aggregate_summary.csv",
            len(config.get("cells", [])),
            lambda row: row.get("experiment_id"),
            {
                str(cell["experiment_id"])
                for cell in config.get("cells", [])
            },
        ),
        (
            "publication_table.csv",
            len(config.get("cells", [])),
            lambda row: (row.get("NBM"), row.get("Representation")),
            {
                (
                    str(cell["nbm"]),
                    suite.REPRESENTATIONS[
                        str(cell["representation"])
                    ]["display_name"],
                )
                for cell in config.get("cells", [])
            },
        ),
        (
            "paired_pr_auc_deltas.csv",
            len(suite.NBMS) * 3,
            lambda row: row.get("comparison_id"),
            {
                row["comparison_id"]
                for row in _expected_comparisons(fold_metrics, config)
            },
        ),
    )
    for filename, expected_count, key_function, expected_keys in csv_contracts:
        path = result_dir / filename
        if not path.exists():
            if not allow_partial:
                failures.append(f"root CSV is missing: {filename}")
            continue
        rows = _load_csv(path)
        require(
            len(rows) == expected_count,
            f"root CSV row count mismatch: {filename}",
        )
        require(
            {key_function(row) for row in rows} == expected_keys,
            f"root CSV row identity mismatch: {filename}",
        )
    return {
        "audit_version": AUDIT_VERSION,
        "suite_version": config.get("suite_version"),
        "protocol_fingerprint": config.get("protocol_fingerprint"),
        "result_dir": str(result_dir),
        "allow_partial": bool(allow_partial),
        "expected_classifier_cells": expected_cells,
        "completed_classifier_cells": completed_cells,
        "expected_representation_cache_tasks": expected_caches,
        "completed_representation_cache_tasks": completed_caches,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    report = audit(
        args.result_dir,
        data_dir=args.data_dir,
        source_suite_dir=args.source_suite_dir,
        allow_partial=args.allow_partial,
    )
    result_dir = args.result_dir.resolve()
    atomic_json_dump(report, result_dir / "AUDIT_REPORT.json")
    lines = [
        f"Audit version: {AUDIT_VERSION}",
        f"Result directory: {result_dir}",
        f"Status: {report['status']}",
        (
            "Classifier cells: "
            f"{report['completed_classifier_cells']}/"
            f"{report['expected_classifier_cells']}"
        ),
        (
            "Representation caches: "
            f"{report['completed_representation_cache_tasks']}/"
            f"{report['expected_representation_cache_tasks']}"
        ),
        f"Failures: {len(report['failures'])}",
        f"Warnings: {len(report['warnings'])}",
    ]
    if report["failures"]:
        lines.append("")
        lines.append("Failures:")
        lines.extend(
            f"{index}. {message}"
            for index, message in enumerate(report["failures"], start=1)
        )
    if report["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(
            f"{index}. {message}"
            for index, message in enumerate(report["warnings"], start=1)
        )
    text = "\n".join(lines) + "\n"
    (result_dir / "AUDIT_REPORT.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
