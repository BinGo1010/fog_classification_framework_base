#!/usr/bin/env python
"""Audit the strict Daphnet Persistence input-representation ablation.

The auditor is independent from the multi-GPU scheduler.  It validates the
canonical four-representation by eight-fold protocol, the frozen Persistence
source, fold-local block representations and common history support, every
completed TCN-M classifier, validation-only threshold selection, window/event
metrics, paired subject-level statistics, and root summaries.

``--allow-partial`` permits missing folds/cells for smoke tests and interrupted
runs.  Any artifact that has already been marked complete is still audited
strictly; corrupt or scientifically incompatible completed work always fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_persistence_input_ablation as suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import RobustChannelScaler
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


AUDIT_VERSION = "daphnet_persistence_input_ablation_audit.v1"
EXPECTED_SUBJECTS = tuple(rf.EXPECTED_LOSO_SUBJECTS)
EXPECTED_REPRESENTATIONS = (
    "raw_support_matched",
    "error_x_minus_mu",
    "standardized_error",
    "standardized_error_clip12",
)
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_CACHE_KEYS = {
    "sigma",
    *{
        f"{split}_{key}"
        for split in EXPECTED_SPLITS
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
}
EXPECTED_SUPPORT_KEYS = {
    f"{split}_{key}"
    for split in EXPECTED_SPLITS
    for key in ("anchor_window_index", "history_window_index", "y")
}
PREDICTION_KEYS = {"window_index", "y_true", "y_prob", "y_pred"}
CLASSIFIER_ARTIFACTS = {
    "best",
    "last",
    "metrics",
    "predictions",
    "validation_predictions",
    "predictions_csv",
}
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
RUNTIME_FIELDS = {
    "data_dir",
    "source_suite_dir",
    "output_dir",
    "device",
    "num_workers",
    "resume",
    "smoke",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the 4-representation x 8-fold Daphnet Persistence "
            "input-representation ablation"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        help=(
            "Fallback canonical NBM suite when the path stored in config.json "
            "is unavailable."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Fallback processed Daphnet directory when the path stored in "
            "config.json is unavailable."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow missing tasks while still strictly validating every "
            "completed artifact."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=float(getattr(suite, "FORMULA_ATOL", 5e-6)),
        help="Absolute/relative tolerance for representation identities.",
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


def value_equal(
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-9,
) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return (
            set(actual) == set(expected)
            and all(
                value_equal(
                    actual[key],
                    expected[key],
                    tolerance=tolerance,
                )
                for key in actual
            )
        )
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if isinstance(actual, (str, bytes)) or isinstance(
            expected, (str, bytes)
        ):
            return actual == expected
        try:
            left = np.asarray(actual)
            right = np.asarray(expected)
            if left.dtype.kind in "fc" or right.dtype.kind in "fc":
                return bool(
                    np.allclose(
                        left.astype(np.float64),
                        right.astype(np.float64),
                        rtol=tolerance,
                        atol=tolerance,
                        equal_nan=False,
                    )
                )
            return bool(np.array_equal(left, right))
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


def validate_mapping(
    report: dict[str, Any],
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    keys: Iterable[str],
    label: str,
    *,
    tolerance: float = 1e-9,
) -> None:
    for key in keys:
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
        if not value_equal(
            actual[key],
            expected[key],
            tolerance=tolerance,
        ):
            add_issue(
                report,
                "failures",
                f"{label}: value mismatch",
                key=key,
                saved=actual[key],
                recomputed=expected[key],
            )


def representation_id(item: Mapping[str, Any]) -> str:
    for key in ("representation", "representation_id", "variant", "id"):
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    raise KeyError(f"Representation has no identifier: {item}")


def normalise_representations() -> list[dict[str, Any]]:
    declared = suite.REPRESENTATIONS
    result: list[dict[str, Any]] = []
    if isinstance(declared, Mapping):
        for key, value in declared.items():
            payload = (
                dict(suite.representation_variant(str(key)))
                if hasattr(suite, "representation_variant")
                else dict(value)
            )
            for field, content in dict(value).items():
                payload.setdefault(field, content)
            payload.setdefault("variant", str(key))
            payload.setdefault("representation", str(key))
            payload.setdefault("representation_id", str(key))
            result.append(payload)
    else:
        for value in declared:
            payload = dict(value)
            identifier = representation_id(payload)
            payload.setdefault("representation", identifier)
            payload.setdefault("representation_id", identifier)
            result.append(payload)
    return result


def _first(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def normalise_comparisons() -> list[dict[str, Any]]:
    declared = suite.COMPARISONS
    rows: list[dict[str, Any]] = []
    if isinstance(declared, Mapping):
        iterable = [
            {"comparison_id": str(key), **dict(value)}
            for key, value in declared.items()
        ]
    else:
        iterable = [dict(value) for value in declared]
    for index, payload in enumerate(iterable):
        comparison = str(
            _first(
                payload,
                ("comparison_id", "id", "name"),
                f"comparison_{index + 1}",
            )
        )
        current = _first(
            payload,
            (
                "new_representation",
                "new",
                "current",
                "numerator",
                "representation",
            ),
        )
        reference = _first(
            payload,
            (
                "reference_representation",
                "reference",
                "baseline",
                "denominator",
            ),
        )
        if current is None or reference is None:
            raise KeyError(
                f"Comparison must identify new/reference representations: "
                f"{payload}"
            )
        rows.append(
            {
                **payload,
                "comparison_id": comparison,
                "new_representation": str(current),
                "reference_representation": str(reference),
                "bootstrap_label": str(
                    payload.get("bootstrap_label", comparison)
                ),
            }
        )
    return rows


def protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RUNTIME_FIELDS | {"protocol_fingerprint"}
    }


def configured_path(
    config: Mapping[str, Any],
    key: str,
) -> Path | None:
    value = config.get(key)
    if value:
        return Path(str(value)).expanduser()
    source = config.get("source")
    if isinstance(source, Mapping):
        value = source.get(key)
        if value:
            return Path(str(value)).expanduser()
    return None


def choose_existing_path(
    primary: Path | None,
    fallback: Path | None,
    label: str,
) -> Path:
    candidates = [path for path in (fallback, primary) if path is not None]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"No reachable {label}; candidates={[str(item) for item in candidates]}"
    )


def validate_protocol(
    report: dict[str, Any],
    result_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    config_path = result_dir / "config.json"
    manifest_path = result_dir / "run_manifest.json"
    if not require(report, config_path.exists(), "Missing config.json"):
        return None, [], []
    config = load_json(config_path)
    if manifest_path.exists():
        run_manifest = load_json(manifest_path)
        expected_manifest = {
            key: value
            for key, value in config.items()
            if key not in RUNTIME_FIELDS
        }
        require(
            report,
            run_manifest == expected_manifest,
            "run_manifest.json differs from immutable config payload",
        )
    else:
        add_issue(report, "failures", "Missing run_manifest.json")

    require(
        report,
        config.get("suite_version") == suite.SUITE_VERSION,
        "Suite version mismatch",
        saved=config.get("suite_version"),
        expected=suite.SUITE_VERSION,
    )
    run_kind = str(config.get("run_kind", "formal"))
    reportable = bool(config.get("reportable", run_kind == "formal"))
    require(
        report,
        run_kind in {"formal", "smoke"},
        "Unknown run_kind",
        run_kind=run_kind,
    )
    require(
        report,
        reportable == (run_kind == "formal"),
        "reportable flag is inconsistent with run_kind",
        run_kind=run_kind,
        reportable=reportable,
    )
    recomputed = canonical_fingerprint(protocol_payload(config))
    require(
        report,
        config.get("protocol_fingerprint") == recomputed,
        "Protocol fingerprint mismatch",
        saved=config.get("protocol_fingerprint"),
        recomputed=recomputed,
    )

    representations = normalise_representations()
    representation_ids = [representation_id(item) for item in representations]
    require(
        report,
        tuple(representation_ids) == EXPECTED_REPRESENTATIONS,
        "Representation order/identity is not the preregistered four-arm set",
        saved=representation_ids,
        expected=list(EXPECTED_REPRESENTATIONS),
    )
    config_representations = config.get("representations")
    if isinstance(config_representations, list):
        config_ids = [
            representation_id(item)
            for item in config_representations
            if isinstance(item, Mapping)
        ]
        require(
            report,
            config_ids == representation_ids,
            "config.json representation grid differs from runner constants",
        )

    comparisons = normalise_comparisons()
    require(
        report,
        len(comparisons) == 4,
        "Exactly four preregistered comparisons are required",
        observed=len(comparisons),
    )
    expected_pairs = {
        ("error_x_minus_mu", "raw_support_matched"),
        ("standardized_error", "error_x_minus_mu"),
        ("standardized_error_clip12", "standardized_error"),
        ("standardized_error_clip12", "raw_support_matched"),
    }
    observed_pairs = {
        (
            item["new_representation"],
            item["reference_representation"],
        )
        for item in comparisons
    }
    require(
        report,
        observed_pairs == expected_pairs,
        "Preregistered comparison pairs changed",
        observed=sorted(observed_pairs),
        expected=sorted(expected_pairs),
    )

    fixed_values = {
        "sampling_rate_hz": 64,
        "n_channels": 9,
        "context_samples": 128,
        "horizon_samples": int(suite.HORIZON_SAMPLES),
        "stride_samples": 16,
        "history_samples": int(suite.HISTORY_SAMPLES),
        "history_blocks": 8,
        "seed": 42,
        "expected_experiments": 4,
        "expected_classifier_cells": 32,
    }
    for key, expected in fixed_values.items():
        require(
            report,
            config.get(key) == expected,
            "Fixed protocol value mismatch",
            key=key,
            saved=config.get(key),
            expected=expected,
        )
    require(
        report,
        tuple(config.get("folds_resolved", ())) == EXPECTED_SUBJECTS,
        "LOSO folds are not the canonical eight subjects",
    )
    require(
        report,
        set(config.get("excluded_subjects", ())) == {"S04", "S10"},
        "Excluded subjects must be exactly S04 and S10",
    )
    require(
        report,
        math.isclose(
            float(config.get("robust_clip", float("nan"))),
            12.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Common Robust scaler clip must be 12",
    )
    residual_clip = config.get("residual_clip", 12.0)
    require(
        report,
        math.isclose(
            float(residual_clip),
            12.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Residual-space clip must be 12",
    )

    classifier = config.get("classifier")
    if not isinstance(classifier, Mapping):
        add_issue(report, "failures", "Missing classifier protocol")
    else:
        dilations = tuple(
            int(value) for value in classifier.get("dilations", ())
        )
        require(
            report,
            dilations == tuple(int(value) for value in suite.TCN_M_DILATIONS),
            "Classifier is not TCN-M",
            dilations=dilations,
        )
        require(
            report,
            int(classifier.get("receptive_field_samples", -1)) == 125,
            "TCN-M receptive field must be 125 samples",
        )
        require(
            report,
            classifier.get("kernel_size", 3) == 3,
            "TCN-M kernel size changed",
        )
        require(
            report,
            classifier.get("convolutions_per_block", 2) == 2,
            "TCN-M convolutions per block changed",
        )

    implementation = config.get("implementation")
    if isinstance(implementation, Mapping) and isinstance(
        implementation.get("files"), Mapping
    ):
        recomputed_files: dict[str, str] = {}
        for relative, expected_hash in implementation["files"].items():
            path = REPO_ROOT / str(relative)
            if not path.exists():
                add_issue(
                    report,
                    "failures",
                    "Implementation file is missing",
                    path=str(path),
                )
                continue
            observed_hash = sha256_file(path)
            recomputed_files[str(relative)] = observed_hash
            require(
                report,
                observed_hash == expected_hash,
                "Implementation file hash changed",
                file=str(relative),
            )
        if len(recomputed_files) == len(implementation["files"]):
            require(
                report,
                canonical_fingerprint(recomputed_files)
                == implementation.get("sha256"),
                "Implementation manifest aggregate hash mismatch",
            )
    else:
        add_issue(report, "failures", "Missing implementation manifest")
    return config, representations, comparisons


def validate_source_and_dataset(
    report: dict[str, Any],
    config: Mapping[str, Any],
    source_suite_dir: Path,
    data_dir: Path,
) -> tuple[dict[str, Any] | None, Any | None, Any | None]:
    source_config_path = source_suite_dir / "config.json"
    source_manifest_path = source_suite_dir / "run_manifest.json"
    if not (
        require(
            report,
            source_config_path.exists(),
            "Canonical source config is missing",
            path=str(source_config_path),
        )
        and require(
            report,
            source_manifest_path.exists(),
            "Canonical source run manifest is missing",
            path=str(source_manifest_path),
        )
    ):
        return None, None, None
    source_config = load_json(source_config_path)
    require(
        report,
        source_config.get("suite_version") == rf.SOURCE_SUITE_VERSION,
        "Unexpected source-suite version",
        saved=source_config.get("suite_version"),
        expected=rf.SOURCE_SUITE_VERSION,
    )
    require(
        report,
        "persistence" in source_config.get("nbms_resolved", ()),
        "Canonical source has no Persistence NBM",
    )
    source_fixed = {
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
    }
    for key, expected in source_fixed.items():
        require(
            report,
            value_equal(source_config.get(key), expected),
            "Canonical source protocol mismatch",
            key=key,
            saved=source_config.get(key),
            expected=expected,
        )
    require(
        report,
        set(source_config.get("excluded_subjects", ())) == {"S04", "S10"},
        "Canonical source excluded-subject set changed",
    )
    source_declared = config.get("source")
    if isinstance(source_declared, Mapping):
        declared_protocol = _first(
            source_declared,
            ("source_protocol_fingerprint", "protocol_fingerprint"),
        )
        if declared_protocol is not None:
            require(
                report,
                declared_protocol == source_config.get("protocol_fingerprint"),
                "Source protocol fingerprint differs from config provenance",
            )
        declared_data = _first(
            source_declared,
            ("source_data_sha256", "data_sha256"),
        )
        if declared_data is not None:
            require(
                report,
                declared_data == source_config.get("data_sha256"),
                "Source data hash differs from config provenance",
            )
    try:
        observed_data_hash = dataset_fingerprint(data_dir)
        require(
            report,
            observed_data_hash == source_config.get("data_sha256"),
            "Processed dataset hash differs from canonical source",
        )
        dataset, windows, _ = rf.load_dataset_and_windows(
            data_dir,
            source_config,
        )
    except Exception as error:  # noqa: BLE001 - collect audit failures.
        add_issue(
            report,
            "failures",
            "Cannot reconstruct canonical dataset/windows",
            error=f"{type(error).__name__}: {error}",
        )
        return source_config, None, None
    require(
        report,
        tuple(dataset.subjects) == EXPECTED_SUBJECTS,
        "Reconstructed dataset subjects differ from canonical LOSO folds",
    )
    return source_config, dataset, windows


def artifact_path(done_path: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact["path"]))
    return path if path.is_absolute() else done_path.parent / path


def validate_done_collect(
    report: dict[str, Any],
    path: Path,
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str | None = None,
) -> dict[str, Any] | None:
    try:
        return validate_done(
            path,
            stage=stage,
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=upstream_sha256,
        )
    except Exception as error:  # noqa: BLE001 - collect all corrupt DONEs.
        add_issue(
            report,
            "failures",
            "Invalid DONE manifest",
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )
        return None


def source_fold_artifacts(
    report: dict[str, Any],
    source_suite_dir: Path,
    source_config: Mapping[str, Any],
    declared_source: Mapping[str, Any],
    subject: str,
) -> dict[str, Any] | None:
    fold_root = source_suite_dir / f"loso_{subject}"
    fold_config_path = fold_root / "fold_config.json"
    scaler_path = fold_root / "scaler.json"
    split_path = fold_root / "split_indices.npz"
    support_path = fold_root / "history_support.npz"
    required = (fold_config_path, scaler_path, split_path, support_path)
    if not all(path.exists() for path in required):
        add_issue(
            report,
            "failures",
            "Canonical source fold artifacts are missing",
            subject=subject,
            missing=[str(path) for path in required if not path.exists()],
        )
        return None
    fold_config = load_json(fold_config_path)
    require(
        report,
        fold_config.get("protocol_fingerprint")
        == source_config.get("protocol_fingerprint"),
        "Canonical source fold protocol mismatch",
        subject=subject,
    )
    require(
        report,
        fold_config.get("test_subject") == subject,
        "Canonical source fold subject mismatch",
        subject=subject,
    )

    nbm_root = fold_root / "persistence"
    nbm_done_path = nbm_root / "nbm" / "DONE.json"
    nbm_done = validate_done_collect(
        report,
        nbm_done_path,
        stage="nbm",
        protocol_fingerprint=str(source_config["protocol_fingerprint"]),
        task_id=f"loso_{subject}/persistence/nbm",
    )
    if nbm_done is None:
        return None
    try:
        nbm_best = artifact_path(
            nbm_done_path,
            nbm_done["artifacts"]["best"],
        )
        nbm_sha = str(nbm_done["artifacts"]["best"]["sha256"])
    except (KeyError, TypeError) as error:
        add_issue(
            report,
            "failures",
            "Malformed source NBM DONE manifest",
            subject=subject,
            error=str(error),
        )
        return None
    residual_done_path = nbm_root / "RESIDUAL_CACHE_DONE.json"
    residual_done = validate_done_collect(
        report,
        residual_done_path,
        stage="residual_cache",
        protocol_fingerprint=str(source_config["protocol_fingerprint"]),
        task_id=f"loso_{subject}/persistence/residual_cache",
        upstream_sha256=nbm_sha,
    )
    if residual_done is None:
        return None
    try:
        residual_cache = artifact_path(
            residual_done_path,
            residual_done["artifacts"]["cache"],
        )
        residual_sha = str(
            residual_done["artifacts"]["cache"]["sha256"]
        )
    except (KeyError, TypeError) as error:
        add_issue(
            report,
            "failures",
            "Malformed source residual DONE manifest",
            subject=subject,
            error=str(error),
        )
        return None
    declared_folds = declared_source.get("folds")
    if isinstance(declared_folds, Mapping) and isinstance(
        declared_folds.get(subject), Mapping
    ):
        declared_fold = declared_folds[subject]
        observed_identity = {
            "source_nbm_best_sha256": nbm_sha,
            "source_residual_cache_sha256": residual_sha,
            "source_residual_done_sha256": sha256_file(
                residual_done_path
            ),
            "source_fold_config_sha256": sha256_file(
                fold_config_path
            ),
            "source_history_support_sha256": sha256_file(
                support_path
            ),
            "source_history_support_bytes": int(
                support_path.stat().st_size
            ),
        }
        for key, observed in observed_identity.items():
            if key in declared_fold:
                require(
                    report,
                    declared_fold[key] == observed,
                    "Configured source-fold provenance changed",
                    subject=subject,
                    key=key,
                )
    return {
        "fold_root": fold_root,
        "fold_config": fold_config,
        "fold_config_path": fold_config_path,
        "scaler_path": scaler_path,
        "split_path": split_path,
        "support_path": support_path,
        "nbm_best": nbm_best,
        "nbm_sha256": nbm_sha,
        "residual_cache": residual_cache,
        "residual_sha256": residual_sha,
        "residual_done_path": residual_done_path,
    }


def load_cache(
    path: Path,
    report: dict[str, Any],
    subject: str,
) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != EXPECTED_CACHE_KEYS:
                add_issue(
                    report,
                    "failures",
                    "Representation cache key set mismatch",
                    subject=subject,
                    observed=sorted(payload.files),
                    expected=sorted(EXPECTED_CACHE_KEYS),
                )
                return None
            return {
                key: np.asarray(payload[key])
                for key in payload.files
            }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load representation cache",
            subject=subject,
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )
        return None


def load_support(
    path: Path,
    report: dict[str, Any],
    subject: str,
) -> dict[str, dict[str, np.ndarray]] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != EXPECTED_SUPPORT_KEYS:
                add_issue(
                    report,
                    "failures",
                    "Common support key set mismatch",
                    subject=subject,
                    observed=sorted(payload.files),
                    expected=sorted(EXPECTED_SUPPORT_KEYS),
                )
                return None
            return {
                split: {
                    "anchor": np.asarray(
                        payload[f"{split}_anchor_window_index"],
                        dtype=np.int64,
                    ),
                    "history": np.asarray(
                        payload[f"{split}_history_window_index"],
                        dtype=np.int64,
                    ),
                    "y": np.asarray(
                        payload[f"{split}_y"],
                        dtype=np.int8,
                    ),
                }
                for split in EXPECTED_SPLITS
            }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load common input support",
            subject=subject,
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )
        return None


def _source_residuals(
    source_cache_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    with np.load(source_cache_path, allow_pickle=False) as payload:
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


def validate_fold_cache_and_support(
    report: dict[str, Any],
    result_dir: Path,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    dataset: Any,
    windows: Any,
    subject: str,
    tolerance: float,
) -> dict[str, Any] | None:
    fold_root = result_dir / f"loso_{subject}"
    fold_config_path = fold_root / "fold_config.json"
    cache_path = fold_root / "representation_cache.npz"
    support_path = fold_root / "input_support.npz"
    done_path = fold_root / "REPRESENTATION_CACHE_DONE.json"
    if not done_path.exists():
        return None
    complete = validate_done_collect(
        report,
        done_path,
        stage="input_representation_cache",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/input_representation_cache",
    )
    if complete is None:
        return None
    if not all(path.exists() for path in (fold_config_path, cache_path, support_path)):
        add_issue(
            report,
            "failures",
            "Completed representation cache lacks fold artifacts",
            subject=subject,
        )
        return None
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "cache",
        "diagnostics",
    }:
        add_issue(
            report,
            "failures",
            "Representation-cache DONE artifact set mismatch",
            subject=subject,
            observed=(
                sorted(artifacts)
                if isinstance(artifacts, Mapping)
                else None
            ),
        )
        return None
    require(
        report,
        artifact_path(done_path, artifacts["cache"]).resolve()
        == cache_path.resolve(),
        "Representation-cache DONE points to another cache",
        subject=subject,
    )

    fold_config = load_json(fold_config_path)
    require(
        report,
        fold_config.get("protocol_fingerprint")
        == config.get("protocol_fingerprint"),
        "Fold protocol mismatch",
        subject=subject,
    )
    require(
        report,
        fold_config.get("test_subject") == subject,
        "Fold test-subject mismatch",
        subject=subject,
    )
    if "source_scaler" in fold_config:
        require(
            report,
            fold_config["source_scaler"]
            == source["fold_config"].get("scaler"),
            "Fold scaler differs from canonical source fold",
            subject=subject,
        )
    source_identity = {
        "source_nbm_best_sha256": source["nbm_sha256"],
        "source_residual_cache_sha256": source["residual_sha256"],
    }
    source_mapping = fold_config.get("source")
    if isinstance(source_mapping, Mapping):
        source_binding = source_mapping.get("source_binding_sha256")
        if source_binding is not None:
            require(
                report,
                complete.get("upstream_nbm_sha256") == source_binding,
                "Representation-cache DONE source binding mismatch",
                subject=subject,
            )
        for key, expected in source_identity.items():
            if key in source_mapping:
                require(
                    report,
                    source_mapping[key] == expected,
                    "Fold source provenance mismatch",
                    subject=subject,
                    key=key,
                )
    source_provenance_path = fold_root / "source_provenance.json"
    if source_provenance_path.exists() and isinstance(
        source_mapping, Mapping
    ):
        require(
            report,
            load_json(source_provenance_path) == dict(source_mapping),
            "source_provenance.json differs from fold_config source",
            subject=subject,
        )
    else:
        add_issue(
            report,
            "failures",
            "Missing source_provenance.json",
            subject=subject,
        )

    cache = load_cache(cache_path, report, subject)
    support = load_support(support_path, report, subject)
    if cache is None or support is None:
        return None
    cache_sha = sha256_file(cache_path)
    support_sha = sha256_file(support_path)
    if "representation_cache_sha256" in fold_config:
        require(
            report,
            fold_config["representation_cache_sha256"] == cache_sha,
            "Fold representation cache hash mismatch",
            subject=subject,
        )
    expected_support_sha = _first(
        fold_config,
        ("input_support_sha256", "history_support_sha256"),
    )
    if expected_support_sha is None and isinstance(source_mapping, Mapping):
        expected_support_sha = source_mapping.get("input_support_sha256")
    if expected_support_sha is not None:
        require(
            report,
            expected_support_sha == support_sha,
            "Fold input support hash mismatch",
            subject=subject,
        )

    sigma = np.asarray(cache["sigma"], dtype=np.float32)
    require(
        report,
        np.asarray(cache["sigma"]).dtype == np.float32,
        "Sigma cache dtype is not float32",
        subject=subject,
        dtype=str(np.asarray(cache["sigma"]).dtype),
    )
    if sigma.shape == (1, 9, 32):
        sigma_block = sigma[0]
    elif sigma.shape == (9, 32):
        sigma_block = sigma
    else:
        add_issue(
            report,
            "failures",
            "Sigma shape must be [9,32] or [1,9,32]",
            subject=subject,
            shape=list(sigma.shape),
        )
        return None
    require(
        report,
        bool(np.isfinite(sigma_block).all()),
        "Sigma contains NaN/Inf",
        subject=subject,
    )
    require(
        report,
        bool(np.all(sigma_block > 0)),
        "Sigma is not strictly positive",
        subject=subject,
    )
    try:
        source_checkpoint = torch.load(
            source["nbm_best"],
            map_location="cpu",
            weights_only=False,
        )
        source_model_config = source_checkpoint["model_config"]
        log_sigma = (
            source_checkpoint["model_state"]["log_sigma"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        minimum = float(source_model_config.get("min_log_sigma", -3.0))
        maximum = float(source_model_config.get("max_log_sigma", 1.5))
        checkpoint_sigma = np.exp(
            np.clip(log_sigma, minimum, maximum)
        ).astype(np.float32, copy=False)
        require(
            report,
            checkpoint_sigma.shape == sigma.shape
            and bool(
                np.allclose(
                    checkpoint_sigma,
                    sigma,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "Cached sigma differs from frozen Persistence checkpoint",
            subject=subject,
            max_abs_diff=(
                float(
                    np.max(
                        np.abs(checkpoint_sigma - sigma),
                        initial=0.0,
                    )
                )
                if checkpoint_sigma.shape == sigma.shape
                else None
            ),
        )
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot reconstruct sigma from source Persistence checkpoint",
            subject=subject,
            error=f"{type(error).__name__}: {error}",
        )

    try:
        scaler_payload = source["fold_config"]["scaler"]
        source_scaler = RobustChannelScaler(
            center=np.asarray(
                scaler_payload["center"],
                dtype=np.float32,
            ),
            scale=np.asarray(
                scaler_payload["scale"],
                dtype=np.float32,
            ),
            clip=float(scaler_payload["clip"]),
        )
        refitted_scaler = dataset.fit_scaler(
            list(source["fold_config"]["train_subjects"]),
            clip=float(scaler_payload["clip"]),
        )
        require(
            report,
            bool(
                np.array_equal(
                    refitted_scaler.center,
                    source_scaler.center,
                )
                and np.array_equal(
                    refitted_scaler.scale,
                    source_scaler.scale,
                )
            ),
            "Source scaler was not fitted exclusively from fold training subjects",
            subject=subject,
        )
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot reconstruct source fold scaler",
            subject=subject,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    try:
        canonical = _source_residuals(source["residual_cache"])
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load canonical source residual cache",
            subject=subject,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    source_support_path = Path(source["support_path"])
    try:
        with np.load(source_support_path, allow_pickle=False) as payload:
            source_support = {
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
                for split in EXPECTED_SPLITS
            }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load canonical source history support",
            subject=subject,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    block_key = {
        "raw_support_matched": "raw",
        "error_x_minus_mu": "error",
        "standardized_error": "standardized_error",
        "standardized_error_clip12": "standardized_error_clip12",
    }
    max_formula_error = {
        "error": 0.0,
        "standardized_error": 0.0,
        "standardized_error_clip12": 0.0,
        "canonical_residual": 0.0,
    }

    for split in EXPECTED_SPLITS:
        for key in (
            "raw",
            "mu",
            "error",
            "standardized_error",
            "standardized_error_clip12",
        ):
            require(
                report,
                np.asarray(cache[f"{split}_{key}"]).dtype == np.float32,
                "Representation block dtype is not float32",
                subject=subject,
                split=split,
                key=key,
                dtype=str(np.asarray(cache[f"{split}_{key}"]).dtype),
            )
        require(
            report,
            np.asarray(cache[f"{split}_y"]).dtype == np.int8,
            "Representation label dtype is not int8",
            subject=subject,
            split=split,
        )
        require(
            report,
            np.asarray(cache[f"{split}_window_index"]).dtype == np.int64,
            "Representation window-index dtype is not int64",
            subject=subject,
            split=split,
        )
        raw = np.asarray(cache[f"{split}_raw"], dtype=np.float32)
        mu = np.asarray(cache[f"{split}_mu"], dtype=np.float32)
        error = np.asarray(cache[f"{split}_error"], dtype=np.float32)
        standard = np.asarray(
            cache[f"{split}_standardized_error"],
            dtype=np.float32,
        )
        clipped = np.asarray(
            cache[f"{split}_standardized_error_clip12"],
            dtype=np.float32,
        )
        y = np.asarray(cache[f"{split}_y"], dtype=np.int8)
        indices = np.asarray(
            cache[f"{split}_window_index"],
            dtype=np.int64,
        )
        arrays = (raw, mu, error, standard, clipped)
        if not all(array.shape == raw.shape for array in arrays):
            add_issue(
                report,
                "failures",
                "Block representation shapes differ",
                subject=subject,
                split=split,
                shapes=[list(array.shape) for array in arrays],
            )
            continue
        require(
            report,
            raw.ndim == 3 and raw.shape[1:] == (9, 32),
            "Block representation is not [N,9,32]",
            subject=subject,
            split=split,
            shape=list(raw.shape),
        )
        require(
            report,
            len(raw) == len(y) == len(indices),
            "Block cache arrays are misaligned",
            subject=subject,
            split=split,
        )
        require(
            report,
            all(bool(np.isfinite(array).all()) for array in arrays),
            "Block representation contains NaN/Inf",
            subject=subject,
            split=split,
        )
        require(
            report,
            bool(np.all(np.abs(raw) <= 12.0 + tolerance)),
            "Robust-scaled raw target exceeds common clip",
            subject=subject,
            split=split,
        )
        reconstructed_raw = np.empty_like(raw)
        reconstructed_mu = np.empty_like(mu)
        for row_index, global_index in enumerate(indices):
            record = dataset.records[
                int(windows.record_index[int(global_index)])
            ]
            target_start = int(windows.target_start[int(global_index)])
            target_end = int(windows.target_end[int(global_index)])
            reconstructed_raw[row_index] = source_scaler.transform(
                record.x[target_start:target_end]
            ).T
            latest = source_scaler.transform(
                record.x[target_start - 1 : target_start]
            )[0]
            reconstructed_mu[row_index] = np.repeat(
                latest[:, None],
                int(suite.HORIZON_SAMPLES),
                axis=1,
            )
        raw_max_abs_diff = float(
            np.max(
                np.abs(raw - reconstructed_raw),
                initial=0.0,
            )
        )
        mu_max_abs_diff = float(
            np.max(
                np.abs(mu - reconstructed_mu),
                initial=0.0,
            )
        )
        require(
            report,
            bool(
                np.allclose(
                    raw,
                    reconstructed_raw,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "Raw blocks differ from fold-scaled physical target samples",
            subject=subject,
            split=split,
            max_abs_diff=raw_max_abs_diff,
        )
        require(
            report,
            bool(
                np.allclose(
                    mu,
                    reconstructed_mu,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "Persistence mu is not the repeated latest context sample",
            subject=subject,
            split=split,
            max_abs_diff=mu_max_abs_diff,
        )
        expected_error = raw - mu
        expected_standard = expected_error / sigma_block[None, :, :]
        expected_clipped = np.clip(expected_standard, -12.0, 12.0)
        formula_errors = {
            "error": float(
                np.max(np.abs(error - expected_error), initial=0.0)
            ),
            "standardized_error": float(
                np.max(
                    np.abs(standard - expected_standard),
                    initial=0.0,
                )
            ),
            "standardized_error_clip12": float(
                np.max(
                    np.abs(clipped - expected_clipped),
                    initial=0.0,
                )
            ),
        }
        for key, difference in formula_errors.items():
            max_formula_error[key] = max(
                max_formula_error[key],
                difference,
            )
        require(
            report,
            bool(
                np.allclose(
                    error,
                    expected_error,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "error != raw - mu",
            subject=subject,
            split=split,
            max_abs_diff=formula_errors["error"],
        )
        require(
            report,
            bool(
                np.allclose(
                    standard,
                    expected_standard,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "standardized_error != error / sigma",
            subject=subject,
            split=split,
            max_abs_diff=formula_errors["standardized_error"],
        )
        require(
            report,
            bool(
                np.allclose(
                    clipped,
                    expected_clipped,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ),
            "clipped standardized error formula mismatch",
            subject=subject,
            split=split,
            max_abs_diff=formula_errors[
                "standardized_error_clip12"
            ],
        )
        require(
            report,
            bool(np.all(np.abs(clipped) <= 12.0 + tolerance)),
            "Clipped standardized error exceeds [-12,12]",
            subject=subject,
            split=split,
        )

        canonical_split = canonical[split]
        require(
            report,
            np.array_equal(
                indices,
                canonical_split["window_index"],
            ),
            "Representation cache block support differs from source cache",
            subject=subject,
            split=split,
        )
        require(
            report,
            np.array_equal(y, canonical_split["y"]),
            "Representation-cache labels differ from source cache",
            subject=subject,
            split=split,
        )
        if clipped.shape == canonical_split["residual"].shape:
            canonical_difference = float(
                np.max(
                    np.abs(clipped - canonical_split["residual"]),
                    initial=0.0,
                )
            )
            max_formula_error["canonical_residual"] = max(
                max_formula_error["canonical_residual"],
                canonical_difference,
            )
            require(
                report,
                bool(
                    np.allclose(
                        clipped,
                        canonical_split["residual"],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                ),
                "Complete representation differs from canonical residual",
                subject=subject,
                split=split,
                max_abs_diff=canonical_difference,
            )
        else:
            add_issue(
                report,
                "failures",
                "Complete/canonical residual shapes differ",
                subject=subject,
                split=split,
            )

        split_support = support[split]
        require(
            report,
            split_support["anchor"].ndim == 1,
            "Anchor support must be one-dimensional",
            subject=subject,
            split=split,
        )
        require(
            report,
            split_support["history"].ndim == 2
            and split_support["history"].shape[1] == 8,
            "History support must have shape [N,8]",
            subject=subject,
            split=split,
            shape=list(split_support["history"].shape),
        )
        require(
            report,
            len(split_support["anchor"])
            == len(split_support["history"])
            == len(split_support["y"]),
            "Common history support arrays are misaligned",
            subject=subject,
            split=split,
        )
        if config.get("run_kind", "formal") == "formal":
            support_matches_source = np.array_equal(
                split_support["anchor"],
                source_support[split]["anchor"],
            ) and np.array_equal(
                split_support["history"],
                source_support[split]["history"],
            )
        else:
            source_rows = {
                int(anchor): row
                for row, anchor in enumerate(
                    source_support[split]["anchor"]
                )
            }
            support_matches_source = all(
                int(anchor) in source_rows
                and np.array_equal(
                    history,
                    source_support[split]["history"][
                        source_rows[int(anchor)]
                    ],
                )
                for anchor, history in zip(
                    split_support["anchor"],
                    split_support["history"],
                )
            )
        require(
            report,
            support_matches_source,
            (
                "Common support differs from canonical residual_h4s support"
                if config.get("run_kind", "formal") == "formal"
                else "Smoke support is not an exact subset of canonical support"
            ),
            subject=subject,
            split=split,
        )
        require(
            report,
            np.array_equal(
                split_support["history"][:, -1],
                split_support["anchor"],
            ),
            "Final history block is not the classifier anchor",
            subject=subject,
            split=split,
        )
        require(
            report,
            np.array_equal(
                split_support["y"],
                windows.label[split_support["anchor"]],
            ),
            "Support labels differ from final target-block labels",
            subject=subject,
            split=split,
        )

        row_lookup = {
            int(window_index): row
            for row, window_index in enumerate(indices.tolist())
        }
        try:
            history_rows = np.asarray(
                [
                    [row_lookup[int(window_index)] for window_index in chain]
                    for chain in split_support["history"]
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            add_issue(
                report,
                "failures",
                "History support references a missing cache block",
                subject=subject,
                split=split,
                window_index=int(error.args[0]),
            )
            continue
        for row in split_support["history"]:
            starts = windows.target_start[row]
            records = windows.record_index[row]
            if not (
                np.all(records == records[0])
                and np.array_equal(
                    np.diff(starts),
                    np.full(7, 32, dtype=np.int64),
                )
            ):
                add_issue(
                    report,
                    "failures",
                    "History blocks are not contiguous record-local 0.5 s blocks",
                    subject=subject,
                    split=split,
                )
                break
        for name, key in block_key.items():
            block = np.asarray(cache[f"{split}_{key}"], dtype=np.float32)
            selected = block[history_rows]
            history = selected.transpose(0, 2, 1, 3).reshape(
                len(history_rows),
                9,
                256,
            )
            require(
                report,
                history.shape
                == (len(split_support["anchor"]), 9, 256),
                "Materialized classifier input shape mismatch",
                subject=subject,
                split=split,
                representation=name,
            )
            require(
                report,
                bool(np.isfinite(history).all()),
                "Materialized classifier input contains NaN/Inf",
                subject=subject,
                split=split,
                representation=name,
            )

    classifier_seed = int(
        fold_config.get(
            "classifier_seed",
            42 + 10000 + EXPECTED_SUBJECTS.index(subject),
        )
    )
    rf.set_seed(classifier_seed, bool(config.get("deterministic", True)))
    classifier_config = config.get("classifier", {})
    reference_model = rf.build_model(
        in_channels=9,
        hidden_channels=int(
            _first(
                classifier_config,
                ("hidden_channels",),
                config.get("classifier_hidden", 48),
            )
        ),
        dropout=float(
            _first(
                classifier_config,
                ("dropout",),
                config.get("classifier_dropout", 0.2),
            )
        ),
        dilations=tuple(int(value) for value in suite.TCN_M_DILATIONS),
    )
    initial_sha = rf.state_dict_sha256(reference_model.state_dict())
    parameter_count = rf.parameter_count(reference_model)
    del reference_model
    expected_initial = _first(
        fold_config,
        (
            "reference_initial_state_sha256",
            "classifier_initial_state_sha256",
        ),
    )
    require(
        report,
        expected_initial == initial_sha,
        "Fold classifier initial-state hash mismatch",
        subject=subject,
        saved=expected_initial,
        recomputed=initial_sha,
    )
    expected_parameters = _first(
        classifier_config,
        ("parameter_count",),
        config.get("shared_parameter_count"),
    )
    if expected_parameters is not None:
        require(
            report,
            int(expected_parameters) == parameter_count,
            "TCN-M parameter count mismatch",
            subject=subject,
        )

    return {
        "fold_root": fold_root,
        "fold_config": fold_config,
        "cache_path": cache_path,
        "cache_sha256": cache_sha,
        "support": support,
        "support_path": support_path,
        "support_sha256": support_sha,
        "classifier_seed": classifier_seed,
        "initial_state_sha256": initial_sha,
        "parameter_count": parameter_count,
        "max_formula_error": max_formula_error,
    }


def load_predictions(
    report: dict[str, Any],
    path: Path,
    label: str,
) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != PREDICTION_KEYS:
                add_issue(
                    report,
                    "failures",
                    "Prediction key set mismatch",
                    cell=label,
                    path=str(path),
                    observed=sorted(payload.files),
                )
                return None
            arrays = {
                "window_index": np.asarray(
                    payload["window_index"],
                    dtype=np.int64,
                ),
                "y_true": np.asarray(payload["y_true"], dtype=np.int8),
                "y_prob": np.asarray(
                    payload["y_prob"],
                    dtype=np.float64,
                ),
                "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
            }
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load predictions",
            cell=label,
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )
        return None
    lengths = {len(array) for array in arrays.values()}
    require(
        report,
        all(array.ndim == 1 for array in arrays.values())
        and len(lengths) == 1,
        "Prediction arrays are not aligned one-dimensional arrays",
        cell=label,
    )
    require(
        report,
        bool(np.isfinite(arrays["y_prob"]).all()),
        "Prediction probabilities contain NaN/Inf",
        cell=label,
    )
    require(
        report,
        bool(
            np.all(
                (arrays["y_prob"] >= 0.0)
                & (arrays["y_prob"] <= 1.0)
            )
        ),
        "Prediction probabilities fall outside [0,1]",
        cell=label,
    )
    require(
        report,
        set(np.unique(arrays["y_true"])).issubset({0, 1})
        and set(np.unique(arrays["y_pred"])).issubset({0, 1}),
        "Predictions/labels are not binary",
        cell=label,
    )
    return arrays


def requested_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    metrics = binary_metrics(y_true, y_prob, threshold)
    return rf.add_requested_metrics(metrics)


def audit_cell(
    report: dict[str, Any],
    result_dir: Path,
    config: Mapping[str, Any],
    representation: Mapping[str, Any],
    subject: str,
    fold: Mapping[str, Any],
    dataset: Any,
    windows: Any,
) -> dict[str, Any] | None:
    name = representation_id(representation)
    task_id = f"{subject}/{name}"
    root = result_dir / f"loso_{subject}" / name
    done_path = root / "DONE.json"
    if not done_path.exists():
        return None
    done = validate_done_collect(
        report,
        done_path,
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    if done is None:
        return None
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != CLASSIFIER_ARTIFACTS:
        add_issue(
            report,
            "failures",
            "Classifier DONE artifact set mismatch",
            cell=task_id,
            observed=(
                sorted(artifacts)
                if isinstance(artifacts, Mapping)
                else None
            ),
        )
        return None
    metadata_done_path = root / "REPRESENTATION_METADATA_DONE.json"
    metadata_done = validate_done_collect(
        report,
        metadata_done_path,
        stage="representation_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{task_id}/representation_metadata",
        upstream_sha256=sha256_file(done_path),
    )
    if metadata_done is None:
        return None
    metadata_path = root / "representation_metadata.json"
    if not metadata_path.exists():
        add_issue(
            report,
            "failures",
            "Completed representation metadata is missing",
            cell=task_id,
        )
        return None
    try:
        expected_metadata = suite.representation_metadata_payload(
            config,
            fold["fold_config"],
            representation,
        )
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot independently construct representation metadata",
            cell=task_id,
            error=f"{type(error).__name__}: {error}",
        )
        return None
    metadata = load_json(metadata_path)
    require(
        report,
        metadata == expected_metadata,
        "Representation metadata differs from protocol",
        cell=task_id,
    )
    metadata_artifacts = metadata_done.get("artifacts")
    require(
        report,
        isinstance(metadata_artifacts, Mapping)
        and set(metadata_artifacts) == {"metadata"}
        and artifact_path(
            metadata_done_path,
            metadata_artifacts["metadata"],
        ).resolve()
        == metadata_path.resolve(),
        "Representation metadata DONE artifact mismatch",
        cell=task_id,
    )

    metrics_path = artifact_path(done_path, artifacts["metrics"])
    predictions_path = artifact_path(done_path, artifacts["predictions"])
    validation_path = artifact_path(
        done_path,
        artifacts["validation_predictions"],
    )
    best_path = artifact_path(done_path, artifacts["best"])
    last_path = artifact_path(done_path, artifacts["last"])
    metrics = load_json(metrics_path)
    test = load_predictions(report, predictions_path, task_id + "/test")
    validation = load_predictions(
        report,
        validation_path,
        task_id + "/validation",
    )
    if test is None or validation is None:
        return None

    for split, arrays in (("test", test), ("validation", validation)):
        expected_support = fold["support"][split]
        require(
            report,
            np.array_equal(
                arrays["window_index"],
                expected_support["anchor"],
            ),
            "Classifier prediction support differs from common support",
            cell=task_id,
            split=split,
        )
        require(
            report,
            np.array_equal(arrays["y_true"], expected_support["y"]),
            "Classifier prediction labels differ from common support",
            cell=task_id,
            split=split,
        )

    threshold = float(metrics.get("threshold", float("nan")))
    require(
        report,
        math.isfinite(threshold) and 0.0 <= threshold <= 1.0,
        "Saved threshold is invalid",
        cell=task_id,
        threshold=threshold,
    )
    selected_threshold, selected_validation = choose_threshold(
        validation["y_true"],
        validation["y_prob"],
    )
    require(
        report,
        value_equal(threshold, selected_threshold, tolerance=1e-12),
        "Threshold was not selected from validation Balanced Accuracy",
        cell=task_id,
        saved=threshold,
        recomputed=selected_threshold,
    )
    require(
        report,
        np.array_equal(
            validation["y_pred"],
            (validation["y_prob"] >= threshold).astype(np.int8),
        ),
        "Validation y_pred differs from probability >= threshold",
        cell=task_id,
    )
    require(
        report,
        np.array_equal(
            test["y_pred"],
            (test["y_prob"] >= threshold).astype(np.int8),
        ),
        "Test y_pred differs from probability >= threshold",
        cell=task_id,
    )

    recomputed_validation = requested_metrics(
        validation["y_true"],
        validation["y_prob"],
        threshold,
    )
    saved_validation = metrics.get("validation")
    if isinstance(saved_validation, Mapping):
        validate_mapping(
            report,
            saved_validation,
            selected_validation,
            selected_validation.keys(),
            task_id + "/selected_validation_metrics",
            tolerance=1e-9,
        )
    else:
        add_issue(
            report,
            "failures",
            "metrics.json lacks validation metrics",
            cell=task_id,
        )

    recomputed_test = requested_metrics(
        test["y_true"],
        test["y_prob"],
        threshold,
    )
    recomputed_events = rf.event_metrics(
        dataset,
        windows,
        test["window_index"],
        test["y_pred"],
    )
    recomputed_test.update(recomputed_events)
    validate_mapping(
        report,
        metrics,
        recomputed_test,
        (*WINDOW_METRIC_KEYS, *EVENT_METRIC_KEYS),
        task_id + "/test_metrics",
        tolerance=1e-9,
    )

    expected_identity = {
        "test_subject": subject,
        "variant": name,
        "experiment_id": representation.get(
            "experiment_id",
            name,
        ),
        "classifier_seed": fold["classifier_seed"],
        "initial_state_sha256": fold["initial_state_sha256"],
    }
    for key, expected in expected_identity.items():
        if key in metrics:
            require(
                report,
                metrics[key] == expected,
                "Classifier metrics identity mismatch",
                cell=task_id,
                key=key,
                saved=metrics[key],
                expected=expected,
            )
    for key, expected in (
        ("initial_state_sha256", fold["initial_state_sha256"]),
        ("input_support_sha256", fold["support_sha256"]),
    ):
        if key in done:
            require(
                report,
                done[key] == expected,
                "Classifier DONE provenance mismatch",
                cell=task_id,
                key=key,
            )

    try:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001
        add_issue(
            report,
            "failures",
            "Cannot load classifier checkpoints",
            cell=task_id,
            error=f"{type(error).__name__}: {error}",
        )
        return None
    for checkpoint_name, checkpoint in (("best", best), ("last", last)):
        require(
            report,
            checkpoint.get("protocol_fingerprint")
            == config.get("protocol_fingerprint"),
            "Classifier checkpoint protocol mismatch",
            cell=task_id,
            checkpoint=checkpoint_name,
        )
        require(
            report,
            checkpoint.get("task_id") == task_id,
            "Classifier checkpoint task mismatch",
            cell=task_id,
            checkpoint=checkpoint_name,
        )
        model_config = checkpoint.get("classifier_config")
        if not isinstance(model_config, Mapping):
            add_issue(
                report,
                "failures",
                "Classifier checkpoint lacks model configuration",
                cell=task_id,
                checkpoint=checkpoint_name,
            )
            continue
        require(
            report,
            tuple(int(value) for value in model_config.get("dilations", ()))
            == tuple(int(value) for value in suite.TCN_M_DILATIONS),
            "Classifier checkpoint is not TCN-M",
            cell=task_id,
            checkpoint=checkpoint_name,
        )
        require(
            report,
            int(model_config.get("receptive_field_samples", -1)) == 125,
            "Classifier checkpoint RF is not 125",
            cell=task_id,
            checkpoint=checkpoint_name,
        )
        require(
            report,
            model_config.get("initial_state_sha256")
            == fold["initial_state_sha256"],
            "Checkpoint initial-state hash differs across representations",
            cell=task_id,
            checkpoint=checkpoint_name,
        )
        require(
            report,
            int(model_config.get("parameter_count", -1))
            == int(fold["parameter_count"]),
            "Checkpoint parameter count differs from TCN-M reference",
            cell=task_id,
            checkpoint=checkpoint_name,
        )

    history = metrics.get("history")
    if isinstance(history, list):
        expected_shuffle = [
            int(fold["classifier_seed"]) + int(row["epoch"])
            for row in history
        ]
        observed_shuffle = [int(row["shuffle_seed"]) for row in history]
        require(
            report,
            observed_shuffle == expected_shuffle,
            "Per-epoch shuffle seeds changed",
            cell=task_id,
        )
    else:
        add_issue(
            report,
            "failures",
            "Classifier metrics lack training history",
            cell=task_id,
        )

    return {
        "subject": subject,
        "representation": name,
        "experiment_id": representation.get("experiment_id", name),
        "metrics": {**metrics, **metadata},
        "test": test,
        "validation": validation,
        "recomputed_test": recomputed_test,
        "recomputed_validation": recomputed_validation,
        "initial_state_sha256": fold["initial_state_sha256"],
        "shuffle_seeds": (
            [int(row["shuffle_seed"]) for row in history]
            if isinstance(history, list)
            else []
        ),
    }


def validate_pure_ablation(
    report: dict[str, Any],
    subject: str,
    cells: Sequence[Mapping[str, Any]],
) -> None:
    if len(cells) < 2:
        return
    first = cells[0]
    for cell in cells[1:]:
        require(
            report,
            np.array_equal(
                first["test"]["window_index"],
                cell["test"]["window_index"],
            )
            and np.array_equal(
                first["test"]["y_true"],
                cell["test"]["y_true"],
            ),
            "Representations do not share exact test support/labels",
            subject=subject,
            left=first["representation"],
            right=cell["representation"],
        )
        require(
            report,
            np.array_equal(
                first["validation"]["window_index"],
                cell["validation"]["window_index"],
            )
            and np.array_equal(
                first["validation"]["y_true"],
                cell["validation"]["y_true"],
            ),
            "Representations do not share exact validation support/labels",
            subject=subject,
            left=first["representation"],
            right=cell["representation"],
        )
        require(
            report,
            first["initial_state_sha256"]
            == cell["initial_state_sha256"],
            "Representations do not share classifier initialization",
            subject=subject,
        )
        require(
            report,
            first["shuffle_seeds"] == cell["shuffle_seeds"],
            "Representations do not share epoch shuffle order",
            subject=subject,
        )


def comparison_statistics(
    config: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    cells_by_representation: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    samples = int(config.get("bootstrap_samples", 100000))
    base_seed = int(config.get("bootstrap_seed", 42))
    for comparison in comparisons:
        comparison_id = str(comparison["comparison_id"])
        current_name = str(comparison["new_representation"])
        reference_name = str(comparison["reference_representation"])
        common: list[str] = []
        differences: list[float] = []
        for subject in EXPECTED_SUBJECTS:
            current = cells_by_representation[current_name].get(subject)
            reference = cells_by_representation[reference_name].get(subject)
            if current is None or reference is None:
                continue
            current_value = current["recomputed_test"].get("pr_auc")
            reference_value = reference["recomputed_test"].get("pr_auc")
            if current_value is None or reference_value is None:
                continue
            common.append(subject)
            differences.append(float(current_value) - float(reference_value))
        array = np.asarray(differences, dtype=np.float64)
        seed = suite.stable_bootstrap_seed(
            base_seed,
            str(comparison["bootstrap_label"]),
        )
        summary = suite.paired_bootstrap_mean_ci(
            array,
            samples,
            seed,
        )
        tie_tolerance = 1e-12
        wins = int(np.sum(array > tie_tolerance))
        losses = int(np.sum(array < -tie_tolerance))
        ties = int(len(array) - wins - losses)
        result[comparison_id] = {
            "comparison_id": comparison_id,
            "new_representation": current_name,
            "reference_representation": reference_name,
            "common_subjects": ",".join(common),
            **summary,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "bootstrap_seed": base_seed,
            "bootstrap_resolved_seed": int(seed),
        }
    return result


def _row_identity(
    row: Mapping[str, Any],
) -> tuple[str, str]:
    representation = str(
        _first(
            row,
            ("representation", "representation_id", "variant"),
            "",
        )
    )
    subject = str(
        _first(row, ("test_subject", "subject", "fold"), "")
    )
    return representation, subject


def _csv_number(value: str | None) -> Any:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"none", "nan", "na"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def audit_root_summaries(
    report: dict[str, Any],
    result_dir: Path,
    config: Mapping[str, Any],
    representations: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
    allow_partial: bool,
) -> None:
    cells_by_representation: dict[
        str, dict[str, Mapping[str, Any]]
    ] = {
        representation_id(item): {} for item in representations
    }
    for cell in completed:
        cells_by_representation[cell["representation"]][
            cell["subject"]
        ] = cell
    aggregates: dict[str, dict[str, Any]] = {}
    for representation in representations:
        name = representation_id(representation)
        rows = [
            cells_by_representation[name][subject]["recomputed_test"]
            for subject in EXPECTED_SUBJECTS
            if subject in cells_by_representation[name]
        ]
        aggregates[name] = aggregate_fold_metrics(
            rows,
            list(suite.CLASSIFICATION_METRICS),
        )
    all_cells_complete = len(completed) == 32
    ranked = sorted(
        (
            (
                -float(aggregates[name]["pr_auc"]["mean"]),
                name,
            )
            for name in EXPECTED_REPRESENTATIONS
            if aggregates[name]["pr_auc"]["mean"] is not None
        ),
        key=lambda item: (item[0], item[1]),
    )
    expected_best = (
        next(
            (
                str(item.get("experiment_id", representation_id(item)))
                for item in representations
                if representation_id(item) == ranked[0][1]
            ),
            ranked[0][1],
        )
        if all_cells_complete and ranked
        else None
    )
    paired = comparison_statistics(
        config,
        comparisons,
        cells_by_representation,
    )

    required_root = (
        "fold_summary.csv",
        "experiment_manifest.csv",
        "aggregate_summary.csv",
        "paired_pr_auc_deltas.csv",
        "publication_table.csv",
        "input_diagnostics.csv",
        "sigma_diagnostics.csv",
        "support_equivalence.json",
        "aggregate_metrics.json",
        "status.json",
    )
    for filename in required_root:
        path = result_dir / filename
        if not path.exists():
            level = "warnings" if allow_partial else "failures"
            add_issue(
                report,
                level,
                "Root summary artifact is missing",
                path=str(path),
            )

    fold_path = result_dir / "fold_summary.csv"
    if fold_path.exists():
        rows = load_csv(fold_path)
        indexed = {_row_identity(row): row for row in rows}
        expected_ids = {
            (cell["representation"], cell["subject"])
            for cell in completed
        }
        require(
            report,
            set(indexed) == expected_ids,
            "fold_summary.csv cell identities differ from completed cells",
        )
        for cell in completed:
            key = (cell["representation"], cell["subject"])
            row = indexed.get(key)
            if row is None:
                continue
            for metric in suite.CLASSIFICATION_METRICS:
                if metric in row:
                    require(
                        report,
                        value_equal(
                            _csv_number(row[metric]),
                            cell["recomputed_test"].get(metric),
                            tolerance=1e-9,
                        ),
                        "fold_summary.csv metric mismatch",
                        cell=f"{key[1]}/{key[0]}",
                        metric=metric,
                    )

    manifest_path = result_dir / "experiment_manifest.csv"
    if manifest_path.exists():
        rows = load_csv(manifest_path)
        indexed = {
            str(
                _first(
                    row,
                    ("representation", "representation_id", "variant"),
                    "",
                )
            ): row
            for row in rows
        }
        require(
            report,
            set(indexed) == set(EXPECTED_REPRESENTATIONS),
            "experiment_manifest.csv representation rows mismatch",
        )
        representation_definitions = {
            representation_id(item): item for item in representations
        }
        for name, definition in representation_definitions.items():
            row = indexed.get(name)
            if row is None:
                continue
            completed_count = len(cells_by_representation[name])
            expected_status = (
                "complete"
                if completed_count == len(EXPECTED_SUBJECTS)
                else ("partial" if completed_count else "pending")
            )
            expectations = {
                "experiment_id": definition.get("experiment_id", name),
                "expected_folds": len(EXPECTED_SUBJECTS),
                "completed_folds": completed_count,
                "status": expected_status,
                "completed_subjects": ",".join(
                    subject
                    for subject in EXPECTED_SUBJECTS
                    if subject in cells_by_representation[name]
                ),
            }
            for key, expected in expectations.items():
                if key in row:
                    saved_value = (
                        _csv_number(row[key])
                        if isinstance(expected, (int, float))
                        else row[key]
                    )
                    require(
                        report,
                        value_equal(saved_value, expected),
                        "experiment_manifest.csv value mismatch",
                        representation=name,
                        key=key,
                    )

    aggregate_path = result_dir / "aggregate_summary.csv"
    if aggregate_path.exists():
        rows = load_csv(aggregate_path)
        indexed = {
            str(
                _first(
                    row,
                    ("representation", "representation_id", "variant"),
                    "",
                )
            ): row
            for row in rows
        }
        require(
            report,
            set(indexed) == set(EXPECTED_REPRESENTATIONS),
            "aggregate_summary.csv representation rows mismatch",
        )
        for name, summary in aggregates.items():
            row = indexed.get(name)
            if row is None:
                continue
            expected_completed = len(cells_by_representation[name])
            if "completed_folds" in row:
                require(
                    report,
                    _csv_number(row["completed_folds"])
                    == expected_completed,
                    "aggregate_summary.csv completed-fold count mismatch",
                    representation=name,
                )
            for metric, values in summary.items():
                for statistic in ("mean", "std"):
                    key = f"{metric}_{statistic}"
                    if key in row:
                        require(
                            report,
                            value_equal(
                                _csv_number(row[key]),
                                values[statistic],
                                tolerance=1e-9,
                            ),
                            "aggregate_summary.csv metric mismatch",
                            representation=name,
                            key=key,
                        )

    delta_path = result_dir / "paired_pr_auc_deltas.csv"
    if delta_path.exists():
        rows = load_csv(delta_path)
        indexed = {
            str(row.get("comparison_id", "")): row for row in rows
        }
        require(
            report,
            set(indexed) == set(paired),
            "paired_pr_auc_deltas.csv comparison rows mismatch",
        )
        aliases = {
            "mean_delta": ("mean_delta", "delta_pr_auc_mean"),
            "ci_low": ("ci_low", "delta_pr_auc_ci_low"),
            "ci_high": ("ci_high", "delta_pr_auc_ci_high"),
            "n_paired_subjects": (
                "n_paired_subjects",
                "delta_pr_auc_n_paired_subjects",
            ),
            "wins": ("wins", "wins_over_8"),
            "ties": ("ties",),
            "losses": ("losses",),
        }
        for comparison_id, expected in paired.items():
            row = indexed.get(comparison_id)
            if row is None:
                continue
            for expected_key, candidates in aliases.items():
                saved_key = next(
                    (key for key in candidates if key in row),
                    None,
                )
                if saved_key is None:
                    add_issue(
                        report,
                        "failures",
                        "paired_pr_auc_deltas.csv missing field",
                        comparison_id=comparison_id,
                        field=expected_key,
                    )
                    continue
                require(
                    report,
                    value_equal(
                        _csv_number(row[saved_key]),
                        expected[expected_key],
                        tolerance=1e-12,
                    ),
                    "paired_pr_auc_deltas.csv value mismatch",
                    comparison_id=comparison_id,
                    field=expected_key,
                )

    aggregate_json_path = result_dir / "aggregate_metrics.json"
    if aggregate_json_path.exists():
        aggregate_json = load_json(aggregate_json_path)
        require(
            report,
            aggregate_json.get("best_experiment") == expected_best,
            "aggregate_metrics.json best_experiment mismatch",
            saved=aggregate_json.get("best_experiment"),
            recomputed=expected_best,
        )
        saved_comparisons = aggregate_json.get(
            "paired_pr_auc_comparisons"
        )
        if isinstance(saved_comparisons, list):
            saved_by_id = {
                str(item.get("comparison_id")): item
                for item in saved_comparisons
                if isinstance(item, Mapping)
            }
            require(
                report,
                set(saved_by_id) == set(paired),
                "aggregate_metrics paired-comparison identities mismatch",
            )
            for comparison_id, expected in paired.items():
                saved = saved_by_id.get(comparison_id)
                if saved is None:
                    continue
                for key in (
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
                ):
                    require(
                        report,
                        value_equal(
                            saved.get(key),
                            expected.get(key),
                            tolerance=1e-12,
                        ),
                        "aggregate_metrics paired-comparison value mismatch",
                        comparison_id=comparison_id,
                        key=key,
                    )
        experiments = aggregate_json.get("experiments")
        if isinstance(experiments, Mapping):
            for representation in representations:
                name = representation_id(representation)
                experiment_id = str(
                    representation.get("experiment_id", name)
                )
                payload = experiments.get(experiment_id)
                if payload is None:
                    payload = experiments.get(name)
                if not isinstance(payload, Mapping):
                    add_issue(
                        report,
                        "failures",
                        "aggregate_metrics.json missing experiment",
                        representation=name,
                    )
                    continue
                saved_macro = payload.get("subject_macro")
                if not isinstance(saved_macro, Mapping):
                    add_issue(
                        report,
                        "failures",
                        "aggregate_metrics experiment lacks subject_macro",
                        representation=name,
                    )
                    continue
                for metric, expected in aggregates[name].items():
                    saved = saved_macro.get(metric)
                    if not isinstance(saved, Mapping):
                        add_issue(
                            report,
                            "failures",
                            "aggregate_metrics subject_macro metric missing",
                            representation=name,
                            metric=metric,
                        )
                        continue
                    for statistic in ("mean", "std", "n_folds"):
                        require(
                            report,
                            value_equal(
                                saved.get(statistic),
                                expected.get(statistic),
                                tolerance=1e-9,
                            ),
                            "aggregate_metrics subject_macro mismatch",
                            representation=name,
                            metric=metric,
                            statistic=statistic,
                        )

    support_equivalence_path = result_dir / "support_equivalence.json"
    if support_equivalence_path.exists():
        support_equivalence = load_json(support_equivalence_path)
        completed_cache_subjects = [
            subject
            for subject in EXPECTED_SUBJECTS
            if (result_dir / f"loso_{subject}" / "REPRESENTATION_CACHE_DONE.json").exists()
        ]
        require(
            report,
            support_equivalence.get("representations")
            == list(EXPECTED_REPRESENTATIONS),
            "support_equivalence.json representation list mismatch",
        )
        require(
            report,
            support_equivalence.get("completed_support_subjects")
            == completed_cache_subjects,
            "support_equivalence.json completed-fold list mismatch",
        )
        require(
            report,
            bool(support_equivalence.get("complete"))
            == (completed_cache_subjects == list(EXPECTED_SUBJECTS)),
            "support_equivalence.json completeness flag mismatch",
        )

    input_diagnostics_path = result_dir / "input_diagnostics.csv"
    if input_diagnostics_path.exists():
        rows = load_csv(input_diagnostics_path)
        expected_rows = (
            len(
                {
                    cell["subject"]
                    for cell in completed
                }
            )
            * len(EXPECTED_SPLITS)
            * len(EXPECTED_REPRESENTATIONS)
        )
        # A cache can legitimately precede every classifier cell during a
        # partial run, so derive the stronger expected count from cache DONEs.
        cache_count = sum(
            (
                result_dir
                / f"loso_{subject}"
                / "REPRESENTATION_CACHE_DONE.json"
            ).exists()
            for subject in EXPECTED_SUBJECTS
        )
        expected_rows = cache_count * 12
        require(
            report,
            len(rows) == expected_rows,
            "input_diagnostics.csv row count mismatch",
            saved=len(rows),
            expected=expected_rows,
        )
        for row in rows:
            if "nonfinite_values" in row:
                require(
                    report,
                    _csv_number(row["nonfinite_values"]) == 0,
                    "input_diagnostics.csv reports non-finite values",
                    subject=row.get("test_subject"),
                    representation=row.get("representation"),
                    split=row.get("split"),
                )

    sigma_diagnostics_path = result_dir / "sigma_diagnostics.csv"
    if sigma_diagnostics_path.exists():
        rows = load_csv(sigma_diagnostics_path)
        cache_count = sum(
            (
                result_dir
                / f"loso_{subject}"
                / "REPRESENTATION_CACHE_DONE.json"
            ).exists()
            for subject in EXPECTED_SUBJECTS
        )
        require(
            report,
            len(rows) == cache_count * 9,
            "sigma_diagnostics.csv row count mismatch",
            saved=len(rows),
            expected=cache_count * 9,
        )
        for row in rows:
            for key in ("sigma_min", "sigma_median", "sigma_max"):
                require(
                    report,
                    float(row[key]) > 0.0,
                    "sigma_diagnostics.csv contains non-positive sigma",
                    subject=row.get("test_subject"),
                    key=key,
                )

    status_path = result_dir / "status.json"
    if status_path.exists():
        status = load_json(status_path)
        completed_count = len(completed)
        expected_status = (
            (
                "complete"
                if bool(config.get("reportable", True))
                else "smoke_complete"
            )
            if completed_count == 32
            else "partial"
        )
        for key, expected in (
            ("expected_experiments", 4),
            ("expected_classifier_cells", 32),
            ("completed_classifier_cells", completed_count),
            ("status", expected_status),
        ):
            if key in status:
                require(
                    report,
                    status[key] == expected,
                    "status.json value mismatch",
                    key=key,
                    saved=status[key],
                    expected=expected,
                )
        if "reportable" in status:
            require(
                report,
                bool(status["reportable"])
                == bool(config.get("reportable", True)),
                "status.json reportable flag mismatch",
            )
        if "best_experiment" in status:
            require(
                report,
                status["best_experiment"] == expected_best,
                "status.json best_experiment mismatch",
                saved=status["best_experiment"],
                recomputed=expected_best,
            )


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "Daphnet Persistence Input-Representation Ablation Audit",
        f"Audit version: {report['audit_version']}",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Completed cells: {report['completed_cells']}/{report['expected_cells']}",
        f"Validated fold caches: {report['validated_fold_caches']}/8",
        f"Failures: {len(report['failures'])}",
        f"Warnings: {len(report['warnings'])}",
    ]
    if report["failures"]:
        lines.extend(["", "Failures:"])
        for index, item in enumerate(report["failures"], start=1):
            lines.append(
                f"{index}. {item['message']} "
                f"{json.dumps({k: v for k, v in item.items() if k != 'message'}, ensure_ascii=False)}"
            )
    if report["warnings"]:
        lines.extend(["", "Warnings:"])
        for index, item in enumerate(report["warnings"], start=1):
            lines.append(
                f"{index}. {item['message']} "
                f"{json.dumps({k: v for k, v in item.items() if k != 'message'}, ensure_ascii=False)}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "suite_version": getattr(suite, "SUITE_VERSION", None),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(result_dir),
        "allow_partial": bool(args.allow_partial),
        "expected_cells": 32,
        "completed_cells": 0,
        "validated_fold_caches": 0,
        "missing_cells": [],
        "formula_max_abs_differences": {},
        "failures": [],
        "warnings": [],
        "status": "fail",
    }
    audited_config: dict[str, Any] | None = None
    complete_path = result_dir / "SUITE_COMPLETE.json"
    try:
        config, representations, comparisons = validate_protocol(
            report,
            result_dir,
        )
        if config is None:
            raise RuntimeError("Protocol cannot be audited")
        audited_config = config
        report["run_kind"] = config.get("run_kind", "formal")
        report["reportable"] = bool(config.get("reportable", True))
        source_suite_dir = choose_existing_path(
            configured_path(config, "source_suite_dir"),
            args.source_suite_dir,
            "canonical source suite",
        )
        data_dir = choose_existing_path(
            configured_path(config, "data_dir"),
            args.data_dir,
            "processed Daphnet data",
        )
        report["source_suite_dir"] = str(source_suite_dir)
        report["data_dir"] = str(data_dir)
        source_config, dataset, windows = validate_source_and_dataset(
            report,
            config,
            source_suite_dir,
            data_dir,
        )
        folds: dict[str, dict[str, Any]] = {}
        if source_config is not None and dataset is not None and windows is not None:
            for subject in EXPECTED_SUBJECTS:
                source = source_fold_artifacts(
                    report,
                    source_suite_dir,
                    source_config,
                    (
                        config["source"]
                        if isinstance(config.get("source"), Mapping)
                        else {}
                    ),
                    subject,
                )
                if source is None:
                    continue
                fold = validate_fold_cache_and_support(
                    report,
                    result_dir,
                    config,
                    source,
                    dataset,
                    windows,
                    subject,
                    float(args.tolerance),
                )
                if fold is not None:
                    folds[subject] = fold
                    report["formula_max_abs_differences"][subject] = fold[
                        "max_formula_error"
                    ]
        report["validated_fold_caches"] = len(folds)

        completed: list[dict[str, Any]] = []
        missing: list[str] = []
        if dataset is not None and windows is not None:
            for subject in EXPECTED_SUBJECTS:
                subject_cells: list[dict[str, Any]] = []
                for representation in representations:
                    name = representation_id(representation)
                    task_id = f"{subject}/{name}"
                    if subject not in folds:
                        classifier_done = (
                            result_dir
                            / f"loso_{subject}"
                            / name
                            / "DONE.json"
                        )
                        if classifier_done.exists():
                            add_issue(
                                report,
                                "failures",
                                "Classifier DONE exists without a valid fold representation cache",
                                cell=task_id,
                            )
                        missing.append(task_id)
                        continue
                    cell = audit_cell(
                        report,
                        result_dir,
                        config,
                        representation,
                        subject,
                        folds[subject],
                        dataset,
                        windows,
                    )
                    if cell is None:
                        missing.append(task_id)
                    else:
                        completed.append(cell)
                        subject_cells.append(cell)
                validate_pure_ablation(report, subject, subject_cells)
        else:
            missing = [
                f"{subject}/{representation_id(representation)}"
                for subject in EXPECTED_SUBJECTS
                for representation in representations
            ]
        report["completed_cells"] = len(completed)
        report["missing_cells"] = missing
        if missing and not args.allow_partial:
            add_issue(
                report,
                "failures",
                "Suite is incomplete; use --allow-partial only for interim audits",
                missing_count=len(missing),
            )
        audit_root_summaries(
            report,
            result_dir,
            config,
            representations,
            comparisons,
            completed,
            bool(args.allow_partial),
        )
    except Exception as error:  # noqa: BLE001 - always emit an audit report.
        add_issue(
            report,
            "failures",
            "Fatal audit exception",
            error=f"{type(error).__name__}: {error}",
        )

    full = (
        report["completed_cells"] == report["expected_cells"]
        and report["validated_fold_caches"] == len(EXPECTED_SUBJECTS)
        and bool(
            audited_config is not None
            and audited_config.get("reportable", True)
            and audited_config.get("run_kind", "formal") == "formal"
        )
    )
    if report["failures"]:
        report["status"] = "fail"
    elif full:
        report["status"] = "pass"
    else:
        report["status"] = "partial_pass"

    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / "AUDIT_REPORT.json"
    text_path = result_dir / "AUDIT_REPORT.txt"
    atomic_json_dump(report, report_path)
    write_text_report(text_path, report)
    if report["status"] == "pass" and full:
        atomic_json_dump(
            {
                "suite_version": suite.SUITE_VERSION,
                "audit_version": AUDIT_VERSION,
                "protocol_fingerprint": (
                    load_json(result_dir / "config.json").get(
                        "protocol_fingerprint"
                    )
                ),
                "expected_classifier_cells": 32,
                "completed_classifier_cells": 32,
                "audit_report_sha256": sha256_file(report_path),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            complete_path,
        )
    elif complete_path.exists():
        complete_path.unlink()

    print(
        f"[input-audit] status={report['status']} "
        f"cells={report['completed_cells']}/{report['expected_cells']} "
        f"fold_caches={report['validated_fold_caches']}/8 "
        f"failures={len(report['failures'])}",
        flush=True,
    )
    for failure in report["failures"]:
        print(
            f"[input-audit] ERROR {failure['message']}: "
            f"{json.dumps({k: v for k, v in failure.items() if k != 'message'}, ensure_ascii=False)}",
            flush=True,
        )
    for warning in report["warnings"]:
        print(
            f"[input-audit] WARNING {warning['message']}: "
            f"{json.dumps({k: v for k, v in warning.items() if k != 'message'}, ensure_ascii=False)}",
            flush=True,
        )
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
