#!/usr/bin/env python
"""Strictly audit the Daphnet Persistence-h4s TCN receptive-field ablation.

The auditor is intentionally independent from training.  It verifies the
canonical eight LOSO folds, immutable Persistence residual-h4s support, the
three six-block TCN variants, DONE artifact hashes, validation-only threshold
selection, per-fold metrics, common support, and root summaries.

``--allow-partial`` is intended for smoke tests and interrupted runs.  Every
completed cell is audited with the same checks, but ``SUITE_COMPLETE.json`` is
created only when all canonical 8 x 3 cells pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import binary_metrics, choose_threshold
from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
    validate_done,
)
from run_cnbr_fog_loso import event_metrics


AUDIT_VERSION = "daphnet_tcn_rf_ablation_audit.v1"
SUITE_VERSION = "daphnet_persistence_h4_tcn_rf_ablation.v1"
SOURCE_SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
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
EXPECTED_VARIANTS: dict[str, dict[str, Any]] = {
    "local": {
        "label": "TCN-S",
        "dilations": (1, 1, 1, 1, 1, 2),
        "receptive_field_samples": 29,
    },
    "medium": {
        "label": "TCN-M",
        "dilations": (1, 2, 4, 8, 8, 8),
        "receptive_field_samples": 125,
    },
    "long": {
        "label": "TCN-L",
        "dilations": (1, 2, 4, 8, 16, 32),
        "receptive_field_samples": 253,
    },
}
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
)
AGGREGATE_METRICS = (
    *SUMMARY_METRICS,
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
)
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
COUNT_METRICS = ("n", "n_normal", "n_fog", "tn", "fp", "fn", "tp")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Daphnet Persistence-h4s TCN RF ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        help=(
            "Fallback location of the completed 5x4 NBM suite. The path saved "
            "in config.json is preferred when it still exists."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Fallback Daphnet processed-data directory required when the path "
            "saved in config.json is not reachable; event metrics are always "
            "recomputed from the processed data."
        ),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--tolerance", type=float, default=2e-6)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def resolved_equal(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def close_enough(
    actual: Any,
    expected: Any,
    tolerance: float,
) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def assert_close(
    actual: Any,
    expected: Any,
    label: str,
    tolerance: float,
) -> None:
    require(
        close_enough(actual, expected, tolerance),
        f"{label}: {actual!r} != {expected!r}",
    )


def first_present(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def configured_path(config: dict[str, Any], kind: str) -> Path | None:
    if kind == "source":
        direct = first_present(
            config,
            ("source_suite_dir", "source_result_dir", "nbm_suite_dir"),
        )
        nested = config.get("source_suite")
        if direct is None and isinstance(nested, dict):
            direct = first_present(nested, ("path", "root", "result_dir"))
    elif kind == "data":
        direct = config.get("data_dir")
        nested = config.get("source_suite")
        if direct is None and isinstance(nested, dict):
            direct = nested.get("data_dir")
    else:
        raise ValueError(kind)
    if direct in (None, ""):
        return None
    return Path(str(direct)).expanduser()


def select_existing_path(
    configured: Path | None,
    fallback: Path | None,
    label: str,
    *,
    required: bool,
) -> Path | None:
    # Config provenance is authoritative when it is reachable.  A CLI path is
    # only a portability fallback for result directories copied from a server.
    if configured is not None and configured.exists():
        return configured.resolve()
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    if required:
        tried = [str(path) for path in (configured, fallback) if path is not None]
        raise FileNotFoundError(f"{label} is unavailable; tried {tried}")
    return None


def protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Use the runner's exact fingerprint projection when available."""

    try:
        from run_daphnet_tcn_rf_ablation import protocol_payload as runner_payload

        return runner_payload(config)
    except (ImportError, AttributeError):
        runtime = {
            "protocol_fingerprint",
            "data_dir",
            "source_suite_dir",
            "output_dir",
            "device",
            "resume",
            "num_workers",
        }
        return {key: value for key, value in config.items() if key not in runtime}


def normalise_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw: Any = first_present(
        config,
        ("variants", "tcn_variants", "rf_variants", "dilation_variants"),
    )
    if raw is None and isinstance(config.get("tcn"), dict):
        raw = config["tcn"].get("variants")
    if isinstance(raw, dict):
        raw = [
            {"name": name, **(value if isinstance(value, dict) else {})}
            for name, value in raw.items()
        ]
    require(isinstance(raw, list) and raw, "config: missing TCN RF variants")
    result: list[dict[str, Any]] = []
    for item in raw:
        require(isinstance(item, dict), "config: every variant must be an object")
        name = str(first_present(item, ("name", "variant", "id")) or "").lower()
        dilations = first_present(item, ("dilations", "dilation"))
        rf = first_present(
            item,
            (
                "receptive_field_samples",
                "receptive_field",
                "rf_samples",
            ),
        )
        result.append(
            {
                **item,
                "name": name,
                "dilations": tuple(int(value) for value in dilations or ()),
                "receptive_field_samples": int(rf) if rf is not None else -1,
            }
        )
    return result


def validate_protocol(
    config: dict[str, Any],
    allow_partial: bool,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    require(
        config.get("suite_version") == SUITE_VERSION,
        f"config: suite_version must be {SUITE_VERSION}",
    )
    require(int(config.get("sampling_rate_hz", -1)) == 64, "config: expected 64 Hz")
    require(int(config.get("n_channels", -1)) == 9, "config: expected 9 channels")
    require(
        tuple(config.get("channel_names", ())) == EXPECTED_CHANNELS,
        "config: unexpected channel names/order",
    )
    require(
        set(config.get("excluded_subjects", ())) == EXPECTED_EXCLUDED,
        "config: exclusions must be exactly S04/S10",
    )
    require(
        tuple(config.get("subjects", ())) == EXPECTED_SUBJECTS,
        "config: post-exclusion subjects are not canonical",
    )
    folds = [str(value) for value in config.get("folds_resolved", ())]
    require(folds and len(folds) == len(set(folds)), "config: invalid fold list")
    require(set(folds).issubset(EXPECTED_SUBJECTS), "config: unknown fold")

    source_nbm = str(
        first_present(config, ("source_nbm", "nbm", "normal_model"))
        or "persistence"
    ).lower()
    input_name = str(
        first_present(config, ("input", "input_name", "history_input"))
        or "residual_h4s"
    ).lower()
    require(source_nbm == "persistence", "config: source NBM is not Persistence")
    require(input_name == "residual_h4s", "config: input is not residual_h4s")
    require(
        int(first_present(config, ("history_samples",)) or -1) == 256,
        "config: residual history is not 256 samples",
    )
    require(
        int(first_present(config, ("history_blocks",)) or -1) == 8,
        "config: residual history is not eight 0.5-second blocks",
    )
    assert_close(
        first_present(config, ("history_seconds",)),
        4.0,
        "config/history_seconds",
        1e-12,
    )
    require(
        int(first_present(config, ("horizon_samples", "block_samples")) or -1)
        == 32,
        "config: source residual block is not 32 samples",
    )

    variants = normalise_variants(config)
    require(
        [item["name"] for item in variants] == list(EXPECTED_VARIANTS),
        "config: variants/order must be local, medium, long",
    )
    for item in variants:
        expected = EXPECTED_VARIANTS[item["name"]]
        require(
            item["dilations"] == expected["dilations"],
            f"config/{item['name']}: wrong dilation schedule",
        )
        require(
            item["receptive_field_samples"]
            == expected["receptive_field_samples"],
            f"config/{item['name']}: wrong receptive field",
        )
        label = first_present(item, ("display_name", "label"))
        if label is not None:
            require(
                str(label) == expected["label"],
                f"config/{item['name']}: wrong TCN label",
            )
        require(int(item.get("n_blocks", -1)) == 6, f"config/{item['name']}: blocks")
        require(
            int(item.get("convolutions_per_block", -1)) == 2,
            f"config/{item['name']}: convolutions per block",
        )
        require(
            int(item.get("kernel_size", -1)) == 3,
            f"config/{item['name']}: kernel size",
        )
        require(
            int(item.get("parameter_count", -1))
            == int(config.get("shared_parameter_count", -2)),
            f"config/{item['name']}: parameter count is not shared",
        )
    reference_initial_hashes = {
        str(item.get("reference_initial_state_sha256", ""))
        for item in variants
    }
    require(
        len(reference_initial_hashes) == 1 and "" not in reference_initial_hashes,
        "config: variants do not share one reference initial state",
    )
    fairness = config.get("fairness_contract")
    require(isinstance(fairness, dict), "config: missing fairness contract")
    require(
        fairness.get("ablation_axis") == "dilations",
        "config: ablation axis is not dilation",
    )
    require(
        fairness.get("same_classifier_seed_within_fold") is True
        and fairness.get("same_initial_state_sha256_within_fold") is True,
        "config: shared seed/initial-state contract is disabled",
    )

    expected_fingerprint = canonical_fingerprint(protocol_payload(config))
    require(
        expected_fingerprint == config.get("protocol_fingerprint"),
        "config: protocol fingerprint mismatch",
    )
    full_protocol = tuple(folds) == EXPECTED_SUBJECTS
    if not allow_partial:
        require(full_protocol, "Full audit requires the canonical eight folds")
    return folds, variants, full_protocol


def source_declared_fingerprint(config: dict[str, Any]) -> str | None:
    direct = first_present(
        config,
        (
            "source_protocol_fingerprint",
            "source_suite_protocol_fingerprint",
        ),
    )
    for nested_name in ("source", "source_suite"):
        nested = config.get(nested_name)
        if direct is None and isinstance(nested, dict):
            direct = first_present(
                nested,
                ("source_protocol_fingerprint", "protocol_fingerprint"),
            )
    return None if direct is None else str(direct)


def source_declared_data_hash(config: dict[str, Any]) -> str | None:
    direct = first_present(config, ("data_sha256", "source_data_sha256"))
    for nested_name in ("source", "source_suite"):
        nested = config.get(nested_name)
        if direct is None and isinstance(nested, dict):
            direct = first_present(
                nested,
                ("source_data_sha256", "data_sha256"),
            )
    return None if direct is None else str(direct)


def validate_source_suite(
    source_root: Path,
    config: dict[str, Any],
    folds: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    source_config = load_json(source_root / "config.json")
    require(
        source_config.get("suite_version") == SOURCE_SUITE_VERSION,
        "source suite has an unexpected suite_version",
    )
    source_protocol = str(source_config.get("protocol_fingerprint", ""))
    declared_source = config.get("source")
    require(
        isinstance(declared_source, dict),
        "config: missing immutable source manifest",
    )
    declared_protocol = source_declared_fingerprint(config)
    require(
        declared_protocol == source_protocol,
        "source suite protocol differs from ablation config",
    )
    require(
        declared_source.get("source_suite_version") == SOURCE_SUITE_VERSION,
        "config: source suite version mismatch",
    )
    require(
        declared_source.get("source_run_manifest_sha256")
        == sha256_file(source_root / "run_manifest.json"),
        "source run_manifest hash differs from ablation config",
    )
    require(
        declared_source.get("source_data_sha256")
        == source_config.get("data_sha256")
        == config.get("data_sha256"),
        "source data hash differs from ablation config",
    )
    require(
        "persistence" in source_config.get("nbms_resolved", ()),
        "source suite has no Persistence NBM",
    )
    source_histories = {
        str(item.get("input")): item
        for item in source_config.get("history_variants", ())
        if isinstance(item, dict)
    }
    require("residual_h4s" in source_histories, "source suite has no h4s support")
    require(
        int(source_histories["residual_h4s"].get("history_samples", -1)) == 256,
        "source suite h4s is not 256 samples",
    )
    require(
        tuple(source_config.get("subjects", ())) == EXPECTED_SUBJECTS,
        "source suite subjects differ from the ablation",
    )

    support: dict[str, dict[str, np.ndarray]] = {}
    residual_hashes: dict[str, str] = {}
    declared_folds = declared_source.get("folds")
    require(
        isinstance(declared_folds, dict)
        and set(declared_folds) == set(EXPECTED_SUBJECTS),
        "config: source fold manifest is incomplete",
    )
    for subject in folds:
        persistence_root = source_root / f"loso_{subject}" / "persistence"
        nbm_done_path = persistence_root / "nbm" / "DONE.json"
        nbm_done = validate_done(
            nbm_done_path,
            stage="nbm",
            protocol_fingerprint=source_protocol,
            task_id=f"loso_{subject}/persistence/nbm",
        )
        require(nbm_done is not None, f"source/{subject}: missing NBM DONE")
        nbm_best = nbm_done.get("artifacts", {}).get("best")
        require(nbm_best is not None, f"source/{subject}: NBM DONE has no best")
        nbm_best_sha256 = str(nbm_best["sha256"])
        residual_done_path = persistence_root / "RESIDUAL_CACHE_DONE.json"
        residual_done = validate_done(
            residual_done_path,
            stage="residual_cache",
            protocol_fingerprint=source_protocol,
            task_id=f"loso_{subject}/persistence/residual_cache",
            upstream_sha256=nbm_best_sha256,
        )
        require(residual_done is not None, f"source/{subject}: missing residual DONE")
        require(
            "cache" in residual_done.get("artifacts", {}),
            f"source/{subject}: residual DONE has no cache",
        )
        cache_entry = residual_done["artifacts"]["cache"]
        residual_hashes[subject] = str(cache_entry["sha256"])
        declared_fold = declared_folds[subject]
        require(
            declared_fold.get("source_nbm_best_sha256") == nbm_best_sha256,
            f"source/{subject}: NBM best hash differs from config",
        )
        require(
            declared_fold.get("source_residual_cache_sha256")
            == residual_hashes[subject],
            f"source/{subject}: residual cache hash differs from config",
        )
        require(
            int(declared_fold.get("source_residual_cache_bytes", -1))
            == int(cache_entry["bytes"]),
            f"source/{subject}: residual cache size differs from config",
        )
        require(
            declared_fold.get("source_residual_done_sha256")
            == sha256_file(residual_done_path),
            f"source/{subject}: residual DONE hash differs from config",
        )
        require(
            declared_fold.get("source_fold_config_sha256")
            == sha256_file(source_root / f"loso_{subject}" / "fold_config.json"),
            f"source/{subject}: fold config hash differs from config",
        )
        history_support_path = (
            source_root / f"loso_{subject}" / "history_support.npz"
        )
        require(
            declared_fold.get("source_history_support_sha256")
            == sha256_file(history_support_path),
            f"source/{subject}: history support hash differs from config",
        )
        require(
            int(declared_fold.get("source_history_support_bytes", -1))
            == int(history_support_path.stat().st_size),
            f"source/{subject}: history support size differs from config",
        )

        source_classifier = persistence_root / "residual_h4s"
        classifier_done = validate_done(
            source_classifier / "DONE.json",
            stage="classifier",
            protocol_fingerprint=source_protocol,
            task_id=f"{subject}/persistence/residual_h4s",
        )
        require(
            classifier_done is not None,
            f"source/{subject}: missing Persistence-h4s classifier DONE",
        )
        split_support: dict[str, np.ndarray] = {}
        for split, filename in (
            ("test", "predictions.npz"),
            ("validation", "validation_predictions.npz"),
        ):
            source_predictions = load_predictions(
                source_classifier / filename,
                f"source/{subject}/{split}",
            )
            split_support[f"{split}_window_index"] = source_predictions[
                "window_index"
            ]
            split_support[f"{split}_y_true"] = source_predictions["y_true"]
        support[subject] = split_support
    return support, residual_hashes


def load_dataset_and_windows(
    data_root: Path,
    source_root: Path,
    config: dict[str, Any],
) -> tuple[DaphnetDataset, WindowTable]:
    source_config = load_json(source_root / "config.json")
    declared_data_hash = source_declared_data_hash(config)
    require(declared_data_hash is not None, "config: missing data_sha256")
    require(
        dataset_fingerprint(data_root) == declared_data_hash,
        "processed Daphnet data fingerprint mismatch",
    )
    source = DaphnetDataset.load(
        data_root,
        flatline_seconds=float(source_config["flatline_seconds"]),
        zero_tolerance=float(source_config["zero_tolerance"]),
    )
    require(
        tuple(source.channel_names) == EXPECTED_CHANNELS,
        "processed data channel order mismatch",
    )
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
    require(
        tuple(dataset.subjects) == EXPECTED_SUBJECTS,
        "processed data post-exclusion subjects mismatch",
    )
    windows = dataset.make_windows(
        warmup_samples=int(source_config["context_samples"]),
        target_samples=int(source_config["horizon_samples"]),
        stride_samples=int(source_config["stride_samples"]),
        fog_fraction_threshold=float(source_config["fog_fraction_threshold"]),
        normal_guard_samples=int(source_config["normal_guard_samples"]),
    )
    require(
        len(windows) == int(source_config["window_count"]),
        "reconstructed WindowTable count mismatch",
    )
    require(
        np.array_equal(
            np.bincount(windows.label, minlength=2),
            np.asarray(source_config["window_class_counts"], dtype=np.int64),
        ),
        "reconstructed WindowTable labels mismatch",
    )
    return dataset, windows


def load_predictions(path: Path, label: str) -> dict[str, np.ndarray]:
    require(path.exists(), f"{label}: missing {path.name}")
    with np.load(path, allow_pickle=False) as payload:
        expected = {"window_index", "y_true", "y_prob", "y_pred"}
        require(set(payload.files) == expected, f"{label}: unexpected NPZ keys")
        arrays = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    require(
        len({array.shape for array in arrays.values()}) == 1,
        f"{label}: prediction shapes differ",
    )
    require(arrays["window_index"].ndim == 1, f"{label}: arrays must be 1D")
    require(len(arrays["window_index"]) > 0, f"{label}: empty predictions")
    require(
        len(np.unique(arrays["window_index"])) == len(arrays["window_index"]),
        f"{label}: duplicate window_index",
    )
    require(np.isin(arrays["y_true"], (0, 1)).all(), f"{label}: invalid y_true")
    require(np.isin(arrays["y_pred"], (0, 1)).all(), f"{label}: invalid y_pred")
    require(np.isfinite(arrays["y_prob"]).all(), f"{label}: non-finite y_prob")
    require(
        np.all((arrays["y_prob"] >= 0.0) & (arrays["y_prob"] <= 1.0)),
        f"{label}: y_prob outside [0,1]",
    )
    return arrays


def requested_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
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


def assert_binary_metrics(
    saved: dict[str, Any],
    recomputed: dict[str, Any],
    label: str,
    tolerance: float,
    *,
    requested: bool,
) -> None:
    for key in CORE_BINARY_METRICS:
        require(key in saved, f"{label}: missing metric {key}")
        assert_close(saved[key], recomputed[key], f"{label}/{key}", tolerance)
    for key in COUNT_METRICS:
        require(
            int(saved[key]) == int(recomputed[key]),
            f"{label}/{key}: {saved[key]} != {recomputed[key]}",
        )
    require(
        saved.get("confusion_matrix") == recomputed.get("confusion_matrix"),
        f"{label}: confusion_matrix mismatch",
    )
    if requested:
        for key, value in requested_metrics(recomputed).items():
            require(key in saved, f"{label}: missing requested metric {key}")
            assert_close(saved[key], value, f"{label}/{key}", tolerance)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    require(isinstance(payload, dict), f"{path}: checkpoint is not an object")
    return payload


def checkpoint_model_config(
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    for candidate in (
        checkpoint.get("classifier_config"),
        checkpoint.get("model_config"),
        metrics.get("model_config"),
        metrics.get("classifier_config"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    raise AssertionError("checkpoint/metrics have no model config")


def get_dilations(
    model_config: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[int, ...]:
    raw = first_present(model_config, ("dilations", "dilation"))
    if raw is None:
        raw = first_present(metrics, ("dilations", "dilation"))
    require(raw is not None, "missing dilation schedule in saved model provenance")
    return tuple(int(value) for value in raw)


def model_invariants(model_config: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "dilations",
        "dilation",
        "receptive_field",
        "receptive_field_samples",
        "receptive_field_seconds",
        "rf_samples",
        "variant",
        "name",
        "label",
    }
    return {key: value for key, value in model_config.items() if key not in ignored}


def state_structure(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    state = checkpoint.get("model_state")
    require(isinstance(state, dict) and state, "checkpoint has no model_state")
    return [
        {
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in state.items()
    ]


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def recompute_initial_state_sha256(
    config: dict[str, Any],
    classifier_seed: int,
) -> str:
    """Independently reconstruct the fold's pre-training parameter state."""

    cpu_rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(classifier_seed))
        model = ResidualTCNClassifier(
            in_channels=int(config["n_channels"]),
            hidden_channels=int(config["classifier_hidden"]),
            dilations=EXPECTED_VARIANTS["local"]["dilations"],
            kernel_size=3,
            dropout=float(config["classifier_dropout"]),
        ).cpu()
        return state_dict_sha256(model.state_dict())
    finally:
        torch.set_rng_state(cpu_rng_state)


def construct_model(
    model_config: dict[str, Any],
    dilations: tuple[int, ...],
) -> ResidualTCNClassifier:
    in_channels = int(
        first_present(model_config, ("in_channels", "input_channels")) or 9
    )
    hidden_channels = int(
        first_present(model_config, ("hidden_channels", "hidden")) or 48
    )
    kernel_size = int(model_config.get("kernel_size", 3))
    dropout = float(model_config.get("dropout", 0.15))
    return ResidualTCNClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dilations=dilations,
        kernel_size=kernel_size,
        dropout=dropout,
    ).cpu()


def done_artifact_path(
    done_path: Path,
    done: dict[str, Any],
    names: Iterable[str],
) -> Path:
    artifacts = done.get("artifacts")
    require(isinstance(artifacts, dict), f"{done_path}: missing artifact map")
    for name in names:
        if name in artifacts:
            result = Path(str(artifacts[name]["path"]))
            if not result.is_absolute():
                result = done_path.parent / result
            return result.resolve()
    raise AssertionError(f"{done_path}: missing artifact aliases {tuple(names)}")


def metric_variant(metrics: dict[str, Any]) -> str:
    value = first_present(metrics, ("variant", "rf_variant", "tcn_variant"))
    if value is not None:
        return str(value).lower()
    experiment_id = str(metrics.get("experiment_id", "")).lower()
    for name in EXPECTED_VARIANTS:
        if experiment_id == name or experiment_id.endswith(f"__{name}"):
            return name
    return ""


def validate_fold_files(
    result_root: Path,
    config: dict[str, Any],
    subject: str,
    source_support: dict[str, np.ndarray],
    source_residual_sha256: str,
) -> dict[str, Any]:
    fold_root = result_root / f"loso_{subject}"
    fold_config = load_json(fold_root / "fold_config.json")
    provenance = load_json(fold_root / "source_provenance.json")
    require(
        fold_config.get("suite_version") == SUITE_VERSION,
        f"{subject}: fold suite mismatch",
    )
    require(
        fold_config.get("protocol_fingerprint") == config["protocol_fingerprint"],
        f"{subject}: fold protocol mismatch",
    )
    require(fold_config.get("test_subject") == subject, f"{subject}: wrong test fold")
    require(
        fold_config.get("input") == "residual_h4s"
        and int(fold_config.get("history_samples", -1)) == 256
        and int(fold_config.get("history_blocks", -1)) == 8,
        f"{subject}: fold input is not Persistence residual_h4s",
    )
    expected_seed = int(config["seed"]) + 10000 + EXPECTED_SUBJECTS.index(subject)
    require(
        int(fold_config.get("classifier_seed", -1)) == expected_seed,
        f"{subject}: classifier seed does not follow the protocol",
    )
    require(
        fold_config.get("source") == provenance,
        f"{subject}: fold/source provenance files differ",
    )
    require(
        provenance.get("source_residual_cache_sha256")
        == source_residual_sha256,
        f"{subject}: fold uses another residual cache",
    )
    configured_source_fold = config["source"]["folds"][subject]
    for key in (
        "source_nbm_best_sha256",
        "source_residual_cache_sha256",
        "source_residual_cache_bytes",
        "source_residual_done_sha256",
        "source_fold_config_sha256",
        "source_history_support_sha256",
        "source_history_support_bytes",
    ):
        require(
            provenance.get(key) == configured_source_fold.get(key),
            f"{subject}: provenance {key} differs from root protocol",
        )

    support_path = fold_root / "input_support.npz"
    require(support_path.exists(), f"{subject}: missing input_support.npz")
    support_sha256 = sha256_file(support_path)
    require(
        provenance.get("input_support_sha256") == support_sha256,
        f"{subject}: input support hash mismatch",
    )
    expected_keys = {
        f"{split}_{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("anchor_window_index", "history_window_index", "y")
    }
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    with np.load(support_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == expected_keys,
            f"{subject}: unexpected input-support keys",
        )
        for split in ("train", "validation", "test"):
            anchors = np.asarray(
                payload[f"{split}_anchor_window_index"], dtype=np.int64
            )
            history = np.asarray(
                payload[f"{split}_history_window_index"], dtype=np.int64
            )
            truth = np.asarray(payload[f"{split}_y"], dtype=np.int8)
            require(
                anchors.ndim == truth.ndim == 1
                and history.ndim == 2
                and history.shape == (len(anchors), 8)
                and len(truth) == len(anchors),
                f"{subject}/{split}: invalid 8-block support shape",
            )
            require(
                len(anchors) > 0 and len(np.unique(anchors)) == len(anchors),
                f"{subject}/{split}: empty or duplicate anchors",
            )
            require(
                np.array_equal(history[:, -1], anchors),
                f"{subject}/{split}: final history block is not the anchor",
            )
            require(
                np.all(np.diff(history, axis=1) > 0),
                f"{subject}/{split}: history is not chronological",
            )
            require(
                np.isin(truth, (0, 1)).all(),
                f"{subject}/{split}: non-binary labels",
            )
            require(
                int(fold_config["history_anchor_counts"][split]) == len(anchors),
                f"{subject}/{split}: stale anchor count",
            )
            split_arrays[split] = {
                "window_index": anchors,
                "y_true": truth,
                "history_window_index": history,
            }
    for split in ("validation", "test"):
        require(
            np.array_equal(
                split_arrays[split]["window_index"],
                source_support[f"{split}_window_index"],
            )
            and np.array_equal(
                split_arrays[split]["y_true"],
                source_support[f"{split}_y_true"],
            ),
            f"{subject}/{split}: input support differs from source h4s",
        )
    reference_initial = str(
        fold_config.get("reference_initial_state_sha256", "")
    )
    require(reference_initial, f"{subject}: missing reference initial-state hash")
    recomputed_initial = recompute_initial_state_sha256(
        config,
        expected_seed,
    )
    require(
        reference_initial == recomputed_initial,
        f"{subject}: reference initial-state hash cannot be reproduced from seed",
    )
    return {
        "root": fold_root,
        "config": fold_config,
        "provenance": provenance,
        "support_sha256": support_sha256,
        "reference_initial_state_sha256": reference_initial,
        "classifier_seed": expected_seed,
        "val_subject": str(fold_config["val_subject"]),
        "support": split_arrays,
    }


def validate_cell(
    result_root: Path,
    config: dict[str, Any],
    subject: str,
    variant: dict[str, Any],
    fold_info: dict[str, Any],
    source_residual_sha256: str,
    dataset: DaphnetDataset,
    windows: WindowTable,
    tolerance: float,
) -> dict[str, Any]:
    name = variant["name"]
    task_id = f"{subject}/{name}"
    cell_root = result_root / f"loso_{subject}" / name
    done_path = cell_root / "DONE.json"
    raw_done = load_json(done_path)
    stage = str(raw_done.get("stage", ""))
    expected_stage = str(config.get("classifier_stage", "rf_classifier"))
    require(
        stage == expected_stage,
        f"{task_id}: DONE stage {stage!r} != {expected_stage!r}",
    )
    done = validate_done(
        done_path,
        stage=expected_stage,
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(done is not None, f"{task_id}: missing DONE")
    require(
        done.get("source_residual_sha256") == source_residual_sha256,
        f"{task_id}: DONE source residual hash mismatch",
    )
    require(
        done.get("input_support_sha256") == fold_info["support_sha256"],
        f"{task_id}: DONE input support hash mismatch",
    )
    require(
        done.get("initial_state_sha256")
        == fold_info["reference_initial_state_sha256"],
        f"{task_id}: DONE initial-state hash mismatch",
    )
    require(
        set(done.get("artifacts", {}))
        == {
            "best",
            "last",
            "metrics",
            "predictions",
            "validation_predictions",
            "predictions_csv",
        },
        f"{task_id}: unexpected DONE artifact map",
    )

    metrics_path = done_artifact_path(done_path, done, ("metrics",))
    test_path = done_artifact_path(done_path, done, ("predictions", "test_predictions"))
    validation_path = done_artifact_path(
        done_path,
        done,
        ("validation_predictions", "val_predictions"),
    )
    best_path = done_artifact_path(
        done_path,
        done,
        ("best", "classifier_best", "best_checkpoint"),
    )
    # A resumable final checkpoint is part of a server-ready experiment.
    last_path = done_artifact_path(
        done_path,
        done,
        ("last", "classifier_last", "last_checkpoint"),
    )

    metrics = load_json(metrics_path)
    require(metric_variant(metrics) == name, f"{task_id}: metrics variant mismatch")
    require(
        metrics.get("experiment_id") == variant["experiment_id"],
        f"{task_id}: experiment id mismatch",
    )
    require(
        metrics.get("display_name") == variant["display_name"],
        f"{task_id}: display name mismatch",
    )
    require(
        str(metrics.get("test_subject")) == subject,
        f"{task_id}: metrics test subject mismatch",
    )
    require(
        str(metrics.get("val_subject")) == fold_info["val_subject"],
        f"{task_id}: metrics validation subject mismatch",
    )
    require(
        str(first_present(metrics, ("input", "input_name")) or "residual_h4s")
        == "residual_h4s",
        f"{task_id}: metrics input is not residual_h4s",
    )
    require(
        str(first_present(metrics, ("nbm", "source_nbm")) or "persistence").lower()
        == "persistence",
        f"{task_id}: metrics NBM is not Persistence",
    )
    require(
        int(metrics.get("history_samples", -1)) == 256
        and int(metrics.get("history_blocks", -1)) == 8,
        f"{task_id}: metrics history shape mismatch",
    )
    assert_close(
        metrics.get("history_seconds"),
        4.0,
        f"{task_id}/history_seconds",
        1e-12,
    )
    saved_source_hash = first_present(
        metrics,
        (
            "source_residual_sha256",
            "source_residual_cache_sha256",
            "residual_cache_sha256",
            "upstream_residual_sha256",
            "upstream_nbm_sha256",
        ),
    )
    require(
        str(saved_source_hash) == source_residual_sha256,
        f"{task_id}: metrics source residual hash mismatch",
    )
    require(
        metrics.get("input_support_sha256") == fold_info["support_sha256"],
        f"{task_id}: metrics input support hash mismatch",
    )

    test = load_predictions(test_path, f"{task_id}/test")
    validation = load_predictions(validation_path, f"{task_id}/validation")
    require(
        np.array_equal(
            test["window_index"],
            fold_info["support"]["test"]["window_index"],
        ),
        f"{task_id}: test support differs from source Persistence-h4s",
    )
    require(
        np.array_equal(test["y_true"], fold_info["support"]["test"]["y_true"]),
        f"{task_id}: test truth differs from source Persistence-h4s",
    )
    require(
        np.array_equal(
            validation["window_index"],
            fold_info["support"]["validation"]["window_index"],
        ),
        f"{task_id}: validation support differs from source Persistence-h4s",
    )
    require(
        np.array_equal(
            validation["y_true"],
            fold_info["support"]["validation"]["y_true"],
        ),
        f"{task_id}: validation truth differs from source Persistence-h4s",
    )

    threshold = float(metrics["threshold"])
    require(0.0 <= threshold <= 1.0, f"{task_id}: invalid threshold")
    require(
        np.array_equal(
            test["y_pred"],
            (test["y_prob"] >= threshold).astype(np.int8),
        ),
        f"{task_id}: test y_pred/threshold mismatch",
    )
    require(
        np.array_equal(
            validation["y_pred"],
            (validation["y_prob"] >= threshold).astype(np.int8),
        ),
        f"{task_id}: validation y_pred/threshold mismatch",
    )
    selected_threshold, selected_validation = choose_threshold(
        validation["y_true"],
        validation["y_prob"],
    )
    assert_close(
        threshold,
        selected_threshold,
        f"{task_id}/selected_threshold",
        1e-12,
    )
    require(
        isinstance(metrics.get("validation"), dict),
        f"{task_id}: missing validation metrics",
    )
    assert_binary_metrics(
        metrics["validation"],
        selected_validation,
        f"{task_id}/validation",
        tolerance,
        requested=False,
    )
    recomputed_test = binary_metrics(test["y_true"], test["y_prob"], threshold)
    assert_binary_metrics(
        metrics,
        recomputed_test,
        f"{task_id}/test",
        tolerance,
        requested=True,
    )
    require(
        np.array_equal(
            test["y_true"],
            windows.label[test["window_index"]],
        )
        and np.array_equal(
            validation["y_true"],
            windows.label[validation["window_index"]],
        ),
        f"{task_id}: saved truth differs from reconstructed WindowTable",
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
            assert_close(
                metrics[key],
                recomputed_events[key],
                f"{task_id}/{key}",
                tolerance,
            )

    best = torch_load(best_path)
    require(
        best.get("protocol_fingerprint") == config["protocol_fingerprint"],
        f"{task_id}: checkpoint protocol mismatch",
    )
    require(best.get("task_id") == task_id, f"{task_id}: checkpoint task mismatch")
    require(best.get("stage") == expected_stage, f"{task_id}: checkpoint stage mismatch")
    require(
        best.get("source_residual_sha256") == source_residual_sha256,
        f"{task_id}: checkpoint source residual hash mismatch",
    )
    require(best.get("variant") == name, f"{task_id}: checkpoint variant mismatch")
    model_config = checkpoint_model_config(best, metrics)
    require(
        metrics.get("classifier_config") == model_config,
        f"{task_id}: checkpoint and metrics model config differ",
    )
    dilations = get_dilations(model_config, metrics)
    expected = EXPECTED_VARIANTS[name]
    require(
        dilations == expected["dilations"],
        f"{task_id}: checkpoint dilation schedule mismatch",
    )
    require(len(dilations) == 6, f"{task_id}: TCN does not have six blocks")
    kernel_size = int(model_config.get("kernel_size", 3))
    receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    require(
        receptive_field == expected["receptive_field_samples"],
        f"{task_id}: computed receptive field is {receptive_field}",
    )
    require(
        int(model_config.get("n_blocks", -1)) == 6
        and int(model_config.get("convolutions_per_block", -1)) == 2
        and model_config.get("global_pooling") == "mean_and_max_over_full_input",
        f"{task_id}: classifier structure differs from the pure ablation",
    )
    saved_rf = first_present(
        model_config,
        ("receptive_field_samples", "receptive_field", "rf_samples"),
    )
    require(saved_rf is not None, f"{task_id}: metrics has no receptive field")
    require(
        int(saved_rf) == receptive_field,
        f"{task_id}: saved receptive field mismatch",
    )
    model = construct_model(model_config, dilations)
    model.load_state_dict(best["model_state"], strict=True)
    model_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    saved_parameters = first_present(
        model_config,
        ("model_parameters", "parameter_count", "n_parameters"),
    )
    require(saved_parameters is not None, f"{task_id}: missing parameter count")
    require(
        int(saved_parameters) == model_parameters,
        f"{task_id}: saved parameter count mismatch",
    )
    require(
        model_parameters
        == int(config["shared_parameter_count"])
        == int(variant["parameter_count"]),
        f"{task_id}: parameter count differs from root protocol",
    )
    with torch.no_grad():
        logits = model.eval()(torch.zeros(2, 9, 256, dtype=torch.float32))
    require(tuple(logits.shape) == (2,), f"{task_id}: invalid model output shape")
    require(torch.isfinite(logits).all().item(), f"{task_id}: non-finite model output")

    classifier_seed = first_present(metrics, ("classifier_seed", "model_seed", "seed"))
    require(classifier_seed is not None, f"{task_id}: missing classifier seed")
    require(
        int(classifier_seed) == fold_info["classifier_seed"],
        f"{task_id}: classifier seed differs from fold protocol",
    )
    checkpoint_seed = first_present(
        best,
        ("classifier_seed", "model_seed", "seed"),
    )
    if checkpoint_seed is not None:
        require(
            int(checkpoint_seed) == int(classifier_seed),
            f"{task_id}: checkpoint/metrics seed mismatch",
        )
    require(
        int(best.get("best_epoch", -1)) == int(metrics.get("best_epoch", -2)),
        f"{task_id}: best epoch mismatch",
    )
    assert_close(
        best.get("best_validation_auprc"),
        metrics.get("best_validation_auprc"),
        f"{task_id}/best_validation_auprc",
        tolerance,
    )
    last = torch_load(last_path)
    for field, expected_value in (
        ("stage", expected_stage),
        ("protocol_fingerprint", config["protocol_fingerprint"]),
        ("task_id", task_id),
        ("source_residual_sha256", source_residual_sha256),
        ("variant", name),
        ("classifier_seed", int(classifier_seed)),
    ):
        require(
            last.get(field) == expected_value,
            f"{task_id}: last checkpoint {field} mismatch",
        )
    require(
        last.get("classifier_config") == model_config,
        f"{task_id}: last checkpoint model config mismatch",
    )
    initialization_hash = first_present(
        best,
        ("initialization_sha256", "initial_state_sha256"),
    )
    if initialization_hash is None:
        initialization_hash = first_present(
            metrics,
            ("initialization_sha256", "initial_state_sha256"),
        )
    require(
        str(initialization_hash)
        == fold_info["reference_initial_state_sha256"],
        f"{task_id}: initial state differs from the shared reference",
    )
    return {
        "metrics": metrics,
        "test_predictions": test,
        "validation_predictions": validation,
        "model_invariants": model_invariants(model_config),
        "state_structure": state_structure(best),
        "model_parameters": model_parameters,
        "classifier_seed": int(classifier_seed),
        "initialization_sha256": initialization_hash,
        "dilations": dilations,
        "receptive_field_samples": receptive_field,
    }


def validate_pure_ablation(
    subject: str,
    evidence_by_variant: dict[str, dict[str, Any]],
    *,
    require_initialization_hash: bool,
) -> None:
    if not evidence_by_variant:
        return
    reference_name = next(iter(evidence_by_variant))
    reference = evidence_by_variant[reference_name]
    for name, evidence in evidence_by_variant.items():
        require(
            np.array_equal(
                evidence["test_predictions"]["window_index"],
                reference["test_predictions"]["window_index"],
            )
            and np.array_equal(
                evidence["test_predictions"]["y_true"],
                reference["test_predictions"]["y_true"],
            ),
            f"{subject}: variants do not share test support/truth",
        )
        require(
            np.array_equal(
                evidence["validation_predictions"]["window_index"],
                reference["validation_predictions"]["window_index"],
            )
            and np.array_equal(
                evidence["validation_predictions"]["y_true"],
                reference["validation_predictions"]["y_true"],
            ),
            f"{subject}: variants do not share validation support/truth",
        )
        require(
            evidence["model_invariants"] == reference["model_invariants"],
            f"{subject}: {name} changes architecture beyond dilation",
        )
        require(
            evidence["state_structure"] == reference["state_structure"],
            f"{subject}: {name} parameter structure differs",
        )
        require(
            evidence["model_parameters"] == reference["model_parameters"],
            f"{subject}: {name} parameter count differs",
        )
        require(
            evidence["classifier_seed"] == reference["classifier_seed"],
            f"{subject}: {name} classifier seed differs",
        )
        for invariant_metric in ("train_counts", "pos_weight"):
            if (
                invariant_metric in evidence["metrics"]
                or invariant_metric in reference["metrics"]
            ):
                require(
                    evidence["metrics"].get(invariant_metric)
                    == reference["metrics"].get(invariant_metric),
                    f"{subject}: {name} {invariant_metric} differs",
                )
    hashes = [
        evidence["initialization_sha256"]
        for evidence in evidence_by_variant.values()
    ]
    if require_initialization_hash:
        require(
            all(value not in (None, "") for value in hashes),
            f"{subject}: shared initialization hash is required",
        )
    nonempty_hashes = [str(value) for value in hashes if value not in (None, "")]
    if nonempty_hashes:
        require(
            len(nonempty_hashes) == len(evidence_by_variant)
            and len(set(nonempty_hashes)) == 1,
            f"{subject}: variants do not share one initialization",
        )


def subject_macro_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in AGGREGATE_METRICS:
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


def pooled_metrics(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.concatenate(
        [item["test_predictions"]["y_true"] for item in evidence]
    ).astype(np.int8, copy=False)
    probability = np.concatenate(
        [item["test_predictions"]["y_prob"] for item in evidence]
    ).astype(np.float64, copy=False)
    prediction = np.concatenate(
        [item["test_predictions"]["y_pred"] for item in evidence]
    ).astype(np.int8, copy=False)
    tn = int(((truth == 0) & (prediction == 0)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    tp = int(((truth == 1) & (prediction == 1)).sum())
    recall_fog = tp / (tp + fn) if tp + fn else 0.0
    recall_nonfog = tn / (tn + fp) if tn + fp else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": int(len(truth)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(truth), 1),
        "balanced_accuracy": 0.5 * (recall_fog + recall_nonfog),
        "macro_f1": 0.5 * (f1_fog + f1_nonfog),
        "roc_auc": (
            float(roc_auc_score(truth, probability))
            if np.unique(truth).size == 2
            else None
        ),
        "pr_auc": (
            float(average_precision_score(truth, probability))
            if np.unique(truth).size == 2
            else None
        ),
        "fog_recall": recall_fog,
        "fog_f1": f1_fog,
        "specificity": recall_nonfog,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.exists(), f"missing root summary {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path.name}: missing header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def row_variant(row: dict[str, str]) -> str:
    for field in ("variant", "rf_variant", "tcn_variant"):
        if row.get(field):
            return row[field].lower()
    experiment_id = row.get("experiment_id", "").lower()
    for name in EXPECTED_VARIANTS:
        if experiment_id == name or experiment_id.endswith(f"__{name}"):
            return name
    return ""


def aggregate_group(
    aggregate: dict[str, Any],
    name: str,
) -> tuple[str, dict[str, Any]]:
    matches = []
    for key, value in aggregate.items():
        if not isinstance(value, dict):
            continue
        declared = str(
            first_present(value, ("variant", "rf_variant", "tcn_variant")) or ""
        ).lower()
        if declared == name or key.lower() == name or key.lower().endswith(f"__{name}"):
            matches.append((key, value))
    require(len(matches) == 1, f"aggregate: expected one group for {name}")
    return matches[0]


def validate_summaries(
    root: Path,
    config: dict[str, Any],
    folds: list[str],
    cells: dict[tuple[str, str], dict[str, Any]],
    tolerance: float,
) -> None:
    expected_cells = len(folds) * len(EXPECTED_VARIANTS)
    completed_cells = len(cells)
    status = load_json(root / "status.json")
    require(
        status.get("suite_version") == SUITE_VERSION,
        "status.json: suite mismatch",
    )
    require(
        status.get("protocol_fingerprint") == config["protocol_fingerprint"],
        "status.json: protocol mismatch",
    )
    for aliases, expected in (
        (("expected_fold_cells", "expected_cells"), expected_cells),
        (("completed_fold_cells", "completed_cells"), completed_cells),
    ):
        saved = first_present(status, aliases)
        require(saved is not None, f"status.json: missing {aliases[0]}")
        require(int(saved) == expected, f"status.json: stale {aliases[0]}")
    expected_status = "complete" if completed_cells == expected_cells else "partial"
    require(status.get("status") == expected_status, "status.json: stale status")

    _, fold_rows = read_csv(root / "fold_summary.csv")
    require(
        len(fold_rows) == completed_cells,
        "fold_summary.csv: stale completed-cell count",
    )
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in fold_rows:
        subject = str(first_present(row, ("test_subject", "subject")) or "")
        name = row_variant(row)
        key = (subject, name)
        require(key not in by_key, f"fold_summary.csv: duplicate {key}")
        by_key[key] = row
    require(
        set(by_key) == set(cells),
        "fold_summary.csv: cell identities differ from audited DONE cells",
    )
    for key, evidence in cells.items():
        row = by_key[key]
        metrics = evidence["metrics"]
        for field in (
            "threshold",
            "n",
            "n_normal",
            "n_fog",
            *SUMMARY_METRICS,
            "tn",
            "fp",
            "fn",
            "tp",
        ):
            require(field in row, f"fold_summary.csv: missing {field}")
            actual = None if row[field].strip() == "" else float(row[field])
            assert_close(
                actual,
                metrics.get(field),
                f"fold_summary/{key}/{field}",
                tolerance,
            )

    _, manifest_rows = read_csv(root / "experiment_manifest.csv")
    require(
        len(manifest_rows) == len(EXPECTED_VARIANTS),
        "experiment_manifest.csv: expected three variants",
    )
    manifest_by_variant = {row_variant(row): row for row in manifest_rows}
    require(
        set(manifest_by_variant) == set(EXPECTED_VARIANTS),
        "experiment_manifest.csv: wrong variant groups",
    )
    for name, row in manifest_by_variant.items():
        definition = next(
            item for item in config["variants"] if item["variant"] == name
        )
        require(
            row.get("experiment_id") == definition["experiment_id"]
            and row.get("display_name") == definition["display_name"],
            f"manifest/{name}: identity mismatch",
        )
        require(
            row.get("dilations") == ",".join(map(str, definition["dilations"])),
            f"manifest/{name}: dilation schedule mismatch",
        )
        require(
            int(row.get("receptive_field_samples", -1))
            == int(definition["receptive_field_samples"])
            and int(row.get("parameter_count", -1))
            == int(config["shared_parameter_count"]),
            f"manifest/{name}: RF/parameter count mismatch",
        )
        completed_subjects = [
            subject for subject in folds if (subject, name) in cells
        ]
        expected_folds_value = first_present(
            row,
            ("expected_folds", "n_expected_folds"),
        )
        completed_folds_value = first_present(
            row,
            ("completed_folds", "n_completed_folds"),
        )
        require(expected_folds_value is not None, f"manifest/{name}: expected_folds")
        require(
            completed_folds_value is not None,
            f"manifest/{name}: completed_folds",
        )
        require(
            int(expected_folds_value) == len(folds),
            f"manifest/{name}: stale expected fold count",
        )
        require(
            int(completed_folds_value) == len(completed_subjects),
            f"manifest/{name}: stale completed fold count",
        )
        group_status = (
            "complete"
            if completed_subjects == folds
            else ("partial" if completed_subjects else "pending")
        )
        require(row.get("status") == group_status, f"manifest/{name}: stale status")
        if "completed_subjects" in row:
            require(
                row["completed_subjects"] == ",".join(completed_subjects),
                f"manifest/{name}: stale completed subjects",
            )

    aggregate = load_json(root / "aggregate_metrics.json")
    used_keys: set[str] = set()
    for name in EXPECTED_VARIANTS:
        evidence = [
            cells[(subject, name)]
            for subject in folds
            if (subject, name) in cells
        ]
        if not evidence:
            continue
        aggregate_key, group = aggregate_group(aggregate, name)
        used_keys.add(aggregate_key)
        definition = next(
            item for item in config["variants"] if item["variant"] == name
        )
        require(
            aggregate_key == definition["experiment_id"],
            f"aggregate/{name}: experiment id mismatch",
        )
        require(
            group.get("display_name") == definition["display_name"]
            and group.get("dilations") == definition["dilations"]
            and int(group.get("receptive_field_samples", -1))
            == int(definition["receptive_field_samples"])
            and int(group.get("parameter_count", -1))
            == int(config["shared_parameter_count"]),
            f"aggregate/{name}: architecture metadata mismatch",
        )
        completed_subjects = [
            subject for subject in folds if (subject, name) in cells
        ]
        require(
            group.get("completed_folds") == completed_subjects,
            f"aggregate/{name}: stale completed folds",
        )
        expected_macro = subject_macro_metrics(
            [item["metrics"] for item in evidence]
        )
        saved_macro = group.get("subject_macro")
        require(isinstance(saved_macro, dict), f"aggregate/{name}: missing macro")
        for metric_name, expected_values in expected_macro.items():
            require(
                metric_name in saved_macro,
                f"aggregate/{name}: missing macro {metric_name}",
            )
            saved_values = saved_macro[metric_name]
            require(
                int(saved_values["n_folds"]) == expected_values["n_folds"],
                f"aggregate/{name}/{metric_name}: fold count",
            )
            for statistic in ("mean", "std"):
                assert_close(
                    saved_values.get(statistic),
                    expected_values[statistic],
                    f"aggregate/{name}/{metric_name}/{statistic}",
                    tolerance,
                )
            if expected_values["n_folds"]:
                for statistic in ("min", "max"):
                    assert_close(
                        saved_values.get(statistic),
                        expected_values[statistic],
                        f"aggregate/{name}/{metric_name}/{statistic}",
                        tolerance,
                    )
        expected_pooled = pooled_metrics(evidence)
        saved_pooled = group.get("pooled")
        require(isinstance(saved_pooled, dict), f"aggregate/{name}: missing pooled")
        for metric_name, expected_value in expected_pooled.items():
            require(
                metric_name in saved_pooled,
                f"aggregate/{name}: missing pooled {metric_name}",
            )
            assert_close(
                saved_pooled[metric_name],
                expected_value,
                f"aggregate/{name}/pooled/{metric_name}",
                tolerance,
            )
    require("paired_deltas" in aggregate, "aggregate: missing paired deltas")
    paired = aggregate["paired_deltas"]
    require(
        isinstance(paired, dict) and set(paired) == {"medium", "long"},
        "aggregate: paired deltas must compare medium/long with local",
    )
    for name in ("medium", "long"):
        local_by_subject = {
            subject: cells[(subject, "local")]
            for subject in folds
            if (subject, "local") in cells
        }
        comparison_by_subject = {
            subject: cells[(subject, name)]
            for subject in folds
            if (subject, name) in cells
        }
        common = [
            subject
            for subject in folds
            if subject in local_by_subject and subject in comparison_by_subject
        ]
        saved = paired[name]
        require(
            saved.get("reference") == "local"
            and saved.get("common_subjects") == common,
            f"aggregate/paired/{name}: support mismatch",
        )
        require(
            set(saved.get("metrics", {})) == set(AGGREGATE_METRICS),
            f"aggregate/paired/{name}: metric keys mismatch",
        )
        for metric_name in AGGREGATE_METRICS:
            deltas = [
                float(comparison_by_subject[subject]["metrics"][metric_name])
                - float(local_by_subject[subject]["metrics"][metric_name])
                for subject in common
                if comparison_by_subject[subject]["metrics"].get(metric_name)
                is not None
                and local_by_subject[subject]["metrics"].get(metric_name) is not None
            ]
            values = np.asarray(deltas, dtype=np.float64)
            saved_metric = saved["metrics"][metric_name]
            require(
                int(saved_metric["n_paired_folds"]) == len(values),
                f"aggregate/paired/{name}/{metric_name}: fold count",
            )
            assert_close(
                saved_metric["mean_delta_vs_local"],
                float(values.mean()) if len(values) else None,
                f"aggregate/paired/{name}/{metric_name}/mean",
                tolerance,
            )
            assert_close(
                saved_metric["std_delta_vs_local"],
                float(values.std(ddof=0)) if len(values) else None,
                f"aggregate/paired/{name}/{metric_name}/std",
                tolerance,
            )
    require(
        set(aggregate) == used_keys | {"paired_deltas"},
        "aggregate_metrics.json: stale or unexpected groups",
    )

    _, aggregate_rows = read_csv(root / "aggregate_summary.csv")
    expected_aggregate_variants = {
        name for name in EXPECTED_VARIANTS if any(key[1] == name for key in cells)
    }
    aggregate_rows_by_variant = {
        row_variant(row): row for row in aggregate_rows
    }
    require(
        set(aggregate_rows_by_variant) == expected_aggregate_variants,
        "aggregate_summary.csv: stale variant rows",
    )
    for name, row in aggregate_rows_by_variant.items():
        _, group = aggregate_group(aggregate, name)
        require(
            int(row["completed_folds"]) == len(group["completed_folds"]),
            f"aggregate_summary/{name}: fold count mismatch",
        )
        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc",
            "fog_recall",
            "fog_f1",
        ):
            for statistic in ("mean", "std"):
                assert_close(
                    float(row[f"{metric_name}_{statistic}"]),
                    group["subject_macro"][metric_name][statistic],
                    f"aggregate_summary/{name}/{metric_name}_{statistic}",
                    tolerance,
                )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.result_dir.resolve()
    require(root.is_dir(), f"result directory does not exist: {root}")
    config = load_json(root / "config.json")
    folds, variants, full_protocol = validate_protocol(config, args.allow_partial)
    source_root = select_existing_path(
        configured_path(config, "source"),
        args.source_suite_dir,
        "source suite directory",
        required=True,
    )
    assert source_root is not None
    source_support, source_residual_hashes = validate_source_suite(
        source_root,
        config,
        folds,
    )

    warnings: list[str] = []
    data_root = select_existing_path(
        configured_path(config, "data"),
        args.data_dir,
        "Daphnet data directory",
        required=True,
    )
    assert data_root is not None
    dataset, windows = load_dataset_and_windows(
        data_root,
        source_root,
        config,
    )

    run_manifest = load_json(root / "run_manifest.json")
    run_manifest_runtime = {
        "data_dir",
        "source_suite_dir",
        "output_dir",
        "device",
        "num_workers",
        "resume",
    }
    require(
        run_manifest
        == {
            key: value
            for key, value in config.items()
            if key not in run_manifest_runtime
        },
        "run_manifest.json differs from the immutable protocol",
    )
    load_json(root / "environment.json")

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    cell_failures: list[str] = []
    require_initialization_hash = True
    for subject in folds:
        subject_done_paths = [
            root / f"loso_{subject}" / name / "DONE.json"
            for name in EXPECTED_VARIANTS
        ]
        fold_config_path = root / f"loso_{subject}" / "fold_config.json"
        if not fold_config_path.exists() and not any(
            path.exists() for path in subject_done_paths
        ):
            missing.extend(
                f"{subject}/{name}" for name in EXPECTED_VARIANTS
            )
            continue
        fold_info = validate_fold_files(
            root,
            config,
            subject,
            source_support[subject],
            source_residual_hashes[subject],
        )
        per_fold: dict[str, dict[str, Any]] = {}
        for variant in variants:
            name = variant["name"]
            done_path = root / f"loso_{subject}" / name / "DONE.json"
            if not done_path.exists():
                missing.append(f"{subject}/{name}")
                continue
            try:
                evidence = validate_cell(
                    root,
                    config,
                    subject,
                    variant,
                    fold_info,
                    source_residual_hashes[subject],
                    dataset,
                    windows,
                    args.tolerance,
                )
                cells[(subject, name)] = evidence
                per_fold[name] = evidence
            except Exception as error:
                cell_failures.append(f"{subject}/{name}: {error}")
        try:
            validate_pure_ablation(
                subject,
                per_fold,
                require_initialization_hash=require_initialization_hash,
            )
        except Exception as error:
            cell_failures.append(f"{subject}/pure_ablation: {error}")

    expected_cells = len(folds) * len(variants)
    if missing and not args.allow_partial:
        cell_failures.extend(f"missing {task_id}" for task_id in missing)
    try:
        validate_summaries(
            root,
            config,
            folds,
            cells,
            args.tolerance,
        )
    except Exception as error:
        cell_failures.append(f"root summaries: {error}")
    full_complete = (
        full_protocol
        and len(cells) == len(EXPECTED_SUBJECTS) * len(EXPECTED_VARIANTS)
        and not missing
        and not cell_failures
    )
    return {
        "audit_version": AUDIT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "source_suite_dir": str(source_root),
        "data_dir": None if data_root is None else str(data_root),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_cells": expected_cells,
        "checked_cells": len(cells),
        "missing_cells": missing,
        "allow_partial": bool(args.allow_partial),
        "full_protocol": bool(full_protocol),
        "full_complete": bool(full_complete),
        "failures": cell_failures,
        "warnings": warnings,
        "status": "pass" if not cell_failures else "fail",
    }


def main() -> None:
    args = parse_args()
    root = args.result_dir.resolve()
    report: dict[str, Any]
    try:
        report = audit(args)
    except Exception as error:
        report = {
            "audit_version": AUDIT_VERSION,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_dir": str(root),
            "allow_partial": bool(args.allow_partial),
            "full_complete": False,
            "failures": [f"fatal: {type(error).__name__}: {error}"],
            "warnings": [],
            "status": "fail",
        }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "AUDIT_REPORT.json"
    atomic_json_dump(report, report_path)
    complete_path = root / "SUITE_COMPLETE.json"
    if report.get("status") == "pass" and report.get("full_complete"):
        atomic_json_dump(
            {
                "format_version": 1,
                "suite_version": SUITE_VERSION,
                "audit_version": AUDIT_VERSION,
                "status": "complete",
                "protocol_fingerprint": report["protocol_fingerprint"],
                "expected_cells": int(report["expected_cells"]),
                "checked_cells": int(report["checked_cells"]),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_report_sha256": sha256_file(report_path),
            },
            complete_path,
        )
    elif complete_path.exists():
        complete_path.unlink()

    print(
        f"[rf-audit] status={report['status']} "
        f"checked={report.get('checked_cells', 0)}/"
        f"{report.get('expected_cells', 24)} "
        f"missing={len(report.get('missing_cells', []))}",
        flush=True,
    )
    for failure in report.get("failures", []):
        print(f"[rf-audit] ERROR {failure}", flush=True)
    for warning in report.get("warnings", []):
        print(f"[rf-audit] WARNING {warning}", flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
