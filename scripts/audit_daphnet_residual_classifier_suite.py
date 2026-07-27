#!/usr/bin/env python
"""Strict independent audit for the Persistence residual-h4s classifier suite.

The training runner is deliberately not imported.  This script reconstructs
the protocol, source provenance, model initialization, validation threshold,
test/event metrics, and root summaries from saved artifacts.  Partial audits
are useful for smoke tests, but ``SUITE_COMPLETE.json`` is written only after
all canonical 8 folds x 4 classifiers pass.
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
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import binary_metrics, choose_threshold
from cnbr_fog.residual_classifiers import (
    CANONICAL_CLASSIFIER_NAMES,
    CLASSIFIER_DISPLAY_NAMES,
    build_residual_classifier,
    parameter_count,
)
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
    validate_done,
)
from run_cnbr_fog_loso import event_metrics


AUDIT_VERSION = "daphnet_residual_classifier_suite_audit.v1"
SUITE_VERSION = "daphnet_persistence_h4_residual_classifier_suite.v1"
SOURCE_SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
CLASSIFIER_STAGE = "residual_classifier"
EXPECTED_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
)
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
EXPECTED_FAMILIES = {
    "mlp": "multilayer_perceptron",
    "cnn1d": "multi_scale_1d_cnn",
    "gru": "gru",
    "transformer": "transformer_encoder",
}
ARCHITECTURE_FIELDS = {
    "mlp": ("hidden_features",),
    "cnn1d": (
        "branch_channels",
        "hidden_channels",
        "head_features",
        "kernel_sizes",
    ),
    "gru": ("hidden_size", "num_layers", "head_features"),
    "transformer": (
        "model_dim",
        "num_heads",
        "num_layers",
        "feedforward_dim",
        "head_features",
    ),
}
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
)
CLASSIFICATION_METRICS = (
    *SUMMARY_METRICS,
    "specificity",
    "precision",
    "mcc",
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
DONE_ARTIFACTS = {
    "best",
    "last",
    "metrics",
    "predictions",
    "validation_predictions",
    "predictions_csv",
}
RUNTIME_CONFIG_FIELDS = {
    "protocol_fingerprint",
    "data_dir",
    "source_suite_dir",
    "output_dir",
    "device",
    "num_workers",
    "resume",
}
RUN_MANIFEST_RUNTIME_FIELDS = RUNTIME_CONFIG_FIELDS - {"protocol_fingerprint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the four-classifier Daphnet Persistence residual-h4s suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        help=(
            "Fallback completed NBM-suite path when the path recorded in "
            "config.json is unavailable."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Fallback processed Daphnet path when the path recorded in "
            "config.json is unavailable."
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
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def first_present(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
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


def configured_path(config: dict[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def select_existing_path(
    configured: Path | None,
    fallback: Path | None,
    label: str,
) -> Path:
    if configured is not None and configured.exists():
        return configured.resolve()
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    tried = [str(path) for path in (configured, fallback) if path is not None]
    raise FileNotFoundError(f"{label} is unavailable; tried {tried}")


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def architecture_kwargs(name: str, architecture: Mapping[str, Any]) -> dict:
    kwargs: dict[str, Any] = {}
    for field in ARCHITECTURE_FIELDS[name]:
        require(field in architecture, f"{name}: architecture missing {field}")
        value = architecture[field]
        if field == "kernel_sizes":
            value = tuple(int(item) for item in value)
        kwargs[field] = value
    return kwargs


def construct_model(name: str, architecture: Mapping[str, Any]) -> torch.nn.Module:
    require(
        str(architecture.get("canonical_name")) == name,
        f"{name}: canonical_name mismatch",
    )
    return build_residual_classifier(
        name,
        in_channels=int(architecture["in_channels"]),
        input_samples=int(architecture["input_samples"]),
        dropout=float(architecture["dropout"]),
        **architecture_kwargs(name, architecture),
    ).cpu()


def recompute_initial_state(
    name: str,
    architecture: Mapping[str, Any],
    seed: int,
) -> tuple[str, int]:
    cpu_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        model = construct_model(name, architecture)
        return state_dict_sha256(model.state_dict()), parameter_count(model)
    finally:
        torch.set_rng_state(cpu_state)


def protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RUNTIME_CONFIG_FIELDS
    }


def validate_implementation(config: dict[str, Any]) -> None:
    implementation = config.get("implementation")
    require(isinstance(implementation, dict), "config: missing implementation")
    files = implementation.get("files")
    require(isinstance(files, dict) and files, "config: missing implementation files")
    require(
        implementation.get("sha256") == canonical_fingerprint(files),
        "config: implementation manifest fingerprint mismatch",
    )
    for relative, expected_hash in files.items():
        path = REPO_ROOT / str(relative)
        require(path.is_file(), f"implementation file unavailable: {relative}")
        require(
            sha256_file(path) == expected_hash,
            f"implementation file changed since protocol creation: {relative}",
        )


def validate_protocol(
    config: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    require(
        config.get("suite_version") == SUITE_VERSION,
        f"config: suite_version must be {SUITE_VERSION}",
    )
    require(int(config.get("sampling_rate_hz", -1)) == 64, "config: expected 64 Hz")
    require(int(config.get("n_channels", -1)) == 9, "config: expected 9 channels")
    require(
        tuple(config.get("channel_names", ())) == EXPECTED_CHANNELS,
        "config: unexpected channel order",
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
    require(
        tuple(folds) == EXPECTED_SUBJECTS,
        "config: this strict suite must contain the canonical eight folds",
    )
    require(config.get("nbm") == "persistence", "config: NBM is not Persistence")
    require(
        config.get("input") == "residual_h4s",
        "config: input is not residual_h4s",
    )
    require(
        int(config.get("history_samples", -1)) == 256
        and int(config.get("history_blocks", -1)) == 8,
        "config: residual history is not eight blocks / 256 samples",
    )
    assert_close(config.get("history_seconds"), 4.0, "config/history_seconds", 1e-12)
    require(
        int(config.get("horizon_samples", -1)) == 32,
        "config: source residual block is not 32 samples",
    )
    require(
        int(config.get("stride_samples", -1)) == 16,
        "config: expected 16-sample source stride",
    )
    names = [str(value) for value in config.get("classifier_names", ())]
    require(
        tuple(names) == CANONICAL_CLASSIFIER_NAMES,
        "config: classifier names/order must be mlp, cnn1d, gru, transformer",
    )
    definitions = config.get("classifiers")
    require(
        isinstance(definitions, list) and len(definitions) == 4,
        "config: expected exactly four classifier definitions",
    )
    require(
        [item.get("classifier") for item in definitions] == names,
        "config: classifier definition order mismatch",
    )
    for definition in definitions:
        name = str(definition["classifier"])
        architecture = definition.get("architecture")
        require(
            isinstance(architecture, dict),
            f"config/{name}: missing architecture",
        )
        require(
            architecture.get("family") == EXPECTED_FAMILIES[name],
            f"config/{name}: wrong family",
        )
        require(
            int(architecture.get("in_channels", -1)) == 9
            and int(architecture.get("input_samples", -1)) == 256,
            f"config/{name}: wrong input shape",
        )
        assert_close(
            architecture.get("dropout"),
            config.get("classifier_dropout"),
            f"config/{name}/dropout",
            1e-12,
        )
        expected_hash, expected_parameters = recompute_initial_state(
            name,
            architecture,
            int(config["seed"]),
        )
        require(
            int(definition.get("parameter_count", -1))
            == int(architecture.get("parameter_count", -2))
            == expected_parameters,
            f"config/{name}: parameter count mismatch",
        )
        require(
            definition.get("protocol_initial_state_sha256") == expected_hash,
            f"config/{name}: protocol initial hash cannot be reproduced",
        )
        require(
            definition.get("display_name") == CLASSIFIER_DISPLAY_NAMES[name],
            f"config/{name}: display name mismatch",
        )
        require(
            definition.get("experiment_id") == f"persistence_h4s__{name}",
            f"config/{name}: experiment id mismatch",
        )
        model = construct_model(name, architecture)
        with torch.no_grad():
            output = model.eval()(torch.zeros(2, 9, 256))
        require(tuple(output.shape) == (2,), f"config/{name}: invalid output shape")
        require(torch.isfinite(output).all().item(), f"config/{name}: non-finite output")

    fairness = config.get("fairness_contract")
    require(isinstance(fairness, dict), "config: missing fairness contract")
    require(
        fairness.get("ablation_axis") == "downstream_classifier_architecture",
        "config: wrong ablation axis",
    )
    require(
        fairness.get("same_classifier_seed_within_fold") is True
        and fairness.get("same_epoch_shuffle_within_fold") is True
        and fairness.get("different_parameter_shapes_expected") is True,
        "config: fairness contract is incomplete",
    )
    require(
        fairness.get("threshold_source")
        == "validation_only_balanced_accuracy",
        "config: threshold rule mismatch",
    )
    validate_implementation(config)
    require(
        canonical_fingerprint(protocol_payload(config))
        == config.get("protocol_fingerprint"),
        "config: protocol fingerprint mismatch",
    )
    return folds, definitions


def load_predictions(path: Path, label: str) -> dict[str, np.ndarray]:
    require(path.exists(), f"{label}: missing prediction file")
    with np.load(path, allow_pickle=False) as payload:
        expected = {"window_index", "y_true", "y_prob", "y_pred"}
        require(set(payload.files) == expected, f"{label}: unexpected NPZ keys")
        result = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    require(
        len({value.shape for value in result.values()}) == 1
        and result["window_index"].ndim == 1,
        f"{label}: arrays are not aligned one-dimensional vectors",
    )
    require(len(result["window_index"]) > 0, f"{label}: empty predictions")
    require(
        len(np.unique(result["window_index"])) == len(result["window_index"]),
        f"{label}: duplicate window indices",
    )
    require(np.isin(result["y_true"], (0, 1)).all(), f"{label}: invalid y_true")
    require(np.isin(result["y_pred"], (0, 1)).all(), f"{label}: invalid y_pred")
    require(np.isfinite(result["y_prob"]).all(), f"{label}: non-finite probability")
    require(
        np.all((result["y_prob"] >= 0.0) & (result["y_prob"] <= 1.0)),
        f"{label}: probability outside [0,1]",
    )
    return result


def validate_source_suite(
    source_root: Path,
    config: dict[str, Any],
    folds: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    source_config = load_json(source_root / "config.json")
    require(
        source_config.get("suite_version") == SOURCE_SUITE_VERSION,
        "source suite version mismatch",
    )
    source_protocol = str(source_config.get("protocol_fingerprint", ""))
    declared = config.get("source")
    require(isinstance(declared, dict), "config: missing immutable source manifest")
    require(
        declared.get("source_suite_version") == SOURCE_SUITE_VERSION,
        "config: declared source suite version mismatch",
    )
    require(
        declared.get("source_protocol_fingerprint") == source_protocol,
        "source protocol differs from classifier suite config",
    )
    require(
        declared.get("source_run_manifest_sha256")
        == sha256_file(source_root / "run_manifest.json"),
        "source run_manifest hash mismatch",
    )
    require(
        declared.get("source_data_sha256")
        == source_config.get("data_sha256")
        == config.get("data_sha256"),
        "source data hash mismatch",
    )
    require(
        tuple(source_config.get("subjects", ())) == EXPECTED_SUBJECTS,
        "source subjects mismatch",
    )
    require(
        "persistence" in source_config.get("nbms_resolved", ()),
        "source has no Persistence NBM",
    )
    histories = {
        item.get("input"): item
        for item in source_config.get("history_variants", ())
        if isinstance(item, dict)
    }
    require(
        "residual_h4s" in histories
        and int(histories["residual_h4s"].get("history_samples", -1)) == 256,
        "source has no canonical residual_h4s",
    )
    declared_folds = declared.get("folds")
    require(
        isinstance(declared_folds, dict)
        and set(declared_folds) == set(EXPECTED_SUBJECTS),
        "config: source fold manifest is incomplete",
    )

    support: dict[str, dict[str, np.ndarray]] = {}
    residual_hashes: dict[str, str] = {}
    for subject in folds:
        source_fold = declared_folds[subject]
        source_fold_config_path = (
            source_root / f"loso_{subject}" / "fold_config.json"
        )
        history_support_path = (
            source_root / f"loso_{subject}" / "history_support.npz"
        )
        persistence_root = source_root / f"loso_{subject}" / "persistence"
        nbm_done_path = persistence_root / "nbm" / "DONE.json"
        nbm_done = validate_done(
            nbm_done_path,
            stage="nbm",
            protocol_fingerprint=source_protocol,
            task_id=f"loso_{subject}/persistence/nbm",
        )
        require(nbm_done is not None, f"source/{subject}: missing NBM DONE")
        best_entry = nbm_done.get("artifacts", {}).get("best")
        require(best_entry is not None, f"source/{subject}: NBM DONE has no best")
        best_sha = str(best_entry["sha256"])
        residual_done_path = persistence_root / "RESIDUAL_CACHE_DONE.json"
        residual_done = validate_done(
            residual_done_path,
            stage="residual_cache",
            protocol_fingerprint=source_protocol,
            task_id=f"loso_{subject}/persistence/residual_cache",
            upstream_sha256=best_sha,
        )
        require(
            residual_done is not None,
            f"source/{subject}: missing residual-cache DONE",
        )
        cache_entry = residual_done.get("artifacts", {}).get("cache")
        require(cache_entry is not None, f"source/{subject}: DONE has no cache")
        cache_path = Path(str(cache_entry["path"]))
        if not cache_path.is_absolute():
            cache_path = residual_done_path.parent / cache_path
        require(cache_path.exists(), f"source/{subject}: residual cache missing")
        residual_hashes[subject] = str(cache_entry["sha256"])
        expected_source = {
            "source_nbm_best_sha256": best_sha,
            "source_residual_cache_sha256": residual_hashes[subject],
            "source_residual_cache_bytes": int(cache_entry["bytes"]),
            "source_residual_done_sha256": sha256_file(residual_done_path),
            "source_fold_config_sha256": sha256_file(source_fold_config_path),
            "source_history_support_sha256": sha256_file(history_support_path),
            "source_history_support_bytes": int(history_support_path.stat().st_size),
        }
        for key, value in expected_source.items():
            require(
                source_fold.get(key) == value,
                f"source/{subject}: configured {key} mismatch",
            )

        with np.load(history_support_path, allow_pickle=False) as payload:
            expected_keys = {
                f"{split}_{suffix}"
                for split in ("train", "validation", "test")
                for suffix in ("anchor_window_index", "history_window_index")
            }
            require(
                set(payload.files) == expected_keys,
                f"source/{subject}: history-support keys mismatch",
            )
            for split in ("train", "validation", "test"):
                anchors = np.asarray(
                    payload[f"{split}_anchor_window_index"], dtype=np.int64
                )
                history = np.asarray(
                    payload[f"{split}_history_window_index"], dtype=np.int64
                )
                require(
                    anchors.ndim == 1
                    and history.shape == (len(anchors), 8)
                    and np.array_equal(history[:, -1], anchors),
                    f"source/{subject}/{split}: malformed h4s support",
                )

        source_classifier = persistence_root / "residual_h4s"
        source_classifier_done = validate_done(
            source_classifier / "DONE.json",
            stage="classifier",
            protocol_fingerprint=source_protocol,
            task_id=f"{subject}/persistence/residual_h4s",
        )
        require(
            source_classifier_done is not None,
            f"source/{subject}: Persistence-h4s classifier is incomplete",
        )
        subject_support: dict[str, np.ndarray] = {}
        for split, filename in (
            ("validation", "validation_predictions.npz"),
            ("test", "predictions.npz"),
        ):
            prediction = load_predictions(
                source_classifier / filename,
                f"source/{subject}/{split}",
            )
            subject_support[f"{split}_window_index"] = prediction["window_index"]
            subject_support[f"{split}_y_true"] = prediction["y_true"]
        support[subject] = subject_support
    return support, residual_hashes


def load_dataset_and_windows(
    data_root: Path,
    source_root: Path,
    config: dict[str, Any],
) -> tuple[DaphnetDataset, WindowTable]:
    source_config = load_json(source_root / "config.json")
    require(
        dataset_fingerprint(data_root) == config.get("data_sha256"),
        "processed data fingerprint mismatch",
    )
    raw = DaphnetDataset.load(
        data_root,
        flatline_seconds=float(source_config["flatline_seconds"]),
        zero_tolerance=float(source_config["zero_tolerance"]),
    )
    require(
        tuple(raw.channel_names) == EXPECTED_CHANNELS,
        "processed data channel order mismatch",
    )
    dataset = DaphnetDataset(
        root=raw.root,
        records=[
            record
            for record in raw.records
            if record.subject_id not in EXPECTED_EXCLUDED
        ],
        sampling_rate_hz=raw.sampling_rate_hz,
        channel_names=raw.channel_names,
    )
    require(
        tuple(dataset.subjects) == EXPECTED_SUBJECTS,
        "processed data subjects mismatch",
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


def validate_run_manifest(root: Path, config: dict[str, Any]) -> None:
    run_manifest = load_json(root / "run_manifest.json")
    expected = {
        key: value
        for key, value in config.items()
        if key not in RUN_MANIFEST_RUNTIME_FIELDS
    }
    require(run_manifest == expected, "run_manifest differs from config protocol")
    load_json(root / "environment.json")


def validate_fold_support(
    root: Path,
    source_root: Path,
    config: dict[str, Any],
    subject: str,
    source_support: dict[str, np.ndarray],
    source_residual_sha256: str,
    windows: WindowTable,
) -> dict[str, Any]:
    fold_root = root / f"loso_{subject}"
    fold_config = load_json(fold_root / "fold_config.json")
    provenance = load_json(fold_root / "source_provenance.json")
    require(
        fold_config.get("suite_version") == SUITE_VERSION,
        f"{subject}: fold suite version mismatch",
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
        f"{subject}: fold input mismatch",
    )
    expected_seed = int(config["seed"]) + 10000 + EXPECTED_SUBJECTS.index(subject)
    require(
        int(fold_config.get("classifier_seed", -1)) == expected_seed,
        f"{subject}: classifier seed rule mismatch",
    )
    require(
        fold_config.get("source") == provenance,
        f"{subject}: provenance files differ",
    )
    require(
        provenance.get("source_residual_cache_sha256")
        == source_residual_sha256,
        f"{subject}: wrong Persistence residual cache",
    )
    configured_source = config["source"]["folds"][subject]
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
            provenance.get(key) == configured_source.get(key),
            f"{subject}: provenance {key} mismatch",
        )
    source_classifier = (
        source_root
        / f"loso_{subject}"
        / "persistence"
        / "residual_h4s"
    )
    for key, source_path in (
        (
            "source_validation_predictions_sha256",
            source_classifier / "validation_predictions.npz",
        ),
        (
            "source_test_predictions_sha256",
            source_classifier / "predictions.npz",
        ),
    ):
        require(
            provenance.get(key) == sha256_file(source_path),
            f"{subject}: provenance {key} mismatch",
        )
    support_path = fold_root / "input_support.npz"
    require(support_path.exists(), f"{subject}: missing input_support.npz")
    support_sha = sha256_file(support_path)
    require(
        provenance.get("input_support_sha256") == support_sha,
        f"{subject}: input support hash mismatch",
    )
    expected_keys = {
        f"{split}_{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("anchor_window_index", "history_window_index", "y")
    }
    split_support: dict[str, dict[str, np.ndarray]] = {}
    with np.load(support_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == expected_keys,
            f"{subject}: input support keys mismatch",
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
                and history.shape == (len(anchors), 8)
                and len(truth) == len(anchors)
                and len(anchors) > 0,
                f"{subject}/{split}: malformed support",
            )
            require(
                len(np.unique(anchors)) == len(anchors),
                f"{subject}/{split}: duplicate anchors",
            )
            require(
                np.array_equal(history[:, -1], anchors)
                and np.all(np.diff(history, axis=1) > 0),
                f"{subject}/{split}: malformed chronological history",
            )
            require(
                np.array_equal(truth, windows.label[anchors]),
                f"{subject}/{split}: labels differ from WindowTable",
            )
            require(
                int(fold_config["history_anchor_counts"][split]) == len(anchors),
                f"{subject}/{split}: stale anchor count",
            )
            split_support[split] = {
                "window_index": anchors,
                "history_window_index": history,
                "y_true": truth,
            }
    for split in ("validation", "test"):
        require(
            np.array_equal(
                split_support[split]["window_index"],
                source_support[f"{split}_window_index"],
            )
            and np.array_equal(
                split_support[split]["y_true"],
                source_support[f"{split}_y_true"],
            ),
            f"{subject}/{split}: differs from source Persistence-h4s support",
        )
    return {
        "root": fold_root,
        "config": fold_config,
        "provenance": provenance,
        "support_sha256": support_sha,
        "classifier_seed": expected_seed,
        "val_subject": str(fold_config["val_subject"]),
        "support": split_support,
    }


def torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    require(isinstance(payload, dict), f"{path}: checkpoint is not an object")
    return payload


def done_artifact_path(
    done_path: Path,
    done: Mapping[str, Any],
    name: str,
) -> Path:
    artifacts = done.get("artifacts")
    require(isinstance(artifacts, dict), f"{done_path}: missing artifact map")
    require(name in artifacts, f"{done_path}: missing artifact {name}")
    path = Path(str(artifacts[name]["path"]))
    if not path.is_absolute():
        path = done_path.parent / path
    return path.resolve()


def requested_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = (int(metrics[key]) for key in ("tn", "fp", "fn", "tp"))
    fog_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    normal_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "macro_f1": 0.5 * (fog_f1 + normal_f1),
        "roc_auc": metrics["auroc"],
        "pr_auc": metrics["auprc"],
        "fog_recall": metrics["sensitivity"],
        "fog_f1": fog_f1,
    }


def assert_binary_metrics(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
    tolerance: float,
    *,
    include_requested: bool,
) -> None:
    for key in CORE_BINARY_METRICS:
        require(key in saved, f"{label}: missing {key}")
        assert_close(saved[key], expected[key], f"{label}/{key}", tolerance)
    for key in COUNT_METRICS:
        require(key in saved, f"{label}: missing {key}")
        require(
            int(saved[key]) == int(expected[key]),
            f"{label}/{key}: count mismatch",
        )
    require(
        saved.get("confusion_matrix") == expected.get("confusion_matrix"),
        f"{label}: confusion matrix mismatch",
    )
    if include_requested:
        for key, value in requested_metrics(expected).items():
            require(key in saved, f"{label}: missing {key}")
            assert_close(saved[key], value, f"{label}/{key}", tolerance)


def validate_checkpoint_base(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    task_id: str,
    name: str,
    source_residual_sha256: str,
    classifier_seed: int,
    architecture: Mapping[str, Any],
    initial_hash: str,
) -> None:
    expected = {
        "stage": CLASSIFIER_STAGE,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": task_id,
        "source_residual_sha256": source_residual_sha256,
        "classifier": name,
        "classifier_seed": classifier_seed,
        "architecture": architecture,
        "initial_state_sha256": initial_hash,
    }
    for key, value in expected.items():
        require(
            checkpoint.get(key) == value,
            f"{task_id}: checkpoint {key} mismatch",
        )


def validate_training_provenance(
    metrics: Mapping[str, Any],
    last: Mapping[str, Any],
    *,
    task_id: str,
    classifier_seed: int,
    train_truth: np.ndarray,
    config: Mapping[str, Any],
    tolerance: float,
) -> None:
    counts = np.bincount(train_truth, minlength=2).astype(np.int64)
    require(
        list(metrics.get("train_counts", ())) == counts.tolist(),
        f"{task_id}: training class counts mismatch",
    )
    expected_pos_weight = min(math.sqrt(counts[0] / counts[1]), 6.0)
    assert_close(
        metrics.get("pos_weight"),
        expected_pos_weight,
        f"{task_id}/pos_weight",
        tolerance,
    )
    history = metrics.get("history")
    require(isinstance(history, list) and history, f"{task_id}: empty history")
    require(
        last.get("history") == history,
        f"{task_id}: metrics/last training histories differ",
    )
    epochs = [int(row.get("epoch", -1)) for row in history]
    require(
        epochs == list(range(1, len(history) + 1)),
        f"{task_id}: non-contiguous epochs",
    )
    require(
        len(history) <= int(config["classifier_epochs"]),
        f"{task_id}: too many training epochs",
    )
    for row in history:
        epoch = int(row["epoch"])
        require(
            int(row.get("shuffle_seed", -1)) == classifier_seed + epoch,
            f"{task_id}: epoch {epoch} shuffle seed mismatch",
        )
        for key in (
            "train_loss",
            "train_auprc",
            "validation_loss",
            "validation_auprc",
        ):
            require(
                math.isfinite(float(row[key])),
                f"{task_id}: non-finite history {key}",
            )
    best_epoch = int(metrics.get("best_epoch", -1))
    require(1 <= best_epoch <= len(history), f"{task_id}: invalid best epoch")
    best_score = float(metrics["best_validation_auprc"])
    assert_close(
        best_score,
        history[best_epoch - 1]["validation_auprc"],
        f"{task_id}/best score history",
        tolerance,
    )
    require(
        all(
            best_score + tolerance >= float(row["validation_auprc"])
            for row in history
        ),
        f"{task_id}: saved best score is not maximal",
    )
    require(
        int(last.get("epoch", -1)) == len(history),
        f"{task_id}: last checkpoint epoch/history mismatch",
    )
    require(
        int(last.get("best_epoch", -1)) == best_epoch,
        f"{task_id}: last checkpoint best epoch mismatch",
    )


def validate_cell(
    root: Path,
    config: dict[str, Any],
    definition: dict[str, Any],
    subject: str,
    fold: dict[str, Any],
    source_residual_sha256: str,
    dataset: DaphnetDataset,
    windows: WindowTable,
    tolerance: float,
) -> dict[str, Any]:
    name = str(definition["classifier"])
    task_id = f"{subject}/{name}"
    cell_root = root / f"loso_{subject}" / name
    done_path = cell_root / "DONE.json"
    done = validate_done(
        done_path,
        stage=CLASSIFIER_STAGE,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    require(done is not None, f"{task_id}: missing DONE")
    require(
        set(done.get("artifacts", {})) == DONE_ARTIFACTS,
        f"{task_id}: DONE artifact map mismatch",
    )
    require(done.get("classifier") == name, f"{task_id}: DONE classifier mismatch")
    require(
        done.get("source_residual_sha256") == source_residual_sha256,
        f"{task_id}: DONE residual hash mismatch",
    )
    require(
        done.get("input_support_sha256") == fold["support_sha256"],
        f"{task_id}: DONE support hash mismatch",
    )
    metrics_path = done_artifact_path(done_path, done, "metrics")
    test_path = done_artifact_path(done_path, done, "predictions")
    validation_path = done_artifact_path(
        done_path, done, "validation_predictions"
    )
    best_path = done_artifact_path(done_path, done, "best")
    last_path = done_artifact_path(done_path, done, "last")

    metrics = load_json(metrics_path)
    architecture = definition["architecture"]
    expected_initial, expected_parameters = recompute_initial_state(
        name,
        architecture,
        fold["classifier_seed"],
    )
    require(
        done.get("initial_state_sha256") == expected_initial,
        f"{task_id}: DONE initial hash cannot be reproduced",
    )
    expected_metadata = {
        "experiment_id": definition["experiment_id"],
        "classifier": name,
        "display_name": definition["display_name"],
        "nbm": "persistence",
        "input": "residual_h4s",
        "history_samples": 256,
        "history_blocks": 8,
        "test_subject": subject,
        "val_subject": fold["val_subject"],
        "classifier_seed": fold["classifier_seed"],
        "architecture": architecture,
        "parameter_count": expected_parameters,
        "initial_state_sha256": expected_initial,
        "source_residual_sha256": source_residual_sha256,
        "input_support_sha256": fold["support_sha256"],
    }
    for key, value in expected_metadata.items():
        require(metrics.get(key) == value, f"{task_id}: metrics {key} mismatch")
    assert_close(
        metrics.get("history_seconds"),
        4.0,
        f"{task_id}/history_seconds",
        1e-12,
    )

    test = load_predictions(test_path, f"{task_id}/test")
    validation = load_predictions(validation_path, f"{task_id}/validation")
    for split, prediction in (("test", test), ("validation", validation)):
        require(
            np.array_equal(
                prediction["window_index"],
                fold["support"][split]["window_index"],
            )
            and np.array_equal(
                prediction["y_true"],
                fold["support"][split]["y_true"],
            ),
            f"{task_id}: {split} support/truth mismatch",
        )
    threshold = float(metrics["threshold"])
    require(0.0 <= threshold <= 1.0, f"{task_id}: invalid threshold")
    require(
        np.array_equal(
            test["y_pred"], (test["y_prob"] >= threshold).astype(np.int8)
        )
        and np.array_equal(
            validation["y_pred"],
            (validation["y_prob"] >= threshold).astype(np.int8),
        ),
        f"{task_id}: threshold/prediction mismatch",
    )
    expected_threshold, expected_validation = choose_threshold(
        validation["y_true"], validation["y_prob"]
    )
    assert_close(
        threshold,
        expected_threshold,
        f"{task_id}/validation-selected threshold",
        1e-12,
    )
    require(
        isinstance(metrics.get("validation"), dict),
        f"{task_id}: missing validation metrics",
    )
    assert_binary_metrics(
        metrics["validation"],
        expected_validation,
        f"{task_id}/validation",
        tolerance,
        include_requested=False,
    )
    expected_test = binary_metrics(test["y_true"], test["y_prob"], threshold)
    assert_binary_metrics(
        metrics,
        expected_test,
        f"{task_id}/test",
        tolerance,
        include_requested=True,
    )
    require(
        np.array_equal(test["y_true"], windows.label[test["window_index"]]),
        f"{task_id}: test truth differs from WindowTable",
    )
    events = event_metrics(
        dataset, windows, test["window_index"], test["y_pred"]
    )
    for key in EVENT_METRICS:
        require(key in metrics, f"{task_id}: missing event metric {key}")
        if isinstance(events[key], (int, np.integer)):
            require(
                int(metrics[key]) == int(events[key]),
                f"{task_id}/{key}: event count mismatch",
            )
        else:
            assert_close(
                metrics[key], events[key], f"{task_id}/{key}", tolerance
            )

    best = torch_load(best_path)
    last = torch_load(last_path)
    validate_checkpoint_base(
        best,
        config=config,
        task_id=task_id,
        name=name,
        source_residual_sha256=source_residual_sha256,
        classifier_seed=fold["classifier_seed"],
        architecture=architecture,
        initial_hash=expected_initial,
    )
    validate_checkpoint_base(
        last,
        config=config,
        task_id=task_id,
        name=name,
        source_residual_sha256=source_residual_sha256,
        classifier_seed=fold["classifier_seed"],
        architecture=architecture,
        initial_hash=expected_initial,
    )
    model = construct_model(name, architecture)
    require(
        parameter_count(model)
        == expected_parameters
        == int(definition["parameter_count"]),
        f"{task_id}: reconstructed parameter count mismatch",
    )
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        output = model.eval()(torch.zeros(2, 9, 256))
    require(tuple(output.shape) == (2,), f"{task_id}: invalid checkpoint output")
    require(torch.isfinite(output).all().item(), f"{task_id}: non-finite output")
    require(
        int(best.get("best_epoch", -1)) == int(metrics["best_epoch"]),
        f"{task_id}: best epoch mismatch",
    )
    assert_close(
        best.get("best_validation_auprc"),
        metrics.get("best_validation_auprc"),
        f"{task_id}/best checkpoint score",
        tolerance,
    )
    validate_training_provenance(
        metrics,
        last,
        task_id=task_id,
        classifier_seed=fold["classifier_seed"],
        train_truth=fold["support"]["train"]["y_true"],
        config=config,
        tolerance=tolerance,
    )
    return {
        "metrics": metrics,
        "test": test,
        "validation": validation,
        "initial_state_sha256": expected_initial,
        "parameter_count": expected_parameters,
        "classifier_seed": fold["classifier_seed"],
    }


def validate_cross_classifier_fairness(
    subject: str,
    evidence: Mapping[str, dict[str, Any]],
) -> None:
    if not evidence:
        return
    reference = next(iter(evidence.values()))
    for name, item in evidence.items():
        for split in ("test", "validation"):
            require(
                np.array_equal(
                    item[split]["window_index"],
                    reference[split]["window_index"],
                )
                and np.array_equal(
                    item[split]["y_true"],
                    reference[split]["y_true"],
                ),
                f"{subject}: {name} does not share {split} support/truth",
            )
        require(
            item["classifier_seed"] == reference["classifier_seed"],
            f"{subject}: classifiers do not share one fold seed",
        )
        require(
            item["metrics"]["train_counts"]
            == reference["metrics"]["train_counts"],
            f"{subject}: classifiers do not share training counts",
        )
        assert_close(
            item["metrics"]["pos_weight"],
            reference["metrics"]["pos_weight"],
            f"{subject}/{name}/shared pos_weight",
            1e-12,
        )
        history = item["metrics"]["history"]
        reference_history = reference["metrics"]["history"]
        common_epochs = min(len(history), len(reference_history))
        require(
            [row["shuffle_seed"] for row in history[:common_epochs]]
            == [
                row["shuffle_seed"]
                for row in reference_history[:common_epochs]
            ],
            f"{subject}: classifiers do not share epoch shuffle seeds",
        )
    # Different architectures are intentionally not required to have equal
    # initial hashes; each hash was independently reconstructed above.


def aggregate_fold_metrics(
    rows: list[dict[str, Any]],
    keys: Iterable[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
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
            "n_folds": int(len(array)),
        }
    return result


def pooled_metrics(
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    y_true = np.concatenate([item["test"]["y_true"] for item in evidence])
    y_prob = np.concatenate([item["test"]["y_prob"] for item in evidence])
    y_pred = np.concatenate([item["test"]["y_pred"] for item in evidence])
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fog_recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fog_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    normal_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (fog_recall + specificity),
        "macro_f1": 0.5 * (fog_f1 + normal_f1),
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
        "fog_recall": fog_recall,
        "fog_f1": fog_f1,
        "specificity": specificity,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.exists(), f"missing root summary {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def csv_number(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def assert_json_metric_map(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
    tolerance: float,
) -> None:
    require(set(saved) == set(expected), f"{label}: metric keys mismatch")
    for key, expected_value in expected.items():
        saved_value = saved[key]
        if isinstance(expected_value, Mapping):
            require(isinstance(saved_value, Mapping), f"{label}/{key}: not object")
            assert_json_metric_map(
                saved_value, expected_value, f"{label}/{key}", tolerance
            )
        elif isinstance(expected_value, (int, np.integer)) and not isinstance(
            expected_value, bool
        ):
            require(int(saved_value) == int(expected_value), f"{label}/{key}")
        elif isinstance(expected_value, (float, np.floating)) or expected_value is None:
            assert_close(saved_value, expected_value, f"{label}/{key}", tolerance)
        else:
            require(saved_value == expected_value, f"{label}/{key}: mismatch")


def paired_deltas(
    cells: Mapping[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in CANONICAL_CLASSIFIER_NAMES:
        if name == "mlp":
            continue
        subjects = [
            subject
            for subject in EXPECTED_SUBJECTS
            if (subject, "mlp") in cells and (subject, name) in cells
        ]
        metric_map: dict[str, Any] = {}
        for metric in CLASSIFICATION_METRICS:
            values = [
                float(cells[(subject, name)]["metrics"][metric])
                - float(cells[(subject, "mlp")]["metrics"][metric])
                for subject in subjects
                if cells[(subject, name)]["metrics"].get(metric) is not None
                and cells[(subject, "mlp")]["metrics"].get(metric) is not None
            ]
            array = np.asarray(values, dtype=np.float64)
            metric_map[metric] = {
                "mean_delta_vs_mlp": (
                    float(array.mean()) if len(array) else None
                ),
                "std_delta_vs_mlp": (
                    float(array.std(ddof=0)) if len(array) else None
                ),
                "n_paired_folds": int(len(array)),
            }
        result[name] = {
            "reference": "mlp",
            "common_subjects": subjects,
            "metrics": metric_map,
        }
    return result


def validate_summaries(
    root: Path,
    config: dict[str, Any],
    definitions: list[dict[str, Any]],
    cells: Mapping[tuple[str, str], dict[str, Any]],
    tolerance: float,
) -> None:
    expected_keys = set(cells)
    _, fold_rows = read_csv(root / "fold_summary.csv")
    require(
        len(fold_rows) == len(expected_keys),
        "fold_summary: stale completed-cell count",
    )
    fold_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in fold_rows:
        key = (row["test_subject"], row["classifier"])
        require(key not in fold_by_key, f"fold_summary: duplicate {key}")
        fold_by_key[key] = row
    require(
        set(fold_by_key) == expected_keys,
        "fold_summary: completed-cell identities mismatch",
    )
    fold_numeric = (
        "history_seconds",
        "history_samples",
        "history_blocks",
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
    )
    for key, row in fold_by_key.items():
        metrics = cells[key]["metrics"]
        for field in (
            "experiment_id",
            "classifier",
            "display_name",
            "nbm",
            "input",
            "test_subject",
            "val_subject",
            "initial_state_sha256",
            "source_residual_sha256",
            "input_support_sha256",
        ):
            require(
                row[field] == str(metrics[field]),
                f"fold_summary/{key}/{field}: mismatch",
            )
        for field in fold_numeric:
            assert_close(
                csv_number(row[field]),
                metrics.get(field),
                f"fold_summary/{key}/{field}",
                tolerance,
            )

    _, manifest_rows = read_csv(root / "experiment_manifest.csv")
    require(len(manifest_rows) == 4, "experiment_manifest: expected four rows")
    manifest_by_name = {row["classifier"]: row for row in manifest_rows}
    require(
        set(manifest_by_name) == set(CANONICAL_CLASSIFIER_NAMES),
        "experiment_manifest: classifier rows mismatch",
    )
    for definition in definitions:
        name = definition["classifier"]
        completed = [
            subject
            for subject in EXPECTED_SUBJECTS
            if (subject, name) in cells
        ]
        row = manifest_by_name[name]
        expected = {
            "experiment_id": definition["experiment_id"],
            "classifier": name,
            "display_name": definition["display_name"],
            "family": definition["architecture"]["family"],
            "parameter_count": str(definition["parameter_count"]),
            "expected_folds": "8",
            "completed_folds": str(len(completed)),
            "status": (
                "complete"
                if len(completed) == 8
                else ("partial" if completed else "pending")
            ),
            "completed_subjects": ",".join(completed),
        }
        require(row == expected, f"experiment_manifest/{name}: row mismatch")

    aggregate = load_json(root / "aggregate_metrics.json")
    expected_group_keys: set[str] = set()
    for definition in definitions:
        name = definition["classifier"]
        evidence = [
            cells[(subject, name)]
            for subject in EXPECTED_SUBJECTS
            if (subject, name) in cells
        ]
        if not evidence:
            continue
        experiment_id = definition["experiment_id"]
        expected_group_keys.add(experiment_id)
        require(experiment_id in aggregate, f"aggregate: missing {experiment_id}")
        group = aggregate[experiment_id]
        expected_group = {
            "classifier": name,
            "display_name": definition["display_name"],
            "architecture": definition["architecture"],
            "parameter_count": definition["parameter_count"],
            "completed_folds": [
                subject
                for subject in EXPECTED_SUBJECTS
                if (subject, name) in cells
            ],
            "subject_macro": aggregate_fold_metrics(
                [item["metrics"] for item in evidence],
                CLASSIFICATION_METRICS,
            ),
            "pooled": pooled_metrics(evidence),
        }
        assert_json_metric_map(
            group,
            expected_group,
            f"aggregate/{name}",
            tolerance,
        )
    expected_paired = paired_deltas(cells)
    require(
        aggregate.get("paired_deltas_vs_mlp") is not None,
        "aggregate: missing paired deltas",
    )
    assert_json_metric_map(
        aggregate["paired_deltas_vs_mlp"],
        expected_paired,
        "aggregate/paired",
        tolerance,
    )
    require(
        set(aggregate) == expected_group_keys | {"paired_deltas_vs_mlp"},
        "aggregate: stale or unexpected groups",
    )

    _, summary_rows = read_csv(root / "aggregate_summary.csv")
    expected_summary_names = {
        name
        for name in CANONICAL_CLASSIFIER_NAMES
        if any((subject, name) in cells for subject in EXPECTED_SUBJECTS)
    }
    summary_by_name = {row["classifier"]: row for row in summary_rows}
    require(
        set(summary_by_name) == expected_summary_names,
        "aggregate_summary: stale classifier rows",
    )
    definition_by_name = {
        definition["classifier"]: definition for definition in definitions
    }
    for name, row in summary_by_name.items():
        definition = definition_by_name[name]
        group = aggregate[definition["experiment_id"]]
        require(
            row["display_name"] == definition["display_name"]
            and int(row["parameter_count"]) == definition["parameter_count"]
            and int(row["completed_folds"]) == len(group["completed_folds"]),
            f"aggregate_summary/{name}: metadata mismatch",
        )
        for metric in SUMMARY_METRICS:
            for statistic in ("mean", "std"):
                assert_close(
                    csv_number(row[f"{metric}_{statistic}"]),
                    group["subject_macro"][metric][statistic],
                    f"aggregate_summary/{name}/{metric}_{statistic}",
                    tolerance,
                )

    status = load_json(root / "status.json")
    expected_status = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_experiments": 4,
        "expected_fold_cells": 32,
        "completed_fold_cells": len(cells),
        "status": "complete" if len(cells) == 32 else "partial",
    }
    require(status == expected_status, "status.json is stale or inconsistent")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.result_dir.resolve()
    require(root.is_dir(), f"result directory does not exist: {root}")
    config = load_json(root / "config.json")
    folds, definitions = validate_protocol(config)
    source_root = select_existing_path(
        configured_path(config, "source_suite_dir"),
        args.source_suite_dir,
        "source suite directory",
    )
    data_root = select_existing_path(
        configured_path(config, "data_dir"),
        args.data_dir,
        "processed Daphnet data directory",
    )
    source_support, source_hashes = validate_source_suite(
        source_root, config, folds
    )
    dataset, windows = load_dataset_and_windows(
        data_root, source_root, config
    )
    validate_run_manifest(root, config)

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    failures: list[str] = []
    for subject in folds:
        done_paths = {
            name: root / f"loso_{subject}" / name / "DONE.json"
            for name in CANONICAL_CLASSIFIER_NAMES
        }
        fold_config_path = root / f"loso_{subject}" / "fold_config.json"
        if not fold_config_path.exists() and not any(
            path.exists() for path in done_paths.values()
        ):
            missing.extend(
                f"{subject}/{name}" for name in CANONICAL_CLASSIFIER_NAMES
            )
            continue
        try:
            fold = validate_fold_support(
                root,
                source_root,
                config,
                subject,
                source_support[subject],
                source_hashes[subject],
                windows,
            )
        except Exception as error:
            failures.append(f"{subject}/fold_support: {error}")
            missing.extend(
                f"{subject}/{name}"
                for name in CANONICAL_CLASSIFIER_NAMES
                if not done_paths[name].exists()
            )
            continue
        per_fold: dict[str, dict[str, Any]] = {}
        for definition in definitions:
            name = definition["classifier"]
            if not done_paths[name].exists():
                missing.append(f"{subject}/{name}")
                continue
            try:
                evidence = validate_cell(
                    root,
                    config,
                    definition,
                    subject,
                    fold,
                    source_hashes[subject],
                    dataset,
                    windows,
                    args.tolerance,
                )
                cells[(subject, name)] = evidence
                per_fold[name] = evidence
            except Exception as error:
                failures.append(f"{subject}/{name}: {type(error).__name__}: {error}")
        try:
            validate_cross_classifier_fairness(subject, per_fold)
        except Exception as error:
            failures.append(
                f"{subject}/cross_classifier_fairness: "
                f"{type(error).__name__}: {error}"
            )

    if missing and not args.allow_partial:
        failures.extend(f"missing {task_id}" for task_id in missing)
    try:
        validate_summaries(root, config, definitions, cells, args.tolerance)
    except Exception as error:
        failures.append(f"root summaries: {type(error).__name__}: {error}")
    full_complete = (
        len(cells) == 32 and not missing and not failures
    )
    return {
        "audit_version": AUDIT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "source_suite_dir": str(source_root),
        "data_dir": str(data_root),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_cells": 32,
        "checked_cells": len(cells),
        "missing_cells": missing,
        "allow_partial": bool(args.allow_partial),
        "full_complete": bool(full_complete),
        "failures": failures,
        "warnings": [],
        "status": "pass" if not failures else "fail",
    }


def main() -> None:
    args = parse_args()
    root = args.result_dir.resolve()
    try:
        report = audit(args)
    except Exception as error:
        report = {
            "audit_version": AUDIT_VERSION,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_dir": str(root),
            "expected_cells": 32,
            "checked_cells": 0,
            "missing_cells": [],
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
    if report.get("status") == "pass" and report.get("full_complete") is True:
        atomic_json_dump(
            {
                "format_version": 1,
                "suite_version": SUITE_VERSION,
                "audit_version": AUDIT_VERSION,
                "status": "complete",
                "protocol_fingerprint": report["protocol_fingerprint"],
                "expected_cells": 32,
                "checked_cells": 32,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_report_sha256": sha256_file(report_path),
            },
            complete_path,
        )
    elif complete_path.exists():
        complete_path.unlink()

    print(
        f"[classifier-audit] status={report['status']} "
        f"checked={report.get('checked_cells', 0)}/32 "
        f"missing={len(report.get('missing_cells', []))}",
        flush=True,
    )
    for failure in report.get("failures", []):
        print(f"[classifier-audit] ERROR {failure}", flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
