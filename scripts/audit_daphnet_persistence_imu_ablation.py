#!/usr/bin/env python
"""Independent audit for the seven-way Persistence IMU-input ablation.

The auditor treats the completed experiment directory as untrusted evidence.  It
reconstructs the immutable protocol, verifies the canonical Persistence source
chain, checks the common four-second history support, independently derives each
sensor-subset fingerprint and initial TCN-M state, validates both completion
stages for every classifier, recomputes window- and event-level metrics from
saved predictions, and checks the root summaries.

``--allow-partial`` permits interim inspection of an interrupted run.  A partial
audit never creates ``SUITE_COMPLETE.json``.  That marker is written only for a
reportable formal protocol after all 7 x 8 = 56 cells pass every check.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_daphnet_tcn_rf_ablation as source_audit
import run_daphnet_persistence_imu_ablation as suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    canonical_fingerprint,
    sha256_file,
    validate_done,
)
from run_cnbr_fog_loso import event_metrics


AUDIT_VERSION = "daphnet_persistence_imu_ablation_audit.v1"
SUITE_VERSION = "daphnet_persistence_h4_tcnm_imu7_loso.v1"
SOURCE_SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
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
EXPECTED_IMU_VARIANTS = (
    "ankle",
    "thigh",
    "trunk",
    "ankle_thigh",
    "ankle_trunk",
    "thigh_trunk",
    "all_three",
)
EXPECTED_VARIANTS = EXPECTED_IMU_VARIANTS
CANONICAL_IMU_VARIANTS = EXPECTED_IMU_VARIANTS
EXPECTED_CLASSIFIER_CELLS = 56
EXPECTED_CELLS = EXPECTED_CLASSIFIER_CELLS
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_DILATIONS = (1, 2, 4, 8, 8, 8)
EXPECTED_RF_SAMPLES = 125
PROJECTION_WEIGHT_KEY = "projection.0.weight"
EXPECTED_VARIANT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "ankle": {
        "display_name": "Ankle",
        "sensor_count": 1,
        "channel_indices": (0, 1, 2),
    },
    "thigh": {
        "display_name": "Thigh",
        "sensor_count": 1,
        "channel_indices": (3, 4, 5),
    },
    "trunk": {
        "display_name": "Trunk",
        "sensor_count": 1,
        "channel_indices": (6, 7, 8),
    },
    "ankle_thigh": {
        "display_name": "Ankle+Thigh",
        "sensor_count": 2,
        "channel_indices": (0, 1, 2, 3, 4, 5),
    },
    "ankle_trunk": {
        "display_name": "Ankle+Trunk",
        "sensor_count": 2,
        "channel_indices": (0, 1, 2, 6, 7, 8),
    },
    "thigh_trunk": {
        "display_name": "Thigh+Trunk",
        "sensor_count": 2,
        "channel_indices": (3, 4, 5, 6, 7, 8),
    },
    "all_three": {
        "display_name": "All three",
        "sensor_count": 3,
        "channel_indices": tuple(range(9)),
    },
}
CLASSIFICATION_METRICS = tuple(rf.CLASSIFICATION_METRICS)
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
RUNTIME_FIELDS = {
    "data_dir",
    "source_suite_dir",
    "output_dir",
    "device",
    "num_workers",
    "resume",
    "smoke",
}
EXPECTED_SUPPORT_KEYS = {
    f"{split}_{suffix}"
    for split in EXPECTED_SPLITS
    for suffix in ("anchor_window_index", "history_window_index", "y")
}
EXPECTED_SOURCE_CACHE_KEYS = {
    f"{split}_{suffix}"
    for split in EXPECTED_SPLITS
    for suffix in ("residual", "y", "window_index")
}
EXPECTED_CLASSIFIER_ARTIFACTS = {
    "best",
    "last",
    "metrics",
    "predictions",
    "validation_predictions",
    "predictions_csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Daphnet Persistence + TCN-M seven-IMU suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        help="Fallback canonical 5x4 NBM suite path for copied results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Fallback processed Daphnet path for copied results",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Audit completed cells without requiring all 56 cells",
    )
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
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=float(tolerance),
            abs_tol=float(tolerance),
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


def protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RUNTIME_FIELDS | {"protocol_fingerprint"}
    }


def configured_path(config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def choose_existing_path(
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


def artifact_path(
    done_path: Path,
    entry: Mapping[str, Any],
) -> Path:
    path = Path(str(entry["path"]))
    return path if path.is_absolute() else (done_path.parent / path)


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return rf.state_dict_sha256(dict(state))


def _common_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return _state_hash(
        {
            key: tensor
            for key, tensor in state.items()
            if key != PROJECTION_WEIGHT_KEY
        }
    )


def independently_reconstruct_initializations(
    *,
    seed: int,
    hidden_channels: int,
    dropout: float,
) -> dict[str, dict[str, Any]]:
    """Rebuild the reference state and every projection slice from first principles."""

    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(int(seed))
        reference = ResidualTCNClassifier(
            in_channels=9,
            hidden_channels=int(hidden_channels),
            dilations=EXPECTED_DILATIONS,
            kernel_size=3,
            dropout=float(dropout),
        ).cpu()
        reference_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in reference.state_dict().items()
        }
        identities: dict[str, dict[str, Any]] = {}
        common_hashes: set[str] = set()
        for name, definition in EXPECTED_VARIANT_DEFINITIONS.items():
            indices = tuple(int(value) for value in definition["channel_indices"])
            model = ResidualTCNClassifier(
                in_channels=len(indices),
                hidden_channels=int(hidden_channels),
                dilations=EXPECTED_DILATIONS,
                kernel_size=3,
                dropout=float(dropout),
            ).cpu()
            expected_state: dict[str, torch.Tensor] = {}
            for key, target in model.state_dict().items():
                source = (
                    reference_state[key][:, indices, :]
                    if key == PROJECTION_WEIGHT_KEY
                    else reference_state[key]
                )
                require(
                    tuple(target.shape) == tuple(source.shape),
                    f"initialization/{name}/{key}: projection slice shape mismatch",
                )
                expected_state[key] = source.detach().clone()
            model.load_state_dict(expected_state, strict=True)
            state = model.state_dict()
            identity = {
                "variant": name,
                "classifier_seed": int(seed),
                "n_channels": len(indices),
                "channel_indices": list(indices),
                "parameter_count": rf.parameter_count(model),
                "initial_state_sha256": _state_hash(state),
                "common_state_sha256": _common_state_hash(state),
                "projection_weight_sha256": _state_hash(
                    {PROJECTION_WEIGHT_KEY: state[PROJECTION_WEIGHT_KEY]}
                ),
            }
            identities[name] = identity
            common_hashes.add(str(identity["common_state_sha256"]))
        require(
            len(common_hashes) == 1,
            "independent reconstruction did not produce one common state",
        )
        return identities
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def expected_input_fingerprint(
    *,
    source_cache_sha256: str,
    support_sha256: str,
    variant: Mapping[str, Any],
) -> str:
    return canonical_fingerprint(
        {
            "source_residual_cache_sha256": str(source_cache_sha256),
            "input_support_sha256": str(support_sha256),
            "representation": (
                "canonical_clipped_standardized_persistence_residual"
            ),
            "channel_indices": list(variant["channel_indices"]),
            "channel_names": list(variant["channel_names"]),
            "n_channels": int(variant["n_channels"]),
            "history_samples": 256,
            "history_blocks": 8,
            "horizon_samples": 32,
            "source_stride_samples": 16,
        }
    )


def validate_protocol(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    require(
        config.get("suite_version") == SUITE_VERSION,
        f"config: expected suite_version {SUITE_VERSION}",
    )
    require(
        canonical_fingerprint(protocol_payload(config))
        == config.get("protocol_fingerprint"),
        "config: protocol fingerprint mismatch",
    )
    require(int(config.get("sampling_rate_hz", -1)) == 64, "config: expected 64 Hz")
    require(
        int(config.get("n_source_channels", -1)) == 9,
        "config: canonical source must have nine channels",
    )
    require(
        tuple(config.get("channel_names", ())) == EXPECTED_CHANNELS,
        "config: canonical channel order changed",
    )
    require(
        tuple(config.get("subjects", ())) == EXPECTED_SUBJECTS
        and tuple(config.get("folds_resolved", ())) == EXPECTED_SUBJECTS,
        "config: LOSO subjects/folds are not canonical",
    )
    require(
        set(config.get("excluded_subjects", ())) == EXPECTED_EXCLUDED,
        "config: exclusions must be exactly S04 and S10",
    )
    require(
        config.get("nbm") == "persistence"
        and config.get("input") == "residual_h4s",
        "config: expected frozen Persistence residual_h4s",
    )
    expected_scalars = {
        "history_samples": 256,
        "history_blocks": 8,
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
        "expected_experiments": 7,
        "expected_classifier_cells": EXPECTED_CLASSIFIER_CELLS,
    }
    for key, expected in expected_scalars.items():
        require(
            int(config.get(key, -1)) == expected,
            f"config/{key}: {config.get(key)!r} != {expected!r}",
        )
    assert_close(config.get("history_seconds"), 4.0, "config/history_seconds", 1e-12)
    classifier = config.get("classifier")
    require(isinstance(classifier, dict), "config: missing classifier block")
    require(
        classifier.get("name") == "tcn_m"
        and tuple(classifier.get("dilations", ())) == EXPECTED_DILATIONS
        and int(classifier.get("receptive_field_samples", -1))
        == EXPECTED_RF_SAMPLES
        and int(classifier.get("kernel_size", -1)) == 3
        and int(classifier.get("convolutions_per_block", -1)) == 2,
        "config: classifier is not the fixed TCN-M",
    )
    require(
        int(classifier.get("hidden_channels", -1)) > 0
        and 0.0 <= float(classifier.get("dropout", -1.0)) < 1.0,
        "config: invalid TCN-M width/dropout",
    )
    if bool(config.get("reportable")):
        formal_values = {
            "hidden_channels": (int(classifier["hidden_channels"]), 48),
            "dropout": (float(classifier["dropout"]), 0.15),
            "classifier_epochs": (int(config["classifier_epochs"]), 12),
            "classifier_patience": (int(config["classifier_patience"]), 4),
            "classifier_lr": (float(config["classifier_lr"]), 1e-3),
            "weight_decay": (float(config["weight_decay"]), 1e-4),
            "batch_size": (int(config["batch_size"]), 256),
            "max_classifier_windows": (
                int(config["max_classifier_windows"]),
                0,
            ),
            "bootstrap_samples": (int(config["bootstrap_samples"]), 100000),
            "bootstrap_seed": (int(config["bootstrap_seed"]), 42),
            "deterministic": (bool(config["deterministic"]), True),
            "amp": (bool(config["amp"]), True),
        }
        changed = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in formal_values.items()
            if actual != expected
        }
        require(not changed, f"config: formal protocol values changed: {changed}")
    variants = config.get("variants")
    require(
        isinstance(variants, list)
        and [item.get("variant") for item in variants]
        == list(EXPECTED_IMU_VARIANTS),
        "config: seven IMU variants/order changed",
    )
    reference_identities = independently_reconstruct_initializations(
        seed=int(config["seed"]),
        hidden_channels=int(classifier["hidden_channels"]),
        dropout=float(classifier["dropout"]),
    )
    for variant in variants:
        name = str(variant["variant"])
        expected = EXPECTED_VARIANT_DEFINITIONS[name]
        indices = expected["channel_indices"]
        names = tuple(EXPECTED_CHANNELS[index] for index in indices)
        require(
            variant.get("display_name") == expected["display_name"]
            and int(variant.get("sensor_count", -1))
            == int(expected["sensor_count"])
            and int(variant.get("n_channels", -1)) == len(indices)
            and tuple(variant.get("channel_indices", ())) == indices
            and tuple(variant.get("channel_names", ())) == names,
            f"config/{name}: sensor definition mismatch",
        )
        require(
            tuple(variant.get("dilations", ())) == EXPECTED_DILATIONS
            and int(variant.get("n_blocks", -1)) == 6
            and int(variant.get("convolutions_per_block", -1)) == 2
            and int(variant.get("kernel_size", -1)) == 3
            and int(variant.get("receptive_field_samples", -1))
            == EXPECTED_RF_SAMPLES,
            f"config/{name}: TCN-M metadata mismatch",
        )
        identity = reference_identities[name]
        require(
            int(variant.get("parameter_count", -1))
            == int(identity["parameter_count"])
            and variant.get("reference_initial_state_sha256")
            == identity["initial_state_sha256"]
            and variant.get("reference_common_state_sha256")
            == identity["common_state_sha256"],
            f"config/{name}: reference initialization mismatch",
        )
        require(
            int(variant.get("input_projection_parameters", -1))
            == int(classifier["hidden_channels"]) * len(indices),
            f"config/{name}: input projection count mismatch",
        )
        assert_close(
            variant.get("input_bandwidth_ratio_vs_all"),
            len(indices) / 9.0,
            f"config/{name}/input_bandwidth_ratio",
            1e-12,
        )
    require(
        config.get("comparisons") == list(suite.COMPARISONS),
        "config: paired comparison definitions changed",
    )
    fairness = config.get("fairness_contract")
    require(isinstance(fairness, dict), "config: missing fairness contract")
    require(
        fairness.get("same_source_scaler_nbm_sigma_and_residual_cache") is True
        and fairness.get("same_fold_history_anchors_and_labels") is True
        and fairness.get("same_tcn_m_common_parameter_values") is True
        and fairness.get("same_post_initialization_rng_state") is True,
        "config: core sensor-ablation fairness contract is disabled",
    )
    implementation = config.get("implementation")
    require(isinstance(implementation, dict), "config: missing implementation manifest")
    files = implementation.get("files")
    require(
        isinstance(files, dict)
        and set(files) == set(suite.IMPLEMENTATION_FILES),
        "config: implementation file set changed",
    )
    current = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in suite.IMPLEMENTATION_FILES
    }
    require(files == current, "config: implementation source hashes changed")
    require(
        implementation.get("sha256") == canonical_fingerprint(current),
        "config: implementation manifest hash mismatch",
    )
    return [dict(item) for item in variants]


def validate_run_manifest(root: Path, config: Mapping[str, Any]) -> None:
    saved = load_json(root / "run_manifest.json")
    expected = {
        key: value for key, value in config.items() if key not in RUNTIME_FIELDS
    }
    require(saved == expected, "run_manifest.json differs from config protocol")


def load_source_cache(
    source_root: Path,
    source_config: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, dict[str, np.ndarray]], str, str]:
    protocol = str(source_config["protocol_fingerprint"])
    persistence = source_root / f"loso_{subject}" / "persistence"
    nbm_done_path = persistence / "nbm" / "DONE.json"
    nbm_done = validate_done(
        nbm_done_path,
        stage="nbm",
        protocol_fingerprint=protocol,
        task_id=f"loso_{subject}/persistence/nbm",
    )
    require(nbm_done is not None, f"source/{subject}: missing NBM DONE")
    best_entry = nbm_done.get("artifacts", {}).get("best")
    require(best_entry is not None, f"source/{subject}: NBM DONE lacks best")
    nbm_sha = str(best_entry["sha256"])
    residual_done_path = persistence / "RESIDUAL_CACHE_DONE.json"
    residual_done = validate_done(
        residual_done_path,
        stage="residual_cache",
        protocol_fingerprint=protocol,
        task_id=f"loso_{subject}/persistence/residual_cache",
        upstream_sha256=nbm_sha,
    )
    require(residual_done is not None, f"source/{subject}: missing residual DONE")
    cache_entry = residual_done.get("artifacts", {}).get("cache")
    require(cache_entry is not None, f"source/{subject}: residual DONE lacks cache")
    cache_path = artifact_path(residual_done_path, cache_entry)
    with np.load(cache_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == EXPECTED_SOURCE_CACHE_KEYS,
            f"source/{subject}: unexpected residual-cache arrays",
        )
        extracted = {
            split: {
                "residual": np.asarray(
                    payload[f"{split}_residual"], dtype=np.float32
                ),
                "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
                "window_index": np.asarray(
                    payload[f"{split}_window_index"], dtype=np.int64
                ),
            }
            for split in EXPECTED_SPLITS
        }
    for split, arrays in extracted.items():
        residual = arrays["residual"]
        require(
            residual.ndim == 3 and residual.shape[1:] == (9, 32),
            f"source/{subject}/{split}: residual shape is not [N,9,32]",
        )
        require(
            len(residual) == len(arrays["y"]) == len(arrays["window_index"])
            and np.isfinite(residual).all(),
            f"source/{subject}/{split}: residual cache is invalid",
        )
    return extracted, str(cache_entry["sha256"]), nbm_sha


def _support_subset_matches(
    local_anchor: np.ndarray,
    local_history: np.ndarray,
    source_anchor: np.ndarray,
    source_history: np.ndarray,
) -> bool:
    lookup = {
        int(anchor): row for row, anchor in enumerate(source_anchor.tolist())
    }
    try:
        source_rows = np.asarray(
            [lookup[int(anchor)] for anchor in local_anchor],
            dtype=np.int64,
        )
    except KeyError:
        return False
    return np.array_equal(local_history, source_history[source_rows])


def validate_fold(
    *,
    result_root: Path,
    source_root: Path,
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    subject: str,
    variants: list[dict[str, Any]],
    dataset: Any,
    windows: Any,
) -> dict[str, Any]:
    fold_root = result_root / f"loso_{subject}"
    fold_config_path = fold_root / "fold_config.json"
    provenance_path = fold_root / "source_provenance.json"
    support_path = fold_root / "input_support.npz"
    fingerprint_path = fold_root / "sensor_input_fingerprints.json"
    initialization_path = fold_root / "sensor_model_initialization.json"
    imu_fold_path = fold_root / "imu_fold_config.json"
    for path in (
        fold_config_path,
        provenance_path,
        support_path,
        fingerprint_path,
        initialization_path,
        imu_fold_path,
    ):
        require(path.exists(), f"{subject}: missing {path.name}")

    fold_config = load_json(fold_config_path)
    provenance = load_json(provenance_path)
    require(
        fold_config.get("suite_version") == SUITE_VERSION
        and fold_config.get("protocol_fingerprint")
        == config["protocol_fingerprint"]
        and fold_config.get("test_subject") == subject,
        f"{subject}: fold identity mismatch",
    )
    require(
        fold_config.get("source") == provenance,
        f"{subject}: fold source and source_provenance.json differ",
    )
    expected_seed = int(config["seed"]) + 10000 + EXPECTED_SUBJECTS.index(subject)
    require(
        int(fold_config.get("classifier_seed", -1)) == expected_seed,
        f"{subject}: classifier seed mismatch",
    )
    require(
        fold_config.get("input") == "residual_h4s"
        and int(fold_config.get("history_samples", -1)) == 256
        and int(fold_config.get("history_blocks", -1)) == 8,
        f"{subject}: fold is not four-second residual_h4s",
    )

    source_features, source_cache_sha, source_nbm_sha = load_source_cache(
        source_root,
        source_config,
        subject,
    )
    declared_source = config["source"]["folds"][subject]
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
            provenance.get(key) == declared_source.get(key),
            f"{subject}: source provenance mismatch for {key}",
        )
    require(
        provenance.get("source_residual_cache_sha256") == source_cache_sha
        and provenance.get("source_nbm_best_sha256") == source_nbm_sha,
        f"{subject}: fold is not bound to canonical Persistence artifacts",
    )

    support_sha = sha256_file(support_path)
    require(
        provenance.get("input_support_sha256") == support_sha,
        f"{subject}: input support SHA mismatch",
    )
    with np.load(support_path, allow_pickle=False) as payload:
        require(
            set(payload.files) == EXPECTED_SUPPORT_KEYS,
            f"{subject}: unexpected input-support arrays",
        )
        support = {
            split: {
                "anchor": np.asarray(
                    payload[f"{split}_anchor_window_index"], dtype=np.int64
                ),
                "history": np.asarray(
                    payload[f"{split}_history_window_index"], dtype=np.int64
                ),
                "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
            }
            for split in EXPECTED_SPLITS
        }
    source_support_path = (
        source_root / f"loso_{subject}" / "history_support.npz"
    )
    with np.load(source_support_path, allow_pickle=False) as payload:
        expected_keys = {
            f"{split}_{suffix}"
            for split in EXPECTED_SPLITS
            for suffix in ("anchor_window_index", "history_window_index")
        }
        require(
            set(payload.files) == expected_keys,
            f"source/{subject}: unexpected history-support arrays",
        )
        source_support = {
            split: {
                "anchor": np.asarray(
                    payload[f"{split}_anchor_window_index"], dtype=np.int64
                ),
                "history": np.asarray(
                    payload[f"{split}_history_window_index"], dtype=np.int64
                ),
            }
            for split in EXPECTED_SPLITS
        }
    for split in EXPECTED_SPLITS:
        anchor = support[split]["anchor"]
        history = support[split]["history"]
        truth = support[split]["y"]
        require(
            anchor.ndim == truth.ndim == 1
            and history.shape == (len(anchor), 8)
            and len(anchor) == len(truth)
            and len(anchor) > 0,
            f"{subject}/{split}: invalid support shapes",
        )
        require(
            len(np.unique(anchor)) == len(anchor)
            and np.array_equal(history[:, -1], anchor)
            and np.isin(truth, (0, 1)).all(),
            f"{subject}/{split}: invalid anchor/history/label support",
        )
        if split != "train" or int(config["max_classifier_windows"]) == 0:
            require(
                np.array_equal(anchor, source_support[split]["anchor"])
                and np.array_equal(history, source_support[split]["history"]),
                f"{subject}/{split}: support differs from source residual_h4s",
            )
        else:
            require(
                _support_subset_matches(
                    anchor,
                    history,
                    source_support[split]["anchor"],
                    source_support[split]["history"],
                ),
                f"{subject}/{split}: smoke support is not a source subset",
            )
        require(
            np.array_equal(truth, windows.label[anchor]),
            f"{subject}/{split}: support labels differ from WindowTable",
        )
        cache_indices = source_features[split]["window_index"]
        cache_labels = source_features[split]["y"]
        row_by_id = {
            int(window_id): row
            for row, window_id in enumerate(cache_indices.tolist())
        }
        try:
            chain_rows = np.asarray(
                [
                    [row_by_id[int(window_id)] for window_id in chain]
                    for chain in history
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise AssertionError(
                f"{subject}/{split}: support references absent residual block"
            ) from error
        require(
            np.array_equal(cache_indices[chain_rows[:, -1]], anchor)
            and np.array_equal(cache_labels[chain_rows[:, -1]], truth),
            f"{subject}/{split}: support/cache alignment mismatch",
        )
        record_ids = windows.record_index[history]
        starts = windows.target_start[history]
        ends = windows.target_end[history]
        require(
            np.all(record_ids == record_ids[:, :1])
            and np.all(np.diff(starts, axis=1) == 32)
            and np.all(ends - starts == 32),
            f"{subject}/{split}: history is not 8 contiguous 0.5-s blocks",
        )
        require(
            int(fold_config["history_anchor_counts"][split]) == len(anchor),
            f"{subject}/{split}: stale anchor count",
        )

    fingerprints = load_json(fingerprint_path)
    expected_fingerprints = {
        str(variant["variant"]): expected_input_fingerprint(
            source_cache_sha256=source_cache_sha,
            support_sha256=support_sha,
            variant=variant,
        )
        for variant in variants
    }
    require(
        fingerprints
        == {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "source_residual_cache_sha256": source_cache_sha,
            "input_support_sha256": support_sha,
            "variants": expected_fingerprints,
        },
        f"{subject}: sensor input fingerprints are not reproducible",
    )

    expected_identities = independently_reconstruct_initializations(
        seed=expected_seed,
        hidden_channels=int(config["classifier"]["hidden_channels"]),
        dropout=float(config["classifier"]["dropout"]),
    )
    initialization = load_json(initialization_path)
    require(
        initialization
        == {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "strategy": (
                "canonical_9ch_reference_common_copy_and_projection_slice"
            ),
            "projection_weight_key": PROJECTION_WEIGHT_KEY,
            "common_state_shared": True,
            "variants": expected_identities,
        },
        f"{subject}: saved common/projection initialization is not reproducible",
    )
    require(
        len(
            {
                identity["common_state_sha256"]
                for identity in expected_identities.values()
            }
        )
        == 1,
        f"{subject}: common initialization differs by sensor subset",
    )
    require(
        fold_config.get("reference_initial_state_sha256")
        == expected_identities["all_three"]["initial_state_sha256"],
        f"{subject}: fold reference is not the reconstructed 9-channel state",
    )

    imu_fold = load_json(imu_fold_path)
    expected_imu_fold = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "classifier_seed": expected_seed,
        "source_fold_config_sha256": sha256_file(fold_config_path),
        "sensor_input_fingerprints_sha256": sha256_file(fingerprint_path),
        "sensor_model_initialization_sha256": sha256_file(initialization_path),
        "source_residual_cache_sha256": source_cache_sha,
        "input_support_sha256": support_sha,
    }
    require(
        imu_fold == expected_imu_fold,
        f"{subject}: imu_fold_config.json mismatch",
    )
    return {
        "root": fold_root,
        "config": fold_config,
        "provenance": provenance,
        "support": support,
        "support_sha256": support_sha,
        "source_cache_sha256": source_cache_sha,
        "source_nbm_sha256": source_nbm_sha,
        "fingerprints": expected_fingerprints,
        "identities": expected_identities,
        "val_subject": str(fold_config["val_subject"]),
        "classifier_seed": expected_seed,
    }


def load_predictions(path: Path, label: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        require(
            set(payload.files)
            == {"window_index", "y_true", "y_prob", "y_pred"},
            f"{label}: unexpected prediction arrays",
        )
        arrays = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    shapes = {array.shape for array in arrays.values()}
    require(
        len(shapes) == 1
        and arrays["window_index"].ndim == 1
        and len(arrays["window_index"]) > 0,
        f"{label}: prediction arrays are empty or misaligned",
    )
    require(
        len(np.unique(arrays["window_index"]))
        == len(arrays["window_index"])
        and np.isin(arrays["y_true"], (0, 1)).all()
        and np.isin(arrays["y_pred"], (0, 1)).all()
        and np.isfinite(arrays["y_prob"]).all()
        and np.all((arrays["y_prob"] >= 0) & (arrays["y_prob"] <= 1)),
        f"{label}: invalid prediction values",
    )
    return arrays


def compare_metric_dict(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    keys: Iterable[str],
    label: str,
    tolerance: float,
) -> None:
    for key in keys:
        require(key in saved, f"{label}: missing {key}")
        if key in COUNT_METRICS:
            require(
                int(saved[key]) == int(expected[key]),
                f"{label}/{key}: {saved[key]!r} != {expected[key]!r}",
            )
        else:
            assert_close(
                saved[key],
                expected[key],
                f"{label}/{key}",
                tolerance,
            )


def expected_imu_metadata(
    *,
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
    variant: Mapping[str, Any],
    subject: str,
) -> dict[str, Any]:
    name = str(variant["variant"])
    identity = fold["identities"][name]
    return {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": f"{subject}/{name}",
        "experiment_id": variant["experiment_id"],
        "variant": name,
        "display_name": variant["display_name"],
        "sensor_count": int(variant["sensor_count"]),
        "n_channels": int(variant["n_channels"]),
        "channel_indices": list(variant["channel_indices"]),
        "channel_names": list(variant["channel_names"]),
        "input_shape": ["batch", int(variant["n_channels"]), 256],
        "sensor_input_sha256": fold["fingerprints"][name],
        "source_residual_cache_sha256": fold["source_cache_sha256"],
        "input_support_sha256": fold["support_sha256"],
        "source_nbm_best_sha256": fold["source_nbm_sha256"],
        "parameter_count": int(identity["parameter_count"]),
        "initial_state_sha256": identity["initial_state_sha256"],
        "common_state_sha256": identity["common_state_sha256"],
        "projection_weight_sha256": identity["projection_weight_sha256"],
        "initialization_strategy": (
            "canonical_9ch_reference_common_copy_and_projection_slice"
        ),
        "history_seconds": 4.0,
        "history_samples": 256,
        "history_blocks": 8,
    }


def expected_classifier_config(
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "in_channels": int(variant["n_channels"]),
        "hidden_channels": int(config["classifier"]["hidden_channels"]),
        "dropout": float(config["classifier"]["dropout"]),
        "kernel_size": 3,
        "dilations": list(EXPECTED_DILATIONS),
        "n_blocks": 6,
        "convolutions_per_block": 2,
        "receptive_field_samples": EXPECTED_RF_SAMPLES,
        "receptive_field_seconds": EXPECTED_RF_SAMPLES / 64.0,
        "parameter_count": int(identity["parameter_count"]),
        "initial_state_sha256": identity["initial_state_sha256"],
        "global_pooling": "mean_and_max_over_full_input",
    }


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    task_id: str,
    config: Mapping[str, Any],
    source_cache_sha256: str,
    classifier_seed: int,
    classifier_config: Mapping[str, Any],
    variant_name: str,
    kind: str,
) -> None:
    expected_header = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "rf_classifier",
        "protocol_fingerprint": config["protocol_fingerprint"],
        "task_id": task_id,
        "source_residual_sha256": source_cache_sha256,
        "variant": variant_name,
        "classifier_seed": classifier_seed,
    }
    for key, expected in expected_header.items():
        require(
            checkpoint.get(key) == expected,
            f"{task_id}/{kind}: checkpoint {key} mismatch",
        )
    require(
        checkpoint.get("classifier_config") == dict(classifier_config),
        f"{task_id}/{kind}: classifier configuration mismatch",
    )
    state = checkpoint.get("model_state")
    require(
        isinstance(state, dict) and state,
        f"{task_id}/{kind}: checkpoint has no model state",
    )
    model = ResidualTCNClassifier(
        in_channels=int(classifier_config["in_channels"]),
        hidden_channels=int(classifier_config["hidden_channels"]),
        dilations=EXPECTED_DILATIONS,
        kernel_size=3,
        dropout=float(classifier_config["dropout"]),
    )
    model.load_state_dict(state, strict=True)
    require(
        rf.parameter_count(model) == int(classifier_config["parameter_count"]),
        f"{task_id}/{kind}: checkpoint parameter count mismatch",
    )
    if kind == "last":
        for key in (
            "optimizer_state",
            "grad_scaler_state",
            "epoch",
            "best_epoch",
            "best_score",
            "bad_epochs",
            "history",
            "rng_state",
        ):
            require(key in checkpoint, f"{task_id}/last: missing {key}")


def audit_cell(
    *,
    result_root: Path,
    config: Mapping[str, Any],
    subject: str,
    variant: Mapping[str, Any],
    fold: Mapping[str, Any],
    dataset: Any,
    windows: Any,
    tolerance: float,
) -> dict[str, Any]:
    name = str(variant["variant"])
    task_id = f"{subject}/{name}"
    root = result_root / f"loso_{subject}" / name
    done_path = root / "DONE.json"
    done = validate_done(
        done_path,
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=task_id,
    )
    require(done is not None, f"{task_id}: missing classifier DONE")
    require(
        set(done.get("artifacts", {})) == EXPECTED_CLASSIFIER_ARTIFACTS,
        f"{task_id}: classifier DONE artifact set mismatch",
    )
    identity = fold["identities"][name]
    require(
        done.get("source_residual_sha256") == fold["source_cache_sha256"]
        and done.get("input_support_sha256") == fold["support_sha256"]
        and done.get("initial_state_sha256")
        == identity["initial_state_sha256"],
        f"{task_id}: classifier DONE provenance mismatch",
    )
    expected_paths = {
        "best": root / "classifier_best.pt",
        "last": root / "classifier_last.pt",
        "metrics": root / "metrics.json",
        "predictions": root / "predictions.npz",
        "validation_predictions": root / "validation_predictions.npz",
        "predictions_csv": root / "predictions.csv",
    }
    for key, expected_path in expected_paths.items():
        require(
            artifact_path(done_path, done["artifacts"][key]).resolve()
            == expected_path.resolve(),
            f"{task_id}: DONE path mismatch for {key}",
        )

    metadata_done_path = root / "IMU_METADATA_DONE.json"
    metadata_done = validate_done(
        metadata_done_path,
        stage="imu_variant_metadata",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{task_id}/imu_metadata",
        upstream_sha256=sha256_file(done_path),
    )
    require(metadata_done is not None, f"{task_id}: missing metadata DONE")
    require(
        set(metadata_done.get("artifacts", {})) == {"metadata"},
        f"{task_id}: metadata DONE artifact set mismatch",
    )
    metadata_path = root / "imu_metadata.json"
    require(
        artifact_path(
            metadata_done_path,
            metadata_done["artifacts"]["metadata"],
        ).resolve()
        == metadata_path.resolve(),
        f"{task_id}: metadata DONE path mismatch",
    )
    metadata = load_json(metadata_path)
    expected_metadata = expected_imu_metadata(
        config=config,
        fold=fold,
        variant=variant,
        subject=subject,
    )
    require(
        metadata == expected_metadata,
        f"{task_id}: sensor metadata/fingerprint mismatch",
    )

    metrics = load_json(root / "metrics.json")
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
        "val_subject": fold["val_subject"],
        "classifier_seed": fold["classifier_seed"],
        "initial_state_sha256": identity["initial_state_sha256"],
        "source_residual_sha256": fold["source_cache_sha256"],
        "input_support_sha256": fold["support_sha256"],
    }
    for key, expected in expected_identity.items():
        require(
            metrics.get(key) == expected,
            f"{task_id}: metrics {key} mismatch",
        )
    classifier_config = expected_classifier_config(config, variant, identity)
    require(
        metrics.get("classifier_config") == classifier_config,
        f"{task_id}: metrics classifier config mismatch",
    )

    test = load_predictions(root / "predictions.npz", f"{task_id}/test")
    validation = load_predictions(
        root / "validation_predictions.npz",
        f"{task_id}/validation",
    )
    for split, arrays in (("test", test), ("validation", validation)):
        expected_support = fold["support"][split]
        require(
            np.array_equal(arrays["window_index"], expected_support["anchor"])
            and np.array_equal(arrays["y_true"], expected_support["y"]),
            f"{task_id}/{split}: prediction support or labels mismatch",
        )

    threshold, validation_expected = choose_threshold(
        validation["y_true"],
        validation["y_prob"],
    )
    assert_close(
        metrics.get("threshold"),
        threshold,
        f"{task_id}/threshold",
        tolerance,
    )
    require(
        np.array_equal(
            validation["y_pred"],
            (validation["y_prob"] >= threshold).astype(np.int8),
        ),
        f"{task_id}: validation y_pred does not use selected threshold",
    )
    compare_metric_dict(
        metrics.get("validation", {}),
        validation_expected,
        keys=(*CORE_BINARY_METRICS, *COUNT_METRICS),
        label=f"{task_id}/validation_metrics",
        tolerance=tolerance,
    )
    require(
        metrics["validation"].get("confusion_matrix")
        == validation_expected.get("confusion_matrix"),
        f"{task_id}: validation confusion matrix mismatch",
    )

    test_expected = binary_metrics(test["y_true"], test["y_prob"], threshold)
    require(
        np.array_equal(
            test["y_pred"],
            (test["y_prob"] >= threshold).astype(np.int8),
        ),
        f"{task_id}: test y_pred does not use validation threshold",
    )
    compare_metric_dict(
        metrics,
        test_expected,
        keys=(*CORE_BINARY_METRICS, *COUNT_METRICS),
        label=f"{task_id}/test_metrics",
        tolerance=tolerance,
    )
    require(
        metrics.get("confusion_matrix") == test_expected["confusion_matrix"],
        f"{task_id}: test confusion matrix mismatch",
    )
    requested = dict(test_expected)
    rf.add_requested_metrics(requested)
    compare_metric_dict(
        metrics,
        requested,
        keys=("macro_f1", "roc_auc", "pr_auc", "fog_recall", "fog_f1"),
        label=f"{task_id}/requested_metrics",
        tolerance=tolerance,
    )
    events = event_metrics(
        dataset,
        windows,
        test["window_index"],
        test["y_pred"],
    )
    compare_metric_dict(
        metrics,
        events,
        keys=EVENT_METRICS,
        label=f"{task_id}/event_metrics",
        tolerance=tolerance,
    )

    train_counts = np.bincount(
        fold["support"]["train"]["y"],
        minlength=2,
    ).astype(int)
    require(
        metrics.get("train_counts") == train_counts.tolist(),
        f"{task_id}: train class counts mismatch",
    )
    expected_pos_weight = min(
        math.sqrt(float(train_counts[0]) / float(train_counts[1])),
        6.0,
    )
    assert_close(
        metrics.get("pos_weight"),
        expected_pos_weight,
        f"{task_id}/pos_weight",
        tolerance,
    )
    history = metrics.get("history")
    require(
        isinstance(history, list)
        and history
        and len(history) <= int(config["classifier_epochs"]),
        f"{task_id}: invalid training history",
    )
    for row in history:
        epoch = int(row["epoch"])
        require(
            int(row["shuffle_seed"]) == int(fold["classifier_seed"]) + epoch,
            f"{task_id}: epoch shuffle seed mismatch",
        )
    require(
        int(metrics.get("best_epoch", -1))
        == int(metrics["history"][int(metrics["best_epoch"]) - 1]["epoch"]),
        f"{task_id}: invalid best epoch",
    )

    best = source_audit.torch_load(root / "classifier_best.pt")
    last = source_audit.torch_load(root / "classifier_last.pt")
    validate_checkpoint(
        best,
        task_id=task_id,
        config=config,
        source_cache_sha256=fold["source_cache_sha256"],
        classifier_seed=fold["classifier_seed"],
        classifier_config=classifier_config,
        variant_name=name,
        kind="best",
    )
    validate_checkpoint(
        last,
        task_id=task_id,
        config=config,
        source_cache_sha256=fold["source_cache_sha256"],
        classifier_seed=fold["classifier_seed"],
        classifier_config=classifier_config,
        variant_name=name,
        kind="last",
    )
    require(
        int(best.get("best_epoch", -1)) == int(metrics["best_epoch"])
        and close_enough(
            best.get("best_validation_auprc"),
            metrics.get("best_validation_auprc"),
            tolerance,
        ),
        f"{task_id}: best checkpoint selection mismatch",
    )
    return {
        "metrics": {**metrics, **metadata},
        "test": test,
        "validation": validation,
    }


def validate_shared_cell_support(
    subject: str,
    cells: Mapping[str, Mapping[str, Any]],
) -> None:
    if not cells:
        return
    first = next(iter(cells.values()))
    for name, evidence in cells.items():
        for split in ("test", "validation"):
            require(
                np.array_equal(
                    evidence[split]["window_index"],
                    first[split]["window_index"],
                )
                and np.array_equal(
                    evidence[split]["y_true"],
                    first[split]["y_true"],
                ),
                f"{subject}/{name}: variants do not share {split} support",
            )


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
            "wins": 0,
            "ties": 0,
            "losses": 0,
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
    tolerance = 1e-12
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_paired_subjects": int(len(values)),
        "bootstrap_samples": int(samples),
        "wins": int((values > tolerance).sum()),
        "ties": int((np.abs(values) <= tolerance).sum()),
        "losses": int((values < -tolerance).sum()),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"missing root summary {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_number(value: str | None) -> Any:
    if value is None or value.strip() == "":
        return None
    return float(value)


def aggregate_evidence(
    config: Mapping[str, Any],
    variants: list[dict[str, Any]],
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    groups: dict[str, dict[str, Any]] = {}
    for variant in variants:
        name = str(variant["variant"])
        subjects = [
            subject
            for subject in EXPECTED_SUBJECTS
            if (subject, name) in cells
        ]
        evidence = [cells[(subject, name)] for subject in subjects]
        metrics = [item["metrics"] for item in evidence]
        macro = (
            aggregate_fold_metrics(metrics, list(CLASSIFICATION_METRICS))
            if metrics
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in CLASSIFICATION_METRICS
            }
        )
        pooled = (
            rf.prediction_metrics(
                np.concatenate([item["test"]["y_true"] for item in evidence]),
                np.concatenate([item["test"]["y_prob"] for item in evidence]),
                np.concatenate([item["test"]["y_pred"] for item in evidence]),
            )
            if evidence
            else None
        )
        groups[name] = {
            "variant": variant,
            "subjects": subjects,
            "macro": macro,
            "pooled": pooled,
        }
    comparisons: list[dict[str, Any]] = []
    for comparison in config["comparisons"]:
        new_name = str(comparison["new"])
        reference_name = str(comparison["reference"])
        common = [
            subject
            for subject in EXPECTED_SUBJECTS
            if (subject, new_name) in cells
            and (subject, reference_name) in cells
        ]
        differences = np.asarray(
            [
                float(cells[(subject, new_name)]["metrics"]["pr_auc"])
                - float(cells[(subject, reference_name)]["metrics"]["pr_auc"])
                for subject in common
            ],
            dtype=np.float64,
        )
        effect = paired_bootstrap_mean_ci(
            differences,
            int(config["bootstrap_samples"]),
            stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                str(comparison["comparison_id"]),
            ),
        )
        comparisons.append(
            {
                **comparison,
                "common_subjects": ",".join(common),
                **effect,
                "bootstrap_seed": int(config["bootstrap_seed"]),
            }
        )
    return groups, comparisons


def validate_root_summaries(
    *,
    root: Path,
    config: Mapping[str, Any],
    variants: list[dict[str, Any]],
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    complete_folds: list[str],
    tolerance: float,
) -> None:
    required = (
        "fold_summary.csv",
        "experiment_manifest.csv",
        "aggregate_summary.csv",
        "publication_table.csv",
        "paired_pr_auc_deltas.csv",
        "sensor_efficiency.csv",
        "aggregate_metrics.json",
        "status.json",
        "support_equivalence.json",
    )
    for filename in required:
        require((root / filename).exists(), f"missing root summary {filename}")

    fold_rows = read_csv(root / "fold_summary.csv")
    require(
        len(fold_rows) == len(cells),
        "fold_summary.csv has a stale completed-cell count",
    )
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in fold_rows:
        key = (row.get("test_subject", ""), row.get("variant", ""))
        require(key not in by_key, f"fold_summary.csv duplicate {key}")
        by_key[key] = row
    require(
        set(by_key) == set(cells),
        "fold_summary.csv cell identities differ from audited cells",
    )
    for key, evidence in cells.items():
        row = by_key[key]
        metrics = evidence["metrics"]
        for field in (
            *CLASSIFICATION_METRICS,
            "threshold",
            "best_validation_auprc",
            "pos_weight",
        ):
            assert_close(
                csv_number(row.get(field)),
                metrics.get(field),
                f"fold_summary/{key}/{field}",
                tolerance,
            )
        for field in ("n", "n_normal", "n_fog", "tn", "fp", "fn", "tp"):
            require(
                int(row[field]) == int(metrics[field]),
                f"fold_summary/{key}/{field} mismatch",
            )
        for field in (
            "experiment_id",
            "display_name",
            "sensor_input_sha256",
            "source_residual_cache_sha256",
            "input_support_sha256",
            "initial_state_sha256",
            "common_state_sha256",
            "projection_weight_sha256",
        ):
            require(
                row.get(field) == str(metrics[field]),
                f"fold_summary/{key}/{field} mismatch",
            )

    groups, comparisons = aggregate_evidence(config, variants, cells)
    manifest_rows = read_csv(root / "experiment_manifest.csv")
    require(len(manifest_rows) == 7, "experiment_manifest.csv must have seven rows")
    manifest = {row["variant"]: row for row in manifest_rows}
    require(
        set(manifest) == set(EXPECTED_IMU_VARIANTS),
        "experiment_manifest.csv variant set mismatch",
    )
    for name, group in groups.items():
        row = manifest[name]
        variant = group["variant"]
        expected_status = (
            "complete"
            if group["subjects"] == list(EXPECTED_SUBJECTS)
            else ("partial" if group["subjects"] else "pending")
        )
        require(
            row["experiment_id"] == variant["experiment_id"]
            and row["display_name"] == variant["display_name"]
            and int(row["sensor_count"]) == int(variant["sensor_count"])
            and int(row["n_channels"]) == int(variant["n_channels"])
            and int(row["parameter_count"]) == int(variant["parameter_count"])
            and int(row["expected_folds"]) == 8
            and int(row["completed_folds"]) == len(group["subjects"])
            and row["status"] == expected_status
            and row["completed_subjects"] == ",".join(group["subjects"]),
            f"experiment_manifest/{name}: stale fields",
        )

    aggregate = load_json(root / "aggregate_metrics.json")
    require(
        aggregate.get("suite_version") == SUITE_VERSION
        and aggregate.get("protocol_fingerprint")
        == config["protocol_fingerprint"]
        and aggregate.get("aggregation_unit") == "held_out_subject"
        and aggregate.get("ranking_metric") == "subject_macro_pr_auc_mean",
        "aggregate_metrics.json header mismatch",
    )
    experiments = aggregate.get("experiments")
    require(
        isinstance(experiments, dict)
        and set(experiments)
        == {variant["experiment_id"] for variant in variants},
        "aggregate_metrics.json experiment set mismatch",
    )
    for name, group in groups.items():
        variant = group["variant"]
        saved = experiments[variant["experiment_id"]]
        for field, expected in variant.items():
            require(
                saved.get(field) == expected,
                f"aggregate/{name}/{field}: variant metadata mismatch",
            )
        require(
            saved.get("completed_folds") == group["subjects"],
            f"aggregate/{name}: completed fold list mismatch",
        )
        macro = saved.get("subject_macro")
        require(isinstance(macro, dict), f"aggregate/{name}: missing macro")
        for metric, expected_values in group["macro"].items():
            saved_values = macro.get(metric)
            require(
                isinstance(saved_values, dict)
                and int(saved_values["n_folds"])
                == int(expected_values["n_folds"]),
                f"aggregate/{name}/{metric}: fold count mismatch",
            )
            for statistic in ("mean", "std"):
                assert_close(
                    saved_values.get(statistic),
                    expected_values.get(statistic),
                    f"aggregate/{name}/{metric}/{statistic}",
                    tolerance,
                )
            if expected_values["n_folds"]:
                for statistic in ("min", "max"):
                    assert_close(
                        saved_values.get(statistic),
                        expected_values.get(statistic),
                        f"aggregate/{name}/{metric}/{statistic}",
                        tolerance,
                    )
        require(
            (saved.get("pooled") is None) == (group["pooled"] is None),
            f"aggregate/{name}: pooled presence mismatch",
        )
        if group["pooled"] is not None:
            for metric, expected in group["pooled"].items():
                assert_close(
                    saved["pooled"].get(metric),
                    expected,
                    f"aggregate/{name}/pooled/{metric}",
                    tolerance,
                )
    require(
        aggregate.get("paired_pr_auc_comparisons") == comparisons,
        "aggregate_metrics.json paired bootstrap results mismatch",
    )

    completed_cells = len(cells)
    formal_complete = (
        completed_cells == EXPECTED_CLASSIFIER_CELLS
        and bool(config["reportable"])
    )
    ranked = sorted(
        groups.values(),
        key=lambda group: (
            -float(group["macro"]["pr_auc"]["mean"])
            if group["macro"]["pr_auc"]["mean"] is not None
            else float("inf"),
            str(group["variant"]["variant"]),
        ),
    )
    expected_best = (
        ranked[0]["variant"]["experiment_id"]
        if formal_complete and ranked
        else None
    )
    require(
        aggregate.get("best_experiment") == expected_best,
        "aggregate_metrics.json best experiment mismatch",
    )

    aggregate_rows = read_csv(root / "aggregate_summary.csv")
    require(
        len(aggregate_rows) == 7,
        "aggregate_summary.csv must contain all seven variants",
    )
    for rank, (row, group) in enumerate(zip(aggregate_rows, ranked), start=1):
        name = str(group["variant"]["variant"])
        require(
            int(row["rank"]) == rank
            and row["variant"] == name
            and int(row["completed_folds"]) == len(group["subjects"]),
            f"aggregate_summary/{name}: rank/count mismatch",
        )
        for metric, expected_values in group["macro"].items():
            for statistic in ("mean", "std"):
                assert_close(
                    csv_number(row.get(f"{metric}_{statistic}")),
                    expected_values[statistic],
                    f"aggregate_summary/{name}/{metric}_{statistic}",
                    tolerance,
                )

    paired_rows = read_csv(root / "paired_pr_auc_deltas.csv")
    require(
        len(paired_rows) == len(comparisons),
        "paired_pr_auc_deltas.csv row count mismatch",
    )
    for row, expected in zip(paired_rows, comparisons):
        for field in (
            "comparison_id",
            "new",
            "reference",
            "interpretation",
            "common_subjects",
        ):
            require(
                row.get(field) == str(expected[field]),
                f"paired_pr_auc/{expected['comparison_id']}/{field} mismatch",
            )
        for field in ("mean_delta", "ci_low", "ci_high"):
            assert_close(
                csv_number(row.get(field)),
                expected[field],
                f"paired_pr_auc/{expected['comparison_id']}/{field}",
                tolerance,
            )
        for field in (
            "n_paired_subjects",
            "wins",
            "ties",
            "losses",
            "bootstrap_samples",
            "bootstrap_seed",
        ):
            require(
                int(row[field]) == int(expected[field]),
                f"paired_pr_auc/{expected['comparison_id']}/{field} mismatch",
            )

    efficiency_rows = read_csv(root / "sensor_efficiency.csv")
    require(
        [row["variant"] for row in efficiency_rows]
        == list(EXPECTED_IMU_VARIANTS),
        "sensor_efficiency.csv variant order mismatch",
    )
    for row, variant in zip(efficiency_rows, variants):
        channels = int(variant["n_channels"])
        require(
            int(row["sensor_count"]) == int(variant["sensor_count"])
            and int(row["n_channels"]) == channels
            and int(row["input_values_per_window"]) == channels * 256
            and int(row["float32_input_bytes_per_window"])
            == channels * 256 * 4
            and int(row["input_projection_parameters"])
            == int(config["classifier"]["hidden_channels"]) * channels
            and int(row["total_parameter_count"])
            == int(variant["parameter_count"]),
            f"sensor_efficiency/{variant['variant']}: mismatch",
        )
        assert_close(
            row["input_bandwidth_ratio_vs_all"],
            channels / 9.0,
            f"sensor_efficiency/{variant['variant']}/ratio",
            tolerance,
        )

    publication_rows = read_csv(root / "publication_table.csv")
    require(
        len(publication_rows) == 7
        and [row["IMU combination"] for row in publication_rows]
        == [variant["display_name"] for variant in variants],
        "publication_table.csv variant rows mismatch",
    )
    for row, variant in zip(publication_rows, variants):
        group = groups[str(variant["variant"])]
        require(
            int(row["Sensors"]) == int(variant["sensor_count"])
            and int(row["Channels"]) == int(variant["n_channels"])
            and int(row["Completed folds"]) == len(group["subjects"]),
            f"publication_table/{variant['variant']}: identity mismatch",
        )

    status = load_json(root / "status.json")
    expected_status = (
        "complete"
        if formal_complete
        else (
            "smoke_complete"
            if completed_cells == EXPECTED_CLASSIFIER_CELLS
            else "partial"
        )
    )
    require(
        status
        == {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_classifier_cells": EXPECTED_CLASSIFIER_CELLS,
            "completed_classifier_cells": completed_cells,
            "status": expected_status,
            "reportable": bool(config["reportable"]),
            "best_experiment": expected_best,
        },
        "status.json mismatch",
    )
    support_equivalence = load_json(root / "support_equivalence.json")
    require(
        support_equivalence.get("suite_version") == SUITE_VERSION
        and support_equivalence.get("protocol_fingerprint")
        == config["protocol_fingerprint"]
        and support_equivalence.get("variants")
        == list(EXPECTED_IMU_VARIANTS)
        and support_equivalence.get("completed_support_subjects")
        == complete_folds
        and support_equivalence.get("expected_subjects")
        == list(EXPECTED_SUBJECTS)
        and bool(support_equivalence.get("complete"))
        == (complete_folds == list(EXPECTED_SUBJECTS)),
        "support_equivalence.json mismatch",
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.result_dir.resolve()
    require(root.is_dir(), f"result directory does not exist: {root}")
    require(
        math.isfinite(args.tolerance) and args.tolerance >= 0,
        "--tolerance must be finite and non-negative",
    )
    config = load_json(root / "config.json")
    variants = validate_protocol(config)
    validate_run_manifest(root, config)

    source_root = choose_existing_path(
        configured_path(config, "source_suite_dir"),
        args.source_suite_dir,
        "source suite directory",
    )
    data_root = choose_existing_path(
        configured_path(config, "data_dir"),
        args.data_dir,
        "processed Daphnet data directory",
    )
    source_config = load_json(source_root / "config.json")
    require(
        source_config.get("suite_version") == SOURCE_SUITE_VERSION,
        "source suite version mismatch",
    )
    source_support, source_hashes = source_audit.validate_source_suite(
        source_root,
        config,
        list(EXPECTED_SUBJECTS),
    )
    dataset, windows = source_audit.load_dataset_and_windows(
        data_root,
        source_root,
        config,
    )
    require(
        tuple(dataset.channel_names) == EXPECTED_CHANNELS,
        "processed data channel order mismatch",
    )
    environment_path = root / "environment.json"
    require(environment_path.exists(), "missing environment.json")
    load_json(environment_path)

    folds: dict[str, dict[str, Any]] = {}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    failures: list[str] = []
    for subject in EXPECTED_SUBJECTS:
        fold_root = root / f"loso_{subject}"
        cell_done_paths = [
            fold_root / name / "DONE.json" for name in EXPECTED_IMU_VARIANTS
        ]
        if not (fold_root / "fold_config.json").exists() and not any(
            path.exists() for path in cell_done_paths
        ):
            missing.extend(
                f"{subject}/{name}" for name in EXPECTED_IMU_VARIANTS
            )
            continue
        try:
            fold = validate_fold(
                result_root=root,
                source_root=source_root,
                config=config,
                source_config=source_config,
                subject=subject,
                variants=variants,
                dataset=dataset,
                windows=windows,
            )
            require(
                fold["source_cache_sha256"] == source_hashes[subject],
                f"{subject}: independently loaded source hash mismatch",
            )
            for split in ("validation", "test"):
                require(
                    np.array_equal(
                        fold["support"][split]["anchor"],
                        source_support[subject][f"{split}_window_index"],
                    )
                    and np.array_equal(
                        fold["support"][split]["y"],
                        source_support[subject][f"{split}_y_true"],
                    ),
                    f"{subject}/{split}: support differs from source classifier",
                )
            folds[subject] = fold
        except Exception as error:
            failures.append(
                f"{subject}/fold: {type(error).__name__}: {error}"
            )
            continue

        per_fold: dict[str, dict[str, Any]] = {}
        for variant in variants:
            name = str(variant["variant"])
            done_path = fold_root / name / "DONE.json"
            metadata_done = fold_root / name / "IMU_METADATA_DONE.json"
            if not done_path.exists() or not metadata_done.exists():
                missing.append(f"{subject}/{name}")
                continue
            try:
                evidence = audit_cell(
                    result_root=root,
                    config=config,
                    subject=subject,
                    variant=variant,
                    fold=fold,
                    dataset=dataset,
                    windows=windows,
                    tolerance=float(args.tolerance),
                )
                cells[(subject, name)] = evidence
                per_fold[name] = evidence
            except Exception as error:
                failures.append(
                    f"{subject}/{name}: {type(error).__name__}: {error}"
                )
        try:
            validate_shared_cell_support(subject, per_fold)
        except Exception as error:
            failures.append(
                f"{subject}/shared_support: {type(error).__name__}: {error}"
            )

    if missing and not args.allow_partial:
        failures.extend(f"missing {task_id}" for task_id in missing)
    complete_folds = [
        subject for subject in EXPECTED_SUBJECTS if subject in folds
    ]
    try:
        validate_root_summaries(
            root=root,
            config=config,
            variants=variants,
            cells=cells,
            complete_folds=complete_folds,
            tolerance=float(args.tolerance),
        )
    except Exception as error:
        failures.append(
            f"root summaries: {type(error).__name__}: {error}"
        )

    full_complete = (
        bool(config.get("reportable"))
        and config.get("run_kind") == "formal"
        and len(folds) == len(EXPECTED_SUBJECTS)
        and len(cells) == EXPECTED_CLASSIFIER_CELLS
        and not missing
        and not failures
    )
    status = (
        "pass"
        if not failures and (args.allow_partial or full_complete)
        else "fail"
    )
    return {
        "audit_version": AUDIT_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "source_suite_dir": str(source_root),
        "data_dir": str(data_root),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_variants": list(EXPECTED_IMU_VARIANTS),
        "expected_subjects": list(EXPECTED_SUBJECTS),
        "expected_cells": EXPECTED_CLASSIFIER_CELLS,
        "checked_cells": len(cells),
        "checked_folds": complete_folds,
        "missing_cells": missing,
        "allow_partial": bool(args.allow_partial),
        "reportable": bool(config.get("reportable")),
        "full_complete": bool(full_complete),
        "failures": failures,
        "warnings": [],
        "status": status,
    }


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"Audit version: {report.get('audit_version')}",
        f"Status: {report.get('status')}",
        f"Checked cells: {report.get('checked_cells')}/{report.get('expected_cells')}",
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
            "expected_cells": EXPECTED_CLASSIFIER_CELLS,
            "checked_cells": 0,
            "missing_cells": [],
            "allow_partial": bool(args.allow_partial),
            "reportable": False,
            "full_complete": False,
            "failures": [f"fatal: {type(error).__name__}: {error}"],
            "warnings": [],
            "status": "fail",
        }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "AUDIT_REPORT.json"
    text_path = root / "AUDIT_REPORT.txt"
    atomic_json_dump(report, report_path)
    write_text_report(text_path, report)
    complete_path = root / "SUITE_COMPLETE.json"
    if report.get("status") == "pass" and report.get("full_complete"):
        atomic_json_dump(
            {
                "format_version": 1,
                "suite_version": SUITE_VERSION,
                "audit_version": AUDIT_VERSION,
                "status": "complete",
                "protocol_fingerprint": report["protocol_fingerprint"],
                "expected_cells": EXPECTED_CLASSIFIER_CELLS,
                "checked_cells": EXPECTED_CLASSIFIER_CELLS,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "audit_report_sha256": sha256_file(report_path),
            },
            complete_path,
        )
    elif complete_path.exists():
        # A stale success marker must never survive a failed or partial audit.
        complete_path.unlink()

    print(
        f"[imu-audit] status={report['status']} "
        f"checked={report.get('checked_cells', 0)}/"
        f"{report.get('expected_cells', EXPECTED_CLASSIFIER_CELLS)} "
        f"missing={len(report.get('missing_cells', []))} "
        f"failures={len(report.get('failures', []))}",
        flush=True,
    )
    if report.get("status") != "pass":
        for failure in report.get("failures", []):
            print(f"[imu-audit] FAIL {failure}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
