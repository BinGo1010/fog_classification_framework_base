#!/usr/bin/env python
"""Deterministic all-five-IMU TF+SVM experiment for processed_NBM_Exp.

Each subject is modelled independently. Roles 6/7 fit the scaler and SVM,
roles 2/3 select hyperparameters and the operating threshold, and roles 0/1
are used only for final window- and event-level metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


WINDOW_METRICS = ("sensitivity", "precision", "specificity", "pr_auc")
EVENT_METRICS = ("event_sensitivity", "false_alarms_per_hour")
ALL_METRICS = WINDOW_METRICS + EVENT_METRICS
FEATURES_PER_CHANNEL = (
    "std",
    "peak_to_peak",
    "mean_abs",
    "diff_rms",
    "line_length",
    "log_power_0p5_3hz",
    "log_power_3_8hz",
    "log_power_8_28hz",
    "log_freeze_index",
    "spectral_entropy_0p5_28hz",
    "dominant_frequency_0p5_28hz",
)
FEATURES_PER_CHANNEL_4F = (
    "std",
    "peak_to_peak",
    "log_power_3_8hz",
    "log_power_8_28hz",
)
FEATURES_PER_CHANNEL_6F = (
    "std",
    "peak_to_peak",
    "mean_abs",
    "log_power_3_8hz",
    "log_power_8_28hz",
    "log_freeze_index",
)
FEATURE_SCHEMAS = {
    "tf330_all5_30ch_v1": FEATURES_PER_CHANNEL,
    "tf120_all5_30ch_4f_v1": FEATURES_PER_CHANNEL_4F,
    "tf180_all5_30ch_6f_v1": FEATURES_PER_CHANNEL_6F,
}
FEATURE_SCHEMA_VERSIONS = {
    "tf330_all5_30ch_v1": 1,
    "tf120_all5_30ch_4f_v1": 2,
    "tf180_all5_30ch_6f_v1": 3,
}
LEGACY_TF330_IMPLEMENTATION_SHA256 = (
    "97107f3911ad7fa292247cd013fcffce18500d394852838f62c84b69eccc8f20"
)
LEGACY_TF120_F1_IMPLEMENTATION_SHA256 = (
    "fd186e36b29de9d2472e79347751bdfc9110b0d4f679ecd2ef8564077a07d693"
)
LEGACY_TF120_SENSITIVITY_IMPLEMENTATION_SHA256 = (
    "3aeb1cd8d2ca597fd47d63a6792ffae11fd58044243176a31a3423098993795f"
)
LEGACY_TF120_MAXPRECISION_IMPLEMENTATION_SHA256 = (
    "0af51f2bf563f4bc117a3836db256a76f14a749593ff8c5c77dbac8526a23c4f"
)
LEGACY_TF120_RECORDMERGE_IMPLEMENTATION_SHA256 = (
    "17cb0abadba2fadb825da8de9473a348eebec79f41d19ef92ea5e6bc7c19179b"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _temporary_path(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=f".{suffix}.tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    return Path(name)


def atomic_json_dump(value: Any, path: Path) -> None:
    temporary = _temporary_path(path, "json")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv_write(
    frame: pd.DataFrame, path: Path, *, float_format: str | None = None
) -> None:
    temporary = _temporary_path(path, "csv")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format=float_format)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text_write(text: str, path: Path) -> None:
    temporary = _temporary_path(path, "text")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_joblib_dump(value: Any, path: Path) -> None:
    temporary = _temporary_path(path, "joblib")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    project_value = Path(str(config.get("project_root", "..")))
    project_root = (
        project_value.resolve()
        if project_value.is_absolute()
        else (config_path.parent / project_value).resolve()
    )
    config["_config_path"] = str(config_path)
    return config, project_root


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def feature_names(
    channel_names: Iterable[str], feature_schema: str = "tf330_all5_30ch_v1"
) -> list[str]:
    if feature_schema not in FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported private TF feature schema: {feature_schema}")
    return [
        f"{channel}__{feature}"
        for channel in channel_names
        for feature in FEATURE_SCHEMAS[feature_schema]
    ]


def _band_power(
    power: np.ndarray,
    frequencies: np.ndarray,
    low: float,
    high: float,
    *,
    include_high: bool,
) -> np.ndarray:
    mask = (frequencies >= low) & (
        frequencies <= high if include_high else frequencies < high
    )
    if not mask.any():
        raise ValueError(f"No FFT bins in band [{low}, {high}]")
    resolution = float(frequencies[1] - frequencies[0])
    return power[:, mask, :].sum(axis=1) * resolution


def extract_tf_features(
    windows: np.ndarray,
    *,
    sampling_rate_hz: float = 64.0,
    high_band_hz: float = 28.0,
    remove_channel_mean: bool = True,
    epsilon: float = 1e-12,
    feature_schema: str = "tf330_all5_30ch_v1",
) -> np.ndarray:
    """Return configured features per physical channel for 2-second windows."""

    values = np.asarray(windows, dtype=np.float64)
    expected_samples = int(round(2.0 * float(sampling_rate_hz)))
    if values.ndim != 3 or values.shape[1] != expected_samples:
        raise ValueError(
            f"windows must have shape [n, {expected_samples}, channels], got {values.shape}"
        )
    if len(values) == 0 or values.shape[2] == 0 or not np.isfinite(values).all():
        raise ValueError("windows must be non-empty and finite")
    if high_band_hz > sampling_rate_hz / 2.0:
        raise ValueError("high-band limit exceeds Nyquist")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if feature_schema not in FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported private TF feature schema: {feature_schema}")

    centered = (
        values - values.mean(axis=1, keepdims=True)
        if remove_channel_mean
        else values
    )
    time_features = [
        centered.std(axis=1),
        np.ptp(centered, axis=1),
    ]
    if feature_schema == "tf180_all5_30ch_6f_v1":
        time_features.append(np.mean(np.abs(centered), axis=1))
    if feature_schema == "tf330_all5_30ch_v1":
        differences = np.diff(centered, axis=1)
        time_features.extend(
            [
                np.mean(np.abs(centered), axis=1),
                np.sqrt(np.mean(np.square(differences), axis=1)),
                np.mean(np.abs(differences), axis=1),
            ]
        )

    taper = np.hanning(values.shape[1]).reshape(1, -1, 1)
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / sampling_rate_hz)
    spectrum = np.fft.rfft(centered * taper, axis=1)
    power = np.square(np.abs(spectrum)) / max(float(np.square(taper).sum()), epsilon)
    freezing = _band_power(power, frequencies, 3.0, 8.0, include_high=False)
    high = _band_power(
        power, frequencies, 8.0, high_band_hz, include_high=True
    )
    frequency_features = [
        np.log1p(freezing),
        np.log1p(high),
    ]
    if feature_schema in {"tf330_all5_30ch_v1", "tf180_all5_30ch_6f_v1"}:
        locomotor = _band_power(
            power, frequencies, 0.5, 3.0, include_high=False
        )
        log_freeze_index = np.log1p(freezing / np.maximum(locomotor, epsilon))
    if feature_schema == "tf180_all5_30ch_6f_v1":
        frequency_features.append(log_freeze_index)
    if feature_schema == "tf330_all5_30ch_v1":
        analysis_mask = (frequencies >= 0.5) & (frequencies <= high_band_hz)
        analysis = power[:, analysis_mask, :]
        normalized = analysis / np.maximum(
            analysis.sum(axis=1, keepdims=True), epsilon
        )
        entropy = -np.sum(
            normalized * np.log(np.maximum(normalized, epsilon)), axis=1
        ) / np.log(float(analysis.shape[1]))
        analysis_frequencies = frequencies[analysis_mask]
        dominant = analysis_frequencies[np.argmax(analysis, axis=1)]
        frequency_features = [
            np.log1p(locomotor),
            *frequency_features,
            log_freeze_index,
            entropy,
            dominant,
        ]
    matrix = np.stack([*time_features, *frequency_features], axis=2)
    result = matrix.reshape(len(values), -1)
    if (
        result.shape[1] != len(feature_names(range(values.shape[2]), feature_schema))
        or not np.isfinite(result).all()
    ):
        raise RuntimeError("Feature extraction produced NaN or Inf")
    return result.astype(np.float32)


def load_index_frame(root: Path, subject: str, fold: int) -> pd.DataFrame:
    path = root / "split_indices" / f"{subject}_outer{fold}_nbm_indices.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    required = {
        "record_id",
        "start_index",
        "end_index_exclusive",
        "role_code",
        "y_binary",
        "allocation_group_id",
        "window_id",
    }
    optional = {
        "connector_id",
        "left_group_id",
        "right_group_id",
        "is_dynamic_connector",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        data = {
            name: payload[name]
            for name in required | (optional & set(payload.files))
        }
    frame = pd.DataFrame(data)
    for column in (
        "record_id",
        "allocation_group_id",
        "window_id",
        "connector_id",
        "left_group_id",
        "right_group_id",
    ):
        if column in frame:
            frame[column] = frame[column].astype(str)
    frame["start_index"] = frame["start_index"].astype(np.int64)
    frame["end_index_exclusive"] = frame["end_index_exclusive"].astype(np.int64)
    frame["role_code"] = frame["role_code"].astype(np.int8)
    frame["y_binary"] = frame["y_binary"].astype(np.int8)
    frame["subject_id"] = str(subject)
    frame["outer_fold"] = int(fold)
    frame["start_time_sec"] = frame["start_index"] / 64.0
    frame["end_time_sec"] = frame["end_index_exclusive"] / 64.0
    if frame["window_id"].duplicated().any():
        raise ValueError(f"Duplicate window IDs in {path}")
    return frame.sort_values(
        ["record_id", "start_index", "role_code"], kind="stable"
    ).reset_index(drop=True)


def validate_job_frame(frame: pd.DataFrame, subject: str, fold: int) -> None:
    expected = {0: 0, 1: 1, 2: 0, 3: 1, 6: 0, 7: 1}
    for role, label in expected.items():
        selected = frame.loc[frame["role_code"] == role]
        if selected.empty or set(selected["y_binary"].unique()) != {label}:
            raise ValueError(
                f"{subject} outer{fold} role {role} is empty or not pure label {label}"
            )


def role_frame(frame: pd.DataFrame, roles: Iterable[int]) -> pd.DataFrame:
    selected = frame.loc[frame["role_code"].isin([int(role) for role in roles])]
    if selected.empty:
        raise ValueError(f"No windows for roles {list(roles)}")
    return selected.reset_index(drop=True)


def load_records(root: Path, frame: pd.DataFrame, channel_count: int) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    for record_id in sorted(frame["record_id"].unique()):
        path = root / "records" / f"{record_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["x"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != channel_count:
            raise ValueError(f"Unexpected record shape in {path}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite signal in {path}")
        records[str(record_id)] = values
    return records


def windows_from_frame(
    frame: pd.DataFrame,
    records: dict[str, np.ndarray],
    channel_indices: tuple[int, ...],
    window_samples: int,
) -> np.ndarray:
    windows = [
        records[str(row.record_id)][
            int(row.start_index) : int(row.end_index_exclusive), channel_indices
        ]
        for row in frame.itertuples(index=False)
    ]
    result = np.stack(windows).astype(np.float32, copy=False)
    if result.shape[1:] != (window_samples, len(channel_indices)):
        raise ValueError(f"Unexpected window matrix: {result.shape}")
    return result


def maximum_fog_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(truth) != len(score) or np.unique(truth).tolist() != [0, 1]:
        raise ValueError("Threshold selection requires aligned binary validation data")
    if not np.isfinite(score).all():
        raise ValueError("Validation score contains NaN or Inf")
    candidates = np.r_[np.unique(score), np.nextafter(score.max(), np.inf)]
    selected: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in candidates:
        prediction = score >= threshold
        tp = int(np.sum((truth == 1) & prediction))
        tn = int(np.sum((truth == 0) & (~prediction)))
        fp = int(np.sum((truth == 0) & prediction))
        fn = int(np.sum((truth == 1) & (~prediction)))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        fog_f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity > 0.0
            else 0.0
        )
        key = (float(fog_f1), float(specificity), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            selected = {
                "threshold": float(threshold),
                "fog_f1": float(fog_f1),
                "sensitivity": float(sensitivity),
                "precision": float(precision),
                "specificity": float(specificity),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "rule": "maximum_validation_fog_f1",
                "tie_break": ["higher_specificity", "larger_threshold"],
                "n_samples": int(len(truth)),
                "n_threshold_candidates": int(len(candidates)),
            }
    if selected is None:
        raise RuntimeError("Threshold selection produced no candidate")
    return selected


def target_sensitivity_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float
) -> dict[str, Any]:
    """Select the highest validation threshold attaining the sensitivity target."""

    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    target = float(target_sensitivity)
    if len(truth) != len(score) or np.unique(truth).tolist() != [0, 1]:
        raise ValueError("Threshold selection requires aligned binary validation data")
    if not np.isfinite(score).all():
        raise ValueError("Validation score contains NaN or Inf")
    if not np.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("Target sensitivity must be in (0, 1]")

    candidates = np.unique(score)
    selected: dict[str, Any] | None = None
    for threshold in candidates[::-1]:
        prediction = score >= threshold
        tp = int(np.sum((truth == 1) & prediction))
        fn = int(np.sum((truth == 1) & (~prediction)))
        sensitivity = tp / (tp + fn)
        if sensitivity + 1e-12 < target:
            continue
        tn = int(np.sum((truth == 0) & (~prediction)))
        fp = int(np.sum((truth == 0) & prediction))
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        fog_f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity > 0.0
            else 0.0
        )
        selected = {
            "threshold": float(threshold),
            "fog_f1": float(fog_f1),
            "sensitivity": float(sensitivity),
            "precision": float(precision),
            "specificity": float(specificity),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "rule": "validation_target_sensitivity",
            "target_sensitivity": target,
            "tie_break": ["largest_threshold"],
            "n_samples": int(len(truth)),
            "n_threshold_candidates": int(len(candidates)),
        }
        break
    if selected is None:
        raise RuntimeError("No validation threshold attains the sensitivity target")
    return selected


def maximum_precision_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, Any]:
    """Maximize validation precision, then sensitivity, specificity and threshold."""

    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(truth) != len(score) or np.unique(truth).tolist() != [0, 1]:
        raise ValueError("Threshold selection requires aligned binary validation data")
    if not np.isfinite(score).all():
        raise ValueError("Validation score contains NaN or Inf")

    candidates = np.unique(score)
    selected: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        prediction = score >= threshold
        tp = int(np.sum((truth == 1) & prediction))
        tn = int(np.sum((truth == 0) & (~prediction)))
        fp = int(np.sum((truth == 0) & prediction))
        fn = int(np.sum((truth == 1) & (~prediction)))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        fog_f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity > 0.0
            else 0.0
        )
        key = (
            float(precision),
            float(sensitivity),
            float(specificity),
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            selected = {
                "threshold": float(threshold),
                "fog_f1": float(fog_f1),
                "sensitivity": float(sensitivity),
                "precision": float(precision),
                "specificity": float(specificity),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "rule": "maximum_validation_precision",
                "tie_break": [
                    "higher_sensitivity",
                    "higher_specificity",
                    "larger_threshold",
                ],
                "n_samples": int(len(truth)),
                "n_threshold_candidates": int(len(candidates)),
            }
    if selected is None:
        raise RuntimeError("Threshold selection produced no candidate")
    return selected


def select_threshold(
    config: dict[str, Any], y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    rule = evaluation.get("threshold_rule")
    if rule == "validation_max_fog_f1":
        return maximum_fog_f1_threshold(y_true, y_score)
    if rule == "validation_target_sensitivity":
        return target_sensitivity_threshold(
            y_true,
            y_score,
            float(evaluation.get("target_sensitivity", float("nan"))),
        )
    if rule == "validation_max_precision":
        return maximum_precision_threshold(y_true, y_score)
    raise ValueError(f"Unsupported threshold rule: {rule}")


def compute_window_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    prediction = (score >= float(threshold)).astype(np.int8)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    both = np.unique(truth).size == 2
    return {
        "threshold": float(threshold),
        "n_windows": int(len(truth)),
        "positive_count": int(np.sum(truth == 1)),
        "negative_count": int(np.sum(truth == 0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "specificity": float(specificity),
        "pr_auc": float(average_precision_score(truth, score)) if both else float("nan"),
        "auroc_audit": float(roc_auc_score(truth, score)) if both else float("nan"),
    }


def _rank_score(decision: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(decision, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def validate_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    data = config.get("data", {})
    for key, expected in {
        "train_roles": [6, 7],
        "validation_roles": [2, 3],
        "test_roles": [0, 1],
    }.items():
        if [int(item) for item in data.get(key, [])] != expected:
            raise ValueError(f"data.{key} must be frozen as {expected}")
    root = resolve_path(project_root, data["root"])
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    quality = json.loads((root / "nbm_quality_report.json").read_text(encoding="utf-8"))
    if not bool(quality.get("overall_pass")):
        raise ValueError("processed_NBM_Exp quality report is not PASS")
    channel_names = tuple(str(item["name"]) for item in schema["channels"])
    sensors = tuple(str(item) for item in schema["selected_sensors"])
    sensor_config = data.get("sensor_config", {})
    indices = tuple(int(item) for item in sensor_config.get("channel_indices", []))
    if schema.get("sampling_rate_hz") != 64 or len(channel_names) != 30:
        raise ValueError("Private all5 protocol requires 64 Hz and 30 channels")
    if tuple(sensor_config.get("sensors", [])) != sensors or indices != tuple(range(30)):
        raise ValueError("sensor_config must use all five IMUs and all 30 channels")
    features = config.get("features", {})
    feature_schema = str(features.get("schema", ""))
    if (
        feature_schema not in FEATURE_SCHEMAS
        or int(features.get("sampling_rate_hz", 0)) != 64
        or int(features.get("window_samples", 0)) != 128
    ):
        raise ValueError("Feature schema must be registered at 64 Hz/128 samples")
    evaluation = config.get("evaluation", {})
    if tuple(evaluation.get("e1_e3_window_metrics", ())) != WINDOW_METRICS:
        raise ValueError("E1/E3 metric contract is not frozen")
    if tuple(evaluation.get("e4_metrics", ())) != ALL_METRICS:
        raise ValueError("E4 metric contract is not frozen")
    if evaluation.get("model_selection") != "validation_pr_auc":
        raise ValueError("Model selection must use validation PR-AUC")
    threshold_rule = evaluation.get("threshold_rule")
    if threshold_rule not in {
        "validation_max_fog_f1",
        "validation_target_sensitivity",
        "validation_max_precision",
    }:
        raise ValueError("Unsupported validation threshold rule")
    if threshold_rule == "validation_target_sensitivity":
        target = float(evaluation.get("target_sensitivity", float("nan")))
        if not np.isfinite(target) or not 0.0 < target <= 1.0:
            raise ValueError("evaluation.target_sensitivity must be in (0, 1]")
    merge_scope = str(
        evaluation.get("event", {}).get(
            "false_alarm_merge_scope", "record_and_allocation_group"
        )
    )
    if merge_scope not in {"record_id", "record_and_allocation_group"}:
        raise ValueError("Unsupported false-alarm merge scope")
    model = config.get("model", {})
    if model.get("kernel") != "rbf" or model.get("class_weight") != "balanced":
        raise ValueError("Model must be balanced RBF-SVC")
    c_grid = [float(value) for value in model.get("grid", {}).get("C", [])]
    gamma_grid = list(model.get("grid", {}).get("gamma", []))
    if not c_grid or not gamma_grid:
        raise ValueError("SVM grid must not be empty")
    return {
        "root": root,
        "subjects": [str(item) for item in data.get("subjects", [])],
        "folds": [int(item) for item in data.get("outer_folds", [0, 1, 2])],
        "channel_names": channel_names,
        "channel_indices": indices,
        "sensors": sensors,
        "C": c_grid,
        "gamma": gamma_grid,
        "feature_schema": feature_schema,
    }


def audit_dataset(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    registered = validate_config(config, project_root)
    counts: list[dict[str, Any]] = []
    for subject in registered["subjects"]:
        for fold in registered["folds"]:
            frame = load_index_frame(registered["root"], subject, fold)
            validate_job_frame(frame, subject, fold)
            role_counts = frame["role_code"].value_counts()
            counts.append(
                {
                    "subject_id": subject,
                    "outer_fold": fold,
                    **{
                        f"role_{role}": int(role_counts.get(role, 0))
                        for role in (0, 1, 2, 3, 6, 7)
                    },
                }
            )
    events = pd.read_csv(registered["root"] / "nbm_fog_event_manifest.csv")
    eligible = int((events["nbm_status"].astype(str) == "eligible").sum())
    return {
        "status": "PASS",
        "dataset_root": str(registered["root"]),
        "subjects": registered["subjects"],
        "outer_folds": registered["folds"],
        "sensors": registered["sensors"],
        "channel_count": len(registered["channel_names"]),
        "feature_schema": registered["feature_schema"],
        "feature_count": len(
            feature_names(registered["channel_names"], registered["feature_schema"])
        ),
        "eligible_annotated_events_before_test_partition": eligible,
        "job_count": len(counts),
        "role_counts": counts,
    }


def _make_pipeline(config: dict[str, Any], c_value: float, gamma: Any) -> Pipeline:
    gamma_value: str | float = (
        str(gamma) if str(gamma) in {"scale", "auto"} else float(gamma)
    )
    model = config["model"]
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "svm",
                SVC(
                    C=float(c_value),
                    gamma=gamma_value,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    cache_size=float(model.get("cache_size_mb", 1024)),
                    tol=float(model.get("tolerance", 1e-3)),
                ),
            ),
        ]
    )


def select_model(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    grid = config["model"]["grid"]
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_model: Pipeline | None = None
    best_selection: dict[str, Any] | None = None
    for index, (c_value, gamma) in enumerate(product(grid["C"], grid["gamma"])):
        model = _make_pipeline(config, float(c_value), gamma)
        model.fit(x_train, y_train)
        score = _rank_score(model.decision_function(x_validation))
        threshold_info = select_threshold(config, y_validation, score)
        metrics = compute_window_metrics(
            y_validation, score, float(threshold_info["threshold"])
        )
        row = {
            "candidate_index": index,
            "C": float(c_value),
            "gamma": gamma,
            "validation_pr_auc": metrics["pr_auc"],
            "validation_sensitivity": metrics["sensitivity"],
            "validation_precision": metrics["precision"],
            "validation_specificity": metrics["specificity"],
            "validation_fog_f1": threshold_info["fog_f1"],
            "threshold": threshold_info["threshold"],
            "selected": False,
        }
        rows.append(row)
        key = (
            float(metrics["pr_auc"]),
            float(threshold_info["fog_f1"]),
            float(threshold_info["specificity"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_selection = {
                **threshold_info,
                "C": float(c_value),
                "gamma": gamma,
                "validation_pr_auc": metrics["pr_auc"],
                "selected_candidate_index": index,
                "model_selection": "maximum_validation_pr_auc",
                "model_selection_tie_break": [
                    "higher_validation_fog_f1",
                    "higher_validation_specificity",
                    "earlier_config_grid_candidate",
                ],
                "score_space": "expit_svm_decision_rank_score_not_calibrated_probability",
            }
    if best_model is None or best_selection is None:
        raise RuntimeError("SVM model selection produced no candidate")
    rows[int(best_selection["selected_candidate_index"])]["selected"] = True
    return best_model, best_selection, pd.DataFrame(rows)


def prediction_frame(
    model: Pipeline,
    features: np.ndarray,
    frame: pd.DataFrame,
    threshold: float,
    split: str,
) -> pd.DataFrame:
    decision = np.asarray(model.decision_function(features), dtype=np.float64)
    score = _rank_score(decision)
    result = frame.copy().reset_index(drop=True)
    result["decision_score"] = decision
    result["prob_fog"] = score
    result["threshold"] = float(threshold)
    result["y_pred"] = (score >= threshold).astype(np.int8)
    result["split"] = split
    result["sensor_config"] = "all5_30ch"
    result["score_semantics"] = "rank_score_not_calibrated_probability"
    return result


def _interval_union_seconds(frame: pd.DataFrame) -> float:
    total = 0.0
    for _, group in frame.groupby("record_id", sort=False):
        intervals = sorted(
            zip(group["start_time_sec"].astype(float), group["end_time_sec"].astype(float))
        )
        if not intervals:
            continue
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        total += end - start
    return float(total)


def evaluate_events(
    predictions: pd.DataFrame,
    event_manifest: pd.DataFrame,
    *,
    merge_gap_sec: float,
    false_alarm_merge_scope: str = "record_and_allocation_group",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate original annotated events and role-0 false-alarm episodes."""

    subject = str(predictions["subject_id"].iloc[0])
    events = event_manifest.loc[
        (event_manifest["subject_id"].astype(str) == subject)
        & (event_manifest["nbm_status"].astype(str) == "eligible")
    ]
    details: list[dict[str, Any]] = []
    fog_test = predictions.loc[predictions["role_code"] == 1]
    for event in events.itertuples(index=False):
        group_ids = {
            item for item in str(event.nbm_allocation_group_ids).split(";") if item
        }
        connector_ids = {
            item for item in str(event.nbm_connector_window_ids).split(";") if item
        }
        candidates = fog_test.loc[
            fog_test["allocation_group_id"].astype(str).isin(group_ids)
            | fog_test["window_id"].astype(str).isin(connector_ids)
        ]
        if candidates.empty:
            continue
        detected = bool((candidates["y_pred"] == 1).any())
        details.append(
            {
                "subject_id": subject,
                "record_id": str(event.record_id),
                "event_id": int(event.event_id),
                "start_time_sec": float(event.start_time_sec),
                "end_time_sec": float(event.end_time_sec),
                "test_window_count": int(len(candidates)),
                "positive_window_count": int((candidates["y_pred"] == 1).sum()),
                "maximum_prob_fog": float(candidates["prob_fog"].max()),
                "detected": detected,
            }
        )
    event_details = pd.DataFrame(details)
    n_events = int(len(event_details))
    n_detected = int(event_details["detected"].sum()) if n_events else 0

    alarm_rows: list[dict[str, Any]] = []
    positives = predictions.loc[
        (predictions["role_code"] == 0) & (predictions["y_pred"] == 1)
    ]
    if false_alarm_merge_scope == "record_id":
        grouped_positives = positives.groupby("record_id", sort=False)
    elif false_alarm_merge_scope == "record_and_allocation_group":
        grouped_positives = positives.groupby(
            ["record_id", "allocation_group_id"], sort=False
        )
    else:
        raise ValueError("Unsupported false-alarm merge scope")
    for keys, group in grouped_positives:
        if false_alarm_merge_scope == "record_id":
            record_id = str(keys)
            allocation_group_id = None
        else:
            record_id = str(keys[0])
            allocation_group_id = str(keys[1])
        ordered = group.sort_values(["start_time_sec", "end_time_sec"], kind="stable")
        current: dict[str, Any] | None = None
        local_id = 0
        for row in ordered.itertuples(index=False):
            start = float(row.start_time_sec)
            end = float(row.end_time_sec)
            if current is None or start > float(current["end_time_sec"]) + merge_gap_sec:
                if current is not None:
                    alarm_rows.append(current)
                    local_id += 1
                current = {
                    "subject_id": subject,
                    "record_id": record_id,
                    "alarm_id": local_id,
                    "start_time_sec": start,
                    "end_time_sec": end,
                    "positive_window_count": 1,
                }
                if allocation_group_id is None:
                    current["allocation_group_ids"] = str(row.allocation_group_id)
                else:
                    current["allocation_group_id"] = allocation_group_id
            else:
                current["end_time_sec"] = max(float(current["end_time_sec"]), end)
                current["positive_window_count"] = int(current["positive_window_count"]) + 1
                if allocation_group_id is None:
                    group_ids = {
                        item
                        for item in str(current["allocation_group_ids"]).split(";")
                        if item
                    }
                    group_ids.add(str(row.allocation_group_id))
                    current["allocation_group_ids"] = ";".join(sorted(group_ids))
        if current is not None:
            alarm_rows.append(current)
    alarms = pd.DataFrame(alarm_rows)
    nonfog_seconds = _interval_union_seconds(
        predictions.loc[predictions["role_code"] == 0]
    )
    nonfog_hours = nonfog_seconds / 3600.0
    metrics = {
        "event_sensitivity": n_detected / n_events if n_events else float("nan"),
        "n_eligible_test_events": n_events,
        "n_detected_test_events": n_detected,
        "n_missed_test_events": n_events - n_detected,
        "n_false_alarm_episodes": int(len(alarms)),
        "nonfog_exposure_hours": float(nonfog_hours),
        "false_alarms_per_hour": (
            float(len(alarms) / nonfog_hours) if nonfog_hours > 0 else float("nan")
        ),
        "event_detection_rule": "any positive permanent-test pure-FoG window mapped to original annotated event",
        "false_alarm_rule": (
            "merge positive role-0 windows within record when interval gap <= configured seconds"
            if false_alarm_merge_scope == "record_id"
            else "merge positive role-0 windows within record/allocation group when interval gap <= configured seconds"
        ),
        "false_alarm_merge_scope": false_alarm_merge_scope,
        "merge_gap_sec": float(merge_gap_sec),
        "exposure_rule": "union of role-0 window intervals per record",
    }
    return metrics, event_details, alarms


def _job_fingerprint(
    config: dict[str, Any], root: Path, subject: str, fold: int
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    split_path = root / "split_indices" / f"{subject}_outer{fold}_nbm_indices.npz"
    feature_schema = str(config.get("features", {}).get("schema", ""))
    threshold_rule = str(config.get("evaluation", {}).get("threshold_rule", ""))
    merge_scope = str(
        config.get("evaluation", {}).get("event", {}).get(
            "false_alarm_merge_scope", "record_and_allocation_group"
        )
    )
    if feature_schema == "tf330_all5_30ch_v1":
        implementation_sha256 = LEGACY_TF330_IMPLEMENTATION_SHA256
    elif threshold_rule == "validation_max_fog_f1":
        implementation_sha256 = LEGACY_TF120_F1_IMPLEMENTATION_SHA256
    elif threshold_rule == "validation_target_sensitivity":
        implementation_sha256 = LEGACY_TF120_SENSITIVITY_IMPLEMENTATION_SHA256
    elif (
        threshold_rule == "validation_max_precision"
        and merge_scope == "record_and_allocation_group"
    ):
        implementation_sha256 = LEGACY_TF120_MAXPRECISION_IMPLEMENTATION_SHA256
    elif (
        feature_schema == "tf120_all5_30ch_4f_v1"
        and threshold_rule == "validation_max_precision"
        and merge_scope == "record_id"
    ):
        implementation_sha256 = LEGACY_TF120_RECORDMERGE_IMPLEMENTATION_SHA256
    else:
        implementation_sha256 = sha256_file(script_path)
    return {
        "format": "private-nbm-tf-svm-all5",
        "version": (
            6
            if feature_schema == "tf180_all5_30ch_6f_v1"
            else FEATURE_SCHEMA_VERSIONS[feature_schema]
            if threshold_rule == "validation_max_fog_f1"
            else 3
            if threshold_rule == "validation_target_sensitivity"
            else 4
            if merge_scope == "record_and_allocation_group"
            else 5
        ),
        "implementation_sha256": implementation_sha256,
        "configuration_hash": stable_hash(public_config(config)),
        "split_index_sha256": sha256_file(split_path),
        "schema_sha256": sha256_file(root / "schema.json"),
        "subject_id": subject,
        "outer_fold": int(fold),
        "sensor_config": "all5_30ch",
        "feature_schema": feature_schema,
    }


def _mean_sd(values: pd.Series) -> tuple[float, float]:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if not len(numbers):
        return float("nan"), float("nan")
    return (
        float(numbers.mean()),
        float(numbers.std(ddof=1)) if len(numbers) > 1 else float("nan"),
    )


def _format_metric(mean: float, sd: float, lower_is_better: bool = False) -> str:
    if not np.isfinite(mean):
        return "NA"
    text = f"{mean:.4f}" if not np.isfinite(sd) else f"{mean:.4f} ± {sd:.4f}"
    return text + (" ↓" if lower_is_better else "")


def _write_main_table(summary: pd.DataFrame, metrics: tuple[str, ...], path: Path) -> None:
    labels = {
        "sensitivity": "Sensitivity",
        "precision": "Precision",
        "specificity": "Specificity",
        "pr_auc": "PR-AUC",
        "event_sensitivity": "Event Sensitivity",
        "false_alarms_per_hour": "False Alarms/hour",
    }
    headers = ["Model", "Sensors", "Channels", "Subjects"] + [labels[item] for item in metrics]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in summary.to_dict(orient="records"):
        cells = [
            str(row["model"]),
            str(int(row["sensors"])),
            str(int(row["channels"])),
            str(int(row["subjects"])),
        ]
        cells.extend(
            _format_metric(
                float(row[f"{metric}_mean"]),
                float(row[f"{metric}_sd"]),
                lower_is_better=metric == "false_alarms_per_hour",
            )
            for metric in metrics
        )
        lines.append("| " + " | ".join(cells) + " |")
    atomic_text_write("\n".join(lines) + "\n", path)


def run_experiment(
    config: dict[str, Any],
    project_root: Path,
    *,
    subjects: Iterable[str] | None = None,
    folds: Iterable[int] | None = None,
) -> dict[str, Any]:
    registered = validate_config(config, project_root)
    feature_schema = registered["feature_schema"]
    features_per_channel = FEATURE_SCHEMAS[feature_schema]
    selected_subjects = list(subjects or registered["subjects"])
    selected_folds = [int(item) for item in (folds or registered["folds"])]
    if set(selected_subjects) - set(registered["subjects"]):
        raise ValueError("Unknown subject selection")
    if set(selected_folds) - set(registered["folds"]):
        raise ValueError("Unknown fold selection")
    audit_dataset(config, project_root)

    run_name = str(config.get("run", {}).get("name", "private_tf_svm_all5_30ch_v1"))
    full_selection = (
        selected_subjects == registered["subjects"]
        and selected_folds == registered["folds"]
    )
    if not full_selection:
        suffix = "_".join([*selected_subjects, *[f"outer{fold}" for fold in selected_folds]])
        run_name = f"_subset/{run_name}_{suffix}"
    run_root = resolve_path(project_root, config["artifacts"]["run_root"]) / run_name
    report_root = resolve_path(project_root, config["artifacts"]["report_root"]) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(public_config(config), run_root / "config_resolved.json")
    names = feature_names(registered["channel_names"], feature_schema)
    atomic_json_dump(
        {
            "schema": feature_schema,
            "sampling_rate_hz": 64,
            "window_samples": 128,
            "sensors": registered["sensors"],
            "channel_names": registered["channel_names"],
            "features_per_channel": features_per_channel,
            "feature_names": names,
            "feature_count": len(names),
            "frequency_band_boundary_rule": "[0.5,3), [3,8), [8,28] Hz",
            "derived_cross_channel_features": False,
        },
        run_root / "feature_schema.json",
    )
    event_manifest = pd.read_csv(registered["root"] / "nbm_fog_event_manifest.csv")
    merge_gap_sec = float(config["evaluation"]["event"].get("merge_gap_sec", 1.0))
    false_alarm_merge_scope = str(
        config["evaluation"]["event"].get(
            "false_alarm_merge_scope", "record_and_allocation_group"
        )
    )
    epsilon = float(config["features"].get("epsilon", 1e-12))

    rows: list[dict[str, Any]] = []
    for subject in selected_subjects:
        for fold in selected_folds:
            job_dir = run_root / "all5_30ch" / subject / f"outer_{fold}"
            job_dir.mkdir(parents=True, exist_ok=True)
            fingerprint = _job_fingerprint(
                config, registered["root"], subject, fold
            )
            fingerprint_path = job_dir / "run_fingerprint.json"
            metrics_path = job_dir / "test_metrics.json"
            if (job_dir / "DONE").exists():
                if not fingerprint_path.exists() or json.loads(
                    fingerprint_path.read_text(encoding="utf-8")
                ) != fingerprint:
                    raise RuntimeError(f"Completed job fingerprint mismatch: {job_dir}")
                if not metrics_path.exists():
                    raise RuntimeError(f"Completed job is missing metrics: {job_dir}")
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            atomic_json_dump(fingerprint, fingerprint_path)

            frame = load_index_frame(registered["root"], subject, fold)
            validate_job_frame(frame, subject, fold)
            records = load_records(
                registered["root"], frame, len(registered["channel_names"])
            )
            windows = windows_from_frame(
                frame,
                records,
                registered["channel_indices"],
                int(config["features"]["window_samples"]),
            )
            features = extract_tf_features(
                windows,
                sampling_rate_hz=64.0,
                high_band_hz=28.0,
                remove_channel_mean=True,
                epsilon=epsilon,
                feature_schema=feature_schema,
            )
            train_mask = frame["role_code"].isin((6, 7)).to_numpy()
            validation_mask = frame["role_code"].isin((2, 3)).to_numpy()
            test_mask = frame["role_code"].isin((0, 1)).to_numpy()
            train_frame = role_frame(frame, (6, 7))
            validation_frame = role_frame(frame, (2, 3))
            test_frame = role_frame(frame, (0, 1))
            model, selection, search = select_model(
                config,
                features[train_mask],
                train_frame["y_binary"].to_numpy(dtype=np.int8),
                features[validation_mask],
                validation_frame["y_binary"].to_numpy(dtype=np.int8),
            )
            threshold = float(selection["threshold"])
            validation_predictions = prediction_frame(
                model,
                features[validation_mask],
                validation_frame,
                threshold,
                "validation",
            )
            test_predictions = prediction_frame(
                model, features[test_mask], test_frame, threshold, "test"
            )
            validation_metrics = compute_window_metrics(
                validation_predictions["y_binary"],
                validation_predictions["prob_fog"],
                threshold,
            )
            test_metrics = compute_window_metrics(
                test_predictions["y_binary"], test_predictions["prob_fog"], threshold
            )
            event_metrics, event_details, alarms = evaluate_events(
                test_predictions,
                event_manifest,
                merge_gap_sec=merge_gap_sec,
                false_alarm_merge_scope=false_alarm_merge_scope,
            )
            test_metrics.update(event_metrics)
            test_metrics.update(
                {
                    "method": "TF+SVM",
                    "sensor_config": "all5_30ch",
                    "sensor_count": 5,
                    "channel_count": 30,
                    "subject_id": subject,
                    "outer_fold": int(fold),
                    "feature_schema": feature_schema,
                    "feature_count": len(names),
                    "C": float(selection["C"]),
                    "gamma": selection["gamma"],
                    "threshold_rule": config["evaluation"]["threshold_rule"],
                    "model_selection": "validation_pr_auc",
                }
            )
            validation_metrics.update(
                {
                    "method": "TF+SVM",
                    "sensor_config": "all5_30ch",
                    "subject_id": subject,
                    "outer_fold": int(fold),
                }
            )
            atomic_json_dump(selection, job_dir / "selection.json")
            atomic_csv_write(search, job_dir / "hyperparameter_search.csv")
            atomic_csv_write(
                validation_predictions, job_dir / "validation_predictions.csv"
            )
            atomic_csv_write(test_predictions, job_dir / "test_predictions.csv")
            atomic_json_dump(validation_metrics, job_dir / "validation_metrics.json")
            atomic_json_dump(test_metrics, metrics_path)
            atomic_csv_write(event_details, job_dir / "test_event_details.csv")
            atomic_csv_write(alarms, job_dir / "test_false_alarm_episodes.csv")
            atomic_joblib_dump(model, job_dir / "model.joblib")
            atomic_text_write("complete\n", job_dir / "DONE")
            rows.append(test_metrics)

    per_job = pd.DataFrame(rows)
    atomic_csv_write(per_job, report_root / "per_subject_fold_metrics.csv")
    subject_rows: list[dict[str, Any]] = []
    for subject, group in per_job.groupby("subject_id", sort=True):
        row: dict[str, Any] = {
            "subject_id": subject,
            "completed_folds": int(group["outer_fold"].nunique()),
            "expected_folds": len(selected_folds),
        }
        for metric in ALL_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[metric] = float(finite.mean()) if len(finite) else float("nan")
        subject_rows.append(row)
    subject_metrics = pd.DataFrame(subject_rows)
    atomic_csv_write(subject_metrics, report_root / "per_subject_averaged_metrics.csv")

    requested_rows: list[dict[str, Any]] = []
    for subject, group in per_job.groupby("subject_id", sort=True):
        row = {"subject_id": subject, "n_folds": int(group["outer_fold"].nunique())}
        for metric, label in (
            ("pr_auc", "AP"),
            ("event_sensitivity", "Event_Sensitivity"),
            ("false_alarms_per_hour", "False_Alarms_per_hour"),
        ):
            mean, sd = _mean_sd(group[metric])
            row[f"{label}_mean"] = mean
            row[f"{label}_SD"] = sd
        requested_rows.append(row)
    requested_metrics = pd.DataFrame(requested_rows)
    requested_path = report_root / "per_subject_AP_EventSensitivity_FAh_mean_SD.csv"
    atomic_csv_write(requested_metrics, requested_path, float_format="%.4f")

    summary_row: dict[str, Any] = {
        "model": (
            "TF+SVM (all5_30ch)"
            if feature_schema == "tf330_all5_30ch_v1"
            else f"TF+SVM (all5_30ch, {len(features_per_channel)}F/ch)"
        ),
        "sensors": 5,
        "channels": 30,
        "subjects": int(len(subject_metrics)),
        "completed_jobs": int(len(per_job)),
        "expected_jobs": int(len(selected_subjects) * len(selected_folds)),
        "aggregation": "average_outer_folds_within_subject_then_subject_mean_sd",
    }
    for metric in ALL_METRICS:
        mean, sd = _mean_sd(subject_metrics[metric])
        summary_row[f"{metric}_mean"] = mean
        summary_row[f"{metric}_sd"] = sd
    summary = pd.DataFrame([summary_row])
    e1_columns = [
        "model",
        "sensors",
        "channels",
        "subjects",
        "completed_jobs",
        "expected_jobs",
        "aggregation",
        *[
            f"{metric}_{suffix}"
            for metric in WINDOW_METRICS
            for suffix in ("mean", "sd")
        ],
    ]
    atomic_csv_write(summary[e1_columns], report_root / "e1_e3_four_metric_main_table.csv")
    atomic_csv_write(summary, report_root / "e4_six_metric_main_table.csv")
    _write_main_table(
        summary, WINDOW_METRICS, report_root / "e1_e3_four_metric_main_table.md"
    )
    _write_main_table(summary, ALL_METRICS, report_root / "e4_six_metric_main_table.md")

    selected_c = pd.to_numeric(per_job["C"], errors="coerce")
    boundary_count = int((selected_c == max(float(x) for x in config["model"]["grid"]["C"])).sum())
    total_events = int(pd.to_numeric(per_job["n_eligible_test_events"]).sum())
    total_detected = int(pd.to_numeric(per_job["n_detected_test_events"]).sum())
    status = {
        "status": "COMPLETE",
        "method": "TF+SVM",
        "sensor_config": "all5_30ch",
        "sensors": registered["sensors"],
        "channel_count": 30,
        "feature_schema": feature_schema,
        "feature_count": len(names),
        "model_selection": "maximum validation PR-AUC on roles 2/3",
        "threshold_rule": (
            "maximum validation FoG-F1 on roles 2/3"
            if config["evaluation"]["threshold_rule"] == "validation_max_fog_f1"
            else (
                "highest threshold attaining validation sensitivity >= "
                f"{float(config['evaluation']['target_sensitivity']):.4f} on roles 2/3"
            )
            if config["evaluation"]["threshold_rule"]
            == "validation_target_sensitivity"
            else "maximum validation Precision on roles 2/3"
        ),
        "test_roles": [0, 1],
        "subjects": selected_subjects,
        "outer_folds": selected_folds,
        "completed_jobs": int(len(per_job)),
        "selected_maximum_C_job_count": boundary_count,
        "event_evaluations_across_folds": total_events,
        "detected_event_evaluations_across_folds": total_detected,
        "false_alarm_merge_scope": false_alarm_merge_scope,
        "run_root": str(run_root),
        "report_root": str(report_root),
        "e1_e3_table": str(report_root / "e1_e3_four_metric_main_table.md"),
        "e4_table": str(report_root / "e4_six_metric_main_table.md"),
        "per_subject_requested_metrics": str(requested_path),
    }
    atomic_json_dump(status, report_root / "report_summary.json")
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Private processed_NBM_Exp all-five-IMU TF+SVM experiment",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--config", default="configs/private_nbm_tf_svm_all5.yaml")
    run = commands.add_parser("run")
    run.add_argument("--config", default="configs/private_nbm_tf_svm_all5.yaml")
    run.add_argument("--subject", action="append")
    run.add_argument("--fold", action="append", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, project_root = load_config(args.config)
        if args.command == "audit":
            result = audit_dataset(config, project_root)
        else:
            result = run_experiment(
                config,
                project_root,
                subjects=args.subject,
                folds=args.fold,
            )
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
