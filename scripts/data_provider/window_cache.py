from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMNS = ["subjectID", "sessionID", "taskID"]
CLINICAL_NUMERIC_COLUMNS = [
    "age",
    "disease_duration",
    "h_y",
    "updrs_iii",
    "fog_q",
    "moca",
    "fes-i",
    "pdq8",
]
LABEL_NORMAL = 0
LABEL_PRE_FOG = 1
LABEL_FOG_BINARY = 1
LABEL_FOG_THREE = 2
CLASS_NAMES_BY_MODE = {
    "binary": np.array(["NORMAL", "FOG"]),
    "three-class": np.array(["NORMAL", "PRE_FOG", "FOG"]),
}
LONG_TREND_FEATURES = (
    "mean",
    "std",
    "delta",
    "slope",
    "fft_energy",
    "fft_entropy",
    "fft_centroid",
    "fft_peak_freq",
    "bandpower_low",
    "bandpower_high",
    "freeze_index",
    "bandpower_ratio",
    "dominant_power",
)
FREQUENCY_TREND_FEATURES = frozenset(
    {
        "fft_energy",
        "fft_entropy",
        "fft_centroid",
        "fft_peak_freq",
        "bandpower_low",
        "bandpower_high",
        "freeze_index",
        "bandpower_ratio",
        "dominant_power",
    }
)
TREND_MULTI_WINDOW_MODES = {"short_plus_long_trend", "long_trend"}
RAW_MULTI_WINDOW_MODES = {"short_plus_long_raw"}
MULTI_WINDOW_MODES = TREND_MULTI_WINDOW_MODES | RAW_MULTI_WINDOW_MODES


def _mode_int(values, default=0):
    values = np.asarray(values)
    if values.size == 0:
        return int(default)
    values = values.astype(np.int64)
    return int(np.bincount(values).argmax())


def _fog_intervals(fog):
    fog = np.asarray(fog).astype(bool)
    if fog.size == 0 or not fog.any():
        return []
    padded = np.r_[False, fog, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(changes[0::2], changes[1::2])]


def _sample_state(fog, label_mode, sampling_rate_hz, pre_fog_seconds):
    fog = np.asarray(fog).astype(bool)
    state = np.zeros(fog.shape[0], dtype=np.int64)
    if label_mode == "binary":
        state[fog] = LABEL_FOG_BINARY
        return state

    pre_samples = int(round(max(0.0, float(pre_fog_seconds)) * float(sampling_rate_hz)))
    prev_fog_end = 0
    for start, end in _fog_intervals(fog):
        if pre_samples > 0:
            pre_start = max(prev_fog_end, start - pre_samples)
            state[pre_start:start] = LABEL_PRE_FOG
        state[start:end] = LABEL_FOG_THREE
        prev_fog_end = end
    return state


def _window_label(values, label_mode, label_rule, fog_threshold):
    values = np.asarray(values).astype(np.int64)
    if label_rule == "center":
        return int(values[len(values) // 2])

    num_classes = len(CLASS_NAMES_BY_MODE[label_mode])
    if label_rule == "majority":
        return int(np.bincount(values, minlength=num_classes).argmax())

    if label_mode == "binary" and label_rule == "threshold":
        return int(float(np.mean(values == LABEL_FOG_BINARY)) >= float(fog_threshold))

    if label_rule in {"priority", "threshold"}:
        if label_mode == "binary":
            return LABEL_FOG_BINARY if np.any(values == LABEL_FOG_BINARY) else LABEL_NORMAL
        if np.any(values == LABEL_FOG_THREE):
            return LABEL_FOG_THREE
        if np.any(values == LABEL_PRE_FOG):
            return LABEL_PRE_FOG
        return LABEL_NORMAL

    raise ValueError("label_rule must be threshold, priority, center, or majority.")


def _normalize_multi_window_config(value):
    if value in (None, False):
        return {"enabled": False}
    if value is True:
        value = {"enabled": True}
    if not isinstance(value, dict):
        raise ValueError("data.windowing.multi_window must be a mapping or boolean.")

    enabled = bool(value.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    mode = str(value.get("mode", "short_plus_long_trend")).lower()
    if mode not in MULTI_WINDOW_MODES:
        available = ", ".join(sorted(MULTI_WINDOW_MODES))
        raise ValueError(f"data.windowing.multi_window.mode must be one of: {available}.")

    trend_features = []
    if mode in TREND_MULTI_WINDOW_MODES:
        trend_features = value.get("trend_features", list(LONG_TREND_FEATURES))
        if isinstance(trend_features, str):
            trend_features = [part.strip() for part in trend_features.split(",") if part.strip()]
        trend_features = [str(feature).lower() for feature in trend_features]
        unknown = [feature for feature in trend_features if feature not in LONG_TREND_FEATURES]
        if unknown:
            raise ValueError(f"Unknown long-window trend feature(s): {unknown}. Available: {LONG_TREND_FEATURES}")
        if not trend_features:
            raise ValueError("data.windowing.multi_window.trend_features cannot be empty.")

    pad = str(value.get("pad", "edge")).lower()
    if pad not in {"edge", "zero"}:
        raise ValueError("data.windowing.multi_window.pad must be edge or zero.")

    normalized = {
        "enabled": True,
        "mode": mode,
        "long_window_size": int(value.get("long_window_size", value.get("window_size", 240))),
        "trend_features": trend_features,
        "pad": pad,
    }
    if normalized["long_window_size"] <= 0:
        raise ValueError("data.windowing.multi_window.long_window_size must be positive.")
    return normalized


def _slice_with_left_pad(values, start, end, pad="edge"):
    values = np.asarray(values)
    if end <= start:
        raise ValueError("Invalid window slice: end must be greater than start.")
    left_pad = max(0, -int(start))
    right_end = min(int(end), len(values))
    body = values[max(0, int(start)):right_end]
    if left_pad:
        if len(body) and pad == "edge":
            pad_values = np.repeat(body[:1], left_pad, axis=0)
        else:
            pad_values = np.zeros((left_pad,) + values.shape[1:], dtype=values.dtype)
        body = np.concatenate([pad_values, body], axis=0)
    expected = int(end) - int(start)
    if len(body) < expected:
        missing = expected - len(body)
        if len(body) and pad == "edge":
            pad_values = np.repeat(body[-1:], missing, axis=0)
        else:
            pad_values = np.zeros((missing,) + values.shape[1:], dtype=values.dtype)
        body = np.concatenate([body, pad_values], axis=0)
    return body[:expected]


def _linear_slope(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.zeros(values.shape[1], dtype=np.float32)
    time = np.linspace(-0.5, 0.5, len(values), dtype=np.float32)
    denom = float(np.sum(time * time))
    centered = values - values.mean(axis=0, keepdims=True)
    return (time[:, None] * centered).sum(axis=0) / max(denom, 1e-8)


def _frequency_trend_values(long_signals, sampling_rate_hz):
    long_signals = np.asarray(long_signals, dtype=np.float32)
    num_channels = int(long_signals.shape[1])
    if len(long_signals) <= 1:
        zeros = np.zeros(num_channels, dtype=np.float32)
        return {
            "fft_energy": zeros,
            "fft_entropy": zeros,
            "fft_centroid": zeros,
            "fft_peak_freq": zeros,
            "bandpower_low": zeros,
            "bandpower_high": zeros,
            "freeze_index": zeros,
            "bandpower_ratio": zeros,
            "dominant_power": zeros,
        }

    centered = long_signals - long_signals.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    power = np.abs(spectrum).astype(np.float64) ** 2
    freqs = np.fft.rfftfreq(len(long_signals), d=1.0 / float(sampling_rate_hz))
    if len(freqs) > 1:
        power = power[1:]
        freqs = freqs[1:]
    total_power = power.sum(axis=0)
    safe_power = np.maximum(total_power, 1e-12)
    low_mask = (freqs >= 0.5) & (freqs < 3.0)
    high_mask = (freqs >= 3.0) & (freqs <= 8.0)
    low_power = power[low_mask].sum(axis=0) if np.any(low_mask) else np.zeros(num_channels, dtype=np.float64)
    high_power = power[high_mask].sum(axis=0) if np.any(high_mask) else np.zeros(num_channels, dtype=np.float64)
    max_power = power.max(axis=0) if power.size else np.zeros(num_channels, dtype=np.float64)

    probabilities = power / safe_power[None, :]
    entropy_denom = np.log(max(2, power.shape[0]))
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=0) / entropy_denom
    centroid = (freqs[:, None] * power).sum(axis=0) / safe_power
    peak_indices = np.argmax(power, axis=0) if power.size else np.zeros(num_channels, dtype=np.int64)
    peak_freq = freqs[peak_indices] if len(freqs) else np.zeros(num_channels, dtype=np.float64)
    freeze_index = np.log1p(high_power / np.maximum(low_power, 1e-12))
    bandpower_ratio = high_power / np.maximum(low_power + high_power, 1e-12)
    dominant_power = max_power / safe_power
    no_signal = total_power <= 1e-12
    entropy[no_signal] = 0.0
    centroid[no_signal] = 0.0
    peak_freq[no_signal] = 0.0
    freeze_index[no_signal] = 0.0
    bandpower_ratio[no_signal] = 0.0
    dominant_power[no_signal] = 0.0

    return {
        "fft_energy": np.log1p(total_power / max(1, len(long_signals))).astype(np.float32),
        "fft_entropy": entropy.astype(np.float32),
        "fft_centroid": centroid.astype(np.float32),
        "fft_peak_freq": peak_freq.astype(np.float32),
        "bandpower_low": np.log1p(low_power).astype(np.float32),
        "bandpower_high": np.log1p(high_power).astype(np.float32),
        "freeze_index": freeze_index.astype(np.float32),
        "bandpower_ratio": bandpower_ratio.astype(np.float32),
        "dominant_power": dominant_power.astype(np.float32),
    }


def _long_trend_values(long_signals, short_size, trend_features, sampling_rate_hz):
    long_signals = np.asarray(long_signals, dtype=np.float32)
    recent = long_signals[-short_size:]
    early = long_signals[:short_size]
    values = {
        "mean": long_signals.mean(axis=0),
        "std": long_signals.std(axis=0),
        "delta": recent.mean(axis=0) - early.mean(axis=0),
        "slope": _linear_slope(long_signals),
    }
    if any(str(feature) in FREQUENCY_TREND_FEATURES for feature in trend_features):
        values.update(_frequency_trend_values(long_signals, sampling_rate_hz))
    return [values[feature].astype(np.float32) for feature in trend_features]


def _combine_short_long_window(short_signals, long_signals, sensor_cols, trend_features, mode, sampling_rate_hz):
    short_size = int(short_signals.shape[0])
    blocks = []
    columns = []
    if mode == "short_plus_long_trend":
        blocks.append(short_signals.T.astype(np.float32))
        columns.extend(sensor_cols)
    trend_values = _long_trend_values(long_signals, short_size, trend_features, sampling_rate_hz)
    for feature, feature_values in zip(trend_features, trend_values):
        blocks.append(np.repeat(feature_values[:, None], short_size, axis=1).astype(np.float32))
        columns.extend([f"{column}_long_{feature}" for column in sensor_cols])
    return np.concatenate(blocks, axis=0), columns


def _align_short_to_long(short_signals, long_size, pad):
    short_signals = np.asarray(short_signals, dtype=np.float32)
    long_size = int(long_size)
    short_size = int(short_signals.shape[0])
    if short_size > long_size:
        return short_signals[-long_size:]
    prefix_size = long_size - short_size
    if prefix_size <= 0:
        return short_signals
    if short_size and pad == "edge":
        prefix = np.repeat(short_signals[:1], prefix_size, axis=0)
    else:
        prefix = np.zeros((prefix_size,) + short_signals.shape[1:], dtype=short_signals.dtype)
    return np.concatenate([prefix, short_signals], axis=0)


def _combine_raw_short_long_window(short_signals, long_signals, sensor_cols, pad):
    short_aligned = _align_short_to_long(short_signals, len(long_signals), pad)
    blocks = [short_aligned.T.astype(np.float32), np.asarray(long_signals, dtype=np.float32).T]
    columns = [f"{column}_short" for column in sensor_cols]
    columns.extend(f"{column}_long" for column in sensor_cols)
    return np.concatenate(blocks, axis=0), columns


def _combine_multi_window(short_signals, long_signals, sensor_cols, multi_window, sampling_rate_hz):
    mode = multi_window["mode"]
    if mode in RAW_MULTI_WINDOW_MODES:
        return _combine_raw_short_long_window(
            short_signals,
            long_signals,
            sensor_cols,
            multi_window["pad"],
        )
    return _combine_short_long_window(
        short_signals,
        long_signals,
        sensor_cols,
        multi_window["trend_features"],
        mode,
        sampling_rate_hz,
    )


def _output_seq_len(window_size, multi_window):
    if multi_window["enabled"] and multi_window["mode"] in RAW_MULTI_WINDOW_MODES:
        return int(multi_window["long_window_size"])
    return int(window_size)


def _split_subjects(subject_ids, subject_has_fog, val_size, test_size, seed):
    subjects = np.asarray(sorted(subject_ids), dtype=np.int64)
    stratify = np.asarray([subject_has_fog[int(s)] for s in subjects], dtype=np.int64)
    holdout_size = val_size + test_size
    train_subjects, holdout_subjects, _, holdout_stratify = train_test_split(
        subjects,
        stratify,
        test_size=holdout_size,
        random_state=seed,
        stratify=stratify if np.bincount(stratify).min() >= 2 else None,
    )
    relative_test_size = test_size / holdout_size
    can_stratify = len(np.unique(holdout_stratify)) > 1 and np.bincount(holdout_stratify).min() >= 2
    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=relative_test_size,
        random_state=seed,
        stratify=holdout_stratify if can_stratify else None,
    )
    return {
        "train": set(map(int, train_subjects)),
        "val": set(map(int, val_subjects)),
        "test": set(map(int, test_subjects)),
    }


def _split_random_windows(y, val_size, test_size, seed):
    indices = np.arange(len(y))
    holdout_size = val_size + test_size
    train_idx, holdout_idx, _, holdout_y = train_test_split(
        indices,
        y,
        test_size=holdout_size,
        random_state=seed,
        stratify=y,
    )
    relative_test_size = test_size / holdout_size
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_size,
        random_state=seed,
        stratify=holdout_y,
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def _select_loso_val_subject(subject_ids, test_subject, subject_pos_fraction):
    remaining = [int(s) for s in sorted(subject_ids) if int(s) != int(test_subject)]
    if not remaining:
        raise ValueError("LOSO requires at least two subjects.")
    target = float(np.mean([subject_pos_fraction[s] for s in remaining]))
    candidates = [s for s in remaining if subject_pos_fraction[s] > 0] or remaining
    return min(candidates, key=lambda s: (abs(subject_pos_fraction[s] - target), s))


def _split_loso(subject_ids, test_subject, subject_pos_fraction, val_subject=None):
    subjects = set(map(int, subject_ids))
    test_subject = int(test_subject)
    if test_subject not in subjects:
        raise ValueError(f"LOSO test subject {test_subject} is not present in the dataset.")
    if val_subject is None:
        val_subject = _select_loso_val_subject(subjects, test_subject, subject_pos_fraction)
    val_subject = int(val_subject)
    if val_subject == test_subject:
        raise ValueError("LOSO validation subject must be different from the test subject.")
    if val_subject not in subjects:
        raise ValueError(f"LOSO validation subject {val_subject} is not present in the dataset.")
    return {
        "train": subjects - {test_subject, val_subject},
        "val": {val_subject},
        "test": {test_subject},
    }


def build_windows(
    sensor_df,
    clinical_df,
    window_size,
    stride,
    fog_threshold,
    keep_activity_zero=False,
    label_mode="binary",
    pre_fog_seconds=3.0,
    label_rule=None,
    sampling_rate_hz=60,
    multi_window=None,
):
    label_mode = str(label_mode).lower()
    if label_mode not in CLASS_NAMES_BY_MODE:
        raise ValueError("label_mode must be binary or three-class.")
    if label_rule is None:
        label_rule = "priority" if label_mode == "three-class" else "threshold"
    label_rule = str(label_rule).lower()
    if label_rule not in {"threshold", "priority", "center", "majority"}:
        raise ValueError("label_rule must be threshold, priority, center, or majority.")
    if label_mode == "three-class" and label_rule == "threshold":
        label_rule = "priority"
    multi_window = _normalize_multi_window_config(multi_window)
    if multi_window["enabled"] and int(multi_window["long_window_size"]) < int(window_size):
        raise ValueError("data.windowing.multi_window.long_window_size must be >= window_size.")

    if not keep_activity_zero and "activity" in sensor_df.columns:
        sensor_df = sensor_df[sensor_df["activity"] > 0].copy()

    sensor_cols = [c for c in sensor_df.columns if ("_acc_" in c or "_gyro_" in c)]
    sensor_df = sensor_df.sort_values(ID_COLUMNS + ["timestamp"]).reset_index(drop=True)
    sensor_df[sensor_cols] = sensor_df.groupby(ID_COLUMNS, group_keys=False)[sensor_cols].apply(
        lambda group: group.interpolate(method="linear", limit_direction="both")
    )
    sensor_df[sensor_cols] = sensor_df[sensor_cols].fillna(sensor_df[sensor_cols].median(numeric_only=True))
    sensor_df[sensor_cols] = sensor_df[sensor_cols].fillna(0.0)

    clinical = clinical_df.copy()
    clinical["gender_male"] = clinical["gender"].str.upper().map({"M": 1.0, "F": 0.0})
    clinical_feature_cols = CLINICAL_NUMERIC_COLUMNS + ["gender_male"]
    clinical = clinical[["subjectID"] + clinical_feature_cols]
    clinical[clinical_feature_cols] = clinical[clinical_feature_cols].fillna(
        clinical[clinical_feature_cols].median(numeric_only=True)
    )

    merged = sensor_df.merge(clinical, on="subjectID", how="left")
    merged = merged.sort_values(ID_COLUMNS + ["timestamp"]).reset_index(drop=True)

    X, y = [], []
    output_sensor_cols = list(sensor_cols)
    meta = {
        "subjectID": [],
        "sessionID": [],
        "taskID": [],
        "activity": [],
        "fog_severity": [],
        "fog_fraction": [],
        "start_timestamp": [],
        "end_timestamp": [],
        "clinical": [],
    }

    for _, group in merged.groupby(ID_COLUMNS, sort=True):
        if len(group) < window_size:
            continue
        signals = group[sensor_cols].to_numpy(dtype=np.float32)
        fog = group["fog"].to_numpy(dtype=np.int64)
        state = _sample_state(fog, label_mode, sampling_rate_hz, pre_fog_seconds)
        activity = group["activity"].to_numpy(dtype=np.int64)
        severity = group["fog_severity"].to_numpy(dtype=np.int64)
        timestamps = group["timestamp"].to_numpy(dtype=np.float32)
        clinical_values = group[clinical_feature_cols].iloc[0].to_numpy(dtype=np.float32)
        ids = group[ID_COLUMNS].iloc[0]

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            fog_window = fog[start:end]
            fog_fraction = float(fog_window.mean())
            label = _window_label(state[start:end], label_mode, label_rule, fog_threshold)

            short_signals = signals[start:end]
            if multi_window["enabled"]:
                long_end = end
                long_start = long_end - int(multi_window["long_window_size"])
                long_signals = _slice_with_left_pad(signals, long_start, long_end, pad=multi_window["pad"])
                x_window, output_sensor_cols = _combine_multi_window(
                    short_signals,
                    long_signals,
                    sensor_cols,
                    multi_window,
                    sampling_rate_hz,
                )
            else:
                x_window = short_signals.T

            X.append(x_window)
            y.append(label)
            meta["subjectID"].append(int(ids["subjectID"]))
            meta["sessionID"].append(int(ids["sessionID"]))
            meta["taskID"].append(int(ids["taskID"]))
            meta["activity"].append(_mode_int(activity[start:end]))
            fog_label = LABEL_FOG_BINARY if label_mode == "binary" else LABEL_FOG_THREE
            if label == fog_label and np.any(fog_window == 1):
                meta["fog_severity"].append(_mode_int(severity[start:end][fog_window == 1]))
            else:
                meta["fog_severity"].append(0)
            meta["fog_fraction"].append(fog_fraction)
            meta["start_timestamp"].append(float(timestamps[start]))
            meta["end_timestamp"].append(float(timestamps[end - 1]))
            meta["clinical"].append(clinical_values)

    if not X:
        raise ValueError("No windows were created. Reduce window_size or check the input data.")

    arrays = {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "sensor_columns": np.asarray(output_sensor_cols),
        "clinical_feature_names": np.asarray(clinical_feature_cols),
        "class_names": CLASS_NAMES_BY_MODE[label_mode],
    }
    for key, values in meta.items():
        if key == "clinical":
            arrays[key] = np.stack(values).astype(np.float32)
        else:
            dtype = np.float32 if key in {"fog_fraction", "start_timestamp", "end_timestamp"} else np.int64
            arrays[key] = np.asarray(values, dtype=dtype)
    return arrays


def _save_mask_split(out_dir, split, arrays, mask):
    payload = {}
    n = len(arrays["y"])
    for key, value in arrays.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n:
            payload[key] = value[mask]
        else:
            payload[key] = value
    np.savez(out_dir / f"{split}.npz", **payload)


def _save_index_split(out_dir, split, arrays, indices):
    payload = {}
    n = len(arrays["y"])
    for key, value in arrays.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n:
            payload[key] = value[indices]
        else:
            payload[key] = value
    np.savez(out_dir / f"{split}.npz", **payload)


def _class_count_summary(y, class_names):
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=len(class_names))
    return {str(name).lower(): int(counts[idx]) for idx, name in enumerate(class_names)}


def _save_subject_splits(out_dir, arrays, splits, base_summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(base_summary)
    summary["splits"] = {}
    class_names = np.asarray(arrays.get("class_names", CLASS_NAMES_BY_MODE["binary"])).astype(str)
    for split, subjects in splits.items():
        mask = np.isin(arrays["subjectID"], list(subjects))
        _save_mask_split(out_dir, split, arrays, mask)
        y_split = arrays["y"][mask]
        summary["splits"][split] = {
            "subjects": sorted(subjects),
            "num_windows": int(mask.sum()),
            "class_counts": _class_count_summary(y_split, class_names),
            "num_positive": int(np.sum(y_split > 0)),
            "positive_fraction": float(np.mean(y_split > 0)) if len(y_split) else 0.0,
        }
    _save_summary(out_dir, summary)
    return summary


def _random_split_summary(arrays, indices):
    y = arrays["y"][indices]
    subjects = arrays["subjectID"][indices]
    class_names = np.asarray(arrays.get("class_names", CLASS_NAMES_BY_MODE["binary"])).astype(str)
    return {
        "num_windows": int(len(indices)),
        "class_counts": _class_count_summary(y, class_names),
        "num_positive": int(np.sum(y > 0)),
        "positive_fraction": float(np.mean(y > 0)) if len(y) else 0.0,
        "subjects": sorted(map(int, np.unique(subjects))),
        "num_subjects": int(len(np.unique(subjects))),
    }


def _save_summary(out_dir, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _summary_matches(out_dir: Path, expected: dict[str, Any], split_strategy: str) -> bool:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        return False
    for key, value in expected.items():
        if key == "multi_window" and value == {"enabled": False} and key not in summary:
            continue
        if key == "split_strategy" and value == "subject" and key not in summary:
            continue
        if key == "seed" and key not in summary:
            continue
        if summary.get(key) != value:
            return False
    if split_strategy == "loso":
        return any(out_dir.glob("loso_subject_*/train.npz"))
    return all((out_dir / f"{split}.npz").exists() for split in ("train", "val", "test"))


def _cache_dir(wcfg, split_strategy, window_size, stride, fog_threshold, keep_activity_zero, seed):
    if wcfg.get("out_dir"):
        return Path(wcfg["out_dir"])
    threshold = str(fog_threshold).replace(".", "p")
    activity = "keepact0" if keep_activity_zero else "dropact0"
    name = f"fogstar_{split_strategy}_win{window_size}_stride{stride}_thr{threshold}_{activity}_seed{seed}"
    multi_window = _normalize_multi_window_config(wcfg.get("multi_window"))
    if multi_window["enabled"]:
        suffix = f"_mw_{multi_window['mode']}_long{multi_window['long_window_size']}"
        if multi_window["trend_features"]:
            suffix += "_" + "-".join(multi_window["trend_features"])
        name += suffix
    return Path(wcfg.get("cache_root", "data/generated")) / name


def _prepare_arrays(wcfg):
    sensor_csv = Path(wcfg.get("sensor_csv", "data/sensor_data.csv"))
    clinical_csv = Path(wcfg.get("clinical_csv", "data/clinical_data.csv"))
    if not sensor_csv.exists():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_csv}")
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")
    sensor_df = pd.read_csv(sensor_csv)
    clinical_df = pd.read_csv(clinical_csv)
    return build_windows(
        sensor_df,
        clinical_df,
        int(wcfg.get("window_size", 120)),
        int(wcfg.get("stride", 60)),
        float(wcfg.get("fog_threshold", 0.25)),
        keep_activity_zero=bool(wcfg.get("keep_activity_zero", False)),
        label_mode=wcfg.get("label_mode", "binary"),
        pre_fog_seconds=float(wcfg.get("pre_fog_seconds", 3.0)),
        label_rule=wcfg.get("label_rule"),
        sampling_rate_hz=float(wcfg.get("sampling_rate_hz", 60)),
        multi_window=wcfg.get("multi_window"),
    )


def prepare_window_dataset(cfg):
    cfg = copy.deepcopy(cfg)
    dcfg = cfg.get("data", {})
    wcfg = dcfg.get("windowing", {})
    if not wcfg or not bool(wcfg.get("enabled", False)):
        return cfg

    split_strategy = str(wcfg.get("split_strategy", "subject")).lower()
    if split_strategy == "random":
        split_strategy = "random_window"
    if split_strategy not in {"subject", "random_window", "loso"}:
        raise ValueError("data.windowing.split_strategy must be subject, random_window, or loso.")

    window_size = int(wcfg.get("window_size", cfg.get("model", {}).get("seq_len", 120)))
    stride = int(wcfg.get("stride", max(1, window_size // 2)))
    fog_threshold = float(wcfg.get("fog_threshold", 0.25))
    keep_activity_zero = bool(wcfg.get("keep_activity_zero", False))
    label_mode = str(wcfg.get("label_mode", "binary")).lower()
    if label_mode not in CLASS_NAMES_BY_MODE:
        raise ValueError("data.windowing.label_mode must be binary or three-class.")
    label_rule = str(wcfg.get("label_rule", "priority" if label_mode == "three-class" else "threshold")).lower()
    if label_mode == "three-class" and label_rule == "threshold":
        label_rule = "priority"
    pre_fog_seconds = float(wcfg.get("pre_fog_seconds", 3.0))
    multi_window = _normalize_multi_window_config(wcfg.get("multi_window"))
    seed = int(wcfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    out_dir = _cache_dir(wcfg, split_strategy, window_size, stride, fog_threshold, keep_activity_zero, seed)
    expected = {
        "split_strategy": split_strategy,
        "window_size": window_size,
        "stride": stride,
        "fog_threshold": fog_threshold,
        "keep_activity_zero": keep_activity_zero,
        "sampling_rate_hz": int(wcfg.get("sampling_rate_hz", 60)),
        "label_mode": label_mode,
        "label_rule": label_rule,
        "multi_window": multi_window,
    }
    if label_mode == "three-class":
        expected["pre_fog_seconds"] = pre_fog_seconds
    if split_strategy in {"subject", "random_window"}:
        expected["seed"] = seed
    if split_strategy == "loso":
        loso_subjects = wcfg.get("loso_subjects")
        if loso_subjects is None:
            expected["loso_subjects"] = "all"
        else:
            subjects = loso_subjects if isinstance(loso_subjects, (list, tuple)) else [loso_subjects]
            expected["loso_subjects"] = [int(subject) for subject in subjects]
        expected["val_subject"] = wcfg.get("val_subject")

    force = bool(wcfg.get("force", False))
    if force or not _summary_matches(out_dir, expected, split_strategy):
        print(f"Preparing window dataset: {out_dir}")
        arrays = _prepare_arrays(
            {
                **wcfg,
                "window_size": window_size,
                "stride": stride,
                "fog_threshold": fog_threshold,
                "label_mode": label_mode,
                "label_rule": label_rule,
                "pre_fog_seconds": pre_fog_seconds,
                "multi_window": multi_window,
            }
        )
        base_summary = dict(expected)
        base_summary["source"] = {
            "sensor_csv": str(wcfg.get("sensor_csv", "data/sensor_data.csv")),
            "clinical_csv": str(wcfg.get("clinical_csv", "data/clinical_data.csv")),
        }

        if split_strategy == "subject":
            fog_label = LABEL_FOG_BINARY if label_mode == "binary" else LABEL_FOG_THREE
            subject_has_fog = {
                int(subject): int(np.any(arrays["y"][arrays["subjectID"] == subject] == fog_label))
                for subject in np.unique(arrays["subjectID"])
            }
            splits = _split_subjects(
                np.unique(arrays["subjectID"]),
                subject_has_fog,
                float(wcfg.get("val_size", 0.15)),
                float(wcfg.get("test_size", 0.15)),
                seed,
            )
            _save_subject_splits(out_dir, arrays, splits, base_summary)
        elif split_strategy == "random_window":
            out_dir.mkdir(parents=True, exist_ok=True)
            splits = _split_random_windows(
                arrays["y"],
                float(wcfg.get("val_size", 0.15)),
                float(wcfg.get("test_size", 0.15)),
                seed,
            )
            summary = {
                **base_summary,
                "warning": "Windows are randomly split; subjects and neighboring overlapping windows may appear in multiple splits.",
                "splits": {},
            }
            for split, indices in splits.items():
                _save_index_split(out_dir, split, arrays, indices)
                summary["splits"][split] = _random_split_summary(arrays, indices)
            _save_summary(out_dir, summary)
        else:
            subjects = sorted(map(int, np.unique(arrays["subjectID"])))
            fog_label = LABEL_FOG_BINARY if label_mode == "binary" else LABEL_FOG_THREE
            subject_pos_fraction = {
                int(subject): float(np.mean(arrays["y"][arrays["subjectID"] == subject] == fog_label))
                for subject in np.unique(arrays["subjectID"])
            }
            loso_subjects = wcfg.get("loso_subjects") or subjects
            loso_subjects = loso_subjects if isinstance(loso_subjects, (list, tuple)) else [loso_subjects]
            summary = {**base_summary, "folds": {}}
            for test_subject in loso_subjects:
                splits = _split_loso(subjects, test_subject, subject_pos_fraction, val_subject=wcfg.get("val_subject"))
                fold_dir = out_dir / f"loso_subject_{int(test_subject):02d}"
                fold_summary = _save_subject_splits(
                    fold_dir,
                    arrays,
                    splits,
                    {**base_summary, "test_subject": int(test_subject)},
                )
                summary["folds"][fold_dir.name] = fold_summary["splits"]
            _save_summary(out_dir, summary)
    else:
        print(f"Using cached window dataset: {out_dir}")

    cfg.setdefault("data", {})["root"] = str(out_dir)
    cfg["data"]["train_file"] = "train.npz"
    cfg["data"]["val_file"] = "val.npz"
    cfg["data"]["test_file"] = "test.npz"
    if bool(wcfg.get("sync_model_seq_len", True)):
        cfg.setdefault("model", {})["seq_len"] = _output_seq_len(window_size, multi_window)
        if multi_window["enabled"] and multi_window["mode"] in RAW_MULTI_WINDOW_MODES:
            cfg["model"]["short_seq_len"] = window_size
            cfg["model"]["long_seq_len"] = int(multi_window["long_window_size"])
    if split_strategy == "loso":
        cfg.setdefault("experiment", {})["loso_root"] = str(out_dir)
    return cfg


__all__ = ["build_windows", "prepare_window_dataset"]
