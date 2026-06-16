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


def _mode_int(values, default=0):
    values = np.asarray(values)
    if values.size == 0:
        return int(default)
    values = values.astype(np.int64)
    return int(np.bincount(values).argmax())


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


def build_windows(sensor_df, clinical_df, window_size, stride, fog_threshold, keep_activity_zero=False):
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
        activity = group["activity"].to_numpy(dtype=np.int64)
        severity = group["fog_severity"].to_numpy(dtype=np.int64)
        timestamps = group["timestamp"].to_numpy(dtype=np.float32)
        clinical_values = group[clinical_feature_cols].iloc[0].to_numpy(dtype=np.float32)
        ids = group[ID_COLUMNS].iloc[0]

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            fog_window = fog[start:end]
            fog_fraction = float(fog_window.mean())
            label = int(fog_fraction >= fog_threshold)

            X.append(signals[start:end].T)
            y.append(label)
            meta["subjectID"].append(int(ids["subjectID"]))
            meta["sessionID"].append(int(ids["sessionID"]))
            meta["taskID"].append(int(ids["taskID"]))
            meta["activity"].append(_mode_int(activity[start:end]))
            if label:
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
        "sensor_columns": np.asarray(sensor_cols),
        "clinical_feature_names": np.asarray(clinical_feature_cols),
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


def _save_subject_splits(out_dir, arrays, splits, base_summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(base_summary)
    summary["splits"] = {}
    for split, subjects in splits.items():
        mask = np.isin(arrays["subjectID"], list(subjects))
        _save_mask_split(out_dir, split, arrays, mask)
        y_split = arrays["y"][mask]
        summary["splits"][split] = {
            "subjects": sorted(subjects),
            "num_windows": int(mask.sum()),
            "num_positive": int(y_split.sum()),
            "positive_fraction": float(y_split.mean()) if len(y_split) else 0.0,
        }
    _save_summary(out_dir, summary)
    return summary


def _random_split_summary(arrays, indices):
    y = arrays["y"][indices]
    subjects = arrays["subjectID"][indices]
    return {
        "num_windows": int(len(indices)),
        "num_positive": int(y.sum()),
        "positive_fraction": float(y.mean()) if len(y) else 0.0,
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
    seed = int(wcfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    out_dir = _cache_dir(wcfg, split_strategy, window_size, stride, fog_threshold, keep_activity_zero, seed)
    expected = {
        "split_strategy": split_strategy,
        "window_size": window_size,
        "stride": stride,
        "fog_threshold": fog_threshold,
        "keep_activity_zero": keep_activity_zero,
        "sampling_rate_hz": int(wcfg.get("sampling_rate_hz", 60)),
    }
    if split_strategy in {"subject", "random_window"}:
        expected["seed"] = seed

    force = bool(wcfg.get("force", False))
    if force or not _summary_matches(out_dir, expected, split_strategy):
        print(f"Preparing window dataset: {out_dir}")
        arrays = _prepare_arrays({**wcfg, "window_size": window_size, "stride": stride, "fog_threshold": fog_threshold})
        base_summary = dict(expected)
        base_summary["source"] = {
            "sensor_csv": str(wcfg.get("sensor_csv", "data/sensor_data.csv")),
            "clinical_csv": str(wcfg.get("clinical_csv", "data/clinical_data.csv")),
        }

        if split_strategy == "subject":
            subject_has_fog = {
                int(subject): int(arrays["y"][arrays["subjectID"] == subject].sum() > 0)
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
            subject_pos_fraction = {
                int(subject): float(arrays["y"][arrays["subjectID"] == subject].mean())
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
        cfg.setdefault("model", {})["seq_len"] = window_size
    if split_strategy == "loso":
        cfg.setdefault("experiment", {})["loso_root"] = str(out_dir)
    return cfg


__all__ = ["build_windows", "prepare_window_dataset"]
