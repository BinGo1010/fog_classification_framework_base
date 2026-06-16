import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from prepare_fogstar_npz import (
    CLINICAL_NUMERIC_COLUMNS,
    ID_COLUMNS,
    mode_int,
    split_loso,
    split_subjects,
)


CLASS_NAMES = {
    0: "No-FoG",
    1: "Shuffling",
    2: "Trembling",
    3: "Akinesia",
}


def prepare_sensor_and_clinical(sensor_df, clinical_df, keep_activity_zero=False):
    if not keep_activity_zero and "activity" in sensor_df.columns:
        sensor_df = sensor_df[sensor_df["activity"] > 0].copy()

    sensor_cols = [
        c for c in sensor_df.columns
        if ("_acc_" in c or "_gyro_" in c)
    ]
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
    return merged, sensor_cols, clinical_feature_cols


def severity_window_label(fog_window, severity_window, fog_threshold):
    fog_fraction = float(np.mean(fog_window))
    if fog_fraction < fog_threshold:
        return 0, fog_fraction
    fog_severity = severity_window[fog_window == 1]
    fog_severity = fog_severity[fog_severity > 0]
    return mode_int(fog_severity, default=0), fog_fraction


def build_severity_windows(sensor_df, clinical_df, window_size, stride, fog_threshold, keep_activity_zero=False):
    merged, sensor_cols, clinical_feature_cols = prepare_sensor_and_clinical(
        sensor_df,
        clinical_df,
        keep_activity_zero=keep_activity_zero,
    )

    X, y = [], []
    meta = {
        "subjectID": [],
        "sessionID": [],
        "taskID": [],
        "activity": [],
        "fog_binary": [],
        "fog_fraction": [],
        "start_timestamp": [],
        "end_timestamp": [],
        "clinical": [],
    }

    for (_, _, _), group in merged.groupby(ID_COLUMNS, sort=True):
        if len(group) < window_size:
            continue
        signals = group[sensor_cols].to_numpy(dtype=np.float32)
        fog = group["fog"].to_numpy(dtype=np.int64)
        severity = group["fog_severity"].to_numpy(dtype=np.int64)
        activity = group["activity"].to_numpy(dtype=np.int64)
        timestamps = group["timestamp"].to_numpy(dtype=np.float32)
        clinical_values = group[clinical_feature_cols].iloc[0].to_numpy(dtype=np.float32)
        ids = group[ID_COLUMNS].iloc[0]

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            label, fog_fraction = severity_window_label(fog[start:end], severity[start:end], fog_threshold)

            X.append(signals[start:end].T)
            y.append(label)
            meta["subjectID"].append(int(ids["subjectID"]))
            meta["sessionID"].append(int(ids["sessionID"]))
            meta["taskID"].append(int(ids["taskID"]))
            meta["activity"].append(mode_int(activity[start:end]))
            meta["fog_binary"].append(int(label > 0))
            meta["fog_fraction"].append(fog_fraction)
            meta["start_timestamp"].append(float(timestamps[start]))
            meta["end_timestamp"].append(float(timestamps[end - 1]))
            meta["clinical"].append(clinical_values)

    if not X:
        raise ValueError("No windows were created. Reduce --window_size or check the input data.")

    arrays = {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "sensor_columns": np.asarray(sensor_cols),
        "clinical_feature_names": np.asarray(clinical_feature_cols),
        "class_names": np.asarray([CLASS_NAMES[i] for i in range(4)]),
    }
    for key, values in meta.items():
        if key == "clinical":
            arrays[key] = np.stack(values).astype(np.float32)
        elif key in {"fog_fraction", "start_timestamp", "end_timestamp"}:
            arrays[key] = np.asarray(values, dtype=np.float32)
        else:
            arrays[key] = np.asarray(values, dtype=np.int64)
    return arrays


def save_split_by_mask(out_dir, split, arrays, mask):
    payload = {}
    n = len(arrays["y"])
    for key, value in arrays.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n:
            payload[key] = value[mask]
        else:
            payload[key] = value
    np.savez(out_dir / f"{split}.npz", **payload)


def save_split_by_index(out_dir, split, arrays, indices):
    payload = {}
    n = len(arrays["y"])
    for key, value in arrays.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n:
            payload[key] = value[indices]
        else:
            payload[key] = value
    np.savez(out_dir / f"{split}.npz", **payload)


def class_count_dict(y):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=4)
    return {str(i): int(counts[i]) for i in range(4)}


def split_summary(arrays, selector):
    y = arrays["y"][selector]
    subjects = arrays["subjectID"][selector]
    return {
        "num_windows": int(len(y)),
        "class_counts": class_count_dict(y),
        "class_fraction": {str(i): float(np.mean(y == i)) if len(y) else 0.0 for i in range(4)},
        "subjects": sorted(map(int, np.unique(subjects))),
        "num_subjects": int(len(np.unique(subjects))),
    }


def save_subject_splits(out_dir, arrays, splits, base_summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(base_summary)
    summary["splits"] = {}
    for split, subjects in splits.items():
        mask = np.isin(arrays["subjectID"], list(subjects))
        save_split_by_mask(out_dir, split, arrays, mask)
        summary["splits"][split] = split_summary(arrays, mask)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def split_random_windows(y, val_size, test_size, seed):
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


def save_random_window_splits(out_dir, arrays, splits, base_summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(base_summary)
    summary["splits"] = {}
    for split, indices in splits.items():
        save_split_by_index(out_dir, split, arrays, indices)
        summary["splits"][split] = split_summary(arrays, indices)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare FoG-STAR severity windows as 4-class NPZ datasets.")
    parser.add_argument("--sensor_csv", default="data/sensor_data.csv")
    parser.add_argument("--clinical_csv", default="data/clinical_data.csv")
    parser.add_argument("--out_dir", default="data/fogstar_severity_random_window")
    parser.add_argument("--window_size", type=int, default=120, help="Samples per window. At 60 Hz, 120 = 2 seconds.")
    parser.add_argument("--stride", type=int, default=60, help="Window stride in samples. At 60 Hz, 60 = 1 second.")
    parser.add_argument("--fog_threshold", type=float, default=0.25, help="Minimum fraction of FoG samples before assigning severity 1/2/3.")
    parser.add_argument("--keep_activity_zero", action="store_true", help="Keep activity=0 samples. By default they are removed.")
    parser.add_argument("--split_strategy", choices=["random_window", "subject", "loso"], default="random_window")
    parser.add_argument("--loso_subject", type=int, default=None, help="Subject to hold out for LOSO. If omitted, all LOSO folds are generated.")
    parser.add_argument("--val_subject", type=int, default=None, help="Validation subject for a single LOSO fold.")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sensor_df = pd.read_csv(args.sensor_csv)
    clinical_df = pd.read_csv(args.clinical_csv)
    arrays = build_severity_windows(
        sensor_df,
        clinical_df,
        args.window_size,
        args.stride,
        args.fog_threshold,
        keep_activity_zero=args.keep_activity_zero,
    )

    subject_positive_fraction = {
        int(subject): float(np.mean(arrays["y"][arrays["subjectID"] == subject] > 0))
        for subject in np.unique(arrays["subjectID"])
    }
    subject_has_fog = {subject: int(value > 0) for subject, value in subject_positive_fraction.items()}
    base_summary = {
        "split_strategy": args.split_strategy,
        "label_schema": CLASS_NAMES,
        "label_rule": "0 if window FoG fraction < fog_threshold; otherwise majority fog_severity among FoG samples.",
        "window_size": args.window_size,
        "stride": args.stride,
        "fog_threshold": args.fog_threshold,
        "keep_activity_zero": args.keep_activity_zero,
        "sampling_rate_hz": 60,
        "seed": args.seed,
        "overall": split_summary(arrays, np.arange(len(arrays["y"]))),
    }

    out_dir = Path(args.out_dir)
    if args.split_strategy == "random_window":
        splits = split_random_windows(arrays["y"], args.val_size, args.test_size, args.seed)
        summary = save_random_window_splits(out_dir, arrays, splits, base_summary)
    elif args.split_strategy == "subject":
        splits = split_subjects(
            np.unique(arrays["subjectID"]),
            subject_has_fog,
            args.val_size,
            args.test_size,
            args.seed,
        )
        summary = save_subject_splits(out_dir, arrays, splits, base_summary)
    else:
        subjects = sorted(map(int, np.unique(arrays["subjectID"])))
        loso_subjects = [args.loso_subject] if args.loso_subject is not None else subjects
        summary = dict(base_summary)
        summary["folds"] = {}
        for test_subject in loso_subjects:
            splits = split_loso(
                subjects,
                test_subject,
                subject_positive_fraction,
                val_subject=args.val_subject if args.loso_subject is not None else None,
            )
            fold_dir = out_dir / f"loso_subject_{int(test_subject):02d}"
            fold_summary = save_subject_splits(fold_dir, arrays, splits, {**base_summary, "test_subject": int(test_subject)})
            summary["folds"][f"loso_subject_{int(test_subject):02d}"] = fold_summary["splits"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved FoG-STAR severity NPZ dataset to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
