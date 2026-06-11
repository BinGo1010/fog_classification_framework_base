import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_provider.window_cache import build_windows


ID_COLUMNS = ["subjectID", "sessionID", "taskID"]
LABEL_COLUMNS = ["activity", "fog", "fog_severity"]
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


def mode_int(values, default=0):
    values = np.asarray(values)
    if values.size == 0:
        return int(default)
    values = values.astype(np.int64)
    return int(np.bincount(values).argmax())


def split_subjects(subject_ids, subject_has_fog, val_size, test_size, seed):
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
    split_stratify = holdout_stratify
    can_stratify_holdout = (
        len(np.unique(split_stratify)) > 1
        and np.bincount(split_stratify).min() >= 2
    )
    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=relative_test_size,
        random_state=seed,
        stratify=split_stratify if can_stratify_holdout else None,
    )
    return {
        "train": set(map(int, train_subjects)),
        "val": set(map(int, val_subjects)),
        "test": set(map(int, test_subjects)),
    }


def select_loso_val_subject(subject_ids, test_subject, subject_pos_fraction):
    remaining = [int(s) for s in sorted(subject_ids) if int(s) != int(test_subject)]
    if not remaining:
        raise ValueError("LOSO requires at least two subjects.")

    target = float(np.mean([subject_pos_fraction[s] for s in remaining]))
    candidates = [s for s in remaining if subject_pos_fraction[s] > 0]
    if not candidates:
        candidates = remaining
    return min(candidates, key=lambda s: (abs(subject_pos_fraction[s] - target), s))


def split_loso(subject_ids, test_subject, subject_pos_fraction, val_subject=None):
    subjects = set(map(int, subject_ids))
    test_subject = int(test_subject)
    if test_subject not in subjects:
        raise ValueError(f"LOSO test subject {test_subject} is not present in the dataset.")

    if val_subject is None:
        val_subject = select_loso_val_subject(subjects, test_subject, subject_pos_fraction)
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


def save_split(out_dir, split, arrays, mask):
    payload = {}
    n = len(arrays["y"])
    for key, value in arrays.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n:
            payload[key] = value[mask]
        else:
            payload[key] = value
    np.savez(out_dir / f"{split}.npz", **payload)


def save_splits(out_dir, arrays, splits, base_summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(base_summary)
    summary["splits"] = {}
    for split, subjects in splits.items():
        mask = np.isin(arrays["subjectID"], list(subjects))
        save_split(out_dir, split, arrays, mask)
        y_split = arrays["y"][mask]
        summary["splits"][split] = {
            "subjects": sorted(subjects),
            "num_windows": int(mask.sum()),
            "num_positive": int(y_split.sum()),
            "positive_fraction": float(y_split.mean()) if len(y_split) else 0.0,
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare FoG-STAR CSV files as window-level NPZ datasets.")
    parser.add_argument("--sensor_csv", default="data/sensor_data.csv")
    parser.add_argument("--clinical_csv", default="data/clinical_data.csv")
    parser.add_argument("--out_dir", default="data/fogstar_npz")
    parser.add_argument("--window_size", type=int, default=120, help="Samples per window. At 60 Hz, 120 = 2 seconds.")
    parser.add_argument("--stride", type=int, default=60, help="Window stride in samples. At 60 Hz, 60 = 1 second.")
    parser.add_argument("--fog_threshold", type=float, default=0.25, help="Minimum fraction of FoG samples for a positive window.")
    parser.add_argument("--keep_activity_zero", action="store_true", help="Keep activity=0 samples. By default they are removed.")
    parser.add_argument("--split_strategy", choices=["subject", "loso"], default="subject")
    parser.add_argument("--loso_subject", type=int, default=None, help="Subject to hold out for LOSO. If omitted, all LOSO folds are generated.")
    parser.add_argument("--val_subject", type=int, default=None, help="Validation subject for a single LOSO fold. If omitted, one is selected automatically.")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sensor_df = pd.read_csv(args.sensor_csv)
    clinical_df = pd.read_csv(args.clinical_csv)
    arrays = build_windows(
        sensor_df,
        clinical_df,
        args.window_size,
        args.stride,
        args.fog_threshold,
        keep_activity_zero=args.keep_activity_zero,
    )

    subject_has_fog = {}
    for subject in np.unique(arrays["subjectID"]):
        mask = arrays["subjectID"] == subject
        subject_has_fog[int(subject)] = int(arrays["y"][mask].sum() > 0)
    subject_pos_fraction = {
        int(subject): float(arrays["y"][arrays["subjectID"] == subject].mean())
        for subject in np.unique(arrays["subjectID"])
    }

    out_dir = Path(args.out_dir)
    base_summary = {
        "split_strategy": args.split_strategy,
        "window_size": args.window_size,
        "stride": args.stride,
        "fog_threshold": args.fog_threshold,
        "keep_activity_zero": args.keep_activity_zero,
        "sampling_rate_hz": 60,
        "subject_positive_fraction": subject_pos_fraction,
    }
    if args.split_strategy == "subject":
        splits = split_subjects(
            np.unique(arrays["subjectID"]),
            subject_has_fog,
            args.val_size,
            args.test_size,
            args.seed,
        )
        summary = save_splits(out_dir, arrays, splits, base_summary)
    else:
        subjects = sorted(map(int, np.unique(arrays["subjectID"])))
        loso_subjects = [args.loso_subject] if args.loso_subject is not None else subjects
        summary = dict(base_summary)
        summary["folds"] = {}
        for test_subject in loso_subjects:
            splits = split_loso(
                subjects,
                test_subject,
                subject_pos_fraction,
                val_subject=args.val_subject if args.loso_subject is not None else None,
            )
            fold_dir = out_dir / f"loso_subject_{int(test_subject):02d}"
            fold_summary = save_splits(fold_dir, arrays, splits, {**base_summary, "test_subject": int(test_subject)})
            summary["folds"][f"loso_subject_{int(test_subject):02d}"] = fold_summary["splits"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved FoG-STAR NPZ dataset to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
