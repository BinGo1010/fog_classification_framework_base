#!/usr/bin/env python
"""Validate window-level FOG datasets used by LOSO training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate windows.npz and loso_folds.npz for FOG LOSO training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--expected-channels", type=int)
    parser.add_argument("--expected-classes", type=int)
    parser.add_argument(
        "--allow-empty-train",
        action="store_true",
        help="Allow folds with no train windows, useful only for tiny smoke datasets.",
    )
    return parser.parse_args()


def require_keys(files: set[str], required: set[str], path: Path) -> None:
    missing = sorted(required - files)
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")


def normalize_subject_groups(values: np.ndarray) -> list[list[str]]:
    values = np.asarray(values).astype(str)
    if values.ndim == 0:
        return [[str(values.item())]]
    if values.ndim == 1:
        return [[item] for item in values.tolist()]
    groups: list[list[str]] = []
    for row in values:
        groups.append([str(item) for item in row.tolist() if str(item) != ""])
    return groups


def normalize_code_groups(values: np.ndarray | None) -> list[np.ndarray] | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.int64)
    if values.ndim == 1:
        return [np.array([code], dtype=np.int64) for code in values.tolist()]
    return [row[row >= 0].astype(np.int64) for row in values]


def load_windows(data_dir: Path) -> dict[str, np.ndarray]:
    path = data_dir / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing windows.npz: {path}")
    with np.load(path, allow_pickle=True) as data:
        require_keys(set(data.files), {"X", "y", "subject_code", "subjects", "class_names"}, path)
        return {key: data[key] for key in data.files}


def validate_windows(
    arrays: dict[str, np.ndarray],
    expected_channels: int | None,
    expected_classes: int | None,
) -> dict:
    x = np.asarray(arrays["X"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.int64).reshape(-1)
    subject_code = np.asarray(arrays["subject_code"], dtype=np.int64).reshape(-1)
    subjects = np.asarray(arrays["subjects"]).astype(str).reshape(-1)
    class_names = np.asarray(arrays["class_names"]).astype(str).reshape(-1)

    if x.ndim != 3:
        raise ValueError(f"X must be 3D [window,time,channel], got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("X has no windows")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"X/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if subject_code.shape[0] != x.shape[0]:
        raise ValueError("subject_code length does not match X")
    if expected_channels is not None and x.shape[2] != expected_channels:
        raise ValueError(f"X has {x.shape[2]} channels, expected {expected_channels}")
    if expected_classes is not None and len(class_names) != expected_classes:
        raise ValueError(f"class_names has {len(class_names)} classes, expected {expected_classes}")
    if len(class_names) == 0:
        raise ValueError("class_names is empty")
    if not np.isfinite(x).all():
        raise ValueError("X contains non-finite values")

    label_values = set(np.unique(y).astype(int).tolist())
    allowed = set(range(len(class_names)))
    if not label_values.issubset(allowed):
        raise ValueError(f"y has labels outside {sorted(allowed)}: {sorted(label_values)}")
    if subject_code.min(initial=0) < 0 or subject_code.max(initial=-1) >= len(subjects):
        raise ValueError("subject_code contains values outside subjects")

    if "subject" in arrays:
        subject = np.asarray(arrays["subject"]).astype(str).reshape(-1)
        if subject.shape[0] != x.shape[0]:
            raise ValueError("subject length does not match X")
        decoded = subjects[subject_code]
        if not np.array_equal(subject, decoded):
            mismatch = int(np.flatnonzero(subject != decoded)[0])
            raise ValueError(
                f"subject/subject_code mismatch at window {mismatch}: "
                f"{subject[mismatch]} vs {decoded[mismatch]}"
            )

    for key in ("file_id", "start_sample", "end_sample", "native_hz"):
        if key in arrays and np.asarray(arrays[key]).reshape(-1).shape[0] != x.shape[0]:
            raise ValueError(f"{key} length does not match X")
    if "start_sample" in arrays and "end_sample" in arrays:
        start = np.asarray(arrays["start_sample"], dtype=np.int64).reshape(-1)
        end = np.asarray(arrays["end_sample"], dtype=np.int64).reshape(-1)
        if not np.all(start < end):
            raise ValueError("Every start_sample must be smaller than end_sample")

    counts = np.bincount(y, minlength=len(class_names)).astype(int)
    return {
        "windows": int(x.shape[0]),
        "time_steps": int(x.shape[1]),
        "channels": int(x.shape[2]),
        "classes": class_names.tolist(),
        "class_counts": {str(name): int(counts[idx]) for idx, name in enumerate(class_names)},
        "subjects": subjects.tolist(),
    }


def codes_from_subject_groups(
    groups: list[list[str]],
    subject_to_code: dict[str, int],
    fold_key: str,
) -> list[np.ndarray]:
    code_groups: list[np.ndarray] = []
    for fold, group in enumerate(groups):
        missing = [subject for subject in group if subject not in subject_to_code]
        if missing:
            raise ValueError(f"{fold_key}[{fold}] contains unknown subjects: {missing}")
        code_groups.append(np.array([subject_to_code[subject] for subject in group], dtype=np.int64))
    return code_groups


def validate_folds(
    data_dir: Path,
    window_subject_code: np.ndarray,
    subjects: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    allow_empty_train: bool,
) -> dict:
    path = data_dir / "loso_folds.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing loso_folds.npz: {path}")

    with np.load(path, allow_pickle=True) as folds:
        require_keys(
            set(folds.files),
            {"subjects", "fold_test_subjects", "fold_val_subjects", "window_subject_code"},
            path,
        )
        fold_subjects = np.asarray(folds["subjects"]).astype(str).reshape(-1)
        fold_window_subject_code = np.asarray(folds["window_subject_code"], dtype=np.int64).reshape(-1)
        test_groups = normalize_subject_groups(folds["fold_test_subjects"])
        val_groups = normalize_subject_groups(folds["fold_val_subjects"])
        test_code_groups = normalize_code_groups(
            folds["fold_test_subject_codes"] if "fold_test_subject_codes" in folds.files else None
        )
        val_code_groups = normalize_code_groups(
            folds["fold_val_subject_codes"] if "fold_val_subject_codes" in folds.files else None
        )

    if not np.array_equal(fold_subjects, subjects):
        raise ValueError("subjects differ between windows.npz and loso_folds.npz")
    if not np.array_equal(fold_window_subject_code, window_subject_code):
        raise ValueError("window_subject_code differs from windows.npz subject_code")
    if len(test_groups) != len(val_groups):
        raise ValueError("fold_test_subjects and fold_val_subjects have different fold counts")
    if test_code_groups is not None and len(test_code_groups) != len(test_groups):
        raise ValueError("fold_test_subject_codes fold count does not match fold_test_subjects")
    if val_code_groups is not None and len(val_code_groups) != len(val_groups):
        raise ValueError("fold_val_subject_codes fold count does not match fold_val_subjects")

    subject_to_code = {str(subject): idx for idx, subject in enumerate(subjects)}
    if test_code_groups is None:
        test_code_groups = codes_from_subject_groups(test_groups, subject_to_code, "fold_test_subjects")
    if val_code_groups is None:
        val_code_groups = codes_from_subject_groups(val_groups, subject_to_code, "fold_val_subjects")

    seen_test_codes: list[int] = []
    fold_rows: list[dict] = []
    for fold, (test_codes, val_codes) in enumerate(zip(test_code_groups, val_code_groups)):
        if test_codes.size == 0:
            raise ValueError(f"Fold {fold} has no test subjects")
        if val_codes.size == 0:
            raise ValueError(f"Fold {fold} has no validation subjects")
        if np.intersect1d(test_codes, val_codes).size:
            raise ValueError(f"Fold {fold} leaks subjects between test and validation")
        unknown_codes = np.setdiff1d(np.r_[test_codes, val_codes], np.arange(len(subjects)))
        if unknown_codes.size:
            raise ValueError(f"Fold {fold} references unknown subject codes: {unknown_codes.tolist()}")

        test_mask = np.isin(window_subject_code, test_codes)
        val_mask = np.isin(window_subject_code, val_codes)
        train_mask = ~(test_mask | val_mask)
        if not test_mask.any():
            raise ValueError(f"Fold {fold} has no test windows")
        if not val_mask.any():
            raise ValueError(f"Fold {fold} has no validation windows")
        if not allow_empty_train and not train_mask.any():
            raise ValueError(f"Fold {fold} has no train windows")

        row = {
            "fold": fold,
            "train_windows": int(train_mask.sum()),
            "val_windows": int(val_mask.sum()),
            "test_windows": int(test_mask.sum()),
            "test_subject": "|".join(subjects[test_codes].astype(str).tolist()),
            "val_subject": "|".join(subjects[val_codes].astype(str).tolist()),
        }
        for split, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            counts = np.bincount(y[mask], minlength=len(class_names))
            for idx, name in enumerate(class_names):
                row[f"{split}_{str(name).lower()}"] = int(counts[idx])
        fold_rows.append(row)
        seen_test_codes.extend(test_codes.astype(int).tolist())

    missing = sorted(set(range(len(subjects))) - set(seen_test_codes))
    if missing:
        raise ValueError(f"Subjects never used as test subjects: {subjects[missing].tolist()}")

    csv_path = data_dir / "loso_folds.csv"
    if csv_path.exists():
        csv_folds = pd.read_csv(csv_path)
        if "fold" not in csv_folds.columns or len(csv_folds) != len(fold_rows):
            raise ValueError("loso_folds.csv does not match loso_folds.npz fold count")

    return {
        "folds": int(len(fold_rows)),
        "folds_with_empty_train": int(sum(row["train_windows"] == 0 for row in fold_rows)),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    arrays = load_windows(data_dir)
    window_summary = validate_windows(arrays, args.expected_channels, args.expected_classes)
    fold_summary = validate_folds(
        data_dir=data_dir,
        window_subject_code=np.asarray(arrays["subject_code"], dtype=np.int64).reshape(-1),
        subjects=np.asarray(arrays["subjects"]).astype(str).reshape(-1),
        y=np.asarray(arrays["y"], dtype=np.int64).reshape(-1),
        class_names=np.asarray(arrays["class_names"]).astype(str).reshape(-1),
        allow_empty_train=args.allow_empty_train,
    )
    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                **window_summary,
                **fold_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
