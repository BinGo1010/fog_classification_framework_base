#!/usr/bin/env python
"""Export compact LOSO windows into per-fold train/val/test NPZ directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split windows.npz + loso_folds.npz into loso_subject_XX fold directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_code_groups(values: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim == 1:
        return [np.array([code], dtype=np.int64) for code in values.tolist()]
    return [row[row >= 0].astype(np.int64) for row in values]


def normalize_subject_groups(values: np.ndarray) -> list[list[str]]:
    values = np.asarray(values).astype(str)
    if values.ndim == 1:
        return [[item] for item in values.tolist()]
    return [[item for item in row.tolist() if item] for row in values]


def subject_groups_to_codes(groups: list[list[str]], subjects: np.ndarray) -> list[np.ndarray]:
    subject_to_code = {str(subject): idx for idx, subject in enumerate(subjects.astype(str))}
    code_groups: list[np.ndarray] = []
    for group in groups:
        missing = [subject for subject in group if subject not in subject_to_code]
        if missing:
            raise ValueError(f"Unknown subject(s) in fold groups: {missing}")
        code_groups.append(np.array([subject_to_code[subject] for subject in group], dtype=np.int64))
    return code_groups


def arrays_from_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: Path, compress: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def split_payload(windows: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    n = int(np.asarray(windows["y"]).shape[0])
    payload: dict[str, np.ndarray] = {}
    for key, value in windows.items():
        array = np.asarray(value)
        if array.shape and array.shape[0] == n:
            payload[key] = array[mask]
        else:
            payload[key] = array
    if "sensor_columns" not in payload and "feature_names" in payload:
        payload["sensor_columns"] = np.asarray(payload["feature_names"])
    return payload


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"fold_count": len(rows), "folds": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            existing = sorted(output_dir.glob("loso_subject_*/train.npz"))
            if existing:
                raise FileExistsError(f"Fold exports already exist in {output_dir}. Use --overwrite.")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = arrays_from_npz(data_dir / "windows.npz")
    folds = arrays_from_npz(data_dir / "loso_folds.npz")
    for key in ("X", "y", "subject_code", "subjects"):
        if key not in windows:
            raise KeyError(f"windows.npz is missing {key}")

    y = np.asarray(windows["y"]).reshape(-1)
    subject_code = np.asarray(windows["subject_code"], dtype=np.int64).reshape(-1)
    subjects = np.asarray(windows["subjects"]).astype(str).reshape(-1)

    if "fold_test_subject_codes" in folds:
        test_groups = normalize_code_groups(folds["fold_test_subject_codes"])
    else:
        test_groups = subject_groups_to_codes(
            normalize_subject_groups(folds["fold_test_subjects"]),
            subjects,
        )
    if "fold_val_subject_codes" in folds:
        val_groups = normalize_code_groups(folds["fold_val_subject_codes"])
    else:
        val_groups = subject_groups_to_codes(
            normalize_subject_groups(folds["fold_val_subjects"]),
            subjects,
        )
    if len(test_groups) != len(val_groups):
        raise ValueError("Test and validation fold counts differ.")

    class_names = np.asarray(windows.get("class_names", []))
    rows: list[dict[str, Any]] = []
    for fold_idx, (test_codes, val_codes) in enumerate(zip(test_groups, val_groups), start=1):
        test_mask = np.isin(subject_code, test_codes)
        val_mask = np.isin(subject_code, val_codes)
        train_mask = ~(test_mask | val_mask)
        if not test_mask.any() or not val_mask.any() or not train_mask.any():
            raise ValueError(
                f"Fold {fold_idx} has empty split: "
                f"train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
            )

        fold_dir = output_dir / f"loso_subject_{fold_idx:02d}"
        save_npz(fold_dir / "train.npz", args.compress, **split_payload(windows, train_mask))
        save_npz(fold_dir / "val.npz", args.compress, **split_payload(windows, val_mask))
        save_npz(fold_dir / "test.npz", args.compress, **split_payload(windows, test_mask))

        row: dict[str, Any] = {
            "fold": fold_idx,
            "test_subject": "|".join(subjects[test_codes].astype(str).tolist()),
            "val_subject": "|".join(subjects[val_codes].astype(str).tolist()),
            "train_windows": int(train_mask.sum()),
            "val_windows": int(val_mask.sum()),
            "test_windows": int(test_mask.sum()),
        }
        for split, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            counts = np.bincount(y[mask].astype(np.int64), minlength=len(class_names))
            for class_idx, class_name in enumerate(class_names.astype(str)):
                row[f"{split}_{class_name.lower()}"] = int(counts[class_idx])
        rows.append(row)

    write_summary(output_dir / "fold_export_summary.json", rows)
    print(f"Exported {len(rows)} LOSO folds to {output_dir}")


if __name__ == "__main__":
    main()
