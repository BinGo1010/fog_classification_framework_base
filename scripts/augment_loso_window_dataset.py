#!/usr/bin/env python
"""Create leakage-safe, class-balanced LOSO folds from window NPZ data.

Only training windows are augmented. Validation and test windows always remain
the original measured data. Minority classes are oversampled to the largest
class count in each training fold unless ``--target-count`` is specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export class-balanced LOSO folds using IMU augmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing windows.npz and loso_folds.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing loso_subject_XX split directories.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="Per-class training count. 0 uses the largest original class count per fold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rotation-max-degrees",
        type=float,
        default=10.0,
        help="Maximum absolute random 3-D rotation angle per tri-axial sensor.",
    )
    parser.add_argument("--rotation-prob", type=float, default=1.0)
    parser.add_argument(
        "--jitter-std-ratio",
        type=float,
        default=0.01,
        help="Gaussian jitter sigma as a fraction of each training channel's std.",
    )
    parser.add_argument("--jitter-prob", type=float, default=0.5)
    parser.add_argument("--scale-min", type=float, default=0.98)
    parser.add_argument("--scale-max", type=float, default=1.02)
    parser.add_argument("--scale-prob", type=float, default=0.5)
    parser.add_argument(
        "--time-warp-sigma",
        type=float,
        default=0.1,
        help="Log-normal speed variation for endpoint-preserving time warping.",
    )
    parser.add_argument(
        "--time-warp-prob",
        type=float,
        default=0.0,
        help="Disabled by default to preserve PRE_FOG/FOG temporal semantics.",
    )
    parser.add_argument("--time-warp-segments", type=int, default=4)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_args(args: argparse.Namespace) -> None:
    if args.target_count < 0:
        raise ValueError("--target-count must be non-negative.")
    if not 0 <= args.rotation_max_degrees <= 180:
        raise ValueError("--rotation-max-degrees must be in [0, 180].")
    if args.jitter_std_ratio < 0:
        raise ValueError("--jitter-std-ratio must be non-negative.")
    if args.scale_min <= 0 or args.scale_max <= 0 or args.scale_min > args.scale_max:
        raise ValueError("Require 0 < --scale-min <= --scale-max.")
    if args.time_warp_sigma < 0:
        raise ValueError("--time-warp-sigma must be non-negative.")
    if args.time_warp_segments < 2:
        raise ValueError("--time-warp-segments must be at least 2.")
    for name in ("rotation_prob", "jitter_prob", "scale_prob", "time_warp_prob"):
        validate_probability(f"--{name.replace('_', '-')}", float(getattr(args, name)))

    rotation_enabled = args.rotation_max_degrees > 0 and args.rotation_prob > 0
    jitter_enabled = args.jitter_std_ratio > 0 and args.jitter_prob > 0
    scale_enabled = (
        args.scale_prob > 0 and (args.scale_min != 1.0 or args.scale_max != 1.0)
    )
    warp_enabled = args.time_warp_sigma > 0 and args.time_warp_prob > 0
    if not (rotation_enabled or jitter_enabled or scale_enabled or warp_enabled):
        raise ValueError("At least one non-identity augmentation must be enabled.")


def sample_parent_indices(
    indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample with replacement while using every source once per random cycle."""
    indices = np.asarray(indices, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Cannot augment a class with no original training windows.")
    cycles, remainder = divmod(count, int(indices.size))
    parts = [rng.permutation(indices) for _ in range(cycles)]
    if remainder:
        parts.append(rng.permutation(indices)[:remainder])
    return np.concatenate(parts).astype(np.int64, copy=False)


def time_warp(
    window: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    segments: int,
) -> np.ndarray:
    """Apply a smooth monotonic time warp while preserving both endpoints."""
    n_samples = int(window.shape[0])
    if n_samples < 3 or sigma <= 0:
        return window
    target_knots = np.linspace(0.0, n_samples - 1.0, segments + 1)
    speeds = rng.lognormal(mean=0.0, sigma=sigma, size=segments)
    source_knots = np.r_[0.0, np.cumsum(speeds)]
    source_knots *= (n_samples - 1.0) / source_knots[-1]
    source_positions = np.interp(
        np.arange(n_samples, dtype=np.float64), target_knots, source_knots
    )
    source_axis = np.arange(n_samples, dtype=np.float64)
    warped = np.empty_like(window, dtype=np.float32)
    for channel in range(window.shape[1]):
        warped[:, channel] = np.interp(
            source_positions,
            source_axis,
            window[:, channel],
        ).astype(np.float32)
    return warped


def rotation_matrix(
    rng: np.random.Generator,
    max_degrees: float,
) -> np.ndarray:
    """Sample a 3-D axis-angle rotation matrix."""
    axis = rng.normal(size=3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    angle = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    identity = np.eye(3, dtype=np.float64)
    return (
        identity * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    ).astype(np.float32)


def rotate_sensor_triads(
    window: np.ndarray,
    rng: np.random.Generator,
    max_degrees: float,
) -> np.ndarray:
    """Rotate each complete xyz sensor triad independently."""
    out = np.asarray(window, dtype=np.float32).copy()
    for start in range(0, out.shape[1] - 2, 3):
        out[:, start : start + 3] = out[:, start : start + 3] @ rotation_matrix(
            rng, max_degrees
        ).T
    return out


def augment_window(
    window: np.ndarray,
    rng: np.random.Generator,
    channel_std: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    out = np.asarray(window, dtype=np.float32).copy()
    apply_rotation = (
        args.rotation_max_degrees > 0
        and args.rotation_prob > 0
        and rng.random() < args.rotation_prob
    )
    apply_scale = args.scale_prob > 0 and rng.random() < args.scale_prob
    apply_jitter = (
        args.jitter_std_ratio > 0
        and args.jitter_prob > 0
        and rng.random() < args.jitter_prob
    )
    apply_warp = (
        args.time_warp_sigma > 0
        and args.time_warp_prob > 0
        and rng.random() < args.time_warp_prob
    )

    available: list[str] = []
    if args.rotation_max_degrees > 0 and args.rotation_prob > 0:
        available.append("rotation")
    if args.scale_prob > 0 and (args.scale_min != 1.0 or args.scale_max != 1.0):
        available.append("scale")
    if args.jitter_std_ratio > 0 and args.jitter_prob > 0:
        available.append("jitter")
    if args.time_warp_sigma > 0 and args.time_warp_prob > 0:
        available.append("time_warp")
    if not (apply_rotation or apply_scale or apply_jitter or apply_warp):
        forced = str(rng.choice(available))
        apply_rotation = forced == "rotation"
        apply_scale = forced == "scale"
        apply_jitter = forced == "jitter"
        apply_warp = forced == "time_warp"

    methods: list[str] = []
    if apply_rotation:
        out = rotate_sensor_triads(out, rng, args.rotation_max_degrees)
        methods.append("rotation")
    if apply_warp:
        out = time_warp(out, rng, args.time_warp_sigma, args.time_warp_segments)
        methods.append("time_warp")
    if apply_scale:
        n_groups = (out.shape[1] + 2) // 3
        group_scale = rng.uniform(args.scale_min, args.scale_max, size=n_groups)
        scale = np.repeat(group_scale, 3)[: out.shape[1]].astype(np.float32)
        out *= scale[None, :]
        methods.append("scale")
    if apply_jitter:
        sigma = np.asarray(channel_std, dtype=np.float32) * float(args.jitter_std_ratio)
        noise = rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
        out += noise * sigma[None, :]
        methods.append("jitter")

    if not np.isfinite(out).all():
        raise ValueError("Augmentation produced non-finite values.")
    return out.astype(np.float32, copy=False), "+".join(methods)


def window_aligned_keys(windows: dict[str, np.ndarray], n_windows: int) -> list[str]:
    return [
        key
        for key, value in windows.items()
        if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == n_windows
    ]


def original_split_payload(
    windows: dict[str, np.ndarray],
    indices: np.ndarray,
    augmentation_config_json: str,
) -> dict[str, np.ndarray]:
    n_windows = int(np.asarray(windows["y"]).shape[0])
    aligned = set(window_aligned_keys(windows, n_windows))
    ignored = {"is_augmented", "parent_index", "augmentation"}
    payload: dict[str, np.ndarray] = {}
    for key, value in windows.items():
        if key in ignored:
            continue
        array = np.asarray(value)
        payload[key] = array[indices] if key in aligned else array
    payload["is_augmented"] = np.zeros(len(indices), dtype=bool)
    payload["parent_index"] = np.asarray(indices, dtype=np.int64)
    payload["augmentation"] = np.full(len(indices), "original", dtype="U64")
    payload["augmentation_config_json"] = np.array(augmentation_config_json)
    if "sensor_columns" not in payload and "feature_names" in payload:
        payload["sensor_columns"] = np.asarray(payload["feature_names"])
    return payload


def balanced_train_payload(
    windows: dict[str, np.ndarray],
    train_indices: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    augmentation_config_json: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    x = np.asarray(windows["X"], dtype=np.float32)
    y = np.asarray(windows["y"], dtype=np.int64).reshape(-1)
    class_names = np.asarray(windows["class_names"]).astype(str).reshape(-1)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    original_counts = np.bincount(y[train_indices], minlength=len(class_names)).astype(int)
    if np.any(original_counts == 0):
        missing = class_names[original_counts == 0].tolist()
        raise ValueError(f"Training fold has no original windows for class(es): {missing}")
    if target_count < int(original_counts.max()):
        raise ValueError(
            f"Target count {target_count} is below largest original class count "
            f"{int(original_counts.max())}; downsampling is intentionally disabled."
        )

    channel_std = x[train_indices].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    channel_std = np.maximum(channel_std, np.float32(1e-6))
    synthetic_parents: list[np.ndarray] = []
    synthetic_x: list[np.ndarray] = []
    synthetic_methods: list[str] = []
    synthetic_by_class: dict[str, int] = {}
    for class_idx, class_name in enumerate(class_names):
        deficit = int(target_count - original_counts[class_idx])
        synthetic_by_class[str(class_name)] = deficit
        parents = sample_parent_indices(
            train_indices[y[train_indices] == class_idx], deficit, rng
        )
        if parents.size:
            class_x = np.empty((len(parents), *x.shape[1:]), dtype=np.float32)
            for output_idx, parent_idx in enumerate(parents):
                class_x[output_idx], method = augment_window(
                    x[int(parent_idx)], rng, channel_std, args
                )
                synthetic_methods.append(method)
            synthetic_parents.append(parents)
            synthetic_x.append(class_x)

    if synthetic_parents:
        parent_indices = np.concatenate(synthetic_parents).astype(np.int64, copy=False)
        x_synthetic = np.concatenate(synthetic_x, axis=0)
    else:
        parent_indices = np.empty(0, dtype=np.int64)
        x_synthetic = np.empty((0, *x.shape[1:]), dtype=np.float32)

    all_parent_indices = np.r_[train_indices, parent_indices].astype(np.int64, copy=False)
    n_windows = int(y.shape[0])
    aligned = set(window_aligned_keys(windows, n_windows))
    ignored = {"is_augmented", "parent_index", "augmentation"}
    payload: dict[str, np.ndarray] = {}
    for key, value in windows.items():
        if key in ignored:
            continue
        array = np.asarray(value)
        if key not in aligned:
            payload[key] = array
        elif key == "X":
            payload[key] = np.concatenate([x[train_indices], x_synthetic], axis=0)
        else:
            payload[key] = array[all_parent_indices]

    n_original = len(train_indices)
    n_synthetic = len(parent_indices)
    payload["is_augmented"] = np.r_[
        np.zeros(n_original, dtype=bool), np.ones(n_synthetic, dtype=bool)
    ]
    payload["parent_index"] = all_parent_indices
    payload["augmentation"] = np.r_[
        np.full(n_original, "original", dtype="U64"),
        np.asarray(synthetic_methods, dtype="U64"),
    ]
    payload["augmentation_config_json"] = np.array(augmentation_config_json)
    if "sensor_columns" not in payload and "feature_names" in payload:
        payload["sensor_columns"] = np.asarray(payload["feature_names"])

    permutation = rng.permutation(len(all_parent_indices))
    for key, value in list(payload.items()):
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == len(all_parent_indices):
            payload[key] = array[permutation]

    final_counts = np.bincount(
        np.asarray(payload["y"], dtype=np.int64), minlength=len(class_names)
    )
    if not np.all(final_counts == target_count):
        raise RuntimeError(
            f"Balanced train counts are {final_counts.tolist()}, expected {target_count}."
        )
    return payload, synthetic_by_class


def save_npz(path: Path, compress: bool, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir: Path) -> None:
    text = """# Leakage-safe balanced LOSO IMU dataset

Each `loso_subject_XX` directory contains:

- `train.npz`: original training windows plus augmented minority-class windows.
- `val.npz`: original validation windows only.
- `test.npz`: original held-out test windows only.

Training classes are balanced to 1:1:1 independently in every fold. Synthetic
windows use small independent 3-D rotations for each sensor triad, plus optional
weak sensor-triad scaling and Gaussian jitter based only on that fold's
training-channel standard deviations. Time warping is available but disabled
by default to preserve PRE_FOG/FOG temporal semantics. No validation or test
statistics are used by augmentation.

Audit arrays in each split:

- `is_augmented`: true only for synthetic training windows.
- `parent_index`: index of the original source window in the input `windows.npz`.
- `augmentation`: applied transformation names (`original` for measured data).

Do not combine the materialized folds before evaluation: the same original
window can legitimately appear in the training sets of multiple LOSO folds.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if data_dir == output_dir:
        raise ValueError("--output-dir must differ from --data-dir.")
    if output_dir.exists():
        existing = list(output_dir.glob("loso_subject_*/train.npz"))
        if existing and not args.overwrite:
            raise FileExistsError(f"Output folds already exist in {output_dir}; use --overwrite.")
        if args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_npz(data_dir / "windows.npz")
    folds = load_npz(data_dir / "loso_folds.npz")
    required = {"X", "y", "subject_code", "subjects", "class_names"}
    missing = sorted(required - set(windows))
    if missing:
        raise KeyError(f"windows.npz is missing required arrays: {missing}")

    x = np.asarray(windows["X"], dtype=np.float32)
    y = np.asarray(windows["y"], dtype=np.int64).reshape(-1)
    subject_code = np.asarray(windows["subject_code"], dtype=np.int64).reshape(-1)
    subjects = np.asarray(windows["subjects"]).astype(str).reshape(-1)
    class_names = np.asarray(windows["class_names"]).astype(str).reshape(-1)
    if x.ndim != 3 or len(y) != len(x) or len(subject_code) != len(x):
        raise ValueError(
            f"Invalid aligned shapes: X={x.shape}, y={y.shape}, subject_code={subject_code.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("Input X contains non-finite values.")
    if set(np.unique(y).tolist()) != set(range(len(class_names))):
        raise ValueError("Every class name must have at least one input window.")

    if "fold_test_subject_codes" in folds:
        test_groups = normalize_code_groups(folds["fold_test_subject_codes"])
    else:
        test_groups = subject_groups_to_codes(
            normalize_subject_groups(folds["fold_test_subjects"]), subjects
        )
    if "fold_val_subject_codes" in folds:
        val_groups = normalize_code_groups(folds["fold_val_subject_codes"])
    else:
        val_groups = subject_groups_to_codes(
            normalize_subject_groups(folds["fold_val_subjects"]), subjects
        )
    if len(test_groups) != len(val_groups):
        raise ValueError("Test and validation fold counts differ.")

    augmentation_config = {
        "source_data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "policy": "balance each LOSO training split to its largest original class",
        "validation_test_policy": "original windows only",
        "target_count": args.target_count,
        "seed": args.seed,
        "rotation_max_degrees": args.rotation_max_degrees,
        "rotation_prob": args.rotation_prob,
        "jitter_std_ratio": args.jitter_std_ratio,
        "jitter_prob": args.jitter_prob,
        "scale_range": [args.scale_min, args.scale_max],
        "scale_prob": args.scale_prob,
        "time_warp_sigma": args.time_warp_sigma,
        "time_warp_prob": args.time_warp_prob,
        "time_warp_segments": args.time_warp_segments,
        "compress": args.compress,
        "class_names": class_names.tolist(),
    }
    augmentation_config_json = json.dumps(augmentation_config, ensure_ascii=False)
    (output_dir / "augmentation_config.json").write_text(
        json.dumps(augmentation_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Loading {len(x):,} original windows from {data_dir}")
    rows: list[dict[str, Any]] = []
    aggregate_train_counts = np.zeros(len(class_names), dtype=np.int64)
    aggregate_augmented = 0
    for fold_idx, (test_codes, val_codes) in enumerate(zip(test_groups, val_groups), start=1):
        if np.intersect1d(test_codes, val_codes).size:
            raise ValueError(f"Fold {fold_idx} overlaps test and validation subjects.")
        test_indices = np.flatnonzero(np.isin(subject_code, test_codes))
        val_indices = np.flatnonzero(np.isin(subject_code, val_codes))
        train_indices = np.flatnonzero(
            ~np.isin(subject_code, np.r_[test_codes, val_codes])
        )
        if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
            raise ValueError(
                f"Fold {fold_idx} has an empty split: train={len(train_indices)}, "
                f"val={len(val_indices)}, test={len(test_indices)}"
            )

        original_train_counts = np.bincount(
            y[train_indices], minlength=len(class_names)
        ).astype(int)
        target_count = args.target_count or int(original_train_counts.max())
        rng = np.random.default_rng(args.seed + fold_idx * 100_003)
        train_payload, synthetic_by_class = balanced_train_payload(
            windows=windows,
            train_indices=train_indices,
            target_count=target_count,
            rng=rng,
            args=args,
            augmentation_config_json=augmentation_config_json,
        )
        val_payload = original_split_payload(windows, val_indices, augmentation_config_json)
        test_payload = original_split_payload(windows, test_indices, augmentation_config_json)

        fold_dir = output_dir / f"loso_subject_{fold_idx:02d}"
        save_npz(fold_dir / "train.npz", args.compress, train_payload)
        save_npz(fold_dir / "val.npz", args.compress, val_payload)
        save_npz(fold_dir / "test.npz", args.compress, test_payload)

        train_counts = np.bincount(
            np.asarray(train_payload["y"], dtype=np.int64), minlength=len(class_names)
        )
        val_counts = np.bincount(y[val_indices], minlength=len(class_names))
        test_counts = np.bincount(y[test_indices], minlength=len(class_names))
        n_augmented = int(np.asarray(train_payload["is_augmented"]).sum())
        aggregate_train_counts += train_counts
        aggregate_augmented += n_augmented
        row: dict[str, Any] = {
            "fold": fold_idx,
            "test_subject": "|".join(subjects[test_codes].tolist()),
            "val_subject": "|".join(subjects[val_codes].tolist()),
            "original_train_windows": int(len(train_indices)),
            "augmented_train_windows": n_augmented,
            "balanced_train_windows": int(len(train_payload["y"])),
            "val_windows": int(len(val_indices)),
            "test_windows": int(len(test_indices)),
            "target_per_class": int(target_count),
        }
        for class_idx, class_name in enumerate(class_names):
            key = str(class_name).lower()
            row[f"original_train_{key}"] = int(original_train_counts[class_idx])
            row[f"synthetic_train_{key}"] = int(synthetic_by_class[str(class_name)])
            row[f"balanced_train_{key}"] = int(train_counts[class_idx])
            row[f"val_{key}"] = int(val_counts[class_idx])
            row[f"test_{key}"] = int(test_counts[class_idx])
        rows.append(row)
        print(
            f"  fold {fold_idx:02d}: train={len(train_payload['y']):,} "
            f"({n_augmented:,} synthetic), val={len(val_indices):,}, "
            f"test={len(test_indices):,}, per_class={target_count:,}"
        )

    write_csv(output_dir / "fold_summary.csv", rows)
    total_aggregate_train = int(aggregate_train_counts.sum())
    report = {
        "source_data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "fold_count": len(rows),
        "note": "Counts aggregate materialized training folds; windows repeat across LOSO folds by design.",
        "aggregate_balanced_train_windows": total_aggregate_train,
        "aggregate_synthetic_train_windows": int(aggregate_augmented),
        "aggregate_class_counts": {
            str(name): int(aggregate_train_counts[idx])
            for idx, name in enumerate(class_names)
        },
        "aggregate_class_percent": {
            str(name): float(aggregate_train_counts[idx] / total_aggregate_train * 100.0)
            for idx, name in enumerate(class_names)
        },
        "original_global_class_counts": {
            str(name): int(count)
            for name, count in zip(
                class_names, np.bincount(y, minlength=len(class_names)).tolist()
            )
        },
        "folds": rows,
    }
    (output_dir / "augmentation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(output_dir)
    print(
        "Aggregate balanced training counts:",
        {str(name): int(aggregate_train_counts[idx]) for idx, name in enumerate(class_names)},
    )
    print(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
    main()
