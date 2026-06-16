#!/usr/bin/env python
"""Build LOSO window NPZ files from standardized processed FOG records.

Input processed directory contract:

- records/*.npz, each containing exactly:
  - x: [time, channel] float32
  - y_binary: [time] int8, 0=NORMAL, 1=FOG
- manifest.csv with record_id, record_path, subject_id, sampling_rate or sampling_rate_hz.
- config.json or schema.json with channel metadata.

The script does not modify sample-level records. It materializes one window
dataset for a chosen window length, stride, and label mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BINARY_CLASS_NAMES = np.array(["NORMAL", "FOG"])
THREE_CLASS_NAMES = np.array(["NORMAL", "PRE_FOG", "FOG"])
LABEL_NORMAL = 0
LABEL_PRE_FOG = 1
LABEL_FOG_BINARY = 1
LABEL_FOG_THREE = 2


@dataclass(frozen=True)
class Record:
    dataset_id: str
    record_id: str
    path: Path
    subject: str
    source: str
    session_id: str
    task_id: str
    segment_id: str
    hz: float
    n_samples: int


@dataclass
class RecordSummary:
    dataset_id: str
    record_id: str
    subject: str
    source: str
    session_id: str
    task_id: str
    segment_id: str
    sampling_rate_hz: float
    samples: int
    windows: int
    normal_windows: int
    pre_fog_windows: int
    fog_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build window-level LOSO NPZ files from standardized processed records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing records/, manifest.csv, and config.json or schema.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for windows.npz, loso_folds.npz, and summaries.",
    )
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument(
        "--stride-seconds",
        type=float,
        help="Window stride in seconds. If omitted, --overlap is used.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio used only when --stride-seconds is omitted.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("binary", "three-class"),
        default="three-class",
        help="binary gives NORMAL/FOG. three-class derives PRE_FOG before each FOG onset.",
    )
    parser.add_argument(
        "--pre-fog-seconds",
        type=float,
        default=3.0,
        help="Duration before each FOG onset labeled PRE_FOG in three-class mode.",
    )
    parser.add_argument(
        "--label-rule",
        choices=("priority", "center", "majority"),
        default="priority",
        help="Rule for reducing sample labels to one window label.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=0.0,
        help="Resample each window to this Hz. 0 keeps native sampling.",
    )
    parser.add_argument(
        "--nan-policy",
        choices=("error", "zero"),
        default="error",
        help="How to handle NaN values in x before windowing. Infinite values always fail.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help="0 means one test subject per LOSO fold. >=2 groups subjects into this many folds.",
    )
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument(
        "--max-records",
        type=int,
        help="Debug option to process only the first N manifest records.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Require processed/_SUCCESS.json with status=complete before reading records.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write summaries/config only; do not create windows.npz.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed. Smaller output, slower to write/read.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing output directory.",
    )
    return parser.parse_args()


def parse_bool(value: object, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def read_schema(processed_dir: Path) -> tuple[dict, Path]:
    schema_path = processed_dir / "schema.json"
    config_path = processed_dir / "config.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8")), schema_path
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8")), config_path
    raise FileNotFoundError(f"Missing config.json or schema.json under {processed_dir}")


def channel_names(schema: dict) -> list[str]:
    channels = schema.get("channels", [])
    names = []
    for idx, channel in enumerate(channels):
        if isinstance(channel, dict):
            names.append(str(channel.get("name", f"ch{idx:03d}")))
        else:
            names.append(str(channel))
    return names


def load_records(processed_dir: Path, max_records: int | None) -> list[Record]:
    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv: {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    if "usable" in manifest.columns:
        manifest = manifest[manifest["usable"].map(parse_bool)].copy()
    if max_records:
        manifest = manifest.head(max_records).copy()
    if manifest.empty:
        raise ValueError(f"No usable records found in {manifest_path}")

    records: list[Record] = []
    for row in manifest.itertuples(index=False):
        data = row._asdict()
        record_path = Path(data.get("record_path", ""))
        if not record_path.is_absolute():
            record_path = processed_dir / record_path
        hz = float(data.get("sampling_rate_hz") or data.get("sampling_rate") or 0)
        samples = int(float(data.get("n_samples") or 0))
        if hz <= 0:
            raise ValueError(f"Invalid sampling_rate_hz for {data.get('record_id')}: {hz}")
        records.append(
            Record(
                dataset_id=str(data.get("dataset_id", "")),
                record_id=str(data.get("record_id", record_path.stem)),
                path=record_path,
                subject=str(data.get("subject_id", "")),
                source=str(data.get("source_file", "")),
                session_id=str(data.get("session_id", "")),
                task_id=str(data.get("task_id", "")),
                segment_id=str(data.get("segment_id", "")),
                hz=hz,
                n_samples=samples,
            )
        )
    return records


def window_params(hz: float, args: argparse.Namespace) -> tuple[int, int, int, float]:
    window_size = int(round(args.window_seconds * hz))
    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    if args.stride_seconds is not None:
        stride = int(round(args.stride_seconds * hz))
    else:
        if not (0.0 <= args.overlap < 1.0):
            raise ValueError("--overlap must be in [0, 1).")
        stride = int(round(window_size * (1.0 - args.overlap)))
    stride = max(1, stride)

    target_hz = float(args.target_hz) if args.target_hz and args.target_hz > 0 else hz
    target_len = int(round(args.window_seconds * target_hz))
    if target_len <= 0:
        raise ValueError("Output window length must be positive.")
    return window_size, stride, target_len, target_hz


def window_spans(n_samples: int, window_size: int, stride: int) -> Iterable[tuple[int, int]]:
    last_start = n_samples - window_size
    if last_start < 0:
        return
    for start in range(0, last_start + 1, stride):
        yield start, start + window_size


def fog_intervals(fog: np.ndarray) -> list[tuple[int, int]]:
    if not fog.any():
        return []
    padded = np.r_[False, fog, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def build_sample_state(
    y_binary: np.ndarray,
    hz: float,
    label_mode: str,
    pre_fog_seconds: float,
) -> np.ndarray:
    y_binary = np.asarray(y_binary).reshape(-1)
    fog = y_binary.astype(np.int8) > 0

    if label_mode == "binary":
        state = np.zeros(len(y_binary), dtype=np.int8)
        state[fog] = LABEL_FOG_BINARY
        return state

    state = np.zeros(len(y_binary), dtype=np.int8)
    intervals = fog_intervals(fog)
    pre_samples = int(round(max(0.0, pre_fog_seconds) * hz))

    prev_fog_end = 0
    for start, end in intervals:
        if pre_samples > 0:
            pre_start = max(prev_fog_end, start - pre_samples)
            state[pre_start:start] = LABEL_PRE_FOG
        state[start:end] = LABEL_FOG_THREE
        prev_fog_end = end
    return state


def label_window(values: np.ndarray, label_mode: str, rule: str) -> int:
    if rule == "center":
        return int(values[len(values) // 2])

    num_classes = 2 if label_mode == "binary" else 3
    if rule == "majority":
        return int(np.bincount(values.astype(np.int64), minlength=num_classes).argmax())

    if label_mode == "binary":
        return LABEL_FOG_BINARY if np.any(values == LABEL_FOG_BINARY) else LABEL_NORMAL

    if np.any(values == LABEL_FOG_THREE):
        return LABEL_FOG_THREE
    if np.any(values == LABEL_PRE_FOG):
        return LABEL_PRE_FOG
    return LABEL_NORMAL


def resample_window(window: np.ndarray, target_len: int) -> np.ndarray:
    if len(window) == target_len:
        return window.astype(np.float32, copy=False)

    source_positions = np.arange(len(window), dtype=np.float32)
    target_positions = np.linspace(0, len(window) - 1, target_len, dtype=np.float32)
    output = np.empty((target_len, window.shape[1]), dtype=np.float32)
    for col in range(window.shape[1]):
        output[:, col] = np.interp(target_positions, source_positions, window[:, col])
    return output


def load_record_arrays(record: Record, nan_policy: str) -> tuple[np.ndarray, np.ndarray]:
    if not record.path.exists():
        raise FileNotFoundError(f"Missing record file: {record.path}")
    with np.load(record.path) as data:
        keys = set(data.files)
        if keys != {"x", "y_binary"}:
            raise ValueError(f"{record.path} must contain exactly x and y_binary, got {sorted(keys)}")
        x = data["x"].astype(np.float32, copy=False)
        y = data["y_binary"].astype(np.int8, copy=False)
    if x.ndim != 2:
        raise ValueError(f"{record.path}: x must be 2D [time, channel], got {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"{record.path}: y_binary length {y.shape} does not match x {x.shape}")
    if np.isinf(x).any():
        raise ValueError(f"{record.path}: x contains infinite values")
    if np.isnan(x).any():
        if nan_policy == "error":
            raise ValueError(f"{record.path}: x contains NaN values; use --nan-policy zero if intended")
        x = np.nan_to_num(x, nan=0.0, copy=True).astype(np.float32, copy=False)
    return x, y


def summarize_record(record: Record, args: argparse.Namespace) -> RecordSummary:
    x, y_binary = load_record_arrays(record, args.nan_policy)
    window_size, stride, _, _ = window_params(record.hz, args)
    state = build_sample_state(y_binary, record.hz, args.label_mode, args.pre_fog_seconds)

    counts = np.zeros(3, dtype=np.int64)
    total = 0
    for start, end in window_spans(x.shape[0], window_size, stride):
        label = label_window(state[start:end], args.label_mode, args.label_rule)
        counts[label] += 1
        total += 1

    fog_label = LABEL_FOG_BINARY if args.label_mode == "binary" else LABEL_FOG_THREE
    return RecordSummary(
        dataset_id=record.dataset_id,
        record_id=record.record_id,
        subject=record.subject,
        source=record.source,
        session_id=record.session_id,
        task_id=record.task_id,
        segment_id=record.segment_id,
        sampling_rate_hz=record.hz,
        samples=int(x.shape[0]),
        windows=int(total),
        normal_windows=int(counts[LABEL_NORMAL]),
        pre_fog_windows=0 if args.label_mode == "binary" else int(counts[LABEL_PRE_FOG]),
        fog_windows=int(counts[fog_label]),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def allocate_arrays(
    output_dir: Path,
    total_windows: int,
    target_len: int,
    n_channels: int,
) -> tuple[Path, np.memmap, dict[str, np.ndarray]]:
    tmp_dir = output_dir / "_tmp_arrays"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    x_path = tmp_dir / "X.npy"
    x = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_windows, target_len, n_channels),
    )
    arrays = {
        "y": np.empty(total_windows, dtype=np.int8),
        "subject": np.empty(total_windows, dtype="U32"),
        "source": np.empty(total_windows, dtype="U256"),
        "file_id": np.empty(total_windows, dtype="U64"),
        "session_id": np.empty(total_windows, dtype="U64"),
        "task_id": np.empty(total_windows, dtype="U64"),
        "segment_id": np.empty(total_windows, dtype="U64"),
        "start_sample": np.empty(total_windows, dtype=np.int64),
        "end_sample": np.empty(total_windows, dtype=np.int64),
        "native_hz": np.empty(total_windows, dtype=np.float32),
    }
    return tmp_dir, x, arrays


def fill_record_windows(
    record: Record,
    args: argparse.Namespace,
    x_out: np.ndarray,
    arrays: dict[str, np.ndarray],
    offset: int,
) -> int:
    features, y_binary = load_record_arrays(record, args.nan_policy)
    window_size, stride, target_len, _ = window_params(record.hz, args)
    state = build_sample_state(y_binary, record.hz, args.label_mode, args.pre_fog_seconds)

    index = offset
    for start, end in window_spans(features.shape[0], window_size, stride):
        x_out[index] = resample_window(features[start:end], target_len)
        arrays["y"][index] = label_window(state[start:end], args.label_mode, args.label_rule)
        arrays["subject"][index] = record.subject
        arrays["source"][index] = record.source
        arrays["file_id"][index] = record.record_id
        arrays["session_id"][index] = record.session_id
        arrays["task_id"][index] = record.task_id
        arrays["segment_id"][index] = record.segment_id
        arrays["start_sample"][index] = int(start)
        arrays["end_sample"][index] = int(end)
        arrays["native_hz"][index] = float(record.hz)
        index += 1
    return index


def build_subject_groups(
    subjects: np.ndarray,
    window_subject_code: np.ndarray,
    y: np.ndarray,
    num_folds: int,
    fold_seed: int,
    num_classes: int,
) -> list[np.ndarray]:
    if num_folds <= 0 or num_folds >= len(subjects):
        return [np.array([subject], dtype="U32") for subject in subjects]
    if num_folds < 2:
        raise ValueError("--num-folds must be 0 or at least 2.")

    counts = np.zeros((len(subjects), num_classes), dtype=np.int64)
    for subject_code in range(len(subjects)):
        mask = window_subject_code == subject_code
        counts[subject_code] = np.bincount(y[mask], minlength=num_classes)

    target = counts.sum(axis=0).astype(np.float64) / float(num_folds)
    denom = np.maximum(target, 1.0)
    fold_counts = np.zeros((num_folds, num_classes), dtype=np.float64)
    fold_groups: list[list[int]] = [[] for _ in range(num_folds)]

    rng = np.random.default_rng(fold_seed)
    tie_breaker = rng.random(len(subjects))
    order = sorted(
        range(len(subjects)),
        key=lambda idx: (
            -int(counts[idx].max(initial=0)),
            -int(counts[idx].sum()),
            float(tie_breaker[idx]),
            str(subjects[idx]),
        ),
    )

    for fold, subject_code in enumerate(order[:num_folds]):
        fold_groups[fold].append(subject_code)
        fold_counts[fold] += counts[subject_code].astype(np.float64)

    for subject_code in order[num_folds:]:
        subject_counts = counts[subject_code].astype(np.float64)
        best_fold = min(
            range(num_folds),
            key=lambda fold: (
                float(np.square((fold_counts[fold] + subject_counts - target) / denom).sum()),
                len(fold_groups[fold]),
                fold,
            ),
        )
        fold_groups[best_fold].append(subject_code)
        fold_counts[best_fold] += subject_counts

    return [
        np.array([subjects[idx] for idx in sorted(group)], dtype="U32")
        for group in fold_groups
    ]


def padded_subject_array(groups: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(group) for group in groups)
    out = np.full((len(groups), max_len), "", dtype="U32")
    for fold, group in enumerate(groups):
        out[fold, : len(group)] = group
    return out


def padded_code_array(groups: list[np.ndarray], subject_to_code: dict[str, int]) -> np.ndarray:
    max_len = max(len(group) for group in groups)
    out = np.full((len(groups), max_len), -1, dtype=np.int16)
    for fold, group in enumerate(groups):
        out[fold, : len(group)] = [subject_to_code[str(subject)] for subject in group]
    return out


def write_loso_folds(
    output_dir: Path,
    subjects: np.ndarray,
    window_subject_code: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    config: dict,
    num_folds: int,
    fold_seed: int,
) -> None:
    subject_to_code = {str(subject): idx for idx, subject in enumerate(subjects)}
    test_groups = build_subject_groups(
        subjects=subjects,
        window_subject_code=window_subject_code,
        y=y,
        num_folds=num_folds,
        fold_seed=fold_seed,
        num_classes=len(class_names),
    )
    val_groups = [test_groups[(idx + 1) % len(test_groups)] for idx in range(len(test_groups))]

    grouped = not (num_folds <= 0 or num_folds >= len(subjects))
    if grouped:
        fold_test_subjects = padded_subject_array(test_groups)
        fold_val_subjects = padded_subject_array(val_groups)
    else:
        fold_test_subjects = np.array([group[0] for group in test_groups], dtype="U32")
        fold_val_subjects = np.array([group[0] for group in val_groups], dtype="U32")

    fold_test_subject_codes = padded_code_array(test_groups, subject_to_code)
    fold_val_subject_codes = padded_code_array(val_groups, subject_to_code)

    np.savez(
        output_dir / "loso_folds.npz",
        subjects=subjects,
        fold_test_subjects=fold_test_subjects,
        fold_val_subjects=fold_val_subjects,
        fold_test_subject_codes=fold_test_subject_codes,
        fold_val_subject_codes=fold_val_subject_codes,
        window_subject_code=window_subject_code,
        class_names=class_names,
        config_json=np.array(json.dumps(config, ensure_ascii=False)),
    )

    rows: list[dict] = []
    for fold_idx, (test_group, val_group) in enumerate(zip(test_groups, val_groups)):
        test_codes = np.array([subject_to_code[str(subject)] for subject in test_group])
        val_codes = np.array([subject_to_code[str(subject)] for subject in val_group])
        test_mask = np.isin(window_subject_code, test_codes)
        val_mask = np.isin(window_subject_code, val_codes)
        train_mask = ~(test_mask | val_mask)

        row = {
            "fold": fold_idx,
            "test_subject": "|".join(str(subject) for subject in test_group),
            "val_subject": "|".join(str(subject) for subject in val_group),
            "test_subject_count": int(len(test_group)),
            "val_subject_count": int(len(val_group)),
            "train_windows": int(train_mask.sum()),
            "val_windows": int(val_mask.sum()),
            "test_windows": int(test_mask.sum()),
        }
        for split, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            counts = np.bincount(y[mask], minlength=len(class_names))
            for idx, name in enumerate(class_names):
                row[f"{split}_{str(name).lower()}"] = int(counts[idx])
        rows.append(row)

    write_csv(output_dir / "loso_folds.csv", rows)


def save_npz(path: Path, compress: bool, **arrays: np.ndarray) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def write_readme(output_dir: Path, class_names: np.ndarray) -> None:
    classes = "\n".join(f"- `{idx}`: {name}" for idx, name in enumerate(class_names))
    text = f"""# Processed Record Windows

Files:

- `windows.npz`: materialized fixed-length windows and window metadata.
- `loso_folds.npz`: compact LOSO fold metadata.
- `loso_folds.csv`: fold-level class counts for inspection.
- `file_summary.csv`: record-level window/class counts.
- `config.json`: exact generation parameters.

Class mapping:

{classes}

`X` has shape `[window, time, channel]`. `y` has one label per window.
`subject_code` and `loso_folds.npz/window_subject_code` can be used to split
without loading string metadata.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        "windows.npz",
        "loso_folds.npz",
        "loso_folds.csv",
        "file_summary.csv",
        "config.json",
    ]
    existing = [name for name in protected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {existing}. Use --overwrite."
        )


def require_success_marker(processed_dir: Path) -> None:
    success_path = processed_dir / "_SUCCESS.json"
    if not success_path.exists():
        raise FileNotFoundError(f"Missing _SUCCESS.json: {success_path}")
    marker = json.loads(success_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError(f"{success_path} status is not complete: {marker.get('status')}")


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive.")
    if args.stride_seconds is not None and args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be positive.")
    if args.pre_fog_seconds < 0:
        raise ValueError("--pre-fog-seconds must be non-negative.")

    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.processed_dir = processed_dir
    args.output_dir = output_dir
    if args.require_success:
        require_success_marker(processed_dir)
    ensure_output_dir(output_dir, args.overwrite)

    schema, schema_source_path = read_schema(processed_dir)
    features = channel_names(schema)
    if not features:
        raise ValueError(f"No channels found in {schema_source_path}")

    records = load_records(processed_dir, args.max_records)
    if len({record.subject for record in records}) < 2:
        raise ValueError("At least two subjects are needed for LOSO folds.")

    class_names = BINARY_CLASS_NAMES if args.label_mode == "binary" else THREE_CLASS_NAMES
    first_window_size, first_stride, first_target_len, first_target_hz = window_params(
        records[0].hz, args
    )
    for record in records[1:]:
        _, _, target_len, _ = window_params(record.hz, args)
        if target_len != first_target_len:
            raise ValueError(
                "Records produce different target lengths. Set --target-hz to a common value."
            )

    config = {
        "processed_dir": str(processed_dir),
        "dataset_id": schema.get("dataset_id", schema.get("dataset_name", "")),
        "label_mode": args.label_mode,
        "label_rule": args.label_rule,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "overlap": args.overlap if args.stride_seconds is None else None,
        "pre_fog_seconds": args.pre_fog_seconds if args.label_mode == "three-class" else None,
        "target_hz": first_target_hz,
        "target_len": first_target_len,
        "nan_policy": args.nan_policy,
        "require_success": args.require_success,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "class_names": class_names.tolist(),
        "feature_names": features,
        "source_schema": str(schema_source_path),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Scanning {len(records)} records from {processed_dir} ...")
    summaries = [summarize_record(record, args) for record in records]
    write_csv(output_dir / "file_summary.csv", [asdict(summary) for summary in summaries])

    total_windows = int(sum(summary.windows for summary in summaries))
    counts = {
        "NORMAL": int(sum(summary.normal_windows for summary in summaries)),
        "PRE_FOG": int(sum(summary.pre_fog_windows for summary in summaries)),
        "FOG": int(sum(summary.fog_windows for summary in summaries)),
    }
    print(f"Total windows: {total_windows:,}")
    print("Label counts:", counts)

    if args.dry_run:
        write_readme(output_dir, class_names)
        print(f"Dry run complete. Summaries written to {output_dir}")
        return
    if total_windows <= 0:
        raise ValueError("No windows generated. Use shorter --window-seconds or stride.")

    tmp_dir, x_out, arrays = allocate_arrays(
        output_dir=output_dir,
        total_windows=total_windows,
        target_len=first_target_len,
        n_channels=len(features),
    )
    try:
        print("Writing window arrays...")
        offset = 0
        for idx, record in enumerate(records, start=1):
            offset = fill_record_windows(record, args, x_out, arrays, offset)
            if idx % 25 == 0 or idx == len(records):
                print(f"  wrote {idx}/{len(records)} records; {offset:,}/{total_windows:,} windows")
        if offset != total_windows:
            raise RuntimeError(f"Expected {total_windows} windows, wrote {offset}.")

        subjects = np.array(sorted({str(value) for value in arrays["subject"]}), dtype="U32")
        subject_to_code = {subject: idx for idx, subject in enumerate(subjects)}
        window_subject_code = np.array(
            [subject_to_code[str(subject)] for subject in arrays["subject"]],
            dtype=np.int16,
        )

        x_out.flush()
        save_npz(
            output_dir / "windows.npz",
            args.compress,
            X=x_out,
            y=arrays["y"],
            subject=arrays["subject"],
            subject_code=window_subject_code,
            subjects=subjects,
            source=arrays["source"],
            file_id=arrays["file_id"],
            session_id=arrays["session_id"],
            task_id=arrays["task_id"],
            segment_id=arrays["segment_id"],
            start_sample=arrays["start_sample"],
            end_sample=arrays["end_sample"],
            native_hz=arrays["native_hz"],
            class_names=class_names,
            feature_names=np.array(features),
            config_json=np.array(json.dumps(config, ensure_ascii=False)),
        )
        write_loso_folds(
            output_dir=output_dir,
            subjects=subjects,
            window_subject_code=window_subject_code,
            y=arrays["y"],
            class_names=class_names,
            config=config,
            num_folds=args.num_folds,
            fold_seed=args.fold_seed,
        )
        write_readme(output_dir, class_names)
        print(f"Done. Output written to {output_dir}")
    finally:
        del x_out
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
