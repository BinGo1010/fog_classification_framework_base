#!/usr/bin/env python
"""Prepare LOSO NPZ windows for FOG/NORMAL/PRE_FOG classification.

This script processes the Kaggle TLVMC Parkinson's FOG train/tdcsfog and
train/defog CSV files into fixed-length time windows. It creates a compact
window dataset plus LOSO fold metadata. Optionally, it can also export one
materialized train/val/test fold as separate NPZ files.
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


FEATURE_COLUMNS = ("AccV", "AccML", "AccAP")
EVENT_COLUMNS = ("StartHesitation", "Turn", "Walking")
DATASET_HZ = {"tdcsfog": 128, "defog": 100}
CLASS_NAMES = np.array(["NORMAL", "PRE_FOG", "FOG"])
LABEL_NORMAL = 0
LABEL_PRE_FOG = 1
LABEL_FOG = 2


@dataclass(frozen=True)
class FileRecord:
    source: str
    file_id: str
    path: Path
    subject: str
    hz: int


@dataclass
class FileSummary:
    source: str
    file_id: str
    subject: str
    rows: int
    valid_rows: int
    windows: int
    normal_windows: int
    pre_fog_windows: int
    fog_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FOG/NORMAL/PRE_FOG LOSO NPZ windows from tdcsfog/defog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/competition data/Competition Dataset"),
        help="Competition Dataset directory containing train/, metadata, and tasks.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/fog_loso_npz"),
        help="Directory for windows.npz, loso_folds.npz, and summaries.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(DATASET_HZ),
        default=["tdcsfog", "defog"],
        help="Source folders under train/ to process.",
    )
    parser.add_argument(
        "--fog-columns",
        default="StartHesitation,Turn,Walking",
        help=(
            "Comma-separated event columns treated as FOG. Use Walking for "
            "walking-only FOG, or the default to merge all event types."
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--pre-fog-seconds", type=float, default=3.0)
    parser.add_argument(
        "--target-hz",
        type=int,
        default=128,
        help="Each output window is resampled to this many samples per second.",
    )
    parser.add_argument(
        "--label-rule",
        choices=("priority", "center", "majority"),
        default="priority",
        help="Rule for converting per-sample labels into one window label.",
    )
    parser.add_argument(
        "--defog-valid-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For defog, keep only rows where Valid and Task are true.",
    )
    parser.add_argument(
        "--exclude-rest-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For defog, use tasks.csv to remove Rest1/Rest2 intervals.",
    )
    parser.add_argument(
        "--heldout-subject",
        help="Optionally export one materialized LOSO fold for this test subject.",
    )
    parser.add_argument(
        "--val-subject",
        help="Validation subject for --heldout-subject. Defaults to the next subject.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help=(
            "Subject-independent grouped LOSO folds. 0 keeps the original "
            "one-subject-per-fold LOSO; e.g. 10 creates 10 test subject groups "
            "and uses the next group as validation."
        ),
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=42,
        help="Seed used only for deterministic tie-breaking in grouped folds.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed. Smaller files, but substantially slower.",
    )
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        help="Debug option to process only the first N files from each source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and write summary files; do not create windows.npz.",
    )
    return parser.parse_args()


def parse_fog_columns(value: str) -> list[str]:
    columns = [part.strip() for part in value.split(",") if part.strip()]
    if not columns or columns == ["any"]:
        columns = list(EVENT_COLUMNS)

    unknown = sorted(set(columns) - set(EVENT_COLUMNS))
    if unknown:
        allowed = ", ".join(EVENT_COLUMNS)
        raise ValueError(f"Unknown fog column(s): {unknown}. Allowed: {allowed}.")
    return columns


def load_metadata(data_root: Path, source: str) -> dict[str, str]:
    metadata_path = data_root / f"{source}_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = pd.read_csv(metadata_path, usecols=["Id", "Subject"])
    return dict(zip(metadata["Id"].astype(str), metadata["Subject"].astype(str)))


def iter_records(args: argparse.Namespace) -> list[FileRecord]:
    records: list[FileRecord] = []
    for source in args.sources:
        train_dir = args.data_root / "train" / source
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing source directory: {train_dir}")

        id_to_subject = load_metadata(args.data_root, source)
        paths = sorted(train_dir.glob("*.csv"))
        if args.max_files_per_source:
            paths = paths[: args.max_files_per_source]

        for path in paths:
            file_id = path.stem
            subject = id_to_subject.get(file_id)
            if subject is None:
                raise KeyError(f"No metadata subject found for {source}/{file_id}")
            records.append(
                FileRecord(
                    source=source,
                    file_id=file_id,
                    path=path,
                    subject=subject,
                    hz=DATASET_HZ[source],
                )
            )
    return records


def load_tasks(data_root: Path) -> dict[str, pd.DataFrame]:
    tasks_path = data_root / "tasks.csv"
    if not tasks_path.exists():
        return {}

    tasks = pd.read_csv(tasks_path)
    tasks["Id"] = tasks["Id"].astype(str)
    return {file_id: group.copy() for file_id, group in tasks.groupby("Id", sort=False)}


def read_source_csv(record: FileRecord) -> pd.DataFrame:
    usecols = ["Time", *FEATURE_COLUMNS, *EVENT_COLUMNS]
    dtype = {
        "Time": np.int64,
        "AccV": np.float32,
        "AccML": np.float32,
        "AccAP": np.float32,
        "StartHesitation": np.int8,
        "Turn": np.int8,
        "Walking": np.int8,
    }
    if record.source == "defog":
        usecols.extend(["Valid", "Task"])

    return pd.read_csv(record.path, usecols=usecols, dtype=dtype)


def as_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool, copy=False)
    lowered = series.astype(str).str.lower()
    return lowered.isin(("true", "1", "t", "yes")).to_numpy(dtype=bool)


def build_valid_mask(
    record: FileRecord,
    df: pd.DataFrame,
    tasks_by_id: dict[str, pd.DataFrame],
    defog_valid_only: bool,
    exclude_rest_tasks: bool,
) -> np.ndarray:
    valid = np.ones(len(df), dtype=bool)
    if record.source != "defog":
        return valid

    if defog_valid_only:
        valid &= as_bool_array(df["Valid"])
        valid &= as_bool_array(df["Task"])

    if exclude_rest_tasks:
        task_rows = tasks_by_id.get(record.file_id)
        if task_rows is None or task_rows.empty:
            return valid

        time_sec = df["Time"].to_numpy(dtype=np.float64, copy=False) / record.hz
        non_rest = np.zeros(len(df), dtype=bool)
        for task in task_rows.itertuples(index=False):
            task_name = str(task.Task)
            if task_name.lower().startswith("rest"):
                continue
            non_rest |= (time_sec >= float(task.Begin)) & (time_sec < float(task.End))
        valid &= non_rest

    return valid


def build_sample_state(
    df: pd.DataFrame,
    fog_columns: list[str],
    source_hz: int,
    pre_fog_seconds: float,
) -> np.ndarray:
    fog = df[fog_columns].to_numpy(dtype=np.int8, copy=False).any(axis=1)
    pre_fog = np.zeros(len(df), dtype=bool)
    pre_samples = int(round(pre_fog_seconds * source_hz))

    if pre_samples > 0 and fog.any():
        previous = np.r_[False, fog[:-1]]
        onsets = np.flatnonzero(fog & ~previous)
        for onset in onsets:
            start = max(0, onset - pre_samples)
            pre_fog[start:onset] = True

    pre_fog &= ~fog
    state = np.full(len(df), LABEL_NORMAL, dtype=np.int8)
    state[pre_fog] = LABEL_PRE_FOG
    state[fog] = LABEL_FOG
    return state


def contiguous_true_intervals(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    true_idx = np.flatnonzero(mask)
    if len(true_idx) == 0:
        return

    breaks = np.flatnonzero(np.diff(true_idx) != 1)
    starts = np.r_[true_idx[0], true_idx[breaks + 1]]
    ends = np.r_[true_idx[breaks] + 1, true_idx[-1] + 1]
    for start, end in zip(starts, ends):
        yield int(start), int(end)


def window_spans(
    valid_mask: np.ndarray,
    window_size: int,
    stride: int,
) -> Iterable[tuple[int, int]]:
    for interval_start, interval_end in contiguous_true_intervals(valid_mask):
        last_start = interval_end - window_size
        if last_start < interval_start:
            continue
        for start in range(interval_start, last_start + 1, stride):
            yield start, start + window_size


def label_window(state: np.ndarray, start: int, end: int, rule: str) -> int:
    values = state[start:end]
    if rule == "center":
        return int(values[len(values) // 2])
    if rule == "majority":
        return int(np.bincount(values, minlength=3).argmax())

    if np.any(values == LABEL_FOG):
        return LABEL_FOG
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


def summarize_record(
    record: FileRecord,
    args: argparse.Namespace,
    fog_columns: list[str],
    tasks_by_id: dict[str, pd.DataFrame],
) -> FileSummary:
    df = read_source_csv(record)
    state = build_sample_state(df, fog_columns, record.hz, args.pre_fog_seconds)
    valid = build_valid_mask(
        record, df, tasks_by_id, args.defog_valid_only, args.exclude_rest_tasks
    )
    window_size = int(round(args.window_seconds * record.hz))
    stride = int(round(window_size * (1.0 - args.overlap)))
    stride = max(1, stride)

    counts = np.zeros(3, dtype=np.int64)
    n_windows = 0
    for start, end in window_spans(valid, window_size, stride):
        counts[label_window(state, start, end, args.label_rule)] += 1
        n_windows += 1

    return FileSummary(
        source=record.source,
        file_id=record.file_id,
        subject=record.subject,
        rows=len(df),
        valid_rows=int(valid.sum()),
        windows=n_windows,
        normal_windows=int(counts[LABEL_NORMAL]),
        pre_fog_windows=int(counts[LABEL_PRE_FOG]),
        fog_windows=int(counts[LABEL_FOG]),
    )


def write_file_summary(output_dir: Path, summaries: list[FileSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "file_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def allocate_arrays(
    output_dir: Path,
    total_windows: int,
    target_len: int,
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
        shape=(total_windows, target_len, len(FEATURE_COLUMNS)),
    )
    arrays = {
        "y": np.empty(total_windows, dtype=np.int8),
        "subject": np.empty(total_windows, dtype="U16"),
        "source": np.empty(total_windows, dtype="U8"),
        "file_id": np.empty(total_windows, dtype="U16"),
        "start_sample": np.empty(total_windows, dtype=np.int64),
        "end_sample": np.empty(total_windows, dtype=np.int64),
        "native_hz": np.empty(total_windows, dtype=np.int16),
    }
    return tmp_dir, x, arrays


def fill_record_windows(
    record: FileRecord,
    args: argparse.Namespace,
    fog_columns: list[str],
    tasks_by_id: dict[str, pd.DataFrame],
    x: np.ndarray,
    arrays: dict[str, np.ndarray],
    offset: int,
    target_len: int,
) -> int:
    df = read_source_csv(record)
    state = build_sample_state(df, fog_columns, record.hz, args.pre_fog_seconds)
    valid = build_valid_mask(
        record, df, tasks_by_id, args.defog_valid_only, args.exclude_rest_tasks
    )
    features = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32, copy=False)
    times = df["Time"].to_numpy(dtype=np.int64, copy=False)
    window_size = int(round(args.window_seconds * record.hz))
    stride = int(round(window_size * (1.0 - args.overlap)))
    stride = max(1, stride)

    index = offset
    for start, end in window_spans(valid, window_size, stride):
        x[index] = resample_window(features[start:end], target_len)
        arrays["y"][index] = label_window(state, start, end, args.label_rule)
        arrays["subject"][index] = record.subject
        arrays["source"][index] = record.source
        arrays["file_id"][index] = record.file_id
        arrays["start_sample"][index] = int(times[start])
        arrays["end_sample"][index] = int(times[end - 1] + 1)
        arrays["native_hz"][index] = record.hz
        index += 1

    return index


def output_window_length(window_seconds: float, target_hz: int) -> int:
    target_len = int(round(float(window_seconds) * int(target_hz)))
    if target_len <= 0:
        raise ValueError("Output window length must be positive.")
    return target_len


def choose_val_subject(
    subjects: np.ndarray,
    test_subject: str,
    explicit_val_subject: str | None = None,
) -> str:
    subjects_list = [str(subject) for subject in subjects]
    if test_subject not in subjects_list:
        raise ValueError(f"Unknown heldout subject: {test_subject}")
    if explicit_val_subject:
        if explicit_val_subject not in subjects_list:
            raise ValueError(f"Unknown validation subject: {explicit_val_subject}")
        if explicit_val_subject == test_subject:
            raise ValueError("Validation subject must differ from heldout subject.")
        return explicit_val_subject

    test_index = subjects_list.index(test_subject)
    for step in range(1, len(subjects_list)):
        candidate = subjects_list[(test_index + step) % len(subjects_list)]
        if candidate != test_subject:
            return candidate
    raise ValueError("Need at least two subjects to build train/val/test splits.")


def build_subject_groups(
    subjects: np.ndarray,
    window_subject_code: np.ndarray,
    y: np.ndarray,
    num_folds: int,
    fold_seed: int,
) -> list[np.ndarray]:
    if num_folds <= 0 or num_folds >= len(subjects):
        return [np.array([subject], dtype="U16") for subject in subjects]
    if num_folds < 2:
        raise ValueError("--num-folds must be 0 or at least 2.")

    counts = np.zeros((len(subjects), len(CLASS_NAMES)), dtype=np.int64)
    for subject_code in range(len(subjects)):
        mask = window_subject_code == subject_code
        counts[subject_code] = np.bincount(y[mask], minlength=len(CLASS_NAMES))

    target = counts.sum(axis=0).astype(np.float64) / float(num_folds)
    denom = np.maximum(target, 1.0)
    fold_counts = np.zeros((num_folds, len(CLASS_NAMES)), dtype=np.float64)
    fold_groups: list[list[int]] = [[] for _ in range(num_folds)]

    rng = np.random.default_rng(fold_seed)
    tie_breaker = rng.random(len(subjects))
    order = sorted(
        range(len(subjects)),
        key=lambda idx: (
            -int(counts[idx, LABEL_FOG]),
            -int(counts[idx, LABEL_PRE_FOG]),
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

    output = []
    for group in fold_groups:
        group_subjects = np.array([subjects[idx] for idx in sorted(group)], dtype="U16")
        output.append(group_subjects)
    return output


def padded_subject_array(groups: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(group) for group in groups)
    out = np.full((len(groups), max_len), "", dtype="U16")
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
    config: dict,
    num_folds: int = 0,
    fold_seed: int = 42,
) -> None:
    subject_to_code = {str(subject): idx for idx, subject in enumerate(subjects)}
    test_groups = build_subject_groups(
        subjects=subjects,
        window_subject_code=window_subject_code,
        y=y,
        num_folds=num_folds,
        fold_seed=fold_seed,
    )
    val_groups = [test_groups[(idx + 1) % len(test_groups)] for idx in range(len(test_groups))]

    grouped = not (num_folds <= 0 or num_folds >= len(subjects))
    if grouped:
        test_subjects = padded_subject_array(test_groups)
        val_subjects = padded_subject_array(val_groups)
    else:
        test_subjects = np.array([group[0] for group in test_groups], dtype="U16")
        val_subjects = np.array([group[0] for group in val_groups], dtype="U16")

    test_subject_codes = padded_code_array(test_groups, subject_to_code)
    val_subject_codes = padded_code_array(val_groups, subject_to_code)

    folds_path = output_dir / "loso_folds.npz"
    np.savez(
        folds_path,
        subjects=subjects,
        fold_test_subjects=test_subjects,
        fold_val_subjects=val_subjects,
        fold_test_subject_codes=test_subject_codes,
        fold_val_subject_codes=val_subject_codes,
        window_subject_code=window_subject_code,
        class_names=CLASS_NAMES,
        config_json=np.array(json.dumps(config, ensure_ascii=False)),
    )

    rows = []
    for fold_idx, (test_group, val_group) in enumerate(zip(test_groups, val_groups)):
        test_codes = np.array([subject_to_code[str(subject)] for subject in test_group])
        val_codes = np.array([subject_to_code[str(subject)] for subject in val_group])
        test_mask = np.isin(window_subject_code, test_codes)
        val_mask = np.isin(window_subject_code, val_codes)
        train_mask = ~(test_mask | val_mask)
        rows.append(
            {
                "fold": fold_idx,
                "test_subject": "|".join(str(subject) for subject in test_group),
                "val_subject": "|".join(str(subject) for subject in val_group),
                "test_subject_count": int(len(test_group)),
                "val_subject_count": int(len(val_group)),
                "train_windows": int(train_mask.sum()),
                "val_windows": int(val_mask.sum()),
                "test_windows": int(test_mask.sum()),
                "train_normal": int(np.sum(y[train_mask] == LABEL_NORMAL)),
                "train_pre_fog": int(np.sum(y[train_mask] == LABEL_PRE_FOG)),
                "train_fog": int(np.sum(y[train_mask] == LABEL_FOG)),
                "val_normal": int(np.sum(y[val_mask] == LABEL_NORMAL)),
                "val_pre_fog": int(np.sum(y[val_mask] == LABEL_PRE_FOG)),
                "val_fog": int(np.sum(y[val_mask] == LABEL_FOG)),
                "test_normal": int(np.sum(y[test_mask] == LABEL_NORMAL)),
                "test_pre_fog": int(np.sum(y[test_mask] == LABEL_PRE_FOG)),
                "test_fog": int(np.sum(y[test_mask] == LABEL_FOG)),
            }
        )

    with (output_dir / "loso_folds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, compress: bool, **arrays: np.ndarray) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def export_materialized_fold(
    output_dir: Path,
    x: np.ndarray,
    arrays: dict[str, np.ndarray],
    subjects: np.ndarray,
    window_subject_code: np.ndarray,
    heldout_subject: str,
    val_subject: str | None,
    config: dict,
    compress: bool,
) -> None:
    val = choose_val_subject(subjects, heldout_subject, val_subject)
    test_code = int(np.flatnonzero(subjects == heldout_subject)[0])
    val_code = int(np.flatnonzero(subjects == val)[0])

    split_masks = {
        "train": ~(np.isin(window_subject_code, [test_code, val_code])),
        "val": window_subject_code == val_code,
        "test": window_subject_code == test_code,
    }
    fold_dir = output_dir / f"fold_test-{heldout_subject}_val-{val}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for split, mask in split_masks.items():
        indices = np.flatnonzero(mask)
        save_npz(
            fold_dir / f"{split}.npz",
            compress,
            X=x[indices],
            y=arrays["y"][indices],
            subject=arrays["subject"][indices],
            source=arrays["source"][indices],
            file_id=arrays["file_id"][indices],
            start_sample=arrays["start_sample"][indices],
            end_sample=arrays["end_sample"][indices],
            native_hz=arrays["native_hz"][indices],
            class_names=CLASS_NAMES,
            feature_names=np.array(FEATURE_COLUMNS),
            config_json=np.array(json.dumps(config, ensure_ascii=False)),
        )


def write_readme(output_dir: Path) -> None:
    text = """# FOG LOSO NPZ output

Files:

- `windows.npz`: all generated windows, labels, subjects, file ids, and timing metadata.
- `loso_folds.npz`: compact LOSO fold metadata. Split by comparing `window_subject_code` to `fold_*_subject_codes`.
- `loso_folds.csv`: fold-level label counts for quick inspection.
- `file_summary.csv`: per-file row/window/label counts.

Class mapping:

- `0`: NORMAL
- `1`: PRE_FOG
- `2`: FOG

Example split reconstruction:

```python
import numpy as np

data = np.load("windows.npz")
folds = np.load("loso_folds.npz")
subjects = folds["subjects"]
codes = folds["window_subject_code"]

fold = 0
test_subject = folds["fold_test_subjects"][fold]
val_subject = folds["fold_val_subjects"][fold]
test_codes = folds["fold_test_subject_codes"][fold]
val_codes = folds["fold_val_subject_codes"][fold]
test_codes = test_codes[test_codes >= 0]
val_codes = val_codes[val_codes >= 0]

test_idx = np.flatnonzero(np.isin(codes, test_codes))
val_idx = np.flatnonzero(np.isin(codes, val_codes))
train_idx = np.flatnonzero(~np.isin(codes, np.r_[test_codes, val_codes]))

X_train, y_train = data["X"][train_idx], data["y"][train_idx]
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1).")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive.")
    if args.target_hz <= 0:
        raise ValueError("--target-hz must be positive.")

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    args.data_root = data_root
    args.output_dir = output_dir

    fog_columns = parse_fog_columns(args.fog_columns)
    target_len = output_window_length(args.window_seconds, args.target_hz)
    records = iter_records(args)
    tasks_by_id = load_tasks(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_root": str(data_root),
        "sources": args.sources,
        "fog_columns": fog_columns,
        "window_seconds": args.window_seconds,
        "overlap": args.overlap,
        "pre_fog_seconds": args.pre_fog_seconds,
        "target_hz": args.target_hz,
        "target_len": target_len,
        "label_rule": args.label_rule,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "defog_valid_only": args.defog_valid_only,
        "exclude_rest_tasks": args.exclude_rest_tasks,
        "class_names": CLASS_NAMES.tolist(),
        "feature_names": list(FEATURE_COLUMNS),
        "dataset_hz": DATASET_HZ,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Scanning {len(records)} files...")
    summaries: list[FileSummary] = []
    for idx, record in enumerate(records, start=1):
        summaries.append(summarize_record(record, args, fog_columns, tasks_by_id))
        if idx % 50 == 0 or idx == len(records):
            print(f"  scanned {idx}/{len(records)}")

    write_file_summary(output_dir, summaries)
    total_windows = int(sum(summary.windows for summary in summaries))
    print(f"Total windows: {total_windows:,}")
    print(
        "Label counts:",
        {
            "NORMAL": sum(s.normal_windows for s in summaries),
            "PRE_FOG": sum(s.pre_fog_windows for s in summaries),
            "FOG": sum(s.fog_windows for s in summaries),
        },
    )

    if args.dry_run:
        write_readme(output_dir)
        print(f"Dry run complete. Summaries written to {output_dir}")
        return

    tmp_dir, x, arrays = allocate_arrays(output_dir, total_windows, target_len)
    try:
        print("Writing window arrays...")
        offset = 0
        for idx, record in enumerate(records, start=1):
            offset = fill_record_windows(
                record, args, fog_columns, tasks_by_id, x, arrays, offset, target_len
            )
            if idx % 25 == 0 or idx == len(records):
                print(f"  wrote {idx}/{len(records)} files; {offset:,}/{total_windows:,} windows")
        if offset != total_windows:
            raise RuntimeError(f"Expected {total_windows} windows, wrote {offset}.")

        subjects = np.array(sorted(set(arrays["subject"].tolist())), dtype="U16")
        subject_to_code = {subject: i for i, subject in enumerate(subjects)}
        window_subject_code = np.array(
            [subject_to_code[subject] for subject in arrays["subject"]],
            dtype=np.int16,
        )

        windows_path = output_dir / "windows.npz"
        print(f"Saving {windows_path} ...")
        x.flush()
        save_npz(
            windows_path,
            args.compress,
            X=x,
            y=arrays["y"],
            subject=arrays["subject"],
            subject_code=window_subject_code,
            subjects=subjects,
            source=arrays["source"],
            file_id=arrays["file_id"],
            start_sample=arrays["start_sample"],
            end_sample=arrays["end_sample"],
            native_hz=arrays["native_hz"],
            class_names=CLASS_NAMES,
            feature_names=np.array(FEATURE_COLUMNS),
            config_json=np.array(json.dumps(config, ensure_ascii=False)),
        )

        write_loso_folds(
            output_dir,
            subjects,
            window_subject_code,
            arrays["y"],
            config,
            num_folds=args.num_folds,
            fold_seed=args.fold_seed,
        )
        if args.heldout_subject:
            export_materialized_fold(
                output_dir=output_dir,
                x=x,
                arrays=arrays,
                subjects=subjects,
                window_subject_code=window_subject_code,
                heldout_subject=args.heldout_subject,
                val_subject=args.val_subject,
                config=config,
                compress=args.compress,
            )

        write_readme(output_dir)
        print(f"Done. Output written to {output_dir}")
    finally:
        del x
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
