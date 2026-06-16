#!/usr/bin/env python
"""Stream supervised Kaggle FOG CSV records from the competition zip.

This script never extracts the zip. It opens one CSV member at a time, writes
sample-level NPZ records, and leaves windowing / Pre-FOG / normalization to
training code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_NAME = "Kaggle Parkinson's Freezing of Gait Prediction"
DATASET_ID = "kaggle_fog"
DEFAULT_DATASET_PREFIX = "2.Kaggle"
CHANNELS = ("AccV", "AccML", "AccAP")
LABEL_COLUMNS = ("StartHesitation", "Turn", "Walking")
SOURCE_SAMPLING_RATES = {
    "tdcsfog": 128,
    "defog": 100,
    "notype": 100,
}


@dataclass
class ManifestRow:
    dataset_id: str
    record_id: str
    source_file: str
    dataset_part: str
    series_id: str
    subject_id: str
    source_subject_id: str
    run_id: str
    segment_id: int
    record_path: str
    sampling_rate: int
    n_samples: int
    duration_sec: float
    n_normal_samples: int
    n_fog_samples: int
    channels: str
    sensor_positions: str
    filter_valid_only: bool
    filter_task_only: bool


@dataclass
class SourceSummaryRow:
    dataset_id: str
    source_file: str
    dataset_part: str
    series_id: str
    subject_id: str
    source_subject_id: str
    sampling_rate: int
    n_rows: int
    n_kept_rows: int
    n_segments: int
    n_records: int
    n_samples: int
    n_fog_samples: int
    filter_valid_only: bool
    filter_task_only: bool
    status: str


@dataclass
class SegmentBuffer:
    x_parts: list[np.ndarray]
    y_parts: list[np.ndarray]
    n_samples: int = 0
    n_fog_samples: int = 0

    def append(self, df: pd.DataFrame, y_binary: np.ndarray, context: str = "") -> None:
        if len(df) == 0:
            return
        x64 = numeric_matrix(df, CHANNELS)
        finite = np.isfinite(x64)
        if not finite.all():
            bad_rows = int((~finite.all(axis=1)).sum())
            bad_values = int((~finite).sum())
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}features contain NaN or non-finite values: rows={bad_rows} values={bad_values}")
        x = x64.astype(np.float32, copy=False)
        y = y_binary.astype(np.uint8, copy=False)
        self.x_parts.append(x)
        self.y_parts.append(y)
        self.n_samples += int(y.shape[0])
        self.n_fog_samples += int(y.sum())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.x_parts) == 1:
            return self.x_parts[0], self.y_parts[0]
        return np.concatenate(self.x_parts, axis=0), np.concatenate(self.y_parts, axis=0)


@dataclass
class LosoRow:
    fold_id: str
    test_subject_id: str
    split: str
    record_id: str
    subject_id: str
    segment_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sample-level binary records from the Kaggle FOG zip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zip-path", type=Path, default=None, help="Path to the Kaggle competition zip.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root used to auto-locate the Kaggle directory when --zip-path is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to processed_smoke when --smoke-limit > 0, otherwise processed.",
    )
    parser.add_argument(
        "--source",
        choices=("tdcsfog", "defog", "both", "all"),
        default="both",
        help="'both' means tdcsfog + defog. 'all' also includes train/notype using Event as a binary label.",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="For defog/notype, keep only rows with Valid == true.",
    )
    parser.add_argument(
        "--task-only",
        action="store_true",
        help="For defog/notype, keep only rows with Task == true.",
    )
    parser.add_argument("--smoke-limit", type=int, default=0, help="Process only the first N train CSV files per source.")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Rows per pandas read_csv chunk.")
    parser.add_argument("--min-samples", type=int, default=1, help="Drop retained segments shorter than this.")
    parser.add_argument(
        "--record-compression",
        choices=("compressed", "none"),
        default="compressed",
        help="NPZ record compression. 'none' writes faster and uses more disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect selected train members and metadata. Do not create output files.",
    )
    parser.add_argument(
        "--dry-run-output-json",
        type=Path,
        help="Optional JSON report path for --dry-run diagnostics.",
    )
    parser.add_argument(
        "--check-headers",
        action="store_true",
        help="During --dry-run, open selected train CSV members and validate header columns without reading data rows.",
    )
    parser.add_argument(
        "--profile-data",
        action="store_true",
        help="During --dry-run, stream selected train CSV rows and report labels, kept rows, and NaN/non-finite counts without creating records. Implies --check-headers.",
    )
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Fail when a selected train CSV lacks a usable metadata Subject mapping for LOSO.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip source files already present in source_summary.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory first if it exists.")
    parser.add_argument("--stop-after-source-files", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.dry_run_output_json and not args.dry_run:
        parser.error("--dry-run-output-json requires --dry-run")
    if args.profile_data and not args.dry_run:
        parser.error("--profile-data requires --dry-run")
    if args.profile_data:
        args.check_headers = True
    if args.stop_after_source_files < 0:
        parser.error("--stop-after-source-files must be >= 0")
    return args


def find_default_zip(dataset_root: Path) -> Path:
    candidates = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith(DEFAULT_DATASET_PREFIX)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {DEFAULT_DATASET_PREFIX} directory under {dataset_root}, found {candidates}")
    zip_path = candidates[0] / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    return zip_path


def default_output_dir(zip_path: Path, smoke_limit: int) -> Path:
    output_name = "processed_smoke" if smoke_limit > 0 else "processed"
    return zip_path.parent / output_name


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def clean_metadata_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def subject_code(index: int) -> str:
    return f"S{index:03d}"


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if output_dir.exists() and overwrite:
        if output_dir.name not in {"processed", "processed_smoke"}:
            raise ValueError(f"Refusing to overwrite unexpected output directory: {output_dir}")
        shutil.rmtree(output_dir)
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite or --resume")
    (output_dir / "records").mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "_SUCCESS.json"
    if success_path.exists():
        success_path.unlink()


def read_small_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle)


def load_metadata(archive: zipfile.ZipFile) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, str], list[dict[str, object]]]:
    tdcs = read_small_csv(archive, "tdcsfog_metadata.csv")
    defog = read_small_csv(archive, "defog_metadata.csv")
    subjects = read_small_csv(archive, "subjects.csv")

    series_meta: dict[tuple[str, str], dict[str, object]] = {}
    for dataset_part, df in (("tdcsfog", tdcs), ("defog", defog)):
        for row in df.to_dict(orient="records"):
            series_meta[(dataset_part, str(row["Id"]))] = row

    source_subject_ids = sorted(
        {
            str(row.get("Subject"))
            for row in series_meta.values()
            if row.get("Subject") is not None and not pd.isna(row.get("Subject"))
        }
    )
    subject_map = {source_subject_id: subject_code(index + 1) for index, source_subject_id in enumerate(source_subject_ids)}

    subject_rows: list[dict[str, object]] = []
    for row in subjects.sort_values(["Subject", "Visit"]).to_dict(orient="records"):
        source_subject_id = str(row["Subject"])
        subject_rows.append(
            {
                "subject_id": subject_map.get(source_subject_id, ""),
                "source_subject_id": source_subject_id,
                "visit": clean_metadata_value(row.get("Visit", "")),
                "age": clean_metadata_value(row.get("Age", "")),
                "sex": clean_metadata_value(row.get("Sex", "")),
                "years_since_dx": clean_metadata_value(row.get("YearsSinceDx", "")),
                "updrsiii_on": clean_metadata_value(row.get("UPDRSIII_On", "")),
                "updrsiii_off": clean_metadata_value(row.get("UPDRSIII_Off", "")),
                "nfogq": clean_metadata_value(row.get("NFOGQ", "")),
            }
        )

    return series_meta, subject_map, subject_rows


def selected_sources(source: str) -> tuple[str, ...]:
    if source == "both":
        return ("tdcsfog", "defog")
    if source == "all":
        return ("tdcsfog", "defog", "notype")
    return (source,)


def iter_train_members(archive: zipfile.ZipFile, source: str) -> list[tuple[str, str, str]]:
    wanted = selected_sources(source)
    members: list[tuple[str, str, str]] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        if len(parts) != 3 or parts[0] != "train" or parts[2].split(".")[-1].lower() != "csv":
            continue
        dataset_part = parts[1]
        if dataset_part not in wanted:
            continue
        series_id = Path(parts[2]).stem
        members.append((dataset_part, series_id, info.filename))
    return sorted(members, key=lambda item: (item[0], item[2]))


def format_gib(size: int) -> str:
    return f"{size / 1024**3:.3f} GiB"


def zip_fingerprint(zip_path: Path) -> dict[str, object]:
    stat = zip_path.stat()
    return {
        "zip_path": str(zip_path),
        "zip_size": int(stat.st_size),
        "zip_modified_time_ns": int(stat.st_mtime_ns),
    }


def metadata_mapping_issue(
    dataset_part: str,
    series_id: str,
    series_meta: dict[tuple[str, str], dict[str, object]],
    subject_map: dict[str, str],
) -> str | None:
    meta = series_meta.get((dataset_part, series_id))
    if meta is None:
        return "missing series metadata"

    source_subject_id = str(meta.get("Subject", "")).strip()
    if not source_subject_id:
        return "metadata Subject is empty"
    if source_subject_id not in subject_map:
        return f"metadata Subject {source_subject_id!r} is absent from subjects.csv"
    return None


def strict_metadata_issues(
    members: list[tuple[str, str, str]],
    series_meta: dict[tuple[str, str], dict[str, object]],
    subject_map: dict[str, str],
) -> list[str]:
    issues: list[str] = []
    for dataset_part, series_id, member in members:
        issue = metadata_mapping_issue(dataset_part, series_id, series_meta, subject_map)
        if issue:
            issues.append(f"{member}: {issue}")
    return issues


def validate_strict_metadata(
    members: list[tuple[str, str, str]],
    series_meta: dict[tuple[str, str], dict[str, object]],
    subject_map: dict[str, str],
) -> None:
    issues = strict_metadata_issues(members, series_meta, subject_map)
    if not issues:
        return

    preview = "\n".join(issues[:20])
    remaining = len(issues) - 20
    suffix = f"\n... {remaining} more" if remaining > 0 else ""
    raise ValueError(
        "Strict metadata check failed. Selected train CSV members lack a usable Subject mapping:\n"
        f"{preview}{suffix}"
    )


def build_dry_run_report(
    archive: zipfile.ZipFile,
    zip_path: Path,
    members: list[tuple[str, str, str]],
    series_meta: dict[tuple[str, str], dict[str, object]],
    subject_map: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    members = apply_smoke_limit(members, args.smoke_limit)

    info_by_name = {info.filename: info for info in archive.infolist() if not info.is_dir()}
    by_source: dict[str, dict[str, int]] = {}
    missing_metadata = 0
    metadata_issues: list[str] = []
    subjects: set[str] = set()
    headers_checked = 0
    header_issues: list[str] = []
    profile_overall = empty_profile_stats()
    profile_by_source: dict[str, dict[str, Any]] = {}
    profile_skipped_header_issues = 0

    for dataset_part, series_id, member in members:
        info = info_by_name[member]
        stats = by_source.setdefault(
            dataset_part,
            {"files": 0, "compressed_size": 0, "uncompressed_size": 0},
        )
        stats["files"] += 1
        stats["compressed_size"] += info.compress_size
        stats["uncompressed_size"] += info.file_size

        metadata_issue = metadata_mapping_issue(dataset_part, series_id, series_meta, subject_map)
        if metadata_issue:
            missing_metadata += 1
            metadata_issues.append(f"{member}: {metadata_issue}")
        else:
            meta = series_meta[(dataset_part, series_id)]
            source_subject_id = str(meta["Subject"])
            subjects.add(subject_map[source_subject_id])
        member_header_issue = False
        if args.check_headers:
            headers_checked += 1
            missing = missing_header_columns(archive, member, dataset_part)
            if missing:
                header_issues.append(f"{member}: missing {missing}")
                member_header_issue = True
        if args.profile_data:
            if member_header_issue:
                profile_skipped_header_issues += 1
            else:
                member_profile = profile_member_data(archive, member, dataset_part, args)
                merge_profile_stats(profile_overall, member_profile)
                source_profile = profile_by_source.setdefault(dataset_part, empty_profile_stats())
                merge_profile_stats(source_profile, member_profile)

    report = {
        "dry_run": True,
        **zip_fingerprint(zip_path),
        "selected_source": args.source,
        "valid_only": args.valid_only,
        "task_only": args.task_only,
        "strict_metadata": args.strict_metadata,
        "check_headers": args.check_headers,
        "profile_data": args.profile_data,
        "smoke_limit": args.smoke_limit,
        "selected_train_csv_files": len(members),
        "metadata_subjects": len(subjects),
        "members_missing_metadata": missing_metadata,
        "headers_checked": headers_checked,
        "members_with_header_issues": len(header_issues),
        "by_source": {
            source: {
                "files": int(stats["files"]),
                "compressed_size": int(stats["compressed_size"]),
                "compressed_gib": round(int(stats["compressed_size"]) / 1024**3, 6),
                "uncompressed_size": int(stats["uncompressed_size"]),
                "uncompressed_gib": round(int(stats["uncompressed_size"]) / 1024**3, 6),
            }
            for source, stats in sorted(by_source.items())
        },
        "metadata_issues": metadata_issues[:20],
        "metadata_issue_count": len(metadata_issues),
        "header_issues": header_issues[:20],
        "header_issue_count": len(header_issues),
    }
    if args.profile_data:
        report["profile"] = {
            "overall": profile_overall,
            "by_source": {source: stats for source, stats in sorted(profile_by_source.items())},
            "members_skipped_header_issues": profile_skipped_header_issues,
        }
    return report


def print_dry_run_report(report: dict[str, Any]) -> None:
    print("dry_run: true")
    print(f"selected_source: {report['selected_source']}")
    print(f"valid_only: {report['valid_only']}")
    print(f"task_only: {report['task_only']}")
    print(f"smoke_limit: {report['smoke_limit']}")
    print(f"selected_train_csv_files: {report['selected_train_csv_files']}")
    print(f"metadata_subjects: {report['metadata_subjects']}")
    print(f"members_missing_metadata: {report['members_missing_metadata']}")
    print(f"headers_checked: {report['headers_checked']}")
    print(f"members_with_header_issues: {report['members_with_header_issues']}")
    print(f"profile_data: {report['profile_data']}")
    for source, stats in sorted(report["by_source"].items()):
        print(
            f"{source}: files={stats['files']} "
            f"compressed={format_gib(stats['compressed_size'])} "
            f"uncompressed={format_gib(stats['uncompressed_size'])}"
        )
    if report.get("profile"):
        overall = report["profile"]["overall"]
        print(
            "profile_overall: "
            f"files={overall['files_profiled']} rows={overall['rows']} kept={overall['kept_rows']} "
            f"normal={overall['normal_samples']} fog={overall['fog_samples']} "
            f"normal_sec={overall['normal_duration_sec']} fog_sec={overall['fog_duration_sec']} "
            f"x_nan={overall['x_nan_values']} x_nonfinite={overall['x_nonfinite_values']} "
            f"label_invalid={overall['label_invalid_rows']} "
            f"label_nonbinary={overall['label_nonbinary_rows']}"
        )


def dry_run_summary(
    archive: zipfile.ZipFile,
    zip_path: Path,
    members: list[tuple[str, str, str]],
    series_meta: dict[tuple[str, str], dict[str, object]],
    subject_map: dict[str, str],
    args: argparse.Namespace,
) -> None:
    report = build_dry_run_report(archive, zip_path, members, series_meta, subject_map, args)
    print_dry_run_report(report)
    if args.dry_run_output_json:
        args.dry_run_output_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.dry_run_output_json, report)

    metadata_issues = report["metadata_issues"]
    if args.strict_metadata and metadata_issues:
        preview = "\n".join(str(issue) for issue in metadata_issues)
        remaining = int(report["metadata_issue_count"]) - len(metadata_issues)
        suffix = f"\n... {remaining} more" if remaining > 0 else ""
        raise ValueError(
            "Strict metadata check failed. Selected train CSV members lack a usable Subject mapping:\n"
            f"{preview}{suffix}"
        )
    header_issues = report["header_issues"]
    if header_issues:
        raise ValueError("CSV header check failed:\n" + "\n".join(str(issue) for issue in header_issues))


def series_columns(dataset_part: str) -> list[str]:
    if dataset_part in {"tdcsfog", "defog"}:
        columns = ["Time", *CHANNELS, *LABEL_COLUMNS]
    elif dataset_part == "notype":
        columns = ["Time", *CHANNELS, "Event"]
    else:
        raise ValueError(f"Unexpected dataset_part: {dataset_part}")
    if dataset_part in {"defog", "notype"}:
        columns.extend(["Valid", "Task"])
    return columns


def missing_header_columns(archive: zipfile.ZipFile, member: str, dataset_part: str) -> list[str]:
    expected = set(series_columns(dataset_part))
    with archive.open(member) as handle:
        header = pd.read_csv(handle, nrows=0)
    return sorted(expected - set(header.columns))


def iter_series_csv_chunks(
    archive: zipfile.ZipFile,
    member: str,
    dataset_part: str,
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    columns = series_columns(dataset_part)
    with archive.open(member) as handle:
        for chunk in pd.read_csv(handle, usecols=columns, chunksize=chunk_size):
            yield chunk


def apply_smoke_limit(members: list[tuple[str, str, str]], smoke_limit: int) -> list[tuple[str, str, str]]:
    if smoke_limit <= 0:
        return members
    selected: list[tuple[str, str, str]] = []
    counts: dict[str, int] = {}
    for member in members:
        dataset_part = member[0]
        count = counts.get(dataset_part, 0)
        if count >= smoke_limit:
            continue
        selected.append(member)
        counts[dataset_part] = count + 1
    return selected


def label_columns_for_part(dataset_part: str) -> tuple[str, ...]:
    if dataset_part == "notype":
        return ("Event",)
    return LABEL_COLUMNS


def binary_labels(df: pd.DataFrame, dataset_part: str) -> np.ndarray:
    label_columns = label_columns_for_part(dataset_part)
    labels = numeric_matrix(df, label_columns)
    finite = np.isfinite(labels)
    if not finite.all():
        bad_rows = int((~finite.all(axis=1)).sum())
        bad_values = int((~finite).sum())
        raise ValueError(
            f"{dataset_part} labels contain NaN or non-finite values: rows={bad_rows} values={bad_values}"
        )
    binary = (labels == 0.0) | (labels == 1.0)
    if not binary.all():
        bad_rows = int((~binary.all(axis=1)).sum())
        bad_values = int((~binary).sum())
        raise ValueError(
            f"{dataset_part} labels must be binary 0/1 values: rows={bad_rows} values={bad_values}"
        )
    return labels.max(axis=1).astype(np.uint8)


def bool_series_to_numpy(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy() != 0
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "t", "yes", "y"}).to_numpy()


def keep_mask(df: pd.DataFrame, dataset_part: str, valid_only: bool, task_only: bool) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    if dataset_part in {"defog", "notype"}:
        if valid_only:
            mask &= bool_series_to_numpy(df["Valid"])
        if task_only:
            mask &= bool_series_to_numpy(df["Task"])
    return mask


def empty_profile_stats() -> dict[str, Any]:
    return {
        "files_profiled": 0,
        "chunks": 0,
        "rows": 0,
        "kept_rows": 0,
        "valid_rows": 0,
        "task_rows": 0,
        "normal_samples": 0,
        "fog_samples": 0,
        "profiled_duration_sec": 0.0,
        "kept_duration_sec": 0.0,
        "normal_duration_sec": 0.0,
        "fog_duration_sec": 0.0,
        "x_nan_values": 0,
        "x_nonfinite_values": 0,
        "x_kept_nan_values": 0,
        "x_kept_nonfinite_values": 0,
        "label_nan_values": 0,
        "label_nonfinite_values": 0,
        "label_invalid_rows": 0,
        "kept_label_invalid_rows": 0,
        "label_nonbinary_values": 0,
        "label_nonbinary_rows": 0,
        "kept_label_nonbinary_rows": 0,
        "x_nan_by_channel": {channel: 0 for channel in CHANNELS},
        "x_nonfinite_by_channel": {channel: 0 for channel in CHANNELS},
        "x_kept_nan_by_channel": {channel: 0 for channel in CHANNELS},
        "x_kept_nonfinite_by_channel": {channel: 0 for channel in CHANNELS},
    }


def merge_profile_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            nested = target.setdefault(key, {})
            for nested_key, nested_value in value.items():
                nested[nested_key] = int(nested.get(nested_key, 0)) + int(nested_value)
        else:
            current = target.get(key, 0.0 if isinstance(value, float) else 0)
            total = current + value
            target[key] = round(float(total), 9) if isinstance(current, float) or isinstance(value, float) else int(total)


def add_channel_counts(stats: dict[str, Any], key: str, counts: np.ndarray) -> None:
    by_channel = stats[key]
    for idx, channel in enumerate(CHANNELS):
        by_channel[channel] = int(by_channel[channel]) + int(counts[idx])


def numeric_matrix(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> np.ndarray:
    numeric = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=np.float64)


def profile_binary_labels(label_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_rows = np.isfinite(label_values).all(axis=1)
    safe_labels = np.where(np.isfinite(label_values), label_values, 0.0)
    y_binary = (safe_labels.max(axis=1) > 0).astype(np.uint8)
    binary_values = (safe_labels == 0.0) | (safe_labels == 1.0)
    binary_rows = finite_rows & binary_values.all(axis=1)
    return y_binary, finite_rows, binary_rows


def profile_member_data(
    archive: zipfile.ZipFile,
    member: str,
    dataset_part: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats = empty_profile_stats()
    stats["files_profiled"] = 1
    label_columns = label_columns_for_part(dataset_part)
    sampling_rate = SOURCE_SAMPLING_RATES[dataset_part]

    for chunk in iter_series_csv_chunks(archive, member, dataset_part, args.chunk_size):
        keep = keep_mask(chunk, dataset_part, args.valid_only, args.task_only)
        labels = numeric_matrix(chunk, list(label_columns))
        y_binary, finite_label_rows, binary_label_rows = profile_binary_labels(labels)
        kept_clean_labels = keep & finite_label_rows & binary_label_rows
        y_kept = y_binary[kept_clean_labels]

        x = numeric_matrix(chunk, CHANNELS)
        x_nan = np.isnan(x)
        x_nonfinite = ~np.isfinite(x)
        x_kept = x[keep]
        x_kept_nan = np.isnan(x_kept)
        x_kept_nonfinite = ~np.isfinite(x_kept)
        label_nan = np.isnan(labels)
        label_nonfinite = ~np.isfinite(labels)
        label_nonbinary = np.isfinite(labels) & ~((labels == 0.0) | (labels == 1.0))

        chunk_rows = int(len(chunk))
        kept_rows = int(keep.sum())
        fog_samples = int(y_kept.sum())
        normal_samples = int(len(y_kept) - fog_samples)

        stats["chunks"] += 1
        stats["rows"] += chunk_rows
        stats["kept_rows"] += kept_rows
        if dataset_part in {"defog", "notype"}:
            stats["valid_rows"] += int(bool_series_to_numpy(chunk["Valid"]).sum())
            stats["task_rows"] += int(bool_series_to_numpy(chunk["Task"]).sum())
        else:
            stats["valid_rows"] += int(len(chunk))
            stats["task_rows"] += int(len(chunk))
        stats["fog_samples"] += fog_samples
        stats["normal_samples"] += normal_samples
        stats["profiled_duration_sec"] = round(float(stats["profiled_duration_sec"]) + chunk_rows / sampling_rate, 9)
        stats["kept_duration_sec"] = round(float(stats["kept_duration_sec"]) + kept_rows / sampling_rate, 9)
        stats["normal_duration_sec"] = round(float(stats["normal_duration_sec"]) + normal_samples / sampling_rate, 9)
        stats["fog_duration_sec"] = round(float(stats["fog_duration_sec"]) + fog_samples / sampling_rate, 9)
        stats["x_nan_values"] += int(x_nan.sum())
        stats["x_nonfinite_values"] += int(x_nonfinite.sum())
        stats["x_kept_nan_values"] += int(x_kept_nan.sum())
        stats["x_kept_nonfinite_values"] += int(x_kept_nonfinite.sum())
        stats["label_nan_values"] += int(label_nan.sum())
        stats["label_nonfinite_values"] += int(label_nonfinite.sum())
        stats["label_invalid_rows"] += int((~finite_label_rows).sum())
        stats["kept_label_invalid_rows"] += int((keep & ~finite_label_rows).sum())
        stats["label_nonbinary_values"] += int(label_nonbinary.sum())
        stats["label_nonbinary_rows"] += int((finite_label_rows & ~binary_label_rows).sum())
        stats["kept_label_nonbinary_rows"] += int((keep & finite_label_rows & ~binary_label_rows).sum())

        add_channel_counts(stats, "x_nan_by_channel", x_nan.sum(axis=0))
        add_channel_counts(stats, "x_nonfinite_by_channel", x_nonfinite.sum(axis=0))
        add_channel_counts(stats, "x_kept_nan_by_channel", x_kept_nan.sum(axis=0))
        add_channel_counts(stats, "x_kept_nonfinite_by_channel", x_kept_nonfinite.sum(axis=0))

    return stats


def dataclass_fieldnames(cls: type[object]) -> list[str]:
    return [field.name for field in fields(cls)]


def write_csv(path: Path, rows: list[object], fieldnames: list[str] | None = None) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    if rows:
        first = rows[0]
        fieldnames = list(asdict(first).keys()) if not isinstance(first, dict) else list(first.keys())
    elif fieldnames is None:
        fieldnames = []
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row) if not isinstance(row, dict) else row)
    tmp_path.replace(path)


def write_json(path: Path, value: object) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def read_existing_config(output_dir: Path) -> dict[str, object]:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_resume_compatibility(output_dir: Path, zip_path: Path, args: argparse.Namespace) -> None:
    if not args.resume:
        return
    config = read_existing_config(output_dir)
    if not config:
        return

    expected = {
        "selected_source": args.source,
        "valid_only": args.valid_only,
        "task_only": args.task_only,
        "strict_metadata": args.strict_metadata,
        "record_compression": args.record_compression,
        "min_samples": args.min_samples,
    }
    defaults = {
        "strict_metadata": False,
        "record_compression": "compressed",
        "min_samples": 1,
    }
    mismatches: list[str] = []
    for key, requested in expected.items():
        existing = config.get(key, defaults.get(key))
        if existing != requested:
            mismatches.append(f"{key}: existing={existing!r}, requested={requested!r}")

    existing_source_zip = config.get("source_zip")
    if existing_source_zip:
        try:
            same_zip = Path(str(existing_source_zip)).resolve() == zip_path.resolve()
        except OSError:
            same_zip = str(existing_source_zip) == str(zip_path)
        if not same_zip:
            mismatches.append(f"source_zip: existing={existing_source_zip!r}, requested={str(zip_path)!r}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"Cannot --resume with incompatible preprocessing settings: {joined}")


def save_record_npz(path: Path, x: np.ndarray, y_binary: np.ndarray, compression: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.npz")
    if compression == "compressed":
        np.savez_compressed(tmp_path, x=x, y_binary=y_binary)
    elif compression == "none":
        np.savez(tmp_path, x=x, y_binary=y_binary)
    else:
        raise ValueError(f"Unsupported record compression: {compression}")
    tmp_path.replace(path)


def read_existing_manifest(output_dir: Path) -> list[ManifestRow]:
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.exists():
        return []
    rows: list[ManifestRow] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ManifestRow(
                    dataset_id=row["dataset_id"],
                    record_id=row["record_id"],
                    source_file=row["source_file"],
                    dataset_part=row["dataset_part"],
                    series_id=row["series_id"],
                    subject_id=row["subject_id"],
                    source_subject_id=row["source_subject_id"],
                    run_id=row["run_id"],
                    segment_id=int(row["segment_id"]),
                    record_path=row["record_path"],
                    sampling_rate=int(row["sampling_rate"]),
                    n_samples=int(row["n_samples"]),
                    duration_sec=float(row["duration_sec"]),
                    n_normal_samples=int(row["n_normal_samples"]),
                    n_fog_samples=int(row["n_fog_samples"]),
                    channels=row["channels"],
                    sensor_positions=row["sensor_positions"],
                    filter_valid_only=row["filter_valid_only"].lower() == "true",
                    filter_task_only=row["filter_task_only"].lower() == "true",
                )
            )
    return rows


def read_existing_source_summary(output_dir: Path) -> list[SourceSummaryRow]:
    summary_path = output_dir / "source_summary.csv"
    if not summary_path.exists():
        return []
    rows: list[SourceSummaryRow] = []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                SourceSummaryRow(
                    dataset_id=row["dataset_id"],
                    source_file=row["source_file"],
                    dataset_part=row["dataset_part"],
                    series_id=row["series_id"],
                    subject_id=row["subject_id"],
                    source_subject_id=row["source_subject_id"],
                    sampling_rate=int(row["sampling_rate"]),
                    n_rows=int(row["n_rows"]),
                    n_kept_rows=int(row["n_kept_rows"]),
                    n_segments=int(row["n_segments"]),
                    n_records=int(row["n_records"]),
                    n_samples=int(row["n_samples"]),
                    n_fog_samples=int(row["n_fog_samples"]),
                    filter_valid_only=row["filter_valid_only"].lower() == "true",
                    filter_task_only=row["filter_task_only"].lower() == "true",
                    status=row["status"],
                )
            )
    return rows


def next_segment_ids_from_manifest(rows: Iterable[ManifestRow]) -> dict[str, int]:
    next_ids: dict[str, int] = {}
    for row in rows:
        next_ids[row.subject_id] = max(next_ids.get(row.subject_id, 0), row.segment_id + 1)
    return next_ids


def process_members(
    archive: zipfile.ZipFile,
    members: list[tuple[str, str, str]],
    output_dir: Path,
    zip_path: Path,
    series_meta: dict[tuple[str, str], dict[str, object]],
    subject_map: dict[str, str],
    subject_rows: list[dict[str, object]],
    existing_rows: list[ManifestRow],
    existing_source_rows: list[SourceSummaryRow],
    args: argparse.Namespace,
) -> tuple[list[ManifestRow], list[SourceSummaryRow]]:
    records_dir = output_dir / "records"
    manifest_rows = list(existing_rows)
    source_summary_rows = list(existing_source_rows)
    processed_sources = {row.source_file for row in existing_source_rows if row.status == "complete"}
    if not processed_sources:
        processed_sources = {row.source_file for row in existing_rows}
    next_segment_ids = next_segment_ids_from_manifest(existing_rows)
    channel_json = compact_json(list(CHANNELS))
    sensor_positions_json = compact_json(["lower_back"])

    members = apply_smoke_limit(members, args.smoke_limit)
    completed_this_run = 0

    for dataset_part, series_id, member in members:
        if args.resume and member in processed_sources:
            continue
        meta = series_meta.get((dataset_part, series_id), {})
        source_subject_id = str(meta.get("Subject", f"{dataset_part}_{series_id}"))
        subject_id = subject_map.get(source_subject_id)
        if not subject_id:
            subject_id = subject_code(len(subject_map) + 1)
            subject_map[source_subject_id] = subject_id

        sampling_rate = SOURCE_SAMPLING_RATES[dataset_part]
        source_row_count = 0
        source_kept_count = 0
        source_segment_count = 0
        source_record_count = 0
        source_sample_count = 0
        source_fog_count = 0
        active_segment: SegmentBuffer | None = None

        def emit_segment(segment: SegmentBuffer) -> None:
            nonlocal source_segment_count
            nonlocal source_record_count
            nonlocal source_sample_count
            nonlocal source_fog_count
            if segment.n_samples < args.min_samples:
                return

            segment_id = next_segment_ids.get(subject_id, 0)
            next_segment_ids[subject_id] = segment_id + 1
            record_id = f"{subject_id}_seg{segment_id:03d}"
            record_relpath = Path("records") / f"{record_id}.npz"

            x, y_binary = segment.arrays()
            save_record_npz(records_dir / record_relpath.name, x=x, y_binary=y_binary, compression=args.record_compression)

            n_samples = segment.n_samples
            n_fog_samples = segment.n_fog_samples
            source_segment_count += 1
            source_record_count += 1
            source_sample_count += n_samples
            source_fog_count += n_fog_samples
            manifest_rows.append(
                ManifestRow(
                    dataset_id=DATASET_ID,
                    record_id=record_id,
                    source_file=member,
                    dataset_part=dataset_part,
                    series_id=series_id,
                    subject_id=subject_id,
                    source_subject_id=source_subject_id,
                    run_id=f"{dataset_part}_{series_id}",
                    segment_id=segment_id,
                    record_path=record_relpath.as_posix(),
                    sampling_rate=sampling_rate,
                    n_samples=n_samples,
                    duration_sec=round(float(n_samples / sampling_rate), 9),
                    n_normal_samples=int(n_samples - n_fog_samples),
                    n_fog_samples=n_fog_samples,
                    channels=channel_json,
                    sensor_positions=sensor_positions_json,
                    filter_valid_only=args.valid_only,
                    filter_task_only=args.task_only,
                )
            )

        for chunk in iter_series_csv_chunks(archive, member, dataset_part, args.chunk_size):
            try:
                y_chunk = binary_labels(chunk, dataset_part)
            except ValueError as exc:
                raise ValueError(f"{member}: {exc}") from exc
            keep = keep_mask(chunk, dataset_part, args.valid_only, args.task_only)
            source_row_count += int(len(chunk))
            source_kept_count += int(keep.sum())

            if len(chunk) == 0:
                continue

            if active_segment is not None and not bool(keep[0]):
                emit_segment(active_segment)
                active_segment = None

            for start, stop in contiguous_true_segments(keep):
                segment_slice = chunk.iloc[start:stop]
                y_slice = y_chunk[start:stop]

                if active_segment is not None and start == 0:
                    active_segment.append(segment_slice, y_slice, context=member)
                    if stop < len(chunk):
                        emit_segment(active_segment)
                        active_segment = None
                    continue

                segment = SegmentBuffer(x_parts=[], y_parts=[])
                segment.append(segment_slice, y_slice, context=member)
                if stop == len(chunk):
                    active_segment = segment
                else:
                    emit_segment(segment)

        if active_segment is not None:
            emit_segment(active_segment)
            active_segment = None

        source_summary_rows.append(
            SourceSummaryRow(
                dataset_id=DATASET_ID,
                source_file=member,
                dataset_part=dataset_part,
                series_id=series_id,
                subject_id=subject_id,
                source_subject_id=source_subject_id,
                sampling_rate=sampling_rate,
                n_rows=source_row_count,
                n_kept_rows=source_kept_count,
                n_segments=source_segment_count,
                n_records=source_record_count,
                n_samples=source_sample_count,
                n_fog_samples=source_fog_count,
                filter_valid_only=args.valid_only,
                filter_task_only=args.task_only,
                status="complete",
            )
        )
        completed_this_run += 1
        write_processing_metadata(output_dir, zip_path, args, manifest_rows, source_summary_rows, subject_rows)
        if args.stop_after_source_files and completed_this_run >= args.stop_after_source_files:
            raise RuntimeError(f"Stopped after {completed_this_run} source files for checkpoint testing.")

    return manifest_rows, source_summary_rows


def build_loso_rows(manifest_rows: list[ManifestRow]) -> list[LosoRow]:
    subjects = sorted({row.subject_id for row in manifest_rows})
    rows: list[LosoRow] = []
    for test_subject_id in subjects:
        fold_id = f"loso_{test_subject_id}"
        for record in manifest_rows:
            rows.append(
                LosoRow(
                    fold_id=fold_id,
                    test_subject_id=test_subject_id,
                    split="test" if record.subject_id == test_subject_id else "train",
                    record_id=record.record_id,
                    subject_id=record.subject_id,
                    segment_id=record.segment_id,
                )
            )
    return rows


def write_config(
    output_dir: Path,
    zip_path: Path,
    args: argparse.Namespace,
    manifest_rows: list[ManifestRow],
    source_summary_rows: list[SourceSummaryRow],
) -> None:
    config = {
        "dataset_name": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "source_zip": str(zip_path),
        "record_format": "npz",
        "record_compression": args.record_compression,
        "record_contents": ["x", "y_binary"],
        "x_dtype": "float32",
        "y_binary_dtype": "uint8",
        "channels": list(CHANNELS),
        "sensor_positions": ["lower_back"],
        "source_sampling_rates": SOURCE_SAMPLING_RATES,
        "label_rule": {
            "tdcsfog": "max(StartHesitation, Turn, Walking)",
            "defog": "max(StartHesitation, Turn, Walking)",
            "notype": "Event when --source all is used",
        },
        "selected_source": args.source,
        "valid_only": args.valid_only,
        "task_only": args.task_only,
        "strict_metadata": args.strict_metadata,
        "chunk_size": args.chunk_size,
        "min_samples": args.min_samples,
        "metadata_checkpointing": "per_source",
        "windowing": False,
        "pre_fog_labeling": False,
        "normalization": False,
        "resampling": False,
        "split_strategy": "LOSO by source Subject",
        "summary": {
            "record_count": len(manifest_rows),
            "source_file_count": len(source_summary_rows),
            "subject_count": len({row.subject_id for row in manifest_rows}),
            "total_samples": int(sum(row.n_samples for row in manifest_rows)),
            "total_fog_samples": int(sum(row.n_fog_samples for row in manifest_rows)),
            "total_normal_samples": int(sum(row.n_normal_samples for row in manifest_rows)),
            "records_by_source": {
                source: int(sum(1 for row in manifest_rows if row.dataset_part == source))
                for source in sorted({row.dataset_part for row in manifest_rows})
            },
            "source_files_by_source": {
                source: int(sum(1 for row in source_summary_rows if row.dataset_part == source))
                for source in sorted({row.dataset_part for row in source_summary_rows})
            },
        },
        "notes": [
            "The source zip is read directly; CSV files are not extracted.",
            "Train CSV members are processed chunk-by-chunk; only the currently open retained segment is buffered before writing one NPZ record.",
            "manifest.csv, source_summary.csv, subjects.csv, loso_folds.csv, and config.json are checkpointed after each completed source CSV; _SUCCESS.json is written only after the full selected run completes.",
            "Only train CSV files are processed. Test CSV files and unlabeled parquet files are skipped.",
            "Window length, step, Pre-FOG labeling, normalization, and fold-specific imputation are left to training code.",
        ],
    }
    write_json(output_dir / "config.json", config)


def write_success_marker(
    output_dir: Path,
    zip_path: Path,
    args: argparse.Namespace,
    manifest_rows: list[ManifestRow],
    source_summary_rows: list[SourceSummaryRow],
) -> None:
    marker = {
        "status": "complete",
        "dataset_id": DATASET_ID,
        "source_zip": str(zip_path),
        "selected_source": args.source,
        "valid_only": args.valid_only,
        "task_only": args.task_only,
        "strict_metadata": args.strict_metadata,
        "chunk_size": args.chunk_size,
        "min_samples": args.min_samples,
        "record_compression": args.record_compression,
        "smoke_limit": args.smoke_limit,
        "record_count": len(manifest_rows),
        "source_file_count": len(source_summary_rows),
        "subject_count": len({row.subject_id for row in manifest_rows}),
        "total_samples": int(sum(row.n_samples for row in manifest_rows)),
        "total_fog_samples": int(sum(row.n_fog_samples for row in manifest_rows)),
    }
    write_json(output_dir / "_SUCCESS.json", marker)


def write_processing_metadata(
    output_dir: Path,
    zip_path: Path,
    args: argparse.Namespace,
    manifest_rows: list[ManifestRow],
    source_summary_rows: list[SourceSummaryRow],
    subject_rows: list[dict[str, object]],
) -> None:
    write_csv(output_dir / "manifest.csv", manifest_rows, fieldnames=dataclass_fieldnames(ManifestRow))
    write_csv(output_dir / "source_summary.csv", source_summary_rows, fieldnames=dataclass_fieldnames(SourceSummaryRow))
    write_csv(output_dir / "subjects.csv", subject_rows)
    write_csv(output_dir / "loso_folds.csv", build_loso_rows(manifest_rows), fieldnames=dataclass_fieldnames(LosoRow))
    write_config(output_dir, zip_path, args, manifest_rows, source_summary_rows)


def main() -> None:
    args = parse_args()
    zip_path = (args.zip_path or find_default_zip(args.dataset_root)).resolve()
    output_dir = (args.output_dir or default_output_dir(zip_path, args.smoke_limit)).resolve()

    with zipfile.ZipFile(zip_path) as archive:
        series_meta, subject_map, subject_rows = load_metadata(archive)
        members = iter_train_members(archive, args.source)
        if args.dry_run:
            dry_run_summary(archive, zip_path, members, series_meta, subject_map, args)
            return
        if args.strict_metadata:
            validate_strict_metadata(apply_smoke_limit(members, args.smoke_limit), series_meta, subject_map)

    validate_resume_compatibility(output_dir, zip_path, args)
    prepare_output_dir(output_dir, args.overwrite, args.resume)
    existing_rows = read_existing_manifest(output_dir) if args.resume else []
    existing_source_rows = read_existing_source_summary(output_dir) if args.resume else []

    with zipfile.ZipFile(zip_path) as archive:
        series_meta, subject_map, subject_rows = load_metadata(archive)
        members = iter_train_members(archive, args.source)
        manifest_rows, source_summary_rows = process_members(
            archive=archive,
            members=members,
            output_dir=output_dir,
            zip_path=zip_path,
            series_meta=series_meta,
            subject_map=subject_map,
            subject_rows=subject_rows,
            existing_rows=existing_rows,
            existing_source_rows=existing_source_rows,
            args=args,
        )

    write_processing_metadata(output_dir, zip_path, args, manifest_rows, source_summary_rows, subject_rows)
    write_success_marker(output_dir, zip_path, args, manifest_rows, source_summary_rows)

    print(
        f"Wrote {len(manifest_rows)} records for {len({row.subject_id for row in manifest_rows})} subjects "
        f"from {len(source_summary_rows)} source files to {output_dir}"
    )


if __name__ == "__main__":
    main()
