#!/usr/bin/env python
"""Preprocess Stanford IMU FOG data into binary sample-level records."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ID = "stanford_imu_fog"
SAMPLING_RATE_HZ = 128
GRAVITY = 9.80665
AXES = ("x", "y", "z")
STREAMS = ("acc", "gyro")
SUBSETS = {
    "imus6_subjects7": ("chest", "lumbar", "ankle_l", "ankle_r", "foot_l", "foot_r"),
    "imus11_subjects4": (
        "chest",
        "lumbar",
        "ankle_l",
        "ankle_r",
        "foot_l",
        "foot_r",
        "head",
        "thigh_l",
        "thigh_r",
        "wrist_l",
        "wrist_r",
    ),
}


@dataclass
class ManifestRow:
    dataset_id: str
    subset_id: str
    record_id: str
    record_path: str
    source_file: str
    source_subset_dir: str
    subject_id: str
    segment_id: int
    visit: str
    source_condition_token: str
    trial_id: str
    source_start_row: int
    source_end_row: int
    source_start_time: float
    source_end_time: float
    sampling_rate_hz: int
    estimated_sampling_rate_hz: float
    n_samples: int
    duration_sec: float
    n_normal_samples: int
    n_fog_samples: int
    fog_event_count: int
    has_fog: bool
    usable: bool
    notes: str


@dataclass
class FogEventRow:
    dataset_id: str
    subset_id: str
    record_id: str
    subject_id: str
    segment_id: int
    event_id: int
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float


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
        description="Build binary sample-level processed records for Stanford IMU FOG data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/5.Stanford imu-fog-detection/data"),
        help="Stanford data directory containing raw/ and config workbooks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/5.Stanford imu-fog-detection/processed"),
        help="Output directory containing one processed folder per subset.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=sorted(SUBSETS),
        default=sorted(SUBSETS),
        help="Raw subsets to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete selected subset output directories first if they already exist.",
    )
    return parser.parse_args()


def channel_columns(sensors: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for sensor in sensors:
        columns.extend(f"imu_{sensor}_a{axis}" for axis in AXES)
        columns.extend(f"imu_{sensor}_g{axis}" for axis in AXES)
    return columns


def schema_channels(sensors: tuple[str, ...]) -> list[dict[str, str]]:
    channels: list[dict[str, str]] = []
    for sensor in sensors:
        for axis in AXES:
            channels.append(
                {
                    "name": f"imu_{sensor}_a{axis}",
                    "sensor": sensor,
                    "modality": "accelerometer",
                    "axis": axis,
                    "unit": "g",
                    "source_unit": "m/s^2",
                }
            )
        for axis in AXES:
            channels.append(
                {
                    "name": f"imu_{sensor}_g{axis}",
                    "sensor": sensor,
                    "modality": "gyroscope",
                    "axis": axis,
                    "unit": "rad/s",
                    "source_unit": "rad/s",
                }
            )
    return channels


def parse_source_name(path: Path) -> tuple[str, str, str, str]:
    pattern = re.compile(
        r"pt(?P<subject>\d+)_visit_(?P<visit>[^_]+)_tbc_walklr_"
        r"(?P<condition>[^_]+)_trial_(?P<trial>\d+)\.xlsx"
    )
    match = pattern.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected Stanford filename: {path.name}")
    subject_id = f"P{int(match.group('subject')):02d}"
    return subject_id, match.group("visit"), match.group("condition"), match.group("trial")


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


def count_fog_events(y_binary: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    events = contiguous_true_segments(y_binary.astype(bool))
    return len(events), events


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def estimate_sampling_rate(time_values: np.ndarray) -> float:
    if time_values.size < 2:
        return float("nan")
    diffs = np.diff(time_values.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    return float(1.0 / np.median(diffs))


def process_file(
    path: Path,
    subset_id: str,
    records_dir: Path,
    sensors: tuple[str, ...],
    next_segment_ids: dict[str, int],
) -> tuple[ManifestRow, list[FogEventRow]]:
    subject_id, visit, condition, trial = parse_source_name(path)
    segment_id = next_segment_ids.get(subject_id, 0)
    next_segment_ids[subject_id] = segment_id + 1

    channels = channel_columns(sensors)
    required_columns = ["subject_ID", "time", *channels, "freeze_label"]
    df = pd.read_excel(path, usecols=required_columns)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    df = df.loc[:, required_columns]

    label_values = df["freeze_label"].to_numpy()
    observed_labels = sorted({int(v) for v in pd.Series(label_values).dropna().unique()})
    if not set(observed_labels).issubset({0, 1}):
        raise ValueError(f"{path.name} has unexpected freeze_label values: {observed_labels}")
    if pd.isna(label_values).any():
        raise ValueError(f"{path.name} has missing freeze_label values")

    x = df.loc[:, channels].to_numpy(dtype=np.float32)
    if np.isnan(x).any():
        nan_columns = df.loc[:, channels].columns[df.loc[:, channels].isna().any()].tolist()
        raise ValueError(f"{path.name} has missing selected IMU values in columns: {nan_columns}")

    acc_indices = [idx for idx, column in enumerate(channels) if re.search(r"_a[xyz]$", column)]
    x[:, acc_indices] = x[:, acc_indices] / np.float32(GRAVITY)
    y_binary = label_values.astype(np.int8)

    record_id = f"{subject_id}_seg{segment_id:03d}"
    record_path = Path("records") / f"{record_id}.npz"
    np.savez_compressed(records_dir / record_path.name, x=x.astype(np.float32), y_binary=y_binary)

    fog_event_count, events = count_fog_events(y_binary)
    n_samples = int(y_binary.size)
    n_fog_samples = int(y_binary.sum())
    n_normal_samples = int(n_samples - n_fog_samples)
    time_values = df["time"].to_numpy(dtype=np.float64)
    estimated_hz = estimate_sampling_rate(time_values)

    manifest_row = ManifestRow(
        dataset_id=DATASET_ID,
        subset_id=subset_id,
        record_id=record_id,
        record_path=record_path.as_posix(),
        source_file=path.name,
        source_subset_dir=subset_id,
        subject_id=subject_id,
        segment_id=segment_id,
        visit=visit,
        source_condition_token=condition,
        trial_id=trial,
        source_start_row=0,
        source_end_row=n_samples - 1,
        source_start_time=float(time_values[0]) if n_samples else float("nan"),
        source_end_time=float(time_values[-1]) if n_samples else float("nan"),
        sampling_rate_hz=SAMPLING_RATE_HZ,
        estimated_sampling_rate_hz=estimated_hz,
        n_samples=n_samples,
        duration_sec=float(n_samples / SAMPLING_RATE_HZ),
        n_normal_samples=n_normal_samples,
        n_fog_samples=n_fog_samples,
        fog_event_count=fog_event_count,
        has_fog=n_fog_samples > 0,
        usable=n_samples > 0,
        notes="one source workbook is treated as one walking-trial record",
    )

    event_rows: list[FogEventRow] = []
    for event_id, (event_start, event_stop) in enumerate(events):
        event_rows.append(
            FogEventRow(
                dataset_id=DATASET_ID,
                subset_id=subset_id,
                record_id=record_id,
                subject_id=subject_id,
                segment_id=segment_id,
                event_id=event_id,
                start_index=event_start,
                end_index=event_stop - 1,
                start_time_sec=float(event_start / SAMPLING_RATE_HZ),
                end_time_sec=float((event_stop - 1) / SAMPLING_RATE_HZ),
                duration_sec=float((event_stop - event_start) / SAMPLING_RATE_HZ),
            )
        )

    return manifest_row, event_rows


def build_loso_rows(manifest_rows: list[ManifestRow]) -> list[LosoRow]:
    subjects = sorted({row.subject_id for row in manifest_rows})
    rows: list[LosoRow] = []
    for test_subject in subjects:
        fold_id = f"loso_{test_subject}"
        for record in manifest_rows:
            rows.append(
                LosoRow(
                    fold_id=fold_id,
                    test_subject_id=test_subject,
                    split="test" if record.subject_id == test_subject else "train",
                    record_id=record.record_id,
                    subject_id=record.subject_id,
                    segment_id=record.segment_id,
                )
            )
    return rows


def write_schema(output_dir: Path, subset_id: str, sensors: tuple[str, ...]) -> None:
    schema = {
        "dataset_id": DATASET_ID,
        "subset_id": subset_id,
        "source": "Stanford imu-fog-detection",
        "record_format": "npz",
        "record_arrays": {
            "x": "[time, channel] float32 IMU data; acceleration in g, gyroscope in rad/s",
            "y_binary": "[time] int8, 0=NORMAL, 1=FOG",
        },
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "source_acceleration_unit": "m/s^2",
        "processed_acceleration_unit": "g",
        "gyroscope_unit": "rad/s",
        "label_binary": {"0": "NORMAL", "1": "FOG"},
        "selected_sensors": list(sensors),
        "channels": schema_channels(sensors),
        "notes": [
            "Each source workbook is one walking trial.",
            "No windows or pre-FOG labels are materialized in processed records.",
        ],
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_report(
    output_dir: Path,
    subset_id: str,
    source_files: list[Path],
    manifest_rows: list[ManifestRow],
    event_rows: list[FogEventRow],
) -> None:
    subject_ids = sorted({row.subject_id for row in manifest_rows})
    subjects_without_fog = sorted(
        subject_id
        for subject_id in subject_ids
        if sum(row.n_fog_samples for row in manifest_rows if row.subject_id == subject_id) == 0
    )
    report = {
        "dataset_id": DATASET_ID,
        "subset_id": subset_id,
        "source_file_count": len(source_files),
        "record_count": len(manifest_rows),
        "subject_count": len(subject_ids),
        "subjects": subject_ids,
        "subjects_without_fog": subjects_without_fog,
        "total_samples": int(sum(row.n_samples for row in manifest_rows)),
        "total_normal_samples": int(sum(row.n_normal_samples for row in manifest_rows)),
        "total_fog_samples": int(sum(row.n_fog_samples for row in manifest_rows)),
        "fog_event_count": len(event_rows),
        "record_arrays": ["x", "y_binary"],
        "acceleration_converted_from_mps2_to_g": True,
        "uncertainties": [
            "The source filename token walklr_0/1 is retained as source_condition_token but not interpreted.",
            "Each workbook is treated as one continuous walking-trial record unless missing values or unexpected labels are found.",
            "The imus11_subjects4 files are also present in imus6_subjects7; subsets are processed separately to avoid mixing duplicate trials.",
        ],
    }
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_subset(data_root: Path, output_root: Path, subset_id: str, overwrite: bool) -> None:
    sensors = SUBSETS[subset_id]
    raw_dir = data_root / "raw" / subset_id
    output_dir = output_root / subset_id
    records_dir = output_dir / "records"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw subset directory not found: {raw_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    records_dir.mkdir(parents=True)

    source_files = sorted(raw_dir.glob("pt*.xlsx"))
    if not source_files:
        raise FileNotFoundError(f"No Stanford xlsx files found in {raw_dir}")

    manifest_rows: list[ManifestRow] = []
    event_rows: list[FogEventRow] = []
    next_segment_ids: dict[str, int] = {}

    for idx, source_file in enumerate(source_files, 1):
        print(f"[{subset_id}] {idx}/{len(source_files)} {source_file.name}")
        manifest_row, file_event_rows = process_file(
            source_file, subset_id, records_dir, sensors, next_segment_ids
        )
        manifest_rows.append(manifest_row)
        event_rows.extend(file_event_rows)

    write_csv(output_dir / "manifest.csv", manifest_rows)
    write_csv(output_dir / "fog_events.csv", event_rows)
    write_csv(output_dir / "loso_folds.csv", build_loso_rows(manifest_rows))
    write_schema(output_dir, subset_id, sensors)
    write_report(output_dir, subset_id, source_files, manifest_rows, event_rows)

    print(f"Wrote {len(manifest_rows)} {subset_id} records to {output_dir}")


def main() -> None:
    args = parse_args()
    for subset_id in args.subsets:
        process_subset(args.data_root, args.output_dir, subset_id, args.overwrite)


if __name__ == "__main__":
    main()
