#!/usr/bin/env python
"""Preprocess Multimodal FOG filtered data into binary sample-level records."""

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


DATASET_ID = "multimodal_fog"
SAMPLING_RATE_HZ = 500
SENSORS = ("lshank", "rshank", "waist", "arm")
SIGNALS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

# Actual files have 61 columns: exported index, TIME, 25 EEG, 5 EMG,
# 4 * (6 IMU channels + NC/SC), and Label.
IMU_USECOLS = (
    32,
    33,
    34,
    35,
    36,
    37,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    55,
    56,
    57,
    58,
)
LABEL_COL = 60


@dataclass
class ManifestRow:
    dataset_id: str
    record_id: str
    record_path: str
    source_file: str
    subject_id: str
    source_subject_dir: str
    session_id: str
    task_id: str
    segment_id: int
    source_start_row: int
    source_end_row: int
    source_start_time: str
    source_end_time: str
    sampling_rate_hz: int
    n_samples: int
    duration_sec: float
    n_normal_samples: int
    n_fog_samples: int
    fog_event_count: int
    has_fog: bool
    usable: bool
    all_zero_channels: str
    notes: str


@dataclass
class FogEventRow:
    dataset_id: str
    record_id: str
    subject_id: str
    session_id: str
    task_id: str
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
        description="Build binary sample-level processed records for Multimodal FOG filtered data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/4.Multimodal Dataset/Filtered Data"),
        help="Filtered Data directory containing subject/task_n.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/4.Multimodal Dataset/processed"),
        help="Output directory for records and metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )
    return parser.parse_args()


def channel_names() -> list[str]:
    return [f"{sensor}_{signal}" for sensor in SENSORS for signal in SIGNALS]


def schema_channels() -> list[dict[str, str]]:
    channels: list[dict[str, str]] = []
    for sensor in SENSORS:
        for signal in SIGNALS:
            modality, axis = signal.split("_", 1)
            channels.append(
                {
                    "name": f"{sensor}_{signal}",
                    "sensor": sensor,
                    "modality": "accelerometer" if modality == "acc" else "gyroscope",
                    "axis": axis,
                    "unit": "filtered_source_value",
                    "source_column_note": "NC/SC columns are excluded",
                }
            )
    return channels


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


def last_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        pos = handle.tell()
        if pos == 0:
            return ""
        buffer = bytearray()
        pos -= 1
        while pos >= 0:
            handle.seek(pos)
            char = handle.read(1)
            if char == b"\n" and buffer:
                break
            if char not in (b"\n", b"\r"):
                buffer.extend(char)
            pos -= 1
        return buffer[::-1].decode("utf-8", errors="ignore")


def first_and_last_times(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
    last = last_line(path).strip()
    first_parts = first.split(",") if first else []
    last_parts = last.split(",") if last else []
    first_time = first_parts[1] if len(first_parts) > 1 else ""
    last_time = last_parts[1] if len(last_parts) > 1 else ""
    return first_time, last_time


def parse_task_file(path: Path, data_dir: Path) -> tuple[str, str, str, str]:
    rel = path.relative_to(data_dir)
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected filtered data path: {path}")
    subject_dir = parts[0]
    session_id = "default" if len(parts) == 2 else "/".join(parts[1:-1])
    task_match = re.fullmatch(r"task_(?P<task>\d+)\.txt", parts[-1])
    if not task_match:
        raise ValueError(f"Unexpected task filename: {path.name}")
    subject_id = f"M{int(subject_dir):03d}"
    task_id = f"task_{int(task_match.group('task'))}"
    return subject_id, subject_dir, session_id, task_id


def process_file(
    path: Path,
    data_dir: Path,
    records_dir: Path,
    next_segment_ids: dict[str, int],
) -> tuple[ManifestRow, list[FogEventRow]]:
    subject_id, subject_dir, session_id, task_id = parse_task_file(path, data_dir)
    segment_id = next_segment_ids.get(subject_id, 0)
    next_segment_ids[subject_id] = segment_id + 1

    usecols = [*IMU_USECOLS, LABEL_COL]
    df = pd.read_csv(path, header=None, usecols=usecols)
    if df.shape[1] != len(usecols):
        raise ValueError(f"{path} did not yield expected selected columns")

    x = df.loc[:, list(IMU_USECOLS)].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path} has non-finite selected IMU values")

    label_values = df.loc[:, LABEL_COL].to_numpy()
    if pd.isna(label_values).any():
        raise ValueError(f"{path} has missing label values")
    observed_labels = sorted({int(v) for v in pd.Series(label_values).dropna().unique()})
    if not set(observed_labels).issubset({0, 1}):
        raise ValueError(f"{path} has unexpected label values: {observed_labels}")
    y_binary = label_values.astype(np.int8)

    record_id = f"{subject_id}_seg{segment_id:03d}"
    record_path = Path("records") / f"{record_id}.npz"
    np.savez_compressed(records_dir / record_path.name, x=x.astype(np.float32), y_binary=y_binary)

    fog_event_count, events = count_fog_events(y_binary)
    n_samples = int(y_binary.size)
    n_fog_samples = int(y_binary.sum())
    n_normal_samples = int(n_samples - n_fog_samples)
    first_time, final_time = first_and_last_times(path)

    zero_channels = [
        name
        for name, is_zero in zip(channel_names(), np.all(x == 0, axis=0))
        if bool(is_zero)
    ]
    source_file = path.relative_to(data_dir).as_posix()

    manifest_row = ManifestRow(
        dataset_id=DATASET_ID,
        record_id=record_id,
        record_path=record_path.as_posix(),
        source_file=source_file,
        subject_id=subject_id,
        source_subject_dir=subject_dir,
        session_id=session_id,
        task_id=task_id,
        segment_id=segment_id,
        source_start_row=0,
        source_end_row=n_samples - 1,
        source_start_time=first_time,
        source_end_time=final_time,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        n_samples=n_samples,
        duration_sec=float(n_samples / SAMPLING_RATE_HZ),
        n_normal_samples=n_normal_samples,
        n_fog_samples=n_fog_samples,
        fog_event_count=fog_event_count,
        has_fog=n_fog_samples > 0,
        usable=n_samples > 0,
        all_zero_channels=";".join(zero_channels),
        notes="one filtered task file is treated as one labeled task record",
    )

    event_rows: list[FogEventRow] = []
    for event_id, (event_start, event_stop) in enumerate(events):
        event_rows.append(
            FogEventRow(
                dataset_id=DATASET_ID,
                record_id=record_id,
                subject_id=subject_id,
                session_id=session_id,
                task_id=task_id,
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


def write_schema(output_dir: Path) -> None:
    schema = {
        "dataset_id": DATASET_ID,
        "source": "Multimodal Dataset of Freezing of Gait in Parkinson's Disease",
        "record_format": "npz",
        "record_arrays": {
            "x": "[time, channel] float32 selected IMU channels from filtered task files",
            "y_binary": "[time] int8, 0=NORMAL, 1=FOG",
        },
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "label_binary": {"0": "NORMAL", "1": "FOG"},
        "selected_modalities": ["accelerometer", "gyroscope"],
        "excluded_modalities": ["EEG", "EMG", "ECG", "EOG", "NC", "SC"],
        "channels": schema_channels(),
        "source_file_columns": {
            "0": "exported row index; excluded",
            "1": "TIME string; stored only in manifest start/end fields",
            "2-26": "EEG; excluded",
            "27-31": "EMG/ECG/EOG; excluded",
            "32-59": "four inertial/SC sensor groups; only acc/gyro columns selected",
            "60": "Label, 0=FOG-free, 1=FOG",
        },
        "notes": [
            "Values are retained in the filtered dataset's source scale; no unit conversion, normalization, windowing, or pre-FOG labeling is applied.",
            "Some participants may have all-zero channels for sensors that were not worn; these are reported per record in manifest.csv.",
            "Subject 008 has OFF_1 and OFF_2 sessions, both assigned to subject_id M008 for LOSO.",
        ],
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_report(
    output_dir: Path,
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
        "uncertainties": [
            "Filtered task files provide source-scale numeric IMU values; no reliable physical unit conversion is documented for these columns.",
            "Only IMU accelerometer and gyroscope columns are selected for x; EEG/EMG/ECG/EOG/NC/SC are intentionally excluded for the current IMU FOG framework.",
            "All-zero selected channels are retained and reported because the README states some participants wore only two inertial sensors.",
        ],
    }
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    records_dir = output_dir / "records"

    if not data_dir.exists():
        raise FileNotFoundError(f"Filtered Data directory not found: {data_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    records_dir.mkdir(parents=True)

    source_files = sorted(data_dir.rglob("task_*.txt"))
    if not source_files:
        raise FileNotFoundError(f"No task_*.txt files found in {data_dir}")

    manifest_rows: list[ManifestRow] = []
    event_rows: list[FogEventRow] = []
    next_segment_ids: dict[str, int] = {}

    for idx, source_file in enumerate(source_files, 1):
        print(f"[multimodal] {idx}/{len(source_files)} {source_file.relative_to(data_dir)}")
        manifest_row, file_event_rows = process_file(
            source_file, data_dir, records_dir, next_segment_ids
        )
        manifest_rows.append(manifest_row)
        event_rows.extend(file_event_rows)

    write_csv(output_dir / "manifest.csv", manifest_rows)
    write_csv(output_dir / "fog_events.csv", event_rows)
    write_csv(output_dir / "loso_folds.csv", build_loso_rows(manifest_rows))
    write_schema(output_dir)
    write_report(output_dir, source_files, manifest_rows, event_rows)

    print(f"Wrote {len(manifest_rows)} records from {len(source_files)} task files to {output_dir}")


if __name__ == "__main__":
    main()
