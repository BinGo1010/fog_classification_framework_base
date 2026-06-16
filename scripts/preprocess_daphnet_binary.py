#!/usr/bin/env python
"""Preprocess Daphnet FOG data into binary sample-level records.

The output intentionally keeps each record minimal: every NPZ contains only
``x`` and ``y_binary``. Metadata, sensor layout, LOSO folds, and FOG events are
stored alongside the records.
"""

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


DATASET_ID = "daphnet"
SAMPLING_RATE_HZ = 64
RAW_COLUMNS = (
    "time_ms",
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
    "annotation",
)
CHANNELS = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)


@dataclass
class ManifestRow:
    dataset_id: str
    record_id: str
    record_path: str
    source_file: str
    subject_id: str
    run_id: str
    segment_id: int
    source_start_row: int
    source_end_row: int
    sampling_rate_hz: int
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
    record_id: str
    subject_id: str
    run_id: str
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
        description="Build binary sample-level processed records for Daphnet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/1.Daphnet Freezing of Gait Dataset/dataset"),
        help="Directory containing Daphnet S<subject>R<run>.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/1.Daphnet Freezing of Gait Dataset/processed"),
        help="Output directory for records and metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )
    return parser.parse_args()


def parse_source_name(path: Path) -> tuple[str, str]:
    match = re.fullmatch(r"S(?P<subject>\d{2})R(?P<run>\d{2})\.txt", path.name)
    if not match:
        raise ValueError(f"Unexpected Daphnet filename: {path.name}")
    return f"S{match.group('subject')}", f"R{match.group('run')}"


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


def process_file(
    path: Path,
    records_dir: Path,
    next_segment_ids: dict[str, int],
) -> tuple[list[ManifestRow], list[FogEventRow]]:
    subject_id, run_id = parse_source_name(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    annotation = df["annotation"].to_numpy(dtype=np.int8)
    experiment_segments = contiguous_true_segments(annotation != 0)

    manifest_rows: list[ManifestRow] = []
    event_rows: list[FogEventRow] = []

    for segment_id, (start, stop) in enumerate(experiment_segments):
        subject_segment_id = next_segment_ids.get(subject_id, 0)
        next_segment_ids[subject_id] = subject_segment_id + 1

        segment = df.iloc[start:stop].copy()
        raw_annotation = segment["annotation"].to_numpy(dtype=np.int8)
        y_binary = (raw_annotation == 2).astype(np.int8)
        x = (segment.loc[:, CHANNELS].to_numpy(dtype=np.float32) / 1000.0).astype(np.float32)

        record_id = f"{subject_id}_seg{subject_segment_id:03d}"
        record_path = Path("records") / f"{record_id}.npz"
        np.savez_compressed(records_dir / record_path.name, x=x, y_binary=y_binary)

        fog_event_count, events = count_fog_events(y_binary)
        n_samples = int(y_binary.size)
        n_fog_samples = int(y_binary.sum())
        n_normal_samples = int(n_samples - n_fog_samples)
        duration_sec = float(n_samples / SAMPLING_RATE_HZ)

        manifest_rows.append(
            ManifestRow(
                dataset_id=DATASET_ID,
                record_id=record_id,
                record_path=record_path.as_posix(),
                source_file=path.name,
                subject_id=subject_id,
                run_id=run_id,
                segment_id=subject_segment_id,
                source_start_row=start,
                source_end_row=stop - 1,
                sampling_rate_hz=SAMPLING_RATE_HZ,
                n_samples=n_samples,
                duration_sec=duration_sec,
                n_normal_samples=n_normal_samples,
                n_fog_samples=n_fog_samples,
                fog_event_count=fog_event_count,
                has_fog=n_fog_samples > 0,
                usable=n_samples > 0,
                notes="task protocol is mixed/unknown in released txt files",
            )
        )

        for event_id, (event_start, event_stop) in enumerate(events):
            start_time_sec = float(event_start / SAMPLING_RATE_HZ)
            end_time_sec = float((event_stop - 1) / SAMPLING_RATE_HZ)
            event_rows.append(
                FogEventRow(
                    dataset_id=DATASET_ID,
                    record_id=record_id,
                    subject_id=subject_id,
                    run_id=run_id,
                    segment_id=subject_segment_id,
                    event_id=event_id,
                    start_index=event_start,
                    end_index=event_stop - 1,
                    start_time_sec=start_time_sec,
                    end_time_sec=end_time_sec,
                    duration_sec=float((event_stop - event_start) / SAMPLING_RATE_HZ),
                )
            )

    return manifest_rows, event_rows


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
    channels = []
    for name in CHANNELS:
        sensor, modality, axis = name.split("_", 2)
        channels.append(
            {
                "name": name,
                "sensor": sensor,
                "modality": modality,
                "axis": axis,
                "unit": "g",
            }
        )
    schema = {
        "dataset_id": DATASET_ID,
        "source": "Daphnet Freezing of Gait Dataset",
        "record_format": "npz",
        "record_arrays": {
            "x": "[time, channel] float32 IMU acceleration in g",
            "y_binary": "[time] int8, 0=NORMAL, 1=FOG",
        },
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "source_signal_unit": "mg",
        "processed_signal_unit": "g",
        "label_binary": {"0": "NORMAL", "1": "FOG"},
        "source_annotation": {
            "0": "not part of experiment; excluded from records",
            "1": "experiment, no freeze",
            "2": "freeze",
        },
        "channels": channels,
        "task_protocol": {
            "released_txt_task_labels": False,
            "description": (
                "The documentation lists straight walking, numerous turns, and ADL tasks, "
                "but the released txt files do not include per-sample task ids."
            ),
        },
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
        "excluded_annotation_0": True,
        "record_arrays": ["x", "y_binary"],
        "uncertainties": [
            "The released Daphnet txt files do not contain per-sample a/b/c walking-task labels.",
            "Annotation 1 can include stand, walk, or turn; all experiment/non-freeze samples are kept as NORMAL.",
            "FOG boundaries were annotated from video and may have a few hundred milliseconds of jitter, per dataset documentation.",
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
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    records_dir.mkdir(parents=True)

    source_files = sorted(data_dir.glob("S??R??.txt"))
    if not source_files:
        raise FileNotFoundError(f"No Daphnet txt files found in {data_dir}")

    manifest_rows: list[ManifestRow] = []
    event_rows: list[FogEventRow] = []
    next_segment_ids: dict[str, int] = {}
    for source_file in source_files:
        file_manifest_rows, file_event_rows = process_file(source_file, records_dir, next_segment_ids)
        manifest_rows.extend(file_manifest_rows)
        event_rows.extend(file_event_rows)

    write_csv(output_dir / "manifest.csv", manifest_rows)
    write_csv(output_dir / "fog_events.csv", event_rows)
    write_csv(output_dir / "loso_folds.csv", build_loso_rows(manifest_rows))
    write_schema(output_dir)
    write_report(output_dir, source_files, manifest_rows, event_rows)

    print(
        f"Wrote {len(manifest_rows)} records from {len(source_files)} source files "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
