#!/usr/bin/env python
"""Preprocess FoG-STAR into sample-level binary FOG records.

Each record NPZ intentionally contains only ``x`` and ``y_binary``. Windowing,
normalization, resampling, and Pre-FOG labeling are left to downstream training
code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_NAME = "FoG-STAR"
DATASET_VERSION = "3.0"
SAMPLING_RATE = 60
WALKING_ACTIVITY_IDS = (1, 6, 7)
EXCLUDED_ACTIVITY_IDS = (0, 2, 3, 4, 5)
SENSOR_POSITIONS = ("ankleL", "ankleR", "back", "wrist")
CHANNELS = (
    "ankleL_acc_x",
    "ankleL_acc_y",
    "ankleL_acc_z",
    "ankleL_gyro_x",
    "ankleL_gyro_y",
    "ankleL_gyro_z",
    "ankleR_acc_x",
    "ankleR_acc_y",
    "ankleR_acc_z",
    "ankleR_gyro_x",
    "ankleR_gyro_y",
    "ankleR_gyro_z",
    "back_acc_x",
    "back_acc_y",
    "back_acc_z",
    "back_gyro_x",
    "back_gyro_y",
    "back_gyro_z",
    "wrist_acc_x",
    "wrist_acc_y",
    "wrist_acc_z",
    "wrist_gyro_x",
    "wrist_gyro_y",
    "wrist_gyro_z",
)


@dataclass
class ManifestRow:
    dataset_id: str
    record_id: str
    source_file: str
    subject_id: str
    source_subject_id: int
    run_id: str
    segment_id: int
    record_path: str
    sampling_rate: int
    sampling_rate_hz: int
    n_samples: int
    n_normal_samples: int
    n_fog_samples: int
    fog_event_count: int
    has_fog: bool
    usable: bool
    duration_sec: float
    channels: str
    sensor_positions: str


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
        description="Build sample-level binary processed records for FoG-STAR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/6.FoG-STAR"),
        help="Directory containing sensor_data.csv and clinical_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/6.FoG-STAR/processed"),
        help="Output directory for records and metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )
    parser.add_argument(
        "--exclude-no-fog-subjects",
        action="store_true",
        help="Keep only subjects that have at least one FOG sample in retained walking/turning activities.",
    )
    return parser.parse_args()


def subject_code(raw_subject_id: int) -> str:
    return f"S{int(raw_subject_id):02d}"


def run_code(subject_id: str, session_id: int, task_id: int) -> str:
    return f"{subject_id}_sess{int(session_id):02d}_task{int(task_id):02d}"


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def clean_metadata_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(rows[0]).keys()) if not isinstance(rows[0], dict) else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if not isinstance(row, dict) else row)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to rebuild it")
        if not output_dir.name.startswith("processed"):
            raise ValueError(f"Refusing to overwrite unexpected output directory: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "records").mkdir(parents=True, exist_ok=True)


def validate_sensor_frame(df: pd.DataFrame) -> None:
    missing = [column for column in (*CHANNELS, "activity", "fog", "subjectID", "sessionID", "taskID") if column not in df]
    if missing:
        raise ValueError(f"sensor_data.csv is missing required columns: {missing}")
    fog_values = set(pd.unique(df["fog"].dropna()).tolist())
    if not fog_values.issubset({0, 1}):
        raise ValueError(f"Unexpected fog values: {sorted(fog_values)}")


def subjects_with_retained_fog(sensor_df: pd.DataFrame) -> set[int]:
    kept = sensor_df.loc[sensor_df["activity"].isin(WALKING_ACTIVITY_IDS)]
    return set(map(int, kept.loc[kept["fog"] == 1, "subjectID"].unique().tolist()))


def build_records(sensor_df: pd.DataFrame, output_dir: Path) -> list[ManifestRow]:
    records_dir = output_dir / "records"
    manifest_rows: list[ManifestRow] = []
    next_segment_ids: dict[str, int] = {}
    channel_json = compact_json(list(CHANNELS))
    position_json = compact_json(list(SENSOR_POSITIONS))

    grouped = sensor_df.groupby(["subjectID", "sessionID", "taskID"], sort=True)
    for (raw_subject_id, session_id, task_id), run_df in grouped:
        subject_id = subject_code(raw_subject_id)
        run_id = run_code(subject_id, session_id, task_id)
        mask = run_df["activity"].isin(WALKING_ACTIVITY_IDS).to_numpy()

        for start, stop in contiguous_true_segments(mask):
            segment_id = next_segment_ids.get(subject_id, 0)
            next_segment_ids[subject_id] = segment_id + 1
            record_id = f"{subject_id}_seg{segment_id:03d}"
            record_relpath = Path("records") / f"{record_id}.npz"

            segment = run_df.iloc[start:stop]
            x = segment.loc[:, CHANNELS].to_numpy(dtype=np.float32)
            y_binary = segment["fog"].to_numpy(dtype=np.uint8)

            if not set(np.unique(y_binary).tolist()).issubset({0, 1}):
                raise ValueError(f"Unexpected binary labels found in {record_id}")

            np.savez_compressed(output_dir / record_relpath, x=x, y_binary=y_binary)

            n_samples = int(y_binary.size)
            n_fog_samples = int(y_binary.sum())
            n_normal_samples = int(n_samples - n_fog_samples)
            fog_event_count = len(contiguous_true_segments(y_binary.astype(bool)))
            manifest_rows.append(
                ManifestRow(
                    dataset_id=DATASET_NAME,
                    record_id=record_id,
                    source_file="sensor_data.csv",
                    subject_id=subject_id,
                    source_subject_id=int(raw_subject_id),
                    run_id=run_id,
                    segment_id=segment_id,
                    record_path=record_relpath.as_posix(),
                    sampling_rate=SAMPLING_RATE,
                    sampling_rate_hz=SAMPLING_RATE,
                    n_samples=n_samples,
                    n_normal_samples=n_normal_samples,
                    n_fog_samples=n_fog_samples,
                    fog_event_count=fog_event_count,
                    has_fog=n_fog_samples > 0,
                    usable=n_samples > 0,
                    duration_sec=round(float(n_samples / SAMPLING_RATE), 9),
                    channels=channel_json,
                    sensor_positions=position_json,
                )
            )

    return manifest_rows


def build_subject_rows(clinical_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clinical_df = clinical_df.sort_values("subjectID")
    for row in clinical_df.to_dict(orient="records"):
        rows.append(
            {
                "subject_id": subject_code(row["subjectID"]),
                "source_subject_id": int(row["subjectID"]),
                "age": clean_metadata_value(row.get("age", "")),
                "gender": clean_metadata_value(row.get("gender", "")),
                "disease_duration": clean_metadata_value(row.get("disease_duration", "")),
                "h_y": clean_metadata_value(row.get("h_y", "")),
                "updrs_iii": clean_metadata_value(row.get("updrs_iii", "")),
                "fog_q": clean_metadata_value(row.get("fog_q", "")),
                "moca": clean_metadata_value(row.get("moca", "")),
                "fes_i": clean_metadata_value(row.get("fes-i", row.get("fes_i", ""))),
                "pdq8": clean_metadata_value(row.get("pdq8", "")),
            }
        )
    return rows


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
    sensor_df: pd.DataFrame,
    manifest_rows: list[ManifestRow],
    subject_rows: list[dict[str, object]],
    excluded_no_fog_subjects: list[int],
) -> None:
    kept_mask = sensor_df["activity"].isin(WALKING_ACTIVITY_IDS)
    kept_df = sensor_df.loc[kept_mask]
    sensor_nan_frame = sensor_df.loc[:, CHANNELS].isna()
    sensor_nan_mask = sensor_nan_frame.any(axis=1)
    kept_sensor_nan_frame = sensor_nan_frame.loc[kept_mask]
    kept_sensor_nan_mask = sensor_nan_mask.loc[kept_mask]
    total_samples = int(sum(row.n_samples for row in manifest_rows))
    total_fog_samples = int(kept_df["fog"].sum())
    config = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "source_files": ["sensor_data.csv", "clinical_data.csv"],
        "record_format": "npz",
        "record_contents": ["x", "y_binary"],
        "x_dtype": "float32",
        "y_binary_dtype": "uint8",
        "sampling_rate": SAMPLING_RATE,
        "label_source": "fog",
        "negative_label": 0,
        "positive_label": 1,
        "walking_activity_ids": list(WALKING_ACTIVITY_IDS),
        "excluded_activity_ids": list(EXCLUDED_ACTIVITY_IDS),
        "activity_labels": {
            "0": "unlabeled_or_unknown",
            "1": "Walk",
            "2": "Sit",
            "3": "Stand",
            "4": "Sit-to-stand",
            "5": "Stand-to-sit",
            "6": "Turn-right",
            "7": "Turn-left",
        },
        "task_labels": {
            "1": "Timed Up-and-Go",
            "2": "Stand 1min",
            "3": "Walk back/forth",
            "4": "Walk+Doorway",
            "5": "Walk+Water",
            "6": "Walk+Count",
            "7": "360 turn",
        },
        "channels": list(CHANNELS),
        "sensor_positions": list(SENSOR_POSITIONS),
        "record_segmentation": "Within each subject/session/task run, keep contiguous samples whose activity is in [1,6,7]; segment_id is subject-level and zero-based.",
        "subject_filtering": {
            "exclude_no_fog_subjects": bool(excluded_no_fog_subjects),
            "no_fog_definition": "subject has zero FOG samples after retaining walking/turning activities [1,6,7]",
            "excluded_source_subject_ids": excluded_no_fog_subjects,
            "excluded_subject_ids": [subject_code(subject) for subject in excluded_no_fog_subjects],
        },
        "split_strategy": "LOSO by subject",
        "windowing": False,
        "pre_fog_labeling": False,
        "normalization": False,
        "resampling": False,
        "missing_value_policy": "preserve_nan",
        "summary": {
            "subject_count": len(subject_rows),
            "record_count": len(manifest_rows),
            "total_samples": total_samples,
            "total_normal_samples": int(total_samples - total_fog_samples),
            "total_fog_samples": total_fog_samples,
            "records_with_fog": 0,
            "excluded_no_fog_subject_count": len(excluded_no_fog_subjects),
            "excluded_no_fog_subjects": [subject_code(subject) for subject in excluded_no_fog_subjects],
            "source_rows_with_any_sensor_nan": int(sensor_nan_mask.sum()),
            "kept_rows_with_any_sensor_nan": int(kept_sensor_nan_mask.sum()),
            "kept_fog_rows_with_any_sensor_nan": int(
                (kept_sensor_nan_mask.to_numpy() & (kept_df["fog"].to_numpy() == 1)).sum()
            ),
            "source_sensor_nan_cells": int(sensor_nan_frame.sum().sum()),
            "kept_sensor_nan_cells": int(kept_sensor_nan_frame.sum().sum()),
            "kept_activity_counts": {str(k): int(v) for k, v in kept_df["activity"].value_counts().sort_index().items()},
            "kept_task_ids": [int(v) for v in sorted(kept_df["taskID"].unique().tolist())],
        },
        "notes": [
            "Records are sample-level time series only; no windows or Pre-FOG labels are generated.",
            "Task-level filtering is not applied; any sample labeled Walk, Turn-right, or Turn-left is retained.",
            "Activity value 0 appears in sensor_data.csv but is not documented as a walking activity, so it is excluded.",
            "Sensor NaN values are preserved because the dataset does not provide an official imputation/drop rule.",
        ],
    }

    # Compute this separately so config summary does not need per-record labels.
    records_with_fog = 0
    for row in manifest_rows:
        with np.load(output_dir / row.record_path) as record_npz:
            records_with_fog += int(record_npz["y_binary"].sum() > 0)
    config["summary"]["records_with_fog"] = records_with_fog

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def write_success_marker(output_dir: Path, manifest_rows: list[ManifestRow], subject_rows: list[dict[str, object]]) -> None:
    payload = {
        "status": "complete",
        "dataset_name": DATASET_NAME,
        "record_count": len(manifest_rows),
        "subject_count": len(subject_rows),
        "total_samples": int(sum(row.n_samples for row in manifest_rows)),
        "total_fog_samples": int(sum(row.n_fog_samples for row in manifest_rows)),
    }
    with (output_dir / "_SUCCESS.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    sensor_path = data_dir / "sensor_data.csv"
    clinical_path = data_dir / "clinical_data.csv"

    prepare_output_dir(output_dir, args.overwrite)

    sensor_df = pd.read_csv(sensor_path)
    validate_sensor_frame(sensor_df)
    clinical_df = pd.read_csv(clinical_path)

    excluded_no_fog_subjects: list[int] = []
    if args.exclude_no_fog_subjects:
        fog_subjects = subjects_with_retained_fog(sensor_df)
        all_subjects = set(map(int, sensor_df["subjectID"].unique().tolist()))
        excluded_no_fog_subjects = sorted(all_subjects - fog_subjects)
        sensor_df = sensor_df.loc[sensor_df["subjectID"].isin(sorted(fog_subjects))].copy()
        clinical_df = clinical_df.loc[clinical_df["subjectID"].isin(sorted(fog_subjects))].copy()
        if len(fog_subjects) < 2:
            raise ValueError("At least two FOG-positive subjects are needed for LOSO after filtering.")

    manifest_rows = build_records(sensor_df, output_dir)
    subject_rows = build_subject_rows(clinical_df)
    loso_rows = build_loso_rows(manifest_rows)

    write_csv(output_dir / "manifest.csv", manifest_rows)
    write_csv(output_dir / "subjects.csv", subject_rows)
    write_csv(output_dir / "loso_folds.csv", loso_rows)
    write_config(output_dir, sensor_df, manifest_rows, subject_rows, excluded_no_fog_subjects)
    write_success_marker(output_dir, manifest_rows, subject_rows)

    total_samples = int(sum(row.n_samples for row in manifest_rows))
    print(
        f"Wrote {len(manifest_rows)} records, {len(subject_rows)} subjects, "
        f"{len(loso_rows)} LOSO rows, {total_samples} samples to {output_dir}"
    )
    if excluded_no_fog_subjects:
        print(
            "Excluded no-FOG subjects: "
            + ", ".join(subject_code(subject) for subject in excluded_no_fog_subjects)
        )


if __name__ == "__main__":
    main()
