from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def write_zip_csv(archive: zipfile.ZipFile, name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        archive.writestr(name, "")
        return
    fieldnames = list(rows[0].keys())
    lines = []
    sink = _ListWriter(lines)
    writer = csv.DictWriter(sink, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    archive.writestr(name, "".join(lines))


class _ListWriter:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def write(self, value: str) -> None:
        self.lines.append(value)


def make_synthetic_kaggle_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [
                {"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"},
                {"Id": "td002", "Subject": "subjA", "Visit": 1, "Test": 2, "Medication": "on"},
            ],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [
                {"Id": "df001", "Subject": "subjB", "Visit": 1, "Medication": "off"},
                {"Id": "df002", "Subject": "subjB", "Visit": 1, "Medication": "off"},
                {"Id": "df003", "Subject": "subjB", "Visit": 1, "Medication": "off"},
            ],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])

        write_zip_csv(
            archive,
            "train/tdcsfog/td001.csv",
            [
                {"Time": 0, "AccV": 1.0, "AccML": 0.1, "AccAP": -0.1, "StartHesitation": 0, "Turn": 0, "Walking": 0},
                {"Time": 1, "AccV": 2.0, "AccML": 0.2, "AccAP": -0.2, "StartHesitation": 1, "Turn": 0, "Walking": 0},
                {"Time": 2, "AccV": 3.0, "AccML": 0.3, "AccAP": -0.3, "StartHesitation": 0, "Turn": 0, "Walking": 0},
            ],
        )
        write_zip_csv(
            archive,
            "train/tdcsfog/td002.csv",
            [
                {"Time": 0, "AccV": 4.0, "AccML": 0.4, "AccAP": -0.4, "StartHesitation": 0, "Turn": 1, "Walking": 0},
                {"Time": 1, "AccV": 5.0, "AccML": 0.5, "AccAP": -0.5, "StartHesitation": 0, "Turn": 0, "Walking": 0},
            ],
        )
        write_zip_csv(
            archive,
            "train/defog/df001.csv",
            [
                {
                    "Time": 0,
                    "AccV": 10.0,
                    "AccML": 1.0,
                    "AccAP": -1.0,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "false",
                    "Task": "false",
                },
                {
                    "Time": 1,
                    "AccV": 11.0,
                    "AccML": 1.1,
                    "AccAP": -1.1,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "true",
                    "Task": "true",
                },
                {
                    "Time": 2,
                    "AccV": 12.0,
                    "AccML": 1.2,
                    "AccAP": -1.2,
                    "StartHesitation": 0,
                    "Turn": 1,
                    "Walking": 0,
                    "Valid": "true",
                    "Task": "true",
                },
                {
                    "Time": 3,
                    "AccV": 13.0,
                    "AccML": 1.3,
                    "AccAP": -1.3,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 1,
                    "Valid": "true",
                    "Task": "false",
                },
                {
                    "Time": 4,
                    "AccV": 14.0,
                    "AccML": 1.4,
                    "AccAP": -1.4,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 1,
                    "Valid": "true",
                    "Task": "true",
                },
            ],
        )
        write_zip_csv(
            archive,
            "train/defog/df002.csv",
            [
                {
                    "Time": 0,
                    "AccV": 20.0,
                    "AccML": 2.0,
                    "AccAP": -2.0,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 1,
                    "Valid": "true",
                    "Task": "true",
                },
                {
                    "Time": 1,
                    "AccV": 21.0,
                    "AccML": 2.1,
                    "AccAP": -2.1,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "true",
                    "Task": "true",
                },
            ],
        )
        write_zip_csv(
            archive,
            "train/defog/df003.csv",
            [
                {
                    "Time": 0,
                    "AccV": 30.0,
                    "AccML": 3.0,
                    "AccAP": -3.0,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "false",
                    "Task": "true",
                },
                {
                    "Time": 1,
                    "AccV": 31.0,
                    "AccML": 3.1,
                    "AccAP": -3.1,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 1,
                    "Valid": "true",
                    "Task": "false",
                },
            ],
        )


def test_kaggle_zip_inventory_reports_path_buckets(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    output_dir = tmp_path / "inventory"
    make_synthetic_kaggle_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "inspect_kaggle_fog_zip.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "path_buckets:" in result.stdout
    inventory = pd.read_csv(output_dir / "kaggle_zip_inventory.csv")
    summary = json.loads((output_dir / "kaggle_zip_inventory_summary.json").read_text(encoding="utf-8"))
    assert "path_bucket" in inventory.columns
    assert summary["groups"]["tdcsfog"]["file_count"] == 2
    assert summary["groups"]["defog"]["file_count"] == 3
    assert summary["groups"]["metadata"]["file_count"] == 6
    assert summary["path_buckets"]["train/tdcsfog"]["file_count"] == 2
    assert summary["path_buckets"]["train/defog"]["file_count"] == 3
    assert summary["path_buckets"]["/"]["file_count"] == 7
    assert not (tmp_path / "processed").exists()


def run_synthetic_kaggle_preprocess(zip_path: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--record-compression",
            "none",
            "--strict-metadata",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def run_processed_validator(processed_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_processed_records.py"),
            str(processed_dir),
            "--expected-channels",
            "3",
            "--require-success",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_validate_processed_records_rejects_schema_inconsistencies(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    processed_dir = tmp_path / "processed"
    make_synthetic_kaggle_zip(zip_path)
    run_synthetic_kaggle_preprocess(zip_path, processed_dir)

    valid = run_processed_validator(processed_dir)
    assert valid.returncode == 0, valid.stderr

    unsafe_dir = tmp_path / "processed_unsafe_path"
    shutil.copytree(processed_dir, unsafe_dir)
    manifest = pd.read_csv(unsafe_dir / "manifest.csv")
    manifest.loc[0, "record_path"] = "../outside.npz"
    manifest.to_csv(unsafe_dir / "manifest.csv", index=False)
    unsafe = run_processed_validator(unsafe_dir)
    assert unsafe.returncode != 0
    assert "record_path escapes processed_dir" in unsafe.stderr

    loso_mismatch_dir = tmp_path / "processed_loso_mismatch"
    shutil.copytree(processed_dir, loso_mismatch_dir)
    loso = pd.read_csv(loso_mismatch_dir / "loso_folds.csv")
    loso.loc[0, "subject_id"] = "S999"
    loso.to_csv(loso_mismatch_dir / "loso_folds.csv", index=False)
    loso_mismatch = run_processed_validator(loso_mismatch_dir)
    assert loso_mismatch.returncode != 0
    assert "does not match manifest" in loso_mismatch.stderr

    dtype_mismatch_dir = tmp_path / "processed_dtype_mismatch"
    shutil.copytree(processed_dir, dtype_mismatch_dir)
    manifest = pd.read_csv(dtype_mismatch_dir / "manifest.csv")
    record_path = dtype_mismatch_dir / str(manifest.iloc[0]["record_path"])
    with np.load(record_path) as record:
        x = record["x"].astype(np.float64)
        y = record["y_binary"]
    np.savez(record_path, x=x, y_binary=y)
    dtype_mismatch = run_processed_validator(dtype_mismatch_dir)
    assert dtype_mismatch.returncode != 0
    assert "x dtype" in dtype_mismatch.stderr


def test_kaggle_streaming_preprocess_smoke(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    output_dir = tmp_path / "processed"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--chunk-size",
            "2",
            "--record-compression",
            "none",
            "--strict-metadata",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_processed_records.py"),
            str(output_dir),
            "--expected-channels",
            "3",
            "--require-success",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    validation_summary = json.loads(validation.stdout)
    assert validation_summary["source_summary"] is True
    assert validation_summary["source_summary_checked_against_manifest"] is True
    assert validation_summary["source_files"] == 5
    assert validation_summary["zero_record_sources"] == 1

    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert len(manifest) == 5
    assert set(manifest["dataset_part"]) == {"tdcsfog", "defog"}
    assert manifest["n_samples"].sum() == 10
    assert manifest["n_fog_samples"].sum() == 5
    source_summary = pd.read_csv(output_dir / "source_summary.csv")
    assert len(source_summary) == 5
    zero_record_source = source_summary[source_summary["source_file"] == "train/defog/df003.csv"].iloc[0]
    assert zero_record_source["status"] == "complete"
    assert zero_record_source["n_kept_rows"] == 0
    assert zero_record_source["n_records"] == 0
    boundary_source = source_summary[source_summary["source_file"] == "train/defog/df001.csv"].iloc[0]
    assert boundary_source["n_records"] == 2
    assert boundary_source["n_samples"] == 3
    tdcs_boundary_source = source_summary[source_summary["source_file"] == "train/tdcsfog/td001.csv"].iloc[0]
    assert tdcs_boundary_source["n_records"] == 1
    assert tdcs_boundary_source["n_samples"] == 3

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["windowing"] is False
    assert config["pre_fog_labeling"] is False
    assert config["record_compression"] == "none"
    assert config["strict_metadata"] is True
    assert config["summary"]["source_file_count"] == 5
    success = json.loads((output_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["status"] == "complete"
    assert success["record_count"] == 5
    assert success["source_file_count"] == 5
    assert success["record_compression"] == "none"
    assert success["strict_metadata"] is True
    assert not list(output_dir.rglob("*.tmp*"))

    first_record = manifest.sort_values("record_id").iloc[0]
    with zipfile.ZipFile(output_dir / first_record["record_path"]) as npz_archive:
        assert {info.compress_type for info in npz_archive.infolist()} == {zipfile.ZIP_STORED}
    with np.load(output_dir / first_record["record_path"]) as record:
        assert set(record.files) == {"x", "y_binary"}
        assert record["x"].shape[1] == 3

    window_dir = tmp_path / "windows_dry_run"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_processed_record_windows.py"),
            "--processed-dir",
            str(output_dir),
            "--output-dir",
            str(window_dir),
            "--window-seconds",
            "0.01",
            "--stride-seconds",
            "0.01",
            "--label-mode",
            "binary",
            "--target-hz",
            "100",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert (window_dir / "config.json").exists()
    assert (window_dir / "file_summary.csv").exists()
    assert not (window_dir / "windows.npz").exists()

    materialized_window_dir = tmp_path / "windows_materialized"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_processed_record_windows.py"),
            "--processed-dir",
            str(output_dir),
            "--output-dir",
            str(materialized_window_dir),
            "--window-seconds",
            "0.01",
            "--stride-seconds",
            "0.01",
            "--label-mode",
            "binary",
            "--target-hz",
            "100",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_window_dataset.py"),
            str(materialized_window_dir),
            "--expected-channels",
            "3",
            "--expected-classes",
            "2",
            "--allow-empty-train",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with np.load(materialized_window_dir / "windows.npz") as windows:
        assert windows["X"].shape == (10, 1, 3)
        assert windows["y"].shape == (10,)

    source_summary.loc[source_summary["source_file"] == "train/defog/df001.csv", "n_records"] = 99
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    corrupted_validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_processed_records.py"),
            str(output_dir),
            "--expected-channels",
            "3",
            "--require-success",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert corrupted_validation.returncode != 0
    assert "n_records" in corrupted_validation.stderr


def test_kaggle_dry_run_does_not_create_output_and_limits_per_source(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    dry_run_report_path = tmp_path / "dry_run_report.json"
    make_synthetic_kaggle_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--smoke-limit",
            "1",
            "--check-headers",
            "--strict-metadata",
            "--dry-run",
            "--dry-run-output-json",
            str(dry_run_report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert not output_dir.exists()
    assert "dry_run: true" in result.stdout
    assert "selected_train_csv_files: 2" in result.stdout
    assert "members_missing_metadata: 0" in result.stdout
    assert "headers_checked: 2" in result.stdout
    assert "members_with_header_issues: 0" in result.stdout
    assert "defog: files=1" in result.stdout
    assert "tdcsfog: files=1" in result.stdout
    dry_run_report = json.loads(dry_run_report_path.read_text(encoding="utf-8"))
    assert dry_run_report["dry_run"] is True
    assert dry_run_report["zip_path"] == str(zip_path.resolve())
    assert dry_run_report["zip_size"] == zip_path.stat().st_size
    assert dry_run_report["zip_modified_time_ns"] == zip_path.stat().st_mtime_ns
    assert dry_run_report["check_headers"] is True
    assert dry_run_report["strict_metadata"] is True
    assert dry_run_report["selected_train_csv_files"] == 2
    assert dry_run_report["by_source"]["defog"]["files"] == 1
    assert dry_run_report["by_source"]["tdcsfog"]["files"] == 1


def test_kaggle_dry_run_profile_data_reports_nan_without_output(tmp_path: Path) -> None:
    zip_path = tmp_path / "profile_nan_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    report_path = tmp_path / "profile_report.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td001.csv",
            [
                {"Time": 0, "AccV": 1.0, "AccML": None, "AccAP": -0.1, "StartHesitation": 0, "Turn": 0, "Walking": 0},
                {"Time": 1, "AccV": 2.0, "AccML": 0.2, "AccAP": -0.2, "StartHesitation": 0, "Turn": 1, "Walking": 0},
                {"Time": 2, "AccV": 3.0, "AccML": 0.3, "AccAP": -0.3, "StartHesitation": 0, "Turn": 0, "Walking": 0},
            ],
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "tdcsfog",
            "--check-headers",
            "--strict-metadata",
            "--dry-run",
            "--profile-data",
            "--chunk-size",
            "2",
            "--dry-run-output-json",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "profile_data: True" in result.stdout
    assert (
        "profile_overall: files=1 rows=3 kept=3 normal=2 fog=1 "
        "normal_sec=0.015625 fog_sec=0.0078125 x_nan=1 x_nonfinite=1"
    ) in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    profile = report["profile"]["overall"]
    assert profile["files_profiled"] == 1
    assert profile["chunks"] == 2
    assert profile["rows"] == 3
    assert profile["kept_rows"] == 3
    assert profile["normal_samples"] == 2
    assert profile["fog_samples"] == 1
    assert profile["profiled_duration_sec"] == 0.0234375
    assert profile["kept_duration_sec"] == 0.0234375
    assert profile["normal_duration_sec"] == 0.015625
    assert profile["fog_duration_sec"] == 0.0078125
    assert profile["x_nan_values"] == 1
    assert profile["x_nonfinite_values"] == 1
    assert profile["x_nan_by_channel"]["AccML"] == 1
    assert profile["x_kept_nan_by_channel"]["AccML"] == 1
    assert not output_dir.exists()

    materialized_output_dir = tmp_path / "processed_bad_x_should_not_succeed"
    materialized = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(materialized_output_dir),
            "--source",
            "tdcsfog",
            "--strict-metadata",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert materialized.returncode != 0
    assert "features contain NaN or non-finite values" in materialized.stderr
    assert "train/tdcsfog/td001.csv" in materialized.stderr
    assert not (materialized_output_dir / "_SUCCESS.json").exists()
    records_dir = materialized_output_dir / "records"
    assert not records_dir.exists() or list(records_dir.glob("*.npz")) == []


def test_kaggle_dry_run_profile_data_reports_label_nan_without_crashing(tmp_path: Path) -> None:
    zip_path = tmp_path / "profile_label_nan_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    report_path = tmp_path / "profile_label_report.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td001.csv",
            [
                {"Time": 0, "AccV": 1.0, "AccML": 0.1, "AccAP": -0.1, "StartHesitation": 0, "Turn": 0, "Walking": 0},
                {"Time": 1, "AccV": 2.0, "AccML": 0.2, "AccAP": -0.2, "StartHesitation": 0, "Turn": None, "Walking": 0},
                {"Time": 2, "AccV": 3.0, "AccML": 0.3, "AccAP": -0.3, "StartHesitation": 0, "Turn": 0, "Walking": 1},
            ],
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "tdcsfog",
            "--strict-metadata",
            "--dry-run",
            "--profile-data",
            "--dry-run-output-json",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "headers_checked: 1" in result.stdout
    assert "label_invalid=1" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    profile = report["profile"]["overall"]
    assert report["check_headers"] is True
    assert profile["rows"] == 3
    assert profile["kept_rows"] == 3
    assert profile["label_nan_values"] == 1
    assert profile["label_nonfinite_values"] == 1
    assert profile["label_invalid_rows"] == 1
    assert profile["kept_label_invalid_rows"] == 1
    assert profile["normal_samples"] == 1
    assert profile["fog_samples"] == 1
    assert not output_dir.exists()


def test_kaggle_dry_run_profile_data_reports_nonbinary_labels(tmp_path: Path) -> None:
    zip_path = tmp_path / "profile_nonbinary_label_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    report_path = tmp_path / "profile_nonbinary_label_report.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td001.csv",
            [
                {"Time": 0, "AccV": 1.0, "AccML": 0.1, "AccAP": -0.1, "StartHesitation": 0, "Turn": 0, "Walking": 0},
                {"Time": 1, "AccV": 2.0, "AccML": 0.2, "AccAP": -0.2, "StartHesitation": 0, "Turn": 2, "Walking": 0},
                {"Time": 2, "AccV": 3.0, "AccML": 0.3, "AccAP": -0.3, "StartHesitation": 0, "Turn": 0, "Walking": 1},
            ],
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "tdcsfog",
            "--strict-metadata",
            "--dry-run",
            "--profile-data",
            "--dry-run-output-json",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "label_nonbinary=1" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    profile = report["profile"]["overall"]
    assert profile["rows"] == 3
    assert profile["kept_rows"] == 3
    assert profile["label_invalid_rows"] == 0
    assert profile["label_nonbinary_values"] == 1
    assert profile["label_nonbinary_rows"] == 1
    assert profile["kept_label_nonbinary_rows"] == 1
    assert profile["normal_samples"] == 1
    assert profile["fog_samples"] == 1
    assert not output_dir.exists()

    materialized_output_dir = tmp_path / "processed_nonbinary_should_not_succeed"
    materialized = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(materialized_output_dir),
            "--source",
            "tdcsfog",
            "--strict-metadata",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert materialized.returncode != 0
    assert "labels must be binary 0/1 values" in materialized.stderr
    assert "train/tdcsfog/td001.csv" in materialized.stderr
    assert not (materialized_output_dir / "_SUCCESS.json").exists()
    records_dir = materialized_output_dir / "records"
    assert not records_dir.exists() or list(records_dir.glob("*.npz")) == []


def test_kaggle_strict_metadata_fails_before_output_creation(tmp_path: Path) -> None:
    zip_path = tmp_path / "missing_metadata_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    dry_run_report_path = tmp_path / "missing_metadata_dry_run.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        for series_id in ("td001", "td_missing"):
            write_zip_csv(
                archive,
                f"train/tdcsfog/{series_id}.csv",
                [
                    {
                        "Time": 0,
                        "AccV": 1.0,
                        "AccML": 0.1,
                        "AccAP": -0.1,
                        "StartHesitation": 0,
                        "Turn": 0,
                        "Walking": 0,
                    }
                ],
            )

    dry_run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "tdcsfog",
            "--check-headers",
            "--strict-metadata",
            "--dry-run",
            "--dry-run-output-json",
            str(dry_run_report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert dry_run.returncode != 0
    assert "members_missing_metadata: 1" in dry_run.stdout
    assert "Strict metadata check failed" in dry_run.stderr
    assert "train/tdcsfog/td_missing.csv" in dry_run.stderr
    dry_run_report = json.loads(dry_run_report_path.read_text(encoding="utf-8"))
    assert dry_run_report["members_missing_metadata"] == 1
    assert dry_run_report["metadata_issue_count"] == 1
    assert "train/tdcsfog/td_missing.csv" in dry_run_report["metadata_issues"][0]
    assert not output_dir.exists()

    materialized = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "tdcsfog",
            "--strict-metadata",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert materialized.returncode != 0
    assert "Strict metadata check failed" in materialized.stderr
    assert not output_dir.exists()


def test_kaggle_dry_run_header_check_fails_on_missing_columns(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad_header_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    dry_run_report_path = tmp_path / "bad_header_dry_run.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                }
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td001.csv",
            [
                {"Time": 0, "AccV": 1.0, "AccML": 0.1, "AccAP": -0.1, "StartHesitation": 0, "Turn": 0}
            ],
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "tdcsfog",
            "--check-headers",
            "--dry-run",
            "--dry-run-output-json",
            str(dry_run_report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "CSV header check failed" in result.stderr
    assert "Walking" in result.stderr
    dry_run_report = json.loads(dry_run_report_path.read_text(encoding="utf-8"))
    assert dry_run_report["members_with_header_issues"] == 1
    assert "Walking" in dry_run_report["header_issues"][0]
    assert not output_dir.exists()


def test_kaggle_overwrite_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    output_dir = tmp_path / "processed_should_not_exist"
    make_synthetic_kaggle_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--overwrite",
            "--resume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr
    assert not output_dir.exists()


def test_kaggle_storage_estimate_uses_zip_directory_only(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    report_path = tmp_path / "storage_report.json"
    smoke_report_path = tmp_path / "storage_smoke_report.json"
    processed_dir = tmp_path / "processed"
    make_synthetic_kaggle_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "estimate_kaggle_fog_storage.py"),
            "--zip-path",
            str(zip_path),
            "--source",
            "both",
            "--suite-config",
            str(ROOT / "configs" / "kaggle_full_suite.json"),
            "--output-json",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selected_train_csv_files"] == 5
    assert report["smoke_limit"] == 0
    assert report["available_train_csv_files"] == 5
    assert report["source_counts"] == {"defog": 3, "tdcsfog": 2}
    assert report["available_source_counts"] == {"defog": 3, "tdcsfog": 2}
    assert report["selected_uncompressed_bytes"] > 0
    assert report["estimated_processed_budget_bytes"] == report["selected_uncompressed_bytes"]
    assert len(report["window_budgets"]) == 1
    assert "status:" in result.stdout
    assert not processed_dir.exists()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "estimate_kaggle_fog_storage.py"),
            "--zip-path",
            str(zip_path),
            "--source",
            "both",
            "--smoke-limit",
            "1",
            "--output-json",
            str(smoke_report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    assert smoke_report["smoke_limit"] == 1
    assert smoke_report["available_train_csv_files"] == 5
    assert smoke_report["selected_train_csv_files"] == 2
    assert smoke_report["source_counts"] == {"defog": 1, "tdcsfog": 1}
    assert smoke_report["available_source_counts"] == {"defog": 3, "tdcsfog": 2}
    assert smoke_report["selected_uncompressed_bytes"] < report["selected_uncompressed_bytes"]
    assert not processed_dir.exists()


def test_kaggle_preflight_writes_structured_json_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "preflight_report.json"
    make_synthetic_kaggle_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--skip-pytest",
            "--suite-config",
            "configs/kaggle_full_suite.json",
            "--output-json",
            str(report_path),
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "preflight_report_json:" in result.stdout
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["processed_output_guard"]["no_processed_output_created"] is True
    assert report["extracted_competition_data"]["exists"] is False
    assert report["zip_inventory"]["total_file_count"] > 0
    assert report["zip_structure"]["ok"] is True
    assert report["zip_structure"]["selected_supervised_train_csv_files"] == 5
    assert report["zip_structure"]["required_path_buckets"]["train/tdcsfog"]["csv_count"] == 2
    assert report["zip_structure"]["required_path_buckets"]["train/defog"]["csv_count"] == 3
    assert report["zip_structure"]["required_metadata_files"]["subjects.csv"]["exists"] is True
    assert report["storage_estimate"]["selected_train_csv_files"] == 5
    assert report["storage_estimate"]["zip_size"] == zip_path.stat().st_size
    assert report["storage_estimate"]["zip_modified_time_ns"] == zip_path.stat().st_mtime_ns
    assert {
        Path(item["suite_config"]).name for item in report["storage_estimate"]["window_budgets"]
    } == {"kaggle_full_suite.json"}
    assert report["streaming_dry_run"]["dry_run"] is True
    assert report["streaming_dry_run"]["selected_train_csv_files"] == 5
    assert report["streaming_dry_run"]["members_missing_metadata"] == 0
    assert report["streaming_dry_run"]["headers_checked"] == 5
    assert report["streaming_dry_run"]["check_headers"] is True
    assert report["streaming_dry_run"]["by_source"]["defog"]["files"] == 3
    assert report["streaming_dry_run"]["by_source"]["tdcsfog"]["files"] == 2
    assert report["suite_preflight"]["ok"] is True
    assert Path(report["suite_preflight"]["suite_config"]).name == "kaggle_full_suite.json"
    assert report["suite_preflight"]["errors"] == []
    assert any(
        warning["message"] == "Processed directory does not exist"
        for warning in report["suite_preflight"]["warnings"]
    )
    assert not any(
        warning["message"].startswith("Window output")
        for warning in report["suite_preflight"]["warnings"]
    )
    assert Path(report["suite_dry_run"]["config"]).name == "kaggle_full_suite.json"
    assert report["suite_dry_run"]["validated_experiment_configs"] is True
    assert report["pytest"]["ran"] is False
    assert all(step["status"] == "passed" for step in report["steps"])


def test_kaggle_preflight_smoke_limit_aligns_storage_and_streaming(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "smoke_limit_preflight_report.json"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--skip-pytest",
            "--suite-config",
            "configs/kaggle_smoke_suite.json",
            "--smoke-limit",
            "1",
            "--output-json",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["preflight_options"]["smoke_limit"] == 1
    assert report["zip_structure"]["selected_supervised_train_csv_files"] == 5
    assert report["storage_estimate"]["selected_train_csv_files"] == 2
    assert report["storage_estimate"]["source_counts"] == {"defog": 1, "tdcsfog": 1}
    assert report["streaming_dry_run"]["selected_train_csv_files"] == 2
    assert report["streaming_dry_run"]["smoke_limit"] == 1
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()


def test_kaggle_powershell_preflight_writes_structured_json_report(tmp_path: Path) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not available")

    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "powershell_preflight_report.json"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.ps1"),
            "-RepoRoot",
            str(ROOT),
            "-DatasetRoot",
            str(dataset_root),
            "-SuiteConfig",
            "configs/kaggle_full_suite.json",
            "-SkipPytest",
            "-OutputJson",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert report_path.exists()
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert report["status"] == "passed"
    assert report["processed_output_guard"]["no_processed_output_created"] is True
    assert report["zip_structure"]["ok"] is True
    assert report["zip_structure"]["selected_supervised_train_csv_files"] == 5
    assert report["storage_estimate"]["selected_train_csv_files"] == 5
    assert report["streaming_dry_run"]["selected_train_csv_files"] == 5
    assert report["suite_preflight"]["ok"] is True
    assert report["suite_preflight"]["errors"] == []
    assert Path(report["suite_dry_run"]["config"]).name == "kaggle_full_suite.json"
    assert report["suite_dry_run"]["validated_experiment_configs"] is True
    assert all(step["status"] == "passed" for step in report["steps"])

    status_path = tmp_path / "status_from_powershell_preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--preflight-json",
            str(report_path),
            "--dry-run-json",
            str(tmp_path / "missing_smoke_dry_run.json"),
            "--full-dry-run-json",
            str(tmp_path / "missing_full_dry_run.json"),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["preflight"]["exists"] is True
    assert status["preflight"]["status"] == "passed"
    assert status["preflight"]["zip_structure"]["exists"] is True
    assert status["preflight"]["zip_structure"]["ok"] is True
    assert status["preflight"]["suite_preflight"]["ok"] is True


def test_kaggle_powershell_preflight_smoke_limit_aligns_storage_and_streaming(tmp_path: Path) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not available")

    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "powershell_smoke_limit_preflight_report.json"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.ps1"),
            "-RepoRoot",
            str(ROOT),
            "-DatasetRoot",
            str(dataset_root),
            "-SuiteConfig",
            "configs/kaggle_smoke_suite.json",
            "-SmokeLimit",
            "1",
            "-SkipPytest",
            "-OutputJson",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert report["status"] == "passed"
    assert report["preflight_options"]["smoke_limit"] == 1
    assert report["zip_structure"]["selected_supervised_train_csv_files"] == 5
    assert report["storage_estimate"]["selected_train_csv_files"] == 2
    assert report["storage_estimate"]["source_counts"] == {"defog": 1, "tdcsfog": 1}
    assert report["streaming_dry_run"]["selected_train_csv_files"] == 2
    assert report["streaming_dry_run"]["smoke_limit"] == 1
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()


def test_kaggle_preflight_fails_on_missing_required_train_bucket(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "missing_bucket_preflight_report.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td001", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df001", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td001.csv",
            [
                {"Time": 0, "AccV": 1.0, "AccML": 0.1, "AccAP": -0.1, "StartHesitation": 0, "Turn": 0, "Walking": 0}
            ],
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--skip-pytest",
            "--output-json",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "preflight_report_json:" in result.stdout
    assert report_path.exists()
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["processed_output_guard"]["no_processed_output_created"] is True
    assert report["zip_structure"]["ok"] is False
    assert "Missing required supervised train CSV bucket: train/defog" in report["zip_structure"]["errors"]
    assert report["storage_estimate"] is None
    assert any(
        step["name"] == "Validate zip supervised structure" and step["status"] == "failed"
        for step in report["steps"]
    )


def test_kaggle_powershell_preflight_writes_failure_json_report(tmp_path: Path) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not available")

    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "powershell_failed_preflight_report.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td_ok", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td_missing.csv",
            [
                {
                    "Time": 0,
                    "AccV": 1.0,
                    "AccML": 0.1,
                    "AccAP": -0.1,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                }
            ],
        )
        write_zip_csv(
            archive,
            "train/defog/df_dummy.csv",
            [
                {
                    "Time": 0,
                    "AccV": 2.0,
                    "AccML": 0.2,
                    "AccAP": -0.2,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "true",
                    "Task": "true",
                }
            ],
        )

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.ps1"),
            "-RepoRoot",
            str(ROOT),
            "-DatasetRoot",
            str(dataset_root),
            "-SkipPytest",
            "-OutputJson",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert report_path.exists()
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert report["status"] == "failed"
    assert report["processed_output_guard"]["no_processed_output_created"] is True
    assert report["streaming_dry_run"]["members_missing_metadata"] == 1
    assert any(step["name"] == "Streaming dry-run only" and step["status"] == "failed" for step in report["steps"])
    assert report["error"]["type"] in {"RuntimeException", "Exception"}


def test_kaggle_preflight_fails_when_storage_is_insufficient(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "insufficient_storage_preflight_report.json"
    make_synthetic_kaggle_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--skip-pytest",
            "--suite-config",
            "configs/kaggle_full_suite.json",
            "--reserve-gib",
            "1000000000",
            "--output-json",
            str(report_path),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "preflight_report_json:" in result.stdout
    assert report_path.exists()
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["storage_estimate"]["status"] == "insufficient_free_space"
    assert report["processed_output_guard"]["no_processed_output_created"] is True
    assert any(
        step["name"] == "Estimate supervised storage budget" and step["status"] == "failed"
        for step in report["steps"]
    )
    assert report["error"]["type"] == "CalledProcessError"


def test_kaggle_preflight_writes_failure_json_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    report_path = tmp_path / "failed_preflight_report.json"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td_ok", "Subject": "subjA", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_dummy", "Subject": "subjB", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjA",
                    "Visit": 1,
                    "Age": 65,
                    "Sex": "M",
                    "YearsSinceDx": 7,
                    "UPDRSIII_On": 20,
                    "UPDRSIII_Off": 30,
                    "NFOGQ": 10,
                },
                {
                    "Subject": "subjB",
                    "Visit": 1,
                    "Age": 70,
                    "Sex": "F",
                    "YearsSinceDx": 9,
                    "UPDRSIII_On": 25,
                    "UPDRSIII_Off": 35,
                    "NFOGQ": 12,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/tdcsfog/td_missing.csv",
            [
                {
                    "Time": 0,
                    "AccV": 1.0,
                    "AccML": 0.1,
                    "AccAP": -0.1,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                }
            ],
        )
        write_zip_csv(
            archive,
            "train/defog/df_dummy.csv",
            [
                {
                    "Time": 0,
                    "AccV": 2.0,
                    "AccML": 0.2,
                    "AccAP": -0.2,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "true",
                    "Task": "true",
                }
            ],
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_kaggle_fog_preflight.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--skip-pytest",
            "--output-json",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "preflight_report_json:" in result.stdout
    assert report_path.exists()
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["processed_output_guard"]["no_processed_output_created"] is True
    assert report["streaming_dry_run"]["members_missing_metadata"] == 1
    assert report["streaming_dry_run"]["metadata_issue_count"] == 1
    assert "train/tdcsfog/td_missing.csv" in report["streaming_dry_run"]["metadata_issues"][0]
    assert report["error"]["type"] == "CalledProcessError"
    assert any(step["name"] == "Streaming dry-run only" and step["status"] == "failed" for step in report["steps"])


def test_kaggle_python_smoke_launcher_defaults_to_safe_dry_run(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    dry_run_path = tmp_path / "launcher_dry_run.json"
    log_path = tmp_path / "launcher.log"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_smoke_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--smoke-limit",
            "1",
            "--no-preflight",
            "--no-suite",
            "--dry-run-json",
            str(dry_run_path),
            "--log-path",
            str(log_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    assert dry_run["dry_run"] is True
    assert dry_run["selected_train_csv_files"] == 2
    assert dry_run["headers_checked"] == 2
    log_text = log_path.read_text(encoding="utf-8")
    assert "start Kaggle smoke pipeline execute=False" in log_text
    assert "selected_train_csv_files: 2" in log_text
    assert "members_with_header_issues: 0" in log_text
    assert "skip processed_smoke validation because --execute was not provided" in log_text


def test_kaggle_python_smoke_launcher_can_profile_data_in_safe_dry_run(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    dry_run_path = tmp_path / "launcher_profile_dry_run.json"
    log_path = tmp_path / "launcher_profile.log"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_smoke_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--smoke-limit",
            "1",
            "--no-preflight",
            "--no-suite",
            "--profile-data",
            "--dry-run-json",
            str(dry_run_path),
            "--log-path",
            str(log_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    profile = dry_run["profile"]["overall"]
    assert dry_run["profile_data"] is True
    assert dry_run["headers_checked"] == 2
    assert profile["files_profiled"] == 2
    assert profile["rows"] == 8
    assert profile["kept_rows"] == 6
    assert profile["normal_samples"] == 3
    assert profile["fog_samples"] == 3
    assert profile["profiled_duration_sec"] == 0.0734375
    assert profile["kept_duration_sec"] == 0.0534375
    assert profile["normal_duration_sec"] == 0.025625
    assert profile["fog_duration_sec"] == 0.0278125
    log_text = log_path.read_text(encoding="utf-8")
    assert "--profile-data" in log_text
    assert "profile_overall: files=2 rows=8 kept=6 normal=3 fog=3 normal_sec=0.025625 fog_sec=0.0278125" in log_text


def test_kaggle_python_full_launcher_defaults_to_safe_dry_run(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    dry_run_path = tmp_path / "full_launcher_dry_run.json"
    log_path = tmp_path / "full_launcher.log"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_full_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--no-preflight",
            "--no-suite",
            "--dry-run-json",
            str(dry_run_path),
            "--log-path",
            str(log_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    assert dry_run["dry_run"] is True
    assert dry_run["selected_train_csv_files"] == 5
    assert dry_run["headers_checked"] == 5
    assert dry_run["by_source"]["defog"]["files"] == 3
    assert dry_run["by_source"]["tdcsfog"]["files"] == 2
    log_text = log_path.read_text(encoding="utf-8")
    assert "start Kaggle full pipeline execute=False" in log_text
    assert "selected_train_csv_files: 5" in log_text
    assert "members_with_header_issues: 0" in log_text
    assert "skip processed validation because --execute was not provided" in log_text


def test_kaggle_python_launchers_can_post_check_window_dry_run_after_execute(tmp_path: Path) -> None:
    existing_tmp_outputs = set((ROOT / "outputs").glob("_tmp_window_dry_run_*"))
    cases = [
        (
            "start_kaggle_smoke_pipeline.py",
            ["--smoke-limit", "1"],
            "processed_smoke",
        ),
        (
            "start_kaggle_full_pipeline.py",
            [],
            "processed",
        ),
    ]
    for launcher, extra_args, processed_name in cases:
        dataset_root = tmp_path / launcher / "dataset"
        kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
        kaggle_dir.mkdir(parents=True)
        zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
        dry_run_path = tmp_path / launcher / "streaming_dry_run.json"
        log_path = tmp_path / launcher / "launcher.log"
        make_synthetic_kaggle_zip(zip_path)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / launcher),
                "--repo",
                str(ROOT),
                "--dataset-root",
                str(dataset_root),
                "--execute",
                "--overwrite",
                "--no-preflight",
                "--allow-execute-without-preflight",
                "--allow-execute-without-status-gate",
                "--no-suite",
                "--post-check-window-dry-run",
                "--record-compression",
                "none",
                "--dry-run-json",
                str(dry_run_path),
                "--log-path",
                str(log_path),
                *extra_args,
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        processed_dir = kaggle_dir / processed_name
        assert (processed_dir / "_SUCCESS.json").exists()
        assert (processed_dir / "manifest.csv").exists()
        assert list((processed_dir / "records").glob("*.npz"))
        log_text = log_path.read_text(encoding="utf-8")
        assert "check_processed_pipeline.py" in log_text
        assert "--require-success" in log_text
        assert "Processed pipeline check passed." in log_text
        assert set((ROOT / "outputs").glob("_tmp_window_dry_run_*")) == existing_tmp_outputs


def test_kaggle_launchers_require_preflight_for_execute(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    make_synthetic_kaggle_zip(zip_path)

    common_args = [
        "--repo",
        str(ROOT),
        "--dataset-root",
        str(dataset_root),
        "--execute",
        "--no-preflight",
        "--no-suite",
        "--log-path",
        str(tmp_path / "blocked_launcher.log"),
    ]
    for launcher in ("start_kaggle_full_pipeline.py", "start_kaggle_smoke_pipeline.py"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / launcher),
                *common_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert "--execute with --no-preflight requires --allow-execute-without-preflight" in result.stderr

    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()


def test_kaggle_execute_status_gate_blocks_without_ready_preflight(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    make_synthetic_kaggle_zip(zip_path)

    cases = [
        (
            "start_kaggle_full_pipeline.py",
            ["--dry-run-json", str(tmp_path / "blocked_full_dry_run.json")],
            "ready_for_full_execute is false",
            tmp_path / "blocked_full_status.json",
            "ready_for_full_execute",
        ),
        (
            "start_kaggle_smoke_pipeline.py",
            ["--smoke-limit", "1", "--dry-run-json", str(tmp_path / "blocked_smoke_dry_run.json")],
            "ready_for_smoke_execute is false",
            tmp_path / "blocked_smoke_status.json",
            "ready_for_smoke_execute",
        ),
    ]
    for launcher, extra_args, expected_error, status_path, ready_key in cases:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / launcher),
                "--repo",
                str(ROOT),
                "--dataset-root",
                str(dataset_root),
                "--execute",
                "--no-preflight",
                "--allow-execute-without-preflight",
                "--no-suite",
                "--no-validation",
                "--preflight-json",
                str(tmp_path / f"{launcher}.missing_preflight.json"),
                "--log-path",
                str(tmp_path / f"{launcher}.log"),
                "--status-json",
                str(status_path),
                *extra_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        combined_output = result.stdout + result.stderr
        assert expected_error in combined_output
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["recommendations"][ready_key] is False

    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()


def test_kaggle_smoke_launcher_execute_status_gate_accepts_matching_preflight(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    preflight_path = tmp_path / "smoke_preflight.json"
    dry_run_path = tmp_path / "smoke_dry_run.json"
    status_path = tmp_path / "smoke_status.json"
    log_path = tmp_path / "smoke_execute.log"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_smoke_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--smoke-limit",
            "1",
            "--execute",
            "--no-suite",
            "--no-validation",
            "--preflight-json",
            str(preflight_path),
            "--dry-run-json",
            str(dry_run_path),
            "--status-json",
            str(status_path),
            "--log-path",
            str(log_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert preflight["preflight_options"]["smoke_limit"] == 1
    assert preflight["storage_estimate"]["selected_train_csv_files"] == 2
    assert preflight["streaming_dry_run"]["selected_train_csv_files"] == 2
    assert status["smoke_dry_run"]["matches_preflight"] is True
    assert status["recommendations"]["ready_for_smoke_execute"] is True
    assert (kaggle_dir / "processed_smoke" / "_SUCCESS.json").exists()
    assert list((kaggle_dir / "processed_smoke" / "records").glob("*.npz"))
    assert not (kaggle_dir / "processed").exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "--smoke-limit 1 --skip-pytest" in log_text
    assert "ready_for_smoke_execute: True" in log_text


def test_kaggle_full_launcher_execute_status_gate_accepts_matching_preflight(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    preflight_path = tmp_path / "full_preflight.json"
    dry_run_path = tmp_path / "full_dry_run.json"
    status_path = tmp_path / "full_status.json"
    log_path = tmp_path / "full_execute.log"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_full_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--execute",
            "--no-suite",
            "--no-validation",
            "--preflight-json",
            str(preflight_path),
            "--dry-run-json",
            str(dry_run_path),
            "--status-json",
            str(status_path),
            "--log-path",
            str(log_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert preflight["preflight_options"]["smoke_limit"] == 0
    assert preflight["storage_estimate"]["selected_train_csv_files"] == 5
    assert preflight["streaming_dry_run"]["selected_train_csv_files"] == 5
    assert status["full_dry_run"]["matches_preflight"] is True
    assert status["recommendations"]["ready_for_full_execute"] is True
    assert (kaggle_dir / "processed" / "_SUCCESS.json").exists()
    assert list((kaggle_dir / "processed" / "records").glob("*.npz"))
    assert not (kaggle_dir / "processed_smoke").exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "--suite-config" in log_text
    assert "--skip-pytest" in log_text
    assert "ready_for_full_execute: True" in log_text


def test_kaggle_status_summarizes_reports_without_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    dry_run_path = tmp_path / "status_dry_run.json"
    full_dry_run_path = tmp_path / "status_full_dry_run.json"
    status_path = tmp_path / "kaggle_status.json"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_smoke_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--smoke-limit",
            "1",
            "--no-preflight",
            "--no-suite",
            "--dry-run-json",
            str(dry_run_path),
            "--log-path",
            str(tmp_path / "status_launcher.log"),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_full_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--no-preflight",
            "--no-suite",
            "--dry-run-json",
            str(full_dry_run_path),
            "--log-path",
            str(tmp_path / "status_full_launcher.log"),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(tmp_path / "missing_preflight.json"),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "processed_smoke_exists: False" in result.stdout
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["zip"]["exists"] is True
    assert status["preflight"]["exists"] is False
    assert status["smoke_dry_run"]["selected_train_csv_files"] == 2
    assert status["smoke_dry_run"]["zip_matches_current"] is True
    assert status["smoke_dry_run"]["metadata_issue_count"] == 0
    assert status["full_dry_run"]["selected_train_csv_files"] == 5
    assert status["full_dry_run"]["zip_matches_current"] is True
    assert status["full_dry_run"]["metadata_issue_count"] == 0
    assert status["processed"]["exists"] is False
    assert status["processed_smoke"]["exists"] is False
    assert status["recommendations"]["status_only"] is True
    assert status["recommendations"]["ready_for_full_execute"] is False
    assert status["recommendations"]["ready_for_smoke_suite"] is False

    stale_report = json.loads(full_dry_run_path.read_text(encoding="utf-8"))
    stale_report["zip_size"] = stale_report["zip_size"] + 1
    full_dry_run_path.write_text(json.dumps(stale_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(tmp_path / "missing_preflight.json"),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    stale_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert stale_status["full_dry_run"]["zip_matches_current"] is False
    assert stale_status["recommendations"]["ready_for_full_execute"] is False


def test_kaggle_status_blocks_profile_data_issues(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    dry_run_path = tmp_path / "profile_dry_run.json"
    full_dry_run_path = tmp_path / "missing_full_dry_run.json"
    preflight_path = tmp_path / "smoke_preflight.json"
    status_path = tmp_path / "profile_status.json"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_smoke_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--smoke-limit",
            "1",
            "--no-preflight",
            "--no-suite",
            "--profile-data",
            "--dry-run-json",
            str(dry_run_path),
            "--log-path",
            str(tmp_path / "profile_status_launcher.log"),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    dry_run_report = json.loads(dry_run_path.read_text(encoding="utf-8"))
    dry_run_report["profile"]["overall"]["x_kept_nonfinite_values"] = 1
    dry_run_path.write_text(json.dumps(dry_run_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    zip_stat = zip_path.stat()
    preflight_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "storage_estimate": {
                    "status": "ok",
                    "zip_size": zip_stat.st_size,
                    "zip_modified_time_ns": zip_stat.st_mtime_ns,
                },
                "processed_output_guard": {"no_processed_output_created": True},
                "suite_dry_run": {"config": str(ROOT / "configs" / "kaggle_smoke_suite.json")},
                "streaming_dry_run": {"selected_train_csv_files": 2},
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(status_path),
            "--require-ready",
            "smoke",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "ready_for_smoke_execute is false" in (result.stdout + result.stderr)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["smoke_dry_run"]["profile_data"] is True
    assert status["smoke_dry_run"]["profile_clean_for_execute"] is False
    assert status["smoke_dry_run"]["profile"]["x_kept_nonfinite_values"] == 1
    assert status["recommendations"]["ready_for_smoke_execute"] is False
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()

    dry_run_report["profile"]["overall"]["x_kept_nonfinite_values"] = 0
    dry_run_report["profile"]["overall"]["kept_label_nonbinary_rows"] = 1
    dry_run_path.write_text(json.dumps(dry_run_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    nonbinary_status_path = tmp_path / "profile_nonbinary_status.json"
    nonbinary_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(nonbinary_status_path),
            "--require-ready",
            "smoke",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert nonbinary_result.returncode != 0
    nonbinary_status = json.loads(nonbinary_status_path.read_text(encoding="utf-8"))
    assert nonbinary_status["smoke_dry_run"]["profile_clean_for_execute"] is False
    assert nonbinary_status["smoke_dry_run"]["profile"]["kept_label_nonbinary_rows"] == 1
    assert nonbinary_status["recommendations"]["ready_for_smoke_execute"] is False


def test_kaggle_status_requires_matching_preflight_suite(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    kaggle_dir = dataset_root / "2.Kaggle Parkinson's Freezing of Gait Prediction"
    kaggle_dir.mkdir(parents=True)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    smoke_dry_run_path = tmp_path / "smoke_dry_run.json"
    full_dry_run_path = tmp_path / "full_dry_run.json"
    preflight_path = tmp_path / "preflight.json"
    status_path = tmp_path / "status.json"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_smoke_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--smoke-limit",
            "1",
            "--no-preflight",
            "--no-suite",
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--log-path",
            str(tmp_path / "smoke.log"),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "start_kaggle_full_pipeline.py"),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--no-preflight",
            "--no-suite",
            "--dry-run-json",
            str(full_dry_run_path),
            "--log-path",
            str(tmp_path / "full.log"),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    zip_stat = zip_path.stat()
    preflight_report = {
        "status": "passed",
        "storage_estimate": {
            "status": "ok",
            "zip_size": zip_stat.st_size,
            "zip_modified_time_ns": zip_stat.st_mtime_ns,
        },
        "processed_output_guard": {"no_processed_output_created": True},
        "suite_dry_run": {"config": str(ROOT / "configs" / "kaggle_full_suite.json")},
        "streaming_dry_run": {"selected_train_csv_files": 5},
    }
    preflight_path.write_text(json.dumps(preflight_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    full_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert full_status["preflight"]["suite_matches_full"] is True
    assert full_status["preflight"]["suite_matches_smoke"] is False
    assert full_status["preflight"]["suite_preflight"]["exists"] is False
    assert full_status["recommendations"]["ready_for_full_execute"] is True
    assert full_status["recommendations"]["ready_for_smoke_execute"] is False

    suite_preflight_blocking = dict(preflight_report)
    suite_preflight_blocking["suite_preflight"] = {
        "ok": False,
        "warnings": [],
        "errors": [{"message": "Synthetic suite preflight error"}],
    }
    preflight_path.write_text(json.dumps(suite_preflight_blocking, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    blocked_suite_preflight_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert blocked_suite_preflight_status["preflight"]["suite_preflight"]["exists"] is True
    assert blocked_suite_preflight_status["preflight"]["suite_preflight"]["ok"] is False
    assert blocked_suite_preflight_status["preflight"]["suite_preflight"]["error_count"] == 1
    assert blocked_suite_preflight_status["recommendations"]["ready_for_full_execute"] is False

    zip_structure_blocking = dict(preflight_report)
    zip_structure_blocking["zip_structure"] = {
        "ok": False,
        "errors": ["Synthetic zip structure error"],
        "warnings": [],
    }
    preflight_path.write_text(json.dumps(zip_structure_blocking, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    zip_structure_status_path = tmp_path / "zip_structure_status.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(zip_structure_status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    blocked_zip_structure_status = json.loads(zip_structure_status_path.read_text(encoding="utf-8"))
    assert blocked_zip_structure_status["preflight"]["zip_structure"]["exists"] is True
    assert blocked_zip_structure_status["preflight"]["zip_structure"]["ok"] is False
    assert blocked_zip_structure_status["preflight"]["zip_structure"]["error_count"] == 1
    assert blocked_zip_structure_status["recommendations"]["ready_for_full_execute"] is False
    preflight_path.write_text(json.dumps(preflight_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    processed_dir = kaggle_dir / "processed"
    processed_dir.mkdir()
    records_dir = processed_dir / "records"
    records_dir.mkdir()
    (records_dir / "orphan.npz").write_bytes(b"placeholder")
    partial_status_path = tmp_path / "partial_status.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(partial_status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    partial_status = json.loads(partial_status_path.read_text(encoding="utf-8"))
    assert partial_status["processed"]["exists"] is True
    assert partial_status["processed"]["success_exists"] is False
    assert partial_status["processed"]["complete"] is False
    assert partial_status["processed"]["partial"] is True
    assert partial_status["processed"]["records_npz_files"] == 1
    assert partial_status["recommendations"]["ready_for_full_execute"] is False
    assert partial_status["recommendations"]["ready_for_full_suite"] is False

    blocked_existing = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--require-ready",
            "full",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert blocked_existing.returncode != 0
    allowed_existing = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--require-ready",
            "full",
            "--allow-existing-output",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert allowed_existing.returncode == 0
    (records_dir / "orphan.npz").unlink()
    records_dir.rmdir()
    processed_dir.rmdir()

    mismatched_full_preflight = dict(preflight_report)
    mismatched_full_preflight["streaming_dry_run"] = {"selected_train_csv_files": 2}
    preflight_path.write_text(json.dumps(mismatched_full_preflight, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    mismatched_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert mismatched_status["full_dry_run"]["matches_preflight"] is False
    assert mismatched_status["recommendations"]["ready_for_full_execute"] is False

    preflight_report["suite_dry_run"]["config"] = str(ROOT / "configs" / "kaggle_smoke_suite.json")
    preflight_report["streaming_dry_run"] = {"selected_train_csv_files": 2}
    preflight_path.write_text(json.dumps(preflight_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_fog_status.py"),
            "--repo-root",
            str(ROOT),
            "--dataset-root",
            str(dataset_root),
            "--dry-run-json",
            str(smoke_dry_run_path),
            "--full-dry-run-json",
            str(full_dry_run_path),
            "--preflight-json",
            str(preflight_path),
            "--output-json",
            str(status_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    smoke_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert smoke_status["preflight"]["suite_matches_full"] is False
    assert smoke_status["preflight"]["suite_matches_smoke"] is True
    assert smoke_status["recommendations"]["ready_for_full_execute"] is False
    assert smoke_status["recommendations"]["ready_for_smoke_execute"] is True
    assert not (kaggle_dir / "processed").exists()
    assert not (kaggle_dir / "processed_smoke").exists()


def test_kaggle_interrupted_checkpoint_can_resume(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    output_dir = tmp_path / "processed_interrupted"
    make_synthetic_kaggle_zip(zip_path)

    interrupted = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--stop-after-source-files",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert interrupted.returncode != 0
    assert "Stopped after 2 source files" in interrupted.stderr
    assert not (output_dir / "_SUCCESS.json").exists()
    partial_manifest = pd.read_csv(output_dir / "manifest.csv")
    partial_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    partial_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert len(partial_manifest) == 3
    assert len(partial_source_summary) == 2
    assert partial_config["metadata_checkpointing"] == "per_source"
    assert partial_config["summary"]["source_file_count"] == 2

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--resume",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    resumed_manifest = pd.read_csv(output_dir / "manifest.csv")
    resumed_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    assert len(resumed_manifest) == 5
    assert resumed_manifest["record_id"].is_unique
    assert len(resumed_source_summary) == 5
    assert resumed_source_summary["source_file"].is_unique
    assert (output_dir / "_SUCCESS.json").exists()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_processed_records.py"),
            str(output_dir),
            "--expected-channels",
            "3",
            "--require-success",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_kaggle_zero_record_checkpoint_writes_csv_headers(tmp_path: Path) -> None:
    zip_path = tmp_path / "zero_record_checkpoint_kaggle.zip"
    output_dir = tmp_path / "processed_zero_checkpoint"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_zip_csv(
            archive,
            "tdcsfog_metadata.csv",
            [{"Id": "td_dummy", "Subject": "subjT", "Visit": 1, "Test": 1, "Medication": "on"}],
        )
        write_zip_csv(
            archive,
            "defog_metadata.csv",
            [{"Id": "df_zero", "Subject": "subjZ", "Visit": 1, "Medication": "off"}],
        )
        write_zip_csv(
            archive,
            "subjects.csv",
            [
                {
                    "Subject": "subjT",
                    "Visit": 1,
                    "Age": 60,
                    "Sex": "M",
                    "YearsSinceDx": 5,
                    "UPDRSIII_On": 18,
                    "UPDRSIII_Off": 24,
                    "NFOGQ": 7,
                },
                {
                    "Subject": "subjZ",
                    "Visit": 1,
                    "Age": 72,
                    "Sex": "F",
                    "YearsSinceDx": 11,
                    "UPDRSIII_On": 28,
                    "UPDRSIII_Off": 40,
                    "NFOGQ": 14,
                },
            ],
        )
        for name in ("daily_metadata.csv", "events.csv", "tasks.csv", "sample_submission.csv"):
            write_zip_csv(archive, name, [{"Id": "dummy"}])
        write_zip_csv(
            archive,
            "train/defog/df_zero.csv",
            [
                {
                    "Time": 0,
                    "AccV": 1.0,
                    "AccML": 0.1,
                    "AccAP": -0.1,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 0,
                    "Valid": "false",
                    "Task": "true",
                },
                {
                    "Time": 1,
                    "AccV": 2.0,
                    "AccML": 0.2,
                    "AccAP": -0.2,
                    "StartHesitation": 0,
                    "Turn": 0,
                    "Walking": 1,
                    "Valid": "true",
                    "Task": "false",
                },
            ],
        )

    interrupted = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "defog",
            "--valid-only",
            "--task-only",
            "--stop-after-source-files",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert interrupted.returncode != 0
    assert not (output_dir / "_SUCCESS.json").exists()
    partial_manifest = pd.read_csv(output_dir / "manifest.csv")
    partial_loso = pd.read_csv(output_dir / "loso_folds.csv")
    partial_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    assert partial_manifest.empty
    assert {"record_id", "record_path", "subject_id", "n_samples"}.issubset(partial_manifest.columns)
    assert partial_loso.empty
    assert {"fold_id", "test_subject_id", "split", "record_id"}.issubset(partial_loso.columns)
    assert len(partial_source_summary) == 1
    assert int(partial_source_summary.iloc[0]["n_records"]) == 0
    assert int(partial_source_summary.iloc[0]["n_kept_rows"]) == 0

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "defog",
            "--valid-only",
            "--task-only",
            "--resume",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    resumed_manifest = pd.read_csv(output_dir / "manifest.csv")
    resumed_loso = pd.read_csv(output_dir / "loso_folds.csv")
    resumed_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    success = json.loads((output_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert resumed_manifest.empty
    assert resumed_loso.empty
    assert len(resumed_source_summary) == 1
    assert success["record_count"] == 0
    assert success["source_file_count"] == 1


def test_kaggle_resume_completes_remaining_sources(tmp_path: Path) -> None:
    zip_path = tmp_path / "synthetic_kaggle.zip"
    output_dir = tmp_path / "processed_resume"
    make_synthetic_kaggle_zip(zip_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--smoke-limit",
            "1",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    first_manifest = pd.read_csv(output_dir / "manifest.csv")
    assert len(first_manifest) == 3
    first_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    assert len(first_source_summary) == 2
    assert (output_dir / "_SUCCESS.json").exists()
    first_success = json.loads((output_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert first_success["min_samples"] == 1

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--resume",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    resumed_manifest = pd.read_csv(output_dir / "manifest.csv")
    assert len(resumed_manifest) == 5
    assert resumed_manifest["record_id"].is_unique
    resumed_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    assert len(resumed_source_summary) == 5
    assert resumed_source_summary["source_file"].is_unique
    assert resumed_source_summary["n_records"].sum() == 5
    assert not list(output_dir.rglob("*.tmp*"))

    success = json.loads((output_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["record_count"] == 5
    assert success["source_file_count"] == 5
    assert success["total_samples"] == 10
    assert success["min_samples"] == 1
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--resume",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    second_resumed_manifest = pd.read_csv(output_dir / "manifest.csv")
    second_resumed_source_summary = pd.read_csv(output_dir / "source_summary.csv")
    assert len(second_resumed_manifest) == 5
    assert len(second_resumed_source_summary) == 5

    incompatible_resume = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--record-compression",
            "none",
            "--resume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert incompatible_resume.returncode != 0
    assert "record_compression" in incompatible_resume.stderr
    assert (output_dir / "_SUCCESS.json").exists()

    incompatible_min_samples = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--zip-path",
            str(zip_path),
            "--output-dir",
            str(output_dir),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--min-samples",
            "2",
            "--resume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert incompatible_min_samples.returncode != 0
    assert "min_samples" in incompatible_min_samples.stderr

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_processed_records.py"),
            str(output_dir),
            "--expected-channels",
            "3",
            "--require-success",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
