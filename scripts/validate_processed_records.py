#!/usr/bin/env python
"""Validate sample-level processed record datasets.

Expected directory layout:

    processed_dir/
      records/*.npz
      manifest.csv
      loso_folds.csv
      config.json or schema.json

Each record NPZ must contain only ``x`` and ``y_binary``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed sample-level FOG record datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("--expected-channels", type=int)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Validate at most this many records; 0 validates all records.",
    )
    parser.add_argument(
        "--allow-nan",
        action="store_true",
        help="Allow NaN values in x. Infinite values are always rejected.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Require and validate _SUCCESS.json completion marker.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def resolve_record_path(processed_dir: Path, record_path_value: object) -> Path:
    if pd.isna(record_path_value):
        raise ValueError("manifest.csv contains an empty record_path value")
    raw_path = Path(str(record_path_value))
    if raw_path.is_absolute():
        raise ValueError(f"manifest.csv record_path must be relative, got {raw_path}")

    processed_root = processed_dir.resolve()
    resolved = (processed_root / raw_path).resolve()
    try:
        resolved.relative_to(processed_root)
    except ValueError as exc:
        raise ValueError(
            f"manifest.csv record_path escapes processed_dir: {record_path_value}"
        ) from exc
    return resolved


def validate_manifest(processed_dir: Path) -> pd.DataFrame:
    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    require_columns(
        manifest,
        {
            "record_id",
            "record_path",
            "subject_id",
            "segment_id",
            "n_samples",
        },
        manifest_path,
    )
    if manifest["record_id"].duplicated().any():
        duplicates = manifest.loc[manifest["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id values in manifest: {duplicates[:10]}")
    if manifest["record_path"].duplicated().any():
        duplicates = manifest.loc[manifest["record_path"].duplicated(), "record_path"].tolist()
        raise ValueError(f"Duplicate record_path values in manifest: {duplicates[:10]}")
    if manifest.empty:
        raise ValueError("manifest.csv has no records")
    for record_path in manifest["record_path"]:
        resolve_record_path(processed_dir, record_path)
    return manifest


def load_schema_or_config(processed_dir: Path) -> dict[str, object]:
    schema_path = processed_dir / "schema.json"
    config_path = processed_dir / "config.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def validate_schema(
    processed_dir: Path,
    expected_channels: int | None,
) -> tuple[int | None, dict[str, object]]:
    schema = load_schema_or_config(processed_dir)
    channels = schema.get("channels", [])
    if expected_channels is not None and channels and len(channels) != expected_channels:
        raise ValueError(
            f"schema/config describes {len(channels)} channels, expected {expected_channels}"
        )
    if expected_channels is None and channels:
        expected_channels = len(channels)
    return expected_channels, schema


def validate_records(
    processed_dir: Path,
    manifest: pd.DataFrame,
    expected_channels: int | None,
    max_records: int,
    allow_nan: bool,
    schema: dict[str, object],
) -> dict[str, int]:
    records = manifest if max_records <= 0 else manifest.head(max_records)
    total_samples = 0
    total_normal = 0
    total_fog = 0
    records_with_fog = 0
    records_with_nan = 0
    nan_cells = 0

    for row in records.itertuples(index=False):
        record_path = resolve_record_path(processed_dir, row.record_path)
        if not record_path.exists():
            raise FileNotFoundError(f"Missing record file: {record_path}")
        with np.load(record_path) as record:
            keys = set(record.files)
            if keys != {"x", "y_binary"}:
                raise ValueError(f"{record_path} keys are {sorted(keys)}, expected x/y_binary")
            x = record["x"]
            y = record["y_binary"]

        expected_x_dtype = schema.get("x_dtype")
        if expected_x_dtype and str(x.dtype) != str(expected_x_dtype):
            raise ValueError(f"{record_path} x dtype is {x.dtype}, expected {expected_x_dtype}")
        expected_y_dtype = schema.get("y_binary_dtype")
        if expected_y_dtype and str(y.dtype) != str(expected_y_dtype):
            raise ValueError(f"{record_path} y_binary dtype is {y.dtype}, expected {expected_y_dtype}")
        if x.ndim != 2:
            raise ValueError(f"{record_path} x must be 2D, got shape {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"{record_path} y_binary must be 1D, got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{record_path} x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        if expected_channels is not None and x.shape[1] != expected_channels:
            raise ValueError(
                f"{record_path} has {x.shape[1]} channels, expected {expected_channels}"
            )
        if np.isinf(x).any():
            raise ValueError(f"{record_path} contains infinite x values")
        record_nan_cells = int(np.isnan(x).sum())
        if record_nan_cells:
            if not allow_nan:
                raise ValueError(f"{record_path} contains NaN x values; pass --allow-nan if expected")
            records_with_nan += 1
            nan_cells += record_nan_cells
        unique_labels = set(np.unique(y).astype(int).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"{record_path} has labels outside {{0,1}}: {sorted(unique_labels)}")

        n_samples = int(y.size)
        n_fog = int(y.sum())
        n_normal = int(n_samples - n_fog)
        if int(row.n_samples) != n_samples:
            raise ValueError(f"{record_path} n_samples mismatch against manifest")
        if hasattr(row, "n_fog_samples") and int(row.n_fog_samples) != n_fog:
            raise ValueError(f"{record_path} n_fog_samples mismatch against manifest")
        if hasattr(row, "n_normal_samples") and int(row.n_normal_samples) != n_normal:
            raise ValueError(f"{record_path} n_normal_samples mismatch against manifest")

        total_samples += n_samples
        total_normal += n_normal
        total_fog += n_fog
        records_with_fog += int(n_fog > 0)

    return {
        "validated_records": int(len(records)),
        "validated_samples": total_samples,
        "validated_normal_samples": total_normal,
        "validated_fog_samples": total_fog,
        "validated_records_with_fog": records_with_fog,
        "validated_records_with_nan": records_with_nan,
        "validated_nan_cells": nan_cells,
    }


def validate_loso(processed_dir: Path, manifest: pd.DataFrame) -> None:
    folds_path = processed_dir / "loso_folds.csv"
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing loso_folds.csv: {folds_path}")
    folds = pd.read_csv(folds_path)
    require_columns(
        folds,
        {"fold_id", "test_subject_id", "split", "record_id", "subject_id", "segment_id"},
        folds_path,
    )
    manifest_records = set(manifest["record_id"].astype(str))
    fold_records = set(folds["record_id"].astype(str))
    unknown_records = sorted(fold_records - manifest_records)
    if unknown_records:
        raise ValueError(f"loso_folds.csv references unknown records: {unknown_records[:10]}")

    bad_splits = sorted(set(folds["split"].astype(str)) - {"train", "test"})
    if bad_splits:
        raise ValueError(f"loso_folds.csv has unsupported split values: {bad_splits}")

    manifest_lookup = {
        str(row.record_id): (str(row.subject_id), int(row.segment_id))
        for row in manifest.itertuples(index=False)
    }
    for row in folds.itertuples(index=False):
        record_id = str(row.record_id)
        if record_id not in manifest_lookup:
            continue
        expected_subject_id, expected_segment_id = manifest_lookup[record_id]
        if str(row.subject_id) != expected_subject_id or int(row.segment_id) != expected_segment_id:
            raise ValueError(
                "loso_folds.csv row does not match manifest for "
                f"{record_id}: subject_id={row.subject_id!r} segment_id={row.segment_id!r}, "
                f"expected subject_id={expected_subject_id!r} segment_id={expected_segment_id!r}"
            )

    subjects = sorted(manifest["subject_id"].astype(str).unique())
    for fold_id, group in folds.groupby("fold_id"):
        test_subjects = set(group.loc[group["split"] == "test", "subject_id"].astype(str))
        train_subjects = set(group.loc[group["split"] == "train", "subject_id"].astype(str))
        declared = set(group["test_subject_id"].astype(str).unique())
        if len(declared) != 1:
            raise ValueError(f"{fold_id} has multiple test_subject_id values: {sorted(declared)}")
        declared_subject = next(iter(declared))
        if test_subjects != {declared_subject}:
            raise ValueError(
                f"{fold_id} test subjects {sorted(test_subjects)} do not match {declared_subject}"
            )
        if test_subjects & train_subjects:
            raise ValueError(f"{fold_id} leaks subjects across train/test")
        fold_record_ids = group["record_id"].astype(str).tolist()
        expected_records = set(manifest_records)
        if set(fold_record_ids) != expected_records or len(fold_record_ids) != len(expected_records):
            raise ValueError(f"{fold_id} does not include every manifest record exactly once")
    missing_folds = sorted(set(subjects) - set(folds["test_subject_id"].astype(str).unique()))
    if missing_folds:
        raise ValueError(f"Missing LOSO folds for subjects: {missing_folds}")


def validate_source_summary(processed_dir: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, object]]:
    summary_path = processed_dir / "source_summary.csv"
    if not summary_path.exists():
        return None, {"source_summary": False}

    source_summary = pd.read_csv(summary_path)
    require_columns(
        source_summary,
        {
            "source_file",
            "dataset_part",
            "subject_id",
            "n_records",
            "n_samples",
            "n_fog_samples",
            "status",
        },
        summary_path,
    )
    if source_summary["source_file"].duplicated().any():
        duplicates = source_summary.loc[source_summary["source_file"].duplicated(), "source_file"].astype(str).tolist()
        raise ValueError(f"Duplicate source_file values in source_summary.csv: {duplicates[:10]}")

    bad_status = source_summary.loc[source_summary["status"].astype(str) != "complete", "source_file"].astype(str).tolist()
    if bad_status:
        raise ValueError(f"source_summary.csv has non-complete sources: {bad_status[:10]}")

    if "source_file" not in manifest.columns:
        return source_summary, {
            "source_summary": True,
            "source_files": int(len(source_summary)),
            "source_summary_checked_against_manifest": False,
        }

    manifest_by_source = manifest.copy()
    manifest_by_source["source_file"] = manifest_by_source["source_file"].astype(str)
    manifest_grouped = manifest_by_source.groupby("source_file", dropna=False).agg(
        n_records=("record_id", "count"),
        n_samples=("n_samples", "sum"),
        n_fog_samples=("n_fog_samples", "sum") if "n_fog_samples" in manifest.columns else ("record_id", "count"),
    )

    summary_sources = set(source_summary["source_file"].astype(str))
    manifest_sources = set(manifest["source_file"].astype(str))
    missing_summary_sources = sorted(manifest_sources - summary_sources)
    if missing_summary_sources:
        raise ValueError(f"source_summary.csv is missing manifest sources: {missing_summary_sources[:10]}")

    zero_record_sources = 0
    for row in source_summary.itertuples(index=False):
        source_file = str(row.source_file)
        if source_file in manifest_grouped.index:
            grouped_row = manifest_grouped.loc[source_file]
            expected_records = int(grouped_row["n_records"])
            expected_samples = int(grouped_row["n_samples"])
            expected_fog = int(grouped_row["n_fog_samples"]) if "n_fog_samples" in manifest.columns else int(row.n_fog_samples)
        else:
            expected_records = 0
            expected_samples = 0
            expected_fog = 0
            zero_record_sources += 1

        if int(row.n_records) != expected_records:
            raise ValueError(f"{summary_path} {source_file} n_records={row.n_records} expected {expected_records}")
        if int(row.n_samples) != expected_samples:
            raise ValueError(f"{summary_path} {source_file} n_samples={row.n_samples} expected {expected_samples}")
        if int(row.n_fog_samples) != expected_fog:
            raise ValueError(f"{summary_path} {source_file} n_fog_samples={row.n_fog_samples} expected {expected_fog}")

    return source_summary, {
        "source_summary": True,
        "source_files": int(len(source_summary)),
        "zero_record_sources": int(zero_record_sources),
        "source_summary_checked_against_manifest": True,
    }


def validate_success_marker(
    processed_dir: Path,
    manifest: pd.DataFrame,
    require_success: bool,
    source_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    success_path = processed_dir / "_SUCCESS.json"
    if not success_path.exists():
        if require_success:
            raise FileNotFoundError(f"Missing _SUCCESS.json: {success_path}")
        return {"success_marker": False}

    marker = json.loads(success_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError(f"{success_path} status is not complete: {marker.get('status')}")

    expected = {
        "record_count": int(len(manifest)),
        "subject_count": int(manifest["subject_id"].astype(str).nunique()),
        "total_samples": int(manifest["n_samples"].astype(int).sum()),
    }
    if "n_fog_samples" in manifest.columns:
        expected["total_fog_samples"] = int(manifest["n_fog_samples"].astype(int).sum())
    if source_summary is not None:
        expected["source_file_count"] = int(len(source_summary))

    for key, value in expected.items():
        if key in marker and int(marker[key]) != value:
            raise ValueError(f"{success_path} {key}={marker[key]} does not match expected {value}")

    return {
        "success_marker": True,
        "success_status": str(marker.get("status")),
    }


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir
    manifest = validate_manifest(processed_dir)
    expected_channels, schema = validate_schema(processed_dir, args.expected_channels)
    allow_nan = args.allow_nan or schema.get("missing_value_policy") == "preserve_nan"
    stats = validate_records(processed_dir, manifest, expected_channels, args.max_records, allow_nan, schema)
    validate_loso(processed_dir, manifest)
    source_summary, source_stats = validate_source_summary(processed_dir, manifest)
    success_stats = validate_success_marker(processed_dir, manifest, args.require_success, source_summary)
    summary = {
        "processed_dir": str(processed_dir),
        "manifest_records": int(len(manifest)),
        "subjects": sorted(manifest["subject_id"].astype(str).unique().tolist()),
        "expected_channels": expected_channels,
        "allow_nan": allow_nan,
        **source_stats,
        **success_stats,
        **stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
