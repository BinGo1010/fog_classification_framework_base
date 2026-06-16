#!/usr/bin/env python
"""Inspect the Kaggle FOG competition zip without extracting records."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET_PREFIX = "2.Kaggle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read only the zip central directory and write a small inventory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Path to tlvmc-parkinsons-freezing-gait-prediction.zip.",
    )
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
        help="Directory for kaggle_zip_inventory.csv and kaggle_zip_inventory_summary.json.",
    )
    return parser.parse_args()


def find_default_zip(dataset_root: Path) -> Path:
    candidates = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith(DEFAULT_DATASET_PREFIX)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {DEFAULT_DATASET_PREFIX} directory under {dataset_root}, found {candidates}")
    zip_path = candidates[0] / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    return zip_path


def classify_member(name: str) -> str:
    normalized = name.replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    if "tdcsfog" in parts:
        return "tdcsfog"
    if "defog" in parts:
        return "defog"
    if "notype" in parts:
        return "notype"
    filename = parts[-1] if parts else normalized
    if "metadata" in filename or filename in {"subjects.csv", "events.csv", "tasks.csv"}:
        return "metadata"
    return "auxiliary"


def path_bucket(name: str, group: str) -> str:
    normalized = name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    lowered = [part.lower() for part in parts]
    if group == "metadata":
        return "/"
    if len(parts) >= 2 and lowered[0] in {"train", "test"}:
        return f"{lowered[0]}/{lowered[1]}"
    if lowered[:1] == ["unlabeled"]:
        return "unlabeled"
    return "/"


def summarize_members(infos: list[zipfile.ZipInfo]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "file_count": 0,
            "compressed_size": 0,
            "uncompressed_size": 0,
            "csv_count": 0,
            "largest_members": [],
            "sample_paths": [],
        }
    )
    path_buckets: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "file_count": 0,
            "compressed_size": 0,
            "uncompressed_size": 0,
            "csv_count": 0,
            "sample_paths": [],
        }
    )

    for info in infos:
        if info.is_dir():
            continue
        group = classify_member(info.filename)
        bucket = path_bucket(info.filename, group)
        suffix = Path(info.filename).suffix.lower()
        row = {
            "group": group,
            "path_bucket": bucket,
            "path": info.filename,
            "filename": Path(info.filename).name,
            "suffix": suffix,
            "compressed_size": info.compress_size,
            "uncompressed_size": info.file_size,
        }
        rows.append(row)

        summary = groups[group]
        summary["file_count"] = int(summary["file_count"]) + 1
        summary["compressed_size"] = int(summary["compressed_size"]) + info.compress_size
        summary["uncompressed_size"] = int(summary["uncompressed_size"]) + info.file_size
        if suffix == ".csv":
            summary["csv_count"] = int(summary["csv_count"]) + 1
        if len(summary["sample_paths"]) < 12:
            summary["sample_paths"].append(info.filename)
        summary["largest_members"].append(
            {
                "path": info.filename,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
            }
        )

        bucket_summary = path_buckets[bucket]
        bucket_summary["file_count"] = int(bucket_summary["file_count"]) + 1
        bucket_summary["compressed_size"] = int(bucket_summary["compressed_size"]) + info.compress_size
        bucket_summary["uncompressed_size"] = int(bucket_summary["uncompressed_size"]) + info.file_size
        if suffix == ".csv":
            bucket_summary["csv_count"] = int(bucket_summary["csv_count"]) + 1
        if len(bucket_summary["sample_paths"]) < 12:
            bucket_summary["sample_paths"].append(info.filename)

    for summary in groups.values():
        summary["largest_members"] = sorted(
            summary["largest_members"],
            key=lambda item: int(item["uncompressed_size"]),
            reverse=True,
        )[:12]

    total_compressed = sum(int(row["compressed_size"]) for row in rows)
    total_uncompressed = sum(int(row["uncompressed_size"]) for row in rows)
    summary = {
        "total_file_count": len(rows),
        "total_compressed_size": total_compressed,
        "total_uncompressed_size": total_uncompressed,
        "groups": dict(sorted(groups.items())),
        "path_buckets": dict(sorted(path_buckets.items())),
    }
    return rows, summary


def write_inventory(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kaggle_zip_inventory.csv"
    json_path = output_dir / "kaggle_zip_inventory_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "path_bucket",
                "path",
                "filename",
                "suffix",
                "compressed_size",
                "uncompressed_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def format_gib(size: int) -> str:
    return f"{size / 1024**3:.3f} GiB"


def main() -> None:
    args = parse_args()
    zip_path = (args.zip_path or find_default_zip(args.dataset_root)).resolve()
    output_dir = (args.output_dir or (zip_path.parent / "inventory")).resolve()

    with zipfile.ZipFile(zip_path) as archive:
        rows, summary = summarize_members(archive.infolist())

    summary["zip_path"] = str(zip_path)
    summary["zip_size"] = zip_path.stat().st_size
    write_inventory(output_dir, rows, summary)

    print(f"zip_path: {zip_path}")
    print(f"zip_size: {format_gib(summary['zip_size'])}")
    print(f"inventory_dir: {output_dir}")
    print(f"files: {summary['total_file_count']}")
    print(f"uncompressed_total: {format_gib(summary['total_uncompressed_size'])}")
    for group, data in summary["groups"].items():
        print(
            f"{group}: files={data['file_count']} csv={data['csv_count']} "
            f"compressed={format_gib(data['compressed_size'])} "
            f"uncompressed={format_gib(data['uncompressed_size'])}"
        )
    print("path_buckets:")
    for bucket, data in summary["path_buckets"].items():
        print(
            f"  {bucket}: files={data['file_count']} csv={data['csv_count']} "
            f"compressed={format_gib(data['compressed_size'])} "
            f"uncompressed={format_gib(data['uncompressed_size'])}"
        )


if __name__ == "__main__":
    main()
