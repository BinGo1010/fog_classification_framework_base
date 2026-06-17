#!/usr/bin/env python
"""Backfill the shared FOG results overview from existing sweep summary CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fog_results_overview import REPO_ROOT, update_overview_many


DEFAULT_SUMMARY_PATTERNS = [
    "outputs/fogstar_mlp_prefog_sweep_summary.csv",
    "outputs/fogstar_mlp_window_sweep_prefog0p5_summary.csv",
    "outputs/fogstar_mlp_long_short_sweep_summary.csv",
    "outputs/fogstar_dualwindow_raw_long_short_sweep_summary.csv",
    "outputs/fogstar_dualcnn_raw_long_short_sweep_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge existing FOG sweep summary CSVs into outputs/fog_results_overview.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "summary_csv",
        nargs="*",
        type=Path,
        help="Existing sweep summary CSV files. Defaults to known FoG sweep summaries.",
    )
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=Path("outputs/fog_results_overview.csv"),
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also include dry-run or incomplete rows whose status is missing_summary.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def infer_sweep(path: Path) -> str:
    name = path.name.lower()
    if "dualwindow" in name and "long_short" in name:
        return "dualwindow_raw_long_short"
    if "dualcnn" in name and "long_short" in name:
        return "dualcnn_raw_long_short"
    if "long_short" in name:
        return "long_short"
    if "prefog" in name:
        return "prefog"
    if "window" in name:
        return "window"
    return "manual"


def read_rows(path: Path, include_missing: bool) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if include_missing:
        return rows
    return [row for row in rows if row.get("status") != "missing_summary"]


def main() -> None:
    args = parse_args()
    summary_paths = args.summary_csv or [Path(pattern) for pattern in DEFAULT_SUMMARY_PATTERNS]
    total = 0
    overview_path = resolve(args.overview_csv)
    for summary_path in summary_paths:
        summary_path = resolve(summary_path)
        rows = read_rows(summary_path, args.include_missing)
        if not rows:
            continue
        overview_path = update_overview_many(overview_path, rows, sweep=infer_sweep(summary_path))
        total += len(rows)
        print(f"[OVERVIEW] merged {len(rows)} rows from {summary_path}")
    print(f"[OVERVIEW] wrote {overview_path} ({total} rows merged)")


if __name__ == "__main__":
    main()
