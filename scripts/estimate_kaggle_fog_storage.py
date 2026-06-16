#!/usr/bin/env python
"""Estimate Kaggle FOG supervised storage needs without extracting records."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PREFIX = "2.Kaggle"
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Estimate supervised Kaggle FOG output storage from the zip central directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "dataset")
    parser.add_argument(
        "--source",
        choices=("tdcsfog", "defog", "both", "all"),
        default="both",
        help="'both' means train/tdcsfog + train/defog. 'all' also includes train/notype.",
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        action="append",
        default=[],
        help="Experiment suite config used to estimate unique window output budgets.",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help="Select only the first N train CSV files per source for smoke-budget estimates. 0 selects all.",
    )
    parser.add_argument(
        "--processed-multiplier",
        type=float,
        default=1.0,
        help="Budgeted processed bytes as a multiple of selected supervised CSV uncompressed bytes.",
    )
    parser.add_argument(
        "--window-multiplier",
        type=float,
        default=1.25,
        help="Extra multiplier for window arrays/metadata after accounting for overlap.",
    )
    parser.add_argument("--reserve-gib", type=float, default=5.0, help="Additional free-space reserve.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write the estimate report.")
    parser.add_argument("--fail-if-insufficient", action="store_true")
    args = parser.parse_args()
    if args.smoke_limit < 0:
        parser.error("--smoke-limit must be >= 0")
    return args


def find_default_zip(dataset_root: Path) -> Path:
    candidates = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith(DEFAULT_DATASET_PREFIX)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {DEFAULT_DATASET_PREFIX} directory under {dataset_root}, found {candidates}")
    zip_path = candidates[0] / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    return zip_path


def selected_sources(source: str) -> set[str]:
    if source == "both":
        return {"tdcsfog", "defog"}
    if source == "all":
        return {"tdcsfog", "defog", "notype"}
    return {source}


def is_train_csv(info: zipfile.ZipInfo, source: str) -> bool:
    if info.is_dir():
        return False
    parts = info.filename.replace("\\", "/").split("/")
    return len(parts) == 3 and parts[0] == "train" and parts[1] == source and parts[2].lower().endswith(".csv")


def format_gib(size: int | float) -> str:
    return f"{float(size) / GIB:.3f} GiB"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def window_duplication_factor(windowing: dict[str, Any]) -> float:
    window_seconds = float(windowing.get("window_seconds", 1.0))
    stride_seconds = windowing.get("stride_seconds")
    if stride_seconds is not None:
        stride = float(stride_seconds)
        if stride <= 0:
            raise ValueError(f"stride_seconds must be positive, got {stride_seconds!r}")
        return max(1.0, window_seconds / stride)
    overlap = float(windowing.get("overlap", 0.5))
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")
    return max(1.0, 1.0 / (1.0 - overlap))


def suite_window_budgets(
    suite_configs: list[Path],
    repo_root: Path,
    processed_budget: int,
    window_multiplier: float,
) -> list[dict[str, Any]]:
    budgets: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    for suite_config in suite_configs:
        suite_path = resolve_path(suite_config, repo_root)
        suite = load_json(suite_path)
        for entry in suite.get("experiments", []):
            config_value = entry["config"] if isinstance(entry, dict) else entry
            experiment_path = resolve_path(Path(config_value), repo_root)
            experiment = load_json(experiment_path)
            windowing = experiment.get("windowing") or {}
            output_value = windowing.get("output_dir")
            if not output_value:
                continue
            output_dir = resolve_path(Path(output_value), repo_root)
            output_key = str(output_dir)
            if output_key in seen_outputs:
                continue
            seen_outputs.add(output_key)
            duplication = window_duplication_factor(windowing)
            estimated_bytes = int(processed_budget * duplication * window_multiplier)
            budgets.append(
                {
                    "suite_config": str(suite_path),
                    "experiment_config": str(experiment_path),
                    "output_dir": output_key,
                    "label_mode": windowing.get("label_mode", ""),
                    "window_seconds": windowing.get("window_seconds", ""),
                    "stride_seconds": windowing.get("stride_seconds", ""),
                    "target_hz": windowing.get("target_hz", ""),
                    "duplication_factor": duplication,
                    "estimated_bytes": estimated_bytes,
                    "estimated_gib": round(estimated_bytes / GIB, 6),
                }
            )
    return budgets


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    zip_path = (args.zip_path or find_default_zip(args.dataset_root)).resolve()
    wanted = selected_sources(args.source)

    source_counts = {source: 0 for source in sorted(wanted)}
    available_source_counts = {source: 0 for source in sorted(wanted)}
    source_compressed = {source: 0 for source in sorted(wanted)}
    source_uncompressed = {source: 0 for source in sorted(wanted)}
    skipped_uncompressed = 0
    selected_members: list[str] = []
    source_infos: dict[str, list[zipfile.ZipInfo]] = {source: [] for source in sorted(wanted)}

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            matched_source = None
            for source in wanted:
                if is_train_csv(info, source):
                    matched_source = source
                    break
            if matched_source is None:
                if not info.is_dir():
                    skipped_uncompressed += info.file_size
                continue
            source_infos[matched_source].append(info)

    for source in sorted(wanted):
        infos = sorted(source_infos[source], key=lambda item: item.filename)
        available_source_counts[source] = len(infos)
        selected = infos[: args.smoke_limit] if args.smoke_limit > 0 else infos
        skipped = infos[len(selected) :]
        for info in selected:
            selected_members.append(info.filename)
            source_counts[source] += 1
            source_compressed[source] += info.compress_size
            source_uncompressed[source] += info.file_size
        for info in skipped:
            skipped_uncompressed += info.file_size

    selected_compressed = int(sum(source_compressed.values()))
    selected_uncompressed = int(sum(source_uncompressed.values()))
    processed_budget = int(selected_uncompressed * args.processed_multiplier)
    window_budgets = suite_window_budgets(args.suite_config, repo_root, processed_budget, args.window_multiplier)
    window_budget = int(sum(item["estimated_bytes"] for item in window_budgets))
    reserve_bytes = int(args.reserve_gib * GIB)
    required_free = processed_budget + window_budget + reserve_bytes
    usage = shutil.disk_usage(zip_path.parent)
    status = "ok" if usage.free >= required_free else "insufficient_free_space"
    zip_stat = zip_path.stat()

    return {
        "zip_path": str(zip_path),
        "zip_size": zip_stat.st_size,
        "zip_modified_time_ns": int(zip_stat.st_mtime_ns),
        "zip_size_gib": round(zip_stat.st_size / GIB, 6),
        "selected_source": args.source,
        "smoke_limit": args.smoke_limit,
        "available_train_csv_files": int(sum(available_source_counts.values())),
        "available_source_counts": available_source_counts,
        "selected_train_csv_files": len(selected_members),
        "source_counts": source_counts,
        "source_compressed_bytes": source_compressed,
        "source_uncompressed_bytes": source_uncompressed,
        "selected_compressed_bytes": selected_compressed,
        "selected_uncompressed_bytes": selected_uncompressed,
        "skipped_uncompressed_bytes": int(skipped_uncompressed),
        "processed_multiplier": args.processed_multiplier,
        "estimated_processed_budget_bytes": processed_budget,
        "estimated_processed_budget_gib": round(processed_budget / GIB, 6),
        "window_multiplier": args.window_multiplier,
        "window_budgets": window_budgets,
        "estimated_window_budget_bytes": window_budget,
        "estimated_window_budget_gib": round(window_budget / GIB, 6),
        "reserve_gib": args.reserve_gib,
        "required_free_bytes": required_free,
        "required_free_gib": round(required_free / GIB, 6),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / GIB, 6),
        "status": status,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"zip_path: {report['zip_path']}")
    print(f"zip_size: {format_gib(report['zip_size'])}")
    print(f"selected_source: {report['selected_source']}")
    print(f"smoke_limit: {report['smoke_limit']}")
    print(f"available_train_csv_files: {report['available_train_csv_files']}")
    print(f"selected_train_csv_files: {report['selected_train_csv_files']}")
    print(f"selected_supervised_csv_uncompressed: {format_gib(report['selected_uncompressed_bytes'])}")
    print(f"selected_supervised_csv_compressed: {format_gib(report['selected_compressed_bytes'])}")
    print(f"skipped_uncompressed: {format_gib(report['skipped_uncompressed_bytes'])}")
    for source, count in report["source_counts"].items():
        size = report["source_uncompressed_bytes"][source]
        print(f"{source}: files={count} uncompressed={format_gib(size)}")
    print(f"estimated_processed_budget: {format_gib(report['estimated_processed_budget_bytes'])}")
    print(f"estimated_window_budget: {format_gib(report['estimated_window_budget_bytes'])}")
    for item in report["window_budgets"]:
        print(
            "window_output: "
            f"{item['output_dir']} duplication={item['duplication_factor']:.3f} "
            f"budget={format_gib(item['estimated_bytes'])}"
        )
    print(f"reserve: {report['reserve_gib']:.3f} GiB")
    print(f"required_free: {format_gib(report['required_free_bytes'])}")
    print(f"free: {format_gib(report['free_bytes'])}")
    print(f"status: {report['status']}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_report(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.fail_if_insufficient and report["status"] != "ok":
        raise SystemExit(f"Insufficient free space: need {format_gib(report['required_free_bytes'])}, free {format_gib(report['free_bytes'])}")


if __name__ == "__main__":
    main()
