#!/usr/bin/env python
"""Read-only status summary for the Kaggle FOG pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GIB = 1024**3
SMOKE_SUITE_CONFIG = "kaggle_smoke_suite.json"
FULL_SUITE_CONFIG = "kaggle_full_suite.json"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize Kaggle FOG zip/preflight/dry-run/processed status without creating records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "dataset")
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--dry-run-json", type=Path, default=None)
    parser.add_argument("--full-dry-run-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--require-ready",
        choices=("smoke", "full"),
        default=None,
        help="Exit non-zero unless the requested execute path is ready.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="With --require-ready, allow the target processed directory to already exist.",
    )
    return parser.parse_args()


def find_kaggle_dir(dataset_root: Path) -> Path:
    matches = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("2.Kaggle")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one 2.Kaggle* directory under {dataset_root}, found {matches}")
    return matches[0]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def success_status(path: Path) -> dict[str, Any] | None:
    marker = read_json(path)
    if marker is None:
        return None
    return {
        "status": marker.get("status", ""),
        "record_count": marker.get("record_count"),
        "source_file_count": marker.get("source_file_count"),
        "total_samples": marker.get("total_samples"),
        "total_fog_samples": marker.get("total_fog_samples"),
    }


def processed_dir_status(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.csv"
    source_summary_path = path / "source_summary.csv"
    success_path = path / "_SUCCESS.json"
    records_dir = path / "records"
    exists = path.exists()
    success = success_status(success_path) if exists else None
    records_npz_files = len(list(records_dir.glob("*.npz"))) if records_dir.exists() else None
    return {
        "path": str(path),
        "exists": exists,
        "success_exists": success_path.exists() if exists else None,
        "complete": bool(success and success.get("status") == "complete"),
        "partial": bool(exists and not (success and success.get("status") == "complete")),
        "success": success,
        "manifest_records": count_csv_rows(manifest_path) if exists else None,
        "source_summary_files": count_csv_rows(source_summary_path) if exists else None,
        "records_npz_files": records_npz_files if exists else None,
    }


def report_matches_zip(report: dict[str, Any] | None, zip_path: Path) -> bool:
    if not report or not zip_path.exists():
        return False
    stat = zip_path.stat()
    if report.get("zip_size") != stat.st_size:
        return False
    modified = report.get("zip_modified_time_ns")
    if modified is None:
        return False
    return int(modified) == int(stat.st_mtime_ns)


def preflight_matches_zip(report: dict[str, Any] | None, zip_path: Path) -> bool:
    if not report or not zip_path.exists():
        return False
    storage = report.get("storage_estimate") or {}
    return report_matches_zip(storage, zip_path)


def preflight_suite_config_name(report: dict[str, Any] | None) -> str | None:
    if not report:
        return None
    suite_config = report.get("suite_dry_run", {}).get("config")
    if not suite_config:
        return None
    return Path(str(suite_config)).name


def preflight_suite_matches(report: dict[str, Any] | None, expected_name: str) -> bool:
    return preflight_suite_config_name(report) == expected_name


def suite_preflight_status(report: dict[str, Any] | None) -> dict[str, Any]:
    suite_preflight = (report or {}).get("suite_preflight") if report else None
    warnings = suite_preflight.get("warnings", []) if isinstance(suite_preflight, dict) else []
    errors = suite_preflight.get("errors", []) if isinstance(suite_preflight, dict) else []
    return {
        "exists": isinstance(suite_preflight, dict),
        "ok": suite_preflight.get("ok") if isinstance(suite_preflight, dict) else None,
        "warning_count": len(warnings),
        "error_count": len(errors),
        "warnings": warnings,
        "errors": errors,
    }


def suite_preflight_allows_execute(report: dict[str, Any] | None) -> bool:
    status = suite_preflight_status(report)
    return not status["exists"] or status["ok"] is True


def zip_structure_status(report: dict[str, Any] | None) -> dict[str, Any]:
    structure = (report or {}).get("zip_structure") if report else None
    errors = structure.get("errors", []) if isinstance(structure, dict) else []
    warnings = structure.get("warnings", []) if isinstance(structure, dict) else []
    return {
        "exists": isinstance(structure, dict),
        "ok": structure.get("ok") if isinstance(structure, dict) else None,
        "selected_supervised_train_csv_files": (
            structure.get("selected_supervised_train_csv_files") if isinstance(structure, dict) else None
        ),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def zip_structure_allows_execute(report: dict[str, Any] | None) -> bool:
    status = zip_structure_status(report)
    return not status["exists"] or status["ok"] is True


def dry_run_status(path: Path, report: dict[str, Any] | None, zip_path: Path) -> dict[str, Any]:
    profile = dry_run_profile_status(report)
    return {
        "path": str(path),
        "exists": report is not None,
        "zip_matches_current": report_matches_zip(report, zip_path),
        "selected_train_csv_files": report.get("selected_train_csv_files") if report else None,
        "members_missing_metadata": report.get("members_missing_metadata") if report else None,
        "members_with_header_issues": report.get("members_with_header_issues") if report else None,
        "metadata_issue_count": report.get("metadata_issue_count") if report else None,
        "header_issue_count": report.get("header_issue_count") if report else None,
        "by_source": report.get("by_source") if report else None,
        "profile_data": profile["profile_data"],
        "profile_clean_for_execute": profile["clean_for_execute"],
        "profile": profile["summary"],
    }


def dry_run_matches_preflight(report: dict[str, Any] | None, preflight: dict[str, Any] | None) -> bool:
    if not report or not preflight:
        return False
    streaming = preflight.get("streaming_dry_run") or {}
    if streaming.get("selected_train_csv_files") != report.get("selected_train_csv_files"):
        return False
    optional_keys = (
        "selected_source",
        "valid_only",
        "task_only",
        "strict_metadata",
        "check_headers",
        "smoke_limit",
    )
    for key in optional_keys:
        if key in streaming and streaming.get(key) != report.get(key):
            return False
    return True


def dry_run_profile_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report or not report.get("profile_data"):
        return {"profile_data": False, "clean_for_execute": None, "summary": None}

    profile = report.get("profile") or {}
    overall = profile.get("overall") or {}
    summary = {
        "files_profiled": overall.get("files_profiled"),
        "rows": overall.get("rows"),
        "kept_rows": overall.get("kept_rows"),
        "normal_samples": overall.get("normal_samples"),
        "fog_samples": overall.get("fog_samples"),
        "profiled_duration_sec": overall.get("profiled_duration_sec"),
        "kept_duration_sec": overall.get("kept_duration_sec"),
        "normal_duration_sec": overall.get("normal_duration_sec"),
        "fog_duration_sec": overall.get("fog_duration_sec"),
        "x_nan_values": overall.get("x_nan_values"),
        "x_nonfinite_values": overall.get("x_nonfinite_values"),
        "x_kept_nan_values": overall.get("x_kept_nan_values"),
        "x_kept_nonfinite_values": overall.get("x_kept_nonfinite_values"),
        "label_invalid_rows": overall.get("label_invalid_rows"),
        "kept_label_invalid_rows": overall.get("kept_label_invalid_rows"),
        "label_nonbinary_values": overall.get("label_nonbinary_values"),
        "label_nonbinary_rows": overall.get("label_nonbinary_rows"),
        "kept_label_nonbinary_rows": overall.get("kept_label_nonbinary_rows"),
        "members_skipped_header_issues": profile.get("members_skipped_header_issues"),
    }
    blocking_counts = [
        int(summary.get("x_kept_nonfinite_values") or 0),
        int(summary.get("kept_label_invalid_rows") or 0),
        int(summary.get("kept_label_nonbinary_rows") or 0),
        int(summary.get("members_skipped_header_issues") or 0),
    ]
    return {
        "profile_data": True,
        "clean_for_execute": bool(profile and overall and all(value == 0 for value in blocking_counts)),
        "summary": summary,
    }


def dry_run_profile_allows_execute(report: dict[str, Any] | None) -> bool:
    profile = dry_run_profile_status(report)
    return profile["clean_for_execute"] is not False


def dry_run_has_no_issues(report: dict[str, Any] | None) -> bool:
    return bool(
        report
        and report.get("metadata_issue_count", 0) == 0
        and report.get("header_issue_count", 0) == 0
        and report.get("members_missing_metadata", 0) == 0
        and report.get("members_with_header_issues", 0) == 0
        and dry_run_profile_allows_execute(report)
    )


def preflight_is_safe(report: dict[str, Any] | None, zip_path: Path) -> bool:
    return bool(
        report
        and preflight_matches_zip(report, zip_path)
        and report.get("status") == "passed"
        and report.get("storage_estimate", {}).get("status") == "ok"
        and report.get("processed_output_guard", {}).get("no_processed_output_created") is True
        and zip_structure_allows_execute(report)
        and suite_preflight_allows_execute(report)
    )


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve()
    kaggle_dir = find_kaggle_dir(dataset_root)
    zip_path = kaggle_dir / "tlvmc-parkinsons-freezing-gait-prediction.zip"
    preflight_path = (args.preflight_json or (repo_root / "outputs" / "kaggle_preflight_report.json")).resolve()
    dry_run_path = (args.dry_run_json or (repo_root / "outputs" / "kaggle_smoke_streaming_dry_run.json")).resolve()
    full_dry_run_path = (
        args.full_dry_run_json or (repo_root / "outputs" / "kaggle_full_streaming_dry_run.json")
    ).resolve()

    preflight = read_json(preflight_path)
    dry_run = read_json(dry_run_path)
    full_dry_run = read_json(full_dry_run_path)
    preflight_suite_status = suite_preflight_status(preflight)
    preflight_zip_structure_status = zip_structure_status(preflight)
    processed = processed_dir_status(kaggle_dir / "processed")
    processed_smoke = processed_dir_status(kaggle_dir / "processed_smoke")
    smoke_dry_run_matches_preflight = dry_run_matches_preflight(dry_run, preflight)
    full_dry_run_matches_preflight = dry_run_matches_preflight(full_dry_run, preflight)

    ready_for_smoke_preprocess_inputs = bool(
        zip_path.exists()
        and preflight_is_safe(preflight, zip_path)
        and preflight_suite_matches(preflight, SMOKE_SUITE_CONFIG)
        and smoke_dry_run_matches_preflight
        and dry_run_has_no_issues(dry_run)
        and report_matches_zip(dry_run, zip_path)
    )
    ready_for_smoke_execute = bool(ready_for_smoke_preprocess_inputs and not processed_smoke["exists"])
    ready_for_smoke_suite = bool(processed_smoke["success"] and processed_smoke["success"].get("status") == "complete")
    ready_for_full_preprocess_inputs = bool(
        zip_path.exists()
        and preflight_is_safe(preflight, zip_path)
        and preflight_suite_matches(preflight, FULL_SUITE_CONFIG)
        and full_dry_run_matches_preflight
        and dry_run_has_no_issues(full_dry_run)
        and report_matches_zip(full_dry_run, zip_path)
    )
    ready_for_full_execute = bool(ready_for_full_preprocess_inputs and not processed["exists"])
    ready_for_full_suite = bool(processed["success"] and processed["success"].get("status") == "complete")

    return {
        "repo_root": str(repo_root),
        "dataset_root": str(dataset_root),
        "kaggle_dir": str(kaggle_dir),
        "zip": {
            "path": str(zip_path),
            "exists": zip_path.exists(),
            "size_bytes": zip_path.stat().st_size if zip_path.exists() else None,
            "size_gib": round(zip_path.stat().st_size / GIB, 6) if zip_path.exists() else None,
        },
        "preflight": {
            "path": str(preflight_path),
            "exists": preflight is not None,
            "status": preflight.get("status") if preflight else None,
            "zip_matches_current": preflight_matches_zip(preflight, zip_path),
            "suite_config": preflight.get("suite_dry_run", {}).get("config") if preflight else None,
            "suite_config_name": preflight_suite_config_name(preflight),
            "suite_matches_smoke": preflight_suite_matches(preflight, SMOKE_SUITE_CONFIG),
            "suite_matches_full": preflight_suite_matches(preflight, FULL_SUITE_CONFIG),
            "selected_train_csv_files": (
                preflight.get("streaming_dry_run", {}).get("selected_train_csv_files") if preflight else None
            ),
            "members_missing_metadata": (
                preflight.get("streaming_dry_run", {}).get("members_missing_metadata") if preflight else None
            ),
            "members_with_header_issues": (
                preflight.get("streaming_dry_run", {}).get("members_with_header_issues") if preflight else None
            ),
            "storage_status": preflight.get("storage_estimate", {}).get("status") if preflight else None,
            "no_processed_output_created": (
                preflight.get("processed_output_guard", {}).get("no_processed_output_created") if preflight else None
            ),
            "zip_structure": preflight_zip_structure_status,
            "suite_preflight": preflight_suite_status,
        },
        "smoke_dry_run": {
            **dry_run_status(dry_run_path, dry_run, zip_path),
            "matches_preflight": smoke_dry_run_matches_preflight,
        },
        "full_dry_run": {
            **dry_run_status(full_dry_run_path, full_dry_run, zip_path),
            "matches_preflight": full_dry_run_matches_preflight,
        },
        "processed": processed,
        "processed_smoke": processed_smoke,
        "recommendations": {
            "ready_for_smoke_preprocess_inputs": ready_for_smoke_preprocess_inputs,
            "ready_for_smoke_execute": ready_for_smoke_execute,
            "ready_for_smoke_suite": ready_for_smoke_suite,
            "ready_for_full_preprocess_inputs": ready_for_full_preprocess_inputs,
            "ready_for_full_execute": ready_for_full_execute,
            "ready_for_full_suite": ready_for_full_suite,
            "smoke_execute_command": "python scripts/start_kaggle_smoke_pipeline.py --execute --overwrite",
            "full_execute_command": "python scripts/start_kaggle_full_pipeline.py --execute --overwrite",
            "status_only": True,
        },
    }


def print_status(status: dict[str, Any]) -> None:
    print(f"kaggle_dir: {status['kaggle_dir']}")
    print(f"zip_exists: {status['zip']['exists']}")
    if status["zip"]["exists"]:
        print(f"zip_size_gib: {status['zip']['size_gib']}")
    print(f"preflight_status: {status['preflight']['status']}")
    print(f"preflight_zip_matches_current: {status['preflight']['zip_matches_current']}")
    print(f"preflight_suite_config: {status['preflight']['suite_config_name']}")
    print(f"preflight_suite_matches_smoke: {status['preflight']['suite_matches_smoke']}")
    print(f"preflight_suite_matches_full: {status['preflight']['suite_matches_full']}")
    print(f"preflight_suite_preflight_exists: {status['preflight']['suite_preflight']['exists']}")
    print(f"preflight_suite_preflight_ok: {status['preflight']['suite_preflight']['ok']}")
    print(f"preflight_suite_preflight_warnings: {status['preflight']['suite_preflight']['warning_count']}")
    print(f"preflight_suite_preflight_errors: {status['preflight']['suite_preflight']['error_count']}")
    print(f"preflight_zip_structure_exists: {status['preflight']['zip_structure']['exists']}")
    print(f"preflight_zip_structure_ok: {status['preflight']['zip_structure']['ok']}")
    print(f"preflight_zip_structure_errors: {status['preflight']['zip_structure']['error_count']}")
    print(f"preflight_selected_train_csv_files: {status['preflight']['selected_train_csv_files']}")
    print(f"smoke_dry_run_exists: {status['smoke_dry_run']['exists']}")
    print(f"smoke_dry_run_zip_matches_current: {status['smoke_dry_run']['zip_matches_current']}")
    print(f"smoke_dry_run_matches_preflight: {status['smoke_dry_run']['matches_preflight']}")
    print(f"smoke_dry_run_selected_train_csv_files: {status['smoke_dry_run']['selected_train_csv_files']}")
    print(f"smoke_dry_run_profile_clean_for_execute: {status['smoke_dry_run']['profile_clean_for_execute']}")
    print(f"full_dry_run_exists: {status['full_dry_run']['exists']}")
    print(f"full_dry_run_zip_matches_current: {status['full_dry_run']['zip_matches_current']}")
    print(f"full_dry_run_matches_preflight: {status['full_dry_run']['matches_preflight']}")
    print(f"full_dry_run_selected_train_csv_files: {status['full_dry_run']['selected_train_csv_files']}")
    print(f"full_dry_run_profile_clean_for_execute: {status['full_dry_run']['profile_clean_for_execute']}")
    print(f"processed_exists: {status['processed']['exists']}")
    print(f"processed_complete: {status['processed']['complete']}")
    print(f"processed_partial: {status['processed']['partial']}")
    print(f"processed_smoke_exists: {status['processed_smoke']['exists']}")
    print(f"processed_smoke_complete: {status['processed_smoke']['complete']}")
    print(f"processed_smoke_partial: {status['processed_smoke']['partial']}")
    print(f"ready_for_smoke_execute: {status['recommendations']['ready_for_smoke_execute']}")
    print(f"ready_for_smoke_preprocess_inputs: {status['recommendations']['ready_for_smoke_preprocess_inputs']}")
    print(f"ready_for_smoke_suite: {status['recommendations']['ready_for_smoke_suite']}")
    print(f"ready_for_full_execute: {status['recommendations']['ready_for_full_execute']}")
    print(f"ready_for_full_preprocess_inputs: {status['recommendations']['ready_for_full_preprocess_inputs']}")
    print(f"ready_for_full_suite: {status['recommendations']['ready_for_full_suite']}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    status = build_status(args)
    print_status(status)
    if args.output_json:
        write_json(args.output_json.resolve(), status)
        print(f"status_json: {args.output_json.resolve()}")
    if args.require_ready:
        key = (
            f"ready_for_{args.require_ready}_preprocess_inputs"
            if args.allow_existing_output
            else f"ready_for_{args.require_ready}_execute"
        )
        if not status["recommendations"].get(key):
            raise SystemExit(f"{key} is false; refusing to proceed")


if __name__ == "__main__":
    main()
