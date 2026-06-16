#!/usr/bin/env python
"""Cross-platform Kaggle FOG preflight that does not create processed records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


GIB = 1024**3


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run safe Kaggle FOG checks without extracting data or creating processed records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "dataset")
    parser.add_argument("--pytest-basetemp", type=Path, default=repo_root / "outputs" / "pytest-kaggle-preflight")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=None,
        help="Suite config to validate during the dry-run. Defaults to configs/kaggle_smoke_suite.json.",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=5.0,
        help="Free-space reserve passed to the storage estimator.",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help="Optional per-source train CSV limit passed to storage and streaming dry-runs. 0 checks all selected train CSV files.",
    )
    parser.add_argument(
        "--allow-insufficient-storage",
        action="store_true",
        help="Report insufficient storage without failing preflight. The default is fail-safe.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to write a structured preflight report.")
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be >= 0")
    if args.smoke_limit < 0:
        parser.error("--smoke-limit must be >= 0")
    return args


def find_kaggle_dir(dataset_root: Path) -> Path:
    matches = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("2.Kaggle")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one 2.Kaggle* directory under {dataset_root}, found {matches}")
    return matches[0]


def run_step(name: str, cmd: list[str], cwd: Path, steps: list[dict[str, Any]], capture: bool = False) -> str:
    print(f"\n== {name} ==", flush=True)
    if capture:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        output = result.stdout
    else:
        result = subprocess.run(cmd, cwd=cwd)
        output = ""

    steps.append(
        {
            "name": name,
            "status": "passed" if result.returncode == 0 else "failed",
            "command": cmd,
            "captured_stdout": bool(capture),
            "returncode": result.returncode,
        }
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout if capture else None,
            stderr=result.stderr if capture else None,
        )
    return output


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / GIB:.3f} GiB"


def directory_file_stats(path: Path) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        count += 1
        total_bytes += item.stat().st_size
    return count, total_bytes


def extracted_competition_data_stats(kaggle_dir: Path) -> dict[str, Any]:
    extracted_path = kaggle_dir / "competition data"
    exists = extracted_path.exists()
    stats: dict[str, Any] = {
        "exists": exists,
        "path": str(extracted_path),
        "file_count": 0,
        "size_bytes": 0,
        "size_gib": 0.0,
        "status": "ignored_by_zip_streaming_pipeline",
    }
    if exists:
        file_count, total_bytes = directory_file_stats(extracted_path)
        stats.update(
            {
                "file_count": file_count,
                "size_bytes": total_bytes,
                "size_gib": round(total_bytes / GIB, 6),
            }
        )
    return stats


def print_extracted_competition_data(stats: dict[str, Any]) -> None:
    print(f"extracted_competition_data exists: {stats['exists']}")
    if not stats["exists"]:
        return

    print(f"extracted_competition_data files: {stats['file_count']}")
    print(f"extracted_competition_data size: {format_gib(int(stats['size_bytes']))}")
    print("extracted_competition_data status: ignored by zip-streaming Kaggle pipeline")


def parse_scalar(value: str) -> object:
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


def parse_key_value_output(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if " " in key:
            continue
        parsed[key] = parse_scalar(value)
    return parsed


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def object_count(value: object, key: str) -> int:
    if isinstance(value, dict):
        return int(value.get(key, 0) or 0)
    return 0


def build_zip_structure_report(kaggle_dir: Path) -> dict[str, Any]:
    inventory_path = kaggle_dir / "inventory" / "kaggle_zip_inventory_summary.json"
    summary = read_json_if_exists(inventory_path)
    if summary is None:
        return {
            "ok": False,
            "inventory_path": str(inventory_path),
            "required_path_buckets": {},
            "required_metadata_files": {},
            "errors": [f"Missing zip inventory summary: {inventory_path}"],
            "warnings": [],
        }

    path_buckets = summary.get("path_buckets", {})
    required_bucket_names = ("train/tdcsfog", "train/defog")
    required_path_buckets: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for bucket_name in required_bucket_names:
        bucket = path_buckets.get(bucket_name, {}) if isinstance(path_buckets, dict) else {}
        file_count = object_count(bucket, "file_count")
        csv_count = object_count(bucket, "csv_count")
        required_path_buckets[bucket_name] = {
            "exists": file_count > 0 and csv_count > 0,
            "file_count": file_count,
            "csv_count": csv_count,
            "compressed_size": object_count(bucket, "compressed_size"),
            "uncompressed_size": object_count(bucket, "uncompressed_size"),
        }
        if file_count <= 0 or csv_count <= 0:
            errors.append(f"Missing required supervised train CSV bucket: {bucket_name}")

    groups = summary.get("groups", {})
    metadata_group = groups.get("metadata", {}) if isinstance(groups, dict) else {}
    metadata_paths = set(str(path) for path in metadata_group.get("sample_paths", []) if path)
    required_metadata_names = (
        "tdcsfog_metadata.csv",
        "defog_metadata.csv",
        "subjects.csv",
        "events.csv",
        "tasks.csv",
        "daily_metadata.csv",
    )
    required_metadata_files = {
        name: {"exists": name in metadata_paths}
        for name in required_metadata_names
    }
    for name, status in required_metadata_files.items():
        if not status["exists"]:
            errors.append(f"Missing required metadata file: {name}")

    skipped_path_buckets: dict[str, dict[str, Any]] = {}
    for bucket_name in ("train/notype", "unlabeled"):
        bucket = path_buckets.get(bucket_name, {}) if isinstance(path_buckets, dict) else {}
        skipped_path_buckets[bucket_name] = {
            "file_count": object_count(bucket, "file_count"),
            "csv_count": object_count(bucket, "csv_count"),
            "compressed_size": object_count(bucket, "compressed_size"),
            "uncompressed_size": object_count(bucket, "uncompressed_size"),
        }

    return {
        "ok": not errors,
        "inventory_path": str(inventory_path),
        "required_path_buckets": required_path_buckets,
        "required_metadata_files": required_metadata_files,
        "selected_supervised_train_csv_files": int(
            sum(bucket["csv_count"] for bucket in required_path_buckets.values())
        ),
        "skipped_path_buckets": skipped_path_buckets,
        "errors": errors,
        "warnings": [],
    }


def validate_zip_structure(kaggle_dir: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    name = "Validate zip supervised structure"
    print(f"\n== {name} ==", flush=True)
    report = build_zip_structure_report(kaggle_dir)
    for bucket_name, bucket in report["required_path_buckets"].items():
        print(
            f"{bucket_name}: files={bucket['file_count']} csv={bucket['csv_count']} "
            f"uncompressed={format_gib(int(bucket['uncompressed_size']))}"
        )
    for metadata_name, status in report["required_metadata_files"].items():
        print(f"{metadata_name}: exists={status['exists']}")
    for error in report["errors"]:
        print(f"zip_structure_error: {error}", file=sys.stderr)

    steps.append(
        {
            "name": name,
            "status": "passed" if report["ok"] else "failed",
            "command": [],
            "captured_stdout": False,
            "returncode": 0 if report["ok"] else 1,
        }
    )
    return report


def resolve_repo_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_and_remove_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = read_json_if_exists(path)
    if path.exists():
        path.unlink()
    return value


def build_preflight_report(
    *,
    status: str,
    repo_root: Path,
    dataset_root: Path,
    kaggle_dir: Path,
    processed_path: Path,
    smoke_path: Path,
    had_processed: bool,
    had_smoke: bool,
    extracted_stats: dict[str, Any],
    storage_report: dict[str, Any] | None,
    streaming_report: dict[str, Any] | None,
    suite_preflight_report: dict[str, Any] | None,
    zip_structure_report: dict[str, Any] | None,
    smoke_limit: int,
    streaming_stdout: str,
    suite_config: Path,
    steps: list[dict[str, Any]],
    pytest_basetemp: Path,
    pytest_ran: bool,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_processed_after = processed_path.exists()
    has_smoke_after = smoke_path.exists()
    report: dict[str, Any] = {
        "status": status,
        "repo_root": str(repo_root),
        "dataset_root": str(dataset_root),
        "kaggle_dir": str(kaggle_dir),
        "processed_output_guard": {
            "processed_path": str(processed_path),
            "processed_smoke_path": str(smoke_path),
            "processed_exists_before": had_processed,
            "processed_smoke_exists_before": had_smoke,
            "processed_exists_after": has_processed_after,
            "processed_smoke_exists_after": has_smoke_after,
            "no_processed_output_created": (had_processed or not has_processed_after)
            and (had_smoke or not has_smoke_after),
        },
        "extracted_competition_data": extracted_stats,
        "zip_inventory": read_json_if_exists(kaggle_dir / "inventory" / "kaggle_zip_inventory_summary.json"),
        "zip_structure": zip_structure_report,
        "preflight_options": {
            "smoke_limit": smoke_limit,
            "suite_config": str(suite_config),
        },
        "storage_estimate": storage_report,
        "streaming_dry_run": streaming_report if streaming_report is not None else parse_key_value_output(streaming_stdout),
        "suite_preflight": suite_preflight_report,
        "suite_dry_run": {
            "config": str(suite_config),
            "validated_experiment_configs": status == "passed"
            and any(step["name"] == "Kaggle suite config dry-run" and step["status"] == "passed" for step in steps),
        },
        "pytest": {
            "ran": pytest_ran,
            "basetemp": str(pytest_basetemp),
        },
        "steps": steps,
    }
    if error is not None:
        report["error"] = error
    return report


def command_error_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        payload.update(
            {
                "returncode": exc.returncode,
                "cmd": exc.cmd,
                "stdout": exc.output if isinstance(exc.output, str) else "",
                "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve()
    suite_config = resolve_repo_path(args.suite_config or Path("configs/kaggle_smoke_suite.json"), repo_root)
    kaggle_dir = find_kaggle_dir(dataset_root)
    processed_path = kaggle_dir / "processed"
    smoke_path = kaggle_dir / "processed_smoke"
    had_processed = processed_path.exists()
    had_smoke = smoke_path.exists()
    steps: list[dict[str, Any]] = []
    extracted_stats = extracted_competition_data_stats(kaggle_dir)
    storage_report_path: Path | None = None
    streaming_report_path: Path | None = None
    suite_preflight_report_path: Path | None = None
    zip_structure_report: dict[str, Any] | None = None
    streaming_stdout = ""

    def write_preflight_report(status: str, error: dict[str, Any] | None = None) -> None:
        if not args.output_json:
            return
        storage_report = read_and_remove_json(storage_report_path)
        streaming_report = read_and_remove_json(streaming_report_path)
        suite_preflight_report = read_and_remove_json(suite_preflight_report_path)
        report = build_preflight_report(
            status=status,
            repo_root=repo_root,
            dataset_root=dataset_root,
            kaggle_dir=kaggle_dir,
            processed_path=processed_path,
            smoke_path=smoke_path,
            had_processed=had_processed,
            had_smoke=had_smoke,
            extracted_stats=extracted_stats,
            storage_report=storage_report,
            streaming_report=streaming_report,
            suite_preflight_report=suite_preflight_report,
            zip_structure_report=zip_structure_report,
            smoke_limit=args.smoke_limit,
            streaming_stdout=streaming_stdout,
            suite_config=suite_config,
            steps=steps,
            pytest_basetemp=args.pytest_basetemp.resolve(),
            pytest_ran=any(step["name"] == "Synthetic Kaggle tests" for step in steps),
            error=error,
        )
        write_json_atomic(args.output_json.resolve(), report)
        print(f"preflight_report_json: {args.output_json.resolve()}")

    print(f"RepoRoot: {repo_root}")
    print(f"DatasetRoot: {dataset_root}")
    print(f"KaggleDir: {kaggle_dir}")
    print(f"processed exists before: {had_processed}")
    print(f"processed_smoke exists before: {had_smoke}")
    print_extracted_competition_data(extracted_stats)

    try:
        run_step(
            "Compile scripts",
            [
                sys.executable,
                "-m",
                "py_compile",
                str(repo_root / "scripts" / "inspect_kaggle_fog_zip.py"),
                str(repo_root / "scripts" / "estimate_kaggle_fog_storage.py"),
                str(repo_root / "scripts" / "preprocess_kaggle_fog_streaming.py"),
                str(repo_root / "scripts" / "kaggle_fog_status.py"),
                str(repo_root / "scripts" / "run_fog_experiment.py"),
                str(repo_root / "scripts" / "run_fog_suite.py"),
                str(repo_root / "scripts" / "preflight_fog_suite.py"),
                str(repo_root / "scripts" / "start_kaggle_full_pipeline.py"),
                str(repo_root / "scripts" / "start_kaggle_smoke_pipeline.py"),
                str(repo_root / "scripts" / "check_processed_pipeline.py"),
                str(repo_root / "scripts" / "prepare_processed_record_windows.py"),
                str(repo_root / "scripts" / "validate_processed_records.py"),
            ],
            repo_root,
            steps,
        )
        run_step(
            "Inspect zip central directory only",
            [
                sys.executable,
                str(repo_root / "scripts" / "inspect_kaggle_fog_zip.py"),
                "--dataset-root",
                str(dataset_root),
            ],
            repo_root,
            steps,
        )
        zip_structure_report = validate_zip_structure(kaggle_dir, steps)
        if not zip_structure_report["ok"]:
            raise RuntimeError("Kaggle zip supervised structure check failed")
        storage_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "estimate_kaggle_fog_storage.py"),
            "--dataset-root",
            str(dataset_root),
            "--source",
            "both",
            "--suite-config",
            str(suite_config),
            "--reserve-gib",
            str(args.reserve_gib),
            "--smoke-limit",
            str(args.smoke_limit),
        ]
        if not args.allow_insufficient_storage:
            storage_cmd.append("--fail-if-insufficient")
        if args.output_json:
            storage_report_path = args.output_json.resolve().with_name(f".{args.output_json.name}.storage.tmp.json")
            storage_cmd.extend(["--output-json", str(storage_report_path)])
        run_step(
            "Estimate supervised storage budget",
            storage_cmd,
            repo_root,
            steps,
        )
        streaming_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "preprocess_kaggle_fog_streaming.py"),
            "--dataset-root",
            str(dataset_root),
            "--source",
            "both",
            "--valid-only",
            "--task-only",
            "--check-headers",
            "--strict-metadata",
            "--smoke-limit",
            str(args.smoke_limit),
            "--dry-run",
        ]
        if args.output_json:
            streaming_report_path = args.output_json.resolve().with_name(f".{args.output_json.name}.streaming.tmp.json")
            streaming_cmd.extend(["--dry-run-output-json", str(streaming_report_path)])
        streaming_stdout = run_step(
            "Streaming dry-run only",
            streaming_cmd,
            repo_root,
            steps,
            capture=True,
        )
        run_step(
            "Kaggle suite config dry-run",
            [
                sys.executable,
                str(repo_root / "scripts" / "run_fog_suite.py"),
                "--config",
                str(suite_config),
                "--dry-run",
                "--skip-collection",
                "--validate-experiment-configs",
            ],
            repo_root,
            steps,
        )
        suite_preflight_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "preflight_fog_suite.py"),
            "--config",
            str(suite_config),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--allow-missing-processed",
        ]
        if args.output_json:
            suite_preflight_report_path = args.output_json.resolve().with_name(
                f".{args.output_json.name}.suite_preflight.tmp.json"
            )
            suite_preflight_cmd.extend(["--output-json", str(suite_preflight_report_path)])
        run_step(
            "FOG suite preflight before processed",
            suite_preflight_cmd,
            repo_root,
            steps,
        )
        if not args.skip_pytest:
            run_step(
                "Synthetic Kaggle tests",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(repo_root / "tests" / "test_kaggle_streaming_preprocess.py"),
                    "-q",
                    "--basetemp",
                    str(args.pytest_basetemp.resolve()),
                ],
                repo_root,
                steps,
            )

        has_processed_after = processed_path.exists()
        has_smoke_after = smoke_path.exists()
        print(f"\nprocessed exists after: {has_processed_after}")
        print(f"processed_smoke exists after: {has_smoke_after}")

        if not had_processed and has_processed_after:
            raise RuntimeError(f"Preflight created processed unexpectedly: {processed_path}")
        if not had_smoke and has_smoke_after:
            raise RuntimeError(f"Preflight created processed_smoke unexpectedly: {smoke_path}")

        write_preflight_report("passed")
        print("\nKaggle FOG preflight passed without creating processed records.")
    except Exception as exc:
        write_preflight_report("failed", command_error_payload(exc))
        raise


if __name__ == "__main__":
    main()
