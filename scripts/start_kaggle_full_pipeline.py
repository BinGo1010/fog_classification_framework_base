#!/usr/bin/env python
"""Cross-platform launcher for the full supervised Kaggle FOG pipeline.

The default mode is intentionally safe: it runs preflight, a full streaming
dry-run with header checks, and full suite dry-run. Pass --execute to create
processed/ and launch the real full stages.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the full Kaggle supervised pipeline; defaults to safe dry-run mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=Path, default=repo_root)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--record-compression", choices=("compressed", "none"), default="compressed")
    parser.add_argument("--only", choices=("all", "windowing", "validation", "training", "collection"), default="all")
    parser.add_argument("--execute", action="store_true", help="Create processed/ and run real full stages.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument(
        "--allow-execute-without-preflight",
        action="store_true",
        help="Allow --execute with --no-preflight. Intended only for manual recovery after separate checks.",
    )
    parser.add_argument(
        "--allow-execute-without-status-gate",
        action="store_true",
        help="Allow --execute without the status readiness gate. Intended only for manual recovery.",
    )
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument(
        "--post-check-window-dry-run",
        action="store_true",
        help="After execute preprocessing, run check_processed_pipeline.py instead of records-only validation.",
    )
    parser.add_argument("--no-suite", action="store_true")
    parser.add_argument("--no-reuse-existing-windows", action="store_true")
    parser.add_argument("--no-dedupe-windowing", action="store_true")
    parser.add_argument("--no-skip-completed-training", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--dry-run-json", type=Path, default=None)
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument(
        "--profile-data",
        action="store_true",
        help="Add --profile-data to streaming dry-runs to report NaN/non-finite and label counts without creating records.",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.execute and args.no_preflight and not args.allow_execute_without_preflight:
        parser.error("--execute with --no-preflight requires --allow-execute-without-preflight")
    return args


def find_kaggle_dir(dataset_root: Path) -> Path:
    matches = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("2.Kaggle")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one 2.Kaggle* directory under {dataset_root}, found {matches}")
    return matches[0]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def log_line(log_path: Path, message: str) -> None:
    line = f"[{timestamp()}] {message}\n"
    print(line, end="", flush=True)
    append_log(log_path, line)


def format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    log_line(log_path, f"CMD {format_command(command)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_parts: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        append_log(log_path, line)
        output_parts.append(line)
    returncode = process.wait()
    output = "".join(output_parts)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=output)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    dataset_root = (args.dataset_root or (repo / "dataset")).resolve()
    kaggle_dir = find_kaggle_dir(dataset_root)
    processed = kaggle_dir / "processed"
    log_path = (args.log_path or (repo / "outputs" / "logs" / "kaggle_full_pipeline.log")).resolve()
    preflight_json = (args.preflight_json or (repo / "outputs" / "kaggle_preflight_report.json")).resolve()
    dry_run_json = (args.dry_run_json or (repo / "outputs" / "kaggle_full_streaming_dry_run.json")).resolve()
    status_json = (args.status_json or (repo / "outputs" / "kaggle_status.json")).resolve()

    log_line(log_path, f"start Kaggle full pipeline execute={args.execute}")

    if not args.no_preflight:
        run_logged(
            [
                args.python,
                str(repo / "scripts" / "check_kaggle_fog_preflight.py"),
                "--repo-root",
                str(repo),
                "--dataset-root",
                str(dataset_root),
                "--suite-config",
                str(repo / "configs" / "kaggle_full_suite.json"),
                "--skip-pytest",
                "--output-json",
                str(preflight_json),
            ],
            cwd=repo,
            log_path=log_path,
        )

    preprocess_base_command = [
        args.python,
        str(repo / "scripts" / "preprocess_kaggle_fog_streaming.py"),
        "--dataset-root",
        str(dataset_root),
        "--source",
        "both",
        "--valid-only",
        "--task-only",
        "--strict-metadata",
        "--record-compression",
        args.record_compression,
    ]
    dry_run_options = [
        "--check-headers",
        "--dry-run",
        "--dry-run-output-json",
        str(dry_run_json),
    ]
    if args.profile_data:
        dry_run_options.append("--profile-data")
    if args.execute:
        run_logged(
            preprocess_base_command + dry_run_options,
            cwd=repo,
            log_path=log_path,
        )
        if not args.allow_execute_without_status_gate:
            status_command = [
                args.python,
                str(repo / "scripts" / "kaggle_fog_status.py"),
                "--repo-root",
                str(repo),
                "--dataset-root",
                str(dataset_root),
                "--preflight-json",
                str(preflight_json),
                "--full-dry-run-json",
                str(dry_run_json),
                "--output-json",
                str(status_json),
                "--require-ready",
                "full",
            ]
            if args.resume or args.overwrite:
                status_command.append("--allow-existing-output")
            run_logged(
                status_command,
                cwd=repo,
                log_path=log_path,
            )
        preprocess_command = list(preprocess_base_command)
        if args.resume:
            preprocess_command.append("--resume")
        if args.overwrite:
            preprocess_command.append("--overwrite")
    else:
        preprocess_command = preprocess_base_command + dry_run_options
    run_logged(preprocess_command, cwd=repo, log_path=log_path)

    if args.execute and not args.no_validation:
        if args.post_check_window_dry_run:
            run_logged(
                [
                    args.python,
                    str(repo / "scripts" / "check_processed_pipeline.py"),
                    "--processed-dir",
                    str(processed),
                    "--expected-channels",
                    "3",
                    "--require-success",
                    "--window-seconds",
                    "1",
                    "--stride-seconds",
                    "1",
                    "--label-mode",
                    "binary",
                    "--nan-policy",
                    "error",
                    "--target-hz",
                    "100",
                ],
                cwd=repo,
                log_path=log_path,
            )
        else:
            run_logged(
                [
                    args.python,
                    str(repo / "scripts" / "validate_processed_records.py"),
                    str(processed),
                    "--expected-channels",
                    "3",
                    "--require-success",
                ],
                cwd=repo,
                log_path=log_path,
            )
    elif not args.execute:
        log_line(log_path, "skip processed validation because --execute was not provided")

    if not args.no_suite:
        suite_command = [
            args.python,
            str(repo / "scripts" / "run_fog_suite.py"),
            "--config",
            str(repo / "configs" / "kaggle_full_suite.json"),
            "--only",
            args.only,
            "--validate-experiment-configs",
        ]
        if not args.execute:
            suite_command.extend(["--dry-run", "--skip-collection"])
        if args.no_reuse_existing_windows:
            suite_command.append("--no-reuse-existing-windows")
        if args.no_dedupe_windowing:
            suite_command.append("--no-dedupe-windowing")
        if args.no_skip_completed_training:
            suite_command.append("--no-skip-completed-training")
        run_logged(suite_command, cwd=repo, log_path=log_path)

    log_line(log_path, f"finished Kaggle full pipeline execute={args.execute}")


if __name__ == "__main__":
    main()
