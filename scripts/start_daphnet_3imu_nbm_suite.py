#!/usr/bin/env python
"""Launch the Daphnet three-IMU NBM suite with persistent tee-style logging.

Launcher-specific options are parsed here.  Every other argument is forwarded
unchanged to ``run_daphnet_3imu_nbm_suite.py``.  Resume is enabled unless the
forwarded arguments already contain ``--resume`` or ``--no-resume``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start the Daphnet 3-IMU 5-NBM x 4-history LOSO suite. "
            "Unknown options are passed through to the core runner."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: %(prog)s --log-dir outputs/logs "
            "--data-dir /data/daphnet/processed --output-dir /runs/nbm_suite "
            "--device cuda. Use --show-core-help for all training options."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used for the core runner; PYTHON_BIN is also supported",
    )
    logging = parser.add_mutually_exclusive_group()
    logging.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append console output to this file",
    )
    logging.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for a unique timestamped log; defaults to "
            "DAPHNET_SUITE_LOG_DIR or outputs/logs/daphnet_3imu_nbm_suite"
        ),
    )
    parser.add_argument(
        "--show-core-help",
        action="store_true",
        help="Run the core suite with --help instead of starting training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and log path without running it",
    )
    parser.set_defaults(repo_root=repo_root)
    return parser


def resolve_repo_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def default_log_path(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.log_file is not None:
        return resolve_repo_path(args.log_file, repo_root)
    configured_dir = args.log_dir
    if configured_dir is None:
        env_dir = os.environ.get("DAPHNET_SUITE_LOG_DIR")
        configured_dir = (
            Path(env_dir)
            if env_dir
            else Path("outputs/logs/daphnet_3imu_nbm_suite")
        )
    log_dir = resolve_repo_path(configured_dir, repo_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"suite_{stamp}_pid{os.getpid()}.log"


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def strip_separator(arguments: list[str]) -> list[str]:
    if arguments and arguments[0] == "--":
        return arguments[1:]
    return arguments


def build_command(
    args: argparse.Namespace,
    forwarded: list[str],
    repo_root: Path,
) -> list[str]:
    runner = repo_root / "scripts" / "run_daphnet_3imu_nbm_suite.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Core suite runner is missing: {runner}")
    command = [str(args.python), "-u", str(runner)]
    if args.show_core_help:
        return [*command, "--help"]
    if "--resume" not in forwarded and "--no-resume" not in forwarded:
        command.append("--resume")
    command.extend(forwarded)
    return command


def run_logged(command: list[str], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    command_text = format_command(command)
    started = timestamp()
    print(f"[launcher] log={log_path}", flush=True)
    print(f"[launcher] command={command_text}", flush=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        def emit(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")

        emit(f"[{started}] START")
        emit(f"[{started}] cwd={repo_root}")
        emit(f"[{started}] command={command_text}")
        try:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            emit(f"[{timestamp()}] LAUNCH_ERROR {exc}")
            return 127

        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        except KeyboardInterrupt:
            emit(f"[{timestamp()}] INTERRUPTED forwarding termination to child")
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            return_code = 130

        emit(f"[{timestamp()}] END return_code={return_code}")
        return int(return_code)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args, forwarded = parser.parse_known_args(argv)
    forwarded = strip_separator(list(forwarded))
    if args.show_core_help and forwarded:
        parser.error("--show-core-help does not accept forwarded suite arguments")
    log_path = default_log_path(args, repo_root)
    try:
        command = build_command(args, forwarded, repo_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"log={log_path}")
        print(f"command={format_command(command)}")
        return 0
    return run_logged(command, repo_root, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
