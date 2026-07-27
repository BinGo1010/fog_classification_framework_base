#!/usr/bin/env python
"""Schedule one complete baseline-suite LOSO fold per physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_daphnet_baseline_suite.py"
EXPECTED_FOLDS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
RESERVED_PREFIXES = (
    "--data-dir",
    "--output-dir",
    "--folds",
    "--worker-fold",
    "--finalize-only",
    "--device",
    "--resume",
    "--no-resume",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Dynamic multi-GPU scheduler for Daphnet reference baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0-6")
    parser.add_argument("--work-folds", default="all")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--launch-delay", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--allow-partial-audit", action="store_true")
    args, forwarded = parser.parse_known_args()
    for value in forwarded:
        if any(
            value == prefix or value.startswith(prefix + "=")
            for prefix in RESERVED_PREFIXES
        ):
            raise ValueError(
                f"{value} is scheduler-controlled and cannot be forwarded"
            )
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if args.launch_delay < 0 or args.poll_seconds <= 0:
        raise ValueError("delay must be non-negative and polling positive")
    return args, forwarded


def parse_range_list(specification: str) -> list[str]:
    values: list[str] = []
    for item in str(specification).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError(f"Descending GPU range is invalid: {item}")
            expanded = [str(value) for value in range(start, end + 1)]
        else:
            expanded = [str(int(item))]
        for value in expanded:
            if value in values:
                raise ValueError(f"Duplicate GPU id: {value}")
            values.append(value)
    if not values:
        raise ValueError("--gpus resolved to an empty list")
    return values


def parse_folds(specification: str) -> list[str]:
    if str(specification).strip().lower() == "all":
        return list(EXPECTED_FOLDS)
    values = [
        value.strip().upper()
        for value in str(specification).split(",")
        if value.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("--work-folds must resolve to unique folds")
    unknown = sorted(set(values) - set(EXPECTED_FOLDS))
    if unknown:
        raise ValueError(f"Unknown folds {unknown}; expected={EXPECTED_FOLDS}")
    return [fold for fold in EXPECTED_FOLDS if fold in values]


def available_gpu_indices() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{timestamp()}] command={subprocess.list2cmdline(command)}\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"[{timestamp()}] return_code={process.returncode}\n")
        return int(process.returncode)


def base_command(
    args: argparse.Namespace,
    forwarded: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(RUNNER),
        "--resume",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--folds",
        "all",
        *forwarded,
    ]


def main() -> None:
    args, forwarded = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    gpus = parse_range_list(args.gpus)
    folds = parse_folds(args.work_folds)
    available = available_gpu_indices()
    unavailable = [gpu for gpu in gpus if gpu not in available]
    if unavailable:
        raise RuntimeError(
            f"Requested GPUs unavailable: {unavailable}; available={available}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "multigpu_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "multigpu_status.json"
    state: dict[str, Any] = {
        "format_version": 1,
        "scheduler_version": "daphnet_reference_baselines_multigpu.v1",
        "run_id": uuid.uuid4().hex,
        "status": "initializing",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "ended_at": None,
        "scheduler_pid": os.getpid(),
        "repo_root": str(REPO_ROOT),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "log_dir": str(log_dir),
        "python": sys.executable,
        "gpus": gpus,
        "work_folds": folds,
        "max_retries": args.max_retries,
        "forwarded_scientific_args": forwarded,
        "folds": {
            fold: {
                "status": "pending",
                "attempts": 0,
                "gpu": None,
                "pid": None,
                "return_code": None,
                "log": str(log_dir / f"{fold}.log"),
            }
            for fold in folds
        },
    }
    atomic_json_dump(state, status_path)

    initialize = [
        *base_command(args, forwarded),
        "--finalize-only",
        "--device",
        "cpu",
    ]
    return_code = run_logged(initialize, log_dir / "initialize.log")
    if return_code:
        state.update(
            {
                "status": "failed_initialization",
                "updated_at": timestamp(),
                "ended_at": timestamp(),
                "return_code": return_code,
            }
        )
        atomic_json_dump(state, status_path)
        raise SystemExit(return_code)

    pending = list(folds)
    idle_gpus = list(gpus)
    running: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    state["status"] = "running"
    atomic_json_dump(state, status_path)
    try:
        while pending or running:
            while pending and idle_gpus:
                fold = pending.pop(0)
                gpu = idle_gpus.pop(0)
                fold_state = state["folds"][fold]
                fold_state["attempts"] += 1
                log_path = Path(fold_state["log"])
                log_handle = log_path.open("a", encoding="utf-8")
                command = [
                    *base_command(args, forwarded),
                    "--worker-fold",
                    fold,
                    "--device",
                    "cuda",
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment.setdefault("OMP_NUM_THREADS", "1")
                environment.setdefault("MKL_NUM_THREADS", "1")
                log_handle.write(
                    f"[{timestamp()}] gpu={gpu} "
                    f"command={subprocess.list2cmdline(command)}\n"
                )
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                fold_state.update(
                    {
                        "status": "running",
                        "gpu": gpu,
                        "pid": process.pid,
                        "return_code": None,
                    }
                )
                running[fold] = {
                    "process": process,
                    "gpu": gpu,
                    "log_handle": log_handle,
                }
                print(
                    f"[multigpu] launched fold={fold} gpu={gpu} "
                    f"attempt={fold_state['attempts']} pid={process.pid}",
                    flush=True,
                )
                state["updated_at"] = timestamp()
                atomic_json_dump(state, status_path)
                if args.launch_delay:
                    time.sleep(args.launch_delay)

            time.sleep(args.poll_seconds)
            for fold, active in list(running.items()):
                process: subprocess.Popen = active["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                active["log_handle"].write(
                    f"[{timestamp()}] return_code={return_code}\n"
                )
                active["log_handle"].close()
                gpu = active["gpu"]
                idle_gpus.append(gpu)
                idle_gpus.sort(key=gpus.index)
                del running[fold]
                fold_state = state["folds"][fold]
                fold_state["return_code"] = int(return_code)
                if return_code == 0:
                    fold_state["status"] = "complete"
                    print(
                        f"[multigpu] complete fold={fold} gpu={gpu}",
                        flush=True,
                    )
                elif fold_state["attempts"] <= args.max_retries:
                    fold_state["status"] = "retry_pending"
                    pending.append(fold)
                    print(
                        f"[multigpu] retry fold={fold} "
                        f"return_code={return_code}",
                        flush=True,
                    )
                else:
                    fold_state["status"] = "failed"
                    failures.append(fold)
                    print(
                        f"[multigpu] failed fold={fold} "
                        f"return_code={return_code}",
                        flush=True,
                    )
                state["updated_at"] = timestamp()
                atomic_json_dump(state, status_path)
    except BaseException:
        for active in running.values():
            active["process"].terminate()
            active["log_handle"].close()
        state.update(
            {
                "status": "interrupted",
                "updated_at": timestamp(),
                "ended_at": timestamp(),
            }
        )
        atomic_json_dump(state, status_path)
        raise

    finalize_code = run_logged(
        initialize,
        log_dir / "finalize.log",
    )
    audit_code = 0
    if args.audit:
        audit_command = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "audit_daphnet_baseline_suite.py"),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.allow_partial_audit or len(folds) != len(EXPECTED_FOLDS):
            audit_command.append("--allow-partial")
        audit_code = run_logged(audit_command, log_dir / "audit.log")
    status = (
        "complete"
        if not failures and finalize_code == 0 and audit_code == 0
        else "failed"
    )
    state.update(
        {
            "status": status,
            "updated_at": timestamp(),
            "ended_at": timestamp(),
            "failed_folds": failures,
            "finalize_return_code": finalize_code,
            "audit_return_code": audit_code if args.audit else None,
        }
    )
    atomic_json_dump(state, status_path)
    if status != "complete":
        raise SystemExit(1)
    print(f"[multigpu] complete output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
