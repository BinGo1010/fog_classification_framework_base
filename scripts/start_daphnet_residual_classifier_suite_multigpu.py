#!/usr/bin/env python
"""Schedule the residual-classifier comparison one complete fold per GPU.

The LOSO fold is the indivisible scheduling unit: one worker is pinned to one
physical GPU with ``CUDA_VISIBLE_DEVICES`` and trains MLP, 1D-CNN, GRU, and the
lightweight Transformer sequentially for that fold.  With seven GPUs and the
canonical eight folds, the first seven folds start concurrently and the final
fold is assigned to the first GPU that becomes idle.

Arguments not owned by this wrapper are forwarded unchanged to
``run_daphnet_residual_classifier_suite.py``.  This keeps scientific
hyperparameters in the experiment runner while this module owns only process
scheduling, retry, resume, logging, locking, finalization, and optional audit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import start_daphnet_tcn_rf_ablation_multigpu as scheduler_base


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_daphnet_residual_classifier_suite.py"
AUDITOR = REPO_ROOT / "scripts" / "audit_daphnet_residual_classifier_suite.py"
CANONICAL_FOLDS = scheduler_base.CANONICAL_FOLDS
CANONICAL_CLASSIFIERS = ("mlp", "cnn1d", "gru", "transformer")
SCHEDULER_VERSION = "daphnet_residual_classifier_multigpu.v1"
RESERVED_PREFIXES = (
    "--data-dir",
    "--source-suite-dir",
    "--output-dir",
    "--folds",
    "--worker-fold",
    "--finalize-only",
    "--device",
    "--resume",
    "--no-resume",
)

# Reuse the already exercised, side-effect-free scheduling helpers.  Keeping
# these names public also makes this launcher independently helper-testable.
parse_gpu_ids = scheduler_base.parse_gpu_ids
parse_folds = scheduler_base.parse_folds
available_gpu_indices = scheduler_base.available_gpu_indices
atomic_json_dump = scheduler_base.atomic_json_dump
paths_overlap = scheduler_base.paths_overlap
process_is_alive = scheduler_base.process_is_alive
timestamp = scheduler_base.timestamp
command_text = scheduler_base.command_text
run_logged = scheduler_base.run_logged


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse scheduler controls and preserve scientific runner arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic multi-GPU scheduler for the Daphnet Persistence-NBM "
            "residual_h4s classifier comparison"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        default="0-6",
        help="Comma-separated physical GPU ids and/or inclusive ranges",
    )
    parser.add_argument(
        "--work-folds",
        default="all",
        help="all or a comma-separated subset of the canonical eight folds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries after the first worker attempt",
    )
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=2.0,
        help="Seconds between worker launches",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Worker polling interval",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=10.0,
        help="Refresh multigpu_status.json while workers remain active",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the independent result auditor after finalization",
    )
    parser.add_argument(
        "--allow-partial-audit",
        action="store_true",
        help="Permit auditing a deliberately selected fold subset",
    )
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
    if args.launch_delay < 0:
        raise ValueError("--launch-delay must be non-negative")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive")
    return args, forwarded


class OutputDirectoryLock(scheduler_base.OutputDirectoryLock):
    """Prevent concurrent schedulers from sharing one comparison output."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.path = output_dir / ".residual_classifier_scheduler.lock"


def base_command(
    args: argparse.Namespace,
    forwarded: list[str],
) -> list[str]:
    """Build the common, resumable runner invocation."""
    return [
        sys.executable,
        "-u",
        str(RUNNER),
        "--resume",
        "--data-dir",
        str(args.data_dir),
        "--source-suite-dir",
        str(args.source_suite_dir),
        "--output-dir",
        str(args.output_dir),
        "--folds",
        "all",
        *forwarded,
    ]


def main() -> None:
    args, forwarded = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.source_suite_dir = args.source_suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    for label, protected in (
        ("source suite", args.source_suite_dir),
        ("processed data", args.data_dir),
    ):
        if paths_overlap(args.output_dir, protected):
            raise ValueError(
                f"Output directory must be separate from the {label}: "
                f"output={args.output_dir}, protected={protected}"
            )

    if not RUNNER.is_file():
        raise FileNotFoundError(f"Experiment runner is missing: {RUNNER}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Processed data directory is missing: {args.data_dir}"
        )
    if not args.source_suite_dir.is_dir():
        raise FileNotFoundError(
            f"Source NBM suite directory is missing: {args.source_suite_dir}"
        )
    if args.audit and not AUDITOR.is_file():
        raise FileNotFoundError(f"Result auditor is missing: {AUDITOR}")

    gpus = parse_gpu_ids(args.gpus)
    work_folds = parse_folds(args.work_folds)
    available = available_gpu_indices()
    unavailable = [gpu for gpu in gpus if gpu not in available]
    if unavailable:
        raise RuntimeError(
            f"Requested GPUs unavailable: {unavailable}; "
            f"available={available}"
        )
    print(
        f"[multigpu] GPU check passed: selected={gpus} "
        f"available={available}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_lock = OutputDirectoryLock(args.output_dir)
    scheduler_lock.acquire()
    log_dir = args.output_dir / "multigpu_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "multigpu_status.json"
    state: dict[str, Any] = {
        "format_version": 1,
        "scheduler_version": SCHEDULER_VERSION,
        "run_id": uuid.uuid4().hex,
        "status": "initializing",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "ended_at": None,
        "hostname": os.environ.get("HOSTNAME")
        or os.environ.get("COMPUTERNAME"),
        "scheduler_pid": os.getpid(),
        "repo_root": str(REPO_ROOT),
        "data_dir": str(args.data_dir),
        "source_suite_dir": str(args.source_suite_dir),
        "output_dir": str(args.output_dir),
        "log_dir": str(log_dir),
        "python": sys.executable,
        "gpus": gpus,
        "work_folds": work_folds,
        "configured_core_folds": "all",
        "fold_is_scheduling_unit": True,
        "classifiers_per_fold": len(CANONICAL_CLASSIFIERS),
        "classifiers": list(CANONICAL_CLASSIFIERS),
        "max_retries": args.max_retries,
        "max_attempts": args.max_retries + 1,
        "launch_delay_seconds": args.launch_delay,
        "poll_seconds": args.poll_seconds,
        "heartbeat_seconds": args.heartbeat_seconds,
        "audit_enabled": bool(args.audit),
        "audit_allow_partial": bool(
            args.allow_partial_audit
            or set(work_folds) != set(CANONICAL_FOLDS)
        ),
        "forwarded_scientific_args": forwarded,
        "initialization_return_code": None,
        "finalize_return_code": None,
        "audit_return_code": None,
        "folds": {
            fold: {
                "status": "pending",
                "attempts": 0,
                "gpu": None,
                "pid": None,
                "return_code": None,
                "started_at": None,
                "ended_at": None,
                "heartbeat_at": None,
                "log": str(log_dir / f"{fold}.log"),
            }
            for fold in work_folds
        },
    }
    atomic_json_dump(state, status_path)

    initialize_command = [
        *base_command(args, forwarded),
        "--finalize-only",
        "--device",
        "cpu",
    ]
    initialize_code = run_logged(
        label="initialize_protocol",
        command=initialize_command,
        log_path=log_dir / "initialize.log",
    )
    state["initialization_return_code"] = initialize_code
    state["updated_at"] = timestamp()
    if initialize_code:
        state.update(
            {
                "status": "failed_initialization",
                "ended_at": timestamp(),
            }
        )
        atomic_json_dump(state, status_path)
        raise SystemExit(initialize_code)

    pending = list(work_folds)
    idle_gpus = list(gpus)
    running: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    state["status"] = "running"
    state["updated_at"] = timestamp()
    atomic_json_dump(state, status_path)
    last_heartbeat = time.monotonic()

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
                    f"[{timestamp()}] START fold={fold} gpu={gpu} "
                    f"attempt={fold_state['attempts']}\n"
                )
                log_handle.write(
                    f"[{timestamp()}] command={command_text(command)}\n"
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
                started_at = timestamp()
                fold_state.update(
                    {
                        "status": "running",
                        "gpu": gpu,
                        "pid": process.pid,
                        "return_code": None,
                        "started_at": started_at,
                        "ended_at": None,
                        "heartbeat_at": started_at,
                    }
                )
                running[fold] = {
                    "process": process,
                    "gpu": gpu,
                    "log_handle": log_handle,
                }
                print(
                    f"[multigpu] launched fold={fold} gpu={gpu} "
                    f"attempt={fold_state['attempts']} pid={process.pid} "
                    f"log={log_path}",
                    flush=True,
                )
                state["updated_at"] = timestamp()
                atomic_json_dump(state, status_path)
                if args.launch_delay:
                    time.sleep(args.launch_delay)

            if running:
                time.sleep(args.poll_seconds)

            for fold, active in list(running.items()):
                process: subprocess.Popen[Any] = active["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                active["log_handle"].write(
                    f"[{timestamp()}] END return_code={return_code}\n"
                )
                active["log_handle"].flush()
                active["log_handle"].close()
                gpu = active["gpu"]
                idle_gpus.append(gpu)
                idle_gpus.sort(key=gpus.index)
                del running[fold]

                fold_state = state["folds"][fold]
                fold_state.update(
                    {
                        "return_code": int(return_code),
                        "pid": None,
                        "ended_at": timestamp(),
                    }
                )
                if return_code == 0:
                    fold_state["status"] = "complete"
                    print(
                        f"[multigpu] completed fold={fold} gpu={gpu} "
                        f"attempt={fold_state['attempts']}",
                        flush=True,
                    )
                elif fold_state["attempts"] <= args.max_retries:
                    fold_state["status"] = "retry_pending"
                    pending.append(fold)
                    print(
                        f"[multigpu] retrying fold={fold} gpu={gpu} "
                        f"return_code={return_code} "
                        f"next_attempt={fold_state['attempts'] + 1}",
                        flush=True,
                    )
                else:
                    fold_state["status"] = "failed"
                    failures.append(fold)
                    print(
                        f"[multigpu] failed fold={fold} gpu={gpu} "
                        f"attempts={fold_state['attempts']} "
                        f"return_code={return_code}",
                        flush=True,
                    )
                state["updated_at"] = timestamp()
                atomic_json_dump(state, status_path)

            if time.monotonic() - last_heartbeat >= args.heartbeat_seconds:
                heartbeat_at = timestamp()
                state["updated_at"] = heartbeat_at
                for fold in running:
                    state["folds"][fold]["heartbeat_at"] = heartbeat_at
                atomic_json_dump(state, status_path)
                last_heartbeat = time.monotonic()
    except BaseException:
        for fold, active in running.items():
            active["process"].terminate()
            active["log_handle"].write(
                f"[{timestamp()}] scheduler interrupted; worker terminated\n"
            )
            active["log_handle"].flush()
            active["log_handle"].close()
            state["folds"][fold].update(
                {
                    "status": "interrupted",
                    "pid": None,
                    "ended_at": timestamp(),
                }
            )
        state.update(
            {
                "status": "interrupted",
                "updated_at": timestamp(),
                "ended_at": timestamp(),
            }
        )
        atomic_json_dump(state, status_path)
        raise

    state["status"] = "finalizing"
    state["updated_at"] = timestamp()
    atomic_json_dump(state, status_path)
    finalize_code = run_logged(
        label="finalize_results",
        command=initialize_command,
        log_path=log_dir / "finalize.log",
    )
    state["finalize_return_code"] = finalize_code
    state["updated_at"] = timestamp()
    atomic_json_dump(state, status_path)

    audit_code = 0
    if args.audit:
        state["status"] = "auditing"
        state["updated_at"] = timestamp()
        atomic_json_dump(state, status_path)
        audit_command = [
            sys.executable,
            "-u",
            str(AUDITOR),
            "--result-dir",
            str(args.output_dir),
            "--source-suite-dir",
            str(args.source_suite_dir),
            "--data-dir",
            str(args.data_dir),
        ]
        if (
            args.allow_partial_audit
            or set(work_folds) != set(CANONICAL_FOLDS)
            or failures
        ):
            audit_command.append("--allow-partial")
        audit_code = run_logged(
            label="audit",
            command=audit_command,
            log_path=log_dir / "audit.log",
        )
        state["audit_return_code"] = audit_code

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
