#!/usr/bin/env python
"""Schedule one complete TCN receptive-field ablation fold per physical GPU.

The scheduler deliberately treats a LOSO fold as the indivisible unit of work.
Each worker is restricted to one physical GPU with ``CUDA_VISIBLE_DEVICES`` and
trains all three RF variants for that fold.  Scientific arguments unknown to
this wrapper are forwarded unchanged to ``run_daphnet_tcn_rf_ablation.py``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_daphnet_tcn_rf_ablation.py"
AUDITOR = REPO_ROOT / "scripts" / "audit_daphnet_tcn_rf_ablation.py"
CANONICAL_FOLDS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse scheduling arguments and retain unknown scientific arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic multi-GPU scheduler for the Daphnet Persistence-NBM "
            "residual_h4s TCN receptive-field ablation"
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


def parse_gpu_ids(specification: str) -> list[str]:
    """Expand a value such as ``0-3,6`` into unique physical GPU ids."""
    values: list[str] = []
    for item in str(specification).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start < 0 or end < 0:
                raise ValueError(f"GPU ids must be non-negative: {item}")
            if end < start:
                raise ValueError(f"Descending GPU range is invalid: {item}")
            expanded = [str(value) for value in range(start, end + 1)]
        else:
            value = int(item)
            if value < 0:
                raise ValueError(f"GPU ids must be non-negative: {item}")
            expanded = [str(value)]
        for value in expanded:
            if value in values:
                raise ValueError(f"Duplicate GPU id: {value}")
            values.append(value)
    if not values:
        raise ValueError("--gpus resolved to an empty list")
    return values


def parse_folds(specification: str) -> list[str]:
    """Resolve a canonical-order LOSO fold selection."""
    if str(specification).strip().lower() == "all":
        return list(CANONICAL_FOLDS)
    values = [
        value.strip().upper()
        for value in str(specification).split(",")
        if value.strip()
    ]
    if not values:
        raise ValueError("--work-folds resolved to an empty list")
    if len(values) != len(set(values)):
        raise ValueError("--work-folds must resolve to unique folds")
    unknown = sorted(set(values) - set(CANONICAL_FOLDS))
    if unknown:
        raise ValueError(
            f"Unknown folds {unknown}; expected={CANONICAL_FOLDS}"
        )
    return [fold for fold in CANONICAL_FOLDS if fold in values]


def available_gpu_indices() -> list[str]:
    """Return physical GPU indices reported by the NVIDIA driver."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError(
            "nvidia-smi was not found; configure the NVIDIA driver first"
        )
    result = subprocess.run(
        [
            executable,
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
    """Persist scheduler state without exposing a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class OutputDirectoryLock:
    """Prevent two schedulers from writing the same experiment directory."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / ".rf_ablation_scheduler.lock"
        self.token = uuid.uuid4().hex
        self.hostname = os.environ.get("HOSTNAME") or os.environ.get(
            "COMPUTERNAME"
        )
        self.acquired = False

    def acquire(self) -> None:
        payload = {
            "format_version": 1,
            "token": self.token,
            "pid": os.getpid(),
            "hostname": self.hostname,
            "created_at": timestamp(),
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    existing = json.loads(
                        self.path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    existing = {}
                same_host = existing.get("hostname") == self.hostname
                existing_pid = int(existing.get("pid", -1))
                if same_host and not process_is_alive(existing_pid):
                    # Recover only a demonstrably stale same-host lock.
                    self.path.unlink()
                    continue
                raise RuntimeError(
                    "Another scheduler may already be using this output "
                    f"directory. Lock={self.path}, owner={existing}"
                )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.acquired = True
            atexit.register(self.release)
            return
        raise RuntimeError(f"Could not acquire scheduler lock: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("token") == self.token:
                self.path.unlink()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        self.acquired = False


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_logged(
    *,
    label: str,
    command: list[str],
    log_path: Path,
) -> int:
    """Run a foreground protocol stage and append all output to its log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[multigpu] {label}: log={log_path}", flush=True)
    print(f"[multigpu] {label}: command={command_text(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{timestamp()}] START label={label}\n")
        log.write(f"[{timestamp()}] command={command_text(command)}\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"[{timestamp()}] END return_code={process.returncode}\n")
        log.flush()
    print(
        f"[multigpu] {label}: return_code={process.returncode}",
        flush=True,
    )
    return int(process.returncode)


def base_command(
    args: argparse.Namespace,
    forwarded: list[str],
) -> list[str]:
    """Construct the immutable part of every runner invocation."""
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
        raise FileNotFoundError(f"Processed data directory is missing: {args.data_dir}")
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
            f"Requested GPUs unavailable: {unavailable}; available={available}"
        )
    print(
        f"[multigpu] GPU check passed: selected={gpus} available={available}",
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
        "scheduler_version": "daphnet_tcn_rf_ablation_multigpu.v1",
        "run_id": uuid.uuid4().hex,
        "status": "initializing",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "ended_at": None,
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
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
                fold_state.update(
                    {
                        "status": "running",
                        "gpu": gpu,
                        "pid": process.pid,
                        "return_code": None,
                        "started_at": timestamp(),
                        "ended_at": None,
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
