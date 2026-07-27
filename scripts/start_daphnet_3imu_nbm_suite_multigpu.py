#!/usr/bin/env python
"""Schedule independent Daphnet LOSO folds across multiple GPUs.

The scientific protocol remains a single ``--folds all`` experiment.  Each
worker receives one additional ``--worker-fold Sxx`` selector and sees exactly
one physical GPU through ``CUDA_VISIBLE_DEVICES``.  Consequently one fold owns
one GPU for its complete five-NBM / twenty-classifier run, while free GPUs
dynamically take the next pending fold.

Unknown scientific options are forwarded unchanged to the core runner.
Scheduler-controlled core options are deliberately rejected to prevent a
worker from escaping its assigned fold or GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Callable, Sequence

if os.name == "nt":
    from ctypes import wintypes


RUNNER_FILENAME = "run_daphnet_3imu_nbm_suite.py"
AUDITOR_FILENAME = "audit_daphnet_3imu_nbm_suite.py"
SCHEDULER_VERSION = "daphnet_3imu_nbm_multigpu.v1"
DEFAULT_OUTPUT_DIRNAME = "daphnet_3imu_nbm_5x4_loso_seed42"
SCHEDULER_DESCRIPTION = (
    "Run Daphnet 3-IMU LOSO folds on independent GPUs. With defaults, "
    "seven workers start S01..S08 and the first free GPU continues S09; "
    "each fold runs all 5 NBMs and all 20 downstream classifiers."
)
CANONICAL_FOLDS = (
    "S01",
    "S02",
    "S03",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
)
DEFAULT_GPUS = "0,1,2,3,4,5,6"
RESERVED_FORWARDED_OPTIONS = {
    "--help",
    "--data-dir",
    "--output-dir",
    "--folds",
    "--worker-fold",
    "--finalize-only",
    "--device",
    "--resume",
    "--no-resume",
    # These controlled test hooks can make a worker return before its fold is
    # complete, so they are unsafe under scheduler success semantics.
    "--stop-after-completed-tasks",
    "--debug-interrupt-nbm-after-epoch",
    "--debug-interrupt-classifier-after-epoch",
}


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve examples while still displaying argument defaults."""


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Atomically replace a JSON status file, including a durable temp write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
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
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows, os.kill(pid, 0) is not a harmless existence probe:
        # CPython routes it through TerminateProcess. Query the process handle
        # instead so lock inspection can never kill a scheduler or worker.
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            # Access denied means that a process with this PID exists but is
            # protected. Invalid-parameter means it no longer exists. Treat
            # other query failures conservatively as alive.
            return ctypes.get_last_error() != 87
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class SchedulerLock:
    """Exclusive output-directory lock with conservative stale reclamation."""

    def __init__(self, path: Path, output_dir: Path, run_id: str) -> None:
        self.path = path
        self.output_dir = output_dir
        self.run_id = run_id
        self.hostname = socket.gethostname()
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _payload(self) -> dict[str, Any]:
        return {
            "scheduler_version": SCHEDULER_VERSION,
            "token": self.token,
            "run_id": self.run_id,
            "hostname": self.hostname,
            "pid": os.getpid(),
            "output_dir": str(self.output_dir),
            "created_at": timestamp(),
            "argv": [sys.executable, *sys.argv],
        }

    def _create_exclusive(self) -> bool:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(str(self.path), flags, 0o600)
        except FileExistsError:
            return False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self._payload(),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            raise
        self.acquired = True
        return True

    def _recorded_live_children(
        self,
        existing_lock: dict[str, Any],
    ) -> list[tuple[str, int]]:
        """Return same-run child PIDs that survived a dead scheduler.

        A scheduler can be killed with SIGKILL before it has a chance to
        terminate its workers. Reclaiming only on the scheduler PID would then
        duplicate folds. The atomic status is therefore also treated as a
        child-process registry during stale-lock recovery.
        """

        status_path = self.output_dir / "multigpu_status.json"
        if not status_path.exists():
            return []
        try:
            status = load_json_object(status_path)
        except Exception as exc:
            raise RuntimeError(
                "The scheduler owner is dead, but its child status cannot be "
                f"verified safely: {status_path}: {exc}"
            ) from exc
        if (
            status.get("run_id") != existing_lock.get("run_id")
            or status.get("hostname") != self.hostname
        ):
            # The status belongs to a different completed/older invocation,
            # so it is not evidence about this lock owner's children.
            return []

        candidates: list[tuple[str, int]] = []
        launching_stage = status.get("launching_stage")
        if isinstance(launching_stage, dict):
            raise RuntimeError(
                "The scheduler PID is dead and its last durable state contains "
                "a stage launch with an uncommitted child PID "
                f"({launching_stage.get('label', 'unknown')}). Refusing "
                "automatic stale-lock reclamation because that stage may have "
                "started. Inspect the host processes before manually "
                "recovering the scheduler lock."
            )
        launching_folds = status.get("launching_folds", {})
        if isinstance(launching_folds, dict) and launching_folds:
            rendered = ", ".join(sorted(str(fold) for fold in launching_folds))
            raise RuntimeError(
                "The scheduler PID is dead and its last durable state contains "
                f"fold launch(es) with an uncommitted child PID ({rendered}). "
                "Refusing automatic stale-lock reclamation because a worker "
                "may have started. Inspect the host processes before manually "
                "recovering the scheduler lock."
            )
        running_folds = status.get("running_folds", {})
        if isinstance(running_folds, dict):
            for fold, details in running_folds.items():
                if not isinstance(details, dict):
                    continue
                try:
                    pid = int(details.get("pid", 0))
                except (TypeError, ValueError):
                    continue
                if pid > 0:
                    candidates.append((f"fold:{fold}", pid))
        active_stage = status.get("active_stage")
        if isinstance(active_stage, dict):
            try:
                stage_pid = int(active_stage.get("pid", 0))
            except (TypeError, ValueError):
                stage_pid = 0
            if stage_pid > 0:
                candidates.append(
                    (f"stage:{active_stage.get('label', 'unknown')}", stage_pid)
                )
        return [
            (label, pid)
            for label, pid in candidates
            if process_is_alive(pid)
        ]

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            if self._create_exclusive():
                return
            try:
                raw_before = self.path.read_bytes()
                existing = json.loads(raw_before.decode("utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Scheduler lock exists but is unreadable: {self.path}: {exc}"
                ) from exc
            if not isinstance(existing, dict):
                raise RuntimeError(
                    f"Scheduler lock has an invalid payload: {self.path}"
                )
            owner_host = str(existing.get("hostname", ""))
            try:
                owner_pid = int(existing.get("pid", 0))
            except (TypeError, ValueError):
                owner_pid = 0
            if owner_host != self.hostname:
                raise RuntimeError(
                    "Scheduler output is locked by another host and cannot be "
                    f"safely reclaimed: host={owner_host!r} pid={owner_pid} "
                    f"lock={self.path}"
                )
            if process_is_alive(owner_pid):
                raise RuntimeError(
                    "Another scheduler is already active for this output: "
                    f"host={owner_host} pid={owner_pid} lock={self.path}"
                )
            live_children = self._recorded_live_children(existing)
            if live_children:
                rendered = ", ".join(
                    f"{label}=pid{pid}" for label, pid in live_children
                )
                raise RuntimeError(
                    "The scheduler PID is dead, but recorded child processes "
                    f"are still alive ({rendered}). Refusing stale-lock "
                    "reclamation to prevent duplicate fold training. Stop or "
                    "wait for those children, then rerun the same command."
                )

            stale_path = self.path.with_name(
                f"{self.path.name}.stale-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"-pid{owner_pid}-{uuid.uuid4().hex[:8]}.json"
            )
            try:
                os.replace(self.path, stale_path)
            except FileNotFoundError:
                continue
            # The lock is immutable while owned. Verify that the file moved is
            # exactly the stale snapshot inspected above before proceeding.
            try:
                raw_moved = stale_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not verify reclaimed lock {stale_path}: {exc}"
                ) from exc
            if raw_moved != raw_before:
                if not self.path.exists():
                    os.replace(stale_path, self.path)
                raise RuntimeError(
                    "Scheduler lock changed during stale reclamation; retry "
                    f"after inspecting {self.path}"
                )
            print(
                "[multigpu] reclaimed stale same-host lock "
                f"from pid={owner_pid}; archived={stale_path}",
                flush=True,
            )
        raise RuntimeError(f"Could not acquire scheduler lock: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = load_json_object(self.path)
        except FileNotFoundError:
            self.acquired = False
            return
        except Exception as exc:
            print(
                f"[multigpu] warning: cannot inspect lock during release: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
        if existing.get("token") != self.token:
            print(
                "[multigpu] warning: lock ownership changed; leaving it intact: "
                f"{self.path}",
                file=sys.stderr,
                flush=True,
            )
            return
        self.path.unlink()
        self.acquired = False


def parse_gpu_spec(specification: str) -> list[str]:
    values: list[int] = []
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not all(value.strip().isdigit() for value in bounds):
                raise ValueError(
                    f"Invalid GPU range {part!r}; use e.g. 0-6 or 0,1,2"
                )
            first, last = (int(value.strip()) for value in bounds)
            if first > last:
                raise ValueError(f"Descending GPU range is not supported: {part!r}")
            values.extend(range(first, last + 1))
        elif part.isdigit():
            values.append(int(part))
        else:
            raise ValueError(
                f"Invalid GPU identifier {part!r}; non-negative indices are required"
            )
    if not values:
        raise ValueError("At least one GPU must be selected")
    if len(values) != len(set(values)):
        raise ValueError(f"GPU selection contains duplicates: {values}")
    return [str(value) for value in values]


def parse_work_folds(specification: str) -> list[str]:
    if specification.strip().lower() == "all":
        return list(CANONICAL_FOLDS)
    folds = [part.strip().upper() for part in specification.split(",") if part.strip()]
    if not folds:
        raise ValueError("At least one work fold must be selected")
    unknown = [fold for fold in folds if fold not in CANONICAL_FOLDS]
    if unknown:
        raise ValueError(
            f"Unknown work folds {unknown}; allowed folds are {CANONICAL_FOLDS}"
        )
    if len(folds) != len(set(folds)):
        raise ValueError(f"Work-fold selection contains duplicates: {folds}")
    return folds


def strip_forward_separator(arguments: list[str]) -> list[str]:
    if arguments and arguments[0] == "--":
        return arguments[1:]
    return arguments


def forwarded_option_name(token: str) -> str | None:
    if not token.startswith("--"):
        return None
    return token.split("=", 1)[0]


def validate_forwarded(arguments: Sequence[str]) -> None:
    conflicts: set[str] = set()
    for token in arguments:
        if token == "-h":
            conflicts.add(token)
            continue
        name = forwarded_option_name(token)
        if name and any(
            reserved == name or reserved.startswith(name)
            for reserved in RESERVED_FORWARDED_OPTIONS
        ):
            conflicts.add(name)
    if conflicts:
        raise ValueError(
            "These options are controlled by the multi-GPU scheduler and "
            f"cannot be forwarded: {sorted(conflicts)}"
        )


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=SCHEDULER_DESCRIPTION,
        formatter_class=RawDefaultsHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "Example:\n"
            "  %(prog)s --data-dir /home/chb/Documents/FOG/"
            "fog_classification_framework_base/processed "
            f"--output-dir outputs/{DEFAULT_OUTPUT_DIRNAME} "
            "--gpus 0-6\n\n"
            "Unknown scientific arguments, for example --normal-epochs 12, "
            "are passed to every fold worker and both protocol-finalization "
            "calls. Resume is always enabled."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            repo_root
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed"
        ),
        help="Processed nine-channel Daphnet directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / DEFAULT_OUTPUT_DIRNAME,
        help="Shared result directory used by every fold",
    )
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPUS,
        help="Physical GPU indices as comma list and/or inclusive ranges",
    )
    parser.add_argument(
        "--work-folds",
        default="all",
        help=(
            "Folds to execute (all or comma list). The core scientific protocol "
            "still receives --folds all."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries after the first failed attempt; checkpoints are resumed",
    )
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=2.0,
        help="Seconds between worker launches to reduce simultaneous startup I/O",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Per-fold logs; defaults to OUTPUT_DIR/multigpu_logs",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable for workers, finalization, and audit",
    )
    parser.add_argument(
        "--audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the independent strict audit after finalization",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip nvidia-smi validation (use only when nvidia-smi is unavailable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete schedule without creating files or processes",
    )
    return parser


def build_worker_command(
    python: str,
    runner: Path,
    data_dir: Path,
    output_dir: Path,
    fold: str,
    forwarded: Sequence[str],
) -> list[str]:
    return [
        python,
        "-u",
        str(runner),
        "--resume",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--folds",
        "all",
        "--worker-fold",
        fold,
        "--device",
        "cuda",
        *forwarded,
    ]


def build_finalize_command(
    python: str,
    runner: Path,
    data_dir: Path,
    output_dir: Path,
    forwarded: Sequence[str],
) -> list[str]:
    return [
        python,
        "-u",
        str(runner),
        "--resume",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--folds",
        "all",
        "--finalize-only",
        "--device",
        "cpu",
        *forwarded,
    ]


def build_audit_command(
    python: str,
    auditor: Path,
    output_dir: Path,
    allow_partial: bool,
) -> list[str]:
    command = [python, "-u", str(auditor), "--result-dir", str(output_dir)]
    if allow_partial:
        command.append("--allow-partial")
    return command


def check_gpus(gpus: Sequence[str]) -> None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError(
            "nvidia-smi was not found; install/configure the NVIDIA driver or "
            "use --skip-gpu-check after independently verifying the GPUs"
        )
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi GPU query failed: "
            f"return_code={result.returncode} stderr={result.stderr.strip()!r}"
        )
    available = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }
    missing = [gpu for gpu in gpus if gpu not in available]
    if missing:
        raise RuntimeError(
            f"Requested GPUs {missing} are unavailable; nvidia-smi reports "
            f"{sorted(available)}"
        )
    print(
        f"[multigpu] GPU check passed: selected={list(gpus)} "
        f"available={sorted(available)}",
        flush=True,
    )


def child_environment(gpu: str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    if gpu is not None:
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    return environment


def forward_signal(process: subprocess.Popen[Any], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signum)
        else:
            # Workers are created as independent Windows process groups.
            # CTRL_BREAK_EVENT reaches the Python worker and its DataLoader
            # descendants, unlike Popen.send_signal(SIGINT/SIGTERM).
            process.send_signal(signal.CTRL_BREAK_EVENT)
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        pass


def force_kill_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass


class RuntimeSignals:
    def __init__(self) -> None:
        self.stop_requested = False
        self.signal_number: int | None = None
        self.stop_requested_at: float | None = None
        self.active: dict[str, subprocess.Popen[Any]] = {}

    def register(self, label: str, process: subprocess.Popen[Any]) -> None:
        self.active[label] = process

    def unregister(self, label: str) -> None:
        self.active.pop(label, None)

    def request_stop(self, signum: int, _frame: Any) -> None:
        if not self.stop_requested:
            self.stop_requested = True
            self.signal_number = signum
            self.stop_requested_at = time.monotonic()
            try:
                signal_name = signal.Signals(signum).name
            except ValueError:
                signal_name = str(signum)
            print(
                f"\n[multigpu] received {signal_name}; forwarding to "
                f"{len(self.active)} child process(es)",
                file=sys.stderr,
                flush=True,
            )
        for process in list(self.active.values()):
            forward_signal(process, signum)


def install_signal_handlers(
    runtime: RuntimeSignals,
) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, runtime.request_stop)
    return previous


def restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def interruptible_delay(seconds: float, runtime: RuntimeSignals) -> None:
    deadline = time.monotonic() + seconds
    while not runtime.stop_requested and time.monotonic() < deadline:
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def shutdown_active_processes(
    runtime: RuntimeSignals,
    grace_seconds: float = 20.0,
) -> None:
    if not runtime.active:
        return
    signum = runtime.signal_number or signal.SIGTERM
    for process in list(runtime.active.values()):
        forward_signal(process, signum)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in runtime.active.values()):
            return
        time.sleep(0.2)
    for process in list(runtime.active.values()):
        forward_signal(process, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in runtime.active.values()):
            return
        time.sleep(0.2)
    for process in list(runtime.active.values()):
        if process.poll() is None:
            force_kill_process_tree(process)


def write_log_header(
    handle: IO[str],
    label: str,
    command: Sequence[str],
    gpu: str | None,
    attempt: int | None,
) -> None:
    handle.write("\n" + "=" * 88 + "\n")
    handle.write(f"[{timestamp()}] START label={label}")
    if gpu is not None:
        handle.write(f" physical_gpu={gpu}")
    if attempt is not None:
        handle.write(f" attempt={attempt}")
    handle.write("\n")
    handle.write(f"[{timestamp()}] command={format_command(command)}\n")
    handle.flush()


def run_stage(
    *,
    label: str,
    command: list[str],
    log_path: Path,
    repo_root: Path,
    runtime: RuntimeSignals,
    on_started: Callable[
        [subprocess.Popen[Any], dict[str, Any]], None
    ]
    | None = None,
) -> tuple[int, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = timestamp()
    print(
        f"[multigpu] {label}: log={log_path}\n"
        f"[multigpu] {label}: command={format_command(command)}",
        flush=True,
    )
    record: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "return_code": None,
        "log": str(log_path),
        "command": command,
    }
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        write_log_header(log_handle, label, command, None, None)
        try:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=child_environment(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
        except OSError as exc:
            log_handle.write(f"[{timestamp()}] LAUNCH_ERROR {exc}\n")
            record.update(
                {
                    "status": "failed",
                    "ended_at": timestamp(),
                    "return_code": 127,
                    "launch_error": str(exc),
                }
            )
            return 127, record
        runtime.register(label, process)
        record["pid"] = process.pid
        if on_started is not None:
            on_started(process, dict(record))
        try:
            while process.poll() is None:
                if (
                    runtime.stop_requested
                    and runtime.stop_requested_at is not None
                    and time.monotonic() - runtime.stop_requested_at > 20.0
                ):
                    forward_signal(process, signal.SIGTERM)
                if (
                    runtime.stop_requested
                    and runtime.stop_requested_at is not None
                    and time.monotonic() - runtime.stop_requested_at > 25.0
                    and process.poll() is None
                ):
                    force_kill_process_tree(process)
                time.sleep(0.2)
            return_code = int(process.returncode)
        finally:
            runtime.unregister(label)
        ended_at = timestamp()
        log_handle.write(
            f"[{ended_at}] END label={label} return_code={return_code}\n"
        )
    record.update(
        {
            "status": "succeeded" if return_code == 0 else "failed",
            "ended_at": ended_at,
            "return_code": return_code,
        }
    )
    print(
        f"[multigpu] {label}: return_code={return_code}",
        flush=True,
    )
    return return_code, record


@dataclass
class RunningFold:
    fold: str
    gpu: str
    attempt: int
    process: subprocess.Popen[Any]
    log_path: Path
    log_handle: IO[str]
    command: list[str]
    started_at: str


def launch_fold(
    *,
    fold: str,
    gpu: str,
    attempt: int,
    command: list[str],
    log_path: Path,
    repo_root: Path,
    runtime: RuntimeSignals,
) -> tuple[RunningFold | None, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = timestamp()
    record: dict[str, Any] = {
        "attempt": attempt,
        "physical_gpu": gpu,
        "status": "running",
        "pid": None,
        "started_at": started_at,
        "ended_at": None,
        "return_code": None,
        "log": str(log_path),
        "command": command,
    }
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    write_log_header(log_handle, f"fold {fold}", command, gpu, attempt)
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=child_environment(gpu),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
    except OSError as exc:
        ended_at = timestamp()
        log_handle.write(f"[{ended_at}] LAUNCH_ERROR {exc}\n")
        log_handle.close()
        record.update(
            {
                "status": "failed",
                "ended_at": ended_at,
                "return_code": 127,
                "launch_error": str(exc),
            }
        )
        return None, record
    record["pid"] = process.pid
    runtime.register(fold, process)
    job = RunningFold(
        fold=fold,
        gpu=gpu,
        attempt=attempt,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        command=command,
        started_at=started_at,
    )
    print(
        f"[multigpu] launched fold={fold} gpu={gpu} attempt={attempt} "
        f"pid={process.pid} log={log_path}",
        flush=True,
    )
    return job, record


def close_fold_job(job: RunningFold, return_code: int) -> None:
    ended_at = timestamp()
    job.log_handle.write(
        f"[{ended_at}] END fold={job.fold} physical_gpu={job.gpu} "
        f"attempt={job.attempt} return_code={return_code}\n"
    )
    job.log_handle.close()


def refresh_status_lists(
    state: dict[str, Any],
    pending: deque[str],
    running: dict[str, RunningFold],
) -> None:
    state["updated_at"] = timestamp()
    state["pending_folds"] = list(pending)
    state["running_folds"] = {
        fold: {
            "physical_gpu": job.gpu,
            "attempt": job.attempt,
            "pid": job.process.pid,
            "started_at": job.started_at,
            "log": str(job.log_path),
        }
        for fold, job in running.items()
    }
    state["succeeded_folds"] = [
        fold
        for fold in state["work_folds"]
        if state["folds"][fold]["status"] == "succeeded"
    ]
    state["failed_folds"] = [
        fold
        for fold in state["work_folds"]
        if state["folds"][fold]["status"] == "failed"
    ]


def persist_status(
    state: dict[str, Any],
    status_path: Path,
    pending: deque[str],
    running: dict[str, RunningFold],
) -> None:
    refresh_status_lists(state, pending, running)
    atomic_json_dump(state, status_path)


def dry_run_report(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    runner: Path,
    auditor: Path,
    data_dir: Path,
    output_dir: Path,
    log_dir: Path,
    gpus: Sequence[str],
    folds: Sequence[str],
    forwarded: Sequence[str],
) -> None:
    finalize = build_finalize_command(
        args.python, runner, data_dir, output_dir, forwarded
    )
    print(f"scheduler_version={SCHEDULER_VERSION}")
    print(f"repo_root={repo_root}")
    print(f"data_dir={data_dir}")
    print(f"output_dir={output_dir}")
    print(f"log_dir={log_dir}")
    print(f"gpus={','.join(gpus)}")
    print(f"work_folds={','.join(folds)}")
    print(f"max_parallel={min(len(gpus), len(folds))}")
    print(f"forwarded={format_command(forwarded) if forwarded else '(none)'}")
    print(f"initialize={format_command(finalize)}")
    for index, fold in enumerate(folds):
        gpu = gpus[index] if index < len(gpus) else "<first-free>"
        command = build_worker_command(
            args.python,
            runner,
            data_dir,
            output_dir,
            fold,
            forwarded,
        )
        print(
            f"worker[{fold}].env.CUDA_VISIBLE_DEVICES={gpu}\n"
            f"worker[{fold}].log={log_dir / f'{fold}.log'}\n"
            f"worker[{fold}].command={format_command(command)}"
        )
    print(f"finalize={format_command(finalize)}")
    if args.audit:
        audit = build_audit_command(
            args.python,
            auditor,
            output_dir,
            allow_partial=set(folds) != set(CANONICAL_FOLDS),
        )
        print(f"audit={format_command(audit)}")
    else:
        print("audit=(disabled)")


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / RUNNER_FILENAME
    auditor = repo_root / "scripts" / AUDITOR_FILENAME
    parser = build_parser(repo_root)
    args, forwarded_raw = parser.parse_known_args(argv)
    forwarded = strip_forward_separator(list(forwarded_raw))
    try:
        validate_forwarded(forwarded)
        gpus = parse_gpu_spec(args.gpus)
        work_folds = parse_work_folds(args.work_folds)
        if args.max_retries < 0:
            raise ValueError("--max-retries must be non-negative")
        if args.launch_delay < 0:
            raise ValueError("--launch-delay must be non-negative")
    except ValueError as exc:
        parser.error(str(exc))

    data_dir = resolve_path(args.data_dir, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    log_dir = (
        resolve_path(args.log_dir, repo_root)
        if args.log_dir is not None
        else output_dir / "multigpu_logs"
    )

    if args.dry_run:
        dry_run_report(
            args=args,
            repo_root=repo_root,
            runner=runner,
            auditor=auditor,
            data_dir=data_dir,
            output_dir=output_dir,
            log_dir=log_dir,
            gpus=gpus,
            folds=work_folds,
            forwarded=forwarded,
        )
        return 0

    if not runner.is_file():
        parser.error(f"Core suite runner is missing: {runner}")
    if args.audit and not auditor.is_file():
        parser.error(f"Audit script is missing: {auditor}")
    for required_name in ("manifest.csv", "schema.json"):
        required = data_dir / required_name
        if not required.is_file():
            parser.error(f"Processed dataset file is missing: {required}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    lock = SchedulerLock(
        output_dir / ".multigpu_scheduler.lock",
        output_dir,
        run_id,
    )
    try:
        lock.acquire()
    except RuntimeError as exc:
        print(f"[multigpu] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    runtime = RuntimeSignals()
    previous_handlers = install_signal_handlers(runtime)
    status_path = output_dir / "multigpu_status.json"
    pending: deque[str] = deque(work_folds)
    available: deque[str] = deque(gpus)
    running: dict[str, RunningFold] = {}
    state: dict[str, Any] = {
        "format_version": 1,
        "scheduler_version": SCHEDULER_VERSION,
        "run_id": run_id,
        "status": "initializing",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "ended_at": None,
        "hostname": socket.gethostname(),
        "scheduler_pid": os.getpid(),
        "repo_root": str(repo_root),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "python": str(args.python),
        "gpus": list(gpus),
        "work_folds": list(work_folds),
        "configured_core_folds": "all",
        "max_retries": args.max_retries,
        "max_attempts": args.max_retries + 1,
        "launch_delay_seconds": args.launch_delay,
        "audit_enabled": bool(args.audit),
        "audit_allow_partial": set(work_folds) != set(CANONICAL_FOLDS),
        "forwarded_scientific_args": list(forwarded),
        "launching_stage": None,
        "active_stage": None,
        "initialization": {"status": "pending"},
        "finalization": {"status": "pending"},
        "audit": {"status": "pending" if args.audit else "disabled"},
        "folds": {
            fold: {"status": "pending", "attempts": []}
            for fold in work_folds
        },
        "pending_folds": list(work_folds),
        "launching_folds": {},
        "running_folds": {},
        "succeeded_folds": [],
        "failed_folds": [],
    }

    def record_stage_started(
        status_key: str,
        label: str,
        process: subprocess.Popen[Any],
        record: dict[str, Any],
    ) -> None:
        state[status_key] = record
        state["launching_stage"] = None
        state["active_stage"] = {
            "label": label,
            "pid": process.pid,
            "started_at": record["started_at"],
            "log": record["log"],
        }
        persist_status(state, status_path, pending, running)

    def mark_interrupted() -> int:
        state["status"] = "interrupted"
        state["ended_at"] = timestamp()
        state["received_signal"] = runtime.signal_number
        for fold, job in list(running.items()):
            return_code = job.process.poll()
            if return_code is None:
                return_code = -(runtime.signal_number or int(signal.SIGTERM))
            close_fold_job(job, int(return_code))
            runtime.unregister(fold)
            attempt_record = state["folds"][fold]["attempts"][-1]
            attempt_record.update(
                {
                    "status": "interrupted",
                    "ended_at": timestamp(),
                    "return_code": int(return_code),
                }
            )
            state["folds"][fold]["status"] = "interrupted"
        running.clear()
        persist_status(state, status_path, pending, running)
        return 128 + int(runtime.signal_number or signal.SIGTERM)

    try:
        if not args.skip_gpu_check:
            check_gpus(gpus)
        else:
            print("[multigpu] GPU availability check skipped", flush=True)
        persist_status(state, status_path, pending, running)

        # Serialize protocol initialization before workers. This creates or
        # validates config.json/run_manifest.json and pending root summaries,
        # eliminating root metadata races among parallel worker processes.
        initialize_command = build_finalize_command(
            args.python, runner, data_dir, output_dir, forwarded
        )
        state["launching_stage"] = {
            "label": "initialize_protocol",
            "recorded_at": timestamp(),
        }
        persist_status(state, status_path, pending, running)
        initialize_code, initialize_record = run_stage(
            label="initialize_protocol",
            command=initialize_command,
            log_path=log_dir / "initialize.log",
            repo_root=repo_root,
            runtime=runtime,
            on_started=lambda process, record: record_stage_started(
                "initialization",
                "initialize_protocol",
                process,
                record,
            ),
        )
        state["launching_stage"] = None
        state["active_stage"] = None
        state["initialization"] = initialize_record
        persist_status(state, status_path, pending, running)
        if runtime.stop_requested:
            shutdown_active_processes(runtime)
            return mark_interrupted()
        if initialize_code != 0:
            state["status"] = "initialization_failed"
            state["ended_at"] = timestamp()
            persist_status(state, status_path, pending, running)
            return 1

        state["status"] = "running"
        persist_status(state, status_path, pending, running)

        while pending or running:
            if runtime.stop_requested:
                shutdown_active_processes(runtime)
                return mark_interrupted()

            while pending and available and not runtime.stop_requested:
                fold = pending.popleft()
                gpu = available.popleft()
                fold_state = state["folds"][fold]
                attempt = len(fold_state["attempts"]) + 1
                fold_state["status"] = "launching"
                state["launching_folds"][fold] = {
                    "physical_gpu": gpu,
                    "attempt": attempt,
                    "recorded_at": timestamp(),
                }
                persist_status(state, status_path, pending, running)
                command = build_worker_command(
                    args.python,
                    runner,
                    data_dir,
                    output_dir,
                    fold,
                    forwarded,
                )
                job, attempt_record = launch_fold(
                    fold=fold,
                    gpu=gpu,
                    attempt=attempt,
                    command=command,
                    log_path=log_dir / f"{fold}.log",
                    repo_root=repo_root,
                    runtime=runtime,
                )
                del state["launching_folds"][fold]
                fold_state["attempts"].append(attempt_record)
                if job is None:
                    available.append(gpu)
                    if attempt <= args.max_retries:
                        fold_state["status"] = "pending"
                        pending.append(fold)
                        print(
                            f"[multigpu] launch failed fold={fold}; "
                            f"queued retry {attempt + 1}/{args.max_retries + 1}",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        fold_state["status"] = "failed"
                        print(
                            f"[multigpu] fold={fold} exhausted "
                            f"{args.max_retries + 1} launch attempt(s)",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    fold_state["status"] = "running"
                    running[fold] = job
                persist_status(state, status_path, pending, running)
                if pending and available and args.launch_delay > 0:
                    interruptible_delay(args.launch_delay, runtime)

            if runtime.stop_requested:
                continue
            completed = [
                (fold, job, job.process.poll())
                for fold, job in running.items()
                if job.process.poll() is not None
            ]
            if not completed:
                time.sleep(0.5)
                continue

            for fold, job, raw_return_code in completed:
                return_code = int(raw_return_code)
                close_fold_job(job, return_code)
                runtime.unregister(fold)
                del running[fold]
                available.append(job.gpu)
                fold_state = state["folds"][fold]
                attempt_record = fold_state["attempts"][-1]
                attempt_record.update(
                    {
                        "status": (
                            "succeeded" if return_code == 0 else "failed"
                        ),
                        "ended_at": timestamp(),
                        "return_code": return_code,
                    }
                )
                if return_code == 0:
                    fold_state["status"] = "succeeded"
                    print(
                        f"[multigpu] completed fold={fold} gpu={job.gpu} "
                        f"attempt={job.attempt}",
                        flush=True,
                    )
                elif job.attempt <= args.max_retries:
                    fold_state["status"] = "pending"
                    pending.append(fold)
                    print(
                        f"[multigpu] failed fold={fold} gpu={job.gpu} "
                        f"return_code={return_code}; queued retry "
                        f"{job.attempt + 1}/{args.max_retries + 1}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    fold_state["status"] = "failed"
                    print(
                        f"[multigpu] fold={fold} exhausted "
                        f"{args.max_retries + 1} attempt(s); "
                        f"last_return_code={return_code}",
                        file=sys.stderr,
                        flush=True,
                    )
            persist_status(state, status_path, pending, running)

        failed_folds = [
            fold
            for fold in work_folds
            if state["folds"][fold]["status"] != "succeeded"
        ]
        if failed_folds:
            state["status"] = "workers_failed"
            state["ended_at"] = timestamp()
            state["failure_message"] = (
                "One or more folds exhausted retries; rerun the same command "
                "to resume their checkpoints"
            )
            persist_status(state, status_path, pending, running)
            return 1

        # Workers intentionally avoid touching root summaries. Rebuild them
        # once, after every selected fold has completed successfully.
        state["status"] = "finalizing"
        persist_status(state, status_path, pending, running)
        finalize_command = build_finalize_command(
            args.python, runner, data_dir, output_dir, forwarded
        )
        state["launching_stage"] = {
            "label": "finalize_results",
            "recorded_at": timestamp(),
        }
        persist_status(state, status_path, pending, running)
        finalize_code, finalize_record = run_stage(
            label="finalize_results",
            command=finalize_command,
            log_path=log_dir / "finalize.log",
            repo_root=repo_root,
            runtime=runtime,
            on_started=lambda process, record: record_stage_started(
                "finalization",
                "finalize_results",
                process,
                record,
            ),
        )
        state["launching_stage"] = None
        state["active_stage"] = None
        state["finalization"] = finalize_record
        persist_status(state, status_path, pending, running)
        if runtime.stop_requested:
            shutdown_active_processes(runtime)
            return mark_interrupted()
        if finalize_code != 0:
            state["status"] = "finalization_failed"
            state["ended_at"] = timestamp()
            persist_status(state, status_path, pending, running)
            return 1

        if args.audit:
            state["status"] = "auditing"
            persist_status(state, status_path, pending, running)
            audit_command = build_audit_command(
                args.python,
                auditor,
                output_dir,
                allow_partial=set(work_folds) != set(CANONICAL_FOLDS),
            )
            state["launching_stage"] = {
                "label": "audit",
                "recorded_at": timestamp(),
            }
            persist_status(state, status_path, pending, running)
            audit_code, audit_record = run_stage(
                label="audit",
                command=audit_command,
                log_path=log_dir / "audit.log",
                repo_root=repo_root,
                runtime=runtime,
                on_started=lambda process, record: record_stage_started(
                    "audit",
                    "audit",
                    process,
                    record,
                ),
            )
            state["launching_stage"] = None
            state["active_stage"] = None
            state["audit"] = audit_record
            persist_status(state, status_path, pending, running)
            if runtime.stop_requested:
                shutdown_active_processes(runtime)
                return mark_interrupted()
            if audit_code != 0:
                state["status"] = "audit_failed"
                state["ended_at"] = timestamp()
                persist_status(state, status_path, pending, running)
                return 1

        state["status"] = "complete" if args.audit else "complete_unverified"
        state["ended_at"] = timestamp()
        persist_status(state, status_path, pending, running)
        print(
            f"[multigpu] COMPLETE folds={work_folds} status={status_path}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        shutdown_active_processes(runtime)
        for fold, job in list(running.items()):
            return_code = job.process.poll()
            if return_code is None:
                return_code = -int(signal.SIGTERM)
            close_fold_job(job, int(return_code))
            runtime.unregister(fold)
            state["folds"][fold]["status"] = "interrupted"
            attempt_record = state["folds"][fold]["attempts"][-1]
            attempt_record.update(
                {
                    "status": "interrupted",
                    "ended_at": timestamp(),
                    "return_code": int(return_code),
                }
            )
        running.clear()
        state["status"] = "scheduler_error"
        state["ended_at"] = timestamp()
        state["scheduler_error"] = f"{type(exc).__name__}: {exc}"
        try:
            persist_status(state, status_path, pending, running)
        except Exception as status_exc:
            print(
                f"[multigpu] additionally failed to update status: {status_exc}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[multigpu] ERROR {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if runtime.active:
            shutdown_active_processes(runtime, grace_seconds=2.0)
        restore_signal_handlers(previous_handlers)
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
