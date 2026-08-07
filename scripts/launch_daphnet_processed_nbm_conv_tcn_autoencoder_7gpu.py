#!/usr/bin/env python3
"""Launch three folds of the paired residual-representation experiment.

The processed_NBM protocol has three independent outer folds.  Each fold owns
one physical GPU through CUDA_VISIBLE_DEVICES.  Within a fold, one shared NBM
is trained and the r and [r,abs(r),delta(r)] classifiers run sequentially.
Therefore a seven-GPU server uses three cards for the default one-seed
experiment without changing batch semantics. Unknown CLI arguments are
forwarded to every fold worker.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def visible_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def worker_command(
    args: argparse.Namespace,
    fold: int,
    forwarded: list[str],
) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--fold",
        str(fold),
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        *forwarded,
    ]


def aggregate_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--output-root",
        str(args.output_root.resolve()),
        "--aggregate-only",
    ]


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def main() -> None:
    args, forwarded = parse_args()
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) < 3:
        raise ValueError("at least three GPU ids are required for three parallel folds")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"duplicate GPU ids: {gpu_ids}")
    for value in gpu_ids:
        if not value.isdigit():
            raise ValueError(f"GPU ids must be non-negative integers: {value}")
    commands = [worker_command(args, fold, forwarded) for fold in (0, 1, 2)]
    plan = {
        "strategy": "one authoritative outer fold per physical GPU; two paired classifiers per fold",
        "representations": ["r", "r_abs_delta"],
        "nbm_training": "one shared NBM per fold",
        "available_gpu_ids": gpu_ids,
        "jobs": [
            {
                "fold": fold,
                "physical_gpu": gpu_ids[fold],
                "worker_visible_device": "cuda:0",
                "command": command_text(commands[fold]),
            }
            for fold in (0, 1, 2)
        ],
        "unused_gpu_ids": gpu_ids[3:],
        "reason_unused": "only three independent folds; no DDP to avoid changing batch and BatchNorm semantics",
        "aggregate_command": command_text(aggregate_command(args)),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not (args.data_dir / "nbm_protocol.json").exists():
        raise FileNotFoundError(f"processed_NBM protocol missing: {args.data_dir}")
    count = visible_gpu_count()
    for gpu in gpu_ids[:3]:
        if int(gpu) >= count:
            raise ValueError(f"requested GPU {gpu}, but nvidia-smi reports {count} GPUs")
    output_root = args.output_root.resolve()
    log_dir = output_root / "logs" / "parallel_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_env = os.environ.copy()
    old_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    running: list[tuple[int, str, subprocess.Popen[Any], Any, Any]] = []

    def terminate(_signum: int, _frame: Any) -> None:
        for _, _, process, _, _ in running:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, terminate)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, terminate)
    try:
        for fold, command in enumerate(commands):
            gpu = gpu_ids[fold]
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            stdout_path = log_dir / f"fold{fold}.out.log"
            stderr_path = log_dir / f"fold{fold}.err.log"
            stdout_handle = stdout_path.open("a", encoding="utf-8")
            stderr_handle = stderr_path.open("a", encoding="utf-8")
            options: dict[str, Any] = {
                "cwd": str(REPO_ROOT),
                "env": env,
                "stdout": stdout_handle,
                "stderr": stderr_handle,
            }
            if os.name == "nt":
                options["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(command, **options)
            running.append((fold, gpu, process, stdout_handle, stderr_handle))
            print(f"LAUNCH fold={fold} physical_gpu={gpu} pid={process.pid}", flush=True)
        failures = []
        for fold, gpu, process, stdout_handle, stderr_handle in running:
            code = process.wait()
            stdout_handle.close()
            stderr_handle.close()
            print(f"EXIT fold={fold} physical_gpu={gpu} code={code}", flush=True)
            if code != 0:
                failures.append(fold)
        if failures:
            raise RuntimeError(f"fold workers failed: {failures}; inspect {log_dir}")
    finally:
        for _, _, process, stdout_handle, stderr_handle in running:
            if process.poll() is None:
                process.terminate()
            if not stdout_handle.closed:
                stdout_handle.close()
            if not stderr_handle.closed:
                stderr_handle.close()
    subprocess.run(aggregate_command(args), cwd=REPO_ROOT, env=base_env, check=True)
    print(f"COMPLETE {output_root}", flush=True)


if __name__ == "__main__":
    main()
