#!/usr/bin/env python3
"""Run the strict A-D residual-calibration experiment on a GPU job pool.

The default experiment has 3 folds x 4 groups x 3 TCN seeds = 36 training
jobs and 36 post-barrier evaluation jobs.  Up to seven jobs run concurrently,
one process per physical GPU.  The launcher never retrains the frozen NBMs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "run_daphnet_residual_calibration_abcd.py"
GROUPS = ("A", "B", "C", "D")
FOLDS = (0, 1, 2)


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"invalid unique seed list: {value}")
    return seeds


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument(
        "--nbm-source-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_residual_calibration_ABCD_3seed_seed20260807",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--tcn-seeds", default="20260807,20260808,20260809")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "train", "evaluate", "aggregate"),
        default="full",
        help="full=train+seal+evaluate+aggregate; evaluate also aggregates.",
    )
    parser.add_argument("--overwrite", action="store_true")
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


def common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--nbm-source-root",
        str(args.nbm_source_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--tcn-seeds",
        args.tcn_seeds,
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def job_command(
    args: argparse.Namespace,
    stage: str,
    fold: int,
    group: str,
    seed: int,
    forwarded: list[str],
) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--stage",
        stage,
        *common_args(args),
        "--fold",
        str(fold),
        "--group",
        group,
        "--tcn-seed",
        str(seed),
        "--device",
        "cuda",
        *forwarded,
    ]


def singleton_command(
    args: argparse.Namespace,
    stage: str,
    forwarded: list[str],
) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--stage",
        stage,
        *common_args(args),
        *forwarded,
    ]


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def validate_nbm_source(root: Path) -> list[dict[str, str]]:
    artifacts = []
    for fold in FOLDS:
        checkpoint = root / f"fold_{fold}" / "checkpoints" / "conv_tcn_nbm_best.pt"
        frozen = root / f"fold_{fold}" / "nbm_frozen.json"
        if not checkpoint.exists() or not frozen.exists():
            raise FileNotFoundError(f"frozen fold-{fold} NBM missing under {root}")
        artifacts.append(
            {
                "fold": str(fold),
                "checkpoint": str(checkpoint.resolve()),
                "calibration": str(frozen.resolve()),
            }
        )
    return artifacts


def run_pool(
    stage: str,
    jobs: list[dict[str, Any]],
    gpu_ids: list[str],
    output_root: Path,
) -> None:
    log_dir = output_root / "logs" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    available = list(gpu_ids)
    running: list[dict[str, Any]] = []
    stop_requested = False

    def terminate(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        for item in running:
            if item["process"].poll() is None:
                item["process"].terminate()

    previous_sigint = signal.signal(signal.SIGINT, terminate)
    previous_sigterm = None
    if hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    base_env = os.environ.copy()
    old_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    try:
        while pending or running:
            while pending and available and not stop_requested:
                job = pending.pop(0)
                gpu = available.pop(0)
                env = base_env.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                stdout_path = log_dir / f"{job['id']}.out.log"
                stderr_path = log_dir / f"{job['id']}.err.log"
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
                process = subprocess.Popen(job["command"], **options)
                running.append(
                    {
                        **job,
                        "gpu": gpu,
                        "process": process,
                        "stdout_handle": stdout_handle,
                        "stderr_handle": stderr_handle,
                    }
                )
                print(
                    f"LAUNCH stage={stage} job={job['id']} gpu={gpu} pid={process.pid}",
                    flush=True,
                )
            completed = []
            for item in running:
                code = item["process"].poll()
                if code is None:
                    continue
                item["stdout_handle"].close()
                item["stderr_handle"].close()
                available.append(item["gpu"])
                completed.append(item)
                print(
                    f"EXIT stage={stage} job={item['id']} gpu={item['gpu']} code={code}",
                    flush=True,
                )
                if code != 0:
                    for other in running:
                        if other is not item and other["process"].poll() is None:
                            other["process"].terminate()
                    raise RuntimeError(
                        f"{stage} job failed: {item['id']}; inspect {log_dir}"
                    )
            for item in completed:
                running.remove(item)
            if stop_requested:
                raise KeyboardInterrupt
            if not completed and running:
                time.sleep(0.25)
    finally:
        for item in running:
            if item["process"].poll() is None:
                item["process"].terminate()
            if not item["stdout_handle"].closed:
                item["stdout_handle"].close()
            if not item["stderr_handle"].closed:
                item["stderr_handle"].close()
        signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    args, forwarded = parse_args()
    seeds = parse_seed_list(args.tcn_seeds)
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"invalid unique --gpu-ids: {args.gpu_ids}")
    if any(not item.isdigit() for item in gpu_ids):
        raise ValueError("GPU ids must be non-negative integers")
    artifacts = validate_nbm_source(args.nbm_source_root.resolve())
    jobs_spec = [
        (fold, group, seed)
        for fold in FOLDS
        for group in GROUPS
        for seed in seeds
    ]
    train_jobs = [
        {
            "id": f"fold{fold}_group{group}_seed{seed}",
            "command": job_command(args, "train", fold, group, seed, forwarded),
        }
        for fold, group, seed in jobs_spec
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_group{group}_seed{seed}",
            "command": job_command(args, "evaluate", fold, group, seed, forwarded),
        }
        for fold, group, seed in jobs_spec
    ]
    plan = {
        "strategy": "dynamic one-process-per-GPU queue with global train/test barrier",
        "gpu_ids": gpu_ids,
        "folds": list(FOLDS),
        "groups": list(GROUPS),
        "tcn_seeds": list(seeds),
        "training_jobs": len(train_jobs),
        "evaluation_jobs_after_barrier": len(evaluate_jobs),
        "maximum_concurrent_jobs": len(gpu_ids),
        "nbm_retrained": False,
        "frozen_nbm_artifacts": artifacts,
        "example_train_command": command_text(train_jobs[0]["command"]),
        "seal_command": command_text(singleton_command(args, "seal", forwarded)),
        "example_evaluate_command": command_text(evaluate_jobs[0]["command"]),
        "aggregate_command": command_text(singleton_command(args, "aggregate", forwarded)),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not (args.data_dir / "nbm_protocol.json").exists():
        raise FileNotFoundError(f"processed_NBM protocol missing: {args.data_dir}")
    count = visible_gpu_count()
    for gpu in gpu_ids:
        if int(gpu) >= count:
            raise ValueError(f"requested GPU {gpu}, but nvidia-smi reports {count} GPUs")
    output_root = args.output_root.resolve()
    launch_dir = output_root / "logs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    (launch_dir / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_env = os.environ.copy()
    old_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )

    if args.phase in ("full", "train"):
        run_pool("train", train_jobs, gpu_ids, output_root)
        subprocess.run(
            singleton_command(args, "seal", forwarded),
            cwd=REPO_ROOT,
            env=base_env,
            check=True,
        )
    if args.phase in ("full", "evaluate"):
        if not (output_root / "TRAINING_BARRIER.json").exists():
            raise FileNotFoundError("evaluation requires TRAINING_BARRIER.json")
        run_pool("evaluate", evaluate_jobs, gpu_ids, output_root)
        subprocess.run(
            singleton_command(args, "aggregate", forwarded),
            cwd=REPO_ROOT,
            env=base_env,
            check=True,
        )
    elif args.phase == "aggregate":
        subprocess.run(
            singleton_command(args, "aggregate", forwarded),
            cwd=REPO_ROOT,
            env=base_env,
            check=True,
        )
    print(f"COMPLETE phase={args.phase} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
