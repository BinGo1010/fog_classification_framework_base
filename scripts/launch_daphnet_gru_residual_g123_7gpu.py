#!/usr/bin/env python3
"""Run frozen GRU-v1 NBM residual G1/G2/G3 on a seven-GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKER = REPO_ROOT / "scripts" / "run_daphnet_gru_residual_g123.py"
SOURCE_EXPERIMENT = (
    REPO_ROOT
    / "outputs"
    / "daphnet_gru_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161"
    / "nbm_source"
)
GROUPS = ("G1", "G2", "G3")
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161)

from scripts.launch_daphnet_residual_calibration_abcd_7gpu import (
    command_text,
    run_pool,
    visible_gpu_count,
)


def parse_seed_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique seed list: {value}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument("--nbm-source-root", type=Path, default=SOURCE_EXPERIMENT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_gru_nbm300_residual_G1_G2_G3_tcn_ep10pat2_seedset_0_52_161",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--tcn-seeds", default="0,52,161")
    parser.add_argument("--tcn-max-epochs", type=int, default=10)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase", choices=("full", "train", "evaluate", "aggregate"), default="full"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = parse_seed_list(args.tcn_seeds)
    if seeds != SEEDS:
        raise ValueError(f"this experiment requires exact seeds {SEEDS}")
    if args.tcn_max_epochs != 10 or args.tcn_patience != 2:
        raise ValueError("this experiment requires TCN max_epoch=10 and patience=2")
    return seeds


def validate_gpus(value: str, check_hardware: bool) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not item.isdigit() for item in ids):
        raise ValueError(f"invalid unique GPU ids: {value}")
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in ids):
            raise ValueError(f"requested GPU ids {ids}, but nvidia-smi reports {count} GPUs")
    return ids


def validate_sources(root: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in FOLDS:
            directory = root.resolve() / f"seed_{seed}" / f"fold_{fold}"
            checkpoint = directory / "checkpoints" / "gru_nbm_best.pt"
            frozen = directory / "nbm_frozen.json"
            done = directory / "DONE_NBM.json"
            for path in (checkpoint, frozen, done):
                if not path.is_file():
                    raise FileNotFoundError(f"frozen paired GRU-NBM artifact missing: {path}")
            payload = json.loads(frozen.read_text(encoding="utf-8"))
            architecture = payload["training"]["architecture"]
            if (
                int(payload["training"]["seed"]) != seed
                or architecture.get("name") != "gru_reconstruction_nbm_v1"
                or int(architecture.get("parameter_count", -1)) != 31_513
            ):
                raise AssertionError(f"frozen GRU-NBM contract mismatch: {directory}")
            artifacts.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "checkpoint": str(checkpoint.resolve()),
                    "calibration": str(frozen.resolve()),
                }
            )
    return artifacts


def common_args(args: argparse.Namespace, source: Path) -> list[str]:
    values = [
        "--data-dir", str(args.data_dir.resolve()),
        "--nbm-source-root", str(source.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--groups", ",".join(GROUPS),
        "--tcn-seeds", args.tcn_seeds,
        "--tcn-max-epochs", str(args.tcn_max_epochs),
        "--tcn-patience", str(args.tcn_patience),
        "--num-workers", str(args.num_workers),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def job_command(
    args: argparse.Namespace, stage: str, fold: int, group: str, seed: int
) -> list[str]:
    source = args.nbm_source_root.resolve() / f"seed_{seed}"
    return [
        args.python, str(WORKER), "--stage", stage,
        *common_args(args, source),
        "--fold", str(fold), "--group", group,
        "--tcn-seed", str(seed), "--device", "cuda",
    ]


def singleton(args: argparse.Namespace, stage: str) -> list[str]:
    return [
        args.python, str(WORKER), "--stage", stage,
        *common_args(args, args.nbm_source_root.resolve()),
    ]


def main() -> None:
    args = parse_args()
    seeds = validate_contract(args)
    gpu_ids = validate_gpus(args.gpu_ids, not args.dry_run)
    artifacts = validate_sources(args.nbm_source_root.resolve(), seeds)
    specifications = [
        (fold, group, seed)
        for fold in FOLDS
        for group in GROUPS
        for seed in seeds
    ]
    train_jobs = [
        {
            "id": f"fold{fold}_{group}_seed{seed}",
            "command": job_command(args, "train", fold, group, seed),
        }
        for fold, group, seed in specifications
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_{group}_seed{seed}",
            "command": job_command(args, "evaluate", fold, group, seed),
        }
        for fold, group, seed in specifications
    ]
    plan = {
        "strategy": "7-GPU dynamic queue with strict global test barrier",
        "gpu_ids": gpu_ids,
        "folds": list(FOLDS),
        "groups": list(GROUPS),
        "paired_nbm_tcn_seeds": list(seeds),
        "nbm_retrained": False,
        "frozen_nbm_artifacts": artifacts,
        "training_jobs": len(train_jobs),
        "evaluation_jobs_after_barrier": len(evaluate_jobs),
        "maximum_concurrent_jobs": len(gpu_ids),
        "input": "all groups use [r,abs(r),delta(r)] [B,27,128]",
        "G1": "clip((e-b)/sigma), then residual window-axis centering",
        "G2": "clip((e-b)/sigma), no residual second centering",
        "G3": "asinh((e-b)/sigma), no hard clip, no residual second centering",
        "classifier": "same RepresentationTCNM; AdamW lr1e-3; weighted BCE; max10/pat2",
        "example_train": command_text(train_jobs[0]["command"]),
        "seal": command_text(singleton(args, "seal")),
        "example_test": command_text(evaluate_jobs[0]["command"]),
        "aggregate": command_text(singleton(args, "aggregate")),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not (args.data_dir.resolve() / "nbm_protocol.json").is_file():
        raise FileNotFoundError(f"processed_NBM protocol missing: {args.data_dir.resolve()}")

    root = args.output_root.resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    if args.phase in ("full", "train"):
        run_pool("train", train_jobs, gpu_ids, root)
        subprocess.run(singleton(args, "seal"), cwd=REPO_ROOT, env=environment, check=True)
        if args.phase == "train":
            print(f"TRAINING SEALED output={root}", flush=True)
            return
    if args.phase in ("full", "evaluate"):
        if not (root / "TRAINING_BARRIER.json").is_file():
            raise FileNotFoundError("evaluation requires TRAINING_BARRIER.json")
        run_pool("evaluate", evaluate_jobs, gpu_ids, root)
        subprocess.run(singleton(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True)
    elif args.phase == "aggregate":
        subprocess.run(singleton(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True)
    print(f"COMPLETE phase={args.phase} output={root}", flush=True)


if __name__ == "__main__":
    main()
