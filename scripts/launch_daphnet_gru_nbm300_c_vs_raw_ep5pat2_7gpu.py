#!/usr/bin/env python3
"""Seven-GPU launcher for 64-Hz GRU-NBM scheme-C versus centered RAW.

Experiment contract:
  * processed_NBM, 64 Hz, 128-sample windows, 64-sample stride;
  * GRU-NBM max_epoch=300, patience=20;
  * unchanged TCN-M, max_epoch=5, patience=2;
  * exact paired NBM/TCN seeds 0, 52, 161, 5216, 52161;
  * all 30 classifiers and validation thresholds freeze before any test access.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_gru_nbm300_fold.py"
PAIR_WORKER = REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py"
FOLDS = (0, 1, 2)
METHODS = ("FULL_C", "RAW")
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
SEED_TEXT = "0,52,161,5216,52161"

from scripts.launch_daphnet_residual_calibration_abcd_7gpu import (
    command_text,
    parse_seed_list,
    run_pool,
    visible_gpu_count,
)


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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--nbm-seeds", default=SEED_TEXT)
    parser.add_argument("--tcn-seeds", default=SEED_TEXT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "nbm", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> tuple[int, ...]:
    nbm_seeds = parse_seed_list(args.nbm_seeds)
    tcn_seeds = parse_seed_list(args.tcn_seeds)
    if nbm_seeds != REQUIRED_SEEDS or tcn_seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires paired seeds {SEED_TEXT}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("GRU-NBM must use max_epoch=300 and patience=20")
    if args.tcn_max_epochs != 5 or args.tcn_patience != 2:
        raise ValueError("TCN must use max_epoch=5 and patience=2")
    return nbm_seeds


def validate_gpus(value: str, check_hardware: bool) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not item.isdigit() for item in ids):
        raise ValueError(f"invalid unique GPU ids: {value}")
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in ids):
            raise ValueError(f"requested {ids}, but nvidia-smi reports {count} GPUs")
    return ids


def nbm_command(args: argparse.Namespace, fold: int, seed: int) -> list[str]:
    command = [
        args.python,
        str(NBM_WORKER),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve() / "nbm_source" / f"seed_{seed}"),
        "--fold",
        str(fold),
        "--seed",
        str(seed),
        "--required-seeds",
        SEED_TEXT,
        "--sampling-rate-hz",
        "64",
        "--window-samples",
        "128",
        "--stride-samples",
        "64",
        "--device",
        "cuda",
        "--num-workers",
        str(args.num_workers),
        "--nbm-max-epochs",
        "300",
        "--nbm-patience",
        "20",
        "--nbm-hidden",
        "64",
        "--nbm-bottleneck",
        "16",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def pair_common(args: argparse.Namespace, source: Path) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--nbm-source-root",
        str(source.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--nbm-kind",
        "gru",
        "--nbm-seeds",
        args.nbm_seeds,
        "--tcn-seeds",
        args.tcn_seeds,
        "--required-seeds",
        SEED_TEXT,
        "--sampling-rate-hz",
        "64",
        "--window-samples",
        "128",
        "--stride-samples",
        "64",
        "--num-workers",
        str(args.num_workers),
        "--tcn-max-epochs",
        "5",
        "--tcn-patience",
        "2",
        "--required-nbm-max-epochs",
        "300",
        "--required-nbm-patience",
        "20",
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def pair_command(
    args: argparse.Namespace,
    stage: str,
    fold: int,
    method: str,
    seed: int,
) -> list[str]:
    source = args.output_root.resolve() / "nbm_source" / f"seed_{seed}"
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *pair_common(args, source),
        "--fold",
        str(fold),
        "--method",
        method,
        "--nbm-seed",
        str(seed),
        "--tcn-seed",
        str(seed),
        "--device",
        "cuda",
    ]


def singleton(args: argparse.Namespace, stage: str) -> list[str]:
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *pair_common(args, args.output_root.resolve() / "nbm_source"),
    ]


def main() -> None:
    args = parse_args()
    seeds = validate_contract(args)
    gpu_ids = validate_gpus(args.gpu_ids, not args.dry_run)
    root = args.output_root.resolve()

    nbm_jobs = [
        {
            "id": f"fold{fold}_GRU_NBM64_seed{seed}",
            "command": nbm_command(args, fold, seed),
        }
        for fold in FOLDS
        for seed in seeds
    ]
    specs = [
        (fold, method, seed)
        for fold in FOLDS
        for method in METHODS
        for seed in seeds
    ]
    train_jobs = [
        {
            "id": f"fold{fold}_{method}_seed{seed}",
            "command": pair_command(args, "train", fold, method, seed),
        }
        for fold, method, seed in specs
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_{method}_seed{seed}",
            "command": pair_command(args, "evaluate", fold, method, seed),
        }
        for fold, method, seed in specs
    ]
    plan = {
        "strategy": "7-GPU queue; 15 GRU-NBMs then 30-classifier global test barrier",
        "dataset": str(args.data_dir.resolve()),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "gpu_ids": gpu_ids,
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "nbm_backbone": (
            "GRU(9,64)->Linear(64,16)->Linear(16,64)->"
            "128-step zero-input GRU(9,64)->Linear(64,9)"
        ),
        "nbm_seeds": list(seeds),
        "tcn_seeds": list(seeds),
        "seed_policy": "exact paired seeds; no fold offset",
        "nbm_jobs": len(nbm_jobs),
        "classifier_train_jobs": len(train_jobs),
        "post_barrier_test_jobs": len(evaluate_jobs),
        "nbm_training": "max300/pat20, SmoothL1, AdamW lr1e-3, 40/40/20 augmentation",
        "classifier_training": "max5/pat2, weighted BCE, AdamW lr1e-3",
        "full_input": "scheme C [r,abs(r),delta(r)] [B,27,128]",
        "raw_input": "role4 RobustScaler + per-window/per-axis centering [B,9,128]",
        "example_nbm": command_text(nbm_jobs[0]["command"]),
        "example_train": command_text(train_jobs[0]["command"]),
        "seal": command_text(singleton(args, "seal")),
        "example_test": command_text(evaluate_jobs[0]["command"]),
        "aggregate": command_text(singleton(args, "aggregate")),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not (args.data_dir.resolve() / "nbm_protocol.json").exists():
        raise FileNotFoundError(f"processed_NBM protocol missing: {args.data_dir.resolve()}")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )

    if args.phase in ("full", "nbm"):
        run_pool("nbm", nbm_jobs, gpu_ids, root)
        if args.phase == "nbm":
            print(f"64-Hz GRU-NBM COMPLETE output={root / 'nbm_source'}", flush=True)
            return
    if args.phase in ("full", "train"):
        for fold in FOLDS:
            for seed in seeds:
                done = (
                    root
                    / "nbm_source"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / "DONE_NBM.json"
                )
                if not done.exists():
                    raise FileNotFoundError(f"GRU-NBM fold/seed not frozen: {done}")
        run_pool("train", train_jobs, gpu_ids, root)
        subprocess.run(
            singleton(args, "seal"), cwd=REPO_ROOT, env=environment, check=True
        )
        if args.phase == "train":
            print(f"TRAINING SEALED output={root}", flush=True)
            return
    if args.phase in ("full", "evaluate"):
        if not (root / "TRAINING_BARRIER.json").exists():
            raise FileNotFoundError("evaluation requires TRAINING_BARRIER.json")
        run_pool("evaluate", evaluate_jobs, gpu_ids, root)
        subprocess.run(
            singleton(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True
        )
    elif args.phase == "aggregate":
        subprocess.run(
            singleton(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True
        )
    print(f"COMPLETE phase={args.phase} output={root}", flush=True)


if __name__ == "__main__":
    main()
