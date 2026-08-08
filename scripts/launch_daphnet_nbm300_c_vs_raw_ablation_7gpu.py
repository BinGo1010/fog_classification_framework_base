#!/usr/bin/env python3
"""Seven-GPU launcher for the strict FULL-C versus centered-RAW ablation."""

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
NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_conv_tcn_nbm_gaussian200_fold.py"
PAIR_WORKER = REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py"
FOLDS = (0, 1, 2)
METHODS = ("FULL_C", "RAW")

# Reuse the tested dynamic one-process-per-GPU scheduler.
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
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_conv_tcn_nbm300_C_vs_raw_tcn_ep10pat2_3seed_seed20260807",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--tcn-seeds", default="20260807,20260808,20260809")
    parser.add_argument("--nbm-seed", type=int, default=20260807)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=10)
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


def nbm_command(args: argparse.Namespace, fold: int) -> list[str]:
    command = [
        args.python, str(NBM_WORKER),
        "--data-dir", str(args.data_dir.resolve()),
        "--output-root", str((args.output_root.resolve() / "nbm_source")),
        "--fold", str(fold), "--device", "cuda",
        "--seed", str(args.nbm_seed), "--num-workers", str(args.num_workers),
        "--nbm-max-epochs", str(args.nbm_max_epochs),
        "--nbm-patience", str(args.nbm_patience),
        "--nbm-learning-rate", "0.001",
        "--clean-probability", "0.40",
        "--gaussian-probability", "0.40",
        "--mask-probability", "0.20",
        "--gaussian-std", "0.04",
        "--mask-min-samples", "4", "--mask-max-samples", "8",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def common_pair_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--data-dir", str(args.data_dir.resolve()),
        "--nbm-source-root", str((args.output_root.resolve() / "nbm_source")),
        "--output-root", str(args.output_root.resolve()),
        "--tcn-seeds", args.tcn_seeds,
        "--num-workers", str(args.num_workers),
        "--tcn-max-epochs", str(args.tcn_max_epochs),
        "--tcn-patience", str(args.tcn_patience),
        "--required-nbm-max-epochs", str(args.nbm_max_epochs),
        "--required-nbm-patience", str(args.nbm_patience),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def pair_command(
    args: argparse.Namespace, stage: str, fold: int, method: str, seed: int
) -> list[str]:
    return [
        args.python, str(PAIR_WORKER), "--stage", stage,
        *common_pair_args(args), "--fold", str(fold), "--method", method,
        "--tcn-seed", str(seed), "--device", "cuda",
    ]


def singleton_command(args: argparse.Namespace, stage: str) -> list[str]:
    return [args.python, str(PAIR_WORKER), "--stage", stage, *common_pair_args(args)]


def validate_gpu_ids(value: str, check_hardware: bool) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not item.isdigit() for item in ids):
        raise ValueError(f"invalid unique GPU ids: {value}")
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in ids):
            raise ValueError(f"requested {ids}, but nvidia-smi reports {count} GPUs")
    return ids


def validate_fixed_contract(args: argparse.Namespace) -> None:
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this experiment requires NBM max_epoch=300 and patience=20")
    if args.tcn_max_epochs != 10 or args.tcn_patience != 2:
        raise ValueError("this experiment requires TCN max_epoch=10 and patience=2")
    if len(parse_seed_list(args.tcn_seeds)) != 3:
        raise ValueError("strict comparison requires exactly three TCN seeds")


def main() -> None:
    args = parse_args()
    validate_fixed_contract(args)
    seeds = parse_seed_list(args.tcn_seeds)
    gpu_ids = validate_gpu_ids(args.gpu_ids, check_hardware=not args.dry_run)
    root = args.output_root.resolve()
    nbm_jobs = [{
        "id": f"fold{fold}_nbm300", "command": nbm_command(args, fold)
    } for fold in FOLDS]
    specs = [(fold, method, seed) for fold in FOLDS for method in METHODS for seed in seeds]
    train_jobs = [{
        "id": f"fold{fold}_{method}_seed{seed}",
        "command": pair_command(args, "train", fold, method, seed),
    } for fold, method, seed in specs]
    evaluate_jobs = [{
        "id": f"fold{fold}_{method}_seed{seed}",
        "command": pair_command(args, "evaluate", fold, method, seed),
    } for fold, method, seed in specs]
    plan = {
        "strategy": "7-GPU dynamic queue; NBM barrier then 18-train global test barrier",
        "gpu_ids": gpu_ids,
        "folds": list(FOLDS), "methods": list(METHODS), "tcn_seeds": list(seeds),
        "nbm_jobs": len(nbm_jobs), "classifier_train_jobs": len(train_jobs),
        "post_barrier_test_jobs": len(evaluate_jobs),
        "nbm_contract": {
            "max_epochs": 300, "patience": 20, "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
            "augmentation": "40% clean + 40% Gaussian(std=0.04) + 20% light time mask",
        },
        "tcn_contract": {
            "max_epochs": 10, "patience": 2,
            "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
            "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        },
        "full_input": "scheme C [r,abs(r),delta(r)] [B,27,128]",
        "raw_input": "role4 RobustScaler + per-window/per-axis centering [B,9,128]",
        "example_nbm": command_text(nbm_jobs[0]["command"]),
        "example_train": command_text(train_jobs[0]["command"]),
        "seal": command_text(singleton_command(args, "seal")),
        "example_test": command_text(evaluate_jobs[0]["command"]),
        "aggregate": command_text(singleton_command(args, "aggregate")),
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
    environment["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")

    if args.phase in ("full", "nbm"):
        run_pool("nbm", nbm_jobs, gpu_ids, root)
        if args.phase == "nbm":
            print(f"NBM COMPLETE output={root / 'nbm_source'}", flush=True)
            return
    if args.phase in ("full", "train"):
        for fold in FOLDS:
            required = root / "nbm_source" / f"fold_{fold}" / "DONE_NBM.json"
            if not required.exists():
                raise FileNotFoundError(f"NBM fold not frozen: {required}")
        run_pool("train", train_jobs, gpu_ids, root)
        subprocess.run(singleton_command(args, "seal"), cwd=REPO_ROOT, env=environment, check=True)
        if args.phase == "train":
            print(f"TRAINING SEALED output={root}", flush=True)
            return
    if args.phase in ("full", "evaluate"):
        if not (root / "TRAINING_BARRIER.json").exists():
            raise FileNotFoundError("evaluation requires TRAINING_BARRIER.json")
        run_pool("evaluate", evaluate_jobs, gpu_ids, root)
        subprocess.run(singleton_command(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True)
    elif args.phase == "aggregate":
        subprocess.run(singleton_command(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True)
    print(f"COMPLETE phase={args.phase} output={root}", flush=True)


if __name__ == "__main__":
    main()
