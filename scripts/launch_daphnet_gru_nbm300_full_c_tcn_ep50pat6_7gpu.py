#!/usr/bin/env python3
"""Seven-GPU full GRU-NGM + scheme-C TCN50/pat6 experiment.

Frozen contract:
  * Daphnet processed_NBM, 64 Hz, 9 axes, 128-point windows, stride 64;
  * GRU BASE normal-gait model, Mask 4--8, max300/pat20;
  * scheme C [r, abs(r), delta(r)] -> unchanged 27-channel TCN;
  * TCN max_epoch=50, early-stopping patience=6;
  * three protocol folds and paired seeds 0,52,161,5216,52161;
  * all 15 classifiers and validation thresholds are sealed before test access.
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

from scripts import launch_daphnet_gru_nbm300_c_vs_raw_ep5pat2_7gpu as shared


PAIR_WORKER = REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py"
FOLDS = (0, 1, 2)
METHODS = ("FULL_C",)
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
SEED_TEXT = "0,52,161,5216,52161"


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
        / (
            "daphnet_gru_nbm300_FULL_C_tcn_ep50pat6_"
            "seedset_0_52_161_5216_52161"
        ),
    )
    parser.add_argument(
        "--reuse-nbm-source-root",
        type=Path,
        default=None,
        help=(
            "Optional existing nbm_source directory. If omitted, all 15 "
            "GRU-NGMs are trained again as part of the complete experiment."
        ),
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--nbm-seeds", default=SEED_TEXT)
    parser.add_argument("--tcn-seeds", default=SEED_TEXT)
    parser.add_argument("--experiment-methods", default="FULL_C")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=50)
    parser.add_argument("--tcn-patience", type=int, default=6)
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
    nbm_seeds = shared.parse_seed_list(args.nbm_seeds)
    tcn_seeds = shared.parse_seed_list(args.tcn_seeds)
    if nbm_seeds != REQUIRED_SEEDS or tcn_seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires paired seeds {SEED_TEXT}")
    if (args.nbm_max_epochs, args.nbm_patience) != (300, 20):
        raise ValueError("GRU-NGM must use max_epoch=300 and patience=20")
    if (args.tcn_max_epochs, args.tcn_patience) != (50, 6):
        raise ValueError("TCN must use max_epoch=50 and patience=6")
    methods = shared.validate_methods(args.experiment_methods)
    if methods != METHODS:
        raise ValueError("this experiment contains FULL_C only")
    return nbm_seeds


def nbm_source_base(args: argparse.Namespace) -> Path:
    if args.reuse_nbm_source_root is not None:
        return args.reuse_nbm_source_root.resolve()
    return args.output_root.resolve() / "nbm_source"


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
        "--experiment-methods",
        "FULL_C",
        "--sampling-rate-hz",
        "64",
        "--window-samples",
        "128",
        "--stride-samples",
        "64",
        "--num-workers",
        str(args.num_workers),
        "--tcn-max-epochs",
        "50",
        "--tcn-patience",
        "6",
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
    seed: int,
) -> list[str]:
    source = nbm_source_base(args) / f"seed_{seed}"
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *pair_common(args, source),
        "--fold",
        str(fold),
        "--method",
        "FULL_C",
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
        *pair_common(args, nbm_source_base(args)),
    ]


def main() -> None:
    args = parse_args()
    seeds = validate_contract(args)
    gpu_ids = shared.validate_gpus(args.gpu_ids, not args.dry_run)
    root = args.output_root.resolve()
    source_base = nbm_source_base(args)
    reuse_nbm = args.reuse_nbm_source_root is not None

    nbm_jobs = [
        {
            "id": f"fold{fold}_GRU_BASE_NGM_seed{seed}",
            "command": shared.nbm_command(args, fold, seed),
        }
        for fold in FOLDS
        for seed in seeds
    ]
    specs = [(fold, seed) for fold in FOLDS for seed in seeds]
    train_jobs = [
        {
            "id": f"fold{fold}_FULL_C_seed{seed}",
            "command": pair_command(args, "train", fold, seed),
        }
        for fold, seed in specs
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_FULL_C_seed{seed}",
            "command": pair_command(args, "evaluate", fold, seed),
        }
        for fold, seed in specs
    ]
    plan = {
        "experiment": "Daphnet_GRU_BASE_NGM300_FULL_C_TCN50pat6",
        "strategy": (
            f"7-GPU queue; {0 if reuse_nbm else len(nbm_jobs)} GRU-NGMs, "
            f"{len(train_jobs)} TCN classifiers, global test barrier, "
            f"{len(evaluate_jobs)} permanent-test jobs"
        ),
        "dataset": str(args.data_dir.resolve()),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "gpu_ids": gpu_ids,
        "nbm_backbone": (
            "GRU(9,64)->Linear(64,16)->Linear(16,64)->"
            "128-step zero-input GRU(9,64)->Linear(64,9)"
        ),
        "nbm_parameter_count": 31_513,
        "nbm_augmentation": (
            "40% clean, 40% Gaussian std0.04, 20% all-axis Mask4-8"
        ),
        "nbm_training": (
            "role4 fit, clean role5 early-stop/calibration, max300/pat20, "
            "SmoothL1, AdamW lr1e-3"
        ),
        "classifier_input": "scheme C [r,abs(r),delta(r)] [B,27,128]",
        "classifier_training": (
            "roles6/7 weighted BCE, roles2/3 PR-AUC checkpoint and BA threshold, "
            "max50/pat6, AdamW lr1e-3"
        ),
        "seeds": list(seeds),
        "seed_policy": "exact paired NBM/TCN seeds; no fold offset",
        "nbm_source_mode": "reuse_frozen" if reuse_nbm else "fit_new",
        "nbm_source_root": str(source_base),
        "nbm_jobs": 0 if reuse_nbm else len(nbm_jobs),
        "classifier_train_jobs": len(train_jobs),
        "post_barrier_test_jobs": len(evaluate_jobs),
        "example_nbm": (
            None if reuse_nbm else shared.command_text(nbm_jobs[0]["command"])
        ),
        "example_train": shared.command_text(train_jobs[0]["command"]),
        "seal": shared.command_text(singleton(args, "seal")),
        "example_test": shared.command_text(evaluate_jobs[0]["command"]),
        "aggregate": shared.command_text(singleton(args, "aggregate")),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    data_dir = args.data_dir.resolve()
    required = (
        "manifest.csv",
        "schema.json",
        "nbm_protocol.json",
        "nbm_quality_report.json",
    )
    missing = [str(data_dir / name) for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"processed_NBM identity files missing; check --data-dir: {missing}"
        )
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
        if reuse_nbm:
            for fold, seed in specs:
                done = source_base / f"seed_{seed}" / f"fold_{fold}" / "DONE_NBM.json"
                if not done.exists():
                    raise FileNotFoundError(f"reused GRU-NGM is not frozen: {done}")
            print(f"REUSE FROZEN GRU-NGM source={source_base}", flush=True)
        else:
            shared.run_pool("nbm", nbm_jobs, gpu_ids, root)
        if args.phase == "nbm":
            print(f"GRU-NGM READY source={source_base}", flush=True)
            return

    if args.phase in ("full", "train"):
        for fold, seed in specs:
            done = source_base / f"seed_{seed}" / f"fold_{fold}" / "DONE_NBM.json"
            if not done.exists():
                raise FileNotFoundError(f"GRU-NGM fold/seed not frozen: {done}")
        shared.run_pool("train", train_jobs, gpu_ids, root)
        subprocess.run(
            singleton(args, "seal"), cwd=REPO_ROOT, env=environment, check=True
        )
        if args.phase == "train":
            print(f"TRAINING SEALED output={root}", flush=True)
            return

    if args.phase in ("full", "evaluate"):
        if not (root / "TRAINING_BARRIER.json").exists():
            raise FileNotFoundError("evaluation requires TRAINING_BARRIER.json")
        shared.run_pool("evaluate", evaluate_jobs, gpu_ids, root)
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
