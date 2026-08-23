#!/usr/bin/env python3
"""Eight-GPU launcher for compact Transformer-NGM scheme-C TCN on Daphnet.

The experiment trains 3 folds x 5 paired seeds.  Each compact 48,208-parameter
Transformer-NGM is fit on role 4 and selected/calibrated on clean role 5.  The
unchanged 27-channel scheme-C TCN is fit on roles 6/7, selected and thresholded
on roles 2/3, and roles 0/1 are evaluated only after the global barrier.
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

from scripts.launch_daphnet_residual_calibration_abcd_7gpu import (
    command_text,
    parse_seed_list,
    run_pool,
    visible_gpu_count,
)


NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_transformer_ngm_48k_fold.py"
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
            "daphnet_processed_NBM_transformer_ngm48k_global_z16_C_tcn_"
            "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
        ),
    )
    parser.add_argument(
        "--reuse-nbm-source-root",
        type=Path,
        default=None,
        help="Optional already-frozen nbm_source directory; skips NGM fitting.",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default=SEED_TEXT)
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
    seeds = parse_seed_list(args.seeds)
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires exact seeds {SEED_TEXT}")
    if (args.nbm_max_epochs, args.nbm_patience) != (300, 20):
        raise ValueError("Transformer-NGM must use max_epoch=300, patience=20")
    if (args.tcn_max_epochs, args.tcn_patience) != (5, 2):
        raise ValueError("TCN must use max_epoch=5, patience=2")
    return seeds


def validate_gpus(value: str, check_hardware: bool) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not item.isdigit() for item in ids)
    ):
        raise ValueError(f"invalid unique GPU ids: {value}")
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in ids):
            raise ValueError(f"requested {ids}, but nvidia-smi reports {count} GPUs")
    return ids


def nbm_source_base(args: argparse.Namespace) -> Path:
    if args.reuse_nbm_source_root is not None:
        return args.reuse_nbm_source_root.resolve()
    return args.output_root.resolve() / "nbm_source"


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
        "--device",
        "cuda",
        "--num-workers",
        str(args.num_workers),
        "--nbm-max-epochs",
        "300",
        "--nbm-patience",
        "20",
        "--nbm-dropout",
        "0.10",
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
        "transformer_48k",
        "--nbm-seeds",
        args.seeds,
        "--tcn-seeds",
        args.seeds,
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
    gpu_ids = validate_gpus(args.gpu_ids, not args.dry_run)
    root = args.output_root.resolve()
    source_base = nbm_source_base(args)
    reuse_nbm = args.reuse_nbm_source_root is not None

    nbm_jobs = [
        {
            "id": f"fold{fold}_TRANSFORMER_NGM48K_seed{seed}",
            "command": nbm_command(args, fold, seed),
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
        "strategy": (
            f"8-GPU queue; {0 if reuse_nbm else len(nbm_jobs)} compact NGM jobs, "
            f"then {len(train_jobs)} TCN jobs, global barrier, and "
            f"{len(evaluate_jobs)} test jobs"
        ),
        "dataset": str(args.data_dir.resolve()),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "gpu_ids": gpu_ids,
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "seeds": list(seeds),
        "seed_policy": "exact paired NGM/TCN seed; no fold offset",
        "ngm_architecture": (
            "patch8; Linear72->40; Transformer encoder x2 (40/4/80); "
            "global mean->Z16; broadcast; decoder self-attention x1; Linear40->72"
        ),
        "ngm_parameter_count": 48_208,
        "ngm_jobs": 0 if reuse_nbm else len(nbm_jobs),
        "ngm_source_mode": "reuse_frozen" if reuse_nbm else "fit_new",
        "ngm_source_root": str(source_base),
        "classifier_train_jobs": len(train_jobs),
        "post_barrier_test_jobs": len(evaluate_jobs),
        "ngm_training": (
            "role4 fit; clean role5 selection/calibration; max300/pat20; "
            "SmoothL1; AdamW lr1e-3; 40/40/20 clean/Gaussian/Mask4-8"
        ),
        "classifier_training": (
            "roles6/7 weighted BCE; roles2/3 PR-AUC checkpoint and BA threshold; "
            "max5/pat2; AdamW lr1e-3"
        ),
        "classifier_input": "scheme C [r,abs(r),delta(r)] [B,27,128]",
        "example_ngm": None if reuse_nbm else command_text(nbm_jobs[0]["command"]),
        "example_train": command_text(train_jobs[0]["command"]),
        "seal": command_text(singleton(args, "seal")),
        "example_test": command_text(evaluate_jobs[0]["command"]),
        "aggregate": command_text(singleton(args, "aggregate")),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    data_dir = args.data_dir.resolve()
    required_identity = (
        "manifest.csv",
        "schema.json",
        "nbm_protocol.json",
        "nbm_quality_report.json",
    )
    missing = [str(data_dir / name) for name in required_identity if not (data_dir / name).exists()]
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
                    raise FileNotFoundError(f"reused Transformer-NGM is not frozen: {done}")
            print(f"REUSE FROZEN TRANSFORMER-NGM source={source_base}", flush=True)
        else:
            run_pool("nbm", nbm_jobs, gpu_ids, root)
        if args.phase == "nbm":
            print(f"TRANSFORMER-NGM READY source={source_base}", flush=True)
            return

    if args.phase in ("full", "train"):
        for fold, seed in specs:
            done = source_base / f"seed_{seed}" / f"fold_{fold}" / "DONE_NBM.json"
            if not done.exists():
                raise FileNotFoundError(f"Transformer-NGM fold/seed not frozen: {done}")
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
