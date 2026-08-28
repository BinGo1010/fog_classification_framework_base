#!/usr/bin/env python3
"""Launch the strict GRU-NGM [r,delta(r)] TCN50/pat10 experiment on 7 GPUs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import (
    launch_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn_8gpu
    as shared,
)


SOURCE_EXPERIMENT = (
    "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_"
    "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_EXPERIMENT = (
    "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_r_delta_tcn_"
    "source_nbm300pat20_ep50pat10_seedset_0_52_161_5216_52161"
)
TCN_MAX_EPOCHS = 50
TCN_PATIENCE = 10
GPU_COUNT = 7
THIS_FILE = Path(__file__).resolve()
if THIS_FILE not in shared.CRITICAL_CODE:
    shared.CRITICAL_CODE = (*shared.CRITICAL_CODE, THIS_FILE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "outputs" / SOURCE_EXPERIMENT,
        help="Completed 90-channel FULL_C source containing frozen GRU-NGMs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / DEFAULT_EXPERIMENT,
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--seeds", default=",".join(map(str, shared.SEEDS)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=TCN_MAX_EPOCHS)
    parser.add_argument("--tcn-patience", type=int, default=TCN_PATIENCE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-runtime-init-preflight",
        action="store_true",
        help="Only for a non-training dry-run on another runtime.",
    )
    return parser.parse_args()


def validate_contract(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], list[str]]:
    seeds = shared.parse_seed_list(args.seeds)
    if seeds != shared.SEEDS:
        raise ValueError(
            f"paired seeds are frozen to {shared.SEEDS}; received {seeds}"
        )
    if (args.tcn_max_epochs, args.tcn_patience) != (
        TCN_MAX_EPOCHS,
        TCN_PATIENCE,
    ):
        raise ValueError(
            "this experiment requires TCN max_epoch=50 and patience=10"
        )
    if args.batch_size != 128:
        raise ValueError("this comparison requires batch_size=128")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if (
        len(gpu_ids) != GPU_COUNT
        or len(set(gpu_ids)) != GPU_COUNT
        or any(not item.isdigit() for item in gpu_ids)
    ):
        raise ValueError(
            "--gpu-ids must contain seven unique non-negative integers"
        )
    if args.skip_runtime_init_preflight and not args.dry_run:
        raise ValueError("--skip-runtime-init-preflight requires --dry-run")
    return seeds, gpu_ids


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    seeds, gpu_ids = validate_contract(args)

    plan = (
        shared.build_plan(args, seeds)
        if args.dry_run
        else shared.ensure_plan(args, seeds)
    )
    train_jobs = shared.jobs(args, seeds, "train")
    evaluate_jobs = shared.jobs(args, seeds, "evaluate")
    print(
        f"PLAN id={plan['plan_id']} train_jobs={len(train_jobs)} "
        f"evaluate_jobs={len(evaluate_jobs)} "
        f"tcn=max{args.tcn_max_epochs}/pat{args.tcn_patience} "
        f"gpus={','.join(gpu_ids)} source={args.source_root}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: no artifact was written and no job was executed")
        print("FIRST TRAIN:", subprocess.list2cmdline(train_jobs[0]["command"]))
        print("LAST TRAIN:", subprocess.list2cmdline(train_jobs[-1]["command"]))
        return

    if args.phase in ("full", "train"):
        shared.run_pool("train", train_jobs, gpu_ids, args.output_root)
        subprocess.run(
            shared.single_command(args, seeds, "seal"),
            cwd=REPO_ROOT,
            check=True,
        )
    if args.phase in ("full", "evaluate"):
        if not (args.output_root / "TRAINING_BARRIER.json").is_file():
            raise FileNotFoundError(
                "evaluate requires TRAINING_BARRIER.json; run --phase train first"
            )
        shared.run_pool("evaluate", evaluate_jobs, gpu_ids, args.output_root)
    if args.phase in ("full", "evaluate", "aggregate"):
        subprocess.run(
            shared.single_command(args, seeds, "aggregate"),
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
