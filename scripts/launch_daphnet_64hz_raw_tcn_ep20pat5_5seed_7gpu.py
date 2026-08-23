#!/usr/bin/env python3
"""Run the strict 64-Hz Daphnet RAW+TCN experiment on seven GPUs.

This launcher retrains 15 RAW classifiers (3 folds x 5 exact seeds).  It does
not train or run an NBM.  To preserve the preprocessing contract used by the
paired Daphnet experiments, it reuses only the role-4 RobustScaler artifacts
from a completed five-seed GRU-NBM source.  The shared strict worker verifies
that source provenance before training.

Protocol:
  * processed_NBM at 64 Hz, window=128 samples, stride=64 samples;
  * RAW input = role-4 RobustScaler followed by per-window/per-axis centering;
  * TCN max_epoch=20, early-stopping patience=5;
  * exact seeds 0, 52, 161, 5216, 52161, without fold offsets;
  * roles 6/7 train, roles 2/3 select checkpoint and threshold;
  * all 15 jobs are sealed before roles 0/1 are evaluated.
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

PAIR_WORKER = REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py"
FOLDS = (0, 1, 2)
METHODS = ("RAW",)
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
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed_NBM"
        ),
    )
    parser.add_argument(
        "--scaler-source-root",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / (
                "daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_"
                "seedset_0_52_161_5216_52161"
            )
            / "nbm_source"
        ),
        help=(
            "Completed five-seed nbm_source directory. RAW uses only each "
            "fold's role-4 RobustScaler; NBM reconstruction and calibration "
            "are not used."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / (
                "daphnet_64Hz_raw_tcn_ep20pat5_"
                "seedset_0_52_161_5216_52161"
            )
        ),
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--seeds", default=SEED_TEXT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=20)
    parser.add_argument("--tcn-patience", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = parse_seed_list(args.seeds)
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires exact seeds {SEED_TEXT}")
    if args.tcn_max_epochs != 20 or args.tcn_patience != 5:
        raise ValueError("this experiment requires TCN max_epoch=20 and patience=5")
    return seeds


def validate_gpus(value: str, check_hardware: bool) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if (
        len(gpu_ids) != 7
        or len(gpu_ids) != len(set(gpu_ids))
        or any(not item.isdigit() for item in gpu_ids)
    ):
        raise ValueError(
            f"this launcher requires exactly seven unique numeric GPU ids: {value}"
        )
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in gpu_ids):
            raise ValueError(
                f"requested GPU ids {gpu_ids}, but nvidia-smi reports {count} GPUs"
            )
    return gpu_ids


def source_for_seed(args: argparse.Namespace, seed: int) -> Path:
    return args.scaler_source_root.resolve() / f"seed_{seed}"


def common_worker_args(args: argparse.Namespace, source: Path) -> list[str]:
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
        args.seeds,
        "--tcn-seeds",
        args.seeds,
        "--required-seeds",
        SEED_TEXT,
        "--experiment-methods",
        "RAW",
        "--sampling-rate-hz",
        "64",
        "--window-samples",
        "128",
        "--stride-samples",
        "64",
        "--num-workers",
        str(args.num_workers),
        "--tcn-max-epochs",
        "20",
        "--tcn-patience",
        "5",
        "--required-nbm-max-epochs",
        "300",
        "--required-nbm-patience",
        "20",
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def job_command(
    args: argparse.Namespace,
    stage: str,
    fold: int,
    seed: int,
) -> list[str]:
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *common_worker_args(args, source_for_seed(args, seed)),
        "--fold",
        str(fold),
        "--method",
        "RAW",
        "--nbm-seed",
        str(seed),
        "--tcn-seed",
        str(seed),
        "--device",
        "cuda",
    ]


def singleton_command(args: argparse.Namespace, stage: str) -> list[str]:
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *common_worker_args(args, args.scaler_source_root.resolve()),
    ]


def validate_scaler_sources(args: argparse.Namespace, seeds: tuple[int, ...]) -> None:
    checkpoint_name = "gru_nbm_best.pt"
    for fold in FOLDS:
        for seed in seeds:
            directory = source_for_seed(args, seed) / f"fold_{fold}"
            required = (
                directory / "DONE_NBM.json",
                directory / "nbm_frozen.json",
                directory / "checkpoints" / checkpoint_name,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "role-4 scaler source is incomplete; missing " + ", ".join(missing)
                )


def main() -> None:
    args = parse_args()
    seeds = validate_contract(args)
    gpu_ids = validate_gpus(args.gpu_ids, check_hardware=not args.dry_run)
    output_root = args.output_root.resolve()

    specs = [(fold, seed) for fold in FOLDS for seed in seeds]
    train_jobs = [
        {
            "id": f"fold{fold}_RAW_seed{seed}",
            "command": job_command(args, "train", fold, seed),
        }
        for fold, seed in specs
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_RAW_seed{seed}",
            "command": job_command(args, "evaluate", fold, seed),
        }
        for fold, seed in specs
    ]
    plan = {
        "experiment": "64-Hz Daphnet centered-scaled RAW plus TCN",
        "strategy": (
            "7-GPU dynamic queue; 15 classifier jobs; global seal; "
            "15 post-barrier test jobs"
        ),
        "dataset": str(args.data_dir.resolve()),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "gpu_ids": gpu_ids,
        "seeds": list(seeds),
        "seed_policy": "exact seeds; no hidden fold offset",
        "classifier_train_jobs": len(train_jobs),
        "post_barrier_test_jobs": len(evaluate_jobs),
        "nbm_trained_or_inferred": False,
        "scaler_source_root": str(args.scaler_source_root.resolve()),
        "scaler_source_usage": (
            "role-4 RobustScaler only; NBM weights, b and sigma are not used "
            "to construct RAW features"
        ),
        "input": (
            "role-4 RobustScaler then per-window/per-axis temporal centering; "
            "[B,9,128]"
        ),
        "classifier": {
            "architecture": "RepresentationTCNM, input_channels=9",
            "maximum_epochs": 20,
            "early_stopping_patience": 5,
            "batch_size": 128,
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
            "checkpoint_rule": "maximum roles2/3 validation PR-AUC",
            "threshold_rule": (
                "roles2/3: maximum balanced accuracy; ties by FoG F1, "
                "then higher threshold"
            ),
        },
        "role_contract": {
            "scaler_fit": [4],
            "classifier_train": [6, 7],
            "classifier_checkpoint_and_threshold": [2, 3],
            "permanent_test_after_global_seal": [0, 1],
        },
        "example_train": command_text(train_jobs[0]["command"]),
        "seal": command_text(singleton_command(args, "seal")),
        "example_test": command_text(evaluate_jobs[0]["command"]),
        "aggregate": command_text(singleton_command(args, "aggregate")),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    protocol_path = args.data_dir.resolve() / "nbm_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"processed_NBM protocol missing: {protocol_path}")
    validate_scaler_sources(args, seeds)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    launch_plan_path = output_root / "logs" / "launch_plan.json"
    if launch_plan_path.exists() and not args.overwrite:
        existing = json.loads(launch_plan_path.read_text(encoding="utf-8"))
        identity_keys = (
            "dataset",
            "sampling_rate_hz",
            "window_samples",
            "stride_samples",
            "folds",
            "methods",
            "gpu_ids",
            "seeds",
            "scaler_source_root",
            "classifier",
        )
        if any(existing.get(key) != plan.get(key) for key in identity_keys):
            raise RuntimeError(
                "existing output-root belongs to a different experiment; "
                "use a new --output-root"
            )
    launch_plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    if args.phase in ("full", "train"):
        run_pool("train", train_jobs, gpu_ids, output_root)
        subprocess.run(
            singleton_command(args, "seal"),
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        if args.phase == "train":
            print(f"TRAINING SEALED output={output_root}", flush=True)
            return

    if args.phase in ("full", "evaluate"):
        barrier_path = output_root / "TRAINING_BARRIER.json"
        if not barrier_path.is_file():
            raise FileNotFoundError(
                "evaluation requires TRAINING_BARRIER.json; run --phase train first"
            )
        run_pool("evaluate", evaluate_jobs, gpu_ids, output_root)
        subprocess.run(
            singleton_command(args, "aggregate"),
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    elif args.phase == "aggregate":
        subprocess.run(
            singleton_command(args, "aggregate"),
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    print(f"COMPLETE phase={args.phase} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
