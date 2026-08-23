#!/usr/bin/env python
"""Launch the strict r-only GRU-NBM residual ablation on eight GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.launch_daphnet_residual_calibration_abcd_7gpu import run_pool
from scripts.run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_only_tcn import (
    EXPERIMENT_SCHEMA,
    FOLDS,
    REPRESENTATION,
    SEEDS,
    SUBJECTS,
    paired_r_only_tcn,
)

WORKER = REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_only_tcn.py"
SOURCE_EXPERIMENT = (
    "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_"
    "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_EXPERIMENT = (
    "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_r_only_tcn_"
    "source_nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)
CRITICAL_CODE = (
    WORKER,
    Path(__file__).resolve(),
    REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn.py",
    REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_raw_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
    REPO_ROOT / "cnbr_fog" / "data.py",
    REPO_ROOT / "cnbr_fog" / "evaluation.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
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
        default=REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "outputs" / SOURCE_EXPERIMENT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / DEFAULT_EXPERIMENT,
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--phase", choices=("full", "train", "evaluate", "aggregate"), default="full")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-runtime-init-preflight",
        action="store_true",
        help="Only for a non-training dry-run on another runtime; full runs remain fail-closed.",
    )
    return parser.parse_args()


def code_hashes() -> dict[str, str]:
    missing = [str(path) for path in CRITICAL_CODE if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"critical source files missing: {missing}")
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in CRITICAL_CODE}


def validate_source_root(source_root: Path, verify_initialization: bool = True) -> dict[str, Any]:
    done_path = source_root / "DONE.json"
    plan_path = source_root / "EXPERIMENT_PLAN.json"
    barrier_path = source_root / "TRAINING_BARRIER.json"
    if not all(path.is_file() for path in (done_path, plan_path, barrier_path)):
        raise FileNotFoundError(f"completed expanded GRU-NBM source missing: {source_root}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or int(done.get("run_count", 0)) != 120:
        raise AssertionError("expanded source experiment is not complete")
    if int(barrier.get("job_count", 0)) != 120:
        raise AssertionError("expanded source training barrier does not contain 120 jobs")
    hashes_by_seed: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in SEEDS:
                frozen_path = (
                    source_root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"
                    / "FROZEN_TRAIN.json"
                )
                if not frozen_path.is_file():
                    raise FileNotFoundError(f"source training artifact missing: {frozen_path}")
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                hashes_by_seed[seed].add(
                    frozen["tcn_training"]["initial_model_state_sha256"]
                )
    if any(len(values) != 1 for values in hashes_by_seed.values()):
        raise AssertionError("source 90-channel TCN initialization varies within a seed")
    if verify_initialization:
        for seed, hashes in hashes_by_seed.items():
            paired_r_only_tcn(seed, next(iter(hashes)), torch.device("cpu"))
    return {
        "source_done_sha256": sha256_file(done_path),
        "source_plan_sha256": sha256_file(plan_path),
        "source_barrier_sha256": sha256_file(barrier_path),
        "source_plan_id": plan.get("plan_id"),
        "source_training_barrier_id": barrier.get("barrier_id"),
        "source_data_scientific_sha256": plan.get("data_scientific_sha256"),
        "source_initial_state_sha256_by_seed": {
            str(seed): next(iter(values)) for seed, values in hashes_by_seed.items()
        },
        "runtime_initialization_preflight_passed": bool(verify_initialization),
    }


def build_plan(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())
    source = validate_source_root(
        args.source_root.resolve(),
        verify_initialization=not args.skip_runtime_init_preflight,
    )
    if source["source_data_scientific_sha256"] != scientific["sha256"]:
        raise AssertionError("source GRU-NBM experiment used a different scientific dataset")
    plan: dict[str, Any] = {
        "schema": EXPERIMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "data_scientific_sha256": scientific["sha256"],
        "data_file_count": len(scientific["files"]),
        "source_identity": source,
        "code_sha256": code_hashes(),
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(seeds),
        "job_count": len(SUBJECTS) * len(FOLDS) * len(seeds),
        "representation": REPRESENTATION,
        "ablation": "remove abs(r) and delta(r); reuse exact frozen Scaler/GRU-NBM/sigma",
        "raw_channels": 30,
        "tcn_input_channels": 30,
        "paired_initialization_reference_channels": 90,
        "batch_size": args.batch_size,
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "event_metric": "coverage_aware.v2; 2 positive windows; merge gap0.5s; FA/h on evaluated valid Non-FoG union coverage",
    }
    plan["plan_id"] = canonical_fingerprint({key: value for key, value in plan.items() if key != "created_utc"})
    return plan


def ensure_plan(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    path = args.output_root / "EXPERIMENT_PLAN.json"
    proposed = build_plan(args, seeds)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != proposed["plan_id"]:
            raise AssertionError("output-root has a different data/source/code/config identity; use a new root")
        return existing
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(proposed, path)
    return proposed


def common_args(args: argparse.Namespace, seeds: tuple[int, ...]) -> list[str]:
    values = [
        "--data-dir", str(args.data_dir.resolve()),
        "--source-root", str(args.source_root.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--seeds", ",".join(map(str, seeds)),
        "--num-workers", str(args.num_workers),
        "--batch-size", str(args.batch_size),
        "--tcn-max-epochs", str(args.tcn_max_epochs),
        "--tcn-patience", str(args.tcn_patience),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def jobs(args: argparse.Namespace, seeds: tuple[int, ...], stage: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                output.append(
                    {
                        "id": f"{subject}_fold{fold}_seed{seed}",
                        "command": [
                            args.python,
                            str(WORKER),
                            "--stage", stage,
                            "--subject", subject,
                            "--fold", str(fold),
                            "--seed", str(seed),
                            "--device", "cuda:0",
                            *common_args(args, seeds),
                        ],
                    }
                )
    return output


def single_command(args: argparse.Namespace, seeds: tuple[int, ...], stage: str) -> list[str]:
    return [args.python, str(WORKER), "--stage", stage, *common_args(args, seeds)]


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.skip_runtime_init_preflight and not args.dry_run:
        raise ValueError("--skip-runtime-init-preflight is allowed only together with --dry-run")
    seeds = parse_seed_list(args.seeds)
    if seeds != SEEDS:
        raise ValueError(f"original five seeds are frozen to {SEEDS}; received {seeds}")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(not item.isdigit() for item in gpu_ids):
        raise ValueError("--gpu-ids must contain eight unique non-negative integers")
    plan = ensure_plan(args, seeds)
    train_jobs = jobs(args, seeds, "train")
    evaluate_jobs = jobs(args, seeds, "evaluate")
    print(
        f"PLAN id={plan['plan_id']} train_jobs={len(train_jobs)} evaluate_jobs={len(evaluate_jobs)} "
        f"gpus={','.join(gpu_ids)} source={args.source_root}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: no training, sealing, evaluation, or aggregation executed", flush=True)
        print("FIRST TRAIN:", subprocess.list2cmdline(train_jobs[0]["command"]), flush=True)
        print("LAST TRAIN:", subprocess.list2cmdline(train_jobs[-1]["command"]), flush=True)
        return
    if args.phase in ("full", "train"):
        run_pool("train", train_jobs, gpu_ids, args.output_root)
        subprocess.run(single_command(args, seeds, "seal"), cwd=REPO_ROOT, check=True)
    if args.phase in ("full", "evaluate"):
        if not (args.output_root / "TRAINING_BARRIER.json").is_file():
            raise FileNotFoundError("evaluate requires TRAINING_BARRIER.json; run --phase train first")
        run_pool("evaluate", evaluate_jobs, gpu_ids, args.output_root)
    if args.phase in ("full", "evaluate", "aggregate"):
        subprocess.run(single_command(args, seeds, "aggregate"), cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
