#!/usr/bin/env python
"""Launch the five-seed step-augmented GRU-NBM experiment on seven GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as base
from scripts.launch_daphnet_residual_calibration_abcd_7gpu import run_pool
from scripts.run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_step_augmented import (
    BARRIER_SCHEMA,
    EXPERIMENT_SCHEMA,
    FOLDS,
    MODEL_DESCRIPTION,
    NBM_VARIANT,
    SEEDS,
    SUBJECTS,
)


WORKER = (
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_step_augmented.py"
)
LATEST_EVENT_SUMMARY = REPO_ROOT / "scripts" / "summarize_private_raw_tcn_latest_event_metrics.py"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "processed_NBM_Exp_gru_nbm_step_aug40_40_20_C_tcn_5seed"
)
CRITICAL_CODE = (
    WORKER,
    Path(__file__).resolve(),
    LATEST_EVENT_SUMMARY,
    REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn.py",
    REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_raw_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
    REPO_ROOT / "cnbr_fog" / "data.py",
    REPO_ROOT / "cnbr_fog" / "evaluation.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
)


def parse_seed_list(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique seed list: {text}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--phase", choices=("full", "train", "evaluate", "aggregate"), default="full")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-batch-size", type=int, default=16)
    parser.add_argument("--maximum-updates", type=int, default=5000)
    parser.add_argument("--validation-frequency", type=int, default=50)
    parser.add_argument("--validation-patience", type=int, default=20)
    parser.add_argument("--nbm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--nbm-weight-decay", type=float, default=1e-4)
    parser.add_argument("--tcn-batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def code_hashes() -> dict[str, str]:
    missing = [str(path) for path in CRITICAL_CODE if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"critical source files missing: {missing}")
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in CRITICAL_CODE
    }


def build_plan(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())
    plan: dict[str, Any] = {
        "schema": EXPERIMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "data_scientific_sha256": scientific["sha256"],
        "data_file_count": len(scientific["files"]),
        "code_sha256": code_hashes(),
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(seeds),
        "job_count": len(SUBJECTS) * len(FOLDS) * len(seeds),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "raw_channels": 30,
        "tcn_input_channels": 90,
        "nbm_variant": NBM_VARIANT,
        "model": MODEL_DESCRIPTION,
        "nbm_architecture": base.architecture_config(),
        "nbm_augmentation": base.augmentation_config(),
        "augmentation_sampling": "dynamic mutually-exclusive per role-4 window encounter",
        "representation": "scheme C [r,abs(r),delta(r)]",
        "nbm_batch_size": args.nbm_batch_size,
        "nbm_maximum_updates": args.maximum_updates,
        "nbm_validation_frequency": args.validation_frequency,
        "nbm_validation_patience": args.validation_patience,
        "nbm_learning_rate": args.nbm_learning_rate,
        "nbm_weight_decay": args.nbm_weight_decay,
        "tcn_batch_size": args.tcn_batch_size,
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "tcn_initialization_contract": (
            "reset from the same job seed immediately before RepresentationTCNM construction; "
            "all subject/fold jobs sharing a seed must have the same initial-state SHA256"
        ),
        "base_event_metric": (
            "coverage_aware.v2 inside primary metrics; latest 1-s merge metrics "
            "are recomputed after aggregation"
        ),
    }
    plan["plan_id"] = canonical_fingerprint(
        {key: value for key, value in plan.items() if key != "created_utc"}
    )
    return plan


def ensure_plan(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    path = args.output_root / "EXPERIMENT_PLAN.json"
    proposed = build_plan(args, seeds)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != proposed["plan_id"]:
            raise AssertionError(
                "output-root has a different data/code/config identity; use a new output root"
            )
        return existing
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(proposed, path)
    return proposed


def common_args(args: argparse.Namespace, seeds: tuple[int, ...]) -> list[str]:
    values = [
        "--data-dir", str(args.data_dir.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--seeds", ",".join(map(str, seeds)),
        "--num-workers", str(args.num_workers),
        "--nbm-batch-size", str(args.nbm_batch_size),
        "--maximum-updates", str(args.maximum_updates),
        "--validation-frequency", str(args.validation_frequency),
        "--validation-patience", str(args.validation_patience),
        "--nbm-learning-rate", str(args.nbm_learning_rate),
        "--nbm-weight-decay", str(args.nbm_weight_decay),
        "--tcn-batch-size", str(args.tcn_batch_size),
        "--tcn-max-epochs", str(args.tcn_max_epochs),
        "--tcn-patience", str(args.tcn_patience),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def jobs(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    stage: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                output.append(
                    {
                        "id": f"{subject}_fold{fold}_seed{seed}",
                        "command": [
                            args.python,
                            "-u",
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


def single_command(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    stage: str,
) -> list[str]:
    return [
        args.python,
        "-u",
        str(WORKER),
        "--stage",
        stage,
        *common_args(args, seeds),
    ]


def verify_paired_tcn_initialization(
    output_root: Path,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        hashes: dict[str, list[str]] = {}
        for subject in SUBJECTS:
            for fold in FOLDS:
                path = (
                    output_root
                    / "runs"
                    / subject
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                    / "FROZEN_TRAIN.json"
                )
                frozen = json.loads(path.read_text(encoding="utf-8"))
                digest = frozen["tcn_training"]["initial_model_state_sha256"]
                hashes.setdefault(digest, []).append(f"{subject}/fold_{fold}")
        if len(hashes) != 1:
            raise AssertionError(
                f"TCN initialization mismatch within seed {seed}: {list(hashes)}"
            )
        digest, jobs_with_hash = next(iter(hashes.items()))
        if len(jobs_with_hash) != len(SUBJECTS) * len(FOLDS):
            raise AssertionError(f"incomplete TCN initialization audit for seed {seed}")
        per_seed[str(seed)] = {
            "initial_model_state_sha256": digest,
            "verified_job_count": len(jobs_with_hash),
        }
    audit = {
        "schema": "paired_tcn_initialization_audit.v1",
        "status": "verified",
        "rule": "one identical 90-channel TCN initial state per seed across all subjects and folds",
        "seeds": per_seed,
    }
    atomic_json_dump(audit, output_root / "TCN_INITIALIZATION_AUDIT.json")
    return audit


def run_latest_event_summary(args: argparse.Namespace) -> None:
    destination = args.output_root / "latest_event_metrics_1s"
    subprocess.run(
        [
            args.python,
            "-u",
            str(LATEST_EVENT_SUMMARY),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--experiment-root",
            str(args.output_root.resolve()),
            "--output-dir",
            str(destination.resolve()),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    seeds = parse_seed_list(args.seeds)
    if seeds != SEEDS:
        raise ValueError(f"five paired seeds are frozen to {SEEDS}; received {seeds}")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) != 7 or len(set(gpu_ids)) != 7 or any(not value.isdigit() for value in gpu_ids):
        raise ValueError("--gpu-ids must contain seven unique non-negative integers")
    if args.nbm_batch_size != 16:
        raise ValueError("the clean-step comparison freezes NBM batch size at 16")
    if (
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
        args.tcn_batch_size,
        args.tcn_max_epochs,
        args.tcn_patience,
    ) != (5000, 50, 20, 128, 5, 2):
        raise ValueError("step/TCN settings differ from the frozen clean-step comparison")
    if abs(args.nbm_learning_rate - 3e-4) > 1e-12 or abs(args.nbm_weight_decay - 1e-4) > 1e-12:
        raise ValueError("NBM optimizer settings differ from the frozen clean-step comparison")

    plan = ensure_plan(args, seeds)
    train_jobs = jobs(args, seeds, "train")
    evaluate_jobs = jobs(args, seeds, "evaluate")
    print(
        f"PLAN id={plan['plan_id']} train_jobs={len(train_jobs)} "
        f"evaluate_jobs={len(evaluate_jobs)} gpus={','.join(gpu_ids)}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: no training, sealing, test, or aggregation executed", flush=True)
        print("FIRST TRAIN:", subprocess.list2cmdline(train_jobs[0]["command"]), flush=True)
        print("LAST TRAIN:", subprocess.list2cmdline(train_jobs[-1]["command"]), flush=True)
        return
    if args.phase in ("full", "train"):
        run_pool("train", train_jobs, gpu_ids, args.output_root)
        verify_paired_tcn_initialization(args.output_root, seeds)
        subprocess.run(single_command(args, seeds, "seal"), cwd=REPO_ROOT, check=True)
    if args.phase in ("full", "evaluate"):
        barrier = args.output_root / "TRAINING_BARRIER.json"
        if not barrier.is_file():
            raise FileNotFoundError("evaluate requires the sealed TRAINING_BARRIER.json")
        run_pool("evaluate", evaluate_jobs, gpu_ids, args.output_root)
    if args.phase in ("full", "evaluate", "aggregate"):
        subprocess.run(single_command(args, seeds, "aggregate"), cwd=REPO_ROOT, check=True)
        run_latest_event_summary(args)


if __name__ == "__main__":
    main()
