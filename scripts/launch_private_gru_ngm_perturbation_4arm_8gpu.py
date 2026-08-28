#!/usr/bin/env python
"""Launch the paired four-arm, five-seed GRU-NGM training suite on eight GPUs."""

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
from scripts.launch_daphnet_residual_calibration_abcd_7gpu import run_pool
from scripts import run_private_gru_ngm_perturbation_4arm as worker


DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "private_gru_ngm_perturbation_4arm_5seed"
)
WORKER = REPO_ROOT / "scripts" / "run_private_gru_ngm_perturbation_4arm.py"
CRITICAL_CODE = (
    WORKER,
    Path(__file__).resolve(),
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn.py",
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_raw_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "cnbr_fog" / "data.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
)


def parse_csv_values(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique comma-separated values: {text}")
    return values


def parse_int_values(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_csv_values(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--subjects", default=",".join(worker.SUBJECTS))
    parser.add_argument("--folds", default=",".join(map(str, worker.FOLDS)))
    parser.add_argument("--seeds", default=",".join(map(str, worker.SEEDS)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-batch-size", type=int, default=16)
    parser.add_argument("--maximum-updates", type=int, default=5000)
    parser.add_argument("--validation-frequency", type=int, default=50)
    parser.add_argument("--validation-patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def build_plan(
    args: argparse.Namespace,
    subjects: tuple[str, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())
    plan: dict[str, Any] = {
        "schema": worker.PLAN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "data_scientific_sha256": scientific["sha256"],
        "data_file_count": len(scientific["files"]),
        "code_sha256": code_hashes(),
        "subjects": list(subjects),
        "folds": list(folds),
        "seeds": list(seeds),
        "arms": list(worker.ARMS),
        "arm_display_names": worker.ARM_DISPLAY_NAMES,
        "arm_augmentation": {
            arm: worker.augmentation_config(arm) for arm in worker.ARMS
        },
        "job_count": len(subjects)
        * len(folds)
        * len(seeds)
        * len(worker.ARMS),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "raw_channels": 30,
        "architecture": worker.base.architecture_config(),
        "paired_initialization": (
            "same subject/fold/seed uses an identical initial GRU-NGM state "
            "in all four arms"
        ),
        "paired_batch_order": (
            "same subject/fold/seed uses the same role-4 DataLoader seed in all arms"
        ),
        "clean_target": True,
        "clean_role5_validation": True,
        "permanent_test_roles_loaded": False,
        "nbm_batch_size": args.nbm_batch_size,
        "maximum_updates": args.maximum_updates,
        "validation_frequency": args.validation_frequency,
        "validation_patience": args.validation_patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    plan["plan_id"] = canonical_fingerprint(
        {key: value for key, value in plan.items() if key != "created_utc"}
    )
    return plan


def ensure_plan(
    args: argparse.Namespace,
    subjects: tuple[str, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    path = args.output_root / "EXPERIMENT_PLAN.json"
    proposed = build_plan(args, subjects, folds, seeds)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != proposed["plan_id"]:
            raise AssertionError(
                "output root already contains a different data/code/config plan; "
                "use a new --output-root"
            )
        return existing
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(proposed, path)
    return proposed


def common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--num-workers",
        str(args.num_workers),
        "--nbm-batch-size",
        str(args.nbm_batch_size),
        "--maximum-updates",
        str(args.maximum_updates),
        "--validation-frequency",
        str(args.validation_frequency),
        "--validation-patience",
        str(args.validation_patience),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def jobs(
    args: argparse.Namespace,
    subjects: tuple[str, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # Keep the four arms adjacent so an eight-GPU machine trains two paired
    # subject/fold/seed blocks at a time under similar system load.
    for subject in subjects:
        for fold in folds:
            for seed in seeds:
                for arm in worker.ARMS:
                    output.append(
                        {
                            "id": f"{subject}_fold{fold}_seed{seed}_{arm}",
                            "command": [
                                args.python,
                                "-u",
                                str(WORKER),
                                "--arm",
                                arm,
                                "--subject",
                                subject,
                                "--fold",
                                str(fold),
                                "--seed",
                                str(seed),
                                "--device",
                                "cuda:0",
                                *common_args(args),
                            ],
                        }
                    )
    return output


def load_frozen(
    root: Path,
    plan: dict[str, Any],
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    destination = worker.run_dir(root, arm, subject, fold, seed)
    if not worker.validate_completed_train(destination, plan):
        raise FileNotFoundError(f"training job incomplete: {destination}")
    return json.loads(
        (destination / "FROZEN_TRAIN.json").read_text(encoding="utf-8")
    )


def audit_and_summarize(
    root: Path,
    plan: dict[str, Any],
    subjects: tuple[str, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    rows: list[dict[str, Any]] = []
    paired_jobs: dict[str, Any] = {}
    for subject in subjects:
        for fold in folds:
            for seed in seeds:
                frozen_by_arm = {
                    arm: load_frozen(root, plan, arm, subject, fold, seed)
                    for arm in worker.ARMS
                }
                initial_hashes = {
                    value["training"]["initial_model_state_sha256"]
                    for value in frozen_by_arm.values()
                }
                scaler_hashes = {
                    value["scaler_sha256"] for value in frozen_by_arm.values()
                }
                role_counts = {
                    json.dumps(value["role_counts"], sort_keys=True)
                    for value in frozen_by_arm.values()
                }
                if len(initial_hashes) != 1:
                    raise AssertionError(
                        f"four-arm initialization mismatch: {subject}/fold{fold}/seed{seed}"
                    )
                if len(scaler_hashes) != 1:
                    raise AssertionError(
                        f"four-arm scaler mismatch: {subject}/fold{fold}/seed{seed}"
                    )
                if len(role_counts) != 1:
                    raise AssertionError(
                        f"four-arm data-role mismatch: {subject}/fold{fold}/seed{seed}"
                    )

                key = f"{subject}/fold_{fold}/seed_{seed}"
                paired_jobs[key] = {
                    "initial_model_state_sha256": next(iter(initial_hashes)),
                    "scaler_sha256": next(iter(scaler_hashes)),
                    "four_arms_present": True,
                }
                for arm, frozen in frozen_by_arm.items():
                    training = frozen["training"]
                    forbidden_gaussian = arm in ("none", "mask_only")
                    forbidden_mask = arm in ("none", "gaussian_only")
                    if forbidden_gaussian and training["total_gaussian_windows"] != 0:
                        raise AssertionError(f"unexpected Gaussian windows in {key}/{arm}")
                    if forbidden_mask and training["total_masked_windows"] != 0:
                        raise AssertionError(f"unexpected masked windows in {key}/{arm}")
                    if (
                        worker.ARM_PROBABILITIES[arm][1] > 0
                        and training["total_gaussian_windows"] <= 0
                    ):
                        raise AssertionError(f"missing Gaussian windows in {key}/{arm}")
                    if (
                        worker.ARM_PROBABILITIES[arm][2] > 0
                        and training["total_masked_windows"] <= 0
                    ):
                        raise AssertionError(f"missing masked windows in {key}/{arm}")
                    rows.append(
                        {
                            "arm": arm,
                            "arm_display_name": worker.ARM_DISPLAY_NAMES[arm],
                            "subject": subject,
                            "fold": fold,
                            "seed": seed,
                            "updates_completed": training["updates_completed"],
                            "best_step": training["best_step"],
                            "best_clean_validation_smooth_l1": training[
                                "best_clean_validation_smooth_l1"
                            ],
                            "stopped_early": training["stopped_early"],
                            "elapsed_seconds": training["elapsed_seconds"],
                            "empirical_clean_fraction": training[
                                "empirical_clean_fraction"
                            ],
                            "empirical_gaussian_fraction": training[
                                "empirical_gaussian_fraction"
                            ],
                            "empirical_mask_fraction": training[
                                "empirical_mask_fraction"
                            ],
                            "checkpoint_sha256": frozen["checkpoint_sha256"],
                            "initial_model_state_sha256": training[
                                "initial_model_state_sha256"
                            ],
                        }
                    )

    audit = {
        "schema": "private_gru_ngm_perturbation_4arm_pairing_audit.v1",
        "status": "verified",
        "plan_id": plan["plan_id"],
        "rules": [
            "all four arms exist for every subject/fold/seed",
            "all four arms share the same initial GRU-NGM state",
            "all four arms share the same role-4 scaler and role allocation",
            "forbidden corruption modes have zero training windows",
            "each enabled corruption mode has at least one training window",
        ],
        "paired_block_count": len(paired_jobs),
        "jobs": paired_jobs,
    }
    audit_path = root / "PAIRING_AUDIT.json"
    summary_csv = root / "TRAINING_SUMMARY.csv"
    atomic_json_dump(audit, audit_path)
    worker.write_csv(summary_csv, rows)
    summary = {
        "schema": "private_gru_ngm_perturbation_4arm_training_summary.v1",
        "status": "complete",
        "plan_id": plan["plan_id"],
        "job_count": len(rows),
        "expected_job_count": plan["job_count"],
        "arms": list(worker.ARMS),
        "subjects": list(subjects),
        "folds": list(folds),
        "seeds": list(seeds),
        "checkpoint_pattern": (
            "runs/<arm>/<subject>/fold_<fold>/seed_<seed>/checkpoints/"
            + worker.CHECKPOINT_NAME
        ),
    }
    if summary["job_count"] != summary["expected_job_count"]:
        raise AssertionError("training summary job count mismatch")
    summary_path = root / "TRAINING_SUMMARY.json"
    atomic_json_dump(summary, summary_path)
    atomic_json_dump(
        {
            "schema": worker.EXPERIMENT_SCHEMA,
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "plan_id": plan["plan_id"],
            "job_count": len(rows),
            "pairing_audit_sha256": sha256_file(audit_path),
            "training_summary_csv_sha256": sha256_file(summary_csv),
            "training_summary_json_sha256": sha256_file(summary_path),
        },
        root / "DONE.json",
    )


def validate_frozen_settings(args: argparse.Namespace) -> None:
    values = (
        args.nbm_batch_size,
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
    )
    if values != (16, 5000, 50, 20):
        raise ValueError(
            "training must match the previous frozen settings: "
            "batch16, updates5000, validation every50, patience20"
        )
    if abs(args.learning_rate - 3e-4) > 1e-12:
        raise ValueError("learning rate is frozen to 3e-4")
    if abs(args.weight_decay - 1e-4) > 1e-12:
        raise ValueError("weight decay is frozen to 1e-4")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    validate_frozen_settings(args)

    subjects = parse_csv_values(args.subjects)
    folds = parse_int_values(args.folds)
    seeds = parse_int_values(args.seeds)
    gpu_ids = list(parse_csv_values(args.gpu_ids))
    if any(subject not in worker.SUBJECTS for subject in subjects):
        raise ValueError(f"subjects must be selected from {worker.SUBJECTS}")
    if any(fold not in worker.FOLDS for fold in folds):
        raise ValueError(f"folds must be selected from {worker.FOLDS}")
    if seeds != worker.SEEDS:
        raise ValueError(
            f"the five paired seeds are frozen to {worker.SEEDS}; received {seeds}"
        )
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError("--gpu-ids must contain exactly eight unique GPU identifiers")
    if any(not value.isdigit() for value in gpu_ids):
        raise ValueError("GPU identifiers must be non-negative integers")

    plan = ensure_plan(args, subjects, folds, seeds)
    train_jobs = jobs(args, subjects, folds, seeds)
    print(
        f"PLAN id={plan['plan_id']} jobs={len(train_jobs)} "
        f"paired_blocks={len(train_jobs) // len(worker.ARMS)} "
        f"gpus={','.join(gpu_ids)}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: no GRU-NGM training was started", flush=True)
        print(
            "FIRST JOB:", subprocess.list2cmdline(train_jobs[0]["command"]), flush=True
        )
        print(
            "LAST JOB:", subprocess.list2cmdline(train_jobs[-1]["command"]), flush=True
        )
        return

    run_pool("train", train_jobs, gpu_ids, args.output_root)
    audit_and_summarize(args.output_root, plan, subjects, folds, seeds)
    print(
        f"COMPLETE jobs={len(train_jobs)} results={args.output_root}", flush=True
    )


if __name__ == "__main__":
    main()
