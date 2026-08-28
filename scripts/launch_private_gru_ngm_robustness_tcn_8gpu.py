#!/usr/bin/env python3
"""Train matched Scheme-C TCNs for two Private GRU-NGM arms on 8 GPUs."""

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
from scripts import train_private_gru_ngm_robustness_tcn as worker
from scripts.launch_daphnet_residual_calibration_abcd_7gpu import run_pool


WORKER = REPO_ROOT / "scripts" / "train_private_gru_ngm_robustness_tcn.py"
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "private_gru_ngm_robustness_matched_tcn"
CRITICAL_CODE = (
    WORKER,
    Path(__file__).resolve(),
    REPO_ROOT / "scripts" / "run_private_gru_ngm_perturbation_4arm.py",
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn.py",
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_raw_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
    REPO_ROOT / "cnbr_fog" / "data.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
)


def parse_csv_values(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique comma-separated values: {text}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--none-ngm-root", type=Path, required=True)
    parser.add_argument("--gaussian-mask-ngm-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--subjects",
        default="auto",
        help="auto, or comma-separated Private subject IDs such as P01",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--tcn-max-epochs", type=int, default=worker.TCN_MAX_EPOCHS
    )
    parser.add_argument("--tcn-patience", type=int, default=worker.TCN_PATIENCE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "none": args.none_ngm_root.resolve(),
        "gaussian_mask": args.gaussian_mask_ngm_root.resolve(),
    }


def checkpoint_count(root: Path, arm: str, subject: str) -> int:
    return sum(
        worker.source_job_exists(root, arm, subject, fold, seed)
        for fold in worker.FOLDS
        for seed in worker.SEEDS
    )


def resolve_subjects(args: argparse.Namespace) -> tuple[str, ...]:
    roots = source_roots(args)
    if args.subjects.strip().lower() != "auto":
        subjects = parse_csv_values(args.subjects)
        unknown = [subject for subject in subjects if subject not in worker.SUBJECTS]
        if unknown:
            raise ValueError(
                f"unknown Private subjects {unknown}; expected one of {worker.SUBJECTS}"
            )
        return subjects

    expected = len(worker.FOLDS) * len(worker.SEEDS)
    coverage = {
        subject: {
            arm: checkpoint_count(roots[arm], arm, subject)
            for arm in worker.ARMS
        }
        for subject in worker.SUBJECTS
    }
    complete = tuple(
        subject
        for subject, counts in coverage.items()
        if all(counts[arm] == expected for arm in worker.ARMS)
    )
    if complete:
        partial = {
            subject: counts
            for subject, counts in coverage.items()
            if subject not in complete and any(counts.values())
        }
        if partial:
            print(
                f"AUTO SUBJECTS: using complete={complete}; ignoring partial={partial}",
                flush=True,
            )
        return complete
    nonzero = {
        subject: counts
        for subject, counts in coverage.items()
        if any(counts.values())
    }
    raise FileNotFoundError(
        "auto-detection found no subject with all 15 checkpoints in both arms; "
        f"coverage={nonzero}. Expected layout example: "
        "<arm-root>/P01/fold_0/seed_0/checkpoints/gru_ngm_best.pt"
    )


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
) -> dict[str, Any]:
    scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())
    roots = source_roots(args)
    source_jobs: dict[str, Any] = {}
    for subject in subjects:
        for arm in worker.ARMS:
            for fold in worker.FOLDS:
                for seed in worker.SEEDS:
                    key = worker.job_key(arm, subject, fold, seed)
                    source_jobs[key] = worker.inspect_source_artifacts(
                        roots[arm],
                        arm,
                        subject,
                        fold,
                        seed,
                        scientific["sha256"],
                    )
    expected_per_arm = len(subjects) * len(worker.FOLDS) * len(worker.SEEDS)
    plan: dict[str, Any] = {
        "schema": worker.PLAN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "data_scientific_sha256": scientific["sha256"],
        "data_file_count": len(scientific["files"]),
        "code_sha256": code_hashes(),
        "arms": list(worker.ARMS),
        "arm_display_names": worker.ARM_DISPLAY_NAMES,
        "subjects": list(subjects),
        "folds": list(worker.FOLDS),
        "seeds": list(worker.SEEDS),
        "job_count": len(source_jobs),
        "expected_checkpoints_per_arm": expected_per_arm,
        "source_roots": {key: str(value) for key, value in roots.items()},
        "source_jobs": source_jobs,
        "representation": "Scheme C [r,abs(r),delta_t(r)] with 90 channels",
        "ngm_calibration_role": 5,
        "tcn_train_roles": [6, 7],
        "tcn_validation_roles": [2, 3],
        "test_roles_locked": [0, 1],
        "tcn_batch_size": worker.TCN_BATCH_SIZE,
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "tcn_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "tcn_loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "tcn_checkpoint": "maximum clean roles2/3 AP",
        "paired_tcn_initialization": (
            "same subject/fold/seed uses the same 90-channel TCN initial state "
            "in both arms"
        ),
        "test_corruption_used_during_tcn_training": False,
    }
    expected_jobs = 2 * expected_per_arm
    if plan["job_count"] != expected_jobs:
        raise AssertionError(
            f"expected {expected_jobs} matched TCN jobs, found {plan['job_count']}"
        )
    for arm in worker.ARMS:
        count = sum(key.startswith(f"{arm}/") for key in source_jobs)
        if count != expected_per_arm:
            raise AssertionError(
                f"expected {expected_per_arm} source checkpoints for {arm}, "
                f"found {count}"
            )
    plan["plan_id"] = canonical_fingerprint(
        {key: value for key, value in plan.items() if key != "created_utc"}
    )
    return plan


def ensure_plan(
    args: argparse.Namespace,
    subjects: tuple[str, ...],
) -> dict[str, Any]:
    path = args.output_root / "EXPERIMENT_PLAN.json"
    proposed = build_plan(args, subjects)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != proposed["plan_id"]:
            raise AssertionError(
                "output root contains a different source/data/code/config plan; "
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
        "--plan-root",
        str(args.output_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--num-workers",
        str(args.num_workers),
        "--tcn-max-epochs",
        str(args.tcn_max_epochs),
        "--tcn-patience",
        str(args.tcn_patience),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def jobs(
    args: argparse.Namespace,
    subjects: tuple[str, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # Adjacent arms keep paired runs under similar server load.
    for subject in subjects:
        for fold in worker.FOLDS:
            for seed in worker.SEEDS:
                for arm in worker.ARMS:
                    output.append(
                        {
                            "id": (
                                f"{subject}_fold{fold}_seed{seed}_{arm}_TCN"
                            ),
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
    source = plan["source_jobs"][worker.job_key(arm, subject, fold, seed)]
    if not worker.completed_training_is_valid(destination, plan, source):
        raise FileNotFoundError(f"matched TCN training incomplete: {destination}")
    return json.loads(
        (destination / "FROZEN_TCN.json").read_text(encoding="utf-8")
    )


def audit_and_seal(root: Path, plan: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    barrier_jobs: dict[str, Any] = {}
    for subject in plan["subjects"]:
        for fold in worker.FOLDS:
            for seed in worker.SEEDS:
                by_arm = {
                    arm: load_frozen(root, plan, arm, subject, fold, seed)
                    for arm in worker.ARMS
                }
                initialization_hashes = {
                    value["training"]["initial_model_state_sha256"]
                    for value in by_arm.values()
                }
                scaler_hashes = {
                    value["role4_scaler_values_sha256"]
                    for value in by_arm.values()
                }
                pos_weights = {
                    float(value["training"]["pos_weight"])
                    for value in by_arm.values()
                }
                if len(initialization_hashes) != 1:
                    raise AssertionError(
                        "paired TCN initialization mismatch: "
                        f"subject={subject}, fold={fold}, seed={seed}"
                    )
                if len(scaler_hashes) != 1:
                    raise AssertionError(
                        "paired role-4 scaler mismatch: "
                        f"subject={subject}, fold={fold}, seed={seed}"
                    )
                if len(pos_weights) != 1:
                    raise AssertionError(
                        "paired TCN class-weight mismatch: "
                        f"subject={subject}, fold={fold}, seed={seed}"
                    )
                pair_key = f"{subject}/fold_{fold}/seed_{seed}"
                paired[pair_key] = {
                    "initial_tcn_state_sha256": next(iter(initialization_hashes)),
                    "role4_scaler_values_sha256": next(iter(scaler_hashes)),
                    "pos_weight": next(iter(pos_weights)),
                    "both_arms_frozen": True,
                }
                for arm, frozen in by_arm.items():
                    key = worker.job_key(arm, subject, fold, seed)
                    barrier_jobs[key] = {
                        "arm": arm,
                        "subject": subject,
                        "fold": fold,
                        "seed": seed,
                        "source_ngm_checkpoint_sha256": frozen[
                            "source_ngm_checkpoint_sha256"
                        ],
                        "tcn_checkpoint": frozen["tcn_checkpoint"],
                        "tcn_checkpoint_sha256": frozen[
                            "tcn_checkpoint_sha256"
                        ],
                        "calibration_sha256": frozen["calibration_sha256"],
                        "frozen_id": frozen["frozen_id"],
                        "threshold": frozen["threshold"],
                    }
                    rows.append(
                        {
                            "arm": arm,
                            "arm_display_name": worker.ARM_DISPLAY_NAMES[arm],
                            "subject": subject,
                            "fold": fold,
                            "seed": seed,
                            "best_epoch": frozen["training"]["best_epoch"],
                            "epochs_completed": frozen["training"][
                                "epochs_completed"
                            ],
                            "validation_ap": frozen["training"][
                                "best_validation_pr_auc"
                            ],
                            "pos_weight": frozen["training"]["pos_weight"],
                            "threshold": frozen["threshold"],
                            "source_ngm_checkpoint_sha256": frozen[
                                "source_ngm_checkpoint_sha256"
                            ],
                            "tcn_checkpoint_sha256": frozen[
                                "tcn_checkpoint_sha256"
                            ],
                            "initial_tcn_state_sha256": frozen["training"][
                                "initial_model_state_sha256"
                            ],
                        }
                    )

    expected_pairs = len(plan["subjects"]) * len(worker.FOLDS) * len(worker.SEEDS)
    if len(rows) != 2 * expected_pairs or len(paired) != expected_pairs:
        raise AssertionError("matched TCN audit count mismatch")
    audit = {
        "schema": "private_gru_ngm_robustness_tcn_pairing_audit.v1",
        "status": "verified",
        "plan_id": plan["plan_id"],
        "job_count": len(rows),
        "paired_block_count": len(paired),
        "rules": [
            "one matched clean-trained TCN per GRU-NGM checkpoint",
            "both arms share one TCN initialization per subject/fold/seed",
            "both arms share the role-4 scaler and TCN class weight",
            "roles 0/1 were not materialized during TCN training",
        ],
        "pairs": paired,
    }
    audit_path = root / "TCN_PAIRING_AUDIT.json"
    summary_path = root / "TCN_TRAINING_SUMMARY.csv"
    atomic_json_dump(audit, audit_path)
    worker.write_csv(summary_path, rows)
    barrier = {
        "schema": "private_gru_ngm_robustness_tcn_training_barrier.v1",
        "status": "all_matched_ngm_tcn_pipelines_frozen_before_robustness_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "job_count": len(barrier_jobs),
        "test_roles_unlocked_for_next_stage": True,
        "test_corruptions_evaluated": False,
        "jobs": barrier_jobs,
    }
    barrier["barrier_id"] = canonical_fingerprint(
        {key: value for key, value in barrier.items() if key != "created_utc"}
    )
    barrier_path = root / "TCN_TRAINING_BARRIER.json"
    atomic_json_dump(barrier, barrier_path)
    atomic_json_dump(
        {
            "schema": worker.EXPERIMENT_SCHEMA,
            "status": "all_matched_tcn_training_complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "plan_id": plan["plan_id"],
            "job_count": len(rows),
            "pairing_audit_sha256": sha256_file(audit_path),
            "summary_sha256": sha256_file(summary_path),
            "training_barrier_sha256": sha256_file(barrier_path),
        },
        root / "DONE_TCN_TRAINING.json",
    )


def validate_settings(args: argparse.Namespace) -> None:
    if (args.tcn_max_epochs, args.tcn_patience) != (
        worker.TCN_MAX_EPOCHS,
        worker.TCN_PATIENCE,
    ):
        raise ValueError(
            "matched backend must use the previous frozen TCN settings: "
            f"max_epochs={worker.TCN_MAX_EPOCHS}, "
            f"patience={worker.TCN_PATIENCE}"
        )


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.none_ngm_root = args.none_ngm_root.resolve()
    args.gaussian_mask_ngm_root = args.gaussian_mask_ngm_root.resolve()
    args.output_root = args.output_root.resolve()
    validate_settings(args)
    gpu_ids = list(parse_csv_values(args.gpu_ids))
    if len(gpu_ids) != 8 or any(not value.isdigit() for value in gpu_ids):
        raise ValueError(
            "--gpu-ids must contain exactly eight unique non-negative integers"
        )
    subjects = resolve_subjects(args)
    plan = ensure_plan(args, subjects)
    train_jobs = jobs(args, subjects)
    print(
        f"PLAN id={plan['plan_id']} subjects={','.join(subjects)} "
        f"jobs={len(train_jobs)} paired_blocks={len(train_jobs) // 2} "
        f"gpus={','.join(gpu_ids)}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: source preflight passed; no TCN training started", flush=True)
        print(
            "FIRST JOB:", subprocess.list2cmdline(train_jobs[0]["command"]), flush=True
        )
        print(
            "LAST JOB:", subprocess.list2cmdline(train_jobs[-1]["command"]), flush=True
        )
        return

    run_pool("train_tcn", train_jobs, gpu_ids, args.output_root)
    audit_and_seal(args.output_root, plan)
    print(
        f"COMPLETE matched_tcn_jobs={len(train_jobs)} "
        f"results={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
