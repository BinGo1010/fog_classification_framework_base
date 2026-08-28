#!/usr/bin/env python3
"""Train 30 matched Scheme-C TCNs for two Daphnet GRU-NGM arms on 8 GPUs."""

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
from scripts import train_daphnet_gru_ngm_robustness_tcn as worker


WORKER = REPO_ROOT / "scripts" / "train_daphnet_gru_ngm_robustness_tcn.py"
DEFAULT_DATA_DIR = (
    REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "daphnet_gru_ngm_robustness_matched_tcn"
CRITICAL_CODE = (
    WORKER,
    Path(__file__).resolve(),
    REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--none-ngm-root", type=Path, required=True)
    parser.add_argument("--gaussian-mask-ngm-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
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


def code_hashes() -> dict[str, str]:
    missing = [str(path) for path in CRITICAL_CODE if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"critical source files missing: {missing}")
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in CRITICAL_CODE
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())
    roots = source_roots(args)
    source_jobs: dict[str, Any] = {}
    for arm in worker.ARMS:
        for fold in worker.FOLDS:
            for seed in worker.SEEDS:
                key = worker.job_key(arm, fold, seed)
                source_jobs[key] = worker.inspect_source_artifacts(
                    roots[arm], fold, seed, scientific["sha256"]
                )
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
        "folds": list(worker.FOLDS),
        "seeds": list(worker.SEEDS),
        "job_count": len(source_jobs),
        "expected_checkpoints_per_arm": len(worker.FOLDS) * len(worker.SEEDS),
        "source_roots": {key: str(value) for key, value in roots.items()},
        "source_jobs": source_jobs,
        "representation": "Scheme C [r,abs(r),delta_t(r)] with 27 channels",
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
            "same fold/seed uses the same 27-channel TCN initial state in both arms"
        ),
        "test_corruption_used_during_tcn_training": False,
    }
    if plan["job_count"] != 30:
        raise AssertionError(f"expected 30 matched TCN jobs, found {plan['job_count']}")
    for arm in worker.ARMS:
        count = sum(key.startswith(f"{arm}/") for key in source_jobs)
        if count != 15:
            raise AssertionError(f"expected 15 source checkpoints for {arm}, found {count}")
    plan["plan_id"] = canonical_fingerprint(
        {key: value for key, value in plan.items() if key != "created_utc"}
    )
    return plan


def ensure_plan(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / "EXPERIMENT_PLAN.json"
    proposed = build_plan(args)
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


def jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # Adjacent arms make each paired comparison run under similar server load.
    for fold in worker.FOLDS:
        for seed in worker.SEEDS:
            for arm in worker.ARMS:
                output.append(
                    {
                        "id": f"fold{fold}_seed{seed}_{arm}_TCN",
                        "command": [
                            args.python,
                            "-u",
                            str(WORKER),
                            "--arm",
                            arm,
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
    fold: int,
    seed: int,
) -> dict[str, Any]:
    destination = worker.run_dir(root, arm, fold, seed)
    source = plan["source_jobs"][worker.job_key(arm, fold, seed)]
    if not worker.completed_training_is_valid(destination, plan, source):
        raise FileNotFoundError(f"matched TCN training incomplete: {destination}")
    return json.loads(
        (destination / "FROZEN_TCN.json").read_text(encoding="utf-8")
    )


def audit_and_seal(root: Path, plan: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    barrier_jobs: dict[str, Any] = {}
    for fold in worker.FOLDS:
        for seed in worker.SEEDS:
            by_arm = {
                arm: load_frozen(root, plan, arm, fold, seed)
                for arm in worker.ARMS
            }
            initialization_hashes = {
                value["initialization"]["full_c_27ch_state_sha256"]
                for value in by_arm.values()
            }
            scaler_hashes = {
                value["role4_scaler_sha256"] for value in by_arm.values()
            }
            pos_weights = {
                float(value["training"]["pos_weight"])
                for value in by_arm.values()
            }
            if len(initialization_hashes) != 1:
                raise AssertionError(
                    f"paired TCN initialization mismatch: fold={fold}, seed={seed}"
                )
            if len(scaler_hashes) != 1:
                raise AssertionError(
                    f"paired role-4 scaler mismatch: fold={fold}, seed={seed}"
                )
            if len(pos_weights) != 1:
                raise AssertionError(
                    f"paired TCN class-weight mismatch: fold={fold}, seed={seed}"
                )
            pair_key = f"fold_{fold}/seed_{seed}"
            paired[pair_key] = {
                "initial_tcn_state_sha256": next(iter(initialization_hashes)),
                "role4_scaler_sha256": next(iter(scaler_hashes)),
                "pos_weight": next(iter(pos_weights)),
                "both_arms_frozen": True,
            }
            for arm, frozen in by_arm.items():
                key = worker.job_key(arm, fold, seed)
                barrier_jobs[key] = {
                    "arm": arm,
                    "fold": fold,
                    "seed": seed,
                    "source_ngm_checkpoint_sha256": frozen[
                        "source_ngm_checkpoint_sha256"
                    ],
                    "tcn_checkpoint": frozen["tcn_checkpoint"],
                    "tcn_checkpoint_sha256": frozen["tcn_checkpoint_sha256"],
                    "calibration_sha256": frozen["calibration_sha256"],
                    "frozen_id": frozen["frozen_id"],
                    "threshold": frozen["threshold"],
                }
                rows.append(
                    {
                        "arm": arm,
                        "arm_display_name": worker.ARM_DISPLAY_NAMES[arm],
                        "fold": fold,
                        "seed": seed,
                        "best_epoch": frozen["training"]["best_epoch"],
                        "epochs_completed": frozen["training"]["epochs_completed"],
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
                        "initial_tcn_state_sha256": frozen[
                            "initialization"
                        ]["full_c_27ch_state_sha256"],
                    }
                )

    if len(rows) != 30 or len(paired) != 15:
        raise AssertionError("matched TCN audit count mismatch")
    audit = {
        "schema": "daphnet_gru_ngm_robustness_tcn_pairing_audit.v1",
        "status": "verified",
        "plan_id": plan["plan_id"],
        "job_count": len(rows),
        "paired_block_count": len(paired),
        "rules": [
            "one matched clean-trained TCN per GRU-NGM checkpoint",
            "both arms share the same TCN initialization for each fold/seed",
            "both arms share the same role-4 scaler and TCN class weight",
            "roles 0/1 were not accessed during TCN training",
        ],
        "pairs": paired,
    }
    audit_path = root / "TCN_PAIRING_AUDIT.json"
    summary_path = root / "TCN_TRAINING_SUMMARY.csv"
    atomic_json_dump(audit, audit_path)
    worker.write_csv(summary_path, rows)
    barrier = {
        "schema": "daphnet_gru_ngm_robustness_tcn_training_barrier.v1",
        "status": "all_30_ngm_tcn_pipelines_frozen_before_robustness_test",
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
    if (
        args.tcn_max_epochs,
        args.tcn_patience,
    ) != (worker.TCN_MAX_EPOCHS, worker.TCN_PATIENCE):
        raise ValueError(
            "matched backend must use the previous frozen TCN settings: "
            f"max_epochs={worker.TCN_MAX_EPOCHS}, patience={worker.TCN_PATIENCE}"
        )


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.none_ngm_root = args.none_ngm_root.resolve()
    args.gaussian_mask_ngm_root = args.gaussian_mask_ngm_root.resolve()
    args.output_root = args.output_root.resolve()
    validate_settings(args)
    gpu_ids = list(parse_csv_values(args.gpu_ids))
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError("--gpu-ids must contain exactly eight unique GPU identifiers")
    if any(not value.isdigit() for value in gpu_ids):
        raise ValueError("GPU identifiers must be non-negative integers")

    plan = ensure_plan(args)
    train_jobs = jobs(args)
    print(
        f"PLAN id={plan['plan_id']} jobs={len(train_jobs)} "
        f"paired_blocks={len(train_jobs) // 2} gpus={','.join(gpu_ids)}",
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
        f"COMPLETE matched_tcn_jobs={len(train_jobs)} results={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
