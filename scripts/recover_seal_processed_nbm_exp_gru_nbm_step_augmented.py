#!/usr/bin/env python
"""Seal completed step-augmented GRU-NBM jobs after the legacy seal mismatch.

This recovery utility does not train, evaluate, or alter any model artifact. It
validates the frozen training outputs against the original experiment plan and
creates the global barrier required before permanent-test inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as legacy
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_step_augmented as step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0,52,161,5216,52161")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if seeds != tuple(step.SEEDS):
        raise AssertionError(f"expected frozen seeds {tuple(step.SEEDS)}, got {seeds}")
    return seeds


def validate_plan(plan: dict[str, Any], data_dir: Path, output_root: Path, seeds: tuple[int, ...]) -> None:
    expected = {
        "schema": step.EXPERIMENT_SCHEMA,
        "subjects": list(step.SUBJECTS),
        "folds": list(step.FOLDS),
        "seeds": list(seeds),
        "job_count": len(step.SUBJECTS) * len(step.FOLDS) * len(seeds),
        "nbm_batch_size": 16,
        "nbm_maximum_updates": 5000,
        "nbm_validation_frequency": 50,
        "nbm_validation_patience": 20,
        "nbm_learning_rate": 3e-4,
        "nbm_weight_decay": 1e-4,
        "tcn_batch_size": 128,
        "tcn_max_epochs": 5,
        "tcn_patience": 2,
    }
    mismatches = {
        key: {"expected": value, "found": plan.get(key)}
        for key, value in expected.items()
        if plan.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"original experiment plan mismatch: {mismatches}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    seeds = parse_seeds(args.seeds)
    plan_path = output_root / "EXPERIMENT_PLAN.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing original experiment plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, data_dir, output_root, seeds)

    current_data = processed_nbm_scientific_manifest(data_dir)["sha256"]
    if current_data != plan.get("data_scientific_sha256"):
        raise AssertionError("dataset differs from the original experiment plan")

    jobs: dict[str, Any] = {}
    for subject in step.SUBJECTS:
        for fold in step.FOLDS:
            for seed in seeds:
                destination = legacy.run_dir(output_root, subject, fold, seed)
                if not legacy.validate_completed_train(destination, plan):
                    raise FileNotFoundError(f"training job incomplete: {destination}")
                frozen_path = destination / "FROZEN_TRAIN.json"
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                if (frozen["subject"], frozen["fold"], frozen["seed"]) != (subject, fold, seed):
                    raise AssertionError(f"frozen identity mismatch: {destination}")
                key = f"{subject}/fold_{fold}/seed_{seed}"
                jobs[key] = {
                    name: frozen[name]
                    for name in (
                        "frozen_id",
                        "nbm_checkpoint_sha256",
                        "tcn_checkpoint_sha256",
                        "scaler_sha256",
                        "calibration_sha256",
                        "threshold",
                    )
                }
                jobs[key]["frozen_sha256"] = sha256_file(frozen_path)

    recovery_path = Path(__file__).resolve()
    barrier = {
        "schema": step.BARRIER_SCHEMA,
        "status": "sealed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "subjects": list(step.SUBJECTS),
        "folds": list(step.FOLDS),
        "seeds": list(seeds),
        "job_count": len(jobs),
        "jobs": jobs,
        "recovery_audit": {
            "reason": "legacy seal expected batch_size instead of split NBM/TCN batch-size fields",
            "action": "validated existing frozen training artifacts and created the pre-test barrier only",
            "training_rerun": False,
            "permanent_test_access_during_recovery": False,
            "original_data_dir": plan.get("data_dir"),
            "recovery_data_dir": str(data_dir),
            "original_output_root": plan.get("output_root"),
            "recovery_output_root": str(output_root),
            "relocated_copy": (
                plan.get("data_dir") != str(data_dir)
                or plan.get("output_root") != str(output_root)
            ),
            "recovery_script": recovery_path.relative_to(REPO_ROOT).as_posix(),
            "recovery_script_sha256": sha256_file(recovery_path),
        },
    }
    barrier["barrier_id"] = canonical_fingerprint(
        {key: value for key, value in barrier.items() if key != "created_utc"}
    )
    atomic_json_dump(barrier, output_root / "TRAINING_BARRIER.json")
    print(
        f"RECOVERY SEALED jobs={len(jobs)} barrier_id={barrier['barrier_id']} "
        "training_rerun=false permanent_test_access=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
