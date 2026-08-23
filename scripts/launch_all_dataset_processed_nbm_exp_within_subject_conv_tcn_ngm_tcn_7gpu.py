#!/usr/bin/env python
"""Launch 120 within-subject Conv-TCN-NGM+TCN jobs on seven GPUs."""

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
from scripts.run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn import (
    EXPERIMENT_SCHEMA,
    FOLDS,
    NBM_PARAMETER_COUNT,
    NBM_VARIANT,
    SEEDS,
    SUBJECTS,
    architecture_config,
    augmentation_config,
)


WORKER = (
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn.py"
)
DEFAULT_EXPERIMENT = (
    "all_dataset_processed_NBM_Exp_within_subject_conv_tcn_ngm_mask4_8_C_tcn_"
    "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)
CRITICAL_CODE = (
    WORKER,
    Path(__file__).resolve(),
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn.py",
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn.py",
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_raw_tcn.py",
    REPO_ROOT
    / "scripts"
    / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
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
        default=REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / DEFAULT_EXPERIMENT,
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> tuple[tuple[int, ...], list[str]]:
    seeds = parse_seed_list(args.seeds)
    if seeds != SEEDS:
        raise ValueError(f"the five seeds are frozen to {SEEDS}; received {seeds}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("Conv-TCN NGM requires max_epoch=300 and patience=20")
    if args.tcn_max_epochs != 5 or args.tcn_patience != 2:
        raise ValueError("TCN classifier requires max_epoch=5 and patience=2")
    if args.batch_size != 128:
        raise ValueError("this experiment requires batch_size=128")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if (
        len(gpu_ids) not in (7, 8)
        or len(set(gpu_ids)) != len(gpu_ids)
        or any(not value.isdigit() for value in gpu_ids)
    ):
        raise ValueError(
            "--gpu-ids must contain seven unique GPU ids or eight unique GPU ids"
        )
    return seeds, gpu_ids


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
        "gpu_count": len(
            [value for value in args.gpu_ids.split(",") if value.strip()]
        ),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "raw_channels": 30,
        "tcn_input_channels": 90,
        "nbm_variant": NBM_VARIANT,
        "nbm_parameter_count": NBM_PARAMETER_COUNT,
        "nbm_architecture": architecture_config(),
        "nbm_augmentation": augmentation_config(),
        "representation": "scheme C [r,abs(r),delta(r)]",
        "batch_size": args.batch_size,
        "nbm_max_epochs": args.nbm_max_epochs,
        "nbm_patience": args.nbm_patience,
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "event_metric": (
            "allocation group detected by any positive window; role-0 "
            "Non-FoG positive runs merged when gap<=1s; FA/h uses valid "
            "Non-FoG union exposure"
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
                "output-root has a different data/code/config identity; "
                "use a new --output-root"
            )
        return existing
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(proposed, path)
    return proposed


def common_args(args: argparse.Namespace, seeds: tuple[int, ...]) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--seeds",
        ",".join(map(str, seeds)),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--nbm-max-epochs",
        str(args.nbm_max_epochs),
        "--nbm-patience",
        str(args.nbm_patience),
        "--tcn-max-epochs",
        str(args.tcn_max_epochs),
        "--tcn-patience",
        str(args.tcn_patience),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def jobs(
    args: argparse.Namespace, seeds: tuple[int, ...], stage: str
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{subject}_fold{fold}_seed{seed}",
            "command": [
                args.python,
                str(WORKER),
                "--stage",
                stage,
                "--subject",
                subject,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
                *common_args(args, seeds),
            ],
        }
        for subject in SUBJECTS
        for fold in FOLDS
        for seed in seeds
    ]


def single_command(
    args: argparse.Namespace, seeds: tuple[int, ...], stage: str
) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--stage",
        stage,
        *common_args(args, seeds),
    ]


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    seeds, gpu_ids = validate_contract(args)
    plan = ensure_plan(args, seeds)
    train_jobs = jobs(args, seeds, "train")
    evaluate_jobs = jobs(args, seeds, "evaluate")
    print(
        f"PLAN id={plan['plan_id']} train_jobs={len(train_jobs)} "
        f"evaluate_jobs={len(evaluate_jobs)} gpus={','.join(gpu_ids)}",
        flush=True,
    )
    if args.dry_run:
        print(
            "DRY RUN: no training, sealing, evaluation, or aggregation executed",
            flush=True,
        )
        print(
            "FIRST TRAIN:",
            subprocess.list2cmdline(train_jobs[0]["command"]),
            flush=True,
        )
        print(
            "LAST TRAIN:",
            subprocess.list2cmdline(train_jobs[-1]["command"]),
            flush=True,
        )
        return
    if args.phase in ("full", "train"):
        run_pool("train", train_jobs, gpu_ids, args.output_root)
        subprocess.run(
            single_command(args, seeds, "seal"), cwd=REPO_ROOT, check=True
        )
    if args.phase in ("full", "evaluate"):
        if not (args.output_root / "TRAINING_BARRIER.json").is_file():
            raise FileNotFoundError(
                "evaluate requires TRAINING_BARRIER.json; run --phase train first"
            )
        run_pool("evaluate", evaluate_jobs, gpu_ids, args.output_root)
    if args.phase in ("full", "evaluate", "aggregate"):
        subprocess.run(
            single_command(args, seeds, "aggregate"), cwd=REPO_ROOT, check=True
        )


if __name__ == "__main__":
    main()
