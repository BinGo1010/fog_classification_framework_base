#!/usr/bin/env python3
"""Run the Gaussian-augmentation Conv-TCN NBM control and only group C."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_daphnet_residual_calibration_abcd_7gpu import (
    command_text,
    parse_seed_list,
    run_pool,
    visible_gpu_count,
)

NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_conv_tcn_nbm_gaussian200_fold.py"
CLASSIFIER_WORKER = REPO_ROOT / "scripts" / "run_daphnet_residual_calibration_abcd.py"
FOLDS = (0, 1, 2)
GROUP = "C"


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
        / "daphnet_conv_tcn_nbm200_C_gaussian40_3seed_seed20260807",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "nbm", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-seed", type=int, default=20260807)
    parser.add_argument("--nbm-max-epochs", type=int, default=200)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--nbm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--nbm-dropout", type=float, default=0.10)
    parser.add_argument("--clean-probability", type=float, default=0.40)
    parser.add_argument("--gaussian-probability", type=float, default=0.40)
    parser.add_argument("--mask-probability", type=float, default=0.20)
    parser.add_argument("--gaussian-std", type=float, default=0.04)
    parser.add_argument("--mask-min-samples", type=int, default=4)
    parser.add_argument("--mask-max-samples", type=int, default=8)
    parser.add_argument("--tcn-seeds", default="20260807,20260808,20260809")
    parser.add_argument("--tcn-max-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output = args.output_root.resolve()
    return output, output / "nbm_source", output / "c_results"


def nbm_command(args: argparse.Namespace, nbm_root: Path, fold: int) -> list[str]:
    command = [
        args.python,
        str(NBM_WORKER),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(nbm_root),
        "--fold",
        str(fold),
        "--device",
        "cuda",
        "--seed",
        str(args.nbm_seed),
        "--num-workers",
        str(args.num_workers),
        "--nbm-max-epochs",
        str(args.nbm_max_epochs),
        "--nbm-patience",
        str(args.nbm_patience),
        "--nbm-learning-rate",
        str(args.nbm_learning_rate),
        "--nbm-dropout",
        str(args.nbm_dropout),
        "--clean-probability",
        str(args.clean_probability),
        "--gaussian-probability",
        str(args.gaussian_probability),
        "--mask-probability",
        str(args.mask_probability),
        "--gaussian-std",
        str(args.gaussian_std),
        "--mask-min-samples",
        str(args.mask_min_samples),
        "--mask-max-samples",
        str(args.mask_max_samples),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def classifier_common(
    args: argparse.Namespace, nbm_root: Path, classifier_root: Path
) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--nbm-source-root",
        str(nbm_root),
        "--output-root",
        str(classifier_root),
        "--groups",
        GROUP,
        "--tcn-seeds",
        args.tcn_seeds,
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


def classifier_job(
    args: argparse.Namespace,
    nbm_root: Path,
    classifier_root: Path,
    stage: str,
    fold: int,
    seed: int,
) -> list[str]:
    return [
        args.python,
        str(CLASSIFIER_WORKER),
        "--stage",
        stage,
        *classifier_common(args, nbm_root, classifier_root),
        "--fold",
        str(fold),
        "--group",
        GROUP,
        "--tcn-seed",
        str(seed),
        "--device",
        "cuda",
    ]


def singleton(
    args: argparse.Namespace,
    nbm_root: Path,
    classifier_root: Path,
    stage: str,
) -> list[str]:
    return [
        args.python,
        str(CLASSIFIER_WORKER),
        "--stage",
        stage,
        *classifier_common(args, nbm_root, classifier_root),
    ]


def validate_gpu_ids(value: str, hardware: bool) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not item.isdigit() for item in ids):
        raise ValueError(f"invalid unique GPU ids: {value}")
    if hardware:
        count = visible_gpu_count()
        for item in ids:
            if int(item) >= count:
                raise ValueError(f"requested GPU {item}, but only {count} GPUs found")
    return ids


def validate_nbms(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    expected = {
        "clean_probability": args.clean_probability,
        "gaussian_probability": args.gaussian_probability,
        "mask_probability": args.mask_probability,
        "gaussian_std": args.gaussian_std,
        "mask_minimum_samples": args.mask_min_samples,
        "mask_maximum_samples": args.mask_max_samples,
        "mask_all_channels": True,
    }
    artifacts: list[dict[str, Any]] = []
    for fold in FOLDS:
        directory = root / f"fold_{fold}"
        checkpoint = directory / "checkpoints" / "conv_tcn_nbm_best.pt"
        frozen_path = directory / "nbm_frozen.json"
        done_path = directory / "DONE_NBM.json"
        for path in (checkpoint, frozen_path, done_path, directory / "config.json"):
            if not path.exists():
                raise FileNotFoundError(f"missing NBM artifact: {path}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        done = json.loads(done_path.read_text(encoding="utf-8"))
        training = frozen["training"]
        if training["augmentation"] != expected:
            raise AssertionError(f"fold {fold} augmentation is not 40/40/20")
        if int(training["maximum_epochs"]) != args.nbm_max_epochs:
            raise AssertionError(f"fold {fold} maximum-epoch mismatch")
        if int(training["patience"]) != args.nbm_patience:
            raise AssertionError(f"fold {fold} patience mismatch")
        if training["loss"] != "SmoothL1(beta=1.0)":
            raise AssertionError(f"fold {fold} loss mismatch")
        if frozen.get("validation_mask_or_noise") is not False:
            raise AssertionError(f"fold {fold} role-5 validation was augmented")
        if not frozen.get("best_checkpoint_restored_before_calibration", False):
            raise AssertionError(f"fold {fold} best checkpoint was not restored")
        artifacts.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.resolve()),
                "best_epoch": int(done["best_epoch"]),
                "epochs_completed": int(done["epochs_completed"]),
                "best_validation_huber": float(done["best_validation_huber"]),
                "augmentation": done["augmentation"],
            }
        )
    return artifacts


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + previous if previous else ""
    )
    return env


def run_single(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, env=process_env(), check=True)


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.tcn_seeds)
    gpu_ids = validate_gpu_ids(args.gpu_ids, hardware=not args.dry_run)
    output_root, nbm_root, classifier_root = roots(args)
    nbm_jobs = [
        {"id": f"fold{fold}_nbm_gaussian40", "command": nbm_command(args, nbm_root, fold)}
        for fold in FOLDS
    ]
    specs = [(fold, seed) for fold in FOLDS for seed in seeds]
    train_jobs = [
        {
            "id": f"fold{fold}_groupC_seed{seed}",
            "command": classifier_job(
                args, nbm_root, classifier_root, "train", fold, seed
            ),
        }
        for fold, seed in specs
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_groupC_seed{seed}",
            "command": classifier_job(
                args, nbm_root, classifier_root, "evaluate", fold, seed
            ),
        }
        for fold, seed in specs
    ]
    seal = singleton(args, nbm_root, classifier_root, "seal")
    aggregate = singleton(args, nbm_root, classifier_root, "aggregate")
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "conv_tcn_nbm200_gaussian40_control_group_C_only",
        "controlled_change": (
            "NBM role-4 augmentation only: 80% clean + 20% mask becomes "
            "40% clean + 40% Gaussian(std=0.04) + 20% mask"
        ),
        "gpu_ids": gpu_ids,
        "nbm_jobs": len(nbm_jobs),
        "classifier_group": GROUP,
        "classifier_training_jobs": len(train_jobs),
        "classifier_evaluation_jobs": len(evaluate_jobs),
        "tcn_seeds": list(seeds),
        "example_nbm_command": command_text(nbm_jobs[0]["command"]),
        "example_classifier_command": command_text(train_jobs[0]["command"]),
        "seal_command": command_text(seal),
        "aggregate_command": command_text(aggregate),
        "paths": {
            "output_root": str(output_root),
            "nbm_source": str(nbm_root),
            "c_results": str(classifier_root),
        },
        "test_gate": "all 9 C checkpoints and thresholds freeze before roles 0/1",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not (args.data_dir.resolve() / "nbm_protocol.json").exists():
        raise FileNotFoundError(f"processed_NBM protocol missing: {args.data_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.phase in ("full", "nbm"):
        run_pool("nbm_gaussian40", nbm_jobs, gpu_ids, output_root)
        artifacts = validate_nbms(args, nbm_root)
        (output_root / "NBM_BARRIER.json").write_text(
            json.dumps(
                {
                    "status": "all_gaussian40_nbms_frozen",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.phase == "nbm":
        print(f"COMPLETE phase=nbm output={output_root}", flush=True)
        return
    if args.phase in ("full", "train"):
        validate_nbms(args, nbm_root)
        run_pool("classifier_train_C", train_jobs, gpu_ids, output_root)
        run_single(seal)
    if args.phase in ("full", "evaluate"):
        barrier = classifier_root / "TRAINING_BARRIER.json"
        if not barrier.exists():
            raise FileNotFoundError(f"evaluation requires barrier: {barrier}")
        run_pool("classifier_evaluate_C", evaluate_jobs, gpu_ids, output_root)
        run_single(aggregate)
    elif args.phase == "aggregate":
        run_single(aggregate)
    print(f"COMPLETE phase={args.phase} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
