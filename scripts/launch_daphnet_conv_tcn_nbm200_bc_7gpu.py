#!/usr/bin/env python3
"""Train fresh 200-epoch-budget Conv-TCN NBMs, then run B/C on seven GPUs.

Pipeline:
  1. train one role-4/5 NBM per fold (three independent jobs);
  2. freeze each best role-5 checkpoint and its role-5 b/sigma calibration;
  3. train 3 folds x 2 groups x 3 TCN seeds (18 jobs);
  4. seal all checkpoints and validation thresholds behind one global barrier;
  5. evaluate roles 0/1 and aggregate only after the barrier exists.
"""

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

NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_conv_tcn_nbm_200_fold.py"
CLASSIFIER_WORKER = REPO_ROOT / "scripts" / "run_daphnet_residual_calibration_abcd.py"
FOLDS = (0, 1, 2)
GROUPS = ("B", "C")


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
        / "daphnet_conv_tcn_nbm200_BC_3seed_seed20260807",
    )
    parser.add_argument(
        "--nbm-source-root",
        type=Path,
        default=None,
        help="Defaults to <output-root>/nbm_source.",
    )
    parser.add_argument(
        "--classifier-output-root",
        type=Path,
        default=None,
        help="Defaults to <output-root>/bc_results.",
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
    parser.add_argument("--mask-probability", type=float, default=0.20)
    parser.add_argument("--mask-min-samples", type=int, default=4)
    parser.add_argument("--mask-max-samples", type=int, default=8)
    parser.add_argument("--tcn-seeds", default="20260807,20260808,20260809")
    parser.add_argument("--tcn-max-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_root = args.output_root.resolve()
    nbm_root = (
        args.nbm_source_root.resolve()
        if args.nbm_source_root is not None
        else output_root / "nbm_source"
    )
    classifier_root = (
        args.classifier_output_root.resolve()
        if args.classifier_output_root is not None
        else output_root / "bc_results"
    )
    return output_root, nbm_root, classifier_root


def nbm_command(
    args: argparse.Namespace, nbm_root: Path, fold: int
) -> list[str]:
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
        "--mask-probability",
        str(args.mask_probability),
        "--mask-min-samples",
        str(args.mask_min_samples),
        "--mask-max-samples",
        str(args.mask_max_samples),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def classifier_common_args(
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
        ",".join(GROUPS),
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


def classifier_job_command(
    args: argparse.Namespace,
    nbm_root: Path,
    classifier_root: Path,
    stage: str,
    fold: int,
    group: str,
    seed: int,
) -> list[str]:
    return [
        args.python,
        str(CLASSIFIER_WORKER),
        "--stage",
        stage,
        *classifier_common_args(args, nbm_root, classifier_root),
        "--fold",
        str(fold),
        "--group",
        group,
        "--tcn-seed",
        str(seed),
        "--device",
        "cuda",
    ]


def classifier_singleton_command(
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
        *classifier_common_args(args, nbm_root, classifier_root),
    ]


def validate_gpu_ids(value: str, check_hardware: bool) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"invalid unique --gpu-ids: {value}")
    if any(not item.isdigit() for item in gpu_ids):
        raise ValueError("GPU ids must be non-negative integers")
    if check_hardware:
        count = visible_gpu_count()
        for gpu in gpu_ids:
            if int(gpu) >= count:
                raise ValueError(
                    f"requested GPU {gpu}, but nvidia-smi reports {count} GPUs"
                )
    return gpu_ids


def validate_frozen_nbms(args: argparse.Namespace, nbm_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_dir = nbm_root / f"fold_{fold}"
        checkpoint = fold_dir / "checkpoints" / "conv_tcn_nbm_best.pt"
        frozen_path = fold_dir / "nbm_frozen.json"
        done_path = fold_dir / "DONE_NBM.json"
        config_path = fold_dir / "config.json"
        for path in (checkpoint, frozen_path, done_path, config_path):
            if not path.exists():
                raise FileNotFoundError(f"required frozen NBM artifact missing: {path}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        done = json.loads(done_path.read_text(encoding="utf-8"))
        training = frozen["training"]
        if int(training["maximum_epochs"]) != args.nbm_max_epochs:
            raise AssertionError(f"fold {fold} NBM maximum-epoch mismatch")
        if int(training["patience"]) != args.nbm_patience:
            raise AssertionError(f"fold {fold} NBM patience mismatch")
        if training["loss"] != "SmoothL1(beta=1.0)":
            raise AssertionError(f"fold {fold} NBM loss mismatch")
        if not frozen.get("best_checkpoint_restored_before_calibration", False):
            raise AssertionError(f"fold {fold} best NBM was not restored")
        if frozen.get("validation_mask") is not False:
            raise AssertionError(f"fold {fold} role-5 validation must be unmasked")
        artifacts.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.resolve()),
                "best_epoch": int(done["best_epoch"]),
                "epochs_completed": int(done["epochs_completed"]),
                "best_validation_huber": float(done["best_validation_huber"]),
            }
        )
    return artifacts


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    return env


def run_singleton(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, env=subprocess_env(), check=True)


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.tcn_seeds)
    gpu_ids = validate_gpu_ids(args.gpu_ids, check_hardware=not args.dry_run)
    output_root, nbm_root, classifier_root = resolved_roots(args)
    nbm_jobs = [
        {
            "id": f"fold{fold}_nbm200",
            "command": nbm_command(args, nbm_root, fold),
        }
        for fold in FOLDS
    ]
    job_specs = [
        (fold, group, seed)
        for fold in FOLDS
        for group in GROUPS
        for seed in seeds
    ]
    train_jobs = [
        {
            "id": f"fold{fold}_group{group}_seed{seed}",
            "command": classifier_job_command(
                args, nbm_root, classifier_root, "train", fold, group, seed
            ),
        }
        for fold, group, seed in job_specs
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_group{group}_seed{seed}",
            "command": classifier_job_command(
                args, nbm_root, classifier_root, "evaluate", fold, group, seed
            ),
        }
        for fold, group, seed in job_specs
    ]
    seal_command = classifier_singleton_command(
        args, nbm_root, classifier_root, "seal"
    )
    aggregate_command = classifier_singleton_command(
        args, nbm_root, classifier_root, "aggregate"
    )
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "fresh_conv_tcn_nbm200_then_residual_BC",
        "gpu_ids": gpu_ids,
        "nbm": {
            "jobs": len(nbm_jobs),
            "maximum_concurrent_jobs": min(len(gpu_ids), len(nbm_jobs)),
            "max_epochs": args.nbm_max_epochs,
            "patience": args.nbm_patience,
            "loss": "SmoothL1(beta=1.0)",
            "learning_rate": args.nbm_learning_rate,
            "example_command": command_text(nbm_jobs[0]["command"]),
        },
        "classifier": {
            "groups": list(GROUPS),
            "folds": list(FOLDS),
            "tcn_seeds": list(seeds),
            "training_jobs": len(train_jobs),
            "evaluation_jobs_after_global_barrier": len(evaluate_jobs),
            "maximum_concurrent_jobs": len(gpu_ids),
            "example_train_command": command_text(train_jobs[0]["command"]),
            "seal_command": command_text(seal_command),
            "example_evaluate_command": command_text(evaluate_jobs[0]["command"]),
            "aggregate_command": command_text(aggregate_command),
        },
        "paths": {
            "output_root": str(output_root),
            "nbm_source_root": str(nbm_root),
            "classifier_output_root": str(classifier_root),
        },
        "strict_test_gate": (
            "all 18 B/C classifier checkpoints and role-2/3 thresholds must be "
            "sealed before any role-0/1 window is requested"
        ),
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
        run_pool("nbm", nbm_jobs, gpu_ids, output_root)
        artifacts = validate_frozen_nbms(args, nbm_root)
        (output_root / "NBM_BARRIER.json").write_text(
            json.dumps(
                {
                    "status": "all_fold_nbms_frozen",
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
        validate_frozen_nbms(args, nbm_root)
        run_pool("classifier_train", train_jobs, gpu_ids, output_root)
        run_singleton(seal_command)
    if args.phase in ("full", "evaluate"):
        barrier = classifier_root / "TRAINING_BARRIER.json"
        if not barrier.exists():
            raise FileNotFoundError(f"evaluation requires global barrier: {barrier}")
        run_pool("classifier_evaluate", evaluate_jobs, gpu_ids, output_root)
        run_singleton(aggregate_command)
    elif args.phase == "aggregate":
        run_singleton(aggregate_command)
    print(f"COMPLETE phase={args.phase} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
