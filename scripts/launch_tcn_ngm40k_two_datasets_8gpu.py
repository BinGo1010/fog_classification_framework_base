#!/usr/bin/env python3
"""Run both 40k TCN-NGM experiments through one strict eight-GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    sha256_file,
)
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import (
    launch_all_dataset_processed_nbm_exp_within_subject_tcn_ngm40k_tcn_7gpu
    as private_wrapper,
)
from scripts import launch_daphnet_tcn_ngm40k_c_tcn_7gpu as daphnet_wrapper
from scripts.launch_daphnet_residual_calibration_abcd_7gpu import run_pool
from scripts.tcn_ngm_40k import (
    TCN_NGM_9_PARAMETER_COUNT,
    TCN_NGM_30_PARAMETER_COUNT,
    architecture_config,
)


SEEDS = (0, 52, 161, 5216, 52161)
SEED_TEXT = ",".join(map(str, SEEDS))
FOLDS = (0, 1, 2)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "tcn_ngm40k_two_datasets_8gpu_seedset_0_52_161_5216_52161"
)
DUAL_PLAN_SCHEMA = "tcn_ngm40k_two_datasets_8gpu_plan.v1"
DUAL_BARRIER_SCHEMA = "tcn_ngm40k_two_datasets_8gpu_barrier.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daphnet-data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed_NBM"
        ),
    )
    parser.add_argument(
        "--private-data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default=SEED_TEXT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument(
        "--phase",
        choices=("full", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if values != SEEDS:
        raise ValueError(f"this experiment requires exact seeds {SEED_TEXT}")
    return values


def configure_launchers() -> tuple[Any, Any]:
    private_wrapper.configure_launcher()
    daphnet_wrapper.configure_launcher()
    return private_wrapper.launcher, daphnet_wrapper.launcher


def private_args(args: argparse.Namespace, output_root: Path) -> Namespace:
    return Namespace(
        data_dir=args.private_data_dir.resolve(),
        output_root=output_root,
        gpu_ids=args.gpu_ids,
        seeds=args.seeds,
        python=args.python,
        phase=args.phase,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        nbm_max_epochs=args.nbm_max_epochs,
        nbm_patience=args.nbm_patience,
        tcn_max_epochs=args.tcn_max_epochs,
        tcn_patience=args.tcn_patience,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def daphnet_args(args: argparse.Namespace, output_root: Path) -> Namespace:
    return Namespace(
        data_dir=args.daphnet_data_dir.resolve(),
        output_root=output_root,
        gpu_ids=args.gpu_ids,
        nbm_seeds=args.seeds,
        tcn_seeds=args.seeds,
        experiment_methods="FULL_C",
        num_workers=args.num_workers,
        nbm_max_epochs=args.nbm_max_epochs,
        nbm_patience=args.nbm_patience,
        tcn_max_epochs=args.tcn_max_epochs,
        tcn_patience=args.tcn_patience,
        python=args.python,
        phase=args.phase,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def prefixed_jobs(prefix: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**job, "id": f"{prefix}_{job['id']}"} for job in jobs]


def critical_code_hashes(private_launcher: Any, daphnet_launcher: Any) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "tcn_ngm_40k.py",
        private_wrapper.WORKER,
        Path(private_wrapper.__file__).resolve(),
        daphnet_wrapper.NBM_WORKER,
        Path(daphnet_wrapper.__file__).resolve(),
        daphnet_launcher.PAIR_WORKER,
        *private_launcher.CRITICAL_CODE,
    }
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"critical source files missing: {missing}")
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def ensure_dual_plan(
    args: argparse.Namespace,
    private_launcher: Any,
    daphnet_launcher: Any,
    gpu_ids: list[str],
    private_root: Path,
    daphnet_root: Path,
) -> dict[str, Any]:
    private_scientific = processed_nbm_scientific_manifest(
        args.private_data_dir.resolve()
    )
    daphnet_scientific = processed_nbm_scientific_manifest(
        args.daphnet_data_dir.resolve()
    )
    identity = {
        "schema": DUAL_PLAN_SCHEMA,
        "private_data_dir": str(args.private_data_dir.resolve()),
        "daphnet_data_dir": str(args.daphnet_data_dir.resolve()),
        "private_output_root": str(private_root),
        "daphnet_output_root": str(daphnet_root),
        "private_scientific_sha256": private_scientific["sha256"],
        "daphnet_scientific_sha256": daphnet_scientific["sha256"],
        "code_sha256": critical_code_hashes(private_launcher, daphnet_launcher),
        "gpu_ids": gpu_ids,
        "seeds": list(SEEDS),
        "nbm_max_epochs": args.nbm_max_epochs,
        "nbm_patience": args.nbm_patience,
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "batch_size": args.batch_size,
        "daphnet_nbm_parameter_count": TCN_NGM_9_PARAMETER_COUNT,
        "private_nbm_parameter_count": TCN_NGM_30_PARAMETER_COUNT,
        "daphnet_nbm_architecture": architecture_config(9),
        "private_nbm_architecture": architecture_config(30),
        "daphnet_jobs": {
            "nbm": len(FOLDS) * len(SEEDS),
            "classifier_train": len(FOLDS) * len(SEEDS),
            "evaluate": len(FOLDS) * len(SEEDS),
        },
        "private_jobs": {
            "train": 8 * len(FOLDS) * len(SEEDS),
            "evaluate": 8 * len(FOLDS) * len(SEEDS),
        },
        "schedule": [
            "fit all Daphnet TCN-NGMs",
            "joint eight-GPU queue for Daphnet classifier and private full training",
            "seal both training protocols and create a dual barrier",
            "joint eight-GPU queue for post-barrier evaluation",
            "aggregate both datasets independently",
        ],
    }
    plan = {
        **identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": canonical_fingerprint(identity),
    }
    path = args.output_root.resolve() / "DUAL_DATASET_PLAN.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != plan["plan_id"]:
            raise AssertionError(
                "output-root belongs to a different dual-dataset configuration; "
                "use a new --output-root"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(plan, path)
    return plan


def environment() -> dict[str, str]:
    values = os.environ.copy()
    existing = values.get("PYTHONPATH", "")
    values["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return values


def seal_dual_training(
    args: argparse.Namespace,
    private_launcher: Any,
    daphnet_launcher: Any,
    private_ns: Namespace,
    daphnet_ns: Namespace,
) -> dict[str, Any]:
    env = environment()
    subprocess.run(
        daphnet_launcher.singleton(daphnet_ns, "seal"),
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        private_launcher.single_command(private_ns, SEEDS, "seal"),
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    barriers = {
        "daphnet": daphnet_ns.output_root / "TRAINING_BARRIER.json",
        "private": private_ns.output_root / "TRAINING_BARRIER.json",
    }
    missing = [str(path) for path in barriers.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dataset training barriers missing: {missing}")
    payload = {
        "schema": DUAL_BARRIER_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "all Daphnet and private-data model checkpoints and validation-selected "
            "thresholds were frozen before either test queue was launched"
        ),
        "barriers": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in barriers.items()
        },
    }
    payload["barrier_id"] = canonical_fingerprint(
        {key: value for key, value in payload.items() if key != "created_utc"}
    )
    atomic_json_dump(
        payload, args.output_root.resolve() / "DUAL_TRAINING_BARRIER.json"
    )
    return payload


def require_dual_barrier(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root.resolve() / "DUAL_TRAINING_BARRIER.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"evaluation requires the dual training barrier: {path}"
        )
    barrier = json.loads(path.read_text(encoding="utf-8"))
    for item in barrier.get("barriers", {}).values():
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise AssertionError(f"dataset training barrier changed: {source}")
    return barrier


def main() -> None:
    args = parse_args()
    args.daphnet_data_dir = args.daphnet_data_dir.resolve()
    args.private_data_dir = args.private_data_dir.resolve()
    args.output_root = args.output_root.resolve()
    parse_seeds(args.seeds)
    private_launcher, daphnet_launcher = configure_launchers()
    private_root = args.output_root / "private_processed_NBM_Exp"
    daphnet_root = args.output_root / "daphnet_processed_NBM"
    private_ns = private_args(args, private_root)
    daphnet_ns = daphnet_args(args, daphnet_root)

    private_seeds, private_gpus = private_launcher.validate_contract(private_ns)
    daphnet_seeds = daphnet_launcher.validate_contract(daphnet_ns)
    methods = daphnet_launcher.validate_methods(daphnet_ns.experiment_methods)
    gpu_ids = daphnet_launcher.validate_gpus(args.gpu_ids, not args.dry_run)
    if len(gpu_ids) != 8 or gpu_ids != private_gpus:
        raise ValueError("the integrated launcher requires exactly eight GPU ids")
    if private_seeds != SEEDS or daphnet_seeds != SEEDS or methods != ("FULL_C",):
        raise AssertionError("unexpected frozen experiment contract")

    if not args.dry_run:
        for data_dir in (args.daphnet_data_dir, args.private_data_dir):
            if not (data_dir / "nbm_protocol.json").is_file():
                raise FileNotFoundError(f"processed NBM protocol missing: {data_dir}")
    private_launcher.ensure_plan(private_ns, SEEDS)
    plan = ensure_dual_plan(
        args,
        private_launcher,
        daphnet_launcher,
        gpu_ids,
        private_root,
        daphnet_root,
    )

    daphnet_nbm_jobs = [
        {
            "id": f"daphnet_fold{fold}_TCN_NGM40K_seed{seed}",
            "command": daphnet_launcher.nbm_command(daphnet_ns, fold, seed),
        }
        for fold in FOLDS
        for seed in SEEDS
    ]
    daphnet_specs = [
        (fold, method, seed)
        for fold in FOLDS
        for method in methods
        for seed in SEEDS
    ]
    daphnet_train_jobs = [
        {
            "id": f"daphnet_fold{fold}_{method}_seed{seed}",
            "command": daphnet_launcher.pair_command(
                daphnet_ns, "train", fold, method, seed
            ),
        }
        for fold, method, seed in daphnet_specs
    ]
    daphnet_evaluate_jobs = [
        {
            "id": f"daphnet_fold{fold}_{method}_seed{seed}",
            "command": daphnet_launcher.pair_command(
                daphnet_ns, "evaluate", fold, method, seed
            ),
        }
        for fold, method, seed in daphnet_specs
    ]
    private_train_jobs = prefixed_jobs(
        "private",
        private_launcher.jobs(private_ns, SEEDS, "train"),
    )
    private_evaluate_jobs = prefixed_jobs(
        "private",
        private_launcher.jobs(private_ns, SEEDS, "evaluate"),
    )
    joint_train_jobs = daphnet_train_jobs + private_train_jobs
    joint_evaluate_jobs = daphnet_evaluate_jobs + private_evaluate_jobs

    print(
        f"DUAL PLAN id={plan['plan_id']} gpus={','.join(gpu_ids)} "
        f"daphnet_nbm={len(daphnet_nbm_jobs)} "
        f"joint_train={len(joint_train_jobs)} "
        f"joint_evaluate={len(joint_evaluate_jobs)}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: no model training or test inference executed", flush=True)
        print(
            "FIRST DAPHNET NBM:",
            subprocess.list2cmdline(daphnet_nbm_jobs[0]["command"]),
            flush=True,
        )
        print(
            "FIRST JOINT TRAIN:",
            subprocess.list2cmdline(joint_train_jobs[0]["command"]),
            flush=True,
        )
        print(
            "LAST JOINT TRAIN:",
            subprocess.list2cmdline(joint_train_jobs[-1]["command"]),
            flush=True,
        )
        return

    if args.phase in ("full", "train"):
        run_pool("daphnet_nbm", daphnet_nbm_jobs, gpu_ids, args.output_root)
        for fold in FOLDS:
            for seed in SEEDS:
                done = (
                    daphnet_root
                    / "nbm_source"
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / "DONE_NBM.json"
                )
                if not done.is_file():
                    raise FileNotFoundError(f"frozen Daphnet TCN-NGM missing: {done}")
        run_pool("joint_train", joint_train_jobs, gpu_ids, args.output_root)
        barrier = seal_dual_training(
            args,
            private_launcher,
            daphnet_launcher,
            private_ns,
            daphnet_ns,
        )
        print(f"DUAL TRAINING SEALED id={barrier['barrier_id']}", flush=True)
        if args.phase == "train":
            return

    if args.phase in ("full", "evaluate"):
        require_dual_barrier(args)
        run_pool("joint_evaluate", joint_evaluate_jobs, gpu_ids, args.output_root)
        env = environment()
        subprocess.run(
            daphnet_launcher.singleton(daphnet_ns, "aggregate"),
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            private_launcher.single_command(private_ns, SEEDS, "aggregate"),
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    elif args.phase == "aggregate":
        require_dual_barrier(args)
        env = environment()
        subprocess.run(
            daphnet_launcher.singleton(daphnet_ns, "aggregate"),
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            private_launcher.single_command(private_ns, SEEDS, "aggregate"),
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    print(f"COMPLETE phase={args.phase} output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
