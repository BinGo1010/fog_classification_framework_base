#!/usr/bin/env python3
"""Launch frozen Daphnet RAW+TCN test-noise evaluation on 7 or 8 GPUs.

The launcher executes 60 inference jobs (4 SNR levels x 3 folds x 5 frozen
model seeds), then aggregates them with the clean source results.  It never
starts a training job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKER = REPO_ROOT / "scripts" / "evaluate_daphnet_raw_tcn_test_noise_snr.py"
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
SNR_LEVELS = (30, 20, 10, 0)
DEFAULT_SOURCE = (
    REPO_ROOT
    / "outputs"
    / "daphnet_64Hz_raw_tcn_lr3e-3_wd1e-3_batch128_ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_SCALER_SOURCE = (
    REPO_ROOT
    / "outputs"
    / "daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161"
    / "nbm_source"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "daphnet_64Hz_raw_tcn_lr3e-3_wd1e-3_batch128_ep5pat2_test_noise_"
      "snr30_20_10_0_seedset_0_52_161_5216_52161"
)

from scripts.launch_daphnet_residual_calibration_abcd_7gpu import (
    command_text,
    run_pool,
    visible_gpu_count,
)
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import stable_json_hash
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import write_json
from scripts.run_daphnet_residual_calibration_abcd import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed_NBM"
        ),
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--scaler-source-root", type=Path, default=DEFAULT_SCALER_SOURCE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--phase", choices=("full", "evaluate", "aggregate"), default="full"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def gpu_ids(value: str, check_hardware: bool) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if len(ids) not in (7, 8) or len(ids) != len(set(ids)):
        raise ValueError("provide exactly 7 or 8 unique GPU ids")
    if any(not item.isdigit() for item in ids):
        raise ValueError("GPU ids must be non-negative integers")
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in ids):
            raise ValueError(f"requested GPUs {ids}, but only {count} are visible")
    return ids


def source_artifact_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    scaler_root = args.scaler_source_root.resolve()
    barrier = source_root / "TRAINING_BARRIER.json"
    done = source_root / "DONE.json"
    missing = [str(path) for path in (barrier, done) if not path.is_file()]
    artifacts = []
    for seed in SEEDS:
        for fold in FOLDS:
            checkpoint = (
                source_root / "runs" / f"fold_{fold}" / "method_RAW"
                / f"seed_{seed}" / "checkpoints" / "tcn.pt"
            )
            scaler = scaler_root / f"seed_{seed}" / f"fold_{fold}" / "nbm_frozen.json"
            if not checkpoint.is_file() or not scaler.is_file():
                missing.extend(
                    str(path) for path in (checkpoint, scaler) if not path.is_file()
                )
                continue
            artifacts.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "tcn_checkpoint_sha256": sha256_file(checkpoint),
                    "scaler_frozen_json_sha256": sha256_file(scaler),
                }
            )
    if missing:
        raise FileNotFoundError(f"completed frozen source missing: {missing}")
    core = {
        "training_barrier_sha256": sha256_file(barrier),
        "done_sha256": sha256_file(done),
        "artifacts": artifacts,
    }
    return {**core, "sha256": stable_json_hash(core)}


def plan_payload(
    args: argparse.Namespace,
    source_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "daphnet_raw_tcn_test_noise_plan.v1",
        "data_dir": str(args.data_dir.resolve()),
        "source_root": str(args.source_root.resolve()),
        "scaler_source_root": str(args.scaler_source_root.resolve()),
        "folds": list(FOLDS),
        "model_seeds": list(SEEDS),
        "snr_db": list(SNR_LEVELS),
        "evaluation_job_count": len(FOLDS) * len(SEEDS) * len(SNR_LEVELS),
        "training_job_count": 0,
        "batch_size": int(args.batch_size),
        "threshold_policy": "reuse each frozen clean-validation threshold",
        "noise_seed_policy": "depends on fold and SNR only, not model seed",
        "worker_sha256": sha256_file(WORKER),
        "source_artifact_manifest": source_manifest,
    }


def ensure_plan(args: argparse.Namespace, *, validate_source: bool) -> dict[str, Any]:
    if args.batch_size != 128:
        raise ValueError("this frozen source experiment requires batch_size=128")
    source_manifest = source_artifact_manifest(args) if validate_source else None
    plan = plan_payload(args, source_manifest)
    plan["plan_id"] = stable_json_hash(plan)
    path = args.output_root.resolve() / "ROBUSTNESS_PLAN.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != plan:
            raise RuntimeError(
                "existing output-root belongs to a different robustness plan; "
                "use a new --output-root"
            )
    elif not args.dry_run:
        write_json(path, plan)

    return plan


def common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--data-dir", str(args.data_dir.resolve()),
        "--source-root", str(args.source_root.resolve()),
        "--scaler-source-root", str(args.scaler_source_root.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--batch-size", str(args.batch_size),
        "--seeds", ",".join(str(value) for value in SEEDS),
        "--snr-levels", ",".join(str(value) for value in SNR_LEVELS),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def evaluate_command(
    args: argparse.Namespace, fold: int, seed: int, snr_db: int
) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--stage", "evaluate",
        "--fold", str(fold),
        "--seed", str(seed),
        "--snr-db", str(snr_db),
        "--device", "cuda",
        *common_args(args),
    ]


def aggregate_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(WORKER),
        "--stage", "aggregate",
        *common_args(args),
    ]


def evaluation_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "id": f"snr{snr_db}_fold{fold}_seed{seed}",
            "command": evaluate_command(args, fold, seed, snr_db),
        }
        for snr_db in SNR_LEVELS
        for fold in FOLDS
        for seed in SEEDS
    ]


def main() -> None:
    args = parse_args()
    ids = gpu_ids(args.gpu_ids, check_hardware=not args.dry_run)
    plan = ensure_plan(args, validate_source=not args.dry_run)
    jobs = evaluation_jobs(args)
    aggregate = aggregate_command(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"evaluation_jobs={len(jobs)} training_jobs=0 gpu_ids={ids}")
        for job in jobs:
            print(f"[{job['id']}] {command_text(job['command'])}")
        print(f"[aggregate] {command_text(aggregate)}")
        return

    root = args.output_root.resolve()
    if args.phase in ("full", "evaluate"):
        run_pool("noise_test", jobs, ids, root)
    if args.phase in ("full", "aggregate"):
        subprocess.run(aggregate, cwd=REPO_ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
