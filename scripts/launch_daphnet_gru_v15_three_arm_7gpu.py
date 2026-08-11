#!/usr/bin/env python3
"""Seven-GPU launcher for RAW / GRU-v1-C / GRU-v1.5-dec96-C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_daphnet_residual_calibration_abcd_7gpu import (
    command_text,
    parse_seed_list,
    run_pool,
    visible_gpu_count,
)
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest

GRU_V1_WORKER = REPO_ROOT / "scripts" / "run_daphnet_gru_nbm300_fold.py"
GRU_V15_WORKER = REPO_ROOT / "scripts" / "run_daphnet_gru_v15_nbm300_fold.py"
PAIR_WORKER = REPO_ROOT / "scripts" / "run_daphnet_gru_v15_three_arm.py"
CRITICAL_CODE_PATHS = (
    GRU_V1_WORKER,
    GRU_V15_WORKER,
    PAIR_WORKER,
    Path(__file__).resolve(),
    REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "cnbr_fog" / "evaluation.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
)
FOLDS = (0, 1, 2)
METHODS = ("RAW", "GRU_V1_C", "GRU_V15_C")
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
SEED_TEXT = "0,52,161,5216,52161"
EXPERIMENT_ID = (
    "daphnet_gru_v15_dec96_three_arm_nbm300_C_tcn_ep5pat2_"
    "seedset_0_52_161_5216_52161"
)


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
        "--output-root", type=Path, default=REPO_ROOT / "outputs" / EXPERIMENT_ID
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6")
    parser.add_argument("--nbm-seeds", default=SEED_TEXT)
    parser.add_argument("--tcn-seeds", default=SEED_TEXT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("full", "nbm", "train", "evaluate", "aggregate"),
        default="full",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> tuple[int, ...]:
    nbm_seeds = parse_seed_list(args.nbm_seeds)
    tcn_seeds = parse_seed_list(args.tcn_seeds)
    if nbm_seeds != REQUIRED_SEEDS or tcn_seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires paired seeds {SEED_TEXT}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("both GRU NBMs require max_epoch=300 and patience=20")
    if args.tcn_max_epochs != 5 or args.tcn_patience != 2:
        raise ValueError("all TCN arms require max_epoch=5 and patience=2")
    return nbm_seeds


def validate_gpus(value: str, check_hardware: bool) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if (
        not gpu_ids
        or len(gpu_ids) != len(set(gpu_ids))
        or any(not item.isdigit() for item in gpu_ids)
    ):
        raise ValueError(f"invalid unique GPU ids: {value}")
    if check_hardware:
        count = visible_gpu_count()
        if any(int(item) >= count for item in gpu_ids):
            raise ValueError(f"requested {gpu_ids}, but nvidia-smi reports {count} GPUs")
    return gpu_ids


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_root(root: Path, kind: str, seed: int) -> Path:
    return root / "nbm_source" / kind / f"seed_{seed}"


def nbm_command(
    args: argparse.Namespace,
    kind: str,
    fold: int,
    seed: int,
) -> list[str]:
    root = args.output_root.resolve()
    if kind == "gru_v1":
        command = [
            args.python,
            str(GRU_V1_WORKER),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-root",
            str(source_root(root, kind, seed)),
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--required-seeds",
            SEED_TEXT,
            "--sampling-rate-hz",
            "64",
            "--window-samples",
            "128",
            "--stride-samples",
            "64",
            "--device",
            "cuda",
            "--num-workers",
            str(args.num_workers),
            "--nbm-max-epochs",
            "300",
            "--nbm-patience",
            "20",
            "--nbm-hidden",
            "64",
            "--nbm-bottleneck",
            "16",
        ]
    elif kind == "gru_v15_decoder96":
        command = [
            args.python,
            str(GRU_V15_WORKER),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-root",
            str(source_root(root, kind, seed)),
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--required-seeds",
            SEED_TEXT,
            "--device",
            "cuda",
            "--num-workers",
            str(args.num_workers),
            "--nbm-max-epochs",
            "300",
            "--nbm-patience",
            "20",
        ]
    else:
        raise ValueError(f"unsupported NBM kind: {kind}")
    if args.overwrite:
        command.append("--overwrite")
    return command


def pair_common(
    args: argparse.Namespace,
    v1_source: Path,
    v15_source: Path,
) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--gru-v1-source-root",
        str(v1_source.resolve()),
        "--gru-v15-source-root",
        str(v15_source.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--nbm-seeds",
        SEED_TEXT,
        "--tcn-seeds",
        SEED_TEXT,
        "--required-seeds",
        SEED_TEXT,
        "--sampling-rate-hz",
        "64",
        "--window-samples",
        "128",
        "--stride-samples",
        "64",
        "--num-workers",
        str(args.num_workers),
        "--tcn-max-epochs",
        "5",
        "--tcn-patience",
        "2",
        "--required-nbm-max-epochs",
        "300",
        "--required-nbm-patience",
        "20",
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def pair_command(
    args: argparse.Namespace,
    stage: str,
    fold: int,
    method: str,
    seed: int,
) -> list[str]:
    root = args.output_root.resolve()
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *pair_common(
            args,
            source_root(root, "gru_v1", seed),
            source_root(root, "gru_v15_decoder96", seed),
        ),
        "--fold",
        str(fold),
        "--method",
        method,
        "--nbm-seed",
        str(seed),
        "--tcn-seed",
        str(seed),
        "--device",
        "cuda",
    ]


def singleton(args: argparse.Namespace, stage: str) -> list[str]:
    root = args.output_root.resolve()
    return [
        args.python,
        str(PAIR_WORKER),
        "--stage",
        stage,
        *pair_common(
            args,
            root / "nbm_source" / "gru_v1",
            root / "nbm_source" / "gru_v15_decoder96",
        ),
    ]


def verify_existing_output_identity(root: Path, plan: dict[str, object]) -> None:
    launch_plan = root / "logs" / "launch_plan.json"
    experiment_config = root / "experiment_config.json"
    material = (
        root / "nbm_source",
        root / "runs",
        root / "TRAINING_BARRIER.json",
        root / "DONE.json",
    )
    if (
        root.exists()
        and any(path.exists() for path in material)
        and not launch_plan.exists()
        and not experiment_config.exists()
    ):
        raise RuntimeError(
            "output-root contains unverifiable artifacts; choose a clean output-root"
        )
    if launch_plan.exists():
        previous = json.loads(launch_plan.read_text(encoding="utf-8"))
        for key in (
            "experiment_id",
            "dataset",
            "sampling_rate_hz",
            "window_samples",
            "stride_samples",
            "methods",
            "nbm_seeds",
            "tcn_seeds",
            "nbm_architectures",
            "nbm_training",
            "classifier_training",
            "code_sha256",
            "scientific_data_sha256",
        ):
            if previous.get(key) != plan.get(key):
                raise RuntimeError(
                    f"output-root identity mismatch for {key}: "
                    f"existing={previous.get(key)!r}, requested={plan.get(key)!r}"
                )
    if experiment_config.exists():
        frozen = json.loads(experiment_config.read_text(encoding="utf-8"))
        if frozen.get("experiment") != "strict_three_arm_GRU_v1_vs_GRU_v15_decoder96_vs_RAW":
            raise RuntimeError("output-root belongs to another experiment")


def main() -> None:
    args = parse_args()
    seeds = validate_contract(args)
    gpu_ids = validate_gpus(args.gpu_ids, not args.dry_run)
    root = args.output_root.resolve()
    nbm_specs = [
        (kind, fold, seed)
        for kind in ("gru_v1", "gru_v15_decoder96")
        for fold in FOLDS
        for seed in seeds
    ]
    nbm_jobs = [
        {
            "id": f"fold{fold}_{kind}_seed{seed}",
            "command": nbm_command(args, kind, fold, seed),
        }
        for kind, fold, seed in nbm_specs
    ]
    classifier_specs = [
        (fold, method, seed)
        for fold in FOLDS
        for method in METHODS
        for seed in seeds
    ]
    train_jobs = [
        {
            "id": f"fold{fold}_{method}_seed{seed}",
            "command": pair_command(args, "train", fold, method, seed),
        }
        for fold, method, seed in classifier_specs
    ]
    evaluate_jobs = [
        {
            "id": f"fold{fold}_{method}_seed{seed}",
            "command": pair_command(args, "evaluate", fold, method, seed),
        }
        for fold, method, seed in classifier_specs
    ]
    code_sha256 = {
        path.relative_to(REPO_ROOT).as_posix(): file_sha256(path)
        for path in CRITICAL_CODE_PATHS
    }
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    plan: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "strategy": (
            "7-GPU queue; 30 NBMs; 45 paired classifiers; one strict global "
            "barrier; 45 permanent-test jobs"
        ),
        "dataset": str(args.data_dir.resolve()),
        "scientific_data_sha256": scientific_data["sha256"],
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "gpu_ids": gpu_ids,
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "nbm_architectures": {
            "GRU_V1_C": {
                "encoder_hidden": 64,
                "bottleneck": 16,
                "decoder_hidden": 64,
                "parameter_count": 31_513,
            },
            "GRU_V15_C": {
                "encoder_hidden": 64,
                "bottleneck": 16,
                "decoder_hidden": 96,
                "parameter_count": 48_761,
            },
        },
        "single_architecture_variable": "decoder hidden 64 -> 96",
        "nbm_seeds": list(seeds),
        "tcn_seeds": list(seeds),
        "seed_policy": "exact paired seeds; no fold offset",
        "nbm_jobs": len(nbm_jobs),
        "classifier_train_jobs": len(train_jobs),
        "post_barrier_test_jobs": len(evaluate_jobs),
        "nbm_training": (
            "max300/pat20, SmoothL1, AdamW lr1e-3, 40/40/20 augmentation"
        ),
        "classifier_training": (
            "unchanged TCN, max5/pat2, weighted BCE, AdamW lr1e-3"
        ),
        "threshold": "roles 2/3 max balanced accuracy; unchanged across arms",
        "inputs": {
            "RAW": "centered scaled raw [B,9,128]",
            "GRU_V1_C": "scheme C [r,abs(r),delta] [B,27,128]",
            "GRU_V15_C": "scheme C [r,abs(r),delta] [B,27,128]",
        },
        "pre_registered_success": {
            "comparison": "GRU_V15_C minus GRU_V1_C",
            "sensitivity_mean_delta_min": 0.010,
            "auprc_mean_delta_min": -0.005,
            "specificity_mean_delta_min": -0.010,
            "precision_mean_delta_min": -0.010,
            "sensitivity_positive_seed_count_min": 4,
        },
        "code_sha256": code_sha256,
        "example_nbm_v1": command_text(
            next(job["command"] for job in nbm_jobs if "gru_v1_" in job["id"])
        ),
        "example_nbm_v15": command_text(
            next(
                job["command"]
                for job in nbm_jobs
                if "gru_v15_decoder96" in job["id"]
            )
        ),
        "example_train": command_text(train_jobs[0]["command"]),
        "seal": command_text(singleton(args, "seal")),
        "example_test": command_text(evaluate_jobs[0]["command"]),
        "aggregate": command_text(singleton(args, "aggregate")),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not (args.data_dir.resolve() / "nbm_protocol.json").is_file():
        raise FileNotFoundError(
            f"processed_NBM protocol missing: {args.data_dir.resolve()}"
        )
    verify_existing_output_identity(root, plan)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )

    if args.phase in ("full", "nbm"):
        run_pool("nbm", nbm_jobs, gpu_ids, root)
        if args.phase == "nbm":
            print(f"NBM STAGE COMPLETE output={root / 'nbm_source'}", flush=True)
            return
    if args.phase in ("full", "train"):
        for kind in ("gru_v1", "gru_v15_decoder96"):
            for fold in FOLDS:
                for seed in seeds:
                    done = source_root(root, kind, seed) / f"fold_{fold}" / "DONE_NBM.json"
                    if not done.is_file():
                        raise FileNotFoundError(f"NBM not frozen: {done}")
        run_pool("train", train_jobs, gpu_ids, root)
        subprocess.run(
            singleton(args, "seal"), cwd=REPO_ROOT, env=environment, check=True
        )
        if args.phase == "train":
            print(f"TRAINING SEALED output={root}", flush=True)
            return
    if args.phase in ("full", "evaluate"):
        if not (root / "TRAINING_BARRIER.json").is_file():
            raise FileNotFoundError("evaluation requires TRAINING_BARRIER.json")
        run_pool("evaluate", evaluate_jobs, gpu_ids, root)
        subprocess.run(
            singleton(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True
        )
    elif args.phase == "aggregate":
        subprocess.run(
            singleton(args, "aggregate"), cwd=REPO_ROOT, env=environment, check=True
        )
    print(f"COMPLETE phase={args.phase} output={root}", flush=True)


if __name__ == "__main__":
    main()
