#!/usr/bin/env python3
"""Train one Conv-TCN NBM fold with a 200-epoch convergence budget.

Only role 4 fits the RobustScaler and NBM. Role 5 is unmasked validation data
for scheduling, early stopping, best-checkpoint restoration, and frozen b/sigma
calibration. Classifier and permanent-test roles are never transformed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    audit_protocol,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    ConvTCNAutoencoderNBM,
    MaskConfig,
    calibrate,
    centered_scaled_bct,
    plot_nbm_training,
    resolve_device,
    train_nbm,
)

FOLDS = (0, 1, 2)


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
        / "daphnet_conv_tcn_nbm200_bc_3seed_seed20260807"
        / "nbm_source",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=200)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--nbm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--nbm-dropout", type=float, default=0.10)
    parser.add_argument("--mask-probability", type=float, default=0.20)
    parser.add_argument("--mask-min-samples", type=int, default=4)
    parser.add_argument("--mask-max-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_args(args: argparse.Namespace) -> None:
    if args.nbm_max_epochs <= 0 or args.nbm_patience <= 0:
        raise ValueError("NBM max epochs and patience must be positive")
    if args.nbm_learning_rate <= 0:
        raise ValueError("NBM learning rate must be positive")
    if not 0.0 <= args.mask_probability <= 1.0:
        raise ValueError("mask probability must be in [0,1]")
    if not 1 <= args.mask_min_samples <= args.mask_max_samples <= 128:
        raise ValueError("mask sample range must satisfy 1 <= min <= max <= 128")


def run_fold(args: argparse.Namespace, device: torch.device) -> None:
    validate_args(args)
    output_root = args.output_root.resolve()
    fold_dir = output_root / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed NBM fold {args.fold}: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in FOLDS}
    source_audit = audit_protocol(data_dir, rows_by_fold, records)
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    scaler, scaler_points = fit_scaler_unique_role4_points(records, role4)

    mask_config = MaskConfig(
        probability=args.mask_probability,
        minimum_samples=args.mask_min_samples,
        maximum_samples=args.mask_max_samples,
    )
    probe_model = ConvTCNAutoencoderNBM(dropout=args.nbm_dropout)
    with torch.no_grad():
        probe = torch.zeros(2, 9, 128)
        latent = probe_model.encode(probe)
        reconstruction = probe_model(probe)
    if latent.shape != (2, 16, 32) or reconstruction.shape != probe.shape:
        raise AssertionError("Conv-TCN NBM shape preflight failed")

    config: dict[str, Any] = {
        "experiment": "conv_tcn_nbm_smoothl1_200epoch_for_BC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "base_seed": args.seed,
        "effective_seed": args.seed + args.fold,
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {
            str(role): int(np.sum(rows.role == role)) for role in ROLES
        },
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "scaler_unique_raw_points": scaler_points,
        "input_preprocessing": (
            "RobustScaler then per-window/per-axis mean subtraction over 128 samples"
        ),
        "architecture": probe_model.architecture_config(),
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_mask": False,
            "loss": "SmoothL1(beta=1.0) for both training and validation",
            "optimizer": (
                f"AdamW(lr={args.nbm_learning_rate},weight_decay=1e-4)"
            ),
            "scheduler": (
                "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)"
            ),
            "batch_size": 128,
            "max_epochs": args.nbm_max_epochs,
            "early_stopping_patience": args.nbm_patience,
            "gradient_clip": 1.0,
            "mask": asdict(mask_config),
            "checkpoint_rule": "lowest unmasked role-5 validation SmoothL1",
            "restore_best": True,
        },
        "classifier_or_test_roles_accessed": False,
        "source_audit": source_audit,
    }
    write_json(fold_dir / "config.json", config)
    print(
        f"PREFLIGHT NBM fold={args.fold} device={device} "
        f"latent={tuple(latent.shape)} params={config['architecture']['parameter_count']}",
        flush=True,
    )

    role4_x = centered_scaled_bct(scaler, raw_windows(records, role4))
    role5_x = centered_scaled_bct(scaler, raw_windows(records, role5))
    nbm, training = train_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed + args.fold,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
        args.nbm_learning_rate,
        args.nbm_dropout,
        mask_config,
    )
    # train_nbm has already restored the checkpoint with the lowest role-5 loss.
    bias, sigma, calibration = calibrate(nbm, role5_x, device)
    plot_nbm_training(fold_dir, training)

    frozen = {
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": scaler_points,
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "validation_mask": False,
        "best_checkpoint_restored_before_calibration": True,
        "training": {key: value for key, value in training.items() if key != "history"},
        "calibration": calibration,
        "classifier_or_test_roles_accessed": False,
    }
    write_json(fold_dir / "nbm_frozen.json", frozen)
    checkpoint = fold_dir / "checkpoints" / "conv_tcn_nbm_best.pt"
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": training["best_epoch"],
        "epochs_completed": training["epochs_completed"],
        "best_validation_huber": training["best_validation_huber"],
        "maximum_epochs": args.nbm_max_epochs,
        "patience": args.nbm_patience,
        "classifier_or_test_roles_accessed": False,
    }
    write_json(done_path, done)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run_fold(args, device)


if __name__ == "__main__":
    main()
