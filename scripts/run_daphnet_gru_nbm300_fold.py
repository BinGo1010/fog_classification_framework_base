#!/usr/bin/env python3
"""Train one exact-seed GRU reconstruction NBM fold for the paired comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    save_figure_bundle,
    write_csv,
    write_json,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    calibrate,
    prepare_nbm_windows,
    train_nbm,
)

FOLDS = (0, 1, 2)
REQUIRED_SEEDS = (0, 52, 161)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_gru_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161"
        / "nbm_source"
        / "seed_0",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=REQUIRED_SEEDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--nbm-hidden", type=int, default=64)
    parser.add_argument("--nbm-bottleneck", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def architecture(hidden: int, bottleneck: int) -> dict[str, Any]:
    model = GRUReconstructionNBM(channels=9, hidden=hidden, bottleneck=bottleneck)
    return {
        "name": "gru_reconstruction_nbm_v1",
        "input_shape": ["B", 128, 9],
        "encoder": f"one-layer unidirectional GRU(9,{hidden})",
        "encoder_summary": "last hidden state",
        "bottleneck": f"Linear({hidden},{bottleneck})",
        "latent_shape": ["B", bottleneck],
        "decoder_initial_state": f"Linear({bottleneck},{hidden})",
        "decoder": f"one-layer unidirectional GRU(9,{hidden}) with 128-step all-zero input",
        "output": f"Linear({hidden},9), no output activation",
        "skip_connections": False,
        "output_shape": ["B", 128, 9],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def plot_training(fold_dir: Path, run: dict[str, Any]) -> None:
    history = run["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(epochs, [row["validation_huber"] for row in history], label="Role 5 validation")
    ax.axvline(run["summary"]["best_epoch"], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss", title="GRU reconstruction NBM")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, fold_dir / "gru_nbm_training_validation")
    plt.close(fig)


def run(args: argparse.Namespace, device: torch.device) -> None:
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this comparison requires GRU-NBM max_epoch=300 and patience=20")
    if args.nbm_hidden != 64 or args.nbm_bottleneck != 16:
        raise ValueError("the retained GRU architecture requires hidden=64 and bottleneck=16")
    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed GRU-NBM fold/seed: {done_path}", flush=True)
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
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
    role4_x = prepare_nbm_windows(scaler, raw_windows(records, role4), center=True)
    role5_x = prepare_nbm_windows(scaler, raw_windows(records, role5), center=True)
    model_config = architecture(args.nbm_hidden, args.nbm_bottleneck)
    config = {
        "experiment": "GRU_NBM300_schemeC_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact seed; no fold offset",
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "input_preprocessing": "RobustScaler then per-window/per-axis mean subtraction",
        "architecture": model_config,
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_augmentation": False,
            "augmentation": "40% clean, 40% Gaussian(std=0.04), 20% all-axis time mask(4..8)",
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
            "batch_size": 128,
            "maximum_epochs": 300,
            "patience": 20,
            "gradient_clip": 1.0,
            "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
            "restore_best": True,
        },
        "classifier_or_test_roles_accessed": False,
        "source_audit": source_audit,
    }
    write_json(fold_dir / "config.json", config)
    model, run_payload = train_nbm(
        "gru_nbm_best",
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        max_epochs=args.nbm_max_epochs,
        patience=args.nbm_patience,
        bottleneck=args.nbm_bottleneck,
    )
    bias, sigma, calibration = calibrate(model, role5_x, device)
    plot_training(fold_dir, run_payload)
    summary = run_payload["summary"]
    training = {
        **summary,
        "maximum_epochs": 300,
        "patience": 20,
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "augmentation": {
            "clean_probability": 0.40,
            "gaussian_probability": 0.40,
            "mask_probability": 0.20,
            "gaussian_std": 0.04,
            "mask_minimum_samples": 4,
            "mask_maximum_samples": 8,
            "mask_all_channels": True,
        },
        "architecture": model_config,
    }
    checkpoint = fold_dir / "checkpoints" / "gru_nbm_best.pt"
    frozen = {
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_points,
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "validation_mask_or_noise": False,
        "best_checkpoint_restored_before_calibration": True,
        "training": training,
        "calibration": calibration,
        "classifier_or_test_roles_accessed": False,
    }
    write_json(fold_dir / "nbm_frozen.json", frozen)
    write_csv(fold_dir / "logs" / "gru_nbm_history.csv", run_payload["history"])
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "best_validation_huber": summary["best_validation_huber"],
        "maximum_epochs": 300,
        "patience": 20,
        "classifier_or_test_roles_accessed": False,
    }
    write_json(done_path, done)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run(args, device)


if __name__ == "__main__":
    main()
