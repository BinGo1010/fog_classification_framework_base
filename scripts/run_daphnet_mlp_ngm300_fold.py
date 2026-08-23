#!/usr/bin/env python3
"""Train one exact-seed factorized MLP-NGM fold on Daphnet processed_NBM."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, atomic_torch_save
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.mlp_ngm_30x128 import (
    MLP_NGM_9_PARAMETER_COUNT,
    FactorizedMLPNGM9,
    architecture_config as generic_architecture_config,
    reconstruct_bct,
)
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    audit_protocol_dynamic,
    parse_csv_ints,
    raw_windows_dynamic,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    save_figure_bundle,
    write_csv,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    corrupt,
    make_loader,
    prepare_nbm_windows,
    set_seed,
)


FOLDS = (0, 1, 2)
NBM_VARIANT = "FACTORIZED_MLP_NGM_MASK4_8"
CHECKPOINT_NAME = "mlp_ngm_best.pt"
EXPERIMENT_SCHEMA = "daphnet_mlp_ngm300_source.v1"


def architecture() -> dict[str, Any]:
    config = generic_architecture_config(9)
    if int(config["parameter_count"]) != MLP_NGM_9_PARAMETER_COUNT:
        raise RuntimeError("Daphnet MLP-NGM parameter contract changed")
    return config


def augmentation_config() -> dict[str, Any]:
    return {
        "clean_probability": 0.40,
        "gaussian_probability": 0.40,
        "mask_probability": 0.20,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_all_channels": True,
    }


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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--required-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--sampling-rate-hz", type=int, default=64)
    parser.add_argument("--window-samples", type=int, default=128)
    parser.add_argument("--stride-samples", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def reconstruct_ntc(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (128, 9):
        raise ValueError(f"expected [N,128,9], got {values.shape}")
    reconstructed = reconstruct_bct(
        model, np.ascontiguousarray(values.transpose(0, 2, 1)), device, batch_size
    )
    return np.ascontiguousarray(reconstructed.transpose(0, 2, 1))


def train_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    maximum_epochs: int,
    patience: int,
) -> tuple[FactorizedMLPNGM9, dict[str, Any]]:
    set_seed(seed)
    model = FactorizedMLPNGM9(dropout=0.10).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != MLP_NGM_9_PARAMETER_COUNT:
        raise RuntimeError("Daphnet MLP-NGM parameter contract changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_loader = make_loader(train_x, 128, True, seed, num_workers)
    validation_loader = make_loader(validation_x, 128, False, seed, num_workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = fold_dir / "checkpoints" / CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean_ntc,) in train_loader:
            clean_ntc = clean_ntc.to(device, non_blocking=True)
            network_input_ntc, counts = corrupt(
                clean_ntc, augmentation_generator
            )
            mode_counts += counts
            clean_bct = clean_ntc.transpose(1, 2)
            network_input_bct = network_input_ntc.transpose(1, 2)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input_bct), clean_bct)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite MLP-NGM gradient")
            optimizer.step()
            train_total += float(loss.detach()) * len(clean_ntc)
            train_count += len(clean_ntc)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for (clean_ntc,) in validation_loader:
                clean_bct = clean_ntc.to(device, non_blocking=True).transpose(1, 2)
                loss = criterion(model(clean_bct), clean_bct)
                validation_total += float(loss) * len(clean_bct)
                validation_count += len(clean_bct)
        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": learning_rate,
                "clean_windows": int(mode_counts[0]),
                "gaussian_windows": int(mode_counts[1]),
                "masked_windows": int(mode_counts[2]),
                "improved": improved,
            }
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "schema": EXPERIMENT_SCHEMA,
                    "variant": NBM_VARIANT,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_huber": validation_loss,
                    "architecture": architecture(),
                    "augmentation": augmentation_config(),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"MLP-NGM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError("MLP-NGM checkpoint variant mismatch")
    if payload.get("architecture") != architecture():
        raise AssertionError("MLP-NGM checkpoint architecture mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    return model, {
        "seed": seed,
        "fit_windows": int(len(train_x)),
        "calibration_validation_windows": int(len(validation_x)),
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": parameter_count,
        "history": history,
    }


def calibrate(
    model: nn.Module,
    calibration_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct_ntc(model, calibration_x, device)
    error = calibration_x - reconstruction
    bias = np.median(error, axis=(0, 1)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(
        np.abs(error - bias[None, None, :]), axis=(0, 1)
    )
    sigma = np.maximum(sigma_raw, 0.05).astype(np.float32)
    return bias, sigma, {
        "bias": bias.astype(float).tolist(),
        "sigma_raw": sigma_raw.astype(float).tolist(),
        "sigma": sigma.astype(float).tolist(),
        "sigma_floor": 0.05,
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05)
        .astype(int)
        .tolist(),
        "calibration_windows": int(len(calibration_x)),
    }


def plot_training(fold_dir: Path, training: dict[str, Any]) -> None:
    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(
        epochs,
        [row["validation_huber"] for row in history],
        label="Role 5 validation",
    )
    ax.axvline(training["best_epoch"], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss", title="Factorized MLP-NGM")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, fold_dir / "mlp_ngm_training_validation")
    plt.close(fig)


def validate_existing(
    fold_dir: Path,
    args: argparse.Namespace,
    scientific_sha256: str,
) -> None:
    done_path = fold_dir / "DONE_NBM.json"
    frozen_path = fold_dir / "nbm_frozen.json"
    scaler_path = fold_dir / "scaler_role4.json"
    checkpoint = fold_dir / "checkpoints" / CHECKPOINT_NAME
    for path in (done_path, frozen_path, scaler_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete MLP-NGM artifacts: {path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected = {
        "status": "frozen",
        "fold": args.fold,
        "seed": args.seed,
        "maximum_epochs": 300,
        "patience": 20,
        "scientific_data_sha256": scientific_sha256,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise AssertionError(f"stale MLP-NGM DONE_NBM {key}: {done.get(key)!r}")
    training = frozen["training"]
    if training.get("architecture") != architecture():
        raise AssertionError("stale MLP-NGM architecture")
    if frozen.get("scientific_data_sha256") != scientific_sha256:
        raise AssertionError("MLP-NGM scientific dataset changed")
    if done.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise AssertionError("MLP-NGM checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError("MLP-NGM checkpoint variant mismatch")
    if int(payload.get("seed", -1)) != args.seed:
        raise AssertionError("MLP-NGM checkpoint seed mismatch")


def run(args: argparse.Namespace, device: torch.device) -> None:
    required_seeds = parse_csv_ints(args.required_seeds)
    if tuple(required_seeds) != (0, 52, 161, 5216, 52161):
        raise ValueError("this experiment requires seeds 0,52,161,5216,52161")
    if args.seed not in required_seeds:
        raise ValueError(f"seed must be one of {required_seeds}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("MLP-NGM requires max_epoch=300 and patience=20")
    if (
        args.sampling_rate_hz != 64
        or args.window_samples != 128
        or args.stride_samples != 64
    ):
        raise ValueError("Daphnet contract requires 64 Hz, 128 samples, stride 64")

    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    data_dir = args.data_dir.resolve()
    scientific_data = processed_nbm_scientific_manifest(data_dir)
    if done_path.exists() and not args.overwrite:
        validate_existing(fold_dir, args, scientific_data["sha256"])
        print(f"SKIP completed MLP-NGM fold/seed: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)

    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in FOLDS}
    source_audit = audit_protocol_dynamic(
        data_dir,
        rows_by_fold,
        args.sampling_rate_hz,
        args.window_samples,
        args.stride_samples,
    )
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
    atomic_json_dump(
        {
            "fold": args.fold,
            "seed": args.seed,
            "scaler_fit_role": 4,
            "scaler_unique_raw_points": unique_points,
            "scaler": scaler.as_dict(),
            "scientific_data_sha256": scientific_data["sha256"],
        },
        fold_dir / "scaler_role4.json",
    )
    role4_x = prepare_nbm_windows(
        scaler, raw_windows_dynamic(records, role4, 128), center=True
    )
    role5_x = prepare_nbm_windows(
        scaler, raw_windows_dynamic(records, role5, 128), center=True
    )
    config = {
        "experiment": "MLP_NGM300_schemeC_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact seed; no fold offset",
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "input_preprocessing": "RobustScaler then per-window/per-axis mean subtraction",
        "architecture": architecture(),
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_augmentation": False,
            "augmentation": augmentation_config(),
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
        "scientific_data_sha256": scientific_data["sha256"],
    }
    atomic_json_dump(config, fold_dir / "config.json")
    model, run_payload = train_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
    )
    _, _, calibration = calibrate(model, role5_x, device)
    plot_training(fold_dir, run_payload)
    training = {
        **{key: value for key, value in run_payload.items() if key != "history"},
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "augmentation": augmentation_config(),
        "architecture": architecture(),
    }
    checkpoint = fold_dir / "checkpoints" / CHECKPOINT_NAME
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
        "scientific_data_sha256": scientific_data["sha256"],
    }
    atomic_json_dump(frozen, fold_dir / "nbm_frozen.json")
    write_csv(fold_dir / "logs" / "mlp_ngm_history.csv", run_payload["history"])
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": run_payload["best_epoch"],
        "epochs_completed": run_payload["epochs_completed"],
        "best_validation_huber": run_payload["best_validation_huber"],
        "maximum_epochs": 300,
        "patience": 20,
        "classifier_or_test_roles_accessed": False,
        "scientific_data_sha256": scientific_data["sha256"],
    }
    atomic_json_dump(done, done_path)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run(args, device)


if __name__ == "__main__":
    main()
