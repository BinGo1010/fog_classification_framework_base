#!/usr/bin/env python3
"""Train one Conv-TCN NBM fold with dynamic 40/40/20 corruption and composite loss.

Role 4 alone fits the RobustScaler and updates the NBM. Every epoch assigns
each role-4 window to exactly one stochastic branch: 40% clean, 40% additive
Gaussian noise (std=0.04), or 20% all-axis time masking. Role 5 is always
uncorrupted and uses the same composite loss for scheduling, early stopping,
and best-checkpoint selection. The restored best model then supplies role-5
errors for frozen b/sigma calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
    set_seed,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    ConvTCNAutoencoderNBM,
    calibrate,
    centered_scaled_bct,
    nbm_loader,
    resolve_device,
)

FOLDS = (0, 1, 2)


@dataclass(frozen=True)
class DynamicAugmentationConfig:
    clean_probability: float = 0.40
    gaussian_probability: float = 0.40
    mask_probability: float = 0.20
    gaussian_std: float = 0.04
    mask_minimum_samples: int = 4
    mask_maximum_samples: int = 8
    mask_all_channels: bool = True
    assignment: str = "mutually_exclusive_per_window_resampled_every_epoch"

    def validate(self) -> None:
        probabilities = (
            self.clean_probability,
            self.gaussian_probability,
            self.mask_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must be in [0,1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("clean/Gaussian/Mask probabilities must sum to 1")
        if self.gaussian_std <= 0.0:
            raise ValueError("Gaussian standard deviation must be positive")
        if not 1 <= self.mask_minimum_samples <= self.mask_maximum_samples <= 128:
            raise ValueError("mask range must satisfy 1 <= min <= max <= 128")


@dataclass(frozen=True)
class CompositeLossConfig:
    smoothl1_weight: float = 0.70
    correlation_weight: float = 0.15
    first_difference_weight: float = 0.15
    smoothl1_beta: float = 1.0
    correlation_epsilon: float = 1e-8

    def validate(self) -> None:
        weights = (
            self.smoothl1_weight,
            self.correlation_weight,
            self.first_difference_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("composite loss weights must sum to 1")
        if self.smoothl1_beta <= 0.0 or self.correlation_epsilon <= 0.0:
            raise ValueError("loss beta and epsilon must be positive")


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
        / "daphnet_conv_tcn_nbm200_composite_C_3seed_seed20260807"
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
    parser.add_argument("--gaussian-std", type=float, default=0.04)
    parser.add_argument("--mask-min-samples", type=int, default=4)
    parser.add_argument("--mask-max-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dynamic_corruption(
    clean: torch.Tensor,
    config: DynamicAugmentationConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return corrupted input and modes: 0=clean, 1=Gaussian, 2=Mask."""
    config.validate()
    batch_size, _, time_samples = clean.shape
    draws = torch.rand(batch_size, device=clean.device, generator=generator)
    gaussian_boundary = config.clean_probability + config.gaussian_probability
    gaussian = (draws >= config.clean_probability) & (draws < gaussian_boundary)
    masked = draws >= gaussian_boundary
    modes = torch.zeros(batch_size, dtype=torch.int64, device=clean.device)
    modes[gaussian] = 1
    modes[masked] = 2
    corrupted = clean.clone()

    gaussian_indices = torch.nonzero(gaussian, as_tuple=False).flatten()
    if len(gaussian_indices):
        noise = torch.randn(
            (len(gaussian_indices), clean.shape[1], time_samples),
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        corrupted[gaussian_indices] += config.gaussian_std * noise

    for index in torch.nonzero(masked, as_tuple=False).flatten().tolist():
        length = int(
            torch.randint(
                config.mask_minimum_samples,
                config.mask_maximum_samples + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        start = int(
            torch.randint(
                0,
                time_samples - length + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        corrupted[index, :, start : start + length] = 0.0
    return corrupted, modes


def composite_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: CompositeLossConfig,
) -> dict[str, torch.Tensor]:
    config.validate()
    smoothl1 = F.smooth_l1_loss(
        prediction, target, beta=config.smoothl1_beta, reduction="mean"
    )
    prediction_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = torch.sum(prediction_centered * target_centered, dim=-1)
    # Clamp each squared norm before sqrt. Clamping only after sqrt leaves the
    # backward path through sqrt(0), which can create inf/NaN gradients for a
    # constant prediction or target window even though the forward value is
    # finite.
    prediction_norm = torch.sqrt(
        torch.sum(prediction_centered.square(), dim=-1).clamp_min(
            config.correlation_epsilon
        )
    )
    target_norm = torch.sqrt(
        torch.sum(target_centered.square(), dim=-1).clamp_min(
            config.correlation_epsilon
        )
    )
    denominator = (prediction_norm * target_norm).clamp_min(
        config.correlation_epsilon
    )
    correlation = (numerator / denominator).clamp(-1.0, 1.0)
    correlation_loss = (1.0 - correlation).mean()
    prediction_difference = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_difference = target[:, :, 1:] - target[:, :, :-1]
    first_difference = F.smooth_l1_loss(
        prediction_difference,
        target_difference,
        beta=config.smoothl1_beta,
        reduction="mean",
    )
    total = (
        config.smoothl1_weight * smoothl1
        + config.correlation_weight * correlation_loss
        + config.first_difference_weight * first_difference
    )
    return {
        "total": total,
        "smoothl1": smoothl1,
        "correlation": correlation_loss,
        "first_difference": first_difference,
    }


def train_nbm_composite(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    dropout: float,
    augmentation: DynamicAugmentationConfig,
    loss_config: CompositeLossConfig,
) -> tuple[ConvTCNAutoencoderNBM, dict[str, Any]]:
    augmentation.validate()
    loss_config.validate()
    set_seed(seed)
    model = ConvTCNAutoencoderNBM(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    train_loader = nbm_loader(train_x, True, seed, num_workers)
    validation_loader = nbm_loader(validation_x, False, seed, num_workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint = fold_dir / "checkpoints" / "conv_tcn_nbm_best.pt"
    component_names = ("total", "smoothl1", "correlation", "first_difference")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sums = {name: 0.0 for name in component_names}
        augmentation_counts = {"clean": 0, "gaussian": 0, "mask": 0}
        train_n = 0
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            corrupted, modes = dynamic_corruption(
                clean, augmentation, augmentation_generator
            )
            optimizer.zero_grad(set_to_none=True)
            components = composite_reconstruction_loss(
                model(corrupted), clean, loss_config
            )
            if any(
                not bool(torch.isfinite(value).item())
                for value in components.values()
            ):
                values = {
                    name: float(value.detach()) for name, value in components.items()
                }
                raise FloatingPointError(
                    f"non-finite Conv-TCN NBM loss components: {values}"
                )
            components["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Conv-TCN NBM gradient")
            optimizer.step()
            count = len(clean)
            train_n += count
            for name in component_names:
                train_sums[name] += float(components[name].detach()) * count
            augmentation_counts["clean"] += int(torch.sum(modes == 0).item())
            augmentation_counts["gaussian"] += int(torch.sum(modes == 1).item())
            augmentation_counts["mask"] += int(torch.sum(modes == 2).item())

        model.eval()
        validation_sums = {name: 0.0 for name in component_names}
        validation_n = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
                # Role 5 is intentionally complete and uncorrupted.
                components = composite_reconstruction_loss(
                    model(clean), clean, loss_config
                )
                count = len(clean)
                validation_n += count
                for name in component_names:
                    validation_sums[name] += float(components[name]) * count

        train_means = {name: train_sums[name] / train_n for name in component_names}
        validation_means = {
            name: validation_sums[name] / validation_n for name in component_names
        }
        scheduler.step(validation_means["total"])
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        improved = validation_means["total"] < best_loss - 1e-10
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_means.items()},
            **{
                f"validation_{name}": value
                for name, value in validation_means.items()
            },
            "learning_rate": learning_rate_now,
            "clean_windows": augmentation_counts["clean"],
            "gaussian_windows": augmentation_counts["gaussian"],
            "mask_windows": augmentation_counts["mask"],
            "improved": improved,
        }
        history.append(row)
        if improved:
            best_loss = validation_means["total"]
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_composite_loss": best_loss,
                    "validation_components": validation_means,
                    "seed": seed,
                    "architecture": model.architecture_config(),
                    "augmentation": asdict(augmentation),
                    "loss": asdict(loss_config),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"NBM fold={fold_dir.name} epoch={epoch:03d} "
            f"train={train_means['total']:.7f} val={validation_means['total']:.7f} "
            f"val_s1={validation_means['smoothl1']:.7f} "
            f"val_corr={validation_means['correlation']:.7f} "
            f"val_diff={validation_means['first_difference']:.7f} "
            f"lr={learning_rate_now:.2e} aug={augmentation_counts} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(fold_dir / "logs" / "conv_tcn_nbm_history.csv", history)
    return model, {
        "seed": seed,
        "fit_windows": len(train_x),
        "validation_windows": len(validation_x),
        "maximum_epochs": max_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selection_metric": "role5_unmasked_composite_loss",
        "optimizer": f"AdamW(lr={learning_rate}, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "loss": (
            "0.70*SmoothL1(beta=1.0)+0.15*Corr(1-Pearson)+"
            "0.15*FirstDifferenceSmoothL1(beta=1.0)"
        ),
        "loss_config": asdict(loss_config),
        "augmentation": asdict(augmentation),
        "validation_augmentation": "none",
        "architecture": model.architecture_config(),
        "history": history,
    }


def plot_training(fold_dir: Path, training: dict[str, Any]) -> None:
    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(epochs, [row["train_total"] for row in history], label="Train")
    axes[0].plot(
        epochs, [row["validation_total"] for row in history], label="Role 5"
    )
    axes[0].axvline(training["best_epoch"], color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Composite loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for name, label in (
        ("smoothl1", "SmoothL1"),
        ("correlation", "Correlation"),
        ("first_difference", "First difference"),
    ):
        axes[1].plot(
            epochs, [row[f"validation_{name}"] for row in history], label=label
        )
    axes[1].axvline(training["best_epoch"], color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Role-5 loss components", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    save_figure_bundle(fig, fold_dir / "conv_tcn_nbm_training_validation")


def run_fold(args: argparse.Namespace, device: torch.device) -> None:
    if args.nbm_max_epochs <= 0 or args.nbm_patience <= 0:
        raise ValueError("NBM max epochs and patience must be positive")
    if args.nbm_learning_rate <= 0.0:
        raise ValueError("NBM learning rate must be positive")
    augmentation = DynamicAugmentationConfig(
        gaussian_std=args.gaussian_std,
        mask_minimum_samples=args.mask_min_samples,
        mask_maximum_samples=args.mask_max_samples,
    )
    loss_config = CompositeLossConfig()
    augmentation.validate()
    loss_config.validate()
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

    probe_model = ConvTCNAutoencoderNBM(dropout=args.nbm_dropout)
    with torch.no_grad():
        probe = torch.zeros(2, 9, 128)
        latent = probe_model.encode(probe)
        reconstruction = probe_model(probe)
    if latent.shape != (2, 16, 32) or reconstruction.shape != probe.shape:
        raise AssertionError("Conv-TCN NBM shape preflight failed")

    config = {
        "experiment": "conv_tcn_nbm_composite_dynamic_augmentation_for_C",
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
            "validation_augmentation": "none",
            "augmentation": asdict(augmentation),
            "loss": asdict(loss_config),
            "loss_formula": (
                "0.70*SmoothL1+0.15*(1-PearsonCorr_t)+"
                "0.15*SmoothL1(first differences)"
            ),
            "optimizer": f"AdamW(lr={args.nbm_learning_rate},weight_decay=1e-4)",
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
            "batch_size": 128,
            "max_epochs": args.nbm_max_epochs,
            "early_stopping_patience": args.nbm_patience,
            "gradient_clip": 1.0,
            "checkpoint_rule": "lowest unmasked role-5 composite loss",
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
    nbm, training = train_nbm_composite(
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
        augmentation,
        loss_config,
    )
    bias, sigma, calibration = calibrate(nbm, role5_x, device)
    plot_training(fold_dir, training)
    write_json(
        fold_dir / "nbm_frozen.json",
        {
            "scaler": scaler.as_dict(),
            "scaler_fit_role": 4,
            "scaler_unique_raw_points": scaler_points,
            "nbm_train_role": 4,
            "nbm_earlystop_and_calibration_role": 5,
            "validation_mask": False,
            "best_checkpoint_restored_before_calibration": True,
            "training": {
                key: value for key, value in training.items() if key != "history"
            },
            "calibration": calibration,
            "classifier_or_test_roles_accessed": False,
        },
    )
    checkpoint = fold_dir / "checkpoints" / "conv_tcn_nbm_best.pt"
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": training["best_epoch"],
        "epochs_completed": training["epochs_completed"],
        "best_validation_loss": training["best_validation_loss"],
        "selection_metric": training["selection_metric"],
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
