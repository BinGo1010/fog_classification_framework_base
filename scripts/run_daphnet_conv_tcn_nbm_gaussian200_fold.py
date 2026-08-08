#!/usr/bin/env python3
"""Train one Conv-TCN NBM with GRU-matched 40/40/20 augmentation.

The only intended change from the preceding 200-epoch Conv-TCN NBM is its
role-4 input augmentation: 40% clean windows, 40% additive Gaussian-noise
windows (std=0.04), and 20% short all-axis time-mask windows. Role 5 is always
unmasked and selects/restores the lowest-validation-SmoothL1 checkpoint before
b/sigma calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

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
    set_seed,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    ConvTCNAutoencoderNBM,
    atomic_torch_save,
    calibrate,
    centered_scaled_bct,
    nbm_loader,
    plot_nbm_training,
    resolve_device,
)

FOLDS = (0, 1, 2)


@dataclass(frozen=True)
class GRUMatchedAugmentation:
    clean_probability: float = 0.40
    gaussian_probability: float = 0.40
    mask_probability: float = 0.20
    gaussian_std: float = 0.04
    mask_minimum_samples: int = 4
    mask_maximum_samples: int = 8
    mask_all_channels: bool = True

    def validate(self) -> None:
        probabilities = (
            self.clean_probability,
            self.gaussian_probability,
            self.mask_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must be in [0,1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("clean/Gaussian/mask probabilities must sum to 1")
        if self.gaussian_std < 0.0:
            raise ValueError("Gaussian standard deviation must be non-negative")
        if not 1 <= self.mask_minimum_samples <= self.mask_maximum_samples <= 128:
            raise ValueError("mask range must satisfy 1 <= min <= max <= 128")


def gru_matched_augmentation(
    clean: torch.Tensor,
    config: GRUMatchedAugmentation,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Assign each window exclusively to clean, Gaussian, or time-mask input."""
    config.validate()
    if clean.ndim != 3 or clean.shape[1] != 9:
        raise ValueError(f"expected [B,9,T], got {tuple(clean.shape)}")
    output = clean.clone()
    draws = torch.rand(clean.shape[0], device=clean.device, generator=generator)
    gaussian_rows = torch.nonzero(
        draws < config.gaussian_probability, as_tuple=False
    ).flatten()
    mask_rows = torch.nonzero(
        (draws >= config.gaussian_probability)
        & (draws < config.gaussian_probability + config.mask_probability),
        as_tuple=False,
    ).flatten()
    if gaussian_rows.numel():
        noise = torch.randn(
            (int(gaussian_rows.numel()), clean.shape[1], clean.shape[2]),
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        output[gaussian_rows] += config.gaussian_std * noise
    for row in mask_rows.tolist():
        length = int(
            torch.randint(
                config.mask_minimum_samples,
                config.mask_maximum_samples + 1,
                (1,),
                device=clean.device,
                generator=generator,
            )
        )
        start = int(
            torch.randint(
                0,
                clean.shape[-1] - length + 1,
                (1,),
                device=clean.device,
                generator=generator,
            )
        )
        output[row, :, start : start + length] = 0.0
    gaussian_count = int(gaussian_rows.numel())
    mask_count = int(mask_rows.numel())
    return output, {
        "clean_windows": int(clean.shape[0]) - gaussian_count - mask_count,
        "gaussian_windows": gaussian_count,
        "masked_windows": mask_count,
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_conv_tcn_nbm200_C_gaussian40_3seed_seed20260807"
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
    parser.add_argument("--clean-probability", type=float, default=0.40)
    parser.add_argument("--gaussian-probability", type=float, default=0.40)
    parser.add_argument("--mask-probability", type=float, default=0.20)
    parser.add_argument("--gaussian-std", type=float, default=0.04)
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


def train_nbm_gru_matched(
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
    augmentation: GRUMatchedAugmentation,
) -> tuple[ConvTCNAutoencoderNBM, dict[str, Any]]:
    augmentation.validate()
    set_seed(seed)
    model = ConvTCNAutoencoderNBM(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_loader = nbm_loader(train_x, True, seed, num_workers)
    validation_loader = nbm_loader(validation_x, False, seed, num_workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint = fold_dir / "checkpoints" / "conv_tcn_nbm_best.pt"
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        counts = {"clean_windows": 0, "gaussian_windows": 0, "masked_windows": 0}
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            corrupted, batch_counts = gru_matched_augmentation(
                clean, augmentation, augmentation_generator
            )
            for key in counts:
                counts[key] += batch_counts[key]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(corrupted)
            loss = criterion(prediction, clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Conv-TCN NBM gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(clean)
            total_n += len(clean)

        model.eval()
        validation_total = 0.0
        validation_n = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
                loss = criterion(model(clean), clean)
                validation_total += float(loss) * len(clean)
                validation_n += len(clean)
        train_loss = total_loss / total_n
        validation_loss = validation_total / validation_n
        scheduler.step(validation_loss)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": learning_rate_now,
                **counts,
                "improved": improved,
            }
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_huber": validation_loss,
                    "seed": seed,
                    "architecture": model.architecture_config(),
                    "augmentation": asdict(augmentation),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"NBM-GAUSS fold={fold_dir.name} epoch={epoch:03d} "
            f"train={train_loss:.7f} val={validation_loss:.7f} "
            f"lr={learning_rate_now:.2e} clean={counts['clean_windows']} "
            f"gauss={counts['gaussian_windows']} mask={counts['masked_windows']} "
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
        "best_validation_huber": best_loss,
        "optimizer": f"AdamW(lr={learning_rate}, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "loss": "SmoothL1(beta=1.0)",
        "augmentation": asdict(augmentation),
        "architecture": model.architecture_config(),
        "history": history,
    }


def run_fold(args: argparse.Namespace, device: torch.device) -> None:
    if args.nbm_max_epochs <= 0 or args.nbm_patience <= 0:
        raise ValueError("NBM max epochs and patience must be positive")
    if args.nbm_learning_rate <= 0:
        raise ValueError("NBM learning rate must be positive")
    augmentation = GRUMatchedAugmentation(
        clean_probability=args.clean_probability,
        gaussian_probability=args.gaussian_probability,
        mask_probability=args.mask_probability,
        gaussian_std=args.gaussian_std,
        mask_minimum_samples=args.mask_min_samples,
        mask_maximum_samples=args.mask_max_samples,
    )
    augmentation.validate()
    output_root = args.output_root.resolve()
    fold_dir = output_root / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed Gaussian NBM fold {args.fold}: {done_path}", flush=True)
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
        "experiment": "conv_tcn_nbm200_gru_matched_augmentation_for_C",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "base_seed": args.seed,
        "effective_seed": args.seed + args.fold,
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "scaler_unique_raw_points": scaler_points,
        "input_preprocessing": (
            "RobustScaler then per-window/per-axis mean subtraction over 128 samples"
        ),
        "architecture": probe_model.architecture_config(),
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_mask_or_noise": False,
            "loss": "SmoothL1(beta=1.0) for both training and validation",
            "optimizer": f"AdamW(lr={args.nbm_learning_rate},weight_decay=1e-4)",
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
            "batch_size": 128,
            "max_epochs": args.nbm_max_epochs,
            "early_stopping_patience": args.nbm_patience,
            "gradient_clip": 1.0,
            "augmentation": asdict(augmentation),
            "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
            "restore_best": True,
        },
        "classifier_or_test_roles_accessed": False,
        "source_audit": source_audit,
    }
    write_json(fold_dir / "config.json", config)
    print(
        f"PREFLIGHT Gaussian NBM fold={args.fold} device={device} "
        f"latent={tuple(latent.shape)} params={config['architecture']['parameter_count']}",
        flush=True,
    )

    role4_x = centered_scaled_bct(scaler, raw_windows(records, role4))
    role5_x = centered_scaled_bct(scaler, raw_windows(records, role5))
    nbm, training = train_nbm_gru_matched(
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
    )
    bias, sigma, calibration = calibrate(nbm, role5_x, device)
    plot_nbm_training(fold_dir, training)
    frozen = {
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": scaler_points,
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "validation_mask_or_noise": False,
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
        "augmentation": asdict(augmentation),
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
