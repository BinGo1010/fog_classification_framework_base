#!/usr/bin/env python3
"""Train one exact-seed global-bottleneck TCN-v2 reconstruction NBM fold.

The NBM accesses only clean Non-FoG role 4 for fitting and clean Non-FoG role
5 for early stopping and residual calibration.  Its temporal encoder features
cannot reach the decoder: the entire 2-second window must pass through one
global 16-dimensional latent vector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
from cnbr_fog.resume import atomic_torch_save
from scripts.run_daphnet_conv_tcn_nbm_gaussian200_fold import (
    GRUMatchedAugmentation,
    gru_matched_augmentation,
)
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
    TCNResidualStack,
    centered_scaled_bct,
    group_count,
    nbm_loader,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file

FOLDS = (0, 1, 2)
CHANNELS = 9
WINDOW_SAMPLES = 128
BOTTLENECK_DIM = 16
DROPOUT = 0.10
ARCHITECTURE_NAME = "global_bottleneck_tcn_autoencoder_nbm_v2"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


class GlobalBottleneckTCNNBM(nn.Module):
    """Non-causal multiscale TCN normal-template generator with global z."""

    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.dropout_probability = float(dropout)
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(9, 24, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(group_count(24), 24),
            nn.GELU(),
        )
        self.encoder_down32 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.encoder_tcn32 = TCNResidualStack(
            32, (1, 2, 4, 8), self.dropout_probability
        )
        self.encoder_down48 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(48), 48),
            nn.GELU(),
        )
        self.encoder_tcn48 = TCNResidualStack(
            48, (1, 2, 4), self.dropout_probability
        )
        self.to_bottleneck = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(48 * 32, BOTTLENECK_DIM),
            nn.LayerNorm(BOTTLENECK_DIM),
            nn.Tanh(),
        )
        self.latent_dropout = nn.Dropout(self.dropout_probability)

        self.latent_projection = nn.Linear(BOTTLENECK_DIM, 32)
        frequencies = torch.tensor(
            [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00],
            dtype=torch.float32,
        )
        time = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / 64.0
        phase = 2.0 * torch.pi * frequencies[:, None] * time[None, :]
        time_code = torch.cat((torch.sin(phase), torch.cos(phase)), dim=0)
        self.register_buffer("time_code", time_code.unsqueeze(0), persistent=True)
        self.decoder_input = nn.Sequential(
            nn.Conv1d(48, 48, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(48), 48),
            nn.GELU(),
        )
        self.decoder_tcn48 = TCNResidualStack(
            48, (1, 2, 4, 8, 16), self.dropout_probability
        )
        self.decoder_to32 = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.output_head = nn.Conv1d(32, 9, kernel_size=1, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"expected [B,9,128], got {tuple(x.shape)}")
        encoded = self.encoder_tcn32(self.encoder_down32(self.encoder_stem(x)))
        encoded = self.encoder_tcn48(self.encoder_down48(encoded))
        if tuple(encoded.shape[1:]) != (48, 32):
            raise RuntimeError(f"unexpected encoder shape: {tuple(encoded.shape)}")
        z = self.to_bottleneck(encoded)
        if tuple(z.shape[1:]) != (BOTTLENECK_DIM,):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != BOTTLENECK_DIM:
            raise ValueError(f"expected [B,16], got {tuple(z.shape)}")
        conditioned = self.latent_dropout(z)
        latent = self.latent_projection(conditioned).unsqueeze(-1).expand(
            -1, -1, WINDOW_SAMPLES
        )
        time_code = self.time_code.expand(z.shape[0], -1, -1)
        decoded = self.decoder_input(torch.cat((latent, time_code), dim=1))
        decoded = self.decoder_tcn48(decoded)
        return self.output_head(self.decoder_to32(decoded))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction {tuple(reconstruction.shape)} != input {tuple(x.shape)}"
            )
        return reconstruction

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "input_shape": ["B", 9, 128],
            "encoder": [
                "Conv1d(9,24,k=7,s=1,p=3)+GroupNorm+GELU",
                "Conv1d(24,32,k=5,s=2,p=2)+GroupNorm+GELU",
                "TCNResidualStack(32,d=1,2,4,8)",
                "Conv1d(32,48,k=5,s=2,p=2)+GroupNorm+GELU",
                "TCNResidualStack(48,d=1,2,4)",
                "Flatten [B,48,32] to [B,1536]",
            ],
            "bottleneck_projection": "Linear(1536,16)+LayerNorm(16)+Tanh",
            "bottleneck_shape": ["B", 16],
            "latent_regularization": f"Dropout(p={self.dropout_probability}) during fit",
            "decoder_conditioning": {
                "latent": "Linear(16,32), broadcast to [B,32,128]",
                "time_code": "fixed Fourier [B,16,128]",
                "frequencies_hz": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                "time_code_trainable": False,
                "raw_or_encoder_temporal_connection": False,
            },
            "decoder": [
                "concat latent template and Fourier code [B,48,128]",
                "Conv1d(48,48,k=1)+GroupNorm+GELU",
                "TCNResidualStack(48,d=1,2,4,8,16)",
                "Conv1d(48,32,k=5,p=2)+GroupNorm+GELU",
                "Conv1d(32,9,k=1), no output activation",
            ],
            "residual_block": (
                "two same-length k=3 convolutions, GroupNorm, GELU, dropout"
            ),
            "causal": False,
            "encoder_decoder_skip_connections": False,
            "teacher_forcing": False,
            "output_activation": None,
            "output_shape": ["B", 9, 128],
            "dropout": self.dropout_probability,
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


@torch.no_grad()
def reconstruct_tcn_v2(
    model: GlobalBottleneckTCNNBM,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in nbm_loader(x, False, 0, 0):
        prediction = model(batch.to(device, non_blocking=True))
        outputs.append(prediction.cpu().numpy().astype(np.float32))
    if not outputs:
        raise ValueError("cannot reconstruct an empty window collection")
    return np.concatenate(outputs, axis=0)


def train_tcn_v2_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    dropout: float,
    augmentation: GRUMatchedAugmentation,
) -> tuple[GlobalBottleneckTCNNBM, dict[str, Any]]:
    augmentation.validate()
    set_seed(seed)
    model = GlobalBottleneckTCNNBM(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_loader = nbm_loader(train_x, True, seed, num_workers)
    validation_loader = nbm_loader(validation_x, False, seed, num_workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = output_dir / "checkpoints" / "tcn_v2_nbm_best.pt"
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
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
            loss = criterion(model(corrupted), clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite TCN-v2 NBM gradient")
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
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": learning_rate,
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
            f"TCN-v2 NBM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"clean={counts['clean_windows']} gauss={counts['gaussian_windows']} "
            f"mask={counts['masked_windows']} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    history_path = output_dir / "logs" / "tcn_v2_nbm_history.csv"
    write_csv(history_path, history)
    summary = {
        "model_id": "tcn_v2_nbm_best",
        "seed": seed,
        "fit_windows": int(len(train_x)),
        "calibration_validation_windows": int(len(validation_x)),
        "maximum_epochs": int(max_epochs),
        "patience": int(patience),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history_file": str(history_path.relative_to(output_dir)),
        "checkpoint_file": str(checkpoint.relative_to(output_dir)),
    }
    return model, {"summary": summary, "history": history}


def calibrate_tcn_v2(
    model: GlobalBottleneckTCNNBM,
    role5_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct_tcn_v2(model, role5_x, device)
    error = role5_x - reconstruction
    bias = np.median(error, axis=(0, 2)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(
        np.abs(error - bias[None, :, None]), axis=(0, 2)
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
        "calibration_windows": int(len(role5_x)),
    }


def plot_training(output_dir: Path, run: dict[str, Any]) -> None:
    history = run["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(
        epochs,
        [row["validation_huber"] for row in history],
        label="Role 5 validation",
    )
    ax.axvline(
        run["summary"]["best_epoch"], color="black", linestyle="--", linewidth=1
    )
    ax.set(
        xlabel="Epoch",
        ylabel="SmoothL1 loss",
        title="Global-bottleneck TCN-v2 reconstruction NBM",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, output_dir / "tcn_v2_nbm_training_validation")
    plt.close(fig)


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
        / (
            "daphnet_tcn_v2_nbm300_C_vs_raw_tcn_ep5pat2_"
            "seedset_0_52_161_5216_52161"
        )
        / "nbm_source"
        / "seed_0",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--required-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--nbm-dropout", type=float, default=DROPOUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, device: torch.device) -> None:
    required_seeds = parse_csv_ints(args.required_seeds)
    if args.seed not in required_seeds:
        raise ValueError(f"seed must be one of {required_seeds}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this experiment requires TCN-v2 NBM max300/pat20")
    if args.nbm_dropout != DROPOUT:
        raise ValueError(f"this experiment freezes TCN-v2 dropout at {DROPOUT}")

    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed TCN-v2 NBM fold/seed: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != CHANNELS:
        raise AssertionError(
            f"expected 64 Hz/9 channels, got "
            f"{dataset.sampling_rate_hz}/{dataset.n_channels}"
        )
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in FOLDS}
    source_audit = audit_protocol(data_dir, rows_by_fold, records)
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
    role4_x = centered_scaled_bct(scaler, raw_windows(records, role4))
    role5_x = centered_scaled_bct(scaler, raw_windows(records, role5))

    probe = GlobalBottleneckTCNNBM(dropout=args.nbm_dropout)
    architecture = probe.architecture_config()
    with torch.no_grad():
        probe_x = torch.zeros(2, 9, 128)
        probe_z = probe.encode(probe_x)
        probe_y = probe(probe_x)
    if probe_z.shape != (2, 16) or probe_y.shape != probe_x.shape:
        raise AssertionError("TCN-v2 NBM shape preflight failed")
    del probe, probe_x, probe_z, probe_y
    augmentation = GRUMatchedAugmentation()
    augmentation.validate()
    config = {
        "experiment": "TCN_v2_NBM300_schemeC_source",
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
        "role_counts": {
            str(role): int(np.sum(rows.role == role)) for role in ROLES
        },
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "input_preprocessing": (
            "RobustScaler then per-window/per-axis mean subtraction"
        ),
        "architecture": architecture,
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_augmentation": False,
            "augmentation": asdict(augmentation),
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

    model, run_payload = train_tcn_v2_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
        args.nbm_dropout,
        augmentation,
    )
    bias, sigma, calibration = calibrate_tcn_v2(model, role5_x, device)
    plot_training(fold_dir, run_payload)
    summary = run_payload["summary"]
    training = {
        **summary,
        "maximum_epochs": 300,
        "patience": 20,
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "augmentation": asdict(augmentation),
        "architecture": architecture,
    }
    checkpoint = fold_dir / "checkpoints" / "tcn_v2_nbm_best.pt"
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
