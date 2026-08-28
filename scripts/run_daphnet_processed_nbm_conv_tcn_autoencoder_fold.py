#!/usr/bin/env python3
"""Train one processed_NBM outer fold and two paired residual-input groups.

The role protocol is unchanged:
  role 4 -> RobustScaler + Conv-TCN NBM fitting
  role 5 -> NBM early stopping and residual b/sigma calibration
  roles 6/7 -> paired TCN classifier training
  roles 2/3 -> classifier early stopping and threshold selection
  roles 0/1 -> one final test after every tunable quantity is frozen

The NBM input uses [B, C, T] = [B, 9, 128].  Its encoder produces
Z=[B,16,32], and its decoder reconstructs X_hat_N=[B,9,128].  After clipped
standardized residual construction, no second per-window/per-axis centering is
applied.  The paired classifier inputs are r=[B,9,128] and
[r,abs(r),delta_t(r)]=[B,27,128].
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    METRIC_KEYS,
    ROLES,
    SUBJECTS,
    RobustScaler,
    audit_protocol,
    choose_document_threshold,
    classifier_loader,
    classifier_predict,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    metric_summary,
    raw_windows,
    residual_diagnostics,
    save_figure_bundle,
    set_seed,
    write_csv,
    write_json,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import TCNBlock

FS = 64
WINDOW = 128
RESIDUAL_EPS = 1e-6

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TCNResidualBlock(nn.Module):
    """Non-causal same-length residual block at one dilation."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        y = self.dropout(self.norm2(self.conv2(y)))
        return F.gelu(x + y)


class TCNResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        dilations: tuple[int, ...] = (1, 2),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(TCNResidualBlock(channels, dilation, dropout) for dilation in dilations)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvTCNAutoencoderNBM(nn.Module):
    """The requested 9x128 -> 16x32 -> 9x128 normal-behavior model."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.encoder_tcn32 = TCNResidualStack(32, (1, 2), dropout)
        self.encoder_down24 = nn.Sequential(
            nn.Conv1d(32, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(24), 24),
            nn.GELU(),
        )
        self.encoder_tcn24 = TCNResidualStack(24, (1, 2), dropout)
        self.to_bottleneck = nn.Sequential(
            nn.Conv1d(24, 16, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(16), 16),
            nn.GELU(),
        )

        self.decoder_from_bottleneck = nn.Sequential(
            nn.Conv1d(16, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(24), 24),
            nn.GELU(),
        )
        self.decoder_to32 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.decoder_tcn32 = TCNResidualStack(32, (1, 2), dropout)
        self.decoder_to16 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(group_count(16), 16),
            nn.GELU(),
        )
        self.output_head = nn.Conv1d(16, 9, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_tcn32(self.encoder_stem(x))
        x = self.encoder_tcn24(self.encoder_down24(x))
        return self.to_bottleneck(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_from_bottleneck(z)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder_tcn32(self.decoder_to32(x))
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder_to16(x)
        return self.output_head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (9, 128):
            raise ValueError(f"expected [B,9,128], got {tuple(x.shape)}")
        z = self.encode(x)
        if tuple(z.shape[1:]) != (16, 32):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        reconstruction = self.decode(z)
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction shape {tuple(reconstruction.shape)} != {tuple(x.shape)}"
            )
        return reconstruction

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "conv_tcn_autoencoder_nbm_v1",
            "input_shape": ["B", 9, 128],
            "encoder": [
                "Conv1d(9,32,k=7,s=2,p=3)+GroupNorm+GELU",
                "TCNResidualBlock(32,d=1) then TCNResidualBlock(32,d=2)",
                "Conv1d(32,24,k=5,s=2,p=2)+GroupNorm+GELU",
                "TCNResidualBlock(24,d=1) then TCNResidualBlock(24,d=2)",
                "Conv1d(24,16,k=1)+GroupNorm+GELU",
            ],
            "bottleneck_shape": ["B", 16, 32],
            "decoder": [
                "Conv1d(16,24,k=3,p=1)+GroupNorm+GELU",
                "linear interpolation x2",
                "Conv1d(24,32,k=5,p=2)+GroupNorm+GELU",
                "TCNResidualBlock(32,d=1) then TCNResidualBlock(32,d=2)",
                "linear interpolation x2",
                "Conv1d(32,16,k=7,p=3)+GroupNorm+GELU",
                "Conv1d(16,9,k=1), no output activation",
            ],
            "residual_block": "two same-length k=3 convolutions, GroupNorm, GELU, dropout",
            "causal": False,
            "encoder_decoder_skip_connections": False,
            "dropout": self.dropout,
            "output_shape": ["B", 9, 128],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


@dataclass(frozen=True)
class MaskConfig:
    probability: float = 0.20
    minimum_samples: int = 4
    maximum_samples: int = 8
    all_channels: bool = True


def light_time_mask(
    clean: torch.Tensor,
    config: MaskConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Mask one short contiguous all-channel span in selected training windows."""
    if clean.ndim != 3 or clean.shape[1] != 9:
        raise ValueError(f"expected [B,9,T], got {tuple(clean.shape)}")
    if not 0.0 <= config.probability <= 1.0:
        raise ValueError("mask probability must be in [0,1]")
    output = clean.clone()
    selected = torch.nonzero(
        torch.rand(clean.shape[0], device=clean.device, generator=generator)
        < config.probability,
        as_tuple=False,
    ).flatten()
    for row in selected.tolist():
        length = int(
            torch.randint(
                config.minimum_samples,
                config.maximum_samples + 1,
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
    return output, int(selected.numel())


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
        / "daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=50)
    parser.add_argument("--nbm-patience", type=int, default=8)
    parser.add_argument("--nbm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--nbm-dropout", type=float, default=0.10)
    parser.add_argument("--mask-probability", type=float, default=0.20)
    parser.add_argument("--mask-min-samples", type=int, default=4)
    parser.add_argument("--mask-max-samples", type=int, default=8)
    parser.add_argument("--tcn-max-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
    parser.add_argument(
        "--representations",
        default="r,r_abs_delta",
        help="Comma-separated classifier inputs: r and/or r_abs_delta.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def centered_scaled_bct(scaler: RobustScaler, raw: np.ndarray) -> np.ndarray:
    """Robust-scale [N,T,C], center every window/axis, return [N,C,T]."""
    values = scaler.transform(raw)
    values = values - values.mean(axis=1, keepdims=True)
    return np.ascontiguousarray(values.transpose(0, 2, 1), dtype=np.float32)


def nbm_loader(
    x: np.ndarray,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x)).float()),
        batch_size=128,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def reconstruct(
    model: ConvTCNAutoencoderNBM,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    for (batch,) in nbm_loader(x, False, 0, 0):
        outputs.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def train_nbm(
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
    mask_config: MaskConfig,
) -> tuple[ConvTCNAutoencoderNBM, dict[str, Any]]:
    set_seed(seed)
    model = ConvTCNAutoencoderNBM(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
        masked_windows = 0
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            corrupted, masked = light_time_mask(clean, mask_config, augmentation_generator)
            masked_windows += masked
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
        lr = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": lr,
                "masked_windows": masked_windows,
                "clean_windows": total_n - masked_windows,
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
                    "mask": asdict(mask_config),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"NBM fold={fold_dir.name} epoch={epoch:02d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={lr:.2e} mask={masked_windows}/{total_n} "
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
        "mask": asdict(mask_config),
        "architecture": model.architecture_config(),
        "history": history,
    }


def calibrate(
    model: ConvTCNAutoencoderNBM,
    role5_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct(model, role5_x, device)
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
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05).astype(int).tolist(),
        "calibration_windows": len(role5_x),
    }


def standardized_residual(
    model: ConvTCNAutoencoderNBM,
    scaler: RobustScaler,
    bias: np.ndarray,
    sigma: np.ndarray,
    raw: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = centered_scaled_bct(scaler, raw)
    reconstruction = reconstruct(model, x, device)
    residual = (x - reconstruction - bias[None, :, None]) / (
        sigma[None, :, None] + RESIDUAL_EPS
    )
    residual = np.clip(residual, -12.0, 12.0).astype(np.float32)
    # Deliberately no post-residual per-window/per-axis centering.
    return np.ascontiguousarray(residual.transpose(0, 2, 1)), reconstruction


def residual_representation(base_r: np.ndarray, name: str) -> np.ndarray:
    """Build a classifier tensor in [N,T,C] from clipped standardized r."""
    base_r = np.asarray(base_r, dtype=np.float32)
    if base_r.ndim != 3 or base_r.shape[1:] != (128, 9):
        raise ValueError(f"expected base residual [N,128,9], got {base_r.shape}")
    if name == "r":
        return np.ascontiguousarray(base_r)
    if name == "r_abs_delta":
        absolute = np.abs(base_r).astype(np.float32, copy=False)
        delta = np.diff(base_r, axis=1, prepend=base_r[:, :1, :]).astype(
            np.float32, copy=False
        )
        features = np.concatenate([base_r, absolute, delta], axis=2)
        if features.shape[1:] != (128, 27):
            raise AssertionError(f"unexpected r_abs_delta shape: {features.shape}")
        return np.ascontiguousarray(features)
    raise ValueError(f"unknown residual representation: {name}")


def representation_channels(name: str) -> int:
    return {"r": 9, "r_abs_delta": 27}[name]


class RepresentationTCNM(nn.Module):
    """The unchanged TCN-M except for its required input-channel projection."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        channels = (int(input_channels), 32, 64, 64, 128)
        dilations = (1, 2, 4, 8)
        self.blocks = nn.Sequential(
            *(TCNBlock(channels[i], channels[i + 1], dilations[i]) for i in range(4))
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.classifier(self.dropout(x.mean(dim=-1))).squeeze(-1)


def paired_tcn_initial_states(seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """Create paired states; extra 27-channel input weights start at zero."""
    set_seed(seed)
    model_r = RepresentationTCNM(9)
    state_r = {name: tensor.detach().cpu().clone() for name, tensor in model_r.state_dict().items()}
    set_seed(seed)
    model_augmented = RepresentationTCNM(27)
    state_augmented = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model_augmented.state_dict().items()
    }
    for name, target in state_augmented.items():
        source = state_r.get(name)
        if source is None:
            continue
        if target.shape == source.shape:
            target.copy_(source)
        elif (
            target.ndim == 3
            and source.ndim == 3
            and target.shape[0] == source.shape[0]
            and target.shape[2] == source.shape[2]
            and target.shape[1] == 27
            and source.shape[1] == 9
        ):
            target.zero_()
            target[:, :9, :].copy_(source)
        else:
            raise AssertionError(
                f"unhandled paired initialization shape for {name}: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}"
            )

    def state_hash(state: dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name, tensor in state.items():
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(tensor.numpy()).tobytes())
        return digest.hexdigest()

    return state_r, state_augmented, {
        "r": state_hash(state_r),
        "r_abs_delta": state_hash(state_augmented),
    }


def validation_classifier_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in classifier_loader(
            x, y, False, 0, 0, batch_size=batch_size
        ):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            loss = criterion(model(batch_x), batch_y)
            total += float(loss) * len(batch_x)
            count += len(batch_x)
    return total / count


def train_representation_tcn(
    representation: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    group_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    initial_state: dict[str, torch.Tensor],
    reset_seed_after_loading: bool = False,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    input_channels = representation_channels(representation)
    model = RepresentationTCNM(input_channels).to(device)
    model.load_state_dict(initial_state)
    if reset_seed_after_loading:
        # Paired experiments with different input-channel counts consume a
        # different number of random values during temporary model creation.
        # Reset here so dropout and all subsequent stochastic operations start
        # from the same seed as well as using the same DataLoader generator.
        set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    n_pos = int(np.sum(train_y == 1))
    n_neg = int(np.sum(train_y == 0))
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    loader = classifier_loader(
        train_x, train_y, True, seed, num_workers, batch_size=batch_size
    )
    checkpoint = group_dir / "checkpoints" / "tcn.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_pr = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite TCN gradient for {representation}")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_bce = total / count
        validation_bce = validation_classifier_loss(
            model,
            validation_x,
            validation_y,
            criterion,
            device,
            batch_size=batch_size,
        )
        val_true, val_prob = classifier_predict(
            model,
            validation_x,
            validation_y,
            device,
            batch_size=batch_size,
        )
        validation_pr_auc = float(average_precision_score(val_true, val_prob))
        improved = validation_pr_auc > best_pr + 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_bce,
                "validation_weighted_bce": validation_bce,
                "validation_pr_auc": validation_pr_auc,
                "improved": improved,
            }
        )
        if improved:
            best_pr = validation_pr_auc
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_pr_auc": validation_pr_auc,
                    "seed": seed,
                    "representation": representation,
                    "input_channels": input_channels,
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"TCN representation={representation} epoch={epoch:02d} "
            f"train={train_bce:.7f} val={validation_bce:.7f} "
            f"val_pr={validation_pr_auc:.7f} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(group_dir / "logs" / "tcn_history.csv", history)
    return model, {
        "representation": representation,
        "input_channels": input_channels,
        "seed": seed,
        "maximum_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr,
        "pos_weight": pos_weight,
        "n_nonfog_role6": n_neg,
        "n_fog_role7": n_pos,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }


def plot_nbm_training(fold_dir: Path, nbm_training: dict[str, Any]) -> None:
    history = nbm_training["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(epochs, [row["validation_huber"] for row in history], label="Role 5 validation")
    ax.axvline(nbm_training["best_epoch"], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss", title="Conv-TCN NBM training")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, fold_dir / "conv_tcn_nbm_training_validation")
    plt.close(fig)


def plot_classifier_training(
    group_dir: Path,
    representation: str,
    tcn_training: dict[str, Any],
    confusion: list[list[int]],
) -> None:
    history = tcn_training["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_weighted_bce"] for row in history], label="Roles 6/7 train")
    axes[0].plot(
        epochs,
        [row["validation_weighted_bce"] for row in history],
        label="Roles 2/3 validation",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Weighted BCE",
        title=f"TCN loss: {representation}",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_pr_auc"] for row in history])
    axes[1].axvline(tcn_training["best_epoch"], color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Epoch", ylabel="PR-AUC", title="Roles 2/3 model selection")
    axes[1].grid(alpha=0.25)
    save_figure_bundle(fig, group_dir / "tcn_training_validation")
    plt.close(fig)

    cm = np.asarray(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 0.5 * cm.max() else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color=color)
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.set(
        xlabel="Predicted",
        ylabel="True",
        title=f"Permanent test: {representation}",
    )
    save_figure_bundle(fig, group_dir / "test_confusion_matrix")
    plt.close(fig)


def run_fold(args: argparse.Namespace, device: torch.device) -> None:
    if args.fold is None:
        raise ValueError("--fold is required unless --aggregate-only is used")
    output_root = args.output_root.resolve()
    fold_dir = output_root / f"fold_{args.fold}"
    done_path = fold_dir / "DONE.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed fold {args.fold}: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {
        fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in (0, 1, 2)
    }
    source_audit = audit_protocol(args.data_dir.resolve(), rows_by_fold, records)
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    representations = [
        value.strip() for value in args.representations.split(",") if value.strip()
    ]
    if not representations or len(representations) != len(set(representations)):
        raise ValueError(f"invalid --representations: {args.representations}")
    unknown = sorted(set(representations) - {"r", "r_abs_delta"})
    if unknown:
        raise ValueError(f"unknown residual representations: {unknown}")
    scaler, scaler_points = fit_scaler_unique_role4_points(records, role4)
    mask_config = MaskConfig(
        probability=args.mask_probability,
        minimum_samples=args.mask_min_samples,
        maximum_samples=args.mask_max_samples,
    )
    model = ConvTCNAutoencoderNBM(dropout=args.nbm_dropout)
    with torch.no_grad():
        probe = torch.zeros(2, 9, 128)
        latent = model.encode(probe)
        reconstructed = model(probe)
    if latent.shape != (2, 16, 32) or reconstructed.shape != probe.shape:
        raise AssertionError("Conv-TCN NBM shape preflight failed")
    probe_r = np.zeros((2, 128, 9), dtype=np.float32)
    representation_shapes = {
        name: list(residual_representation(probe_r, name).shape)
        for name in representations
    }
    tcn_seed = args.seed + args.fold + 100
    state_r, state_augmented, state_hashes = paired_tcn_initial_states(tcn_seed)
    initial_states = {"r": state_r, "r_abs_delta": state_augmented}
    with torch.no_grad():
        for name in representations:
            classifier_probe = RepresentationTCNM(representation_channels(name))
            classifier_probe.load_state_dict(initial_states[name])
            output = classifier_probe(
                torch.zeros(2, representation_channels(name), 128)
            )
            if output.shape != (2,):
                raise AssertionError(f"classifier shape preflight failed for {name}")
    config = {
        "experiment": "processed_NBM_conv_tcn_autoencoder_paired_residual_representations",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "scaler_unique_raw_points": scaler_points,
        "window_centering": "after scaling, per window/channel mean over time is subtracted",
        "nbm": {
            **model.architecture_config(),
            "training_roles": [4],
            "earlystop_and_calibration_roles": [5],
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": f"AdamW(lr={args.nbm_learning_rate},weight_decay=1e-4)",
            "batch_size": 128,
            "max_epochs": args.nbm_max_epochs,
            "patience": args.nbm_patience,
            "mask": asdict(mask_config),
        },
        "residual": {
            "formula": "clip((X_scaled_centered-Xhat-b)/(sigma+1e-6),-12,12)",
            "post_residual_window_axis_centering": False,
            "representations": {
                "r": {"formula": "r", "classifier_input_shape": ["B", 9, 128]},
                "r_abs_delta": {
                    "formula": "concat(r,abs(r),delta_t(r)); delta_r[0]=0; delta_r[t]=r[t]-r[t-1]",
                    "classifier_input_shape": ["B", 27, 128],
                    "delta_source": "clipped standardized r",
                    "additional_clipping_or_scaling": False,
                },
            },
            "selected_groups": representations,
        },
        "classifier": {
            "same_backbone_except_required_input_channels": True,
            "train_roles": [6, 7],
            "validation_roles": [2, 3],
            "test_roles": [0, 1],
            "pos_weight": "N_role6/N_role7",
            "max_epochs": args.tcn_max_epochs,
            "patience": args.tcn_patience,
            "paired_initialization": (
                "all shape-compatible parameters identical; in the 27-channel first "
                "projection, r weights copy the 9-channel group and the extra 18 input "
                "weights start at zero"
            ),
            "seed": tcn_seed,
            "initial_state_sha256": state_hashes,
            "preflight_feature_shapes_ntc": representation_shapes,
        },
        "threshold": "roles 2/3 balanced accuracy, 0.05..0.95 step 0.01; ties FoG F1 then higher threshold",
        "test_use": (
            "roles 0/1 are not loaded until both groups have frozen TCN checkpoints "
            "and validation-selected thresholds"
        ),
        "source_audit": source_audit,
    }
    write_json(fold_dir / "config.json", config)
    print(
        f"PREFLIGHT fold={args.fold} device={device} latent={tuple(latent.shape)} "
        f"reconstruction={tuple(reconstructed.shape)} params={model.architecture_config()['parameter_count']}",
        flush=True,
    )
    if args.dry_run:
        write_json(fold_dir / "DRY_RUN.json", {"status": "complete", "config": config})
        return

    role4_x = centered_scaled_bct(scaler, raw_windows(records, role4))
    role5_x = centered_scaled_bct(scaler, raw_windows(records, role5))
    nbm_seed = args.seed + args.fold
    nbm, nbm_training = train_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        nbm_seed,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
        args.nbm_learning_rate,
        args.nbm_dropout,
        mask_config,
    )
    bias, sigma, calibration = calibrate(nbm, role5_x, device)
    write_json(
        fold_dir / "nbm_frozen.json",
        {
            "scaler": scaler.as_dict(),
            "scaler_fit_role": 4,
            "scaler_unique_raw_points": scaler_points,
            "nbm_train_role": 4,
            "nbm_earlystop_and_calibration_role": 5,
            "training": {key: value for key, value in nbm_training.items() if key != "history"},
            "calibration": calibration,
        },
    )

    train_base_r, _ = standardized_residual(
        nbm, scaler, bias, sigma, raw_windows(records, role67), device
    )
    validation_base_r, _ = standardized_residual(
        nbm, scaler, bias, sigma, raw_windows(records, role23), device
    )
    write_json(
        fold_dir / "paired_classifier_initialization.json",
        {
            "seed": tcn_seed,
            "state_sha256": state_hashes,
            "shared_parameters_identical": True,
            "r_abs_delta_extra_input_weights": "zero",
        },
    )
    plot_nbm_training(fold_dir, nbm_training)
    frozen_groups: dict[str, Any] = {}
    for representation in representations:
        group_dir = fold_dir / "groups" / representation
        group_dir.mkdir(parents=True, exist_ok=True)
        train_features = residual_representation(train_base_r, representation)
        validation_features = residual_representation(
            validation_base_r, representation
        )
        tcn, tcn_training = train_representation_tcn(
            representation,
            train_features,
            role67.label,
            validation_features,
            role23.label,
            group_dir,
            device,
            tcn_seed,
            args.num_workers,
            args.tcn_max_epochs,
            args.tcn_patience,
            initial_states[representation],
        )
        val_true, val_prob = classifier_predict(
            tcn, validation_features, role23.label, device
        )
        threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
        frozen_groups[representation] = {
            "model": tcn.to("cpu"),
            "training": tcn_training,
            "threshold": threshold,
            "validation": validation_metrics,
            "train_diagnostics": residual_diagnostics(
                train_features, role67.label
            ),
            "validation_diagnostics": residual_diagnostics(
                validation_features, role23.label
            ),
        }
        print(
            f"VALIDATION FROZEN fold={args.fold} representation={representation} "
            f"best_epoch={tcn_training['best_epoch']} threshold={threshold:.2f}",
            flush=True,
        )

    # Strict test gate: roles 0/1 are not loaded or transformed until every
    # comparison group's checkpoint and validation-selected threshold is frozen.
    if set(frozen_groups) != set(representations):
        raise AssertionError("not every comparison group was frozen before testing")
    test_rows = rows.take_role(0, 1)
    test_base_r, _ = standardized_residual(
        nbm, scaler, bias, sigma, raw_windows(records, test_rows), device
    )
    group_results: dict[str, Any] = {}
    for representation in representations:
        group_dir = fold_dir / "groups" / representation
        frozen = frozen_groups[representation]
        tcn = frozen["model"].to(device)
        tcn_training = frozen["training"]
        threshold = float(frozen["threshold"])
        validation_metrics = frozen["validation"]
        test_features = residual_representation(test_base_r, representation)
        test_true, test_prob = classifier_predict(
            tcn, test_features, test_rows.label, device
        )
        test_metrics = binary_metrics(test_true, test_prob, threshold)
        test_pred = (test_prob >= threshold).astype(np.int8)
        subject_metrics = {}
        for subject in SUBJECTS:
            mask = test_rows.subject_id == subject
            subject_metrics[subject] = binary_metrics(
                test_true[mask], test_prob[mask], threshold
            )
        diagnostics = {
            "classifier_train_roles_6_7": frozen["train_diagnostics"],
            "classifier_validation_roles_2_3": frozen[
                "validation_diagnostics"
            ],
            "permanent_test_roles_0_1": residual_diagnostics(
                test_features, test_true
            ),
        }
        result = {
            "fold": args.fold,
            "representation": representation,
            "input_shape": ["B", representation_channels(representation), 128],
            "post_residual_window_axis_centering": False,
            "threshold": threshold,
            "threshold_source_roles": [2, 3],
            "all_group_thresholds_frozen_before_any_test_access": True,
            "nbm_shared_across_groups": True,
            "tcn_initial_state_sha256": state_hashes[representation],
            "tcn_training": {
                key: value for key, value in tcn_training.items() if key != "history"
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "test_by_subject": subject_metrics,
            "feature_diagnostics": diagnostics,
        }
        write_json(group_dir / "metrics.json", result)
        write_csv(
            group_dir / "predictions.csv",
            [
                {
                    "fold": args.fold,
                    "representation": representation,
                    "subject_id": str(test_rows.subject_id[i]),
                    "record_id": str(test_rows.record_id[i]),
                    "window_id": str(test_rows.window_id[i]),
                    "start_index": int(test_rows.start[i]),
                    "end_index_exclusive": int(test_rows.end[i]),
                    "role_code": int(test_rows.role[i]),
                    "y_true": int(test_true[i]),
                    "fog_probability": float(test_prob[i]),
                    "threshold": threshold,
                    "y_pred": int(test_pred[i]),
                }
                for i in range(len(test_rows))
            ],
        )
        np.savez_compressed(
            group_dir / "test_probabilities.npz",
            y_true=test_true,
            y_prob=test_prob,
            y_pred=test_pred,
            subject_id=test_rows.subject_id,
            window_id=test_rows.window_id,
            threshold=np.asarray(threshold),
        )
        plot_classifier_training(
            group_dir,
            representation,
            tcn_training,
            test_metrics["confusion_matrix"],
        )
        write_json(
            group_dir / "DONE.json",
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "threshold": threshold,
                "test": test_metrics,
            },
        )
        group_results[representation] = result
        print(
            f"GROUP COMPLETE fold={args.fold} representation={representation} "
            f"threshold={threshold:.2f} acc={test_metrics['accuracy']:.6f} "
            f"recall={test_metrics['sensitivity']:.6f} "
            f"specificity={test_metrics['specificity']:.6f} "
            f"pr_auc={test_metrics['auprc']:.6f}",
            flush=True,
        )

    fold_result = {
        "fold": args.fold,
        "representations": representations,
        "post_residual_window_axis_centering": False,
        "nbm_training": {
            key: value for key, value in nbm_training.items() if key != "history"
        },
        "groups": group_results,
    }
    write_json(fold_dir / "metrics.json", fold_result)
    write_json(
        done_path,
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "representations": representations,
            "groups": {
                name: {"threshold": result["threshold"], "test": result["test"]}
                for name, result in group_results.items()
            },
        },
    )
    print(
        f"FOLD {args.fold} COMPLETE groups={representations}",
        flush=True,
    )


def aggregate(output_root: Path) -> None:
    fold_results = []
    for fold in (0, 1, 2):
        path = output_root / f"fold_{fold}" / "metrics.json"
        done = output_root / f"fold_{fold}" / "DONE.json"
        if not path.exists() or not done.exists():
            raise FileNotFoundError(f"fold {fold} is incomplete under {output_root}")
        fold_results.append(json.loads(path.read_text(encoding="utf-8")))
    representations = fold_results[0]["representations"]
    if set(representations) != {"r", "r_abs_delta"}:
        raise ValueError(
            f"aggregation requires both comparison groups, found {representations}"
        )
    if any(result["representations"] != representations for result in fold_results):
        raise ValueError("fold representation order/configuration mismatch")
    group_summaries: dict[str, Any] = {}
    for representation in representations:
        results = [fold["groups"][representation] for fold in fold_results]
        aggregate_metrics = metric_summary([result["test"] for result in results])
        subject_aggregate = {
            subject: metric_summary(
                [result["test_by_subject"][subject] for result in results]
            )
            for subject in SUBJECTS
        }
        group_summaries[representation] = {
            "test_mean_std": aggregate_metrics,
            "subject_test_mean_std": subject_aggregate,
        }
        write_csv(
            output_root / f"fold_metrics_{representation}.csv",
            [
                {
                    "fold": result["fold"],
                    "representation": representation,
                    "threshold": result["threshold"],
                    **result["test"],
                }
                for result in results
            ],
        )
        write_csv(
            output_root / f"subject_metrics_mean_{representation}.csv",
            [
                {
                    "subject_id": subject,
                    **{
                        f"{key}_mean": subject_aggregate[subject][key]["mean"]
                        for key in METRIC_KEYS
                    },
                    **{
                        f"{key}_std": subject_aggregate[subject][key]["std"]
                        for key in METRIC_KEYS
                    },
                }
                for subject in SUBJECTS
            ],
        )
    comparison_rows = []
    for key in METRIC_KEYS:
        r_mean = group_summaries["r"]["test_mean_std"][key]["mean"]
        augmented_mean = group_summaries["r_abs_delta"]["test_mean_std"][key][
            "mean"
        ]
        comparison_rows.append(
            {
                "metric": key,
                "r_mean": r_mean,
                "r_abs_delta_mean": augmented_mean,
                "r_abs_delta_minus_r": augmented_mean - r_mean,
            }
        )
    write_csv(output_root / "representation_comparison.csv", comparison_rows)
    write_json(
        output_root / "summary.json",
        {
            "fold_results": fold_results,
            "groups": group_summaries,
            "r_abs_delta_minus_r": {
                row["metric"]: row["r_abs_delta_minus_r"]
                for row in comparison_rows
            },
            "post_residual_window_axis_centering": False,
            "note": "roles 0/1 are the same permanent test windows in all folds",
        },
    )
    write_json(
        output_root / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "groups": group_summaries,
            "comparison": comparison_rows,
        },
    )
    print(json.dumps(group_summaries, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(output_root)
        return
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run_fold(args, device)


if __name__ == "__main__":
    main()
