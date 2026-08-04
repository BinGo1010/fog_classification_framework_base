#!/usr/bin/env python
"""Within-S01 spectral GRU-NBM reconstruction + single-window TCN-M.

Protocol fixed for this experiment:

* nine accelerometer channels at 64 Hz;
* chronological, raw-support-disjoint train/validation/test partitions;
* 2 s windows (128 samples), 1 s stride;
* train-only robust scaling;
* Hann-windowed 128-point log power spectrum, shape [9, 65];
* a 64-dimensional GRU denoising bottleneck trained on clean Non-FoG only;
* signed raw spectral residual ``observed - reconstruction`` (no calibration,
  z-scoring, or clipping);
* one residual window [9, 65] classified along the frequency axis by TCN-M;
* validation-only early stopping and decision-threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset, RobustChannelScaler
from cnbr_fog.evaluation import binary_metrics, choose_threshold
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    dataset_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_spectral_gru_nbm_tcnm.v1"
SUBJECT_ID = "S01"
FS = 64
WINDOW_SAMPLES = 128
STRIDE_SAMPLES = 64
N_FREQ = 65
NORMAL_GUARD_SAMPLES = 32
TRAIN_VALIDATION_CUT_SAMPLE = 50_944
TRAIN_RECORD = "S01_seg000"
CUT_RECORD = "S01_seg001"
TEST_RECORD = "S01_seg002"
EXPECTED_CHANNEL_NAMES = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)


@dataclass(frozen=True)
class Windows:
    record_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    label: np.ndarray
    fog_fraction: np.ndarray
    clean_normal: np.ndarray

    def __len__(self) -> int:
        return int(self.start.size)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def parse_args() -> argparse.Namespace:
    local_data = (
        REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed"
    )
    cloud_data = Path(
        r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="S01 log-power spectral GRU-NBM + single-window TCN-M",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=local_data if local_data.exists() else cloud_data
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "daphnet_s01_spectral_gru_nbm_tcnm_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-epochs", type=int, default=50)
    parser.add_argument("--nbm-patience", type=int, default=10)
    parser.add_argument("--nbm-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=10)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--time-mask-probability", type=float, default=0.30)
    parser.add_argument("--channel-mask-probability", type=float, default=0.20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def load_s01(root: Path) -> DaphnetDataset:
    full = DaphnetDataset.load(root)
    records = [record for record in full.records if record.subject_id == SUBJECT_ID]
    if {record.record_id for record in records} != {TRAIN_RECORD, CUT_RECORD, TEST_RECORD}:
        raise ValueError("Unexpected S01 record set")
    if full.sampling_rate_hz != FS or tuple(full.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError("Unexpected sampling rate or channel schema")
    return DaphnetDataset(
        root=root,
        records=records,
        sampling_rate_hz=full.sampling_rate_hz,
        channel_names=full.channel_names,
    )


def make_windows(dataset: DaphnetDataset) -> Windows:
    rec_idx: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    labels: list[int] = []
    fractions: list[float] = []
    clean: list[bool] = []
    for record_index, record in enumerate(dataset.records):
        invalid_prefix = np.r_[0, np.cumsum(~record.valid, dtype=np.int64)]
        fog_prefix = np.r_[0, np.cumsum(record.y == 1, dtype=np.int64)]
        for start in range(0, len(record.y) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            if invalid_prefix[end] - invalid_prefix[start]:
                continue
            fog_count = int(fog_prefix[end] - fog_prefix[start])
            fraction = fog_count / WINDOW_SAMPLES
            guard_start = max(0, start - NORMAL_GUARD_SAMPLES)
            guard_end = min(len(record.y), end + NORMAL_GUARD_SAMPLES)
            rec_idx.append(record_index)
            starts.append(start)
            ends.append(end)
            labels.append(int(fraction >= 0.5))
            fractions.append(fraction)
            clean.append(bool(fog_prefix[guard_end] - fog_prefix[guard_start] == 0))
    return Windows(
        record_index=np.asarray(rec_idx, dtype=np.int32),
        start=np.asarray(starts, dtype=np.int32),
        end=np.asarray(ends, dtype=np.int32),
        label=np.asarray(labels, dtype=np.int8),
        fog_fraction=np.asarray(fractions, dtype=np.float32),
        clean_normal=np.asarray(clean, dtype=bool),
    )


def make_split(dataset: DaphnetDataset, windows: Windows) -> Split:
    lookup = {record.record_id: i for i, record in enumerate(dataset.records)}
    train = np.flatnonzero(
        (windows.record_index == lookup[TRAIN_RECORD])
        | (
            (windows.record_index == lookup[CUT_RECORD])
            & (windows.end <= TRAIN_VALIDATION_CUT_SAMPLE)
        )
    )
    validation = np.flatnonzero(
        (windows.record_index == lookup[CUT_RECORD])
        & (windows.start >= TRAIN_VALIDATION_CUT_SAMPLE)
    )
    test = np.flatnonzero(windows.record_index == lookup[TEST_RECORD])
    split = Split(train, validation, test)
    for name, indices in split.as_dict().items():
        counts = np.bincount(windows.label[indices], minlength=2)
        if not len(indices) or np.any(counts == 0):
            raise ValueError(f"Invalid {name} split: {counts.tolist()}")
    if int(windows.end[train[windows.record_index[train] == lookup[CUT_RECORD]]].max()) != TRAIN_VALIDATION_CUT_SAMPLE:
        raise AssertionError("Training support does not end at cut")
    if int(windows.start[validation].min()) != TRAIN_VALIDATION_CUT_SAMPLE:
        raise AssertionError("Validation support does not start at cut")
    return split


def fit_scaler(dataset: DaphnetDataset) -> tuple[RobustChannelScaler, int]:
    by_id = {record.record_id: record for record in dataset.records}
    ranges = (
        (by_id[TRAIN_RECORD], 0, len(by_id[TRAIN_RECORD].y)),
        (by_id[CUT_RECORD], 0, TRAIN_VALIDATION_CUT_SAMPLE),
    )
    chunks = [
        record.x[start:end][record.valid[start:end] & (record.y[start:end] == 0)]
        for record, start, end in ranges
    ]
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (
        RobustChannelScaler(center.astype(np.float32), scale.astype(np.float32), 12.0),
        int(len(values)),
    )


def split_clean_normal_indices(
    dataset: DaphnetDataset, windows: Windows, split: Split
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {record.record_id: i for i, record in enumerate(dataset.records)}
    train = split.train[windows.clean_normal[split.train]]
    validation = split.validation[windows.clean_normal[split.validation]]
    train = train[
        (windows.record_index[train] != lookup[CUT_RECORD])
        | (windows.end[train] <= TRAIN_VALIDATION_CUT_SAMPLE - NORMAL_GUARD_SAMPLES)
    ]
    validation = validation[
        (windows.record_index[validation] != lookup[CUT_RECORD])
        | (windows.start[validation] >= TRAIN_VALIDATION_CUT_SAMPLE + NORMAL_GUARD_SAMPLES)
    ]
    return train, validation


def extract_windows(
    dataset: DaphnetDataset,
    windows: Windows,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
) -> np.ndarray:
    result = np.empty((len(indices), WINDOW_SAMPLES, dataset.n_channels), dtype=np.float32)
    for output_index, window_index in enumerate(indices):
        record = dataset.records[int(windows.record_index[window_index])]
        result[output_index] = scaler.transform(
            record.x[int(windows.start[window_index]) : int(windows.end[window_index])]
        )
    return result


class LogPowerSpectrum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hann = torch.hann_window(WINDOW_SAMPLES, periodic=True)
        self.register_buffer("hann", hann)
        self.register_buffer("window_energy", hann.square().sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 128, 9] -> [B, 9, 65]
        transformed = torch.fft.rfft(x * self.hann[None, :, None], dim=1)
        power = transformed.abs().square() / self.window_energy
        return torch.log1p(power).transpose(1, 2).contiguous()


@dataclass(frozen=True)
class SpectralRobustScaler:
    center: np.ndarray
    scale: np.ndarray
    fit_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": [9, N_FREQ],
            "fit_windows": int(self.fit_windows),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "definition": "per-axis/per-frequency median and IQR/1.349 with standard-deviation fallback",
            "clipping": None,
        }


class SpectralRobustTransform(nn.Module):
    def __init__(self, scaler: SpectralRobustScaler) -> None:
        super().__init__()
        self.register_buffer("center", torch.from_numpy(scaler.center)[None])
        self.register_buffer("scale", torch.from_numpy(scaler.scale)[None])

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return (spectrum - self.center) / self.scale


@torch.no_grad()
def fit_spectral_robust_scaler(
    clean_normal_x: np.ndarray, batch_size: int
) -> SpectralRobustScaler:
    if clean_normal_x.ndim != 3 or clean_normal_x.shape[1:] != (WINDOW_SAMPLES, 9):
        raise ValueError(
            f"Expected clean-normal windows [B,{WINDOW_SAMPLES},9], got {clean_normal_x.shape}"
        )
    spectral = LogPowerSpectrum()
    parts: list[np.ndarray] = []
    for start in range(0, len(clean_normal_x), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(clean_normal_x[start : start + batch_size])
        )
        parts.append(spectral(batch).numpy().astype(np.float64, copy=False))
    values = np.concatenate(parts, axis=0)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return SpectralRobustScaler(
        center=np.ascontiguousarray(center, dtype=np.float32),
        scale=np.ascontiguousarray(scale, dtype=np.float32),
        fit_windows=int(len(clean_normal_x)),
    )


class SpectralGRUNBM(nn.Module):
    def __init__(
        self, channels: int = 9, hidden: int = 64, output_nonnegative: bool = True
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(channels, hidden, num_layers=1, batch_first=True)
        decoder: list[nn.Module] = [
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, channels * N_FREQ),
        ]
        if output_nonnegative:
            decoder.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder)
        self.channels = channels
        self.hidden = hidden
        self.output_nonnegative = output_nonnegative

    def forward(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Treat ordered frequency bins as the GRU sequence: [B,65,9].
        _, hidden = self.gru(spectrum.transpose(1, 2))
        latent = hidden[-1]
        reconstruction = self.decoder(latent).reshape(-1, self.channels, N_FREQ)
        return reconstruction, latent


def corrupt_time_domain(x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    corrupted = x + torch.randn_like(x) * float(args.noise_std)
    for row in range(len(corrupted)):
        if torch.rand((), device=x.device) < args.time_mask_probability:
            length = int(torch.randint(4, 13, (), device=x.device).item())
            start = int(
                torch.randint(0, WINDOW_SAMPLES - length + 1, (), device=x.device).item()
            )
            corrupted[row, start : start + length, :] = 0.0
        if torch.rand((), device=x.device) < args.channel_mask_probability:
            channel = int(torch.randint(0, x.shape[2], (), device=x.device).item())
            corrupted[row, :, channel] = 0.0
    return corrupted


def loader(
    x: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    tensors = [torch.from_numpy(np.ascontiguousarray(x))]
    if y is not None:
        tensors.append(torch.from_numpy(np.ascontiguousarray(y)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_gru_nbm(
    args: argparse.Namespace,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    spectral_scaler: SpectralRobustScaler | None = None,
) -> tuple[SpectralGRUNBM, dict[str, Any]]:
    set_seed(args.seed, args.deterministic)
    model = SpectralGRUNBM(output_nonnegative=spectral_scaler is None).to(device)
    spectral = LogPowerSpectrum().to(device)
    spectral_transform: nn.Module = (
        SpectralRobustTransform(spectral_scaler).to(device)
        if spectral_scaler is not None
        else nn.Identity().to(device)
    )
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.nbm_lr, weight_decay=args.weight_decay
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_loader = loader(
        validation_x, None, args.batch_size, False, args.num_workers, args.seed
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.perf_counter()
    checkpoint = output_dir / "gru_nbm_best.pt"
    for epoch in range(1, args.nbm_epochs + 1):
        if bad_epochs >= args.nbm_patience:
            break
        train_loader = loader(
            train_x, None, args.batch_size, True, args.num_workers, args.seed + epoch
        )
        model.train()
        total_loss = 0.0
        total_n = 0
        for (clean_cpu,) in train_loader:
            clean = clean_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                target = spectral_transform(spectral(clean))
                model_input = spectral_transform(
                    spectral(corrupt_time_domain(clean, args))
                )
                reconstruction, _ = model(model_input)
                loss = criterion(reconstruction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(clean)
            total_n += len(clean)
        train_loss = total_loss / total_n

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for (clean_cpu,) in val_loader:
                clean = clean_cpu.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    target = spectral_transform(spectral(clean))
                    reconstruction, _ = model(target)
                    loss = criterion(reconstruction, target)
                val_total += float(loss) * len(clean)
                val_n += len(clean)
        val_loss = val_total / val_n
        improved = val_loss < best_loss - 1e-6
        history.append(
            {
                "epoch": epoch,
                "train_smooth_l1": train_loss,
                "validation_smooth_l1": val_loss,
                "improved": improved,
            }
        )
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {"epoch": epoch, "model_state": model.state_dict()}, checkpoint
            )
        else:
            bad_epochs += 1
        print(
            f"[GRU-NBM] epoch={epoch:02d} train={train_loss:.7f} "
            f"val={val_loss:.7f}{' *' if improved else ''}",
            flush=True,
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    training = {
        "training_windows": int(len(train_x)),
        "validation_windows": int(len(validation_x)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "optimizer": "AdamW",
        "loss": "SmoothL1",
        "learning_rate": args.nbm_lr,
        "weight_decay": args.weight_decay,
        "maximum_epochs": args.nbm_epochs,
        "patience": args.nbm_patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_smooth_l1": best_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint),
        "spectral_robust_standardization": spectral_scaler is not None,
        "decoder_output_activation": (
            "identity" if spectral_scaler is not None else "softplus"
        ),
    }
    atomic_json_dump(training, output_dir / "gru_nbm_training.json")
    return model, training


@torch.no_grad()
def extract_residuals(
    args: argparse.Namespace,
    model: SpectralGRUNBM,
    x: np.ndarray,
    device: torch.device,
    spectral_scaler: SpectralRobustScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    spectral = LogPowerSpectrum().to(device)
    spectral_transform: nn.Module = (
        SpectralRobustTransform(spectral_scaler).to(device)
        if spectral_scaler is not None
        else nn.Identity().to(device)
    )
    observed_parts: list[np.ndarray] = []
    reconstruction_parts: list[np.ndarray] = []
    for (batch_cpu,) in loader(x, None, args.batch_size, False, args.num_workers, 0):
        batch = batch_cpu.to(device, non_blocking=True)
        observed = spectral_transform(spectral(batch))
        reconstruction, _ = model(observed)
        observed_parts.append(observed.float().cpu().numpy())
        reconstruction_parts.append(reconstruction.float().cpu().numpy())
    observed_np = np.concatenate(observed_parts).astype(np.float32)
    reconstruction_np = np.concatenate(reconstruction_parts).astype(np.float32)
    return observed_np, reconstruction_np, observed_np - reconstruction_np


class SpectralResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpectralTCNM(nn.Module):
    def __init__(self, in_channels: int = 9, hidden: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU()
        )
        self.blocks = nn.Sequential(
            *[SpectralResidualBlock(hidden, dilation, 0.20) for dilation in (1, 2, 4, 8)]
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)
        return self.head(pooled).squeeze(1)


def classifier_pass(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for x_cpu, y_cpu in data_loader:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.float().to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.detach()) * len(x)
        total_n += len(x)
        truths.append(y_cpu.numpy().astype(np.int8))
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return total_loss / total_n, np.concatenate(truths), np.concatenate(probabilities)


def metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result["fog_recall"] = result["sensitivity"]
    result["pr_auc"] = result["auprc"]
    result["roc_auc"] = result["auroc"]
    return result


def train_classifier(
    args: argparse.Namespace,
    residuals: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    classifier_seed = args.seed + 10_000
    set_seed(classifier_seed, args.deterministic)
    model = SpectralTCNM().to(device)
    counts = np.bincount(labels["train"], minlength=2).astype(np.float64)
    positive_weight = min(math.sqrt(counts[0] / counts[1]), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay
    )
    val_loader = loader(
        residuals["validation"], labels["validation"], args.batch_size, False,
        args.num_workers, classifier_seed,
    )
    test_loader = loader(
        residuals["test"], labels["test"], args.batch_size, False,
        args.num_workers, classifier_seed,
    )
    history: list[dict[str, Any]] = []
    best_pr = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    checkpoint = output_dir / "tcnm_best.pt"
    started = time.perf_counter()
    for epoch in range(1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
        train_loader = loader(
            residuals["train"], labels["train"], args.batch_size, True,
            args.num_workers, classifier_seed + epoch,
        )
        train_loss, train_true, train_prob = classifier_pass(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_true, val_prob = classifier_pass(
                model, val_loader, criterion, device
            )
        train_pr = float(average_precision_score(train_true, train_prob))
        val_pr = float(average_precision_score(val_true, val_prob))
        improved = val_pr > best_pr + 1e-5
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "train_pr_auc": train_pr,
                "validation_bce": val_loss,
                "validation_pr_auc": val_pr,
                "improved": improved,
            }
        )
        if improved:
            best_pr = val_pr
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {"epoch": epoch, "model_state": model.state_dict()}, checkpoint
            )
        else:
            bad_epochs += 1
        print(
            f"[TCN-M] epoch={epoch:02d} train_bce={train_loss:.7f} "
            f"val_bce={val_loss:.7f} val_pr={val_pr:.6f}{' *' if improved else ''}",
            flush=True,
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    with torch.no_grad():
        _, val_true, val_prob = classifier_pass(model, val_loader, criterion, device)
        _, test_true, test_prob = classifier_pass(model, test_loader, criterion, device)
    threshold, validation_metrics = choose_threshold(val_true, val_prob)
    validation_metrics = metric_aliases(validation_metrics)
    test_metrics = metric_aliases(binary_metrics(test_true, test_prob, threshold))
    training = {
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "positive_class_weight": positive_weight,
        "training_counts_non_fog_fog": counts.astype(int).tolist(),
        "learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "maximum_epochs": args.classifier_epochs,
        "patience": args.classifier_patience,
        "early_stop_metric": "validation PR-AUC",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr,
        "threshold_selection": "validation grid 0.01..0.99, maximum balanced accuracy",
        "selected_threshold": threshold,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    metrics = {"validation": validation_metrics, "test": test_metrics}
    predictions = {
        "validation_true": val_true,
        "validation_probability": val_prob.astype(np.float32),
        "validation_prediction": (val_prob >= threshold).astype(np.int8),
        "test_true": test_true,
        "test_probability": test_prob.astype(np.float32),
        "test_prediction": (test_prob >= threshold).astype(np.int8),
    }
    atomic_json_dump(training, output_dir / "tcnm_training.json")
    atomic_json_dump(metrics, output_dir / "metrics.json")
    return training, metrics, predictions


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "q25": float(np.percentile(flat, 25)),
        "q75": float(np.percentile(flat, 75)),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
    }


def residual_diagnostics(residual: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    score = np.mean(np.abs(residual), axis=(1, 2))
    return {
        "signed_residual": distribution_summary(residual),
        "window_mean_absolute_residual_non_fog": distribution_summary(score[y == 0]),
        "window_mean_absolute_residual_fog": distribution_summary(score[y == 1]),
        "simple_absolute_residual_pr_auc": float(average_precision_score(y, score)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(
    rows: list[dict[str, Any]], train_key: str, val_key: str, title: str, path: Path
) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, [row[train_key] for row in rows], label="Train", linewidth=2)
    axis.plot(epochs, [row[val_key] for row in rows], label="Validation", linewidth=2)
    axis.set(xlabel="Epoch", ylabel="Loss", title=title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confusion(
    metrics: dict[str, Any], path: Path, subject: str = "S01"
) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Windows")
    axis.set_xticks([0, 1], ["Non-FoG", "FoG"])
    axis.set_yticks([0, 1], ["Non-FoG", "FoG"])
    axis.set(
        xlabel="Predicted", ylabel="True", title=f"{subject} test confusion matrix"
    )
    cutoff = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                fontsize=14, color="white" if matrix[row, column] > cutoff else "black",
            )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed, args.deterministic)
    dataset = load_s01(args.data_dir)
    windows = make_windows(dataset)
    split = make_split(dataset, windows)
    scaler, scaler_fit_points = fit_scaler(dataset)
    normal_train, normal_validation = split_clean_normal_indices(
        dataset, windows, split
    )
    window_stats: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        counts = np.bincount(windows.label[indices], minlength=2)
        window_stats[name] = {
            "windows": int(len(indices)),
            "non_fog_windows": int(counts[0]),
            "fog_windows": int(counts[1]),
            "fog_percent": float(100 * counts[1] / counts.sum()),
        }
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "subject": SUBJECT_ID,
        "records": [record.record_id for record in dataset.records],
        "sampling_rate_hz": FS,
        "channels": list(dataset.channel_names),
        "split": {
            "strategy": "chronological record/block split with disjoint raw support",
            "train": "S01_seg000 plus S01_seg001 [0,50944)",
            "validation": "S01_seg001 [50944,end)",
            "test": "S01_seg002",
            "disclosure": "The inherited 50944 cut is an event-free, label-aware exploratory boundary; test record is independent.",
        },
        "windowing": {
            "shape_time_channel": [128, 9],
            "seconds": 2.0,
            "stride_samples": 64,
            "stride_seconds": 1.0,
            "label": "FOG when at least 50% of all 128 samples are FOG",
            "window_statistics": window_stats,
        },
        "scaler": {
            **scaler.as_dict(),
            "fit_split": "train only",
            "fit_class": "valid Non-FoG sample points only",
            "fit_points": scaler_fit_points,
            "scale_definition": "IQR/1.349 with train standard-deviation fallback",
        },
        "spectrum": {
            "input_shape": [9, 65],
            "formula": "log1p(abs(rfft(hann*x))**2 / sum(hann**2))",
            "frequency_range_hz": [0.0, 32.0],
            "frequency_resolution_hz": 0.5,
        },
        "gru_nbm": {
            "input": "corrupted Non-FoG log-power spectrum [B,9,65], transposed internally to [B,65,9]",
            "output": "clean Non-FoG reconstruction [B,9,65]",
            "training_class": "clean Non-FoG only, including 0.5 s label guard",
            "hidden_bottleneck": 64,
            "layers": 1,
            "decoder": "Linear(64,128)-GELU-Dropout(0.1)-Linear(128,585)-Softplus",
            "corruption": {
                "gaussian_noise_std_scaled_units": args.noise_std,
                "time_mask_probability": args.time_mask_probability,
                "time_mask_samples_uniform_inclusive": [4, 12],
                "channel_mask_probability": args.channel_mask_probability,
                "masked_channels": 1,
            },
        },
        "residual": {
            "formula": "observed_log_power - reconstructed_log_power",
            "shape": [9, 65],
            "standardization": None,
            "clipping": None,
        },
        "tcnm": {
            "input": "one current 2 s signed spectral residual [B,9,65]",
            "axis": "frequency, not multiple time windows",
            "hidden_channels": 48,
            "dilations": [1, 2, 4, 8],
            "kernel_size": 3,
            "convolutions_per_block": 2,
            "frequency_receptive_field_bins": 61,
            "normalization": "GroupNorm inside classifier only",
            "threshold": "validation-only maximum balanced accuracy",
        },
        "training": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(output_dir)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "leakage_controls": [
            "No raw sample support overlaps train and validation.",
            "Scaler and classifier class weight use training data only.",
            "GRU-NBM weights use clean Non-FoG training windows only.",
            "Validation selects both model epochs and decision threshold.",
            "Test is evaluated only after all choices are frozen.",
        ],
    }
    atomic_json_dump(config, output_dir / "config.json")
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train=split.train,
        validation=split.validation,
        test=split.test,
        gru_train_clean_normal=normal_train,
        gru_validation_clean_normal=normal_validation,
    )
    print(
        f"device={device} windows={window_stats} "
        f"GRU normal train/val={len(normal_train)}/{len(normal_validation)}",
        flush=True,
    )

    scaled_windows = {
        name: extract_windows(dataset, windows, indices, scaler)
        for name, indices in split.as_dict().items()
    }
    train_normal_x = extract_windows(dataset, windows, normal_train, scaler)
    validation_normal_x = extract_windows(dataset, windows, normal_validation, scaler)
    nbm, nbm_training = train_gru_nbm(
        args, train_normal_x, validation_normal_x, output_dir, device
    )

    residuals: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    process_arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        observed, reconstruction, residual = extract_residuals(
            args, nbm, scaled_windows[name], device
        )
        y = windows.label[indices].copy()
        residuals[name] = residual
        labels[name] = y
        diagnostics[name] = residual_diagnostics(residual, y)
        process_arrays.update(
            {
                f"{name}_observed_log_power": observed,
                f"{name}_reconstructed_log_power": reconstruction,
                f"{name}_signed_residual": residual,
                f"{name}_y": y,
                f"{name}_window_index": indices,
            }
        )
    atomic_json_dump(diagnostics, output_dir / "residual_diagnostics.json")
    atomic_npz_save(output_dir / "spectral_residuals.npz", **process_arrays)

    tcn_training, metrics, predictions = train_classifier(
        args, residuals, labels, output_dir, device
    )
    atomic_npz_save(output_dir / "predictions.npz", **predictions)
    write_csv(output_dir / "gru_nbm_history.csv", nbm_training["history"])
    write_csv(output_dir / "tcnm_history.csv", tcn_training["history"])
    plot_history(
        nbm_training["history"], "train_smooth_l1", "validation_smooth_l1",
        "GRU-NBM spectral reconstruction loss", output_dir / "gru_nbm_loss.png",
    )
    plot_history(
        tcn_training["history"], "train_bce", "validation_bce",
        "TCN-M classification loss", output_dir / "tcnm_loss.png",
    )
    plot_confusion(metrics["test"], output_dir / "test_confusion_matrix.png")

    test = metrics["test"]
    validation = metrics["validation"]
    summary = f"""# S01 spectral GRU-NBM + single-window TCN-M

## Main metrics

| Split | Accuracy | FoG Recall | Specificity | PR-AUC | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['pr_auc']:.6f} | {validation['f1']:.6f} | {validation['balanced_accuracy']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['pr_auc']:.6f} | {test['f1']:.6f} | {test['balanced_accuracy']:.6f} |

Validation-selected threshold: `{tcn_training['selected_threshold']:.4f}`.

Test confusion matrix (rows=true Non-FoG/FoG, columns=predicted Non-FoG/FoG):

```text
{test['confusion_matrix'][0]}
{test['confusion_matrix'][1]}
```

## Training

- Windows: train/validation/test = {window_stats['train']['windows']}/{window_stats['validation']['windows']}/{window_stats['test']['windows']}.
- GRU-NBM clean-normal train/validation = {len(normal_train)}/{len(normal_validation)}.
- GRU-NBM parameters = {nbm_training['parameter_count']:,}; best epoch = {nbm_training['best_epoch']}; best validation SmoothL1 = {nbm_training['best_validation_smooth_l1']:.8f}.
- TCN-M parameters = {tcn_training['parameter_count']:,}; best epoch = {tcn_training['best_epoch']}; best validation PR-AUC = {tcn_training['best_validation_pr_auc']:.6f}.
- Residual is raw signed log-power difference; no residual mean/std fitting, z-score, or clipping was used.

## Interpretation boundary

The GRU output is a reconstruction under the learned Non-FoG training distribution, not an observed counterfactual ideal signal. This is one subject, one chronological split, and one seed. Training residuals are in-sample with respect to the GRU-NBM, which is not test leakage but can create a train/deployment residual-domain gap.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_version": EXPERIMENT_VERSION,
            "main_test_metrics": {
                key: test[key]
                for key in ("accuracy", "fog_recall", "specificity", "pr_auc", "f1", "balanced_accuracy")
            },
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        },
        output_dir / "DONE.json",
    )
    print(
        "COMPLETE "
        f"accuracy={test['accuracy']:.6f} recall={test['fog_recall']:.6f} "
        f"specificity={test['specificity']:.6f} pr_auc={test['pr_auc']:.6f} "
        f"confusion={test['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
