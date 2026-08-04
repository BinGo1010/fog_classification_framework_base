#!/usr/bin/env python
"""Leakage-controlled within-S01 reconstruction NBM + residual TCN-M experiment.

This runner implements ``nonfog_gru_nbm_tcnm_within_subject_v1.md``.  The
training-pool classifier features are out-of-fold reconstruction residuals;
validation and test features share one final NBM fitted only within the train
pool.  The test labels are not consulted until the frozen threshold is chosen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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

from cnbr_fog.data import DaphnetDataset, Record
from cnbr_fog.evaluation import binary_metrics


EXPERIMENT = "nonfog_gru_nbm_tcnm_within_subject_v1"
SUBJECT = "S01"
FS = 64
WINDOW = 128
STRIDE = 64
ENDPOINT_LABEL_SAMPLES = 32
FOG_GUARD = 64
INNER_HALF_GAP = 64
OUTER_HALF_GAP = 128
OUTER_CUT = 50_944
RESIDUAL_EPS = 1e-6

SUBJECT_SPLITS: dict[str, dict[str, Any]] = {
    "S01": {"cut_record": "S01_seg001", "cut_sample": 50_944, "test_record": "S01_seg002", "ignored": []},
    "S02": {"cut_record": "S02_seg000", "cut_sample": 17_152, "test_record": "S02_seg001", "ignored": []},
    "S03": {"cut_record": "S03_seg001", "cut_sample": 8_576, "test_record": "S03_seg002", "ignored": ["S03_seg003"]},
    "S05": {"cut_record": "S05_seg004", "cut_sample": 4_736, "test_record": "S05_seg005", "ignored": []},
    "S06": {"cut_record": "S06_seg001", "cut_sample": 6_144, "test_record": "S06_seg002", "ignored": ["S06_seg003", "S06_seg004"]},
    "S07": {"cut_record": "S07_seg000", "cut_sample": 51_968, "test_record": "S07_seg001", "ignored": []},
    "S08": {"cut_record": "S08_seg002", "cut_sample": 1_920, "test_record": "S08_seg003", "ignored": []},
    "S09": {"cut_record": "S09_seg003", "cut_sample": 15_552, "test_record": "S09_seg004", "ignored": []},
}


@dataclass(frozen=True)
class Interval:
    record_id: str
    start: int
    end: int
    split: str
    block: int


@dataclass
class WindowSet:
    record_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    label: np.ndarray
    fog_fraction: np.ndarray
    clean_normal: np.ndarray
    split: np.ndarray
    block: np.ndarray

    def __len__(self) -> int:
        return int(self.start.size)

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray
    epsilon: float = 1e-6

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.median) / (
            self.iqr + self.epsilon
        )).astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "median": self.median.astype(float).tolist(),
            "iqr": self.iqr.astype(float).tolist(),
            "epsilon": float(self.epsilon),
        }


class GRUReconstructionNBM(nn.Module):
    """Single-direction, skip-free GRU denoising autoencoder."""

    def __init__(self, channels: int = 9, hidden: int = 64, bottleneck: int = 16):
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.bottleneck = int(bottleneck)
        self.encoder = nn.GRU(channels, hidden, num_layers=1, batch_first=True)
        self.to_bottleneck = nn.Linear(hidden, bottleneck)
        self.to_decoder = nn.Linear(bottleneck, hidden)
        self.decoder = nn.GRU(channels, hidden, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected [B,T,{self.channels}], got {tuple(x.shape)}")
        _, h = self.encoder(x)
        z = self.to_bottleneck(h[-1])
        h0 = self.to_decoder(z).unsqueeze(0)
        decoder_input = torch.zeros_like(x)
        decoded, _ = self.decoder(decoder_input, h0)
        return self.output(decoded)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int):
        super().__init__()
        self.left_padding = int(dilation) * (int(kernel) - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, 3, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(0.2)
        self.conv2 = CausalConv1d(out_channels, out_channels, 3, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.drop(F.relu(self.bn1(self.conv1(x))))
        y = F.relu(self.bn2(self.conv2(y)))
        return residual + y


class ResidualTCNM(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (9, 32, 64, 64, 128)
        dilations = (1, 2, 4, 8)
        self.blocks = nn.Sequential(
            *(TCNBlock(channels[i], channels[i + 1], dilations[i]) for i in range(4))
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        y = y.mean(dim=-1)
        return self.classifier(self.dropout(y)).squeeze(-1)


class InceptionModule1d(nn.Module):
    """Standard equal-length InceptionTime module for [B, C, T] input."""

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 32,
        branch_channels: int = 32,
        kernels: tuple[int, int, int] = (39, 19, 9),
    ):
        super().__init__()
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Inception kernels must be odd for symmetric same padding")
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(branch_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        branches = [convolution(bottleneck) for convolution in self.convolutions]
        branches.append(self.pool_branch(x))
        return F.relu(self.batch_norm(torch.cat(branches, dim=1)))


class InceptionResidualShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, source: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
        return F.relu(self.projection(source) + transformed)


class InceptionTimeClassifier(nn.Module):
    """Six Inception modules with a residual connection after every three."""

    def __init__(self, in_channels: int = 9):
        super().__init__()
        self.modules_1_to_3 = nn.ModuleList(
            [
                InceptionModule1d(in_channels),
                InceptionModule1d(128),
                InceptionModule1d(128),
            ]
        )
        self.residual_1 = InceptionResidualShortcut(in_channels)
        self.modules_4_to_6 = nn.ModuleList(
            [InceptionModule1d(128), InceptionModule1d(128), InceptionModule1d(128)]
        )
        self.residual_2 = InceptionResidualShortcut(128)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_source = x
        for module in self.modules_1_to_3:
            x = module(x)
        x = self.residual_1(first_source, x)
        second_source = x
        for module in self.modules_4_to_6:
            x = module(x)
        x = self.residual_2(second_source, x)
        return self.classifier(x.mean(dim=-1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"
        ),
    )
    parser.add_argument("--subject", choices=sorted(SUBJECT_SPLITS), default="S01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"{EXPERIMENT}_S01_seed20260802",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--final-nbm-max-epochs", type=int, default=50)
    parser.add_argument("--final-nbm-patience", type=int, default=8)
    parser.add_argument("--nbm-bottleneck", type=int, default=16)
    parser.add_argument(
        "--classifier",
        choices=("tcn_m", "inception_time"),
        default="tcn_m",
    )
    parser.add_argument(
        "--window-axis-center",
        action="store_true",
        help=(
            "Subtract each window/channel time mean after RobustScaler for the NBM, "
            "and subtract each window/channel residual time mean before TCN-M."
        ),
    )
    parser.add_argument(
        "--distributed-nbm-calibration",
        action="store_true",
        help=(
            "Within every available training block, use the chronological first "
            "80%% of clean non-FoG support for NBM fitting and the final 20%% for "
            "NBM validation/residual calibration, with a 2 s total guard gap."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fog_runs(y: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[0, np.asarray(y, dtype=np.int8), 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _safe_inner_cut(record: Record, nominal: int, low: int, high: int) -> int:
    """Move a 64-sample-grid cut to the nearest FoG-free protected location."""
    nominal = int(round(nominal / STRIDE) * STRIDE)
    for distance in range(0, max(high - low, STRIDE) + STRIDE, STRIDE):
        candidates = [nominal] if distance == 0 else [nominal - distance, nominal + distance]
        for candidate in candidates:
            if candidate - INNER_HALF_GAP - WINDOW - FOG_GUARD < low:
                continue
            if candidate + INNER_HALF_GAP + WINDOW + FOG_GUARD > high:
                continue
            guard = record.y[candidate - INNER_HALF_GAP : candidate + INNER_HALF_GAP]
            if not np.any(guard == 1):
                return int(candidate)
    raise ValueError(
        f"cannot find event-safe inner cut for {record.record_id} near {nominal}"
    )


def _generic_train_blocks(
    records: list[Record], cut_record: str, cut_sample: int
) -> list[Interval]:
    record_order = {record.record_id: index for index, record in enumerate(records)}
    cut_position = record_order[cut_record]
    supports: list[tuple[Record, int, int]] = []
    for index, record in enumerate(records):
        if index < cut_position:
            supports.append((record, 0, len(record.y)))
        elif index == cut_position:
            supports.append((record, 0, cut_sample - OUTER_HALF_GAP))
            break
    total = sum(end - start for _, start, end in supports)
    if total < 5 * (WINDOW + 2 * FOG_GUARD):
        raise ValueError(f"insufficient train support for five blocks: {SUBJECT} total={total}")
    nominal_boundaries = [total * index / 5.0 for index in range(1, 5)]
    cumulative = 0
    boundary_rows: list[tuple[str, int]] = []
    for target in nominal_boundaries:
        for record, start, end in supports:
            length = end - start
            if cumulative <= target < cumulative + length:
                local_nominal = start + int(round(target - cumulative))
                cut = _safe_inner_cut(record, local_nominal, start, end)
                boundary_rows.append((record.record_id, cut))
                break
            cumulative += length
        else:
            raise AssertionError(f"failed to map inner target {target}")
        cumulative = 0
    if len(set(boundary_rows)) != 4:
        raise ValueError(f"inner cuts are not unique: {boundary_rows}")
    boundary_lookup: dict[str, list[int]] = {}
    for record_id, cut in boundary_rows:
        boundary_lookup.setdefault(record_id, []).append(cut)
    intervals: list[Interval] = []
    block = 0
    for record, support_start, support_end in supports:
        cuts = sorted(boundary_lookup.get(record.record_id, []))
        cursor = support_start
        for cut in cuts:
            intervals.append(
                Interval(record.record_id, cursor, cut - INNER_HALF_GAP, "train", block)
            )
            block += 1
            cursor = cut + INNER_HALF_GAP
        intervals.append(Interval(record.record_id, cursor, support_end, "train", block))
    if block != 4 or {item.block for item in intervals} != set(range(5)):
        raise ValueError(f"failed to construct five blocks: {intervals}")
    return intervals


def build_intervals(records: list[Record]) -> list[Interval]:
    by_name = {record.record_id: record for record in records}
    protocol = SUBJECT_SPLITS[SUBJECT]
    required = {protocol["cut_record"], protocol["test_record"], *protocol["ignored"]}
    if not required.issubset(by_name):
        raise ValueError(f"missing {SUBJECT} protocol records: {sorted(required - set(by_name))}")
    cut_record = str(protocol["cut_record"])
    cut_sample = int(protocol["cut_sample"])
    test_record = str(protocol["test_record"])
    if cut_sample >= len(by_name[cut_record].y):
        raise ValueError(f"invalid cut for {cut_record}: {cut_sample}")
    if SUBJECT != "S01":
        intervals = _generic_train_blocks(records, cut_record, cut_sample)
        intervals.extend(
            [
                Interval(
                    cut_record,
                    cut_sample + OUTER_HALF_GAP,
                    len(by_name[cut_record].y),
                    "validation",
                    -1,
                ),
                Interval(test_record, 0, len(by_name[test_record].y), "test", -1),
            ]
        )
        return intervals
    # Blocks 1..4 partition the pre-cut portion of seg001.  A 2 s total gap
    # surrounds each internal boundary.  The outer train/validation gap is 4 s.
    cuts = (0, 12_736, 25_472, 38_208, OUTER_CUT)
    intervals = [
        Interval("S01_seg000", 0, len(by_name["S01_seg000"].y), "train", 0)
    ]
    for block in range(1, 5):
        left = cuts[block - 1] + (INNER_HALF_GAP if block > 1 else 0)
        right = cuts[block] - (INNER_HALF_GAP if block < 4 else OUTER_HALF_GAP)
        intervals.append(Interval("S01_seg001", left, right, "train", block))
    intervals.extend(
        [
            Interval(
                "S01_seg001",
                OUTER_CUT + OUTER_HALF_GAP,
                len(by_name["S01_seg001"].y),
                "validation",
                -1,
            ),
            Interval(
                "S01_seg002",
                0,
                len(by_name["S01_seg002"].y),
                "test",
                -1,
            ),
        ]
    )
    return intervals


def build_windows(records: list[Record], intervals: list[Interval]) -> WindowSet:
    lookup = {record.record_id: i for i, record in enumerate(records)}
    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "record_index",
            "start",
            "end",
            "label",
            "fog_fraction",
            "clean_normal",
            "split",
            "block",
        )
    }
    for interval in intervals:
        rec_idx = lookup[interval.record_id]
        record = records[rec_idx]
        for start in range(interval.start, interval.end - WINDOW + 1, STRIDE):
            end = start + WINDOW
            if not np.all(record.valid[start:end]):
                continue
            endpoint = record.y[end - ENDPOINT_LABEL_SAMPLES : end]
            fraction = float(np.mean(endpoint == 1))
            guard_start = start - FOG_GUARD
            guard_end = end + FOG_GUARD
            clean = bool(
                guard_start >= interval.start
                and guard_end <= interval.end
                and not np.any(record.y[guard_start:guard_end] == 1)
            )
            columns["record_index"].append(rec_idx)
            columns["start"].append(start)
            columns["end"].append(end)
            columns["label"].append(int(fraction >= 0.5))
            columns["fog_fraction"].append(fraction)
            columns["clean_normal"].append(clean)
            columns["split"].append(interval.split)
            columns["block"].append(interval.block)
    return WindowSet(
        record_index=np.asarray(columns["record_index"], dtype=np.int32),
        start=np.asarray(columns["start"], dtype=np.int32),
        end=np.asarray(columns["end"], dtype=np.int32),
        label=np.asarray(columns["label"], dtype=np.int8),
        fog_fraction=np.asarray(columns["fog_fraction"], dtype=np.float32),
        clean_normal=np.asarray(columns["clean_normal"], dtype=bool),
        split=np.asarray(columns["split"], dtype="U10"),
        block=np.asarray(columns["block"], dtype=np.int8),
    )


def raw_windows(records: list[Record], windows: WindowSet, indices: np.ndarray) -> np.ndarray:
    values = np.empty((len(indices), WINDOW, 9), dtype=np.float32)
    for row, index in enumerate(np.asarray(indices, dtype=np.int64)):
        rec = records[int(windows.record_index[index])]
        values[row] = rec.x[int(windows.start[index]) : int(windows.end[index])]
    return values


def window_axis_center(values: np.ndarray) -> np.ndarray:
    """Center every [window,time,channel] sequence independently by channel."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [window,time,channel], got {values.shape}")
    return np.ascontiguousarray(values - values.mean(axis=1, keepdims=True))


def prepare_nbm_windows(
    scaler: RobustScaler, raw: np.ndarray, *, center: bool
) -> np.ndarray:
    scaled = scaler.transform(raw)
    return window_axis_center(scaled) if center else np.ascontiguousarray(scaled)


def fit_scaler_unique_points(
    records: list[Record], windows: WindowSet, clean_indices: np.ndarray
) -> RobustScaler:
    masks: dict[int, np.ndarray] = {}
    for index in np.asarray(clean_indices, dtype=np.int64):
        rec_idx = int(windows.record_index[index])
        masks.setdefault(rec_idx, np.zeros(len(records[rec_idx].y), dtype=bool))
        masks[rec_idx][int(windows.start[index]) : int(windows.end[index])] = True
    chunks = [records[i].x[mask] for i, mask in masks.items() if np.any(mask)]
    values = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    iqr = q75 - q25
    if np.any(iqr <= 1e-6):
        raise ValueError(f"degenerate IQR channels: {np.flatnonzero(iqr <= 1e-6).tolist()}")
    return RobustScaler(median.astype(np.float32), iqr.astype(np.float32))


def corrupt(clean: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, np.ndarray]:
    output = clean.clone()
    modes = torch.rand(clean.shape[0], device=clean.device, generator=generator)
    gaussian = (modes >= 0.4) & (modes < 0.8)
    if torch.any(gaussian):
        noise = torch.randn(
            output[gaussian].shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        output[gaussian] += noise * 0.04
    masked_indices = torch.nonzero(modes >= 0.8, as_tuple=False).flatten().tolist()
    for index in masked_indices:
        length = int(torch.randint(4, 9, (1,), device=clean.device, generator=generator))
        start = int(
            torch.randint(0, WINDOW - length + 1, (1,), device=clean.device, generator=generator)
        )
        output[index, start : start + length, :] = 0.0
    counts = np.asarray(
        [int((modes < 0.4).sum()), int(gaussian.sum()), len(masked_indices)],
        dtype=np.int64,
    )
    return output, counts


def make_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x)).float()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def reconstruct(
    model: GRUReconstructionNBM,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for (batch,) in make_loader(x, batch_size, False, 0, 0):
        prediction = model(batch.to(device, non_blocking=True))
        output.append(prediction.cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def train_nbm(
    name: str,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int = 50,
    patience: int = 8,
    bottleneck: int = 16,
) -> tuple[GRUReconstructionNBM, dict[str, Any]]:
    set_seed(seed)
    if bottleneck <= 0:
        raise ValueError("NBM bottleneck must be positive")
    model = GRUReconstructionNBM(bottleneck=bottleneck).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    train_loader = make_loader(train_x, 128, True, seed, num_workers)
    validation_loader = make_loader(validation_x, 128, False, seed, num_workers)
    criterion = nn.SmoothL1Loss(beta=1.0)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint = output_dir / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("NBM max_epochs and patience must be positive")
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            network_input, counts = corrupt(clean, augmentation_generator)
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            prediction = model(network_input)
            loss = criterion(prediction, clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite NBM gradient in {name}")
            optimizer.step()
            total_loss += float(loss.detach()) * len(clean)
            total_n += len(clean)
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
                loss = criterion(model(clean), clean)
                val_loss += float(loss) * len(clean)
                val_n += len(clean)
        train_loss = total_loss / total_n
        validation_loss = val_loss / val_n
        scheduler.step(validation_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": lr,
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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_huber": validation_loss,
                    "seed": seed,
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"NBM {name} epoch={epoch:02d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={lr:.2e} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    rows_path = output_dir / "logs" / f"{name}_history.csv"
    write_csv(rows_path, history)
    summary = {
        "model_id": name,
        "seed": seed,
        "fit_windows": int(len(train_x)),
        "calibration_validation_windows": int(len(validation_x)),
        "maximum_epochs": int(max_epochs),
        "patience": int(patience),
        "bottleneck": int(bottleneck),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history_file": str(rows_path.relative_to(output_dir)),
        "checkpoint_file": str(checkpoint.relative_to(output_dir)),
    }
    return model, {"summary": summary, "history": history}


def calibrate(
    model: GRUReconstructionNBM,
    calibration_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mu = reconstruct(model, calibration_x, device)
    error = calibration_x - mu
    bias = np.median(error, axis=(0, 1)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(np.abs(error - bias[None, None, :]), axis=(0, 1))
    sigma = np.maximum(sigma_raw, 0.05).astype(np.float32)
    return bias, sigma, {
        "bias": bias.astype(float).tolist(),
        "sigma_raw": sigma_raw.astype(float).tolist(),
        "sigma": sigma.astype(float).tolist(),
        "sigma_floor": 0.05,
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05).astype(int).tolist(),
        "calibration_windows": int(len(calibration_x)),
    }


def normalized_residual(
    model: GRUReconstructionNBM,
    scaler: RobustScaler,
    bias: np.ndarray,
    sigma: np.ndarray,
    raw_x: np.ndarray,
    device: torch.device,
    *,
    center_windows: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = prepare_nbm_windows(scaler, raw_x, center=center_windows)
    mu = reconstruct(model, scaled, device)
    residual_unclipped = (scaled - mu - bias[None, None, :]) / (
        sigma[None, None, :] + RESIDUAL_EPS
    )
    residual = np.clip(residual_unclipped, -12.0, 12.0).astype(np.float32)
    if center_windows:
        # The classifier receives an exactly zero-time-mean sequence per axis.
        # Centering follows residual clipping; therefore final values are not
        # mathematically restricted to [-12, 12].
        residual = window_axis_center(residual)
    return residual, mu, residual_unclipped.astype(np.float32)


def distributed_nbm_fit_cal_indices(
    windows: WindowSet,
    candidate_indices: np.ndarray,
    *,
    fit_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Create time-disjoint 80/20 NBM fit/calibration parts in every block."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    clean = candidate_indices[windows.clean_normal[candidate_indices]]
    fit_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for block in sorted({int(windows.block[index]) for index in clean}):
        group = clean[windows.block[clean] == block]
        order = np.lexsort((windows.start[group], windows.record_index[group]))
        group = group[order]
        if len(group) < 2:
            raise ValueError(
                f"too few clean windows for distributed calibration: "
                f"block={block} n={len(group)}"
            )
        split_position = min(
            max(int(math.floor(fit_ratio * len(group))), 1), len(group) - 1
        )
        fit_seed = group[:split_position]
        cal_seed = group[split_position:]
        boundary_record = int(windows.record_index[cal_seed[0]])
        boundary_sample = int(windows.start[cal_seed[0]])
        fit = fit_seed[
            (windows.record_index[fit_seed] != boundary_record)
            | (windows.end[fit_seed] <= boundary_sample - INNER_HALF_GAP)
        ]
        cal = cal_seed[
            (windows.record_index[cal_seed] != boundary_record)
            | (windows.start[cal_seed] >= boundary_sample + INNER_HALF_GAP)
        ]
        if min(len(fit), len(cal)) == 0:
            raise ValueError(
                f"empty distributed component: block={block} "
                f"fit={len(fit)} cal={len(cal)}"
            )
        fit_same_record = fit[windows.record_index[fit] == boundary_record]
        cal_same_record = cal[windows.record_index[cal] == boundary_record]
        raw_support_gap_samples = None
        if len(fit_same_record) and len(cal_same_record):
            raw_support_gap_samples = int(
                windows.start[cal_same_record].min()
                - windows.end[fit_same_record].max()
            )
            if raw_support_gap_samples < 2 * INNER_HALF_GAP:
                raise AssertionError("distributed NBM fit/calibration guard gap failed")
        fit_parts.append(fit)
        cal_parts.append(cal)
        details.append(
            {
                "block": block,
                "record_indices": sorted(
                    {int(value) for value in windows.record_index[group]}
                ),
                "clean_windows_before_guard": int(len(group)),
                "requested_split_position": int(split_position),
                "boundary_record_index": boundary_record,
                "boundary_sample": boundary_sample,
                "fit_windows": int(len(fit)),
                "calibration_windows": int(len(cal)),
                "raw_support_gap_samples": raw_support_gap_samples,
            }
        )
    if not fit_parts or not cal_parts:
        raise ValueError("distributed NBM split produced no fit/calibration groups")
    fit_idx = np.concatenate(fit_parts).astype(np.int64, copy=False)
    cal_idx = np.concatenate(cal_parts).astype(np.int64, copy=False)
    if np.intersect1d(fit_idx, cal_idx).size:
        raise AssertionError("distributed NBM fit/calibration overlap")
    return fit_idx, cal_idx, details


def train_crossfit(
    records: list[Record],
    windows: WindowSet,
    train_indices: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    center_windows: bool,
    distributed_calibration: bool = False,
    nbm_bottleneck: int = 16,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    residuals = np.empty((len(train_indices), WINDOW, 9), dtype=np.float32)
    assigned = np.zeros(len(train_indices), dtype=bool)
    local_lookup = {int(index): row for row, index in enumerate(train_indices)}
    fold_summaries: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for holdout in range(5):
        if distributed_calibration:
            source_blocks = [b for b in range(5) if b != holdout]
            source_idx = train_indices[
                np.isin(windows.block[train_indices], source_blocks)
            ]
            fit_idx, cal_idx, split_details = distributed_nbm_fit_cal_indices(
                windows, source_idx
            )
            calibration_block = None
            fit_blocks = source_blocks
            calibration_blocks = source_blocks
        else:
            calibration_block = (holdout + 1) % 5
            fit_blocks = [b for b in range(5) if b not in {holdout, calibration_block}]
            calibration_blocks = [calibration_block]
            split_details = []
            fit_idx = train_indices[
                windows.clean_normal[train_indices]
                & np.isin(windows.block[train_indices], fit_blocks)
            ]
            cal_idx = train_indices[
                windows.clean_normal[train_indices]
                & (windows.block[train_indices] == calibration_block)
            ]
        hold_idx = train_indices[windows.block[train_indices] == holdout]
        if min(len(fit_idx), len(cal_idx), len(hold_idx)) == 0:
            raise ValueError(
                f"empty crossfit component fold={holdout}: "
                f"fit={len(fit_idx)} cal={len(cal_idx)} hold={len(hold_idx)}"
            )
        if np.intersect1d(hold_idx, np.r_[fit_idx, cal_idx]).size:
            raise AssertionError("NBM saw a held-out TCN training window")
        scaler = fit_scaler_unique_points(records, windows, fit_idx)
        train_x = prepare_nbm_windows(
            scaler, raw_windows(records, windows, fit_idx), center=center_windows
        )
        cal_x = prepare_nbm_windows(
            scaler, raw_windows(records, windows, cal_idx), center=center_windows
        )
        name = f"inner_fold_{holdout + 1:02d}_nbm"
        model, training = train_nbm(
            name,
            train_x,
            cal_x,
            output_dir,
            device,
            seed + holdout,
            num_workers,
            bottleneck=nbm_bottleneck,
        )
        bias, sigma, calibration = calibrate(model, cal_x, device)
        hold_raw = raw_windows(records, windows, hold_idx)
        fold_residual, _, _ = normalized_residual(
            model,
            scaler,
            bias,
            sigma,
            hold_raw,
            device,
            center_windows=center_windows,
        )
        for source_row, index in enumerate(hold_idx):
            local_row = local_lookup[int(index)]
            if assigned[local_row]:
                raise AssertionError("duplicate OOF assignment")
            residuals[local_row] = fold_residual[source_row]
            assigned[local_row] = True
            rec = records[int(windows.record_index[index])]
            manifest.append(
                {
                    "sample_id": f"{rec.record_id}:{int(windows.start[index])}:{int(windows.end[index])}",
                    "subject_id": SUBJECT,
                    "record_id": rec.record_id,
                    "window_start": int(windows.start[index]),
                    "window_end": int(windows.end[index]),
                    "label": int(windows.label[index]),
                    "outer_split": "train",
                    "inner_fold": holdout + 1,
                    "nbm_model_id": name,
                    "nbm_seen_this_window": False,
                }
            )
        fold_summary = {
            **training["summary"],
            "holdout_block": holdout,
            "calibration_block": calibration_block,
            "calibration_blocks": calibration_blocks,
            "fit_blocks": fit_blocks,
            "calibration_strategy": (
                "distributed_chronological_80_20_per_source_block"
                if distributed_calibration
                else "next_block_cyclic"
            ),
            "distributed_split_details": split_details,
            "holdout_windows": int(len(hold_idx)),
            "holdout_fog_windows": int(windows.label[hold_idx].sum()),
            "scaler": scaler.as_dict(),
            "residual_calibration": calibration,
        }
        fold_summaries.append({**fold_summary, "history": training["history"]})
        write_json(output_dir / "artifacts" / f"{name}.json", fold_summary)
    if not np.all(assigned):
        raise AssertionError(f"unassigned OOF rows: {np.flatnonzero(~assigned).tolist()}")
    if any(row["nbm_seen_this_window"] for row in manifest):
        raise AssertionError("OOF leakage flag failed")
    return residuals, fold_summaries, manifest


def fit_final_nbm(
    records: list[Record],
    windows: WindowSet,
    train_indices: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    center_windows: bool,
    final_max_epochs: int,
    final_patience: int,
    distributed_calibration: bool = False,
    nbm_bottleneck: int = 16,
) -> tuple[GRUReconstructionNBM, RobustScaler, np.ndarray, np.ndarray, dict[str, Any]]:
    if distributed_calibration:
        fit_idx, cal_idx, split_details = distributed_nbm_fit_cal_indices(
            windows, train_indices
        )
        fit_blocks = [0, 1, 2, 3, 4]
        calibration_blocks = [0, 1, 2, 3, 4]
        calibration_block = None
    else:
        fit_idx = train_indices[
            windows.clean_normal[train_indices]
            & np.isin(windows.block[train_indices], [0, 1, 2, 3])
        ]
        cal_idx = train_indices[
            windows.clean_normal[train_indices] & (windows.block[train_indices] == 4)
        ]
        fit_blocks = [0, 1, 2, 3]
        calibration_blocks = [4]
        calibration_block = 4
        split_details = []
    scaler = fit_scaler_unique_points(records, windows, fit_idx)
    fit_x = prepare_nbm_windows(
        scaler, raw_windows(records, windows, fit_idx), center=center_windows
    )
    cal_x = prepare_nbm_windows(
        scaler, raw_windows(records, windows, cal_idx), center=center_windows
    )
    model, training = train_nbm(
        "final_nbm",
        fit_x,
        cal_x,
        output_dir,
        device,
        seed + 100,
        num_workers,
        max_epochs=final_max_epochs,
        patience=final_patience,
        bottleneck=nbm_bottleneck,
    )
    bias, sigma, calibration = calibrate(model, cal_x, device)
    summary = {
        **training["summary"],
        "fit_blocks": fit_blocks,
        "calibration_block": calibration_block,
        "calibration_blocks": calibration_blocks,
        "calibration_strategy": (
            "distributed_chronological_80_20_per_source_block"
            if distributed_calibration
            else "single_block"
        ),
        "distributed_split_details": split_details,
        "scaler": scaler.as_dict(),
        "residual_calibration": calibration,
        "history": training["history"],
    }
    write_json(output_dir / "artifacts" / "final_nbm.json", {
        key: value for key, value in summary.items() if key != "history"
    })
    return model, scaler, bias, sigma, summary


def classifier_loader(
    x: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).float(),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        ),
        batch_size=128,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def classifier_predict(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_x, batch_y in classifier_loader(x, y, False, 0, 0):
        logits = model(batch_x.to(device, non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_y.numpy())
    return np.concatenate(labels).astype(np.int8), np.concatenate(probabilities)


def train_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    classifier_name: str = "tcn_m",
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    if classifier_name == "tcn_m":
        model = ResidualTCNM().to(device)
    elif classifier_name == "inception_time":
        model = InceptionTimeClassifier().to(device)
    else:
        raise ValueError(f"unknown classifier: {classifier_name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = int(train_y.sum())
    n_neg = int(len(train_y) - n_pos)
    if min(n_pos, n_neg) == 0:
        raise ValueError("classifier training requires both classes")
    pos_weight_value = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    loader = classifier_loader(train_x, train_y, True, seed, num_workers)
    checkpoint = output_dir / "checkpoints" / f"{classifier_name}.pt"
    best_pr = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite {classifier_name} gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            total_n += len(batch_x)
        val_true, val_prob = classifier_predict(model, validation_x, validation_y, device)
        val_pr = float(average_precision_score(val_true, val_prob))
        train_loss = total_loss / total_n
        improved = val_pr > best_pr + 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_loss,
                "validation_pr_auc": val_pr,
                "improved": improved,
            }
        )
        if improved:
            best_pr = val_pr
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_pr_auc": val_pr,
                    "seed": seed,
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"{classifier_name} epoch={epoch:02d} train_bce={train_loss:.7f} "
            f"val_pr_auc={val_pr:.7f} stale={stale}/6",
            flush=True,
        )
        if stale >= 6:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(output_dir / "logs" / f"{classifier_name}_history.csv", history)
    return model, {
        "model_name": classifier_name,
        "seed": seed,
        "maximum_epochs": 30,
        "patience": 6,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr,
        "positive_weight": pos_weight_value,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
    }


def choose_document_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    best_key = (-math.inf, -math.inf, -math.inf)
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        key = (
            float(metrics["balanced_accuracy"] or 0.0),
            float(metrics["f1"] or 0.0),
            float(threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def add_requested_macro_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add ACC and unweighted class-macro precision/recall/F1 for C=2."""
    tn = int(metrics["tn"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    tp = int(metrics["tp"])

    def safe_div(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision_non_fog = safe_div(tn, tn + fn)
    precision_fog = safe_div(tp, tp + fp)
    recall_non_fog = safe_div(tn, tn + fp)
    recall_fog = safe_div(tp, tp + fn)
    f1_non_fog = safe_div(2 * tn, 2 * tn + fp + fn)
    f1_fog = safe_div(2 * tp, 2 * tp + fp + fn)
    metrics.update(
        {
            "acc": safe_div(tn + tp, tn + fp + fn + tp),
            "macro_precision": 0.5 * (precision_non_fog + precision_fog),
            "macro_recall": 0.5 * (recall_non_fog + recall_fog),
            "macro_f1": 0.5 * (f1_non_fog + f1_fog),
            "per_class_for_macro": {
                "non_fog": {
                    "precision": precision_non_fog,
                    "recall": recall_non_fog,
                    "f1": f1_non_fog,
                },
                "fog": {
                    "precision": precision_fog,
                    "recall": recall_fog,
                    "f1": f1_fog,
                },
            },
            "macro_zero_division": 0,
        }
    )
    return metrics


def residual_diagnostics(residual: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in (("non_fog", labels == 0), ("fog", labels == 1)):
        values = residual[mask]
        result[name] = {
            "windows": int(mask.sum()),
            "median_absolute_residual": float(np.median(np.abs(values))),
            "mean_absolute_residual": float(np.mean(np.abs(values))),
            "absolute_residual_gt_3_fraction": float(np.mean(np.abs(values) > 3.0)),
            "clip_fraction": float(np.mean(np.abs(values) >= 12.0)),
            "channel_median": np.median(values, axis=(0, 1)).astype(float).tolist(),
            "channel_mad": np.median(
                np.abs(values - np.median(values, axis=(0, 1))[None, None, :]),
                axis=(0, 1),
            ).astype(float).tolist(),
        }
    return result


def save_plots(
    output_dir: Path,
    nbm_runs: list[dict[str, Any]],
    classifier_training: dict[str, Any],
    confusion: list[list[int]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, run in zip(axes.flat, nbm_runs):
        history = run["history"]
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_huber"] for row in history], label="train")
        ax.plot(epochs, [row["validation_huber"] for row in history], label="validation")
        ax.axvline(run["best_epoch"], color="black", linestyle="--", linewidth=0.8)
        ax.set_title(run["model_id"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_dir / "nbm_training_validation_loss.png", dpi=180)
    plt.close(fig)

    classifier_name = str(classifier_training.get("model_name", "tcn_m"))
    display_name = "InceptionTime" if classifier_name == "inception_time" else "TCN-M"
    history = classifier_training["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_weighted_bce"] for row in history])
    axes[0].set_title(f"{display_name} training weighted BCE")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["validation_pr_auc"] for row in history])
    axes[1].axvline(classifier_training["best_epoch"], color="black", linestyle="--")
    axes[1].set_title(f"{display_name} validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / f"{classifier_name}_training_validation.png", dpi=180)
    plt.close(fig)

    cm = np.asarray(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{SUBJECT} {display_name} test confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)


def event_results(
    records: list[Record],
    windows: WindowSet,
    test_indices: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_record_ids = {
        records[int(windows.record_index[index])].record_id for index in test_indices
    }
    for rec_idx, record in enumerate(records):
        if record.record_id not in test_record_ids:
            continue
        relevant = np.flatnonzero(windows.record_index[test_indices] == rec_idx)
        for event_id, (start, end) in enumerate(fog_runs(record.y), 1):
            candidates: list[int] = []
            for local in relevant:
                index = int(test_indices[local])
                endpoint_start = int(windows.end[index]) - ENDPOINT_LABEL_SAMPLES
                overlap = max(0, min(int(windows.end[index]), end) - max(endpoint_start, start))
                if overlap >= ENDPOINT_LABEL_SAMPLES / 2:
                    candidates.append(int(local))
            detected = bool(candidates and np.any(prediction[candidates] == 1))
            rows.append(
                {
                    "record_id": record.record_id,
                    "event_id": event_id,
                    "start_sample": start,
                    "end_sample": end,
                    "duration_seconds": (end - start) / FS,
                    "eligible_positive_windows": len(candidates),
                    "detected": detected,
                }
            )
    return rows


def split_statistics(windows: WindowSet, indices: np.ndarray) -> dict[str, Any]:
    labels = windows.label[indices]
    return {
        "windows": int(len(indices)),
        "non_fog_windows": int(np.sum(labels == 0)),
        "fog_windows": int(np.sum(labels == 1)),
        "clean_normal_windows": int(np.sum(windows.clean_normal[indices])),
    }


def main() -> None:
    global SUBJECT
    args = parse_args()
    if args.nbm_bottleneck <= 0:
        raise ValueError("--nbm-bottleneck must be positive")
    SUBJECT = str(args.subject)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = DaphnetDataset.load(args.data_dir)
    records = [record for record in dataset.records if record.subject_id == SUBJECT]
    if dataset.sampling_rate_hz != FS or dataset.n_channels != 9:
        raise ValueError(
            f"expected 64 Hz/9 channels, got {dataset.sampling_rate_hz}/{dataset.n_channels}"
        )
    intervals = build_intervals(records)
    windows = build_windows(records, intervals)
    train_indices = windows.indices("train")
    validation_indices = windows.indices("validation")
    test_indices = windows.indices("test")
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        if np.unique(windows.label[indices]).size != 2:
            raise ValueError(f"{name} lacks one classification label")
    event_counts = {
        split: int(
            sum(
                1
                for interval in intervals
                if interval.split == split
                for start, end in fog_runs(records[[r.record_id for r in records].index(interval.record_id)].y)
                if start >= interval.start and end <= interval.end
            )
        )
        for split in ("train", "validation", "test")
    }
    # Raw supports are interval-disjoint by construction.  These assertions make
    # the intended gaps machine-verifiable.
    by_record_intervals: dict[str, list[Interval]] = {}
    for item in intervals:
        by_record_intervals.setdefault(item.record_id, []).append(item)
    for record_intervals in by_record_intervals.values():
        record_intervals.sort(key=lambda item: item.start)
        for left, right in zip(record_intervals, record_intervals[1:]):
            if right.start - left.end < 2 * FS:
                raise AssertionError(f"insufficient boundary gap: {left} -> {right}")
    distributed_preflight: dict[str, Any] | None = None
    if args.distributed_nbm_calibration:
        final_fit, final_cal, final_details = distributed_nbm_fit_cal_indices(
            windows, train_indices
        )
        inner_details = []
        for holdout in range(5):
            source = train_indices[windows.block[train_indices] != holdout]
            fold_fit, fold_cal, fold_details = distributed_nbm_fit_cal_indices(
                windows, source
            )
            inner_details.append(
                {
                    "holdout_block": holdout,
                    "fit_windows": int(len(fold_fit)),
                    "calibration_windows": int(len(fold_cal)),
                    "per_block": fold_details,
                }
            )
        distributed_preflight = {
            "final_fit_windows": int(len(final_fit)),
            "final_calibration_windows": int(len(final_cal)),
            "final_per_block": final_details,
            "inner_folds": inner_details,
        }
    protocol = {
        "experiment": EXPERIMENT,
        "framework_document": str(REPO_ROOT / "nonfog_gru_nbm_tcnm_within_subject_v1.md"),
        "framework_sha256": sha256(REPO_ROOT / "nonfog_gru_nbm_tcnm_within_subject_v1.md"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": SUBJECT,
        "data_dir": str(args.data_dir.resolve()),
        "dataset_manifest_sha256": sha256(args.data_dir / "manifest.csv"),
        "seed": args.seed,
        "device": str(device),
        "sampling_rate_hz": FS,
        "channels": list(dataset.channel_names),
        "windowing": {
            "window_samples": WINDOW,
            "window_seconds": WINDOW / FS,
            "stride_samples": STRIDE,
            "stride_seconds": STRIDE / FS,
            "label_rule": "FoG iff >=50% of final 32 samples (0.5 s) are FoG",
            "clean_normal_rule": "full 2 s window plus 1 s pre/post guard contain no FoG",
        },
        "outer_split": {
            "priority": "historical frozen within-subject test record, FoG-positive chronological validation, then proportions",
            "intervals": [asdict(item) for item in intervals],
            "train_validation_total_gap_samples": 2 * OUTER_HALF_GAP,
            "train_validation_total_gap_seconds": 2 * OUTER_HALF_GAP / FS,
            "test_record": SUBJECT_SPLITS[SUBJECT]["test_record"],
            "ignored_records": SUBJECT_SPLITS[SUBJECT]["ignored"],
            "test_segment_disjoint": True,
            "exact_70_15_15_achievable": False,
        },
        "inner_crossfit": {
            "folds": 5,
            "blocks": [0, 1, 2, 3, 4],
            "calibration_rule": (
                "within each of the four non-holdout blocks, chronological first 80% "
                "fits NBM and final 20% validates/calibrates NBM"
                if args.distributed_nbm_calibration
                else "next block cyclically; other three source blocks fit NBM"
            ),
            "inner_total_gap_seconds": 2 * INNER_HALF_GAP / FS,
        },
        "nbm_calibration": {
            "strategy": (
                "distributed_chronological_80_20_per_available_block"
                if args.distributed_nbm_calibration
                else "single_whole_block"
            ),
            "fit_ratio": 0.8 if args.distributed_nbm_calibration else None,
            "calibration_ratio": 0.2 if args.distributed_nbm_calibration else None,
            "total_guard_gap_seconds": (
                2 * INNER_HALF_GAP / FS if args.distributed_nbm_calibration else None
            ),
            "only_changed_variable": bool(args.distributed_nbm_calibration),
            "preflight_counts": distributed_preflight,
        },
        "scaler": {"type": "per-channel median/IQR", "epsilon": 1e-6, "clip": None},
        "window_axis_centering": {
            "enabled": bool(args.window_axis_center),
            "nbm": (
                "after RobustScaler, subtract each window/channel mean over 128 samples "
                "from both denoising input and clean reconstruction target"
                if args.window_axis_center
                else "disabled"
            ),
            "tcn_m": (
                "after standardized-residual clipping, subtract each residual window/channel "
                "mean over 128 samples; no second clipping"
                if args.window_axis_center
                else "disabled"
            ),
        },
        "nbm": {
            "architecture": (
                f"GRU(9,64)->Linear(64,{args.nbm_bottleneck})->"
                f"Linear({args.nbm_bottleneck},64)->zero-input GRU(9,64)->Linear(64,9)"
            ),
            "bottleneck": int(args.nbm_bottleneck),
            "bidirectional": False,
            "skip_connections": False,
            "loss": "SmoothL1Loss(beta=1.0)",
            "augmentation": {"clean": 0.4, "gaussian_std_0.04": 0.4, "mask_4_to_8": 0.2},
            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
            "batch_size": 128,
            "inner_crossfit_max_epochs": 50,
            "inner_crossfit_patience": 8,
            "final_max_epochs": int(args.final_nbm_max_epochs),
            "final_patience": int(args.final_nbm_patience),
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)",
            "gradient_clip": 1.0,
        },
        "residual": {
            "formula": "clip((X_scaled-mu-b)/(sigma+1e-6),-12,12)",
            "b": "per-channel calibration median",
            "sigma": "1.4826 * per-channel calibration MAD, floor 0.05",
        },
        "classifier": {
            "name": args.classifier,
            "architecture": (
                "six InceptionTime modules; each uses shared 1x1 bottleneck 32, "
                "three same-length Conv1d branches with kernels 39/19/9 and 32 "
                "channels each, plus MaxPool3->Conv1x1 branch with 32 channels; "
                "BatchNorm+ReLU; residual shortcut after modules 3 and 6; GAP; Linear(128,1)"
                if args.classifier == "inception_time"
                else "causal TCN blocks 9-32-64-64-128, dilations 1-2-4-8, global average, dropout 0.3, linear"
            ),
            "input_shape": "[batch, 9, 128]",
            "output": "one logit; sigmoid used for FoG probability",
            "loss": "BCEWithLogitsLoss(pos_weight=N_nonFoG/N_FoG)",
            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
            "batch_size": 128,
            "max_epochs": 30,
            "patience": 6,
            "monitor": "validation PR-AUC",
            "threshold": "validation balanced accuracy over 0.05..0.95 step 0.01; ties F1 then higher threshold",
        },
        "split_statistics": {
            "train": split_statistics(windows, train_indices),
            "validation": split_statistics(windows, validation_indices),
            "test": split_statistics(windows, test_indices),
        },
        "fog_events_fully_contained": event_counts,
        "known_specification_choices": [
            "The document did not fix the binary window label rule; the prior S01 endpoint rule is retained.",
            (
                "InceptionTime uses non-causal symmetric same-length convolutions within the available 2 s window."
                if args.classifier == "inception_time"
                else "Causal convolution was selected from the document's 'Causal/same-length' alternative."
            ),
            "TCN residual-add output has no extra activation because none is specified after the addition.",
            "The 70/15/15 ratio is secondary to keeping R02 as an independent test recording and both validation/test FoG-positive.",
        ],
    }
    write_json(output_dir / "config.json", protocol)
    print(
        f"PREFLIGHT device={device} split_stats={protocol['split_statistics']} "
        f"events={event_counts}",
        flush=True,
    )
    if args.dry_run:
        write_json(output_dir / "DRY_RUN.json", {"status": "complete", "protocol": EXPERIMENT})
        return

    oof_residual, inner_runs, oof_manifest = train_crossfit(
        records,
        windows,
        train_indices,
        output_dir,
        device,
        args.seed,
        args.num_workers,
        args.window_axis_center,
        args.distributed_nbm_calibration,
        args.nbm_bottleneck,
    )
    write_csv(output_dir / "artifacts" / "oof_manifest.csv", oof_manifest)
    final_model, final_scaler, final_bias, final_sigma, final_run = fit_final_nbm(
        records,
        windows,
        train_indices,
        output_dir,
        device,
        args.seed,
        args.num_workers,
        args.window_axis_center,
        args.final_nbm_max_epochs,
        args.final_nbm_patience,
        args.distributed_nbm_calibration,
        args.nbm_bottleneck,
    )
    validation_raw = raw_windows(records, windows, validation_indices)
    test_raw = raw_windows(records, windows, test_indices)
    validation_residual, validation_mu, validation_unclipped = normalized_residual(
        final_model,
        final_scaler,
        final_bias,
        final_sigma,
        validation_raw,
        device,
        center_windows=args.window_axis_center,
    )
    test_residual, test_mu, test_unclipped = normalized_residual(
        final_model,
        final_scaler,
        final_bias,
        final_sigma,
        test_raw,
        device,
        center_windows=args.window_axis_center,
    )
    train_y = windows.label[train_indices]
    validation_y = windows.label[validation_indices]
    test_y = windows.label[test_indices]
    save_npz(
        output_dir / "artifacts" / "residuals.npz",
        train_oof_residual=oof_residual,
        train_y=train_y,
        train_window_index=train_indices,
        validation_residual=validation_residual,
        validation_y=validation_y,
        validation_mu=validation_mu,
        validation_unclipped=validation_unclipped,
        validation_window_index=validation_indices,
        test_residual=test_residual,
        test_y=test_y,
        test_mu=test_mu,
        test_unclipped=test_unclipped,
        test_window_index=test_indices,
    )
    diagnostics = {
        "train_oof": residual_diagnostics(oof_residual, train_y),
        "validation_final_nbm": residual_diagnostics(validation_residual, validation_y),
        "test_final_nbm": residual_diagnostics(test_residual, test_y),
        "oof_vs_validation_nonfog_median_abs_ratio": float(
            np.median(np.abs(oof_residual[train_y == 0]))
            / np.median(np.abs(validation_residual[validation_y == 0]))
        ),
        "test_vs_validation_nonfog_median_abs_ratio": float(
            np.median(np.abs(test_residual[test_y == 0]))
            / np.median(np.abs(validation_residual[validation_y == 0]))
        ),
    }
    write_json(output_dir / "residual_diagnostics.json", diagnostics)

    classifier, classifier_training = train_classifier(
        oof_residual,
        train_y,
        validation_residual,
        validation_y,
        output_dir,
        device,
        args.seed,
        args.num_workers,
        args.classifier,
    )
    val_true, val_prob = classifier_predict(
        classifier, validation_residual, validation_y, device
    )
    threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
    test_true, test_prob = classifier_predict(classifier, test_residual, test_y, device)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    validation_metrics = add_requested_macro_metrics(validation_metrics)
    test_metrics = add_requested_macro_metrics(test_metrics)
    val_pred = (val_prob >= threshold).astype(np.int8)
    test_pred = (test_prob >= threshold).astype(np.int8)
    test_events = event_results(records, windows, test_indices, test_pred)
    event_summary = {
        "events": len(test_events),
        "detected": int(sum(bool(row["detected"]) for row in test_events)),
        "event_recall": float(np.mean([row["detected"] for row in test_events])),
        "definition": "event detected when any threshold-positive window has >=16/32 endpoint samples inside the event",
    }
    metrics = {
        "selected_threshold": threshold,
        "threshold_source": "validation only",
        "validation": validation_metrics,
        "test": test_metrics,
        "test_event_detection": event_summary,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "training.json", {
        "inner_nbm": [{key: value for key, value in run.items() if key != "history"} for run in inner_runs],
        "final_nbm": {key: value for key, value in final_run.items() if key != "history"},
        "classifier": {
            key: value for key, value in classifier_training.items() if key != "history"
        },
    })
    prediction_rows: list[dict[str, Any]] = []
    for split, indices, probability, prediction in (
        ("validation", validation_indices, val_prob, val_pred),
        ("test", test_indices, test_prob, test_pred),
    ):
        for local, index in enumerate(indices):
            record = records[int(windows.record_index[index])]
            prediction_rows.append(
                {
                    "split": split,
                    "subject_id": SUBJECT,
                    "record_id": record.record_id,
                    "window_start": int(windows.start[index]),
                    "window_end": int(windows.end[index]),
                    "y_true": int(windows.label[index]),
                    "fog_fraction_final_0p5s": float(windows.fog_fraction[index]),
                    "fog_probability": float(probability[local]),
                    "threshold": threshold,
                    "y_pred": int(prediction[local]),
                }
            )
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "test_event_detection.csv", test_events)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true\\pred": "non-FoG", "non-FoG": test_metrics["tn"], "FoG": test_metrics["fp"]},
            {"true\\pred": "FoG", "non-FoG": test_metrics["fn"], "FoG": test_metrics["tp"]},
        ],
    )
    save_plots(
        output_dir,
        inner_runs + [final_run],
        classifier_training,
        test_metrics["confusion_matrix"],
    )
    warnings: list[str] = []
    ratio = diagnostics["oof_vs_validation_nonfog_median_abs_ratio"]
    if ratio < 0.67 or ratio > 1.5:
        warnings.append(
            f"OOF/validation non-FoG median |r| ratio={ratio:.3f} indicates residual distribution shift."
        )
    test_shift_ratio = diagnostics["test_vs_validation_nonfog_median_abs_ratio"]
    if test_shift_ratio < 0.67 or test_shift_ratio > 1.5:
        warnings.append(
            f"Test/validation non-FoG median |r| ratio={test_shift_ratio:.3f} indicates recording-domain shift."
        )
    epoch_one_models = [
        run["model_id"] for run in inner_runs if int(run["best_epoch"]) == 1
    ]
    if epoch_one_models:
        warnings.append(
            "Validation Huber was best at epoch 1 for " + ", ".join(epoch_one_models)
            + "; continuous-block reconstruction distributions are heterogeneous."
        )
    if test_metrics["auprc"] is not None and test_metrics["auprc"] <= float(np.mean(test_y)):
        warnings.append("Test PR-AUC does not exceed the test FoG prevalence baseline.")
    summary = f"""# S01 Non-FoG GRU-NBM + TCN-M 单被试实验

## 协议

- 输入：9通道、64 Hz、2秒窗口（128点），步长1秒。
- 逐窗口逐轴中心化：{'启用；NBM缩放信号与TCN残差均沿128点时间维减去各轴窗口均值。' if args.window_axis_center else '未启用。'}
- 外层：沿用历史冻结的{SUBJECT}连续块/记录划分，测试段为{SUBJECT_SPLITS[SUBJECT]['test_record']}；训练/验证间隔4秒。
- TCN训练残差：5折连续块交叉拟合，任何训练窗口均未被生成其残差的NBM看到。
- NBM：GRU 64 → 16维瓶颈 → 零输入GRU解码器64；仅clean non-FoG训练。
- 分类器：单分支因果TCN，通道32/64/64/128，膨胀率1/2/4/8。

## 训练

- Final NBM：最佳epoch {final_run['best_epoch']}/{final_run['epochs_completed']}，验证Huber {final_run['best_validation_huber']:.6f}。
- 分类器 {args.classifier}：最佳epoch {classifier_training['best_epoch']}/{classifier_training['epochs_completed']}，验证PR-AUC {classifier_training['best_validation_pr_auc']:.6f}。
- 验证集选择阈值：{threshold:.2f}。

## 测试结果

- Accuracy：{test_metrics['accuracy']:.6f}
- Balanced Accuracy：{test_metrics['balanced_accuracy']:.6f}
- FoG Precision：{test_metrics['precision']:.6f}
- FoG Recall：{test_metrics['sensitivity']:.6f}
- FoG F1：{test_metrics['f1']:.6f}
- Specificity：{test_metrics['specificity']:.6f}
- PR-AUC：{test_metrics['auprc']:.6f}
- ROC-AUC：{test_metrics['auroc']:.6f}
- 混淆矩阵：{test_metrics['confusion_matrix']}
- FoG事件检出：{event_summary['detected']}/{event_summary['events']}，事件召回率 {event_summary['event_recall']:.6f}

## 诊断

- OOF/验证 non-FoG 中位绝对残差比：{ratio:.6f}。
- 测试/验证 non-FoG 中位绝对残差比：{test_shift_ratio:.6f}。
- 测试 non-FoG 中 `|r|>3` 的比例：{diagnostics['test_final_nbm']['non_fog']['absolute_residual_gt_3_fraction']:.6f}。
- 警告：{'; '.join(warnings) if warnings else '未触发预设诊断警告。'}

## 公开的方案选择

- 原方案未规定2秒窗口二分类标签规则；本实验沿用既有S01端点规则：最后0.5秒FoG占比至少50%。
- 文档的“Causal/同长度”存在二选一表述；本实验采用左填充因果卷积。
- 为保留独立R02测试记录及FoG阳性的验证段，外层比例未强行切成精确70/15/15。
"""
    summary = f"""# {SUBJECT} NBM + {args.classifier} result

## Primary test metrics

- ACC: {test_metrics['acc']:.6f}
- Macro-Precision: {test_metrics['macro_precision']:.6f}
- Macro-Recall: {test_metrics['macro_recall']:.6f}
- Macro-F1: {test_metrics['macro_f1']:.6f}
- Confusion matrix: {test_metrics['confusion_matrix']}
- Validation-selected threshold: {threshold:.2f}

Both non-FoG and FoG are included as classes in every macro average. A zero
denominator contributes 0. Threshold selection remains validation-only.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_json(
        output_dir / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "warnings": warnings,
        },
    )
    print(
        f"COMPLETE threshold={threshold:.2f} acc={test_metrics['acc']:.6f} "
        f"macro_precision={test_metrics['macro_precision']:.6f} "
        f"macro_recall={test_metrics['macro_recall']:.6f} "
        f"macro_f1={test_metrics['macro_f1']:.6f} cm={test_metrics['confusion_matrix']} "
        f"events={event_summary['detected']}/{event_summary['events']} warnings={warnings}",
        flush=True,
    )


if __name__ == "__main__":
    main()
