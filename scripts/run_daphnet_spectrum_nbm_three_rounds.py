#!/usr/bin/env python
"""Run the three-round Daphnet Non-FoG spectrum NBM experiment.

Round 1 compares ordinary MSE with energy-balanced SmoothL1 on fixed N=32
subsets.  Round 2 trains a clean MLP spectrum NBM on record/time-block splits.
Round 3 ablates full-spectrum GRU, cropped-spectrum MLP, frequency Conv-AE,
and a shape-energy dual-head Conv-AE on the exact same splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for search_root in (REPO_ROOT, SCRIPTS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_daphnet_nbm_spectrum_small_overfit as small  # noqa: E402


EXPERIMENT = "daphnet_spectrum_nbm_three_rounds_v1"
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
ROUND1_SUBJECTS = ("S03", "S06", "S01")
ROUND1_LOSSES = ("A0_mse", "A1_balanced_smoothl1")
ROUND3_MODELS = ("B0_gru65", "B1_mlp24", "B2_conv24", "B3_shape_energy_conv24")
SEEDS = (20260803, 20260804, 20260805)
FS, WINDOW, STRIDE, CHANNELS = 64, 128, 64, 9
FULL_FREQ = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
CROP_MASK = (FULL_FREQ >= 0.5) & (FULL_FREQ <= 12.0)
CROP_FREQ = FULL_FREQ[CROP_MASK]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs" / "daphnet_spectrum_nbm_three_rounds_seed20260803",
    )
    parser.add_argument("--stage", choices=("round1", "round2", "round3", "all"), default="all")
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--round1-epochs", type=int, default=3000)
    parser.add_argument("--round2-epochs", type=int, default=300)
    parser.add_argument("--round3-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--screen-only", action="store_true", help="Skip extra seeds in rounds 2 and 3.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def csv_values(value: str, cast: Any = str) -> tuple[Any, ...]:
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Empty comma-separated argument")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    def json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2,
            allow_nan=False, default=json_default,
        )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class WindowTable:
    record_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    split: np.ndarray
    label: np.ndarray
    clean_nonfog: np.ndarray
    block_id: np.ndarray

    def indices(self, split: str, *, clean_only: bool = False) -> np.ndarray:
        mask = self.split == split
        if clean_only:
            mask &= self.clean_nonfog
        return np.flatnonzero(mask)


@dataclass(frozen=True)
class SubjectData:
    subject: str
    records: tuple[small.Record, ...]
    windows: WindowTable
    scaler: small.RunRobustScaler
    channel_names: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    metadata: dict[str, list[dict[str, Any]]]
    energy_floor_full: float
    energy_floor_crop: float
    mean_template_full: np.ndarray
    median_template_full: np.ndarray
    mean_template_crop: np.ndarray
    median_template_crop: np.ndarray


def record_protocol(subject: str, records: Sequence[small.Record]) -> dict[str, Any]:
    protocol = small.current.SUBJECT_SPLITS[subject]
    test_record = str(protocol["test_record"])
    validation_record = str(protocol["cut_record"])
    ignored = set(map(str, protocol["ignored"]))
    available = {record.record_id for record in records}
    required = {test_record, validation_record}
    if not required.issubset(available):
        raise ValueError(f"{subject}: missing protocol records {sorted(required - available)}")
    train_records = [
        record.record_id for record in records
        if record.record_id not in {test_record, validation_record} and record.record_id not in ignored
    ]
    temporal = not train_records
    return {
        "test_record": test_record,
        "validation_record": validation_record,
        "train_records": train_records,
        "ignored_records": sorted(ignored),
        "temporal_train_validation": temporal,
    }


def _append_windows(
    columns: dict[str, list[Any]],
    record_index: int,
    record: small.Record,
    segment_start: int,
    segment_end: int,
    split: str,
    block_id: str,
) -> None:
    for start in range(segment_start, segment_end - WINDOW + 1, STRIDE):
        end = start + WINDOW
        if not np.all(record.valid[start:end]):
            continue
        endpoint = record.y[end - 32 : end]
        label = int(float(np.mean(endpoint == 1)) >= 0.5)
        guard_start, guard_end = start - 2 * FS, end + FS
        clean = bool(
            guard_start >= segment_start
            and guard_end <= segment_end
            and not np.any(record.y[guard_start:guard_end] == 1)
        )
        columns["record_index"].append(record_index)
        columns["start"].append(start)
        columns["end"].append(end)
        columns["split"].append(split)
        columns["label"].append(label)
        columns["clean_nonfog"].append(clean)
        columns["block_id"].append(block_id)


def build_record_split_windows(
    subject: str, records: Sequence[small.Record]
) -> tuple[WindowTable, list[dict[str, Any]]]:
    protocol = record_protocol(subject, records)
    columns: dict[str, list[Any]] = defaultdict(list)
    split_rows: list[dict[str, Any]] = []
    train_records = set(protocol["train_records"])
    validation_record = protocol["validation_record"]
    test_record = protocol["test_record"]
    ignored = set(protocol["ignored_records"])
    for record_index, record in enumerate(records):
        if record.record_id in ignored:
            split_rows.append({
                "subject_id": subject, "record_id": record.record_id, "task_id": "not_available_in_release",
                "split": "ignored", "num_nonfog_windows": 0, "num_fog_windows": 0,
                "same_session_group": record.run_id, "notes": "excluded by frozen within-subject protocol",
            })
            continue
        segments: list[tuple[int, int, str, str]]
        if protocol["temporal_train_validation"] and record.record_id == validation_record:
            boundary = int(math.floor(0.8 * len(record.y) / STRIDE) * STRIDE)
            gap = 2 * FS
            segments = [
                (0, max(0, boundary - gap), "train", f"{record.record_id}:train_block"),
                (min(len(record.y), boundary + gap), len(record.y), "validation", f"{record.record_id}:validation_block"),
            ]
        elif record.record_id in train_records:
            segments = [(0, len(record.y), "train", record.record_id)]
        elif record.record_id == validation_record:
            segments = [(0, len(record.y), "validation", record.record_id)]
        elif record.record_id == test_record:
            segments = [(0, len(record.y), "test", record.record_id)]
        else:
            segments = []
        for start, end, split, block in segments:
            before = len(columns["start"])
            _append_windows(columns, record_index, record, start, end, split, block)
            rows = slice(before, len(columns["start"]))
            labels = np.asarray(columns["label"][rows], dtype=np.int8)
            split_rows.append({
                "subject_id": subject, "record_id": record.record_id,
                "task_id": "not_available_in_release", "split": split,
                "num_nonfog_windows": int(np.sum(labels == 0)),
                "num_fog_windows": int(np.sum(labels == 1)),
                "same_session_group": record.run_id,
                "notes": f"samples[{start}:{end}); contiguous time block; guard -2s/+1s",
            })
    windows = WindowTable(
        record_index=np.asarray(columns["record_index"], dtype=np.int32),
        start=np.asarray(columns["start"], dtype=np.int32),
        end=np.asarray(columns["end"], dtype=np.int32),
        split=np.asarray(columns["split"], dtype="U10"),
        label=np.asarray(columns["label"], dtype=np.int8),
        clean_nonfog=np.asarray(columns["clean_nonfog"], dtype=bool),
        block_id=np.asarray(columns["block_id"], dtype="U40"),
    )
    for split in ("train", "validation", "test"):
        if not len(windows.indices(split)):
            raise ValueError(f"{subject}: empty {split} split")
    if not len(windows.indices("train", clean_only=True)) or not len(windows.indices("validation", clean_only=True)):
        raise ValueError(f"{subject}: empty clean Non-FoG train/validation pool")
    return windows, split_rows


def fit_train_scaler(
    records: Sequence[small.Record], windows: WindowTable, indices: np.ndarray
) -> small.RunRobustScaler:
    masks: dict[int, np.ndarray] = {}
    for index in indices:
        record_index = int(windows.record_index[index])
        masks.setdefault(record_index, np.zeros(len(records[record_index].y), dtype=bool))
        masks[record_index][int(windows.start[index]) : int(windows.end[index])] = True
    values = np.concatenate([records[index].x[mask] for index, mask in masks.items()]).astype(np.float64)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = q75 - q25
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return small.RunRobustScaler(center.astype(np.float32), scale.astype(np.float32))


def extract_raw(
    records: Sequence[small.Record], windows: WindowTable, indices: np.ndarray
) -> np.ndarray:
    values = np.empty((len(indices), WINDOW, CHANNELS), dtype=np.float32)
    for row, index in enumerate(indices):
        record = records[int(windows.record_index[index])]
        values[row] = record.x[int(windows.start[index]) : int(windows.end[index])]
    return values


def power_and_log_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hann = np.hanning(WINDOW + 1)[:-1].astype(np.float32)
    transformed = np.fft.rfft(values * hann[None, :, None], axis=1)
    power = (np.square(np.abs(transformed)) / float(np.sum(np.square(hann)))).transpose(0, 2, 1)
    power = np.ascontiguousarray(power, dtype=np.float32)
    log_power = np.ascontiguousarray(np.log1p(power), dtype=np.float32)
    if power.shape[1:] != (CHANNELS, 65) or not np.isfinite(power).all():
        raise FloatingPointError(f"Invalid spectrum {power.shape}")
    return power, log_power


def window_metadata(
    records: Sequence[small.Record], windows: WindowTable, indices: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_index, index in enumerate(indices):
        record = records[int(windows.record_index[index])]
        rows.append({
            "array_index": output_index,
            "window_table_index": int(index),
            "subject_id": record.subject_id,
            "record_id": record.record_id,
            "run_id": record.run_id,
            "task_id": "not_available_in_release",
            "split": str(windows.split[index]),
            "block_id": str(windows.block_id[index]),
            "start_index": int(windows.start[index]),
            "end_index_exclusive": int(windows.end[index]),
            "start_time_sec": float(windows.start[index] / FS),
            "end_time_sec": float(windows.end[index] / FS),
            "label": int(windows.label[index]),
            "clean_nonfog": bool(windows.clean_nonfog[index]),
        })
    return rows


def prepare_subject(dataset: small.DaphnetDataset, subject: str) -> tuple[SubjectData, list[dict[str, Any]]]:
    records = tuple(record for record in dataset.records if record.subject_id == subject)
    windows, split_rows = build_record_split_windows(subject, records)
    train_indices = windows.indices("train", clean_only=True)
    validation_indices = windows.indices("validation", clean_only=True)
    test_indices = windows.indices("test")
    scaler = fit_train_scaler(records, windows, train_indices)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, list[dict[str, Any]]] = {}
    for name, indices in (
        ("train", train_indices), ("validation", validation_indices), ("test", test_indices)
    ):
        scaled = scaler.transform(extract_raw(records, windows, indices))
        power, log_power = power_and_log_spectrum(scaled)
        arrays[f"{name}_power"] = power
        arrays[f"{name}_log"] = log_power
        arrays[f"{name}_label"] = windows.label[indices].astype(np.int8)
        arrays[f"{name}_clean"] = windows.clean_nonfog[indices].astype(bool)
        metadata[name] = window_metadata(records, windows, indices)
    train_energy_full = np.sum(np.abs(arrays["train_log"]), axis=(1, 2))
    train_crop = arrays["train_log"][:, :, CROP_MASK]
    train_energy_crop = np.sum(np.abs(train_crop), axis=(1, 2))
    subject_data = SubjectData(
        subject=subject, records=records, windows=windows, scaler=scaler,
        channel_names=tuple(dataset.channel_names), arrays=arrays, metadata=metadata,
        energy_floor_full=float(np.quantile(train_energy_full, 0.10)),
        energy_floor_crop=float(np.quantile(train_energy_crop, 0.10)),
        mean_template_full=np.mean(arrays["train_log"], axis=0),
        median_template_full=np.median(arrays["train_log"], axis=0),
        mean_template_crop=np.mean(train_crop, axis=0),
        median_template_crop=np.median(train_crop, axis=0),
    )
    return subject_data, split_rows


class SpectrumMLP(nn.Module):
    def __init__(self, frequencies: int, latent: int = 32) -> None:
        super().__init__()
        width = CHANNELS * frequencies
        self.frequencies = frequencies
        self.net = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, latent), nn.GELU(),
            nn.Linear(latent, 128), nn.GELU(), nn.Linear(128, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(len(x), -1)).reshape(len(x), CHANNELS, self.frequencies)


class FrequencyConvAE(nn.Module):
    def __init__(self, latent: int = 32) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(9, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        self.to_latent = nn.Linear(64 * 6, latent)
        self.from_latent = nn.Linear(latent, 64 * 6)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 9, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_conv(x)
        if encoded.shape[1:] != (64, 6):
            raise RuntimeError(f"Conv encoder produced {encoded.shape}")
        return self.to_latent(encoded.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        output = self.decoder(self.from_latent(latent).reshape(-1, 64, 6))
        if output.shape[1:] != (9, 24):
            raise RuntimeError(f"Conv decoder produced {output.shape}; cropping is forbidden")
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ShapeEnergyConvAE(FrequencyConvAE):
    def __init__(self, latent: int = 32) -> None:
        super().__init__(latent=latent)
        self.energy_head = nn.Linear(latent, CHANNELS)

    def forward(self, shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(shape)
        reconstructed_shape = torch.softmax(self.decode(latent), dim=-1)
        reconstructed_energy = self.energy_head(latent)
        return reconstructed_shape, reconstructed_energy


def build_model(model_id: str) -> nn.Module:
    if model_id in {"round1_mlp65", "round2_mlp65"}:
        return SpectrumMLP(65)
    if model_id == "B0_gru65":
        return small.CurrentSpectrumNBM()
    if model_id == "B1_mlp24":
        return SpectrumMLP(24)
    if model_id == "B2_conv24":
        return FrequencyConvAE()
    if model_id == "B3_shape_energy_conv24":
        return ShapeEnergyConvAE()
    raise ValueError(model_id)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sample_metrics(
    actual: np.ndarray, predicted: np.ndarray, energy_floor: float, frequencies: np.ndarray
) -> dict[str, np.ndarray]:
    difference = actual.astype(np.float64) - predicted.astype(np.float64)
    absolute = np.abs(difference)
    target_energy = np.sum(np.abs(actual), axis=(1, 2)).astype(np.float64)
    predicted_energy = np.sum(np.abs(predicted), axis=(1, 2)).astype(np.float64)
    flat_actual = actual.reshape(len(actual), -1).astype(np.float64)
    flat_predicted = predicted.reshape(len(predicted), -1).astype(np.float64)
    denominator = np.linalg.norm(flat_actual, axis=1) * np.linalg.norm(flat_predicted, axis=1)
    locomotor = (frequencies >= 0.5) & (frequencies <= 3.0)
    freeze = (frequencies > 3.0) & (frequencies <= 8.0)
    return {
        "mae": np.mean(absolute, axis=(1, 2)),
        "mse": np.mean(np.square(difference), axis=(1, 2)),
        "nmae": np.sum(absolute, axis=(1, 2)) / (target_energy + EPS),
        "nmae_floor": np.sum(absolute, axis=(1, 2)) / np.maximum(target_energy, energy_floor),
        "cosine": np.divide(
            np.sum(flat_actual * flat_predicted, axis=1), denominator,
            out=np.zeros(len(actual), dtype=np.float64), where=denominator > EPS,
        ),
        "energy_error": np.abs(target_energy - predicted_energy) / (target_energy + EPS),
        "band_05_3_error": np.mean(absolute[:, :, locomotor], axis=(1, 2)),
        "band_3_8_error": np.mean(absolute[:, :, freeze], axis=(1, 2)),
        "target_energy": target_energy,
    }


def aggregate_metrics(
    actual: np.ndarray, predicted: np.ndarray, energy_floor: float, frequencies: np.ndarray
) -> dict[str, Any]:
    arrays = sample_metrics(actual, predicted, energy_floor, frequencies)
    result = {key: float(np.mean(value)) for key, value in arrays.items() if key != "target_energy"}
    result.update({
        "median_nmae_floor": float(np.median(arrays["nmae_floor"])),
        "p90_nmae_floor": float(np.quantile(arrays["nmae_floor"], 0.90)),
        "worst_nmae_floor": float(np.max(arrays["nmae_floor"])),
        "output_target_std_ratio": float(np.std(predicted) / max(float(np.std(actual)), EPS)),
        "all_finite": bool(np.isfinite(predicted).all()),
    })
    return result


def template_skill(actual: np.ndarray, predicted: np.ndarray, template: np.ndarray) -> float:
    model_mae = float(np.mean(np.abs(actual - predicted)))
    template_mae = float(np.mean(np.abs(actual - template[None])))
    return float(1.0 - model_mae / max(template_mae, EPS))


def energy_weights(target: np.ndarray) -> np.ndarray:
    energy = np.mean(target, axis=(1, 2)).astype(np.float64)
    median = float(np.median(energy))
    q10 = float(np.quantile(energy, 0.10))
    weights = np.clip(np.sqrt(median / np.maximum(energy, q10)), 0.5, 3.0)
    return weights.astype(np.float32)


def ordinary_loss(
    reconstruction: torch.Tensor, target: torch.Tensor, loss_name: str,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_name == "A0_mse":
        return nn.functional.mse_loss(reconstruction, target)
    if loss_name == "A1_balanced_smoothl1":
        if weights is None:
            raise ValueError("Balanced loss requires weights")
        per_sample = nn.functional.smooth_l1_loss(reconstruction, target, reduction="none").mean(dim=(1, 2))
        return torch.sum(weights * per_sample) / torch.sum(weights)
    raise ValueError(loss_name)


def shape_energy_targets(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cropped = np.asarray(power[:, :, CROP_MASK], dtype=np.float32)
    total = np.sum(cropped, axis=2)
    shape = cropped / np.maximum(total[:, :, None], 1e-12)
    energy = np.log(total + 1e-12)
    return np.ascontiguousarray(shape), np.ascontiguousarray(energy)


def shape_energy_to_log(shape: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    power = shape * torch.exp(energy.clamp(-20.0, 20.0)).unsqueeze(-1)
    return torch.log1p(power)


def shape_energy_loss(
    predicted_shape: torch.Tensor, predicted_energy: torch.Tensor,
    target_shape: torch.Tensor, target_energy: torch.Tensor,
) -> torch.Tensor:
    shape_loss = nn.functional.smooth_l1_loss(predicted_shape, target_shape)
    emd = torch.mean(torch.abs(torch.cumsum(predicted_shape, dim=-1) - torch.cumsum(target_shape, dim=-1)))
    energy_loss = nn.functional.smooth_l1_loss(predicted_energy, target_energy)
    return 0.6 * shape_loss + 0.2 * emd + 0.2 * energy_loss


@torch.no_grad()
def predict_ordinary(
    model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(np.ascontiguousarray(values[start : start + batch_size])).to(device)
        parts.append(model(tensor).float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


@torch.no_grad()
def predict_shape_energy(
    model: nn.Module, power: np.ndarray, batch_size: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    shapes, energies = shape_energy_targets(power)
    log_parts: list[np.ndarray] = []
    shape_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    for start in range(0, len(power), batch_size):
        shape = torch.from_numpy(shapes[start : start + batch_size]).to(device)
        predicted_shape, predicted_energy = model(shape)
        log_parts.append(shape_energy_to_log(predicted_shape, predicted_energy).float().cpu().numpy())
        shape_parts.append(predicted_shape.float().cpu().numpy())
        energy_parts.append(predicted_energy.float().cpu().numpy())
    return (
        np.ascontiguousarray(np.concatenate(log_parts), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(shape_parts), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(energy_parts), dtype=np.float32),
    )


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_round1_model(
    target: np.ndarray,
    loss_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    run_dir: Path,
) -> tuple[nn.Module, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    model = build_model("round1_mlp65").to(device)
    if any(isinstance(module, (nn.Dropout, nn.BatchNorm1d)) for module in model.modules()):
        raise AssertionError("Round 1 MLP must not contain dropout or batch norm")
    tensor = torch.from_numpy(np.ascontiguousarray(target)).to(device)
    weights = torch.from_numpy(energy_weights(target)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    initial = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    gradient_initial = 0.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(tensor)
        if reconstruction.shape != tensor.shape:
            raise AssertionError("Round 1 reconstruction shape mismatch")
        loss = ordinary_loss(reconstruction, tensor, loss_name, weights)
        loss.backward()
        gradient = math.sqrt(sum(
            float(torch.sum(parameter.grad.detach().square()))
            for parameter in model.parameters() if parameter.grad is not None
        ))
        if epoch == 1:
            gradient_initial = gradient
        if not math.isfinite(gradient) or gradient <= 0:
            raise FloatingPointError("Invalid Round 1 gradient")
        optimizer.step()
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "learning_rate": learning_rate})
        if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
            print(f"[R1 {loss_name}] epoch={epoch:04d}/{epochs} loss={float(loss):.7g}", flush=True)
    prediction = predict_ordinary(model, target, len(target), device)
    final = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    training = {
        "epochs": epochs, "final_loss": history[-1]["train_loss"],
        "gradient_norm_initial": gradient_initial,
        "parameter_delta_l2": float(torch.linalg.vector_norm(final - initial)),
        "parameter_count": parameter_count(model), "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_torch_save(run_dir / "checkpoint.pt", {
        "model_state": model.state_dict(), "loss_name": loss_name, "seed": seed, "training": training,
    })
    write_csv(run_dir / "history.csv", history)
    return model, prediction, history, training


def training_arrays(
    subject: SubjectData, model_id: str, split: str
) -> tuple[np.ndarray, np.ndarray | None]:
    if model_id in {"round2_mlp65", "B0_gru65"}:
        return subject.arrays[f"{split}_log"], None
    if model_id in {"B1_mlp24", "B2_conv24"}:
        return subject.arrays[f"{split}_log"][:, :, CROP_MASK], None
    if model_id == "B3_shape_energy_conv24":
        return shape_energy_targets(subject.arrays[f"{split}_power"])
    raise ValueError(model_id)


def reconstruct_split(
    model_id: str, model: nn.Module, subject: SubjectData, split: str,
    batch_size: int, device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if model_id in {"round2_mlp65", "B0_gru65"}:
        target = subject.arrays[f"{split}_log"]
        return predict_ordinary(model, target, batch_size, device), {}
    if model_id in {"B1_mlp24", "B2_conv24"}:
        target = subject.arrays[f"{split}_log"][:, :, CROP_MASK]
        return predict_ordinary(model, target, batch_size, device), {}
    if model_id == "B3_shape_energy_conv24":
        log_power, shape, energy = predict_shape_energy(
            model, subject.arrays[f"{split}_power"], batch_size, device
        )
        return log_power, {"predicted_shape": shape, "predicted_energy": energy}
    raise ValueError(model_id)


def target_log(subject: SubjectData, model_id: str, split: str) -> np.ndarray:
    values = subject.arrays[f"{split}_log"]
    return values if model_id in {"round2_mlp65", "B0_gru65"} else values[:, :, CROP_MASK]


def model_frequency(model_id: str) -> np.ndarray:
    return FULL_FREQ if model_id in {"round2_mlp65", "B0_gru65"} else CROP_FREQ


def model_energy_floor(subject: SubjectData, model_id: str) -> float:
    return subject.energy_floor_full if model_id in {"round2_mlp65", "B0_gru65"} else subject.energy_floor_crop


def train_early_stopped_model(
    subject: SubjectData,
    model_id: str,
    loss_name: str,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    model = build_model(model_id).to(device)
    if any(isinstance(module, nn.Dropout) and module.p > 0 for module in model.modules()):
        raise AssertionError("Dropout must be disabled")
    train_primary, train_secondary = training_arrays(subject, model_id, "train")
    validation_target = target_log(subject, model_id, "validation")
    train_weights = energy_weights(target_log(subject, model_id, "train"))
    if model_id == "B3_shape_energy_conv24":
        dataset = TensorDataset(torch.from_numpy(train_primary), torch.from_numpy(train_secondary))
    else:
        dataset = TensorDataset(torch.from_numpy(train_primary), torch.from_numpy(train_weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    initial = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    gradient_initial = 0.0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
            generator=generator, num_workers=num_workers, drop_last=False,
            pin_memory=device.type == "cuda",
        )
        model.train()
        total_loss, total_count = 0.0, 0
        maximum_gradient = 0.0
        for first_cpu, second_cpu in loader:
            first = first_cpu.to(device, non_blocking=True)
            second = second_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if model_id == "B3_shape_energy_conv24":
                predicted_shape, predicted_energy = model(first)
                loss = shape_energy_loss(predicted_shape, predicted_energy, first, second)
            else:
                reconstruction = model(first)
                loss = ordinary_loss(reconstruction, first, loss_name, second)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(gradient) or gradient <= 0:
                raise FloatingPointError(f"{subject.subject}/{model_id}: invalid gradient")
            maximum_gradient = max(maximum_gradient, gradient)
            if epoch == 1 and total_count == 0:
                gradient_initial = gradient
            optimizer.step()
            total_loss += float(loss.detach()) * len(first)
            total_count += len(first)
        validation_prediction, _ = reconstruct_split(
            model_id, model, subject, "validation", batch_size, device
        )
        validation = aggregate_metrics(
            validation_target, validation_prediction, model_energy_floor(subject, model_id),
            model_frequency(model_id),
        )
        val_metric = float(validation["nmae_floor"])
        scheduler.step(val_metric)
        improved = val_metric < best_metric - 1e-5
        if improved:
            best_metric, best_epoch, best_state, bad_epochs = val_metric, epoch, clone_state(model), 0
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch, "train_loss": total_loss / total_count,
            "validation_nmae_floor": val_metric,
            "validation_cosine": validation["cosine"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "max_gradient_norm_before_clip": maximum_gradient,
            "improved": improved,
        })
        if epoch == 1 or epoch % 25 == 0 or bad_epochs >= patience:
            print(
                f"[{subject.subject} {model_id} seed={seed}] epoch={epoch:03d}/{max_epochs} "
                f"train={total_loss/total_count:.6g} val_floor={val_metric:.4f} bad={bad_epochs}",
                flush=True,
            )
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("No best model state")
    model.load_state_dict(best_state)
    final = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    training = {
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_validation_nmae_floor": best_metric,
        "final_learning_rate": history[-1]["learning_rate"],
        "gradient_norm_initial": gradient_initial,
        "parameter_delta_l2": float(torch.linalg.vector_norm(final - initial)),
        "parameter_count": parameter_count(model), "elapsed_seconds": time.perf_counter() - started,
        "stopped_early": len(history) < max_epochs,
    }
    atomic_torch_save(run_dir / "checkpoint.pt", {
        "model_id": model_id, "model_state": best_state, "seed": seed,
        "loss_name": loss_name, "training": training,
    })
    write_csv(run_dir / "history.csv", history)
    return model, history, training


def load_model_checkpoint(model_id: str, path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(model_id).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload


def inference_time_ms(
    model_id: str, model: nn.Module, subject: SubjectData, batch_size: int, device: torch.device
) -> float:
    model.eval()
    indices = np.flatnonzero(subject.arrays["test_clean"])
    if not len(indices):
        return float("nan")
    if model_id == "B3_shape_energy_conv24":
        values, _ = shape_energy_targets(subject.arrays["test_power"][indices])
    else:
        values = target_log(subject, model_id, "test")[indices]
    tensor = torch.from_numpy(np.ascontiguousarray(values[:batch_size])).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(20):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / (20 * len(tensor))


def residual_scores(
    actual: np.ndarray, predicted: np.ndarray, frequencies: np.ndarray
) -> dict[str, np.ndarray]:
    absolute = np.abs(actual - predicted)
    locomotor = (frequencies >= 0.5) & (frequencies <= 3.0)
    freeze = (frequencies > 3.0) & (frequencies <= 8.0)
    return {
        "mean": np.mean(absolute, axis=(1, 2)),
        "band_05_3": np.mean(absolute[:, :, locomotor], axis=(1, 2)),
        "band_3_8": np.mean(absolute[:, :, freeze], axis=(1, 2)),
    }


def effect_size(values0: np.ndarray, values1: np.ndarray) -> float:
    values0 = np.asarray(values0, dtype=np.float64)
    values1 = np.asarray(values1, dtype=np.float64)
    if not len(values0) or not len(values1):
        raise ValueError("Effect size requires two non-empty groups")
    sum_squares = float(np.sum(np.square(values0 - np.mean(values0))))
    sum_squares += float(np.sum(np.square(values1 - np.mean(values1))))
    degrees_of_freedom = len(values0) + len(values1) - 2
    if degrees_of_freedom > 0 and sum_squares > 0:
        pooled = math.sqrt(sum_squares / degrees_of_freedom)
    else:
        pooled = math.sqrt(max(float(np.var(np.concatenate((values0, values1)))), EPS))
    return float((np.mean(values1) - np.mean(values0)) / pooled)


def run_status(metrics: dict[str, Any]) -> str:
    checks = (
        metrics["test_nonfog_nmae_floor"] < 0.40,
        metrics["test_nonfog_cossim"] > 0.80,
        metrics["test_median_template_skill"] > 0.0,
        0.9 <= metrics["output_target_std_ratio"] <= 1.1,
    )
    if all(checks):
        return "Pass"
    close = (
        metrics["test_nonfog_nmae_floor"] < 0.50,
        metrics["test_nonfog_cossim"] > 0.70,
        metrics["test_median_template_skill"] > -0.10,
        0.75 <= metrics["output_target_std_ratio"] <= 1.25,
    )
    return "Borderline" if all(close) else "Fail"


def evaluate_trained_run(
    subject: SubjectData,
    model_id: str,
    model: nn.Module,
    seed: int,
    loss_name: str,
    training: dict[str, Any],
    batch_size: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frequencies = model_frequency(model_id)
    floor = model_energy_floor(subject, model_id)
    predictions: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        prediction, extra = reconstruct_split(model_id, model, subject, split, batch_size, device)
        predictions[split] = prediction
        for key, value in extra.items():
            extras[f"{split}_{key}"] = value
        if split != "test":
            actual = target_log(subject, model_id, split)
            split_metrics[split] = aggregate_metrics(actual, prediction, floor, frequencies)
    test_actual_all = target_log(subject, model_id, "test")
    clean_mask = subject.arrays["test_clean"]
    fog_mask = subject.arrays["test_label"] == 1
    if not np.any(clean_mask) or not np.any(fog_mask):
        raise ValueError(f"{subject.subject}: test needs clean Non-FoG and FoG windows")
    test_actual = test_actual_all[clean_mask]
    test_prediction = predictions["test"][clean_mask]
    test_metrics = aggregate_metrics(test_actual, test_prediction, floor, frequencies)
    if model_id in {"round2_mlp65", "B0_gru65"}:
        mean_template, median_template = subject.mean_template_full, subject.median_template_full
    else:
        mean_template, median_template = subject.mean_template_crop, subject.median_template_crop
    scores = residual_scores(test_actual_all, predictions["test"], frequencies)
    eligible = clean_mask | fog_mask
    labels = fog_mask[eligible].astype(np.int8)
    residual = scores["mean"][eligible]
    nonfog_residual, fog_residual = scores["mean"][clean_mask], scores["mean"][fog_mask]
    channel_error = np.mean(np.abs(test_actual - test_prediction), axis=0)
    frequency_error = np.mean(np.abs(test_actual - test_prediction), axis=(0, 1))
    split_error_arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        split_actual = target_log(subject, model_id, split)
        split_absolute = np.abs(split_actual - predictions[split])
        split_error_arrays[f"{split}_channel_frequency_error"] = np.mean(split_absolute, axis=0)
        split_error_arrays[f"{split}_frequency_error"] = np.mean(split_absolute, axis=(0, 1))
    metrics = {
        "subject_id": subject.subject,
        "model_id": model_id,
        "seed": seed,
        "loss_name": loss_name,
        "frequency_range": "0-32Hz" if len(frequencies) == 65 else "0.5-12Hz",
        "num_frequency_bins": len(frequencies),
        "representation": "shape-energy power" if model_id.startswith("B3") else "log-power",
        **training,
        "train_loss": split_metrics["train"]["mse"],
        "train_mse": split_metrics["train"]["mse"],
        "train_mae": split_metrics["train"]["mae"],
        "train_nmae": split_metrics["train"]["nmae"],
        "train_nmae_floor": split_metrics["train"]["nmae_floor"],
        "train_cossim": split_metrics["train"]["cosine"],
        "train_energy_error": split_metrics["train"]["energy_error"],
        "val_mse": split_metrics["validation"]["mse"],
        "val_mae": split_metrics["validation"]["mae"],
        "val_nmae": split_metrics["validation"]["nmae"],
        "val_nmae_floor": split_metrics["validation"]["nmae_floor"],
        "val_cossim": split_metrics["validation"]["cosine"],
        "val_energy_error": split_metrics["validation"]["energy_error"],
        "test_nonfog_mae": test_metrics["mae"],
        "test_nonfog_nmae": test_metrics["nmae"],
        "test_nonfog_nmae_floor": test_metrics["nmae_floor"],
        "test_nonfog_cossim": test_metrics["cosine"],
        "test_energy_error": test_metrics["energy_error"],
        "band_05_3_error": test_metrics["band_05_3_error"],
        "band_3_8_error": test_metrics["band_3_8_error"],
        "mean_template_skill": template_skill(test_actual, test_prediction, mean_template),
        "test_median_template_skill": template_skill(test_actual, test_prediction, median_template),
        "fog_auc": float(roc_auc_score(labels, residual)),
        "fog_prauc": float(average_precision_score(labels, residual)),
        "fog_residual_effect_size": effect_size(nonfog_residual, fog_residual),
        "output_target_std_ratio": test_metrics["output_target_std_ratio"],
        "inference_time_ms": inference_time_ms(model_id, model, subject, batch_size, device),
        "test_nonfog_windows": int(np.sum(clean_mask)),
        "test_fog_windows": int(np.sum(fog_mask)),
    }
    metrics["status"] = run_status(metrics)
    payload = {
        "test_target": test_actual_all,
        "test_reconstruction": predictions["test"],
        "test_label": subject.arrays["test_label"],
        "test_clean_nonfog": clean_mask.astype(np.int8),
        "test_residual_mean": scores["mean"],
        "test_residual_band_05_3": scores["band_05_3"],
        "test_residual_band_3_8": scores["band_3_8"],
        "channel_frequency_error": channel_error,
        "frequency_error": frequency_error,
        "train_target": target_log(subject, model_id, "train"),
        "train_reconstruction": predictions["train"],
        "validation_target": target_log(subject, model_id, "validation"),
        "validation_reconstruction": predictions["validation"],
        **split_error_arrays,
        **extras,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    atomic_npz(run_dir / "predictions.npz", **payload)
    return metrics, payload


def plot_loss_curve(
    history: Sequence[dict[str, Any]], path: Path, *, validation: bool
) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    if validation:
        ax.plot(epochs, [row["validation_nmae_floor"] for row in history], label="validation NMAE floor")
        best = int(np.argmin([row["validation_nmae_floor"] for row in history]))
        ax.axvline(epochs[best], linestyle="--", color="tab:green", label=f"best epoch {epochs[best]}")
        secondary = ax.twinx()
        secondary.step(epochs, [row["learning_rate"] for row in history], color="0.55", alpha=0.5, label="LR")
        secondary.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / metric (log scale)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overlay(
    actual: np.ndarray, predicted: np.ndarray, index: int, frequencies: np.ndarray,
    channel_names: Sequence[str], path: Path, title: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for channel, ax in enumerate(axes.flat):
        ax.plot(frequencies, actual[index, channel], label="target", linewidth=1.2)
        ax.plot(frequencies, predicted[index, channel], "--", label="reconstruction", linewidth=1.1)
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(
    actual: np.ndarray, predicted: np.ndarray, index: int, frequencies: np.ndarray,
    channel_names: Sequence[str], path: Path,
) -> None:
    vmax = float(max(np.max(actual[index]), np.max(predicted[index]), EPS))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    image = axes[0].imshow(actual[index], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[1].imshow(predicted[index], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    error_image = axes[2].imshow(np.abs(actual[index] - predicted[index]), aspect="auto", origin="lower", cmap="magma")
    for ax, title in zip(axes, ("Observed", "Reconstruction", "Absolute residual")):
        ax.set_title(title)
        ticks = np.linspace(0, len(frequencies) - 1, 6, dtype=int)
        ax.set_xticks(ticks, [f"{frequencies[tick]:.1f}" for tick in ticks])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_yticks(range(CHANNELS), channel_names, fontsize=7)
    fig.colorbar(image, ax=axes[:2], shrink=0.82)
    fig.colorbar(error_image, ax=axes[2], shrink=0.82)
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.13, top=0.88, wspace=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def best_median_worst(actual: np.ndarray, predicted: np.ndarray, floor: float, frequencies: np.ndarray) -> tuple[int, int, int]:
    values = sample_metrics(actual, predicted, floor, frequencies)["nmae_floor"]
    order = np.argsort(values)
    return int(order[0]), int(order[len(order) // 2]), int(order[-1])


def plot_channel_error(actual: np.ndarray, predicted: np.ndarray, channel_names: Sequence[str], path: Path) -> None:
    values = np.sum(np.abs(actual - predicted), axis=(0, 2)) / (np.sum(np.abs(actual), axis=(0, 2)) + EPS)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(range(CHANNELS), values)
    ax.set_xticks(range(CHANNELS), channel_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Channel NMAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual_timeline(
    payloads: Sequence[tuple[str, dict[str, np.ndarray]]], metadata: Sequence[dict[str, Any]], path: Path
) -> None:
    fig, axes = plt.subplots(len(payloads), 1, figsize=(14, 3.2 * len(payloads)), squeeze=False, sharex=True)
    times = np.asarray([row["start_time_sec"] for row in metadata])
    for ax, (label, payload) in zip(axes[:, 0], payloads):
        fog = payload["test_label"] == 1
        ax.fill_between(times, 0, 1, where=fog, transform=ax.get_xaxis_transform(), color="tab:red", alpha=0.12, label="FoG")
        ax.plot(times, payload["test_residual_mean"], label="mean residual", linewidth=1)
        ax.plot(times, payload["test_residual_band_05_3"], label="0.5-3 Hz", linewidth=0.9)
        ax.plot(times, payload["test_residual_band_3_8"], label="3-8 Hz", linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=4)
    axes[-1, 0].set_xlabel("Test record time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def nonoverlap_train_positions(subject: SubjectData) -> np.ndarray:
    grouped: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(subject.metadata["train"]):
        grouped[str(row["record_id"])].append(position)
    selected: list[int] = []
    for record_id in sorted(grouped):
        ordered = sorted(grouped[record_id], key=lambda i: subject.metadata["train"][i]["start_index"])
        last_end = -1
        for position in ordered:
            start = int(subject.metadata["train"][position]["start_index"])
            end = int(subject.metadata["train"][position]["end_index_exclusive"])
            if start >= last_end:
                selected.append(position)
                last_end = end
    return np.asarray(selected, dtype=np.int64)


def evenly_select(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) < count:
        raise ValueError(f"Need {count} values, found {len(values)}")
    positions = np.rint(np.linspace(0, len(values) - 1, count)).astype(int)
    if len(np.unique(positions)) != count:
        raise AssertionError("Even selection duplicated a position")
    return values[positions]


def round1_subsets(subject: SubjectData) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    eligible = nonoverlap_train_positions(subject)
    by_record: dict[str, list[int]] = defaultdict(list)
    for position in eligible:
        by_record[str(subject.metadata["train"][int(position)]["record_id"])].append(int(position))
    largest_record = max(by_record, key=lambda record_id: len(by_record[record_id]))
    if len(by_record[largest_record]) >= 96:
        # The protocol asks that each fixed subset stay within one recording/task
        # whenever possible.  Using one common recording also keeps the three
        # loss-comparison subsets on exactly the same acquisition domain.
        eligible = np.asarray(by_record[largest_record], dtype=np.int64)
    energy = np.sum(subject.arrays["train_log"][eligible], axis=(1, 2))
    quartiles = np.array_split(eligible[np.argsort(energy)], 4)
    if any(len(quartile) < 24 for quartile in quartiles):
        raise ValueError(f"{subject.subject}: insufficient windows for three disjoint stratified subsets")
    energy_lookup = {
        int(position): float(np.sum(subject.arrays["train_log"][position])) for position in eligible
    }
    subsets: dict[int, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    for subset_id in (1, 2, 3):
        chosen_parts: list[np.ndarray] = []
        for quartile_index, quartile in enumerate(quartiles, start=1):
            interleaved = quartile[subset_id - 1 :: 3]
            chosen = evenly_select(interleaved, 8)
            chosen_parts.append(chosen)
            for position in chosen:
                row = subject.metadata["train"][int(position)]
                manifest_rows.append({
                    "subject_id": subject.subject, "subset_id": subset_id,
                    "sample_id": row["array_index"], "window_id": f"{row['record_id']}_{row['start_index']:06d}_{row['end_index_exclusive']:06d}",
                    "record_id": row["record_id"], "task_id": row["task_id"],
                    "start_time": row["start_time_sec"], "end_time": row["end_time_sec"],
                    "energy": energy_lookup[int(position)], "energy_quartile": f"Q{quartile_index}",
                })
        selected = np.concatenate(chosen_parts).astype(np.int64)
        if len(selected) != 32 or len(np.unique(selected)) != 32:
            raise AssertionError("Invalid Round 1 subset")
        subsets[subset_id] = selected
    if len({int(value) for subset in subsets.values() for value in subset}) != 96:
        raise AssertionError("Round 1 subsets must be disjoint")
    return subsets, manifest_rows


def quartile_summary(
    metrics: dict[str, np.ndarray], quartiles: Sequence[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    quartile_array = np.asarray(quartiles)
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        values = metrics["nmae_floor"][quartile_array == quartile]
        result[f"{quartile.lower()}_mean_nmae_floor"] = float(np.mean(values))
        result[f"{quartile.lower()}_median_nmae_floor"] = float(np.median(values))
        result[f"{quartile.lower()}_p90_nmae_floor"] = float(np.quantile(values, 0.90))
        result[f"{quartile.lower()}_worst_nmae_floor"] = float(np.max(values))
        result[f"{quartile.lower()}_mae"] = float(np.mean(metrics["mae"][quartile_array == quartile]))
    return result


def plot_round1_comparison(
    subject: str, subset_id: int, target: np.ndarray,
    predictions: dict[str, np.ndarray], quartiles: Sequence[str], floor: float,
    figure_dir: Path,
) -> None:
    arrays = {name: sample_metrics(target, value, floor, FULL_FREQ) for name, value in predictions.items()}
    positions = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, name in zip((-0.17, 0.17), ROUND1_LOSSES):
        values = [arrays[name]["nmae_floor"][np.asarray(quartiles) == f"Q{q}"] for q in range(1, 5)]
        boxes = ax.boxplot(values, positions=positions + offset, widths=0.28, patch_artist=True)
        for box in boxes["boxes"]:
            box.set_alpha(0.55)
        boxes["boxes"][0].set_label(name)
    ax.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylabel("NMAE floor")
    ax.set_title(f"{subject} subset {subset_id}: energy-quartile error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"R1_{subject}_subset{subset_id}_energy_quartile_nmae.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    energy = arrays[ROUND1_LOSSES[0]]["target_energy"]
    for name, marker in zip(ROUND1_LOSSES, ("o", "x")):
        ax.scatter(np.log(energy + EPS), arrays[name]["nmae_floor"], marker=marker, alpha=0.75, label=name)
    ax.set_xlabel("log(total spectral energy)")
    ax.set_ylabel("NMAE floor")
    ax.set_title(f"{subject} subset {subset_id}: error vs energy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"R1_{subject}_subset{subset_id}_error_vs_energy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.36
    for offset, name in zip((-width / 2, width / 2), ROUND1_LOSSES):
        channel = np.sum(np.abs(target - predictions[name]), axis=(0, 2)) / (np.sum(np.abs(target), axis=(0, 2)) + EPS)
        ax.bar(np.arange(CHANNELS) + offset, channel, width, label=name)
    ax.set_xticks(range(CHANNELS), [f"C{i}" for i in range(CHANNELS)])
    ax.set_ylabel("Channel NMAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"R1_{subject}_subset{subset_id}_channel_nmae.png", dpi=180)
    plt.close(fig)


def run_round1(
    args: argparse.Namespace, subjects: dict[str, SubjectData], device: torch.device
) -> dict[str, Any]:
    root = args.output_dir / "round1"
    figure_dir = args.output_dir / "figures" / "round1"
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_manifest: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for subject_id in ROUND1_SUBJECTS:
        subject = subjects[subject_id]
        subsets, manifest_rows = round1_subsets(subject)
        all_manifest.extend(manifest_rows)
        by_subset_manifest = defaultdict(list)
        for row in manifest_rows:
            by_subset_manifest[int(row["subset_id"])].append(row)
        for subset_id, positions in subsets.items():
            target = subject.arrays["train_log"][positions]
            floor = float(np.quantile(np.sum(np.abs(target), axis=(1, 2)), 0.10))
            predictions: dict[str, np.ndarray] = {}
            run_rows: dict[str, dict[str, Any]] = {}
            for loss_name in ROUND1_LOSSES:
                run_dir = root / subject_id / f"subset{subset_id}" / loss_name
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists() and not args.overwrite:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    with np.load(run_dir / "predictions.npz", allow_pickle=False) as payload:
                        prediction = np.asarray(payload["reconstruction"])
                    with (run_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
                        history = [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    _, prediction, history, training = train_round1_model(
                        target, loss_name, SEEDS[0], args.round1_epochs,
                        args.learning_rate, device, run_dir,
                    )
                    per_sample = sample_metrics(target, prediction, floor, FULL_FREQ)
                    quartiles = [row["energy_quartile"] for row in by_subset_manifest[subset_id]]
                    metrics = {
                        "subject_id": subject_id, "subset_id": subset_id,
                        "loss_name": loss_name, **training,
                        **quartile_summary(per_sample, quartiles),
                        "output_target_std_ratio": float(np.std(prediction) / max(float(np.std(target)), EPS)),
                        "mean_template_skill": template_skill(target, prediction, np.mean(target, axis=0)),
                    }
                    atomic_json(metrics_path, metrics)
                    atomic_npz(run_dir / "predictions.npz", target=target, reconstruction=prediction)
                predictions[loss_name] = prediction
                run_rows[loss_name] = metrics
                results.append(metrics)
                per_sample = sample_metrics(target, prediction, floor, FULL_FREQ)
                for sample_index, manifest_row in enumerate(by_subset_manifest[subset_id]):
                    sample_results.append({
                        "subject_id": subject_id,
                        "subset_id": subset_id,
                        "loss_name": loss_name,
                        "sample_id": manifest_row["sample_id"],
                        "window_id": manifest_row["window_id"],
                        "energy_quartile": manifest_row["energy_quartile"],
                        **{key: float(value[sample_index]) for key, value in per_sample.items()},
                    })
                plot_loss_curve(history, figure_dir / f"R1_{subject_id}_subset{subset_id}_{loss_name.split('_')[0]}_loss.png", validation=False)
            quartiles = [row["energy_quartile"] for row in by_subset_manifest[subset_id]]
            plot_round1_comparison(subject_id, subset_id, target, predictions, quartiles, floor, figure_dir)
            balanced = predictions["A1_balanced_smoothl1"]
            best, median, worst = best_median_worst(target, balanced, floor, FULL_FREQ)
            for label, index in (("best", best), ("median", median), ("worst", worst)):
                plot_overlay(
                    target, balanced, index, FULL_FREQ, subject.channel_names,
                    figure_dir / f"R1_{subject_id}_subset{subset_id}_A1_overlay_{label}.png",
                    f"Round 1 {subject_id} subset {subset_id} A1 {label}",
                )
            a0, a1 = run_rows["A0_mse"], run_rows["A1_balanced_smoothl1"]
            q1_improvement = 100.0 * (a0["q1_p90_nmae_floor"] - a1["q1_p90_nmae_floor"]) / max(a0["q1_p90_nmae_floor"], EPS)
            q3_change = 100.0 * (a1["q3_p90_nmae_floor"] - a0["q3_p90_nmae_floor"]) / max(a0["q3_p90_nmae_floor"], EPS)
            q4_change = 100.0 * (a1["q4_p90_nmae_floor"] - a0["q4_p90_nmae_floor"]) / max(a0["q4_p90_nmae_floor"], EPS)
            passed = bool(
                q1_improvement >= 20.0 and q3_change <= 10.0 and q4_change <= 10.0
                and 0.9 <= a1["output_target_std_ratio"] <= 1.1
                and a1["mean_template_skill"] > 0.0
            )
            comparisons.append({
                "subject_id": subject_id, "subset_id": subset_id,
                "q1_p90_improvement_pct": q1_improvement,
                "q3_p90_change_pct": q3_change, "q4_p90_change_pct": q4_change,
                "a1_output_target_std_ratio": a1["output_target_std_ratio"],
                "status": "Pass" if passed else "Fail",
            })
    write_csv(args.output_dir / "manifests" / "round1_subsets.csv", all_manifest)
    write_csv(args.output_dir / "metrics" / "round1_comparisons.csv", comparisons)
    comparison_status = {
        (row["subject_id"], int(row["subset_id"])): row["status"] for row in comparisons
    }
    for row in results:
        row["status"] = comparison_status[(row["subject_id"], int(row["subset_id"]))]
    write_csv(args.output_dir / "metrics" / "round1_metrics.csv", results)
    write_csv(args.output_dir / "metrics" / "round1_sample_metrics.csv", sample_results)
    subject_pass = {
        subject: sum(row["status"] == "Pass" for row in comparisons if row["subject_id"] == subject) >= 2
        for subject in ROUND1_SUBJECTS
    }
    gate = sum(subject_pass.values()) >= 2
    selected_loss = "A1_balanced_smoothl1" if gate else "A0_mse"
    decision = {
        "round1_gate": "Pass" if gate else "Fail",
        "selected_loss": selected_loss,
        "subject_pass": subject_pass,
        "passing_subjects": sum(subject_pass.values()),
        "passing_cells": sum(row["status"] == "Pass" for row in comparisons),
        "total_cells": len(comparisons),
    }
    atomic_json(root / "decision.json", decision)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"{row['subject_id']}-S{row['subset_id']}" for row in comparisons]
    values = [row["q1_p90_improvement_pct"] for row in comparisons]
    colors = ["tab:green" if row["status"] == "Pass" else "tab:red" for row in comparisons]
    ax.bar(labels, values, color=colors)
    ax.axhline(20.0, linestyle="--", color="black", label="20% gate")
    ax.set_ylabel("A1 vs A0 Q1 P90 improvement (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "R1_all_subjects_Q1_P90_improvement.png", dpi=180)
    plt.close(fig)
    report_lines = [
        "# Round 1: low-energy windows and balanced loss", "",
        f"- Gate: {decision['round1_gate']}", f"- Selected Round 2 loss: `{selected_loss}`",
        f"- Passing subjects: {decision['passing_subjects']}/3",
        f"- Passing subject/subset cells: {decision['passing_cells']}/{decision['total_cells']}", "",
        "| Subject | Subset | Q1 P90 improvement | Q3 change | Q4 change | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    report_lines += [
        f"| {row['subject_id']} | {row['subset_id']} | {row['q1_p90_improvement_pct']:.2f}% | "
        f"{row['q3_p90_change_pct']:.2f}% | {row['q4_p90_change_pct']:.2f}% | {row['status']} |"
        for row in comparisons
    ]
    report_path = args.output_dir / "reports" / "round1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"ROUND1 {decision['round1_gate']} selected_loss={selected_loss}", flush=True)
    return decision


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key == "epoch":
                    converted[key] = int(value)
                elif key == "improved":
                    converted[key] = str(value).lower() == "true"
                else:
                    converted[key] = float(value)
            rows.append(converted)
        return rows


def load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def run_record_model(
    args: argparse.Namespace,
    subject: SubjectData,
    model_id: str,
    loss_name: str,
    seed: int,
    max_epochs: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload = load_payload(run_dir / "predictions.npz")
        history = load_history(run_dir / "history.csv")
        return metrics, payload, history
    run_dir.mkdir(parents=True, exist_ok=True)
    model, history, training = train_early_stopped_model(
        subject, model_id, loss_name, seed, max_epochs, args.patience,
        args.batch_size, args.learning_rate, args.num_workers, device, run_dir,
    )
    metrics, payload = evaluate_trained_run(
        subject, model_id, model, seed, loss_name, training,
        args.batch_size, device, run_dir,
    )
    print(
        f"RESULT {subject.subject} {model_id} seed={seed} "
        f"test_floor={metrics['test_nonfog_nmae_floor']:.4f} "
        f"cos={metrics['test_nonfog_cossim']:.4f} skill={metrics['test_median_template_skill']:.3f}",
        flush=True,
    )
    return metrics, payload, history


def round2_subject_figures(
    subject: SubjectData,
    metrics: dict[str, Any],
    payload: dict[str, np.ndarray],
    history: Sequence[dict[str, Any]],
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    subject_id = subject.subject
    plot_loss_curve(history, figure_dir / f"R2_{subject_id}_train_val_loss.png", validation=True)
    clean = payload["test_clean_nonfog"].astype(bool)
    actual = payload["test_target"][clean]
    predicted = payload["test_reconstruction"][clean]
    best, median, worst = best_median_worst(actual, predicted, subject.energy_floor_full, FULL_FREQ)
    for label, index in (("best", best), ("median", median), ("worst", worst)):
        plot_overlay(
            actual, predicted, index, FULL_FREQ, subject.channel_names,
            figure_dir / f"R2_{subject_id}_test_nonfog_overlay_{label}.png",
            f"Round 2 {subject_id} test Non-FoG {label}",
        )
    plot_heatmap(
        actual, predicted, worst, FULL_FREQ, subject.channel_names,
        figure_dir / f"R2_{subject_id}_test_nonfog_heatmap.png",
    )
    split_labels, split_values = [], []
    for split in ("train", "validation"):
        split_labels.append(split)
        split_values.append(float(np.mean(np.abs(payload[f"{split}_target"] - payload[f"{split}_reconstruction"]))))
    split_labels.append("test_nonfog")
    split_values.append(float(np.mean(np.abs(actual - predicted))))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(split_labels, split_values)
    ax.set_ylabel("Mean absolute log-power error")
    ax.set_title(f"{subject_id}: split/record-wise Non-FoG error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / f"R2_{subject_id}_recordwise_nonfog_error.png", dpi=180)
    plt.close(fig)
    plot_residual_timeline([("MLP-AE", payload)], subject.metadata["test"], figure_dir / f"R2_{subject_id}_test_residual_timeline.png")
    nonfog = payload["test_residual_mean"][clean]
    fog = payload["test_residual_mean"][payload["test_label"] == 1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(nonfog, bins=40, density=True, alpha=0.55, label="clean Non-FoG")
    ax.hist(fog, bins=40, density=True, alpha=0.55, label="FoG")
    ax.set_xlabel("Mean absolute residual")
    ax.set_ylabel("Density")
    ax.set_title(f"{subject_id}: residual distributions; AUROC={metrics['fog_auc']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"R2_{subject_id}_nonfog_vs_fog_residual_distribution.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(payload["channel_frequency_error"], aspect="auto", origin="lower", cmap="magma")
    ax.set_yticks(range(CHANNELS), subject.channel_names, fontsize=7)
    ticks = np.arange(0, 65, 8)
    ax.set_xticks(ticks, [f"{FULL_FREQ[tick]:.0f}" for tick in ticks])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"{subject_id}: test Non-FoG channel-frequency MAE")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / f"R2_{subject_id}_channel_frequency_error.png", dpi=180)
    plt.close(fig)


def plot_round2_summary(rows: Sequence[dict[str, Any]], subjects: Sequence[str], figure_dir: Path) -> None:
    aggregate: dict[str, dict[str, tuple[float, float]]] = {}
    metrics = {
        "nonfog_nmae": "test_nonfog_nmae_floor",
        "nonfog_cossim": "test_nonfog_cossim",
        "skill": "test_median_template_skill",
        "best_epoch": "best_epoch",
    }
    for subject in subjects:
        selected = [row for row in rows if row["subject_id"] == subject]
        aggregate[subject] = {
            label: (float(np.mean([row[key] for row in selected])), float(np.std([row[key] for row in selected])))
            for label, key in metrics.items()
        }
    figure_dir.mkdir(parents=True, exist_ok=True)
    for label in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        means = [aggregate[subject][label][0] for subject in subjects]
        stds = [aggregate[subject][label][1] for subject in subjects]
        ax.bar(subjects, means, yerr=stds, capsize=3)
        if label == "nonfog_nmae":
            ax.axhline(0.40, linestyle="--", color="tab:red", label="engineering target")
        elif label == "nonfog_cossim":
            ax.axhline(0.80, linestyle="--", color="tab:red", label="engineering target")
        elif label == "skill":
            ax.axhline(0.0, linestyle="--", color="black")
        ax.set_ylabel(label.replace("_", " "))
        ax.set_title(f"Round 2 {label} (mean ± SD across seeds)")
        ax.grid(axis="y", alpha=0.25)
        if label in {"nonfog_nmae", "nonfog_cossim"}:
            ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"R2_all_subjects_{label}.png", dpi=180)
        plt.close(fig)


def run_round2(
    args: argparse.Namespace,
    subjects_data: dict[str, SubjectData],
    subject_ids: Sequence[str],
    selected_loss: str,
    device: torch.device,
) -> dict[str, Any]:
    root = args.output_dir / "round2"
    figure_root = args.output_dir / "figures" / "round2"
    seeds = (SEEDS[0],) if args.screen_only else SEEDS
    rows: list[dict[str, Any]] = []
    primary_payload: dict[str, dict[str, np.ndarray]] = {}
    for subject_id in subject_ids:
        subject = subjects_data[subject_id]
        for seed in seeds:
            run_dir = root / subject_id / f"seed{seed}"
            metrics, payload, history = run_record_model(
                args, subject, "round2_mlp65", selected_loss, seed,
                args.round2_epochs, device, run_dir,
            )
            rows.append(metrics)
            if seed == SEEDS[0]:
                primary_payload[subject_id] = payload
                round2_subject_figures(subject, metrics, payload, history, figure_root / subject_id)
    write_csv(args.output_dir / "metrics" / "round2_metrics.csv", rows)
    plot_round2_summary(rows, subject_ids, figure_root)
    by_subject = {
        subject: [row for row in rows if row["subject_id"] == subject] for subject in subject_ids
    }
    macro_nmae = float(np.mean([np.mean([row["test_nonfog_nmae_floor"] for row in values]) for values in by_subject.values()]))
    macro_cosine = float(np.mean([np.mean([row["test_nonfog_cossim"] for row in values]) for values in by_subject.values()]))
    positive_skill_subjects = int(sum(np.mean([row["test_median_template_skill"] for row in values]) > 0 for values in by_subject.values()))
    std_in_range_subjects = int(sum(
        0.9 <= np.mean([row["output_target_std_ratio"] for row in values]) <= 1.1
        for values in by_subject.values()
    ))
    best_not_max = int(sum(
        np.median([row["best_epoch"] for row in values]) < args.round2_epochs for values in by_subject.values()
    ))
    converged_train = int(sum(np.mean([row["train_nmae_floor"] for row in values]) < 0.10 for values in by_subject.values()))
    gate = bool(
        macro_nmae < 0.40 and macro_cosine > 0.80 and positive_skill_subjects >= 7
        and best_not_max >= 5 and std_in_range_subjects >= 6
    )
    decision = {
        "round2_gate": "Pass" if gate else "Fail",
        "selected_loss": selected_loss,
        "seeds": list(seeds),
        "macro_test_nonfog_nmae_floor": macro_nmae,
        "macro_test_nonfog_cossim": macro_cosine,
        "positive_median_template_skill_subjects": positive_skill_subjects,
        "output_std_ratio_in_range_subjects": std_in_range_subjects,
        "best_epoch_not_at_max_subjects": best_not_max,
        "train_converged_subjects": converged_train,
        "enter_round3": bool(converged_train >= 7 and best_not_max >= 5),
    }
    atomic_json(root / "decision.json", decision)
    report = [
        "# Round 2: full clean Non-FoG reconstruction", "",
        f"- Engineering gate: {decision['round2_gate']}",
        f"- Macro test Non-FoG NMAE floor: {macro_nmae:.4f}",
        f"- Macro test Non-FoG CosSim: {macro_cosine:.4f}",
        f"- Positive median-template skill: {positive_skill_subjects}/{len(subject_ids)} subjects",
        f"- Output std ratio in [0.9,1.1]: {std_in_range_subjects}/{len(subject_ids)} subjects",
        f"- Train NMAE floor <0.10: {converged_train}/{len(subject_ids)} subjects",
        f"- Enter Round 3: {decision['enter_round3']}", "",
        "| Subject | Seed | Best epoch | Test NMAE floor | CosSim | Median skill | FoG AUROC | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    report += [
        f"| {row['subject_id']} | {row['seed']} | {row['best_epoch']} | {row['test_nonfog_nmae_floor']:.4f} | "
        f"{row['test_nonfog_cossim']:.4f} | {row['test_median_template_skill']:.3f} | {row['fog_auc']:.3f} | {row['status']} |"
        for row in rows
    ]
    report_path = args.output_dir / "reports" / "round2_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"ROUND2 {decision['round2_gate']} macro_floor={macro_nmae:.4f} "
        f"cos={macro_cosine:.4f} enter_round3={decision['enter_round3']}", flush=True,
    )
    return decision


def minmax(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    if high - low <= EPS:
        return {key: 1.0 for key in values}
    if higher_is_better:
        return {key: (value - low) / (high - low) for key, value in values.items()}
    return {key: (high - value) / (high - low) for key, value in values.items()}


def model_aggregate(rows: Sequence[dict[str, Any]], model_ids: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model_id in model_ids:
        selected = [row for row in rows if row["model_id"] == model_id]
        by_subject = defaultdict(list)
        for row in selected:
            by_subject[row["subject_id"]].append(row)
        subject_nmae = [np.mean([row["test_nonfog_nmae_floor"] for row in values]) for values in by_subject.values()]
        subject_cosine = [np.mean([row["test_nonfog_cossim"] for row in values]) for values in by_subject.values()]
        subject_skill = [np.mean([row["test_median_template_skill"] for row in values]) for values in by_subject.values()]
        result.append({
            "model_id": model_id,
            "seeds": ";".join(map(str, sorted({int(row["seed"]) for row in selected}))),
            "run_count": len(selected),
            "macro_nmae_floor": float(np.mean(subject_nmae)),
            "between_subject_nmae_std": float(np.std(subject_nmae)),
            "mean_within_subject_seed_nmae_std": float(np.mean([
                np.std([row["test_nonfog_nmae_floor"] for row in values]) for values in by_subject.values()
            ])),
            "worst_seed_subject_nmae": float(max(row["test_nonfog_nmae_floor"] for row in selected)),
            "macro_cossim": float(np.mean(subject_cosine)),
            "macro_skill": float(np.mean(subject_skill)),
            "macro_fog_auc": float(np.mean([row["fog_auc"] for row in selected])),
            "macro_fog_prauc": float(np.mean([row["fog_prauc"] for row in selected])),
            "parameter_count": int(np.median([row["parameter_count"] for row in selected])),
            "inference_time_ms": float(np.mean([row["inference_time_ms"] for row in selected])),
            "best_epoch_mean": float(np.mean([row["best_epoch"] for row in selected])),
        })
    return result


def score_models(aggregate: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    model_ids = [row["model_id"] for row in aggregate]
    lookup = {row["model_id"]: row for row in aggregate}
    nmae = minmax({model: lookup[model]["macro_nmae_floor"] for model in model_ids}, higher_is_better=False)
    cosine = minmax({model: lookup[model]["macro_cossim"] for model in model_ids}, higher_is_better=True)
    skill = minmax({model: lookup[model]["macro_skill"] for model in model_ids}, higher_is_better=True)
    stability = minmax({model: lookup[model]["mean_within_subject_seed_nmae_std"] for model in model_ids}, higher_is_better=False)
    fog = minmax({model: lookup[model]["macro_fog_auc"] for model in model_ids}, higher_is_better=True)
    complexity = minmax({model: float(lookup[model]["parameter_count"]) for model in model_ids}, higher_is_better=False)
    result = []
    for model in model_ids:
        score = 0.35 * nmae[model] + 0.20 * cosine[model] + 0.15 * skill[model] + 0.15 * stability[model] + 0.10 * fog[model] + 0.05 * complexity[model]
        result.append({
            **lookup[model], "nmae_score": nmae[model], "cossim_score": cosine[model],
            "skill_score": skill[model], "stability_score": stability[model],
            "fog_score": fog[model], "complexity_score": complexity[model],
            "composite_score": score,
        })
    return sorted(result, key=lambda row: (-row["composite_score"], row["macro_nmae_floor"], row["model_id"]))


def plot_round3_summary(
    rows: Sequence[dict[str, Any]], aggregate: Sequence[dict[str, Any]],
    primary_payload: dict[tuple[str, str], dict[str, np.ndarray]],
    subjects_data: dict[str, SubjectData], subject_ids: Sequence[str], figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = list(ROUND3_MODELS)
    primary_rows = [row for row in rows if int(row["seed"]) == SEEDS[0]]
    for metric, filename, ylabel in (
        ("test_nonfog_nmae_floor", "R3_model_nonfog_nmae_comparison.png", "Test Non-FoG NMAE floor"),
        ("test_nonfog_cossim", "R3_model_nonfog_cossim_comparison.png", "Test Non-FoG CosSim"),
        ("test_median_template_skill", "R3_model_skill_comparison.png", "Median-template skill"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        values = [[row[metric] for row in primary_rows if row["model_id"] == model] for model in labels]
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        if metric == "test_median_template_skill":
            ax.axhline(0.0, color="black", linestyle="--")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)

    matrix = np.empty((len(subject_ids), len(labels)), dtype=float)
    for row_index, subject in enumerate(subject_ids):
        values = np.asarray([
            next(row["test_nonfog_nmae_floor"] for row in primary_rows if row["subject_id"] == subject and row["model_id"] == model)
            for model in labels
        ])
        rank = np.argsort(np.argsort(values)) + 1
        matrix[row_index] = rank
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=len(labels), aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(subject_ids)), subject_ids)
    ax.set_title("Round 3 subject-wise NMAE rank (1=best)")
    for i in range(len(subject_ids)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "R3_subject_model_rank_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in labels:
        parts = [primary_payload[(subject, model)]["frequency_error"] for subject in subject_ids]
        minimum = min(len(part) for part in parts)
        frequency = FULL_FREQ if minimum == 65 else CROP_FREQ
        values = np.mean([part[:minimum] for part in parts], axis=0)
        ax.plot(frequency[:minimum], values, label=model)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Round 3 frequency error curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "R3_frequency_error_curve.png", dpi=180)
    plt.close(fig)

    channel_matrix = np.asarray([
        [
            np.mean([
                np.mean(primary_payload[(subject, model)]["channel_frequency_error"][channel])
                for subject in subject_ids
            ])
            for model in labels
        ]
        for channel in range(CHANNELS)
    ])
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(channel_matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(CHANNELS), subjects_data[subject_ids[0]].channel_names, fontsize=8)
    ax.set_title("Round 3 channel-model test Non-FoG MAE")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "R3_channel_model_error_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [row["parameter_count"] for row in aggregate],
        [row["macro_nmae_floor"] for row in aggregate],
    )
    for row in aggregate:
        ax.annotate(row["model_id"], (row["parameter_count"], row["macro_nmae_floor"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Macro test Non-FoG NMAE floor")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "R3_parameter_count_vs_nmae.png", dpi=180)
    plt.close(fig)

    representative = "S03" if "S03" in subject_ids else subject_ids[0]
    subject = subjects_data[representative]
    timeline_payloads: list[tuple[str, dict[str, np.ndarray]]] = []
    for model in labels:
        payload = primary_payload[(representative, model)]
        timeline_payloads.append((model, payload))
        clean = payload["test_clean_nonfog"].astype(bool)
        actual, predicted = payload["test_target"][clean], payload["test_reconstruction"][clean]
        floor = model_energy_floor(subject, model)
        frequency = model_frequency(model)
        _, median, _ = best_median_worst(actual, predicted, floor, frequency)
        plot_overlay(
            actual, predicted, median, frequency, subject.channel_names,
            figure_dir / f"R3_{representative}_{model.split('_')[0]}_overlay.png",
            f"Round 3 {representative} {model}",
        )
    plot_residual_timeline(
        timeline_payloads, subject.metadata["test"],
        figure_dir / f"R3_{representative}_all_models_residual_timeline.png",
    )
    b3 = primary_payload[(representative, "B3_shape_energy_conv24")]
    clean_indices = np.flatnonzero(b3["test_clean_nonfog"].astype(bool))
    actual_shape, actual_energy = shape_energy_targets(subject.arrays["test_power"])
    index = int(clean_indices[len(clean_indices) // 2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    image = axes[0].imshow(actual_shape[index], aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("True spectral shape")
    axes[1].imshow(b3["test_predicted_shape"][index], aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Reconstructed spectral shape")
    fig.colorbar(image, ax=axes)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.88, wspace=0.25)
    fig.savefig(figure_dir / f"R3_{representative}_B3_shape_reconstruction.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(actual_energy[clean_indices].reshape(-1), b3["test_predicted_energy"][clean_indices].reshape(-1), s=7, alpha=0.35)
    low = float(min(actual_energy[clean_indices].min(), b3["test_predicted_energy"][clean_indices].min()))
    high = float(max(actual_energy[clean_indices].max(), b3["test_predicted_energy"][clean_indices].max()))
    ax.plot([low, high], [low, high], "--", color="black")
    ax.set_xlabel("True log energy")
    ax.set_ylabel("Predicted log energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / f"R3_{representative}_B3_energy_true_vs_pred.png", dpi=180)
    plt.close(fig)


def run_round3(
    args: argparse.Namespace,
    subjects_data: dict[str, SubjectData],
    subject_ids: Sequence[str],
    selected_loss: str,
    device: torch.device,
) -> dict[str, Any]:
    root = args.output_dir / "round3"
    figure_dir = args.output_dir / "figures" / "round3"
    rows: list[dict[str, Any]] = []
    primary_payload: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for subject_id in subject_ids:
        subject = subjects_data[subject_id]
        for model_id in ROUND3_MODELS:
            metrics, payload, _ = run_record_model(
                args, subject, model_id, selected_loss, SEEDS[0],
                args.round3_epochs, device, root / subject_id / model_id / f"seed{SEEDS[0]}",
            )
            rows.append(metrics)
            primary_payload[(subject_id, model_id)] = payload
    screening_aggregate = model_aggregate(rows, ROUND3_MODELS)
    screening_scores = score_models(screening_aggregate)
    top_models = [row["model_id"] for row in screening_scores[:2]]
    if not args.screen_only:
        for subject_id in subject_ids:
            subject = subjects_data[subject_id]
            for model_id in top_models:
                for seed in SEEDS[1:]:
                    metrics, _, _ = run_record_model(
                        args, subject, model_id, selected_loss, seed,
                        args.round3_epochs, device, root / subject_id / model_id / f"seed{seed}",
                    )
                    rows.append(metrics)
    write_csv(args.output_dir / "metrics" / "round3_metrics.csv", rows)
    final_aggregate = model_aggregate(rows, ROUND3_MODELS)
    final_scores = score_models(final_aggregate)
    write_csv(args.output_dir / "metrics" / "round3_model_summary.csv", final_scores)
    selected_model = next(row["model_id"] for row in final_scores if row["model_id"] in top_models)
    decision = {
        "round3_complete": True,
        "screening_top_models": top_models,
        "selected_model": selected_model,
        "selected_loss": selected_loss,
        "screening_run_count": len(subject_ids) * len(ROUND3_MODELS),
        "total_run_count": len(rows),
        "scores": final_scores,
    }
    atomic_json(root / "decision.json", decision)
    plot_round3_summary(rows, final_aggregate, primary_payload, subjects_data, subject_ids, figure_dir)
    report = [
        "# Round 3: spectrum representation and backbone ablation", "",
        f"- Screening top two: {', '.join(top_models)}",
        f"- Final selected model: `{selected_model}`",
        f"- Total runs: {len(rows)}", "",
        "| Model | Seeds | Macro NMAE floor | CosSim | Skill | FoG AUROC | Params | Seed SD | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report += [
        f"| {row['model_id']} | {row['seeds']} | {row['macro_nmae_floor']:.4f} | {row['macro_cossim']:.4f} | "
        f"{row['macro_skill']:.3f} | {row['macro_fog_auc']:.3f} | {row['parameter_count']} | "
        f"{row['mean_within_subject_seed_nmae_std']:.4f} | {row['composite_score']:.3f} |"
        for row in final_scores
    ]
    report_path = args.output_dir / "reports" / "round3_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"ROUND3 complete selected={selected_model} top2={top_models} runs={len(rows)}", flush=True)
    return decision


def spectrum_metadata_rows(subjects_data: dict[str, SubjectData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in subjects_data.values():
        for split in ("train", "validation", "test"):
            log_power = subject.arrays[f"{split}_log"]
            for meta, spectrum in zip(subject.metadata[split], log_power):
                peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
                rows.append({
                    **meta,
                    "spectrum_total_energy": float(np.sum(spectrum)),
                    "dominant_channel": subject.channel_names[int(peak[0])],
                    "dominant_frequency_hz": float(FULL_FREQ[int(peak[1])]),
                    "spectrum_min": float(np.min(spectrum)),
                    "spectrum_max": float(np.max(spectrum)),
                    "all_finite": bool(np.isfinite(spectrum).all()),
                })
    return rows


def validate_splits(subjects_data: dict[str, SubjectData]) -> None:
    for subject in subjects_data.values():
        support: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for split, rows in subject.metadata.items():
            for row in rows:
                support[split].append((str(row["record_id"]), int(row["start_index"]), int(row["end_index_exclusive"])))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            for left_record, left_start, left_end in support[left]:
                for right_record, right_start, right_end in support[right]:
                    if left_record == right_record and max(left_start, right_start) < min(left_end, right_end):
                        raise AssertionError(f"{subject.subject}: {left}/{right} support overlap")
        for split in ("train", "validation"):
            if np.any(subject.arrays[f"{split}_label"] != 0):
                raise AssertionError(f"{subject.subject}: FoG entered {split} NBM pool")
        for key, array in subject.arrays.items():
            if key.endswith(("_power", "_log")) and not np.isfinite(array).all():
                raise FloatingPointError(f"{subject.subject}/{key}: non-finite data")


def load_decision(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prerequisite decision: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def final_model_config(round1: dict[str, Any], round3: dict[str, Any]) -> dict[str, Any]:
    selected = str(round3["selected_model"])
    summary = next(row for row in round3["scores"] if row["model_id"] == selected)
    if selected == "B0_gru65":
        frequency_range, bins, representation, backbone = "0-32 Hz", 65, "log-power", "frequency-axis GRU"
    elif selected == "B1_mlp24":
        frequency_range, bins, representation, backbone = "0.5-12 Hz", 24, "log-power", "MLP-AE"
    elif selected == "B2_conv24":
        frequency_range, bins, representation, backbone = "0.5-12 Hz", 24, "log-power", "Frequency-Conv-AE"
    else:
        frequency_range, bins, representation, backbone = "0.5-12 Hz", 24, "shape-energy power", "dual-head Frequency-Conv-AE"
    return {
        "nbm": {
            "representation": {
                "frequency_range": frequency_range, "num_bins": bins,
                "use_log_psd": selected != "B3_shape_energy_conv24",
                "shape_energy_split": selected == "B3_shape_energy_conv24",
            },
            "model": {
                "backbone": backbone, "latent_dim": 32,
                "parameter_count": summary["parameter_count"],
            },
            "training": {
                "loss": round1["selected_loss"] if selected != "B3_shape_energy_conv24" else "0.6 shape SmoothL1 + 0.2 EMD + 0.2 energy SmoothL1",
                "optimizer": "Adam", "learning_rate": 1e-3,
                "max_epochs": 300, "patience": 30, "weight_decay": 0,
                "augmentation": False,
            },
            "calibration": {
                "scaler": "channel RobustScaler (median/IQR with std fallback)",
                "fitted_on": "training clean Non-FoG unique raw samples",
                "residual_standardization": None,
            },
            "performance": {
                "mean_nonfog_nmae_floor": summary["macro_nmae_floor"],
                "mean_nonfog_cossim": summary["macro_cossim"],
                "mean_template_skill": summary["macro_skill"],
                "fog_residual_auc": summary["macro_fog_auc"],
            },
        }
    }


def write_final_report(
    output_dir: Path, round1: dict[str, Any], round2: dict[str, Any], round3: dict[str, Any]
) -> None:
    selected = str(round3["selected_model"])
    selected_summary = next(row for row in round3["scores"] if row["model_id"] == selected)
    frequency_range = "0-32 Hz" if selected == "B0_gru65" else "0.5-12 Hz"
    score_table = "\n".join(
        f"| {row['model_id']} | {row['seeds']} | {row['macro_nmae_floor']:.4f} | "
        f"{row['macro_cossim']:.4f} | {row['macro_skill']:.3f} | "
        f"{row['mean_within_subject_seed_nmae_std']:.4f} | {row['macro_fog_auc']:.3f} | "
        f"{row['composite_score']:.3f} |"
        for row in round3["scores"]
    )
    if selected == "B3_shape_energy_conv24":
        interpretation = "shape-energy 解耦在综合重构、稳定性和复杂度权衡下最佳。"
    elif selected == "B2_conv24":
        interpretation = "缩频后的局部频率卷积有效，shape-energy 双头没有带来足够额外收益。"
    elif selected == "B1_mlp24":
        b2 = next(row for row in round3["scores"] if row["model_id"] == "B2_conv24")
        interpretation = (
            f"B2 的 NMAE_floor 略低于 B1（{b2['macro_nmae_floor']:.4f} vs "
            f"{selected_summary['macro_nmae_floor']:.4f}），但 B1 的 CosSim 更高（"
            f"{selected_summary['macro_cossim']:.4f} vs {b2['macro_cossim']:.4f}）、"
            f"跨种子 NMAE 标准差更低（{selected_summary['mean_within_subject_seed_nmae_std']:.4f} vs "
            f"{b2['mean_within_subject_seed_nmae_std']:.4f}），因此按预设综合评分选择 B1；"
            "24-bin 维度下卷积结构没有稳定的综合优势。"
        )
    else:
        interpretation = "完整 65-bin 的频率序列关系仍有价值，当前 GRU-NBM 综合最佳。"
    if round2["round2_gate"] != "Pass":
        if round2["train_converged_subjects"] >= 7:
            domain_note = (
                "第二轮未达到全部工程门槛，但训练池基本收敛，因此第三轮结果应解释为表示/骨干对记录域偏移的相对缓解，"
                "不是独立记录问题已经完全解决。"
            )
        else:
            domain_note = (
                "第二轮的训练池收敛与独立记录工程门槛均未充分达到；第三轮仍按预注册方案完成，"
                "但其结果只能作为表示/骨干的探索性相对比较。"
            )
    else:
        domain_note = "第二轮达到工程门槛，第三轮比较建立在可用的独立记录重构基础上。"
    report = f"""# Daphnet Non-FoG 频谱 NBM 三轮实验总报告

## 1. 实验目标

依次检验低能量窗口的损失权重、完整 Non-FoG 的独立记录重构，以及频谱范围/表示/骨干消融。FoG 从未参与模型训练或 epoch 选择，仅在冻结测试记录上作残差探索。

## 2. 数据与记录划分

- 8 名实际被试：{', '.join(SUBJECTS)}（Daphnet 中排除无 FoG 的 S04/S10）。
- Train/Validation/Test 先按完整记录划分；仅 S02、S07 在首条记录内使用互不重叠的连续时间块划分 Train/Validation。
- 所有训练/验证 NBM 窗口均为带 FoG 前 2 秒、后 1 秒保护区的 clean Non-FoG。
- Daphnet 发布文件没有逐样本 task ID，因此 task 字段明确记录为 unavailable。

## 3. 第一轮：平衡损失

- 门控：{round1['round1_gate']}
- 选定损失：`{round1['selected_loss']}`
- 通过被试：{round1['passing_subjects']}/3；通过子集：{round1['passing_cells']}/{round1['total_cells']}。

## 4. 第二轮：独立记录纯净重构

- 工程门控：{round2['round2_gate']}
- 宏平均 Test Non-FoG NMAE floor：{round2['macro_test_nonfog_nmae_floor']:.4f}
- 宏平均 Test Non-FoG CosSim：{round2['macro_test_nonfog_cossim']:.4f}
- 相对中位模板正 skill：{round2['positive_median_template_skill_subjects']}/8 名被试
- 训练收敛被试：{round2['train_converged_subjects']}/8

{domain_note}

## 5. 第三轮：表示与骨干消融

- 单种子筛查前两名：{', '.join(round3['screening_top_models'])}
- 最终模型：`{selected}`
- 频率范围：{frequency_range}
- 宏平均 NMAE floor：{selected_summary['macro_nmae_floor']:.4f}
- 宏平均 CosSim：{selected_summary['macro_cossim']:.4f}
- 宏平均中位模板 skill：{selected_summary['macro_skill']:.3f}
- 探索性 FoG 残差 AUROC：{selected_summary['macro_fog_auc']:.3f}

| 模型 | 种子 | NMAE floor | CosSim | 模板 skill | 种子内 NMAE SD | FoG AUROC | 综合分数 |
|---|---|---:|---:|---:|---:|---:|---:|
{score_table}

{interpretation}

## 6. 最终结论与后续

最终配置已写入 `configs/final_nbm_config.json`。若独立记录性能仍整体偏低，下一步优先验证记录级频谱校准和 6 秒 Welch 频谱，再考虑多正常原型；当前证据不支持直接通过增加网络深度解决记录域偏移。
"""
    path = output_dir / "reports" / "nbm_three_round_final_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    subject_ids = csv_values(args.subjects)
    if set(subject_ids) - set(SUBJECTS):
        raise ValueError(f"Unsupported subjects: {sorted(set(subject_ids) - set(SUBJECTS))}")
    if args.stage in {"round1", "all"} and not set(ROUND1_SUBJECTS).issubset(subject_ids):
        raise ValueError("Round 1 requires S03, S06, and S01")
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = small.DaphnetDataset.load(args.data_dir)
    if dataset.sampling_rate_hz != FS or dataset.n_channels != CHANNELS:
        raise ValueError("Expected Daphnet 64 Hz / 9 channels")
    subjects_data: dict[str, SubjectData] = {}
    split_rows: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject, rows = prepare_subject(dataset, subject_id)
        subjects_data[subject_id] = subject
        split_rows.extend(rows)
        print(
            f"DATA {subject_id} train={len(subject.arrays['train_log'])} "
            f"val={len(subject.arrays['validation_log'])} test={len(subject.arrays['test_log'])}", flush=True,
        )
    validate_splits(subjects_data)
    write_csv(args.output_dir / "manifests" / "subject_record_split.csv", split_rows)
    write_csv(args.output_dir / "manifests" / "spectrum_metadata.csv", spectrum_metadata_rows(subjects_data))
    resolved = {
        "experiment": EXPERIMENT, "subjects": list(subject_ids), "stage": args.stage,
        "round1_epochs": args.round1_epochs, "round2_epochs": args.round2_epochs,
        "round3_epochs": args.round3_epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "screen_only": args.screen_only, "device": str(device), "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "spectrum": "log1p(abs(rfft(hann*x))**2/sum(hann**2))",
        "fog_guard": {"before_sec": 2, "after_sec": 1},
    }
    atomic_json(args.output_dir / "configs" / "resolved_config.json", resolved)
    source = REPO_ROOT / "configs" / "daphnet_spectrum_nbm_three_rounds.yaml"
    if source.exists():
        (args.output_dir / "configs" / "base_config.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    if args.stage == "round1":
        run_round1(args, subjects_data, device)
        print(f"COMPLETE round1={args.output_dir / 'reports' / 'round1_report.md'}", flush=True)
        return

    round1 = (
        run_round1(args, subjects_data, device)
        if args.stage == "all"
        else load_decision(args.output_dir / "round1" / "decision.json")
    )
    if args.stage == "round2":
        run_round2(args, subjects_data, subject_ids, round1["selected_loss"], device)
        print(f"COMPLETE round2={args.output_dir / 'reports' / 'round2_report.md'}", flush=True)
        return

    round2 = (
        run_round2(args, subjects_data, subject_ids, round1["selected_loss"], device)
        if args.stage == "all"
        else load_decision(args.output_dir / "round2" / "decision.json")
    )
    round3 = run_round3(args, subjects_data, subject_ids, round1["selected_loss"], device)
    final_config = final_model_config(round1, round3)
    atomic_json(args.output_dir / "configs" / "final_nbm_config.json", final_config)
    write_final_report(args.output_dir, round1, round2, round3)
    audit = {
        "record_split_rows": len(split_rows),
        "spectrum_metadata_rows": len(spectrum_metadata_rows(subjects_data)),
        "round1_runs": len(list((args.output_dir / "round1").rglob("metrics.json"))),
        "round2_runs": len(list((args.output_dir / "round2").rglob("metrics.json"))),
        "round3_runs": len(list((args.output_dir / "round3").rglob("metrics.json"))),
        "checkpoints": len(list(args.output_dir.rglob("checkpoint.pt"))),
        "prediction_files": len(list(args.output_dir.rglob("predictions.npz"))),
        "figure_files": len(list((args.output_dir / "figures").rglob("*.png"))),
        "splits_validated_no_overlap": True,
        "all_train_validation_labels_nonfog": True,
        "task_metadata_available": False,
    }
    atomic_json(args.output_dir / "artifact_audit.json", audit)
    print(f"COMPLETE report={args.output_dir / 'reports' / 'nbm_three_round_final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
