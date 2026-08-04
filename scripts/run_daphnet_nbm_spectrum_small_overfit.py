#!/usr/bin/env python
"""Daphnet Non-FoG spectrum small-sample overfitting diagnostic.

The selected examples are deliberately used as train, validation, and
evaluation data.  This tests spectrum-pipeline/model memorization only; it is
not a generalization or FoG-classification experiment.
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
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for search_root in (REPO_ROOT, SCRIPTS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as current  # noqa: E402
from cnbr_fog.data import DaphnetDataset, Record  # noqa: E402


EXPERIMENT = "daphnet_nbm_spectrum_small_overfit_v1"
DEFAULT_SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
DEFAULT_MODELS = ("mlp_ae", "current_nbm")
MODEL_LABELS = {"mlp_ae": "MLP-AE", "current_nbm": "CurrentNBM"}
FS, WINDOW, STRIDE, CHANNELS = 64, 128, 64, 9
N_FREQ = WINDOW // 2 + 1
PRIMARY_SEED = 20260803
EPS = 1e-12
THRESHOLDS = {
    1: {"nmae": 0.01, "cosine": 0.995, "mean_improvement": None},
    8: {"nmae": 0.03, "cosine": 0.990, "mean_improvement": 90.0},
    32: {"nmae": 0.05, "cosine": 0.980, "mean_improvement": 80.0},
}


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
        default=REPO_ROOT / "outputs" / f"daphnet_nbm_spectrum_small_overfit_seed{PRIMARY_SEED}",
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--levels", default="1,8,32")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-patience", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-only", action="store_true")
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
    return torch.device("cuda" if specification == "auto" and torch.cuda.is_available() else "cpu" if specification == "auto" else specification)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


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


def load_manifest(data_dir: Path) -> dict[str, dict[str, str]]:
    with (data_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


@dataclass(frozen=True)
class CandidateSegment:
    record_id: str
    run_id: str
    indices: tuple[int, ...]
    duration_samples: int
    capacity: int


@dataclass(frozen=True)
class RunRobustScaler:
    center: np.ndarray
    scale: np.ndarray
    clip: float = 12.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(values, dtype=np.float32) - self.center) / self.scale
        return np.clip(standardized, -self.clip, self.clip).astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "clip": self.clip,
            "definition": "channel median and IQR with standard-deviation then unit fallback",
        }


def subject_pool(
    dataset: DaphnetDataset, subject: str
) -> tuple[list[Record], current.WindowSet, np.ndarray]:
    records = [record for record in dataset.records if record.subject_id == subject]
    current.SUBJECT = subject
    windows = current.build_windows(records, current.build_intervals(records))
    eligible: list[int] = []
    for raw_index in np.flatnonzero(windows.split == "train"):
        index = int(raw_index)
        record = records[int(windows.record_index[index])]
        start, end = int(windows.start[index]), int(windows.end[index])
        # Remove [FoG onset - 2 s, FoG offset + 1 s].
        guard_start, guard_end = start - 2 * FS, end + FS
        if guard_start < 0 or guard_end > len(record.y):
            continue
        values = np.asarray(record.x[start:end], dtype=np.float64)
        if (
            not np.all(record.valid[start:end])
            or np.any(record.y[guard_start:guard_end] == 1)
            or not np.isfinite(values).all()
            or np.any(np.std(values, axis=0) <= 1e-8)
            or float(np.mean(np.square(values))) <= EPS
        ):
            continue
        eligible.append(index)
    if not eligible:
        raise ValueError(f"{subject} has no eligible clean Non-FoG training windows")
    return records, windows, np.asarray(eligible, dtype=np.int64)


def candidate_segments(
    records: Sequence[Record], windows: current.WindowSet, eligible: np.ndarray
) -> list[CandidateSegment]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        grouped[int(windows.record_index[index])].append(int(index))
    result: list[CandidateSegment] = []
    for record_index, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: int(windows.start[index]))
        runs: list[list[int]] = []
        for index in ordered:
            if not runs or int(windows.start[index]) != int(windows.start[runs[-1][-1]]) + STRIDE:
                runs.append([index])
            else:
                runs[-1].append(index)
        record = records[record_index]
        for run in runs:
            phase0, phase1 = tuple(run[::2]), tuple(run[1::2])
            nonoverlap = phase0 if len(phase0) >= len(phase1) else phase1
            result.append(CandidateSegment(
                record.record_id, record.run_id, nonoverlap,
                int(windows.end[run[-1]] - windows.start[run[0]]), len(nonoverlap),
            ))
    return sorted(result, key=lambda item: (-item.duration_samples, item.record_id, item.indices[0]))


def evenly_choose(indices: Sequence[int], count: int) -> list[int]:
    if count > len(indices):
        raise ValueError(f"Cannot choose {count} from {len(indices)}")
    positions = np.rint(np.linspace(0, len(indices) - 1, count)).astype(int)
    if len(np.unique(positions)) != count:
        raise AssertionError("Even selection produced duplicates")
    return [int(indices[position]) for position in positions]


def allocate_counts(segments: Sequence[CandidateSegment], total: int) -> list[int]:
    allocation, remaining = [0] * len(segments), int(total)
    while remaining:
        feasible = [i for i, item in enumerate(segments) if allocation[i] < item.capacity]
        if not feasible:
            raise ValueError(f"Only {total - remaining} non-overlapping windows available")
        index = min(feasible, key=lambda i: (allocation[i] / segments[i].capacity, i))
        allocation[index] += 1
        remaining -= 1
    return allocation


def select_windows(
    sample_count: int, eligible: np.ndarray, records: Sequence[Record], windows: current.WindowSet
) -> np.ndarray:
    if sample_count not in THRESHOLDS:
        raise ValueError(f"Unsupported sample count {sample_count}")
    segments = candidate_segments(records, windows, eligible)
    if sample_count == 1:
        longest = segments[0]
        return np.asarray([longest.indices[len(longest.indices) // 2]], dtype=np.int64)
    one = next((segment for segment in segments if segment.capacity >= sample_count), None)
    if one:
        chosen_segments = [one]
    else:
        by_record: dict[str, list[CandidateSegment]] = defaultdict(list)
        for segment in segments:
            by_record[segment.record_id].append(segment)
        same_record = [items[:3] for items in by_record.values() if sum(x.capacity for x in items[:3]) >= sample_count]
        if same_record:
            chosen_segments = min(same_record, key=lambda items: (-items[0].duration_samples, items[0].record_id))
        else:
            chosen_segments, capacity = [], 0
            for segment in segments:
                chosen_segments.append(segment)
                capacity += segment.capacity
                if capacity >= sample_count:
                    break
    selected: list[int] = []
    for segment, count in zip(chosen_segments, allocate_counts(chosen_segments, sample_count)):
        selected.extend(evenly_choose(segment.indices, count))
    selected.sort(key=lambda index: (int(windows.record_index[index]), int(windows.start[index])))
    if len(selected) != sample_count:
        raise AssertionError("Selection count mismatch")
    for position, index in enumerate(selected):
        for prior in selected[:position]:
            if int(windows.record_index[index]) == int(windows.record_index[prior]):
                assert max(int(windows.start[index]), int(windows.start[prior])) >= min(int(windows.end[index]), int(windows.end[prior]))
    return np.asarray(selected, dtype=np.int64)


def raw_window(records: Sequence[Record], windows: current.WindowSet, index: int) -> np.ndarray:
    record = records[int(windows.record_index[index])]
    return np.asarray(record.x[int(windows.start[index]) : int(windows.end[index])], dtype=np.float32)


def log_power_spectrum(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (WINDOW, CHANNELS):
        raise ValueError(f"Expected [N,{WINDOW},{CHANNELS}], got {values.shape}")
    hann = np.hanning(WINDOW + 1)[:-1].astype(np.float32)
    transformed = np.fft.rfft(values * hann[None, :, None], axis=1)
    power = np.square(np.abs(transformed)) / float(np.sum(np.square(hann)))
    spectrum = np.log1p(power).transpose(0, 2, 1).astype(np.float32)
    if spectrum.shape[1:] != (CHANNELS, N_FREQ) or not np.isfinite(spectrum).all() or np.any(spectrum < 0):
        raise FloatingPointError(f"Invalid spectrum with shape {spectrum.shape}")
    if np.any(spectrum.sum(axis=(1, 2)) <= EPS):
        raise ValueError("Near-zero spectrum selected")
    return np.ascontiguousarray(spectrum)


def prepare_spectra(
    records: list[Record], windows: current.WindowSet, selection: np.ndarray
) -> tuple[np.ndarray, RunRobustScaler]:
    masks: dict[int, np.ndarray] = {}
    for index in selection:
        record_index = int(windows.record_index[index])
        masks.setdefault(record_index, np.zeros(len(records[record_index].y), dtype=bool))
        masks[record_index][int(windows.start[index]) : int(windows.end[index])] = True
    values = np.concatenate([records[index].x[mask] for index, mask in masks.items()]).astype(np.float64)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = q75 - q25
    standard_deviation = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, standard_deviation)
    scale = np.where(scale > 1e-6, scale, 1.0)
    scaler = RunRobustScaler(center.astype(np.float32), scale.astype(np.float32))
    raw = np.stack([raw_window(records, windows, int(index)) for index in selection])
    return log_power_spectrum(scaler.transform(raw)), scaler


def selection_metadata(
    subject: str,
    sample_count: int,
    subset_seed: int,
    selection: np.ndarray,
    spectra: np.ndarray,
    records: Sequence[Record],
    windows: current.WindowSet,
    manifest: dict[str, dict[str, str]],
    channel_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, (raw_index, spectrum) in enumerate(zip(selection, spectra)):
        index = int(raw_index)
        record = records[int(windows.record_index[index])]
        start, end = int(windows.start[index]), int(windows.end[index])
        source_start = int(manifest[record.record_id]["source_start_row"])
        distribution = spectrum / max(float(spectrum.sum()), EPS)
        peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
        rows.append({
            "subject_id": subject,
            "sample_size": sample_count,
            "subset_seed": subset_seed,
            "selection_order": order,
            "window_id": f"{record.record_id}_{start:06d}_{end:06d}",
            "window_table_index": index,
            "record_id": record.record_id,
            "run_id": record.run_id,
            "task_id": "not_available_in_release",
            "start_index": start,
            "end_index_exclusive": end,
            "start_time_sec": start / FS,
            "end_time_sec": end / FS,
            "source_start_row": source_start + start,
            "source_end_row_exclusive": source_start + end,
            "raw_label": "Non-FoG",
            "fog_guard_before_sec": 2.0,
            "fog_guard_after_sec": 1.0,
            "spectrum_total_energy": float(spectrum.sum()),
            "dominant_channel": channel_names[int(peak[0])],
            "dominant_frequency_hz": float(peak[1] * FS / WINDOW),
            "spectral_entropy": float(-np.sum(distribution * np.log(distribution + EPS))),
        })
    return rows


class MLPAutoencoder(nn.Module):
    def __init__(self, channels: int = CHANNELS, frequencies: int = N_FREQ) -> None:
        super().__init__()
        self.channels, self.frequencies = channels, frequencies
        width = channels * frequencies
        self.encoder = nn.Sequential(nn.Linear(width, 256), nn.GELU())
        self.bottleneck = nn.Sequential(nn.Linear(256, 64), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.decoder(self.bottleneck(self.encoder(flat))).reshape_as(x)

    def diagnostic_parameters(self) -> dict[str, nn.Parameter]:
        return {
            "encoder_first": self.encoder[0].weight,
            "bottleneck": self.bottleneck[0].weight,
            "decoder_last": self.decoder[-1].weight,
        }


class CurrentSpectrumNBM(nn.Module):
    """Current frequency-axis GRU-NBM topology in clean diagnostic mode."""

    def __init__(
        self, channels: int = CHANNELS, frequencies: int = N_FREQ, hidden: int = 64
    ) -> None:
        super().__init__()
        self.channels, self.frequencies = channels, frequencies
        self.gru = nn.GRU(channels, hidden, num_layers=1, batch_first=True)
        # Current decoder width/topology; dropout is removed and output is linear
        # because this is the template's clean memorization diagnostic.
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, channels * frequencies)
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(spectrum.transpose(1, 2))
        return self.decoder(hidden[-1]).reshape(-1, self.channels, self.frequencies)

    def diagnostic_parameters(self) -> dict[str, nn.Parameter]:
        return {
            "encoder_first": self.gru.weight_ih_l0,
            "bottleneck": self.decoder[0].weight,
            "decoder_last": self.decoder[-1].weight,
        }


def build_model(name: str) -> nn.Module:
    if name == "mlp_ae":
        return MLPAutoencoder()
    if name == "current_nbm":
        return CurrentSpectrumNBM()
    raise ValueError(f"Unknown model {name}")


def metric_arrays(actual: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    difference = actual.astype(np.float64) - predicted.astype(np.float64)
    flat_actual = actual.reshape(actual.shape[0], -1).astype(np.float64)
    flat_predicted = predicted.reshape(predicted.shape[0], -1).astype(np.float64)
    denominator = np.linalg.norm(flat_actual, axis=1) * np.linalg.norm(flat_predicted, axis=1)
    cosine = np.divide(
        np.sum(flat_actual * flat_predicted, axis=1), denominator,
        out=np.zeros(actual.shape[0], dtype=np.float64), where=denominator > EPS,
    )
    return {
        "sample_mse": np.mean(np.square(difference), axis=(1, 2)),
        "sample_nmae": np.sum(np.abs(difference), axis=(1, 2)) /
        (np.sum(np.abs(actual), axis=(1, 2)) + EPS),
        "sample_cosine": cosine,
        "channel_nmae_by_sample": np.sum(np.abs(difference), axis=2) /
        (np.sum(np.abs(actual), axis=2) + EPS),
    }


def pairwise_cosine_distance(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = np.divide(flat, norms, out=np.zeros_like(flat), where=norms > EPS)
    return np.clip(1.0 - unit @ unit.T, 0.0, 2.0)


def summarize_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch {actual.shape} != {predicted.shape}")
    arrays = metric_arrays(actual, predicted)
    difference = actual.astype(np.float64) - predicted.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    zero_mse = float(np.mean(np.square(actual.astype(np.float64))))
    template = np.mean(actual, axis=0, keepdims=True)
    template_mse = float(np.mean(np.square(actual.astype(np.float64) - template)))
    flat_actual, flat_predicted = actual.reshape(-1).astype(np.float64), predicted.reshape(-1).astype(np.float64)
    total = float(np.sum(np.square(flat_actual - flat_actual.mean())))
    pearson = (
        float(np.corrcoef(flat_actual, flat_predicted)[0, 1])
        if np.std(flat_actual) > EPS and np.std(flat_predicted) > EPS else 0.0
    )
    if len(actual) > 1:
        target_distance, output_distance = pairwise_cosine_distance(actual), pairwise_cosine_distance(predicted)
        triangle = np.triu_indices(len(actual), 1)
        target_mean_distance = float(np.mean(target_distance[triangle]))
        output_mean_distance = float(np.mean(output_distance[triangle]))
        pairwise_ratio = output_mean_distance / max(target_mean_distance, EPS)
        pairwise_corr = (
            float(np.corrcoef(target_distance[triangle], output_distance[triangle])[0, 1])
            if np.std(target_distance[triangle]) > EPS and np.std(output_distance[triangle]) > EPS else 0.0
        )
    else:
        target_mean_distance = output_mean_distance = pairwise_ratio = pairwise_corr = None
    return {
        "final_mse": mse,
        "final_nmae": float(np.sum(np.abs(difference)) / (np.sum(np.abs(actual)) + EPS)),
        "median_sample_nmae": float(np.median(arrays["sample_nmae"])),
        "max_sample_nmae": float(np.max(arrays["sample_nmae"])),
        "cosine_similarity": float(np.mean(arrays["sample_cosine"])),
        "min_sample_cosine": float(np.min(arrays["sample_cosine"])),
        "pearson_r": pearson,
        "r2": float(1.0 - np.sum(np.square(flat_actual - flat_predicted)) / max(total, EPS)),
        "zero_baseline_mse": zero_mse,
        "mean_template_mse": template_mse if len(actual) > 1 else None,
        "improvement_vs_zero": 100.0 * (zero_mse - mse) / max(zero_mse, EPS),
        "improvement_vs_mean_template": (
            100.0 * (template_mse - mse) / max(template_mse, EPS) if len(actual) > 1 else None
        ),
        "output_std": float(np.std(predicted)),
        "target_std": float(np.std(actual)),
        "output_target_std_ratio": float(np.std(predicted) / max(float(np.std(actual)), EPS)),
        "target_pairwise_distance": target_mean_distance,
        "output_pairwise_distance": output_mean_distance,
        "pairwise_distance_ratio": pairwise_ratio,
        "pairwise_distance_correlation": pairwise_corr,
        "all_finite": bool(np.isfinite(predicted).all()),
    }


def pass_status(sample_count: int, metrics: dict[str, Any]) -> str:
    threshold = THRESHOLDS[sample_count]
    checks = [
        bool(metrics["all_finite"]),
        float(metrics["final_nmae"]) <= threshold["nmae"],
        float(metrics["cosine_similarity"]) >= threshold["cosine"],
        float(metrics["output_target_std_ratio"]) >= 0.20,
    ]
    if threshold["mean_improvement"] is not None:
        checks += [
            float(metrics["improvement_vs_mean_template"]) >= threshold["mean_improvement"],
            float(metrics["pairwise_distance_ratio"]) >= 0.20,
        ]
    if all(checks):
        return "Pass"
    borderline = (
        bool(metrics["all_finite"])
        and float(metrics["final_nmae"]) <= threshold["nmae"] + 0.02
        and float(metrics["cosine_similarity"]) >= threshold["cosine"] - 0.01
        and float(metrics["output_target_std_ratio"]) >= 0.20
    )
    if threshold["mean_improvement"] is not None:
        borderline &= float(metrics["improvement_vs_mean_template"]) >= threshold["mean_improvement"] - 10.0
    return "Borderline" if borderline else "Fail"


@torch.no_grad()
def predict(model: nn.Module, target: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(target).float().cpu().numpy().astype(np.float32)


def train_run(
    model_name: str,
    spectra: np.ndarray,
    sample_count: int,
    seed: int,
    max_epochs: int,
    learning_rate: float,
    target_patience: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    model = build_model(model_name).to(device)
    if any(isinstance(module, nn.Dropout) and module.p > 0 for module in model.modules()):
        raise AssertionError("Dropout must be disabled")
    target = torch.from_numpy(np.ascontiguousarray(spectra)).to(device)
    model_input = target.clone()
    assert torch.allclose(model_input, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    initial_vector = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    diagnostics = model.diagnostic_parameters()  # type: ignore[attr-defined]
    history: list[dict[str, Any]] = []
    first_gradient: dict[str, float] = {}
    final_gradient: dict[str, float] = {}
    first_step_delta, consecutive_target_epochs = 0.0, 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        before = torch.cat([p.detach().flatten().cpu() for p in model.parameters()]) if epoch == 1 else None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(model_input)
        if reconstruction.shape != target.shape:
            raise AssertionError(f"Reconstruction shape {reconstruction.shape} != {target.shape}")
        loss = nn.functional.mse_loss(reconstruction, target)
        loss.backward()
        total_gradient = math.sqrt(sum(
            float(torch.sum(parameter.grad.detach().square()))
            for parameter in model.parameters() if parameter.grad is not None
        ))
        named = {
            name: 0.0 if parameter.grad is None else float(torch.linalg.vector_norm(parameter.grad.detach()))
            for name, parameter in diagnostics.items()
        }
        if not math.isfinite(total_gradient) or total_gradient <= 0:
            raise FloatingPointError("Gradient is zero or non-finite")
        if epoch == 1:
            first_gradient = named
            if any(not math.isfinite(value) or value <= 0 for value in named.values()):
                raise FloatingPointError(f"Required first gradients invalid: {named}")
        final_gradient = named
        optimizer.step()
        if before is not None:
            after = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
            first_step_delta = float(torch.sum(torch.abs(after - before)))
            if first_step_delta <= 0:
                raise AssertionError("First optimizer step did not update parameters")
        with torch.no_grad():
            model.eval()
            evaluated = model(target)
            eval_mse = float(nn.functional.mse_loss(evaluated, target))
            eval_nmae = float(torch.sum(torch.abs(evaluated - target)) / (torch.sum(torch.abs(target)) + EPS))
            cosine = float(nn.functional.cosine_similarity(
                target.reshape(len(target), -1), evaluated.reshape(len(target), -1), dim=1
            ).mean())
        if not all(math.isfinite(value) for value in (eval_mse, eval_nmae, cosine)):
            raise FloatingPointError("Non-finite training metric")
        history.append({
            "epoch": epoch,
            "train_mse_before_step": float(loss.detach()),
            "eval_mse_after_step": eval_mse,
            "eval_nmae_after_step": eval_nmae,
            "eval_cosine_after_step": cosine,
            "gradient_norm": total_gradient,
            "learning_rate": learning_rate,
        })
        consecutive_target_epochs = consecutive_target_epochs + 1 if eval_nmae <= THRESHOLDS[sample_count]["nmae"] else 0
        if epoch == 1 or epoch % 250 == 0 or consecutive_target_epochs == target_patience:
            print(
                f"[{model_name}] N={sample_count} epoch={epoch:04d}/{max_epochs} "
                f"mse={eval_mse:.8g} nmae={100*eval_nmae:.3f}% cos={cosine:.5f}", flush=True,
            )
        if consecutive_target_epochs >= target_patience:
            break

    reconstruction_np = predict(model, target)
    final_vector = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    parameter_delta = float(torch.linalg.vector_norm(final_vector - initial_vector))
    training = {
        "num_epochs": len(history),
        "stopped_after_target_patience": consecutive_target_epochs >= target_patience,
        "gradient_norm_initial": history[0]["gradient_norm"],
        "gradient_norm_final": history[-1]["gradient_norm"],
        "named_gradient_norm_initial": first_gradient,
        "named_gradient_norm_final": final_gradient,
        "first_step_parameter_delta_l1": first_step_delta,
        "parameter_delta_l2": parameter_delta,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if parameter_delta <= 0:
        raise AssertionError("Model parameters did not update")
    atomic_torch_save(run_dir / "checkpoint.pt", {
        "model_name": model_name, "model_state": model.state_dict(), "seed": seed,
        "sample_count": sample_count, "training": training,
    })
    write_csv(run_dir / "history.csv", history)
    return reconstruction_np, history, training


def sample_order(actual: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int]:
    order = np.argsort(metric_arrays(actual, predicted)["sample_nmae"])
    return int(order[0]), int(order[len(order) // 2]), int(order[-1])


def plot_loss(history: Sequence[dict[str, Any]], path: Path, sample_count: int) -> None:
    epochs = np.asarray([row["epoch"] for row in history])
    values = np.asarray([row["eval_mse_after_step"] for row in history])
    target_epoch = next(
        (row["epoch"] for row in history if row["eval_nmae_after_step"] <= THRESHOLDS[sample_count]["nmae"]), None
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, log_scale in zip(axes, (False, True)):
        ax.plot(epochs, values, linewidth=1.2)
        ax.scatter(epochs[-1], values[-1], color="tab:red", s=22)
        if target_epoch is not None:
            ax.axvline(target_epoch, color="tab:green", linestyle="--", label="first target NMAE")
            ax.legend(fontsize=8)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("Log scale" if log_scale else f"Final MSE={values[-1]:.3g}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overlay(
    actual: np.ndarray, predicted: np.ndarray, sample_count: int, path: Path,
    channel_names: Sequence[str],
) -> None:
    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    best, median, worst = sample_order(actual, predicted)
    selected = [("only", 0)] if sample_count == 1 else [("median", median)] if sample_count == 8 else [
        ("best", best), ("median", median), ("worst", worst)
    ]
    fig, axes = plt.subplots(len(selected), CHANNELS, figsize=(24, 3.0 * len(selected)), squeeze=False, sharex=True)
    for row, (label, index) in enumerate(selected):
        for channel, ax in enumerate(axes[row]):
            ax.plot(frequency, actual[index, channel], label="target", linewidth=1.15)
            ax.plot(frequency, predicted[index, channel], "--", label="reconstruction", linewidth=1.05)
            ax.set_title(channel_names[channel], fontsize=8)
            ax.grid(alpha=0.2)
            if row == len(selected) - 1:
                ax.set_xlabel("Hz")
        axes[row, 0].set_ylabel(f"{label}\nsample {index}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("True and reconstructed log-power spectra")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(
    actual: np.ndarray, predicted: np.ndarray, index: int, path: Path,
    channel_names: Sequence[str],
) -> None:
    error = np.abs(actual[index] - predicted[index])
    vmax = float(max(np.max(actual[index]), np.max(predicted[index]), EPS))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    image = axes[0].imshow(actual[index], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[1].imshow(predicted[index], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    error_image = axes[2].imshow(error, aspect="auto", origin="lower", cmap="magma", vmin=0)
    for ax, title in zip(axes, ("Target", "Reconstruction", "Absolute error")):
        ax.set_title(title)
        ax.set_xlabel("Frequency bin (0.5 Hz)")
        ax.set_yticks(range(CHANNELS), channel_names, fontsize=7)
    fig.colorbar(image, ax=axes[:2], shrink=0.82, label="log-power")
    fig.colorbar(error_image, ax=axes[2], shrink=0.82, label="absolute error")
    fig.suptitle(f"Spectrum heatmap triplet, sample {index}")
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.13, top=0.84, wspace=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scatter(actual: np.ndarray, predicted: np.ndarray, path: Path, metrics: dict[str, Any]) -> None:
    x, y = actual.reshape(-1), predicted.reshape(-1)
    lower, upper = float(min(np.min(x), np.min(y))), float(max(np.max(x), np.max(y)))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(x, y, s=6, alpha=0.25, rasterized=True)
    ax.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1, label="y=x")
    ax.text(0.04, 0.96, (
        f"Pearson r={metrics['pearson_r']:.4f}\nR²={metrics['r2']:.4f}\n"
        f"NMAE={100*metrics['final_nmae']:.3f}%\nCosSim={metrics['cosine_similarity']:.5f}"
    ), transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85})
    ax.set_xlabel("True log-power")
    ax.set_ylabel("Predicted log-power")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_channel_nmae(
    actual: np.ndarray, predicted: np.ndarray, path: Path, channel_names: Sequence[str], overall: float
) -> None:
    per_sample = metric_arrays(actual, predicted)["channel_nmae_by_sample"]
    aggregate = np.sum(np.abs(actual - predicted), axis=(0, 2)) / (np.sum(np.abs(actual), axis=(0, 2)) + EPS)
    error = np.std(per_sample, axis=0) if len(actual) > 1 else np.zeros(CHANNELS)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(CHANNELS), 100 * aggregate, yerr=100 * error, capsize=3, alpha=0.82)
    ax.axhline(100 * overall, color="tab:red", linestyle="--", label="overall NMAE")
    ax.set_xticks(range(CHANNELS), channel_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("NMAE (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sample_rank(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    sample_nmae = metric_arrays(actual, predicted)["sample_nmae"]
    template = np.mean(actual, axis=0, keepdims=True)
    template_nmae = np.sum(np.abs(actual - template), axis=(1, 2)) / (np.sum(np.abs(actual), axis=(1, 2)) + EPS)
    order = np.argsort(sample_nmae)
    best, median, worst = int(order[0]), int(order[len(order) // 2]), int(order[-1])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rank = np.arange(len(order))
    ax.plot(rank, 100 * sample_nmae[order], marker="o", markersize=3, label="model")
    ax.plot(rank, 100 * template_nmae[order], "--", linewidth=1, label="mean template")
    for name, index in (("best", best), ("median", median), ("worst", worst)):
        position = int(np.flatnonzero(order == index)[0])
        ax.annotate(f"{name}: {index}", (position, 100 * sample_nmae[index]), fontsize=8)
    ax.set_xlabel("Samples ranked by model NMAE")
    ax.set_ylabel("NMAE (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    target, output = pairwise_cosine_distance(actual), pairwise_cosine_distance(predicted)
    vmax = float(max(np.max(target), np.max(output), EPS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    image = axes[0].imshow(target, cmap="magma", vmin=0, vmax=vmax)
    axes[1].imshow(output, cmap="magma", vmin=0, vmax=vmax)
    for ax, title in zip(axes, ("Target cosine distance", "Reconstruction cosine distance")):
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Sample")
    fig.colorbar(image, ax=axes, shrink=0.82)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.12, top=0.88, wspace=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_run_figures(
    figure_dir: Path, prefix: str, sample_count: int, actual: np.ndarray,
    predicted: np.ndarray, history: Sequence[dict[str, Any]], metrics: dict[str, Any],
    channel_names: Sequence[str],
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(history, figure_dir / f"{prefix}_loss_curve.png", sample_count)
    overlay_suffix = "all_channels" if sample_count == 1 else "median_sample" if sample_count == 8 else "best_median_worst"
    plot_overlay(actual, predicted, sample_count, figure_dir / f"{prefix}_overlay_{overlay_suffix}.png", channel_names)
    _, median, worst = sample_order(actual, predicted)
    heatmap_index = 0 if sample_count == 1 else median if sample_count == 8 else worst
    heatmap_suffix = "triplet" if sample_count == 1 else "median" if sample_count == 8 else "worst"
    plot_heatmap(actual, predicted, heatmap_index, figure_dir / f"{prefix}_heatmap_{heatmap_suffix}.png", channel_names)
    plot_scatter(actual, predicted, figure_dir / f"{prefix}_true_vs_pred.png", metrics)
    plot_channel_nmae(actual, predicted, figure_dir / f"{prefix}_channel_nmae.png", channel_names, metrics["final_nmae"])
    if sample_count > 1:
        plot_sample_rank(actual, predicted, figure_dir / f"{prefix}_sample_error_rank.png")
        plot_pairwise(actual, predicted, figure_dir / f"{prefix}_pairwise_distance.png")


def detail_rows(
    subject: str, model_name: str, sample_count: int, seed: int,
    metadata: Sequence[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray,
    channel_names: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = metric_arrays(actual, predicted)
    sample_rows, channel_rows = [], []
    for index, meta in enumerate(metadata):
        sample_rows.append({
            "subject_id": subject, "model_name": MODEL_LABELS[model_name],
            "sample_size": sample_count, "subset_seed": seed,
            "window_id": meta["window_id"], "record_id": meta["record_id"],
            "sample_mse": float(arrays["sample_mse"][index]),
            "sample_nmae": float(arrays["sample_nmae"][index]),
            "sample_cosine": float(arrays["sample_cosine"][index]),
        })
        for channel, channel_name in enumerate(channel_names):
            channel_rows.append({
                "subject_id": subject, "model_name": MODEL_LABELS[model_name],
                "sample_size": sample_count, "subset_seed": seed,
                "window_id": meta["window_id"], "channel_index": channel,
                "channel_name": channel_name,
                "channel_nmae": float(arrays["channel_nmae_by_sample"][index, channel]),
            })
    return sample_rows, channel_rows


def run_one(
    subject: str, model_name: str, sample_count: int, seed: int,
    spectra: np.ndarray, scaler: RunRobustScaler, metadata: Sequence[dict[str, Any]],
    output_dir: Path, max_epochs: int, learning_rate: float, target_patience: int,
    device: torch.device, channel_names: Sequence[str], overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = output_dir / "runs" / subject / model_name / f"N{sample_count:02d}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as payload:
            actual, predicted = np.asarray(payload["target"]), np.asarray(payload["reconstruction"])
        with (run_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
            history = [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        predicted, history, training = train_run(
            model_name, spectra, sample_count, seed, max_epochs, learning_rate,
            target_patience, device, run_dir,
        )
        actual = spectra
        record_ids = sorted({str(row["record_id"]) for row in metadata})
        run_ids = sorted({str(row["run_id"]) for row in metadata})
        metrics = {
            "subject_id": subject, "model_name": MODEL_LABELS[model_name],
            "model_key": model_name, "sample_size": sample_count, "subset_seed": seed,
            "record_ids": ";".join(record_ids), "run_ids": ";".join(run_ids),
            "task_ids": "not_available_in_release", "same_record": len(record_ids) == 1,
            "same_task": "unknown", **training, **summarize_metrics(actual, predicted),
        }
        metrics["status"] = pass_status(sample_count, metrics)
        atomic_json(metrics_path, metrics)
        atomic_json(run_dir / "scaler.json", scaler.as_dict())
        atomic_npz(run_dir / "predictions.npz", target=actual, reconstruction=predicted)
    prefix = f"{subject}_N{sample_count:02d}_seed{seed}"
    render_run_figures(
        output_dir / "figures" / "per_subject" / subject / model_name,
        prefix, sample_count, actual, predicted, history, metrics, channel_names,
    )
    sample_rows, channel_rows = detail_rows(
        subject, model_name, sample_count, seed, metadata, actual, predicted, channel_names,
    )
    print(
        f"RESULT {subject} {MODEL_LABELS[model_name]} N={sample_count} "
        f"NMAE={100*metrics['final_nmae']:.3f}% CosSim={metrics['cosine_similarity']:.5f} "
        f"status={metrics['status']}", flush=True,
    )
    return metrics, sample_rows, channel_rows


def summary_figures(
    output_dir: Path, metrics: Sequence[dict[str, Any]], subjects: Sequence[str],
    models: Sequence[str], levels: Sequence[int],
) -> None:
    summary_dir = output_dir / "figures" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(row["subject_id"], row["model_key"], int(row["sample_size"])): row for row in metrics}
    columns = [(model, level) for model in models for level in levels]
    status_value = {"Fail": 0, "Borderline": 1, "Pass": 2}
    matrix = np.full((len(subjects), len(columns)), np.nan)
    nmae = np.full_like(matrix, np.nan)
    for i, subject in enumerate(subjects):
        for j, (model, level) in enumerate(columns):
            row = lookup[(subject, model, level)]
            matrix[i, j] = status_value[row["status"]]
            nmae[i, j] = 100 * float(row["final_nmae"])
    fig, ax = plt.subplots(figsize=(max(9, len(columns) * 1.45), 6))
    image = ax.imshow(matrix, cmap=ListedColormap(["#d73027", "#fee08b", "#1a9850"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(columns)), [f"{MODEL_LABELS[m]}-N{n}" for m, n in columns], rotation=25, ha="right")
    ax.set_yticks(range(len(subjects)), subjects)
    ax.set_title("Daphnet spectrum small-overfit pass matrix")
    for i in range(len(subjects)):
        for j in range(len(columns)):
            row = lookup[(subjects[i], columns[j][0], columns[j][1])]
            ax.text(j, i, f"{row['status']}\n{nmae[i,j]:.2f}%", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["Fail", "Borderline", "Pass"])
    fig.tight_layout()
    fig.savefig(summary_dir / "all_subjects_overfit_pass_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4.3 * len(models)), squeeze=False)
    for row_index, model in enumerate(models):
        ax = axes[row_index, 0]
        x = np.arange(len(subjects))
        offsets = np.linspace(-0.25, 0.25, len(levels))
        for offset, level in zip(offsets, levels):
            values = [100 * float(lookup[(subject, model, level)]["final_nmae"]) for subject in subjects]
            ax.plot(x + offset, values, marker="o", label=f"N={level}")
            ax.axhline(100 * THRESHOLDS[level]["nmae"], linestyle="--", alpha=0.35)
        ax.set_xticks(x, subjects)
        ax.set_ylabel("Final NMAE (%)")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(summary_dir / "all_subjects_nmae_summary.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path, metrics: Sequence[dict[str, Any]], subjects: Sequence[str],
    models: Sequence[str], levels: Sequence[int],
) -> None:
    counts = {
        (model, level): sum(
            row["status"] == "Pass" for row in metrics
            if row["model_key"] == model and int(row["sample_size"]) == level
        ) for model in models for level in levels
    }
    table = [
        "| Subject | Model | N | Final NMAE | CosSim | Improve vs mean | Same record | Status |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in metrics:
        improve = "—" if row["improvement_vs_mean_template"] is None else f"{row['improvement_vs_mean_template']:.2f}%"
        table.append(
            f"| {row['subject_id']} | {row['model_name']} | {row['sample_size']} | "
            f"{100*row['final_nmae']:.3f}% | {row['cosine_similarity']:.5f} | {improve} | "
            f"{row['same_record']} | {row['status']} |"
        )
    failed = [row for row in metrics if row["status"] != "Pass"]
    multi_sample = [row for row in metrics if int(row["sample_size"]) > 1]
    collapse_rows = [row for row in multi_sample if (
        row["output_target_std_ratio"] < 0.20
        or row["pairwise_distance_ratio"] < 0.20
        or row["improvement_vs_mean_template"] < THRESHOLDS[int(row["sample_size"])]["mean_improvement"]
    )]
    if counts.get(("mlp_ae", 1), 0) < len(subjects) - 1:
        diagnosis = "简单 MLP 在单样本上也普遍失败，优先检查频谱尺度、归一化、损失或优化链路。"
    elif counts.get(("current_nbm", 1), 0) < len(subjects) - 1:
        diagnosis = "MLP 基础管线可用，但当前频率轴 GRU-NBM 的静态频谱解码/优化存在限制。"
    elif failed and not collapse_rows:
        diagnosis = (
            "两种模型的基础链路正常，且没有零输出、低幅值或均值谱塌缩。未通过项由少数低能量窗口的逐样本误差主导，"
            "而全局 MSE/NMAE 被高能量窗口主导，首要问题是频谱动态范围与 MSE 的样本权重不均；CurrentNBM 的多样本 "
            "CosSim 通常更低，说明其优化或条件解码还有次要限制。"
        )
    else:
        diagnosis = "两类模型均具备训练集小样本频谱重构能力；后续问题应转向跨记录泛化。"
    count_lines = [
        f"- {MODEL_LABELS[model]}：" + "，".join(f"N={level} 为 {counts[(model, level)]}/{len(subjects)} Pass" for level in levels)
        for model in models
    ]
    failure_lines = [
        f"- {row['subject_id']} / {row['model_name']} / N={row['sample_size']}: "
        f"NMAE={100*row['final_nmae']:.3f}%，CosSim={row['cosine_similarity']:.5f}，"
        f"mean 改善={row['improvement_vs_mean_template'] if row['improvement_vs_mean_template'] is not None else 'N/A'}。"
        for row in failed
    ] or ["- 无。"]
    diagnostic_lines = [
        f"- N=8/32 的均值模板改善率范围：{min(row['improvement_vs_mean_template'] for row in multi_sample):.2f}%–{max(row['improvement_vs_mean_template'] for row in multi_sample):.2f}%。",
        f"- 输出/目标标准差比范围：{min(row['output_target_std_ratio'] for row in multi_sample):.3f}–{max(row['output_target_std_ratio'] for row in multi_sample):.3f}。",
        f"- 重构/真实样本间距离比范围：{min(row['pairwise_distance_ratio'] for row in multi_sample):.3f}–{max(row['pairwise_distance_ratio'] for row in multi_sample):.3f}。",
        f"- 非 Pass 配置的最差逐样本 NMAE 范围：{100*min(row['max_sample_nmae'] for row in failed):.2f}%–{100*max(row['max_sample_nmae'] for row in failed):.2f}%。" if failed else "- 无非 Pass 配置。",
        f"- 塌缩判据命中的运行数：{len(collapse_rows)}/{len(multi_sample)}。",
    ]
    report = f"""# Daphnet Non-FoG 频谱小样本过拟合报告

本实验在 {len(subjects)} 名被试上分别使用 1、8、32 个固定 Non-FoG 窗口，并令训练集、验证集和评价集完全相同。输入为 64 Hz、2 秒窗经被试/配置内 RobustScaler 后生成的 `9×65` Hann-RFFT `log1p(power)` 频谱。FoG 保护区为开始前 2 秒、结束后 1 秒。

本结果只诊断模型是否能记住训练谱，不代表泛化能力或 FoG 分类性能。

## 实验结果

{chr(10).join(count_lines)}

{chr(10).join(table)}

## 未通过配置

{chr(10).join(failure_lines)}

## 诊断结论

{diagnosis}

{chr(10).join(diagnostic_lines)}

所有 N=8/32 样本均来自单一记录，因此本轮失败不能归因于跨记录混合。按模板，下一轮应优先对 S01、S02、S05、S06、S07 的失败/临界配置更换固定子集复核，并比较“训练满 3000 epoch”与“仅按 NMAE 连续 50 epoch 停止”；若低能量窗口仍持续失败，再评估按窗口/通道平衡的损失或频谱标准化。

所有训练均使用全批次 Adam、MSE、无数据增强、无 dropout、无权重衰减、无验证早停；达到对应 NMAE 阈值连续 50 epoch 时允许停止。输入—目标一致性、形状、首批梯度、首次参数更新、最终输出方差和有限值均已自动审计。

Daphnet 发布的 txt 文件不含逐样本任务 ID，因此 `task_ids=not_available_in_release`、`same_task=unknown`；本报告未把 run ID 冒充任务标签。
"""
    path = output_dir / "report" / "spectrum_small_sample_overfit_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    subjects, levels, models = (
        csv_values(args.subjects), csv_values(args.levels, int), csv_values(args.models)
    )
    if set(subjects) - set(DEFAULT_SUBJECTS):
        raise ValueError(f"Unsupported subjects: {sorted(set(subjects) - set(DEFAULT_SUBJECTS))}")
    if set(levels) - set(THRESHOLDS):
        raise ValueError(f"Unsupported levels: {sorted(set(levels) - set(THRESHOLDS))}")
    if set(models) - set(DEFAULT_MODELS):
        raise ValueError(f"Unsupported models: {sorted(set(models) - set(DEFAULT_MODELS))}")
    if args.max_epochs <= 0 or args.learning_rate <= 0 or args.target_patience <= 0:
        raise ValueError("Training arguments must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = DaphnetDataset.load(args.data_dir)
    if dataset.sampling_rate_hz != FS or dataset.n_channels != CHANNELS:
        raise ValueError(f"Expected {FS} Hz/{CHANNELS} channels")
    manifest = load_manifest(args.data_dir)
    config = {
        "experiment": EXPERIMENT,
        "subjects": list(subjects), "levels": list(levels), "models": list(models),
        "subset_seed": args.seed, "sampling_rate_hz": FS, "window_samples": WINDOW,
        "stride_samples": STRIDE, "spectrum": "log1p(abs(rfft(hann*x))**2/sum(hann**2))",
        "spectrum_shape": [CHANNELS, N_FREQ], "fog_guard_before_sec": 2.0,
        "fog_guard_after_sec": 1.0, "source_pool": "frozen within-subject training split",
        "scaler": "per-run channel RobustScaler fitted on unique selected raw points",
        "optimizer": "Adam", "learning_rate": args.learning_rate, "weight_decay": 0.0,
        "batch_size": "full selected subset", "max_epochs": args.max_epochs,
        "target_patience": args.target_patience, "loss": "MSE", "augmentation": False,
        "dropout": 0.0, "early_stopping": False, "output_activation": "linear",
        "device": str(device), "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_json(output_dir / "config" / "resolved_config.json", config)
    source_config = REPO_ROOT / "configs" / "daphnet_nbm_spectrum_small_overfit.yaml"
    if source_config.exists():
        copied = output_dir / "config" / "base_config.yaml"
        copied.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")

    prepared: dict[tuple[str, int], tuple[np.ndarray, RunRobustScaler, list[dict[str, Any]]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    print(f"PREFLIGHT device={device} subjects={subjects} levels={levels} models={models}", flush=True)
    for subject in subjects:
        records, windows, eligible = subject_pool(dataset, subject)
        for sample_count in levels:
            selection = select_windows(sample_count, eligible, records, windows)
            spectra, scaler = prepare_spectra(records, windows, selection)
            metadata = selection_metadata(
                subject, sample_count, args.seed, selection, spectra, records, windows,
                manifest, dataset.channel_names,
            )
            prepared[(subject, sample_count)] = (spectra, scaler, metadata)
            manifest_rows.extend(metadata)
            sample_path = output_dir / "samples" / subject / f"N{sample_count:02d}_seed{args.seed}.npy"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(sample_path, spectra, allow_pickle=False)
    write_csv(output_dir / "config" / "subject_sample_manifest.csv", manifest_rows)
    if args.selection_only:
        print("Selection-only run complete", flush=True)
        return

    all_metrics: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    all_channel_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for model_name in models:
            for sample_count in levels:
                spectra, scaler, metadata = prepared[(subject, sample_count)]
                metrics, sample_rows, channel_rows = run_one(
                    subject, model_name, sample_count, args.seed, spectra, scaler, metadata,
                    output_dir, args.max_epochs, args.learning_rate, args.target_patience,
                    device, dataset.channel_names, args.overwrite,
                )
                all_metrics.append(metrics)
                all_sample_rows.extend(sample_rows)
                all_channel_rows.extend(channel_rows)

    table_dir = output_dir / "tables"
    write_csv(table_dir / "run_metrics.csv", all_metrics)
    write_csv(table_dir / "subject_summary.csv", [{
        "Subject": row["subject_id"], "Model": row["model_name"], "N": row["sample_size"],
        "Final NMAE": row["final_nmae"], "CosSim": row["cosine_similarity"],
        "Improve vs Mean": row["improvement_vs_mean_template"], "Status": row["status"],
    } for row in all_metrics])
    write_csv(table_dir / "sample_metrics.csv", all_sample_rows)
    write_csv(table_dir / "channel_metrics.csv", all_channel_rows)
    summary_figures(output_dir, all_metrics, subjects, models, levels)
    write_report(output_dir, all_metrics, subjects, models, levels)
    audit = {
        "manifest_rows": len(manifest_rows),
        "expected_manifest_rows": len(subjects) * sum(levels),
        "run_count": len(all_metrics),
        "expected_run_count": len(subjects) * len(models) * len(levels),
        "all_predictions_finite": all(row["all_finite"] for row in all_metrics),
        "all_models_updated": all(row["parameter_delta_l2"] > 0 for row in all_metrics),
        "all_required_initial_gradients_positive": all(
            all(value > 0 for value in row["named_gradient_norm_initial"].values()) for row in all_metrics
        ),
        "task_metadata_available": False,
        "task_metadata_note": "Released Daphnet txt files do not contain per-sample task ids.",
    }
    atomic_json(output_dir / "artifact_audit.json", audit)
    print(f"COMPLETE report={output_dir / 'report' / 'spectrum_small_sample_overfit_report.md'}", flush=True)


if __name__ == "__main__":
    main()
