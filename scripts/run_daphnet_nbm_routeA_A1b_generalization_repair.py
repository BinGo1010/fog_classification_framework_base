#!/usr/bin/env python
"""Run the Non-FoG-only Route A A1b diagnostic repair experiment.

The script reuses the frozen A1 split and checkpoints.  It never reads FoG
windows for model selection.  Deployment calibration samples are removed from
the clean Non-FoG evaluation set with a five-second guard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for location in (REPO_ROOT, SCRIPTS_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_daphnet_nbm_routeA_final_residual_validation as a1  # noqa: E402
import run_daphnet_nbm_tcdae_three_rounds as base  # noqa: E402
import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as legacy  # noqa: E402
from cnbr_fog.data import DaphnetDataset  # noqa: E402


EXPERIMENT = "daphnet_nbm_routeA_A1b_generalization_repair_v1"
SUBJECTS = a1.SUBJECTS
SEEDS = a1.SEEDS
FAILED_SUBJECTS = ("S02", "S05", "S06", "S09")
ORIGINAL_PASS_SUBJECTS = ("S01", "S07", "S08")
CHANNELS = 9
FS = a1.FS
WINDOW = a1.WINDOW
CALIBRATION_DURATIONS = (10, 20, 30, 60)
CALIBRATION_METHODS = ("C0", "C1", "C2", "C3")
C4_EPOCHS = (5, 10, 20)
LOSSES = ("L0", "L1", "L2", "L3", "L4")
CONTEXTS = {"W0": 128, "W1": 256, "W2": 384}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_nbm_routeA_final_residual_validation_v1"
        / "routeA_final_residual_validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / EXPERIMENT,
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            r"C:\Users\bin\Downloads\Daphnet_NBM_RouteA_A1b_generalization_repair_template.md"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("diagnostics", "calibration", "loss", "context", "retest"),
        default="retest",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def original_results(parent_root: Path) -> list[dict[str, Any]]:
    return [
        read_json(path)
        for path in sorted((parent_root / "A1_nonfog_generalization").rglob("run_metrics.json"))
    ]


def result_lookup(results: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["subject_id"]), int(row["seed"])): row for row in results}


def load_original_model(
    lookup: dict[tuple[str, int], dict[str, Any]],
    subject: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        Path(lookup[(subject, seed)]["run_dir"]) / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = base.build_model("M3_tcdae_long")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def unique_raw_points(
    item: a1.PreparedSubject, indices: Sequence[int]
) -> np.ndarray:
    masks: dict[int, np.ndarray] = {}
    for raw_index in np.asarray(indices, dtype=np.int64):
        rec_idx = int(item.windows.record_index[raw_index])
        masks.setdefault(rec_idx, np.zeros(len(item.records[rec_idx].y), dtype=bool))
        masks[rec_idx][
            int(item.windows.start[raw_index]) : int(item.windows.end[raw_index])
        ] = True
    return np.concatenate(
        [item.records[rec_idx].x[mask] for rec_idx, mask in masks.items() if np.any(mask)],
        axis=0,
    ).astype(np.float32)


@dataclass
class CalibrationPlan:
    subject: str
    requested_seconds: int
    actual_seconds: float
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    calibration_values: np.ndarray
    calibration_median: np.ndarray
    calibration_scale: np.ndarray
    manifest_rows: list[dict[str, Any]]


def calibration_plan(item: a1.PreparedSubject, duration_seconds: int) -> CalibrationPlan:
    """Take early unique clean points and guard them from final evaluation."""
    if duration_seconds == 0:
        return CalibrationPlan(
            subject=item.subject,
            requested_seconds=0,
            actual_seconds=0.0,
            calibration_indices=np.empty(0, dtype=np.int64),
            evaluation_indices=np.asarray(item.test_indices, dtype=np.int64).copy(),
            calibration_values=np.empty((0, CHANNELS), dtype=np.float32),
            calibration_median=item.scaler.median.copy(),
            calibration_scale=item.scaler.iqr.copy(),
            manifest_rows=[
                {
                    "subject_id": item.subject,
                    "requested_seconds": 0,
                    "record_id": "NOT APPLICABLE",
                    "window_id": "NOT APPLICABLE",
                    "new_unique_samples": 0,
                    "cumulative_unique_samples": 0,
                    "cumulative_seconds": 0.0,
                    "clean_nonfog": True,
                }
            ],
        )
    target_points = int(duration_seconds * FS)
    masks: dict[int, np.ndarray] = {
        index: np.zeros(len(record.y), dtype=bool)
        for index, record in enumerate(item.records)
    }
    ordered = sorted(
        np.asarray(item.test_indices, dtype=np.int64).tolist(),
        key=lambda index: (
            int(item.windows.record_index[index]),
            int(item.windows.start[index]),
        ),
    )
    contributing: list[int] = []
    manifest: list[dict[str, Any]] = []
    accumulated = 0
    for raw_index in ordered:
        if accumulated >= target_points:
            break
        rec_idx = int(item.windows.record_index[raw_index])
        start = int(item.windows.start[raw_index])
        end = int(item.windows.end[raw_index])
        available = np.flatnonzero(~masks[rec_idx][start:end])
        needed = target_points - accumulated
        chosen = available[:needed]
        if chosen.size:
            masks[rec_idx][start + chosen] = True
            contributing.append(raw_index)
            accumulated += int(chosen.size)
            record = item.records[rec_idx]
            manifest.append(
                {
                    "subject_id": item.subject,
                    "requested_seconds": duration_seconds,
                    "record_id": record.record_id,
                    "window_id": f"{record.record_id}:{start}:{end}",
                    "new_unique_samples": int(chosen.size),
                    "cumulative_unique_samples": accumulated,
                    "cumulative_seconds": accumulated / FS,
                    "clean_nonfog": True,
                }
            )
    chunks = [
        item.records[rec_idx].x[mask]
        for rec_idx, mask in masks.items()
        if np.any(mask)
    ]
    if not chunks:
        raise ValueError(f"{item.subject} has no deployment calibration samples")
    values = np.concatenate(chunks, axis=0).astype(np.float32)
    median = np.median(values, axis=0).astype(np.float32)
    mad = np.median(np.abs(values - median), axis=0).astype(np.float32)
    scale = np.maximum(1.4826 * mad, 1e-6).astype(np.float32)
    evaluation: list[int] = []
    guard = 5 * FS
    for raw_index in ordered:
        rec_idx = int(item.windows.record_index[raw_index])
        start = int(item.windows.start[raw_index])
        end = int(item.windows.end[raw_index])
        lower = max(0, start - guard)
        upper = min(len(masks[rec_idx]), end + guard)
        if not np.any(masks[rec_idx][lower:upper]):
            evaluation.append(raw_index)
    if len(evaluation) < 5:
        raise ValueError(
            f"{item.subject} duration={duration_seconds}s leaves only {len(evaluation)} evaluation windows"
        )
    return CalibrationPlan(
        subject=item.subject,
        requested_seconds=duration_seconds,
        actual_seconds=len(values) / FS,
        calibration_indices=np.asarray(contributing, dtype=np.int64),
        evaluation_indices=np.asarray(evaluation, dtype=np.int64),
        calibration_values=values,
        calibration_median=median,
        calibration_scale=scale,
        manifest_rows=manifest,
    )


def method_center_scale(
    item: a1.PreparedSubject, plan: CalibrationPlan, method: str
) -> tuple[np.ndarray, np.ndarray]:
    if method == "C0":
        return item.scaler.median, item.scaler.iqr
    if method in ("C1", "C3"):
        return plan.calibration_median, item.scaler.iqr
    if method == "C2":
        return plan.calibration_median, plan.calibration_scale
    if method == "C4":
        raise ValueError("C4 must specify its base preprocessing method")
    raise ValueError(f"unknown calibration method {method}")


def transform_raw_windows(
    item: a1.PreparedSubject,
    indices: Sequence[int],
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = legacy.raw_windows(item.records, item.windows, np.asarray(indices, dtype=np.int64))
    scaled = (raw.astype(np.float32) - center) / (scale + 1e-6)
    return legacy.window_axis_center(scaled)


def calibration_arrays(
    item: a1.PreparedSubject,
    plan: CalibrationPlan,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    center, scale = method_center_scale(item, plan, method)
    calibration_x = (
        transform_raw_windows(item, plan.calibration_indices, center=center, scale=scale)
        if len(plan.calibration_indices)
        else np.empty((0, WINDOW, CHANNELS), dtype=np.float32)
    )
    evaluation_x = transform_raw_windows(
        item, plan.evaluation_indices, center=center, scale=scale
    )
    return calibration_x, evaluation_x


class ContextM3(nn.Module):
    """M3-long with variable context and a fixed central 128-sample output."""

    def __init__(self, input_samples: int = 128) -> None:
        super().__init__()
        if input_samples not in CONTEXTS.values():
            raise ValueError(f"unsupported context length {input_samples}")
        self.input_samples = int(input_samples)
        self.core = base.build_model("M3_tcdae_long")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != CHANNELS or x.shape[2] != self.input_samples:
            raise ValueError(
                f"expected [B,{CHANNELS},{self.input_samples}], got {tuple(x.shape)}"
            )
        latent = self.core.encoder_stage3(
            self.core.encoder_stage2(self.core.encoder_stage1(x))
        )
        decoded = self.core.decoder_stage1(
            F.interpolate(latent, scale_factor=2.0, mode="linear", align_corners=False)
        )
        decoded = self.core.decoder_final(
            F.interpolate(decoded, scale_factor=2.0, mode="linear", align_corners=False)
        )
        start = (decoded.shape[-1] - WINDOW) // 2
        return decoded[..., start : start + WINDOW], latent

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "M3_tcdae_long_central_target",
            "input_samples": self.input_samples,
            "output_samples": WINDOW,
            "latent_shape": ["batch", 48, self.input_samples // 4],
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "long_skip": False,
        }


def context_arrays(
    item: a1.PreparedSubject,
    indices: Sequence[int],
    context_samples: int,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extra = (context_samples - WINDOW) // 2
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    kept: list[int] = []
    for raw_index in np.asarray(indices, dtype=np.int64):
        rec_idx = int(item.windows.record_index[raw_index])
        record = item.records[rec_idx]
        target_start = int(item.windows.start[raw_index])
        target_end = int(item.windows.end[raw_index])
        start = target_start - extra
        end = target_end + extra
        guard_start = max(0, start - a1.FOG_GUARD)
        guard_end = min(len(record.y), end + a1.FOG_GUARD)
        if start < 0 or end > len(record.y):
            continue
        if not record.valid[start:end].all() or np.any(record.y[guard_start:guard_end]):
            continue
        raw_context = record.x[start:end].astype(np.float32)
        scaled_context = (raw_context - center) / (scale + 1e-6)
        central = scaled_context[extra : extra + WINDOW]
        window_center = central.mean(axis=0, keepdims=True)
        inputs.append(scaled_context - window_center)
        targets.append(central - window_center)
        kept.append(int(raw_index))
    if not inputs:
        raise ValueError(f"{item.subject} has no valid W={context_samples} windows")
    return (
        np.ascontiguousarray(np.stack(inputs).astype(np.float32)),
        np.ascontiguousarray(np.stack(targets).astype(np.float32)),
        np.asarray(kept, dtype=np.int64),
    )


def pair_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    x = torch.from_numpy(np.ascontiguousarray(inputs.transpose(0, 2, 1))).float()
    y = torch.from_numpy(np.ascontiguousarray(targets.transpose(0, 2, 1))).float()
    return DataLoader(
        TensorDataset(x, y),
        batch_size=64,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def correlation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = predicted - predicted.mean(dim=-1, keepdim=True)
    actual = target - target.mean(dim=-1, keepdim=True)
    numerator = torch.sum(pred * actual, dim=-1)
    denominator = torch.sqrt(
        torch.sum(pred.square(), dim=-1) * torch.sum(actual.square(), dim=-1) + 1e-8
    )
    return 1.0 - torch.mean(numerator / denominator)


def structural_loss(name: str, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(predicted, target)
    if name == "L0":
        return mse
    if name == "L1":
        channel_mse = torch.mean((predicted - target).square(), dim=(0, 2))
        channel_var = torch.var(target, dim=(0, 2), unbiased=False)
        return torch.mean(channel_mse / (channel_var + 1e-6))
    corr = correlation_loss(predicted, target)
    if name == "L2":
        return 0.8 * mse + 0.2 * corr
    difference = F.mse_loss(
        predicted[..., 1:] - predicted[..., :-1],
        target[..., 1:] - target[..., :-1],
    )
    if name == "L3":
        return 0.7 * mse + 0.15 * corr + 0.15 * difference
    if name == "L4":
        return 0.7 * F.smooth_l1_loss(predicted, target) + 0.15 * corr + 0.15 * difference
    raise ValueError(f"unknown loss {name}")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    loss_name: str,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_x, batch_y in pair_loader(inputs, targets, shuffle=False, seed=0, workers=0):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predicted, _ = model(batch_x)
        total += float(structural_loss(loss_name, predicted, batch_y)) * len(batch_x)
        count += len(batch_x)
    return total / count


@torch.no_grad()
def predict_pairs(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predicted: list[np.ndarray] = []
    latent: list[np.ndarray] = []
    for batch_x, _ in pair_loader(inputs, targets, shuffle=False, seed=0, workers=0):
        reconstruction, representation = model(batch_x.to(device))
        predicted.append(reconstruction.transpose(1, 2).cpu().numpy().astype(np.float32))
        latent.append(representation.cpu().numpy().astype(np.float32))
    return np.concatenate(predicted), np.concatenate(latent)


def train_repair_model(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    calibration_inputs: np.ndarray,
    calibration_targets: np.ndarray,
    run_dir: Path,
    *,
    subject: str,
    seed: int,
    loss_name: str,
    context_name: str,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
) -> tuple[ContextM3, list[dict[str, Any]], dict[str, Any]]:
    required = (run_dir / "best_model.pt", run_dir / "last_model.pt", run_dir / "training_log.csv")
    model = ContextM3(CONTEXTS[context_name]).to(device)
    if all(path.exists() for path in required):
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        history = [dict(row) for row in read_csv(run_dir / "training_log.csv")]
        return model, history, dict(checkpoint["training"])
    run_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(seed)
    model = ContextM3(CONTEXTS[context_name]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_batches = pair_loader(
        train_inputs, train_targets, shuffle=True, seed=seed, workers=workers
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0
    last_train = math.inf
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        max_gradient = 0.0
        for batch_x, batch_y in train_batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(batch_x)
            loss = structural_loss(loss_name, predicted, batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite A1b gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
            max_gradient = max(max_gradient, float(gradient))
        last_train = total / count
        validation = evaluate_loss(
            model, calibration_inputs, calibration_targets, loss_name, device
        )
        improved = validation < best_loss - 1e-8
        if improved:
            best_loss = validation
            best_epoch = epoch
            best_state = base.clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        last_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or improved or epoch == max_epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": last_train,
                    "calibration_loss": validation,
                    "max_gradient_norm_before_clip": max_gradient,
                    "improved": improved,
                    "bad_epochs": bad_epochs,
                }
            )
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"TRAIN {subject} {loss_name}/{context_name} seed={seed} "
                f"epoch={epoch}/{max_epochs} train={last_train:.6g} "
                f"cal={validation:.6g} best={best_loss:.6g}@{best_epoch}",
                flush=True,
            )
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("repair training produced no best state")
    training = {
        "subject_id": subject,
        "seed": seed,
        "loss": loss_name,
        "context": context_name,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_calibration_loss": best_loss,
        "last_train_loss": last_train,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    base.torch_save(
        run_dir / "last_model.pt",
        {"model_state": base.clone_state(model), "training": training},
    )
    base.torch_save(
        run_dir / "best_model.pt",
        {"model_state": best_state, "training": training},
    )
    base.write_csv(run_dir / "training_log.csv", history)
    model.load_state_dict(best_state)
    return model, history, training


def fine_tune_decoder(
    model: nn.Module,
    calibration_x: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    tuned = deepcopy(model).to(device)
    for parameter in tuned.parameters():
        parameter.requires_grad = False
    decoder_final = tuned.decoder_final if hasattr(tuned, "decoder_final") else tuned.core.decoder_final
    for parameter in decoder_final.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(
        [p for p in tuned.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0
    )
    batches = a1.loader(calibration_x, shuffle=True, seed=seed, workers=0)
    history: list[dict[str, Any]] = []
    base.set_seed(seed)
    for epoch in range(1, epochs + 1):
        tuned.train()
        total = 0.0
        count = 0
        for (batch,) in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = tuned(batch)
            loss = F.mse_loss(predicted, batch)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in tuned.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        history.append({"epoch": epoch, "calibration_mse": total / count})
    return tuned, history


def plot_training_history(history: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [int(row["epoch"]) for row in history]
    train_key = "train_loss" if "train_loss" in history[0] else "calibration_mse"
    ax.plot(epochs, [float(row[train_key]) for row in history], label=train_key)
    if "calibration_loss" in history[0]:
        ax.plot(epochs, [float(row["calibration_loss"]) for row in history], label="calibration")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x0 = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y0 = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denominator = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    return float(np.sum(x0 * y0) / denominator) if denominator > 1e-12 else 0.0


def lagged_metrics(
    actual: np.ndarray, predicted: np.ndarray, maximum_lag: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    correlations = np.empty((len(actual), actual.shape[2]), dtype=np.float32)
    lags = np.empty_like(correlations, dtype=np.int16)
    for window_index in range(len(actual)):
        for channel in range(actual.shape[2]):
            scores: list[float] = []
            for lag in range(-maximum_lag, maximum_lag + 1):
                if lag < 0:
                    source = actual[window_index, -lag:, channel]
                    estimate = predicted[window_index, :lag, channel]
                elif lag > 0:
                    source = actual[window_index, :-lag, channel]
                    estimate = predicted[window_index, lag:, channel]
                else:
                    source = actual[window_index, :, channel]
                    estimate = predicted[window_index, :, channel]
                scores.append(safe_corr(source, estimate))
            best = int(np.argmax(scores))
            correlations[window_index, channel] = scores[best]
            lags[window_index, channel] = best - maximum_lag
    return correlations, lags


def band_energy(values: np.ndarray, low: float, high: float) -> np.ndarray:
    frequency = np.fft.rfftfreq(values.shape[1], d=1.0 / FS)
    mask = (frequency >= low) & (frequency <= high)
    spectrum = np.abs(np.fft.rfft(values.astype(np.float64), axis=1)) ** 2
    return np.mean(spectrum[:, mask, :], axis=1).astype(np.float32)


def dominant_frequency(values: np.ndarray) -> np.ndarray:
    frequency = np.fft.rfftfreq(values.shape[1], d=1.0 / FS)
    mask = (frequency >= 0.5) & (frequency <= 10.0)
    spectrum = np.abs(np.fft.rfft(values.astype(np.float64), axis=1)) ** 2
    selected = spectrum[:, mask, :]
    return frequency[mask][np.argmax(selected, axis=1)].astype(np.float32)


def domain_statistics(
    item: a1.PreparedSubject,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    train_raw = legacy.raw_windows(item.records, item.windows, item.train_indices)
    test_raw = legacy.raw_windows(item.records, item.windows, item.test_indices)
    train_points = unique_raw_points(item, item.train_indices)
    test_points = unique_raw_points(item, item.test_indices)
    arrays: dict[str, np.ndarray] = {
        "train_rms": np.sqrt(np.mean(train_raw.astype(np.float64) ** 2, axis=1)),
        "test_rms": np.sqrt(np.mean(test_raw.astype(np.float64) ** 2, axis=1)),
        "train_low": band_energy(train_raw, 0.5, 3.0),
        "test_low": band_energy(test_raw, 0.5, 3.0),
        "train_high": band_energy(train_raw, 3.0, 8.0),
        "test_high": band_energy(test_raw, 3.0, 8.0),
        "train_full": band_energy(train_raw, 0.5, 10.0),
        "test_full": band_energy(test_raw, 0.5, 10.0),
        "train_dominant": dominant_frequency(train_raw),
        "test_dominant": dominant_frequency(test_raw),
    }
    rows: list[dict[str, Any]] = []
    train_median = np.median(train_points, axis=0)
    test_median = np.median(test_points, axis=0)
    train_q25, train_q75 = np.percentile(train_points, [25.0, 75.0], axis=0)
    test_q25, test_q75 = np.percentile(test_points, [25.0, 75.0], axis=0)
    train_mad = np.median(np.abs(train_points - train_median), axis=0)
    test_mad = np.median(np.abs(test_points - test_median), axis=0)
    for channel, name in enumerate(item.channel_names):
        median_shift = abs(test_median[channel] - train_median[channel]) / (
            train_q75[channel] - train_q25[channel] + 1e-8
        )
        scale_shift = abs(
            math.log((test_mad[channel] + 1e-8) / (train_mad[channel] + 1e-8))
        )
        rows.append(
            {
                "subject_id": item.subject,
                "channel": name,
                "train_median": float(train_median[channel]),
                "test_median": float(test_median[channel]),
                "train_iqr": float(train_q75[channel] - train_q25[channel]),
                "test_iqr": float(test_q75[channel] - test_q25[channel]),
                "train_mad": float(train_mad[channel]),
                "test_mad": float(test_mad[channel]),
                "train_rms": float(np.median(arrays["train_rms"][:, channel])),
                "test_rms": float(np.median(arrays["test_rms"][:, channel])),
                "train_std": float(np.std(train_points[:, channel])),
                "test_std": float(np.std(test_points[:, channel])),
                "train_dominant_frequency": float(
                    np.median(arrays["train_dominant"][:, channel])
                ),
                "test_dominant_frequency": float(
                    np.median(arrays["test_dominant"][:, channel])
                ),
                "train_energy_0p5_3hz": float(np.median(arrays["train_low"][:, channel])),
                "test_energy_0p5_3hz": float(np.median(arrays["test_low"][:, channel])),
                "train_energy_3_8hz": float(np.median(arrays["train_high"][:, channel])),
                "test_energy_3_8hz": float(np.median(arrays["test_high"][:, channel])),
                "train_energy_0p5_10hz": float(np.median(arrays["train_full"][:, channel])),
                "test_energy_0p5_10hz": float(np.median(arrays["test_full"][:, channel])),
                "median_shift": float(median_shift),
                "scale_shift": float(scale_shift),
            }
        )
    return rows, arrays


def classify_failure(
    ordinary: np.ndarray,
    lagged: np.ndarray,
    lags: np.ndarray,
    amplitude: np.ndarray,
    spectral_full: np.ndarray,
    domain_rows: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    channel_corr = np.median(ordinary, axis=0)
    flags = {
        "Amplitude shift": bool(np.median(np.abs(np.log(np.clip(amplitude, 1e-6, None)))) > 0.35),
        "Phase shift": bool(np.median(lagged - ordinary) >= 0.12 and np.median(np.abs(lags)) >= 2),
        "Spectral shift": bool(np.median(spectral_full) >= 0.90),
        "Channel-specific failure": bool(np.median(channel_corr) - np.min(channel_corr) >= 0.20),
        "Global record shift": bool(
            sum(
                float(row["median_shift"]) >= 0.50 or float(row["scale_shift"]) >= 0.50
                for row in domain_rows
            )
            >= 5
        ),
    }
    active = [name for name, enabled in flags.items() if enabled]
    if len(active) >= 2:
        return "Mixed", flags
    return (active[0] if active else "Low waveform coverage"), flags


def diagnostic_plots(
    subject: str,
    channel_names: Sequence[str],
    actual: np.ndarray,
    predicted: np.ndarray,
    ordinary: np.ndarray,
    nrmse: np.ndarray,
    amplitude: np.ndarray,
    lagged: np.ndarray,
    lags: np.ndarray,
    frequency_arrays: dict[str, np.ndarray],
    domain_rows: Sequence[dict[str, Any]],
    domain_arrays: dict[str, np.ndarray],
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(channel_names))

    def bar_plot(path: str, series: Sequence[tuple[str, np.ndarray]], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        width = 0.8 / len(series)
        for index, (label, values) in enumerate(series):
            ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=label)
        ax.set_xticks(x, channel_names, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(figures / path, dpi=150)
        plt.close(fig)

    bar_plot(
        "channel_pearson_comparison.png",
        (("ordinary", np.median(ordinary, axis=0)), ("lagged", np.median(lagged, axis=0))),
        "Pearson",
    )
    bar_plot("channel_nrmse_comparison.png", (("NRMSE", np.median(nrmse, axis=0)),), "NRMSE")
    bar_plot(
        "channel_amplitude_ratio.png",
        (("amplitude", np.median(amplitude, axis=0)),),
        "std(pred)/std(actual)",
    )
    bar_plot(
        "channel_frequency_error.png",
        (
            ("0.5-3 Hz", np.median(frequency_arrays["low"], axis=0)),
            ("3-8 Hz", np.median(frequency_arrays["high"], axis=0)),
            ("0.5-10 Hz", np.median(frequency_arrays["full"], axis=0)),
        ),
        "spectral NRMSE",
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ordinary.ravel(), lagged.ravel(), s=6, alpha=0.25)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8)
    ax.set(xlabel="ordinary Pearson", ylabel="maximum-lag Pearson", xlim=(-1, 1), ylim=(-1, 1))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "pearson_vs_lagged_pearson.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lags.ravel(), bins=np.arange(-16.5, 17.5, 1), color="#4472c4")
    ax.set(xlabel="best lag (samples)", ylabel="window-channel count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "best_lag_distribution.png", dpi=150)
    plt.close(fig)

    bar_plot(
        "train_test_channel_median_shift.png",
        (("robust median shift", np.asarray([float(row["median_shift"]) for row in domain_rows])),),
        "normalized shift",
    )
    bar_plot(
        "train_test_channel_scale_shift.png",
        (("MAD log-ratio", np.asarray([float(row["scale_shift"]) for row in domain_rows])),),
        "absolute log scale ratio",
    )

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for channel, ax in enumerate(axes.ravel()):
        ax.boxplot(
            [domain_arrays["train_rms"][:, channel], domain_arrays["test_rms"][:, channel]],
            tick_labels=("train", "test"),
            showfliers=False,
        )
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"{subject} train/test RMS")
    fig.tight_layout()
    fig.savefig(figures / "train_test_rms_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for channel, ax in enumerate(axes.ravel()):
        ax.boxplot(
            [domain_arrays["train_full"][:, channel], domain_arrays["test_full"][:, channel]],
            tick_labels=("train", "test"),
            showfliers=False,
        )
        ax.set_yscale("log")
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"{subject} 0.5-10 Hz energy")
    fig.tight_layout()
    fig.savefig(figures / "train_test_spectral_distribution.png", dpi=150)
    plt.close(fig)

    window_nrmse = np.median(nrmse, axis=1)
    worst = int(np.argmax(window_nrmse))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    t = np.arange(WINDOW) / FS
    for channel, ax in enumerate(axes.ravel()):
        ax.plot(t, actual[worst, :, channel], label="target", linewidth=0.9)
        ax.plot(t, predicted[worst, :, channel], "--", label="reconstruction", linewidth=0.9)
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{subject} worst clean Non-FoG window")
    fig.tight_layout()
    fig.savefig(figures / "worst_window_waveform.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    heat = np.abs(actual[worst] - predicted[worst]).T
    image = ax.imshow(heat, aspect="auto", origin="lower", cmap="magma")
    ax.set_yticks(np.arange(CHANNELS), channel_names)
    ax.set_xlabel("sample")
    fig.colorbar(image, ax=ax, label="absolute residual")
    fig.tight_layout()
    fig.savefig(figures / "worst_window_residual_heatmap.png", dpi=150)
    plt.close(fig)


def run_diagnostics(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    prior_results: Sequence[dict[str, Any]],
    *,
    skip_figures: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage = root / "A1b0_diagnostics"
    summary_path = root / "tables" / "A1b0_failure_diagnosis.csv"
    if summary_path.exists():
        summary = [dict(row) for row in read_csv(summary_path)]
        channels = [dict(row) for row in read_csv(root / "tables" / "A1b0_channel_diagnostics.csv")]
        domain = [dict(row) for row in read_csv(root / "tables" / "A1b0_train_test_domain_shift.csv")]
        return summary, channels, domain
    summaries: list[dict[str, Any]] = []
    channel_output: list[dict[str, Any]] = []
    domain_output: list[dict[str, Any]] = []
    failure_flags: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        rows = sorted(
            [row for row in prior_results if row["subject_id"] == subject],
            key=lambda row: float(row["median_nrmse"]),
        )
        representative = rows[len(rows) // 2]
        with np.load(Path(representative["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
            actual = np.asarray(payload["target"])
            predicted = np.asarray(payload["reconstruction"])
        arrays = base.metric_arrays(actual, predicted)
        lagged, lags = lagged_metrics(actual, predicted)
        _, low = a1.spectral_error(actual, predicted, 0.5, 3.0)
        _, high = a1.spectral_error(actual, predicted, 3.0, 8.0)
        _, full = a1.spectral_error(actual, predicted, 0.5, 10.0)
        difference_error = np.mean(
            (
                np.diff(actual, axis=1) - np.diff(predicted, axis=1)
            )
            ** 2,
            axis=1,
        )
        domain_rows, domain_arrays = domain_statistics(prepared[subject])
        failure_type, flags = classify_failure(
            arrays["correlation"],
            lagged,
            lags,
            arrays["amplitude_ratio"],
            full,
            domain_rows,
        )
        channel_medians = np.median(arrays["correlation"], axis=0)
        worst_channel = int(np.argmin(channel_medians))
        summaries.append(
            {
                "subject_id": subject,
                "representative_seed": int(representative["seed"]),
                "original_a1_status": representative["pass_status"],
                "worst_channel": prepared[subject].channel_names[worst_channel],
                "worst_channel_pearson": float(channel_medians[worst_channel]),
                "worst_channel_lagged_pearson": float(np.median(lagged[:, worst_channel])),
                "worst_channel_best_lag": float(np.median(lags[:, worst_channel])),
                "median_pearson": float(np.median(arrays["correlation"])),
                "median_lagged_pearson": float(np.median(lagged)),
                "median_abs_best_lag": float(np.median(np.abs(lags))),
                "median_nrmse": float(np.median(arrays["nrmse"])),
                "median_amplitude_ratio": float(np.median(arrays["amplitude_ratio"])),
                "median_shift": float(np.median([row["median_shift"] for row in domain_rows])),
                "scale_shift": float(np.median([row["scale_shift"] for row in domain_rows])),
                "failure_type": failure_type,
            }
        )
        for channel, name in enumerate(prepared[subject].channel_names):
            channel_output.append(
                {
                    "subject_id": subject,
                    "channel": name,
                    "mse": float(
                        np.median(np.mean((actual - predicted) ** 2, axis=1)[:, channel])
                    ),
                    "pearson": float(np.median(arrays["correlation"][:, channel])),
                    "lagged_pearson": float(np.median(lagged[:, channel])),
                    "best_lag": float(np.median(lags[:, channel])),
                    "median_abs_best_lag": float(np.median(np.abs(lags[:, channel]))),
                    "nrmse": float(np.median(arrays["nrmse"][:, channel])),
                    "amplitude_ratio": float(np.median(arrays["amplitude_ratio"][:, channel])),
                    "first_difference_mse": float(np.median(difference_error[:, channel])),
                    "spectral_nrmse_0p5_3hz": float(np.median(low[:, channel])),
                    "spectral_nrmse_3_8hz": float(np.median(high[:, channel])),
                    "spectral_nrmse_0p5_10hz": float(np.median(full[:, channel])),
                }
            )
        domain_output.extend(domain_rows)
        failure_flags.append({"subject_id": subject, **flags, "Mixed": failure_type == "Mixed"})
        base.write_csv(stage / subject / "channel_metrics.csv", channel_output[-CHANNELS:])
        base.write_csv(stage / subject / "domain_shift.csv", domain_rows)
        base.write_json(stage / subject / "diagnostic_summary.json", summaries[-1])
        if not skip_figures:
            diagnostic_plots(
                subject,
                prepared[subject].channel_names,
                actual,
                predicted,
                arrays["correlation"],
                arrays["nrmse"],
                arrays["amplitude_ratio"],
                lagged,
                lags,
                {"low": low, "high": high, "full": full},
                domain_rows,
                domain_arrays,
                stage / subject / "figures",
            )
        print(
            f"A1b0 {subject} type={failure_type} corr={np.median(arrays['correlation']):.3f} "
            f"lagged={np.median(lagged):.3f} |lag|={np.median(np.abs(lags)):.1f}",
            flush=True,
        )
    base.write_csv(summary_path, summaries)
    base.write_csv(root / "tables" / "A1b0_channel_diagnostics.csv", channel_output)
    base.write_csv(root / "tables" / "A1b0_train_test_domain_shift.csv", domain_output)
    if not skip_figures:
        names = ["Amplitude shift", "Phase shift", "Spectral shift", "Channel-specific failure", "Global record shift", "Mixed"]
        matrix = np.asarray(
            [[float(bool(row[name])) for name in names] for row in failure_flags]
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(SUBJECTS)), SUBJECTS)
        for y in range(len(SUBJECTS)):
            for x in range(len(names)):
                ax.text(x, y, "YES" if matrix[y, x] else "", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="diagnostic flag")
        fig.tight_layout()
        output = root / "figures" / "all_subject_failure_type_matrix.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
    return summaries, channel_output, domain_output


def evaluation_artifacts(
    item: a1.PreparedSubject,
    evaluation_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    template: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, arrays = a1.reconstruction_metrics(actual, predicted, template)
    lagged, lags = lagged_metrics(actual, predicted)
    metrics.update(
        {
            "median_lagged_pearson": float(np.median(lagged)),
            "median_best_lag": float(np.median(lags)),
            "median_abs_best_lag": float(np.median(np.abs(lags))),
        }
    )
    metrics["strict_pass"] = a1.a1_pass(metrics)
    metrics["waveform_collapse"] = a1.waveform_collapse(metrics)
    metadata = a1.window_metadata(
        item.subject,
        item.records,
        item.windows,
        evaluation_indices,
        "evaluation_clean_nonfog",
    )
    window_rows: list[dict[str, Any]] = []
    for index, row in enumerate(metadata):
        window_rows.append(
            {
                **row,
                "mse": float(arrays["window_mse"][index]),
                "zero_improvement_pct": float(arrays["window_improvement_pct"][index]),
                "pearson_median": float(arrays["window_corr"][index]),
                "lagged_pearson_median": float(np.median(lagged[index])),
                "best_lag_median": float(np.median(lags[index])),
                "nrmse_median": float(arrays["window_nrmse"][index]),
                "amplitude_ratio_median": float(arrays["window_amplitude_ratio"][index]),
            }
        )
    channel_rows: list[dict[str, Any]] = []
    for channel, name in enumerate(item.channel_names):
        channel_rows.append(
            {
                "subject_id": item.subject,
                "channel": name,
                "mse": float(np.mean((actual[:, :, channel] - predicted[:, :, channel]) ** 2)),
                "pearson": float(np.median(arrays["correlation"][:, channel])),
                "lagged_pearson": float(np.median(lagged[:, channel])),
                "best_lag": float(np.median(lags[:, channel])),
                "nrmse": float(np.median(arrays["nrmse"][:, channel])),
                "amplitude_ratio": float(np.median(arrays["amplitude_ratio"][:, channel])),
            }
        )
    return metrics, {**arrays, "lagged": lagged, "lags": lags}, window_rows, channel_rows


def save_evaluation_run(
    run_dir: Path,
    *,
    config: dict[str, Any],
    plan: CalibrationPlan,
    item: a1.PreparedSubject,
    model: nn.Module,
    actual: np.ndarray,
    predicted: np.ndarray,
    latent: np.ndarray,
    result: dict[str, Any],
    window_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    training_log: Sequence[dict[str, Any]],
    save_model: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base.write_json(run_dir / "config.json", config)
    split_rows: list[dict[str, Any]] = []
    for split, indices in (
        ("train_clean_nonfog", item.train_indices),
        ("early_stopping_clean_nonfog", item.calibration_indices),
        ("deployment_calibration_clean_nonfog", plan.calibration_indices),
        ("final_evaluation_clean_nonfog", plan.evaluation_indices),
    ):
        split_rows.extend(a1.window_metadata(item.subject, item.records, item.windows, indices, split))
    base.write_csv(run_dir / "split_manifest.csv", split_rows)
    base.write_csv(run_dir / "calibration_manifest.csv", plan.manifest_rows)
    base.write_csv(run_dir / "training_log.csv", list(training_log) or [{"epoch": 0, "status": "no retraining"}])
    base.write_csv(run_dir / "window_metrics.csv", window_rows)
    base.write_csv(run_dir / "channel_metrics.csv", channel_rows)
    base.write_json(run_dir / "run_metrics.json", result)
    if save_model:
        state = base.clone_state(model)
        checkpoint = {
            "experiment": EXPERIMENT,
            "config": config,
            "model_state": state,
        }
        base.torch_save(run_dir / "best_model.pt", checkpoint)
        base.torch_save(run_dir / "last_model.pt", checkpoint)
    legacy.save_npz(
        run_dir / "predictions.npz",
        target=actual,
        reconstruction=predicted,
        latent=latent,
    )


def evaluate_calibration_setting(
    item: a1.PreparedSubject,
    model: nn.Module,
    plan: CalibrationPlan,
    method: str,
    device: torch.device,
    *,
    c4_epochs: int = 0,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_x, evaluation_x = calibration_arrays(item, plan, method)
    training_log: list[dict[str, Any]] = []
    evaluated_model = model
    if c4_epochs:
        evaluated_model, training_log = fine_tune_decoder(
            model,
            calibration_x,
            epochs=c4_epochs,
            seed=seed,
            device=device,
        )
    predicted, latent = a1.predict_model(evaluated_model, evaluation_x, device)
    template = np.repeat(
        np.mean(item.train_x, axis=0, keepdims=True).astype(np.float32),
        len(evaluation_x),
        axis=0,
    )
    metrics, arrays, window_rows, channel_rows = evaluation_artifacts(
        item, plan.evaluation_indices, evaluation_x, predicted, template
    )
    payload = {
        "model": evaluated_model,
        "calibration_x": calibration_x,
        "actual": evaluation_x,
        "predicted": predicted,
        "latent": latent,
        "arrays": arrays,
        "window_rows": window_rows,
        "channel_rows": channel_rows,
        "training_log": training_log,
    }
    return metrics, payload


def setting_rank(rows: Sequence[dict[str, Any]]) -> tuple[float, ...]:
    return (
        float(sum(as_bool(row["strict_pass"]) for row in rows)),
        float(np.median([float(row["median_corr"]) for row in rows])),
        -float(np.median([float(row["median_nrmse"]) for row in rows])),
        -float(np.median([float(row["nrmse_p90"]) for row in rows])),
        -float(np.median([float(row["median_abs_best_lag"]) for row in rows])),
    )


def calibration_summary_plots(
    rows: Sequence[dict[str, Any]], figures: Path, *, prefix: str = ""
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    methods = sorted(set(str(row["method"]) for row in rows))
    subjects = sorted(set(str(row["subject_id"]) for row in rows))
    durations = sorted(set(int(row["requested_calibration_seconds"]) for row in rows))
    seed = min(int(row["seed"]) for row in rows)
    screen = [row for row in rows if int(row["seed"]) == seed]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        values = []
        for duration in durations:
            selected = [
                float(row["median_nrmse"])
                for row in screen
                if row["method"] == method
                and int(row["requested_calibration_seconds"]) == duration
            ]
            values.append(float(np.median(selected)) if selected else np.nan)
        ax.plot(durations, values, marker="o", label=method)
    ax.set(xlabel="calibration seconds", ylabel="median NRMSE")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}calibration_duration_curve.png", dpi=150)
    plt.close(fig)

    best_by_method = []
    for method in methods:
        selected = [row for row in screen if row["method"] == method]
        best_by_method.append(
            (
                method,
                float(np.median([float(row["median_corr"]) for row in selected])),
                float(np.median([float(row["median_nrmse"]) for row in selected])),
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([row[0] for row in best_by_method], [row[1] for row in best_by_method])
    axes[0].set_ylabel("median Pearson")
    axes[1].bar([row[0] for row in best_by_method], [row[2] for row in best_by_method])
    axes[1].set_ylabel("median NRMSE")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}calibration_method_comparison.png", dpi=150)
    plt.close(fig)

    for metric, filename, ylabel in (
        ("median_corr", "before_after_pearson.png", "Pearson"),
        ("median_nrmse", "before_after_nrmse.png", "NRMSE"),
        ("nrmse_p90", "before_after_p90.png", "NRMSE P90"),
    ):
        candidate_rows = [row for row in rows if row.get("selected_candidate")]
        if not candidate_rows:
            candidate_rows = screen
        candidate = {str(row["subject_id"]): float(row[metric]) for row in candidate_rows}
        baseline = {}
        for subject in subjects:
            base_rows = [row for row in screen if row["subject_id"] == subject and row["method"] == "C0"]
            if base_rows:
                baseline[subject] = float(base_rows[0][metric])
        common = [subject for subject in subjects if subject in candidate and subject in baseline]
        x = np.arange(len(common))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.18, [baseline[s] for s in common], 0.36, label="C0")
        ax.bar(x + 0.18, [candidate[s] for s in common], 0.36, label="selected")
        ax.set_xticks(x, common)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{prefix}{filename}", dpi=150)
        plt.close(fig)

    selected = [row for row in rows if row.get("selected_candidate")]
    if selected:
        matrix = np.zeros((len(subjects), len(SEEDS)), dtype=float)
        for y, subject in enumerate(subjects):
            for x, seed_value in enumerate(SEEDS):
                matches = [
                    row for row in selected
                    if row["subject_id"] == subject and int(row["seed"]) == seed_value
                ]
                matrix[y, x] = float(bool(matches and matches[0]["strict_pass"]))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(SEEDS)), SEEDS)
        ax.set_yticks(np.arange(len(subjects)), subjects)
        for y in range(len(subjects)):
            for x in range(len(SEEDS)):
                ax.text(x, y, "PASS" if matrix[y, x] else "FAIL", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(figures / f"{prefix}subject_pass_matrix_calibration.png", dpi=150)
        plt.close(fig)


def run_calibration(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    prior_lookup: dict[tuple[str, int], dict[str, Any]],
    *,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    stage = root / "A1b1_record_calibration"
    screen_path = root / "tables" / "A1b1_calibration_screen.csv"
    screen_rows: list[dict[str, Any]] = [] if overwrite or not screen_path.exists() else [dict(row) for row in read_csv(screen_path)]
    if not screen_rows:
        for subject in FAILED_SUBJECTS:
            item = prepared[subject]
            model = load_original_model(prior_lookup, subject, SEEDS[0], device)
            for duration in CALIBRATION_DURATIONS:
                plan = calibration_plan(item, duration)
                for method in CALIBRATION_METHODS:
                    run_dir = stage / method / subject / f"duration{duration}s" / f"seed{SEEDS[0]}"
                    if (run_dir / "run_metrics.json").exists() and not overwrite:
                        result = read_json(run_dir / "run_metrics.json")
                        screen_rows.append(result)
                        print(f"RESUME A1b1 screen {subject} {method}/{duration}s", flush=True)
                        continue
                    metrics, payload = evaluate_calibration_setting(
                        item, model, plan, method, device, seed=SEEDS[0]
                    )
                    config = {
                        "stage": "A1b1_calibration_screen",
                        "subject_id": subject,
                        "seed": SEEDS[0],
                        "method": method,
                        "requested_calibration_seconds": duration,
                        "decoder_fine_tune_epochs": 0,
                        "test_fog_used": False,
                    }
                    result = {
                        **config,
                        "actual_calibration_seconds": plan.actual_seconds,
                        "evaluation_windows": len(plan.evaluation_indices),
                        **metrics,
                        "run_dir": str(run_dir.resolve()),
                    }
                    save_evaluation_run(
                        run_dir,
                        config=config,
                        plan=plan,
                        item=item,
                        model=payload["model"],
                        actual=payload["actual"],
                        predicted=payload["predicted"],
                        latent=payload["latent"],
                        result=result,
                        window_rows=payload["window_rows"],
                        channel_rows=payload["channel_rows"],
                        training_log=payload["training_log"],
                    )
                    screen_rows.append(result)
                    print(
                        f"A1b1 screen {subject} {method}/{duration}s "
                        f"corr={metrics['median_corr']:.3f} nrmse={metrics['median_nrmse']:.3f} "
                        f"p90={metrics['nrmse_p90']:.3f} {'PASS' if metrics['strict_pass'] else 'FAIL'}",
                        flush=True,
                    )
        base.write_csv(screen_path, screen_rows)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in screen_rows:
        if str(row["method"]) == "C0":
            continue
        grouped.setdefault((str(row["method"]), int(row["requested_calibration_seconds"])), []).append(row)
    best_base_key = max(grouped, key=lambda key: setting_rank(grouped[key]))
    best_base_rows = grouped[best_base_key]
    baseline_by_subject = {
        (str(row["subject_id"]), int(row["requested_calibration_seconds"])): row
        for row in screen_rows
        if str(row["method"]) == "C0"
    }
    improved = sum(
        (
            as_bool(row["strict_pass"])
            and not as_bool(baseline_by_subject[(str(row["subject_id"]), int(row["requested_calibration_seconds"]))]["strict_pass"])
        )
        or (
            float(row["median_corr"])
            - float(baseline_by_subject[(str(row["subject_id"]), int(row["requested_calibration_seconds"]))]["median_corr"])
            >= 0.03
            and float(baseline_by_subject[(str(row["subject_id"]), int(row["requested_calibration_seconds"]))]["median_nrmse"])
            - float(row["median_nrmse"])
            >= 0.03
        )
        for row in best_base_rows
    )
    c4_needed = improved < 2
    c4_rows: list[dict[str, Any]] = []
    c4_path = root / "tables" / "A1b1_c4_screen.csv"
    if c4_needed:
        c4_rows = [] if overwrite or not c4_path.exists() else [dict(row) for row in read_csv(c4_path)]
        if not c4_rows:
            method, duration = best_base_key
            for subject in FAILED_SUBJECTS:
                item = prepared[subject]
                plan = calibration_plan(item, duration)
                model = load_original_model(prior_lookup, subject, SEEDS[0], device)
                for epochs in C4_EPOCHS:
                    run_dir = stage / "C4" / subject / f"epochs{epochs}" / f"seed{SEEDS[0]}"
                    if (run_dir / "run_metrics.json").exists() and not overwrite:
                        c4_rows.append(read_json(run_dir / "run_metrics.json"))
                        print(f"RESUME A1b1 C4 {subject} epochs={epochs}", flush=True)
                        continue
                    metrics, payload = evaluate_calibration_setting(
                        item,
                        model,
                        plan,
                        method,
                        device,
                        c4_epochs=epochs,
                        seed=SEEDS[0],
                    )
                    config = {
                        "stage": "A1b1_C4_screen",
                        "subject_id": subject,
                        "seed": SEEDS[0],
                        "method": "C4",
                        "base_preprocessing_method": method,
                        "requested_calibration_seconds": duration,
                        "decoder_fine_tune_epochs": epochs,
                        "test_fog_used": False,
                    }
                    result = {
                        **config,
                        "actual_calibration_seconds": plan.actual_seconds,
                        "evaluation_windows": len(plan.evaluation_indices),
                        **metrics,
                        "run_dir": str(run_dir.resolve()),
                    }
                    save_evaluation_run(
                        run_dir,
                        config=config,
                        plan=plan,
                        item=item,
                        model=payload["model"],
                        actual=payload["actual"],
                        predicted=payload["predicted"],
                        latent=payload["latent"],
                        result=result,
                        window_rows=payload["window_rows"],
                        channel_rows=payload["channel_rows"],
                        training_log=payload["training_log"],
                    )
                    c4_rows.append(result)
                    print(
                        f"A1b1 C4 {subject} epochs={epochs} corr={metrics['median_corr']:.3f} "
                        f"nrmse={metrics['median_nrmse']:.3f} {'PASS' if metrics['strict_pass'] else 'FAIL'}",
                        flush=True,
                    )
            base.write_csv(c4_path, c4_rows)

    selected = {
        "method": best_base_key[0],
        "duration_seconds": int(best_base_key[1]),
        "c4_epochs": 0,
        "screen_rank": setting_rank(best_base_rows),
        "screen_improved_failed_subjects": improved,
    }
    if c4_rows:
        c4_grouped = {
            epochs: [row for row in c4_rows if int(row["decoder_fine_tune_epochs"]) == epochs]
            for epochs in C4_EPOCHS
        }
        best_epochs = max(c4_grouped, key=lambda epochs: setting_rank(c4_grouped[epochs]))
        if setting_rank(c4_grouped[best_epochs]) > setting_rank(best_base_rows):
            selected["c4_epochs"] = int(best_epochs)
            selected["screen_rank"] = setting_rank(c4_grouped[best_epochs])

    if improved < 2 and int(selected["c4_epochs"]) == 0:
        selected.update(
            {
                "method": "C0",
                "duration_seconds": 0,
                "selection_reason": (
                    "C1-C3 did not clearly improve at least 2/4 failed subjects and C4 "
                    "did not outrank the best non-fine-tuned setting; retain no record calibration."
                ),
            }
        )

    expansion_path = root / "tables" / "A1b1_calibration_expansion.csv"
    expansion_rows = [] if overwrite or not expansion_path.exists() else [dict(row) for row in read_csv(expansion_path)]
    if expansion_rows and (
        str(expansion_rows[0]["method"]) != str(selected["method"])
        or int(expansion_rows[0]["requested_calibration_seconds"]) != int(selected["duration_seconds"])
        or int(expansion_rows[0]["decoder_fine_tune_epochs"]) != int(selected["c4_epochs"])
    ):
        expansion_rows = []
    elif expansion_rows:
        expansion_rows = [
            read_json(Path(str(row["run_dir"])) / "run_metrics.json")
            for row in expansion_rows
        ]
    if not expansion_rows:
        for subject in SUBJECTS:
            item = prepared[subject]
            plan = calibration_plan(item, int(selected["duration_seconds"]))
            for seed in SEEDS:
                model = load_original_model(prior_lookup, subject, seed, device)
                metrics, payload = evaluate_calibration_setting(
                    item,
                    model,
                    plan,
                    str(selected["method"]),
                    device,
                    c4_epochs=int(selected["c4_epochs"]),
                    seed=seed,
                )
                run_dir = stage / "selected" / subject / f"seed{seed}"
                config = {
                    "stage": "A1b1_selected_expansion",
                    "subject_id": subject,
                    "seed": seed,
                    "method": str(selected["method"]),
                    "requested_calibration_seconds": int(selected["duration_seconds"]),
                    "decoder_fine_tune_epochs": int(selected["c4_epochs"]),
                    "test_fog_used": False,
                }
                result = {
                    **config,
                    "actual_calibration_seconds": plan.actual_seconds,
                    "evaluation_windows": len(plan.evaluation_indices),
                    **metrics,
                    "selected_candidate": True,
                    "run_dir": str(run_dir.resolve()),
                }
                save_evaluation_run(
                    run_dir,
                    config=config,
                    plan=plan,
                    item=item,
                    model=payload["model"],
                    actual=payload["actual"],
                    predicted=payload["predicted"],
                    latent=payload["latent"],
                    result=result,
                    window_rows=payload["window_rows"],
                    channel_rows=payload["channel_rows"],
                    training_log=payload["training_log"],
                )
                expansion_rows.append(result)
                print(
                    f"A1b1 selected {subject} seed={seed} corr={metrics['median_corr']:.3f} "
                    f"nrmse={metrics['median_nrmse']:.3f} {'PASS' if metrics['strict_pass'] else 'FAIL'}",
                    flush=True,
                )
        base.write_csv(expansion_path, expansion_rows)

    gate_rows = []
    for row in expansion_rows:
        gate_rows.append(
            {
                **row,
                "split_type": prepared[str(row["subject_id"])].disclosure["split_type"],
                "test_record_or_block": prepared[str(row["subject_id"])].disclosure.get(
                    "test_record", prepared[str(row["subject_id"])].disclosure.get("record_id")
                ),
            }
        )
    gate = a1.evaluate_a1_gate(gate_rows)
    selected.update(
        {
            "expansion_gate": gate,
            "advance_directly_to_retest": bool(gate["advance_to_A2"]),
        }
    )
    base.write_json(root / "reports" / "A1b1_selected_calibration.json", selected)
    if not skip_figures:
        calibration_summary_plots(
            [*screen_rows, *c4_rows, *expansion_rows], stage / "figures"
        )
        # The remaining two requested calibration views are explicit audit plots.
        chosen_rows = [row for row in expansion_rows if int(row["seed"]) == SEEDS[0]]
        names = [str(row["subject_id"]) for row in chosen_rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(names, [float(row["median_amplitude_ratio"]) for row in chosen_rows])
        ax.axhspan(0.75, 1.25, alpha=0.15, color="green")
        ax.set_ylabel("amplitude ratio after calibration")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(stage / "figures" / "channel_shift_before_after.png", dpi=150)
        plt.close(fig)
        representative = chosen_rows[0]
        with np.load(Path(representative["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
            a1.plot_best_median_worst(
                payload["target"],
                payload["reconstruction"],
                stage / "figures" / "waveform_before_after_calibration.png",
                prepared[str(representative["subject_id"])].channel_names,
            )
    return selected, expansion_rows, gate


def fine_tune_decoder_pairs(
    model: nn.Module,
    calibration_inputs: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    tuned = deepcopy(model).to(device)
    for parameter in tuned.parameters():
        parameter.requires_grad = False
    decoder_final = tuned.decoder_final if hasattr(tuned, "decoder_final") else tuned.core.decoder_final
    for parameter in decoder_final.parameters():
        parameter.requires_grad = True
    parameters = [parameter for parameter in tuned.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=0.0)
    batches = pair_loader(
        calibration_inputs,
        calibration_targets,
        shuffle=True,
        seed=seed,
        workers=0,
    )
    base.set_seed(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        tuned.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = tuned(batch_x)
            loss = F.mse_loss(predicted, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        history.append({"epoch": epoch, "calibration_mse": total / count})
    return tuned, history


def plan_with_evaluation(plan: CalibrationPlan, indices: np.ndarray) -> CalibrationPlan:
    return CalibrationPlan(
        subject=plan.subject,
        requested_seconds=plan.requested_seconds,
        actual_seconds=plan.actual_seconds,
        calibration_indices=plan.calibration_indices,
        evaluation_indices=np.asarray(indices, dtype=np.int64),
        calibration_values=plan.calibration_values,
        calibration_median=plan.calibration_median,
        calibration_scale=plan.calibration_scale,
        manifest_rows=plan.manifest_rows,
    )


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "pass")


def repair_rank(rows: Sequence[dict[str, Any]]) -> tuple[float, ...]:
    """Frozen repair selection priority from the template."""
    return (
        float(np.median([float(row["median_corr"]) for row in rows])),
        -float(np.median([float(row["median_nrmse"]) for row in rows])),
        -float(np.median([float(row["nrmse_p90"]) for row in rows])),
        -float(
            np.median(
                [abs(float(row["median_amplitude_ratio"]) - 1.0) for row in rows]
            )
        ),
        float(sum(as_bool(row["strict_pass"]) for row in rows)),
    )


def run_trained_setting(
    root: Path,
    stage_name: str,
    item: a1.PreparedSubject,
    plan: CalibrationPlan,
    prior_lookup: dict[tuple[str, int], dict[str, Any]],
    selected_calibration: dict[str, Any],
    *,
    loss_name: str,
    context_name: str,
    seed: int,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    run_dir = root / stage_name / loss_name / context_name / item.subject / f"seed{seed}"
    metrics_path = run_dir / "run_metrics.json"
    if metrics_path.exists() and not overwrite:
        return read_json(metrics_path)
    context_samples = CONTEXTS[context_name]
    train_inputs, train_targets, train_kept = context_arrays(
        item,
        item.train_indices,
        context_samples,
        center=item.scaler.median,
        scale=item.scaler.iqr,
    )
    calibration_inputs, calibration_targets, calibration_kept = context_arrays(
        item,
        item.calibration_indices,
        context_samples,
        center=item.scaler.median,
        scale=item.scaler.iqr,
    )
    deployment_center, deployment_scale = method_center_scale(
        item, plan, str(selected_calibration["method"])
    )
    if len(plan.calibration_indices):
        deployment_inputs, deployment_targets, deployment_kept = context_arrays(
            item,
            plan.calibration_indices,
            context_samples,
            center=deployment_center,
            scale=deployment_scale,
        )
    else:
        deployment_inputs = np.empty((0, context_samples, CHANNELS), dtype=np.float32)
        deployment_targets = np.empty((0, WINDOW, CHANNELS), dtype=np.float32)
        deployment_kept = np.empty(0, dtype=np.int64)
    evaluation_inputs, evaluation_targets, evaluation_kept = context_arrays(
        item,
        plan.evaluation_indices,
        context_samples,
        center=deployment_center,
        scale=deployment_scale,
    )
    training: dict[str, Any]
    if loss_name == "L0" and context_name == "W0":
        original = load_original_model(prior_lookup, item.subject, seed, device)
        model = ContextM3(WINDOW).to(device)
        model.core.load_state_dict(original.state_dict())
        history = [{"epoch": 0, "status": "reused frozen original A1 MSE checkpoint"}]
        training = {
            "source": "original_A1_checkpoint",
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": base.clone_state(model),
            "training": training,
        }
        base.torch_save(run_dir / "best_model.pt", checkpoint)
        base.torch_save(run_dir / "last_model.pt", checkpoint)
    else:
        model, history, training = train_repair_model(
            train_inputs,
            train_targets,
            calibration_inputs,
            calibration_targets,
            run_dir,
            subject=item.subject,
            seed=seed,
            loss_name=loss_name,
            context_name=context_name,
            max_epochs=max_epochs,
            patience=patience,
            workers=workers,
            device=device,
        )
    c4_history: list[dict[str, Any]] = []
    if int(selected_calibration["c4_epochs"]):
        model, c4_history = fine_tune_decoder_pairs(
            model,
            deployment_inputs,
            deployment_targets,
            epochs=int(selected_calibration["c4_epochs"]),
            seed=seed,
            device=device,
        )
        tuned_checkpoint = {
            "model_state": base.clone_state(model),
            "training": training,
            "deployment_fine_tune": c4_history,
        }
        base.torch_save(run_dir / "best_model.pt", tuned_checkpoint)
    started = time.perf_counter()
    predicted, latent = predict_pairs(model, evaluation_inputs, evaluation_targets, device)
    inference_ms = 1000.0 * (time.perf_counter() - started) / max(len(evaluation_inputs), 1)
    template = np.repeat(
        np.mean(item.train_x, axis=0, keepdims=True).astype(np.float32),
        len(evaluation_targets),
        axis=0,
    )
    metrics, _, window_rows, channel_rows = evaluation_artifacts(
        item,
        evaluation_kept,
        evaluation_targets,
        predicted,
        template,
    )
    effective_plan = plan_with_evaluation(plan, evaluation_kept)
    config = {
        "experiment": EXPERIMENT,
        "stage": stage_name,
        "subject_id": item.subject,
        "seed": seed,
        "loss": loss_name,
        "context": context_name,
        "input_samples": context_samples,
        "calibration_method": selected_calibration["method"],
        "calibration_duration_seconds": selected_calibration["duration_seconds"],
        "decoder_fine_tune_epochs": selected_calibration["c4_epochs"],
        "train_windows": len(train_inputs),
        "early_stopping_windows": len(calibration_inputs),
        "deployment_calibration_windows": len(deployment_inputs),
        "evaluation_windows": len(evaluation_inputs),
        "test_fog_used": False,
    }
    result = {
        **config,
        **metrics,
        **training,
        "median_lagged_pearson": metrics["median_lagged_pearson"],
        "inference_ms_per_window": inference_ms,
        "run_dir": str(run_dir.resolve()),
    }
    if c4_history:
        base.write_csv(run_dir / "deployment_fine_tune_log.csv", c4_history)
    save_evaluation_run(
        run_dir,
        config=config,
        plan=effective_plan,
        item=item,
        model=model,
        actual=evaluation_targets,
        predicted=predicted,
        latent=latent,
        result=result,
        window_rows=window_rows,
        channel_rows=channel_rows,
        training_log=history,
        save_model=False,
    )
    print(
        f"{stage_name} {item.subject} {loss_name}/{context_name} seed={seed} "
        f"corr={metrics['median_corr']:.3f} nrmse={metrics['median_nrmse']:.3f} "
        f"p90={metrics['nrmse_p90']:.3f} {'PASS' if metrics['strict_pass'] else 'FAIL'}",
        flush=True,
    )
    return result


def ablation_pass_matrix(
    rows: Sequence[dict[str, Any]],
    row_key: str,
    subjects: Sequence[str],
    path: Path,
) -> None:
    settings = sorted(set(str(row[row_key]) for row in rows))
    matrix = np.full((len(subjects), len(settings)), np.nan)
    for y, subject in enumerate(subjects):
        for x, setting in enumerate(settings):
            selected = [
                row for row in rows if row["subject_id"] == subject and row[row_key] == setting
            ]
            if selected:
                matrix[y, x] = float(as_bool(selected[0]["strict_pass"]))
    fig, ax = plt.subplots(figsize=(max(7, len(settings) * 1.3), max(4, len(subjects) * 0.65)))
    ax.imshow(np.nan_to_num(matrix, nan=0.5), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(settings)), settings)
    ax.set_yticks(np.arange(len(subjects)), subjects)
    for y in range(len(subjects)):
        for x in range(len(settings)):
            label = "NA" if np.isnan(matrix[y, x]) else ("PASS" if matrix[y, x] else "FAIL")
            ax.text(x, y, label, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loss_ablation_plots(
    root: Path,
    screen_rows: Sequence[dict[str, Any]],
    expansion_rows: Sequence[dict[str, Any]],
    selected_loss: str,
    prepared: dict[str, a1.PreparedSubject],
) -> None:
    figures = root / "A1b2_loss_ablation" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    losses = list(LOSSES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for subject in ("S05", "S09"):
        selected = [row for row in screen_rows if row["subject_id"] == subject]
        axes[0].plot(losses, [float(next(row["median_corr"] for row in selected if row["loss"] == loss)) for loss in losses], marker="o", label=subject)
        axes[1].plot(losses, [float(next(row["median_nrmse"] for row in selected if row["loss"] == loss)) for loss in losses], marker="o", label=subject)
    axes[0].set_ylabel("Pearson")
    axes[1].set_ylabel("NRMSE")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "loss_ablation_subject_summary.png", dpi=150)
    plt.close(fig)

    for metric, filename in (
        ("pearson", "loss_ablation_channel_pearson.png"),
        ("nrmse", "loss_ablation_channel_nrmse.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(CHANNELS)
        width = 0.8 / len(losses)
        for index, loss in enumerate(losses):
            result = next(row for row in screen_rows if row["subject_id"] == "S05" and row["loss"] == loss)
            channels = read_csv(Path(result["run_dir"]) / "channel_metrics.csv")
            ax.bar(
                x - 0.4 + width / 2 + index * width,
                [float(row[metric]) for row in channels],
                width,
                label=loss,
            )
        ax.set_xticks(x, prepared["S05"].channel_names, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncols=len(losses), fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for subject in ("S05", "S09"):
        selected = [row for row in screen_rows if row["subject_id"] == subject]
        ax.plot(losses, [float(next(row["nrmse_p90"] for row in selected if row["loss"] == loss)) for loss in losses], marker="o", label=subject)
    ax.set_ylabel("NRMSE P90")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "loss_ablation_p90.png", dpi=150)
    plt.close(fig)

    representative = next(
        row for row in screen_rows if row["subject_id"] == "S05" and row["loss"] == selected_loss
    )
    with np.load(Path(representative["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
        a1.plot_best_median_worst(
            payload["target"],
            payload["reconstruction"],
            figures / "loss_ablation_waveforms.png",
            prepared["S05"].channel_names,
        )
    ablation_pass_matrix(
        screen_rows,
        "loss",
        ("S05", "S09"),
        figures / "loss_ablation_pass_matrix.png",
    )


def run_loss_ablation(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    prior_lookup: dict[tuple[str, int], dict[str, Any]],
    selected_calibration: dict[str, Any],
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    screen_path = root / "tables" / "A1b2_loss_screen.csv"
    screen_rows: list[dict[str, Any]] = []
    for subject in ("S05", "S09"):
        item = prepared[subject]
        plan = calibration_plan(item, int(selected_calibration["duration_seconds"]))
        for loss_name in LOSSES:
            screen_rows.append(
                run_trained_setting(
                    root,
                    "A1b2_loss_ablation",
                    item,
                    plan,
                    prior_lookup,
                    selected_calibration,
                    loss_name=loss_name,
                    context_name="W0",
                    seed=SEEDS[0],
                    max_epochs=max_epochs,
                    patience=patience,
                    workers=workers,
                    device=device,
                    overwrite=overwrite,
                )
            )
    base.write_csv(screen_path, screen_rows)
    grouped = {
        loss_name: [row for row in screen_rows if row["loss"] == loss_name]
        for loss_name in LOSSES
    }
    selected_loss = max(grouped, key=lambda loss: repair_rank(grouped[loss]))
    expansion_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        item = prepared[subject]
        plan = calibration_plan(item, int(selected_calibration["duration_seconds"]))
        expansion_rows.append(
            run_trained_setting(
                root,
                "A1b2_loss_ablation",
                item,
                plan,
                prior_lookup,
                selected_calibration,
                loss_name=selected_loss,
                context_name="W0",
                seed=SEEDS[0],
                max_epochs=max_epochs,
                patience=patience,
                workers=workers,
                device=device,
                overwrite=overwrite,
            )
        )
    base.write_csv(root / "tables" / "A1b2_loss_expansion.csv", expansion_rows)
    decision = {
        "selected_loss": selected_loss,
        "screen_rank": repair_rank(grouped[selected_loss]),
        "one_seed_subject_pass_count": sum(as_bool(row["strict_pass"]) for row in expansion_rows),
        "selection_priority": [
            "Pearson",
            "NRMSE",
            "NRMSE P90",
            "amplitude ratio",
            "pass count",
        ],
    }
    base.write_json(root / "reports" / "A1b2_selected_loss.json", decision)
    if not skip_figures:
        loss_ablation_plots(root, screen_rows, expansion_rows, selected_loss, prepared)
    return selected_loss, expansion_rows, decision


def context_ablation_plots(
    root: Path,
    screen_rows: Sequence[dict[str, Any]],
    selected_context: str,
    screen_subjects: Sequence[str],
    prepared: dict[str, a1.PreparedSubject],
) -> None:
    figures = root / "A1b3_context_ablation" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    settings = list(CONTEXTS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for subject in screen_subjects:
        rows = [row for row in screen_rows if row["subject_id"] == subject]
        axes[0].plot(settings, [float(next(row["median_corr"] for row in rows if row["context"] == setting)) for setting in settings], marker="o", label=subject)
        axes[1].plot(settings, [float(next(row["median_nrmse"] for row in rows if row["context"] == setting)) for setting in settings], marker="o", label=subject)
        axes[2].plot(settings, [float(next(row["nrmse_p90"] for row in rows if row["context"] == setting)) for setting in settings], marker="o", label=subject)
    for ax, label in zip(axes, ("Pearson", "NRMSE", "NRMSE P90")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "context_length_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for subject in screen_subjects:
        rows = [row for row in screen_rows if row["subject_id"] == subject]
        ax.plot(settings, [float(next(row["median_abs_best_lag"] for row in rows if row["context"] == setting)) for setting in settings], marker="o", label=subject)
    ax.set_ylabel("median |best lag| (samples)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "context_length_lag_error.png", dpi=150)
    plt.close(fig)

    representative = next(
        row for row in screen_rows
        if row["subject_id"] == screen_subjects[0] and row["context"] == selected_context
    )
    with np.load(Path(representative["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
        a1.plot_best_median_worst(
            payload["target"],
            payload["reconstruction"],
            figures / "context_length_waveforms.png",
            prepared[screen_subjects[0]].channel_names,
        )
    ablation_pass_matrix(
        screen_rows,
        "context",
        screen_subjects,
        figures / "context_length_subject_pass_matrix.png",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    reference_subject = screen_subjects[0]
    rows = [row for row in screen_rows if row["subject_id"] == reference_subject]
    axes[0].bar(settings, [int(next(row["parameter_count"] for row in rows if row["context"] == setting)) for setting in settings])
    axes[0].set_ylabel("parameters")
    axes[1].bar(settings, [float(next(row["inference_ms_per_window"] for row in rows if row["context"] == setting)) for setting in settings])
    axes[1].set_ylabel("inference ms/window")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "context_length_parameter_latency.png", dpi=150)
    plt.close(fig)


def run_context_ablation(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    prior_lookup: dict[tuple[str, int], dict[str, Any]],
    selected_calibration: dict[str, Any],
    selected_loss: str,
    diagnostic_rows: Sequence[dict[str, Any]],
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    phase_subjects = [
        str(row["subject_id"])
        for row in diagnostic_rows
        if str(row["subject_id"]) in ("S02", "S06")
        and float(row["median_lagged_pearson"]) - float(row["median_pearson"]) >= 0.12
    ]
    screen_subjects = tuple(dict.fromkeys(("S05", "S09", *phase_subjects)))
    screen_rows: list[dict[str, Any]] = []
    for subject in screen_subjects:
        item = prepared[subject]
        plan = calibration_plan(item, int(selected_calibration["duration_seconds"]))
        for context_name in CONTEXTS:
            source_stage = (
                "A1b2_loss_ablation" if context_name == "W0" else "A1b3_context_ablation"
            )
            screen_rows.append(
                run_trained_setting(
                    root,
                    source_stage,
                    item,
                    plan,
                    prior_lookup,
                    selected_calibration,
                    loss_name=selected_loss,
                    context_name=context_name,
                    seed=SEEDS[0],
                    max_epochs=max_epochs,
                    patience=patience,
                    workers=workers,
                    device=device,
                    overwrite=overwrite,
                )
            )
    base.write_csv(root / "tables" / "A1b3_context_screen.csv", screen_rows)
    grouped = {
        context_name: [row for row in screen_rows if row["context"] == context_name]
        for context_name in CONTEXTS
    }
    baseline_corr = float(np.median([float(row["median_corr"]) for row in grouped["W0"]]))
    baseline_nrmse = float(np.median([float(row["median_nrmse"]) for row in grouped["W0"]]))
    baseline_lag = float(
        np.median([float(row["median_abs_best_lag"]) for row in grouped["W0"]])
    )
    eligible_contexts = ["W0"]
    for context_name in ("W1", "W2"):
        candidate_corr = float(
            np.median([float(row["median_corr"]) for row in grouped[context_name]])
        )
        candidate_nrmse = float(
            np.median([float(row["median_nrmse"]) for row in grouped[context_name]])
        )
        candidate_lag = float(
            np.median(
                [float(row["median_abs_best_lag"]) for row in grouped[context_name]]
            )
        )
        if (
            candidate_corr > baseline_corr
            and candidate_nrmse < baseline_nrmse
            and candidate_lag <= baseline_lag
        ):
            eligible_contexts.append(context_name)
    selected_context = max(
        eligible_contexts, key=lambda context: repair_rank(grouped[context])
    )
    expansion_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        item = prepared[subject]
        plan = calibration_plan(item, int(selected_calibration["duration_seconds"]))
        source_stage = (
            "A1b2_loss_ablation"
            if selected_context == "W0"
            else "A1b3_context_ablation"
        )
        expansion_rows.append(
            run_trained_setting(
                root,
                source_stage,
                item,
                plan,
                prior_lookup,
                selected_calibration,
                loss_name=selected_loss,
                context_name=selected_context,
                seed=SEEDS[0],
                max_epochs=max_epochs,
                patience=patience,
                workers=workers,
                device=device,
                overwrite=overwrite,
            )
        )
    base.write_csv(root / "tables" / "A1b3_context_expansion.csv", expansion_rows)
    decision = {
        "screen_subjects": list(screen_subjects),
        "selected_context": selected_context,
        "input_samples": CONTEXTS[selected_context],
        "screen_rank": repair_rank(grouped[selected_context]),
        "eligible_contexts": eligible_contexts,
        "eligibility_rule": (
            "relative to W0, aggregate Pearson must increase, aggregate NRMSE must decrease, "
            "and median absolute best lag must not increase"
        ),
        "one_seed_subject_pass_count": sum(as_bool(row["strict_pass"]) for row in expansion_rows),
        "parameter_count": int(expansion_rows[0]["parameter_count"]),
    }
    base.write_json(root / "reports" / "A1b3_selected_context.json", decision)
    if not skip_figures:
        context_ablation_plots(
            root, screen_rows, selected_context, screen_subjects, prepared
        )
    return selected_context, expansion_rows, decision


def waveform_grid(
    actual: np.ndarray,
    predicted: np.ndarray,
    index: int,
    channel_names: Sequence[str],
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    t = np.arange(WINDOW) / FS
    for channel, ax in enumerate(axes.ravel()):
        ax.plot(t, actual[index, :, channel], label="target", linewidth=0.9)
        ax.plot(t, predicted[index, :, channel], "--", label="reconstruction", linewidth=0.9)
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_retest_subject(
    root: Path,
    subject: str,
    rows: Sequence[dict[str, Any]],
    channel_names: Sequence[str],
) -> None:
    representative = sorted(rows, key=lambda row: float(row["median_nrmse"]))[len(rows) // 2]
    run_dir = Path(representative["run_dir"])
    figures = root / "A1_retest" / subject / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as payload:
        actual = np.asarray(payload["target"])
        predicted = np.asarray(payload["reconstruction"])
    arrays = base.metric_arrays(actual, predicted)
    lagged, lags = lagged_metrics(actual, predicted)
    _, low = a1.spectral_error(actual, predicted, 0.5, 3.0)
    _, high = a1.spectral_error(actual, predicted, 3.0, 8.0)
    order = np.argsort(np.median(arrays["nrmse"], axis=1))
    median_index = int(order[len(order) // 2])
    worst_index = int(order[-1])

    log_rows = read_csv(run_dir / "training_log.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    numeric = [row for row in log_rows if row.get("train_loss") not in (None, "")]
    if numeric:
        ax.plot([int(row["epoch"]) for row in numeric], [float(row["train_loss"]) for row in numeric], label="train")
        ax.plot([int(row["epoch"]) for row in numeric], [float(row["calibration_loss"]) for row in numeric], label="early stopping")
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Frozen original A1 checkpoint reused", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="epoch", ylabel="loss")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "training_curve.png", dpi=150)
    plt.close(fig)

    waveform_grid(
        actual,
        predicted,
        median_index,
        channel_names,
        figures / "test_median_window_waveform.png",
        f"{subject} median clean Non-FoG window",
    )
    waveform_grid(
        actual,
        predicted,
        worst_index,
        channel_names,
        figures / "test_worst_window_waveform.png",
        f"{subject} worst clean Non-FoG window",
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(
        np.abs(actual[worst_index] - predicted[worst_index]).T,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    ax.set_yticks(np.arange(CHANNELS), channel_names)
    ax.set_xlabel("sample")
    fig.colorbar(image, ax=ax, label="absolute residual")
    fig.tight_layout()
    fig.savefig(figures / "test_residual_heatmap.png", dpi=150)
    plt.close(fig)

    for values, filename, ylabel in (
        (np.median(arrays["correlation"], axis=0), "channel_pearson.png", "Pearson"),
        (np.median(arrays["nrmse"], axis=0), "channel_nrmse.png", "NRMSE"),
        (np.median(arrays["amplitude_ratio"], axis=0), "channel_amplitude_ratio.png", "amplitude ratio"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(channel_names, values)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(channel_names, np.median(lagged, axis=0))
    axes[0].set_ylabel("lagged Pearson")
    axes[1].bar(channel_names, np.median(np.abs(lags), axis=0))
    axes[1].set_ylabel("median |best lag|")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "lagged_correlation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(CHANNELS)
    ax.bar(x - 0.18, np.median(low, axis=0), 0.36, label="0.5-3 Hz")
    ax.bar(x + 0.18, np.median(high, axis=0), 0.36, label="3-8 Hz")
    ax.set_xticks(x, channel_names, rotation=35, ha="right")
    ax.set_ylabel("spectral NRMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "frequency_error.png", dpi=150)
    plt.close(fig)

    window_nrmse = np.median(arrays["nrmse"], axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.sort(window_nrmse))
    ax.axhline(1.30, linestyle="--", color="red", label="P90 gate")
    ax.set(xlabel="sorted window", ylabel="window median NRMSE")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "window_nrmse_sorted.png", dpi=150)
    plt.close(fig)

    energy = np.median(np.sqrt(np.mean(actual.astype(np.float64) ** 2, axis=1)), axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(energy, window_nrmse, s=10, alpha=0.5)
    ax.set(xlabel="window RMS energy", ylabel="window NRMSE")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "window_energy_vs_nrmse.png", dpi=150)
    plt.close(fig)

    window_rows = read_csv(run_dir / "window_metrics.csv")
    groups: dict[str, list[float]] = {}
    for row in window_rows:
        groups.setdefault(str(row["record_id"]), []).append(float(row["nrmse_median"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(groups), [float(np.median(values)) for values in groups.values()])
    ax.set_ylabel("median NRMSE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "record_or_timeblock_nrmse.png", dpi=150)
    plt.close(fig)


def render_retest_global(
    root: Path,
    retest_rows: Sequence[dict[str, Any]],
    prior_results: Sequence[dict[str, Any]],
) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(SUBJECTS), len(SEEDS)), dtype=float)
    for y, subject in enumerate(SUBJECTS):
        for x, seed in enumerate(SEEDS):
            row = next(
                row for row in retest_rows
                if row["subject_id"] == subject and int(row["seed"]) == seed
            )
            matrix[y, x] = float(as_bool(row["strict_pass"]))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(SEEDS)), SEEDS)
    ax.set_yticks(np.arange(len(SUBJECTS)), SUBJECTS)
    for y in range(len(SUBJECTS)):
        for x in range(len(SEEDS)):
            ax.text(x, y, "PASS" if matrix[y, x] else "FAIL", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figures / "a1_retest_pass_matrix.png", dpi=150)
    plt.close(fig)

    prior_lookup_local = {(row["subject_id"], int(row["seed"])): row for row in prior_results}
    for metric, filename, ylabel in (
        ("median_corr", "baseline_vs_final_pearson.png", "Pearson"),
        ("median_nrmse", "baseline_vs_final_nrmse.png", "NRMSE"),
        ("nrmse_p90", "baseline_vs_final_p90.png", "NRMSE P90"),
    ):
        baseline = [
            float(np.median([float(prior_lookup_local[(subject, seed)][metric]) for seed in SEEDS]))
            for subject in SUBJECTS
        ]
        final = [
            float(np.median([float(row[metric]) for row in retest_rows if row["subject_id"] == subject]))
            for subject in SUBJECTS
        ]
        x = np.arange(len(SUBJECTS))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - 0.18, baseline, 0.36, label="original A1")
        ax.bar(x + 0.18, final, 0.36, label="A1-Retest")
        ax.set_xticks(x, SUBJECTS)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for subject in SUBJECTS:
        values = [float(row["median_corr"]) for row in retest_rows if row["subject_id"] == subject]
        axes[0].plot(SEEDS, values, marker="o", label=subject)
        values = [float(row["median_nrmse"]) for row in retest_rows if row["subject_id"] == subject]
        axes[1].plot(SEEDS, values, marker="o", label=subject)
    axes[0].set_ylabel("Pearson")
    axes[1].set_ylabel("NRMSE")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(ncols=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "all_subject_seed_stability.png", dpi=150)
    plt.close(fig)

    summary = []
    for subject in SUBJECTS:
        rows = [row for row in retest_rows if row["subject_id"] == subject]
        summary.append(
            (
                subject,
                float(np.median([float(row["median_corr"]) for row in rows])),
                float(np.median([float(row["median_nrmse"]) for row in rows])),
                sum(as_bool(row["strict_pass"]) for row in rows),
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter([row[1] for row in summary], [row[2] for row in summary])
    for subject, corr, nrmse, _ in summary:
        axes[0].annotate(subject, (corr, nrmse))
    axes[0].axvline(0.5, linestyle="--", color="red")
    axes[0].axhline(0.85, linestyle="--", color="red")
    axes[0].set(xlabel="median Pearson", ylabel="median NRMSE")
    axes[1].bar([row[0] for row in summary], [row[3] for row in summary])
    axes[1].axhline(2, linestyle="--", color="red")
    axes[1].set_ylabel("passing seeds")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "all_subject_generalization_summary.png", dpi=150)
    plt.close(fig)


def run_retest(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    prior_lookup: dict[tuple[str, int], dict[str, Any]],
    prior_results: Sequence[dict[str, Any]],
    selected_calibration: dict[str, Any],
    selected_loss: str,
    selected_context: str,
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        item = prepared[subject]
        plan = calibration_plan(item, int(selected_calibration["duration_seconds"]))
        for seed in SEEDS:
            rows.append(
                run_trained_setting(
                    root,
                    "A1_retest",
                    item,
                    plan,
                    prior_lookup,
                    selected_calibration,
                    loss_name=selected_loss,
                    context_name=selected_context,
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                    workers=workers,
                    device=device,
                    overwrite=overwrite,
                )
            )
    base.write_csv(root / "tables" / "A1_retest_run_metrics.csv", rows)
    gate_rows = [
        {
            **row,
            "split_type": prepared[str(row["subject_id"])].disclosure["split_type"],
            "test_record_or_block": prepared[str(row["subject_id"])].disclosure.get(
                "test_record", prepared[str(row["subject_id"])].disclosure.get("record_id")
            ),
        }
        for row in rows
    ]
    gate = a1.evaluate_a1_gate(gate_rows)
    tail_risk = any(
        float(np.median([float(row["nrmse_p95"]) for row in rows if row["subject_id"] == subject])) > 1.50
        for subject in SUBJECTS
    )
    if int(gate["subject_pass_count"]) < 5:
        status = "FAIL"
    elif tail_risk:
        status = "CONDITIONAL PASS"
    else:
        status = "STRICT PASS"
    gate["a1_retest_status"] = status
    gate["tail_risk_definition"] = "any subject median seed P95 > 1.50"
    gate["tail_risk"] = tail_risk
    gate["eligible_for_A2"] = status in ("STRICT PASS", "CONDITIONAL PASS")
    base.write_json(root / "reports" / "A1_retest_gate.json", gate)
    if not skip_figures:
        for subject in SUBJECTS:
            render_retest_subject(
                root,
                subject,
                [row for row in rows if row["subject_id"] == subject],
                prepared[subject].channel_names,
            )
        render_retest_global(root, rows, prior_results)
    return rows, gate


def write_protocol(
    root: Path,
    args: argparse.Namespace,
    prepared: dict[str, a1.PreparedSubject],
) -> None:
    protocol = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "template": str(args.template.resolve()),
        "template_sha256": sha256(args.template),
        "parent_A1_root": str(args.parent_root.resolve()),
        "parent_frozen_config_sha256": sha256(args.parent_root / "A0_protocol" / "frozen_config.json"),
        "parent_split_manifest_sha256": sha256(args.parent_root / "A0_protocol" / "data_split_manifest.csv"),
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "failed_subject_screen": list(FAILED_SUBJECTS),
        "loss_context_screen": ["S05", "S09", "plus S02/S06 when lag diagnosis qualifies"],
        "calibration_durations_seconds": list(CALIBRATION_DURATIONS),
        "calibration_methods": list(CALIBRATION_METHODS),
        "c4_epochs": list(C4_EPOCHS),
        "losses": list(LOSSES),
        "contexts": CONTEXTS,
        "deployment_calibration_policy": {
            "source": "early unique samples covered by clean Non-FoG windows in frozen A1 test record/time block",
            "evaluation_exclusion": "all windows within five seconds of any calibration sample",
            "calibration_fog_samples_allowed": 0,
            "evaluation_fog_samples_allowed": 0,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "maximum_epochs": args.max_epochs,
            "patience": args.patience,
        },
        "split_disclosures": {
            subject: item.disclosure for subject, item in prepared.items()
        },
        "test_fog_used": False,
        "interpretation_boundary": (
            "A1b uses the already observed A1 clean Non-FoG benchmark for diagnosis and method selection. "
            "A1-Retest is a frozen-benchmark confirmation, not a new untouched external test."
        ),
    }
    base.write_json(root / "protocol" / "frozen_A1b_protocol.json", protocol)


def write_reports(
    root: Path,
    diagnostics: Sequence[dict[str, Any]],
    selected_calibration: dict[str, Any],
    calibration_gate: dict[str, Any],
    loss_decision: dict[str, Any] | None,
    context_decision: dict[str, Any] | None,
    retest_rows: Sequence[dict[str, Any]] | None,
    retest_gate: dict[str, Any] | None,
) -> None:
    reports = root / "reports"
    diagnostic_lines = "\n".join(
        f"- {row['subject_id']}: {row['failure_type']}; Pearson={float(row['median_pearson']):.3f}, "
        f"lagged Pearson={float(row['median_lagged_pearson']):.3f}, "
        f"|lag|={float(row['median_abs_best_lag']):.1f}, NRMSE={float(row['median_nrmse']):.3f}."
        for row in diagnostics
    )
    write_text(
        reports / "A1b0_failure_diagnosis_report.md",
        f"""# A1b-0 未见 clean Non-FoG 失败诊断

本阶段复用原 A1 冻结检查点和预测，不重新训练，不读取测试 FoG。

{diagnostic_lines}

失败类型是预注册启发式诊断标签，用于确定修复顺序，不作为分类结论。
""",
    )
    calibration_subject_lines = "\n".join(
        f"- {row['subject_id']}: {row['seed_passes']}/3 seeds，"
        f"Pearson={float(row['median_corr']):.3f}，NRMSE={float(row['median_nrmse']):.3f}，"
        f"P90={float(row['median_nrmse_p90']):.3f}，"
        f"{'PASS' if as_bool(row['subject_pass']) else 'FAIL'}。"
        for row in calibration_gate["subject_summary"]
    )
    write_text(
        reports / "A1b1_record_calibration_report.md",
        f"""# A1b-1 记录级 clean Non-FoG 校准

- 冻结方法：{selected_calibration['method']}
- 校准时长：{selected_calibration['duration_seconds']} 秒
- Decoder 轻量微调：{selected_calibration['c4_epochs']} epochs
- 扩展门控：{calibration_gate['status']}，{calibration_gate['subject_pass_count']}/7 subjects
- 校准样本与评价窗口之间保留 5 秒保护区；校准和评价均为 clean Non-FoG。

{calibration_subject_lines}
""",
    )
    if loss_decision is None:
        write_text(
            reports / "A1b2_loss_ablation_report.md",
            "# A1b-2 波形结构损失\n\n状态：NOT RUN。记录级校准已达到直接正式复测条件。\n",
        )
    else:
        write_text(
            reports / "A1b2_loss_ablation_report.md",
            f"""# A1b-2 波形结构损失

- 选择损失：{loss_decision['selected_loss']}
- 全部 7 被试单种子通过：{loss_decision['one_seed_subject_pass_count']}/7
- 选择顺序：Pearson → NRMSE → P90 → 振幅保持 → 通过数。
""",
        )
    if context_decision is None:
        write_text(
            reports / "A1b3_context_ablation_report.md",
            "# A1b-3 上下文长度\n\n状态：NOT RUN。记录级校准已达到直接正式复测条件。\n",
        )
    else:
        write_text(
            reports / "A1b3_context_ablation_report.md",
            f"""# A1b-3 上下文长度

- 筛选被试：{', '.join(context_decision['screen_subjects'])}
- 选择上下文：{context_decision['selected_context']}（{context_decision['input_samples']} samples）
- 参数量：{context_decision['parameter_count']}
- 全部 7 被试单种子通过：{context_decision['one_seed_subject_pass_count']}/7
""",
        )
    if retest_rows is None or retest_gate is None:
        return
    subject_lines = "\n".join(
        f"- {row['subject_id']}: {row['seed_passes']}/3 seeds，"
        f"Pearson={float(row['median_corr']):.3f}，NRMSE={float(row['median_nrmse']):.3f}，"
        f"P90={float(row['median_nrmse_p90']):.3f}，"
        f"{'PASS' if as_bool(row['subject_pass']) else 'FAIL'}。"
        for row in retest_gate["subject_summary"]
    )
    eligibility = (
        "允许进入 A2，但仍须严格执行 A2 门控。"
        if retest_gate["eligible_for_A2"]
        else "不得进入 FoG 残差分离；应转向多原型、条件化或记录级 NBM。"
    )
    write_text(
        reports / "A1_retest_report.md",
        f"""# A1-Retest 正式复测

- 状态：**{retest_gate['a1_retest_status']}**
- 运行通过：{retest_gate['run_pass_count']}/{retest_gate['run_total']}
- 被试通过：{retest_gate['subject_pass_count']}/7
- 明显波形塌缩被试：{retest_gate['waveform_collapse_subject_count']}
- 尾部风险：{'是' if retest_gate['tail_risk'] else '否'}

{subject_lines}

{eligibility}

边界说明：按模板，本轮诊断、方案选择与复测复用了原 A1 clean Non-FoG 基准。因此这是冻结基准上的确认性复测，不等同于新的外部未见测试；全过程未使用测试 FoG。
""",
    )
    write_text(
        reports / "A1b_final_generalization_repair_report.md",
        f"""# Daphnet NBM Route A A1b 最终总结

A1b 已按顺序完成诊断、条件校准/损失/上下文修复和 7 被试 × 3 种子复测。

- 最终校准：{selected_calibration['method']} / {selected_calibration['duration_seconds']} 秒 / C4={selected_calibration['c4_epochs']} epochs
- 最终损失：{loss_decision['selected_loss'] if loss_decision else 'L0'}
- 最终上下文：{context_decision['selected_context'] if context_decision else 'W0'}
- A1-Retest：**{retest_gate['a1_retest_status']}**，{retest_gate['subject_pass_count']}/7 subjects
- 测试 FoG 用于选择：否

{eligibility}
""",
    )


def audit_outputs(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    selected_duration: int,
) -> dict[str, Any]:
    predictions = list(root.rglob("predictions.npz"))
    checkpoints = list(root.rglob("*_model.pt"))
    prediction_errors: list[str] = []
    for path in predictions:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if not all(np.isfinite(payload[key]).all() for key in payload.files):
                    prediction_errors.append(f"non-finite: {path}")
                if "target" in payload and "reconstruction" in payload:
                    if payload["target"].shape != payload["reconstruction"].shape:
                        prediction_errors.append(f"shape mismatch: {path}")
        except Exception as error:
            prediction_errors.append(f"{path}: {error}")
    checkpoint_errors: list[str] = []
    for path in checkpoints:
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            checkpoint_errors.append(f"{path}: {error}")
    calibration_audit = {}
    for subject, item in prepared.items():
        plan = calibration_plan(item, selected_duration)
        calibration_ids = set(plan.calibration_indices.tolist())
        evaluation_ids = set(plan.evaluation_indices.tolist())
        calibration_audit[subject] = {
            "calibration_windows": len(calibration_ids),
            "evaluation_windows": len(evaluation_ids),
            "window_id_overlap": len(calibration_ids & evaluation_ids),
            "calibration_all_clean_nonfog": bool(
                np.all(item.windows.clean_normal[plan.calibration_indices])
            ),
            "evaluation_all_clean_nonfog": bool(
                np.all(item.windows.clean_normal[plan.evaluation_indices])
            ),
        }
    configs = list(root.rglob("config.json"))
    test_fog_flags = [
        read_json(path).get("test_fog_used")
        for path in configs
        if "test_fog_used" in read_json(path)
    ]
    output = {
        "run_metrics": len(list(root.rglob("run_metrics.json"))),
        "prediction_files": len(predictions),
        "checkpoint_files": len(checkpoints),
        "all_predictions_finite_and_shape_aligned": not prediction_errors,
        "prediction_errors": prediction_errors,
        "all_checkpoints_loadable": not checkpoint_errors,
        "checkpoint_errors": checkpoint_errors,
        "all_explicit_test_fog_flags_false": all(flag is False for flag in test_fog_flags),
        "explicit_test_fog_flag_count": len(test_fog_flags),
        "calibration_evaluation_audit": calibration_audit,
        "diagnostic_subject_figures": len(list((root / "A1b0_diagnostics").glob("S*/figures/*.png"))),
        "retest_subject_figures": len(list((root / "A1_retest").glob("S*/figures/*.png"))),
        "global_figures": len(list((root / "figures").glob("*.png"))),
        "temporary_files": len([path for path in root.rglob("*") if path.is_file() and ".tmp-" in path.name]),
        "zero_byte_files": len([path for path in root.rglob("*") if path.is_file() and path.stat().st_size == 0]),
    }
    base.write_json(root / "reports" / "artifact_audit.json", output)
    return output


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.parent_root = args.parent_root.resolve()
    args.output_dir = args.output_dir.resolve()
    root = args.output_dir / "routeA_A1b_generalization_repair"
    root.mkdir(parents=True, exist_ok=True)
    if not args.parent_root.exists():
        raise FileNotFoundError(f"parent A1 output missing: {args.parent_root}")
    dataset = DaphnetDataset.load(args.data_dir)
    prepared = {subject: a1.prepare_subject(dataset, subject) for subject in SUBJECTS}
    prior_results = original_results(args.parent_root)
    if len(prior_results) != 21:
        raise ValueError(f"expected 21 parent A1 runs, found {len(prior_results)}")
    prior_lookup = result_lookup(prior_results)
    write_protocol(root, args, prepared)
    device = base.resolve_device(args.device)

    diagnostics, _, _ = run_diagnostics(
        root, prepared, prior_results, skip_figures=args.skip_figures
    )
    if args.stop_after == "diagnostics":
        print("A1b stopped after diagnostics by request", flush=True)
        return
    selected_calibration, _, calibration_gate = run_calibration(
        root,
        prepared,
        prior_lookup,
        device=device,
        overwrite=args.overwrite,
        skip_figures=args.skip_figures,
    )
    if args.stop_after == "calibration":
        write_reports(root, diagnostics, selected_calibration, calibration_gate, None, None, None, None)
        print("A1b stopped after calibration by request", flush=True)
        return

    loss_decision: dict[str, Any] | None = None
    context_decision: dict[str, Any] | None = None
    if bool(selected_calibration["advance_directly_to_retest"]):
        selected_loss = "L0"
        selected_context = "W0"
        for stage, reason in (
            ("A1b2_loss_ablation", "record-level calibration already reached >=5/7 subjects"),
            ("A1b3_context_ablation", "record-level calibration already reached >=5/7 subjects"),
        ):
            base.write_json(root / stage / "status.json", {"status": "NOT RUN", "reason": reason})
    else:
        selected_loss, _, loss_decision = run_loss_ablation(
            root,
            prepared,
            prior_lookup,
            selected_calibration,
            max_epochs=args.max_epochs,
            patience=args.patience,
            workers=args.num_workers,
            device=device,
            overwrite=args.overwrite,
            skip_figures=args.skip_figures,
        )
        if args.stop_after == "loss":
            write_reports(
                root,
                diagnostics,
                selected_calibration,
                calibration_gate,
                loss_decision,
                None,
                None,
                None,
            )
            print("A1b stopped after loss ablation by request", flush=True)
            return
        selected_context, _, context_decision = run_context_ablation(
            root,
            prepared,
            prior_lookup,
            selected_calibration,
            selected_loss,
            diagnostics,
            max_epochs=args.max_epochs,
            patience=args.patience,
            workers=args.num_workers,
            device=device,
            overwrite=args.overwrite,
            skip_figures=args.skip_figures,
        )
        if args.stop_after == "context":
            write_reports(
                root,
                diagnostics,
                selected_calibration,
                calibration_gate,
                loss_decision,
                context_decision,
                None,
                None,
            )
            print("A1b stopped after context ablation by request", flush=True)
            return

    frozen = {
        "calibration": {
            "method": selected_calibration["method"],
            "duration_seconds": selected_calibration["duration_seconds"],
            "c4_epochs": selected_calibration["c4_epochs"],
        },
        "loss": selected_loss,
        "context": selected_context,
        "input_samples": CONTEXTS[selected_context],
        "seeds": list(SEEDS),
        "gate": "unchanged original A1 gate",
        "test_fog_used": False,
        "frozen_before_A1_retest": True,
    }
    base.write_json(root / "protocol" / "frozen_final_repair.json", frozen)
    retest_rows, retest_gate = run_retest(
        root,
        prepared,
        prior_lookup,
        prior_results,
        selected_calibration,
        selected_loss,
        selected_context,
        max_epochs=args.max_epochs,
        patience=args.patience,
        workers=args.num_workers,
        device=device,
        overwrite=args.overwrite,
        skip_figures=args.skip_figures,
    )
    write_reports(
        root,
        diagnostics,
        selected_calibration,
        calibration_gate,
        loss_decision,
        context_decision,
        retest_rows,
        retest_gate,
    )
    audit = audit_outputs(
        root, prepared, int(selected_calibration["duration_seconds"])
    )
    print(
        f"A1b COMPLETE status={retest_gate['a1_retest_status']} "
        f"subjects={retest_gate['subject_pass_count']}/7 audit={json.dumps(audit, ensure_ascii=False)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
