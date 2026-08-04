#!/usr/bin/env python
"""Run zero-shot waveform-loss and strictly causal context experiments on Daphnet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for location in (REPO_ROOT, SCRIPTS_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_daphnet_nbm_routeA_A1b_generalization_repair as a1b  # noqa: E402
import run_daphnet_nbm_routeA_final_residual_validation as a1  # noqa: E402
import run_daphnet_nbm_tcdae_three_rounds as base  # noqa: E402
import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as legacy  # noqa: E402
from cnbr_fog.data import DaphnetDataset  # noqa: E402


EXPERIMENT = "daphnet_nbm_zero_shot_causal_waveform_context_v1"
SUBJECTS = a1.SUBJECTS
SEEDS = a1.SEEDS
FAILED_SUBJECTS = ("S02", "S05", "S06", "S09")
ORIGINAL_PASS_SUBJECTS = ("S01", "S07", "S08")
LOSS_SCREEN_SUBJECTS = ("S05", "S09")
FS = 64
WINDOW = 128
CHANNELS = 9
CONTEXTS = {"W0": 128, "W1_causal": 256, "W2_causal": 384}
LOSSES = ("L0", "L1", "L2", "L3", "L4", "L5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument(
        "--parent-a1",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_nbm_routeA_final_residual_validation_v1"
        / "routeA_final_residual_validation",
    )
    parser.add_argument(
        "--parent-a1b",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_nbm_routeA_A1b_generalization_repair_v1"
        / "routeA_A1b_generalization_repair",
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
            r"C:\Users\bin\Downloads\Daphnet_NBM_zero_shot_causal_waveform_context_experiment_template.md"
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
        choices=("diagnostics", "loss", "context", "final"),
        default="final",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "pass")


class CausalContextM3(nn.Module):
    """M3-long decoding the most recent two seconds from causal history."""

    def __init__(self, input_samples: int = 128) -> None:
        super().__init__()
        if input_samples not in CONTEXTS.values():
            raise ValueError(f"unsupported context length {input_samples}")
        self.input_samples = int(input_samples)
        self.core = base.build_model("M3_tcdae_long")

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (CHANNELS, self.input_samples)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected[0]},{expected[1]}], got {tuple(values.shape)}")
        latent = self.core.encoder_stage3(
            self.core.encoder_stage2(self.core.encoder_stage1(values))
        )
        decoded = self.core.decoder_stage1(
            F.interpolate(latent, scale_factor=2.0, mode="linear", align_corners=False)
        )
        decoded = self.core.decoder_final(
            F.interpolate(decoded, scale_factor=2.0, mode="linear", align_corners=False)
        )
        return decoded[..., -WINDOW:], latent

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "M3_tcdae_long_causal_recent_target",
            "input_shape": ["batch", CHANNELS, self.input_samples],
            "target_shape": ["batch", CHANNELS, WINDOW],
            "latent_shape": ["batch", 48, self.input_samples // 4],
            "target_position": "most recent 128 samples",
            "future_samples": 0,
            "long_skip": False,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def masked_correlation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = predicted - predicted.mean(dim=-1, keepdim=True)
    actual = target - target.mean(dim=-1, keepdim=True)
    target_std = torch.sqrt(torch.mean(actual.square(), dim=-1) + 1e-8)
    mask = target_std >= 1e-3
    numerator = torch.sum(pred * actual, dim=-1)
    denominator = torch.sqrt(
        torch.sum(pred.square(), dim=-1) * torch.sum(actual.square(), dim=-1) + 1e-8
    )
    correlation = numerator / denominator
    if torch.any(mask):
        return 1.0 - torch.mean(correlation[mask])
    return torch.zeros((), dtype=predicted.dtype, device=predicted.device)


def log_spectrum_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    frequency = torch.fft.rfftfreq(WINDOW, d=1.0 / FS, device=predicted.device)
    mask = (frequency >= 0.5) & (frequency <= 10.0)
    pred_spectrum = torch.log1p(torch.abs(torch.fft.rfft(predicted, dim=-1)))[..., mask]
    target_spectrum = torch.log1p(torch.abs(torch.fft.rfft(target, dim=-1)))[..., mask]
    return F.l1_loss(pred_spectrum, target_spectrum)


def waveform_loss(name: str, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(predicted, target)
    if name == "L0":
        return mse
    if name == "L1":
        per_window_channel_mse = torch.mean((predicted - target).square(), dim=-1)
        per_window_channel_variance = torch.var(target, dim=-1, unbiased=False)
        return torch.mean(per_window_channel_mse / (per_window_channel_variance + 1e-3))
    correlation = masked_correlation_loss(predicted, target)
    if name == "L2":
        return 0.8 * mse + 0.2 * correlation
    delta = F.mse_loss(
        predicted[..., 1:] - predicted[..., :-1],
        target[..., 1:] - target[..., :-1],
    )
    if name == "L3":
        return 0.70 * mse + 0.15 * correlation + 0.15 * delta
    if name == "L4":
        return 0.70 * F.smooth_l1_loss(predicted, target, beta=1.0) + 0.15 * correlation + 0.15 * delta
    if name == "L5":
        return 0.60 * mse + 0.15 * correlation + 0.15 * delta + 0.10 * log_spectrum_loss(predicted, target)
    raise ValueError(f"unknown loss {name}")


def interval_for_target(
    item: a1.PreparedSubject,
    raw_index: int,
) -> legacy.Interval | None:
    rec_idx = int(item.windows.record_index[raw_index])
    record_id = item.records[rec_idx].record_id
    split = str(item.windows.split[raw_index])
    start = int(item.windows.start[raw_index])
    end = int(item.windows.end[raw_index])
    intervals, _ = a1.subject_intervals(item.subject, item.records)
    for interval in intervals:
        if (
            interval.record_id == record_id
            and interval.split == split
            and start >= interval.start
            and end <= interval.end
        ):
            return interval
    return None


def causal_context_arrays(
    item: a1.PreparedSubject,
    indices: Sequence[int],
    input_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    kept: list[int] = []
    manifest: list[dict[str, Any]] = []
    for raw_index in np.asarray(indices, dtype=np.int64):
        raw_index_int = int(raw_index)
        rec_idx = int(item.windows.record_index[raw_index_int])
        record = item.records[rec_idx]
        target_start = int(item.windows.start[raw_index_int])
        target_end = int(item.windows.end[raw_index_int])
        input_end = target_end
        input_start = input_end - input_samples
        interval = interval_for_target(item, raw_index_int)
        if interval is None or input_start < interval.start or input_end > interval.end:
            continue
        if input_start < 0 or input_end > len(record.y):
            continue
        guard_start = max(interval.start, input_start - a1.FOG_GUARD)
        guard_end = min(interval.end, input_end + a1.FOG_GUARD)
        if not record.valid[input_start:input_end].all():
            continue
        if np.any(record.y[guard_start:guard_end]):
            continue
        raw_input = record.x[input_start:input_end].astype(np.float32)
        scaled_input = item.scaler.transform(raw_input)
        target_offset = input_samples - WINDOW
        scaled_target = scaled_input[target_offset:]
        target_center = scaled_target.mean(axis=0, keepdims=True)
        inputs.append(scaled_input - target_center)
        targets.append(scaled_target - target_center)
        kept.append(raw_index_int)
        manifest.append(
            {
                "subject_id": item.subject,
                "record_id": record.record_id,
                "split": str(item.windows.split[raw_index_int]),
                "input_start": input_start,
                "input_end_exclusive": input_end,
                "target_start": target_start,
                "target_end_exclusive": target_end,
                "input_samples": input_samples,
                "history_before_target_samples": input_samples - WINDOW,
                "future_samples_after_target": 0,
                "interval_start": interval.start,
                "interval_end": interval.end,
                "input_within_same_split": True,
                "input_all_valid": True,
                "input_guard_clean_nonfog": True,
            }
        )
    if not inputs:
        raise ValueError(f"{item.subject} has no causal T={input_samples} windows")
    return (
        np.ascontiguousarray(np.stack(inputs).astype(np.float32)),
        np.ascontiguousarray(np.stack(targets).astype(np.float32)),
        np.asarray(kept, dtype=np.int64),
        manifest,
    )


@torch.no_grad()
def evaluate_objective(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    loss_name: str,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_x, batch_y in a1b.pair_loader(inputs, targets, shuffle=False, seed=0, workers=0):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predicted, _ = model(batch_x)
        total += float(waveform_loss(loss_name, predicted, batch_y)) * len(batch_x)
        count += len(batch_x)
    return total / count


def train_model(
    item: a1.PreparedSubject,
    input_samples: int,
    loss_name: str,
    seed: int,
    run_dir: Path,
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
) -> tuple[CausalContextM3, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    required = (run_dir / "best_model.pt", run_dir / "last_model.pt", run_dir / "training_log.csv")
    train_x, train_y, train_indices, train_manifest = causal_context_arrays(
        item, item.train_indices, input_samples
    )
    calibration_x, calibration_y, calibration_indices, calibration_manifest = causal_context_arrays(
        item, item.calibration_indices, input_samples
    )
    model = CausalContextM3(input_samples).to(device)
    if not overwrite and all(path.exists() for path in required):
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        return model, read_csv(run_dir / "training_log.csv"), checkpoint["training"], {
            "train_indices": train_indices,
            "calibration_indices": calibration_indices,
            "train_manifest": train_manifest,
            "calibration_manifest": calibration_manifest,
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(seed)
    model = CausalContextM3(input_samples).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = a1b.pair_loader(train_x, train_y, shuffle=True, seed=seed, workers=workers)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0
    last_train_loss = math.inf
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        maximum_gradient = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(batch_x)
            loss = waveform_loss(loss_name, predicted, batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite zero-shot causal gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
            maximum_gradient = max(maximum_gradient, float(gradient))
        last_train_loss = total / count
        validation_loss = evaluate_objective(
            model, calibration_x, calibration_y, loss_name, device
        )
        improved = validation_loss < best_loss - 1e-8
        if improved:
            best_loss = validation_loss
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
                    "train_loss": last_train_loss,
                    "validation_loss": validation_loss,
                    "max_gradient_norm_before_clip": maximum_gradient,
                    "improved": improved,
                    "bad_epochs": bad_epochs,
                }
            )
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"TRAIN causal {item.subject} {loss_name} T={input_samples} seed={seed} "
                f"epoch={epoch}/{max_epochs} train={last_train_loss:.6g} "
                f"val={validation_loss:.6g} best={best_loss:.6g}@{best_epoch}",
                flush=True,
            )
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("causal training produced no best checkpoint")
    training = {
        "subject_id": item.subject,
        "seed": seed,
        "loss": loss_name,
        "input_samples": input_samples,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_validation_loss": best_loss,
        "last_train_loss": last_train_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(train_x),
        "validation_windows": len(calibration_x),
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
    return model, history, training, {
        "train_indices": train_indices,
        "calibration_indices": calibration_indices,
        "train_manifest": train_manifest,
        "calibration_manifest": calibration_manifest,
    }


def scalar_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert CSV strings and numpy scalars into JSON-safe Python scalars."""
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            output[key] = value.item()
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in ("true", "false"):
                output[key] = stripped.lower() == "true"
            else:
                try:
                    output[key] = float(stripped) if any(c in stripped for c in ".eE") else int(stripped)
                except (ValueError, TypeError):
                    output[key] = value
        else:
            output[key] = value
    return output


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_reused_run(
    source_dir: Path,
    destination: Path,
    *,
    stage: str,
    loss_name: str,
    context_name: str = "W0",
) -> dict[str, Any]:
    """Create an auditable, storage-efficient view of a frozen parent run."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "best_model.pt",
        "last_model.pt",
        "predictions.npz",
        "training_log.csv",
        "window_metrics.csv",
        "channel_metrics.csv",
        "split_manifest.csv",
    ):
        source = source_dir / name
        if source.exists():
            link_or_copy(source, destination / name)
    source_config = read_json(source_dir / "config.json")
    result = read_json(source_dir / "run_metrics.json")
    result.update(
        {
            "experiment": EXPERIMENT,
            "stage": stage,
            "loss": loss_name,
            "context": context_name,
            "input_samples": CONTEXTS[context_name],
            "test_fog_used": False,
            "test_nonfog_calibration_used": False,
            "future_samples_after_target": 0,
            "source": "frozen_parent_run_reuse",
            "source_run_dir": str(source_dir.resolve()),
            "run_dir": str(destination.resolve()),
        }
    )
    config = {
        "experiment": EXPERIMENT,
        "stage": stage,
        "subject_id": result["subject_id"],
        "seed": int(result["seed"]),
        "loss": loss_name,
        "context": context_name,
        "input_samples": CONTEXTS[context_name],
        "target_samples": WINDOW,
        "future_samples_after_target": 0,
        "test_nonfog_calibration": False,
        "test_fog_used_for_selection": False,
        "source_run_dir": str(source_dir.resolve()),
        "source_config_sha256": sha256(source_dir / "config.json"),
        "source_config": source_config,
    }
    base.write_json(destination / "config.json", config)
    base.write_json(destination / "run_metrics.json", result)
    split_rows = read_csv(destination / "split_manifest.csv")
    base.write_json(destination / "split_manifest.json", split_rows)
    base.write_json(
        destination / "context_manifest.json",
        {
            "context": context_name,
            "input_samples": CONTEXTS[context_name],
            "target_samples": WINDOW,
            "history_before_target_samples": 0,
            "future_samples_after_target": 0,
            "same_split_required": True,
            "source": "W0 target-aligned parent windows",
        },
    )
    return result


def inference_latency_ms(
    model: nn.Module, inputs: np.ndarray, targets: np.ndarray, device: torch.device
) -> float:
    sample_count = min(len(inputs), 128)
    tensor = torch.from_numpy(
        np.ascontiguousarray(inputs[:sample_count].transpose(0, 2, 1))
    ).float().to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(10):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / (10 * sample_count)


def save_causal_run(
    run_dir: Path,
    *,
    item: a1.PreparedSubject,
    model: CausalContextM3,
    loss_name: str,
    context_name: str,
    seed: int,
    training: dict[str, Any],
    manifests: dict[str, Any],
    evaluation_manifest: list[dict[str, Any]],
    evaluation_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    latent: np.ndarray,
    result: dict[str, Any],
    window_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
) -> None:
    config = {
        "experiment": EXPERIMENT,
        "stage": result["stage"],
        "subject_id": item.subject,
        "seed": seed,
        "loss": loss_name,
        "context": context_name,
        "input_samples": CONTEXTS[context_name],
        "target_samples": WINDOW,
        "architecture": model.architecture_config(),
        "preprocessing": "training-only RobustScaler followed by target-window centering",
        "test_nonfog_calibration": False,
        "test_fog_used_for_selection": False,
        "decoder_fine_tuning": False,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "maximum_epochs": training["last_epoch"],
        "patience": 100,
    }
    split_rows: list[dict[str, Any]] = []
    for split, indices in (
        ("train_clean_nonfog", manifests["train_indices"]),
        ("early_stopping_clean_nonfog", manifests["calibration_indices"]),
        ("final_evaluation_clean_nonfog", evaluation_indices),
    ):
        split_rows.extend(a1.window_metadata(item.subject, item.records, item.windows, indices, split))
    base.write_json(run_dir / "config.json", config)
    base.write_csv(run_dir / "split_manifest.csv", split_rows)
    base.write_json(run_dir / "split_manifest.json", split_rows)
    context_payload = {
        "policy": {
            "target_split_controls_full_input": True,
            "history_must_remain_in_same_split_interval": True,
            "overlap_across_split_boundary": False,
            "future_samples_after_target": 0,
        },
        "train": manifests["train_manifest"],
        "early_stopping": manifests["calibration_manifest"],
        "evaluation": evaluation_manifest,
    }
    base.write_json(run_dir / "context_manifest.json", context_payload)
    base.write_csv(run_dir / "window_metrics.csv", window_rows)
    base.write_csv(run_dir / "channel_metrics.csv", channel_rows)
    base.write_json(run_dir / "run_metrics.json", result)
    legacy.save_npz(
        run_dir / "predictions.npz", target=actual, reconstruction=predicted, latent=latent
    )


def evaluate_causal_model(
    item: a1.PreparedSubject,
    model: CausalContextM3,
    input_samples: int,
    device: torch.device,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = item.test_indices if indices is None else np.asarray(indices, dtype=np.int64)
    test_x, test_y, kept, manifest = causal_context_arrays(item, selected, input_samples)
    predicted, latent = a1b.predict_pairs(model, test_x, test_y, device)
    template = np.repeat(
        np.mean(item.train_x, axis=0, keepdims=True).astype(np.float32), len(test_y), axis=0
    )
    metrics, arrays, window_rows, channel_rows = a1b.evaluation_artifacts(
        item, kept, test_y, predicted, template
    )
    return metrics, {
        "inputs": test_x,
        "actual": test_y,
        "predicted": predicted,
        "latent": latent,
        "arrays": arrays,
        "kept_indices": kept,
        "manifest": manifest,
        "window_rows": window_rows,
        "channel_rows": channel_rows,
    }


def run_causal_setting(
    root: Path,
    stage: str,
    item: a1.PreparedSubject,
    loss_name: str,
    context_name: str,
    seed: int,
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    evaluation_indices: Sequence[int] | None = None,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    run_dir = root / stage / loss_name / context_name / item.subject / f"seed{seed}"
    metrics_file = run_dir / "run_metrics.json"
    if metrics_file.exists() and not overwrite:
        return read_json(metrics_file)
    if initial_checkpoint is None:
        model, _, training, manifests = train_model(
            item,
            CONTEXTS[context_name],
            loss_name,
            seed,
            run_dir,
            max_epochs=max_epochs,
            patience=patience,
            workers=workers,
            device=device,
            overwrite=overwrite,
        )
    else:
        model = CausalContextM3(CONTEXTS[context_name]).to(device)
        checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        training = dict(checkpoint.get("training", {}))
        training.setdefault("last_epoch", int(training.get("final_epoch", 0)))
        training.setdefault("best_epoch", int(training.get("best_epoch", 0)))
        training.setdefault("best_validation_loss", float(training.get("best_calibration_loss", math.nan)))
        _, _, train_indices, train_manifest = causal_context_arrays(
            item, item.train_indices, CONTEXTS[context_name]
        )
        _, _, calibration_indices, calibration_manifest = causal_context_arrays(
            item, item.calibration_indices, CONTEXTS[context_name]
        )
        manifests = {
            "train_indices": train_indices,
            "calibration_indices": calibration_indices,
            "train_manifest": train_manifest,
            "calibration_manifest": calibration_manifest,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(initial_checkpoint, run_dir / "best_model.pt")
        link_or_copy(initial_checkpoint.parent / "last_model.pt", run_dir / "last_model.pt")
        if (initial_checkpoint.parent / "training_log.csv").exists():
            link_or_copy(initial_checkpoint.parent / "training_log.csv", run_dir / "training_log.csv")
    metrics, payload = evaluate_causal_model(
        item,
        model,
        CONTEXTS[context_name],
        device,
        indices=evaluation_indices,
    )
    latency = inference_latency_ms(model, payload["inputs"], payload["actual"], device)
    result = {
        "experiment": EXPERIMENT,
        "stage": stage,
        "subject_id": item.subject,
        "seed": seed,
        "split_type": item.disclosure["split_type"],
        "test_record_or_block": item.disclosure.get("test_record", item.subject),
        "loss": loss_name,
        "context": context_name,
        "input_samples": CONTEXTS[context_name],
        "history_before_target_samples": CONTEXTS[context_name] - WINDOW,
        "future_samples_after_target": 0,
        "train_windows": len(manifests["train_indices"]),
        "early_stopping_windows": len(manifests["calibration_indices"]),
        "evaluation_windows": len(payload["actual"]),
        "test_nonfog_calibration_used": False,
        "test_fog_used": False,
        **metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_ms_per_window": latency,
        "best_epoch": training.get("best_epoch"),
        "last_epoch": training.get("last_epoch"),
        "best_calibration_loss": training.get("best_validation_loss"),
        "run_dir": str(run_dir.resolve()),
    }
    save_causal_run(
        run_dir,
        item=item,
        model=model,
        loss_name=loss_name,
        context_name=context_name,
        seed=seed,
        training=training,
        manifests=manifests,
        evaluation_manifest=payload["manifest"],
        evaluation_indices=payload["kept_indices"],
        actual=payload["actual"],
        predicted=payload["predicted"],
        latent=payload["latent"],
        result=result,
        window_rows=payload["window_rows"],
        channel_rows=payload["channel_rows"],
    )
    print(
        f"DONE {stage} {item.subject} {loss_name} {context_name} seed={seed} "
        f"corr={result['median_corr']:.3f} nrmse={result['median_nrmse']:.3f} "
        f"p90={result['nrmse_p90']:.3f} pass={result['strict_pass']}",
        flush=True,
    )
    return result


def spectral_entropy(values: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(values.astype(np.float64), axis=1)) ** 2
    frequency = np.fft.rfftfreq(values.shape[1], d=1.0 / FS)
    selected = spectrum[:, (frequency >= 0.5) & (frequency <= 10.0), :]
    probability = selected / np.maximum(selected.sum(axis=1, keepdims=True), 1e-12)
    return (-np.sum(probability * np.log(probability + 1e-12), axis=1)).astype(np.float32)


def bar_figure(
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str,
    path: Path,
    *,
    comparison: Sequence[float] | None = None,
    comparison_label: str = "comparison",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if comparison is None:
        ax.bar(x, values)
    else:
        ax.bar(x - 0.2, values, width=0.4, label="ordinary")
        ax.bar(x + 0.2, comparison, width=0.4, label=comparison_label)
        ax.legend()
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_failure_diagnostics(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    parent_results: Sequence[dict[str, Any]],
    parent_a1b: Path,
    *,
    skip_figures: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse frozen A1 predictions and add every diagnostic requested by the template."""
    parent_summary = {
        str(row["subject_id"]): scalar_row(row)
        for row in read_csv(parent_a1b / "tables" / "A1b0_failure_diagnosis.csv")
    }
    parent_channels = read_csv(parent_a1b / "tables" / "A1b0_channel_diagnostics.csv")
    parent_domain = read_csv(parent_a1b / "tables" / "A1b0_train_test_domain_shift.csv")
    result_lookup = {
        (str(row["subject_id"]), int(row["seed"])): row for row in parent_results
    }
    diagnosis_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        item = prepared[subject]
        summary = parent_summary[subject]
        seed = int(summary["representative_seed"])
        source = Path(result_lookup[(subject, seed)]["run_dir"])
        with np.load(source / "predictions.npz", allow_pickle=False) as payload:
            actual = payload["target"]
            predicted = payload["reconstruction"]
        arrays = base.metric_arrays(actual, predicted)
        lagged, lags = a1b.lagged_metrics(actual, predicted)
        delta = np.mean(
            np.square(np.diff(actual, axis=1) - np.diff(predicted, axis=1)), axis=1
        )
        mae = np.mean(np.abs(actual - predicted), axis=1)
        local_channels = [
            scalar_row(row)
            for row in parent_channels
            if str(row["subject_id"]) == subject
        ]
        for channel_index, row in enumerate(local_channels):
            row.update(
                {
                    "mae": float(np.median(mae[:, channel_index])),
                    "first_difference_mse": float(np.median(delta[:, channel_index])),
                    "representative_seed": seed,
                }
            )
            channel_rows.append(row)
        domain_statistics, domain_arrays = a1b.domain_statistics(item)
        for channel_index, row in enumerate(domain_statistics):
            row = scalar_row(row)
            row.update(
                {
                    "train_spectral_entropy": float(
                        np.median(spectral_entropy(legacy.raw_windows(item.records, item.windows, item.train_indices))[:, channel_index])
                    ),
                    "test_spectral_entropy": float(
                        np.median(spectral_entropy(legacy.raw_windows(item.records, item.windows, item.test_indices))[:, channel_index])
                    ),
                }
            )
            row["spectral_entropy_shift"] = abs(
                float(row["test_spectral_entropy"]) - float(row["train_spectral_entropy"])
            )
            domain_rows.append(row)
        lag_gain = float(np.median(lagged) - np.median(arrays["correlation"]))
        maximum_band_error = max(
            float(np.mean([float(row["spectral_nrmse_0p5_3hz"]) for row in local_channels])),
            float(np.mean([float(row["spectral_nrmse_3_8hz"]) for row in local_channels])),
        )
        frequency_mismatch = maximum_band_error >= 0.80 and float(summary["median_pearson"]) < 0.50
        diagnosis = {
            **summary,
            "lag_gain": lag_gain,
            "median_mae": float(np.median(mae)),
            "median_first_difference_mse": float(np.median(delta)),
            "median_spectral_error_0p5_10hz": float(
                np.median([float(row["spectral_nrmse_0p5_10hz"]) for row in local_channels])
            ),
            "frequency_mismatch_trigger": frequency_mismatch,
            "phase_mismatch_trigger": lag_gain >= 0.12,
            "source_run_dir": str(source),
        }
        diagnosis_rows.append(diagnosis)
        subject_root = root / "E1_failure_diagnostics" / subject
        subject_root.mkdir(parents=True, exist_ok=True)
        base.write_csv(subject_root / "channel_diagnostics.csv", local_channels)
        base.write_csv(
            subject_root / "train_test_domain_shift.csv",
            [row for row in domain_rows if row["subject_id"] == subject],
        )
        if skip_figures:
            continue
        labels = [str(row["channel"]) for row in local_channels]
        ordinary = [float(row["pearson"]) for row in local_channels]
        lagged_values = [float(row["lagged_pearson"]) for row in local_channels]
        bar_figure(labels, ordinary, "Pearson", subject_root / "channel_pearson.png")
        bar_figure(labels, lagged_values, "Lagged Pearson", subject_root / "channel_lagged_pearson.png")
        bar_figure(
            labels,
            ordinary,
            "Pearson",
            subject_root / "pearson_vs_lagged_pearson.png",
            comparison=lagged_values,
            comparison_label="lagged",
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(lags.ravel(), bins=np.arange(-16.5, 17.5, 1), color="#4472C4")
        ax.set_xlabel("best lag (samples)")
        ax.set_ylabel("window-channel count")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(subject_root / "best_lag_distribution.png", dpi=150)
        plt.close(fig)
        for name, key, ylabel in (
            ("channel_nrmse.png", "nrmse", "NRMSE"),
            ("channel_amplitude_ratio.png", "amplitude_ratio", "amplitude ratio"),
            ("channel_delta_error.png", "first_difference_mse", "first-difference MSE"),
        ):
            bar_figure(labels, [float(row[key]) for row in local_channels], ylabel, subject_root / name)
        bar_figure(
            labels,
            [float(row["spectral_nrmse_0p5_3hz"]) for row in local_channels],
            "spectral NRMSE",
            subject_root / "channel_spectral_error.png",
            comparison=[float(row["spectral_nrmse_3_8hz"]) for row in local_channels],
            comparison_label="3-8 Hz",
        )
        local_domain = [row for row in domain_rows if row["subject_id"] == subject]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - 0.3, [float(row["median_shift"]) for row in local_domain], 0.2, label="median")
        ax.bar(x - 0.1, [float(row["scale_shift"]) for row in local_domain], 0.2, label="scale")
        ax.bar(x + 0.1, [abs(math.log((float(row["test_energy_0p5_3hz"]) + 1e-8) / (float(row["train_energy_0p5_3hz"]) + 1e-8))) for row in local_domain], 0.2, label="0.5-3 Hz")
        ax.bar(x + 0.3, [float(row["spectral_entropy_shift"]) for row in local_domain], 0.2, label="entropy")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("robust train-test shift")
        ax.legend(ncol=4)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(subject_root / "train_test_feature_shift.png", dpi=150)
        plt.close(fig)
        worst = int(np.argmin(np.median(arrays["correlation"], axis=1)))
        time_axis = np.arange(WINDOW) / FS
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for channel_index, ax in enumerate(axes):
            ax.plot(time_axis, actual[worst, :, channel_index], label="target")
            ax.plot(time_axis, predicted[worst, :, channel_index], label="reconstruction", alpha=0.85)
            ax.set_ylabel(labels[channel_index])
            ax.grid(alpha=0.2)
        axes[0].legend()
        axes[-1].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(subject_root / "worst_window_waveform.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        image = ax.imshow((actual[worst] - predicted[worst]).T, aspect="auto", cmap="coolwarm")
        ax.set_yticks(range(CHANNELS), labels)
        ax.set_xlabel("sample")
        fig.colorbar(image, ax=ax, label="residual")
        fig.tight_layout()
        fig.savefig(subject_root / "worst_window_residual_heatmap.png", dpi=150)
        plt.close(fig)
        failure_labels = np.where(
            np.median(arrays["correlation"], axis=1) < 0.3,
            "shape",
            np.where(np.median(arrays["amplitude_ratio"], axis=1) < 0.65, "amplitude", "boundary/pass"),
        )
        names, counts = np.unique(failure_labels, return_counts=True)
        bar_figure(names.tolist(), counts.tolist(), "window count", subject_root / "window_failure_type.png")
    base.write_csv(root / "tables" / "E1_failure_diagnostics.csv", diagnosis_rows)
    base.write_csv(root / "tables" / "E1_channel_diagnostics.csv", channel_rows)
    base.write_csv(root / "tables" / "E1_train_test_domain_shift.csv", domain_rows)
    triggered = [str(row["subject_id"]) for row in diagnosis_rows if row["frequency_mismatch_trigger"]]
    base.write_json(
        root / "reports" / "E1_frequency_trigger.json",
        {
            "L5_triggered": bool(triggered),
            "trigger_subjects": triggered,
            "rule": "channel-mean 0.5-3 or 3-8 Hz spectral NRMSE >= 0.80 and Pearson < 0.50",
            "selection_screen_subjects": list(LOSS_SCREEN_SUBJECTS),
        },
    )
    if not skip_figures:
        (root / "figures").mkdir(parents=True, exist_ok=True)
        matrix_names = ["phase", "amplitude/scale", "local waveform", "frequency", "global domain"]
        matrix = []
        for row in diagnosis_rows:
            failure = str(row["failure_type"]).lower()
            matrix.append(
                [
                    float(row["phase_mismatch_trigger"]),
                    float(float(row["median_amplitude_ratio"]) < 0.75 or float(row["median_amplitude_ratio"]) > 1.25),
                    float(float(row["median_pearson"]) < 0.50 and float(row["median_first_difference_mse"]) > 0.1),
                    float(row["frequency_mismatch_trigger"]),
                    float("global" in failure or float(row["median_shift"]) >= 0.8),
                ]
            )
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(np.asarray(matrix), cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(matrix_names)), matrix_names, rotation=30, ha="right")
        ax.set_yticks(range(len(SUBJECTS)), SUBJECTS)
        fig.colorbar(image, ax=ax, label="diagnostic flag")
        fig.tight_layout()
        fig.savefig(root / "figures" / "all_subject_failure_type_matrix.png", dpi=150)
        plt.close(fig)
        bar_figure(
            list(SUBJECTS),
            [float(row["lag_gain"]) for row in diagnosis_rows],
            "Lagged Pearson - Pearson",
            root / "figures" / "all_subject_lag_gain.png",
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in diagnosis_rows:
            ax.scatter(float(row["median_shift"]), float(row["median_nrmse"]), s=60)
            ax.text(float(row["median_shift"]), float(row["median_nrmse"]), str(row["subject_id"]))
        ax.set_xlabel("median train-test domain shift")
        ax.set_ylabel("test NRMSE")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "figures" / "all_subject_domain_shift_vs_nrmse.png", dpi=150)
        plt.close(fig)
    return diagnosis_rows, channel_rows, domain_rows


def result_rank(rows: Sequence[dict[str, Any]]) -> tuple[float, ...]:
    lag_gaps = [float(row["median_lagged_pearson"]) - float(row["median_corr"]) for row in rows]
    amplitude_error = [abs(float(row["median_amplitude_ratio"]) - 1.0) for row in rows]
    return (
        float(np.median([float(row["median_corr"]) for row in rows])),
        -float(np.median([float(row["median_nrmse"]) for row in rows])),
        -float(np.median([float(row["nrmse_p90"]) for row in rows])),
        -float(np.median(lag_gaps)),
        -float(np.median(amplitude_error)),
        float(sum(as_bool(row["strict_pass"]) for row in rows)),
    )


def grouped_metric_plot(
    rows: Sequence[dict[str, Any]],
    setting_key: str,
    settings: Sequence[str],
    metric: str,
    path: Path,
    *,
    subjects: Sequence[str] = LOSS_SCREEN_SUBJECTS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for subject in subjects:
        values = []
        for setting in settings:
            selected = [
                float(row[metric])
                for row in rows
                if str(row["subject_id"]) == subject and str(row[setting_key]) == setting
            ]
            values.append(float(np.median(selected)) if selected else math.nan)
        ax.plot(settings, values, marker="o", label=subject)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pass_matrix_plot(
    rows: Sequence[dict[str, Any]],
    setting_key: str,
    settings: Sequence[str],
    subjects: Sequence[str],
    path: Path,
) -> None:
    matrix = np.zeros((len(subjects), len(settings)), dtype=float)
    for y, subject in enumerate(subjects):
        for x, setting in enumerate(settings):
            values = [
                as_bool(row["strict_pass"])
                for row in rows
                if str(row["subject_id"]) == subject and str(row[setting_key]) == setting
            ]
            matrix[y, x] = float(np.mean(values)) if values else np.nan
    fig, ax = plt.subplots(figsize=(max(7, len(settings) * 1.2), max(4, len(subjects) * 0.65)))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(settings)), settings, rotation=30, ha="right")
    ax.set_yticks(range(len(subjects)), subjects)
    fig.colorbar(image, ax=ax, label="PASS fraction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def materialize_parent_loss_screen(
    root: Path, parent_a1b: Path, losses: Sequence[str]
) -> list[dict[str, Any]]:
    parent_rows = read_csv(parent_a1b / "tables" / "A1b2_loss_screen.csv")
    rows: list[dict[str, Any]] = []
    for raw in parent_rows:
        if str(raw["loss"]) not in losses or str(raw["subject_id"]) not in LOSS_SCREEN_SUBJECTS:
            continue
        row = scalar_row(raw)
        source = Path(str(row["run_dir"]))
        destination = (
            root
            / "E3_waveform_loss"
            / str(row["loss"])
            / "W0"
            / str(row["subject_id"])
            / f"seed{int(row['seed'])}"
        )
        rows.append(
            materialize_reused_run(
                source,
                destination,
                stage="E3_waveform_loss_screen",
                loss_name=str(row["loss"]),
            )
        )
    return rows


def run_loss_ablation(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    parent_a1b: Path,
    frequency_trigger: dict[str, Any],
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    # L0/L2/L3/L4 are byte-for-byte frozen C0/W0 parent runs. L1 is rerun with
    # the template's per-window-channel normalization and epsilon=1e-3.
    screen_rows = materialize_parent_loss_screen(root, parent_a1b, ("L0", "L2", "L3", "L4"))
    for subject in LOSS_SCREEN_SUBJECTS:
        screen_rows.append(
            run_causal_setting(
                root,
                "E3_waveform_loss",
                prepared[subject],
                "L1",
                "W0",
                SEEDS[0],
                max_epochs=max_epochs,
                patience=patience,
                workers=workers,
                device=device,
                overwrite=overwrite,
            )
        )
    if bool(frequency_trigger["L5_triggered"]):
        for subject in LOSS_SCREEN_SUBJECTS:
            screen_rows.append(
                run_causal_setting(
                    root,
                    "E3_waveform_loss",
                    prepared[subject],
                    "L5",
                    "W0",
                    SEEDS[0],
                    max_epochs=max_epochs,
                    patience=patience,
                    workers=workers,
                    device=device,
                    overwrite=overwrite,
                )
            )
    else:
        base.write_json(
            root / "E3_waveform_loss" / "L5_freq_optional" / "status.json",
            {"status": "NOT RUN", "reason": "E1 did not meet the frozen F4 frequency trigger"},
        )
    for row in screen_rows:
        row["lag_gain"] = float(row["median_lagged_pearson"]) - float(row["median_corr"])
    candidate_losses = sorted({str(row["loss"]) for row in screen_rows})
    grouped = {
        loss: [row for row in screen_rows if str(row["loss"]) == loss]
        for loss in candidate_losses
    }
    eligible = [
        loss
        for loss, rows in grouped.items()
        if any(as_bool(row["strict_pass"]) for row in rows)
    ]
    if not eligible:
        eligible = ["L0"]
    selected_loss = max(eligible, key=lambda loss: result_rank(grouped[loss]))
    base.write_csv(root / "tables" / "E3_loss_screen.csv", screen_rows)
    # The frozen selected loss is now verified on the two difficult cases and
    # formally expanded to all seven subjects and all three seeds.
    full_rows: list[dict[str, Any]] = []
    if selected_loss == "L4":
        for raw in read_csv(parent_a1b / "tables" / "A1_retest_run_metrics.csv"):
            row = scalar_row(raw)
            source = Path(str(row["run_dir"]))
            destination = (
                root
                / "E3_waveform_loss"
                / selected_loss
                / "full_retest"
                / str(row["subject_id"])
                / f"seed{int(row['seed'])}"
            )
            reused = materialize_reused_run(
                source,
                destination,
                stage="E3_waveform_loss_full_retest",
                loss_name=selected_loss,
            )
            item = prepared[str(row["subject_id"])]
            reused.update(
                {
                    "split_type": item.disclosure["split_type"],
                    "test_record_or_block": item.disclosure.get("test_record", item.subject),
                }
            )
            base.write_json(destination / "run_metrics.json", reused)
            full_rows.append(reused)
    else:
        for subject in SUBJECTS:
            for seed in SEEDS:
                full_rows.append(
                    run_causal_setting(
                        root,
                        "E3_waveform_loss_full_retest",
                        prepared[subject],
                        selected_loss,
                        "W0",
                        seed,
                        max_epochs=max_epochs,
                        patience=patience,
                        workers=workers,
                        device=device,
                        overwrite=overwrite,
                    )
                )
    gate = a1.evaluate_a1_gate(full_rows)
    passing_original = {
        row["subject_id"]: bool(row["subject_pass"])
        for row in gate["subject_summary"]
        if row["subject_id"] in ORIGINAL_PASS_SUBJECTS
    }
    decision = {
        "selected_loss": selected_loss,
        "eligible_losses": eligible,
        "screen_ranks": {loss: result_rank(rows) for loss, rows in grouped.items()},
        "selection_subjects": list(LOSS_SCREEN_SUBJECTS),
        "difficult_case_verification_subjects": ["S02", "S06"],
        "full_retest_gate": gate,
        "original_pass_group_failures": sum(not value for value in passing_original.values()),
        "selection_rule": "Pearson, NRMSE, P90, lag gap, amplitude, then PASS count; at least one of S05/S09 must PASS",
        "L1_epsilon": 1e-3,
        "L5_conditionally_triggered": bool(frequency_trigger["L5_triggered"]),
    }
    base.write_csv(root / "tables" / "E3_selected_loss_full_retest.csv", full_rows)
    base.write_json(root / "reports" / "E3_selected_loss.json", decision)
    if not skip_figures:
        figures = root / "figures"
        for filename, metric in (
            ("loss_ablation_pearson.png", "median_corr"),
            ("loss_ablation_nrmse.png", "median_nrmse"),
            ("loss_ablation_p90.png", "nrmse_p90"),
            ("loss_ablation_lag_gain.png", "lag_gain"),
            ("loss_ablation_amplitude_ratio.png", "median_amplitude_ratio"),
        ):
            grouped_metric_plot(screen_rows, "loss", candidate_losses, metric, figures / filename)
        pass_matrix_plot(
            screen_rows,
            "loss",
            candidate_losses,
            LOSS_SCREEN_SUBJECTS,
            figures / "loss_ablation_pass_matrix.png",
        )
        representative = next(
            row for row in full_rows if row["subject_id"] == "S05" and int(row["seed"]) == SEEDS[0]
        )
        with np.load(Path(representative["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
            a1.plot_best_median_worst(
                payload["target"],
                payload["reconstruction"],
                figures / "loss_ablation_best_worst_waveform.png",
                prepared["S05"].channel_names,
            )
        matrix = np.full((len(SUBJECTS), CHANNELS), np.nan)
        for y, subject in enumerate(SUBJECTS):
            run = next(row for row in full_rows if row["subject_id"] == subject and int(row["seed"]) == SEEDS[0])
            channels = read_csv(Path(run["run_dir"]) / "channel_metrics.csv")
            matrix[y] = [float(row["pearson"]) for row in channels]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(CHANNELS), prepared["S01"].channel_names, rotation=35, ha="right")
        ax.set_yticks(range(len(SUBJECTS)), SUBJECTS)
        fig.colorbar(image, ax=ax, label="Pearson")
        fig.tight_layout()
        fig.savefig(figures / "loss_ablation_channel_heatmap.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        for subject in SUBJECTS:
            values = [float(row["median_corr"]) for row in full_rows if row["subject_id"] == subject]
            ax.plot(range(len(SEEDS)), values, marker="o", label=subject)
        ax.set_xticks(range(len(SEEDS)), [str(seed) for seed in SEEDS])
        ax.set_ylabel("Pearson")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "loss_ablation_seed_stability.png", dpi=150)
        plt.close(fig)
        history = read_csv(Path(representative["run_dir"]) / "training_log.csv")
        a1b.plot_training_history(history, figures / "loss_ablation_training_curve.png")
    return selected_loss, full_rows, decision


def run_context_ablation(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    parent_a1b: Path,
    selected_loss: str,
    selected_loss_rows: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    lag_extra = [
        str(row["subject_id"])
        for row in diagnostics
        if str(row["subject_id"]) in ("S02", "S06") and float(row["lag_gain"]) >= 0.12
    ]
    screen_subjects = tuple(dict.fromkeys((*LOSS_SCREEN_SUBJECTS, *lag_extra)))
    screen_rows: list[dict[str, Any]] = []
    common_support_rows: list[dict[str, Any]] = []
    for subject in screen_subjects:
        item = prepared[subject]
        _, _, common_indices, common_manifest = causal_context_arrays(
            item, item.test_indices, CONTEXTS["W2_causal"]
        )
        common_support_rows.extend(common_manifest)
        source_run = next(
            row
            for row in selected_loss_rows
            if str(row["subject_id"]) == subject and int(row["seed"]) == SEEDS[0]
        )
        screen_rows.append(
            run_causal_setting(
                root,
                "E4_causal_context",
                item,
                selected_loss,
                "W0",
                SEEDS[0],
                max_epochs=max_epochs,
                patience=patience,
                workers=workers,
                device=device,
                overwrite=overwrite,
                evaluation_indices=common_indices,
                initial_checkpoint=Path(str(source_run["run_dir"])) / "best_model.pt",
            )
        )
        for context_name in ("W1_causal", "W2_causal"):
            screen_rows.append(
                run_causal_setting(
                    root,
                    "E4_causal_context",
                    item,
                    selected_loss,
                    context_name,
                    SEEDS[0],
                    max_epochs=max_epochs,
                    patience=patience,
                    workers=workers,
                    device=device,
                    overwrite=overwrite,
                    evaluation_indices=common_indices,
                )
            )
    for row in screen_rows:
        row["lag_gain"] = float(row["median_lagged_pearson"]) - float(row["median_corr"])
    base.write_csv(root / "tables" / "E4_context_common_support_manifest.csv", common_support_rows)
    base.write_csv(root / "tables" / "E4_context_screen.csv", screen_rows)
    grouped = {
        context: [row for row in screen_rows if str(row["context"]) == context]
        for context in CONTEXTS
    }
    baseline = {str(row["subject_id"]): row for row in grouped["W0"]}
    eligibility: dict[str, dict[str, Any]] = {
        "W0": {"eligible": True, "reason": "causal 2-second baseline"}
    }
    eligible = ["W0"]
    for context_name in ("W1_causal", "W2_causal"):
        candidates = grouped[context_name]
        improved_subjects = [
            str(row["subject_id"])
            for row in candidates
            if float(row["median_corr"]) > float(baseline[str(row["subject_id"])]["median_corr"])
        ]
        baseline_rows = list(baseline.values())
        corr_better = np.median([float(row["median_corr"]) for row in candidates]) > np.median(
            [float(row["median_corr"]) for row in baseline_rows]
        )
        nrmse_ok = np.median([float(row["median_nrmse"]) for row in candidates]) <= np.median(
            [float(row["median_nrmse"]) for row in baseline_rows]
        )
        p90_ok = np.median([float(row["nrmse_p90"]) for row in candidates]) <= np.median(
            [float(row["nrmse_p90"]) for row in baseline_rows]
        )
        lag_gap_ok = np.median(
            [float(row["median_lagged_pearson"]) - float(row["median_corr"]) for row in candidates]
        ) <= np.median(
            [float(row["median_lagged_pearson"]) - float(row["median_corr"]) for row in baseline_rows]
        )
        best_lag_ok = np.median([float(row["median_abs_best_lag"]) for row in candidates]) <= np.median(
            [float(row["median_abs_best_lag"]) for row in baseline_rows]
        )
        pass_ok = sum(as_bool(row["strict_pass"]) for row in candidates) >= sum(
            as_bool(row["strict_pass"]) for row in baseline_rows
        )
        is_eligible = (
            len(improved_subjects) >= 2
            and corr_better
            and nrmse_ok
            and p90_ok
            and lag_gap_ok
            and best_lag_ok
            and pass_ok
        )
        eligibility[context_name] = {
            "eligible": is_eligible,
            "improved_failed_subjects": improved_subjects,
            "at_least_two_failed_subjects_improved": len(improved_subjects) >= 2,
            "aggregate_pearson_improved": bool(corr_better),
            "aggregate_nrmse_not_worse": bool(nrmse_ok),
            "aggregate_p90_not_worse": bool(p90_ok),
            "lag_gap_not_worse": bool(lag_gap_ok),
            "absolute_best_lag_not_worse": bool(best_lag_ok),
            "screen_pass_count_not_lower": bool(pass_ok),
        }
        if is_eligible:
            eligible.append(context_name)
    selected_context = max(eligible, key=lambda context: result_rank(grouped[context]))
    full_rows: list[dict[str, Any]] = []
    if selected_context == "W0":
        for row in selected_loss_rows:
            full_rows.append(
                {
                    **row,
                    "stage": "E4_selected_context_full_retest",
                    "context": "W0",
                    "input_samples": WINDOW,
                    "history_before_target_samples": 0,
                    "future_samples_after_target": 0,
                    "source": "E3 selected-loss W0 reuse",
                }
            )
    else:
        for subject in SUBJECTS:
            for seed in SEEDS:
                full_rows.append(
                    run_causal_setting(
                        root,
                        "E4_selected_context_full_retest",
                        prepared[subject],
                        selected_loss,
                        selected_context,
                        seed,
                        max_epochs=max_epochs,
                        patience=patience,
                        workers=workers,
                        device=device,
                        overwrite=overwrite,
                    )
                )
    full_gate = a1.evaluate_a1_gate(full_rows)
    # Existing centered context is retained only as an explicitly non-causal upper bound.
    upper_bound_rows: list[dict[str, Any]] = []
    if selected_loss == "L4":
        for raw in read_csv(parent_a1b / "tables" / "A1b3_context_screen.csv"):
            row = scalar_row(raw)
            if str(row["subject_id"]) in screen_subjects and str(row["context"]) == "W1":
                upper_bound_rows.append(
                    {
                        **row,
                        "context": "W1_center_upper_bound",
                        "causal": False,
                        "future_samples_after_target": 64,
                        "selection_eligible": False,
                        "source": "prior centered-context experiment",
                    }
                )
        if upper_bound_rows:
            base.write_csv(root / "tables" / "E4_noncausal_upper_bound.csv", upper_bound_rows)
    decision = {
        "screen_subjects": list(screen_subjects),
        "fair_comparison_support": "intersection induced by W2 causal history; identical target indices per subject",
        "selected_context": selected_context,
        "input_samples": CONTEXTS[selected_context],
        "eligible_contexts": eligible,
        "eligibility_audit": eligibility,
        "full_retest_gate": full_gate,
        "noncausal_upper_bound_selection_eligible": False,
        "causal_definition": "input_end == target_end and future_samples_after_target == 0",
    }
    base.write_csv(root / "tables" / "E4_selected_context_full_retest.csv", full_rows)
    base.write_json(root / "reports" / "E4_selected_context.json", decision)
    if not skip_figures:
        figures = root / "figures"
        settings = list(CONTEXTS)
        for filename, metric in (
            ("context_pearson.png", "median_corr"),
            ("context_lagged_pearson.png", "median_lagged_pearson"),
            ("context_lag_gain.png", "lag_gain"),
            ("context_best_lag.png", "median_abs_best_lag"),
            ("context_nrmse.png", "median_nrmse"),
            ("context_p90.png", "nrmse_p90"),
            ("context_parameter_count.png", "parameter_count"),
            ("context_inference_latency.png", "inference_ms_per_window"),
        ):
            grouped_metric_plot(
                screen_rows,
                "context",
                settings,
                metric,
                figures / filename,
                subjects=screen_subjects,
            )
        pass_matrix_plot(
            screen_rows,
            "context",
            settings,
            screen_subjects,
            figures / "context_pass_matrix.png",
        )
        representative_runs = [
            row for row in screen_rows if str(row["subject_id"]) == screen_subjects[0]
        ]
        fig, axes = plt.subplots(len(representative_runs), 1, figsize=(11, 3 * len(representative_runs)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, row in zip(axes, representative_runs):
            with np.load(Path(row["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
                actual = payload["target"]
                predicted = payload["reconstruction"]
                idx = int(np.argsort(np.median(base.metric_arrays(actual, predicted)["correlation"], axis=1))[len(actual) // 2])
                ax.plot(actual[idx, :, 0], label="target")
                ax.plot(predicted[idx, :, 0], label=str(row["context"]))
                ax.legend()
                ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "context_waveform_comparison.png", dpi=150)
        plt.close(fig)
        if upper_bound_rows:
            combined = [*screen_rows, *upper_bound_rows]
            grouped_metric_plot(
                combined,
                "context",
                ["W0", "W1_causal", "W1_center_upper_bound"],
                "median_corr",
                figures / "causal_vs_noncausal_upper_bound.png",
                subjects=screen_subjects,
            )
    return selected_context, full_rows, decision, upper_bound_rows


def add_lagged_metrics(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    if "median_lagged_pearson" not in output:
        with np.load(Path(str(output["run_dir"])) / "predictions.npz", allow_pickle=False) as payload:
            lagged, lags = a1b.lagged_metrics(payload["target"], payload["reconstruction"])
        output.update(
            {
                "median_lagged_pearson": float(np.median(lagged)),
                "median_best_lag": float(np.median(lags)),
                "median_abs_best_lag": float(np.median(np.abs(lags))),
            }
        )
    output["lag_gain"] = float(output["median_lagged_pearson"]) - float(output["median_corr"])
    return output


def materialize_final_candidate(
    root: Path,
    candidate: str,
    rows: Sequence[dict[str, Any]],
    *,
    loss_name: str,
    context_name: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = add_lagged_metrics(raw)
        source = Path(str(row["run_dir"]))
        destination = (
            root
            / "final_combination"
            / candidate
            / str(row["subject_id"])
            / f"seed{int(row['seed'])}"
        )
        reused = materialize_reused_run(
            source,
            destination,
            stage=f"final_combination_{candidate}",
            loss_name=loss_name,
            context_name=context_name,
        )
        reused.update(
            {
                "median_lagged_pearson": row["median_lagged_pearson"],
                "median_best_lag": row["median_best_lag"],
                "median_abs_best_lag": row["median_abs_best_lag"],
                "lag_gain": row["lag_gain"],
                "candidate": candidate,
            }
        )
        base.write_json(destination / "run_metrics.json", reused)
        output.append(reused)
    return output


def candidate_summary(candidate: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gate = a1.evaluate_a1_gate(rows)
    return {
        "candidate": candidate,
        "loss": str(rows[0]["loss"]),
        "context": str(rows[0]["context"]),
        "run_pass_count": gate["run_pass_count"],
        "run_total": gate["run_total"],
        "subject_pass_count": gate["subject_pass_count"],
        "subject_total": gate["subject_total"],
        "waveform_collapse_subject_count": gate["waveform_collapse_subject_count"],
        "median_pearson": float(np.median([float(row["median_corr"]) for row in rows])),
        "median_lagged_pearson": float(np.median([float(row["median_lagged_pearson"]) for row in rows])),
        "median_lag_gain": float(np.median([float(row["lag_gain"]) for row in rows])),
        "median_nrmse": float(np.median([float(row["median_nrmse"]) for row in rows])),
        "median_p90": float(np.median([float(row["nrmse_p90"]) for row in rows])),
        "status": gate["status"],
        "gate": gate,
    }


def run_final_combination(
    root: Path,
    prepared: dict[str, a1.PreparedSubject],
    parent_results: Sequence[dict[str, Any]],
    selected_loss: str,
    selected_loss_rows: Sequence[dict[str, Any]],
    selected_context: str,
    selected_context_rows: Sequence[dict[str, Any]],
    *,
    max_epochs: int,
    patience: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    skip_figures: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    f0_source = [add_lagged_metrics(row) for row in parent_results]
    if selected_context == "W0":
        f2_source = f0_source
        f3_source = list(selected_loss_rows)
    else:
        f2_source: list[dict[str, Any]] = []
        for subject in SUBJECTS:
            for seed in SEEDS:
                f2_source.append(
                    run_causal_setting(
                        root,
                        "final_combination_F2_training",
                        prepared[subject],
                        "L0",
                        selected_context,
                        seed,
                        max_epochs=max_epochs,
                        patience=patience,
                        workers=workers,
                        device=device,
                        overwrite=overwrite,
                    )
                )
        f3_source = list(selected_context_rows)
    candidate_rows = {
        "F0_baseline": materialize_final_candidate(
            root, "F0_baseline", f0_source, loss_name="L0", context_name="W0"
        ),
        "F1_best_loss": materialize_final_candidate(
            root,
            "F1_best_loss",
            selected_loss_rows,
            loss_name=selected_loss,
            context_name="W0",
        ),
        "F2_best_context": materialize_final_candidate(
            root,
            "F2_best_context",
            f2_source,
            loss_name="L0",
            context_name=selected_context,
        ),
        "F3_combined": materialize_final_candidate(
            root,
            "F3_combined",
            f3_source,
            loss_name=selected_loss,
            context_name=selected_context,
        ),
    }
    summaries = [candidate_summary(name, rows) for name, rows in candidate_rows.items()]
    baseline = summaries[0]
    baseline_subjects = {
        row["subject_id"]: bool(row["subject_pass"])
        for row in baseline["gate"]["subject_summary"]
    }
    for summary in summaries:
        subjects = {
            row["subject_id"]: bool(row["subject_pass"])
            for row in summary["gate"]["subject_summary"]
        }
        original_pass_group_degradation = sum(
            baseline_subjects[subject] and not subjects[subject]
            for subject in ORIGINAL_PASS_SUBJECTS
        )
        summary.update(
            {
                "original_pass_group_degradation_count": original_pass_group_degradation,
                "p90_below_F0": summary["median_p90"] < baseline["median_p90"] - 1e-12,
                "lag_gap_below_F0": summary["median_lag_gain"] < baseline["median_lag_gain"] - 1e-12,
                "secondary_constraints_pass": (
                    original_pass_group_degradation <= 1
                    and summary["waveform_collapse_subject_count"] <= 2
                    and (
                        summary["candidate"] == "F0_baseline"
                        or (
                            summary["median_p90"] < baseline["median_p90"]
                            and summary["median_lag_gain"] <= baseline["median_lag_gain"]
                        )
                    )
                ),
            }
        )
    eligible = [
        summary
        for summary in summaries[1:]
        if summary["status"] == "PASS" and summary["secondary_constraints_pass"]
    ]
    if eligible:
        # Prefer the simpler independent module when F1 and F3 are numerically identical (W0).
        selected = max(
            eligible,
            key=lambda row: (
                int(row["subject_pass_count"]),
                int(row["run_pass_count"]),
                float(row["median_pearson"]),
                -float(row["median_nrmse"]),
                -int(row["candidate"] == "F3_combined" and selected_context == "W0"),
            ),
        )
    else:
        selected = baseline
    if selected["candidate"] == "F0_baseline":
        positioning = "P0_current + M3_tcdae_long remains a partial zero-shot baseline, not a stable NBM"
    elif selected["candidate"] == "F1_best_loss":
        positioning = "zero-shot waveform-constrained deterministic NBM"
    elif selected["candidate"] == "F2_best_context":
        positioning = "zero-shot causal long-context deterministic NBM"
    else:
        positioning = "zero-shot causal waveform-constrained normal-behavior model"
    selected_rows = candidate_rows[str(selected["candidate"])]
    subject_p95 = {
        subject: float(np.median([float(row.get("nrmse_p95", math.nan)) for row in selected_rows if row["subject_id"] == subject]))
        for subject in SUBJECTS
    }
    tail_risk_subjects = [subject for subject, value in subject_p95.items() if np.isfinite(value) and value > 1.50]
    decision = {
        "selected_candidate": selected["candidate"],
        "selected_loss": selected_loss if selected["candidate"] in ("F1_best_loss", "F3_combined") else "L0",
        "selected_context": selected_context if selected["candidate"] in ("F2_best_context", "F3_combined") else "W0",
        "formal_status": selected["status"],
        "reported_status": "CONDITIONAL PASS" if selected["status"] == "PASS" and tail_risk_subjects else selected["status"],
        "positioning": positioning,
        "eligible_for_final_residual_validation": selected["subject_pass_count"] >= 5,
        "subject_pass_count": selected["subject_pass_count"],
        "run_pass_count": selected["run_pass_count"],
        "tail_risk_subjects_nrmse_p95_gt_1p50": tail_risk_subjects,
        "subject_median_nrmse_p95": subject_p95,
        "candidate_summaries": summaries,
        "selection_note": "Only independently eligible loss/context modules were combined; a centered future-looking context was excluded.",
    }
    all_rows = [row for rows in candidate_rows.values() for row in rows]
    base.write_csv(root / "tables" / "final_combination_run_metrics.csv", all_rows)
    base.write_csv(
        root / "tables" / "final_combination_summary.csv",
        [{key: value for key, value in row.items() if key != "gate"} for row in summaries],
    )
    base.write_json(root / "reports" / "final_decision.json", decision)
    if not skip_figures:
        settings = [summary["candidate"] for summary in summaries]
        figures = root / "figures"
        for filename, metric in (
            ("final_combination_pearson.png", "median_pearson"),
            ("final_combination_nrmse.png", "median_nrmse"),
            ("final_combination_p90.png", "median_p90"),
            ("final_combination_subject_passes.png", "subject_pass_count"),
        ):
            bar_figure(settings, [float(row[metric]) for row in summaries], metric, figures / filename)
    return all_rows, decision


def write_protocol(
    root: Path,
    args: argparse.Namespace,
    prepared: dict[str, a1.PreparedSubject],
) -> None:
    protocol = root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    if args.template.exists():
        shutil.copy2(args.template, protocol / "experiment_template.md")
    files = [
        args.template,
        Path(__file__),
        REPO_ROOT / "configs" / "daphnet_nbm_zero_shot_causal_waveform_context.yaml",
        REPO_ROOT / "scripts" / "run_daphnet_nbm_routeA_final_residual_validation.py",
        REPO_ROOT / "scripts" / "run_daphnet_nbm_routeA_A1b_generalization_repair.py",
    ]
    hashes = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in files if path.exists()
    ]
    base.write_csv(protocol / "input_hashes.csv", hashes)
    preflight: list[dict[str, Any]] = []
    for subject, item in prepared.items():
        row: dict[str, Any] = {
            "subject_id": subject,
            "split_type": item.disclosure["split_type"],
            "train_W0": len(item.train_indices),
            "validation_W0": len(item.calibration_indices),
            "test_W0": len(item.test_indices),
        }
        for context_name in ("W1_causal", "W2_causal"):
            input_samples = CONTEXTS[context_name]
            for split, indices in (
                ("train", item.train_indices),
                ("validation", item.calibration_indices),
                ("test", item.test_indices),
            ):
                _, _, kept, manifest = causal_context_arrays(item, indices, input_samples)
                if any(not entry["input_within_same_split"] or entry["future_samples_after_target"] for entry in manifest):
                    raise AssertionError("causal preflight found boundary leakage")
                row[f"{split}_{context_name}"] = len(kept)
        preflight.append(row)
    base.write_csv(protocol / "causal_preflight.csv", preflight)
    frozen = {
        "experiment": EXPERIMENT,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "parent_a1": str(args.parent_a1),
        "parent_a1b": str(args.parent_a1b),
        "test_nonfog_calibration": False,
        "test_fog_used_for_training_selection_or_preprocessing": False,
        "decoder_fine_tuning": False,
        "future_information_in_main_experiments": False,
        "context_common_support_policy": "W0/W1/W2 screen evaluated on W2-eligible target intersection",
        "loss_selection": "S05/S09 seed 20260802; full 7x3 verification",
        "context_selection": "S05/S09 plus only S02/S06 with lag gain >=0.12; full 7x3 verification if selected",
        "formal_gate": "unchanged A1: each subject >=2/3 run PASS; overall >=5/7; collapse subjects <=2",
        "losses": list(LOSSES),
        "contexts": CONTEXTS,
    }
    base.write_json(protocol / "frozen_protocol.json", frozen)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(output)


def write_reports(
    root: Path,
    diagnostics: Sequence[dict[str, Any]],
    loss_decision: dict[str, Any] | None,
    context_decision: dict[str, Any] | None,
    final_decision: dict[str, Any] | None,
) -> None:
    e1_rows = [
        (
            row["subject_id"],
            f"{float(row['median_pearson']):.3f}",
            f"{float(row['median_lagged_pearson']):.3f}",
            f"{float(row['lag_gain']):.3f}",
            f"{float(row['median_nrmse']):.3f}",
            f"{float(row['median_amplitude_ratio']):.3f}",
            row["worst_channel"],
            row["failure_type"],
        )
        for row in diagnostics
    ]
    trigger = read_json(root / "reports" / "E1_frequency_trigger.json")
    e1 = f"""# 实验 1：Zero-shot 未见 Non-FoG 失效诊断

本阶段复用原 A1 冻结模型与预测，不重新训练，也未使用测试记录统计量做校准。最大时滞范围固定为 ±16 点。

{markdown_table(['被试','Pearson','Lagged Pearson','Lag Gain','NRMSE','振幅比','最差通道','主失败类型'], e1_rows)}

结论：S02、S06 的普通与 lagged Pearson 均低，主要不是单纯相位偏移；S05、S09 仅有有限 lag gain，仍以局部波形/混合失配为主。F4 条件触发为 `{trigger['L5_triggered']}`，触发被试为 `{', '.join(trigger['trigger_subjects']) or '无'}`；L5 {'进入 S05/S09 条件筛选' if trigger['L5_triggered'] else '按预注册规则不运行'}。
"""
    write_text(root / "reports" / "E1_failure_diagnostics_report.md", e1)
    if loss_decision is not None:
        gate = loss_decision["full_retest_gate"]
        e3 = f"""# 实验 3：波形损失消融

- 筛选被试：S05、S09；固定种子：20260802。
- L0/L2/L3/L4 复用严格 C0/W0 的冻结运行；L1 按模板采用逐窗口逐通道 NMSE 与 `epsilon=1e-3` 重新训练。
- L5 条件触发：{loss_decision['L5_conditionally_triggered']}。
- 冻结最佳损失：**{loss_decision['selected_loss']}**。
- 全量复测：运行 {gate['run_pass_count']}/{gate['run_total']}，被试 {gate['subject_pass_count']}/7，正式门控 **{gate['status']}**。
- 原通过组退化数：{loss_decision['original_pass_group_failures']}（要求不超过 1）。

最佳损失随后在 S02、S06 困难病例及全部 7 被试 × 3 种子上复核；选择未使用测试 FoG。
"""
        write_text(root / "reports" / "E3_waveform_loss_report.md", e3)
    if context_decision is not None:
        gate = context_decision["full_retest_gate"]
        e4 = f"""# 实验 4：严格因果长上下文

- 筛选被试：{', '.join(context_decision['screen_subjects'])}。
- 公平支持集：{context_decision['fair_comparison_support']}。
- W1/W2 均满足 `input_end == target_end`、未来样本数为 0，且完整输入历史位于目标所属的同一冻结区间。
- 合格上下文：{', '.join(context_decision['eligible_contexts'])}。
- 冻结最佳上下文：**{context_decision['selected_context']}**（{context_decision['input_samples']/FS:.0f} 秒输入）。
- 全量复测：运行 {gate['run_pass_count']}/{gate['run_total']}，被试 {gate['subject_pass_count']}/7，门控 **{gate['status']}**。
- W1-Center 仅作为非因果上限，未进入选择。
"""
        write_text(root / "reports" / "E4_causal_context_report.md", e4)
    if final_decision is not None:
        summaries = final_decision["candidate_summaries"]
        summary_rows = [
            (
                row["candidate"],
                row["loss"],
                row["context"],
                f"{row['run_pass_count']}/{row['run_total']}",
                f"{row['subject_pass_count']}/7",
                f"{row['median_pearson']:.3f}",
                f"{row['median_lagged_pearson']:.3f}",
                f"{row['median_nrmse']:.3f}",
                f"{row['median_p90']:.3f}",
                row["status"],
            )
            for row in summaries
        ]
        selected = next(row for row in summaries if row["candidate"] == final_decision["selected_candidate"])
        subjects = selected["gate"]["subject_summary"]
        subject_rows = [
            (
                row["subject_id"],
                f"{row['seed_passes']}/3",
                "PASS" if row["subject_pass"] else "FAIL",
                f"{row['median_corr']:.3f}",
                f"{row['median_nrmse']:.3f}",
                f"{row['median_nrmse_p90']:.3f}",
            )
            for row in subjects
        ]
        final = f"""# Daphnet NBM Zero-shot 因果波形/上下文实验最终报告

## 总体结果

{markdown_table(['模型','Loss','Context','通过运行','通过被试','Pearson','Lagged Pearson','NRMSE','P90','状态'], summary_rows)}

## 最终选择

- 候选：**{final_decision['selected_candidate']}**。
- 正式状态：**{final_decision['formal_status']}**；考虑 P95 尾部风险后的报告状态：**{final_decision['reported_status']}**。
- 方法定位：**{final_decision['positioning']}**。
- 可进入最终残差验证：**{final_decision['eligible_for_final_residual_validation']}**。
- 尾部风险被试（被试内种子中位 P95 > 1.50）：{', '.join(final_decision['tail_risk_subjects_nrmse_p95_gt_1p50']) or '无'}。

## 最终被试级结果

{markdown_table(['被试','种子通过','被试状态','Pearson','NRMSE','P90'], subject_rows)}

## 结论边界

该结论仅针对完全不使用测试记录适配数据的 zero-shot 未见 Non-FoG 重构。它不证明 FoG 残差必然更大，也不证明残差一定提高分类；后续仍需按模板开展残差偏置/尺度、FoG–Non-FoG 分离和分类消融。非因果 W1-Center 从未参与最终选择。
"""
        write_text(root / "reports" / "final_report.md", final)


def audit_outputs(root: Path) -> dict[str, Any]:
    prediction_files = list(root.rglob("predictions.npz"))
    checkpoint_files = list(root.rglob("best_model.pt"))
    config_files = list(root.rglob("config.json"))
    context_files = list(root.rglob("context_manifest.json"))
    finite_predictions = 0
    for path in prediction_files:
        with np.load(path, allow_pickle=False) as payload:
            if all(np.isfinite(payload[key]).all() for key in payload.files):
                finite_predictions += 1
    unique_checkpoints: dict[str, Path] = {}
    for path in checkpoint_files:
        unique_checkpoints.setdefault(sha256(path), path)
    loadable_checkpoints = 0
    for path in unique_checkpoints.values():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state" in payload and payload["model_state"]:
            loadable_checkpoints += 1
    zero_shot_configs = 0
    for path in config_files:
        config = read_json(path)
        if (
            not as_bool(config.get("test_nonfog_calibration", False))
            and not as_bool(config.get("test_fog_used_for_selection", False))
        ):
            zero_shot_configs += 1
    causal_manifests = 0
    for path in context_files:
        payload = read_json(path)
        policy = payload.get("policy", payload)
        if int(policy.get("future_samples_after_target", 0)) == 0:
            causal_manifests += 1
    temporary = [path for path in root.rglob("*") if path.is_file() and ".tmp-" in path.name]
    zero_byte = [path for path in root.rglob("*") if path.is_file() and path.stat().st_size == 0]
    final_runs = list((root / "final_combination").rglob("run_metrics.json"))
    audit = {
        "prediction_files": len(prediction_files),
        "finite_prediction_files": finite_predictions,
        "checkpoint_files": len(checkpoint_files),
        "unique_checkpoint_hashes": len(unique_checkpoints),
        "loadable_unique_checkpoints": loadable_checkpoints,
        "config_files": len(config_files),
        "zero_shot_config_files": zero_shot_configs,
        "context_manifest_files": len(context_files),
        "causal_context_manifest_files": causal_manifests,
        "final_candidate_run_metrics": len(final_runs),
        "expected_final_candidate_runs": 84,
        "temporary_files": len(temporary),
        "zero_byte_files": len(zero_byte),
        "all_predictions_finite": finite_predictions == len(prediction_files),
        "all_unique_checkpoints_loadable": loadable_checkpoints == len(unique_checkpoints),
        "all_configs_zero_shot": zero_shot_configs == len(config_files),
        "all_main_context_manifests_causal": causal_manifests == len(context_files),
        "final_run_count_complete": len(final_runs) == 84,
        "status": "PASS",
    }
    if not all(
        audit[key]
        for key in (
            "all_predictions_finite",
            "all_unique_checkpoints_loadable",
            "all_configs_zero_shot",
            "all_main_context_manifests_causal",
            "final_run_count_complete",
        )
    ) or temporary or zero_byte:
        audit["status"] = "FAIL"
    base.write_json(root / "reports" / "artifact_audit.json", audit)
    return audit


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.parent_a1 = args.parent_a1.resolve()
    args.parent_a1b = args.parent_a1b.resolve()
    args.output_dir = args.output_dir.resolve()
    root = args.output_dir / "routeA_zero_shot_causal_repair"
    root.mkdir(parents=True, exist_ok=True)
    for required in (args.data_dir, args.parent_a1, args.parent_a1b, args.template):
        if not required.exists():
            raise FileNotFoundError(required)
    dataset = DaphnetDataset.load(args.data_dir)
    prepared = {subject: a1.prepare_subject(dataset, subject) for subject in SUBJECTS}
    parent_results = a1b.original_results(args.parent_a1)
    if len(parent_results) != 21:
        raise ValueError(f"expected 21 frozen A1 runs, found {len(parent_results)}")
    write_protocol(root, args, prepared)
    device = base.resolve_device(args.device)
    diagnostics, _, _ = run_failure_diagnostics(
        root,
        prepared,
        parent_results,
        args.parent_a1b,
        skip_figures=args.skip_figures,
    )
    write_reports(root, diagnostics, None, None, None)
    if args.stop_after == "diagnostics":
        print("ZERO-SHOT CAUSAL EXPERIMENT stopped after E1 diagnostics", flush=True)
        return
    frequency_trigger = read_json(root / "reports" / "E1_frequency_trigger.json")
    selected_loss, loss_rows, loss_decision = run_loss_ablation(
        root,
        prepared,
        args.parent_a1b,
        frequency_trigger,
        max_epochs=args.max_epochs,
        patience=args.patience,
        workers=args.num_workers,
        device=device,
        overwrite=args.overwrite,
        skip_figures=args.skip_figures,
    )
    write_reports(root, diagnostics, loss_decision, None, None)
    if args.stop_after == "loss":
        print(f"ZERO-SHOT CAUSAL EXPERIMENT stopped after E3 loss; selected={selected_loss}", flush=True)
        return
    selected_context, context_rows, context_decision, _ = run_context_ablation(
        root,
        prepared,
        args.parent_a1b,
        selected_loss,
        loss_rows,
        diagnostics,
        max_epochs=args.max_epochs,
        patience=args.patience,
        workers=args.num_workers,
        device=device,
        overwrite=args.overwrite,
        skip_figures=args.skip_figures,
    )
    write_reports(root, diagnostics, loss_decision, context_decision, None)
    if args.stop_after == "context":
        print(f"ZERO-SHOT CAUSAL EXPERIMENT stopped after E4 context; selected={selected_context}", flush=True)
        return
    _, final_decision = run_final_combination(
        root,
        prepared,
        parent_results,
        selected_loss,
        loss_rows,
        selected_context,
        context_rows,
        max_epochs=args.max_epochs,
        patience=args.patience,
        workers=args.num_workers,
        device=device,
        overwrite=args.overwrite,
        skip_figures=args.skip_figures,
    )
    write_reports(root, diagnostics, loss_decision, context_decision, final_decision)
    audit = audit_outputs(root)
    print(
        f"ZERO-SHOT CAUSAL COMPLETE candidate={final_decision['selected_candidate']} "
        f"status={final_decision['reported_status']} subjects={final_decision['subject_pass_count']}/7 "
        f"audit={audit['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
