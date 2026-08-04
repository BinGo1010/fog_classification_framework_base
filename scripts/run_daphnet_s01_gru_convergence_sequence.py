#!/usr/bin/env python
"""Sequential, leakage-safe S01 GRU convergence experiments.

This runner addresses six diagnostic questions in a fixed order while never
opening the held-out S01 R02 array.  Every arm reuses the exact 978/295
clean-normal train/validation master windows and the train-only RobustScaler
from ``diagnose_daphnet_s01_gru_convergence``.  Short forecast horizons are
prefixes of the same two-second target, so context, prediction origin, guard,
and sample support remain paired across arms.

Stages
------
1. ``overfit``: fixed 32-window mean-only overfit test.
2. ``long_mean``: 2 s mean-only training with step-budgeted RMSE early stop.
3. ``sigma_detach``: joint NLL with/without sigma-to-encoder gradient flow.
4. ``horizon``: 0.25/0.5/1/2 s mean-only forecast ablation.
5. ``decoder``: parameter-matched direct/shared/TCN/GRU decoders.
6. ``modes``: train-only context clustering plus conditioned/MoE predictors.

The script is intentionally separate from the completed earlier diagnostics;
changing those source files would invalidate the source hashes recorded in
their DONE manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import diagnose_daphnet_s01_gru_convergence as diagnostic  # noqa: E402
from cnbr_fog.data import DaphnetDataset, RobustChannelScaler, WindowTable  # noqa: E402
from cnbr_fog.nbm import GRUNBM, gaussian_nll_sigma, parameter_count  # noqa: E402
from cnbr_fog.nbm_representations import calibrate_fixed_sigma  # noqa: E402
from cnbr_fog.gru_convergence_models import (  # noqa: E402
    DECODER_NAMES,
    ClusterConditionedGRUMeanForecaster,
    GRUMeanForecaster,
    JointDirectGRUForecaster,
    MoEGRUMeanForecaster,
)
from cnbr_fog.gru_mode_analysis import (  # noqa: E402
    GRUContextModeAnalyzer,
    summarize_train_validation_clusters,
)
from cnbr_fog.gru_predictor_artifact import ARTIFACT_SCHEMA_VERSION  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_convergence_sequence.v4"
HORIZONS: tuple[tuple[str, int], ...] = (
    ("h025", 16),
    ("h050", 32),
    ("h100", 64),
    ("h200", 128),
)
STAGE_ORDER = (
    "overfit",
    "long_mean",
    "sigma_detach",
    "horizon",
    "decoder",
    "modes",
    "finalize",
)
STAGE_DIRECTORIES = {
    "overfit": "01_overfit",
    "long_mean": "02_long_mean",
    "sigma_detach": "03_sigma_detach",
    "horizon": "04_horizon",
    "decoder": "05_decoder",
    "modes": "06_modes",
    "finalize": "07_finalize",
}
STAGE_PREREQUISITE = {
    "long_mean": "overfit",
    "sigma_detach": "long_mean",
    "horizon": "sigma_detach",
    "decoder": "horizon",
    "modes": "decoder",
    "finalize": "modes",
}
MIN_DELTA_RMSE = 1e-4
MIN_DELTA_NLL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential S01 GRU convergence experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "daphnet_s01_gru_convergence_sequence_v4"
        ),
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER),
        help="Comma-separated ordered stage subset",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-steps", type=int, default=32)
    parser.add_argument("--overfit-max-steps", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def parse_stages(specification: str) -> tuple[str, ...]:
    requested: list[str] = []
    for token in specification.split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in STAGE_ORDER:
            raise ValueError(f"Unknown stage {name!r}; use {STAGE_ORDER}")
        if name not in requested:
            requested.append(name)
    if not requested:
        raise ValueError("At least one stage is required")
    positions = [STAGE_ORDER.index(name) for name in requested]
    if positions != sorted(positions):
        raise ValueError("Stages must be supplied in scientific execution order")
    return tuple(requested)


def validate_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[int, ...]]:
    stages = parse_stages(args.stages)
    seeds = diagnostic.parse_int_list(args.seeds)
    if min(
        args.hidden_channels,
        args.batch_size,
        args.max_steps,
        args.patience,
        args.min_steps,
        args.overfit_max_steps,
    ) <= 0:
        raise ValueError("All size/step/patience values must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if args.min_steps > args.max_steps:
        raise ValueError("min_steps cannot exceed max_steps")
    return stages, seeds


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def numeric_stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def horizon_seconds(samples: int) -> float:
    return int(samples) / float(diagnostic.base.SAMPLING_RATE_HZ)


def split_sequence(
    sequence: torch.Tensor,
    horizon_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(horizon_samples)
    if not 1 <= horizon <= diagnostic.base.TARGET_SAMPLES:
        raise ValueError("horizon_samples must be in [1,128]")
    expected = diagnostic.base.CONTEXT_SAMPLES + diagnostic.base.TARGET_SAMPLES
    if sequence.ndim != 3 or int(sequence.shape[-1]) != expected:
        raise ValueError(f"Expected [batch,channel,{expected}], got {sequence.shape}")
    context = sequence[:, :, : diagnostic.base.CONTEXT_SAMPLES]
    target = sequence[
        :,
        :,
        diagnostic.base.CONTEXT_SAMPLES : diagnostic.base.CONTEXT_SAMPLES + horizon,
    ]
    return context, target


def _mean_output(
    model: nn.Module,
    context: torch.Tensor,
    mode: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(model, "forward_mean"):
        function = getattr(model, "forward_mean")
        # The current conditioned and MoE models infer their routing strictly
        # from context.  Frozen KMeans modes are diagnostic/evaluation strata,
        # not privileged inputs.  A future hard-conditioned model must opt in
        # explicitly before labels are passed into its forward path.
        if mode is not None and bool(getattr(model, "uses_external_mode", False)):
            return function(context, mode)
        return function(context)
    output = model(context)
    if isinstance(output, tuple):
        return output[0]
    return output


def _joint_output(
    model: nn.Module,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(context)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("Joint model must return (mean,sigma)")
    return output


def mode_for_indices(
    window_indices: torch.Tensor,
    mode_by_window: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor | None:
    if mode_by_window is None:
        return None
    values = mode_by_window[window_indices.numpy()]
    if np.any(values < 0):
        raise AssertionError("Missing mode assignment for an evaluated window")
    return torch.as_tensor(values, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_mean(
    model: nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    horizon_samples: int,
    batch_size: int,
    device: torch.device,
    mode_by_window: np.ndarray | None = None,
) -> dict[str, Any]:
    loader = diagnostic.base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        0,
        device.type == "cuda",
    )
    model.eval()
    horizon = int(horizon_samples)
    squared_sum = 0.0
    absolute_sum = 0.0
    values = 0
    windows_seen = 0
    horizon_squared = np.zeros(horizon, dtype=np.float64)
    channel_squared = np.zeros(dataset.n_channels, dtype=np.float64)
    per_mode: dict[int, dict[str, float]] = {}
    for sequence, _, window_index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context, target = split_sequence(sequence, horizon)
        mode = mode_for_indices(window_index, mode_by_window, device)
        mean = _mean_output(model, context.float(), mode).float()
        if mean.shape != target.shape:
            raise RuntimeError(f"mean shape {mean.shape} != target {target.shape}")
        error = target.float() - mean
        squared = error.square().double()
        squared_sum += float(squared.sum().cpu())
        absolute_sum += float(error.abs().double().sum().cpu())
        values += int(error.numel())
        windows_seen += int(error.shape[0])
        array = squared.cpu().numpy()
        horizon_squared += array.sum(axis=(0, 1))
        channel_squared += array.sum(axis=(0, 2))
        if mode is not None:
            mode_array = mode.cpu().numpy()
            error_array = error.double().cpu().numpy()
            for label in np.unique(mode_array):
                selected = error_array[mode_array == label]
                item = per_mode.setdefault(
                    int(label), {"squared_sum": 0.0, "absolute_sum": 0.0, "values": 0}
                )
                item["squared_sum"] += float(np.square(selected).sum())
                item["absolute_sum"] += float(np.abs(selected).sum())
                item["values"] += int(selected.size)
    if windows_seen != len(indices) or values <= 0:
        raise AssertionError("Mean evaluation support changed")
    result: dict[str, Any] = {
        "windows": windows_seen,
        "horizon_samples": horizon,
        "horizon_seconds": horizon_seconds(horizon),
        "mse_scaled": squared_sum / values,
        "rmse_scaled": math.sqrt(squared_sum / values),
        "mae_scaled": absolute_sum / values,
        "per_channel_rmse_scaled": np.sqrt(
            channel_squared / (windows_seen * horizon)
        ).tolist(),
        "per_horizon_rmse_scaled": np.sqrt(
            horizon_squared / (windows_seen * dataset.n_channels)
        ).tolist(),
    }
    if per_mode:
        result["per_mode"] = {
            str(label): {
                "rmse_scaled": math.sqrt(item["squared_sum"] / item["values"]),
                "mae_scaled": item["absolute_sum"] / item["values"],
                "values": int(item["values"]),
                "windows": int(item["values"] // (dataset.n_channels * horizon)),
            }
            for label, item in sorted(per_mode.items())
        }
    return result


@torch.no_grad()
def evaluate_joint(
    model: nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    horizon_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = diagnostic.base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        0,
        device.type == "cuda",
    )
    model.eval()
    sums = {
        "nll": 0.0,
        "log_sigma": 0.0,
        "half_z2": 0.0,
        "squared": 0.0,
        "absolute": 0.0,
        "sigma": 0.0,
        "z2": 0.0,
        "cover1": 0.0,
        "cover196": 0.0,
    }
    values = 0
    windows_seen = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context, target = split_sequence(sequence, horizon_samples)
        mean, sigma = _joint_output(model, context.float())
        mean = mean.float()
        sigma = sigma.float()
        error = target.float() - mean
        z = error / sigma
        nll = torch.log(sigma) + 0.5 * z.square()
        count = int(error.numel())
        values += count
        windows_seen += int(error.shape[0])
        sums["nll"] += float(nll.double().sum().cpu())
        sums["log_sigma"] += float(torch.log(sigma).double().sum().cpu())
        sums["half_z2"] += float((0.5 * z.square()).double().sum().cpu())
        sums["squared"] += float(error.square().double().sum().cpu())
        sums["absolute"] += float(error.abs().double().sum().cpu())
        sums["sigma"] += float(sigma.double().sum().cpu())
        sums["z2"] += float(z.square().double().sum().cpu())
        sums["cover1"] += float((z.abs() <= 1).sum().cpu())
        sums["cover196"] += float((z.abs() <= 1.96).sum().cpu())
    if windows_seen != len(indices) or values <= 0:
        raise AssertionError("Joint evaluation support changed")
    return {
        "windows": windows_seen,
        "gaussian_nll": sums["nll"] / values,
        "mean_log_sigma": sums["log_sigma"] / values,
        "mean_half_standardized_squared_error": sums["half_z2"] / values,
        "forecast_rmse_scaled": math.sqrt(sums["squared"] / values),
        "forecast_mae_scaled": sums["absolute"] / values,
        "mean_sigma_scaled": sums["sigma"] / values,
        "standardized_residual_rms": math.sqrt(sums["z2"] / values),
        "coverage_abs_z_le_1": sums["cover1"] / values,
        "coverage_abs_z_le_1p96": sums["cover196"] / values,
    }


def collect_targets_and_means(
    model: nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    horizon_samples: int,
    batch_size: int,
    device: torch.device,
    mode_by_window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loader = diagnostic.base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        0,
        device.type == "cuda",
    )
    targets: list[np.ndarray] = []
    means: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for sequence, _, window_index in loader:
            sequence = sequence.to(device, non_blocking=True)
            context, target = split_sequence(sequence, horizon_samples)
            mode = mode_for_indices(window_index, mode_by_window, device)
            mean = _mean_output(model, context.float(), mode)
            targets.append(target.float().cpu().numpy())
            means.append(mean.float().cpu().numpy())
    return np.concatenate(targets), np.concatenate(means)


def persistence_baseline(
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    horizon_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    def collect(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loader = diagnostic.base.sequence_loader(
            dataset, windows, indices, scaler, batch_size, False, 0, False
        )
        lasts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for sequence, _, _ in loader:
            context, target = split_sequence(sequence, horizon_samples)
            lasts.append(context[:, :, -1:].numpy())
            targets.append(target.numpy())
        return np.concatenate(lasts), np.concatenate(targets)

    train_last, train_target = collect(train_indices)
    val_last, val_target = collect(validation_indices)
    train_mean = np.repeat(train_last, horizon_samples, axis=2)
    val_mean = np.repeat(val_last, horizon_samples, axis=2)
    sigma = calibrate_fixed_sigma(train_target - train_mean)

    def metrics(target: np.ndarray, mean: np.ndarray) -> dict[str, float]:
        error = target.astype(np.float64) - mean.astype(np.float64)
        z = error / sigma.astype(np.float64)
        return {
            "rmse_scaled": float(np.sqrt(np.mean(error**2))),
            "mae_scaled": float(np.mean(np.abs(error))),
            "gaussian_nll": float(np.mean(np.log(sigma) + 0.5 * z**2)),
            "z_rms": float(np.sqrt(np.mean(z**2))),
            "coverage_95": float(np.mean(np.abs(z) <= 1.96)),
        }

    return {
        "definition": "Repeat final context point; train-only channel-by-horizon sigma",
        "horizon_samples": int(horizon_samples),
        "train": metrics(train_target, train_mean),
        "validation": metrics(val_target, val_mean),
    }


def _last_slope(history: Sequence[Mapping[str, Any]], key: str, count: int = 5) -> float:
    rows = history[-count:]
    if len(rows) < 2:
        return 0.0
    x = np.arange(len(rows), dtype=np.float64)
    y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def aggregate_run_summaries(
    summaries: Sequence[Mapping[str, Any]],
    scalar_keys: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runs": len(summaries),
        "patience_stop_count": sum(
            item["stop_reason"] == "validation_patience" for item in summaries
        ),
        "maximum_step_stop_count": sum(
            item["stop_reason"] == "maximum_steps" for item in summaries
        ),
    }
    for key in scalar_keys:
        result[key] = numeric_stats([float(item[key]) for item in summaries])
    return result


def _root_protocol_fingerprint(root: Path) -> str:
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Root protocol is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = str(config.get("protocol_fingerprint", ""))
    if not fingerprint:
        raise RuntimeError(f"Root protocol has no fingerprint: {config_path}")
    unsigned = dict(config)
    unsigned.pop("protocol_fingerprint", None)
    if canonical_fingerprint(unsigned) != fingerprint:
        raise RuntimeError(f"Root protocol fingerprint mismatch: {config_path}")
    return fingerprint


def _validate_completed_stage(stage_dir: Path) -> dict[str, Any]:
    done_path = stage_dir / "DONE.json"
    if not done_path.exists():
        raise FileNotFoundError(f"Required stage is not complete: {done_path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    expected_stage = stage_dir.name
    if done.get("status") != "complete" or done.get("stage") != expected_stage:
        raise RuntimeError(f"Invalid completion manifest: {done_path}")
    if done.get("experiment_version") != EXPERIMENT_VERSION:
        raise RuntimeError(f"Stage experiment version mismatch: {done_path}")

    root_fingerprint = _root_protocol_fingerprint(stage_dir.parent)
    config_path = stage_dir / "config.json"
    if config_path.exists():
        stage_config = json.loads(config_path.read_text(encoding="utf-8"))
        stage_fingerprint = str(stage_config.get("protocol_fingerprint", ""))
        if stage_config.get("suite_protocol_fingerprint") != root_fingerprint:
            raise RuntimeError(f"Stage is not linked to this suite: {config_path}")
        unsigned = dict(stage_config)
        unsigned.pop("protocol_fingerprint", None)
        if canonical_fingerprint(unsigned) != stage_fingerprint:
            raise RuntimeError(f"Stage config fingerprint mismatch: {config_path}")
        if done.get("protocol_fingerprint") != stage_fingerprint:
            raise RuntimeError(f"DONE/config fingerprint mismatch: {done_path}")
    elif expected_stage == "07_finalize":
        if done.get("protocol_fingerprint") != root_fingerprint:
            raise RuntimeError(f"Finalize/root fingerprint mismatch: {done_path}")
    else:
        raise FileNotFoundError(f"Completed stage config is missing: {config_path}")

    declared = dict(done.get("artifacts", {}))
    actual_paths = {
        str(path.relative_to(stage_dir)).replace("\\", "/"): path
        for path in stage_dir.rglob("*")
        if path.is_file() and path.name != "DONE.json"
    }
    if set(declared) != set(actual_paths):
        missing = sorted(set(declared) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(declared))
        raise RuntimeError(
            f"Stage artifact inventory mismatch: missing={missing}, extra={extra}"
        )
    for relative, expected_hash in declared.items():
        if sha256_file(actual_paths[relative]) != expected_hash:
            raise RuntimeError(
                f"Stage artifact hash mismatch: {actual_paths[relative]}"
            )
    return done


def _stage_ready(stage_dir: Path, resume: bool) -> dict[str, Any] | None:
    done = stage_dir / "DONE.json"
    if done.exists():
        if not resume:
            raise FileExistsError(f"Completed stage exists: {stage_dir}")
        return _validate_completed_stage(stage_dir)
    if stage_dir.exists() and any(stage_dir.iterdir()) and not resume:
        raise FileExistsError(f"Incomplete non-empty stage directory: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    return None


def _finish_stage(
    stage_dir: Path,
    stage: str,
    protocol_fingerprint: str,
    summary_file: str = "aggregate.json",
) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(stage_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file() and path.name != "DONE.json"
    }
    payload = {
        "status": "complete",
        "stage": stage,
        "completed_utc": utc_now(),
        "experiment_version": EXPERIMENT_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "summary_file": summary_file,
        "test_record_evaluated": False,
        "artifacts": artifacts,
    }
    atomic_json_dump(payload, stage_dir / "DONE.json")
    return payload


def _load_stage_summary(root: Path, stage: str) -> dict[str, Any]:
    stage_dir = root / stage
    done = _validate_completed_stage(stage_dir)
    path = stage_dir / str(done.get("summary_file", "aggregate.json"))
    if not path.exists():
        raise FileNotFoundError(f"Required completed stage summary missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config(model: nn.Module) -> dict[str, Any]:
    if hasattr(model, "model_config"):
        return dict(getattr(model, "model_config")())
    return {"class_name": type(model).__name__}


def _fixed_sigma_metrics(
    train_target: np.ndarray,
    train_mean: np.ndarray,
    validation_target: np.ndarray,
    validation_mean: np.ndarray,
) -> dict[str, Any]:
    sigma = calibrate_fixed_sigma(train_target - train_mean)

    def calculate(target: np.ndarray, mean: np.ndarray) -> dict[str, float]:
        error = target.astype(np.float64) - mean.astype(np.float64)
        sigma64 = sigma.astype(np.float64)
        z = error / sigma64
        return {
            "rmse_scaled": float(np.sqrt(np.mean(error**2))),
            "mae_scaled": float(np.mean(np.abs(error))),
            "gaussian_nll": float(np.mean(np.log(sigma64) + 0.5 * z**2)),
            "standardized_residual_rms": float(np.sqrt(np.mean(z**2))),
            "coverage_abs_z_le_1": float(np.mean(np.abs(z) <= 1.0)),
            "coverage_abs_z_le_1p96": float(np.mean(np.abs(z) <= 1.96)),
        }

    return {
        "definition": "Train-only analytic channel-by-horizon RMS sigma",
        "sigma_shape": list(sigma.shape),
        "sigma_sha256": diagnostic.array_sha256(sigma),
        "sigma": sigma.tolist(),
        "train": calculate(train_target, train_mean),
        "validation": calculate(validation_target, validation_mean),
    }


def train_mean_run(
    *,
    run_dir: Path,
    model_factory: Callable[[], nn.Module],
    seed: int,
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    horizon_samples: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    protocol_fingerprint: str,
    device: torch.device,
    amp: bool,
    mode_by_window: np.ndarray | None = None,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    resume_path = run_dir / "resume.pt"
    if summary_path.exists() and best_path.exists():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(f"Completed run protocol mismatch: {run_dir}")
        if completed.get("best_checkpoint_sha256") != sha256_file(best_path):
            raise RuntimeError(f"Completed run checkpoint hash mismatch: {best_path}")
        if not (run_dir / "history.csv").exists():
            diagnostic.write_csv(run_dir / "history.csv", completed["history"])
        return completed
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostic.set_seed(seed, True)
    model = model_factory().to(device)
    # Pair stochastic training noise across architecture arms independently of
    # how many RNG draws each constructor consumed.
    diagnostic.set_seed(seed + 1_000_000, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, patience // 3),
        threshold=MIN_DELTA_RMSE,
        threshold_mode="abs",
        min_lr=1e-5,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(amp and device.type == "cuda")
    )
    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = 0
    best_step = 0
    bad_epochs = 0
    cumulative_steps = 0
    epoch = 0
    gradient_clip_steps = 0
    total_gradient_steps = 0
    elapsed_before = 0.0
    started = time.perf_counter()
    if resume_path.exists():
        resume_state = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume_state.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(f"Resume checkpoint protocol mismatch: {resume_path}")
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        grad_scaler.load_state_dict(resume_state["grad_scaler_state"])
        history = list(resume_state["history"])
        best_rmse = float(resume_state["best_rmse"])
        best_epoch = int(resume_state["best_epoch"])
        best_step = int(resume_state["best_step"])
        bad_epochs = int(resume_state["bad_epochs"])
        cumulative_steps = int(resume_state["cumulative_steps"])
        epoch = int(resume_state["epoch"])
        gradient_clip_steps = int(resume_state["gradient_clip_steps"])
        total_gradient_steps = int(resume_state["total_gradient_steps"])
        elapsed_before = float(resume_state.get("elapsed_seconds", 0.0))
        if not best_path.exists():
            raise FileNotFoundError(f"Resume checkpoint has no best model: {best_path}")
        print(
            f"[mean seed={seed} h={horizon_samples}] resume epoch={epoch} "
            f"steps={cumulative_steps}",
            flush=True,
        )

    while cumulative_steps < max_steps and not (
        cumulative_steps >= min_steps and bad_epochs >= patience
    ):
        epoch += 1
        diagnostic.set_seed(seed + 2_000_000 + epoch, True)
        model.train()
        loader = diagnostic.base.sequence_loader(
            dataset,
            windows,
            train_indices,
            scaler,
            batch_size,
            True,
            0,
            device.type == "cuda",
            seed=seed + epoch,
        )
        squared_sum = 0.0
        values = 0
        gradient_norms: list[float] = []
        for sequence, _, window_index in loader:
            if cumulative_steps >= max_steps:
                break
            sequence = sequence.to(device, non_blocking=True)
            context, target = split_sequence(sequence, horizon_samples)
            mode = mode_for_indices(window_index, mode_by_window, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type, enabled=bool(amp and device.type == "cuda")
            ):
                mean = _mean_output(model, context, mode)
                error = target - mean
                loss = 0.5 * error.square().mean()
                if hasattr(model, "regularization_loss"):
                    loss = loss + getattr(model, "regularization_loss")()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite mean-only loss")
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite mean-only gradient")
            gradient_norms.append(norm)
            gradient_clip_steps += int(norm > 5.0)
            total_gradient_steps += 1
            grad_scaler.step(optimizer)
            grad_scaler.update()
            cumulative_steps += 1
            squared_sum += float(error.detach().square().double().sum().cpu())
            values += int(error.numel())

        train = evaluate_mean(
            model,
            dataset,
            windows,
            train_indices,
            scaler,
            horizon_samples,
            batch_size,
            device,
            mode_by_window,
        )
        validation = evaluate_mean(
            model,
            dataset,
            windows,
            validation_indices,
            scaler,
            horizon_samples,
            batch_size,
            device,
            mode_by_window,
        )
        scheduler.step(validation["rmse_scaled"])
        improved = validation["rmse_scaled"] < best_rmse - MIN_DELTA_RMSE
        row = {
            "epoch": epoch,
            "cumulative_optimizer_steps": cumulative_steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_mse_scaled": squared_sum / max(values, 1),
            "train_rmse_scaled": train["rmse_scaled"],
            "train_mae_scaled": train["mae_scaled"],
            "validation_rmse_scaled": validation["rmse_scaled"],
            "validation_mae_scaled": validation["mae_scaled"],
            "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
            "max_preclip_gradient_norm": float(np.max(gradient_norms)),
            "improved": improved,
        }
        history.append(row)
        if improved:
            best_rmse = float(validation["rmse_scaled"])
            best_epoch = epoch
            best_step = cumulative_steps
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "seed": seed,
                    "epoch": epoch,
                    "optimizer_steps": cumulative_steps,
                    "horizon_samples": int(horizon_samples),
                    "validation_clean_normal_rmse": best_rmse,
                    "model_config": _model_config(model),
                    "model_state": model.state_dict(),
                },
                best_path,
            )
        else:
            bad_epochs += 1
        print(
            f"[mean seed={seed} h={horizon_samples}] epoch={epoch:03d} "
            f"steps={cumulative_steps:04d}/{max_steps} "
            f"train={train['rmse_scaled']:.6f} "
            f"val={validation['rmse_scaled']:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        atomic_torch_save(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "seed": seed,
                "epoch": epoch,
                "cumulative_steps": cumulative_steps,
                "best_rmse": best_rmse,
                "best_epoch": best_epoch,
                "best_step": best_step,
                "bad_epochs": bad_epochs,
                "gradient_clip_steps": gradient_clip_steps,
                "total_gradient_steps": total_gradient_steps,
                "elapsed_seconds": (
                    elapsed_before + time.perf_counter() - started
                ),
                "history": history,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
            },
            resume_path,
        )
        if cumulative_steps >= min_steps and bad_epochs >= patience:
            break

    stop_reason = (
        "validation_patience"
        if cumulative_steps >= min_steps and bad_epochs >= patience
        else "maximum_steps"
    )
    atomic_torch_save(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "seed": seed,
            "epoch": epoch,
            "optimizer_steps": cumulative_steps,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_validation_rmse": best_rmse,
            "stop_reason": stop_reason,
            "model_config": _model_config(model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        run_dir / "last.pt",
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_train = evaluate_mean(
        model,
        dataset,
        windows,
        train_indices,
        scaler,
        horizon_samples,
        batch_size,
        device,
        mode_by_window,
    )
    best_validation = evaluate_mean(
        model,
        dataset,
        windows,
        validation_indices,
        scaler,
        horizon_samples,
        batch_size,
        device,
        mode_by_window,
    )
    persistence = persistence_baseline(
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        horizon_samples,
        batch_size,
    )
    train_target, train_mean = collect_targets_and_means(
        model,
        dataset,
        windows,
        train_indices,
        scaler,
        horizon_samples,
        batch_size,
        device,
        mode_by_window,
    )
    val_target, val_mean = collect_targets_and_means(
        model,
        dataset,
        windows,
        validation_indices,
        scaler,
        horizon_samples,
        batch_size,
        device,
        mode_by_window,
    )
    fixed_sigma = _fixed_sigma_metrics(
        train_target, train_mean, val_target, val_mean
    )
    summary = {
        "protocol_fingerprint": protocol_fingerprint,
        "seed": seed,
        "model_config": _model_config(model),
        "parameter_count": int(parameter_count(model)),
        "horizon_samples": int(horizon_samples),
        "horizon_seconds": horizon_seconds(horizon_samples),
        "epochs_completed": len(history),
        "cumulative_optimizer_steps": cumulative_steps,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_validation_rmse": best_validation["rmse_scaled"],
        "best_validation_mae": best_validation["mae_scaled"],
        "persistence_validation_rmse": persistence["validation"]["rmse_scaled"],
        "rmse_skill_vs_persistence": (
            1.0
            - best_validation["rmse_scaled"]
            / persistence["validation"]["rmse_scaled"]
        ),
        "last_five_validation_rmse_slope_per_epoch": _last_slope(
            history, "validation_rmse_scaled"
        ),
        "gradient_clip_step_fraction": gradient_clip_steps
        / max(total_gradient_steps, 1),
        "best": {"train": best_train, "validation": best_validation},
        "persistence": persistence,
        "fixed_sigma_calibration": fixed_sigma,
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(best_path),
        "history": history,
    }
    atomic_json_dump(summary, summary_path)
    diagnostic.write_csv(run_dir / "history.csv", history)
    return summary


def train_joint_run(
    *,
    run_dir: Path,
    model_factory: Callable[[], nn.Module],
    seed: int,
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    horizon_samples: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    patience: int,
    protocol_fingerprint: str,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_nll.pt"
    resume_path = run_dir / "resume.pt"
    if summary_path.exists() and best_path.exists():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(f"Completed run protocol mismatch: {run_dir}")
        if completed.get("best_checkpoint_sha256") != sha256_file(best_path):
            raise RuntimeError(f"Completed run checkpoint hash mismatch: {best_path}")
        if not (run_dir / "history.csv").exists():
            diagnostic.write_csv(run_dir / "history.csv", completed["history"])
        return completed
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostic.set_seed(seed, True)
    model = model_factory().to(device)
    diagnostic.set_seed(seed + 1_000_000, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, patience // 3),
        threshold=MIN_DELTA_NLL,
        threshold_mode="abs",
        min_lr=1e-5,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(amp and device.type == "cuda")
    )
    best_nll = float("inf")
    best_epoch = 0
    best_step = 0
    best_rmse_seen = float("inf")
    best_rmse_step = 0
    bad_epochs = 0
    cumulative_steps = 0
    epoch = 0
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    started = time.perf_counter()
    if resume_path.exists():
        resume_state = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume_state.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(f"Resume checkpoint protocol mismatch: {resume_path}")
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        grad_scaler.load_state_dict(resume_state["grad_scaler_state"])
        history = list(resume_state["history"])
        best_nll = float(resume_state["best_nll"])
        best_epoch = int(resume_state["best_epoch"])
        best_step = int(resume_state["best_step"])
        best_rmse_seen = float(resume_state["best_rmse_seen"])
        best_rmse_step = int(resume_state["best_rmse_step"])
        bad_epochs = int(resume_state["bad_epochs"])
        cumulative_steps = int(resume_state["cumulative_steps"])
        epoch = int(resume_state["epoch"])
        elapsed_before = float(resume_state.get("elapsed_seconds", 0.0))
        if not best_path.exists():
            raise FileNotFoundError(f"Resume checkpoint has no best model: {best_path}")
        print(
            f"[joint seed={seed} h={horizon_samples}] resume epoch={epoch} "
            f"steps={cumulative_steps}",
            flush=True,
        )

    while cumulative_steps < max_steps and not (
        cumulative_steps >= min_steps and bad_epochs >= patience
    ):
        epoch += 1
        diagnostic.set_seed(seed + 2_000_000 + epoch, True)
        model.train()
        loader = diagnostic.base.sequence_loader(
            dataset,
            windows,
            train_indices,
            scaler,
            batch_size,
            True,
            0,
            device.type == "cuda",
            seed=seed + epoch,
        )
        losses: list[float] = []
        for sequence, _, _ in loader:
            if cumulative_steps >= max_steps:
                break
            sequence = sequence.to(device, non_blocking=True)
            context, target = split_sequence(sequence, horizon_samples)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type, enabled=bool(amp and device.type == "cuda")
            ):
                mean, sigma = _joint_output(model, context)
                loss = gaussian_nll_sigma(target, mean, sigma)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite joint loss")
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite joint gradient")
            grad_scaler.step(optimizer)
            grad_scaler.update()
            cumulative_steps += 1
            losses.append(float(loss.detach()))

        train = evaluate_joint(
            model,
            dataset,
            windows,
            train_indices,
            scaler,
            horizon_samples,
            batch_size,
            device,
        )
        validation = evaluate_joint(
            model,
            dataset,
            windows,
            validation_indices,
            scaler,
            horizon_samples,
            batch_size,
            device,
        )
        scheduler.step(validation["gaussian_nll"])
        improved = validation["gaussian_nll"] < best_nll - MIN_DELTA_NLL
        if validation["forecast_rmse_scaled"] < best_rmse_seen:
            best_rmse_seen = float(validation["forecast_rmse_scaled"])
            best_rmse_step = cumulative_steps
        row = {
            "epoch": epoch,
            "cumulative_optimizer_steps": cumulative_steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_nll": float(np.mean(losses)),
            "train_nll": train["gaussian_nll"],
            "train_rmse": train["forecast_rmse_scaled"],
            "validation_nll": validation["gaussian_nll"],
            "validation_rmse": validation["forecast_rmse_scaled"],
            "validation_mean_log_sigma": validation["mean_log_sigma"],
            "validation_half_z2": validation[
                "mean_half_standardized_squared_error"
            ],
            "improved": improved,
        }
        history.append(row)
        if improved:
            best_nll = float(validation["gaussian_nll"])
            best_epoch = epoch
            best_step = cumulative_steps
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "seed": seed,
                    "epoch": epoch,
                    "optimizer_steps": cumulative_steps,
                    "validation_nll": best_nll,
                    "model_config": _model_config(model),
                    "model_state": model.state_dict(),
                },
                best_path,
            )
        else:
            bad_epochs += 1
        print(
            f"[joint seed={seed} h={horizon_samples}] epoch={epoch:03d} "
            f"steps={cumulative_steps:04d}/{max_steps} "
            f"nll={validation['gaussian_nll']:.6f} "
            f"rmse={validation['forecast_rmse_scaled']:.6f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        atomic_torch_save(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "seed": seed,
                "epoch": epoch,
                "cumulative_steps": cumulative_steps,
                "best_nll": best_nll,
                "best_epoch": best_epoch,
                "best_step": best_step,
                "best_rmse_seen": best_rmse_seen,
                "best_rmse_step": best_rmse_step,
                "bad_epochs": bad_epochs,
                "elapsed_seconds": (
                    elapsed_before + time.perf_counter() - started
                ),
                "history": history,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
            },
            resume_path,
        )
        if cumulative_steps >= min_steps and bad_epochs >= patience:
            break

    stop_reason = (
        "validation_patience"
        if cumulative_steps >= min_steps and bad_epochs >= patience
        else "maximum_steps"
    )
    atomic_torch_save(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "seed": seed,
            "epoch": epoch,
            "optimizer_steps": cumulative_steps,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_validation_nll": best_nll,
            "stop_reason": stop_reason,
            "model_config": _model_config(model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        run_dir / "last.pt",
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_train = evaluate_joint(
        model,
        dataset,
        windows,
        train_indices,
        scaler,
        horizon_samples,
        batch_size,
        device,
    )
    best_validation = evaluate_joint(
        model,
        dataset,
        windows,
        validation_indices,
        scaler,
        horizon_samples,
        batch_size,
        device,
    )
    summary = {
        "protocol_fingerprint": protocol_fingerprint,
        "seed": seed,
        "model_config": _model_config(model),
        "parameter_count": int(parameter_count(model)),
        "horizon_samples": int(horizon_samples),
        "epochs_completed": len(history),
        "cumulative_optimizer_steps": cumulative_steps,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_validation_nll": best_validation["gaussian_nll"],
        "nll_selected_validation_rmse": best_validation["forecast_rmse_scaled"],
        "minimum_validation_rmse_seen": best_rmse_seen,
        "minimum_validation_rmse_step": best_rmse_step,
        "best": {"train": best_train, "validation": best_validation},
        "last_five_validation_nll_slope_per_epoch": _last_slope(
            history, "validation_nll"
        ),
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(best_path),
        "history": history,
    }
    atomic_json_dump(summary, summary_path)
    diagnostic.write_csv(run_dir / "history.csv", history)
    return summary


def run_overfit_stage(
    args: argparse.Namespace,
    root: Path,
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    scaler: RobustChannelScaler,
    device: torch.device,
) -> dict[str, Any]:
    stage = "01_overfit"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)
    order = diagnostic.temporal_spread_order(len(train_indices))
    selected = train_indices[np.asarray(order[:32], dtype=np.int64)]
    if len(np.unique(selected)) != 32:
        raise AssertionError("Overfit support must contain 32 unique windows")
    atomic_npz_save(stage_dir / "support.npz", window_indices=selected)

    loader = diagnostic.base.sequence_loader(
        dataset, windows, selected, scaler, 32, False, 0, False
    )
    sequence, _, observed_indices = next(iter(loader))
    if not np.array_equal(observed_indices.numpy(), selected):
        raise AssertionError("Overfit loader changed fixed window order")
    alignment_rows: list[dict[str, Any]] = []
    for row, window_index in enumerate(selected.tolist()):
        rec_index = int(windows.record_index[window_index])
        record = dataset.records[rec_index]
        start = int(windows.start[window_index])
        target_start = int(windows.target_start[window_index])
        target_end = int(windows.target_end[window_index])
        if target_start != start + diagnostic.base.CONTEXT_SAMPLES:
            raise AssertionError("Context/target coordinate alignment failed")
        if target_end != target_start + diagnostic.base.TARGET_SAMPLES:
            raise AssertionError("Target length alignment failed")
        expected_last = scaler.transform(
            record.x[target_start - 1 : target_start]
        )[0]
        expected_first = scaler.transform(record.x[target_start : target_start + 1])[0]
        np.testing.assert_allclose(
            sequence[row, :, diagnostic.base.CONTEXT_SAMPLES - 1].numpy(),
            expected_last,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sequence[row, :, diagnostic.base.CONTEXT_SAMPLES].numpy(),
            expected_first,
            rtol=0,
            atol=1e-6,
        )
        alignment_rows.append(
            {
                "window_index": window_index,
                "record_id": record.record_id,
                "context_start": start,
                "context_end_exclusive": target_start,
                "target_start": target_start,
                "target_end_exclusive": target_end,
            }
        )
    diagnostic.write_csv(stage_dir / "support.csv", alignment_rows)

    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "purpose": "Fixed 32-window mean-only implementation overfit test",
        "seed": 42,
        "window_count": 32,
        "window_index_sha256": diagnostic.array_sha256(selected),
        "dropout": 0.0,
        "weight_decay": 0.0,
        "amp": False,
        "learning_rate": args.learning_rate,
        "maximum_optimizer_steps": args.overfit_max_steps,
        "success_rule": "final RMSE <= 10% of initial RMSE",
        "target_alignment_checked_against_raw_record": True,
        "test_record_evaluated": False,
    }
    config["suite_protocol_fingerprint"] = _root_protocol_fingerprint(root)
    fingerprint = canonical_fingerprint(config)
    config["protocol_fingerprint"] = fingerprint
    atomic_json_dump(config, stage_dir / "config.json")

    diagnostic.set_seed(42, True)
    model = GRUNBM(
        in_channels=dataset.n_channels,
        horizon=diagnostic.base.TARGET_SAMPLES,
        hidden_channels=args.hidden_channels,
        num_layers=1,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    sequence = sequence.to(device)
    context, target = split_sequence(sequence, diagnostic.base.TARGET_SAMPLES)
    model.eval()
    with torch.no_grad():
        initial_mean, _ = model(context.float())
        initial_rmse = float(torch.sqrt((target - initial_mean).square().mean()))
    threshold = initial_rmse * 0.10
    history: list[dict[str, Any]] = [
        {
            "optimizer_step": 0,
            "train_rmse_scaled": initial_rmse,
            "preclip_gradient_norm": None,
        }
    ]
    final_rmse = initial_rmse
    started = time.perf_counter()
    model.train()
    for step in range(1, args.overfit_max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        mean, _ = model(context)
        error = target - mean
        loss = 0.5 * error.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite overfit loss")
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite overfit gradient")
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.overfit_max_steps:
            model.eval()
            with torch.no_grad():
                check_mean, _ = model(context.float())
                final_rmse = float(
                    torch.sqrt((target.float() - check_mean).square().mean())
                )
            model.train()
            history.append(
                {
                    "optimizer_step": step,
                    "train_rmse_scaled": final_rmse,
                    "preclip_gradient_norm": norm,
                }
            )
            print(
                f"[overfit] step={step:04d} rmse={final_rmse:.8f} "
                f"target={threshold:.8f}",
                flush=True,
            )
            if final_rmse <= threshold:
                break
    success = final_rmse <= threshold
    atomic_torch_save(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": fingerprint,
            "model_config": model.model_config(),
            "model_state": model.state_dict(),
            "fixed_window_indices": selected,
            "initial_rmse": initial_rmse,
            "final_rmse": final_rmse,
            "success": success,
        },
        stage_dir / "overfit_model.pt",
    )
    aggregate = {
        "success": success,
        "initial_rmse_scaled": initial_rmse,
        "final_rmse_scaled": final_rmse,
        "relative_rmse_reduction": 1.0 - final_rmse / initial_rmse,
        "success_threshold_rmse": threshold,
        "optimizer_steps": int(history[-1]["optimizer_step"]),
        "target_alignment_verified": True,
        "reshape_output_shape_verified": list(check_mean.shape) == [32, 9, 128],
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    diagnostic.write_csv(stage_dir / "history.csv", history)
    _finish_stage(stage_dir, stage, fingerprint)
    return aggregate


def _mean_stage_config(
    args: argparse.Namespace,
    stage: str,
    horizon_samples: int,
    seeds: Sequence[int],
    support: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "model_name": model_name,
        "objective": "0.5 * mean squared future error",
        "horizon_samples": int(horizon_samples),
        "horizon_seconds": horizon_seconds(horizon_samples),
        "master_support_horizon_samples": 128,
        "short_target_definition": "prefix of the same master H200 target",
        "support": support,
        "seeds": list(seeds),
        "training_rng_protocol": (
            "model initialized with seed; stochastic training reset each epoch "
            "to seed + 2000000 + epoch"
        ),
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "maximum_optimizer_steps": args.max_steps,
        "minimum_optimizer_steps": args.min_steps,
        "patience_evaluations": args.patience,
        "early_stop_metric": "validation clean-normal RMSE",
        "min_delta_rmse": MIN_DELTA_RMSE,
        "scheduler": "ReduceLROnPlateau factor=0.5",
        "test_record_evaluated": False,
    }


def run_long_mean_stage(
    args: argparse.Namespace,
    root: Path,
    seeds: Sequence[int],
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    support: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    stage = "02_long_mean"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)
    config = _mean_stage_config(
        args,
        stage,
        diagnostic.base.TARGET_SAMPLES,
        seeds,
        support,
        "current_grunbm_direct_mean_path",
    )
    config["suite_protocol_fingerprint"] = _root_protocol_fingerprint(root)
    fingerprint = canonical_fingerprint(config)
    config["protocol_fingerprint"] = fingerprint
    atomic_json_dump(config, stage_dir / "config.json")
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        summaries.append(
            train_mean_run(
                run_dir=stage_dir / "runs" / f"seed_{seed}",
                model_factory=lambda: GRUNBM(
                    in_channels=dataset.n_channels,
                    horizon=diagnostic.base.TARGET_SAMPLES,
                    hidden_channels=args.hidden_channels,
                    num_layers=1,
                    dropout=args.dropout,
                ),
                seed=seed,
                dataset=dataset,
                windows=windows,
                train_indices=train_indices,
                validation_indices=validation_indices,
                scaler=scaler,
                horizon_samples=diagnostic.base.TARGET_SAMPLES,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                max_steps=args.max_steps,
                min_steps=args.min_steps,
                patience=args.patience,
                protocol_fingerprint=fingerprint,
                device=device,
                amp=args.amp,
            )
        )
    keys = (
        "epochs_completed",
        "cumulative_optimizer_steps",
        "best_epoch",
        "best_step",
        "best_validation_rmse",
        "best_validation_mae",
        "persistence_validation_rmse",
        "rmse_skill_vs_persistence",
        "last_five_validation_rmse_slope_per_epoch",
        "gradient_clip_step_fraction",
        "elapsed_seconds",
    )
    aggregate = aggregate_run_summaries(summaries, keys)
    aggregate["mean_predictor_converged_count"] = sum(
        item["stop_reason"] == "validation_patience"
        and abs(item["last_five_validation_rmse_slope_per_epoch"]) < 5e-4
        for item in summaries
    )
    aggregate["convergence_rule"] = (
        "validation patience and abs(last-five RMSE slope)<5e-4"
    )
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    diagnostic.write_csv(
        stage_dir / "run_table.csv",
        [
            {
                key: item[key]
                for key in ("seed", "stop_reason", *keys)
            }
            for item in summaries
        ],
    )
    _finish_stage(stage_dir, stage, fingerprint)
    return aggregate


def run_horizon_stage(
    args: argparse.Namespace,
    root: Path,
    seeds: Sequence[int],
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    support: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    stage = "04_horizon"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "purpose": "Paired direct-GRU future-horizon mean convergence ablation",
        "horizons": [
            {"id": name, "samples": samples, "seconds": horizon_seconds(samples)}
            for name, samples in HORIZONS
        ],
        "same_master_window_indices": True,
        "same_context_and_prediction_origin": True,
        "target_definition": "first H samples of the same H200 target",
        "support": support,
        "seeds": list(seeds),
        "training_rng_protocol": (
            "model initialized with seed; stochastic training reset each epoch "
            "to seed + 2000000 + epoch"
        ),
        "training": _mean_stage_config(
            args, stage, 128, seeds, support, "pure_mean_common_gru_direct_decoder"
        ),
        "selection_rule": (
            f"longest horizon with >={math.ceil(0.8 * len(seeds))}/{len(seeds)} "
            "patience+flat-slope stops and mean persistence skill>=5%; fallback "
            "to greatest mean skill"
        ),
        "test_record_evaluated": False,
    }
    config["suite_protocol_fingerprint"] = _root_protocol_fingerprint(root)
    fingerprint = canonical_fingerprint(config)
    config["protocol_fingerprint"] = fingerprint
    atomic_json_dump(config, stage_dir / "config.json")
    arms: dict[str, Any] = {}
    for horizon_id, horizon in HORIZONS:
        summaries: list[dict[str, Any]] = []
        for seed in seeds:
            summaries.append(
                train_mean_run(
                    run_dir=(
                        stage_dir / "arms" / horizon_id / "runs" / f"seed_{seed}"
                    ),
                    model_factory=lambda h=horizon: GRUMeanForecaster(
                        in_channels=dataset.n_channels,
                        horizon=h,
                        hidden_channels=args.hidden_channels,
                        num_layers=1,
                        dropout=args.dropout,
                        decoder="direct",
                    ),
                    seed=seed,
                    dataset=dataset,
                    windows=windows,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    scaler=scaler,
                    horizon_samples=horizon,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    max_steps=args.max_steps,
                    min_steps=args.min_steps,
                    patience=args.patience,
                    protocol_fingerprint=fingerprint,
                    device=device,
                    amp=args.amp,
                )
            )
        keys = (
            "best_validation_rmse",
            "best_validation_mae",
            "rmse_skill_vs_persistence",
            "best_step",
            "cumulative_optimizer_steps",
            "last_five_validation_rmse_slope_per_epoch",
            "elapsed_seconds",
        )
        arm = aggregate_run_summaries(summaries, keys)
        arm.update(
            {
                "horizon_samples": horizon,
                "horizon_seconds": horizon_seconds(horizon),
                "parameter_count": summaries[0]["parameter_count"],
                "converged_count": sum(
                    item["stop_reason"] == "validation_patience"
                    and abs(item["last_five_validation_rmse_slope_per_epoch"]) < 5e-4
                    for item in summaries
                ),
            }
        )
        arms[horizon_id] = arm
        atomic_json_dump(arm, stage_dir / "arms" / horizon_id / "aggregate.json")
        diagnostic.write_csv(
            stage_dir / "arms" / horizon_id / "run_table.csv",
            [
                {
                    "seed": item["seed"],
                    "stop_reason": item["stop_reason"],
                    **{key: item[key] for key in keys},
                }
                for item in summaries
            ],
        )

    eligible = [
        (name, samples)
        for name, samples in HORIZONS
        if arms[name]["converged_count"] >= math.ceil(0.8 * len(seeds))
        and arms[name]["rmse_skill_vs_persistence"]["mean"] >= 0.05
    ]
    if eligible:
        selected_id, selected_samples = max(eligible, key=lambda item: item[1])
        selection_reason = "longest_converged_horizon_with_at_least_5pct_skill"
    else:
        selected_id, selected_samples = max(
            HORIZONS,
            key=lambda item: arms[item[0]]["rmse_skill_vs_persistence"]["mean"],
        )
        selection_reason = "fallback_greatest_mean_persistence_skill"
    aggregate = {
        "arms": arms,
        "selected_horizon_id": selected_id,
        "selected_horizon_samples": selected_samples,
        "selected_horizon_seconds": horizon_seconds(selected_samples),
        "selection_reason": selection_reason,
    }
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    diagnostic.write_csv(
        stage_dir / "horizon_table.csv",
        [
            {
                "horizon_id": name,
                "horizon_samples": samples,
                "horizon_seconds": horizon_seconds(samples),
                "converged_count": arms[name]["converged_count"],
                "patience_stop_count": arms[name]["patience_stop_count"],
                "best_validation_rmse_mean": arms[name]["best_validation_rmse"]["mean"],
                "best_validation_rmse_std": arms[name]["best_validation_rmse"]["std"],
                "persistence_skill_mean": arms[name]["rmse_skill_vs_persistence"]["mean"],
                "best_step_mean": arms[name]["best_step"]["mean"],
            }
            for name, samples in HORIZONS
        ],
    )
    _finish_stage(stage_dir, stage, fingerprint)
    return aggregate


def _paired_comparison(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    key: str,
    *,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Return seed-paired candidate-vs-reference diagnostics."""

    reference_by_seed = {int(item["seed"]): float(item[key]) for item in reference}
    candidate_by_seed = {int(item["seed"]): float(item[key]) for item in candidate}
    shared = sorted(set(reference_by_seed) & set(candidate_by_seed))
    if not shared:
        raise ValueError("Paired comparison has no shared seeds")
    signed_delta = [candidate_by_seed[s] - reference_by_seed[s] for s in shared]
    relative_gain = [
        (
            (reference_by_seed[s] - candidate_by_seed[s])
            if lower_is_better
            else (candidate_by_seed[s] - reference_by_seed[s])
        )
        / max(abs(reference_by_seed[s]), 1e-12)
        for s in shared
    ]
    return {
        "seeds": shared,
        "candidate_minus_reference": signed_delta,
        "candidate_minus_reference_stats": numeric_stats(signed_delta),
        "relative_gain": relative_gain,
        "relative_gain_stats": numeric_stats(relative_gain),
        "candidate_win_count": sum(value > 0.0 for value in relative_gain),
    }


def _encoder_sha256(model: nn.Module) -> str:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise TypeError(f"{type(model).__name__} has no exposed encoder")
    return state_sha256(encoder.state_dict())


def run_sigma_detach_stage(
    args: argparse.Namespace,
    root: Path,
    seeds: Sequence[int],
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    support: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    stage = "03_sigma_detach"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)
    arms_spec = (("joint", False), ("sigma_state_detach", True))
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "purpose": "Isolate sigma-head gradient competition at the shared encoder",
        "arms": [
            {"name": name, "sigma_state_detach": detach}
            for name, detach in arms_spec
        ],
        "shared_mean_and_sigma_architecture": True,
        "identical_initial_state_within_seed": True,
        "objective": "joint Gaussian NLL",
        "selection_metric": "validation clean-normal Gaussian NLL",
        "horizon_samples": diagnostic.base.TARGET_SAMPLES,
        "horizon_seconds": 2.0,
        "support": support,
        "seeds": list(seeds),
        "training_rng_protocol": (
            "model initialized with seed; stochastic training reset each epoch "
            "to seed + 2000000 + epoch"
        ),
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "maximum_optimizer_steps": args.max_steps,
        "minimum_optimizer_steps": args.min_steps,
        "patience_evaluations": args.patience,
        "interpretation_rule": (
            "sigma competition supported if detach improves NLL-selected RMSE "
            "by >=1% on average and wins >=80% of paired seeds"
        ),
        "test_record_evaluated": False,
    }
    config["suite_protocol_fingerprint"] = _root_protocol_fingerprint(root)
    fingerprint = canonical_fingerprint(config)
    config["protocol_fingerprint"] = fingerprint
    atomic_json_dump(config, stage_dir / "config.json")

    initial_hashes: dict[str, dict[str, str]] = {}
    for seed in seeds:
        by_arm: dict[str, str] = {}
        for arm_name, detach in arms_spec:
            diagnostic.set_seed(seed, True)
            probe = JointDirectGRUForecaster(
                in_channels=dataset.n_channels,
                horizon=diagnostic.base.TARGET_SAMPLES,
                hidden_channels=args.hidden_channels,
                num_layers=1,
                dropout=args.dropout,
                sigma_state_detach=detach,
            )
            by_arm[arm_name] = state_sha256(probe.state_dict())
        if len(set(by_arm.values())) != 1:
            raise AssertionError("Sigma-detach arms do not share initial weights")
        initial_hashes[str(seed)] = by_arm
    atomic_json_dump(initial_hashes, stage_dir / "initial_state_hashes.json")

    summaries_by_arm: dict[str, list[dict[str, Any]]] = {}
    arms: dict[str, Any] = {}
    keys = (
        "best_validation_nll",
        "nll_selected_validation_rmse",
        "minimum_validation_rmse_seen",
        "best_step",
        "minimum_validation_rmse_step",
        "cumulative_optimizer_steps",
        "last_five_validation_nll_slope_per_epoch",
        "elapsed_seconds",
    )
    for arm_name, detach in arms_spec:
        summaries: list[dict[str, Any]] = []
        for seed in seeds:
            summaries.append(
                train_joint_run(
                    run_dir=stage_dir / "arms" / arm_name / "runs" / f"seed_{seed}",
                    model_factory=lambda d=detach: JointDirectGRUForecaster(
                        in_channels=dataset.n_channels,
                        horizon=diagnostic.base.TARGET_SAMPLES,
                        hidden_channels=args.hidden_channels,
                        num_layers=1,
                        dropout=args.dropout,
                        sigma_state_detach=d,
                    ),
                    seed=seed,
                    dataset=dataset,
                    windows=windows,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    scaler=scaler,
                    horizon_samples=diagnostic.base.TARGET_SAMPLES,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    max_steps=args.max_steps,
                    min_steps=args.min_steps,
                    patience=args.patience,
                    protocol_fingerprint=fingerprint,
                    device=device,
                    amp=args.amp,
                )
            )
        summaries_by_arm[arm_name] = summaries
        arm = aggregate_run_summaries(summaries, keys)
        arm.update(
            {
                "sigma_state_detach": detach,
                "parameter_count": summaries[0]["parameter_count"],
            }
        )
        arms[arm_name] = arm
        atomic_json_dump(arm, stage_dir / "arms" / arm_name / "aggregate.json")
        diagnostic.write_csv(
            stage_dir / "arms" / arm_name / "run_table.csv",
            [
                {
                    "seed": item["seed"],
                    "stop_reason": item["stop_reason"],
                    **{key: item[key] for key in keys},
                }
                for item in summaries
            ],
        )

    rmse_comparison = _paired_comparison(
        summaries_by_arm["joint"],
        summaries_by_arm["sigma_state_detach"],
        "nll_selected_validation_rmse",
    )
    nll_comparison = _paired_comparison(
        summaries_by_arm["joint"],
        summaries_by_arm["sigma_state_detach"],
        "best_validation_nll",
    )
    required = math.ceil(0.8 * len(seeds))
    supported = bool(
        rmse_comparison["relative_gain_stats"]["mean"] >= 0.01
        and rmse_comparison["candidate_win_count"] >= required
    )
    aggregate = {
        "arms": arms,
        "paired_detach_vs_joint_rmse": rmse_comparison,
        "paired_detach_vs_joint_nll": nll_comparison,
        "sigma_gradient_competition_supported": supported,
        "required_seed_wins": required,
        "scope": (
            "Detachment blocks sigma-loss gradients into the encoder; sigma still "
            "weights the mean term in joint NLL, so this is not equivalent to mean-only."
        ),
    }
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    _finish_stage(stage_dir, stage, fingerprint)
    return aggregate


def run_decoder_stage(
    args: argparse.Namespace,
    root: Path,
    seeds: Sequence[int],
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    support: Mapping[str, Any],
    device: torch.device,
    selected_horizon: int,
) -> dict[str, Any]:
    stage = "05_decoder"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)
    comparison_horizons = sorted({int(selected_horizon), 128})
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "purpose": "Test whether the direct 9xH output map is the mean bottleneck",
        "decoders": list(DECODER_NAMES),
        "comparison_horizons_samples": comparison_horizons,
        "same_encoder_architecture": True,
        "identical_encoder_initial_state_within_seed_and_horizon": True,
        "decoder_parameter_matching": "nearest width to direct decoder budget",
        "maximum_decoder_parameter_budget_relative_error": 0.03,
        "teacher_forcing": False,
        "support": support,
        "seeds": list(seeds),
        "training": _mean_stage_config(
            args, stage, selected_horizon, seeds, support, "common_encoder_decoder_ablation"
        ),
        "bottleneck_rule": (
            "supported if a non-direct decoder improves paired validation RMSE "
            "by >=2% on average and wins >=80% of seeds"
        ),
        "test_record_evaluated": False,
    }
    config["suite_protocol_fingerprint"] = _root_protocol_fingerprint(root)
    fingerprint = canonical_fingerprint(config)
    config["protocol_fingerprint"] = fingerprint
    atomic_json_dump(config, stage_dir / "config.json")

    initial_hashes: dict[str, Any] = {}
    parameter_audit: dict[str, Any] = {}
    for horizon in comparison_horizons:
        horizon_key = f"h{horizon:03d}"
        initial_hashes[horizon_key] = {}
        parameter_audit[horizon_key] = {}
        for seed in seeds:
            hashes: dict[str, str] = {}
            for decoder in DECODER_NAMES:
                diagnostic.set_seed(seed, True)
                probe = GRUMeanForecaster(
                    in_channels=dataset.n_channels,
                    horizon=horizon,
                    hidden_channels=args.hidden_channels,
                    num_layers=1,
                    dropout=args.dropout,
                    decoder=decoder,
                )
                hashes[decoder] = _encoder_sha256(probe)
                ratio = float(
                    probe.model_config()["decoder_to_direct_parameter_ratio"]
                )
                if abs(ratio - 1.0) > 0.03:
                    raise AssertionError(
                        f"Decoder parameter match failed: {decoder} ratio={ratio}"
                    )
                if str(seed) == str(seeds[0]):
                    parameter_audit[horizon_key][decoder] = {
                        "total_parameters": int(parameter_count(probe)),
                        "model_config": probe.model_config(),
                    }
            if len(set(hashes.values())) != 1:
                raise AssertionError("Decoder arms do not share encoder initialisation")
            initial_hashes[horizon_key][str(seed)] = hashes
    atomic_json_dump(initial_hashes, stage_dir / "initial_encoder_hashes.json")
    atomic_json_dump(parameter_audit, stage_dir / "parameter_audit.json")

    all_summaries: dict[int, dict[str, list[dict[str, Any]]]] = {}
    horizon_results: dict[str, Any] = {}
    keys = (
        "best_validation_rmse",
        "best_validation_mae",
        "rmse_skill_vs_persistence",
        "best_step",
        "cumulative_optimizer_steps",
        "last_five_validation_rmse_slope_per_epoch",
        "gradient_clip_step_fraction",
        "elapsed_seconds",
    )
    required = math.ceil(0.8 * len(seeds))
    for horizon in comparison_horizons:
        horizon_key = f"h{horizon:03d}"
        all_summaries[horizon] = {}
        arm_results: dict[str, Any] = {}
        for decoder in DECODER_NAMES:
            summaries: list[dict[str, Any]] = []
            for seed in seeds:
                summaries.append(
                    train_mean_run(
                        run_dir=(
                            stage_dir
                            / "arms"
                            / horizon_key
                            / decoder
                            / "runs"
                            / f"seed_{seed}"
                        ),
                        model_factory=lambda d=decoder, h=horizon: GRUMeanForecaster(
                            in_channels=dataset.n_channels,
                            horizon=h,
                            hidden_channels=args.hidden_channels,
                            num_layers=1,
                            dropout=args.dropout,
                            decoder=d,
                        ),
                        seed=seed,
                        dataset=dataset,
                        windows=windows,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        scaler=scaler,
                        horizon_samples=horizon,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        dropout=args.dropout,
                        max_steps=args.max_steps,
                        min_steps=args.min_steps,
                        patience=args.patience,
                        protocol_fingerprint=fingerprint,
                        device=device,
                        amp=args.amp,
                    )
                )
            all_summaries[horizon][decoder] = summaries
            arm = aggregate_run_summaries(summaries, keys)
            arm.update(
                {
                    "horizon_samples": horizon,
                    "parameter_count": summaries[0]["parameter_count"],
                    "converged_count": sum(
                        item["stop_reason"] == "validation_patience"
                        and abs(item["last_five_validation_rmse_slope_per_epoch"]) < 5e-4
                        for item in summaries
                    ),
                }
            )
            arm_results[decoder] = arm
            arm_dir = stage_dir / "arms" / horizon_key / decoder
            atomic_json_dump(arm, arm_dir / "aggregate.json")
            diagnostic.write_csv(
                arm_dir / "run_table.csv",
                [
                    {
                        "seed": item["seed"],
                        "stop_reason": item["stop_reason"],
                        **{key: item[key] for key in keys},
                    }
                    for item in summaries
                ],
            )
        comparisons = {
            decoder: _paired_comparison(
                all_summaries[horizon]["direct"],
                all_summaries[horizon][decoder],
                "best_validation_rmse",
            )
            for decoder in DECODER_NAMES
            if decoder != "direct"
        }
        best_decoder = min(
            DECODER_NAMES,
            key=lambda name: arm_results[name]["best_validation_rmse"]["mean"],
        )
        supported_alternatives = [
            name
            for name, comparison in comparisons.items()
            if comparison["relative_gain_stats"]["mean"] >= 0.02
            and comparison["candidate_win_count"] >= required
        ]
        horizon_results[horizon_key] = {
            "horizon_samples": horizon,
            "horizon_seconds": horizon_seconds(horizon),
            "arms": arm_results,
            "paired_vs_direct": comparisons,
            "selected_decoder": best_decoder,
            "decoder_bottleneck_supported": bool(supported_alternatives),
            "supported_alternatives": supported_alternatives,
        }
        atomic_json_dump(
            horizon_results[horizon_key],
            stage_dir / "arms" / horizon_key / "aggregate.json",
        )

    selected_key = f"h{int(selected_horizon):03d}"
    aggregate = {
        "horizons": horizon_results,
        "selected_horizon_samples": int(selected_horizon),
        "selected_horizon_key": selected_key,
        "selected_decoder": horizon_results[selected_key]["selected_decoder"],
        "decoder_bottleneck_supported_at_selected_horizon": horizon_results[
            selected_key
        ]["decoder_bottleneck_supported"],
    }
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    table_rows: list[dict[str, Any]] = []
    for horizon_key, result in horizon_results.items():
        for decoder, arm in result["arms"].items():
            table_rows.append(
                {
                    "horizon_key": horizon_key,
                    "horizon_samples": result["horizon_samples"],
                    "decoder": decoder,
                    "parameter_count": arm["parameter_count"],
                    "converged_count": arm["converged_count"],
                    "validation_rmse_mean": arm["best_validation_rmse"]["mean"],
                    "validation_rmse_std": arm["best_validation_rmse"]["std"],
                    "persistence_skill_mean": arm["rmse_skill_vs_persistence"]["mean"],
                    "selected": decoder == result["selected_decoder"],
                }
            )
    diagnostic.write_csv(stage_dir / "decoder_table.csv", table_rows)
    _finish_stage(stage_dir, stage, fingerprint)
    return aggregate


def collect_scaled_contexts(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect context-only arrays and verify their exact window order."""

    loader = diagnostic.base.sequence_loader(
        dataset, windows, indices, scaler, batch_size, False, 0, False
    )
    contexts: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for sequence, _, window_index in loader:
        context, _ = split_sequence(sequence, diagnostic.base.TARGET_SAMPLES)
        contexts.append(context.numpy())
        observed.append(window_index.numpy())
    context_array = np.concatenate(contexts)
    observed_array = np.concatenate(observed).astype(np.int64, copy=False)
    if not np.array_equal(observed_array, np.asarray(indices, dtype=np.int64)):
        raise AssertionError("Context collection changed window order")
    return context_array, observed_array


@torch.no_grad()
def routing_diagnostics(
    model: nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if hasattr(model, "cluster_probabilities"):
        probability_function = getattr(model, "cluster_probabilities")
    elif hasattr(model, "routing_probabilities"):
        probability_function = getattr(model, "routing_probabilities")
    else:
        return None
    loader = diagnostic.base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        0,
        device.type == "cuda",
    )
    model.eval()
    values: list[np.ndarray] = []
    for sequence, _, _ in loader:
        context, _ = split_sequence(sequence.to(device), 1)
        values.append(probability_function(context.float()).cpu().numpy())
    probabilities = np.concatenate(values).astype(np.float64)
    hard = np.argmax(probabilities, axis=1)
    mean_probability = probabilities.mean(axis=0)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    )
    normalized_entropy = entropy / math.log(probabilities.shape[1])
    return {
        "windows": int(len(probabilities)),
        "routes": int(probabilities.shape[1]),
        "mean_probability": mean_probability.tolist(),
        "hard_assignment_fraction": (
            np.bincount(hard, minlength=probabilities.shape[1]) / len(hard)
        ).tolist(),
        "mean_normalized_entropy": float(normalized_entropy.mean()),
        "minimum_mean_route_probability": float(mean_probability.min()),
        "maximum_mean_route_probability": float(mean_probability.max()),
    }


def run_modes_stage(
    args: argparse.Namespace,
    root: Path,
    seeds: Sequence[int],
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    support: Mapping[str, Any],
    device: torch.device,
    selected_horizon: int,
) -> dict[str, Any]:
    stage = "06_modes"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)

    train_context, observed_train = collect_scaled_contexts(
        dataset, windows, train_indices, scaler, args.batch_size
    )
    validation_context, observed_validation = collect_scaled_contexts(
        dataset, windows, validation_indices, scaler, args.batch_size
    )
    analyzer = GRUContextModeAnalyzer(
        k_candidates=(2, 3, 4, 5, 6),
        min_cluster_fraction=0.10,
        pca_components=0.95,
        random_state=42,
        n_init=20,
        channel_names=dataset.channel_names,
    )
    train_assignments = analyzer.fit_predict_train(train_context)
    validation_assignments = analyzer.assign(validation_context)
    cluster_summary = summarize_train_validation_clusters(
        train_assignments,
        validation_assignments,
        n_clusters=analyzer.selected_k_,
    )
    fit_summary = analyzer.fit_summary()
    atomic_json_dump(fit_summary, stage_dir / "mode_fit.json")
    atomic_json_dump(cluster_summary, stage_dir / "cluster_support.json")
    pca_components = (
        analyzer.pca_.components_
        if analyzer.pca_ is not None
        else np.empty((0, len(analyzer.feature_names_)), dtype=np.float64)
    )
    pca_mean = (
        analyzer.pca_.mean_
        if analyzer.pca_ is not None
        else np.empty((0,), dtype=np.float64)
    )
    atomic_npz_save(
        stage_dir / "frozen_mode_model.npz",
        train_window_indices=observed_train,
        validation_window_indices=observed_validation,
        train_assignments=train_assignments,
        validation_assignments=validation_assignments,
        feature_scaler_mean=analyzer.scaler_.mean_,
        feature_scaler_scale=analyzer.scaler_.scale_,
        pca_components=pca_components,
        pca_mean=pca_mean,
        kmeans_centers=analyzer.kmeans_.cluster_centers_,
        raw_to_canonical_cluster=analyzer.raw_to_canonical_cluster_,
    )
    mode_by_window = np.full(len(windows), -1, dtype=np.int64)
    mode_by_window[observed_train] = train_assignments
    mode_by_window[observed_validation] = validation_assignments

    arm_names = ("global_direct", "latent_soft_conditioned", "moe")

    def build_model(name: str) -> nn.Module:
        common = {
            "in_channels": dataset.n_channels,
            "horizon": int(selected_horizon),
            "hidden_channels": args.hidden_channels,
            "num_layers": 1,
            "dropout": args.dropout,
        }
        if name == "global_direct":
            return GRUMeanForecaster(**common, decoder="direct")
        if name == "latent_soft_conditioned":
            return ClusterConditionedGRUMeanForecaster(
                **common, n_clusters=analyzer.selected_k_
            )
        if name == "moe":
            return MoEGRUMeanForecaster(
                **common, n_experts=analyzer.selected_k_
            )
        raise ValueError(name)

    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "purpose": "Measure normal-regime heterogeneity and test context-only routing",
        "mode_fit": fit_summary,
        "mode_assignment": (
            "fit scaler/PCA/KMeans on train contexts only; validation uses frozen assign"
        ),
        "targets_labels_and_residuals_used_for_clustering": False,
        "mode_labels_used_as_predictor_inputs": False,
        "predictor_arms": list(arm_names),
        "conditioning": "learned soft context-only route; no target and no teacher forcing",
        "horizon_samples": int(selected_horizon),
        "support": support,
        "seeds": list(seeds),
        "training": _mean_stage_config(
            args, stage, selected_horizon, seeds, support, "normal_mode_ablation"
        ),
        "heterogeneity_rule": "max/min global per-mode RMSE >=1.25",
        "conditioning_benefit_rule": (
            "candidate improves paired validation RMSE >=2% on average and wins >=80% seeds"
        ),
        "test_record_evaluated": False,
    }
    config["suite_protocol_fingerprint"] = _root_protocol_fingerprint(root)
    fingerprint = canonical_fingerprint(config)
    config["protocol_fingerprint"] = fingerprint
    atomic_json_dump(config, stage_dir / "config.json")

    initial_hashes: dict[str, dict[str, str]] = {}
    for seed in seeds:
        hashes: dict[str, str] = {}
        for name in arm_names:
            diagnostic.set_seed(seed, True)
            hashes[name] = _encoder_sha256(build_model(name))
        if len(set(hashes.values())) != 1:
            raise AssertionError("Mode arms do not share encoder initialisation")
        initial_hashes[str(seed)] = hashes
    atomic_json_dump(initial_hashes, stage_dir / "initial_encoder_hashes.json")

    summaries_by_arm: dict[str, list[dict[str, Any]]] = {}
    arms: dict[str, Any] = {}
    keys = (
        "best_validation_rmse",
        "best_validation_mae",
        "rmse_skill_vs_persistence",
        "best_step",
        "cumulative_optimizer_steps",
        "last_five_validation_rmse_slope_per_epoch",
        "gradient_clip_step_fraction",
        "elapsed_seconds",
    )
    for name in arm_names:
        summaries: list[dict[str, Any]] = []
        for seed in seeds:
            run_dir = stage_dir / "arms" / name / "runs" / f"seed_{seed}"
            summary = train_mean_run(
                run_dir=run_dir,
                model_factory=lambda n=name: build_model(n),
                seed=seed,
                dataset=dataset,
                windows=windows,
                train_indices=train_indices,
                validation_indices=validation_indices,
                scaler=scaler,
                horizon_samples=selected_horizon,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                max_steps=args.max_steps,
                min_steps=args.min_steps,
                patience=args.patience,
                protocol_fingerprint=fingerprint,
                device=device,
                amp=args.amp,
                mode_by_window=mode_by_window,
            )
            if name != "global_direct":
                model = build_model(name).to(device)
                checkpoint = torch.load(
                    run_dir / "best.pt", map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                summary["routing"] = {
                    "train": routing_diagnostics(
                        model,
                        dataset,
                        windows,
                        train_indices,
                        scaler,
                        args.batch_size,
                        device,
                    ),
                    "validation": routing_diagnostics(
                        model,
                        dataset,
                        windows,
                        validation_indices,
                        scaler,
                        args.batch_size,
                        device,
                    ),
                }
                atomic_json_dump(summary, run_dir / "summary.json")
            summaries.append(summary)
        summaries_by_arm[name] = summaries
        arm = aggregate_run_summaries(summaries, keys)
        arm.update(
            {
                "parameter_count": summaries[0]["parameter_count"],
                "converged_count": sum(
                    item["stop_reason"] == "validation_patience"
                    and abs(item["last_five_validation_rmse_slope_per_epoch"]) < 5e-4
                    for item in summaries
                ),
            }
        )
        arms[name] = arm
        atomic_json_dump(arm, stage_dir / "arms" / name / "aggregate.json")
        diagnostic.write_csv(
            stage_dir / "arms" / name / "run_table.csv",
            [
                {
                    "seed": item["seed"],
                    "stop_reason": item["stop_reason"],
                    **{key: item[key] for key in keys},
                }
                for item in summaries
            ],
        )

    per_cluster: dict[str, Any] = {}
    observed_cluster_rmse: list[float] = []
    for cluster_id in range(analyzer.selected_k_):
        cluster_values: list[float] = []
        window_counts: list[int] = []
        for summary in summaries_by_arm["global_direct"]:
            per_mode = summary["best"]["validation"].get("per_mode", {})
            item = per_mode.get(str(cluster_id))
            if item is not None:
                cluster_values.append(float(item["rmse_scaled"]))
                window_counts.append(int(item["windows"]))
        if cluster_values:
            stats = numeric_stats(cluster_values)
            per_cluster[str(cluster_id)] = {
                "rmse": stats,
                "validation_windows": window_counts[0],
            }
            observed_cluster_rmse.append(stats["mean"])
        else:
            per_cluster[str(cluster_id)] = {
                "rmse": None,
                "validation_windows": 0,
            }
    heterogeneity_ratio = (
        max(observed_cluster_rmse) / max(min(observed_cluster_rmse), 1e-12)
        if len(observed_cluster_rmse) >= 2
        else 1.0
    )
    comparisons = {
        name: _paired_comparison(
            summaries_by_arm["global_direct"],
            summaries_by_arm[name],
            "best_validation_rmse",
        )
        for name in arm_names
        if name != "global_direct"
    }
    required = math.ceil(0.8 * len(seeds))
    supported_conditioners = [
        name
        for name, comparison in comparisons.items()
        if comparison["relative_gain_stats"]["mean"] >= 0.02
        and comparison["candidate_win_count"] >= required
    ]
    selected_arm = min(
        arm_names,
        key=lambda name: arms[name]["best_validation_rmse"]["mean"],
    )
    aggregate = {
        "selected_horizon_samples": int(selected_horizon),
        "selected_k": analyzer.selected_k_,
        "selected_silhouette": analyzer.selected_silhouette_,
        "cluster_support": cluster_summary,
        "global_validation_per_cluster": per_cluster,
        "heterogeneity_rmse_max_min_ratio": heterogeneity_ratio,
        "substantial_normal_mode_heterogeneity": heterogeneity_ratio >= 1.25,
        "arms": arms,
        "paired_vs_global": comparisons,
        "conditioning_supported": bool(supported_conditioners),
        "supported_conditioners": supported_conditioners,
        "selected_arm": selected_arm,
        "scope": (
            "Hard clusters stratify errors only. Conditioned/MoE predictors learn "
            "their own soft context-only routes, so validation targets never leak into routing."
        ),
    }
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    diagnostic.write_csv(
        stage_dir / "mode_arm_table.csv",
        [
            {
                "arm": name,
                "parameters": arm["parameter_count"],
                "converged_count": arm["converged_count"],
                "validation_rmse_mean": arm["best_validation_rmse"]["mean"],
                "validation_rmse_std": arm["best_validation_rmse"]["std"],
                "persistence_skill_mean": arm["rmse_skill_vs_persistence"]["mean"],
                "selected": name == selected_arm,
            }
            for name, arm in arms.items()
        ],
    )
    _finish_stage(stage_dir, stage, fingerprint)
    return aggregate


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _fmt(value: float, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def _build_report(
    overfit: Mapping[str, Any],
    long_mean: Mapping[str, Any],
    sigma_detach: Mapping[str, Any],
    horizon: Mapping[str, Any],
    decoder: Mapping[str, Any],
    modes: Mapping[str, Any],
    final: Mapping[str, Any],
) -> str:
    horizon_rows = []
    for horizon_id, samples in HORIZONS:
        arm = horizon["arms"][horizon_id]
        horizon_rows.append(
            "| {seconds:g} | {samples} | {rmse} ± {std} | {skill:.2%} | {conv}/{runs} |".format(
                seconds=horizon_seconds(samples),
                samples=samples,
                rmse=_fmt(arm["best_validation_rmse"]["mean"]),
                std=_fmt(arm["best_validation_rmse"]["std"]),
                skill=arm["rmse_skill_vs_persistence"]["mean"],
                conv=arm["converged_count"],
                runs=arm["runs"],
            )
        )
    selected_key = decoder["selected_horizon_key"]
    decoder_rows = []
    for name, arm in decoder["horizons"][selected_key]["arms"].items():
        decoder_rows.append(
            "| {name} | {params} | {rmse} ± {std} | {skill:.2%} | {conv}/{runs} |".format(
                name=name,
                params=arm["parameter_count"],
                rmse=_fmt(arm["best_validation_rmse"]["mean"]),
                std=_fmt(arm["best_validation_rmse"]["std"]),
                skill=arm["rmse_skill_vs_persistence"]["mean"],
                conv=arm["converged_count"],
                runs=arm["runs"],
            )
        )
    mode_rows = []
    for name, arm in modes["arms"].items():
        mode_rows.append(
            "| {name} | {params} | {rmse} ± {std} | {skill:.2%} | {conv}/{runs} |".format(
                name=name,
                params=arm["parameter_count"],
                rmse=_fmt(arm["best_validation_rmse"]["mean"]),
                std=_fmt(arm["best_validation_rmse"]["std"]),
                skill=arm["rmse_skill_vs_persistence"]["mean"],
                conv=arm["converged_count"],
                runs=arm["runs"],
            )
        )
    cause_lines = []
    cause_lines.append(
        "- 实现/张量对齐：{}。32 窗口最终 RMSE {}，初始 {}。".format(
            "排除为首要问题" if overfit["success"] else "仍是问题",
            _fmt(overfit["final_rmse_scaled"], 7),
            _fmt(overfit["initial_rmse_scaled"], 7),
        )
    )
    cause_lines.append(
        "- 2 秒均值训练预算：{}/{} 个种子满足预注册的验证收敛规则；平均技能值为 {:.2%}。".format(
            long_mean["mean_predictor_converged_count"],
            long_mean["runs"],
            long_mean["rmse_skill_vs_persistence"]["mean"],
        )
    )
    cause_lines.append(
        "- σ-head 对共享编码器的直接梯度干扰：{}；detach 相对 RMSE 平均改善 {:.2%}。".format(
            "有支持" if sigma_detach["sigma_gradient_competition_supported"] else "无充分支持",
            sigma_detach["paired_detach_vs_joint_rmse"]["relative_gain_stats"]["mean"],
        )
    )
    cause_lines.append(
        "- 预测长度：按预注册的纯预测规则选择 {} 秒；它是未来预测/等待 target 的时长，不是 FoG 提前预警量，也不把 6 个 TCN 残差块解释成 12 秒。".format(
            _fmt(horizon["selected_horizon_seconds"], 2)
        )
    )
    cause_lines.append(
        "- 固定协议下的解码器归纳偏置：{}；所选解码器为 {}。相同参数预算并不等于相同计算量或各架构已分别调到最优。".format(
            "有支持" if decoder["decoder_bottleneck_supported_at_selected_horizon"] else "无充分支持",
            decoder["selected_decoder"],
        )
    )
    cause_lines.append(
        "- context 区域预测难度异质性：全局模型各簇 RMSE 最大/最小比为 {}，{}；潜在 soft-routing/MoE {}。这些臂容量不同，不能把收益单独归因于硬聚类模式。".format(
            _fmt(modes["heterogeneity_rmse_max_min_ratio"], 3),
            "达到阈值" if modes["substantial_normal_mode_heterogeneity"] else "未达到阈值",
            "得到支持" if modes["conditioning_supported"] else "未得到稳定收益支持",
        )
    )
    selected = final["selected_candidate"]
    return f"""# S01 GRU 正常行为预测器顺序诊断报告

## 结论

最终验证候选为 `{selected['name']}`，预测长度 {selected['horizon_seconds']:g} 秒，{selected['run_count']} 种子平均验证 RMSE 为 {_fmt(selected['validation_rmse_mean'])}，相对持久性基线技能值为 {selected['persistence_skill_mean']:.2%}。按本实验预注册规则，该**架构族**在 {selected['run_count']} 种子聚合上{'达到' if final['architecture_family_convergence_achieved'] else '尚未达到'}验证收敛条件；固定落盘的 seed {final['fixed_seed']} checkpoint 则{'满足' if final['fixed_seed_checkpoint_convergence_achieved'] else '不满足'}同一单次运行规则，两者不混为一谈。

这里的“收敛”是操作性判据：32 窗口可过拟合、至少 80% 随机种子因验证耐心停止且末 5 次 RMSE 斜率绝对值小于 5e-4、并且平均 RMSE 至少优于持久性基线 5%。其中优化稳定性与相对基线有用性仍分别报告。多个 seed 只反映初始化/训练噪声变异，不是多份独立数据；978/295 个 1 秒步长窗口高度重叠。阈值均为工程阈值，它不是参数达到数学驻点的证明。

## 固定数据边界

- 仅使用源记录 S01R01：`S01_seg000` 全段与 `S01_seg001` 在样点 50944 处作时间切分，得到冻结的 978 个 clean-normal 训练窗口和 295 个 clean-normal 验证窗口；不存在 R03 数据。
- Robust Scaler 只用正常训练原始点拟合；所有短 horizon 都截取同一个 2 秒 target 的前缀。
- context 恒为 128 点（2 秒），步长不因本诊断改变。
- 本 suite 未打开 R02、未建 loader、未前向；因此本报告没有测试集泛化指标，也没有 FoG 分类指标。R02 在更早的分类器 pilot 中曾被评估，已不是 publication-level pristine test，此历史接触不能被本次隔离声明抹去。

## 逐步结果

{chr(10).join(cause_lines)}

### Horizon 消融

| 未来长度（秒） | 点数 | 验证 RMSE | 相对持久性技能 | 收敛种子 |
|---:|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

### 解码器消融（所选 horizon）

| 解码器 | 参数量 | 验证 RMSE | 相对持久性技能 | 收敛种子 |
|---|---:|---:|---:|---:|
{chr(10).join(decoder_rows)}

### 正常模式消融

训练 context 的 63 个统计特征经训练集 StandardScaler/PCA 后选择 k={modes['selected_k']}（silhouette={_fmt(modes['selected_silhouette'], 4)}）；验证集只由冻结聚类器分配。硬簇只用于分层误差，不作为预测器输入。conditioned/MoE 学习的是另一套 context-only soft route，且增加了参数；所以这是“潜在路由/额外容量”的探索性消融，不是对 KMeans 模式条件化的因果检验。

| 架构 | 参数量 | 验证 RMSE | 相对持久性技能 | 收敛种子 |
|---|---:|---:|---:|---:|
{chr(10).join(mode_rows)}

## 与 SCADA CNBM 的关键差异

SCADA 图中的 C→Y 是低频、宽特征的条件回归，289 长度窗口用右对齐、补零和 mask 处理不等长历史，并可用风机 embedding 表达设备差异；缺失值用正常训练均值填补，均值/标准差标准化，head 输出 log-variance，损失是 masked NLL。S01 FoG 是 64 Hz 连续 IMU：输入为 9×128 的强自相关时序，输出为 9×H；无效区间整窗排除，采用训练正常原始点的 median/IQR Robust Scaler，完整固定窗口无需 mask，尺度实现使用 σ/log-σ 约定。窗口数量并不等于独立样本数量，步态相位、站立/转向/直行等工况可能形成多模式或异方差条件分布，但本次聚类不能证明“多峰”。

因此迁移 SCADA 思路时保留的是“训练正常集专属预处理、μ 与 σ 明确建模、验证早停、冻结后再形成残差”，不能机械照搬 289 步 padding/mask 或把单被试窗口数当成大量独立工况。本实验先用 MSE 让 μ 真正学会未来，再用训练残差解析校准固定 σ；joint-NLL 的 detach 只用于判断 σ 分支是否干扰共享编码器。

## 选择偏差与下一阶段

最终 horizon/架构由同一验证集反复用于早停、调度和多臂选择，结果可能明显乐观；`final_predictor.pt` 固定使用预先指定的第一个种子，而不是挑选最优种子。下一步应冻结该方案，再在独立边界上评估。若 H<128，当前接收 `[9,128]` 残差的 TCN-M 不能直接复用：必须重设分类器输入长度，或明确用滚动残差缓冲补成 2 秒，并重新验证感受野与分类阈值。无论 H 多长，残差只能在对应 target 到达后形成；TCN-M 输入应使用训练残差解析校准的 σ，而不是 mean-only 模型的占位单位 σ。
"""


def _artifact_constructor_spec(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert descriptive model metadata into strict executable kwargs."""

    encoder = model_config["encoder"]
    common: dict[str, Any] = {
        "in_channels": int(model_config["in_channels"]),
        "horizon": int(model_config["horizon"]),
        "hidden_channels": int(encoder["hidden_channels"]),
        "num_layers": int(encoder["num_layers"]),
        "dropout": float(encoder["dropout"]),
    }
    model_name = str(model_config["name"])
    if model_name == "gru_mean":
        decoder = model_config["decoder"]
        common["decoder"] = str(decoder["name"])
        if decoder["name"] != "direct":
            common["decoder_width"] = int(decoder["width"])
        if decoder["name"] == "tcn":
            common.update(
                {
                    "tcn_dilations": list(decoder["dilations"]),
                    "tcn_kernel_size": int(decoder["kernel_size"]),
                    "decoder_dropout": float(decoder["dropout"]),
                }
            )
        return {"model_class": "GRUMeanForecaster", "constructor_kwargs": common}
    if model_name == "gru_cluster_conditioned_mean":
        common.update(
            {
                "n_clusters": int(model_config["n_clusters"]),
                "cluster_embedding_dim": int(
                    model_config["cluster_embedding_dim"]
                ),
                "routing_temperature": float(model_config["routing_temperature"]),
            }
        )
        return {
            "model_class": "ClusterConditionedGRUMeanForecaster",
            "constructor_kwargs": common,
        }
    if model_name == "gru_moe_mean":
        common.update(
            {
                "n_experts": int(model_config["n_experts"]),
                "routing_temperature": float(model_config["routing_temperature"]),
            }
        )
        return {"model_class": "MoEGRUMeanForecaster", "constructor_kwargs": common}
    raise ValueError(f"Unsupported final mean model config: {model_name!r}")


def run_finalize_stage(
    args: argparse.Namespace,
    root: Path,
    seeds: Sequence[int],
    scaler: RobustChannelScaler,
    suite_fingerprint: str,
) -> dict[str, Any]:
    stage = "07_finalize"
    stage_dir = root / stage
    if _stage_ready(stage_dir, args.resume) is not None:
        return _load_stage_summary(root, stage)
    overfit = _load_stage_summary(root, "01_overfit")
    long_mean = _load_stage_summary(root, "02_long_mean")
    sigma_detach = _load_stage_summary(root, "03_sigma_detach")
    horizon = _load_stage_summary(root, "04_horizon")
    decoder = _load_stage_summary(root, "05_decoder")
    modes = _load_stage_summary(root, "06_modes")
    selected_horizon = int(horizon["selected_horizon_samples"])
    selected_horizon_key = f"h{selected_horizon:03d}"
    required = math.ceil(0.8 * len(seeds))

    candidates: list[dict[str, Any]] = []

    def add_candidate(
        name: str,
        source: str,
        arm: Mapping[str, Any],
        run_dir: Path,
    ) -> None:
        converged_count = int(arm["converged_count"])
        skill = float(arm["rmse_skill_vs_persistence"]["mean"])
        candidates.append(
            {
                "name": name,
                "source": source,
                "horizon_samples": selected_horizon,
                "horizon_seconds": horizon_seconds(selected_horizon),
                "validation_rmse_mean": float(
                    arm["best_validation_rmse"]["mean"]
                ),
                "validation_rmse_std": float(
                    arm["best_validation_rmse"]["std"]
                ),
                "persistence_skill_mean": skill,
                "converged_count": converged_count,
                "required_converged_count": required,
                "run_count": int(arm["runs"]),
                "eligible": converged_count >= required and skill >= 0.05,
                "parameter_count": int(arm["parameter_count"]),
                "fixed_seed_run_dir": str(run_dir.relative_to(root)).replace("\\", "/"),
            }
        )

    for name, arm in decoder["horizons"][selected_horizon_key]["arms"].items():
        add_candidate(
            f"decoder_{name}",
            "05_decoder",
            arm,
            root
            / "05_decoder"
            / "arms"
            / selected_horizon_key
            / name
            / "runs"
            / f"seed_{seeds[0]}",
        )
    for name, arm in modes["arms"].items():
        if name == "global_direct":
            # This is the same pure-mean direct architecture already present
            # in the decoder stage; keep it as the mode baseline but do not
            # duplicate it in final cross-stage selection.
            continue
        add_candidate(
            f"mode_{name}",
            "06_modes",
            arm,
            root / "06_modes" / "arms" / name / "runs" / f"seed_{seeds[0]}",
        )
    eligible = [item for item in candidates if item["eligible"]]
    pool = eligible if eligible else candidates
    selected = min(
        pool,
        key=lambda item: (item["validation_rmse_mean"], item["parameter_count"]),
    )
    run_dir = root / selected["fixed_seed_run_dir"]
    checkpoint_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fixed_seed_converged = bool(
        source_summary["stop_reason"] == "validation_patience"
        and abs(source_summary["last_five_validation_rmse_slope_per_epoch"]) < 5e-4
        and source_summary["rmse_skill_vs_persistence"] >= 0.05
    )
    sigma = np.asarray(
        source_summary["fixed_sigma_calibration"]["sigma"], dtype=np.float32
    )
    constructor_spec = _artifact_constructor_spec(checkpoint["model_config"])
    final_checkpoint = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "suite_protocol_fingerprint": suite_fingerprint,
        "selection_scope": "validation-selected; R02 not evaluated",
        "selected_candidate": selected,
        "fixed_seed": int(seeds[0]),
        "fixed_seed_converged_by_operational_rule": fixed_seed_converged,
        "model_config": checkpoint["model_config"],
        **constructor_spec,
        "model_state": checkpoint["model_state"],
        "horizon_samples": selected_horizon,
        "horizon_seconds": horizon_seconds(selected_horizon),
        "fixed_sigma_definition": (
            "channel-by-horizon RMS of training residuals from selected checkpoint"
        ),
        "fixed_sigma": sigma,
        "fixed_sigma_sha256": diagnostic.array_sha256(sigma),
        "inference_contract": {
            "input": "Robust-Scaled context [batch,9,128]",
            "output": "mean and calibrated fixed sigma [batch,9,horizon]",
            "residual": "clip((target - mean) / fixed_sigma, -12, 12)",
            "loader": "cnbr_fog.gru_predictor_artifact.load_gru_predictor_artifact",
            "mean_model_unit_sigma_must_not_be_used": True,
        },
        "robust_scaler": scaler.as_dict(),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_summary_sha256": sha256_file(summary_path),
        "test_record_evaluated": False,
    }
    final_path = stage_dir / "final_predictor.pt"
    atomic_torch_save(final_checkpoint, final_path)
    architecture_converged = bool(overfit["success"] and selected["eligible"])
    convergence_achieved = bool(architecture_converged and fixed_seed_converged)
    aggregate = {
        "selection_rule": (
            "lowest multi-seed mean validation RMSE among candidates meeting >=80% "
            "patience+flat-slope convergence and >=5% persistence skill; fallback "
            "to lowest RMSE if none eligible"
        ),
        "required_converged_seed_count": required,
        "candidates": candidates,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected,
        "fixed_seed": int(seeds[0]),
        "overfit_success": bool(overfit["success"]),
        "architecture_family_convergence_achieved": architecture_converged,
        "fixed_seed_checkpoint_convergence_achieved": fixed_seed_converged,
        "convergence_achieved": convergence_achieved,
        "final_predictor": "final_predictor.pt",
        "final_predictor_sha256": sha256_file(final_path),
        "test_record_evaluated": False,
    }
    atomic_json_dump(aggregate, stage_dir / "aggregate.json")
    diagnostic.write_csv(stage_dir / "candidate_table.csv", candidates)
    report = _build_report(
        overfit, long_mean, sigma_detach, horizon, decoder, modes, aggregate
    )
    _atomic_text(stage_dir / "report.md", report)
    _atomic_text(root / "report.md", report)
    _finish_stage(stage_dir, stage, suite_fingerprint)
    return aggregate


def suite_protocol_payload(
    args: argparse.Namespace,
    stages: Sequence[str],
    seeds: Sequence[int],
    dataset: DaphnetDataset,
    support_metadata: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "cnbr_fog" / "data.py",
        REPO_ROOT / "cnbr_fog" / "gru_convergence_models.py",
        REPO_ROOT / "cnbr_fog" / "gru_mode_analysis.py",
        REPO_ROOT / "cnbr_fog" / "gru_predictor_artifact.py",
        REPO_ROOT / "cnbr_fog" / "models.py",
        REPO_ROOT / "cnbr_fog" / "nbm.py",
        REPO_ROOT / "cnbr_fog" / "nbm_representations.py",
        REPO_ROOT / "cnbr_fog" / "resume.py",
        SCRIPTS_DIR / "diagnose_daphnet_s01_gru_convergence.py",
        SCRIPTS_DIR / "run_daphnet_s01_gru_h200_tcnm.py",
    )
    data_dir = args.data_dir.resolve()
    input_paths = (
        data_dir / "manifest.csv",
        data_dir / "schema.json",
        data_dir / "records" / "S01_seg000.npz",
        data_dir / "records" / "S01_seg001.npz",
    )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_order": list(STAGE_ORDER),
        # Stage subsets are an execution/resume concern and deliberately do
        # not alter the scientific protocol fingerprint.
        "available_stages": list(STAGE_ORDER),
        "stage_subset_does_not_change_protocol": True,
        "data_dir": str(args.data_dir.resolve()),
        "records_loaded": [record.record_id for record in dataset.records],
        "record_lengths": {
            record.record_id: int(len(record.x)) for record in dataset.records
        },
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channels": list(dataset.channel_names),
        "context_samples": diagnostic.base.CONTEXT_SAMPLES,
        "master_target_samples": diagnostic.base.TARGET_SAMPLES,
        "support": support_metadata,
        "seeds": list(seeds),
        "hyperparameters": {
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "maximum_optimizer_steps": args.max_steps,
            "patience": args.patience,
            "minimum_optimizer_steps": args.min_steps,
            "overfit_maximum_optimizer_steps": args.overfit_max_steps,
            "amp": bool(args.amp),
        },
        "device_type": device.type,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in source_paths
        },
        "loaded_input_sha256": {
            str(path.relative_to(data_dir)).replace("\\", "/"): sha256_file(path)
            for path in input_paths
        },
        "test_boundary": {
            "record": diagnostic.base.TEST_RECORD,
            "array_opened": False,
            "evaluated": False,
        },
    }


def main() -> None:
    args = parse_args()
    stages, seeds = validate_args(args)
    device = diagnostic.resolve_device(args.device)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        support_metadata,
    ) = diagnostic.prepare_support(args.data_dir)
    protocol = suite_protocol_payload(
        args, stages, seeds, dataset, support_metadata, device
    )
    suite_fingerprint = canonical_fingerprint(protocol)
    protocol["protocol_fingerprint"] = suite_fingerprint
    config_path = root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != suite_fingerprint:
            raise RuntimeError(
                "Existing output protocol differs; choose a new output directory"
            )
    else:
        if any(root.iterdir()):
            raise FileExistsError(f"Output directory lacks config but is non-empty: {root}")
        atomic_json_dump(protocol, config_path)
        atomic_json_dump(scaler.as_dict(), root / "scaler.json")
        atomic_npz_save(
            root / "locked_support.npz",
            clean_normal_train_window_index=train_indices,
            clean_normal_validation_window_index=validation_indices,
        )
    runtime = {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "protocol_fingerprint": suite_fingerprint,
        "requested_stages_this_invocation": list(stages),
    }
    atomic_json_dump(runtime, root / "runtime.json")
    print(
        f"Protocol {suite_fingerprint} device={device} "
        f"train/val={len(train_indices)}/{len(validation_indices)}; "
        f"test_record={diagnostic.base.TEST_RECORD} excluded",
        flush=True,
    )

    results: dict[str, Any] = {}
    progress_path = root / "PROGRESS.json"
    if progress_path.exists():
        prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed_stages = list(prior_progress.get("completed_stages", []))
    else:
        completed_stages = []
    for stage in stages:
        print(f"=== stage: {stage} ===", flush=True)
        prerequisite = STAGE_PREREQUISITE.get(stage)
        if prerequisite is not None:
            prerequisite_summary = _load_stage_summary(
                root, STAGE_DIRECTORIES[prerequisite]
            )
            if prerequisite == "overfit" and not prerequisite_summary["success"]:
                raise RuntimeError(
                    "32-window overfit gate failed; later experiments are invalid"
                )
        if stage == "overfit":
            results[stage] = run_overfit_stage(
                args, root, dataset, windows, train_indices, scaler, device
            )
            if not results[stage]["success"]:
                raise RuntimeError(
                    "32-window overfit gate failed; stop before expensive stages"
                )
        elif stage == "long_mean":
            results[stage] = run_long_mean_stage(
                args,
                root,
                seeds,
                dataset,
                windows,
                train_indices,
                validation_indices,
                scaler,
                support_metadata,
                device,
            )
        elif stage == "sigma_detach":
            results[stage] = run_sigma_detach_stage(
                args,
                root,
                seeds,
                dataset,
                windows,
                train_indices,
                validation_indices,
                scaler,
                support_metadata,
                device,
            )
        elif stage == "horizon":
            results[stage] = run_horizon_stage(
                args,
                root,
                seeds,
                dataset,
                windows,
                train_indices,
                validation_indices,
                scaler,
                support_metadata,
                device,
            )
        elif stage == "decoder":
            horizon_result = results.get("horizon") or _load_stage_summary(
                root, "04_horizon"
            )
            results[stage] = run_decoder_stage(
                args,
                root,
                seeds,
                dataset,
                windows,
                train_indices,
                validation_indices,
                scaler,
                support_metadata,
                device,
                int(horizon_result["selected_horizon_samples"]),
            )
        elif stage == "modes":
            horizon_result = results.get("horizon") or _load_stage_summary(
                root, "04_horizon"
            )
            results[stage] = run_modes_stage(
                args,
                root,
                seeds,
                dataset,
                windows,
                train_indices,
                validation_indices,
                scaler,
                support_metadata,
                device,
                int(horizon_result["selected_horizon_samples"]),
            )
        elif stage == "finalize":
            results[stage] = run_finalize_stage(
                args, root, seeds, scaler, suite_fingerprint
            )
        else:
            raise AssertionError(stage)
        if stage not in completed_stages:
            completed_stages.append(stage)
        completed_stages.sort(key=STAGE_ORDER.index)
        atomic_json_dump(
            {
                "last_completed_stage": stage,
                "completed_stages": completed_stages,
                "updated_utc": utc_now(),
                "test_record_evaluated": False,
            },
            progress_path,
        )

    if "finalize" in results:
        artifacts = {}
        for relative in (
            "config.json",
            "scaler.json",
            "locked_support.npz",
            "report.md",
            "07_finalize/aggregate.json",
            "07_finalize/final_predictor.pt",
        ):
            path = root / relative
            artifacts[relative] = sha256_file(path)
        atomic_json_dump(
            {
                "status": "complete",
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": suite_fingerprint,
                "completed_utc": utc_now(),
                "convergence_achieved": results["finalize"][
                    "convergence_achieved"
                ],
                "architecture_family_convergence_achieved": results["finalize"][
                    "architecture_family_convergence_achieved"
                ],
                "fixed_seed_checkpoint_convergence_achieved": results["finalize"][
                    "fixed_seed_checkpoint_convergence_achieved"
                ],
                "test_record_evaluated": False,
                "artifacts": artifacts,
            },
            root / "DONE.json",
        )
    print(f"Results: {root}", flush=True)


if __name__ == "__main__":
    main()
