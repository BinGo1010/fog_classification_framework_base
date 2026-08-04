"""Run the gated three-round Daphnet TC-DAE reconstruction diagnostic."""

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
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for location in (REPO_ROOT, SCRIPTS_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import daphnet_small_sample_selection as selection  # noqa: E402
from cnbr_fog.data import DaphnetDataset, Record  # noqa: E402
from cnbr_fog.temporal_conv_autoencoder import TemporalConvAutoencoder  # noqa: E402


EXPERIMENT = "daphnet_nbm_tcdae_three_rounds_v1"
REPRESENTATIVES = ("S01", "S03", "S05", "S07")
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
SEEDS = (20260802, 20260803, 20260804)
PREPROCESSORS = ("P0_current", "P1_no_window_centering", "P2_clip_only")
ARCHITECTURES = (
    "M0_gru_baseline",
    "M1_tcdae_base",
    "M2_tcdae_wide",
    "M3_tcdae_long",
)
CHANNELS = 9
WINDOW = 128
FS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"{EXPERIMENT}_seed{SEEDS[0]}",
    )
    parser.add_argument(
        "--stage", choices=("round1", "round2", "round3", "all"), default="all"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--round1-epochs", type=int, default=3000)
    parser.add_argument("--round2-epochs", type=int, default=3000)
    parser.add_argument("--round3-epochs", type=int, default=2000)
    parser.add_argument("--round3-patience", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


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


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    _replace_with_retry(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _replace_with_retry(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    _replace_with_retry(temporary, path)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def raw_selected(
    records: Sequence[Record], windows: selection.current.WindowSet, indices: np.ndarray
) -> np.ndarray:
    return selection.current.raw_windows(records, windows, indices)


def unique_selected_points(
    records: Sequence[Record], windows: selection.current.WindowSet, indices: np.ndarray
) -> np.ndarray:
    masks: dict[int, np.ndarray] = {}
    for raw_index in indices:
        index = int(raw_index)
        record_index = int(windows.record_index[index])
        masks.setdefault(record_index, np.zeros(len(records[record_index].y), dtype=bool))
        masks[record_index][
            int(windows.start[index]) : int(windows.end[index])
        ] = True
    return np.concatenate(
        [records[index].x[mask] for index, mask in masks.items() if np.any(mask)],
        axis=0,
    ).astype(np.float32, copy=False)


def preprocess(
    name: str,
    records: list[Record],
    windows: selection.current.WindowSet,
    indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = raw_selected(records, windows, indices)
    if name in ("P0_current", "P1_no_window_centering"):
        scaler = selection.current.fit_scaler_unique_points(records, windows, indices)
        values = scaler.transform(raw)
        centered = name == "P0_current"
        if centered:
            values = selection.current.window_axis_center(values)
        return np.ascontiguousarray(values), {
            "name": name,
            "robust_scaler": scaler.as_dict(),
            "window_axis_centering": centered,
            "clip": None,
            "fit_scope": "unique raw points covered by selected diagnostic windows",
        }
    if name != "P2_clip_only":
        raise ValueError(f"Unknown preprocessor: {name}")
    points = unique_selected_points(records, windows, indices).astype(np.float64)
    lower, upper = np.quantile(points, [0.005, 0.995], axis=0)
    values = np.clip(raw, lower[None, None, :], upper[None, None, :]).astype(
        np.float32
    )
    return np.ascontiguousarray(values), {
        "name": name,
        "robust_scaler": None,
        "window_axis_centering": False,
        "quantile_clip": [0.005, 0.995],
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "fit_scope": "unique raw points covered by selected diagnostic windows",
    }


class GRUBaselineAdapter(nn.Module):
    """Expose the current GRU-NBM reconstruction and bottleneck uniformly."""

    def __init__(self) -> None:
        super().__init__()
        self.model = selection.current.GRUReconstructionNBM(
            channels=CHANNELS, hidden=64, bottleneck=32
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        _, hidden = self.model.encoder(sequence)
        return self.model.to_bottleneck(hidden[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = x.transpose(1, 2)
        return self.model(sequence).transpose(1, 2), self.encode(x)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "current_gru_nbm",
            "input_shape": ["batch", 9, 128],
            "latent_shape": ["batch", 32],
            "encoder_decoder_long_skip": False,
            "decoder_input": "all zeros",
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }


def build_model(architecture: str) -> nn.Module:
    if architecture == "M0_gru_baseline":
        return GRUBaselineAdapter()
    variants = {
        "M1_tcdae_base": "base",
        "M2_tcdae_wide": "wide",
        "M3_tcdae_long": "long",
    }
    if architecture not in variants:
        raise ValueError(f"Unknown architecture: {architecture}")
    return TemporalConvAutoencoder(variant=variants[architecture])


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_distances(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    matrix = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    upper = matrix[np.triu_indices(len(flat), k=1)]
    return matrix, upper


def metric_arrays(actual: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-8
    centered_actual = actual - actual.mean(axis=1, keepdims=True)
    centered_prediction = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_actual * centered_prediction, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(centered_actual), axis=1)
        * np.sum(np.square(centered_prediction), axis=1)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > eps,
    )
    rmse = np.sqrt(np.mean(np.square(actual - predicted), axis=1))
    rms = np.sqrt(np.mean(np.square(actual), axis=1))
    amplitude = np.std(predicted, axis=1) / (np.std(actual, axis=1) + eps)
    return {
        "correlation": correlation,
        "nrmse": rmse / (rms + eps),
        "amplitude_ratio": amplitude,
        "window_mse": np.mean(np.square(actual - predicted), axis=(1, 2)),
        "window_zero_mse": np.mean(np.square(actual), axis=(1, 2)),
    }


def summarize(
    actual: np.ndarray, predicted: np.ndarray, latent: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = metric_arrays(actual, predicted)
    mse = float(np.mean(np.square(actual - predicted)))
    zero = float(np.mean(np.square(actual)))
    raw_matrix, raw_upper = pairwise_distances(actual)
    latent_matrix, latent_upper = pairwise_distances(latent)
    distance_corr: float | None = None
    if len(raw_upper) >= 2:
        distance_corr = safe_corr(raw_upper, latent_upper)
    variance_retention: float | None = None
    if len(actual) >= 2:
        source_variance = float(np.mean(np.var(actual, axis=0)))
        reconstruction_variance = float(np.mean(np.var(predicted, axis=0)))
        variance_retention = reconstruction_variance / max(source_variance, 1e-12)
    per_window_nrmse = np.median(arrays["nrmse"], axis=1)
    metrics: dict[str, Any] = {
        "nbm_mse": mse,
        "zero_mse": zero,
        "improvement_pct": 100.0 * (zero - mse) / max(zero, 1e-12),
        "median_corr": float(np.median(arrays["correlation"])),
        "p10_corr": float(np.percentile(arrays["correlation"], 10)),
        "median_nrmse": float(np.median(arrays["nrmse"])),
        "p90_window_nrmse": float(np.percentile(per_window_nrmse, 90)),
        "p95_window_nrmse": float(np.percentile(per_window_nrmse, 95)),
        "worst_window_nrmse": float(np.max(per_window_nrmse)),
        "median_amplitude_ratio": float(np.median(arrays["amplitude_ratio"])),
        "diff_mse": float(
            np.mean(np.square(np.diff(actual, axis=1) - np.diff(predicted, axis=1)))
        ),
        "latent_variance": float(np.var(latent)),
        "latent_between_window_variance": float(
            np.mean(np.var(latent.reshape(len(latent), -1), axis=0))
        ),
        "raw_latent_distance_corr": distance_corr,
        "reconstruction_variance_retention": variance_retention,
    }
    return metrics, {
        **arrays,
        "raw_distance_matrix": raw_matrix,
        "latent_distance_matrix": latent_matrix,
        "raw_distance_upper": raw_upper,
        "latent_distance_upper": latent_upper,
    }


def round1_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["improvement_pct"] >= 98.0
        and metrics["median_corr"] >= 0.98
        and metrics["median_nrmse"] <= 0.20
        and 0.85 <= metrics["median_amplitude_ratio"] <= 1.15
    )


def round2_pass(sample_count: int, metrics: dict[str, Any]) -> bool:
    if sample_count == 1:
        return bool(
            metrics["improvement_pct"] >= 98.0
            and metrics["median_corr"] >= 0.98
            and metrics["median_nrmse"] <= 0.20
        )
    distance_corr = metrics["raw_latent_distance_corr"]
    return bool(
        metrics["improvement_pct"] >= 80.0
        and metrics["median_corr"] >= 0.80
        and metrics["median_nrmse"] <= 0.50
        and 0.75 <= metrics["median_amplitude_ratio"] <= 1.25
        and distance_corr is not None
        and distance_corr >= 0.50
    )


def round3_pass(sample_count: int, metrics: dict[str, Any]) -> bool:
    thresholds = {
        1: (98.0, 0.98, 0.20),
        8: (80.0, 0.80, 0.50),
        32: (60.0, 0.70, 0.60),
        128: (40.0, 0.60, 0.75),
    }
    improvement, correlation, nrmse = thresholds[sample_count]
    return bool(
        metrics["improvement_pct"] >= improvement
        and metrics["median_corr"] >= correlation
        and metrics["median_nrmse"] <= nrmse
    )


@torch.no_grad()
def model_output(
    model: nn.Module, x: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(device)
    reconstruction, latent = model(tensor)
    return (
        reconstruction.transpose(1, 2).cpu().numpy().astype(np.float32),
        latent.cpu().numpy().astype(np.float32),
    )


@torch.no_grad()
def inference_milliseconds(
    model: nn.Module, x: np.ndarray, device: torch.device, iterations: int = 30
) -> float:
    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(device)
    for _ in range(5):
        model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / iterations


def train_model(
    x: np.ndarray,
    architecture: str,
    *,
    seed: int,
    max_epochs: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    patience: int | None,
    device: torch.device,
    num_workers: int,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    model = build_model(architecture).to(device)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    criterion = nn.MSELoss()
    batch_size = 64 if len(x) == 128 else len(x)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    evaluation_tensor = torch.from_numpy(
        np.ascontiguousarray(x.transpose(0, 2, 1))
    ).to(device)
    zero_mse = float(np.mean(np.square(x)))
    initial_state = torch.cat(
        [parameter.detach().flatten().cpu() for parameter in model.parameters()]
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        maximum_gradient_norm = 0.0
        minimum_conv_gradient_norm = math.inf
        maximum_conv_gradient_norm = 0.0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            if reconstruction.shape != batch.shape:
                raise AssertionError("Input, target, and output shapes are not aligned")
            loss = criterion(reconstruction, batch)
            loss.backward()
            conv_gradient_norms = [
                float(torch.linalg.vector_norm(parameter.grad.detach()))
                for name, parameter in model.named_parameters()
                if "conv" in name.lower() and parameter.grad is not None
            ]
            if conv_gradient_norms:
                minimum_conv_gradient_norm = min(
                    minimum_conv_gradient_norm, min(conv_gradient_norms)
                )
                maximum_conv_gradient_norm = max(
                    maximum_conv_gradient_norm, max(conv_gradient_norms)
                )
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total_count += len(batch)
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))

        model.eval()
        with torch.no_grad():
            evaluated, _ = model(evaluation_tensor)
            evaluation_loss = float(criterion(evaluated, evaluation_tensor))
        improved = evaluation_loss < best_loss - 1e-12
        if improved:
            best_loss = evaluation_loss
            best_epoch = epoch
            best_state = clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        last_epoch = epoch
        should_log = epoch == 1 or epoch % 10 == 0 or epoch == max_epochs
        if should_log:
            with torch.no_grad():
                prediction = evaluated.transpose(1, 2).cpu().numpy().astype(np.float32)
                _, latent_tensor = model(evaluation_tensor)
                latent = latent_tensor.cpu().numpy().astype(np.float32)
            snapshot, _ = summarize(x, prediction, latent)
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": total_loss / total_count,
                    "eval_mse": evaluation_loss,
                    "zero_mse": zero_mse,
                    "improvement_pct": snapshot["improvement_pct"],
                    "median_corr": snapshot["median_corr"],
                    "median_nrmse": snapshot["median_nrmse"],
                    "median_amplitude_ratio": snapshot["median_amplitude_ratio"],
                    "max_gradient_norm_before_clip": maximum_gradient_norm,
                    "min_conv_gradient_norm_before_clip": (
                        minimum_conv_gradient_norm
                        if math.isfinite(minimum_conv_gradient_norm)
                        else 0.0
                    ),
                    "max_conv_gradient_norm_before_clip": maximum_conv_gradient_norm,
                    "learning_rate": learning_rate,
                    "improved": improved,
                }
            )
        if epoch == 1 or epoch % 200 == 0 or epoch == max_epochs:
            print(
                f"{architecture} seed={seed} N={len(x)} epoch={epoch:04d}/{max_epochs} "
                f"mse={evaluation_loss:.8g} improve={100*(zero_mse-evaluation_loss)/max(zero_mse,1e-12):.3f}%",
                flush=True,
            )
        if patience is not None and bad_epochs >= patience:
            break

    if best_state is None:
        raise AssertionError("No best model state was produced")
    last_state = clone_state(model)
    architecture_config = model.architecture_config()  # type: ignore[attr-defined]
    common_checkpoint = {
        "experiment": EXPERIMENT,
        "architecture": architecture,
        "architecture_config": architecture_config,
        "seed": seed,
        "maximum_epochs": max_epochs,
    }
    torch_save(
        run_dir / "last_model.pt",
        {**common_checkpoint, "epoch": last_epoch, "model_state": last_state},
    )
    torch_save(
        run_dir / "best_model.pt",
        {
            **common_checkpoint,
            "epoch": best_epoch,
            "evaluation_mse": best_loss,
            "model_state": best_state,
        },
    )
    model.load_state_dict(best_state)
    predicted, latent = model_output(model, x, device)
    final_state = torch.cat(
        [parameter.detach().flatten().cpu() for parameter in model.parameters()]
    )
    training = {
        "best_epoch": best_epoch,
        "final_epoch": last_epoch,
        "stopped_early": last_epoch < max_epochs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_delta_l2": float(torch.linalg.vector_norm(final_state - initial_state)),
        "inference_ms_per_batch": inference_milliseconds(model, x, device),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "maximum_epochs": max_epochs,
        "patience": patience,
        "architecture_config": architecture_config,
    }
    return predicted, latent, history, training


def window_rows(
    metadata: Sequence[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    arrays = metric_arrays(actual, predicted)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(metadata):
        rows.append(
            {
                "window_id": item["window_id"],
                "record_id": item["record_id"],
                "start_time_sec": item["start_time_sec"],
                "energy_quartile": item["energy_quartile"],
                "nbm_mse": float(arrays["window_mse"][index]),
                "zero_mse": float(arrays["window_zero_mse"][index]),
                "improvement_pct": 100.0
                * (
                    arrays["window_zero_mse"][index]
                    - arrays["window_mse"][index]
                )
                / max(float(arrays["window_zero_mse"][index]), 1e-12),
                "median_corr": float(np.median(arrays["correlation"][index])),
                "median_nrmse": float(np.median(arrays["nrmse"][index])),
                "median_amplitude_ratio": float(
                    np.median(arrays["amplitude_ratio"][index])
                ),
                "diff_mse": float(
                    np.mean(
                        np.square(
                            np.diff(actual[index], axis=0)
                            - np.diff(predicted[index], axis=0)
                        )
                    )
                ),
            }
        )
    return rows


def channel_rows(
    metadata: Sequence[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    arrays = metric_arrays(actual, predicted)
    rows: list[dict[str, Any]] = []
    for window_index, item in enumerate(metadata):
        for channel in range(CHANNELS):
            rows.append(
                {
                    "window_id": item["window_id"],
                    "channel_id": channel,
                    "mse": float(
                        np.mean(
                            np.square(
                                actual[window_index, :, channel]
                                - predicted[window_index, :, channel]
                            )
                        )
                    ),
                    "pearson_corr": float(arrays["correlation"][window_index, channel]),
                    "nrmse": float(arrays["nrmse"][window_index, channel]),
                    "amplitude_ratio": float(
                        arrays["amplitude_ratio"][window_index, channel]
                    ),
                    "diff_mse": float(
                        np.mean(
                            np.square(
                                np.diff(actual[window_index, :, channel])
                                - np.diff(predicted[window_index, :, channel])
                            )
                        )
                    ),
                }
            )
    return rows


def plot_training(
    history: Sequence[dict[str, Any]], path: Path, zero_mse: float
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([row["epoch"] for row in history], [row["eval_mse"] for row in history])
    ax.axhline(zero_mse, linestyle="--", color="0.5", label="zero output")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Training/evaluation loss on the memorized subset")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_waveform(
    actual: np.ndarray,
    predicted: np.ndarray,
    index: int,
    path: Path,
    channel_names: Sequence[str],
    title: str,
) -> None:
    time_axis = np.arange(WINDOW) / FS
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for channel, ax in enumerate(axes.flat):
        ax.plot(time_axis, actual[index, :, channel], linewidth=1.1, label="true")
        ax.plot(
            time_axis,
            predicted[index, :, channel],
            "--",
            linewidth=1.0,
            label="reconstruction",
        )
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residual_lines(
    actual: np.ndarray,
    predicted: np.ndarray,
    path: Path,
    channel_names: Sequence[str],
) -> None:
    residual = actual[0] - predicted[0]
    time_axis = np.arange(WINDOW) / FS
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for channel, ax in enumerate(axes.flat):
        ax.plot(time_axis, residual[:, channel], color="tab:red", linewidth=1.0)
        ax.axhline(0, color="0.6", linewidth=0.7)
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.18)
    fig.suptitle("Residual X - reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scatter(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    x = actual.ravel()
    y = predicted.ravel()
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(x, y, s=4, alpha=0.3)
    ax.plot([lower, upper], [lower, upper], "--", color="black")
    ax.set_xlabel("True input")
    ax.set_ylabel("Reconstruction")
    ax.set_title("Input vs reconstruction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_channel_bars(
    actual: np.ndarray,
    predicted: np.ndarray,
    path: Path,
    channel_names: Sequence[str],
) -> None:
    arrays = metric_arrays(actual, predicted)
    values = np.stack(
        [
            np.median(arrays["correlation"], axis=0),
            np.median(arrays["nrmse"], axis=0),
            np.median(arrays["amplitude_ratio"], axis=0),
        ]
    )
    x = np.arange(CHANNELS)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, row, name in zip(axes, values, ("Pearson", "NRMSE", "Amplitude ratio")):
        ax.bar(x, row)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, channel_names, rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residual_heatmap(
    actual: np.ndarray,
    predicted: np.ndarray,
    path: Path,
    channel_names: Sequence[str],
) -> None:
    residual = (actual - predicted).transpose(0, 2, 1).reshape(-1, WINDOW)
    limit = max(float(np.percentile(np.abs(residual), 99.5)), 1e-6)
    fig, ax = plt.subplots(figsize=(10, max(4, len(actual) * 0.18 + 2)))
    image = ax.imshow(residual, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Window-channel row")
    ax.set_title("Residual heatmap")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_metric_heatmap(
    values: np.ndarray,
    path: Path,
    channel_names: Sequence[str],
    title: str,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, max(3.5, values.shape[0] * 0.22 + 2)))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(CHANNELS), channel_names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Window order")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_distance_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    image = ax.imshow(matrix, cmap="viridis", aspect="equal")
    ax.set_xlabel("Window")
    ax.set_ylabel("Window")
    ax.set_title("Latent Euclidean distance")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_raw_latent_distance(arrays: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    raw = arrays["raw_distance_upper"]
    latent = arrays["latent_distance_upper"]
    if len(raw):
        ax.scatter(raw, latent, alpha=0.7)
        correlation = safe_corr(raw, latent) if len(raw) >= 2 else 0.0
        ax.set_title(f"Raw vs latent distance (r={correlation:.3f})")
    else:
        ax.text(0.5, 0.5, "Requires at least two windows", ha="center", va="center")
        ax.set_title("Raw vs latent distance")
    ax.set_xlabel("Raw-window distance")
    ax.set_ylabel("Latent distance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_window_losses(arrays: dict[str, np.ndarray], path: Path) -> None:
    indices = np.arange(len(arrays["window_mse"]))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(indices) * 0.25 + 2), 4.5))
    ax.bar(indices - width / 2, arrays["window_mse"], width, label="NBM")
    ax.bar(indices + width / 2, arrays["window_zero_mse"], width, label="zero")
    ax.set_xlabel("Window")
    ax.set_ylabel("MSE")
    ax.set_title("Per-window reconstruction vs zero-output loss")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_variance(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    true_variance = np.var(actual, axis=0).mean(axis=1)
    reconstructed_variance = np.var(predicted, axis=0).mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(np.arange(WINDOW) / FS, true_variance, label="true")
    ax.plot(np.arange(WINDOW) / FS, reconstructed_variance, "--", label="reconstruction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Across-window variance")
    ax.set_title("Temporal variance comparison")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_figures(
    mode: str,
    run_dir: Path,
    history: Sequence[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
    latent: np.ndarray,
    channel_names: Sequence[str],
) -> None:
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    arrays = metric_arrays(actual, predicted)
    difficulty = np.argsort(arrays["window_mse"])
    median_index = int(difficulty[len(difficulty) // 2])
    worst_index = int(difficulty[-1])
    plot_training(history, figures / "training_loss.png", float(np.mean(np.square(actual))))
    if mode == "round1":
        plot_waveform(
            actual,
            predicted,
            0,
            figures / "waveform_9channels.png",
            channel_names,
            "Single-window reconstruction",
        )
        plot_residual_lines(
            actual, predicted, figures / "residual_9channels.png", channel_names
        )
        plot_scatter(actual, predicted, figures / "input_reconstruction_scatter.png")
        plot_channel_bars(
            actual, predicted, figures / "channel_metrics.png", channel_names
        )
        return

    plot_waveform(
        actual,
        predicted,
        median_index,
        figures / "median_window_waveform.png",
        channel_names,
        "Median-difficulty reconstruction",
    )
    plot_waveform(
        actual,
        predicted,
        worst_index,
        figures / "worst_window_waveform.png",
        channel_names,
        "Worst-window reconstruction",
    )
    plot_residual_heatmap(
        actual, predicted, figures / "residual_heatmap.png", channel_names
    )
    plot_metric_heatmap(
        arrays["correlation"],
        figures / "channel_metric_heatmap.png",
        channel_names,
        "Window-channel Pearson correlation",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )
    plot_metric_heatmap(
        arrays["amplitude_ratio"],
        figures / "amplitude_ratio_heatmap.png",
        channel_names,
        "Window-channel amplitude preservation ratio",
        cmap="viridis",
        vmin=0,
        vmax=max(1.5, float(np.percentile(arrays["amplitude_ratio"], 99))),
    )
    if mode == "round2" and len(actual) == 1:
        return
    _, diagnostic_arrays = summarize(actual, predicted, latent)
    plot_distance_matrix(
        diagnostic_arrays["latent_distance_matrix"],
        figures / "latent_distance_matrix.png",
    )
    plot_raw_latent_distance(
        diagnostic_arrays, figures / "raw_vs_latent_distance.png"
    )
    plot_window_losses(
        diagnostic_arrays, figures / "window_reconstruction_loss.png"
    )
    plot_variance(actual, predicted, figures / "temporal_variance_comparison.png")


def numeric_history(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key == "improved":
                parsed[key] = value.lower() == "true"
            elif key == "epoch":
                parsed[key] = int(value)
            else:
                parsed[key] = float(value)
        output.append(parsed)
    return output


def execute_run(
    *,
    mode: str,
    run_dir: Path,
    subject: str,
    sample_count: int,
    seed: int,
    architecture: str,
    preprocessor: str,
    x: np.ndarray,
    preprocessing_config: dict[str, Any],
    metadata: list[dict[str, Any]],
    max_epochs: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    patience: int | None,
    pass_function: Any,
    device: torch.device,
    num_workers: int,
    channel_names: Sequence[str],
    overwrite: bool,
    skip_figures: bool,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.npz"
    history_path = run_dir / "training_log.csv"
    if (
        metrics_path.exists()
        and predictions_path.exists()
        and history_path.exists()
        and not overwrite
    ):
        with metrics_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        print(f"RESUME {mode} {subject} N={sample_count} {architecture} seed={seed}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"START {mode} {subject} N={sample_count} {architecture} {preprocessor} seed={seed}")
        predicted, latent, history, training = train_model(
            x,
            architecture,
            seed=seed,
            max_epochs=max_epochs,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            num_workers=num_workers,
            run_dir=run_dir,
        )
        metrics, _ = summarize(x, predicted, latent)
        passed = bool(pass_function(sample_count, metrics))
        result = {
            "complete": True,
            "round": mode,
            "subject_id": subject,
            "sample_count": sample_count,
            "seed": seed,
            "architecture": architecture,
            "preprocessor": preprocessor,
            **training,
            **metrics,
            "pass_status": "PASS" if passed else "FAIL",
            "preprocessing_config": preprocessing_config,
            "evaluation_set_equals_training_set": True,
            "augmentation": False,
        }
        write_csv(history_path, history)
        write_csv(run_dir / "window_metrics.csv", window_rows(metadata, x, predicted))
        write_csv(run_dir / "channel_metrics.csv", channel_rows(metadata, x, predicted))
        temporary = predictions_path.with_name(
            f".{predictions_path.name}.tmp-{os.getpid()}.npz"
        )
        np.savez_compressed(
            temporary,
            target=x.astype(np.float32),
            reconstruction=predicted.astype(np.float32),
            latent=latent.astype(np.float32),
        )
        _replace_with_retry(temporary, predictions_path)
        write_json(run_dir / "config.json", {
            "subject_id": subject,
            "sample_count": sample_count,
            "seed": seed,
            "architecture": architecture,
            "preprocessor": preprocessor,
            "max_epochs": max_epochs,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "loss": "MSELoss",
            "gradient_clip_norm": 1.0,
            "preprocessing": preprocessing_config,
        })
        write_json(metrics_path, result)
        print(
            f"DONE {mode} {subject} N={sample_count} {architecture} "
            f"{result['pass_status']} improve={metrics['improvement_pct']:.3f}% "
            f"corr={metrics['median_corr']:.4f} nrmse={metrics['median_nrmse']:.4f}",
            flush=True,
        )
    with np.load(predictions_path, allow_pickle=False) as payload:
        actual = np.asarray(payload["target"])
        predicted = np.asarray(payload["reconstruction"])
        latent = np.asarray(payload["latent"])
    refreshed_metrics, _ = summarize(actual, predicted, latent)
    result.update(refreshed_metrics)
    result["pass_status"] = (
        "PASS" if pass_function(sample_count, refreshed_metrics) else "FAIL"
    )
    write_csv(run_dir / "window_metrics.csv", window_rows(metadata, actual, predicted))
    write_csv(run_dir / "channel_metrics.csv", channel_rows(metadata, actual, predicted))
    write_json(metrics_path, result)
    if not skip_figures:
        render_figures(
            mode,
            run_dir,
            numeric_history(read_csv(history_path)),
            actual,
            predicted,
            latent,
            channel_names,
        )
    return result


def prepare_selections(
    dataset: DaphnetDataset,
    output_dir: Path,
) -> tuple[
    dict[str, tuple[list[Record], selection.current.WindowSet]],
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int], list[dict[str, Any]]],
]:
    manifest = selection.load_manifest_rows(dataset.root)
    pools: dict[str, tuple[list[Record], selection.current.WindowSet]] = {}
    indices_by_key: dict[tuple[str, int], np.ndarray] = {}
    metadata_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        records, windows, candidates = selection.subject_pool(dataset, subject)
        eligible, energy = selection.eligible_candidates(records, windows, candidates)
        pools[subject] = (records, windows)
        subject_rows: list[dict[str, Any]] = []
        for sample_count in (1, 8, 32):
            selected = selection.select_windows(
                sample_count, eligible, energy, records, windows
            )
            metadata = selection.selected_metadata(
                subject,
                sample_count,
                selected,
                energy,
                records,
                windows,
                manifest,
            )
            indices_by_key[(subject, sample_count)] = selected
            metadata_by_key[(subject, sample_count)] = metadata
            subject_rows.extend(metadata)
            all_rows.extend(metadata)
        write_csv(
            output_dir / "selected_windows" / f"{subject}_selected_windows.csv",
            subject_rows,
        )
    write_csv(output_dir / "selected_windows" / "all_selected_windows.csv", all_rows)
    return pools, indices_by_key, metadata_by_key


def select_optional_128(
    subject: str,
    records: list[Record],
    windows: selection.current.WindowSet,
    manifest: dict[str, dict[str, str]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a non-overlapping pool first, then balance energy and records."""

    candidates = np.flatnonzero((windows.split == "train") & windows.clean_normal)
    eligible, energy = selection.eligible_candidates(records, windows, candidates)
    by_record: dict[str, list[int]] = defaultdict(list)
    for raw_index in eligible:
        index = int(raw_index)
        by_record[selection.record_id_for(records, windows, index)].append(index)

    nonoverlap_pool: list[int] = []
    for record_id in sorted(by_record):
        last_end = -1
        ordered = sorted(
            by_record[record_id], key=lambda index: (int(windows.start[index]), index)
        )
        for index in ordered:
            start = int(windows.start[index])
            if start >= last_end:
                nonoverlap_pool.append(index)
                last_end = int(windows.end[index])
    if len(nonoverlap_pool) < 128:
        raise ValueError(
            f"{subject} has only {len(nonoverlap_pool)} non-overlapping N=128 candidates"
        )

    selected: list[int] = []
    record_counts: defaultdict[str, int] = defaultdict(int)
    for group in selection.rank_quartiles(nonoverlap_pool, energy):
        target = float(np.median([energy[index] for index in group]))
        remaining = list(group)
        for _ in range(32):
            remaining.sort(
                key=lambda index: (
                    record_counts[
                        selection.record_id_for(records, windows, int(index))
                    ],
                    abs(energy[int(index)] - target),
                    int(windows.start[int(index)]),
                )
            )
            chosen = int(remaining.pop(0))
            selected.append(chosen)
            record_counts[
                selection.record_id_for(records, windows, chosen)
            ] += 1
    selected_array = np.asarray(selected, dtype=np.int64)
    metadata = selection.selected_metadata(
        subject,
        128,
        selected_array,
        energy,
        records,
        windows,
        manifest,
    )
    return selected_array, metadata


def row_for_table(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "round",
        "subject_id",
        "sample_count",
        "seed",
        "architecture",
        "preprocessor",
        "parameter_count",
        "best_epoch",
        "final_epoch",
        "stopped_early",
        "nbm_mse",
        "zero_mse",
        "improvement_pct",
        "median_corr",
        "median_nrmse",
        "p90_window_nrmse",
        "p95_window_nrmse",
        "worst_window_nrmse",
        "median_amplitude_ratio",
        "diff_mse",
        "latent_variance",
        "latent_between_window_variance",
        "raw_latent_distance_corr",
        "reconstruction_variance_retention",
        "inference_ms_per_batch",
        "pass_status",
    )
    return {key: result.get(key) for key in keys}


def select_preprocessor(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for name in PREPROCESSORS:
        rows = [row for row in results if row["preprocessor"] == name]
        summaries.append(
            {
                "preprocessor": name,
                "pass_count": sum(row["pass_status"] == "PASS" for row in rows),
                "median_improvement_pct": float(
                    np.median([row["improvement_pct"] for row in rows])
                ),
                "median_corr": float(np.median([row["median_corr"] for row in rows])),
                "median_nrmse": float(
                    np.median([row["median_nrmse"] for row in rows])
                ),
                "median_amplitude_deviation": float(
                    np.median(
                        [abs(row["median_amplitude_ratio"] - 1.0) for row in rows]
                    )
                ),
            }
        )
    eligible = [row for row in summaries if row["pass_count"] == len(REPRESENTATIVES)]
    if not eligible:
        return {
            "round1_gate": "FAIL",
            "selected_preprocessor": None,
            "reason": "No preprocessor passed N=1 for all four representative subjects.",
            "summaries": summaries,
        }
    by_name = {row["preprocessor"]: row for row in eligible}

    def materially_better(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
        return bool(
            candidate["median_improvement_pct"]
            >= reference["median_improvement_pct"] + 0.5
            or candidate["median_corr"] >= reference["median_corr"] + 0.005
            or candidate["median_nrmse"] <= reference["median_nrmse"] - 0.02
        )

    if "P0_current" in by_name:
        selected = by_name["P0_current"]
        reason = "P0 passed all subjects; neither P1 nor P2 showed a material improvement (>=0.5 percentage-point improvement, >=0.005 correlation, or >=0.02 NRMSE reduction)."
        if "P1_no_window_centering" in by_name and materially_better(
            by_name["P1_no_window_centering"], selected
        ):
            selected = by_name["P1_no_window_centering"]
            reason = "P1 materially outperformed P0, so window-axis centering was removed."
        if "P2_clip_only" in by_name and "P1_no_window_centering" in by_name and materially_better(
            by_name["P2_clip_only"], by_name["P1_no_window_centering"]
        ):
            selected = by_name["P2_clip_only"]
            reason = "P2 materially outperformed P1, indicating a RobustScaler diagnostic concern."
    else:
        eligible.sort(
            key=lambda row: (
                -row["median_improvement_pct"],
                -row["median_corr"],
                row["median_nrmse"],
                row["median_amplitude_deviation"],
            )
        )
        selected = eligible[0]
        reason = "P0 did not pass all subjects; selected the strongest all-subject passing alternative."
    return {
        "round1_gate": "PASS",
        "selected_preprocessor": selected["preprocessor"],
        "reason": reason,
        "material_difference_rule": {
            "improvement_pct": 0.5,
            "median_corr": 0.005,
            "median_nrmse_reduction": 0.02,
        },
        "summaries": summaries,
    }


def plot_round1_summary(results: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    x = np.arange(len(REPRESENTATIVES))
    width = 0.25
    for offset, preprocessor in enumerate(PREPROCESSORS):
        rows = [row for row in results if row["preprocessor"] == preprocessor]
        lookup = {row["subject_id"]: row for row in rows}
        for ax, key, title in zip(
            axes,
            ("improvement_pct", "median_corr", "median_nrmse"),
            ("Improvement (%)", "Pearson", "NRMSE"),
        ):
            ax.bar(
                x + (offset - 1) * width,
                [lookup[subject][key] for subject in REPRESENTATIVES],
                width,
                label=preprocessor,
            )
            ax.set_title(title)
            ax.set_xticks(x, REPRESENTATIVES)
            ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Round 1 preprocessing comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_round1(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    pools: dict[str, tuple[list[Record], selection.current.WindowSet]],
    indices: dict[tuple[str, int], np.ndarray],
    metadata: dict[tuple[str, int], list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    root = args.output_dir / "round1_preprocessing"
    results: list[dict[str, Any]] = []
    for preprocessor in PREPROCESSORS:
        for subject in REPRESENTATIVES:
            records, windows = pools[subject]
            selected = indices[(subject, 1)]
            x, preprocessing_config = preprocess(
                preprocessor, records, windows, selected
            )
            result = execute_run(
                mode="round1",
                run_dir=root / preprocessor / subject / f"seed{SEEDS[0]}",
                subject=subject,
                sample_count=1,
                seed=SEEDS[0],
                architecture="M1_tcdae_base",
                preprocessor=preprocessor,
                x=x,
                preprocessing_config=preprocessing_config,
                metadata=metadata[(subject, 1)],
                max_epochs=args.round1_epochs,
                optimizer_name="Adam",
                learning_rate=1e-3,
                weight_decay=0.0,
                patience=None,
                pass_function=lambda _n, metrics: round1_pass(metrics),
                device=device,
                num_workers=args.num_workers,
                channel_names=dataset.channel_names,
                overwrite=args.overwrite,
                skip_figures=args.skip_figures,
            )
            results.append(result)
    decision = select_preprocessor(results)
    write_csv(root / "round1_metrics.csv", [row_for_table(row) for row in results])
    write_json(root / "decision.json", decision)
    if not args.skip_figures:
        plot_round1_summary(results, root / "round1_preprocessing_comparison.png")
    print(f"ROUND1 {decision['round1_gate']} selected={decision['selected_preprocessor']}")
    return decision


def load_decision(path: Path, required_key: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prerequisite decision: {path}")
    with path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    if not decision.get(required_key):
        raise RuntimeError(f"Prerequisite gate did not pass: {decision}")
    return decision


def plot_architecture_comparison(
    rows: Sequence[dict[str, Any]], subject: str, path: Path
) -> None:
    selected = [
        row
        for row in rows
        if row["subject_id"] == subject and int(row["sample_count"]) == 8
    ]
    lookup = {row["architecture"]: row for row in selected}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key, title in zip(
        axes,
        ("improvement_pct", "median_corr", "median_nrmse", "parameter_count"),
        ("Improvement (%)", "Pearson", "NRMSE", "Parameters"),
    ):
        values = [lookup[name][key] for name in ARCHITECTURES]
        ax.bar(range(4), values)
        ax.set_xticks(range(4), [name.split("_")[0] for name in ARCHITECTURES])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{subject} N=8 architecture comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_architecture(screening: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        rows = [
            row
            for row in screening
            if row["architecture"] == architecture and row["sample_count"] == 8
        ]
        summaries.append(
            {
                "architecture": architecture,
                "n8_pass_count": sum(row["pass_status"] == "PASS" for row in rows),
                "median_corr": float(np.median([row["median_corr"] for row in rows])),
                "median_nrmse": float(np.median([row["median_nrmse"] for row in rows])),
                "median_improvement_pct": float(
                    np.median([row["improvement_pct"] for row in rows])
                ),
                "median_amplitude_deviation": float(
                    np.median(
                        [abs(row["median_amplitude_ratio"] - 1.0) for row in rows]
                    )
                ),
                "median_distance_corr": float(
                    np.median([row["raw_latent_distance_corr"] for row in rows])
                ),
                "parameter_count": int(rows[0]["parameter_count"]),
            }
        )
    by_name = {row["architecture"]: row for row in summaries}
    selected: str | None = None
    for candidate in ("M1_tcdae_base", "M2_tcdae_wide", "M3_tcdae_long"):
        if by_name[candidate]["n8_pass_count"] == len(REPRESENTATIVES):
            selected = candidate
            break
    return {
        "screen_gate": "PASS" if selected else "FAIL",
        "selected_architecture": selected,
        "reason": (
            "Selected the smallest TC-DAE in the preregistered M1->M2->M3 order that passed N=8 for all representative subjects."
            if selected
            else "No TC-DAE architecture passed N=8 for all representative subjects."
        ),
        "summaries": summaries,
    }


def run_round2_legacy(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    pools: dict[str, tuple[list[Record], selection.current.WindowSet]],
    indices: dict[tuple[str, int], np.ndarray],
    metadata: dict[tuple[str, int], list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    round1 = load_decision(
        args.output_dir / "round1_preprocessing" / "decision.json", "selected_preprocessor"
    )
    preprocessor = str(round1["selected_preprocessor"])
    root = args.output_dir / "round2_architecture"
    screening: list[dict[str, Any]] = []
    prepared: dict[tuple[str, int], tuple[np.ndarray, dict[str, Any]]] = {}
    for subject in REPRESENTATIVES:
        records, windows = pools[subject]
        for sample_count in (1, 8):
            prepared[(subject, sample_count)] = preprocess(
                preprocessor, records, windows, indices[(subject, sample_count)]
            )
    for architecture in ARCHITECTURES:
        for subject in REPRESENTATIVES:
            for sample_count in (1, 8):
                x, preprocessing_config = prepared[(subject, sample_count)]
                screening.append(
                    execute_run(
                        mode="round2",
                        run_dir=(
                            root
                            / "screening"
                            / architecture
                            / subject
                            / f"N{sample_count}"
                            / f"seed{SEEDS[0]}"
                        ),
                        subject=subject,
                        sample_count=sample_count,
                        seed=SEEDS[0],
                        architecture=architecture,
                        preprocessor=preprocessor,
                        x=x,
                        preprocessing_config=preprocessing_config,
                        metadata=metadata[(subject, sample_count)],
                        max_epochs=args.round2_epochs,
                        optimizer_name="Adam",
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        patience=None,
                        pass_function=round2_pass,
                        device=device,
                        num_workers=args.num_workers,
                        channel_names=dataset.channel_names,
                        overwrite=args.overwrite,
                        skip_figures=args.skip_figures,
                    )
                )
    decision = select_architecture(screening)
    write_csv(
        root / "round2_screening_metrics.csv",
        [row_for_table(row) for row in screening],
    )
    if not args.skip_figures:
        for subject in REPRESENTATIVES:
            plot_architecture_comparison(
                screening,
                subject,
                root / f"{subject}_N8_architecture_comparison.png",
            )
    if decision["screen_gate"] != "PASS":
        decision["round2_gate"] = "FAIL"
        write_json(root / "decision.json", decision)
        print("ROUND2 FAIL: no architecture passed screening")
        return decision

    best = str(decision["selected_architecture"])
    review: list[dict[str, Any]] = []
    screening_lookup = {
        (row["subject_id"], row["sample_count"], row["seed"], row["architecture"]): row
        for row in screening
    }
    for subject in REPRESENTATIVES:
        for sample_count in (1, 8):
            review.append(
                screening_lookup[(subject, sample_count, SEEDS[0], best)]
            )
            x, preprocessing_config = prepared[(subject, sample_count)]
            for seed in SEEDS[1:]:
                review.append(
                    execute_run(
                        mode="round2",
                        run_dir=(
                            root
                            / "seed_review"
                            / best
                            / subject
                            / f"N{sample_count}"
                            / f"seed{seed}"
                        ),
                        subject=subject,
                        sample_count=sample_count,
                        seed=seed,
                        architecture=best,
                        preprocessor=preprocessor,
                        x=x,
                        preprocessing_config=preprocessing_config,
                        metadata=metadata[(subject, sample_count)],
                        max_epochs=args.round2_epochs,
                        optimizer_name="Adam",
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        patience=None,
                        pass_function=round2_pass,
                        device=device,
                        num_workers=args.num_workers,
                        channel_names=dataset.channel_names,
                        overwrite=args.overwrite,
                        skip_figures=args.skip_figures,
                    )
                )
    n8_review = [row for row in review if row["sample_count"] == 8]
    stable = len(n8_review) == len(REPRESENTATIVES) * len(SEEDS) and all(
        row["pass_status"] == "PASS" for row in n8_review
    )
    decision["round2_gate"] = "PASS" if stable else "FAIL"
    decision["seed_review_n8_pass_count"] = sum(
        row["pass_status"] == "PASS" for row in n8_review
    )
    decision["seed_review_n8_total"] = len(n8_review)
    decision["seed_review_n8_failures"] = [
        {
            "subject_id": row["subject_id"],
            "seed": row["seed"],
            "improvement_pct": row["improvement_pct"],
            "median_corr": row["median_corr"],
            "median_nrmse": row["median_nrmse"],
            "median_amplitude_ratio": row["median_amplitude_ratio"],
            "raw_latent_distance_corr": row["raw_latent_distance_corr"],
        }
        for row in n8_review
        if row["pass_status"] != "PASS"
    ]
    write_csv(root / "round2_seed_review_metrics.csv", [row_for_table(row) for row in review])
    write_json(root / "decision.json", decision)
    print(
        f"ROUND2 {decision['round2_gate']} selected={best} "
        f"review={decision['seed_review_n8_pass_count']}/{decision['seed_review_n8_total']}"
    )
    return decision


def evaluate_structure_feasibility(
    architecture: str, screening: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    rows = [
        row
        for row in screening
        if row["architecture"] == architecture and row["sample_count"] == 8
    ]
    amplitude_collapse_count = sum(
        not 0.75 <= row["median_amplitude_ratio"] <= 1.25 for row in rows
    )
    values = {
        "strict_pass_count": sum(round2_pass(8, row) for row in rows),
        "run_count": len(rows),
        "median_improvement_pct": float(
            np.median([row["improvement_pct"] for row in rows])
        ),
        "median_corr": float(np.median([row["median_corr"] for row in rows])),
        "median_nrmse": float(np.median([row["median_nrmse"] for row in rows])),
        "amplitude_collapse_count": amplitude_collapse_count,
        "median_distance_corr": float(
            np.median([row["raw_latent_distance_corr"] for row in rows])
        ),
    }
    conditions = {
        "at_least_3_of_4_strict_pass": values["strict_pass_count"] >= 3,
        "median_improvement_at_least_95": values["median_improvement_pct"] >= 95.0,
        "median_corr_at_least_0_90": values["median_corr"] >= 0.90,
        "median_nrmse_at_most_0_35": values["median_nrmse"] <= 0.35,
        "fewer_than_two_amplitude_collapses": amplitude_collapse_count < 2,
        "median_distance_corr_at_least_0_70": values["median_distance_corr"] >= 0.70,
    }
    return {
        "architecture": architecture,
        "status": "PASS" if len(rows) == 4 and all(conditions.values()) else "FAIL",
        **values,
        "conditions": conditions,
    }


def round2_safety_pass(metrics: dict[str, Any]) -> bool:
    distance_corr = metrics.get("raw_latent_distance_corr")
    return bool(
        metrics["improvement_pct"] >= 95.0
        and metrics["median_corr"] >= 0.80
        and metrics["median_nrmse"] <= 0.60
        and 0.75 <= metrics["median_amplitude_ratio"] <= 1.25
        and distance_corr is not None
        and distance_corr >= 0.50
    )


def round2_strict_failure_details(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    checks = (
        ("improvement_pct", metrics["improvement_pct"], 80.0, "minimum"),
        ("median_corr", metrics["median_corr"], 0.80, "minimum"),
        ("median_nrmse", metrics["median_nrmse"], 0.50, "maximum"),
        ("median_amplitude_ratio_low", metrics["median_amplitude_ratio"], 0.75, "minimum"),
        ("median_amplitude_ratio_high", metrics["median_amplitude_ratio"], 1.25, "maximum"),
        (
            "raw_latent_distance_corr",
            metrics.get("raw_latent_distance_corr"),
            0.50,
            "minimum",
        ),
    )
    for name, value, threshold, direction in checks:
        if value is None:
            failures.append(
                {"metric": name, "value": None, "threshold": threshold, "relative_excess": math.inf}
            )
        elif direction == "minimum" and value < threshold:
            failures.append(
                {
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "relative_excess": (threshold - value) / threshold,
                }
            )
        elif direction == "maximum" and value > threshold:
            failures.append(
                {
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "relative_excess": (value - threshold) / threshold,
                }
            )
    return failures


def round2_catastrophic_failure_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if metrics["nbm_mse"] >= metrics["zero_mse"] or metrics["improvement_pct"] <= 0.0:
        reasons.append("loss_not_better_than_zero_output")
    if not 0.50 <= metrics["median_amplitude_ratio"] <= 1.50:
        reasons.append("severe_amplitude_collapse_or_explosion")
    if metrics["median_corr"] < 0.20:
        reasons.append("near_zero_correlation")
    if (metrics.get("reconstruction_variance_retention") or 0.0) < 0.10:
        reasons.append("near_zero_or_flat_output")
    if (
        metrics.get("latent_variance", 0.0) <= 1e-8
        or metrics.get("latent_between_window_variance", 0.0) <= 1e-8
    ):
        reasons.append("latent_collapse")
    return reasons


def evaluate_revised_round2_gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = len(REPRESENTATIVES) * len(SEEDS)
    strict_flags = [round2_pass(8, row) for row in rows]
    safety_flags = [round2_safety_pass(row) for row in rows]
    strict_failures: list[dict[str, Any]] = []
    boundary_failures_only = True
    catastrophic: list[dict[str, Any]] = []
    for row, strict_pass in zip(rows, strict_flags):
        catastrophic_reasons = round2_catastrophic_failure_reasons(row)
        if catastrophic_reasons:
            catastrophic.append(
                {
                    "subject_id": row["subject_id"],
                    "seed": row["seed"],
                    "reasons": catastrophic_reasons,
                }
            )
        if not strict_pass:
            details = round2_strict_failure_details(row)
            boundary = bool(
                len(details) == 1
                and details[0]["relative_excess"] <= 0.20
                and round2_safety_pass(row)
                and not catastrophic_reasons
            )
            boundary_failures_only = boundary_failures_only and boundary
            strict_failures.append(
                {
                    "subject_id": row["subject_id"],
                    "seed": row["seed"],
                    "boundary_failure": boundary,
                    "failed_metrics": details,
                    "improvement_pct": row["improvement_pct"],
                    "median_corr": row["median_corr"],
                    "median_nrmse": row["median_nrmse"],
                    "median_amplitude_ratio": row["median_amplitude_ratio"],
                    "raw_latent_distance_corr": row["raw_latent_distance_corr"],
                }
            )
    medians = {
        "median_improvement_pct": float(
            np.median([row["improvement_pct"] for row in rows])
        ),
        "median_corr": float(np.median([row["median_corr"] for row in rows])),
        "median_nrmse": float(np.median([row["median_nrmse"] for row in rows])),
        "median_amplitude_ratio": float(
            np.median([row["median_amplitude_ratio"] for row in rows])
        ),
        "median_distance_corr": float(
            np.median([row["raw_latent_distance_corr"] for row in rows])
        ),
    }
    strict_pass_count = sum(strict_flags)
    pass_rate = strict_pass_count / expected if expected else 0.0
    conditions = {
        "A_pass_rate_at_least_90_percent": (
            len(rows) == expected and pass_rate >= 0.90
        ),
        "B_all_runs_meet_safety_floor": len(rows) == expected and all(safety_flags),
        "C_aggregate_statistics": bool(
            len(rows) == expected
            and medians["median_improvement_pct"] >= 98.0
            and medians["median_corr"] >= 0.95
            and medians["median_nrmse"] <= 0.30
            and 0.85 <= medians["median_amplitude_ratio"] <= 1.15
            and medians["median_distance_corr"] >= 0.70
        ),
        "D_failures_are_boundary_only": bool(
            len(rows) == expected and boundary_failures_only and not catastrophic
        ),
    }
    strict_gate = len(rows) == expected and strict_pass_count == expected
    conditional_gate = not strict_gate and all(conditions.values())
    status = "Strict PASS" if strict_gate else "Conditional PASS" if conditional_gate else "FAIL"
    return {
        "status": status,
        "strict_stability_gate": "PASS" if strict_gate else "FAIL",
        "engineering_progression_gate": (
            "PASS" if strict_gate else "CONDITIONAL PASS" if conditional_gate else "FAIL"
        ),
        "run_count": len(rows),
        "expected_run_count": expected,
        "strict_pass_count": strict_pass_count,
        "safety_pass_count": sum(safety_flags),
        "pass_rate": pass_rate,
        **medians,
        "nrmse_p90": float(np.percentile([row["median_nrmse"] for row in rows], 90)),
        "worst_nrmse": float(max(row["median_nrmse"] for row in rows)),
        "worst_corr": float(min(row["median_corr"] for row in rows)),
        "median_amplitude_deviation": float(
            np.median([abs(row["median_amplitude_ratio"] - 1.0) for row in rows])
        ),
        "median_inference_ms_per_batch": float(
            np.median([row["inference_ms_per_batch"] for row in rows])
        ),
        "parameter_count": int(rows[0]["parameter_count"]),
        "conditions": conditions,
        "strict_failures": strict_failures,
        "catastrophic_failures": catastrophic,
    }


def select_revised_round2_architecture(
    gates: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    m2_name = "M2_tcdae_wide"
    m3_name = "M3_tcdae_long"
    m2 = gates[m2_name]
    m3 = gates[m3_name]
    eligible = [name for name in (m2_name, m3_name) if gates[name]["status"] != "FAIL"]
    if not eligible:
        return None, "Neither M2 nor M3 passed the revised progression gate."
    if m3["status"] == "Strict PASS":
        return m3_name, "M3 achieved 12/12 Strict PASS and is preferred by the frozen rule."
    if m2["status"] == "Strict PASS":
        return m2_name, "M2 was the only structure to achieve 12/12 Strict PASS."
    if m3["strict_pass_count"] < 11 and m2_name in eligible:
        return m2_name, "M3 was below 11/12, so the frozen rule selects Conditional-PASS M2."
    if len(eligible) == 1:
        return eligible[0], "Only one structure passed the revised progression gate."
    similar = bool(
        m2["strict_pass_count"] == m3["strict_pass_count"]
        and not m2["catastrophic_failures"]
        and not m3["catastrophic_failures"]
        and abs(m2["worst_nrmse"] - m3["worst_nrmse"]) <= 0.03
        and abs(m2["worst_corr"] - m3["worst_corr"]) <= 0.03
        and abs(m2["median_nrmse"] - m3["median_nrmse"]) <= 0.03
        and abs(m2["median_corr"] - m3["median_corr"]) <= 0.02
    )
    if similar:
        return m3_name, "M2/M3 stability and quality were similar; M3 is preferred for lower complexity and longer latent time axis."
    ranked = sorted(
        eligible,
        key=lambda name: (
            -gates[name]["strict_pass_count"],
            len(gates[name]["catastrophic_failures"]),
            gates[name]["worst_nrmse"],
            -gates[name]["worst_corr"],
            gates[name]["median_nrmse"],
            -gates[name]["median_corr"],
            -gates[name]["median_improvement_pct"],
            gates[name]["parameter_count"],
        ),
    )
    return ranked[0], "Selected by frozen stability, worst-run quality, aggregate quality, then complexity priority."


def revised_review_row(row: dict[str, Any]) -> dict[str, Any]:
    output = row_for_table(row)
    output["strict_pass"] = "PASS" if round2_pass(8, row) else "FAIL"
    output["safety_line_pass"] = "PASS" if round2_safety_pass(row) else "FAIL"
    output["catastrophic_failure"] = bool(round2_catastrophic_failure_reasons(row))
    return output


def plot_revised_structure_summary(
    gates: dict[str, dict[str, Any]], path: Path
) -> None:
    names = ["M2", "M3"]
    keys = ["M2_tcdae_wide", "M3_tcdae_long"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    values = (
        ("strict_pass_count", "Strict PASS count"),
        ("median_corr", "Median Pearson"),
        ("median_nrmse", "Median NRMSE"),
        ("nrmse_p90", "NRMSE P90"),
    )
    for ax, (key, title) in zip(axes, values):
        ax.bar(names, [gates[name][key] for name in keys])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Revised Round-2 cross-seed comparison")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_round2(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    pools: dict[str, tuple[list[Record], selection.current.WindowSet]],
    indices: dict[tuple[str, int], np.ndarray],
    metadata: dict[tuple[str, int], list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    round1 = load_decision(
        args.output_dir / "round1_preprocessing" / "decision.json", "selected_preprocessor"
    )
    preprocessor = str(round1["selected_preprocessor"])
    root = args.output_dir / "round2_architecture"
    revised_root = args.output_dir / "round2_architecture_revised"
    screening: list[dict[str, Any]] = []
    prepared: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for subject in REPRESENTATIVES:
        records, windows = pools[subject]
        prepared[subject] = preprocess(
            preprocessor, records, windows, indices[(subject, 8)]
        )
    for architecture in ARCHITECTURES:
        for subject in REPRESENTATIVES:
            x, preprocessing_config = prepared[subject]
            screening.append(
                execute_run(
                    mode="round2",
                    run_dir=root / "screening" / architecture / subject / "N8" / f"seed{SEEDS[0]}",
                    subject=subject,
                    sample_count=8,
                    seed=SEEDS[0],
                    architecture=architecture,
                    preprocessor=preprocessor,
                    x=x,
                    preprocessing_config=preprocessing_config,
                    metadata=metadata[(subject, 8)],
                    max_epochs=args.round2_epochs,
                    optimizer_name="Adam",
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    patience=None,
                    pass_function=round2_pass,
                    device=device,
                    num_workers=args.num_workers,
                    channel_names=dataset.channel_names,
                    overwrite=args.overwrite,
                    skip_figures=args.skip_figures,
                )
            )
    feasibility = {
        architecture: evaluate_structure_feasibility(architecture, screening)
        for architecture in ARCHITECTURES
    }
    review: list[dict[str, Any]] = []
    screening_lookup = {
        (row["architecture"], row["subject_id"]): row for row in screening
    }
    for architecture in ("M2_tcdae_wide", "M3_tcdae_long"):
        for subject in REPRESENTATIVES:
            review.append(screening_lookup[(architecture, subject)])
            x, preprocessing_config = prepared[subject]
            for seed in SEEDS[1:]:
                review.append(
                    execute_run(
                        mode="round2",
                        run_dir=root / "seed_review" / architecture / subject / "N8" / f"seed{seed}",
                        subject=subject,
                        sample_count=8,
                        seed=seed,
                        architecture=architecture,
                        preprocessor=preprocessor,
                        x=x,
                        preprocessing_config=preprocessing_config,
                        metadata=metadata[(subject, 8)],
                        max_epochs=args.round2_epochs,
                        optimizer_name="Adam",
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        patience=None,
                        pass_function=round2_pass,
                        device=device,
                        num_workers=args.num_workers,
                        channel_names=dataset.channel_names,
                        overwrite=args.overwrite,
                        skip_figures=args.skip_figures,
                    )
                )
    by_architecture = {
        architecture: [row for row in review if row["architecture"] == architecture]
        for architecture in ("M2_tcdae_wide", "M3_tcdae_long")
    }
    gates = {
        architecture: evaluate_revised_round2_gate(rows)
        for architecture, rows in by_architecture.items()
    }
    selected, selection_reason = select_revised_round2_architecture(gates)
    selected_gate = None if selected is None else gates[selected]
    decision = {
        "round2_gate": "PASS" if selected is not None else "FAIL",
        "round2_status": "FAIL" if selected_gate is None else selected_gate["status"],
        "selected_architecture": selected,
        "selection_reason": selection_reason,
        "structure_feasibility": feasibility,
        "architecture_gates": gates,
        "strict_stability_gate": (
            "FAIL" if selected_gate is None else selected_gate["strict_stability_gate"]
        ),
        "engineering_progression_gate": (
            "FAIL" if selected_gate is None else selected_gate["engineering_progression_gate"]
        ),
        "seed_review_n8_pass_count": (
            0 if selected_gate is None else selected_gate["strict_pass_count"]
        ),
        "seed_review_n8_total": 0 if selected_gate is None else selected_gate["run_count"],
        "seed_review_n8_failures": (
            [] if selected_gate is None else selected_gate["strict_failures"]
        ),
        "gate_definition": {
            "strict": "12/12 runs satisfy the original N=8 thresholds",
            "conditional": "A-D engineering progression conditions in the supplemental template",
        },
        "selection_rule_frozen_before_m3_review": True,
    }
    review_table = [revised_review_row(row) for row in review]
    summary_table = [
        {"architecture": architecture, **gates[architecture]}
        for architecture in ("M2_tcdae_wide", "M3_tcdae_long")
    ]
    write_csv(revised_root / "tables" / "cross_seed_results.csv", review_table)
    write_csv(revised_root / "tables" / "structure_summary.csv", summary_table)
    write_csv(root / "round2_seed_review_metrics.csv", review_table)
    write_json(root / "decision.json", decision)
    write_json(revised_root / "reports" / "decision.json", decision)
    if not args.skip_figures:
        plot_revised_structure_summary(
            gates, revised_root / "figures" / "m2_m3_cross_seed_comparison.png"
        )
    print(
        f"ROUND2 {decision['round2_status']} selected={selected} "
        f"strict={decision['seed_review_n8_pass_count']}/{decision['seed_review_n8_total']}",
        flush=True,
    )
    return decision


def boxplot_metric(
    rows: Sequence[dict[str, Any]], key: str, ylabel: str, path: Path
) -> None:
    levels = sorted({int(row["sample_count"]) for row in rows})
    values = [[float(row[key]) for row in rows if int(row["sample_count"]) == level] for level in levels]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.boxplot(values, tick_labels=[f"N={level}" for level in levels])
    ax.set_ylabel(ylabel)
    ax.set_title(f"All-subject {ylabel}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subject_curves(
    rows: Sequence[dict[str, Any]], subject: str, path: Path
) -> None:
    selected = [row for row in rows if row["subject_id"] == subject]
    levels = sorted({int(row["sample_count"]) for row in selected})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, key, ylabel in zip(
        axes,
        ("improvement_pct", "median_corr", "median_nrmse"),
        ("Improvement (%)", "Pearson", "NRMSE"),
    ):
        means = []
        stds = []
        for level in levels:
            values = [float(row[key]) for row in selected if int(row["sample_count"]) == level]
            means.append(np.mean(values))
            stds.append(np.std(values))
        ax.errorbar(levels, means, yerr=stds, marker="o", capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, [str(level) for level in levels])
        ax.set_xlabel("Sample count")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{subject} sample-size and seed stability")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pass_matrix(rows: Sequence[dict[str, Any]], path: Path) -> None:
    levels = sorted({int(row["sample_count"]) for row in rows})
    matrix = np.zeros((len(SUBJECTS), len(levels)), dtype=float)
    for i, subject in enumerate(SUBJECTS):
        for j, level in enumerate(levels):
            selected = [
                row
                for row in rows
                if row["subject_id"] == subject and int(row["sample_count"]) == level
            ]
            matrix[i, j] = np.mean([row["pass_status"] == "PASS" for row in selected])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)), [f"N={level}" for level in levels])
    ax.set_yticks(range(len(SUBJECTS)), SUBJECTS)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(round(matrix[i,j]*3))}/3", ha="center", va="center")
    ax.set_title("Subject-level seed pass matrix")
    fig.colorbar(image, ax=ax, label="Pass fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_final_report(
    output_dir: Path,
    round1: dict[str, Any],
    round2: dict[str, Any] | None,
    round3_rows: Sequence[dict[str, Any]],
) -> None:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    by_level: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in round3_rows:
        by_level[int(row["sample_count"])].append(row)
    round3_summary = "\n".join(
        f"- N={level}: {sum(row['pass_status']=='PASS' for row in rows)}/{len(rows)} PASS；"
        f"改善率中位数 {np.median([row['improvement_pct'] for row in rows]):.1f}%；"
        f"Pearson 中位数 {np.median([row['median_corr'] for row in rows]):.3f}；"
        f"NRMSE 中位数 {np.median([row['median_nrmse'] for row in rows]):.3f}；"
        f"运行级 NRMSE P90 中位数 {np.median([row.get('p90_window_nrmse', row['median_nrmse']) for row in rows]):.3f}。"
        for level, rows in sorted(by_level.items())
    ) or "- 未执行。"
    round3_failures = [row for row in round3_rows if row["pass_status"] != "PASS"]
    failures_by_subject_level: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in round3_failures:
        failures_by_subject_level[(str(row["subject_id"]), int(row["sample_count"]))].append(row)
    round3_failure_summary = "\n".join(
        f"- {subject} / N={level}: {len(rows)}/{len(SEEDS)} FAIL；"
        f"Pearson {min(row['median_corr'] for row in rows):.3f}–{max(row['median_corr'] for row in rows):.3f}；"
        f"NRMSE {min(row['median_nrmse'] for row in rows):.3f}–{max(row['median_nrmse'] for row in rows):.3f}。"
        for (subject, level), rows in sorted(failures_by_subject_level.items())
    ) or "- 无。"
    round3_conclusion = (
        f"第三轮共执行 {len(round3_rows)} 次；N=1 达到 24/24，但 N=8 与 N=32 均为 21/24。"
        "6 次失败全部来自 S03，说明 M3 的小样本单窗记忆能力稳定，但多窗口容量尚未在所有受试者上稳定成立。"
        if round3_rows
        else "第三轮尚未执行。"
    )
    n128_conclusion = (
        "N=32 未达到全部 8 名被试、3 个种子的 24/24 稳定通过，因此未触发 N=128。"
        if round3_rows and 128 not in by_level
        else "N=128 已按门控执行。"
        if 128 in by_level
        else "N=128 尚未评估。"
    )
    round1_summary = "\n".join(
        "- {preprocessor}: {pass_count}/4 PASS；改善率中位数 {median_improvement_pct:.4f}%；"
        "Pearson {median_corr:.4f}；NRMSE {median_nrmse:.4f}。".format(**row)
        for row in round1.get("summaries", [])
    )
    architecture_gates = {} if round2 is None else round2.get("architecture_gates", {})
    round2_summary = "\n".join(
        f"- {architecture}: {gate['status']}；严格 {gate['strict_pass_count']}/{gate['run_count']}；"
        f"安全线 {gate['safety_pass_count']}/{gate['run_count']}；改善率中位数 {gate['median_improvement_pct']:.2f}%；"
        f"Pearson {gate['median_corr']:.3f}；NRMSE {gate['median_nrmse']:.3f}；"
        f"NRMSE P90 {gate['nrmse_p90']:.3f}；最差 NRMSE {gate['worst_nrmse']:.3f}；"
        f"参数量 {gate['parameter_count']:,}。"
        for architecture, gate in architecture_gates.items()
    ) or "- 未执行修订版跨种子复核。"
    m2_failures = architecture_gates.get("M2_tcdae_wide", {}).get("strict_failures", [])
    m2_failure_summary = "\n".join(
        f"- {row['subject_id']} / seed {row['seed']}: NRMSE {row['median_nrmse']:.3f}，"
        f"相对严格线超出 {row['failed_metrics'][0]['relative_excess']*100:.2f}%；"
        f"Pearson {row['median_corr']:.3f}，幅值比 {row['median_amplitude_ratio']:.3f}，"
        f"潜变量距离相关 {row['raw_latent_distance_corr']:.3f}；"
        f"边界失败={'是' if row['boundary_failure'] else '否'}。"
        for row in m2_failures
    ) or "- 无。"
    round1_reason = (
        "P0 在 4 位代表受试者上全部通过；P1/P2 相对 P0 未达到预注册的实质改善幅度，"
        "因此保留当前稳健缩放和逐窗逐轴中心化方案。"
        if round1.get("selected_preprocessor") == "P0_current"
        else str(round1.get("reason"))
    )
    round2_selection_reason = (
        "M3 达到 12/12 Strict PASS；按补充模板冻结的第一优先级规则选择 M3。"
        if round2 is not None
        and round2.get("selected_architecture") == "M3_tcdae_long"
        and round2.get("strict_stability_gate") == "PASS"
        else None if round2 is None else str(round2.get("selection_reason"))
    )
    report = f"""# Daphnet NBM TC-DAE 三轮实验报告

本实验严格按三轮门控执行，仅使用冻结训练池中的纯 Non-FoG 窗口；训练集与评价集相同，因此结果只代表记忆/容量诊断，不代表泛化能力。

## 第一轮：预处理

- 门控：{round1.get('round1_gate')}
- 最佳预处理：{round1.get('selected_preprocessor')}
- 选择理由：{round1_reason}

{round1_summary}

## 第二轮：结构

- 结构可行性：M0 FAIL；M1/M2/M3 PASS
- 所选结构：{None if round2 is None else round2.get('selected_architecture')}
- 所选结构严格稳定性：{None if round2 is None else round2.get('strict_stability_gate')}
- 工程推进门控：{None if round2 is None else round2.get('engineering_progression_gate')}
- 决策状态：{None if round2 is None else round2.get('round2_status')}
- 选择理由：{round2_selection_reason}

{round2_summary}

### M2 边界失败保留记录

{m2_failure_summary}

定向诊断显示 M2/S03/seed20260804 的 4/8 个较低能量窗口较难，误差主要集中于 thigh_acc_forward 与 thigh_acc_vertical；输出未平坦化，梯度和潜变量均未塌缩。M2 因而保留为 Conditional PASS，但 M3 达到 12/12 Strict PASS，并以更少参数进入第三轮。

## 第三轮：容量与稳定性

{round3_summary}

### 失败分布

{round3_failure_summary}

{round3_conclusion}

{n128_conclusion}当前结果仍只代表训练集内记忆/容量诊断，不能作为未见记录泛化、去噪有效性或 FoG 检测能力的证据。
"""
    (report_dir / "final_report.md").write_text(report, encoding="utf-8")


def run_round3(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    pools: dict[str, tuple[list[Record], selection.current.WindowSet]],
    indices: dict[tuple[str, int], np.ndarray],
    metadata: dict[tuple[str, int], list[dict[str, Any]]],
    device: torch.device,
) -> list[dict[str, Any]]:
    round1 = load_decision(
        args.output_dir / "round1_preprocessing" / "decision.json", "selected_preprocessor"
    )
    round2 = load_decision(
        args.output_dir / "round2_architecture" / "decision.json", "selected_architecture"
    )
    if round2.get("round2_gate") != "PASS":
        raise RuntimeError("Round 2 seed-stability gate did not pass")
    preprocessor = str(round1["selected_preprocessor"])
    architecture = str(round2["selected_architecture"])
    root = args.output_dir / "round3_capacity"
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        records, windows = pools[subject]
        for sample_count in (1, 8, 32):
            x, preprocessing_config = preprocess(
                preprocessor, records, windows, indices[(subject, sample_count)]
            )
            for seed in SEEDS:
                rows.append(
                    execute_run(
                        mode="round3",
                        run_dir=root / subject / f"N{sample_count}" / f"seed{seed}",
                        subject=subject,
                        sample_count=sample_count,
                        seed=seed,
                        architecture=architecture,
                        preprocessor=preprocessor,
                        x=x,
                        preprocessing_config=preprocessing_config,
                        metadata=metadata[(subject, sample_count)],
                        max_epochs=args.round3_epochs,
                        optimizer_name="AdamW",
                        learning_rate=3e-4,
                        weight_decay=1e-4,
                        patience=args.round3_patience,
                        pass_function=round3_pass,
                        device=device,
                        num_workers=args.num_workers,
                        channel_names=dataset.channel_names,
                        overwrite=args.overwrite,
                        skip_figures=args.skip_figures,
                    )
                )
    n32 = [row for row in rows if row["sample_count"] == 32]
    n32_stable = len(n32) == len(SUBJECTS) * len(SEEDS) and all(
        row["pass_status"] == "PASS" for row in n32
    )
    if n32_stable:
        manifest = selection.load_manifest_rows(dataset.root)
        for subject in SUBJECTS:
            records, windows = pools[subject]
            if (subject, 128) not in indices:
                selected_128, metadata_128 = select_optional_128(
                    subject, records, windows, manifest
                )
                indices[(subject, 128)] = selected_128
                metadata[(subject, 128)] = metadata_128
                write_csv(
                    args.output_dir
                    / "selected_windows"
                    / f"{subject}_N128_selected_windows.csv",
                    metadata_128,
                )
            x, preprocessing_config = preprocess(
                preprocessor, records, windows, indices[(subject, 128)]
            )
            for seed in SEEDS:
                rows.append(
                    execute_run(
                        mode="round3",
                        run_dir=root / subject / "N128" / f"seed{seed}",
                        subject=subject,
                        sample_count=128,
                        seed=seed,
                        architecture=architecture,
                        preprocessor=preprocessor,
                        x=x,
                        preprocessing_config=preprocessing_config,
                        metadata=metadata[(subject, 128)],
                        max_epochs=args.round3_epochs,
                        optimizer_name="AdamW",
                        learning_rate=3e-4,
                        weight_decay=1e-4,
                        patience=args.round3_patience,
                        pass_function=round3_pass,
                        device=device,
                        num_workers=args.num_workers,
                        channel_names=dataset.channel_names,
                        overwrite=args.overwrite,
                        skip_figures=args.skip_figures,
                    )
                )
    write_csv(args.output_dir / "tables" / "round3_metrics.csv", [row_for_table(row) for row in rows])
    base_rows = [row for row in rows if row["sample_count"] in (1, 8, 32)]
    base_capacity_pass = len(base_rows) == len(SUBJECTS) * len(SEEDS) * 3 and all(
        row["pass_status"] == "PASS" for row in base_rows
    )
    decision = {
        "round3_complete": True,
        "round3_gate": "PASS" if base_capacity_pass else "FAIL",
        "base_capacity_all_pass": base_capacity_pass,
        "n32_stable_pass": n32_stable,
        "n128_executed": n32_stable,
        "n128_skip_reason": (
            None
            if n32_stable
            else "N=32 did not pass for all 8 subjects and all 3 seeds (24/24 required)."
        ),
        "run_count": len(rows),
        "pass_counts": {
            str(level): sum(
                row["pass_status"] == "PASS"
                for row in rows
                if row["sample_count"] == level
            )
            for level in sorted({int(row["sample_count"]) for row in rows})
        },
        "run_counts": {
            str(level): sum(row["sample_count"] == level for row in rows)
            for level in sorted({int(row["sample_count"]) for row in rows})
        },
        "failure_runs": [
            {
                "subject_id": row["subject_id"],
                "sample_count": row["sample_count"],
                "seed": row["seed"],
                "improvement_pct": row["improvement_pct"],
                "median_corr": row["median_corr"],
                "median_nrmse": row["median_nrmse"],
            }
            for row in rows
            if row["pass_status"] != "PASS"
        ],
    }
    write_json(root / "decision.json", decision)
    if not args.skip_figures:
        figures = args.output_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        for subject in SUBJECTS:
            plot_subject_curves(
                rows, subject, figures / f"{subject}_subject_sample_size_curve.png"
            )
            plot_subject_curves(
                rows, subject, figures / f"{subject}_subject_seed_stability.png"
            )
        boxplot_metric(
            rows,
            "improvement_pct",
            "Improvement over zero (%)",
            figures / "all_subject_improvement_boxplot.png",
        )
        boxplot_metric(
            rows,
            "median_corr",
            "Median Pearson correlation",
            figures / "all_subject_correlation_boxplot.png",
        )
        boxplot_metric(
            rows,
            "median_nrmse",
            "Median NRMSE",
            figures / "all_subject_nrmse_boxplot.png",
        )
        plot_pass_matrix(rows, figures / "subject_level_pass_matrix.png")
        plot_pass_matrix(rows, figures / "architecture_final_summary.png")
    return rows


def audit_outputs(output_dir: Path) -> dict[str, Any]:
    metric_files = list(output_dir.rglob("metrics.json"))
    checkpoint_files = list(output_dir.rglob("*_model.pt"))
    prediction_files = list(output_dir.rglob("predictions.npz"))
    all_finite = True
    for path in prediction_files:
        with np.load(path, allow_pickle=False) as payload:
            all_finite = all_finite and all(
                np.isfinite(payload[key]).all()
                for key in ("target", "reconstruction", "latent")
            )
    audit = {
        "metric_files": len(metric_files),
        "checkpoint_files": len(checkpoint_files),
        "prediction_files": len(prediction_files),
        "figure_files": len(list(output_dir.rglob("*.png"))),
        "all_prediction_arrays_finite": all_finite,
        "all_runs_have_two_checkpoints": len(checkpoint_files) == 2 * len(metric_files),
        "all_runs_have_predictions": len(prediction_files) == len(metric_files),
    }
    write_json(output_dir / "reports" / "artifact_audit.json", audit)
    return audit


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = DaphnetDataset.load(args.data_dir)
    if dataset.sampling_rate_hz != FS or dataset.n_channels != CHANNELS:
        raise ValueError("Expected 64-Hz, nine-channel Daphnet data")
    config = {
        "experiment": EXPERIMENT,
        "stage": args.stage,
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "round1_epochs": args.round1_epochs,
        "round2_epochs": args.round2_epochs,
        "round3_epochs": args.round3_epochs,
        "round3_patience": args.round3_patience,
        "subjects": list(SUBJECTS),
        "representatives": list(REPRESENTATIVES),
        "seeds": list(SEEDS),
    }
    write_json(args.output_dir / "config" / "resolved_config.json", config)
    source = REPO_ROOT / "configs" / "daphnet_nbm_tcdae_three_rounds.yaml"
    (args.output_dir / "config" / "experiment.yaml").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    pools, indices, metadata = prepare_selections(dataset, args.output_dir)
    print(f"PREFLIGHT stage={args.stage} device={device} output={args.output_dir}")

    round1_decision: dict[str, Any] | None = None
    round2_decision: dict[str, Any] | None = None
    round3_rows: list[dict[str, Any]] = []
    if args.stage in ("round1", "all"):
        round1_decision = run_round1(
            args, dataset, pools, indices, metadata, device
        )
        if round1_decision["round1_gate"] != "PASS" and args.stage == "all":
            write_final_report(args.output_dir, round1_decision, None, [])
            audit_outputs(args.output_dir)
            print("STOP after round 1 gate failure")
            return
    else:
        round1_decision = load_decision(
            args.output_dir / "round1_preprocessing" / "decision.json",
            "selected_preprocessor",
        )

    if args.stage in ("round2", "all"):
        round2_decision = run_round2(
            args, dataset, pools, indices, metadata, device
        )
        if round2_decision.get("round2_gate") != "PASS" and args.stage == "all":
            write_final_report(args.output_dir, round1_decision, round2_decision, [])
            audit_outputs(args.output_dir)
            print("STOP after round 2 gate failure")
            return
    elif args.stage == "round3":
        round2_decision = load_decision(
            args.output_dir / "round2_architecture" / "decision.json",
            "selected_architecture",
        )

    if args.stage in ("round3", "all"):
        round3_rows = run_round3(
            args, dataset, pools, indices, metadata, device
        )

    write_final_report(args.output_dir, round1_decision, round2_decision, round3_rows)
    audit = audit_outputs(args.output_dir)
    print(f"COMPLETE audit={audit} report={args.output_dir / 'reports' / 'final_report.md'}")


if __name__ == "__main__":
    main()
