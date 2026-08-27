"""Run a clean 16/32-window overfit capacity test for the current GRU-NBM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
from cnbr_fog.resume import atomic_json_dump, sha256_file
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    RobustScaler,
)


DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
DEFAULT_SOURCE_EXPERIMENT = (
    REPO_ROOT
    / "outputs"
    / "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_"
    "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "private_gru_nbm_small_sample_overfit_P02_fold0"
)
SUBSET_SIZES = (16, 32)
DEFAULT_SEEDS = (0, 52, 161)
FIXED_CHANNELS = (0, 9, 15)
FIXED_CHANNEL_LABELS = (
    "Lumbar Acc-x",
    "Left ankle Gyro-x",
    "Right ankle Gyro-x",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subject", default="P02")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--selection-seed", type=int, default=20260827)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"invalid seed list: {text!r}")
    return seeds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_scaler(path: Path) -> RobustScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))["scaler"]
    return RobustScaler(
        np.asarray(payload["median"], dtype=np.float32),
        np.asarray(payload["iqr"], dtype=np.float32),
        float(payload["epsilon"]),
    )


def smooth_l1_numpy(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.abs(target.astype(np.float64) - prediction.astype(np.float64))
    loss = np.where(error < 1.0, 0.5 * error**2, error - 0.5)
    return float(loss.mean())


def temporal_correlation(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    left = target.astype(np.float64) - target.mean(axis=1, keepdims=True)
    right = prediction.astype(np.float64) - prediction.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-12,
    )


def angular_error_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (left - right)))) * 180.0 / np.pi


def reconstruction_metrics(
    target: np.ndarray,
    reconstruction: np.ndarray,
    sampling_rate_hz: int,
) -> dict[str, float]:
    error = reconstruction.astype(np.float64) - target.astype(np.float64)
    correlation = temporal_correlation(target, reconstruction)
    target_std = target.astype(np.float64).std(axis=1)
    reconstruction_std = reconstruction.astype(np.float64).std(axis=1)
    variance_ratio = reconstruction_std / np.maximum(target_std, 1e-12)
    target_ptp = np.ptp(target.astype(np.float64), axis=1)
    reconstruction_ptp = np.ptp(reconstruction.astype(np.float64), axis=1)
    amplitude_ratio = reconstruction_ptp / np.maximum(target_ptp, 1e-12)

    channels = np.asarray(FIXED_CHANNELS, dtype=int)
    target_fixed = target[:, :, channels]
    reconstruction_fixed = reconstruction[:, :, channels]
    peak_error = np.abs(
        np.argmax(target_fixed, axis=1) - np.argmax(reconstruction_fixed, axis=1)
    ) / sampling_rate_hz
    trough_error = np.abs(
        np.argmin(target_fixed, axis=1) - np.argmin(reconstruction_fixed, axis=1)
    ) / sampling_rate_hz

    target_spectrum = np.fft.rfft(
        target_fixed - target_fixed.mean(axis=1, keepdims=True), axis=1
    )
    reconstruction_spectrum = np.fft.rfft(
        reconstruction_fixed - reconstruction_fixed.mean(axis=1, keepdims=True), axis=1
    )
    frequencies = np.fft.rfftfreq(target.shape[1], d=1.0 / sampling_rate_hz)
    target_bins = np.argmax(np.abs(target_spectrum[:, 1:, :]) ** 2, axis=1) + 1
    reconstruction_bins = (
        np.argmax(np.abs(reconstruction_spectrum[:, 1:, :]) ** 2, axis=1) + 1
    )
    target_frequency = frequencies[target_bins]
    reconstruction_frequency = frequencies[reconstruction_bins]
    dominant_frequency_error = np.abs(target_frequency - reconstruction_frequency)

    left_index = FIXED_CHANNELS.index(9)
    right_index = FIXED_CHANNELS.index(15)
    joint_power = (
        np.abs(target_spectrum[:, 1:, left_index]) ** 2
        + np.abs(target_spectrum[:, 1:, right_index]) ** 2
    )
    phase_bins = np.argmax(joint_power, axis=1) + 1
    window_index = np.arange(len(target))
    target_phase = np.angle(
        target_spectrum[window_index, phase_bins, left_index]
        * np.conj(target_spectrum[window_index, phase_bins, right_index])
    )
    reconstruction_phase = np.angle(
        reconstruction_spectrum[window_index, phase_bins, left_index]
        * np.conj(reconstruction_spectrum[window_index, phase_bins, right_index])
    )
    phase_error = angular_error_degrees(reconstruction_phase, target_phase)

    return {
        "smooth_l1": smooth_l1_numpy(target, reconstruction),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "temporal_correlation_mean": float(np.nanmean(correlation)),
        "temporal_correlation_median": float(np.nanmedian(correlation)),
        "variance_ratio_median": float(np.nanmedian(variance_ratio)),
        "amplitude_ratio_fixed_channels_median": float(np.nanmedian(amplitude_ratio[:, channels])),
        "peak_timing_error_fixed_channels_median_sec": float(np.median(peak_error)),
        "trough_timing_error_fixed_channels_median_sec": float(np.median(trough_error)),
        "dominant_frequency_error_fixed_channels_median_hz": float(
            np.median(dominant_frequency_error)
        ),
        "dominant_frequency_exact_bin_fraction": float(
            np.mean(target_bins == reconstruction_bins)
        ),
        "left_right_phase_error_median_deg": float(np.median(phase_error)),
    }


def train_one(
    train_x: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    worker.set_seed(seed)
    model = GRUReconstructionNBM(
        channels=worker.RAW_CHANNELS,
        hidden=worker.HIDDEN,
        bottleneck=worker.BOTTLENECK,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != worker.NBM_PARAMETER_COUNT:
        raise AssertionError(f"GRU-NBM parameter mismatch: {parameter_count}")
    if any(isinstance(module, nn.Dropout) for module in model.modules()):
        raise AssertionError("current GRU-NBM unexpectedly contains dropout")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    clean = torch.from_numpy(np.ascontiguousarray(train_x)).float().to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(clean)
        loss = criterion(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "train_smooth_l1": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(
                f"size={len(train_x):02d} seed={seed} epoch={epoch:04d}/{epochs} "
                f"loss={float(loss.detach()):.8f}",
                flush=True,
            )
    return model, history


@torch.no_grad()
def reconstruct(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    return model(tensor).cpu().numpy().astype(np.float32)


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"subset_size", "seed", "epochs", "parameter_count"}
    metric_names = [key for key in rows[0] if key not in excluded]
    output: list[dict[str, Any]] = []
    for subset_size in SUBSET_SIZES:
        selected = [row for row in rows if row["subset_size"] == subset_size]
        summary: dict[str, Any] = {
            "subset_size": subset_size,
            "seed_count": len(selected),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=0))
        output.append(summary)
    return output


def choose_median_window(target: np.ndarray, reconstruction: np.ndarray) -> int:
    rmse = np.sqrt(np.mean((target - reconstruction) ** 2, axis=(1, 2)))
    order = np.argsort(rmse, kind="stable")
    return int(order[len(order) // 2])


def plot_results(
    output_dir: Path,
    histories: dict[tuple[int, int], list[dict[str, Any]]],
    target: np.ndarray,
    reconstruction: np.ndarray,
    representative_index: int,
    seeds: tuple[int, ...],
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.2, 10.5),
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.0, 1.0, 1.0], "hspace": 0.42},
    )
    for axis, subset_size in zip(axes[:2], SUBSET_SIZES):
        for seed in seeds:
            history = histories[(subset_size, seed)]
            axis.plot(
                [row["epoch"] for row in history],
                [row["train_smooth_l1"] for row in history],
                linewidth=1.0,
                label=f"seed {seed}",
            )
        axis.set_yscale("log")
        axis.set_ylabel("SmoothL1")
        axis.set_title(f"{subset_size}-window clean overfit training loss", loc="left")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=len(seeds), fontsize=7)
    axes[1].set_xlabel("Epoch")

    time = np.arange(target.shape[1]) / worker.SAMPLING_RATE_HZ
    for axis, channel, label in zip(axes[2:], FIXED_CHANNELS, FIXED_CHANNEL_LABELS):
        axis.plot(
            time,
            target[representative_index, :, channel],
            color="#4A4A4A",
            linewidth=1.0,
            label="Target",
        )
        axis.plot(
            time,
            reconstruction[representative_index, :, channel],
            color="#2878B5",
            linewidth=1.1,
            label="Reconstruction",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    axes[2].set_title(
        f"32-window, seed-{seeds[0]} median-RMSE memorized training window",
        loc="left",
    )
    axes[2].legend(frameon=False, ncol=2, fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "GRU-NBM small-sample overfit capacity test\n"
        "No noise, no mask, no dropout, no early stopping",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.07)
    stem = output_dir / "gru_nbm_small_sample_overfit"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    source_experiment = args.source_experiment.resolve()
    output_dir = args.output_dir.resolve()
    seeds = parse_seeds(args.seeds)
    if args.epochs <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimization settings")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")

    dataset = DaphnetDataset.load(data_dir)
    rows = raw_base.load_subject_rows(data_dir, dataset, args.subject, args.fold)
    role4 = rows.take_role(4)
    if len(role4) < max(SUBSET_SIZES) or not np.all(role4.label == 0):
        raise AssertionError("role 4 does not contain enough clean Non-FoG windows")
    source_run = worker.run_dir(source_experiment, args.subject, args.fold, 0)
    scaler_path = source_run / "scaler_role4.json"
    scaler = load_scaler(scaler_path)
    raw = raw_base.raw_windows(dataset, role4)
    scaled = scaler.transform(raw)
    centered = np.ascontiguousarray(
        scaled - scaled.mean(axis=1, keepdims=True), dtype=np.float32
    )

    rng = np.random.default_rng(args.selection_seed)
    selected_positions = rng.permutation(len(role4))[: max(SUBSET_SIZES)]
    selected_window_ids = [str(role4.window_id[index]) for index in selected_positions]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "selected_clean_nonfog_windows.csv",
        [
            {
                "nested_order": order + 1,
                "included_in_16": order < 16,
                "included_in_32": True,
                "window_id": selected_window_ids[order],
                "record_id": str(role4.record_id[position]),
                "start_index": int(role4.start[position]),
                "end_index_exclusive": int(role4.end[position]),
            }
            for order, position in enumerate(selected_positions)
        ],
    )

    device = worker.resolve_device(args.device)
    all_metrics: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    histories: dict[tuple[int, int], list[dict[str, Any]]] = {}
    saved_reconstructions: dict[tuple[int, int], np.ndarray] = {}
    baselines: dict[int, dict[str, float]] = {}
    for subset_size in SUBSET_SIZES:
        train_x = centered[selected_positions[:subset_size]]
        zero = np.zeros_like(train_x)
        mean_template = np.broadcast_to(train_x.mean(axis=0, keepdims=True), train_x.shape)
        baselines[subset_size] = {
            "zero_output_smooth_l1": smooth_l1_numpy(train_x, zero),
            "mean_template_smooth_l1": smooth_l1_numpy(train_x, mean_template),
        }
        for seed in seeds:
            model, history = train_one(
                train_x,
                seed,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                device,
            )
            reconstruction = reconstruct(model, train_x, device)
            metrics = reconstruction_metrics(
                train_x, reconstruction, int(dataset.sampling_rate_hz)
            )
            metrics["loss_ratio_to_zero_baseline"] = (
                metrics["smooth_l1"] / baselines[subset_size]["zero_output_smooth_l1"]
            )
            metrics["loss_ratio_to_mean_template"] = (
                metrics["smooth_l1"]
                / baselines[subset_size]["mean_template_smooth_l1"]
            )
            all_metrics.append(
                {
                    "subset_size": subset_size,
                    "seed": seed,
                    "epochs": args.epochs,
                    "parameter_count": worker.NBM_PARAMETER_COUNT,
                    **metrics,
                }
            )
            histories[(subset_size, seed)] = history
            all_history_rows.extend(
                {
                    "subset_size": subset_size,
                    "seed": seed,
                    **row,
                }
                for row in history
            )
            saved_reconstructions[(subset_size, seed)] = reconstruction
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "subset_size": subset_size,
                    "seed": seed,
                    "epochs": args.epochs,
                    "training_contract": "clean full-batch overfit; no augmentation or stopping",
                },
                output_dir / f"gru_nbm_size{subset_size}_seed{seed}_final.pt",
            )

    aggregate = aggregate_metrics(all_metrics)
    write_csv(output_dir / "per_run_metrics.csv", all_metrics)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(output_dir / "training_history.csv", all_history_rows)

    reference_seed = seeds[0]
    target32 = centered[selected_positions[:32]]
    reconstruction32 = saved_reconstructions[(32, reference_seed)]
    representative_index = choose_median_window(target32, reconstruction32)
    plot_results(
        output_dir,
        histories,
        target32,
        reconstruction32,
        representative_index,
        seeds,
        args.dpi,
    )
    np.savez_compressed(
        output_dir / "representative_training_window.npz",
        target=target32[representative_index],
        reconstruction=reconstruction32[representative_index],
        window_id=np.asarray(selected_window_ids[representative_index]),
        channel_names=np.asarray(dataset.channel_names),
    )

    metadata = {
        "schema": "private_gru_nbm_small_sample_overfit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "network capacity diagnostic; not a generalization experiment",
        "data_dir": str(data_dir),
        "subject": args.subject,
        "fold": args.fold,
        "source_role": 4,
        "source_role4_windows": len(role4),
        "nested_subset_sizes": list(SUBSET_SIZES),
        "selection_seed": args.selection_seed,
        "seeds": list(seeds),
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "input_shape": ["B", worker.WINDOW_SAMPLES, worker.RAW_CHANNELS],
        "architecture": worker.architecture_config(),
        "parameter_count": worker.NBM_PARAMETER_COUNT,
        "preprocessing": (
            "frozen official role-4 RobustScaler followed by per-window/per-axis "
            "temporal centering"
        ),
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "training": {
            "input_equals_target": True,
            "gaussian_noise": False,
            "time_mask": False,
            "dropout_modules": 0,
            "early_stopping": False,
            "validation_set": None,
            "scheduler": None,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batching": "one full batch per epoch",
            "epochs": args.epochs,
            "gradient_clipping_norm": 1.0,
            "loss": "SmoothL1(beta=1.0)",
        },
        "baselines": baselines,
        "representative_window_rule": (
            "median final RMSE among the 32 memorized windows for the first seed"
        ),
        "representative_window_id": selected_window_ids[representative_index],
        "metrics": {
            "per_run": all_metrics,
            "aggregate": aggregate,
        },
        "output_sha256": {},
    }
    for path in output_dir.iterdir():
        if path.is_file() and path.name != "audit.json":
            metadata["output_sha256"][path.name] = sha256_file(path)
    atomic_json_dump(metadata, output_dir / "audit.json")
    print(f"COMPLETE output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
