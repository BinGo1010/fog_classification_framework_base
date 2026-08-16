#!/usr/bin/env python3
"""Visualize a representative pure-FoG window before/after the frozen GRU-NBM.

The representative window is selected deterministically as the permanent-test
FoG window whose whole-window reconstruction MAE is closest to the median over
all permanent-test FoG windows.  This avoids selecting the most visually
striking reconstruction.  Curves are shown in the exact Robust-scaled and
per-window/per-axis centered space seen by the NBM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import raw_windows_dynamic
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    RobustScaler,
    load_fold_rows,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    prepare_nbm_windows,
    reconstruct,
)


CHANNELS = (
    ("Ankle", "Forward"),
    ("Ankle", "Vertical"),
    ("Ankle", "Lateral"),
    ("Thigh", "Forward"),
    ("Thigh", "Vertical"),
    ("Thigh", "Lateral"),
    ("Trunk", "Forward"),
    ("Trunk", "Vertical"),
    ("Trunk", "Lateral"),
)


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
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_gru_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "gru_nbm_fog_reconstruction",
    )
    parser.add_argument("--seed", type=int, default=0, choices=(0, 52, 161))
    parser.add_argument("--fold", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "n": int(values.size),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    flat_x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    flat_y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    flat_x -= flat_x.mean(axis=1, keepdims=True)
    flat_y -= flat_y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(flat_x * flat_x, axis=1) * np.sum(flat_y * flat_y, axis=1)
    )
    return np.divide(
        np.sum(flat_x * flat_y, axis=1),
        denominator,
        out=np.full(len(x), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def load_frozen_model(
    experiment_dir: Path,
    seed: int,
    fold: int,
    device: torch.device,
) -> tuple[GRUReconstructionNBM, RobustScaler, dict[str, Any]]:
    source = experiment_dir / "nbm_source" / f"seed_{seed}" / f"fold_{fold}"
    frozen = json.loads((source / "nbm_frozen.json").read_text(encoding="utf-8"))
    architecture = frozen["training"]["architecture"]
    if architecture["name"] != "gru_reconstruction_nbm_v1":
        raise AssertionError(f"unexpected architecture: {architecture['name']}")
    if architecture["parameter_count"] != 31_513:
        raise AssertionError("unexpected GRU-NBM parameter count")
    scaler_config = frozen["scaler"]
    scaler = RobustScaler(
        median=np.asarray(scaler_config["median"], dtype=np.float32),
        iqr=np.asarray(scaler_config["iqr"], dtype=np.float32),
        epsilon=float(scaler_config["epsilon"]),
    )
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    checkpoint = source / "checkpoints" / "gru_nbm_best.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, scaler, {
        "source": str(source.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "best_epoch": int(frozen["training"]["best_epoch"]),
        "best_role5_smoothl1": float(frozen["training"]["best_validation_huber"]),
        "parameter_count": int(architecture["parameter_count"]),
    }


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = DaphnetDataset.load(args.data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(args.data_dir, args.fold).take_role(0, 1)
    if set(np.unique(rows.role).tolist()) != {0, 1}:
        raise AssertionError("expected permanent-test roles 0 and 1 only")
    raw = raw_windows_dynamic(records, rows, window_samples=128)
    model, scaler, model_meta = load_frozen_model(
        args.experiment_dir, args.seed, args.fold, device
    )
    model_input = prepare_nbm_windows(scaler, raw, center=True)
    reconstructed = reconstruct(model, model_input, device)
    if model_input.shape != reconstructed.shape or model_input.shape[1:] != (128, 9):
        raise AssertionError(
            f"unexpected input/output shapes: {model_input.shape}, {reconstructed.shape}"
        )

    residual = model_input - reconstructed
    mae = np.mean(np.abs(residual), axis=(1, 2))
    rmse = np.sqrt(np.mean(residual * residual, axis=(1, 2)))
    correlation = rowwise_correlation(model_input, reconstructed)
    fog_indices = np.flatnonzero(rows.role == 1)
    nonfog_indices = np.flatnonzero(rows.role == 0)
    fog_median_mae = float(np.median(mae[fog_indices]))
    selected = int(fog_indices[np.argmin(np.abs(mae[fog_indices] - fog_median_mae))])

    # The manifest guarantees role 1 is a pure 128/128-sample FoG window.
    record = records[str(rows.record_id[selected])]
    start = int(rows.start[selected])
    end = int(rows.end[selected])
    if not np.all(np.asarray(record.y[start:end]) == 1):
        raise AssertionError("selected role-1 window is not 128/128 pure FoG")

    metrics = {
        "selection_rule": (
            "permanent-test pure-FoG window with reconstruction MAE closest "
            "to the median across all permanent-test pure-FoG windows"
        ),
        "seed": args.seed,
        "fold": args.fold,
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "test_counts": {"nonfog": int(len(nonfog_indices)), "fog": int(len(fog_indices))},
        "selected_window": {
            "subject_id": str(rows.subject_id[selected]),
            "record_id": str(rows.record_id[selected]),
            "window_id": str(rows.window_id[selected]),
            "start_index": start,
            "end_index_exclusive": end,
            "fog_samples": 128,
            "mae": float(mae[selected]),
            "rmse": float(rmse[selected]),
            "pearson_r": float(correlation[selected]),
        },
        "permanent_test_reconstruction": {
            "nonfog_mae": percentile_summary(mae[nonfog_indices]),
            "fog_mae": percentile_summary(mae[fog_indices]),
            "nonfog_rmse": percentile_summary(rmse[nonfog_indices]),
            "fog_rmse": percentile_summary(rmse[fog_indices]),
            "nonfog_pearson_r": percentile_summary(correlation[nonfog_indices]),
            "fog_pearson_r": percentile_summary(correlation[fog_indices]),
            "fog_to_nonfog_median_mae_ratio": float(
                np.median(mae[fog_indices]) / np.median(mae[nonfog_indices])
            ),
        },
        "model": model_meta,
        "curve_space": "role-4 Robust-scaled then per-window/per-axis centered",
        "interpretation_guardrail": (
            "The reconstruction is a learned normal-manifold projection, not a paired "
            "ground-truth Non-FoG signal for this FoG window."
        ),
    }
    (args.output_dir / "gru_nbm_fog_reconstruction_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = args.output_dir / "gru_nbm_fog_reconstruction_source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "time_sec",
                "channel_index",
                "sensor",
                "axis",
                "fog_input_scaled_centered",
                "gru_nbm_reconstruction_scaled_centered",
                "residual",
            ),
        )
        writer.writeheader()
        for channel, (sensor, axis) in enumerate(CHANNELS):
            for sample in range(128):
                writer.writerow(
                    {
                        "time_sec": sample / 64.0,
                        "channel_index": channel,
                        "sensor": sensor,
                        "axis": axis,
                        "fog_input_scaled_centered": float(model_input[selected, sample, channel]),
                        "gru_nbm_reconstruction_scaled_centered": float(
                            reconstructed[selected, sample, channel]
                        ),
                        "residual": float(residual[selected, sample, channel]),
                    }
                )

    configure_plotting()
    time = np.arange(128, dtype=np.float64) / 64.0
    input_color = "#315A8C"
    reconstruction_color = "#D0705B"
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.15, 5.55),
        sharex=True,
        constrained_layout=True,
    )
    for channel, ax in enumerate(axes.flat):
        sensor, axis = CHANNELS[channel]
        ax.plot(
            time,
            model_input[selected, :, channel],
            color=input_color,
            linewidth=1.05,
            label="Observed pure-FoG input",
            zorder=2,
        )
        ax.plot(
            time,
            reconstructed[selected, :, channel],
            color=reconstruction_color,
            linewidth=1.20,
            label="GRU-NBM reconstruction",
            zorder=3,
        )
        ax.axhline(0.0, color="#B8B8B8", linewidth=0.55, zorder=1)
        ax.set_title(f"{sensor} — {axis}", fontsize=8.2, pad=3)
        ax.set_xlim(0.0, 2.0)
        ax.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Scaled, centered\nacceleration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        handlelength=2.6,
        columnspacing=2.0,
    )
    fig.suptitle(
        "Pure-FoG input and its GRU-NBM normal-manifold reconstruction",
        fontsize=10,
        y=1.075,
    )
    fig.text(
        0.5,
        -0.025,
        (
            f"Median-error permanent-test FoG window; "
            f"{rows.subject_id[selected]}, {rows.record_id[selected]}, samples {start}:{end}. "
            f"Window MAE={mae[selected]:.3f}, r={correlation[selected]:.3f}."
        ),
        ha="center",
        va="top",
        fontsize=7,
        color="#555555",
    )
    save_figure(fig, args.output_dir / "gru_nbm_fog_input_vs_reconstruction")
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
