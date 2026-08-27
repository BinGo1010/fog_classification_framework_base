"""Compare P01 GRU-NBM reconstructions from the original and clean-step training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
ORIGINAL_RUN = (
    REPO_ROOT
    / "outputs"
    / "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
    / "runs"
    / "P01"
    / "fold_0"
    / "seed_0"
)
CURRENT_RUN = (
    REPO_ROOT
    / "outputs"
    / "private_P01_gru_nbm_step5000_val50_pat20_lr3e4_seed0"
    / "fold_0"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "P01_gru_nbm_training_reconstruction_comparison"
CHANNEL_INDICES = (0, 10, 16)
CHANNEL_LABELS = (
    "Lumbar acceleration x",
    "Left shank angular velocity y",
    "Right shank angular velocity y",
)
METHODS = (
    ("Original DAE training", "40% clean / 40% Gaussian / 20% mask"),
    ("Current clean-step training", "100% clean; step-based early stopping"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--original-run", type=Path, default=ORIGINAL_RUN)
    parser.add_argument("--current-run", type=Path, default=CURRENT_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_scaler(path: Path) -> RobustScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))["scaler"]
    return RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float32),
        iqr=np.asarray(payload["iqr"], dtype=np.float32),
        epsilon=float(payload.get("epsilon", 1e-6)),
    )


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = worker.build_nbm_from_checkpoint(payload, device)
    model.eval()
    return model


def reconstruct_physical(
    model: torch.nn.Module,
    scaler: RobustScaler,
    raw_window: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = scaler.transform(raw_window[None, :, :])
    temporal_mean = scaled.mean(axis=1, keepdims=True)
    centered = np.ascontiguousarray(scaled - temporal_mean, dtype=np.float32)
    with torch.no_grad():
        prediction = (
            model(torch.from_numpy(centered).to(device))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    reconstructed_scaled = prediction + temporal_mean
    reconstructed_raw = (
        reconstructed_scaled * (scaler.iqr[None, None, :] + scaler.epsilon)
        + scaler.median[None, None, :]
    )
    return centered[0], prediction[0], reconstructed_raw[0].astype(np.float32)


def smooth_l1(target: np.ndarray, prediction: np.ndarray) -> float:
    absolute = np.abs(np.asarray(target) - np.asarray(prediction))
    return float(np.mean(np.where(absolute < 1.0, 0.5 * absolute**2, absolute - 0.5)))


def correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if float(np.std(target)) <= 1e-12 or float(np.std(prediction)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(target, prediction)[0, 1])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.original_run = args.original_run.resolve()
    args.current_run = args.current_run.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = worker.resolve_device(args.device)

    dataset = DaphnetDataset.load(args.data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 30:
        raise AssertionError("expected the processed_NBM_Exp 64-Hz, 30-channel dataset")
    schema = json.loads((args.data_dir / "schema.json").read_text(encoding="utf-8"))
    channel_names = [item["name"] for item in schema["channels"]]
    expected_names = ("imu_lumbar_ax", "imu_ankle_l_gy", "imu_ankle_r_gy")
    actual_names = tuple(channel_names[index] for index in CHANNEL_INDICES)
    if actual_names != expected_names:
        raise AssertionError(f"channel contract changed: {actual_names}")

    rows = raw_base.load_subject_rows(args.data_dir, dataset, "P01", 0)
    role5 = rows.take_role(5)
    if len(role5) == 0 or not np.all(role5.label == 0):
        raise AssertionError("role 5 must be non-empty clean Non-FoG")
    raw_role5 = raw_base.raw_windows(dataset, role5)

    original_scaler = load_scaler(args.original_run / "scaler_role4.json")
    current_scaler = load_scaler(args.current_run / "scaler_role4.json")
    median_difference = float(np.max(np.abs(original_scaler.median - current_scaler.median)))
    iqr_difference = float(np.max(np.abs(original_scaler.iqr - current_scaler.iqr)))
    if median_difference > 1e-7 or iqr_difference > 1e-7:
        raise AssertionError("the two runs did not use the same role-4 RobustScaler")

    selection_domain = worker.centered_scaled_ntc(current_scaler, raw_role5)
    motion_energy = np.mean(selection_domain[:, :, CHANNEL_INDICES] ** 2, axis=(1, 2))
    median_energy = float(np.median(motion_energy))
    selected_index = int(np.argmin(np.abs(motion_energy - median_energy)))
    raw_window = raw_role5[selected_index]

    original_checkpoint = args.original_run / "checkpoints" / "gru_nbm_best.pt"
    current_checkpoint = args.current_run / "checkpoints" / "gru_nbm_best.pt"
    models = (
        load_model(original_checkpoint, device),
        load_model(current_checkpoint, device),
    )
    scalers = (original_scaler, current_scaler)
    reconstructions: list[np.ndarray] = []
    centered_targets: list[np.ndarray] = []
    centered_predictions: list[np.ndarray] = []
    for model, scaler in zip(models, scalers):
        target, prediction, reconstruction = reconstruct_physical(
            model, scaler, raw_window, device
        )
        centered_targets.append(target)
        centered_predictions.append(prediction)
        reconstructions.append(reconstruction)

    time_seconds = np.arange(raw_window.shape[0], dtype=np.float64) / dataset.sampling_rate_hz
    colors = ("#D97706", "#2563EB")
    observed_color = "#4B5563"
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 8.4), sharex=True)
    metric_rows: list[dict[str, Any]] = []
    for row_index, (channel_index, label) in enumerate(
        zip(CHANNEL_INDICES, CHANNEL_LABELS)
    ):
        values_for_limits = [raw_window[:, channel_index]] + [
            reconstruction[:, channel_index] for reconstruction in reconstructions
        ]
        minimum = min(float(np.min(values)) for values in values_for_limits)
        maximum = max(float(np.max(values)) for values in values_for_limits)
        padding = max((maximum - minimum) * 0.10, 1e-3)
        for column_index, ((title, subtitle), reconstruction) in enumerate(
            zip(METHODS, reconstructions)
        ):
            axis = axes[row_index, column_index]
            axis.plot(
                time_seconds,
                raw_window[:, channel_index],
                color=observed_color,
                linewidth=1.35,
                label="Observed signal",
                zorder=2,
            )
            axis.plot(
                time_seconds,
                reconstruction[:, channel_index],
                color=colors[column_index],
                linewidth=1.65,
                label="NBM reconstruction",
                zorder=3,
            )
            axis.set_ylim(minimum - padding, maximum + padding)
            axis.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                axis.set_title(f"{title}\n{subtitle}", fontsize=12, pad=10)
            if column_index == 0:
                unit = "g" if channel_index == 0 else "rad/s"
                axis.set_ylabel(f"{label}\n({unit})", fontsize=10)
            target = centered_targets[column_index][:, channel_index]
            prediction = centered_predictions[column_index][:, channel_index]
            corr = correlation(target, prediction)
            amplitude_ratio = float(
                np.std(reconstruction[:, channel_index])
                / max(float(np.std(raw_window[:, channel_index])), 1e-12)
            )
            metric_rows.append(
                {
                    "subject": "P01",
                    "fold": 0,
                    "seed": 0,
                    "role": 5,
                    "window_id": str(role5.window_id[selected_index]),
                    "record_id": str(role5.record_id[selected_index]),
                    "start_index": int(role5.start[selected_index]),
                    "channel_index": channel_index,
                    "channel_name": channel_names[channel_index],
                    "method": title,
                    "centered_smooth_l1": smooth_l1(target, prediction),
                    "physical_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    raw_window[:, channel_index]
                                    - reconstruction[:, channel_index]
                                )
                                ** 2
                            )
                        )
                    ),
                    "temporal_correlation": corr,
                    "amplitude_ratio": amplitude_ratio,
                }
            )
            axis.text(
                0.985,
                0.94,
                f"r = {corr:.2f}\nAmplitude ratio = {amplitude_ratio:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.88},
            )
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    fig.legend(
        handles=[
            Line2D([0], [0], color=observed_color, linewidth=1.35),
            Line2D([0], [0], color=colors[0], linewidth=1.65),
            Line2D([0], [0], color=colors[1], linewidth=1.65),
        ],
        labels=[
            "Observed signal",
            "Original DAE reconstruction",
            "Current clean-step reconstruction",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "P01 clean Non-FoG reconstruction under two GRU-NBM training strategies",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.995, 0.925), h_pad=1.25, w_pad=1.15)
    stem = args.output_dir / "P01_original_vs_clean_step_gru_nbm_reconstruction"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=args.dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=args.dpi, facecolor="white")
    plt.close(fig)

    metrics_path = args.output_dir / "reconstruction_metrics.csv"
    write_csv(metrics_path, metric_rows)
    audit = {
        "schema": "p01_gru_nbm_training_reconstruction_comparison.v1",
        "subject": "P01",
        "fold": 0,
        "seed": 0,
        "window_role": 5,
        "window_label": "clean Non-FoG",
        "selection_rule": (
            "role-5 window whose robust-scaled and window-centered mean-square "
            "energy over the three predeclared channels is closest to the role-5 median"
        ),
        "selected_role5_index": selected_index,
        "selected_window_id": str(role5.window_id[selected_index]),
        "selected_record_id": str(role5.record_id[selected_index]),
        "selected_start_index": int(role5.start[selected_index]),
        "selected_end_index": int(role5.end[selected_index]),
        "selected_motion_energy": float(motion_energy[selected_index]),
        "median_motion_energy": median_energy,
        "channels": [
            {
                "index": index,
                "name": channel_names[index],
                "display_label": label,
            }
            for index, label in zip(CHANNEL_INDICES, CHANNEL_LABELS)
        ],
        "display_domain": (
            "physical units after restoring the observed window mean and inverting "
            "the role-4 RobustScaler"
        ),
        "scaler_max_absolute_median_difference": median_difference,
        "scaler_max_absolute_iqr_difference": iqr_difference,
        "artifacts_sha256": {
            "original_nbm_checkpoint": sha256_file(original_checkpoint),
            "current_nbm_checkpoint": sha256_file(current_checkpoint),
            "png": sha256_file(stem.with_suffix(".png")),
            "pdf": sha256_file(stem.with_suffix(".pdf")),
            "svg": sha256_file(stem.with_suffix(".svg")),
            "metrics": sha256_file(metrics_path),
        },
    }
    atomic_json_dump(audit, args.output_dir / "audit.json")
    print(json.dumps(audit, indent=2), flush=True)
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
