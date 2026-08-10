#!/usr/bin/env python3
"""Plot clean, Gaussian, and time-mask NBM inputs from one real Daphnet window."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
)

FS = 64
WINDOW_SAMPLES = 128
DISPLAY_CHANNELS = (0, 3, 6)
DISPLAY_LABELS = ("Ankle AP", "Thigh AP", "Trunk AP")

COLORS = {
    "clean": "#2F6B9A",
    "gaussian": "#D98524",
    "mask": "#B84A5A",
    "reference": "#A3A8AE",
    "zero": "#D8DADD",
}


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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "nbm_dynamic_augmentation",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--gaussian-std", type=float, default=0.04)
    return parser.parse_args()


def select_representative_window(centered: np.ndarray) -> int:
    selected = centered[:, :, DISPLAY_CHANNELS]
    channel_energy = np.sqrt(np.mean(np.square(selected), axis=1))
    valid = np.all(channel_energy > 0.05, axis=1)
    if not np.any(valid):
        raise RuntimeError("no role-4 window has dynamic signal on all displayed IMUs")
    energy = np.sqrt(np.mean(np.square(selected), axis=(1, 2)))
    target = np.median(energy[valid])
    candidates = np.flatnonzero(valid)
    return int(candidates[np.argmin(np.abs(energy[candidates] - target))])


def export_source_data(
    path: Path,
    time: np.ndarray,
    clean: np.ndarray,
    gaussian: np.ndarray,
    masked: np.ndarray,
) -> None:
    fieldnames = ["time_s"]
    for prefix in ("clean", "gaussian", "mask"):
        fieldnames.extend(f"{prefix}_{label.lower().replace(' ', '_')}" for label in DISPLAY_LABELS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, value in enumerate(time):
            row: dict[str, float] = {"time_s": float(value)}
            for prefix, values in (
                ("clean", clean),
                ("gaussian", gaussian),
                ("mask", masked),
            ):
                for channel_index, label in enumerate(DISPLAY_LABELS):
                    key = f"{prefix}_{label.lower().replace(' ', '_')}"
                    row[key] = float(values[index, channel_index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()

    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir, args.fold)
    role4 = rows.take_role(4)
    scaler, scaler_points = fit_scaler_unique_role4_points(records, role4)
    raw = raw_windows(records, role4)
    centered = scaler.transform(raw)
    centered -= centered.mean(axis=1, keepdims=True)
    window_index = select_representative_window(centered)
    clean_all = centered[window_index].astype(np.float32)

    noise_rng = np.random.default_rng(args.seed)
    gaussian_all = clean_all + noise_rng.normal(
        0.0, args.gaussian_std, size=clean_all.shape
    ).astype(np.float32)
    mask_rng = np.random.default_rng(args.seed)
    mask_length = int(mask_rng.integers(4, 9))
    mask_start = int(mask_rng.integers(0, WINDOW_SAMPLES - mask_length + 1))
    mask_end = mask_start + mask_length
    masked_all = clean_all.copy()
    masked_all[mask_start:mask_end, :] = 0.0

    clean = clean_all[:, DISPLAY_CHANNELS]
    gaussian = gaussian_all[:, DISPLAY_CHANNELS]
    masked = masked_all[:, DISPLAY_CHANNELS]
    time = np.arange(WINDOW_SAMPLES, dtype=np.float32) / FS

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.2, 4.2),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.79, wspace=0.18, hspace=0.16)
    fig.suptitle(
        "Dynamic per-window NBM augmentation (mutually exclusive; resampled every epoch)",
        x=0.535,
        y=0.965,
        fontsize=10,
        fontweight="bold",
    )
    column_titles = (
        ("a", "Clean input", "40% of windows", COLORS["clean"]),
        ("b", "Gaussian input", r"40%; $\sigma=0.04$", COLORS["gaussian"]),
        (
            "c",
            "Time-mask input",
            f"20%; {mask_length} samples ({1000 * mask_length / FS:.0f} ms)",
            COLORS["mask"],
        ),
    )
    for column, (letter, title, subtitle, color) in enumerate(column_titles):
        position = axes[0, column].get_position()
        center = (position.x0 + position.x1) / 2
        fig.text(center, 0.885, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)
        fig.text(center, 0.843, subtitle, ha="center", va="center", fontsize=7, color="#44484D")
        fig.text(position.x0 - 0.02, 0.905, letter, ha="left", va="center", fontsize=10, fontweight="bold")

    for row, label in enumerate(DISPLAY_LABELS):
        row_values = np.concatenate((clean[:, row], gaussian[:, row], masked[:, row]))
        lower, upper = float(np.min(row_values)), float(np.max(row_values))
        padding = max(0.08, 0.10 * (upper - lower))
        limits = (lower - padding, upper + padding)
        for column in range(3):
            ax = axes[row, column]
            ax.axhline(0.0, color=COLORS["zero"], linewidth=0.6, zorder=0)
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(*limits)
            ax.tick_params(labelsize=6, length=2.5, pad=2)
            if column == 0:
                ax.set_ylabel(label, fontsize=7)
            else:
                ax.tick_params(labelleft=False)
                ax.spines["left"].set_visible(False)
        axes[row, 0].plot(time, clean[:, row], color=COLORS["clean"], linewidth=1.05)
        axes[row, 1].plot(
            time,
            clean[:, row],
            color=COLORS["reference"],
            linewidth=0.8,
            linestyle="--",
            alpha=0.9,
        )
        axes[row, 1].plot(time, gaussian[:, row], color=COLORS["gaussian"], linewidth=0.85)
        axes[row, 2].plot(
            time,
            clean[:, row],
            color=COLORS["reference"],
            linewidth=0.8,
            linestyle="--",
            alpha=0.9,
        )
        axes[row, 2].axvspan(
            mask_start / FS,
            mask_end / FS,
            color=COLORS["mask"],
            alpha=0.14,
            linewidth=0,
        )
        axes[row, 2].plot(time, masked[:, row], color=COLORS["mask"], linewidth=0.95)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
    axes[0, 1].text(
        0.98,
        0.94,
        "dashed: clean reference",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#686D72",
    )
    axes[0, 2].annotate(
        "same interval on all 9 axes",
        xy=((mask_start + mask_length / 2) / FS, 0.0),
        xytext=(1.05, 0.90),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=6,
        color=COLORS["mask"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["mask"], "lw": 0.7},
    )
    fig.text(
        0.535,
        0.035,
        "Representative clean non-FoG window after Robust scaling and per-axis centering; one axis per IMU shown.",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#555A60",
    )

    stem = output_dir / "nbm_dynamic_augmentation_clean_gaussian_mask"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    export_source_data(
        output_dir / "source_data.csv", time, clean, gaussian, masked
    )
    metadata = {
        "figure_claim": (
            "Clean, Gaussian, and time-mask branches apply mutually exclusive "
            "transformations to the same NBM training window."
        ),
        "archetype": "schematic-led quantitative triptych",
        "backend": "Python/matplotlib",
        "fold": args.fold,
        "role": 4,
        "selected_window_index_within_role4": window_index,
        "subject_id": str(role4.subject_id[window_index]),
        "record_id": str(role4.record_id[window_index]),
        "window_id": str(role4.window_id[window_index]),
        "scaler_unique_raw_points": scaler_points,
        "gaussian_std": args.gaussian_std,
        "mask_start_sample": mask_start,
        "mask_end_sample_exclusive": mask_end,
        "mask_length_samples": mask_length,
        "mask_duration_ms": 1000.0 * mask_length / FS,
        "displayed_channels": [
            {"index": index, "label": label}
            for index, label in zip(DISPLAY_CHANNELS, DISPLAY_LABELS)
        ],
        "all_nine_channels_were_augmented": True,
        "figure_note": "Only one axis per IMU is displayed to preserve readability.",
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_contract.md").write_text(
        """# Figure contract

- Core conclusion: the three NBM augmentation branches are mutually exclusive and preserve, perturb, or briefly remove the same clean signal window.
- Archetype: schematic-led quantitative triptych.
- Backend: Python/matplotlib only.
- Final size: two-column, 7.2 x 4.2 inches.
- Panel a: unchanged clean input (40%).
- Panel b: additive Gaussian noise with standard deviation 0.04 (40%).
- Panel c: one synchronous all-axis time mask of 4-8 samples (20%).
- Source data: one median-energy role-4 clean non-FoG window; full plotted traces exported to CSV.
- Reviewer risk: this is a representative augmentation illustration, not a distributional or performance comparison.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
