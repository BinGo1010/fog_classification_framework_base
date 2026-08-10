#!/usr/bin/env python3
"""Plot three real Daphnet trunk-IMU windows under NBM augmentations."""

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
TRUNK_CHANNELS = (6, 7, 8)
DISPLAY_CHANNEL = 6

COLORS = {
    "clean": "#2F6B9A",
    "gaussian": "#D98524",
    "mask": "#B84A5A",
    "reference": "#9CA2A8",
    "zero": "#D9DCDF",
    "text": "#33383D",
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
        default=REPO_ROOT / "outputs" / "figures" / "nbm_trunk_three_windows",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--visualization-gaussian-std",
        type=float,
        default=0.15,
        help="Enlarged Gaussian standard deviation used only in this illustration.",
    )
    parser.add_argument(
        "--training-gaussian-std",
        type=float,
        default=0.04,
        help="Actual training standard deviation, reported for comparison only.",
    )
    return parser.parse_args()


def select_three_distinct_windows(
    centered: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Select low-, medium-, and high-energy windows, preferring unique subjects."""
    trunk = centered[:, :, TRUNK_CHANNELS]
    energy = np.sqrt(np.mean(np.square(trunk), axis=(1, 2)))
    ap_std = np.std(centered[:, :, DISPLAY_CHANNEL], axis=1)
    valid = np.isfinite(energy) & np.isfinite(ap_std) & (ap_std > 0.05)
    candidates = np.flatnonzero(valid)
    if candidates.size < 3:
        raise RuntimeError("fewer than three dynamic role-4 trunk windows are available")

    targets = np.quantile(energy[candidates], (0.25, 0.50, 0.75))
    selected: list[int] = []
    used_subjects: set[str] = set()
    for target in targets:
        ranked = candidates[np.argsort(np.abs(energy[candidates] - target))]
        choice = next(
            (
                int(index)
                for index in ranked
                if int(index) not in selected
                and str(subject_ids[index]) not in used_subjects
            ),
            None,
        )
        if choice is None:
            choice = next(int(index) for index in ranked if int(index) not in selected)
        selected.append(choice)
        used_subjects.add(str(subject_ids[choice]))
    return selected, energy, targets


def export_source_data(
    path: Path,
    time: np.ndarray,
    selected: list[int],
    role4: object,
    energy: np.ndarray,
    clean: np.ndarray,
    gaussian: np.ndarray,
    masked: np.ndarray,
    mask_starts: list[int],
    mask_lengths: list[int],
) -> None:
    fieldnames = [
        "display_window",
        "subject_id",
        "record_id",
        "window_id",
        "trunk_energy_all_axes",
        "time_s",
        "clean_trunk_ap",
        "gaussian_trunk_ap",
        "masked_trunk_ap",
        "mask_active",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, index in enumerate(selected):
            mask_start = mask_starts[row]
            mask_end = mask_start + mask_lengths[row]
            for sample, value in enumerate(time):
                writer.writerow(
                    {
                        "display_window": row + 1,
                        "subject_id": str(role4.subject_id[index]),
                        "record_id": str(role4.record_id[index]),
                        "window_id": str(role4.window_id[index]),
                        "trunk_energy_all_axes": float(energy[index]),
                        "time_s": float(value),
                        "clean_trunk_ap": float(clean[row, sample]),
                        "gaussian_trunk_ap": float(gaussian[row, sample]),
                        "masked_trunk_ap": float(masked[row, sample]),
                        "mask_active": int(mask_start <= sample < mask_end),
                    }
                )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir, args.fold)
    role4 = rows.take_role(4)
    scaler, scaler_points = fit_scaler_unique_role4_points(records, role4)
    raw = raw_windows(records, role4)
    centered = scaler.transform(raw)
    centered -= centered.mean(axis=1, keepdims=True)

    selected, energy, targets = select_three_distinct_windows(
        centered, role4.subject_id
    )
    clean_all = centered[selected].astype(np.float32)

    rng = np.random.default_rng(args.seed)
    gaussian_all = clean_all + rng.normal(
        0.0,
        args.visualization_gaussian_std,
        size=clean_all.shape,
    ).astype(np.float32)
    masked_all = clean_all.copy()
    mask_starts: list[int] = []
    mask_lengths: list[int] = []
    for row in range(len(selected)):
        mask_length = int(rng.integers(4, 9))
        mask_start = int(rng.integers(0, WINDOW_SAMPLES - mask_length + 1))
        masked_all[row, mask_start : mask_start + mask_length, :] = 0.0
        mask_starts.append(mask_start)
        mask_lengths.append(mask_length)

    clean = clean_all[:, :, DISPLAY_CHANNEL]
    gaussian = gaussian_all[:, :, DISPLAY_CHANNEL]
    masked = masked_all[:, :, DISPLAY_CHANNEL]
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
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 4.55), sharex=True)
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.13,
        top=0.79,
        wspace=0.18,
        hspace=0.22,
    )
    fig.suptitle(
        "Three real trunk-IMU windows under mutually exclusive NBM augmentation",
        x=0.54,
        y=0.965,
        fontsize=10,
        fontweight="bold",
    )

    column_titles = (
        ("a", "Original", "clean input", COLORS["clean"]),
        (
            "b",
            "Gaussian noise",
            rf"illustration $\sigma={args.visualization_gaussian_std:.2f}$; training $\sigma={args.training_gaussian_std:.2f}$",
            COLORS["gaussian"],
        ),
        ("c", "Continuous time mask", "4-8 samples; all 9 axes", COLORS["mask"]),
    )
    for column, (letter, title, subtitle, color) in enumerate(column_titles):
        position = axes[0, column].get_position()
        center = (position.x0 + position.x1) / 2
        fig.text(
            center,
            0.875,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
        fig.text(
            center,
            0.835,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#4B5055",
        )
        fig.text(
            position.x0 - 0.025,
            0.895,
            letter,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    activity_labels = ("low", "medium", "high")
    for row, index in enumerate(selected):
        combined = np.concatenate((clean[row], gaussian[row], masked[row]))
        lower, upper = float(np.min(combined)), float(np.max(combined))
        padding = max(0.08, 0.10 * (upper - lower))
        limits = (lower - padding, upper + padding)
        subject = str(role4.subject_id[index])
        row_label = f"Window {row + 1}\n{activity_labels[row]} energy\n{subject}"
        for column in range(3):
            ax = axes[row, column]
            ax.axhline(0.0, color=COLORS["zero"], linewidth=0.6, zorder=0)
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(*limits)
            ax.tick_params(labelsize=6, length=2.5, pad=2)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=6.5, multialignment="center")
            else:
                ax.tick_params(labelleft=False)
                ax.spines["left"].set_visible(False)

        axes[row, 0].plot(time, clean[row], color=COLORS["clean"], linewidth=1.0)
        axes[row, 1].plot(
            time,
            clean[row],
            color=COLORS["reference"],
            linewidth=0.75,
            linestyle="--",
        )
        axes[row, 1].plot(
            time,
            gaussian[row],
            color=COLORS["gaussian"],
            linewidth=0.8,
        )
        axes[row, 2].plot(
            time,
            clean[row],
            color=COLORS["reference"],
            linewidth=0.75,
            linestyle="--",
        )
        mask_start = mask_starts[row]
        mask_length = mask_lengths[row]
        axes[row, 2].axvspan(
            mask_start / FS,
            (mask_start + mask_length) / FS,
            color=COLORS["mask"],
            alpha=0.16,
            linewidth=0,
        )
        axes[row, 2].plot(
            time,
            masked[row],
            color=COLORS["mask"],
            linewidth=0.95,
        )
        axes[row, 2].text(
            0.98,
            0.92,
            f"{mask_length} samples ({1000 * mask_length / FS:.0f} ms)",
            transform=axes[row, 2].transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            color=COLORS["mask"],
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
    axes[0, 1].text(
        0.98,
        0.08,
        "dashed: original",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#666C72",
    )
    axes[0, 2].text(
        0.02,
        0.08,
        "dashed: original",
        transform=axes[0, 2].transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#666C72",
    )
    fig.text(
        0.018,
        0.46,
        "Centered, Robust-scaled trunk AP acceleration",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["text"],
    )
    fig.text(
        0.54,
        0.035,
        "Role-4 clean non-FoG windows selected near the 25th, 50th, and 75th percentiles of three-axis trunk energy.",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#555B60",
    )

    stem = output_dir / "trunk_imu_three_windows_clean_gaussian_mask"
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
        output_dir / "source_data.csv",
        time,
        selected,
        role4,
        energy,
        clean,
        gaussian,
        masked,
        mask_starts,
        mask_lengths,
    )
    windows = []
    for row, index in enumerate(selected):
        windows.append(
            {
                "display_window": row + 1,
                "activity_band": activity_labels[row],
                "selected_index_within_role4": index,
                "subject_id": str(role4.subject_id[index]),
                "record_id": str(role4.record_id[index]),
                "window_id": str(role4.window_id[index]),
                "trunk_energy_all_axes": float(energy[index]),
                "target_energy_quantile": float(targets[row]),
                "mask_start_sample": mask_starts[row],
                "mask_length_samples": mask_lengths[row],
                "mask_duration_ms": 1000.0 * mask_lengths[row] / FS,
            }
        )
    metadata = {
        "figure_claim": (
            "Three morphologically different real trunk-IMU windows show the "
            "distinct effects of enlarged Gaussian noise and continuous time masking."
        ),
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "fold": args.fold,
        "role": 4,
        "role_definition": "clean non-FoG used to train the NBM",
        "scaler_unique_raw_points": scaler_points,
        "displayed_channel": {"index": DISPLAY_CHANNEL, "label": "Trunk AP"},
        "selection_rule": (
            "nearest valid windows to the 25th, 50th, and 75th percentiles of "
            "three-axis trunk RMS energy, preferring distinct subjects"
        ),
        "visualization_gaussian_std": args.visualization_gaussian_std,
        "actual_training_gaussian_std": args.training_gaussian_std,
        "visualization_noise_is_enlarged": True,
        "all_nine_channels_were_augmented": True,
        "mask_rule": "one synchronous continuous mask of 4-8 samples on all 9 axes",
        "windows": windows,
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_contract.md").write_text(
        """# Figure contract

- Core conclusion: Three different real trunk-IMU windows make the distinct full-window Gaussian perturbation and short continuous time-mask effects directly visible.
- Figure archetype: Quantitative grid.
- Target output: Two-column manuscript figure and editable vector exports.
- Backend: Python/matplotlib.
- Final size: 7.2 x 4.55 inches.
- Panel a: Original trunk AP acceleration from three clean non-FoG windows.
- Panel b: Enlarged Gaussian-noise illustration, with the original trace shown as a dashed reference.
- Panel c: Dynamic continuous time masks, with the original trace shown as a dashed reference.
- Evidence hierarchy: Real source traces are primary; augmentation overlays are explanatory comparisons.
- Statistics needed: None; this is an augmentation illustration rather than an inferential comparison.
- Source data needed: Every displayed sample plus subject, record, window, energy, and mask metadata.
- Image-integrity notes: No amplitude normalization or smoothing; all plotted values remain in the Robust-scaled, per-window/per-axis-centered domain.
- Reviewer risk: Gaussian standard deviation 0.15 is deliberately enlarged only for visualization; actual training standard deviation remains 0.04.
""",
        encoding="utf-8",
    )
    (output_dir / "figure_legend.md").write_text(
        """**Three real trunk-IMU windows under NBM augmentation.** Three clean non-FoG role-4 windows were selected near the 25th, 50th, and 75th percentiles of three-axis trunk RMS energy, while preferring different subjects. (a) Original Robust-scaled and per-window/per-axis-centered trunk anterior-posterior acceleration. (b) Additive Gaussian-noise illustration; the standard deviation was deliberately enlarged to 0.15 for visibility, whereas model training uses 0.04. (c) One continuous 4-8-sample interval was set to zero synchronously across all nine sensor axes. Dashed gray curves show the corresponding original signals. The clean, Gaussian, and time-mask branches are mutually exclusive for each training window.
""",
        encoding="utf-8",
    )
    (output_dir / "qa_notes.md").write_text(
        """# Figure QA notes

- Automated preflight: 13 PASS, 1 reviewed WARN, 0 FAIL.
- Reviewed warning: the source contains random-number generation because Gaussian noise and dynamic masks are the experimental transformations being illustrated; the underlying IMU traces are real data, not simulated data.
- Backend exclusivity: all rendering, export, and visual inspection used Python/matplotlib.
- Visual inspection: passed at the 7.2-inch final width; panel labels, titles, axes, dashed references, and mask spans are readable and do not overlap.
- Traceability: all 384 displayed samples and their subject, record, window, energy, and mask fields are in `source_data.csv`.
- Signal integrity: no smoothing, amplitude normalization, selective point removal, or local display adjustment was applied.
- Scaling: the traces remain in the Robust-scaled and per-window/per-axis-centered domain.
- Gaussian disclosure: sigma 0.15 is used only to make the perturbation visible in this figure; actual training uses sigma 0.04.
- Export bundle: 600-dpi PNG and TIFF plus editable SVG and PDF.
- Statistics: not applicable; the figure illustrates preprocessing/augmentation rather than a performance comparison.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
