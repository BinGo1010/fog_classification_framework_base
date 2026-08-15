#!/usr/bin/env python3
"""Export Original, Gaussian-noise, and Time-mask traces as separate axisless panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_nbm_paper_augmentation_triptych import (
    COLORS,
    WINDOW_SAMPLES,
    load_subject_traces,
)

FS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_trunk_three_windows"
        / "source_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_paper_augmentation_axisless_panels",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--visualization-gaussian-std", type=float, default=0.30)
    parser.add_argument("--panel-width", type=float, default=3.5)
    parser.add_argument("--panel-height", type=float, default=1.65)
    return parser.parse_args()


def draw_reference_key(ax: plt.Axes, y: float) -> None:
    ax.plot(
        [0.62, 0.70],
        [y, y],
        transform=ax.transAxes,
        color=COLORS["reference"],
        linewidth=1.0,
        linestyle="--",
        clip_on=False,
        zorder=5,
    )
    ax.text(
        0.72,
        y,
        "Original signal",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color="#777D82",
        zorder=5,
    )


def draw_mask_key(ax: plt.Axes, y: float) -> None:
    ax.add_patch(
        Rectangle(
            (0.62, y - 0.0125),
            0.08,
            0.025,
            transform=ax.transAxes,
            facecolor=COLORS["mask"],
            edgecolor=COLORS["mask"],
            linewidth=0.8,
            alpha=0.16,
            clip_on=False,
            zorder=5,
        )
    )
    ax.text(
        0.72,
        y,
        "Masked interval",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color=COLORS["mask"],
        zorder=5,
    )


def save_panel(
    output_dir: Path,
    stem: str,
    title: str,
    color: str,
    time: np.ndarray,
    displayed: np.ndarray,
    clean: np.ndarray,
    y_limit: float,
    width: float,
    height: float,
    mask_active: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.80)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=5)

    if title != "Original":
        ax.plot(
            time,
            clean,
            color=COLORS["reference"],
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )

    if title == "Time mask":
        if mask_active is None:
            raise RuntimeError("Time mask panel requires mask metadata")
        mask_indices = np.flatnonzero(mask_active)
        if mask_indices.size == 0:
            raise RuntimeError("Time mask panel contains no masked samples")
        mask_start = int(mask_indices[0])
        mask_end = int(mask_indices[-1] + 1)
        ax.axvspan(
            mask_start / FS,
            mask_end / FS,
            ymin=0.02,
            ymax=0.62,
            color=COLORS["mask"],
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )

    ax.plot(time, displayed, color=color, linewidth=1.35, zorder=2)

    if title == "Gaussian noise":
        draw_reference_key(ax, 0.88)
    elif title == "Time mask":
        draw_reference_key(ax, 0.87)
        draw_mask_key(ax, 0.70)

    path = output_dir / stem
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    traces = load_subject_traces(args.input_csv.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    gaussian = traces["S02"]["clean"] + rng.normal(
        0.0,
        args.visualization_gaussian_std,
        size=WINDOW_SAMPLES,
    )
    all_values = np.concatenate(
        (
            traces["S03"]["clean"],
            traces["S02"]["clean"],
            gaussian,
            traces["S09"]["clean"],
            traces["S09"]["masked"],
        )
    )
    y_limit = 1.08 * float(np.max(np.abs(all_values)))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
        }
    )

    save_panel(
        output_dir,
        "01_original",
        "Original",
        COLORS["original"],
        traces["S03"]["time"],
        traces["S03"]["clean"],
        traces["S03"]["clean"],
        y_limit,
        args.panel_width,
        args.panel_height,
    )
    save_panel(
        output_dir,
        "02_gaussian_noise",
        "Gaussian noise",
        COLORS["gaussian"],
        traces["S02"]["time"],
        gaussian,
        traces["S02"]["clean"],
        y_limit,
        args.panel_width,
        args.panel_height,
    )
    save_panel(
        output_dir,
        "03_time_mask",
        "Time mask",
        COLORS["mask"],
        traces["S09"]["time"],
        traces["S09"]["masked"],
        traces["S09"]["clean"],
        y_limit,
        args.panel_width,
        args.panel_height,
        traces["S09"]["mask_active"],
    )

    metadata = {
        "backend": "Python/matplotlib",
        "layout": "three separate axisless panels",
        "titles_retained": True,
        "keys_retained": True,
        "axes_removed": True,
        "shared_y_limit": y_limit,
        "visualization_gaussian_std": args.visualization_gaussian_std,
        "panel_size_inches": [args.panel_width, args.panel_height],
        "outputs": ["01_original", "02_gaussian_noise", "03_time_mask"],
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_contract.md").write_text(
        """# Figure contract

- Core conclusion: Three standalone signals show unchanged, Gaussian-noised, and continuously masked NBM inputs without coordinate axes.
- Figure archetype: Three independent method illustrations.
- Backend: Python/matplotlib only.
- Output: Three separate 3.5 x 1.65 inch images.
- Retained elements: signal, processing-method title, and applicable direct keys.
- Removed elements: axes, ticks, tick labels, axis titles, grid, subject labels, and panel letters.
- Integrity: all panels use the same y-range and no amplitude normalization or smoothing.
""",
        encoding="utf-8",
    )
    (output_dir / "qa_notes.md").write_text(
        """# Figure QA notes

- All three panels use the same y-limit even though the coordinate axes are hidden.
- No smoothing, interpolation, or amplitude normalization was applied.
- Gaussian noise is illustrative with sigma 0.30; the actual training value remains sigma 0.04.
- The Time-mask key is isolated from the real mask span by an unbordered white background.
- Each panel is exported as editable SVG/PDF and 600-dpi PNG/TIFF.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
