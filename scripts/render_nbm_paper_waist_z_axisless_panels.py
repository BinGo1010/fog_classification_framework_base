#!/usr/bin/env python3
"""Render the S01/S02/S03 waist-IMU Z-axis augmentation panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 64
WINDOW_SAMPLES = 128
SUBJECTS = ("S01", "S02", "S03")
ORIGINAL_SIGNAL_COLOR = "#000000"
ZERO_BASELINE_COLOR = "#D9DCDF"
COLORS = {
    "original": "#2F6B9A",
    "gaussian": "#D98524",
    "mask": "#B84A5A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_paper_augmentation_axisless_panels"
        / "source_data.csv",
    )
    parser.add_argument(
        "--slice-metadata",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_paper_augmentation_axisless_panels"
        / "slice_metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_paper_augmentation_axisless_panels",
    )
    parser.add_argument("--panel-width", type=float, default=3.5)
    parser.add_argument("--panel-height", type=float, default=1.65)
    return parser.parse_args()


def load_traces(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["subject_id"], []).append(row)

    traces: dict[str, dict[str, np.ndarray]] = {}
    for subject in SUBJECTS:
        rows = grouped.get(subject, [])
        rows.sort(key=lambda row: float(row["plot_time_s"]))
        if len(rows) != WINDOW_SAMPLES:
            raise RuntimeError(
                f"{subject} must contribute exactly {WINDOW_SAMPLES} source-data rows"
            )
        traces[subject] = {
            "time": np.asarray([float(row["plot_time_s"]) for row in rows]),
            "clean": np.asarray([float(row["clean_waist_z"]) for row in rows]),
            "displayed": np.asarray(
                [float(row["displayed_waist_z"]) for row in rows]
            ),
            "mask_active": np.asarray([int(row["mask_active"]) for row in rows]),
        }
    return traces


def save_panel(
    output_dir: Path,
    stem: str,
    condition: str,
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
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_axis_off()
    ax.axhline(
        0.0,
        color=ZERO_BASELINE_COLOR,
        linewidth=0.65,
        linestyle="--",
        zorder=0.5,
    )

    if condition != "Original":
        ax.plot(
            time,
            clean,
            color=ORIGINAL_SIGNAL_COLOR,
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )
    if condition == "Time mask":
        if mask_active is None:
            raise RuntimeError("Time-mask panel requires mask metadata")
        mask_indices = np.flatnonzero(mask_active)
        if mask_indices.size == 0:
            raise RuntimeError("Time-mask panel contains no masked samples")
        start = int(mask_indices[0])
        end = int(mask_indices[-1] + 1)
        ax.axvspan(
            start / FS,
            end / FS,
            ymin=0.02,
            ymax=0.62,
            color=COLORS["mask"],
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )

    ax.plot(time, displayed, color=color, linewidth=1.35, zorder=2)
    path = output_dir / stem
    fig.savefig(path.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    traces = load_traces(args.input_csv.resolve())
    slice_metadata = json.loads(args.slice_metadata.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_values = np.concatenate(
        (
            traces["S01"]["clean"],
            traces["S02"]["clean"],
            traces["S02"]["displayed"],
            traces["S03"]["clean"],
            traces["S03"]["displayed"],
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
        traces["S01"]["time"],
        traces["S01"]["displayed"],
        traces["S01"]["clean"],
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
        traces["S02"]["displayed"],
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
        traces["S03"]["time"],
        traces["S03"]["displayed"],
        traces["S03"]["clean"],
        y_limit,
        args.panel_width,
        args.panel_height,
        traces["S03"]["mask_active"],
    )

    metadata = {
        "backend": "Python/matplotlib",
        "layout": "three separate axisless panels",
        "titles_retained": False,
        "keys_retained": False,
        "original_panel_color": COLORS["original"],
        "reference_signal_color": ORIGINAL_SIGNAL_COLOR,
        "zero_baseline_retained": True,
        "zero_baseline_color": ZERO_BASELINE_COLOR,
        "zero_baseline_linestyle": "dashed",
        "displayed_channel": slice_metadata["displayed_channel"],
        "selection_rule": slice_metadata["selection_rule"],
        "pooled_target_trunk_energy": slice_metadata[
            "pooled_target_trunk_energy"
        ],
        "axes_removed": True,
        "shared_y_limit": y_limit,
        "panel_size_inches": [args.panel_width, args.panel_height],
        "windows": slice_metadata["windows"],
        "outputs": ["01_original", "02_gaussian_noise", "03_time_mask"],
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_contract.md").write_text(
        """# Figure contract

- Core conclusion: Three standalone waist-IMU Z-axis signals from S01, S02, and S03 show unchanged, Gaussian-noised, and continuously masked NBM inputs without coordinate axes.
- Figure archetype: Three independent method illustrations.
- Backend: Python/matplotlib only.
- Output: Three separate 3.5 x 1.65 inch images.
- Panel mapping: S01 Original, S02 Gaussian noise, and S03 Time mask.
- Signal: `trunk_acc_vertical`, interpreted as waist IMU Z-axis acceleration.
- Selection: Each subject's valid role-4 window nearest the pooled median three-axis trunk energy across S01-S03.
- Retained elements: Signals, a shared gray dashed zero baseline, and the Time-mask interval shading.
- Removed elements: Processing-method titles, direct keys, axes, ticks, tick labels, axis titles, grid, subject labels, and panel letters.
- Integrity: All panels use the same y-range and no smoothing; values retain the fold-0 RobustScaler and per-window centering used for NBM inputs.
""",
        encoding="utf-8",
    )
    (output_dir / "qa_notes.md").write_text(
        """# Figure QA notes

- All three panels use the same y-limit even though the coordinate axes are hidden.
- No smoothing, interpolation, or post-selection amplitude normalization was applied.
- Source mapping: S01/S02/S03 waist IMU Z-axis (`trunk_acc_vertical`, channel index 7).
- Selection: Each subject's valid role-4 window nearest the pooled median three-axis trunk energy across S01-S03.
- Gaussian noise is illustrative with sigma 0.30; the actual training value remains sigma 0.04.
- The Original panel is blue; augmented panels retain black dashed original-signal references and contain no keys or text labels.
- Each panel contains the same thin gray dashed horizontal baseline at y = 0, drawn beneath the signal traces.
- All formats retain the exact 3.5 x 1.65 inch panel canvas; no content is clipped.
- Each panel is exported as editable SVG/PDF and 600-dpi PNG/TIFF.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
