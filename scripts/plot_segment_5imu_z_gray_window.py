"""Plot five IMU z-acceleration channels with a configurable gray time window."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["path.simplify"] = False


AZ_COLUMNS = tuple(f"imu{index}_az" for index in range(1, 6))
IMU_COLORS = ("#0F4D92", "#3775BA", "#42949E", "#9A4D8E", "#606060")
GRAY_SHADE = "#D9D9D9"
GRAY_EDGE = "#606060"


# =============================================================================
# Manual settings for running directly from PyCharm.
# Edit this block, then click Run. Command-line arguments are not required.
# GRAY_START_S and GRAY_END_S are seconds relative to the segment start.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "segments"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "figures"
OUTPUT_BASENAME = ""  # Empty: automatic name; example: "seg000_gray_0_20s"
SEGMENT_ID = "seg000"
GRAY_START_S = 0.0
GRAY_END_S = 20.0
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-name", default=OUTPUT_BASENAME)
    parser.add_argument("--segment-id", default=SEGMENT_ID)
    parser.add_argument("--gray-start", type=float, default=GRAY_START_S)
    parser.add_argument("--gray-end", type=float, default=GRAY_END_S)
    return parser.parse_args()


def load_segment(
    input_dir: Path,
    segment_id: str,
) -> tuple[pd.DataFrame, pd.Series]:
    summary_paths = sorted(input_dir.glob("*_segments_summary.csv"))
    if len(summary_paths) != 1:
        raise ValueError(f"Expected exactly one segment summary CSV, found {len(summary_paths)}")
    summary = pd.read_csv(summary_paths[0])
    summary_row = summary.loc[summary["segment_id"] == segment_id]
    if len(summary_row) != 1:
        raise ValueError(f"Expected one summary row for {segment_id}, found {len(summary_row)}")
    row = summary_row.iloc[0]

    usecols = ["relative_time", *AZ_COLUMNS]
    frame = pd.read_csv(input_dir / str(row["file"]), usecols=usecols)
    values = frame[usecols].to_numpy(dtype=float)
    if len(frame) == 0:
        raise ValueError(f"{row['file']} contains no samples")
    if not np.isfinite(values).all():
        raise ValueError(f"{row['file']} contains missing or non-finite target values")
    if not frame["relative_time"].is_monotonic_increasing:
        raise ValueError(f"{row['file']} has non-monotonic relative time")
    return frame, row


def validate_gray_window(
    gray_start_s: float,
    gray_end_s: float,
    segment_duration_s: float,
) -> None:
    if not np.isfinite([gray_start_s, gray_end_s]).all():
        raise ValueError("Gray-window boundaries must be finite numbers")
    if gray_start_s < 0:
        raise ValueError("GRAY_START_S must be at least 0")
    if gray_end_s <= gray_start_s:
        raise ValueError("GRAY_END_S must be greater than GRAY_START_S")
    if gray_end_s > segment_duration_s:
        raise ValueError(
            f"GRAY_END_S={gray_end_s:g} exceeds segment duration "
            f"{segment_duration_s:.3f} s"
        )


def make_figure(
    frame: pd.DataFrame,
    summary_row: pd.Series,
    gray_start_s: float,
    gray_end_s: float,
) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    relative_time = frame["relative_time"].to_numpy(dtype=float)
    for row_index, (axis, az_column, color) in enumerate(
        zip(axes, AZ_COLUMNS, IMU_COLORS, strict=True)
    ):
        axis.axvspan(
            gray_start_s,
            gray_end_s,
            color=GRAY_SHADE,
            alpha=0.72,
            zorder=0,
        )
        axis.axvline(gray_start_s, color=GRAY_EDGE, linewidth=0.7, linestyle=":", zorder=1)
        axis.axvline(gray_end_s, color=GRAY_EDGE, linewidth=0.7, linestyle=":", zorder=1)
        axis.plot(
            relative_time,
            frame[az_column],
            color=color,
            linewidth=0.6,
            zorder=2,
        )
        axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, linestyle="--", zorder=1)
        axis.set_xlim(float(relative_time[0]), float(relative_time[-1]))
        axis.set_ylabel(f"IMU {row_index + 1}\n{az_column}\n(raw units)", fontsize=7.5)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        axis.tick_params(direction="out", length=2.5, width=0.65)

    gray_midpoint = (gray_start_s + gray_end_s) / 2
    axes[0].text(
        gray_midpoint,
        0.96,
        f"Gray window: {gray_start_s:g}-{gray_end_s:g} s",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=GRAY_EDGE,
    )
    axes[-1].xaxis.set_major_locator(MultipleLocator(10))
    axes[-1].set_xlabel("Relative time from segment start (s)", fontsize=8.5)

    start_world = str(summary_row["start_pc_world_datetime_local"])
    end_world = str(summary_row["end_pc_world_datetime_local"])
    segment_id = str(summary_row["segment_id"])
    figure.suptitle(
        f"{segment_id}: five-IMU z-axis acceleration with a configurable gray window\n"
        f"World time {start_world} - {end_world}; n = {len(frame):,}",
        fontsize=9.5,
        y=1.02,
    )
    figure.align_ylabels(axes)
    return figure


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    frame, summary_row = load_segment(args.input_dir, args.segment_id)
    segment_duration_s = float(frame["relative_time"].iloc[-1])
    validate_gray_window(args.gray_start, args.gray_end, segment_duration_s)
    figure = make_figure(frame, summary_row, args.gray_start, args.gray_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    automatic_name = (
        f"{args.segment_id}_5imu_az_gray_"
        f"{number_token(args.gray_start)}s_{number_token(args.gray_end)}s"
    )
    output_name = args.output_name.strip() or automatic_name
    output_base = args.output_dir / output_name
    svg_path = output_base.with_suffix(".svg")
    pdf_path = output_base.with_suffix(".pdf")
    tiff_path = output_base.with_suffix(".tiff")
    png_path = output_base.with_suffix(".png")

    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("Figure generation complete:")
    print(f"  segment: {args.segment_id}")
    print(f"  gray window: {args.gray_start:g}-{args.gray_end:g} s")
    for path in (png_path, svg_path, pdf_path, tiff_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
