"""Plot five IMU z-acceleration channels with a 60-fps gray time window."""

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
# Gray-window timecode format: MM:SS:FF, where FF is 00-59 at FPS = 60.
# The defaults below mark the final 60 seconds of seg001 on the 60-fps grid.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "segments"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "figures"
OUTPUT_BASENAME = ""  # Empty: automatic name; example: "seg001_last_60s"
SEGMENT_ID = "seg001"
FPS = 60
GRAY_START_TC = "01:27:38"
GRAY_END_TC = "02:42:38"
TIME_TICK_SECONDS = 20.0
WORLD_TIME_FORMAT = "%H:%M:%S"
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-name", default=OUTPUT_BASENAME)
    parser.add_argument("--segment-id", default=SEGMENT_ID)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--gray-start", default=GRAY_START_TC)
    parser.add_argument("--gray-end", default=GRAY_END_TC)
    return parser.parse_args()


def parse_relative_timecode(value: str, fps: int) -> tuple[int, float]:
    if fps <= 0:
        raise ValueError("FPS must be a positive integer")
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timecode {value!r}; use MM:SS:FF")
    try:
        minute, second, frame = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid numeric timecode {value!r}") from error
    if minute < 0 or not 0 <= second <= 59:
        raise ValueError(f"Minute/second portion out of range in {value!r}")
    if not 0 <= frame < fps:
        raise ValueError(f"Frame must be between 0 and {fps - 1}, got {frame}")
    total_frames = (minute * 60 + second) * fps + frame
    return total_frames, total_frames / fps


def format_relative_timecode(total_frames: int, fps: int) -> str:
    whole_seconds, frame = divmod(total_frames, fps)
    minute, second = divmod(whole_seconds, 60)
    return f"{minute:02d}:{second:02d}:{frame:02d}"


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
    fps: int,
) -> None:
    if gray_start_s < 0:
        raise ValueError("Gray-window start must be at least 00:00:00")
    if gray_end_s <= gray_start_s:
        raise ValueError("Gray-window end must be later than its start")
    half_frame_s = 0.5 / fps
    if gray_end_s > segment_duration_s + half_frame_s:
        raise ValueError(
            f"Gray-window end {gray_end_s:.6f} s exceeds segment duration "
            f"{segment_duration_s:.6f} s by more than half a frame"
        )


def make_figure(
    frame: pd.DataFrame,
    summary_row: pd.Series,
    gray_start_frames: int,
    gray_end_frames: int,
    gray_start_s: float,
    gray_end_s: float,
    fps: int,
) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    relative_time = frame["relative_time"].to_numpy(dtype=float)
    plot_end_s = max(float(relative_time[-1]), gray_end_s)
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
            linewidth=0.55,
            zorder=2,
        )
        axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, linestyle="--", zorder=1)
        axis.set_xlim(0.0, plot_end_s)
        axis.set_ylabel(f"IMU {row_index + 1}\n{az_column}\n(raw units)", fontsize=7.5)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        axis.tick_params(direction="out", length=2.5, width=0.65)

    gray_start_label = format_relative_timecode(gray_start_frames, fps)
    gray_end_label = format_relative_timecode(gray_end_frames, fps)
    gray_midpoint = (gray_start_s + gray_end_s) / 2
    axes[0].text(
        gray_midpoint,
        0.96,
        f"Gray window: {gray_start_label}-{gray_end_label} ({gray_end_s - gray_start_s:.3f} s)",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=GRAY_EDGE,
    )
    axes[-1].xaxis.set_major_locator(MultipleLocator(TIME_TICK_SECONDS))
    axes[-1].set_xlabel(
        f"Relative time from segment start (s); gray boundaries use MM:SS:FF at {fps} fps",
        fontsize=8.5,
    )

    start_world_timestamp = pd.Timestamp(summary_row["start_pc_world_datetime_local"])
    aligned_ticks = np.arange(0.0, plot_end_s + 1e-9, TIME_TICK_SECONDS)
    world_tick_labels = [
        (start_world_timestamp + pd.Timedelta(seconds=float(tick))).strftime(
            WORLD_TIME_FORMAT
        )
        for tick in aligned_ticks
    ]
    world_axis = axes[0].secondary_xaxis("top")
    world_axis.set_xticks(aligned_ticks)
    world_axis.set_xticklabels(world_tick_labels, fontsize=7.5)
    recording_date = start_world_timestamp.strftime("%Y-%m-%d")
    world_axis.set_xlabel(
        f"World time on {recording_date} (Asia/Shanghai, HH:MM:SS)",
        fontsize=8.5,
        labelpad=3,
    )
    world_axis.tick_params(direction="out", length=2.5, width=0.65, pad=2)

    gray_sample_count = int(
        frame["relative_time"].between(gray_start_s, gray_end_s, inclusive="both").sum()
    )
    segment_id = str(summary_row["segment_id"])
    figure.suptitle(
        f"{segment_id}: five-IMU z-axis acceleration with a frame-precision gray window\n"
        f"total n = {len(frame):,}, gray-window n = {gray_sample_count:,}",
        fontsize=9.5,
        y=1.04,
    )
    figure.align_ylabels(axes)
    return figure


def main() -> None:
    args = parse_args()
    gray_start_frames, gray_start_s = parse_relative_timecode(args.gray_start, args.fps)
    gray_end_frames, gray_end_s = parse_relative_timecode(args.gray_end, args.fps)
    frame, summary_row = load_segment(args.input_dir, args.segment_id)
    segment_duration_s = float(frame["relative_time"].iloc[-1])
    validate_gray_window(gray_start_s, gray_end_s, segment_duration_s, args.fps)
    figure = make_figure(
        frame,
        summary_row,
        gray_start_frames,
        gray_end_frames,
        gray_start_s,
        gray_end_s,
        args.fps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_token = format_relative_timecode(gray_start_frames, args.fps).replace(":", "")
    end_token = format_relative_timecode(gray_end_frames, args.fps).replace(":", "")
    automatic_name = (
        f"{args.segment_id}_5imu_az_gray_{start_token}_{end_token}_{args.fps}fps"
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
    print(f"  gray window: {args.gray_start}-{args.gray_end} at {args.fps} fps")
    for path in (png_path, svg_path, pdf_path, tiff_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
