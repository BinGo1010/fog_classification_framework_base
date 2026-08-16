"""Plot five IMU z-acceleration channels around a target world-time interval."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
TARGET_SHADE = "#DDEAF6"
NO_DATA_SHADE = "#E2E2E2"


# =============================================================================
# Manual settings for running directly from PyCharm.
# Edit only this block for the usual workflow; no command-line arguments needed.
# Timecode format: HH:MM:SS:FF, where FF is 00-59 when FPS = 60.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "segments"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "figures"
OUTPUT_BASENAME = ""  # Empty: automatic name; example: "seg001_freezing_1"
SEGMENT_ID = "seg001"
TARGET_START = "18:42:38:20"
TARGET_END = "18:42:48:20"
FPS = 60
CONTEXT_SECONDS = 5.0
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing segment CSV files and their summary.",
    )
    parser.add_argument("--segment-id", default=SEGMENT_ID)
    parser.add_argument(
        "--target-start",
        default=TARGET_START,
        help="Local timecode as HH:MM:SS or HH:MM:SS:FF.",
    )
    parser.add_argument(
        "--target-end",
        default=TARGET_END,
        help="Local timecode as HH:MM:SS or HH:MM:SS:FF.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second for FF.")
    parser.add_argument("--context-seconds", type=float, default=CONTEXT_SECONDS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for SVG, PDF, TIFF, and PNG outputs.",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_BASENAME,
        help="Custom output basename without an extension; empty uses an automatic name.",
    )
    return parser.parse_args()


def parse_timecode(recording_date: str, value: str, fps: int) -> pd.Timestamp:
    if fps <= 0:
        raise ValueError("FPS must be a positive integer")
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"Invalid timecode {value!r}; use HH:MM:SS or HH:MM:SS:FF")
    try:
        hour, minute, second = (int(part) for part in parts[:3])
        frame = int(parts[3]) if len(parts) == 4 else 0
    except ValueError as error:
        raise ValueError(f"Invalid numeric timecode {value!r}") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        raise ValueError(f"Clock portion out of range in {value!r}")
    if not 0 <= frame < fps:
        raise ValueError(f"Frame must be between 0 and {fps - 1}, got {frame}")

    base = pd.Timestamp(f"{recording_date} {hour:02d}:{minute:02d}:{second:02d}")
    frame_offset_ns = round(frame * 1_000_000_000 / fps)
    return base + pd.Timedelta(frame_offset_ns, unit="ns")


def format_timecode(value: pd.Timestamp, fps: int) -> str:
    whole_second = value.floor("s")
    fraction_ns = value.value - whole_second.value
    frame = round(fraction_ns * fps / 1_000_000_000)
    if frame >= fps:
        whole_second += pd.Timedelta(seconds=1)
        frame = 0
    return f"{whole_second:%H:%M:%S}:{frame:02d}"


def load_window(
    input_dir: Path,
    segment_id: str,
    target_start_text: str,
    target_end_text: str,
    context_seconds: float,
    fps: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    summary_paths = sorted(input_dir.glob("*_segments_summary.csv"))
    if len(summary_paths) != 1:
        raise ValueError(f"Expected exactly one segment summary CSV, found {len(summary_paths)}")
    summary = pd.read_csv(summary_paths[0])
    summary_row = summary.loc[summary["segment_id"] == segment_id]
    if len(summary_row) != 1:
        raise ValueError(f"Expected one summary row for {segment_id}, found {len(summary_row)}")
    row = summary_row.iloc[0]

    segment_start = pd.Timestamp(row["start_pc_world_datetime_local"])
    segment_end = pd.Timestamp(row["end_pc_world_datetime_local"])
    recording_date = segment_start.strftime("%Y-%m-%d")
    target_start = parse_timecode(recording_date, target_start_text, fps)
    target_end = parse_timecode(recording_date, target_end_text, fps)
    if target_end <= target_start:
        raise ValueError("Target end must be later than target start")
    if context_seconds < 0:
        raise ValueError("Context seconds must be non-negative")
    requested_start = target_start - pd.Timedelta(seconds=context_seconds)
    requested_end = target_end + pd.Timedelta(seconds=context_seconds)

    segment_path = input_dir / str(row["file"])
    usecols = ["relative_time", *AZ_COLUMNS]
    frame = pd.read_csv(segment_path, usecols=usecols)
    values = frame[usecols].to_numpy(dtype=float)
    if len(frame) == 0:
        raise ValueError(f"{segment_path.name} contains no samples")
    if not np.isfinite(values).all():
        raise ValueError(f"{segment_path.name} contains missing or non-finite target values")
    if not frame["relative_time"].is_monotonic_increasing:
        raise ValueError(f"{segment_path.name} has non-monotonic relative time")

    elapsed = frame["relative_time"] - frame["relative_time"].iloc[0]
    frame["world_time"] = segment_start + pd.to_timedelta(elapsed, unit="s")
    endpoint_error_s = abs((frame["world_time"].iloc[-1] - segment_end).total_seconds())
    if endpoint_error_s > 0.02:
        raise ValueError(
            f"{segment_id} world-time endpoint differs from the summary by "
            f"{endpoint_error_s:.6f} s"
        )

    window_mask = frame["world_time"].between(requested_start, requested_end, inclusive="both")
    window = frame.loc[window_mask].copy()
    if window.empty:
        raise ValueError("The requested world-time window contains no segment samples")
    return window, target_start, target_end, requested_start, requested_end, segment_end


def make_figure(
    window: pd.DataFrame,
    segment_id: str,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    segment_end: pd.Timestamp,
    fps: int,
) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    no_data_start = max(segment_end, requested_start)
    has_missing_tail = no_data_start < requested_end
    for row_index, (axis, az_column, color) in enumerate(
        zip(axes, AZ_COLUMNS, IMU_COLORS, strict=True)
    ):
        axis.axvspan(target_start, target_end, color=TARGET_SHADE, alpha=0.8, zorder=0)
        axis.axvline(target_start, color="#0F4D92", linewidth=0.7, linestyle=":", zorder=1)
        axis.axvline(target_end, color="#0F4D92", linewidth=0.7, linestyle=":", zorder=1)
        if has_missing_tail:
            axis.axvspan(no_data_start, requested_end, color=NO_DATA_SHADE, alpha=0.85, zorder=0)
        axis.plot(
            window["world_time"],
            window[az_column],
            color=color,
            linewidth=0.8,
            zorder=2,
        )
        axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, linestyle="--", zorder=1)
        axis.set_xlim(requested_start, requested_end)
        axis.set_ylabel(f"IMU {row_index + 1}\n{az_column}\n(raw units)", fontsize=7.5)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        axis.tick_params(direction="out", length=2.5, width=0.65)

    target_midpoint = target_start + (target_end - target_start) / 2
    target_start_label = format_timecode(target_start, fps)
    target_end_label = format_timecode(target_end, fps)
    axes[0].text(
        target_midpoint,
        0.96,
        f"Target {target_start_label}-{target_end_label}",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#0F4D92",
    )
    if has_missing_tail:
        no_data_midpoint = no_data_start + (requested_end - no_data_start) / 2
        axes[0].text(
            no_data_midpoint,
            0.96,
            f"No {segment_id} data",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#606060",
        )

    axes[-1].xaxis.set_major_locator(mdates.SecondLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel(
        f"World time (Asia/Shanghai); target timecode HH:MM:SS:FF at {fps} fps",
        fontsize=8.5,
    )

    available_start = window["world_time"].iloc[0]
    available_end = window["world_time"].iloc[-1]
    available_start_label = available_start.strftime("%H:%M:%S.%f")[:-3]
    available_end_label = available_end.strftime("%H:%M:%S.%f")[:-3]
    requested_start_label = format_timecode(requested_start, fps)
    requested_end_label = format_timecode(requested_end, fps)
    target_count = int(
        window["world_time"].between(target_start, target_end, inclusive="both").sum()
    )
    figure.suptitle(
        f"{segment_id}: five-IMU z-axis acceleration around the target interval\n"
        f"Requested {requested_start_label}-{requested_end_label}; "
        f"available {available_start_label}-{available_end_label}; "
        f"window n = {len(window):,}, target n = {target_count:,}",
        fontsize=9.5,
        y=1.02,
    )
    figure.align_ylabels(axes)
    return figure


def main() -> None:
    args = parse_args()
    window, target_start, target_end, requested_start, requested_end, segment_end = load_window(
        args.input_dir,
        args.segment_id,
        args.target_start,
        args.target_end,
        args.context_seconds,
        args.fps,
    )
    figure = make_figure(
        window,
        args.segment_id,
        target_start,
        target_end,
        requested_start,
        requested_end,
        segment_end,
        args.fps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_token = format_timecode(target_start, args.fps).replace(":", "")
    end_token = format_timecode(target_end, args.fps).replace(":", "")
    automatic_name = (
        f"{args.segment_id}_5imu_az_target_{start_token}_{end_token}_world_time"
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
    for path in (png_path, svg_path, pdf_path, tiff_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
