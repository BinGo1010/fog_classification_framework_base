"""Remove configured non-experimental intervals and plot retained five-IMU z data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


# =============================================================================
# Manual settings for running directly from PyCharm.
# Timecode format is MM:SS:FF, where FF is 00-59 at FPS = 60.
# Both boundaries are removed. Source CSV files are preserved unchanged.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "segments"
OUTPUT_DATA_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "segments_experimental"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "figures" / "experimental_only"
FPS = 60
EXCLUSION_WINDOWS = {
    "seg000": ("00:00:00", "00:22:00"),
    "seg001": ("01:27:38", "02:42:38"),
}
TIME_TICK_SECONDS = 10.0
COMBINED_TIME_TICK_SECONDS = 20.0
WORLD_TIME_FORMAT = "%H:%M:%S"
# =============================================================================


@dataclass
class RetainedSegment:
    segment_id: str
    frame: pd.DataFrame
    source_relative_time: np.ndarray
    start_world_time: pd.Timestamp
    end_world_time: pd.Timestamp
    exclusion_start_tc: str
    exclusion_end_tc: str
    exclusion_start_s: float
    exclusion_end_s: float
    input_samples: int
    excluded_samples: int
    source_file: str
    output_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-data-dir", type=Path, default=OUTPUT_DATA_DIR)
    parser.add_argument("--output-figure-dir", type=Path, default=OUTPUT_FIGURE_DIR)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def parse_timecode(value: str, fps: int) -> tuple[int, float]:
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


def find_summary(input_dir: Path) -> tuple[Path, pd.DataFrame]:
    paths = sorted(input_dir.glob("*_segments_summary.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one segment summary CSV, found {len(paths)}")
    summary = pd.read_csv(paths[0])
    return paths[0], summary


def format_local_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def trim_segment(
    input_dir: Path,
    output_data_dir: Path,
    summary_row: pd.Series,
    segment_id: str,
    exclusion: tuple[str, str],
    fps: int,
) -> tuple[RetainedSegment, pd.Series, dict[str, object]]:
    source_file = str(summary_row["file"])
    source_path = input_dir / source_file
    frame = pd.read_csv(source_path)
    required_columns = {"time", "relative_time", *AZ_COLUMNS}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{source_file} is missing columns: {missing_columns}")
    if len(frame) == 0:
        raise ValueError(f"{source_file} contains no samples")

    numeric_values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{source_file} contains missing or non-finite numeric values")
    source_relative = frame["relative_time"].to_numpy(dtype=float)
    if not pd.Series(source_relative).is_monotonic_increasing:
        raise ValueError(f"{source_file} has non-monotonic relative time")

    exclusion_start_tc, exclusion_end_tc = exclusion
    _, exclusion_start_s = parse_timecode(exclusion_start_tc, fps)
    _, exclusion_end_s = parse_timecode(exclusion_end_tc, fps)
    if exclusion_end_s <= exclusion_start_s:
        raise ValueError(f"Invalid exclusion for {segment_id}: end must follow start")
    half_frame_s = 0.5 / fps
    if exclusion_start_s < -half_frame_s or exclusion_start_s > source_relative[-1] + half_frame_s:
        raise ValueError(f"Exclusion start for {segment_id} is outside the segment")
    if exclusion_end_s > source_relative[-1] + half_frame_s:
        raise ValueError(
            f"Exclusion end for {segment_id} exceeds the segment by more than half a frame"
        )

    excluded_mask = (source_relative >= exclusion_start_s) & (
        source_relative <= exclusion_end_s
    )
    retained_mask = ~excluded_mask
    excluded_samples = int(excluded_mask.sum())
    if excluded_samples == 0:
        raise ValueError(f"Exclusion for {segment_id} removed no samples")
    if not retained_mask.any():
        raise ValueError(f"Exclusion for {segment_id} removed the whole segment")

    retained_source_relative = source_relative[retained_mask]
    retained_frame = frame.loc[retained_mask].copy().reset_index(drop=True)
    first_source_relative_s = float(retained_source_relative[0])
    last_source_relative_s = float(retained_source_relative[-1])
    retained_frame["relative_time"] = retained_source_relative - first_source_relative_s

    source_deltas = np.diff(source_relative)
    positive_deltas = source_deltas[source_deltas > 0]
    if len(positive_deltas) == 0:
        raise ValueError(f"Cannot determine sample period for {segment_id}")
    sample_period_s = float(np.median(positive_deltas))
    retained_deltas = np.diff(retained_frame["relative_time"].to_numpy(dtype=float))
    if len(retained_deltas) and float(retained_deltas.max()) > 1.5 * sample_period_s:
        raise ValueError(
            f"Trimming left an internal time gap in {segment_id}; split output is required"
        )

    source_start_world = pd.Timestamp(summary_row["start_pc_world_datetime_local"])
    retained_start_world = source_start_world + pd.Timedelta(seconds=first_source_relative_s)
    retained_end_world = source_start_world + pd.Timedelta(seconds=last_source_relative_s)
    retained_duration_s = float(retained_frame["relative_time"].iloc[-1] + sample_period_s)

    output_file = f"{Path(source_file).stem}_experimental.csv"
    output_path = output_data_dir / output_file
    retained_frame.to_csv(output_path, index=False, float_format="%.9f")

    output_summary_row = summary_row.copy()
    output_summary_row["file"] = output_file
    output_summary_row["start_time"] = float(retained_frame["time"].iloc[0])
    output_summary_row["end_time"] = float(retained_frame["time"].iloc[-1])
    output_summary_row["samples"] = len(retained_frame)
    output_summary_row["duration_s"] = retained_duration_s
    output_summary_row["duration_s_by_time_span"] = retained_duration_s
    output_summary_row["start_pc_world_timestamp"] = (
        float(summary_row["start_pc_world_timestamp"]) + first_source_relative_s
    )
    output_summary_row["end_pc_world_timestamp"] = (
        float(summary_row["start_pc_world_timestamp"]) + last_source_relative_s
    )
    output_summary_row["start_pc_world_datetime_local"] = format_local_timestamp(
        retained_start_world
    )
    output_summary_row["end_pc_world_datetime_local"] = format_local_timestamp(
        retained_end_world
    )
    output_summary_row["source_file"] = source_file
    output_summary_row["source_samples"] = len(frame)
    output_summary_row["excluded_samples"] = excluded_samples
    output_summary_row["exclusion_start_timecode"] = exclusion_start_tc
    output_summary_row["exclusion_end_timecode"] = exclusion_end_tc
    output_summary_row["fps"] = fps

    record = RetainedSegment(
        segment_id=segment_id,
        frame=retained_frame,
        source_relative_time=retained_source_relative,
        start_world_time=retained_start_world,
        end_world_time=retained_end_world,
        exclusion_start_tc=exclusion_start_tc,
        exclusion_end_tc=exclusion_end_tc,
        exclusion_start_s=exclusion_start_s,
        exclusion_end_s=exclusion_end_s,
        input_samples=len(frame),
        excluded_samples=excluded_samples,
        source_file=source_file,
        output_file=output_file,
    )
    report_row: dict[str, object] = {
        "segment_id": segment_id,
        "source_file": source_file,
        "output_file": output_file,
        "exclusion_start_timecode": exclusion_start_tc,
        "exclusion_end_timecode": exclusion_end_tc,
        "exclusion_start_s": exclusion_start_s,
        "exclusion_end_s": exclusion_end_s,
        "input_samples": len(frame),
        "excluded_samples": excluded_samples,
        "output_samples": len(retained_frame),
        "first_retained_source_relative_s": first_source_relative_s,
        "last_retained_source_relative_s": last_source_relative_s,
        "retained_duration_s": retained_duration_s,
        "retained_world_start": format_local_timestamp(retained_start_world),
        "retained_world_end": format_local_timestamp(retained_end_world),
    }
    return record, output_summary_row, report_row


def add_world_axis(
    axis: plt.Axes,
    start_world_time: pd.Timestamp,
    plot_end_s: float,
    tick_seconds: float,
    label_prefix: str = "World time",
) -> None:
    ticks = np.arange(0.0, plot_end_s + 1e-9, tick_seconds)
    labels = [
        (start_world_time + pd.Timedelta(seconds=float(tick))).strftime(WORLD_TIME_FORMAT)
        for tick in ticks
    ]
    world_axis = axis.secondary_xaxis("top")
    world_axis.set_xticks(ticks)
    world_axis.set_xticklabels(labels, fontsize=7.5)
    recording_date = start_world_time.strftime("%Y-%m-%d")
    world_axis.set_xlabel(
        f"{label_prefix} on {recording_date} (Asia/Shanghai)", fontsize=8.2, labelpad=3
    )
    world_axis.tick_params(direction="out", length=2.5, width=0.65, pad=2)


def draw_segment_axes(
    axes: np.ndarray,
    segment: RetainedSegment,
    tick_seconds: float,
) -> None:
    relative_time = segment.frame["relative_time"].to_numpy(dtype=float)
    plot_end_s = float(relative_time[-1])
    for row_index, (axis, column, color) in enumerate(
        zip(axes, AZ_COLUMNS, IMU_COLORS, strict=True)
    ):
        axis.plot(relative_time, segment.frame[column], color=color, linewidth=0.55)
        axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, linestyle="--", zorder=0)
        axis.set_xlim(0.0, plot_end_s)
        axis.set_ylabel(f"IMU {row_index + 1}\n{column}\n(raw units)", fontsize=7.5)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        axis.tick_params(direction="out", length=2.5, width=0.65)
    axes[-1].xaxis.set_major_locator(MultipleLocator(tick_seconds))
    axes[-1].set_xlabel("Retained experimental time from new segment start (s)", fontsize=8.5)
    add_world_axis(axes[0], segment.start_world_time, plot_end_s, tick_seconds)


def make_individual_figure(segment: RetainedSegment, fps: int) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    draw_segment_axes(axes, segment, TIME_TICK_SECONDS)
    output_samples = len(segment.frame)
    figure.suptitle(
        f"{segment.segment_id}: retained five-IMU z-axis acceleration after trimming\n"
        f"removed {segment.exclusion_start_tc}-{segment.exclusion_end_tc} at {fps} fps; "
        f"n = {output_samples:,} retained / {segment.input_samples:,} source",
        fontsize=9.5,
        y=1.04,
    )
    figure.align_ylabels(axes)
    return figure


def make_combined_figure(segments: list[RetainedSegment], fps: int) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(7.2, 8.6),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    for column_index, segment in enumerate(segments):
        relative_time = segment.frame["relative_time"].to_numpy(dtype=float)
        plot_end_s = float(relative_time[-1])
        for row_index, (column, color) in enumerate(zip(AZ_COLUMNS, IMU_COLORS, strict=True)):
            axis = axes[row_index, column_index]
            axis.plot(relative_time, segment.frame[column], color=color, linewidth=0.5)
            axis.axhline(0.0, color="#A8A8A8", linewidth=0.5, linestyle="--", zorder=0)
            axis.set_xlim(0.0, plot_end_s)
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.4)
            axis.tick_params(direction="out", length=2.3, width=0.6)
        axes[-1, column_index].xaxis.set_major_locator(
            MultipleLocator(COMBINED_TIME_TICK_SECONDS)
        )
        axes[-1, column_index].set_xlabel(
            f"{segment.segment_id}: retained experimental time (s)", fontsize=7.8
        )
        add_world_axis(
            axes[0, column_index],
            segment.start_world_time,
            plot_end_s,
            COMBINED_TIME_TICK_SECONDS,
            label_prefix=f"{segment.segment_id} world time",
        )

    for row_index, column in enumerate(AZ_COLUMNS):
        axes[row_index, 0].set_ylabel(
            f"IMU {row_index + 1}\n{column} (raw units)", fontsize=7.5
        )
    removal_text = " | ".join(
        f"{segment.segment_id} removed {segment.exclusion_start_tc}-{segment.exclusion_end_tc}"
        for segment in segments
    )
    figure.suptitle(
        "Retained five-IMU z-axis acceleration after non-experimental trimming\n"
        f"{removal_text} at {fps} fps",
        fontsize=9.5,
        y=1.04,
    )
    figure.align_ylabels(axes[:, 0])
    return figure


def save_figure(figure: plt.Figure, output_base: Path) -> None:
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_figure_dir.mkdir(parents=True, exist_ok=True)

    summary_path, summary = find_summary(args.input_dir)
    retained_segments: list[RetainedSegment] = []
    output_summary_rows: list[pd.Series] = []
    report_rows: list[dict[str, object]] = []
    for segment_id, exclusion in EXCLUSION_WINDOWS.items():
        matched = summary.loc[summary["segment_id"] == segment_id]
        if len(matched) != 1:
            raise ValueError(f"Expected one summary row for {segment_id}, found {len(matched)}")
        segment, summary_row, report_row = trim_segment(
            args.input_dir,
            args.output_data_dir,
            matched.iloc[0],
            segment_id,
            exclusion,
            args.fps,
        )
        retained_segments.append(segment)
        output_summary_rows.append(summary_row)
        report_rows.append(report_row)

    subject_id = str(summary.iloc[0]["subject_id"])
    output_summary_path = args.output_data_dir / f"{subject_id}_experimental_segments_summary.csv"
    pd.DataFrame(output_summary_rows).to_csv(output_summary_path, index=False)
    report_path = args.output_data_dir / f"{subject_id}_trimming_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False, float_format="%.9f")

    for segment in retained_segments:
        individual = make_individual_figure(segment, args.fps)
        save_figure(
            individual,
            args.output_figure_dir / f"{segment.segment_id}_experimental_5imu_az_full",
        )
    combined = make_combined_figure(retained_segments, args.fps)
    combined_base = args.output_figure_dir / "seg000_seg001_experimental_5imu_az_full"
    save_figure(combined, combined_base)

    print("Experimental-only outputs generated; source files were not changed.")
    print(f"  source summary: {summary_path}")
    print(f"  output summary: {output_summary_path}")
    print(f"  trimming report: {report_path}")
    for row in report_rows:
        print(
            f"  {row['segment_id']}: {row['input_samples']:,} -> "
            f"{row['output_samples']:,} samples "
            f"({row['excluded_samples']:,} excluded); "
            f"world time {row['retained_world_start']} - {row['retained_world_end']}"
        )
    print(f"  combined figure: {combined_base}.png/.svg/.pdf/.tiff")


if __name__ == "__main__":
    main()
