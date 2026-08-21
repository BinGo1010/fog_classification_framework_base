"""Add frame-precision FOG binary labels and plot retained five-IMU z data."""

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
FOG_FILL = "#E69F00"
FOG_EDGE = "#A65F00"


# =============================================================================
# Manual settings for running directly from PyCharm.
# Target format is HH:MM:SS:FF, where FF is 00-59 at FPS = 60.
# Target boundaries are inclusive when assigning y_binary = 1.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "segments_experimental"
OUTPUT_DATA_DIR = (
    PROJECT_ROOT / "dataset" / "All_dataset" / "segments_experimental_labeled"
)
OUTPUT_FIGURE_DIR = (
    PROJECT_ROOT / "dataset" / "All_dataset" / "figures" / "experimental_labeled"
)
FPS = 60
FOG_WINDOWS = {
    "seg000": [
        ("fog1", "18:40:34:25", "18:40:41:30"),
    ],
    "seg001": [
        ("fog1", "18:42:16:20", "18:42:20:55"),
        ("fog2", "18:42:38:20", "18:42:48:20"),
    ],
}
TIME_TICK_SECONDS = 10.0
COMBINED_TIME_TICK_SECONDS = 20.0
WORLD_TIME_FORMAT = "%H:%M:%S"
# =============================================================================


@dataclass(frozen=True)
class FogInterval:
    name: str
    start_timecode: str
    end_timecode: str
    start_world: pd.Timestamp
    end_world: pd.Timestamp
    start_relative_s: float
    end_relative_s: float
    positive_samples: int
    first_positive_world: pd.Timestamp
    last_positive_world: pd.Timestamp


@dataclass
class LabeledSegment:
    segment_id: str
    frame: pd.DataFrame
    start_world_time: pd.Timestamp
    end_world_time: pd.Timestamp
    intervals: list[FogInterval]
    source_file: str
    output_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-data-dir", type=Path, default=OUTPUT_DATA_DIR)
    parser.add_argument("--output-figure-dir", type=Path, default=OUTPUT_FIGURE_DIR)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def parse_world_timecode(value: str, recording_date: pd.Timestamp, fps: int) -> pd.Timestamp:
    if fps <= 0:
        raise ValueError("FPS must be a positive integer")
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid target {value!r}; use HH:MM:SS:FF")
    try:
        hour, minute, second, frame = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid numeric target {value!r}") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        raise ValueError(f"Clock portion out of range in {value!r}")
    if not 0 <= frame < fps:
        raise ValueError(f"Frame must be between 0 and {fps - 1}, got {frame}")
    seconds_since_midnight = hour * 3600 + minute * 60 + second + frame / fps
    return recording_date.normalize() + pd.Timedelta(seconds=seconds_since_midnight)


def format_local_timestamp(value: pd.Timestamp, decimals: int = 6) -> str:
    text = value.strftime("%Y-%m-%d %H:%M:%S.%f")
    return text[: -(6 - decimals)] if decimals < 6 else text


def find_summary(input_dir: Path) -> tuple[Path, pd.DataFrame]:
    paths = sorted(input_dir.glob("*_experimental_segments_summary.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one experimental segment summary CSV, found {len(paths)}"
        )
    return paths[0], pd.read_csv(paths[0])


def label_segment(
    input_dir: Path,
    output_data_dir: Path,
    summary_row: pd.Series,
    segment_id: str,
    configured_windows: list[tuple[str, str, str]],
    fps: int,
) -> tuple[LabeledSegment, pd.Series, list[dict[str, object]]]:
    source_file = str(summary_row["file"])
    frame = pd.read_csv(input_dir / source_file)
    required_columns = {"relative_time", *AZ_COLUMNS}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{source_file} is missing columns: {missing_columns}")
    if len(frame) == 0:
        raise ValueError(f"{source_file} contains no samples")
    numeric_values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{source_file} contains missing or non-finite numeric values")

    relative_time = frame["relative_time"].to_numpy(dtype=float)
    if not pd.Series(relative_time).is_monotonic_increasing:
        raise ValueError(f"{source_file} has non-monotonic relative time")
    start_world = pd.Timestamp(summary_row["start_pc_world_datetime_local"])
    end_world = pd.Timestamp(summary_row["end_pc_world_datetime_local"])
    world_time = start_world + pd.to_timedelta(frame["relative_time"], unit="s")
    endpoint_error_s = abs((world_time.iloc[-1] - end_world).total_seconds())
    if endpoint_error_s > 0.001:
        raise ValueError(
            f"{segment_id} endpoint differs from its summary by {endpoint_error_s:.6f} s"
        )

    frame["y_binary"] = np.zeros(len(frame), dtype=np.int8)
    intervals: list[FogInterval] = []
    report_rows: list[dict[str, object]] = []
    occupied = np.zeros(len(frame), dtype=bool)
    for event_name, start_timecode, end_timecode in configured_windows:
        start_target = parse_world_timecode(start_timecode, start_world, fps)
        end_target = parse_world_timecode(end_timecode, start_world, fps)
        if end_target <= start_target:
            raise ValueError(f"{segment_id} {event_name}: target end must follow target start")
        if start_target < start_world or end_target > end_world + pd.Timedelta(seconds=0.01):
            raise ValueError(f"{segment_id} {event_name}: target lies outside retained data")

        positive_mask = np.asarray((world_time >= start_target) & (world_time <= end_target))
        positive_samples = int(positive_mask.sum())
        if positive_samples == 0:
            raise ValueError(f"{segment_id} {event_name}: target matched no samples")
        if np.any(occupied & positive_mask):
            raise ValueError(f"{segment_id} {event_name}: target overlaps another FOG event")
        occupied |= positive_mask
        frame.loc[positive_mask, "y_binary"] = 1

        positive_positions = np.flatnonzero(positive_mask)
        first_positive_world = world_time.iloc[int(positive_positions[0])]
        last_positive_world = world_time.iloc[int(positive_positions[-1])]
        interval = FogInterval(
            name=event_name,
            start_timecode=start_timecode,
            end_timecode=end_timecode,
            start_world=start_target,
            end_world=end_target,
            start_relative_s=float((start_target - start_world).total_seconds()),
            end_relative_s=float((end_target - start_world).total_seconds()),
            positive_samples=positive_samples,
            first_positive_world=first_positive_world,
            last_positive_world=last_positive_world,
        )
        intervals.append(interval)
        report_rows.append(
            {
                "segment_id": segment_id,
                "event": event_name,
                "target_start_timecode": start_timecode,
                "target_end_timecode": end_timecode,
                "target_start_world": format_local_timestamp(start_target),
                "target_end_world": format_local_timestamp(end_target),
                "first_positive_sample_world": format_local_timestamp(first_positive_world),
                "last_positive_sample_world": format_local_timestamp(last_positive_world),
                "positive_samples": positive_samples,
            }
        )

    if not set(frame["y_binary"].unique()).issubset({0, 1}):
        raise ValueError(f"{segment_id} produced a non-binary y_binary column")
    expected_positive = sum(interval.positive_samples for interval in intervals)
    actual_positive = int(frame["y_binary"].sum())
    if actual_positive != expected_positive:
        raise ValueError(
            f"{segment_id} positive-count mismatch: {actual_positive} != {expected_positive}"
        )

    output_file = f"{Path(source_file).stem}_y_binary.csv"
    frame.to_csv(output_data_dir / output_file, index=False, float_format="%.9f")
    output_summary_row = summary_row.copy()
    output_summary_row["file"] = output_file
    output_summary_row["label_column"] = "y_binary"
    output_summary_row["y_binary_positive_samples"] = actual_positive
    output_summary_row["y_binary_negative_samples"] = len(frame) - actual_positive
    output_summary_row["y_binary_positive_fraction"] = actual_positive / len(frame)

    labeled = LabeledSegment(
        segment_id=segment_id,
        frame=frame,
        start_world_time=start_world,
        end_world_time=end_world,
        intervals=intervals,
        source_file=source_file,
        output_file=output_file,
    )
    return labeled, output_summary_row, report_rows


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
    date_label = start_world_time.strftime("%Y-%m-%d")
    world_axis.set_xlabel(
        f"{label_prefix} on {date_label} (Asia/Shanghai)", fontsize=8.2, labelpad=3
    )
    world_axis.tick_params(direction="out", length=2.5, width=0.65, pad=2)


def shade_fog_intervals(axis: plt.Axes, intervals: list[FogInterval], add_labels: bool) -> None:
    for interval in intervals:
        axis.axvspan(
            interval.start_relative_s,
            interval.end_relative_s,
            color=FOG_FILL,
            alpha=0.20,
            zorder=0,
        )
        axis.axvline(
            interval.start_relative_s,
            color=FOG_EDGE,
            linewidth=0.65,
            linestyle=":",
            zorder=1,
        )
        axis.axvline(
            interval.end_relative_s,
            color=FOG_EDGE,
            linewidth=0.65,
            linestyle=":",
            zorder=1,
        )
        if add_labels:
            axis.text(
                (interval.start_relative_s + interval.end_relative_s) / 2,
                0.97,
                f"{interval.name}\ny=1",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.8,
                fontweight="bold",
                color=FOG_EDGE,
            )


def draw_individual_axes(axes: np.ndarray, segment: LabeledSegment) -> None:
    relative_time = segment.frame["relative_time"].to_numpy(dtype=float)
    plot_end_s = float(relative_time[-1])
    for row_index, (axis, column, color) in enumerate(
        zip(axes, AZ_COLUMNS, IMU_COLORS, strict=True)
    ):
        shade_fog_intervals(axis, segment.intervals, add_labels=row_index == 0)
        axis.plot(relative_time, segment.frame[column], color=color, linewidth=0.55, zorder=2)
        axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, linestyle="--", zorder=1)
        axis.set_xlim(0.0, plot_end_s)
        axis.set_ylabel(f"IMU {row_index + 1}\n{column}\n(raw units)", fontsize=7.5)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        axis.tick_params(direction="out", length=2.5, width=0.65)
    axes[-1].xaxis.set_major_locator(MultipleLocator(TIME_TICK_SECONDS))
    axes[-1].set_xlabel("Retained experimental time from new segment start (s)", fontsize=8.5)
    add_world_axis(axes[0], segment.start_world_time, plot_end_s, TIME_TICK_SECONDS)


def make_individual_figure(segment: LabeledSegment, fps: int) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    draw_individual_axes(axes, segment)
    positive_samples = int(segment.frame["y_binary"].sum())
    figure.suptitle(
        f"{segment.segment_id}: five-IMU z-axis acceleration with FOG labels\n"
        f"y_binary = 1: n = {positive_samples:,} / {len(segment.frame):,}; "
        f"target precision = 1/{fps} s",
        fontsize=9.5,
        y=1.04,
    )
    figure.align_ylabels(axes)
    return figure


def make_combined_figure(segments: list[LabeledSegment], fps: int) -> plt.Figure:
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
            shade_fog_intervals(axis, segment.intervals, add_labels=row_index == 0)
            axis.plot(relative_time, segment.frame[column], color=color, linewidth=0.5, zorder=2)
            axis.axhline(0.0, color="#A8A8A8", linewidth=0.5, linestyle="--", zorder=1)
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
    counts = " | ".join(
        f"{segment.segment_id}: {int(segment.frame['y_binary'].sum()):,}/{len(segment.frame):,} positive"
        for segment in segments
    )
    figure.suptitle(
        "Five-IMU z-axis acceleration with frame-precision FOG regions\n"
        f"{counts}; target precision = 1/{fps} s",
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

    labeled_segments: list[LabeledSegment] = []
    output_summary_rows: list[pd.Series] = []
    label_report_rows: list[dict[str, object]] = []
    for segment_id, configured_windows in FOG_WINDOWS.items():
        matched = summary.loc[summary["segment_id"] == segment_id]
        if len(matched) != 1:
            raise ValueError(f"Expected one summary row for {segment_id}, found {len(matched)}")
        segment, output_summary_row, report_rows = label_segment(
            args.input_dir,
            args.output_data_dir,
            matched.iloc[0],
            segment_id,
            configured_windows,
            args.fps,
        )
        labeled_segments.append(segment)
        output_summary_rows.append(output_summary_row)
        label_report_rows.extend(report_rows)

    subject_id = str(summary.iloc[0]["subject_id"])
    output_summary_path = (
        args.output_data_dir / f"{subject_id}_experimental_y_binary_segments_summary.csv"
    )
    pd.DataFrame(output_summary_rows).to_csv(output_summary_path, index=False)
    report_path = args.output_data_dir / f"{subject_id}_y_binary_label_report.csv"
    pd.DataFrame(label_report_rows).to_csv(report_path, index=False)

    for segment in labeled_segments:
        individual = make_individual_figure(segment, args.fps)
        save_figure(
            individual,
            args.output_figure_dir / f"{segment.segment_id}_experimental_5imu_az_y_binary",
        )
    combined = make_combined_figure(labeled_segments, args.fps)
    combined_base = args.output_figure_dir / "seg000_seg001_experimental_5imu_az_y_binary"
    save_figure(combined, combined_base)

    print("FOG binary-label outputs generated; source files were not changed.")
    print(f"  source summary: {summary_path}")
    print(f"  output summary: {output_summary_path}")
    print(f"  label report: {report_path}")
    for segment in labeled_segments:
        print(
            f"  {segment.segment_id}: y_binary=1 for "
            f"{int(segment.frame['y_binary'].sum()):,}/{len(segment.frame):,} samples"
        )
        for interval in segment.intervals:
            print(
                f"    {interval.name}: {interval.start_timecode}-{interval.end_timecode}, "
                f"{interval.positive_samples:,} samples, actual matched "
                f"{format_local_timestamp(interval.first_positive_world)} - "
                f"{format_local_timestamp(interval.last_positive_world)}"
            )
    print(f"  combined figure: {combined_base}.png/.svg/.pdf/.tiff")


if __name__ == "__main__":
    main()
