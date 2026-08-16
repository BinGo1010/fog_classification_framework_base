"""Plot z-axis acceleration from all five IMUs for two time segments."""

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
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["path.simplify"] = False


AZ_COLUMNS = tuple(f"imu{index}_az" for index in range(1, 6))
IMU_COLORS = ("#0F4D92", "#3775BA", "#42949E", "#9A4D8E", "#606060")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/All_dataset/segments"),
        help="Directory containing the two segment CSV files and their summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/All_dataset/figures"),
        help="Directory for SVG, PDF, TIFF, and PNG outputs.",
    )
    return parser.parse_args()


def load_segments(
    input_dir: Path,
) -> list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]]:
    segment_paths = sorted(input_dir.glob("*_seg[0-9][0-9][0-9].csv"))
    if len(segment_paths) != 2:
        raise ValueError(f"Expected exactly two segment CSV files, found {len(segment_paths)}")

    summary_paths = sorted(input_dir.glob("*_segments_summary.csv"))
    if len(summary_paths) != 1:
        raise ValueError(f"Expected exactly one segment summary CSV, found {len(summary_paths)}")
    summary = pd.read_csv(summary_paths[0])

    usecols = ["relative_time", *AZ_COLUMNS]
    segments: list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]] = []
    for path in segment_paths:
        segment_id = path.stem.rsplit("_", maxsplit=1)[-1]
        frame = pd.read_csv(path, usecols=usecols)
        values = frame[usecols].to_numpy(dtype=float)
        if len(frame) == 0:
            raise ValueError(f"{path.name} contains no samples")
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} contains missing or non-finite target values")
        if not frame["relative_time"].is_monotonic_increasing:
            raise ValueError(f"{path.name} has non-monotonic relative time")

        summary_row = summary.loc[summary["segment_id"] == segment_id]
        if len(summary_row) != 1:
            raise ValueError(f"Expected one summary row for {segment_id}, found {len(summary_row)}")
        start_world_time = pd.Timestamp(summary_row.iloc[0]["start_pc_world_datetime_local"])
        end_world_time = pd.Timestamp(summary_row.iloc[0]["end_pc_world_datetime_local"])
        elapsed = frame["relative_time"] - frame["relative_time"].iloc[0]
        frame["world_time"] = start_world_time + pd.to_timedelta(elapsed, unit="s")
        endpoint_error_s = abs((frame["world_time"].iloc[-1] - end_world_time).total_seconds())
        if endpoint_error_s > 0.02:
            raise ValueError(
                f"{segment_id} world-time endpoint differs from the summary by "
                f"{endpoint_error_s:.6f} s"
            )
        segments.append((segment_id, frame, start_world_time, end_world_time))
    return segments


def make_figure(
    segments: list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(7.2, 8.6),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    for column_index, (segment_id, frame, start_time, end_time) in enumerate(segments):
        start_clock = start_time.strftime("%H:%M:%S.%f")[:-3]
        end_clock = end_time.strftime("%H:%M:%S.%f")[:-3]
        duration_s = (end_time - start_time).total_seconds()
        axes[0, column_index].set_title(
            f"{segment_id} | {start_clock} - {end_clock}\n"
            f"n = {len(frame):,}, duration = {duration_s:.2f} s",
            fontsize=8.5,
            fontweight="bold",
            pad=6,
        )

        for row_index, (az_column, color) in enumerate(zip(AZ_COLUMNS, IMU_COLORS, strict=True)):
            axis = axes[row_index, column_index]
            axis.plot(
                frame["world_time"],
                frame[az_column],
                color=color,
                linewidth=0.55,
            )
            axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, linestyle="--", zorder=0)
            axis.set_xlim(frame["world_time"].iloc[0], frame["world_time"].iloc[-1])
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
            axis.tick_params(direction="out", length=2.5, width=0.65)

    for row_index, az_column in enumerate(AZ_COLUMNS):
        axes[row_index, 0].set_ylabel(
            f"IMU {row_index + 1}\n{az_column} (raw units)",
            fontsize=7.5,
        )

    for column_index in range(2):
        bottom_axis = axes[-1, column_index]
        bottom_axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        bottom_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        bottom_axis.set_xlabel("World time (Asia/Shanghai, HH:MM:SS)", fontsize=8)

    recording_date = segments[0][2].strftime("%Y-%m-%d")
    figure.suptitle(
        f"Five-IMU z-axis acceleration across two segments | {recording_date}",
        fontsize=10,
        y=1.015,
    )
    figure.align_ylabels(axes[:, 0])
    return figure


def main() -> None:
    args = parse_args()
    segments = load_segments(args.input_dir)
    figure = make_figure(segments)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / "imu1_to_imu5_az_two_segments_world_time"
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


if __name__ == "__main__":
    main()
