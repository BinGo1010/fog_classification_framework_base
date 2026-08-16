"""Plot IMU1 z-axis acceleration for the two All_dataset segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


SEGMENT_COLORS = ("#0F4D92", "#42949E")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/All_dataset/segments"),
        help="Directory containing the two segment CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/All_dataset/figures"),
        help="Directory for SVG, PDF, and PNG outputs.",
    )
    return parser.parse_args()


def load_segments(
    input_dir: Path,
) -> list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]]:
    paths = sorted(input_dir.glob("*_seg[0-9][0-9][0-9].csv"))
    if len(paths) != 2:
        raise ValueError(f"Expected exactly two segment CSV files, found {len(paths)}")

    summary_paths = sorted(input_dir.glob("*_segments_summary.csv"))
    if len(summary_paths) != 1:
        raise ValueError(f"Expected exactly one segment summary CSV, found {len(summary_paths)}")
    summary = pd.read_csv(summary_paths[0])

    segments: list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]] = []
    for path in paths:
        segment_id = path.stem.rsplit("_", maxsplit=1)[-1]
        frame = pd.read_csv(path, usecols=["relative_time", "imu1_az"])
        values = frame[["relative_time", "imu1_az"]].to_numpy(dtype=float)
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
        nrows=2,
        ncols=1,
        figsize=(7.2, 4.8),
        sharey=True,
        constrained_layout=True,
    )

    for index, (axis, (segment_id, frame, start_world_time, end_world_time), color) in enumerate(
        zip(axes, segments, SEGMENT_COLORS, strict=True)
    ):
        world_time = frame["world_time"]
        duration_s = (end_world_time - start_world_time).total_seconds()
        start_clock = start_world_time.strftime("%H:%M:%S.%f")[:-3]
        end_clock = end_world_time.strftime("%H:%M:%S.%f")[:-3]
        acceleration_z = frame["imu1_az"].to_numpy()
        axis.plot(world_time, acceleration_z, color=color, linewidth=0.65)
        axis.axhline(0.0, color="#A8A8A8", linewidth=0.6, linestyle="--", zorder=0)
        axis.set_xlim(world_time.iloc[0], world_time.iloc[-1])
        axis.set_title(
            f"{segment_id}  |  {start_clock} - {end_clock}  |  "
            f"n = {len(frame):,}, duration = {duration_s:.2f} s",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
            pad=5,
        )
        axis.text(
            -0.075,
            1.03,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axis.set_xlabel("World time (Asia/Shanghai, HH:MM:SS)")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        axis.tick_params(direction="out", length=3, width=0.7)

    figure.supylabel("IMU1 z-axis acceleration, imu1_az (raw units)", x=-0.01)
    recording_date = segments[0][2].strftime("%Y-%m-%d")
    figure.suptitle(
        f"IMU1 z-axis acceleration across two segments | {recording_date}",
        fontsize=10,
        y=1.025,
    )
    return figure


def main() -> None:
    args = parse_args()
    segments = load_segments(args.input_dir)
    figure = make_figure(segments)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / "imu1_az_two_segments"
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
