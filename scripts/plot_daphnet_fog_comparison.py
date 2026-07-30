"""Plot matched non-FOG and FOG acceleration windows from one Daphnet subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SENSOR_NAMES = ("Ankle", "Thigh", "Waist (trunk)")
AXIS_NAMES = ("Forward", "Vertical", "Lateral")
AXIS_COLORS = ("#1565C0", "#EF6C00", "#6A1B9A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=Path(
            "dataset/1.Daphnet Freezing of Gait Dataset/"
            "processed/records/S01_seg001.npz"
        ),
    )
    parser.add_argument("--fog-start", type=int, default=47552)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    parser.add_argument(
        "--sensor",
        choices=("all", "ankle", "thigh", "trunk"),
        default="all",
        help="Plot all sensors or one sensor only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/daphnet_S01_fog_vs_nonfog.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_samples = int(round(args.duration * args.sampling_rate))

    with np.load(args.record) as record:
        x = np.asarray(record["x"], dtype=np.float32)
        labels = np.asarray(record["y_binary"], dtype=np.int8)

    fog_slice = slice(args.fog_start, args.fog_start + window_samples)
    nonfog_slice = slice(args.fog_start - window_samples, args.fog_start)
    if nonfog_slice.start < 0 or fog_slice.stop > len(labels):
        raise ValueError("Requested windows extend beyond the record.")
    if np.any(labels[nonfog_slice] != 0):
        raise ValueError("The selected non-FOG window is not label-pure.")
    if np.any(labels[fog_slice] != 1):
        raise ValueError("The selected FOG window is not label-pure.")

    time_s = np.arange(window_samples) / args.sampling_rate
    windows = (
        ("non-FOG", x[nonfog_slice], "#2E7D32"),
        ("FOG", x[fog_slice], "#C62828"),
    )
    sensor_indices = (
        range(len(SENSOR_NAMES))
        if args.sensor == "all"
        else ({"ankle": 0, "thigh": 1, "trunk": 2}[args.sensor],)
    )
    sensor_indices = tuple(sensor_indices)
    figure_height = 9 if len(sensor_indices) > 1 else 4.5

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        len(sensor_indices),
        2,
        figsize=(14, figure_height),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.78 if len(sensor_indices) == 1 else 0.86,
        bottom=0.20 if len(sensor_indices) == 1 else 0.105,
        hspace=0.36,
        wspace=0.12,
    )

    for row, sensor_idx in enumerate(sensor_indices):
        sensor = SENSOR_NAMES[sensor_idx]
        channel_start = sensor_idx * 3
        for col, (state, values, title_color) in enumerate(windows):
            ax = axes[row, col]
            for axis_idx, (axis_name, color) in enumerate(
                zip(AXIS_NAMES, AXIS_COLORS)
            ):
                ax.plot(
                    time_s,
                    values[:, channel_start + axis_idx],
                    color=color,
                    linewidth=0.9,
                    alpha=0.92,
                    label=axis_name,
                )
            ax.set_title(
                f"{sensor} acceleration — {state}",
                color=title_color,
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylabel("Acceleration (g)")
            ax.set_xlim(0, args.duration)
            ax.grid(True, color="#D9DEE5", linewidth=0.6, alpha=0.8)
            if row == 0:
                ax.legend(
                    loc="upper right",
                    ncol=3,
                    frameon=True,
                    fontsize=9,
                    columnspacing=1.0,
                )
            if row == len(sensor_indices) - 1:
                ax.set_xlabel("Time within selected window (s)")

    record_name = args.record.stem
    sensor_title = (
        "all three sensors"
        if args.sensor == "all"
        else f"{SENSOR_NAMES[sensor_indices[0]]} sensor"
    )
    fig.suptitle(
        f"Daphnet subject S01: matched {args.duration:g}-second "
        f"non-FOG and FOG windows — {sensor_title}\n"
        f"Record {record_name}, 64 Hz, raw processed acceleration",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "Labels: non-FOG = normal experiment activity; FOG = annotated "
        "freezing-of-gait episode. Windows are contiguous at the FOG onset.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#424242",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "record": str(args.record),
        "sampling_rate_hz": args.sampling_rate,
        "duration_s": args.duration,
        "nonfog_samples": [nonfog_slice.start, nonfog_slice.stop],
        "fog_samples": [fog_slice.start, fog_slice.stop],
        "nonfog_record_time_s": [
            nonfog_slice.start / args.sampling_rate,
            nonfog_slice.stop / args.sampling_rate,
        ],
        "fog_record_time_s": [
            fog_slice.start / args.sampling_rate,
            fog_slice.stop / args.sampling_rate,
        ],
        "output": str(args.output),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
