"""Plot all reconstructed processed Daphnet signals for subject S01."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


DEFAULT_INPUT = Path(
    "dataset/processed/daphnet_loso_3class_prefog2_win1_stride0p5/windows.npz"
)
DEFAULT_OUTPUT = Path("outputs/figures/daphnet_S01_all_processed_raw.png")
SAMPLING_RATE = 64.0
BASE_STRIDE = 32
SENSORS = (
    ("Ankle", slice(0, 3)),
    ("Thigh", slice(3, 6)),
    ("Trunk", slice(6, 9)),
)
AXES = ("Forward", "Vertical", "Lateral")
AXIS_COLORS = ("#1565C0", "#EF6C00", "#6A1B9A")
STATE_COLORS = {1: "#F6B94A", 2: "#E85A57"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subject", default="S01")
    return parser.parse_args()


def reconstruct_record(
    windows: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """Reconstruct one record exactly from its 50%-overlapping windows."""
    order = np.argsort(starts)
    windows = windows[order]
    starts = starts[order]
    if starts[0] != 0 or not np.all(np.diff(starts) == BASE_STRIDE):
        raise ValueError("Record windows are not a complete 32-sample-stride sequence")
    if not np.allclose(windows[:-1, BASE_STRIDE:], windows[1:, :BASE_STRIDE]):
        raise ValueError("Overlapping signal samples do not match")
    parts = [windows[0]]
    parts.extend(window[-BASE_STRIDE:] for window in windows[1:])
    return np.concatenate(parts, axis=0)


def label_spans(
    labels: np.ndarray,
    starts: np.ndarray,
    label: int,
) -> list[tuple[float, float]]:
    """Union overlapping one-second windows carrying a selected label."""
    intervals = sorted(
        (int(start), int(start + 64))
        for start, current_label in zip(starts, labels)
        if int(current_label) == label
    )
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, stop in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start / SAMPLING_RATE / 60.0, stop / SAMPLING_RATE / 60.0) for start, stop in merged]


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        selected = np.asarray(data["subject"]) == args.subject
        x = np.asarray(data["X"])[selected]
        y = np.asarray(data["y"])[selected]
        file_id = np.asarray(data["file_id"])[selected]
        source = np.asarray(data["source"])[selected]
        start_sample = np.asarray(data["start_sample"])[selected]

    record_ids = sorted(np.unique(file_id))
    if not record_ids:
        raise ValueError(f"No records found for {args.subject}")

    records = []
    total_samples = 0
    for record_id in record_ids:
        current = file_id == record_id
        signal = reconstruct_record(x[current], start_sample[current])
        order = np.argsort(start_sample[current])
        record_labels = y[current][order]
        record_starts = start_sample[current][order]
        records.append(
            {
                "record_id": str(record_id),
                "source": str(source[current][0]),
                "signal": signal,
                "labels": record_labels,
                "starts": record_starts,
            }
        )
        total_samples += len(signal)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        len(SENSORS),
        len(records),
        figsize=(19, 10),
        squeeze=False,
    )
    for column, record in enumerate(records):
        signal = record["signal"]
        time_minutes = np.arange(len(signal)) / SAMPLING_RATE / 60.0
        pre_fog_spans = label_spans(record["labels"], record["starts"], 1)
        fog_spans = label_spans(record["labels"], record["starts"], 2)
        duration_minutes = len(signal) / SAMPLING_RATE / 60.0
        for row, (sensor_name, channels) in enumerate(SENSORS):
            ax = axes[row, column]
            for axis_index, (axis_name, color) in enumerate(zip(AXES, AXIS_COLORS)):
                ax.plot(
                    time_minutes,
                    signal[:, channels.start + axis_index],
                    color=color,
                    linewidth=0.32,
                    alpha=0.78,
                    rasterized=True,
                )
            for start, stop in pre_fog_spans:
                ax.axvspan(start, stop, color=STATE_COLORS[1], alpha=0.22, linewidth=0)
            for start, stop in fog_spans:
                ax.axvspan(start, stop, color=STATE_COLORS[2], alpha=0.22, linewidth=0)
            ax.set_xlim(0, duration_minutes)
            ax.set_ylabel(f"{sensor_name}\nacceleration (g)")
            ax.grid(True, color="#D9DEE5", linewidth=0.55, alpha=0.75)
            if row == 0:
                ax.set_title(
                    f"{record['record_id']} · {record['source']}\n"
                    f"{duration_minutes:.2f} min · {len(signal):,} samples",
                    fontsize=12,
                    fontweight="bold",
                )
            if row == len(SENSORS) - 1:
                ax.set_xlabel("Time within record (minutes)")

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=axis_name)
        for axis_name, color in zip(AXES, AXIS_COLORS)
    ]
    legend_handles.extend(
        (
            Patch(facecolor=STATE_COLORS[1], alpha=0.35, label="PRE_FOG window label"),
            Patch(facecolor=STATE_COLORS[2], alpha=0.35, label="FOG window label"),
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncols=5,
        frameon=True,
        fontsize=10,
    )
    fig.suptitle(
        f"Daphnet {args.subject}: all processed acceleration samples (64 Hz)",
        fontsize=18,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        f"All {total_samples:,} processed samples are plotted; no display downsampling. "
        "Signals are reconstructed exactly from matching 50%-overlapping windows. "
        "Shading represents processed one-second window labels.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.995, 0.89), h_pad=1.2, w_pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"records={len(records)} samples={total_samples} duration_min={total_samples / SAMPLING_RATE / 60:.6f}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
