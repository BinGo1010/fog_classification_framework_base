"""Show all raw S01 Daphnet data and compare it with processed records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


RAW_COLUMNS = (
    "time_ms",
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
    "annotation",
)
SENSORS = (
    ("Ankle", slice(1, 4)),
    ("Thigh", slice(4, 7)),
    ("Trunk", slice(7, 10)),
)
AXES = ("Forward", "Vertical", "Lateral")
AXIS_COLORS = ("#1565C0", "#EF6C00", "#6A1B9A")
LABEL_COLORS = ("#B7BDC7", "#66A66D", "#E15B58")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("dataset/1.Daphnet Freezing of Gait Dataset/dataset"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("dataset/1.Daphnet Freezing of Gait Dataset/processed"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("outputs/figures/daphnet_S01_raw_all_labels_012.png"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("outputs/figures/daphnet_S01_raw_vs_processed.png"),
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["subject_id"] == "S01"]


def load_raw_records(raw_dir: Path) -> dict[str, np.ndarray]:
    records = {}
    for path in sorted(raw_dir.glob("S01R*.txt")):
        values = np.loadtxt(path, dtype=np.int32)
        if values.ndim != 2 or values.shape[1] != len(RAW_COLUMNS):
            raise ValueError(f"Unexpected raw shape for {path}: {values.shape}")
        labels = np.unique(values[:, -1])
        if not np.all(np.isin(labels, (0, 1, 2))):
            raise ValueError(f"Unexpected labels in {path}: {labels}")
        records[path.name] = values
    if not records:
        raise FileNotFoundError(f"No S01 raw TXT files found in {raw_dir}")
    return records


def time_minutes(raw: np.ndarray) -> np.ndarray:
    return (raw[:, 0].astype(np.float64) - float(raw[0, 0])) / 60_000.0


def state_strip(ax: plt.Axes, raw: np.ndarray) -> None:
    time = time_minutes(raw)
    cmap = ListedColormap(LABEL_COLORS)
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), cmap.N)
    ax.imshow(
        raw[np.newaxis, :, -1],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(time[0], time[-1], 0, 1),
        rasterized=True,
    )
    ax.set_yticks([0.5], ["Label"])
    ax.set_xlim(time[0], time[-1])
    ax.grid(False)


def make_raw_figure(records: dict[str, np.ndarray], output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        4,
        len(records),
        figsize=(18, 11),
        gridspec_kw={"height_ratios": (0.18, 1, 1, 1)},
        squeeze=False,
    )
    for column, (name, raw) in enumerate(records.items()):
        time = time_minutes(raw)
        counts = dict(zip(*np.unique(raw[:, -1], return_counts=True)))
        state_strip(axes[0, column], raw)
        axes[0, column].set_title(
            f"{name} · {len(raw):,} samples · {time[-1]:.2f} min\n"
            f"label 0: {counts.get(0, 0):,}   1: {counts.get(1, 0):,}   2: {counts.get(2, 0):,}",
            fontsize=11.5,
            fontweight="bold",
        )
        for row, (sensor_name, channels) in enumerate(SENSORS, start=1):
            ax = axes[row, column]
            for channel, axis_name, color in zip(range(channels.start, channels.stop), AXES, AXIS_COLORS):
                ax.plot(
                    time,
                    raw[:, channel],
                    color=color,
                    linewidth=0.28,
                    alpha=0.76,
                    rasterized=True,
                )
            ax.set_xlim(time[0], time[-1])
            ax.set_ylabel(f"{sensor_name}\nraw acceleration (mg)")
            ax.grid(True, color="#D9DEE5", linewidth=0.55, alpha=0.75)
            if row == 3:
                ax.set_xlabel("Raw recording time (minutes)")

    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=axis_name)
        for axis_name, color in zip(AXES, AXIS_COLORS)
    ]
    handles.extend(
        Patch(facecolor=color, label=f"raw label {label}")
        for label, color in enumerate(LABEL_COLORS)
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncols=6)
    fig.suptitle(
        "Daphnet S01: complete unprocessed raw signals with labels 0 / 1 / 2",
        fontsize=18,
        fontweight="bold",
        y=0.993,
    )
    fig.text(
        0.5,
        0.012,
        "All original TXT rows are plotted without display downsampling. "
        "Raw acceleration unit: mg; original timestamps are retained.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.995, 0.90), h_pad=0.7, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate_and_collect_processed(
    records: dict[str, np.ndarray],
    manifest: list[dict[str, str]],
    processed_dir: Path,
) -> tuple[dict[str, list[dict[str, object]]], float, int]:
    segments: dict[str, list[dict[str, object]]] = {name: [] for name in records}
    max_signal_error = 0.0
    label_mismatch = 0
    for row in manifest:
        source = row["source_file"]
        raw = records[source]
        start = int(row["source_start_row"])
        stop = int(row["source_end_row"]) + 1
        record_path = processed_dir / row["record_path"]
        with np.load(record_path, allow_pickle=False) as data:
            x_processed = np.asarray(data["x"])
            y_processed = np.asarray(data["y_binary"])
        raw_x_g = (raw[start:stop, 1:10].astype(np.float32) / 1000.0).astype(np.float32)
        raw_y_binary = (raw[start:stop, -1] == 2).astype(np.int8)
        if x_processed.shape != raw_x_g.shape or y_processed.shape != raw_y_binary.shape:
            raise ValueError(f"Shape mismatch for {record_path}")
        max_signal_error = max(max_signal_error, float(np.max(np.abs(x_processed - raw_x_g))))
        label_mismatch += int(np.count_nonzero(y_processed != raw_y_binary))
        if np.any(raw[start:stop, -1] == 0):
            raise ValueError(f"Processed segment {record_path} contains raw label 0")
        segments[source].append(
            {
                "record_id": row["record_id"],
                "start": start,
                "stop": stop,
                "x": x_processed,
            }
        )
    return segments, max_signal_error, label_mismatch


def make_comparison_figure(
    records: dict[str, np.ndarray],
    segments: dict[str, list[dict[str, object]]],
    max_signal_error: float,
    label_mismatch: int,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, len(records), figsize=(18, 10), squeeze=False)
    for column, (name, raw) in enumerate(records.items()):
        time = time_minutes(raw)
        for row, (sensor_name, raw_channels) in enumerate(SENSORS):
            ax = axes[row, column]
            for raw_channel in range(raw_channels.start, raw_channels.stop):
                ax.plot(
                    time,
                    raw[:, raw_channel] / 1000.0,
                    color="#AEB3BA",
                    linewidth=0.25,
                    alpha=0.46,
                    rasterized=True,
                )
            processed_channels = slice(raw_channels.start - 1, raw_channels.stop - 1)
            for segment in segments[name]:
                start = int(segment["start"])
                stop = int(segment["stop"])
                segment_time = time[start:stop]
                signal = np.asarray(segment["x"])
                for channel, color in zip(range(processed_channels.start, processed_channels.stop), AXIS_COLORS):
                    ax.plot(
                        segment_time,
                        signal[:, channel],
                        color=color,
                        linewidth=0.38,
                        alpha=0.80,
                        rasterized=True,
                    )
            ax.set_xlim(time[0], time[-1])
            ax.set_ylabel(f"{sensor_name}\nacceleration (g)")
            ax.grid(True, color="#D9DEE5", linewidth=0.55, alpha=0.75)
            if row == 0:
                retained = sum(int(segment["stop"]) - int(segment["start"]) for segment in segments[name])
                ax.set_title(
                    f"{name}: raw {len(raw):,} → processed {retained:,} samples "
                    f"({100.0 * retained / len(raw):.1f}% retained)",
                    fontsize=11.5,
                    fontweight="bold",
                )
            if row == 2:
                ax.set_xlabel("Original raw recording time (minutes)")

    handles = [Patch(facecolor="#AEB3BA", alpha=0.55, label="Raw (labels 0/1/2, converted mg→g)")]
    handles.extend(
        Line2D([0], [0], color=color, linewidth=2, label=f"Processed {axis_name}")
        for axis_name, color in zip(AXES, AXIS_COLORS)
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncols=4)
    fig.suptitle(
        "Daphnet S01: raw vs processed signals on the original timeline",
        fontsize=18,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Processed keeps every contiguous raw label-1/2 interval, removes label 0, "
        f"converts mg→g, and maps label 1→0 / label 2→1. "
        f"Pointwise signal max error: {max_signal_error:.1e} g; label mismatches: {label_mismatch}.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.995, 0.89), h_pad=1.0, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_raw_records(args.raw_dir)
    manifest = read_manifest(args.processed_dir / "manifest.csv")
    segments, max_signal_error, label_mismatch = validate_and_collect_processed(
        records, manifest, args.processed_dir
    )
    make_raw_figure(records, args.raw_output)
    make_comparison_figure(
        records,
        segments,
        max_signal_error,
        label_mismatch,
        args.comparison_output,
    )

    raw_counts = np.sum(
        [np.bincount(raw[:, -1], minlength=3) for raw in records.values()], axis=0
    )
    processed_samples = sum(
        int(segment["stop"]) - int(segment["start"])
        for source_segments in segments.values()
        for segment in source_segments
    )
    print(f"raw_counts_0_1_2={raw_counts.tolist()}")
    print(f"raw_total={int(raw_counts.sum())} processed_total={processed_samples}")
    print(f"max_signal_error_g={max_signal_error:.9g} label_mismatch={label_mismatch}")
    print(f"Saved: {args.raw_output.resolve()}")
    print(f"Saved: {args.comparison_output.resolve()}")


if __name__ == "__main__":
    main()
