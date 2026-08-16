"""Compare onset and offset transitions of P06's first two FOG events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "dataset" / "5.Stanford imu-fog-detection" / "processed_NGM"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "figures"

SUBJECT_ID = "P06"
CHANNEL_NAME = "imu_lumbar_az"
TRANSITION_WINDOW_SEC = 3.0

EVENT_COLORS = ("#315F85", "#C27335")
SIGNAL_COLOR = "#243447"
NONFOG_COLOR = "#EAF2F8"
FOG_COLOR = "#E9968A"
MIXED_COLOR = "#EEEAE3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
        }
    )


def load_events(
    input_dir: Path,
    event_count: int,
) -> tuple[list[dict[str, Any]], float]:
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    sampling_rate = float(schema["sampling_rate_hz"])
    channel_names = [channel["name"] for channel in schema["channels"]]
    channel_index = channel_names.index(CHANNEL_NAME)

    events = [
        row
        for row in read_csv(input_dir / "fog_events.csv")
        if row["subject_id"] == SUBJECT_ID
    ]
    events.sort(key=lambda row: (int(row["segment_id"]), int(row["event_id"])))
    if event_count <= 0:
        raise ValueError("event_count must be positive")
    if len(events) < event_count:
        raise ValueError(f"{SUBJECT_ID} has fewer than {event_count} FOG events")

    selected: list[dict[str, Any]] = []
    record_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    margin = int(round(TRANSITION_WINDOW_SEC * sampling_rate))
    for ordinal, event in enumerate(events[:event_count], start=1):
        record_id = event["record_id"]
        if record_id not in record_cache:
            with np.load(
                input_dir / "records" / f"{record_id}.npz", allow_pickle=False
            ) as record:
                record_cache[record_id] = (
                    np.asarray(record["x"], dtype=np.float32),
                    np.asarray(record["y_binary"], dtype=np.int8),
                )
        x, y = record_cache[record_id]
        start = int(event["start_index"])
        end = int(event["end_index"])
        if not np.all(y[start : end + 1] == 1):
            raise ValueError(f"event labels do not match {record_id} event {event['event_id']}")
        context_start = max(0, start - margin)
        context_end = min(len(y), end + 1 + margin)
        sample_indices = np.arange(context_start, context_end, dtype=np.int64)
        selected.append(
            {
                "ordinal": ordinal,
                "record_id": record_id,
                "event_id": int(event["event_id"]),
                "start": start,
                "end": end,
                "duration_sec": (end - start + 1) / sampling_rate,
                "sample_indices": sample_indices,
                "signal": x[sample_indices, channel_index],
                "label": y[sample_indices],
                "time_sec": sample_indices / sampling_rate,
                "relative_onset_sec": (sample_indices - start) / sampling_rate,
                "relative_offset_sec": (sample_indices - (end + 1)) / sampling_rate,
                "record_signal": x[:, channel_index],
                "record_label": y,
            }
        )
    return selected, sampling_rate


def load_first_two_events(input_dir: Path) -> tuple[list[dict[str, Any]], float]:
    return load_events(input_dir, event_count=2)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.int8)
    padded = np.pad(values, (1, 1), mode="constant")
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
        ha="left",
        color="#17212B",
    )


def plot_event_context(
    axis: plt.Axes,
    event: dict[str, Any],
    sampling_rate: float,
    color: str,
) -> None:
    time = np.asarray(event["time_sec"], dtype=np.float64)
    values = np.asarray(event["signal"], dtype=np.float32)
    labels = np.asarray(event["label"], dtype=np.int8)
    start_time = int(event["start"]) / sampling_rate
    offset_time = (int(event["end"]) + 1) / sampling_rate

    axis.set_facecolor(NONFOG_COLOR)
    for run_start, run_end in true_runs(labels == 1):
        axis.axvspan(
            float(time[run_start]),
            float(time[run_end - 1] + 1.0 / sampling_rate),
            color=FOG_COLOR,
            alpha=0.82,
            zorder=0,
        )
    axis.axvline(start_time, color=color, linestyle="--", linewidth=0.8, zorder=1)
    axis.axvline(offset_time, color=color, linestyle="--", linewidth=0.8, zorder=1)
    axis.plot(time, values, color=SIGNAL_COLOR, linewidth=0.65, zorder=2)
    axis.text(
        start_time,
        0.97,
        "onset",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=6.5,
        color=color,
    )
    axis.text(
        offset_time,
        0.97,
        "offset",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=6.5,
        color=color,
    )
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_title(
        f"Event {event['ordinal']}: {start_time:.3f}–{offset_time:.3f} s "
        f"(duration {event['duration_sec']:.1f} s)",
        loc="left",
        pad=4.0,
    )
    axis.set_xlabel("Time within trial (s)")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))


def shade_onset_comparison(axis: plt.Axes, durations: list[float]) -> None:
    shared_fog_end = min(durations)
    final_fog_end = max(durations)
    axis.axvspan(-TRANSITION_WINDOW_SEC, 0.0, color=NONFOG_COLOR, zorder=0)
    axis.axvspan(0.0, shared_fog_end, color=FOG_COLOR, alpha=0.72, zorder=0)
    if final_fog_end > shared_fog_end:
        axis.axvspan(
            shared_fog_end,
            final_fog_end,
            facecolor=MIXED_COLOR,
            edgecolor="#B8ADA0",
            hatch="////",
            linewidth=0.0,
            zorder=0,
        )
    if final_fog_end < TRANSITION_WINDOW_SEC:
        axis.axvspan(
            final_fog_end,
            TRANSITION_WINDOW_SEC,
            color=NONFOG_COLOR,
            zorder=0,
        )


def shade_offset_comparison(axis: plt.Axes, durations: list[float]) -> None:
    first_onset = -max(durations)
    shared_fog_start = -min(durations)
    if first_onset > -TRANSITION_WINDOW_SEC:
        axis.axvspan(
            -TRANSITION_WINDOW_SEC, first_onset, color=NONFOG_COLOR, zorder=0
        )
    if shared_fog_start > first_onset:
        axis.axvspan(
            first_onset,
            shared_fog_start,
            facecolor=MIXED_COLOR,
            edgecolor="#B8ADA0",
            hatch="////",
            linewidth=0.0,
            zorder=0,
        )
    axis.axvspan(shared_fog_start, 0.0, color=FOG_COLOR, alpha=0.72, zorder=0)
    axis.axvspan(0.0, TRANSITION_WINDOW_SEC, color=NONFOG_COLOR, zorder=0)


def plot_aligned_comparison(
    axis: plt.Axes,
    events: list[dict[str, Any]],
    alignment: str,
) -> None:
    durations = [float(event["duration_sec"]) for event in events]
    if alignment == "onset":
        shade_onset_comparison(axis, durations)
        relative_key = "relative_onset_sec"
        title = "Onset-aligned transition: Non-FOG → FOG"
        boundary_text = "FOG onset"
    elif alignment == "offset":
        shade_offset_comparison(axis, durations)
        relative_key = "relative_offset_sec"
        title = "Offset-aligned transition: FOG → Non-FOG"
        boundary_text = "FOG offset"
    else:
        raise ValueError(alignment)

    for event, color in zip(events, EVENT_COLORS):
        relative_time = np.asarray(event[relative_key], dtype=np.float64)
        values = np.asarray(event["signal"], dtype=np.float32)
        keep = (relative_time >= -TRANSITION_WINDOW_SEC) & (
            relative_time <= TRANSITION_WINDOW_SEC
        )
        axis.plot(
            relative_time[keep],
            values[keep],
            color=color,
            linewidth=0.85,
            label=f"Event {event['ordinal']} ({event['duration_sec']:.1f} s)",
            zorder=2,
        )
        other_boundary = (
            float(event["duration_sec"])
            if alignment == "onset"
            else -float(event["duration_sec"])
        )
        axis.axvline(
            other_boundary,
            color=color,
            linestyle=":",
            linewidth=0.85,
            alpha=0.9,
            zorder=1,
        )

    axis.axvline(0.0, color="#151A20", linestyle="--", linewidth=0.9, zorder=1)
    axis.text(
        0.0,
        0.97,
        boundary_text,
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=6.5,
        color="#151A20",
    )
    axis.set_xlim(-TRANSITION_WINDOW_SEC, TRANSITION_WINDOW_SEC)
    axis.set_title(title, loc="left", pad=4.0)
    axis.set_xlabel("Time relative to boundary (s)")
    axis.legend(loc="lower right", handlelength=2.0)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))


def create_figure(events: list[dict[str, Any]], sampling_rate: float) -> plt.Figure:
    configure_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.45),
        sharey=True,
        gridspec_kw={"height_ratios": [1.0, 1.08]},
    )

    plot_event_context(axes[0, 0], events[0], sampling_rate, EVENT_COLORS[0])
    plot_event_context(axes[0, 1], events[1], sampling_rate, EVENT_COLORS[1])
    plot_aligned_comparison(axes[1, 0], events, "onset")
    plot_aligned_comparison(axes[1, 1], events, "offset")

    all_values = np.concatenate(
        [np.asarray(event["signal"], dtype=np.float32) for event in events]
    )
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    margin = max(0.08, 0.05 * (y_max - y_min))
    for label, axis in zip("abcd", axes.flat):
        axis.set_ylim(y_min - margin, y_max + margin)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        axis.grid(axis="y", color="#FFFFFF", linewidth=0.6, alpha=0.95)
        axis.tick_params(colors="#4B5563")
        add_panel_label(axis, label)
    axes[0, 0].set_ylabel("Lumbar az (g)")
    axes[1, 0].set_ylabel("Lumbar az (g)")

    state_handles = [
        Patch(facecolor=NONFOG_COLOR, edgecolor="#B8C8D5", label="Both Non-FOG"),
        Patch(facecolor=FOG_COLOR, edgecolor="none", label="Both FOG"),
        Patch(
            facecolor=MIXED_COLOR,
            edgecolor="#B8ADA0",
            hatch="////",
            label="Different states",
        ),
    ]
    figure.legend(
        handles=state_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.944),
        ncol=3,
        handlelength=1.8,
        columnspacing=1.6,
    )
    figure.suptitle(
        "P06 lumbar z-axis acceleration: first two FOG transitions",
        x=0.5,
        y=0.992,
        fontsize=10.5,
        fontweight="semibold",
        color="#17212B",
    )
    figure.text(
        0.5,
        0.916,
        "All 64 Hz samples shown; dotted colored lines mark each event's opposite boundary",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#56616D",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.095,
        top=0.855,
        hspace=0.42,
        wspace=0.16,
    )
    return figure


def write_source_data(path: Path, events: list[dict[str, Any]]) -> None:
    fields = [
        "subject_id",
        "record_id",
        "event_ordinal",
        "event_id",
        "sample_index",
        "time_sec",
        "relative_onset_sec",
        "relative_offset_sec",
        "lumbar_az_g",
        "y_binary",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            arrays = zip(
                event["sample_indices"],
                event["time_sec"],
                event["relative_onset_sec"],
                event["relative_offset_sec"],
                event["signal"],
                event["label"],
            )
            for sample_index, time, onset, offset, value, label in arrays:
                writer.writerow(
                    {
                        "subject_id": SUBJECT_ID,
                        "record_id": event["record_id"],
                        "event_ordinal": event["ordinal"],
                        "event_id": event["event_id"],
                        "sample_index": int(sample_index),
                        "time_sec": float(time),
                        "relative_onset_sec": float(onset),
                        "relative_offset_sec": float(offset),
                        "lumbar_az_g": float(value),
                        "y_binary": int(label),
                    }
                )


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events, sampling_rate = load_first_two_events(input_dir)
    figure = create_figure(events, sampling_rate)
    stem = output_dir / "stanford_ngm_P06_first2_fog_transition_comparison"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    write_source_data(stem.with_name(f"{stem.name}_source.csv"), events)

    print(
        json.dumps(
            {
                "subject_id": SUBJECT_ID,
                "channel": CHANNEL_NAME,
                "sampling_rate_hz": sampling_rate,
                "transition_window_sec": TRANSITION_WINDOW_SEC,
                "events": [
                    {
                        "record_id": event["record_id"],
                        "event_id": event["event_id"],
                        "start_sec": int(event["start"]) / sampling_rate,
                        "offset_sec": (int(event["end"]) + 1) / sampling_rate,
                        "duration_sec": event["duration_sec"],
                    }
                    for event in events
                ],
                "output_stem": str(stem),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
