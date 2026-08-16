"""Show and compare P06's first five lumbar-z FOG transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import plot_stanford_ngm_p06_first2_fog_transitions as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "dataset" / "5.Stanford imu-fog-detection" / "processed_NGM"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "figures"

EVENT_COUNT = 5
EVENT_COLORS = ("#315F85", "#C27335", "#587A52", "#8A5A83", "#8A7046")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    base.configure_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def shade_actual_states(
    axis: plt.Axes,
    relative_time: np.ndarray,
    state_matrix: np.ndarray,
    sampling_rate: float,
) -> None:
    all_nonfog = np.all(state_matrix == 0, axis=0)
    all_fog = np.all(state_matrix == 1, axis=0)
    categories = np.full(relative_time.shape, 2, dtype=np.int8)
    categories[all_nonfog] = 0
    categories[all_fog] = 1
    styles = {
        0: {
            "facecolor": base.NONFOG_COLOR,
            "edgecolor": "none",
            "hatch": None,
        },
        1: {"facecolor": base.FOG_COLOR, "edgecolor": "none", "hatch": None},
        2: {
            "facecolor": base.MIXED_COLOR,
            "edgecolor": "#B8ADA0",
            "hatch": "////",
        },
    }
    half_sample = 0.5 / sampling_rate
    for category, style in styles.items():
        for start, end in base.true_runs(categories == category):
            left = max(-base.TRANSITION_WINDOW_SEC, relative_time[start] - half_sample)
            right = min(
                base.TRANSITION_WINDOW_SEC,
                relative_time[end - 1] + half_sample,
            )
            axis.axvspan(
                left,
                right,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=0.0,
                zorder=0,
            )


def plot_five_event_alignment(
    axis: plt.Axes,
    events: list[dict[str, Any]],
    sampling_rate: float,
    alignment: str,
) -> None:
    margin = int(round(base.TRANSITION_WINDOW_SEC * sampling_rate))
    sample_offsets = np.arange(-margin, margin + 1, dtype=np.int64)
    relative_time = sample_offsets / sampling_rate
    signals: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for event in events:
        center = (
            int(event["start"])
            if alignment == "onset"
            else int(event["end"]) + 1
        )
        indices = center + sample_offsets
        record_signal = np.asarray(event["record_signal"], dtype=np.float32)
        record_label = np.asarray(event["record_label"], dtype=np.int8)
        if indices[0] < 0 or indices[-1] >= len(record_signal):
            raise ValueError(
                f"{event['record_id']} event {event['event_id']} lacks a full window"
            )
        signals.append(record_signal[indices])
        labels.append(record_label[indices])

    state_matrix = np.stack(labels)
    shade_actual_states(axis, relative_time, state_matrix, sampling_rate)
    for event, values, color in zip(events, signals, EVENT_COLORS):
        axis.plot(
            relative_time,
            values,
            color=color,
            linewidth=0.72,
            alpha=0.96,
            zorder=2,
        )
        opposite_boundary = (
            float(event["duration_sec"])
            if alignment == "onset"
            else -float(event["duration_sec"])
        )
        axis.axvline(
            opposite_boundary,
            color=color,
            linestyle=":",
            linewidth=0.75,
            alpha=0.9,
            zorder=1,
        )

    if alignment == "onset":
        title = "Onset-aligned comparison: Non-FOG → FOG"
        boundary_text = "FOG onset"
    elif alignment == "offset":
        title = "Offset-aligned comparison: FOG → Non-FOG"
        boundary_text = "FOG offset"
    else:
        raise ValueError(alignment)
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
    axis.set_xlim(-base.TRANSITION_WINDOW_SEC, base.TRANSITION_WINDOW_SEC)
    axis.set_title(title, loc="left", pad=4.0)
    axis.set_xlabel("Time relative to boundary (s)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))


def create_figure(events: list[dict[str, Any]], sampling_rate: float) -> plt.Figure:
    configure_style()
    figure = plt.figure(figsize=(7.15, 8.85))
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=(1.0, 1.0, 1.0, 1.18),
        left=0.09,
        right=0.985,
        bottom=0.065,
        top=0.84,
        hspace=0.5,
        wspace=0.16,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, :]),
        figure.add_subplot(grid[3, 0]),
        figure.add_subplot(grid[3, 1]),
    ]

    for axis, event, color in zip(axes[:5], events, EVENT_COLORS):
        base.plot_event_context(axis, event, sampling_rate, color)
    plot_five_event_alignment(axes[5], events, sampling_rate, "onset")
    plot_five_event_alignment(axes[6], events, sampling_rate, "offset")

    all_values = np.concatenate(
        [np.asarray(event["signal"], dtype=np.float32) for event in events]
    )
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_margin = max(0.08, 0.05 * (y_max - y_min))
    for label, axis in zip("abcdefg", axes):
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        axis.grid(axis="y", color="#FFFFFF", linewidth=0.6, alpha=0.95)
        axis.tick_params(colors="#4B5563")
        base.add_panel_label(axis, label)
    for axis in (axes[0], axes[2], axes[4], axes[5]):
        axis.set_ylabel("Lumbar az (g)")
    for axis in (axes[1], axes[3], axes[6]):
        axis.tick_params(labelleft=False)

    event_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.2,
            label=f"Event {event['ordinal']} ({event['duration_sec']:.1f} s)",
        )
        for event, color in zip(events, EVENT_COLORS)
    ]
    state_handles = [
        Patch(
            facecolor=base.NONFOG_COLOR,
            edgecolor="#B8C8D5",
            label="All Non-FOG",
        ),
        Patch(facecolor=base.FOG_COLOR, edgecolor="none", label="All FOG"),
        Patch(
            facecolor=base.MIXED_COLOR,
            edgecolor="#B8ADA0",
            hatch="////",
            label="Mixed states",
        ),
    ]
    figure.legend(
        handles=event_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956),
        ncol=5,
        handlelength=1.6,
        columnspacing=1.1,
    )
    figure.legend(
        handles=state_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=3,
        handlelength=1.8,
        columnspacing=1.5,
    )
    figure.suptitle(
        "P06 lumbar z-axis acceleration: first five FOG transitions",
        x=0.5,
        y=0.992,
        fontsize=10.5,
        fontweight="semibold",
        color="#17212B",
    )
    figure.text(
        0.5,
        0.885,
        "All 64 Hz samples shown; ±3 s alignment windows use actual sample labels",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#56616D",
    )
    return figure


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events, sampling_rate = base.load_events(input_dir, event_count=EVENT_COUNT)
    figure = create_figure(events, sampling_rate)
    stem = output_dir / "stanford_ngm_P06_first5_fog_transition_comparison"
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
    base.write_source_data(stem.with_name(f"{stem.name}_source.csv"), events)

    print(
        json.dumps(
            {
                "subject_id": base.SUBJECT_ID,
                "channel": base.CHANNEL_NAME,
                "sampling_rate_hz": sampling_rate,
                "event_count": len(events),
                "transition_window_sec": base.TRANSITION_WINDOW_SEC,
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
