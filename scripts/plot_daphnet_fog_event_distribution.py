"""Summarize and plot Daphnet FOG event counts and duration distributions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EVENTS = Path(
    "dataset/1.Daphnet Freezing of Gait Dataset/processed/fog_events.csv"
)
DEFAULT_FIGURE = Path("outputs/figures/daphnet_fog_event_distribution")
DEFAULT_SUMMARY = Path(
    "outputs/statistics/daphnet_fog_event_summary_by_subject.csv"
)
SUBJECTS = tuple(f"S{index:02d}" for index in range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def load_durations(path: Path) -> dict[str, np.ndarray]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            duration = float(row["duration_sec"])
            if duration <= 0:
                raise ValueError(f"FOG event duration must be positive, got {duration}")
            grouped[row["subject_id"]].append(duration)
    return {
        subject: np.asarray(grouped.get(subject, []), dtype=float)
        for subject in SUBJECTS
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "event_count": 0,
            "total_duration_sec": 0.0,
            "total_duration_min": 0.0,
            "mean_duration_sec": np.nan,
            "median_duration_sec": np.nan,
            "q1_duration_sec": np.nan,
            "q3_duration_sec": np.nan,
            "min_duration_sec": np.nan,
            "max_duration_sec": np.nan,
        }
    return {
        "event_count": int(values.size),
        "total_duration_sec": float(values.sum()),
        "total_duration_min": float(values.sum() / 60.0),
        "mean_duration_sec": float(values.mean()),
        "median_duration_sec": float(np.median(values)),
        "q1_duration_sec": float(np.quantile(values, 0.25)),
        "q3_duration_sec": float(np.quantile(values, 0.75)),
        "min_duration_sec": float(values.min()),
        "max_duration_sec": float(values.max()),
    }


def write_summary(
    path: Path,
    durations: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows = [{"subject": subject, **describe(durations[subject])} for subject in SUBJECTS]
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def make_figure(
    durations: dict[str, np.ndarray],
    rows: list[dict[str, float | int | str]],
    output_stem: Path,
) -> None:
    configure_style()
    counts = np.asarray([int(row["event_count"]) for row in rows])
    total_minutes = np.asarray([float(row["total_duration_min"]) for row in rows])
    all_durations = np.concatenate([values for values in durations.values() if values.size])
    overall_median = float(np.median(all_durations))

    fig = plt.figure(figsize=(7.2, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.83, 1.45), hspace=0.52, wspace=0.30)
    count_ax = fig.add_subplot(grid[0, 0])
    total_ax = fig.add_subplot(grid[0, 1])
    distribution_ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(SUBJECTS))

    count_color = "#4C78A8"
    duration_color = "#E07B39"
    neutral_color = "#747B84"

    count_bars = count_ax.bar(x, counts, color=count_color, width=0.70)
    count_ax.set_title("FOG event count by subject", fontweight="bold")
    count_ax.set_ylabel("Number of events")
    count_ax.set_xticks(x, SUBJECTS)
    count_ax.set_ylim(0, max(counts) * 1.16)
    count_ax.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    count_ax.grid(axis="x", visible=False)
    for bar, value in zip(count_bars, counts):
        count_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.022,
            str(value),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=count_color if value else neutral_color,
            fontweight="bold",
        )
    add_panel_label(count_ax, "a")

    total_bars = total_ax.bar(x, total_minutes, color=duration_color, width=0.70)
    total_ax.set_title("Cumulative FOG duration by subject", fontweight="bold")
    total_ax.set_ylabel("Total duration (min)")
    total_ax.set_xticks(x, SUBJECTS)
    total_ax.set_ylim(0, max(total_minutes) * 1.18)
    total_ax.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    total_ax.grid(axis="x", visible=False)
    for bar, value in zip(total_bars, total_minutes):
        total_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_minutes) * 0.024,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=duration_color if value else neutral_color,
            fontweight="bold",
        )
    add_panel_label(total_ax, "b")

    valid_positions = [index + 1 for index, subject in enumerate(SUBJECTS) if durations[subject].size]
    valid_values = [durations[subject] for subject in SUBJECTS if durations[subject].size]
    box = distribution_ax.boxplot(
        valid_values,
        positions=valid_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1E3348", "linewidth": 1.25},
        whiskerprops={"color": neutral_color, "linewidth": 0.8},
        capprops={"color": neutral_color, "linewidth": 0.8},
        boxprops={"edgecolor": count_color, "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#C8D9E8")
        patch.set_alpha(0.80)

    for position, values in zip(valid_positions, valid_values):
        order = np.argsort(values, kind="stable")
        jitter = np.empty(len(values), dtype=float)
        jitter[order] = np.linspace(-0.20, 0.20, num=len(values))
        distribution_ax.scatter(
            position + jitter,
            values,
            s=9,
            color=count_color,
            alpha=0.60,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        distribution_ax.text(
            position,
            47.0,
            f"n={len(values)}",
            ha="center",
            va="top",
            fontsize=6.2,
            color=neutral_color,
        )
    for position in (4, 10):
        distribution_ax.text(
            position,
            0.60,
            "no events",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=neutral_color,
        )

    distribution_ax.axhline(
        overall_median,
        color=duration_color,
        linestyle="--",
        linewidth=1.0,
        label=f"Overall median = {overall_median:.2f} s",
    )
    distribution_ax.set_yscale("log")
    distribution_ax.set_ylim(0.4, 50)
    distribution_ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    distribution_ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
    distribution_ax.set_xticks(np.arange(1, len(SUBJECTS) + 1), SUBJECTS)
    distribution_ax.set_xlabel("Subject")
    distribution_ax.set_ylabel("Single-event duration (s, log scale)")
    distribution_ax.set_title(
        "Distribution of individual FOG event durations",
        fontweight="bold",
    )
    distribution_ax.grid(axis="y", which="major", color="#D9DEE5", linewidth=0.55)
    distribution_ax.grid(axis="x", visible=False)
    distribution_ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    add_panel_label(distribution_ax, "c")

    fig.suptitle(
        "Daphnet FOG events: frequency and duration heterogeneity",
        fontsize=10,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Each point is one contiguous sample-level FOG event. Boxes show median and IQR; whiskers show 1.5×IQR. "
        f"All {len(all_durations)} events are shown; S04 and S10 have no events.",
        ha="center",
        fontsize=6.2,
        color="#4A4F55",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.10)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    durations = load_durations(args.events)
    rows = write_summary(args.summary, durations)
    make_figure(durations, rows, args.figure)

    all_durations = np.concatenate([values for values in durations.values() if values.size])
    print(f"events={len(all_durations)} total_sec={all_durations.sum():.6f}")
    print(
        f"mean={all_durations.mean():.6f} median={np.median(all_durations):.6f} "
        f"q1={np.quantile(all_durations, 0.25):.6f} "
        f"q3={np.quantile(all_durations, 0.75):.6f} "
        f"min={all_durations.min():.6f} max={all_durations.max():.6f}"
    )
    print(f"Saved: {args.summary.resolve()}")
    print(f"Saved: {args.figure.with_suffix('.png').resolve()}")
    print(f"Saved: {args.figure.with_suffix('.svg').resolve()}")
    print(f"Saved: {args.figure.with_suffix('.pdf').resolve()}")
    print(f"Saved: {args.figure.with_suffix('.tiff').resolve()}")


if __name__ == "__main__":
    main()
