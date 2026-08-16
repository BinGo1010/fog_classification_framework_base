"""Plot P06 lumbar z-axis acceleration with sample-level FOG annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

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

SIGNAL_COLOR = "#243447"
NONFOG_COLOR = "#EAF2F8"
FOG_COLOR = "#E9968A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.int8)
    padded = np.pad(values, (1, 1), mode="constant")
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def load_subject_records(
    input_dir: Path,
) -> tuple[list[dict[str, object]], float]:
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    sampling_rate = float(schema["sampling_rate_hz"])
    channel_names = [channel["name"] for channel in schema["channels"]]
    if CHANNEL_NAME not in channel_names:
        raise ValueError(f"channel not found: {CHANNEL_NAME}")
    channel_index = channel_names.index(CHANNEL_NAME)

    rows = [
        row
        for row in read_manifest(input_dir / "manifest.csv")
        if row["subject_id"] == SUBJECT_ID
    ]
    rows.sort(key=lambda row: int(row["segment_id"]))
    if not rows:
        raise ValueError(f"no records found for {SUBJECT_ID}")

    records: list[dict[str, object]] = []
    for row in rows:
        with np.load(input_dir / row["record_path"], allow_pickle=False) as record:
            x = np.asarray(record["x"], dtype=np.float32)
            y = np.asarray(record["y_binary"], dtype=np.int8)
        if len(x) != len(y) or not np.isin(y, (0, 1)).all():
            raise ValueError(f"invalid record: {row['record_id']}")
        records.append(
            {
                "record_id": row["record_id"],
                "segment_id": int(row["segment_id"]),
                "signal": x[:, channel_index],
                "label": y,
            }
        )
    return records, sampling_rate


def plot_records(
    records: list[dict[str, object]],
    sampling_rate: float,
) -> plt.Figure:
    configure_style()
    n_columns = 2
    n_rows = int(np.ceil(len(records) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.15, 9.2),
        sharey=True,
        squeeze=False,
    )

    all_values = np.concatenate(
        [np.asarray(record["signal"], dtype=np.float32) for record in records]
    )
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_margin = max(0.08, 0.04 * (y_max - y_min))

    for index, (axis, record) in enumerate(zip(axes.flat, records)):
        values = np.asarray(record["signal"], dtype=np.float32)
        labels = np.asarray(record["label"], dtype=np.int8)
        time = np.arange(len(values), dtype=np.float64) / sampling_rate

        axis.set_facecolor(NONFOG_COLOR)
        for start, end in true_runs(labels == 1):
            axis.axvspan(
                start / sampling_rate,
                end / sampling_rate,
                facecolor=FOG_COLOR,
                edgecolor="none",
                alpha=0.76,
                zorder=0,
            )
        axis.plot(
            time,
            values,
            color=SIGNAL_COLOR,
            linewidth=0.48,
            solid_capstyle="round",
            rasterized=False,
            zorder=2,
        )
        fog_samples = int(np.count_nonzero(labels == 1))
        fog_seconds = fog_samples / sampling_rate
        fog_percent = 100.0 * fog_samples / len(labels)
        axis.set_title(
            f"{record['record_id']}  |  FOG {fog_seconds:.1f} s ({fog_percent:.1f}%)",
            loc="left",
            pad=2.0,
            color="#20262E",
        )
        axis.set_xlim(0.0, len(values) / sampling_rate)
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.grid(axis="y", color="#FFFFFF", linewidth=0.55, alpha=0.9)
        axis.tick_params(colors="#4B5563")
        if index % n_columns == 1:
            axis.tick_params(labelleft=False)

    for axis in axes.flat[len(records) :]:
        axis.set_visible(False)

    handles = [
        Patch(facecolor=NONFOG_COLOR, edgecolor="#B8C8D5", label="Non-FOG"),
        Patch(facecolor=FOG_COLOR, edgecolor="none", label="FOG"),
        plt.Line2D([0], [0], color=SIGNAL_COLOR, linewidth=0.9, label="Lumbar az"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.966),
        ncol=3,
        handlelength=1.7,
        columnspacing=1.6,
    )
    figure.suptitle(
        "P06 lumbar IMU z-axis acceleration across 14 walking trials",
        x=0.5,
        y=0.993,
        fontsize=10.0,
        fontweight="semibold",
        color="#17212B",
    )
    figure.text(
        0.5,
        0.941,
        "64 Hz FIR-preprocessed signal; shaded backgrounds show sample-level labels",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#56616D",
    )
    figure.supxlabel("Time within trial (s)", x=0.51, y=0.018)
    figure.supylabel("Lumbar z-axis acceleration, az (g)", x=0.012)
    figure.subplots_adjust(
        left=0.085,
        right=0.988,
        bottom=0.055,
        top=0.907,
        hspace=0.53,
        wspace=0.12,
    )
    return figure


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, sampling_rate = load_subject_records(input_dir)
    figure = plot_records(records, sampling_rate)
    stem = output_dir / "stanford_ngm_P06_lumbar_az_fog_regions"
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

    print(
        json.dumps(
            {
                "subject_id": SUBJECT_ID,
                "channel": CHANNEL_NAME,
                "sampling_rate_hz": sampling_rate,
                "record_count": len(records),
                "sample_count": int(
                    sum(len(np.asarray(record["label"])) for record in records)
                ),
                "output_stem": str(stem),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
