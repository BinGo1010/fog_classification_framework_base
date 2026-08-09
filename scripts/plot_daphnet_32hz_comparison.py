"""Plot reproducible before/after validation of Daphnet 64-to-32 Hz processing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preprocess_daphnet_32hz as preprocessing


SOURCE_ROOT = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed"
OUTPUT_32HZ_ROOT = (
    ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_32Hz"
)
DEFAULT_FIGURE_DIR = ROOT / "outputs" / "daphnet_32hz_preprocessing_comparison"

RAW_COLOR = "#7A7F87"
FILTERED_COLOR = "#2D6A8A"
DOWNSAMPLED_COLOR = "#D97941"
FOG_COLOR = "#D95F59"
GRID_COLOR = "#D9DDE2"

CHANNEL_NAMES = (
    "Ankle forward",
    "Ankle vertical",
    "Ankle lateral",
    "Thigh forward",
    "Thigh vertical",
    "Thigh lateral",
    "Trunk forward",
    "Trunk vertical",
    "Trunk lateral",
)
VERTICAL_CHANNELS = (1, 4, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--processed-32hz", type=Path, default=OUTPUT_32HZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_record(root: Path, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / "records" / f"{record_id}.npz", allow_pickle=False) as record:
        return np.asarray(record["x"]), np.asarray(record["y_binary"])


def select_representative_event(
    source_root: Path,
    segment_seconds: float,
) -> dict[str, Any]:
    events = read_csv(source_root / "fog_events.csv")
    manifest = {row["record_id"]: row for row in read_csv(source_root / "manifest.csv")}
    half_span = segment_seconds / 2.0
    eligible: list[dict[str, Any]] = []
    for row in events:
        event = dict(row)
        start = float(row["start_time_sec"])
        end = float(row["end_time_sec"])
        center = (start + end) / 2.0
        last_time = (int(manifest[row["record_id"]]["n_samples"]) - 1) / 64.0
        if center - half_span >= 0.0 and center + half_span <= last_time:
            event["center_time_sec"] = center
            event["record_last_time_sec"] = last_time
            eligible.append(event)
    if not eligible:
        raise RuntimeError("no FoG event has enough temporal margin for the requested segment")

    durations = np.asarray([float(row["duration_sec"]) for row in eligible])
    median_duration = float(np.median(durations))
    selected = min(
        eligible,
        key=lambda row: (
            abs(float(row["duration_sec"]) - median_duration),
            row["record_id"],
            int(row["event_id"]),
        ),
    )
    selected["eligible_event_count"] = len(eligible)
    selected["eligible_duration_median_sec"] = median_duration
    selected["selection_rule"] = (
        "duration closest to the median among events with a full centered display margin"
    )
    return selected


def fog_intervals(
    labels: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for start, end in preprocessing.true_runs(np.asarray(labels) == 1):
        left = start / fs
        right = end / fs
        if right > start_time and left < end_time:
            intervals.append((max(left, start_time), min(right, end_time)))
    return intervals


def add_fog_spans(ax: plt.Axes, intervals: Sequence[tuple[float, float]]) -> None:
    for left, right in intervals:
        ax.axvspan(left, right, color=FOG_COLOR, alpha=0.10, linewidth=0, zorder=0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def calculate_psd_summary(
    source_root: Path,
    processed_root: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    manifest = read_csv(source_root / "manifest.csv")
    raw_psd: list[np.ndarray] = []
    processed_psd: list[np.ndarray] = []
    frequency: np.ndarray | None = None

    for row in manifest:
        record_id = row["record_id"]
        x_raw, _ = load_record(source_root, record_id)
        x_processed, _ = load_record(processed_root, record_id)
        f_raw, p_raw = signal.welch(
            x_raw,
            fs=64,
            axis=0,
            nperseg=2048,
            noverlap=1024,
            detrend="constant",
            scaling="density",
        )
        f_processed, p_processed = signal.welch(
            x_processed,
            fs=32,
            axis=0,
            nperseg=1024,
            noverlap=512,
            detrend="constant",
            scaling="density",
        )
        use_raw = f_raw <= 16.0 + 1e-12
        if frequency is None:
            frequency = f_processed
        if not np.allclose(f_raw[use_raw], f_processed, rtol=0.0, atol=1e-12):
            raise AssertionError("raw and processed Welch grids differ below 16 Hz")
        raw_psd.append(p_raw[use_raw].T)
        processed_psd.append(p_processed.T)

    assert frequency is not None
    raw = np.concatenate(raw_psd, axis=0)
    processed = np.concatenate(processed_psd, axis=0)
    tiny = np.finfo(np.float64).tiny

    rows: list[dict[str, Any]] = []
    for index, f in enumerate(frequency):
        rows.append(
            {
                "frequency_hz": float(f),
                "raw_median_psd_db": float(10 * np.log10(max(np.median(raw[:, index]), tiny))),
                "raw_q25_psd_db": float(10 * np.log10(max(np.quantile(raw[:, index], 0.25), tiny))),
                "raw_q75_psd_db": float(10 * np.log10(max(np.quantile(raw[:, index], 0.75), tiny))),
                "processed_median_psd_db": float(
                    10 * np.log10(max(np.median(processed[:, index]), tiny))
                ),
                "processed_q25_psd_db": float(
                    10 * np.log10(max(np.quantile(processed[:, index], 0.25), tiny))
                ),
                "processed_q75_psd_db": float(
                    10 * np.log10(max(np.quantile(processed[:, index], 0.75), tiny))
                ),
                "curve_count_per_method": int(raw.shape[0]),
            }
        )
    return frequency, rows


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def make_main_figure(
    output_dir: Path,
    event: dict[str, Any],
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_filtered: np.ndarray,
    x_32hz: np.ndarray,
    y_32hz: np.ndarray,
    psd_rows: Sequence[dict[str, Any]],
    coefficients: np.ndarray,
    segment_seconds: float,
) -> None:
    center = float(event["center_time_sec"])
    start_time = center - segment_seconds / 2.0
    end_time = center + segment_seconds / 2.0
    t_raw = np.arange(len(x_raw)) / 64.0
    t_32hz = np.arange(len(x_32hz)) / 32.0
    raw_mask = (t_raw >= start_time) & (t_raw <= end_time)
    processed_mask = (t_32hz >= start_time) & (t_32hz <= end_time)
    intervals = fog_intervals(y_raw, 64.0, start_time, end_time)

    fig = plt.figure(figsize=(7.15, 8.15), constrained_layout=False)
    grid = GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=(1.0, 1.0, 1.0, 0.65, 1.35),
        hspace=0.42,
        wspace=0.34,
        left=0.10,
        right=0.98,
        top=0.97,
        bottom=0.075,
    )
    time_axes = [fig.add_subplot(grid[index, :]) for index in range(3)]
    label_ax = fig.add_subplot(grid[3, :])
    psd_ax = fig.add_subplot(grid[4, 0])
    response_ax = fig.add_subplot(grid[4, 1])

    for axis_index, (ax, channel) in enumerate(zip(time_axes, VERTICAL_CHANNELS)):
        add_fog_spans(ax, intervals)
        ax.plot(
            t_raw[raw_mask],
            x_raw[raw_mask, channel],
            color=RAW_COLOR,
            linewidth=0.65,
            alpha=0.58,
            label="Raw 64 Hz" if axis_index == 0 else None,
            zorder=1,
        )
        ax.plot(
            t_raw[raw_mask],
            x_filtered[raw_mask, channel],
            color=FILTERED_COLOR,
            linewidth=1.0,
            label="FIR-aligned 64 Hz" if axis_index == 0 else None,
            zorder=2,
        )
        ax.plot(
            t_32hz[processed_mask],
            x_32hz[processed_mask, channel],
            color=DOWNSAMPLED_COLOR,
            linewidth=0.75,
            marker="o",
            markersize=1.7,
            markevery=8,
            label="Final 32 Hz" if axis_index == 0 else None,
            zorder=3,
        )
        ax.set_xlim(start_time, end_time)
        ax.set_ylabel(f"{CHANNEL_NAMES[channel]}\nacceleration (g)")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.65)
        if axis_index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time from record start (s)")
        add_panel_label(ax, chr(ord("a") + axis_index))
    time_axes[0].legend(loc="upper right", ncol=3, handlelength=2.3, columnspacing=1.2)

    add_fog_spans(label_ax, intervals)
    label_ax.step(
        t_raw[raw_mask],
        y_raw[raw_mask],
        where="post",
        color=RAW_COLOR,
        linewidth=1.0,
        label="Original label 64 Hz",
    )
    label_ax.step(
        t_32hz[processed_mask],
        y_32hz[processed_mask],
        where="post",
        color=DOWNSAMPLED_COLOR,
        linewidth=1.0,
        linestyle="--",
        label="Nearest-neighbour label 32 Hz",
    )
    label_ax.set_xlim(start_time, end_time)
    label_ax.set_ylim(-0.15, 1.25)
    label_ax.set_yticks((0, 1), labels=("Non-FoG", "FoG"))
    label_ax.set_xlabel("Time from record start (s)")
    label_ax.legend(loc="upper right", ncol=2)
    label_ax.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.65)
    add_panel_label(label_ax, "d")

    frequency = np.asarray([float(row["frequency_hz"]) for row in psd_rows])
    raw_med = np.asarray([float(row["raw_median_psd_db"]) for row in psd_rows])
    raw_q25 = np.asarray([float(row["raw_q25_psd_db"]) for row in psd_rows])
    raw_q75 = np.asarray([float(row["raw_q75_psd_db"]) for row in psd_rows])
    out_med = np.asarray([float(row["processed_median_psd_db"]) for row in psd_rows])
    out_q25 = np.asarray([float(row["processed_q25_psd_db"]) for row in psd_rows])
    out_q75 = np.asarray([float(row["processed_q75_psd_db"]) for row in psd_rows])
    psd_ax.fill_between(frequency, raw_q25, raw_q75, color=RAW_COLOR, alpha=0.13, linewidth=0)
    psd_ax.fill_between(
        frequency, out_q25, out_q75, color=FILTERED_COLOR, alpha=0.13, linewidth=0
    )
    psd_ax.plot(frequency, raw_med, color=RAW_COLOR, linewidth=1.0, label="Raw 64 Hz")
    psd_ax.plot(
        frequency,
        out_med,
        color=FILTERED_COLOR,
        linewidth=1.15,
        label="Filtered + 32 Hz",
    )
    psd_ax.axvline(14.0, color=DOWNSAMPLED_COLOR, linestyle="--", linewidth=0.8)
    psd_ax.text(13.7, psd_ax.get_ylim()[1], "14 Hz", color=DOWNSAMPLED_COLOR, ha="right", va="top")
    psd_ax.set_xlim(0, 16)
    psd_ax.set_xlabel("Frequency (Hz)")
    psd_ax.set_ylabel(r"PSD (dB g$^2$ Hz$^{-1}$)")
    psd_ax.set_title("All records and channels: median [IQR]")
    psd_ax.legend(loc="lower left")
    psd_ax.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.65)
    add_panel_label(psd_ax, "e")

    frequencies, response = signal.freqz(coefficients, worN=16384, fs=64)
    response_db = 20 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
    response_ax.plot(frequencies, response_db, color=FILTERED_COLOR, linewidth=1.25)
    response_ax.axvline(14.0, color=DOWNSAMPLED_COLOR, linestyle="--", linewidth=0.8)
    response_ax.axvline(16.0, color=FOG_COLOR, linestyle=":", linewidth=0.9)
    response_ax.axhline(-6.0, color=RAW_COLOR, linestyle=":", linewidth=0.65)
    response_ax.scatter(
        [14.0, 16.0],
        [np.interp(14.0, frequencies, response_db), np.interp(16.0, frequencies, response_db)],
        color=[DOWNSAMPLED_COLOR, FOG_COLOR],
        s=14,
        zorder=3,
    )
    response_ax.text(13.7, -6.0, "−6.02 dB", color=DOWNSAMPLED_COLOR, ha="right", va="bottom")
    response_ax.text(15.7, -55.0, "−55.95 dB", color=FOG_COLOR, ha="right", va="bottom")
    response_ax.text(
        16.35,
        -31.0,
        "16 Hz\nnew Nyquist",
        color=FOG_COLOR,
        ha="left",
        va="center",
    )
    response_ax.set_xlim(0, 32)
    response_ax.set_ylim(-90, 5)
    response_ax.set_xlabel("Frequency (Hz)")
    response_ax.set_ylabel("Gain (dB)")
    response_ax.set_title("65-tap FIR response")
    response_ax.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.65)
    add_panel_label(response_ax, "f")

    fig.text(
        0.10,
        0.992,
        (
            f"64→32 Hz preprocessing validation | {event['record_id']}, "
            f"FoG event {event['event_id']} ({float(event['duration_sec']):.2f} s)"
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )
    fig.text(
        0.98,
        0.992,
        "Shading: FoG interval",
        ha="right",
        va="top",
        fontsize=6.5,
        color=FOG_COLOR,
    )
    save_figure(fig, output_dir / "daphnet_64hz_vs_32hz_comparison")
    plt.close(fig)


def make_all_channels_figure(
    output_dir: Path,
    event: dict[str, Any],
    x_raw: np.ndarray,
    x_32hz: np.ndarray,
    y_raw: np.ndarray,
    segment_seconds: float,
) -> None:
    center = float(event["center_time_sec"])
    start_time = center - segment_seconds / 2.0
    end_time = center + segment_seconds / 2.0
    t_raw = np.arange(len(x_raw)) / 64.0
    t_32hz = np.arange(len(x_32hz)) / 32.0
    raw_mask = (t_raw >= start_time) & (t_raw <= end_time)
    processed_mask = (t_32hz >= start_time) & (t_32hz <= end_time)
    intervals = fog_intervals(y_raw, 64.0, start_time, end_time)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.15, 5.8),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.91, wspace=0.28, hspace=0.30)
    for channel, ax in enumerate(axes.flat):
        add_fog_spans(ax, intervals)
        ax.plot(
            t_raw[raw_mask],
            x_raw[raw_mask, channel],
            color=RAW_COLOR,
            linewidth=0.6,
            alpha=0.56,
            label="Raw 64 Hz" if channel == 0 else None,
        )
        ax.plot(
            t_32hz[processed_mask],
            x_32hz[processed_mask, channel],
            color=FILTERED_COLOR,
            linewidth=0.9,
            label="Filtered + 32 Hz" if channel == 0 else None,
        )
        ax.set_title(CHANNEL_NAMES[channel], loc="left", pad=2.0)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.60)
        if channel % 3 == 0:
            ax.set_ylabel("Acceleration (g)")
        if channel >= 6:
            ax.set_xlabel("Time (s)")
        add_panel_label(ax, chr(ord("a") + channel))
    axes.flat[0].legend(loc="upper right")
    fig.suptitle(
        f"All nine channels | {event['record_id']}, representative FoG segment",
        x=0.085,
        y=0.975,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "daphnet_64hz_vs_32hz_all_channels")
    plt.close(fig)


def write_source_data(
    output_dir: Path,
    event: dict[str, Any],
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_filtered: np.ndarray,
    x_32hz: np.ndarray,
    y_32hz: np.ndarray,
    psd_rows: Sequence[dict[str, Any]],
    coefficients: np.ndarray,
    segment_seconds: float,
) -> None:
    center = float(event["center_time_sec"])
    start_time = center - segment_seconds / 2.0
    end_time = center + segment_seconds / 2.0
    start_index = int(np.ceil(start_time * 64))
    end_index = int(np.floor(end_time * 64))
    segment_rows: list[dict[str, Any]] = []
    for source_index in range(start_index, end_index + 1):
        row: dict[str, Any] = {
            "time_sec": source_index / 64.0,
            "source_index_64hz": source_index,
            "label_64hz": int(y_raw[source_index]),
        }
        for channel, name in enumerate(CHANNEL_NAMES):
            key = name.lower().replace(" ", "_")
            row[f"raw64_{key}_g"] = float(x_raw[source_index, channel])
            row[f"fir_aligned64_{key}_g"] = float(x_filtered[source_index, channel])
        if source_index % 2 == 0:
            output_index = source_index // 2
            row["output_index_32hz"] = output_index
            row["label_32hz"] = int(y_32hz[output_index])
            for channel, name in enumerate(CHANNEL_NAMES):
                key = name.lower().replace(" ", "_")
                row[f"final32_{key}_g"] = float(x_32hz[output_index, channel])
        segment_rows.append(row)
    write_csv(output_dir / "source_data_selected_segment.csv", segment_rows)
    write_csv(output_dir / "source_data_psd_summary.csv", psd_rows)
    write_csv(
        output_dir / "source_data_fir_response.csv",
        [
            {
                "frequency_hz": float(f),
                "gain_db": float(gain),
            }
            for f, gain in zip(
                *(
                    lambda freq, response: (
                        freq,
                        20
                        * np.log10(
                            np.maximum(np.abs(response), np.finfo(np.float64).tiny)
                        ),
                    )
                )(*signal.freqz(coefficients, worN=16384, fs=64))
            )
        ],
    )


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    processed_root = args.processed_32hz.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    event = select_representative_event(source_root, args.segment_seconds)
    record_id = str(event["record_id"])
    x_raw, y_raw = load_record(source_root, record_id)
    x_32hz, y_32hz = load_record(processed_root, record_id)
    coefficients = preprocessing.design_fir()
    x_filtered = preprocessing.filter_and_align_64hz(x_raw, coefficients)
    _, psd_rows = calculate_psd_summary(source_root, processed_root)

    make_main_figure(
        output_dir,
        event,
        x_raw,
        y_raw,
        x_filtered,
        x_32hz,
        y_32hz,
        psd_rows,
        coefficients,
        args.segment_seconds,
    )
    make_all_channels_figure(
        output_dir,
        event,
        x_raw,
        x_32hz,
        y_raw,
        args.segment_seconds,
    )
    write_source_data(
        output_dir,
        event,
        x_raw,
        y_raw,
        x_filtered,
        x_32hz,
        y_32hz,
        psd_rows,
        coefficients,
        args.segment_seconds,
    )

    metadata = {
        "core_conclusion": (
            "The 14 Hz FIR suppresses near-Nyquist high-frequency content while preserving "
            "low-frequency gait morphology and FoG label timing on the 32 Hz grid."
        ),
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "representative_event": event,
        "segment_seconds": args.segment_seconds,
        "psd_scope": {
            "records": len(read_csv(source_root / "manifest.csv")),
            "channels_per_record": 9,
            "curve_count_per_method": int(psd_rows[0]["curve_count_per_method"]),
            "summary": "median and interquartile range",
        },
        "formats": ["png", "svg", "pdf", "tiff"],
        "integrity_notes": [
            "The representative event was selected by a deterministic median-duration rule.",
            "The PSD summary includes all 35 records and all nine channels.",
            "No smoothing beyond the specified FIR preprocessing was added to time-domain traces.",
            "FoG shading is derived from the original 64 Hz label array.",
        ],
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
