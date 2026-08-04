"""Plot Daphnet spectra at true 0.5 Hz and 0.25 Hz resolutions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


DEFAULT_INPUT = Path(
    "dataset/processed/daphnet_loso_3class_prefog2_win1_stride0p5/windows.npz"
)
DEFAULT_OUTPUT_DIR = Path("outputs/figures")
BASE_WINDOW_SAMPLES = 64
BASE_STRIDE_SAMPLES = 32
TRUNK_CHANNELS = slice(6, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    parser.add_argument("--max-frequency", type=float, default=10.0)
    return parser.parse_args()


def rebuild_pure_state_windows(
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    file_id: np.ndarray,
    start_sample: np.ndarray,
    target_samples: int,
) -> dict[tuple[str, int], np.ndarray]:
    """Rebuild longer windows from overlapping 1 s windows in one record."""
    if target_samples < BASE_WINDOW_SAMPLES:
        raise ValueError("target_samples must be at least 64")
    if (target_samples - BASE_WINDOW_SAMPLES) % BASE_STRIDE_SAMPLES:
        raise ValueError("target_samples is incompatible with the 32-sample stride")

    required = 1 + (target_samples - BASE_WINDOW_SAMPLES) // BASE_STRIDE_SAMPLES
    output_stride = target_samples // 2
    if output_stride % BASE_STRIDE_SAMPLES:
        raise ValueError("A 50% output overlap must align with the base stride")
    index_stride = output_stride // BASE_STRIDE_SAMPLES
    rebuilt: defaultdict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for current_file in np.unique(file_id):
        indices = np.flatnonzero(file_id == current_file)
        indices = indices[np.argsort(start_sample[indices])]
        for offset in range(0, len(indices) - required + 1, index_stride):
            selected = indices[offset : offset + required]
            if not np.all(np.diff(start_sample[selected]) == BASE_STRIDE_SAMPLES):
                continue
            label = int(y[selected[0]])
            if label not in (0, 2) or not np.all(y[selected] == label):
                continue
            parts = [x[selected[0], :, TRUNK_CHANNELS]]
            parts.extend(x[index, -BASE_STRIDE_SAMPLES:, TRUNK_CHANNELS] for index in selected[1:])
            long_window = np.concatenate(parts, axis=0)
            if long_window.shape != (target_samples, 3):
                raise RuntimeError(f"Unexpected rebuilt shape: {long_window.shape}")
            rebuilt[(str(subject[selected[0]]), label)].append(long_window)

    return {
        key: np.asarray(windows, dtype=np.float64)
        for key, windows in rebuilt.items()
    }


def relative_psd(
    windows: np.ndarray,
    sampling_rate: float,
    max_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    frequency, psd = welch(
        windows,
        fs=sampling_rate,
        axis=1,
        nperseg=windows.shape[1],
        noverlap=0,
        detrend="constant",
        window="hann",
        scaling="density",
    )
    mean_psd = psd.mean(axis=(0, 2))
    display = (frequency >= 0.5) & (frequency <= max_frequency)
    band_power = np.trapz(mean_psd[display], frequency[display])
    return frequency, mean_psd / max(float(band_power), np.finfo(float).tiny)


def make_figure(
    rebuilt: dict[tuple[str, int], np.ndarray],
    target_samples: int,
    sampling_rate: float,
    max_frequency: float,
    output: Path,
) -> None:
    subjects = [f"S{index:02d}" for index in range(1, 11)]
    states = (("non-FOG", 0, "#247A4A"), ("FOG", 2, "#C43C39"))
    duration = target_samples / sampling_rate
    resolution = sampling_rate / target_samples

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(5, 2, figsize=(14, 16), sharex=True, sharey=True)
    for ax, subject_id in zip(axes.ravel(), subjects):
        for state_name, label, color in states:
            windows = rebuilt.get((subject_id, label))
            if windows is None or len(windows) == 0:
                continue
            frequency, power = relative_psd(windows, sampling_rate, max_frequency)
            keep = (frequency >= 0.5) & (frequency <= max_frequency)
            ax.plot(
                frequency[keep],
                10.0 * np.log10(np.maximum(power[keep], 1e-12)),
                color=color,
                linewidth=2.0,
                label=f"{state_name} (n={len(windows):,})",
            )
        if (subject_id, 2) not in rebuilt:
            ax.text(
                0.97,
                0.74,
                "No annotated\nFOG windows",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="#A63230",
                fontsize=9.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "fc": "#FFF4F3", "ec": "#D99A97"},
            )
        ax.axvspan(0.5, 3.0, color="#4BAE4F", alpha=0.07)
        ax.axvspan(3.0, 8.0, color="#E4514F", alpha=0.055)
        ax.axvline(3.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(subject_id, fontsize=13, fontweight="bold")
        ax.set_xlim(0.5, max_frequency)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, color="#D9DEE5", linewidth=0.6, alpha=0.75)

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative PSD (dB/Hz)")

    fig.suptitle(
        "Daphnet S01–S10: trunk acceleration spectra\n"
        f"{duration:g} s windows at 64 Hz — frequency resolution Δf = {resolution:g} Hz",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Only continuous, same-record, same-label windows are used. "
        "Each curve is normalized by its 0.5–10 Hz power; PRE_FOG is excluded. "
        "S04 and S10 have no annotated FOG windows.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.035, 0.027, 0.995, 0.972), h_pad=1.4, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output.resolve()}")


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        x = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"])
        subject = np.asarray(data["subject"])
        file_id = np.asarray(data["file_id"])
        start_sample = np.asarray(data["start_sample"])

    for target_samples, resolution_tag in ((128, "0p5Hz"), (256, "0p25Hz")):
        rebuilt = rebuild_pure_state_windows(
            x, y, subject, file_id, start_sample, target_samples
        )
        output = args.output_dir / (
            f"daphnet_S01_S10_nonfog_fog_spectra_{resolution_tag}.png"
        )
        make_figure(
            rebuilt,
            target_samples,
            args.sampling_rate,
            args.max_frequency,
            output,
        )


if __name__ == "__main__":
    main()
