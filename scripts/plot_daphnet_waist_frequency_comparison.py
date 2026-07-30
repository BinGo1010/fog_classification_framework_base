"""Visualize frequency-domain differences in Daphnet waist FOG/non-FOG data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, welch


FS = 64.0
WINDOW_SAMPLES = 512
FOG_START = 47552
WAIST_CHANNELS = slice(6, 9)
RECORD = Path(
    "dataset/1.Daphnet Freezing of Gait Dataset/"
    "processed/records/S01_seg001.npz"
)
OUTPUT = Path("outputs/figures/daphnet_S01_waist_frequency_features.png")


def pooled_freeze_index(values: np.ndarray) -> float:
    """Return pooled three-axis 3–8 Hz / 0.5–3 Hz power."""
    centered = values - values.mean(axis=0, keepdims=True)
    frequency = np.fft.rfftfreq(len(values), d=1.0 / FS)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    locomotor = power[(frequency >= 0.5) & (frequency < 3.0)].sum()
    freeze = power[(frequency >= 3.0) & (frequency <= 8.0)].sum()
    return float(freeze / max(locomotor, 1e-12))


def sliding_freeze_index(
    values: np.ndarray,
    window_samples: int = 128,
    step_samples: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(values) - window_samples + 1, step_samples)
    centers = (starts + window_samples / 2) / FS
    scores = np.asarray(
        [
            pooled_freeze_index(values[start : start + window_samples])
            for start in starts
        ]
    )
    return centers, scores


def pooled_welch(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency, axis_psd = welch(
        values,
        fs=FS,
        axis=0,
        nperseg=256,
        noverlap=192,
        detrend="constant",
        scaling="density",
    )
    return frequency, axis_psd.mean(axis=1)


def pooled_spectrogram(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_spectrograms = []
    for channel in range(values.shape[1]):
        frequency, time_s, power = spectrogram(
            values[:, channel],
            fs=FS,
            nperseg=128,
            noverlap=112,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        axis_spectrograms.append(power)
    pooled = np.mean(axis_spectrograms, axis=0)
    return frequency, time_s, 10.0 * np.log10(np.maximum(pooled, 1e-12))


def main() -> None:
    with np.load(RECORD) as record:
        x = np.asarray(record["x"], dtype=np.float64)
        y = np.asarray(record["y_binary"], dtype=np.int8)

    nonfog = x[FOG_START - WINDOW_SAMPLES : FOG_START, WAIST_CHANNELS]
    fog = x[FOG_START : FOG_START + WINDOW_SAMPLES, WAIST_CHANNELS]
    if np.any(y[FOG_START - WINDOW_SAMPLES : FOG_START] != 0):
        raise ValueError("Selected non-FOG segment is not label-pure.")
    if np.any(y[FOG_START : FOG_START + WINDOW_SAMPLES] != 1):
        raise ValueError("Selected FOG segment is not label-pure.")

    segments = {"non-FOG": nonfog, "FOG": fog}
    state_colors = {"non-FOG": "#2E7D32", "FOG": "#C62828"}
    whole_fi = {state: pooled_freeze_index(values) for state, values in segments.items()}
    spectra = {state: pooled_welch(values) for state, values in segments.items()}
    spectrograms = {
        state: pooled_spectrogram(values) for state, values in segments.items()
    }

    all_spectrogram_values = np.concatenate(
        [item[2].ravel() for item in spectrograms.values()]
    )
    vmin, vmax = np.percentile(all_spectrogram_values, [5, 99])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(
        left=0.07,
        right=0.94,
        top=0.84,
        bottom=0.09,
        hspace=0.36,
        wspace=0.23,
    )

    image = None
    for ax, state in zip(axes[0], ("non-FOG", "FOG")):
        frequency, time_s, power_db = spectrograms[state]
        keep = frequency <= 10.0
        image = ax.pcolormesh(
            time_s,
            frequency[keep],
            power_db[keep],
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhspan(0.5, 3.0, color="#43A047", alpha=0.10)
        ax.axhspan(3.0, 8.0, color="#E53935", alpha=0.10)
        ax.axhline(3.0, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_title(
            f"Waist three-axis spectrogram — {state}",
            color=state_colors[state],
            fontweight="bold",
        )
        ax.set_xlabel("Time within selected window (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 10)
        ax.grid(False)

    colorbar_ax = fig.add_axes([0.955, 0.535, 0.012, 0.305])
    fig.colorbar(image, cax=colorbar_ax, label="Mean PSD (dB/Hz)")

    psd_ax = axes[1, 0]
    psd_ax.axvspan(0.5, 3.0, color="#43A047", alpha=0.13, label="Walking band")
    psd_ax.axvspan(3.0, 8.0, color="#E53935", alpha=0.10, label="Freeze band")
    for state in ("non-FOG", "FOG"):
        frequency, power = spectra[state]
        keep = (frequency >= 0.25) & (frequency <= 10.0)
        psd_ax.semilogy(
            frequency[keep],
            power[keep],
            color=state_colors[state],
            linewidth=2.0,
            label=state,
        )
    psd_ax.axvline(3.0, color="#616161", linestyle="--", linewidth=1.0)
    psd_ax.set_title("Average three-axis power spectrum", fontweight="bold")
    psd_ax.set_xlabel("Frequency (Hz)")
    psd_ax.set_ylabel("Power spectral density (g²/Hz)")
    psd_ax.set_xlim(0.25, 10.0)
    psd_ax.legend(fontsize=9, ncol=2)

    fi_ax = axes[1, 1]
    for state in ("non-FOG", "FOG"):
        time_s, fi = sliding_freeze_index(segments[state])
        fi_ax.plot(
            time_s,
            fi,
            color=state_colors[state],
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=state,
        )
        fi_ax.axhline(
            whole_fi[state],
            color=state_colors[state],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
    fi_ax.set_yscale("log")
    fi_ax.set_title("Sliding Freeze Index (2-second windows)", fontweight="bold")
    fi_ax.set_xlabel("Time within selected window (s)")
    fi_ax.set_ylabel("Freeze / walking band power ratio")
    fi_ax.set_xlim(0, 8)
    fi_ax.legend()

    ratio = whole_fi["FOG"] / whole_fi["non-FOG"]
    fig.suptitle(
        "Why raw waist signals look similar: the difference is clearer in frequency space\n"
        f"Subject S01 — full-window Freeze Index: "
        f"non-FOG {whole_fi['non-FOG']:.2f}, FOG {whole_fi['FOG']:.2f} "
        f"({ratio:.2f}× higher)",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Green band: locomotor rhythm (0.5–3 Hz). Red band: freezing-related "
        "rhythm (3–8 Hz). Dashed lines in the FI panel show each 8-second mean.",
        ha="center",
        fontsize=10,
        color="#424242",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"non-FOG Freeze Index: {whole_fi['non-FOG']:.6f}")
    print(f"FOG Freeze Index: {whole_fi['FOG']:.6f}")
    print(f"FOG/non-FOG ratio: {ratio:.6f}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
