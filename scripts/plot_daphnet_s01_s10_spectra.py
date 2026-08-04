"""Plot subject-wise trunk spectra for Daphnet non-FOG and FOG windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


DEFAULT_INPUT = Path(
    "dataset/processed/daphnet_loso_3class_prefog2_win1_stride0p5/windows.npz"
)
DEFAULT_OUTPUT = Path("outputs/figures/daphnet_S01_S10_nonfog_fog_spectra.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    parser.add_argument("--max-frequency", type=float, default=10.0)
    return parser.parse_args()


def ensemble_relative_psd(
    windows: np.ndarray,
    sampling_rate: float,
    max_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average three-axis, per-window Welch PSD and normalize its display band."""
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
    relative_psd = mean_psd / max(float(band_power), np.finfo(float).tiny)
    return frequency, relative_psd


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        x = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"])
        subject = np.asarray(data["subject"])
        class_names = tuple(str(item) for item in data["class_names"])

    if class_names != ("NORMAL", "PRE_FOG", "FOG"):
        raise ValueError(f"Unexpected class mapping: {class_names}")
    if x.ndim != 3 or x.shape[2] < 9:
        raise ValueError(f"Expected [window, time, >=9 channels], got {x.shape}")

    subjects = [f"S{index:02d}" for index in range(1, 11)]
    state_specs = (("non-FOG", 0, "#247A4A"), ("FOG", 2, "#C43C39"))
    spectra: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = {}
    for subject_id in subjects:
        for state_name, label, _ in state_specs:
            selected = (subject == subject_id) & (y == label)
            count = int(selected.sum())
            if count == 0:
                spectra[(subject_id, state_name)] = (
                    np.asarray([], dtype=float),
                    np.asarray([], dtype=float),
                    0,
                )
                continue
            frequency, relative_psd = ensemble_relative_psd(
                x[selected, :, 6:9], args.sampling_rate, args.max_frequency
            )
            spectra[(subject_id, state_name)] = (frequency, relative_psd, count)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(5, 2, figsize=(14, 16), sharex=True, sharey=True)
    for ax, subject_id in zip(axes.ravel(), subjects):
        for state_name, _, color in state_specs:
            frequency, relative_psd, count = spectra[(subject_id, state_name)]
            if count == 0:
                continue
            keep = (frequency >= 0.5) & (frequency <= args.max_frequency)
            ax.plot(
                frequency[keep],
                10.0 * np.log10(np.maximum(relative_psd[keep], 1e-12)),
                color=color,
                linewidth=2.1,
                label=f"{state_name} (n={count:,})",
            )
        if spectra[(subject_id, "FOG")][2] == 0:
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
        ax.set_xlim(0.5, args.max_frequency)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, color="#D9DEE5", linewidth=0.6, alpha=0.75)

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative PSD (dB/Hz)")

    fig.suptitle(
        "Daphnet S01–S10: trunk acceleration spectra\n"
        "non-FOG vs FOG (three-axis ensemble Welch PSD)",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Each curve is normalized by its 0.5–10 Hz power. "
        "Green shading: locomotor band (0.5–3 Hz); red shading: freezing band (3–8 Hz). "
        "PRE_FOG windows are excluded; S04 and S10 have no annotated FOG windows.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.035, 0.027, 0.995, 0.972), h_pad=1.4, w_pad=1.1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
