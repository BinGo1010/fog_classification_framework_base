"""Compare trunk Freeze Index between Daphnet non-FOG and FOG states."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


DEFAULT_INPUT = Path(
    "dataset/processed/daphnet_loso_3class_prefog2_win1_stride0p5/windows.npz"
)
DEFAULT_OUTPUT = Path("outputs/figures/daphnet_S01_S10_freeze_index_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    return parser.parse_args()


def ensemble_freeze_index(windows: np.ndarray, sampling_rate: float) -> float:
    """Return pooled 3–8 Hz / 0.5–3 Hz power across windows and trunk axes."""
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
    locomotor_power = mean_psd[(frequency >= 0.5) & (frequency < 3.0)].sum()
    freeze_power = mean_psd[(frequency >= 3.0) & (frequency <= 8.0)].sum()
    return float(freeze_power / max(locomotor_power, np.finfo(float).tiny))


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        x = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"])
        subject = np.asarray(data["subject"])
        class_names = tuple(str(item) for item in data["class_names"])

    if class_names != ("NORMAL", "PRE_FOG", "FOG"):
        raise ValueError(f"Unexpected class mapping: {class_names}")

    subjects = [f"S{index:02d}" for index in range(1, 11)]
    values = np.full((len(subjects), 2), np.nan, dtype=float)
    counts = np.zeros((len(subjects), 2), dtype=int)
    for subject_index, subject_id in enumerate(subjects):
        for state_index, label in enumerate((0, 2)):
            selected = (subject == subject_id) & (y == label)
            counts[subject_index, state_index] = int(selected.sum())
            if selected.any():
                values[subject_index, state_index] = ensemble_freeze_index(
                    x[selected, :, 6:9], args.sampling_rate
                )

    x_position = np.arange(len(subjects), dtype=float)
    width = 0.36
    colors = ("#247A4A", "#C43C39")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (index_ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": (2.1, 1.0)},
    )

    for state_index, (state_name, color, offset) in enumerate(
        (("non-FOG", colors[0], -width / 2), ("FOG", colors[1], width / 2))
    ):
        bar_values = np.nan_to_num(values[:, state_index], nan=0.0)
        bars = index_ax.bar(
            x_position + offset,
            bar_values,
            width,
            color=color,
            alpha=0.92,
            label=state_name,
            edgecolor="white",
            linewidth=0.7,
        )
        for subject_index, bar in enumerate(bars):
            value = values[subject_index, state_index]
            if np.isfinite(value):
                index_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.18,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=color,
                    fontweight="bold",
                )
            elif state_index == 1:
                index_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.25,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=color,
                    fontweight="bold",
                )

    index_ax.set_xticks(x_position, subjects)
    index_ax.set_ylabel("Freeze Index (3–8 Hz / 0.5–3 Hz power)")
    index_ax.set_title("Freeze Index by subject and gait state", fontweight="bold")
    index_ax.set_ylim(0, max(12.0, float(np.nanmax(values)) * 1.18))
    index_ax.legend(loc="upper right", ncols=2)
    index_ax.grid(axis="x", visible=False)

    valid = np.isfinite(values[:, 1])
    ratio = values[valid, 1] / values[valid, 0]
    ratio_subjects = np.asarray(subjects)[valid]
    ratio_bars = ratio_ax.bar(
        ratio_subjects,
        ratio,
        color="#4B66A1",
        width=0.62,
        alpha=0.92,
        edgecolor="white",
    )
    ratio_ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
    ratio_ax.set_ylabel("FOG / non-FOG FI ratio")
    ratio_ax.set_xlabel("Subject")
    ratio_ax.set_title(
        "Within-subject increase during FOG "
        f"(median {np.median(ratio):.2f}×)",
        fontweight="bold",
    )
    ratio_ax.set_ylim(0, max(4.5, float(ratio.max()) * 1.18))
    ratio_ax.grid(axis="x", visible=False)
    for bar, value in zip(ratio_bars, ratio):
        ratio_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#354B7C",
        )

    fig.suptitle(
        "Daphnet S01–S10: trunk Freeze Index comparison (64 Hz)",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        "Welch PSD pooled over all one-second windows and three trunk axes. "
        "PRE_FOG is excluded. S04 and S10 have no annotated FOG windows.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.035, 0.04, 0.995, 0.955), h_pad=2.1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("subject,nonfog_fi,fog_fi,fog_nonfog_ratio,nonfog_n,fog_n")
    for subject_id, state_values, state_counts in zip(subjects, values, counts):
        current_ratio = state_values[1] / state_values[0]
        print(
            f"{subject_id},{state_values[0]:.6f},{state_values[1]:.6f},"
            f"{current_ratio:.6f},{state_counts[0]},{state_counts[1]}"
        )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
