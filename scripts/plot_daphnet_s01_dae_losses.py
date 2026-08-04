#!/usr/bin/env python
"""Plot the recorded S01 DAE training and validation loss curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_dae_tcnm_seed42"
            / "dae_training_history.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "daphnet_s01_dae_tcnm_seed42"
        ),
    )
    return parser.parse_args()


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty training history: {path}")
    keys = (
        "epoch",
        "train_total_loss",
        "validation_total_loss",
    )
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in keys
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#d6dbe3", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Combined loss", fontsize=12)
    axis.tick_params(labelsize=10)


def save_training_plot(history: dict[str, np.ndarray], path: Path) -> None:
    epochs = history["epoch"]
    values = history["train_total_loss"]
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    axis.plot(epochs, values, color="#2468b4", linewidth=2.2)
    axis.scatter(
        [epochs[0], epochs[-1]],
        [values[0], values[-1]],
        color=["#e07a1f", "#18864b"],
        s=42,
        zorder=3,
    )
    axis.annotate(
        f"Epoch 1: {values[0]:.6f}",
        (epochs[0], values[0]),
        xytext=(12, -4),
        textcoords="offset points",
        fontsize=10,
    )
    axis.annotate(
        f"Epoch {int(epochs[-1])}: {values[-1]:.6f}",
        (epochs[-1], values[-1]),
        xytext=(-132, 14),
        textcoords="offset points",
        fontsize=10,
    )
    axis.set_title(
        "S01 DAE Training Combined Loss (Raw Epoch Values)",
        fontsize=14,
        pad=12,
    )
    axis.set_xlim(float(epochs[0]), float(epochs[-1]))
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_validation_plot(history: dict[str, np.ndarray], path: Path) -> None:
    epochs = history["epoch"]
    values = history["validation_total_loss"]
    best_index = int(np.argmin(values))
    best_epoch = int(epochs[best_index])
    best_value = float(values[best_index])
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    axis.plot(epochs, values, color="#9b4b9d", linewidth=2.0)
    axis.scatter(
        [best_epoch],
        [best_value],
        color="#18864b",
        edgecolor="white",
        linewidth=1.2,
        s=70,
        zorder=4,
        label=f"Best epoch {best_epoch}: {best_value:.6f}",
    )
    axis.axvline(
        best_epoch,
        color="#18864b",
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
    )
    axis.annotate(
        f"Best: epoch {best_epoch}\nloss = {best_value:.6f}",
        (best_epoch, best_value),
        xytext=(-150, 42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#18864b"},
        fontsize=10,
    )
    axis.set_title(
        "S01 DAE Clean-Validation Combined Loss (Raw Epoch Values)",
        fontsize=14,
        pad=12,
    )
    axis.set_xlim(float(epochs[0]), float(epochs[-1]))
    axis.legend(frameon=False, loc="upper right", fontsize=10)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    history = read_history(args.history.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "dae_training_loss.png"
    validation_path = output_dir / "dae_validation_loss.png"
    save_training_plot(history, training_path)
    save_validation_plot(history, validation_path)
    print(training_path)
    print(validation_path)


if __name__ == "__main__":
    main()
