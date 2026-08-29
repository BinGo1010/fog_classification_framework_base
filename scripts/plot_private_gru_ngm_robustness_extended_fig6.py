#!/usr/bin/env python3
"""Create the publication figures for the extended Private robustness test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_private_gru_ngm_robustness_fig6 as plot


plot.DEFAULT_RESULTS = (
    REPO_ROOT
    / "outputs"
    / "private_gru_ngm_robustness_matched_tcn"
    / "robustness_extended"
)
plot.DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "private_gru_ngm_robustness_extended_figures"
)

plot.EXPECTED_GAUSSIAN_LEVELS = (
    0.0,
    0.02,
    0.04,
    0.08,
    0.12,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
)
plot.EXPECTED_MASK_LEVELS = (
    0.0,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
)
plot.GAUSSIAN_X_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
plot.MASK_X_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
plot.MASK_X_TICK_LABELS = ("0", "20", "40", "60", "80", "100")
plot.SHOW_PANEL_LABELS = True

plot.PLOT_TEXT.update(
    {
        "gaussian_title": "",
        "mask_title": "",
        "gaussian_xlabel": (
            r"Gaussian noise standard deviation, $\sigma_{\mathrm{test}}$"
        ),
        "mask_xlabel": r"Temporal masking ratio, $\rho_{\mathrm{mask}}$ (%)",
        "ylabel": "Average precision (AP)",
        "none_legend": "Without augmentation",
        "gaussian_mask_legend": "Gaussian + masking augmentation",
        "legend_title": "",
    }
)

plot.PLOT_STYLE.update(
    {
        "figure_size": (4.2, 3.1),
        "combined_figure_size": (4.2, 6.0),
        "font_size": 8.0,
        "axis_label_size": 8.5,
        "title_size": 9.0,
        "panel_label_size": 9.0,
        "tick_label_size": 8.0,
        "legend_font_size": 7.5,
        "legend_title_size": 7.5,
        "gaussian_legend_location": "upper left",
        "mask_legend_location": "upper left",
        "legend_frame": False,
        "line_width": 1.7,
        "marker_size": 4.4,
        "marker_edge_width": 1.0,
        "gaussian_band_alpha": 0.10,
        "mask_band_alpha": 0.10,
        "grid_alpha": 0.25,
        "y_limits": (0.18, 0.63),
        "y_ticks": np.arange(0.20, 0.61, 0.10),
        "gaussian_y_limits": (0.50, 0.62),
        "gaussian_y_ticks": np.arange(0.50, 0.621, 0.02),
        "mask_y_limits": (0.18, 0.63),
        "mask_y_ticks": np.arange(0.20, 0.61, 0.10),
        "gaussian_x_limits": (-0.025, 1.025),
        "mask_x_limits": (-0.025, 1.025),
        "gaussian_none_curve_offset": 0.0,
        "gaussian_gaussian_mask_curve_offset": 0.0,
        "mask_none_curve_offset": 0.0,
        "mask_gaussian_mask_curve_offset": 0.0,
    }
)


if __name__ == "__main__":
    plot.main()
