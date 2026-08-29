#!/usr/bin/env python3
"""Plot the two Private GRU-NGM robustness figures from aggregated CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = Path(
    r"E:\fog-merged\outputs\private_gru_ngm_robustness_matched_tcn"
)
DEFAULT_RESULTS = (
    LOCAL_RESULTS
    if LOCAL_RESULTS.is_dir()
    else REPO_ROOT / "outputs" / "private_gru_ngm_robustness_matched_tcn"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "private_gru_ngm_robustness_figures"
EXPECTED_GAUSSIAN_LEVELS = (0.0, 0.02, 0.04, 0.08, 0.12)
EXPECTED_MASK_LEVELS = (0.0, 0.025, 0.05, 0.10, 0.15)
GAUSSIAN_X_TICKS = (0.0, 0.02, 0.04, 0.08, 0.12)
MASK_X_TICKS = (0.0, 0.025, 0.05, 0.10, 0.15)
MASK_X_TICK_LABELS = ("0", "2.5", "5", "10", "15")
SHOW_PANEL_LABELS = False

# ============================================================================
# USER-EDITABLE PLOT SETTINGS
# Modify the text, font sizes, legend, axes, colors, and line styles here.
# Empty title strings remove the titles from the exported figures.
# ============================================================================
PLOT_TEXT = {
    "gaussian_title": "",
    "mask_title": "",
    "gaussian_xlabel": (
        r"Gaussian Noise Standard Deviation, $\sigma_{\mathrm{test}}$"
    ),
    "mask_xlabel": r"Temporal Masking Ratio, $\rho_{\mathrm{mask}}$ (%)",
    "ylabel": "Average precision (AP)",
    "none_legend": "Gaussian + Masking Augmentation",
    "gaussian_mask_legend": "Without Augmentation",
    "legend_title": "",
}

PLOT_STYLE = {
    # 单张图尺寸：(宽, 高)，单位为英寸。
    "figure_size": (6.2, 4.2),
    # 两张图上下组合后的尺寸：(宽, 高)，单位为英寸。
    "combined_figure_size": (6.2, 8.0),
    # 全局默认字体大小，单位为 pt。
    "font_size": 12.0,
    # 横轴和纵轴标题字体大小，单位为 pt。
    "axis_label_size": 15,
    # 分面标题字体大小；当前标题为空，因此暂不显示。
    "title_size": 9.5,
    # 合并图左上角分面编号的字体大小。
    "panel_label_size": 9.5,
    # 分面标题字重，可设为 "normal" 或 "bold"。
    "title_weight": "bold",
    # 横纵坐标刻度数字的字体大小，单位为 pt。
    "tick_label_size": 15,
    # 图例中两条曲线名称的字体大小，单位为 pt。
    "legend_font_size": 12,
    # 图例标题字体大小；legend_title 为空时不显示标题。
    "legend_title_size": 12,
    # Gaussian 单图的图例位置，使用 Matplotlib 的位置名称。
    "gaussian_legend_location": "upper left",
    # Temporal masking 单图的图例位置。
    "mask_legend_location": "upper left",
    # 是否绘制图例边框。
    "legend_frame": False,
    # 图例背景透明度：1.0 为完全不透明，0.0 为完全透明。
    "legend_frame_alpha": 0.95,
    # 图例边框颜色。
    "legend_edge_color": "#999999",
    # 均值曲线宽度。
    "line_width": 1.8,
    # 数据点标记大小。
    "marker_size": 4.8,
    # 数据点标记边缘宽度。
    "marker_edge_width": 1.2,
    # Gaussian 图误差带透明度；数值越小，阴影越淡。
    "gaussian_band_alpha": 0.10,
    # Temporal masking 图误差带透明度。
    "mask_band_alpha": 0.10,
    # 网格线透明度。
    "grid_alpha": 0.35,
    # 两张图共用的纵轴显示范围：(最小值, 最大值)。
    "y_limits": (0.54, 0.63),
    # 纵轴刻度位置：从 0.54 到 0.63，间隔为 0.02。
    "y_ticks": np.arange(0.54, 0.631, 0.02),
    # Gaussian 图横轴显示范围。
    "gaussian_x_limits": (-0.004, 0.124),
    # Temporal masking 图横轴显示范围。
    "mask_x_limits": (-0.005, 0.155),
    # Gaussian 图中 none（无扰动训练）曲线的整体纵向偏移量；正值上移，负值下移。
    "gaussian_none_curve_offset": 0.0,
    # Gaussian 图中 gaussian_mask（Gaussian + Mask 训练）曲线的整体纵向偏移量。
    "gaussian_gaussian_mask_curve_offset": 0.0,
    # Mask 图中 none（无扰动训练）曲线的整体纵向偏移量；正值上移，负值下移。
    "mask_none_curve_offset": 0.0,
    # Mask 图中 gaussian_mask（Gaussian + Mask 训练）曲线的整体纵向偏移量。
    "mask_gaussian_mask_curve_offset": 0.0,
}

COLORS = {
    "none": "#1f77b4",
    "gaussian_mask": "#2ca02c",
}
LINESTYLES = {
    "none": "--",
    "gaussian_mask": "-",
}
MARKERS = {
    "none": "o",
    "gaussian_mask": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--band",
        choices=("sem", "std"),
        default="sem",
        help="uncertainty band shown around the mean curve",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"non-finite values in column {name}")
    return values


def validate_levels(
    rows: list[dict[str, Any]],
    name: str,
    expected: tuple[float, ...],
) -> None:
    actual = numeric(rows, name)
    reference = np.asarray(expected, dtype=np.float64)
    if actual.shape != reference.shape or not np.allclose(
        actual, reference, rtol=0.0, atol=1e-12
    ):
        raise AssertionError(
            f"unexpected {name} grid: {actual.tolist()} != {reference.tolist()}"
        )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": PLOT_STYLE["font_size"],
            "axes.labelsize": PLOT_STYLE["axis_label_size"],
            "axes.titlesize": PLOT_STYLE["title_size"],
            "axes.titleweight": PLOT_STYLE["title_weight"],
            "xtick.labelsize": PLOT_STYLE["tick_label_size"],
            "ytick.labelsize": PLOT_STYLE["tick_label_size"],
            "legend.fontsize": PLOT_STYLE["legend_font_size"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def plot_curves(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    x_column: str,
    band: str,
    band_alpha: float,
    figure_key: str,
) -> None:
    x = numeric(rows, x_column)
    series = (
        (
            "none",
            PLOT_TEXT["none_legend"],
            "no_perturbation_ap_mean",
            f"no_perturbation_ap_{band}",
        ),
        (
            "gaussian_mask",
            PLOT_TEXT["gaussian_mask_legend"],
            "gaussian_mask_ap_mean",
            f"gaussian_mask_ap_{band}",
        ),
    )
    for series_key, label, mean_column, std_column in series:
        offset_key = f"{figure_key}_{series_key}_curve_offset"
        mean = numeric(rows, mean_column) + float(PLOT_STYLE[offset_key])
        std = numeric(rows, std_column)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[series_key],
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=COLORS[series_key],
            linestyle=LINESTYLES[series_key],
            linewidth=PLOT_STYLE["line_width"],
            marker=MARKERS[series_key],
            markersize=PLOT_STYLE["marker_size"],
            markerfacecolor="white",
            markeredgecolor=COLORS[series_key],
            markeredgewidth=PLOT_STYLE["marker_edge_width"],
            label=label,
            zorder=3,
        )


def decorate_axes(
    ax: plt.Axes,
    show_ylabel: bool = True,
    y_limits: tuple[float, float] | None = None,
    y_ticks: np.ndarray | tuple[float, ...] | None = None,
) -> None:
    if show_ylabel:
        ax.set_ylabel(PLOT_TEXT["ylabel"])
    limits = PLOT_STYLE["y_limits"] if y_limits is None else y_limits
    ticks = PLOT_STYLE["y_ticks"] if y_ticks is None else y_ticks
    ax.set_ylim(*limits)
    ax.set_yticks(ticks)
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        alpha=PLOT_STYLE["grid_alpha"],
    )
    ax.tick_params(direction="out", length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    return png, pdf, svg


def plot_gaussian(
    rows: list[dict[str, Any]], output_dir: Path, dpi: int, band: str
) -> tuple[Path, Path, Path]:
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figure_size"], constrained_layout=True
    )
    plot_curves(
        ax,
        rows,
        "sigma_test",
        band,
        PLOT_STYLE["gaussian_band_alpha"],
        "gaussian",
    )
    ax.set_xticks(GAUSSIAN_X_TICKS)
    ax.set_xlim(*PLOT_STYLE["gaussian_x_limits"])
    ax.set_xlabel(PLOT_TEXT["gaussian_xlabel"])
    if PLOT_TEXT["gaussian_title"]:
        ax.set_title(PLOT_TEXT["gaussian_title"])
    decorate_axes(
        ax,
        y_limits=PLOT_STYLE.get("gaussian_y_limits"),
        y_ticks=PLOT_STYLE.get("gaussian_y_ticks"),
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        borderaxespad=0.0,
        frameon=PLOT_STYLE["legend_frame"],
        framealpha=PLOT_STYLE["legend_frame_alpha"],
        edgecolor=PLOT_STYLE["legend_edge_color"],
        title=(
            PLOT_TEXT["legend_title"].format(band=band.upper())
            if PLOT_TEXT["legend_title"]
            else None
        ),
        title_fontsize=PLOT_STYLE["legend_title_size"],
    )
    paths = save_figure(
        fig, output_dir, "Fig6a_Private_Gaussian_Noise_Robustness", dpi
    )
    plt.close(fig)
    return paths


def plot_mask(
    rows: list[dict[str, Any]], output_dir: Path, dpi: int, band: str
) -> tuple[Path, Path, Path]:
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figure_size"], constrained_layout=True
    )
    plot_curves(
        ax,
        rows,
        "rho_mask",
        band,
        PLOT_STYLE["mask_band_alpha"],
        "mask",
    )
    ax.set_xticks(MASK_X_TICKS, MASK_X_TICK_LABELS)
    ax.set_xlim(*PLOT_STYLE["mask_x_limits"])
    ax.set_xlabel(PLOT_TEXT["mask_xlabel"])
    if PLOT_TEXT["mask_title"]:
        ax.set_title(PLOT_TEXT["mask_title"])
    decorate_axes(
        ax,
        y_limits=PLOT_STYLE.get("mask_y_limits"),
        y_ticks=PLOT_STYLE.get("mask_y_ticks"),
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        borderaxespad=0.0,
        frameon=PLOT_STYLE["legend_frame"],
        framealpha=PLOT_STYLE["legend_frame_alpha"],
        edgecolor=PLOT_STYLE["legend_edge_color"],
        title=(
            PLOT_TEXT["legend_title"].format(band=band.upper())
            if PLOT_TEXT["legend_title"]
            else None
        ),
        title_fontsize=PLOT_STYLE["legend_title_size"],
    )
    paths = save_figure(
        fig, output_dir, "Fig6b_Private_Temporal_Masking_Robustness", dpi
    )
    plt.close(fig)
    return paths


def plot_combined(
    gaussian_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
    band: str,
) -> tuple[Path, Path, Path]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=PLOT_STYLE["combined_figure_size"],
        sharey=True,
    )
    gaussian_ax, mask_ax = axes
    plot_curves(
        gaussian_ax,
        gaussian_rows,
        "sigma_test",
        band,
        PLOT_STYLE["gaussian_band_alpha"],
        "gaussian",
    )
    gaussian_ax.set_xticks(GAUSSIAN_X_TICKS)
    gaussian_ax.set_xlim(*PLOT_STYLE["gaussian_x_limits"])
    gaussian_ax.set_xlabel(PLOT_TEXT["gaussian_xlabel"])
    decorate_axes(gaussian_ax, show_ylabel=False)

    plot_curves(
        mask_ax,
        mask_rows,
        "rho_mask",
        band,
        PLOT_STYLE["mask_band_alpha"],
        "mask",
    )
    mask_ax.set_xticks(MASK_X_TICKS, MASK_X_TICK_LABELS)
    mask_ax.set_xlim(*PLOT_STYLE["mask_x_limits"])
    mask_ax.set_xlabel(PLOT_TEXT["mask_xlabel"])
    decorate_axes(mask_ax, show_ylabel=False)
    fig.supylabel(
        PLOT_TEXT["ylabel"],
        x=0.025,
        fontsize=PLOT_STYLE["axis_label_size"],
    )
    handles, labels = gaussian_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.15, 0.995),
        ncol=1,
        frameon=PLOT_STYLE["legend_frame"],
        framealpha=PLOT_STYLE["legend_frame_alpha"],
        edgecolor=PLOT_STYLE["legend_edge_color"],
        fontsize=PLOT_STYLE["legend_font_size"],
    )
    if SHOW_PANEL_LABELS:
        for ax, label in zip(axes, ("a", "b")):
            ax.text(
                0.0,
                1.015,
                label,
                transform=ax.transAxes,
                fontsize=PLOT_STYLE["panel_label_size"],
                fontweight="bold",
                ha="left",
                va="bottom",
            )
    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        top=0.88,
        bottom=0.09,
        hspace=0.27,
    )
    paths = save_figure(
        fig, output_dir, "Fig6_Private_Robustness_Combined", dpi
    )
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    configure_matplotlib()
    gaussian_rows = read_csv(results_dir / "FIG1_GAUSSIAN_NOISE_AP.csv")
    mask_rows = read_csv(results_dir / "FIG2_TEMPORAL_MASK_AP.csv")
    validate_levels(
        gaussian_rows, "sigma_test", EXPECTED_GAUSSIAN_LEVELS
    )
    validate_levels(mask_rows, "rho_mask", EXPECTED_MASK_LEVELS)
    gaussian_paths = plot_gaussian(
        gaussian_rows, output_dir, args.dpi, args.band
    )
    mask_paths = plot_mask(mask_rows, output_dir, args.dpi, args.band)
    combined_paths = plot_combined(
        gaussian_rows, mask_rows, output_dir, args.dpi, args.band
    )
    for path in (*gaussian_paths, *mask_paths, *combined_paths):
        print(path)


if __name__ == "__main__":
    main()
