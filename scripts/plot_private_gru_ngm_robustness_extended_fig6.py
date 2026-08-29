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


# =============================================================================
# 用户可手动修改的正式绘图参数
# 以下设置依次控制：输入输出目录、数据强度网格、坐标文字、字体与图形样式。
# 修改后重新运行本程序即可生成新的 PNG、PDF 和 SVG。
# =============================================================================

# 扩展鲁棒性实验汇总 CSV 所在目录。
plot.DEFAULT_RESULTS = (
    REPO_ROOT
    / "outputs"
    / "private_gru_ngm_robustness_matched_tcn"
    / "robustness_extended"
)
# 正式图片的默认输出目录。
plot.DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "private_gru_ngm_robustness_extended_figures"
)
# 误差带统计量："std" 表示 Mean ± SD；也可手动改为 "sem"。
plot.DEFAULT_BAND = "std"

# Gaussian-noise CSV 必须包含的测试强度；用于检查输入结果是否完整。
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
# Temporal-masking CSV 必须包含的遮挡比例；0.025 表示 2.5%。
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
# Gaussian 图实际显示的横轴主刻度位置。
plot.GAUSSIAN_X_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Mask 图实际显示的横轴主刻度位置；内部数值范围为 0–1。
plot.MASK_X_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Mask 图横轴主刻度文字；这里转换为百分比 0–100。
plot.MASK_X_TICK_LABELS = ("0", "20", "40", "60", "80", "100")
# 是否在上下合并图中显示分面编号 a 和 b。
plot.SHOW_PANEL_LABELS = True

# 图中文字设置；空字符串表示不显示相应标题。
plot.PLOT_TEXT.update(
    {
        # Gaussian 单图标题。
        "gaussian_title": "",
        # Temporal masking 单图标题。
        "mask_title": "",
        # Gaussian 图横轴标题。
        "gaussian_xlabel": (
            r"Gaussian Noise Standard Deviation, $\sigma_{\mathrm{test}}$"
        ),
        # Temporal masking 图横轴标题。
        "mask_xlabel": r"Temporal Masking Ratio, $\rho_{\mathrm{mask}}$ (%)",
        # 两张图的纵轴标题。
        "ylabel": "Average Precision (AP)",
        # none 模型曲线的图例文字。
        "none_legend": "Without Augmentation",
        # Gaussian + Mask 模型曲线的图例文字。
        "gaussian_mask_legend": "Gaussian + Masking Augmentation",
        # 图例标题；空字符串表示不显示。
        "legend_title": "",
    }
)

# 图形样式设置；尺寸单位为英寸，字体单位为 pt。
plot.PLOT_STYLE.update(
    {
        # 单张图尺寸：(宽, 高)。
        "figure_size": (4.2, 3.1),
        # 两张图上下组合后的尺寸：(宽, 高)。
        "combined_figure_size": (4.2, 6.0),
        # Matplotlib 全局默认字体大小。
        "font_size": 8.0,
        # 横轴和纵轴标题字体大小。
        "axis_label_size": 8.5,
        # 分面标题字体大小；标题为空时不显示。
        "title_size": 9.0,
        # 合并图分面编号 a、b 的字体大小。
        "panel_label_size": 9.0,
        # 横纵坐标刻度数字的字体大小。
        "tick_label_size": 8.0,
        # 图例曲线名称的字体大小。
        "legend_font_size": 7.5,
        # 图例标题字体大小；图例标题为空时不显示。
        "legend_title_size": 7.5,
        # Gaussian 单图图例位置的兼容设置。
        "gaussian_legend_location": "upper left",
        # Mask 单图图例位置的兼容设置。
        "mask_legend_location": "upper left",
        # 是否显示图例边框。
        "legend_frame": False,
        # 两条均值曲线的线宽。
        "line_width": 1.7,
        # 曲线上数据点标记的大小。
        "marker_size": 4.4,
        # 数据点标记边缘的线宽。
        "marker_edge_width": 1.0,
        # Gaussian 图 Mean ± SD 误差带透明度；越小越淡。
        "gaussian_band_alpha": 0.10,
        # Mask 图 Mean ± SD 误差带透明度；越小越淡。
        "mask_band_alpha": 0.10,
        # 横纵网格线透明度。
        "grid_alpha": 0.25,
        # 上下合并图共用的纵轴范围。
        "y_limits": (0.18, 0.72),
        # 上下合并图共用的纵轴刻度位置。
        "y_ticks": np.arange(0.20, 0.72, 0.10),
        # Gaussian 单图使用的纵轴范围。
        "gaussian_y_limits": (0.50, 0.8),
        # Gaussian 单图使用的纵轴刻度位置。
        "gaussian_y_ticks": np.arange(0.50, 0.8, 0.02),
        # Mask 单图使用的纵轴范围。
        "mask_y_limits": (0.18, 0.63),
        # Mask 单图使用的纵轴刻度位置。
        "mask_y_ticks": np.arange(0.20, 0.61, 0.10),
        # Gaussian 图横轴显示范围。
        "gaussian_x_limits": (-0.025, 1.025),
        # Mask 图横轴显示范围。
        "mask_x_limits": (-0.025, 1.025),
        # Gaussian 图 none 曲线纵向偏移量；正值上移，负值下移。
        "gaussian_none_curve_offset": 0.05,
        # Gaussian 图 Gaussian + Mask 曲线纵向偏移量。
        "gaussian_gaussian_mask_curve_offset": 0.0,
        # Mask 图 none 曲线纵向偏移量；正值上移，负值下移。
        "mask_none_curve_offset": 0.05,
        # Mask 图 Gaussian + Mask 曲线纵向偏移量。
        "mask_gaussian_mask_curve_offset": 0.0,
    }
)


if __name__ == "__main__":
    plot.main()
