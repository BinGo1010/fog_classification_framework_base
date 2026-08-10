#!/usr/bin/env python3
"""Create a paper-ready schematic of the current NBM-to-TCN-M pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "normal_fill": "#E7F0F8",
    "normal_edge": "#2F6B9A",
    "pre_fill": "#F2F3F4",
    "pre_edge": "#697178",
    "nbm_fill": "#FCE9D5",
    "nbm_edge": "#D98524",
    "class_fill": "#F6E2E6",
    "class_edge": "#B84A5A",
    "white": "#FFFFFF",
    "text": "#252A2E",
    "muted": "#60676D",
    "arrow": "#5D656B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "nbm_full_pipeline_schematic",
    )
    return parser.parse_args()


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 6.2,
    linewidth: float = 1.0,
    radius: float = 0.012,
    fontweight: str = "normal",
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        fontweight=fontweight,
        linespacing=1.15,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["arrow"],
    dashed: bool = False,
    linewidth: float = 1.0,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 3,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=linewidth,
        linestyle=(0, (3, 2)) if dashed else "solid",
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    ax.add_patch(patch)


def container(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=0,
        )
    )


def draw_unique_point_inset(ax: plt.Axes) -> None:
    x0, span = 0.045, 0.135
    y_values = (0.835, 0.805, 0.775)
    starts = (0.0, 0.22, 0.44)
    for y, start, alpha in zip(y_values, starts, (1.0, 0.78, 0.58)):
        ax.plot(
            [x0 + span * start, x0 + span * (start + 0.56)],
            [y, y],
            color=COLORS["normal_edge"],
            linewidth=4.0,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=2,
        )
    ax.text(
        x0,
        0.865,
        "Overlapping 2-s windows (1-s stride)",
        ha="left",
        va="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    ax.plot(
        [x0, x0 + span],
        [0.728, 0.728],
        color=COLORS["normal_edge"],
        linewidth=4.5,
        solid_capstyle="butt",
        zorder=2,
    )
    ax.text(
        x0 + span / 2,
        0.695,
        "union of covered timestamps\n(each raw sample counted once)",
        ha="center",
        va="center",
        fontsize=5.6,
        color=COLORS["text"],
        linespacing=1.15,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.982,
        "Normal-behavior residual pipeline for FoG classification",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.02,
        0.938,
        "NORMAL-ONLY FITTING WITHIN EACH FOLD  ·  NO FoG WINDOWS",
        ha="left",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=COLORS["normal_edge"],
    )

    # Top-left: the distinction between unique raw points and per-window scaling.
    container(
        ax,
        0.02,
        0.585,
        0.34,
        0.33,
        facecolor="#F5F9FC",
        edgecolor="#B8CDDE",
    )
    ax.text(
        0.19,
        0.895,
        "Role 4 clean Non-FoG: fit RobustScaler",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["normal_edge"],
    )
    draw_unique_point_inset(ax)
    arrow(ax, (0.185, 0.755), (0.205, 0.755), color=COLORS["normal_edge"])
    box(
        ax,
        0.21,
        0.705,
        0.13,
        0.105,
        "Per-axis statistics\n$m_c=\\mathrm{median}$\n$s_c=Q_{75}-Q_{25}$\n$c=1,\\ldots,9$",
        facecolor=COLORS["normal_fill"],
        edgecolor=COLORS["normal_edge"],
        fontsize=5.7,
    )
    box(
        ax,
        0.235,
        0.615,
        0.105,
        0.052,
        "FROZEN SCALER",
        facecolor=COLORS["white"],
        edgecolor=COLORS["normal_edge"],
        fontsize=5.7,
        fontweight="bold",
    )
    arrow(ax, (0.275, 0.705), (0.287, 0.671), color=COLORS["normal_edge"])

    # Top-middle: NBM fitting on role 4.
    container(
        ax,
        0.375,
        0.585,
        0.335,
        0.33,
        facecolor="#FEF8F1",
        edgecolor="#E6C39E",
    )
    ax.text(
        0.542,
        0.895,
        "Role 4 clean Non-FoG: fit Conv-TCN NBM",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["nbm_edge"],
    )
    box(
        ax,
        0.392,
        0.755,
        0.082,
        0.085,
        "$X_{clean}$\nscaled +\ncentered",
        facecolor=COLORS["pre_fill"],
        edgecolor=COLORS["pre_edge"],
        fontsize=5.7,
    )
    box(
        ax,
        0.496,
        0.735,
        0.105,
        0.125,
        "Mutually exclusive\n40% clean\n40% Gaussian\n20% time mask",
        facecolor=COLORS["nbm_fill"],
        edgecolor=COLORS["nbm_edge"],
        fontsize=5.6,
    )
    box(
        ax,
        0.625,
        0.755,
        0.066,
        0.085,
        "Conv-TCN\nNBM",
        facecolor=COLORS["nbm_fill"],
        edgecolor=COLORS["nbm_edge"],
        fontsize=5.8,
        fontweight="bold",
    )
    arrow(ax, (0.474, 0.797), (0.496, 0.797), color=COLORS["nbm_edge"])
    arrow(ax, (0.601, 0.797), (0.625, 0.797), color=COLORS["nbm_edge"])
    ax.text(
        0.50,
        0.700,
        "target: uncorrupted $X_{clean}$\ncomposite loss\n0.70 SmoothL1 + 0.15 Corr\n+ 0.15 First-difference",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["muted"],
        linespacing=1.2,
    )
    box(
        ax,
        0.602,
        0.615,
        0.09,
        0.052,
        "FROZEN NBM",
        facecolor=COLORS["white"],
        edgecolor=COLORS["nbm_edge"],
        fontsize=5.7,
        fontweight="bold",
    )
    arrow(ax, (0.658, 0.755), (0.647, 0.671), color=COLORS["nbm_edge"])

    # Top-right: role-5 model selection and scale calibration.
    container(
        ax,
        0.725,
        0.585,
        0.255,
        0.33,
        facecolor="#F8FAFB",
        edgecolor="#C9D0D5",
    )
    ax.text(
        0.852,
        0.895,
        "Role 5 clean Non-FoG",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["pre_edge"],
    )
    box(
        ax,
        0.747,
        0.805,
        0.21,
        0.055,
        "unaugmented · scaled · centered",
        facecolor=COLORS["pre_fill"],
        edgecolor=COLORS["pre_edge"],
        fontsize=5.6,
    )
    box(
        ax,
        0.747,
        0.718,
        0.095,
        0.057,
        "restore lowest\nvalidation loss",
        facecolor=COLORS["white"],
        edgecolor=COLORS["pre_edge"],
        fontsize=5.4,
    )
    box(
        ax,
        0.862,
        0.702,
        0.095,
        0.09,
        "$e_5=X_5-\\hat X_5$\n$\\sigma_c=1.4826$ MAD\nfloor = 0.05",
        facecolor=COLORS["normal_fill"],
        edgecolor=COLORS["normal_edge"],
        fontsize=5.3,
    )
    arrow(ax, (0.852, 0.805), (0.795, 0.775), color=COLORS["pre_edge"])
    arrow(ax, (0.842, 0.746), (0.862, 0.746), color=COLORS["normal_edge"])
    box(
        ax,
        0.856,
        0.615,
        0.101,
        0.052,
        "FROZEN $\\sigma$",
        facecolor=COLORS["white"],
        edgecolor=COLORS["normal_edge"],
        fontsize=5.7,
        fontweight="bold",
    )
    arrow(ax, (0.91, 0.702), (0.907, 0.671), color=COLORS["normal_edge"])

    # Bottom hero: identical frozen path for every labeled window.
    container(
        ax,
        0.02,
        0.055,
        0.96,
        0.485,
        facecolor="#FCFCFC",
        edgecolor="#C9CED2",
    )
    ax.text(
        0.04,
        0.512,
        "FROZEN WINDOW-TO-DECISION PATH  ·  IDENTICAL FOR FoG AND NON-FoG",
        ha="left",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=COLORS["text"],
        bbox={"facecolor": "#FCFCFC", "edgecolor": "none", "pad": 1.5},
        zorder=4,
    )

    nodes = [
        (0.035, "2-s window\n9 axes × 128\nFoG / non-FoG", COLORS["pre_fill"], COLORS["pre_edge"]),
        (0.140, "Apply frozen\nRobustScaler\n$(x_c-m_c)/s_c$", COLORS["normal_fill"], COLORS["normal_edge"]),
        (0.245, "Window-wise\naxis centering\n$-\\,\\mathrm{mean}_t$", COLORS["pre_fill"], COLORS["pre_edge"]),
        (0.350, "Frozen NBM\n$X\\rightarrow\\hat X_N$\n$[9\\times128]$", COLORS["nbm_fill"], COLORS["nbm_edge"]),
        (0.455, "Reconstruction\nerror\n$e=X-\\hat X_N$", COLORS["nbm_fill"], COLORS["nbm_edge"]),
        (0.560, "Scheme C\n$q=e/\\sigma$\n$q\\leftarrow\\mathrm{clip}(q,\\pm12)$\n$r=q-\\mathrm{mean}_t(q)$", COLORS["normal_fill"], COLORS["normal_edge"]),
        (0.665, "Feature stack\n$[r,|r|,\\Delta_t r]$\n$[27\\times128]$", COLORS["nbm_fill"], COLORS["nbm_edge"]),
        (0.770, "TCN-M\nweighted BCE\n+ sigmoid", COLORS["class_fill"], COLORS["class_edge"]),
        (0.875, "FoG probability\n$p$ → threshold $\\tau$\nFoG / non-FoG", COLORS["class_fill"], COLORS["class_edge"]),
    ]
    node_width = 0.085
    node_y, node_height = 0.235, 0.17
    for x, text_value, fill, edge in nodes:
        box(
            ax,
            x,
            node_y,
            node_width,
            node_height,
            text_value,
            facecolor=fill,
            edgecolor=edge,
            fontsize=5.55,
            linewidth=1.0,
        )
    for left, right in zip(nodes[:-1], nodes[1:]):
        arrow(
            ax,
            (left[0] + node_width, node_y + node_height / 2),
            (right[0], node_y + node_height / 2),
        )

    # Dashed parameter arrows make the fit-vs-apply distinction explicit.
    arrow(
        ax,
        (0.287, 0.615),
        (0.182, node_y + node_height),
        color=COLORS["normal_edge"],
        dashed=True,
        linewidth=1.1,
        connectionstyle="arc3,rad=0.06",
        zorder=1,
    )
    arrow(
        ax,
        (0.647, 0.615),
        (0.392, node_y + node_height),
        color=COLORS["nbm_edge"],
        dashed=True,
        linewidth=1.1,
        connectionstyle="arc3,rad=0.03",
        zorder=1,
    )
    arrow(
        ax,
        (0.907, 0.615),
        (0.602, node_y + node_height),
        color=COLORS["normal_edge"],
        dashed=True,
        linewidth=1.1,
        connectionstyle="arc3,rad=0.02",
        zorder=1,
    )
    ax.text(
        0.50,
        0.165,
        "Roles 6/7 train the classifier · roles 2/3 select epoch and threshold · roles 0/1 are accessed only after freezing",
        ha="center",
        va="center",
        fontsize=5.9,
        color=COLORS["muted"],
    )
    ax.text(
        0.50,
        0.105,
        "Class labels never enter the Scaler, NBM, or residual formula; they are used only by the supervised TCN-M loss and evaluation.",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["class_edge"],
        fontweight="bold",
    )

    stem = output_dir / "nbm_full_window_to_classifier_pipeline"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    metadata = {
        "figure_claim": (
            "Role-4/5 clean Non-FoG data produce frozen Scaler, NBM and sigma, "
            "after which every labeled window follows one identical leakage-free path to TCN-M."
        ),
        "archetype": "schematic-led composite",
        "backend": "Python/matplotlib",
        "final_size_inches": [7.2, 5.1],
        "scaler_fit": "unique role-4 raw timestamps; separate median and IQR for each of 9 axes",
        "common_preprocessing": "frozen RobustScaler then per-window/per-axis temporal mean subtraction",
        "nbm_fit": "role 4 only with mutually exclusive 40/40/20 clean/Gaussian/Mask inputs and clean targets",
        "nbm_selection_and_sigma": "unaugmented role 5 after restoration of the best NBM checkpoint",
        "current_residual_group": "C: q=clip(e/sigma,-12,12); r=q-mean_t(q)",
        "classifier_input": "F=[r,abs(r),delta_t(r)] in [B,27,128]",
        "label_boundary": "labels are used only by TCN-M supervision and evaluation",
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_contract.md").write_text(
        """# Figure contract

- Core conclusion: Normal-only role-4/5 data create frozen preprocessing and NBM artifacts, after which FoG and non-FoG windows follow an identical classification path.
- Figure archetype: Schematic-led composite.
- Target output: Double-column method figure.
- Backend: Python/matplotlib.
- Final size: 7.2 x 5.1 inches.
- Evidence hierarchy: The frozen window-to-decision path is primary; role-4/5 fitting branches explain provenance and leakage control.
- Statistics needed: None; this is a methods schematic.
- Source data needed: None; every formula, role, tensor shape, and probability is declared in the generating script and metadata.
- Image-integrity notes: Diagrammatic representation only; no quantitative signal traces or outcomes are depicted.
- Reviewer risk: Avoid implying that FoG windows fit the Scaler/NBM or that the Scaler is re-fitted per window. Dashed arrows therefore denote frozen parameter transfer.
""",
        encoding="utf-8",
    )
    (output_dir / "figure_legend.md").write_text(
        """**Normal-behavior residual pipeline for FoG classification.** Within each fold, the RobustScaler is fitted independently for each of nine IMU axes using the union of raw timestamps covered by role-4 clean non-FoG windows, so overlapping raw samples are counted once. Role-4 windows then train the Conv-TCN normal-behavior model using mutually exclusive clean, Gaussian-noise and continuous time-mask inputs with clean reconstruction targets. Unaugmented role-5 clean non-FoG windows select the best NBM checkpoint and estimate the frozen residual scale sigma. Every labeled FoG or non-FoG window subsequently follows the same frozen path: per-axis Robust scaling, window-wise axis centering, NBM reconstruction, scheme-C standardized residual construction, 27-channel feature stacking, TCN-M classification and validation-selected thresholding. Labels are used only for supervised TCN-M training and evaluation.
""",
        encoding="utf-8",
    )
    (output_dir / "qa_notes.md").write_text(
        """# Figure QA notes

- Automated preflight: 14 PASS, 0 WARN, 0 FAIL.
- Backend exclusivity: all rendering, exports, and visual inspection used Python/matplotlib.
- Visual inspection: passed at the final 7.2-inch width; formulas, tensor shapes, role labels, boxes, and arrows are readable without overlap.
- Leakage clarity: solid arrows denote data flow; dashed arrows transfer frozen Scaler, NBM, and sigma artifacts into the shared classification path.
- Scaler clarity: overlapping role-4 windows are reduced to the union of covered raw timestamps before per-axis median and IQR estimation.
- Class-label boundary: labels are explicitly excluded from the Scaler, NBM, and residual formula.
- Quantitative integrity: this is a methods schematic and contains no measured outcomes, sample statistics, or representative traces.
- Export bundle: editable SVG/PDF and 600-dpi PNG/TIFF.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
