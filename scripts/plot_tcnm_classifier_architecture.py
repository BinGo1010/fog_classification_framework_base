"""Render the downstream 27-channel residual TCN-M classifier schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from plot_conv_tcn_nbm_architecture import COLORS, arrow, container, rounded_box


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "figures" / "tcnm_classifier_architecture"
OUTPUT_STEM = OUTPUT_DIR / "tcnm_classifier_architecture"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


CLASSIFIER_COLORS = {
    "feature_bg": "#F1EEF8",
    "feature_fill": "#E7E1F3",
    "feature_edge": "#70619E",
    "head_bg": "#EAF3FA",
    "head_fill": "#D8EAF6",
    "head_edge": "#4C82AD",
    "prob_fill": "#D9EDF0",
    "prob_edge": "#3E7F86",
    "decision_fill": "#DDEED8",
    "decision_edge": "#57844F",
    "skip_fill": "#EDF4F8",
    "skip_edge": "#2E6F9E",
}


def draw_main_classifier(ax: plt.Axes) -> None:
    y, h = 5.58, 2.42
    cy = y + h / 2

    rounded_box(
        ax,
        0.18,
        y,
        1.82,
        h,
        "Residual tensor\n$\\mathbf{F}=[r,|r|,\\Delta_t r]$\n[B, 27, 128]",
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["neutral_edge"],
        fontsize=6.8,
        weight="bold",
        linewidth=1.15,
    )

    container(
        ax,
        2.28,
        5.00,
        6.92,
        3.78,
        "Causal temporal feature extractor",
        facecolor=CLASSIFIER_COLORS["feature_bg"],
        edgecolor=CLASSIFIER_COLORS["feature_edge"],
    )

    blocks = [
        (2.55, "TCN block 1\n27$\\rightarrow$32\nk3 · $d$1\n[B, 32, 128]"),
        (4.18, "TCN block 2\n32$\\rightarrow$64\nk3 · $d$2\n[B, 64, 128]"),
        (5.81, "TCN block 3\n64$\\rightarrow$64\nk3 · $d$4\n[B, 64, 128]"),
        (7.44, "TCN block 4\n64$\\rightarrow$128\nk3 · $d$8\n[B, 128, 128]"),
    ]
    block_w = 1.47
    for x, text in blocks:
        rounded_box(
            ax,
            x,
            y,
            block_w,
            h,
            text,
            facecolor=CLASSIFIER_COLORS["feature_fill"],
            edgecolor=CLASSIFIER_COLORS["feature_edge"],
            fontsize=5.8,
        )
    for left, right in zip(blocks[:-1], blocks[1:]):
        arrow(
            ax,
            (left[0] + block_w, cy),
            (right[0], cy),
            color=CLASSIFIER_COLORS["feature_edge"],
            mutation_scale=7.0,
        )

    container(
        ax,
        9.52,
        5.00,
        4.43,
        3.78,
        "Classification head",
        facecolor=CLASSIFIER_COLORS["head_bg"],
        edgecolor=CLASSIFIER_COLORS["head_edge"],
    )
    head_blocks = [
        (9.78, 1.18, "Temporal GAP\nmean over $t$\n[B, 128]"),
        (11.17, 1.05, "Dropout\n$p$ = 0.30"),
        (12.43, 1.22, "Linear\n128$\\rightarrow$1\nlogit $z$"),
    ]
    for x, width, text in head_blocks:
        rounded_box(
            ax,
            x,
            y,
            width,
            h,
            text,
            facecolor=CLASSIFIER_COLORS["head_fill"],
            edgecolor=CLASSIFIER_COLORS["head_edge"],
            fontsize=5.9,
        )
    for left, right in zip(head_blocks[:-1], head_blocks[1:]):
        arrow(
            ax,
            (left[0] + left[1], cy),
            (right[0], cy),
            color=CLASSIFIER_COLORS["head_edge"],
            mutation_scale=7.0,
        )

    rounded_box(
        ax,
        14.28,
        y,
        1.38,
        h,
        "Sigmoid\n$p_{\\mathrm{FoG}}$\n$=\\sigma(z)$",
        facecolor=CLASSIFIER_COLORS["prob_fill"],
        edgecolor=CLASSIFIER_COLORS["prob_edge"],
        fontsize=6.6,
        weight="bold",
        linewidth=1.15,
    )
    rounded_box(
        ax,
        16.02,
        y,
        1.78,
        h,
        "Validation\nthreshold $\\tau^*$\n\n$p_{\\mathrm{FoG}}\\geq\\tau^* \\rightarrow$ FoG\n$p_{\\mathrm{FoG}}<\\tau^* \\rightarrow$ Non-FoG",
        facecolor=CLASSIFIER_COLORS["decision_fill"],
        edgecolor=CLASSIFIER_COLORS["decision_edge"],
        fontsize=5.4,
        weight="bold",
        linewidth=1.15,
    )

    arrow(ax, (2.00, cy), (2.55, cy))
    arrow(ax, (8.91, cy), (9.78, cy))
    arrow(ax, (13.65, cy), (14.28, cy))
    arrow(ax, (15.66, cy), (16.02, cy))

    ax.text(
        16.91,
        5.28,
        "selected on roles 2/3",
        ha="center",
        va="center",
        fontsize=5.4,
        color=CLASSIFIER_COLORS["decision_edge"],
        style="italic",
    )


def draw_tcn_block_inset(ax: plt.Axes) -> None:
    container(
        ax,
        1.25,
        0.48,
        15.50,
        3.67,
        "Causal TCN residual block  ($d\\in\\{1,2,4,8\\}$)",
        facecolor="#F8F9FA",
        edgecolor=CLASSIFIER_COLORS["feature_edge"],
    )

    y, h, cy = 1.33, 1.12, 1.89
    rounded_box(
        ax,
        1.62,
        y,
        0.78,
        h,
        "$\\mathbf{h}$",
        facecolor="white",
        edgecolor=COLORS["neutral_edge"],
        fontsize=8.0,
        weight="bold",
    )
    rounded_box(
        ax,
        2.72,
        y,
        1.46,
        h,
        "Causal Conv1D\n$C_{in}\\rightarrow C_{out}$\nk3 · dilation $d$\nleft pad $2d$",
        facecolor=CLASSIFIER_COLORS["feature_fill"],
        edgecolor=CLASSIFIER_COLORS["feature_edge"],
        fontsize=5.6,
    )
    rounded_box(
        ax,
        4.50,
        y,
        1.48,
        h,
        "BatchNorm\nReLU · Dropout\n$p$ = 0.20",
        facecolor=CLASSIFIER_COLORS["feature_fill"],
        edgecolor=CLASSIFIER_COLORS["feature_edge"],
        fontsize=5.8,
    )
    rounded_box(
        ax,
        6.30,
        y,
        1.48,
        h,
        "Causal Conv1D\n$C_{out}\\rightarrow C_{out}$\nk3 · dilation $d$\nleft pad $2d$",
        facecolor=CLASSIFIER_COLORS["feature_fill"],
        edgecolor=CLASSIFIER_COLORS["feature_edge"],
        fontsize=5.6,
    )
    rounded_box(
        ax,
        8.10,
        y,
        1.30,
        h,
        "BatchNorm\nReLU",
        facecolor=CLASSIFIER_COLORS["feature_fill"],
        edgecolor=CLASSIFIER_COLORS["feature_edge"],
        fontsize=5.9,
    )

    add_center = (10.02, cy)
    ax.add_patch(
        Circle(
            add_center,
            radius=0.31,
            facecolor="white",
            edgecolor=CLASSIFIER_COLORS["skip_edge"],
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.text(
        *add_center,
        "+",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=CLASSIFIER_COLORS["skip_edge"],
        zorder=5,
    )
    rounded_box(
        ax,
        10.65,
        y,
        0.90,
        h,
        "$\\mathbf{h}'$",
        facecolor="white",
        edgecolor=COLORS["neutral_edge"],
        fontsize=8.0,
        weight="bold",
    )

    for start_x, end_x in [
        (2.40, 2.72),
        (4.18, 4.50),
        (5.98, 6.30),
        (7.78, 8.10),
        (9.40, 9.71),
        (10.33, 10.65),
    ]:
        arrow(
            ax,
            (start_x, cy),
            (end_x, cy),
            color=CLASSIFIER_COLORS["feature_edge"],
            mutation_scale=7.0,
        )

    shortcut_y = 3.02
    rounded_box(
        ax,
        5.26,
        2.62,
        2.18,
        0.82,
        "Shortcut\nIdentity: $C_{in}=C_{out}$\nConv1D 1$\\times$1: otherwise",
        facecolor=CLASSIFIER_COLORS["skip_fill"],
        edgecolor=CLASSIFIER_COLORS["skip_edge"],
        fontsize=5.4,
    )
    ax.plot(
        [2.01, 2.01, 5.26],
        [y + h, shortcut_y, shortcut_y],
        color=CLASSIFIER_COLORS["skip_edge"],
        linewidth=1.05,
        zorder=2,
    )
    arrow(
        ax,
        (7.44, shortcut_y),
        (10.02, shortcut_y),
        color=CLASSIFIER_COLORS["skip_edge"],
        linewidth=1.05,
        mutation_scale=7.0,
    )
    arrow(
        ax,
        (10.02, shortcut_y),
        (10.02, add_center[1] + 0.31),
        color=CLASSIFIER_COLORS["skip_edge"],
        linewidth=1.05,
        mutation_scale=7.0,
    )

    ax.text(
        12.15,
        2.04,
        "Causal and length preserving\nNo activation after residual addition\nTemporal length remains 128",
        ha="left",
        va="center",
        fontsize=6.0,
        color=COLORS["muted"],
        linespacing=1.35,
    )


def render() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.25, 4.15))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        9.0,
        9.56,
        "Residual TCN classifier (TCN-M)",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        9.0,
        9.18,
        "27-channel residual representation  →  causal temporal encoding  →  FoG probability and decision",
        ha="center",
        va="center",
        fontsize=6.1,
        color=COLORS["muted"],
    )

    draw_main_classifier(ax)
    draw_tcn_block_inset(ax)

    ax.text(
        9.0,
        0.14,
        "BN, BatchNorm1d; GAP, global average pooling. Sigmoid is applied to the output logit; the validation-selected threshold is frozen before testing.",
        ha="center",
        va="center",
        fontsize=5.35,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    render()
