"""Render a publication-ready schematic of the Conv-TCN NBM architecture."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "figures" / "conv_tcn_nbm_architecture"
OUTPUT_STEM = OUTPUT_DIR / "conv_tcn_nbm_architecture"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COLORS = {
    "ink": "#25313B",
    "muted": "#66737D",
    "neutral_fill": "#F2F4F6",
    "neutral_edge": "#7C8993",
    "encoder_bg": "#FFF4E5",
    "encoder_fill": "#FBE3BF",
    "encoder_edge": "#C98227",
    "tcn_fill": "#ECE8F6",
    "tcn_edge": "#7566A3",
    "latent_fill": "#F8D99B",
    "latent_edge": "#A66A00",
    "decoder_bg": "#EAF3FA",
    "decoder_fill": "#D8EAF6",
    "decoder_edge": "#4C82AD",
    "output_fill": "#DDEED8",
    "output_edge": "#57844F",
    "residual_bg": "#F8F9FA",
    "accent": "#2E6F9E",
}


def rounded_box(
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
    radius: float = 0.12,
    weight: str = "normal",
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
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
        color=COLORS["ink"],
        fontweight=weight,
        linespacing=1.16,
        zorder=zorder + 1,
    )
    return patch


def container(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=1.15,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.18,
        y + height - 0.32,
        title,
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=edgecolor,
        zorder=2,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    linewidth: float = 1.15,
    mutation_scale: float = 9.0,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color or COLORS["ink"],
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=zorder,
        )
    )


def draw_main_architecture(ax: plt.Axes) -> None:
    main_y = 5.00
    main_h = 3.78

    rounded_box(
        ax,
        0.20,
        5.66,
        1.55,
        2.30,
        "Processed IMU\nwindow\n$\\mathbf{X}^{\\mathrm{in}}$\n[B, 9, 128]",
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["neutral_edge"],
        fontsize=7.0,
        weight="bold",
    )

    container(
        ax,
        2.05,
        main_y,
        5.45,
        main_h,
        "Encoder",
        facecolor=COLORS["encoder_bg"],
        edgecolor=COLORS["encoder_edge"],
    )

    encoder_boxes = [
        (
            2.30,
            "Conv1D\n9$\\rightarrow$32\nk7 · s2 · p3\nGN + GELU\n[B, 32, 64]",
            COLORS["encoder_fill"],
            COLORS["encoder_edge"],
        ),
        (
            3.59,
            "TCN stack\n2 ResBlocks\n$d$: 1$\\rightarrow$2\n[B, 32, 64]",
            COLORS["tcn_fill"],
            COLORS["tcn_edge"],
        ),
        (
            4.88,
            "Downsample\n32$\\rightarrow$24\nk5 · s2 · p2\nGN + GELU\n[B, 24, 32]",
            COLORS["encoder_fill"],
            COLORS["encoder_edge"],
        ),
        (
            6.17,
            "TCN stack\n2 ResBlocks\n$d$: 1$\\rightarrow$2\n[B, 24, 32]",
            COLORS["tcn_fill"],
            COLORS["tcn_edge"],
        ),
    ]
    box_y, box_w, box_h = 5.60, 1.08, 2.40
    for x, text, fill, edge in encoder_boxes:
        rounded_box(
            ax,
            x,
            box_y,
            box_w,
            box_h,
            text,
            facecolor=fill,
            edgecolor=edge,
            fontsize=5.75,
        )
    for left, right in zip(encoder_boxes[:-1], encoder_boxes[1:]):
        arrow(
            ax,
            (left[0] + box_w, box_y + box_h / 2),
            (right[0], box_y + box_h / 2),
            color=COLORS["encoder_edge"],
            mutation_scale=7.5,
        )

    rounded_box(
        ax,
        7.82,
        5.63,
        1.65,
        2.36,
        "Bottleneck\nConv1D\n24$\\rightarrow$16\nk1 · GN · GELU\n$\\mathbf{H}$ [B, 16, 32]",
        facecolor=COLORS["latent_fill"],
        edgecolor=COLORS["latent_edge"],
        fontsize=6.3,
        weight="bold",
        linewidth=1.2,
    )

    container(
        ax,
        9.80,
        main_y,
        6.00,
        main_h,
        "Decoder",
        facecolor=COLORS["decoder_bg"],
        edgecolor=COLORS["decoder_edge"],
    )
    decoder_boxes = [
        (
            10.04,
            "Conv1D\n16$\\rightarrow$24\nk3 · p1\nGN + GELU\n[B, 24, 32]",
            COLORS["decoder_fill"],
            COLORS["decoder_edge"],
        ),
        (
            11.17,
            "Linear\nupsample $\\times$2\n[B, 24, 64]",
            COLORS["decoder_fill"],
            COLORS["decoder_edge"],
        ),
        (
            12.30,
            "Conv1D\n24$\\rightarrow$32\nk5 · p2\nGN + GELU\n[B, 32, 64]",
            COLORS["decoder_fill"],
            COLORS["decoder_edge"],
        ),
        (
            13.43,
            "TCN stack\n2 ResBlocks\n$d$: 1$\\rightarrow$2\n[B, 32, 64]",
            COLORS["tcn_fill"],
            COLORS["tcn_edge"],
        ),
        (
            14.56,
            "Upsample $\\times$2\nConv 32$\\rightarrow$16\nk7 · p3\nGN + GELU\n[B, 16, 128]",
            COLORS["decoder_fill"],
            COLORS["decoder_edge"],
        ),
    ]
    dec_w = 0.98
    for x, text, fill, edge in decoder_boxes:
        rounded_box(
            ax,
            x,
            box_y,
            dec_w,
            box_h,
            text,
            facecolor=fill,
            edgecolor=edge,
            fontsize=5.45,
        )
    for left, right in zip(decoder_boxes[:-1], decoder_boxes[1:]):
        arrow(
            ax,
            (left[0] + dec_w, box_y + box_h / 2),
            (right[0], box_y + box_h / 2),
            color=COLORS["decoder_edge"],
            mutation_scale=7.0,
        )

    rounded_box(
        ax,
        16.14,
        5.63,
        1.66,
        2.36,
        "Output head\nConv1D 16$\\rightarrow$9\nk=1 · linear\n$\\widehat{\\mathbf{X}}^{\\mathrm{NBM}}$\n[B, 9, 128]",
        facecolor=COLORS["output_fill"],
        edgecolor=COLORS["output_edge"],
        fontsize=6.25,
        weight="bold",
        linewidth=1.2,
    )

    center_y = box_y + box_h / 2
    arrow(ax, (1.75, center_y), (2.30, center_y))
    arrow(ax, (7.25, center_y), (7.82, center_y))
    arrow(ax, (9.47, center_y), (10.04, center_y))
    arrow(ax, (15.54, center_y), (16.14, center_y))

def draw_residual_inset(ax: plt.Axes) -> None:
    container(
        ax,
        2.05,
        0.50,
        13.75,
        3.65,
        "TCN residual block  (reused at dilation $d$ = 1 and 2)",
        facecolor=COLORS["residual_bg"],
        edgecolor=COLORS["tcn_edge"],
    )

    y, h = 1.42, 1.05
    rounded_box(
        ax,
        2.38,
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
        3.48,
        y,
        1.25,
        h,
        "Conv1D\n$C\\rightarrow C$\nk=3\ndilation=$d$\npadding=$d$",
        facecolor=COLORS["tcn_fill"],
        edgecolor=COLORS["tcn_edge"],
        fontsize=5.9,
    )
    rounded_box(
        ax,
        5.05,
        y,
        1.50,
        h,
        "GroupNorm\nGELU · Dropout",
        facecolor=COLORS["tcn_fill"],
        edgecolor=COLORS["tcn_edge"],
        fontsize=6.0,
    )
    rounded_box(
        ax,
        6.87,
        y,
        1.25,
        h,
        "Conv1D\n$C\\rightarrow C$\nk=3\ndilation=$d$\npadding=$d$",
        facecolor=COLORS["tcn_fill"],
        edgecolor=COLORS["tcn_edge"],
        fontsize=5.9,
    )
    rounded_box(
        ax,
        8.44,
        y,
        1.25,
        h,
        "GroupNorm\nDropout",
        facecolor=COLORS["tcn_fill"],
        edgecolor=COLORS["tcn_edge"],
        fontsize=6.0,
    )

    add_center = (10.32, y + h / 2)
    add = Circle(
        add_center,
        radius=0.31,
        facecolor="white",
        edgecolor=COLORS["accent"],
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(add)
    ax.text(
        *add_center,
        "+",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["accent"],
        zorder=5,
    )
    rounded_box(
        ax,
        10.96,
        y,
        0.90,
        h,
        "GELU",
        facecolor=COLORS["decoder_fill"],
        edgecolor=COLORS["decoder_edge"],
        fontsize=6.4,
        weight="bold",
    )
    rounded_box(
        ax,
        12.18,
        y,
        0.84,
        h,
        "$\\mathbf{h}'$",
        facecolor="white",
        edgecolor=COLORS["neutral_edge"],
        fontsize=8.0,
        weight="bold",
    )

    cy = y + h / 2
    chain = [(3.16, 3.48), (4.73, 5.05), (6.55, 6.87), (8.12, 8.44)]
    for start_x, end_x in chain:
        arrow(
            ax,
            (start_x, cy),
            (end_x, cy),
            color=COLORS["tcn_edge"],
            mutation_scale=7.0,
        )
    arrow(ax, (9.69, cy), (10.01, cy), color=COLORS["tcn_edge"], mutation_scale=7.0)
    arrow(ax, (10.63, cy), (10.96, cy), color=COLORS["tcn_edge"], mutation_scale=7.0)
    arrow(ax, (11.86, cy), (12.18, cy), color=COLORS["tcn_edge"], mutation_scale=7.0)

    skip_y = 3.05
    ax.plot(
        [2.77, 2.77, 10.32],
        [y + h, skip_y, skip_y],
        color=COLORS["accent"],
        linewidth=1.05,
        zorder=2,
    )
    arrow(
        ax,
        (10.32, skip_y),
        (10.32, add_center[1] + 0.31),
        color=COLORS["accent"],
        linewidth=1.05,
        mutation_scale=7.0,
    )
    ax.text(
        6.55,
        skip_y + 0.12,
        "identity shortcut",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["accent"],
        style="italic",
    )

    ax.text(
        13.45,
        1.96,
        "Non-causal\nLength preserving\nDropout $p$ = 0.10",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
        linespacing=1.35,
    )


def render() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.25, 4.15))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_aspect("auto")
    ax.axis("off")

    ax.text(
        9.0,
        9.56,
        "Conv–TCN normal-behavior model",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        9.0,
        9.18,
        "9-channel, 2-s IMU window  →  compressed representation  →  full-window reconstruction",
        ha="center",
        va="center",
        fontsize=6.1,
        color=COLORS["muted"],
    )

    draw_main_architecture(ax)
    draw_residual_inset(ax)

    ax.text(
        9.0,
        0.15,
        "GN, GroupNorm (8 groups); GELU, Gaussian error linear unit. No encoder–decoder skip connections; output layer has no activation.",
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
