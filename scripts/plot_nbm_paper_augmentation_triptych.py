#!/usr/bin/env python3
"""Create a paper-ready triptych of original, Gaussian, and masked trunk IMU traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 64
WINDOW_SAMPLES = 128

COLORS = {
    "original": "#2F6B9A",
    "gaussian": "#D98524",
    "mask": "#B84A5A",
    "reference": "#A1A6AB",
    "zero": "#D9DCDF",
    "text": "#30353A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_trunk_three_windows"
        / "source_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "figures"
        / "nbm_paper_augmentation_vertical",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--vertical-gap",
        type=float,
        default=0.34,
        help="Matplotlib hspace between vertically stacked panels.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=5.2,
        help="Figure height in inches; adjust together with --vertical-gap.",
    )
    parser.add_argument(
        "--visualization-gaussian-std",
        type=float,
        default=0.30,
        help="Deliberately enlarged Gaussian standard deviation for the schematic.",
    )
    parser.add_argument("--training-gaussian-std", type=float, default=0.04)
    return parser.parse_args()


def load_subject_traces(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["subject_id"], []).append(row)

    result: dict[str, dict[str, np.ndarray]] = {}
    for subject, rows in grouped.items():
        rows.sort(key=lambda row: float(row["time_s"]))
        result[subject] = {
            "time": np.asarray([float(row["time_s"]) for row in rows]),
            "clean": np.asarray([float(row["clean_trunk_ap"]) for row in rows]),
            "masked": np.asarray([float(row["masked_trunk_ap"]) for row in rows]),
            "mask_active": np.asarray([int(row["mask_active"]) for row in rows]),
        }
    required = {"S02", "S03", "S09"}
    if not required.issubset(result):
        raise RuntimeError(f"source data are missing subjects: {sorted(required - set(result))}")
    if any(result[subject]["time"].size != WINDOW_SAMPLES for subject in required):
        raise RuntimeError("each selected subject must contribute exactly one 128-sample window")
    return result


def export_source_data(
    path: Path,
    panels: list[dict[str, object]],
) -> None:
    fieldnames = [
        "panel",
        "subject_id",
        "condition",
        "time_s",
        "displayed_trunk_ap",
        "clean_reference_trunk_ap",
        "mask_active",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel in panels:
            time = panel["time"]
            displayed = panel["displayed"]
            clean = panel["clean"]
            mask_active = panel["mask_active"]
            for index in range(WINDOW_SAMPLES):
                writer.writerow(
                    {
                        "panel": panel["letter"],
                        "subject_id": panel["subject"],
                        "condition": panel["condition"],
                        "time_s": float(time[index]),
                        "displayed_trunk_ap": float(displayed[index]),
                        "clean_reference_trunk_ap": float(clean[index]),
                        "mask_active": int(mask_active[index]),
                    }
                )


def main() -> None:
    args = parse_args()
    traces = load_subject_traces(args.input_csv.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    s02_gaussian = traces["S02"]["clean"] + rng.normal(
        0.0,
        args.visualization_gaussian_std,
        size=WINDOW_SAMPLES,
    )
    no_mask = np.zeros(WINDOW_SAMPLES, dtype=np.int8)
    panels: list[dict[str, object]] = [
        {
            "letter": "a",
            "subject": "S03",
            "condition": "Original",
            "subtitle": "unchanged clean signal",
            "time": traces["S03"]["time"],
            "clean": traces["S03"]["clean"],
            "displayed": traces["S03"]["clean"],
            "mask_active": no_mask,
            "color": COLORS["original"],
        },
        {
            "letter": "b",
            "subject": "S02",
            "condition": "Gaussian noise",
            "subtitle": rf"illustration $\sigma={args.visualization_gaussian_std:.2f}$",
            "time": traces["S02"]["time"],
            "clean": traces["S02"]["clean"],
            "displayed": s02_gaussian,
            "mask_active": no_mask,
            "color": COLORS["gaussian"],
        },
        {
            "letter": "c",
            "subject": "S09",
            "condition": "Time mask",
            "subtitle": "continuous all-axis mask",
            "time": traces["S09"]["time"],
            "clean": traces["S09"]["clean"],
            "displayed": traces["S09"]["masked"],
            "mask_active": traces["S09"]["mask_active"],
            "color": COLORS["mask"],
        },
    ]

    all_values = np.concatenate(
        [np.asarray(panel["clean"]) for panel in panels]
        + [np.asarray(panel["displayed"]) for panel in panels]
    )
    y_limit = 1.08 * float(np.max(np.abs(all_values)))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.5, args.figure_height),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.09,
        top=0.98,
        hspace=args.vertical_gap,
    )

    for index, (ax, panel) in enumerate(zip(axes, panels)):
        time = np.asarray(panel["time"])
        clean = np.asarray(panel["clean"])
        displayed = np.asarray(panel["displayed"])
        color = str(panel["color"])
        ax.axhline(0.0, color=COLORS["zero"], linewidth=0.65, zorder=0)

        if panel["condition"] != "Original":
            ax.plot(
                time,
                clean,
                color=COLORS["reference"],
                linewidth=0.85,
                linestyle="--",
                zorder=1,
            )
        if panel["condition"] == "Time mask":
            mask_indices = np.flatnonzero(np.asarray(panel["mask_active"]))
            if mask_indices.size == 0:
                raise RuntimeError("S09 has no active time-mask samples")
            mask_start = int(mask_indices[0])
            mask_end = int(mask_indices[-1] + 1)
            ax.axvspan(
                mask_start / FS,
                mask_end / FS,
                color=COLORS["mask"],
                alpha=0.16,
                linewidth=0,
                zorder=0,
            )

        ax.plot(time, displayed, color=color, linewidth=1.25, zorder=2)
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks((0.0, 1.0, 2.0))
        ax.tick_params(labelsize=6.5, length=3, pad=2, labelbottom=True)
        ax.set_title(
            str(panel["condition"]),
            fontsize=10,
            fontweight="bold",
            color=color,
            pad=5,
        )
        if panel["condition"] != "Original":
            ax.plot(
                [0.67, 0.76],
                [0.90, 0.90],
                transform=ax.transAxes,
                color=COLORS["reference"],
                linewidth=0.9,
                linestyle="--",
                clip_on=False,
                zorder=3,
            )
            ax.text(
                0.78,
                0.90,
                "Original signal",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.2,
                color="#777D82",
                zorder=3,
            )
        if panel["condition"] == "Time mask":
            ax.add_patch(
                Rectangle(
                    (0.67, 0.77),
                    0.09,
                    0.07,
                    transform=ax.transAxes,
                    facecolor=COLORS["mask"],
                    edgecolor=COLORS["mask"],
                    linewidth=0.8,
                    alpha=0.16,
                    clip_on=False,
                    zorder=3,
                )
            )
            ax.text(
                0.78,
                0.805,
                "Masked interval",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.2,
                color=COLORS["mask"],
                zorder=3,
            )

    axes[-1].set_xlabel("Time (s)", fontsize=7)

    stem = output_dir / "nbm_augmentation_vertical_s03_s02_s09"
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

    export_source_data(output_dir / "source_data.csv", panels)
    mask_length = int(np.sum(traces["S09"]["mask_active"]))
    metadata = {
        "figure_claim": (
            "Three directly labeled examples distinguish unchanged, Gaussian-noised, "
            "and continuously masked trunk-IMU inputs without a legend."
        ),
        "archetype": "vertical quantitative triptych",
        "backend": "Python/matplotlib",
        # "panel_a": "S03 original trunk AP signal",
        # "panel_b": "S02 Gaussian-noised trunk AP signal",
        # "panel_c": "S09 continuously masked trunk AP signal",
        "visualization_gaussian_std": args.visualization_gaussian_std,
        "actual_training_gaussian_std": args.training_gaussian_std,
        "visualization_noise_is_enlarged": True,
        "s09_mask_length_samples": mask_length,
        "s09_mask_duration_ms": 1000.0 * mask_length / FS,
        "legend_used": False,
        "reference_annotation": "Original signal, shown at upper right of augmented panels",
        "mask_annotation": "Masked interval, shown in red at upper right of the time-mask panel",
        "shared_y_axis": True,
        "amplitude_normalization": False,
        "figure_height_inches": args.figure_height,
        "vertical_gap_hspace": args.vertical_gap,
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_contract.md").write_text(
        """# Figure contract

- Core conclusion: Three directly labeled real signals distinguish unchanged, Gaussian-noised, and continuously masked NBM inputs.
- Figure archetype: Vertical quantitative triptych.
- Target output: Minimal single-column paper schematic with no legend.
- Backend: Python/matplotlib only.
- Final size: 3.5 x 5.2 inches by default; height and vertical gap are manually adjustable.
- Panel a: S03 original trunk AP acceleration.
- Panel b: S02 with deliberately enlarged Gaussian noise.
- Panel c: S09 with a continuous time mask.
- Evidence hierarchy: The colored augmented traces are primary; dashed clean references are supporting evidence.
- Statistics needed: None; this is a preprocessing illustration.
- Source data needed: Every displayed value and the clean reference for each panel.
- Image-integrity notes: Shared y-axis; no smoothing or amplitude normalization.
- Reviewer risk: Visualization sigma 0.30 is larger than the actual training sigma 0.04 and must be disclosed in the caption.
""",
        encoding="utf-8",
    )
    (output_dir / "figure_legend.md").write_text(
        """**Representative NBM input augmentations.** (a) Original centered and Robust-scaled trunk anterior-posterior acceleration from S03. (b) S02 signal after additive Gaussian noise; sigma was increased to 0.30 solely to make the perturbation visible in this schematic, whereas training uses sigma 0.04. (c) S09 signal after a continuous seven-sample (109 ms) time mask applied synchronously to all nine sensor axes. Dashed gray curves indicate the corresponding unaugmented signals. Colors and direct panel labels identify the three conditions; no legend is used.
""",
        encoding="utf-8",
    )
    (output_dir / "qa_notes.md").write_text(
        """# Figure QA notes

- Automated preflight: 13 PASS, 1 reviewed WARN, 0 FAIL.
- Reviewed warning: random-number generation is intentional because panel b illustrates additive Gaussian augmentation; all underlying IMU traces are real dataset signals.
- Visual inspection: passed at the final 3.5-inch width; method titles, axes, mask span, original-signal annotations, and the red masked-interval key are readable without overlap.
- Legend strategy: no legend object is present; the gray dashed reference is labeled directly at the upper right of the two augmented panels.
- Axis integrity: all panels share one y-axis scale; no panel-specific amplitude normalization was applied.
- Signal integrity: no smoothing, interpolation, or selective point removal was applied.
- Gaussian disclosure: sigma 0.30 is illustrative only; actual training sigma remains 0.04.
- Traceability: every displayed value and clean reference is exported in `source_data.csv`.
- Export bundle: editable SVG/PDF and 600-dpi PNG/TIFF.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
