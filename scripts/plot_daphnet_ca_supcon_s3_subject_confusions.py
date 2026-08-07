#!/usr/bin/env python3
"""Plot per-subject S3 test confusion matrices across the three training seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.linewidth": 0.8,
    }
)


SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
SEEDS = (2026, 2027, 2028)
CELL_NAMES = ("tn", "fp", "fn", "tp")


def load_s3_test_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "subject_id",
        "seed",
        "method",
        "split",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "threshold",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    result = frame.loc[(frame["method"] == "S3") & (frame["split"] == "test")].copy()
    result = result.loc[result["subject_id"].isin(SUBJECTS) & result["seed"].isin(SEEDS)]
    result = result.sort_values(["subject_id", "seed"]).reset_index(drop=True)
    expected = {(subject, seed) for subject in SUBJECTS for seed in SEEDS}
    observed = set(result[["subject_id", "seed"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"S3 test runs are incomplete; missing={sorted(expected - observed)}")
    for row in result.itertuples(index=False):
        if int(row.tn + row.fp) <= 0 or int(row.fn + row.tp) <= 0:
            raise ValueError(f"Both true classes are required: {row.subject_id}/{row.seed}")
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for subject, group in frame.groupby("subject_id", sort=False):
        row: dict[str, float | str | int] = {
            "subject_id": subject,
            "n_seeds": len(group),
            "test_nonfog_windows": int(group.iloc[0].tn + group.iloc[0].fp),
            "test_fog_windows": int(group.iloc[0].fn + group.iloc[0].tp),
        }
        for cell in CELL_NAMES:
            values = group[cell].to_numpy(dtype=float)
            row[f"{cell}_mean"] = float(values.mean())
            row[f"{cell}_sd"] = float(values.std(ddof=1))
        for metric in ("sensitivity", "specificity", "balanced_accuracy", "f1", "threshold"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def cell_annotation(name: str, mean: float, sd: float, rate: float) -> str:
    return f"{name.upper()}\n{mean:.1f}±{sd:.1f}\n{rate * 100:.1f}%"


def plot_confusions(summary: pd.DataFrame, output_base: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.25), constrained_layout=True)
    cmap = mpl.colormaps["Blues"]
    panel_labels = tuple("abcdefg")
    for panel, subject in enumerate(SUBJECTS):
        axis = axes.flat[panel]
        row = summary.loc[summary["subject_id"] == subject].iloc[0]
        mean = np.asarray(
            [[row["tn_mean"], row["fp_mean"]], [row["fn_mean"], row["tp_mean"]]],
            dtype=float,
        )
        sd = np.asarray(
            [[row["tn_sd"], row["fp_sd"]], [row["fn_sd"], row["tp_sd"]]],
            dtype=float,
        )
        normalized = mean / mean.sum(axis=1, keepdims=True)
        axis.imshow(normalized, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        names = np.asarray([["TN", "FP"], ["FN", "TP"]])
        for true_index in range(2):
            for predicted_index in range(2):
                rate = normalized[true_index, predicted_index]
                axis.text(
                    predicted_index,
                    true_index,
                    cell_annotation(
                        names[true_index, predicted_index],
                        mean[true_index, predicted_index],
                        sd[true_index, predicted_index],
                        rate,
                    ),
                    ha="center",
                    va="center",
                    color="white" if rate >= 0.55 else "#272727",
                    fontsize=5.8,
                    linespacing=1.15,
                )
        axis.set_xticks((0, 1), ("Non-FoG", "FoG"))
        axis.set_yticks((0, 1), ("Non-FoG", "FoG"))
        axis.tick_params(length=0, pad=2)
        axis.set_title(
            f"{subject}  BAcc={row['balanced_accuracy_mean']:.3f}",
            fontsize=7,
            pad=4,
        )
        axis.text(
            -0.14,
            1.06,
            panel_labels[panel],
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if panel % 4 == 0:
            axis.set_ylabel("True class")
        if panel >= 4:
            axis.set_xlabel("Predicted class")
        for spine in axis.spines.values():
            spine.set_visible(False)

    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    legend_axis.text(
        0.02,
        0.96,
        "S3 test matrices",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        transform=legend_axis.transAxes,
    )
    legend_axis.text(
        0.02,
        0.82,
        "Each subject is trained independently.\n"
        "Cells: mean count ± s.d. (3 seeds)\n"
        "and row-normalized percentage.\n\n"
        "Rows are true classes; columns are\n"
        "predicted classes. Color encodes the\n"
        "row-normalized percentage (0–100%).\n\n"
        "The same frozen test windows are used\n"
        "for all three seeds.",
        ha="left",
        va="top",
        fontsize=6.5,
        linespacing=1.35,
        transform=legend_axis.transAxes,
    )
    colorbar_axis = legend_axis.inset_axes([0.02, 0.08, 0.85, 0.08])
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((0.0, 0.5, 1.0))
    colorbar.set_ticklabels(("0%", "50%", "100%"))
    colorbar.ax.tick_params(length=2, pad=1, labelsize=6)
    figure.suptitle(
        "S3 per-subject test confusion matrices",
        fontsize=9,
        fontweight="bold",
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def write_markdown(frame: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# S3方法：各被试测试集混淆矩阵",
        "",
        "每名被试、每个随机种子对应一个独立训练模型。矩阵定义为 `[[TN, FP], [FN, TP]]`。",
        "3种子均值没有将测试集视为三倍样本，仅用于描述随机种子波动。",
        "",
        "| 被试 | Seed 2026 | Seed 2027 | Seed 2028 | 三种子均值 | Recall | Specificity | Balanced Acc. |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for subject in SUBJECTS:
        group = frame.loc[frame["subject_id"] == subject].set_index("seed")
        mean = summary.loc[summary["subject_id"] == subject].iloc[0]
        matrices = []
        for seed in SEEDS:
            row = group.loc[seed]
            matrices.append(f"[[{int(row.tn)}, {int(row.fp)}], [{int(row.fn)}, {int(row.tp)}]]")
        mean_matrix = (
            f"[[{mean.tn_mean:.1f}, {mean.fp_mean:.1f}], "
            f"[{mean.fn_mean:.1f}, {mean.tp_mean:.1f}]]"
        )
        lines.append(
            f"| {subject} | {matrices[0]} | {matrices[1]} | {matrices[2]} | {mean_matrix} | "
            f"{mean.sensitivity_mean:.3f} | {mean.specificity_mean:.3f} | "
            f"{mean.balanced_accuracy_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 图形说明",
            "",
            "- 色阶按真实类别逐行归一化，避免Non-FoG多数类掩盖FoG召回情况。",
            "- 每格依次显示单元名称、3种子计数均值±样本标准差、行归一化百分比。",
            "- `n=3`表示训练随机种子重复；每名被试的测试窗口在三个种子中相同。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/daphnet_ca_supcon_subject_v1",
        help="Completed CA-SupCon experiment directory",
    )
    args = parser.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    output = root / "S3_subject_confusion_matrices"
    frame = load_s3_test_metrics(root / "all_metrics.csv")
    summary = summarize(frame)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "s3_subject_seed_confusion_matrices.csv", index=False)
    summary.to_csv(output / "s3_subject_confusion_summary.csv", index=False)
    write_markdown(frame, summary, output / "S3_subject_confusion_matrices.md")
    plot_confusions(summary, output / "S3_subject_confusion_matrices")
    print(f"Saved S3 per-subject confusion matrices to: {output}")


if __name__ == "__main__":
    main()
