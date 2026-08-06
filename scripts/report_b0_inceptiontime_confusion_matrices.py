from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "outputs/daphnet_full_subject_tcndae_inceptiontime_server_v1/"
    "full_subject_binary_experiment"
)

SPLIT_LABELS = {
    "train": "Train",
    "validation": "Record validation",
    "test": "Outer test",
    "official_test_seed_median": "Official outer test",
}


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float | int]:
    total = tn + fp + fn + tp
    precision_0 = safe_div(tn, tn + fn)
    precision_1 = safe_div(tp, tp + fp)
    recall_0 = safe_div(tn, tn + fp)
    recall_1 = safe_div(tp, tp + fn)
    f1_0 = safe_div(2 * precision_0 * recall_0, precision_0 + recall_0)
    f1_1 = safe_div(2 * precision_1 * recall_1, precision_1 + recall_1)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_evaluations": int(total),
        "accuracy": safe_div(tn + tp, total),
        "macro_precision": (precision_0 + precision_1) / 2,
        "macro_recall": (recall_0 + recall_1) / 2,
        "macro_f1": (f1_0 + f1_1) / 2,
    }


def restore_counts(row: pd.Series) -> dict[str, int]:
    total = int(row["n_windows"])
    positive = int(row["positive_windows"])
    negative = total - positive
    tp = int(round(float(row["recall"]) * positive)) if positive else 0
    tn = int(round(float(row["specificity"]) * negative)) if negative else 0
    counts = {"tn": tn, "fp": negative - tn, "fn": positive - tp, "tp": tp}
    if min(counts.values()) < 0 or sum(counts.values()) != total:
        raise ValueError(f"Invalid reconstructed confusion counts: {counts}")
    return counts


def reconstruct_run_counts(run_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject_id",
        "fold_id",
        "method",
        "seed",
        "split",
        "n_windows",
        "positive_windows",
        "recall",
        "specificity",
    }
    missing = required.difference(run_metrics.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = run_metrics.loc[run_metrics["method"].eq("B0")].copy()
    restored = pd.DataFrame([restore_counts(row) for _, row in data.iterrows()])
    return pd.concat([data.reset_index(drop=True), restored], axis=1)


def seed_median_fold_counts(run_counts: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject_id", "fold_id", "split"]
    seed_counts = run_counts.groupby(keys, observed=True)["seed"].nunique()
    if not seed_counts.eq(3).all():
        bad = seed_counts.loc[~seed_counts.eq(3)]
        raise ValueError(f"Expected three seeds per fold/split; found:\n{bad}")
    grouped = (
        run_counts.groupby(keys, as_index=False, observed=True)[["tn", "fp", "fn", "tp"]]
        .median()
    )
    for column in ["tn", "fp", "fn", "tp"]:
        grouped[column] = grouped[column].round().astype(int)
    grouped["n_evaluations"] = grouped[["tn", "fp", "fn", "tp"]].sum(axis=1)
    grouped["aggregation"] = "component-wise median across 3 seeds"
    return grouped


def summarize_counts(fold_counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_columns = ["tn", "fp", "fn", "tp"]
    overall_rows: list[dict[str, object]] = []
    for split, frame in fold_counts.groupby("split", observed=True):
        sums = frame[count_columns].sum().astype(int)
        row: dict[str, object] = {
            "split": split,
            "aggregation": "fold sum after component-wise 3-seed median",
            "n_outer_folds": int(frame[["subject_id", "fold_id"]].drop_duplicates().shape[0]),
        }
        row.update(metrics_from_counts(**sums.to_dict()))
        overall_rows.append(row)

    subject_rows: list[dict[str, object]] = []
    for (subject_id, split), frame in fold_counts.groupby(
        ["subject_id", "split"], observed=True
    ):
        sums = frame[count_columns].sum().astype(int)
        row = {
            "subject_id": subject_id,
            "split": split,
            "n_outer_folds": int(frame["fold_id"].nunique()),
        }
        row.update(metrics_from_counts(**sums.to_dict()))
        subject_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(subject_rows)


def official_test_row(predictions: pd.DataFrame) -> dict[str, object]:
    data = predictions.loc[predictions["method"].eq("B0")].copy()
    y_true = data["y_true"].astype(int).to_numpy()
    y_pred = data["y_pred"].astype(int).to_numpy()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    row: dict[str, object] = {
        "split": "official_test_seed_median",
        "aggregation": (
            "unique outer-test windows; median probability and majority vote of "
            "seed-specific thresholded labels across 3 seeds"
        ),
        "n_outer_folds": int(data[["subject_id", "record_id"]].drop_duplicates().shape[0]),
    }
    row.update(metrics_from_counts(tn, fp, fn, tp))
    return row


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_confusion_matrices(
    overall: pd.DataFrame,
    output_stem: Path,
    figure_title: str = "Raw + InceptionTime: train, validation and outer-test confusion matrices",
) -> None:
    configure_plotting()
    main = overall.loc[overall["split"].isin(["train", "validation", "test"])].copy()
    main["split"] = pd.Categorical(
        main["split"], categories=["train", "validation", "test"], ordered=True
    )
    main = main.sort_values("split")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.64), constrained_layout=True)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "confusion_blue", ["#F4F7FA", "#9EC5E6", "#2F6FA7"]
    )
    for panel, (ax, (_, row)) in enumerate(zip(axes, main.iterrows())):
        counts = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        row_total = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            row_total,
            out=np.zeros_like(counts, dtype=float),
            where=row_total != 0,
        )
        ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(2):
            for j in range(2):
                color = "white" if normalized[i, j] >= 0.58 else "#17212B"
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]:,}\n{normalized[i, j] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7.2,
                    fontweight="bold" if i == j else "normal",
                )
        split = str(row["split"])
        ax.set_title(
            f"{SPLIT_LABELS[split]}\nACC {row['accuracy']:.3f} | Macro-F1 {row['macro_f1']:.3f}",
            pad=6,
        )
        ax.set_xticks([0, 1], ["Non-FoG", "FoG"])
        ax.set_yticks([0, 1], ["Non-FoG", "FoG"])
        ax.set_xlabel("Predicted class")
        if panel == 0:
            ax.set_ylabel("True class")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            -0.12,
            1.08,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    fig.suptitle(figure_title, fontsize=10)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def matrix_markdown(row: pd.Series) -> str:
    return (
        "| True \\ Predicted | Non-FoG | FoG |\n"
        "|---|---:|---:|\n"
        f"| Non-FoG | {int(row['tn']):,} | {int(row['fp']):,} |\n"
        f"| FoG | {int(row['fn']):,} | {int(row['tp']):,} |\n"
    )


def write_report(
    overall: pd.DataFrame,
    output_path: Path,
    method_title: str = "Raw + InceptionTime",
) -> None:
    main_order = ["train", "validation", "test", "official_test_seed_median"]
    indexed = overall.set_index("split")
    lines = [
        f"# {method_title}三集合混淆矩阵与主指标",
        "",
        "## 统计口径",
        "",
        "- 训练、验证、外层测试：每个外层折先对3个种子的TN/FP/FN/TP逐项取中位数，再跨30折汇总。",
        "- 训练和验证矩阵统计的是模型-窗口评估次数；同一窗口可能在不同外层折中被不同模型评估。",
        "- Official outer test中，每个外层测试窗口仅出现一次：概率取3种子中位数，类别取3个种子各自验证阈值预测的多数票。",
        "- Macro指标均为Non-FoG与FoG两个类别的算术平均，zero-division按0处理。",
        "",
        "## 四个主指标",
        "",
        "| 集合 | ACC | Macro-Precision | Macro-Recall | Macro-F1 | 评估数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in main_order:
        row = indexed.loc[split]
        lines.append(
            f"| {SPLIT_LABELS[split]} | {row['accuracy']:.4f} | "
            f"{row['macro_precision']:.4f} | {row['macro_recall']:.4f} | "
            f"{row['macro_f1']:.4f} | {int(row['n_evaluations']):,} |"
        )
    for split in main_order:
        row = indexed.loc[split]
        lines.extend(["", f"## {SPLIT_LABELS[split]}混淆矩阵", "", matrix_markdown(row).rstrip()])
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 三集合主矩阵用于观察模型拟合和泛化差距；最终测试性能优先引用Official outer test。",
            "- 因类别不平衡，ACC可能被Non-FoG多数类抬高；模型比较应同时查看Macro-Recall与Macro-F1。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit_dir = root / "analysis_nbm_three_set_audit"
    output_dir = root / "analysis_b0_confusion_matrices"
    run_metrics = pd.read_csv(audit_dir / "b0_control_run_fold_split_metrics.csv")
    pooled_predictions = pd.read_csv(root / "predictions" / "seed_median_pooled_predictions.csv")

    run_counts = reconstruct_run_counts(run_metrics)
    fold_counts = seed_median_fold_counts(run_counts)
    overall, subjects = summarize_counts(fold_counts)
    overall = pd.concat([overall, pd.DataFrame([official_test_row(pooled_predictions)])], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_counts.to_csv(output_dir / "run_seed_confusion_counts.csv", index=False)
    fold_counts.to_csv(output_dir / "fold_seed_median_confusion_counts.csv", index=False)
    subjects.to_csv(output_dir / "subject_split_metrics.csv", index=False)
    overall.to_csv(output_dir / "overall_split_metrics.csv", index=False)
    plot_confusion_matrices(overall, output_dir / "raw_inceptiontime_confusion_matrices")
    write_report(overall, output_dir / "raw_inceptiontime_confusion_matrix_report.md")
    print(overall.to_string(index=False))
    print(f"COMPLETE {output_dir}")


if __name__ == "__main__":
    main()
