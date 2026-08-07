#!/usr/bin/env python3
"""Export S06 test probability, PR, and threshold diagnostics for S0-S3."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


METHODS = ("S0", "S1", "S2", "S3")
SEEDS = (2026, 2027, 2028)
SEED_COLORS = {2026: "#0F4D92", 2027: "#42949E", 2028: "#9A4D8E"}
METRIC_COLORS = {
    "recall": "#B64342",
    "specificity": "#0F4D92",
    "f1": "#D08A18",
}


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_path = root / "all_metrics.csv"
    if not metric_path.is_file():
        raise FileNotFoundError(metric_path)
    metrics = pd.read_csv(metric_path)
    metrics = metrics.loc[
        (metrics["subject_id"] == "S06")
        & (metrics["split"] == "test")
        & metrics["method"].isin(METHODS)
        & metrics["seed"].isin(SEEDS)
    ].copy()
    if len(metrics) != len(METHODS) * len(SEEDS):
        raise ValueError(f"Expected 12 S06 test metric rows, found {len(metrics)}")

    prediction_parts: list[pd.DataFrame] = []
    for method in METHODS:
        for seed in SEEDS:
            path = root / "S06" / f"seed_{seed}" / method / "predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, keep_default_na=False)
            frame = frame.loc[frame["evaluation_split"] == "test"].copy()
            if len(frame) != 351 or int(frame["y_binary"].sum()) != 32:
                raise ValueError(
                    f"Unexpected S06 test population for {method}/{seed}: "
                    f"n={len(frame)}, FoG={int(frame['y_binary'].sum())}"
                )
            frame["method"] = method
            frame["seed"] = seed
            prediction_parts.append(frame)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    expected_windows: set[str] | None = None
    for (_, _), group in predictions.groupby(["method", "seed"]):
        windows = set(group["window_id"].astype(str))
        if expected_windows is None:
            expected_windows = windows
        elif windows != expected_windows:
            raise ValueError("Test window IDs differ across method/seed models")
    return metrics.sort_values(["method", "seed"]), predictions


def threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    predicted = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))
    fp = int(np.sum(predicted & negative))
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return recall, specificity, f1


def build_curve_tables(
    metrics: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_grid = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                np.geomspace(1e-8, 1.0, 801),
                metrics["threshold"].to_numpy(dtype=float),
            )
        )
    )
    threshold_rows: list[dict[str, float | int | str]] = []
    pr_rows: list[dict[str, float | int | str | None]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        for seed in SEEDS:
            group = predictions.loc[
                (predictions["method"] == method) & (predictions["seed"] == seed)
            ].sort_values("window_id")
            labels = group["y_binary"].to_numpy(dtype=np.int8)
            probability = group["probability"].to_numpy(dtype=float)
            selected_threshold = float(
                metrics.loc[
                    (metrics["method"] == method) & (metrics["seed"] == seed),
                    "threshold",
                ].iloc[0]
            )
            precision, recall, pr_threshold = precision_recall_curve(labels, probability)
            average_precision = float(average_precision_score(labels, probability))
            for index, (precision_value, recall_value) in enumerate(zip(precision, recall)):
                pr_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "point_index": index,
                        "recall": float(recall_value),
                        "precision": float(precision_value),
                        "threshold": (
                            float(pr_threshold[index]) if index < len(pr_threshold) else None
                        ),
                    }
                )
            for threshold in threshold_grid:
                recall_value, specificity_value, f1_value = threshold_metrics(
                    labels, probability, float(threshold)
                )
                threshold_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "threshold": float(threshold),
                        "recall": recall_value,
                        "specificity": specificity_value,
                        "f1": f1_value,
                    }
                )
            fog_probability = probability[labels == 1]
            nonfog_probability = probability[labels == 0]
            recall_selected, specificity_selected, f1_selected = threshold_metrics(
                labels, probability, selected_threshold
            )
            summary_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_pr_auc": average_precision,
                    "test_prevalence": float(labels.mean()),
                    "validation_selected_threshold": selected_threshold,
                    "recall_at_selected_threshold": recall_selected,
                    "specificity_at_selected_threshold": specificity_selected,
                    "f1_at_selected_threshold": f1_selected,
                    "fog_probability_min": float(fog_probability.min()),
                    "fog_probability_median": float(np.median(fog_probability)),
                    "fog_probability_mean": float(fog_probability.mean()),
                    "fog_probability_max": float(fog_probability.max()),
                    "nonfog_probability_min": float(nonfog_probability.min()),
                    "nonfog_probability_median": float(np.median(nonfog_probability)),
                    "nonfog_probability_mean": float(nonfog_probability.mean()),
                    "nonfog_probability_max": float(nonfog_probability.max()),
                    "fog_windows_at_or_above_selected_threshold": int(
                        np.sum(fog_probability >= selected_threshold)
                    ),
                    "nonfog_windows_at_or_above_selected_threshold": int(
                        np.sum(nonfog_probability >= selected_threshold)
                    ),
                }
            )
    return (
        pd.DataFrame(threshold_rows),
        pd.DataFrame(pr_rows),
        pd.DataFrame(summary_rows),
    )


def build_fog_window_table(predictions: pd.DataFrame) -> pd.DataFrame:
    fog = predictions.loc[predictions["y_binary"] == 1].copy()
    metadata_columns = [
        "window_id",
        "record_id",
        "run_id",
        "group_id",
        "event_cluster_id",
        "overlapping_event_ids",
        "start_index",
        "end_index_exclusive",
        "start_time_sec",
        "end_time_sec",
    ]
    metadata = (
        fog.loc[(fog["method"] == "S0") & (fog["seed"] == 2026), metadata_columns]
        .drop_duplicates("window_id")
        .set_index("window_id")
    )
    pivot = fog.pivot(index="window_id", columns=["method", "seed"], values="probability")
    pivot.columns = [f"{method}_seed{seed}" for method, seed in pivot.columns]
    result = metadata.join(pivot).reset_index()
    for method in METHODS:
        columns = [f"{method}_seed{seed}" for seed in SEEDS]
        result[f"{method}_mean_probability"] = result[columns].mean(axis=1)
        result[f"{method}_min_probability"] = result[columns].min(axis=1)
        result[f"{method}_max_probability"] = result[columns].max(axis=1)
    result = result.sort_values(["record_id", "start_index"]).reset_index(drop=True)
    if len(result) != 32:
        raise ValueError(f"Expected 32 unique FoG windows, found {len(result)}")
    return result


def plot_diagnostic_grid(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold_curves: pd.DataFrame,
    pr_curves: pd.DataFrame,
    output_base: Path,
) -> None:
    all_positive = predictions.loc[predictions["probability"] > 0, "probability"].to_numpy()
    # Log axes receive strictly positive values; minimum is an explicit epsilon guard.
    minimum = max(float(all_positive.min()) * 0.75, 1e-8) if len(all_positive) else 1e-8
    bins = np.geomspace(minimum, 1.0, 55)
    figure, axes = plt.subplots(4, 3, figsize=(7.2, 8.2), constrained_layout=True)
    panel_index = 0
    for row_index, method in enumerate(METHODS):
        method_predictions = predictions.loc[predictions["method"] == method]
        method_metrics = metrics.loc[metrics["method"] == method]

        histogram_axis = axes[row_index, 0]
        for label, name, color in (
            (0, "Non-FoG", "#767676"),
            (1, "FoG", "#B64342"),
        ):
            values = method_predictions.loc[
                method_predictions["y_binary"] == label, "probability"
            ].to_numpy(dtype=float)
            values = np.clip(values, minimum, 1.0)
            histogram_axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.25,
                color=color,
                label=f"{name} ({len(values)} model-window scores)",
            )
        histogram_axis.set_xscale("log")
        histogram_axis.set_xlim(minimum, 1.0)
        histogram_axis.set_ylabel("Density")
        histogram_axis.set_title(f"{method} probability distributions")
        histogram_axis.legend(fontsize=5.2, loc="upper left")
        histogram_axis.grid(axis="x", alpha=0.18)

        pr_axis = axes[row_index, 1]
        for seed in SEEDS:
            curve = pr_curves.loc[
                (pr_curves["method"] == method) & (pr_curves["seed"] == seed)
            ]
            pr_auc = float(
                method_metrics.loc[method_metrics["seed"] == seed, "auprc"].iloc[0]
            )
            pr_axis.plot(
                curve["recall"],
                curve["precision"],
                color=SEED_COLORS[seed],
                linewidth=1.1,
                label=f"{seed}: {pr_auc:.3f}",
            )
        prevalence = 32 / 351
        pr_axis.axhline(
            prevalence,
            color="#767676",
            linestyle="--",
            linewidth=0.9,
            label=f"prevalence={prevalence:.3f}",
        )
        pr_axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
        pr_axis.set_title(f"{method} test PR curves")
        pr_axis.legend(fontsize=5.3, loc="upper right")
        pr_axis.grid(alpha=0.18)

        threshold_axis = axes[row_index, 2]
        curves = threshold_curves.loc[threshold_curves["method"] == method]
        aggregate = curves.groupby("threshold")[["recall", "specificity", "f1"]].agg(
            ["mean", "std"]
        )
        positive_thresholds = aggregate.index.to_numpy(dtype=float) > 0
        x_values = aggregate.index.to_numpy(dtype=float)[positive_thresholds]
        for metric_name in ("recall", "specificity", "f1"):
            mean = aggregate[(metric_name, "mean")].to_numpy(dtype=float)[positive_thresholds]
            std = aggregate[(metric_name, "std")].fillna(0.0).to_numpy(dtype=float)[
                positive_thresholds
            ]
            color = METRIC_COLORS[metric_name]
            threshold_axis.plot(
                x_values,
                mean,
                color=color,
                linewidth=1.2,
                label=f"{metric_name} mean",
            )
            threshold_axis.fill_between(
                x_values,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        for selected in method_metrics["threshold"].to_numpy(dtype=float):
            threshold_axis.axvline(selected, color="#4D4D4D", linewidth=0.7, alpha=0.35)
        threshold_axis.set_xscale("log")
        threshold_axis.set_xlim(max(minimum, 1e-8), 1.0)
        threshold_axis.set_ylim(-0.02, 1.02)
        threshold_axis.set(xlabel="Decision threshold", ylabel="Metric")
        threshold_axis.set_title(f"{method} threshold sensitivity")
        threshold_axis.legend(fontsize=5.1, loc="center right")
        threshold_axis.grid(axis="x", alpha=0.18)

        for column_index in range(3):
            axis = axes[row_index, column_index]
            axis.text(
                -0.14,
                1.04,
                chr(ord("a") + panel_index),
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            panel_index += 1

    for axis in axes[-1, :]:
        axis.set_xlabel(axis.get_xlabel())
    figure.suptitle(
        "S06 test diagnostics: probability overlap, PR ranking and threshold behavior",
        fontsize=9,
        fontweight="bold",
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    prevalence = 32 / 351
    lines = [
        "# S06测试集概率与PR诊断",
        "",
        f"测试集含32个FoG窗口和319个Non-FoG窗口，FoG比例为 `{prevalence:.4f}`。",
        "每个方法均包含2026、2027、2028三个独立训练种子。",
        "",
        "## PR-AUC及验证集所选阈值",
        "",
        "| 方法 | Seed | PR-AUC | 验证阈值 | Recall | Specificity | F1 | FoG≥阈值 | Non-FoG≥阈值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["method", "seed"]).itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.seed} | {row.test_pr_auc:.4f} | "
            f"{row.validation_selected_threshold:.4f} | "
            f"{row.recall_at_selected_threshold:.3f} | "
            f"{row.specificity_at_selected_threshold:.3f} | "
            f"{row.f1_at_selected_threshold:.3f} | "
            f"{row.fog_windows_at_or_above_selected_threshold} | "
            f"{row.nonfog_windows_at_or_above_selected_threshold} |"
        )
    lines.extend(
        [
            "",
            "## 方法级PR-AUC",
            "",
            "| 方法 | 三种子均值±SD | 相对随机排序基准 |",
            "|---|---:|---:|",
        ]
    )
    for method, group in summary.groupby("method", sort=False):
        mean = group["test_pr_auc"].mean()
        sd = group["test_pr_auc"].std(ddof=1)
        lines.append(f"| {method} | {mean:.4f} ± {sd:.4f} | {mean - prevalence:+.4f} |")
    lines.extend(
        [
            "",
            "## 数据说明",
            "",
            "- PR-AUC采用测试集Average Precision；随机排序的期望基准为测试集FoG比例。",
            "- 阈值来自对应种子的验证集，测试集没有重新选阈值。",
            "- 阈值扫描仅作诊断，不用于回写或修改测试结果。",
            "- 概率直方图使用对数横轴；只在绘图时将非正概率显示到最小正刻度，CSV保留原值。",
            "- `s06_fog32_window_probabilities.csv`逐行保存全部32个FoG窗口在12个模型中的概率。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/daphnet_ca_supcon_subject_v1",
        help="Completed CA-SupCon experiment directory",
    )
    args = parser.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    output = root / "S06_probability_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions = load_inputs(root)
    threshold_curves, pr_curves, summary = build_curve_tables(metrics, predictions)
    fog_windows = build_fog_window_table(predictions)

    metrics.to_csv(output / "s06_test_metrics_s0_s3.csv", index=False)
    predictions.to_csv(output / "s06_test_probabilities_all_windows.csv", index=False)
    summary.to_csv(output / "s06_probability_pr_summary.csv", index=False)
    threshold_curves.to_csv(output / "s06_threshold_metrics.csv", index=False)
    pr_curves.to_csv(output / "s06_pr_curve_points.csv", index=False)
    fog_windows.to_csv(output / "s06_fog32_window_probabilities.csv", index=False)
    write_report(summary, output / "S06_probability_diagnostic_report.md")
    plot_diagnostic_grid(
        metrics,
        predictions,
        threshold_curves,
        pr_curves,
        output / "S06_probability_PR_threshold_diagnostics",
    )
    print(f"Saved S06 diagnostics to: {output}")


if __name__ == "__main__":
    main()
