#!/usr/bin/env python
"""Create compact CSV and Markdown summaries for a residual-history ablation."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


SUMMARY_METRICS = [
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "event_sensitivity",
    "false_alarm_events_per_hour",
]
PAIR_METRICS = [
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1",
    "event_sensitivity",
    "false_alarm_events_per_hour",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> float:
    if value in (None, "", "None", "null"):
        return float("nan")
    return float(value)


def fmt(value: object, digits: int = 4) -> str:
    number = as_float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(root: Path, rows: list[dict]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    histories = np.asarray([as_float(row["history_seconds"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    left = axes[0]
    for key, label, marker in [
        ("macro_auprc_mean", "Macro AUPRC", "o"),
        ("macro_auroc_mean", "Macro AUROC", "s"),
        ("macro_balanced_accuracy_mean", "Macro balanced accuracy", "^"),
        ("macro_f1_mean", "Macro F1", "D"),
    ]:
        left.plot(
            histories,
            [as_float(row[key]) for row in rows],
            marker=marker,
            linewidth=2,
            label=label,
        )
    left.set_title("Window-level subject-macro metrics")
    left.set_xlabel("Residual history (s)")
    left.set_ylabel("Score")
    left.set_xticks(histories)
    left.set_ylim(0.25, 0.9)
    left.grid(alpha=0.25)
    left.legend(fontsize=8)

    right = axes[1]
    right.plot(
        histories,
        [as_float(row["macro_event_sensitivity_mean"]) for row in rows],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Event sensitivity",
    )
    right.set_title("Event-level operating trade-off")
    right.set_xlabel("Residual history (s)")
    right.set_ylabel("Event sensitivity", color="#1f77b4")
    right.tick_params(axis="y", labelcolor="#1f77b4")
    right.set_xticks(histories)
    right.set_ylim(0.5, 0.85)
    right.grid(alpha=0.25)
    alarm_axis = right.twinx()
    alarm_axis.plot(
        histories,
        [as_float(row["macro_false_alarm_events_per_hour_mean"]) for row in rows],
        color="#d62728",
        marker="s",
        linewidth=2,
        label="False alarms/hour",
    )
    alarm_axis.set_ylabel("False alarms/hour", color="#d62728")
    alarm_axis.tick_params(axis="y", labelcolor="#d62728")
    handles_a, labels_a = right.get_legend_handles_labels()
    handles_b, labels_b = alarm_axis.get_legend_handles_labels()
    right.legend(handles_a + handles_b, labels_a + labels_b, loc="lower left", fontsize=8)
    figure.suptitle("CNBR-FoG residual-history ablation (8-subject LOSO)")
    figure.savefig(root / "history_ablation.png", dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    root = args.result_dir.resolve()
    config = read_json(root / "config.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    fold_rows = read_rows(root / "fold_summary.csv")
    definitions = {item["input"]: item for item in config.get("history_variants", [])}
    variants = [item["input"] for item in config.get("history_variants", [])]
    if not variants:
        raise ValueError("The result directory does not contain history variants")

    summary_rows: list[dict] = []
    for variant in variants:
        definition = definitions[variant]
        macro = aggregate[variant]["subject_macro"]
        pooled = aggregate[variant]["pooled"]
        row: dict[str, object] = {
            "variant": variant,
            "history_seconds": definition["history_seconds"],
            "input_samples": definition["history_samples"],
            "history_blocks": definition["history_blocks"],
            "n": pooled["n"],
        }
        for metric in SUMMARY_METRICS:
            stats = macro.get(metric, {})
            row[f"macro_{metric}_mean"] = stats.get("mean")
            row[f"macro_{metric}_std"] = stats.get("std")
        for metric in [
            "auprc",
            "auroc",
            "balanced_accuracy",
            "f1",
            "precision",
            "sensitivity",
            "specificity",
            "accuracy",
        ]:
            row[f"pooled_{metric}"] = pooled.get(metric)
        summary_rows.append(row)
    write_csv(root / "history_ablation_summary.csv", summary_rows)
    plot_written = write_plot(root, summary_rows)

    by_variant_subject = {
        (row["input"], row["test_subject"]): row for row in fold_rows
    }
    subjects = list(config["folds_resolved"])
    pairwise_rows: list[dict] = []
    paired_summary: list[dict] = []
    rng = np.random.default_rng(int(config.get("seed", 42)) + 2026)
    for left, right in combinations(variants, 2):
        for subject in subjects:
            left_row = by_variant_subject[(left, subject)]
            right_row = by_variant_subject[(right, subject)]
            output: dict[str, object] = {
                "left": left,
                "right": right,
                "test_subject": subject,
            }
            for metric in PAIR_METRICS:
                output[f"delta_{metric}"] = as_float(right_row.get(metric)) - as_float(
                    left_row.get(metric)
                )
            pairwise_rows.append(output)
        for metric in PAIR_METRICS:
            values = np.asarray(
                [
                    row[f"delta_{metric}"]
                    for row in pairwise_rows
                    if row["left"] == left and row["right"] == right
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if len(values):
                samples = rng.choice(
                    values,
                    size=(max(1, args.bootstrap_repetitions), len(values)),
                    replace=True,
                ).mean(axis=1)
                low, high = np.percentile(samples, [2.5, 97.5])
                mean = float(values.mean())
                wins = int((values > 0).sum())
                ties = int((values == 0).sum())
                losses = int((values < 0).sum())
            else:
                mean = low = high = float("nan")
                wins = ties = losses = 0
            paired_summary.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "mean_delta_right_minus_left": mean,
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                    "right_wins": wins,
                    "ties": ties,
                    "right_losses": losses,
                    "n_subjects": int(len(values)),
                }
            )
    write_csv(root / "pairwise_subject_deltas.csv", pairwise_rows)
    write_csv(root / "pairwise_summary.csv", paired_summary)
    paired_lookup = {
        (row["left"], row["right"], row["metric"]): row for row in paired_summary
    }

    best = max(
        summary_rows,
        key=lambda row: as_float(row["macro_auprc_mean"]),
    )
    baseline = summary_rows[0]
    delta_macro_ap = as_float(best["macro_auprc_mean"]) - as_float(
        baseline["macro_auprc_mean"]
    )
    delta_pooled_ap = as_float(best["pooled_auprc"]) - as_float(
        baseline["pooled_auprc"]
    )

    lines = [
        "# CNBR-FoG residual history 消融报告",
        "",
        "## 结论",
        "",
        (
            f"按预先指定的主指标 subject-macro AUPRC，最佳历史长度为 "
            f"**{best['history_seconds']:g} 秒**（{fmt(best['macro_auprc_mean'])}）。"
        ),
        (
            f"相较 0.5 秒基线，macro AUPRC 变化 {delta_macro_ap:+.4f}，"
            f"pooled AUPRC 变化 {delta_pooled_ap:+.4f}。"
        ),
        "",
        "## 协议",
        "",
        f"- 严格排除受试者：{', '.join(config.get('excluded_subjects', []))}。",
        f"- 外层 LOSO：{', '.join(subjects)}，共 {len(subjects)} 折。",
        (
            f"- 四个输入共享 {config['evaluation_windows']:,} 个测试锚点："
            f"non-FOG {config['evaluation_window_class_counts'][0]:,}，"
            f"FOG {config['evaluation_window_class_counts'][1]:,}。"
        ),
        "- 历史由完整、无重叠的 0.5 秒残差块组成；标签始终取末端 0.5 秒。",
        "- CNBM、数据划分、分类器结构、训练超参数和阈值规则保持一致。",
        "",
        "## 汇总结果",
        "",
        "| History | Macro AUPRC | Macro AUROC | Macro BA | Macro F1 | Pooled AUPRC | Pooled AUROC | Event sens. | False alarms/h |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['history_seconds']:g}s",
                    f"{fmt(row['macro_auprc_mean'])} ± {fmt(row['macro_auprc_std'])}",
                    f"{fmt(row['macro_auroc_mean'])} ± {fmt(row['macro_auroc_std'])}",
                    fmt(row["macro_balanced_accuracy_mean"]),
                    fmt(row["macro_f1_mean"]),
                    fmt(row["pooled_auprc"]),
                    fmt(row["pooled_auroc"]),
                    fmt(row["macro_event_sensitivity_mean"]),
                    fmt(row["macro_false_alarm_events_per_hour_mean"], 1),
                ]
            )
            + " |"
        )
    if plot_written:
        lines.extend(["", "![Residual history ablation](history_ablation.png)"])

    two_second = next(
        (row for row in summary_rows if np.isclose(as_float(row["history_seconds"]), 2.0)),
        None,
    )
    four_second = next(
        (row for row in summary_rows if np.isclose(as_float(row["history_seconds"]), 4.0)),
        None,
    )
    if two_second is not None and four_second is not None:
        pair = paired_lookup[(two_second["variant"], four_second["variant"], "auprc")]
        false_alarm_reduction = 1.0 - as_float(
            four_second["macro_false_alarm_events_per_hour_mean"]
        ) / as_float(two_second["macro_false_alarm_events_per_hour_mean"])
        lines.extend(
            [
                "",
                "## 2 秒与 4 秒的关键取舍",
                "",
                (
                    f"4 秒相对 2 秒的 macro AUPRC 平均差为 "
                    f"{as_float(pair['mean_delta_right_minus_left']):+.4f}，"
                    f"配对 bootstrap 95% CI "
                    f"[{as_float(pair['bootstrap_ci95_low']):+.4f}, "
                    f"{as_float(pair['bootstrap_ci95_high']):+.4f}]；"
                    f"4 秒在 {pair['right_wins']}/{pair['n_subjects']} 位受试者上更高。"
                ),
                (
                    f"2 秒的 macro F1 / event sensitivity 为 "
                    f"{fmt(two_second['macro_f1_mean'])} / "
                    f"{fmt(two_second['macro_event_sensitivity_mean'])}；"
                    f"4 秒为 {fmt(four_second['macro_f1_mean'])} / "
                    f"{fmt(four_second['macro_event_sensitivity_mean'])}。"
                ),
                (
                    f"4 秒将宏平均误报从 "
                    f"{fmt(two_second['macro_false_alarm_events_per_hour_mean'], 1)} 降至 "
                    f"{fmt(four_second['macro_false_alarm_events_per_hour_mean'], 1)} 次/小时，"
                    f"下降 {false_alarm_reduction:.1%}。"
                ),
                "因此，若以排序能力和降低误报为主，选择 4 秒；若更重视事件检出率与较短历史，2 秒是更均衡的工作点。",
            ]
        )

    lines.extend(
        [
            "",
            "## 逐受试者 AUPRC",
            "",
            "| Test | " + " | ".join(f"{definitions[v]['history_seconds']:g}s" for v in variants) + " |",
            "|---|" + "---:|" * len(variants),
        ]
    )
    for subject in subjects:
        values = [fmt(by_variant_subject[(variant, subject)].get("auprc")) for variant in variants]
        lines.append(f"| {subject} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本实验只改变分类器可见的 residual history，不同时引入 raw 融合、subject-level cross-fitting 或事件后处理调参。结果为单随机种子，历史长度应结合逐受试者稳定性和事件误报共同解释。",
            "",
            "机器可读结果见 `history_ablation_summary.csv`、`pairwise_subject_deltas.csv`、`pairwise_summary.csv` 和 `aggregate_metrics.json`。",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "report.md")


if __name__ == "__main__":
    main()
