#!/usr/bin/env python3
"""Aggregate CA-SUPCON-SUBJECT-V1 outputs into CSV, plots, and a report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTOCOL_ID = "CA-SUPCON-SUBJECT-V1"
FORMAL_SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
METHODS = ("S0", "S1", "S2", "S3")
METRICS = (
    "auprc",
    "precision",
    "sensitivity",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "auroc",
    "accuracy",
)


def parse_csv(text: str, cast) -> list[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def collect_rows(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("S*/seed_*/S[0-3]/metrics.json")):
        payload = read_json(path)
        if payload.get("protocol_id") != PROTOCOL_ID:
            continue
        subject = path.parents[2].name
        seed = int(path.parents[1].name.removeprefix("seed_"))
        method = str(payload["method"])
        for split in ("validation", "test"):
            metrics = payload[split]
            row: dict[str, Any] = {
                "subject_id": subject,
                "seed": seed,
                "method": method,
                "split": split,
                "threshold": payload["selected_threshold"],
                "n": metrics.get("n"),
                "n_fog": metrics.get("n_fog"),
                "n_nonfog": metrics.get("n_normal"),
                "tn": metrics.get("tn"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "tp": metrics.get("tp"),
            }
            row.update({key: metrics.get(key) for key in METRICS})
            if split == "test":
                event = payload.get("test_event", {})
                row.update(
                    {
                        "fog_event_sensitivity": event.get("fog_event_sensitivity"),
                        "detected_fog_events": event.get("detected_fog_events"),
                        "total_fog_events": event.get("total_fog_events_with_pure_windows"),
                        "false_alarms_per_hour": event.get("false_alarms_per_hour"),
                        "mean_detection_latency_sec": event.get("mean_detection_latency_sec"),
                    }
                )
            metric_rows.append(row)
        representation_rows.append(
            {
                "subject_id": subject,
                "seed": seed,
                "method": method,
                **payload.get("representation", {}),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(representation_rows)


def validate_completeness(
    frame: pd.DataFrame,
    subjects: list[str],
    seeds: list[int],
    allow_incomplete: bool,
) -> list[str]:
    available = set(
        frame.loc[frame["split"] == "test", ["subject_id", "seed", "method"]]
        .itertuples(index=False, name=None)
    )
    expected = {(subject, seed, method) for subject in subjects for seed in seeds for method in METHODS}
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    messages: list[str] = []
    if missing:
        messages.append(f"Missing {len(missing)} runs: {missing[:12]}")
    if unexpected:
        messages.append(f"Unexpected {len(unexpected)} runs: {unexpected[:12]}")
    if missing and not allow_incomplete:
        raise RuntimeError(messages[0])
    return messages


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[frame["split"] == "test"].copy()
    columns = list(METRICS) + [
        "fog_event_sensitivity",
        "false_alarms_per_hour",
        "mean_detection_latency_sec",
    ]
    rows: list[dict[str, Any]] = []
    for method, group in test.groupby("method", sort=True):
        for metric in columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[frame["split"] == "test"].copy()
    comparisons = (("H1", "S1", "S0"), ("H2", "S2", "S1"), ("H3", "S3", "S2"))
    rows: list[dict[str, Any]] = []
    for hypothesis, candidate, reference in comparisons:
        for metric in METRICS:
            pivot = test.pivot_table(
                index=["subject_id", "seed"], columns="method", values=metric, aggfunc="first"
            )
            if candidate not in pivot or reference not in pivot:
                continue
            delta = (pivot[candidate] - pivot[reference]).dropna()
            for (subject, seed), value in delta.items():
                rows.append(
                    {
                        "hypothesis": hypothesis,
                        "candidate": candidate,
                        "reference": reference,
                        "subject_id": subject,
                        "seed": int(seed),
                        "metric": metric,
                        "delta": float(value),
                    }
                )
    return pd.DataFrame(rows)


def gate_summary(delta: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    definitions = {
        "H1": ("S1", "S0", ("auprc", "f1", "balanced_accuracy", "sensitivity")),
        "H2": ("S2", "S1", ("auprc", "f1")),
        "H3": ("S3", "S2", ("auprc", "f1", "balanced_accuracy")),
    }
    for hypothesis, (candidate, reference, main_metrics) in definitions.items():
        subset = delta.loc[delta["hypothesis"] == hypothesis]
        metrics: dict[str, Any] = {}
        for metric in sorted(set(main_metrics) | {"specificity"}):
            values = subset.loc[subset["metric"] == metric, "delta"].to_numpy(dtype=float)
            metrics[metric] = {
                "n_pairs": int(len(values)),
                "mean_delta": float(values.mean()) if len(values) else None,
                "median_delta": float(np.median(values)) if len(values) else None,
                "positive_fraction": float((values > 0).mean()) if len(values) else None,
                "specificity_drop_over_5pp_fraction": (
                    float((values < -0.05).mean()) if metric == "specificity" and len(values) else None
                ),
            }
        main_positive = [
            metrics[metric]["positive_fraction"] for metric in main_metrics if metrics[metric]["positive_fraction"] is not None
        ]
        specificity_warning = metrics["specificity"]["specificity_drop_over_5pp_fraction"]
        supported = bool(
            main_positive
            and sum(value > 0.5 for value in main_positive) >= math.ceil(len(main_positive) / 2)
            and (specificity_warning is None or specificity_warning <= 0.25)
        )
        result[hypothesis] = {
            "candidate": candidate,
            "reference": reference,
            "automatic_summary_only": True,
            "supported_by_majority_rule": supported,
            "metrics": metrics,
        }
    return result


def collect_data_audits(output_root: Path, subjects: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        path = output_root / subject / "data_audit.json"
        if not path.exists():
            continue
        payload = read_json(path)
        for split, stats in payload["splits"].items():
            rows.append({"subject_id": subject, "split": split, **stats})
    return pd.DataFrame(rows)


def plot_seed_scatter(frame: pd.DataFrame, output: Path) -> None:
    test = frame.loc[frame["split"] == "test"].copy()
    plot_metrics = ("auprc", "f1", "sensitivity", "specificity", "balanced_accuracy")
    colors = {method: color for method, color in zip(METHODS, ("#6B7280", "#2A9D8F", "#E9A03B", "#C44E52"))}
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    for axis, metric in zip(axes.flat, plot_metrics):
        for method_index, method in enumerate(METHODS):
            group = test.loc[test["method"] == method]
            jitter = np.asarray(
                [((hash((row.subject_id, int(row.seed))) % 100) / 100.0 - 0.5) * 0.22 for row in group.itertuples()]
            )
            axis.scatter(
                np.full(len(group), method_index) + jitter,
                group[metric],
                s=28,
                alpha=0.75,
                color=colors[method],
                label=method,
            )
        axis.set_xticks(range(4), METHODS)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("CA-SupCon test metrics: each point is one subject/seed")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_aggregate_confusions(frame: pd.DataFrame, output: Path) -> None:
    test = frame.loc[frame["split"] == "test"]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))
    for axis, method in zip(axes, METHODS):
        group = test.loc[test["method"] == method]
        matrix = np.asarray(
            [[group["tn"].sum(), group["fp"].sum()], [group["fn"].sum(), group["tp"].sum()]],
            dtype=int,
        )
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        axis.set_xticks((0, 1), ("Non-FoG", "FoG"), rotation=20)
        axis.set_yticks((0, 1), ("Non-FoG", "FoG"))
        axis.set_title(method)
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.suptitle("Pooled test confusion counts (descriptive; not a pooled-subject model)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NE"
    return f"{float(value):.{digits}f}"


def markdown_report(
    output_root: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    gates: dict[str, Any],
    audits: pd.DataFrame,
    warnings_list: list[str],
) -> str:
    test = frame.loc[frame["split"] == "test"]
    subject_count = int(test["subject_id"].nunique())
    seed_count = int(test["seed"].nunique())
    expected_runs = (
        test["subject_id"].nunique() * test["seed"].nunique() * len(METHODS)
        if not test.empty
        else 0
    )
    pivot = summary.pivot(index="method", columns="metric", values="mean")
    lines = [
        "# 单被试 CA-SupCon FoG 类别不平衡验证实验报告",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 数据：`processed_CA_pure` 冻结划分；训练集拟合 RobustScaler；验证/测试保持自然比例",
        f"- 完成运行：{len(test)} / {expected_runs} 个被试-种子-方法组合",
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if warnings_list:
        lines.extend(["## 完整性告警", ""] + [f"- {item}" for item in warnings_list] + [""])
    lines.extend(
        [
            "## 冻结数据统计",
            "",
            "| 被试 | 集合 | FoG事件 | FoG窗口 | Non-FoG片段 | Non-FoG窗口 | FoG比例 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audits.sort_values(["subject_id", "split"]).itertuples():
        lines.append(
            f"| {row.subject_id} | {row.split} | {row.n_fog_events} | {row.n_fog} | "
            f"{row.n_nonfog_segments} | {row.n_nonfog} | {row.fog_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            f"## 测试集主要结果（{subject_count}被试 x {seed_count}种子宏平均）",
            "",
            "| 方法 | PR-AUC | Precision | Recall | F1 | Specificity | Balanced Acc. | MCC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = pivot.loc[method] if method in pivot.index else pd.Series(dtype=float)
        lines.append(
            f"| {method} | {fmt(row.get('auprc'))} | {fmt(row.get('precision'))} | "
            f"{fmt(row.get('sensitivity'))} | {fmt(row.get('f1'))} | "
            f"{fmt(row.get('specificity'))} | {fmt(row.get('balanced_accuracy'))} | "
            f"{fmt(row.get('mcc'))} |"
        )
    lines.extend(["", "## 假设方向审计", ""])
    descriptions = {
        "H1": "S1 相对 S0：事件感知平衡采样",
        "H2": "S2 相对 S1：CA-SupCon 表征",
        "H3": "S3 相对 S2：平衡分类器重训",
    }
    for key in ("H1", "H2", "H3"):
        gate = gates[key]
        status = "多数配对支持" if gate["supported_by_majority_rule"] else "未达到多数配对支持"
        pr = gate["metrics"]["auprc"]
        f1 = gate["metrics"]["f1"]
        specificity = gate["metrics"]["specificity"]
        lines.append(
            f"- **{key}（{descriptions[key]}）**：{status}；PR-AUC平均差 {fmt(pr['mean_delta'])}，"
            f"F1平均差 {fmt(f1['mean_delta'])}，Specificity下降超过5个百分点的配对比例 "
            f"{fmt(specificity['specificity_drop_over_5pp_fraction'])}。"
        )
    lines.extend(
        [
            "",
            "> 注：以上门控是自动方向汇总，不替代对逐被试、逐种子、事件级结果和特征图的人工判断。"
            "测试阈值完全来自对应验证集，温度选择也完全来自验证集 linear probe。",
            "",
            "## 输出索引",
            "",
            "- `all_metrics.csv`：验证/测试逐运行指标",
            "- `paired_deltas.csv`：H1/H2/H3 配对差值",
            "- `representation_diagnostics.csv`：类内与类间距离",
            "- `seed_metric_scatter.png`：逐被试/种子散点",
            "- `aggregate_confusion_matrices.png`：S0–S3 描述性汇总混淆矩阵",
            "- 每个 `Sxx/seed_xxxx/Sx/`：曲线、t-SNE、事件时序、假阳性案例、预测明细与检查点",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subjects", default=",".join(FORMAL_SUBJECTS))
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    subjects = parse_csv(args.subjects, str)
    seeds = parse_csv(args.seeds, int)
    metrics, representation = collect_rows(output_root)
    if metrics.empty:
        raise RuntimeError(f"No {PROTOCOL_ID} metrics found under {output_root}")
    warnings_list = validate_completeness(metrics, subjects, seeds, args.allow_incomplete)
    summary = summarize_metrics(metrics)
    delta = paired_deltas(metrics)
    gates = gate_summary(delta)
    audits = collect_data_audits(output_root, subjects)

    metrics.to_csv(output_root / "all_metrics.csv", index=False)
    summary.to_csv(output_root / "method_metric_summary.csv", index=False)
    delta.to_csv(output_root / "paired_deltas.csv", index=False)
    representation.to_csv(output_root / "representation_diagnostics.csv", index=False)
    audits.to_csv(output_root / "data_split_audit.csv", index=False)
    write_json(output_root / "hypothesis_gate_summary.json", gates)
    plot_seed_scatter(metrics, output_root / "seed_metric_scatter.png")
    plot_aggregate_confusions(metrics, output_root / "aggregate_confusion_matrices.png")
    report = markdown_report(output_root, metrics, summary, gates, audits, warnings_list)
    (output_root / "CA_SupCon_experiment_report.md").write_text(report, encoding="utf-8")
    write_json(
        output_root / "aggregation_complete.json",
        {
            "protocol_id": PROTOCOL_ID,
            "subjects": subjects,
            "seeds": seeds,
            "n_test_runs": int((metrics["split"] == "test").sum()),
            "warnings": warnings_list,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Aggregation complete: {output_root / 'CA_SupCon_experiment_report.md'}")


if __name__ == "__main__":
    main()
