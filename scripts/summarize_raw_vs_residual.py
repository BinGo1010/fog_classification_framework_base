#!/usr/bin/env python
"""Audit and summarize matched raw-TCN versus CNBR-residual TCN experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
]
DISPLAY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Acc.",
    "macro_f1": "Macro-F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "fog_recall": "FoG Recall",
    "fog_f1": "FoG F1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def history_map(config: dict) -> dict[float, str]:
    return {
        float(item["history_seconds"]): str(item["input"])
        for item in config["history_variants"]
    }


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall_fog = tp / (tp + fn) if tp + fn else 0.0
    recall_nonfog = tn / (tn + fp) if tn + fp else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "accuracy": (tn + tp) / len(y_true),
        "balanced_accuracy": 0.5 * (recall_fog + recall_nonfog),
        "macro_f1": 0.5 * (f1_fog + f1_nonfog),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "fog_recall": recall_fog,
        "fog_f1": f1_fog,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def load_predictions(
    root: Path,
    subject: str,
    input_name: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    fold_root = root / f"loso_{subject}" / input_name
    saved_metrics = load_json(fold_root / "metrics.json")
    with np.load(fold_root / "predictions.npz", allow_pickle=False) as payload:
        arrays = {
            "window_index": np.asarray(payload["window_index"], dtype=np.int64),
            "y_true": np.asarray(payload["y_true"], dtype=np.int8),
            "y_prob": np.asarray(payload["y_prob"], dtype=np.float64),
            "y_pred": np.asarray(payload["y_pred"], dtype=np.int8),
        }
    expected = (
        arrays["y_prob"] >= float(saved_metrics["threshold"])
    ).astype(np.int8)
    if not np.array_equal(expected, arrays["y_pred"]):
        raise AssertionError(f"Threshold mismatch: {root.name}/{subject}/{input_name}")
    recomputed = classification_metrics(
        arrays["y_true"], arrays["y_prob"], arrays["y_pred"]
    )
    saved_aliases = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "auroc",
        "pr_auc": "auprc",
        "fog_recall": "sensitivity",
        "fog_f1": "f1",
    }
    for key, saved_key in saved_aliases.items():
        if not np.isclose(
            recomputed[key],
            float(saved_metrics[saved_key]),
            rtol=1e-8,
            atol=1e-8,
        ):
            raise AssertionError(
                f"Saved metric mismatch: {root.name}/{subject}/{input_name}/{key}"
            )
    return saved_metrics, arrays


def exact_signflip_p(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    observed = abs(float(deltas.mean()))
    null = [
        abs(float(np.mean(deltas * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(deltas))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bootstrap_ci(
    deltas: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = np.asarray(deltas, dtype=np.float64)[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def audit_protocol(raw_dir: Path, residual_dir: Path) -> tuple[dict, dict]:
    raw_config = load_json(raw_dir / "config.json")
    residual_config = load_json(residual_dir / "config.json")
    if bool(raw_config.get("uses_nbm", True)):
        raise AssertionError("Raw run is not marked uses_nbm=false")
    for key in (
        "subjects",
        "folds_resolved",
        "excluded_subjects",
        "context_samples",
        "horizon_samples",
        "stride_samples",
        "evaluation_windows",
        "evaluation_window_class_counts",
        "classifier_hidden",
        "dropout",
        "classifier_epochs",
        "classifier_patience",
        "batch_size",
        "classifier_lr",
        "weight_decay",
        "max_classifier_windows",
        "robust_clip",
        "seed",
    ):
        if raw_config[key] != residual_config[key]:
            raise AssertionError(f"Protocol mismatch for {key}")
    if set(raw_config["excluded_subjects"]) != {"S04", "S10"}:
        raise AssertionError("Expected strict exclusion of S04 and S10")
    if raw_config["evaluation_windows"] != 53387:
        raise AssertionError("Unexpected common support size")
    if raw_config["evaluation_window_class_counts"] != [46449, 6938]:
        raise AssertionError("Unexpected common support class counts")

    raw_histories = history_map(raw_config)
    residual_histories = history_map(residual_config)
    if set(raw_histories) != set(residual_histories):
        raise AssertionError("History durations differ")

    support_keys = [
        "train_anchor_window_index",
        "validation_anchor_window_index",
        "test_anchor_window_index",
        "train_history_window_index",
        "validation_history_window_index",
        "test_history_window_index",
    ]
    for subject in raw_config["folds_resolved"]:
        raw_fold = raw_dir / f"loso_{subject}"
        residual_fold = residual_dir / f"loso_{subject}"
        if (raw_fold / "normal_predictor_best.pt").exists():
            raise AssertionError(f"Raw fold contains NBM checkpoint: {subject}")
        raw_fold_config = load_json(raw_fold / "fold_config.json")
        residual_fold_config = load_json(residual_fold / "fold_config.json")
        for key in ("test_subject", "val_subject", "train_subjects", "scaler"):
            if raw_fold_config[key] != residual_fold_config[key]:
                raise AssertionError(f"Fold mismatch {subject}/{key}")
        with np.load(
            raw_fold / "history_support.npz", allow_pickle=False
        ) as raw_support, np.load(
            residual_fold / "history_support.npz", allow_pickle=False
        ) as residual_support:
            for key in support_keys:
                if not np.array_equal(raw_support[key], residual_support[key]):
                    raise AssertionError(f"Support mismatch {subject}/{key}")
    return raw_config, residual_config


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    residual_dir = args.residual_dir.resolve()
    output_dir = (args.output_dir or raw_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config, residual_config = audit_protocol(raw_dir, residual_dir)
    subjects = list(raw_config["folds_resolved"])
    raw_histories = history_map(raw_config)
    residual_histories = history_map(residual_config)
    histories = sorted(raw_histories)

    per_subject_rows: list[dict] = []
    fold_values: dict[tuple[float, str, str], list[float]] = {}
    pooled_arrays: dict[tuple[float, str, str], list[np.ndarray]] = {}
    for history in histories:
        raw_name = raw_histories[history]
        residual_name = residual_histories[history]
        for subject in subjects:
            raw_saved, raw_arrays = load_predictions(raw_dir, subject, raw_name)
            residual_saved, residual_arrays = load_predictions(
                residual_dir, subject, residual_name
            )
            for key in ("window_index", "y_true"):
                if not np.array_equal(raw_arrays[key], residual_arrays[key]):
                    raise AssertionError(
                        f"Raw/residual target mismatch: {history}s/{subject}/{key}"
                    )
            for key in ("test_subject", "val_subject", "classifier_seed"):
                if raw_saved[key] != residual_saved[key]:
                    raise AssertionError(
                        f"Classifier protocol mismatch: {history}s/{subject}/{key}"
                    )
            raw_metrics = classification_metrics(
                raw_arrays["y_true"], raw_arrays["y_prob"], raw_arrays["y_pred"]
            )
            residual_metrics = classification_metrics(
                residual_arrays["y_true"],
                residual_arrays["y_prob"],
                residual_arrays["y_pred"],
            )
            row = {
                "history_seconds": history,
                "test_subject": subject,
                "val_subject": raw_saved["val_subject"],
                "n": len(raw_arrays["y_true"]),
                "n_fog": int(raw_arrays["y_true"].sum()),
                "raw_threshold": raw_saved["threshold"],
                "residual_threshold": residual_saved["threshold"],
            }
            for metric in METRICS:
                raw_value = raw_metrics[metric]
                residual_value = residual_metrics[metric]
                row[f"raw_{metric}"] = raw_value
                row[f"residual_{metric}"] = residual_value
                row[f"delta_{metric}"] = residual_value - raw_value
                fold_values.setdefault((history, "raw", metric), []).append(raw_value)
                fold_values.setdefault(
                    (history, "residual", metric), []
                ).append(residual_value)
            per_subject_rows.append(row)
            for representation, arrays in (
                ("raw", raw_arrays),
                ("residual", residual_arrays),
            ):
                for key in ("y_true", "y_prob", "y_pred"):
                    pooled_arrays.setdefault((history, representation, key), []).append(
                        arrays[key]
                    )

    pooled_metrics_by_rep: dict[tuple[float, str], dict] = {}
    for history in histories:
        for representation in ("raw", "residual"):
            arrays = {
                key: np.concatenate(
                    pooled_arrays[(history, representation, key)]
                )
                for key in ("y_true", "y_prob", "y_pred")
            }
            pooled_metrics_by_rep[(history, representation)] = (
                classification_metrics(
                    arrays["y_true"], arrays["y_prob"], arrays["y_pred"]
                )
            )

    raw_metric_rows: list[dict] = []
    for history in histories:
        row = {
            "history_seconds": history,
            "n_subjects": len(subjects),
            "n_windows": raw_config["evaluation_windows"],
        }
        pooled = pooled_metrics_by_rep[(history, "raw")]
        for metric in METRICS:
            values = np.asarray(
                fold_values[(history, "raw", metric)], dtype=np.float64
            )
            row[f"subject_macro_{metric}"] = float(values.mean())
            row[f"subject_std_{metric}"] = float(values.std(ddof=0))
            row[f"pooled_{metric}"] = pooled[metric]
        raw_metric_rows.append(row)

    rng = np.random.default_rng(args.seed)
    paired_summary_rows: list[dict] = []
    for history in histories:
        for metric in METRICS:
            raw_values = np.asarray(
                fold_values[(history, "raw", metric)], dtype=np.float64
            )
            residual_values = np.asarray(
                fold_values[(history, "residual", metric)], dtype=np.float64
            )
            deltas = residual_values - raw_values
            low, high = bootstrap_ci(deltas, rng, args.bootstrap_samples)
            paired_summary_rows.append(
                {
                    "history_seconds": history,
                    "metric": metric,
                    "delta_definition": "residual_minus_raw",
                    "n_pairs": len(deltas),
                    "raw_subject_mean": float(raw_values.mean()),
                    "raw_subject_std": float(raw_values.std(ddof=0)),
                    "residual_subject_mean": float(residual_values.mean()),
                    "residual_subject_std": float(residual_values.std(ddof=0)),
                    "delta_mean": float(deltas.mean()),
                    "delta_std": float(deltas.std(ddof=0)),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "residual_wins": int((deltas > 1e-12).sum()),
                    "ties": int((np.abs(deltas) <= 1e-12).sum()),
                    "residual_losses": int((deltas < -1e-12).sum()),
                    "exact_signflip_p": exact_signflip_p(deltas),
                    "raw_pooled": pooled_metrics_by_rep[(history, "raw")][metric],
                    "residual_pooled": pooled_metrics_by_rep[
                        (history, "residual")
                    ][metric],
                    "pooled_delta": (
                        pooled_metrics_by_rep[(history, "residual")][metric]
                        - pooled_metrics_by_rep[(history, "raw")][metric]
                    ),
                }
            )

    write_csv(output_dir / "raw_classification_metrics.csv", raw_metric_rows)
    write_csv(
        output_dir / "raw_vs_residual_paired_subjects.csv", per_subject_rows
    )
    write_csv(
        output_dir / "raw_vs_residual_paired_summary.csv", paired_summary_rows
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    colors = {"raw": "#3B82F6", "residual": "#EF4444"}
    for representation in ("raw", "residual"):
        for metric, marker in (
            ("pr_auc", "o"),
            ("roc_auc", "s"),
            ("balanced_accuracy", "^"),
            ("macro_f1", "D"),
        ):
            values = [
                np.mean(fold_values[(history, representation, metric)])
                for history in histories
            ]
            axes[0].plot(
                histories,
                values,
                marker=marker,
                color=colors[representation],
                linestyle="-" if metric in {"pr_auc", "roc_auc"} else "--",
                alpha=1.0 if metric in {"pr_auc", "roc_auc"} else 0.65,
                label=f"{representation} {DISPLAY[metric]}",
            )
    axes[0].set_title("Subject-macro discrimination metrics")
    axes[0].set_xlabel("Input history (s)")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(histories)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    for representation in ("raw", "residual"):
        for metric, marker in (("fog_recall", "o"), ("fog_f1", "s")):
            values = [
                np.mean(fold_values[(history, representation, metric)])
                for history in histories
            ]
            axes[1].plot(
                histories,
                values,
                marker=marker,
                color=colors[representation],
                linestyle="-" if metric == "fog_recall" else "--",
                label=f"{representation} {DISPLAY[metric]}",
            )
    axes[1].set_title("Subject-macro FoG detection metrics")
    axes[1].set_xlabel("Input history (s)")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(histories)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    fig.savefig(output_dir / "raw_vs_residual_comparison.png", dpi=180)
    plt.close(fig)

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    summary_lookup = {
        (float(row["history_seconds"]), row["metric"]): row
        for row in paired_summary_rows
    }
    raw_best_pr = max(
        raw_metric_rows, key=lambda row: float(row["subject_macro_pr_auc"])
    )
    four_pr = summary_lookup[(4.0, "pr_auc")]
    four_recall = summary_lookup[(4.0, "fog_recall")]

    report: list[str] = [
        "# Raw-TCN 与 CNBR residual-TCN 严格配对报告",
        "",
        "## 协议",
        "",
        "- S04、S10 在缩放、窗口构造和 LOSO 之前完全排除。",
        "- 测试受试者为 S01、S02、S03、S05、S06、S07、S08、S09。",
        "- raw 与 residual 使用相同的 53,387 个公共测试锚点（46,449 non-FoG；6,938 FoG）。",
        "- 两者共享每折 train/validation/test 受试者、scaler、历史支持、TCN、随机种子、训练预算和阈值规则。",
        "- raw-TCN 不创建、训练或调用 NBM；输入为训练折 scaler 处理后的原始腰部三轴加速度。",
        "",
        "## 主要结论",
        "",
        (
            f"- Raw-TCN 的最高 subject-macro PR-AUC 出现在 "
            f"{float(raw_best_pr['history_seconds']):g} 秒："
            f"{fmt(float(raw_best_pr['subject_macro_pr_auc']))}。"
        ),
        (
            f"- 在 4 秒配对下，residual 相对 raw 的 subject-macro PR-AUC "
            f"差值为 {float(four_pr['delta_mean']):+.4f}，95% CI "
            f"[{float(four_pr['delta_ci95_low']):+.4f}, "
            f"{float(four_pr['delta_ci95_high']):+.4f}]；"
            "未显示稳定的 residual 排序优势。"
        ),
        (
            f"- 4 秒 residual 的 FoG Recall 比 raw 高 "
            f"{float(four_recall['delta_mean']):+.4f}，但其 95% CI "
            f"[{float(four_recall['delta_ci95_low']):+.4f}, "
            f"{float(four_recall['delta_ci95_high']):+.4f}] 跨过 0。"
        ),
        (
            f"- 4 秒 pooled PR-AUC 则由 residual 占优："
            f"{fmt(float(four_pr['residual_pooled']))} vs "
            f"{fmt(float(four_pr['raw_pooled']))}。"
            "这说明受试者等权结果与窗口池化结果存在异质性。"
        ),
        "- 总体上，CNBR residual 更倾向于提高 FoG Recall，而 raw-TCN 在部分历史长度上保留了更高 Accuracy 或排序指标；不存在所有指标一致占优的单一表征。",
        "",
        "## Raw-TCN subject-macro 指标",
        "",
        "| History | Accuracy | Balanced Acc. | Macro-F1 | ROC-AUC | PR-AUC | FoG Recall | FoG F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in raw_metric_rows:
        cells = [
            f"{row['history_seconds']:g}s",
            *[
                f"{fmt(row[f'subject_macro_{metric}'])} ± "
                f"{fmt(row[f'subject_std_{metric}'])}"
                for metric in METRICS
            ],
        ]
        report.append("| " + " | ".join(cells) + " |")
    report.extend(
        [
            "",
            "## Raw-TCN pooled 指标",
            "",
            "| History | Accuracy | Balanced Acc. | Macro-F1 | ROC-AUC | PR-AUC | FoG Recall | FoG F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in raw_metric_rows:
        cells = [
            f"{row['history_seconds']:g}s",
            *[fmt(row[f"pooled_{metric}"]) for metric in METRICS],
        ]
        report.append("| " + " | ".join(cells) + " |")

    report.extend(
        [
            "",
            "## Residual − raw 的 subject-macro 配对差值",
            "",
            "正值表示 residual-TCN 更高；负值表示 raw-TCN 更高。",
            "",
            "| History | Accuracy | Balanced Acc. | Macro-F1 | ROC-AUC | PR-AUC | FoG Recall | FoG F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for history in histories:
        cells = [f"{history:g}s"]
        for metric in METRICS:
            item = summary_lookup[(history, metric)]
            cells.append(
                f"{float(item['delta_mean']):+.4f} "
                f"[{float(item['delta_ci95_low']):+.4f}, "
                f"{float(item['delta_ci95_high']):+.4f}]"
            )
        report.append("| " + " | ".join(cells) + " |")
    report.extend(
        [
            "",
            "区间为按 8 位受试者配对 bootstrap 的 95% CI。",
            "",
            "![Raw versus residual comparison](raw_vs_residual_comparison.png)",
            "",
            "## 解释边界",
            "",
            "raw_hD 与 residual_hD 的分类器输入长度相同，但 residual 的每个 0.5 秒块还由前置 2 秒 context 条件化。因此该实验隔离的是“是否使用 CNBR residual 表征”的系统级差异，并不是底层原始观察范围完全相同的比较。",
            "",
            "本结果为单随机种子。显著性比较以受试者为配对单位；不能把高度重叠的窗口当作独立样本。",
            "",
            "审计状态：`PAIRING_AUDIT_OK`。",
        ]
    )
    (output_dir / "raw_vs_residual_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("PAIRING_AUDIT_OK")
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
