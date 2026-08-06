from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_nbm_routeA_A2_A4 as prior  # noqa: E402
import run_daphnet_nbm_routeA_final_residual_validation as route_a  # noqa: E402


EXPERIMENT = "daphnet_nbm_routeA_A5_v1"
SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
SELECTION_SUBJECTS = ("S01", "S05", "S08", "S09")
DIAGNOSTIC_ONLY_DURING_SELECTION = ("S02", "S06", "S07")
SEEDS = (20260802, 20260803, 20260804)
SCORES = ("S0", "S1", "S2", "S3")
FS = 64
WINDOW = 128
STRIDE_SECONDS = 1.0
PR_MARGIN_ABOVE_RANDOM = 0.02

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, ensure_ascii=False, default=json_default, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_median(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    return float(np.median(values)) if values else math.nan


def component_scores(residual: np.ndarray) -> np.ndarray:
    """Return template A5 scores S0, S1 and S2 for every window."""
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (WINDOW, 9):
        raise ValueError(f"expected [N,{WINDOW},9] residual, got {values.shape}")
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float64)
    absolute = np.abs(values)
    s0 = np.median(absolute, axis=(1, 2))
    channel_error = np.median(absolute, axis=1)
    s1 = np.mean(np.sort(channel_error, axis=1)[:, -3:], axis=1)

    window = np.hanning(WINDOW).reshape(1, WINDOW, 1)
    spectrum = np.abs(np.fft.rfft(values * window, axis=1)) ** 2
    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    low = np.mean(spectrum[:, (frequency >= 0.5) & (frequency <= 3.0), :], axis=(1, 2))
    high = np.mean(spectrum[:, (frequency >= 3.0) & (frequency <= 8.0), :], axis=(1, 2))
    s2 = low + high
    return np.column_stack((s0, s1, s2)).astype(np.float64)


def fit_component_scale(train_components: np.ndarray) -> np.ndarray:
    """Positive train-only scaling keeps S3 interpretable as an anomaly magnitude."""
    train = np.asarray(train_components, dtype=np.float64)
    scale = np.median(train, axis=0)
    fallback = np.percentile(train, 75, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    return np.maximum(scale, 1e-12)


def combine_components(components: np.ndarray, scale: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(components, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    return values @ np.asarray(weights, dtype=np.float64)


def simplex_weights(step: float = 0.1) -> list[tuple[float, float, float]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one exactly")
    return [
        (i / units, j / units, (units - i - j) / units)
        for i in range(units + 1)
        for j in range(units - i + 1)
    ]


def cliffs_delta(normal: np.ndarray, fog: np.ndarray) -> float:
    normal = np.asarray(normal, dtype=np.float64)
    fog = np.asarray(fog, dtype=np.float64)
    return float(
        (sum(float(value > ref) - float(value < ref) for value in fog for ref in normal))
        / (len(normal) * len(fog))
    )


def separation_metrics(normal: np.ndarray, fog: np.ndarray, train_normal: np.ndarray) -> dict[str, float]:
    normal = np.asarray(normal, dtype=np.float64)
    fog = np.asarray(fog, dtype=np.float64)
    train_normal = np.asarray(train_normal, dtype=np.float64)
    if min(len(normal), len(fog), len(train_normal)) == 0:
        raise ValueError("A5 requires non-empty train Non-FoG, evaluation Non-FoG, and FoG")
    y = np.concatenate((np.zeros(len(normal), dtype=int), np.ones(len(fog), dtype=int)))
    score = np.concatenate((normal, fog))
    threshold = float(np.percentile(train_normal, 95))
    prevalence = float(len(fog) / len(y))
    if len(normal) >= 2 and len(fog) >= 2:
        pooled = math.sqrt(
            max(
                ((len(normal) - 1) * np.var(normal, ddof=1) + (len(fog) - 1) * np.var(fog, ddof=1))
                / max(len(normal) + len(fog) - 2, 1),
                1e-12,
            )
        )
        correction = 1.0 - 3.0 / max(4.0 * (len(normal) + len(fog)) - 9.0, 1.0)
        hedges = float(((np.mean(fog) - np.mean(normal)) / pooled) * correction)
    else:
        hedges = math.nan
    nonfog_median = float(np.median(normal))
    ratio = float(np.median(fog) / max(nonfog_median, 1e-12))
    false_alarm_fraction = float(np.mean(normal > threshold))
    return {
        "nonfog_p50": nonfog_median,
        "nonfog_p90": float(np.percentile(normal, 90)),
        "nonfog_p95": float(np.percentile(normal, 95)),
        "fog_p50": float(np.percentile(fog, 50)),
        "fog_p90": float(np.percentile(fog, 90)),
        "fog_to_nonfog_median_ratio": ratio,
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "random_pr_baseline": prevalence,
        "pr_margin_over_random": float(average_precision_score(y, score) - prevalence),
        "recall_at_train_nonfog_p95": float(np.mean(fog > threshold)),
        "nonfog_false_alarm_fraction": false_alarm_fraction,
        "false_alarm_windows_per_minute": float(false_alarm_fraction * 60.0 / STRIDE_SECONDS),
        "cliffs_delta": cliffs_delta(normal, fog),
        "hedges_g": hedges,
        "train_nonfog_p95_threshold": threshold,
    }


def run_gate(metrics: dict[str, float]) -> tuple[bool, bool]:
    usable = bool(
        metrics["auroc"] >= 0.65
        and metrics["fog_to_nonfog_median_ratio"] > 1.10
        and metrics["cliffs_delta"] > 0.20
        and metrics["recall_at_train_nonfog_p95"] >= 0.20
    )
    strong = bool(
        metrics["auroc"] >= 0.75
        and metrics["pr_margin_over_random"] >= PR_MARGIN_ABOVE_RANDOM
        and metrics["false_alarm_windows_per_minute"] <= 0.5
    )
    return usable, strong


def score_arrays(
    component_sets: dict[str, np.ndarray], scale: np.ndarray, score_name: str, weights: Sequence[float]
) -> dict[str, np.ndarray]:
    if score_name in ("S0", "S1", "S2"):
        index = SCORES.index(score_name)
        return {name: values[:, index] for name, values in component_sets.items()}
    if score_name == "S3":
        return {name: combine_components(values, scale, weights) for name, values in component_sets.items()}
    raise ValueError(score_name)


def summarize_candidate(rows: Sequence[dict[str, Any]], score_name: str) -> dict[str, Any]:
    chosen = [
        row
        for row in rows
        if row["score"] == score_name and row["subject_id"] in SELECTION_SUBJECTS
    ]
    return {
        "score": score_name,
        "selection_runs": len(chosen),
        "median_validation_auroc": finite_median(chosen, "auroc"),
        "median_validation_average_precision": finite_median(chosen, "average_precision"),
        "median_validation_cliffs_delta": finite_median(chosen, "cliffs_delta"),
        "median_validation_fog_nonfog_ratio": finite_median(chosen, "fog_to_nonfog_median_ratio"),
        "median_validation_false_alarm_per_minute": finite_median(chosen, "false_alarm_windows_per_minute"),
    }


def save_figure(fig: plt.Figure, root: Path, name: str) -> None:
    output = root / "figures" / name
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def subject_summary(rows: Sequence[dict[str, Any]], subject: str) -> dict[str, Any]:
    chosen = [row for row in rows if row["subject_id"] == subject]
    summary = {
        "subject_id": subject,
        "seed_runs": len(chosen),
        "evaluable_run_count": sum(bool(row["evaluable"]) for row in chosen),
        "usable_run_count": sum(bool(row["usable_run_pass"]) for row in chosen),
        "strong_run_count": sum(bool(row["strong_run_pass"]) for row in chosen),
    }
    for key in (
        "auroc",
        "average_precision",
        "random_pr_baseline",
        "pr_margin_over_random",
        "fog_to_nonfog_median_ratio",
        "recall_at_train_nonfog_p95",
        "false_alarm_windows_per_minute",
        "cliffs_delta",
        "hedges_g",
        "validation_to_test_auroc_drop",
        "validation_to_test_pr_drop",
    ):
        summary[f"median_{key}"] = finite_median(chosen, key)
    summary["usable_subject_pass"] = bool(
        summary["evaluable_run_count"] == len(SEEDS)
        and summary["median_auroc"] >= 0.65
        and summary["median_fog_to_nonfog_median_ratio"] > 1.10
        and summary["median_cliffs_delta"] > 0.20
        and summary["median_recall_at_train_nonfog_p95"] >= 0.20
    )
    summary["strong_subject_pass"] = bool(
        summary["evaluable_run_count"] == len(SEEDS)
        and summary["median_auroc"] >= 0.75
        and summary["median_pr_margin_over_random"] >= PR_MARGIN_ABOVE_RANDOM
        and summary["median_false_alarm_windows_per_minute"] <= 0.5
    )
    return summary


def plot_candidate_comparison(root: Path, candidates: Sequence[dict[str, Any]]) -> None:
    names = [row["score"] for row in candidates]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for panel, (ax, key, title) in enumerate(zip(
        axes,
        ("median_validation_auroc", "median_validation_average_precision", "median_validation_cliffs_delta"),
        ("Validation AUROC", "Validation PR-AUC", "Validation Cliff's delta"),
    )):
        ax.bar(x, [row[key] for row in candidates], color="#4472C4")
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, root, "validation_score_comparison")
    plt.close(fig)


def plot_subject_metrics(root: Path, summaries: Sequence[dict[str, Any]]) -> None:
    names = [row["subject_id"] for row in summaries]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = (
        ("median_auroc", "Test AUROC", 0.65),
        ("median_average_precision", "Test PR-AUC", None),
        ("median_fog_to_nonfog_median_ratio", "FoG / Non-FoG median", 1.10),
        ("median_false_alarm_windows_per_minute", "False alarms / min", 0.5),
    )
    for panel, (ax, (key, title, threshold)) in enumerate(zip(axes.flat, panels)):
        colors = ["#4472C4" if row["usable_subject_pass"] else "#C55A11" for row in summaries]
        ax.bar(x, [row[key] for row in summaries], color=colors)
        if key == "median_average_precision":
            ax.plot(x, [row["median_random_pr_baseline"] for row in summaries], "ko--", label="random")
            ax.legend()
        if threshold is not None:
            ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for index, row in enumerate(summaries):
            if row["evaluable_run_count"] == 0:
                ax.text(index, 0.03, "NE", ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.text(-0.10, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, root, "selected_score_subject_metrics")
    plt.close(fig)


def plot_gate_matrix(root: Path, summaries: Sequence[dict[str, Any]]) -> None:
    matrix = np.asarray([
        [-1.0, -1.0]
        if row["evaluable_run_count"] == 0
        else [float(row["usable_subject_pass"]), float(row["strong_subject_pass"])]
        for row in summaries
    ])
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap=ListedColormap(["#BFBFBF", "#C55A11", "#4472C4"]),
        aspect="auto",
    )
    ax.set_xticks([0, 1], ["Usable", "Strong"])
    ax.set_yticks(np.arange(len(summaries)), [row["subject_id"] for row in summaries])
    for y in range(len(summaries)):
        for x in range(2):
            label = "NE" if matrix[y, x] < 0 else "PASS" if matrix[y, x] > 0 else "FAIL"
            ax.text(x, y, label, ha="center", va="center")
    ax.set_title("A5 subject gates")
    fig.tight_layout()
    save_figure(fig, root, "subject_gate_matrix")
    plt.close(fig)


def plot_curves(
    root: Path,
    selected_arrays: dict[tuple[str, int], dict[str, np.ndarray]],
) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for subject in SUBJECTS:
        fog_stack = np.stack([selected_arrays[(subject, seed)]["test_fog"] for seed in SEEDS])
        if fog_stack.shape[1] == 0:
            continue
        normal = np.median(
            np.stack([selected_arrays[(subject, seed)]["test_nonfog"] for seed in SEEDS]), axis=0
        )
        fog = np.median(fog_stack, axis=0)
        y = np.concatenate((np.zeros(len(normal), dtype=int), np.ones(len(fog), dtype=int)))
        score = np.concatenate((normal, fog))
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        ax_roc.plot(fpr, tpr, label=f"{subject} ({roc_auc_score(y, score):.3f})")
        ax_pr.plot(recall, precision, label=f"{subject} ({average_precision_score(y, score):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="A5 test ROC")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="A5 test precision-recall")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    ax_roc.grid(alpha=0.25)
    ax_pr.grid(alpha=0.25)
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    save_figure(fig_roc, root, "selected_score_test_roc")
    save_figure(fig_pr, root, "selected_score_test_pr")
    plt.close(fig_roc)
    plt.close(fig_pr)


def render_report(root: Path, gate: dict[str, Any]) -> None:
    subject_lines = [
        "| 被试 | 可用运行 | AUROC | PR-AUC/随机 | FoG÷Non-FoG | Recall@Q95 | FA/min | Cliff's δ | 可用门控 | 强门控 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in gate["subject_summaries"]:
        if row["evaluable_run_count"] == 0:
            subject_lines.append(
                f"| {row['subject_id']} | 0/3（无测试 FoG） | NE | NE | NE | NE | NE | NE | FAIL | FAIL |"
            )
            continue
        subject_lines.append(
            f"| {row['subject_id']} | {row['usable_run_count']}/3（可估计 {row['evaluable_run_count']}/3） | {row['median_auroc']:.3f} | "
            f"{row['median_average_precision']:.3f}/{row['median_random_pr_baseline']:.3f} | "
            f"{row['median_fog_to_nonfog_median_ratio']:.3f} | "
            f"{row['median_recall_at_train_nonfog_p95']:.1%} | "
            f"{row['median_false_alarm_windows_per_minute']:.2f} | "
            f"{row['median_cliffs_delta']:.3f} | "
            f"{'PASS' if row['usable_subject_pass'] else 'FAIL'} | "
            f"{'PASS' if row['strong_subject_pass'] else 'FAIL'} |"
        )
    candidate_lines = [
        "| 分数 | 验证 AUROC | 验证 PR-AUC | Cliff's δ | FoG÷Non-FoG | FA/min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gate["score_candidates"]:
        candidate_lines.append(
            f"| {row['score']} | {row['median_validation_auroc']:.3f} | "
            f"{row['median_validation_average_precision']:.3f} | "
            f"{row['median_validation_cliffs_delta']:.3f} | "
            f"{row['median_validation_fog_nonfog_ratio']:.3f} | "
            f"{row['median_validation_false_alarm_per_minute']:.2f} |"
        )
    weights = gate["s3_weights"]
    report = f"""# Daphnet NBM Route A：A5 FoG / Non-FoG 残差分离诊断

生成时间（UTC）：{gate['completed_utc']}

## 冻结方案

- NBM：`M3_tcdae_long + L4 + W0 + D0`。
- 残差校准：`C0_clipnone`。
- A4 残差表示：`R5 = [R, |R|, ΔR]`。
- A5 候选分数：S0 全局绝对残差、S1 Top-3 通道残差、S2 频谱残差、S3 验证集加权组合。
- S3 权重：S0={weights[0]:.1f}、S1={weights[1]:.1f}、S2={weights[2]:.1f}。
- 最终分数：`{gate['selected_score']}`，仅由验证集选择；测试 FoG 未参与选型。

## 门控结论

- A5 状态：**{gate['status']}**。
- 残差可用门控：{gate['usable_subject_count']}/7 被试通过（要求至少 5/7）。
- 强残差门控：{gate['strong_subject_count']}/7 被试通过（要求至少 4/7）。
- 是否允许进入 A6：**{'是' if gate['eligible_for_A6'] else '否'}**。
- 1 秒窗口步长用于 FA/min；这纠正了早期 A2–A4 脚本中 0.5 秒代理换算，不改动原始窗口或预测。

## 验证集分数选择

{chr(10).join(candidate_lines)}

## 冻结后测试结果

{chr(10).join(subject_lines)}

## 解释边界

A5 只判断冻结 NBM 残差能否区分 FoG 与 Non-FoG，不训练分类器，也不证明残差相对 Raw 信号具有增量价值。S02、S06 未通过 A1-Retest，因此其 A5 结果必须结合正常重构失败解释；只有 A6 的公平分类消融才能判断 Raw+Residual 是否优于 Raw-only。
"""
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "routeA_A5_residual_separation_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT
        / "outputs"
        / "daphnet_nbm_routeA_A1b_generalization_repair_v1"
        / "routeA_A1b_generalization_repair",
    )
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=ROOT / "outputs" / "daphnet_nbm_routeA_A2_A4_v1" / "routeA_A2_A4",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT / "routeA_A5",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--weight-step", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    parent = args.parent.resolve()
    prior_root = args.prior_root.resolve()
    a4_gate = json.loads(
        (prior_root / "A4_representation_ablation" / "A4_gate.json").read_text(encoding="utf-8")
    )
    if a4_gate.get("status") != "PASS" or a4_gate.get("selected_representation") != "R5":
        raise RuntimeError("A5 requires the frozen A4 R5 PASS result")
    a1_gate = json.loads((parent / "reports" / "A1_retest_gate.json").read_text(encoding="utf-8"))
    if not a1_gate.get("eligible_for_A2"):
        raise RuntimeError("A1-Retest does not authorize residual experiments")

    protocol = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": list(SUBJECTS),
        "selection_subjects": list(SELECTION_SUBJECTS),
        "diagnostic_only_during_selection": list(DIAGNOSTIC_ONLY_DURING_SELECTION),
        "seeds": list(SEEDS),
        "frozen_pipeline": "M3_tcdae_long+L4+W0+D0+C0_clipnone+R5",
        "candidate_scores": list(SCORES),
        "s3_weight_grid_step": args.weight_step,
        "score_and_weight_selection_split": "validation only",
        "test_fog_used_for_selection": False,
        "window_samples": WINDOW,
        "sampling_rate_hz": FS,
        "stride_seconds_for_false_alarm_rate": STRIDE_SECONDS,
        "pr_margin_above_random_for_strong_gate": PR_MARGIN_ABOVE_RANDOM,
        "gate": {
            "usable_subject": {"auroc": 0.65, "ratio_exclusive": 1.10, "cliffs_delta_exclusive": 0.20, "recall_q95": 0.20},
            "usable_overall_subject_count": 5,
            "strong_subject": {"auroc": 0.75, "pr_margin_over_random": PR_MARGIN_ABOVE_RANDOM, "false_alarm_per_minute": 0.5},
            "strong_overall_subject_count": 4,
        },
    }
    write_json(root / "protocol" / "frozen_A5_protocol.json", protocol)

    dataset = route_a.DaphnetDataset.load(args.data_dir.resolve())
    prepared = {subject: route_a.prepare_subject(dataset, subject) for subject in SUBJECTS}
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    print(f"A5 device={device} subjects={','.join(SUBJECTS)}", flush=True)

    all_components: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    scales: dict[tuple[str, int], np.ndarray] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            checkpoint = prior.a2_checkpoint(prior_root, parent, "D0", subject, seed)
            model = prior.load_model(checkpoint, device)
            item = prepared[subject]
            raw_sets = {
                "train_nonfog": item.train_x,
                "validation_nonfog": prior.prepared_windows(item, prior.indices(item, "validation", 0)),
                "validation_fog": prior.prepared_windows(item, prior.indices(item, "validation", 1)),
                "test_nonfog": item.test_x,
                "test_fog": prior.prepared_windows(item, prior.indices(item, "test", 1)),
            }
            sets: dict[str, prior.ResidualBundle] = {}
            for split, values in raw_sets.items():
                if len(values):
                    sets[split] = prior.bundle(model, values, device)
                else:
                    empty = np.asarray(values, dtype=np.float32)
                    sets[split] = prior.ResidualBundle(empty, empty.copy(), empty.copy())
            calibrated = prior.calibrated_sets(sets, "C0", None, args.sigma_min)
            components = {name: component_scores(bundle.residual) for name, bundle in calibrated.items()}
            all_components[(subject, seed)] = components
            scales[(subject, seed)] = fit_component_scale(components["train_nonfog"])
            run_dir = root / "scalar_scores" / subject / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                run_dir / "component_scores.npz",
                **components,
                train_component_scale=scales[(subject, seed)],
            )
            checkpoint_rows.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "train_nonfog_windows": len(components["train_nonfog"]),
                    "validation_nonfog_windows": len(components["validation_nonfog"]),
                    "validation_fog_windows": len(components["validation_fog"]),
                    "test_nonfog_windows": len(components["test_nonfog"]),
                    "test_fog_windows": len(components["test_fog"]),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"A5 inference {subject} seed={seed} done", flush=True)
    write_csv(root / "tables" / "checkpoint_and_window_audit.csv", checkpoint_rows)

    weight_rows: list[dict[str, Any]] = []
    for weights in simplex_weights(args.weight_step):
        rows: list[dict[str, Any]] = []
        for subject in SELECTION_SUBJECTS:
            for seed in SEEDS:
                components = all_components[(subject, seed)]
                arrays = score_arrays(components, scales[(subject, seed)], "S3", weights)
                rows.append(
                    separation_metrics(
                        arrays["validation_nonfog"], arrays["validation_fog"], arrays["train_nonfog"]
                    )
                )
        weight_rows.append(
            {
                "weight_s0": weights[0],
                "weight_s1": weights[1],
                "weight_s2": weights[2],
                "selection_runs": len(rows),
                "median_validation_auroc": finite_median(rows, "auroc"),
                "median_validation_average_precision": finite_median(rows, "average_precision"),
                "median_validation_cliffs_delta": finite_median(rows, "cliffs_delta"),
                "median_validation_false_alarm_per_minute": finite_median(rows, "false_alarm_windows_per_minute"),
            }
        )
    best_weight_row = max(
        weight_rows,
        key=lambda row: (
            row["median_validation_auroc"],
            row["median_validation_cliffs_delta"],
            row["median_validation_average_precision"],
            -row["median_validation_false_alarm_per_minute"],
            row["weight_s0"],
            row["weight_s1"],
        ),
    )
    weights = (
        float(best_weight_row["weight_s0"]),
        float(best_weight_row["weight_s1"]),
        float(best_weight_row["weight_s2"]),
    )
    for row in weight_rows:
        row["selected"] = bool(
            np.allclose([row["weight_s0"], row["weight_s1"], row["weight_s2"]], weights)
        )
    write_csv(root / "tables" / "S3_validation_weight_search.csv", weight_rows)
    write_json(root / "protocol" / "frozen_S3_weights.json", {"weights": weights, "selection_split": "validation only"})

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    array_cache: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for subject in SUBJECTS:
        for seed in SEEDS:
            for score_name in SCORES:
                arrays = score_arrays(all_components[(subject, seed)], scales[(subject, seed)], score_name, weights)
                array_cache[(subject, seed, score_name)] = arrays
                validation = separation_metrics(
                    arrays["validation_nonfog"], arrays["validation_fog"], arrays["train_nonfog"]
                )
                usable, strong = run_gate(validation)
                validation_rows.append(
                    {
                        "stage": "A5_residual_separation",
                        "selection_split": "validation",
                        "score": score_name,
                        "subject_id": subject,
                        "seed": seed,
                        "usable_run_pass": usable,
                        "strong_run_pass": strong,
                        **validation,
                    }
                )
                evaluable = bool(len(arrays["test_nonfog"]) and len(arrays["test_fog"]))
                if evaluable:
                    test = separation_metrics(arrays["test_nonfog"], arrays["test_fog"], arrays["train_nonfog"])
                    usable, strong = run_gate(test)
                else:
                    test = {
                        key: math.nan
                        for key in (
                            "nonfog_p50", "nonfog_p90", "nonfog_p95", "fog_p50", "fog_p90",
                            "fog_to_nonfog_median_ratio", "auroc", "average_precision",
                            "random_pr_baseline", "pr_margin_over_random", "recall_at_train_nonfog_p95",
                            "nonfog_false_alarm_fraction", "false_alarm_windows_per_minute", "cliffs_delta",
                            "hedges_g", "train_nonfog_p95_threshold",
                        )
                    }
                    usable, strong = False, False
                test_rows.append(
                    {
                        "stage": "A5_residual_separation",
                        "report_split": "test_after_freeze",
                        "score": score_name,
                        "subject_id": subject,
                        "seed": seed,
                        "evaluable": evaluable,
                        "not_evaluable_reason": "" if evaluable else "no test FoG windows in frozen split",
                        "test_nonfog_windows": len(arrays["test_nonfog"]),
                        "test_fog_windows": len(arrays["test_fog"]),
                        "usable_run_pass": usable,
                        "strong_run_pass": strong,
                        "validation_to_test_auroc_drop": validation["auroc"] - test["auroc"],
                        "validation_to_test_pr_drop": validation["average_precision"] - test["average_precision"],
                        **test,
                    }
                )
    write_csv(root / "tables" / "all_validation_score_metrics.csv", validation_rows)
    write_csv(root / "tables" / "all_test_score_metrics.csv", test_rows)

    candidates = [summarize_candidate(validation_rows, score_name) for score_name in SCORES]
    selected_candidate = max(
        candidates,
        key=lambda row: (
            row["median_validation_auroc"],
            row["median_validation_cliffs_delta"],
            row["median_validation_average_precision"],
            -row["median_validation_false_alarm_per_minute"],
            -SCORES.index(row["score"]),
        ),
    )
    selected_score = str(selected_candidate["score"])
    selected_test = [row for row in test_rows if row["score"] == selected_score]
    selected_validation = [row for row in validation_rows if row["score"] == selected_score]
    write_csv(root / "tables" / "selected_score_validation_metrics.csv", selected_validation)
    write_csv(root / "tables" / "selected_score_test_metrics.csv", selected_test)

    summaries = [subject_summary(selected_test, subject) for subject in SUBJECTS]
    usable_count = sum(bool(row["usable_subject_pass"]) for row in summaries)
    strong_count = sum(bool(row["strong_subject_pass"]) for row in summaries)
    status = "STRONG PASS" if strong_count >= 4 else "PASS" if usable_count >= 5 else "FAIL"
    gate = {
        "stage": "A5_residual_separation",
        "status": status,
        "selected_score": selected_score,
        "s3_weights": weights,
        "selection_split": "validation only",
        "selection_subjects": list(SELECTION_SUBJECTS),
        "test_subjects": list(SUBJECTS),
        "test_fog_used_for_selection": False,
        "score_candidates": candidates,
        "subject_summaries": summaries,
        "usable_subject_count": usable_count,
        "usable_subject_required": 5,
        "strong_subject_count": strong_count,
        "strong_subject_required": 4,
        "usable_gate_pass": bool(usable_count >= 5),
        "strong_gate_pass": bool(strong_count >= 4),
        "eligible_for_A6": bool(usable_count >= 5),
        "false_alarm_stride_seconds": STRIDE_SECONDS,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "A5_gate.json", gate)
    write_csv(root / "tables" / "subject_summary.csv", summaries)

    selected_arrays = {
        (subject, seed): array_cache[(subject, seed, selected_score)]
        for subject in SUBJECTS
        for seed in SEEDS
    }
    plot_candidate_comparison(root, candidates)
    plot_subject_metrics(root, summaries)
    plot_gate_matrix(root, summaries)
    plot_curves(root, selected_arrays)
    render_report(root, gate)
    write_json(
        root / "FINAL_RESULTS.json",
        {
            "experiment": EXPERIMENT,
            "A5": gate,
            "prior_A4_gate": str(prior_root / "A4_representation_ablation" / "A4_gate.json"),
            "completed_utc": gate["completed_utc"],
        },
    )
    print(
        f"A5 COMPLETE status={status} selected={selected_score} "
        f"usable={usable_count}/7 strong={strong_count}/7 results={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
