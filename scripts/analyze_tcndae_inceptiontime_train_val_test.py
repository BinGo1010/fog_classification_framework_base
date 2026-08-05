from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_full_subject_nbm_residual_binary as exp
import run_daphnet_full_subject_nbm_residual_inceptiontime as inception
import run_daphnet_full_subject_tcndae_inceptiontime as tcndae


DEFAULT_ROOT = (
    ROOT / "outputs" / "daphnet_full_subject_tcndae_inceptiontime_server_v1"
    / "full_subject_binary_experiment"
)
DEFAULT_METHODS = ("B1", "B2", "B3")
SPLITS = ("train", "validation", "test")
METRICS = ("pr_auc", "roc_auc", "fog_f1", "recall", "precision",
           "specificity", "balanced_accuracy", "mcc")
DISPLAY_METRICS = ("pr_auc", "fog_f1", "balanced_accuracy")
DISPLAY_NAMES = {"pr_auc": "PR-AUC", "fog_f1": "FoG F1",
                 "balanced_accuracy": "Balanced accuracy"}
METHOD_SHORT = {"B1": "B1 Residual", "B2": "B2 R5", "B3": "B3 Raw+R5"}
SPLIT_COLORS = {"train": "#4477AA", "validation": "#EE6677", "test": "#228833"}


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    return (frame["y_true"].to_numpy(dtype=int),
            frame["y_prob"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=int))


def metric_row(y_true: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = exp.binary_metrics(y_true, probability, prediction)
    return {metric: result[metric] for metric in METRICS} | {
        "n_windows": int(len(y_true)),
        "positive_windows": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)) if len(y_true) else math.nan,
    }


def train_probabilities(run_dir: Path, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model = inception.InceptionTimeClassifier(inputs.shape[2]).to(device)
    payload = torch.load(run_dir / "inceptiontime_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    probability = exp.predict_classifier(model, inputs, device)
    del model
    return probability


def collect_run_metrics(root: Path, methods: tuple[str, ...], device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fold_summary = pd.read_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv")
    expected = len(fold_summary) * len(methods) * len(exp.SEEDS)
    completed = 0
    for fold_info in fold_summary.itertuples(index=False):
        subject = str(fold_info.subject_id)
        fold_id = str(fold_info.fold_id)
        cache = root / "splits" / "outer_folds" / subject / fold_id / "representations.npz"
        arrays = dict(np.load(cache, allow_pickle=False))
        representations = exp.representation_arrays(
            arrays["train_x"], arrays["train_reconstruction_oof"]
        )
        inner = arrays["inner_fold"]
        validation_fold = int(arrays["validation_inner_fold"][0])
        train_mask = (inner >= 0) & (inner != validation_fold)
        train_y = arrays["train_y"].astype(int)[train_mask]
        for method in methods:
            train_x = representations[method][train_mask]
            for seed in exp.SEEDS:
                run_dir = root / tcndae.METHOD_DIRS[method] / subject / fold_id / f"seed{seed}"
                run_result = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
                threshold = float(run_result["threshold"])
                train_probability = train_probabilities(run_dir, train_x, device)
                train_prediction = (train_probability >= threshold).astype(int)
                common = {"subject_id": subject, "fold_id": fold_id, "method": method,
                          "method_name": tcndae.METHOD_NAMES[method], "seed": int(seed),
                          "threshold": threshold}
                rows.append(common | {"split": "train"}
                            | metric_row(train_y, train_probability, train_prediction))
                for split, filename in (("validation", "validation_predictions.csv"),
                                        ("test", "test_predictions.csv")):
                    y_true, probability, prediction = read_predictions(run_dir / filename)
                    rows.append(common | {"split": split}
                                | metric_row(y_true, probability, prediction))
                completed += 1
                if completed == 1 or completed % 15 == 0 or completed == expected:
                    print(f"ANALYSIS {completed}/{expected} {subject}/{fold_id} {method} seed={seed}",
                          flush=True)
        del arrays, representations
    return pd.DataFrame(rows)


def aggregate_metrics(run_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = list(METRICS) + ["n_windows", "positive_windows", "prevalence", "threshold"]
    seed_level = (
        run_frame.groupby(["subject_id", "method", "method_name", "seed", "split"], as_index=False)[numeric]
        .median(numeric_only=True)
    )
    subject_level = (
        seed_level.groupby(["subject_id", "method", "method_name", "split"], as_index=False)[numeric]
        .median(numeric_only=True)
    )
    macro = (
        subject_level.groupby(["method", "method_name", "split"], as_index=False)[list(METRICS)]
        .mean(numeric_only=True)
    )
    return seed_level, subject_level, macro


def wide_subject_table(subject_level: pd.DataFrame) -> pd.DataFrame:
    frame = subject_level.pivot(index=["subject_id", "method", "method_name"],
                                columns="split", values=list(METRICS))
    frame.columns = [f"{metric}_{split}" for metric, split in frame.columns]
    frame = frame.reset_index()
    for metric in METRICS:
        frame[f"{metric}_train_minus_test"] = frame[f"{metric}_train"] - frame[f"{metric}_test"]
        frame[f"{metric}_validation_minus_test"] = (
            frame[f"{metric}_validation"] - frame[f"{metric}_test"]
        )
    return frame


def heatmap_figure(subject_level: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
    })
    subjects = list(exp.SUBJECTS)
    methods = [method for method in DEFAULT_METHODS if method in set(subject_level["method"])]
    fig, axes = plt.subplots(len(DISPLAY_METRICS), len(methods), figsize=(7.2, 7.1),
                             constrained_layout=True)
    image = None
    for row_index, metric in enumerate(DISPLAY_METRICS):
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            selected = subject_level[(subject_level["method"] == method)]
            matrix = (selected.pivot(index="subject_id", columns="split", values=metric)
                      .reindex(index=subjects, columns=SPLITS).to_numpy(dtype=float))
            image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    value = matrix[y, x]
                    color = "white" if np.isfinite(value) and value >= 0.58 else "#222222"
                    axis.text(x, y, "NA" if not np.isfinite(value) else f"{value:.2f}",
                              ha="center", va="center", fontsize=6.1, color=color)
            axis.set_xticks(range(3), ["Train", "Validation", "Test"], rotation=25, ha="right")
            axis.set_yticks(range(len(subjects)))
            axis.set_yticklabels(subjects if column_index == 0 else [])
            axis.tick_params(length=0)
            if row_index == 0:
                axis.set_title(METHOD_SHORT[method], fontsize=8, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(DISPLAY_NAMES[metric], fontsize=8, fontweight="bold")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("Score", fontsize=7)
    fig.suptitle("Subject-level train-validation-test performance\n"
                 "Median across outer folds, then median across three seeds",
                 fontsize=10, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def gap_figure(wide: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
    })
    subjects = list(exp.SUBJECTS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True, constrained_layout=True)
    x = np.arange(len(subjects))
    for axis, method in zip(axes, DEFAULT_METHODS):
        selected = wide[wide["method"] == method].set_index("subject_id").reindex(subjects)
        for split in SPLITS:
            axis.plot(x, selected[f"pr_auc_{split}"], marker="o", markersize=3.4,
                      linewidth=1.15, color=SPLIT_COLORS[split], label=split.title())
        axis.set_title(METHOD_SHORT[method], fontsize=8, fontweight="bold")
        axis.set_xticks(x, subjects, rotation=45, ha="right")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("PR-AUC")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Generalization profile by subject", fontsize=10, fontweight="bold", y=1.13)
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def markdown_report(subject_level: pd.DataFrame, macro: pd.DataFrame, wide: pd.DataFrame,
                    output: Path) -> None:
    lines = [
        "# TCN-DAE残差方法：训练/验证/测试诊断",
        "",
        "## 统计口径",
        "",
        "- 方法：B1残差、B2 R5残差增强、B3原始信号与R5融合。",
        "- 每个外层折先计算指标；每名被试先取外层折中位数，再取3个分类种子中位数。",
        "- 因此这里的测试列表示典型外层折，不等同于主报告将全部互斥外层测试预测汇总后的被试级指标。",
        "- 训练指标由最佳检查点重新推理得到，并使用该运行在验证集选择的阈值。",
        "- 验证集同时参与early stopping和阈值选择，因此验证F1偏乐观；外层测试指标才是泛化证据。",
        "- 被purge的inner_fold<0窗口不属于分类器训练集，未纳入训练指标。",
        "",
        "## 全被试宏平均",
        "",
        "| 方法 | 集合 | PR-AUC | ROC-AUC | FoG F1 | Recall | Specificity | BAcc | MCC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in DEFAULT_METHODS:
        for split in SPLITS:
            row = macro[(macro["method"] == method) & (macro["split"] == split)].iloc[0]
            lines.append(
                f"| {method} | {split} | {row.pr_auc:.4f} | {row.roc_auc:.4f} | "
                f"{row.fog_f1:.4f} | {row.recall:.4f} | {row.specificity:.4f} | "
                f"{row.balanced_accuracy:.4f} | {row.mcc:.4f} |"
            )
    lines += ["", "## 各被试PR-AUC", "",
              "| 方法 | 被试 | Train | Validation | Test | Train-Test | Validation-Test |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for method in DEFAULT_METHODS:
        selected = wide[wide["method"] == method].set_index("subject_id")
        for subject in exp.SUBJECTS:
            row = selected.loc[subject]
            lines.append(
                f"| {method} | {subject} | {row.pr_auc_train:.4f} | "
                f"{row.pr_auc_validation:.4f} | {row.pr_auc_test:.4f} | "
                f"{row.pr_auc_train_minus_test:+.4f} | "
                f"{row.pr_auc_validation_minus_test:+.4f} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = set(methods) - set(tcndae.METHOD_DIRS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = root / "analysis_train_validation_test"
    output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        subject_level = pd.read_csv(output / "subject_split_metrics.csv")
        macro = pd.read_csv(output / "macro_train_validation_test.csv")
        wide = pd.read_csv(output / "subject_train_validation_test_wide.csv")
        heatmap_figure(subject_level, output / "subject_split_performance_heatmaps")
        gap_figure(wide, output / "subject_pr_auc_generalization_profiles")
        markdown_report(subject_level, macro, wide, output / "train_validation_test_report.md")
        print(f"RENDER COMPLETE {output}", flush=True)
        return
    run_frame = collect_run_metrics(root, methods, device)
    seed_level, subject_level, macro = aggregate_metrics(run_frame)
    wide = wide_subject_table(subject_level)
    run_frame.to_csv(output / "run_fold_split_metrics.csv", index=False, encoding="utf-8-sig")
    seed_level.to_csv(output / "subject_seed_split_metrics.csv", index=False, encoding="utf-8-sig")
    subject_level.to_csv(output / "subject_split_metrics.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(output / "subject_train_validation_test_wide.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(output / "macro_train_validation_test.csv", index=False, encoding="utf-8-sig")
    heatmap_figure(subject_level, output / "subject_split_performance_heatmaps")
    gap_figure(wide, output / "subject_pr_auc_generalization_profiles")
    markdown_report(subject_level, macro, wide, output / "train_validation_test_report.md")
    manifest = {
        "methods": list(methods), "subjects": list(exp.SUBJECTS), "splits": list(SPLITS),
        "run_rows": len(run_frame), "expected_run_rows": 30 * len(methods) * 3 * 3,
        "aggregation": "metric per outer-fold run; median folds within subject-seed; median seeds",
        "training_predictions": "recomputed from inceptiontime_best.pt",
        "validation_predictions": "saved validation_predictions.csv",
        "test_predictions": "saved test_predictions.csv",
        "validation_is_model_selection_data": True,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
