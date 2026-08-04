#!/usr/bin/env python
"""Summarize the eight window-axis-centered within-subject experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "outputs" / "nonfog_gru_nbm_tcnm_within_subject_v1_window_axis_centered_seed20260802"
S01 = ROOT / "outputs" / "nonfog_gru_nbm_tcnm_within_subject_v1_S01_window_axis_centered_seed20260802"
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=SUITE)
    parser.add_argument("--s01-dir", type=Path, default=None)
    return parser.parse_args()


def subject_dir(subject: str, suite: Path, s01_dir: Path) -> Path:
    return s01_dir if subject == "S01" else suite / subject


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit(subject: str, directory: Path) -> dict:
    done = read_json(directory / "DONE.json")
    config = read_json(directory / "config.json")
    metrics = read_json(directory / "metrics.json")
    if done["status"] != "complete":
        raise AssertionError(f"{subject} is incomplete")
    if not config["window_axis_centering"]["enabled"]:
        raise AssertionError(f"{subject} did not enable window-axis centering")
    with (directory / "artifacts" / "oof_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in manifest]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError(f"{subject} has duplicate OOF sample IDs")
    if any(row["nbm_seen_this_window"] != "False" for row in manifest):
        raise AssertionError(f"{subject} has NBM-seen OOF windows")
    with np.load(directory / "artifacts" / "residuals.npz") as arrays:
        centered_error = {
            split: float(np.max(np.abs(arrays[f"{split}_residual"].mean(axis=1))))
            for split in ("validation", "test")
        }
        centered_error["train_oof"] = float(
            np.max(np.abs(arrays["train_oof_residual"].mean(axis=1)))
        )
        if max(centered_error.values()) > 2e-5:
            raise AssertionError(f"{subject} residual centering failed: {centered_error}")
    if metrics["selected_threshold"] != metrics["validation"]["threshold"]:
        raise AssertionError(f"{subject} threshold provenance mismatch")
    return {
        "oof_windows": len(manifest),
        "maximum_center_mean_error": max(centered_error.values()),
        "warnings": done.get("warnings", []),
    }


def main() -> None:
    args = parse_args()
    suite = args.suite_dir.resolve()
    default_s01 = suite / "S01"
    s01_dir = (
        args.s01_dir.resolve()
        if args.s01_dir is not None
        else (default_s01 if default_s01.is_dir() else S01)
    )
    suite.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    audits: dict[str, dict] = {}
    for subject in SUBJECTS:
        directory = subject_dir(subject, suite, s01_dir)
        audits[subject] = audit(subject, directory)
        metrics = read_json(directory / "metrics.json")
        training = read_json(directory / "training.json")
        classifier_training = training.get("classifier", training.get("tcn_m"))
        if classifier_training is None:
            raise KeyError(f"{subject} training.json lacks classifier training")
        test = metrics["test"]
        event = metrics["test_event_detection"]
        rows.append(
            {
                "subject": subject,
                "test_windows": test["n"],
                "test_non_fog_windows": test["n_normal"],
                "test_fog_windows": test["n_fog"],
                "threshold": metrics["selected_threshold"],
                "accuracy": test["accuracy"],
                "balanced_accuracy": test["balanced_accuracy"],
                "fog_precision": test["precision"],
                "fog_recall": test["sensitivity"],
                "specificity": test["specificity"],
                "fog_f1": test["f1"],
                "pr_auc": test["auprc"],
                "roc_auc": test["auroc"],
                "tn": test["tn"],
                "fp": test["fp"],
                "fn": test["fn"],
                "tp": test["tp"],
                "fog_events": event["events"],
                "detected_events": event["detected"],
                "event_recall": event["event_recall"],
                "final_nbm_best_epoch": training["final_nbm"]["best_epoch"],
                "classifier": classifier_training.get("model_name", "tcn_m"),
                "classifier_best_epoch": classifier_training["best_epoch"],
                "validation_pr_auc": classifier_training["best_validation_pr_auc"],
            }
        )
    csv_path = suite / "aggregate_main_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "fog_precision",
        "fog_recall",
        "specificity",
        "fog_f1",
        "pr_auc",
        "roc_auc",
        "event_recall",
    )
    macro = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        macro[name] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    tn = sum(row["tn"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tp = sum(row["tp"] for row in rows)
    micro = {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "fog_recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "fog_precision": tp / (tp + fp),
        "fog_f1": 2 * tp / (2 * tp + fp + fn),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "detected_events": sum(row["detected_events"] for row in rows),
        "fog_events": sum(row["fog_events"] for row in rows),
    }
    micro["event_recall"] = micro["detected_events"] / micro["fog_events"]
    limitations = [
        {
            "subject": row["subject"],
            "reason": f"only {row['test_fog_windows']} FoG test window(s)",
        }
        for row in rows
        if row["test_fog_windows"] < 10
    ]
    payload = {
        "experiment": "nonfog_gru_nbm_tcnm_within_subject_v1_window_axis_centered",
        "subjects": list(SUBJECTS),
        "n_subjects": len(SUBJECTS),
        "per_subject": rows,
        "macro_subject_statistics": macro,
        "micro_threshold_dependent_statistics": micro,
        "audit": audits,
        "small_positive_test_sets": limitations,
        "pr_auc_aggregation_note": "PR-AUC is summarized only as the unweighted subject macro mean; probabilities from separately trained within-subject models are not pooled.",
    }
    with (suite / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    labels = [row["subject"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - width, [row["fog_recall"] for row in rows], width, label="FoG Recall")
    ax.bar(x, [row["specificity"] for row in rows], width, label="Specificity")
    ax.bar(x + width, [row["pr_auc"] for row in rows], width, label="PR-AUC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Window-axis-centered within-subject results")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(suite / "aggregate_main_metrics.png", dpi=180)
    plt.close(fig)

    lines = [
        "# 逐窗口逐轴中心化单被试实验汇总",
        "",
        "|被试|测试窗口(FoG)|阈值|Accuracy|Balanced Acc|FoG Recall|Specificity|PR-AUC|F1|混淆矩阵|事件检出|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['subject']}|{row['test_windows']} ({row['test_fog_windows']})|"
            f"{row['threshold']:.2f}|{row['accuracy']:.4f}|{row['balanced_accuracy']:.4f}|"
            f"{row['fog_recall']:.4f}|{row['specificity']:.4f}|{row['pr_auc']:.4f}|"
            f"{row['fog_f1']:.4f}|[[{row['tn']},{row['fp']}],[{row['fn']},{row['tp']}]]|"
            f"{row['detected_events']}/{row['fog_events']}|"
        )
    lines.extend(
        [
            "",
            "## 宏平均",
            "",
            *[
                f"- {name}: {macro[name]['mean']:.6f} ± {macro[name]['std_population']:.6f}"
                for name in metric_names
            ],
            "",
            "## 合并混淆计数",
            "",
            f"- Confusion matrix: {micro['confusion_matrix']}",
            f"- Accuracy: {micro['accuracy']:.6f}",
            f"- FoG Recall: {micro['fog_recall']:.6f}",
            f"- Specificity: {micro['specificity']:.6f}",
            f"- Event detection: {micro['detected_events']}/{micro['fog_events']}",
            "",
            "PR-AUC不做概率池化，只报告各被试等权宏平均，因为每名被试使用独立训练模型和独立验证阈值。",
        ]
    )
    (suite / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"macro": macro, "micro": micro, "limitations": limitations}, ensure_ascii=False))


if __name__ == "__main__":
    main()
