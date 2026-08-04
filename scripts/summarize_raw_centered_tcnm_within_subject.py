#!/usr/bin/env python
"""Audit and summarize the eight raw-centered TCN-M ablation runs."""

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
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
DEFAULT_SUITE = (
    ROOT / "outputs" / "raw_window_axis_centered_tcnm_within_subject_v1_seed20260802"
)
DEFAULT_BASELINE = (
    ROOT
    / "outputs"
    / "nonfog_gru_nbm_tcnm_within_subject_v1_window_axis_centered_finalnbm100_pat10_seed20260802"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit(subject: str, directory: Path) -> dict:
    done = read_json(directory / "DONE.json")
    config = read_json(directory / "config.json")
    metrics = read_json(directory / "metrics.json")
    scalers = read_json(directory / "scalers.json")
    if done["status"] != "complete":
        raise AssertionError(f"{subject}: incomplete run")
    if not all(
        config.get(key) is True
        for key in ("nbm_removed", "residual_removed", "residual_clipping_removed")
    ):
        raise AssertionError(f"{subject}: ablation flags are incomplete")
    if metrics["selected_threshold"] != metrics["validation"]["threshold"]:
        raise AssertionError(f"{subject}: threshold was not frozen from validation")
    manifest_path = directory / "artifacts" / "oof_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{subject}: duplicate OOF windows")
    if any(row["scaler_seen_this_window"] != "False" for row in manifest):
        raise AssertionError(f"{subject}: OOF scaler leakage")
    errors = scalers["maximum_absolute_window_axis_mean"]
    if max(errors.values()) > 1e-4:
        raise AssertionError(f"{subject}: centering audit failed: {errors}")
    with np.load(directory / "artifacts" / "features.npz") as arrays:
        if len(arrays["train_y"]) != len(manifest):
            raise AssertionError(f"{subject}: manifest/feature count mismatch")
        if not np.array_equal(arrays["test_y"], np.asarray(
            [0] * metrics["test"]["n_normal"] + [1] * metrics["test"]["n_fog"]
        )):
            # Labels need not be class-sorted, so verify counts if the strict
            # ordering comparison fails.
            labels = arrays["test_y"]
            if int((labels == 0).sum()) != metrics["test"]["n_normal"] or int(
                (labels == 1).sum()
            ) != metrics["test"]["n_fog"]:
                raise AssertionError(f"{subject}: test label counts mismatch")
    return {
        "oof_windows": len(manifest),
        "maximum_center_mean_error": max(errors.values()),
        "warnings": done.get("warnings", []),
    }


def row_from_run(subject: str, directory: Path) -> dict:
    metrics = read_json(directory / "metrics.json")
    training = read_json(directory / "training.json")
    test = metrics["test"]
    event = metrics["test_event_detection"]
    return {
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
        "tcn_best_epoch": training["best_epoch"],
        "validation_pr_auc": training["best_validation_pr_auc"],
    }


def main() -> None:
    args = parse_args()
    suite = args.suite_dir.resolve()
    baseline = args.baseline_dir.resolve()
    rows = []
    audits = {}
    for subject in SUBJECTS:
        directory = suite / subject
        audits[subject] = audit(subject, directory)
        rows.append(row_from_run(subject, directory))

    with (suite / "aggregate_main_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
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
    macro = {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "std_population": float(np.std([row[name] for row in rows], ddof=0)),
        }
        for name in metric_names
    }
    tn, fp, fn, tp = (
        sum(row[name] for row in rows) for name in ("tn", "fp", "fn", "tp")
    )
    micro = {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "fog_precision": tp / (tp + fp),
        "fog_recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "fog_f1": 2 * tp / (2 * tp + fp + fn),
    }

    comparison = []
    if baseline.is_dir():
        for row in rows:
            base_test = read_json(baseline / row["subject"] / "metrics.json")["test"]
            comparison.append(
                {
                    "subject": row["subject"],
                    "delta_accuracy": row["accuracy"] - base_test["accuracy"],
                    "delta_balanced_accuracy": row["balanced_accuracy"]
                    - base_test["balanced_accuracy"],
                    "delta_fog_recall": row["fog_recall"] - base_test["sensitivity"],
                    "delta_specificity": row["specificity"] - base_test["specificity"],
                    "delta_pr_auc": row["pr_auc"] - base_test["auprc"],
                    "delta_fog_f1": row["fog_f1"] - base_test["f1"],
                }
            )
        with (suite / "comparison_vs_final_nbm100.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)

    payload = {
        "experiment": "raw_window_axis_centered_tcnm_within_subject_v1",
        "subjects": list(SUBJECTS),
        "per_subject": rows,
        "macro_subject_statistics": macro,
        "micro_threshold_dependent_statistics": micro,
        "audit": audits,
        "comparison_vs_final_nbm100": comparison,
        "pr_auc_aggregation_note": (
            "PR-AUC is reported as the unweighted subject macro mean; independently "
            "trained within-subject probabilities are not pooled."
        ),
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
    ax.set_title("Raw-centered TCN-M within-subject ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(suite / "aggregate_main_metrics.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"macro": macro, "micro": micro}, ensure_ascii=False))


if __name__ == "__main__":
    main()
