#!/usr/bin/env python
"""Summarize completed within-subject spectral blocked-crossfit runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--window-axis-centering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Summarize the per-window per-axis centered experiment directories.",
    )
    parser.add_argument(
        "--spectral-robust-standardization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Summarize runs using the shared training-only spectral Robust scaler.",
    )
    return parser.parse_args()


def subject_dir(
    subject: str,
    window_axis_centering: bool,
    spectral_robust_standardization: bool,
) -> Path:
    suffix = "_window_axis_centered" if window_axis_centering else ""
    spectral_suffix = (
        "_spectral_robust" if spectral_robust_standardization else ""
    )
    name = (
        f"daphnet_{subject.lower()}_spectral_gru_nbm_blocked5"
        f"{suffix}{spectral_suffix}_tcnm_seed42"
    )
    return REPO_ROOT / "outputs" / name


def main() -> None:
    args = parse_args()
    suffix = "_window_axis_centered" if args.window_axis_centering else ""
    spectral_suffix = (
        "_spectral_robust" if args.spectral_robust_standardization else ""
    )
    output_dir = (
        REPO_ROOT
        / "outputs"
        / f"daphnet_spectral_gru_nbm_blocked5{suffix}{spectral_suffix}_within_subject_seed42_summary"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Summary directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for subject in SUBJECTS:
        directory = subject_dir(
            subject,
            args.window_axis_centering,
            args.spectral_robust_standardization,
        )
        if not (directory / "DONE.json").exists():
            raise FileNotFoundError(f"Incomplete subject result: {directory}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        actual_spectral_standardization = bool(
            config["spectrum"]["robust_standardization"]["enabled"]
        )
        if actual_spectral_standardization != bool(
            args.spectral_robust_standardization
        ):
            raise ValueError(f"Spectral standardization mismatch: {directory}")
        training = json.loads(
            (directory / "tcnm_training.json").read_text(encoding="utf-8")
        )
        test = metrics["test"]
        rows.append(
            {
                "subject": subject,
                "test_windows": test["n"],
                "test_non_fog_windows": test["n_normal"],
                "test_fog_windows": test["n_fog"],
                "test_fog_percent": 100.0 * test["n_fog"] / test["n"],
                "threshold": test["threshold"],
                "accuracy": test["accuracy"],
                "fog_recall": test["fog_recall"],
                "specificity": test["specificity"],
                "pr_auc": test["pr_auc"],
                "f1": test["f1"],
                "balanced_accuracy": test["balanced_accuracy"],
                "tn": test["tn"],
                "fp": test["fp"],
                "fn": test["fn"],
                "tp": test["tp"],
                "tcn_best_epoch": training["best_epoch"],
                "result_dir": str(directory.resolve()),
            }
        )

    metric_keys = (
        "accuracy",
        "fog_recall",
        "specificity",
        "pr_auc",
        "f1",
        "balanced_accuracy",
    )
    macro = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std_population": float(np.std([row[key] for row in rows], ddof=0)),
        }
        for key in metric_keys
    }
    tn = sum(row["tn"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tp = sum(row["tp"] for row in rows)
    pooled = {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "fog_recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "note": "Count-pooled threshold metrics; each subject used its own validation-selected threshold.",
    }
    payload = {
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "seed": 42,
        "per_window_per_axis_temporal_centering": bool(
            args.window_axis_centering
        ),
        "per_axis_per_frequency_spectral_robust_standardization": bool(
            args.spectral_robust_standardization
        ),
        "rows": rows,
        "macro_unweighted": macro,
        "count_pooled": pooled,
        "cautions": [
            "This is within-subject evaluation, not LOSO or cross-subject generalization.",
            "Each subject uses a predefined chronological split and its own validation-selected threshold.",
            "PR-AUC prevalence baselines differ strongly between subjects.",
            "S06 test has one FoG window and S09 has four, so their positive-class metrics are unstable.",
        ],
    }
    (output_dir / "main_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "main_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table_rows = "\n".join(
        f"| {row['subject']} | {row['test_windows']} | {row['test_fog_windows']} "
        f"| {row['threshold']:.2f} | {row['accuracy']:.4f} "
        f"| {row['fog_recall']:.4f} | {row['specificity']:.4f} "
        f"| {row['pr_auc']:.4f} | {row['f1']:.4f} "
        f"| {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        for row in rows
    )
    centering_text = (
        "Robust-scaled windows are centered independently per window and axis "
        "before corruption/FFT."
        if args.window_axis_centering
        else "Per-window per-axis temporal centering is disabled."
    )
    spectral_text = (
        "One per-axis/per-frequency Robust scaler is fitted on outer-training clean Non-FoG log-power spectra and shared by every fold and final NBM; residual-error mean/std is not fitted."
        if args.spectral_robust_standardization
        else "Training-spectrum Robust standardization is disabled."
    )
    text = f"""# Daphnet within-subject blocked-crossfit summary

All subjects use the S01 spectral protocol: 2 s/1 s stride, [9,65] log-power spectrum, 64-dimensional denoising GRU-NBM, raw signed spectral residual, five blocked OOF folds with 0.5 s purge, and one-window frequency-axis TCN-M. S04 and S10 are excluded.

{centering_text}

{spectral_text}

| Subject | Test N | Test FoG | Threshold | Accuracy | FoG Recall | Specificity | PR-AUC | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

## Aggregate

| Metric | Macro mean | Macro population SD |
|---|---:|---:|
| Accuracy | {macro['accuracy']['mean']:.4f} | {macro['accuracy']['std_population']:.4f} |
| FoG Recall | {macro['fog_recall']['mean']:.4f} | {macro['fog_recall']['std_population']:.4f} |
| Specificity | {macro['specificity']['mean']:.4f} | {macro['specificity']['std_population']:.4f} |
| PR-AUC | {macro['pr_auc']['mean']:.4f} | {macro['pr_auc']['std_population']:.4f} |
| F1 | {macro['f1']['mean']:.4f} | {macro['f1']['std_population']:.4f} |
| Balanced accuracy | {macro['balanced_accuracy']['mean']:.4f} | {macro['balanced_accuracy']['std_population']:.4f} |

Count-pooled confusion matrix: `[[{tn}, {fp}], [{fn}, {tp}]]`; pooled Accuracy={pooled['accuracy']:.4f}, FoG Recall={pooled['fog_recall']:.4f}, Specificity={pooled['specificity']:.4f}. This pooling combines subject-specific validation thresholds and is descriptive only.

S06 has only one test FoG window and S09 has four. These two subjects' Recall and PR-AUC should not be interpreted as stable patient-level estimates. Results describe within-subject temporal generalization, not unseen-subject generalization.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
