#!/usr/bin/env python
"""Summarize seed-42 capacity-matched Raw-TCN within-subject runs."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
METRICS = ("accuracy", "fog_recall", "specificity", "pr_auc", "f1", "balanced_accuracy")


def full_dir(subject: str, window_axis_centering: bool) -> Path:
    suffix = "_window_axis_centered" if window_axis_centering else ""
    name = (
        f"daphnet_{subject.lower()}_spectral_gru_nbm_blocked5"
        f"{suffix}_tcnm_seed42"
    )
    return REPO_ROOT / "outputs" / name


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--window-axis-centering",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    centering_suffix = "_window_axis_centered" if args.window_axis_centering else ""
    output_dir = (
        REPO_ROOT
        / "outputs"
        / f"daphnet_raw_tcnm_capacitymatched{centering_suffix}_seed42_summary"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Summary directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    comparison_rows: list[dict] = []
    for subject in SUBJECTS:
        raw_dir = (
            REPO_ROOT
            / "outputs"
            / f"daphnet_{subject.lower()}_raw_tcnm_capacitymatched{centering_suffix}_seed42"
        )
        if not (raw_dir / "DONE.json").exists():
            raise FileNotFoundError(raw_dir)
        raw_config = json.loads(
            (raw_dir / "config.json").read_text(encoding="utf-8")
        )
        actual_centering = bool(
            raw_config["windowing"]["per_window_per_axis_temporal_centering"][
                "enabled"
            ]
        )
        if actual_centering != bool(args.window_axis_centering):
            raise ValueError(f"Centering configuration mismatch: {raw_dir}")
        raw = json.loads((raw_dir / "metrics.json").read_text(encoding="utf-8"))["test"]
        full = json.loads(
            (full_dir(subject, args.window_axis_centering) / "metrics.json").read_text(encoding="utf-8")
        )["test"]
        row = {
            "subject": subject,
            "test_windows": raw["n"],
            "test_non_fog_windows": raw["n_normal"],
            "test_fog_windows": raw["n_fog"],
            "threshold": raw["threshold"],
            **{key: raw[key] for key in METRICS},
            "tn": raw["tn"],
            "fp": raw["fp"],
            "fn": raw["fn"],
            "tp": raw["tp"],
            "result_dir": str(raw_dir.resolve()),
        }
        rows.append(row)
        comparison_rows.append(
            {
                "subject": subject,
                **{f"raw_{key}": raw[key] for key in METRICS},
                **{f"full_nbm_{key}": full[key] for key in METRICS},
                **{f"raw_minus_full_{key}": raw[key] - full[key] for key in METRICS},
            }
        )

    macro = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std_population": float(np.std([row[key] for row in rows], ddof=0)),
        }
        for key in METRICS
    }
    macro_comparison = {
        key: {
            "raw_mean": float(np.mean([row[f"raw_{key}"] for row in comparison_rows])),
            "full_nbm_mean": float(
                np.mean([row[f"full_nbm_{key}"] for row in comparison_rows])
            ),
            "raw_minus_full": float(
                np.mean([row[f"raw_minus_full_{key}"] for row in comparison_rows])
            ),
        }
        for key in METRICS
    }
    tn, fp, fn, tp = (
        sum(row[key] for row in rows) for key in ("tn", "fp", "fn", "tp")
    )
    pooled = {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "fog_recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }
    payload = {
        "experiment": "capacity-matched Raw-TCN, no FFT/NBM/residual/crossfit",
        "per_window_per_axis_temporal_centering": bool(
            args.window_axis_centering
        ),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "seed": 42,
        "rows": rows,
        "macro_unweighted": macro,
        "count_pooled": pooled,
        "paired_descriptive_comparison_to_existing_full_nbm": comparison_rows,
        "macro_comparison": macro_comparison,
        "cautions": [
            "One seed only; paired deltas are descriptive, not inferential.",
            "S06 test contains one FoG window and S09 contains four.",
            "Threshold transfer and false-positive counts vary substantially by subject.",
            "These are within-subject chronological tests, not unseen-subject generalization.",
        ],
    }
    (output_dir / "main_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, data in (
        ("main_metrics.csv", rows),
        ("comparison_to_full_nbm.csv", comparison_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    table = "\n".join(
        f"| {row['subject']} | {row['test_windows']} | {row['test_fog_windows']} "
        f"| {row['threshold']:.2f} | {row['accuracy']:.4f} | {row['fog_recall']:.4f} "
        f"| {row['specificity']:.4f} | {row['pr_auc']:.4f} | {row['f1']:.4f} "
        f"| {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        for row in rows
    )
    comparison_table = "\n".join(
        f"| {key} | {values['raw_mean']:.4f} | {values['full_nbm_mean']:.4f} "
        f"| {values['raw_minus_full']:+.4f} |"
        for key, values in macro_comparison.items()
    )
    centering_description = (
        "After train-fitted Robust scaling, every 2 s window and axis is centered independently over its 128 time samples."
        if args.window_axis_centering
        else "Per-window per-axis temporal centering is disabled."
    )
    summary = f"""# Capacity-matched Raw-TCN, seed 42

Raw input is Robust-scaled `[B,9,128]`. FFT, log-power spectrum, GRU-NBM, residual generation, augmentation, and blocked crossfit are absent. TCN-M remains 57,121 parameters with four `[1,2,4,8]` dilation blocks.

{centering_description}

| Subject | Test N | FoG N | Threshold | Accuracy | FoG Recall | Specificity | PR-AUC | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Raw-TCN aggregate

| Metric | Macro mean | Population SD |
|---|---:|---:|
| Accuracy | {macro['accuracy']['mean']:.4f} | {macro['accuracy']['std_population']:.4f} |
| FoG Recall | {macro['fog_recall']['mean']:.4f} | {macro['fog_recall']['std_population']:.4f} |
| Specificity | {macro['specificity']['mean']:.4f} | {macro['specificity']['std_population']:.4f} |
| PR-AUC | {macro['pr_auc']['mean']:.4f} | {macro['pr_auc']['std_population']:.4f} |
| F1 | {macro['f1']['mean']:.4f} | {macro['f1']['std_population']:.4f} |
| Balanced accuracy | {macro['balanced_accuracy']['mean']:.4f} | {macro['balanced_accuracy']['std_population']:.4f} |

Count-pooled confusion matrix: `[[{tn}, {fp}], [{fn}, {tp}]]`; Accuracy={pooled['accuracy']:.4f}, FoG Recall={pooled['fog_recall']:.4f}, Specificity={pooled['specificity']:.4f}. Subject-specific thresholds make this descriptive only.

## Descriptive comparison with existing full NBM runs

| Metric | Raw macro | Full NBM macro | Raw - Full |
|---|---:|---:|---:|
{comparison_table}

Only one seed was run, so the differences are descriptive. Threshold transfer and false-positive counts vary substantially by subject. S06 has one test FoG window and S09 has four.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
