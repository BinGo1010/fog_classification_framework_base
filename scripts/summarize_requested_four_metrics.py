#!/usr/bin/env python
"""Report ACC, Macro-Precision, Macro-Recall and Macro-F1 from saved runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
DEFAULT_SUITE = (
    ROOT
    / "outputs"
    / "nonfog_gru_nbm_inceptiontime_within_subject_distributed_calibration_bottleneck32_finalnbm200_pat10_seed20260802"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def requested_metrics(test: dict) -> dict:
    tn, fp, fn, tp = (int(test[key]) for key in ("tn", "fp", "fn", "tp"))
    precision = (safe_div(tn, tn + fn), safe_div(tp, tp + fp))
    recall = (safe_div(tn, tn + fp), safe_div(tp, tp + fn))
    f1 = (
        safe_div(2 * tn, 2 * tn + fp + fn),
        safe_div(2 * tp, 2 * tp + fp + fn),
    )
    return {
        "acc": safe_div(tn + tp, tn + fp + fn + tp),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    suite = parse_args().suite_dir.resolve()
    rows = []
    for subject in SUBJECTS:
        with (suite / subject / "metrics.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {
            "subject": subject,
            "threshold": payload["selected_threshold"],
            **requested_metrics(payload["test"]),
        }
        rows.append(row)

    macro_subject = {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "std_population": float(np.std([row[name] for row in rows], ddof=0)),
        }
        for name in ("acc", "macro_precision", "macro_recall", "macro_f1")
    }
    combined_counts = {
        key: int(sum(row[key] for row in rows)) for key in ("tn", "fp", "fn", "tp")
    }
    combined = requested_metrics(combined_counts)
    result = {
        "definitions": {
            "classes": ["non-FoG", "FoG"],
            "class_count": 2,
            "acc": "sum_c TP_c / N",
            "macro_precision": "mean_c TP_c/(TP_c+FP_c)",
            "macro_recall": "mean_c TP_c/(TP_c+FN_c)",
            "macro_f1": "mean_c 2TP_c/(2TP_c+FP_c+FN_c)",
            "zero_division": 0,
        },
        "per_subject": rows,
        "subject_macro_statistics": macro_subject,
        "combined_window_counts": combined,
    }
    with (suite / "aggregate_four_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (suite / "aggregate_four_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
