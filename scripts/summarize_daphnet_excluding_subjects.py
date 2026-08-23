#!/usr/bin/env python3
"""Recompute frozen Daphnet test metrics after excluding named subjects."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.evaluation import binary_metrics


METRICS = ("sensitivity", "precision", "specificity", "auprc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--method", default="FULL_C")
    parser.add_argument("--exclude-subjects", default="S07")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.experiment_dir.resolve()
    excluded = tuple(
        item.strip() for item in args.exclude_subjects.split(",") if item.strip()
    )
    if not excluded or len(excluded) != len(set(excluded)):
        raise ValueError(f"invalid excluded-subject list: {excluded}")

    paths = sorted(
        root.glob(
            f"runs/fold_*/method_{args.method}/seed_*/test_predictions.csv"
        )
    )
    if not paths:
        raise FileNotFoundError(f"no frozen predictions found under {root}")

    run_rows: list[dict[str, Any]] = []
    observed_subjects: set[str] = set()
    for path in paths:
        rows = read_rows(path)
        observed_subjects.update(row["subject_id"] for row in rows)
        kept = [row for row in rows if row["subject_id"] not in excluded]
        if not kept:
            raise ValueError(f"subject exclusion removed every row: {path}")
        folds = {int(row["fold"]) for row in rows}
        seeds = {int(row["tcn_seed"]) for row in rows}
        methods = {row["method"] for row in rows}
        thresholds = {float(row["threshold"]) for row in rows}
        if len(folds) != 1 or len(seeds) != 1 or methods != {args.method}:
            raise AssertionError(f"prediction identity mismatch: {path}")
        if len(thresholds) != 1:
            raise AssertionError(f"multiple thresholds in one frozen run: {path}")
        y_true = np.asarray([int(row["y_true"]) for row in kept], dtype=np.int8)
        y_prob = np.asarray(
            [float(row["fog_probability"]) for row in kept], dtype=np.float64
        )
        threshold = next(iter(thresholds))
        metrics = binary_metrics(y_true, y_prob, threshold)
        run_rows.append(
            {
                "fold": next(iter(folds)),
                "seed": next(iter(seeds)),
                "method": args.method,
                "excluded_subjects": ",".join(excluded),
                "threshold": threshold,
                "test_windows": int(metrics["n"]),
                "nonfog_windows": int(metrics["n_normal"]),
                "fog_windows": int(metrics["n_fog"]),
                **{key: float(metrics[key]) for key in METRICS},
                **{key: int(metrics[key]) for key in ("tn", "fp", "fn", "tp")},
            }
        )

    missing = set(excluded) - observed_subjects
    if missing:
        raise ValueError(f"excluded subjects are absent from predictions: {missing}")
    fold_ids = sorted({int(row["fold"]) for row in run_rows})
    seed_ids = sorted({int(row["seed"]) for row in run_rows})
    if len(run_rows) != len(fold_ids) * len(seed_ids):
        raise AssertionError("incomplete fold-seed prediction grid")

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_seed[int(row["seed"])].append(row)
    seed_rows = []
    for seed in seed_ids:
        rows = by_seed[seed]
        if sorted(int(row["fold"]) for row in rows) != fold_ids:
            raise AssertionError(f"incomplete folds for seed {seed}")
        seed_rows.append(
            {
                "seed": seed,
                "folds": len(rows),
                **{
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in METRICS
                },
            }
        )

    summary = {
        "experiment": str(root),
        "method": args.method,
        "evaluation_only": True,
        "model_retrained": False,
        "threshold_changed": False,
        "excluded_subjects": list(excluded),
        "remaining_subjects": sorted(observed_subjects - set(excluded)),
        "folds": fold_ids,
        "seeds": seed_ids,
        "aggregation": (
            "pool remaining-subject windows within each fold-seed run; mean over "
            "folds within seed; mean and population SD over seeds"
        ),
        "metrics": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in seed_rows])),
                "std": float(np.std([float(row[key]) for row in seed_rows], ddof=0)),
                "n": len(seed_rows),
            }
            for key in METRICS
        },
    }

    suffix = "_".join(excluded)
    prefix = f"{args.method.lower()}_exclude_{suffix}"
    write_rows(root / f"{prefix}_run_metrics_{len(run_rows)}.csv", run_rows)
    write_rows(root / f"{prefix}_seed_macro.csv", seed_rows)
    with (root / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
