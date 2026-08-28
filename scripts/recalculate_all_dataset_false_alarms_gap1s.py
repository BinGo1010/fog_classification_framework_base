#!/usr/bin/env python
"""Recalculate FA/hour from positive decisions inside true Non-FoG periods.

Definition implemented here:

* only role-0 rows (pure true Non-FoG windows) with y_pred=1 are false-positive
  decisions;
* each decision time is the window end, because the detector emits one result
  per stride/update;
* within one record, positive decision times separated by <= 1 second belong
  to one false-alarm event; a larger separation starts a new event;
* decisions from different records are never merged;
* the denominator is the union of evaluated role-0 window support intersected
  with record.valid and sample-level y==0, expressed in hours.

The sealed original metrics are not modified.  Versioned recalculation files
are written below the experiment root.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest


SUBJECTS = tuple(f"P{index:02d}" for index in range(1, 9))
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
METRIC_KEYS = (
    "sensitivity",
    "precision",
    "specificity",
    "pr_auc",
    "event_sensitivity",
    "false_alarms_per_hour",
)
SCHEMA = "nonfog_positive_decision_false_alarm.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-positive-decisions", type=int, default=1)
    parser.add_argument(
        "--recalculation-dir-name",
        default="false_alarm_gap1s_recalculation",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n": int(len(array)),
    }


def recalculate_run(
    dataset: Any,
    prediction_rows: list[dict[str, str]],
    merge_gap_seconds: float,
    minimum_positive_decisions: int = 1,
) -> dict[str, Any]:
    if minimum_positive_decisions < 1:
        raise ValueError("minimum_positive_decisions must be >= 1")
    record_lookup = {record.record_id: record for record in dataset.records}
    by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        role = int(row["role_code"])
        y_true = int(row["y_true"])
        if role not in (0, 1) or y_true != role:
            raise AssertionError(f"unexpected permanent-test role/label row: {row}")
        if row["record_id"] not in record_lookup:
            raise KeyError(f"unknown record_id: {row['record_id']}")
        by_record[row["record_id"]].append(row)

    false_alarm_events = 0
    false_positive_decisions = 0
    evaluated_nonfog_samples = 0
    per_record: list[dict[str, Any]] = []
    maximum_gap_samples = int(round(float(merge_gap_seconds) * dataset.sampling_rate_hz))

    for record_id, rows in sorted(by_record.items()):
        record = record_lookup[record_id]
        coverage = np.zeros(len(record.y), dtype=bool)
        positive_decision_samples: list[int] = []
        for row in rows:
            if int(row["role_code"]) != 0:
                continue
            start = int(row["start_index"])
            end = int(row["end_index_exclusive"])
            coverage[start:end] = True
            if int(row["y_pred"]) == 1:
                positive_decision_samples.append(end)

        valid_nonfog_samples = int(np.sum(coverage & record.valid & (record.y == 0)))
        evaluated_nonfog_samples += valid_nonfog_samples
        decision_times = sorted(set(positive_decision_samples))
        record_events = 0
        previous: int | None = None
        active_positive_decisions = 0
        for decision_sample in decision_times:
            if previous is None or decision_sample - previous > maximum_gap_samples:
                if active_positive_decisions >= minimum_positive_decisions:
                    record_events += 1
                active_positive_decisions = 1
            else:
                active_positive_decisions += 1
            previous = decision_sample
        if active_positive_decisions >= minimum_positive_decisions:
            record_events += 1
        false_alarm_events += record_events
        false_positive_decisions += len(decision_times)
        per_record.append(
            {
                "record_id": record_id,
                "false_positive_decisions": len(decision_times),
                "false_alarm_events": record_events,
                "evaluated_nonfog_seconds": valid_nonfog_samples / dataset.sampling_rate_hz,
            }
        )

    evaluated_nonfog_hours = evaluated_nonfog_samples / (
        dataset.sampling_rate_hz * 3600.0
    )
    return {
        "false_alarm_metric_schema": SCHEMA,
        "merge_gap_seconds": float(merge_gap_seconds),
        "decision_time": "end_index_exclusive / sampling_rate_hz",
        "minimum_positive_decisions": int(minimum_positive_decisions),
        "same_record_only": True,
        "false_positive_support": "role_code=0, y_true=0, y_pred=1",
        "denominator": "union(role0 window support) & record.valid & sample_y==0",
        "false_positive_decisions": int(false_positive_decisions),
        "false_alarm_events": int(false_alarm_events),
        "evaluated_nonfog_hours": float(evaluated_nonfog_hours),
        "false_alarms_per_hour": (
            float(false_alarm_events / evaluated_nonfog_hours)
            if evaluated_nonfog_hours
            else None
        ),
        "per_record": per_record,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    if args.merge_gap_seconds < 0:
        raise ValueError("--merge-gap-seconds must be non-negative")
    if args.minimum_positive_decisions < 1:
        raise ValueError("--minimum-positive-decisions must be >= 1")
    done = json.loads((output_root / "DONE.json").read_text(encoding="utf-8"))
    if done.get("status") != "complete" or int(done.get("run_count", -1)) != 120:
        raise AssertionError("source experiment is not a complete 120-run result")
    plan = json.loads((output_root / "EXPERIMENT_PLAN.json").read_text(encoding="utf-8"))
    current_data_sha = processed_nbm_scientific_manifest(data_dir)["sha256"]
    if plan.get("data_scientific_sha256") != current_data_sha:
        raise AssertionError("current dataset differs from the sealed experiment data")
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64:
        raise AssertionError(f"expected 64 Hz, got {dataset.sampling_rate_hz}")

    original_runs = read_csv(output_root / "run_metrics.csv")
    original_lookup = {
        (row["subject"], int(row["fold"]), int(row["seed"])): row
        for row in original_runs
    }
    if len(original_lookup) != 120:
        raise AssertionError(f"expected 120 unique run metrics, got {len(original_lookup)}")

    run_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = output_root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"
                predictions_path = run_dir / "test_predictions.csv"
                done_test_path = run_dir / "DONE_TEST.json"
                done_test = json.loads(done_test_path.read_text(encoding="utf-8"))
                if done_test.get("predictions_sha256") != sha256_file(predictions_path):
                    raise AssertionError(f"prediction hash mismatch: {run_dir}")
                recalculated = recalculate_run(
                    dataset,
                    read_csv(predictions_path),
                    args.merge_gap_seconds,
                    args.minimum_positive_decisions,
                )
                original = original_lookup[(subject, fold, seed)]
                run_rows.append(
                    {
                        "subject": subject,
                        "fold": fold,
                        "seed": seed,
                        "threshold": float(original["threshold"]),
                        "sensitivity": float(original["sensitivity"]),
                        "precision": float(original["precision"]),
                        "specificity": float(original["specificity"]),
                        "pr_auc": float(original["pr_auc"]),
                        "event_sensitivity": float(original["event_sensitivity"]),
                        "false_alarms_per_hour": recalculated[
                            "false_alarms_per_hour"
                        ],
                        "false_positive_decisions": recalculated[
                            "false_positive_decisions"
                        ],
                        "false_alarm_events": recalculated["false_alarm_events"],
                        "evaluated_nonfog_hours": recalculated[
                            "evaluated_nonfog_hours"
                        ],
                        "original_false_alarms_per_hour": float(
                            original["false_alarms_per_hour"]
                        ),
                    }
                )
                for record_row in recalculated["per_record"]:
                    detail_rows.append(
                        {
                            "subject": subject,
                            "fold": fold,
                            "seed": seed,
                            **record_row,
                        }
                    )

    subject_seed_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            selected = [
                row
                for row in run_rows
                if row["subject"] == subject and row["seed"] == seed
            ]
            subject_seed_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    **{
                        key: float(np.mean([row[key] for row in selected]))
                        for key in METRIC_KEYS
                    },
                }
            )

    subject_summary_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        selected = [row for row in subject_seed_rows if row["subject"] == subject]
        summary_row: dict[str, Any] = {"subject": subject}
        for key in METRIC_KEYS:
            summary = mean_std(row[key] for row in selected)
            summary_row[f"{key}_mean"] = summary["mean"]
            summary_row[f"{key}_std"] = summary["std"]
        subject_summary_rows.append(summary_row)

    overall_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        selected = [row for row in subject_seed_rows if row["seed"] == seed]
        overall_seed_rows.append(
            {
                "seed": seed,
                **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in METRIC_KEYS
                },
            }
        )
    overall = {
        key: mean_std(row[key] for row in overall_seed_rows)
        for key in METRIC_KEYS
    }

    destination = output_root / args.recalculation_dir_name
    write_csv(destination / "run_metrics_gap1s.csv", run_rows)
    write_csv(destination / "record_false_alarm_details_gap1s.csv", detail_rows)
    write_csv(destination / "subject_seed_metrics_gap1s.csv", subject_seed_rows)
    write_csv(destination / "subject_summary_gap1s.csv", subject_summary_rows)
    write_csv(destination / "overall_seed_metrics_gap1s.csv", overall_seed_rows)
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_output_root": str(output_root),
        "source_barrier_id": done["barrier_id"],
        "source_summary_sha256": done["summary_sha256"],
        "data_scientific_sha256": current_data_sha,
        "definition": {
            "eligible_predictions": "role0 true Non-FoG windows predicted FoG",
            "decision_timestamp": "window end_index_exclusive / 64 Hz",
            "minimum_positive_decisions": args.minimum_positive_decisions,
            "merge_rule": (
                "within the same record, successive positive decision timestamps "
                f"with separation <= {args.merge_gap_seconds:g} s are one false alarm"
            ),
            "different_records_never_merged": True,
            "denominator": (
                "union of evaluated role0 window support intersected with valid "
                "sample-level Non-FoG, in hours"
            ),
        },
        "aggregation": (
            "subject/seed macro mean of 3 folds; subject mean+population SD over "
            "5 seeds; overall subject-macro per seed then mean+population SD"
        ),
        "subjects": subject_summary_rows,
        "overall": overall,
    }
    atomic_json_dump(summary, destination / "summary_gap1s.json")
    atomic_json_dump(
        {
            "schema": SCHEMA,
            "status": "complete",
            "run_count": len(run_rows),
            "summary_sha256": sha256_file(destination / "summary_gap1s.json"),
        },
        destination / "DONE.json",
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
