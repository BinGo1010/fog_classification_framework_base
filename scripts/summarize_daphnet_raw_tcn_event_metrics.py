#!/usr/bin/env python
"""Add event-level metrics to the frozen Daphnet RAW+TCN test outputs.

This script performs evaluation only.  It does not train a model, alter a
checkpoint, change a validation-selected threshold, or regenerate test
probabilities.

Definitions
-----------
* Each permanent-test FoG allocation group is one annotated FoG event.
* A FoG event is detected when at least one of its pure-FoG test windows is
  classified as FoG.
* Consecutive positive predictions on pure Non-FoG test windows within the
  same record are merged into one false-alarm episode.
* Non-FoG exposure is the union duration of the pure Non-FoG test-window
  intervals, so overlapping windows are not counted repeatedly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXPERIMENT = Path(
    "outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_"
    "seedset_0_52_161_5216_52161"
)
DEFAULT_DATASET = Path(
    "dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM"
)
EXPECTED_FOLDS = (0, 1, 2)
EXPECTED_SEEDS = (0, 52, 161, 5216, 52161)
FS = 64
STRIDE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.mean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "n": len(numbers),
    }


def interval_union_samples(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    if current_end <= current_start:
        raise ValueError("Non-positive monitoring interval")
    for start, end in ordered[1:]:
        if end <= start:
            raise ValueError("Non-positive monitoring interval")
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def positive_episodes(rows: list[dict[str, Any]]) -> int:
    """Count contiguous positive runs at the fixed one-second update grid."""

    positives = sorted(
        int(row["start_index"]) for row in rows if int(row["y_pred"]) == 1
    )
    if not positives:
        return 0
    if len(positives) != len(set(positives)):
        raise ValueError("Duplicate positive window start within one record")
    return 1 + sum(
        int(current - previous > STRIDE)
        for previous, current in zip(positives, positives[1:])
    )


def load_window_lookup(dataset_dir: Path) -> dict[tuple[int, str], dict[str, str]]:
    lookup: dict[tuple[int, str], dict[str, str]] = {}
    path = dataset_dir / "nbm_window_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role_code"] not in {"0", "1"}:
                continue
            if row["active_for_outer_fold"].lower() != "true":
                continue
            key = (int(row["outer_fold_id"]), row["window_id"])
            if key in lookup:
                raise ValueError(f"Duplicate active test window key: {key}")
            lookup[key] = row
    return lookup


def load_fog_groups(dataset_dir: Path) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for row in read_csv(dataset_dir / "nbm_group_manifest.csv"):
        if row["permanent_partition"] != "permanent_test":
            continue
        if row["class_label"] != "FOG":
            continue
        if int(row["event_count"]) != 1:
            raise ValueError(
                f"Event-level evaluation requires one event per test group: {row['group_id']}"
            )
        groups[row["group_id"]] = row
    return groups


def parse_run_path(path: Path) -> tuple[int, int]:
    fold_match = re.fullmatch(r"fold_(\d+)", path.parents[2].name)
    seed_match = re.fullmatch(r"seed_(\d+)", path.parent.name)
    if fold_match is None or seed_match is None:
        raise ValueError(f"Unexpected prediction path: {path}")
    return int(fold_match.group(1)), int(seed_match.group(1))


def evaluate_run(
    prediction_path: Path,
    window_lookup: dict[tuple[int, str], dict[str, str]],
    fog_groups: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold, seed = parse_run_path(prediction_path)
    predictions = read_csv(prediction_path)
    joined: list[dict[str, Any]] = []
    for prediction in predictions:
        key = (fold, prediction["window_id"])
        manifest = window_lookup.get(key)
        if manifest is None:
            raise ValueError(f"Prediction is absent from active test manifest: {key}")
        if prediction["role_code"] != manifest["role_code"]:
            raise ValueError(f"Role mismatch for {key}")
        if int(prediction["y_true"]) != int(manifest["y_binary"]):
            raise ValueError(f"Label mismatch for {key}")
        joined.append(
            {
                **prediction,
                "fold": fold,
                "seed": seed,
                "start_index": int(prediction["start_index"]),
                "end_index_exclusive": int(prediction["end_index_exclusive"]),
                "role_code": int(prediction["role_code"]),
                "y_true": int(prediction["y_true"]),
                "y_pred": int(prediction["y_pred"]),
                "allocation_group_id": manifest["allocation_group_id"],
            }
        )

    event_rows = [row for row in joined if row["role_code"] == 1]
    nonfog_rows = [row for row in joined if row["role_code"] == 0]
    if any(row["y_true"] != 1 for row in event_rows):
        raise ValueError("Role-1 test population contains a Non-FoG label")
    if any(row["y_true"] != 0 for row in nonfog_rows):
        raise ValueError("Role-0 test population contains a FoG label")

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        group_id = str(row["allocation_group_id"])
        if group_id not in fog_groups:
            raise ValueError(f"Unknown permanent-test FoG group: {group_id!r}")
        by_event[group_id].append(row)

    subjects = sorted({str(row["subject_id"]) for row in joined})
    subject_rows: list[dict[str, Any]] = []
    total_detected = 0
    total_events = 0
    total_false_alarms = 0
    total_exposure_samples = 0

    for subject in subjects:
        subject_events = {
            group_id: rows
            for group_id, rows in by_event.items()
            if fog_groups[group_id]["subject_id"] == subject
        }
        detected = sum(
            int(any(int(row["y_pred"]) == 1 for row in rows))
            for rows in subject_events.values()
        )
        event_count = len(subject_events)

        subject_nonfog = [row for row in nonfog_rows if row["subject_id"] == subject]
        by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subject_nonfog:
            by_record[str(row["record_id"])].append(row)
        false_alarms = sum(positive_episodes(rows) for rows in by_record.values())
        exposure_samples = sum(
            interval_union_samples(
                (int(row["start_index"]), int(row["end_index_exclusive"]))
                for row in rows
            )
            for rows in by_record.values()
        )
        exposure_hours = exposure_samples / FS / 3600.0
        subject_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "subject_id": subject,
                "detected_events": detected,
                "total_events": event_count,
                "event_sensitivity": detected / event_count if event_count else math.nan,
                "false_alarm_episodes": false_alarms,
                "nonfog_exposure_hours": exposure_hours,
                "false_alarm_events_per_hour": (
                    false_alarms / exposure_hours if exposure_hours else math.nan
                ),
            }
        )
        total_detected += detected
        total_events += event_count
        total_false_alarms += false_alarms
        total_exposure_samples += exposure_samples

    if total_events != 19:
        raise ValueError(f"Expected 19 permanent-test FoG events, found {total_events}")
    total_exposure_hours = total_exposure_samples / FS / 3600.0
    run_row = {
        "fold": fold,
        "seed": seed,
        "threshold": float(predictions[0]["threshold"]),
        "detected_events": total_detected,
        "total_events": total_events,
        "event_sensitivity": total_detected / total_events,
        "false_alarm_episodes": total_false_alarms,
        "nonfog_exposure_hours": total_exposure_hours,
        "false_alarm_events_per_hour": total_false_alarms / total_exposure_hours,
    }
    return run_row, subject_rows


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    window_lookup = load_window_lookup(dataset_dir)
    fog_groups = load_fog_groups(dataset_dir)
    prediction_paths = sorted(
        experiment_dir.glob("runs/fold_*/method_RAW/seed_*/test_predictions.csv")
    )
    if len(prediction_paths) != len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS):
        raise ValueError(
            f"Expected 15 frozen RAW prediction files, found {len(prediction_paths)}"
        )

    run_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        run_row, run_subject_rows = evaluate_run(
            prediction_path, window_lookup, fog_groups
        )
        run_rows.append(run_row)
        subject_rows.extend(run_subject_rows)

    observed = {(int(row["fold"]), int(row["seed"])) for row in run_rows}
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    if observed != expected:
        raise ValueError(f"Run grid mismatch: missing={expected-observed}, extra={observed-expected}")

    seed_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        rows = [row for row in run_rows if int(row["seed"]) == seed]
        seed_rows.append(
            {
                "seed": seed,
                "folds": len(rows),
                "event_sensitivity": statistics.mean(
                    float(row["event_sensitivity"]) for row in rows
                ),
                "false_alarm_events_per_hour": statistics.mean(
                    float(row["false_alarm_events_per_hour"]) for row in rows
                ),
            }
        )

    summary = {
        "experiment": str(experiment_dir),
        "method": "RAW+TCN",
        "evaluation_only": True,
        "model_retrained": False,
        "threshold_changed": False,
        "folds": list(EXPECTED_FOLDS),
        "seeds": list(EXPECTED_SEEDS),
        "sampling_rate_hz": FS,
        "stride_samples": STRIDE,
        "event_definition": (
            "one permanent-test FoG allocation group; detected if any pure-FoG "
            "test window is predicted positive"
        ),
        "false_alarm_definition": (
            "one contiguous run of positive predictions on pure Non-FoG test "
            "windows within a record"
        ),
        "exposure_definition": (
            "union duration of pure Non-FoG test-window intervals"
        ),
        "aggregation": (
            "mean over 3 folds within each seed, then mean and sample SD over 5 seeds"
        ),
        "event_sensitivity": mean_sd(
            row["event_sensitivity"] for row in seed_rows
        ),
        "false_alarm_events_per_hour": mean_sd(
            row["false_alarm_events_per_hour"] for row in seed_rows
        ),
    }

    run_columns = [
        "fold",
        "seed",
        "threshold",
        "detected_events",
        "total_events",
        "event_sensitivity",
        "false_alarm_episodes",
        "nonfog_exposure_hours",
        "false_alarm_events_per_hour",
    ]
    subject_columns = [
        "fold",
        "seed",
        "subject_id",
        "detected_events",
        "total_events",
        "event_sensitivity",
        "false_alarm_episodes",
        "nonfog_exposure_hours",
        "false_alarm_events_per_hour",
    ]
    seed_columns = [
        "seed",
        "folds",
        "event_sensitivity",
        "false_alarm_events_per_hour",
    ]
    write_csv(experiment_dir / "raw_event_metrics_15runs.csv", run_rows, run_columns)
    write_csv(
        experiment_dir / "raw_event_metrics_by_subject_15runs.csv",
        subject_rows,
        subject_columns,
    )
    write_csv(
        experiment_dir / "raw_event_metrics_5seed_macro.csv", seed_rows, seed_columns
    )
    with (experiment_dir / "raw_event_metrics_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    event = summary["event_sensitivity"]
    false_alarm = summary["false_alarm_events_per_hour"]
    report = "\n".join(
        [
            "# Daphnet RAW+TCN event-level test metrics",
            "",
            "This is an evaluation-only supplement based on frozen test predictions. "
            "No model was retrained and no validation-selected threshold was changed.",
            "",
            "## Result",
            "",
            "| Metric | Mean ± SD |",
            "|---|---:|",
            f"| Event Sensitivity | {event['mean']:.4f} ± {event['std']:.4f} |",
            (
                "| False Alarms/hour | "
                f"{false_alarm['mean']:.2f} ± {false_alarm['std']:.2f} |"
            ),
            "",
            "The mean and sample SD are calculated over five seed-level values. "
            "Each seed-level value is the mean of the three fixed folds.",
            "",
            "## Frozen evaluation definitions",
            "",
            "- One permanent-test FoG allocation group is one annotated event.",
            "- An event is detected if any of its pure-FoG test windows is predicted positive.",
            "- Consecutive positive predictions on pure Non-FoG windows within one record "
            "are merged into one false-alarm episode.",
            "- False Alarms/hour uses the union duration of Non-FoG test-window intervals; "
            "overlapping windows are not counted repeatedly.",
            "",
        ]
    )
    (experiment_dir / "RAW_EVENT_METRICS_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
