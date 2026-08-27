#!/usr/bin/env python3
"""Recompute subject AP, event sensitivity, and FA/h under the latest rule.

This is a read-only post-processing step for the completed within-subject
processed_NBM_Exp GRU-BASE Mask4--8 NBM + scheme-C TCN experiment.  It does
not retrain a model, change a threshold, or overwrite the original summary.

Aggregation is fixed to:
  1. compute each metric independently for every subject/fold/seed test run;
  2. within each subject and seed, macro-average the three folds;
  3. within each subject, report mean and population SD over five seeds.

Event rules are fixed to:
  * one permanent-test FoG allocation group is one reference event;
  * any positive window in that group detects the event;
  * false alarms use only role-0, true Non-FoG windows;
  * within one record, positive decisions whose start times are at most 1 s
    apart are merged, irrespective of allocation-group boundaries;
  * different records are never merged;
  * the denominator is the union coverage of evaluated valid Non-FoG samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base


SUBJECTS = tuple(f"P{index:02d}" for index in range(1, 9))
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
MERGE_GAP_SECONDS = 1.0
OUTPUT_NAME = "subject_AP_EventSensitivity_FAh_latest_mean_SD_4sig.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / (
            "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_"
            "tcn_nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=f"Default: <output-root>/{OUTPUT_NAME}",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("refusing to write an empty result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def permanent_fog_groups(data_dir: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    manifest = data_dir / "nbm_window_manifest.csv"
    for row in read_csv(manifest):
        if row["active_for_outer_fold"].strip().lower() != "true":
            continue
        if int(row["role_code"]) != 1:
            continue
        window_id = row["window_id"]
        group_id = row["allocation_group_id"]
        if not group_id:
            raise AssertionError(f"role-1 window has no allocation group: {window_id}")
        previous = lookup.setdefault(window_id, group_id)
        if previous != group_id:
            raise AssertionError(f"window changes allocation group: {window_id}")
    if not lookup:
        raise ValueError(f"no permanent-test FoG groups found in {manifest}")
    return lookup


def validate_prediction_identity(
    prediction_rows: list[dict[str, str]],
    test_rows: Any,
    subject: str,
    fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prediction_rows) != len(test_rows):
        raise AssertionError(
            f"prediction length mismatch for {subject}/fold{fold}/seed{seed}: "
            f"{len(prediction_rows)} != {len(test_rows)}"
        )
    probability = np.empty(len(test_rows), dtype=np.float64)
    prediction = np.empty(len(test_rows), dtype=np.int8)
    target = np.empty(len(test_rows), dtype=np.int8)
    for index, row in enumerate(prediction_rows):
        expected = (
            subject,
            fold,
            seed,
            str(test_rows.record_id[index]),
            int(test_rows.start[index]),
            int(test_rows.end[index]),
            int(test_rows.role[index]),
            str(test_rows.window_id[index]),
            int(test_rows.label[index]),
        )
        observed = (
            row["subject_id"],
            int(row["fold"]),
            int(row["seed"]),
            row["record_id"],
            int(row["start_index"]),
            int(row["end_index_exclusive"]),
            int(row["role_code"]),
            row["window_id"],
            int(row["y_true"]),
        )
        if observed != expected:
            raise AssertionError(
                f"prediction identity mismatch at row {index}: "
                f"observed={observed}, expected={expected}"
            )
        probability[index] = float(row["probability"])
        prediction[index] = int(row["y_pred"])
        target[index] = int(row["y_true"])
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"non-finite probability in {subject}/fold{fold}/seed{seed}")
    if not np.all(np.isin(prediction, (0, 1))):
        raise ValueError("predictions must be binary")
    return target, probability, prediction


def latest_event_metrics(
    dataset: DaphnetDataset,
    test_rows: Any,
    prediction: np.ndarray,
    group_lookup: dict[str, str],
) -> tuple[float, float, int, int, int, float]:
    group_predictions: dict[str, list[int]] = defaultdict(list)
    for window_id, role, label, predicted in zip(
        test_rows.window_id,
        test_rows.role,
        test_rows.label,
        prediction,
    ):
        if int(role) != 1:
            continue
        if int(label) != 1:
            raise AssertionError("role-1 event window is not true FoG")
        key = str(window_id)
        if key not in group_lookup:
            raise KeyError(f"role-1 window absent from allocation manifest: {key}")
        group_predictions[group_lookup[key]].append(int(predicted))
    if not group_predictions:
        raise ValueError("no permanent-test FoG allocation groups were evaluated")
    event_count = len(group_predictions)
    detected_events = sum(int(any(values)) for values in group_predictions.values())

    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for record_index, start, end, role, label, predicted in zip(
        test_rows.record_index,
        test_rows.start,
        test_rows.end,
        test_rows.role,
        test_rows.label,
        prediction,
    ):
        if int(role) != 0:
            continue
        if int(label) != 0:
            raise AssertionError("role-0 false-alarm support is not true Non-FoG")
        by_record[int(record_index)].append(
            (int(start), int(end), int(predicted))
        )

    false_alarm_events = 0
    nonfog_seconds = 0.0
    maximum_start_gap = int(round(dataset.sampling_rate_hz * MERGE_GAP_SECONDS))
    for record_index, record_rows in by_record.items():
        record_rows.sort(key=lambda item: item[0])
        positive_starts = [start for start, _, pred in record_rows if pred == 1]
        if positive_starts:
            false_alarm_events += 1 + sum(
                int(current - previous > maximum_start_gap)
                for previous, current in zip(positive_starts, positive_starts[1:])
            )
        record = dataset.records[record_index]
        coverage = np.zeros(len(record.y), dtype=bool)
        for start, end, _ in record_rows:
            coverage[start:end] = True
        nonfog_seconds += float(
            np.sum(coverage & record.valid & (record.y == 0))
        ) / float(dataset.sampling_rate_hz)
    nonfog_hours = nonfog_seconds / 3600.0
    if nonfog_hours <= 0.0:
        raise ValueError("evaluated Non-FoG exposure is empty")
    return (
        detected_events / event_count,
        false_alarm_events / nonfog_hours,
        detected_events,
        event_count,
        false_alarm_events,
        nonfog_hours,
    )


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size != len(SEEDS) or not np.all(np.isfinite(array)):
        raise ValueError(f"expected {len(SEEDS)} finite seed values, got {array}")
    return float(np.mean(array)), float(np.std(array, ddof=0))


def format_4sig(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"cannot format non-finite value: {value}")
    if value == 0.0:
        return "0.000"
    magnitude = math.floor(math.log10(abs(value)))
    decimal_places = 3 - magnitude
    if decimal_places >= 0:
        return f"{value:.{decimal_places}f}"
    return f"{value:.3e}"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else output_root / OUTPUT_NAME
    )

    done = read_json(output_root / "DONE.json")
    plan = read_json(output_root / "EXPERIMENT_PLAN.json")
    barrier = read_json(output_root / "TRAINING_BARRIER.json")
    if done.get("status") != "complete" or int(done.get("run_count", -1)) != 120:
        raise AssertionError("source experiment is not a complete 120-run result")
    if tuple(plan.get("subjects", ())) != SUBJECTS:
        raise AssertionError("source subject contract changed")
    if tuple(plan.get("folds", ())) != FOLDS or tuple(plan.get("seeds", ())) != SEEDS:
        raise AssertionError("source fold/seed contract changed")
    if plan.get("nbm_variant") != "GRU_BASE_MASK4_8":
        raise AssertionError("source is not the requested GRU BASE Mask4-8 experiment")
    data_sha = processed_nbm_scientific_manifest(data_dir)["sha256"]
    if data_sha != plan.get("data_scientific_sha256"):
        raise AssertionError("current processed_NBM_Exp differs from the trained dataset")
    if done.get("barrier_id") != barrier.get("barrier_id"):
        raise AssertionError("source completion marker and training barrier disagree")

    dataset = DaphnetDataset.load(data_dir)
    group_lookup = permanent_fog_groups(data_dir)
    run_metrics: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            rows = raw_base.load_subject_rows(data_dir, dataset, subject, fold)
            test_rows = rows.take_role(0, 1)
            for seed in SEEDS:
                destination = output_root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"
                done_test = read_json(destination / "DONE_TEST.json")
                predictions_path = destination / "test_predictions.csv"
                if done_test.get("barrier_id") != barrier.get("barrier_id"):
                    raise AssertionError(f"barrier mismatch: {destination}")
                if done_test.get("predictions_sha256") != sha256_file(predictions_path):
                    raise AssertionError(f"prediction artifact hash mismatch: {destination}")
                prediction_rows = read_csv(predictions_path)
                target, probability, prediction = validate_prediction_identity(
                    prediction_rows, test_rows, subject, fold, seed
                )
                if np.unique(target).size != 2:
                    raise ValueError(f"AP requires both classes: {subject}/fold{fold}/seed{seed}")
                ap = float(average_precision_score(target, probability))
                (
                    event_sensitivity,
                    false_alarms_per_hour,
                    detected_events,
                    event_count,
                    false_alarm_events,
                    nonfog_hours,
                ) = latest_event_metrics(dataset, test_rows, prediction, group_lookup)
                run_metrics.append(
                    {
                        "subject": subject,
                        "fold": fold,
                        "seed": seed,
                        "AP": ap,
                        "Event_Sensitivity": event_sensitivity,
                        "False_Alarms_per_hour": false_alarms_per_hour,
                        "detected_events": detected_events,
                        "reference_events": event_count,
                        "false_alarm_events": false_alarm_events,
                        "nonfog_hours": nonfog_hours,
                    }
                )
    if len(run_metrics) != 120:
        raise AssertionError(f"expected 120 run metrics, got {len(run_metrics)}")

    subject_seed: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            selected = [
                row
                for row in run_metrics
                if row["subject"] == subject and row["seed"] == seed
            ]
            if len(selected) != len(FOLDS):
                raise AssertionError(f"missing fold for {subject}/seed{seed}")
            subject_seed.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "AP": float(np.mean([row["AP"] for row in selected])),
                    "Event_Sensitivity": float(
                        np.mean([row["Event_Sensitivity"] for row in selected])
                    ),
                    "False_Alarms_per_hour": float(
                        np.mean([row["False_Alarms_per_hour"] for row in selected])
                    ),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        selected = [row for row in subject_seed if row["subject"] == subject]
        if len(selected) != len(SEEDS):
            raise AssertionError(f"missing seed for {subject}")
        ap_mean, ap_sd = mean_sd(row["AP"] for row in selected)
        event_mean, event_sd = mean_sd(
            row["Event_Sensitivity"] for row in selected
        )
        fah_mean, fah_sd = mean_sd(
            row["False_Alarms_per_hour"] for row in selected
        )
        summary_rows.append(
            {
                "Subject": subject,
                "AP_mean": format_4sig(ap_mean),
                "AP_SD": format_4sig(ap_sd),
                "Event_Sensitivity_mean": format_4sig(event_mean),
                "Event_Sensitivity_SD": format_4sig(event_sd),
                "False_Alarms_per_hour_mean": format_4sig(fah_mean),
                "False_Alarms_per_hour_SD": format_4sig(fah_sd),
            }
        )
    write_csv(output_csv, summary_rows)
    print(output_csv)
    for row in summary_rows:
        print(
            f"{row['Subject']}: AP={row['AP_mean']}±{row['AP_SD']}; "
            f"EventSens={row['Event_Sensitivity_mean']}±{row['Event_Sensitivity_SD']}; "
            f"FA/h={row['False_Alarms_per_hour_mean']}±{row['False_Alarms_per_hour_SD']}"
        )


if __name__ == "__main__":
    main()
