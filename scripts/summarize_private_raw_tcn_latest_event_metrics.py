"""Recompute private-dataset Raw+TCN AP and event metrics under the latest rule.

The permanent-test predictions are read only.  Within each record, positive
window intervals are merged when their temporal supports overlap or their gap
is at most one second, irrespective of allocation-group boundaries.  Records
are never merged.  A merged cluster forms an alarm only when it contains at
least two positive windows; isolated positive windows are discarded.  False
alarms are constructed exclusively from alarms on evaluated, truly Non-FoG
windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file


DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "all_dataset_processed_NBM_Exp_within_subject_raw_tcn_"
    "ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "processed_NBM_Exp_raw_tcn_latest_event_metrics_1s_min2"
)
SUBJECTS = tuple(f"P{index:02d}" for index in range(1, 9))
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
MERGE_GAP_SECONDS = 1.0
MINIMUM_POSITIVE_WINDOWS = 2
EXPECTED_WINDOW_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def format_four_significant(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric cannot be formatted: {value}")
    if value == 0.0:
        return "0.000"
    decimals = 3 - math.floor(math.log10(abs(value)))
    if decimals >= 0:
        return f"{value:.{decimals}f}"
    factor = 10.0 ** (-decimals)
    rounded = round(value / factor) * factor
    return f"{rounded:.0f}"


def load_allocation_groups(
    manifest_path: Path,
) -> dict[tuple[int, str], str]:
    mapping: dict[tuple[int, str], str] = {}
    for row in read_csv(manifest_path):
        key = (int(row["outer_fold_id"]), row["window_id"])
        if key in mapping:
            raise AssertionError(f"duplicate manifest key: {key}")
        mapping[key] = row["allocation_group_id"]
    return mapping


def merge_intervals(
    rows: list[dict[str, Any]],
    maximum_gap_samples: int,
) -> list[dict[str, Any]]:
    """Merge positive 2-s window supports within one record."""

    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row["start"], row["end"], row["window_id"]))
    events: list[dict[str, Any]] = []
    active = {
        "start": int(ordered[0]["start"]),
        "end": int(ordered[0]["end"]),
        "window_ids": [str(ordered[0]["window_id"])],
        "allocation_groups": {str(ordered[0]["allocation_group_id"])},
    }
    for row in ordered[1:]:
        start = int(row["start"])
        end = int(row["end"])
        gap = start - int(active["end"])
        if gap <= maximum_gap_samples:
            active["end"] = max(int(active["end"]), end)
            active["window_ids"].append(str(row["window_id"]))
            active["allocation_groups"].add(str(row["allocation_group_id"]))
            continue
        events.append(active)
        active = {
            "start": start,
            "end": end,
            "window_ids": [str(row["window_id"])],
            "allocation_groups": {str(row["allocation_group_id"])},
        }
    events.append(active)
    for event in events:
        event["positive_window_count"] = len(event["window_ids"])
        event["allocation_group_count"] = len(event["allocation_groups"])
        event["cross_allocation_group"] = event["allocation_group_count"] > 1
        event["allocation_groups"] = sorted(event["allocation_groups"])
    return events


def verify_run_artifacts(run_dir: Path) -> None:
    done_path = run_dir / "DONE_TEST.json"
    predictions_path = run_dir / "test_predictions.csv"
    metrics_path = run_dir / "metrics.json"
    if not done_path.is_file() or not predictions_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"incomplete sealed test run: {run_dir}")
    done = read_json(done_path)
    if done.get("predictions_sha256") != sha256_file(predictions_path):
        raise AssertionError(f"prediction hash mismatch: {run_dir}")
    if done.get("metrics_sha256") != sha256_file(metrics_path):
        raise AssertionError(f"metrics hash mismatch: {run_dir}")


def evaluate_run(
    dataset: Any,
    subject: str,
    fold: int,
    seed: int,
    prediction_rows: list[dict[str, str]],
    allocation_group_by_window: dict[tuple[int, str], str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record_lookup = {record.record_id: record for record in dataset.records}
    expected_window_samples = int(round(EXPECTED_WINDOW_SECONDS * dataset.sampling_rate_hz))
    maximum_gap_samples = int(round(MERGE_GAP_SECONDS * dataset.sampling_rate_hz))
    parsed: list[dict[str, Any]] = []
    for row in prediction_rows:
        if (row["subject_id"], int(row["fold"]), int(row["seed"])) != (
            subject,
            fold,
            seed,
        ):
            raise AssertionError(f"prediction identity mismatch: {row}")
        start = int(row["start_index"])
        end = int(row["end_index_exclusive"])
        if end - start != expected_window_samples:
            raise AssertionError(f"unexpected window duration: {row['window_id']}")
        y_true = int(row["y_true"])
        role_code = int(row["role_code"])
        if role_code not in (0, 1) or y_true != role_code:
            raise AssertionError(f"test role/label mismatch: {row['window_id']}")
        manifest_key = (fold, row["window_id"])
        if manifest_key not in allocation_group_by_window:
            raise KeyError(f"window missing from allocation manifest: {manifest_key}")
        parsed.append(
            {
                "record_id": row["record_id"],
                "start": start,
                "end": end,
                "window_id": row["window_id"],
                "y_true": y_true,
                "probability": float(row["probability"]),
                "y_pred": int(row["y_pred"]),
                "allocation_group_id": allocation_group_by_window[manifest_key],
            }
        )
    if not parsed or len({row["window_id"] for row in parsed}) != len(parsed):
        raise AssertionError(f"empty or duplicate predictions: {subject}/fold{fold}/seed{seed}")

    labels = np.asarray([row["y_true"] for row in parsed], dtype=np.int8)
    probabilities = np.asarray([row["probability"] for row in parsed], dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise AssertionError(f"AP requires both classes: {subject}/fold{fold}/seed{seed}")
    ap = float(average_precision_score(labels, probabilities))

    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_record[row["record_id"]].append(row)

    evaluable_true_events = 0
    detected_true_events = 0
    false_alarm_events = 0
    evaluated_nonfog_samples = 0
    cross_group_false_alarm_events = 0
    event_audit: list[dict[str, Any]] = []
    for record_id, record_rows in sorted(by_record.items()):
        if record_id not in record_lookup:
            raise KeyError(f"unknown record: {record_id}")
        record = record_lookup[record_id]
        nonfog_coverage = np.zeros(len(record.y), dtype=bool)
        fog_coverage = np.zeros(len(record.y), dtype=bool)
        for row in record_rows:
            target = nonfog_coverage if row["y_true"] == 0 else fog_coverage
            target[row["start"] : row["end"]] = True
        if np.any(nonfog_coverage & (record.y != 0)):
            raise AssertionError(f"Non-FoG test coverage crosses true FoG: {record_id}")
        if np.any(fog_coverage & (record.y != 1)):
            raise AssertionError(f"FoG test coverage crosses true Non-FoG: {record_id}")
        evaluated_nonfog_samples += int(
            np.sum(nonfog_coverage & record.valid & (record.y == 0))
        )

        positive_fog_rows = [
            row for row in record_rows if row["y_true"] == 1 and row["y_pred"] == 1
        ]
        positive_fog_alarms = [
            event
            for event in merge_intervals(positive_fog_rows, maximum_gap_samples)
            if int(event["positive_window_count"]) >= MINIMUM_POSITIVE_WINDOWS
        ]
        true_events = [
            interval
            for interval in boolean_runs(record.y == 1)
            if max(interval[0], 0) < min(interval[1], len(record.y))
            and np.any(fog_coverage[interval[0] : interval[1]])
        ]
        evaluable_true_events += len(true_events)
        for true_start, true_end in true_events:
            matching_alarms = [
                event
                for event in positive_fog_alarms
                if max(true_start, int(event["start"]))
                < min(true_end, int(event["end"]))
            ]
            detected = bool(matching_alarms)
            matching_window_ids = {
                window_id
                for alarm in matching_alarms
                for window_id in alarm["window_ids"]
            }
            matching_groups = {
                group
                for alarm in matching_alarms
                for group in alarm["allocation_groups"]
            }
            detected_true_events += int(detected)
            event_audit.append(
                {
                    "subject": subject,
                    "fold": fold,
                    "seed": seed,
                    "record_id": record_id,
                    "event_type": "reference_fog",
                    "start_index": true_start,
                    "end_index_exclusive": true_end,
                    "positive_window_count": len(matching_window_ids),
                    "allocation_group_count": len(matching_groups),
                    "cross_allocation_group": len(matching_groups) > 1,
                    "detected": detected,
                }
            )

        positive_nonfog_rows = [
            row for row in record_rows if row["y_true"] == 0 and row["y_pred"] == 1
        ]
        nonfog_positive_clusters = merge_intervals(
            positive_nonfog_rows, maximum_gap_samples
        )
        merged_false_alarms = [
            event
            for event in nonfog_positive_clusters
            if int(event["positive_window_count"]) >= MINIMUM_POSITIVE_WINDOWS
        ]
        discarded_isolated = [
            event
            for event in nonfog_positive_clusters
            if int(event["positive_window_count"]) < MINIMUM_POSITIVE_WINDOWS
        ]
        false_alarm_events += len(merged_false_alarms)
        cross_group_false_alarm_events += sum(
            int(event["cross_allocation_group"]) for event in merged_false_alarms
        )
        for event in merged_false_alarms:
            event_audit.append(
                {
                    "subject": subject,
                    "fold": fold,
                    "seed": seed,
                    "record_id": record_id,
                    "event_type": "false_alarm",
                    "start_index": event["start"],
                    "end_index_exclusive": event["end"],
                    "positive_window_count": event["positive_window_count"],
                    "allocation_group_count": event["allocation_group_count"],
                    "cross_allocation_group": event["cross_allocation_group"],
                    "detected": "",
                }
            )
        for event in discarded_isolated:
            event_audit.append(
                {
                    "subject": subject,
                    "fold": fold,
                    "seed": seed,
                    "record_id": record_id,
                    "event_type": "discarded_isolated_nonfog_positive",
                    "start_index": event["start"],
                    "end_index_exclusive": event["end"],
                    "positive_window_count": event["positive_window_count"],
                    "allocation_group_count": event["allocation_group_count"],
                    "cross_allocation_group": event["cross_allocation_group"],
                    "detected": "",
                }
            )

    if evaluable_true_events <= 0 or evaluated_nonfog_samples <= 0:
        raise AssertionError(f"undefined event metric denominator: {subject}/fold{fold}/seed{seed}")
    nonfog_hours = evaluated_nonfog_samples / dataset.sampling_rate_hz / 3600.0
    return (
        {
            "subject": subject,
            "fold": fold,
            "seed": seed,
            "ap": ap,
            "event_sensitivity": detected_true_events / evaluable_true_events,
            "false_alarms_per_hour": false_alarm_events / nonfog_hours,
            "evaluable_true_events": evaluable_true_events,
            "detected_true_events": detected_true_events,
            "false_alarm_events": false_alarm_events,
            "cross_allocation_group_false_alarm_events": cross_group_false_alarm_events,
            "evaluated_nonfog_hours": nonfog_hours,
            "minimum_positive_windows_per_alarm": MINIMUM_POSITIVE_WINDOWS,
            "discarded_isolated_nonfog_positive_clusters": sum(
                row["event_type"] == "discarded_isolated_nonfog_positive"
                for row in event_audit
            ),
            "test_windows": len(parsed),
            "test_nonfog_windows": int(np.sum(labels == 0)),
            "test_fog_windows": int(np.sum(labels == 1)),
        },
        event_audit,
    )


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("cannot aggregate empty or non-finite values")
    return float(array.mean())


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    for path in (
        data_dir / "nbm_protocol.json",
        data_dir / "nbm_window_manifest.csv",
        experiment_root / "TRAINING_BARRIER.json",
        experiment_root / "DONE.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    barrier = read_json(experiment_root / "TRAINING_BARRIER.json")
    if barrier.get("status") != "sealed":
        raise AssertionError("training barrier is not sealed")
    dataset = DaphnetDataset.load(data_dir)
    if int(dataset.sampling_rate_hz) != 64:
        raise AssertionError(f"unexpected sampling rate: {dataset.sampling_rate_hz}")
    allocation_groups = load_allocation_groups(data_dir / "nbm_window_manifest.csv")

    fold_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = (
                    experiment_root
                    / "runs"
                    / subject
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                )
                verify_run_artifacts(run_dir)
                metrics, events = evaluate_run(
                    dataset,
                    subject,
                    fold,
                    seed,
                    read_csv(run_dir / "test_predictions.csv"),
                    allocation_groups,
                )
                stored = read_json(run_dir / "metrics.json")
                metrics["stored_ap_absolute_difference"] = abs(
                    metrics["ap"] - float(stored["pr_auc"])
                )
                if metrics["stored_ap_absolute_difference"] > 1e-12:
                    raise AssertionError(f"AP mismatch against sealed metric: {run_dir}")
                fold_rows.append(metrics)
                event_rows.extend(events)
    if len(fold_rows) != len(SUBJECTS) * len(FOLDS) * len(SEEDS):
        raise AssertionError(f"expected 120 runs, found {len(fold_rows)}")

    subject_seed_rows: list[dict[str, Any]] = []
    metric_names = ("ap", "event_sensitivity", "false_alarms_per_hour")
    for subject in SUBJECTS:
        for seed in SEEDS:
            selected = [
                row
                for row in fold_rows
                if row["subject"] == subject and row["seed"] == seed
            ]
            if len(selected) != 3:
                raise AssertionError(f"expected three folds: {subject}/seed{seed}")
            subject_seed_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    **{
                        metric: mean(row[metric] for row in selected)
                        for metric in metric_names
                    },
                }
            )

    unrounded_summary: list[dict[str, Any]] = []
    formatted_summary: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        selected = [row for row in subject_seed_rows if row["subject"] == subject]
        if len(selected) != 5:
            raise AssertionError(f"expected five seed summaries: {subject}")
        unrounded: dict[str, Any] = {"subject": subject}
        formatted: dict[str, Any] = {"subject": subject}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            metric_mean = float(values.mean())
            metric_sd = float(values.std(ddof=0))
            unrounded[f"{metric}_mean"] = metric_mean
            unrounded[f"{metric}_sd"] = metric_sd
            formatted[f"{metric}_mean"] = format_four_significant(metric_mean)
            formatted[f"{metric}_sd"] = format_four_significant(metric_sd)
        unrounded_summary.append(unrounded)
        formatted_summary.append(formatted)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "raw_tcn_fold_metrics_unrounded.csv", fold_rows)
    write_csv(output_dir / "raw_tcn_subject_seed_metrics_unrounded.csv", subject_seed_rows)
    write_csv(output_dir / "raw_tcn_subject_summary_unrounded.csv", unrounded_summary)
    write_csv(
        output_dir / "raw_tcn_subject_AP_EventSensitivity_FAh_mean_sd_4sig.csv",
        formatted_summary,
    )
    fah_summary = [
        {
            "subject": row["subject"],
            "false_alarms_per_hour_mean": f"{row['false_alarms_per_hour_mean']:.3f}",
            "false_alarms_per_hour_sd": f"{row['false_alarms_per_hour_sd']:.3f}",
        }
        for row in unrounded_summary
    ]
    macro_fah_by_seed = [
        mean(
            row["false_alarms_per_hour"]
            for row in subject_seed_rows
            if row["seed"] == seed
        )
        for seed in SEEDS
    ]
    fah_summary.append(
        {
            "subject": "MACRO",
            "false_alarms_per_hour_mean": f"{np.mean(macro_fah_by_seed):.3f}",
            "false_alarms_per_hour_sd": f"{np.std(macro_fah_by_seed, ddof=0):.3f}",
        }
    )
    write_csv(
        output_dir / "raw_tcn_subject_FAh_mean_sd_3dec.csv",
        fah_summary,
    )
    write_csv(output_dir / "raw_tcn_event_audit.csv", event_rows)
    metadata = {
        "schema": "private_raw_tcn_latest_event_metrics.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(experiment_root),
        "source_barrier_id": barrier.get("barrier_id"),
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "run_count": len(fold_rows),
        "ap_definition": "sklearn.metrics.average_precision_score on sealed test windows",
        "event_sensitivity_definition": (
            "number of evaluable contiguous reference FoG events overlapped by a merged "
            "alarm containing at least two positive pure-FoG test windows divided by "
            "evaluable reference events"
        ),
        "false_alarm_definition": (
            "FoG-positive predictions on pure Non-FoG test windows; within each record, "
            "2-s window supports that overlap or have a gap <=1 s are one false alarm; "
            "a merged cluster must contain at least two positive windows; isolated "
            "positive windows are discarded; records never merge; allocation-group "
            "boundaries are ignored"
        ),
        "minimum_positive_windows_per_alarm": MINIMUM_POSITIVE_WINDOWS,
        "false_alarm_denominator": (
            "union duration of evaluated test-window coverage intersected with valid true "
            "Non-FoG samples, converted to hours"
        ),
        "aggregation": (
            "unweighted mean over three folds within each subject/seed; unweighted mean "
            "and population SD (ddof=0) over five seed-level fold means"
        ),
        "rounding": "final subject CSV displays exactly four significant digits",
        "maximum_ap_recompute_difference": max(
            row["stored_ap_absolute_difference"] for row in fold_rows
        ),
        "cross_allocation_group_false_alarm_events": int(
            sum(row["cross_allocation_group_false_alarm_events"] for row in fold_rows)
        ),
        "output_sha256": {},
    }
    for name in (
        "raw_tcn_fold_metrics_unrounded.csv",
        "raw_tcn_subject_seed_metrics_unrounded.csv",
        "raw_tcn_subject_summary_unrounded.csv",
        "raw_tcn_subject_AP_EventSensitivity_FAh_mean_sd_4sig.csv",
        "raw_tcn_subject_FAh_mean_sd_3dec.csv",
        "raw_tcn_event_audit.csv",
    ):
        metadata["output_sha256"][name] = sha256_file(output_dir / name)
    atomic_json_dump(metadata, output_dir / "statistical_contract.json")
    print(
        f"COMPLETE runs={len(fold_rows)} subjects={len(formatted_summary)} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
