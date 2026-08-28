from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.recalculate_all_dataset_false_alarms_gap1s import recalculate_run
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_worker


def row(record: str, start: int, end: int, prediction: int) -> dict[str, str]:
    return {
        "record_id": record,
        "start_index": str(start),
        "end_index_exclusive": str(end),
        "role_code": "0",
        "y_true": "0",
        "y_pred": str(prediction),
    }


def test_gap1s_merges_only_within_record_and_counts_single_positive() -> None:
    records = [
        SimpleNamespace(
            record_id="A",
            y=np.zeros(512, dtype=np.int8),
            valid=np.ones(512, dtype=bool),
        ),
        SimpleNamespace(
            record_id="B",
            y=np.zeros(256, dtype=np.int8),
            valid=np.ones(256, dtype=bool),
        ),
    ]
    dataset = SimpleNamespace(records=records, sampling_rate_hz=64)
    predictions = [
        row("A", 0, 128, 1),
        row("A", 64, 192, 1),  # decision separation = 1 s: same alarm
        row("A", 192, 320, 1),  # decision separation = 2 s: new alarm
        row("B", 0, 128, 1),  # different record: always new alarm
    ]
    result = recalculate_run(dataset, predictions, merge_gap_seconds=1.0)
    assert result["minimum_positive_decisions"] == 1
    assert result["false_positive_decisions"] == 4
    assert result["false_alarm_events"] == 3
    assert result["same_record_only"] is True


def test_gap_greater_than_1s_starts_new_alarm() -> None:
    record_a = SimpleNamespace(
        record_id="A",
        y=np.zeros(512, dtype=np.int8),
        valid=np.ones(512, dtype=bool),
    )
    dataset = SimpleNamespace(records=[record_a], sampling_rate_hz=64)
    predictions = [row("A", 0, 128, 1), row("A", 128, 256, 1)]
    result = recalculate_run(dataset, predictions, merge_gap_seconds=1.0)
    assert result["false_alarm_events"] == 2


def test_minimum_two_discards_isolated_positive_decisions() -> None:
    record_a = SimpleNamespace(
        record_id="A",
        y=np.zeros(640, dtype=np.int8),
        valid=np.ones(640, dtype=bool),
    )
    dataset = SimpleNamespace(records=[record_a], sampling_rate_hz=64)
    predictions = [
        row("A", 0, 128, 1),
        row("A", 64, 192, 1),  # pair: one alarm
        row("A", 256, 384, 1),  # isolated: discarded
    ]
    result = recalculate_run(
        dataset,
        predictions,
        merge_gap_seconds=1.0,
        minimum_positive_decisions=2,
    )
    assert result["false_positive_decisions"] == 3
    assert result["false_alarm_events"] == 1
    assert result["minimum_positive_decisions"] == 2


def test_future_worker_uses_the_same_gap1s_false_alarm_definition() -> None:
    records = [
        SimpleNamespace(
            record_id="A",
            y=np.zeros(512, dtype=np.int8),
            valid=np.ones(512, dtype=bool),
        ),
        SimpleNamespace(
            record_id="B",
            y=np.zeros(256, dtype=np.int8),
            valid=np.ones(256, dtype=bool),
        ),
    ]
    dataset = SimpleNamespace(records=records, sampling_rate_hz=64)
    rows = raw_worker.RoleRows(
        subject_id=np.asarray(["P01"] * 4),
        record_index=np.asarray([0, 0, 0, 1], dtype=np.int32),
        record_id=np.asarray(["A", "A", "A", "B"]),
        start=np.asarray([0, 64, 192, 0], dtype=np.int32),
        end=np.asarray([128, 192, 320, 128], dtype=np.int32),
        role=np.zeros(4, dtype=np.int8),
        label=np.zeros(4, dtype=np.int8),
        window_id=np.asarray(["a", "b", "c", "d"]),
    )
    result = raw_worker.nonfog_false_alarm_metrics(
        dataset, rows, np.ones(4, dtype=np.int8), merge_gap_seconds=1.0
    )
    assert result["false_positive_decisions"] == 4
    assert result["false_alarm_events"] == 3
    assert result["false_alarm_minimum_positive_decisions"] == 1
