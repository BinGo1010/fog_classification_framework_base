from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_group_3folds as group3


def test_balanced_assignment_prioritizes_group_count() -> None:
    groups = [
        {
            "group_id": f"g{index}",
            "window_count": weight,
            "duration_sec": float(weight),
        }
        for index, weight in enumerate([40, 20, 15, 12, 10, 9, 7, 6, 4])
    ]
    assigned = group3.assign_balanced_folds(groups, 3)
    assert set(assigned.values()) == {0, 1, 2}
    assert Counter(assigned.values()) == {0: 3, 1: 3, 2: 3}
    assert assigned == group3.assign_balanced_folds(groups, 3)


def test_fold_roles_exclude_entire_nearby_train_group() -> None:
    groups = [
        {
            "group_id": "S01|FOG|record|E000",
            "subject_id": "S01",
            "subject_scope": "formal_main7",
            "class_label": "FOG",
            "group_type": "fog_event",
            "source_group_id": "record:E000",
            "assigned_test_fold": 0,
            "record_id": "record",
            "run_id": "R01",
            "segment_id": 0,
            "start_index": 0,
            "end_index_exclusive": 128,
            "window_count": 1,
            "duration_sec": 2.0,
        },
        {
            "group_id": "S01|FOG|record|E001",
            "subject_id": "S01",
            "subject_scope": "formal_main7",
            "class_label": "FOG",
            "group_type": "fog_event",
            "source_group_id": "record:E001",
            "assigned_test_fold": 1,
            "record_id": "record",
            "run_id": "R01",
            "segment_id": 0,
            "start_index": 256,
            "end_index_exclusive": 384,
            "window_count": 1,
            "duration_sec": 2.0,
        },
        {
            "group_id": "S01|FOG|record|E002",
            "subject_id": "S01",
            "subject_scope": "formal_main7",
            "class_label": "FOG",
            "group_type": "fog_event",
            "source_group_id": "record:E002",
            "assigned_test_fold": 2,
            "record_id": "record",
            "run_id": "R01",
            "segment_id": 0,
            "start_index": 2048,
            "end_index_exclusive": 2176,
            "window_count": 1,
            "duration_sec": 2.0,
        },
    ]
    _, lookup = group3.fold_group_roles(groups, 3)
    assert lookup[(0, groups[0]["group_id"])] == "outer_test"
    assert lookup[(0, groups[1]["group_id"])] == "excluded_outer_train_embargo"
    assert lookup[(0, groups[2]["group_id"])] == "outer_train"


def test_group_ids_keep_subjects_and_events_isolated() -> None:
    fog = {
        "subject_id": "S02",
        "record_id": "S02_seg000",
        "event_id": "3",
        "window_id": "fog-window",
        "class_label": "FOG",
    }
    clean = {
        "subject_id": "S02",
        "clean_block_id": "S02_seg000_cleanblock004",
        "window_id": "clean-window",
        "class_label": "CLEAN_NONFOG",
    }
    assert group3.row_group_id(fog) == "S02|FOG|S02_seg000|E003"
    assert group3.row_group_id(clean) == "S02|NONFOG|S02_seg000_cleanblock004"
