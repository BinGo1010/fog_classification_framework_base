from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_group_3folds_3inner as inner3


def test_enrich_outer_group_roles_restores_intervals() -> None:
    roles = [{"group_id": "g0", "outer_role": "outer_train"}]
    groups = [{"group_id": "g0", "start_index": "64", "end_index_exclusive": "192"}]
    enriched = inner3.enrich_outer_group_roles(roles, groups)
    assert enriched[0]["start_index"] == "64"
    assert enriched[0]["end_index_exclusive"] == "192"


def test_choose_earlystop_groups_is_deterministic_and_group_level() -> None:
    groups = [
        {"group_id": f"g{i}", "window_count": weight}
        for i, weight in enumerate([40, 30, 20, 10, 5, 5])
    ]
    selected = inner3.choose_earlystop_groups(groups, 0.15)
    assert selected
    assert len(selected) < len(groups)
    assert selected == inner3.choose_earlystop_groups(list(reversed(groups)), 0.15)
    selected_windows = sum(row["window_count"] for row in groups if row["group_id"] in selected)
    assert abs(selected_windows - 0.15 * 110) <= 5


def test_assign_inner_folds_balances_each_class_independently() -> None:
    rows = []
    for label in ("CLEAN_NONFOG", "FOG"):
        for i in range(7):
            rows.append(
                {
                    "fold_id": "0",
                    "outer_role": "outer_train",
                    "subject_id": "S01",
                    "subject_scope": "formal_main7",
                    "class_label": label,
                    "group_id": f"{label}-{i}",
                    "window_count": str(20 - i),
                    "duration_sec": str(20 - i),
                }
            )
    assignments, lookup = inner3.assign_inner_folds(rows)
    for label in ("CLEAN_NONFOG", "FOG"):
        counts = [
            sum(row["class_label"] == label and row["assigned_inner_fold"] == fold for row in assignments)
            for fold in range(3)
        ]
        assert max(counts) - min(counts) <= 1
    assert len(lookup) == 14


def test_inner_embargo_excludes_entire_candidate_group() -> None:
    rows = [
        {
            "fold_id": "0",
            "outer_role": "outer_train",
            "group_id": "holdout",
            "subject_id": "S01",
            "subject_scope": "formal_main7",
            "class_label": "FOG",
            "group_type": "fog_event",
            "source_group_id": "e0",
            "assigned_test_fold": "1",
            "record_id": "rec",
            "run_id": "R1",
            "segment_id": "0",
            "start_index": "0",
            "end_index_exclusive": "128",
            "window_count": "1",
            "duration_sec": "2",
            "embargo_conflict_group_ids": "",
        },
        {
            "fold_id": "0",
            "outer_role": "outer_train",
            "group_id": "near-clean",
            "subject_id": "S01",
            "subject_scope": "formal_main7",
            "class_label": "CLEAN_NONFOG",
            "group_type": "clean_block",
            "source_group_id": "c0",
            "assigned_test_fold": "2",
            "record_id": "rec",
            "run_id": "R1",
            "segment_id": "0",
            "start_index": "256",
            "end_index_exclusive": "384",
            "window_count": "1",
            "duration_sec": "2",
            "embargo_conflict_group_ids": "",
        },
    ]
    for i in range(5):
        rows.append(
            {
                **rows[1],
                "group_id": f"far-clean-{i}",
                "source_group_id": f"c{i + 1}",
                "record_id": f"rec-far-{i}",
                "window_count": "10",
            }
        )
    assignments = {(0, "holdout"): 0, (0, "near-clean"): 1}
    assignments.update({(0, f"far-clean-{i}"): 1 + i % 2 for i in range(5)})
    roles, _ = inner3.build_pipeline_group_roles(rows, assignments)
    near_pipeline_a = next(
        row for row in roles if row["group_id"] == "near-clean" and row["pipeline_id"] == "A"
    )
    assert near_pipeline_a["inner_role"] == "excluded_inner_train_embargo"
    assert near_pipeline_a["active_for_pipeline"] is False
