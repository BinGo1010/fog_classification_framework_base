from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_nbm_splits as splits


def make_group(
    group_id: str,
    window_count: int,
    *,
    start_index: int = 0,
) -> splits.AllocationGroup:
    return splits.AllocationGroup(
        group_id=group_id,
        group_kind="clean_nonfog_block",
        class_label=splits.CLASS_NONFOG,
        subject_id="S01",
        record_id="S01_seg000",
        run_id="R01",
        segment_id=0,
        start_index=start_index,
        end_index_exclusive=(
            start_index + splits.WINDOW + (window_count - 1) * splits.STRIDE
        ),
        window_ids=tuple(f"{group_id}:w{index}" for index in range(window_count)),
    )


def make_connector(
    connector_id: str,
    left_group_id: str,
    right_group_id: str,
    *,
    start_index: int = 64,
) -> splits.CleanConnector:
    return splits.CleanConnector(
        connector_id=connector_id,
        window_id=f"{connector_id}:window",
        subject_id="S01",
        record_id="S01_seg000",
        run_id="R01",
        segment_id=0,
        start_index=start_index,
        end_index_exclusive=start_index + splits.WINDOW,
        left_group_id=left_group_id,
        right_group_id=right_group_id,
    )


def make_allocated_group(
    group_id: str,
    window_count: int,
    *,
    subject_id: str,
    class_label: str,
    permanent_partition: str,
    assigned_fold: int | None,
) -> splits.AllocationGroup:
    return splits.AllocationGroup(
        group_id=group_id,
        group_kind=(
            "fog_event_cluster"
            if class_label == splits.CLASS_FOG
            else "clean_nonfog_block"
        ),
        class_label=class_label,
        subject_id=subject_id,
        record_id=f"{subject_id}_seg000",
        run_id="R01",
        segment_id=0,
        start_index=0,
        end_index_exclusive=splits.WINDOW + (window_count - 1) * splits.STRIDE,
        window_ids=tuple(f"{group_id}:w{index}" for index in range(window_count)),
        permanent_partition=permanent_partition,
        assigned_development_fold=assigned_fold,
    )


def overlap_row(
    window_id: str,
    start_index: int,
    role: str,
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "subject_id": "S01",
        "record_id": "S01_seg000",
        "source_file": "S01R01.txt",
        "outer_fold_id": 0,
        "start_index": start_index,
        "end_index_exclusive": start_index + splits.WINDOW,
        "source_start_row": start_index,
        "source_end_row_exclusive": start_index + splits.WINDOW,
        "final_role": role,
    }


def test_purity_requires_all_128_labels_to_agree() -> None:
    assert splits.purity_from_fog_samples(0) == splits.PURE_NONFOG
    assert all(
        splits.purity_from_fog_samples(fog_samples) == splits.MIXED
        for fog_samples in range(1, splits.WINDOW)
    )
    assert splits.purity_from_fog_samples(splits.WINDOW) == splits.PURE_FOG

    with pytest.raises(ValueError):
        splits.purity_from_fog_samples(-1)
    with pytest.raises(ValueError):
        splits.purity_from_fog_samples(splits.WINDOW + 1)


def test_clean_partition_has_at_most_60_second_cores_and_one_connector_per_cut() -> None:
    starts = [index * splits.STRIDE for index in range(120)]

    blocks, connectors = splits.partition_clean_start_run(starts)

    assert len(blocks) > 1
    assert len(connectors) == len(blocks) - 1
    covered = [start for block in blocks for start in block] + list(connectors)
    assert sorted(covered) == starts
    assert len({start for block in blocks for start in block} | set(connectors)) == len(starts)

    for block in blocks:
        assert block
        assert all(
            right - left == splits.STRIDE
            for left, right in zip(block, block[1:])
        )
        assert block[-1] + splits.WINDOW - block[0] <= splits.MAX_CLEAN_BLOCK_SAMPLES

    for left, connector, right in zip(blocks, connectors, blocks[1:]):
        assert left[-1] + splits.STRIDE == connector
        assert connector + splits.STRIDE == right[0]
        assert left[-1] + splits.WINDOW == right[0]
        assert connector < left[-1] + splits.WINDOW
        assert connector + splits.WINDOW > right[0]


def test_connector_same_role_is_counted_and_cross_role_is_excluded() -> None:
    left = make_group("left", 2)
    right = make_group("right", 3, start_index=192)
    connector = make_connector("connector", left.group_id, right.group_id, start_index=128)
    labels = ("A", "B")

    same_role = splits.effective_label_counts(
        [left, right],
        [connector],
        {left.group_id: "A", right.group_id: "A"},
        labels,
    )
    cross_role = splits.effective_label_counts(
        [left, right],
        [connector],
        {left.group_id: "A", right.group_id: "B"},
        labels,
    )

    assert same_role == {"A": 6, "B": 0}
    assert cross_role == {"A": 2, "B": 3}


def test_connector_aware_optimizer_is_deterministic_under_reversed_input() -> None:
    groups = [
        make_group("g0", 4),
        make_group("g1", 3),
        make_group("g2", 2),
        make_group("g3", 1),
    ]
    connectors = [
        make_connector("c01", "g0", "g1"),
        make_connector("c12", "g1", "g2"),
        make_connector("c23", "g2", "g3"),
    ]
    targets = {"A": 6.0, "B": 6.0}
    minimums = {"A": 1, "B": 1}

    forward = splits.optimize_group_labels(groups, connectors, targets, minimums)
    reversed_input = splits.optimize_group_labels(
        list(reversed(groups)),
        list(reversed(connectors)),
        targets,
        minimums,
    )

    assert forward == reversed_input
    assignments, counts, score = forward
    assert set(assignments) == {group.group_id for group in groups}
    assert sum(counts.values()) in {10, 11, 12, 13}
    assert score >= 0.0


def test_cross_role_overlap_detects_one_second_overlap_but_not_touching_windows() -> None:
    overlapping = splits.cross_role_overlap_audit(
        [
            overlap_row("left", 0, splits.NBM_TRAIN_CLEAN),
            overlap_row("right", 64, splits.NBM_EARLYSTOP_CLEAN),
        ]
    )
    touching = splits.cross_role_overlap_audit(
        [
            overlap_row("left", 0, splits.NBM_TRAIN_CLEAN),
            overlap_row("right", 128, splits.NBM_EARLYSTOP_CLEAN),
        ]
    )

    assert overlapping["pass"] is False
    assert overlapping["cross_role_raw_overlap_pair_count"] == 1
    assert touching["pass"] is True
    assert touching["cross_role_raw_overlap_pair_count"] == 0


def test_optimizer_recovers_exact_375_window_clean_targets() -> None:
    targets = {
        splits.PERMANENT_TEST_NONFOG: 75.0,
        splits.EXTERNAL_VALIDATION_NONFOG: 100.0,
        splits.NBM_TRAIN_CLEAN: 96.0,
        splits.NBM_EARLYSTOP_CLEAN: 24.0,
        splits.CLASSIFIER_TRAIN_CLEAN: 80.0,
    }
    groups = [
        make_group(f"exact_{label}", int(target))
        for label, target in targets.items()
    ]

    assignments, counts, score = splits.optimize_group_labels(
        groups,
        [],
        targets,
        {label: 1 for label in targets},
    )

    assert counts == {label: int(target) for label, target in targets.items()}
    assert score == pytest.approx(0.0)
    assert set(assignments.values()) == set(targets)


def test_full_inventory_target_fractions_reconcile_by_class() -> None:
    clean_roles = (
        splits.PERMANENT_TEST_NONFOG,
        splits.EXTERNAL_VALIDATION_NONFOG,
        splits.NBM_TRAIN_CLEAN,
        splits.NBM_EARLYSTOP_CLEAN,
        splits.CLASSIFIER_TRAIN_CLEAN,
    )
    fog_roles = (
        splits.PERMANENT_TEST_FOG,
        splits.EXTERNAL_VALIDATION_FOG,
        splits.CLASSIFIER_TRAIN_FOG,
    )

    assert sum(splits.FULL_INVENTORY_TARGETS[role] for role in clean_roles) == pytest.approx(1.0)
    assert sum(splits.FULL_INVENTORY_TARGETS[role] for role in fog_roles) == pytest.approx(1.0)
    assert splits.WINDOW == 2 * splits.FS
    assert splits.STRIDE == splits.FS


def test_subject_fold_alignment_improves_aggregate_without_changing_groups() -> None:
    fog_validation_counts = {
        "S01": (17, 17, 17),
        "S02": (37, 37, 36),
        "S03": (54, 53, 53),
        "S05": (93, 93, 92),
        "S06": (39, 27, 25),
        "S07": (11, 11, 10),
        "S08": (47, 47, 46),
        "S09": (59, 58, 58),
    }
    groups: list[splits.AllocationGroup] = []
    for subject_id, counts in fog_validation_counts.items():
        for fold, count in enumerate(counts):
            groups.append(
                make_allocated_group(
                    f"{subject_id}_fog_fold{fold}",
                    count,
                    subject_id=subject_id,
                    class_label=splits.CLASS_FOG,
                    permanent_partition="development",
                    assigned_fold=fold,
                )
            )
    groups.append(
        make_allocated_group(
            "S01_fog_test",
            264,
            subject_id="S01",
            class_label=splits.CLASS_FOG,
            permanent_partition="permanent_test",
            assigned_fold=None,
        )
    )
    clean = make_allocated_group(
        "S06_clean_fold0",
        5,
        subject_id="S06",
        class_label=splits.CLASS_NONFOG,
        permanent_partition="development",
        assigned_fold=0,
    )
    groups.append(clean)
    clean_roles = {
        (1, clean.group_id): splits.NBM_TRAIN_CLEAN,
        (2, clean.group_id): splits.CLASSIFIER_TRAIN_CLEAN,
    }

    realigned_roles, rows, quality = splits.align_subject_outer_fold_labels(
        groups, clean_roles, []
    )

    assert quality["pass"] is True
    assert quality["aggregate_fog_validation_counts_before"] == [357, 343, 337]
    assert quality["aggregate_fog_validation_counts_after"] == [343, 343, 351]
    assert quality["aggregate_fog_classifier_train_counts_after"] == [694, 694, 686]
    assert quality["subjects_relabelled"] == ["S06"]
    assert quality["max_absolute_error_percentage_points_after"] < quality[
        "max_absolute_error_percentage_points_before"
    ]
    assert clean.assigned_development_fold == 2
    assert realigned_roles == {
        (1, clean.group_id): splits.NBM_TRAIN_CLEAN,
        (0, clean.group_id): splits.CLASSIFIER_TRAIN_CLEAN,
    }
    s06 = next(row for row in rows if row["subject_id"] == "S06")
    assert (
        s06["new_fold0_uses_old_fold"],
        s06["new_fold1_uses_old_fold"],
        s06["new_fold2_uses_old_fold"],
    ) == (2, 1, 0)
