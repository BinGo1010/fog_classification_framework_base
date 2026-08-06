"""Build within-subject three-fold grouped CV manifests from processed_A5_50.

FoG events and clean Non-FoG blocks are indivisible groups.  Every group is
assigned to exactly one outer test fold and belongs to the outer-train side of
the other two folds.  Inner model selection must be performed only inside each
outer-train partition by the downstream A6 experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_a5_splits as base  # noqa: E402


N_SPLITS = 3
DATASET_ID = "daphnet_Group_3folds"
SOURCE_WINDOW_MANIFEST = "a5_50_window_manifest.csv"
SOURCE_EVENT_MANIFEST = "a5_50_fog_event_manifest.csv"
SOURCE_BLOCK_MANIFEST = "a5_50_clean_block_manifest.csv"
GROUP_MANIFEST = "group3_group_assignments.csv"
FOLD_GROUP_ROLES = "group3_fold_group_roles.csv"
WINDOW_MANIFEST = "group3_window_manifest.csv"
EVENT_MANIFEST = "group3_fog_event_manifest.csv"
BLOCK_MANIFEST = "group3_clean_block_manifest.csv"
SUMMARY = "group3_fold_summary.csv"
QUALITY = "group3_quality_report.json"
PROTOCOL = "group3_protocol.json"


def parse_args() -> argparse.Namespace:
    dataset_root = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root / "processed_A5_50")
    parser.add_argument("--output", type=Path, default=dataset_root / "processed_Group_3folds")
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    return parser.parse_args()


def as_int(value: Any) -> int:
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def clean_group_id(row: dict[str, Any]) -> str:
    block = str(row.get("clean_block_id", "")).strip()
    if not block:
        raise ValueError(f"clean window lacks clean_block_id: {row.get('window_id')}")
    return f"{row['subject_id']}|NONFOG|{block}"


def fog_group_id(row: dict[str, Any]) -> str:
    event = str(row.get("event_id", "")).strip()
    if not event:
        raise ValueError(f"FoG window lacks event_id: {row.get('window_id')}")
    return f"{row['subject_id']}|FOG|{row['record_id']}|E{as_int(event):03d}"


def row_group_id(row: dict[str, Any]) -> str:
    return fog_group_id(row) if str(row["class_label"]).upper() == "FOG" else clean_group_id(row)


def assign_balanced_folds(
    groups: Sequence[dict[str, Any]],
    n_splits: int,
) -> dict[str, int]:
    """Deterministic GroupKFold-style greedy balancing by window count."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if len(groups) < n_splits:
        raise ValueError(f"need at least {n_splits} groups, got {len(groups)}")
    ordered = sorted(
        groups,
        key=lambda row: (
            -as_int(row["window_count"]),
            -as_float(row["duration_sec"]),
            str(row["group_id"]),
        ),
    )
    window_totals = [0] * n_splits
    duration_totals = [0.0] * n_splits
    group_totals = [0] * n_splits
    assigned: dict[str, int] = {}
    for row in ordered:
        fold = min(
            range(n_splits),
            key=lambda index: (
                group_totals[index],
                window_totals[index],
                duration_totals[index],
                index,
            ),
        )
        assigned[str(row["group_id"])] = fold
        window_totals[fold] += as_int(row["window_count"])
        duration_totals[fold] += as_float(row["duration_sec"])
        group_totals[fold] += 1
    return assigned


def build_groups(
    window_rows: Sequence[dict[str, str]],
    event_rows: Sequence[dict[str, str]],
    block_rows: Sequence[dict[str, str]],
    n_splits: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[dict[str, str]]]]:
    windows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in window_rows:
        windows_by_group[row_group_id(row)].append(row)

    source_events = {
        (row["subject_id"], row["record_id"], as_int(row["event_id"])): row
        for row in event_rows
        if row.get("strict_rule_status") == "retained"
    }
    source_blocks = {row["clean_block_id"]: row for row in block_rows}
    groups: list[dict[str, Any]] = []
    for group_id, rows in sorted(windows_by_group.items()):
        first = rows[0]
        label = str(first["class_label"]).upper()
        records = {row["record_id"] for row in rows}
        if len(records) != 1:
            raise ValueError(f"group spans records: {group_id} -> {records}")
        starts = [as_int(row["start_index"]) for row in rows]
        ends = [as_int(row["end_index_exclusive"]) for row in rows]
        if label == "FOG":
            source = source_events[(first["subject_id"], first["record_id"], as_int(first["event_id"]))]
            duration = as_float(source["duration_sec"])
            source_role = source["a5_role"]
            source_split = source["a5_split"]
            group_type = "fog_event"
            source_group_id = f"{first['record_id']}:E{as_int(first['event_id']):03d}"
        else:
            source = source_blocks[first["clean_block_id"]]
            duration = (as_int(source["end_index_exclusive"]) - as_int(source["start_index"])) / base.FS
            source_role = source["a5_role"]
            source_split = source["a5_split"]
            group_type = "clean_block"
            source_group_id = first["clean_block_id"]
        groups.append(
            {
                "group_id": group_id,
                "subject_id": first["subject_id"],
                "subject_scope": first["subject_scope"],
                "class_label": label,
                "group_type": group_type,
                "source_group_id": source_group_id,
                "record_id": first["record_id"],
                "run_id": first["run_id"],
                "segment_id": as_int(first["segment_id"]),
                "start_index": min(starts),
                "end_index_exclusive": max(ends),
                "duration_sec": duration,
                "window_count": len(rows),
                "source_a5_role": source_role,
                "source_a5_split": source_split,
            }
        )

    assignments: dict[str, int] = {}
    by_subject_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_subject_class[(group["subject_id"], group["class_label"])].append(group)
    for (_, _), selected in sorted(by_subject_class.items()):
        if len(selected) < n_splits:
            # Clean-only controls have no FoG group at all, but every existing
            # class must still be foldable.
            raise ValueError(f"insufficient groups for {selected[0]['subject_id']} {selected[0]['class_label']}")
        assignments.update(assign_balanced_folds(selected, n_splits))
    for group in groups:
        group["assigned_test_fold"] = assignments[group["group_id"]]
    return groups, assignments, windows_by_group


def interval_distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    return base.interval_distance(
        as_int(a["start_index"]),
        as_int(a["end_index_exclusive"]),
        as_int(b["start_index"]),
        as_int(b["end_index_exclusive"]),
    )


def fold_group_roles(
    groups: Sequence[dict[str, Any]],
    n_splits: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], str]]:
    groups_by_subject_record: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        groups_by_subject_record[(group["subject_id"], group["record_id"])].append(group)
    role_lookup: dict[tuple[int, str], str] = {}
    rows: list[dict[str, Any]] = []
    for fold in range(n_splits):
        for group in groups:
            if as_int(group["assigned_test_fold"]) == fold:
                role = "outer_test"
                embargo_conflicts: list[str] = []
            else:
                tests = [
                    other
                    for other in groups_by_subject_record[(group["subject_id"], group["record_id"])]
                    if as_int(other["assigned_test_fold"]) == fold
                ]
                embargo_conflicts = [
                    str(other["group_id"])
                    for other in tests
                    if interval_distance(group, other) < base.INTER_SPLIT_EMBARGO
                ]
                role = "excluded_outer_train_embargo" if embargo_conflicts else "outer_train"
            role_lookup[(fold, str(group["group_id"]))] = role
            rows.append(
                {
                    "fold_id": fold,
                    "group_id": group["group_id"],
                    "subject_id": group["subject_id"],
                    "subject_scope": group["subject_scope"],
                    "class_label": group["class_label"],
                    "group_type": group["group_type"],
                    "source_group_id": group["source_group_id"],
                    "assigned_test_fold": group["assigned_test_fold"],
                    "outer_role": role,
                    "active_for_fold": role != "excluded_outer_train_embargo",
                    "embargo_conflict_group_ids": "|".join(embargo_conflicts),
                    "record_id": group["record_id"],
                    "run_id": group["run_id"],
                    "segment_id": group["segment_id"],
                    "window_count": group["window_count"],
                    "duration_sec": group["duration_sec"],
                }
            )
    return rows, role_lookup


def expanded_window_manifest(
    source_rows: Sequence[dict[str, str]],
    assignments: dict[str, int],
    role_lookup: dict[tuple[int, str], str],
    n_splits: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_index, row in enumerate(source_rows):
        group_id = row_group_id(row)
        label = str(row["class_label"]).upper()
        for fold in range(n_splits):
            outer_role = role_lookup[(fold, group_id)]
            active = outer_role != "excluded_outer_train_embargo"
            class_suffix = "fog" if label == "FOG" else "nonfog"
            a6_role = f"{outer_role}_{class_suffix}" if active else outer_role
            item: dict[str, Any] = {
                "fold_id": fold,
                "outer_role": outer_role,
                "a6_role": a6_role,
                "active_for_fold": active,
                "assigned_test_fold": assignments[group_id],
                "cv_group_id": group_id,
                "cv_group_type": "fog_event" if label == "FOG" else "clean_block",
                "source_window_row_index": source_index,
                "source_a5_role": row["a5_role"],
                "source_a5_split": row["a5_split"],
            }
            item.update(row)
            output.append(item)
    output.sort(
        key=lambda row: (
            as_int(row["fold_id"]),
            row["subject_id"],
            0 if row["outer_role"] == "outer_train" else 1 if row["outer_role"] == "outer_test" else 2,
            row["class_label"],
            row["record_id"],
            as_int(row["start_index"]),
        )
    )
    return output


def fold_summary(
    fold_rows: Sequence[dict[str, Any]],
    fold_group_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: Counter[tuple[str, int, str, str]] = Counter()
    durations: defaultdict[tuple[str, int, str, str], float] = defaultdict(float)
    for row in fold_group_rows:
        key = (row["subject_id"], as_int(row["fold_id"]), row["outer_role"], row["class_label"])
        groups[key] += 1
        durations[key] += as_float(row["duration_sec"])
    windows: Counter[tuple[str, int, str, str]] = Counter(
        (row["subject_id"], as_int(row["fold_id"]), row["outer_role"], row["class_label"])
        for row in fold_rows
    )
    output: list[dict[str, Any]] = []
    for key in sorted(set(groups) | set(windows)):
        subject, fold, outer_role, label = key
        first = next(row for row in fold_rows if row["subject_id"] == subject)
        output.append(
            {
                "subject_id": subject,
                "subject_scope": first["subject_scope"],
                "fold_id": fold,
                "outer_role": outer_role,
                "class_label": label,
                "group_count": groups[key],
                "window_count": windows[key],
                "duration_sec": durations[key],
            }
        )
    return output


def update_source_manifests(
    event_rows: Sequence[dict[str, str]],
    block_rows: Sequence[dict[str, str]],
    assignments: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    for row in event_rows:
        item: dict[str, Any] = dict(row)
        if row.get("strict_rule_status") == "retained":
            item["cv_group_id"] = fog_group_id(row)
            item["assigned_test_fold"] = assignments[item["cv_group_id"]]
        else:
            item["cv_group_id"] = ""
            item["assigned_test_fold"] = ""
        events.append(item)
    blocks: list[dict[str, Any]] = []
    for row in block_rows:
        item = dict(row)
        pseudo = {
            "subject_id": row["subject_id"],
            "clean_block_id": row["clean_block_id"],
        }
        item["cv_group_id"] = clean_group_id(pseudo)
        item["assigned_test_fold"] = assignments[item["cv_group_id"]]
        blocks.append(item)
    return events, blocks


def active_rows(rows: Iterable[dict[str, Any]], subject: str, fold: int, role: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["subject_id"] == subject
        and as_int(row["fold_id"]) == fold
        and row["outer_role"] == role
    ]


def leakage_audit(
    fold_rows: Sequence[dict[str, Any]],
    groups: Sequence[dict[str, Any]],
    n_splits: int,
) -> dict[str, Any]:
    subjects = sorted({row["subject_id"] for row in fold_rows})
    duplicate_window_ids = 0
    train_test_window_overlap = 0
    train_test_group_overlap = 0
    cross_split_interval_violations = 0
    examples: list[dict[str, Any]] = []
    empty_formal_partitions: list[str] = []
    for subject in subjects:
        scope = next(row["subject_scope"] for row in fold_rows if row["subject_id"] == subject)
        for fold in range(n_splits):
            selected = [row for row in fold_rows if row["subject_id"] == subject and as_int(row["fold_id"]) == fold]
            counts = Counter(row["window_id"] for row in selected)
            duplicate_window_ids += sum(count - 1 for count in counts.values() if count > 1)
            train = [row for row in selected if row["outer_role"] == "outer_train"]
            test = [row for row in selected if row["outer_role"] == "outer_test"]
            train_ids = {row["window_id"] for row in train}
            test_ids = {row["window_id"] for row in test}
            train_test_window_overlap += len(train_ids & test_ids)
            train_groups = {row["cv_group_id"] for row in train}
            test_groups = {row["cv_group_id"] for row in test}
            train_test_group_overlap += len(train_groups & test_groups)
            by_record_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in test:
                by_record_test[row["record_id"]].append(row)
            for row in train:
                for other in by_record_test.get(row["record_id"], []):
                    distance = base.interval_distance(
                        as_int(row["start_index"]),
                        as_int(row["end_index_exclusive"]),
                        as_int(other["start_index"]),
                        as_int(other["end_index_exclusive"]),
                    )
                    if distance < base.INTER_SPLIT_EMBARGO:
                        cross_split_interval_violations += 1
                        if len(examples) < 10:
                            examples.append(
                                {
                                    "subject_id": subject,
                                    "fold_id": fold,
                                    "train_window": row["window_id"],
                                    "test_window": other["window_id"],
                                    "distance_samples": distance,
                                }
                            )
            if scope == "formal_main7":
                for role in ("outer_train", "outer_test"):
                    for label in ("CLEAN_NONFOG", "FOG"):
                        if not any(row["outer_role"] == role and row["class_label"] == label for row in selected):
                            empty_formal_partitions.append(f"{subject}:fold{fold}:{role}:{label}")
    test_exposure = Counter(
        row["window_id"] for row in fold_rows if row["outer_role"] == "outer_test"
    )
    test_once_failures = sorted(window_id for window_id, count in test_exposure.items() if count != 1)
    group_test_exposure = Counter(
        row["group_id"]
        for row in groups
        for _ in [row["assigned_test_fold"]]
    )
    group_once_failures = sorted(group_id for group_id, count in group_test_exposure.items() if count != 1)
    passed = bool(
        duplicate_window_ids == 0
        and train_test_window_overlap == 0
        and train_test_group_overlap == 0
        and cross_split_interval_violations == 0
        and not empty_formal_partitions
        and not test_once_failures
        and not group_once_failures
    )
    return {
        "duplicate_window_ids_within_subject_fold": duplicate_window_ids,
        "train_test_window_overlap": train_test_window_overlap,
        "train_test_group_overlap": train_test_group_overlap,
        "cross_split_pairs_inside_5s_embargo": cross_split_interval_violations,
        "empty_formal_partitions": empty_formal_partitions,
        "windows_not_tested_exactly_once": test_once_failures[:50],
        "groups_not_assigned_exactly_once": group_once_failures[:50],
        "examples": examples,
        "pass": passed,
    }


def save_indices(
    output: Path,
    source_rows: Sequence[dict[str, str]],
    fold_rows: Sequence[dict[str, Any]],
    n_splits: int,
) -> None:
    index_dir = output / "split_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted({row["subject_id"] for row in source_rows})
    for subject in subjects:
        payload: dict[str, np.ndarray] = {
            "source_window_ids": np.asarray(
                [row["window_id"] for row in source_rows if row["subject_id"] == subject],
                dtype=str,
            )
        }
        for fold in range(n_splits):
            for role in ("outer_train", "outer_test", "excluded_outer_train_embargo"):
                selected = active_rows(fold_rows, subject, fold, role)
                payload[f"fold{fold}_{role}_source_rows"] = np.asarray(
                    [as_int(row["source_window_row_index"]) for row in selected], dtype=np.int64
                )
                for label, suffix in (("CLEAN_NONFOG", "nonfog"), ("FOG", "fog")):
                    payload[f"fold{fold}_{role}_{suffix}_source_rows"] = np.asarray(
                        [
                            as_int(row["source_window_row_index"])
                            for row in selected
                            if row["class_label"] == label
                        ],
                        dtype=np.int64,
                    )
        np.savez_compressed(index_dir / f"{subject}_group3_indices.npz", **payload)


def quality_report(
    source_rows: Sequence[dict[str, str]],
    fold_rows: Sequence[dict[str, Any]],
    groups: Sequence[dict[str, Any]],
    group_role_rows: Sequence[dict[str, Any]],
    n_splits: int,
) -> dict[str, Any]:
    leakage = leakage_audit(fold_rows, groups, n_splits)
    subjects: dict[str, Any] = {}
    for subject in sorted({row["subject_id"] for row in source_rows}):
        selected_groups = [group for group in groups if group["subject_id"] == subject]
        fold_details: dict[str, Any] = {}
        for fold in range(n_splits):
            selected = [row for row in fold_rows if row["subject_id"] == subject and as_int(row["fold_id"]) == fold]
            fold_details[str(fold)] = {
                role: {
                    label: sum(
                        1
                        for row in selected
                        if row["outer_role"] == role and row["class_label"] == label
                    )
                    for label in ("CLEAN_NONFOG", "FOG")
                }
                for role in ("outer_train", "outer_test", "excluded_outer_train_embargo")
            }
        subjects[subject] = {
            "subject_scope": source_rows[next(i for i, row in enumerate(source_rows) if row["subject_id"] == subject)]["subject_scope"],
            "clean_group_count": sum(group["class_label"] == "CLEAN_NONFOG" for group in selected_groups),
            "fog_group_count": sum(group["class_label"] == "FOG" for group in selected_groups),
            "folds": fold_details,
        }
    assignment_counts = Counter(as_int(group["assigned_test_fold"]) for group in groups)
    excluded_groups = sum(row["outer_role"] == "excluded_outer_train_embargo" for row in group_role_rows)
    overall_pass = bool(
        leakage["pass"]
        and len(fold_rows) == len(source_rows) * n_splits
        and set(assignment_counts) == set(range(n_splits))
    )
    return {
        "overall_pass": overall_pass,
        "source_window_count": len(source_rows),
        "expanded_fold_window_row_count": len(fold_rows),
        "n_splits": n_splits,
        "group_count": len(groups),
        "test_group_assignment_counts": dict(sorted(assignment_counts.items())),
        "embargo_excluded_train_group_fold_pairs": excluded_groups,
        "leakage_audit": leakage,
        "subjects": subjects,
    }


def copy_source_files(source: Path, build: Path) -> None:
    for name in (
        "manifest.csv",
        "fog_events.csv",
        "loso_folds.csv",
        "preprocessing_report.json",
        "a5_50_event_drop_summary.csv",
        "a5_50_protocol.json",
        "a5_50_quality_report.json",
        "README_A5_50.md",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, build / name)
    shutil.copytree(source / "records", build / "records")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    n_splits = int(args.folds)
    if n_splits != N_SPLITS:
        raise ValueError(f"this dataset contract requires exactly {N_SPLITS} folds")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    build = output.with_name(f"{output.name}.__building_{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)

    source_rows = base.read_csv(source / SOURCE_WINDOW_MANIFEST)
    event_rows = base.read_csv(source / SOURCE_EVENT_MANIFEST)
    block_rows = base.read_csv(source / SOURCE_BLOCK_MANIFEST)
    source_quality = json.loads((source / "a5_50_quality_report.json").read_text(encoding="utf-8"))
    if not source_quality.get("overall_pass"):
        raise RuntimeError("processed_A5_50 quality gate is not PASS")

    groups, assignments, _ = build_groups(source_rows, event_rows, block_rows, n_splits)
    group_role_rows, role_lookup = fold_group_roles(groups, n_splits)
    fold_rows = expanded_window_manifest(source_rows, assignments, role_lookup, n_splits)
    summary_rows = fold_summary(fold_rows, group_role_rows)
    updated_events, updated_blocks = update_source_manifests(event_rows, block_rows, assignments)
    quality = quality_report(source_rows, fold_rows, groups, group_role_rows, n_splits)
    if not quality["overall_pass"]:
        base.write_json(build / "FAILED_GROUP3_QUALITY_REPORT.json", quality)
        base.write_csv(build / GROUP_MANIFEST, groups)
        base.write_csv(build / FOLD_GROUP_ROLES, group_role_rows)
        raise RuntimeError(f"Group-3-fold quality gate failed; inspect {build}")

    copy_source_files(source, build)
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    schema["within_subject_group_3folds"] = {
        "source_dataset": "processed_A5_50",
        "window_manifest": WINDOW_MANIFEST,
        "group_manifest": GROUP_MANIFEST,
        "fold_group_roles": FOLD_GROUP_ROLES,
        "event_manifest": EVENT_MANIFEST,
        "clean_block_manifest": BLOCK_MANIFEST,
        "fold_summary": SUMMARY,
        "n_splits": n_splits,
        "group_unit_fog": "strict retained FoG event",
        "group_unit_nonfog": "clean Non-FoG block",
        "outer_fold_roles": ["outer_train", "outer_test"],
        "inner_validation_policy": "create only from outer_train in the downstream experiment",
    }
    base.write_json(build / "schema.json", schema)
    base.write_csv(build / GROUP_MANIFEST, groups)
    base.write_csv(build / FOLD_GROUP_ROLES, group_role_rows)
    base.write_csv(build / WINDOW_MANIFEST, fold_rows)
    base.write_csv(build / EVENT_MANIFEST, updated_events)
    base.write_csv(build / BLOCK_MANIFEST, updated_blocks)
    base.write_csv(build / SUMMARY, summary_rows)
    base.write_json(build / QUALITY, quality)
    save_indices(build, source_rows, fold_rows, n_splits)

    protocol = {
        "dataset_id": DATASET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_window_manifest": str(source / SOURCE_WINDOW_MANIFEST),
        "source_window_manifest_sha256": base.sha256(source / SOURCE_WINDOW_MANIFEST),
        "source_event_manifest_sha256": base.sha256(source / SOURCE_EVENT_MANIFEST),
        "source_clean_block_manifest_sha256": base.sha256(source / SOURCE_BLOCK_MANIFEST),
        "n_splits": n_splits,
        "scope": "within-subject only; subjects are never pooled across folds",
        "fog_group": "record_id + event_id; all windows from an event stay together",
        "nonfog_group": "clean_block_id; all windows from a clean block stay together",
        "assignment": "deterministic greedy GroupKFold-style balance by group count, then window count and duration",
        "outer_test_policy": "each group is outer_test exactly once and outer_train in the other folds unless removed by embargo",
        "embargo_seconds": base.INTER_SPLIT_EMBARGO / base.FS,
        "embargo_policy": "exclude an entire outer-train group when it lies within 5 s of an outer-test group in the same record",
        "inner_validation_policy": "not pre-split; use grouped inner CV only within outer_train",
        "nbm_policy": "NBM may use only outer-train clean Non-FoG; FoG never enters NBM fitting or early stopping",
        "classifier_policy": "classifier outer-train may use outer-train Non-FoG and FoG; evaluate only on the matching outer-test fold",
        "window_samples": base.WINDOW,
        "stride_samples": base.STRIDE,
        "sampling_rate_hz": base.FS,
    }
    base.write_json(build / PROTOCOL, protocol)
    (build / "README_GROUP_3FOLDS.md").write_text(
        "# Daphnet processed_Group_3folds\n\n"
        "This is a within-subject outer Group K-fold dataset with K=3, derived from processed_A5_50. "
        "FoG events and clean Non-FoG blocks are indivisible groups. Each group is tested in exactly one fold.\n\n"
        "Use `group3_window_manifest.csv` and filter `active_for_fold=True`. For each fold, train the "
        "NBM only with `outer_train` clean Non-FoG. Train the classifier with `outer_train` Non-FoG and "
        "FoG. Create any inner validation split only from outer-train groups. Never use outer-test for "
        "model selection, score selection, calibration, or threshold tuning.\n",
        encoding="utf-8",
    )
    build.replace(output)
    print(json.dumps({"output": str(output), "quality": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
