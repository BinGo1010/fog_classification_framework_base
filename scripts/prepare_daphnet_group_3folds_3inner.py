"""Add concrete grouped inner three-fold pipelines to processed_Group_3folds.

For every subject and outer fold, active outer-train FoG events and clean
Non-FoG blocks are assigned to one of three inner holdout folds.  Pipeline A/B/C
holds out inner fold 0/1/2 respectively, trains its NBM only on eligible clean
Non-FoG groups from the other folds, and predicts the matching outer-test set.

Classifier features are cross-fitted: every active outer-train group receives
residual features from the one inner NBM for which that group was held out.
Each pipeline trains its classifier on OOF features from the other inner folds,
validates on its own held-out OOF features, predicts outer-test, and the three
outer-test probabilities can then be averaged.
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
import prepare_daphnet_group_3folds as outer3  # noqa: E402


N_INNER_SPLITS = 3
EARLYSTOP_FRACTION = 0.15
PIPELINE_NAMES = {0: "A", 1: "B", 2: "C"}
DATASET_ID = "daphnet_Group_3folds_3inner"
SOURCE_WINDOW_MANIFEST = "group3_window_manifest.csv"
SOURCE_GROUP_ROLES = "group3_fold_group_roles.csv"
SOURCE_GROUP_ASSIGNMENTS = "group3_group_assignments.csv"
INNER_ASSIGNMENTS = "inner3_group_assignments.csv"
INNER_GROUP_ROLES = "inner3_pipeline_group_roles.csv"
INNER_WINDOW_MANIFEST = "inner3_pipeline_window_manifest.csv"
INNER_SUMMARY = "inner3_pipeline_summary.csv"
INNER_QUALITY = "inner3_quality_report.json"
INNER_PROTOCOL = "inner3_protocol.json"


def parse_args() -> argparse.Namespace:
    dataset_root = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root / "processed_Group_3folds")
    parser.add_argument("--output", type=Path, default=dataset_root / "processed_Group_3folds_3inner")
    return parser.parse_args()


def as_int(value: Any) -> int:
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def enrich_outer_group_roles(
    role_rows: Sequence[dict[str, str]],
    group_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Restore interval fields omitted from group3_fold_group_roles.csv."""
    groups = {row["group_id"]: row for row in group_rows}
    output: list[dict[str, str]] = []
    for row in role_rows:
        source = groups[row["group_id"]]
        item = dict(row)
        item["start_index"] = source["start_index"]
        item["end_index_exclusive"] = source["end_index_exclusive"]
        output.append(item)
    return output


def choose_earlystop_groups(
    groups: Sequence[dict[str, Any]],
    fraction: float = EARLYSTOP_FRACTION,
) -> set[str]:
    """Choose a deterministic group-level subset closest to the window target."""
    if len(groups) < 2:
        raise ValueError(f"need at least two clean groups for train/early-stop, got {len(groups)}")
    ordered = sorted(groups, key=lambda row: str(row["group_id"]))
    total_windows = sum(as_int(row["window_count"]) for row in ordered)
    target_windows = total_windows * fraction
    target_groups = max(1, round(len(ordered) * fraction))

    # DP over attainable window sums.  For an equal sum, retain the subset whose
    # group count is closer to the target, then the lexicographically first IDs.
    states: dict[int, tuple[str, ...]] = {0: ()}
    for row in ordered:
        group_id = str(row["group_id"])
        weight = as_int(row["window_count"])
        updates = dict(states)
        for subtotal, subset in states.items():
            candidate = subset + (group_id,)
            new_total = subtotal + weight
            current = updates.get(new_total)
            if current is None or (
                abs(len(candidate) - target_groups), candidate
            ) < (
                abs(len(current) - target_groups), current
            ):
                updates[new_total] = candidate
        states = updates

    candidates = [
        (subtotal, subset)
        for subtotal, subset in states.items()
        if subset and len(subset) < len(ordered)
    ]
    _, selected = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target_windows),
            abs(len(item[1]) - target_groups),
            item[1],
        ),
    )
    return set(selected)


def assign_inner_folds(
    outer_group_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], int]]:
    """Assign active outer-train groups within each subject/outer-fold/class."""
    active = [row for row in outer_group_rows if row["outer_role"] == "outer_train"]
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in active:
        grouped[(row["subject_id"], as_int(row["fold_id"]), row["class_label"])].append(row)

    assignments: dict[tuple[int, str], int] = {}
    output: list[dict[str, Any]] = []
    for (subject, outer_fold, label), groups in sorted(grouped.items()):
        assigned = outer3.assign_balanced_folds(
            [
                {
                    "group_id": row["group_id"],
                    "window_count": as_int(row["window_count"]),
                    "duration_sec": as_float(row["duration_sec"]),
                }
                for row in groups
            ],
            N_INNER_SPLITS,
        )
        for row in groups:
            inner_fold = assigned[row["group_id"]]
            assignments[(outer_fold, row["group_id"])] = inner_fold
            item: dict[str, Any] = {
                "outer_fold_id": outer_fold,
                "assigned_inner_fold": inner_fold,
                "oof_feature_generator_pipeline": PIPELINE_NAMES[inner_fold],
            }
            item.update(row)
            output.append(item)
    output.sort(
        key=lambda row: (
            row["subject_id"],
            as_int(row["outer_fold_id"]),
            row["class_label"],
            as_int(row["assigned_inner_fold"]),
            row["group_id"],
        )
    )
    return output, assignments


def interval_distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    return base.interval_distance(
        as_int(a["start_index"]),
        as_int(a["end_index_exclusive"]),
        as_int(b["start_index"]),
        as_int(b["end_index_exclusive"]),
    )


def build_pipeline_group_roles(
    outer_group_rows: Sequence[dict[str, str]],
    assignments: dict[tuple[int, str], int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], set[str]]]:
    """Expand every outer fold into pipelines A/B/C with concrete roles."""
    by_subject_outer: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in outer_group_rows:
        by_subject_outer[(row["subject_id"], as_int(row["fold_id"]))].append(row)

    earlystop_lookup: dict[tuple[str, int, int], set[str]] = {}
    inner_embargo_lookup: dict[tuple[str, int, int], dict[str, list[str]]] = {}
    for (subject, outer_fold), rows in sorted(by_subject_outer.items()):
        active_train = [row for row in rows if row["outer_role"] == "outer_train"]
        for pipeline in range(N_INNER_SPLITS):
            holdout = [
                row
                for row in active_train
                if assignments[(outer_fold, row["group_id"])] == pipeline
            ]
            conflicts: dict[str, list[str]] = defaultdict(list)
            for candidate in active_train:
                if assignments[(outer_fold, candidate["group_id"])] == pipeline:
                    continue
                for held in holdout:
                    if candidate["record_id"] != held["record_id"]:
                        continue
                    if interval_distance(candidate, held) < base.INTER_SPLIT_EMBARGO:
                        conflicts[candidate["group_id"]].append(held["group_id"])
            inner_embargo_lookup[(subject, outer_fold, pipeline)] = dict(conflicts)
            clean_fit = [
                row
                for row in active_train
                if row["class_label"] == "CLEAN_NONFOG"
                and assignments[(outer_fold, row["group_id"])] != pipeline
                and row["group_id"] not in conflicts
            ]
            earlystop_lookup[(subject, outer_fold, pipeline)] = choose_earlystop_groups(clean_fit)

    output: list[dict[str, Any]] = []
    for (subject, outer_fold), rows in sorted(by_subject_outer.items()):
        for pipeline in range(N_INNER_SPLITS):
            earlystop = earlystop_lookup[(subject, outer_fold, pipeline)]
            conflicts = inner_embargo_lookup[(subject, outer_fold, pipeline)]
            for row in rows:
                outer_role = row["outer_role"]
                label = row["class_label"]
                assigned = assignments.get((outer_fold, row["group_id"]))
                if outer_role == "excluded_outer_train_embargo":
                    inner_role = "excluded_outer_train_embargo"
                    nbm_role = "excluded"
                    classifier_role = "excluded"
                    active = False
                    generator = ""
                    conflict_ids = row.get("embargo_conflict_group_ids", "")
                elif outer_role == "outer_test":
                    inner_role = "outer_test"
                    nbm_role = "outer_test_transform"
                    classifier_role = "classifier_outer_test_predict"
                    active = True
                    generator = PIPELINE_NAMES[pipeline]
                    conflict_ids = ""
                elif assigned == pipeline:
                    inner_role = "inner_holdout_oof"
                    nbm_role = "oof_transform_only_nonfog" if label == "CLEAN_NONFOG" else "oof_transform_only_fog"
                    classifier_role = "classifier_validation_oof"
                    active = True
                    generator = PIPELINE_NAMES[pipeline]
                    conflict_ids = ""
                elif row["group_id"] in conflicts:
                    inner_role = "excluded_inner_train_embargo"
                    nbm_role = "excluded"
                    classifier_role = "excluded"
                    active = False
                    generator = PIPELINE_NAMES[assigned] if assigned is not None else ""
                    conflict_ids = "|".join(sorted(conflicts[row["group_id"]]))
                else:
                    inner_role = "inner_fit"
                    if label == "CLEAN_NONFOG":
                        nbm_role = (
                            "nbm_earlystop_clean"
                            if row["group_id"] in earlystop
                            else "nbm_weight_train_clean"
                        )
                    else:
                        nbm_role = "nbm_not_used_fog"
                    classifier_role = "classifier_train_oof"
                    active = True
                    generator = PIPELINE_NAMES[assigned]
                    conflict_ids = ""
                item: dict[str, Any] = {
                    "outer_fold_id": outer_fold,
                    "pipeline_index": pipeline,
                    "pipeline_id": PIPELINE_NAMES[pipeline],
                    "assigned_inner_fold": "" if assigned is None else assigned,
                    "inner_role": inner_role,
                    "nbm_role": nbm_role,
                    "classifier_role": classifier_role,
                    "active_for_pipeline": active,
                    "oof_feature_generator_pipeline": generator,
                    "inner_embargo_conflict_group_ids": conflict_ids,
                }
                item.update(row)
                output.append(item)
    output.sort(
        key=lambda row: (
            row["subject_id"],
            as_int(row["outer_fold_id"]),
            as_int(row["pipeline_index"]),
            row["outer_role"],
            row["class_label"],
            row["group_id"],
        )
    )
    return output, earlystop_lookup


def expand_window_manifest(
    source_rows: Sequence[dict[str, str]],
    group_roles: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    role_lookup = {
        (as_int(row["outer_fold_id"]), as_int(row["pipeline_index"]), row["group_id"]): row
        for row in group_roles
    }
    output: list[dict[str, Any]] = []
    for row in source_rows:
        outer_fold = as_int(row["fold_id"])
        group_id = row["cv_group_id"]
        for pipeline in range(N_INNER_SPLITS):
            role = role_lookup[(outer_fold, pipeline, group_id)]
            item: dict[str, Any] = {
                "outer_fold_id": outer_fold,
                "pipeline_index": pipeline,
                "pipeline_id": PIPELINE_NAMES[pipeline],
                "assigned_inner_fold": role["assigned_inner_fold"],
                "inner_role": role["inner_role"],
                "nbm_role": role["nbm_role"],
                "classifier_role": role["classifier_role"],
                "active_for_pipeline": role["active_for_pipeline"],
                "oof_feature_generator_pipeline": role["oof_feature_generator_pipeline"],
                "inner_embargo_conflict_group_ids": role["inner_embargo_conflict_group_ids"],
            }
            item.update(row)
            output.append(item)
    output.sort(
        key=lambda row: (
            row["subject_id"],
            as_int(row["outer_fold_id"]),
            as_int(row["pipeline_index"]),
            row["inner_role"],
            row["class_label"],
            row["record_id"],
            as_int(row["start_index"]),
        )
    )
    return output


def build_summary(
    window_rows: Sequence[dict[str, Any]],
    group_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    window_counts: Counter[tuple[Any, ...]] = Counter()
    group_counts: Counter[tuple[Any, ...]] = Counter()
    durations: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    keys = (
        "subject_id",
        "subject_scope",
        "outer_fold_id",
        "pipeline_id",
        "class_label",
        "inner_role",
        "nbm_role",
        "classifier_role",
    )
    for row in window_rows:
        key = tuple(row[name] for name in keys)
        window_counts[key] += 1
    for row in group_rows:
        key = tuple(row[name] for name in keys)
        group_counts[key] += 1
        durations[key] += as_float(row["duration_sec"])
    output: list[dict[str, Any]] = []
    for key in sorted(set(window_counts) | set(group_counts)):
        item = dict(zip(keys, key))
        item.update(
            {
                "group_count": group_counts[key],
                "window_count": window_counts[key],
                "duration_sec": durations[key],
            }
        )
        output.append(item)
    return output


def select_windows(
    rows: Iterable[dict[str, Any]],
    subject: str,
    outer_fold: int,
    pipeline: int,
    *,
    nbm_role: str | None = None,
    classifier_role: str | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["subject_id"] == subject
        and as_int(row["outer_fold_id"]) == outer_fold
        and as_int(row["pipeline_index"]) == pipeline
        and (nbm_role is None or row["nbm_role"] == nbm_role)
        and (classifier_role is None or row["classifier_role"] == classifier_role)
        and (label is None or row["class_label"] == label)
    ]


def save_indices(
    output: Path,
    source_outer_rows: Sequence[dict[str, str]],
    window_rows: Sequence[dict[str, Any]],
) -> None:
    index_dir = output / "inner_split_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted({row["subject_id"] for row in source_outer_rows})
    for subject in subjects:
        payload: dict[str, np.ndarray] = {}
        for outer_fold in range(N_INNER_SPLITS):
            for pipeline in range(N_INNER_SPLITS):
                prefix = f"outer{outer_fold}_pipeline{PIPELINE_NAMES[pipeline]}"
                specs = {
                    "nbm_train_nonfog": {"nbm_role": "nbm_weight_train_clean"},
                    "nbm_earlystop_nonfog": {"nbm_role": "nbm_earlystop_clean"},
                    "oof_holdout_nonfog": {
                        "classifier_role": "classifier_validation_oof",
                        "label": "CLEAN_NONFOG",
                    },
                    "oof_holdout_fog": {
                        "classifier_role": "classifier_validation_oof",
                        "label": "FOG",
                    },
                    "classifier_train_nonfog": {
                        "classifier_role": "classifier_train_oof",
                        "label": "CLEAN_NONFOG",
                    },
                    "classifier_train_fog": {
                        "classifier_role": "classifier_train_oof",
                        "label": "FOG",
                    },
                    "outer_test_nonfog": {
                        "classifier_role": "classifier_outer_test_predict",
                        "label": "CLEAN_NONFOG",
                    },
                    "outer_test_fog": {
                        "classifier_role": "classifier_outer_test_predict",
                        "label": "FOG",
                    },
                }
                for suffix, filters in specs.items():
                    selected = select_windows(
                        window_rows,
                        subject,
                        outer_fold,
                        pipeline,
                        **filters,
                    )
                    payload[f"{prefix}_{suffix}_source_rows"] = np.asarray(
                        [as_int(row["source_window_row_index"]) for row in selected],
                        dtype=np.int64,
                    )
                    payload[f"{prefix}_{suffix}_window_ids"] = np.asarray(
                        [row["window_id"] for row in selected],
                        dtype=str,
                    )
                classifier_train = select_windows(
                    window_rows,
                    subject,
                    outer_fold,
                    pipeline,
                    classifier_role="classifier_train_oof",
                )
                payload[f"{prefix}_classifier_train_oof_generator_pipelines"] = np.asarray(
                    [row["oof_feature_generator_pipeline"] for row in classifier_train],
                    dtype=str,
                )
        np.savez_compressed(index_dir / f"{subject}_outer3_inner3_indices.npz", **payload)


def quality_report(
    source_outer_rows: Sequence[dict[str, str]],
    assignment_rows: Sequence[dict[str, Any]],
    group_rows: Sequence[dict[str, Any]],
    window_rows: Sequence[dict[str, Any]],
    source_outer_quality: dict[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    subjects = sorted({row["subject_id"] for row in source_outer_rows})

    # Every active outer-train group is assigned once per outer fold.
    expected = {
        (as_int(row["fold_id"]), row["group_id"])
        for row in base.read_csv(Path(source_outer_quality["source_group_roles_path"]))
        if row["outer_role"] == "outer_train"
    } if source_outer_quality.get("source_group_roles_path") else {
        (as_int(row["outer_fold_id"]), row["group_id"]) for row in assignment_rows
    }
    actual_counts = Counter(
        (as_int(row["outer_fold_id"]), row["group_id"]) for row in assignment_rows
    )
    if set(actual_counts) != expected or any(count != 1 for count in actual_counts.values()):
        problems.append("active outer-train group assignment is not exactly once")

    # Group-count balance for every subject/outer-fold/class.
    balance: dict[str, Any] = {}
    by_partition: dict[tuple[str, int, str], Counter[int]] = defaultdict(Counter)
    for row in assignment_rows:
        by_partition[(row["subject_id"], as_int(row["outer_fold_id"]), row["class_label"])][
            as_int(row["assigned_inner_fold"])
        ] += 1
    for key, counts in sorted(by_partition.items()):
        values = [counts[index] for index in range(N_INNER_SPLITS)]
        balance[":".join(map(str, key))] = values
        if max(values) - min(values) > 1 or min(values) == 0:
            problems.append(f"inner group-count imbalance: {key} -> {values}")

    group_lookup = {
        (
            row["subject_id"],
            as_int(row["outer_fold_id"]),
            as_int(row["pipeline_index"]),
            row["group_id"],
        ): row
        for row in group_rows
    }
    empty_required: list[str] = []
    oof_generator_leaks: list[str] = []
    for subject in subjects:
        scope = next(row["subject_scope"] for row in source_outer_rows if row["subject_id"] == subject)
        for outer_fold in range(N_INNER_SPLITS):
            for pipeline in range(N_INNER_SPLITS):
                selected = [
                    row
                    for row in group_rows
                    if row["subject_id"] == subject
                    and as_int(row["outer_fold_id"]) == outer_fold
                    and as_int(row["pipeline_index"]) == pipeline
                ]
                nbm_train = {row["group_id"] for row in selected if row["nbm_role"] == "nbm_weight_train_clean"}
                nbm_stop = {row["group_id"] for row in selected if row["nbm_role"] == "nbm_earlystop_clean"}
                holdout = {row["group_id"] for row in selected if row["inner_role"] == "inner_holdout_oof"}
                classifier_train = {row["group_id"] for row in selected if row["classifier_role"] == "classifier_train_oof"}
                classifier_val = {row["group_id"] for row in selected if row["classifier_role"] == "classifier_validation_oof"}
                if nbm_train & nbm_stop or (nbm_train | nbm_stop) & holdout:
                    problems.append(f"NBM role overlap: {subject}:outer{outer_fold}:pipeline{pipeline}")
                if classifier_train & classifier_val:
                    problems.append(f"classifier train/validation overlap: {subject}:outer{outer_fold}:pipeline{pipeline}")
                if classifier_val != holdout:
                    problems.append(f"classifier validation is not inner holdout: {subject}:outer{outer_fold}:pipeline{pipeline}")
                if not nbm_train or not nbm_stop or not holdout:
                    empty_required.append(f"{subject}:outer{outer_fold}:pipeline{pipeline}")
                if scope != "clean_only_control":
                    for label in ("CLEAN_NONFOG", "FOG"):
                        if not any(row["classifier_role"] == "classifier_train_oof" and row["class_label"] == label for row in selected):
                            empty_required.append(f"{subject}:outer{outer_fold}:pipeline{pipeline}:classifier_train:{label}")
                        if not any(row["classifier_role"] == "classifier_validation_oof" and row["class_label"] == label for row in selected):
                            empty_required.append(f"{subject}:outer{outer_fold}:pipeline{pipeline}:classifier_val:{label}")

                # Every OOF target must be absent from the generator NBM's fit/stop groups.
                for row in selected:
                    if row["outer_role"] != "outer_train" or row["assigned_inner_fold"] == "":
                        continue
                    generator = as_int(row["assigned_inner_fold"])
                    generated_role = group_lookup[(subject, outer_fold, generator, row["group_id"])]
                    if generated_role["inner_role"] != "inner_holdout_oof" or generated_role["nbm_role"] in {
                        "nbm_weight_train_clean",
                        "nbm_earlystop_clean",
                    }:
                        oof_generator_leaks.append(f"{subject}:outer{outer_fold}:{row['group_id']}")

    if empty_required:
        problems.append("required partitions are empty")
    if oof_generator_leaks:
        problems.append("OOF feature generator saw its target group")

    # All outer-test windows appear once in each of the three pipelines.
    outer_test_counts = Counter(
        (as_int(row["outer_fold_id"]), row["window_id"])
        for row in window_rows
        if row["classifier_role"] == "classifier_outer_test_predict"
    )
    outer_test_bad = [key for key, count in outer_test_counts.items() if count != N_INNER_SPLITS]
    if outer_test_bad:
        problems.append("outer-test windows are not represented in all three pipelines")

    # Expanded row count must be exactly 3x the source outer manifest.
    expected_rows = len(source_outer_rows) * N_INNER_SPLITS
    if len(window_rows) != expected_rows:
        problems.append(f"expanded window row count {len(window_rows)} != {expected_rows}")

    return {
        "overall_pass": not problems and bool(source_outer_quality.get("overall_pass")),
        "source_outer_quality_pass": bool(source_outer_quality.get("overall_pass")),
        "source_outer_window_rows": len(source_outer_rows),
        "expanded_inner_pipeline_window_rows": len(window_rows),
        "active_outer_train_group_assignments": len(assignment_rows),
        "inner_group_count_balance": balance,
        "empty_required_partitions": empty_required,
        "oof_generator_leaks": oof_generator_leaks[:50],
        "outer_test_pipeline_count_failures": outer_test_bad[:50],
        "problems": problems,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    source_quality = json.loads((source / "group3_quality_report.json").read_text(encoding="utf-8"))
    if not source_quality.get("overall_pass"):
        raise RuntimeError("processed_Group_3folds quality gate is not PASS")

    build = output.with_name(f"{output.name}.__building_{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, build, dirs_exist_ok=True)

    source_outer_rows = base.read_csv(source / SOURCE_WINDOW_MANIFEST)
    source_group_rows = enrich_outer_group_roles(
        base.read_csv(source / SOURCE_GROUP_ROLES),
        base.read_csv(source / SOURCE_GROUP_ASSIGNMENTS),
    )
    assignment_rows, assignments = assign_inner_folds(source_group_rows)
    pipeline_group_rows, _ = build_pipeline_group_roles(source_group_rows, assignments)
    window_rows = expand_window_manifest(source_outer_rows, pipeline_group_rows)
    summary_rows = build_summary(window_rows, pipeline_group_rows)

    # Pass the source path only in-memory so quality_report can audit the source
    # role universe without persisting an environment-specific dependency.
    source_quality_for_audit = dict(source_quality)
    source_quality_for_audit["source_group_roles_path"] = str(source / SOURCE_GROUP_ROLES)
    quality = quality_report(
        source_outer_rows,
        assignment_rows,
        pipeline_group_rows,
        window_rows,
        source_quality_for_audit,
    )
    if not quality["overall_pass"]:
        base.write_json(build / "FAILED_INNER3_QUALITY_REPORT.json", quality)
        base.write_csv(build / INNER_ASSIGNMENTS, assignment_rows)
        base.write_csv(build / INNER_GROUP_ROLES, pipeline_group_rows)
        raise RuntimeError(f"inner-3 quality gate failed; inspect {build}")

    base.write_csv(build / INNER_ASSIGNMENTS, assignment_rows)
    base.write_csv(build / INNER_GROUP_ROLES, pipeline_group_rows)
    base.write_csv(build / INNER_WINDOW_MANIFEST, window_rows)
    base.write_csv(build / INNER_SUMMARY, summary_rows)
    base.write_json(build / INNER_QUALITY, quality)
    save_indices(build, source_outer_rows, window_rows)

    protocol = {
        "dataset_id": DATASET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_window_manifest": str(source / SOURCE_WINDOW_MANIFEST),
        "source_window_manifest_sha256": base.sha256(source / SOURCE_WINDOW_MANIFEST),
        "outer_splits": 3,
        "inner_splits": 3,
        "scope": "within-subject only; each subject and outer fold is split independently",
        "inner_assignment": "active outer-train groups only; deterministic balance by group count, then windows and duration, separately for each class",
        "fog_group": "retained FoG event; indivisible",
        "nonfog_group": "clean Non-FoG block; indivisible",
        "inner_embargo_seconds": base.INTER_SPLIT_EMBARGO / base.FS,
        "inner_embargo_policy": "exclude an entire inner-train group from a pipeline when it lies within 5 s of that pipeline's inner holdout group in the same record",
        "nbm_policy": "pipeline k trains and early-stops only on eligible clean Non-FoG groups assigned to the other inner folds",
        "nbm_earlystop_policy": "deterministic group subset closest to 15% of eligible clean windows; never window-randomized",
        "oof_policy": "each active outer-train group is transformed by the one NBM whose assigned inner fold holds that group out",
        "classifier_policy": "pipeline k trains on eligible OOF features from other inner folds and validates on OOF features assigned to k",
        "outer_test_policy": "pipelines A/B/C independently predict the unchanged matching outer-test fold; average their probabilities",
        "forbidden": [
            "FoG in NBM weight fitting or NBM early stopping",
            "in-sample NBM residuals as classifier features",
            "outer-test use for residual choice, tuning, calibration, or threshold selection",
        ],
    }
    base.write_json(build / INNER_PROTOCOL, protocol)

    schema = json.loads((build / "schema.json").read_text(encoding="utf-8"))
    schema["within_subject_outer3_inner3"] = {
        "inner_assignment_manifest": INNER_ASSIGNMENTS,
        "pipeline_group_roles": INNER_GROUP_ROLES,
        "pipeline_window_manifest": INNER_WINDOW_MANIFEST,
        "pipeline_summary": INNER_SUMMARY,
        "quality_report": INNER_QUALITY,
        "protocol": INNER_PROTOCOL,
        "pipeline_map": PIPELINE_NAMES,
        "probability_aggregation": "arithmetic mean of A/B/C outer-test probabilities",
    }
    base.write_json(build / "schema.json", schema)
    (build / "README_GROUP_3FOLDS_3INNER.md").write_text(
        "# Daphnet processed_Group_3folds_3inner\n\n"
        "This dataset adds concrete grouped inner 3-fold pipelines to every within-subject outer fold. "
        "A/B/C correspond to assigned inner folds 0/1/2. All splits operate on whole FoG events or "
        "whole clean Non-FoG blocks.\n\n"
        "For pipeline k, use `nbm_weight_train_clean` and `nbm_earlystop_clean` to fit the NBM. "
        "Generate validation residuals only for `classifier_validation_oof`. Classifier training rows "
        "are `classifier_train_oof`; their residuals must be loaded from the pipeline named by "
        "`oof_feature_generator_pipeline`, not recomputed by the current pipeline's NBM. "
        "Predict `classifier_outer_test_predict` independently with A/B/C and average probabilities.\n\n"
        "The outer-test fold is unchanged and is forbidden for model selection, calibration, or threshold tuning.\n",
        encoding="utf-8",
    )

    build.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "quality_pass": quality["overall_pass"],
                "assignment_rows": len(assignment_rows),
                "pipeline_group_rows": len(pipeline_group_rows),
                "pipeline_window_rows": len(window_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
