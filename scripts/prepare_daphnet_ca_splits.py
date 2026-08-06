"""Build the leakage-safe Daphnet processed_CA window dataset.

The canonical continuous records are preserved.  CA adds 2 s windows on a
1 s grid, labels a window as FOG when at least 1 s (64/128 samples) is FOG,
and creates deterministic within-subject train/validation/test assignments.

Assignments are group based rather than window random:

* nearby complete FOG events are clustered when their possible 2 s windows
  could violate the five-second cross-split embargo;
* clean Non-FOG windows are partitioned into at-most-60-second contiguous
  blocks with a five-second inter-block embargo;
* event clusters and clean blocks are allocated to 65%/15%/20% targets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_a5_splits as base  # noqa: E402


FS = 64
WINDOW = 128
STRIDE = 64
MIN_FOG_SAMPLES = 64
INTER_SPLIT_EMBARGO = 5 * FS
EVENT_CLUSTER_GAP = 2 * (WINDOW - 1) + INTER_SPLIT_EMBARGO
CLEAN_FOG_GUARD = 7 * FS
CLEAN_BLOCK_SECONDS = 60
SPLITS = ("train", "validation", "test")
TARGETS = {"train": 0.65, "validation": 0.15, "test": 0.20}
SPLIT_CODES = {name: index for index, name in enumerate(SPLITS)}
GROUP_KIND_CODES = {"fog_event_cluster": 0, "clean_nonfog_block": 1}


@dataclass
class Group:
    group_id: str
    kind: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    start_index: int
    end_index_exclusive: int
    starts: tuple[int, ...]
    events: tuple[dict[str, Any], ...]
    fog_samples_by_start: dict[int, int]
    split: str = ""

    @property
    def window_count(self) -> int:
        return len(self.starts)

    @property
    def fog_window_count(self) -> int:
        return sum(
            int(self.fog_samples_by_start[start] >= MIN_FOG_SAMPLES)
            for start in self.starts
        )

    @property
    def nonfog_window_count(self) -> int:
        return self.window_count - self.fog_window_count

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def fog_duration_samples(self) -> int:
        return sum(
            int(event["end_index_exclusive"]) - int(event["start_index"])
            for event in self.events
        )

    def metrics(self) -> dict[str, float]:
        return {
            "window_count": float(self.window_count),
            "fog_window_count": float(self.fog_window_count),
            "nonfog_window_count": float(self.nonfog_window_count),
            "event_count": float(self.event_count),
            "fog_duration_samples": float(self.fog_duration_samples),
        }


def parse_args() -> argparse.Namespace:
    dataset_root = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root / "processed")
    parser.add_argument("--output", type=Path, default=dataset_root / "processed_CA")
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "019fd1bb-dca3-7063-b479-4ec71f2425f8"
            / "daphnet_fog_segment_summary.csv"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the split quality report without writing processed_CA.",
    )
    return parser.parse_args()


def subject_scope(subject: str) -> str:
    if subject in base.MAIN_SUBJECTS:
        return "formal_main7"
    if subject in base.DIAGNOSTIC_SUBJECTS:
        return "diagnostic_only"
    if subject in base.CLEAN_ONLY_SUBJECTS:
        return "clean_only_control"
    return "other"


def prepare_events(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    typed: list[dict[str, Any]] = []
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subject_ordinals: Counter[str] = Counter()
    ordered = sorted(
        rows,
        key=lambda row: (
            row["subject_id"],
            int(row["segment_id"]),
            int(row["start_index"]),
        ),
    )
    for raw in ordered:
        row: dict[str, Any] = dict(raw)
        subject = str(raw["subject_id"])
        row.update(
            {
                "event_id": int(raw["event_id"]),
                "start_index": int(raw["start_index"]),
                "end_index": int(raw["end_index"]),
                "end_index_exclusive": int(raw["end_index"]) + 1,
                "duration_sec": float(raw["duration_sec"]),
                "subject_event_ordinal": subject_ordinals[subject],
                "ca_event_cluster_id": "",
                "ca_split": "",
                "ca_window_count": 0,
                "ca_fog_window_count": 0,
            }
        )
        subject_ordinals[subject] += 1
        typed.append(row)
        by_record[str(row["record_id"])].append(row)
    return typed, by_record


def cluster_events(
    records: Sequence[base.Record],
    manifest: dict[str, dict[str, str]],
    events_by_record: dict[str, list[dict[str, Any]]],
) -> list[Group]:
    groups: list[Group] = []
    for record in records:
        events = sorted(
            events_by_record.get(record.record_id, []),
            key=lambda event: (int(event["start_index"]), int(event["event_id"])),
        )
        if not events:
            continue
        clusters: list[list[dict[str, Any]]] = []
        current = [events[0]]
        current_end = int(events[0]["end_index_exclusive"])
        for event in events[1:]:
            gap = int(event["start_index"]) - current_end
            if gap < EVENT_CLUSTER_GAP:
                current.append(event)
                current_end = max(current_end, int(event["end_index_exclusive"]))
            else:
                clusters.append(current)
                current = [event]
                current_end = int(event["end_index_exclusive"])
        clusters.append(current)

        for cluster_index, cluster in enumerate(clusters):
            cluster_start = min(int(event["start_index"]) for event in cluster)
            cluster_end = max(int(event["end_index_exclusive"]) for event in cluster)
            starts: list[int] = []
            fog_samples: dict[int, int] = {}
            for start in range(0, len(record.y) - WINDOW + 1, STRIDE):
                end = start + WINDOW
                if end <= cluster_start or start >= cluster_end:
                    continue
                if not record.valid[start:end].all():
                    continue
                starts.append(start)
                fog_samples[start] = int(np.sum(record.y[start:end] == 1))
            group_id = f"{record.record_id}_fogcluster{cluster_index:03d}"
            group = Group(
                group_id=group_id,
                kind="fog_event_cluster",
                subject_id=record.subject_id,
                record_id=record.record_id,
                run_id=record.run_id,
                segment_id=int(manifest[record.record_id]["segment_id"]),
                start_index=cluster_start,
                end_index_exclusive=cluster_end,
                starts=tuple(starts),
                events=tuple(cluster),
                fog_samples_by_start=fog_samples,
            )
            for event in cluster:
                event["ca_event_cluster_id"] = group_id
            groups.append(group)
    return groups


def true_if_clean_with_guard(record: base.Record, start: int) -> bool:
    end = start + WINDOW
    if not record.valid[start:end].all():
        return False
    guard_start = max(0, start - CLEAN_FOG_GUARD)
    guard_end = min(len(record.y), end + CLEAN_FOG_GUARD)
    return not bool(np.any(record.y[guard_start:guard_end] == 1))


def consecutive_runs(starts: Sequence[int]) -> Iterable[list[int]]:
    if not starts:
        return
    current = [int(starts[0])]
    for start in starts[1:]:
        if int(start) == current[-1] + STRIDE:
            current.append(int(start))
        else:
            yield current
            current = [int(start)]
    yield current


def clean_groups(
    records: Sequence[base.Record], manifest: dict[str, dict[str, str]]
) -> list[Group]:
    groups: list[Group] = []
    for record in records:
        candidates = [
            start
            for start in range(0, len(record.y) - WINDOW + 1, STRIDE)
            if true_if_clean_with_guard(record, start)
        ]
        block_number = 0
        for run in consecutive_runs(candidates):
            position = 0
            while position < len(run):
                first = int(run[position])
                boundary = first + CLEAN_BLOCK_SECONDS * FS
                stop = position
                while stop < len(run) and int(run[stop]) + WINDOW <= boundary:
                    stop += 1
                chosen = tuple(int(value) for value in run[position:stop])
                if not chosen:
                    chosen = (int(run[position]),)
                    stop = position + 1
                group_id = f"{record.record_id}_cleanblock{block_number:03d}"
                groups.append(
                    Group(
                        group_id=group_id,
                        kind="clean_nonfog_block",
                        subject_id=record.subject_id,
                        record_id=record.record_id,
                        run_id=record.run_id,
                        segment_id=int(manifest[record.record_id]["segment_id"]),
                        start_index=chosen[0],
                        end_index_exclusive=chosen[-1] + WINDOW,
                        starts=chosen,
                        events=(),
                        fog_samples_by_start={start: 0 for start in chosen},
                    )
                )
                block_number += 1
                next_start = chosen[-1] + WINDOW + INTER_SPLIT_EMBARGO
                position = stop
                while position < len(run) and int(run[position]) < next_start:
                    position += 1
    return groups


def empty_counts(metrics: Sequence[str]) -> dict[str, dict[str, float]]:
    return {split: {metric: 0.0 for metric in metrics} for split in SPLITS}


def clone_counts(
    counts: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    return {split: dict(values) for split, values in counts.items()}


def add_metrics(
    counts: dict[str, dict[str, float]], split: str, metrics: dict[str, float], sign: float = 1.0
) -> None:
    for metric, value in metrics.items():
        if metric in counts[split]:
            counts[split][metric] += sign * float(value)


def allocation_objective(
    counts: dict[str, dict[str, float]],
    totals: dict[str, float],
    weights: dict[str, float],
    group_counts: dict[str, int],
    require_groups: bool,
    require_positive_metric: str | None,
    bounds_metric: str | None,
) -> float:
    score = 0.0
    for metric, weight in weights.items():
        total = totals.get(metric, 0.0)
        if total <= 0:
            continue
        for split in SPLITS:
            fraction = counts[split][metric] / total
            score += weight * (fraction - TARGETS[split]) ** 2
    if require_groups and any(group_counts[split] <= 0 for split in SPLITS):
        score += 1_000_000.0
    if require_positive_metric and totals.get(require_positive_metric, 0.0) > 0:
        if any(counts[split][require_positive_metric] <= 0 for split in SPLITS):
            score += 1_000_000.0
    if bounds_metric and totals.get(bounds_metric, 0.0) > 0:
        bounds = {
            "train": (0.60, 0.70),
            "validation": (0.15, 0.20),
            "test": (0.15, 0.25),
        }
        total = totals[bounds_metric]
        for split, (lower, upper) in bounds.items():
            fraction = counts[split][bounds_metric] / total
            if fraction < lower:
                score += 1_000.0 + 10_000.0 * (lower - fraction) ** 2
            elif fraction > upper:
                score += 1_000.0 + 10_000.0 * (fraction - upper) ** 2
    return score


def allocate_groups(
    groups: Sequence[Group],
    *,
    metrics: Sequence[str],
    weights: dict[str, float],
    primary_metric: str,
    initial_counts: dict[str, dict[str, float]] | None = None,
    require_groups: bool = True,
    require_positive_metric: str | None = None,
    bounds_metric: str | None = None,
) -> None:
    if not groups:
        return
    counts = empty_counts(metrics) if initial_counts is None else clone_counts(initial_counts)
    totals = {metric: sum(group.metrics()[metric] for group in groups) for metric in metrics}
    if initial_counts is not None:
        for metric in metrics:
            totals[metric] += sum(initial_counts[split][metric] for split in SPLITS)
    group_counts = {split: 0 for split in SPLITS}

    ordered = sorted(
        groups,
        key=lambda group: (
            -group.metrics()[primary_metric],
            -group.window_count,
            group.record_id,
            group.start_index,
            group.group_id,
        ),
    )
    seeded: set[str] = set()
    if len(ordered) >= 3:
        seed_splits = ("train", "test", "validation")
        seed_candidates = ordered
        if require_positive_metric:
            positive = [
                group
                for group in ordered
                if group.metrics()[require_positive_metric] > 0
            ]
            if len(positive) >= 3:
                seed_candidates = positive
        for split, group in zip(seed_splits, seed_candidates[:3]):
            group.split = split
            seeded.add(group.group_id)
            add_metrics(counts, split, group.metrics())
            group_counts[split] += 1

    for group in ordered:
        if group.group_id in seeded:
            continue
        values = group.metrics()

        def deficit(split: str) -> tuple[float, float, int]:
            target = TARGETS[split] * max(totals[primary_metric], 1.0)
            normalized = (target - counts[split][primary_metric]) / max(target, 1.0)
            absolute = target - counts[split][primary_metric]
            return normalized, absolute, -SPLITS.index(split)

        split = max(SPLITS, key=deficit)
        group.split = split
        add_metrics(counts, split, values)
        group_counts[split] += 1

    current = allocation_objective(
        counts,
        totals,
        weights,
        group_counts,
        require_groups,
        require_positive_metric,
        bounds_metric,
    )
    for _ in range(200):
        best: tuple[float, str, Group, Group | None] | None = None
        for group in ordered:
            source = group.split
            values = group.metrics()
            for target in SPLITS:
                if target == source:
                    continue
                candidate_counts = clone_counts(counts)
                candidate_groups = dict(group_counts)
                add_metrics(candidate_counts, source, values, -1.0)
                add_metrics(candidate_counts, target, values, 1.0)
                candidate_groups[source] -= 1
                candidate_groups[target] += 1
                score = allocation_objective(
                    candidate_counts,
                    totals,
                    weights,
                    candidate_groups,
                    require_groups,
                    require_positive_metric,
                    bounds_metric,
                )
                if score + 1e-12 < current and (best is None or score < best[0] - 1e-12):
                    best = (score, target, group, None)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if left.split == right.split:
                    continue
                candidate_counts = clone_counts(counts)
                add_metrics(candidate_counts, left.split, left.metrics(), -1.0)
                add_metrics(candidate_counts, right.split, right.metrics(), -1.0)
                add_metrics(candidate_counts, left.split, right.metrics(), 1.0)
                add_metrics(candidate_counts, right.split, left.metrics(), 1.0)
                score = allocation_objective(
                    candidate_counts,
                    totals,
                    weights,
                    group_counts,
                    require_groups,
                    require_positive_metric,
                    bounds_metric,
                )
                if score + 1e-12 < current and (best is None or score < best[0] - 1e-12):
                    best = (score, right.split, left, right)
        if best is None:
            break
        score, target, left, right = best
        if right is None:
            source = left.split
            add_metrics(counts, source, left.metrics(), -1.0)
            add_metrics(counts, target, left.metrics(), 1.0)
            group_counts[source] -= 1
            group_counts[target] += 1
            left.split = target
        else:
            left_split, right_split = left.split, right.split
            add_metrics(counts, left_split, left.metrics(), -1.0)
            add_metrics(counts, right_split, right.metrics(), -1.0)
            add_metrics(counts, left_split, right.metrics(), 1.0)
            add_metrics(counts, right_split, left.metrics(), 1.0)
            left.split, right.split = right_split, left_split
        current = score


def allocate_all(event_groups: Sequence[Group], clean: Sequence[Group]) -> None:
    subjects = sorted({group.subject_id for group in event_groups} | {group.subject_id for group in clean})
    for subject in subjects:
        subject_events = [group for group in event_groups if group.subject_id == subject]
        if subject_events:
            allocate_groups(
                subject_events,
                metrics=(
                    "window_count",
                    "fog_window_count",
                    "nonfog_window_count",
                    "event_count",
                    "fog_duration_samples",
                ),
                weights={
                    "window_count": 0.5,
                    "fog_window_count": 6.0,
                    "nonfog_window_count": 0.5,
                    "event_count": 3.0,
                    "fog_duration_samples": 1.0,
                },
                primary_metric="fog_window_count",
                require_groups=True,
                require_positive_metric="fog_window_count",
            )

        initial = empty_counts(("nonfog_window_count", "window_count"))
        for group in subject_events:
            initial[group.split]["nonfog_window_count"] += group.nonfog_window_count
            initial[group.split]["window_count"] += group.nonfog_window_count
        subject_clean = [group for group in clean if group.subject_id == subject]
        allocate_groups(
            subject_clean,
            metrics=("nonfog_window_count", "window_count"),
            weights={"nonfog_window_count": 8.0, "window_count": 1.0},
            primary_metric="nonfog_window_count",
            initial_counts=initial,
            require_groups=True,
            require_positive_metric="nonfog_window_count",
            bounds_metric="nonfog_window_count",
        )


def overlapping_event_ids(group: Group, start: int, end: int) -> str:
    values = [
        int(event["event_id"])
        for event in group.events
        if start < int(event["end_index_exclusive"])
        and end > int(event["start_index"])
    ]
    return ";".join(str(value) for value in values)


def window_rows(
    groups: Sequence[Group], manifest: dict[str, dict[str, str]], record_order: dict[str, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        manifest_row = manifest[group.record_id]
        source_start = int(manifest_row["source_start_row"])
        for start in group.starts:
            end = start + WINDOW
            fog_samples = int(group.fog_samples_by_start[start])
            y_binary = int(fog_samples >= MIN_FOG_SAMPLES)
            transition = group.kind == "fog_event_cluster" and y_binary == 0
            rows.append(
                {
                    "window_id": f"{group.record_id}:{start}:{end}",
                    "subject_id": group.subject_id,
                    "subject_scope": subject_scope(group.subject_id),
                    "record_id": group.record_id,
                    "run_id": group.run_id,
                    "segment_id": group.segment_id,
                    "source_file": manifest_row["source_file"],
                    "ca_split": group.split,
                    "class_label": "FOG" if y_binary else "NONFOG",
                    "y_binary": y_binary,
                    "nonfog_subtype": "TRANSITION" if transition else ("CLEAN" if not y_binary else ""),
                    "group_id": group.group_id,
                    "group_kind": group.kind,
                    "start_index": start,
                    "end_index_exclusive": end,
                    "start_time_sec": start / FS,
                    "end_time_sec": end / FS,
                    "source_start_row": source_start + start,
                    "source_end_row_exclusive": source_start + end,
                    "fog_samples_in_2s": fog_samples,
                    "fog_duration_in_2s_sec": fog_samples / FS,
                    "full_2s_fog_fraction": fog_samples / WINDOW,
                    "label_rule": "FOG iff fog_samples_in_2s >= 64",
                    "event_cluster_id": group.group_id if group.kind == "fog_event_cluster" else "",
                    "overlapping_event_ids": overlapping_event_ids(group, start, end) if group.events else "",
                    "clean_nonfog": bool(group.kind == "clean_nonfog_block"),
                    "clean_fog_guard_sec_each_side": CLEAN_FOG_GUARD / FS if group.kind == "clean_nonfog_block" else "",
                    "inter_split_embargo_sec": INTER_SPLIT_EMBARGO / FS,
                }
            )
    rows.sort(
        key=lambda row: (
            row["subject_id"],
            SPLIT_CODES[row["ca_split"]],
            record_order[row["record_id"]],
            int(row["start_index"]),
        )
    )
    return rows


def group_rows(groups: Sequence[Group]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in sorted(
        groups,
        key=lambda item: (
            item.subject_id,
            SPLIT_CODES[item.split],
            item.record_id,
            item.start_index,
        ),
    ):
        rows.append(
            {
                "group_id": group.group_id,
                "group_kind": group.kind,
                "subject_id": group.subject_id,
                "subject_scope": subject_scope(group.subject_id),
                "record_id": group.record_id,
                "run_id": group.run_id,
                "segment_id": group.segment_id,
                "ca_split": group.split,
                "start_index": group.start_index,
                "end_index_exclusive": group.end_index_exclusive,
                "start_time_sec": group.start_index / FS,
                "end_time_sec": group.end_index_exclusive / FS,
                "window_count": group.window_count,
                "fog_window_count": group.fog_window_count,
                "nonfog_window_count": group.nonfog_window_count,
                "event_count": group.event_count,
                "fog_duration_sec": group.fog_duration_samples / FS,
                "maximum_clean_block_seconds": CLEAN_BLOCK_SECONDS if group.kind == "clean_nonfog_block" else "",
                "inter_split_embargo_sec": INTER_SPLIT_EMBARGO / FS,
            }
        )
    return rows


def update_event_rows(events: Sequence[dict[str, Any]], groups: Sequence[Group]) -> None:
    by_id = {group.group_id: group for group in groups if group.kind == "fog_event_cluster"}
    for event in events:
        group = by_id[str(event["ca_event_cluster_id"])]
        event["ca_split"] = group.split
        event_start = int(event["start_index"])
        event_end = int(event["end_index_exclusive"])
        selected = [
            start
            for start in group.starts
            if start < event_end and start + WINDOW > event_start
        ]
        event["ca_window_count"] = len(selected)
        event["ca_fog_window_count"] = sum(
            int(group.fog_samples_by_start[start] >= MIN_FOG_SAMPLES)
            for start in selected
        )


def leakage_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    duplicate_ids = len(rows) - len({str(row["window_id"]) for row in rows})
    overlapping = 0
    inside_embargo = 0
    examples: list[dict[str, Any]] = []
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_record[str(row["record_id"])].append(row)
    for record_id, selected in by_record.items():
        ordered = sorted(selected, key=lambda row: int(row["start_index"]))
        for index, left in enumerate(ordered):
            left_end = int(left["end_index_exclusive"])
            for right in ordered[index + 1 :]:
                right_start = int(right["start_index"])
                if right_start >= left_end + INTER_SPLIT_EMBARGO:
                    break
                if left["ca_split"] == right["ca_split"]:
                    continue
                distance = base.interval_distance(
                    int(left["start_index"]),
                    left_end,
                    right_start,
                    int(right["end_index_exclusive"]),
                )
                if distance == 0:
                    overlapping += 1
                if distance < INTER_SPLIT_EMBARGO:
                    inside_embargo += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "record_id": record_id,
                                "left": left["window_id"],
                                "left_split": left["ca_split"],
                                "right": right["window_id"],
                                "right_split": right["ca_split"],
                                "distance_samples": distance,
                            }
                        )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["ca_split"]))
    return {
        "duplicate_window_ids": duplicate_ids,
        "groups_crossing_splits": sum(len(values) > 1 for values in group_splits.values()),
        "cross_split_overlapping_window_pairs": overlapping,
        "cross_split_pairs_inside_5s_embargo": inside_embargo,
        "examples": examples,
        "pass": bool(
            duplicate_ids == 0
            and not any(len(values) > 1 for values in group_splits.values())
            and overlapping == 0
            and inside_embargo == 0
        ),
    }


def split_summary(
    rows: Sequence[dict[str, Any]], groups: Sequence[Group], events: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    output: list[dict[str, Any]] = []
    for subject in subjects + ["ALL"]:
        subject_rows = list(rows) if subject == "ALL" else [row for row in rows if row["subject_id"] == subject]
        subject_groups = list(groups) if subject == "ALL" else [group for group in groups if group.subject_id == subject]
        subject_events = list(events) if subject == "ALL" else [event for event in events if event["subject_id"] == subject]
        total_windows = len(subject_rows)
        total_fog = sum(int(row["y_binary"]) for row in subject_rows)
        total_nonfog = total_windows - total_fog
        total_events = len(subject_events)
        total_duration = sum(float(event["duration_sec"]) for event in subject_events)
        for split in SPLITS:
            chosen = [row for row in subject_rows if row["ca_split"] == split]
            fog = sum(int(row["y_binary"]) for row in chosen)
            nonfog = len(chosen) - fog
            chosen_events = [event for event in subject_events if event["ca_split"] == split]
            duration = sum(float(event["duration_sec"]) for event in chosen_events)
            output.append(
                {
                    "subject_id": subject,
                    "subject_scope": "aggregate" if subject == "ALL" else subject_scope(subject),
                    "ca_split": split,
                    "target_fraction": TARGETS[split],
                    "window_count": len(chosen),
                    "window_fraction": len(chosen) / total_windows if total_windows else "",
                    "fog_window_count": fog,
                    "fog_window_fraction": fog / total_fog if total_fog else "",
                    "nonfog_window_count": nonfog,
                    "nonfog_window_fraction": nonfog / total_nonfog if total_nonfog else "",
                    "group_count": sum(group.split == split for group in subject_groups),
                    "fog_event_count": len(chosen_events),
                    "fog_event_fraction": len(chosen_events) / total_events if total_events else "",
                    "fog_duration_sec": duration,
                    "fog_duration_fraction": duration / total_duration if total_duration else "",
                }
            )
    return output


def source_summary_audit(
    summary_rows: Sequence[dict[str, str]],
    manifest_rows: Sequence[dict[str, str]],
    events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    summary = {row["record_id"]: row for row in summary_rows}
    event_counts = Counter(str(event["record_id"]) for event in events)
    event_durations: dict[str, float] = defaultdict(float)
    for event in events:
        event_durations[str(event["record_id"])] += float(event["duration_sec"])
    output: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        record_id = manifest["record_id"]
        source = summary.get(record_id)
        if source is None:
            output.append({"record_id": record_id, "status": "MISSING_IN_REFERENCE_SUMMARY"})
            continue
        count_match = int(source["manifest_fog_event_count"]) == event_counts[record_id]
        duration_match = abs(float(source["fog_total_duration_sec"]) - event_durations[record_id]) <= 1e-6
        samples_match = int(source["segment_samples"]) == int(manifest["n_samples"])
        output.append(
            {
                "record_id": record_id,
                "subject_id": manifest["subject_id"],
                "summary_segment_samples": int(source["segment_samples"]),
                "manifest_n_samples": int(manifest["n_samples"]),
                "summary_fog_event_count": int(source["manifest_fog_event_count"]),
                "source_fog_event_count": event_counts[record_id],
                "summary_fog_duration_sec": float(source["fog_total_duration_sec"]),
                "source_fog_duration_sec": event_durations[record_id],
                "samples_match": samples_match,
                "event_count_match": count_match,
                "fog_duration_match": duration_match,
                "status": "PASS" if samples_match and count_match and duration_match else "FAIL",
            }
        )
    extra = sorted(set(summary) - {row["record_id"] for row in manifest_rows})
    for record_id in extra:
        output.append({"record_id": record_id, "status": "EXTRA_IN_REFERENCE_SUMMARY"})
    return output, bool(output and all(row["status"] == "PASS" for row in output))


def quality_report(
    rows: Sequence[dict[str, Any]],
    groups: Sequence[Group],
    events: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    source_summary_pass: bool,
) -> dict[str, Any]:
    leakage = leakage_audit(rows)
    by_subject = defaultdict(dict)
    for row in summaries:
        if row["subject_id"] == "ALL":
            continue
        by_subject[row["subject_id"]][row["ca_split"]] = row
    subject_gates: dict[str, Any] = {}
    for subject, split_rows in sorted(by_subject.items()):
        fractions = {split: float(split_rows[split]["window_fraction"]) for split in SPLITS}
        window_gate = bool(
            0.60 <= fractions["train"] <= 0.70
            and 0.15 <= fractions["validation"] <= 0.20
            and 0.15 <= fractions["test"] <= 0.25
        )
        fog_available = sum(int(split_rows[split]["fog_window_count"]) for split in SPLITS) > 0
        fog_in_all = bool(
            not fog_available
            or all(int(split_rows[split]["fog_window_count"]) > 0 for split in SPLITS)
        )
        subject_gates[subject] = {
            "window_fractions": fractions,
            "window_ratio_gate_pass": window_gate,
            "fog_windows_present_in_all_splits_or_not_applicable": fog_in_all,
        }
    label_pass = all(
        int(row["y_binary"]) == (int(row["fog_samples_in_2s"]) >= MIN_FOG_SAMPLES)
        for row in rows
    )
    event_split_count = defaultdict(set)
    for event in events:
        event_split_count[(event["record_id"], int(event["event_id"]))].add(event["ca_split"])
    event_pass = all(len(values) == 1 for values in event_split_count.values())
    report = {
        "overall_pass": False,
        "source_summary_audit_pass": source_summary_pass,
        "window_count": len(rows),
        "group_count": len(groups),
        "fog_event_count": len(events),
        "all_windows_length_128": all(
            int(row["end_index_exclusive"]) - int(row["start_index"]) == WINDOW
            for row in rows
        ),
        "all_windows_grid_aligned_stride64": all(
            int(row["start_index"]) % STRIDE == 0 for row in rows
        ),
        "label_rule_exact_pass": label_pass,
        "complete_fog_event_single_split_pass": event_pass,
        "leakage_audit": leakage,
        "subjects": subject_gates,
        "all_subject_window_ratio_gates_pass": all(
            item["window_ratio_gate_pass"] for item in subject_gates.values()
        ),
        "all_fog_subjects_have_fog_windows_in_each_split": all(
            item["fog_windows_present_in_all_splits_or_not_applicable"]
            for item in subject_gates.values()
        ),
    }
    report["overall_pass"] = bool(
        source_summary_pass
        and report["all_windows_length_128"]
        and report["all_windows_grid_aligned_stride64"]
        and label_pass
        and event_pass
        and leakage["pass"]
        and report["all_subject_window_ratio_gates_pass"]
        and report["all_fog_subjects_have_fog_windows_in_each_split"]
    )
    return report


def save_indices(
    root: Path,
    subjects: Sequence[str],
    rows: Sequence[dict[str, Any]],
    groups: Sequence[Group],
    record_order: dict[str, int],
) -> None:
    split_dir = root / "split_indices"
    split_dir.mkdir(parents=True, exist_ok=True)
    group_index = {
        group.group_id: index
        for index, group in enumerate(
            sorted(groups, key=lambda item: (item.subject_id, item.record_id, item.start_index, item.group_id))
        )
    }
    for subject in subjects:
        selected = sorted(
            (row for row in rows if row["subject_id"] == subject),
            key=lambda row: (
                SPLIT_CODES[row["ca_split"]],
                record_order[row["record_id"]],
                int(row["start_index"]),
            ),
        )
        np.savez_compressed(
            split_dir / f"{subject}_ca_window_indices.npz",
            record_index=np.asarray([record_order[row["record_id"]] for row in selected], dtype=np.int16),
            start_index=np.asarray([row["start_index"] for row in selected], dtype=np.int32),
            end_index_exclusive=np.asarray([row["end_index_exclusive"] for row in selected], dtype=np.int32),
            split_code=np.asarray([SPLIT_CODES[row["ca_split"]] for row in selected], dtype=np.int8),
            y_binary=np.asarray([row["y_binary"] for row in selected], dtype=np.int8),
            fog_samples_in_2s=np.asarray([row["fog_samples_in_2s"] for row in selected], dtype=np.int16),
            group_index=np.asarray([group_index[row["group_id"]] for row in selected], dtype=np.int32),
            group_kind_code=np.asarray([GROUP_KIND_CODES[row["group_kind"]] for row in selected], dtype=np.int8),
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    manifest_rows = base.read_csv(source / "manifest.csv")
    manifest = {row["record_id"]: row for row in manifest_rows}
    records = base.load_records(source, manifest_rows)
    record_order = {record.record_id: index for index, record in enumerate(records)}
    events, events_by_record = prepare_events(base.read_csv(source / "fog_events.csv"))
    reference_summary = base.read_csv(summary_path)
    source_audit_rows, source_summary_pass = source_summary_audit(
        reference_summary, manifest_rows, events
    )

    fog_groups = cluster_events(records, manifest, events_by_record)
    clean = clean_groups(records, manifest)
    allocate_all(fog_groups, clean)
    groups = fog_groups + clean
    update_event_rows(events, fog_groups)
    rows = window_rows(groups, manifest, record_order)
    summaries = split_summary(rows, groups, events)
    quality = quality_report(
        rows, groups, events, summaries, source_summary_pass
    )
    payload = {
        "output": str(output),
        "dry_run": bool(args.dry_run),
        "quality": quality,
        "aggregate_split_summary": [
            row for row in summaries if row["subject_id"] == "ALL"
        ],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not quality["overall_pass"]:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))

    build = output.with_name(f"{output.name}.__building_{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.csv", "fog_events.csv", "loso_folds.csv", "preprocessing_report.json"):
        shutil.copy2(source / name, build / name)
    shutil.copy2(summary_path, build / "ca_source_fog_segment_summary.csv")
    shutil.copytree(source / "records", build / "records")

    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    schema["ca_split"] = {
        "window_manifest": "ca_window_manifest.csv",
        "group_manifest": "ca_group_manifest.csv",
        "event_manifest": "ca_fog_event_manifest.csv",
        "summary": "ca_split_summary.csv",
        "quality_report": "ca_quality_report.json",
        "split_codes": SPLIT_CODES,
        "group_kind_codes": GROUP_KIND_CODES,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "fog_definition": "FOG when at least 64 of 128 samples (>=1 second) are FOG",
        "target_split_fractions": TARGETS,
        "assignment_unit": "complete nearby FOG-event cluster or contiguous clean Non-FOG block",
        "clean_block_max_seconds": CLEAN_BLOCK_SECONDS,
        "clean_fog_guard_seconds_each_side": CLEAN_FOG_GUARD / FS,
        "inter_split_embargo_seconds": INTER_SPLIT_EMBARGO / FS,
    }
    base.write_json(build / "schema.json", schema)
    base.write_csv(build / "ca_window_manifest.csv", rows)
    base.write_csv(build / "ca_group_manifest.csv", group_rows(groups))
    base.write_csv(build / "ca_fog_event_manifest.csv", events)
    base.write_csv(build / "ca_split_summary.csv", summaries)
    base.write_csv(build / "ca_source_summary_audit.csv", source_audit_rows)
    base.write_json(build / "ca_split_codes.json", SPLIT_CODES)
    base.write_json(build / "ca_group_kind_codes.json", GROUP_KIND_CODES)
    base.write_json(build / "ca_quality_report.json", quality)
    subjects = sorted({record.subject_id for record in records})
    save_indices(build, subjects, rows, groups, record_order)

    protocol = {
        "dataset_id": "daphnet_CA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed": str(source),
        "source_manifest_sha256": base.sha256(source / "manifest.csv"),
        "source_fog_events_sha256": base.sha256(source / "fog_events.csv"),
        "reference_summary": str(summary_path),
        "reference_summary_sha256": base.sha256(summary_path),
        "subjects": subjects,
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "window_label": "FOG iff the 2 s window contains at least 64 FOG samples (>=1 s)",
        "split_targets": TARGETS,
        "split_interpretation": "fractions of window records within each subject",
        "random_window_split": False,
        "fog_assignment": "complete event clusters; events closer than 9 s remain together",
        "clean_assignment": "contiguous clean blocks of at most 60 s",
        "transition_policy": "windows overlapping an event but containing <64 FOG samples are NONFOG transition windows and follow the event cluster split",
        "clean_fog_guard_seconds_each_side": CLEAN_FOG_GUARD / FS,
        "inter_split_embargo_seconds": INTER_SPLIT_EMBARGO / FS,
    }
    base.write_json(build / "ca_protocol.json", protocol)
    (build / "README_CA.md").write_text(
        "# Daphnet processed_CA\n\n"
        "This directory preserves the canonical continuous NPZ records and adds a leakage-safe "
        "within-subject CA train/validation/test window split.\n\n"
        "- Window: 2 s (128 samples), stride 1 s (64 samples).\n"
        "- Label: FOG when at least 64 samples (1 s) in the full window are FOG.\n"
        "- Split target: train/validation/test = 65%/15%/20% of window records.\n"
        "- Assignment is deterministic and group based; no random window split is used.\n"
        "- Complete nearby FOG events remain together; clean Non-FOG is assigned in contiguous blocks.\n"
        "- A five-second cross-split embargo is enforced.\n\n"
        "Authoritative files: `ca_window_manifest.csv`, `ca_group_manifest.csv`, "
        "`ca_fog_event_manifest.csv`, `ca_split_summary.csv`, and `ca_quality_report.json`.\n",
        encoding="utf-8",
    )
    build.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
