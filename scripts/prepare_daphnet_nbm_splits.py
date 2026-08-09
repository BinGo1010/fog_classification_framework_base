"""Build the strict-purity Daphnet split used by the NBM/classifier pipeline.

The canonical continuous records are preserved.  Window candidates are always
anchored to the beginning of each record on a 1 s grid.  A 2 s candidate is
retained only when all labels agree; mixed candidates are audited and removed
before any allocation.  The sample count is inferred from the source manifest,
so the same protocol supports both the canonical 64 Hz records (128/64 samples)
and the FIR-downsampled 32 Hz records (64/32 samples).

Allocation is deterministic and within subject.  Complete FoG event clusters
and at-most-60-second clean Non-FoG blocks are the indivisible allocation
groups.  A permanent class-wise 20% test pool is frozen first.  The remaining
groups are assigned to three rotating development folds.  In each outer fold,
the held development fold is external validation and the other two folds are
training.  Clean training groups are further assigned disjoint NBM-weight,
NBM-early-stop, and classifier-training roles.

Clean runs longer than 60 seconds have a single connector window between
neighboring allocation blocks.  The connector is retained in an outer fold
only when both neighboring blocks have the same final role; otherwise it is
reported as ``excluded_cross_pool_boundary``.  This retains every safe
connector without allowing raw-sample overlap between different active roles.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_a5_splits as base  # noqa: E402
import prepare_daphnet_ca_splits as ca  # noqa: E402


DATASET_ID = "daphnet_NBM"
FS = 64
SOURCE_FS = 64
WINDOW = 128
STRIDE = 64
OUTER_FOLDS = 3
MAX_CLEAN_BLOCK_SECONDS = 60
MAX_CLEAN_BLOCK_SAMPLES = MAX_CLEAN_BLOCK_SECONDS * FS
MAX_SUBJECT_RATIO_ERROR_PERCENTAGE_POINTS = 8.0
MAX_AGGREGATE_RATIO_ERROR_PERCENTAGE_POINTS = 1.25
# Strict-purity windows are contained inside one annotated FoG event, so no
# legacy 5 s event guard is needed here.  Overlapping annotations (if any) are
# still merged by ``cluster_record_events``; otherwise every complete event is
# an indivisible singleton cluster.
EVENT_CLUSTER_GAP = 0

EXPECTED_SOURCE_INVENTORIES = {
    64: {
        "candidate_windows": 17_790,
        "pure_nonfog_windows": 15_593,
        "pure_fog_windows": 1_301,
        "mixed_windows": 896,
        "summary_rows": 35,
    },
    32: {
        "candidate_windows": 17_790,
        "pure_nonfog_windows": 15_596,
        "pure_fog_windows": 1_307,
        "mixed_windows": 887,
        "summary_rows": 35,
    },
}
EXPECTED_SOURCE_INVENTORY = dict(EXPECTED_SOURCE_INVENTORIES[FS])

CLASS_NONFOG = "NONFOG"
CLASS_FOG = "FOG"
PURE_NONFOG = "PURE_NONFOG"
PURE_FOG = "PURE_FOG"
MIXED = "MIXED_BOUNDARY"

PERMANENT_TEST_NONFOG = "permanent_test_nonfog"
PERMANENT_TEST_FOG = "permanent_test_fog"
EXTERNAL_VALIDATION_NONFOG = "external_validation_nonfog"
EXTERNAL_VALIDATION_FOG = "external_validation_fog"
NBM_TRAIN_CLEAN = "nbm_train_clean"
NBM_EARLYSTOP_CLEAN = "nbm_earlystop_clean"
CLASSIFIER_TRAIN_CLEAN = "classifier_train_clean"
CLASSIFIER_TRAIN_FOG = "classifier_train_fog"
EXCLUDED_BOUNDARY = "excluded_cross_pool_boundary"
EXCLUDED_NO_PURE_FOG = "excluded_no_pure_fog_window"

ACTIVE_ROLES = (
    PERMANENT_TEST_NONFOG,
    PERMANENT_TEST_FOG,
    EXTERNAL_VALIDATION_NONFOG,
    EXTERNAL_VALIDATION_FOG,
    NBM_TRAIN_CLEAN,
    NBM_EARLYSTOP_CLEAN,
    CLASSIFIER_TRAIN_CLEAN,
    CLASSIFIER_TRAIN_FOG,
)
ROLE_CODES = {role: index for index, role in enumerate(ACTIVE_ROLES)}

FULL_INVENTORY_TARGETS = {
    PERMANENT_TEST_NONFOG: 1.0 / 5.0,
    PERMANENT_TEST_FOG: 1.0 / 5.0,
    EXTERNAL_VALIDATION_NONFOG: 4.0 / 15.0,
    EXTERNAL_VALIDATION_FOG: 4.0 / 15.0,
    NBM_TRAIN_CLEAN: 0.256,
    NBM_EARLYSTOP_CLEAN: 0.064,
    CLASSIFIER_TRAIN_CLEAN: 16.0 / 75.0,
    CLASSIFIER_TRAIN_FOG: 8.0 / 15.0,
}
ROLE_CLASS = {
    PERMANENT_TEST_NONFOG: CLASS_NONFOG,
    PERMANENT_TEST_FOG: CLASS_FOG,
    EXTERNAL_VALIDATION_NONFOG: CLASS_NONFOG,
    EXTERNAL_VALIDATION_FOG: CLASS_FOG,
    NBM_TRAIN_CLEAN: CLASS_NONFOG,
    NBM_EARLYSTOP_CLEAN: CLASS_NONFOG,
    CLASSIFIER_TRAIN_CLEAN: CLASS_NONFOG,
    CLASSIFIER_TRAIN_FOG: CLASS_FOG,
}

WINDOW_MANIFEST = "nbm_window_manifest.csv"
GROUP_MANIFEST = "nbm_group_manifest.csv"
FOLD_GROUP_ROLES = "nbm_fold_group_roles.csv"
CONNECTOR_MANIFEST = "nbm_connector_manifest.csv"
FOLD_ALIGNMENT_MANIFEST = "nbm_subject_fold_alignment.csv"
EXCLUDED_AUDIT = "nbm_excluded_window_audit.csv"
SPLIT_SUMMARY = "nbm_split_summary.csv"
POOL_COUNT_REPORT = "NBM_POOL_COUNT_REPORT.md"
SOURCE_SUMMARY_COPY = "nbm_source_fog_segment_summary.csv"
SOURCE_SUMMARY_AUDIT = "nbm_source_summary_audit.csv"
QUALITY_REPORT = "nbm_quality_report.json"
PROTOCOL = "nbm_protocol.json"
ROLE_CODES_JSON = "nbm_role_codes.json"


def configure_sampling(
    sampling_rate_hz: int,
    *,
    source_sampling_rate_hz: int | None = None,
    source_dataset_id: str = "daphnet",
) -> None:
    """Configure the module-wide temporal grid before any split work."""

    global DATASET_ID, FS, SOURCE_FS, WINDOW, STRIDE, MAX_CLEAN_BLOCK_SAMPLES
    global EXPECTED_SOURCE_INVENTORY

    rate = int(sampling_rate_hz)
    source_rate = rate if source_sampling_rate_hz is None else int(source_sampling_rate_hz)
    if rate <= 0 or source_rate <= 0:
        raise ValueError("sampling rates must be positive")
    if source_rate % rate != 0:
        raise ValueError(
            "source sampling rate must be an integer multiple of the processed rate"
        )
    if rate not in EXPECTED_SOURCE_INVENTORIES:
        raise ValueError(
            f"unsupported sampling rate {rate}; expected one of "
            f"{sorted(EXPECTED_SOURCE_INVENTORIES)}"
        )
    FS = rate
    SOURCE_FS = source_rate
    WINDOW = 2 * FS
    STRIDE = FS
    MAX_CLEAN_BLOCK_SAMPLES = MAX_CLEAN_BLOCK_SECONDS * FS
    EXPECTED_SOURCE_INVENTORY = dict(EXPECTED_SOURCE_INVENTORIES[FS])
    DATASET_ID = (
        "daphnet_NBM"
        if source_dataset_id == "daphnet" and FS == 64
        else f"{source_dataset_id}_NBM"
    )


def source_offset_for_processed_index(index: int) -> int:
    """Map a processed-record sample boundary to its source-file coordinate."""

    numerator = int(index) * SOURCE_FS
    if numerator % FS != 0:
        raise ValueError(
            f"processed index {index} has no exact source coordinate at {SOURCE_FS}/{FS} Hz"
        )
    return numerator // FS


@dataclass(frozen=True)
class CandidateWindow:
    window_id: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    source_file: str
    source_start_row: int
    start_index: int
    end_index_exclusive: int
    fog_samples_in_2s: int
    purity_label: str

    @property
    def class_label(self) -> str:
        if self.purity_label == PURE_FOG:
            return CLASS_FOG
        if self.purity_label == PURE_NONFOG:
            return CLASS_NONFOG
        return "MIXED"

    @property
    def y_binary(self) -> int:
        return int(self.purity_label == PURE_FOG)


@dataclass
class AllocationGroup:
    group_id: str
    group_kind: str
    class_label: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    start_index: int
    end_index_exclusive: int
    window_ids: tuple[str, ...]
    event_ids: tuple[int, ...] = ()
    eligible_for_allocation: bool = True
    permanent_partition: str = ""
    assigned_development_fold: int | None = None

    @property
    def window_count(self) -> int:
        return len(self.window_ids)


@dataclass(frozen=True)
class CleanConnector:
    connector_id: str
    window_id: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    start_index: int
    end_index_exclusive: int
    left_group_id: str
    right_group_id: str


def parse_args() -> argparse.Namespace:
    dataset_root = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=dataset_root / "processed")
    parser.add_argument("--output", type=Path, default=dataset_root / "processed_NBM")
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "019fd1bb-dca3-7063-b479-4ec71f2425f8"
            / "daphnet_fog_segment_summary.csv"
        ),
        help="User-supplied record-level segment summary used for source auditing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute allocation and quality results without writing the output dataset.",
    )
    return parser.parse_args()


def purity_from_fog_samples(fog_samples: int) -> str:
    value = int(fog_samples)
    if value == 0:
        return PURE_NONFOG
    if value == WINDOW:
        return PURE_FOG
    if not 0 < value < WINDOW:
        raise ValueError(f"fog_samples must be in [0, {WINDOW}], got {value}")
    return MIXED


def consecutive_start_runs(starts: Sequence[int]) -> list[list[int]]:
    if not starts:
        return []
    ordered = [int(value) for value in starts]
    if ordered != sorted(set(ordered)):
        raise ValueError("starts must be sorted and unique")
    output: list[list[int]] = []
    current = [ordered[0]]
    for start in ordered[1:]:
        if start == current[-1] + STRIDE:
            current.append(start)
        else:
            output.append(current)
            current = [start]
    output.append(current)
    return output


def normalize_fog_event_rows(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize event ``end_index`` to the canonical inclusive convention."""

    normalized: list[dict[str, Any]] = []
    convention_counts: Counter[str] = Counter()
    problems: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        start = int(raw["start_index"])
        end = int(raw["end_index"])
        duration_samples = int(round(float(raw["duration_sec"]) * FS))
        if end - start + 1 == duration_samples:
            convention = "inclusive"
            inclusive_end = end
        elif end - start == duration_samples:
            convention = "exclusive_normalized_to_inclusive"
            inclusive_end = end - 1
        else:
            problems.append(
                {
                    "record_id": raw.get("record_id", ""),
                    "event_id": raw.get("event_id", ""),
                    "start_index": start,
                    "end_index": end,
                    "duration_samples": duration_samples,
                }
            )
            continue
        convention_counts[convention] += 1
        row["start_index"] = start
        row["end_index"] = inclusive_end
        row["start_time_sec"] = start / FS
        row["end_time_sec"] = inclusive_end / FS
        row["duration_sec"] = duration_samples / FS
        normalized.append(row)
    audit = {
        "pass": not problems and len(normalized) == len(rows),
        "input_event_count": len(rows),
        "normalized_event_count": len(normalized),
        "input_convention_counts": dict(sorted(convention_counts.items())),
        "output_end_index_convention": "inclusive_last_fog_sample",
        "problem_count": len(problems),
        "examples": problems[:20],
    }
    return normalized, audit


def partition_clean_start_run(starts: Sequence[int]) -> tuple[list[tuple[int, ...]], list[int]]:
    """Split one pure-clean run into non-overlapping cores plus connectors.

    Core blocks span at most 60 seconds.  Neighboring cores are separated by
    exactly one connector candidate.  The two closest core windows then touch
    but do not overlap; the connector overlaps each by one second.
    """

    ordered = [int(value) for value in starts]
    if not ordered:
        return [], []
    if any(right - left != STRIDE for left, right in zip(ordered, ordered[1:])):
        raise ValueError("partition_clean_start_run requires one contiguous stride grid")
    max_core_windows = 1 + (MAX_CLEAN_BLOCK_SAMPLES - WINDOW) // STRIDE
    blocks: list[tuple[int, ...]] = []
    connectors: list[int] = []
    position = 0
    while len(ordered) - position > max_core_windows:
        remaining = len(ordered) - position
        core_count = max_core_windows
        if remaining - core_count == 1:
            core_count -= 1
        if core_count <= 0 or remaining - core_count < 2:
            raise AssertionError("unable to leave a connector and a non-empty right block")
        blocks.append(tuple(ordered[position : position + core_count]))
        connectors.append(ordered[position + core_count])
        position += core_count + 1
    blocks.append(tuple(ordered[position:]))
    if len(connectors) != len(blocks) - 1:
        raise AssertionError("connector/core partition is inconsistent")
    for block in blocks:
        if not block or block[-1] + WINDOW - block[0] > MAX_CLEAN_BLOCK_SAMPLES:
            raise AssertionError("clean core exceeds the 60-second limit")
    return blocks, connectors


def enumerate_candidates(
    records: Sequence[base.Record],
    manifest: Mapping[str, Mapping[str, str]],
) -> list[CandidateWindow]:
    candidates: list[CandidateWindow] = []
    for record in records:
        metadata = manifest[record.record_id]
        source_start = int(metadata["source_start_row"])
        for start in range(0, len(record.y) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            fog_samples = int(np.count_nonzero(record.y[start:end] == 1))
            candidates.append(
                CandidateWindow(
                    window_id=f"{record.record_id}:{start}:{end}",
                    subject_id=record.subject_id,
                    record_id=record.record_id,
                    run_id=record.run_id,
                    segment_id=int(metadata["segment_id"]),
                    source_file=str(metadata["source_file"]),
                    source_start_row=source_start,
                    start_index=start,
                    end_index_exclusive=end,
                    fog_samples_in_2s=fog_samples,
                    purity_label=purity_from_fog_samples(fog_samples),
                )
            )
    return candidates


def source_provenance_audit(
    manifest_rows: Sequence[dict[str, str]],
    records: Sequence[base.Record],
    candidates: Sequence[CandidateWindow],
) -> dict[str, Any]:
    """Validate record and window coordinates in original source-file space."""

    record_lookup = {record.record_id: record for record in records}
    manifest_lookup = {row["record_id"]: row for row in manifest_rows}
    problems: list[dict[str, Any]] = []
    by_source: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row in manifest_rows:
        record_id = str(row["record_id"])
        start = int(row["source_start_row"])
        end_inclusive = int(row["source_end_row"])
        expected_source = end_inclusive - start + 1
        record = record_lookup.get(record_id)
        actual = -1 if record is None else len(record.y)
        declared = int(row["n_samples"])
        declared_source = int(row.get("source_n_samples", declared))
        expected_processed = ((declared_source - 1) * FS) // SOURCE_FS + 1
        if (
            start < 0
            or expected_source != declared_source
            or expected_processed != declared
            or declared != actual
        ):
            problems.append(
                {
                    "kind": "record_interval_length",
                    "record_id": record_id,
                    "source_file": row["source_file"],
                    "source_start_row": start,
                    "source_end_row": end_inclusive,
                    "source_interval_samples": expected_source,
                    "manifest_source_samples": declared_source,
                    "expected_processed_samples": expected_processed,
                    "manifest_processed_samples": declared,
                    "npz_processed_samples": actual,
                }
            )
        by_source[str(row["source_file"])].append(
            (start, end_inclusive + 1, record_id)
        )
    for source_file, intervals in sorted(by_source.items()):
        ordered = sorted(intervals)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] < left[1]:
                problems.append(
                    {
                        "kind": "source_record_overlap",
                        "source_file": source_file,
                        "left_record_id": left[2],
                        "left_interval": [left[0], left[1]],
                        "right_record_id": right[2],
                        "right_interval": [right[0], right[1]],
                    }
                )
    for candidate in candidates:
        row = manifest_lookup[candidate.record_id]
        record_start = int(row["source_start_row"])
        record_end_exclusive = int(row["source_end_row"]) + 1
        absolute_start = candidate.source_start_row + source_offset_for_processed_index(
            candidate.start_index
        )
        absolute_end = candidate.source_start_row + source_offset_for_processed_index(
            candidate.end_index_exclusive
        )
        expected_source_window = source_offset_for_processed_index(WINDOW)
        if (
            candidate.source_start_row != record_start
            or absolute_start < record_start
            or absolute_end > record_end_exclusive
            or absolute_end - absolute_start != expected_source_window
        ):
            problems.append(
                {
                    "kind": "window_absolute_interval",
                    "window_id": candidate.window_id,
                    "record_id": candidate.record_id,
                    "source_file": candidate.source_file,
                    "window_interval": [absolute_start, absolute_end],
                    "record_interval": [record_start, record_end_exclusive],
                }
            )
            if len(problems) >= 100:
                break
    return {
        "pass": not problems,
        "record_count": len(manifest_rows),
        "source_file_count": len(by_source),
        "candidate_window_count": len(candidates),
        "problem_count": len(problems),
        "examples": problems[:20],
    }


def cluster_record_events(events: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        events,
        key=lambda event: (int(event["start_index"]), int(event["event_id"])),
    )
    if not ordered:
        return []
    clusters: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    current_end = int(ordered[0]["end_index_exclusive"])
    for event in ordered[1:]:
        gap = int(event["start_index"]) - current_end
        if gap < EVENT_CLUSTER_GAP:
            current.append(event)
            current_end = max(current_end, int(event["end_index_exclusive"]))
        else:
            clusters.append(current)
            current = [event]
            current_end = int(event["end_index_exclusive"])
    clusters.append(current)
    return clusters


def build_fog_groups(
    candidates: Sequence[CandidateWindow],
    events_by_record: Mapping[str, Sequence[dict[str, Any]]],
) -> tuple[list[AllocationGroup], dict[str, str], dict[tuple[str, int], str]]:
    pure_by_record: dict[str, list[CandidateWindow]] = defaultdict(list)
    for candidate in candidates:
        if candidate.purity_label == PURE_FOG:
            pure_by_record[candidate.record_id].append(candidate)

    groups: list[AllocationGroup] = []
    window_to_group: dict[str, str] = {}
    event_to_group: dict[tuple[str, int], str] = {}
    for record_id, events in sorted(events_by_record.items()):
        for cluster_index, cluster in enumerate(cluster_record_events(events)):
            group_id = f"{record_id}_fogcluster{cluster_index:03d}"
            starts = min(int(event["start_index"]) for event in cluster)
            ends = max(int(event["end_index_exclusive"]) for event in cluster)
            selected: list[CandidateWindow] = []
            for candidate in pure_by_record.get(record_id, []):
                containing = [
                    event
                    for event in cluster
                    if int(event["start_index"]) <= candidate.start_index
                    and int(event["end_index_exclusive"]) >= candidate.end_index_exclusive
                ]
                if containing:
                    if len(containing) != 1:
                        raise AssertionError(f"pure FoG window belongs to multiple events: {candidate.window_id}")
                    selected.append(candidate)
            first = cluster[0]
            group = AllocationGroup(
                group_id=group_id,
                group_kind="fog_event_cluster",
                class_label=CLASS_FOG,
                subject_id=str(first["subject_id"]),
                record_id=record_id,
                run_id=str(first["run_id"]),
                segment_id=int(first["segment_id"]),
                start_index=starts,
                end_index_exclusive=ends,
                window_ids=tuple(
                    candidate.window_id
                    for candidate in sorted(selected, key=lambda item: item.start_index)
                ),
                event_ids=tuple(int(event["event_id"]) for event in cluster),
                eligible_for_allocation=bool(selected),
            )
            groups.append(group)
            for event in cluster:
                event_to_group[(record_id, int(event["event_id"]))] = group_id
            for candidate in selected:
                if candidate.window_id in window_to_group:
                    raise AssertionError(f"duplicate pure FoG assignment: {candidate.window_id}")
                window_to_group[candidate.window_id] = group_id

    expected = {candidate.window_id for candidate in candidates if candidate.purity_label == PURE_FOG}
    if set(window_to_group) != expected:
        missing = sorted(expected - set(window_to_group))[:10]
        extra = sorted(set(window_to_group) - expected)[:10]
        raise RuntimeError(f"FoG event grouping mismatch; missing={missing}, extra={extra}")
    return groups, window_to_group, event_to_group


def build_clean_groups(
    candidates: Sequence[CandidateWindow],
) -> tuple[list[AllocationGroup], list[CleanConnector], dict[str, str]]:
    by_record: dict[str, list[CandidateWindow]] = defaultdict(list)
    lookup = {candidate.window_id: candidate for candidate in candidates}
    for candidate in candidates:
        if candidate.purity_label == PURE_NONFOG:
            by_record[candidate.record_id].append(candidate)

    groups: list[AllocationGroup] = []
    connectors: list[CleanConnector] = []
    window_to_group: dict[str, str] = {}
    for record_id, selected in sorted(by_record.items()):
        selected = sorted(selected, key=lambda item: item.start_index)
        by_start = {candidate.start_index: candidate for candidate in selected}
        block_number = 0
        connector_number = 0
        for run in consecutive_start_runs([candidate.start_index for candidate in selected]):
            core_blocks, connector_starts = partition_clean_start_run(run)
            run_groups: list[AllocationGroup] = []
            for core in core_blocks:
                first = by_start[core[0]]
                group_id = f"{record_id}_cleanblock{block_number:03d}"
                block_number += 1
                group = AllocationGroup(
                    group_id=group_id,
                    group_kind="clean_nonfog_block",
                    class_label=CLASS_NONFOG,
                    subject_id=first.subject_id,
                    record_id=record_id,
                    run_id=first.run_id,
                    segment_id=first.segment_id,
                    start_index=core[0],
                    end_index_exclusive=core[-1] + WINDOW,
                    window_ids=tuple(by_start[start].window_id for start in core),
                )
                groups.append(group)
                run_groups.append(group)
                for window_id in group.window_ids:
                    if window_id in window_to_group:
                        raise AssertionError(f"duplicate clean core assignment: {window_id}")
                    window_to_group[window_id] = group_id
            for index, start in enumerate(connector_starts):
                candidate = by_start[start]
                connector = CleanConnector(
                    connector_id=f"{record_id}_cleanconnector{connector_number:03d}",
                    window_id=candidate.window_id,
                    subject_id=candidate.subject_id,
                    record_id=record_id,
                    run_id=candidate.run_id,
                    segment_id=candidate.segment_id,
                    start_index=start,
                    end_index_exclusive=start + WINDOW,
                    left_group_id=run_groups[index].group_id,
                    right_group_id=run_groups[index + 1].group_id,
                )
                connector_number += 1
                connectors.append(connector)

    connector_ids = {connector.window_id for connector in connectors}
    expected = {
        candidate.window_id for candidate in candidates if candidate.purity_label == PURE_NONFOG
    }
    if set(window_to_group) & connector_ids:
        raise AssertionError("clean core and connector windows overlap by identity")
    if set(window_to_group) | connector_ids != expected:
        missing = sorted(expected - (set(window_to_group) | connector_ids))[:10]
        raise RuntimeError(f"clean grouping does not reconcile; missing={missing}")
    if any(window_id not in lookup for window_id in connector_ids):
        raise AssertionError("connector references an unknown candidate")
    return groups, connectors, window_to_group


def effective_label_counts(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[CleanConnector],
    assignments: Mapping[str, str],
    labels: Sequence[str],
) -> dict[str, int]:
    selected_ids = {group.group_id for group in groups}
    counts = {label: 0 for label in labels}
    for group in groups:
        counts[assignments[group.group_id]] += group.window_count
    for connector in connectors:
        if connector.left_group_id not in selected_ids or connector.right_group_id not in selected_ids:
            continue
        left = assignments[connector.left_group_id]
        right = assignments[connector.right_group_id]
        if left == right:
            counts[left] += 1
    return counts


def normalized_target_error(counts: Mapping[str, int], targets: Mapping[str, float]) -> float:
    return float(
        sum(
            ((float(counts[label]) - float(target)) / max(float(target), 1.0)) ** 2
            for label, target in targets.items()
        )
    )


def optimize_group_labels(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[CleanConnector],
    targets: Mapping[str, float],
    minimum_group_counts: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, int], float]:
    """Deterministically optimize whole-group labels by retained window count."""

    labels = tuple(targets)
    ordered = sorted(groups, key=lambda group: (-group.window_count, group.group_id))
    required = sum(int(minimum_group_counts.get(label, 0)) for label in labels)
    if len(ordered) < required:
        raise ValueError(f"need at least {required} groups for {labels}, got {len(ordered)}")
    if any(group.window_count <= 0 for group in ordered):
        raise ValueError("only positive-window groups may be optimized")

    assignments: dict[str, str] = {}
    core_counts = {label: 0 for label in labels}
    for group in ordered:
        choices = []
        for label in labels:
            trial = dict(core_counts)
            trial[label] += group.window_count
            choices.append(
                (
                    normalized_target_error(trial, targets),
                    -float(targets[label]),
                    labels.index(label),
                    label,
                )
            )
        label = min(choices)[-1]
        assignments[group.group_id] = label
        core_counts[label] += group.window_count

    def group_counts(mapping: Mapping[str, str]) -> Counter[str]:
        return Counter(mapping.values())

    # Repair minimum group-count constraints with the least costly moves.
    while True:
        current_groups = group_counts(assignments)
        missing = [
            label
            for label in labels
            if current_groups[label] < int(minimum_group_counts.get(label, 0))
        ]
        if not missing:
            break
        target_label = missing[0]
        best: tuple[float, str] | None = None
        for group in sorted(ordered, key=lambda item: item.group_id):
            source_label = assignments[group.group_id]
            if current_groups[source_label] <= int(minimum_group_counts.get(source_label, 0)):
                continue
            trial = dict(assignments)
            trial[group.group_id] = target_label
            score = normalized_target_error(
                effective_label_counts(ordered, connectors, trial, labels), targets
            )
            candidate = (score, group.group_id)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError(f"cannot satisfy minimum counts for {target_label}")
        assignments[best[1]] = target_label

    current_counts = effective_label_counts(ordered, connectors, assignments, labels)
    current_score = normalized_target_error(current_counts, targets)

    # Coordinate descent using whole-group moves and swaps.  Connector counts
    # are recomputed for every trial, so boundary retention participates in the
    # objective instead of being a post-hoc approximation.
    for _ in range(max(20, len(ordered) * 4)):
        counts_by_label = group_counts(assignments)
        best_move: tuple[float, str, str] | None = None
        for group in sorted(ordered, key=lambda item: item.group_id):
            source_label = assignments[group.group_id]
            if counts_by_label[source_label] <= int(minimum_group_counts.get(source_label, 0)):
                continue
            for target_label in labels:
                if target_label == source_label:
                    continue
                trial = dict(assignments)
                trial[group.group_id] = target_label
                score = normalized_target_error(
                    effective_label_counts(ordered, connectors, trial, labels), targets
                )
                candidate = (score, group.group_id, target_label)
                if score + 1e-12 < current_score and (
                    best_move is None or candidate < best_move
                ):
                    best_move = candidate
        if best_move is not None:
            current_score, group_id, target_label = best_move
            assignments[group_id] = target_label
            current_counts = effective_label_counts(ordered, connectors, assignments, labels)
            continue

        best_swap: tuple[float, str, str] | None = None
        by_id = sorted(ordered, key=lambda item: item.group_id)
        for left_index, left in enumerate(by_id):
            for right in by_id[left_index + 1 :]:
                if assignments[left.group_id] == assignments[right.group_id]:
                    continue
                trial = dict(assignments)
                trial[left.group_id], trial[right.group_id] = (
                    trial[right.group_id],
                    trial[left.group_id],
                )
                score = normalized_target_error(
                    effective_label_counts(ordered, connectors, trial, labels), targets
                )
                candidate = (score, left.group_id, right.group_id)
                if score + 1e-12 < current_score and (
                    best_swap is None or candidate < best_swap
                ):
                    best_swap = candidate
        if best_swap is None:
            break
        current_score, left_id, right_id = best_swap
        assignments[left_id], assignments[right_id] = (
            assignments[right_id],
            assignments[left_id],
        )
        current_counts = effective_label_counts(ordered, connectors, assignments, labels)

    return assignments, current_counts, float(current_score)


def allocate_subject_groups(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[CleanConnector],
    candidates: Sequence[CandidateWindow],
) -> tuple[dict[tuple[int, str], str], list[dict[str, Any]]]:
    """Freeze permanent test/folds and assign clean train roles per outer fold."""

    final_clean_roles: dict[tuple[int, str], str] = {}
    allocation_audit: list[dict[str, Any]] = []
    eligible = [group for group in groups if group.eligible_for_allocation]
    subjects = sorted({candidate.subject_id for candidate in candidates})
    inventory = Counter(
        (candidate.subject_id, candidate.class_label)
        for candidate in candidates
        if candidate.purity_label != MIXED
    )

    for subject in subjects:
        subject_connectors = [item for item in connectors if item.subject_id == subject]
        for class_label in (CLASS_NONFOG, CLASS_FOG):
            selected = [
                group
                for group in eligible
                if group.subject_id == subject and group.class_label == class_label
            ]
            total_windows = int(inventory[(subject, class_label)])
            if not selected:
                if total_windows:
                    raise RuntimeError(f"{subject} {class_label} has windows but no groups")
                continue
            relevant_connectors = subject_connectors if class_label == CLASS_NONFOG else []
            permanent, counts, score = optimize_group_labels(
                selected,
                relevant_connectors,
                targets={
                    "development": 0.80 * total_windows,
                    "permanent_test": 0.20 * total_windows,
                },
                minimum_group_counts={"development": 3, "permanent_test": 1},
            )
            for group in selected:
                group.permanent_partition = permanent[group.group_id]
            allocation_audit.append(
                {
                    "subject_id": subject,
                    "class_label": class_label,
                    "allocation_stage": "permanent_test",
                    "outer_fold_id": "",
                    "targets": json.dumps(
                        {"development": 0.80 * total_windows, "permanent_test": 0.20 * total_windows},
                        sort_keys=True,
                    ),
                    "actual_counts": json.dumps(counts, sort_keys=True),
                    "normalized_squared_error": score,
                }
            )

            development = [group for group in selected if group.permanent_partition == "development"]
            development_ids = {group.group_id for group in development}
            dev_connectors = [
                item
                for item in relevant_connectors
                if item.left_group_id in development_ids and item.right_group_id in development_ids
            ]
            fold_labels = tuple(f"fold{index}" for index in range(OUTER_FOLDS))
            folds, counts, score = optimize_group_labels(
                development,
                dev_connectors,
                targets={label: (4.0 / 15.0) * total_windows for label in fold_labels},
                minimum_group_counts={label: 1 for label in fold_labels},
            )
            for group in development:
                group.assigned_development_fold = int(folds[group.group_id].replace("fold", ""))
            allocation_audit.append(
                {
                    "subject_id": subject,
                    "class_label": class_label,
                    "allocation_stage": "development_three_folds",
                    "outer_fold_id": "",
                    "targets": json.dumps(
                        {label: (4.0 / 15.0) * total_windows for label in fold_labels},
                        sort_keys=True,
                    ),
                    "actual_counts": json.dumps(counts, sort_keys=True),
                    "normalized_squared_error": score,
                }
            )

        clean_total = int(inventory[(subject, CLASS_NONFOG)])
        clean_groups = [
            group
            for group in eligible
            if group.subject_id == subject and group.class_label == CLASS_NONFOG
        ]
        if not clean_groups:
            continue
        for outer_fold in range(OUTER_FOLDS):
            training = [
                group
                for group in clean_groups
                if group.permanent_partition == "development"
                and group.assigned_development_fold != outer_fold
            ]
            training_ids = {group.group_id for group in training}
            train_connectors = [
                item
                for item in subject_connectors
                if item.left_group_id in training_ids and item.right_group_id in training_ids
            ]
            assignments, counts, score = optimize_group_labels(
                training,
                train_connectors,
                targets={
                    NBM_TRAIN_CLEAN: 0.256 * clean_total,
                    NBM_EARLYSTOP_CLEAN: 0.064 * clean_total,
                    CLASSIFIER_TRAIN_CLEAN: (16.0 / 75.0) * clean_total,
                },
                minimum_group_counts={
                    NBM_TRAIN_CLEAN: 1,
                    NBM_EARLYSTOP_CLEAN: 1,
                    CLASSIFIER_TRAIN_CLEAN: 1,
                },
            )
            for group in training:
                final_clean_roles[(outer_fold, group.group_id)] = assignments[group.group_id]
            allocation_audit.append(
                {
                    "subject_id": subject,
                    "class_label": CLASS_NONFOG,
                    "allocation_stage": "clean_training_roles",
                    "outer_fold_id": outer_fold,
                    "targets": json.dumps(
                        {
                            NBM_TRAIN_CLEAN: 0.256 * clean_total,
                            NBM_EARLYSTOP_CLEAN: 0.064 * clean_total,
                            CLASSIFIER_TRAIN_CLEAN: (16.0 / 75.0) * clean_total,
                        },
                        sort_keys=True,
                    ),
                    "actual_counts": json.dumps(counts, sort_keys=True),
                    "normalized_squared_error": score,
                }
            )
    return final_clean_roles, allocation_audit


def align_subject_outer_fold_labels(
    groups: Sequence[AllocationGroup],
    clean_training_roles: Mapping[tuple[int, str], str],
    allocation_audit: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, str], str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Align each subject's arbitrary fold labels to improve aggregate FoG balance.

    The within-subject partitions are already frozen at this point.  This step
    may only permute their labels, and applies one permutation consistently to
    every class and role of that subject.  Thus no group membership changes and
    every development group still validates once and trains twice.
    """

    identity = tuple(range(OUTER_FOLDS))
    all_permutations = tuple(permutations(range(OUTER_FOLDS)))
    subjects = sorted({group.subject_id for group in groups})
    fog_counts_by_subject: dict[str, tuple[int, ...]] = {}
    for subject in subjects:
        fog_counts_by_subject[subject] = tuple(
            sum(
                group.window_count
                for group in groups
                if group.subject_id == subject
                and group.class_label == CLASS_FOG
                and group.eligible_for_allocation
                and group.permanent_partition == "development"
                and group.assigned_development_fold == fold
            )
            for fold in range(OUTER_FOLDS)
        )

    total_fog_inventory = sum(
        group.window_count
        for group in groups
        if group.class_label == CLASS_FOG and group.eligible_for_allocation
    )
    development_fog_total = sum(sum(counts) for counts in fog_counts_by_subject.values())

    def aggregate_error(validation_counts: tuple[int, ...]) -> tuple[float, float]:
        if total_fog_inventory <= 0:
            return 0.0, 0.0
        validation_target = (4.0 / 15.0) * total_fog_inventory
        training_target = (8.0 / 15.0) * total_fog_inventory
        errors = [
            100.0 * (count - validation_target) / total_fog_inventory
            for count in validation_counts
        ]
        errors.extend(
            100.0
            * ((development_fog_total - count) - training_target)
            / total_fog_inventory
            for count in validation_counts
        )
        return max(map(abs, errors), default=0.0), sum(error * error for error in errors)

    # DP states are aggregate validation counts.  Equivalent states retain the
    # path with the fewest relabelled fold positions, then the lexical path.
    states: dict[
        tuple[int, ...],
        tuple[int, tuple[tuple[int, ...], ...]],
    ] = {(0,) * OUTER_FOLDS: (0, ())}
    for subject in subjects:
        old_counts = fog_counts_by_subject[subject]
        unique_options: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
        for new_to_old in all_permutations:
            permuted = tuple(old_counts[old_fold] for old_fold in new_to_old)
            moved = sum(new_fold != old_fold for new_fold, old_fold in enumerate(new_to_old))
            candidate = (moved, tuple(new_to_old))
            if permuted not in unique_options or candidate < unique_options[permuted]:
                unique_options[permuted] = candidate
        next_states: dict[
            tuple[int, ...],
            tuple[int, tuple[tuple[int, ...], ...]],
        ] = {}
        for state, (prior_moved, prior_path) in states.items():
            for option_counts, (option_moved, new_to_old) in unique_options.items():
                new_state = tuple(
                    state[fold] + option_counts[fold] for fold in range(OUTER_FOLDS)
                )
                metadata = (prior_moved + option_moved, prior_path + (new_to_old,))
                if new_state not in next_states or metadata < next_states[new_state]:
                    next_states[new_state] = metadata
        states = next_states

    if not states:
        raise RuntimeError("subject fold-label alignment produced no feasible state")
    best_counts, (moved_positions, best_path) = min(
        states.items(),
        key=lambda item: (
            *aggregate_error(item[0]),
            item[1][0],
            item[1][1],
        ),
    )
    mappings = {subject: best_path[index] for index, subject in enumerate(subjects)}
    before_counts = tuple(
        sum(fog_counts_by_subject[subject][fold] for subject in subjects)
        for fold in range(OUTER_FOLDS)
    )

    alignment_rows: list[dict[str, Any]] = []
    old_to_new_by_subject: dict[str, dict[int, int]] = {}
    for subject in subjects:
        new_to_old = mappings[subject]
        old_to_new = {old_fold: new_fold for new_fold, old_fold in enumerate(new_to_old)}
        old_to_new_by_subject[subject] = old_to_new
        old_counts = fog_counts_by_subject[subject]
        new_counts = tuple(old_counts[old_fold] for old_fold in new_to_old)
        alignment_rows.append(
            {
                "subject_id": subject,
                "new_fold0_uses_old_fold": new_to_old[0],
                "new_fold1_uses_old_fold": new_to_old[1],
                "new_fold2_uses_old_fold": new_to_old[2],
                "relabelled_fold_position_count": sum(
                    new_fold != old_fold
                    for new_fold, old_fold in enumerate(new_to_old)
                ),
                "old_fog_validation_counts": json.dumps(old_counts),
                "new_fog_validation_counts": json.dumps(new_counts),
            }
        )

    subject_by_group = {group.group_id: group.subject_id for group in groups}
    realigned_clean_roles: dict[tuple[int, str], str] = {}
    for (old_outer_fold, group_id), role in clean_training_roles.items():
        subject = subject_by_group[group_id]
        new_outer_fold = old_to_new_by_subject[subject][old_outer_fold]
        key = (new_outer_fold, group_id)
        if key in realigned_clean_roles:
            raise AssertionError(f"duplicate clean role after fold alignment: {key}")
        realigned_clean_roles[key] = role

    for group in groups:
        if (
            group.permanent_partition == "development"
            and group.assigned_development_fold is not None
        ):
            group.assigned_development_fold = old_to_new_by_subject[group.subject_id][
                group.assigned_development_fold
            ]

    for row in allocation_audit:
        subject = str(row["subject_id"])
        new_to_old = mappings[subject]
        old_to_new = old_to_new_by_subject[subject]
        if row["allocation_stage"] == "development_three_folds":
            old_counts = json.loads(str(row["actual_counts"]))
            row["actual_counts"] = json.dumps(
                {
                    f"fold{new_fold}": old_counts[f"fold{old_fold}"]
                    for new_fold, old_fold in enumerate(new_to_old)
                },
                sort_keys=True,
            )
        elif row["allocation_stage"] == "clean_training_roles":
            row["outer_fold_id"] = old_to_new[int(row["outer_fold_id"])]

    before_error = aggregate_error(before_counts)
    after_error = aggregate_error(best_counts)
    quality = {
        "pass": after_error <= before_error and len(best_path) == len(subjects),
        "objective": "minimize aggregate FoG max absolute percentage-point error, then SSE",
        "aggregate_fog_validation_counts_before": list(before_counts),
        "aggregate_fog_validation_counts_after": list(best_counts),
        "aggregate_fog_classifier_train_counts_before": [
            development_fog_total - count for count in before_counts
        ],
        "aggregate_fog_classifier_train_counts_after": [
            development_fog_total - count for count in best_counts
        ],
        "max_absolute_error_percentage_points_before": before_error[0],
        "max_absolute_error_percentage_points_after": after_error[0],
        "sum_squared_error_before": before_error[1],
        "sum_squared_error_after": after_error[1],
        "subject_count": len(subjects),
        "subjects_relabelled": [
            subject for subject in subjects if mappings[subject] != identity
        ],
        "relabelled_fold_position_count": moved_positions,
    }
    return realigned_clean_roles, alignment_rows, quality


def core_group_role(
    group: AllocationGroup,
    outer_fold: int,
    clean_training_roles: Mapping[tuple[int, str], str],
) -> str:
    if not group.eligible_for_allocation:
        return EXCLUDED_NO_PURE_FOG
    if group.permanent_partition == "permanent_test":
        return PERMANENT_TEST_FOG if group.class_label == CLASS_FOG else PERMANENT_TEST_NONFOG
    if group.permanent_partition != "development" or group.assigned_development_fold is None:
        raise AssertionError(f"incomplete allocation for {group.group_id}")
    if group.assigned_development_fold == outer_fold:
        return EXTERNAL_VALIDATION_FOG if group.class_label == CLASS_FOG else EXTERNAL_VALIDATION_NONFOG
    if group.class_label == CLASS_FOG:
        return CLASSIFIER_TRAIN_FOG
    return clean_training_roles[(outer_fold, group.group_id)]


def candidate_base_row(candidate: CandidateWindow) -> dict[str, Any]:
    source_start = candidate.source_start_row + source_offset_for_processed_index(
        candidate.start_index
    )
    source_end = candidate.source_start_row + source_offset_for_processed_index(
        candidate.end_index_exclusive
    )
    return {
        "window_id": candidate.window_id,
        "subject_id": candidate.subject_id,
        "record_id": candidate.record_id,
        "run_id": candidate.run_id,
        "segment_id": candidate.segment_id,
        "source_file": candidate.source_file,
        "start_index": candidate.start_index,
        "end_index_exclusive": candidate.end_index_exclusive,
        "start_time_sec": candidate.start_index / FS,
        "end_time_sec": candidate.end_index_exclusive / FS,
        "source_start_row": source_start,
        "source_end_row_exclusive": source_end,
        "fog_samples_in_2s": candidate.fog_samples_in_2s,
        "full_2s_fog_fraction": candidate.fog_samples_in_2s / WINDOW,
        "purity_label": candidate.purity_label,
        "class_label": candidate.class_label,
        "y_binary": candidate.y_binary,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "window_alignment": f"record_start_stride{STRIDE}",
        "label_rule": (
            f"PURE_FOG iff {WINDOW}/{WINDOW} FOG; "
            f"PURE_NONFOG iff 0/{WINDOW} FOG"
        ),
    }


def materialize_manifests(
    candidates: Sequence[CandidateWindow],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[CleanConnector],
    core_window_to_group: Mapping[str, str],
    clean_training_roles: Mapping[tuple[int, str], str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_lookup = {candidate.window_id: candidate for candidate in candidates}
    group_lookup = {group.group_id: group for group in groups}
    connector_by_window = {connector.window_id: connector for connector in connectors}

    group_rows: list[dict[str, Any]] = []
    fold_group_rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: (item.subject_id, item.class_label, item.group_id)):
        group_rows.append(
            {
                "group_id": group.group_id,
                "group_kind": group.group_kind,
                "class_label": group.class_label,
                "subject_id": group.subject_id,
                "record_id": group.record_id,
                "run_id": group.run_id,
                "segment_id": group.segment_id,
                "start_index": group.start_index,
                "end_index_exclusive": group.end_index_exclusive,
                "start_time_sec": group.start_index / FS,
                "end_time_sec": group.end_index_exclusive / FS,
                "core_window_count": group.window_count,
                "event_ids": ";".join(map(str, group.event_ids)),
                "event_count": len(group.event_ids),
                "eligible_for_allocation": group.eligible_for_allocation,
                "permanent_partition": (
                    group.permanent_partition if group.eligible_for_allocation else EXCLUDED_NO_PURE_FOG
                ),
                "assigned_development_fold": (
                    "" if group.assigned_development_fold is None else group.assigned_development_fold
                ),
                "allocation_unit_indivisible": True,
            }
        )
        for outer_fold in range(OUTER_FOLDS):
            role = core_group_role(group, outer_fold, clean_training_roles)
            fold_group_rows.append(
                {
                    "outer_fold_id": outer_fold,
                    "group_id": group.group_id,
                    "group_kind": group.group_kind,
                    "class_label": group.class_label,
                    "subject_id": group.subject_id,
                    "record_id": group.record_id,
                    "permanent_partition": (
                        group.permanent_partition if group.eligible_for_allocation else EXCLUDED_NO_PURE_FOG
                    ),
                    "assigned_development_fold": (
                        "" if group.assigned_development_fold is None else group.assigned_development_fold
                    ),
                    "final_role": role,
                    "active_for_outer_fold": role in ACTIVE_ROLES,
                    "core_window_count": group.window_count,
                }
            )

    connector_rows = [
        {
            "connector_id": connector.connector_id,
            "window_id": connector.window_id,
            "subject_id": connector.subject_id,
            "record_id": connector.record_id,
            "run_id": connector.run_id,
            "segment_id": connector.segment_id,
            "start_index": connector.start_index,
            "end_index_exclusive": connector.end_index_exclusive,
            "start_time_sec": connector.start_index / FS,
            "end_time_sec": connector.end_index_exclusive / FS,
            "left_group_id": connector.left_group_id,
            "right_group_id": connector.right_group_id,
            "retention_policy": "retain per outer fold iff neighboring groups have the same final role",
        }
        for connector in sorted(connectors, key=lambda item: (item.subject_id, item.record_id, item.start_index))
    ]

    window_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.purity_label == MIXED:
            item = candidate_base_row(candidate)
            item.update(
                {
                    "outer_fold_id": "",
                    "exclusion_stage": "purity_filter",
                    "exclusion_reason": "mixed_boundary_window",
                    "left_group_id": "",
                    "right_group_id": "",
                    "left_role": "",
                    "right_role": "",
                }
            )
            excluded_rows.append(item)
            continue

        if candidate.window_id in connector_by_window:
            connector = connector_by_window[candidate.window_id]
            left_group = group_lookup[connector.left_group_id]
            right_group = group_lookup[connector.right_group_id]
            for outer_fold in range(OUTER_FOLDS):
                left_role = core_group_role(left_group, outer_fold, clean_training_roles)
                right_role = core_group_role(right_group, outer_fold, clean_training_roles)
                if left_role == right_role and left_role in ACTIVE_ROLES:
                    item = candidate_base_row(candidate)
                    item.update(
                        {
                            "outer_fold_id": outer_fold,
                            "final_role": left_role,
                            "role_code": ROLE_CODES[left_role],
                            "active_for_outer_fold": True,
                            "allocation_group_id": "",
                            "group_kind": "clean_connector",
                            "connector_id": connector.connector_id,
                            "left_group_id": connector.left_group_id,
                            "right_group_id": connector.right_group_id,
                            "is_dynamic_connector": True,
                        }
                    )
                    window_rows.append(item)
                else:
                    item = candidate_base_row(candidate)
                    item.update(
                        {
                            "outer_fold_id": outer_fold,
                            "exclusion_stage": "cross_pool_boundary",
                            "exclusion_reason": EXCLUDED_BOUNDARY,
                            "connector_id": connector.connector_id,
                            "left_group_id": connector.left_group_id,
                            "right_group_id": connector.right_group_id,
                            "left_role": left_role,
                            "right_role": right_role,
                        }
                    )
                    excluded_rows.append(item)
            continue

        group_id = core_window_to_group[candidate.window_id]
        group = group_lookup[group_id]
        for outer_fold in range(OUTER_FOLDS):
            role = core_group_role(group, outer_fold, clean_training_roles)
            if role not in ACTIVE_ROLES:
                raise AssertionError(f"eligible candidate received inactive role: {candidate.window_id}")
            item = candidate_base_row(candidate)
            item.update(
                {
                    "outer_fold_id": outer_fold,
                    "final_role": role,
                    "role_code": ROLE_CODES[role],
                    "active_for_outer_fold": True,
                    "allocation_group_id": group_id,
                    "group_kind": group.group_kind,
                    "connector_id": "",
                    "left_group_id": "",
                    "right_group_id": "",
                    "is_dynamic_connector": False,
                }
            )
            window_rows.append(item)

    window_rows.sort(
        key=lambda row: (
            row["subject_id"],
            int(row["outer_fold_id"]),
            ROLE_CODES[row["final_role"]],
            row["record_id"],
            int(row["start_index"]),
        )
    )
    excluded_rows.sort(
        key=lambda row: (
            row["subject_id"],
            -1 if row["outer_fold_id"] == "" else int(row["outer_fold_id"]),
            row["record_id"],
            int(row["start_index"]),
        )
    )
    return window_rows, group_rows, fold_group_rows, connector_rows, excluded_rows


def cross_role_overlap_audit(window_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_partition: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        by_partition[
            (row["subject_id"], row["source_file"], int(row["outer_fold_id"]))
        ].append(row)
    violations: list[dict[str, Any]] = []
    overlapping_same_role_pairs = 0
    for (subject, source_file, outer_fold), rows in sorted(by_partition.items()):
        ordered = sorted(
            rows,
            key=lambda row: (int(row["source_start_row"]), row["window_id"]),
        )
        for index, left in enumerate(ordered):
            left_end = int(left["source_end_row_exclusive"])
            for right in ordered[index + 1 :]:
                if int(right["source_start_row"]) >= left_end:
                    break
                if left["final_role"] == right["final_role"]:
                    overlapping_same_role_pairs += 1
                    continue
                violations.append(
                    {
                        "subject_id": subject,
                        "source_file": source_file,
                        "outer_fold_id": outer_fold,
                        "left_window_id": left["window_id"],
                        "left_role": left["final_role"],
                        "right_window_id": right["window_id"],
                        "right_role": right["final_role"],
                    }
                )
    return {
        "pass": not violations,
        "cross_role_raw_overlap_pair_count": len(violations),
        "same_role_overlap_pair_count": overlapping_same_role_pairs,
        "examples": violations[:20],
    }


def reference_summary_audit(
    summary_rows: Sequence[dict[str, str]],
    manifest_rows: Sequence[dict[str, str]],
    records: Sequence[base.Record],
    events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    summary = {row["record_id"]: row for row in summary_rows}
    manifest = {row["record_id"]: row for row in manifest_rows}
    record_lookup = {record.record_id: record for record in records}
    events_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_record[str(event["record_id"])].append(event)

    output: list[dict[str, Any]] = []
    all_ids = sorted(set(summary) | set(manifest) | set(record_lookup))
    for record_id in all_ids:
        source = summary.get(record_id)
        metadata = manifest.get(record_id)
        record = record_lookup.get(record_id)
        if source is None or metadata is None or record is None:
            output.append(
                {
                    "record_id": record_id,
                    "status": "MISSING_COMPONENT",
                    "in_summary": source is not None,
                    "in_manifest": metadata is not None,
                    "in_npz": record is not None,
                }
            )
            continue
        selected_events = sorted(
            events_by_record.get(record_id, []), key=lambda row: int(row["start_index"])
        )
        actual_event_durations = [float(row["duration_sec"]) for row in selected_events]
        text = str(source.get("individual_event_durations_sec", "")).strip()
        summary_event_durations = [float(value.strip()) for value in text.split(";") if value.strip()]
        manifest_source_samples = int(
            metadata.get("source_n_samples", metadata["n_samples"])
        )
        expected_processed_samples = (
            ((manifest_source_samples - 1) * FS) // SOURCE_FS + 1
        )
        resampling_tolerance_sec = 0.0 if FS == SOURCE_FS else 1.0 / FS
        event_total_tolerance_sec = resampling_tolerance_sec * max(
            1, len(selected_events)
        )
        summary_fog_duration_from_samples = int(source["fog_samples"]) / SOURCE_FS
        processed_fog_duration_from_samples = int(np.sum(record.y == 1)) / FS
        checks = {
            "subject_match": str(source["subject_id"]) == str(metadata["subject_id"]),
            "source_file_match": str(source["source_file"]) == str(metadata["source_file"]),
            "run_id_match": str(source["run_id"]) == str(metadata["run_id"]),
            "segment_id_match": int(source["segment_id"]) == int(metadata["segment_id"]),
            "segment_samples_match": (
                int(source["segment_samples"]) == manifest_source_samples
                and len(record.y) == int(metadata["n_samples"])
                and len(record.y) == expected_processed_samples
            ),
            "segment_duration_match": abs(
                float(source["segment_duration_sec"])
                - manifest_source_samples / SOURCE_FS
            ) <= 1e-6,
            "fog_samples_match": abs(
                summary_fog_duration_from_samples - processed_fog_duration_from_samples
            ) <= event_total_tolerance_sec + 1e-12,
            "fog_event_count_match": int(source["manifest_fog_event_count"]) == len(selected_events),
            "fog_total_duration_match": abs(
                float(source["fog_total_duration_sec"]) - sum(actual_event_durations)
            ) <= event_total_tolerance_sec + 1e-12,
            "fog_total_duration_from_samples_match": abs(
                float(source["fog_total_duration_sec"])
                - processed_fog_duration_from_samples
            ) <= event_total_tolerance_sec + 1e-12,
            "individual_event_durations_match": (
                len(summary_event_durations) == len(actual_event_durations)
                and all(
                    abs(left - right) <= resampling_tolerance_sec + 1e-12
                    for left, right in zip(summary_event_durations, actual_event_durations)
                )
            ),
        }
        output.append(
            {
                "record_id": record_id,
                "subject_id": record.subject_id,
                "summary_segment_samples": int(source["segment_samples"]),
                "manifest_source_n_samples": manifest_source_samples,
                "manifest_n_samples": int(metadata["n_samples"]),
                "npz_n_samples": len(record.y),
                "summary_fog_samples": int(source["fog_samples"]),
                "npz_fog_samples": int(np.sum(record.y == 1)),
                "summary_fog_duration_from_samples_sec": summary_fog_duration_from_samples,
                "npz_fog_duration_from_samples_sec": processed_fog_duration_from_samples,
                "resampling_tolerance_sec_per_event": resampling_tolerance_sec,
                "event_total_tolerance_sec": event_total_tolerance_sec,
                "summary_fog_event_count": int(source["manifest_fog_event_count"]),
                "manifest_fog_event_count": len(selected_events),
                **checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    passed = bool(
        len(summary_rows) == EXPECTED_SOURCE_INVENTORY["summary_rows"]
        and len(summary) == len(summary_rows)
        and output
        and all(row["status"] == "PASS" for row in output)
    )
    return output, passed


def event_annotation_audit(
    records: Sequence[base.Record], events: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Reconcile every half-open FoG event interval to the NPZ label runs."""

    events_by_record: dict[str, list[tuple[int, int]]] = defaultdict(list)
    duration_failures: list[dict[str, Any]] = []
    for event in events:
        start = int(event["start_index"])
        end = int(event["end_index_exclusive"])
        events_by_record[str(event["record_id"])].append((start, end))
        expected_duration = (end - start) / FS
        if abs(float(event["duration_sec"]) - expected_duration) > 1e-9:
            duration_failures.append(
                {
                    "record_id": event["record_id"],
                    "event_id": int(event["event_id"]),
                    "reported_duration_sec": float(event["duration_sec"]),
                    "interval_duration_sec": expected_duration,
                }
            )
    mismatches: list[dict[str, Any]] = []
    derived_total = 0
    for record in records:
        labels = np.asarray(record.y, dtype=np.int8)
        padded = np.pad(labels == 1, (1, 1), constant_values=False)
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        derived = [
            (int(start), int(end))
            for start, end in zip(transitions[0::2], transitions[1::2])
        ]
        annotated = sorted(events_by_record.get(record.record_id, []))
        derived_total += len(derived)
        if derived != annotated:
            mismatches.append(
                {
                    "record_id": record.record_id,
                    "derived_event_count": len(derived),
                    "annotated_event_count": len(annotated),
                    "derived_intervals": derived[:20],
                    "annotated_intervals": annotated[:20],
                }
            )
    annotated_total = sum(len(values) for values in events_by_record.values())
    return {
        "pass": (
            not mismatches
            and not duration_failures
            and derived_total == annotated_total == 237
        ),
        "derived_event_count": derived_total,
        "annotated_event_count": annotated_total,
        "mismatch_count": len(mismatches),
        "duration_mismatch_count": len(duration_failures),
        "examples": mismatches[:10],
        "duration_examples": duration_failures[:10],
    }


def build_split_summary(
    candidates: Sequence[CandidateWindow],
    window_rows: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    subjects = sorted({candidate.subject_id for candidate in candidates})
    inventory = Counter(
        (candidate.subject_id, candidate.class_label)
        for candidate in candidates
        if candidate.purity_label != MIXED
    )
    output: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLDS):
        fold_rows = [row for row in window_rows if int(row["outer_fold_id"]) == outer_fold]
        fold_excluded = [
            row
            for row in excluded_rows
            if row["exclusion_reason"] == EXCLUDED_BOUNDARY
            and int(row["outer_fold_id"]) == outer_fold
        ]
        for subject in subjects + ["ALL"]:
            selected = fold_rows if subject == "ALL" else [row for row in fold_rows if row["subject_id"] == subject]
            excluded = fold_excluded if subject == "ALL" else [row for row in fold_excluded if row["subject_id"] == subject]
            for role in ACTIVE_ROLES:
                class_label = ROLE_CLASS[role]
                role_rows = [row for row in selected if row["final_role"] == role]
                full_inventory = (
                    sum(inventory[(item, class_label)] for item in subjects)
                    if subject == "ALL"
                    else inventory[(subject, class_label)]
                )
                output.append(
                    {
                        "subject_id": subject,
                        "outer_fold_id": outer_fold,
                        "pool_role": role,
                        "class_label": class_label,
                        "window_count": len(role_rows),
                        "allocation_group_count": len(
                            {row["allocation_group_id"] for row in role_rows if row["allocation_group_id"]}
                        ),
                        "dynamic_connector_count": sum(bool(row["is_dynamic_connector"]) for row in role_rows),
                        "full_class_inventory_before_boundary_exclusions": full_inventory,
                        "window_fraction_of_full_class_inventory": (
                            len(role_rows) / full_inventory if full_inventory else ""
                        ),
                        "target_fraction_of_full_class_inventory": FULL_INVENTORY_TARGETS[role],
                        "absolute_target_window_count": (
                            FULL_INVENTORY_TARGETS[role] * full_inventory if full_inventory else ""
                        ),
                        "window_count_deviation_from_target": (
                            len(role_rows) - FULL_INVENTORY_TARGETS[role] * full_inventory
                            if full_inventory
                            else ""
                        ),
                        "percentage_point_deviation_from_target": (
                            100.0
                            * (
                                len(role_rows) / full_inventory
                                - FULL_INVENTORY_TARGETS[role]
                            )
                            if full_inventory
                            else ""
                        ),
                    }
                )
            full_clean = (
                sum(inventory[(item, CLASS_NONFOG)] for item in subjects)
                if subject == "ALL"
                else inventory[(subject, CLASS_NONFOG)]
            )
            output.append(
                {
                    "subject_id": subject,
                    "outer_fold_id": outer_fold,
                    "pool_role": EXCLUDED_BOUNDARY,
                    "class_label": CLASS_NONFOG,
                    "window_count": len(excluded),
                    "allocation_group_count": 0,
                    "dynamic_connector_count": len(excluded),
                    "full_class_inventory_before_boundary_exclusions": full_clean,
                    "window_fraction_of_full_class_inventory": (
                        len(excluded) / full_clean if full_clean else ""
                    ),
                    "target_fraction_of_full_class_inventory": 0.0,
                    "absolute_target_window_count": 0.0,
                    "window_count_deviation_from_target": len(excluded),
                    "percentage_point_deviation_from_target": (
                        100.0 * len(excluded) / full_clean if full_clean else ""
                    ),
                }
            )
    return output


def build_pool_count_report(
    split_summary: Sequence[dict[str, Any]], quality: Mapping[str, Any]
) -> str:
    """Return a compact human-readable aggregate count report."""

    role_labels = {
        PERMANENT_TEST_NONFOG: "永久测试 Non-FoG",
        PERMANENT_TEST_FOG: "永久测试 FoG",
        NBM_TRAIN_CLEAN: "NBM 参数训练 Non-FoG",
        NBM_EARLYSTOP_CLEAN: "NBM 内部早停 Non-FoG",
        CLASSIFIER_TRAIN_CLEAN: "分类器训练 Non-FoG",
        CLASSIFIER_TRAIN_FOG: "分类器训练 FoG",
        EXTERNAL_VALIDATION_NONFOG: "外部验证 Non-FoG",
        EXTERNAL_VALIDATION_FOG: "外部验证 FoG",
        EXCLUDED_BOUNDARY: "跨池边界排除 Non-FoG",
    }
    aggregate = [row for row in split_summary if row["subject_id"] == "ALL"]
    by_key = {
        (int(row["outer_fold_id"]), str(row["pool_role"])): row
        for row in aggregate
    }
    inventory = quality["confirmed_source_inventory"]
    lines = [
        f"# Daphnet {DATASET_ID} 池样本数报告",
        "",
        "## 严格窗口库存",
        "",
        f"- 候选窗口：{int(inventory['candidate_windows']):,}",
        f"- 纯 Non-FoG：{int(inventory['pure_nonfog_windows']):,}",
        f"- 纯 FoG：{int(inventory['pure_fog_windows']):,}",
        f"- 混合边界窗口（删除）：{int(inventory['mixed_windows']):,}",
        "",
        "比例分母为每类完整纯窗口库存；跨池边界排除单列，不重新归一化。",
        "",
    ]
    ordered_roles = (
        PERMANENT_TEST_NONFOG,
        PERMANENT_TEST_FOG,
        NBM_TRAIN_CLEAN,
        NBM_EARLYSTOP_CLEAN,
        CLASSIFIER_TRAIN_CLEAN,
        CLASSIFIER_TRAIN_FOG,
        EXTERNAL_VALIDATION_NONFOG,
        EXTERNAL_VALIDATION_FOG,
        EXCLUDED_BOUNDARY,
    )
    for outer_fold in range(OUTER_FOLDS):
        lines.extend(
            [
                f"## 外层轮转 {outer_fold}",
                "",
                "| 池 | 窗口数 | 实际比例 | 目标比例 | 偏差（百分点） |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for role in ordered_roles:
            row = by_key[(outer_fold, role)]
            actual = float(row["window_fraction_of_full_class_inventory"])
            target = float(row["target_fraction_of_full_class_inventory"])
            deviation = float(row["percentage_point_deviation_from_target"])
            lines.append(
                f"| {role_labels[role]} | {int(row['window_count']):,} | "
                f"{actual:.2%} | {target:.2%} | {deviation:+.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 质量门控",
            "",
            f"- 总体：{'PASS' if quality['overall_pass'] else 'FAIL'}",
            "- 不同池之间原始样本重叠：0",
            "- 激活的混合窗口：0",
            "- FoG 进入 NBM 参数训练/早停：0",
            "- 永久测试在三轮中保持不变：是",
            "",
            "各受试者、各轮、各池的完整明细见 `nbm_split_summary.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


def quality_report(
    candidates: Sequence[CandidateWindow],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[CleanConnector],
    core_window_to_group: Mapping[str, str],
    window_rows: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    source_summary_pass: bool,
    event_audit: Mapping[str, Any],
    provenance_audit: Mapping[str, Any],
    split_summary_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    inventory = Counter(candidate.purity_label for candidate in candidates)
    confirmed = {
        "candidate_windows": len(candidates),
        "pure_nonfog_windows": inventory[PURE_NONFOG],
        "pure_fog_windows": inventory[PURE_FOG],
        "mixed_windows": inventory[MIXED],
        "summary_rows": len(summary_rows),
    }
    inventory_match = confirmed == EXPECTED_SOURCE_INVENTORY
    candidate_ids = {candidate.window_id for candidate in candidates}
    mixed_ids = {candidate.window_id for candidate in candidates if candidate.purity_label == MIXED}
    connector_ids = {connector.window_id for connector in connectors}
    core_ids = set(core_window_to_group)
    source_reconciles = bool(
        len(candidate_ids) == len(candidates)
        and core_ids.isdisjoint(connector_ids)
        and core_ids.isdisjoint(mixed_ids)
        and connector_ids.isdisjoint(mixed_ids)
        and core_ids | connector_ids | mixed_ids == candidate_ids
    )
    overlap = cross_role_overlap_audit(window_rows)
    mixed_active = sum(row["purity_label"] == MIXED for row in window_rows)
    nonpure_active = sum(
        int(row["fog_samples_in_2s"]) not in {0, WINDOW} for row in window_rows
    )
    fog_in_nbm = sum(
        row["class_label"] == CLASS_FOG
        and row["final_role"] in {NBM_TRAIN_CLEAN, NBM_EARLYSTOP_CLEAN}
        for row in window_rows
    )
    duplicate_fold_windows = 0
    per_fold_ids: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in window_rows:
        per_fold_ids[(row["subject_id"], int(row["outer_fold_id"]))][row["window_id"]] += 1
    for counts in per_fold_ids.values():
        duplicate_fold_windows += sum(value - 1 for value in counts.values() if value > 1)

    test_sets: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in window_rows:
        if row["final_role"] in {PERMANENT_TEST_NONFOG, PERMANENT_TEST_FOG}:
            test_sets[(row["subject_id"], row["class_label"], int(row["outer_fold_id"]))].add(
                row["window_id"]
            )
    fixed_test_failures: list[str] = []
    for subject, class_label in sorted({(key[0], key[1]) for key in test_sets}):
        values = [test_sets[(subject, class_label, fold)] for fold in range(OUTER_FOLDS)]
        if not all(value == values[0] for value in values[1:]):
            fixed_test_failures.append(f"{subject}:{class_label}")

    boundary_excluded = [
        row for row in excluded_rows if row["exclusion_reason"] == EXCLUDED_BOUNDARY
    ]
    mixed_excluded = [
        row
        for row in excluded_rows
        if row["exclusion_reason"] == "mixed_boundary_window"
    ]
    mixed_exclusion_audit_complete = bool(
        len(mixed_excluded) == len(mixed_ids)
        and {row["window_id"] for row in mixed_excluded} == mixed_ids
        and all(row["outer_fold_id"] == "" for row in mixed_excluded)
    )
    connector_decisions_complete = len(boundary_excluded) + sum(
        bool(row["is_dynamic_connector"]) for row in window_rows
    ) == len(connectors) * OUTER_FOLDS

    # In every rotation, each pure source candidate is represented exactly
    # once: either in one active role or as one explicitly purged connector.
    pure_ids = candidate_ids - mixed_ids
    fold_candidate_accounting_failures: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLDS):
        active = [
            row for row in window_rows if int(row["outer_fold_id"]) == outer_fold
        ]
        purged = [
            row
            for row in boundary_excluded
            if int(row["outer_fold_id"]) == outer_fold
        ]
        active_ids = {row["window_id"] for row in active}
        purged_ids = {row["window_id"] for row in purged}
        if (
            len(active_ids) != len(active)
            or len(purged_ids) != len(purged)
            or active_ids & purged_ids
            or active_ids | purged_ids != pure_ids
        ):
            fold_candidate_accounting_failures.append(
                {
                    "outer_fold_id": outer_fold,
                    "active_rows": len(active),
                    "active_unique_ids": len(active_ids),
                    "boundary_excluded_rows": len(purged),
                    "boundary_excluded_unique_ids": len(purged_ids),
                    "missing_pure_ids": len(pure_ids - active_ids - purged_ids),
                    "unexpected_ids": len((active_ids | purged_ids) - pure_ids),
                    "active_excluded_overlap": len(active_ids & purged_ids),
                }
            )

    # Audit the role sequence of every indivisible core group independently of
    # the row-level checks.  Development groups validate once and train twice;
    # permanent-test groups remain test-only in all three rotations.
    group_fold_roles: dict[tuple[str, int], set[str]] = defaultdict(set)
    group_fold_window_counts: Counter[tuple[str, int]] = Counter()
    for row in window_rows:
        group_id = str(row.get("allocation_group_id", ""))
        if not group_id:
            continue
        key = (group_id, int(row["outer_fold_id"]))
        group_fold_roles[key].add(str(row["final_role"]))
        group_fold_window_counts[key] += 1
    group_rotation_failures: list[str] = []
    for group in groups:
        if not group.eligible_for_allocation:
            continue
        observed: list[str] = []
        for outer_fold in range(OUTER_FOLDS):
            key = (group.group_id, outer_fold)
            roles = group_fold_roles.get(key, set())
            if len(roles) != 1 or group_fold_window_counts[key] != group.window_count:
                group_rotation_failures.append(
                    f"{group.group_id}:fold{outer_fold}:roles={sorted(roles)}:"
                    f"windows={group_fold_window_counts[key]}/{group.window_count}"
                )
                observed.append("")
            else:
                observed.append(next(iter(roles)))
        if group.permanent_partition == "permanent_test":
            expected = (
                PERMANENT_TEST_FOG
                if group.class_label == CLASS_FOG
                else PERMANENT_TEST_NONFOG
            )
            if any(role != expected for role in observed):
                group_rotation_failures.append(
                    f"{group.group_id}:permanent_test_roles={observed}"
                )
            continue
        validation_role = (
            EXTERNAL_VALIDATION_FOG
            if group.class_label == CLASS_FOG
            else EXTERNAL_VALIDATION_NONFOG
        )
        allowed_training = (
            {CLASSIFIER_TRAIN_FOG}
            if group.class_label == CLASS_FOG
            else {NBM_TRAIN_CLEAN, NBM_EARLYSTOP_CLEAN, CLASSIFIER_TRAIN_CLEAN}
        )
        validation_count = sum(role == validation_role for role in observed)
        training_count = sum(role in allowed_training for role in observed)
        if (
            validation_count != 1
            or training_count != 2
            or group.assigned_development_fold is None
            or observed[group.assigned_development_fold] != validation_role
        ):
            group_rotation_failures.append(
                f"{group.group_id}:fold={group.assigned_development_fold}:roles={observed}"
            )
    active_role_nonempty_failures: list[str] = []
    subjects = sorted({candidate.subject_id for candidate in candidates})
    for subject in subjects:
        has_fog = any(
            candidate.subject_id == subject and candidate.purity_label == PURE_FOG
            for candidate in candidates
        )
        for outer_fold in range(OUTER_FOLDS):
            roles = {
                row["final_role"]
                for row in window_rows
                if row["subject_id"] == subject and int(row["outer_fold_id"]) == outer_fold
            }
            required = {
                PERMANENT_TEST_NONFOG,
                EXTERNAL_VALIDATION_NONFOG,
                NBM_TRAIN_CLEAN,
                NBM_EARLYSTOP_CLEAN,
                CLASSIFIER_TRAIN_CLEAN,
            }
            if has_fog:
                required |= {
                    PERMANENT_TEST_FOG,
                    EXTERNAL_VALIDATION_FOG,
                    CLASSIFIER_TRAIN_FOG,
                }
            for role in sorted(required - roles):
                active_role_nonempty_failures.append(f"{subject}:fold{outer_fold}:{role}")

    ratio_rows = [
        row
        for row in split_summary_rows
        if row["pool_role"] in ACTIVE_ROLES
        and row["percentage_point_deviation_from_target"] != ""
    ]
    subject_ratio_rows = [row for row in ratio_rows if row["subject_id"] != "ALL"]
    aggregate_ratio_rows = [row for row in ratio_rows if row["subject_id"] == "ALL"]
    worst_subject_ratio = max(
        subject_ratio_rows,
        key=lambda row: abs(float(row["percentage_point_deviation_from_target"])),
    )
    worst_aggregate_ratio = max(
        aggregate_ratio_rows,
        key=lambda row: abs(float(row["percentage_point_deviation_from_target"])),
    )
    subject_ratio_max = abs(
        float(worst_subject_ratio["percentage_point_deviation_from_target"])
    )
    aggregate_ratio_max = abs(
        float(worst_aggregate_ratio["percentage_point_deviation_from_target"])
    )
    ratio_gate_pass = bool(
        subject_ratio_max <= MAX_SUBJECT_RATIO_ERROR_PERCENTAGE_POINTS + 1e-12
        and aggregate_ratio_max
        <= MAX_AGGREGATE_RATIO_ERROR_PERCENTAGE_POINTS + 1e-12
    )
    ratio_quality = {
        "pass": ratio_gate_pass,
        "subject_max_absolute_error_percentage_points": subject_ratio_max,
        "subject_tolerance_percentage_points": MAX_SUBJECT_RATIO_ERROR_PERCENTAGE_POINTS,
        "worst_subject_row": dict(worst_subject_ratio),
        "aggregate_max_absolute_error_percentage_points": aggregate_ratio_max,
        "aggregate_tolerance_percentage_points": MAX_AGGREGATE_RATIO_ERROR_PERCENTAGE_POINTS,
        "worst_aggregate_row": dict(worst_aggregate_ratio),
        "note": "FoG event groups are indivisible; S06 sets the attainable subject-level tolerance.",
    }

    report = {
        "overall_pass": False,
        "confirmed_source_inventory": confirmed,
        "expected_source_inventory": EXPECTED_SOURCE_INVENTORY,
        "confirmed_source_inventory_match": inventory_match,
        "source_summary_audit_pass": source_summary_pass,
        "event_annotation_audit": dict(event_audit),
        "source_provenance_audit": dict(provenance_audit),
        "candidate_identity_partition_reconciles": source_reconciles,
        "candidate_count_before_any_cross_pool_boundary_exclusion": len(candidates),
        "pure_count_before_any_cross_pool_boundary_exclusion": inventory[PURE_NONFOG] + inventory[PURE_FOG],
        "mixed_boundary_window_count": inventory[MIXED],
        "mixed_exclusion_audit_complete": mixed_exclusion_audit_complete,
        "allocation_group_count": sum(group.eligible_for_allocation for group in groups),
        "zero_window_fog_event_cluster_count": sum(not group.eligible_for_allocation for group in groups),
        "clean_connector_count": len(connectors),
        "cross_pool_boundary_exclusion_rows_across_outer_folds": len(boundary_excluded),
        "connector_decisions_complete": connector_decisions_complete,
        "fold_candidate_accounting_failures": fold_candidate_accounting_failures,
        "active_window_rows_across_outer_folds": len(window_rows),
        "duplicate_window_rows_within_subject_outer_fold": duplicate_fold_windows,
        "mixed_windows_active": mixed_active,
        "nonpure_windows_active": nonpure_active,
        "fog_windows_in_nbm_fit_or_earlystop": fog_in_nbm,
        "permanent_test_fixed_across_outer_folds_failures": fixed_test_failures,
        "group_rotation_failures": group_rotation_failures,
        "required_active_role_empty_failures": active_role_nonempty_failures,
        "cross_role_raw_overlap_audit": overlap,
        "ratio_quality": ratio_quality,
    }
    report["overall_pass"] = bool(
        inventory_match
        and source_summary_pass
        and bool(event_audit.get("pass"))
        and bool(provenance_audit.get("pass"))
        and source_reconciles
        and mixed_exclusion_audit_complete
        and connector_decisions_complete
        and not fold_candidate_accounting_failures
        and duplicate_fold_windows == 0
        and mixed_active == 0
        and nonpure_active == 0
        and fog_in_nbm == 0
        and not fixed_test_failures
        and not group_rotation_failures
        and not active_role_nonempty_failures
        and overlap["pass"]
        and ratio_gate_pass
    )
    return report


def update_event_manifest(
    events: Sequence[dict[str, Any]],
    event_to_group: Mapping[tuple[str, int], str],
    group_lookup: Mapping[str, AllocationGroup],
    candidates: Sequence[CandidateWindow],
) -> list[dict[str, Any]]:
    pure_fog = [candidate for candidate in candidates if candidate.purity_label == PURE_FOG]
    output: list[dict[str, Any]] = []
    for source in events:
        row = dict(source)
        key = (str(row["record_id"]), int(row["event_id"]))
        group_id = event_to_group[key]
        group = group_lookup[group_id]
        count = sum(
            candidate.record_id == row["record_id"]
            and int(row["start_index"]) <= candidate.start_index
            and int(row["end_index_exclusive"]) >= candidate.end_index_exclusive
            for candidate in pure_fog
        )
        row.update(
            {
                "nbm_event_cluster_id": group_id,
                "nbm_pure_fog_window_count": count,
                "nbm_cluster_pure_fog_window_count": group.window_count,
                "nbm_cluster_status": (
                    "eligible" if group.eligible_for_allocation else EXCLUDED_NO_PURE_FOG
                ),
                "nbm_permanent_partition": (
                    group.permanent_partition if group.eligible_for_allocation else EXCLUDED_NO_PURE_FOG
                ),
                "nbm_assigned_development_fold": (
                    "" if group.assigned_development_fold is None else group.assigned_development_fold
                ),
            }
        )
        output.append(row)
    return output


def save_indices(
    output: Path,
    window_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, str]],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[CleanConnector],
) -> dict[str, Any]:
    index_dir = output / "split_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    record_index = {row["record_id"]: index for index, row in enumerate(manifest_rows)}
    eligible_groups = sorted(
        (group for group in groups if group.eligible_for_allocation),
        key=lambda group: (group.subject_id, group.class_label, group.group_id),
    )
    group_index = {group.group_id: index for index, group in enumerate(eligible_groups)}
    ordered_connectors = sorted(
        connectors,
        key=lambda item: (item.subject_id, item.record_id, item.start_index),
    )
    connector_index = {
        connector.connector_id: index
        for index, connector in enumerate(ordered_connectors)
    }
    base.write_csv(
        index_dir / "record_lookup.csv",
        [
            {
                "record_index": index,
                "record_id": row["record_id"],
                "subject_id": row["subject_id"],
                "source_file": row["source_file"],
                "record_path": row["record_path"],
            }
            for index, row in enumerate(manifest_rows)
        ],
    )
    base.write_csv(
        index_dir / "group_lookup.csv",
        [
            {
                "group_index": index,
                "group_id": group.group_id,
                "group_kind": group.group_kind,
                "class_label": group.class_label,
                "subject_id": group.subject_id,
                "record_id": group.record_id,
            }
            for index, group in enumerate(eligible_groups)
        ],
    )
    base.write_csv(
        index_dir / "connector_lookup.csv",
        [
            {
                "connector_index": index,
                "connector_id": connector.connector_id,
                "window_id": connector.window_id,
                "left_group_id": connector.left_group_id,
                "right_group_id": connector.right_group_id,
            }
            for index, connector in enumerate(ordered_connectors)
        ],
    )
    verification_problems: list[str] = []
    verified_rows = 0
    verified_files = 0
    subjects = sorted({row["subject_id"] for row in window_rows})
    for subject in subjects:
        for outer_fold in range(OUTER_FOLDS):
            selected = [
                row
                for row in window_rows
                if row["subject_id"] == subject and int(row["outer_fold_id"]) == outer_fold
            ]
            selected.sort(
                key=lambda row: (
                    ROLE_CODES[row["final_role"]],
                    record_index[row["record_id"]],
                    int(row["start_index"]),
                )
            )
            payload = {
                "record_index": np.asarray(
                    [record_index[row["record_id"]] for row in selected], dtype=np.int16
                ),
                "record_id": np.asarray([row["record_id"] for row in selected], dtype=str),
                "start_index": np.asarray(
                    [row["start_index"] for row in selected], dtype=np.int32
                ),
                "end_index_exclusive": np.asarray(
                    [row["end_index_exclusive"] for row in selected], dtype=np.int32
                ),
                "role_code": np.asarray(
                    [row["role_code"] for row in selected], dtype=np.int8
                ),
                "y_binary": np.asarray(
                    [row["y_binary"] for row in selected], dtype=np.int8
                ),
                "group_index": np.asarray(
                    [
                        group_index[row["allocation_group_id"]]
                        if row["allocation_group_id"]
                        else -1
                        for row in selected
                    ],
                    dtype=np.int32,
                ),
                "allocation_group_id": np.asarray(
                    [row["allocation_group_id"] for row in selected], dtype=str
                ),
                "connector_index": np.asarray(
                    [
                        connector_index[row["connector_id"]]
                        if row["connector_id"]
                        else -1
                        for row in selected
                    ],
                    dtype=np.int32,
                ),
                "connector_id": np.asarray(
                    [row["connector_id"] for row in selected], dtype=str
                ),
                "left_group_id": np.asarray(
                    [row["left_group_id"] for row in selected], dtype=str
                ),
                "right_group_id": np.asarray(
                    [row["right_group_id"] for row in selected], dtype=str
                ),
                "is_dynamic_connector": np.asarray(
                    [row["is_dynamic_connector"] for row in selected], dtype=bool
                ),
                "window_id": np.asarray(
                    [row["window_id"] for row in selected], dtype=str
                ),
            }
            path = index_dir / f"{subject}_outer{outer_fold}_nbm_indices.npz"
            np.savez_compressed(
                path,
                **payload,
            )
            with np.load(path, allow_pickle=False) as restored:
                missing = sorted(set(payload) - set(restored.files))
                unequal = [
                    key
                    for key, expected in payload.items()
                    if key in restored.files and not np.array_equal(restored[key], expected)
                ]
                lengths = {key: len(restored[key]) for key in restored.files}
            if missing or unequal or any(length != len(selected) for length in lengths.values()):
                verification_problems.append(
                    f"{path.name}:missing={missing}:unequal={unequal}:lengths={lengths}"
                )
            verified_files += 1
            verified_rows += len(selected)
    return {
        "pass": not verification_problems and verified_files == len(subjects) * OUTER_FOLDS,
        "verified_file_count": verified_files,
        "expected_file_count": len(subjects) * OUTER_FOLDS,
        "verified_row_count": verified_rows,
        "lookup_files": ["record_lookup.csv", "group_lookup.csv", "connector_lookup.csv"],
        "problems": verification_problems[:20],
    }


def copy_canonical_layout(source: Path, build: Path) -> None:
    for name in ("manifest.csv", "fog_events.csv", "loso_folds.csv", "preprocessing_report.json"):
        shutil.copy2(source / name, build / name)
    for name in (
        "fir_kaiser65_cutoff14hz.csv",
        "record_resampling_audit.csv",
        "README_32Hz.md",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, build / name)
    shutil.copytree(source / "records", build / "records")


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
    if not manifest_rows:
        raise ValueError(f"empty source manifest: {source / 'manifest.csv'}")
    sampling_rates = {int(row["sampling_rate_hz"]) for row in manifest_rows}
    source_sampling_rates = {
        int(row.get("source_sampling_rate_hz", row["sampling_rate_hz"]))
        for row in manifest_rows
    }
    if len(sampling_rates) != 1 or len(source_sampling_rates) != 1:
        raise ValueError(
            "all records must use one processed rate and one source sampling rate"
        )
    source_schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    configure_sampling(
        next(iter(sampling_rates)),
        source_sampling_rate_hz=next(iter(source_sampling_rates)),
        source_dataset_id=str(source_schema.get("dataset_id", "daphnet")),
    )
    manifest = {row["record_id"]: row for row in manifest_rows}
    records = base.load_records(source, manifest_rows)
    normalized_event_rows, event_input_audit = normalize_fog_event_rows(
        base.read_csv(source / "fog_events.csv")
    )
    if not event_input_audit["pass"]:
        raise RuntimeError(json.dumps(event_input_audit, ensure_ascii=False, indent=2))
    events, events_by_record = ca.prepare_events(normalized_event_rows)
    summary_rows = base.read_csv(summary_path)
    summary_audit, source_summary_pass = reference_summary_audit(
        summary_rows, manifest_rows, records, events
    )
    event_audit = event_annotation_audit(records, events)

    candidates = enumerate_candidates(records, manifest)
    provenance_audit = source_provenance_audit(manifest_rows, records, candidates)
    fog_groups, fog_window_to_group, event_to_group = build_fog_groups(
        candidates, events_by_record
    )
    clean_groups, connectors, clean_window_to_group = build_clean_groups(candidates)
    groups = fog_groups + clean_groups
    core_window_to_group = dict(fog_window_to_group)
    overlap_ids = set(core_window_to_group) & set(clean_window_to_group)
    if overlap_ids:
        raise AssertionError(f"class group identities overlap: {sorted(overlap_ids)[:10]}")
    core_window_to_group.update(clean_window_to_group)

    clean_training_roles, allocation_audit = allocate_subject_groups(
        groups, connectors, candidates
    )
    clean_training_roles, fold_alignment_rows, fold_alignment_quality = (
        align_subject_outer_fold_labels(groups, clean_training_roles, allocation_audit)
    )
    window_rows, group_rows, fold_group_rows, connector_rows, excluded_rows = materialize_manifests(
        candidates,
        groups,
        connectors,
        core_window_to_group,
        clean_training_roles,
    )
    split_summary = build_split_summary(candidates, window_rows, excluded_rows)
    quality = quality_report(
        candidates,
        groups,
        connectors,
        core_window_to_group,
        window_rows,
        excluded_rows,
        summary_rows,
        source_summary_pass,
        event_audit,
        provenance_audit,
        split_summary,
    )
    quality["fold_label_alignment_audit"] = fold_alignment_quality
    quality["fog_event_input_convention_audit"] = event_input_audit
    quality["overall_pass"] = bool(
        quality["overall_pass"]
        and fold_alignment_quality["pass"]
        and event_input_audit["pass"]
    )
    pool_count_report = build_pool_count_report(split_summary, quality)
    payload = {
        "output": str(output),
        "dry_run": bool(args.dry_run),
        "quality": quality,
        "aggregate_split_summary": [row for row in split_summary if row["subject_id"] == "ALL"],
    }
    if not quality["overall_pass"]:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    build = output.with_name(f"{output.name}.__building_{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)
    copy_canonical_layout(source, build)
    base.write_csv(build / "fog_events.csv", normalized_event_rows)
    shutil.copy2(summary_path, build / SOURCE_SUMMARY_COPY)

    group_lookup = {group.group_id: group for group in groups}
    event_rows = update_event_manifest(events, event_to_group, group_lookup, candidates)
    base.write_csv(build / WINDOW_MANIFEST, window_rows)
    base.write_csv(build / GROUP_MANIFEST, group_rows)
    base.write_csv(build / FOLD_GROUP_ROLES, fold_group_rows)
    base.write_csv(build / CONNECTOR_MANIFEST, connector_rows)
    base.write_csv(build / FOLD_ALIGNMENT_MANIFEST, fold_alignment_rows)
    base.write_csv(build / EXCLUDED_AUDIT, excluded_rows)
    base.write_csv(build / SPLIT_SUMMARY, split_summary)
    (build / POOL_COUNT_REPORT).write_text(pool_count_report, encoding="utf-8")
    base.write_csv(build / SOURCE_SUMMARY_AUDIT, summary_audit)
    base.write_csv(build / "nbm_allocation_optimization_audit.csv", allocation_audit)
    base.write_csv(build / "nbm_fog_event_manifest.csv", event_rows)
    base.write_json(build / ROLE_CODES_JSON, ROLE_CODES)
    index_audit = save_indices(build, window_rows, manifest_rows, groups, connectors)
    quality["index_export_audit"] = index_audit
    quality["overall_pass"] = bool(quality["overall_pass"] and index_audit["pass"])
    base.write_json(build / QUALITY_REPORT, quality)
    if not quality["overall_pass"]:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))

    source_schema["nbm_split"] = {
        "window_manifest": WINDOW_MANIFEST,
        "group_manifest": GROUP_MANIFEST,
        "fold_group_roles": FOLD_GROUP_ROLES,
        "connector_manifest": CONNECTOR_MANIFEST,
        "subject_fold_alignment_manifest": FOLD_ALIGNMENT_MANIFEST,
        "excluded_window_audit": EXCLUDED_AUDIT,
        "split_summary": SPLIT_SUMMARY,
        "pool_count_report": POOL_COUNT_REPORT,
        "quality_report": QUALITY_REPORT,
        "protocol": PROTOCOL,
        "role_codes": ROLE_CODES_JSON,
        "split_indices_directory": "split_indices",
        "split_index_lookup_files": [
            "split_indices/record_lookup.csv",
            "split_indices/group_lookup.csv",
            "split_indices/connector_lookup.csv",
        ],
        "split_index_npz_pattern": "split_indices/{subject_id}_outer{fold_id}_nbm_indices.npz",
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "outer_folds": OUTER_FOLDS,
        "pure_fog_definition": f"fog_samples_in_2s == {WINDOW}",
        "pure_nonfog_definition": "fog_samples_in_2s == 0",
        "mixed_window_policy": f"exclude 1..{WINDOW - 1} FOG samples",
        "allocation_scope": "within subject",
        "permanent_test_fraction_per_class": 0.20,
        "target_fractions_of_full_class_inventory": FULL_INVENTORY_TARGETS,
    }
    base.write_json(build / "schema.json", source_schema)

    protocol = {
        "dataset_id": DATASET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed": str(source),
        "source_manifest_sha256": base.sha256(source / "manifest.csv"),
        "source_fog_events_sha256": base.sha256(source / "fog_events.csv"),
        "source_fog_event_end_index_convention_audit": event_input_audit,
        "output_fog_event_end_index_convention": "inclusive_last_fog_sample",
        "user_summary_source": str(summary_path),
        "user_summary_copied_as": SOURCE_SUMMARY_COPY,
        "user_summary_sha256": base.sha256(summary_path),
        "user_summary_row_count": len(summary_rows),
        "user_summary_audit_pass": source_summary_pass,
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "window_anchor": f"start at sample 0 of each record, then +{STRIDE} samples",
        "source_inventory_before_cross_pool_boundary_exclusions": quality[
            "confirmed_source_inventory"
        ],
        "signal_validity_or_flatline_filter_applied": False,
        "extra_fog_guard_applied": False,
        "pure_fog_rule": f"all {WINDOW} samples are FOG",
        "pure_nonfog_rule": f"all {WINDOW} samples are Non-FoG",
        "mixed_rule": f"exclude every window with 1..{WINDOW - 1} FOG samples",
        "scope": "split independently within every subject",
        "permanent_test": "freeze approximately 20% of retained windows in each class by whole groups",
        "development_folds": "assign remaining groups to three folds by retained window count; rotate one validation fold",
        "subject_fold_label_alignment": (
            "permute each subject's already-frozen fold labels consistently across classes "
            "to reduce aggregate FoG imbalance; group membership never changes"
        ),
        "fog_group": (
            "each complete annotated FoG event is indivisible; only overlapping "
            "annotations are merged, with no extra inter-event guard"
        ),
        "clean_group": "contiguous non-overlapping cores spanning at most 60 seconds",
        "clean_connector": (
            "one pure connector between adjacent cores; retain per outer fold only when both "
            "neighboring cores have the same final role"
        ),
        "boundary_exclusion": EXCLUDED_BOUNDARY,
        "clean_training_targets_of_full_clean_inventory": {
            NBM_TRAIN_CLEAN: 0.256,
            NBM_EARLYSTOP_CLEAN: 0.064,
            CLASSIFIER_TRAIN_CLEAN: 16.0 / 75.0,
        },
        "fog_training_target_of_full_fog_inventory": {CLASSIFIER_TRAIN_FOG: 8.0 / 15.0},
        "allocation": "deterministic whole-group optimization by retained window count",
        "ratio_quality_tolerances_percentage_points": {
            "per_subject": MAX_SUBJECT_RATIO_ERROR_PERCENTAGE_POINTS,
            "aggregate": MAX_AGGREGATE_RATIO_ERROR_PERCENTAGE_POINTS,
        },
        "forbidden": [
            "random window allocation",
            "mixed 2 s windows in any active pool",
            "FoG windows in NBM fitting or NBM early stopping",
            "raw-sample overlap between different active roles within an outer fold",
            "permanent-test use for tuning, early stopping, calibration, or threshold selection",
        ],
    }
    base.write_json(build / PROTOCOL, protocol)
    (build / "README_NBM.md").write_text(
        f"# Daphnet {DATASET_ID}\n\n"
        "This directory preserves the canonical continuous records and adds a strict-purity, "
        "within-subject split for the NBM and downstream classifier.\n\n"
        f"- Window: 2 s ({WINDOW} samples), anchored to each record, stride 1 s ({STRIDE} samples).\n"
        f"- PURE_FOG: {WINDOW}/{WINDOW} samples are FoG. "
        f"PURE_NONFOG: 0/{WINDOW} samples are FoG.\n"
        "- Mixed candidates are excluded and listed in `nbm_excluded_window_audit.csv`.\n"
        "- A fixed class-wise approximately 20% permanent test pool is shared by all folds.\n"
        "- The remaining groups form three rotating folds; one validates and two train.\n"
        "- Subject fold labels are aligned after allocation to improve aggregate balance; this "
        "does not change any subject's group membership or validation rotation.\n"
        "- NBM train, NBM early-stop, and classifier clean-training pools are disjoint.\n"
        "- Dynamic clean connectors are retained only when both neighboring blocks have the "
        "same final role in that outer fold.\n\n"
        "Use `nbm_window_manifest.csv` or `split_indices/*_nbm_indices.npz` as the authoritative "
        "active-window selection. Read `NBM_POOL_COUNT_REPORT.md` for aggregate counts and "
        "consult `nbm_quality_report.json` before training. The NPZ files include both stable "
        "string identifiers and compact integer indices; their lookup tables are stored beside "
        "them in `split_indices/`.\n",
        encoding="utf-8",
    )
    build.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
