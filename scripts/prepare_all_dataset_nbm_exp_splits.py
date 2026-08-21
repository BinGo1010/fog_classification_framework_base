"""Build the 64 Hz within-subject NBM experiment split for All_dataset.

The source records remain continuous.  Candidate windows are anchored at the
start of every record, use 128 samples (2 s), and advance by 64 samples (1 s).
Only 128/128 Non-FoG and 128/128 FoG windows are eligible.  Mixed windows and
the one-stride connector at every cross-pool boundary are excluded explicitly.

The output follows the existing ``processed_NBM`` training contract: canonical
continuous ``records/``, CSV manifests, JSON protocol/quality reports, and one
``split_indices/{subject}_outer{fold}_nbm_indices.npz`` file per subject and
rotation.  Defaults below are intentionally editable for direct PyCharm use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


# =============================================================================
# Manual settings for direct execution from PyCharm.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "dataset" / "All_dataset" / "processed_NGM"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp"
P08_LABELED_ROOT = (
    PROJECT_ROOT / "dataset" / "All_dataset" / "segments_experimental_labeled"
)
DRY_RUN = False
# =============================================================================


DATASET_ID = "stanford_imu_fog_5imu_64hz_nbm_exp"
SUBSET_ID = "imus5_subjects8"
SUBJECTS = tuple(f"P{index:02d}" for index in range(1, 9))
FS = 64
WINDOW = 128
STRIDE = 64
OUTER_FOLDS = 3
MAX_NONFOG_CORE_WINDOWS = 14
MAX_FOG_CORE_WINDOWS = 16

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
ROLE_TARGETS = {
    PERMANENT_TEST_NONFOG: 0.20,
    PERMANENT_TEST_FOG: 0.20,
    EXTERNAL_VALIDATION_NONFOG: 4.0 / 15.0,
    EXTERNAL_VALIDATION_FOG: 4.0 / 15.0,
    NBM_TRAIN_CLEAN: 0.256,
    NBM_EARLYSTOP_CLEAN: 0.064,
    CLASSIFIER_TRAIN_CLEAN: 16.0 / 75.0,
    CLASSIFIER_TRAIN_FOG: 8.0 / 15.0,
}

EXPECTED_INVENTORY = {
    "record_count": 62,
    "subject_count": 8,
    "sample_count": 339_594,
    "nonfog_sample_count": 253_878,
    "fog_sample_count": 85_716,
    "fog_event_count": 214,
    "candidate_window_count": 5_221,
    "pure_nonfog_window_count": 3_498,
    "pure_fog_window_count": 957,
    "mixed_window_count": 766,
}


@dataclass(frozen=True)
class Record:
    record_id: str
    subject_id: str
    run_id: str
    segment_id: int
    record_path: str
    source_file: str
    source_sampling_rate_hz: int
    source_start_row: int
    source_n_samples: int
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Candidate:
    window_id: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    source_file: str
    start_index: int
    end_index_exclusive: int
    fog_samples_in_2s: int
    purity_label: str

    @property
    def class_label(self) -> str:
        if self.purity_label == PURE_NONFOG:
            return CLASS_NONFOG
        if self.purity_label == PURE_FOG:
            return CLASS_FOG
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
    parent_event_id: int | None = None
    subblock_ordinal: int = 0
    split_from_continuous_event: bool = False
    eligible_for_allocation: bool = True
    permanent_partition: str = ""
    assigned_development_fold: int | None = None

    @property
    def window_count(self) -> int:
        return len(self.window_ids)


@dataclass(frozen=True)
class Connector:
    connector_id: str
    window_id: str
    class_label: str
    boundary_kind: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    start_index: int
    end_index_exclusive: int
    left_group_id: str
    right_group_id: str
    parent_event_id: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--p08-labeled-root", type=Path, default=P08_LABELED_ROOT)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def binary_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("labels must be a one-dimensional binary array")
    if not len(values):
        return []
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    boundaries = np.r_[0, changes, len(values)]
    return [
        (int(start), int(end), int(values[start]))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def p08_summary_rows(labeled_root: Path) -> dict[str, dict[str, str]]:
    paths = sorted(labeled_root.glob("*_experimental_y_binary_segments_summary.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"expected exactly one P08 labeled summary CSV, found {len(paths)}"
        )
    rows = read_csv(paths[0])
    output = {str(row["segment_id"]): row for row in rows}
    if set(output) != {"seg000", "seg001"}:
        raise ValueError(f"unexpected P08 segments in {paths[0]}: {sorted(output)}")
    return output


def synthesize_p08_manifest_row(
    record_id: str,
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    summary: Mapping[str, str],
) -> dict[str, Any]:
    segment_token = record_id.split("_", 1)[1]
    segment_id = int(segment_token.replace("seg", ""))
    source_n = int(summary["samples"])
    expected_n = int(math.floor((source_n - 1) * FS / 100.0)) + 1
    if expected_n != len(y):
        raise AssertionError(f"P08 resampling length mismatch for {record_id}")
    events = sum(label == 1 for _, _, label in binary_runs(y))
    return {
        "dataset_id": DATASET_ID,
        "subset_id": SUBSET_ID,
        "record_id": record_id,
        "record_path": f"records/{path.name}",
        "source_file": str(summary["file"]),
        "source_subset_dir": "segments_experimental_labeled",
        "subject_id": "P08",
        "run_id": record_id,
        "segment_id": segment_id,
        "visit": "",
        "source_condition_token": "",
        "trial_id": segment_id,
        "source_start_row": 0,
        "source_end_row": source_n - 1,
        "source_start_time": summary["start_time"],
        "source_end_time": summary["end_time"],
        "sampling_rate_hz": FS,
        "estimated_sampling_rate_hz": FS,
        "n_samples": len(y),
        "duration_sec": len(y) / FS,
        "n_normal_samples": int(np.count_nonzero(y == 0)),
        "n_fog_samples": int(np.count_nonzero(y == 1)),
        "fog_event_count": events,
        "has_fog": bool(np.any(y == 1)),
        "usable": True,
        "notes": (
            "P08 experimental interval; 100 Hz five-IMU signal filtered with "
            "65-tap Kaiser(beta=5) 28 Hz FIR and time-aligned to 64 Hz"
        ),
        "source_sampling_rate_hz": 100,
        "source_n_samples": source_n,
        "source_record_path": f"processed_NGM_P08/{path.name}",
        "downsampling_method": (
            "FIR65 cutoff28Hz Kaiser(beta=5), reflect32, delay compensation32, "
            "linear interpolation 100to64Hz"
        ),
        "label_resampling": "nearest source label on aligned 64 Hz grid",
        "start_pc_world_datetime_local": summary.get(
            "start_pc_world_datetime_local", ""
        ),
        "end_pc_world_datetime_local": summary.get("end_pc_world_datetime_local", ""),
    }


def canonicalize_source_row(row: Mapping[str, str]) -> dict[str, Any]:
    record_id = str(row["record_id"])
    visit = str(row.get("visit", "")).strip()
    trial = str(row.get("trial_id", "")).strip()
    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        run_id = f"visit{visit}_trial{trial}" if visit or trial else record_id
    output: dict[str, Any] = dict(row)
    output.update(
        {
            "dataset_id": DATASET_ID,
            "subset_id": SUBSET_ID,
            "run_id": run_id,
            "sampling_rate_hz": FS,
        }
    )
    return output


def load_records_and_manifest(
    source_root: Path, labeled_root: Path
) -> tuple[list[Record], list[dict[str, Any]], dict[str, str]]:
    source_manifest_rows = read_csv(source_root / "manifest.csv")
    source_by_id = {row["record_id"]: row for row in source_manifest_rows}
    if len(source_by_id) != len(source_manifest_rows):
        raise ValueError("source manifest has duplicate record_id values")
    p08_summary = p08_summary_rows(labeled_root)
    record_paths = sorted((source_root / "records").glob("*.npz"))
    records: list[Record] = []
    manifest_rows: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    for path in record_paths:
        record_id = path.stem
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"x", "y_binary"}:
                raise ValueError(f"unexpected NPZ arrays in {path}: {payload.files}")
            x = payload["x"]
            y = payload["y_binary"]
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 30:
            raise ValueError(f"expected float32 [N,30] x in {path}, got {x.dtype} {x.shape}")
        if y.dtype != np.int8 or y.shape != (len(x),) or not np.isin(y, (0, 1)).all():
            raise ValueError(f"invalid y_binary in {path}: {y.dtype} {y.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite signal values in {path}")
        subject_id = record_id.split("_", 1)[0]
        segment_token = record_id.split("_", 1)[1]
        if subject_id == "P08":
            row = synthesize_p08_manifest_row(
                record_id, path, x, y, p08_summary[segment_token]
            )
        else:
            if record_id not in source_by_id:
                raise ValueError(f"record missing from source manifest: {record_id}")
            row = canonicalize_source_row(source_by_id[record_id])
            if int(row["n_samples"]) != len(y):
                raise ValueError(f"source manifest sample mismatch for {record_id}")
            if int(row["n_normal_samples"]) != int(np.count_nonzero(y == 0)):
                raise ValueError(f"source manifest Non-FoG mismatch for {record_id}")
            if int(row["n_fog_samples"]) != int(np.count_nonzero(y == 1)):
                raise ValueError(f"source manifest FoG mismatch for {record_id}")
        row["record_path"] = f"records/{path.name}"
        row["usable"] = True
        records.append(
            Record(
                record_id=record_id,
                subject_id=subject_id,
                run_id=str(row["run_id"]),
                segment_id=int(row["segment_id"]),
                record_path=str(row["record_path"]),
                source_file=str(row["source_file"]),
                source_sampling_rate_hz=int(row.get("source_sampling_rate_hz", FS)),
                source_start_row=int(row.get("source_start_row", 0)),
                source_n_samples=int(row.get("source_n_samples", len(y))),
                x=x,
                y=y,
            )
        )
        manifest_rows.append(row)
        record_hashes[record_id] = sha256(path)
    records.sort(key=lambda item: item.record_id)
    manifest_rows.sort(key=lambda item: str(item["record_id"]))
    if set(source_by_id) != {record.record_id for record in records if record.subject_id != "P08"}:
        raise ValueError("P01-P07 records and source manifest do not reconcile")
    return records, manifest_rows, record_hashes


def make_fog_events(records: Sequence[Record]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        event_id = 0
        for start, end, label in binary_runs(record.y):
            if label != 1:
                continue
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subset_id": SUBSET_ID,
                    "record_id": record.record_id,
                    "subject_id": record.subject_id,
                    "segment_id": record.segment_id,
                    "event_id": event_id,
                    "start_index": start,
                    "end_index": end - 1,
                    "start_time_sec": start / FS,
                    "end_time_sec": (end - 1) / FS,
                    "duration_sec": (end - start) / FS,
                }
            )
            event_id += 1
    return rows


def purity_label(fog_samples: int) -> str:
    if fog_samples == 0:
        return PURE_NONFOG
    if fog_samples == WINDOW:
        return PURE_FOG
    if 0 < fog_samples < WINDOW:
        return MIXED
    raise ValueError(f"invalid FoG count: {fog_samples}")


def enumerate_candidates(records: Sequence[Record]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for record in records:
        prefix = np.r_[0, np.cumsum(record.y == 1, dtype=np.int64)]
        for start in range(0, len(record.y) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            fog_samples = int(prefix[end] - prefix[start])
            candidates.append(
                Candidate(
                    window_id=f"{record.record_id}:{start}:{end}",
                    subject_id=record.subject_id,
                    record_id=record.record_id,
                    run_id=record.run_id,
                    segment_id=record.segment_id,
                    source_file=record.source_file,
                    start_index=start,
                    end_index_exclusive=end,
                    fog_samples_in_2s=fog_samples,
                    purity_label=purity_label(fog_samples),
                )
            )
    return candidates


def consecutive_candidate_runs(candidates: Sequence[Candidate]) -> list[list[Candidate]]:
    selected = sorted(candidates, key=lambda item: item.start_index)
    if not selected:
        return []
    output: list[list[Candidate]] = []
    current = [selected[0]]
    for candidate in selected[1:]:
        if candidate.start_index == current[-1].start_index + STRIDE:
            current.append(candidate)
        else:
            output.append(current)
            current = [candidate]
    output.append(current)
    return output


def balanced_run_parts(
    run: Sequence[Candidate], block_count: int
) -> tuple[list[tuple[Candidate, ...]], list[Candidate]]:
    """Split a pure run into blocks separated by one candidate connector."""

    ordered = tuple(sorted(run, key=lambda item: item.start_index))
    block_count = int(block_count)
    if block_count < 1 or len(ordered) < 2 * block_count - 1:
        raise ValueError(
            f"cannot split {len(ordered)} candidates into {block_count} non-empty blocks"
        )
    core_total = len(ordered) - (block_count - 1)
    base_size, remainder = divmod(core_total, block_count)
    sizes = [base_size + int(index >= block_count - remainder) for index in range(block_count)]
    blocks: list[tuple[Candidate, ...]] = []
    connectors: list[Candidate] = []
    position = 0
    for index, size in enumerate(sizes):
        blocks.append(tuple(ordered[position : position + size]))
        position += size
        if index + 1 < block_count:
            connectors.append(ordered[position])
            position += 1
    if position != len(ordered) or any(not block for block in blocks):
        raise AssertionError("balanced run partition failed")
    return blocks, connectors


def minimum_run_block_counts(
    runs: Sequence[Sequence[Candidate]],
    minimum_groups: int,
    maximum_core_windows: int,
) -> list[int]:
    """Use the fewest connectors while meeting block-size and feasibility limits."""

    counts = [
        max(1, int(math.ceil((len(run) + 1) / (maximum_core_windows + 1))))
        for run in runs
    ]
    for index, (run, count) in enumerate(zip(runs, counts)):
        maximum = (len(run) + 1) // 2
        counts[index] = min(count, maximum)
    while sum(counts) < minimum_groups:
        options = [
            (
                len(run) / (counts[index] + 1),
                len(run),
                -index,
                index,
            )
            for index, run in enumerate(runs)
            if counts[index] < (len(run) + 1) // 2
        ]
        if not options:
            raise ValueError(
                f"cannot create {minimum_groups} allocation groups from pure runs"
            )
        counts[max(options)[-1]] += 1
    return counts


def containing_event_id(record: Record, candidate: Candidate) -> int:
    event_id = 0
    for start, end, label in binary_runs(record.y):
        if label != 1:
            continue
        if start <= candidate.start_index and end >= candidate.end_index_exclusive:
            return event_id
        event_id += 1
    raise AssertionError(f"pure FoG window is not inside one event: {candidate.window_id}")


def build_groups(
    records: Sequence[Record], candidates: Sequence[Candidate]
) -> tuple[list[AllocationGroup], list[Connector], dict[str, str]]:
    record_lookup = {record.record_id: record for record in records}
    by_subject_class: dict[tuple[str, str], list[list[Candidate]]] = defaultdict(list)
    for record in records:
        record_candidates = [item for item in candidates if item.record_id == record.record_id]
        for class_label, purity in ((CLASS_NONFOG, PURE_NONFOG), (CLASS_FOG, PURE_FOG)):
            selected = [item for item in record_candidates if item.purity_label == purity]
            by_subject_class[(record.subject_id, class_label)].extend(
                consecutive_candidate_runs(selected)
            )

    groups: list[AllocationGroup] = []
    connectors: list[Connector] = []
    window_to_group: dict[str, str] = {}
    for subject in SUBJECTS:
        for class_label in (CLASS_NONFOG, CLASS_FOG):
            runs = by_subject_class[(subject, class_label)]
            if not runs:
                raise ValueError(f"{subject} has no pure {class_label} window")
            minimum_groups = 6 if class_label == CLASS_NONFOG else 4
            block_counts = minimum_run_block_counts(
                runs,
                minimum_groups,
                (
                    MAX_NONFOG_CORE_WINDOWS
                    if class_label == CLASS_NONFOG
                    else MAX_FOG_CORE_WINDOWS
                ),
            )
            group_serial_by_record: Counter[str] = Counter()
            connector_serial_by_record: Counter[str] = Counter()
            for run, block_count in zip(runs, block_counts):
                blocks, connector_candidates = balanced_run_parts(run, block_count)
                first = run[0]
                event_id = (
                    containing_event_id(record_lookup[first.record_id], first)
                    if class_label == CLASS_FOG
                    else None
                )
                run_groups: list[AllocationGroup] = []
                for ordinal, block in enumerate(blocks):
                    record_id = block[0].record_id
                    serial = group_serial_by_record[record_id]
                    group_serial_by_record[record_id] += 1
                    stem = "cleanblock" if class_label == CLASS_NONFOG else "fogblock"
                    group_id = f"{record_id}_{stem}{serial:03d}"
                    split_fog = class_label == CLASS_FOG and block_count > 1
                    group = AllocationGroup(
                        group_id=group_id,
                        group_kind=(
                            "clean_nonfog_block"
                            if class_label == CLASS_NONFOG
                            else (
                                "fog_event_subblock" if split_fog else "fog_event_block"
                            )
                        ),
                        class_label=class_label,
                        subject_id=block[0].subject_id,
                        record_id=record_id,
                        run_id=block[0].run_id,
                        segment_id=block[0].segment_id,
                        start_index=block[0].start_index,
                        end_index_exclusive=block[-1].end_index_exclusive,
                        window_ids=tuple(item.window_id for item in block),
                        event_ids=() if event_id is None else (event_id,),
                        parent_event_id=event_id,
                        subblock_ordinal=ordinal,
                        split_from_continuous_event=split_fog,
                    )
                    groups.append(group)
                    run_groups.append(group)
                    for item in block:
                        if item.window_id in window_to_group:
                            raise AssertionError(f"duplicate group window: {item.window_id}")
                        window_to_group[item.window_id] = group_id
                for index, item in enumerate(connector_candidates):
                    serial = connector_serial_by_record[item.record_id]
                    connector_serial_by_record[item.record_id] += 1
                    class_token = "clean" if class_label == CLASS_NONFOG else "fog"
                    connectors.append(
                        Connector(
                            connector_id=f"{item.record_id}_{class_token}connector{serial:03d}",
                            window_id=item.window_id,
                            class_label=class_label,
                            boundary_kind=(
                                "nonfog_run_block_split"
                                if class_label == CLASS_NONFOG
                                else "continuous_fog_event_block_split"
                            ),
                            subject_id=item.subject_id,
                            record_id=item.record_id,
                            run_id=item.run_id,
                            segment_id=item.segment_id,
                            start_index=item.start_index,
                            end_index_exclusive=item.end_index_exclusive,
                            left_group_id=run_groups[index].group_id,
                            right_group_id=run_groups[index + 1].group_id,
                            parent_event_id=event_id,
                        )
                    )

    pure_ids = {item.window_id for item in candidates if item.purity_label != MIXED}
    connector_ids = {item.window_id for item in connectors}
    if set(window_to_group) & connector_ids:
        raise AssertionError("core groups and connectors overlap by window identity")
    if set(window_to_group) | connector_ids != pure_ids:
        missing = sorted(pure_ids - set(window_to_group) - connector_ids)[:10]
        extra = sorted((set(window_to_group) | connector_ids) - pure_ids)[:10]
        raise AssertionError(f"pure-window grouping mismatch: missing={missing}, extra={extra}")
    p08_fog_splits = [
        group
        for group in groups
        if group.subject_id == "P08"
        and group.class_label == CLASS_FOG
        and group.split_from_continuous_event
    ]
    p08_fog_connectors = [
        item
        for item in connectors
        if item.subject_id == "P08" and item.class_label == CLASS_FOG
    ]
    if len(p08_fog_splits) != 2 or len(p08_fog_connectors) != 1:
        raise AssertionError(
            "P08 FoG must use exactly two subblocks and one audited connector"
        )
    return groups, connectors, window_to_group


def effective_label_counts(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    assignments: Mapping[str, str],
    labels: Sequence[str],
) -> dict[str, int]:
    group_ids = {group.group_id for group in groups}
    counts = {label: 0 for label in labels}
    for group in groups:
        counts[assignments[group.group_id]] += group.window_count
    for connector in connectors:
        if (
            connector.left_group_id not in group_ids
            or connector.right_group_id not in group_ids
        ):
            continue
        left = assignments[connector.left_group_id]
        right = assignments[connector.right_group_id]
        if left == right:
            counts[left] += 1
    return counts


def fraction_objective(
    counts: Mapping[str, int], target_fractions: Mapping[str, float]
) -> tuple[float, float, int]:
    total = int(sum(counts.values()))
    if total <= 0:
        return float("inf"), float("inf"), 0
    errors = [
        100.0 * (float(counts[label]) / total - float(target_fractions[label]))
        for label in target_fractions
    ]
    return max(map(abs, errors)), sum(value * value for value in errors), -total


def assignment_is_better(
    left: tuple[float, float, int], right: tuple[float, float, int]
) -> bool:
    if left[0] < right[0] - 1e-12:
        return True
    if left[0] > right[0] + 1e-12:
        return False
    if left[1] < right[1] - 1e-12:
        return True
    if left[1] > right[1] + 1e-12:
        return False
    return left[2] < right[2]


def optimize_group_labels_heuristic(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    target_fractions: Mapping[str, float],
    minimum_group_counts: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, int], dict[str, Any]]:
    """Deterministically minimize retained-window fraction error.

    The lexicographic objective is: smallest maximum absolute percentage-point
    error, then smallest squared error, then greatest retained-window count.
    Exact enumeration is used for small group sets; larger sets use deterministic
    multi-start coordinate descent with whole-group moves and swaps.
    """

    labels = tuple(target_fractions)
    if not math.isclose(sum(target_fractions.values()), 1.0, abs_tol=1e-12):
        raise ValueError("target fractions must sum to one")
    ordered = tuple(sorted(groups, key=lambda item: item.group_id))
    minimum_required = sum(int(minimum_group_counts.get(label, 0)) for label in labels)
    if len(ordered) < minimum_required:
        raise ValueError(
            f"need at least {minimum_required} groups for {labels}, got {len(ordered)}"
        )
    group_by_id = {group.group_id: group for group in ordered}

    def valid(mapping: Mapping[str, str]) -> bool:
        counts = Counter(mapping.values())
        return all(
            counts[label] >= int(minimum_group_counts.get(label, 0))
            for label in labels
        )

    def score(mapping: Mapping[str, str]) -> tuple[float, float, int]:
        return fraction_objective(
            effective_label_counts(ordered, connectors, mapping, labels),
            target_fractions,
        )

    def lexical(mapping: Mapping[str, str]) -> tuple[int, ...]:
        return tuple(labels.index(mapping[group.group_id]) for group in ordered)

    best_mapping: dict[str, str] | None = None
    best_score: tuple[float, float, int] | None = None
    evaluated = 0

    if len(ordered) <= 10:
        for values in itertools.product(labels, repeat=len(ordered)):
            mapping = {
                group.group_id: values[index] for index, group in enumerate(ordered)
            }
            if not valid(mapping):
                continue
            evaluated += 1
            current = score(mapping)
            if (
                best_score is None
                or assignment_is_better(current, best_score)
                or (current == best_score and lexical(mapping) < lexical(best_mapping or {}))
            ):
                best_mapping = mapping
                best_score = current
        method = "exact_enumeration"
    else:
        method = "deterministic_multistart_move_swap_descent"
        group_orders = [
            tuple(sorted(ordered, key=lambda item: (-item.window_count, item.group_id))),
            tuple(sorted(ordered, key=lambda item: (item.window_count, item.group_id))),
            ordered,
            tuple(reversed(ordered)),
        ]
        label_orders = list(itertools.permutations(labels))[: min(12, math.factorial(len(labels)))]
        seeds: list[dict[str, str]] = []
        for order_index, group_order in enumerate(group_orders):
            for label_order in label_orders:
                unassigned = list(group_order)
                mapping: dict[str, str] = {}
                # Reserve the smallest available groups for mandatory slots,
                # with rotations providing deterministic alternative starts.
                mandatory_pool = sorted(
                    unassigned, key=lambda item: (item.window_count, item.group_id)
                )
                if order_index % 2:
                    mandatory_pool.reverse()
                for label in label_order:
                    for _ in range(int(minimum_group_counts.get(label, 0))):
                        group = mandatory_pool.pop(0)
                        mapping[group.group_id] = label
                        unassigned.remove(group)
                for group in sorted(
                    unassigned, key=lambda item: (-item.window_count, item.group_id)
                ):
                    choices: list[tuple[tuple[float, float, int], int, str]] = []
                    for label in label_order:
                        trial = dict(mapping)
                        trial[group.group_id] = label
                        selected_groups = [
                            group_by_id[group_id] for group_id in trial
                        ]
                        selected_connectors = [
                            connector
                            for connector in connectors
                            if connector.left_group_id in trial
                            and connector.right_group_id in trial
                        ]
                        counts = effective_label_counts(
                            selected_groups, selected_connectors, trial, labels
                        )
                        choices.append(
                            (
                                fraction_objective(counts, target_fractions),
                                label_order.index(label),
                                label,
                            )
                        )
                    mapping[group.group_id] = min(choices)[-1]
                if valid(mapping):
                    seeds.append(mapping)

        for seed in seeds:
            mapping = dict(seed)
            current_score = score(mapping)
            for _ in range(max(40, len(ordered) * 6)):
                label_group_counts = Counter(mapping.values())
                best_action: tuple[str, str, str] | None = None
                next_score = current_score
                for group in ordered:
                    source_label = mapping[group.group_id]
                    if label_group_counts[source_label] <= int(
                        minimum_group_counts.get(source_label, 0)
                    ):
                        continue
                    for target_label in labels:
                        if target_label == source_label:
                            continue
                        trial = dict(mapping)
                        trial[group.group_id] = target_label
                        trial_score = score(trial)
                        evaluated += 1
                        action = ("move", group.group_id, target_label)
                        if assignment_is_better(trial_score, next_score) or (
                            trial_score == next_score
                            and best_action is not None
                            and action < best_action
                        ):
                            next_score = trial_score
                            best_action = action
                if best_action is not None and assignment_is_better(
                    next_score, current_score
                ):
                    _, group_id, target_label = best_action
                    mapping[group_id] = target_label
                    current_score = next_score
                    continue

                best_swap: tuple[str, str] | None = None
                next_score = current_score
                for left_index, left in enumerate(ordered):
                    for right in ordered[left_index + 1 :]:
                        if mapping[left.group_id] == mapping[right.group_id]:
                            continue
                        trial = dict(mapping)
                        trial[left.group_id], trial[right.group_id] = (
                            trial[right.group_id],
                            trial[left.group_id],
                        )
                        trial_score = score(trial)
                        evaluated += 1
                        swap = (left.group_id, right.group_id)
                        if assignment_is_better(trial_score, next_score) or (
                            trial_score == next_score
                            and best_swap is not None
                            and swap < best_swap
                        ):
                            next_score = trial_score
                            best_swap = swap
                if best_swap is None or not assignment_is_better(
                    next_score, current_score
                ):
                    break
                left_id, right_id = best_swap
                mapping[left_id], mapping[right_id] = mapping[right_id], mapping[left_id]
                current_score = next_score

            if (
                best_score is None
                or assignment_is_better(current_score, best_score)
                or (
                    current_score == best_score
                    and lexical(mapping) < lexical(best_mapping or {})
                )
            ):
                best_mapping = mapping
                best_score = current_score

    if best_mapping is None or best_score is None:
        raise RuntimeError(f"no feasible allocation for labels {labels}")
    counts = effective_label_counts(ordered, connectors, best_mapping, labels)
    group_counts = Counter(best_mapping.values())
    audit = {
        "method": method,
        "objective_order": [
            "minimize_max_absolute_percentage_point_error",
            "minimize_sum_squared_percentage_point_error",
            "maximize_retained_window_count",
        ],
        "evaluated_assignment_or_neighbor_count": evaluated,
        "maximum_absolute_percentage_point_error": best_score[0],
        "sum_squared_percentage_point_error": best_score[1],
        "retained_window_count": -best_score[2],
        "group_counts": dict(group_counts),
        "locally_optimal_under_single_group_moves_and_swaps": True,
    }
    return best_mapping, counts, audit


def optimize_group_labels(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    target_fractions: Mapping[str, float],
    minimum_group_counts: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, int], dict[str, Any]]:
    """Solve the whole-group allocation as a deterministic mixed-integer program.

    Targets use the complete pure-window inventory represented by the selected
    cores plus their possible connectors.  Three lexicographic solves minimize
    maximum count error, then total absolute count error, then maximize retained
    connectors.  Final reports additionally express every pool over the actual
    retained-window denominator requested by the user.
    """

    labels = tuple(target_fractions)
    ordered = tuple(sorted(groups, key=lambda item: item.group_id))
    group_position = {group.group_id: index for index, group in enumerate(ordered)}
    selected_ids = set(group_position)
    relevant_connectors = tuple(
        sorted(
            (
                item
                for item in connectors
                if item.left_group_id in selected_ids
                and item.right_group_id in selected_ids
            ),
            key=lambda item: item.connector_id,
        )
    )
    if not math.isclose(sum(target_fractions.values()), 1.0, abs_tol=1e-12):
        raise ValueError("target fractions must sum to one")
    minimum_required = sum(int(minimum_group_counts.get(label, 0)) for label in labels)
    if len(ordered) < minimum_required:
        raise ValueError(
            f"need at least {minimum_required} groups for {labels}, got {len(ordered)}"
        )

    group_count = len(ordered)
    label_count = len(labels)
    connector_count = len(relevant_connectors)
    x_offset = 0
    z_offset = group_count * label_count
    absolute_offset = z_offset + connector_count * label_count
    maximum_offset = absolute_offset + label_count
    variable_count = maximum_offset + 1

    def x_index(group_index: int, label_index: int) -> int:
        return x_offset + group_index * label_count + label_index

    def z_index(connector_index: int, label_index: int) -> int:
        return z_offset + connector_index * label_count + label_index

    potential_total = sum(group.window_count for group in ordered) + connector_count
    rational_targets = {
        label: Fraction(str(target_fractions[label])).limit_denominator(1000)
        for label in labels
    }
    count_scale = math.lcm(
        *(fraction.denominator for fraction in rational_targets.values())
    )
    targets = {
        label: float(rational_targets[label]) * potential_total for label in labels
    }
    scaled_targets = {
        label: int(rational_targets[label] * count_scale) * potential_total
        for label in labels
    }
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float], low: float, high: float
    ) -> None:
        row = np.zeros(variable_count, dtype=np.float64)
        for index, value in coefficients.items():
            row[index] = value
        rows.append(row)
        lower.append(low)
        upper.append(high)

    for group_index in range(group_count):
        add_constraint(
            {x_index(group_index, label_index): 1.0 for label_index in range(label_count)},
            1.0,
            1.0,
        )
    for label_index, label in enumerate(labels):
        add_constraint(
            {x_index(group_index, label_index): 1.0 for group_index in range(group_count)},
            float(minimum_group_counts.get(label, 0)),
            np.inf,
        )

    for connector_index, item in enumerate(relevant_connectors):
        left = group_position[item.left_group_id]
        right = group_position[item.right_group_id]
        for label_index in range(label_count):
            z = z_index(connector_index, label_index)
            x_left = x_index(left, label_index)
            x_right = x_index(right, label_index)
            add_constraint({z: 1.0, x_left: -1.0}, -np.inf, 0.0)
            add_constraint({z: 1.0, x_right: -1.0}, -np.inf, 0.0)
            add_constraint({z: -1.0, x_left: 1.0, x_right: 1.0}, -np.inf, 1.0)

    count_coefficients: list[dict[int, float]] = []
    for label_index, label in enumerate(labels):
        coefficients = {
            x_index(group_index, label_index): float(group.window_count * count_scale)
            for group_index, group in enumerate(ordered)
        }
        coefficients.update(
            {
                z_index(connector_index, label_index): float(count_scale)
                for connector_index in range(connector_count)
            }
        )
        count_coefficients.append(coefficients)
        positive = dict(coefficients)
        positive[absolute_offset + label_index] = -1.0
        add_constraint(positive, -np.inf, scaled_targets[label])
        negative = {index: -value for index, value in coefficients.items()}
        negative[absolute_offset + label_index] = -1.0
        add_constraint(negative, -np.inf, -scaled_targets[label])
        maximum_positive = dict(coefficients)
        maximum_positive[maximum_offset] = -1.0
        add_constraint(maximum_positive, -np.inf, scaled_targets[label])
        maximum_negative = {index: -value for index, value in coefficients.items()}
        maximum_negative[maximum_offset] = -1.0
        add_constraint(maximum_negative, -np.inf, -scaled_targets[label])

    matrix = np.vstack(rows)
    constraint = LinearConstraint(
        matrix, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    )
    variable_lower = np.zeros(variable_count, dtype=np.float64)
    variable_upper = np.full(variable_count, np.inf, dtype=np.float64)
    variable_upper[:absolute_offset] = 1.0
    bounds = Bounds(variable_lower, variable_upper)
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:absolute_offset] = 1

    def solve(
        objective: np.ndarray, extra: Sequence[tuple[Mapping[int, float], float, float]] = ()
    ) -> Any:
        if extra:
            extra_rows: list[np.ndarray] = []
            extra_lower: list[float] = []
            extra_upper: list[float] = []
            for coefficients, low, high in extra:
                row = np.zeros(variable_count, dtype=np.float64)
                for index, value in coefficients.items():
                    row[index] = value
                extra_rows.append(row)
                extra_lower.append(low)
                extra_upper.append(high)
            combined = LinearConstraint(
                np.vstack([matrix, *extra_rows]),
                np.r_[lower, extra_lower],
                np.r_[upper, extra_upper],
            )
        else:
            combined = constraint
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=combined,
            options={
                "disp": False,
                "presolve": True,
                "mip_rel_gap": 0.0,
                "time_limit": 2.0,
            },
        )
        if result.x is None:
            raise RuntimeError(f"MILP allocation failed: {result.message}")
        return result

    # Exact rational scaling makes D and every absolute deviation integral.
    # The weights below therefore implement the three lexicographic stages in
    # one MILP: one unit of D dominates every possible lower-order change; one
    # unit of total absolute error dominates every connector decision.
    absolute_weight = float(connector_count + 1)
    maximum_possible_absolute = 2 * count_scale * potential_total
    maximum_weight = float(
        maximum_possible_absolute * absolute_weight + connector_count + 1
    )
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[maximum_offset] = maximum_weight
    objective[absolute_offset:maximum_offset] = absolute_weight
    objective[z_offset:absolute_offset] = -1.0
    # A tiny stable tie-break keeps repeated runs deterministic without ever
    # trading one retained connector.
    for group_index in range(group_count):
        for label_index in range(label_count):
            objective[x_index(group_index, label_index)] += (
                1e-8 * (group_index + 1) * (label_index + 1)
            )
    solution = solve(objective)

    assignments: dict[str, str] = {}
    for group_index, group in enumerate(ordered):
        values = np.asarray(
            [
                solution.x[x_index(group_index, label_index)]
                for label_index in range(label_count)
            ]
        )
        label_index = int(np.argmax(values))
        if values[label_index] < 0.5:
            raise AssertionError(f"MILP returned fractional group assignment: {group.group_id}")
        assignments[group.group_id] = labels[label_index]
    counts = effective_label_counts(
        ordered, relevant_connectors, assignments, labels
    )
    group_counts = Counter(assignments.values())
    if any(
        group_counts[label] < int(minimum_group_counts.get(label, 0))
        for label in labels
    ):
        raise AssertionError("MILP group-count minimum was not satisfied")
    retained = sum(counts.values())
    count_errors = {label: counts[label] - targets[label] for label in labels}
    retained_fraction_score = fraction_objective(counts, target_fractions)
    audit = {
        "method": "scipy_optimize_milp_three_stage_lexicographic",
        "objective_order": [
            "minimize_max_absolute_window_count_error_against_full_pure_inventory",
            "minimize_total_absolute_window_count_error",
            "maximize_retained_connector_windows",
        ],
        "potential_pure_window_count": potential_total,
        "exact_rational_count_scale": count_scale,
        "retained_window_count": retained,
        "excluded_connector_count": potential_total - retained,
        "target_window_counts": targets,
        "actual_window_counts": counts,
        "window_count_errors": count_errors,
        "maximum_absolute_window_count_error": max(map(abs, count_errors.values())),
        "total_absolute_window_count_error": sum(map(abs, count_errors.values())),
        "retained_denominator_maximum_absolute_percentage_point_error": (
            retained_fraction_score[0]
        ),
        "retained_denominator_sum_squared_percentage_point_error": (
            retained_fraction_score[1]
        ),
        "group_counts": dict(group_counts),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "solver_proven_optimal": bool(solution.success),
        "solver_mip_gap": (
            None
            if getattr(solution, "mip_gap", None) is None
            else float(solution.mip_gap)
        ),
    }
    return assignments, counts, audit


def allocate_groups(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
) -> tuple[dict[tuple[int, str], str], list[dict[str, Any]]]:
    clean_training_roles: dict[tuple[int, str], str] = {}
    audit_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for class_label in (CLASS_NONFOG, CLASS_FOG):
            selected = [
                group
                for group in groups
                if group.subject_id == subject and group.class_label == class_label
            ]
            selected_ids = {group.group_id for group in selected}
            selected_connectors = [
                connector
                for connector in connectors
                if connector.class_label == class_label
                and connector.left_group_id in selected_ids
                and connector.right_group_id in selected_ids
            ]
            labels = ("permanent_test", "fold0", "fold1", "fold2")
            minima = (
                {
                    "permanent_test": 1,
                    "fold0": 1,
                    "fold1": 2,
                    "fold2": 2,
                }
                if class_label == CLASS_NONFOG
                else {label: 1 for label in labels}
            )
            assignments, counts, audit = optimize_group_labels(
                selected,
                selected_connectors,
                target_fractions={
                    "permanent_test": 0.20,
                    "fold0": 4.0 / 15.0,
                    "fold1": 4.0 / 15.0,
                    "fold2": 4.0 / 15.0,
                },
                minimum_group_counts=minima,
            )
            for group in selected:
                label = assignments[group.group_id]
                if label == "permanent_test":
                    group.permanent_partition = "permanent_test"
                    group.assigned_development_fold = None
                else:
                    group.permanent_partition = "development"
                    group.assigned_development_fold = int(label.replace("fold", ""))
            audit_rows.append(
                {
                    "subject_id": subject,
                    "class_label": class_label,
                    "allocation_stage": "joint_permanent_test_and_three_development_folds",
                    "outer_fold_id": "",
                    "target_fractions_of_retained_windows": json.dumps(
                        {
                            "permanent_test": 0.20,
                            "fold0": 4.0 / 15.0,
                            "fold1": 4.0 / 15.0,
                            "fold2": 4.0 / 15.0,
                        },
                        sort_keys=True,
                    ),
                    "actual_counts": json.dumps(counts, sort_keys=True),
                    "optimization": json.dumps(audit, sort_keys=True),
                }
            )

        subject_nonfog = [
            group
            for group in groups
            if group.subject_id == subject and group.class_label == CLASS_NONFOG
        ]
        subject_nonfog_connectors = [
            item
            for item in connectors
            if item.subject_id == subject and item.class_label == CLASS_NONFOG
        ]
        subject_group_lookup = {group.group_id: group for group in subject_nonfog}
        for outer_fold in range(OUTER_FOLDS):
            training = [
                group
                for group in subject_nonfog
                if group.permanent_partition == "development"
                and group.assigned_development_fold != outer_fold
            ]
            training_ids = {group.group_id for group in training}
            training_connectors = [
                item
                for item in subject_nonfog_connectors
                if item.left_group_id in training_ids and item.right_group_id in training_ids
                and base_partition_key(subject_group_lookup[item.left_group_id])
                == base_partition_key(subject_group_lookup[item.right_group_id])
            ]
            assignments, counts, audit = optimize_group_labels(
                training,
                training_connectors,
                target_fractions={
                    NBM_TRAIN_CLEAN: 0.48,
                    NBM_EARLYSTOP_CLEAN: 0.12,
                    CLASSIFIER_TRAIN_CLEAN: 0.40,
                },
                minimum_group_counts={
                    NBM_TRAIN_CLEAN: 1,
                    NBM_EARLYSTOP_CLEAN: 1,
                    CLASSIFIER_TRAIN_CLEAN: 1,
                },
            )
            for group in training:
                clean_training_roles[(outer_fold, group.group_id)] = assignments[
                    group.group_id
                ]
            audit_rows.append(
                {
                    "subject_id": subject,
                    "class_label": CLASS_NONFOG,
                    "allocation_stage": "clean_training_roles",
                    "outer_fold_id": outer_fold,
                    "target_fractions_of_retained_windows": json.dumps(
                        {
                            NBM_TRAIN_CLEAN: 0.48,
                            NBM_EARLYSTOP_CLEAN: 0.12,
                            CLASSIFIER_TRAIN_CLEAN: 0.40,
                        },
                        sort_keys=True,
                    ),
                    "actual_counts": json.dumps(counts, sort_keys=True),
                    "optimization": json.dumps(audit, sort_keys=True),
                }
            )
    return clean_training_roles, audit_rows


def base_partition_key(group: AllocationGroup) -> str:
    if group.permanent_partition == "permanent_test":
        return "permanent_test"
    if (
        group.permanent_partition == "development"
        and group.assigned_development_fold is not None
    ):
        return f"fold{group.assigned_development_fold}"
    raise AssertionError(f"incomplete base partition for {group.group_id}")


def align_subject_fold_labels(
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    clean_training_roles: Mapping[tuple[int, str], str],
    allocation_audit: list[dict[str, Any]],
) -> tuple[dict[tuple[int, str], str], list[dict[str, Any]], dict[str, Any]]:
    """Permute subject-local fold names without changing any membership."""

    permutations = tuple(itertools.permutations(range(OUTER_FOLDS)))
    counts_by_subject: dict[str, tuple[int, ...]] = {}
    for subject in SUBJECTS:
        subject_groups = [
            group
            for group in groups
            if group.subject_id == subject
            and group.class_label == CLASS_FOG
            and group.permanent_partition == "development"
        ]
        subject_ids = {group.group_id for group in subject_groups}
        subject_connectors = [
            item
            for item in connectors
            if item.subject_id == subject
            and item.class_label == CLASS_FOG
            and item.left_group_id in subject_ids
            and item.right_group_id in subject_ids
        ]
        mapping = {
            group.group_id: f"fold{group.assigned_development_fold}"
            for group in subject_groups
        }
        effective = effective_label_counts(
            subject_groups,
            subject_connectors,
            mapping,
            tuple(f"fold{fold}" for fold in range(OUTER_FOLDS)),
        )
        counts_by_subject[subject] = tuple(
            effective[f"fold{fold}"] for fold in range(OUTER_FOLDS)
        )

    fog_inventory = sum(group.window_count for group in groups if group.class_label == CLASS_FOG)
    fog_inventory += sum(item.class_label == CLASS_FOG for item in connectors)
    target = (4.0 / 15.0) * fog_inventory

    def state_score(values: tuple[int, ...]) -> tuple[float, float]:
        errors = [100.0 * (value - target) / fog_inventory for value in values]
        return max(map(abs, errors)), sum(value * value for value in errors)

    states: dict[tuple[int, ...], tuple[int, tuple[tuple[int, ...], ...]]] = {
        (0, 0, 0): (0, ())
    }
    for subject in SUBJECTS:
        next_states: dict[
            tuple[int, ...], tuple[int, tuple[tuple[int, ...], ...]]
        ] = {}
        old = counts_by_subject[subject]
        unique: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
        for new_to_old in permutations:
            values = tuple(old[index] for index in new_to_old)
            metadata = (
                sum(new != old_index for new, old_index in enumerate(new_to_old)),
                new_to_old,
            )
            if values not in unique or metadata < unique[values]:
                unique[values] = metadata
        for state, (moved, path) in states.items():
            for values, (extra_moved, new_to_old) in unique.items():
                new_state = tuple(state[index] + values[index] for index in range(3))
                metadata = (moved + extra_moved, path + (new_to_old,))
                if new_state not in next_states or metadata < next_states[new_state]:
                    next_states[new_state] = metadata
        states = next_states
    best_state, (moved, path) = min(
        states.items(), key=lambda item: (*state_score(item[0]), item[1])
    )
    mappings = {subject: path[index] for index, subject in enumerate(SUBJECTS)}
    old_to_new: dict[str, dict[int, int]] = {}
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        new_to_old = mappings[subject]
        reverse = {old: new for new, old in enumerate(new_to_old)}
        old_to_new[subject] = reverse
        old_counts = counts_by_subject[subject]
        new_counts = tuple(old_counts[old] for old in new_to_old)
        rows.append(
            {
                "subject_id": subject,
                "new_fold0_uses_old_fold": new_to_old[0],
                "new_fold1_uses_old_fold": new_to_old[1],
                "new_fold2_uses_old_fold": new_to_old[2],
                "relabelled_fold_position_count": sum(
                    new != old for new, old in enumerate(new_to_old)
                ),
                "old_fog_validation_counts": json.dumps(old_counts),
                "new_fog_validation_counts": json.dumps(new_counts),
            }
        )

    for group in groups:
        if group.assigned_development_fold is not None:
            group.assigned_development_fold = old_to_new[group.subject_id][
                group.assigned_development_fold
            ]
    realigned: dict[tuple[int, str], str] = {}
    subject_by_group = {group.group_id: group.subject_id for group in groups}
    for (old_outer, group_id), role in clean_training_roles.items():
        subject = subject_by_group[group_id]
        realigned[(old_to_new[subject][old_outer], group_id)] = role
    for row in allocation_audit:
        subject = str(row["subject_id"])
        if row["allocation_stage"] == "clean_training_roles":
            row["outer_fold_id"] = old_to_new[subject][int(row["outer_fold_id"])]
        elif row["allocation_stage"].startswith("joint_"):
            counts = json.loads(str(row["actual_counts"]))
            new_to_old = mappings[subject]
            row["actual_counts"] = json.dumps(
                {
                    "permanent_test": counts["permanent_test"],
                    **{
                        f"fold{new}": counts[f"fold{old}"]
                        for new, old in enumerate(new_to_old)
                    },
                },
                sort_keys=True,
            )
    before = tuple(
        sum(counts_by_subject[subject][fold] for subject in SUBJECTS)
        for fold in range(3)
    )
    quality = {
        "pass": state_score(best_state) <= state_score(before),
        "aggregate_fog_validation_counts_before": list(before),
        "aggregate_fog_validation_counts_after": list(best_state),
        "maximum_absolute_percentage_point_error_before": state_score(before)[0],
        "maximum_absolute_percentage_point_error_after": state_score(best_state)[0],
        "subjects_relabelled": [
            subject for subject in SUBJECTS if mappings[subject] != (0, 1, 2)
        ],
        "relabelled_fold_position_count": moved,
    }
    return realigned, rows, quality


def group_role(
    group: AllocationGroup,
    outer_fold: int,
    clean_training_roles: Mapping[tuple[int, str], str],
) -> str:
    if group.permanent_partition == "permanent_test":
        return (
            PERMANENT_TEST_FOG
            if group.class_label == CLASS_FOG
            else PERMANENT_TEST_NONFOG
        )
    if (
        group.permanent_partition != "development"
        or group.assigned_development_fold is None
    ):
        raise AssertionError(f"incomplete group allocation: {group.group_id}")
    if group.assigned_development_fold == outer_fold:
        return (
            EXTERNAL_VALIDATION_FOG
            if group.class_label == CLASS_FOG
            else EXTERNAL_VALIDATION_NONFOG
        )
    if group.class_label == CLASS_FOG:
        return CLASSIFIER_TRAIN_FOG
    return clean_training_roles[(outer_fold, group.group_id)]


def candidate_base_row(
    candidate: Candidate, record_lookup: Mapping[str, Record]
) -> dict[str, Any]:
    record = record_lookup[candidate.record_id]
    source_scale = record.source_sampling_rate_hz / FS
    source_start_position = record.source_start_row + candidate.start_index * source_scale
    source_end_position = (
        record.source_start_row + candidate.end_index_exclusive * source_scale
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
        "source_start_row": int(math.floor(source_start_position)),
        "source_end_row_exclusive": int(math.ceil(source_end_position)),
        "source_start_position": source_start_position,
        "source_end_position_exclusive": source_end_position,
        "source_sampling_rate_hz": record.source_sampling_rate_hz,
        "fog_samples_in_2s": candidate.fog_samples_in_2s,
        "full_2s_fog_fraction": candidate.fog_samples_in_2s / WINDOW,
        "purity_label": candidate.purity_label,
        "class_label": candidate.class_label,
        "y_binary": candidate.y_binary,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "window_alignment": f"record_start_stride{STRIDE}",
        "label_rule": "PURE_FOG iff 128/128 FOG; PURE_NONFOG iff 0/128 FOG",
    }


def materialize_manifests(
    records: Sequence[Record],
    candidates: Sequence[Candidate],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    core_window_to_group: Mapping[str, str],
    clean_training_roles: Mapping[tuple[int, str], str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    record_lookup = {record.record_id: record for record in records}
    group_lookup = {group.group_id: group for group in groups}
    connector_by_window = {item.window_id: item for item in connectors}

    group_rows: list[dict[str, Any]] = []
    fold_group_rows: list[dict[str, Any]] = []
    for group in sorted(
        groups, key=lambda item: (item.subject_id, item.class_label, item.group_id)
    ):
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
                "parent_event_id": (
                    "" if group.parent_event_id is None else group.parent_event_id
                ),
                "subblock_ordinal": group.subblock_ordinal,
                "split_from_continuous_event": group.split_from_continuous_event,
                "eligible_for_allocation": group.eligible_for_allocation,
                "permanent_partition": group.permanent_partition,
                "assigned_development_fold": (
                    ""
                    if group.assigned_development_fold is None
                    else group.assigned_development_fold
                ),
                "base_partition": base_partition_key(group),
                "allocation_unit_indivisible": True,
            }
        )
        for outer_fold in range(OUTER_FOLDS):
            role = group_role(group, outer_fold, clean_training_roles)
            fold_group_rows.append(
                {
                    "outer_fold_id": outer_fold,
                    "group_id": group.group_id,
                    "group_kind": group.group_kind,
                    "class_label": group.class_label,
                    "subject_id": group.subject_id,
                    "record_id": group.record_id,
                    "permanent_partition": group.permanent_partition,
                    "assigned_development_fold": (
                        ""
                        if group.assigned_development_fold is None
                        else group.assigned_development_fold
                    ),
                    "base_partition": base_partition_key(group),
                    "final_role": role,
                    "active_for_outer_fold": True,
                    "core_window_count": group.window_count,
                }
            )

    connector_rows = [
        {
            "connector_id": item.connector_id,
            "window_id": item.window_id,
            "class_label": item.class_label,
            "boundary_kind": item.boundary_kind,
            "subject_id": item.subject_id,
            "record_id": item.record_id,
            "run_id": item.run_id,
            "segment_id": item.segment_id,
            "start_index": item.start_index,
            "end_index_exclusive": item.end_index_exclusive,
            "start_time_sec": item.start_index / FS,
            "end_time_sec": item.end_index_exclusive / FS,
            "left_group_id": item.left_group_id,
            "right_group_id": item.right_group_id,
            "left_base_partition": base_partition_key(group_lookup[item.left_group_id]),
            "right_base_partition": base_partition_key(group_lookup[item.right_group_id]),
            "parent_event_id": (
                "" if item.parent_event_id is None else item.parent_event_id
            ),
            "retention_policy": (
                "exclude in all rotations when base partitions differ; otherwise retain "
                "only when both final roles match"
            ),
        }
        for item in sorted(
            connectors, key=lambda value: (value.subject_id, value.record_id, value.start_index)
        )
    ]

    window_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.purity_label == MIXED:
            row = candidate_base_row(candidate, record_lookup)
            row.update(
                {
                    "outer_fold_id": "",
                    "exclusion_stage": "purity_filter",
                    "exclusion_reason": "mixed_boundary_window",
                    "connector_id": "",
                    "left_group_id": "",
                    "right_group_id": "",
                    "left_base_partition": "",
                    "right_base_partition": "",
                    "left_role": "",
                    "right_role": "",
                }
            )
            excluded_rows.append(row)
            continue

        connector = connector_by_window.get(candidate.window_id)
        if connector is not None:
            left_group = group_lookup[connector.left_group_id]
            right_group = group_lookup[connector.right_group_id]
            left_base = base_partition_key(left_group)
            right_base = base_partition_key(right_group)
            for outer_fold in range(OUTER_FOLDS):
                left_role = group_role(left_group, outer_fold, clean_training_roles)
                right_role = group_role(right_group, outer_fold, clean_training_roles)
                same_base = left_base == right_base
                if same_base and left_role == right_role:
                    row = candidate_base_row(candidate, record_lookup)
                    row.update(
                        {
                            "outer_fold_id": outer_fold,
                            "final_role": left_role,
                            "role_code": ROLE_CODES[left_role],
                            "active_for_outer_fold": True,
                            "allocation_group_id": "",
                            "group_kind": (
                                "nonfog_connector"
                                if connector.class_label == CLASS_NONFOG
                                else "fog_connector"
                            ),
                            "connector_id": connector.connector_id,
                            "left_group_id": connector.left_group_id,
                            "right_group_id": connector.right_group_id,
                            "left_base_partition": left_base,
                            "right_base_partition": right_base,
                            "is_dynamic_connector": True,
                        }
                    )
                    window_rows.append(row)
                else:
                    row = candidate_base_row(candidate, record_lookup)
                    row.update(
                        {
                            "outer_fold_id": outer_fold,
                            "exclusion_stage": "cross_pool_boundary",
                            "exclusion_reason": EXCLUDED_BOUNDARY,
                            "connector_id": connector.connector_id,
                            "left_group_id": connector.left_group_id,
                            "right_group_id": connector.right_group_id,
                            "left_base_partition": left_base,
                            "right_base_partition": right_base,
                            "left_role": left_role,
                            "right_role": right_role,
                        }
                    )
                    excluded_rows.append(row)
            continue

        group_id = core_window_to_group[candidate.window_id]
        group = group_lookup[group_id]
        for outer_fold in range(OUTER_FOLDS):
            role = group_role(group, outer_fold, clean_training_roles)
            row = candidate_base_row(candidate, record_lookup)
            row.update(
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
                    "left_base_partition": "",
                    "right_base_partition": "",
                    "is_dynamic_connector": False,
                }
            )
            window_rows.append(row)

    window_rows.sort(
        key=lambda row: (
            row["subject_id"],
            int(row["outer_fold_id"]),
            ROLE_CODES[str(row["final_role"])],
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


def build_split_summary(
    candidates: Sequence[Candidate],
    window_rows: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory = Counter(
        (item.subject_id, item.class_label)
        for item in candidates
        if item.purity_label != MIXED
    )
    rows: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLDS):
        fold_windows = [
            row for row in window_rows if int(row["outer_fold_id"]) == outer_fold
        ]
        fold_excluded = [
            row
            for row in excluded_rows
            if row["exclusion_reason"] == EXCLUDED_BOUNDARY
            and int(row["outer_fold_id"]) == outer_fold
        ]
        for subject in (*SUBJECTS, "ALL"):
            selected = (
                fold_windows
                if subject == "ALL"
                else [row for row in fold_windows if row["subject_id"] == subject]
            )
            excluded = (
                fold_excluded
                if subject == "ALL"
                else [row for row in fold_excluded if row["subject_id"] == subject]
            )
            for class_label in (CLASS_NONFOG, CLASS_FOG):
                full_inventory = (
                    sum(inventory[(item, class_label)] for item in SUBJECTS)
                    if subject == "ALL"
                    else inventory[(subject, class_label)]
                )
                retained_class = sum(
                    row["class_label"] == class_label for row in selected
                )
                class_roles = [
                    role for role in ACTIVE_ROLES if ROLE_CLASS[role] == class_label
                ]
                for role in class_roles:
                    role_rows = [row for row in selected if row["final_role"] == role]
                    target = ROLE_TARGETS[role]
                    fraction_full = len(role_rows) / full_inventory
                    fraction_retained = len(role_rows) / retained_class
                    rows.append(
                        {
                            "subject_id": subject,
                            "outer_fold_id": outer_fold,
                            "pool_role": role,
                            "class_label": class_label,
                            "window_count": len(role_rows),
                            "allocation_group_count": len(
                                {
                                    row["allocation_group_id"]
                                    for row in role_rows
                                    if row["allocation_group_id"]
                                }
                            ),
                            "dynamic_connector_count": sum(
                                bool(row["is_dynamic_connector"]) for row in role_rows
                            ),
                            "full_class_inventory_before_boundary_exclusions": full_inventory,
                            "retained_active_class_window_count": retained_class,
                            "window_fraction_of_full_class_inventory": fraction_full,
                            "window_fraction_of_retained_active_class_windows": fraction_retained,
                            "target_fraction_of_full_class_inventory": target,
                            "target_fraction_of_retained_active_class_windows": target,
                            "absolute_target_window_count": target * retained_class,
                            "window_count_deviation_from_retained_target": (
                                len(role_rows) - target * retained_class
                            ),
                            "percentage_point_deviation_from_target": (
                                100.0 * (fraction_full - target)
                            ),
                            "retained_percentage_point_deviation_from_target": (
                                100.0 * (fraction_retained - target)
                            ),
                        }
                    )
                boundary_rows = [
                    row for row in excluded if row["class_label"] == class_label
                ]
                rows.append(
                    {
                        "subject_id": subject,
                        "outer_fold_id": outer_fold,
                        "pool_role": EXCLUDED_BOUNDARY,
                        "class_label": class_label,
                        "window_count": len(boundary_rows),
                        "allocation_group_count": 0,
                        "dynamic_connector_count": len(boundary_rows),
                        "full_class_inventory_before_boundary_exclusions": full_inventory,
                        "retained_active_class_window_count": retained_class,
                        "window_fraction_of_full_class_inventory": (
                            len(boundary_rows) / full_inventory
                        ),
                        "window_fraction_of_retained_active_class_windows": "",
                        "target_fraction_of_full_class_inventory": 0.0,
                        "target_fraction_of_retained_active_class_windows": 0.0,
                        "absolute_target_window_count": 0.0,
                        "window_count_deviation_from_retained_target": len(boundary_rows),
                        "percentage_point_deviation_from_target": (
                            100.0 * len(boundary_rows) / full_inventory
                        ),
                        "retained_percentage_point_deviation_from_target": "",
                    }
                )
    return rows


def cross_role_overlap_audit(
    window_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_partition: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        by_partition[
            (row["subject_id"], row["record_id"], int(row["outer_fold_id"]))
        ].append(row)
    violations: list[dict[str, Any]] = []
    same_role_overlap_pairs = 0
    for (subject, record_id, outer_fold), rows in sorted(by_partition.items()):
        ordered = sorted(rows, key=lambda row: int(row["start_index"]))
        for left_index, left in enumerate(ordered):
            left_end = int(left["end_index_exclusive"])
            for right in ordered[left_index + 1 :]:
                if int(right["start_index"]) >= left_end:
                    break
                if left["final_role"] == right["final_role"]:
                    same_role_overlap_pairs += 1
                else:
                    violations.append(
                        {
                            "subject_id": subject,
                            "record_id": record_id,
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
        "same_role_overlap_pair_count": same_role_overlap_pairs,
        "examples": violations[:20],
    }


def quality_report(
    records: Sequence[Record],
    events: Sequence[dict[str, Any]],
    candidates: Sequence[Candidate],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    core_window_to_group: Mapping[str, str],
    window_rows: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
    split_summary: Sequence[dict[str, Any]],
    fold_alignment_quality: Mapping[str, Any],
) -> dict[str, Any]:
    purity_counts = Counter(item.purity_label for item in candidates)
    confirmed = {
        "record_count": len(records),
        "subject_count": len({record.subject_id for record in records}),
        "sample_count": sum(len(record.y) for record in records),
        "nonfog_sample_count": sum(
            int(np.count_nonzero(record.y == 0)) for record in records
        ),
        "fog_sample_count": sum(
            int(np.count_nonzero(record.y == 1)) for record in records
        ),
        "fog_event_count": len(events),
        "candidate_window_count": len(candidates),
        "pure_nonfog_window_count": purity_counts[PURE_NONFOG],
        "pure_fog_window_count": purity_counts[PURE_FOG],
        "mixed_window_count": purity_counts[MIXED],
    }
    inventory_match = confirmed == EXPECTED_INVENTORY

    candidate_ids = {item.window_id for item in candidates}
    pure_ids = {
        item.window_id for item in candidates if item.purity_label != MIXED
    }
    mixed_ids = candidate_ids - pure_ids
    connector_ids = {item.window_id for item in connectors}
    grouped_ids = set(core_window_to_group)
    group_partition_pass = bool(
        not (connector_ids & grouped_ids)
        and connector_ids | grouped_ids == pure_ids
        and sum(group.window_count for group in groups) == len(grouped_ids)
    )

    mixed_excluded = [
        row
        for row in excluded_rows
        if row["exclusion_reason"] == "mixed_boundary_window"
    ]
    mixed_exclusion_pass = bool(
        len(mixed_excluded) == len(mixed_ids)
        and {row["window_id"] for row in mixed_excluded} == mixed_ids
        and all(row["outer_fold_id"] == "" for row in mixed_excluded)
    )
    boundary_excluded = [
        row for row in excluded_rows if row["exclusion_reason"] == EXCLUDED_BOUNDARY
    ]
    dynamic_active = [row for row in window_rows if row["is_dynamic_connector"]]
    connector_decisions_pass = bool(
        len(boundary_excluded) + len(dynamic_active) == len(connectors) * OUTER_FOLDS
    )

    fold_accounting_failures: list[dict[str, Any]] = []
    duplicate_active_windows = 0
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
        duplicate_active_windows += len(active) - len(active_ids)
        if active_ids & purged_ids or active_ids | purged_ids != pure_ids:
            fold_accounting_failures.append(
                {
                    "outer_fold_id": outer_fold,
                    "active_count": len(active),
                    "boundary_excluded_count": len(purged),
                    "missing_pure_window_count": len(pure_ids - active_ids - purged_ids),
                    "active_excluded_overlap_count": len(active_ids & purged_ids),
                }
            )

    candidate_lookup = {item.window_id: item for item in candidates}
    nonpure_active = sum(
        candidate_lookup[row["window_id"]].purity_label == MIXED for row in window_rows
    )
    role_class_mismatches = [
        row["window_id"]
        for row in window_rows
        if ROLE_CLASS[row["final_role"]] != row["class_label"]
        or int(row["y_binary"]) != int(row["class_label"] == CLASS_FOG)
    ]
    fog_in_nbm = sum(
        row["y_binary"] == 1
        and row["final_role"] in {NBM_TRAIN_CLEAN, NBM_EARLYSTOP_CLEAN}
        for row in window_rows
    )

    fixed_test_failures: list[str] = []
    for subject in SUBJECTS:
        for role in (PERMANENT_TEST_NONFOG, PERMANENT_TEST_FOG):
            sets = [
                {
                    row["window_id"]
                    for row in window_rows
                    if row["subject_id"] == subject
                    and int(row["outer_fold_id"]) == outer_fold
                    and row["final_role"] == role
                }
                for outer_fold in range(OUTER_FOLDS)
            ]
            if any(values != sets[0] for values in sets[1:]):
                fixed_test_failures.append(f"{subject}:{role}")

    group_roles: dict[tuple[str, int], set[str]] = defaultdict(set)
    group_window_counts: Counter[tuple[str, int]] = Counter()
    for row in window_rows:
        group_id = str(row["allocation_group_id"])
        if group_id:
            key = (group_id, int(row["outer_fold_id"]))
            group_roles[key].add(str(row["final_role"]))
            group_window_counts[key] += 1
    rotation_failures: list[str] = []
    for group in groups:
        observed: list[str] = []
        for outer_fold in range(OUTER_FOLDS):
            key = (group.group_id, outer_fold)
            roles = group_roles[key]
            if len(roles) != 1 or group_window_counts[key] != group.window_count:
                rotation_failures.append(
                    f"{group.group_id}:outer{outer_fold}:roles={sorted(roles)}:"
                    f"count={group_window_counts[key]}/{group.window_count}"
                )
                observed.append("")
            else:
                observed.append(next(iter(roles)))
        if group.permanent_partition == "permanent_test":
            expected_role = (
                PERMANENT_TEST_FOG
                if group.class_label == CLASS_FOG
                else PERMANENT_TEST_NONFOG
            )
            if any(role != expected_role for role in observed):
                rotation_failures.append(f"{group.group_id}:permanent:{observed}")
        else:
            validation_role = (
                EXTERNAL_VALIDATION_FOG
                if group.class_label == CLASS_FOG
                else EXTERNAL_VALIDATION_NONFOG
            )
            training_roles = (
                {CLASSIFIER_TRAIN_FOG}
                if group.class_label == CLASS_FOG
                else {NBM_TRAIN_CLEAN, NBM_EARLYSTOP_CLEAN, CLASSIFIER_TRAIN_CLEAN}
            )
            if (
                sum(role == validation_role for role in observed) != 1
                or sum(role in training_roles for role in observed) != 2
                or observed[group.assigned_development_fold or 0] != validation_role
            ):
                rotation_failures.append(f"{group.group_id}:development:{observed}")

    empty_role_failures: list[str] = []
    for subject in SUBJECTS:
        for outer_fold in range(OUTER_FOLDS):
            present = {
                row["final_role"]
                for row in window_rows
                if row["subject_id"] == subject
                and int(row["outer_fold_id"]) == outer_fold
            }
            for role in ACTIVE_ROLES:
                if role not in present:
                    empty_role_failures.append(f"{subject}:outer{outer_fold}:{role}")

    fixed_base_boundary_active = []
    connector_lookup = {item.connector_id: item for item in connectors}
    group_lookup = {group.group_id: group for group in groups}
    for row in dynamic_active:
        connector = connector_lookup[row["connector_id"]]
        if base_partition_key(group_lookup[connector.left_group_id]) != base_partition_key(
            group_lookup[connector.right_group_id]
        ):
            fixed_base_boundary_active.append(row["window_id"])

    size_failures = [
        group.group_id
        for group in groups
        if group.window_count
        > (
            MAX_NONFOG_CORE_WINDOWS
            if group.class_label == CLASS_NONFOG
            else MAX_FOG_CORE_WINDOWS
        )
    ]
    p08_split_groups = [
        group
        for group in groups
        if group.subject_id == "P08"
        and group.class_label == CLASS_FOG
        and group.split_from_continuous_event
    ]
    p08_fog_connectors = [
        item
        for item in connectors
        if item.subject_id == "P08" and item.class_label == CLASS_FOG
    ]
    p08_split_pass = bool(
        sorted(group.window_count for group in p08_split_groups) == [3, 4]
        and len(p08_fog_connectors) == 1
    )

    overlap = cross_role_overlap_audit(window_rows)
    ratio_rows = [
        row
        for row in split_summary
        if row["pool_role"] in ACTIVE_ROLES
        and row["retained_percentage_point_deviation_from_target"] != ""
    ]
    subject_ratio_rows = [row for row in ratio_rows if row["subject_id"] != "ALL"]
    aggregate_ratio_rows = [row for row in ratio_rows if row["subject_id"] == "ALL"]
    worst_subject = max(
        subject_ratio_rows,
        key=lambda row: abs(float(row["retained_percentage_point_deviation_from_target"])),
    )
    worst_aggregate = max(
        aggregate_ratio_rows,
        key=lambda row: abs(float(row["retained_percentage_point_deviation_from_target"])),
    )
    aggregate_ratio_pass = bool(
        abs(float(worst_aggregate["retained_percentage_point_deviation_from_target"]))
        <= 1.25 + 1e-12
    )

    report = {
        "overall_pass": False,
        "confirmed_source_inventory": confirmed,
        "expected_source_inventory": EXPECTED_INVENTORY,
        "confirmed_source_inventory_match": inventory_match,
        "subjects": sorted({record.subject_id for record in records}),
        "record_npz_contract_pass": all(
            record.x.dtype == np.float32
            and record.y.dtype == np.int8
            and record.x.shape == (len(record.y), 30)
            and np.isfinite(record.x).all()
            and np.isin(record.y, (0, 1)).all()
            for record in records
        ),
        "candidate_identity_partition_reconciles": group_partition_pass,
        "mixed_exclusion_audit_complete": mixed_exclusion_pass,
        "connector_decisions_complete": connector_decisions_pass,
        "fixed_base_boundary_connectors_active": fixed_base_boundary_active,
        "fold_candidate_accounting_failures": fold_accounting_failures,
        "duplicate_active_window_rows_within_outer_fold": duplicate_active_windows,
        "mixed_or_nonpure_windows_active": nonpure_active,
        "role_class_or_label_mismatches": role_class_mismatches[:20],
        "fog_windows_in_nbm_fit_or_earlystop": fog_in_nbm,
        "permanent_test_fixed_across_outer_folds_failures": fixed_test_failures,
        "group_rotation_failures": rotation_failures,
        "required_active_role_empty_failures": empty_role_failures,
        "group_core_size_limit_failures": size_failures,
        "p08_fog_3_connector_4_split_pass": p08_split_pass,
        "cross_role_raw_overlap_audit": overlap,
        "fold_label_alignment_audit": dict(fold_alignment_quality),
        "mixed_window_count": len(mixed_ids),
        "allocation_group_count": len(groups),
        "connector_count": len(connectors),
        "boundary_exclusion_rows_across_outer_folds": len(boundary_excluded),
        "boundary_exclusion_window_counts_by_outer_fold": {
            str(fold): sum(
                int(row["outer_fold_id"]) == fold for row in boundary_excluded
            )
            for fold in range(OUTER_FOLDS)
        },
        "active_window_rows_across_outer_folds": len(window_rows),
        "ratio_quality_retained_window_denominator": {
            "pass": aggregate_ratio_pass,
            "aggregate_tolerance_percentage_points": 1.25,
            "subject_level_is_reported_not_gated_due_to_integer_granularity": True,
            "worst_subject_row": dict(worst_subject),
            "worst_aggregate_row": dict(worst_aggregate),
        },
    }
    report["overall_pass"] = bool(
        inventory_match
        and report["record_npz_contract_pass"]
        and group_partition_pass
        and mixed_exclusion_pass
        and connector_decisions_pass
        and not fixed_base_boundary_active
        and not fold_accounting_failures
        and duplicate_active_windows == 0
        and nonpure_active == 0
        and not role_class_mismatches
        and fog_in_nbm == 0
        and not fixed_test_failures
        and not rotation_failures
        and not empty_role_failures
        and not size_failures
        and p08_split_pass
        and overlap["pass"]
        and bool(fold_alignment_quality["pass"])
        and aggregate_ratio_pass
    )
    return report


def build_fog_event_manifest(
    events: Sequence[dict[str, Any]],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
) -> list[dict[str, Any]]:
    groups_by_event: dict[tuple[str, int], list[AllocationGroup]] = defaultdict(list)
    connectors_by_event: dict[tuple[str, int], list[Connector]] = defaultdict(list)
    for group in groups:
        if group.class_label == CLASS_FOG and group.parent_event_id is not None:
            groups_by_event[(group.record_id, group.parent_event_id)].append(group)
    for item in connectors:
        if item.class_label == CLASS_FOG and item.parent_event_id is not None:
            connectors_by_event[(item.record_id, item.parent_event_id)].append(item)
    rows: list[dict[str, Any]] = []
    for event in events:
        key = (str(event["record_id"]), int(event["event_id"]))
        event_groups = sorted(groups_by_event[key], key=lambda item: item.start_index)
        event_connectors = sorted(
            connectors_by_event[key], key=lambda item: item.start_index
        )
        row = dict(event)
        row.update(
            {
                "nbm_allocation_group_ids": ";".join(
                    group.group_id for group in event_groups
                ),
                "nbm_allocation_group_count": len(event_groups),
                "nbm_pure_fog_core_window_count": sum(
                    group.window_count for group in event_groups
                ),
                "nbm_connector_window_ids": ";".join(
                    item.window_id for item in event_connectors
                ),
                "nbm_connector_count": len(event_connectors),
                "nbm_continuous_event_split": len(event_groups) > 1,
                "nbm_status": "eligible" if event_groups else "no_pure_2s_fog_window",
            }
        )
        rows.append(row)
    return rows


def save_indices(
    output: Path,
    window_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
) -> dict[str, Any]:
    index_dir = output / "split_indices"
    index_dir.mkdir(parents=True, exist_ok=False)
    record_index = {
        str(row["record_id"]): index for index, row in enumerate(manifest_rows)
    }
    ordered_groups = sorted(
        groups, key=lambda item: (item.subject_id, item.class_label, item.group_id)
    )
    group_index = {group.group_id: index for index, group in enumerate(ordered_groups)}
    ordered_connectors = sorted(
        connectors, key=lambda item: (item.subject_id, item.record_id, item.start_index)
    )
    connector_index = {
        item.connector_id: index for index, item in enumerate(ordered_connectors)
    }
    write_csv(
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
    write_csv(
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
            for index, group in enumerate(ordered_groups)
        ],
    )
    write_csv(
        index_dir / "connector_lookup.csv",
        [
            {
                "connector_index": index,
                "connector_id": item.connector_id,
                "window_id": item.window_id,
                "left_group_id": item.left_group_id,
                "right_group_id": item.right_group_id,
            }
            for index, item in enumerate(ordered_connectors)
        ],
    )

    expected_keys = {
        "record_index",
        "record_id",
        "start_index",
        "end_index_exclusive",
        "role_code",
        "y_binary",
        "group_index",
        "allocation_group_id",
        "connector_index",
        "connector_id",
        "left_group_id",
        "right_group_id",
        "is_dynamic_connector",
        "window_id",
    }
    verification_problems: list[str] = []
    verified_rows = 0
    for subject in SUBJECTS:
        for outer_fold in range(OUTER_FOLDS):
            selected = [
                row
                for row in window_rows
                if row["subject_id"] == subject
                and int(row["outer_fold_id"]) == outer_fold
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
            np.savez_compressed(path, **payload)
            with np.load(path, allow_pickle=False) as restored:
                if set(restored.files) != expected_keys:
                    verification_problems.append(
                        f"{path.name}:keys={sorted(restored.files)}"
                    )
                lengths = {key: len(restored[key]) for key in restored.files}
                unequal = [
                    key
                    for key, expected in payload.items()
                    if key not in restored.files
                    or not np.array_equal(restored[key], expected)
                ]
            if unequal or any(length != len(selected) for length in lengths.values()):
                verification_problems.append(
                    f"{path.name}:unequal={unequal}:lengths={lengths}"
                )
            verified_rows += len(selected)
    return {
        "pass": not verification_problems,
        "verified_file_count": len(SUBJECTS) * OUTER_FOLDS,
        "expected_file_count": 24,
        "verified_row_count": verified_rows,
        "expected_keys": sorted(expected_keys),
        "problems": verification_problems[:20],
    }


def build_loso_rows(manifest_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_subject in SUBJECTS:
        for record in manifest_rows:
            rows.append(
                {
                    "fold_id": f"loso_{test_subject}",
                    "test_subject_id": test_subject,
                    "split": (
                        "test" if record["subject_id"] == test_subject else "train"
                    ),
                    "record_id": record["record_id"],
                    "subject_id": record["subject_id"],
                    "segment_id": record["segment_id"],
                }
            )
    return rows


def ordered_manifest_rows(
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred = [
        "dataset_id",
        "subset_id",
        "record_id",
        "record_path",
        "source_file",
        "source_subset_dir",
        "subject_id",
        "run_id",
        "segment_id",
        "visit",
        "source_condition_token",
        "trial_id",
        "source_start_row",
        "source_end_row",
        "source_start_time",
        "source_end_time",
        "sampling_rate_hz",
        "estimated_sampling_rate_hz",
        "n_samples",
        "duration_sec",
        "n_normal_samples",
        "n_fog_samples",
        "fog_event_count",
        "has_fog",
        "usable",
        "notes",
        "source_sampling_rate_hz",
        "source_n_samples",
        "source_record_path",
        "downsampling_method",
        "label_resampling",
        "start_pc_world_datetime_local",
        "end_pc_world_datetime_local",
    ]
    extra = sorted(
        {
            key
            for row in manifest_rows
            for key in row
            if key not in set(preferred)
        }
    )
    fields = preferred + extra
    return [{field: row.get(field, "") for field in fields} for row in manifest_rows]


def record_integrity_rows(
    records: Sequence[Record], record_hashes: Mapping[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": record.record_id,
            "subject_id": record.subject_id,
            "segment_id": record.segment_id,
            "n_samples": len(record.y),
            "n_channels": record.x.shape[1],
            "x_dtype": str(record.x.dtype),
            "y_binary_dtype": str(record.y.dtype),
            "finite_signal": bool(np.isfinite(record.x).all()),
            "binary_labels": bool(np.isin(record.y, (0, 1)).all()),
            "n_normal_samples": int(np.count_nonzero(record.y == 0)),
            "n_fog_samples": int(np.count_nonzero(record.y == 1)),
            "source_sampling_rate_hz": record.source_sampling_rate_hz,
            "processed_sampling_rate_hz": FS,
            "record_sha256": record_hashes[record.record_id],
        }
        for record in records
    ]


def build_pool_count_report(
    split_summary: Sequence[dict[str, Any]], quality: Mapping[str, Any]
) -> str:
    labels = {
        PERMANENT_TEST_NONFOG: "永久测试 Non-FoG",
        PERMANENT_TEST_FOG: "永久测试 FoG",
        EXTERNAL_VALIDATION_NONFOG: "外部验证 Non-FoG",
        EXTERNAL_VALIDATION_FOG: "外部验证 FoG",
        NBM_TRAIN_CLEAN: "NBM 参数训练 Non-FoG",
        NBM_EARLYSTOP_CLEAN: "NBM 内部早停 Non-FoG",
        CLASSIFIER_TRAIN_CLEAN: "分类器训练 Non-FoG",
        CLASSIFIER_TRAIN_FOG: "分类器训练 FoG",
    }
    lines = [
        "# processed_NBM_Exp 池样本数报告",
        "",
        "## 严格窗口库存",
        "",
        f"- 候选窗口：{quality['confirmed_source_inventory']['candidate_window_count']:,}",
        f"- 纯 Non-FoG：{quality['confirmed_source_inventory']['pure_nonfog_window_count']:,}",
        f"- 纯 FoG：{quality['confirmed_source_inventory']['pure_fog_window_count']:,}",
        f"- 混合窗口（删除）：{quality['confirmed_source_inventory']['mixed_window_count']:,}",
        "",
        "实际比例以每一轮、每一类别最终保留的激活窗口数为分母。",
        "",
    ]
    aggregate = [row for row in split_summary if row["subject_id"] == "ALL"]
    for outer_fold in range(OUTER_FOLDS):
        lines.extend(
            [
                f"## 三折轮转 {outer_fold}",
                "",
                "| 池 | 窗口数 | 保留后比例 | 目标比例 | 偏差（百分点） |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for role in ACTIVE_ROLES:
            row = next(
                item
                for item in aggregate
                if int(item["outer_fold_id"]) == outer_fold
                and item["pool_role"] == role
            )
            lines.append(
                f"| {labels[role]} | {int(row['window_count']):,} | "
                f"{float(row['window_fraction_of_retained_active_class_windows']):.2%} | "
                f"{float(row['target_fraction_of_retained_active_class_windows']):.2%} | "
                f"{float(row['retained_percentage_point_deviation_from_target']):+.2f} |"
            )
        for class_label in (CLASS_NONFOG, CLASS_FOG):
            row = next(
                item
                for item in aggregate
                if int(item["outer_fold_id"]) == outer_fold
                and item["pool_role"] == EXCLUDED_BOUNDARY
                and item["class_label"] == class_label
            )
            lines.append(
                f"| 跨池边界删除 {class_label} | {int(row['window_count']):,} | - | 0 | - |"
            )
        lines.append("")

    lines.extend(
        [
            "## 各受试者明细",
            "",
            "| 受试者 | 轮转 | test Non | test FoG | val Non | val FoG | "
            "NBM train | NBM early | clf Non | clf FoG | 边界删除 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for subject in SUBJECTS:
        for outer_fold in range(OUTER_FOLDS):
            selected = [
                row
                for row in split_summary
                if row["subject_id"] == subject
                and int(row["outer_fold_id"]) == outer_fold
            ]
            count = {
                role: int(
                    next(row for row in selected if row["pool_role"] == role)[
                        "window_count"
                    ]
                )
                for role in ACTIVE_ROLES
            }
            boundary = sum(
                int(row["window_count"])
                for row in selected
                if row["pool_role"] == EXCLUDED_BOUNDARY
            )
            lines.append(
                f"| {subject} | {outer_fold} | {count[PERMANENT_TEST_NONFOG]} | "
                f"{count[PERMANENT_TEST_FOG]} | {count[EXTERNAL_VALIDATION_NONFOG]} | "
                f"{count[EXTERNAL_VALIDATION_FOG]} | {count[NBM_TRAIN_CLEAN]} | "
                f"{count[NBM_EARLYSTOP_CLEAN]} | {count[CLASSIFIER_TRAIN_CLEAN]} | "
                f"{count[CLASSIFIER_TRAIN_FOG]} | {boundary} |"
            )
    lines.extend(
        [
            "",
            "## 质量门控",
            "",
            f"- 总体：{'PASS' if quality['overall_pass'] else 'FAIL'}",
            "- 不同池之间原始样本重叠：0",
            "- 激活混合窗口：0",
            "- FoG 进入 NBM 参数训练或早停：0",
            "- 永久测试在三轮中固定：是",
            "- P08 长 FoG 段：3 窗 + 1 个连接窗 + 4 窗",
            "",
        ]
    )
    return "\n".join(lines)


def build_schema(
    source_root: Path, quality: Mapping[str, Any]
) -> dict[str, Any]:
    source_schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    source_schema.update(
        {
            "dataset_id": DATASET_ID,
            "subset_id": SUBSET_ID,
            "sampling_rate_hz": FS,
            "subject_ids": list(SUBJECTS),
            "record_count": EXPECTED_INVENTORY["record_count"],
        }
    )
    notes = list(source_schema.get("notes", []))
    notes.extend(
        [
            "P08 is included; all records contain five IMU slots and 30 channels.",
            "P01-P07 use 128-to-64 Hz FIR decimation; P08 uses a separately designed "
            "100-to-64 Hz FIR plus interpolation. One coefficient table is not shared.",
            "P08 anatomical sensor-name mapping was not independently re-verified; channel "
            "slot order is preserved exactly from its prepared five-IMU record.",
        ]
    )
    source_schema["notes"] = notes
    source_schema["nbm_split"] = {
        "window_manifest": "nbm_window_manifest.csv",
        "group_manifest": "nbm_group_manifest.csv",
        "fold_group_roles": "nbm_fold_group_roles.csv",
        "connector_manifest": "nbm_connector_manifest.csv",
        "excluded_window_audit": "nbm_excluded_window_audit.csv",
        "split_summary": "nbm_split_summary.csv",
        "quality_report": "nbm_quality_report.json",
        "protocol": "nbm_protocol.json",
        "role_codes": "nbm_role_codes.json",
        "split_indices_directory": "split_indices",
        "split_index_npz_pattern": (
            "split_indices/{subject_id}_outer{fold_id}_nbm_indices.npz"
        ),
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "outer_folds": OUTER_FOLDS,
        "pure_fog_definition": "fog_samples_in_2s == 128",
        "pure_nonfog_definition": "fog_samples_in_2s == 0",
        "mixed_window_policy": "exclude 1..127 FoG samples",
        "allocation_scope": "within each subject",
        "target_fractions": ROLE_TARGETS,
        "quality_pass": quality["overall_pass"],
    }
    return source_schema


def build_protocol(
    source_root: Path,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "subset_id": SUBSET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_records": str((source_root / "records").resolve()),
        "source_manifest": str((source_root / "manifest.csv").resolve()),
        "source_manifest_limitation": (
            "source metadata contains P01-P07 only; output metadata was rebuilt from all "
            "62 record files and explicit P08 provenance"
        ),
        "sampling_rate_hz": FS,
        "window_seconds": 2.0,
        "window_samples": WINDOW,
        "stride_seconds": 1.0,
        "stride_samples": STRIDE,
        "window_anchor": "sample 0 of each independent continuous record",
        "pure_fog_rule": "all 128 labels are 1",
        "pure_nonfog_rule": "all 128 labels are 0",
        "mixed_rule": "exclude every window containing both 0 and 1",
        "extra_fog_guard_applied": False,
        "scope": "independent allocation within every subject P01-P08",
        "base_partition": {
            "method": "joint deterministic retained-window optimization",
            "targets": {
                "permanent_test": 0.20,
                "development_fold0": 4.0 / 15.0,
                "development_fold1": 4.0 / 15.0,
                "development_fold2": 4.0 / 15.0,
            },
            "outer_rotation": "one development fold validates; the other two train",
        },
        "training_pool_targets": {
            NBM_TRAIN_CLEAN: 0.256,
            NBM_EARLYSTOP_CLEAN: 0.064,
            CLASSIFIER_TRAIN_CLEAN: 16.0 / 75.0,
            CLASSIFIER_TRAIN_FOG: 8.0 / 15.0,
        },
        "continuous_grouping": {
            "nonfog_maximum_core_windows": MAX_NONFOG_CORE_WINDOWS,
            "nonfog_maximum_core_support_seconds": (
                MAX_NONFOG_CORE_WINDOWS + 1
            ),
            "fog_maximum_core_windows": MAX_FOG_CORE_WINDOWS,
            "fog_maximum_core_support_seconds": MAX_FOG_CORE_WINDOWS + 1,
            "balanced_partition": True,
            "connector_between_neighboring_blocks": "one 128-sample candidate",
            "p08_long_fog_split": "3 core windows + 1 connector + 4 core windows",
        },
        "connector_policy": (
            "if neighboring base partitions differ, exclude in all rotations; otherwise "
            "retain only in rotations where both final roles match"
        ),
        "allocation_objective": [
            "minimize maximum absolute retained-window percentage-point error",
            "minimize squared retained-window percentage-point error",
            "maximize retained-window count",
        ],
        "ratio_denominator": "active retained windows of the same class in that rotation",
        "source_inventory": quality["confirmed_source_inventory"],
        "preprocessing_provenance": {
            "P01-P07": (
                "source records already FIR low-pass filtered at 128 Hz and decimated to "
                "64 Hz; 65 taps, cutoff 28 Hz, Kaiser beta 5"
            ),
            "P08": (
                "source records already FIR low-pass filtered at 100 Hz then interpolated "
                "to 64 Hz; 65 taps, cutoff 28 Hz, Kaiser beta 5"
            ),
            "signals_reprocessed_by_this_script": False,
            "labels_reprocessed_by_this_script": False,
        },
        "forbidden": [
            "random window allocation",
            "mixed windows in active pools",
            "cross-role raw-sample overlap within an outer rotation",
            "FoG windows in NBM fitting or early stopping",
            "permanent-test use for tuning, calibration, or threshold selection",
        ],
    }


def validate_written_dataset(
    build: Path,
    source_root: Path,
    manifest_rows: Sequence[dict[str, Any]],
    record_hashes: Mapping[str, str],
    window_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    problems: list[str] = []
    persisted_manifest = read_csv(build / "manifest.csv")
    if len(persisted_manifest) != 62:
        problems.append(f"manifest_rows={len(persisted_manifest)}")
    if {row["subject_id"] for row in persisted_manifest} != set(SUBJECTS):
        problems.append("manifest_subjects")
    for row in manifest_rows:
        record_id = str(row["record_id"])
        target = build / str(row["record_path"])
        source = source_root / str(row["record_path"])
        if not target.is_file() or sha256(target) != record_hashes[record_id]:
            problems.append(f"record_copy_hash:{record_id}")
        if sha256(source) != record_hashes[record_id]:
            problems.append(f"source_record_changed:{record_id}")
    schema = json.loads((build / "schema.json").read_text(encoding="utf-8"))
    if schema.get("sampling_rate_hz") != 64 or len(schema.get("channels", [])) != 30:
        problems.append("schema_contract")
    if len(read_csv(build / "loso_folds.csv")) != 496:
        problems.append("loso_row_count")

    manifest_by_key = {
        (
            row["subject_id"],
            int(row["outer_fold_id"]),
            row["window_id"],
        ): row
        for row in window_rows
    }
    indexed_rows = 0
    for subject in SUBJECTS:
        for outer_fold in range(OUTER_FOLDS):
            path = build / "split_indices" / f"{subject}_outer{outer_fold}_nbm_indices.npz"
            with np.load(path, allow_pickle=False) as payload:
                for index in range(len(payload["window_id"])):
                    key = (subject, outer_fold, str(payload["window_id"][index]))
                    row = manifest_by_key.get(key)
                    if row is None:
                        problems.append(f"index_missing_manifest:{key}")
                        continue
                    if (
                        int(payload["start_index"][index]) != int(row["start_index"])
                        or int(payload["end_index_exclusive"][index])
                        != int(row["end_index_exclusive"])
                        or int(payload["role_code"][index]) != int(row["role_code"])
                        or int(payload["y_binary"][index]) != int(row["y_binary"])
                    ):
                        problems.append(f"index_manifest_mismatch:{key}")
                    indexed_rows += 1
    if indexed_rows != len(window_rows):
        problems.append(f"indexed_rows={indexed_rows}:manifest_rows={len(window_rows)}")

    # Exercise the project's generic continuous-record loader.  This detects
    # missing run_id fields and channel/schema mismatches before publication.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from cnbr_fog.data import DaphnetDataset

        dataset = DaphnetDataset.load(build)
        if len(dataset.records) != 62 or dataset.n_channels != 30:
            problems.append(
                f"generic_loader_contract:{len(dataset.records)}:{dataset.n_channels}"
            )
    except Exception as error:  # pragma: no cover - reported in output QA
        problems.append(f"generic_loader_error:{type(error).__name__}:{error}")
    return {
        "pass": not problems,
        "record_copy_count": len(manifest_rows),
        "index_manifest_rows_verified": indexed_rows,
        "generic_loader_record_count": 62 if not problems else "see problems",
        "problems": problems[:50],
    }


def write_dataset(
    build: Path,
    source_root: Path,
    records: Sequence[Record],
    manifest_rows: Sequence[dict[str, Any]],
    record_hashes: Mapping[str, str],
    events: Sequence[dict[str, Any]],
    event_manifest: Sequence[dict[str, Any]],
    window_rows: Sequence[dict[str, Any]],
    group_rows: Sequence[dict[str, Any]],
    fold_group_rows: Sequence[dict[str, Any]],
    connector_rows: Sequence[dict[str, Any]],
    alignment_rows: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
    split_summary: Sequence[dict[str, Any]],
    allocation_audit: Sequence[dict[str, Any]],
    groups: Sequence[AllocationGroup],
    connectors: Sequence[Connector],
    quality: dict[str, Any],
) -> dict[str, Any]:
    build.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_root / "records", build / "records")
    ordered_manifest = ordered_manifest_rows(manifest_rows)
    write_csv(build / "manifest.csv", ordered_manifest)
    write_csv(build / "fog_events.csv", events)
    write_csv(build / "loso_folds.csv", build_loso_rows(manifest_rows))
    write_csv(build / "record_integrity_audit.csv", record_integrity_rows(records, record_hashes))
    write_csv(build / "nbm_window_manifest.csv", window_rows)
    write_csv(build / "nbm_group_manifest.csv", group_rows)
    write_csv(build / "nbm_fold_group_roles.csv", fold_group_rows)
    write_csv(build / "nbm_connector_manifest.csv", connector_rows)
    write_csv(build / "nbm_subject_fold_alignment.csv", alignment_rows)
    write_csv(build / "nbm_excluded_window_audit.csv", excluded_rows)
    write_csv(build / "nbm_split_summary.csv", split_summary)
    write_csv(build / "nbm_allocation_optimization_audit.csv", allocation_audit)
    write_csv(build / "nbm_fog_event_manifest.csv", event_manifest)
    write_json(build / "nbm_role_codes.json", ROLE_CODES)

    source_audit = source_root / "record_resampling_audit.csv"
    if source_audit.exists():
        shutil.copy2(source_audit, build / "source_P01_P07_record_resampling_audit.csv")
    source_fir = source_root / "fir_kaiser65_cutoff28hz.csv"
    if source_fir.exists():
        shutil.copy2(source_fir, build / "fir_kaiser65_cutoff28hz_128to64_P01_P07.csv")

    index_audit = save_indices(build, window_rows, ordered_manifest, groups, connectors)
    quality["index_export_audit"] = index_audit
    quality["overall_pass"] = bool(quality["overall_pass"] and index_audit["pass"])
    write_json(build / "schema.json", build_schema(source_root, quality))
    write_json(build / "nbm_protocol.json", build_protocol(source_root, quality))
    write_json(
        build / "preprocessing_report.json",
        {
            "dataset_id": DATASET_ID,
            "subset_id": SUBSET_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "split-only; signals and point labels copied without modification",
            "source_record_directory": str((source_root / "records").resolve()),
            "inventory": quality["confirmed_source_inventory"],
            "P01_P07_preprocessing": "128 Hz to 64 Hz FIR decimation in source dataset",
            "P08_preprocessing": "100 Hz to 64 Hz FIR plus interpolation before this split",
            "quality_pass": quality["overall_pass"],
        },
    )
    pool_report = build_pool_count_report(split_summary, quality)
    (build / "NBM_POOL_COUNT_REPORT.md").write_text(pool_report, encoding="utf-8")
    (build / "README_NBM.md").write_text(
        "# All_dataset processed_NBM_Exp\n\n"
        "This directory contains P01-P08 continuous 64 Hz, 30-channel records and a "
        "strict-purity within-subject three-fold NBM/classifier split.\n\n"
        "- Window: 128 samples (2 s); stride: 64 samples (1 s).\n"
        "- FoG requires 128/128 FoG points; Non-FoG requires 128/128 Non-FoG points.\n"
        "- Mixed and cross-pool boundary windows are explicitly audited and excluded.\n"
        "- Permanent test is fixed across rotations; one development fold validates and "
        "two train.\n"
        "- Use `split_indices/*_nbm_indices.npz` or `nbm_window_manifest.csv`.\n"
        "- Read `NBM_POOL_COUNT_REPORT.md` for counts and `nbm_quality_report.json` "
        "before training.\n",
        encoding="utf-8",
    )
    validation = validate_written_dataset(
        build, source_root, ordered_manifest, record_hashes, window_rows
    )
    quality["persisted_dataset_validation"] = validation
    quality["overall_pass"] = bool(quality["overall_pass"] and validation["pass"])
    # Re-write metadata after persisted validation so the final report is authoritative.
    write_json(build / "nbm_quality_report.json", quality)
    write_json(build / "schema.json", build_schema(source_root, quality))
    write_json(build / "nbm_protocol.json", build_protocol(source_root, quality))
    (build / "NBM_POOL_COUNT_REPORT.md").write_text(
        build_pool_count_report(split_summary, quality), encoding="utf-8"
    )
    if not quality["overall_pass"]:
        raise RuntimeError(json.dumps(quality, ensure_ascii=False, indent=2))
    return validation


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    labeled_root = args.p08_labeled_root.resolve()
    if not (source_root / "records").is_dir():
        raise FileNotFoundError(source_root / "records")
    if not labeled_root.is_dir():
        raise FileNotFoundError(labeled_root)
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    records, manifest_rows, record_hashes = load_records_and_manifest(
        source_root, labeled_root
    )
    events = make_fog_events(records)
    candidates = enumerate_candidates(records)
    groups, connectors, core_window_to_group = build_groups(records, candidates)
    clean_training_roles, allocation_audit = allocate_groups(groups, connectors)
    clean_training_roles, alignment_rows, alignment_quality = align_subject_fold_labels(
        groups, connectors, clean_training_roles, allocation_audit
    )
    (
        window_rows,
        group_rows,
        fold_group_rows,
        connector_rows,
        excluded_rows,
    ) = materialize_manifests(
        records,
        candidates,
        groups,
        connectors,
        core_window_to_group,
        clean_training_roles,
    )
    split_summary = build_split_summary(candidates, window_rows, excluded_rows)
    quality = quality_report(
        records,
        events,
        candidates,
        groups,
        connectors,
        core_window_to_group,
        window_rows,
        excluded_rows,
        split_summary,
        alignment_quality,
    )
    event_manifest = build_fog_event_manifest(events, groups, connectors)
    payload = {
        "output": str(output),
        "dry_run": bool(args.dry_run),
        "quality": quality,
        "aggregate_split_summary": [
            row for row in split_summary if row["subject_id"] == "ALL"
        ],
    }
    if not quality["overall_pass"]:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    build = output.with_name(f"{output.name}.__building_{os.getpid()}")
    if build.exists():
        raise FileExistsError(f"staging directory already exists: {build}")
    try:
        write_dataset(
            build,
            source_root,
            records,
            manifest_rows,
            record_hashes,
            events,
            event_manifest,
            window_rows,
            group_rows,
            fold_group_rows,
            connector_rows,
            alignment_rows,
            excluded_rows,
            split_summary,
            allocation_audit,
            groups,
            connectors,
            quality,
        )
        build.replace(output)
    except Exception:
        if build.exists():
            shutil.rmtree(build)
        raise
    payload["quality"] = quality
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
