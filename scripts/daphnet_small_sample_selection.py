"""Shared leakage-safe Daphnet small-sample selection primitives."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as current
from cnbr_fog.data import DaphnetDataset, Record


FS = 64


def load_manifest_rows(data_dir: Path) -> dict[str, dict[str, str]]:
    with (Path(data_dir) / "manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def subject_pool(
    dataset: DaphnetDataset, subject: str
) -> tuple[list[Record], current.WindowSet, np.ndarray]:
    records = [record for record in dataset.records if record.subject_id == subject]
    current.SUBJECT = subject
    intervals = current.build_intervals(records)
    windows = current.build_windows(records, intervals)
    candidates = np.flatnonzero((windows.split == "train") & windows.clean_normal)
    if not len(candidates):
        raise ValueError(f"{subject} has no clean Non-FoG training windows")
    return records, windows, candidates.astype(np.int64)


def window_values(
    records: Sequence[Record], windows: current.WindowSet, index: int
) -> np.ndarray:
    record = records[int(windows.record_index[index])]
    return np.asarray(
        record.x[int(windows.start[index]) : int(windows.end[index])],
        dtype=np.float32,
    )


def eligible_candidates(
    records: Sequence[Record], windows: current.WindowSet, candidates: np.ndarray
) -> tuple[np.ndarray, dict[int, float]]:
    eligible: list[int] = []
    energy: dict[int, float] = {}
    for raw_index in candidates:
        index = int(raw_index)
        values = window_values(records, windows, index).astype(np.float64)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        if np.any(np.std(values, axis=0) <= 1e-8) or np.any(q75 - q25 <= 1e-6):
            continue
        eligible.append(index)
        energy[index] = float(np.mean(np.square(values)))
    if not eligible:
        raise ValueError("All candidate windows have a constant or degenerate channel")
    return np.asarray(eligible, dtype=np.int64), energy


def record_id_for(
    records: Sequence[Record], windows: current.WindowSet, index: int
) -> str:
    return records[int(windows.record_index[index])].record_id


def overlaps(
    windows: current.WindowSet,
    records: Sequence[Record],
    candidate: int,
    selected: Iterable[int],
) -> bool:
    candidate_record = record_id_for(records, windows, candidate)
    candidate_start = int(windows.start[candidate])
    candidate_end = int(windows.end[candidate])
    for index in selected:
        if record_id_for(records, windows, int(index)) != candidate_record:
            continue
        start = int(windows.start[int(index)])
        end = int(windows.end[int(index)])
        if max(candidate_start, start) < min(candidate_end, end):
            return True
    return False


def rank_quartiles(indices: Sequence[int], energy: dict[int, float]) -> list[list[int]]:
    ordered = sorted((int(index) for index in indices), key=lambda x: (energy[x], x))
    return [list(map(int, part)) for part in np.array_split(ordered, 4)]


def choose_from_group(
    group: Sequence[int],
    count: int,
    energy: dict[int, float],
    windows: current.WindowSet,
    records: Sequence[Record],
    selected: list[int],
    record_counts: Counter[str] | None = None,
    record_cap: int | None = None,
) -> None:
    if count <= 0:
        return
    target = float(np.median([energy[int(index)] for index in group]))
    remaining = [int(index) for index in group]
    for _ in range(count):
        feasible = [
            index
            for index in remaining
            if not overlaps(windows, records, index, selected)
            and (
                record_cap is None
                or record_counts is None
                or record_counts[record_id_for(records, windows, index)] < record_cap
            )
        ]
        if not feasible and record_cap is not None:
            feasible = [
                index
                for index in remaining
                if not overlaps(windows, records, index, selected)
            ]
        if not feasible:
            raise ValueError(
                f"Unable to choose {count} non-overlapping windows from group of {len(group)}"
            )
        feasible.sort(
            key=lambda index: (
                record_counts[record_id_for(records, windows, index)]
                if record_counts is not None
                else 0,
                abs(energy[index] - target),
                int(windows.start[index]),
                index,
            )
        )
        chosen = feasible[0]
        selected.append(chosen)
        remaining.remove(chosen)
        if record_counts is not None:
            record_counts[record_id_for(records, windows, chosen)] += 1


def allocate_proportional(
    total: int, capacities: dict[str, int], minimum: int
) -> dict[str, int]:
    keys = list(capacities)
    if minimum * len(keys) > total:
        minimum = total // len(keys)
    allocation = {key: minimum for key in keys}
    left = total - sum(allocation.values())
    weights = np.asarray(
        [max(0, capacities[key] - minimum) for key in keys], dtype=float
    )
    if weights.sum() <= 0:
        weights = np.ones(len(keys), dtype=float)
    raw = left * weights / weights.sum()
    floors = np.floor(raw).astype(int)
    for key, value in zip(keys, floors):
        allocation[key] += int(value)
    remainder = total - sum(allocation.values())
    order = sorted(
        range(len(keys)), key=lambda i: (-(raw[i] - floors[i]), keys[i])
    )
    for index in order[:remainder]:
        allocation[keys[index]] += 1
    return allocation


def quota_by_quartile(total: int) -> list[int]:
    values = [total // 4] * 4
    for index in range(total % 4):
        values[index] += 1
    return values


def select_within_record(
    indices: Sequence[int],
    total: int,
    energy: dict[int, float],
    windows: current.WindowSet,
    records: Sequence[Record],
    selected: list[int],
) -> None:
    groups = rank_quartiles(indices, energy)
    for group, count in zip(groups, quota_by_quartile(total)):
        choose_from_group(group, count, energy, windows, records, selected)


def select_windows(
    sample_count: int,
    eligible: np.ndarray,
    energy: dict[int, float],
    records: Sequence[Record],
    windows: current.WindowSet,
) -> np.ndarray:
    if sample_count not in (1, 8, 32):
        raise ValueError("Shared selector supports N=1, N=8, and N=32")
    if len(eligible) < sample_count:
        raise ValueError(f"Need {sample_count} eligible windows, found {len(eligible)}")
    if sample_count == 1:
        target = float(np.median([energy[int(index)] for index in eligible]))
        chosen = min(
            map(int, eligible), key=lambda index: (abs(energy[index] - target), index)
        )
        return np.asarray([chosen], dtype=np.int64)
    if sample_count == 8:
        selected: list[int] = []
        counts: Counter[str] = Counter()
        record_count = len(
            {record_id_for(records, windows, int(index)) for index in eligible}
        )
        cap = 4 if record_count >= 2 else None
        for group in rank_quartiles(eligible, energy):
            choose_from_group(
                group,
                2,
                energy,
                windows,
                records,
                selected,
                record_counts=counts,
                record_cap=cap,
            )
        return np.asarray(selected, dtype=np.int64)

    by_record: dict[str, list[int]] = defaultdict(list)
    for index in eligible:
        by_record[record_id_for(records, windows, int(index))].append(int(index))
    selected = []
    if len(by_record) >= 4:
        chosen_records = sorted(
            by_record, key=lambda key: (-len(by_record[key]), key)
        )[:4]
        for record_id in chosen_records:
            select_within_record(
                by_record[record_id], 8, energy, windows, records, selected
            )
    elif len(by_record) >= 2:
        allocation = allocate_proportional(
            32, {key: len(value) for key, value in by_record.items()}, minimum=8
        )
        for record_id in sorted(by_record):
            select_within_record(
                by_record[record_id],
                allocation[record_id],
                energy,
                windows,
                records,
                selected,
            )
    else:
        only = next(iter(by_record.values()))
        ordered = sorted(only, key=lambda index: (int(windows.start[index]), index))
        for block in np.array_split(ordered, 4):
            select_within_record(
                list(map(int, block)), 8, energy, windows, records, selected
            )
    if len(selected) != 32:
        raise AssertionError(f"Selected {len(selected)} windows, expected 32")
    return np.asarray(selected, dtype=np.int64)


def selected_metadata(
    subject: str,
    sample_count: int,
    selected: np.ndarray,
    energy: dict[int, float],
    records: Sequence[Record],
    windows: current.WindowSet,
    manifest: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    groups = rank_quartiles(selected, energy)
    quartiles = {
        index: group_index + 1
        for group_index, group in enumerate(groups)
        for index in group
    }
    rows: list[dict[str, Any]] = []
    for order, raw_index in enumerate(selected):
        index = int(raw_index)
        record = records[int(windows.record_index[index])]
        start = int(windows.start[index])
        end = int(windows.end[index])
        source_start = int(manifest[record.record_id]["source_start_row"])
        rows.append(
            {
                "subject_id": subject,
                "sample_count": sample_count,
                "selection_order": order,
                "window_id": f"{record.record_id}_{start:06d}_{end:06d}",
                "window_table_index": index,
                "record_id": record.record_id,
                "run_id": record.run_id,
                "start_index": start,
                "end_index_exclusive": end,
                "start_time_sec": start / FS,
                "end_time_sec": end / FS,
                "source_start_row": source_start + start,
                "source_end_row_exclusive": source_start + end,
                "energy": energy[index],
                "energy_quartile": f"Q{quartiles[index]}",
                "clean_normal": True,
                "fog_guard_sec_each_side": 1.0,
            }
        )
    return rows


__all__ = [
    "DaphnetDataset",
    "Record",
    "current",
    "eligible_candidates",
    "load_manifest_rows",
    "overlaps",
    "rank_quartiles",
    "record_id_for",
    "select_windows",
    "selected_metadata",
    "subject_pool",
]
