"""Create the leakage-safe Daphnet A5 data split beside the canonical processed data.

The source continuous NPZ records are preserved.  A5 adds event-level FoG
assignments and block-level clean Non-FoG window assignments.  Splits are made
before model training and are materialized as auditable CSV/JSON/NPZ manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FS = 64
WINDOW = 128
STRIDE = 64
ENDPOINT_LABEL_SAMPLES = 32
# A FoG-positive 2 s window may begin about 1 s before the event because the
# final 0.5 s carries the label.  Six seconds at the event level therefore
# guarantees at least five seconds between the edges of different-split windows.
CLEAN_FOG_GUARD = 6 * FS
INTER_SPLIT_EMBARGO = 5 * FS
CLEAN_BLOCK_SECONDS = 60
SEEDS = (20260802, 20260803, 20260804)

CLEAN_ROLES = (
    "nbm_internal_train_nonfog",
    "nbm_internal_earlystop_nonfog",
    "external_validation_nonfog",
    "external_test_nonfog",
)
CLEAN_TARGETS = {
    "nbm_internal_train_nonfog": 0.48,
    "nbm_internal_earlystop_nonfog": 0.12,
    "external_validation_nonfog": 0.18,
    "external_test_nonfog": 0.22,
}
FOG_ROLES = ("external_validation_fog", "external_test_fog")
ROLE_TO_SPLIT = {
    "nbm_internal_train_nonfog": "nbm_internal_train",
    "nbm_internal_earlystop_nonfog": "nbm_internal_earlystop",
    "external_validation_nonfog": "external_validation",
    "external_test_nonfog": "external_test",
    "external_validation_fog": "external_validation",
    "external_test_fog": "external_test",
}
ROLE_CODES = {role: index for index, role in enumerate(CLEAN_ROLES + FOG_ROLES)}
MAIN_SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
DIAGNOSTIC_SUBJECTS = ("S03",)
CLEAN_ONLY_SUBJECTS = ("S04", "S10")


@dataclass(frozen=True)
class Record:
    record_id: str
    subject_id: str
    run_id: str
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class CleanBlock:
    block_id: str
    subject_id: str
    record_id: str
    run_id: str
    segment_id: int
    starts: tuple[int, ...]

    @property
    def n_windows(self) -> int:
        return len(self.starts)

    @property
    def start(self) -> int:
        return self.starts[0]

    @property
    def end(self) -> int:
        return self.starts[-1] + WINDOW


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base = root / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=base / "processed")
    parser.add_argument("--output", type=Path, default=base / "processed_A5")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Recompute A5 manifests in an existing output while preserving copied records.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def valid_signal_mask(x: np.ndarray, flatline_seconds: float = 1.0) -> np.ndarray:
    values = np.asarray(x)
    valid = np.isfinite(values).all(axis=1)
    minimum = max(1, int(round(flatline_seconds * FS)))
    groups = (
        [values[:, start : start + 3] for start in range(0, values.shape[1], 3)]
        if values.shape[1] % 3 == 0
        else [values]
    )
    for group in groups:
        zero = np.max(np.abs(group), axis=1) <= 1e-8
        for start, end in true_runs(zero):
            if end - start >= minimum:
                valid[start:end] = False
    return valid


def load_records(source: Path, manifest_rows: Sequence[dict[str, str]]) -> list[Record]:
    records: list[Record] = []
    for row in manifest_rows:
        if str(row.get("usable", "true")).strip().lower() not in {"1", "true", "yes"}:
            continue
        path = source / row["record_path"]
        with np.load(path, allow_pickle=False) as payload:
            x = np.asarray(payload["x"], dtype=np.float32)
            y = np.asarray(payload["y_binary"], dtype=np.int8)
        if x.ndim != 2 or y.shape != (len(x),):
            raise ValueError(f"Invalid record arrays in {path}: x={x.shape}, y={y.shape}")
        if len(x) != int(row["n_samples"]):
            raise ValueError(f"Manifest length mismatch in {path}")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f"Non-binary labels in {path}")
        records.append(
            Record(
                record_id=row["record_id"],
                subject_id=row["subject_id"],
                run_id=row["run_id"],
                x=x,
                y=y,
                valid=valid_signal_mask(x),
            )
        )
    if not records:
        raise ValueError(f"No records loaded from {source}")
    return records


def subject_scope(subject: str) -> str:
    if subject in MAIN_SUBJECTS:
        return "formal_main7"
    if subject in DIAGNOSTIC_SUBJECTS:
        return "diagnostic_only"
    if subject in CLEAN_ONLY_SUBJECTS:
        return "clean_only_control"
    raise ValueError(subject)


def validation_event(record_id: str, event_id: int) -> bool:
    """Frozen event allocation agreed from the supplied segment summary."""
    subject = record_id[:3]
    if subject == "S01":
        return record_id == "S01_seg001" and event_id <= 10
    if subject == "S02":
        return record_id == "S02_seg000" or (record_id == "S02_seg001" and event_id <= 2)
    if subject == "S03":
        # E01 and E02 in seg001 are only 1.5 s apart and must remain in one split.
        # Including E02 keeps both event-count and duration ratios inside 40-50%.
        return record_id == "S03_seg000" or (record_id == "S03_seg001" and event_id <= 2)
    if subject == "S05":
        return record_id in {"S05_seg000", "S05_seg001", "S05_seg002"} or (
            record_id == "S05_seg003" and event_id == 0
        )
    if subject == "S06":
        return record_id == "S06_seg000" and event_id <= 3
    if subject == "S07":
        return record_id == "S07_seg000" and event_id <= 10
    if subject == "S08":
        return record_id in {"S08_seg001", "S08_seg002"}
    if subject == "S09":
        return record_id in {"S09_seg000", "S09_seg001"} or (
            record_id == "S09_seg002" and event_id <= 1
        )
    if subject in CLEAN_ONLY_SUBJECTS:
        raise ValueError(f"{subject} unexpectedly has a FoG event")
    raise ValueError(f"No event plan for {record_id}")


def prepare_events(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    result: list[dict[str, Any]] = []
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordinal: Counter[str] = Counter()
    ordered = sorted(
        rows,
        key=lambda row: (row["subject_id"], int(row["segment_id"]), int(row["start_index"])),
    )
    for raw in ordered:
        subject = raw["subject_id"]
        event_id = int(raw["event_id"])
        split = "external_validation" if validation_event(raw["record_id"], event_id) else "external_test"
        row: dict[str, Any] = dict(raw)
        row.update(
            {
                "event_id": event_id,
                "subject_event_ordinal": ordinal[subject],
                "start_index": int(raw["start_index"]),
                "end_index": int(raw["end_index"]),
                "end_index_exclusive": int(raw["end_index"]) + 1,
                "duration_sec": float(raw["duration_sec"]),
                "a5_split": split,
                "a5_role": f"{split}_fog",
                "subject_scope": subject_scope(subject),
            }
        )
        ordinal[subject] += 1
        result.append(row)
        by_record[row["record_id"]].append(row)
    return result, by_record


def candidate_clean_starts(record: Record) -> list[int]:
    starts: list[int] = []
    for start in range(0, len(record.y) - WINDOW + 1, STRIDE):
        end = start + WINDOW
        guard_start = start - CLEAN_FOG_GUARD
        guard_end = end + CLEAN_FOG_GUARD
        if guard_start < 0 or guard_end > len(record.y):
            continue
        if not record.valid[start:end].all():
            continue
        if record.y[guard_start:guard_end].any():
            continue
        starts.append(start)
    return starts


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


def clean_blocks(records: Sequence[Record], manifest: dict[str, dict[str, str]]) -> list[CleanBlock]:
    blocks: list[CleanBlock] = []
    for record in records:
        block_number = 0
        for run in consecutive_runs(candidate_clean_starts(record)):
            position = 0
            while position < len(run):
                first = run[position]
                block_boundary = first + CLEAN_BLOCK_SECONDS * FS
                stop = position
                while stop < len(run) and run[stop] + WINDOW <= block_boundary:
                    stop += 1
                chosen = tuple(run[position:stop])
                if not chosen:
                    chosen = (run[position],)
                    stop = position + 1
                blocks.append(
                    CleanBlock(
                        block_id=f"{record.record_id}_cleanblock{block_number:03d}",
                        subject_id=record.subject_id,
                        record_id=record.record_id,
                        run_id=record.run_id,
                        segment_id=int(manifest[record.record_id]["segment_id"]),
                        starts=chosen,
                    )
                )
                block_number += 1
                next_start = chosen[-1] + WINDOW + INTER_SPLIT_EMBARGO
                position = stop
                while position < len(run) and run[position] < next_start:
                    position += 1
    return blocks


def allocate_clean_blocks(blocks: Sequence[CleanBlock]) -> dict[str, str]:
    by_subject: dict[str, list[CleanBlock]] = defaultdict(list)
    for block in blocks:
        by_subject[block.subject_id].append(block)
    allocation: dict[str, str] = {}
    for subject, subject_blocks in sorted(by_subject.items()):
        total = sum(block.n_windows for block in subject_blocks)
        targets = {role: CLEAN_TARGETS[role] * total for role in CLEAN_ROLES}
        counts = {role: 0 for role in CLEAN_ROLES}
        ordered = sorted(
            subject_blocks,
            key=lambda block: (-block.n_windows, block.record_id, block.start, block.block_id),
        )
        for block in ordered:
            def score(role: str) -> tuple[float, float, int]:
                normalized_deficit = (targets[role] - counts[role]) / max(targets[role], 1.0)
                absolute_deficit = targets[role] - counts[role]
                return normalized_deficit, absolute_deficit, -CLEAN_ROLES.index(role)

            role = max(CLEAN_ROLES, key=score)
            allocation[block.block_id] = role
            counts[role] += block.n_windows
        if any(counts[role] == 0 for role in CLEAN_ROLES):
            raise RuntimeError(f"{subject} has an empty clean role after block allocation: {counts}")
    return allocation


def interval_distance(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a <= start_b:
        return start_b - end_a
    if end_b <= start_a:
        return start_a - end_b
    return 0


def fog_windows(
    record: Record,
    events: Sequence[dict[str, Any]],
    manifest_row: dict[str, str],
) -> tuple[list[dict[str, Any]], Counter[tuple[str, int]]]:
    rows: list[dict[str, Any]] = []
    coverage: Counter[tuple[str, int]] = Counter()
    for start in range(0, len(record.y) - WINDOW + 1, STRIDE):
        end = start + WINDOW
        if not record.valid[start:end].all():
            continue
        endpoint_start = end - ENDPOINT_LABEL_SAMPLES
        fraction = float(np.mean(record.y[endpoint_start:end] == 1))
        if fraction < 0.5:
            continue
        overlaps = [
            max(0, min(end, int(event["end_index_exclusive"])) - max(endpoint_start, int(event["start_index"])))
            for event in events
        ]
        if not overlaps or max(overlaps) < ENDPOINT_LABEL_SAMPLES / 2:
            continue
        event = events[int(np.argmax(overlaps))]
        split = str(event["a5_split"])
        other_events = [candidate for candidate in events if candidate["a5_split"] != split]
        if any(
            interval_distance(
                start,
                end,
                int(candidate["start_index"]),
                int(candidate["end_index_exclusive"]),
            )
            < INTER_SPLIT_EMBARGO
            for candidate in other_events
        ):
            continue
        role = f"{split}_fog"
        row = window_row(
            record=record,
            manifest_row=manifest_row,
            start=start,
            role=role,
            fog_fraction=fraction,
            block_id="",
            event=event,
        )
        rows.append(row)
        coverage[(record.record_id, int(event["event_id"]))] += 1
    return rows, coverage


def fallback_fog_window(
    record: Record,
    event: dict[str, Any],
    events: Sequence[dict[str, Any]],
    manifest_row: dict[str, str],
    occupied: set[str],
) -> dict[str, Any] | None:
    """Cover a short event that the 1 s grid misses, while preserving the endpoint rule."""
    split = str(event["a5_split"])
    event_start = int(event["start_index"])
    event_end = int(event["end_index_exclusive"])
    lower = max(0, event_start - WINDOW)
    upper = min(len(record.y) - WINDOW, event_end)
    other_events = [candidate for candidate in events if candidate["a5_split"] != split]
    candidates: list[tuple[tuple[float, ...], int, float]] = []
    for start in range(lower, upper + 1):
        end = start + WINDOW
        window_id = f"{record.record_id}:{start}:{end}"
        if window_id in occupied or not record.valid[start:end].all():
            continue
        endpoint_start = end - ENDPOINT_LABEL_SAMPLES
        overlap = max(0, min(end, event_end) - max(endpoint_start, event_start))
        if overlap < ENDPOINT_LABEL_SAMPLES / 2:
            continue
        fraction = float(np.mean(record.y[endpoint_start:end] == 1))
        if fraction < 0.5:
            continue
        if any(
            interval_distance(
                start,
                end,
                int(candidate["start_index"]),
                int(candidate["end_index_exclusive"]),
            )
            < INTER_SPLIT_EMBARGO
            for candidate in other_events
        ):
            continue
        grid_distance = min(start % STRIDE, STRIDE - (start % STRIDE))
        center_distance = abs((endpoint_start + end) / 2.0 - (event_start + event_end) / 2.0)
        candidates.append(((-float(overlap), float(grid_distance), center_distance, float(start)), start, fraction))
    if not candidates:
        return None
    _, start, fraction = min(candidates, key=lambda item: item[0])
    return window_row(
        record=record,
        manifest_row=manifest_row,
        start=start,
        role=f"{split}_fog",
        fog_fraction=fraction,
        block_id="",
        event=event,
        alignment="event_fallback",
    )


def window_row(
    *,
    record: Record,
    manifest_row: dict[str, str],
    start: int,
    role: str,
    fog_fraction: float,
    block_id: str,
    event: dict[str, Any] | None,
    alignment: str = "stride64",
) -> dict[str, Any]:
    end = start + WINDOW
    source_start = int(manifest_row["source_start_row"])
    is_fog = role.endswith("_fog")
    return {
        "window_id": f"{record.record_id}:{start}:{end}",
        "subject_id": record.subject_id,
        "subject_scope": subject_scope(record.subject_id),
        "record_id": record.record_id,
        "run_id": record.run_id,
        "segment_id": int(manifest_row["segment_id"]),
        "source_file": manifest_row["source_file"],
        "a5_role": role,
        "a5_split": ROLE_TO_SPLIT[role],
        "class_label": "FOG" if is_fog else "CLEAN_NONFOG",
        "y_binary": int(is_fog),
        "start_index": start,
        "end_index_exclusive": end,
        "start_time_sec": start / FS,
        "end_time_sec": end / FS,
        "source_start_row": source_start + start,
        "source_end_row_exclusive": source_start + end,
        "endpoint_fog_fraction": fog_fraction,
        "clean_nonfog": not is_fog,
        "clean_fog_guard_sec_each_side": CLEAN_FOG_GUARD / FS if not is_fog else "",
        "inter_split_embargo_sec": INTER_SPLIT_EMBARGO / FS,
        "clean_block_id": block_id,
        "event_id": "" if event is None else int(event["event_id"]),
        "subject_event_ordinal": "" if event is None else int(event["subject_event_ordinal"]),
        "window_alignment": alignment,
    }


def select_n8(
    subject: str,
    rows: Sequence[dict[str, Any]],
    records: dict[str, Record],
) -> list[dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        if row["subject_id"] != subject or row["a5_role"] != "nbm_internal_train_nonfog":
            continue
        values = records[row["record_id"]].x[int(row["start_index"]) : int(row["end_index_exclusive"])]
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        if np.any(np.std(values, axis=0) <= 1e-8) or np.any(q75 - q25 <= 1e-6):
            continue
        candidates.append((row, float(np.mean(np.square(values.astype(np.float64))))))
    if len(candidates) < 8:
        raise RuntimeError(f"{subject} has only {len(candidates)} eligible N=8 candidates")
    candidates.sort(key=lambda item: (item[1], item[0]["record_id"], item[0]["start_index"]))
    groups = np.array_split(np.arange(len(candidates)), 4)
    selected: list[tuple[dict[str, Any], float, int]] = []
    record_counts: Counter[str] = Counter()
    record_count = len({item[0]["record_id"] for item in candidates})
    cap = 4 if record_count >= 2 else None
    for quartile, group_indices in enumerate(groups, start=1):
        group = [candidates[int(index)] for index in group_indices]
        target = float(np.median([energy for _, energy in group]))
        for _ in range(2):
            feasible = []
            for row, energy in group:
                overlaps_selected = any(
                    row["record_id"] == existing["record_id"]
                    and max(int(row["start_index"]), int(existing["start_index"]))
                    < min(int(row["end_index_exclusive"]), int(existing["end_index_exclusive"]))
                    for existing, _, _ in selected
                )
                if overlaps_selected:
                    continue
                if cap is not None and record_counts[row["record_id"]] >= cap:
                    continue
                feasible.append((row, energy))
            if not feasible and cap is not None:
                for row, energy in group:
                    if not any(
                        row["record_id"] == existing["record_id"]
                        and max(int(row["start_index"]), int(existing["start_index"]))
                        < min(int(row["end_index_exclusive"]), int(existing["end_index_exclusive"]))
                        for existing, _, _ in selected
                    ):
                        feasible.append((row, energy))
            if not feasible:
                raise RuntimeError(f"Unable to select non-overlapping N=8 windows for {subject} Q{quartile}")
            row, energy = min(
                feasible,
                key=lambda item: (
                    record_counts[item[0]["record_id"]],
                    abs(item[1] - target),
                    item[0]["record_id"],
                    item[0]["start_index"],
                ),
            )
            selected.append((row, energy, quartile))
            record_counts[row["record_id"]] += 1
            group.remove((row, energy))
    output: list[dict[str, Any]] = []
    for order, (row, energy, quartile) in enumerate(selected):
        output.append(
            {
                "subject_id": subject,
                "subject_scope": subject_scope(subject),
                "selection_order": order,
                "window_id": row["window_id"],
                "record_id": row["record_id"],
                "run_id": row["run_id"],
                "start_index": row["start_index"],
                "end_index_exclusive": row["end_index_exclusive"],
                "energy": energy,
                "energy_quartile": f"Q{quartile}",
                "applies_to_seeds": ";".join(map(str, SEEDS)),
                "seed_invariant_selection": True,
            }
        )
    return output


def summarize(
    subjects: Sequence[str],
    windows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        subject_windows = [row for row in windows if row["subject_id"] == subject]
        subject_events = [row for row in events if row["subject_id"] == subject]
        clean_total = sum(row["class_label"] == "CLEAN_NONFOG" for row in subject_windows)
        event_total = len(subject_events)
        duration_total = sum(float(row["duration_sec"]) for row in subject_events)
        for role in CLEAN_ROLES + FOG_ROLES:
            chosen_windows = [row for row in subject_windows if row["a5_role"] == role]
            split = ROLE_TO_SPLIT[role]
            chosen_events = [row for row in subject_events if row["a5_split"] == split] if role.endswith("_fog") else []
            event_count = len(chosen_events)
            event_duration = sum(float(row["duration_sec"]) for row in chosen_events)
            rows.append(
                {
                    "subject_id": subject,
                    "subject_scope": subject_scope(subject),
                    "a5_role": role,
                    "a5_split": split,
                    "class_label": "FOG" if role.endswith("_fog") else "CLEAN_NONFOG",
                    "window_count": len(chosen_windows),
                    "clean_window_fraction": len(chosen_windows) / clean_total if clean_total and not role.endswith("_fog") else "",
                    "fog_event_count": event_count,
                    "fog_event_fraction": event_count / event_total if event_total and role.endswith("_fog") else "",
                    "fog_duration_sec": event_duration if role.endswith("_fog") else "",
                    "fog_duration_fraction": event_duration / duration_total if duration_total and role.endswith("_fog") else "",
                }
            )

    def fog_ratio(subject: str, split: str, key: str) -> float:
        selected = [row for row in events if row["subject_id"] == subject and row["a5_split"] == split]
        all_rows = [row for row in events if row["subject_id"] == subject]
        if key == "count":
            return len(selected) / len(all_rows)
        return sum(float(row["duration_sec"]) for row in selected) / sum(
            float(row["duration_sec"]) for row in all_rows
        )

    main_events = [row for row in events if row["subject_id"] in MAIN_SUBJECTS]
    main_validation = [row for row in main_events if row["a5_split"] == "external_validation"]
    quality = {
        "subjects": {},
        "main7_aggregate": {
            "fog_events_total": len(main_events),
            "validation_fog_events": len(main_validation),
            "validation_fog_event_fraction": len(main_validation) / len(main_events),
            "test_fog_events": len(main_events) - len(main_validation),
            "test_fog_event_fraction": 1.0 - len(main_validation) / len(main_events),
            "fog_duration_total_sec": sum(float(row["duration_sec"]) for row in main_events),
            "validation_fog_duration_sec": sum(float(row["duration_sec"]) for row in main_validation),
        },
    }
    total_duration = quality["main7_aggregate"]["fog_duration_total_sec"]
    validation_duration = quality["main7_aggregate"]["validation_fog_duration_sec"]
    quality["main7_aggregate"].update(
        {
            "validation_fog_duration_fraction": validation_duration / total_duration,
            "test_fog_duration_sec": total_duration - validation_duration,
            "test_fog_duration_fraction": 1.0 - validation_duration / total_duration,
        }
    )
    for subject in subjects:
        subject_rows = [row for row in rows if row["subject_id"] == subject]
        clean_rows = [row for row in subject_rows if row["class_label"] == "CLEAN_NONFOG"]
        clean_fractions = {row["a5_role"]: float(row["clean_window_fraction"]) for row in clean_rows}
        item: dict[str, Any] = {
            "scope": subject_scope(subject),
            "clean_window_fractions": clean_fractions,
            "clean_fraction_max_abs_deviation": max(
                abs(clean_fractions[role] - CLEAN_TARGETS[role]) for role in CLEAN_ROLES
            ),
        }
        if subject not in CLEAN_ONLY_SUBJECTS:
            item.update(
                {
                    "validation_fog_event_fraction": fog_ratio(subject, "external_validation", "count"),
                    "test_fog_event_fraction": fog_ratio(subject, "external_test", "count"),
                    "validation_fog_duration_fraction": fog_ratio(subject, "external_validation", "duration"),
                    "test_fog_duration_fraction": fog_ratio(subject, "external_test", "duration"),
                }
            )
        quality["subjects"][subject] = item
    return rows, quality


def leakage_audit(windows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    duplicate_ids = len(windows) - len({str(row["window_id"]) for row in windows})
    internal_fog = sum(
        row["y_binary"] == 1 and row["a5_split"].startswith("nbm_internal") for row in windows
    )
    close_cross_split_pairs = 0
    overlap_cross_split_pairs = 0
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        by_record[row["record_id"]].append(row)
    examples: list[dict[str, Any]] = []
    for record_id, record_rows in by_record.items():
        ordered = sorted(record_rows, key=lambda row: (int(row["start_index"]), int(row["end_index_exclusive"])))
        for i, left in enumerate(ordered):
            left_end = int(left["end_index_exclusive"])
            for right in ordered[i + 1 :]:
                right_start = int(right["start_index"])
                if right_start >= left_end + INTER_SPLIT_EMBARGO:
                    break
                if left["a5_split"] == right["a5_split"]:
                    continue
                distance = interval_distance(
                    int(left["start_index"]),
                    left_end,
                    right_start,
                    int(right["end_index_exclusive"]),
                )
                if distance == 0:
                    overlap_cross_split_pairs += 1
                if distance < INTER_SPLIT_EMBARGO:
                    close_cross_split_pairs += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "record_id": record_id,
                                "left": left["window_id"],
                                "left_split": left["a5_split"],
                                "right": right["window_id"],
                                "right_split": right["a5_split"],
                                "distance_samples": distance,
                            }
                        )
    return {
        "duplicate_window_ids": duplicate_ids,
        "internal_fog_windows": internal_fog,
        "cross_split_overlapping_window_pairs": overlap_cross_split_pairs,
        "cross_split_pairs_inside_5s_embargo": close_cross_split_pairs,
        "examples": examples,
        "pass": duplicate_ids == 0 and internal_fog == 0 and overlap_cross_split_pairs == 0,
    }


def save_index_npz(root: Path, subjects: Sequence[str], rows: Sequence[dict[str, Any]], record_order: dict[str, int]) -> None:
    split_dir = root / "split_indices"
    split_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        chosen = sorted(
            (row for row in rows if row["subject_id"] == subject),
            key=lambda row: (ROLE_CODES[row["a5_role"]], record_order[row["record_id"]], row["start_index"]),
        )
        np.savez_compressed(
            split_dir / f"{subject}_a5_window_indices.npz",
            record_index=np.asarray([record_order[row["record_id"]] for row in chosen], dtype=np.int16),
            start_index=np.asarray([row["start_index"] for row in chosen], dtype=np.int32),
            end_index_exclusive=np.asarray([row["end_index_exclusive"] for row in chosen], dtype=np.int32),
            role_code=np.asarray([ROLE_CODES[row["a5_role"]] for row in chosen], dtype=np.int8),
            y_binary=np.asarray([row["y_binary"] for row in chosen], dtype=np.int8),
            event_id=np.asarray([row["event_id"] if row["event_id"] != "" else -1 for row in chosen], dtype=np.int16),
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and not args.update_existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not source.exists():
        raise FileNotFoundError(source)
    build = output if args.update_existing else output.with_name(f"{output.name}.__building_{os.getpid()}")
    build.mkdir(parents=True, exist_ok=args.update_existing)

    manifest_rows = read_csv(source / "manifest.csv")
    manifest = {row["record_id"]: row for row in manifest_rows}
    event_rows, events_by_record = prepare_events(read_csv(source / "fog_events.csv"))
    ordered_records = load_records(source, manifest_rows)
    records = {record.record_id: record for record in ordered_records}
    record_order = {record.record_id: index for index, record in enumerate(ordered_records)}

    blocks = clean_blocks(ordered_records, manifest)
    allocation = allocate_clean_blocks(blocks)
    window_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for block in blocks:
        role = allocation[block.block_id]
        record = records[block.record_id]
        manifest_row = manifest[block.record_id]
        for start in block.starts:
            window_rows.append(
                window_row(
                    record=record,
                    manifest_row=manifest_row,
                    start=start,
                    role=role,
                    fog_fraction=0.0,
                    block_id=block.block_id,
                    event=None,
                )
            )
        block_rows.append(
            {
                "clean_block_id": block.block_id,
                "subject_id": block.subject_id,
                "subject_scope": subject_scope(block.subject_id),
                "record_id": block.record_id,
                "run_id": block.run_id,
                "segment_id": block.segment_id,
                "a5_role": role,
                "a5_split": ROLE_TO_SPLIT[role],
                "start_index": block.start,
                "end_index_exclusive": block.end,
                "start_time_sec": block.start / FS,
                "end_time_sec": block.end / FS,
                "window_count": block.n_windows,
                "maximum_block_seconds": CLEAN_BLOCK_SECONDS,
                "inter_block_embargo_sec": INTER_SPLIT_EMBARGO / FS,
            }
        )

    event_window_coverage: Counter[tuple[str, int]] = Counter()
    for record in ordered_records:
        rows, coverage = fog_windows(record, events_by_record.get(record.record_id, []), manifest[record.record_id])
        window_rows.extend(rows)
        event_window_coverage.update(coverage)
    occupied = {str(row["window_id"]) for row in window_rows}
    for event in event_rows:
        key = (event["record_id"], int(event["event_id"]))
        if event_window_coverage[key] > 0:
            continue
        record = records[event["record_id"]]
        fallback = fallback_fog_window(
            record,
            event,
            events_by_record[event["record_id"]],
            manifest[event["record_id"]],
            occupied,
        )
        if fallback is not None:
            window_rows.append(fallback)
            occupied.add(str(fallback["window_id"]))
            event_window_coverage[key] += 1
    for event in event_rows:
        event["a5_window_count"] = event_window_coverage[(event["record_id"], int(event["event_id"]))]

    window_rows.sort(
        key=lambda row: (row["subject_id"], ROLE_CODES[row["a5_role"]], record_order[row["record_id"]], row["start_index"])
    )
    subjects = sorted({record.subject_id for record in ordered_records})
    n8_rows = [item for subject in subjects for item in select_n8(subject, window_rows, records)]
    summary_rows, quality = summarize(subjects, window_rows, event_rows)
    leakage = leakage_audit(window_rows)
    zero_window_events = [
        f"{row['record_id']}:E{int(row['event_id']):02d}" for row in event_rows if int(row["a5_window_count"]) == 0
    ]
    quality.update(
        {
            "leakage_audit": leakage,
            "fog_events_without_a5_windows": zero_window_events,
            "all_fog_events_have_windows": not zero_window_events,
            "main7_fog_ratio_gate_pass": all(
                0.40 <= quality["subjects"][subject]["validation_fog_event_fraction"] <= 0.50
                and 0.40 <= quality["subjects"][subject]["validation_fog_duration_fraction"] <= 0.50
                for subject in MAIN_SUBJECTS
            ),
            "clean_ratio_tolerance": 0.06,
            "clean_ratio_gate_pass": all(
                quality["subjects"][subject]["clean_fraction_max_abs_deviation"] <= 0.06
                for subject in subjects
            ),
        }
    )
    quality["overall_pass"] = bool(
        quality["leakage_audit"]["pass"]
        and quality["all_fog_events_have_windows"]
        and quality["main7_fog_ratio_gate_pass"]
        and quality["clean_ratio_gate_pass"]
    )
    if not quality["overall_pass"]:
        write_json(build / "FAILED_QUALITY_REPORT.json", quality)
        raise RuntimeError(f"A5 split quality gate failed; inspect {build / 'FAILED_QUALITY_REPORT.json'}")

    for name in ("manifest.csv", "fog_events.csv", "loso_folds.csv", "preprocessing_report.json"):
        shutil.copy2(source / name, build / name)
    if not (build / "records").exists():
        shutil.copytree(source / "records", build / "records")
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    schema["a5_split"] = {
        "window_manifest": "a5_window_manifest.csv",
        "event_manifest": "a5_fog_event_manifest.csv",
        "clean_block_manifest": "a5_clean_block_manifest.csv",
        "summary": "a5_split_summary.csv",
        "role_codes": ROLE_CODES,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "endpoint_label_samples": ENDPOINT_LABEL_SAMPLES,
        "clean_fog_guard_seconds_each_side": CLEAN_FOG_GUARD / FS,
        "inter_split_embargo_seconds": INTER_SPLIT_EMBARGO / FS,
    }
    write_json(build / "schema.json", schema)
    write_csv(build / "a5_window_manifest.csv", window_rows)
    write_csv(build / "a5_fog_event_manifest.csv", event_rows)
    write_csv(build / "a5_clean_block_manifest.csv", block_rows)
    write_csv(build / "a5_split_summary.csv", summary_rows)
    write_csv(build / "a5_n8_training_selection.csv", n8_rows)
    write_json(build / "a5_role_codes.json", ROLE_CODES)
    save_index_npz(build, subjects, window_rows, record_order)

    protocol = {
        "dataset_id": "daphnet_A5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed": str(source),
        "source_manifest_sha256": sha256(source / "manifest.csv"),
        "source_fog_events_sha256": sha256(source / "fog_events.csv"),
        "subjects": subjects,
        "formal_main_subjects": list(MAIN_SUBJECTS),
        "diagnostic_subjects": list(DIAGNOSTIC_SUBJECTS),
        "clean_only_controls": list(CLEAN_ONLY_SUBJECTS),
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "window_label": "FOG when at least half of the final 32 samples are FOG",
        "clean_definition": "valid window with no FOG in the window or five seconds on either side",
        "clean_block_max_seconds": CLEAN_BLOCK_SECONDS,
        "inter_split_embargo_seconds": INTER_SPLIT_EMBARGO / FS,
        "clean_targets": CLEAN_TARGETS,
        "fog_targets": {"external_validation": [0.40, 0.50], "external_test": [0.50, 0.60]},
        "n8_selection": {
            "source_role": "nbm_internal_train_nonfog",
            "windows_per_subject": 8,
            "energy_quartile_quota": 2,
            "non_overlapping": True,
            "seed_invariant": True,
            "seeds": list(SEEDS),
        },
    }
    write_json(build / "a5_protocol.json", protocol)
    write_json(build / "a5_quality_report.json", quality)
    readme = """# Daphnet processed_A5\n\nThis directory preserves the canonical continuous records and adds the frozen A5 split.\n\n- `a5_window_manifest.csv`: authoritative window-level roles.\n- `a5_fog_event_manifest.csv`: event-level validation/test assignment.\n- `a5_clean_block_manifest.csv`: clean block allocation before windowing.\n- `a5_n8_training_selection.csv`: frozen non-overlapping N=8 subset from internal training.\n- `a5_split_summary.csv`: per-subject counts and ratios.\n- `a5_quality_report.json`: leakage and ratio gates.\n- `split_indices/*.npz`: compact indices referencing the copied continuous records.\n\nNBM training and early stopping contain only clean Non-FoG.  Residual/threshold selection uses external validation; external test remains frozen.\n"""
    (build / "README_A5.md").write_text(readme, encoding="utf-8")
    if build != output:
        build.replace(output)
    print(json.dumps({"output": str(output), "quality": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
