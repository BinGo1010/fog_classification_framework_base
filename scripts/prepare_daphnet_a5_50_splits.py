"""Build processed_A5_50 with a strict last-one-second-all-FoG label rule."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_a5_splits as base  # noqa: E402


LAST_FOG_SAMPLES = 64
STRICT_RULE = "last_1s_all_fog"


def strict_validation_event(record_id: str, event_id: int) -> bool:
    # Under the strict rule, three short S09 validation events disappear.
    # Moving seg002 E02 (2.89 s) to validation restores both count and duration
    # ratios while moving the single chronological boundary later.
    if record_id == "S09_seg002" and event_id == 2:
        return True
    return base.validation_event(record_id, event_id)


def parse_args() -> argparse.Namespace:
    dataset_root = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root / "processed")
    parser.add_argument("--output", type=Path, default=dataset_root / "processed_A5_50")
    return parser.parse_args()


def prepare_events(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    output: list[dict[str, Any]] = []
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordinal: Counter[str] = Counter()
    ordered = sorted(rows, key=lambda row: (row["subject_id"], int(row["segment_id"]), int(row["start_index"])))
    for raw in ordered:
        subject = raw["subject_id"]
        event_id = int(raw["event_id"])
        split = "external_validation" if strict_validation_event(raw["record_id"], event_id) else "external_test"
        row: dict[str, Any] = dict(raw)
        row.update(
            {
                "event_id": event_id,
                "subject_event_ordinal": ordinal[subject],
                "start_index": int(raw["start_index"]),
                "end_index": int(raw["end_index"]),
                "end_index_exclusive": int(raw["end_index"]) + 1,
                "duration_sec": float(raw["duration_sec"]),
                "planned_split_before_strict_filter": split,
                "a5_split": split,
                "a5_role": f"{split}_fog",
                "subject_scope": base.subject_scope(subject),
                "strict_rule_status": "pending",
                "strict_window_count_before_embargo": 0,
                "a5_window_count": 0,
            }
        )
        ordinal[subject] += 1
        output.append(row)
        by_record[row["record_id"]].append(row)
    return output, by_record


def strict_windows_before_embargo(
    record: base.Record,
    events: Sequence[dict[str, Any]],
    manifest_row: dict[str, str],
) -> tuple[list[dict[str, Any]], Counter[tuple[str, int]]]:
    rows: list[dict[str, Any]] = []
    coverage: Counter[tuple[str, int]] = Counter()
    for start in range(0, len(record.y) - base.WINDOW + 1, base.STRIDE):
        end = start + base.WINDOW
        if not record.valid[start:end].all():
            continue
        last_start = end - LAST_FOG_SAMPLES
        last_fraction = float(np.mean(record.y[last_start:end] == 1))
        if last_fraction < 1.0:
            continue
        containing = [
            event
            for event in events
            if int(event["start_index"]) <= last_start
            and int(event["end_index_exclusive"]) >= end
        ]
        if len(containing) != 1:
            continue
        event = containing[0]
        role = str(event["a5_role"])
        row = base.window_row(
            record=record,
            manifest_row=manifest_row,
            start=start,
            role=role,
            fog_fraction=last_fraction,
            block_id="",
            event=event,
            alignment="stride64",
        )
        row.update(
            {
                "label_rule": STRICT_RULE,
                "last_1s_fog_fraction": last_fraction,
                "full_2s_fog_fraction": float(np.mean(record.y[start:end] == 1)),
            }
        )
        rows.append(row)
        coverage[(record.record_id, int(event["event_id"]))] += 1
    return rows, coverage


def filter_cross_split_embargo(
    rows: Sequence[dict[str, Any]],
    retained_events_by_record: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        split = str(row["a5_split"])
        other_events = [
            event
            for event in retained_events_by_record[row["record_id"]]
            if event["a5_split"] != split
        ]
        if any(
            base.interval_distance(
                int(row["start_index"]),
                int(row["end_index_exclusive"]),
                int(event["start_index"]),
                int(event["end_index_exclusive"]),
            )
            < base.INTER_SPLIT_EMBARGO
            for event in other_events
        ):
            continue
        output.append(row)
    return output


def add_clean_rows(
    records: Sequence[base.Record],
    manifest: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = base.clean_blocks(records, manifest)
    allocation = base.allocate_clean_blocks(blocks)
    windows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    record_lookup = {record.record_id: record for record in records}
    for block in blocks:
        role = allocation[block.block_id]
        record = record_lookup[block.record_id]
        for start in block.starts:
            row = base.window_row(
                record=record,
                manifest_row=manifest[block.record_id],
                start=start,
                role=role,
                fog_fraction=0.0,
                block_id=block.block_id,
                event=None,
                alignment="stride64",
            )
            row.update(
                {
                    "label_rule": "clean_nonfog_with_6s_fog_guard",
                    "last_1s_fog_fraction": 0.0,
                    "full_2s_fog_fraction": 0.0,
                }
            )
            windows.append(row)
        block_rows.append(
            {
                "clean_block_id": block.block_id,
                "subject_id": block.subject_id,
                "subject_scope": base.subject_scope(block.subject_id),
                "record_id": block.record_id,
                "run_id": block.run_id,
                "segment_id": block.segment_id,
                "a5_role": role,
                "a5_split": base.ROLE_TO_SPLIT[role],
                "start_index": block.start,
                "end_index_exclusive": block.end,
                "start_time_sec": block.start / base.FS,
                "end_time_sec": block.end / base.FS,
                "window_count": block.n_windows,
                "maximum_block_seconds": base.CLEAN_BLOCK_SECONDS,
                "inter_block_embargo_sec": base.INTER_SPLIT_EMBARGO / base.FS,
            }
        )
    return windows, block_rows


def dropped_summary(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subjects = sorted({event["subject_id"] for event in events})
    for subject in subjects:
        selected = [event for event in events if event["subject_id"] == subject]
        kept = [event for event in selected if event["strict_rule_status"] == "retained"]
        dropped = [event for event in selected if event["strict_rule_status"] != "retained"]
        rows.append(
            {
                "subject_id": subject,
                "subject_scope": base.subject_scope(subject),
                "source_event_count": len(selected),
                "retained_event_count": len(kept),
                "dropped_event_count": len(dropped),
                "retained_event_fraction": len(kept) / len(selected) if selected else "",
                "source_fog_duration_sec": sum(float(event["duration_sec"]) for event in selected),
                "retained_fog_duration_sec": sum(float(event["duration_sec"]) for event in kept),
                "dropped_fog_duration_sec": sum(float(event["duration_sec"]) for event in dropped),
                "strict_fog_window_count": sum(int(event["a5_window_count"]) for event in kept),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    build = output.with_name(f"{output.name}.__building_{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)

    manifest_rows = base.read_csv(source / "manifest.csv")
    manifest = {row["record_id"]: row for row in manifest_rows}
    records = base.load_records(source, manifest_rows)
    record_lookup = {record.record_id: record for record in records}
    record_order = {record.record_id: index for index, record in enumerate(records)}
    events, events_by_record = prepare_events(base.read_csv(source / "fog_events.csv"))

    preliminary_fog: list[dict[str, Any]] = []
    pre_coverage: Counter[tuple[str, int]] = Counter()
    for record in records:
        rows, coverage = strict_windows_before_embargo(record, events_by_record.get(record.record_id, []), manifest[record.record_id])
        preliminary_fog.extend(rows)
        pre_coverage.update(coverage)
    for event in events:
        count = pre_coverage[(event["record_id"], int(event["event_id"]))]
        event["strict_window_count_before_embargo"] = count
        if count:
            event["strict_rule_status"] = "retained"
        else:
            event["strict_rule_status"] = "excluded_no_last1s_all_fog_window"
            event["a5_split"] = "excluded"
            event["a5_role"] = "excluded_fog_event"

    retained_events = [event for event in events if event["strict_rule_status"] == "retained"]
    retained_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in retained_events:
        retained_by_record[event["record_id"]].append(event)
    retained_keys = {(event["record_id"], int(event["event_id"])) for event in retained_events}
    preliminary_fog = [
        row
        for row in preliminary_fog
        if (row["record_id"], int(row["event_id"])) in retained_keys
    ]
    fog_rows = filter_cross_split_embargo(preliminary_fog, retained_by_record)
    post_coverage = Counter((row["record_id"], int(row["event_id"])) for row in fog_rows)
    for event in events:
        event["a5_window_count"] = post_coverage[(event["record_id"], int(event["event_id"]))]

    clean_rows, block_rows = add_clean_rows(records, manifest)
    window_rows = clean_rows + fog_rows
    window_rows.sort(
        key=lambda row: (
            row["subject_id"],
            base.ROLE_CODES[row["a5_role"]],
            record_order[row["record_id"]],
            int(row["start_index"]),
        )
    )
    subjects = sorted({record.subject_id for record in records})
    n8_rows = [item for subject in subjects for item in base.select_n8(subject, window_rows, record_lookup)]
    summary_rows, quality = base.summarize(subjects, window_rows, retained_events)
    leakage = base.leakage_audit(window_rows)
    zero_retained = [
        f"{event['record_id']}:E{int(event['event_id']):02d}"
        for event in retained_events
        if int(event["a5_window_count"]) == 0
    ]
    strict_fog = [row for row in window_rows if row["a5_role"].endswith("_fog")]
    quality.update(
        {
            "strict_fog_definition": {
                "window_samples": base.WINDOW,
                "stride_samples": base.STRIDE,
                "last_samples_checked": LAST_FOG_SAMPLES,
                "required_last_1s_fog_fraction": 1.0,
                "event_without_qualifying_window_policy": "exclude_entire_event",
                "fallback_windows_allowed": False,
            },
            "source_event_count": len(events),
            "retained_event_count": len(retained_events),
            "dropped_event_count": len(events) - len(retained_events),
            "retained_events_without_post_embargo_windows": zero_retained,
            "all_strict_fog_windows_grid_aligned": all(int(row["start_index"]) % base.STRIDE == 0 for row in strict_fog),
            "all_windows_length_128": all(int(row["end_index_exclusive"]) - int(row["start_index"]) == base.WINDOW for row in window_rows),
            "all_strict_fog_last1s_fraction_one": all(float(row["last_1s_fog_fraction"]) == 1.0 for row in strict_fog),
            "leakage_audit": leakage,
            "main7_fog_ratio_gate_pass": all(
                0.40 <= quality["subjects"][subject]["validation_fog_event_fraction"] <= 0.50
                and 0.40 <= quality["subjects"][subject]["validation_fog_duration_fraction"] <= 0.50
                for subject in base.MAIN_SUBJECTS
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
        and not zero_retained
        and quality["all_strict_fog_windows_grid_aligned"]
        and quality["all_windows_length_128"]
        and quality["all_strict_fog_last1s_fraction_one"]
        and quality["main7_fog_ratio_gate_pass"]
        and quality["clean_ratio_gate_pass"]
    )
    if not quality["overall_pass"]:
        base.write_json(build / "FAILED_QUALITY_REPORT.json", quality)
        base.write_csv(build / "strict_event_audit.csv", events)
        base.write_csv(build / "strict_event_drop_summary.csv", dropped_summary(events))
        raise RuntimeError(f"Strict A5-50 gate failed; inspect {build}")

    for name in ("manifest.csv", "fog_events.csv", "loso_folds.csv", "preprocessing_report.json"):
        shutil.copy2(source / name, build / name)
    shutil.copytree(source / "records", build / "records")
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    schema["a5_50_split"] = {
        "window_manifest": "a5_50_window_manifest.csv",
        "event_manifest": "a5_50_fog_event_manifest.csv",
        "strict_event_drop_summary": "a5_50_event_drop_summary.csv",
        "window_samples": base.WINDOW,
        "stride_samples": base.STRIDE,
        "fog_definition": "all final 64 samples (1 second) are FOG",
        "minimum_full_window_fog_fraction_implied": 0.5,
        "fallback_windows_allowed": False,
    }
    base.write_json(build / "schema.json", schema)
    base.write_csv(build / "a5_50_window_manifest.csv", window_rows)
    base.write_csv(build / "a5_50_fog_event_manifest.csv", events)
    base.write_csv(build / "a5_50_clean_block_manifest.csv", block_rows)
    base.write_csv(build / "a5_50_split_summary.csv", summary_rows)
    base.write_csv(build / "a5_50_event_drop_summary.csv", dropped_summary(events))
    base.write_csv(build / "a5_50_n8_training_selection.csv", n8_rows)
    base.write_json(build / "a5_50_role_codes.json", base.ROLE_CODES)
    base.save_index_npz(build, subjects, window_rows, record_order)
    protocol = {
        "dataset_id": "daphnet_A5_50",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed": str(source),
        "source_manifest_sha256": base.sha256(source / "manifest.csv"),
        "source_fog_events_sha256": base.sha256(source / "fog_events.csv"),
        "formal_main_subjects": list(base.MAIN_SUBJECTS),
        "diagnostic_subjects": list(base.DIAGNOSTIC_SUBJECTS),
        "clean_only_controls": list(base.CLEAN_ONLY_SUBJECTS),
        "sampling_rate_hz": base.FS,
        "window_samples": base.WINDOW,
        "stride_samples": base.STRIDE,
        "fog_rule": "the final 64 samples are all FOG; this implies at least 50% FOG in the 2 s window",
        "ineligible_event_policy": "exclude the entire FoG event",
        "event_fallback_windows": False,
        "clean_targets": base.CLEAN_TARGETS,
        "clean_fog_guard_seconds_each_side": base.CLEAN_FOG_GUARD / base.FS,
        "inter_split_embargo_seconds": base.INTER_SPLIT_EMBARGO / base.FS,
        "seeds": list(base.SEEDS),
    }
    base.write_json(build / "a5_50_protocol.json", protocol)
    base.write_json(build / "a5_50_quality_report.json", quality)
    (build / "README_A5_50.md").write_text(
        "# Daphnet processed_A5_50\n\n"
        "Strict FoG window: 2 s (128 samples), 1 s stride, and all final 1 s samples are FoG. "
        "FoG events without a qualifying strict-grid window are excluded in full. No fallback windows are used.\n",
        encoding="utf-8",
    )
    build.replace(output)
    print(json.dumps({"output": str(output), "quality": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
