"""Derive processed_CA_pure by removing every mixed 2 s CA window.

Only windows with 0/128 FOG samples (PURE_NONFOG) or 128/128 FOG samples
(PURE_FOG) are retained.  The frozen within-subject CA group and split
assignments are preserved to support paired ablation experiments.
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
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_daphnet_a5_splits as base  # noqa: E402
import prepare_daphnet_ca_splits as ca  # noqa: E402


WINDOW = 128
SPLITS = ca.SPLITS
SPLIT_CODES = ca.SPLIT_CODES


def parse_args() -> argparse.Namespace:
    dataset_root = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root / "processed_CA")
    parser.add_argument("--output", type=Path, default=dataset_root / "processed_CA_pure")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def classify_filter(row: dict[str, str]) -> str:
    fog_samples = int(row["fog_samples_in_2s"])
    if fog_samples == 0:
        return "PURE_NONFOG"
    if fog_samples == WINDOW:
        return "PURE_FOG"
    if fog_samples < WINDOW // 2:
        return "MIXED_1_TO_63"
    return "MIXED_64_TO_127"


def pure_rows(source_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in source_rows:
        category = classify_filter(source)
        row: dict[str, Any] = dict(source)
        row["purity_label"] = category
        if category.startswith("MIXED_"):
            row["pure_exclusion_reason"] = "mixed_boundary_window"
            excluded.append(row)
            continue
        is_fog = category == "PURE_FOG"
        row.update(
            {
                "class_label": "FOG" if is_fog else "NONFOG",
                "y_binary": int(is_fog),
                "nonfog_subtype": "" if is_fog else "PURE_NONFOG",
                "pure_window": True,
                "label_rule": (
                    "PURE_FOG iff fog_samples_in_2s == 128; "
                    "PURE_NONFOG iff fog_samples_in_2s == 0; mixed windows excluded"
                ),
            }
        )
        retained.append(row)
    return retained, excluded


def filter_summary(
    source_rows: Sequence[dict[str, str]],
    retained: Sequence[dict[str, Any]],
    excluded: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    subjects = sorted({row["subject_id"] for row in source_rows})
    output: list[dict[str, Any]] = []
    for subject in subjects + ["ALL"]:
        source_scope = list(source_rows) if subject == "ALL" else [row for row in source_rows if row["subject_id"] == subject]
        pure_scope = list(retained) if subject == "ALL" else [row for row in retained if row["subject_id"] == subject]
        excluded_scope = list(excluded) if subject == "ALL" else [row for row in excluded if row["subject_id"] == subject]
        for split in SPLITS:
            source_split = [row for row in source_scope if row["ca_split"] == split]
            pure_split = [row for row in pure_scope if row["ca_split"] == split]
            excluded_split = [row for row in excluded_scope if row["ca_split"] == split]
            output.append(
                {
                    "subject_id": subject,
                    "ca_split": split,
                    "source_ca_window_count": len(source_split),
                    "retained_pure_window_count": len(pure_split),
                    "retained_fraction": len(pure_split) / len(source_split) if source_split else "",
                    "pure_fog_window_count": sum(row["purity_label"] == "PURE_FOG" for row in pure_split),
                    "pure_nonfog_window_count": sum(row["purity_label"] == "PURE_NONFOG" for row in pure_split),
                    "removed_mixed_window_count": len(excluded_split),
                    "removed_mixed_1_to_63_count": sum(row["purity_label"] == "MIXED_1_TO_63" for row in excluded_split),
                    "removed_mixed_64_to_127_count": sum(row["purity_label"] == "MIXED_64_TO_127" for row in excluded_split),
                }
            )
    return output


def split_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    subjects = sorted({row["subject_id"] for row in rows})
    output: list[dict[str, Any]] = []
    for subject in subjects + ["ALL"]:
        scope = list(rows) if subject == "ALL" else [row for row in rows if row["subject_id"] == subject]
        total = len(scope)
        total_fog = sum(int(row["y_binary"]) for row in scope)
        total_nonfog = total - total_fog
        for split in SPLITS:
            chosen = [row for row in scope if row["ca_split"] == split]
            fog = sum(int(row["y_binary"]) for row in chosen)
            nonfog = len(chosen) - fog
            output.append(
                {
                    "subject_id": subject,
                    "subject_scope": "aggregate" if subject == "ALL" else chosen[0]["subject_scope"] if chosen else "",
                    "ca_split": split,
                    "source_target_fraction": ca.TARGETS[split],
                    "window_count": len(chosen),
                    "window_fraction": len(chosen) / total if total else "",
                    "pure_fog_window_count": fog,
                    "pure_fog_window_fraction": fog / total_fog if total_fog else "",
                    "pure_nonfog_window_count": nonfog,
                    "pure_nonfog_window_fraction": nonfog / total_nonfog if total_nonfog else "",
                }
            )
    return output


def group_manifest(
    source_groups: Sequence[dict[str, str]],
    source_rows: Sequence[dict[str, str]],
    retained: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_counts = Counter(row["group_id"] for row in source_rows)
    retained_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        retained_by_group[str(row["group_id"])].append(row)
    output: list[dict[str, Any]] = []
    for source in source_groups:
        group_id = source["group_id"]
        selected = retained_by_group[group_id]
        row: dict[str, Any] = dict(source)
        pure_fog = sum(int(item["y_binary"]) for item in selected)
        row.update(
            {
                "source_ca_window_count": source_counts[group_id],
                "pure_window_count": len(selected),
                "pure_fog_window_count": pure_fog,
                "pure_nonfog_window_count": len(selected) - pure_fog,
                "removed_mixed_window_count": source_counts[group_id] - len(selected),
                "pure_group_status": "retained" if selected else "no_pure_windows",
            }
        )
        output.append(row)
    return output


def event_manifest(
    source_events: Sequence[dict[str, str]],
    source_rows: Sequence[dict[str, str]],
    retained: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    retained_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_record[row["record_id"]].append(row)
    for row in retained:
        retained_by_record[row["record_id"]].append(row)
    output: list[dict[str, Any]] = []
    for source in source_events:
        record_id = source["record_id"]
        start = int(source["start_index"])
        end = int(source.get("end_index_exclusive") or int(source["end_index"]) + 1)

        def overlaps(row: dict[str, Any]) -> bool:
            return int(row["start_index"]) < end and int(row["end_index_exclusive"]) > start

        source_window_rows = [row for row in source_by_record[record_id] if overlaps(row)]
        pure_window_rows = [row for row in retained_by_record[record_id] if overlaps(row)]
        pure_fog = sum(int(row["y_binary"]) for row in pure_window_rows)
        row: dict[str, Any] = dict(source)
        row.update(
            {
                "source_ca_overlapping_window_count": len(source_window_rows),
                "pure_overlapping_window_count": len(pure_window_rows),
                "pure_fog_window_count": pure_fog,
                "pure_nonfog_window_count": len(pure_window_rows) - pure_fog,
                "pure_event_status": "has_pure_fog" if pure_fog else "no_pure_fog_window",
            }
        )
        output.append(row)
    return output


def ratio_quality(summary: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_subject: dict[str, dict[str, float]] = defaultdict(dict)
    for row in summary:
        if row["subject_id"] != "ALL":
            by_subject[row["subject_id"]][row["ca_split"]] = float(row["window_fraction"])
    subjects: dict[str, Any] = {}
    for subject, fractions in sorted(by_subject.items()):
        strict = bool(
            0.60 <= fractions["train"] <= 0.70
            and 0.15 <= fractions["validation"] <= 0.20
            and 0.15 <= fractions["test"] <= 0.25
        )
        approximate = bool(
            0.60 <= fractions["train"] <= 0.70
            and 0.14 <= fractions["validation"] <= 0.21
            and 0.15 <= fractions["test"] <= 0.25
        )
        subjects[subject] = {
            "window_fractions": fractions,
            "strict_ratio_gate_pass": strict,
            "approximate_ratio_gate_pass": approximate,
        }
    return {
        "subjects": subjects,
        "strict_ratio_pass_subjects": sum(item["strict_ratio_gate_pass"] for item in subjects.values()),
        "approximate_ratio_pass_subjects": sum(item["approximate_ratio_gate_pass"] for item in subjects.values()),
        "all_subjects_approximate_ratio_gate_pass": all(item["approximate_ratio_gate_pass"] for item in subjects.values()),
    }


def quality_report(
    source_rows: Sequence[dict[str, str]],
    retained: Sequence[dict[str, Any]],
    excluded: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ratio = ratio_quality(summary)
    leakage = ca.leakage_audit(retained)
    source_ids = {row["window_id"] for row in source_rows}
    retained_ids = {row["window_id"] for row in retained}
    excluded_ids = {row["window_id"] for row in excluded}
    pure_values = {int(row["fog_samples_in_2s"]) for row in retained}
    report = {
        "overall_pass": False,
        "source_ca_window_count": len(source_rows),
        "retained_pure_window_count": len(retained),
        "removed_mixed_window_count": len(excluded),
        "pure_fog_window_count": sum(int(row["y_binary"]) for row in retained),
        "pure_nonfog_window_count": sum(not int(row["y_binary"]) for row in retained),
        "removed_mixed_1_to_63_count": sum(row["purity_label"] == "MIXED_1_TO_63" for row in excluded),
        "removed_mixed_64_to_127_count": sum(row["purity_label"] == "MIXED_64_TO_127" for row in excluded),
        "retained_plus_excluded_reconciles_to_source": bool(
            len(retained) + len(excluded) == len(source_rows)
            and retained_ids.isdisjoint(excluded_ids)
            and retained_ids | excluded_ids == source_ids
        ),
        "retained_fog_sample_values": sorted(pure_values),
        "only_0_or_128_fog_samples_pass": pure_values.issubset({0, WINDOW}),
        "all_windows_length_128": all(
            int(row["end_index_exclusive"]) - int(row["start_index"]) == WINDOW
            for row in retained
        ),
        "all_source_split_assignments_preserved": all(
            next(source for source in source_rows if source["window_id"] == row["window_id"])["ca_split"] == row["ca_split"]
            for row in retained
        ),
        "leakage_audit": leakage,
        "ratio_quality": ratio,
    }
    report["overall_pass"] = bool(
        report["retained_plus_excluded_reconciles_to_source"]
        and report["only_0_or_128_fog_samples_pass"]
        and report["all_windows_length_128"]
        and report["all_source_split_assignments_preserved"]
        and leakage["pass"]
        and ratio["all_subjects_approximate_ratio_gate_pass"]
    )
    return report


def save_indices(
    root: Path,
    rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, str]],
    source_groups: Sequence[dict[str, str]],
) -> None:
    split_dir = root / "split_indices"
    split_dir.mkdir(parents=True, exist_ok=True)
    record_order = {row["record_id"]: index for index, row in enumerate(manifest_rows)}
    group_index = {row["group_id"]: index for index, row in enumerate(source_groups)}
    subjects = sorted({row["subject_id"] for row in rows})
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
            split_dir / f"{subject}_ca_pure_window_indices.npz",
            record_index=np.asarray([record_order[row["record_id"]] for row in selected], dtype=np.int16),
            start_index=np.asarray([row["start_index"] for row in selected], dtype=np.int32),
            end_index_exclusive=np.asarray([row["end_index_exclusive"] for row in selected], dtype=np.int32),
            split_code=np.asarray([SPLIT_CODES[row["ca_split"]] for row in selected], dtype=np.int8),
            y_binary=np.asarray([row["y_binary"] for row in selected], dtype=np.int8),
            fog_samples_in_2s=np.asarray([row["fog_samples_in_2s"] for row in selected], dtype=np.int16),
            group_index=np.asarray([group_index[row["group_id"]] for row in selected], dtype=np.int32),
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    source_quality = json.loads((source / "ca_quality_report.json").read_text(encoding="utf-8"))
    if not source_quality.get("overall_pass"):
        raise RuntimeError("Source processed_CA quality gate is not PASS")

    source_rows = base.read_csv(source / "ca_window_manifest.csv")
    source_groups = base.read_csv(source / "ca_group_manifest.csv")
    source_events = base.read_csv(source / "ca_fog_event_manifest.csv")
    manifest_rows = base.read_csv(source / "manifest.csv")
    retained, excluded = pure_rows(source_rows)
    summaries = split_summary(retained)
    filters = filter_summary(source_rows, retained, excluded)
    groups = group_manifest(source_groups, source_rows, retained)
    events = event_manifest(source_events, source_rows, retained)
    quality = quality_report(source_rows, retained, excluded, summaries)
    payload = {
        "output": str(output),
        "dry_run": bool(args.dry_run),
        "quality": quality,
        "aggregate_split_summary": [row for row in summaries if row["subject_id"] == "ALL"],
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
    for optional in ("ca_source_fog_segment_summary.csv", "ca_source_summary_audit.csv"):
        if (source / optional).exists():
            shutil.copy2(source / optional, build / optional)
    shutil.copytree(source / "records", build / "records")

    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    source_ca_schema = schema.pop("ca_split", None)
    schema["ca_pure_split"] = {
        "window_manifest": "ca_window_manifest.csv",
        "group_manifest": "ca_group_manifest.csv",
        "event_manifest": "ca_fog_event_manifest.csv",
        "summary": "ca_split_summary.csv",
        "filter_summary": "ca_pure_filter_summary.csv",
        "window_samples": WINDOW,
        "stride_samples": 64,
        "pure_fog_definition": "all 128 samples are FOG",
        "pure_nonfog_definition": "all 128 samples are NONFOG",
        "mixed_window_policy": "exclude every window with 1-127 FOG samples",
        "split_policy": "preserve frozen processed_CA group and split assignments",
        "source_ca_schema": source_ca_schema,
    }
    base.write_json(build / "schema.json", schema)
    base.write_csv(build / "ca_window_manifest.csv", retained)
    base.write_csv(build / "ca_group_manifest.csv", groups)
    base.write_csv(build / "ca_fog_event_manifest.csv", events)
    base.write_csv(build / "ca_split_summary.csv", summaries)
    base.write_csv(build / "ca_pure_filter_summary.csv", filters)
    base.write_csv(build / "ca_pure_excluded_windows.csv", excluded)
    base.write_json(build / "ca_split_codes.json", SPLIT_CODES)
    base.write_json(build / "ca_pure_quality_report.json", quality)
    # Compatibility name for readers that expect the standard CA quality filename.
    base.write_json(build / "ca_quality_report.json", quality)
    save_indices(build, retained, manifest_rows, source_groups)

    protocol = {
        "dataset_id": "daphnet_CA_pure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed_ca": str(source),
        "source_ca_window_manifest_sha256": base.sha256(source / "ca_window_manifest.csv"),
        "sampling_rate_hz": 64,
        "window_samples": WINDOW,
        "stride_samples": 64,
        "pure_fog_rule": "fog_samples_in_2s == 128",
        "pure_nonfog_rule": "fog_samples_in_2s == 0",
        "mixed_boundary_rule": "remove windows with 1 <= fog_samples_in_2s <= 127",
        "source_split_preserved": True,
        "random_resplit": False,
        "paired_ablation_ready": True,
    }
    base.write_json(build / "ca_protocol.json", protocol)
    (build / "README_CA_PURE.md").write_text(
        "# Daphnet processed_CA_pure\n\n"
        "A strict-purity derivative of processed_CA.\n\n"
        "- PURE_FOG: all 128 samples in the 2 s window are FOG.\n"
        "- PURE_NONFOG: all 128 samples are NONFOG.\n"
        "- Every mixed window containing 1-127 FOG samples is excluded.\n"
        "- The original within-subject CA group and train/validation/test assignments are preserved.\n"
        "- No random re-splitting is performed.\n",
        encoding="utf-8",
    )
    build.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
