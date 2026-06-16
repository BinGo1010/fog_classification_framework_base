#!/usr/bin/env python
"""Prepare patch-token LOSO blocks for FOG/NORMAL/PRE_FOG sequence classification."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_fog_loso_npz import (
    CLASS_NAMES,
    DATASET_HZ,
    FEATURE_COLUMNS,
    LABEL_FOG,
    LABEL_NORMAL,
    LABEL_PRE_FOG,
    build_sample_state,
    build_valid_mask,
    choose_val_subject,
    contiguous_true_intervals,
    iter_records,
    load_tasks,
    parse_fog_columns,
    read_source_csv,
    resample_window,
    save_npz,
)


@dataclass
class PatchFileSummary:
    source: str
    file_id: str
    subject: str
    rows: int
    valid_rows: int
    patches: int
    blocks: int
    normal_patches: int
    pre_fog_patches: int
    fog_patches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build long patch-token blocks for Transformer-BiLSTM FOG classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/competition data/Competition Dataset"),
        help="Competition Dataset directory containing train/, metadata, and tasks.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/fog_patch_blocks_seq128"),
        help="Output directory for patch_blocks.npz and LOSO metadata.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(DATASET_HZ),
        default=["tdcsfog", "defog"],
    )
    parser.add_argument(
        "--fog-columns",
        default="StartHesitation,Turn,Walking",
        help="Comma-separated event columns treated as FOG.",
    )
    parser.add_argument("--pre-fog-seconds", type=float, default=3.0)
    parser.add_argument(
        "--patch-seconds",
        type=float,
        default=18 / 128,
        help="Native-duration represented by one token. 18/128 maps defog to about 14 samples.",
    )
    parser.add_argument(
        "--target-hz",
        type=int,
        default=128,
        help="Feature patches are resampled to target_hz before flattening.",
    )
    parser.add_argument(
        "--target-patch-samples",
        type=int,
        default=18,
        help="Flattened feature dimension is target_patch_samples * 3.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Number of patch tokens in each model block.",
    )
    parser.add_argument(
        "--block-stride-tokens",
        type=int,
        default=32,
        help="Rolling stride between neighboring blocks, in patch tokens.",
    )
    parser.add_argument(
        "--patch-label-rule",
        choices=("priority", "center", "majority"),
        default="priority",
        help="Rule for reducing per-sample labels inside one patch.",
    )
    parser.add_argument(
        "--defog-valid-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For defog, keep only rows where Valid and Task are true.",
    )
    parser.add_argument(
        "--exclude-rest-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For defog, use tasks.csv to remove Rest1/Rest2 intervals.",
    )
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        help="Debug option to process only the first N files from each source.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed. Smaller files, slower to write/read.",
    )
    return parser.parse_args()


def label_patch(state: np.ndarray, start: int, end: int, rule: str) -> int:
    values = state[start:end]
    if len(values) == 0:
        return LABEL_NORMAL
    if rule == "center":
        return int(values[len(values) // 2])
    if rule == "majority":
        return int(np.bincount(values, minlength=3).argmax())
    if np.any(values == LABEL_FOG):
        return LABEL_FOG
    if np.any(values == LABEL_PRE_FOG):
        return LABEL_PRE_FOG
    return LABEL_NORMAL


def block_starts(num_tokens: int, seq_len: int, stride: int) -> list[int]:
    if num_tokens <= 0:
        return []
    if num_tokens <= seq_len:
        return [0]

    starts = list(range(0, num_tokens - seq_len + 1, stride))
    last = num_tokens - seq_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def write_file_summary(output_dir: Path, summaries: list[PatchFileSummary]) -> None:
    if not summaries:
        return
    with (output_dir / "file_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def write_loso_folds(
    output_dir: Path,
    subjects: np.ndarray,
    patch_subject_code: np.ndarray,
    block_subject_code: np.ndarray,
    patch_y: np.ndarray,
    config: dict,
) -> None:
    test_subjects = subjects.copy()
    val_subjects = np.array(
        [choose_val_subject(subjects, str(subject)) for subject in test_subjects],
        dtype="U16",
    )

    np.savez(
        output_dir / "loso_folds.npz",
        subjects=subjects,
        fold_test_subjects=test_subjects,
        fold_val_subjects=val_subjects,
        patch_subject_code=patch_subject_code,
        block_subject_code=block_subject_code,
        class_names=CLASS_NAMES,
        config_json=np.array(json.dumps(config, ensure_ascii=False)),
    )

    rows = []
    for fold_idx, (test_subject, val_subject) in enumerate(zip(test_subjects, val_subjects)):
        test_code = int(np.flatnonzero(subjects == test_subject)[0])
        val_code = int(np.flatnonzero(subjects == val_subject)[0])
        train_codes = [code for code in range(len(subjects)) if code not in (test_code, val_code)]
        train_patch_mask = np.isin(patch_subject_code, train_codes)
        val_patch_mask = patch_subject_code == val_code
        test_patch_mask = patch_subject_code == test_code
        train_block_mask = np.isin(block_subject_code, train_codes)
        val_block_mask = block_subject_code == val_code
        test_block_mask = block_subject_code == test_code
        rows.append(
            {
                "fold": fold_idx,
                "test_subject": str(test_subject),
                "val_subject": str(val_subject),
                "train_blocks": int(train_block_mask.sum()),
                "val_blocks": int(val_block_mask.sum()),
                "test_blocks": int(test_block_mask.sum()),
                "train_patches": int(train_patch_mask.sum()),
                "val_patches": int(val_patch_mask.sum()),
                "test_patches": int(test_patch_mask.sum()),
                "train_normal": int(np.sum(patch_y[train_patch_mask] == LABEL_NORMAL)),
                "train_pre_fog": int(np.sum(patch_y[train_patch_mask] == LABEL_PRE_FOG)),
                "train_fog": int(np.sum(patch_y[train_patch_mask] == LABEL_FOG)),
                "val_normal": int(np.sum(patch_y[val_patch_mask] == LABEL_NORMAL)),
                "val_pre_fog": int(np.sum(patch_y[val_patch_mask] == LABEL_PRE_FOG)),
                "val_fog": int(np.sum(patch_y[val_patch_mask] == LABEL_FOG)),
                "test_normal": int(np.sum(patch_y[test_patch_mask] == LABEL_NORMAL)),
                "test_pre_fog": int(np.sum(patch_y[test_patch_mask] == LABEL_PRE_FOG)),
                "test_fog": int(np.sum(patch_y[test_patch_mask] == LABEL_FOG)),
            }
        )

    with (output_dir / "loso_folds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.block_stride_tokens <= 0:
        raise ValueError("--block-stride-tokens must be positive.")
    if args.target_patch_samples <= 0:
        raise ValueError("--target-patch-samples must be positive.")

    fog_columns = parse_fog_columns(args.fog_columns)
    records = iter_records(args)
    tasks_by_id = load_tasks(args.data_root)
    subjects = np.array(sorted({record.subject for record in records}), dtype="U16")
    subject_to_code = {subject: idx for idx, subject in enumerate(subjects)}
    source_names = np.array(args.sources, dtype="U8")
    source_to_code = {source: idx for idx, source in enumerate(source_names)}
    file_ids = np.array([record.file_id for record in records], dtype="U16")
    file_to_code = {record.file_id: idx for idx, record in enumerate(records)}

    patch_features: list[np.ndarray] = []
    patch_labels: list[int] = []
    patch_subject_codes: list[int] = []
    patch_source_codes: list[int] = []
    patch_file_codes: list[int] = []
    patch_start_samples: list[int] = []
    patch_end_samples: list[int] = []
    patch_native_hz: list[int] = []
    block_patch_ids: list[np.ndarray] = []
    block_subject_codes: list[int] = []
    block_source_codes: list[int] = []
    block_file_codes: list[int] = []
    summaries: list[PatchFileSummary] = []

    print(f"Processing {len(records)} files from {args.data_root}")
    for record_idx, record in enumerate(records, start=1):
        df = read_source_csv(record)
        state = build_sample_state(df, fog_columns, record.hz, args.pre_fog_seconds)
        valid = build_valid_mask(
            record,
            df,
            tasks_by_id,
            args.defog_valid_only,
            args.exclude_rest_tasks,
        )
        features = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32, copy=False)
        times = df["Time"].to_numpy(dtype=np.int64, copy=False)
        native_patch_size = max(1, int(round(args.patch_seconds * record.hz)))
        record_patch_ids: list[int] = []
        record_block_count = 0

        for interval_start, interval_end in contiguous_true_intervals(valid):
            num_native = interval_end - interval_start
            num_patches = num_native // native_patch_size
            if num_patches <= 0:
                continue

            interval_patch_ids = []
            for patch_idx in range(num_patches):
                start = interval_start + patch_idx * native_patch_size
                end = start + native_patch_size
                patch_id = len(patch_labels)
                patch_features.append(
                    resample_window(
                        features[start:end],
                        args.target_patch_samples,
                    ).reshape(-1)
                )
                patch_labels.append(label_patch(state, start, end, args.patch_label_rule))
                patch_subject_codes.append(subject_to_code[record.subject])
                patch_source_codes.append(source_to_code[record.source])
                patch_file_codes.append(file_to_code[record.file_id])
                patch_start_samples.append(int(times[start]))
                patch_end_samples.append(int(times[end - 1] + 1))
                patch_native_hz.append(record.hz)
                interval_patch_ids.append(patch_id)
                record_patch_ids.append(patch_id)

            interval_patch_ids_arr = np.asarray(interval_patch_ids, dtype=np.int32)
            for start_token in block_starts(
                len(interval_patch_ids_arr),
                args.seq_len,
                args.block_stride_tokens,
            ):
                ids = np.full(args.seq_len, -1, dtype=np.int32)
                selected = interval_patch_ids_arr[start_token : start_token + args.seq_len]
                ids[: len(selected)] = selected
                block_patch_ids.append(ids)
                block_subject_codes.append(subject_to_code[record.subject])
                block_source_codes.append(source_to_code[record.source])
                block_file_codes.append(file_to_code[record.file_id])
                record_block_count += 1

        if record_patch_ids:
            labels = np.asarray([patch_labels[idx] for idx in record_patch_ids], dtype=np.int8)
            counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        else:
            counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        summaries.append(
            PatchFileSummary(
                source=record.source,
                file_id=record.file_id,
                subject=record.subject,
                rows=len(df),
                valid_rows=int(valid.sum()),
                patches=len(record_patch_ids),
                blocks=record_block_count,
                normal_patches=int(counts[LABEL_NORMAL]),
                pre_fog_patches=int(counts[LABEL_PRE_FOG]),
                fog_patches=int(counts[LABEL_FOG]),
            )
        )
        if record_idx % 100 == 0 or record_idx == len(records):
            print(
                f"  {record_idx:04d}/{len(records)} files, "
                f"patches={len(patch_labels)}, blocks={len(block_patch_ids)}"
            )

    patch_X = np.asarray(patch_features, dtype=np.float32)
    patch_y = np.asarray(patch_labels, dtype=np.int8)
    patch_subject_code = np.asarray(patch_subject_codes, dtype=np.int16)
    patch_source_code = np.asarray(patch_source_codes, dtype=np.int8)
    patch_file_code = np.asarray(patch_file_codes, dtype=np.int32)
    block_patch_ids_arr = np.asarray(block_patch_ids, dtype=np.int32)
    block_subject_code = np.asarray(block_subject_codes, dtype=np.int16)
    block_source_code = np.asarray(block_source_codes, dtype=np.int8)
    block_file_code = np.asarray(block_file_codes, dtype=np.int32)

    config = {
        "data_root": str(args.data_root),
        "sources": args.sources,
        "fog_columns": fog_columns,
        "pre_fog_seconds": args.pre_fog_seconds,
        "patch_seconds": args.patch_seconds,
        "target_hz": args.target_hz,
        "target_patch_samples": args.target_patch_samples,
        "feature_dim": int(args.target_patch_samples * len(FEATURE_COLUMNS)),
        "seq_len": args.seq_len,
        "block_stride_tokens": args.block_stride_tokens,
        "patch_label_rule": args.patch_label_rule,
        "defog_valid_only": args.defog_valid_only,
        "exclude_rest_tasks": args.exclude_rest_tasks,
        "class_names": CLASS_NAMES.tolist(),
        "feature_names": list(FEATURE_COLUMNS),
        "dataset_hz": DATASET_HZ,
        "native_patch_sizes": {
            source: max(1, int(round(args.patch_seconds * hz)))
            for source, hz in DATASET_HZ.items()
        },
    }

    save_npz(
        args.output_dir / "patch_blocks.npz",
        args.compress,
        patch_X=patch_X,
        patch_y=patch_y,
        patch_subject_code=patch_subject_code,
        patch_source_code=patch_source_code,
        patch_file_code=patch_file_code,
        patch_start_sample=np.asarray(patch_start_samples, dtype=np.int64),
        patch_end_sample=np.asarray(patch_end_samples, dtype=np.int64),
        patch_native_hz=np.asarray(patch_native_hz, dtype=np.int16),
        block_patch_ids=block_patch_ids_arr,
        block_subject_code=block_subject_code,
        block_source_code=block_source_code,
        block_file_code=block_file_code,
        subjects=subjects,
        source_names=source_names,
        file_ids=file_ids,
        class_names=CLASS_NAMES,
        config_json=np.array(json.dumps(config, ensure_ascii=False)),
    )
    write_loso_folds(
        args.output_dir,
        subjects,
        patch_subject_code,
        block_subject_code,
        patch_y,
        config,
    )
    write_file_summary(args.output_dir, summaries)

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    counts = np.bincount(patch_y, minlength=len(CLASS_NAMES))
    readme = [
        "# FOG patch-token LOSO dataset",
        "",
        f"- patches: {len(patch_y)}",
        f"- blocks: {len(block_patch_ids_arr)}",
        f"- patch_X: {tuple(patch_X.shape)}",
        f"- block_patch_ids: {tuple(block_patch_ids_arr.shape)}",
        f"- class counts: {dict(zip(CLASS_NAMES.tolist(), counts.astype(int).tolist()))}",
        f"- subjects/folds: {len(subjects)}",
        "",
        "Each block stores patch ids. Padded token ids are -1 and must be masked in loss/evaluation.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("Done.")
    print("\n".join(readme[2:7]))


if __name__ == "__main__":
    main()
