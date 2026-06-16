#!/usr/bin/env python
"""Derive a patch-block dataset with a shorter PRE_FOG horizon.

This is useful when raw Kaggle CSV files are not extracted but an existing
patch dataset with a longer PRE_FOG horizon is available. FOG patches are kept
unchanged; PRE_FOG is recomputed as patches overlapping the target horizon
before each patch-level FOG onset.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


LABEL_NORMAL = 0
LABEL_PRE_FOG = 1
LABEL_FOG = 2
CLASS_NAMES = np.array(["NORMAL", "PRE_FOG", "FOG"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel PRE_FOG horizon in patch_blocks.npz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/processed/fog_patch_blocks_seq128"),
        help="Existing patch dataset with patch_blocks.npz and loso_folds.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the relabeled dataset.",
    )
    parser.add_argument("--pre-fog-seconds", type=float, required=True)
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed for patch_blocks.npz.",
    )
    return parser.parse_args()


def save_npz(path: Path, compress: bool, **arrays) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def relabel_prefog(
    patch_y: np.ndarray,
    patch_file_code: np.ndarray,
    patch_start_sample: np.ndarray,
    patch_end_sample: np.ndarray,
    patch_native_hz: np.ndarray,
    pre_fog_seconds: float,
) -> np.ndarray:
    new_y = patch_y.copy()
    new_y[new_y == LABEL_PRE_FOG] = LABEL_NORMAL

    for file_code in np.unique(patch_file_code):
        ids = np.flatnonzero(patch_file_code == file_code)
        ids = ids[np.argsort(patch_start_sample[ids], kind="stable")]
        old_labels = patch_y[ids]
        fog_local = np.flatnonzero(old_labels == LABEL_FOG)
        if fog_local.size == 0:
            continue

        previous_fog = np.r_[False, old_labels[:-1] == LABEL_FOG]
        onset_local = np.flatnonzero((old_labels == LABEL_FOG) & ~previous_fog)
        for onset_pos in onset_local:
            onset_id = ids[onset_pos]
            hz = int(patch_native_hz[onset_id])
            horizon_start = int(round(patch_start_sample[onset_id] - pre_fog_seconds * hz))
            horizon_end = int(patch_start_sample[onset_id])
            overlap = (
                (patch_end_sample[ids] > horizon_start)
                & (patch_start_sample[ids] < horizon_end)
                & (old_labels != LABEL_FOG)
            )
            new_y[ids[overlap]] = LABEL_PRE_FOG

    return new_y.astype(np.int8, copy=False)


def write_loso_folds(output_dir: Path, base_folds, arrays: dict[str, np.ndarray]) -> None:
    subjects = base_folds["subjects"]
    test_subjects = base_folds["fold_test_subjects"]
    val_subjects = base_folds["fold_val_subjects"]
    patch_subject_code = arrays["patch_subject_code"]
    block_subject_code = arrays["block_subject_code"]
    patch_y = arrays["patch_y"]

    np.savez(
        output_dir / "loso_folds.npz",
        subjects=subjects,
        fold_test_subjects=test_subjects,
        fold_val_subjects=val_subjects,
        patch_subject_code=patch_subject_code,
        block_subject_code=block_subject_code,
        class_names=CLASS_NAMES,
        config_json=np.array(json.dumps(arrays["config"], ensure_ascii=False)),
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
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_dir / "patch_blocks.npz", allow_pickle=True) as src:
        config = json.loads(str(src["config_json"].item()))
        patch_y = src["patch_y"]
        new_y = relabel_prefog(
            patch_y=patch_y,
            patch_file_code=src["patch_file_code"],
            patch_start_sample=src["patch_start_sample"],
            patch_end_sample=src["patch_end_sample"],
            patch_native_hz=src["patch_native_hz"],
            pre_fog_seconds=args.pre_fog_seconds,
        )
        config["pre_fog_seconds"] = float(args.pre_fog_seconds)
        config["derived_from"] = str(args.input_dir)
        config["derivation_method"] = "patch-level FOG onset relabel"

        save_npz(
            args.output_dir / "patch_blocks.npz",
            args.compress,
            patch_X=src["patch_X"],
            patch_y=new_y,
            patch_subject_code=src["patch_subject_code"],
            patch_source_code=src["patch_source_code"],
            patch_file_code=src["patch_file_code"],
            patch_start_sample=src["patch_start_sample"],
            patch_end_sample=src["patch_end_sample"],
            patch_native_hz=src["patch_native_hz"],
            block_patch_ids=src["block_patch_ids"],
            block_subject_code=src["block_subject_code"],
            block_source_code=src["block_source_code"],
            block_file_code=src["block_file_code"],
            subjects=src["subjects"],
            source_names=src["source_names"],
            file_ids=src["file_ids"],
            class_names=CLASS_NAMES,
            config_json=np.array(json.dumps(config, ensure_ascii=False)),
        )

        arrays = {
            "patch_y": new_y,
            "patch_subject_code": src["patch_subject_code"].copy(),
            "block_subject_code": src["block_subject_code"].copy(),
            "config": config,
        }

    with np.load(args.input_dir / "loso_folds.npz", allow_pickle=True) as folds:
        write_loso_folds(args.output_dir, folds, arrays)

    for name in ("README.md", "file_summary.csv"):
        src_path = args.input_dir / name
        if src_path.exists():
            shutil.copy2(src_path, args.output_dir / name)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = np.bincount(new_y.astype(int), minlength=3).tolist()
    print(f"Output: {args.output_dir}")
    print(f"Patch label counts NORMAL/PRE_FOG/FOG: {counts}")


if __name__ == "__main__":
    main()
