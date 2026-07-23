from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "augment_loso_window_dataset.py"


def make_tiny_loso_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    subjects = np.array(["S01", "S02", "S03", "S04"])
    subject_code = np.repeat(np.arange(4, dtype=np.int16), 6)
    y = np.tile(np.array([0, 0, 0, 1, 2, 2], dtype=np.int8), 4)
    x = rng.normal(size=(len(y), 8, 3)).astype(np.float32)
    data_dir.mkdir(parents=True)
    np.savez(
        data_dir / "windows.npz",
        X=x,
        y=y,
        subject=subjects[subject_code],
        subject_code=subject_code,
        subjects=subjects,
        file_id=np.array([f"window_{idx:03d}" for idx in range(len(y))]),
        class_names=np.array(["NORMAL", "PRE_FOG", "FOG"]),
        feature_names=np.array(["acc_x", "acc_y", "acc_z"]),
    )
    np.savez(
        data_dir / "loso_folds.npz",
        subjects=subjects,
        fold_test_subjects=subjects,
        fold_val_subjects=np.roll(subjects, -1),
        fold_test_subject_codes=np.arange(4, dtype=np.int16)[:, None],
        fold_val_subject_codes=np.roll(np.arange(4, dtype=np.int16), -1)[:, None],
        window_subject_code=subject_code,
    )
    return x, y, subject_code


def test_balances_only_training_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "input"
    output_dir = tmp_path / "augmented"
    source_x, source_y, source_subject_code = make_tiny_loso_dataset(data_dir)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "123",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with (output_dir / "augmentation_report.json").open(encoding="utf-8") as file:
        report = json.load(file)
    assert report["fold_count"] == 4
    assert list(report["aggregate_class_percent"].values()) == pytest.approx(
        [100.0 / 3.0] * 3
    )

    for fold_idx in range(1, 5):
        fold_dir = output_dir / f"loso_subject_{fold_idx:02d}"
        with np.load(fold_dir / "train.npz") as train:
            counts = np.bincount(train["y"], minlength=3)
            assert counts.tolist() == [6, 6, 6]
            assert int(train["is_augmented"].sum()) == 6
            parent_index = train["parent_index"]
            is_augmented = train["is_augmented"]
            np.testing.assert_array_equal(
                train["X"][~is_augmented], source_x[parent_index[~is_augmented]]
            )
            assert np.all(
                np.any(
                    train["X"][is_augmented] != source_x[parent_index[is_augmented]],
                    axis=(1, 2),
                )
            )

            test_code = fold_idx - 1
            val_code = fold_idx % 4
            assert not np.isin(
                source_subject_code[parent_index], [test_code, val_code]
            ).any()

        for split, expected_code in (("val", fold_idx % 4), ("test", fold_idx - 1)):
            with np.load(fold_dir / f"{split}.npz") as data:
                assert len(data["y"]) == 6
                assert not data["is_augmented"].any()
                np.testing.assert_array_equal(
                    data["X"], source_x[data["parent_index"]]
                )
                np.testing.assert_array_equal(
                    data["y"], source_y[data["parent_index"]]
                )
                assert np.all(data["subject_code"] == expected_code)
