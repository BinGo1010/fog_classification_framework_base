from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scripts.run_daphnet_processed_nbm_loso_s01_gru_v1_c_tcn as loso
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import RoleRows


def rows(subjects: list[str], roles: list[int], labels: list[int]) -> RoleRows:
    count = len(subjects)
    return RoleRows(
        subject_id=np.asarray(subjects, dtype="U3"),
        record_id=np.asarray([f"R{i}" for i in range(count)], dtype="U32"),
        start=np.arange(count, dtype=np.int32) * 128,
        end=np.arange(1, count + 1, dtype=np.int32) * 128,
        role=np.asarray(roles, dtype=np.int8),
        label=np.asarray(labels, dtype=np.int8),
        window_id=np.asarray([f"W{i}" for i in range(count)], dtype="U96"),
    )


def test_fixed_loso_identity() -> None:
    assert loso.TEST_SUBJECT == "S01"
    assert loso.VALIDATION_SUBJECT == "S02"
    assert "S01" not in loso.DEVELOPMENT_SUBJECTS
    assert "S02" not in loso.TRAIN_SUBJECTS
    assert set(loso.TRAIN_SUBJECTS) == {"S03", "S05", "S06", "S07", "S08", "S09"}
    assert loso.SOURCE_OUTER_FOLD == 0
    assert loso.NBM_SEED == loso.TCN_SEED == 0


def test_subject_filter_is_disjoint() -> None:
    source = rows(["S01", "S02", "S03", "S01"], [0, 4, 7, 1], [0, 0, 1, 1])
    heldout = loso.take_subjects(source, ("S01",))
    development = loso.take_subjects(source, ("S02", "S03"))
    assert set(heldout.subject_id.tolist()) == {"S01"}
    assert set(development.subject_id.tolist()) == {"S02", "S03"}
    assert set(heldout.window_id.tolist()).isdisjoint(development.window_id.tolist())
    assert set(heldout.role.tolist()) == {0, 1}


def test_scaler_rejects_s01() -> None:
    source = rows(["S01"], [4], [0])
    with pytest.raises(AssertionError, match="S01 leaked"):
        loso.fit_scaler_unique_points({}, source)


def test_training_contract_is_frozen(tmp_path: Path) -> None:
    class Args:
        nbm_max_epochs = 299
        nbm_patience = 20
        tcn_max_epochs = 5
        tcn_patience = 2

    with pytest.raises(ValueError, match="max300"):
        loso.validate_contract(Args())


def test_architecture_and_scheme_c_contract() -> None:
    architecture = loso.architecture_config()
    assert architecture["parameter_count"] == 31_513
    assert architecture["latent_shape"] == ["B", 16]
    augmentation = loso.augmentation_config("MASK8_12")
    assert augmentation["mask_minimum_samples"] == 8
    assert augmentation["mask_maximum_samples"] == 12
    assert augmentation["gaussian_std"] == pytest.approx(0.04)


def test_real_processed_nbm_split_counts_and_subject_isolation() -> None:
    data_dir = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM"
    )
    source = loso.load_fold_rows(data_dir, loso.SOURCE_OUTER_FOLD)
    train = loso.take_subjects(source, loso.TRAIN_SUBJECTS)
    validation = loso.take_subjects(source, (loso.VALIDATION_SUBJECT,))
    test = loso.take_subjects(source, (loso.TEST_SUBJECT,))
    splits = {
        "nbm_train": train.take_role(4),
        "nbm_validation": validation.take_role(5),
        "classifier_train": train.take_role(6, 7),
        "classifier_validation": validation.take_role(2, 3),
        "test": test,
    }
    assert (len(splits["nbm_train"]), int(splits["nbm_train"].label.sum())) == (2129, 0)
    assert (len(splits["nbm_validation"]), int(splits["nbm_validation"].label.sum())) == (76, 0)
    assert (
        int(np.sum(splits["classifier_train"].label == 0)),
        int(np.sum(splits["classifier_train"].label == 1)),
    ) == (1773, 587)
    assert (
        int(np.sum(splits["classifier_validation"].label == 0)),
        int(np.sum(splits["classifier_validation"].label == 1)),
    ) == (315, 37)
    assert (
        int(np.sum(test.label == 0)),
        int(np.sum(test.label == 1)),
    ) == (1744, 64)
    assert set(test.role.tolist()) == set(range(8))
    test_ids = set(test.window_id.tolist())
    for name, split in splits.items():
        if name != "test":
            assert test_ids.isdisjoint(split.window_id.tolist())
