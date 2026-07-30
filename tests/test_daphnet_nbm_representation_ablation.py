from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.histories import make_common_history_plan
import run_daphnet_nbm_representation_ablation as suite


def _args(tmp_path: Path, *, smoke: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        folds="all",
        worker_fold="",
        finalize_only=False,
        smoke=smoke,
        seed=42,
        classifier_hidden=48,
        classifier_dropout=0.15,
        classifier_epochs=12,
        classifier_patience=4,
        classifier_lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_classifier_windows=0,
        bootstrap_samples=100_000,
        bootstrap_seed=42,
        num_workers=0,
        device="cpu",
        amp=True,
        deterministic=True,
        resume=True,
        debug_interrupt_classifier_after_epoch=0,
        stop_after_completed_tasks=0,
    )


def _dataset_and_windows(tmp_path: Path) -> tuple[DaphnetDataset, WindowTable]:
    records = [
        Record(
            record_id=f"{subject}_R01",
            subject_id=subject,
            run_id="R01",
            x=np.zeros((192, 9), dtype=np.float32),
            y=np.zeros(192, dtype=np.int8),
            valid=np.ones(192, dtype=bool),
        )
        for subject in suite.EXPECTED_LOSO_SUBJECTS
    ]
    dataset = DaphnetDataset(
        root=tmp_path,
        records=records,
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    labels = np.asarray([0, 1], dtype=np.int8)
    windows = WindowTable(
        record_index=np.asarray([0, 1], dtype=np.int32),
        start=np.asarray([0, 0], dtype=np.int32),
        target_start=np.asarray([128, 128], dtype=np.int32),
        target_end=np.asarray([160, 160], dtype=np.int32),
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    return dataset, windows


def _source_config() -> dict:
    return {
        "excluded_subjects": ["S04", "S10"],
    }


def test_protocol_is_four_by_three_with_96_tcn_m_cells(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    dataset, windows = _dataset_and_windows(tmp_path)
    protocol = suite.build_protocol(
        args,
        source_manifest={"source_protocol_fingerprint": "a" * 64},
        source_config=_source_config(),
        dataset=dataset,
        windows=windows,
        data_sha256="b" * 64,
        device=torch.device("cpu"),
    )
    assert tuple(suite.NBMS) == (
        "persistence",
        "gru",
        "tcn",
        "transformer",
    )
    assert tuple(suite.REPRESENTATIONS) == (
        "error_x_minus_mu",
        "fixed_standardized_error",
        "dynamic_standardized_error",
    )
    assert len(protocol["cells"]) == 12
    assert protocol["expected_classifier_cells"] == 96
    assert {
        (cell["nbm"], cell["representation"])
        for cell in protocol["cells"]
    } == {
        (nbm, representation)
        for nbm in suite.NBMS
        for representation in suite.REPRESENTATIONS
    }
    assert protocol["context_samples"] == 128
    assert protocol["horizon_samples"] == 32
    assert protocol["history_samples"] == 256
    assert protocol["history_blocks"] == 8
    assert protocol["fixed_sigma"]["calibration_split"] == (
        "source normal_validation_window_index"
    )
    assert protocol["fixed_sigma"]["test_subject_used"] is False
    assert protocol["classifier"]["dilations"] == [1, 2, 4, 8, 8, 8]
    assert protocol["classifier"]["receptive_field_samples"] == 125
    assert protocol["classifier"]["parameter_count"] == 89_329
    assert protocol["reportable"] is True


def test_formal_arguments_are_locked_and_smoke_can_reduce(
    tmp_path: Path,
) -> None:
    formal = _args(tmp_path)
    suite.validate_args(formal)
    reduced = copy.copy(formal)
    reduced.classifier_epochs = 1
    reduced.batch_size = 8
    reduced.max_classifier_windows = 16
    reduced.bootstrap_samples = 100
    with pytest.raises(ValueError, match="--smoke"):
        suite.validate_args(reduced)
    reduced.smoke = True
    suite.validate_args(reduced)
    wrong_seed = copy.copy(reduced)
    wrong_seed.seed = 43
    with pytest.raises(ValueError, match="seed 42"):
        suite.validate_args(wrong_seed)


def test_history_materialization_is_shared_and_chronological() -> None:
    rows = np.arange(20, dtype=np.int64)
    target_start = 128 + rows.astype(np.int32) * suite.STRIDE_SAMPLES
    windows = WindowTable(
        record_index=np.zeros(len(rows), dtype=np.int32),
        start=target_start - suite.CONTEXT_SAMPLES,
        target_start=target_start,
        target_end=target_start + suite.HORIZON_SAMPLES,
        label=(rows % 3 == 0).astype(np.int8),
        fog_fraction=(rows % 3 == 0).astype(np.float32),
        clean_normal=rows % 3 != 0,
    )
    plan = make_common_history_plan(
        windows,
        rows,
        suite.HORIZON_SAMPLES,
        suite.STRIDE_SAMPLES,
        suite.HISTORY_SAMPLES,
    )
    block = np.broadcast_to(
        rows[:, None, None],
        (len(rows), 9, suite.HORIZON_SAMPLES),
    ).astype(np.float32)
    features = {
        split: {
            name: block
            for name in suite.REPRESENTATIONS
        }
        | {
            "y": windows.label,
            "window_index": rows,
        }
        for split in ("train", "validation", "test")
    }
    plans = {
        split: plan
        for split in ("train", "validation", "test")
    }
    outputs = [
        suite.materialize_inputs(features, plans, name)["train"][name]
        for name in suite.REPRESENTATIONS
    ]
    assert all(array.shape[1:] == (9, 256) for array in outputs)
    assert all(np.array_equal(outputs[0], array) for array in outputs[1:])
    first = outputs[0][0, 0].reshape(8, 32)
    np.testing.assert_array_equal(first[:, 0], np.arange(0, 16, 2))


def test_task_directories_keep_nbm_and_representation_separate(
    tmp_path: Path,
) -> None:
    roots = {
        suite.task_root_for(tmp_path, "S01", nbm, representation)
        for nbm in suite.NBMS
        for representation in suite.REPRESENTATIONS
    }
    assert len(roots) == 12
    assert (
        tmp_path
        / "loso_S01"
        / "transformer"
        / "fixed_standardized_error"
    ) in roots
