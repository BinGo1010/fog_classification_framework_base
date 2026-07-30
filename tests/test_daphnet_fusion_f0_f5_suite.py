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
import run_daphnet_fusion_f0_f5_suite as suite


def _args(tmp_path: Path, *, smoke: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        folds="all",
        worker_fold="",
        finalize_only=False,
        cache_only=False,
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
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
        "excluded_subjects": ["S04", "S10"],
        "sampling_rate_hz": 64,
        "nbm_hidden": 48,
        "nbm_dropout": 0.15,
        "linear_ar_seconds": 2.0,
        "gru_layers": 1,
        "transformer_heads": 4,
        "transformer_layers": 2,
        "transformer_ffn": 96,
    }


def test_protocol_fixes_six_inputs_48_cells_and_tcn_m(
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
    assert suite.FUSION_IDS == ("F0", "F1", "F2", "F3", "F4", "F5")
    assert protocol["expected_experiments"] == 6
    assert protocol["expected_classifier_cells"] == 48
    assert protocol["folds_resolved"] == [
        "S01",
        "S02",
        "S03",
        "S05",
        "S06",
        "S07",
        "S08",
        "S09",
    ]
    assert protocol["excluded_subjects"] == ["S04", "S10"]
    assert protocol["history_samples"] == 256
    assert protocol["history_blocks"] == 8
    assert protocol["classifier"]["dilations"] == [1, 2, 4, 8, 8, 8]
    assert protocol["classifier"]["receptive_field_samples"] == 125
    assert protocol["classifier"]["parameter_count_by_in_channels"] == {
        "9": 89_329,
        "18": 89_761,
        "27": 90_193,
    }
    assert protocol["primary_comparison"] == "F2_minus_F0"
    assert {
        item["comparison_id"] for item in protocol["comparisons"]
    } == {
        "F1_minus_F0",
        "F2_minus_F0",
        "F2_minus_F3",
        "F2_minus_F1",
        "F4_minus_F2",
        "F5_minus_F2",
        "F5_minus_F3",
    }
    assert protocol["source_model_reconstruction"][
        "transformer_layers"
    ] == 2
    assert protocol["reportable"] is True


def test_aligned_initialization_is_exact_within_width_and_shared_across_backbone() -> None:
    states, counts, hashes, backbone_hash = suite._aligned_reference_states(
        10042,
        48,
        0.15,
        True,
    )
    assert counts == {9: 89_329, 18: 89_761, 27: 90_193}
    assert len(set(hashes.values())) == 3
    for channels, state in states.items():
        assert state["projection.0.weight"].shape[1] == channels
        for name, value in states[27].items():
            if name == "projection.0.weight":
                continue
            assert torch.equal(state[name], value)
        assert torch.equal(
            state["projection.0.weight"],
            states[27]["projection.0.weight"][:, :channels],
        )
    assert isinstance(backbone_hash, str) and len(backbone_hash) == 64


def test_history_materialization_preserves_slots_and_chronology() -> None:
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
    raw = np.broadcast_to(
        rows[:, None, None],
        (len(rows), 9, suite.HORIZON_SAMPLES),
    ).astype(np.float32)
    error = raw + 100.0
    sigma = np.full_like(raw, 2.0)
    features = {
        split: {
            "raw": raw,
            "error": error,
            "sigma": sigma,
            "y": windows.label,
            "window_index": rows,
        }
        for split in ("train", "validation", "test")
    }
    plans = {split: plan for split in features}
    outputs = {
        name: suite.materialize_fusion_inputs(
            features,
            plans,
            name,
        )["train"][name]
        for name in suite.FUSION_IDS
    }
    for name, values in outputs.items():
        assert values.shape[1:] == (
            suite.FUSION_REPRESENTATIONS[name]["in_channels"],
            256,
        )
    first = outputs["F0"][0, 0].reshape(8, 32)
    np.testing.assert_array_equal(first[:, 0], np.arange(0, 16, 2))
    np.testing.assert_array_equal(outputs["F2"][:, :9], outputs["F0"])
    np.testing.assert_array_equal(outputs["F2"][:, 9:], outputs["F1"])
    np.testing.assert_array_equal(outputs["F3"][:, :9], outputs["F0"])
    assert np.count_nonzero(outputs["F3"][:, 9:]) == 0


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


def test_task_roots_are_one_directory_per_fold_and_fusion(
    tmp_path: Path,
) -> None:
    roots = {
        suite.task_root_for(tmp_path, "S01", name)
        for name in suite.FUSION_IDS
    }
    assert len(roots) == 6
    assert tmp_path / "loso_S01" / "F4" in roots
