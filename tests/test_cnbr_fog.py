from __future__ import annotations

import numpy as np
import torch

from cnbr_fog.data import DaphnetTrunkDataset, Record, WindowTable, valid_signal_mask
from cnbr_fog.evaluation import binary_metrics, choose_threshold
from cnbr_fog.histories import (
    history_block_count,
    make_block_history_input,
    make_common_history_plan,
    make_history_input,
    materialize_nonoverlap_residual_history,
)
from cnbr_fog.models import ConditionalNormalPredictor, ResidualTCNClassifier, gaussian_nll


def make_record(record_id: str, subject: str, y: np.ndarray) -> Record:
    x = np.stack(
        [
            np.linspace(0.0, 1.0, len(y)),
            np.linspace(1.0, 0.0, len(y)),
            np.ones(len(y)),
        ],
        axis=1,
    ).astype(np.float32)
    return Record(
        record_id=record_id,
        subject_id=subject,
        run_id="R01",
        x=x,
        y=y.astype(np.int8),
        valid=np.ones(len(y), dtype=bool),
    )


def test_flatline_rule_is_label_independent_and_run_based():
    x = np.ones((20, 3), dtype=np.float32)
    x[5:13] = 0.0
    mask = valid_signal_mask(x, sampling_rate_hz=4, flatline_seconds=2.0)
    assert mask[:5].all()
    assert not mask[5:13].any()
    assert mask[13:].all()


def test_flatline_rule_rejects_one_missing_sensor_triad():
    x = np.ones((20, 9), dtype=np.float32)
    x[4:14, 3:6] = 0.0
    mask = valid_signal_mask(x, sampling_rate_hz=4, flatline_seconds=2.0)
    assert mask[:4].all()
    assert not mask[4:14].any()
    assert mask[14:].all()


def test_windows_are_record_local_and_target_majority_labeled():
    y1 = np.zeros(20, dtype=np.int8)
    y1[12:16] = 1
    y2 = np.zeros(20, dtype=np.int8)
    dataset = DaphnetTrunkDataset(
        root=".",
        records=[make_record("r1", "S01", y1), make_record("r2", "S02", y2)],
        sampling_rate_hz=4,
    )
    windows = dataset.make_windows(
        warmup_samples=4,
        target_samples=4,
        stride_samples=4,
        fog_fraction_threshold=0.5,
        normal_guard_samples=1,
    )
    assert len(windows) == 8
    assert set(windows.record_index.tolist()) == {0, 1}
    assert np.all(windows.target_end - windows.start == 8)
    positive = np.flatnonzero(windows.label == 1)
    assert len(positive) == 1
    assert windows.fog_fraction[positive[0]] == 1.0
    assert not windows.clean_normal[positive[0]]


def test_models_preserve_expected_shapes_and_finite_loss():
    predictor = ConditionalNormalPredictor(
        in_channels=3,
        horizon=8,
        hidden_channels=8,
        dilations=(1, 2),
        dropout=0.0,
    )
    context = torch.randn(4, 3, 16)
    target = torch.randn(4, 3, 8)
    mean, logvar = predictor(context)
    assert mean.shape == target.shape
    assert logvar.shape == target.shape
    assert torch.isfinite(gaussian_nll(target, mean, logvar))

    classifier = ResidualTCNClassifier(
        in_channels=3,
        hidden_channels=8,
        dilations=(1, 2),
        dropout=0.0,
    )
    logits = classifier(target)
    assert logits.shape == (4,)


def test_single_class_metrics_are_explicitly_undefined():
    metrics = binary_metrics(np.zeros(5), np.linspace(0.1, 0.5, 5), threshold=0.5)
    assert metrics["sensitivity"] is None
    assert metrics["f1"] is None
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None


def test_threshold_uses_validation_balanced_accuracy():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    y_prob = np.array([0.1, 0.2, 0.4, 0.45, 0.7, 0.9])
    threshold, metrics = choose_threshold(y_true, y_prob)
    assert 0.4 < threshold <= 0.45
    assert metrics["balanced_accuracy"] == 1.0


def make_window_table(n: int, record_index: np.ndarray | None = None) -> WindowTable:
    target_start = np.arange(n, dtype=np.int32) * 16
    if record_index is None:
        record_index = np.zeros(n, dtype=np.int32)
    return WindowTable(
        record_index=np.asarray(record_index, dtype=np.int32),
        start=target_start - 128,
        target_start=target_start,
        target_end=target_start + 32,
        label=(np.arange(n) % 3 == 0).astype(np.int8),
        fog_fraction=(np.arange(n) % 3 == 0).astype(np.float32),
        clean_normal=np.ones(n, dtype=bool),
    )


def test_nonoverlap_history_uses_horizon_spaced_blocks_and_common_anchors():
    windows = make_window_table(20)
    indices = np.arange(20, dtype=np.int64)
    plan = make_common_history_plan(windows, indices, 32, 16, 256)
    assert plan.anchor_window_indices.tolist() == list(range(14, 20))
    assert plan.max_chain_rows[0].tolist() == [0, 2, 4, 6, 8, 10, 12, 14]

    blocks = np.repeat(np.arange(20, dtype=np.float32)[:, None, None], 3 * 32, axis=2)
    blocks = blocks.reshape(20, 3, 32)
    half_second = materialize_nonoverlap_residual_history(blocks, plan, 32, 32, 16)
    one_second = materialize_nonoverlap_residual_history(blocks, plan, 64, 32, 16)
    four_seconds = materialize_nonoverlap_residual_history(blocks, plan, 256, 32, 16)
    assert half_second.shape == (6, 3, 32)
    assert np.all(half_second[0] == 14)
    assert np.all(one_second[0, :, :32] == 12)
    assert np.all(one_second[0, :, 32:] == 14)
    for block_position, source_row in enumerate(range(0, 15, 2)):
        section = four_seconds[0, :, block_position * 32 : (block_position + 1) * 32]
        assert np.all(section == source_row)

    extracted = {
        "residual": blocks,
        "y": windows.label.copy(),
        "window_index": indices,
    }
    built = make_history_input(extracted, plan, "residual_h2s", 128, 32, 16)
    assert np.array_equal(built["window_index"], plan.anchor_window_indices)
    assert np.array_equal(built["y"], windows.label[plan.anchor_window_indices])

    raw_blocks = blocks + 100.0
    extracted["raw"] = raw_blocks
    raw_built = make_block_history_input(
        extracted,
        plan,
        "raw",
        "raw_h2s",
        128,
        32,
        16,
    )
    assert raw_built["raw_h2s"].shape == (6, 3, 128)
    assert np.all(raw_built["raw_h2s"][0, :, :32] == 108)
    assert np.all(raw_built["raw_h2s"][0, :, -32:] == 114)
    assert np.array_equal(raw_built["window_index"], plan.anchor_window_indices)
    assert np.array_equal(raw_built["y"], built["y"])


def test_history_plan_never_crosses_records_or_missing_blocks():
    record_index = np.r_[np.zeros(15, dtype=np.int32), np.ones(15, dtype=np.int32)]
    windows = make_window_table(30, record_index=record_index)
    # Restart local target time for the second record, as in real record-local windows.
    windows.target_start[15:] = np.arange(15, dtype=np.int32) * 16
    windows.target_end[15:] = windows.target_start[15:] + 32
    plan = make_common_history_plan(windows, np.arange(30), 32, 16, 256)
    assert plan.anchor_window_indices.tolist() == [14, 29]
    for chain in plan.max_chain_rows:
        records = windows.record_index[chain]
        assert np.unique(records).size == 1

    kept = np.delete(np.arange(30), 2)
    gap_plan = make_common_history_plan(windows, kept, 32, 16, 256)
    assert 14 not in gap_plan.anchor_window_indices
    assert 29 in gap_plan.anchor_window_indices


def test_history_parameter_validation_is_explicit():
    assert history_block_count(256, 32, 16) == 8
    for invalid in (16, 48):
        try:
            history_block_count(invalid, 32, 16)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid history length: {invalid}")
