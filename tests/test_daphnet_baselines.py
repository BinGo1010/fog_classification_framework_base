from __future__ import annotations

import numpy as np
import pytest
import torch

from cnbr_fog.data import Record, RobustChannelScaler, WindowTable
from daphnet_baselines import (
    CNNGRUClassifier,
    HistoryWindowDataset,
    TimeFrequencyFeatureExtractor,
    freeze_index_features,
    materialize_history_windows,
)


FS = 64
SAMPLES = 256


def sine(frequency_hz: float, amplitude: float = 1.0) -> np.ndarray:
    time = np.arange(SAMPLES, dtype=np.float64) / FS
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def test_freeze_index_separates_locomotor_and_freeze_bands() -> None:
    windows = np.stack([sine(1.0), sine(5.0)], axis=0)[:, None, :]
    features = freeze_index_features(windows, FS, 0)
    assert features["score"][0] < 1e-10
    assert features["score"][1] > 1.0 - 1e-10
    assert np.isfinite(features["freeze_index"]).all()


def test_freeze_index_three_hz_belongs_only_to_freeze_band() -> None:
    window = sine(3.0)[None, None, :]
    features = freeze_index_features(window, FS, 0)
    assert features["locomotor_power"][0] < 1e-10
    assert features["freeze_power"][0] > 1.0
    assert features["score"][0] > 1.0 - 1e-10


def test_freeze_index_constant_is_finite_zero_and_scale_invariant() -> None:
    constant = np.full((2, 3, SAMPLES), 7.0, dtype=np.float64)
    inactive = freeze_index_features(constant, FS, [0, 1, 2])
    assert np.array_equal(inactive["score"], np.zeros(2))
    mixed = (sine(1.0) + 0.5 * sine(5.0))[None, None, :]
    base = freeze_index_features(mixed, FS, 0)
    scaled = freeze_index_features(9.0 * mixed, FS, 0)
    np.testing.assert_allclose(base["score"], scaled["score"], atol=1e-12)
    np.testing.assert_allclose(
        scaled["total_power"],
        81.0 * base["total_power"],
        rtol=1e-12,
    )


def test_freeze_index_power_pool_is_axis_rotation_invariant() -> None:
    triad = np.stack(
        [sine(1.0), 0.5 * sine(5.0), 0.25 * sine(7.0)],
        axis=0,
    )
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    original = freeze_index_features(
        triad[None],
        FS,
        [0, 1, 2],
        aggregation="power_pool",
    )
    rotated = freeze_index_features(
        (rotation @ triad)[None],
        FS,
        [0, 1, 2],
        aggregation="power_pool",
    )
    np.testing.assert_allclose(original["score"], rotated["score"], atol=1e-12)


def test_time_frequency_schema_is_stable_and_finite() -> None:
    names = (
        "ankle_acc_forward",
        "ankle_acc_vertical",
        "ankle_acc_lateral",
        "thigh_acc_forward",
        "thigh_acc_vertical",
        "thigh_acc_lateral",
        "trunk_acc_forward",
        "trunk_acc_vertical",
        "trunk_acc_lateral",
    )
    extractor = TimeFrequencyFeatureExtractor(FS, names)
    values = np.zeros((3, 9, SAMPLES), dtype=np.float32)
    features = extractor.transform(values)
    assert features.shape == (3, 306)
    assert len(extractor.feature_names()) == 306
    assert len(set(extractor.feature_names())) == 306
    assert np.isfinite(features).all()


def _synthetic_history_inputs() -> tuple[list[Record], WindowTable]:
    x = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    record = Record(
        record_id="record",
        subject_id="S01",
        run_id="R01",
        x=x,
        y=np.zeros(40, dtype=np.int8),
        valid=np.ones(40, dtype=bool),
    )
    windows = WindowTable(
        record_index=np.asarray([0, 0], dtype=np.int32),
        start=np.asarray([0, 8], dtype=np.int32),
        target_start=np.asarray([8, 16], dtype=np.int32),
        target_end=np.asarray([16, 24], dtype=np.int32),
        label=np.asarray([0, 1], dtype=np.int8),
        fog_fraction=np.asarray([0.0, 1.0], dtype=np.float32),
        clean_normal=np.asarray([True, False]),
    )
    return [record], windows


def test_history_materialization_is_causal_and_label_aligned() -> None:
    records, windows = _synthetic_history_inputs()
    values, labels, indices = materialize_history_windows(
        records,
        windows,
        [1],
        history_samples=12,
    )
    assert values.shape == (1, 3, 12)
    np.testing.assert_array_equal(values[0].T, records[0].x[12:24])
    np.testing.assert_array_equal(labels, [1])
    np.testing.assert_array_equal(indices, [1])


def test_history_helpers_reject_invalid_indices() -> None:
    records, windows = _synthetic_history_inputs()
    scaler = RobustChannelScaler(
        center=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
    )
    with pytest.raises(IndexError):
        materialize_history_windows(records, windows, [-1], 8)
    with pytest.raises(IndexError):
        HistoryWindowDataset(records, windows, [2], 8, scaler)


def test_cnn_gru_output_shape_and_gradient() -> None:
    model = CNNGRUClassifier(
        in_channels=9,
        cnn_channels=(8, 16),
        gru_hidden=12,
    )
    batch = torch.randn(4, 9, SAMPLES)
    logits = model(batch)
    assert logits.shape == (4,)
    logits.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
