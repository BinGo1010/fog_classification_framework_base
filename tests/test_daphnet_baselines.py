from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from cnbr_fog.data import (
    DaphnetDataset,
    Record,
    RobustChannelScaler,
    WindowTable,
)
from cnbr_fog.histories import make_common_history_plan
from daphnet_baselines import (
    CNNGRUClassifier,
    HistoryWindowDataset,
    TimeFrequencyFeatureExtractor,
    freeze_index_features,
    load_dataset,
    materialize_history_windows,
    resolve_sensor_channel_indices,
)
from scripts.run_cnbr_fog_loso import event_metrics


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


def test_manifest_adapter_supports_private_subjects(tmp_path) -> None:
    record_root = tmp_path / "records"
    record_root.mkdir()
    channel_names = (
        "waist_acc_forward",
        "waist_acc_vertical",
        "waist_acc_lateral",
    )
    with (tmp_path / "schema.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "channels": [
                    {"name": name, "unit": "g"} for name in channel_names
                ]
            },
            handle,
        )
    rows = []
    for index, subject in enumerate(("P01", "P02", "P03")):
        time = np.arange(768, dtype=np.float32)
        x = np.stack(
            [
                1.0 + 0.01 * time + index,
                2.0 + np.sin(time / 5.0),
                3.0 + np.cos(time / 7.0),
            ],
            axis=1,
        ).astype(np.float32)
        y = np.zeros(768, dtype=np.int8)
        y[400:496] = 1
        relative = f"records/{subject}.npz"
        np.savez_compressed(tmp_path / relative, x=x, y_binary=y)
        rows.append(
            {
                "record_path": relative,
                "record_id": f"{subject}_record",
                "subject_id": subject,
                "run_id": "R01",
                "n_samples": len(x),
                "sampling_rate_hz": 64,
                "usable": "true",
            }
        )
    with (tmp_path / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_dataset(
        "manifest_npz",
        tmp_path,
        excluded_subjects=(),
        flatline_seconds=1.0,
        zero_tolerance=1e-8,
    )
    assert tuple(loaded.dataset.subjects) == ("P01", "P02", "P03")
    assert loaded.default_fi_channels == ("waist_acc_vertical",)
    assert resolve_sensor_channel_indices(
        "all",
        loaded.dataset.channel_names,
    ) == (0, 1, 2)
    windows = loaded.dataset.make_windows(
        warmup_samples=128,
        target_samples=32,
        stride_samples=16,
        fog_fraction_threshold=0.5,
        normal_guard_samples=32,
    )
    plan = make_common_history_plan(
        windows,
        np.arange(len(windows), dtype=np.int64),
        horizon_samples=32,
        stride_samples=16,
        max_history_samples=256,
    )
    for subject in loaded.dataset.subjects:
        indices = loaded.dataset.window_indices_for_subjects(
            windows,
            [subject],
        )
        eligible = np.intersect1d(
            indices,
            plan.anchor_window_indices,
            assume_unique=True,
        )
        assert set(windows.label[eligible]) == {0, 1}
    with pytest.raises(ValueError, match="unavailable"):
        resolve_sensor_channel_indices(
            "ankle",
            loaded.dataset.channel_names,
        )


def test_event_metrics_use_evaluated_nonfog_coverage_and_split_gaps() -> None:
    record = Record(
        record_id="record",
        subject_id="S01",
        run_id="R01",
        x=np.ones((40, 3), dtype=np.float32),
        y=np.zeros(40, dtype=np.int8),
        valid=np.ones(40, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=".",
        records=[record],
        sampling_rate_hz=4,
        channel_names=("a", "b", "c"),
    )
    windows = WindowTable(
        record_index=np.zeros(4, dtype=np.int32),
        start=np.asarray([0, 1, 10, 11], dtype=np.int32),
        target_start=np.asarray([0, 1, 10, 11], dtype=np.int32),
        target_end=np.asarray([2, 3, 12, 13], dtype=np.int32),
        label=np.zeros(4, dtype=np.int8),
        fog_fraction=np.zeros(4, dtype=np.float32),
        clean_normal=np.ones(4, dtype=bool),
    )
    metrics = event_metrics(
        dataset,
        windows,
        np.arange(4, dtype=np.int64),
        np.ones(4, dtype=np.int8),
        minimum_positive_windows=2,
        merge_gap_seconds=0.5,
    )
    evaluated_seconds = 6.0 / 4.0
    assert metrics["event_metric_version"] == "coverage_aware.v2"
    assert metrics["predicted_events"] == 2
    assert metrics["false_alarm_events"] == 2
    assert metrics["evaluated_nonfog_hours"] == pytest.approx(
        evaluated_seconds / 3600.0
    )
    assert metrics["false_alarm_events_per_hour"] == pytest.approx(
        2.0 / (evaluated_seconds / 3600.0)
    )
