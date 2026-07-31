from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cnbr_fog.data import DaphnetDataset, Record, RobustChannelScaler, WindowTable
from cnbr_fog.h200_crossfit import (
    assemble_oof_gaussians,
    audit_crossfit_provenance,
    convert_to_outer_scaler_primitives,
    ensemble_gaussians,
    extract_gaussian_forecasts,
    temporal_clean_normal_split,
)


SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07")


def _toy_dataset_and_windows() -> tuple[DaphnetDataset, WindowTable, dict[str, np.ndarray]]:
    records: list[Record] = []
    record_index: list[int] = []
    starts: list[int] = []
    target_starts: list[int] = []
    target_ends: list[int] = []
    labels: list[int] = []
    subject_windows: dict[str, np.ndarray] = {}
    cursor = 0
    for subject_index, subject in enumerate(SUBJECTS):
        time = np.arange(80, dtype=np.float32)
        x = np.stack(
            [
                10.0 + 2.0 * time + subject_index,
                20.0 + 4.0 * (time + 0.5) + subject_index,
            ],
            axis=1,
        ).astype(np.float32)
        records.append(
            Record(
                record_id=f"{subject}_R01",
                subject_id=subject,
                run_id="R01",
                x=x,
                y=np.zeros(len(x), dtype=np.int8),
                valid=np.ones(len(x), dtype=bool),
            )
        )
        local_indices: list[int] = []
        for local, start in enumerate(range(0, 26, 2)):
            local_indices.append(cursor)
            record_index.append(subject_index)
            starts.append(start)
            target_starts.append(start + 4)
            target_ends.append(start + 6)
            labels.append(0)
            cursor += 1
        subject_windows[subject] = np.asarray(local_indices, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int8)
    windows = WindowTable(
        record_index=np.asarray(record_index, dtype=np.int32),
        start=np.asarray(starts, dtype=np.int32),
        target_start=np.asarray(target_starts, dtype=np.int32),
        target_end=np.asarray(target_ends, dtype=np.int32),
        label=label_array,
        fog_fraction=label_array.astype(np.float32),
        clean_normal=np.ones(len(label_array), dtype=bool),
    )
    dataset = DaphnetDataset(
        root=Path("."),
        records=records,
        sampling_rate_hz=64,
        channel_names=("c0", "c1"),
    )
    return dataset, windows, subject_windows


class RepeatLastGaussian(torch.nn.Module):
    def __init__(self, horizon: int = 2) -> None:
        super().__init__()
        self.horizon = horizon
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = context[:, :, -1:].repeat(1, 1, self.horizon)
        sigma = torch.full_like(mean, 0.5) + self.anchor * 0.0
        return mean, sigma


def _physical_forecast(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    *,
    predictor_id: str,
    train_subjects: tuple[str, ...],
    heldout_subjects: tuple[str, ...],
    mean_value: float = 0.0,
) -> dict:
    targets = []
    for index in indices:
        record = dataset.records[int(windows.record_index[index])]
        targets.append(
            record.x[
                int(windows.target_start[index]) : int(windows.target_end[index])
            ].T
        )
    target = np.ascontiguousarray(np.stack(targets), dtype=np.float32)
    return {
        "target": target,
        "mu": np.full_like(target, mean_value, dtype=np.float32),
        "sigma": np.ones_like(target, dtype=np.float32),
        "y": np.asarray(windows.label[indices], dtype=np.int8),
        "window_index": np.asarray(indices, dtype=np.int64),
        "provenance": {
            "predictor_id": predictor_id,
            "predictor_train_subjects": list(train_subjects),
            "scaler_fit_subjects": list(train_subjects),
            "heldout_subjects": list(heldout_subjects),
        },
    }


def test_temporal_clean_normal_split_uses_record_tail_with_raw_embargo() -> None:
    dataset, windows, subject_windows = _toy_dataset_and_windows()
    selected = np.concatenate([subject_windows["S01"], subject_windows["S02"]])
    split = temporal_clean_normal_split(
        dataset,
        windows,
        ("S01", "S02"),
        candidate_indices=selected,
        validation_fraction=0.25,
    )

    train = split["train_window_index"]
    validation = split["validation_window_index"]
    assert len(train) > 0 and len(validation) > 0
    assert set(train).isdisjoint(set(validation))
    assert split["raw_support_overlap"] is False
    assert {row["subject_id"] for row in split["records"]} == {"S01", "S02"}
    for record_index in np.unique(windows.record_index[np.r_[train, validation]]):
        record_train = train[windows.record_index[train] == record_index]
        record_validation = validation[
            windows.record_index[validation] == record_index
        ]
        assert windows.target_end[record_train].max() <= windows.start[
            record_validation
        ].min()
        # Validation is the clean-normal chronological tail of the record.
        assert windows.target_end[record_validation].min() > windows.target_end[
            record_train
        ].max()


def test_temporal_split_rejects_nonclean_candidate() -> None:
    dataset, windows, subject_windows = _toy_dataset_and_windows()
    clean = windows.clean_normal.copy()
    clean[subject_windows["S01"][0]] = False
    changed = WindowTable(
        record_index=windows.record_index,
        start=windows.start,
        target_start=windows.target_start,
        target_end=windows.target_end,
        label=windows.label,
        fog_fraction=windows.fog_fraction,
        clean_normal=clean,
    )
    with pytest.raises(ValueError, match="clean-normal"):
        temporal_clean_normal_split(
            dataset,
            changed,
            ("S01",),
            candidate_indices=subject_windows["S01"],
        )


def test_extract_gaussian_forecasts_returns_physical_units_and_metadata() -> None:
    dataset, windows, subject_windows = _toy_dataset_and_windows()
    scaler = RobustChannelScaler(
        center=np.asarray([10.0, 20.0], dtype=np.float32),
        scale=np.asarray([2.0, 4.0], dtype=np.float32),
        clip=100.0,
    )
    indices = subject_windows["S01"][:2]
    model = RepeatLastGaussian()
    model.train()
    result = extract_gaussian_forecasts(
        model,
        dataset,
        windows,
        indices,
        scaler,
        batch_size=1,
        device="cpu",
        predictor_id="p0",
        predictor_train_subjects=("S02", "S03", "S05", "S06"),
        scaler_fit_subjects=("S02", "S03", "S05", "S06"),
        heldout_subjects=("S01", "S07"),
    )

    assert model.training is True
    assert result["target"].shape == (2, 2, 2)
    np.testing.assert_array_equal(result["window_index"], indices)
    np.testing.assert_array_equal(result["y"], windows.label[indices])
    # First context ends at t=3.  The model repeats that scaled sample, which
    # converts back to the physical values at t=3.
    np.testing.assert_allclose(result["mu"][0, :, 0], [16.0, 34.0])
    np.testing.assert_allclose(result["mu"][0, :, 1], [16.0, 34.0])
    np.testing.assert_allclose(result["sigma"][0, :, 0], [1.0, 2.0])
    expected_target = dataset.records[0].x[4:6].T
    np.testing.assert_allclose(result["target"][0], expected_target)
    assert result["provenance"]["predictor_id"] == "p0"


def test_assemble_oof_gaussians_enforces_unique_unseen_subject_ownership() -> None:
    dataset, windows, subject_windows = _toy_dataset_and_windows()
    expected = np.asarray(
        [subject_windows[subject][0] for subject in reversed(SUBJECTS)],
        dtype=np.int64,
    )
    heldout_groups = (("S01", "S02"), ("S03", "S05"), ("S06", "S07"))
    forecasts = []
    for fold, heldout in enumerate(heldout_groups):
        train = tuple(subject for subject in SUBJECTS if subject not in heldout)
        indices = np.asarray(
            [subject_windows[subject][0] for subject in reversed(heldout)],
            dtype=np.int64,
        )
        forecasts.append(
            _physical_forecast(
                dataset,
                windows,
                indices,
                predictor_id=f"inner_{fold}",
                train_subjects=train,
                heldout_subjects=heldout,
                mean_value=float(fold),
            )
        )

    result = assemble_oof_gaussians(
        forecasts,
        dataset,
        windows,
        expected,
        outer_train_subjects=SUBJECTS,
        validation_subjects=("S08",),
        test_subjects=("S09",),
        scheme="3fold",
    )
    np.testing.assert_array_equal(result["window_index"], expected)
    np.testing.assert_array_equal(result["y"], windows.label[expected])
    assert result["provenance_audit"]["status"] == "pass"
    assert len(np.unique(result["source_predictor_id"])) == 3

    tampered = [dict(forecast) for forecast in forecasts]
    tampered[0] = dict(tampered[0])
    tampered[0]["provenance"] = dict(tampered[0]["provenance"])
    tampered[0]["provenance"]["scaler_fit_subjects"] = [
        "S01",
        "S03",
        "S05",
        "S06",
    ]
    report = audit_crossfit_provenance(
        tampered,
        outer_train_subjects=SUBJECTS,
        dataset=dataset,
        windows=windows,
        expected_window_indices=expected,
        scheme="3fold",
    )
    assert report["status"] == "fail"
    assert any("scaler subjects differ" in failure for failure in report["failures"])
    with pytest.raises(ValueError, match="provenance audit failed"):
        assemble_oof_gaussians(
            tampered,
            dataset,
            windows,
            expected,
            outer_train_subjects=SUBJECTS,
            scheme="3fold",
        )


def test_ensemble_gaussians_aligns_endpoints_and_matches_physical_moments() -> None:
    target = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    first = {
        "target": target,
        "mu": np.full_like(target, 1.0),
        "sigma": np.full_like(target, 2.0),
        "y": np.asarray([0, 1], dtype=np.int8),
        "window_index": np.asarray([10, 11], dtype=np.int64),
    }
    second = {
        "target": target[::-1].copy(),
        "mu": np.full_like(target, 3.0)[::-1].copy(),
        "sigma": np.full_like(target, 4.0)[::-1].copy(),
        "y": np.asarray([1, 0], dtype=np.int8),
        "window_index": np.asarray([11, 10], dtype=np.int64),
    }
    result = ensemble_gaussians(
        [first, second], expected_window_indices=np.asarray([10, 11])
    )
    np.testing.assert_allclose(result["mu"], 2.0)
    # E[sigma^2 + mu^2] - E[mu]^2 = 15 - 4 = 11.
    np.testing.assert_allclose(result["sigma"], np.sqrt(11.0), rtol=1e-6)
    np.testing.assert_array_equal(result["target"], target)
    np.testing.assert_array_equal(result["y"], [0, 1])
    assert result["ensemble_size"] == 2

    bad = dict(second)
    bad["window_index"] = np.asarray([11, 12], dtype=np.int64)
    with pytest.raises(ValueError, match="endpoint set"):
        ensemble_gaussians([first, bad])


def test_convert_to_outer_scaler_primitives_reports_raw_and_z_clipping() -> None:
    forecast = {
        "target": np.asarray([[[14.0, 8.0], [20.0, 28.0]]], dtype=np.float32),
        "mu": np.asarray([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32),
        "sigma": np.asarray([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32),
        "y": np.asarray([1], dtype=np.int8),
        "window_index": np.asarray([7], dtype=np.int64),
    }
    scaler = RobustChannelScaler(
        center=np.asarray([10.0, 20.0], dtype=np.float32),
        scale=np.asarray([2.0, 4.0], dtype=np.float32),
        clip=1.0,
    )
    result = convert_to_outer_scaler_primitives(
        forecast, scaler, z_clip=1.0
    )
    np.testing.assert_allclose(result["raw"], [[[1.0, -1.0], [0.0, 1.0]]])
    np.testing.assert_allclose(result["mu"], 0.0)
    np.testing.assert_allclose(result["sigma"], 0.5)
    np.testing.assert_allclose(result["error"], result["raw"])
    np.testing.assert_allclose(result["z"], [[[1.0, -1.0], [0.0, 1.0]]])
    np.testing.assert_allclose(result["log_sigma"], np.log(0.5))
    assert result["diagnostics"]["raw_clip_rate"] == pytest.approx(0.5)
    assert result["diagnostics"]["z_clip_rate"] == pytest.approx(0.75)
    assert result["clip_rate"] == pytest.approx(0.75)
