from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_daphnet_s01_gru_convergence as diagnostic
import diagnose_daphnet_s01_gru_mean_only as mean_only


DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)


def _prepared_support():
    if not (DATA_DIR / "manifest.csv").exists():
        pytest.skip("Daphnet processed data are not available")
    return diagnostic.prepare_support(DATA_DIR)


def test_loader_never_opens_held_out_r02_array(monkeypatch) -> None:
    if not (DATA_DIR / "manifest.csv").exists():
        pytest.skip("Daphnet processed data are not available")
    opened: list[str] = []
    original_load = diagnostic.np.load

    def tracking_load(path, *args, **kwargs):
        opened.append(str(Path(path).resolve()))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(diagnostic.np, "load", tracking_load)
    dataset = diagnostic.load_train_validation_dataset(DATA_DIR)

    assert [record.record_id for record in dataset.records] == [
        diagnostic.base.TRAIN_RECORD,
        diagnostic.base.TRAIN_VALIDATION_CUT_RECORD,
    ]
    assert len(opened) == 2
    assert all("seg002" not in path.lower() for path in opened)


def test_frozen_clean_support_and_model_capacity() -> None:
    dataset, _, train, validation, scaler, metadata = _prepared_support()
    assert [record.record_id for record in dataset.records] == [
        "S01_seg000",
        "S01_seg001",
    ]
    assert len(train) == 978
    assert len(validation) == 295
    assert metadata["scaler"]["fit_points"] == 67_135
    assert scaler.center.shape == (9,)
    model = diagnostic.GRUNBM(
        in_channels=9,
        horizon=128,
        hidden_channels=48,
        num_layers=1,
        dropout=0.1,
    )
    assert diagnostic.parameter_count(model) == 123_744


def test_duration_fraction_subsets_are_nested_and_measure_raw_time() -> None:
    dataset, windows, train, _, _, _ = _prepared_support()
    subsets, audit = diagnostic.duration_fraction_subsets(
        dataset,
        windows,
        train,
        diagnostic.DEFAULT_FRACTIONS,
    )

    expected_seconds = {0.25: 267.0, 0.5: 495.0, 0.75: 767.0, 1.0: 1020.0}
    previous: set[int] = set()
    for fraction in diagnostic.DEFAULT_FRACTIONS:
        selected = set(map(int, subsets[fraction]))
        assert previous.issubset(selected)
        previous = selected
        arm = audit["fractions"][f"{fraction:g}"]
        assert arm["unique_raw_support_seconds"] == expected_seconds[fraction]
        assert abs(arm["achieved_full_duration_fraction"] - fraction) <= 0.03
        assert set(arm["windows_by_record"]) == {"S01_seg000", "S01_seg001"}
    assert np.array_equal(subsets[1.0], train)


def test_matched_exposure_equalizes_steps_without_adding_new_windows() -> None:
    dataset, windows, train, _, _, _ = _prepared_support()
    subsets, _ = diagnostic.duration_fraction_subsets(
        dataset,
        windows,
        train,
        diagnostic.DEFAULT_FRACTIONS,
    )
    for fraction, selected in subsets.items():
        exposure_a = diagnostic.matched_epoch_exposure(selected, len(train), 123)
        exposure_b = diagnostic.matched_epoch_exposure(selected, len(train), 123)
        assert len(exposure_a) == len(train) == 978
        assert np.array_equal(exposure_a, exposure_b)
        assert set(map(int, exposure_a)).issubset(set(map(int, selected)))
        assert int(np.ceil(len(exposure_a) / 256)) == 4
        if fraction < 1.0:
            assert len(np.unique(exposure_a)) < len(exposure_a)
        else:
            assert np.array_equal(exposure_a, selected)


def test_minimum_epoch_guarantees_epoch8_diagnostic() -> None:
    assert diagnostic.MIN_EPOCHS == 8
    assert diagnostic.MIN_DELTA == pytest.approx(1e-4)
    assert diagnostic.DEFAULT_SEEDS == (42, 43, 44, 45, 46)
    assert diagnostic.DEFAULT_FRACTIONS == (0.25, 0.5, 0.75, 1.0)


def test_mean_only_ablation_evaluates_the_same_clean_support() -> None:
    dataset, windows, train, validation, scaler, _ = _prepared_support()
    model = diagnostic.GRUNBM(
        in_channels=9,
        horizon=128,
        hidden_channels=48,
        num_layers=1,
        dropout=0.1,
    )
    metrics = mean_only.evaluate_mean(
        model,
        dataset,
        windows,
        validation[:3],
        scaler,
        batch_size=3,
        device=diagnostic.torch.device("cpu"),
    )
    assert metrics["windows"] == 3
    assert np.isfinite(metrics["rmse_scaled"])
    assert metrics["per_channel_rmse_scaled"] and len(
        metrics["per_horizon_rmse_scaled"]
    ) == 128
    assert len(train) == 978 and len(validation) == 295
