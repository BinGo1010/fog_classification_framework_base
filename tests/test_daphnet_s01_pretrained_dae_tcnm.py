from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_s01_gru_h200_tcnm as core
import run_daphnet_s01_pretrained_dae_tcnm as experiment


DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DAE_DIR = REPO_ROOT / "outputs" / "daphnet_s01_dae_only_max200_seed42"


def test_completed_dae_artifact_loads_as_frozen_best_epoch() -> None:
    if not (DAE_DIR / "DONE.json").is_file():
        pytest.skip("completed DAE-only artifact is unavailable")
    model, scaler, config, training, done = experiment.load_frozen_dae(
        DAE_DIR, torch.device("cpu")
    )
    assert done["scope"] == "dae_only"
    assert config["execution_scope"] == "dae_only"
    assert training["best_epoch"] == 120
    assert training["epochs_completed"] == 135
    assert training["stopped_early"] is True
    assert model.training is False
    assert scaler.mean.shape == scaler.std.shape == (9,)
    assert np.all(scaler.std > 0)


def test_source_and_current_s01_splits_are_identical() -> None:
    if not (DATA_DIR / "manifest.csv").is_file() or not DAE_DIR.is_dir():
        pytest.skip("S01 data or DAE-only artifact is unavailable")
    dataset = core.load_s01_dataset(DATA_DIR)
    base = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base)
    split = core.make_split(dataset, windows)
    normal_train = core.normal_support_indices(
        dataset, windows, "train", split.train
    )
    normal_validation = core.normal_support_indices(
        dataset, windows, "validation", split.validation
    )
    experiment.verify_source_split(
        DAE_DIR, split, normal_train, normal_validation
    )
    assert [len(split.train), len(split.validation), len(split.test)] == [
        1090,
        351,
        447,
    ]
    assert [len(normal_train), len(normal_validation)] == [978, 295]


def test_classifier_version_is_restored_after_failure(monkeypatch) -> None:
    original = core.EXPERIMENT_VERSION

    def fail(*_args, **_kwargs):
        assert core.EXPERIMENT_VERSION == experiment.EXPERIMENT_VERSION
        raise RuntimeError("expected")

    monkeypatch.setattr(core, "train_classifier", fail)
    with pytest.raises(RuntimeError, match="expected"):
        experiment.train_classifier(
            object(), {}, object(), object(), Path("unused"), "fingerprint", object()
        )
    assert core.EXPERIMENT_VERSION == original

