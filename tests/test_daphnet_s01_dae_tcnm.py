from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_s01_dae_tcnm as experiment
import run_daphnet_s01_gru_h200_tcnm as core


DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)


def _prepared():
    if not (DATA_DIR / "manifest.csv").exists():
        pytest.skip("Daphnet processed data are unavailable")
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
    return dataset, windows, split


def test_dae_uses_exact_baseline_windows_and_clean_normal_support() -> None:
    dataset, windows, split = _prepared()
    assert [len(split.train), len(split.validation), len(split.test)] == [
        1090,
        351,
        447,
    ]
    normal_train = core.normal_support_indices(
        dataset, windows, "train", split.train
    )
    normal_validation = core.normal_support_indices(
        dataset, windows, "validation", split.validation
    )
    assert [len(normal_train), len(normal_validation)] == [978, 295]

    raw = experiment.extract_target_windows(dataset, windows, normal_train)
    assert raw.shape == (978, 9, 128)
    scaler = experiment.ChannelZScoreScaler.fit_channel_time(raw)
    standardized = scaler.transform_channel_time(raw)
    np.testing.assert_allclose(standardized.mean(axis=(0, 2)), 0, atol=2e-5)
    np.testing.assert_allclose(standardized.std(axis=(0, 2)), 1, atol=2e-5)


def test_fixed_sigma_adapter_is_channel_by_time_and_train_only() -> None:
    rng = np.random.default_rng(3)
    errors = rng.normal(size=(978, 9, 128)).astype(np.float32)
    sigma = experiment.calibrate_fixed_sigma(
        errors, epsilon=experiment.FIXED_SIGMA_EPSILON
    )
    assert sigma.shape == (1, 9, 128)
    assert np.isfinite(sigma).all()
    assert np.all(sigma > 0)

    target = np.asarray([[[3.0, -30.0]]], dtype=np.float32)
    reconstruction = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    known_sigma = np.asarray([[[2.0, 2.0]]], dtype=np.float32)
    residual = np.clip(
        (target - reconstruction) / known_sigma,
        -experiment.RESIDUAL_CLIP,
        experiment.RESIDUAL_CLIP,
    )
    np.testing.assert_array_equal(
        residual, np.asarray([[[1.0, -12.0]]], dtype=np.float32)
    )


def test_classifier_version_is_restored_if_reused_core_raises(monkeypatch) -> None:
    original = core.EXPERIMENT_VERSION

    def fail(*_args, **_kwargs):
        assert core.EXPERIMENT_VERSION == experiment.EXPERIMENT_VERSION
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(core, "train_classifier", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        experiment.run_classifier(
            object(), {}, object(), object(), Path("unused"), "fingerprint", object()
        )
    assert core.EXPERIMENT_VERSION == original
