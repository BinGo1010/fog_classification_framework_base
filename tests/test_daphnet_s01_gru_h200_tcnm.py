from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_s01_gru_h200_tcnm as experiment


DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)


def _prepared_protocol():
    if not (DATA_DIR / "manifest.csv").exists():
        pytest.skip("Daphnet processed data are not available")
    dataset = experiment.load_s01_dataset(DATA_DIR)
    base = dataset.make_windows(
        warmup_samples=experiment.CONTEXT_SAMPLES,
        target_samples=experiment.TARGET_SAMPLES,
        stride_samples=experiment.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=experiment.NORMAL_GUARD_SAMPLES,
    )
    windows = experiment.endpoint_relabel(dataset, base)
    split = experiment.make_split(dataset, windows)
    return dataset, windows, split


def test_s01_split_is_disjoint_and_has_expected_support() -> None:
    dataset, windows, split = _prepared_protocol()
    expected = {
        "train": (1090, [1042, 48], 978),
        "validation": (351, [316, 35], 295),
        "test": (447, [423, 24], 401),
    }
    for name, indices in split.as_dict().items():
        size, class_counts, normal_count = expected[name]
        assert len(indices) == size
        assert np.bincount(windows.label[indices], minlength=2).tolist() == class_counts
        normal = experiment.normal_support_indices(
            dataset, windows, name, indices
        )
        assert len(normal) == normal_count

    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)
    assert not set(split.validation) & set(split.test)
    lookup = experiment.record_lookup(dataset)
    cut_record = lookup[experiment.TRAIN_VALIDATION_CUT_RECORD]
    train_at_cut = split.train[windows.record_index[split.train] == cut_record]
    validation_at_cut = split.validation[
        windows.record_index[split.validation] == cut_record
    ]
    assert windows.target_end[train_at_cut].max() == 50_944
    assert windows.start[validation_at_cut].min() == 50_944


def test_s01_scaler_and_architectures_match_declared_protocol() -> None:
    dataset, windows, _ = _prepared_protocol()
    scaler, metadata = experiment.fit_training_scaler(dataset)
    assert metadata["fit_points"] == 67_135
    assert scaler.center.shape == (9,)
    assert scaler.scale.shape == (9,)
    assert np.all(scaler.scale > 0)
    assert scaler.clip == 12.0

    nbm = experiment.GRUNBM(
        in_channels=9,
        horizon=128,
        hidden_channels=48,
        num_layers=1,
        dropout=0.1,
    )
    classifier = experiment.build_rf125_classifier(
        "tcn_m",
        in_channels=9,
        input_samples=128,
        hidden_channels=48,
        dropout=0.15,
    )
    assert experiment.parameter_count(nbm) == 123_744
    architecture = classifier.architecture_config()
    assert architecture["parameter_count"] == 89_329
    assert architecture["dilations"] == [1, 2, 4, 8, 8, 8]
    assert architecture["local_receptive_field_samples"] == 125

    event_windows = experiment.event_scoring_windows(windows)
    assert np.all(
        event_windows.target_end - event_windows.target_start
        == experiment.STRIDE_SAMPLES
    )
    assert np.array_equal(event_windows.label, windows.label)
