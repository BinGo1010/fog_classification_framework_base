from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_full_subject_nbm_residual_binary",
    SCRIPTS / "run_daphnet_full_subject_nbm_residual_binary.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_representations_have_frozen_shapes_and_delta_origin() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 128, 9)).astype(np.float32)
    reconstruction = rng.normal(size=(5, 128, 9)).astype(np.float32)
    values = runner.representation_arrays(x, reconstruction)
    assert {key: value.shape for key, value in values.items()} == {
        "B0": (5, 128, 9), "B1": (5, 128, 9),
        "B2": (5, 128, 27), "B3": (5, 128, 36),
    }
    np.testing.assert_allclose(values["B2"][:, 0, 18:27], 0.0, atol=1e-7)


def test_tcn_only_differs_in_input_channel_projection() -> None:
    parameter_counts = {}
    for method, channels in runner.METHOD_CHANNELS.items():
        model = runner.FixedTCNClassifier(channels)
        output = model(torch.randn(3, channels, 128))
        assert output.shape == (3,)
        parameter_counts[method] = sum(parameter.numel() for parameter in model.parameters())
        config = model.architecture_config()
        assert config["dilations"] == [1, 2, 4, 8]
    assert parameter_counts["B0"] == parameter_counts["B1"]
    assert parameter_counts["B2"] - parameter_counts["B1"] == (27 - 9) * 64 * 5
    assert parameter_counts["B3"] - parameter_counts["B2"] == (36 - 27) * 64 * 5


def test_threshold_selection_maximizes_positive_f1() -> None:
    truth = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.4, 0.45, 0.9])
    threshold = runner.select_threshold(truth, probability)
    prediction = probability >= threshold
    assert runner.f1_score(truth, prediction) == 1.0


def test_event_metrics_detects_events_and_false_alarm_episode() -> None:
    rows = []
    truth = [0, 1, 1, 0, 0, 1]
    prediction = [1, 0, 1, 0, 0, 1]
    for index, (actual, predicted) in enumerate(zip(truth, prediction)):
        rows.append({"record_id": "r0", "window_start": index * 64,
                     "y_true": actual, "y_pred": predicted})
    metrics = runner.event_metrics(rows)
    assert metrics["total_events"] == 2
    assert metrics["detected_events"] == 2
    assert metrics["false_alarm_episodes"] == 1
    assert metrics["median_detection_latency_seconds"] == 0.5


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = [0.01, 0.04, 0.03, 0.2]
    adjusted = runner.holm_adjust(raw)
    order = np.argsort(raw)
    sorted_adjusted = np.asarray(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= -1e-12)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
