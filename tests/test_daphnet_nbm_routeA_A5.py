from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_nbm_routeA_A5 as a5


def test_component_scores_follow_template_definitions() -> None:
    residual = np.zeros((2, 128, 9), dtype=np.float32)
    residual[0, :, 0] = 1.0
    residual[0, :, 1] = 2.0
    residual[0, :, 2] = 3.0
    scores = a5.component_scores(residual)
    assert scores.shape == (2, 3)
    assert scores[0, 0] == 0.0
    assert np.isclose(scores[0, 1], 2.0)
    assert scores[0, 2] > 0.0
    assert np.allclose(scores[1], 0.0)
    assert a5.component_scores(np.empty((0, 128, 9), dtype=np.float32)).shape == (0, 3)


def test_s3_scaling_and_simplex_grid_are_train_only_and_positive() -> None:
    train = np.asarray([[1.0, 2.0, 4.0], [3.0, 4.0, 8.0]])
    scale = a5.fit_component_scale(train)
    combined = a5.combine_components(train, scale, (0.2, 0.3, 0.5))
    assert np.all(scale > 0)
    assert np.all(combined > 0)
    grid = a5.simplex_weights(0.1)
    assert len(grid) == 66
    assert all(np.isclose(sum(weights), 1.0) for weights in grid)


def test_a5_gate_uses_one_second_false_alarm_rate() -> None:
    normal = np.asarray([1.0] * 95 + [3.0] * 5)
    fog = np.asarray([4.0] * 20)
    train = np.asarray([1.0] * 100)
    metrics = a5.separation_metrics(normal, fog, train)
    assert np.isclose(metrics["false_alarm_windows_per_minute"], 3.0)
    usable, strong = a5.run_gate(metrics)
    assert usable
    assert not strong
