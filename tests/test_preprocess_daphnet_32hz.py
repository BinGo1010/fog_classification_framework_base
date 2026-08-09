from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preprocess_daphnet_32hz as preprocessing


def test_fir_is_65_tap_symmetric_unity_gain() -> None:
    coefficients = preprocessing.design_fir()

    assert coefficients.shape == (65,)
    assert np.array_equal(coefficients, coefficients[::-1])
    assert float(np.sum(coefficients)) == pytest.approx(1.0, abs=1e-12)
    assert preprocessing.GROUP_DELAY_SAMPLES == 32
    assert preprocessing.MIRROR_PAD_SAMPLES == 32


def test_constant_signal_remains_constant_after_filter_and_resample() -> None:
    coefficients = preprocessing.design_fir()
    x = np.full((257, 9), 2.5, dtype=np.float32)
    y = np.zeros(257, dtype=np.int8)

    x_32hz, y_32hz, audit = preprocessing.preprocess_record(x, y, coefficients)

    assert x_32hz.shape == (129, 9)
    assert y_32hz.shape == (129,)
    assert np.allclose(x_32hz, 2.5, rtol=0.0, atol=1e-6)
    assert np.count_nonzero(y_32hz) == 0
    assert audit["endpoint_truncation_sec"] == pytest.approx(0.0)


def test_nearest_labels_use_even_64hz_samples_on_aligned_grid() -> None:
    labels = np.asarray([0, 1, 1, 0, 0, 1, 1], dtype=np.int8)

    output, nearest = preprocessing.resample_labels_nearest_32hz(labels)

    assert nearest.tolist() == [0, 2, 4, 6]
    assert output.tolist() == [0, 1, 0, 1]


def test_delay_compensation_keeps_centered_impulse_aligned() -> None:
    coefficients = preprocessing.design_fir()
    x = np.zeros((257, 9), dtype=np.float32)
    x[128, :] = 1.0

    aligned = preprocessing.filter_and_align_64hz(x, coefficients)
    downsampled, positions = preprocessing.resample_acceleration_32hz(aligned)

    assert int(np.argmax(aligned[:, 0])) == 128
    assert int(np.argmax(downsampled[:, 0])) == 64
    assert positions[64] == pytest.approx(128.0)
