from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preprocess_stanford_ngm_64hz as preprocessing


def test_fir_is_symmetric_and_has_unity_dc_gain() -> None:
    coefficients = preprocessing.design_fir()

    assert coefficients.shape == (65,)
    assert np.array_equal(coefficients, coefficients[::-1])
    assert float(np.sum(coefficients)) == pytest.approx(1.0, abs=1e-12)
    assert preprocessing.KAISER_BETA == 5.0
    assert preprocessing.FIR_CUTOFF_HZ == 28.0


def test_constant_signal_remains_constant_after_filter_and_decimation() -> None:
    coefficients = preprocessing.design_fir()
    x = np.full((257, 30), 2.5, dtype=np.float32)
    y = np.zeros(257, dtype=np.int8)

    x_64hz, y_64hz, audit = preprocessing.preprocess_record(x, y, coefficients)

    assert x_64hz.shape == (129, 30)
    assert y_64hz.shape == (129,)
    assert np.allclose(x_64hz, 2.5, rtol=0.0, atol=1e-6)
    assert audit["endpoint_truncation_sec"] == pytest.approx(0.0)


def test_nearest_labels_use_even_128hz_samples() -> None:
    labels = np.asarray([0, 1, 1, 0, 0, 1, 1], dtype=np.int8)

    output, nearest = preprocessing.resample_labels_nearest_64hz(labels)

    assert nearest.tolist() == [0, 2, 4, 6]
    assert output.tolist() == [0, 1, 0, 1]


def test_delay_compensation_keeps_centered_impulse_aligned() -> None:
    coefficients = preprocessing.design_fir()
    x = np.zeros((257, 30), dtype=np.float32)
    x[128, :] = 1.0

    aligned = preprocessing.filter_and_align_128hz(x, coefficients)
    downsampled, source_indices = preprocessing.resample_signal_64hz(aligned)

    assert int(np.argmax(aligned[:, 0])) == 128
    assert int(np.argmax(downsampled[:, 0])) == 64
    assert source_indices[64] == 128


def test_filter_meets_documented_stopband_check() -> None:
    response = preprocessing.filter_response(preprocessing.design_fir())

    assert response["gain_db_at_28hz"] == pytest.approx(-6.02, abs=0.05)
    assert response["maximum_stopband_gain_db_at_or_above_32hz"] < -50.0
