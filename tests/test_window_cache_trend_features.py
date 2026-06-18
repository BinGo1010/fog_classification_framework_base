from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_provider.window_cache import _long_trend_values


def test_frequency_trend_features_are_finite_and_use_hz() -> None:
    sampling_rate_hz = 60
    t = np.arange(360, dtype=np.float32) / sampling_rate_hz
    sine_2hz = np.sin(2 * np.pi * 2.0 * t)
    zeros = np.zeros_like(sine_2hz)
    long_signals = np.stack([sine_2hz, zeros], axis=1).astype(np.float32)

    features = ["fft_energy", "fft_entropy", "fft_centroid", "fft_peak_freq"]
    fft_energy, fft_entropy, fft_centroid, fft_peak_freq = _long_trend_values(
        long_signals,
        short_size=60,
        trend_features=features,
        sampling_rate_hz=sampling_rate_hz,
    )

    assert np.all(np.isfinite(fft_energy))
    assert np.all(np.isfinite(fft_entropy))
    assert np.all(np.isfinite(fft_centroid))
    assert np.all(np.isfinite(fft_peak_freq))
    assert fft_energy[0] > 0
    assert fft_energy[1] == 0
    assert abs(float(fft_peak_freq[0]) - 2.0) < 0.2
    assert fft_peak_freq[1] == 0
