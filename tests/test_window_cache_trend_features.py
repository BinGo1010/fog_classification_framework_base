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


def test_bandpower_features_separate_low_and_high_bands() -> None:
    sampling_rate_hz = 60
    t = np.arange(360, dtype=np.float32) / sampling_rate_hz
    sine_2hz = np.sin(2 * np.pi * 2.0 * t)
    sine_5hz = np.sin(2 * np.pi * 5.0 * t)
    long_signals = np.stack([sine_2hz, sine_5hz], axis=1).astype(np.float32)

    features = ["bandpower_low", "bandpower_high", "freeze_index", "bandpower_ratio", "dominant_power"]
    bandpower_low, bandpower_high, freeze_index, bandpower_ratio, dominant_power = _long_trend_values(
        long_signals,
        short_size=60,
        trend_features=features,
        sampling_rate_hz=sampling_rate_hz,
    )

    assert bandpower_low[0] > bandpower_high[0]
    assert bandpower_high[1] > bandpower_low[1]
    assert freeze_index[1] > freeze_index[0]
    assert bandpower_ratio[1] > bandpower_ratio[0]
    assert np.all(dominant_power >= 0)
    assert np.all(dominant_power <= 1)


def test_bandpower_features_are_zero_safe() -> None:
    long_signals = np.zeros((360, 3), dtype=np.float32)
    features = ["bandpower_low", "bandpower_high", "freeze_index", "bandpower_ratio", "dominant_power"]
    values = _long_trend_values(
        long_signals,
        short_size=60,
        trend_features=features,
        sampling_rate_hz=60,
    )

    for feature_values in values:
        assert np.all(np.isfinite(feature_values))
        assert np.all(feature_values == 0)
