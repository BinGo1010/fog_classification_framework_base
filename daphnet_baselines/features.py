"""Domain and handcrafted time-frequency features for FoG baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


EPSILON = 1e-12


def _validate_windows(windows: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("windows must have shape [window, channel, time]")
    if values.shape[-1] < 4:
        raise ValueError("windows are too short for spectral analysis")
    if int(sampling_rate_hz) <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not np.isfinite(values).all():
        raise ValueError("windows contain non-finite values")
    return values


def _one_sided_power(
    windows: np.ndarray,
    sampling_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = _validate_windows(windows, sampling_rate_hz)
    centered = values - values.mean(axis=-1, keepdims=True)
    taper = np.hanning(values.shape[-1]).astype(np.float64)
    taper_energy = max(float(np.square(taper).sum()), EPSILON)
    spectrum = np.fft.rfft(centered * taper, axis=-1)
    power = np.square(np.abs(spectrum)) / taper_energy
    if values.shape[-1] % 2 == 0:
        power[..., 1:-1] *= 2.0
    elif power.shape[-1] > 1:
        power[..., 1:] *= 2.0
    frequencies = np.fft.rfftfreq(
        values.shape[-1],
        d=1.0 / float(sampling_rate_hz),
    )
    return frequencies, power


def _band_power(
    power: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
    *,
    include_high: bool,
) -> np.ndarray:
    mask = frequencies >= float(low_hz)
    mask &= (
        frequencies <= float(high_hz)
        if include_high
        else frequencies < float(high_hz)
    )
    if not bool(mask.any()):
        raise ValueError(
            f"No FFT bins fall in [{low_hz}, {high_hz}] Hz; "
            "increase the input duration"
        )
    spacing = (
        float(frequencies[1] - frequencies[0])
        if len(frequencies) > 1
        else 1.0
    )
    return power[..., mask].sum(axis=-1) * spacing


def freeze_index_features(
    windows: np.ndarray,
    sampling_rate_hz: int,
    channel_index: int | Sequence[int],
    *,
    locomotor_band_hz: tuple[float, float] = (0.5, 3.0),
    freeze_band_hz: tuple[float, float] = (3.0, 8.0),
    aggregation: str = "power_pool",
    squared_ratio: bool = False,
) -> dict[str, np.ndarray]:
    """Compute the validation-thresholded Freeze Index input score.

    The canonical score is freeze-band power divided by locomotor-band power.
    ``squared_ratio`` exposes the squared-power variant used in part of the
    clinical literature without silently changing the default definition.
    """

    values = _validate_windows(windows, sampling_rate_hz)
    if isinstance(channel_index, (int, np.integer)):
        channel_indices = np.asarray([int(channel_index)], dtype=np.int64)
    else:
        channel_indices = np.asarray(tuple(channel_index), dtype=np.int64)
    if channel_indices.ndim != 1 or len(channel_indices) == 0:
        raise ValueError("channel_index must identify at least one channel")
    if len(np.unique(channel_indices)) != len(channel_indices):
        raise ValueError("Freeze Index channels must not contain duplicates")
    if channel_indices.min() < 0 or channel_indices.max() >= values.shape[1]:
        raise IndexError(f"channel indices {channel_indices.tolist()} are outside the input")
    if aggregation not in {"power_pool", "mean", "max"}:
        raise ValueError("aggregation must be power_pool, mean, or max")

    # The Bächlin-style FI uses mean-removed raw acceleration, no taper and no
    # zero-padding.  The absolute FFT scale cancels in the ratio.  Keep this
    # separate from the Hann-windowed PSD used by the generic feature extractor.
    selected = values[:, channel_indices]
    centered = selected - selected.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=-1)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(
        values.shape[-1],
        d=1.0 / float(sampling_rate_hz),
    )
    channel_locomotor = _band_power(
        power,
        frequencies,
        locomotor_band_hz[0],
        locomotor_band_hz[1],
        include_high=False,
    )
    channel_freeze = _band_power(
        power,
        frequencies,
        freeze_band_hz[0],
        freeze_band_hz[1],
        include_high=True,
    )

    if squared_ratio:
        channel_ratio = np.square(channel_freeze) / (
            np.square(channel_locomotor) + EPSILON
        )
    else:
        channel_ratio = channel_freeze / (channel_locomotor + EPSILON)
    channel_score = channel_ratio / (1.0 + channel_ratio)

    if aggregation == "power_pool":
        locomotor = channel_locomotor.mean(axis=1)
        freeze = channel_freeze.mean(axis=1)
        if squared_ratio:
            ratio = np.square(freeze) / (np.square(locomotor) + EPSILON)
        else:
            ratio = freeze / (locomotor + EPSILON)
        score = ratio / (1.0 + ratio)
    else:
        reducer = np.mean if aggregation == "mean" else np.max
        score = reducer(channel_score, axis=1)
        ratio = score / np.maximum(1.0 - score, EPSILON)
        locomotor = reducer(channel_locomotor, axis=1)
        freeze = reducer(channel_freeze, axis=1)

    ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e12, neginf=0.0)
    ratio = np.maximum(ratio, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    inactive = (locomotor + freeze) <= EPSILON
    ratio[inactive] = 0.0
    score[inactive] = 0.0
    return {
        "freeze_index": ratio.astype(np.float64, copy=False),
        "score": score.astype(np.float64, copy=False),
        "locomotor_power": locomotor.astype(np.float64, copy=False),
        "freeze_power": freeze.astype(np.float64, copy=False),
        "total_power": (locomotor + freeze).astype(np.float64, copy=False),
        "channel_locomotor_power": channel_locomotor.astype(
            np.float64, copy=False
        ),
        "channel_freeze_power": channel_freeze.astype(
            np.float64, copy=False
        ),
    }


def _safe_skew_and_kurtosis(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variance = np.mean(np.square(centered), axis=-1)
    scale = np.sqrt(np.maximum(variance, EPSILON))
    standardized = centered / scale[..., None]
    skew = np.mean(np.power(standardized, 3), axis=-1)
    kurtosis = np.mean(np.power(standardized, 4), axis=-1) - 3.0
    inactive = variance <= EPSILON
    skew[inactive] = 0.0
    kurtosis[inactive] = 0.0
    return skew, kurtosis


def _correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first_centered = first - first.mean(axis=-1, keepdims=True)
    second_centered = second - second.mean(axis=-1, keepdims=True)
    numerator = np.sum(first_centered * second_centered, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(first_centered), axis=-1)
        * np.sum(np.square(second_centered), axis=-1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPSILON,
    )


@dataclass(frozen=True)
class TimeFrequencyFeatureExtractor:
    """Deterministic temporal, spectral, and tri-axial correlation features."""

    sampling_rate_hz: int
    channel_names: tuple[str, ...]
    include_triad_magnitudes: bool = True
    batch_size: int = 2048
    locomotor_band_hz: tuple[float, float] = (0.5, 3.0)
    freeze_band_hz: tuple[float, float] = (3.0, 8.0)
    analysis_band_hz: tuple[float, float] = (0.5, 15.0)

    def __init__(
        self,
        sampling_rate_hz: int,
        channel_names: Sequence[str],
        include_triad_magnitudes: bool = True,
        batch_size: int = 2048,
        locomotor_band_hz: tuple[float, float] = (0.5, 3.0),
        freeze_band_hz: tuple[float, float] = (3.0, 8.0),
        analysis_band_hz: tuple[float, float] = (0.5, 15.0),
    ) -> None:
        object.__setattr__(self, "sampling_rate_hz", int(sampling_rate_hz))
        object.__setattr__(
            self,
            "channel_names",
            tuple(str(name) for name in channel_names),
        )
        object.__setattr__(
            self,
            "include_triad_magnitudes",
            bool(include_triad_magnitudes),
        )
        object.__setattr__(self, "batch_size", int(batch_size))
        object.__setattr__(
            self,
            "locomotor_band_hz",
            tuple(float(value) for value in locomotor_band_hz),
        )
        object.__setattr__(
            self,
            "freeze_band_hz",
            tuple(float(value) for value in freeze_band_hz),
        )
        object.__setattr__(
            self,
            "analysis_band_hz",
            tuple(float(value) for value in analysis_band_hz),
        )
        if self.sampling_rate_hz <= 0 or self.batch_size <= 0:
            raise ValueError("sampling rate and batch size must be positive")
        if not self.channel_names:
            raise ValueError("channel_names must not be empty")

    def _augment_channels(
        self,
        windows: np.ndarray,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        if (
            not self.include_triad_magnitudes
            or windows.shape[1] % 3 != 0
        ):
            return windows, self.channel_names
        magnitudes = []
        magnitude_names = []
        for start in range(0, windows.shape[1], 3):
            magnitudes.append(
                np.sqrt(np.sum(np.square(windows[:, start : start + 3]), axis=1))
            )
            prefix = self.channel_names[start].split("_acc_", 1)[0]
            magnitude_names.append(f"{prefix}_acc_magnitude")
        augmented = np.concatenate(
            [windows, np.stack(magnitudes, axis=1)],
            axis=1,
        )
        return augmented, (*self.channel_names, *magnitude_names)

    def feature_names(self) -> tuple[str, ...]:
        channel_names = self.channel_names
        if self.include_triad_magnitudes and len(channel_names) % 3 == 0:
            magnitude_names = tuple(
                f"{channel_names[start].split('_acc_', 1)[0]}_acc_magnitude"
                for start in range(0, len(channel_names), 3)
            )
            channel_names = (*channel_names, *magnitude_names)
        per_channel = (
            "mean",
            "std",
            "rms",
            "minimum",
            "maximum",
            "peak_to_peak",
            "median",
            "iqr",
            "mad",
            "absolute_mean",
            "zero_crossing_rate",
            "derivative_rms",
            "skewness",
            "excess_kurtosis",
            "total_band_power",
            "locomotor_power",
            "freeze_power",
            "high_band_power",
            "freeze_index_log1p",
            "dominant_frequency",
            "spectral_centroid",
            "spectral_entropy",
            "locomotor_power_fraction",
            "freeze_power_fraction",
        )
        names = [
            f"{channel}__{feature}"
            for channel in channel_names
            for feature in per_channel
        ]
        # Correlations retain body-axis structure and are computed only from
        # physical channels, never from derived magnitudes.
        if len(self.channel_names) % 3 == 0:
            for start in range(0, len(self.channel_names), 3):
                for first, second in ((0, 1), (0, 2), (1, 2)):
                    names.append(
                        f"corr__{self.channel_names[start + first]}"
                        f"__{self.channel_names[start + second]}"
                    )
            if len(self.channel_names) >= 6:
                triads = len(self.channel_names) // 3
                for first_triad in range(triads):
                    for second_triad in range(first_triad + 1, triads):
                        for axis in range(3):
                            names.append(
                                f"corr__{self.channel_names[3 * first_triad + axis]}"
                                f"__{self.channel_names[3 * second_triad + axis]}"
                            )
        return tuple(names)

    def _transform_batch(self, windows: np.ndarray) -> np.ndarray:
        values = _validate_windows(windows, self.sampling_rate_hz)
        if values.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, got {values.shape[1]}"
            )
        physical = values
        values, _ = self._augment_channels(values)
        centered = values - values.mean(axis=-1, keepdims=True)
        median = np.median(values, axis=-1)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=-1)
        skew, kurtosis = _safe_skew_and_kurtosis(centered)
        signs = np.signbit(centered)
        zero_crossing = np.mean(signs[..., 1:] != signs[..., :-1], axis=-1)
        derivative = np.diff(values, axis=-1) * float(self.sampling_rate_hz)

        frequencies, power = _one_sided_power(values, self.sampling_rate_hz)
        locomotor = _band_power(
            power,
            frequencies,
            self.locomotor_band_hz[0],
            self.locomotor_band_hz[1],
            include_high=False,
        )
        freeze = _band_power(
            power,
            frequencies,
            self.freeze_band_hz[0],
            self.freeze_band_hz[1],
            include_high=True,
        )
        high = _band_power(
            power,
            frequencies,
            np.nextafter(self.freeze_band_hz[1], np.inf),
            self.analysis_band_hz[1],
            include_high=True,
        )
        analysis_mask = (
            (frequencies >= self.analysis_band_hz[0])
            & (frequencies <= self.analysis_band_hz[1])
        )
        analysis_power = power[..., analysis_mask]
        analysis_frequencies = frequencies[analysis_mask]
        total = _band_power(
            power,
            frequencies,
            self.analysis_band_hz[0],
            self.analysis_band_hz[1],
            include_high=True,
        )
        normalized_power = np.divide(
            analysis_power,
            analysis_power.sum(axis=-1, keepdims=True),
            out=np.zeros_like(analysis_power),
            where=analysis_power.sum(axis=-1, keepdims=True) > EPSILON,
        )
        spectral_entropy = -np.sum(
            np.where(
                normalized_power > 0,
                normalized_power * np.log(normalized_power + EPSILON),
                0.0,
            ),
            axis=-1,
        )
        if analysis_power.shape[-1] > 1:
            spectral_entropy /= np.log(float(analysis_power.shape[-1]))
        centroid = np.sum(
            analysis_power * analysis_frequencies,
            axis=-1,
        )
        centroid = np.divide(
            centroid,
            analysis_power.sum(axis=-1),
            out=np.zeros_like(centroid),
            where=analysis_power.sum(axis=-1) > EPSILON,
        )
        dominant_indices = np.argmax(analysis_power, axis=-1)
        dominant = analysis_frequencies[dominant_indices]
        dominant[analysis_power.sum(axis=-1) <= EPSILON] = 0.0

        per_channel = np.stack(
            [
                values.mean(axis=-1),
                values.std(axis=-1),
                np.sqrt(np.mean(np.square(values), axis=-1)),
                values.min(axis=-1),
                values.max(axis=-1),
                np.ptp(values, axis=-1),
                median,
                q75 - q25,
                np.median(np.abs(values - median[..., None]), axis=-1),
                np.mean(np.abs(values), axis=-1),
                zero_crossing,
                np.sqrt(np.mean(np.square(derivative), axis=-1)),
                skew,
                kurtosis,
                total,
                locomotor,
                freeze,
                high,
                np.log1p(freeze / (locomotor + EPSILON)),
                dominant,
                centroid,
                spectral_entropy,
                locomotor / (total + EPSILON),
                freeze / (total + EPSILON),
            ],
            axis=-1,
        ).reshape(len(values), -1)

        correlations: list[np.ndarray] = []
        if physical.shape[1] % 3 == 0:
            for start in range(0, physical.shape[1], 3):
                for first, second in ((0, 1), (0, 2), (1, 2)):
                    correlations.append(
                        _correlation(
                            physical[:, start + first],
                            physical[:, start + second],
                        )
                    )
            triads = physical.shape[1] // 3
            for first_triad in range(triads):
                for second_triad in range(first_triad + 1, triads):
                    for axis in range(3):
                        correlations.append(
                            _correlation(
                                physical[:, 3 * first_triad + axis],
                                physical[:, 3 * second_triad + axis],
                            )
                        )
        if correlations:
            per_channel = np.concatenate(
                [per_channel, np.stack(correlations, axis=1)],
                axis=1,
            )
        result = np.nan_to_num(
            per_channel,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        expected = len(self.feature_names())
        if result.shape[1] != expected:
            raise AssertionError(
                f"Feature implementation produced {result.shape[1]}, expected {expected}"
            )
        return result.astype(np.float32, copy=False)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        values = _validate_windows(windows, self.sampling_rate_hz)
        chunks = [
            self._transform_batch(values[start : start + self.batch_size])
            for start in range(0, len(values), self.batch_size)
        ]
        if not chunks:
            return np.empty((0, len(self.feature_names())), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(chunks, axis=0))
