"""Reusable models and metrics for the Daphnet NBM E0--E3 study.

The module is deliberately independent of the experiment runner so the
pre-registered mechanisms (C1-MAD, true bottlenecks and masked prediction)
can be unit-tested without loading the Daphnet dataset.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cnbr_fog.temporal_conv_autoencoder import ConvNormGELU, ResidualConvBlock


FS = 64
WINDOW = 128
CHANNELS = 9
EPSILON = 1e-8
BANDS = ((0.5, 3.0), (3.0, 8.0), (8.0, 15.0))


def _parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


class TrueBottleneckAE(nn.Module):
    """E2 autoencoder with a genuinely smaller latent tensor than the input."""

    VARIANTS = ("P24", "P16")

    def __init__(self, variant: str = "P24") -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant
        self.encoder_stage1 = nn.Sequential(
            ConvNormGELU(CHANNELS, 32, kernel_size=7, stride=2, padding=3),
            ResidualConvBlock(32),
        )
        if variant == "P24":
            self.encoder_stage2 = nn.Sequential(
                ConvNormGELU(32, 24, kernel_size=5, stride=2, padding=2),
                ResidualConvBlock(24),
            )
            self.encoder_stage3: nn.Module = nn.Identity()
            latent_channels, latent_samples = 24, 32
            self.decoder_stage1 = nn.Sequential(
                ConvNormGELU(24, 32, kernel_size=5, padding=2),
                ResidualConvBlock(32),
            )
            self.decoder_stage2: nn.Module = nn.Identity()
        else:
            self.encoder_stage2 = nn.Sequential(
                ConvNormGELU(32, 32, kernel_size=5, stride=2, padding=2),
                ResidualConvBlock(32),
            )
            self.encoder_stage3 = nn.Sequential(
                ConvNormGELU(32, 32, kernel_size=5, stride=2, padding=2),
                ResidualConvBlock(32),
            )
            latent_channels, latent_samples = 32, 16
            self.decoder_stage1 = nn.Sequential(
                ConvNormGELU(32, 32, kernel_size=5, padding=2),
                ResidualConvBlock(32),
            )
            self.decoder_stage2 = nn.Sequential(
                ConvNormGELU(32, 32, kernel_size=5, padding=2),
                ResidualConvBlock(32),
            )
        self.latent_channels = latent_channels
        self.latent_samples = latent_samples
        self.decoder_final = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, CHANNELS, kernel_size=1),
        )

    @staticmethod
    def _up(values: torch.Tensor) -> torch.Tensor:
        return F.interpolate(values, scale_factor=2.0, mode="linear", align_corners=False)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or tuple(values.shape[1:]) != (CHANNELS, WINDOW):
            raise ValueError(f"expected [B,{CHANNELS},{WINDOW}], got {tuple(values.shape)}")
        latent = self.encoder_stage3(self.encoder_stage2(self.encoder_stage1(values)))
        decoded = self.decoder_stage1(self._up(latent))
        if self.variant == "P16":
            decoded = self.decoder_stage2(self._up(decoded))
        reconstruction = self.decoder_final(self._up(decoded))
        if reconstruction.shape[-1] != WINDOW:
            raise RuntimeError(f"decoder produced {reconstruction.shape[-1]} samples")
        return reconstruction, latent

    def architecture_config(self) -> dict[str, Any]:
        latent_elements = self.latent_channels * self.latent_samples
        return {
            "name": "E2_true_bottleneck_temporal_autoencoder",
            "variant": self.variant,
            "input_shape": ["batch", CHANNELS, WINDOW],
            "latent_shape": ["batch", self.latent_channels, self.latent_samples],
            "latent_elements": latent_elements,
            "input_elements": CHANNELS * WINDOW,
            "compression_ratio": CHANNELS * WINDOW / latent_elements,
            "parameter_count": _parameter_count(self),
            "long_skip": False,
        }


class HistoryPredictor(nn.Module):
    """E3 causal four-second context to following two-second target model."""

    def __init__(self, latent_channels: int = 24, input_channels: int = CHANNELS) -> None:
        super().__init__()
        if latent_channels not in (24, 48):
            raise ValueError("latent_channels must be 24 or 48")
        if input_channels not in (CHANNELS, CHANNELS + 1):
            raise ValueError("input_channels must be 9 or 10")
        self.latent_channels = int(latent_channels)
        self.input_channels = int(input_channels)
        self.encoder_stage1 = nn.Sequential(
            ConvNormGELU(input_channels, 32, kernel_size=7, stride=2, padding=3),
            ResidualConvBlock(32),
        )
        self.encoder_stage2 = nn.Sequential(
            ConvNormGELU(32, latent_channels, kernel_size=5, stride=2, padding=2),
            ResidualConvBlock(latent_channels),
        )
        self.encoder_stage3 = nn.Sequential(
            ConvNormGELU(
                latent_channels,
                latent_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            ResidualConvBlock(latent_channels),
        )
        self.decoder_stage1 = nn.Sequential(
            ConvNormGELU(latent_channels, 32, kernel_size=5, padding=2),
            ResidualConvBlock(32),
        )
        self.decoder_stage2 = nn.Sequential(
            ConvNormGELU(32, 32, kernel_size=5, padding=2),
            ResidualConvBlock(32),
        )
        self.decoder_final = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, CHANNELS, kernel_size=1),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.input_channels, 256)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected[0]},{expected[1]}], got {tuple(values.shape)}")
        latent = self.encoder_stage3(self.encoder_stage2(self.encoder_stage1(values)))
        decoded = self.decoder_stage1(
            F.interpolate(latent, scale_factor=2.0, mode="linear", align_corners=False)
        )
        decoded = self.decoder_stage2(
            F.interpolate(decoded, scale_factor=2.0, mode="linear", align_corners=False)
        )
        target = self.decoder_final(decoded)
        if target.shape[-1] != WINDOW:
            raise RuntimeError(f"history predictor produced {target.shape[-1]} samples")
        return target, latent

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "E3_causal_history_predictor",
            "input_shape": ["batch", self.input_channels, 256],
            "target_shape": ["batch", CHANNELS, WINDOW],
            "latent_shape": ["batch", self.latent_channels, 32],
            "latent_elements": self.latent_channels * 32,
            "target_visible_in_input": self.input_channels == CHANNELS + 1,
            "parameter_count": _parameter_count(self),
            "long_skip": False,
        }


def correlation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = predicted - predicted.mean(dim=-1, keepdim=True)
    actual = target - target.mean(dim=-1, keepdim=True)
    numerator = torch.sum(pred * actual, dim=-1)
    denominator = torch.sqrt(
        torch.sum(pred.square(), dim=-1) * torch.sum(actual.square(), dim=-1) + EPSILON
    )
    return 1.0 - torch.mean(numerator / denominator)


def l4_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = F.mse_loss(
        predicted[..., 1:] - predicted[..., :-1],
        target[..., 1:] - target[..., :-1],
    )
    return (
        0.70 * F.smooth_l1_loss(predicted, target)
        + 0.15 * correlation_loss(predicted, target)
        + 0.15 * difference
    )


def masked_l4_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L4 restricted to a temporal mask shared by all nine signal channels."""
    if mask.ndim != 3 or mask.shape[1] != 1 or mask.shape[-1] != target.shape[-1]:
        raise ValueError("mask must have shape [B,1,T]")
    weights = mask.to(dtype=target.dtype).expand_as(target)
    denominator = torch.clamp(weights.sum(), min=1.0)
    huber = F.smooth_l1_loss(predicted, target, reduction="none")
    huber = torch.sum(huber * weights) / denominator

    pred_center = predicted - (
        torch.sum(predicted * weights, dim=-1, keepdim=True)
        / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1.0)
    )
    target_center = target - (
        torch.sum(target * weights, dim=-1, keepdim=True)
        / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1.0)
    )
    numerator = torch.sum(pred_center * target_center * weights, dim=-1)
    denom = torch.sqrt(
        torch.sum(pred_center.square() * weights, dim=-1)
        * torch.sum(target_center.square() * weights, dim=-1)
        + EPSILON
    )
    correlation = 1.0 - torch.mean(numerator / denom)

    pair_mask = weights[..., 1:] * weights[..., :-1]
    pred_delta = predicted[..., 1:] - predicted[..., :-1]
    target_delta = target[..., 1:] - target[..., :-1]
    delta = torch.sum((pred_delta - target_delta).square() * pair_mask) / torch.clamp(
        pair_mask.sum(), min=1.0
    )
    return 0.70 * huber + 0.15 * correlation + 0.15 * delta


def e3b_objective(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return 0.70 * masked_l4_loss(predicted, target, mask) + 0.30 * l4_loss(
        predicted, target
    )


def random_block_mask(
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    minimum: int = 16,
    maximum: int = 32,
) -> torch.Tensor:
    """Draw one or two contiguous 0.25--0.5 s target masks per example."""
    mask = torch.zeros((batch_size, 1, WINDOW), dtype=torch.bool, device=device)
    counts = torch.randint(1, 3, (batch_size,), generator=generator)
    lengths = torch.randint(minimum, maximum + 1, (batch_size, 2), generator=generator)
    starts = torch.randint(0, WINDOW - minimum + 1, (batch_size, 2), generator=generator)
    for batch_index in range(batch_size):
        for block_index in range(int(counts[batch_index])):
            length = int(lengths[batch_index, block_index])
            start = min(int(starts[batch_index, block_index]), WINDOW - length)
            mask[batch_index, 0, start : start + length] = True
    return mask


def fixed_quarter_masks(batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    masks: list[torch.Tensor] = []
    for quarter in range(4):
        mask = torch.zeros((batch_size, 1, WINDOW), dtype=torch.bool, device=device)
        mask[..., quarter * 32 : (quarter + 1) * 32] = True
        masks.append(mask)
    return tuple(masks)


def build_e3b_input(
    history: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Concatenate two-second history, masked target and a binary mask channel."""
    if history.shape[-1] != WINDOW or target.shape[-1] != WINDOW:
        raise ValueError("history and target must both contain 128 samples")
    masked_target = target.masked_fill(mask.expand_as(target), 0.0)
    signal = torch.cat((history, masked_target), dim=-1)
    observed = torch.ones(
        (len(target), 1, 2 * WINDOW), dtype=target.dtype, device=target.device
    )
    observed[..., WINDOW:] = (~mask).to(dtype=target.dtype)
    return torch.cat((signal, observed), dim=1)


@dataclass(frozen=True)
class C1Parameters:
    center: np.ndarray
    scale: np.ndarray
    epsilon: float = 1e-6

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "epsilon": float(self.epsilon),
            "clip": None,
        }


def fit_c1_mad(residual: np.ndarray, epsilon: float = 1e-6) -> C1Parameters:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (WINDOW, CHANNELS) or len(values) == 0:
        raise ValueError(f"expected non-empty [N,{WINDOW},{CHANNELS}] residual")
    center = np.median(values, axis=(0, 1))
    mad = np.median(np.abs(values - center.reshape(1, 1, -1)), axis=(0, 1))
    scale = 1.4826 * mad
    fallback = np.std(values, axis=(0, 1))
    scale = np.where(scale > epsilon, scale, fallback)
    scale = np.maximum(scale, epsilon)
    return C1Parameters(center.astype(np.float64), scale.astype(np.float64), epsilon)


def apply_c1(residual: np.ndarray, parameters: C1Parameters) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    calibrated = (values - parameters.center.reshape(1, 1, -1)) / (
        parameters.scale.reshape(1, 1, -1) + parameters.epsilon
    )
    return np.ascontiguousarray(calibrated.astype(np.float32))


def chronological_calibration_split(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Split validation Non-FoG chronologically and remove overlapping windows."""
    if len(rows) < 4:
        raise ValueError("at least four validation Non-FoG windows are required")
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            str(rows[index]["record_id"]),
            int(rows[index]["start_index"]),
            str(rows[index]["window_id"]),
        ),
    )
    cut = max(1, min(len(order) - 2, len(order) // 2))
    calibration = order[:cut]
    threshold_candidates = order[cut:]
    last_by_record: dict[str, int] = {}
    for index in calibration:
        row = rows[index]
        record = str(row["record_id"])
        last_by_record[record] = max(
            last_by_record.get(record, -1), int(row["end_index_exclusive"])
        )
    threshold = [
        index
        for index in threshold_candidates
        if int(rows[index]["start_index"])
        >= last_by_record.get(str(rows[index]["record_id"]), -1)
    ]
    if not threshold:
        # A different record is always non-overlapping; otherwise retain one
        # stride embargo by moving the cut left until a legal threshold exists.
        for candidate_cut in range(cut - 1, 0, -1):
            calibration = order[:candidate_cut]
            last_by_record = {}
            for index in calibration:
                row = rows[index]
                record = str(row["record_id"])
                last_by_record[record] = max(
                    last_by_record.get(record, -1), int(row["end_index_exclusive"])
                )
            threshold = [
                index
                for index in order[candidate_cut:]
                if int(rows[index]["start_index"])
                >= last_by_record.get(str(rows[index]["record_id"]), -1)
            ]
            if threshold:
                break
    if not calibration or not threshold:
        raise ValueError("unable to form non-overlapping chronological calibration split")
    audit: list[dict[str, Any]] = []
    calibration_set, threshold_set = set(calibration), set(threshold)
    for index, row in enumerate(rows):
        audit.append(
            {
                "window_id": row["window_id"],
                "record_id": row["record_id"],
                "start_index": int(row["start_index"]),
                "end_index_exclusive": int(row["end_index_exclusive"]),
                "c1_role": (
                    "calibration"
                    if index in calibration_set
                    else "score_threshold"
                    if index in threshold_set
                    else "embargo_dropped"
                ),
            }
        )
    return (
        np.asarray(calibration, dtype=np.int64),
        np.asarray(threshold, dtype=np.int64),
        audit,
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def reconstruction_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    channel_names: Sequence[str] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return V1 summary, per-channel rows and per-window rows in IQR units."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 3 or actual.shape[1:] != (
        WINDOW,
        CHANNELS,
    ):
        raise ValueError("actual and predicted must share [N,128,9]")
    names = tuple(channel_names or [f"channel_{index}" for index in range(CHANNELS)])
    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    psd_actual = np.abs(np.fft.rfft(actual * np.hanning(WINDOW)[None, :, None], axis=1)) ** 2
    psd_predicted = np.abs(
        np.fft.rfft(predicted * np.hanning(WINDOW)[None, :, None], axis=1)
    ) ** 2
    psd_mask = (frequency >= 0.5) & (frequency <= 15.0)
    error = predicted - actual
    rmse_wc = np.sqrt(np.mean(np.square(error), axis=1))
    corr_wc = np.empty((len(actual), CHANNELS), dtype=np.float64)
    delta_corr_wc = np.empty_like(corr_wc)
    lag_wc = np.empty_like(corr_wc)
    psd_distance_wc = np.mean(
        np.abs(np.log(psd_actual[:, psd_mask] + EPSILON) - np.log(psd_predicted[:, psd_mask] + EPSILON)),
        axis=1,
    )
    band_errors: dict[tuple[float, float], np.ndarray] = {}
    for low, high in BANDS:
        mask = (frequency >= low) & (frequency < high if high < 15.0 else frequency <= high)
        source_power = np.sum(psd_actual[:, mask], axis=1)
        estimate_power = np.sum(psd_predicted[:, mask], axis=1)
        band_errors[(low, high)] = np.abs(estimate_power - source_power) / (
            source_power + EPSILON
        )
    for window_index in range(len(actual)):
        for channel in range(CHANNELS):
            source = actual[window_index, :, channel]
            estimate = predicted[window_index, :, channel]
            corr_wc[window_index, channel] = _safe_corr(source, estimate)
            delta_corr_wc[window_index, channel] = _safe_corr(
                np.diff(source), np.diff(estimate)
            )
            source_centered = source - np.mean(source)
            estimate_centered = estimate - np.mean(estimate)
            correlation = np.correlate(source_centered, estimate_centered, mode="full")
            lag_wc[window_index, channel] = int(np.argmax(correlation) - (WINDOW - 1))

    channel_rows: list[dict[str, Any]] = []
    for channel, name in enumerate(names):
        row: dict[str, Any] = {
            "channel_index": channel,
            "channel_name": name,
            "nrmse_median": float(np.median(rmse_wc[:, channel])),
            "nrmse_p90": float(np.percentile(rmse_wc[:, channel], 90)),
            "nrmse_p95": float(np.percentile(rmse_wc[:, channel], 95)),
            "pearson_median": float(np.nanmedian(corr_wc[:, channel])),
            "delta_pearson_median": float(np.nanmedian(delta_corr_wc[:, channel])),
            "psd_log_distance_median": float(np.median(psd_distance_wc[:, channel])),
            "absolute_lag_median_samples": float(np.median(np.abs(lag_wc[:, channel]))),
        }
        for low, high in BANDS:
            row[f"band_{low:g}_{high:g}_relative_error_median"] = float(
                np.median(band_errors[(low, high)][:, channel])
            )
        channel_rows.append(row)

    window_rows: list[dict[str, Any]] = []
    for index in range(len(actual)):
        row = {
            "window_local_index": index,
            "nrmse_median": float(np.median(rmse_wc[index])),
            "pearson_median": float(np.nanmedian(corr_wc[index])),
            "delta_pearson_median": float(np.nanmedian(delta_corr_wc[index])),
            "psd_log_distance_median": float(np.median(psd_distance_wc[index])),
            "absolute_lag_median_samples": float(np.median(np.abs(lag_wc[index]))),
        }
        for low, high in BANDS:
            row[f"band_{low:g}_{high:g}_relative_error_median"] = float(
                np.median(band_errors[(low, high)][index])
            )
        window_rows.append(row)
    summary = {
        "windows": int(len(actual)),
        "nrmse_median": float(np.median(rmse_wc)),
        "nrmse_p90": float(np.percentile(rmse_wc, 90)),
        "nrmse_p95": float(np.percentile(rmse_wc, 95)),
        "pearson_median": float(np.nanmedian(corr_wc)),
        "delta_pearson_median": float(np.nanmedian(delta_corr_wc)),
        "psd_log_distance_median": float(np.median(psd_distance_wc)),
        "absolute_lag_median_samples": float(np.median(np.abs(lag_wc))),
        "band_0.5_3_relative_error_median": float(np.median(band_errors[(0.5, 3.0)])),
        "band_3_8_relative_error_median": float(np.median(band_errors[(3.0, 8.0)])),
        "band_8_15_relative_error_median": float(np.median(band_errors[(8.0, 15.0)])),
    }
    return summary, channel_rows, window_rows


def score_shift_metrics(validation: np.ndarray, test: np.ndarray) -> dict[str, float]:
    validation = np.asarray(validation, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if len(validation) == 0 or len(test) == 0:
        raise ValueError("shift metrics require non-empty validation and test arrays")
    validation_median = float(np.median(validation))
    test_median = float(np.median(test))
    mad = float(np.median(np.abs(validation - validation_median)))
    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        wasserstein = float(wasserstein_distance(validation, test))
        ks = float(ks_2samp(validation, test, method="auto").statistic)
    except ImportError:  # pragma: no cover - scipy is present in the experiment env
        wasserstein = math.nan
        ks = math.nan
    result = {
        "validation_nonfog_median": validation_median,
        "test_nonfog_median": test_median,
        "shift_median": abs(test_median - validation_median),
        "validation_nonfog_mad": mad,
        "shift_robust": abs(test_median - validation_median) / (1.4826 * mad + EPSILON),
        "wasserstein_distance": wasserstein,
        "ks_statistic_descriptive": ks,
    }
    for quantile in (90.0, 95.0, 99.0):
        validation_q = float(np.percentile(validation, quantile))
        test_q = float(np.percentile(test, quantile))
        key = int(quantile)
        result[f"validation_nonfog_q{key}"] = validation_q
        result[f"test_nonfog_q{key}"] = test_q
        result[f"tail_shift_q{key}"] = abs(test_q - validation_q)
    return result


def _alarm_episodes(
    rows: Sequence[dict[str, Any]], scores: np.ndarray, high: float, low: float
) -> list[dict[str, Any]]:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            str(rows[index]["record_id"]),
            int(rows[index]["start_index"]),
        ),
    )
    episodes: list[dict[str, Any]] = []
    active = False
    above = 0
    below = 0
    start_position: int | None = None
    previous_record: str | None = None
    previous_start: int | None = None

    def close(end_position: int) -> None:
        nonlocal active, above, below, start_position
        if start_position is not None:
            selected = ordered[start_position : end_position + 1]
            episodes.append(
                {
                    "record_id": str(rows[selected[0]]["record_id"]),
                    "start_index": int(rows[selected[0]]["start_index"]),
                    "end_index_exclusive": int(rows[selected[-1]]["end_index_exclusive"]),
                    "row_indices": selected,
                }
            )
        active = False
        above = 0
        below = 0
        start_position = None

    for position, index in enumerate(ordered):
        record = str(rows[index]["record_id"])
        start = int(rows[index]["start_index"])
        discontinuity = (
            previous_record is not None
            and (record != previous_record or start - int(previous_start) > FS + 1)
        )
        if discontinuity and active:
            close(position - 1)
        if discontinuity:
            above = below = 0
        value = float(scores[index])
        if not active:
            above = above + 1 if value > high else 0
            if above >= 2:
                active = True
                start_position = position - 1
                below = 0
        else:
            below = below + 1 if value < low else 0
            if below >= 2:
                close(position)
        previous_record, previous_start = record, start
    if active:
        close(len(ordered) - 1)

    merged: list[dict[str, Any]] = []
    for episode in episodes:
        if (
            merged
            and episode["record_id"] == merged[-1]["record_id"]
            and int(episode["start_index"]) - int(merged[-1]["end_index_exclusive"])
            < 5 * FS
        ):
            merged[-1]["end_index_exclusive"] = episode["end_index_exclusive"]
            merged[-1]["row_indices"] = sorted(
                set(merged[-1]["row_indices"]) | set(episode["row_indices"])
            )
        else:
            merged.append(dict(episode))
    return merged


def threshold_metrics(
    rows: Sequence[dict[str, Any]],
    scores: np.ndarray,
    validation_nonfog_scores: np.ndarray,
    *,
    quantile: float,
) -> dict[str, Any]:
    """Window and event metrics for Q95 or Q99.2 deployment thresholds."""
    scores = np.asarray(scores, dtype=np.float64)
    reference = np.asarray(validation_nonfog_scores, dtype=np.float64)
    if len(rows) != len(scores) or len(reference) == 0:
        raise ValueError("threshold rows/scores/reference mismatch")
    high = float(np.percentile(reference, quantile))
    low = 0.8 * high
    labels = np.asarray([int(row["y_binary"]) for row in rows], dtype=np.int8)
    fog_mask = labels == 1
    nonfog_mask = ~fog_mask
    predictions = scores > high
    episodes = _alarm_episodes(rows, scores, high, low)
    detected_events: set[str] = set()
    delays: list[float] = []
    false_events = 0
    true_event_starts: dict[str, int] = {}
    for row in rows:
        event_id = str(row.get("event_id", "")).strip()
        if int(row["y_binary"]) == 1 and event_id:
            true_event_starts[event_id] = min(
                true_event_starts.get(event_id, int(row["start_index"])),
                int(row["start_index"]),
            )
    for episode in episodes:
        event_ids = {
            str(rows[index].get("event_id", "")).strip()
            for index in episode["row_indices"]
            if int(rows[index]["y_binary"]) == 1
            and str(rows[index].get("event_id", "")).strip()
        }
        if not event_ids:
            false_events += 1
            continue
        for event_id in event_ids:
            if event_id not in detected_events:
                delay = max(
                    0.0,
                    (int(episode["start_index"]) - true_event_starts[event_id]) / FS,
                )
                delays.append(delay)
            detected_events.add(event_id)
    nonfog_minutes = max(float(np.sum(nonfog_mask)) / 60.0, 1.0 / 60.0)
    return {
        "quantile": float(quantile),
        "threshold": high,
        "hysteresis_low_threshold": low,
        "window_recall": float(np.mean(predictions[fog_mask])) if np.any(fog_mask) else math.nan,
        "window_false_alarm_per_minute": float(np.mean(predictions[nonfog_mask]) * 60.0)
        if np.any(nonfog_mask)
        else math.nan,
        "alarm_event_count": len(episodes),
        "false_alarm_event_count": false_events,
        "event_false_alarm_per_minute": false_events / nonfog_minutes,
        "fog_event_count": len(true_event_starts),
        "detected_fog_event_count": len(detected_events),
        "event_fog_recall": len(detected_events) / len(true_event_starts)
        if true_event_starts
        else math.nan,
        "median_detection_delay_seconds": float(np.median(delays)) if delays else math.nan,
    }


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    *,
    device: torch.device,
    iterations: int = 20,
) -> dict[str, Any]:
    """Count parameters, approximate Conv1d MACs and measured batch-one latency."""
    macs = 0
    hooks: list[Any] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        assert isinstance(module, nn.Conv1d)
        macs += int(
            output.shape[0]
            * output.shape[1]
            * output.shape[2]
            * (module.in_channels // module.groups)
            * module.kernel_size[0]
        )

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(hook))
    model = model.to(device).eval()
    sample = torch.zeros(input_shape, dtype=torch.float32, device=device)
    with torch.no_grad():
        model(sample)
    for item in hooks:
        item.remove()
    encoder = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "encoder" in name
    )
    total = _parameter_count(model)
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return {
        "total_parameters": total,
        "encoder_parameters": int(encoder),
        "decoder_parameters": int(total - encoder),
        "approximate_conv1d_macs_per_window": int(macs),
        "inference_ms_per_window_batch1": 1000.0
        * (time.perf_counter() - started)
        / iterations,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(len(values), np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted.tolist()
    order = finite[np.argsort(values[finite])]
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(order) - reverse_rank + 1
        running = min(running, values[index] * len(order) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_subject_statistics(
    left: dict[str, float],
    right: dict[str, float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    subjects = sorted(set(left) & set(right))
    difference = np.asarray([right[subject] - left[subject] for subject in subjects])
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return {
            "subjects": 0,
            "median_difference": math.nan,
            "bootstrap_ci_low": math.nan,
            "bootstrap_ci_high": math.nan,
            "wilcoxon_p": math.nan,
            "rank_biserial": math.nan,
            "improved_subjects": 0,
        }
    generator = np.random.default_rng(seed)
    boot = np.asarray(
        [
            np.median(generator.choice(difference, size=len(difference), replace=True))
            for _ in range(max(1, bootstrap_samples))
        ]
    )
    try:
        from scipy.stats import wilcoxon

        p_value = (
            float(wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue)
            if np.any(np.abs(difference) > 1e-12)
            else 1.0
        )
    except (ImportError, ValueError):
        p_value = math.nan
    absolute = np.abs(difference)
    ranks = np.argsort(np.argsort(absolute)) + 1
    positive = float(np.sum(ranks[difference > 0]))
    negative = float(np.sum(ranks[difference < 0]))
    denominator = positive + negative
    effect = (positive - negative) / denominator if denominator else 0.0
    return {
        "subjects": int(len(difference)),
        "median_difference": float(np.median(difference)),
        "bootstrap_ci_low": float(np.percentile(boot, 2.5)),
        "bootstrap_ci_high": float(np.percentile(boot, 97.5)),
        "wilcoxon_p": p_value,
        "rank_biserial": effect,
        "improved_subjects": int(np.sum(difference > 0)),
    }


__all__ = [
    "BANDS",
    "C1Parameters",
    "HistoryPredictor",
    "TrueBottleneckAE",
    "apply_c1",
    "benjamini_hochberg",
    "build_e3b_input",
    "chronological_calibration_split",
    "e3b_objective",
    "fit_c1_mad",
    "fixed_quarter_masks",
    "l4_loss",
    "paired_subject_statistics",
    "profile_model",
    "random_block_mask",
    "reconstruction_rows",
    "score_shift_metrics",
    "threshold_metrics",
]
