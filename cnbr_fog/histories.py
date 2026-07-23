"""Leakage-safe construction of rolling residual histories.

The conditional normal predictor emits overlapping fixed-horizon residual
blocks.  Histories use horizon-spaced blocks so their physical target intervals
are contiguous but never overlap.  Every sample therefore occurs exactly once,
all segments share the same forecast-lead distribution, and the 0.5-second
variant is identical to the original single-block representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .data import WindowTable


@dataclass(frozen=True)
class HistoryPlan:
    """Rows forming the common maximum-length history for each anchor."""

    anchor_rows: np.ndarray
    anchor_window_indices: np.ndarray
    max_chain_rows: np.ndarray

    def take(self, rows: np.ndarray) -> "HistoryPlan":
        rows = np.asarray(rows, dtype=np.int64)
        return HistoryPlan(
            anchor_rows=self.anchor_rows[rows],
            anchor_window_indices=self.anchor_window_indices[rows],
            max_chain_rows=self.max_chain_rows[rows],
        )


def history_block_count(
    history_samples: int,
    horizon_samples: int,
    stride_samples: int,
) -> int:
    """Return non-overlapping forecast blocks needed for an exact history."""

    history_samples = int(history_samples)
    horizon_samples = int(horizon_samples)
    stride_samples = int(stride_samples)
    if min(history_samples, horizon_samples, stride_samples) <= 0:
        raise ValueError("history, horizon, and stride samples must be positive")
    if history_samples < horizon_samples:
        raise ValueError("history must be at least as long as the forecast horizon")
    if history_samples % horizon_samples:
        raise ValueError("history must be divisible by the forecast horizon")
    if horizon_samples % stride_samples:
        raise ValueError("forecast horizon must be divisible by stride")
    return history_samples // horizon_samples


def make_common_history_plan(
    windows: WindowTable,
    window_indices: np.ndarray,
    horizon_samples: int,
    stride_samples: int,
    max_history_samples: int,
) -> HistoryPlan:
    """Find anchors with a complete record-local chain up to ``max_history``.

    ``max_chain_rows`` contains local row positions into ``window_indices``;
    global window ids are never assumed to be contiguous.
    """

    indices = np.asarray(window_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("window_indices must be one-dimensional")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("window_indices must not contain duplicates")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(windows)):
        raise IndexError("window index outside WindowTable")

    horizon_samples = int(horizon_samples)
    stride_samples = int(stride_samples)
    chain_length = history_block_count(
        max_history_samples, horizon_samples, stride_samples
    )

    lookup: dict[tuple[int, int], int] = {}
    for row, global_index in enumerate(indices):
        rec = int(windows.record_index[global_index])
        target_start = int(windows.target_start[global_index])
        target_end = int(windows.target_end[global_index])
        if target_end - target_start != horizon_samples:
            raise ValueError("all residual blocks must match horizon_samples")
        key = (rec, target_start)
        if key in lookup:
            raise ValueError(f"duplicate record/target start: {key}")
        lookup[key] = row

    anchor_rows: list[int] = []
    chains: list[list[int]] = []
    for anchor_row, global_index in enumerate(indices):
        rec = int(windows.record_index[global_index])
        anchor_start = int(windows.target_start[global_index])
        starts = [
            anchor_start - offset * horizon_samples
            for offset in range(chain_length - 1, -1, -1)
        ]
        chain = [lookup.get((rec, start)) for start in starts]
        if any(row is None for row in chain):
            continue
        anchor_rows.append(anchor_row)
        chains.append([int(row) for row in chain])

    chain_array = np.asarray(chains, dtype=np.int64)
    if chain_array.size == 0:
        chain_array = np.empty((0, chain_length), dtype=np.int64)
    anchor_array = np.asarray(anchor_rows, dtype=np.int64)
    return HistoryPlan(
        anchor_rows=anchor_array,
        anchor_window_indices=indices[anchor_array],
        max_chain_rows=chain_array,
    )


def materialize_nonoverlap_residual_history(
    residual_blocks: np.ndarray,
    plan: HistoryPlan,
    history_samples: int,
    horizon_samples: int,
    stride_samples: int,
) -> np.ndarray:
    """Concatenate horizon-spaced blocks into an exact causal history."""

    blocks = np.asarray(residual_blocks)
    if blocks.ndim != 3:
        raise ValueError("residual_blocks must have shape [window, channel, time]")
    if blocks.shape[2] != int(horizon_samples):
        raise ValueError("residual block length does not match horizon_samples")
    if plan.max_chain_rows.size and plan.max_chain_rows.max() >= len(blocks):
        raise IndexError("HistoryPlan refers to a residual row that is unavailable")

    block_count = history_block_count(
        history_samples, horizon_samples, stride_samples
    )
    if block_count > plan.max_chain_rows.shape[1]:
        raise ValueError("HistoryPlan is shorter than the requested history")
    chain = plan.max_chain_rows[:, -block_count:]
    selected = blocks[chain]
    # [anchor, block, channel, time] -> [anchor, channel, block * time]
    return selected.transpose(0, 2, 1, 3).reshape(
        len(plan.anchor_rows), blocks.shape[1], int(history_samples)
    )


def make_history_input(
    extracted: Mapping[str, np.ndarray],
    plan: HistoryPlan,
    name: str,
    history_samples: int,
    horizon_samples: int,
    stride_samples: int,
) -> dict[str, np.ndarray]:
    """Materialize one classifier input while preserving anchor labels/ids."""

    return make_block_history_input(
        extracted=extracted,
        plan=plan,
        source_key="residual",
        name=name,
        history_samples=history_samples,
        horizon_samples=horizon_samples,
        stride_samples=stride_samples,
    )


def make_block_history_input(
    extracted: Mapping[str, np.ndarray],
    plan: HistoryPlan,
    source_key: str,
    name: str,
    history_samples: int,
    horizon_samples: int,
    stride_samples: int,
) -> dict[str, np.ndarray]:
    """Build a history from any aligned block representation.

    ``source_key`` may identify uncertainty-standardised residual blocks or
    robust-scaled raw IMU blocks.  Labels and anchor ids are always taken from
    the final 0.5-second block, so changing the representation does not change
    the classification task.
    """

    if source_key not in extracted:
        raise KeyError(f"Missing history source {source_key!r}")
    source_indices = np.asarray(extracted["window_index"], dtype=np.int64)
    if not np.array_equal(source_indices[plan.anchor_rows], plan.anchor_window_indices):
        raise ValueError("HistoryPlan and extracted feature rows are misaligned")
    return {
        name: materialize_nonoverlap_residual_history(
            extracted[source_key],
            plan,
            history_samples,
            horizon_samples,
            stride_samples,
        ),
        "y": np.asarray(extracted["y"])[plan.anchor_rows],
        "window_index": plan.anchor_window_indices,
    }
