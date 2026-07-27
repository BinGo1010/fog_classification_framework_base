"""Causal raw-history materialisation shared by the baseline methods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cnbr_fog.data import Record, RobustChannelScaler, WindowTable


def _history_bounds(
    records: Sequence[Record],
    windows: WindowTable,
    window_index: int,
    history_samples: int,
) -> tuple[int, int, int]:
    window_index = int(window_index)
    history_samples = int(history_samples)
    if history_samples <= 0:
        raise ValueError("history_samples must be positive")
    if window_index < 0 or window_index >= len(windows):
        raise IndexError(f"window_index {window_index} is outside WindowTable")
    record_index = int(windows.record_index[window_index])
    end = int(windows.target_end[window_index])
    start = end - history_samples
    if start < 0:
        raise ValueError(
            f"Window {window_index} has no complete {history_samples}-sample history"
        )
    record = records[record_index]
    if end > len(record.x):
        raise IndexError(f"Window {window_index} ends outside record {record.record_id}")
    if not bool(record.valid[start:end].all()):
        raise ValueError(f"Window {window_index} history contains invalid samples")
    return record_index, start, end


def materialize_history_windows(
    records: Sequence[Record],
    windows: WindowTable,
    window_indices: np.ndarray | Sequence[int],
    history_samples: int,
    scaler: RobustChannelScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return histories as ``[window, channel, time]`` with anchor labels."""

    indices = np.asarray(window_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("window_indices must be one-dimensional")
    if len(indices) == 0:
        channel_count = int(records[0].x.shape[1])
        return (
            np.empty((0, channel_count, int(history_samples)), dtype=np.float32),
            np.empty(0, dtype=np.int8),
            indices,
        )
    if indices.min() < 0 or indices.max() >= len(windows):
        raise IndexError("window_indices contain an index outside WindowTable")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("window_indices must not contain duplicates")
    channel_count = int(records[0].x.shape[1])
    result = np.empty(
        (len(indices), channel_count, int(history_samples)),
        dtype=np.float32,
    )
    for row, window_index in enumerate(indices):
        record_index, start, end = _history_bounds(
            records,
            windows,
            int(window_index),
            history_samples,
        )
        sequence = records[record_index].x[start:end]
        if scaler is not None:
            sequence = scaler.transform(sequence)
        result[row] = np.asarray(sequence, dtype=np.float32).T
    return (
        np.ascontiguousarray(result),
        np.asarray(windows.label[indices], dtype=np.int8),
        indices,
    )


class HistoryWindowDataset(Dataset):
    """Lazy robust-scaled raw histories for CNN-GRU training."""

    def __init__(
        self,
        records: Sequence[Record],
        windows: WindowTable,
        window_indices: np.ndarray | Sequence[int],
        history_samples: int,
        scaler: RobustChannelScaler,
        channel_indices: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        self.records = records
        self.windows = windows
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.history_samples = int(history_samples)
        self.scaler = scaler
        if channel_indices is None:
            channel_indices = np.arange(records[0].x.shape[1], dtype=np.int64)
        self.channel_indices = np.asarray(channel_indices, dtype=np.int64)
        if self.channel_indices.ndim != 1 or len(self.channel_indices) == 0:
            raise ValueError("channel_indices must be a non-empty vector")
        if (
            self.channel_indices.min() < 0
            or self.channel_indices.max() >= records[0].x.shape[1]
        ):
            raise IndexError("channel_indices contain an unavailable channel")
        if len(np.unique(self.channel_indices)) != len(self.channel_indices):
            raise ValueError("channel_indices must not contain duplicates")
        if (
            len(self.window_indices)
            and (
                self.window_indices.min() < 0
                or self.window_indices.max() >= len(windows)
            )
        ):
            raise IndexError("window_indices contain an index outside WindowTable")
        if len(np.unique(self.window_indices)) != len(self.window_indices):
            raise ValueError("window_indices must not contain duplicates")

    def __len__(self) -> int:
        return int(len(self.window_indices))

    def __getitem__(self, item: int):
        window_index = int(self.window_indices[item])
        record_index, start, end = _history_bounds(
            self.records,
            self.windows,
            window_index,
            self.history_samples,
        )
        sequence = self.scaler.transform(self.records[record_index].x[start:end])
        sequence = np.ascontiguousarray(sequence[:, self.channel_indices].T)
        return (
            torch.from_numpy(sequence),
            torch.tensor(int(self.windows.label[window_index]), dtype=torch.long),
            torch.tensor(window_index, dtype=torch.long),
        )
