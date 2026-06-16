from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import register_dataset


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _channel_parts(name: str):
    parts = name.split("_")
    if len(parts) != 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def resolve_channel_indices(
    sensor_columns,
    channel_indices=None,
    channel_names=None,
    imu_positions=None,
    sensor_types=None,
    axes=None,
):
    sensor_columns = [str(c) for c in sensor_columns]
    if channel_names:
        name_to_idx = {name: idx for idx, name in enumerate(sensor_columns)}
        missing = [name for name in channel_names if name not in name_to_idx]
        if missing:
            raise KeyError(f"Unknown channel name(s): {missing}. Available: {sensor_columns}")
        return [name_to_idx[name] for name in channel_names]

    if channel_indices:
        indices = [int(idx) for idx in channel_indices]
        bad = [idx for idx in indices if idx < 0 or idx >= len(sensor_columns)]
        if bad:
            raise IndexError(f"Channel index out of range: {bad}. Num channels: {len(sensor_columns)}")
        return indices

    imu_positions = set(_as_list(imu_positions) or [])
    sensor_types = set(_as_list(sensor_types) or [])
    axes = set(_as_list(axes) or [])

    indices = []
    for idx, name in enumerate(sensor_columns):
        position, sensor_type, axis = _channel_parts(name)
        if imu_positions and position not in imu_positions:
            continue
        if sensor_types and sensor_type not in sensor_types:
            continue
        if axes and axis not in axes:
            continue
        indices.append(idx)

    if not indices:
        raise ValueError(
            "No channels selected. Check channel_names/channel_indices/imu_positions/sensor_types/axes."
        )
    return indices


@register_dataset("ChannelSubsetNPZDataset")
class ChannelSubsetNPZDataset(Dataset):
    """NPZ time-series dataset with configurable channel selection.

    Expected npz keys:
      X: [N, C, T] or [N, T, C]
      y: [N]
      sensor_columns: [C] channel names such as ankleL_acc_x
    """

    def __init__(
        self,
        file_path: str | Path,
        x_key: str = "X",
        y_key: str = "y",
        input_format: str = "NCT",
        normalize: str = "none",
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        metadata_keys: Optional[list[str]] = None,
        channel_indices: Optional[list[int]] = None,
        channel_names: Optional[list[str]] = None,
        imu_positions: Optional[list[str]] = None,
        sensor_types: Optional[list[str]] = None,
        axes: Optional[list[str]] = None,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        data = np.load(self.file_path, allow_pickle=True)
        X = data[x_key].astype("float32")
        y = data[y_key].astype("int64")
        if input_format.upper() == "NTC":
            X = np.transpose(X, (0, 2, 1))
        if X.ndim != 3:
            raise ValueError(f"X must be 3D [N,C,T], got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if "sensor_columns" not in data:
            raise KeyError(f"'sensor_columns' key is required for channel selection: {self.file_path}")

        self.sensor_columns = np.asarray(data["sensor_columns"]).astype(str)
        self.channel_indices = resolve_channel_indices(
            self.sensor_columns,
            channel_indices=channel_indices,
            channel_names=channel_names,
            imu_positions=imu_positions,
            sensor_types=sensor_types,
            axes=axes,
        )
        self.selected_channel_names = self.sensor_columns[self.channel_indices].tolist()
        X = X[:, self.channel_indices, :]

        if not np.isfinite(X).all():
            raise ValueError(f"X contains NaN or infinite values: {self.file_path}")
        self.mean, self.std = mean, std
        if normalize == "zscore":
            if self.mean is None or self.std is None:
                self.mean = X.mean(axis=(0, 2), keepdims=True)
                self.std = X.std(axis=(0, 2), keepdims=True) + 1e-8
            X = (X - self.mean) / self.std
        elif normalize not in ("none", None):
            raise ValueError("normalize must be 'none' or 'zscore'")

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.num_channels = int(X.shape[1])
        self.metadata = {}
        for key in metadata_keys or []:
            if key not in data:
                raise KeyError(f"Metadata key '{key}' not found in {self.file_path}")
            value = data[key]
            if len(value) != len(y):
                raise ValueError(f"Metadata key '{key}' length mismatch: {len(value)} vs {len(y)}")
            if np.issubdtype(value.dtype, np.number):
                self.metadata[key] = torch.from_numpy(value)
            else:
                self.metadata[key] = value

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.X[idx], "y": self.y[idx], "index": idx}
        for key, value in self.metadata.items():
            item[key] = value[idx]
        return item
