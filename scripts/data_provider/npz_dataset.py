from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from .registry import register_dataset


@register_dataset("NPZTimeSeriesDataset")
class NPZTimeSeriesDataset(Dataset):
    """Window-level time-series classification dataset.

    Expected npz keys:
      X: [N, C, T] or [N, T, C]
      y: [N]
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
