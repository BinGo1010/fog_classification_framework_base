"""Continuous-record loading and leakage-safe windowing for Daphnet IMU data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    """One contiguous Daphnet segment."""

    record_id: str
    subject_id: str
    run_id: str
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class RobustChannelScaler:
    """Median/IQR scaler fitted on valid non-FOG training samples only."""

    center: np.ndarray
    scale: np.ndarray
    clip: float = 12.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float32) - self.center) / self.scale
        return np.clip(z, -self.clip, self.clip).astype(np.float32, copy=False)

    def as_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "clip": float(self.clip),
        }


@dataclass(frozen=True)
class WindowTable:
    """Compact metadata for record-local context/target windows."""

    record_index: np.ndarray
    start: np.ndarray
    target_start: np.ndarray
    target_end: np.ndarray
    label: np.ndarray
    fog_fraction: np.ndarray
    clean_normal: np.ndarray

    def __len__(self) -> int:
        return int(self.start.size)

    def take(self, indices: np.ndarray | Sequence[int]) -> "WindowTable":
        idx = np.asarray(indices, dtype=np.int64)
        return WindowTable(
            record_index=self.record_index[idx],
            start=self.start[idx],
            target_start=self.target_start[idx],
            target_end=self.target_end[idx],
            label=self.label[idx],
            fog_fraction=self.fog_fraction[idx],
            clean_normal=self.clean_normal[idx],
        )

    def as_metadata(self, records: Sequence[Record]) -> list[dict]:
        rows: list[dict] = []
        for i in range(len(self)):
            rec = records[int(self.record_index[i])]
            rows.append(
                {
                    "window_index": i,
                    "subject_id": rec.subject_id,
                    "record_id": rec.record_id,
                    "run_id": rec.run_id,
                    "window_start": int(self.start[i]),
                    "target_start": int(self.target_start[i]),
                    "target_end_exclusive": int(self.target_end[i]),
                    "fog_fraction": float(self.fog_fraction[i]),
                    "y_true": int(self.label[i]),
                    "clean_normal": bool(self.clean_normal[i]),
                }
            )
        return rows


def _runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield half-open runs of True values."""

    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def valid_signal_mask(
    x: np.ndarray,
    sampling_rate_hz: int,
    flatline_seconds: float = 1.0,
    zero_tolerance: float = 1e-8,
) -> np.ndarray:
    """Mark finite samples and long sensor-triad zero flatlines as invalid.

    The rule is label independent and is fixed before any fold is evaluated.
    Complete Daphnet data contain three independent tri-axial accelerometers.
    A window is invalid if any complete sensor triad has a long all-zero run;
    this catches both the known S10 tail and a missing thigh sensor in one S03
    record without consulting labels or subject ids.
    """

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("x must have shape [time, channel]")
    valid = np.isfinite(x).all(axis=1)
    minimum = max(1, int(round(float(flatline_seconds) * int(sampling_rate_hz))))
    groups = (
        [x[:, start : start + 3] for start in range(0, x.shape[1], 3)]
        if x.shape[1] % 3 == 0
        else [x]
    )
    for group in groups:
        zero = np.max(np.abs(group), axis=1) <= float(zero_tolerance)
        for start, end in _runs(zero):
            if end - start >= minimum:
                valid[start:end] = False
    return valid


class DaphnetTrunkDataset:
    """In-memory view of processed Daphnet continuous IMU records.

    The historical class name is retained for compatibility.  Both the
    trunk-only three-channel export and the complete three-sensor/nine-channel
    export are supported.
    """

    def __init__(
        self,
        root: Path,
        records: list[Record],
        sampling_rate_hz: int,
        channel_names: Sequence[str] | None = None,
    ):
        if not records:
            raise ValueError("records must not be empty")
        channel_counts = {int(record.x.shape[1]) for record in records}
        if len(channel_counts) != 1:
            raise ValueError(f"Mixed channel counts are unsupported: {channel_counts}")
        n_channels = channel_counts.pop()
        if channel_names is None:
            channel_names = tuple(f"channel_{index}" for index in range(n_channels))
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names has {len(channel_names)} entries for {n_channels} channels"
            )
        self.root = Path(root)
        self.records = records
        self.sampling_rate_hz = int(sampling_rate_hz)
        self.channel_names = tuple(str(name) for name in channel_names)
        self.n_channels = n_channels
        self.subjects = sorted({record.subject_id for record in records})

    @classmethod
    def load(
        cls,
        root: str | Path,
        flatline_seconds: float = 1.0,
        zero_tolerance: float = 1e-8,
    ) -> "DaphnetTrunkDataset":
        root = Path(root)
        manifest_path = root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing Daphnet manifest: {manifest_path}")

        schema_path = root / "schema.json"
        schema_channel_names: tuple[str, ...] | None = None
        if schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            channels = schema.get("channels")
            if channels:
                schema_channel_names = tuple(str(channel["name"]) for channel in channels)

        records: list[Record] = []
        sampling_rates: set[int] = set()
        channel_counts: set[int] = set()
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("usable", "true")).strip().lower() not in {"1", "true", "yes"}:
                    continue
                path = root / row["record_path"]
                with np.load(path, allow_pickle=False) as payload:
                    if set(payload.files) != {"x", "y_binary"}:
                        raise ValueError(f"Unexpected arrays in {path}: {payload.files}")
                    x = np.asarray(payload["x"], dtype=np.float32)
                    y = np.asarray(payload["y_binary"], dtype=np.int8)
                if x.ndim != 2 or x.shape[1] < 1 or y.shape != (x.shape[0],):
                    raise ValueError(f"Invalid shapes in {path}: x={x.shape}, y={y.shape}")
                channel_counts.add(int(x.shape[1]))
                if not set(np.unique(y)).issubset({0, 1}):
                    raise ValueError(f"Non-binary labels in {path}")
                expected = int(row["n_samples"])
                if len(x) != expected:
                    raise ValueError(f"Manifest mismatch for {path}: {len(x)} != {expected}")
                fs = int(row["sampling_rate_hz"])
                sampling_rates.add(fs)
                records.append(
                    Record(
                        record_id=row["record_id"],
                        subject_id=row["subject_id"],
                        run_id=row["run_id"],
                        x=x,
                        y=y,
                        valid=valid_signal_mask(x, fs, flatline_seconds, zero_tolerance),
                    )
                )
        if not records:
            raise ValueError(f"No usable records found under {root}")
        if len(sampling_rates) != 1:
            raise ValueError(f"Mixed sampling rates are unsupported: {sampling_rates}")
        if len(channel_counts) != 1:
            raise ValueError(f"Mixed channel counts are unsupported: {channel_counts}")
        n_channels = next(iter(channel_counts))
        if schema_channel_names is not None and len(schema_channel_names) != n_channels:
            raise ValueError(
                f"Schema declares {len(schema_channel_names)} channels, records contain "
                f"{n_channels}"
            )
        return cls(
            root,
            records,
            sampling_rates.pop(),
            channel_names=schema_channel_names,
        )

    def subject_record_indices(self, subject_ids: Iterable[str]) -> np.ndarray:
        selected = set(subject_ids)
        return np.asarray(
            [i for i, record in enumerate(self.records) if record.subject_id in selected],
            dtype=np.int32,
        )

    def fit_scaler(
        self,
        subject_ids: Iterable[str],
        clip: float = 12.0,
    ) -> RobustChannelScaler:
        selected = set(subject_ids)
        chunks = [
            rec.x[(rec.y == 0) & rec.valid]
            for rec in self.records
            if rec.subject_id in selected
        ]
        chunks = [chunk for chunk in chunks if len(chunk)]
        if not chunks:
            raise ValueError("No valid non-FOG samples available to fit the scaler")
        values = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
        center = np.median(values, axis=0)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        scale = (q75 - q25) / 1.349
        fallback = np.std(values, axis=0)
        scale = np.where(scale > 1e-6, scale, fallback)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return RobustChannelScaler(
            center=center.astype(np.float32),
            scale=scale.astype(np.float32),
            clip=float(clip),
        )

    def make_windows(
        self,
        warmup_samples: int,
        target_samples: int,
        stride_samples: int,
        fog_fraction_threshold: float = 0.5,
        normal_guard_samples: int = 32,
    ) -> WindowTable:
        warmup_samples = int(warmup_samples)
        target_samples = int(target_samples)
        stride_samples = int(stride_samples)
        total = warmup_samples + target_samples
        if min(warmup_samples, target_samples, stride_samples) <= 0:
            raise ValueError("warmup, target and stride must be positive")
        if not 0.0 < fog_fraction_threshold <= 1.0:
            raise ValueError("fog_fraction_threshold must be in (0, 1]")

        record_index: list[int] = []
        starts: list[int] = []
        target_starts: list[int] = []
        target_ends: list[int] = []
        labels: list[int] = []
        fractions: list[float] = []
        clean: list[bool] = []

        for rec_idx, record in enumerate(self.records):
            n = len(record.y)
            if n < total:
                continue
            invalid_prefix = np.r_[0, np.cumsum(~record.valid, dtype=np.int64)]
            fog_prefix = np.r_[0, np.cumsum(record.y == 1, dtype=np.int64)]
            for start in range(0, n - total + 1, stride_samples):
                end = start + total
                if invalid_prefix[end] - invalid_prefix[start] != 0:
                    continue
                target_start = start + warmup_samples
                target_end = end
                fog_count = int(fog_prefix[target_end] - fog_prefix[target_start])
                fraction = fog_count / float(target_samples)
                guard_start = max(0, start - int(normal_guard_samples))
                guard_end = min(n, end + int(normal_guard_samples))
                is_clean = bool(fog_prefix[guard_end] - fog_prefix[guard_start] == 0)
                record_index.append(rec_idx)
                starts.append(start)
                target_starts.append(target_start)
                target_ends.append(target_end)
                labels.append(int(fraction >= fog_fraction_threshold))
                fractions.append(fraction)
                clean.append(is_clean)

        return WindowTable(
            record_index=np.asarray(record_index, dtype=np.int32),
            start=np.asarray(starts, dtype=np.int32),
            target_start=np.asarray(target_starts, dtype=np.int32),
            target_end=np.asarray(target_ends, dtype=np.int32),
            label=np.asarray(labels, dtype=np.int8),
            fog_fraction=np.asarray(fractions, dtype=np.float32),
            clean_normal=np.asarray(clean, dtype=bool),
        )

    def window_indices_for_subjects(
        self,
        windows: WindowTable,
        subject_ids: Iterable[str],
        clean_normal_only: bool = False,
    ) -> np.ndarray:
        rec_idx = set(self.subject_record_indices(subject_ids).tolist())
        mask = np.fromiter(
            (int(index) in rec_idx for index in windows.record_index),
            dtype=bool,
            count=len(windows),
        )
        if clean_normal_only:
            mask &= windows.clean_normal
        return np.flatnonzero(mask)


class SequenceWindowDataset(Dataset):
    """Return robust-scaled [channel,time] windows and labels."""

    def __init__(
        self,
        records: Sequence[Record],
        windows: WindowTable,
        indices: np.ndarray | Sequence[int],
        scaler: RobustChannelScaler,
    ):
        self.records = records
        self.windows = windows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scaler = scaler

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        window_index = int(self.indices[item])
        rec_index = int(self.windows.record_index[window_index])
        start = int(self.windows.start[window_index])
        end = int(self.windows.target_end[window_index])
        sequence = self.scaler.transform(self.records[rec_index].x[start:end])
        x = torch.from_numpy(np.ascontiguousarray(sequence.T))
        y = torch.tensor(int(self.windows.label[window_index]), dtype=torch.long)
        return x, y, torch.tensor(window_index, dtype=torch.long)


# Preferred generic name for new code.  Keep DaphnetTrunkDataset import
# compatibility for the already completed trunk-only experiments.
DaphnetDataset = DaphnetTrunkDataset
