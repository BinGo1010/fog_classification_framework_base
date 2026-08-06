#!/usr/bin/env python3
"""Run the frozen single-subject CA-SupCon experiment on processed_CA_pure.

Protocol: CA-SUPCON-SUBJECT-V1

Each invocation owns exactly one subject and one visible GPU.  The seven-GPU
launcher starts seven independent invocations, which is preferable to DDP for
this within-subject experiment because no gradients or samples may cross
subjects.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnbr_fog.evaluation import binary_metrics, choose_threshold  # noqa: E402


PROTOCOL_ID = "CA-SUPCON-SUBJECT-V1"
FORMAL_SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
SPLITS = ("train", "validation", "test")
METHODS = ("S0", "S1", "S2", "S3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    """Convert numpy/torch values and non-finite floats to strict JSON."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class RobustScale:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.center[None, :, None]) / self.scale[None, :, None]).astype(
            np.float32
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": "median and IQR fitted on training windows only",
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


def fit_robust_scale(x_train: np.ndarray) -> RobustScale:
    """Match RobustScaler's default median/IQR definition, channel-wise."""

    if x_train.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {x_train.shape}")
    flattened = x_train.transpose(0, 2, 1).reshape(-1, x_train.shape[1]).astype(np.float64)
    center = np.median(flattened, axis=0)
    q25, q75 = np.percentile(flattened, (25.0, 75.0), axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return RobustScale(center.astype(np.float32), scale.astype(np.float32))


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def _event_sampling_id(row: pd.Series) -> str:
    event_id = str(row.get("overlapping_event_ids", "")).strip()
    if event_id and event_id.lower() != "nan":
        return f"{row['record_id']}:fog_event:{event_id}"
    return str(row["group_id"])


def add_sampling_groups(metadata: pd.DataFrame, segment_seconds: int = 20) -> pd.DataFrame:
    """Add event/segment IDs without altering the frozen data split.

    Pure FoG windows use original event IDs.  Pure Non-FoG windows use
    deterministic 20-s virtual segments within the already frozen group.  The
    virtual segment is a sampling device only; it never moves a window between
    train/validation/test.
    """

    result = metadata.copy()
    sampling_groups: list[str] = [""] * len(result)
    for _, indices in result.groupby("group_id", sort=False).groups.items():
        indices = list(indices)
        group_rows = result.loc[indices]
        group_start = int(group_rows["start_index"].min())
        for index, row in group_rows.iterrows():
            if int(row["y_binary"]) == 1:
                value = _event_sampling_id(row)
            else:
                offset = int(row["start_index"]) - group_start
                segment = offset // (segment_seconds * 64)
                value = f"{row['group_id']}:nonfog_{segment_seconds}s:{segment:03d}"
            sampling_groups[result.index.get_loc(index)] = value
    result["sampling_group_id"] = sampling_groups
    return result


def load_subject_data(data_dir: Path, subject: str) -> tuple[dict[str, SplitData], RobustScale, dict[str, Any]]:
    manifest_path = data_dir / "ca_window_manifest.csv"
    protocol_path = data_dir / "ca_protocol.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing protocol: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as handle:
        dataset_protocol = json.load(handle)
    required_protocol = {
        "window_samples": 128,
        "stride_samples": 64,
        "source_split_preserved": True,
        "random_resplit": False,
    }
    for key, expected in required_protocol.items():
        if dataset_protocol.get(key) != expected:
            raise ValueError(f"processed_CA_pure protocol mismatch: {key}={dataset_protocol.get(key)!r}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required_columns = {
        "window_id",
        "subject_id",
        "subject_scope",
        "record_id",
        "ca_split",
        "y_binary",
        "group_id",
        "start_index",
        "end_index_exclusive",
        "start_time_sec",
        "end_time_sec",
        "fog_samples_in_2s",
        "overlapping_event_ids",
        "pure_window",
    }
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    subject_rows = manifest.loc[manifest["subject_id"] == subject].copy()
    if subject_rows.empty:
        raise ValueError(f"Subject {subject} not found in {manifest_path}")
    if not subject_rows["pure_window"].astype(bool).all():
        raise ValueError("Non-pure windows found in processed_CA_pure manifest")
    valid_purity = (
        ((subject_rows["y_binary"] == 0) & (subject_rows["fog_samples_in_2s"] == 0))
        | ((subject_rows["y_binary"] == 1) & (subject_rows["fog_samples_in_2s"] == 128))
    )
    if not bool(valid_purity.all()):
        raise ValueError("Pure-window labels disagree with fog_samples_in_2s")
    if set(subject_rows["ca_split"].unique()) != set(SPLITS):
        raise ValueError(f"{subject} does not contain exactly {SPLITS}")

    records: dict[str, np.ndarray] = {}
    raw_splits: dict[str, SplitData] = {}
    audit_splits: dict[str, Any] = {}
    for split in SPLITS:
        rows = subject_rows.loc[subject_rows["ca_split"] == split].copy().reset_index(drop=True)
        windows: list[np.ndarray] = []
        for row in rows.itertuples(index=False):
            record_id = str(row.record_id)
            if record_id not in records:
                record_path = data_dir / "records" / f"{record_id}.npz"
                if not record_path.is_file():
                    raise FileNotFoundError(f"Missing continuous record: {record_path}")
                with np.load(record_path, allow_pickle=False) as payload:
                    records[record_id] = payload["x"].astype(np.float32, copy=False)
            start = int(row.start_index)
            end = int(row.end_index_exclusive)
            window = records[record_id][start:end]
            if window.shape != (128, 9):
                raise ValueError(f"Bad window {row.window_id}: {window.shape}")
            windows.append(window.T.copy())
        x = np.stack(windows).astype(np.float32)
        y = rows["y_binary"].to_numpy(dtype=np.int64)
        if np.unique(y).size != 2:
            raise ValueError(f"{subject}/{split} lacks one class; CA-SupCon experiment is undefined")
        rows = add_sampling_groups(rows)
        raw_splits[split] = SplitData(x=x, y=y, metadata=rows)
        audit_splits[split] = {
            "n_windows": int(len(rows)),
            "n_fog": int(y.sum()),
            "n_nonfog": int((y == 0).sum()),
            "fog_fraction": float(y.mean()),
            "n_fog_events": int(rows.loc[y == 1, "sampling_group_id"].nunique()),
            "n_nonfog_segments": int(rows.loc[y == 0, "sampling_group_id"].nunique()),
            "n_frozen_groups": int(rows["group_id"].nunique()),
        }

    frozen_groups = {
        split: set(raw_splits[split].metadata["group_id"].astype(str)) for split in SPLITS
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = frozen_groups[left] & frozen_groups[right]
        if overlap:
            raise ValueError(f"Frozen group leakage between {left}/{right}: {sorted(overlap)[:3]}")

    scaler = fit_robust_scale(raw_splits["train"].x)
    splits = {
        key: SplitData(scaler.transform(value.x), value.y, value.metadata)
        for key, value in raw_splits.items()
    }
    audit = {
        "protocol_id": PROTOCOL_ID,
        "subject_id": subject,
        "subject_scope": str(subject_rows["subject_scope"].iloc[0]),
        "manifest": str(manifest_path.resolve()),
        "dataset_protocol": dataset_protocol,
        "split_frozen": True,
        "random_resplit": False,
        "scaler_fit_split": "train",
        "splits": audit_splits,
    }
    return splits, scaler, audit


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, split: SplitData) -> None:
        self.x = torch.from_numpy(split.x)
        self.y = torch.from_numpy(split.y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.x[index], self.y[index], index


class EventAwareBalancedBatchSampler(BatchSampler):
    """Hierarchical class -> event/segment -> window sampler.

    Every batch is 1:1.  With batch_size=32, each class contributes 16 windows
    from at least four groups when four groups exist, with no group contributing
    more than four windows.  Small groups are sampled with replacement.
    """

    def __init__(
        self,
        labels: Sequence[int] | np.ndarray,
        groups: Sequence[str],
        batch_size: int = 32,
        max_windows_per_group: int = 4,
        min_groups_per_class: int = 4,
        seed: int = 0,
        steps_per_epoch: int | None = None,
    ) -> None:
        if batch_size < 8 or batch_size % 2:
            raise ValueError("Balanced batch_size must be even and >= 8")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = np.asarray(groups, dtype=str)
        if len(self.labels) != len(self.groups):
            raise ValueError("labels/groups length mismatch")
        if set(np.unique(self.labels)) != {0, 1}:
            raise ValueError("Balanced sampler requires both classes")
        self.batch_size = int(batch_size)
        self.per_class = self.batch_size // 2
        self.max_windows_per_group = int(max_windows_per_group)
        self.min_groups_per_class = int(min_groups_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.by_class: dict[int, dict[str, np.ndarray]] = {}
        for label in (0, 1):
            label_indices = np.flatnonzero(self.labels == label)
            label_groups: dict[str, np.ndarray] = {}
            for group in np.unique(self.groups[label_indices]):
                label_groups[str(group)] = label_indices[self.groups[label_indices] == group]
            self.by_class[label] = label_groups
            if len(label_groups) < self.min_groups_per_class:
                warnings.warn(
                    f"Class {label} has only {len(label_groups)} sampling groups; "
                    f"template target is {self.min_groups_per_class}",
                    RuntimeWarning,
                )
        required_groups = math.ceil(self.per_class / self.max_windows_per_group)
        self.groups_per_class = max(self.min_groups_per_class, required_groups)
        majority = max(int((self.labels == 0).sum()), int((self.labels == 1).sum()))
        self.steps_per_epoch = int(steps_per_epoch or math.ceil(majority / self.per_class))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_class(self, label: int, rng: np.random.Generator) -> list[int]:
        mapping = self.by_class[label]
        names = np.asarray(sorted(mapping), dtype=object)
        n_groups = min(self.groups_per_class, len(names))
        chosen = rng.choice(names, size=n_groups, replace=False).tolist()
        base, remainder = divmod(self.per_class, n_groups)
        quotas = [base + (1 if i < remainder else 0) for i in range(n_groups)]
        if max(quotas) > self.max_windows_per_group:
            raise RuntimeError(
                f"Cannot satisfy max_windows_per_group={self.max_windows_per_group} "
                f"with only {n_groups} groups for class {label}"
            )
        result: list[int] = []
        for group, quota in zip(chosen, quotas):
            candidates = mapping[str(group)]
            selected = rng.choice(candidates, size=quota, replace=len(candidates) < quota)
            result.extend(int(item) for item in selected)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        for _ in range(self.steps_per_epoch):
            batch = self._sample_class(0, rng) + self._sample_class(1, rng)
            rng.shuffle(batch)
            yield batch

    def audit(self) -> dict[str, Any]:
        return {
            "hierarchy": "class -> event_or_nonfog_segment -> window",
            "batch_size": self.batch_size,
            "windows_per_class": self.per_class,
            "max_windows_per_group": self.max_windows_per_group,
            "target_min_groups_per_class": self.min_groups_per_class,
            "actual_groups": {str(label): len(groups) for label, groups in self.by_class.items()},
            "steps_per_epoch": self.steps_per_epoch,
        }


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def natural_loader(
    split: SplitData,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        WindowDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_init if num_workers else None,
        persistent_workers=bool(num_workers),
    )


def balanced_loader(
    split: SplitData,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, EventAwareBalancedBatchSampler]:
    sampler = EventAwareBalancedBatchSampler(
        split.y,
        split.metadata["sampling_group_id"].astype(str).tolist(),
        batch_size=batch_size,
        seed=seed,
    )
    loader = DataLoader(
        WindowDataset(split),
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init if num_workers else None,
        persistent_workers=bool(num_workers),
    )
    return loader, sampler


class TCNMResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNMEncoder(nn.Module):
    output_dim = 96

    def __init__(self, in_channels: int = 9, hidden: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(TCNMResidualBlock(hidden, dilation) for dilation in (1, 2, 4, 8))
        )
        self.output_dim = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        return torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)


class TCNMClassifier(nn.Module):
    def __init__(self, encoder: TCNMEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or TCNMEncoder()
        self.classifier = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x)).squeeze(1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(h), dim=1)


class FrozenEncoderClassifier(nn.Module):
    def __init__(self, encoder: TCNMEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.classifier = nn.Linear(self.encoder.output_dim, 1)

    def train(self, mode: bool = True) -> "FrozenEncoderClassifier":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x)
        return self.classifier(h).squeeze(1)


class ClassAwareSupConLoss(nn.Module):
    """Single-view supervised contrastive loss averaged equally by class."""

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or labels.ndim != 1 or len(z) != len(labels):
            raise ValueError("Expected z=[B,D], labels=[B]")
        z = nn.functional.normalize(z, dim=1)
        labels = labels.view(-1)
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        valid_mask = ~self_mask
        exp_logits = torch.exp(logits) * valid_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1)
        if bool((positive_count == 0).any()):
            raise ValueError("Every SupCon anchor needs at least one same-class positive")
        anchor_loss = -(log_prob * positive_mask).sum(dim=1) / positive_count
        class_losses = [anchor_loss[labels == label].mean() for label in torch.unique(labels)]
        if len(class_losses) != 2:
            raise ValueError("CA-SupCon batch must contain both classes")
        return torch.stack(class_losses).mean()


def amp_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    scaler: Any | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    losses = 0.0
    count = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for x_cpu, y_cpu, index_cpu in loader:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.float().to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with amp_context(amp):
                logits = model(x)
                loss = criterion(logits, y)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
            scaler.step(optimizer)
            scaler.update()
        losses += float(loss.detach()) * len(x)
        count += len(x)
        truths.append(y_cpu.numpy().astype(np.int8))
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        indices.append(index_cpu.numpy().astype(np.int64))
    return losses / max(count, 1), np.concatenate(truths), np.concatenate(probabilities), np.concatenate(indices)


def score_pr_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if np.unique(y).size == 2 else -math.inf


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
    train_sampler: EventAwareBalancedBatchSampler | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scaler = make_grad_scaler(amp)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, _, _, _ = classifier_epoch(
            model, train_loader, device, optimizer, amp, scaler
        )
        val_loss, val_y, val_probability, _ = classifier_epoch(
            model, validation_loader, device, None, amp
        )
        val_pr_auc = score_pr_auc(val_y, val_probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_pr_auc": val_pr_auc,
            }
        )
        if val_pr_auc > best_score + 1e-8:
            best_score = val_pr_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No valid classifier checkpoint")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_validation_pr_auc": best_score}


def clone_encoder_from_state(state: dict[str, torch.Tensor], device: torch.device) -> TCNMEncoder:
    encoder = TCNMEncoder()
    encoder.load_state_dict(state)
    return encoder.to(device)


def linear_probe(
    encoder_state: dict[str, torch.Tensor],
    splits: dict[str, SplitData],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    balanced: bool,
    max_epochs: int,
    patience: int,
) -> tuple[FrozenEncoderClassifier, list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed, args.deterministic)
    model = FrozenEncoderClassifier(clone_encoder_from_state(encoder_state, device)).to(device)
    if balanced:
        train_loader, sampler = balanced_loader(
            splits["train"], args.batch_size, seed, args.num_workers
        )
    else:
        train_loader = natural_loader(
            splits["train"], args.batch_size, True, seed, args.num_workers
        )
        sampler = None
    validation_loader = natural_loader(
        splits["validation"], args.batch_size, False, seed, args.num_workers
    )
    model, history, selection = train_classifier(
        model,
        train_loader,
        validation_loader,
        device,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp=args.amp and device.type == "cuda",
        train_sampler=sampler,
    )
    selection["sampling"] = "event_aware_1_to_1" if balanced else "natural"
    return model, history, selection


def train_supcon_candidate(
    temperature: float,
    splits: dict[str, SplitData],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed, args.deterministic)
    encoder = TCNMEncoder().to(device)
    projection = ProjectionHead(encoder.output_dim).to(device)
    criterion = ClassAwareSupConLoss(temperature)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    train_loader, sampler = balanced_loader(
        splits["train"], args.batch_size, seed, args.num_workers
    )
    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_probe: dict[str, Any] = {}
    probe_epochs = set(range(args.probe_every, args.supcon_epochs + 1, args.probe_every))
    probe_epochs.add(args.supcon_epochs)
    for epoch in range(1, args.supcon_epochs + 1):
        encoder.train()
        projection.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        total_n = 0
        for x_cpu, y_cpu, _ in train_loader:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.amp and device.type == "cuda"):
                z = projection(encoder(x))
                loss = criterion(z, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projection.parameters()), 5.0
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(x)
            total_n += len(x)
        row: dict[str, Any] = {
            "epoch": epoch,
            "supcon_train_loss": total_loss / max(total_n, 1),
            "temperature": temperature,
        }
        if epoch in probe_epochs:
            current_state = {
                key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
            }
            probe_model, _, probe_selection = linear_probe(
                current_state,
                splits,
                args,
                device,
                seed=seed + 50_000 + epoch,
                balanced=False,
                max_epochs=args.selection_probe_epochs,
                patience=args.selection_probe_patience,
            )
            val_loader = natural_loader(
                splits["validation"], args.batch_size, False, seed, args.num_workers
            )
            _, val_y, val_probability, _ = classifier_epoch(
                probe_model, val_loader, device, None, args.amp and device.type == "cuda"
            )
            threshold, val_metrics = choose_threshold(val_y, val_probability)
            key = (float(val_metrics["auprc"] or -math.inf), float(val_metrics["f1"] or -math.inf))
            row.update(
                {
                    "probe_validation_pr_auc": val_metrics["auprc"],
                    "probe_validation_f1": val_metrics["f1"],
                    "probe_validation_threshold": threshold,
                }
            )
            if key > best_key:
                best_key = key
                best_state = current_state
                best_probe = {
                    "temperature": temperature,
                    "encoder_epoch": epoch,
                    "validation_pr_auc": val_metrics["auprc"],
                    "validation_f1": val_metrics["f1"],
                    "validation_threshold": threshold,
                    "linear_probe": probe_selection,
                }
            del probe_model
        history.append(row)
    if best_state is None:
        raise RuntimeError("No SupCon encoder selected by validation linear probe")
    best_probe["sampler_audit"] = sampler.audit()
    return best_state, history, best_probe


@torch.no_grad()
def extract_features(
    encoder: TCNMEncoder, loader: DataLoader, device: torch.device, amp: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for x_cpu, y_cpu, index_cpu in loader:
        x = x_cpu.to(device, non_blocking=True)
        with amp_context(amp):
            h = encoder(x)
        features.append(h.float().cpu().numpy())
        labels.append(y_cpu.numpy().astype(np.int8))
        indices.append(index_cpu.numpy().astype(np.int64))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(indices)


def _mean_pairwise_distance(x: np.ndarray, limit: int = 500) -> float | None:
    if len(x) < 2:
        return None
    if len(x) > limit:
        selection = np.linspace(0, len(x) - 1, limit, dtype=int)
        x = x[selection]
    delta = x[:, None, :] - x[None, :, :]
    distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
    upper = distance[np.triu_indices(len(x), k=1)]
    return float(upper.mean())


def representation_diagnostics(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    nonfog = features[labels == 0]
    fog = features[labels == 1]
    intra_nonfog = _mean_pairwise_distance(nonfog)
    intra_fog = _mean_pairwise_distance(fog)
    center_distance = float(np.linalg.norm(nonfog.mean(axis=0) - fog.mean(axis=0)))
    intra_values = [value for value in (intra_nonfog, intra_fog) if value is not None]
    mean_intra = float(np.mean(intra_values)) if intra_values else None
    return {
        "split": "test",
        "feature": "TCN-M pooled h (not projection z)",
        "fog_intra_class_distance": intra_fog,
        "nonfog_intra_class_distance": intra_nonfog,
        "class_center_distance": center_distance,
        "between_to_within_ratio": center_distance / mean_intra if mean_intra else None,
    }


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def event_metrics(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    rows = metadata.copy().reset_index(drop=True)
    rows["label"] = labels.astype(int)
    rows["probability"] = probabilities
    rows["prediction"] = (probabilities >= threshold).astype(int)
    positive = rows.loc[rows["label"] == 1].copy()
    positive["event_eval_id"] = positive.apply(_event_sampling_id, axis=1)
    detected = 0
    latencies: list[float] = []
    event_rows: list[dict[str, Any]] = []
    for event_id, group in positive.groupby("event_eval_id", sort=False):
        predicted = group.loc[group["prediction"] == 1]
        is_detected = not predicted.empty
        detected += int(is_detected)
        latency = None
        if is_detected:
            latency = float(predicted["start_time_sec"].min() - group["start_time_sec"].min())
            latencies.append(latency)
        event_rows.append(
            {
                "event_id": str(event_id),
                "detected": is_detected,
                "latency_sec": latency,
                "n_pure_fog_windows": int(len(group)),
            }
        )

    nonfog = rows.loc[rows["label"] == 0].copy()
    false_positive = nonfog.loc[nonfog["prediction"] == 1]
    false_positive_episodes = 0
    for _, group in false_positive.groupby("record_id", sort=False):
        starts = sorted(group["start_time_sec"].astype(float).tolist())
        previous = None
        for start in starts:
            if previous is None or start - previous > 1.01:
                false_positive_episodes += 1
            previous = start
    exposure_seconds = 0.0
    for _, group in nonfog.groupby("record_id", sort=False):
        exposure_seconds += _union_duration(
            zip(group["start_time_sec"], group["end_time_sec"])
        )
    exposure_hours = exposure_seconds / 3600.0
    total_events = int(positive["event_eval_id"].nunique())
    return {
        "event_definition": "record_id + original overlapping_event_ids",
        "total_fog_events_with_pure_windows": total_events,
        "detected_fog_events": detected,
        "fog_event_sensitivity": detected / total_events if total_events else None,
        "mean_detection_latency_sec": float(np.mean(latencies)) if latencies else None,
        "false_positive_episodes": false_positive_episodes,
        "nonfog_exposure_hours_unique_union": exposure_hours,
        "false_alarms_per_hour": false_positive_episodes / exposure_hours if exposure_hours else None,
        "false_positive_windows_per_hour": len(false_positive) / exposure_hours if exposure_hours else None,
        "events": event_rows,
    }


def save_history_plot(history: list[dict[str, Any]], path: Path, title: str) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for column in ("train_loss", "validation_loss", "supcon_train_loss"):
        if column in frame and frame[column].notna().any():
            axis.plot(frame["epoch"], frame[column], label=column)
    probe_col = "probe_validation_pr_auc"
    if probe_col in frame and frame[probe_col].notna().any():
        second = axis.twinx()
        second.scatter(frame["epoch"], frame[probe_col], color="#B24A3B", label=probe_col)
        second.set_ylabel("Linear-probe validation PR-AUC")
        second.set_ylim(0.0, 1.02)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_evaluation_plots(
    method_dir: Path,
    method: str,
    split: SplitData,
    labels: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    features: np.ndarray,
) -> None:
    ordered_probability = np.empty(len(split.y), dtype=np.float64)
    ordered_probability[indices] = probabilities
    labels = split.y
    predictions = (ordered_probability >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(labels, ordered_probability)
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    axis.plot(recall, precision, color="#276FBF")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.set_title(f"{method} test PR curve")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(method_dir / "test_pr_curve.png", dpi=180)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, ordered_probability)
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    axis.plot(false_positive_rate, true_positive_rate, color="#2A9D6F")
    axis.plot([0, 1], [0, 1], "--", color="0.6")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axis.set_title(f"{method} test ROC curve")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(method_dir / "test_roc_curve.png", dpi=180)
    plt.close(fig)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(4.8, 4.4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks((0, 1), ("Non-FoG", "FoG"))
    axis.set_yticks((0, 1), ("Non-FoG", "FoG"))
    axis.set(xlabel="Predicted", ylabel="True", title=f"{method} test confusion matrix")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(method_dir / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 4.5))
    bins = np.linspace(0, 1, 31)
    axis.hist(ordered_probability[labels == 0], bins=bins, alpha=0.65, label="Non-FoG")
    axis.hist(ordered_probability[labels == 1], bins=bins, alpha=0.65, label="FoG")
    axis.axvline(threshold, color="black", linestyle="--", label=f"val threshold={threshold:.2f}")
    axis.set(xlabel="Predicted FoG probability", ylabel="Windows", title=f"{method} probability distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(method_dir / "test_probability_distribution.png", dpi=180)
    plt.close(fig)

    if len(features) >= 5:
        sample_indices = np.arange(len(features))
        if len(sample_indices) > 1500:
            rng = np.random.default_rng(0)
            sample_indices = np.sort(rng.choice(sample_indices, 1500, replace=False))
        perplexity = max(2, min(30, (len(sample_indices) - 1) // 3))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=0,
        ).fit_transform(features[sample_indices])
        fig, axis = plt.subplots(figsize=(6.0, 5.2))
        for label, name, color in ((0, "Non-FoG", "#4C78A8"), (1, "FoG", "#E45756")):
            mask = labels[sample_indices] == label
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=16, alpha=0.72, label=name, color=color)
        axis.set_title(f"{method} test TCN-M features (t-SNE)")
        axis.legend()
        axis.set_xticks([])
        axis.set_yticks([])
        fig.tight_layout()
        fig.savefig(method_dir / "test_embedding_tsne.png", dpi=180)
        plt.close(fig)

    metadata = split.metadata.copy().reset_index(drop=True)
    metadata["probability"] = ordered_probability
    metadata["prediction"] = predictions
    positive = metadata.loc[metadata["y_binary"] == 1]
    if not positive.empty:
        chosen_group = positive["group_id"].value_counts().index[0]
        timeline = metadata.loc[metadata["group_id"] == chosen_group].sort_values("start_time_sec")
        fig, axis = plt.subplots(figsize=(8.0, 4.2))
        axis.plot(timeline["start_time_sec"], timeline["probability"], marker="o", label="FoG probability")
        axis.step(timeline["start_time_sec"], timeline["y_binary"], where="mid", label="Pure-window label", alpha=0.7)
        axis.axhline(threshold, color="black", linestyle="--", label="Validation threshold")
        axis.set(xlabel="Time (s)", ylabel="Value", ylim=(-0.03, 1.03), title=f"{method} representative FoG event cluster")
        axis.legend(loc="best")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(method_dir / "test_representative_fog_timeline.png", dpi=180)
        plt.close(fig)

    false_positive = np.flatnonzero((labels == 0) & (predictions == 1))
    if len(false_positive):
        chosen = int(false_positive[np.argmax(ordered_probability[false_positive])])
        time_axis = np.arange(split.x.shape[-1]) / 64.0
        fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.8), sharex=True)
        for imu, axis in enumerate(axes):
            for channel in range(3):
                axis.plot(time_axis, split.x[chosen, imu * 3 + channel], linewidth=0.9, label=f"axis {channel + 1}")
            axis.set_ylabel(f"IMU {imu + 1}")
            axis.legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{method} strongest false-positive Non-FoG, p={ordered_probability[chosen]:.3f}")
        fig.tight_layout()
        fig.savefig(method_dir / "test_false_positive_case.png", dpi=180)
        plt.close(fig)


def evaluate_method(
    method: str,
    model: nn.Module,
    splits: dict[str, SplitData],
    args: argparse.Namespace,
    device: torch.device,
    method_dir: Path,
    training_history: list[dict[str, Any]],
    training_selection: dict[str, Any],
    scaler: RobustScale,
    seed: int,
) -> dict[str, Any]:
    validation_loader = natural_loader(
        splits["validation"], args.batch_size, False, seed, args.num_workers
    )
    test_loader = natural_loader(splits["test"], args.batch_size, False, seed, args.num_workers)
    _, val_y, val_probability, val_indices = classifier_epoch(
        model, validation_loader, device, None, args.amp and device.type == "cuda"
    )
    threshold, val_metrics = choose_threshold(val_y, val_probability)
    _, test_y, test_probability, test_indices = classifier_epoch(
        model, test_loader, device, None, args.amp and device.type == "cuda"
    )
    test_metrics = binary_metrics(test_y, test_probability, threshold)
    encoder = model.encoder
    test_features, feature_y, feature_indices = extract_features(
        encoder, test_loader, device, args.amp and device.type == "cuda"
    )
    if not np.array_equal(feature_indices, test_indices):
        raise RuntimeError("Feature/prediction evaluation order mismatch")
    representation = representation_diagnostics(test_features, feature_y)
    test_event_metrics = event_metrics(
        splits["test"].metadata.iloc[test_indices].reset_index(drop=True),
        test_y,
        test_probability,
        threshold,
    )
    predictions = pd.concat(
        [
            splits["validation"].metadata.iloc[val_indices].reset_index(drop=True).assign(
                evaluation_split="validation",
                probability=val_probability,
                prediction=(val_probability >= threshold).astype(int),
            ),
            splits["test"].metadata.iloc[test_indices].reset_index(drop=True).assign(
                evaluation_split="test",
                probability=test_probability,
                prediction=(test_probability >= threshold).astype(int),
            ),
        ],
        ignore_index=True,
    )
    method_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(method_dir / "predictions.csv", index=False)
    save_history_plot(training_history, method_dir / "training_history.png", f"{method} training")
    save_evaluation_plots(
        method_dir,
        method,
        splits["test"],
        test_y,
        test_probability,
        test_indices,
        threshold,
        test_features,
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "method": method,
        "seed": seed,
        "threshold_source": "validation only; maximize balanced accuracy, tie by F1 then higher threshold",
        "selected_threshold": threshold,
        "training_selection": training_selection,
        "validation": val_metrics,
        "test": test_metrics,
        "test_event": test_event_metrics,
        "representation": representation,
        "scaler": scaler.to_dict(),
    }
    write_json(method_dir / "metrics.json", payload)
    checkpoint = {
        "protocol_id": PROTOCOL_ID,
        "method": method,
        "seed": seed,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_threshold": threshold,
        "scaler": scaler.to_dict(),
        "training_selection": training_selection,
    }
    torch.save(checkpoint, method_dir / "checkpoint.pt")
    return payload


def run_subject(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    subject_dir = output_root / args.subject
    complete_path = subject_dir / "subject_complete.json"
    if complete_path.exists() and not args.overwrite:
        print(f"[resume] {args.subject} already complete: {complete_path}", flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    splits, robust_scale, data_audit = load_subject_data(data_dir, args.subject)
    subject_dir.mkdir(parents=True, exist_ok=True)
    write_json(subject_dir / "data_audit.json", data_audit)
    print(json.dumps(data_audit["splits"], ensure_ascii=False), flush=True)

    requested_methods = tuple(args.methods)
    all_seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_dir = subject_dir / f"seed_{seed}"
        seed_complete = seed_dir / "seed_complete.json"
        if seed_complete.exists() and not args.overwrite:
            print(f"[resume] {args.subject} seed={seed} already complete", flush=True)
            with seed_complete.open("r", encoding="utf-8") as handle:
                all_seed_results.append(json.load(handle))
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_results: dict[str, Any] = {
            "protocol_id": PROTOCOL_ID,
            "subject_id": args.subject,
            "seed": seed,
            "started_utc": utc_now(),
            "methods": {},
        }

        for method in ("S0", "S1"):
            if method not in requested_methods:
                continue
            print(f"[{args.subject} seed={seed}] training {method}", flush=True)
            set_seed(seed, args.deterministic)
            model = TCNMClassifier().to(device)
            if method == "S0":
                train_loader = natural_loader(
                    splits["train"], args.batch_size, True, seed, args.num_workers
                )
                sampler = None
                sampler_audit = {"sampling": "natural class prevalence"}
            else:
                train_loader, sampler = balanced_loader(
                    splits["train"], args.batch_size, seed, args.num_workers
                )
                sampler_audit = sampler.audit()
            validation_loader = natural_loader(
                splits["validation"], args.batch_size, False, seed, args.num_workers
            )
            model, history, selection = train_classifier(
                model,
                train_loader,
                validation_loader,
                device,
                max_epochs=args.supervised_epochs,
                patience=args.early_stopping_patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                amp=args.amp and device.type == "cuda",
                train_sampler=sampler,
            )
            selection.update(
                {
                    "loss": "unweighted BCEWithLogitsLoss",
                    "sampling": "natural" if method == "S0" else "event_aware_1_to_1",
                    "sampler_audit": sampler_audit,
                }
            )
            seed_results["methods"][method] = evaluate_method(
                method,
                model,
                splits,
                args,
                device,
                seed_dir / method,
                history,
                selection,
                robust_scale,
                seed,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if "S2" in requested_methods or "S3" in requested_methods:
            candidates: list[dict[str, Any]] = []
            candidate_states: dict[float, dict[str, torch.Tensor]] = {}
            candidate_histories: dict[float, list[dict[str, Any]]] = {}
            for temperature in args.temperatures:
                print(
                    f"[{args.subject} seed={seed}] CA-SupCon temperature={temperature:.2f}",
                    flush=True,
                )
                encoder_state, history, selection = train_supcon_candidate(
                    temperature, splits, args, device, seed + 100_000
                )
                candidate_states[temperature] = encoder_state
                candidate_histories[temperature] = history
                candidates.append(selection)
                temperature_dir = seed_dir / "temperature_selection" / f"tau_{temperature:.2f}"
                temperature_dir.mkdir(parents=True, exist_ok=True)
                save_history_plot(
                    history,
                    temperature_dir / "supcon_history.png",
                    f"CA-SupCon temperature={temperature:.2f}",
                )
                write_json(temperature_dir / "selection_metrics.json", selection)
            candidates.sort(
                key=lambda item: (
                    float(item.get("validation_pr_auc") or -math.inf),
                    float(item.get("validation_f1") or -math.inf),
                    -float(item["temperature"]),
                ),
                reverse=True,
            )
            selected = candidates[0]
            selected_temperature = float(selected["temperature"])
            selected_state = candidate_states[selected_temperature]
            temperature_selection = {
                "selection_split": "validation only",
                "primary_metric": "linear-probe PR-AUC",
                "tie_breaker": "linear-probe FoG F1, then lower temperature",
                "candidates": candidates,
                "selected_temperature": selected_temperature,
                "selected_encoder_epoch": selected["encoder_epoch"],
            }
            write_json(seed_dir / "temperature_selection.json", temperature_selection)

            classifier_seed = seed + 200_000
            for method, balanced in (("S2", False), ("S3", True)):
                if method not in requested_methods:
                    continue
                print(
                    f"[{args.subject} seed={seed}] training frozen head {method} "
                    f"balanced={balanced}",
                    flush=True,
                )
                model, head_history, head_selection = linear_probe(
                    selected_state,
                    splits,
                    args,
                    device,
                    seed=classifier_seed,
                    balanced=balanced,
                    max_epochs=args.classifier_epochs,
                    patience=args.early_stopping_patience,
                )
                history = [
                    {**row, "stage": "supcon_pretrain"}
                    for row in candidate_histories[selected_temperature]
                ] + [
                    {**row, "epoch": row["epoch"] + args.supcon_epochs, "stage": "frozen_head"}
                    for row in head_history
                ]
                selection = {
                    "encoder": temperature_selection,
                    "frozen_encoder": True,
                    "projection_head_discarded": True,
                    "classifier": head_selection,
                    "loss": "unweighted BCEWithLogitsLoss",
                }
                seed_results["methods"][method] = evaluate_method(
                    method,
                    model,
                    splits,
                    args,
                    device,
                    seed_dir / method,
                    history,
                    selection,
                    robust_scale,
                    seed,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        missing_methods = set(requested_methods) - set(seed_results["methods"])
        if missing_methods:
            raise RuntimeError(f"Seed {seed} did not produce methods: {sorted(missing_methods)}")
        seed_results["completed_utc"] = utc_now()
        write_json(seed_complete, seed_results)
        all_seed_results.append(seed_results)

    write_json(
        complete_path,
        {
            "protocol_id": PROTOCOL_ID,
            "subject_id": args.subject,
            "seeds": args.seeds,
            "methods": list(requested_methods),
            "completed_utc": utc_now(),
            "seed_count": len(all_seed_results),
        },
    )
    print(f"[complete] {args.subject}: {complete_path}", flush=True)


def parse_csv_values(text: str, cast) -> list[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_CA_pure"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "daphnet_ca_supcon_subject_v1"),
    )
    parser.add_argument("--subject", required=True, choices=FORMAL_SUBJECTS)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--methods", default="S0,S1,S2,S3")
    parser.add_argument("--temperatures", default="0.07,0.10,0.20")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--supervised-epochs", type=int, default=50)
    parser.add_argument("--supcon-epochs", type=int, default=60)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--selection-probe-epochs", type=int, default=20)
    parser.add_argument("--selection-probe-patience", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.seeds = parse_csv_values(args.seeds, int)
    args.methods = parse_csv_values(args.methods, str)
    args.temperatures = parse_csv_values(args.temperatures, float)
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if not set(args.methods).issubset(METHODS):
        raise ValueError(f"Methods must be a subset of {METHODS}")
    if "S3" in args.methods and "S2" not in args.methods:
        warnings.warn("S3 requested without S2 output; S2 representation will still be trained internally")
    if not args.temperatures:
        raise ValueError("At least one temperature is required")
    if args.batch_size != 32:
        warnings.warn("Template default batch_size is 32; override is intended for smoke tests only")
    for name in (
        "supervised_epochs",
        "supcon_epochs",
        "classifier_epochs",
        "probe_every",
        "selection_probe_epochs",
        "early_stopping_patience",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_subject(args)


if __name__ == "__main__":
    main()
