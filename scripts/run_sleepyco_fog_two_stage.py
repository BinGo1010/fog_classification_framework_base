#!/usr/bin/env python
"""Two-stage SleePyCo-style LOSO training for 3-class FOG detection.

Stage 1: supervised contrastive pretraining on single IMU epochs.
Stage 2: sequence fine-tuning with 5-epoch context, seq-to-one or seq2seq.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses import FocalLoss, SupConLoss
from models.sleepyco_fog import SleePyCoFogCRL, SleePyCoFogSequenceClassifier


DEFAULT_CLASS_NAMES = np.array(["NORMAL", "PRE_FOG", "FOG"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain SleePyCo CNN and fine-tune CNN+GRU/TCN on FOG LOSO NPZ data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "dataset/processed/fogstar_loso_activity1_notask2_7_3class_pre_fog5p0s_win90"
        ),
        help="Directory containing windows.npz and loso_folds.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sleepyco_fogstar_two_stage_win90"),
    )
    parser.add_argument(
        "--stage",
        choices=("pretrain", "finetune", "both"),
        default="both",
        help="Run only CRL pretraining, only fine-tuning, or both.",
    )
    parser.add_argument(
        "--baselines",
        default="seq2one_gru,seq2seq_gru",
        help="Comma-separated: seq2one_gru, seq2seq_gru, seq2seq_tcn.",
    )
    parser.add_argument("--folds", default="all", help="'all', '0:13', or '0,3,7'.")
    parser.add_argument("--seq-len", type=int, default=5, help="Number of IMU epochs per context.")
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument(
        "--target-position",
        choices=("center", "last"),
        default="center",
        help="Seq-to-one target inside the 5-epoch context.",
    )
    parser.add_argument(
        "--strict-consecutive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require equal start_sample step inside a sequence when start_sample exists.",
    )

    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--pretrain-patience", type=int, default=8)
    parser.add_argument("--finetune-patience", type=int, default=10)
    parser.add_argument("--pretrain-batch-size", type=int, default=512)
    parser.add_argument("--finetune-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--gru-pool",
        choices=("attn", "center", "last", "mean"),
        default="attn",
        help="Pooling used by seq2one_gru.",
    )
    parser.add_argument("--tcn-levels", type=int, default=3)
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze CNN backbone during fine-tuning.",
    )
    parser.add_argument(
        "--load-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrain checkpoint before fine-tuning.",
    )

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        choices=("ce", "focal"),
        default="focal",
        help="Fine-tuning classification loss.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced", "manual"),
        default="balanced",
        help="Class weights for CE/focal alpha.",
    )
    parser.add_argument(
        "--class-weights",
        default="1,4,2",
        help="Manual NORMAL,PRE_FOG,FOG loss weights.",
    )
    parser.add_argument(
        "--sampler",
        choices=("none", "balanced", "manual"),
        default="manual",
        help="Weighted sampler for train data.",
    )
    parser.add_argument(
        "--sampler-class-weights",
        default="1,4,2",
        help="Manual NORMAL,PRE_FOG,FOG sample weights.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        default="auto",
        help="'auto' uses len(train); otherwise pass an integer.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip finished folds/baselines with metrics.json.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_folds(spec: str, num_folds: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(num_folds))
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else num_folds
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        folds = list(range(start, stop, step))
    else:
        folds = [int(item.strip()) for item in spec.split(",") if item.strip()]
    bad = [fold for fold in folds if fold < 0 or fold >= num_folds]
    if bad:
        raise ValueError(f"Fold index out of range: {bad}; num_folds={num_folds}")
    return folds


def parse_float_list(value: str, expected: int) -> np.ndarray:
    arr = np.asarray([float(x.strip()) for x in value.split(",") if x.strip()], dtype=np.float32)
    if arr.size != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {value!r}.")
    if np.any(arr < 0):
        raise ValueError("Weights must be non-negative.")
    return arr


def samples_per_epoch(value: str, train_size: int) -> int:
    if str(value).strip().lower() == "auto":
        return int(train_size)
    n = int(value)
    if n <= 0:
        raise ValueError("--samples-per-epoch must be positive or auto.")
    return n


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=to_serializable)


def pick_key(npz: np.lib.npyio.NpzFile, candidates: tuple[str, ...], required: bool = True):
    for key in candidates:
        if key in npz.files:
            return npz[key], key
    if required:
        raise KeyError(f"None of keys {candidates} found. Available keys: {npz.files}")
    return None, None


def derive_subject_codes(subject_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subjects, inverse = np.unique(subject_values.astype(str), return_inverse=True)
    return inverse.astype(np.int32), subjects.astype(str)


def load_dataset(data_dir: Path) -> dict:
    data_dir = data_dir.resolve()
    windows_path = data_dir / "windows.npz"
    if not windows_path.exists():
        candidates = sorted(data_dir.glob("*.npz"))
        for path in candidates:
            with np.load(path, allow_pickle=True) as probe:
                if any(k in probe.files for k in ("X", "x", "windows", "data")) and any(
                    k in probe.files for k in ("y", "labels", "label")
                ):
                    windows_path = path
                    break
    if not windows_path.exists():
        raise FileNotFoundError(f"No window NPZ found under {data_dir}")

    win = np.load(windows_path, allow_pickle=True)
    x, x_key = pick_key(win, ("X", "x", "windows", "data", "features"))
    y, y_key = pick_key(win, ("y", "labels", "label", "target"))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape [N,T,C] or [N,C,T], got {x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] <= 32 and x.shape[2] > x.shape[1]:
        x = np.ascontiguousarray(x.transpose(0, 2, 1))

    class_names, _ = pick_key(win, ("class_names", "classes"), required=False)
    class_names = np.asarray(class_names).astype(str) if class_names is not None else DEFAULT_CLASS_NAMES

    subject_code, _ = pick_key(
        win,
        ("subject_code", "window_subject_code", "subject_idx", "subject_id_code"),
        required=False,
    )
    subjects, _ = pick_key(win, ("subjects", "subject_names"), required=False)
    if subject_code is None:
        subject_values, _ = pick_key(win, ("subject", "subject_id", "subjects_per_window"))
        subject_code, subjects = derive_subject_codes(np.asarray(subject_values))
    else:
        subject_code = np.asarray(subject_code, dtype=np.int32).reshape(-1)
        if subjects is None:
            subjects = np.asarray([str(x) for x in sorted(np.unique(subject_code))])
        else:
            subjects = np.asarray(subjects).astype(str)

    file_id, _ = pick_key(win, ("file_id", "recording_id", "series_id", "file"), required=False)
    if file_id is None:
        file_id = np.zeros(y.shape[0], dtype=np.int32)
    else:
        file_id = np.asarray(file_id)

    start_sample, _ = pick_key(win, ("start_sample", "start", "window_start"), required=False)
    if start_sample is None:
        start_sample = np.arange(y.shape[0], dtype=np.int64)
        has_start_sample = False
    else:
        start_sample = np.asarray(start_sample, dtype=np.int64).reshape(-1)
        has_start_sample = True

    folds_path = data_dir / "loso_folds.npz"
    folds = np.load(folds_path, allow_pickle=True) if folds_path.exists() else None
    fold_test_subjects = None
    fold_val_subjects = None
    fold_test_subject_codes = None
    fold_val_subject_codes = None
    if folds is not None:
        fold_test_subjects, _ = pick_key(folds, ("fold_test_subjects", "test_subjects"), required=False)
        fold_val_subjects, _ = pick_key(folds, ("fold_val_subjects", "val_subjects"), required=False)
        fold_test_subject_codes, _ = pick_key(folds, ("fold_test_subject_codes", "test_subject_codes"), required=False)
        fold_val_subject_codes, _ = pick_key(folds, ("fold_val_subject_codes", "val_subject_codes"), required=False)
    if fold_test_subjects is None:
        fold_test_subjects = subjects
    if fold_val_subjects is None:
        fold_val_subjects = np.roll(fold_test_subjects, -1)

    return {
        "data_dir": data_dir,
        "windows_path": windows_path,
        "x_key": x_key,
        "y_key": y_key,
        "X": x,
        "y": y,
        "class_names": class_names,
        "subjects": np.asarray(subjects).astype(str),
        "subject_code": subject_code,
        "file_id": file_id,
        "start_sample": start_sample,
        "has_start_sample": has_start_sample,
        "fold_test_subjects": np.asarray(fold_test_subjects).astype(str),
        "fold_val_subjects": np.asarray(fold_val_subjects).astype(str),
        "fold_test_subject_codes": (
            np.asarray(fold_test_subject_codes, dtype=np.int64)
            if fold_test_subject_codes is not None
            else None
        ),
        "fold_val_subject_codes": (
            np.asarray(fold_val_subject_codes, dtype=np.int64)
            if fold_val_subject_codes is not None
            else None
        ),
    }


def subject_name_to_code(subjects: np.ndarray, subject_code: np.ndarray, subject: str) -> int:
    unique_codes = set(int(x) for x in np.unique(subject_code))
    match = np.flatnonzero(subjects.astype(str) == str(subject))
    if match.size == 0:
        try:
            parsed = int(subject)
        except ValueError as exc:
            raise KeyError(f"Subject {subject!r} not found in subjects array.") from exc
        if parsed in unique_codes:
            return parsed
        raise KeyError(f"Subject code {parsed} not found in window subject_code.")

    matched_subject = str(subjects[int(match[0])])
    try:
        parsed = int(matched_subject)
        if parsed in unique_codes:
            return parsed
    except ValueError:
        pass
    return int(match[0])


def split_indices(data: dict, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    subjects = data["subjects"]
    code = data["subject_code"]
    if data.get("fold_test_subject_codes") is not None:
        test_codes = np.asarray(data["fold_test_subject_codes"][fold], dtype=np.int64)
        val_codes = np.asarray(data["fold_val_subject_codes"][fold], dtype=np.int64)
        test_codes = test_codes[test_codes >= 0]
        val_codes = val_codes[val_codes >= 0]
        test_idx = np.flatnonzero(np.isin(code, test_codes))
        val_idx = np.flatnonzero(np.isin(code, val_codes))
        train_idx = np.flatnonzero(~np.isin(code, np.r_[test_codes, val_codes]))
        test_subject = "|".join(str(subjects[int(subject_code)]) for subject_code in test_codes)
        val_subject = "|".join(str(subjects[int(subject_code)]) for subject_code in val_codes)
        return train_idx, val_idx, test_idx, test_subject, val_subject

    test_subject = str(data["fold_test_subjects"][fold])
    val_subject = str(data["fold_val_subjects"][fold])
    test_code = subject_name_to_code(subjects, code, test_subject)
    val_code = subject_name_to_code(subjects, code, val_subject)
    test_idx = np.flatnonzero(code == test_code)
    val_idx = np.flatnonzero(code == val_code)
    train_idx = np.flatnonzero((code != test_code) & (code != val_code))
    return train_idx, val_idx, test_idx, test_subject, val_subject


def compute_norm(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x[train_idx].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


class IMUAugment:
    def __init__(
        self,
        jitter_std: float = 0.03,
        scale_range: tuple[float, float] = (0.8, 1.2),
        max_shift_ratio: float = 0.1,
        zero_mask_ratio: float = 0.1,
    ):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.max_shift_ratio = max_shift_ratio
        self.zero_mask_ratio = zero_mask_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, copy=True)
        channels, samples = out.shape
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(channels, 1))
        out *= scale.astype(np.float32)
        max_shift = int(round(samples * self.max_shift_ratio))
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift + 1)
            out = np.roll(out, shift=shift, axis=-1)
        if self.zero_mask_ratio > 0:
            width = max(1, int(round(samples * self.zero_mask_ratio)))
            start = np.random.randint(0, max(samples - width + 1, 1))
            out[:, start : start + width] = 0.0
        if self.jitter_std > 0:
            out += np.random.normal(0.0, self.jitter_std, size=out.shape).astype(np.float32)
        return out.astype(np.float32, copy=False)


class PretrainEpochDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
    ):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.transform = IMUAugment()

    def __len__(self) -> int:
        return int(self.indices.size)

    def _get_epoch(self, raw_idx: int) -> np.ndarray:
        epoch = (self.x[raw_idx] - self.mean) / self.std
        return np.ascontiguousarray(epoch.T.astype(np.float32))

    def __getitem__(self, idx: int):
        raw_idx = int(self.indices[idx])
        epoch = self._get_epoch(raw_idx)
        if self.augment:
            view1 = self.transform(epoch)
            view2 = self.transform(epoch)
        else:
            view1 = epoch
            view2 = epoch.copy()
        return torch.from_numpy(view1), torch.from_numpy(view2), torch.tensor(self.y[raw_idx]).long()


def build_sequences(
    indices: np.ndarray,
    subject_code: np.ndarray,
    file_id: np.ndarray,
    start_sample: np.ndarray,
    seq_len: int,
    seq_stride: int,
    strict_consecutive: bool,
    has_start_sample: bool,
) -> np.ndarray:
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx in indices:
        groups[(int(subject_code[idx]), str(file_id[idx]))].append(int(idx))

    sequences: list[np.ndarray] = []
    for group_indices in groups.values():
        order = np.asarray(group_indices, dtype=np.int64)
        sort_key = start_sample[order] if has_start_sample else order
        order = order[np.argsort(sort_key, kind="stable")]
        if order.size < seq_len:
            continue

        expected_step = None
        if strict_consecutive and has_start_sample and order.size > 1:
            diffs = np.diff(start_sample[order])
            diffs = diffs[diffs > 0]
            if diffs.size:
                expected_step = int(np.median(diffs))

        for start in range(0, order.size - seq_len + 1, seq_stride):
            seq = order[start : start + seq_len]
            if expected_step is not None:
                diffs = np.diff(start_sample[seq])
                if not np.all(diffs == expected_step):
                    continue
            sequences.append(seq)

    if not sequences:
        return np.empty((0, seq_len), dtype=np.int64)
    return np.stack(sequences).astype(np.int64)


class SequenceEpochDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sequences: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        seq2seq: bool,
        target_position: str,
    ):
        self.x = x
        self.y = y
        self.sequences = sequences
        self.mean = mean
        self.std = std
        self.seq2seq = seq2seq
        self.target_idx = sequences.shape[1] // 2 if target_position == "center" else sequences.shape[1] - 1

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int):
        raw_indices = self.sequences[idx]
        epochs = (self.x[raw_indices] - self.mean) / self.std
        epochs = np.ascontiguousarray(epochs.transpose(0, 2, 1).astype(np.float32))
        labels = self.y[raw_indices].astype(np.int64)
        if self.seq2seq:
            target = torch.from_numpy(labels)
        else:
            target = torch.tensor(labels[self.target_idx]).long()
        return torch.from_numpy(epochs), target


def build_sampler(
    labels: np.ndarray,
    num_classes: int,
    mode: str,
    manual_weights: str,
    samples_value: str,
) -> WeightedRandomSampler | None:
    if mode == "none":
        return None
    if mode == "manual":
        class_weight = parse_float_list(manual_weights, num_classes).astype(np.float64)
    else:
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        class_weight = np.zeros(num_classes, dtype=np.float64)
        nonzero = counts > 0
        class_weight[nonzero] = counts.sum() / (num_classes * counts[nonzero])
    sample_weight = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=samples_per_epoch(samples_value, labels.size),
        replacement=True,
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, sampler, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def class_weight_tensor(
    labels: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor | None:
    if args.class_weight == "none":
        return None
    if args.class_weight == "manual":
        weights = parse_float_list(args.class_weights, num_classes)
    else:
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(labels: np.ndarray, num_classes: int, args: argparse.Namespace, device: torch.device):
    weight = class_weight_tensor(labels, num_classes, args, device)
    if args.loss == "focal":
        return FocalLoss(weight=weight, gamma=args.focal_gamma)
    return torch.nn.CrossEntropyLoss(weight=weight)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1, len(class_names))
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if len(class_names) == 2:
            y_positive = (y_true == labels[1]).astype(np.int8)
            positive_prob = y_prob[:, 1]
            try:
                metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_positive, positive_prob))
            except ValueError:
                metrics["roc_auc_ovr_macro"] = None
            try:
                metrics["pr_auc_macro"] = float(average_precision_score(y_positive, positive_prob))
            except ValueError:
                metrics["pr_auc_macro"] = None
        else:
            y_bin = label_binarize(y_true, classes=labels)
            try:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
                )
            except ValueError:
                metrics["roc_auc_ovr_macro"] = None
            try:
                metrics["pr_auc_macro"] = float(average_precision_score(y_bin, y_prob, average="macro"))
            except ValueError:
                metrics["pr_auc_macro"] = None

    for i, name in enumerate(class_names):
        prefix = str(name).lower()
        metrics[f"{prefix}_precision"] = float(precision[i])
        metrics[f"{prefix}_recall"] = float(recall[i])
        metrics[f"{prefix}_f1"] = float(f1[i])
        metrics[f"{prefix}_support"] = int(support[i])
    return metrics


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    row = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        if isinstance(value, (int, float)) or value is None:
            row[f"{prefix}_{key}"] = value
    return row


def pretrain_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    for view1, view2, labels in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                features = model.project(torch.cat([view1, view2], dim=0))
                f1, f2 = torch.chunk(features, 2, dim=0)
                loss = criterion(torch.stack([f1, f2], dim=1), labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch = int(labels.size(0))
        total_loss += float(loss.detach().item()) * batch
        total_n += batch
    return total_loss / max(total_n, 1)


def finetune_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool, freeze_backbone: bool = False):
    train = optimizer is not None
    model.train(train)
    if freeze_backbone:
        model.backbone.eval()
    total_loss = 0.0
    total_n = 0
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(xb)
                if model.is_seq2seq:
                    loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    prob = torch.softmax(logits.detach(), dim=-1).reshape(-1, logits.size(-1))
                    target = yb.reshape(-1)
                else:
                    loss = criterion(logits, yb)
                    prob = torch.softmax(logits.detach(), dim=-1)
                    target = yb
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_n = int(target.numel())
        total_loss += float(loss.detach().item()) * batch_n
        total_n += batch_n
        y_true.append(target.detach().cpu().numpy())
        y_prob.append(prob.detach().cpu().numpy())
    return total_loss / max(total_n, 1), np.concatenate(y_true), np.concatenate(y_prob)


def run_pretrain_fold(
    fold: int,
    args: argparse.Namespace,
    data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> Path:
    fold_dir = args.output_dir / "pretrain" / f"fold_{fold:03d}"
    ckpt_path = fold_dir / "pretrain_best.pt"
    if args.resume and ckpt_path.exists():
        print(f"[fold {fold:03d}] pretrain resume: {ckpt_path}")
        return ckpt_path

    fold_dir.mkdir(parents=True, exist_ok=True)
    x, y = data["X"], data["y"]
    train_ds = PretrainEpochDataset(x, y, train_idx, mean, std, augment=True)
    val_ds = PretrainEpochDataset(x, y, val_idx, mean, std, augment=False)
    train_labels = y[train_idx]
    sampler = build_sampler(
        train_labels,
        len(data["class_names"]),
        args.sampler,
        args.sampler_class_weights,
        args.samples_per_epoch,
    )
    train_loader = make_loader(train_ds, args.pretrain_batch_size, True, sampler, args.num_workers, device)
    val_loader = make_loader(val_ds, args.pretrain_batch_size, False, None, args.num_workers, device)

    model = SleePyCoFogCRL(
        in_channels=x.shape[2],
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = SupConLoss(temperature=args.temperature, base_temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_loss = float("inf")
    bad_epochs = 0
    log_path = fold_dir / "pretrain_log.csv"
    if log_path.exists():
        log_path.unlink()

    print(f"[fold {fold:03d}] pretrain train={len(train_ds)} val={len(val_ds)}")
    for epoch in range(1, args.pretrain_epochs + 1):
        start = time.perf_counter()
        train_loss = pretrain_epoch(model, train_loader, criterion, optimizer, scaler, device, args.amp)
        with torch.no_grad():
            val_loss = pretrain_epoch(model, val_loader, criterion, None, scaler, device, args.amp)
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
        append_csv(
            log_path,
            {
                "fold": fold,
                "epoch": epoch,
                "epoch_sec": round(time.perf_counter() - start, 3),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_loss,
            },
        )
        print(
            f"[fold {fold:03d}] pretrain epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} {'*' if improved else ''}"
        )
        if bad_epochs >= args.pretrain_patience:
            print(f"[fold {fold:03d}] pretrain early stopping at epoch {epoch}")
            break

    return ckpt_path


def sequence_target_labels(y: np.ndarray, sequences: np.ndarray, target_position: str) -> np.ndarray:
    target_idx = sequences.shape[1] // 2 if target_position == "center" else sequences.shape[1] - 1
    return y[sequences[:, target_idx]].astype(np.int64)


def run_finetune_fold(
    fold: int,
    baseline: str,
    args: argparse.Namespace,
    data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pretrain_ckpt: Path | None,
    test_subject: str,
    val_subject: str,
    device: torch.device,
) -> dict:
    fold_dir = args.output_dir / baseline / f"fold_{fold:03d}"
    metrics_path = fold_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            print(f"[fold {fold:03d}] {baseline} resume: metrics already exist")
            return json.load(f)

    fold_dir.mkdir(parents=True, exist_ok=True)
    x, y = data["X"], data["y"]
    seq2seq = baseline.startswith("seq2seq")
    train_seq = build_sequences(
        train_idx,
        data["subject_code"],
        data["file_id"],
        data["start_sample"],
        args.seq_len,
        args.seq_stride,
        args.strict_consecutive,
        data["has_start_sample"],
    )
    val_seq = build_sequences(
        val_idx,
        data["subject_code"],
        data["file_id"],
        data["start_sample"],
        args.seq_len,
        args.seq_stride,
        args.strict_consecutive,
        data["has_start_sample"],
    )
    test_seq = build_sequences(
        test_idx,
        data["subject_code"],
        data["file_id"],
        data["start_sample"],
        args.seq_len,
        args.seq_stride,
        args.strict_consecutive,
        data["has_start_sample"],
    )
    if train_seq.size == 0 or val_seq.size == 0 or test_seq.size == 0:
        raise RuntimeError(
            f"Empty sequence split in fold {fold}: train={train_seq.shape}, "
            f"val={val_seq.shape}, test={test_seq.shape}"
        )

    train_ds = SequenceEpochDataset(x, y, train_seq, mean, std, seq2seq, args.target_position)
    val_ds = SequenceEpochDataset(x, y, val_seq, mean, std, seq2seq, args.target_position)
    test_ds = SequenceEpochDataset(x, y, test_seq, mean, std, seq2seq, args.target_position)

    if seq2seq:
        sampler_labels = sequence_target_labels(y, train_seq, args.target_position)
        loss_labels = y[train_seq.reshape(-1)]
    else:
        sampler_labels = sequence_target_labels(y, train_seq, args.target_position)
        loss_labels = sampler_labels
    sampler = build_sampler(
        sampler_labels,
        len(data["class_names"]),
        args.sampler,
        args.sampler_class_weights,
        args.samples_per_epoch,
    )
    train_loader = make_loader(train_ds, args.finetune_batch_size, True, sampler, args.num_workers, device)
    val_loader = make_loader(val_ds, args.finetune_batch_size, False, None, args.num_workers, device)
    test_loader = make_loader(test_ds, args.finetune_batch_size, False, None, args.num_workers, device)

    model = SleePyCoFogSequenceClassifier(
        in_channels=x.shape[2],
        num_classes=len(data["class_names"]),
        baseline=baseline,
        feature_dim=args.feature_dim,
        num_scales=args.num_scales,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gru_pool=args.gru_pool,
        tcn_levels=args.tcn_levels,
    ).to(device)

    loaded_pretrain = False
    if args.load_pretrained and pretrain_ckpt is not None:
        if not pretrain_ckpt.exists():
            raise FileNotFoundError(f"Missing pretrain checkpoint: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
        model.backbone.load_state_dict(ckpt["backbone"], strict=False)
        loaded_pretrain = True

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

    criterion = build_criterion(loss_labels, len(data["class_names"]), args, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    log_path = fold_dir / "finetune_log.csv"
    if log_path.exists():
        log_path.unlink()

    fold_start = time.perf_counter()
    print(
        f"[fold {fold:03d}] {baseline} test={test_subject} val={val_subject} "
        f"seq train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"pretrained={loaded_pretrain}"
    )
    for epoch in range(1, args.finetune_epochs + 1):
        start = time.perf_counter()
        train_loss, train_true, train_prob = finetune_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.amp,
            freeze_backbone=args.freeze_backbone,
        )
        train_metrics = compute_metrics(train_true, train_prob, data["class_names"])
        with torch.no_grad():
            val_loss, val_true, val_prob = finetune_epoch(
                model,
                val_loader,
                criterion,
                None,
                scaler,
                device,
                args.amp,
                freeze_backbone=args.freeze_backbone,
            )
        val_metrics = compute_metrics(val_true, val_prob, data["class_names"])
        val_metrics["loss"] = val_loss
        current = val_metrics["f1_macro"]
        improved = current > best_score
        if improved:
            best_score = current
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "args": vars(args),
                },
                fold_dir / "finetune_best.pt",
            )
        else:
            bad_epochs += 1
        append_csv(
            log_path,
            {
                "fold": fold,
                "epoch": epoch,
                "epoch_sec": round(time.perf_counter() - start, 3),
                "train_loss": train_loss,
                "train_f1_macro": train_metrics["f1_macro"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_f1_macro": val_metrics["f1_macro"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "best_epoch": best_epoch,
            },
        )
        print(
            f"[fold {fold:03d}] {baseline} epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_f1={val_metrics['f1_macro']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} {'*' if improved else ''}"
        )
        if bad_epochs >= args.finetune_patience:
            print(f"[fold {fold:03d}] {baseline} early stopping at epoch {epoch}")
            break

    best = torch.load(fold_dir / "finetune_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    with torch.no_grad():
        test_loss, test_true, test_prob = finetune_epoch(
            model,
            test_loader,
            criterion,
            None,
            scaler,
            device,
            args.amp,
            freeze_backbone=args.freeze_backbone,
        )
    test_metrics = compute_metrics(test_true, test_prob, data["class_names"])
    test_metrics["loss"] = test_loss

    metrics = {
        "fold": fold,
        "baseline": baseline,
        "test_subject": test_subject,
        "val_subject": val_subject,
        "best_epoch": int(best_epoch),
        "elapsed_sec": float(time.perf_counter() - fold_start),
        "loaded_pretrain": loaded_pretrain,
        "train_windows": int(train_idx.size),
        "val_windows": int(val_idx.size),
        "test_windows": int(test_idx.size),
        "train_sequences": int(train_seq.shape[0]),
        "val_sequences": int(val_seq.shape[0]),
        "test_sequences": int(test_seq.shape[0]),
        "train_counts": np.bincount(loss_labels, minlength=len(data["class_names"])).tolist(),
        "best_val": best["val_metrics"],
        "test": test_metrics,
    }
    save_json(metrics_path, metrics)
    return metrics


def aggregate(rows: list[dict]) -> dict:
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "normal_recall",
        "pre_fog_recall",
        "fog_recall",
        "normal_f1",
        "pre_fog_f1",
        "fog_f1",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    ]
    summary = {}
    for key in metric_keys:
        values = [row.get(f"test_{key}") for row in rows if row.get(f"test_{key}") is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "config.json", vars(args))

    data = load_dataset(args.data_dir)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    allowed = {"seq2one_gru", "seq2seq_gru", "seq2seq_tcn"}
    unknown = sorted(set(baselines) - allowed)
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")

    folds = parse_folds(args.folds, len(data["fold_test_subjects"]))
    print(
        f"[INFO] data={data['windows_path']} X={data['X'].shape} "
        f"classes={data['class_names'].tolist()} folds={folds} device={device}"
    )

    rows_by_baseline: dict[str, list[dict]] = {baseline: [] for baseline in baselines}
    for fold in folds:
        train_idx, val_idx, test_idx, test_subject, val_subject = split_indices(data, fold)
        mean, std = compute_norm(data["X"], train_idx)
        pretrain_ckpt = args.output_dir / "pretrain" / f"fold_{fold:03d}" / "pretrain_best.pt"

        if args.stage in ("pretrain", "both"):
            pretrain_ckpt = run_pretrain_fold(
                fold, args, data, train_idx, val_idx, mean, std, device
            )

        if args.stage in ("finetune", "both"):
            for baseline in baselines:
                metrics = run_finetune_fold(
                    fold,
                    baseline,
                    args,
                    data,
                    train_idx,
                    val_idx,
                    test_idx,
                    mean,
                    std,
                    pretrain_ckpt if args.load_pretrained else None,
                    test_subject,
                    val_subject,
                    device,
                )
                row = {
                    "fold": fold,
                    "baseline": baseline,
                    "test_subject": test_subject,
                    "val_subject": val_subject,
                    "best_epoch": metrics["best_epoch"],
                }
                row.update(flatten_metrics("test", metrics["test"]))
                rows_by_baseline[baseline].append(row)
                write_csv(args.output_dir / baseline / "summary.csv", rows_by_baseline[baseline])
                save_json(args.output_dir / baseline / "aggregate.json", aggregate(rows_by_baseline[baseline]))


if __name__ == "__main__":
    main()
