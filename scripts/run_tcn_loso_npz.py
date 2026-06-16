#!/usr/bin/env python
"""Run TCN LOSO classification experiments from compact NPZ windows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path

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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tcn import TCNClassifier


DEFAULT_CLASS_NAMES = np.array(["NORMAL", "PRE_FOG", "FOG"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate TCN on a subset of compact LOSO NPZ folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/processed/fog_loso_npz"),
        help="Directory containing windows.npz and loso_folds.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tcn_loso_npz_20fold"),
        help="Experiment output directory.",
    )
    parser.add_argument(
        "--folds",
        default="0:20",
        help="Fold indices, e.g. '0:20' or '0,3,7'.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="balanced",
        help="Use inverse-frequency train-set class weights.",
    )
    parser.add_argument(
        "--sampler",
        choices=("none", "weighted", "manual"),
        default="none",
        help=(
            "Training sampler. 'weighted' uses inverse-frequency weights; "
            "'manual' uses --sampler-class-weights."
        ),
    )
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=1.0,
        help="Power applied to inverse class-frequency sampler weights.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        default="auto",
        help="'auto' keeps one epoch at len(train), or pass an integer number of samples.",
    )
    parser.add_argument(
        "--sampler-class-weights",
        default="1,4,2",
        help="Comma-separated per-class sample weights for --sampler manual.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip folds that already have metrics.json.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a torch device string.",
    )
    return parser.parse_args()


def parse_folds(spec: str, num_folds: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(num_folds))
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid fold range: {spec}")
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
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=to_serializable)


def class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def normalize_split(
    x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x[train_idx].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def transform(indices: np.ndarray) -> np.ndarray:
        transformed = (x[indices].astype(np.float32, copy=False) - mean) / std
        return np.ascontiguousarray(transformed.transpose(0, 2, 1))

    return transform(train_idx), transform(val_idx), transform(test_idx), mean, std


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def parse_samples_per_epoch(value: str, train_size: int) -> int:
    value = str(value).strip().lower()
    if value == "auto":
        return int(train_size)
    samples = int(value)
    if samples <= 0:
        raise ValueError("--samples-per-epoch must be positive or 'auto'.")
    return samples


def parse_sampler_class_weights(value: str, class_names: np.ndarray) -> np.ndarray:
    weights = np.array(
        [float(part.strip()) for part in str(value).split(",") if part.strip()],
        dtype=np.float64,
    )
    if len(weights) != len(class_names):
        raise ValueError(
            f"--sampler-class-weights must contain {len(class_names)} values: "
            f"{','.join(str(name) for name in class_names)}."
        )
    if np.any(weights < 0):
        raise ValueError("--sampler-class-weights must be non-negative.")
    if not np.any(weights > 0):
        raise ValueError("--sampler-class-weights must contain at least one positive weight.")
    return weights


def build_train_sampler(
    args: argparse.Namespace,
    y_train: np.ndarray,
    num_classes: int,
    class_names: np.ndarray,
) -> WeightedRandomSampler | None:
    if args.sampler == "none":
        return None

    if args.sampler == "manual":
        class_weight = parse_sampler_class_weights(args.sampler_class_weights, class_names)
    else:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        inv = np.zeros_like(counts, dtype=np.float64)
        nonzero = counts > 0
        inv[nonzero] = 1.0 / counts[nonzero]
        class_weight = np.power(inv, float(args.sampler_power))

    sample_weight = class_weight[y_train]
    num_samples = parse_samples_per_epoch(args.samples_per_epoch, len(y_train))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: np.ndarray) -> dict:
    num_classes = len(class_names)
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(num_classes))
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
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred, adjusted=False)
            ),
            "f1_macro": float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if num_classes == 2:
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
                metrics["pr_auc_macro"] = float(
                    average_precision_score(y_bin, y_prob, average="macro")
                )
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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
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
                loss = criterion(logits, yb)
            if train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        prob = torch.softmax(logits.detach(), dim=1)
        bs = int(yb.size(0))
        total_loss += float(loss.detach().item()) * bs
        total_n += bs
        y_true.append(yb.detach().cpu().numpy())
        y_prob.append(prob.detach().cpu().numpy())

    return total_loss / max(total_n, 1), np.concatenate(y_true), np.concatenate(y_prob)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp: bool,
    class_names: np.ndarray,
) -> dict:
    with torch.no_grad():
        loss, y_true, y_prob = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            amp=amp,
        )
    metrics = compute_metrics(y_true, y_prob, class_names)
    metrics["loss"] = float(loss)
    return metrics


def run_fold(
    fold: int,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    subjects: np.ndarray,
    fold_test_subjects: np.ndarray,
    fold_val_subjects: np.ndarray,
    window_subject_code: np.ndarray,
    device: torch.device,
    fold_test_subject_codes: np.ndarray | None = None,
    fold_val_subject_codes: np.ndarray | None = None,
) -> dict:
    fold_dir = args.output_dir / f"fold_{fold:03d}"
    metrics_path = fold_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[fold {fold:03d}] resume: metrics already exist")
        return metrics

    fold_dir.mkdir(parents=True, exist_ok=True)
    if fold_test_subject_codes is not None:
        test_codes = np.asarray(fold_test_subject_codes[fold], dtype=np.int64)
        val_codes = np.asarray(fold_val_subject_codes[fold], dtype=np.int64)
        test_codes = test_codes[test_codes >= 0]
        val_codes = val_codes[val_codes >= 0]
        test_subject = "|".join(str(subjects[int(subject_code)]) for subject_code in test_codes)
        val_subject = "|".join(str(subjects[int(subject_code)]) for subject_code in val_codes)
        test_idx = np.flatnonzero(np.isin(window_subject_code, test_codes))
        val_idx = np.flatnonzero(np.isin(window_subject_code, val_codes))
        train_idx = np.flatnonzero(~np.isin(window_subject_code, np.r_[test_codes, val_codes]))
    else:
        test_subject = str(fold_test_subjects[fold])
        val_subject = str(fold_val_subjects[fold])
        test_code = int(np.flatnonzero(subjects == test_subject)[0])
        val_code = int(np.flatnonzero(subjects == val_subject)[0])

        test_idx = np.flatnonzero(window_subject_code == test_code)
        val_idx = np.flatnonzero(window_subject_code == val_code)
        train_idx = np.flatnonzero((window_subject_code != test_code) & (window_subject_code != val_code))

    t_prepare = time.perf_counter()
    x_train, x_val, x_test, mean, std = normalize_split(x, train_idx, val_idx, test_idx)
    y_train = y[train_idx].astype(np.int64, copy=False)
    y_val = y[val_idx].astype(np.int64, copy=False)
    y_test = y[test_idx].astype(np.int64, copy=False)
    num_classes = len(class_names)

    pin_memory = device.type == "cuda"
    train_sampler = build_train_sampler(args, y_train, num_classes, class_names)
    train_loader = make_loader(
        x_train,
        y_train,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        x_val, y_val, args.batch_size, False, args.num_workers, pin_memory
    )
    test_loader = make_loader(
        x_test, y_test, args.batch_size, False, args.num_workers, pin_memory
    )
    prepare_sec = time.perf_counter() - t_prepare

    model = TCNClassifier(
        in_channels=x.shape[2],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        levels=args.levels,
        kernel_size=args.kernel_size,
    ).to(device)
    weight = (
        class_weights(y_train, num_classes, device)
        if args.class_weight == "balanced"
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    fold_start = time.perf_counter()
    log_path = fold_dir / "train_log.csv"
    if log_path.exists():
        log_path.unlink()

    print(
        f"[fold {fold:03d}] test={test_subject} val={val_subject} "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"sampler={args.sampler}"
    )
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_true, train_prob = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
        )
        train_metrics = compute_metrics(train_true, train_prob, class_names)
        train_metrics["loss"] = float(train_loss)
        val_metrics = evaluate(model, val_loader, criterion, device, args.amp, class_names)
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
                    "class_names": class_names,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "args": vars(args),
                },
                fold_dir / "best.pt",
            )
        else:
            bad_epochs += 1

        row = {
            "fold": fold,
            "epoch": epoch,
            "epoch_sec": round(time.perf_counter() - epoch_start, 3),
            "train_loss": train_metrics["loss"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "best_epoch": best_epoch,
        }
        append_csv(log_path, row)
        print(
            f"[fold {fold:03d}] epoch {epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"{'*' if improved else ''}"
        )
        if bad_epochs >= args.patience:
            print(f"[fold {fold:03d}] early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, criterion, device, args.amp, class_names)
    best_val_metrics = checkpoint["val_metrics"]
    elapsed_sec = time.perf_counter() - fold_start

    metrics = {
        "fold": fold,
        "test_subject": test_subject,
        "val_subject": val_subject,
        "best_epoch": int(best_epoch),
        "prepare_sec": float(prepare_sec),
        "elapsed_sec": float(elapsed_sec),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "train_counts": np.bincount(y_train, minlength=num_classes).tolist(),
        "val_counts": np.bincount(y_val, minlength=num_classes).tolist(),
        "test_counts": np.bincount(y_test, minlength=num_classes).tolist(),
        "best_val": best_val_metrics,
        "test": test_metrics,
    }
    save_json(metrics_path, metrics)

    del model, optimizer, criterion, train_loader, val_loader, test_loader
    del x_train, x_val, x_test, y_train, y_val, y_test
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def class_count_fields(prefix: str, counts: list[int], class_names: np.ndarray) -> dict:
    return {
        f"{prefix}_{str(name).lower()}": int(counts[idx])
        for idx, name in enumerate(class_names)
    }


def aggregate(rows: list[dict], class_names: np.ndarray) -> dict:
    count_keys = {"windows"} | {str(name).lower() for name in class_names}
    metric_keys = sorted(
        {
            key.removeprefix("test_")
            for row in rows
            for key, value in row.items()
            if key.startswith("test_")
            and isinstance(value, (int, float))
            and not key.endswith("_support")
            and key.removeprefix("test_") not in count_keys
        }
    )
    summary = {}
    for key in metric_keys:
        values = [
            row[f"test_{key}"]
            for row in rows
            if row.get(f"test_{key}") is not None
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"Loading data from {args.data_dir}")
    with np.load(args.data_dir / "windows.npz") as data:
        x = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
        class_names = (
            data["class_names"].astype(str)
            if "class_names" in data.files
            else DEFAULT_CLASS_NAMES
        )
    with np.load(args.data_dir / "loso_folds.npz") as folds_npz:
        subjects = folds_npz["subjects"]
        fold_test_subjects = folds_npz["fold_test_subjects"]
        fold_val_subjects = folds_npz["fold_val_subjects"]
        window_subject_code = folds_npz["window_subject_code"]
        fold_test_subject_codes = (
            folds_npz["fold_test_subject_codes"]
            if "fold_test_subject_codes" in folds_npz.files
            else None
        )
        fold_val_subject_codes = (
            folds_npz["fold_val_subject_codes"]
            if "fold_val_subject_codes" in folds_npz.files
            else None
        )

    folds = parse_folds(args.folds, len(fold_test_subjects))
    if class_names.ndim != 1 or len(class_names) == 0:
        raise ValueError(f"Invalid class_names: {class_names}")
    label_values = set(np.unique(y).astype(int).tolist())
    allowed_labels = set(range(len(class_names)))
    if not label_values.issubset(allowed_labels):
        raise ValueError(
            f"y contains labels outside {sorted(allowed_labels)}: {sorted(label_values)}"
        )
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "folds_expanded": folds,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "class_names": class_names.tolist(),
            "input_channels": int(x.shape[2]),
        },
    )
    print(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    print(f"Data: X={x.shape} classes={class_names.tolist()}")
    print(f"Running folds: {folds}")

    rows = []
    total_start = time.perf_counter()
    for fold in folds:
        metrics = run_fold(
            fold,
            args,
            x,
            y,
            class_names,
            subjects,
            fold_test_subjects,
            fold_val_subjects,
            window_subject_code,
            device,
            fold_test_subject_codes,
            fold_val_subject_codes,
        )
        row = {
            "fold": metrics["fold"],
            "test_subject": metrics["test_subject"],
            "val_subject": metrics["val_subject"],
            "best_epoch": metrics["best_epoch"],
            "elapsed_sec": round(metrics["elapsed_sec"], 3),
            "train_windows": metrics["train_windows"],
            "val_windows": metrics["val_windows"],
            "test_windows": metrics["test_windows"],
        }
        row.update(class_count_fields("train", metrics["train_counts"], class_names))
        row.update(class_count_fields("val", metrics["val_counts"], class_names))
        row.update(class_count_fields("test", metrics["test_counts"], class_names))
        row.update(flatten_metrics("best_val", metrics["best_val"]))
        row.update(flatten_metrics("test", metrics["test"]))
        rows.append(row)
        write_csv(args.output_dir / "summary.csv", rows)
        save_json(
            args.output_dir / "aggregate.json",
            {
                "folds": rows,
                "aggregate": aggregate(rows, class_names),
                "elapsed_sec": time.perf_counter() - total_start,
            },
        )

    final = {
        "folds": rows,
        "aggregate": aggregate(rows, class_names),
        "elapsed_sec": time.perf_counter() - total_start,
    }
    save_json(args.output_dir / "aggregate.json", final)
    print("Done.")
    print(json.dumps(final["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
