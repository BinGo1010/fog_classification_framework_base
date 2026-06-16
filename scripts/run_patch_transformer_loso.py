#!/usr/bin/env python
"""Train/evaluate patch Transformer-BiLSTM LOSO experiments."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.patch_transformer_lstm import PatchTransformerBiLSTMClassifier


CLASS_NAMES = np.array(["NORMAL", "PRE_FOG", "FOG"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate patch-token Transformer-BiLSTM on LOSO folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/processed/fog_patch_blocks_seq128"),
        help="Directory containing patch_blocks.npz and loso_folds.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/patch_transformer_loso_13fold"),
    )
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,10,15,16,17,19")
    parser.add_argument(
        "--val-strategy",
        choices=("fold", "eventful"),
        default="eventful",
        help="Use stored fold validation subject, or choose an eventful validation subject.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--roll-pos-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly roll learned positional encoding during training.",
    )
    parser.add_argument(
        "--loss-class-weights",
        default="1,6,3",
        help="Comma-separated CE weights for NORMAL,PRE_FOG,FOG, or 'none'.",
    )
    parser.add_argument(
        "--sampler",
        choices=("none", "weighted", "manual"),
        default="manual",
        help="Block sampler. Manual uses --sampler-class-weights.",
    )
    parser.add_argument(
        "--sampler-class-weights",
        default="1,12,2",
        help="Comma-separated block weights by token class for NORMAL,PRE_FOG,FOG.",
    )
    parser.add_argument("--sampler-power", type=float, default=0.75)
    parser.add_argument(
        "--samples-per-epoch",
        default="auto",
        help="'auto' uses len(train_blocks), otherwise an integer.",
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
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_folds(spec: str, num_folds: int) -> list[int]:
    spec = str(spec).strip()
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


def parse_class_weight_arg(value: str, device: torch.device) -> torch.Tensor | None:
    value = str(value).strip().lower()
    if value in ("none", ""):
        return None
    weights = np.array([float(part.strip()) for part in value.split(",") if part.strip()])
    if len(weights) != len(CLASS_NAMES):
        raise ValueError("--loss-class-weights must be 'none' or 3 comma-separated numbers.")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def parse_sampler_class_weights(value: str) -> np.ndarray:
    weights = np.array([float(part.strip()) for part in value.split(",") if part.strip()])
    if len(weights) != len(CLASS_NAMES):
        raise ValueError("--sampler-class-weights must contain 3 values.")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("--sampler-class-weights must be non-negative and not all zero.")
    return weights.astype(np.float64)


def parse_samples_per_epoch(value: str, train_size: int) -> int:
    value = str(value).strip().lower()
    if value == "auto":
        return int(train_size)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--samples-per-epoch must be positive or 'auto'.")
    return parsed


def compute_channel_norm(
    patch_x: np.ndarray,
    patch_ids: np.ndarray,
    target_patch_samples: int,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(patch_ids), chunk_size):
        ids = patch_ids[start : start + chunk_size]
        chunk = patch_x[ids].reshape(-1, target_patch_samples, 3).astype(np.float64)
        total += chunk.sum(axis=(0, 1))
        total_sq += np.square(chunk).sum(axis=(0, 1))
        count += chunk.shape[0] * chunk.shape[1]
    mean = total / max(count, 1)
    var = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_patch_features(
    patch_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    target_patch_samples: int,
) -> np.ndarray:
    shaped = patch_x.reshape(-1, target_patch_samples, 3)
    normalized = (shaped - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.ascontiguousarray(normalized.reshape(patch_x.shape).astype(np.float32))


class PatchBlockDataset(Dataset):
    def __init__(
        self,
        patch_x: np.ndarray,
        patch_y: np.ndarray,
        block_patch_ids: np.ndarray,
        block_indices: np.ndarray,
    ) -> None:
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.block_patch_ids = block_patch_ids
        self.block_indices = block_indices.astype(np.int64, copy=False)
        self.input_dim = patch_x.shape[1]

    def __len__(self) -> int:
        return len(self.block_indices)

    def __getitem__(self, index: int):
        block_id = self.block_indices[index]
        patch_ids = self.block_patch_ids[block_id]
        mask = patch_ids >= 0
        x = np.zeros((len(patch_ids), self.input_dim), dtype=np.float32)
        y = np.zeros(len(patch_ids), dtype=np.int64)
        valid_ids = patch_ids[mask]
        if len(valid_ids):
            x[mask] = self.patch_x[valid_ids]
            y[mask] = self.patch_y[valid_ids].astype(np.int64, copy=False)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.from_numpy(patch_ids.astype(np.int64, copy=False)),
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def choose_val_code(
    fold: int,
    args: argparse.Namespace,
    subjects: np.ndarray,
    fold_val_subjects: np.ndarray,
    test_code: int,
    patch_subject_code: np.ndarray,
    patch_y: np.ndarray,
) -> int:
    if args.val_strategy == "fold":
        val_subject = str(fold_val_subjects[fold])
        return int(np.flatnonzero(subjects == val_subject)[0])

    all_mask = patch_subject_code != test_code
    global_counts = np.bincount(patch_y[all_mask], minlength=len(CLASS_NAMES)).astype(np.float64)
    global_frac = global_counts / np.maximum(global_counts.sum(), 1.0)
    candidates = []
    for code in range(len(subjects)):
        if code == test_code:
            continue
        mask = patch_subject_code == code
        counts = np.bincount(patch_y[mask], minlength=len(CLASS_NAMES)).astype(np.float64)
        if counts[1] <= 0 or counts[2] <= 0:
            continue
        frac = counts / np.maximum(counts.sum(), 1.0)
        score = float(np.abs(frac - global_frac).sum())
        candidates.append((score, -min(counts[1], counts[2]), code))
    if not candidates:
        val_subject = str(fold_val_subjects[fold])
        return int(np.flatnonzero(subjects == val_subject)[0])
    candidates.sort()
    return int(candidates[0][2])


def block_sample_weights(
    block_patch_ids: np.ndarray,
    train_block_idx: np.ndarray,
    patch_y: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if args.sampler == "none":
        return None
    if args.sampler == "manual":
        class_weight = parse_sampler_class_weights(args.sampler_class_weights)
    else:
        train_patch_ids = np.unique(block_patch_ids[train_block_idx].ravel())
        train_patch_ids = train_patch_ids[train_patch_ids >= 0]
        counts = np.bincount(patch_y[train_patch_ids], minlength=len(CLASS_NAMES)).astype(np.float64)
        inv = np.zeros_like(counts)
        nonzero = counts > 0
        inv[nonzero] = 1.0 / counts[nonzero]
        class_weight = np.power(inv, float(args.sampler_power))

    weights = np.empty(len(train_block_idx), dtype=np.float64)
    for i, block_id in enumerate(train_block_idx):
        patch_ids = block_patch_ids[block_id]
        patch_ids = patch_ids[patch_ids >= 0]
        if len(patch_ids) == 0:
            weights[i] = class_weight[0]
        else:
            token_weights = class_weight[patch_y[patch_ids]]
            weights[i] = float(token_weights.max())
    return weights


def masked_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    loss = loss.reshape(target.shape)
    mask_f = mask.to(dtype=loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict:
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
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        }

    y_bin = label_binarize(y_true, classes=labels)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
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

    for i, name in enumerate(CLASS_NAMES):
        prefix = name.lower()
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


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for xb, yb, mask, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(xb, mask=mask)
            loss = masked_ce_loss(logits, yb, mask, criterion)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tokens = int(mask.sum().item())
        total_loss += float(loss.detach().item()) * tokens
        total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def evaluate_aggregated(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    patch_y: np.ndarray,
    eval_patch_ids: np.ndarray,
    num_classes: int,
    device: torch.device,
    amp: bool,
) -> dict:
    model.eval()
    prob_sum = np.zeros((len(patch_y), num_classes), dtype=np.float64)
    counts = np.zeros(len(patch_y), dtype=np.int32)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb, mask, patch_ids in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mask_dev = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(xb, mask=mask_dev)
                loss = masked_ce_loss(logits, yb, mask_dev, criterion)

            prob = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            patch_ids_np = patch_ids.numpy()
            mask_np = mask.numpy().astype(bool)
            flat_ids = patch_ids_np[mask_np]
            flat_prob = prob[mask_np]
            np.add.at(prob_sum, flat_ids, flat_prob)
            np.add.at(counts, flat_ids, 1)

            tokens = int(mask_np.sum())
            total_loss += float(loss.detach().item()) * tokens
            total_tokens += tokens

    covered = eval_patch_ids[counts[eval_patch_ids] > 0]
    if len(covered) != len(eval_patch_ids):
        missing = len(eval_patch_ids) - len(covered)
        print(f"Warning: {missing} eval patches had no block prediction and were ignored.")
    y_prob = prob_sum[covered] / counts[covered, None]
    metrics = compute_metrics(patch_y[covered], y_prob, num_classes)
    metrics["loss"] = total_loss / max(total_tokens, 1)
    metrics["eval_patches"] = int(len(covered))
    return metrics


def run_fold(
    fold: int,
    args: argparse.Namespace,
    arrays: dict,
    config: dict,
    device: torch.device,
) -> dict:
    fold_dir = args.output_dir / f"fold_{fold:03d}"
    metrics_path = fold_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[fold {fold:03d}] resume: metrics already exist")
        return metrics

    fold_dir.mkdir(parents=True, exist_ok=True)
    patch_x = arrays["patch_X"]
    patch_y = arrays["patch_y"].astype(np.int64, copy=False)
    patch_subject_code = arrays["patch_subject_code"]
    block_patch_ids = arrays["block_patch_ids"]
    block_subject_code = arrays["block_subject_code"]
    subjects = arrays["subjects"]
    fold_test_subjects = arrays["fold_test_subjects"]
    fold_val_subjects = arrays["fold_val_subjects"]
    target_patch_samples = int(config["target_patch_samples"])
    seq_len = int(config["seq_len"])

    test_subject = str(fold_test_subjects[fold])
    test_code = int(np.flatnonzero(subjects == test_subject)[0])
    val_code = choose_val_code(
        fold,
        args,
        subjects,
        fold_val_subjects,
        test_code,
        patch_subject_code,
        patch_y,
    )
    val_subject = str(subjects[val_code])
    train_block_idx = np.flatnonzero((block_subject_code != test_code) & (block_subject_code != val_code))
    val_block_idx = np.flatnonzero(block_subject_code == val_code)
    test_block_idx = np.flatnonzero(block_subject_code == test_code)
    train_patch_ids = np.flatnonzero((patch_subject_code != test_code) & (patch_subject_code != val_code))
    val_patch_ids = np.flatnonzero(patch_subject_code == val_code)
    test_patch_ids = np.flatnonzero(patch_subject_code == test_code)

    t_prepare = time.perf_counter()
    mean, std = compute_channel_norm(patch_x, train_patch_ids, target_patch_samples)
    norm_patch_x = normalize_patch_features(patch_x, mean, std, target_patch_samples)

    train_dataset = PatchBlockDataset(norm_patch_x, patch_y, block_patch_ids, train_block_idx)
    val_dataset = PatchBlockDataset(norm_patch_x, patch_y, block_patch_ids, val_block_idx)
    test_dataset = PatchBlockDataset(norm_patch_x, patch_y, block_patch_ids, test_block_idx)

    sampler_weights = block_sample_weights(block_patch_ids, train_block_idx, patch_y, args)
    train_sampler = None
    if sampler_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights, dtype=torch.double),
            num_samples=parse_samples_per_epoch(args.samples_per_epoch, len(train_block_idx)),
            replacement=True,
        )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    prepare_sec = time.perf_counter() - t_prepare

    model = PatchTransformerBiLSTMClassifier(
        input_dim=patch_x.shape[1],
        num_classes=len(CLASS_NAMES),
        seq_len=seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.encoder_layers,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        roll_pos_encoding=args.roll_pos_encoding,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=parse_class_weight_arg(args.loss_class_weights, device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        f"train_blocks={len(train_block_idx)} val_blocks={len(val_block_idx)} "
        f"test_blocks={len(test_block_idx)} sampler={args.sampler}"
    )
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.amp,
        )
        val_metrics = evaluate_aggregated(
            model,
            val_loader,
            criterion,
            patch_y,
            val_patch_ids,
            len(CLASS_NAMES),
            device,
            args.amp,
        )
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
                    "class_names": CLASS_NAMES,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "args": vars(args),
                    "config": config,
                },
                fold_dir / "best.pt",
            )
        else:
            bad_epochs += 1

        row = {
            "fold": fold,
            "epoch": epoch,
            "epoch_sec": round(time.perf_counter() - epoch_start, 3),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_pre_fog_recall": val_metrics["pre_fog_recall"],
            "val_fog_recall": val_metrics["fog_recall"],
            "best_epoch": best_epoch,
        }
        append_csv(log_path, row)
        print(
            f"[fold {fold:03d}] epoch {epoch:02d} "
            f"loss={train_loss:.4f} val_f1={val_metrics['f1_macro']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"pre_r={val_metrics['pre_fog_recall']:.3f} "
            f"fog_r={val_metrics['fog_recall']:.3f} {'*' if improved else ''}"
        )
        if bad_epochs >= args.patience:
            print(f"[fold {fold:03d}] early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_aggregated(
        model,
        test_loader,
        criterion,
        patch_y,
        test_patch_ids,
        len(CLASS_NAMES),
        device,
        args.amp,
    )
    elapsed_sec = time.perf_counter() - fold_start

    metrics = {
        "fold": fold,
        "test_subject": test_subject,
        "val_subject": val_subject,
        "best_epoch": int(best_epoch),
        "prepare_sec": float(prepare_sec),
        "elapsed_sec": float(elapsed_sec),
        "train_blocks": int(len(train_block_idx)),
        "val_blocks": int(len(val_block_idx)),
        "test_blocks": int(len(test_block_idx)),
        "train_patches": int(len(train_patch_ids)),
        "val_patches": int(len(val_patch_ids)),
        "test_patches": int(len(test_patch_ids)),
        "train_counts": np.bincount(patch_y[train_patch_ids], minlength=len(CLASS_NAMES)).tolist(),
        "val_counts": np.bincount(patch_y[val_patch_ids], minlength=len(CLASS_NAMES)).tolist(),
        "test_counts": np.bincount(patch_y[test_patch_ids], minlength=len(CLASS_NAMES)).tolist(),
        "best_val": checkpoint["val_metrics"],
        "test": test_metrics,
    }
    save_json(metrics_path, metrics)

    del model, optimizer, criterion, train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset, norm_patch_x
    if device.type == "cuda":
        torch.cuda.empty_cache()
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
        values = [row[f"test_{key}"] for row in rows if row.get(f"test_{key}") is not None]
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

    print(f"Loading patch blocks from {args.data_dir}")
    with np.load(args.data_dir / "patch_blocks.npz") as data:
        arrays = {
            "patch_X": data["patch_X"].astype(np.float32, copy=False),
            "patch_y": data["patch_y"].astype(np.int64, copy=False),
            "patch_subject_code": data["patch_subject_code"],
            "block_patch_ids": data["block_patch_ids"],
            "block_subject_code": data["block_subject_code"],
            "subjects": data["subjects"],
        }
        config = json.loads(str(data["config_json"].item()))
    with np.load(args.data_dir / "loso_folds.npz") as folds_npz:
        arrays["fold_test_subjects"] = folds_npz["fold_test_subjects"]
        arrays["fold_val_subjects"] = folds_npz["fold_val_subjects"]

    folds = parse_folds(args.folds, len(arrays["fold_test_subjects"]))
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "folds_expanded": folds,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "dataset_config": config,
        },
    )
    print(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    print(f"patch_X={arrays['patch_X'].shape}, blocks={arrays['block_patch_ids'].shape}")
    print(f"Running folds: {folds}")

    rows = []
    total_start = time.perf_counter()
    for fold in folds:
        metrics = run_fold(fold, args, arrays, config, device)
        row = {
            "fold": metrics["fold"],
            "test_subject": metrics["test_subject"],
            "val_subject": metrics["val_subject"],
            "best_epoch": metrics["best_epoch"],
            "elapsed_sec": round(metrics["elapsed_sec"], 3),
            "train_blocks": metrics["train_blocks"],
            "val_blocks": metrics["val_blocks"],
            "test_blocks": metrics["test_blocks"],
            "train_patches": metrics["train_patches"],
            "val_patches": metrics["val_patches"],
            "test_patches": metrics["test_patches"],
            "train_normal": metrics["train_counts"][0],
            "train_pre_fog": metrics["train_counts"][1],
            "train_fog": metrics["train_counts"][2],
            "val_normal": metrics["val_counts"][0],
            "val_pre_fog": metrics["val_counts"][1],
            "val_fog": metrics["val_counts"][2],
            "test_normal": metrics["test_counts"][0],
            "test_pre_fog": metrics["test_counts"][1],
            "test_fog": metrics["test_counts"][2],
        }
        row.update(flatten_metrics("best_val", metrics["best_val"]))
        row.update(flatten_metrics("test", metrics["test"]))
        rows.append(row)
        write_csv(args.output_dir / "summary.csv", rows)
        save_json(
            args.output_dir / "aggregate.json",
            {
                "folds": rows,
                "aggregate": aggregate(rows),
                "elapsed_sec": time.perf_counter() - total_start,
            },
        )

    final = {
        "folds": rows,
        "aggregate": aggregate(rows),
        "elapsed_sec": time.perf_counter() - total_start,
    }
    save_json(args.output_dir / "aggregate.json", final)
    print("Done.")
    print(json.dumps(final["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
