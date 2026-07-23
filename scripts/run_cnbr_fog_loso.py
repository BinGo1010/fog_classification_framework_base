#!/usr/bin/env python
"""Run conditional normal-behaviour residual FoG detection with strict LOSO.

The normal predictor sees only a historical trunk-accelerometer context and is
trained only on clean non-FOG context/target pairs.  It forecasts a Gaussian
future block.  The uncertainty-standardised forecast error is then classified
with a small TCN.  A raw-target TCN with the same classifier architecture is
included as a representation baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetTrunkDataset, SequenceWindowDataset, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
    save_json,
)
from cnbr_fog.histories import (
    HistoryPlan,
    history_block_count,
    make_common_history_plan,
    make_history_input,
)
from cnbr_fog.models import (
    ConditionalNormalPredictor,
    ResidualTCNClassifier,
    gaussian_nll,
)


METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "auroc",
    "auprc",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional normal-behaviour residual FoG LOSO experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"E:\fog-merged\dataset\1.Daphnet Freezing of Gait Dataset\processed_trunk"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cnbr_fog_daphnet_trunk_loso"),
    )
    parser.add_argument(
        "--folds",
        default="all",
        help="all, comma-separated subjects (S01,S03), or numeric indices (0,2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-subjects",
        default="",
        help="Comma-separated subjects removed before scaling, windowing, and LOSO",
    )
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.25)
    parser.add_argument("--normal-guard-seconds", type=float, default=0.5)
    parser.add_argument("--fog-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--flatline-seconds", type=float, default=1.0)
    parser.add_argument("--robust-clip", type=float, default=12.0)
    parser.add_argument("--residual-clip", type=float, default=12.0)
    parser.add_argument("--normal-hidden", type=int, default=48)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--normal-epochs", type=int, default=8)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--normal-patience", type=int, default=3)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--normal-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-normal-windows",
        type=int,
        default=30000,
        help="Deterministic cap per fold; 0 uses every clean normal window",
    )
    parser.add_argument(
        "--max-classifier-windows",
        type=int,
        default=0,
        help="Stratified deterministic cap for smoke tests; 0 uses every window",
    )
    parser.add_argument(
        "--baselines",
        default="residual,raw",
        help="Comma-separated inputs: residual and/or raw",
    )
    parser.add_argument(
        "--residual-history-seconds",
        default="",
        help=(
            "Optional comma-separated classifier histories (for example "
            "0.5,1,2,4). Empty preserves the original single-block inputs."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def parse_folds(spec: str, subjects: list[str]) -> list[str]:
    spec = str(spec).strip()
    if spec.lower() == "all":
        return subjects
    result: list[str] = []
    for value in spec.split(","):
        value = value.strip().upper()
        if not value:
            continue
        if value in subjects:
            result.append(value)
        elif value.isdigit() and 0 <= int(value) < len(subjects):
            result.append(subjects[int(value)])
        else:
            raise ValueError(f"Unknown fold {value!r}; subjects={subjects}")
    return result


def parse_subject_list(spec: str) -> list[str]:
    """Parse a comma-separated subject list without silently accepting repeats."""

    values = [value.strip().upper() for value in str(spec).split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate subjects in {spec!r}")
    return values


def parse_history_variants(
    spec: str,
    sampling_rate_hz: int,
    horizon_samples: int,
    stride_samples: int,
) -> list[tuple[str, float, int]]:
    """Resolve history seconds into stable variant ids and exact sample lengths."""

    if not str(spec).strip():
        return []
    seconds: list[float] = []
    for value in str(spec).split(","):
        value = value.strip()
        if not value:
            continue
        duration = float(value)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid residual history duration: {value!r}")
        if duration not in seconds:
            seconds.append(duration)
    variants: list[tuple[str, float, int]] = []
    seen_samples: set[int] = set()
    for duration in sorted(seconds):
        samples = int(round(duration * sampling_rate_hz))
        if samples in seen_samples:
            raise ValueError("Multiple history durations resolve to the same sample count")
        history_block_count(samples, horizon_samples, stride_samples)
        resolved = samples / float(sampling_rate_hz)
        label = f"{resolved:g}".replace(".", "p")
        variants.append((f"residual_h{label}s", resolved, samples))
        seen_samples.add(samples)
    return variants


def select_validation_subject(
    test_subject: str,
    subjects: list[str],
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
) -> str:
    """Choose the next cyclic training subject containing both classes."""

    start = subjects.index(test_subject)
    for offset in range(1, len(subjects)):
        candidate = subjects[(start + offset) % len(subjects)]
        indices = dataset.window_indices_for_subjects(windows, [candidate])
        labels = windows.label[indices]
        if np.unique(labels).size == 2:
            return candidate
    raise RuntimeError("No validation subject with both classes is available")


def deterministic_subsample(
    indices: np.ndarray,
    maximum: int,
    seed: int,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(indices, size=maximum, replace=False))
    picked: list[np.ndarray] = []
    labels_for_indices = np.asarray(labels)[indices]
    classes, counts = np.unique(labels_for_indices, return_counts=True)
    allocation = np.maximum(1, np.floor(maximum * counts / counts.sum()).astype(int))
    while allocation.sum() > maximum:
        allocation[np.argmax(allocation)] -= 1
    while allocation.sum() < maximum:
        room = counts - allocation
        allocation[np.argmax(room)] += 1
    for cls, count in zip(classes, allocation):
        cls_indices = indices[labels_for_indices == cls]
        picked.append(rng.choice(cls_indices, size=min(int(count), len(cls_indices)), replace=False))
    return np.sort(np.concatenate(picked))


def make_sequence_loader(
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        SequenceWindowDataset(dataset.records, windows, indices, scaler),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def normal_epoch(
    model: ConditionalNormalPredictor,
    loader: DataLoader,
    context_samples: int,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :context_samples]
        target = sequence[:, :, context_samples:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                mean, logvar = model(context)
                loss = gaussian_nll(target, mean, logvar)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        batch = int(sequence.size(0))
        total_loss += float(loss.detach()) * batch
        total_n += batch
    return total_loss / max(total_n, 1)


def train_normal_predictor(
    fold_dir: Path,
    args: argparse.Namespace,
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    scaler_stats,
    context_samples: int,
    horizon_samples: int,
    device: torch.device,
) -> tuple[ConditionalNormalPredictor, dict]:
    checkpoint = fold_dir / "normal_predictor_best.pt"
    model = ConditionalNormalPredictor(
        in_channels=3,
        horizon=horizon_samples,
        hidden_channels=args.normal_hidden,
        dropout=args.dropout,
    ).to(device)
    if args.resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        return model, payload["training"]

    pin = device.type == "cuda"
    train_loader = make_sequence_loader(
        dataset,
        windows,
        train_indices,
        scaler_stats,
        args.batch_size,
        True,
        args.num_workers,
        pin,
    )
    val_loader = make_sequence_loader(
        dataset,
        windows,
        val_indices,
        scaler_stats,
        args.batch_size,
        False,
        args.num_workers,
        pin,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.normal_lr, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.normal_epochs + 1):
        train_loss = normal_epoch(
            model,
            train_loader,
            context_samples,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            val_loss = normal_epoch(
                model, val_loader, context_samples, device, args.amp
            )
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        improved = val_loss < best_loss - 1e-5
        print(
            f"    normal epoch={epoch:02d} train_nll={train_loss:.5f} "
            f"val_nll={val_loss:.5f}{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "training": {
                        "best_epoch": best_epoch,
                        "best_val_nll": best_loss,
                        "train_windows": int(len(train_indices)),
                        "val_windows": int(len(val_indices)),
                        "elapsed_sec": float(time.perf_counter() - start_time),
                        "history": history,
                    },
                },
                checkpoint,
            )
        else:
            bad_epochs += 1
        if bad_epochs >= args.normal_patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model, payload["training"]


@torch.no_grad()
def extract_inputs(
    model: ConditionalNormalPredictor,
    args: argparse.Namespace,
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler_stats,
    context_samples: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = make_sequence_loader(
        dataset,
        windows,
        indices,
        scaler_stats,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    model.eval()
    residuals: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    means: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :context_samples]
        target = sequence[:, :, context_samples:]
        with torch.amp.autocast(device.type, enabled=args.amp and device.type == "cuda"):
            mean, logvar = model(context)
            z = (target - mean) * torch.exp(-0.5 * logvar)
        z = z.clamp(-args.residual_clip, args.residual_clip)
        residuals.append(z.float().cpu().numpy())
        raw_targets.append(target.float().cpu().numpy())
        means.append(mean.float().cpu().numpy())
        logvars.append(logvar.float().cpu().numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())
    return {
        "residual": np.concatenate(residuals).astype(np.float32, copy=False),
        "raw": np.concatenate(raw_targets).astype(np.float32, copy=False),
        "mean": np.concatenate(means).astype(np.float32, copy=False),
        "logvar": np.concatenate(logvars).astype(np.float32, copy=False),
        "y": np.concatenate(labels).astype(np.int8, copy=False),
        "window_index": np.concatenate(window_indices).astype(np.int64, copy=False),
    }


def feature_diagnostics(extracted: dict[str, np.ndarray]) -> dict:
    error = extracted["raw"] - extracted["mean"]
    z = extracted["residual"]
    result = {
        "forecast_rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "forecast_mae": float(np.mean(np.abs(error.astype(np.float64)))),
        "mean_predicted_sigma": float(np.mean(np.exp(0.5 * extracted["logvar"]))),
    }
    for label, name in [(0, "normal"), (1, "fog")]:
        mask = extracted["y"] == label
        if not mask.any():
            continue
        values = z[mask].astype(np.float64)
        result[f"{name}_z_mean"] = float(values.mean())
        result[f"{name}_z_std"] = float(values.std())
        result[f"{name}_z_abs_mean"] = float(np.abs(values).mean())
        result[f"{name}_z_rms"] = float(np.sqrt(np.mean(values**2)))
    return result


def history_support_summary(
    extracted: dict[str, np.ndarray],
    plan: HistoryPlan,
) -> dict:
    labels = np.asarray(extracted["y"], dtype=np.int8)[plan.anchor_rows]
    return {
        "windows": int(len(plan.anchor_rows)),
        "class_counts": np.bincount(labels, minlength=2).astype(int).tolist(),
    }


def make_anchor_raw_input(
    extracted: dict[str, np.ndarray],
    plan: HistoryPlan,
) -> dict[str, np.ndarray]:
    """Restrict the raw-target baseline to the common history anchor set."""

    return {
        "raw": np.asarray(extracted["raw"])[plan.anchor_rows],
        "y": np.asarray(extracted["y"])[plan.anchor_rows],
        "window_index": plan.anchor_window_indices,
    }


def classifier_epoch(
    model: ResidualTCNClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    truths: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        batch = int(y.numel())
        total_loss += float(loss.detach()) * batch
        total_n += batch
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
    return total_loss / max(total_n, 1), np.concatenate(truths), np.concatenate(probs)


def array_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def train_classifier(
    fold_dir: Path,
    name: str,
    args: argparse.Namespace,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model_dir = fold_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "classifier_best.pt"
    metrics_path = model_dir / "metrics.json"
    prediction_path = model_dir / "predictions.npz"
    if args.resume and metrics_path.exists() and checkpoint.exists() and prediction_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with np.load(prediction_path, allow_pickle=False) as payload:
            saved_window_index = np.asarray(payload["window_index"], dtype=np.int64)
            saved_y_true = np.asarray(payload["y_true"], dtype=np.int8)
            saved_y_prob = np.asarray(payload["y_prob"], dtype=np.float32)
        # AMP may produce float16 probabilities.  Always promote before thresholding
        # so persisted decisions exactly match binary_metrics' float64 comparison.
        saved_y_pred = (
            saved_y_prob.astype(np.float64) >= float(metrics["threshold"])
        ).astype(np.int8)
        np.savez_compressed(
            prediction_path,
            window_index=saved_window_index,
            y_true=saved_y_true,
            y_prob=saved_y_prob,
            y_pred=saved_y_pred,
        )
        return metrics, saved_y_prob, saved_y_pred

    x_train = train[name]
    y_train = train["y"]
    x_val = val[name]
    y_val = val["y"]
    x_test = test[name]
    y_test = test["y"]
    model = ResidualTCNClassifier(
        in_channels=x_train.shape[1],
        hidden_channels=args.classifier_hidden,
        dropout=args.dropout,
    ).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    ratio = counts[0] / max(counts[1], 1.0)
    pos_weight = torch.tensor(min(math.sqrt(ratio), 6.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    pin = device.type == "cuda"
    train_loader = array_loader(
        x_train, y_train, args.batch_size, True, args.num_workers, pin
    )
    val_loader = array_loader(
        x_val, y_val, args.batch_size, False, args.num_workers, pin
    )
    test_loader = array_loader(
        x_test, y_test, args.batch_size, False, args.num_workers, pin
    )

    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.classifier_epochs + 1):
        train_loss, train_true, train_prob = classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            val_loss, val_true, val_prob = classifier_epoch(
                model, val_loader, criterion, device, args.amp
            )
        score = float(average_precision_score(val_true, val_prob))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auprc": float(average_precision_score(train_true, train_prob)),
                "val_loss": val_loss,
                "val_auprc": score,
            }
        )
        improved = score > best_score + 1e-5
        print(
            f"    {name} epoch={epoch:02d} train_loss={train_loss:.5f} "
            f"val_auprc={score:.5f}{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_auprc": best_score,
                    "history": history,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1
        if bad_epochs >= args.classifier_patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    with torch.no_grad():
        _, val_true, val_prob = classifier_epoch(
            model, val_loader, criterion, device, args.amp
        )
        _, test_true, test_prob = classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, val_metrics = choose_threshold(val_true, val_prob)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (
        np.asarray(test_prob, dtype=np.float64) >= float(threshold)
    ).astype(np.int8)
    metrics = {
        **test_metrics,
        "input": name,
        "best_epoch": int(payload["best_epoch"]),
        "best_val_auprc": float(payload["best_val_auprc"]),
        "validation": val_metrics,
        "train_counts": counts.astype(int).tolist(),
        "pos_weight": float(pos_weight.item()),
        "elapsed_sec": float(time.perf_counter() - start_time),
        "history": payload["history"],
    }
    save_json(metrics_path, metrics)
    np.savez_compressed(
        prediction_path,
        window_index=test["window_index"],
        y_true=test_true,
        y_prob=test_prob.astype(np.float32),
        y_pred=test_pred,
    )
    return metrics, test_prob, test_pred


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def event_metrics(
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    y_pred: np.ndarray,
    minimum_positive_windows: int = 2,
    merge_gap_seconds: float = 0.5,
) -> dict:
    """Compute overlap-based event detection metrics on one held-out subject."""

    fs = dataset.sampling_rate_hz
    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for global_index, pred in zip(window_indices, y_pred):
        rec_idx = int(windows.record_index[global_index])
        by_record[rec_idx].append(
            (
                int(windows.target_start[global_index]),
                int(windows.target_end[global_index]),
                int(pred),
            )
        )

    true_total = 0
    true_detected = 0
    predicted_total = 0
    matched_predictions = 0
    delays: list[float] = []
    evaluated_seconds = 0.0
    merge_gap = int(round(merge_gap_seconds * fs))

    for rec_idx, rows in by_record.items():
        record = dataset.records[rec_idx]
        evaluated_seconds += float(record.valid.sum()) / fs
        rows.sort(key=lambda item: item[0])
        predicted_intervals: list[tuple[int, int, int]] = []
        pred_values = np.asarray([row[2] for row in rows], dtype=np.int8)
        for run_start, run_end in _boolean_runs(pred_values == 1):
            if run_end - run_start < minimum_positive_windows:
                continue
            interval_start = rows[run_start][0]
            interval_end = rows[run_end - 1][1]
            decision_sample = rows[run_start + minimum_positive_windows - 1][1]
            if predicted_intervals and interval_start - predicted_intervals[-1][1] <= merge_gap:
                previous = predicted_intervals[-1]
                predicted_intervals[-1] = (previous[0], interval_end, previous[2])
            else:
                predicted_intervals.append((interval_start, interval_end, decision_sample))

        true_intervals = _boolean_runs(record.y == 1)
        # Only score events that overlap at least one evaluated target window.
        target_coverage = [(row[0], row[1]) for row in rows]
        true_intervals = [
            interval
            for interval in true_intervals
            if any(max(interval[0], start) < min(interval[1], end) for start, end in target_coverage)
        ]
        true_total += len(true_intervals)
        predicted_total += len(predicted_intervals)
        used_predictions: set[int] = set()
        for true_start, true_end in true_intervals:
            matches = [
                index
                for index, (pred_start, pred_end, _) in enumerate(predicted_intervals)
                if index not in used_predictions
                and max(true_start, pred_start) < min(true_end, pred_end)
            ]
            if not matches:
                continue
            match = min(matches, key=lambda index: predicted_intervals[index][0])
            used_predictions.add(match)
            true_detected += 1
            _, _, decision_sample = predicted_intervals[match]
            delays.append(max(0.0, (decision_sample - true_start) / fs))
        matched_predictions += len(used_predictions)

    false_events = predicted_total - matched_predictions
    return {
        "evaluable_true_events": int(true_total),
        "detected_true_events": int(true_detected),
        "predicted_events": int(predicted_total),
        "false_alarm_events": int(false_events),
        "event_sensitivity": true_detected / true_total if true_total else None,
        "false_alarm_events_per_hour": (
            false_events / (evaluated_seconds / 3600.0) if evaluated_seconds else None
        ),
        "median_detection_delay_sec": float(np.median(delays)) if delays else None,
        "mean_detection_delay_sec": float(np.mean(delays)) if delays else None,
        "evaluated_hours": evaluated_seconds / 3600.0,
    }


def write_predictions_csv(
    path: Path,
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    window_indices: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "record_id",
        "run_id",
        "window_start",
        "target_start",
        "target_end_exclusive",
        "fog_fraction",
        "y_true",
        "y_prob",
        "y_pred",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for global_index, probability, prediction in zip(window_indices, y_prob, y_pred):
            rec = dataset.records[int(windows.record_index[global_index])]
            writer.writerow(
                {
                    "subject_id": rec.subject_id,
                    "record_id": rec.record_id,
                    "run_id": rec.run_id,
                    "window_start": int(windows.start[global_index]),
                    "target_start": int(windows.target_start[global_index]),
                    "target_end_exclusive": int(windows.target_end[global_index]),
                    "fog_fraction": float(windows.fog_fraction[global_index]),
                    "y_true": int(windows.label[global_index]),
                    "y_prob": float(probability),
                    "y_pred": int(prediction),
                }
            )


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = [
        "input",
        "test_subject",
        "val_subject",
        "threshold",
        "n",
        "n_fog",
        *METRIC_KEYS,
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pooled_metrics(
    rows: list[dict],
    output_dir: Path | None = None,
    baseline: str | None = None,
) -> dict:
    if not rows:
        return {}
    tn = sum(int(row["tn"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision and sensitivity else 0.0
    result = {
        "n": tn + fp + fn + tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(tn + fp + fn + tp, 1),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            0.5 * (sensitivity + specificity)
            if sensitivity is not None and specificity is not None
            else None
        ),
        "f1": f1,
    }
    if output_dir is not None and baseline is not None:
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for row in rows:
            path = output_dir / f"loso_{row['test_subject']}" / baseline / "predictions.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as payload:
                truths.append(np.asarray(payload["y_true"], dtype=np.int8))
                probabilities.append(np.asarray(payload["y_prob"], dtype=np.float64))
        if truths:
            y_true = np.concatenate(truths)
            y_prob = np.concatenate(probabilities)
            if np.unique(y_true).size == 2:
                result["auroc"] = float(roc_auc_score(y_true, y_prob))
                result["auprc"] = float(average_precision_score(y_true, y_prob))
    return result


def run_fold(
    args: argparse.Namespace,
    dataset: DaphnetTrunkDataset,
    windows: WindowTable,
    test_subject: str,
    context_samples: int,
    horizon_samples: int,
    stride_samples: int,
    baselines: list[str],
    history_variants: list[tuple[str, float, int]],
    device: torch.device,
) -> list[dict]:
    fold_index = dataset.subjects.index(test_subject)
    set_seed(args.seed + fold_index)
    fold_dir = args.output_dir / f"loso_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_subject = select_validation_subject(test_subject, dataset.subjects, dataset, windows)
    train_subjects = [
        subject
        for subject in dataset.subjects
        if subject not in {test_subject, val_subject}
    ]
    scaler_stats = dataset.fit_scaler(train_subjects, clip=args.robust_clip)
    train_indices = dataset.window_indices_for_subjects(windows, train_subjects)
    val_indices = dataset.window_indices_for_subjects(windows, [val_subject])
    test_indices = dataset.window_indices_for_subjects(windows, [test_subject])
    normal_train_indices = dataset.window_indices_for_subjects(
        windows, train_subjects, clean_normal_only=True
    )
    normal_val_indices = dataset.window_indices_for_subjects(
        windows, [val_subject], clean_normal_only=True
    )
    normal_train_indices = deterministic_subsample(
        normal_train_indices,
        args.max_normal_windows,
        args.seed + dataset.subjects.index(test_subject),
    )
    history_mode = bool(history_variants)
    if not history_mode:
        train_indices = deterministic_subsample(
            train_indices,
            args.max_classifier_windows,
            args.seed + 100 + fold_index,
            windows.label,
        )
    print(
        f"[fold {test_subject}] train={train_subjects} val={val_subject} "
        f"source windows train/val/test={len(train_indices)}/{len(val_indices)}/{len(test_indices)} "
        f"normal={len(normal_train_indices)}/{len(normal_val_indices)}",
        flush=True,
    )

    model, normal_training = train_normal_predictor(
        fold_dir,
        args,
        dataset,
        windows,
        normal_train_indices,
        normal_val_indices,
        scaler_stats,
        context_samples,
        horizon_samples,
        device,
    )
    train_features = extract_inputs(
        model,
        args,
        dataset,
        windows,
        train_indices,
        scaler_stats,
        context_samples,
        device,
    )
    val_features = extract_inputs(
        model,
        args,
        dataset,
        windows,
        val_indices,
        scaler_stats,
        context_samples,
        device,
    )
    test_features = extract_inputs(
        model,
        args,
        dataset,
        windows,
        test_indices,
        scaler_stats,
        context_samples,
        device,
    )
    diagnostics = {
        "train": feature_diagnostics(train_features),
        "validation": feature_diagnostics(val_features),
        "test": feature_diagnostics(test_features),
    }

    train_plan: HistoryPlan | None = None
    val_plan: HistoryPlan | None = None
    test_plan: HistoryPlan | None = None
    support: dict | None = None
    if history_mode:
        max_history_samples = max(samples for _, _, samples in history_variants)
        train_plan = make_common_history_plan(
            windows,
            train_features["window_index"],
            horizon_samples,
            stride_samples,
            max_history_samples,
        )
        val_plan = make_common_history_plan(
            windows,
            val_features["window_index"],
            horizon_samples,
            stride_samples,
            max_history_samples,
        )
        test_plan = make_common_history_plan(
            windows,
            test_features["window_index"],
            horizon_samples,
            stride_samples,
            max_history_samples,
        )
        if min(len(train_plan.anchor_rows), len(val_plan.anchor_rows), len(test_plan.anchor_rows)) == 0:
            raise RuntimeError(f"Empty common history support in fold {test_subject}")
        if args.max_classifier_windows > 0:
            plan_rows = np.arange(len(train_plan.anchor_rows), dtype=np.int64)
            plan_labels = train_features["y"][train_plan.anchor_rows]
            selected = deterministic_subsample(
                plan_rows,
                args.max_classifier_windows,
                args.seed + 100 + fold_index,
                plan_labels,
            )
            train_plan = train_plan.take(selected)
        support = {
            "policy": "maximum_history_common_anchors",
            "train": history_support_summary(train_features, train_plan),
            "validation": history_support_summary(val_features, val_plan),
            "test": history_support_summary(test_features, test_plan),
        }
        np.savez_compressed(
            fold_dir / "history_support.npz",
            train_anchor_window_index=train_plan.anchor_window_indices,
            validation_anchor_window_index=val_plan.anchor_window_indices,
            test_anchor_window_index=test_plan.anchor_window_indices,
            train_history_window_index=train_features["window_index"][train_plan.max_chain_rows],
            validation_history_window_index=val_features["window_index"][val_plan.max_chain_rows],
            test_history_window_index=test_features["window_index"][test_plan.max_chain_rows],
        )
        print(
            f"[fold {test_subject}] common anchors train/val/test="
            f"{len(train_plan.anchor_rows)}/{len(val_plan.anchor_rows)}/"
            f"{len(test_plan.anchor_rows)}",
            flush=True,
        )
    save_json(
        fold_dir / "fold_config.json",
        {
            "test_subject": test_subject,
            "val_subject": val_subject,
            "train_subjects": train_subjects,
            "scaler": scaler_stats.as_dict(),
            "normal_training": normal_training,
            "diagnostics": diagnostics,
            "history_support": support,
        },
    )

    rows: list[dict] = []
    if history_mode:
        input_names = [name for name, _, _ in history_variants if "residual" in baselines]
        if "raw" in baselines:
            input_names.append("raw")
    else:
        input_names = baselines
    history_lookup = {
        name: (seconds, samples) for name, seconds, samples in history_variants
    }
    for input_name in input_names:
        classifier_seed = args.seed + 10000 + fold_index
        set_seed(classifier_seed)
        if input_name in history_lookup:
            assert train_plan is not None and val_plan is not None and test_plan is not None
            history_seconds, history_samples = history_lookup[input_name]
            classifier_train = make_history_input(
                train_features,
                train_plan,
                input_name,
                history_samples,
                horizon_samples,
                stride_samples,
            )
            classifier_val = make_history_input(
                val_features,
                val_plan,
                input_name,
                history_samples,
                horizon_samples,
                stride_samples,
            )
            classifier_test = make_history_input(
                test_features,
                test_plan,
                input_name,
                history_samples,
                horizon_samples,
                stride_samples,
            )
        elif history_mode and input_name == "raw":
            assert train_plan is not None and val_plan is not None and test_plan is not None
            history_seconds = None
            history_samples = horizon_samples
            classifier_train = make_anchor_raw_input(train_features, train_plan)
            classifier_val = make_anchor_raw_input(val_features, val_plan)
            classifier_test = make_anchor_raw_input(test_features, test_plan)
        else:
            history_seconds = args.horizon_seconds if input_name == "residual" else None
            history_samples = horizon_samples
            classifier_train = train_features
            classifier_val = val_features
            classifier_test = test_features
        metrics, test_prob, test_pred = train_classifier(
            fold_dir,
            input_name,
            args,
            classifier_train,
            classifier_val,
            classifier_test,
            device,
        )
        events = event_metrics(
            dataset,
            windows,
            classifier_test["window_index"],
            test_pred,
        )
        metrics.update(events)
        metrics.update(
            {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "history_seconds": history_seconds,
                "input_samples": int(classifier_train[input_name].shape[-1]),
                "history_blocks": (
                    history_block_count(history_samples, horizon_samples, stride_samples)
                    if input_name in history_lookup
                    else 1
                ),
                "classifier_seed": classifier_seed,
            }
        )
        save_json(fold_dir / input_name / "metrics.json", metrics)
        write_predictions_csv(
            fold_dir / input_name / "predictions.csv",
            dataset,
            windows,
            classifier_test["window_index"],
            test_prob,
            test_pred,
        )
        rows.append(metrics)
        print(
            f"[fold {test_subject}] {input_name} AUPRC={metrics['auprc']} "
            f"F1={metrics['f1']} Sens={metrics['sensitivity']} "
            f"Spec={metrics['specificity']}",
            flush=True,
        )
        del classifier_train, classifier_val, classifier_test
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del train_features, val_features, test_features, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = DaphnetTrunkDataset.load(
        args.data_dir, flatline_seconds=args.flatline_seconds
    )
    source_subjects = list(dataset.subjects)
    excluded_subjects = parse_subject_list(args.exclude_subjects)
    unknown_exclusions = sorted(set(excluded_subjects) - set(source_subjects))
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded subjects: {unknown_exclusions}")
    if excluded_subjects:
        excluded_set = set(excluded_subjects)
        dataset = DaphnetTrunkDataset(
            root=dataset.root,
            records=[
                record for record in dataset.records if record.subject_id not in excluded_set
            ],
            sampling_rate_hz=dataset.sampling_rate_hz,
        )
    if len(dataset.subjects) < 3:
        raise ValueError("At least three subjects are required for train/validation/test")
    fs = dataset.sampling_rate_hz
    context_samples = int(round(args.context_seconds * fs))
    horizon_samples = int(round(args.horizon_seconds * fs))
    stride_samples = int(round(args.stride_seconds * fs))
    guard_samples = int(round(args.normal_guard_seconds * fs))
    windows = dataset.make_windows(
        warmup_samples=context_samples,
        target_samples=horizon_samples,
        stride_samples=stride_samples,
        fog_fraction_threshold=args.fog_fraction_threshold,
        normal_guard_samples=guard_samples,
    )
    baselines = [value.strip() for value in args.baselines.split(",") if value.strip()]
    unknown = sorted(set(baselines) - {"residual", "raw"})
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")
    history_variants = parse_history_variants(
        args.residual_history_seconds,
        fs,
        horizon_samples,
        stride_samples,
    )
    if history_variants and "residual" not in baselines:
        raise ValueError("Residual history variants require --baselines residual")
    if history_variants:
        input_names = [name for name, _, _ in history_variants]
        if "raw" in baselines:
            input_names.append("raw")
        max_history_samples = max(samples for _, _, samples in history_variants)
        global_support = make_common_history_plan(
            windows,
            np.arange(len(windows), dtype=np.int64),
            horizon_samples,
            stride_samples,
            max_history_samples,
        )
        evaluation_window_indices = global_support.anchor_window_indices
    else:
        input_names = baselines
        evaluation_window_indices = np.arange(len(windows), dtype=np.int64)
    folds = parse_folds(args.folds, dataset.subjects)
    fold_record_indices = set(dataset.subject_record_indices(folds).astype(int).tolist())
    evaluation_window_indices = evaluation_window_indices[
        np.fromiter(
            (
                int(record_index) in fold_record_indices
                for record_index in windows.record_index[evaluation_window_indices]
            ),
            dtype=bool,
            count=len(evaluation_window_indices),
        )
    ]
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "sampling_rate_hz": fs,
        "context_samples": context_samples,
        "horizon_samples": horizon_samples,
        "stride_samples": stride_samples,
        "normal_guard_samples": guard_samples,
        "source_subjects": source_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects": dataset.subjects,
        "folds_resolved": folds,
        "inputs_resolved": input_names,
        "history_variants": [
            {
                "input": name,
                "history_seconds": seconds,
                "history_samples": samples,
                "history_blocks": history_block_count(
                    samples, horizon_samples, stride_samples
                ),
            }
            for name, seconds, samples in history_variants
        ],
        "history_construction": (
            "non_overlapping_horizon_spaced_blocks_common_maximum_support"
            if history_variants
            else None
        ),
        "records": len(dataset.records),
        "windows": len(windows),
        "window_class_counts": np.bincount(windows.label, minlength=2).tolist(),
        "evaluation_windows": int(len(evaluation_window_indices)),
        "evaluation_window_class_counts": np.bincount(
            windows.label[evaluation_window_indices], minlength=2
        ).astype(int).tolist(),
        "invalid_samples": int(sum((~record.valid).sum() for record in dataset.records)),
    }
    save_json(args.output_dir / "config.json", config)
    print(
        f"[INFO] device={device} records={len(dataset.records)} subjects={dataset.subjects} "
        f"windows={len(windows)} common={config['evaluation_windows']} "
        f"counts={config['evaluation_window_class_counts']} folds={folds} inputs={input_names}",
        flush=True,
    )

    all_rows: list[dict] = []
    for test_subject in folds:
        all_rows.extend(
            run_fold(
                args,
                dataset,
                windows,
                test_subject,
                context_samples,
                horizon_samples,
                stride_samples,
                baselines,
                history_variants,
                device,
            )
        )
        write_summary_csv(args.output_dir / "fold_summary.csv", all_rows)

    aggregate: dict[str, dict] = {}
    for input_name in input_names:
        rows = [row for row in all_rows if row["input"] == input_name]
        aggregate[input_name] = {
            "subject_macro": aggregate_fold_metrics(rows, METRIC_KEYS),
            "pooled": pooled_metrics(rows, args.output_dir, input_name),
            "completed_folds": [row["test_subject"] for row in rows],
        }
    save_json(args.output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
