#!/usr/bin/env python
"""Post-hoc threshold tuning for patch Transformer-BiLSTM experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.patch_transformer_lstm import PatchTransformerBiLSTMClassifier
from run_patch_transformer_loso import (
    CLASS_NAMES,
    PatchBlockDataset,
    normalize_patch_features,
    parse_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune PRE_FOG/FOG thresholds on validation patches and evaluate test patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/processed/fog_patch_blocks_seq128"))
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,10,15,16,17,19")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.05:1.00:0.05")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_threshold_grid(spec: str) -> np.ndarray:
    parts = [float(part) for part in spec.split(":")]
    if len(parts) == 1:
        return np.asarray(parts, dtype=np.float64)
    if len(parts) != 3:
        raise ValueError("--thresholds must be a value or start:stop:step.")
    start, stop, step = parts
    values = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    return np.clip(values, 1e-6, None)


def threshold_predict(y_prob: np.ndarray, pre_threshold: float, fog_threshold: float) -> np.ndarray:
    scores = y_prob.copy()
    scores[:, 1] /= pre_threshold
    scores[:, 2] /= fog_threshold
    return scores.argmax(axis=1)


def metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    for i, name in enumerate(CLASS_NAMES):
        prefix = name.lower()
        metrics[f"{prefix}_precision"] = float(precision[i])
        metrics[f"{prefix}_recall"] = float(recall[i])
        metrics[f"{prefix}_f1"] = float(f1[i])
        metrics[f"{prefix}_support"] = int(support[i])
    return metrics


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> tuple[float, float, dict]:
    best = (-1.0, -1.0, 0.5, 0.5, {})
    for pre_t in grid:
        for fog_t in grid:
            pred = threshold_predict(y_prob, pre_t, fog_t)
            macro = f1_score(
                y_true,
                pred,
                labels=list(range(len(CLASS_NAMES))),
                average="macro",
                zero_division=0,
            )
            bacc = balanced_accuracy_score(y_true, pred)
            if (macro, bacc) > (best[0], best[1]):
                best = (float(macro), float(bacc), float(pre_t), float(fog_t), metrics_from_pred(y_true, pred))
    return best[2], best[3], best[4]


def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    patch_y: np.ndarray,
    eval_patch_ids: np.ndarray,
    num_classes: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prob_sum = np.zeros((len(patch_y), num_classes), dtype=np.float64)
    counts = np.zeros(len(patch_y), dtype=np.int32)
    with torch.no_grad():
        for xb, _, mask, patch_ids in loader:
            xb = xb.to(device, non_blocking=True)
            mask_dev = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(xb, mask=mask_dev)
            prob = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            patch_ids_np = patch_ids.numpy()
            mask_np = mask.numpy().astype(bool)
            flat_ids = patch_ids_np[mask_np]
            np.add.at(prob_sum, flat_ids, prob[mask_np])
            np.add.at(counts, flat_ids, 1)

    covered = eval_patch_ids[counts[eval_patch_ids] > 0]
    y_prob = prob_sum[covered] / counts[covered, None]
    return patch_y[covered], y_prob


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    keys = [
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
    ]
    out = {}
    for key in keys:
        arr = np.asarray([row[f"test_{key}"] for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return out


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    device = resolve_device(args.device)
    grid = parse_threshold_grid(args.thresholds)

    with np.load(args.data_dir / "patch_blocks.npz") as data:
        patch_x = data["patch_X"].astype(np.float32, copy=False)
        patch_y = data["patch_y"].astype(np.int64, copy=False)
        patch_subject_code = data["patch_subject_code"]
        block_patch_ids = data["block_patch_ids"]
        block_subject_code = data["block_subject_code"]
        subjects = data["subjects"]
        config = json.loads(str(data["config_json"].item()))
    with np.load(args.data_dir / "loso_folds.npz") as folds_npz:
        fold_test_subjects = folds_npz["fold_test_subjects"]

    folds = parse_folds(args.folds, len(fold_test_subjects))
    rows = []
    for fold in folds:
        metrics_path = args.experiment_dir / f"fold_{fold:03d}" / "metrics.json"
        checkpoint_path = args.experiment_dir / f"fold_{fold:03d}" / "best.pt"
        with metrics_path.open("r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint["args"]
        mean = checkpoint["normalization_mean"]
        std = checkpoint["normalization_std"]
        norm_patch_x = normalize_patch_features(
            patch_x,
            mean,
            std,
            int(config["target_patch_samples"]),
        )

        test_code = int(np.flatnonzero(subjects == saved_metrics["test_subject"])[0])
        val_code = int(np.flatnonzero(subjects == saved_metrics["val_subject"])[0])
        val_block_idx = np.flatnonzero(block_subject_code == val_code)
        test_block_idx = np.flatnonzero(block_subject_code == test_code)
        val_patch_ids = np.flatnonzero(patch_subject_code == val_code)
        test_patch_ids = np.flatnonzero(patch_subject_code == test_code)

        model = PatchTransformerBiLSTMClassifier(
            input_dim=patch_x.shape[1],
            num_classes=len(CLASS_NAMES),
            seq_len=int(config["seq_len"]),
            d_model=int(ckpt_args["d_model"]),
            num_heads=int(ckpt_args["num_heads"]),
            num_encoder_layers=int(ckpt_args["encoder_layers"]),
            lstm_layers=int(ckpt_args["lstm_layers"]),
            dropout=float(ckpt_args["dropout"]),
            roll_pos_encoding=bool(ckpt_args.get("roll_pos_encoding", True)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])

        val_loader = DataLoader(
            PatchBlockDataset(norm_patch_x, patch_y, block_patch_ids, val_block_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            PatchBlockDataset(norm_patch_x, patch_y, block_patch_ids, test_block_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )
        y_val, p_val = collect_probs(model, val_loader, patch_y, val_patch_ids, len(CLASS_NAMES), device, args.amp)
        pre_t, fog_t, val_metrics = tune_thresholds(y_val, p_val, grid)
        y_test, p_test = collect_probs(model, test_loader, patch_y, test_patch_ids, len(CLASS_NAMES), device, args.amp)
        test_pred = threshold_predict(p_test, pre_t, fog_t)
        test_metrics = metrics_from_pred(y_test, test_pred)

        row = {
            "fold": fold,
            "test_subject": saved_metrics["test_subject"],
            "val_subject": saved_metrics["val_subject"],
            "pre_threshold": pre_t,
            "fog_threshold": fog_t,
        }
        for key, value in val_metrics.items():
            if key != "confusion_matrix":
                row[f"val_{key}"] = value
        for key, value in test_metrics.items():
            if key != "confusion_matrix":
                row[f"test_{key}"] = value
        rows.append(row)
        print(
            f"[fold {fold:03d}] thresholds pre={pre_t:.2f} fog={fog_t:.2f} "
            f"test_f1={test_metrics['f1_macro']:.4f} "
            f"pre_r={test_metrics['pre_fog_recall']:.3f} fog_r={test_metrics['fog_recall']:.3f}"
        )
        del model, norm_patch_x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.experiment_dir / "summary_threshold.csv", rows)
    payload = {"folds": rows, "aggregate": aggregate(rows)}
    with (args.experiment_dir / "aggregate_threshold.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
