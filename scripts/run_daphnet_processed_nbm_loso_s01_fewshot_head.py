#!/usr/bin/env python3
"""Strict S01 few-shot personalization of the frozen LOSO GRU-NBM + TCN.

The source scaler, GRU-v1 MASK8_12 NBM, residual scale, and TCN are loaded from
the completed zero-shot S01 LOSO run.  S01_seg002 is reserved for
personalization: complete windows in 150--300 s form the support set and
complete windows in 300--350 s form the calibration set.  The final query set
is exclusively S01_seg000 and S01_seg001 and is not materialized until the
personalization barrier has been written and verified.

Three paired arms are evaluated on exactly the same query windows:

* ZERO_SHOT: frozen source network and its original S02 threshold;
* THRESHOLD_ONLY: frozen source network, exact S01 calibration threshold;
* HEAD_FINE_TUNE: only the final Linear TCN head is updated on S01 support,
  with epoch selection and exact threshold calibration on S01 calibration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.resume import atomic_json_dump, atomic_torch_save, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_gru_mask_strength_nbm300_fold import (
    PARAMETER_COUNT,
    architecture_config,
    checkpoint_name,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    RoleRows,
    RobustScaler,
    load_fold_rows,
    write_csv,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
    classifier_predict,
)
from scripts.run_daphnet_processed_nbm_loso_s01_gru_v1_c_tcn import (
    SOURCE_OUTER_FOLD,
    TEST_SUBJECT,
    feature_values,
    manifest_rows,
    resolve_device,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    set_seed,
)

EXPERIMENT_VERSION = "processed_nbm_loso_s01_fewshot_head.v1"
PERSONALIZATION_SEED = 0
ZERO_SHOT = "ZERO_SHOT"
THRESHOLD_ONLY = "THRESHOLD_ONLY"
HEAD_FINE_TUNE = "HEAD_FINE_TUNE"
ARMS = (ZERO_SHOT, THRESHOLD_ONLY, HEAD_FINE_TUNE)

SUPPORT_RECORD = "S01_seg002"
SUPPORT_START = 150 * 64
SUPPORT_END = 300 * 64
CALIBRATION_RECORD = "S01_seg002"
CALIBRATION_START = 300 * 64
CALIBRATION_END = 350 * 64
QUERY_RECORDS = ("S01_seg000", "S01_seg001")
EXPECTED_SPLIT_COUNTS = {
    "support": {"windows": 136, "nonfog": 127, "fog": 9},
    "calibration": {"windows": 41, "nonfog": 36, "fog": 5},
    "query": {"windows": 1380, "nonfog": 1330, "fog": 50},
}

HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4
HEAD_BATCH_SIZE = 64
HEAD_MAX_EPOCHS = 30
HEAD_PATIENCE = 5
HEAD_GRADIENT_CLIP = 1.0
PERSONALIZATION_BARRIER = "PERSONALIZATION_BARRIER.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_loso_S01_gru_mask8_12_C_tcn_ep5pat2_seed0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_loso_S01_fewshot_head_seed0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _take_mask(rows: RoleRows, mask: np.ndarray) -> RoleRows:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(rows),):
        raise ValueError("row mask shape mismatch")
    return RoleRows(
        *(getattr(rows, field)[mask] for field in rows.__dataclass_fields__)
    )


def _split_count(rows: RoleRows) -> dict[str, int]:
    return {
        "windows": int(len(rows)),
        "nonfog": int(np.sum(rows.label == 0)),
        "fog": int(np.sum(rows.label == 1)),
    }


def build_personalization_splits(rows: RoleRows) -> dict[str, RoleRows]:
    """Build the pre-registered, record-disjoint query personalization split."""

    subject = rows.subject_id == TEST_SUBJECT
    support = (
        subject
        & (rows.record_id == SUPPORT_RECORD)
        & (rows.start >= SUPPORT_START)
        & (rows.end <= SUPPORT_END)
    )
    calibration = (
        subject
        & (rows.record_id == CALIBRATION_RECORD)
        & (rows.start >= CALIBRATION_START)
        & (rows.end <= CALIBRATION_END)
    )
    query = subject & np.isin(rows.record_id, np.asarray(QUERY_RECORDS))
    splits = {
        "support": _take_mask(rows, support),
        "calibration": _take_mask(rows, calibration),
        "query": _take_mask(rows, query),
    }
    audit_personalization_splits(splits)
    return splits


def _raw_point_overlap(left: RoleRows, right: RoleRows) -> int:
    overlap = 0
    shared_records = set(left.record_id.tolist()) & set(right.record_id.tolist())
    for record_id in shared_records:
        left_intervals = [
            (int(start), int(end))
            for record, start, end in zip(left.record_id, left.start, left.end)
            if str(record) == str(record_id)
        ]
        right_intervals = [
            (int(start), int(end))
            for record, start, end in zip(right.record_id, right.start, right.end)
            if str(record) == str(record_id)
        ]
        for left_start, left_end in left_intervals:
            for right_start, right_end in right_intervals:
                overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
    return int(overlap)


def audit_personalization_splits(splits: dict[str, RoleRows]) -> dict[str, Any]:
    """Fail closed on counts, classes, duplicate windows, or raw-point leakage."""

    if set(splits) != set(EXPECTED_SPLIT_COUNTS):
        raise AssertionError(f"unexpected split names: {sorted(splits)}")
    all_window_ids: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for name, rows in splits.items():
        if set(rows.subject_id.tolist()) != {TEST_SUBJECT}:
            raise AssertionError(f"{name} is not exclusively {TEST_SUBJECT}")
        counts[name] = _split_count(rows)
        if counts[name] != EXPECTED_SPLIT_COUNTS[name]:
            raise AssertionError(
                f"{name} count mismatch: {counts[name]} != {EXPECTED_SPLIT_COUNTS[name]}"
            )
        if set(rows.label.astype(int).tolist()) != {0, 1}:
            raise AssertionError(f"{name} lacks a class")
        ids = rows.window_id.astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate windows inside {name}")
        all_window_ids.extend(ids)
    if len(all_window_ids) != len(set(all_window_ids)):
        raise AssertionError("a window is assigned to multiple personalization splits")
    pair_overlap = {
        "support_calibration": _raw_point_overlap(
            splits["support"], splits["calibration"]
        ),
        "support_query": _raw_point_overlap(splits["support"], splits["query"]),
        "calibration_query": _raw_point_overlap(
            splits["calibration"], splits["query"]
        ),
    }
    if any(pair_overlap.values()):
        raise AssertionError(f"cross-split raw-point leakage: {pair_overlap}")
    if set(splits["support"].record_id.tolist()) != {SUPPORT_RECORD}:
        raise AssertionError("support record contract changed")
    if set(splits["calibration"].record_id.tolist()) != {CALIBRATION_RECORD}:
        raise AssertionError("calibration record contract changed")
    if set(splits["query"].record_id.tolist()) != set(QUERY_RECORDS):
        raise AssertionError("query records contract changed")
    return {
        "counts": counts,
        "cross_split_raw_point_overlap": pair_overlap,
        "support_interval_samples": [SUPPORT_START, SUPPORT_END],
        "calibration_interval_samples": [CALIBRATION_START, CALIBRATION_END],
        "query_records": list(QUERY_RECORDS),
    }


def exact_threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    """Return an exhaustive score-derived threshold set, not a coarse grid."""

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not len(probabilities) or not np.all(np.isfinite(probabilities)):
        raise ValueError("threshold probabilities must be finite and non-empty")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("threshold probabilities must lie in [0,1]")
    unique = np.unique(probabilities)
    midpoints = (
        unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
        if len(unique) > 1
        else np.empty(0, dtype=np.float64)
    )
    # With the decision rule ``p >= threshold``, a threshold equal to the
    # maximum score still predicts that maximum-scored sample positive.  The
    # next representable float above max(score) is therefore required to
    # exhaustively include the all-negative decision partition (including the
    # boundary case max(score) == 1.0).
    above_maximum = (
        1.0
        if unique[-1] < 1.0
        else np.nextafter(unique[-1], np.inf)
    )
    return np.unique(
        np.concatenate(([0.0], unique, midpoints, [above_maximum]))
    )


def choose_exact_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, dict[str, Any], int]:
    """Maximize balanced accuracy; break ties by F1 then higher threshold."""

    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("threshold calibration requires both classes")
    candidates = exact_threshold_candidates(y_prob)
    best_key = (-math.inf, -math.inf, -math.inf)
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        key = (
            float(metrics["balanced_accuracy"] or 0.0),
            float(metrics["f1"] or 0.0),
            float(threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics, int(len(candidates))


def freeze_for_head_only(model: RepresentationTCNM) -> dict[str, Any]:
    """Freeze the TCN body and every BN statistic; expose only Linear head."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.blocks.eval()
    model.dropout.eval()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if trainable != ["classifier.weight", "classifier.bias"]:
        raise AssertionError(f"unexpected trainable parameters: {trainable}")
    if any(module.training for module in model.blocks.modules()):
        raise AssertionError("TCN blocks/BN are not in evaluation mode")
    return {
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "frozen_parameter_count": int(
            sum(p.numel() for p in model.parameters() if not p.requires_grad)
        ),
        "batchnorm_running_statistics_frozen": True,
        "dropout_disabled": True,
    }


def implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_loso_s01_gru_v1_c_tcn.py",
        REPO_ROOT / "scripts" / "run_daphnet_gru_mask_strength_nbm300_fold.py",
        REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
        REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
        REPO_ROOT / "cnbr_fog" / "evaluation.py",
    )
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths}


def _window_id_sha256(rows: RoleRows) -> str:
    digest = hashlib.sha256()
    for window_id in rows.window_id.astype(str).tolist():
        digest.update(window_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_source_artifacts(source_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Verify the completed zero-shot source and return only needed metadata."""

    source_dir = source_dir.resolve()
    data_dir = data_dir.resolve()
    required = {
        "done": source_dir / "DONE.json",
        "barrier": source_dir / "TRAINING_BARRIER.json",
        "metrics": source_dir / "metrics.json",
        "nbm_checkpoint": source_dir / "checkpoints" / checkpoint_name("MASK8_12"),
        "tcn_checkpoint": source_dir / "checkpoints" / "tcn.pt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    done = json.loads(required["done"].read_text(encoding="utf-8"))
    barrier = json.loads(required["barrier"].read_text(encoding="utf-8"))
    metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
    scientific = processed_nbm_scientific_manifest(data_dir)
    if done.get("status") != "complete" or barrier.get("status") != (
        "all_training_validation_and_thresholds_frozen"
    ):
        raise RuntimeError("zero-shot source is not complete/frozen")
    if done.get("metrics_sha256") != sha256_file(required["metrics"]):
        raise RuntimeError("source metrics hash mismatch")
    if barrier.get("scientific_data_sha256") != scientific["sha256"]:
        raise RuntimeError("source/data scientific hash mismatch")
    if barrier.get("nbm_checkpoint_sha256") != sha256_file(required["nbm_checkpoint"]):
        raise RuntimeError("source NBM checkpoint hash mismatch")
    if barrier.get("tcn_checkpoint_sha256") != sha256_file(required["tcn_checkpoint"]):
        raise RuntimeError("source TCN checkpoint hash mismatch")
    if metrics.get("test_subject") != TEST_SUBJECT:
        raise RuntimeError("source is not the S01 LOSO fold")
    threshold = float(barrier["threshold"])
    if not math.isclose(threshold, float(metrics["classifier"]["threshold"]), abs_tol=0.0):
        raise RuntimeError("source threshold metadata mismatch")
    scaler = metrics["scaler"]
    sigma = metrics["nbm"]["sigma_used_in_scheme_c"]
    if len(scaler["median"]) != 9 or len(scaler["iqr"]) != 9 or len(sigma) != 9:
        raise RuntimeError("source scaler/sigma channel contract mismatch")
    return {
        "source_dir": str(source_dir),
        "scientific_data_sha256": scientific["sha256"],
        "threshold": threshold,
        "scaler": scaler,
        "sigma": sigma,
        "paths": {name: str(path) for name, path in required.items()},
        "sha256": {name: sha256_file(path) for name, path in required.items()},
        "source_experiment_version": done.get("experiment_version"),
    }


def _load_frozen_models(
    source: dict[str, Any], device: torch.device
) -> tuple[GRUReconstructionNBM, RepresentationTCNM]:
    nbm = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    if sum(p.numel() for p in nbm.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("GRU-v1 architecture parameter count changed")
    nbm_payload = torch.load(
        source["paths"]["nbm_checkpoint"], map_location=device, weights_only=False
    )
    if nbm_payload.get("variant") != "MASK8_12":
        raise RuntimeError("source NBM is not MASK8_12")
    if nbm_payload.get("architecture") != architecture_config():
        raise RuntimeError("source NBM architecture mismatch")
    nbm.load_state_dict(nbm_payload["model_state"], strict=True)
    nbm.eval()
    for parameter in nbm.parameters():
        parameter.requires_grad_(False)

    tcn = RepresentationTCNM(27).to(device)
    tcn_payload = torch.load(
        source["paths"]["tcn_checkpoint"], map_location=device, weights_only=False
    )
    if tcn_payload.get("representation") != "r_abs_delta":
        raise RuntimeError("source TCN representation mismatch")
    if int(tcn_payload.get("input_channels", -1)) != 27:
        raise RuntimeError("source TCN channel contract mismatch")
    tcn.load_state_dict(tcn_payload["model_state"], strict=True)
    tcn.eval()
    return nbm, tcn


def _head_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).float(),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        ),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def _predict_loader(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch_x, batch_y in _head_loader(
        x, y, shuffle=False, seed=0, num_workers=0
    ):
        logits = model(batch_x.to(device, non_blocking=True))
        labels.append(batch_y.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels).astype(np.int8), np.concatenate(probabilities)


def _validation_bce(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in _head_loader(
            x, y, shuffle=False, seed=0, num_workers=0
        ):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            loss = criterion(model(batch_x), batch_y)
            total += float(loss) * len(batch_x)
            count += len(batch_x)
    return total / count


def fine_tune_head(
    source_model: RepresentationTCNM,
    support_x: np.ndarray,
    support_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
) -> tuple[RepresentationTCNM, dict[str, Any]]:
    """Update only classifier.{weight,bias}; select epoch by calibration PR-AUC."""

    set_seed(PERSONALIZATION_SEED)
    model = copy.deepcopy(source_model).to(device)
    freeze = freeze_for_head_only(model)
    n_pos = int(np.sum(support_y == 1))
    n_neg = int(np.sum(support_y == 0))
    if (n_neg, n_pos) != (127, 9):
        raise AssertionError("support class contract changed")
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    loader = _head_loader(
        support_x,
        support_y,
        shuffle=True,
        seed=PERSONALIZATION_SEED,
        num_workers=num_workers,
    )
    checkpoint = output_dir / "checkpoints" / "tcn_head_finetuned_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_pr = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, HEAD_MAX_EPOCHS + 1):
        # Calling model.train() would mutate frozen BN running statistics and
        # enable dropout.  Keep the whole network in eval mode; Linear has no
        # train/eval-dependent behavior and remains differentiable.
        model.eval()
        model.classifier.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.classifier.parameters(), HEAD_GRADIENT_CLIP
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite head gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        support_bce = total / count
        calibration_bce = _validation_bce(
            model, calibration_x, calibration_y, criterion, device
        )
        cal_true, cal_prob = _predict_loader(
            model, calibration_x, calibration_y, device
        )
        cal_pr = float(average_precision_score(cal_true, cal_prob))
        improved = cal_pr > best_pr + 1e-10
        history.append(
            {
                "epoch": epoch,
                "support_weighted_bce": support_bce,
                "calibration_weighted_bce": calibration_bce,
                "calibration_pr_auc": cal_pr,
                "improved": improved,
            }
        )
        if improved:
            best_pr = cal_pr
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "head_state": model.classifier.state_dict(),
                    "epoch": epoch,
                    "calibration_pr_auc": cal_pr,
                    "seed": PERSONALIZATION_SEED,
                    "trainable_parameter_names": freeze["trainable_parameter_names"],
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"head epoch={epoch:02d} support_bce={support_bce:.7f} "
            f"cal_bce={calibration_bce:.7f} cal_pr={cal_pr:.7f} "
            f"stale={stale}/{HEAD_PATIENCE}",
            flush=True,
        )
        if stale >= HEAD_PATIENCE:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("trainable_parameter_names") != freeze["trainable_parameter_names"]:
        raise RuntimeError("fine-tuned checkpoint parameter contract mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_for_head_only(model)
    write_csv(output_dir / "head_finetune_history.csv", history)
    return model, {
        "seed": PERSONALIZATION_SEED,
        "optimizer": "AdamW",
        "learning_rate": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "batch_size": HEAD_BATCH_SIZE,
        "maximum_epochs": HEAD_MAX_EPOCHS,
        "patience": HEAD_PATIENCE,
        "gradient_clip": HEAD_GRADIENT_CLIP,
        "loss": "weighted BCEWithLogitsLoss",
        "pos_weight": pos_weight,
        "epoch_selection": "highest S01 calibration PR-AUC; first epoch on ties",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_calibration_pr_auc": best_pr,
        "freeze": freeze,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": history,
    }


def validate_personalization_barrier(
    path: Path, source_meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Query gate: validate barrier and every artifact hash it binds."""

    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("personalization barrier is missing; query access denied")
    barrier = json.loads(path.read_text(encoding="utf-8"))
    if barrier.get("status") != "personalization_frozen_query_not_accessed":
        raise RuntimeError("personalization barrier status mismatch")
    if barrier.get("experiment_version") != EXPERIMENT_VERSION:
        raise RuntimeError("personalization barrier version mismatch")
    checkpoint = Path(barrier["head_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != barrier.get(
        "head_checkpoint_sha256"
    ):
        raise RuntimeError("personalization head checkpoint hash mismatch")
    if barrier.get("implementation_sha256") != implementation_hashes():
        raise RuntimeError("personalization implementation hash mismatch")
    if source_meta is not None:
        if barrier.get("scientific_data_sha256") != source_meta.get(
            "scientific_data_sha256"
        ):
            raise RuntimeError("barrier scientific data hash mismatch")
        if barrier.get("source_artifact_sha256") != source_meta.get("sha256"):
            raise RuntimeError("barrier source artifact hash mismatch")
    return barrier


def _plot_confusions(output_dir: Path, arm_metrics: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for axis, arm in zip(axes, ARMS):
        matrix = np.asarray(arm_metrics[arm]["confusion_matrix"], dtype=int)
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(
            xticks=(0, 1),
            yticks=(0, 1),
            xticklabels=("Non-FoG", "FoG"),
            yticklabels=("Non-FoG", "FoG"),
            xlabel="Predicted",
            ylabel="True",
            title=arm,
        )
    figure.savefig(output_dir / "query_confusion_matrices.png", dpi=180)
    figure.savefig(output_dir / "query_confusion_matrices.svg")
    plt.close(figure)


def _resume_if_complete(output_dir: Path) -> bool:
    done_path = output_dir / "DONE.json"
    metrics_path = output_dir / "metrics.json"
    if not done_path.is_file():
        return False
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or not metrics_path.is_file():
        raise RuntimeError("incomplete/stale personalization output exists")
    if done.get("metrics_sha256") != sha256_file(metrics_path):
        raise RuntimeError("personalization metrics hash mismatch")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("implementation_sha256") != implementation_hashes():
        raise RuntimeError("implementation changed; rerun with --overwrite")
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)
    return True


def run(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir:
        raise ValueError("personalization output must not overwrite zero-shot source")
    if output_dir.joinpath("DONE.json").is_file() and not args.overwrite:
        if _resume_if_complete(output_dir):
            return
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "seed": PERSONALIZATION_SEED,
                    "support": "S01_seg002 complete windows in [150,300) s",
                    "calibration": "S01_seg002 complete windows in [300,350) s",
                    "query": list(QUERY_RECORDS),
                    "expected_counts": EXPECTED_SPLIT_COUNTS,
                    "arms": list(ARMS),
                    "head_training": {
                        "lr": HEAD_LR,
                        "weight_decay": HEAD_WEIGHT_DECAY,
                        "batch_size": HEAD_BATCH_SIZE,
                        "max_epochs": HEAD_MAX_EPOCHS,
                        "patience": HEAD_PATIENCE,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    set_seed(PERSONALIZATION_SEED)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    source = validate_source_artifacts(source_dir, data_dir)
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 9:
        raise AssertionError("expected 64-Hz, 9-channel processed_NBM")
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir, SOURCE_OUTER_FOLD)
    splits = build_personalization_splits(rows)
    split_audit = audit_personalization_splits(splits)
    nbm, source_tcn = _load_frozen_models(source, device)
    scaler = RobustScaler(
        median=np.asarray(source["scaler"]["median"], dtype=np.float32),
        iqr=np.asarray(source["scaler"]["iqr"], dtype=np.float32),
        epsilon=float(source["scaler"].get("epsilon", 1e-6)),
    )
    sigma = np.asarray(source["sigma"], dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    support_x, support_feature = feature_values(
        nbm, scaler, sigma, records, splits["support"], device
    )
    calibration_x, calibration_feature = feature_values(
        nbm, scaler, sigma, records, splits["calibration"], device
    )
    support_y = splits["support"].label.astype(np.int8)
    calibration_y = splits["calibration"].label.astype(np.int8)

    calibration_true, source_calibration_prob = classifier_predict(
        source_tcn, calibration_x, calibration_y, device
    )
    if not np.array_equal(calibration_true, calibration_y):
        raise AssertionError("source calibration label order changed")
    threshold_only, threshold_only_cal_metrics, threshold_only_candidates = (
        choose_exact_threshold(calibration_y, source_calibration_prob)
    )
    fine_tuned_tcn, head_training = fine_tune_head(
        source_tcn,
        support_x,
        support_y,
        calibration_x,
        calibration_y,
        output_dir,
        device,
        args.num_workers,
    )
    fine_cal_true, fine_cal_prob = classifier_predict(
        fine_tuned_tcn, calibration_x, calibration_y, device
    )
    if not np.array_equal(fine_cal_true, calibration_y):
        raise AssertionError("fine-tuned calibration label order changed")
    fine_threshold, fine_cal_metrics, fine_candidates = choose_exact_threshold(
        calibration_y, fine_cal_prob
    )
    zero_cal_metrics = binary_metrics(
        calibration_y, source_calibration_prob, source["threshold"]
    )

    barrier = {
        "status": "personalization_frozen_query_not_accessed",
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": PERSONALIZATION_SEED,
        "scientific_data_sha256": source["scientific_data_sha256"],
        "source_artifact_sha256": source["sha256"],
        "support_window_ids_sha256": _window_id_sha256(splits["support"]),
        "calibration_window_ids_sha256": _window_id_sha256(splits["calibration"]),
        "query_window_ids_sha256": _window_id_sha256(splits["query"]),
        "head_checkpoint": head_training["checkpoint"],
        "head_checkpoint_sha256": head_training["checkpoint_sha256"],
        "thresholds": {
            ZERO_SHOT: source["threshold"],
            THRESHOLD_ONLY: threshold_only,
            HEAD_FINE_TUNE: fine_threshold,
        },
        "implementation_sha256": implementation_hashes(),
    }
    barrier_path = output_dir / PERSONALIZATION_BARRIER
    atomic_json_dump(barrier, barrier_path)
    validate_personalization_barrier(barrier_path, source)

    # QUERY ACCESS GATE: no query raw window or feature is materialized above.
    query_x, query_feature = feature_values(
        nbm, scaler, sigma, records, splits["query"], device
    )
    query_y = splits["query"].label.astype(np.int8)
    query_true, source_query_prob = classifier_predict(
        source_tcn, query_x, query_y, device
    )
    fine_query_true, fine_query_prob = classifier_predict(
        fine_tuned_tcn, query_x, query_y, device
    )
    if not np.array_equal(query_true, query_y) or not np.array_equal(
        fine_query_true, query_y
    ):
        raise AssertionError("query label order changed")
    thresholds = barrier["thresholds"]
    probabilities = {
        ZERO_SHOT: source_query_prob,
        THRESHOLD_ONLY: source_query_prob,
        HEAD_FINE_TUNE: fine_query_prob,
    }
    query_metrics = {
        arm: binary_metrics(query_y, probabilities[arm], float(thresholds[arm]))
        for arm in ARMS
    }

    prediction_rows: list[dict[str, Any]] = []
    query_manifest = manifest_rows(splits["query"], "query")
    for arm in ARMS:
        threshold = float(thresholds[arm])
        predictions = (probabilities[arm] >= threshold).astype(np.int8)
        for index, metadata in enumerate(query_manifest):
            prediction_rows.append(
                {
                    **metadata,
                    "arm": arm,
                    "fog_probability": float(probabilities[arm][index]),
                    "threshold": threshold,
                    "y_pred": int(predictions[index]),
                }
            )
    write_csv(output_dir / "query_predictions.csv", prediction_rows)
    np.savez_compressed(
        output_dir / "query_probabilities.npz",
        y_true=query_y,
        zero_shot_prob=source_query_prob,
        threshold_only_prob=source_query_prob,
        head_fine_tune_prob=fine_query_prob,
        zero_shot_threshold=np.asarray(float(thresholds[ZERO_SHOT])),
        threshold_only_threshold=np.asarray(float(thresholds[THRESHOLD_ONLY])),
        head_fine_tune_threshold=np.asarray(float(thresholds[HEAD_FINE_TUNE])),
    )
    all_manifest: list[dict[str, Any]] = []
    for name, split in splits.items():
        all_manifest.extend(manifest_rows(split, name))
    write_csv(output_dir / "split_manifest.csv", all_manifest)
    _plot_confusions(output_dir, query_metrics)

    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "seed": PERSONALIZATION_SEED,
        "dataset": str(data_dir),
        "source": {
            key: value for key, value in source.items() if key not in {"scaler", "sigma"}
        },
        "protocol": {
            "description": (
                "S01 few-shot personalization with fixed record/time split; "
                "query is accessed once only after model/threshold barrier"
            ),
            "split_audit": split_audit,
            "support_seconds": [150, 300],
            "calibration_seconds": [300, 350],
            "query_records": list(QUERY_RECORDS),
            "scheme_c": "F=[r,abs(r),delta(r)] [B,27,128]",
            "nbm_scaler_sigma_frozen": True,
        },
        "features": {
            "support": support_feature,
            "calibration": calibration_feature,
            "query": query_feature,
        },
        "arms": {
            ZERO_SHOT: {
                "network_update": "none",
                "threshold_source": "original S02 development subject",
                "threshold": float(thresholds[ZERO_SHOT]),
                "calibration_metrics_for_audit_only": zero_cal_metrics,
                "query_metrics": query_metrics[ZERO_SHOT],
            },
            THRESHOLD_ONLY: {
                "network_update": "none",
                "threshold_source": "S01_seg002 [300,350) s calibration",
                "threshold_rule": "exact candidates; max BA, ties F1 then higher",
                "candidate_count": threshold_only_candidates,
                "threshold": float(thresholds[THRESHOLD_ONLY]),
                "calibration_metrics": threshold_only_cal_metrics,
                "query_metrics": query_metrics[THRESHOLD_ONLY],
            },
            HEAD_FINE_TUNE: {
                "network_update": "TCN classifier Linear(128,1) only",
                "training": {
                    key: value for key, value in head_training.items() if key != "history"
                },
                "threshold_source": "S01_seg002 [300,350) s calibration",
                "threshold_rule": "exact candidates; max BA, ties F1 then higher",
                "candidate_count": fine_candidates,
                "threshold": float(thresholds[HEAD_FINE_TUNE]),
                "calibration_metrics": fine_cal_metrics,
                "query_metrics": query_metrics[HEAD_FINE_TUNE],
            },
        },
        "personalization_barrier": barrier,
        "implementation_sha256": implementation_hashes(),
    }
    atomic_json_dump(result, output_dir / "metrics.json")
    atomic_json_dump(
        {
            "status": "complete",
            "experiment_version": EXPERIMENT_VERSION,
            "query_windows": int(len(query_y)),
            "arms": {
                arm: {
                    "threshold": float(query_metrics[arm]["threshold"]),
                    "sensitivity": query_metrics[arm]["sensitivity"],
                    "precision": query_metrics[arm]["precision"],
                    "specificity": query_metrics[arm]["specificity"],
                    "pr_auc": query_metrics[arm]["auprc"],
                }
                for arm in ARMS
            },
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        },
        output_dir / "DONE.json",
    )
    print(json.dumps(query_metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
