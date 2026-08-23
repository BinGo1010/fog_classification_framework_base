#!/usr/bin/env python
"""Strict r-only ablation of GRU-BASE-NBM + TCN on processed_NBM_Exp.

The frozen role-4 Scaler, GRU-NBM, and role-5 residual calibration are reused
from the completed expanded scheme-C experiment.  Only the TCN is retrained,
using centered standardized residual r [B,30,128] instead of
[r,abs(r),delta(r)] [B,90,128].  Permanent roles 0/1 remain locked until all
120 TCN checkpoints and validation-selected thresholds are globally sealed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
)
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as expanded
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import RepresentationTCNM
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    RobustScaler,
    set_seed,
)


SUBJECTS = expanded.SUBJECTS
FOLDS = expanded.FOLDS
SEEDS = expanded.SEEDS
ROLES = expanded.ROLES
RAW_CHANNELS = 30
REFERENCE_TCN_CHANNELS = 90
TCN_INPUT_CHANNELS = 30
TCN_PARAMETER_COUNT = 135_969
NBM_VARIANT = expanded.NBM_VARIANT
REPRESENTATION = "r_only"
METRIC_KEYS = expanded.METRIC_KEYS
EXPERIMENT_SCHEMA = "all_dataset_within_subject_gru_nbm_r_only_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_gru_nbm_r_only_tcn_barrier.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seed_list(value: str) -> tuple[int, ...]:
    return expanded.parse_seed_list(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "seal", "evaluate", "aggregate"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_dir(root: Path, subject: str, fold: int, seed: int) -> Path:
    return root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"


def require_job_args(args: argparse.Namespace) -> tuple[str, int, int]:
    if args.subject is None or args.fold is None or args.seed is None:
        raise ValueError(f"stage={args.stage} requires --subject, --fold, and --seed")
    return str(args.subject), int(args.fold), int(args.seed)


def load_plan(root: Path) -> dict[str, Any]:
    path = root / "EXPERIMENT_PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"launcher plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != EXPERIMENT_SCHEMA:
        raise AssertionError(f"unexpected plan schema: {plan.get('schema')}")
    return plan


def validate_plan_args(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "source_root": str(args.source_root.resolve()),
        "batch_size": int(args.batch_size),
        "tcn_max_epochs": int(args.tcn_max_epochs),
        "tcn_patience": int(args.tcn_patience),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}")


def source_artifact_paths(source_root: Path, subject: str, fold: int, seed: int) -> dict[str, Path]:
    root = expanded.run_dir(source_root, subject, fold, seed)
    return {
        "root": root,
        "done": root / "DONE_TRAIN.json",
        "frozen": root / "FROZEN_TRAIN.json",
        "nbm_checkpoint": root / "checkpoints" / "gru_nbm_best.pt",
        "scaler": root / "scaler_role4.json",
        "calibration": root / "calibration_role5.json",
    }


def load_frozen_source(
    source_root: Path,
    subject: str,
    fold: int,
    seed: int,
    data_scientific_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    paths = source_artifact_paths(source_root, subject, fold, seed)
    if not all(path.is_file() for name, path in paths.items() if name != "root"):
        raise FileNotFoundError(f"frozen expanded-experiment NBM source missing: {paths['root']}")
    done = json.loads(paths["done"].read_text(encoding="utf-8"))
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8"))
    if frozen.get("schema") != expanded.EXPERIMENT_SCHEMA:
        raise AssertionError("source is not the required expanded GRU-NBM experiment")
    if (frozen.get("subject"), frozen.get("fold"), frozen.get("seed")) != (subject, fold, seed):
        raise AssertionError("source fold/subject/seed identity mismatch")
    if frozen.get("nbm_variant") != NBM_VARIANT:
        raise AssertionError(f"source NBM variant mismatch: {frozen.get('nbm_variant')}")
    if frozen.get("data_scientific_sha256") != data_scientific_sha256:
        raise AssertionError("source NBM was trained on a different scientific dataset identity")
    if done.get("frozen_sha256") != sha256_file(paths["frozen"]):
        raise AssertionError("source FROZEN artifact changed after DONE_TRAIN")
    if done.get("frozen_id") != frozen.get("frozen_id"):
        raise AssertionError("source DONE/FROZEN identity mismatch")
    expected_hashes = {
        "nbm_checkpoint_sha256": paths["nbm_checkpoint"],
        "scaler_sha256": paths["scaler"],
        "calibration_sha256": paths["calibration"],
    }
    for name, path in expected_hashes.items():
        if frozen.get(name) != sha256_file(path):
            raise AssertionError(f"source artifact hash mismatch: {name}")
    scaler_payload = json.loads(paths["scaler"].read_text(encoding="utf-8"))
    calibration_payload = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    scaler = RobustScaler(
        np.asarray(scaler_payload["scaler"]["median"], dtype=np.float32),
        np.asarray(scaler_payload["scaler"]["iqr"], dtype=np.float32),
        float(scaler_payload["scaler"]["epsilon"]),
    )
    sigma = np.asarray(calibration_payload["sigma"], dtype=np.float32)
    if sigma.shape != (RAW_CHANNELS,):
        raise AssertionError(f"source sigma shape mismatch: {sigma.shape}")
    nbm = GRUReconstructionNBM(
        channels=RAW_CHANNELS,
        hidden=expanded.HIDDEN,
        bottleneck=expanded.BOTTLENECK,
    ).to(device)
    checkpoint = torch.load(paths["nbm_checkpoint"], map_location=device, weights_only=False)
    if checkpoint.get("seed") != seed or checkpoint.get("architecture") != expanded.architecture_config():
        raise AssertionError("source GRU-NBM checkpoint semantic identity mismatch")
    nbm.load_state_dict(checkpoint["model_state"])
    source_bundle = {
        "source_frozen_id": frozen["frozen_id"],
        "source_frozen_sha256": sha256_file(paths["frozen"]),
        "source_done_sha256": sha256_file(paths["done"]),
        **{name: sha256_file(path) for name, path in expected_hashes.items()},
        "source_reference_tcn_initial_state_sha256": frozen["tcn_training"][
            "initial_model_state_sha256"
        ],
    }
    return {
        "model": nbm,
        "scaler": scaler,
        "sigma": sigma,
        "bundle": source_bundle,
        "paths": paths,
    }


def r_only_features(
    model: nn.Module,
    scaler: RobustScaler,
    sigma: np.ndarray,
    raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x = expanded.centered_scaled_ntc(scaler, raw)
    x_hat = expanded.reconstruct(model, x, device, batch_size)
    q = np.clip((x - x_hat) / (sigma[None, None, :] + 1e-6), -12.0, 12.0)
    r = (q - q.mean(axis=1, keepdims=True)).astype(np.float32, copy=False)
    if r.shape[1:] != (128, RAW_CHANNELS):
        raise AssertionError(f"unexpected r-only shape: {r.shape}")
    return np.ascontiguousarray(r.transpose(0, 2, 1), dtype=np.float32)


def paired_r_only_tcn(
    seed: int,
    expected_reference_hash: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, str]]:
    """Regenerate expanded TCN initialization, then select the shared r subset."""

    set_seed(seed)
    reference = RepresentationTCNM(REFERENCE_TCN_CHANNELS)
    reference_state = {
        name: tensor.detach().cpu().clone() for name, tensor in reference.state_dict().items()
    }
    reference_hash = expanded.state_dict_sha256(reference_state)
    if reference_hash != expected_reference_hash:
        raise AssertionError(
            "cannot reproduce source 90-channel TCN initialization; code/runtime identity drifted"
        )
    rng_after_reference = capture_rng_state()
    target = RepresentationTCNM(TCN_INPUT_CHANNELS)
    target_state = {
        name: tensor.detach().cpu().clone() for name, tensor in target.state_dict().items()
    }
    for name, target_tensor in target_state.items():
        source_tensor = reference_state[name]
        if target_tensor.shape == source_tensor.shape:
            target_tensor.copy_(source_tensor)
        elif (
            target_tensor.ndim == 3
            and source_tensor.ndim == 3
            and target_tensor.shape[0] == source_tensor.shape[0]
            and target_tensor.shape[2] == source_tensor.shape[2]
            and target_tensor.shape[1] == TCN_INPUT_CHANNELS
            and source_tensor.shape[1] == REFERENCE_TCN_CHANNELS
        ):
            target_tensor.copy_(source_tensor[:, :TCN_INPUT_CHANNELS, :])
        else:
            raise AssertionError(
                f"unhandled paired TCN parameter {name}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
    target.load_state_dict(target_state)
    target = target.to(device)
    restore_rng_state(rng_after_reference)
    parameter_count = sum(parameter.numel() for parameter in target.parameters())
    if parameter_count != TCN_PARAMETER_COUNT:
        raise RuntimeError(f"r-only TCN parameter contract changed: {parameter_count}")
    return target, {
        "reference_90ch_initial_state_sha256": reference_hash,
        "r_only_30ch_initial_state_sha256": expanded.state_dict_sha256(target_state),
    }


def train_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    destination: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
    maximum_epochs: int,
    patience: int,
    expected_reference_hash: str,
) -> tuple[nn.Module, dict[str, Any]]:
    model, initialization = paired_r_only_tcn(seed, expected_reference_hash, device)
    n_nonfog = int(np.sum(train_y == 0))
    n_fog = int(np.sum(train_y == 1))
    if min(n_nonfog, n_fog) == 0:
        raise ValueError("roles6/7 must contain both classes")
    pos_weight = n_nonfog / n_fog
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = expanded.tcn_loader(train_x, train_y, batch_size, True, seed, workers)
    checkpoint = destination / "checkpoints" / "tcn_r_only.pt"
    best_pr_auc = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite r-only TCN gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_bce = total / count
        validation_bce = expanded.validation_tcn_loss(
            model, validation_x, validation_y, criterion, device, batch_size
        )
        val_true, val_prob = expanded.predict(
            model, validation_x, validation_y, device, batch_size
        )
        validation_pr_auc = float(average_precision_score(val_true, val_prob))
        improved = validation_pr_auc > best_pr_auc + 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_bce,
                "validation_weighted_bce": validation_bce,
                "validation_pr_auc": validation_pr_auc,
                "improved": improved,
            }
        )
        if improved:
            best_pr_auc = validation_pr_auc
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "schema": EXPERIMENT_SCHEMA,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_pr_auc": validation_pr_auc,
                    "input_channels": TCN_INPUT_CHANNELS,
                    "representation": REPRESENTATION,
                    **initialization,
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"TCN-r epoch={epoch:03d} train={train_bce:.7f} val={validation_bce:.7f} "
            f"val_pr_auc={validation_pr_auc:.7f} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return model, {
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "n_nonfog_role6": n_nonfog,
        "n_fog_role7": n_fog,
        "pos_weight": pos_weight,
        "parameter_count": TCN_PARAMETER_COUNT,
        **initialization,
        "history": history,
    }


def training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ablation": "remove abs(r) and delta(r); retain centered standardized r only",
        "frozen_source": "same per-job role4 Scaler, GRU-BASE Mask4-8 NBM, and role5 sigma as expanded experiment",
        "source_trainable_parameters_updated": False,
        "residual": "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); r=q-mean_t(q)",
        "input_shape": ["B", 30, 128],
        "tcn": "RepresentationTCNM 30->32->64->64->128; dilations1/2/4/8; GAP; one logit",
        "paired_initialization": "shared layers and first 30 input channels copied from exact regenerated 90-channel source initialization",
        "train_roles": [6, 7],
        "validation_roles": [2, 3],
        "test_roles": [0, 1],
        "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "batch_size": args.batch_size,
        "maximum_epochs": args.tcn_max_epochs,
        "patience": args.tcn_patience,
        "checkpoint": "maximum roles2/3 PR-AUC",
        "threshold": "roles2/3 grid0.05..0.95 step0.01; max balanced accuracy; ties F1 then higher threshold",
    }


def validate_completed_train(destination: Path, plan: dict[str, Any]) -> bool:
    done_path = destination / "DONE_TRAIN.json"
    if not done_path.is_file():
        return False
    frozen_path = destination / "FROZEN_TRAIN.json"
    checkpoint_path = destination / "checkpoints" / "tcn_r_only.pt"
    history_path = destination / "tcn_history.csv"
    if not all(path.is_file() for path in (frozen_path, checkpoint_path, history_path)):
        raise FileNotFoundError(f"incomplete completed train job: {destination}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    valid = (
        done.get("frozen_sha256") == sha256_file(frozen_path)
        and done.get("frozen_id") == frozen.get("frozen_id")
        and frozen.get("tcn_checkpoint_sha256") == sha256_file(checkpoint_path)
        and frozen.get("tcn_history_sha256") == sha256_file(history_path)
        and frozen.get("data_scientific_sha256") == plan["data_scientific_sha256"]
        and frozen.get("code_sha256") == plan["code_sha256"]
    )
    if not valid:
        raise AssertionError(f"completed train job failed validation: {destination}")
    return True


def run_train(args: argparse.Namespace) -> None:
    subject, fold, seed = require_job_args(args)
    root = args.output_root.resolve()
    destination = run_dir(root, subject, fold, seed)
    plan = load_plan(root)
    validate_plan_args(args, plan)
    if not args.overwrite and validate_completed_train(destination, plan):
        print(f"SKIP validated completed train job: {destination}", flush=True)
        return
    device = raw_base.resolve_device(args.device)
    source = load_frozen_source(
        args.source_root.resolve(), subject, fold, seed,
        plan["data_scientific_sha256"], device,
    )
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != RAW_CHANNELS:
        raise AssertionError(f"expected 64Hz/30 channels, got {dataset.sampling_rate_hz}/{dataset.n_channels}")
    rows = raw_base.load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    train_x = r_only_features(
        source["model"], source["scaler"], source["sigma"],
        raw_base.raw_windows(dataset, role67), device, args.batch_size,
    )
    validation_x = r_only_features(
        source["model"], source["scaler"], source["sigma"],
        raw_base.raw_windows(dataset, role23), device, args.batch_size,
    )
    model, training = train_tcn(
        train_x, role67.label, validation_x, role23.label, destination, device,
        seed, args.batch_size, args.num_workers, args.tcn_max_epochs,
        args.tcn_patience, source["bundle"]["source_reference_tcn_initial_state_sha256"],
    )
    val_true, val_prob = expanded.predict(
        model, validation_x, role23.label, device, args.batch_size
    )
    threshold, validation_metrics = raw_base.choose_threshold(val_true, val_prob)
    history_path = destination / "tcn_history.csv"
    expanded.write_csv(history_path, training["history"])
    checkpoint_path = destination / "checkpoints" / "tcn_r_only.pt"
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_before_permanent_test",
        "created_utc": utc_now(),
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "representation": REPRESENTATION,
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "code_sha256": plan["code_sha256"],
        "source_bundle": source["bundle"],
        "tcn_checkpoint_sha256": sha256_file(checkpoint_path),
        "tcn_history_sha256": sha256_file(history_path),
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "validation_metrics": validation_metrics,
        "tcn_training": {key: value for key, value in training.items() if key != "history"},
        "training_contract": training_contract(args),
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "permanent_test_materialized": False,
    }
    frozen["frozen_id"] = canonical_fingerprint(frozen)
    frozen_path = destination / "FROZEN_TRAIN.json"
    atomic_json_dump(frozen, frozen_path)
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "status": "train_complete",
            "subject": subject,
            "fold": fold,
            "seed": seed,
            "frozen_id": frozen["frozen_id"],
            "frozen_sha256": sha256_file(frozen_path),
        },
        destination / "DONE_TRAIN.json",
    )
    print(
        f"TRAIN COMPLETE subject={subject} fold={fold} seed={seed} "
        f"best_epoch={training['best_epoch']} threshold={threshold:.2f} "
        f"val_pr_auc={validation_metrics['auprc']:.6f}",
        flush=True,
    )


def load_and_validate_barrier(
    root: Path, subject: str, fold: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "TRAINING_BARRIER.json"
    if not path.is_file():
        raise FileNotFoundError("permanent test is locked until TRAINING_BARRIER.json exists")
    barrier = json.loads(path.read_text(encoding="utf-8"))
    if barrier.get("schema") != BARRIER_SCHEMA or barrier.get("status") != "sealed":
        raise AssertionError("invalid or unsealed training barrier")
    key = f"{subject}/fold_{fold}/seed_{seed}"
    sealed = barrier.get("jobs", {}).get(key)
    if sealed is None:
        raise KeyError(f"job absent from barrier: {key}")
    destination = run_dir(root, subject, fold, seed)
    frozen_path = destination / "FROZEN_TRAIN.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if sha256_file(frozen_path) != sealed["frozen_sha256"] or frozen["frozen_id"] != sealed["frozen_id"]:
        raise AssertionError(f"frozen artifact changed after seal: {key}")
    if sha256_file(destination / "checkpoints" / "tcn_r_only.pt") != sealed["tcn_checkpoint_sha256"]:
        raise AssertionError(f"TCN checkpoint changed after seal: {key}")
    return barrier, frozen


def run_evaluate(args: argparse.Namespace) -> None:
    subject, fold, seed = require_job_args(args)
    root = args.output_root.resolve()
    destination = run_dir(root, subject, fold, seed)
    barrier, frozen = load_and_validate_barrier(root, subject, fold, seed)
    current_data = processed_nbm_scientific_manifest(args.data_dir.resolve())["sha256"]
    if current_data != barrier["data_scientific_sha256"]:
        raise AssertionError("dataset changed after training barrier")
    done_path = destination / "DONE_TEST.json"
    if done_path.is_file() and not args.overwrite:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        artifacts = {
            "metrics_sha256": destination / "metrics.json",
            "predictions_sha256": destination / "test_predictions.csv",
            "probabilities_sha256": destination / "test_probabilities.npz",
        }
        if (
            done.get("barrier_id") == barrier["barrier_id"]
            and all(path.is_file() and done.get(name) == sha256_file(path) for name, path in artifacts.items())
        ):
            print(f"SKIP validated completed evaluate job: {destination}", flush=True)
            return
        raise AssertionError("existing test result failed barrier/artifact validation")
    device = raw_base.resolve_device(args.device)
    source = load_frozen_source(
        args.source_root.resolve(), subject, fold, seed,
        barrier["data_scientific_sha256"], device,
    )
    if source["bundle"] != frozen["source_bundle"]:
        raise AssertionError("frozen NBM source changed after ablation barrier")
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    rows = raw_base.load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    test_rows = rows.take_role(0, 1)
    test_x = r_only_features(
        source["model"], source["scaler"], source["sigma"],
        raw_base.raw_windows(dataset, test_rows), device, args.batch_size,
    )
    model = RepresentationTCNM(TCN_INPUT_CHANNELS).to(device)
    checkpoint = torch.load(
        destination / "checkpoints" / "tcn_r_only.pt",
        map_location=device,
        weights_only=False,
    )
    if (
        checkpoint.get("seed") != seed
        or checkpoint.get("input_channels") != TCN_INPUT_CHANNELS
        or checkpoint.get("representation") != REPRESENTATION
    ):
        raise AssertionError("r-only TCN checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model_state"])
    y_true, probability = expanded.predict(model, test_x, test_rows.label, device, args.batch_size)
    threshold = float(frozen["threshold"])
    y_pred = (probability >= threshold).astype(np.int8)
    metrics = binary_metrics(y_true, probability, threshold)
    metrics.update(raw_base.event_metrics(dataset, test_rows, y_pred))
    metrics["pr_auc"] = metrics["auprc"]
    metrics["false_alarms_per_hour"] = metrics["false_alarm_events_per_hour"]
    metrics.update(
        {
            "schema": EXPERIMENT_SCHEMA,
            "subject": subject,
            "fold": fold,
            "seed": seed,
            "representation": REPRESENTATION,
            "barrier_id": barrier["barrier_id"],
            "frozen_id": frozen["frozen_id"],
        }
    )
    metrics_path = destination / "metrics.json"
    predictions_path = destination / "test_predictions.csv"
    probabilities_path = destination / "test_probabilities.npz"
    atomic_json_dump(metrics, metrics_path)
    expanded.write_csv(
        predictions_path,
        (
            {
                "subject_id": subject,
                "fold": fold,
                "seed": seed,
                "record_id": str(test_rows.record_id[index]),
                "start_index": int(test_rows.start[index]),
                "end_index_exclusive": int(test_rows.end[index]),
                "role_code": int(test_rows.role[index]),
                "window_id": str(test_rows.window_id[index]),
                "y_true": int(y_true[index]),
                "probability": float(probability[index]),
                "threshold": threshold,
                "y_pred": int(y_pred[index]),
            }
            for index in range(len(test_rows))
        ),
    )
    atomic_npz_save(
        probabilities_path,
        y_true=y_true,
        probability=probability.astype(np.float32),
        y_pred=y_pred,
        threshold=np.asarray(threshold),
        window_id=test_rows.window_id,
    )
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "status": "test_complete",
            "subject": subject,
            "fold": fold,
            "seed": seed,
            "barrier_id": barrier["barrier_id"],
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
            "probabilities_sha256": sha256_file(probabilities_path),
        },
        done_path,
    )
    print(
        f"TEST COMPLETE subject={subject} fold={fold} seed={seed} "
        f"sens={metrics['sensitivity']:.6f} precision={metrics['precision']:.6f} "
        f"spec={metrics['specificity']:.6f} pr_auc={metrics['pr_auc']:.6f} "
        f"event_sens={metrics['event_sensitivity']} fa_h={metrics['false_alarms_per_hour']:.6f}",
        flush=True,
    )


def run_seal(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    plan = load_plan(root)
    validate_plan_args(args, plan)
    current_data = processed_nbm_scientific_manifest(args.data_dir.resolve())["sha256"]
    if current_data != plan["data_scientific_sha256"]:
        raise AssertionError("dataset changed after plan creation")
    seeds = parse_seed_list(args.seeds)
    jobs: dict[str, Any] = {}
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                destination = run_dir(root, subject, fold, seed)
                if not validate_completed_train(destination, plan):
                    raise FileNotFoundError(f"training job incomplete: {destination}")
                frozen_path = destination / "FROZEN_TRAIN.json"
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                if (frozen["subject"], frozen["fold"], frozen["seed"]) != (subject, fold, seed):
                    raise AssertionError(f"frozen identity mismatch: {destination}")
                source = load_frozen_source(
                    args.source_root.resolve(), subject, fold, seed,
                    plan["data_scientific_sha256"], torch.device("cpu"),
                )
                if source["bundle"] != frozen["source_bundle"]:
                    raise AssertionError(f"source bundle mismatch: {destination}")
                key = f"{subject}/fold_{fold}/seed_{seed}"
                jobs[key] = {
                    "frozen_id": frozen["frozen_id"],
                    "frozen_sha256": sha256_file(frozen_path),
                    "tcn_checkpoint_sha256": frozen["tcn_checkpoint_sha256"],
                    "threshold": frozen["threshold"],
                    "source_bundle": frozen["source_bundle"],
                }
    core = {
        "schema": BARRIER_SCHEMA,
        "status": "sealed",
        "created_utc": utc_now(),
        "plan_id": plan["plan_id"],
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "source_root": str(args.source_root.resolve()),
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(seeds),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    core["barrier_id"] = canonical_fingerprint(
        {key: value for key, value in core.items() if key != "created_utc"}
    )
    atomic_json_dump(core, root / "TRAINING_BARRIER.json")
    print(f"SEALED {len(jobs)} jobs barrier_id={core['barrier_id']}", flush=True)


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    return expanded.mean_std(values)


def run_aggregate(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    seeds = parse_seed_list(args.seeds)
    barrier = json.loads((root / "TRAINING_BARRIER.json").read_text(encoding="utf-8"))
    if barrier.get("schema") != BARRIER_SCHEMA or barrier.get("status") != "sealed":
        raise AssertionError("strict training barrier missing")
    run_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                destination = run_dir(root, subject, fold, seed)
                paths = {
                    "done": destination / "DONE_TEST.json",
                    "metrics_sha256": destination / "metrics.json",
                    "predictions_sha256": destination / "test_predictions.csv",
                    "probabilities_sha256": destination / "test_probabilities.npz",
                }
                if not all(path.is_file() for path in paths.values()):
                    raise FileNotFoundError(f"test job incomplete: {destination}")
                done = json.loads(paths["done"].read_text(encoding="utf-8"))
                metrics = json.loads(paths["metrics_sha256"].read_text(encoding="utf-8"))
                if done.get("barrier_id") != barrier["barrier_id"] or metrics.get("barrier_id") != barrier["barrier_id"]:
                    raise AssertionError(f"barrier mismatch: {destination}")
                for name in ("metrics_sha256", "predictions_sha256", "probabilities_sha256"):
                    if done.get(name) != sha256_file(paths[name]):
                        raise AssertionError(f"{name} mismatch: {destination}")
                run_rows.append(
                    {
                        "subject": subject,
                        "fold": fold,
                        "seed": seed,
                        "threshold": metrics["threshold"],
                        **{key: metrics[key] for key in METRIC_KEYS},
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tp": metrics["tp"],
                        "evaluable_true_events": metrics["evaluable_true_events"],
                        "detected_true_events": metrics["detected_true_events"],
                        "false_alarm_events": metrics["false_alarm_events"],
                        "evaluated_nonfog_hours": metrics["evaluated_nonfog_hours"],
                    }
                )
    subject_seed_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in seeds:
            selected = [row for row in run_rows if row["subject"] == subject and row["seed"] == seed]
            subject_seed_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    **{key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS},
                }
            )
    subject_summary_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        selected = [row for row in subject_seed_rows if row["subject"] == subject]
        row: dict[str, Any] = {"subject": subject}
        for key in METRIC_KEYS:
            summary = mean_std(item[key] for item in selected)
            row[f"{key}_mean"] = summary["mean"]
            row[f"{key}_std"] = summary["std"]
        subject_summary_rows.append(row)
    overall_seed_rows = []
    for seed in seeds:
        selected = [row for row in subject_seed_rows if row["seed"] == seed]
        overall_seed_rows.append(
            {
                "seed": seed,
                **{key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS},
            }
        )
    overall = {key: mean_std(row[key] for row in overall_seed_rows) for key in METRIC_KEYS}
    expanded.write_csv(root / "run_metrics.csv", run_rows)
    expanded.write_csv(root / "subject_seed_metrics.csv", subject_seed_rows)
    expanded.write_csv(root / "subject_summary.csv", subject_summary_rows)
    expanded.write_csv(root / "overall_seed_metrics.csv", overall_seed_rows)
    summary = {
        "schema": EXPERIMENT_SCHEMA,
        "model": "frozen GRU-BASE Mask4-8 NBM + r-only 30-channel TCN",
        "ablation": "remove abs(r) and delta(r) from expanded scheme C",
        "aggregation": "subject/seed macro mean of 3 folds; subject mean+population SD over 5 seeds; overall subject-macro per seed then mean+population SD",
        "event_metric": {
            "version": raw_base.EVENT_METRIC_VERSION,
            "minimum_positive_windows": 2,
            "merge_gap_seconds": 0.5,
            "false_alarm_denominator": "union coverage of evaluated valid Non-FoG samples",
        },
        "subjects": subject_summary_rows,
        "overall": overall,
    }
    atomic_json_dump(summary, root / "summary.json")
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "status": "complete",
            "completed_utc": utc_now(),
            "run_count": len(run_rows),
            "barrier_id": barrier["barrier_id"],
            "summary_sha256": sha256_file(root / "summary.json"),
        },
        root / "DONE.json",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.stage == "train":
        run_train(args)
    elif args.stage == "seal":
        run_seal(args)
    elif args.stage == "evaluate":
        run_evaluate(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
