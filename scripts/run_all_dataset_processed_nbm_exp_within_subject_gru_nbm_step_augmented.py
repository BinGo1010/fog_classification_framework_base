#!/usr/bin/env python
"""Step-trained augmented GRU-NBM + scheme-C TCN on processed_NBM_Exp.

Only the role-4 NBM input corruption differs from the clean-step comparison:
each window draw is mutually exclusively clean (0.40), Gaussian-corrupted
(0.40), or continuously time-masked (0.20). The reconstruction target remains
the clean window. All jobs are trained and sealed before permanent-test access.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as base


EXPERIMENT_SCHEMA = "all_dataset_within_subject_gru_nbm_step_augmented_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_gru_nbm_step_augmented_tcn_barrier.v1"
NBM_VARIANT = "GRU_STEP_AUGMENTED_40CLEAN_40GAUSSIAN_20MASK4_8"
MODEL_DESCRIPTION = "step-trained augmented GRU-NBM + scheme-C 90-channel TCN"
SUBJECTS = base.SUBJECTS
FOLDS = base.FOLDS
SEEDS = base.SEEDS
ROLES = base.ROLES

# Reuse the audited data, architecture, residual, TCN, barrier, and aggregation
# implementations while giving this experiment an independent artifact schema.
base.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
base.BARRIER_SCHEMA = BARRIER_SCHEMA
base.NBM_VARIANT = NBM_VARIANT
base.MODEL_DESCRIPTION = MODEL_DESCRIPTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "seal", "evaluate", "aggregate"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-batch-size", type=int, default=16)
    parser.add_argument("--maximum-updates", type=int, default=5000)
    parser.add_argument("--validation-frequency", type=int, default=50)
    parser.add_argument("--validation-patience", type=int, default=20)
    parser.add_argument("--nbm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--nbm-weight-decay", type=float, default=1e-4)
    parser.add_argument("--tcn-batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def clean_validation_loss(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    criterion = nn.SmoothL1Loss(beta=1.0)
    total = 0.0
    count = 0
    for (batch,) in base.nbm_loader(values, batch_size, False, 0, 0):
        batch = batch.to(device, non_blocking=True)
        loss = criterion(model(batch), batch)
        total += float(loss) * len(batch)
        count += len(batch)
    if count == 0:
        raise ValueError("empty role-5 NBM validation set")
    return total / count


def train_nbm_by_updates(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    destination: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
    maximum_updates: int,
    validation_frequency: int,
    validation_patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, dict[str, Any]]:
    base.set_seed(seed)
    model = base.GRUReconstructionNBM(
        channels=base.RAW_CHANNELS,
        hidden=base.HIDDEN,
        bottleneck=base.BOTTLENECK,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != base.NBM_PARAMETER_COUNT:
        raise AssertionError("GRU-NBM parameter contract changed")
    if any(isinstance(module, nn.Dropout) for module in model.modules()):
        raise AssertionError("fixed GRU-NBM unexpectedly contains dropout")
    initial_state = base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    loader = base.nbm_loader(train_x, batch_size, True, seed, workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = destination / "checkpoints" / base.NBM_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_loss = clean_validation_loss(model, validation_x, device, batch_size)
    best_step = 0
    stale_validations = 0
    history: list[dict[str, Any]] = []
    total_mode_counts = np.zeros(3, dtype=np.int64)
    base.atomic_torch_save(
        {
            "schema": EXPERIMENT_SCHEMA,
            "model_state": model.state_dict(),
            "seed": seed,
            "step": 0,
            "validation_huber": best_loss,
            "initial_model_state_sha256": initial_state,
            "architecture": base.architecture_config(),
            "augmentation": base.augmentation_config(),
        },
        checkpoint,
    )
    history.append(
        {
            "update_step": 0,
            "last_train_smooth_l1": None,
            "mean_train_smooth_l1_since_validation": None,
            "validation_smooth_l1": best_loss,
            "clean_windows_since_validation": 0,
            "gaussian_windows_since_validation": 0,
            "masked_windows_since_validation": 0,
            "improved": True,
            "stale_validations": 0,
        }
    )

    updates = 0
    recent_losses: list[float] = []
    recent_mode_counts = np.zeros(3, dtype=np.int64)
    while updates < maximum_updates and stale_validations < validation_patience:
        for (clean,) in loader:
            model.train()
            clean = clean.to(device, non_blocking=True)
            network_input, counts = base.corrupt_gru_base(clean, augmentation_generator)
            recent_mode_counts += counts
            total_mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite NBM loss at update {updates + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite NBM gradient at update {updates + 1}")
            optimizer.step()
            updates += 1
            recent_losses.append(float(loss.detach()))

            if updates % validation_frequency == 0 or updates == maximum_updates:
                validation_loss = clean_validation_loss(
                    model, validation_x, device, batch_size
                )
                improved = validation_loss < best_loss - 1e-10
                if improved:
                    best_loss = validation_loss
                    best_step = updates
                    stale_validations = 0
                    base.atomic_torch_save(
                        {
                            "schema": EXPERIMENT_SCHEMA,
                            "model_state": model.state_dict(),
                            "seed": seed,
                            "step": updates,
                            "validation_huber": validation_loss,
                            "initial_model_state_sha256": initial_state,
                            "architecture": base.architecture_config(),
                            "augmentation": base.augmentation_config(),
                        },
                        checkpoint,
                    )
                else:
                    stale_validations += 1
                history.append(
                    {
                        "update_step": updates,
                        "last_train_smooth_l1": recent_losses[-1],
                        "mean_train_smooth_l1_since_validation": float(np.mean(recent_losses)),
                        "validation_smooth_l1": validation_loss,
                        "clean_windows_since_validation": int(recent_mode_counts[0]),
                        "gaussian_windows_since_validation": int(recent_mode_counts[1]),
                        "masked_windows_since_validation": int(recent_mode_counts[2]),
                        "improved": improved,
                        "stale_validations": stale_validations,
                    }
                )
                print(
                    f"GRU-NBM seed={seed} step={updates:04d}/{maximum_updates} "
                    f"train={float(np.mean(recent_losses)):.7f} val={validation_loss:.7f} "
                    f"best={best_loss:.7f}@{best_step} stale={stale_validations}/"
                    f"{validation_patience} modes={recent_mode_counts.tolist()}",
                    flush=True,
                )
                recent_losses.clear()
                recent_mode_counts[:] = 0
            if updates >= maximum_updates or stale_validations >= validation_patience:
                break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != base.architecture_config():
        raise AssertionError("best NBM checkpoint architecture mismatch")
    if payload.get("augmentation") != base.augmentation_config():
        raise AssertionError("best NBM checkpoint augmentation mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    restored_loss = clean_validation_loss(model, validation_x, device, batch_size)
    if abs(restored_loss - best_loss) > 1e-7:
        raise AssertionError("restored validation loss does not match best checkpoint")
    total_windows = int(total_mode_counts.sum())
    return model, {
        "maximum_updates": maximum_updates,
        "updates_completed": updates,
        "validation_frequency_updates": validation_frequency,
        "validation_patience_checks": validation_patience,
        "patience_equivalent_updates": validation_frequency * validation_patience,
        "best_step": best_step,
        "best_validation_smooth_l1": best_loss,
        "restored_validation_smooth_l1": restored_loss,
        "stopped_early": updates < maximum_updates,
        "initial_model_state_sha256": initial_state,
        "parameter_count": parameter_count,
        "total_clean_windows": int(total_mode_counts[0]),
        "total_gaussian_windows": int(total_mode_counts[1]),
        "total_masked_windows": int(total_mode_counts[2]),
        "empirical_clean_fraction": float(total_mode_counts[0] / total_windows),
        "empirical_gaussian_fraction": float(total_mode_counts[1] / total_windows),
        "empirical_mask_fraction": float(total_mode_counts[2] / total_windows),
        "history": history,
    }


def training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scaler": "per-axis median/IQR fitted on unique role4 raw samples",
        "nbm_preprocessing": "RobustScaler then per-window/per-axis time centering",
        "nbm": base.architecture_config(),
        "augmentation": base.augmentation_config(),
        "augmentation_sampling": "dynamic independent mutually-exclusive draw per role-4 window encounter",
        "nbm_loss": "SmoothL1(beta=1.0), corrupted input predicts clean target",
        "nbm_optimizer": f"AdamW(lr={args.nbm_learning_rate},weight_decay={args.nbm_weight_decay})",
        "nbm_scheduler": None,
        "nbm_batch_size": args.nbm_batch_size,
        "nbm_maximum_updates": args.maximum_updates,
        "nbm_validation_frequency_updates": args.validation_frequency,
        "nbm_validation_patience_checks": args.validation_patience,
        "nbm_checkpoint": "minimum clean role5 SmoothL1",
        "gradient_clip_global_norm": 1.0,
        "calibration": "after restoring best NBM, role5 b=median(e), sigma=max(1.4826*MAD(e-b),0.05)",
        "scheme_c": "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); r=q-mean_t(q); [r,abs(r),delta(r)]",
        "scheme_c_uses_bias_b": False,
        "tcn_input_shape": ["B", 90, 128],
        "tcn": "RepresentationTCNM 90->32->64->64->128; dilations1/2/4/8; GAP; one logit",
        "classifier_train_roles": [6, 7],
        "classifier_validation_roles": [2, 3],
        "classifier_test_roles": [0, 1],
        "tcn_loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "tcn_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "tcn_batch_size": args.tcn_batch_size,
        "tcn_maximum_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "tcn_checkpoint": "maximum roles2/3 AP",
        "tcn_initialization": "reset from the job seed immediately before TCN construction",
        "threshold": "roles2/3 grid 0.05..0.95 step0.01; max balanced accuracy; ties F1 then higher threshold",
    }


def validate_plan_args(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "nbm_batch_size": int(args.nbm_batch_size),
        "nbm_maximum_updates": int(args.maximum_updates),
        "nbm_validation_frequency": int(args.validation_frequency),
        "nbm_validation_patience": int(args.validation_patience),
        "nbm_learning_rate": float(args.nbm_learning_rate),
        "nbm_weight_decay": float(args.nbm_weight_decay),
        "tcn_batch_size": int(args.tcn_batch_size),
        "tcn_max_epochs": int(args.tcn_max_epochs),
        "tcn_patience": int(args.tcn_patience),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}")


def run_train(args: argparse.Namespace) -> None:
    subject, fold, seed = base.require_job_args(args)
    destination = base.run_dir(args.output_root.resolve(), subject, fold, seed)
    plan = base.load_plan(args.output_root.resolve())
    validate_plan_args(args, plan)
    if not args.overwrite and base.validate_completed_train(destination, plan):
        print(f"SKIP validated completed train job: {destination}", flush=True)
        return

    dataset = DaphnetDataset.load(args.data_dir.resolve())
    if dataset.sampling_rate_hz != base.SAMPLING_RATE_HZ or dataset.n_channels != base.RAW_CHANNELS:
        raise AssertionError("expected processed_NBM_Exp 64-Hz/30-channel data")
    rows = base.raw_base.load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    role4, role5 = rows.take_role(4), rows.take_role(5)
    role67, role23 = rows.take_role(6, 7), rows.take_role(2, 3)
    scaler, scaler_points = base.raw_base.fit_scaler_unique_role4_points(dataset, role4)
    role4_x = base.centered_scaled_ntc(scaler, base.raw_base.raw_windows(dataset, role4))
    role5_x = base.centered_scaled_ntc(scaler, base.raw_base.raw_windows(dataset, role5))
    device = base.resolve_device(args.device)
    nbm, nbm_training = train_nbm_by_updates(
        role4_x,
        role5_x,
        destination,
        device,
        seed,
        args.nbm_batch_size,
        args.num_workers,
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
        args.nbm_learning_rate,
        args.nbm_weight_decay,
    )
    bias, sigma, calibration = base.calibrate(
        nbm, role5_x, device, args.nbm_batch_size
    )
    scaler_path = destination / "scaler_role4.json"
    calibration_path = destination / "calibration_role5.json"
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "subject": subject,
            "fold": fold,
            "seed": seed,
            "fit_role": 4,
            "unique_raw_points": scaler_points,
            "scaler": scaler.as_dict(),
        },
        scaler_path,
    )
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "subject": subject,
            "fold": fold,
            "seed": seed,
            "source_role": 5,
            **calibration,
        },
        calibration_path,
    )

    train_x = base.scheme_c_features(
        nbm,
        scaler,
        sigma,
        base.raw_base.raw_windows(dataset, role67),
        device,
        args.nbm_batch_size,
    )
    validation_x = base.scheme_c_features(
        nbm,
        scaler,
        sigma,
        base.raw_base.raw_windows(dataset, role23),
        device,
        args.nbm_batch_size,
    )
    tcn, tcn_training = base.train_tcn(
        train_x,
        role67.label,
        validation_x,
        role23.label,
        destination,
        device,
        seed,
        args.tcn_batch_size,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
    )
    val_true, val_prob = base.predict(
        tcn, validation_x, role23.label, device, args.tcn_batch_size
    )
    threshold, validation_metrics = base.raw_base.choose_threshold(val_true, val_prob)
    nbm_history_path = destination / "nbm_history.csv"
    tcn_history_path = destination / "tcn_history.csv"
    base.write_csv(nbm_history_path, nbm_training["history"])
    base.write_csv(tcn_history_path, tcn_training["history"])
    nbm_checkpoint = destination / "checkpoints" / base.NBM_CHECKPOINT_NAME
    tcn_checkpoint = destination / "checkpoints" / "tcn.pt"
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_before_permanent_test",
        "created_utc": base.utc_now(),
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "nbm_variant": NBM_VARIANT,
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "code_sha256": plan["code_sha256"],
        "nbm_checkpoint_sha256": sha256_file(nbm_checkpoint),
        "tcn_checkpoint_sha256": sha256_file(tcn_checkpoint),
        "scaler_sha256": sha256_file(scaler_path),
        "calibration_sha256": sha256_file(calibration_path),
        "nbm_history_sha256": sha256_file(nbm_history_path),
        "tcn_history_sha256": sha256_file(tcn_history_path),
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "validation_metrics": validation_metrics,
        "nbm_training": {key: value for key, value in nbm_training.items() if key != "history"},
        "tcn_training": {key: value for key, value in tcn_training.items() if key != "history"},
        "calibration": calibration,
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
        f"nbm_best_step={nbm_training['best_step']} "
        f"tcn_best={tcn_training['best_epoch']} threshold={threshold:.2f} "
        f"val_ap={validation_metrics['auprc']:.6f} "
        f"tcn_init={tcn_training['initial_model_state_sha256']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.nbm_batch_size not in (16, 32):
        raise ValueError("NBM batch size must be 16 or 32")
    if min(
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
        args.tcn_batch_size,
        args.tcn_max_epochs,
        args.tcn_patience,
    ) <= 0:
        raise ValueError("all integer training settings must be positive")
    if args.maximum_updates % args.validation_frequency != 0:
        raise ValueError("maximum updates must be divisible by validation frequency")
    if args.stage == "train":
        run_train(args)
    elif args.stage == "seal":
        base.run_seal(args)
    elif args.stage == "evaluate":
        # NBM inference is sample-independent and contains no BatchNorm, so the
        # base evaluator may safely use the TCN batch size for both frozen models.
        args.batch_size = args.tcn_batch_size
        base.run_evaluate(args)
    else:
        base.run_aggregate(args)


if __name__ == "__main__":
    main()
