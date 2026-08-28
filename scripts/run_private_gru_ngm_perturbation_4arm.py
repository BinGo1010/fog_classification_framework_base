#!/usr/bin/env python
"""Train one frozen GRU-NGM perturbation-ablation job on processed_NBM_Exp.

The only experimental factor is the dynamic role-4 input corruption arm.  All
arms reconstruct the original clean window and select their checkpoint on the
same uncorrupted role-5 validation windows.  Permanent-test roles are never
loaded by this training-only worker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as base


EXPERIMENT_SCHEMA = "private_gru_ngm_perturbation_4arm_training.v1"
PLAN_SCHEMA = "private_gru_ngm_perturbation_4arm_plan.v1"
ARMS = ("none", "gaussian_only", "mask_only", "gaussian_mask")
ARM_DISPLAY_NAMES = {
    "none": "No perturbation",
    "gaussian_only": "Gaussian only",
    "mask_only": "Mask only",
    "gaussian_mask": "Gaussian + Mask",
}
SEEDS = base.SEEDS
SUBJECTS = base.SUBJECTS
FOLDS = base.FOLDS
CHECKPOINT_NAME = "gru_ngm_best.pt"

# These probabilities nest the single-corruption controls inside the previously
# frozen 40% clean / 40% Gaussian / 20% mask training mixture.  Thus adding the
# second corruption never changes exposure to the first one.
ARM_PROBABILITIES: dict[str, tuple[float, float, float]] = {
    "none": (1.00, 0.00, 0.00),
    "gaussian_only": (0.60, 0.40, 0.00),
    "mask_only": (0.80, 0.00, 0.20),
    "gaussian_mask": (0.40, 0.40, 0.20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--subject", choices=SUBJECTS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-batch-size", type=int, default=16)
    parser.add_argument("--maximum-updates", type=int, default=5000)
    parser.add_argument("--validation-frequency", type=int, default=50)
    parser.add_argument("--validation-patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_dir(
    root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> Path:
    return root / "runs" / arm / subject / f"fold_{fold}" / f"seed_{seed}"


def augmentation_config(arm: str) -> dict[str, Any]:
    if arm not in ARM_PROBABILITIES:
        raise ValueError(f"unknown perturbation arm: {arm}")
    clean, gaussian, mask = ARM_PROBABILITIES[arm]
    if not math.isclose(clean + gaussian + mask, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"probabilities do not sum to one for {arm}")
    return {
        "arm": arm,
        "display_name": ARM_DISPLAY_NAMES[arm],
        "clean_probability": clean,
        "gaussian_probability": gaussian,
        "mask_probability": mask,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_length_sampling": "discrete_uniform_inclusive",
        "mask_contiguous": True,
        "mask_all_channels": True,
        "mask_replacement_value": 0.0,
        "augmentation_roles": [4],
        "validation_augmentation": False,
        "target": "original clean role-4 window",
        "sampling": "dynamic mutually-exclusive draw per window encounter",
    }


def corrupt_for_arm(
    clean: torch.Tensor,
    arm: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return corrupted inputs and counts in [clean, Gaussian, mask] order."""

    if clean.ndim != 3 or tuple(clean.shape[1:]) != (
        base.WINDOW_SAMPLES,
        base.RAW_CHANNELS,
    ):
        raise ValueError(f"expected [B,128,30], got {tuple(clean.shape)}")
    config = augmentation_config(arm)
    clean_probability = float(config["clean_probability"])
    gaussian_probability = float(config["gaussian_probability"])
    gaussian_upper = clean_probability + gaussian_probability

    output = clean.clone()
    modes = torch.rand(clean.shape[0], device=clean.device, generator=generator)
    gaussian = (modes >= clean_probability) & (modes < gaussian_upper)
    masked = modes >= gaussian_upper

    if torch.any(gaussian):
        noise = torch.randn(
            output[gaussian].shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        output[gaussian] += float(config["gaussian_std"]) * noise

    masked_indices = torch.nonzero(masked, as_tuple=False).flatten().tolist()
    minimum = int(config["mask_minimum_samples"])
    maximum = int(config["mask_maximum_samples"])
    for index in masked_indices:
        length = int(
            torch.randint(
                minimum,
                maximum + 1,
                (1,),
                device=clean.device,
                generator=generator,
            )
        )
        start = int(
            torch.randint(
                0,
                base.WINDOW_SAMPLES - length + 1,
                (1,),
                device=clean.device,
                generator=generator,
            )
        )
        output[index, start : start + length, :] = 0.0

    counts = np.asarray(
        [
            int((modes < clean_probability).sum()),
            int(gaussian.sum()),
            len(masked_indices),
        ],
        dtype=np.int64,
    )
    if int(counts.sum()) != len(clean):
        raise AssertionError("corruption-mode accounting mismatch")
    return output, counts


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
        raise ValueError("empty clean role-5 validation set")
    return total / count


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def train_ngm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    destination: Path,
    device: torch.device,
    arm: str,
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
        raise AssertionError("GRU-NGM parameter contract changed")
    if any(isinstance(module, nn.Dropout) for module in model.modules()):
        raise AssertionError("frozen GRU-NGM unexpectedly contains dropout")

    initial_state = base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    loader = base.nbm_loader(train_x, batch_size, True, seed, workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = destination / "checkpoints" / CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_loss = clean_validation_loss(model, validation_x, device, batch_size)
    best_step = 0
    stale_validations = 0
    history: list[dict[str, Any]] = []
    total_mode_counts = np.zeros(3, dtype=np.int64)
    config = augmentation_config(arm)
    atomic_torch_save(
        {
            "schema": EXPERIMENT_SCHEMA,
            "model_state": model.state_dict(),
            "arm": arm,
            "seed": seed,
            "step": 0,
            "validation_smooth_l1": best_loss,
            "initial_model_state_sha256": initial_state,
            "architecture": base.architecture_config(),
            "augmentation": config,
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
    started = time.perf_counter()
    while updates < maximum_updates and stale_validations < validation_patience:
        for (clean,) in loader:
            model.train()
            clean = clean.to(device, non_blocking=True)
            network_input, counts = corrupt_for_arm(
                clean, arm, augmentation_generator
            )
            recent_mode_counts += counts
            total_mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite GRU-NGM loss at update {updates + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite GRU-NGM gradient at update {updates + 1}"
                )
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
                    atomic_torch_save(
                        {
                            "schema": EXPERIMENT_SCHEMA,
                            "model_state": model.state_dict(),
                            "arm": arm,
                            "seed": seed,
                            "step": updates,
                            "validation_smooth_l1": validation_loss,
                            "initial_model_state_sha256": initial_state,
                            "architecture": base.architecture_config(),
                            "augmentation": config,
                        },
                        checkpoint,
                    )
                else:
                    stale_validations += 1
                history.append(
                    {
                        "update_step": updates,
                        "last_train_smooth_l1": recent_losses[-1],
                        "mean_train_smooth_l1_since_validation": float(
                            np.mean(recent_losses)
                        ),
                        "validation_smooth_l1": validation_loss,
                        "clean_windows_since_validation": int(
                            recent_mode_counts[0]
                        ),
                        "gaussian_windows_since_validation": int(
                            recent_mode_counts[1]
                        ),
                        "masked_windows_since_validation": int(
                            recent_mode_counts[2]
                        ),
                        "improved": improved,
                        "stale_validations": stale_validations,
                    }
                )
                print(
                    f"GRU-NGM arm={arm} seed={seed} "
                    f"step={updates:04d}/{maximum_updates} "
                    f"train={float(np.mean(recent_losses)):.7f} "
                    f"clean_val={validation_loss:.7f} "
                    f"best={best_loss:.7f}@{best_step} "
                    f"stale={stale_validations}/{validation_patience} "
                    f"modes={recent_mode_counts.tolist()}",
                    flush=True,
                )
                recent_losses.clear()
                recent_mode_counts[:] = 0
            if updates >= maximum_updates or stale_validations >= validation_patience:
                break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != base.architecture_config():
        raise AssertionError("best GRU-NGM checkpoint architecture mismatch")
    if payload.get("augmentation") != config or payload.get("arm") != arm:
        raise AssertionError("best GRU-NGM checkpoint arm mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    restored_loss = clean_validation_loss(model, validation_x, device, batch_size)
    if abs(restored_loss - best_loss) > 1e-7:
        raise AssertionError("restored clean validation loss differs from best checkpoint")

    total_windows = int(total_mode_counts.sum())
    if total_windows <= 0:
        raise AssertionError("no role-4 windows were used for training")
    return model, {
        "arm": arm,
        "maximum_updates": maximum_updates,
        "updates_completed": updates,
        "validation_frequency_updates": validation_frequency,
        "validation_patience_checks": validation_patience,
        "patience_equivalent_updates": validation_frequency
        * validation_patience,
        "best_step": best_step,
        "best_clean_validation_smooth_l1": best_loss,
        "restored_clean_validation_smooth_l1": restored_loss,
        "stopped_early": updates < maximum_updates,
        "initial_model_state_sha256": initial_state,
        "parameter_count": parameter_count,
        "elapsed_seconds": time.perf_counter() - started,
        "total_clean_windows": int(total_mode_counts[0]),
        "total_gaussian_windows": int(total_mode_counts[1]),
        "total_masked_windows": int(total_mode_counts[2]),
        "empirical_clean_fraction": float(total_mode_counts[0] / total_windows),
        "empirical_gaussian_fraction": float(
            total_mode_counts[1] / total_windows
        ),
        "empirical_mask_fraction": float(total_mode_counts[2] / total_windows),
        "history": history,
    }


def load_plan(root: Path) -> dict[str, Any]:
    path = root / "EXPERIMENT_PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"launcher plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise AssertionError(f"unexpected experiment plan schema: {plan.get('schema')}")
    return plan


def validate_plan_args(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "nbm_batch_size": int(args.nbm_batch_size),
        "maximum_updates": int(args.maximum_updates),
        "validation_frequency": int(args.validation_frequency),
        "validation_patience": int(args.validation_patience),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(
                f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}"
            )
    if args.arm not in plan.get("arms", []):
        raise AssertionError(f"arm absent from plan: {args.arm}")
    if args.subject not in plan.get("subjects", []):
        raise AssertionError(f"subject absent from plan: {args.subject}")
    if args.fold not in plan.get("folds", []):
        raise AssertionError(f"fold absent from plan: {args.fold}")
    if args.seed not in plan.get("seeds", []):
        raise AssertionError(f"seed absent from plan: {args.seed}")


def training_contract(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    return {
        "scaler": "per-axis median/IQR fitted on unique role-4 raw samples",
        "preprocessing": "RobustScaler then per-window/per-axis time centering",
        "architecture": base.architecture_config(),
        "augmentation": augmentation_config(arm),
        "loss": "SmoothL1(beta=1.0), perturbed input predicts original clean window",
        "optimizer": (
            f"AdamW(lr={args.learning_rate},weight_decay={args.weight_decay})"
        ),
        "scheduler": None,
        "batch_size": args.nbm_batch_size,
        "maximum_updates": args.maximum_updates,
        "validation_frequency_updates": args.validation_frequency,
        "validation_patience_checks": args.validation_patience,
        "checkpoint": "minimum uncorrupted role-5 SmoothL1",
        "gradient_clip_global_norm": 1.0,
        "permanent_test_roles_loaded": False,
    }


def validate_completed_train(destination: Path, plan: dict[str, Any]) -> bool:
    done_path = destination / "DONE_TRAIN.json"
    if not done_path.is_file():
        return False
    frozen_path = destination / "FROZEN_TRAIN.json"
    checkpoint = destination / "checkpoints" / CHECKPOINT_NAME
    scaler_path = destination / "scaler_role4.json"
    history_path = destination / "ngm_history.csv"
    required = (frozen_path, checkpoint, scaler_path, history_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete completed job: {destination}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    valid = (
        done.get("frozen_sha256") == sha256_file(frozen_path)
        and done.get("frozen_id") == frozen.get("frozen_id")
        and frozen.get("plan_id") == plan.get("plan_id")
        and frozen.get("data_scientific_sha256")
        == plan.get("data_scientific_sha256")
        and frozen.get("code_sha256") == plan.get("code_sha256")
        and frozen.get("checkpoint_sha256") == sha256_file(checkpoint)
        and frozen.get("scaler_sha256") == sha256_file(scaler_path)
        and frozen.get("history_sha256") == sha256_file(history_path)
    )
    if not valid:
        raise AssertionError(f"completed training artifacts failed validation: {destination}")
    return True


def run_train(args: argparse.Namespace) -> None:
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    plan = load_plan(args.output_root)
    validate_plan_args(args, plan)
    destination = run_dir(
        args.output_root, args.arm, args.subject, args.fold, args.seed
    )
    if not args.overwrite and validate_completed_train(destination, plan):
        print(f"SKIP validated completed training job: {destination}", flush=True)
        return

    dataset = DaphnetDataset.load(args.data_dir)
    if (
        dataset.sampling_rate_hz != base.SAMPLING_RATE_HZ
        or dataset.n_channels != base.RAW_CHANNELS
    ):
        raise AssertionError(
            "expected processed_NBM_Exp with 64-Hz, 30-channel windows"
        )
    rows = base.raw_base.load_subject_rows(
        args.data_dir, dataset, args.subject, args.fold
    )
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    if len(role4) == 0 or len(role5) == 0:
        raise ValueError("training requires non-empty role 4 and role 5")
    scaler, scaler_points = base.raw_base.fit_scaler_unique_role4_points(
        dataset, role4
    )
    role4_x = base.centered_scaled_ntc(
        scaler, base.raw_base.raw_windows(dataset, role4)
    )
    role5_x = base.centered_scaled_ntc(
        scaler, base.raw_base.raw_windows(dataset, role5)
    )
    device = base.resolve_device(args.device)
    model, training = train_ngm(
        role4_x,
        role5_x,
        destination,
        device,
        args.arm,
        args.seed,
        args.nbm_batch_size,
        args.num_workers,
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
        args.learning_rate,
        args.weight_decay,
    )
    del model

    scaler_path = destination / "scaler_role4.json"
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "subject": args.subject,
            "fold": args.fold,
            "seed": args.seed,
            "fit_role": 4,
            "unique_raw_points": scaler_points,
            "scaler": scaler.as_dict(),
        },
        scaler_path,
    )
    history_path = destination / "ngm_history.csv"
    write_csv(history_path, training["history"])
    checkpoint = destination / "checkpoints" / CHECKPOINT_NAME
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_training_complete",
        "plan_id": plan["plan_id"],
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "code_sha256": plan["code_sha256"],
        "arm": args.arm,
        "arm_display_name": ARM_DISPLAY_NAMES[args.arm],
        "subject": args.subject,
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "scaler_sha256": sha256_file(scaler_path),
        "history_sha256": sha256_file(history_path),
        "training": {
            key: value for key, value in training.items() if key != "history"
        },
        "training_contract": training_contract(args, args.arm),
        "role_counts": {
            str(role): int(np.sum(rows.role == role)) for role in base.ROLES
        },
        "permanent_test_materialized": False,
    }
    frozen["frozen_id"] = canonical_fingerprint(frozen)
    frozen_path = destination / "FROZEN_TRAIN.json"
    atomic_json_dump(frozen, frozen_path)
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "status": "train_complete",
            "arm": args.arm,
            "subject": args.subject,
            "fold": args.fold,
            "seed": args.seed,
            "frozen_id": frozen["frozen_id"],
            "frozen_sha256": sha256_file(frozen_path),
        },
        destination / "DONE_TRAIN.json",
    )
    print(
        f"TRAIN COMPLETE arm={args.arm} subject={args.subject} "
        f"fold={args.fold} seed={args.seed} "
        f"best_step={training['best_step']} "
        f"clean_val={training['best_clean_validation_smooth_l1']:.7f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
