#!/usr/bin/env python3
"""Train one clean Scheme-C TCN matched to one frozen Private GRU-NGM.

The source GRU-NGM and its role-4 scaler remain frozen.  This training-only
stage recalibrates residual scale on clean role 5, trains the 90-channel TCN on
clean roles 6/7, selects its checkpoint on clean roles 2/3 AP, and never
materializes role-0/1 test windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as base


EXPERIMENT_SCHEMA = "private_gru_ngm_robustness_matched_tcn.v1"
PLAN_SCHEMA = "private_gru_ngm_robustness_matched_tcn_plan.v1"
ARMS = ("none", "gaussian_mask")
ARM_DISPLAY_NAMES = {
    "none": "No perturbation",
    "gaussian_mask": "Gaussian + Mask",
}
SUBJECTS = base.SUBJECTS
FOLDS = base.FOLDS
SEEDS = base.SEEDS
SOURCE_CHECKPOINT_NAME = "gru_ngm_best.pt"
TCN_MAX_EPOCHS = 5
TCN_PATIENCE = 2
TCN_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--subject", choices=SUBJECTS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=TCN_MAX_EPOCHS)
    parser.add_argument("--tcn-patience", type=int, default=TCN_PATIENCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def job_key(arm: str, subject: str, fold: int, seed: int) -> str:
    return f"{arm}/{subject}/fold_{fold}/seed_{seed}"


def run_dir(
    root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> Path:
    return root / "runs" / arm / subject / f"fold_{fold}" / f"seed_{seed}"


def _checkpoint_in(directory: Path) -> Path | None:
    for path in (
        directory / "checkpoints" / SOURCE_CHECKPOINT_NAME,
        directory / SOURCE_CHECKPOINT_NAME,
    ):
        if path.is_file():
            return path
    return None


def source_run_candidates(
    root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> tuple[Path, ...]:
    """Accept an arm root, experiment root, runs root, or subject root."""

    return (
        root / subject / f"fold_{fold}" / f"seed_{seed}",
        root / arm / subject / f"fold_{fold}" / f"seed_{seed}",
        root / "runs" / arm / subject / f"fold_{fold}" / f"seed_{seed}",
        root / f"fold_{fold}" / f"seed_{seed}",
    )


def resolve_source_run_dir(
    root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> Path:
    matches = [
        candidate.resolve()
        for candidate in source_run_candidates(
            root.resolve(), arm, subject, fold, seed
        )
        if _checkpoint_in(candidate) is not None
    ]
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        expected = "\n".join(
            str(path)
            for path in source_run_candidates(root, arm, subject, fold, seed)
        )
        raise FileNotFoundError(
            "no Private 30-channel GRU-NGM checkpoint found for "
            f"arm={arm}, subject={subject}, fold={fold}, seed={seed}; checked:\n"
            f"{expected}"
        )
    if len(unique) != 1:
        raise RuntimeError(
            "ambiguous Private GRU-NGM source for "
            f"arm={arm}, subject={subject}, fold={fold}, seed={seed}: {unique}"
        )
    return unique[0]


def source_job_exists(
    root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> bool:
    return any(
        _checkpoint_in(candidate) is not None
        for candidate in source_run_candidates(root, arm, subject, fold, seed)
    )


def scaler_dict_from_path(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scaler = payload.get("scaler", payload)
    required = ("median", "iqr", "epsilon")
    if not all(key in scaler for key in required):
        raise KeyError(f"invalid role-4 scaler artifact: {path}")
    median = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    if median.shape != (base.RAW_CHANNELS,) or iqr.shape != (base.RAW_CHANNELS,):
        raise ValueError(
            f"Private scaler must contain {base.RAW_CHANNELS} channels: {path}"
        )
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(iqr)):
        raise FloatingPointError(f"non-finite scaler values: {path}")
    if np.any(iqr <= 0):
        raise ValueError(f"non-positive scaler IQR values: {path}")
    return {
        "median": median.astype(float).tolist(),
        "iqr": iqr.astype(float).tolist(),
        "epsilon": float(scaler["epsilon"]),
    }


def scaler_from_dict(payload: dict[str, Any]) -> base.RobustScaler:
    return base.RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float32),
        iqr=np.asarray(payload["iqr"], dtype=np.float32),
        epsilon=float(payload["epsilon"]),
    )


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def inspect_source_artifacts(
    root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
    scientific_data_sha256: str | None = None,
) -> dict[str, Any]:
    directory = resolve_source_run_dir(root, arm, subject, fold, seed)
    checkpoint = _checkpoint_in(directory)
    if checkpoint is None:
        raise FileNotFoundError(
            directory / "checkpoints" / SOURCE_CHECKPOINT_NAME
        )
    scaler_path = directory / "scaler_role4.json"
    if not scaler_path.is_file():
        raise FileNotFoundError(scaler_path)
    scaler = scaler_dict_from_path(scaler_path)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise KeyError(f"source checkpoint has no model_state: {checkpoint}")
    expected_architecture = base.architecture_config()
    if payload.get("architecture") != expected_architecture:
        raise AssertionError(f"source GRU-NGM architecture mismatch: {checkpoint}")
    expected_fields = {
        "arm": arm,
        "seed": seed,
    }
    for key, expected in expected_fields.items():
        if payload.get(key) is not None and payload.get(key) != expected:
            raise AssertionError(
                f"source checkpoint {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    probe = base.GRUReconstructionNBM(
        channels=base.RAW_CHANNELS,
        hidden=base.HIDDEN,
        bottleneck=base.BOTTLENECK,
    )
    probe.load_state_dict(payload["model_state"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    if parameter_count != base.NBM_PARAMETER_COUNT:
        raise AssertionError("Private GRU-NGM parameter contract changed")
    del probe

    frozen_path = directory / "FROZEN_TRAIN.json"
    done_path = directory / "DONE_TRAIN.json"
    checkpoint_hash = sha256_file(checkpoint)
    scaler_file_hash = sha256_file(scaler_path)
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("arm", arm),
            ("subject", subject),
            ("fold", fold),
            ("seed", seed),
            ("checkpoint_sha256", checkpoint_hash),
            ("scaler_sha256", scaler_file_hash),
        ):
            if frozen.get(key) != expected:
                raise AssertionError(
                    f"source FROZEN_TRAIN {key} mismatch: "
                    f"{frozen.get(key)!r} != {expected!r}"
                )
        source_data_hash = frozen.get("data_scientific_sha256")
        if (
            scientific_data_sha256 is not None
            and source_data_hash != scientific_data_sha256
        ):
            raise AssertionError(
                f"source GRU-NGM dataset mismatch: {source_data_hash} != "
                f"{scientific_data_sha256}"
            )
        if done_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("frozen_sha256") != sha256_file(frozen_path):
                raise AssertionError(f"source DONE_TRAIN hash mismatch: {done_path}")
            if done.get("frozen_id") != frozen.get("frozen_id"):
                raise AssertionError(f"source DONE_TRAIN id mismatch: {done_path}")

    return {
        "source_root": str(root.resolve()),
        "source_run_dir": str(directory),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_name": checkpoint.name,
        "checkpoint_seed": payload.get("seed"),
        "checkpoint_step": payload.get("step"),
        "parameter_count": parameter_count,
        "scaler_source": str(scaler_path.resolve()),
        "scaler_source_sha256": scaler_file_hash,
        "scaler": scaler,
        "scaler_values_sha256": canonical_fingerprint(scaler),
        "frozen_train_sha256": _optional_sha256(frozen_path),
        "done_train_sha256": _optional_sha256(done_path),
    }


def load_plan(plan_root: Path) -> dict[str, Any]:
    path = plan_root.resolve() / "EXPERIMENT_PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"TCN experiment plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise AssertionError(f"unexpected TCN plan schema: {plan.get('schema')}")
    return plan


def validate_args_against_plan(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if (args.tcn_max_epochs, args.tcn_patience) != (
        TCN_MAX_EPOCHS,
        TCN_PATIENCE,
    ):
        raise ValueError(
            f"matched TCN is frozen to max_epochs={TCN_MAX_EPOCHS}, "
            f"patience={TCN_PATIENCE}"
        )
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tcn_batch_size": TCN_BATCH_SIZE,
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(
                f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}"
            )
    key = job_key(args.arm, args.subject, args.fold, args.seed)
    if key not in plan.get("source_jobs", {}):
        raise KeyError(f"job absent from frozen plan: {key}")


def load_source_model(
    source: dict[str, Any], device: torch.device
) -> base.GRUReconstructionNBM:
    checkpoint = Path(source["checkpoint"])
    scaler_path = Path(source["scaler_source"])
    if sha256_file(checkpoint) != source["checkpoint_sha256"]:
        raise AssertionError(f"source GRU-NGM checkpoint changed: {checkpoint}")
    if sha256_file(scaler_path) != source["scaler_source_sha256"]:
        raise AssertionError(f"source role-4 scaler artifact changed: {scaler_path}")
    current_scaler = scaler_dict_from_path(scaler_path)
    if canonical_fingerprint(current_scaler) != source["scaler_values_sha256"]:
        raise AssertionError(f"source role-4 scaler values changed: {scaler_path}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = base.build_nbm_from_checkpoint(payload, device)
    model.eval()
    return model


def feature_diagnostics(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(features.shape),
        "finite": bool(np.all(np.isfinite(features))),
        "nonfog_windows": int(np.sum(labels == 0)),
        "fog_windows": int(np.sum(labels == 1)),
        "maximum_absolute_value": float(np.max(np.abs(features))),
    }


def completed_training_is_valid(
    destination: Path,
    plan: dict[str, Any],
    source: dict[str, Any],
) -> bool:
    done_path = destination / "DONE_TCN.json"
    if not done_path.is_file():
        return False
    frozen_path = destination / "FROZEN_TCN.json"
    checkpoint = destination / "checkpoints" / "tcn.pt"
    history = destination / "tcn_history.csv"
    calibration = destination / "calibration_role5.json"
    required = (frozen_path, checkpoint, history, calibration)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete completed TCN job: {destination}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    valid = (
        done.get("frozen_sha256") == sha256_file(frozen_path)
        and done.get("frozen_id") == frozen.get("frozen_id")
        and frozen.get("plan_id") == plan.get("plan_id")
        and frozen.get("source_ngm_checkpoint_sha256")
        == source.get("checkpoint_sha256")
        and frozen.get("tcn_checkpoint_sha256") == sha256_file(checkpoint)
        and frozen.get("tcn_history_sha256") == sha256_file(history)
        and frozen.get("calibration_sha256") == sha256_file(calibration)
        and frozen.get("test_roles_accessed") is False
    )
    if not valid:
        raise AssertionError(f"completed TCN artifacts failed validation: {destination}")
    return True


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def run_train(args: argparse.Namespace) -> None:
    args.data_dir = args.data_dir.resolve()
    args.plan_root = args.plan_root.resolve()
    args.output_root = args.output_root.resolve()
    plan = load_plan(args.plan_root)
    validate_args_against_plan(args, plan)
    key = job_key(args.arm, args.subject, args.fold, args.seed)
    source = plan["source_jobs"][key]
    destination = run_dir(
        args.output_root, args.arm, args.subject, args.fold, args.seed
    )
    if not args.overwrite and completed_training_is_valid(destination, plan, source):
        print(f"SKIP validated completed TCN job: {destination}", flush=True)
        return

    current_scientific = processed_nbm_scientific_manifest(args.data_dir)["sha256"]
    if current_scientific != plan["data_scientific_sha256"]:
        raise AssertionError("Private scientific dataset changed after plan freeze")
    dataset = DaphnetDataset.load(args.data_dir)
    if (
        dataset.sampling_rate_hz != base.SAMPLING_RATE_HZ
        or dataset.n_channels != base.RAW_CHANNELS
    ):
        raise AssertionError(
            f"expected Private 64-Hz/{base.RAW_CHANNELS}-channel data, got "
            f"{dataset.sampling_rate_hz}/{dataset.n_channels}"
        )
    rows = base.raw_base.load_subject_rows(
        args.data_dir, dataset, args.subject, args.fold
    )
    role5 = rows.take_role(5)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    if min(len(role5), len(role67), len(role23)) <= 0:
        raise ValueError("roles 5/6/7/2/3 must all be non-empty")
    if not np.array_equal(role67.label, np.isin(role67.role, [7]).astype(np.int8)):
        raise AssertionError("classifier training labels do not match roles 6/7")
    if not np.array_equal(role23.label, np.isin(role23.role, [3]).astype(np.int8)):
        raise AssertionError("classifier validation labels do not match roles 2/3")

    scaler = scaler_from_dict(source["scaler"])
    device = base.resolve_device(args.device)
    model = load_source_model(source, device)
    role5_x = base.centered_scaled_ntc(
        scaler, base.raw_base.raw_windows(dataset, role5)
    )
    bias, sigma, calibration = base.calibrate(
        model, role5_x, device, TCN_BATCH_SIZE
    )
    if (
        bias.shape != (base.RAW_CHANNELS,)
        or sigma.shape != (base.RAW_CHANNELS,)
        or np.any(sigma < 0.05)
    ):
        raise AssertionError("invalid clean role-5 GRU-NGM calibration")
    train_x = base.scheme_c_features(
        model,
        scaler,
        sigma,
        base.raw_base.raw_windows(dataset, role67),
        device,
        TCN_BATCH_SIZE,
    )
    validation_x = base.scheme_c_features(
        model,
        scaler,
        sigma,
        base.raw_base.raw_windows(dataset, role23),
        device,
        TCN_BATCH_SIZE,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    destination.mkdir(parents=True, exist_ok=True)
    tcn, training = base.train_tcn(
        train_x,
        role67.label,
        validation_x,
        role23.label,
        destination,
        device,
        args.seed,
        TCN_BATCH_SIZE,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
    )
    val_true, val_probability = base.predict(
        tcn,
        validation_x,
        role23.label,
        device,
        TCN_BATCH_SIZE,
    )
    threshold, validation_metrics = base.raw_base.choose_threshold(
        val_true, val_probability
    )
    del tcn

    calibration_path = destination / "calibration_role5.json"
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "arm": args.arm,
            "subject": args.subject,
            "fold": args.fold,
            "seed": args.seed,
            "source_role": 5,
            "source_ngm_checkpoint_sha256": source["checkpoint_sha256"],
            **calibration,
        },
        calibration_path,
    )
    history_path = destination / "tcn_history.csv"
    write_csv(history_path, training["history"])
    checkpoint = destination / "checkpoints" / "tcn.pt"
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_before_robustness_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "arm": args.arm,
        "arm_display_name": ARM_DISPLAY_NAMES[args.arm],
        "subject": args.subject,
        "fold": args.fold,
        "seed": args.seed,
        "source_ngm": source,
        "source_ngm_checkpoint_sha256": source["checkpoint_sha256"],
        "role4_scaler_values_sha256": source["scaler_values_sha256"],
        "calibration_sha256": sha256_file(calibration_path),
        "tcn_checkpoint": str(checkpoint.resolve()),
        "tcn_checkpoint_sha256": sha256_file(checkpoint),
        "tcn_history_sha256": sha256_file(history_path),
        "representation": {
            "name": "Scheme C",
            "formula": (
                "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
                "r=q-mean_t(q); F=[r,abs(r),delta_t(r)]"
            ),
            "input_shape": ["B", 90, 128],
        },
        "roles": {
            "ngm_calibration": [5],
            "tcn_train": [6, 7],
            "tcn_validation": [2, 3],
            "test_not_materialized": [0, 1],
        },
        "training": {
            key: value for key, value in training.items() if key != "history"
        },
        "threshold": float(threshold),
        "threshold_source_roles": [2, 3],
        "threshold_not_used_for_ap": True,
        "validation": validation_metrics,
        "feature_diagnostics": {
            "roles_6_7_train": feature_diagnostics(train_x, role67.label),
            "roles_2_3_validation": feature_diagnostics(
                validation_x, role23.label
            ),
        },
        "test_roles_accessed": False,
        "test_corruption_used_during_tcn_training": False,
    }
    frozen["frozen_id"] = canonical_fingerprint(frozen)
    frozen_path = destination / "FROZEN_TCN.json"
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
            "test_roles_accessed": False,
        },
        destination / "DONE_TCN.json",
    )
    print(
        f"TCN TRAIN COMPLETE arm={args.arm} subject={args.subject} "
        f"fold={args.fold} seed={args.seed} "
        f"best_epoch={training['best_epoch']} "
        f"validation_ap={training['best_validation_pr_auc']:.7f}",
        flush=True,
    )


def main() -> None:
    run_train(parse_args())


if __name__ == "__main__":
    main()
