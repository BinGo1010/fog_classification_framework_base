"""Run one private-data subject with step-based clean GRU-NBM training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts import summarize_private_raw_tcn_latest_event_metrics as latest_event
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import RepresentationTCNM
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    RobustScaler,
)


DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
SUBJECT = "P01"
FOLDS = (0, 1, 2)
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nbm-batch-size", type=int, default=16)
    parser.add_argument("--maximum-updates", type=int, default=5000)
    parser.add_argument("--validation-frequency", type=int, default=50)
    parser.add_argument("--validation-patience", type=int, default=20)
    parser.add_argument("--nbm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--nbm-weight-decay", type=float, default=1e-4)
    parser.add_argument("--tcn-batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def clean_loss(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    criterion = nn.SmoothL1Loss(beta=1.0)
    model.eval()
    total = 0.0
    count = 0
    for (batch,) in worker.nbm_loader(values, batch_size, False, 0, 0):
        batch = batch.to(device, non_blocking=True)
        loss = criterion(model(batch), batch)
        total += float(loss) * len(batch)
        count += len(batch)
    return total / count


def train_nbm_by_updates(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    destination: Path,
    device: torch.device,
    batch_size: int,
    maximum_updates: int,
    validation_frequency: int,
    validation_patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, dict[str, Any]]:
    worker.set_seed(SEED)
    model = GRUReconstructionNBM(
        channels=worker.RAW_CHANNELS,
        hidden=worker.HIDDEN,
        bottleneck=worker.BOTTLENECK,
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != worker.NBM_PARAMETER_COUNT:
        raise AssertionError("GRU-NBM parameter contract changed")
    if any(isinstance(module, nn.Dropout) for module in model.modules()):
        raise AssertionError("fixed GRU-NBM unexpectedly contains dropout")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    loader = worker.nbm_loader(train_x, batch_size, True, SEED, 0)
    checkpoint = destination / "checkpoints" / worker.NBM_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = clean_loss(model, validation_x, device, batch_size)
    best_step = 0
    stale_validations = 0
    worker.atomic_torch_save(
        {
            "schema": "private_p01_gru_nbm_step_training.v1",
            "model_state": model.state_dict(),
            "seed": SEED,
            "step": 0,
            "validation_huber": best_loss,
            "architecture": worker.architecture_config(),
        },
        checkpoint,
    )
    history.append(
        {
            "update_step": 0,
            "last_train_smooth_l1": None,
            "mean_train_smooth_l1_since_validation": None,
            "validation_smooth_l1": best_loss,
            "improved": True,
            "stale_validations": 0,
        }
    )
    updates = 0
    recent_losses: list[float] = []
    while updates < maximum_updates and stale_validations < validation_patience:
        for (clean,) in loader:
            model.train()
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(clean), clean)
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
                validation_loss = clean_loss(
                    model, validation_x, device, batch_size
                )
                improved = validation_loss < best_loss - 1e-10
                if improved:
                    best_loss = validation_loss
                    best_step = updates
                    stale_validations = 0
                    worker.atomic_torch_save(
                        {
                            "schema": "private_p01_gru_nbm_step_training.v1",
                            "model_state": model.state_dict(),
                            "seed": SEED,
                            "step": updates,
                            "validation_huber": validation_loss,
                            "architecture": worker.architecture_config(),
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
                        "improved": improved,
                        "stale_validations": stale_validations,
                    }
                )
                print(
                    f"NBM fold={destination.name} step={updates:04d}/{maximum_updates} "
                    f"train={float(np.mean(recent_losses)):.7f} "
                    f"val={validation_loss:.7f} best={best_loss:.7f}@{best_step} "
                    f"stale={stale_validations}/{validation_patience}",
                    flush=True,
                )
                recent_losses.clear()
            if updates >= maximum_updates or stale_validations >= validation_patience:
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != worker.architecture_config():
        raise AssertionError("best NBM checkpoint architecture mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    restored_validation_loss = clean_loss(model, validation_x, device, batch_size)
    if abs(restored_validation_loss - best_loss) > 1e-7:
        raise AssertionError("restored validation loss does not match best checkpoint")
    return model, {
        "maximum_updates": maximum_updates,
        "updates_completed": updates,
        "validation_frequency": validation_frequency,
        "validation_patience": validation_patience,
        "best_step": best_step,
        "best_validation_smooth_l1": best_loss,
        "restored_validation_smooth_l1": restored_validation_loss,
        "stopped_early": updates < maximum_updates,
        "history": history,
    }


def load_scaler(path: Path) -> RobustScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))["scaler"]
    return RobustScaler(
        np.asarray(payload["median"], dtype=np.float32),
        np.asarray(payload["iqr"], dtype=np.float32),
        float(payload["epsilon"]),
    )


def train_fold(
    args: argparse.Namespace,
    dataset: Any,
    fold: int,
    device: torch.device,
) -> dict[str, Any]:
    destination = args.output_dir / f"fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    rows = raw_base.load_subject_rows(args.data_dir, dataset, SUBJECT, fold)
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, scaler_points = raw_base.fit_scaler_unique_role4_points(dataset, role4)
    role4_x = worker.centered_scaled_ntc(
        scaler, raw_base.raw_windows(dataset, role4)
    )
    role5_x = worker.centered_scaled_ntc(
        scaler, raw_base.raw_windows(dataset, role5)
    )
    nbm, nbm_training = train_nbm_by_updates(
        role4_x,
        role5_x,
        destination,
        device,
        args.nbm_batch_size,
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
        args.nbm_learning_rate,
        args.nbm_weight_decay,
    )
    _, sigma, calibration = worker.calibrate(
        nbm, role5_x, device, args.nbm_batch_size
    )
    scaler_path = destination / "scaler_role4.json"
    calibration_path = destination / "calibration_role5.json"
    atomic_json_dump(
        {
            "schema": "private_p01_gru_nbm_step_training.v1",
            "subject": SUBJECT,
            "fold": fold,
            "seed": SEED,
            "fit_role": 4,
            "unique_raw_points": scaler_points,
            "scaler": scaler.as_dict(),
        },
        scaler_path,
    )
    atomic_json_dump(
        {
            "schema": "private_p01_gru_nbm_step_training.v1",
            "subject": SUBJECT,
            "fold": fold,
            "seed": SEED,
            "source_role": 5,
            **calibration,
        },
        calibration_path,
    )

    train_x = worker.scheme_c_features(
        nbm,
        scaler,
        sigma,
        raw_base.raw_windows(dataset, role67),
        device,
        args.tcn_batch_size,
    )
    validation_x = worker.scheme_c_features(
        nbm,
        scaler,
        sigma,
        raw_base.raw_windows(dataset, role23),
        device,
        args.tcn_batch_size,
    )
    tcn, tcn_training = worker.train_tcn(
        train_x,
        role67.label,
        validation_x,
        role23.label,
        destination,
        device,
        SEED,
        args.tcn_batch_size,
        0,
        args.tcn_max_epochs,
        args.tcn_patience,
    )
    validation_true, validation_probability = worker.predict(
        tcn,
        validation_x,
        role23.label,
        device,
        args.tcn_batch_size,
    )
    threshold, validation_metrics = raw_base.choose_threshold(
        validation_true, validation_probability
    )
    write_csv(destination / "nbm_history.csv", nbm_training["history"])
    write_csv(destination / "tcn_history.csv", tcn_training["history"])
    frozen = {
        "schema": "private_p01_gru_nbm_step_training.v1",
        "status": "frozen_before_test",
        "subject": SUBJECT,
        "fold": fold,
        "seed": SEED,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "nbm_training": {
            key: value for key, value in nbm_training.items() if key != "history"
        },
        "tcn_training": {
            key: value for key, value in tcn_training.items() if key != "history"
        },
        "role_counts": {
            str(role): int(np.sum(rows.role == role)) for role in raw_base.ROLES
        },
        "artifact_sha256": {
            "nbm": sha256_file(
                destination / "checkpoints" / worker.NBM_CHECKPOINT_NAME
            ),
            "tcn": sha256_file(destination / "checkpoints" / "tcn.pt"),
            "scaler": sha256_file(scaler_path),
            "calibration": sha256_file(calibration_path),
        },
    }
    atomic_json_dump(frozen, destination / "FROZEN_TRAIN.json")
    print(
        f"FROZEN fold={fold} nbm_best_step={nbm_training['best_step']} "
        f"nbm_val={nbm_training['best_validation_smooth_l1']:.7f} "
        f"tcn_best_epoch={tcn_training['best_epoch']} threshold={threshold:.2f}",
        flush=True,
    )
    return frozen


def evaluate_fold(
    args: argparse.Namespace,
    dataset: Any,
    fold: int,
    device: torch.device,
    allocation_groups: dict[tuple[int, str], str],
) -> dict[str, Any]:
    destination = args.output_dir / f"fold_{fold}"
    frozen = json.loads((destination / "FROZEN_TRAIN.json").read_text(encoding="utf-8"))
    rows = raw_base.load_subject_rows(args.data_dir, dataset, SUBJECT, fold)
    test_rows = rows.take_role(0, 1)
    scaler = load_scaler(destination / "scaler_role4.json")
    calibration = json.loads(
        (destination / "calibration_role5.json").read_text(encoding="utf-8")
    )
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    nbm_payload = torch.load(
        destination / "checkpoints" / worker.NBM_CHECKPOINT_NAME,
        map_location=device,
        weights_only=False,
    )
    nbm = worker.build_nbm_from_checkpoint(nbm_payload, device)
    test_x = worker.scheme_c_features(
        nbm,
        scaler,
        sigma,
        raw_base.raw_windows(dataset, test_rows),
        device,
        args.tcn_batch_size,
    )
    tcn = RepresentationTCNM(worker.TCN_INPUT_CHANNELS).to(device)
    tcn_payload = torch.load(
        destination / "checkpoints" / "tcn.pt",
        map_location=device,
        weights_only=False,
    )
    tcn.load_state_dict(tcn_payload["model_state"], strict=True)
    y_true, probability = worker.predict(
        tcn, test_x, test_rows.label, device, args.tcn_batch_size
    )
    threshold = float(frozen["threshold"])
    y_pred = (probability >= threshold).astype(np.int8)
    window_metrics = worker.binary_metrics(y_true, probability, threshold)
    prediction_rows: list[dict[str, Any]] = []
    for index in range(len(test_rows)):
        prediction_rows.append(
            {
                "subject_id": SUBJECT,
                "fold": fold,
                "seed": SEED,
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
        )
    write_csv(destination / "test_predictions.csv", prediction_rows)
    latest_metrics, event_audit = latest_event.evaluate_run(
        dataset,
        SUBJECT,
        fold,
        SEED,
        [{key: str(value) for key, value in row.items()} for row in prediction_rows],
        allocation_groups,
    )
    write_csv(destination / "event_audit.csv", event_audit)
    result = {
        "subject": SUBJECT,
        "fold": fold,
        "seed": SEED,
        "threshold": threshold,
        "accuracy": window_metrics["accuracy"],
        "sensitivity": window_metrics["sensitivity"],
        "precision": window_metrics["precision"],
        "specificity": window_metrics["specificity"],
        "f1": window_metrics["f1"],
        "ap": latest_metrics["ap"],
        "event_sensitivity": latest_metrics["event_sensitivity"],
        "false_alarms_per_hour": latest_metrics["false_alarms_per_hour"],
        "tn": window_metrics["tn"],
        "fp": window_metrics["fp"],
        "fn": window_metrics["fn"],
        "tp": window_metrics["tp"],
        "false_alarm_events": latest_metrics["false_alarm_events"],
        "evaluated_nonfog_hours": latest_metrics["evaluated_nonfog_hours"],
        "evaluable_true_events": latest_metrics["evaluable_true_events"],
        "detected_true_events": latest_metrics["detected_true_events"],
    }
    atomic_json_dump(result, destination / "test_metrics.json")
    print(
        f"TEST fold={fold} AP={result['ap']:.6f} sens={result['sensitivity']:.6f} "
        f"spec={result['specificity']:.6f} event_sens={result['event_sensitivity']:.6f} "
        f"FA/h={result['false_alarms_per_hour']:.6f}",
        flush=True,
    )
    return result


def plot_histories(output_dir: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=False)
    for fold in FOLDS:
        nbm_rows = []
        with (output_dir / f"fold_{fold}" / "nbm_history.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            nbm_rows = list(csv.DictReader(handle))
        steps = np.asarray([int(row["update_step"]) for row in nbm_rows])
        val = np.asarray([float(row["validation_smooth_l1"]) for row in nbm_rows])
        axes[0].plot(steps, val, linewidth=1.2, label=f"fold {fold}")
        train_steps = [
            int(row["update_step"])
            for row in nbm_rows
            if row["mean_train_smooth_l1_since_validation"] not in ("", "None")
        ]
        train_loss = [
            float(row["mean_train_smooth_l1_since_validation"])
            for row in nbm_rows
            if row["mean_train_smooth_l1_since_validation"] not in ("", "None")
        ]
        axes[1].plot(train_steps, train_loss, linewidth=1.2, label=f"fold {fold}")
    axes[0].set_title("Held-out role-5 clean Non-FoG validation loss", loc="left")
    axes[0].set_ylabel("SmoothL1")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_title("Role-4 clean Non-FoG training loss", loc="left")
    axes[1].set_ylabel("SmoothL1")
    axes[1].set_xlabel("Optimizer update step")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"{SUBJECT} GRU-NBM step-based clean training",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = output_dir / f"{SUBJECT.lower()}_gru_nbm_step_training_losses"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    global SUBJECT
    args = parse_args()
    SUBJECT = str(args.subject).strip().upper()
    if not SUBJECT.startswith("P") or not SUBJECT[1:].isdigit():
        raise ValueError("subject must use the Pxx form, for example P02")
    args.data_dir = args.data_dir.resolve()
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "outputs"
            / f"private_{SUBJECT}_gru_nbm_step5000_val50_pat20_lr3e4_seed0"
        )
    args.output_dir = args.output_dir.resolve()
    if args.nbm_batch_size not in (16, 32):
        raise ValueError("NBM batch size must be 16 or 32")
    if min(
        args.maximum_updates,
        args.validation_frequency,
        args.validation_patience,
        args.tcn_batch_size,
        args.tcn_max_epochs,
        args.tcn_patience,
        args.dpi,
    ) <= 0:
        raise ValueError("all integer settings must be positive")
    if args.maximum_updates % args.validation_frequency != 0:
        raise ValueError("maximum updates must be divisible by validation frequency")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DaphnetDataset.load(args.data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 30:
        raise AssertionError("expected processed_NBM_Exp 64-Hz/30-channel dataset")
    device = worker.resolve_device(args.device)

    frozen_rows = [train_fold(args, dataset, fold, device) for fold in FOLDS]
    barrier = {
        "schema": "private_subject_gru_nbm_step_training_barrier.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": SUBJECT,
        "seed": SEED,
        "folds": list(FOLDS),
        "status": "all_three_folds_frozen_before_test",
        "jobs": {
            f"fold_{row['fold']}": {
                "threshold": row["threshold"],
                "artifact_sha256": row["artifact_sha256"],
                "frozen_sha256": sha256_file(
                    args.output_dir / f"fold_{row['fold']}" / "FROZEN_TRAIN.json"
                ),
            }
            for row in frozen_rows
        },
    }
    atomic_json_dump(barrier, args.output_dir / "TRAINING_BARRIER.json")

    allocation_groups = latest_event.load_allocation_groups(
        args.data_dir / "nbm_window_manifest.csv"
    )
    fold_metrics = [
        evaluate_fold(args, dataset, fold, device, allocation_groups) for fold in FOLDS
    ]
    write_csv(args.output_dir / "fold_test_metrics.csv", fold_metrics)
    metric_names = (
        "accuracy",
        "sensitivity",
        "precision",
        "specificity",
        "f1",
        "ap",
        "event_sensitivity",
        "false_alarms_per_hour",
    )
    summary: dict[str, Any] = {
        "subject": SUBJECT,
        "seed": SEED,
        "fold_count": len(FOLDS),
    }
    for metric in metric_names:
        values = np.asarray([row[metric] for row in fold_metrics], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_sd_across_folds"] = float(values.std(ddof=0))
    write_csv(args.output_dir / "subject_summary.csv", [summary])
    plot_histories(args.output_dir, args.dpi)
    audit = {
        "schema": "private_subject_gru_nbm_step_training_experiment.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "local single-subject training-strategy experiment",
        "subject": SUBJECT,
        "seed": SEED,
        "folds": list(FOLDS),
        "fixed_architecture": worker.architecture_config(),
        "nbm_training": {
            "clean_input_equals_target": True,
            "gaussian_noise": False,
            "time_mask": False,
            "dropout_modules": 0,
            "latent_dimension": 16,
            "decoder_input": "128-step all-zero sequence",
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW",
            "batch_size": args.nbm_batch_size,
            "maximum_updates": args.maximum_updates,
            "validation_frequency_updates": args.validation_frequency,
            "early_stopping_patience_validations": args.validation_patience,
            "patience_equivalent_updates": (
                args.validation_frequency * args.validation_patience
            ),
            "initial_learning_rate": args.nbm_learning_rate,
            "weight_decay": args.nbm_weight_decay,
            "gradient_clip_global_norm": 1.0,
            "checkpoint_monitor": "minimum held-out role-5 clean Non-FoG SmoothL1",
        },
        "tcn_training": {
            "unchanged": True,
            "batch_size": args.tcn_batch_size,
            "maximum_epochs": args.tcn_max_epochs,
            "patience": args.tcn_patience,
            "checkpoint_monitor": "maximum roles-2/3 AP",
        },
        "test_event_metric": (
            "latest 1-s interval-gap merge; isolated Non-FoG positive retained; "
            "merge within record across allocation groups"
        ),
        "test_started_after_training_barrier": True,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "output_sha256": {},
    }
    for path in args.output_dir.rglob("*"):
        if path.is_file() and path.name != "audit.json":
            audit["output_sha256"][str(path.relative_to(args.output_dir))] = sha256_file(path)
    atomic_json_dump(audit, args.output_dir / "audit.json")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
