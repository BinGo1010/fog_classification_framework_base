#!/usr/bin/env python
"""Run the complete 5-NBM x 4-history Daphnet three-IMU LOSO suite.

Defaults implement the core protocol:

- complete ankle + thigh + trunk accelerometers (9 channels);
- S04 and S10 removed before scaling/windowing;
- Persistence, Linear-AR, GRU, TCN, and Transformer NBMs;
- every NBM returns ``(mu, sigma)``;
- residual histories of 0.5, 1, 2, and 4 seconds;
- one fixed downstream TCN architecture;
- atomic ``last``/``best`` checkpoints and epoch-boundary resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Required by deterministic CUDA matrix multiplications. This must be set
# before importing torch so the same entry point works on a fresh GPU server.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "daphnet_3imu_nbm_5x4_loso_seed42"
)
IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/__init__.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/resume.py",
)
EXPECTED_CHANNEL_NAMES = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)
EXPECTED_LOSO_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset, SequenceWindowDataset, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.histories import (
    HistoryPlan,
    history_block_count,
    make_common_history_plan,
    make_history_input,
)
from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.nbm import (
    NBM_NAMES,
    NormalBehaviourModel,
    build_nbm,
    canonical_nbm_name,
    gaussian_nll_sigma,
    parameter_count,
)
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    capture_rng_state,
    dataset_fingerprint,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_checkpoint,
    validate_done,
)
from run_cnbr_fog_loso import (
    deterministic_subsample,
    event_metrics,
    parse_folds,
    parse_subject_list,
    write_predictions_csv,
)


SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
    "specificity",
    "precision",
    "mcc",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
]


def implementation_manifest() -> dict[str, Any]:
    """Fingerprint every source file that defines the executable protocol."""

    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {
        "sha256": canonical_fingerprint(files),
        "files": files,
    }


def environment_payload(device: torch.device) -> dict[str, Any]:
    """Capture the runtime used by the latest invocation for provenance."""

    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "selected_device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": [sys.executable, *sys.argv],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daphnet three-IMU 5x4 NBM residual LOSO suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help=(
            "Execute exactly one fold while keeping --folds as the shared "
            "scientific protocol. Used by the multi-GPU scheduler."
        ),
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Validate the shared protocol and rebuild root summaries without "
            "training. Used after parallel fold workers finish."
        ),
    )
    parser.add_argument("--nbms", default=",".join(NBM_NAMES))
    parser.add_argument("--history-seconds", default="0.5,1,2,4")
    parser.add_argument("--exclude-subjects", default="S04,S10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.25)
    parser.add_argument("--normal-guard-seconds", type=float, default=0.5)
    parser.add_argument("--fog-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--flatline-seconds", type=float, default=1.0)
    parser.add_argument("--zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--robust-clip", type=float, default=12.0)
    parser.add_argument("--residual-clip", type=float, default=12.0)

    parser.add_argument("--nbm-hidden", type=int, default=48)
    parser.add_argument("--nbm-dropout", type=float, default=0.1)
    parser.add_argument("--linear-ar-seconds", type=float, default=0.5)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ffn", type=int, default=128)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)

    parser.add_argument("--normal-epochs", type=int, default=8)
    parser.add_argument("--normal-patience", type=int, default=3)
    parser.add_argument("--normal-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-normal-windows", type=int, default=30000)
    parser.add_argument("--max-classifier-windows", type=int, default=0)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cache-residuals", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )

    # Controlled interruption hooks are useful for proving resume semantics and
    # are inert in normal runs.
    parser.add_argument("--stop-after-completed-tasks", type=int, default=0)
    parser.add_argument("--debug-interrupt-nbm-after-epoch", type=int, default=0)
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch", type=int, default=0
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if not args.cache_residuals:
        raise ValueError(
            "The core suite requires residual-cache saving for exact recovery "
            "and audit; omit --no-cache-residuals"
        )
    positive_integers = {
        "normal_epochs": args.normal_epochs,
        "normal_patience": args.normal_patience,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "nbm_hidden": args.nbm_hidden,
        "classifier_hidden": args.classifier_hidden,
        "gru_layers": args.gru_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ffn": args.transformer_ffn,
    }
    invalid = [name for name, value in positive_integers.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These integer options must be positive: {invalid}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_normal_windows < 0 or args.max_classifier_windows < 0:
        raise ValueError("Window caps must be non-negative (zero means unlimited)")
    positive_floats = {
        "context_seconds": args.context_seconds,
        "horizon_seconds": args.horizon_seconds,
        "stride_seconds": args.stride_seconds,
        "flatline_seconds": args.flatline_seconds,
        "robust_clip": args.robust_clip,
        "residual_clip": args.residual_clip,
        "linear_ar_seconds": args.linear_ar_seconds,
        "normal_lr": args.normal_lr,
        "classifier_lr": args.classifier_lr,
    }
    invalid = [
        name
        for name, value in positive_floats.items()
        if not math.isfinite(float(value)) or float(value) <= 0
    ]
    if invalid:
        raise ValueError(f"These numeric options must be finite and positive: {invalid}")
    if args.normal_guard_seconds < 0:
        raise ValueError("--normal-guard-seconds must be non-negative")
    if args.weight_decay < 0 or args.zero_tolerance < 0:
        raise ValueError("--weight-decay and --zero-tolerance must be non-negative")
    if not 0.0 < args.fog_fraction_threshold <= 1.0:
        raise ValueError("--fog-fraction-threshold must be in (0, 1]")
    if not 0.0 <= args.nbm_dropout < 1.0:
        raise ValueError("--nbm-dropout must be in [0, 1)")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0, 1)")


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {spec}")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def parse_nbms(spec: str) -> list[str]:
    result: list[str] = []
    for value in str(spec).split(","):
        if not value.strip():
            continue
        canonical = canonical_nbm_name(value)
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("At least one NBM is required")
    return result


def parse_histories(
    spec: str,
    sampling_rate_hz: int,
    horizon_samples: int,
    stride_samples: int,
) -> list[tuple[str, float, int]]:
    durations: list[float] = []
    for value in str(spec).split(","):
        if not value.strip():
            continue
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid history duration {value!r}")
        if duration not in durations:
            durations.append(duration)
    variants: list[tuple[str, float, int]] = []
    for duration in sorted(durations):
        samples = int(round(duration * sampling_rate_hz))
        history_block_count(samples, horizon_samples, stride_samples)
        resolved = samples / float(sampling_rate_hz)
        label = f"{resolved:g}".replace(".", "p")
        variants.append((f"residual_h{label}s", resolved, samples))
    if not variants:
        raise ValueError("At least one residual history is required")
    return variants


def make_sequence_loader(
    dataset: DaphnetDataset,
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


def atomic_csv_write(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_or_validate_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(arrays):
                raise ValueError(f"Saved array keys differ in {path}")
            for key, expected in arrays.items():
                if not np.array_equal(np.asarray(payload[key]), np.asarray(expected)):
                    raise ValueError(f"Saved array mismatch in {path}: {key}")
        return
    atomic_npz_save(path, **arrays)


def save_or_validate_json(path: Path, payload: dict) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(f"Saved JSON is incompatible: {path}")
        return
    atomic_json_dump(payload, path)


def checkpoint_base(
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    upstream_nbm_sha256: str | None = None,
) -> dict:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
    }
    if upstream_nbm_sha256 is not None:
        payload["upstream_nbm_sha256"] = upstream_nbm_sha256
    return payload


def normal_epoch(
    model: NormalBehaviourModel,
    loader: DataLoader,
    context_samples: int,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
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
            with torch.amp.autocast(
                device.type, enabled=amp and device.type == "cuda"
            ):
                mean, sigma = model(context)
                loss = gaussian_nll_sigma(target, mean, sigma)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(sequence.shape[0])
        total_loss += float(loss.detach()) * batch
        total_n += batch
    return total_loss / max(total_n, 1)


def build_model(
    args: argparse.Namespace,
    name: str,
    in_channels: int,
    horizon_samples: int,
    context_samples: int,
    sampling_rate_hz: int,
) -> NormalBehaviourModel:
    return build_nbm(
        name,
        in_channels,
        horizon_samples,
        hidden_channels=args.nbm_hidden,
        dropout=args.nbm_dropout,
        linear_ar_order=int(round(args.linear_ar_seconds * sampling_rate_hz)),
        gru_layers=args.gru_layers,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ffn=args.transformer_ffn,
        max_context_samples=context_samples,
    )


def train_nbm_resumable(
    args: argparse.Namespace,
    nbm_name: str,
    nbm_root: Path,
    protocol_fingerprint: str,
    fold_seed: int,
    dataset: DaphnetDataset,
    windows: WindowTable,
    normal_train_indices: np.ndarray,
    normal_val_indices: np.ndarray,
    scaler,
    context_samples: int,
    horizon_samples: int,
    device: torch.device,
) -> tuple[NormalBehaviourModel, dict, str]:
    task_id = f"{nbm_root.parent.name}/{nbm_name}/nbm"
    stage_root = nbm_root / "nbm"
    stage_root.mkdir(parents=True, exist_ok=True)
    best_path = stage_root / "best.pt"
    last_path = stage_root / "last.pt"
    training_path = stage_root / "training.json"
    done_path = stage_root / "DONE.json"

    set_seed(fold_seed, args.deterministic)
    model = build_model(
        args,
        nbm_name,
        dataset.n_channels,
        horizon_samples,
        context_samples,
        dataset.sampling_rate_hz,
    ).to(device)
    completed = validate_done(
        done_path,
        stage="nbm",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
    )
    if completed is not None:
        payload = torch.load(best_path, map_location=device, weights_only=False)
        validate_checkpoint(
            payload,
            stage="nbm",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
        )
        model.load_state_dict(payload["model_state"])
        with training_path.open("r", encoding="utf-8") as handle:
            training = json.load(handle)
        return model, training, sha256_file(best_path)

    pin = device.type == "cuda"
    train_loader = make_sequence_loader(
        dataset,
        windows,
        normal_train_indices,
        scaler,
        args.batch_size,
        True,
        args.num_workers,
        pin,
    )
    val_loader = make_sequence_loader(
        dataset,
        windows,
        normal_val_indices,
        scaler,
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
    start_epoch = 0
    best_epoch = 0
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    elapsed_before = 0.0
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        validate_checkpoint(
            payload,
            stage="nbm",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
        )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        grad_scaler.load_state_dict(payload["grad_scaler_state"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_loss = float(payload["best_loss"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        restore_rng_state(payload["rng_state"])
        print(
            f"    [{nbm_name}] resume NBM at epoch {start_epoch + 1}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.normal_epochs + 1):
        if bad_epochs >= args.normal_patience:
            break
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
        history.append(
            {"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss}
        )
        improved = val_loss < best_loss - 1e-5
        if improved:
            best_epoch = epoch
            best_loss = val_loss
            bad_epochs = 0
            atomic_torch_save(
                {
                    **checkpoint_base(
                        stage="nbm",
                        protocol_fingerprint=protocol_fingerprint,
                        task_id=task_id,
                    ),
                    "model_name": nbm_name,
                    "seed": int(fold_seed),
                    "model_config": model.model_config(),
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_nll": best_loss,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch_save(
            {
                **checkpoint_base(
                    stage="nbm",
                    protocol_fingerprint=protocol_fingerprint,
                    task_id=task_id,
                ),
                "model_name": nbm_name,
                "seed": int(fold_seed),
                "model_config": model.model_config(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_sec": elapsed,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
        print(
            f"    [{nbm_name}] epoch={epoch:02d} train_nll={train_loss:.5f} "
            f"val_nll={val_loss:.5f}{' *' if improved else ''}",
            flush=True,
        )
        interrupt_marker = stage_root / ".debug_interrupted_once"
        if (
            args.debug_interrupt_nbm_after_epoch > 0
            and epoch >= args.debug_interrupt_nbm_after_epoch
            and not interrupt_marker.exists()
        ):
            atomic_json_dump({"interrupted_after_epoch": epoch}, interrupt_marker)
            raise RuntimeError("Intentional NBM interruption after checkpoint")

    if not best_path.exists():
        raise RuntimeError(f"NBM did not produce a best checkpoint: {task_id}")
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    validate_checkpoint(
        best_payload,
        stage="nbm",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
    )
    model.load_state_dict(best_payload["model_state"])
    training = {
        "model_name": nbm_name,
        "seed": int(fold_seed),
        "model_config": model.model_config(),
        "parameter_count": parameter_count(model),
        "best_epoch": int(best_payload["best_epoch"]),
        "best_val_nll": float(best_payload["best_val_nll"]),
        "train_windows": int(len(normal_train_indices)),
        "validation_windows": int(len(normal_val_indices)),
        "epochs_completed": int(history[-1]["epoch"]),
        "elapsed_sec": float(
            elapsed_before + time.perf_counter() - started
        ),
        "history": history,
    }
    atomic_json_dump(training, training_path)
    atomic_json_dump(
        done_payload(
            stage="nbm",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            relative_to=stage_root,
            artifacts={
                "best": best_path.resolve(),
                "last": last_path.resolve(),
                "training": training_path.resolve(),
            },
        ),
        done_path,
    )
    return model, training, sha256_file(best_path)


@torch.no_grad()
def extract_residual_blocks(
    args: argparse.Namespace,
    model: NormalBehaviourModel,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler,
    context_samples: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    loader = make_sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    model.eval()
    residuals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    squared_error = 0.0
    absolute_error = 0.0
    sigma_sum = 0.0
    n_values = 0
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :context_samples]
        target = sequence[:, :, context_samples:]
        with torch.amp.autocast(
            device.type, enabled=args.amp and device.type == "cuda"
        ):
            mean, sigma = model(context)
            residual = (target - mean) / sigma
        residual = residual.clamp(-args.residual_clip, args.residual_clip)
        error = (target - mean).float()
        squared_error += float(error.square().sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        sigma_sum += float(sigma.float().sum().cpu())
        n_values += int(error.numel())
        residuals.append(residual.float().cpu().numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())
    features = {
        "residual": np.ascontiguousarray(
            np.concatenate(residuals).astype(np.float32, copy=False)
        ),
        "y": np.concatenate(labels).astype(np.int8, copy=False),
        "window_index": np.concatenate(window_indices).astype(
            np.int64, copy=False
        ),
    }
    diagnostics = {
        "windows": int(len(features["y"])),
        "class_counts": np.bincount(
            features["y"], minlength=2
        ).astype(int).tolist(),
        "forecast_rmse": math.sqrt(squared_error / max(n_values, 1)),
        "forecast_mae": absolute_error / max(n_values, 1),
        "mean_sigma": sigma_sum / max(n_values, 1),
        "residual_abs_mean": float(
            np.abs(features["residual"].astype(np.float64)).mean()
        ),
        "residual_rms": float(
            np.sqrt(
                np.mean(features["residual"].astype(np.float64) ** 2)
            )
        ),
    }
    return features, diagnostics


def load_or_extract_residual_cache(
    args: argparse.Namespace,
    nbm_root: Path,
    protocol_fingerprint: str,
    nbm_name: str,
    nbm_sha256: str,
    model: NormalBehaviourModel,
    dataset: DaphnetDataset,
    windows: WindowTable,
    split_indices: dict[str, np.ndarray],
    scaler,
    context_samples: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    task_id = f"{nbm_root.parent.name}/{nbm_name}/residual_cache"
    cache_path = nbm_root / "residual_cache.npz"
    diagnostics_path = nbm_root / "residual_diagnostics.json"
    done_path = nbm_root / "RESIDUAL_CACHE_DONE.json"
    if args.cache_residuals:
        complete = validate_done(
            done_path,
            stage="residual_cache",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=nbm_sha256,
        )
        if complete is not None:
            with np.load(cache_path, allow_pickle=False) as payload:
                features = {
                    split: {
                        "residual": np.asarray(
                            payload[f"{split}_residual"], dtype=np.float32
                        ),
                        "y": np.asarray(payload[f"{split}_y"], dtype=np.int8),
                        "window_index": np.asarray(
                            payload[f"{split}_window_index"], dtype=np.int64
                        ),
                    }
                    for split in ("train", "validation", "test")
                }
            with diagnostics_path.open("r", encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            return features, diagnostics

    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        features[split], diagnostics[split] = extract_residual_blocks(
            args,
            model,
            dataset,
            windows,
            split_indices[split],
            scaler,
            context_samples,
            device,
        )
    atomic_json_dump(diagnostics, diagnostics_path)
    if args.cache_residuals:
        atomic_npz_save(
            cache_path,
            **{
                f"{split}_{key}": features[split][key]
                for split in ("train", "validation", "test")
                for key in ("residual", "y", "window_index")
            },
        )
        atomic_json_dump(
            done_payload(
                stage="residual_cache",
                protocol_fingerprint=protocol_fingerprint,
                task_id=task_id,
                upstream_sha256=nbm_sha256,
                relative_to=nbm_root,
                artifacts={
                    "cache": cache_path.resolve(),
                    "diagnostics": diagnostics_path.resolve(),
                },
            ),
            done_path,
        )
    return features, diagnostics


def classifier_epoch(
    model: ResidualTCNClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type, enabled=amp and device.type == "cuda"
            ):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(y.numel())
        total_loss += float(loss.detach()) * batch
        total_n += batch
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probabilities.append(
            torch.sigmoid(logits.detach()).float().cpu().numpy()
        )
    return (
        total_loss / max(total_n, 1),
        np.concatenate(truths),
        np.concatenate(probabilities),
    )


def add_requested_metrics(metrics: dict) -> dict:
    tn, fp, fn, tp = [int(metrics[key]) for key in ("tn", "fp", "fn", "tp")]
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    metrics["macro_f1"] = 0.5 * (f1_nonfog + f1_fog)
    metrics["roc_auc"] = metrics.get("auroc")
    metrics["pr_auc"] = metrics.get("auprc")
    metrics["fog_recall"] = metrics.get("sensitivity")
    metrics["fog_f1"] = f1_fog
    return metrics


def train_classifier_resumable(
    args: argparse.Namespace,
    task_root: Path,
    task_id: str,
    protocol_fingerprint: str,
    upstream_nbm_sha256: str,
    classifier_seed: int,
    input_name: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    dataset: DaphnetDataset,
    windows: WindowTable,
    metadata: dict,
    device: torch.device,
) -> dict:
    task_root.mkdir(parents=True, exist_ok=True)
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    metrics_path = task_root / "metrics.json"
    predictions_path = task_root / "predictions.npz"
    validation_predictions_path = task_root / "validation_predictions.npz"
    predictions_csv_path = task_root / "predictions.csv"
    done_path = task_root / "DONE.json"
    complete = validate_done(
        done_path,
        stage="classifier",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
        upstream_sha256=upstream_nbm_sha256,
    )
    if complete is not None:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    set_seed(classifier_seed, args.deterministic)
    x_train = train[input_name]
    y_train = train["y"]
    x_val = validation[input_name]
    y_val = validation["y"]
    x_test = test[input_name]
    y_test = test["y"]
    model = ResidualTCNClassifier(
        in_channels=x_train.shape[1],
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
    ).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    pos_weight_value = min(math.sqrt(counts[0] / max(counts[1], 1.0)), 6.0)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
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

    start_epoch = 0
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict] = []
    elapsed_before = 0.0
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        validate_checkpoint(
            payload,
            stage="classifier",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=upstream_nbm_sha256,
        )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        grad_scaler.load_state_dict(payload["grad_scaler_state"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        restore_rng_state(payload["rng_state"])
        print(
            f"      [{input_name}] resume classifier at epoch {start_epoch + 1}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
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
                "train_auprc": float(
                    average_precision_score(train_true, train_prob)
                ),
                "validation_loss": val_loss,
                "validation_auprc": score,
            }
        )
        improved = score > best_score + 1e-5
        if improved:
            best_epoch = epoch
            best_score = score
            bad_epochs = 0
            atomic_torch_save(
                {
                    **checkpoint_base(
                        stage="classifier",
                        protocol_fingerprint=protocol_fingerprint,
                        task_id=task_id,
                        upstream_nbm_sha256=upstream_nbm_sha256,
                    ),
                    "input_name": input_name,
                    "classifier_seed": int(classifier_seed),
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_auprc": best_score,
                    "classifier_config": {
                        "in_channels": int(x_train.shape[1]),
                        "hidden_channels": args.classifier_hidden,
                        "dropout": args.classifier_dropout,
                    },
                },
                best_path,
            )
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch_save(
            {
                **checkpoint_base(
                    stage="classifier",
                    protocol_fingerprint=protocol_fingerprint,
                    task_id=task_id,
                    upstream_nbm_sha256=upstream_nbm_sha256,
                ),
                "input_name": input_name,
                "classifier_seed": int(classifier_seed),
                "classifier_config": {
                    "in_channels": int(x_train.shape[1]),
                    "hidden_channels": args.classifier_hidden,
                    "dropout": args.classifier_dropout,
                },
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_sec": elapsed,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
        print(
            f"      [{input_name}] epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} val_auprc={score:.5f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        interrupt_marker = task_root / ".debug_interrupted_once"
        if (
            args.debug_interrupt_classifier_after_epoch > 0
            and epoch >= args.debug_interrupt_classifier_after_epoch
            and not interrupt_marker.exists()
        ):
            atomic_json_dump({"interrupted_after_epoch": epoch}, interrupt_marker)
            raise RuntimeError(
                "Intentional classifier interruption after checkpoint"
            )

    if not best_path.exists():
        raise RuntimeError(f"Classifier did not produce a best checkpoint: {task_id}")
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    validate_checkpoint(
        best_payload,
        stage="classifier",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
        upstream_sha256=upstream_nbm_sha256,
    )
    model.load_state_dict(best_payload["model_state"])
    with torch.no_grad():
        _, val_true, val_prob = classifier_epoch(
            model, val_loader, criterion, device, args.amp
        )
        _, test_true, test_prob = classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, validation_metrics = choose_threshold(val_true, val_prob)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (
        np.asarray(test_prob, dtype=np.float64) >= float(threshold)
    ).astype(np.int8)
    test_metrics.update(
        event_metrics(
            dataset,
            windows,
            test["window_index"],
            test_pred,
        )
    )
    test_metrics.update(
        {
            **metadata,
            "input": input_name,
            "classifier_seed": classifier_seed,
            "best_epoch": int(best_payload["best_epoch"]),
            "best_validation_auprc": float(
                best_payload["best_validation_auprc"]
            ),
            "validation": validation_metrics,
            "train_counts": counts.astype(int).tolist(),
            "pos_weight": float(pos_weight_value),
            "elapsed_sec": float(
                elapsed_before + time.perf_counter() - started
            ),
            "history": history,
            "upstream_nbm_sha256": upstream_nbm_sha256,
        }
    )
    add_requested_metrics(test_metrics)
    atomic_json_dump(test_metrics, metrics_path)
    atomic_npz_save(
        predictions_path,
        window_index=test["window_index"],
        y_true=test_true,
        y_prob=np.asarray(test_prob, dtype=np.float32),
        y_pred=test_pred,
    )
    validation_pred = (
        np.asarray(val_prob, dtype=np.float64) >= float(threshold)
    ).astype(np.int8)
    atomic_npz_save(
        validation_predictions_path,
        window_index=validation["window_index"],
        y_true=val_true,
        y_prob=np.asarray(val_prob, dtype=np.float32),
        y_pred=validation_pred,
    )
    write_predictions_csv(
        predictions_csv_path,
        dataset,
        windows,
        test["window_index"],
        test_prob,
        test_pred,
    )
    atomic_json_dump(
        done_payload(
            stage="classifier",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=upstream_nbm_sha256,
            relative_to=task_root,
            artifacts={
                "best": best_path.resolve(),
                "last": last_path.resolve(),
                "metrics": metrics_path.resolve(),
                "predictions": predictions_path.resolve(),
                "validation_predictions": validation_predictions_path.resolve(),
                "predictions_csv": predictions_csv_path.resolve(),
            },
        ),
        done_path,
    )
    return test_metrics


def prediction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall_fog = tp / (tp + fn) if tp + fn else 0.0
    recall_nonfog = tn / (tn + fp) if tn + fp else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (recall_fog + recall_nonfog),
        "macro_f1": 0.5 * (f1_fog + f1_nonfog),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else None
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else None
        ),
        "fog_recall": recall_fog,
        "fog_f1": f1_fog,
        "specificity": recall_nonfog,
    }


def refresh_summaries(
    output_dir: Path,
    config: dict,
) -> None:
    fold_rows: list[dict] = []
    manifest_rows: list[dict] = []
    aggregate: dict[str, dict] = {}
    expected_folds = list(config["folds_resolved"])
    for nbm_name in config["nbms_resolved"]:
        for variant in config["history_variants"]:
            input_name = variant["input"]
            experiment_id = f"{nbm_name}__{input_name.removeprefix('residual_')}"
            group_rows: list[dict] = []
            truths: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            predictions: list[np.ndarray] = []
            completed: list[str] = []
            for subject in expected_folds:
                root = (
                    output_dir
                    / f"loso_{subject}"
                    / nbm_name
                    / input_name
                )
                metrics_path = root / "metrics.json"
                predictions_path = root / "predictions.npz"
                done_path = root / "DONE.json"
                if not (
                    metrics_path.exists()
                    and predictions_path.exists()
                    and done_path.exists()
                ):
                    continue
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                with np.load(predictions_path, allow_pickle=False) as payload:
                    truths.append(np.asarray(payload["y_true"], dtype=np.int8))
                    probabilities.append(
                        np.asarray(payload["y_prob"], dtype=np.float64)
                    )
                    predictions.append(
                        np.asarray(payload["y_pred"], dtype=np.int8)
                    )
                group_rows.append(metrics)
                fold_rows.append(metrics)
                completed.append(subject)
            if group_rows:
                aggregate[experiment_id] = {
                    "nbm": nbm_name,
                    "input": input_name,
                    "history_seconds": variant["history_seconds"],
                    "completed_folds": completed,
                    "subject_macro": aggregate_fold_metrics(
                        group_rows, CLASSIFICATION_METRICS
                    ),
                    "pooled": prediction_metrics(
                        np.concatenate(truths),
                        np.concatenate(probabilities),
                        np.concatenate(predictions),
                    ),
                }
            manifest_rows.append(
                {
                    "experiment_id": experiment_id,
                    "nbm": nbm_name,
                    "history_seconds": variant["history_seconds"],
                    "history_samples": variant["history_samples"],
                    "expected_folds": len(expected_folds),
                    "completed_folds": len(completed),
                    "status": (
                        "complete"
                        if completed == expected_folds
                        else ("partial" if completed else "pending")
                    ),
                    "completed_subjects": ",".join(completed),
                }
            )
    fold_columns = [
        "experiment_id",
        "nbm",
        "input",
        "history_seconds",
        "history_samples",
        "history_blocks",
        "test_subject",
        "val_subject",
        "classifier_seed",
        "threshold",
        "n",
        "n_normal",
        "n_fog",
        *CLASSIFICATION_METRICS,
        "tn",
        "fp",
        "fn",
        "tp",
        "best_epoch",
        "best_validation_auprc",
        "upstream_nbm_sha256",
    ]
    atomic_csv_write(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        [
            "experiment_id",
            "nbm",
            "history_seconds",
            "history_samples",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    atomic_json_dump(aggregate, output_dir / "aggregate_metrics.json")
    total_cells = len(manifest_rows) * len(expected_folds)
    completed_cells = sum(int(row["completed_folds"]) for row in manifest_rows)
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_experiments": len(manifest_rows),
            "expected_fold_cells": total_cells,
            "completed_fold_cells": completed_cells,
            "status": "complete" if completed_cells == total_cells else "partial",
        },
        output_dir / "status.json",
    )


def build_protocol(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    source_subjects: list[str],
    excluded_subjects: list[str],
    folds: list[str],
    nbms: list[str],
    histories: list[tuple[str, float, int]],
    context_samples: int,
    horizon_samples: int,
    stride_samples: int,
    guard_samples: int,
    data_sha256: str,
    windows: WindowTable,
    evaluation_indices: np.ndarray,
    device: torch.device,
) -> dict:
    protocol = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channel_names": list(dataset.channel_names),
        "n_channels": dataset.n_channels,
        "source_subjects": source_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects": dataset.subjects,
        "folds_resolved": folds,
        "nbms_resolved": nbms,
        "history_variants": [
            {
                "input": name,
                "history_seconds": seconds,
                "history_samples": samples,
                "history_blocks": history_block_count(
                    samples, horizon_samples, stride_samples
                ),
            }
            for name, seconds, samples in histories
        ],
        "context_samples": context_samples,
        "horizon_samples": horizon_samples,
        "stride_samples": stride_samples,
        "normal_guard_samples": guard_samples,
        "fog_fraction_threshold": args.fog_fraction_threshold,
        "flatline_seconds": args.flatline_seconds,
        "zero_tolerance": args.zero_tolerance,
        "robust_clip": args.robust_clip,
        "residual_clip": args.residual_clip,
        "seed": args.seed,
        "nbm_hidden": args.nbm_hidden,
        "nbm_dropout": args.nbm_dropout,
        "linear_ar_seconds": args.linear_ar_seconds,
        "gru_layers": args.gru_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ffn": args.transformer_ffn,
        "classifier_hidden": args.classifier_hidden,
        "classifier_dropout": args.classifier_dropout,
        "normal_epochs": args.normal_epochs,
        "normal_patience": args.normal_patience,
        "normal_lr": args.normal_lr,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_normal_windows": args.max_normal_windows,
        "max_classifier_windows": args.max_classifier_windows,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "cache_residuals": args.cache_residuals,
        "window_count": len(windows),
        "window_class_counts": np.bincount(
            windows.label, minlength=2
        ).astype(int).tolist(),
        "evaluation_windows": int(len(evaluation_indices)),
        "evaluation_window_class_counts": np.bincount(
            windows.label[evaluation_indices], minlength=2
        ).astype(int).tolist(),
    }
    fingerprint = canonical_fingerprint(protocol)
    return {
        **protocol,
        "protocol_fingerprint": fingerprint,
        # Runtime locations are recorded for provenance but deliberately kept
        # out of the protocol hash so a resumable run can be moved to another
        # server without changing the scientific experiment.
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "resume": args.resume,
        "num_workers": args.num_workers,
    }


def prepare_fold(
    args: argparse.Namespace,
    config: dict,
    dataset: DaphnetDataset,
    windows: WindowTable,
    test_subject: str,
    histories: list[tuple[str, float, int]],
) -> tuple[
    Path,
    str,
    list[str],
    Any,
    dict[str, np.ndarray],
    dict[str, HistoryPlan],
    np.ndarray,
    np.ndarray,
]:
    fold_root = args.output_dir / f"loso_{test_subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    subjects = dataset.subjects
    start = subjects.index(test_subject)
    val_subject = ""
    for offset in range(1, len(subjects)):
        candidate = subjects[(start + offset) % len(subjects)]
        candidate_indices = dataset.window_indices_for_subjects(
            windows, [candidate]
        )
        if np.unique(windows.label[candidate_indices]).size == 2:
            val_subject = candidate
            break
    if not val_subject:
        raise RuntimeError("No validation subject with both classes")
    train_subjects = [
        subject
        for subject in subjects
        if subject not in {test_subject, val_subject}
    ]
    scaler = dataset.fit_scaler(train_subjects, clip=args.robust_clip)
    split_indices = {
        "train": dataset.window_indices_for_subjects(windows, train_subjects),
        "validation": dataset.window_indices_for_subjects(
            windows, [val_subject]
        ),
        "test": dataset.window_indices_for_subjects(windows, [test_subject]),
    }
    normal_train = dataset.window_indices_for_subjects(
        windows, train_subjects, clean_normal_only=True
    )
    normal_validation = dataset.window_indices_for_subjects(
        windows, [val_subject], clean_normal_only=True
    )
    fold_index = subjects.index(test_subject)
    normal_train = deterministic_subsample(
        normal_train,
        args.max_normal_windows,
        args.seed + fold_index,
    )
    maximum_history = max(samples for _, _, samples in histories)
    plans = {
        split: make_common_history_plan(
            windows,
            indices,
            config["horizon_samples"],
            config["stride_samples"],
            maximum_history,
        )
        for split, indices in split_indices.items()
    }
    if min(len(plan.anchor_rows) for plan in plans.values()) == 0:
        raise RuntimeError(f"Empty history support in fold {test_subject}")
    if args.max_classifier_windows > 0:
        plan_rows = np.arange(len(plans["train"].anchor_rows), dtype=np.int64)
        plan_labels = windows.label[plans["train"].anchor_window_indices]
        selected = deterministic_subsample(
            plan_rows,
            args.max_classifier_windows,
            args.seed + 100 + fold_index,
            plan_labels,
        )
        plans["train"] = plans["train"].take(selected)

    fold_config = {
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "excluded_subjects": config["excluded_subjects"],
        "channel_names": list(dataset.channel_names),
        "scaler": scaler.as_dict(),
        "source_window_counts": {
            split: int(len(indices)) for split, indices in split_indices.items()
        },
        "history_anchor_counts": {
            split: int(len(plan.anchor_rows)) for split, plan in plans.items()
        },
    }
    save_or_validate_json(fold_root / "fold_config.json", fold_config)
    save_or_validate_json(fold_root / "scaler.json", scaler.as_dict())
    save_or_validate_npz(
        fold_root / "split_indices.npz",
        train_window_index=split_indices["train"],
        validation_window_index=split_indices["validation"],
        test_window_index=split_indices["test"],
        normal_train_window_index=normal_train,
        normal_validation_window_index=normal_validation,
    )
    save_or_validate_npz(
        fold_root / "history_support.npz",
        **{
            f"{split}_anchor_window_index": plan.anchor_window_indices
            for split, plan in plans.items()
        },
        **{
            f"{split}_history_window_index": split_indices[split][
                plan.max_chain_rows
            ]
            for split, plan in plans.items()
        },
    )
    return (
        fold_root,
        val_subject,
        train_subjects,
        scaler,
        split_indices,
        plans,
        normal_train,
        normal_validation,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    worker_mode = bool(str(args.worker_fold).strip())
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if set(parse_subject_list(args.exclude_subjects)) != {"S04", "S10"}:
        raise ValueError("The core suite requires exactly --exclude-subjects S04,S10")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new output directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed, args.deterministic)

    data_sha256 = dataset_fingerprint(args.data_dir)
    dataset = DaphnetDataset.load(
        args.data_dir,
        flatline_seconds=args.flatline_seconds,
        zero_tolerance=args.zero_tolerance,
    )
    if dataset.n_channels != 9:
        raise ValueError(
            f"Core three-IMU suite requires 9 channels, got {dataset.n_channels}"
        )
    if tuple(dataset.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError(
            "Expected ordered ankle/thigh/trunk three-axis channels, got "
            f"{dataset.channel_names}"
        )
    source_subjects = list(dataset.subjects)
    excluded_subjects = parse_subject_list(args.exclude_subjects)
    excluded = set(excluded_subjects)
    dataset = DaphnetDataset(
        root=dataset.root,
        records=[
            record
            for record in dataset.records
            if record.subject_id not in excluded
        ],
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
    )
    if tuple(dataset.subjects) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            "Core suite requires the eight post-exclusion subjects "
            f"{EXPECTED_LOSO_SUBJECTS}, got {tuple(dataset.subjects)}"
        )
    fs = dataset.sampling_rate_hz
    if fs != 64:
        raise ValueError(f"Core Daphnet suite requires 64 Hz data, got {fs} Hz")
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
    nbms = parse_nbms(args.nbms)
    histories = parse_histories(
        args.history_seconds, fs, horizon_samples, stride_samples
    )
    folds = parse_folds(args.folds, dataset.subjects)
    if worker_mode and tuple(folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            "Parallel worker mode requires --folds all so every worker shares "
            "the canonical eight-fold protocol"
        )
    execution_folds = list(folds)
    if str(args.worker_fold).strip():
        worker_folds = parse_folds(args.worker_fold, dataset.subjects)
        if len(worker_folds) != 1:
            raise ValueError("--worker-fold must resolve to exactly one subject")
        if worker_folds[0] not in folds:
            raise ValueError(
                f"Worker fold {worker_folds[0]} is outside configured folds {folds}"
            )
        execution_folds = worker_folds
    global_plan = make_common_history_plan(
        windows,
        np.arange(len(windows), dtype=np.int64),
        horizon_samples,
        stride_samples,
        max(samples for _, _, samples in histories),
    )
    fold_records = set(
        dataset.subject_record_indices(folds).astype(int).tolist()
    )
    evaluation_indices = global_plan.anchor_window_indices[
        np.fromiter(
            (
                int(record_index) in fold_records
                for record_index in windows.record_index[
                    global_plan.anchor_window_indices
                ]
            ),
            dtype=bool,
            count=len(global_plan.anchor_window_indices),
        )
    ]
    config = build_protocol(
        args,
        dataset,
        source_subjects,
        excluded_subjects,
        folds,
        nbms,
        histories,
        context_samples,
        horizon_samples,
        stride_samples,
        guard_samples,
        data_sha256,
        windows,
        evaluation_indices,
        device,
    )
    config_path = args.output_dir / "config.json"
    if worker_mode and not config_path.exists():
        raise RuntimeError(
            "Parallel worker bootstrap is missing config.json; run the same "
            "protocol once with --finalize-only before launching workers"
        )
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_fingerprint") != config["protocol_fingerprint"]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new output directory"
            )
    # Refresh only runtime/provenance fields on a compatible resume. The
    # scientific portion is protected by protocol_fingerprint and duplicated
    # in immutable run_manifest.json.
    if not worker_mode:
        atomic_json_dump(config, config_path)
    runtime_only = {
        "data_dir",
        "output_dir",
        "device",
        "resume",
        "num_workers",
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode and not run_manifest_path.exists():
        raise RuntimeError(
            "Parallel worker bootstrap is missing run_manifest.json"
        )
    run_manifest = {
        key: value for key, value in config.items() if key not in runtime_only
    }
    if worker_mode:
        with run_manifest_path.open("r", encoding="utf-8") as handle:
            existing_run_manifest = json.load(handle)
        if existing_run_manifest != run_manifest:
            raise ValueError(
                f"Saved JSON is incompatible: {run_manifest_path}"
            )
    else:
        save_or_validate_json(run_manifest_path, run_manifest)
    current_environment = environment_payload(device)
    if worker_mode:
        current_environment.update(
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "worker_fold": execution_folds[0],
            }
        )
        atomic_json_dump(
            current_environment,
            args.output_dir
            / "worker_environments"
            / f"loso_{execution_folds[0]}.json",
        )
    else:
        atomic_json_dump(
            current_environment,
            args.output_dir / "environment.json",
        )
        refresh_summaries(args.output_dir, config)
    print(
        f"[INFO] suite={SUITE_VERSION} device={device} channels={dataset.n_channels} "
        f"subjects={dataset.subjects} windows={len(windows)} "
        f"common={config['evaluation_windows']} "
        f"counts={config['evaluation_window_class_counts']} "
        f"configured_folds={folds} execution_folds={execution_folds} "
        f"nbms={nbms} histories={[v[1] for v in histories]}",
        flush=True,
    )

    if args.finalize_only:
        with (args.output_dir / "status.json").open(
            "r", encoding="utf-8"
        ) as handle:
            status = json.load(handle)
        print("[INFO] finalize-only: root summaries refreshed", flush=True)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        return

    completed_this_run = 0
    for test_subject in execution_folds:
        fold_index = dataset.subjects.index(test_subject)
        (
            fold_root,
            val_subject,
            train_subjects,
            scaler,
            split_indices,
            plans,
            normal_train_indices,
            normal_val_indices,
        ) = prepare_fold(args, config, dataset, windows, test_subject, histories)
        print(
            f"[fold {test_subject}] train={train_subjects} val={val_subject} "
            f"anchors={{{', '.join(f'{k}:{len(v.anchor_rows)}' for k, v in plans.items())}}}",
            flush=True,
        )
        for nbm_name in nbms:
            nbm_root = fold_root / nbm_name
            nbm_root.mkdir(parents=True, exist_ok=True)
            model, normal_training, nbm_sha256 = train_nbm_resumable(
                args,
                nbm_name,
                nbm_root,
                config["protocol_fingerprint"],
                args.seed + fold_index,
                dataset,
                windows,
                normal_train_indices,
                normal_val_indices,
                scaler,
                context_samples,
                horizon_samples,
                device,
            )
            features, residual_diagnostics = load_or_extract_residual_cache(
                args,
                nbm_root,
                config["protocol_fingerprint"],
                nbm_name,
                nbm_sha256,
                model,
                dataset,
                windows,
                split_indices,
                scaler,
                context_samples,
                device,
            )
            atomic_json_dump(
                {
                    "protocol_fingerprint": config["protocol_fingerprint"],
                    "nbm": nbm_name,
                    "nbm_sha256": nbm_sha256,
                    "normal_training": normal_training,
                    "residual_diagnostics": residual_diagnostics,
                },
                nbm_root / "nbm_summary.json",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for input_name, history_seconds, history_samples in histories:
                classifier_seed = args.seed + 10000 + fold_index
                set_seed(classifier_seed, args.deterministic)
                classifier_inputs = {
                    split: make_history_input(
                        features[split],
                        plans[split],
                        input_name,
                        history_samples,
                        horizon_samples,
                        stride_samples,
                    )
                    for split in ("train", "validation", "test")
                }
                experiment_id = (
                    f"{nbm_name}__{input_name.removeprefix('residual_')}"
                )
                metrics = train_classifier_resumable(
                    args,
                    nbm_root / input_name,
                    f"{test_subject}/{nbm_name}/{input_name}",
                    config["protocol_fingerprint"],
                    nbm_sha256,
                    classifier_seed,
                    input_name,
                    classifier_inputs["train"],
                    classifier_inputs["validation"],
                    classifier_inputs["test"],
                    dataset,
                    windows,
                    {
                        "experiment_id": experiment_id,
                        "nbm": nbm_name,
                        "history_seconds": history_seconds,
                        "history_samples": history_samples,
                        "history_blocks": history_block_count(
                            history_samples,
                            horizon_samples,
                            stride_samples,
                        ),
                        "test_subject": test_subject,
                        "val_subject": val_subject,
                    },
                    device,
                )
                print(
                    f"[fold {test_subject}] {experiment_id} "
                    f"PR-AUC={metrics['pr_auc']:.4f} "
                    f"BA={metrics['balanced_accuracy']:.4f} "
                    f"FoG-F1={metrics['fog_f1']:.4f}",
                    flush=True,
                )
                del classifier_inputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                completed_this_run += 1
                if not worker_mode:
                    refresh_summaries(args.output_dir, config)
                if (
                    args.stop_after_completed_tasks > 0
                    and completed_this_run >= args.stop_after_completed_tasks
                ):
                    raise RuntimeError(
                        "Intentional stop after completed classifier tasks"
                    )
            del features
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if worker_mode:
        print(
            json.dumps(
                {
                    "suite_version": SUITE_VERSION,
                    "protocol_fingerprint": config["protocol_fingerprint"],
                    "worker_fold": execution_folds[0],
                    "classifier_cells_visited": completed_this_run,
                    "status": "worker_complete",
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    refresh_summaries(args.output_dir, config)
    with (args.output_dir / "status.json").open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
