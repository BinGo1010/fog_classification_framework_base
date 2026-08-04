#!/usr/bin/env python
"""Mean-only S01 GRU ablation with no access to the held-out R02 record.

This diagnostic keeps the GRU encoder and direct 128-step mean path used by
the Gaussian NBM, but optimizes only future-mean squared error.  It exists to
test whether Gaussian-NLL improvement is dominated by the sigma branch rather
than by learning a useful future trajectory.  It is an ablation, not the model
selection protocol: early stopping therefore uses validation clean-normal
RMSE rather than NLL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_daphnet_s01_gru_convergence as diagnostic  # noqa: E402
from cnbr_fog.data import DaphnetDataset, RobustChannelScaler, WindowTable  # noqa: E402
from cnbr_fog.nbm import GRUNBM, parameter_count  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_mean_only_ablation.v1"
MIN_DELTA_RMSE = 1e-4
MIN_EPOCHS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S01 GRU future-mean-only ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "daphnet_s01_gru_mean_only_40ep_pat6",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.max_epochs not in {30, 40}:
        raise ValueError("--max-epochs must be 30 or 40")
    if args.patience not in {5, 6}:
        raise ValueError("--patience must be 5 or 6")
    if args.batch_size <= 0 or args.hidden_channels <= 0:
        raise ValueError("Batch and hidden sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    return diagnostic.parse_int_list(args.seeds)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@torch.no_grad()
def evaluate_mean(
    model: GRUNBM,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = diagnostic.base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        0,
        device.type == "cuda",
    )
    model.eval()
    squared_sum = 0.0
    absolute_sum = 0.0
    values = 0
    horizon_squared = np.zeros(diagnostic.base.TARGET_SAMPLES, dtype=np.float64)
    channel_squared = np.zeros(dataset.n_channels, dtype=np.float64)
    window_count = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, : diagnostic.base.CONTEXT_SAMPLES]
        target = sequence[:, :, diagnostic.base.CONTEXT_SAMPLES :]
        mean, _ = model(context.float())
        error = target.float() - mean.float()
        squared = error.square().double()
        squared_sum += float(squared.sum().cpu())
        absolute_sum += float(error.abs().double().sum().cpu())
        values += int(error.numel())
        window_count += int(error.shape[0])
        array = squared.cpu().numpy()
        horizon_squared += array.sum(axis=(0, 1))
        channel_squared += array.sum(axis=(0, 2))
    if window_count != len(indices) or values <= 0:
        raise AssertionError("Mean-only evaluation support changed")
    rmse = math.sqrt(squared_sum / values)
    result = {
        "windows": window_count,
        "mse_scaled": squared_sum / values,
        "rmse_scaled": rmse,
        "mae_scaled": absolute_sum / values,
        "per_channel_rmse_scaled": np.sqrt(
            channel_squared / (window_count * diagnostic.base.TARGET_SAMPLES)
        ).tolist(),
        "per_horizon_rmse_scaled": np.sqrt(
            horizon_squared / (window_count * dataset.n_channels)
        ).tolist(),
    }
    if not all(
        math.isfinite(float(result[key]))
        for key in ("mse_scaled", "rmse_scaled", "mae_scaled")
    ):
        raise FloatingPointError("Non-finite mean-only evaluation")
    return result


def train_mean_epoch(
    model: GRUNBM,
    loader,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.train()
    squared_sum = 0.0
    values = 0
    gradient_norms: list[float] = []
    clipped = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, : diagnostic.base.CONTEXT_SAMPLES]
        target = sequence[:, :, diagnostic.base.CONTEXT_SAMPLES :]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device.type, enabled=bool(amp and device.type == "cuda")
        ):
            mean, _ = model(context)
            error = target - mean
            loss = 0.5 * error.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite mean-only loss")
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite mean-only gradient")
        clipped += int(norm > 5.0)
        gradient_norms.append(norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        squared_sum += float(error.detach().square().double().sum().cpu())
        values += int(error.numel())
    return {
        "optimization_mse_scaled": squared_sum / values,
        "optimizer_steps": len(gradient_norms),
        "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
        "max_preclip_gradient_norm": float(np.max(gradient_norms)),
        "gradient_clip_step_fraction": clipped / len(gradient_norms),
    }


def run_seed(
    args: argparse.Namespace,
    seed: int,
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    protocol_fingerprint: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    diagnostic.set_seed(seed, True)
    model = GRUNBM(
        in_channels=dataset.n_channels,
        horizon=diagnostic.base.TARGET_SAMPLES,
        hidden_channels=args.hidden_channels,
        num_layers=1,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )
    best_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    cumulative_steps = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    started = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        if epoch > MIN_EPOCHS and bad_epochs >= args.patience:
            break
        loader = diagnostic.base.sequence_loader(
            dataset,
            windows,
            train_indices,
            scaler,
            args.batch_size,
            True,
            0,
            device.type == "cuda",
            seed=seed + epoch,
        )
        optimization = train_mean_epoch(
            model, loader, optimizer, grad_scaler, device, args.amp
        )
        cumulative_steps += int(optimization["optimizer_steps"])
        train = evaluate_mean(
            model, dataset, windows, train_indices, scaler, args.batch_size, device
        )
        validation = evaluate_mean(
            model,
            dataset,
            windows,
            validation_indices,
            scaler,
            args.batch_size,
            device,
        )
        improved = validation["rmse_scaled"] < best_rmse - MIN_DELTA_RMSE
        row = {
            "epoch": epoch,
            "cumulative_optimizer_steps": cumulative_steps,
            **optimization,
            "train_eval_mse_scaled": train["mse_scaled"],
            "train_eval_rmse_scaled": train["rmse_scaled"],
            "train_eval_mae_scaled": train["mae_scaled"],
            "validation_mse_scaled": validation["mse_scaled"],
            "validation_rmse_scaled": validation["rmse_scaled"],
            "validation_mae_scaled": validation["mae_scaled"],
            "improved": improved,
        }
        history.append(row)
        if improved:
            best_rmse = float(validation["rmse_scaled"])
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_clean_normal_rmse": best_rmse,
                    "model_config": model.model_config(),
                    "model_state": model.state_dict(),
                },
                best_path,
            )
        else:
            bad_epochs += 1
        atomic_torch_save(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "seed": seed,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_clean_normal_rmse": best_rmse,
                "bad_epochs": bad_epochs,
                "model_config": model.model_config(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            last_path,
        )
        print(
            f"[mean-only seed={seed}] epoch={epoch:02d} "
            f"train_rmse={train['rmse_scaled']:.6f} "
            f"val_rmse={validation['rmse_scaled']:.6f}"
            f"{' *' if improved else ''}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_train = evaluate_mean(
        model, dataset, windows, train_indices, scaler, args.batch_size, device
    )
    best_validation = evaluate_mean(
        model,
        dataset,
        windows,
        validation_indices,
        scaler,
        args.batch_size,
        device,
    )
    persistence = diagnostic.persistence_baseline(
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        args.batch_size,
        0,
    )
    epoch8 = history[7]
    summary = {
        "seed": seed,
        "epochs_completed": len(history),
        "cumulative_optimizer_steps": cumulative_steps,
        "stop_reason": (
            "validation_patience"
            if bad_epochs >= args.patience
            else "maximum_epochs"
        ),
        "best_epoch": best_epoch,
        "epoch8_validation_rmse": epoch8["validation_rmse_scaled"],
        "best_validation_rmse": best_validation["rmse_scaled"],
        "validation_rmse_improvement_after_epoch8": (
            epoch8["validation_rmse_scaled"] - best_validation["rmse_scaled"]
        ),
        "best": {"train": best_train, "validation": best_validation},
        "persistence_validation_rmse": persistence["validation"][
            "forecast_rmse_scaled"
        ],
        "rmse_skill_vs_persistence": (
            1.0
            - best_validation["rmse_scaled"]
            / persistence["validation"]["forecast_rmse_scaled"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history": history,
    }
    atomic_json_dump(summary, run_dir / "summary.json")
    diagnostic.write_csv(run_dir / "history.csv", history)
    return summary


def aggregate_runs(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "epochs_completed",
        "cumulative_optimizer_steps",
        "best_epoch",
        "epoch8_validation_rmse",
        "best_validation_rmse",
        "validation_rmse_improvement_after_epoch8",
        "persistence_validation_rmse",
        "rmse_skill_vs_persistence",
    )
    result: dict[str, Any] = {
        "runs": len(summaries),
        "patience_stop_count": sum(
            item["stop_reason"] == "validation_patience" for item in summaries
        ),
        "maximum_epoch_stop_count": sum(
            item["stop_reason"] == "maximum_epochs" for item in summaries
        ),
    }
    for key in keys:
        values = np.asarray([float(item[key]) for item in summaries])
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def main() -> None:
    args = parse_args()
    seeds = validate_args(args)
    device = diagnostic.resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, windows, train, validation, scaler, support = (
        diagnostic.prepare_support(args.data_dir)
    )
    with torch.random.fork_rng(devices=[]):
        model = GRUNBM(
            dataset.n_channels,
            diagnostic.base.TARGET_SAMPLES,
            hidden_channels=args.hidden_channels,
            num_layers=1,
            dropout=args.dropout,
        )
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_file": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": utc_now(),
        "purpose": (
            "Ablate Gaussian NLL: optimize only the future mean with 0.5*MSE."
        ),
        "not_a_model_selection_protocol": True,
        "records_used": [record.record_id for record in dataset.records],
        "held_out_test_record": diagnostic.base.TEST_RECORD,
        "test_policy": "R02 array is never opened and receives zero forward passes.",
        "support": support,
        "train_windows": len(train),
        "validation_windows": len(validation),
        "model": {
            "config": model.model_config(),
            "total_parameter_count": parameter_count(model),
            "effective_mean_path_parameter_count": 67_296,
            "ignored_sigma_output_parameter_count": 56_448,
            "sigma_output_ignored": True,
        },
        "training": {
            "objective": "0.5 * mean((target - mean)^2)",
            "seeds": list(seeds),
            "maximum_epochs": args.max_epochs,
            "minimum_epochs": MIN_EPOCHS,
            "patience": args.patience,
            "early_stop_metric": "validation clean-normal RMSE",
            "min_delta_rmse": MIN_DELTA_RMSE,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "optimizer_steps_per_epoch": math.ceil(len(train) / args.batch_size),
            "amp": args.amp,
            "deterministic_algorithms_strict": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    config["protocol_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in config.items() if key not in {"created_utc", "environment"}}
    )
    atomic_json_dump(config, output_dir / "config.json")
    summaries = [
        run_seed(
            args,
            seed,
            dataset,
            windows,
            train,
            validation,
            scaler,
            config["protocol_fingerprint"],
            output_dir,
            device,
        )
        for seed in seeds
    ]
    aggregate = aggregate_runs(summaries)
    atomic_json_dump(aggregate, output_dir / "aggregate.json")
    diagnostic.write_csv(
        output_dir / "run_table.csv",
        [
            {
                "seed": item["seed"],
                "epochs_completed": item["epochs_completed"],
                "stop_reason": item["stop_reason"],
                "best_epoch": item["best_epoch"],
                "epoch8_validation_rmse": item["epoch8_validation_rmse"],
                "best_validation_rmse": item["best_validation_rmse"],
                "improvement_after_epoch8": item[
                    "validation_rmse_improvement_after_epoch8"
                ],
                "persistence_validation_rmse": item[
                    "persistence_validation_rmse"
                ],
                "rmse_skill_vs_persistence": item[
                    "rmse_skill_vs_persistence"
                ],
            }
            for item in summaries
        ],
    )
    artifacts = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "DONE.json"
    }
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "test_record_evaluated": False,
            "artifacts": artifacts,
        },
        output_dir / "DONE.json",
    )
    print(
        "COMPLETE "
        f"mean_best_val_rmse={aggregate['best_validation_rmse']['mean']:.6f} "
        f"mean_skill={aggregate['rmse_skill_vs_persistence']['mean']:.4%} "
        f"max_epoch_stops={aggregate['maximum_epoch_stop_count']}/{len(seeds)} "
        "test_evaluated=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
