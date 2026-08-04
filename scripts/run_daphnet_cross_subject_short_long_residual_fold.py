#!/usr/bin/env python
"""One LOSO fold comparing 2 s residual TCN-M with 2 s + 6 s fusion."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

import run_daphnet_cross_subject_fold as cross
import run_daphnet_loso_s01_gru_h200_tcnm as loso
import run_daphnet_s01_gru_h200_tcnm as core
from cnbr_fog.h200_feasibility import RF125TCNFeatureEncoder
from cnbr_fog.histories import (
    make_common_history_plan,
    materialize_nonoverlap_residual_history,
)
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "daphnet_cross_subject_gru_residual_short2_long6_tcnm.v1"
ARMS = ("short_2s", "short_2s_long_6s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-subject GRU residual 2 s versus 2 s + 6 s TCN-M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-subject", choices=cross.TEST_SUBJECTS, required=True)
    parser.add_argument("--validation-subject", choices=cross.TEST_SUBJECTS, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normal-epochs", type=int, default=50)
    parser.add_argument("--normal-patience", type=int, default=6)
    parser.add_argument("--normal-lr", type=float, default=1e-3)
    parser.add_argument("--nbm-hidden", type=int, default=48)
    parser.add_argument("--nbm-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--target-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class Short2sTCNM(nn.Module):
    """RF125 TCN-M that sees only the terminal 2 s of common 6 s support."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = RF125TCNFeatureEncoder(
            in_channels=channels,
            input_samples=128,
            hidden_channels=hidden,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(history[:, :, -128:])).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "arm": "short_2s",
            "input_support_samples": 384,
            "visible_short_samples": 128,
            "encoder": self.encoder.architecture_config(),
            "fusion": None,
            "parameter_count": core.parameter_count(self),
        }


class Short2sLong6sTCNM(nn.Module):
    """Two RF125 TCN-M encoders with pooled late feature fusion."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        # Construct short first so its seed-aligned initialization matches the
        # single-branch control when each arm resets the same classifier seed.
        self.short_encoder = RF125TCNFeatureEncoder(
            in_channels=channels,
            input_samples=128,
            hidden_channels=hidden,
            dropout=dropout,
        )
        self.long_encoder = RF125TCNFeatureEncoder(
            in_channels=channels,
            input_samples=384,
            hidden_channels=hidden,
            dropout=dropout,
        )
        fused = self.short_encoder.output_features + self.long_encoder.output_features
        self.head = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        short_features = self.short_encoder(history[:, :, -128:])
        long_features = self.long_encoder(history)
        return self.head(torch.cat([short_features, long_features], dim=1)).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "arm": "short_2s_long_6s",
            "short_encoder": self.short_encoder.architecture_config(),
            "long_encoder": self.long_encoder.architecture_config(),
            "fusion": "late concatenation of per-branch temporal mean and max pooling",
            "head_input_features": (
                self.short_encoder.output_features + self.long_encoder.output_features
            ),
            "parameter_count": core.parameter_count(self),
        }


def build_model(arm: str, channels: int, args: argparse.Namespace) -> nn.Module:
    if arm == "short_2s":
        return Short2sTCNM(channels, args.classifier_hidden, args.classifier_dropout)
    if arm == "short_2s_long_6s":
        return Short2sLong6sTCNM(
            channels, args.classifier_hidden, args.classifier_dropout
        )
    raise ValueError(f"Unknown arm: {arm}")


def history_features(windows, split, residual_features):
    result: dict[str, dict[str, np.ndarray]] = {}
    support: dict[str, dict[str, Any]] = {}
    for name, indices in split.as_dict().items():
        plan = make_common_history_plan(
            windows,
            indices,
            horizon_samples=core.TARGET_SAMPLES,
            stride_samples=core.STRIDE_SAMPLES,
            max_history_samples=384,
        )
        extracted = residual_features[name]
        history = materialize_nonoverlap_residual_history(
            extracted["residual"],
            plan,
            history_samples=384,
            horizon_samples=core.TARGET_SAMPLES,
            stride_samples=core.STRIDE_SAMPLES,
        ).astype(np.float32, copy=False)
        y = np.asarray(extracted["y"])[plan.anchor_rows].astype(np.int8, copy=False)
        window_index = plan.anchor_window_indices.astype(np.int64, copy=False)
        if not np.array_equal(
            history[:, :, -core.TARGET_SAMPLES :],
            extracted["residual"][plan.anchor_rows],
        ):
            raise AssertionError(f"{name}: terminal short residual differs from anchor")
        result[name] = {
            "history": np.ascontiguousarray(history),
            "y": y,
            "window_index": window_index,
        }
        block_indices = np.asarray(indices)[plan.max_chain_rows]
        support[name] = {
            "windows_before_history": int(len(indices)),
            "anchors_after_history": int(len(window_index)),
            "class_counts": np.bincount(y, minlength=2).astype(int).tolist(),
            "anchor_window_index": window_index,
            "block_window_index": block_indices,
            "block_target_start": windows.target_start[block_indices],
            "block_target_end": windows.target_end[block_indices],
        }
        if np.any(np.diff(windows.target_start[block_indices], axis=1) != 128):
            raise AssertionError(f"{name}: history blocks are not separated by 2 seconds")
        if np.any(windows.target_end[block_indices[:, :-1]] != windows.target_start[block_indices[:, 1:]]):
            raise AssertionError(f"{name}: history blocks are not contiguous")
    return result, support


def train_arm(
    arm: str,
    args: argparse.Namespace,
    output_dir: Path,
    features,
    dataset,
    windows,
    protocol_fingerprint: str,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = features["train"]["history"]
    y_train = features["train"]["y"]
    x_validation = features["validation"]["history"]
    y_validation = features["validation"]["y"]
    x_test = features["test"]["history"]
    y_test = features["test"]["y"]
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"{arm}: classifier train lacks a class")

    classifier_seed = args.seed + 10_000
    core.set_seed(classifier_seed, args.deterministic)
    model = build_model(arm, dataset.n_channels, args).to(device)
    architecture = model.architecture_config()
    pos_weight_value = min(float(np.sqrt(counts[0] / counts[1])), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )
    val_loader = core.array_loader(
        x_validation, y_validation, args.batch_size, False, args.num_workers,
        device.type == "cuda",
    )
    test_loader = core.array_loader(
        x_test, y_test, args.batch_size, False, args.num_workers,
        device.type == "cuda",
    )
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    checkpoint_path = output_dir / "classifier_best.pt"
    started = time.perf_counter()
    for epoch in range(1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
        train_loader = core.array_loader(
            x_train, y_train, args.batch_size, True, args.num_workers,
            device.type == "cuda", seed=classifier_seed + epoch,
        )
        train_loss, train_true, train_probability = core.classifier_epoch(
            model, train_loader, criterion, device, args.amp, optimizer, grad_scaler
        )
        with torch.no_grad():
            validation_loss, validation_true, validation_probability = core.classifier_epoch(
                model, val_loader, criterion, device, args.amp
            )
        validation_pr = float(
            average_precision_score(validation_true, validation_probability)
        )
        improved = validation_pr > best_score + 1e-5
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": classifier_seed + epoch,
                "train_bce": train_loss,
                "train_pr_auc": float(
                    average_precision_score(train_true, train_probability)
                ),
                "validation_bce": validation_loss,
                "validation_pr_auc": validation_pr,
                "improved": improved,
            }
        )
        if improved:
            best_score = validation_pr
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "arm": arm,
                    "seed": classifier_seed,
                    "epoch": epoch,
                    "architecture": architecture,
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        print(
            f"[{arm}] epoch={epoch:02d} train_bce={train_loss:.6f} "
            f"val_pr={validation_pr:.6f}{' *' if improved else ''}",
            flush=True,
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    with torch.no_grad():
        _, validation_true, validation_probability = core.classifier_epoch(
            model, val_loader, criterion, device, args.amp
        )
        _, test_true, test_probability = core.classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, validation_metrics = core.choose_threshold(
        validation_true, validation_probability
    )
    validation_metrics = core.enrich_metrics(validation_metrics)
    test_metrics = core.enrich_metrics(
        core.binary_metrics(test_true, test_probability, threshold)
    )
    validation_prediction = (validation_probability >= threshold).astype(np.int8)
    test_prediction = (test_probability >= threshold).astype(np.int8)
    test_metrics.update(
        core.event_metrics(
            dataset,
            core.event_scoring_windows(windows),
            features["test"]["window_index"],
            test_prediction,
            minimum_positive_windows=1,
            merge_gap_seconds=0.5,
        )
    )
    training = {
        "arm": arm,
        "seed": classifier_seed,
        "architecture": architecture,
        "optimizer": "AdamW",
        "learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "loss": "BCEWithLogitsLoss",
        "train_counts_non_fog_fog": counts.astype(int).tolist(),
        "positive_class_weight": pos_weight_value,
        "maximum_epochs": args.classifier_epochs,
        "patience": args.classifier_patience,
        "early_stop_metric": "validation PR-AUC",
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_score,
        "epochs_completed": len(history),
        "selected_threshold": threshold,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "test_positive_prevalence_baseline_pr_auc": float(np.mean(test_true)),
    }
    atomic_json_dump(training, output_dir / "classifier_training.json")
    atomic_json_dump(metrics, output_dir / "metrics.json")
    core.write_csv(output_dir / "classifier_training_history.csv", history)
    core.write_csv(
        output_dir / "validation_predictions.csv",
        core.prediction_rows(
            dataset, windows, features["validation"]["window_index"],
            validation_true, validation_probability, validation_prediction,
        ),
    )
    core.write_csv(
        output_dir / "test_predictions.csv",
        core.prediction_rows(
            dataset, windows, features["test"]["window_index"],
            test_true, test_probability, test_prediction,
        ),
    )
    core.plot_classifier_losses(output_dir, training)
    core.plot_test_confusion_matrix(output_dir, test_metrics, loso.TEST_SUBJECT)
    return training, metrics


def write_fold_summary(output_dir, args, nbm_training, arm_results, support):
    lines = [
        f"# {args.test_subject} short/long residual LOSO fold",
        "",
        f"- Train subjects: {', '.join(loso.TRAIN_SUBJECTS)}",
        f"- Validation subject: {args.validation_subject}",
        f"- Test subject: {args.test_subject}",
        "- Context/forecast/stride: 2/2/1 seconds",
        "- Common classifier anchors require a complete 6-second, three-block history.",
        f"- GRU maximum/best/completed epoch: {nbm_training['maximum_epochs']}/{nbm_training['best_epoch']}/{nbm_training['epochs_completed']}",
        f"- Test windows before/after history support: {support['test']['windows_before_history']}/{support['test']['anchors_after_history']}",
        "",
        "| Arm | Parameters | TCN best/completed | Threshold | Accuracy | FoG Recall | Specificity | PR-AUC | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        training, metrics = arm_results[arm]
        test = metrics["test"]
        lines.append(
            f"| {arm} | {training['architecture']['parameter_count']:,} "
            f"| {training['best_epoch']}/{training['epochs_completed']} "
            f"| {training['selected_threshold']:.2f} | {test['accuracy']:.4f} "
            f"| {test['fog_recall']:.4f} | {test['specificity']:.4f} "
            f"| {test['pr_auc']:.4f} | {test['tn']}/{test['fp']}/{test['fn']}/{test['tp']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.architecture = "gru_residual"
    cross.configure(args)
    core.EXPERIMENT_VERSION = VERSION
    loso.EXPERIMENT_VERSION = VERSION
    core.validate_args(args)
    if core.CONTEXT_SAMPLES != 128 or core.TARGET_SAMPLES != 128 or core.STRIDE_SAMPLES != 64:
        raise ValueError("This experiment requires context/forecast/stride = 2/2/1 seconds")
    device = core.resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"Completed output already exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = loso.load_dataset(args.data_dir.resolve())
    base_windows = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base_windows)
    split = loso.make_split(dataset, windows)
    scaler, scaler_metadata = loso.fit_training_scaler(dataset)
    point_stats = loso.point_statistics(dataset)
    window_stats = loso.window_statistics(dataset, windows, split)
    clip_stats = loso.scaler_clip_statistics(dataset, scaler)
    protocol = cross.build_protocol(
        args, VERSION, dataset, point_stats, window_stats, scaler_metadata,
        clip_stats, device,
    )
    protocol["experiment_version"] = VERSION
    protocol["short_long_experiment"] = {
        "arms": list(ARMS),
        "forecast_block_seconds": 2.0,
        "short_seconds": 2.0,
        "long_seconds": 6.0,
        "long_block_count": 3,
        "block_rule": "three horizon-spaced contiguous non-overlapping residual blocks",
        "common_anchor_support": True,
        "fusion": "two RF125 TCN-M encoders; temporal mean/max pooling; late concatenation",
        "capacity_caveat": "dual arm has a second encoder and is not parameter matched",
    }
    protocol["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in protocol.items()
            if key not in {"created_utc", "environment", "protocol_fingerprint"}
        }
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(scaler_metadata, output_dir / "scaler.json")
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train_window_index=split.train,
        validation_window_index=split.validation,
        test_window_index=split.test,
    )
    print(
        f"Protocol {protocol['protocol_fingerprint']}\ndevice={device} "
        f"subjects={loso.subject_groups()} window_counts="
        f"{ {name: len(value) for name, value in split.as_dict().items()} }",
        flush=True,
    )
    if args.dry_run:
        plans = {
            name: make_common_history_plan(windows, indices, 128, 64, 384)
            for name, indices in split.as_dict().items()
        }
        atomic_json_dump(
            {
                "status": "dry_run_complete",
                "anchors": {name: len(plan.anchor_rows) for name, plan in plans.items()},
                "protocol_fingerprint": protocol["protocol_fingerprint"],
            },
            output_dir / "DRY_RUN.json",
        )
        return

    nbm, nbm_training = core.train_nbm(
        args, dataset, windows, split, scaler, output_dir,
        protocol["protocol_fingerprint"], device,
    )
    core.plot_nbm_losses(output_dir, nbm_training)
    residual_features: dict[str, dict[str, np.ndarray]] = {}
    residual_diagnostics: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        residual_features[name], residual_diagnostics[name] = core.extract_residuals(
            args, nbm, dataset, windows, indices, scaler, device
        )
    atomic_json_dump(residual_diagnostics, output_dir / "residual_diagnostics.json")
    atomic_npz_save(
        output_dir / "residual_cache.npz",
        **{
            f"{name}_{field}": values[field]
            for name, values in residual_features.items()
            for field in ("residual", "y", "window_index")
        },
    )
    features, support = history_features(windows, split, residual_features)
    support_json = {
        name: {
            key: value
            for key, value in values.items()
            if key not in {"anchor_window_index", "block_window_index", "block_target_start", "block_target_end"}
        }
        for name, values in support.items()
    }
    atomic_json_dump(support_json, output_dir / "history_support.json")
    atomic_npz_save(
        output_dir / "history_support.npz",
        **{
            f"{name}_{field}": values[field]
            for name, values in support.items()
            for field in ("anchor_window_index", "block_window_index", "block_target_start", "block_target_end")
        },
    )
    arm_results = {}
    for arm in ARMS:
        arm_results[arm] = train_arm(
            arm, args, output_dir / arm, features, dataset, windows,
            protocol["protocol_fingerprint"], device,
        )
    write_fold_summary(output_dir, args, nbm_training, arm_results, support_json)
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": core.utc_now(),
            "experiment_version": VERSION,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "artifacts": {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in sorted(output_dir.rglob("*"))
                if path.is_file()
            },
        },
        done_path,
    )
    print("COMPLETE " + " ".join(
        f"{arm}_pr_auc={arm_results[arm][1]['test']['pr_auc']:.6f}" for arm in ARMS
    ), flush=True)


if __name__ == "__main__":
    main()
