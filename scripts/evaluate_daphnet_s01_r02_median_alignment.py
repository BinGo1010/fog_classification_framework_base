#!/usr/bin/env python
"""Diagnostic record-level median alignment for S01_seg002/R02.

This is deliberately a transductive diagnostic: the per-channel test median is
computed from the complete S01_seg002 record before frozen-pipeline inference.
No labels are used to estimate that median, but future test samples are used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)
from cnbr_fog.rf125_classifiers import build_rf125_classifier  # noqa: E402
import run_daphnet_s01_dae_tcnm as dae_core  # noqa: E402
import run_daphnet_s01_gru_h200_tcnm as core  # noqa: E402
import run_daphnet_s01_pretrained_dae_tcnm as frozen_core  # noqa: E402


EXPERIMENT_VERSION = "daphnet_s01_r02_record_median_alignment.v1"
TEST_RECORD = "S01_seg002"
TEST_RUN = "R02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic full-record median alignment of S01_seg002/R02",
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
        "--dae-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_dae_only_max200_seed42"
        ),
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_dae_max200_best120_tcnm50_seed42"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_dae_tcnm50_r02_record_median_alignment"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_artifact_directory(directory: Path) -> dict[str, Any]:
    done_path = directory / "DONE.json"
    if not done_path.is_file():
        raise FileNotFoundError(done_path)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete":
        raise ValueError(f"incomplete source artifact: {directory}")
    for name, expected in done.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source artifact hash mismatch: {path}")
    return done


def training_nonfog_median(
    dataset: core.DaphnetDataset,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    points = 0
    for record, start, end in core.training_ranges(dataset):
        mask = record.valid[start:end] & (record.y[start:end] == 0)
        chunks.append(record.x[start:end][mask])
        points += int(mask.sum())
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    median = np.median(values, axis=0)
    if points != len(values) or not np.isfinite(median).all():
        raise AssertionError("invalid training Non-FoG median support")
    return median.astype(np.float32), points


def load_classifier(
    pipeline_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, Any], dict[str, Any]]:
    training = json.loads(
        (pipeline_dir / "classifier_training.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (pipeline_dir / "metrics.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        pipeline_dir / "classifier_best.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("epoch") != training.get("best_epoch"):
        raise ValueError("classifier checkpoint/training best epoch mismatch")
    if training.get("checkpoint_sha256") != sha256_file(
        pipeline_dir / "classifier_best.pt"
    ):
        raise ValueError("classifier checkpoint hash mismatch")
    architecture = checkpoint["architecture"]
    model = build_rf125_classifier(
        architecture["canonical_name"],
        in_channels=int(architecture["in_channels"]),
        input_samples=int(architecture["input_samples"]),
        dropout=float(architecture["dropout"]),
        hidden_channels=int(architecture["hidden_channels"]),
        dilations=tuple(int(x) for x in architecture["dilations"]),
        kernel_size=int(architecture["kernel_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    threshold = float(training["selected_threshold"])
    if threshold != float(metrics["validation"]["threshold"]):
        raise ValueError("classifier threshold metadata mismatch")
    return model, threshold, training, metrics


@torch.no_grad()
def classifier_probabilities(
    model: nn.Module,
    residual: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    loader = core.array_loader(
        residual,
        labels,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    criterion = nn.BCEWithLogitsLoss()
    _, returned_labels, probabilities = core.classifier_epoch(
        model,
        loader,
        criterion,
        device,
        args.amp,
    )
    if not np.array_equal(returned_labels, labels):
        raise AssertionError("classifier inference changed label order")
    return np.asarray(probabilities, dtype=np.float32)


def json_channel_mapping(
    names: tuple[str, ...] | list[str], values: np.ndarray
) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(names, np.asarray(values).reshape(-1))
    }


def build_protocol(
    args: argparse.Namespace,
    dataset: core.DaphnetDataset,
    dae_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    classifier_training: dict[str, Any],
    train_median: np.ndarray,
    test_median: np.ndarray,
    train_points: int,
    test_points: int,
    device: torch.device,
) -> dict[str, Any]:
    shift = train_median - test_median
    payload: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": utc_now(),
        "diagnostic_only": True,
        "transductive_test_time_adaptation": True,
        "online_causal": False,
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "test_record": TEST_RECORD,
        "test_run": TEST_RUN,
        "source_dae": {
            "directory": str(args.dae_dir.resolve()),
            "protocol_fingerprint": dae_config["protocol_fingerprint"],
            "best_checkpoint_sha256": sha256_file(args.dae_dir / "dae_best.pt"),
            "weights_frozen": True,
        },
        "source_residual_classifier_pipeline": {
            "directory": str(args.pipeline_dir.resolve()),
            "protocol_fingerprint": pipeline_config["protocol_fingerprint"],
            "fixed_sigma_sha256": sha256_file(
                args.pipeline_dir / "fixed_sigma.npy"
            ),
            "classifier_checkpoint_sha256": sha256_file(
                args.pipeline_dir / "classifier_best.pt"
            ),
            "classifier_best_epoch": classifier_training["best_epoch"],
            "classifier_maximum_epochs": classifier_training["maximum_epochs"],
            "selected_threshold": classifier_training["selected_threshold"],
            "all_weights_and_threshold_frozen": True,
        },
        "alignment": {
            "domain": "raw physical acceleration in g, before DAE z-score",
            "formula": "x_aligned[c,t] = x[c,t] - median_test[c] + median_train_nonfog[c]",
            "train_median_support": (
                "all valid sample-level Non-FoG points in the declared training ranges"
            ),
            "train_median_points": int(train_points),
            "test_median_support": (
                "all valid points in the complete S01_seg002 record; labels not used"
            ),
            "test_median_points": int(test_points),
            "complete_test_record_used_before_inference": True,
            "includes_future_test_samples": True,
            "includes_unlabelled_mixture_of_nonfog_and_fog": True,
            "per_channel_train_nonfog_median_g": json_channel_mapping(
                dataset.channel_names, train_median
            ),
            "per_channel_test_record_median_g": json_channel_mapping(
                dataset.channel_names, test_median
            ),
            "per_channel_additive_shift_g": json_channel_mapping(
                dataset.channel_names, shift
            ),
        },
        "frozen_inference": {
            "dae_scaler_refit": False,
            "dae_weights_updated": False,
            "fixed_sigma_refit": False,
            "classifier_weights_updated": False,
            "threshold_reselected": False,
            "test_labels_used_for_alignment": False,
            "only_test_raw_signal_changed": True,
        },
        "interpretation_limit": (
            "The complete test record is used to estimate its channel medians. "
            "This is an unsupervised transductive diagnostic and must not be reported "
            "as an untouched-test or causal online result."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    payload["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_utc", "environment"}
        }
    )
    return payload


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    baseline: dict[str, Any],
    aligned: dict[str, Any],
    aligned_diagnostics: dict[str, Any],
) -> None:
    delta = {
        key: aligned[key] - baseline[key]
        for key in ("accuracy", "fog_recall", "specificity", "pr_auc")
    }
    text = f"""# S01_seg002/R02 diagnostic record-median alignment

This is a transductive diagnostic. The complete test record was used to estimate
its channel medians before inference; no labels were used for alignment.

## Result

| Test signal | Accuracy | FoG recall | Specificity | PR-AUC |
|---|---:|---:|---:|---:|
| Original | {baseline['accuracy']:.6f} | {baseline['fog_recall']:.6f} | {baseline['specificity']:.6f} | {baseline['pr_auc']:.6f} |
| Median aligned | {aligned['accuracy']:.6f} | {aligned['fog_recall']:.6f} | {aligned['specificity']:.6f} | {aligned['pr_auc']:.6f} |
| Difference | {delta['accuracy']:+.6f} | {delta['fog_recall']:+.6f} | {delta['specificity']:+.6f} | {delta['pr_auc']:+.6f} |

Frozen validation threshold: {aligned['threshold']:.4f}.
Aligned confusion matrix: `[[{aligned['tn']}, {aligned['fp']}], [{aligned['fn']}, {aligned['tp']}]]`.

Aligned Non-FoG/FoG residual RMS: {aligned_diagnostics['non_fog']['residual_clipped_rms']:.6f} /
{aligned_diagnostics['fog']['residual_clipped_rms']:.6f}.

## Alignment

`x_aligned[c,t] = x[c,t] - median_test_record[c] + median_train_nonfog[c]`

- Train median support: {protocol['alignment']['train_median_points']} valid Non-FoG training points.
- Test median support: {protocol['alignment']['test_median_points']} valid points from complete S01_seg002.
- DAE, scaler, fixed sigma, TCN-M weights and threshold remained frozen.

This result cannot be treated as an untouched-test or causal online score because
future test samples contribute to the record-level median.
"""
    temporary = output_dir / f".summary.md.tmp-{os.getpid()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid batch-size or num-workers")
    device = core.resolve_device(args.device)
    dae_dir = args.dae_dir.resolve()
    pipeline_dir = args.pipeline_dir.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(done_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dae_done = verify_artifact_directory(dae_dir)
    pipeline_done = verify_artifact_directory(pipeline_dir)
    dae_model, scaler, dae_config, _, _ = frozen_core.load_frozen_dae(
        dae_dir, device
    )
    pipeline_config = json.loads(
        (pipeline_dir / "config.json").read_text(encoding="utf-8")
    )
    current_fingerprint = dataset_fingerprint(args.data_dir)
    if current_fingerprint != dae_config["dataset_fingerprint_sha256"]:
        raise ValueError("current/source DAE dataset fingerprint mismatch")
    if current_fingerprint != pipeline_config["dataset_fingerprint_sha256"]:
        raise ValueError("current/source pipeline dataset fingerprint mismatch")
    if pipeline_config["source_dae"]["best_checkpoint_sha256"] != sha256_file(
        dae_dir / "dae_best.pt"
    ):
        raise ValueError("pipeline does not reference the requested DAE checkpoint")

    dataset = core.load_s01_dataset(args.data_dir)
    base_windows = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base_windows)
    split = core.make_split(dataset, windows)
    with np.load(pipeline_dir / "split_indices.npz", allow_pickle=False) as archive:
        for name, indices in split.as_dict().items():
            if not np.array_equal(archive[f"{name}_window_index"], indices):
                raise ValueError(f"pipeline/current {name} split mismatch")

    lookup = core.record_lookup(dataset)
    record = dataset.records[lookup[TEST_RECORD]]
    if record.run_id != TEST_RUN:
        raise ValueError(f"expected {TEST_RECORD}/{TEST_RUN}, got {record.run_id}")
    if not np.array_equal(
        split.test, np.flatnonzero(windows.record_index == lookup[TEST_RECORD])
    ):
        raise ValueError("test split is not exactly S01_seg002")

    train_median, train_points = training_nonfog_median(dataset)
    test_mask = record.valid
    test_values = record.x[test_mask].astype(np.float64, copy=False)
    test_median = np.median(test_values, axis=0).astype(np.float32)
    shift = train_median - test_median
    aligned_record_median = np.median(
        record.x[test_mask].astype(np.float64) + shift.astype(np.float64),
        axis=0,
    )
    np.testing.assert_allclose(
        aligned_record_median,
        train_median.astype(np.float64),
        rtol=0,
        atol=2e-7,
    )

    classifier, threshold, classifier_training, source_metrics = load_classifier(
        pipeline_dir, device
    )
    protocol = build_protocol(
        args,
        dataset,
        dae_config,
        pipeline_config,
        classifier_training,
        train_median,
        test_median,
        train_points,
        int(test_mask.sum()),
        device,
    )
    protocol["source_artifact_verification"] = {
        "dae_artifacts_verified": len(dae_done.get("artifacts", {})),
        "pipeline_artifacts_verified": len(pipeline_done.get("artifacts", {})),
    }
    protocol["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in protocol.items()
            if key not in {"created_utc", "environment", "protocol_fingerprint"}
        }
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(
        {
            "formula": protocol["alignment"]["formula"],
            "channel_names": list(dataset.channel_names),
            "train_nonfog_median_g": train_median.tolist(),
            "test_record_median_g": test_median.tolist(),
            "additive_shift_g": shift.tolist(),
            "aligned_test_record_median_g": aligned_record_median.tolist(),
            "train_nonfog_points": train_points,
            "test_record_valid_points": int(test_mask.sum()),
            "test_record_nonfog_points_not_used_for_estimation_filter": int(
                np.sum((record.y == 0) & test_mask)
            ),
            "test_record_fog_points_not_used_for_estimation_filter": int(
                np.sum((record.y == 1) & test_mask)
            ),
            "labels_used_for_test_median": False,
        },
        output_dir / "alignment.json",
    )

    raw_target = dae_core.extract_target_windows(
        dataset, windows, split.test
    )
    aligned_raw_target = (
        raw_target
        + shift[None, :, None].astype(np.float32)
    ).astype(np.float32)
    original_scaled = np.ascontiguousarray(
        scaler.transform_channel_time(raw_target)
    )
    aligned_scaled = np.ascontiguousarray(
        scaler.transform_channel_time(aligned_raw_target)
    )
    original_reconstruction, _ = dae_core.reconstruct(
        args, dae_model, original_scaled, device
    )
    aligned_reconstruction, aligned_latent = dae_core.reconstruct(
        args, dae_model, aligned_scaled, device
    )
    fixed_sigma = np.load(
        pipeline_dir / "fixed_sigma.npy", allow_pickle=False
    ).astype(np.float32)
    with np.load(
        pipeline_dir / "residual_process.npz", allow_pickle=False
    ) as source_residual:
        expected_original = source_residual["test_residual_clipped"]
        reproduced_original = np.clip(
            (original_scaled - original_reconstruction) / fixed_sigma,
            -dae_core.RESIDUAL_CLIP,
            dae_core.RESIDUAL_CLIP,
        ).astype(np.float32)
        np.testing.assert_allclose(
            reproduced_original, expected_original, rtol=0, atol=0
        )

    aligned_error = aligned_scaled - aligned_reconstruction
    aligned_unclipped = aligned_error / fixed_sigma
    aligned_residual = np.clip(
        aligned_unclipped,
        -dae_core.RESIDUAL_CLIP,
        dae_core.RESIDUAL_CLIP,
    ).astype(np.float32)
    labels = windows.label[split.test].astype(np.int8, copy=True)
    probabilities = classifier_probabilities(
        classifier, aligned_residual, labels, args, device
    )
    predictions = (probabilities >= threshold).astype(np.int8)
    aligned_metrics = core.enrich_metrics(
        core.binary_metrics(labels, probabilities, threshold)
    )
    baseline_metrics = source_metrics["test"]
    aligned_diagnostics = dae_core.residual_diagnostics(
        aligned_scaled,
        aligned_reconstruction,
        fixed_sigma,
        aligned_unclipped,
        aligned_residual,
        labels,
        scaler,
    )
    comparison = {
        "diagnostic_only": True,
        "baseline_original_test": baseline_metrics,
        "record_median_aligned_test": aligned_metrics,
        "aligned_minus_original": {
            key: float(aligned_metrics[key] - baseline_metrics[key])
            for key in (
                "accuracy",
                "fog_recall",
                "specificity",
                "pr_auc",
                "roc_auc",
                "balanced_accuracy",
            )
        },
        "threshold_frozen": threshold,
        "threshold_reselected_on_test": False,
    }
    atomic_json_dump(comparison, output_dir / "metrics.json")
    atomic_json_dump(
        aligned_diagnostics, output_dir / "aligned_residual_diagnostics.json"
    )
    core.write_csv(
        output_dir / "aligned_test_predictions.csv",
        core.prediction_rows(
            dataset,
            windows,
            split.test,
            labels,
            probabilities,
            predictions,
        ),
    )
    atomic_npz_save(
        output_dir / "aligned_test_process.npz",
        window_index=split.test,
        y_true=labels,
        raw_target_g=raw_target,
        aligned_raw_target_g=aligned_raw_target,
        aligned_target_scaled=aligned_scaled,
        aligned_reconstruction_scaled=aligned_reconstruction,
        aligned_error_scaled=aligned_error.astype(np.float32),
        fixed_sigma=fixed_sigma,
        aligned_residual_unclipped=aligned_unclipped.astype(np.float32),
        aligned_residual_clipped=aligned_residual,
        aligned_latent=aligned_latent,
        fog_probability=probabilities,
        y_pred=predictions,
        train_nonfog_median_g=train_median,
        test_record_median_g=test_median,
        additive_shift_g=shift,
    )
    write_summary(
        output_dir,
        protocol,
        baseline_metrics,
        aligned_metrics,
        aligned_diagnostics,
    )
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        },
        done_path,
    )
    print(
        "COMPLETE "
        f"accuracy={aligned_metrics['accuracy']:.6f} "
        f"recall={aligned_metrics['fog_recall']:.6f} "
        f"specificity={aligned_metrics['specificity']:.6f} "
        f"pr_auc={aligned_metrics['pr_auc']:.6f} "
        f"confusion={aligned_metrics['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
