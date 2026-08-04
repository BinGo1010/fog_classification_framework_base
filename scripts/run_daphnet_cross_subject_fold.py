#!/usr/bin/env python
"""Run one subject-disjoint fold for GRU-residual or raw-target TCN-M."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

import run_daphnet_loso_s01_gru_h200_tcnm as loso
import run_daphnet_s01_gru_h200_tcnm as core
import run_daphnet_single_subject_raw_tcnm as raw
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_SUBJECTS = tuple(f"S{index:02d}" for index in range(1, 11))
TEST_SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
ARCHITECTURES = ("gru_residual", "raw_target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One subject-disjoint GRU-residual/raw-target TCN-M fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--test-subject", choices=TEST_SUBJECTS, required=True)
    parser.add_argument("--validation-subject", choices=TEST_SUBJECTS, required=True)
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
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> str:
    if args.test_subject == args.validation_subject:
        raise ValueError("Test and validation subjects must differ")
    for name in ("context", "target", "stride"):
        seconds = float(getattr(args, f"{name}_seconds"))
        samples = seconds * core.SAMPLING_RATE_HZ
        if seconds <= 0 or not float(samples).is_integer():
            raise ValueError(f"--{name}-seconds must resolve to positive whole samples")
        setattr(core, f"{name.upper()}_SAMPLES", int(samples))
    if core.LABEL_SAMPLES > core.TARGET_SAMPLES:
        raise ValueError("The final 0.5-second label support exceeds the target")

    training = tuple(
        subject
        for subject in ALL_SUBJECTS
        if subject not in {args.test_subject, args.validation_subject}
    )
    loso.TEST_SUBJECT = args.test_subject
    loso.VALIDATION_SUBJECT = args.validation_subject
    loso.TRAIN_SUBJECTS = training
    loso.EXPECTED_SUBJECTS = ALL_SUBJECTS
    version = f"daphnet_cross_subject_{args.architecture}_tcnm.v1"
    loso.EXPERIMENT_VERSION = version
    core.EXPERIMENT_VERSION = version
    return version


def build_protocol(args, version, dataset, point_stats, window_stats, scaler_metadata, clip_stats, device):
    payload = loso.build_protocol(
        args, dataset, point_stats, window_stats, scaler_metadata, clip_stats, device
    )
    payload.update(
        {
            "experiment_version": version,
            "outer_fold": args.test_subject,
            "architecture_variant": args.architecture,
            "split": {
                "strategy": "subject-disjoint outer test and inner validation",
                "train_subjects": list(loso.TRAIN_SUBJECTS),
                "validation_subject": args.validation_subject,
                "test_subject": args.test_subject,
                "validation_selection": (
                    "next cyclic FoG-positive subject; fixed before model training"
                ),
                "raw_support_overlap": False,
            },
            "leakage_controls": [
                "Robust scaler uses valid non-FoG points from training subjects only.",
                (
                    "GRU uses clean-normal training-subject windows only."
                    if args.architecture == "gru_residual"
                    else "No GRU or residual model is fitted in the raw-target ablation."
                ),
                f"{args.validation_subject} selects model epoch(s) and classification threshold.",
                f"{args.test_subject} is used once for final evaluation only.",
                "Train, validation, and test subjects are mutually disjoint.",
            ],
            "known_limitation": (
                "One deterministic validation-subject rotation and one random seed are reported. "
                "Overlapping 0.5-second decisions within a subject are not independent."
            ),
        }
    )
    payload["training"]["execution_scope"] = f"cross_subject_{args.architecture}"
    if args.architecture == "raw_target":
        payload.pop("normal_behaviour_model", None)
        payload.pop("residual", None)
        for key in ("normal_epochs_max", "normal_patience", "normal_learning_rate"):
            payload["training"].pop(key, None)
        payload["ablation"] = {
            "removed_modules": ["GRU normal-behaviour predictor", "standardized residual"],
            "classifier_input": "Robust-scaled raw target signal",
            "classifier_input_shape": ["batch", dataset.n_channels, core.TARGET_SAMPLES],
            "context_usage": (
                "Retained only for identical endpoints/window membership; not input to TCN-M."
            ),
        }
        payload["classifier"]["input"] = "Robust-scaled raw target"
    else:
        payload["normal_behaviour_model"]["training_input"] = (
            "clean-normal windows from training subjects only"
        )
    payload["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_utc", "environment", "protocol_fingerprint"}
        }
    )
    return payload


def write_summary(output_dir: Path, args, protocol, nbm_training, classifier_training, metrics):
    test = metrics["test"]
    model_text = (
        "GRU prediction -> standardized residual -> TCN-M"
        if args.architecture == "gru_residual"
        else "Robust-scaled raw target -> TCN-M"
    )
    gru_line = (
        f"- GRU maximum/best/completed epoch: {nbm_training['maximum_epochs']}/"
        f"{nbm_training['best_epoch']}/{nbm_training['epochs_completed']}.\n"
        if nbm_training is not None
        else "- GRU and residual modules removed.\n"
    )
    text = f"""# {args.test_subject} cross-subject fold

- Architecture: {model_text}.
- Train subjects: {', '.join(loso.TRAIN_SUBJECTS)}.
- Validation subject: {args.validation_subject}.
- Test subject: {args.test_subject}.
- Context/target/stride: {core.CONTEXT_SAMPLES/64:g}/{core.TARGET_SAMPLES/64:g}/{core.STRIDE_SAMPLES/64:g} seconds.
{gru_line}- TCN-M maximum/best/completed epoch: {classifier_training['maximum_epochs']}/{classifier_training['best_epoch']}/{classifier_training['epochs_completed']}.
- Validation-selected threshold: {classifier_training['selected_threshold']:.4f}.

| Test metric | Value |
|---|---:|
| Accuracy | {test['accuracy']:.6f} |
| FoG recall | {test['fog_recall']:.6f} |
| Specificity | {test['specificity']:.6f} |
| PR-AUC | {test['pr_auc']:.6f} |

Test confusion matrix: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}.
"""
    path = output_dir / "summary.md"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    version = configure(args)
    core.validate_args(args)
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
    protocol = build_protocol(
        args, version, dataset, point_stats, window_stats, scaler_metadata, clip_stats, device
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(scaler_metadata, output_dir / "scaler.json")
    core.write_csv(output_dir / "split_manifest.csv", core.split_manifest_rows(dataset, windows, split))
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train_window_index=split.train,
        validation_window_index=split.validation,
        test_window_index=split.test,
    )
    print(
        f"Protocol {protocol['protocol_fingerprint']}\narchitecture={args.architecture} "
        f"device={device} subjects={loso.subject_groups()} "
        f"window_counts={ {name: len(value) for name, value in split.as_dict().items()} }",
        flush=True,
    )
    if args.dry_run:
        atomic_json_dump(
            {"status": "dry_run_complete", "experiment_version": version,
             "protocol_fingerprint": protocol["protocol_fingerprint"]},
            output_dir / "DRY_RUN.json",
        )
        return

    nbm_training = None
    features: dict[str, dict[str, np.ndarray]] = {}
    if args.architecture == "gru_residual":
        nbm, nbm_training = core.train_nbm(
            args, dataset, windows, split, scaler, output_dir,
            protocol["protocol_fingerprint"], device,
        )
        core.plot_nbm_losses(output_dir, nbm_training)
        diagnostics: dict[str, Any] = {}
        for name, indices in split.as_dict().items():
            features[name], diagnostics[name] = core.extract_residuals(
                args, nbm, dataset, windows, indices, scaler, device
            )
        atomic_json_dump(diagnostics, output_dir / "residual_diagnostics.json")
        cache_name = "residual_cache.npz"
    else:
        diagnostics = {}
        for name, indices in split.as_dict().items():
            features[name], diagnostics[name] = raw.direct_target_features(
                args, dataset, windows, indices, scaler, device
            )
        atomic_json_dump(diagnostics, output_dir / "direct_input_diagnostics.json")
        cache_name = "raw_target_cache.npz"
    atomic_npz_save(
        output_dir / cache_name,
        **{
            f"{split_name}_{field}": values[field]
            for split_name, values in features.items()
            for field in ("residual", "y", "window_index")
        },
    )
    classifier_training, metrics = core.train_classifier(
        args, features, dataset, windows, output_dir,
        protocol["protocol_fingerprint"], device,
    )
    core.plot_classifier_losses(output_dir, classifier_training)
    core.plot_test_confusion_matrix(output_dir, metrics["test"], args.test_subject)
    write_summary(output_dir, args, protocol, nbm_training, classifier_training, metrics)
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": core.utc_now(),
            "experiment_version": version,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        },
        done_path,
    )
    test = metrics["test"]
    print(
        f"COMPLETE test_accuracy={test['accuracy']:.6f} "
        f"test_fog_recall={test['fog_recall']:.6f} "
        f"test_specificity={test['specificity']:.6f} "
        f"test_pr_auc={test['pr_auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
