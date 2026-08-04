#!/usr/bin/env python
"""Freeze a converged subject DAE, retain residuals, and train converged TCN-M."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_s01_pretrained_dae_tcnm as base  # noqa: E402
import run_daphnet_s09_gru_h200_tcnm as s09  # noqa: E402
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint  # noqa: E402


EXPERIMENT_VERSION = "daphnet_single_subject_pretrained_dae_tcnm.v1"
_BASE_BUILD_PROTOCOL = base.build_protocol
_BASE_TRAIN_CLASSIFIER = base.train_classifier
_BASE_LOAD_FROZEN_DAE = base.load_frozen_dae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a converged residual TCN-M from a frozen subject DAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subject", choices=sorted(s09.SPLIT_CONFIGS), default="S09")
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
    parser.add_argument("--dae-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=200)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_protocol(
    args: argparse.Namespace,
    dataset,
    windows,
    split,
    source_config: dict[str, Any],
    source_training: dict[str, Any],
    source_done: dict[str, Any],
    device,
) -> dict[str, Any]:
    payload = _BASE_BUILD_PROTOCOL(
        args,
        dataset,
        windows,
        split,
        source_config,
        source_training,
        source_done,
        device,
    )
    calibration_windows = int(
        source_config["normalization"]["fit_windows"]
    )
    payload.update(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "subject": s09.SUBJECT_ID,
            "records": [record.record_id for record in dataset.records],
            "split": {
                "same_as_source_dae": True,
                "strategy": "chronological record/block split with disjoint raw support",
                "train": (
                    f"complete records {list(s09.TRAIN_RECORDS)} plus "
                    f"{s09.CUT_RECORD} before sample {s09.CUT_SAMPLE}"
                ),
                "validation": f"{s09.CUT_RECORD} from sample {s09.CUT_SAMPLE}",
                "test": f"all eligible windows of {s09.TEST_RECORD}",
                "ignored_records": list(s09.IGNORED_RECORDS),
                "counts": {
                    name: int(len(indices))
                    for name, indices in split.as_dict().items()
                },
                "test_used_for_fitting_or_selection": False,
            },
            "residual": {
                "error": "target_scaled - frozen_dae_reconstruction_scaled",
                "sigma": (
                    f"sqrt(mean(error^2 over {calibration_windows} clean-normal "
                    "training windows, axis=window) + 1e-6), separately by "
                    "channel and within-window time position"
                ),
                "sigma_shape": [1, dataset.n_channels, base.core.TARGET_SAMPLES],
                "formula": "clip(error / fixed_sigma, -12, 12)",
                "fixed_sigma_fit_split": "clean-normal train only",
                "test_used_to_calibrate_sigma": False,
                "full_process_arrays_saved": True,
            },
            "classifier": {
                **payload["classifier"],
                "early_stopping": (
                    "validation PR-AUC with minimum improvement 1e-5; "
                    f"patience {args.classifier_patience}"
                ),
            },
            "convergence_acceptance_rule": {
                "dae": {
                    "source_status": "early_stopped_after_full_patience",
                    "full_post_best_patience": True,
                    "stopped_before_maximum": True,
                    "finite_history": True,
                    "best_checkpoint_metric_reproduced": True,
                    "minimum_learning_rate_reductions": 2,
                },
                "classifier_metric": "validation PR-AUC",
                "classifier_minimum_improvement": 1e-5,
                "classifier_maximum_epochs": args.classifier_epochs,
                "classifier_patience": args.classifier_patience,
                "classifier_required_epochs_after_best": args.classifier_patience,
                "classifier_stopped_before_maximum": True,
                "classifier_finite_history": True,
                "classifier_best_checkpoint_metric_reproduced": True,
                "failure_policy": (
                    "No DONE artifact is written unless both DAE and TCN-M have "
                    "observed their full post-best patience."
                ),
            },
            "leakage_controls": [
                "The source DAE, its scaler, and declared hashes are frozen and verified.",
                f"Fixed sigma uses source-matched {s09.SUBJECT_ID} clean-normal training windows only.",
                f"TCN-M weights use {s09.SUBJECT_ID} training residuals and labels only.",
                f"{s09.SUBJECT_ID} validation selects TCN-M epoch and decision threshold.",
                f"{s09.TEST_RECORD} is used only for final evaluation after all selection.",
            ],
            "interpretation_limits": [
                "The DAE reconstructs the already observed target rather than forecasting it.",
                "Fixed sigma and training residuals are in-sample for the DAE.",
                "Some held-out records contain very few positive windows, so recall can be highly uncertain.",
            ],
        }
    )
    payload["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_utc", "environment", "protocol_fingerprint"}
        }
    )
    return payload


def load_frozen_dae(dae_dir: Path, device):
    result = _BASE_LOAD_FROZEN_DAE(dae_dir, device)
    _, _, config, training, _ = result
    expected_records = list(s09.EXPECTED_RECORDS)
    if config.get("subject") != s09.SUBJECT_ID:
        raise ValueError(
            f"source DAE subject is {config.get('subject')!r}, expected {s09.SUBJECT_ID}"
        )
    if config.get("records") != expected_records:
        raise ValueError(
            f"source DAE records are {config.get('records')!r}, expected "
            f"{expected_records!r}"
        )
    if training.get("epochs_after_best", -1) < training.get(
        "early_stopping_patience", 0
    ):
        raise ValueError("source DAE did not observe full post-best patience")
    source_audit = json.loads(
        (Path(dae_dir) / "convergence_audit.json").read_text(encoding="utf-8")
    )
    if source_audit.get("accepted") is not True:
        raise ValueError("source DAE convergence audit was not accepted")
    return result


def train_classifier(
    args: argparse.Namespace,
    features,
    dataset,
    windows,
    output_dir: Path,
    protocol_fingerprint: str,
    device,
):
    training, metrics = _BASE_TRAIN_CLASSIFIER(
        args,
        features,
        dataset,
        windows,
        output_dir,
        protocol_fingerprint,
        device,
    )
    epochs_after_best = int(
        training["epochs_completed"] - training["best_epoch"]
    )
    full_patience = epochs_after_best >= int(args.classifier_patience)
    stopped_before_maximum = (
        training["epochs_completed"] < training["maximum_epochs"]
    )
    finite_history = all(
        math.isfinite(float(value))
        for row in training["history"]
        for key, value in row.items()
        if key in {
            "train_bce",
            "train_pr_auc",
            "validation_bce",
            "validation_pr_auc",
        }
    )
    reproduced_best_pr_auc = math.isclose(
        float(metrics["validation"]["pr_auc"]),
        float(training["best_validation_pr_auc"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    classifier_accepted = bool(
        full_patience
        and stopped_before_maximum
        and finite_history
        and reproduced_best_pr_auc
    )
    if classifier_accepted:
        status = "early_stopped_after_full_patience"
    elif not finite_history:
        status = "rejected_non_finite_history"
    elif not reproduced_best_pr_auc:
        status = "rejected_best_checkpoint_metric_mismatch"
    elif not stopped_before_maximum:
        status = "maximum_epoch_reached"
    elif not full_patience:
        status = "insufficient_post_best_patience"
    elif training["best_epoch"] == training["epochs_completed"]:
        status = "maximum_epoch_reached_while_validation_still_improving"
    else:
        status = "rejected_unspecified_convergence_condition"
    training.update(
        {
            "epochs_after_best": epochs_after_best,
            "full_post_best_patience_observed": full_patience,
            "convergence_status": status,
        }
    )
    atomic_json_dump(training, output_dir / "classifier_training.json")
    dae_training = json.loads(
        (Path(args.dae_dir) / "dae_training.json").read_text(encoding="utf-8")
    )
    source_dae_audit = json.loads(
        (Path(args.dae_dir) / "convergence_audit.json").read_text(
            encoding="utf-8"
        )
    )
    dae_accepted = source_dae_audit.get("accepted") is True
    atomic_json_dump(
        {
            "accepted": bool(dae_accepted and classifier_accepted),
            "dae": {
                "accepted": dae_accepted,
                "metric": "clean validation combined DAE loss",
                "maximum_epochs": dae_training["maximum_epochs"],
                "patience": dae_training["early_stopping_patience"],
                "best_epoch": dae_training["best_epoch"],
                "epochs_completed": dae_training["epochs_completed"],
                "epochs_after_best": dae_training["epochs_after_best"],
                "status": dae_training["convergence_status"],
            },
            "classifier": {
                "accepted": classifier_accepted,
                "metric": "validation PR-AUC",
                "minimum_improvement": 1e-5,
                "maximum_epochs": args.classifier_epochs,
                "patience": args.classifier_patience,
                "best_epoch": training["best_epoch"],
                "epochs_completed": training["epochs_completed"],
                "epochs_after_best": epochs_after_best,
                "stopped_before_maximum": stopped_before_maximum,
                "finite_history": finite_history,
                "best_validation_pr_auc": training[
                    "best_validation_pr_auc"
                ],
                "recomputed_best_validation_pr_auc": metrics["validation"][
                    "pr_auc"
                ],
                "best_checkpoint_metric_reproduced": reproduced_best_pr_auc,
                "status": status,
            },
        },
        output_dir / "convergence_audit.json",
    )
    base.core.plot_classifier_losses(output_dir, training)
    base.core.plot_test_confusion_matrix(output_dir, metrics["test"], s09.SUBJECT_ID)
    if not classifier_accepted:
        failed_conditions = []
        if not full_patience:
            failed_conditions.append("full_post_best_patience")
        if not stopped_before_maximum:
            failed_conditions.append("stopped_before_maximum")
        if not finite_history:
            failed_conditions.append("finite_history")
        if not reproduced_best_pr_auc:
            failed_conditions.append("best_checkpoint_metric_reproduced")
        raise RuntimeError(
            "TCN-M convergence audit failed: " + ", ".join(failed_conditions)
        )
    return training, metrics


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    sigma: dict[str, Any],
    residual: dict[str, Any],
    classifier: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    validation = metrics["validation"]
    test = metrics["test"]
    text = f"""# {s09.SUBJECT_ID} frozen DAE residual + TCN-M result

## Convergence

- DAE best/completed epoch: {protocol['source_dae']['best_epoch']} / {protocol['source_dae']['epochs_completed']}.
- DAE status: `{protocol['source_dae']['convergence_status']}`.
- TCN-M best/completed/max epoch: {classifier['best_epoch']} / {classifier['epochs_completed']} / {classifier['maximum_epochs']}.
- TCN-M epochs after best / patience: {classifier['epochs_after_best']} / {classifier['patience']}.
- TCN-M status: `{classifier['convergence_status']}`.

## Classification

| Split | Accuracy | FoG recall | Specificity | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['pr_auc']:.6f} | {validation['roc_auc']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['pr_auc']:.6f} | {test['roc_auc']:.6f} |

- Validation-selected threshold: {classifier['selected_threshold']:.4f}.
- Test confusion matrix: `[[{test['tn']}, {test['fp']}], [{test['fn']}, {test['tp']}]]`.
- Fixed-sigma calibration windows: {sigma['calibration_windows']}.
- Test residual RMS, Non-FoG/FoG: {residual['test']['non_fog']['residual_clipped_rms']:.6f} / {residual['test']['fog']['residual_clipped_rms']:.6f}.

`residual_process.npz` retains the scaled target, reconstruction, error,
unclipped and clipped standardized residual, latent vector, labels, and window
indices for train, validation, and test.

Per-subject recall must be interpreted together with the number of positive
test windows and the confusion matrix.
"""
    temporary = output_dir / f".summary.md.tmp-{os.getpid()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")


def configure_subject_adapter() -> None:
    global EXPERIMENT_VERSION
    EXPERIMENT_VERSION = (
        f"daphnet_{s09.SUBJECT_ID.lower()}_pretrained_dae_tcnm.v1"
    )
    base.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    base.parse_args = parse_args
    base.build_protocol = build_protocol
    base.load_frozen_dae = load_frozen_dae
    base.train_classifier = train_classifier
    base.write_summary = write_summary
    base.core.SUBJECT_ID = s09.SUBJECT_ID
    base.core.load_s01_dataset = s09.load_dataset
    base.core.make_split = s09.make_split
    base.core.point_statistics = s09.point_statistics
    base.core.window_statistics = s09.window_statistics
    base.core.normal_support_indices = s09.normal_support_indices


def main() -> None:
    args = parse_args()
    s09.configure_subject(args.subject)
    if args.dae_dir is None:
        if args.subject == "S09":
            args.dae_dir = (
                REPO_ROOT
                / "outputs"
                / "daphnet_s09_dae_only_max300_patience30_seed42"
            )
        else:
            args.dae_dir = (
                REPO_ROOT
                / "outputs"
                / (
                    f"daphnet_{args.subject.lower()}_dae_only_"
                    "s09protocol_max300_patience30_seed42"
                )
            )
    if args.output_dir is None:
        if args.subject == "S09":
            args.output_dir = (
                REPO_ROOT
                / "outputs"
                / "daphnet_s09_dae_converged_tcnm_max200_patience20_seed42"
            )
        else:
            args.output_dir = (
                REPO_ROOT
                / "outputs"
                / (
                    f"daphnet_{args.subject.lower()}_dae_converged_"
                    "tcnm_s09protocol_max200_patience20_seed42"
                )
            )
    configure_subject_adapter()
    base.parse_args = lambda: args
    base.main()


if __name__ == "__main__":
    main()
