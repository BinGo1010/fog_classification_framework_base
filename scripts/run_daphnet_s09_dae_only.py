#!/usr/bin/env python
"""Train the 64 Hz denoising autoencoder on an audited within-subject split.

This is a thin subject adapter around the audited S01 DAE implementation.  It
keeps the architecture, corruption, loss, optimizer, scheduler, and windowing
unchanged while replacing all subject-specific data and split functions with
the transparent chronological protocol configured by ``--subject``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_s01_dae_tcnm as base  # noqa: E402
import run_daphnet_s09_gru_h200_tcnm as s09  # noqa: E402
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint  # noqa: E402


EXPERIMENT_VERSION = "daphnet_single_subject_dae_only.v1"
_BASE_BUILD_PROTOCOL = base.build_protocol
_BASE_TRAIN_DAE = base.train_dae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a converged clean-normal single-subject DAE",
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dae-epochs", type=int, default=300)
    parser.add_argument("--dae-patience", type=int, default=30)
    parser.add_argument("--dae-lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dae-dropout", type=float, default=0.10)
    # Unused in DAE-only mode, but required by the shared argument validator.
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
    args = parser.parse_args()
    args.dae_only = True
    return args


def build_protocol(
    args: argparse.Namespace,
    dataset,
    point_stats: dict[str, Any],
    window_stats: dict[str, Any],
    scaler,
    scaler_fit_windows: int,
    corruption,
    device,
) -> dict[str, Any]:
    payload = _BASE_BUILD_PROTOCOL(
        args,
        dataset,
        point_stats,
        window_stats,
        scaler,
        scaler_fit_windows,
        corruption,
        device,
    )
    payload.update(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "subject": s09.SUBJECT_ID,
            "records": [record.record_id for record in dataset.records],
            "split": {
                "strategy": "chronological record/block split with disjoint raw support",
                "train": (
                    f"complete records {list(s09.TRAIN_RECORDS)} plus "
                    f"{s09.CUT_RECORD}[0,{s09.CUT_SAMPLE}); complete window "
                    f"support must end at or before sample {s09.CUT_SAMPLE}"
                ),
                "validation": (
                    f"{s09.CUT_RECORD}[{s09.CUT_SAMPLE},end); complete window "
                    f"support must begin at or after sample {s09.CUT_SAMPLE}"
                ),
                "test": f"all eligible windows of {s09.TEST_RECORD}",
                "ignored_records": list(s09.IGNORED_RECORDS),
                "cut_record": s09.CUT_RECORD,
                "cut_sample": s09.CUT_SAMPLE,
                "raw_train_validation_support_overlap": False,
                "test_used_for_fitting_or_selection": False,
                "cut_selection_disclosure": (
                    "The nominal 70 percent chronological pre-test cut was rounded "
                    "to the 64-sample grid and selected between FOG events so train "
                    "and validation retain both classes. Test was not inspected to "
                    "choose the cut."
                ),
            },
            "leakage_controls": [
                f"Z-score statistics and DAE weights use {s09.SUBJECT_ID} clean-normal training windows only.",
                f"Clean-normal {s09.SUBJECT_ID} validation windows select the DAE epoch.",
                f"{s09.TEST_RECORD} is not used by this DAE-only training stage.",
                "Train and validation context-target supports are raw-sample disjoint.",
            ],
            "convergence_acceptance_rule": {
                "metric": "clean validation combined DAE loss",
                "minimum_improvement": 1e-8,
                "required_terminal_condition": "early_stopped_after_full_patience",
                "maximum_epochs": args.dae_epochs,
                "patience": args.dae_patience,
                "stopped_before_maximum": True,
                "finite_history": True,
                "best_checkpoint_metric_reproduced": True,
                "minimum_learning_rate_reductions": 2,
                "failure_policy": (
                    "A maximum-epoch run with continuing validation improvement is "
                    "not accepted as converged and cannot be consumed downstream."
                ),
            },
            "interpretation_limits": [
                "This DAE reconstructs an observed two-second target; it is not a future predictor.",
                f"The within-{s09.SUBJECT_ID} validation split is chronological rather than an independent subject.",
                "Some held-out records contain very few positive windows, so per-subject recall can be highly uncertain.",
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


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    training: dict[str, Any],
) -> None:
    first = training["history"][0]
    final = training["history"][-1]
    text = f"""# {s09.SUBJECT_ID} DAE-only training result

- Architecture input/output: `[B,9,128]`, 64 Hz, observed-target denoising reconstruction.
- Clean-normal train/validation windows: {training['train_windows']} / {training['validation_windows']}.
- Maximum epochs / patience: {training['maximum_epochs']} / {training['early_stopping_patience']}.
- Epochs completed / best epoch: {training['epochs_completed']} / {training['best_epoch']}.
- Epochs observed after the best epoch: {training['epochs_after_best']}.
- Best clean-validation combined loss: {training['best_validation_total_loss']:.9f}.
- Stop status: `{training['convergence_status']}`.

| Loss | Epoch 1 | Final epoch |
|---|---:|---:|
| Train combined | {first['train_total_loss']:.9f} | {final['train_total_loss']:.9f} |
| Validation combined | {first['validation_total_loss']:.9f} | {final['validation_total_loss']:.9f} |

Downstream use is permitted only when the stop status is
`early_stopped_after_full_patience`. Protocol fingerprint:
`{protocol['protocol_fingerprint']}`.
"""
    temporary = output_dir / f".summary.md.tmp-{os.getpid()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")


def train_dae(*args, **kwargs):
    model, training = _BASE_TRAIN_DAE(*args, **kwargs)
    namespace = args[0] if args else kwargs["args"]
    validation_clean = args[2] if len(args) > 2 else kwargs["validation_clean"]
    device = args[5] if len(args) > 5 else kwargs["device"]
    validation_loader = base.array_loader(
        validation_clean,
        namespace.batch_size,
        shuffle=False,
        num_workers=namespace.num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        recomputed_validation, _ = base.dae_epoch(
            model,
            validation_loader,
            device,
            amp=namespace.amp,
        )
    recomputed_best_loss = float(recomputed_validation["total"])
    best_loss_matches = math.isclose(
        recomputed_best_loss,
        float(training["best_validation_total_loss"]),
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    finite_history = all(
        math.isfinite(float(value))
        for row in training["history"]
        for key, value in row.items()
        if key.endswith("_loss") or "learning_rate" in key
    )
    lr_reductions = sum(
        float(row["next_learning_rate"])
        < float(row["learning_rate_used"]) - 1e-15
        for row in training["history"]
    )
    accepted = bool(
        training["convergence_status"]
        == "early_stopped_after_full_patience"
        and training["epochs_after_best"] >= training["early_stopping_patience"]
        and training["epochs_completed"] < training["maximum_epochs"]
        and lr_reductions >= 2
        and finite_history
        and best_loss_matches
    )
    output_dir = args[3] if len(args) > 3 else kwargs["output_dir"]
    atomic_json_dump(
        {
            "accepted": accepted,
            "metric": "clean validation combined DAE loss",
            "minimum_improvement": 1e-8,
            "maximum_epochs": training["maximum_epochs"],
            "patience": training["early_stopping_patience"],
            "best_epoch": training["best_epoch"],
            "epochs_completed": training["epochs_completed"],
            "epochs_after_best": training["epochs_after_best"],
            "stopped_before_maximum": (
                training["epochs_completed"] < training["maximum_epochs"]
            ),
            "learning_rate_reductions": lr_reductions,
            "minimum_required_learning_rate_reductions": 2,
            "finite_history": finite_history,
            "best_validation_total_loss": training[
                "best_validation_total_loss"
            ],
            "recomputed_best_validation_total_loss": recomputed_best_loss,
            "best_checkpoint_metric_reproduced": best_loss_matches,
            "status": training["convergence_status"],
        },
        Path(output_dir) / "convergence_audit.json",
    )
    if not accepted:
        failed_conditions = []
        if training["convergence_status"] != "early_stopped_after_full_patience":
            failed_conditions.append("early_stopped_after_full_patience")
        if training["epochs_after_best"] < training["early_stopping_patience"]:
            failed_conditions.append("full_post_best_patience")
        if training["epochs_completed"] >= training["maximum_epochs"]:
            failed_conditions.append("stopped_before_maximum")
        if lr_reductions < 2:
            failed_conditions.append("minimum_learning_rate_reductions")
        if not finite_history:
            failed_conditions.append("finite_history")
        if not best_loss_matches:
            failed_conditions.append("best_checkpoint_metric_reproduced")
        raise RuntimeError(
            "DAE convergence audit failed: " + ", ".join(failed_conditions)
        )
    return model, training


def configure_subject_adapter() -> None:
    global EXPERIMENT_VERSION
    EXPERIMENT_VERSION = f"daphnet_{s09.SUBJECT_ID.lower()}_dae_only.v1"
    base.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    base.parse_args = parse_args
    base.build_protocol = build_protocol
    base.train_dae = train_dae
    base.write_dae_only_summary = write_summary
    base.core.SUBJECT_ID = s09.SUBJECT_ID
    base.core.load_s01_dataset = s09.load_dataset
    base.core.make_split = s09.make_split
    base.core.point_statistics = s09.point_statistics
    base.core.window_statistics = s09.window_statistics
    base.core.normal_support_indices = s09.normal_support_indices


def main() -> None:
    args = parse_args()
    s09.configure_subject(args.subject)
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "outputs"
            / (
                f"daphnet_{args.subject.lower()}_dae_only_"
                "s09protocol_max300_patience30_seed42"
            )
        )
    configure_subject_adapter()
    base.parse_args = lambda: args
    base.main()


if __name__ == "__main__":
    main()
