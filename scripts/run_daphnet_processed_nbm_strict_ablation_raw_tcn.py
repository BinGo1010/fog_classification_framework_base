#!/usr/bin/env python
"""Strict NBM ablation on processed_NBM: centered scaled raw windows -> same TCN.

This runner is intentionally paired with
``run_daphnet_processed_nbm_centered_residual_tcn.py``.  It preserves the
authoritative three folds, role ranges, RobustScaler fit support, TCN code,
TCN initialization seeds, optimizer, loss, early stopping, and threshold rule.
Only the GRU-NBM reconstruction and standardized-residual transformation are
removed.  Role 5 is deliberately unused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    METRIC_KEYS,
    ROLES,
    SUBJECTS,
    ResidualTCNM,
    audit_protocol,
    choose_document_threshold,
    classifier_predict,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    metric_summary,
    prepare_nbm_windows,
    raw_windows,
    set_seed,
    train_tcn,
    write_csv,
    write_json,
)

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def save_figure_bundle(fig: Any, stem: Path) -> None:
    """Save a screen preview plus editable publication-safe vector versions."""
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_strict_ablation_centered_raw_tcn_roles8_seed20260807",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_centered_residual_tcn_roles8_seed20260807",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def initial_state_sha256(seed: int) -> str:
    """Deterministic fingerprint of the TCN parameters before optimization."""
    set_seed(seed)
    model = ResidualTCNM().cpu()
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().numpy()).tobytes())
    return digest.hexdigest()


def strict_contract(
    args: argparse.Namespace,
    reference_dir: Path,
    rows_by_fold: dict[int, Any],
    records: dict[str, Any],
) -> dict[str, Any]:
    reference_config = json.loads((reference_dir / "config.json").read_text(encoding="utf-8"))
    if tuple(reference_config["included_subjects"]) != SUBJECTS:
        raise AssertionError("included subjects differ from the complete-model run")
    if reference_config["seed"] != args.seed:
        raise AssertionError("base seed differs from the complete-model run")
    reference_tcn = reference_config["tcn"]
    expected = {
        "batch_size": 128,
        "max_epochs": args.tcn_max_epochs,
        "patience": args.tcn_patience,
        "train_roles": [6, 7],
        "validation_roles": [2, 3],
        "gradient_clip": 1.0,
    }
    for key, value in expected.items():
        if reference_tcn[key] != value:
            raise AssertionError(f"TCN contract mismatch for {key}: {reference_tcn[key]} != {value}")

    folds = []
    for fold, rows in rows_by_fold.items():
        role4 = rows.take_role(4)
        scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
        frozen = json.loads(
            (reference_dir / f"fold_{fold}" / "nbm_frozen.json").read_text(encoding="utf-8")
        )
        reference_scaler = frozen["scaler"]
        if not np.array_equal(scaler.median, np.asarray(reference_scaler["median"], dtype=np.float32)):
            raise AssertionError(f"fold {fold} scaler median differs from complete model")
        if not np.array_equal(scaler.iqr, np.asarray(reference_scaler["iqr"], dtype=np.float32)):
            raise AssertionError(f"fold {fold} scaler IQR differs from complete model")
        if unique_points != frozen["scaler_unique_raw_points"]:
            raise AssertionError(f"fold {fold} scaler support count differs")
        reference_metrics = json.loads(
            (reference_dir / f"fold_{fold}" / "metrics.json").read_text(encoding="utf-8")
        )
        n6 = int(np.sum(rows.role == 6))
        n7 = int(np.sum(rows.role == 7))
        pos_weight = n6 / n7
        if pos_weight != reference_metrics["tcn_training"]["pos_weight"]:
            raise AssertionError(f"fold {fold} pos_weight differs from complete model")
        tcn_seed = args.seed + fold + 100
        folds.append(
            {
                "fold": fold,
                "same_window_ids_by_role": {
                    str(role): int(np.sum(rows.role == role)) for role in ROLES
                },
                "scaler_exactly_matches_complete_model": True,
                "scaler_unique_raw_points": unique_points,
                "tcn_seed": tcn_seed,
                "tcn_initial_state_sha256": initial_state_sha256(tcn_seed),
                "pos_weight": pos_weight,
                "pos_weight_exactly_matches_complete_model": True,
            }
        )
    return {
        "reference_dir": str(reference_dir),
        "same_subjects": True,
        "same_three_outer_folds": True,
        "same_roles_6_7_train": True,
        "same_roles_2_3_validation_threshold": True,
        "same_roles_0_1_test": True,
        "same_tcn_class_and_training_function": True,
        "same_tcn_hyperparameters": True,
        "same_tcn_initialization_seeds": True,
        "same_threshold_function": True,
        "same_window_axis_centering": True,
        "role5_used": False,
        "only_removed_components": [
            "GRU reconstruction NBM",
            "role-5 residual bias b",
            "role-5 residual scale sigma",
            "standardized residual transformation and clipping",
            "post-residual centering (replaced by the same centering on scaled raw input)",
        ],
        "folds": folds,
    }


def plot_fold(fold_dir: Path, tcn_run: dict, confusion: list[list[int]]) -> None:
    history = tcn_run["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_weighted_bce"] for row in history], label="Roles 6/7 train")
    axes[0].plot(
        epochs,
        [row["validation_weighted_bce"] for row in history],
        label="Roles 2/3 validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE", title="Strict-ablation TCN loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_pr_auc"] for row in history])
    axes[1].axvline(tcn_run["best_epoch"], color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Epoch", ylabel="PR-AUC", title="Roles 2/3 model selection")
    axes[1].grid(alpha=0.25)
    save_figure_bundle(fig, fold_dir / "tcn_training_validation")
    plt.close(fig)

    cm = np.asarray(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 0.5 * cm.max() else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color=color)
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.set(xlabel="Predicted", ylabel="True", title="Strict-ablation permanent test")
    save_figure_bundle(fig, fold_dir / "test_confusion_matrix")
    plt.close(fig)


def run_fold(
    fold: int,
    rows: Any,
    records: dict[str, Any],
    output_dir: Path,
    reference_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    role4 = rows.take_role(4)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)

    # This is the complete model's pre-NBM tensor, sent directly to the same TCN.
    train_x = prepare_nbm_windows(scaler, raw_windows(records, role67), center=True)
    val_x = prepare_nbm_windows(scaler, raw_windows(records, role23), center=True)
    train_y = role67.label
    val_y = role23.label
    tcn_seed = args.seed + fold + 100
    tcn, tcn_run = train_tcn(
        train_x,
        train_y,
        val_x,
        val_y,
        fold_dir,
        device,
        tcn_seed,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
    )
    val_true, val_prob = classifier_predict(tcn, val_x, val_y, device)
    threshold, val_metrics = choose_document_threshold(val_true, val_prob)

    # The permanent test tensor is not materialized until best checkpoint and
    # validation-only threshold have both been frozen.
    test_rows = rows.take_role(0, 1)
    test_x = prepare_nbm_windows(scaler, raw_windows(records, test_rows), center=True)
    test_true, test_prob = classifier_predict(tcn, test_x, test_rows.label, device)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(np.int8)
    subject_metrics = {}
    for subject in SUBJECTS:
        mask = test_rows.subject_id == subject
        subject_metrics[subject] = binary_metrics(test_true[mask], test_prob[mask], threshold)

    reference_metrics = json.loads(
        (reference_dir / f"fold_{fold}" / "metrics.json").read_text(encoding="utf-8")
    )
    comparison = {
        key: {
            "complete_nbm_residual_tcn": reference_metrics["test"][key],
            "strict_nbm_ablation_raw_tcn": test_metrics[key],
            "ablation_minus_complete": (
                test_metrics[key] - reference_metrics["test"][key]
                if test_metrics[key] is not None and reference_metrics["test"][key] is not None
                else None
            ),
        }
        for key in METRIC_KEYS
    }
    result = {
        "fold": fold,
        "tcn_seed": tcn_seed,
        "tcn_initial_state_sha256": initial_state_sha256(tcn_seed),
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_points,
        "role5_windows_unused": int(np.sum(rows.role == 5)),
        "input": "RobustScaler(role4) -> per-window/per-axis time centering -> same TCN",
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "validation": val_metrics,
        "test": test_metrics,
        "test_by_subject": subject_metrics,
        "tcn_training": {key: value for key, value in tcn_run.items() if key != "history"},
        "comparison_to_complete_model": comparison,
    }
    write_json(fold_dir / "metrics.json", result)
    write_csv(
        fold_dir / "predictions.csv",
        [
            {
                "fold": fold,
                "subject_id": str(test_rows.subject_id[i]),
                "record_id": str(test_rows.record_id[i]),
                "window_id": str(test_rows.window_id[i]),
                "start_index": int(test_rows.start[i]),
                "end_index_exclusive": int(test_rows.end[i]),
                "role_code": int(test_rows.role[i]),
                "y_true": int(test_true[i]),
                "fog_probability": float(test_prob[i]),
                "threshold": threshold,
                "y_pred": int(test_pred[i]),
            }
            for i in range(len(test_rows))
        ],
    )
    write_csv(
        fold_dir / "test_by_subject.csv",
        [
            {
                "subject_id": subject,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
            for subject, metrics in subject_metrics.items()
        ],
    )
    write_csv(
        fold_dir / "test_confusion_matrix.csv",
        [
            {"true\\pred": "non-FoG", "non-FoG": test_metrics["tn"], "FoG": test_metrics["fp"]},
            {"true\\pred": "FoG", "non-FoG": test_metrics["fn"], "FoG": test_metrics["tp"]},
        ],
    )
    np.savez_compressed(
        fold_dir / "test_probabilities.npz",
        y_true=test_true,
        y_prob=test_prob,
        y_pred=test_pred,
        subject_id=test_rows.subject_id,
        window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    plot_fold(fold_dir, tcn_run, test_metrics["confusion_matrix"])
    print(
        f"FOLD {fold} COMPLETE threshold={threshold:.2f} acc={test_metrics['accuracy']:.6f} "
        f"recall={test_metrics['sensitivity']:.6f} specificity={test_metrics['specificity']:.6f} "
        f"pr_auc={test_metrics['auprc']:.6f} cm={test_metrics['confusion_matrix']}",
        flush=True,
    )
    return result


def plot_paired_comparison(output_dir: Path, results: list[dict[str, Any]]) -> None:
    panels = (
        ("accuracy", "Accuracy"),
        ("sensitivity", "FoG recall"),
        ("specificity", "Specificity"),
        ("auprc", "PR-AUC"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        complete = np.asarray(
            [r["comparison_to_complete_model"][key]["complete_nbm_residual_tcn"] for r in results]
        )
        ablated = np.asarray(
            [r["comparison_to_complete_model"][key]["strict_nbm_ablation_raw_tcn"] for r in results]
        )
        for fold, (left, right) in enumerate(zip(complete, ablated)):
            ax.plot([0, 1], [left, right], color="#9aa0a6", linewidth=1, marker="o", markersize=4)
            ax.text(1.03, right, f"F{fold}", va="center", fontsize=6)
        ax.scatter([0, 1], [complete.mean(), ablated.mean()], color="#1f77b4", s=45, zorder=3)
        ax.set_xticks([0, 1], ["NBM residual", "NBM ablated"])
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Strict paired NBM ablation on the same permanent test windows", fontsize=9)
    save_figure_bundle(fig, output_dir / "strict_ablation_paired_comparison")
    plt.close(fig)


def render_existing_plots(output_dir: Path, folds: list[int]) -> None:
    results = []
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold}"
        metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        with (fold_dir / "logs" / "tcn_history.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            history = list(csv.DictReader(handle))
        for row in history:
            row["epoch"] = int(row["epoch"])
            row["train_weighted_bce"] = float(row["train_weighted_bce"])
            row["validation_weighted_bce"] = float(row["validation_weighted_bce"])
            row["validation_pr_auc"] = float(row["validation_pr_auc"])
        plot_fold(
            fold_dir,
            {**metrics["tcn_training"], "history": history},
            metrics["test"]["confusion_matrix"],
        )
        results.append(metrics)
        print(f"RENDERED fold={fold}", flush=True)
    if len(results) == 3:
        plot_paired_comparison(output_dir, results)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    done_path = output_dir / "DONE.json"
    if args.render_only:
        folds = [args.fold] if args.fold is not None else [0, 1, 2]
        render_existing_plots(output_dir, folds)
        return
    if done_path.exists() and not args.overwrite:
        raise FileExistsError(f"completed output exists: {done_path}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in (0, 1, 2)}
    source_audit = audit_protocol(data_dir, rows_by_fold, records)
    contract = strict_contract(args, reference_dir, rows_by_fold, records)
    write_json(output_dir / "preflight_source_audit.json", source_audit)
    write_json(output_dir / "strict_ablation_contract.json", contract)
    config = {
        "experiment": "processed_NBM_strict_NBM_ablation_centered_scaled_raw_TCN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "reference_complete_model_dir": str(reference_dir),
        "included_subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "seed": args.seed,
        "device": str(device),
        "window": {"samples": 128, "seconds": 2.0, "stride_samples": 64, "stride_seconds": 1.0},
        "scaler": "same per-channel median/IQR fitted on same unique role-4 raw points",
        "input": "scaled raw X, centered independently over time for every window/channel",
        "role5": "unused",
        "tcn": {
            "same_implementation_as_complete_model": True,
            "architecture": "causal residual TCN 9/32/64/64/128, kernel 3, dilations 1/2/4/8, GAP, dropout 0.3, one logit",
            "train_roles": [6, 7],
            "validation_roles": [2, 3],
            "test_roles": [0, 1],
            "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "batch_size": 128,
            "max_epochs": args.tcn_max_epochs,
            "patience": args.tcn_patience,
            "monitor": "role-2/3 PR-AUC",
            "gradient_clip": 1.0,
            "initialization_seed_by_fold": {str(f): args.seed + f + 100 for f in (0, 1, 2)},
        },
        "threshold": "roles 2/3 balanced accuracy over 0.05..0.95 step 0.01; ties FoG F1 then higher threshold",
        "test_timing": "roles 0/1 materialized and inferred only after best TCN and threshold are frozen",
        "figure_contract": {
            "core_conclusion": "Estimate the contribution of NBM residualization under a paired, sample-matched classifier comparison.",
            "evidence": "fold metrics, paired deltas, subject metrics, confusion matrices, training curves",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "exports": ["300-dpi PNG", "editable SVG", "editable-text PDF"],
        },
    }
    write_json(output_dir / "config.json", config)
    print("PREFLIGHT STRICT CONTRACT PASS", flush=True)
    if args.dry_run:
        write_json(output_dir / "DRY_RUN.json", {"status": "complete"})
        return

    folds = [args.fold] if args.fold is not None else [0, 1, 2]
    results = [
        run_fold(fold, rows_by_fold[fold], records, output_dir, reference_dir, args, device)
        for fold in folds
    ]
    fold_rows = [
        {"fold": r["fold"], "threshold": r["threshold"], **r["test"]} for r in results
    ]
    aggregate = metric_summary([r["test"] for r in results])
    subject_aggregate = {
        subject: metric_summary([r["test_by_subject"][subject] for r in results])
        for subject in SUBJECTS
    }
    reference_aggregate = metric_summary(
        [
            {
                key: r["comparison_to_complete_model"][key]["complete_nbm_residual_tcn"]
                for key in METRIC_KEYS
            }
            for r in results
        ]
    )
    deltas = {
        key: aggregate[key]["mean"] - reference_aggregate[key]["mean"] for key in METRIC_KEYS
    }
    summary = {
        "strict_ablation_fold_results": results,
        "strict_ablation_test_mean_std": aggregate,
        "complete_model_test_mean_std": reference_aggregate,
        "strict_ablation_minus_complete_mean": deltas,
        "subject_test_mean_std": subject_aggregate,
        "note": "The permanent test windows are identical across folds; mean/std quantify model-fold variability.",
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)
    write_csv(
        output_dir / "comparison_to_complete_model.csv",
        [
            {
                "metric": key,
                "complete_mean": reference_aggregate[key]["mean"],
                "ablation_mean": aggregate[key]["mean"],
                "ablation_minus_complete": deltas[key],
            }
            for key in METRIC_KEYS
        ],
    )
    write_csv(
        output_dir / "subject_metrics_mean.csv",
        [
            {
                "subject_id": subject,
                **{f"{key}_mean": subject_aggregate[subject][key]["mean"] for key in METRIC_KEYS},
                **{f"{key}_std": subject_aggregate[subject][key]["std"] for key in METRIC_KEYS},
            }
            for subject in SUBJECTS
        ],
    )
    plot_paired_comparison(output_dir, results)
    write_json(
        done_path,
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "strict_ablation_test_mean_std": aggregate,
            "strict_ablation_minus_complete_mean": deltas,
        },
    )
    print("ALL COMPLETE", json.dumps({"ablation": aggregate, "delta": deltas}), flush=True)


if __name__ == "__main__":
    main()
