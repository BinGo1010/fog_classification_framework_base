#!/usr/bin/env python
"""Audit and summarize eight within-subject DAE-residual TCN-M runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump, sha256_file  # noqa: E402
import run_daphnet_s09_gru_h200_tcnm as split_core  # noqa: E402


INCLUDED = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
ALL_SUBJECTS = tuple(f"S{number:02d}" for number in range(1, 11))
EXCLUDED = {
    "S04": "No sample-level FoG in any valid record.",
    "S10": "No sample-level FoG in any valid record.",
}
REQUIRED_DAE = {
    "config.json",
    "convergence_audit.json",
    "dae_best.pt",
    "dae_training.json",
    "split_indices.npz",
}
REQUIRED_PIPELINE = {
    "classifier_best.pt",
    "classifier_training.json",
    "config.json",
    "convergence_audit.json",
    "fixed_sigma.npy",
    "metrics.json",
    "predictions.npz",
    "residual_process.npz",
    "split_indices.npz",
    "test_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_single_subject_dae_tcnm_s09protocol_summary"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def subject_dirs(root: Path, subject: str) -> tuple[Path, Path]:
    lower = subject.lower()
    if subject == "S09":
        return (
            root / "daphnet_s09_dae_only_max300_patience30_seed42",
            root / "daphnet_s09_dae_converged_tcnm_max200_patience20_seed42",
        )
    return (
        root
        / f"daphnet_{lower}_dae_only_s09protocol_max300_patience30_seed42",
        root
        / (
            f"daphnet_{lower}_dae_converged_"
            "tcnm_s09protocol_max200_patience20_seed42"
        ),
    )


def verify_done(directory: Path, required: set[str]) -> dict[str, Any]:
    done_path = directory / "DONE.json"
    if not done_path.is_file():
        raise FileNotFoundError(done_path)
    done = load_json(done_path)
    if done.get("status") != "complete":
        raise ValueError(f"incomplete run: {directory}")
    declared = done.get("artifacts", {})
    missing = required - set(declared)
    if missing:
        raise ValueError(f"{directory} DONE lacks required artifacts: {sorted(missing)}")
    for name, expected in declared.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
    return done


def audit_residual_process(path: Path) -> dict[str, float]:
    maximum_error_difference = 0.0
    maximum_unclipped_difference = 0.0
    maximum_clipped_difference = 0.0
    with np.load(path, allow_pickle=False) as archive:
        sigma = archive["fixed_sigma"]
        if sigma.shape != (1, 9, 128) or not np.isfinite(sigma).all():
            raise ValueError(f"invalid fixed sigma in {path}")
        for split in ("train", "validation", "test"):
            target = archive[f"{split}_target_scaled"]
            reconstruction = archive[f"{split}_reconstruction_scaled"]
            error = archive[f"{split}_error_scaled"]
            unclipped = archive[f"{split}_residual_unclipped"]
            clipped = archive[f"{split}_residual_clipped"]
            maximum_error_difference = max(
                maximum_error_difference,
                float(np.max(np.abs(error - (target - reconstruction)))),
            )
            maximum_unclipped_difference = max(
                maximum_unclipped_difference,
                float(np.max(np.abs(unclipped - error / sigma))),
            )
            maximum_clipped_difference = max(
                maximum_clipped_difference,
                float(np.max(np.abs(clipped - np.clip(unclipped, -12, 12)))),
            )
    if any(
        value != 0.0
        for value in (
            maximum_error_difference,
            maximum_unclipped_difference,
            maximum_clipped_difference,
        )
    ):
        raise AssertionError(f"residual formula mismatch in {path}")
    return {
        "error_formula_max_abs": maximum_error_difference,
        "unclipped_formula_max_abs": maximum_unclipped_difference,
        "clip_formula_max_abs": maximum_clipped_difference,
    }


def matrix_from_csv(path: Path) -> tuple[np.ndarray, int]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            matrix[int(row["y_true"]), int(row["y_pred"])] += 1
            rows += 1
    return matrix, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "RESULTS_DONE.json"
    if done_path.exists():
        raise FileExistsError(done_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    included_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, Any] = {}
    residual_audits: dict[str, Any] = {}
    for subject in INCLUDED:
        dae_dir, pipeline_dir = subject_dirs(outputs_root, subject)
        dae_done = verify_done(dae_dir, REQUIRED_DAE)
        pipeline_done = verify_done(pipeline_dir, REQUIRED_PIPELINE)
        dae_config = load_json(dae_dir / "config.json")
        pipeline_config = load_json(pipeline_dir / "config.json")
        dae_convergence = load_json(dae_dir / "convergence_audit.json")
        convergence = load_json(pipeline_dir / "convergence_audit.json")
        dae_training = load_json(dae_dir / "dae_training.json")
        classifier = load_json(pipeline_dir / "classifier_training.json")
        metrics = load_json(pipeline_dir / "metrics.json")
        test = metrics["test"]
        if dae_config.get("subject") != subject or pipeline_config.get("subject") != subject:
            raise ValueError(f"subject metadata mismatch for {subject}")
        if dae_convergence.get("accepted") is not True or convergence.get("accepted") is not True:
            raise ValueError(f"convergence not accepted for {subject}")
        with np.load(dae_dir / "split_indices.npz", allow_pickle=False) as source_split:
            with np.load(pipeline_dir / "split_indices.npz", allow_pickle=False) as target_split:
                for key in source_split.files:
                    if key not in target_split or not np.array_equal(
                        source_split[key], target_split[key]
                    ):
                        raise ValueError(f"source/downstream split mismatch: {subject}/{key}")
        matrix, prediction_rows = matrix_from_csv(pipeline_dir / "test_predictions.csv")
        stored_matrix = np.asarray(test["confusion_matrix"], dtype=np.int64)
        if prediction_rows != int(test["n"]) or not np.array_equal(matrix, stored_matrix):
            raise AssertionError(f"prediction/metric mismatch for {subject}")
        residual_audits[subject] = audit_residual_process(
            pipeline_dir / "residual_process.npz"
        )
        split_core.configure_subject(subject)
        included_rows.append(
            {
                "subject": subject,
                "status": "complete",
                "test_record": split_core.TEST_RECORD,
                "ignored_records": ",".join(split_core.IGNORED_RECORDS),
                "test_windows": int(test["n"]),
                "test_non_fog_windows": int(test["n_normal"]),
                "test_fog_windows": int(test["n_fog"]),
                "dae_max_epochs": int(dae_training["maximum_epochs"]),
                "dae_best_epoch": int(dae_training["best_epoch"]),
                "dae_epochs_completed": int(dae_training["epochs_completed"]),
                "dae_patience": int(dae_training["early_stopping_patience"]),
                "tcnm_max_epochs": int(classifier["maximum_epochs"]),
                "tcnm_best_epoch": int(classifier["best_epoch"]),
                "tcnm_epochs_completed": int(classifier["epochs_completed"]),
                "tcnm_patience": int(classifier["patience"]),
                "threshold": float(classifier["selected_threshold"]),
                "accuracy": float(test["accuracy"]),
                "fog_recall": float(test["fog_recall"]),
                "specificity": float(test["specificity"]),
                "precision": float(test["precision"]),
                "pr_auc": float(test["pr_auc"]),
                "tn": int(test["tn"]),
                "fp": int(test["fp"]),
                "fn": int(test["fn"]),
                "tp": int(test["tp"]),
            }
        )
        source_hashes[subject] = {
            "dae_done_sha256": sha256_file(dae_dir / "DONE.json"),
            "pipeline_done_sha256": sha256_file(pipeline_dir / "DONE.json"),
            "metrics_sha256": sha256_file(pipeline_dir / "metrics.json"),
            "dae_declared_artifacts": len(dae_done["artifacts"]),
            "pipeline_declared_artifacts": len(pipeline_done["artifacts"]),
        }

    pooled_matrix = np.asarray(
        [
            [sum(row["tn"] for row in included_rows), sum(row["fp"] for row in included_rows)],
            [sum(row["fn"] for row in included_rows), sum(row["tp"] for row in included_rows)],
        ],
        dtype=np.int64,
    )
    total_windows = int(pooled_matrix.sum())
    if total_windows != 3341:
        raise AssertionError(f"unexpected pooled test windows: {total_windows}")
    if int(pooled_matrix[1].sum()) != 377:
        raise AssertionError("unexpected pooled FoG test windows")
    macro_metrics = {
        key: float(np.mean([row[key] for row in included_rows]))
        for key in ("accuracy", "fog_recall", "specificity", "precision", "pr_auc")
    }
    aggregate = {
        "experiment": "within-subject DAE reconstruction residual plus TCN-M",
        "included_subjects": list(INCLUDED),
        "excluded_subjects": EXCLUDED,
        "uniform_training": {
            "dae_max_epochs": 300,
            "dae_patience": 30,
            "tcnm_max_epochs": 200,
            "tcnm_patience": 20,
            "validation_only_epoch_and_threshold_selection": True,
        },
        "subject_macro": macro_metrics,
        "pooled_confusion_matrix": pooled_matrix.tolist(),
        "pooled_test_windows": total_windows,
        "runs": included_rows,
        "residual_formula_audits": residual_audits,
        "source_hashes": source_hashes,
    }
    atomic_json_dump(aggregate, output_dir / "aggregate.json")

    csv_rows: list[dict[str, Any]] = []
    by_subject = {row["subject"]: row for row in included_rows}
    for subject in ALL_SUBJECTS:
        if subject in by_subject:
            csv_rows.append(dict(by_subject[subject]))
        else:
            blank = {key: "" for key in included_rows[0]}
            blank.update(
                {
                    "subject": subject,
                    "status": "excluded_no_fog",
                    "test_record": "not_applicable",
                }
            )
            csv_rows.append(blank)
    write_csv(output_dir / "summary.csv", csv_rows)

    lines = [
        "# Daphnet single-subject DAE residual + TCN-M summary",
        "",
        "S04 and S10 were not trained because their valid data contain no FoG.",
        "",
        "| Subject | Test N/F | DAE best/stop | TCN-M best/stop | Threshold | Accuracy | FoG recall | Specificity | PR-AUC | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subject in ALL_SUBJECTS:
        if subject not in by_subject:
            lines.append(f"| {subject} | excluded: no FoG | — | — | — | — | — | — | — | — |")
            continue
        row = by_subject[subject]
        lines.append(
            f"| {subject} | {row['test_non_fog_windows']}/{row['test_fog_windows']} "
            f"| {row['dae_best_epoch']}/{row['dae_epochs_completed']} "
            f"| {row['tcnm_best_epoch']}/{row['tcnm_epochs_completed']} "
            f"| {row['threshold']:.2f} | {row['accuracy']:.4f} "
            f"| {row['fog_recall']:.4f} | {row['specificity']:.4f} "
            f"| {row['pr_auc']:.4f} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    lines.extend(
        [
            "",
            "## Subject-macro across the eight trained subjects",
            "",
            *[f"- {key}: {value:.6f}" for key, value in macro_metrics.items()],
            "",
            f"Pooled confusion matrix: `{pooled_matrix.tolist()}`.",
            "",
            "S06 has only one positive test window and S09 has five; their recall is highly uncertain.",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 5, figsize=(19.0, 7.8), constrained_layout=True)
    for axis, subject in zip(axes.flat, ALL_SUBJECTS):
        if subject not in by_subject:
            axis.axis("off")
            axis.text(
                0.5,
                0.55,
                subject,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                transform=axis.transAxes,
            )
            axis.text(
                0.5,
                0.38,
                "Excluded\nNo valid FoG",
                ha="center",
                va="center",
                fontsize=12,
                transform=axis.transAxes,
            )
            continue
        row = by_subject[subject]
        matrix = np.asarray(
            [[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.int64
        )
        local_maximum = int(matrix.max())
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(local_maximum, 1))
        axis.set_xticks([0, 1], labels=["Non-FoG", "FoG"])
        axis.set_yticks([0, 1], labels=["Non-FoG", "FoG"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{subject} | threshold={row['threshold']:.2f}\n"
            f"Acc={row['accuracy']:.3f}, Recall={row['fog_recall']:.3f}",
            fontsize=10.5,
        )
        cutoff = local_maximum / 2.0
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="white" if matrix[i, j] > cutoff else "black",
                )
    figure.suptitle(
        "Daphnet within-subject DAE residual + TCN-M confusion matrices",
        fontsize=16,
    )
    figure.savefig(output_dir / "confusion_matrices.png", dpi=180)
    plt.close(figure)

    atomic_json_dump(
        {
            "status": "complete",
            "included_subjects": list(INCLUDED),
            "excluded_subjects": EXCLUDED,
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        },
        done_path,
    )
    print(output_dir / "summary.md")
    print(output_dir / "confusion_matrices.png")


if __name__ == "__main__":
    main()
