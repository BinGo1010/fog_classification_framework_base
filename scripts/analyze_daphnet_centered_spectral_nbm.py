#!/usr/bin/env python
"""Audit reconstruction quality and residual separation of the centered spectral NBM.

This script intentionally evaluates the NBM output itself.  It does not use TCN-M
probabilities or a classifier-selected threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
RUN_PATTERN = (
    "daphnet_{subject}_spectral_gru_nbm_blocked5_"
    "window_axis_centered_tcnm_seed42"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "daphnet_centered_spectral_nbm_reconstruction_audit_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-15:
        raise ValueError(f"Cannot divide by {denominator!r}")
    return float(numerator / denominator)


def smooth_l1_mean(residual: np.ndarray) -> float:
    absolute = np.abs(residual.astype(np.float64, copy=False))
    losses = np.where(absolute < 1.0, 0.5 * absolute**2, absolute - 0.5)
    return float(np.mean(losses))


def reconstruction_metrics(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    median_template: np.ndarray,
    mean_template: np.ndarray,
) -> dict[str, float]:
    observed64 = observed.astype(np.float64, copy=False)
    reconstructed64 = reconstructed.astype(np.float64, copy=False)
    residual = observed64 - reconstructed64
    median_template_residual = observed64 - median_template[None]
    mean_template_residual = observed64 - mean_template[None]
    absolute = np.abs(residual)
    squared = residual**2
    denominator = float(np.sum(np.abs(observed64)))
    centered_sum_squares = float(np.sum((observed64 - np.mean(observed64)) ** 2))
    model_huber = smooth_l1_mean(residual)
    template_huber = smooth_l1_mean(median_template_residual)
    model_mae = float(np.mean(absolute))
    model_mse = float(np.mean(squared))
    median_template_mae = float(np.mean(np.abs(median_template_residual)))
    mean_template_mse = float(np.mean(mean_template_residual**2))

    observed_flat = observed64.reshape(len(observed64), -1)
    reconstructed_flat = reconstructed64.reshape(len(reconstructed64), -1)
    cosine_denominator = np.linalg.norm(observed_flat, axis=1) * np.linalg.norm(
        reconstructed_flat, axis=1
    )
    cosine = np.divide(
        np.sum(observed_flat * reconstructed_flat, axis=1),
        cosine_denominator,
        out=np.zeros_like(cosine_denominator),
        where=cosine_denominator > 1e-15,
    )

    return {
        "mae": model_mae,
        "rmse": float(np.sqrt(model_mse)),
        "relative_mae": safe_ratio(float(np.sum(absolute)), denominator),
        "global_r2": float(1.0 - np.sum(squared) / centered_sum_squares),
        "mean_window_cosine": float(np.mean(cosine)),
        "smooth_l1": model_huber,
        "median_template_mae": median_template_mae,
        "mae_skill_vs_train_non_fog_median_template": float(
            1.0 - model_mae / median_template_mae
        ),
        "mean_template_mse": mean_template_mse,
        "mse_skill_vs_train_non_fog_mean_template": float(
            1.0 - model_mse / mean_template_mse
        ),
        "template_smooth_l1": template_huber,
        "smooth_l1_skill_vs_train_non_fog_median_template": float(
            1.0 - model_huber / template_huber
        ),
    }


def cohen_d(non_fog_scores: np.ndarray, fog_scores: np.ndarray) -> float:
    n0 = len(non_fog_scores)
    n1 = len(fog_scores)
    denominator_df = n0 + n1 - 2
    if denominator_df <= 0:
        return 0.0
    non_fog_sum_squares = (
        (n0 - 1) * float(np.var(non_fog_scores, ddof=1)) if n0 > 1 else 0.0
    )
    fog_sum_squares = (
        (n1 - 1) * float(np.var(fog_scores, ddof=1)) if n1 > 1 else 0.0
    )
    pooled_variance = (non_fog_sum_squares + fog_sum_squares) / denominator_df
    if pooled_variance <= 1e-15:
        return 0.0
    return float((np.mean(fog_scores) - np.mean(non_fog_scores)) / np.sqrt(pooled_variance))


def separation_metrics(residual: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    labels = labels.astype(np.int8, copy=False)
    non_fog = labels == 0
    fog = labels == 1
    if not np.any(non_fog) or not np.any(fog):
        raise ValueError("Both non-FoG and FoG windows are required")

    absolute = np.abs(residual.astype(np.float64, copy=False))
    score = np.mean(absolute, axis=(1, 2))
    non_fog_score = score[non_fog]
    fog_score = score[fog]
    auc = float(roc_auc_score(labels, score))
    ap = float(average_precision_score(labels, score))
    prevalence = float(np.mean(labels))
    non_fog_q95 = float(np.quantile(non_fog_score, 0.95))
    cell_non_fog = np.mean(absolute[non_fog], axis=0)
    cell_fog = np.mean(absolute[fog], axis=0)

    return {
        "windows": int(len(labels)),
        "non_fog_windows": int(np.sum(non_fog)),
        "fog_windows": int(np.sum(fog)),
        "fog_prevalence": prevalence,
        "non_fog_score_mean": float(np.mean(non_fog_score)),
        "fog_score_mean": float(np.mean(fog_score)),
        "fog_to_non_fog_mean_ratio": safe_ratio(
            float(np.mean(fog_score)), float(np.mean(non_fog_score))
        ),
        "non_fog_score_median": float(np.median(non_fog_score)),
        "fog_score_median": float(np.median(fog_score)),
        "fog_to_non_fog_median_ratio": safe_ratio(
            float(np.median(fog_score)), float(np.median(non_fog_score))
        ),
        "magnitude_roc_auc": auc,
        "rank_biserial": float(2.0 * auc - 1.0),
        "magnitude_pr_auc": ap,
        "pr_auc_prevalence_lift": safe_ratio(ap, prevalence),
        "cohen_d": cohen_d(non_fog_score, fog_score),
        "axis_frequency_cells_fog_mean_gt_non_fog_fraction": float(
            np.mean(cell_fog > cell_non_fog)
        ),
        "non_fog_score_q95": non_fog_q95,
        "fog_fraction_above_non_fog_q95": float(np.mean(fog_score > non_fog_q95)),
    }


def flattened_row(
    subject: str,
    config: dict[str, Any],
    train_oof_nf: dict[str, float],
    validation_nf: dict[str, float],
    validation_separation: dict[str, float | int],
    test_nf: dict[str, float],
    test_separation: dict[str, float | int],
) -> dict[str, Any]:
    training = config["gru_nbm"]["final_training"]
    row: dict[str, Any] = {
        "subject": subject,
        "nbm_parameters": training["parameter_count"],
        "nbm_train_non_fog_windows": training["training_windows"],
        "nbm_early_stop_non_fog_windows": training["validation_windows"],
        "nbm_best_epoch": training["best_epoch"],
        "nbm_epochs_completed": training["epochs_completed"],
        "nbm_best_validation_smooth_l1": training["best_validation_smooth_l1"],
        "train_oof_non_fog_mae": train_oof_nf["mae"],
        "validation_non_fog_mae": validation_nf["mae"],
        "test_non_fog_reconstruction_mae": test_nf["mae"],
        "test_non_fog_reconstruction_rmse": test_nf["rmse"],
        "test_non_fog_relative_mae": test_nf["relative_mae"],
        "test_non_fog_global_r2": test_nf["global_r2"],
        "test_non_fog_mean_window_cosine": test_nf["mean_window_cosine"],
        "test_non_fog_smooth_l1": test_nf["smooth_l1"],
        "test_non_fog_median_template_mae": test_nf["median_template_mae"],
        "test_non_fog_mae_skill_vs_template": test_nf[
            "mae_skill_vs_train_non_fog_median_template"
        ],
        "test_non_fog_mse_skill_vs_template": test_nf[
            "mse_skill_vs_train_non_fog_mean_template"
        ],
        "test_non_fog_nbm_skill_vs_template": test_nf[
            "smooth_l1_skill_vs_train_non_fog_median_template"
        ],
    }
    for prefix, metrics in (
        ("validation", validation_separation),
        ("test", test_separation),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "Subject",
        "Test NF/FoG",
        "NF MAE",
        "Rel. MAE",
        "Cosine",
        "MAE skill",
        "MSE skill",
        "FoG MAE",
        "Mean ratio",
        "ROC-AUC",
        "PR-AUC/base",
        "Cells FoG>NF",
        "FoG>NF q95",
    )
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                (
                    row["subject"],
                    f'{row["test_non_fog_windows"]}/{row["test_fog_windows"]}',
                    f'{row["test_non_fog_reconstruction_mae"]:.4f}',
                    f'{row["test_non_fog_relative_mae"]:.3f}',
                    f'{row["test_non_fog_mean_window_cosine"]:.3f}',
                    f'{row["test_non_fog_mae_skill_vs_template"]:.3f}',
                    f'{row["test_non_fog_mse_skill_vs_template"]:.3f}',
                    f'{row["test_fog_score_mean"]:.4f}',
                    f'{row["test_fog_to_non_fog_mean_ratio"]:.3f}',
                    f'{row["test_magnitude_roc_auc"]:.3f}',
                    f'{row["test_magnitude_pr_auc"]:.3f}/{row["test_fog_prevalence"]:.3f}',
                    f'{row["test_axis_frequency_cells_fog_mean_gt_non_fog_fraction"]:.1%}',
                    f'{row["test_fog_fraction_above_non_fog_q95"]:.1%}',
                )
            )
            + "|"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for subject in SUBJECTS:
        run_dir = outputs_dir / RUN_PATTERN.format(subject=subject.lower())
        config = read_json(run_dir / "config.json")
        if not config["windowing"]["per_window_per_axis_temporal_centering"]["enabled"]:
            raise AssertionError(f"{subject}: window-axis centering is not enabled")
        if config["residual"]["standardization"] is not None:
            raise AssertionError(f"{subject}: residual is unexpectedly standardized")
        if not config["gru_nbm"]["input_output"].startswith(
            "[B,9,65] log-power spectrum"
        ):
            raise AssertionError(f"{subject}: unexpected NBM input/output")

        npz_path = run_dir / "spectral_residuals_crossfit.npz"
        with np.load(npz_path) as arrays:
            train_y = arrays["train_y"]
            train_observed = arrays["train_observed_log_power"]
            train_reconstructed = arrays["train_reconstructed_log_power"]
            train_nf = train_y == 0
            train_non_fog_observed = train_observed[train_nf].astype(np.float64)
            median_template = np.median(train_non_fog_observed, axis=0)
            mean_template = np.mean(train_non_fog_observed, axis=0)
            train_oof_nf_metrics = reconstruction_metrics(
                train_observed[train_nf],
                train_reconstructed[train_nf],
                median_template,
                mean_template,
            )

            split_details: dict[str, Any] = {}
            for split in ("validation", "test"):
                observed = arrays[f"{split}_observed_log_power"]
                reconstructed = arrays[f"{split}_reconstructed_log_power"]
                residual = arrays[f"{split}_signed_residual"]
                labels = arrays[f"{split}_y"]
                identity_error = float(np.max(np.abs(residual - (observed - reconstructed))))
                if identity_error > 5e-7:
                    raise AssertionError(
                        f"{subject} {split}: saved residual identity error={identity_error}"
                    )
                non_fog = labels == 0
                split_details[split] = {
                    "non_fog_reconstruction": reconstruction_metrics(
                        observed[non_fog],
                        reconstructed[non_fog],
                        median_template,
                        mean_template,
                    ),
                    "residual_separation": separation_metrics(residual, labels),
                }

        validation = split_details["validation"]
        test = split_details["test"]
        row = flattened_row(
            subject,
            config,
            train_oof_nf_metrics,
            validation["non_fog_reconstruction"],
            validation["residual_separation"],
            test["non_fog_reconstruction"],
            test["residual_separation"],
        )
        rows.append(row)
        details[subject] = {
            "run_dir": str(run_dir),
            "train_oof_non_fog_reconstruction": train_oof_nf_metrics,
            **split_details,
        }

    macro_keys = (
        "test_non_fog_reconstruction_mae",
        "test_non_fog_relative_mae",
        "test_non_fog_global_r2",
        "test_non_fog_mean_window_cosine",
        "test_non_fog_mae_skill_vs_template",
        "test_non_fog_mse_skill_vs_template",
        "test_non_fog_nbm_skill_vs_template",
        "test_fog_to_non_fog_mean_ratio",
        "test_fog_to_non_fog_median_ratio",
        "test_magnitude_roc_auc",
        "test_magnitude_pr_auc",
        "test_pr_auc_prevalence_lift",
        "test_axis_frequency_cells_fog_mean_gt_non_fog_fraction",
        "test_fog_fraction_above_non_fog_q95",
    )
    macro = {key: float(np.mean([float(row[key]) for row in rows])) for key in macro_keys}
    validation_macro_keys = (
        "validation_fog_to_non_fog_mean_ratio",
        "validation_magnitude_roc_auc",
        "validation_magnitude_pr_auc",
        "validation_pr_auc_prevalence_lift",
        "validation_axis_frequency_cells_fog_mean_gt_non_fog_fraction",
        "validation_fog_fraction_above_non_fog_q95",
    )
    validation_macro = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in validation_macro_keys
    }
    counts = {
        "subjects_with_fog_mean_residual_gt_non_fog": int(
            sum(float(row["test_fog_to_non_fog_mean_ratio"]) > 1.0 for row in rows)
        ),
        "subjects_with_residual_auc_gt_chance": int(
            sum(float(row["test_magnitude_roc_auc"]) > 0.5 for row in rows)
        ),
        "subjects_with_nbm_positive_mae_skill": int(
            sum(
                float(row["test_non_fog_mae_skill_vs_template"]) > 0.0
                for row in rows
            )
        ),
        "subjects_with_nbm_positive_mse_skill": int(
            sum(
                float(row["test_non_fog_mse_skill_vs_template"]) > 0.0
                for row in rows
            )
        ),
        "subjects": len(rows),
    }

    csv_path = output_dir / "per_subject_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "scope": (
            "Centered spectral GRU-NBM runs used in the prior best-NBM versus "
            "Raw-TCN comparison; spectral robust standardization is disabled."
        ),
        "score_definition": "window mean absolute raw log-power residual over 9x65 cells",
        "template_definition": (
            "per-axis/per-frequency median or mean observed spectrum of outer-training non-FoG windows"
        ),
        "important_limitations": [
            "Test windows overlap because the 2 s window advances by 1 s.",
            "S06 has one FoG test window and S09 has five; their FoG estimates are unstable.",
            "Train reconstructions are out-of-fold, whereas validation/test use the final NBM.",
            "One seed and one chronological split were evaluated.",
        ],
        "validation_macro_subject_mean": validation_macro,
        "test_macro_subject_mean": macro,
        "counts": counts,
        "per_subject": rows,
        "details": details,
    }
    json_path = output_dir / "audit.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    summary_lines = [
        "# Centered spectral GRU-NBM reconstruction audit (seed 42)",
        "",
        "The table uses the held-out test split and evaluates the NBM directly; no TCN-M score is used.",
        "Residual score is the per-window mean absolute raw log-power residual over all 9x65 cells.",
        "MAE skill is `1 - NBM MAE / train-non-FoG-median-template MAE`.",
        "MSE skill is `1 - NBM MSE / train-non-FoG-mean-template MSE`.",
        "Positive skill means the conditional NBM beats the corresponding fixed normal-spectrum baseline.",
        "",
        markdown_table(rows),
        "",
        "## Subject-macro diagnostics",
        "",
        "### Validation",
        "",
        *[f"- {key}: {value:.6f}" for key, value in validation_macro.items()],
        "",
        "### Test",
        "",
        *[f"- {key}: {value:.6f}" for key, value in macro.items()],
        *[f"- {key}: {value}" for key, value in counts.items()],
        "",
        "## Limits",
        "",
        "- Test windows overlap (2 s windows, 1 s stride), so they are not IID replicates.",
        "- S06 has only one FoG test window and S09 has five.",
        "- Train residuals are produced by fold NBMs; validation/test residuals use the final NBM.",
        "- Results are from one seed and one chronological split per subject.",
        "",
    ]
    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "macro": macro, "counts": counts}))


if __name__ == "__main__":
    main()
