from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_full_subject_nbm_residual_binary as exp
import run_daphnet_full_subject_tcndae_inceptiontime as tcndae
import run_daphnet_nbm_tcdae_three_rounds as reconstruction
import analyze_tcndae_inceptiontime_train_val_test as classifier_analysis


DEFAULT_ROOT = (
    ROOT / "outputs" / "daphnet_full_subject_tcndae_inceptiontime_server_v1"
    / "full_subject_binary_experiment"
)
SETS = ("nbm_train", "record_validation", "outer_test")
SET_LABELS = {"nbm_train": "NBM train", "record_validation": "Record validation",
              "outer_test": "Outer test"}
CHANNEL_NAMES = (
    "ankle_forward", "ankle_vertical", "ankle_lateral",
    "thigh_forward", "thigh_vertical", "thigh_lateral",
    "trunk_forward", "trunk_vertical", "trunk_lateral",
)
QUALITY_METRICS = (
    "improvement_pct", "median_corr", "median_nrmse", "median_amplitude_ratio",
    "spectral_cosine_distance", "residual_rms", "latent_between_window_variance",
    "raw_latent_distance_corr",
)


def predict(model: torch.nn.Module, inputs: np.ndarray, device: torch.device,
            batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    reconstructions: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(inputs[start:start + batch_size].transpose(0, 2, 1))
            ).float().to(device)
            output, latent = model(batch)
            reconstructions.append(output.transpose(1, 2).cpu().numpy().astype(np.float32))
            latents.append(latent.cpu().numpy().astype(np.float32))
    return np.concatenate(reconstructions), np.concatenate(latents)


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = tcndae.TCNDAE().to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload


def deterministic_subset(count: int, maximum: int = 256) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64))


def spectral_cosine(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    frequencies = np.fft.rfftfreq(actual.shape[1], d=1.0 / exp.FS)
    selected = (frequencies >= 0.5) & (frequencies <= 10.0)
    actual_power = np.square(np.abs(np.fft.rfft(actual, axis=1)[:, selected, :]))
    predicted_power = np.square(np.abs(np.fft.rfft(predicted, axis=1)[:, selected, :]))
    numerator = np.sum(actual_power * predicted_power, axis=1)
    denominator = np.linalg.norm(actual_power, axis=1) * np.linalg.norm(predicted_power, axis=1)
    similarity = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                           where=denominator > 1e-12)
    return np.clip(1.0 - similarity, 0.0, 1.0)


def quality_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    rules = (
        (float(metrics["improvement_pct"]) >= 95.0, "improvement<95"),
        (float(metrics["median_corr"]) >= 0.80, "Pearson<0.80"),
        (float(metrics["median_nrmse"]) <= 0.60, "NRMSE>0.60"),
        (0.75 <= float(metrics["median_amplitude_ratio"]) <= 1.25, "amplitude_outside"),
        (float(metrics["raw_latent_distance_corr"]) >= 0.50, "latent_distance<0.50"),
    )
    failures.extend(label for passed, label in rules if not passed)
    return not failures, failures


def summarize_set(actual: np.ndarray, predicted: np.ndarray, latent: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrays = reconstruction.metric_arrays(actual, predicted)
    residual = actual - predicted
    residual_window_rms = np.sqrt(np.mean(np.square(residual), axis=(1, 2)))
    spectrum = spectral_cosine(actual, predicted)
    subset = deterministic_subset(len(actual))
    raw_upper = reconstruction.pairwise_distances(actual[subset])[1]
    latent_upper = reconstruction.pairwise_distances(latent[subset])[1]
    distance_corr = reconstruction.safe_corr(raw_upper, latent_upper) if len(raw_upper) >= 2 else math.nan
    mse = float(np.mean(np.square(residual)))
    zero_mse = float(np.mean(np.square(actual)))
    per_window_nrmse = np.median(arrays["nrmse"], axis=1)
    metrics: dict[str, Any] = {
        "n_windows": len(actual),
        "nbm_mse": mse,
        "zero_mse": zero_mse,
        "improvement_pct": 100.0 * (zero_mse - mse) / max(zero_mse, 1e-12),
        "median_corr": float(np.median(arrays["correlation"])),
        "corr_p10": float(np.percentile(arrays["correlation"], 10)),
        "median_nrmse": float(np.median(arrays["nrmse"])),
        "nrmse_p90": float(np.percentile(per_window_nrmse, 90)),
        "nrmse_p95": float(np.percentile(per_window_nrmse, 95)),
        "median_amplitude_ratio": float(np.median(arrays["amplitude_ratio"])),
        "amplitude_ratio_p10": float(np.percentile(arrays["amplitude_ratio"], 10)),
        "amplitude_ratio_p90": float(np.percentile(arrays["amplitude_ratio"], 90)),
        "spectral_cosine_distance": float(np.median(spectrum)),
        "spectral_cosine_distance_p90": float(np.percentile(spectrum, 90)),
        "residual_rms": float(np.median(residual_window_rms)),
        "residual_rms_p90": float(np.percentile(residual_window_rms, 90)),
        "residual_rms_p95": float(np.percentile(residual_window_rms, 95)),
        "latent_variance": float(np.var(latent)),
        "latent_between_window_variance": float(
            np.mean(np.var(latent.reshape(len(latent), -1), axis=0))
        ) if len(latent) >= 2 else 0.0,
        "latent_active_fraction": float(
            np.mean(np.var(latent.reshape(len(latent), -1), axis=0) > 1e-6)
        ) if len(latent) >= 2 else 0.0,
        "raw_latent_distance_corr": float(distance_corr),
        "latent_distance_sample_windows": len(subset),
    }
    passed, failures = quality_gate(metrics)
    metrics["safety_pass"] = passed
    metrics["failure_reasons"] = ";".join(failures)
    channel_rows: list[dict[str, Any]] = []
    for channel, name in enumerate(CHANNEL_NAMES):
        channel_residual = residual[:, :, channel]
        rms = np.sqrt(np.mean(np.square(channel_residual), axis=1))
        absolute = np.abs(channel_residual).ravel()
        signed = channel_residual.ravel()
        channel_rows.append({
            "channel_index": channel, "channel": name,
            "residual_rms_q10": float(np.percentile(rms, 10)),
            "residual_rms_q50": float(np.percentile(rms, 50)),
            "residual_rms_q90": float(np.percentile(rms, 90)),
            "residual_rms_q95": float(np.percentile(rms, 95)),
            "abs_residual_q50": float(np.percentile(absolute, 50)),
            "abs_residual_q90": float(np.percentile(absolute, 90)),
            "abs_residual_q95": float(np.percentile(absolute, 95)),
            "signed_residual_q05": float(np.percentile(signed, 5)),
            "signed_residual_q50": float(np.percentile(signed, 50)),
            "signed_residual_q95": float(np.percentile(signed, 95)),
            "median_corr": float(np.median(arrays["correlation"][:, channel])),
            "median_nrmse": float(np.median(arrays["nrmse"][:, channel])),
            "median_amplitude_ratio": float(np.median(arrays["amplitude_ratio"][:, channel])),
            "spectral_cosine_distance": float(np.median(spectrum[:, channel])),
        })
    return metrics, channel_rows


def positions_for_keys(keys: Sequence[str], lookup: dict[str, int]) -> np.ndarray:
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise KeyError(f"missing {len(missing)} window keys, first={missing[0]}")
    return np.asarray([lookup[key] for key in keys], dtype=np.int64)


def manifest_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def audit_outer_fold(root: Path, subject: str, fold_id: str, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_dir = root / "splits" / "outer_folds" / subject / fold_id
    arrays = dict(np.load(fold_dir / "representations.npz", allow_pickle=False))
    manifest = pd.read_csv(fold_dir / "split_manifest.csv")
    train_manifest = manifest[manifest["role"] == "outer_train"].reset_index(drop=True)
    test_manifest = manifest[manifest["role"] == "outer_test"].reset_index(drop=True)
    if len(train_manifest) != len(arrays["train_x"]) or len(test_manifest) != len(arrays["test_x"]):
        raise AssertionError(f"manifest/array length mismatch {subject}/{fold_id}")
    train_lookup = {str(key): index for index, key in enumerate(train_manifest["window_key"])}
    rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for held in range(exp.INNER_K):
        model_dir = fold_dir / "nbm_oof_models" / f"inner{held}"
        model, checkpoint = load_model(model_dir / "nbm_best.pt", device)
        split = json.loads((model_dir / "split_manifest.json").read_text(encoding="utf-8"))
        training_indices = positions_for_keys(checkpoint["train_window_keys"], train_lookup)
        held_keys = set(str(key) for key in split["held_window_keys"])
        held_strict_keys = [str(row.window_key) for row in train_manifest.itertuples(index=False)
                            if str(row.window_key) in held_keys
                            and manifest_bool(row.strict_clean_nonfog)]
        validation_indices = positions_for_keys(held_strict_keys, train_lookup)
        for set_name, indices in (("nbm_train", training_indices),
                                  ("record_validation", validation_indices)):
            actual = arrays["train_x"][indices]
            predicted, latent = predict(model, actual, device)
            metrics, channels = summarize_set(actual, predicted, latent)
            common = {"subject_id": subject, "fold_id": fold_id, "set": set_name,
                      "inner_model": held, "model_role": "oof_tcndae"}
            rows.append(common | metrics)
            channel_rows.extend(common | value for value in channels)
        del model
    test_indices = np.flatnonzero(
        test_manifest["strict_clean_nonfog"].map(manifest_bool).to_numpy(dtype=bool)
    )
    if len(test_indices) == 0:
        raise ValueError(f"no strict Non-FoG outer-test windows {subject}/{fold_id}")
    final_model, _ = load_model(fold_dir / "final_nbm_models" / "nbm_best.pt", device)
    actual = arrays["test_x"][test_indices]
    predicted, latent = predict(final_model, actual, device)
    metrics, channels = summarize_set(actual, predicted, latent)
    common = {"subject_id": subject, "fold_id": fold_id, "set": "outer_test",
              "inner_model": "final", "model_role": "final_tcndae"}
    rows.append(common | metrics)
    channel_rows.extend(common | value for value in channels)
    return rows, channel_rows


def aggregate(raw: pd.DataFrame, channels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in raw.columns if column not in {
        "subject_id", "fold_id", "set", "inner_model", "model_role",
        "safety_pass", "failure_reasons",
    }]
    fold = raw.groupby(["subject_id", "fold_id", "set"], as_index=False)[numeric].median(numeric_only=True)
    gate_values = fold.apply(lambda row: quality_gate(row.to_dict()), axis=1)
    fold["safety_pass"] = [value[0] for value in gate_values]
    fold["failure_reasons"] = [";".join(value[1]) for value in gate_values]
    subject = fold.groupby(["subject_id", "set"], as_index=False)[numeric].median(numeric_only=True)
    subject_gate = subject.apply(lambda row: quality_gate(row.to_dict()), axis=1)
    subject["safety_pass"] = [value[0] for value in subject_gate]
    subject["failure_reasons"] = [";".join(value[1]) for value in subject_gate]
    overall = subject.groupby("set", as_index=False)[numeric].median(numeric_only=True)
    overall_gate = overall.apply(lambda row: quality_gate(row.to_dict()), axis=1)
    overall["safety_pass"] = [value[0] for value in overall_gate]
    overall["failure_reasons"] = [";".join(value[1]) for value in overall_gate]
    channel_numeric = [column for column in channels.columns if column not in {
        "subject_id", "fold_id", "set", "inner_model", "model_role", "channel_index", "channel",
    }]
    subject_channel = channels.groupby(
        ["subject_id", "set", "channel_index", "channel"], as_index=False
    )[channel_numeric].median(numeric_only=True)
    return fold, subject, overall, subject_channel


def wide_subject(subject: pd.DataFrame) -> pd.DataFrame:
    keep = list(QUALITY_METRICS) + ["nrmse_p90", "residual_rms_p90", "safety_pass"]
    wide = subject.pivot(index="subject_id", columns="set", values=keep)
    wide.columns = [f"{metric}_{set_name}" for metric, set_name in wide.columns]
    wide = wide.reset_index()
    for metric in ("median_nrmse", "spectral_cosine_distance", "residual_rms"):
        wide[f"{metric}_test_minus_train"] = (
            wide[f"{metric}_outer_test"] - wide[f"{metric}_nbm_train"]
        )
    for metric in ("improvement_pct", "median_corr", "median_amplitude_ratio",
                   "raw_latent_distance_corr"):
        wide[f"{metric}_train_minus_test"] = (
            wide[f"{metric}_nbm_train"] - wide[f"{metric}_outer_test"]
        )
    return wide


def link_classifier_diagnostics(root: Path, wide: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    classifier_path = root / "analysis_train_validation_test" / "subject_train_validation_test_wide.csv"
    if not classifier_path.exists():
        return pd.DataFrame(), []
    classifier = pd.read_csv(classifier_path)
    rows: list[dict[str, Any]] = []
    for method in ("B1", "B2", "B3"):
        selected = classifier[classifier["method"] == method][
            ["subject_id", "pr_auc_train_minus_test", "pr_auc_validation_minus_test"]
        ]
        joined = wide.merge(selected, on="subject_id", how="inner")
        for metric in ("median_nrmse_test_minus_train", "median_corr_train_minus_test",
                       "residual_rms_test_minus_train"):
            rho, p_value = spearmanr(joined[metric], joined["pr_auc_train_minus_test"])
            rows.append({"classifier_method": method, "nbm_gap_metric": metric,
                         "spearman_rho": float(rho), "p_value_descriptive": float(p_value),
                         "n_subjects": len(joined)})
    return classifier, rows


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def quality_heatmap(subject: pd.DataFrame, path: Path) -> None:
    configure_plotting()
    panels = (
        ("median_corr", "Pearson", 0.0, 1.0, "Blues"),
        ("median_nrmse", "NRMSE", 0.0, 1.2, "Reds"),
        ("improvement_pct", "Improvement (%)", 0.0, 100.0, "Blues"),
        ("median_amplitude_ratio", "Amplitude ratio", 0.0, 1.5, "Blues"),
        ("spectral_cosine_distance", "Spectral distance", 0.0, 1.0, "Reds"),
        ("residual_rms", "Residual RMS", 0.0, max(0.5, float(subject["residual_rms"].quantile(.95))), "Reds"),
        ("raw_latent_distance_corr", "Latent distance corr.", -0.2, 1.0, "Blues"),
        ("latent_between_window_variance", "Latent variance", 0.0,
         max(0.1, float(subject["latent_between_window_variance"].quantile(.95))), "Blues"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 5.3), constrained_layout=True)
    for axis, (metric, title, minimum, maximum, cmap) in zip(axes.flat, panels):
        matrix = (subject.pivot(index="subject_id", columns="set", values=metric)
                  .reindex(index=exp.SUBJECTS, columns=SETS).to_numpy(dtype=float))
        image = axis.imshow(matrix, vmin=minimum, vmax=maximum, cmap=cmap, aspect="auto")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                label = f"{value:.2f}" if metric != "improvement_pct" else f"{value:.0f}"
                relative = (value - minimum) / max(maximum - minimum, 1e-12)
                axis.text(x, y, label, ha="center", va="center", fontsize=5.7,
                          color="white" if relative > .62 else "#222222")
        axis.set_title(title, fontsize=8, fontweight="bold")
        axis.set_xticks(range(3), [SET_LABELS[value] for value in SETS], rotation=28, ha="right")
        axis.set_yticks(range(len(exp.SUBJECTS)), exp.SUBJECTS)
        axis.tick_params(length=0)
        fig.colorbar(image, ax=axis, fraction=.045, pad=.025)
    fig.suptitle("TCN-DAE reconstruction audit on strict Non-FoG windows",
                 fontsize=10, fontweight="bold")
    save_figure(fig, path)


def channel_heatmap(subject_channel: pd.DataFrame, path: Path) -> None:
    configure_plotting()
    test = subject_channel[subject_channel["set"] == "outer_test"]
    matrix = (test.pivot(index="subject_id", columns="channel", values="residual_rms_q90")
              .reindex(index=exp.SUBJECTS, columns=CHANNEL_NAMES).to_numpy(dtype=float))
    fig, axis = plt.subplots(figsize=(7.25, 2.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Reds", vmin=0,
                        vmax=max(.5, float(np.nanpercentile(matrix, 95))), aspect="auto")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            axis.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=5.5)
    axis.set_xticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES, rotation=40, ha="right")
    axis.set_yticks(range(len(exp.SUBJECTS)), exp.SUBJECTS)
    axis.set_title("Outer-test channel residual RMS (90th percentile)", fontsize=9, fontweight="bold")
    colorbar = fig.colorbar(image, ax=axis, fraction=.025, pad=.02)
    colorbar.set_label("Scaled residual RMS")
    save_figure(fig, path)


def association_figure(wide: pd.DataFrame, classifier: pd.DataFrame, path: Path) -> None:
    if classifier.empty:
        return
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)
    for axis, method in zip(axes, ("B1", "B2", "B3")):
        values = classifier[classifier["method"] == method][
            ["subject_id", "pr_auc_train_minus_test"]
        ].merge(wide[["subject_id", "median_nrmse_test_minus_train"]], on="subject_id")
        axis.scatter(values["median_nrmse_test_minus_train"], values["pr_auc_train_minus_test"],
                     color="#4477AA", s=24)
        for row in values.itertuples(index=False):
            axis.annotate(row.subject_id,
                          (row.median_nrmse_test_minus_train, row.pr_auc_train_minus_test),
                          fontsize=5.8, xytext=(2, 2), textcoords="offset points")
        rho = spearmanr(values["median_nrmse_test_minus_train"],
                        values["pr_auc_train_minus_test"]).statistic
        axis.set_title(f"{method}  rho={rho:.2f}", fontsize=8, fontweight="bold")
        axis.axvline(0, color="#888888", linestyle="--", linewidth=.7)
        axis.set_xlabel("NBM test - train NRMSE")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Classifier train - test PR-AUC")
    fig.suptitle("Does NBM reconstruction degradation explain classifier overfitting?",
                 fontsize=9.5, fontweight="bold")
    save_figure(fig, path)


def diagnosis_table(wide: pd.DataFrame, classifier: pd.DataFrame,
                    b0_wide: pd.DataFrame | None = None) -> pd.DataFrame:
    output = wide.copy()
    if not classifier.empty:
        gap = classifier.groupby("subject_id", as_index=False)["pr_auc_train_minus_test"].mean()
        output = output.merge(gap.rename(columns={"pr_auc_train_minus_test": "classifier_mean_pr_gap"}),
                              on="subject_id", how="left")
    if b0_wide is not None and not b0_wide.empty:
        output = output.merge(
            b0_wide[["subject_id", "pr_auc_train_minus_test"]].rename(
                columns={"pr_auc_train_minus_test": "b0_raw_pr_gap"}
            ), on="subject_id", how="left"
        )
    diagnoses: list[str] = []
    for row in output.itertuples(index=False):
        train_pass = bool(getattr(row, "safety_pass_nbm_train"))
        test_pass = bool(getattr(row, "safety_pass_outer_test"))
        large_nbm_gap = (
            float(getattr(row, "median_nrmse_test_minus_train")) > .15
            or float(getattr(row, "median_corr_train_minus_test")) > .10
        )
        classifier_gap = float(getattr(row, "classifier_mean_pr_gap", 0.0)) > .25
        raw_gap = float(getattr(row, "b0_raw_pr_gap", 0.0)) > .25
        if not train_pass and raw_gap:
            diagnosis = ("NBM poor fit + classifier/domain shift" if not large_nbm_gap
                         else "NBM poor fit/shift + classifier/domain shift")
        elif not train_pass and classifier_gap:
            diagnosis = ("NBM poor fit + residual-pipeline overfit" if not large_nbm_gap
                         else "NBM poor fit/shift + residual-pipeline overfit")
        elif not train_pass:
            diagnosis = "NBM underfit/poor fit" if not large_nbm_gap else "NBM poor fit + shift"
        elif (not test_pass or large_nbm_gap) and raw_gap:
            diagnosis = "NBM shift + classifier/domain shift"
        elif not test_pass or large_nbm_gap:
            diagnosis = "NBM overfit/record shift"
        elif classifier_gap:
            diagnosis = "classifier-dominant overfit"
        else:
            diagnosis = "no major overfit signal"
        diagnoses.append(diagnosis)
    output["diagnosis"] = diagnoses
    return output


def write_report(overall: pd.DataFrame, diagnosis: pd.DataFrame,
                 associations: list[dict[str, Any]], path: Path,
                 b0_macro: pd.DataFrame | None = None) -> None:
    lines = [
        "# TCN-DAE NBM三集合重建审计",
        "",
        "## 审计定义",
        "",
        "- NBM训练集：每个OOF TCN-DAE实际参与优化的严格Non-FoG窗口。",
        "- Record-separated验证集：对应OOF模型完全未见的inner-held严格Non-FoG窗口。",
        "- 外层测试集：最终TCN-DAE完全未见的外层测试记录/时间块中的严格Non-FoG窗口。",
        "- 训练/验证指标先在3个inner模型间取中位数，再在被试的外层折间取中位数。",
        "- 潜变量距离相关最多使用均匀抽取的256个窗口，避免二次方距离矩阵支配计算。",
        "- 频谱距离为0.5–10 Hz功率谱余弦距离，越小越好。",
        "",
        "## 总体中位数（8被试）",
        "",
        "| 集合 | 改善率% | Pearson | NRMSE | 幅值比 | 频谱距离 | 残差RMS | 潜变量方差 | 距离相关 | 门控 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for set_name in SETS:
        row = overall[overall["set"] == set_name].iloc[0]
        lines.append(
            f"| {SET_LABELS[set_name]} | {row.improvement_pct:.2f} | {row.median_corr:.3f} | "
            f"{row.median_nrmse:.3f} | {row.median_amplitude_ratio:.3f} | "
            f"{row.spectral_cosine_distance:.3f} | {row.residual_rms:.3f} | "
            f"{row.latent_between_window_variance:.4g} | {row.raw_latent_distance_corr:.3f} | "
            f"{'PASS' if row.safety_pass else 'FAIL'} |"
        )
    lines += ["", "## 被试级诊断", "",
              "| 被试 | Train NRMSE | Validation NRMSE | Test NRMSE | Test-Train | "
              "Train Pearson | Test Pearson | 残差方法PR差距 | B0 PR差距 | 诊断 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in diagnosis.itertuples(index=False):
        lines.append(
            f"| {row.subject_id} | {row.median_nrmse_nbm_train:.3f} | "
            f"{row.median_nrmse_record_validation:.3f} | {row.median_nrmse_outer_test:.3f} | "
            f"{row.median_nrmse_test_minus_train:+.3f} | {row.median_corr_nbm_train:.3f} | "
            f"{row.median_corr_outer_test:.3f} | {getattr(row, 'classifier_mean_pr_gap', math.nan):.3f} | "
            f"{getattr(row, 'b0_raw_pr_gap', math.nan):.3f} | "
            f"{row.diagnosis} |"
        )
    if b0_macro is not None and not b0_macro.empty:
        lines += ["", "## 不经过NBM的B0分类器对照", "",
                  "| 集合 | PR-AUC | ROC-AUC | FoG F1 | BAcc | MCC |",
                  "|---|---:|---:|---:|---:|---:|"]
        for set_name in classifier_analysis.SPLITS:
            row = b0_macro[b0_macro["split"] == set_name].iloc[0]
            lines.append(f"| {set_name} | {row.pr_auc:.3f} | {row.roc_auc:.3f} | "
                         f"{row.fog_f1:.3f} | {row.balanced_accuracy:.3f} | {row.mcc:.3f} |")
    if associations:
        lines += ["", "## NBM差距与分类器过拟合的描述性相关", "",
                  "| 分类方法 | NBM差距 | Spearman rho | p（描述性） | n |",
                  "|---|---|---:|---:|---:|"]
        for row in associations:
            lines.append(f"| {row['classifier_method']} | {row['nbm_gap_metric']} | "
                         f"{row['spearman_rho']:.3f} | {row['p_value_descriptive']:.3f} | "
                         f"{row['n_subjects']} |")
    lines += ["", "## 解释边界", "",
              "- 8名被试的相关分析仅作机制定位，不作为显著性证据。",
              "- 如果NBM训练集已经不通过安全线，应优先解释为欠拟合/重建能力不足，而非过拟合。",
              "- 只有训练质量良好而record-validation或outer-test显著下降，才支持NBM过拟合或记录域偏移。",
              "- 分类器验证集参与early stopping与阈值选择；最终泛化判断以外层测试为准。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--b0-control-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "analysis_nbm_three_set_audit"
    output.mkdir(parents=True, exist_ok=True)
    if args.b0_control_only:
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        run_frame = classifier_analysis.collect_run_metrics(root, ("B0",), device)
        seed_level, subject_level, b0_macro = classifier_analysis.aggregate_metrics(run_frame)
        b0_wide = classifier_analysis.wide_subject_table(subject_level)
        run_frame.to_csv(output / "b0_control_run_fold_split_metrics.csv", index=False,
                         encoding="utf-8-sig")
        seed_level.to_csv(output / "b0_control_subject_seed_metrics.csv", index=False,
                          encoding="utf-8-sig")
        subject_level.to_csv(output / "b0_control_subject_split_metrics.csv", index=False,
                             encoding="utf-8-sig")
        b0_wide.to_csv(output / "b0_control_subject_wide.csv", index=False, encoding="utf-8-sig")
        b0_macro.to_csv(output / "b0_control_macro.csv", index=False, encoding="utf-8-sig")
        wide = pd.read_csv(output / "subject_nbm_gaps.csv")
        overall = pd.read_csv(output / "overall_set_metrics.csv")
        classifier_path = root / "analysis_train_validation_test" / "subject_train_validation_test_wide.csv"
        classifier = pd.read_csv(classifier_path)
        _, associations = link_classifier_diagnostics(root, wide)
        diagnosis = diagnosis_table(wide, classifier, b0_wide)
        diagnosis.to_csv(output / "nbm_vs_classifier_diagnosis.csv", index=False,
                         encoding="utf-8-sig")
        write_report(overall, diagnosis, associations,
                     output / "nbm_three_set_audit_report.md", b0_macro=b0_macro)
        print(f"B0 CONTROL COMPLETE {output}", flush=True)
        return
    if args.render_only:
        subject = pd.read_csv(output / "subject_set_metrics.csv")
        subject_channel = pd.read_csv(output / "subject_set_channel_metrics.csv")
        wide = pd.read_csv(output / "subject_nbm_gaps.csv")
        overall = pd.read_csv(output / "overall_set_metrics.csv")
        association_path = output / "nbm_classifier_gap_correlations.csv"
        associations = (pd.read_csv(association_path).to_dict("records")
                        if association_path.exists() else [])
        classifier_path = root / "analysis_train_validation_test" / "subject_train_validation_test_wide.csv"
        classifier = pd.read_csv(classifier_path) if classifier_path.exists() else pd.DataFrame()
        b0_wide_path = output / "b0_control_subject_wide.csv"
        b0_wide = pd.read_csv(b0_wide_path) if b0_wide_path.exists() else None
        diagnosis = diagnosis_table(wide, classifier, b0_wide)
        diagnosis.to_csv(output / "nbm_vs_classifier_diagnosis.csv", index=False,
                         encoding="utf-8-sig")
        quality_heatmap(subject, output / "nbm_three_set_quality_heatmaps")
        channel_heatmap(subject_channel, output / "outer_test_channel_residual_q90")
        association_figure(wide, classifier, output / "nbm_classifier_overfit_association")
        b0_macro_path = output / "b0_control_macro.csv"
        b0_macro = pd.read_csv(b0_macro_path) if b0_macro_path.exists() else None
        write_report(overall, diagnosis, associations, output / "nbm_three_set_audit_report.md",
                     b0_macro=b0_macro)
        print(f"RENDER COMPLETE {output}", flush=True)
        return
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    fold_summary = pd.read_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv")
    raw_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for position, row in enumerate(fold_summary.itertuples(index=False), 1):
        local, local_channels = audit_outer_fold(root, str(row.subject_id), str(row.fold_id), device)
        raw_rows.extend(local)
        channel_rows.extend(local_channels)
        print(f"AUDIT {position}/{len(fold_summary)} {row.subject_id}/{row.fold_id}", flush=True)
    raw = pd.DataFrame(raw_rows)
    channels = pd.DataFrame(channel_rows)
    fold, subject, overall, subject_channel = aggregate(raw, channels)
    wide = wide_subject(subject)
    classifier, associations = link_classifier_diagnostics(root, wide)
    diagnosis = diagnosis_table(wide, classifier)
    raw.to_csv(output / "inner_model_set_metrics.csv", index=False, encoding="utf-8-sig")
    channels.to_csv(output / "inner_model_channel_residual_quantiles.csv", index=False,
                    encoding="utf-8-sig")
    fold.to_csv(output / "fold_set_metrics.csv", index=False, encoding="utf-8-sig")
    subject.to_csv(output / "subject_set_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "overall_set_metrics.csv", index=False, encoding="utf-8-sig")
    subject_channel.to_csv(output / "subject_set_channel_metrics.csv", index=False,
                           encoding="utf-8-sig")
    wide.to_csv(output / "subject_nbm_gaps.csv", index=False, encoding="utf-8-sig")
    diagnosis.to_csv(output / "nbm_vs_classifier_diagnosis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(associations).to_csv(output / "nbm_classifier_gap_correlations.csv", index=False,
                                     encoding="utf-8-sig")
    quality_heatmap(subject, output / "nbm_three_set_quality_heatmaps")
    channel_heatmap(subject_channel, output / "outer_test_channel_residual_q90")
    association_figure(wide, classifier, output / "nbm_classifier_overfit_association")
    write_report(overall, diagnosis, associations, output / "nbm_three_set_audit_report.md")
    manifest = {
        "subjects": list(exp.SUBJECTS), "outer_folds": len(fold_summary), "sets": list(SETS),
        "strict_nonfog_only": True,
        "record_validation": "strict clean held inner fold evaluated by its OOF TCN-DAE",
        "outer_test": "strict clean outer test windows evaluated by final TCN-DAE",
        "spectral_distance": "median cosine distance of 0.5-10 Hz power spectrum",
        "distance_correlation_sample_cap": 256,
        "gate": {"improvement_min": 95, "pearson_min": .80, "nrmse_max": .60,
                 "amplitude_range": [.75, 1.25], "latent_distance_min": .50},
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
