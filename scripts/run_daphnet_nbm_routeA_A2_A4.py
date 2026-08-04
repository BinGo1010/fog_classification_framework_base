from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_nbm_routeA_A1b_generalization_repair as a1b
import run_daphnet_nbm_routeA_final_residual_validation as a1
import run_daphnet_nbm_tcdae_three_rounds as base
import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as legacy


EXPERIMENT = "daphnet_nbm_routeA_A2_A4_v1"
SUBJECTS = ("S01", "S05", "S07", "S08", "S09")
SELECTION_SUBJECTS = ("S01", "S05", "S08", "S09")
SPARSE_DIAGNOSTIC_SUBJECTS = ("S07",)
SEEDS = (20260802, 20260803, 20260804)
DENOISING = ("D0", "D1", "D2", "D3")
CONDITIONS = ("clean", "gaussian_0p01", "gaussian_0p03", "mask_8", "mask_16", "single_axis_mask")
CALIBRATIONS = ("C0", "C1", "C2")
CLIPS: tuple[float | None, ...] = (None, 6.0, 12.0)
REPRESENTATIONS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
WINDOW = 128
CHANNELS = 9
FS = 64
STRIDE_SECONDS = 32 / FS


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.median(values)) if values else math.nan


def stable_seed(subject: str, seed: int, offset: int) -> int:
    return int(seed + 100_000 * int(subject[1:]) + offset)


def corrupt_mixture(clean: np.ndarray, scheme: str, seed: int) -> np.ndarray:
    """Frozen training/validation mixture; every corrupted input targets clean."""
    output = np.asarray(clean, dtype=np.float32).copy()
    if scheme == "D0":
        return output
    rng = np.random.default_rng(seed)
    modes = rng.random(len(output))
    if scheme == "D1":
        gaussian = modes >= 0.70
        masking = np.zeros(len(output), dtype=bool)
        axis_masking = np.zeros(len(output), dtype=bool)
    elif scheme == "D2":
        gaussian = np.zeros(len(output), dtype=bool)
        masking = modes >= 0.60
        axis_masking = np.zeros(len(output), dtype=bool)
    elif scheme == "D3":
        gaussian = (modes >= 0.20) & (modes < 0.60)
        masking = (modes >= 0.60) & (modes < 0.90)
        axis_masking = modes >= 0.90
    else:
        raise ValueError(f"unknown scheme {scheme}")
    indices = np.flatnonzero(gaussian)
    if len(indices):
        std = rng.uniform(0.01, 0.03, size=(len(indices), 1, 1)).astype(np.float32)
        output[indices] += rng.normal(size=output[indices].shape).astype(np.float32) * std
    for index in np.flatnonzero(masking):
        length = int(rng.integers(8, 17))
        start = int(rng.integers(0, WINDOW - length + 1))
        output[index, start:start + length, :] = 0.0
    for index in np.flatnonzero(axis_masking):
        length = int(rng.integers(4, 9))
        start = int(rng.integers(0, WINDOW - length + 1))
        channel = int(rng.integers(0, CHANNELS))
        output[index, start:start + length, channel] = 0.0
    return np.ascontiguousarray(output)


def corrupt_condition(clean: np.ndarray, condition: str, seed: int) -> np.ndarray:
    output = np.asarray(clean, dtype=np.float32).copy()
    if condition == "clean":
        return output
    rng = np.random.default_rng(seed)
    if condition.startswith("gaussian"):
        std = 0.01 if condition.endswith("0p01") else 0.03
        output += rng.normal(0.0, std, size=output.shape).astype(np.float32)
    elif condition in ("mask_8", "mask_16"):
        length = 8 if condition == "mask_8" else 16
        for index in range(len(output)):
            start = int(rng.integers(0, WINDOW - length + 1))
            output[index, start:start + length, :] = 0.0
    elif condition == "single_axis_mask":
        for index in range(len(output)):
            length = int(rng.integers(4, 9))
            start = int(rng.integers(0, WINDOW - length + 1))
            channel = int(rng.integers(0, CHANNELS))
            output[index, start:start + length, channel] = 0.0
    else:
        raise ValueError(condition)
    return np.ascontiguousarray(output)


def load_model(checkpoint: Path, device: torch.device) -> a1b.ContextM3:
    model = a1b.ContextM3(WINDOW).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def train_denoiser(
    item: a1.PreparedSubject, scheme: str, seed: int, run_dir: Path,
    device: torch.device, max_epochs: int, patience: int, workers: int,
) -> tuple[a1b.ContextM3, dict[str, Any]]:
    required = (run_dir / "best_model.pt", run_dir / "last_model.pt", run_dir / "training_log.csv")
    if all(path.exists() for path in required):
        model = load_model(run_dir / "best_model.pt", device)
        payload = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        return model, dict(payload["training"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_input = corrupt_mixture(item.train_x, scheme, stable_seed(item.subject, seed, 11))
    val_input = corrupt_mixture(item.calibration_x, scheme, stable_seed(item.subject, seed, 29))
    base.set_seed(seed)
    model = a1b.ContextM3(WINDOW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batches = a1b.pair_loader(train_input, item.train_x, shuffle=True, seed=seed, workers=workers)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0
    last_train = math.inf
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        max_gradient = 0.0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(batch_x)
            loss = a1b.structural_loss("L4", predicted, batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite A2 gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
            max_gradient = max(max_gradient, float(gradient))
        last_train = total / count
        validation = a1b.evaluate_loss(model, val_input, item.calibration_x, "L4", device)
        improved = validation < best_loss - 1e-8
        if improved:
            best_loss = validation
            best_epoch = epoch
            best_state = base.clone_state(model)
            bad = 0
        else:
            bad += 1
        last_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or improved or epoch == max_epochs:
            history.append({"epoch": epoch, "train_loss": last_train, "validation_loss": validation,
                            "max_gradient_norm_before_clip": max_gradient, "improved": improved,
                            "bad_epochs": bad})
        if epoch == 1 or epoch % 100 == 0:
            print(f"A2 {scheme} {item.subject} seed={seed} epoch={epoch}/{max_epochs} "
                  f"train={last_train:.6g} val={validation:.6g} best={best_loss:.6g}@{best_epoch}", flush=True)
        if bad >= patience:
            break
    if best_state is None:
        raise AssertionError("A2 produced no checkpoint")
    training = {"stage": "A2_denoising", "scheme": scheme, "subject_id": item.subject,
                "seed": seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
                "best_validation_loss": best_loss, "last_train_loss": last_train,
                "elapsed_seconds": time.perf_counter() - started,
                "train_windows": len(item.train_x), "validation_windows": len(item.calibration_x),
                "loss": "L4", "input_context": "W0"}
    base.torch_save(run_dir / "last_model.pt", {"model_state": base.clone_state(model), "training": training})
    base.torch_save(run_dir / "best_model.pt", {"model_state": best_state, "training": training})
    write_csv(run_dir / "training_log.csv", history)
    model.load_state_dict(best_state)
    return model, training


def prediction_metrics(clean: np.ndarray, corrupted: np.ndarray, predicted: np.ndarray,
                       template: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics, arrays = a1.reconstruction_metrics(clean, predicted, template)
    corruption_window_mse = np.mean(np.square(corrupted - clean), axis=(1, 2))
    prediction_window_mse = np.mean(np.square(predicted - clean), axis=(1, 2))
    corruption_mse = float(np.mean(corruption_window_mse))
    metrics["corruption_mse"] = corruption_mse
    metrics["recovery_improvement_pct"] = (
        100.0 * (corruption_mse - metrics["mse"]) / max(corruption_mse, 1e-12)
        if corruption_mse > 1e-12 else None
    )
    metrics["negative_recovery_window_fraction"] = (
        float(np.mean(prediction_window_mse > corruption_window_mse))
        if corruption_mse > 1e-12 else None
    )
    arrays["corruption_window_mse"] = corruption_window_mse
    arrays["prediction_window_mse"] = prediction_window_mse
    return metrics, arrays


def a2_checkpoint(root: Path, parent: Path, scheme: str, subject: str, seed: int) -> Path:
    if scheme == "D0":
        return parent / "A1_retest" / "L4" / "W0" / subject / f"seed{seed}" / "best_model.pt"
    return root / "A2_denoising" / scheme / subject / f"seed{seed}" / "best_model.pt"


def plot_worst_waveform(path: Path, clean: np.ndarray, corrupted: np.ndarray,
                        predicted: np.ndarray, window_nrmse: np.ndarray, title: str) -> None:
    index = int(np.argmax(window_nrmse))
    channel = int(np.argmax(np.sqrt(np.mean(np.square(clean[index] - predicted[index]), axis=0))))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.plot(clean[index, :, channel], label="clean target", lw=1.5)
    plt.plot(corrupted[index, :, channel], label="corrupted input", lw=1.0, alpha=.7)
    plt.plot(predicted[index, :, channel], label="reconstruction", lw=1.2)
    plt.title(f"{title}; worst window={index}, channel={channel}")
    plt.xlabel("sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def evaluate_a2_run(model: nn.Module, item: a1.PreparedSubject, scheme: str, seed: int,
                    run_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    marker = run_dir / "metrics_all_conditions.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    template, _ = a1.nearest_training_windows(item.train_x, item.test_x)
    rows: list[dict[str, Any]] = []
    saved_clean: dict[str, np.ndarray] = {}
    for offset, condition in enumerate(CONDITIONS, 1):
        corrupted = corrupt_condition(item.test_x, condition, stable_seed(item.subject, 20260850, offset))
        predicted, latent = a1b.predict_pairs(model, corrupted, item.test_x, device)
        metrics, arrays = prediction_metrics(item.test_x, corrupted, predicted, template)
        row = {"stage": "A2_denoising", "scheme": scheme, "subject_id": item.subject,
               "seed": seed, "condition": condition, **metrics}
        rows.append(row)
        if condition == "clean":
            saved_clean = {"actual": item.test_x, "prediction": predicted, "latent": latent,
                           "window_nrmse": arrays["window_nrmse"]}
        if condition == "mask_16":
            plot_worst_waveform(run_dir / "worst_waveform_mask16.png", item.test_x, corrupted,
                                predicted, arrays["window_nrmse"], f"{scheme} {item.subject} seed={seed}")
    np.savez_compressed(run_dir / "clean_predictions.npz", **saved_clean)
    write_json(marker, rows)
    write_csv(run_dir / "metrics_all_conditions.csv", rows)
    return rows


def make_a2_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schemes: list[dict[str, Any]] = []
    baseline = [row for row in rows if row["scheme"] == "D0"]
    for scheme in DENOISING:
        current = [row for row in rows if row["scheme"] == scheme]
        subject_rows: list[dict[str, Any]] = []
        for subject in SUBJECTS:
            clean = [row for row in current if row["subject_id"] == subject and row["condition"] == "clean"]
            clean0 = [row for row in baseline if row["subject_id"] == subject and row["condition"] == "clean"]
            nrmse = median(clean, "median_nrmse")
            nrmse0 = median(clean0, "median_nrmse")
            corr = median(clean, "median_corr")
            corr0 = median(clean0, "median_corr")
            p90 = median(clean, "nrmse_p90")
            p900 = median(clean0, "nrmse_p90")
            amp = median(clean, "median_amplitude_ratio")
            preserve = bool(nrmse <= nrmse0 * 1.10 + 1e-12 and corr >= corr0 - 0.05 - 1e-12
                            and p90 <= p900 + 1e-12 and 0.5 <= amp <= 1.5)
            subject_rows.append({"subject_id": subject, "clean_nrmse": nrmse, "D0_clean_nrmse": nrmse0,
                                 "clean_pearson": corr, "D0_clean_pearson": corr0,
                                 "clean_nrmse_p90": p90, "D0_clean_nrmse_p90": p900,
                                 "clean_amplitude_ratio": amp, "clean_preserved": preserve})
        wins: list[dict[str, Any]] = []
        for condition in CONDITIONS[1:]:
            value = median((row for row in current if row["condition"] == condition), "recovery_improvement_pct")
            value0 = median((row for row in baseline if row["condition"] == condition), "recovery_improvement_pct")
            wins.append({"condition": condition, "median_recovery_improvement_pct": value,
                         "D0_median_recovery_improvement_pct": value0, "better_than_D0": value > value0 + 1e-12})
        clean_all = [row for row in current if row["condition"] == "clean"]
        clean0_all = [row for row in baseline if row["condition"] == "clean"]
        preserve_count = sum(row["clean_preserved"] for row in subject_rows)
        win_count = sum(row["better_than_D0"] for row in wins)
        aggregate = {
            "median_clean_nrmse": median(clean_all, "median_nrmse"),
            "D0_median_clean_nrmse": median(clean0_all, "median_nrmse"),
            "median_clean_pearson": median(clean_all, "median_corr"),
            "D0_median_clean_pearson": median(clean0_all, "median_corr"),
            "median_clean_nrmse_p90": median(clean_all, "nrmse_p90"),
            "D0_median_clean_nrmse_p90": median(clean0_all, "nrmse_p90"),
            "median_clean_amplitude_ratio": median(clean_all, "median_amplitude_ratio"),
        }
        aggregate_preserve = bool(
            aggregate["median_clean_nrmse"] <= aggregate["D0_median_clean_nrmse"] * 1.10 + 1e-12
            and aggregate["median_clean_pearson"] >= aggregate["D0_median_clean_pearson"] - 0.05 - 1e-12
            and aggregate["median_clean_nrmse_p90"] <= aggregate["D0_median_clean_nrmse_p90"] + 1e-12
            and 0.5 <= aggregate["median_clean_amplitude_ratio"] <= 1.5
        )
        collapse = any(not 0.5 <= row["clean_amplitude_ratio"] <= 1.5 for row in subject_rows)
        passed = scheme == "D0" or bool(preserve_count >= 4 and win_count >= 3 and aggregate_preserve and not collapse)
        schemes.append({"scheme": scheme, "subject_clean_preservation_count": preserve_count,
                        "subject_total": len(SUBJECTS), "corruption_win_count": win_count,
                        "aggregate_clean_preserved": aggregate_preserve, "amplitude_collapse": collapse,
                        "gate_pass": passed, "subject_details": subject_rows,
                        "corruption_details": wins, **aggregate})
    candidates = [row for row in schemes if row["scheme"] != "D0" and row["gate_pass"]]
    if candidates:
        selected = max(candidates, key=lambda row: (row["corruption_win_count"],
                       row["subject_clean_preservation_count"], -row["median_clean_nrmse_p90"],
                       -DENOISING.index(row["scheme"])))
        status = "PASS"
        reason = "at least one denoising candidate satisfied every frozen A2 gate"
    else:
        selected = schemes[0]
        status = "D0 FALLBACK PASS"
        reason = "D1-D3 did not satisfy every gate; template-mandated clean-autoencoder fallback retained"
    return {"stage": "A2_denoising_ablation", "status": status, "advance_to_A3": True,
            "selected_scheme": selected["scheme"], "selection_reason": reason,
            "test_fog_used": False, "schemes": schemes}


@dataclass
class ResidualBundle:
    x: np.ndarray
    xhat: np.ndarray
    residual: np.ndarray


def indices(item: a1.PreparedSubject, split: str, label: int) -> np.ndarray:
    mask = (item.windows.split == split) & (item.windows.label == label)
    if label == 0:
        mask &= item.windows.clean_normal
    return np.flatnonzero(mask).astype(np.int64)


def prepared_windows(item: a1.PreparedSubject, raw_indices: np.ndarray) -> np.ndarray:
    raw = legacy.raw_windows(item.records, item.windows, raw_indices)
    return legacy.prepare_nbm_windows(item.scaler, raw, center=True)


def bundle(model: nn.Module, values: np.ndarray, device: torch.device) -> ResidualBundle:
    predicted, _ = a1.predict_model(model, values, device)
    return ResidualBundle(values, predicted, values - predicted)


def fit_residual_calibration(train_residual: np.ndarray, sigma_min: float = 0.05) -> dict[str, np.ndarray]:
    center = np.median(train_residual, axis=(0, 1), keepdims=True).astype(np.float32)
    mad = np.median(np.abs(train_residual - center), axis=(0, 1), keepdims=True).astype(np.float32)
    scale = np.maximum(1.4826 * mad, sigma_min).astype(np.float32)
    return {"center": center, "scale": scale}


def apply_residual_calibration(values: np.ndarray, stats: dict[str, np.ndarray], method: str,
                               clip: float | None) -> tuple[np.ndarray, float]:
    if method == "C0":
        output = np.asarray(values, dtype=np.float32).copy()
    elif method in ("C1", "C2"):
        output = ((values - stats["center"]) / stats["scale"]).astype(np.float32)
        if method == "C2":
            output -= output.mean(axis=1, keepdims=True)
    else:
        raise ValueError(method)
    saturation = 0.0
    if clip is not None:
        saturation = float(np.mean(np.abs(output) > clip))
        output = np.clip(output, -clip, clip)
    return np.ascontiguousarray(output), saturation


def residual_score(values: np.ndarray) -> np.ndarray:
    return np.median(np.abs(values), axis=(1, 2)).astype(np.float64)


def cliffs_delta(normal: np.ndarray, fog: np.ndarray) -> float:
    combined = np.concatenate((normal, fog))
    ranks = rankdata(combined, method="average")
    rank_fog = float(np.sum(ranks[len(normal):]))
    u = rank_fog - len(fog) * (len(fog) + 1) / 2.0
    return float(2.0 * u / (len(normal) * len(fog)) - 1.0)


def separation_metrics(normal: np.ndarray, fog: np.ndarray, train_normal: np.ndarray) -> dict[str, float]:
    normal = np.asarray(normal, dtype=np.float64)
    fog = np.asarray(fog, dtype=np.float64)
    train_normal = np.asarray(train_normal, dtype=np.float64)
    y = np.concatenate((np.zeros(len(normal), dtype=int), np.ones(len(fog), dtype=int)))
    score = np.concatenate((normal, fog))
    threshold = float(np.percentile(train_normal, 95))
    if len(normal) >= 2 and len(fog) >= 2:
        pooled = math.sqrt(max(((len(normal) - 1) * np.var(normal, ddof=1) +
                                (len(fog) - 1) * np.var(fog, ddof=1)) /
                               max(len(normal) + len(fog) - 2, 1), 1e-12))
        d = float((np.mean(fog) - np.mean(normal)) / pooled)
        correction = 1.0 - 3.0 / max(4.0 * (len(normal) + len(fog)) - 9.0, 1.0)
        hedges = float(d * correction)
    else:
        # S07 has one validation FoG window. Rank metrics remain descriptive,
        # but a variance-based standardized effect is not estimable.
        hedges = math.nan
    return {
        "nonfog_p50": float(np.percentile(normal, 50)), "nonfog_p90": float(np.percentile(normal, 90)),
        "nonfog_p95": float(np.percentile(normal, 95)), "fog_p50": float(np.percentile(fog, 50)),
        "fog_p90": float(np.percentile(fog, 90)),
        "fog_to_nonfog_median_ratio": float(np.median(fog) / max(np.median(normal), 1e-12)),
        "auroc": float(roc_auc_score(y, score)), "average_precision": float(average_precision_score(y, score)),
        "recall_at_train_nonfog_p95": float(np.mean(fog > threshold)),
        "nonfog_false_alarm_fraction": float(np.mean(normal > threshold)),
        "false_alarm_windows_per_minute_proxy": float(np.mean(normal > threshold) * 60.0 / STRIDE_SECONDS),
        "cliffs_delta": cliffs_delta(normal, fog), "hedges_g": hedges,
        "train_nonfog_p50": float(np.median(train_normal)),
        "nonfog_median_drift_ratio": float(np.median(normal) / max(np.median(train_normal), 1e-12)),
    }


def prepare_residual_sets(model: nn.Module, item: a1.PreparedSubject, device: torch.device) -> dict[str, ResidualBundle]:
    val_nf = prepared_windows(item, indices(item, "validation", 0))
    val_fog = prepared_windows(item, indices(item, "validation", 1))
    test_nf = item.test_x
    test_fog = prepared_windows(item, indices(item, "test", 1))
    return {"train_nonfog": bundle(model, item.train_x, device), "validation_nonfog": bundle(model, val_nf, device),
            "validation_fog": bundle(model, val_fog, device), "test_nonfog": bundle(model, test_nf, device),
            "test_fog": bundle(model, test_fog, device)}


def run_a3(root: Path, parent: Path, prepared: dict[str, a1.PreparedSubject], scheme: str,
           device: torch.device, sigma_min: float) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, ResidualBundle]]]:
    stage = root / "A3_residual_calibration"
    validation_rows: list[dict[str, Any]] = []
    residual_sets: dict[tuple[str, int], dict[str, ResidualBundle]] = {}
    for subject in SUBJECTS:
        for seed in SEEDS:
            model = load_model(a2_checkpoint(root, parent, scheme, subject, seed), device)
            sets = prepare_residual_sets(model, prepared[subject], device)
            residual_sets[(subject, seed)] = sets
            stats = fit_residual_calibration(sets["train_nonfog"].residual, sigma_min)
            run_dir = stage / subject / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "calibration_stats.npz", **stats)
            for method in CALIBRATIONS:
                for clip in CLIPS:
                    name = f"{method}_clip{'none' if clip is None else int(clip)}"
                    calibrated: dict[str, np.ndarray] = {}
                    saturation: dict[str, float] = {}
                    for split, values in sets.items():
                        calibrated[split], saturation[split] = apply_residual_calibration(values.residual, stats, method, clip)
                    metrics = separation_metrics(residual_score(calibrated["validation_nonfog"]),
                                                 residual_score(calibrated["validation_fog"]),
                                                 residual_score(calibrated["train_nonfog"]))
                    validation_rows.append({"stage": "A3_residual_calibration", "selection_split": "validation",
                                            "scheme": name, "method": method, "clip": "none" if clip is None else clip,
                                            "subject_id": subject, "seed": seed,
                                            "validation_fog_windows": len(calibrated["validation_fog"]),
                                            "saturation_fraction_nonfog": saturation["validation_nonfog"],
                                            "saturation_fraction_fog": saturation["validation_fog"], **metrics})
    write_csv(stage / "validation_metrics.csv", validation_rows)
    baseline = [r for r in validation_rows if r["scheme"] == "C0_clipnone" and r["subject_id"] in SELECTION_SUBJECTS]
    schemes: list[dict[str, Any]] = []
    for method in CALIBRATIONS:
        for clip in CLIPS:
            name = f"{method}_clip{'none' if clip is None else int(clip)}"
            rows = [r for r in validation_rows if r["scheme"] == name and r["subject_id"] in SELECTION_SUBJECTS]
            ratio = median(rows, "fog_to_nonfog_median_ratio")
            effect = median(rows, "cliffs_delta")
            base_ratio = median(baseline, "fog_to_nonfog_median_ratio")
            base_effect = median(baseline, "cliffs_delta")
            valid = len(rows) == len(SELECTION_SUBJECTS) * len(SEEDS) and all(np.isfinite(float(r["auroc"])) for r in rows)
            preserve = bool(ratio >= 0.90 * base_ratio and effect >= 0.90 * base_effect)
            saturation = max(median(rows, "saturation_fraction_nonfog"), median(rows, "saturation_fraction_fog"))
            passed = bool(valid and preserve and saturation <= 0.01)
            schemes.append({"scheme": name, "method": method, "clip": "none" if clip is None else clip,
                            "valid_selection_runs": len(rows), "median_validation_nonfog_p95": median(rows, "nonfog_p95"),
                            "median_validation_nonfog_p90": median(rows, "nonfog_p90"),
                            "median_validation_false_alarm_per_minute_proxy": median(rows, "false_alarm_windows_per_minute_proxy"),
                            "median_validation_auroc": median(rows, "auroc"),
                            "median_validation_fog_nonfog_ratio": ratio, "C0_ratio": base_ratio,
                            "median_validation_cliffs_delta": effect, "C0_cliffs_delta": base_effect,
                            "maximum_median_saturation_fraction": saturation,
                            "effect_preserved": preserve, "gate_pass": passed})
    passed = [row for row in schemes if row["gate_pass"]]
    if not passed:
        selected = next(row for row in schemes if row["scheme"] == "C0_clipnone")
        status = "C0 FALLBACK PASS"
    else:
        selected = min(passed, key=lambda row: (row["median_validation_nonfog_p95"],
                       row["median_validation_false_alarm_per_minute_proxy"],
                       -row["median_validation_auroc"], CALIBRATIONS.index(row["method"])))
        status = "PASS"
    gate = {"stage": "A3_residual_calibration", "status": status, "advance_to_A4": True,
            "selected_scheme": selected["scheme"], "selection_subjects": list(SELECTION_SUBJECTS),
            "sparse_validation_diagnostic_subjects": list(SPARSE_DIAGNOSTIC_SUBJECTS),
            "selection_split": "validation only", "test_fog_used_for_selection": False, "schemes": schemes}
    write_json(stage / "A3_gate.json", gate)
    method = str(selected["method"])
    clip = None if selected["clip"] == "none" else float(selected["clip"])
    test_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            sets = residual_sets[(subject, seed)]
            stats = fit_residual_calibration(sets["train_nonfog"].residual, sigma_min)
            cal: dict[str, np.ndarray] = {}
            sat: dict[str, float] = {}
            for split, values in sets.items():
                cal[split], sat[split] = apply_residual_calibration(values.residual, stats, method, clip)
            metrics = separation_metrics(residual_score(cal["test_nonfog"]), residual_score(cal["test_fog"]),
                                         residual_score(cal["train_nonfog"]))
            test_rows.append({"stage": "A3_residual_calibration", "report_split": "test_after_freeze",
                              "scheme": selected["scheme"], "subject_id": subject, "seed": seed,
                              "test_nonfog_windows": len(cal["test_nonfog"]), "test_fog_windows": len(cal["test_fog"]),
                              "saturation_fraction_nonfog": sat["test_nonfog"],
                              "saturation_fraction_fog": sat["test_fog"], **metrics})
    write_csv(stage / "test_metrics_after_freeze.csv", test_rows)
    gate["test_summary_after_freeze"] = {
        "median_auroc": median(test_rows, "auroc"), "median_average_precision": median(test_rows, "average_precision"),
        "median_cliffs_delta": median(test_rows, "cliffs_delta"),
        "median_fog_nonfog_ratio": median(test_rows, "fog_to_nonfog_median_ratio")}
    write_json(stage / "A3_gate.json", gate)
    return gate, residual_sets


def spectral_features(values: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(values, axis=1)).astype(np.float64)
    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    features: list[np.ndarray] = []
    for low, high in ((0.5, 3.0), (3.0, 8.0), (0.5, 10.0)):
        mask = (frequency >= low) & (frequency <= high)
        features.append(np.mean(np.square(spectrum[:, mask, :]), axis=1))
    band = (frequency >= 0.5) & (frequency <= 10.0)
    band_spectrum = spectrum[:, band, :]
    band_frequency = frequency[band]
    dominant = band_frequency[np.argmax(band_spectrum, axis=1)]
    probability = band_spectrum / np.maximum(np.sum(band_spectrum, axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=1) / math.log(len(band_frequency))
    features.extend((dominant, entropy))
    return np.concatenate(features, axis=1).astype(np.float32)


def build_representation(name: str, x: np.ndarray, xhat: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if name == "R0":
        return np.ascontiguousarray(residual.astype(np.float32))
    if name == "R1":
        return np.ascontiguousarray(np.abs(residual).astype(np.float32))
    if name == "R2":
        return np.ascontiguousarray(np.diff(residual, axis=1, prepend=residual[:, :1, :]).astype(np.float32))
    if name == "R3":
        rms = np.sqrt(np.mean(np.square(residual), axis=1))
        center = np.median(residual, axis=1, keepdims=True)
        mad = np.median(np.abs(residual - center), axis=1)
        p90 = np.percentile(np.abs(residual), 90, axis=1)
        peak = np.ptp(residual, axis=1)
        return np.concatenate((rms, mad, p90, peak), axis=1).astype(np.float32)
    if name == "R4":
        return spectral_features(residual)
    if name == "R5":
        delta = np.diff(residual, axis=1, prepend=residual[:, :1, :])
        return np.concatenate((residual, np.abs(residual), delta), axis=2).astype(np.float32)
    if name == "R6":
        return np.concatenate((x, xhat, residual), axis=2).astype(np.float32)
    raise ValueError(name)


def representation_score(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    train_flat = train.reshape(len(train), -1).astype(np.float64)
    value_flat = values.reshape(len(values), -1).astype(np.float64)
    center = np.median(train_flat, axis=0, keepdims=True)
    mad = np.median(np.abs(train_flat - center), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 0.05)
    return np.median(np.abs((value_flat - center) / scale), axis=1)


def calibrated_sets(sets: dict[str, ResidualBundle], method: str, clip: float | None,
                    sigma_min: float) -> dict[str, ResidualBundle]:
    stats = fit_residual_calibration(sets["train_nonfog"].residual, sigma_min)
    result: dict[str, ResidualBundle] = {}
    for split, values in sets.items():
        residual, _ = apply_residual_calibration(values.residual, stats, method, clip)
        result[split] = ResidualBundle(values.x, values.xhat, residual)
    return result


def run_a4(root: Path, residual_sets: dict[tuple[str, int], dict[str, ResidualBundle]],
           a3_gate: dict[str, Any], sigma_min: float) -> dict[str, Any]:
    stage = root / "A4_representation_ablation"
    selected = next(row for row in a3_gate["schemes"] if row["scheme"] == a3_gate["selected_scheme"])
    method = str(selected["method"])
    clip = None if selected["clip"] == "none" else float(selected["clip"])
    validation_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for subject in SUBJECTS:
        for seed in SEEDS:
            sets = calibrated_sets(residual_sets[(subject, seed)], method, clip, sigma_min)
            for representation in REPRESENTATIONS:
                reps = {split: build_representation(representation, value.x, value.xhat, value.residual)
                        for split, value in sets.items()}
                cache[(subject, seed, representation)] = reps
                train_score = representation_score(reps["train_nonfog"], reps["train_nonfog"])
                nf_score = representation_score(reps["train_nonfog"], reps["validation_nonfog"])
                fog_score = representation_score(reps["train_nonfog"], reps["validation_fog"])
                metrics = separation_metrics(nf_score, fog_score, train_score)
                validation_rows.append({"stage": "A4_representation_ablation", "selection_split": "validation",
                                        "representation": representation, "subject_id": subject, "seed": seed,
                                        "shape": "x".join(str(v) for v in reps["validation_nonfog"].shape[1:]),
                                        "finite": bool(np.isfinite(reps["validation_nonfog"]).all() and np.isfinite(reps["validation_fog"]).all()),
                                        "validation_fog_windows": len(fog_score), **metrics})
                if seed == SEEDS[0]:
                    sample_dir = stage / "tensor_samples" / subject
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_dir / f"{representation}.npz",
                                        validation_nonfog=reps["validation_nonfog"][:8],
                                        validation_fog=reps["validation_fog"][:8])
    write_csv(stage / "validation_metrics.csv", validation_rows)
    summaries: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        rows = [r for r in validation_rows if r["representation"] == representation and r["subject_id"] in SELECTION_SUBJECTS]
        summaries.append({"representation": representation, "valid_selection_runs": len(rows),
                          "all_finite": all(r["finite"] for r in rows),
                          "median_validation_auroc": median(rows, "auroc"),
                          "median_validation_average_precision": median(rows, "average_precision"),
                          "median_validation_cliffs_delta": median(rows, "cliffs_delta"),
                          "median_validation_fog_nonfog_ratio": median(rows, "fog_to_nonfog_median_ratio"),
                          "median_validation_false_alarm_per_minute_proxy": median(rows, "false_alarm_windows_per_minute_proxy")})
    valid = [r for r in summaries if r["all_finite"] and r["valid_selection_runs"] == len(SELECTION_SUBJECTS) * len(SEEDS)]
    selected_rep = max(valid, key=lambda row: (row["median_validation_auroc"],
                       row["median_validation_cliffs_delta"], row["median_validation_average_precision"],
                       -REPRESENTATIONS.index(row["representation"])))
    gate = {"stage": "A4_representation_ablation", "status": "PASS",
            "selected_representation": selected_rep["representation"],
            "selection_split": "validation only", "selection_subjects": list(SELECTION_SUBJECTS),
            "sparse_validation_diagnostic_subjects": list(SPARSE_DIAGNOSTIC_SUBJECTS),
            "test_fog_used_for_selection": False, "representations": summaries,
            "automatic_next_stage": "STOPPED AFTER A4 AS AUTHORIZED"}
    write_json(stage / "A4_gate.json", gate)
    test_rows: list[dict[str, Any]] = []
    representation = selected_rep["representation"]
    for subject in SUBJECTS:
        for seed in SEEDS:
            reps = cache[(subject, seed, representation)]
            train_score = representation_score(reps["train_nonfog"], reps["train_nonfog"])
            nf_score = representation_score(reps["train_nonfog"], reps["test_nonfog"])
            fog_score = representation_score(reps["train_nonfog"], reps["test_fog"])
            metrics = separation_metrics(nf_score, fog_score, train_score)
            test_rows.append({"stage": "A4_representation_ablation", "report_split": "test_after_freeze",
                              "representation": representation, "subject_id": subject, "seed": seed,
                              "test_nonfog_windows": len(nf_score), "test_fog_windows": len(fog_score), **metrics})
    write_csv(stage / "test_metrics_after_freeze.csv", test_rows)
    gate["test_summary_after_freeze"] = {
        "median_auroc": median(test_rows, "auroc"), "median_average_precision": median(test_rows, "average_precision"),
        "median_cliffs_delta": median(test_rows, "cliffs_delta"),
        "median_fog_nonfog_ratio": median(test_rows, "fog_to_nonfog_median_ratio")}
    write_json(stage / "A4_gate.json", gate)
    return gate


def render_report(root: Path, a2_gate: dict[str, Any], a3_gate: dict[str, Any] | None,
                  a4_gate: dict[str, Any] | None) -> None:
    lines = ["# Daphnet NBM Route A：A2–A4 实验报告", "",
             f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
             "## 结论", "",
             f"- A2 状态：**{a2_gate['status']}**；选中 `{a2_gate['selected_scheme']}`。",
             f"- A2 选择未使用测试 FoG：`{str(not a2_gate['test_fog_used']).lower()}`。"]
    if a3_gate is not None:
        lines += [f"- A3 状态：**{a3_gate['status']}**；选中 `{a3_gate['selected_scheme']}`。",
                  f"- A3 冻结后测试中位 AUROC：{a3_gate['test_summary_after_freeze']['median_auroc']:.4f}。"]
    if a4_gate is not None:
        lines += [f"- A4 状态：**{a4_gate['status']}**；选中 `{a4_gate['selected_representation']}`。",
                  f"- A4 冻结后测试中位 AUROC：{a4_gate['test_summary_after_freeze']['median_auroc']:.4f}。",
                  "- 已按本次授权在 A4 后停止，未自动进入 A5。"]
    lines += ["", "## 防泄漏说明", "",
              "A2 只用测试 clean Non-FoG 及其合成扰动做去噪门控；A3/A4 仅用验证集 FoG/Non-FoG 选型。"
              "所有测试 FoG 指标均在方案冻结后计算，不参与排序。", "",
              "## 文件索引", "",
              "- `A2_denoising/A2_gate.json`：A2 完整门控。",
              "- `A3_residual_calibration/A3_gate.json`：A3 选择与冻结后测试摘要。",
              "- `A4_representation_ablation/A4_gate.json`：A4 选择与冻结后测试摘要。"]
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "A2_A4_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument("--parent", type=Path, default=ROOT / "outputs" / "daphnet_nbm_routeA_A1b_generalization_repair_v1" / "routeA_A1b_generalization_repair")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / EXPERIMENT / "routeA_A2_A4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "daphnet_nbm_routeA_A2_A4.yaml")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    parent = args.parent.resolve()
    root.mkdir(parents=True, exist_ok=True)
    a1_gate = json.loads((parent / "reports" / "A1_retest_gate.json").read_text(encoding="utf-8"))
    if not a1_gate.get("eligible_for_A2"):
        raise RuntimeError("A1 gate does not authorize A2")
    protocol = {"experiment": EXPERIMENT, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(args.config.resolve()), "parent": str(parent),
                "A1_gate_status": a1_gate["a1_retest_status"], "eligible_subjects": list(SUBJECTS),
                "diagnostic_only_subjects": ["S02", "S06"], "seeds": list(SEEDS),
                "frozen_model": "M3_tcdae_long+C0+L4+W0", "test_fog_used_for_selection": False}
    write_json(root / "protocol" / "frozen_A2_A4_protocol.json", protocol)
    dataset = a1.DaphnetDataset.load(args.data_dir.resolve())
    prepared = {subject: a1.prepare_subject(dataset, subject) for subject in SUBJECTS}
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    print(f"device={device} eligible_subjects={','.join(SUBJECTS)}", flush=True)
    all_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for scheme in DENOISING:
        for subject in SUBJECTS:
            item = prepared[subject]
            for seed in SEEDS:
                run_dir = root / "A2_denoising" / scheme / subject / f"seed{seed}"
                if scheme == "D0":
                    checkpoint = a2_checkpoint(root, parent, scheme, subject, seed)
                    if not checkpoint.exists():
                        raise FileNotFoundError(checkpoint)
                    model = load_model(checkpoint, device)
                    training = {"stage": "A2_denoising", "scheme": "D0", "subject_id": subject,
                                "seed": seed, "source_checkpoint": str(checkpoint), "reused_frozen_A1b_model": True}
                    run_dir.mkdir(parents=True, exist_ok=True)
                    write_json(run_dir / "source_checkpoint.json", training)
                else:
                    model, training = train_denoiser(item, scheme, seed, run_dir, device,
                                                     args.max_epochs, args.patience, args.workers)
                training_rows.append(training)
                rows = evaluate_a2_run(model, item, scheme, seed, run_dir, device)
                all_rows.extend(rows)
                print(f"A2 DONE {scheme} {subject} seed={seed}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    write_csv(root / "A2_denoising" / "all_metrics.csv", all_rows)
    write_csv(root / "A2_denoising" / "training_summary.csv", training_rows)
    a2_gate = make_a2_gate(all_rows)
    write_json(root / "A2_denoising" / "A2_gate.json", a2_gate)
    a3_gate: dict[str, Any] | None = None
    a4_gate: dict[str, Any] | None = None
    if a2_gate["advance_to_A3"]:
        print(f"A2 {a2_gate['status']} selected={a2_gate['selected_scheme']}; entering A3", flush=True)
        a3_gate, sets = run_a3(root, parent, prepared, a2_gate["selected_scheme"], device, args.sigma_min)
        if a3_gate["advance_to_A4"]:
            print(f"A3 {a3_gate['status']} selected={a3_gate['selected_scheme']}; entering A4", flush=True)
            a4_gate = run_a4(root, sets, a3_gate, args.sigma_min)
    render_report(root, a2_gate, a3_gate, a4_gate)
    final = {"experiment": EXPERIMENT, "A2": a2_gate, "A3": a3_gate, "A4": a4_gate,
             "completed_utc": datetime.now(timezone.utc).isoformat(), "test_fog_used_for_selection": False}
    write_json(root / "FINAL_RESULTS.json", final)
    print(f"COMPLETE results={root}", flush=True)


if __name__ == "__main__":
    main()
