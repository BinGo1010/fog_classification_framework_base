#!/usr/bin/env python3
"""Run one strict S01 LOSO fold on processed_NBM: GRU-v1 NBM -> scheme-C TCN."""

from __future__ import annotations

import argparse
import csv
import json
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

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.resume import atomic_json_dump, atomic_torch_save, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_gru_mask_strength_nbm300_fold import (
    PARAMETER_COUNT,
    architecture_config,
    augmentation_config,
    calibrate_gru_mask_strength,
    checkpoint_name,
    reconstruct_gru_mask_strength,
    train_gru_mask_strength_nbm,
)
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    build_scheme_c_features,
    paired_tcn_initial_states,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    METRIC_KEYS,
    ROLES,
    SUBJECTS,
    RoleRows,
    RobustScaler,
    load_fold_rows,
    raw_windows,
    write_csv,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    classifier_predict,
    plot_classifier_training,
    train_representation_tcn,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    choose_document_threshold,
    prepare_nbm_windows,
    residual_diagnostics,
    set_seed,
)

EXPERIMENT_VERSION = "processed_nbm_loso_s01_gru_mask8_12_c_tcn.v1"
TEST_SUBJECT = "S01"
VALIDATION_SUBJECT = "S02"
TRAIN_SUBJECTS = tuple(
    subject for subject in SUBJECTS if subject not in {TEST_SUBJECT, VALIDATION_SUBJECT}
)
DEVELOPMENT_SUBJECTS = (VALIDATION_SUBJECT, *TRAIN_SUBJECTS)
SOURCE_OUTER_FOLD = 0
NBM_SEED = 0
TCN_SEED = 0
WINDOW_SAMPLES = 128


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
        / "daphnet_processed_NBM_loso_S01_gru_mask8_12_C_tcn_ep5pat2_seed0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def validate_contract(args: argparse.Namespace) -> None:
    if (args.nbm_max_epochs, args.nbm_patience) != (300, 20):
        raise ValueError("GRU-v1 NBM is frozen to max300/pat20")
    if (args.tcn_max_epochs, args.tcn_patience) != (5, 2):
        raise ValueError("TCN is frozen to max5/pat2")


def implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "run_daphnet_gru_mask_strength_nbm300_fold.py",
        REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py",
        REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
        REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
        REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
        REPO_ROOT / "cnbr_fog" / "data.py",
        REPO_ROOT / "cnbr_fog" / "evaluation.py",
    )
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths}


def take_subjects(rows: RoleRows, subjects: tuple[str, ...]) -> RoleRows:
    mask = np.isin(rows.subject_id, np.asarray(subjects, dtype="U3"))
    return RoleRows(*(getattr(rows, field)[mask] for field in rows.__dataclass_fields__))


def fit_scaler_unique_points(
    records: dict[str, Any], rows: RoleRows
) -> tuple[RobustScaler, int]:
    if not len(rows) or not np.all(rows.role == 4) or not np.all(rows.label == 0):
        raise AssertionError("Scaler/NBM fit must contain development role-4 clean windows only")
    if TEST_SUBJECT in set(rows.subject_id.tolist()):
        raise AssertionError("S01 leaked into scaler/NBM fit")
    masks: dict[str, np.ndarray] = {}
    for record_id, start, end in zip(rows.record_id, rows.start, rows.end):
        record_id = str(record_id)
        masks.setdefault(record_id, np.zeros(len(records[record_id].x), dtype=bool))
        masks[record_id][int(start) : int(end)] = True
    values = np.concatenate(
        [records[record_id].x[mask] for record_id, mask in masks.items() if mask.any()],
        axis=0,
    ).astype(np.float64, copy=False)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    iqr = q75 - q25
    if np.any(iqr <= 1e-6):
        raise ValueError(f"degenerate scaler channels: {np.flatnonzero(iqr <= 1e-6)}")
    return RobustScaler(median.astype(np.float32), iqr.astype(np.float32)), int(len(values))


@torch.no_grad()
def clean_reconstruction_loss(
    model: torch.nn.Module, values: np.ndarray, device: torch.device
) -> float:
    reconstruction = reconstruct_gru_mask_strength(model, values, device)
    return float(
        nn.functional.smooth_l1_loss(
            torch.from_numpy(reconstruction), torch.from_numpy(values), beta=1.0
        )
    )


def feature_values(
    model: torch.nn.Module,
    scaler: RobustScaler,
    sigma: np.ndarray,
    records: dict[str, Any],
    rows: RoleRows,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = prepare_nbm_windows(scaler, raw_windows(records, rows), center=True)
    reconstruction = reconstruct_gru_mask_strength(model, scaled, device)
    error_bct = np.ascontiguousarray((scaled - reconstruction).transpose(0, 2, 1))
    features, clip_stats = build_scheme_c_features(
        error_bct, rows.label, sigma, WINDOW_SAMPLES
    )
    return features, {
        "windows": int(len(rows)),
        "subjects": sorted(set(rows.subject_id.tolist())),
        "shape": list(features.shape),
        "clip_statistics": clip_stats,
        "diagnostics": residual_diagnostics(features, rows.label),
    }


def manifest_rows(rows: RoleRows, split: str) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "subject_id": str(rows.subject_id[index]),
            "record_id": str(rows.record_id[index]),
            "window_id": str(rows.window_id[index]),
            "start_index": int(rows.start[index]),
            "end_index_exclusive": int(rows.end[index]),
            "source_role": int(rows.role[index]),
            "y_binary": int(rows.label[index]),
        }
        for index in range(len(rows))
    ]


def plot_nbm_history(output_dir: Path, training: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = training["history"]
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    axis.plot(epochs, [row["train_huber"] for row in history], label="Development role 4")
    axis.plot(
        epochs,
        [row["validation_huber"] for row in history],
        label="Development role 5",
    )
    axis.axvline(training["summary"]["best_epoch"], color="black", linestyle="--")
    axis.set(xlabel="Epoch", ylabel="SmoothL1", title="GRU-v1 NBM LOSO-S01")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "gru_nbm_training_validation.png", dpi=180)
    figure.savefig(output_dir / "gru_nbm_training_validation.svg")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    validate_contract(args)
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.is_file() and not args.overwrite:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        metrics_path = output_dir / "metrics.json"
        if (
            done.get("status") != "complete"
            or done.get("experiment_version") != EXPERIMENT_VERSION
            or not metrics_path.is_file()
        ):
            raise RuntimeError("existing output does not match this experiment")
        if done.get("metrics_sha256") != sha256_file(metrics_path):
            raise RuntimeError("existing metrics hash does not match DONE.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("implementation_sha256") != implementation_hashes():
            raise RuntimeError(
                "implementation changed; rerun with --overwrite or use a new output-dir"
            )
        print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)
        return
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "source_outer_fold": SOURCE_OUTER_FOLD,
                    "test_subject": TEST_SUBJECT,
                    "validation_subject": VALIDATION_SUBJECT,
                    "train_subjects": list(TRAIN_SUBJECTS),
                    "roles": {
                        "NBM/scaler fit": [4],
                        "NBM early stop/calibration": [5],
                        "TCN fit": [6, 7],
                        "TCN early stop/threshold": [2, 3],
                        "S01 outer test": list(ROLES),
                    },
                    "NBM": (
                        "GRU-v1 H64-z16-H64, 40/40/20 augmentation with "
                        "Mask8-12, scheme C, max300/pat20"
                    ),
                    "TCN": "27-channel RepresentationTCNM, max5/pat2",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    set_seed(NBM_SEED)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    data_dir = args.data_dir.resolve()
    scientific = processed_nbm_scientific_manifest(data_dir)
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 9:
        raise AssertionError("expected processed_NBM at 64 Hz with 9 channels")
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir, SOURCE_OUTER_FOLD)
    training_subject_rows = take_subjects(rows, TRAIN_SUBJECTS)
    validation_subject_rows = take_subjects(rows, (VALIDATION_SUBJECT,))
    heldout = take_subjects(rows, (TEST_SUBJECT,))
    split_rows = {
        "nbm_train": training_subject_rows.take_role(4),
        "nbm_validation": validation_subject_rows.take_role(5),
        "classifier_train": training_subject_rows.take_role(6, 7),
        "classifier_validation": validation_subject_rows.take_role(2, 3),
        # In a genuine outer LOSO fold, every retained pure window of the
        # held-out subject is test data.  The original role labels remain only
        # as provenance; none of them controls training for S01.
        "test": heldout,
    }
    for name, split in split_rows.items():
        if not len(split):
            raise AssertionError(f"empty split: {name}")
        subjects = set(split.subject_id.tolist())
        if name == "test" and subjects != {TEST_SUBJECT}:
            raise AssertionError("test split is not exclusively S01")
        if name != "test" and TEST_SUBJECT in subjects:
            raise AssertionError(f"S01 leaked into {name}")
        if name in {"nbm_train", "classifier_train"} and subjects != set(TRAIN_SUBJECTS):
            raise AssertionError(f"wrong training subjects in {name}: {subjects}")
        if name in {"nbm_validation", "classifier_validation"} and subjects != {
            VALIDATION_SUBJECT
        }:
            raise AssertionError(f"wrong validation subject in {name}: {subjects}")
    if set(split_rows["classifier_train"].label.tolist()) != {0, 1}:
        raise AssertionError("classifier training lacks a class")
    if set(split_rows["classifier_validation"].label.tolist()) != {0, 1}:
        raise AssertionError("classifier validation lacks a class")
    if set(split_rows["test"].label.tolist()) != {0, 1}:
        raise AssertionError("S01 test lacks a class")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    scaler, scaler_points = fit_scaler_unique_points(records, split_rows["nbm_train"])
    nbm_train_x = prepare_nbm_windows(
        scaler, raw_windows(records, split_rows["nbm_train"]), center=True
    )
    nbm_validation_x = prepare_nbm_windows(
        scaler, raw_windows(records, split_rows["nbm_validation"]), center=True
    )
    model, nbm_training = train_gru_mask_strength_nbm(
        nbm_train_x,
        nbm_validation_x,
        output_dir,
        device,
        NBM_SEED,
        args.num_workers,
        "MASK8_12",
        args.nbm_max_epochs,
        args.nbm_patience,
    )
    bias, sigma, calibration = calibrate_gru_mask_strength(
        model, nbm_validation_x, device
    )
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise AssertionError("GRU-v1 parameter count changed")
    nbm_train_clean_huber = clean_reconstruction_loss(model, nbm_train_x, device)
    nbm_validation_clean_huber = clean_reconstruction_loss(
        model, nbm_validation_x, device
    )
    nbm_domain_shift = {
        "train_subjects_role4_clean_huber": nbm_train_clean_huber,
        "validation_subject_S02_role5_clean_huber": nbm_validation_clean_huber,
        "ratio_validation_over_train": (
            nbm_validation_clean_huber / max(nbm_train_clean_huber, 1e-12)
        ),
    }

    train_x, train_feature = feature_values(
        model, scaler, sigma, records, split_rows["classifier_train"], device
    )
    validation_x, validation_feature = feature_values(
        model, scaler, sigma, records, split_rows["classifier_validation"], device
    )
    _, initial_state, initial_hashes = paired_tcn_initial_states(TCN_SEED)
    classifier, tcn_training = train_representation_tcn(
        "r_abs_delta",
        train_x,
        split_rows["classifier_train"].label,
        validation_x,
        split_rows["classifier_validation"].label,
        output_dir,
        device,
        TCN_SEED,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
        initial_state,
        reset_seed_after_loading=True,
    )
    validation_true, validation_prob = classifier_predict(
        classifier,
        validation_x,
        split_rows["classifier_validation"].label,
        device,
    )
    threshold, validation_metrics = choose_document_threshold(
        validation_true, validation_prob
    )

    # All trainable/tunable quantities are now frozen; S01 features are first
    # materialized only below this point.
    training_barrier = {
        "status": "all_training_validation_and_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_subject": TEST_SUBJECT,
        "threshold": float(threshold),
        "nbm_checkpoint_sha256": sha256_file(
            output_dir / "checkpoints" / checkpoint_name("MASK8_12")
        ),
        "tcn_checkpoint_sha256": sha256_file(output_dir / "checkpoints" / "tcn.pt"),
        "scientific_data_sha256": scientific["sha256"],
        "implementation_sha256": implementation_hashes(),
    }
    atomic_json_dump(training_barrier, output_dir / "TRAINING_BARRIER.json")

    test_x, test_feature = feature_values(
        model, scaler, sigma, records, split_rows["test"], device
    )
    test_true, test_prob = classifier_predict(
        classifier, test_x, split_rows["test"].label, device
    )
    test_metrics = binary_metrics(test_true, test_prob, float(threshold))
    test_pred = (test_prob >= float(threshold)).astype(np.int8)
    write_csv(
        output_dir / "test_predictions.csv",
        [
            {
                **manifest_rows(split_rows["test"], "test")[index],
                "fog_probability": float(test_prob[index]),
                "threshold": float(threshold),
                "y_pred": int(test_pred[index]),
            }
            for index in range(len(test_true))
        ],
    )
    np.savez_compressed(
        output_dir / "test_probabilities.npz",
        y_true=test_true,
        y_prob=test_prob,
        y_pred=test_pred,
        threshold=np.asarray(float(threshold)),
    )
    all_manifest = []
    for name, split in split_rows.items():
        all_manifest.extend(manifest_rows(split, name))
    write_csv(output_dir / "split_manifest.csv", all_manifest)
    plot_nbm_history(output_dir, nbm_training)
    plot_classifier_training(
        output_dir,
        "GRU_V1_C_LOSO_S01",
        tcn_training,
        test_metrics["confusion_matrix"],
    )

    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_dir),
        "scientific_data_sha256": scientific["sha256"],
        "source_outer_fold": SOURCE_OUTER_FOLD,
        "test_subject": TEST_SUBJECT,
        "validation_subject": VALIDATION_SUBJECT,
        "train_subjects": list(TRAIN_SUBJECTS),
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "split_policy": (
            "Nested subject-disjoint LOSO on canonical processed_NBM outer0: "
            "fit on S03,S05,S06,S07,S08,S09 roles4 and 6/7; validate/calibrate "
            "on S02 roles5 and 2/3; evaluate every retained pure S01 window "
            "(original S01 roles0-7 are provenance only)"
        ),
        "split_counts": {
            name: {
                "windows": int(len(split)),
                "nonfog": int(np.sum(split.label == 0)),
                "fog": int(np.sum(split.label == 1)),
                "subjects": sorted(set(split.subject_id.tolist())),
                "roles": sorted(set(split.role.astype(int).tolist())),
            }
            for name, split in split_rows.items()
        },
        "scaler": {
            **scaler.as_dict(),
            "fit_unique_raw_points": scaler_points,
            "fit_subjects": list(TRAIN_SUBJECTS),
            "fit_role": 4,
        },
        "nbm": {
            "architecture": architecture_config(),
            "augmentation": augmentation_config("MASK8_12"),
            "training": nbm_training["summary"],
            "domain_shift_diagnostic": nbm_domain_shift,
            "calibration": calibration,
            "bias_used_in_scheme_c": False,
            "sigma_used_in_scheme_c": sigma.astype(float).tolist(),
        },
        "feature_contract": (
            "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
            "r=q-mean_t(q); F=[r,abs(r),delta] [B,27,128]"
        ),
        "classifier": {
            **{key: value for key, value in tcn_training.items() if key != "history"},
            "initial_27ch_state_sha256": initial_hashes["r_abs_delta"],
            "threshold": float(threshold),
            "threshold_source": "development subjects roles2/3",
            "threshold_rule": "max balanced accuracy; ties F1 then higher threshold",
        },
        "features": {
            "classifier_train": train_feature,
            "classifier_validation": validation_feature,
            "test": test_feature,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_barrier": training_barrier,
        "implementation_sha256": implementation_hashes(),
    }
    atomic_json_dump(result, output_dir / "metrics.json")
    atomic_json_dump(
        {
            "status": "complete",
            "experiment_version": EXPERIMENT_VERSION,
            "test_subject": TEST_SUBJECT,
            "threshold": float(threshold),
            "sensitivity": float(test_metrics["sensitivity"]),
            "precision": float(test_metrics["precision"]),
            "specificity": float(test_metrics["specificity"]),
            "pr_auc": float(test_metrics["auprc"]),
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        },
        done_path,
    )
    print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
