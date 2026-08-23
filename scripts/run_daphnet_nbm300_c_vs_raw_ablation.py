#!/usr/bin/env python3
"""Strict paired TCN representation ablations on processed_NBM.

FULL_C
    role-4 RobustScaler -> per-window/per-axis centering -> selected frozen NBM
    -> e=X-Xhat -> scheme-C r -> [r,abs(r),delta(r)] -> TCN [B,27,T].

RAW
    the identical role-4 RobustScaler -> per-window/per-axis centering
    -> TCN [B,9,T].  The NBM, role-5 b/sigma, and residual path are absent.

RESIDUAL_R
    the identical FULL_C path through centered scheme-C residual r, followed
    directly by TCN [B,9,T]; abs(r) and delta(r) are the only removed terms.

Stages enforce a global test barrier: all configured classifiers and validation-only
thresholds must be frozen before roles 0/1 can be materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    RobustScaler,
    choose_document_threshold,
    classifier_predict,
    load_fold_rows,
    residual_diagnostics,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
    centered_scaled_bct,
    paired_tcn_initial_states,
    plot_classifier_training,
    train_representation_tcn,
)
from scripts.run_daphnet_residual_calibration_abcd import (
    load_frozen_nbm,
    reconstruction_error,
    sha256_file,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    prepare_nbm_windows,
    reconstruct as reconstruct_gru,
)
from scripts.run_daphnet_gru_v2_nbm300_fold import (
    ARCHITECTURE_NAME as GRU_V2_ARCHITECTURE_NAME,
    PhaseConditionedGRUNBM,
    reconstruct_gru_v2,
)
from scripts.run_daphnet_tcn_v2_nbm300_fold import (
    ARCHITECTURE_NAME as TCN_V2_ARCHITECTURE_NAME,
    GlobalBottleneckTCNNBM,
    reconstruct_tcn_v2,
)
from scripts.run_daphnet_restcn_attention_pool_nbm300_fold import (
    ARCHITECTURE_NAME as TCN_ATTN_Z16_ARCHITECTURE_NAME,
    CHECKPOINT_NAME as TCN_ATTN_Z16_CHECKPOINT_NAME,
    ResTCNSingleQueryAttentionPoolNBM,
    reconstruct_attention_pool_nbm,
)
from scripts.run_daphnet_transformer_nbm300_fold import (
    PatchTransformerNBM,
    reconstruct_transformer,
)
from scripts.run_daphnet_transformer_ngm_48k_fold import (
    ARCHITECTURE_NAME as TRANSFORMER_48K_ARCHITECTURE_NAME,
    PARAMETER_COUNT as TRANSFORMER_48K_PARAMETER_COUNT,
    PatchTransformerNGM48K,
    reconstruct_transformer_48k,
)
from scripts.mlp_ngm_30x128 import (
    MLP_NGM_9_PARAMETER_COUNT,
    FactorizedMLPNGM9,
    reconstruct_bct as reconstruct_mlp_bct,
)

FOLDS = (0, 1, 2)
METHODS = ("FULL_C", "RAW")
SUPPORTED_METHODS = (*METHODS, "RESIDUAL_R")
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auprc",
    "auroc",
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


def parse_csv_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods or len(methods) != len(set(methods)):
        raise ValueError(f"invalid unique method list: {value}")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(
            f"unsupported methods {unknown}; expected a subset of {SUPPORTED_METHODS}"
        )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train", "seal", "evaluate", "aggregate"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument(
        "--nbm-source-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_conv_tcn_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161"
        / "nbm_source",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_conv_tcn_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--method", choices=SUPPORTED_METHODS)
    parser.add_argument(
        "--experiment-methods",
        default=",".join(METHODS),
        help=(
            "Comma-separated methods sealed and aggregated together. "
            "RESIDUAL_R is scheme-C r without abs(r) or delta(r)."
        ),
    )
    parser.add_argument(
        "--nbm-kind",
        choices=(
            "conv_tcn",
            "gru",
            "gru_v2",
            "tcn_v2",
            "tcn_attn_z16",
            "transformer",
            "transformer_48k",
            "mlp",
        ),
        default="conv_tcn",
    )
    parser.add_argument("--nbm-seed", type=int)
    parser.add_argument("--tcn-seed", type=int)
    parser.add_argument("--nbm-seeds", default="0,52,161")
    parser.add_argument("--tcn-seeds", default="0,52,161")
    parser.add_argument("--required-seeds", default="0,52,161")
    parser.add_argument("--sampling-rate-hz", type=int, default=64)
    parser.add_argument("--window-samples", type=int, default=128)
    parser.add_argument("--stride-samples", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=10)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--required-nbm-max-epochs", type=int, default=300)
    parser.add_argument("--required-nbm-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def require_job_args(args: argparse.Namespace) -> None:
    if (
        args.fold is None
        or args.method is None
        or args.nbm_seed is None
        or args.tcn_seed is None
    ):
        raise ValueError(
            f"--fold, --method, --nbm-seed and --tcn-seed are required for {args.stage}"
        )
    if args.nbm_seed != args.tcn_seed:
        raise ValueError("strict paired repeats require NBM seed == TCN seed")
    methods = parse_csv_methods(args.experiment_methods)
    if args.method not in methods:
        raise ValueError(
            f"job method {args.method!r} is absent from --experiment-methods {methods}"
        )
    required_seeds = parse_csv_ints(args.required_seeds)
    if args.nbm_seed not in required_seeds:
        raise ValueError(f"seed must be one of {required_seeds}")


def job_id(fold: int, method: str, seed: int) -> str:
    return f"fold{fold}_method{method}_seed{seed}"


def job_dir(root: Path, fold: int, method: str, seed: int) -> Path:
    return root / "runs" / f"fold_{fold}" / f"method_{method}" / f"seed_{seed}"


def expected_jobs(
    seeds: tuple[int, ...],
    methods: tuple[str, ...] = METHODS,
) -> list[tuple[int, str, int]]:
    return [
        (fold, method, seed)
        for fold in FOLDS
        for method in methods
        for seed in seeds
    ]


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records_rows(data_dir: Path, fold: int) -> tuple[dict[str, Any], Any]:
    dataset = DaphnetDataset.load(data_dir.resolve())
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir.resolve(), fold)
    expected = np.isin(rows.role, [1, 3, 7]).astype(np.int8)
    if not np.array_equal(rows.label, expected):
        raise AssertionError(f"role/label mismatch in fold {fold}")
    return records, rows


def audit_protocol_dynamic(
    data_dir: Path,
    rows_by_fold: dict[int, Any],
    sampling_rate_hz: int,
    window_samples: int,
    stride_samples: int,
) -> dict[str, Any]:
    quality_path = data_dir / "nbm_quality_report.json"
    protocol_path = data_dir / "nbm_protocol.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not quality.get("overall_pass", False):
        raise AssertionError("processed_NBM quality report does not pass")
    expected = {
        "sampling_rate_hz": sampling_rate_hz,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
    }
    for key, value in expected.items():
        if int(protocol[key]) != value:
            raise AssertionError(
                f"protocol {key}={protocol[key]} does not match requested {value}"
            )
    fold_rows = []
    reference_test_ids: set[str] | None = None
    for fold, rows in sorted(rows_by_fold.items()):
        lengths = np.asarray(rows.end - rows.start, dtype=np.int64)
        if not np.all(lengths == window_samples):
            raise AssertionError(f"fold {fold} contains an invalid window length")
        expected_labels = np.isin(rows.role, [1, 3, 7]).astype(np.int8)
        if not np.array_equal(rows.label, expected_labels):
            raise AssertionError(f"fold {fold} role/label mismatch")
        role_counts = {str(role): int(np.sum(rows.role == role)) for role in ROLES}
        if any(count <= 0 for count in role_counts.values()):
            raise AssertionError(f"fold {fold} has an empty required role")
        test_ids = set(rows.window_id[np.isin(rows.role, [0, 1])].tolist())
        if reference_test_ids is None:
            reference_test_ids = test_ids
        elif test_ids != reference_test_ids:
            raise AssertionError("permanent role-0/1 windows differ across folds")
        fold_rows.append({"fold": fold, "role_counts": role_counts})
    return {
        "overall_pass": True,
        "dataset_id": protocol.get("dataset_id"),
        **expected,
        "quality_report_sha256": sha256_file(quality_path),
        "protocol_sha256": sha256_file(protocol_path),
        "folds": fold_rows,
        "permanent_test_identical_across_folds": True,
    }


def build_test_data_manifest(
    data_dir: Path,
    rows_by_fold: dict[int, Any],
) -> dict[str, Any]:
    """Bind the permanent-test rows and underlying record bytes to the seal."""
    data_dir = data_dir.resolve()
    fold_manifests: dict[str, dict[str, Any]] = {}
    referenced_records: set[str] = set()
    fold_hashes: set[str] = set()
    for fold, rows in sorted(rows_by_fold.items()):
        indices = np.flatnonzero(np.isin(rows.role, [0, 1]))
        entries = [
            {
                "subject_id": str(rows.subject_id[index]),
                "record_id": str(rows.record_id[index]),
                "window_id": str(rows.window_id[index]),
                "start": int(rows.start[index]),
                "end": int(rows.end[index]),
                "role": int(rows.role[index]),
                "label": int(rows.label[index]),
            }
            for index in indices
        ]
        entries.sort(
            key=lambda item: (
                item["subject_id"],
                item["record_id"],
                item["start"],
                item["end"],
                item["window_id"],
            )
        )
        window_sha256 = stable_json_hash(entries)
        fold_hashes.add(window_sha256)
        fold_manifests[str(fold)] = {
            "window_count": len(entries),
            "window_manifest_sha256": window_sha256,
        }
        referenced_records.update(item["record_id"] for item in entries)
    if len(fold_hashes) != 1:
        raise AssertionError("permanent role-0/1 manifests differ across folds")

    record_files = []
    for record_id in sorted(referenced_records):
        path = data_dir / "records" / f"{record_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"permanent-test record file missing: {path}")
        record_files.append(
            {
                "record_id": record_id,
                "relative_path": path.relative_to(data_dir).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    core = {
        "schema": "permanent_test_data_manifest.v1",
        "definition": (
            "canonical role0/1 subject,record,start,end,role,label,window_id plus "
            "SHA256 of every referenced record npz"
        ),
        "protocol_sha256": sha256_file(data_dir / "nbm_protocol.json"),
        "quality_report_sha256": sha256_file(
            data_dir / "nbm_quality_report.json"
        ),
        "folds": fold_manifests,
        "record_files": record_files,
    }
    return {**core, "sha256": stable_json_hash(core)}


def barrier_identity_payload(barrier: dict[str, Any]) -> dict[str, Any]:
    """Canonical fields protected by barrier_id; timestamps are excluded."""
    return {
        "barrier_schema": barrier["barrier_schema"],
        "status": barrier["status"],
        "folds": barrier["folds"],
        "methods": barrier["methods"],
        "nbm_seeds": barrier["nbm_seeds"],
        "tcn_seeds": barrier["tcn_seeds"],
        "job_count": barrier["job_count"],
        "strict_test_gate": barrier["strict_test_gate"],
        "source_audit": barrier["source_audit"],
        "test_data_manifest_sha256": barrier["test_data_manifest"]["sha256"],
        "jobs": barrier["jobs"],
    }


def load_and_validate_barrier(path: Path) -> dict[str, Any]:
    barrier = json.loads(path.read_text(encoding="utf-8"))
    schema = barrier.get("barrier_schema")
    if schema == "strict_test_barrier.v2":
        expected = stable_json_hash(barrier_identity_payload(barrier))
        if barrier.get("barrier_id") != expected:
            raise AssertionError("TRAINING_BARRIER identity hash mismatch")
    elif schema is not None or "barrier_id" in barrier:
        raise AssertionError(f"unsupported TRAINING_BARRIER schema: {schema}")
    return barrier


def require_strict_barrier_for_tcn_v2(
    barrier: dict[str, Any],
    nbm_kind: str,
) -> None:
    if (
        nbm_kind in ("tcn_v2", "tcn_attn_z16")
        and barrier.get("barrier_schema") != "strict_test_barrier.v2"
    ):
        raise RuntimeError(
            f"{nbm_kind} requires strict_test_barrier.v2; rerun classifier train/seal "
            "or use a clean output-root"
        )


def load_scaler_only(
    source_root: Path,
    fold: int,
    nbm_kind: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    """Load only the role-4 scaler; do not instantiate NBM or read b/sigma."""
    fold_dir = source_root.resolve() / f"fold_{fold}"
    frozen_path = fold_dir / "nbm_frozen.json"
    checkpoint_names = {
        "conv_tcn": "conv_tcn_nbm_best.pt",
        "gru": "gru_nbm_best.pt",
        "gru_v2": "gru_v2_nbm_best.pt",
        "tcn_v2": "tcn_v2_nbm_best.pt",
        "tcn_attn_z16": TCN_ATTN_Z16_CHECKPOINT_NAME,
        "transformer": "transformer_nbm_best.pt",
        "transformer_48k": "transformer_nbm_best.pt",
        "mlp": "mlp_ngm_best.pt",
    }
    checkpoint_name = checkpoint_names[nbm_kind]
    checkpoint = fold_dir / "checkpoints" / checkpoint_name
    if not frozen_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"frozen NBM/scaler artifacts missing: {fold_dir}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload = frozen["scaler"]
    scaler = RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float32),
        iqr=np.asarray(payload["iqr"], dtype=np.float32),
        epsilon=float(payload.get("epsilon", 1e-6)),
    )
    manifest = {
        "fold": fold,
        "nbm_kind": nbm_kind,
        "scaler_fit_role": int(frozen["scaler_fit_role"]),
        "scaler_unique_raw_points": int(frozen["scaler_unique_raw_points"]),
        "scaler_sha256": stable_json_hash(payload),
        "frozen_json": str(frozen_path.resolve()),
        "frozen_json_sha256": sha256_file(frozen_path),
        "nbm_checkpoint": str(checkpoint.resolve()),
        "nbm_checkpoint_sha256": sha256_file(checkpoint),
    }
    return scaler, manifest, frozen


def validate_nbm_contract(frozen: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = frozen["training"]
    augmentation = training["augmentation"]
    architecture = training["architecture"]
    expected_aug = {
        "clean_probability": 0.40,
        "gaussian_probability": 0.40,
        "mask_probability": 0.20,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_all_channels": True,
    }
    checks = {
        "maximum_epochs": int(training["maximum_epochs"]) == args.required_nbm_max_epochs,
        "patience": int(training["patience"]) == args.required_nbm_patience,
        "loss": str(training["loss"]) == "SmoothL1(beta=1.0)",
        "optimizer_lr": "lr=0.001" in str(training["optimizer"]),
        "augmentation": all(augmentation.get(key) == value for key, value in expected_aug.items()),
        "best_checkpoint_restored_before_calibration": bool(
            frozen.get("best_checkpoint_restored_before_calibration", False)
        ),
        "validation_unaugmented": frozen.get("validation_mask_or_noise") is False,
        "exact_seed": (
            args.nbm_seed is not None
            and int(training["seed"]) == args.nbm_seed
        ),
        "architecture": str(architecture["name"]).startswith(
            {
                "conv_tcn": "conv_tcn",
                "gru": "gru_reconstruction",
                "gru_v2": GRU_V2_ARCHITECTURE_NAME,
                "tcn_v2": TCN_V2_ARCHITECTURE_NAME,
                "tcn_attn_z16": TCN_ATTN_Z16_ARCHITECTURE_NAME,
                "transformer": "transformer_patch_autoencoder",
                "transformer_48k": "tiny_patch_transformer_ngm",
                "mlp": "factorized_mlp_ngm",
            }[args.nbm_kind]
        ),
        "gru_v2_architecture_details": (
            args.nbm_kind != "gru_v2"
            or (
                architecture.get("name") == GRU_V2_ARCHITECTURE_NAME
                and architecture.get("input_shape") == ["B", 128, 9]
                and architecture.get("encoder", {}).get("layers") == 1
                and architecture.get("encoder", {}).get("hidden_per_direction")
                == 96
                and architecture.get("bottleneck_shape") == ["B", 16]
                and architecture.get("decoder", {}).get("layers") == 2
                and architecture.get("decoder", {}).get("hidden") == 96
                and architecture.get("decoder_conditioning", {}).get(
                    "raw_or_encoder_token_connection"
                )
                is False
                and architecture.get("encoder_decoder_skip_connections") is False
                and architecture.get("teacher_forcing") is False
                and int(architecture.get("parameter_count", -1)) == 172_697
            )
        ),
        "tcn_v2_architecture_details": (
            args.nbm_kind != "tcn_v2"
            or (
                architecture.get("name") == TCN_V2_ARCHITECTURE_NAME
                and architecture.get("input_shape") == ["B", 9, 128]
                and architecture.get("bottleneck_shape") == ["B", 16]
                and architecture.get("output_shape") == ["B", 9, 128]
                and architecture.get("decoder_conditioning", {}).get(
                    "raw_or_encoder_temporal_connection"
                )
                is False
                and architecture.get("decoder_conditioning", {}).get(
                    "time_code_trainable"
                )
                is False
                and architecture.get("encoder_decoder_skip_connections") is False
                and architecture.get("teacher_forcing") is False
                and float(architecture.get("dropout", -1.0)) == 0.10
                and int(architecture.get("parameter_count", -1)) == 186_065
            )
        ),
        "tcn_attn_z16_architecture_details": (
            args.nbm_kind != "tcn_attn_z16"
            or (
                architecture.get("name") == TCN_ATTN_Z16_ARCHITECTURE_NAME
                and architecture.get("input_shape") == ["B", 9, 128]
                and architecture.get("encoder_token_shape") == ["B", 32, 48]
                and architecture.get("attention_pool", {}).get("query_shape")
                == [1, 1, 48]
                and architecture.get("attention_pool", {}).get("heads") == 4
                and architecture.get("attention_pool", {}).get("embed_dim") == 48
                and architecture.get("attention_pool", {}).get(
                    "position_code_trainable"
                )
                is False
                and architecture.get("attention_pool", {}).get(
                    "raw_or_encoder_token_residual_bypass"
                )
                is False
                and architecture.get("bottleneck_shape") == ["B", 16]
                and architecture.get("output_shape") == ["B", 9, 128]
                and architecture.get("decoder_conditioning", {}).get(
                    "raw_or_encoder_temporal_connection"
                )
                is False
                and architecture.get("decoder_conditioning", {}).get(
                    "time_code_trainable"
                )
                is False
                and architecture.get("encoder_decoder_skip_connections") is False
                and architecture.get("input_output_global_residual") is False
                and architecture.get("teacher_forcing") is False
                and float(architecture.get("dropout", -1.0)) == 0.10
                and int(architecture.get("parameter_count", -1)) == 171_905
            )
        ),
        "transformer_architecture_details": (
            args.nbm_kind != "transformer"
            or (
                architecture.get("input_shape") == ["B", 9, 128]
                and architecture.get("patchify", {}).get("patch_size") == 8
                and architecture.get("patchify", {}).get("token_shape")
                == ["B", 16, 72]
                and architecture.get("encoder", {}).get("layers") == 4
                and architecture.get("encoder", {}).get("d_model") == 192
                and architecture.get("encoder", {}).get("heads") == 6
                and architecture.get("encoder", {}).get("ffn") == 576
                and architecture.get("bottleneck_shape") == ["B", 8, 64]
                and architecture.get("decoder", {}).get("layers") == 2
                and architecture.get("decoder", {}).get("d_model") == 192
                and architecture.get("decoder", {}).get("heads") == 6
                and architecture.get("decoder", {}).get("ffn") == 576
                and architecture.get("encoder_decoder_skip_connections") is False
                and int(architecture.get("parameter_count", -1)) == 2_329_736
            )
        ),
        "transformer_48k_architecture_details": (
            args.nbm_kind != "transformer_48k"
            or (
                architecture.get("name") == TRANSFORMER_48K_ARCHITECTURE_NAME
                and architecture.get("input_shape") == ["B", 9, 128]
                and architecture.get("patchify", {}).get("patch_size") == 8
                and architecture.get("patchify", {}).get("token_shape")
                == ["B", 16, 72]
                and architecture.get("encoder", {}).get("layers") == 2
                and architecture.get("encoder", {}).get("d_model") == 40
                and architecture.get("encoder", {}).get("heads") == 4
                and architecture.get("encoder", {}).get("ffn") == 80
                and architecture.get("bottleneck_shape") == ["B", 16]
                and architecture.get("decoder", {}).get("layers") == 1
                and architecture.get("decoder", {}).get("d_model") == 40
                and architecture.get("decoder", {}).get("heads") == 4
                and architecture.get("decoder", {}).get("ffn") == 80
                and architecture.get("encoder_decoder_skip_connections") is False
                and architecture.get("cross_attention") is False
                and architecture.get("teacher_forcing") is False
                and architecture.get("raw_input_bypass") is False
                and int(architecture.get("parameter_count", -1))
                == TRANSFORMER_48K_PARAMETER_COUNT
            )
        ),
        "mlp_architecture_details": (
            args.nbm_kind != "mlp"
            or (
                architecture.get("name") == "factorized_mlp_ngm_v1_9channel"
                and architecture.get("input_shape") == ["B", 9, 128]
                and architecture.get("bottleneck_shape") == ["B", 16, 32]
                and architecture.get("output_shape") == ["B", 9, 128]
                and architecture.get("channel_mlp") == "9->16->9"
                and architecture.get("encoder_decoder_skip_connections") is False
                and architecture.get("output_activation") is None
                and int(architecture.get("parameter_count", -1))
                == MLP_NGM_9_PARAMETER_COUNT
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"frozen NBM does not match requested best configuration: {failed}")
    return {
        "max_epochs": args.required_nbm_max_epochs,
        "patience": args.required_nbm_patience,
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "augmentation": expected_aug,
        "checkpoint_rule": "lowest unaugmented role-5 validation SmoothL1",
        "all_checks_passed": True,
        "seed": args.nbm_seed,
        "nbm_kind": args.nbm_kind,
    }


def load_frozen_gru_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[GRUReconstructionNBM, RobustScaler, np.ndarray, np.ndarray, dict[str, Any]]:
    scaler, artifact, frozen = load_scaler_only(source_root, fold, "gru")
    training = frozen["training"]
    if training["architecture"]["name"] != "gru_reconstruction_nbm_v1":
        raise AssertionError("unexpected frozen GRU-NBM architecture")
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if bias.shape != (9,) or sigma.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError("invalid frozen GRU-NBM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
    }
    return model, scaler, bias, sigma, manifest


def load_frozen_gru_v2_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[
    PhaseConditionedGRUNBM,
    RobustScaler,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    scaler, artifact, frozen = load_scaler_only(source_root, fold, "gru_v2")
    training = frozen["training"]
    architecture = training["architecture"]
    if architecture["name"] != GRU_V2_ARCHITECTURE_NAME:
        raise AssertionError("unexpected frozen GRU-v2 NBM architecture")
    if int(architecture.get("parameter_count", -1)) != 172_697:
        raise AssertionError("unexpected frozen GRU-v2 NBM parameter count")
    model = PhaseConditionedGRUNBM().to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture:
        raise AssertionError("GRU-v2 checkpoint/frozen architecture mismatch")
    model.load_state_dict(payload["model_state"])
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if bias.shape != (9,) or sigma.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError("invalid frozen GRU-v2 NBM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
        "architecture": architecture,
    }
    return model, scaler, bias, sigma, manifest


def load_frozen_tcn_v2_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[
    GlobalBottleneckTCNNBM,
    RobustScaler,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    scaler, artifact, frozen = load_scaler_only(source_root, fold, "tcn_v2")
    training = frozen["training"]
    architecture = training["architecture"]
    if architecture.get("name") != TCN_V2_ARCHITECTURE_NAME:
        raise AssertionError("unexpected frozen TCN-v2 NBM architecture")
    if int(architecture.get("parameter_count", -1)) != 186_065:
        raise AssertionError("unexpected frozen TCN-v2 NBM parameter count")
    if architecture.get("bottleneck_shape") != ["B", 16]:
        raise AssertionError("unexpected frozen TCN-v2 NBM bottleneck")
    model = GlobalBottleneckTCNNBM(
        dropout=float(architecture.get("dropout", 0.10))
    ).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture:
        raise AssertionError("TCN-v2 checkpoint/frozen architecture mismatch")
    if payload.get("augmentation") != training.get("augmentation"):
        raise AssertionError("TCN-v2 checkpoint/frozen augmentation mismatch")
    if int(payload.get("seed", -1)) != int(training["seed"]):
        raise AssertionError("TCN-v2 checkpoint/frozen seed mismatch")
    if int(payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError("TCN-v2 checkpoint/frozen best epoch mismatch")
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=1e-7,
        atol=1e-10,
    ):
        raise AssertionError("TCN-v2 checkpoint/frozen validation loss mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if (
        bias.shape != (9,)
        or sigma.shape != (9,)
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma < 0.05)
    ):
        raise AssertionError("invalid frozen TCN-v2 NBM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
        "architecture": architecture,
    }
    return model, scaler, bias, sigma, manifest


def load_frozen_tcn_attention_z16_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[
    ResTCNSingleQueryAttentionPoolNBM,
    RobustScaler,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    scaler, artifact, frozen = load_scaler_only(
        source_root, fold, "tcn_attn_z16"
    )
    training = frozen["training"]
    architecture = training["architecture"]
    if architecture.get("name") != TCN_ATTN_Z16_ARCHITECTURE_NAME:
        raise AssertionError("unexpected frozen TCN-attention-Z16 architecture")
    if int(architecture.get("parameter_count", -1)) != 171_905:
        raise AssertionError("unexpected frozen TCN-attention-Z16 parameter count")
    if architecture.get("bottleneck_shape") != ["B", 16]:
        raise AssertionError("unexpected frozen TCN-attention-Z16 bottleneck")
    if architecture.get("encoder_decoder_skip_connections") is not False:
        raise AssertionError("TCN-attention-Z16 must not contain long skips")

    model = ResTCNSingleQueryAttentionPoolNBM(
        dropout=float(architecture.get("dropout", 0.10))
    ).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture:
        raise AssertionError(
            "TCN-attention-Z16 checkpoint/frozen architecture mismatch"
        )
    if payload.get("augmentation") != training.get("augmentation"):
        raise AssertionError(
            "TCN-attention-Z16 checkpoint/frozen augmentation mismatch"
        )
    if int(payload.get("seed", -1)) != int(training["seed"]):
        raise AssertionError("TCN-attention-Z16 checkpoint/frozen seed mismatch")
    if int(payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError(
            "TCN-attention-Z16 checkpoint/frozen best epoch mismatch"
        )
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=1e-7,
        atol=1e-10,
    ):
        raise AssertionError(
            "TCN-attention-Z16 checkpoint/frozen validation loss mismatch"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if (
        bias.shape != (9,)
        or sigma.shape != (9,)
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma < 0.05)
    ):
        raise AssertionError("invalid frozen TCN-attention-Z16 role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
        "architecture": architecture,
    }
    return model, scaler, bias, sigma, manifest


def load_frozen_transformer_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[PatchTransformerNBM, RobustScaler, np.ndarray, np.ndarray, dict[str, Any]]:
    scaler, artifact, frozen = load_scaler_only(source_root, fold, "transformer")
    training = frozen["training"]
    architecture = training["architecture"]
    if architecture["name"] != "transformer_patch_autoencoder_nbm_v1":
        raise AssertionError("unexpected frozen Transformer-NBM architecture")
    dropout = float(architecture.get("dropout", 0.10))
    model = PatchTransformerNBM(dropout=dropout).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if bias.shape != (9,) or sigma.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError("invalid frozen Transformer-NBM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
    }
    return model, scaler, bias, sigma, manifest


def load_frozen_transformer_48k_ngm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[
    PatchTransformerNGM48K,
    RobustScaler,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    scaler, artifact, frozen = load_scaler_only(
        source_root, fold, "transformer_48k"
    )
    training = frozen["training"]
    architecture = training["architecture"]
    probe = PatchTransformerNGM48K(dropout=0.10)
    expected_architecture = probe.architecture_config()
    del probe
    if architecture != expected_architecture:
        raise AssertionError("unexpected frozen compact Transformer-NGM architecture")
    model = PatchTransformerNGM48K(dropout=0.10).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != expected_architecture:
        raise AssertionError("compact Transformer-NGM checkpoint architecture mismatch")
    if int(payload.get("seed", -1)) != int(training["seed"]):
        raise AssertionError("compact Transformer-NGM checkpoint seed mismatch")
    if int(payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError("compact Transformer-NGM checkpoint epoch mismatch")
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=1e-7,
        atol=1e-10,
    ):
        raise AssertionError(
            "compact Transformer-NGM checkpoint/frozen validation loss mismatch"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if (
        bias.shape != (9,)
        or sigma.shape != (9,)
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma < 0.05)
    ):
        raise AssertionError("invalid compact Transformer-NGM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
        "architecture": architecture,
    }
    return model, scaler, bias, sigma, manifest


def load_frozen_mlp_ngm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[
    FactorizedMLPNGM9,
    RobustScaler,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    scaler, artifact, frozen = load_scaler_only(source_root, fold, "mlp")
    training = frozen["training"]
    architecture = training["architecture"]
    if architecture.get("name") != "factorized_mlp_ngm_v1_9channel":
        raise AssertionError("unexpected frozen MLP-NGM architecture")
    if int(architecture.get("parameter_count", -1)) != MLP_NGM_9_PARAMETER_COUNT:
        raise AssertionError("unexpected frozen MLP-NGM parameter count")
    model = FactorizedMLPNGM9(dropout=0.10).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture:
        raise AssertionError("MLP-NGM checkpoint architecture mismatch")
    if int(payload.get("seed", -1)) != int(training["seed"]):
        raise AssertionError("MLP-NGM checkpoint seed mismatch")
    if int(payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError("MLP-NGM checkpoint epoch mismatch")
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=1e-7,
        atol=1e-10,
    ):
        raise AssertionError("MLP-NGM checkpoint/frozen validation loss mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if (
        bias.shape != (9,)
        or sigma.shape != (9,)
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma < 0.05)
    ):
        raise AssertionError("invalid MLP-NGM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
        "architecture": architecture,
    }
    return model, scaler, bias, sigma, manifest


def raw_windows_dynamic(
    records: dict[str, Any],
    rows: Any,
    window_samples: int,
) -> np.ndarray:
    lengths = np.asarray(rows.end - rows.start, dtype=np.int64)
    if not np.all(lengths == window_samples):
        raise AssertionError(
            f"expected {window_samples}-sample windows, got {np.unique(lengths).tolist()}"
        )
    values = np.empty((len(rows), window_samples, 9), dtype=np.float32)
    for index, (record_id, start, end) in enumerate(
        zip(rows.record_id, rows.start, rows.end)
    ):
        values[index] = records[str(record_id)].x[int(start) : int(end)]
    return values


def raw_features(
    scaler: RobustScaler,
    raw: np.ndarray,
    window_samples: int = 128,
) -> np.ndarray:
    """RobustScaler then per-window/per-axis centering; return [N,T,9]."""
    bct = centered_scaled_bct(scaler, raw)
    if bct.shape[1:] != (9, window_samples):
        raise AssertionError(f"unexpected RAW tensor shape: {bct.shape}")
    if not np.all(np.isfinite(bct)):
        raise FloatingPointError("RAW tensor contains NaN or infinity after scaling/centering")
    # ``bct`` is intentionally float32 because it is the tensor sent to the
    # classifier.  Summing many potentially large scaled values in float32 can
    # leave a small apparent mean (observed around 1.6e-5) even though the
    # window mean was subtracted correctly.  Recheck with float64 accumulation
    # and use a scale-aware float32 tolerance so this audit catches genuine
    # centering failures without rejecting ordinary round-off.
    axis_means = np.mean(bct, axis=2, dtype=np.float64)
    maximum_axis_mean = float(np.max(np.abs(axis_means)))
    maximum_signal = float(np.max(np.abs(bct)))
    centering_tolerance = max(
        1e-5,
        64.0 * float(np.finfo(np.float32).eps) * max(1.0, maximum_signal),
    )
    if maximum_axis_mean > centering_tolerance:
        raise AssertionError(
            "RAW per-window/per-axis centering failed: "
            f"max_mean={maximum_axis_mean}, tolerance={centering_tolerance}, "
            f"max_signal={maximum_signal}"
        )
    return np.ascontiguousarray(bct.transpose(0, 2, 1), dtype=np.float32)


def build_scheme_c_features(
    error_bct: np.ndarray,
    labels: np.ndarray,
    sigma: np.ndarray,
    window_samples: int,
    expand: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scheme C for arbitrary T, optionally without residual expansion.

    ``expand=True`` returns [r, |r|, delta(r)] with 27 channels.  The strict
    residual-expansion ablation uses ``expand=False`` and returns only r with
    9 channels; every operation before this final representation choice is
    identical.
    """
    error = np.asarray(error_bct, dtype=np.float32)
    if error.ndim != 3 or error.shape[1:] != (9, window_samples):
        raise ValueError(
            f"expected error [N,9,{window_samples}], got {error.shape}"
        )
    labels = np.asarray(labels, dtype=np.int8)
    unclipped = error / (sigma[None, :, None] + 1e-6)
    clip_mask = np.abs(unclipped) > 12.0
    clipped = np.clip(unclipped, -12.0, 12.0).astype(np.float32)
    residual = clipped - clipped.mean(axis=2, keepdims=True)
    maximum_mean = float(
        np.max(np.abs(np.mean(residual, axis=2, dtype=np.float64)))
    )
    tolerance = max(
        1e-5,
        64.0
        * float(np.finfo(np.float32).eps)
        * max(1.0, float(np.max(np.abs(residual)))),
    )
    if maximum_mean > tolerance:
        raise AssertionError(
            f"scheme-C centering failed: max_mean={maximum_mean}, tolerance={tolerance}"
        )
    if expand:
        absolute = np.abs(residual).astype(np.float32, copy=False)
        delta = np.diff(
            residual, axis=2, prepend=residual[:, :, :1]
        ).astype(np.float32, copy=False)
        features = np.concatenate([residual, absolute, delta], axis=1)
        expected_channels = 27
    else:
        features = residual
        expected_channels = 9
    if features.shape[1:] != (expected_channels, window_samples):
        raise AssertionError(f"unexpected scheme-C tensor shape: {features.shape}")

    def clip_record(mask: np.ndarray) -> dict[str, Any]:
        points = int(mask.size)
        count = int(np.count_nonzero(mask))
        return {"points": points, "clipped": count, "rate": count / points if points else None}

    clip_stats = {
        "applicable": True,
        "definition": "abs(e/(sigma+1e-6))>12 before clipping",
        "overall": clip_record(clip_mask),
        "nonfog": clip_record(clip_mask[labels == 0]),
        "fog": clip_record(clip_mask[labels == 1]),
    }
    return np.ascontiguousarray(features.transpose(0, 2, 1)), clip_stats


def make_features(
    method: str,
    scaler: RobustScaler,
    raw: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    nbm_source_root: Path,
    fold: int,
    nbm_kind: str,
    window_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "RAW":
        values = raw_features(scaler, raw, window_samples)
        return values, {
            "formula": "RobustScaler(role4); Xc=X-mean_t(X), input Xc",
            "shape": ["B", 9, window_samples],
            "uses_nbm": False,
            "uses_role5_b_sigma": False,
            "uses_residual": False,
            "removed_nbm_kind": nbm_kind,
        }
    if nbm_kind == "conv_tcn":
        if window_samples != 128:
            raise ValueError("Conv-TCN NBM v1 is fixed to 128 samples")
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_nbm(
            nbm_source_root, fold, device
        )
        error = reconstruction_error(nbm, full_scaler, raw, device)
    elif nbm_kind == "gru":
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_gru_nbm(
            nbm_source_root, fold, device
        )
        scaled = prepare_nbm_windows(full_scaler, raw, center=True)
        reconstruction = reconstruct_gru(nbm, scaled, device)
        error_ntc = (scaled - reconstruction).astype(np.float32, copy=False)
        error = np.ascontiguousarray(error_ntc.transpose(0, 2, 1))
    elif nbm_kind == "gru_v2":
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_gru_v2_nbm(
            nbm_source_root, fold, device
        )
        scaled = prepare_nbm_windows(full_scaler, raw, center=True)
        reconstruction = reconstruct_gru_v2(nbm, scaled, device)
        error_ntc = (scaled - reconstruction).astype(np.float32, copy=False)
        error = np.ascontiguousarray(error_ntc.transpose(0, 2, 1))
    elif nbm_kind == "tcn_v2":
        if window_samples != 128:
            raise ValueError("TCN-v2 NBM is fixed to 128 samples")
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_tcn_v2_nbm(
            nbm_source_root, fold, device
        )
        scaled = centered_scaled_bct(full_scaler, raw)
        reconstruction = reconstruct_tcn_v2(nbm, scaled, device)
        error = (scaled - reconstruction).astype(np.float32, copy=False)
    elif nbm_kind == "tcn_attn_z16":
        if window_samples != 128:
            raise ValueError("TCN-attention-Z16 NBM is fixed to 128 samples")
        nbm, full_scaler, bias, sigma, nbm_manifest = (
            load_frozen_tcn_attention_z16_nbm(
                nbm_source_root, fold, device
            )
        )
        scaled = centered_scaled_bct(full_scaler, raw)
        reconstruction = reconstruct_attention_pool_nbm(nbm, scaled, device)
        error = (scaled - reconstruction).astype(np.float32, copy=False)
    elif nbm_kind == "transformer":
        if window_samples != 128:
            raise ValueError("Transformer-NBM v1 is fixed to 128 samples")
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_transformer_nbm(
            nbm_source_root, fold, device
        )
        scaled = centered_scaled_bct(full_scaler, raw)
        reconstruction = reconstruct_transformer(nbm, scaled, device)
        error = (scaled - reconstruction).astype(np.float32, copy=False)
    elif nbm_kind == "transformer_48k":
        if window_samples != 128:
            raise ValueError("compact Transformer-NGM is fixed to 128 samples")
        nbm, full_scaler, bias, sigma, nbm_manifest = (
            load_frozen_transformer_48k_ngm(nbm_source_root, fold, device)
        )
        scaled = centered_scaled_bct(full_scaler, raw)
        reconstruction = reconstruct_transformer_48k(nbm, scaled, device)
        error = (scaled - reconstruction).astype(np.float32, copy=False)
    elif nbm_kind == "mlp":
        if window_samples != 128:
            raise ValueError("MLP-NGM is fixed to 128 samples")
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_mlp_ngm(
            nbm_source_root, fold, device
        )
        scaled = centered_scaled_bct(full_scaler, raw)
        reconstruction = reconstruct_mlp_bct(nbm, scaled, device)
        error = (scaled - reconstruction).astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported NBM kind: {nbm_kind}")
    if stable_json_hash(full_scaler.as_dict()) != stable_json_hash(scaler.as_dict()):
        raise AssertionError("NBM and classifier role-4 scalers differ")
    expand = method == "FULL_C"
    if method not in ("FULL_C", "RESIDUAL_R"):
        raise ValueError(f"unsupported residual method: {method}")
    values, clip_stats = build_scheme_c_features(
        error, labels, sigma, window_samples, expand=expand
    )
    del nbm, error
    if device.type == "cuda":
        torch.cuda.empty_cache()
    formula = "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); r=q-mean_t(q)"
    if expand:
        formula += "; F=[r,abs(r),delta_t(r)]"
    else:
        formula += "; F=r (no residual expansion)"
    return values, {
        "formula": formula,
        "shape": ["B", 27 if expand else 9, window_samples],
        "residual_expansion": (
            "[r,abs(r),delta_t(r)]" if expand else "none; r only"
        ),
        "uses_nbm": True,
        "uses_role5_b_sigma": True,
        "uses_bias_b": False,
        "uses_sigma": True,
        "nbm_kind": nbm_kind,
        "nbm": nbm_manifest,
        "clip_statistics": clip_stats,
    }


def paired_initialization(seed: int, method: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_state, full_state, hashes = paired_tcn_initial_states(seed)
    pair_id = hashlib.sha256(
        f"{seed}:{hashes['r']}:{hashes['r_abs_delta']}".encode("utf-8")
    ).hexdigest()
    return (full_state if method == "FULL_C" else raw_state), {
        "seed": seed,
        "pair_id": pair_id,
        "raw_9ch_state_sha256": hashes["r"],
        "full_c_27ch_state_sha256": hashes["r_abs_delta"],
        "pairing_rule": (
            "all shape-compatible weights are identical; the 27-channel first layer "
            "copies the first 9 channels and initializes the extra 18 channels to zero"
        ),
    }


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    directory = job_dir(args.output_root.resolve(), args.fold, args.method, args.tcn_seed)
    done_path = directory / "DONE_TRAIN.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP frozen training job: {done_path}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    records, rows = load_records_rows(args.data_dir, args.fold)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, artifact_manifest, frozen_nbm = load_scaler_only(
        args.nbm_source_root, args.fold, args.nbm_kind
    )
    nbm_contract = validate_nbm_contract(frozen_nbm, args)
    train_x, train_feature = make_features(
        args.method, scaler,
        raw_windows_dynamic(records, role67, args.window_samples),
        role67.label, device, args.nbm_source_root, args.fold, args.nbm_kind,
        args.window_samples,
    )
    validation_x, validation_feature = make_features(
        args.method, scaler,
        raw_windows_dynamic(records, role23, args.window_samples),
        role23.label, device, args.nbm_source_root, args.fold, args.nbm_kind,
        args.window_samples,
    )
    initial_state, initialization = paired_initialization(args.tcn_seed, args.method)
    representation = "r_abs_delta" if args.method == "FULL_C" else "r"
    model, training = train_representation_tcn(
        representation,
        train_x,
        role67.label,
        validation_x,
        role23.label,
        directory,
        device,
        args.tcn_seed,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
        initial_state,
        reset_seed_after_loading=True,
    )
    val_true, val_prob = classifier_predict(model, validation_x, role23.label, device)
    threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
    checkpoint = directory / "checkpoints" / "tcn.pt"
    frozen = {
        "job_id": job_id(args.fold, args.method, args.tcn_seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "method": args.method,
        "nbm_kind": args.nbm_kind,
        "nbm_seed": args.nbm_seed,
        "tcn_seed": args.tcn_seed,
        "input_shape": train_feature["shape"],
        "feature": train_feature,
        "validation_feature": validation_feature,
        "role4_scaler_artifact": artifact_manifest,
        "nbm_contract": nbm_contract,
        "role5_policy": (
            "RAW reads only the role-4 scaler fields; it does not load NBM weights, "
            "b, sigma, or use role-5 windows"
            if args.method == "RAW"
            else "Residual methods use frozen role-5 sigma; b is used to estimate "
            "sigma but is not subtracted in scheme C"
        ),
        "roles": {"classifier_train": [6, 7], "classifier_validation": [2, 3], "test_not_accessed": [0, 1]},
        "test_roles_accessed": False,
        "initialization": initialization,
        "training": {key: value for key, value in training.items() if key != "history"},
        "threshold": float(threshold),
        "threshold_source_roles": [2, 3],
        "threshold_rule": "max balanced accuracy; ties FoG F1 then higher threshold; 0.05..0.95 step 0.01",
        "validation": validation_metrics,
        "feature_diagnostics": {
            "roles_6_7_train": residual_diagnostics(train_x, role67.label),
            "roles_2_3_validation": residual_diagnostics(validation_x, role23.label),
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    write_json(directory / "frozen_validation.json", frozen)
    write_json(done_path, {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": frozen["job_id"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "threshold": threshold,
        "test_roles_accessed": False,
    })
    print(
        f"TRAIN FROZEN {frozen['job_id']} best_epoch={training['best_epoch']} threshold={threshold:.2f}",
        flush=True,
    )


def run_seal(args: argparse.Namespace) -> None:
    nbm_seeds = parse_csv_ints(args.nbm_seeds)
    seeds = parse_csv_ints(args.tcn_seeds)
    required_seeds = parse_csv_ints(args.required_seeds)
    methods = parse_csv_methods(args.experiment_methods)
    if nbm_seeds != seeds:
        raise ValueError("strict paired repeats require identical NBM and TCN seed lists")
    if seeds != required_seeds:
        raise ValueError(f"this experiment requires seeds {required_seeds}")
    root = args.output_root.resolve()
    entries = []
    for fold, method, seed in expected_jobs(seeds, methods):
        directory = job_dir(root, fold, method, seed)
        frozen_path = directory / "frozen_validation.json"
        checkpoint = directory / "checkpoints" / "tcn.pt"
        if not (directory / "DONE_TRAIN.json").exists() or not frozen_path.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"training job incomplete: {job_id(fold, method, seed)}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen["test_roles_accessed"] is not False:
            raise AssertionError(f"premature test access: {frozen['job_id']}")
        if sha256_file(checkpoint) != frozen["checkpoint_sha256"]:
            raise AssertionError(f"classifier checkpoint changed: {frozen['job_id']}")
        entries.append({
            "job_id": frozen["job_id"], "fold": fold, "method": method, "tcn_seed": seed,
            "nbm_seed": frozen["nbm_seed"],
            "nbm_kind": frozen["nbm_kind"],
            "threshold": frozen["threshold"], "checkpoint_sha256": frozen["checkpoint_sha256"],
            "pair_id": frozen["initialization"]["pair_id"],
            "scaler_sha256": frozen["role4_scaler_artifact"]["scaler_sha256"],
            "nbm_checkpoint_sha256": frozen["role4_scaler_artifact"]["nbm_checkpoint_sha256"],
            "pos_weight": frozen["training"]["pos_weight"],
            "tcn_max_epochs": frozen["training"]["maximum_epochs"],
            "tcn_patience": frozen["training"]["patience"],
        })
    for fold in FOLDS:
        same_fold = [item for item in entries if item["fold"] == fold]
        for key in ("scaler_sha256", "pos_weight"):
            if len({item[key] for item in same_fold}) != 1:
                raise AssertionError(f"fold {fold} mismatch: {key}")
        for seed in seeds:
            paired = [item for item in same_fold if item["tcn_seed"] == seed]
            if (
                len(paired) != len(methods)
                or len({item["method"] for item in paired}) != len(methods)
                or len({item["pair_id"] for item in paired}) != 1
            ):
                raise AssertionError(f"fold {fold}, seed {seed} is not a valid paired initialization")
            if len({item["nbm_seed"] for item in paired}) != 1 or paired[0]["nbm_seed"] != seed:
                raise AssertionError(f"fold {fold}, seed {seed} NBM/TCN seeds are not paired")
            if len({item["nbm_checkpoint_sha256"] for item in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} methods do not share one NBM")
            if len({item["nbm_kind"] for item in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} NBM backbone mismatch")
    if len({item["nbm_kind"] for item in entries}) != 1:
        raise AssertionError("one experiment cannot mix NBM backbone kinds")
    if any(item["tcn_max_epochs"] != args.tcn_max_epochs for item in entries):
        raise AssertionError("TCN maximum epoch mismatch")
    if any(item["tcn_patience"] != args.tcn_patience for item in entries):
        raise AssertionError("TCN patience mismatch")
    rows_by_fold = {fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS}
    source_audit = audit_protocol_dynamic(
        args.data_dir.resolve(),
        rows_by_fold,
        args.sampling_rate_hz,
        args.window_samples,
        args.stride_samples,
    )
    test_data_manifest = build_test_data_manifest(
        args.data_dir.resolve(), rows_by_fold
    )
    barrier = {
        "barrier_schema": "strict_test_barrier.v2",
        "status": f"all_{'_and_'.join(methods)}_classifiers_and_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(FOLDS), "methods": list(methods),
        "nbm_seeds": list(nbm_seeds), "tcn_seeds": list(seeds),
        "job_count": len(entries),
        "strict_test_gate": "roles 0/1 may be accessed only after this global barrier",
        "source_audit": source_audit,
        "test_data_manifest": test_data_manifest,
        "jobs": entries,
    }
    barrier["barrier_id"] = stable_json_hash(barrier_identity_payload(barrier))
    write_json(root / "TRAINING_BARRIER.json", barrier)
    method_inputs = {
        "FULL_C": (
            "role4 scaler + window-axis centering + NBM + scheme C "
            f"[r,abs(r),delta] [B,27,{args.window_samples}]"
        ),
        "RESIDUAL_R": (
            "role4 scaler + window-axis centering + NBM + scheme C "
            f"r only [B,9,{args.window_samples}]; no abs/delta expansion"
        ),
        "RAW": (
            "role4 scaler + window-axis centering + RAW "
            f"[B,9,{args.window_samples}]"
        ),
    }
    experiment_config = {
        "experiment": (
            f"strict_paired_{entries[0]['nbm_kind']}_NBM_"
            f"{'_vs_'.join(methods)}_TCN"
        ),
        "methods": {method: method_inputs[method] for method in methods},
        "nbm": "max_epoch=300, patience=20, SmoothL1, lr=1e-3, augmentation=40% clean/40% Gaussian(std=.04)/20% mask",
        "paired_seeds": list(seeds),
        "nbm_kind": entries[0]["nbm_kind"],
        "sampling_rate_hz": args.sampling_rate_hz,
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "seed_policy": "exact seeds; no hidden fold offset",
        "tcn": f"max_epoch={args.tcn_max_epochs}, patience={args.tcn_patience}, paired seed/loader order",
        "roles": {str(key): value for key, value in ROLES.items()},
        "threshold": "roles 2/3 balanced accuracy; ties FoG F1 then higher threshold",
        "global_test_barrier_jobs": len(entries),
        "barrier_schema": barrier["barrier_schema"],
        "barrier_id": barrier["barrier_id"],
        "test_data_manifest_sha256": test_data_manifest["sha256"],
    }
    if "FULL_C" in methods:
        experiment_config["full"] = method_inputs["FULL_C"]
    if "RAW" in methods:
        experiment_config["ablation"] = method_inputs["RAW"]
    if "RESIDUAL_R" in methods:
        experiment_config["residual_expansion_ablation"] = method_inputs[
            "RESIDUAL_R"
        ]
    write_json(root / "experiment_config.json", experiment_config)
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def sealed_job(args: argparse.Namespace) -> dict[str, Any]:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.exists():
        raise FileNotFoundError("TRAINING_BARRIER.json missing; roles 0/1 access forbidden")
    barrier = load_and_validate_barrier(barrier_path)
    require_strict_barrier_for_tcn_v2(barrier, args.nbm_kind)
    target = job_id(args.fold, args.method, args.tcn_seed)
    matches = [item for item in barrier["jobs"] if item["job_id"] == target]
    if len(matches) != 1:
        raise AssertionError(f"job not sealed: {target}")
    if matches[0]["nbm_kind"] != args.nbm_kind:
        raise AssertionError("requested NBM backbone differs from the sealed experiment")
    sealed = dict(matches[0])
    sealed["barrier_schema"] = barrier.get("barrier_schema", "legacy.v1")
    sealed["barrier_id"] = barrier.get("barrier_id", sha256_file(barrier_path))
    sealed["test_data_manifest_sha256"] = (
        barrier.get("test_data_manifest", {}).get("sha256")
    )
    return sealed


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key in ("train_weighted_bce", "validation_weighted_bce", "validation_pr_auc"):
            row[key] = float(row[key])
    return rows


def validate_completed_test_artifacts(
    directory: Path,
    sealed: dict[str, Any],
) -> dict[str, Any]:
    """Return a completed result only when it belongs to the current seal."""
    done_path = directory / "DONE_TEST.json"
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "test_predictions.csv"
    probabilities_path = directory / "test_probabilities.npz"
    required = (done_path, metrics_path, predictions_path, probabilities_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete sealed test artifacts: {missing}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or done.get("job_id") != sealed["job_id"]:
        raise AssertionError("DONE_TEST does not identify the sealed job")
    if result.get("job_id") != sealed["job_id"]:
        raise AssertionError("metrics do not identify the sealed job")
    if sealed["barrier_schema"] == "strict_test_barrier.v2":
        expected_fields = {
            "barrier_id": sealed["barrier_id"],
            "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
            "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
            "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
            "scaler_sha256": sealed["scaler_sha256"],
        }
        for optional_key in (
            "nbm_frozen_sha256",
            "done_nbm_sha256",
            "frozen_validation_sha256",
            "scientific_data_sha256",
            "feature_contract_sha256",
        ):
            if optional_key in sealed:
                expected_fields[optional_key] = sealed[optional_key]
        for key, expected in expected_fields.items():
            if result.get(key) != expected or done.get(key) != expected:
                raise AssertionError(
                    f"completed test artifact does not match current seal: {key}"
                )
        if float(result.get("threshold", np.nan)) != float(sealed["threshold"]):
            raise AssertionError("completed test threshold does not match current seal")
        artifact_hashes = {
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
            "probabilities_sha256": sha256_file(probabilities_path),
        }
        for key, actual in artifact_hashes.items():
            if done.get(key) != actual:
                raise AssertionError(f"completed test file hash mismatch: {key}")
    return result


def run_evaluate(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    sealed = sealed_job(args)
    directory = job_dir(args.output_root.resolve(), args.fold, args.method, args.tcn_seed)
    done_path = directory / "DONE_TEST.json"
    current_test_manifest = None
    if sealed["barrier_schema"] == "strict_test_barrier.v2":
        rows_by_fold = {
            fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
        }
        current_test_manifest = build_test_data_manifest(
            args.data_dir.resolve(), rows_by_fold
        )
        if (
            current_test_manifest["sha256"]
            != sealed["test_data_manifest_sha256"]
        ):
            raise AssertionError("permanent-test data changed after the global seal")
    if done_path.exists() and not args.overwrite:
        validate_completed_test_artifacts(directory, sealed)
        print(f"SKIP completed test job: {done_path}", flush=True)
        return
    checkpoint = directory / "checkpoints" / "tcn.pt"
    if sha256_file(checkpoint) != sealed["checkpoint_sha256"]:
        raise AssertionError("sealed TCN checkpoint changed")
    frozen = json.loads((directory / "frozen_validation.json").read_text(encoding="utf-8"))
    if float(frozen["threshold"]) != float(sealed["threshold"]):
        raise AssertionError("sealed threshold changed")

    # The permanent test roles are first requested only after the global barrier.
    records, rows = load_records_rows(args.data_dir, args.fold)
    test_rows = rows.take_role(0, 1)
    scaler, artifact_manifest, _ = load_scaler_only(
        args.nbm_source_root, args.fold, args.nbm_kind
    )
    if artifact_manifest["scaler_sha256"] != sealed["scaler_sha256"]:
        raise AssertionError("sealed role-4 scaler changed")
    if (
        artifact_manifest["nbm_checkpoint_sha256"]
        != sealed["nbm_checkpoint_sha256"]
    ):
        raise AssertionError("sealed NBM checkpoint changed")
    test_x, test_feature = make_features(
        args.method, scaler,
        raw_windows_dynamic(records, test_rows, args.window_samples),
        test_rows.label, device, args.nbm_source_root, args.fold, args.nbm_kind,
        args.window_samples,
    )
    input_channels = 27 if args.method == "FULL_C" else 9
    model = RepresentationTCNM(input_channels).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_true, test_prob = classifier_predict(model, test_x, test_rows.label, device)
    threshold = float(sealed["threshold"])
    metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(np.int8)
    by_subject = {}
    for subject in SUBJECTS:
        mask = test_rows.subject_id == subject
        by_subject[subject] = binary_metrics(test_true[mask], test_prob[mask], threshold)
    result = {
        "job_id": sealed["job_id"], "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold, "method": args.method, "tcn_seed": args.tcn_seed,
        "nbm_seed": args.nbm_seed,
        "nbm_kind": args.nbm_kind,
        "threshold": threshold, "threshold_source_roles": [2, 3],
        "strict_global_test_barrier_verified": True, "test_roles": [0, 1],
        "barrier_schema": sealed["barrier_schema"],
        "barrier_id": sealed["barrier_id"],
        "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
        "test": metrics, "test_by_subject": by_subject,
        "test_feature": test_feature,
        "test_feature_diagnostics": residual_diagnostics(test_x, test_true),
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
    }
    write_json(directory / "metrics.json", result)
    write_csv(directory / "test_predictions.csv", [{
        "fold": args.fold, "method": args.method, "tcn_seed": args.tcn_seed,
        "subject_id": str(test_rows.subject_id[i]), "record_id": str(test_rows.record_id[i]),
        "window_id": str(test_rows.window_id[i]), "start_index": int(test_rows.start[i]),
        "end_index_exclusive": int(test_rows.end[i]), "role_code": int(test_rows.role[i]),
        "y_true": int(test_true[i]), "fog_probability": float(test_prob[i]),
        "threshold": threshold, "y_pred": int(test_pred[i]),
    } for i in range(len(test_rows))])
    np.savez_compressed(
        directory / "test_probabilities.npz", y_true=test_true, y_prob=test_prob,
        y_pred=test_pred, subject_id=test_rows.subject_id, window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    history = load_history(directory / "logs" / "tcn_history.csv")
    plot_classifier_training(
        directory, args.method,
        {**frozen["training"], "history": history}, metrics["confusion_matrix"],
    )
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "test_predictions.csv"
    probabilities_path = directory / "test_probabilities.npz"
    write_json(done_path, {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": sealed["job_id"], "test": metrics,
        "barrier_id": sealed["barrier_id"],
        "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "probabilities_sha256": sha256_file(probabilities_path),
    })
    print(
        f"TEST COMPLETE {sealed['job_id']} acc={metrics['accuracy']:.6f} "
        f"recall={metrics['sensitivity']:.6f} spec={metrics['specificity']:.6f} "
        f"pr_auc={metrics['auprc']:.6f}", flush=True,
    )


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "n": int(len(array))}


def run_aggregate(args: argparse.Namespace) -> None:
    nbm_seeds = parse_csv_ints(args.nbm_seeds)
    seeds = parse_csv_ints(args.tcn_seeds)
    required_seeds = parse_csv_ints(args.required_seeds)
    methods = parse_csv_methods(args.experiment_methods)
    if nbm_seeds != seeds:
        raise ValueError("strict paired repeats require identical NBM and TCN seed lists")
    if seeds != required_seeds:
        raise ValueError(f"this experiment requires seeds {required_seeds}")
    root = args.output_root.resolve()
    barrier_path = root / "TRAINING_BARRIER.json"
    if not barrier_path.exists():
        raise FileNotFoundError("cannot aggregate without global barrier")
    barrier = load_and_validate_barrier(barrier_path)
    if tuple(barrier.get("methods", ())) != methods:
        raise AssertionError(
            "requested experiment methods differ from the sealed method order"
        )
    require_strict_barrier_for_tcn_v2(barrier, args.nbm_kind)
    barrier_schema = barrier.get("barrier_schema", "legacy.v1")
    barrier_id = barrier.get("barrier_id", sha256_file(barrier_path))
    test_data_manifest_sha256 = barrier.get("test_data_manifest", {}).get(
        "sha256"
    )
    if barrier_schema == "strict_test_barrier.v2":
        rows_by_fold = {
            fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
        }
        current_test_manifest = build_test_data_manifest(
            args.data_dir.resolve(), rows_by_fold
        )
        if current_test_manifest["sha256"] != test_data_manifest_sha256:
            raise AssertionError("permanent-test data changed after the global seal")
    barrier_jobs = {item["job_id"]: item for item in barrier["jobs"]}
    results = []
    for fold, method, seed in expected_jobs(seeds, methods):
        directory = job_dir(root, fold, method, seed)
        target = job_id(fold, method, seed)
        if target not in barrier_jobs:
            raise AssertionError(f"job absent from global barrier: {target}")
        if not (directory / "DONE_TEST.json").exists():
            raise FileNotFoundError(f"test incomplete: {job_id(fold, method, seed)}")
        sealed = {
            **barrier_jobs[target],
            "barrier_schema": barrier_schema,
            "barrier_id": barrier_id,
            "test_data_manifest_sha256": test_data_manifest_sha256,
        }
        result = validate_completed_test_artifacts(directory, sealed)
        if (
            int(result["fold"]) != fold
            or result["method"] != method
            or int(result["tcn_seed"]) != seed
            or result["nbm_kind"] != args.nbm_kind
        ):
            raise AssertionError(f"test result identity mismatch: {target}")
        results.append(result)
    run_rows = [{
        "fold": r["fold"], "method": r["method"], "nbm_kind": r["nbm_kind"],
        "nbm_seed": r["nbm_seed"], "tcn_seed": r["tcn_seed"],
        "threshold": r["threshold"],
        **{key: r["test"][key] for key in METRIC_KEYS},
        **{key: r["test"][key] for key in ("tn", "fp", "fn", "tp")},
    } for r in results]
    write_csv(root / f"run_metrics_{len(results)}.csv", run_rows)
    seed_rows = []
    for method in methods:
        for seed in seeds:
            subset = [r for r in results if r["method"] == method and r["tcn_seed"] == seed]
            seed_rows.append({
                "method": method, "tcn_seed": seed,
                **{key: float(np.mean([r["test"][key] for r in subset])) for key in METRIC_KEYS},
            })
    write_csv(root / "seed_macro_over_3folds.csv", seed_rows)
    summary: dict[str, Any] = {}
    summary_rows = []
    for method in methods:
        method_rows = [row for row in seed_rows if row["method"] == method]
        summary[method] = {key: mean_std(row[key] for row in method_rows) for key in METRIC_KEYS}
        for key in METRIC_KEYS:
            summary_rows.append({"method": method, "metric": key, **summary[method][key]})
    write_csv(root / f"method_summary_{len(seeds)}seed_mean_std.csv", summary_rows)
    paired_comparisons: dict[str, Any] = {}
    if len(methods) == 2:
        reference, ablation = methods
        comparison_name = f"{reference}_minus_{ablation}"
        deltas = []
        for seed in seeds:
            reference_row = next(
                row
                for row in seed_rows
                if row["method"] == reference and row["tcn_seed"] == seed
            )
            ablation_row = next(
                row
                for row in seed_rows
                if row["method"] == ablation and row["tcn_seed"] == seed
            )
            deltas.append(
                {
                    "tcn_seed": seed,
                    **{
                        key: reference_row[key] - ablation_row[key]
                        for key in METRIC_KEYS
                    },
                }
            )
        delta_summary = {
            key: mean_std(row[key] for row in deltas) for key in METRIC_KEYS
        }
        write_csv(root / f"paired_delta_{comparison_name}_by_seed.csv", deltas)
        write_csv(
            root / f"paired_delta_{comparison_name}_summary.csv",
            [{"metric": key, **value} for key, value in delta_summary.items()],
        )
        paired_comparisons[comparison_name] = delta_summary
    subject_rows = []
    subject_json: dict[str, Any] = {method: {} for method in methods}
    for method in methods:
        for subject in SUBJECTS:
            per_seed = []
            for seed in seeds:
                subset = [r for r in results if r["method"] == method and r["tcn_seed"] == seed]
                per_seed.append({
                    key: float(np.mean([r["test_by_subject"][subject][key] for r in subset]))
                    for key in METRIC_KEYS
                })
            stats = {key: mean_std(item[key] for item in per_seed) for key in METRIC_KEYS}
            subject_json[method][subject] = stats
            subject_rows.append({
                "method": method, "subject_id": subject,
                **{f"{key}_mean": stats[key]["mean"] for key in METRIC_KEYS},
                **{f"{key}_std": stats[key]["std"] for key in METRIC_KEYS},
            })
    write_csv(root / f"subject_metrics_{len(seeds)}seed_mean_std.csv", subject_rows)
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metrics": summary,
        "paired_comparisons": paired_comparisons,
        "subject_metrics": subject_json,
        "definition": (
            f"within each seed macro-average 3 folds; mean±population SD across "
            f"{len(seeds)} seeds"
        ),
        "strict_global_test_barrier": True,
        "barrier_schema": barrier_schema,
        "barrier_id": barrier_id,
        "test_data_manifest_sha256": test_data_manifest_sha256,
        "nbm_kind": results[0]["nbm_kind"],
        "run_count": len(results),
        "methods": list(methods),
    }
    for comparison_name, comparison_summary in paired_comparisons.items():
        # Keep the historical top-level paired-delta key while also exposing
        # all comparisons through the generic mapping above.
        final[f"paired_delta_{comparison_name}"] = comparison_summary
    write_json(root / "summary.json", final)
    write_json(root / "DONE.json", {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(results), "methods": list(methods),
        "nbm_seeds": list(seeds), "tcn_seeds": list(seeds),
    })
    print(json.dumps(final["primary_metrics"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    if args.stage == "seal":
        run_seal(args)
        return
    if args.stage == "aggregate":
        run_aggregate(args)
        return
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    if args.stage == "train":
        run_train(args, device)
    else:
        run_evaluate(args, device)


if __name__ == "__main__":
    main()
