#!/usr/bin/env python
"""Run the preregistered Transformer-NBM fusion F0--F5 LOSO suite.

The suite freezes the canonical Transformer normal-behaviour checkpoint in
each fold, replays it once to obtain robust-scaled raw targets, signed forecast
errors, and conditional standard deviations, then trains six TCN-M diagnostic
readouts on an identical four-second HistoryPlan:

F0 raw, F1 error, F2 raw+error, F3 raw+zero, F4 raw+z+log(sigma), and
F5 raw+Gaussian-NLL.  S04 and S10 remain excluded by the canonical source
protocol.  One worker owns a complete held-out-subject fold so the script can
be scheduled safely across independent GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_3imu_nbm_suite as core
import run_daphnet_nbm_representation_ablation as nbm_rep
import run_daphnet_persistence_input_ablation as input_ablation
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.data import DaphnetDataset, RobustChannelScaler, WindowTable
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.fusion_representations import (
    FUSION_REPRESENTATION_NAMES,
    FUSION_REPRESENTATION_REGISTRY,
    build_fusion_representation,
)
from cnbr_fog.histories import (
    HistoryPlan,
    make_block_history_input,
    make_common_history_plan,
)
from cnbr_fog.nbm import NormalBehaviourModel
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    done_payload,
    sha256_file,
    validate_checkpoint,
    validate_done,
)


SUITE_VERSION = "daphnet_transformer_fusion_f0_f5_h4_tcnm_loso.v1"
SOURCE_SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
SOURCE_NBM = "transformer"
EXPECTED_CHANNEL_NAMES = rf.EXPECTED_CHANNEL_NAMES
EXPECTED_LOSO_SUBJECTS = rf.EXPECTED_LOSO_SUBJECTS
CLASSIFICATION_METRICS = tuple(rf.CLASSIFICATION_METRICS)

CONTEXT_SAMPLES = 128
HORIZON_SAMPLES = 32
STRIDE_SAMPLES = 16
HISTORY_SECONDS = 4.0
HISTORY_SAMPLES = 256
HISTORY_BLOCKS = 8
TCN_M_DILATIONS = (1, 2, 4, 8, 8, 8)
TCN_M_RF_SAMPLES = 125
Z_CLIP = 12.0
SOURCE_REPLAY_TOLERANCE_AMP = 2e-2
SOURCE_REPLAY_TOLERANCE_NO_AMP = 2e-3

FUSION_IDS = tuple(FUSION_REPRESENTATION_NAMES)
FUSION_REPRESENTATIONS: dict[str, dict[str, Any]] = {
    name: {
        "display_name": f"{name} {spec.display_name}",
        "in_channels": int(spec.output_channels),
        "shape": spec.shape,
        "formula": spec.formula,
    }
    for name, spec in FUSION_REPRESENTATION_REGISTRY.items()
}
COMPARISONS: tuple[dict[str, str], ...] = (
    {
        "comparison_id": "F1_minus_F0",
        "new": "F1",
        "reference": "F0",
        "interpretation": "Transformer centering versus matched raw input",
    },
    {
        "comparison_id": "F2_minus_F0",
        "new": "F2",
        "reference": "F0",
        "interpretation": "primary raw-plus-error fusion comparison",
    },
    {
        "comparison_id": "F2_minus_F3",
        "new": "F2",
        "reference": "F3",
        "interpretation": "error information beyond 18-channel capacity",
    },
    {
        "comparison_id": "F2_minus_F1",
        "new": "F2",
        "reference": "F1",
        "interpretation": "raw support added to signed error",
    },
    {
        "comparison_id": "F4_minus_F2",
        "new": "F4",
        "reference": "F2",
        "interpretation": "explicit conditional uncertainty representation",
    },
    {
        "comparison_id": "F5_minus_F2",
        "new": "F5",
        "reference": "F2",
        "interpretation": "Gaussian surprise map versus signed error",
    },
    {
        "comparison_id": "F5_minus_F3",
        "new": "F5",
        "reference": "F3",
        "interpretation": (
            "Gaussian surprise information beyond 18-channel capacity"
        ),
    },
)

IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_fusion_f0_f5_suite.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_daphnet_nbm_representation_ablation.py",
    "scripts/run_daphnet_persistence_input_ablation.py",
    "scripts/run_daphnet_3imu_nbm_suite.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/fusion_representations.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/nbm.py",
    "cnbr_fog/resume.py",
)

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed"
)
DEFAULT_SOURCE_SUITE_DIR = (
    REPO_ROOT / "outputs" / "daphnet_3imu_nbm_5x4_loso_seed42"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daphnet_transformer_fusion_f0_f5_h4_tcnm_loso_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daphnet Transformer-NBM F0--F5 fusion, residual_h4s, "
            "TCN-M, 8-fold LOSO"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        default=DEFAULT_SOURCE_SUITE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help="Run exactly one complete held-out-subject fold",
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Replay and validate Transformer primitives without classifiers",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow reduced non-reportable epochs/windows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-classifier-windows", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
    )
    parser.add_argument("--stop-after-completed-tasks", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    if int(args.seed) != 42:
        raise ValueError("The formal protocol fixes --seed 42")
    positive = {
        "classifier_hidden": args.classifier_hidden,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [name for name, value in positive.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {invalid}")
    if args.max_classifier_windows < 0 or args.num_workers < 0:
        raise ValueError("Window cap and num-workers must be non-negative")
    if args.stop_after_completed_tasks < 0:
        raise ValueError("--stop-after-completed-tasks must be non-negative")
    if not math.isfinite(args.classifier_lr) or args.classifier_lr <= 0:
        raise ValueError("--classifier-lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0,1)")
    if not args.smoke:
        formal = {
            "classifier_hidden": (int(args.classifier_hidden), 48),
            "classifier_dropout": (float(args.classifier_dropout), 0.15),
            "classifier_epochs": (int(args.classifier_epochs), 12),
            "classifier_patience": (int(args.classifier_patience), 4),
            "classifier_lr": (float(args.classifier_lr), 1e-3),
            "weight_decay": (float(args.weight_decay), 1e-4),
            "batch_size": (int(args.batch_size), 256),
            "max_classifier_windows": (
                int(args.max_classifier_windows),
                0,
            ),
            "bootstrap_samples": (
                int(args.bootstrap_samples),
                100000,
            ),
            "bootstrap_seed": (int(args.bootstrap_seed), 42),
            "deterministic": (bool(args.deterministic), True),
            "amp": (bool(args.amp), True),
        }
        changed = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in formal.items()
            if actual != expected
        }
        if changed:
            raise ValueError(
                "Formal protocol options changed; add --smoke for a "
                f"non-reportable run: {changed}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def build_source_manifest(
    source_suite_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full, source_config = nbm_rep.build_source_manifest(source_suite_dir)
    folds: dict[str, Any] = {}
    for subject in EXPECTED_LOSO_SUBJECTS:
        source_fold = full["folds"][subject]
        folds[subject] = {
            key: value
            for key, value in source_fold.items()
            if key != "models"
        }
        folds[subject]["models"] = {
            SOURCE_NBM: source_fold["models"][SOURCE_NBM]
        }
    selected = {
        key: value
        for key, value in full.items()
        if key != "folds"
    }
    selected.update(
        {
            "selected_nbm": SOURCE_NBM,
            "folds": folds,
        }
    )
    return selected, source_config


def experiment_id(fusion_id: str) -> str:
    return f"transformer_h4s_tcnm__{fusion_id}"


def _aligned_reference_states(
    seed: int,
    hidden_channels: int,
    dropout: float,
    deterministic: bool,
) -> tuple[
    dict[int, dict[str, torch.Tensor]],
    dict[int, int],
    dict[int, str],
    str,
]:
    """Build shape-specific states with an exactly shared TCN-M backbone."""
    max_channels = max(
        int(spec["in_channels"])
        for spec in FUSION_REPRESENTATIONS.values()
    )
    rf.set_seed(seed, deterministic)
    largest = rf.build_model(
        in_channels=max_channels,
        hidden_channels=hidden_channels,
        dropout=dropout,
        dilations=TCN_M_DILATIONS,
    )
    largest_state = {
        name: value.detach().cpu().clone()
        for name, value in largest.state_dict().items()
    }
    del largest
    states: dict[int, dict[str, torch.Tensor]] = {}
    counts: dict[int, int] = {}
    hashes: dict[int, str] = {}
    for in_channels in sorted(
        {
            int(spec["in_channels"])
            for spec in FUSION_REPRESENTATIONS.values()
        }
    ):
        model = rf.build_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            dilations=TCN_M_DILATIONS,
        )
        state: dict[str, torch.Tensor] = {}
        for name, target in model.state_dict().items():
            source = largest_state[name]
            if name == "projection.0.weight":
                if source.ndim != 3 or source.shape[1] < in_channels:
                    raise AssertionError("Unexpected TCN input projection")
                source = source[:, :in_channels, :]
            if source.shape != target.shape:
                raise AssertionError(
                    f"Unshareable TCN state {name}: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            state[name] = source.clone()
        model.load_state_dict(state, strict=True)
        states[in_channels] = state
        counts[in_channels] = rf.parameter_count(model)
        hashes[in_channels] = rf.state_dict_sha256(model.state_dict())
        del model
    shared_backbone = {
        name: value
        for name, value in largest_state.items()
        if name != "projection.0.weight"
    }
    shared_backbone_hash = rf.state_dict_sha256(shared_backbone)
    return states, counts, hashes, shared_backbone_hash


def experiment_grid(
    args: argparse.Namespace,
    sampling_rate_hz: int,
) -> tuple[
    list[dict[str, Any]],
    dict[int, int],
    dict[int, str],
    str,
]:
    if rf.convolutional_receptive_field(TCN_M_DILATIONS) != TCN_M_RF_SAMPLES:
        raise AssertionError("TCN-M receptive field changed")
    _, counts, hashes, backbone_hash = _aligned_reference_states(
        args.seed,
        int(args.classifier_hidden),
        float(args.classifier_dropout),
        bool(args.deterministic),
    )
    cells: list[dict[str, Any]] = []
    for fusion_id, definition in FUSION_REPRESENTATIONS.items():
        channels = int(definition["in_channels"])
        cells.append(
            {
                "variant": fusion_id,
                "experiment_id": experiment_id(fusion_id),
                "fusion_id": fusion_id,
                **definition,
                "dilations": list(TCN_M_DILATIONS),
                "kernel_size": 3,
                "n_blocks": len(TCN_M_DILATIONS),
                "convolutions_per_block": 2,
                "receptive_field_samples": TCN_M_RF_SAMPLES,
                "receptive_field_seconds": (
                    TCN_M_RF_SAMPLES / float(sampling_rate_hz)
                ),
                "parameter_count": counts[channels],
                "template_initial_state_sha256": hashes[channels],
            }
        )
    return cells, counts, hashes, backbone_hash


def build_protocol(
    args: argparse.Namespace,
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    required = {
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "seed": 42,
        "robust_clip": 12.0,
        "residual_clip": Z_CLIP,
    }
    for key, expected in required.items():
        if source_config.get(key) != expected:
            raise ValueError(
                f"Canonical source {key}={source_config.get(key)!r}; "
                f"expected {expected!r}"
            )
    if tuple(source_config.get("excluded_subjects", [])) != ("S04", "S10"):
        raise ValueError("The formal protocol must exclude S04 and S10")
    cells, counts, hashes, backbone_hash = experiment_grid(
        args,
        dataset.sampling_rate_hz,
    )
    scientific = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "channel_names": list(dataset.channel_names),
        "n_channels": int(dataset.n_channels),
        "subjects": list(dataset.subjects),
        "excluded_subjects": ["S04", "S10"],
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbm": SOURCE_NBM,
        "nbm_policy": "frozen canonical fold-specific Transformer checkpoint",
        "source_model_reconstruction": {
            key: source_config[key]
            for key in (
                "sampling_rate_hz",
                "nbm_hidden",
                "nbm_dropout",
                "linear_ar_seconds",
                "gru_layers",
                "transformer_heads",
                "transformer_layers",
                "transformer_ffn",
            )
        },
        "context_seconds": 2.0,
        "context_samples": CONTEXT_SAMPLES,
        "horizon_seconds": 0.5,
        "horizon_samples": HORIZON_SAMPLES,
        "predictor_stride_seconds": 0.25,
        "stride_samples": STRIDE_SAMPLES,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "history_block_spacing_samples": HORIZON_SAMPLES,
        "z_clip": Z_CLIP,
        "gaussian_nll": {
            "formula": "log(sigma) + 0.5 * ((x - mu) / sigma) ** 2",
            "uses_unclipped_z": True,
            "omits_constant_half_log_2pi": True,
        },
        "fusion_representations": cells,
        "expected_experiments": len(cells),
        "expected_primitive_cache_tasks": len(EXPECTED_LOSO_SUBJECTS),
        "expected_classifier_cells": (
            len(EXPECTED_LOSO_SUBJECTS) * len(cells)
        ),
        "comparisons": list(COMPARISONS),
        "primary_comparison": "F2_minus_F0",
        "classifier": {
            "name": "TCN-M",
            "hidden_channels": int(args.classifier_hidden),
            "dropout": float(args.classifier_dropout),
            "kernel_size": rf.KERNEL_SIZE,
            "convolutions_per_block": rf.CONVS_PER_BLOCK,
            "dilations": list(TCN_M_DILATIONS),
            "receptive_field_samples": TCN_M_RF_SAMPLES,
            "receptive_field_seconds": (
                TCN_M_RF_SAMPLES / float(dataset.sampling_rate_hz)
            ),
            "parameter_count_by_in_channels": {
                str(key): value for key, value in counts.items()
            },
            "template_initial_state_sha256_by_in_channels": {
                str(key): value for key, value in hashes.items()
            },
            "shared_backbone_initial_state_sha256": backbone_hash,
            "fold_initialization_seed_rule": "42 + 10000 + fold_index",
            "global_pooling": "mean_and_max_over_full_4s_input",
        },
        "classifier_epochs": int(args.classifier_epochs),
        "classifier_patience": int(args.classifier_patience),
        "classifier_lr": float(args.classifier_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_classifier_windows": int(args.max_classifier_windows),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "run_kind": "smoke" if args.smoke else "formal",
        "reportable": not bool(args.smoke),
        "fairness_contract": {
            "same_fold_scaler_and_transformer_checkpoint": True,
            "same_anchor_history_and_labels": True,
            "same_tcn_m_backbone_initialization": True,
            "same_full_initialization_within_equal_channel_groups": True,
            "equal_channel_groups": [
                ["F0", "F1"],
                ["F2", "F3", "F5"],
                ["F4"],
            ],
            "F3_controls_18_channel_capacity": True,
            "F3_does_not_control_27_channel_F4_capacity": True,
            "same_epoch_shuffle_rule": "classifier_seed + epoch",
            "validation_only_early_stopping_and_threshold": True,
            "test_subject_never_selects_model_or_threshold": True,
        },
        "window_count": int(len(windows)),
        "window_class_counts": np.bincount(
            windows.label,
            minlength=2,
        ).astype(int).tolist(),
    }
    fingerprint = canonical_fingerprint(scientific)
    return {
        **scientific,
        "protocol_fingerprint": fingerprint,
        "data_dir": str(args.data_dir),
        "source_suite_dir": str(args.source_suite_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "num_workers": int(args.num_workers),
        "resume": bool(args.resume),
        "smoke": bool(args.smoke),
    }


def _primitive_cache_keys() -> set[str]:
    return {
        f"{split}_{key}"
        for split in ("train", "validation", "test")
        for key in ("raw", "error", "sigma", "y", "window_index")
    }


@torch.no_grad()
def _extract_primitives(
    args: argparse.Namespace,
    model: NormalBehaviourModel,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    labels: np.ndarray,
    canonical_dynamic: np.ndarray,
    scaler: RobustChannelScaler,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    loader = core.make_sequence_loader(
        dataset,
        windows,
        np.asarray(indices, dtype=np.int64),
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    model = model.to(device)
    model.eval()
    raw_chunks: list[np.ndarray] = []
    error_chunks: list[np.ndarray] = []
    sigma_chunks: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    observed_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        target = sequence[:, :, CONTEXT_SAMPLES:].float()
        with torch.amp.autocast(
            device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            mean, sigma = model(context)
        raw_chunks.append(target.cpu().numpy().astype(np.float32))
        error_chunks.append(
            (target - mean.float()).cpu().numpy().astype(np.float32)
        )
        sigma_chunks.append(sigma.float().cpu().numpy().astype(np.float32))
        observed_labels.append(y.numpy())
        observed_indices.append(index.numpy())
    primitives = {
        "raw": np.ascontiguousarray(np.concatenate(raw_chunks)),
        "error": np.ascontiguousarray(np.concatenate(error_chunks)),
        "sigma": np.ascontiguousarray(np.concatenate(sigma_chunks)),
    }
    seen_y = np.concatenate(observed_labels).astype(np.int8, copy=False)
    seen_index = np.concatenate(observed_indices).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(seen_y, labels):
        raise ValueError("Transformer replay changed labels")
    if not np.array_equal(seen_index, indices):
        raise ValueError("Transformer replay changed window order")
    sigma = primitives["sigma"]
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Transformer sigma must be finite and positive")
    replay = np.clip(
        primitives["error"] / sigma,
        -Z_CLIP,
        Z_CLIP,
    ).astype(np.float32)
    canonical = np.asarray(canonical_dynamic, dtype=np.float32)
    tolerance = (
        SOURCE_REPLAY_TOLERANCE_AMP
        if args.amp and device.type == "cuda"
        else SOURCE_REPLAY_TOLERANCE_NO_AMP
    )
    max_abs_diff = float(
        np.max(np.abs(replay.astype(np.float64) - canonical))
    )
    if not np.allclose(
        replay,
        canonical,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise AssertionError(
            "Replayed Transformer z differs from canonical source cache: "
            f"max_abs_diff={max_abs_diff}, tolerance={tolerance}"
        )
    diagnostics = {
        key: input_ablation._array_diagnostics(value)
        for key, value in primitives.items()
    }
    for fusion_id in FUSION_IDS:
        derived = build_fusion_representation(
            fusion_id,
            primitives["raw"],
            primitives["error"],
            primitives["sigma"],
        )
        diagnostics[fusion_id] = input_ablation._array_diagnostics(
            derived
        )
        del derived
    diagnostics["canonical_dynamic_replay_max_abs_diff"] = max_abs_diff
    diagnostics["canonical_dynamic_replay_tolerance"] = tolerance
    diagnostics["unclipped_z_gt_12_fraction"] = float(
        (np.abs(primitives["error"] / sigma) > Z_CLIP).mean()
    )
    return primitives, diagnostics


def load_or_create_primitive_cache(
    args: argparse.Namespace,
    config: dict[str, Any],
    subject: str,
    fold_root: Path,
    dataset: DaphnetDataset,
    windows: WindowTable,
    source_features: Mapping[str, Mapping[str, np.ndarray]],
    source_provenance: Mapping[str, Any],
    scaler: RobustChannelScaler,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    str,
]:
    cache_path = fold_root / "fusion_primitives.npz"
    diagnostics_path = fold_root / "fusion_diagnostics.json"
    done_path = fold_root / "FUSION_PRIMITIVE_CACHE_DONE.json"
    task_id = f"{subject}/fusion_primitive_cache"
    upstream = canonical_fingerprint(
        {
            **source_provenance,
            "source_scaler_sha256": config["source"]["folds"][subject][
                "source_scaler_sha256"
            ],
            "source_split_indices_sha256": config["source"]["folds"][
                subject
            ]["source_split_indices_sha256"],
            "source_history_support_sha256": config["source"]["folds"][
                subject
            ]["source_history_support_sha256"],
            "primitive_formula_version": "raw_error_sigma.v1",
        }
    )
    completed = validate_done(
        done_path,
        stage="fusion_primitive_cache",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        upstream_sha256=upstream,
    )
    if completed is not None:
        with np.load(cache_path, allow_pickle=False) as payload:
            if set(payload.files) != _primitive_cache_keys():
                raise ValueError(
                    f"Unexpected fusion primitive arrays: {subject}"
                )
            features = {
                split: {
                    "raw": np.asarray(
                        payload[f"{split}_raw"],
                        dtype=np.float32,
                    ),
                    "error": np.asarray(
                        payload[f"{split}_error"],
                        dtype=np.float32,
                    ),
                    "sigma": np.asarray(
                        payload[f"{split}_sigma"],
                        dtype=np.float32,
                    ),
                    "y": np.asarray(
                        payload[f"{split}_y"],
                        dtype=np.int8,
                    ),
                    "window_index": np.asarray(
                        payload[f"{split}_window_index"],
                        dtype=np.int64,
                    ),
                }
                for split in ("train", "validation", "test")
            }
        return features, _load_json(diagnostics_path), sha256_file(cache_path)

    checkpoint_path = (
        args.source_suite_dir
        / f"loso_{subject}"
        / SOURCE_NBM
        / "nbm"
        / "best.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint(
        checkpoint,
        stage="nbm",
        protocol_fingerprint=config["source"][
            "source_protocol_fingerprint"
        ],
        task_id=f"loso_{subject}/{SOURCE_NBM}/nbm",
    )
    model = nbm_rep._build_source_model(
        SOURCE_NBM,
        checkpoint,
        config["source_model_reconstruction"],
    )
    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source = source_features[split]
        primitives, split_diagnostics = _extract_primitives(
            args,
            model,
            dataset,
            windows,
            np.asarray(source["window_index"], dtype=np.int64),
            np.asarray(source["y"], dtype=np.int8),
            np.asarray(
                source["dynamic_standardized_error"],
                dtype=np.float32,
            ),
            scaler,
            device,
        )
        features[split] = {
            **primitives,
            "y": np.asarray(source["y"], dtype=np.int8),
            "window_index": np.asarray(
                source["window_index"],
                dtype=np.int64,
            ),
        }
        diagnostics[split] = split_diagnostics
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    atomic_npz_save(
        cache_path,
        **{
            f"{split}_{key}": features[split][key]
            for split in ("train", "validation", "test")
            for key in ("raw", "error", "sigma", "y", "window_index")
        },
    )
    atomic_json_dump(diagnostics, diagnostics_path)
    atomic_json_dump(
        done_payload(
            stage="fusion_primitive_cache",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
            upstream_sha256=upstream,
            relative_to=fold_root,
            artifacts={
                "cache": cache_path,
                "diagnostics": diagnostics_path,
            },
        ),
        done_path,
    )
    return features, diagnostics, sha256_file(cache_path)


def prepare_fold(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
    device: torch.device,
) -> tuple[
    Path,
    dict[str, dict[str, np.ndarray]],
    dict[str, HistoryPlan],
    dict[str, Any],
    dict[int, dict[str, torch.Tensor]],
]:
    fold_root = args.output_dir / f"loso_{subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    source_fold_root = args.source_suite_dir / f"loso_{subject}"
    source_fold_config = _load_json(source_fold_root / "fold_config.json")
    if source_fold_config.get("test_subject") != subject:
        raise ValueError(f"Source fold subject mismatch: {subject}")
    scaler = nbm_rep._load_scaler(source_fold_root / "scaler.json")
    source_features, source_provenance = nbm_rep._load_source_cache(
        args,
        config,
        subject,
        SOURCE_NBM,
    )
    features, diagnostics, cache_sha = load_or_create_primitive_cache(
        args,
        config,
        subject,
        fold_root,
        dataset,
        windows,
        source_features,
        source_provenance,
        scaler,
        device,
    )
    plans: dict[str, HistoryPlan] = {}
    support_arrays: dict[str, np.ndarray] = {}
    source_support_path = source_fold_root / "history_support.npz"
    with np.load(source_support_path, allow_pickle=False) as source_support:
        for split in ("train", "validation", "test"):
            indices = np.asarray(
                features[split]["window_index"],
                dtype=np.int64,
            )
            plan = make_common_history_plan(
                windows,
                indices,
                HORIZON_SAMPLES,
                STRIDE_SAMPLES,
                HISTORY_SAMPLES,
            )
            if len(plan.anchor_rows) == 0:
                raise RuntimeError(f"Empty history support: {subject}/{split}")
            if not np.array_equal(
                plan.anchor_window_indices,
                source_support[f"{split}_anchor_window_index"],
            ):
                raise ValueError(
                    f"Anchor support differs from source: {subject}/{split}"
                )
            chain_indices = indices[plan.max_chain_rows]
            if not np.array_equal(
                chain_indices,
                source_support[f"{split}_history_window_index"],
            ):
                raise ValueError(
                    f"History support differs from source: {subject}/{split}"
                )
            cap_this_split = (
                args.max_classifier_windows > 0
                and (split == "train" or args.smoke)
            )
            if cap_this_split:
                rows = np.arange(len(plan.anchor_rows), dtype=np.int64)
                anchor_labels = windows.label[
                    plan.anchor_window_indices
                ]
                split_offset = {
                    "train": 0,
                    "validation": 1,
                    "test": 2,
                }[split]
                selected = rf.deterministic_subsample(
                    rows,
                    args.max_classifier_windows,
                    args.seed
                    + 100
                    + EXPECTED_LOSO_SUBJECTS.index(subject)
                    + split_offset,
                    anchor_labels,
                )
                plan = plan.take(selected)
                chain_indices = indices[plan.max_chain_rows]
            plans[split] = plan
            support_arrays[f"{split}_anchor_window_index"] = (
                plan.anchor_window_indices
            )
            support_arrays[f"{split}_history_window_index"] = chain_indices
            support_arrays[f"{split}_y"] = np.asarray(
                features[split]["y"][plan.anchor_rows],
                dtype=np.int8,
            )
    support_path = fold_root / "input_support.npz"
    core.save_or_validate_npz(support_path, **support_arrays)
    support_sha = sha256_file(support_path)
    input_fingerprints = {
        fusion_id: canonical_fingerprint(
            {
                "primitive_cache_sha256": cache_sha,
                "input_support_sha256": support_sha,
                "fusion": definition,
                "history_samples": HISTORY_SAMPLES,
                "history_blocks": HISTORY_BLOCKS,
                "z_clip": Z_CLIP,
            }
        )
        for fusion_id, definition in FUSION_REPRESENTATIONS.items()
    }
    core.save_or_validate_json(
        fold_root / "fusion_input_fingerprints.json",
        {
            "protocol_fingerprint": config["protocol_fingerprint"],
            "fusion_inputs": input_fingerprints,
        },
    )
    fold_index = EXPECTED_LOSO_SUBJECTS.index(subject)
    classifier_seed = int(args.seed) + 10000 + fold_index
    reference_states, counts, hashes, backbone_hash = (
        _aligned_reference_states(
            classifier_seed,
            int(args.classifier_hidden),
            float(args.classifier_dropout),
            bool(args.deterministic),
        )
    )
    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "val_subject": source_fold_config["val_subject"],
        "train_subjects": source_fold_config["train_subjects"],
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256_by_in_channels": {
            str(key): value for key, value in hashes.items()
        },
        "parameter_count_by_in_channels": {
            str(key): value for key, value in counts.items()
        },
        "shared_backbone_initial_state_sha256": backbone_hash,
        "source": {
            **source_provenance,
            "source_fold_config_sha256": sha256_file(
                source_fold_root / "fold_config.json"
            ),
            "source_scaler_sha256": sha256_file(
                source_fold_root / "scaler.json"
            ),
            "source_split_indices_sha256": sha256_file(
                source_fold_root / "split_indices.npz"
            ),
            "source_history_support_sha256": sha256_file(
                source_support_path
            ),
        },
        "primitive_cache_sha256": cache_sha,
        "input_support_sha256": support_sha,
        "fusion_input_fingerprints": input_fingerprints,
        "history_anchor_counts": {
            split: int(len(plan.anchor_rows))
            for split, plan in plans.items()
        },
        "shared_support_by_all_fusions": True,
        "diagnostics_present": bool(diagnostics),
    }
    core.save_or_validate_json(fold_root / "fold_config.json", fold_config)
    core.save_or_validate_json(
        fold_root / "source_provenance.json",
        fold_config["source"],
    )
    return (
        fold_root,
        features,
        plans,
        fold_config,
        reference_states,
    )


def materialize_fusion_inputs(
    features: Mapping[str, Mapping[str, np.ndarray]],
    plans: Mapping[str, HistoryPlan],
    fusion_id: str,
) -> dict[str, dict[str, np.ndarray]]:
    expected_channels = int(
        FUSION_REPRESENTATIONS[fusion_id]["in_channels"]
    )
    inputs: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        source = features[split]
        block = build_fusion_representation(
            fusion_id,
            np.asarray(source["raw"], dtype=np.float32),
            np.asarray(source["error"], dtype=np.float32),
            np.asarray(source["sigma"], dtype=np.float32),
        )
        extracted = {
            fusion_id: block,
            "y": np.asarray(source["y"], dtype=np.int8),
            "window_index": np.asarray(
                source["window_index"],
                dtype=np.int64,
            ),
        }
        payload = make_block_history_input(
            extracted=extracted,
            plan=plans[split],
            source_key=fusion_id,
            name=fusion_id,
            history_samples=HISTORY_SAMPLES,
            horizon_samples=HORIZON_SAMPLES,
            stride_samples=STRIDE_SAMPLES,
        )
        array = np.asarray(payload[fusion_id], dtype=np.float32)
        if array.shape[1:] != (expected_channels, HISTORY_SAMPLES):
            raise ValueError(
                f"Unexpected fusion history: "
                f"{fusion_id}/{split}/{array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Non-finite fusion history: {fusion_id}/{split}"
            )
        inputs[split] = payload
    return inputs


def task_root_for(
    output_dir: Path,
    subject: str,
    fusion_id: str,
) -> Path:
    return output_dir / f"loso_{subject}" / fusion_id


def train_cell(
    args: argparse.Namespace,
    config: dict[str, Any],
    cell: dict[str, Any],
    task_root: Path,
    fold_config: dict[str, Any],
    reference_states: Mapping[int, Mapping[str, torch.Tensor]],
    inputs: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    device: torch.device,
) -> dict[str, Any]:
    fusion_id = str(cell["fusion_id"])
    in_channels = int(cell["in_channels"])
    input_binding = fold_config["fusion_input_fingerprints"][fusion_id]
    classifier_fold_config = {
        **fold_config,
        "reference_initial_state_sha256": fold_config[
            "reference_initial_state_sha256_by_in_channels"
        ][str(in_channels)],
        "_reference_initial_state": reference_states[in_channels],
        "source": {
            "source_residual_cache_sha256": input_binding,
            "input_support_sha256": fold_config["input_support_sha256"],
        },
    }
    original_input = rf.INPUT_NAME
    original_nbm = rf.SOURCE_NBM
    rf.INPUT_NAME = fusion_id
    rf.SOURCE_NBM = SOURCE_NBM
    try:
        metrics = rf.train_classifier_resumable(
            args,
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "shared_parameter_count": cell["parameter_count"],
            },
            cell,
            task_root,
            classifier_fold_config,
            inputs,
            dataset,
            windows,
            device,
        )
    finally:
        rf.INPUT_NAME = original_input
        rf.SOURCE_NBM = original_nbm
    expected = {
        "experiment_id": cell["experiment_id"],
        "variant": fusion_id,
        "input": fusion_id,
        "nbm": SOURCE_NBM,
        "test_subject": fold_config["test_subject"],
        "source_residual_sha256": input_binding,
        "input_support_sha256": fold_config["input_support_sha256"],
        "initial_state_sha256": fold_config[
            "reference_initial_state_sha256_by_in_channels"
        ][str(in_channels)],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"Classifier identity mismatch: {fusion_id}/{key}"
            )
    return metrics


def _load_completed_cell(
    output_dir: Path,
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    subject: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    fusion_id = str(cell["fusion_id"])
    root = task_root_for(output_dir, subject, fusion_id)
    done = validate_done(
        root / "DONE.json",
        stage="rf_classifier",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/{fusion_id}",
    )
    if done is None:
        return None
    expected_artifacts = {
        "best",
        "last",
        "metrics",
        "predictions",
        "validation_predictions",
        "predictions_csv",
    }
    if set(done.get("artifacts", {})) != expected_artifacts:
        raise ValueError(f"Classifier DONE artifact mismatch: {root}")
    fold_root = output_dir / f"loso_{subject}"
    fold_config = _load_json(fold_root / "fold_config.json")
    cache_sha = sha256_file(fold_root / "fusion_primitives.npz")
    support_sha = sha256_file(fold_root / "input_support.npz")
    input_binding = fold_config["fusion_input_fingerprints"][fusion_id]
    in_channels = int(cell["in_channels"])
    initial_hash = fold_config[
        "reference_initial_state_sha256_by_in_channels"
    ][str(in_channels)]
    if cache_sha != fold_config["primitive_cache_sha256"]:
        raise ValueError(f"Primitive cache changed: {subject}")
    if support_sha != fold_config["input_support_sha256"]:
        raise ValueError(f"Input support changed: {subject}")
    expected_done = {
        "source_residual_sha256": input_binding,
        "input_support_sha256": support_sha,
        "initial_state_sha256": initial_hash,
    }
    for key, value in expected_done.items():
        if done.get(key) != value:
            raise ValueError(f"Classifier DONE mismatch: {root}/{key}")
    metrics = _load_json(root / "metrics.json")
    expected_metrics = {
        "experiment_id": cell["experiment_id"],
        "variant": fusion_id,
        "test_subject": subject,
        "val_subject": fold_config["val_subject"],
        "classifier_seed": fold_config["classifier_seed"],
        "nbm": SOURCE_NBM,
        "input": fusion_id,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "source_residual_sha256": input_binding,
        "input_support_sha256": support_sha,
        "initial_state_sha256": initial_hash,
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise ValueError(f"Metrics identity mismatch: {root}/{key}")
    classifier_config = metrics.get("classifier_config", {})
    expected_classifier_config = {
        "in_channels": in_channels,
        "hidden_channels": config["classifier"]["hidden_channels"],
        "dropout": config["classifier"]["dropout"],
        "kernel_size": rf.KERNEL_SIZE,
        "dilations": list(TCN_M_DILATIONS),
        "n_blocks": len(TCN_M_DILATIONS),
        "convolutions_per_block": rf.CONVS_PER_BLOCK,
        "receptive_field_samples": TCN_M_RF_SAMPLES,
        "receptive_field_seconds": cell["receptive_field_seconds"],
        "parameter_count": cell["parameter_count"],
        "initial_state_sha256": initial_hash,
        "global_pooling": "mean_and_max_over_full_input",
    }
    if classifier_config != expected_classifier_config:
        raise ValueError(f"Classifier configuration mismatch: {root}")
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        expected_keys = {"window_index", "y_true", "y_prob", "y_pred"}
        if set(payload.files) != expected_keys:
            raise ValueError(f"Prediction array set mismatch: {root}")
        arrays = {
            key: np.asarray(payload[key])
            for key in expected_keys
        }
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError(f"Prediction lengths differ: {root}")
    if not np.isfinite(arrays["y_prob"]).all():
        raise ValueError(f"Non-finite prediction probabilities: {root}")
    with np.load(
        fold_root / "input_support.npz",
        allow_pickle=False,
    ) as support:
        if not np.array_equal(
            arrays["window_index"],
            support["test_anchor_window_index"],
        ):
            raise ValueError(f"Prediction support changed: {root}")
        if not np.array_equal(arrays["y_true"], support["test_y"]):
            raise ValueError(f"Prediction labels changed: {root}")
    return metrics, arrays


def _format_mean_sd(
    summary: Mapping[str, Any],
    metric: str,
) -> str:
    payload = summary.get(metric, {})
    if not isinstance(payload, Mapping):
        return ""
    mean, std = payload.get("mean"), payload.get("std")
    if mean is None or std is None:
        return ""
    precision = 3 if metric in {
        "false_alarm_events_per_hour",
        "median_detection_delay_sec",
    } else 4
    return f"{float(mean):.{precision}f} +/- {float(std):.{precision}f}"


def refresh_summaries(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    cells = list(config["fusion_representations"])
    rows_by_fusion: dict[str, dict[str, dict[str, Any]]] = {
        str(cell["fusion_id"]): {} for cell in cells
    }
    fold_rows: list[dict[str, Any]] = []
    aggregate_payload: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    publication_rows: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []

    for cell in cells:
        fusion_id = str(cell["fusion_id"])
        group: list[dict[str, Any]] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed_subjects: list[str] = []
        for subject in EXPECTED_LOSO_SUBJECTS:
            completed = _load_completed_cell(
                output_dir,
                config,
                cell,
                subject,
            )
            if completed is None:
                continue
            metrics, arrays = completed
            enriched = {
                **metrics,
                "fusion_id": fusion_id,
                "formula": cell["formula"],
                "in_channels": cell["in_channels"],
                "parameter_count": cell["parameter_count"],
            }
            group.append(enriched)
            fold_rows.append(enriched)
            rows_by_fusion[fusion_id][subject] = enriched
            truths.append(np.asarray(arrays["y_true"], dtype=np.int8))
            probabilities.append(
                np.asarray(arrays["y_prob"], dtype=np.float64)
            )
            predictions.append(
                np.asarray(arrays["y_pred"], dtype=np.int8)
            )
            completed_subjects.append(subject)
        macro = (
            aggregate_fold_metrics(
                group,
                list(CLASSIFICATION_METRICS),
            )
            if group
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in CLASSIFICATION_METRICS
            }
        )
        pooled = (
            rf.prediction_metrics(
                np.concatenate(truths),
                np.concatenate(probabilities),
                np.concatenate(predictions),
            )
            if truths
            else None
        )
        aggregate_payload[cell["experiment_id"]] = {
            **cell,
            "completed_folds": completed_subjects,
            "subject_macro": macro,
            "pooled": pooled,
        }
        experiment_rows.append(
            {
                "experiment_id": cell["experiment_id"],
                "fusion_id": fusion_id,
                "display_name": cell["display_name"],
                "formula": cell["formula"],
                "in_channels": cell["in_channels"],
                "parameter_count": cell["parameter_count"],
                "expected_folds": len(EXPECTED_LOSO_SUBJECTS),
                "completed_folds": len(completed_subjects),
                "status": (
                    "complete"
                    if completed_subjects == list(EXPECTED_LOSO_SUBJECTS)
                    else ("partial" if completed_subjects else "pending")
                ),
                "completed_subjects": ",".join(completed_subjects),
            }
        )
        row: dict[str, Any] = {
            "experiment_id": cell["experiment_id"],
            "fusion_id": fusion_id,
            "display_name": cell["display_name"],
            "in_channels": cell["in_channels"],
            "parameter_count": cell["parameter_count"],
            "completed_folds": len(completed_subjects),
        }
        for metric in CLASSIFICATION_METRICS:
            row[f"{metric}_mean"] = macro[metric]["mean"]
            row[f"{metric}_std"] = macro[metric]["std"]
        aggregate_rows.append(row)
        publication_rows.append(
            {
                "Fusion input": cell["display_name"],
                "Channels": cell["in_channels"],
                "PR-AUC": _format_mean_sd(macro, "pr_auc"),
                "BA": _format_mean_sd(macro, "balanced_accuracy"),
                "Macro-F1": _format_mean_sd(macro, "macro_f1"),
                "AUROC": _format_mean_sd(macro, "roc_auc"),
                "FoG Recall": _format_mean_sd(macro, "fog_recall"),
                "Specificity": _format_mean_sd(macro, "specificity"),
                "FoG Precision": _format_mean_sd(macro, "precision"),
                "FoG F1": _format_mean_sd(macro, "fog_f1"),
                "Event Sensitivity": _format_mean_sd(
                    macro,
                    "event_sensitivity",
                ),
                "FA/h": _format_mean_sd(
                    macro,
                    "false_alarm_events_per_hour",
                ),
                "Delay (s)": _format_mean_sd(
                    macro,
                    "median_detection_delay_sec",
                ),
                "Completed folds": len(completed_subjects),
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        differences: list[float] = []
        common_subjects: list[str] = []
        for subject in EXPECTED_LOSO_SUBJECTS:
            new = rows_by_fusion[comparison["new"]].get(subject)
            reference = rows_by_fusion[
                comparison["reference"]
            ].get(subject)
            if new is None or reference is None:
                continue
            differences.append(
                float(new["pr_auc"]) - float(reference["pr_auc"])
            )
            common_subjects.append(subject)
        effect = input_ablation.paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            input_ablation.stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                comparison["comparison_id"],
            ),
        )
        comparison_rows.append(
            {
                **comparison,
                "common_subjects": ",".join(common_subjects),
                **effect,
                "bootstrap_seed": int(config["bootstrap_seed"]),
            }
        )

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["pr_auc_mean"])
            if row["pr_auc_mean"] is not None
            else float("inf"),
            row["fusion_id"],
        )
    )
    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(aggregate_rows, start=1)
    ]
    completed_cells = len(fold_rows)
    expected_cells = int(config["expected_classifier_cells"])
    completed_cache_subjects = [
        subject
        for subject in EXPECTED_LOSO_SUBJECTS
        if validate_done(
            output_dir
            / f"loso_{subject}"
            / "FUSION_PRIMITIVE_CACHE_DONE.json",
            stage="fusion_primitive_cache",
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=f"{subject}/fusion_primitive_cache",
        )
        is not None
    ]
    integrity_complete = (
        completed_cells == expected_cells
        and completed_cache_subjects == list(EXPECTED_LOSO_SUBJECTS)
    )
    reportable_complete = integrity_complete and bool(config["reportable"])
    best_experiment = (
        ranked_rows[0]["experiment_id"]
        if reportable_complete and ranked_rows
        else None
    )
    fold_columns = [
        "experiment_id",
        "variant",
        "fusion_id",
        "display_name",
        "formula",
        "in_channels",
        "parameter_count",
        "test_subject",
        "val_subject",
        "classifier_seed",
        "threshold",
        "n",
        "n_normal",
        "n_fog",
        *CLASSIFICATION_METRICS,
        "tn",
        "fp",
        "fn",
        "tp",
        "best_epoch",
        "best_validation_auprc",
        "pos_weight",
        "elapsed_sec",
        "source_residual_sha256",
        "input_support_sha256",
        "initial_state_sha256",
    ]
    aggregate_columns = [
        "rank",
        "experiment_id",
        "fusion_id",
        "display_name",
        "in_channels",
        "parameter_count",
        "completed_folds",
        *[
            field
            for metric in CLASSIFICATION_METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ],
    ]
    _write_csv(
        output_dir / "experiment_manifest.csv",
        experiment_rows,
        list(experiment_rows[0])
        if experiment_rows
        else [
            "experiment_id",
            "fusion_id",
            "display_name",
            "formula",
            "in_channels",
            "parameter_count",
            "expected_folds",
            "completed_folds",
            "status",
            "completed_subjects",
        ],
    )
    _write_csv(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    _write_csv(
        output_dir / "aggregate_summary.csv",
        ranked_rows,
        aggregate_columns,
    )
    _write_csv(
        output_dir / "paired_pr_auc_deltas.csv",
        comparison_rows,
        [
            "comparison_id",
            "new",
            "reference",
            "interpretation",
            "common_subjects",
            "mean_delta",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "wins",
            "ties",
            "losses",
            "bootstrap_samples",
            "bootstrap_seed",
        ],
    )
    _write_csv(
        output_dir / "publication_table.csv",
        publication_rows,
        [
            "Fusion input",
            "Channels",
            "PR-AUC",
            "BA",
            "Macro-F1",
            "AUROC",
            "FoG Recall",
            "Specificity",
            "FoG Precision",
            "FoG F1",
            "Event Sensitivity",
            "FA/h",
            "Delay (s)",
            "Completed folds",
        ],
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "aggregation_unit": "held_out_subject",
            "ranking_metric": "subject_macro_pr_auc_mean",
            "primary_comparison": config["primary_comparison"],
            "best_experiment": best_experiment,
            "experiments": aggregate_payload,
            "paired_pr_auc_comparisons": comparison_rows,
        },
        output_dir / "aggregate_metrics.json",
    )
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_primitive_cache_tasks": len(EXPECTED_LOSO_SUBJECTS),
            "completed_primitive_cache_tasks": len(
                completed_cache_subjects
            ),
            "expected_classifier_cells": expected_cells,
            "completed_classifier_cells": completed_cells,
            "status": (
                "complete"
                if reportable_complete
                else (
                    "smoke_complete"
                    if integrity_complete
                    else "partial"
                )
            ),
            "reportable": bool(config["reportable"]),
            "best_experiment": best_experiment,
        },
        output_dir / "status.json",
    )


def initialize_protocol(
    args: argparse.Namespace,
    device: torch.device,
    worker_mode: bool,
) -> tuple[dict[str, Any], DaphnetDataset, WindowTable]:
    source_manifest, source_config = build_source_manifest(
        args.source_suite_dir
    )
    dataset, windows, data_sha256 = rf.load_dataset_and_windows(
        args.data_dir,
        source_config,
    )
    config = build_protocol(
        args,
        source_manifest,
        source_config,
        dataset,
        windows,
        data_sha256,
        device,
    )
    config_path = args.output_dir / "config.json"
    if worker_mode and not config_path.exists():
        raise RuntimeError(
            "Missing config.json; initialize once with --finalize-only"
        )
    if config_path.exists():
        existing = _load_json(config_path)
        if existing.get("protocol_fingerprint") != config[
            "protocol_fingerprint"
        ]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new "
                "output directory"
            )
    if not worker_mode:
        atomic_json_dump(config, config_path)
    runtime_fields = {
        "data_dir",
        "source_suite_dir",
        "output_dir",
        "device",
        "num_workers",
        "resume",
        "smoke",
    }
    run_manifest = {
        key: value
        for key, value in config.items()
        if key not in runtime_fields
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if _load_json(run_manifest_path) != run_manifest:
            raise ValueError("Saved run manifest is incompatible")
    else:
        core.save_or_validate_json(run_manifest_path, run_manifest)
    return config, dataset, windows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.source_suite_dir = args.source_suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    rf.validate_output_path(
        args.output_dir,
        args.source_suite_dir,
        args.data_dir,
    )
    worker_mode = bool(str(args.worker_fold).strip())
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = rf.resolve_device(args.device)
    rf.set_seed(args.seed, args.deterministic)
    configured_folds = rf.parse_folds(
        args.folds,
        list(EXPECTED_LOSO_SUBJECTS),
    )
    if tuple(configured_folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("The formal suite requires --folds all")
    execution_folds = list(configured_folds)
    if worker_mode:
        execution_folds = rf.parse_folds(
            str(args.worker_fold),
            list(EXPECTED_LOSO_SUBJECTS),
        )
        if len(execution_folds) != 1:
            raise ValueError("--worker-fold must select exactly one subject")

    config, dataset, windows = initialize_protocol(
        args,
        device,
        worker_mode,
    )
    environment = rf.environment_payload(device)
    environment["protocol_fingerprint"] = config["protocol_fingerprint"]
    if worker_mode:
        environment["worker_fold"] = execution_folds[0]
        atomic_json_dump(
            environment,
            args.output_dir
            / "worker_environments"
            / f"loso_{execution_folds[0]}.json",
        )
    else:
        atomic_json_dump(
            environment,
            args.output_dir / "environment.json",
        )
        refresh_summaries(args.output_dir, config)
    print(
        f"[INFO] suite={SUITE_VERSION} device={device} "
        f"folds={execution_folds} nbm={SOURCE_NBM} "
        f"fusions={list(FUSION_IDS)} classifier=TCN-M",
        flush=True,
    )
    if args.finalize_only:
        refresh_summaries(args.output_dir, config)
        print("[INFO] finalize-only: summaries refreshed", flush=True)
        print(
            json.dumps(
                _load_json(args.output_dir / "status.json"),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    completed_this_run = 0
    for subject in execution_folds:
        (
            fold_root,
            features,
            plans,
            fold_config,
            reference_states,
        ) = prepare_fold(
            args,
            config,
            dataset,
            windows,
            subject,
            device,
        )
        print(
            f"[fold {subject}] train={fold_config['train_subjects']} "
            f"val={fold_config['val_subject']} "
            f"anchors={fold_config['history_anchor_counts']}",
            flush=True,
        )
        if args.cache_only:
            print(
                f"[fold {subject}] fusion primitive cache ready",
                flush=True,
            )
            continue
        observed_hashes: dict[int, set[str]] = {}
        for cell in config["fusion_representations"]:
            fusion_id = str(cell["fusion_id"])
            completed = _load_completed_cell(
                args.output_dir,
                config,
                cell,
                subject,
            )
            if completed is not None:
                metrics = completed[0]
                print(
                    f"[fold {subject}] {cell['display_name']} "
                    "validated complete; skip",
                    flush=True,
                )
            else:
                inputs = materialize_fusion_inputs(
                    features,
                    plans,
                    fusion_id,
                )
                metrics = train_cell(
                    args,
                    config,
                    cell,
                    task_root_for(
                        args.output_dir,
                        subject,
                        fusion_id,
                    ),
                    fold_config,
                    reference_states,
                    inputs,
                    dataset,
                    windows,
                    device,
                )
                del inputs
            channels = int(cell["in_channels"])
            observed_hashes.setdefault(channels, set()).add(
                str(metrics["initial_state_sha256"])
            )
            completed_this_run += 1
            print(
                f"[fold {subject}] {cell['display_name']} "
                f"PR-AUC={metrics['pr_auc']:.4f} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"FoG-F1={metrics['fog_f1']:.4f}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                args.stop_after_completed_tasks > 0
                and completed_this_run >= args.stop_after_completed_tasks
            ):
                raise RuntimeError(
                    "Intentional stop after completed classifier tasks"
                )
        if any(len(hashes) != 1 for hashes in observed_hashes.values()):
            raise AssertionError(
                f"Equal-width initial states differ in fold {subject}"
            )
        if not worker_mode:
            refresh_summaries(args.output_dir, config)
        del features, reference_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if worker_mode:
        print(
            json.dumps(
                {
                    "suite_version": SUITE_VERSION,
                    "protocol_fingerprint": config[
                        "protocol_fingerprint"
                    ],
                    "worker_fold": execution_folds[0],
                    "classifier_cells_visited": completed_this_run,
                    "status": "worker_complete",
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    refresh_summaries(args.output_dir, config)
    print(
        json.dumps(
            _load_json(args.output_dir / "status.json"),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
