#!/usr/bin/env python
"""Strict TCN convolutional receptive-field ablation on Persistence residual_h4s.

This suite reuses the immutable Persistence NBM residual caches produced by
``run_daphnet_3imu_nbm_suite.py``.  It never retrains or re-runs the NBM.
For every one of the eight canonical LOSO folds, three six-block TCN
classifiers are trained.  The dilation schedule is the only architectural
difference:

* local  / TCN-S: (1, 1, 1, 1, 1, 2), receptive field 29 samples;
* medium / TCN-M: (1, 2, 4, 8, 8, 8), receptive field 125 samples;
* long   / TCN-L: (1, 2, 4, 8, 16, 32), receptive field 253 samples.

The quoted receptive field is the local convolutional-feature receptive field.
The existing classifier uses global mean/max pooling, so the final readout can
aggregate features across the complete four-second input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# Required by deterministic CUDA matrix multiplication.  It must be set before
# torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset, WindowTable
from cnbr_fog.evaluation import (
    aggregate_fold_metrics,
    binary_metrics,
    choose_threshold,
)
from cnbr_fog.histories import (
    HistoryPlan,
    history_block_count,
    make_common_history_plan,
    make_history_input,
)
from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    capture_rng_state,
    dataset_fingerprint,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_done,
)
from run_cnbr_fog_loso import (
    deterministic_subsample,
    event_metrics,
    parse_folds,
    write_predictions_csv,
)


SUITE_VERSION = "daphnet_persistence_h4_tcn_rf_ablation.v1"
SOURCE_SUITE_VERSION = "daphnet_3imu_nbm_suite.v1"
SOURCE_NBM = "persistence"
INPUT_NAME = "residual_h4s"
HISTORY_SECONDS = 4.0
HISTORY_SAMPLES = 256
HISTORY_BLOCKS = 8
KERNEL_SIZE = 3
CONVS_PER_BLOCK = 2
EXPECTED_CHANNEL_NAMES = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)
EXPECTED_LOSO_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
)
TCN_VARIANTS: dict[str, dict[str, Any]] = {
    "local": {
        "display_name": "TCN-S",
        "dilations": (1, 1, 1, 1, 1, 2),
        "receptive_field_samples": 29,
    },
    "medium": {
        "display_name": "TCN-M",
        "dilations": (1, 2, 4, 8, 8, 8),
        "receptive_field_samples": 125,
    },
    "long": {
        "display_name": "TCN-L",
        "dilations": (1, 2, 4, 8, 16, 32),
        "receptive_field_samples": 253,
    },
}
CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "fog_f1",
    "specificity",
    "precision",
    "mcc",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
]
IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/__init__.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
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
    / "daphnet_persistence_h4_tcn_rf_ablation_seed42"
)


def convolutional_receptive_field(
    dilations: tuple[int, ...] | list[int],
    kernel_size: int = KERNEL_SIZE,
    convolutions_per_block: int = CONVS_PER_BLOCK,
) -> int:
    """Return the local theoretical receptive field of stacked same-pad convs."""

    if kernel_size <= 0 or convolutions_per_block <= 0:
        raise ValueError("kernel_size and convolutions_per_block must be positive")
    if not dilations or any(int(value) <= 0 for value in dilations):
        raise ValueError("dilations must be a non-empty sequence of positive values")
    return 1 + int(convolutions_per_block) * (int(kernel_size) - 1) * sum(
        int(value) for value in dilations
    )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes independent of torch.save."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def implementation_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {"sha256": canonical_fingerprint(files), "files": files}


def environment_payload(device: torch.device) -> dict[str, Any]:
    cuda_devices: list[str] = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "selected_device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": [sys.executable, *sys.argv],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict Persistence residual_h4s TCN local receptive-field ablation"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-suite-dir", type=Path, default=DEFAULT_SOURCE_SUITE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help="Run exactly one fold; used by the multi-GPU scheduler.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Initialize/validate the protocol and rebuild root summaries only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-classifier-windows",
        type=int,
        default=0,
        help="Training-only deterministic cap; zero uses all common anchors.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
        help="Testing hook for exact epoch-boundary resume.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.finalize_only and str(args.worker_fold).strip():
        raise ValueError("--finalize-only and --worker-fold cannot be combined")
    positive_integer = {
        "classifier_hidden": args.classifier_hidden,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "batch_size": args.batch_size,
    }
    invalid = [key for key, value in positive_integer.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These options must be positive integers: {invalid}")
    if args.max_classifier_windows < 0 or args.num_workers < 0:
        raise ValueError("Window cap and num-workers must be non-negative")
    if 0 < args.max_classifier_windows < 2:
        raise ValueError("--max-classifier-windows must be zero or at least two")
    if not math.isfinite(args.classifier_lr) or args.classifier_lr <= 0:
        raise ValueError("--classifier-lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.classifier_dropout < 1.0:
        raise ValueError("--classifier-dropout must be in [0, 1)")


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {spec}")
    return device


def paths_overlap(left: Path, right: Path) -> bool:
    """Return True when either resolved path contains the other."""

    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def validate_output_path(
    output_dir: Path,
    source_suite_dir: Path,
    data_dir: Path,
) -> None:
    for label, protected in (
        ("source suite", source_suite_dir),
        ("processed data", data_dir),
    ):
        if paths_overlap(output_dir, protected):
            raise ValueError(
                f"Output directory must be separate from the {label}: "
                f"output={output_dir}, protected={protected}"
            )


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def atomic_csv_write(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_or_validate_json(path: Path, payload: dict) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(f"Saved JSON is incompatible: {path}")
        return
    atomic_json_dump(payload, path)


def save_or_validate_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(arrays):
                raise ValueError(f"Saved array keys differ in {path}")
            for key, expected in arrays.items():
                if not np.array_equal(payload[key], np.asarray(expected)):
                    raise ValueError(f"Saved array mismatch in {path}: {key}")
        return
    atomic_npz_save(path, **arrays)


def _artifact_path(done_path: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact["path"]))
    return path if path.is_absolute() else done_path.parent / path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_source_config(source_config: dict) -> None:
    required = {
        "suite_version": SOURCE_SUITE_VERSION,
        "sampling_rate_hz": 64,
        "n_channels": 9,
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
    }
    for key, value in required.items():
        if source_config.get(key) != value:
            raise ValueError(
                f"Source suite {key}={source_config.get(key)!r}; expected {value!r}"
            )
    if tuple(source_config.get("channel_names", [])) != EXPECTED_CHANNEL_NAMES:
        raise ValueError("Source suite channel order is not the canonical 3-IMU order")
    if tuple(source_config.get("subjects", [])) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("Source suite is not the canonical post-exclusion cohort")
    if tuple(source_config.get("folds_resolved", [])) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError("Source suite does not contain the canonical eight LOSO folds")
    if set(source_config.get("excluded_subjects", [])) != {"S04", "S10"}:
        raise ValueError("Source suite must exclude exactly S04 and S10")
    if SOURCE_NBM not in source_config.get("nbms_resolved", []):
        raise ValueError("Source suite does not contain the Persistence NBM")
    h4 = [
        item
        for item in source_config.get("history_variants", [])
        if item.get("input") == INPUT_NAME
    ]
    if len(h4) != 1:
        raise ValueError("Source suite must contain exactly one residual_h4s variant")
    if (
        int(h4[0].get("history_samples", -1)) != HISTORY_SAMPLES
        or int(h4[0].get("history_blocks", -1)) != HISTORY_BLOCKS
    ):
        raise ValueError("Source residual_h4s is not the expected 256-sample history")


def _read_done_metadata(
    path: Path,
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str | None = None,
) -> dict:
    """Read a DONE manifest and check identity/size without hashing artifacts.

    Parallel workers use this for folds they do not consume, avoiding seven
    processes concurrently re-hashing every residual cache.  The bootstrap and
    finalizer fully hash all folds, and each worker fully validates its own
    source cache before loading it.
    """

    payload = _load_json(path)
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Source DONE {key} mismatch in {path}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    if (
        upstream_sha256 is not None
        and payload.get("upstream_nbm_sha256") != upstream_sha256
    ):
        raise ValueError(f"Source DONE upstream mismatch in {path}")
    for artifact in payload.get("artifacts", {}).values():
        artifact_path = _artifact_path(path, artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_path)
        if artifact_path.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"Source artifact size mismatch: {artifact_path}")
    return payload


def build_source_manifest(
    source_suite_dir: Path,
    *,
    verify_artifacts: bool = True,
) -> tuple[dict, dict]:
    """Validate source DONE chains and describe immutable cache inputs."""

    config_path = source_suite_dir / "config.json"
    run_manifest_path = source_suite_dir / "run_manifest.json"
    if not config_path.exists() or not run_manifest_path.exists():
        raise FileNotFoundError(
            f"Source suite lacks config/run_manifest under {source_suite_dir}"
        )
    source_config = _load_json(config_path)
    validate_source_config(source_config)
    source_protocol = str(source_config["protocol_fingerprint"])
    folds: dict[str, dict[str, Any]] = {}
    for subject in EXPECTED_LOSO_SUBJECTS:
        nbm_done_path = (
            source_suite_dir / f"loso_{subject}" / SOURCE_NBM / "nbm" / "DONE.json"
        )
        validation = validate_done if verify_artifacts else _read_done_metadata
        nbm_done = validation(
            nbm_done_path,
            stage="nbm",
            protocol_fingerprint=source_protocol,
            task_id=f"loso_{subject}/{SOURCE_NBM}/nbm",
        )
        if nbm_done is None:
            raise FileNotFoundError(nbm_done_path)
        residual_done_path = (
            source_suite_dir
            / f"loso_{subject}"
            / SOURCE_NBM
            / "RESIDUAL_CACHE_DONE.json"
        )
        residual_done = validation(
            residual_done_path,
            stage="residual_cache",
            protocol_fingerprint=source_protocol,
            task_id=f"loso_{subject}/{SOURCE_NBM}/residual_cache",
            upstream_sha256=residual_done_upstream(nbm_done),
        )
        if residual_done is None:
            raise FileNotFoundError(residual_done_path)
        cache_entry = residual_done["artifacts"].get("cache")
        if not cache_entry:
            raise ValueError(f"Source residual DONE lacks cache artifact: {subject}")
        cache_path = _artifact_path(residual_done_path, cache_entry)
        source_history_support_path = (
            source_suite_dir / f"loso_{subject}" / "history_support.npz"
        )
        if not source_history_support_path.exists():
            raise FileNotFoundError(source_history_support_path)
        folds[subject] = {
            "source_nbm_best_sha256": residual_done_upstream(nbm_done),
            "source_residual_cache_sha256": str(cache_entry["sha256"]),
            "source_residual_cache_bytes": int(cache_entry["bytes"]),
            "source_residual_done_sha256": sha256_file(residual_done_path),
            "source_fold_config_sha256": sha256_file(
                source_suite_dir / f"loso_{subject}" / "fold_config.json"
            ),
            "source_history_support_sha256": sha256_file(
                source_history_support_path
            ),
            "source_history_support_bytes": int(
                source_history_support_path.stat().st_size
            ),
            # The path is deliberately omitted from the scientific protocol.
            "_cache_path": str(cache_path),
        }
    scientific_folds = {
        subject: {key: value for key, value in payload.items() if not key.startswith("_")}
        for subject, payload in folds.items()
    }
    return (
        {
            "source_suite_version": source_config["suite_version"],
            "source_protocol_fingerprint": source_protocol,
            "source_run_manifest_sha256": sha256_file(run_manifest_path),
            "source_data_sha256": source_config["data_sha256"],
            "source_seed": int(source_config["seed"]),
            "nbm": SOURCE_NBM,
            "input": INPUT_NAME,
            "history_seconds": HISTORY_SECONDS,
            "history_samples": HISTORY_SAMPLES,
            "history_blocks": HISTORY_BLOCKS,
            "folds": scientific_folds,
        },
        source_config,
    )


def residual_done_upstream(nbm_done: Mapping[str, Any]) -> str:
    best = nbm_done.get("artifacts", {}).get("best")
    if not best:
        raise ValueError("Source NBM DONE lacks the best checkpoint artifact")
    return str(best["sha256"])


def load_dataset_and_windows(
    data_dir: Path,
    source_config: dict,
) -> tuple[DaphnetDataset, WindowTable, str]:
    data_sha256 = dataset_fingerprint(data_dir)
    if data_sha256 != source_config["data_sha256"]:
        raise ValueError(
            "Processed Daphnet data hash differs from the source NBM suite"
        )
    dataset = DaphnetDataset.load(
        data_dir,
        flatline_seconds=float(source_config["flatline_seconds"]),
        zero_tolerance=float(source_config["zero_tolerance"]),
    )
    if dataset.n_channels != 9 or tuple(dataset.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError("The receptive-field suite requires canonical 9-channel data")
    excluded = set(source_config["excluded_subjects"])
    dataset = DaphnetDataset(
        root=dataset.root,
        records=[
            record for record in dataset.records if record.subject_id not in excluded
        ],
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
    )
    if tuple(dataset.subjects) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            f"Expected subjects {EXPECTED_LOSO_SUBJECTS}, got {dataset.subjects}"
        )
    windows = dataset.make_windows(
        warmup_samples=int(source_config["context_samples"]),
        target_samples=int(source_config["horizon_samples"]),
        stride_samples=int(source_config["stride_samples"]),
        fog_fraction_threshold=float(source_config["fog_fraction_threshold"]),
        normal_guard_samples=int(source_config["normal_guard_samples"]),
    )
    if len(windows) != int(source_config["window_count"]):
        raise ValueError("Reconstructed WindowTable differs from source suite")
    expected_counts = np.asarray(
        source_config["window_class_counts"], dtype=np.int64
    )
    actual_counts = np.bincount(windows.label, minlength=2)
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError("Reconstructed window labels differ from source suite")
    return dataset, windows, data_sha256


def build_model(
    *,
    in_channels: int,
    hidden_channels: int,
    dropout: float,
    dilations: tuple[int, ...],
) -> ResidualTCNClassifier:
    return ResidualTCNClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dilations=dilations,
        kernel_size=KERNEL_SIZE,
        dropout=dropout,
    )


def variant_protocol(
    args: argparse.Namespace,
    sampling_rate_hz: int,
) -> tuple[list[dict[str, Any]], int]:
    variants: list[dict[str, Any]] = []
    counts: list[int] = []
    hashes: list[str] = []
    for name, definition in TCN_VARIANTS.items():
        dilations = tuple(int(value) for value in definition["dilations"])
        receptive_field = convolutional_receptive_field(dilations)
        if receptive_field != int(definition["receptive_field_samples"]):
            raise AssertionError(f"Incorrect canonical receptive field for {name}")
        set_seed(args.seed, args.deterministic)
        model = build_model(
            in_channels=len(EXPECTED_CHANNEL_NAMES),
            hidden_channels=args.classifier_hidden,
            dropout=args.classifier_dropout,
            dilations=dilations,
        )
        count = parameter_count(model)
        initial_hash = state_dict_sha256(model.state_dict())
        counts.append(count)
        hashes.append(initial_hash)
        variants.append(
            {
                "variant": name,
                "display_name": definition["display_name"],
                "experiment_id": f"persistence_h4s__tcn_{name}",
                "dilations": list(dilations),
                "n_blocks": len(dilations),
                "convolutions_per_block": CONVS_PER_BLOCK,
                "kernel_size": KERNEL_SIZE,
                "receptive_field_samples": receptive_field,
                "receptive_field_seconds": receptive_field
                / float(sampling_rate_hz),
                "parameter_count": count,
                "reference_initial_state_sha256": initial_hash,
            }
        )
        del model
    if len(set(counts)) != 1:
        raise AssertionError(f"Variant parameter counts differ: {counts}")
    if len(set(hashes)) != 1:
        raise AssertionError("Variant initial states differ before training")
    return variants, counts[0]


def build_protocol(
    args: argparse.Namespace,
    source_manifest: dict,
    source_config: dict,
    dataset: DaphnetDataset,
    windows: WindowTable,
    data_sha256: str,
    device: torch.device,
) -> dict:
    variants, shared_parameter_count = variant_protocol(
        args, dataset.sampling_rate_hz
    )
    protocol = {
        "suite_version": SUITE_VERSION,
        "implementation": implementation_manifest(),
        "data_sha256": data_sha256,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channel_names": list(dataset.channel_names),
        "n_channels": dataset.n_channels,
        "excluded_subjects": list(source_config["excluded_subjects"]),
        "subjects": list(dataset.subjects),
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "source": source_manifest,
        "nbm": SOURCE_NBM,
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "context_samples": int(source_config["context_samples"]),
        "horizon_samples": int(source_config["horizon_samples"]),
        "stride_samples": int(source_config["stride_samples"]),
        "window_count": len(windows),
        "variants": variants,
        "shared_parameter_count": shared_parameter_count,
        "classifier_hidden": args.classifier_hidden,
        "classifier_dropout": args.classifier_dropout,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_classifier_windows": args.max_classifier_windows,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "fairness_contract": {
            "ablation_axis": "dilations",
            "receptive_field_scope": "local_convolutional_features",
            "global_pooling_note": (
                "The final mean/max readout aggregates features over all 256 input "
                "samples; 29/125/253 are convolutional feature receptive fields."
            ),
            "shared_fields": [
                "source_persistence_residual_cache",
                "residual_h4s_window_ids_and_labels",
                "six_residual_blocks",
                "two_convolutions_per_block",
                "kernel_size",
                "hidden_channels",
                "dropout",
                "normalization",
                "classification_head",
                "initial_parameter_values",
                "training_sample_order",
                "optimizer",
                "class_weight",
                "early_stopping",
                "validation_threshold_rule",
            ],
            "same_classifier_seed_within_fold": True,
            "same_initial_state_sha256_within_fold": True,
            "epoch_shuffle_seed_rule": "classifier_seed + epoch",
            "threshold_source": "validation_only_balanced_accuracy",
        },
    }
    fingerprint = canonical_fingerprint(protocol)
    return {
        **protocol,
        "protocol_fingerprint": fingerprint,
        # Runtime locations and worker mechanics are not scientific variables.
        "data_dir": str(args.data_dir),
        "source_suite_dir": str(args.source_suite_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "num_workers": args.num_workers,
        "resume": args.resume,
    }


def checkpoint_base(
    *,
    protocol_fingerprint: str,
    task_id: str,
    source_residual_sha256: str,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "rf_classifier",
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
        "source_residual_sha256": source_residual_sha256,
    }


def validate_rf_checkpoint(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    task_id: str,
    source_residual_sha256: str,
) -> None:
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "rf_classifier",
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
        "source_residual_sha256": source_residual_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Incompatible checkpoint {key}: {payload.get(key)!r} != {value!r}"
            )


def classifier_epoch(
    model: ResidualTCNClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type, enabled=amp and device.type == "cuda"
            ):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(y.numel())
        total_loss += float(loss.detach()) * batch
        total_n += batch
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probabilities.append(
            torch.sigmoid(logits.detach()).float().cpu().numpy()
        )
    if not truths:
        raise RuntimeError("Classifier DataLoader is empty")
    return (
        total_loss / total_n,
        np.concatenate(truths),
        np.concatenate(probabilities),
    )


def array_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    shuffle_seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle:
        if shuffle_seed is None:
            raise ValueError("A deterministic shuffle seed is required")
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def add_requested_metrics(metrics: dict) -> dict:
    tn, fp, fn, tp = [int(metrics[key]) for key in ("tn", "fp", "fn", "tp")]
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    metrics["macro_f1"] = 0.5 * (f1_nonfog + f1_fog)
    metrics["roc_auc"] = metrics.get("auroc")
    metrics["pr_auc"] = metrics.get("auprc")
    metrics["fog_recall"] = metrics.get("sensitivity")
    metrics["fog_f1"] = f1_fog
    return metrics


def _load_source_cache(
    args: argparse.Namespace,
    config: dict,
    subject: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    source_root = (
        args.source_suite_dir / f"loso_{subject}" / SOURCE_NBM
    )
    nbm_done_path = source_root / "nbm" / "DONE.json"
    nbm_done = validate_done(
        nbm_done_path,
        stage="nbm",
        protocol_fingerprint=config["source"]["source_protocol_fingerprint"],
        task_id=f"loso_{subject}/{SOURCE_NBM}/nbm",
    )
    if nbm_done is None:
        raise FileNotFoundError(nbm_done_path)
    source_nbm_sha = residual_done_upstream(nbm_done)
    residual_done_path = source_root / "RESIDUAL_CACHE_DONE.json"
    residual_done = validate_done(
        residual_done_path,
        stage="residual_cache",
        protocol_fingerprint=config["source"]["source_protocol_fingerprint"],
        task_id=f"loso_{subject}/{SOURCE_NBM}/residual_cache",
        upstream_sha256=source_nbm_sha,
    )
    if residual_done is None:
        raise FileNotFoundError(residual_done_path)
    expected = config["source"]["folds"][subject]
    cache_entry = residual_done["artifacts"]["cache"]
    if str(cache_entry["sha256"]) != expected["source_residual_cache_sha256"]:
        raise ValueError(f"Source cache hash changed for {subject}")
    cache_path = _artifact_path(residual_done_path, cache_entry)
    expected_keys = {
        f"{split}_{key}"
        for split in ("train", "validation", "test")
        for key in ("residual", "y", "window_index")
    }
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError(
                f"Unexpected Persistence residual cache arrays in {cache_path}"
            )
        extracted = {
            split: {
                key: np.asarray(payload[f"{split}_{key}"])
                for key in ("residual", "y", "window_index")
            }
            for split in ("train", "validation", "test")
        }
    provenance = {
        "source_protocol_fingerprint": config["source"][
            "source_protocol_fingerprint"
        ],
        "source_nbm": SOURCE_NBM,
        "source_nbm_best_sha256": source_nbm_sha,
        "source_residual_cache_sha256": str(cache_entry["sha256"]),
        "source_residual_cache_bytes": int(cache_entry["bytes"]),
        "source_residual_done_sha256": sha256_file(residual_done_path),
    }
    return extracted, provenance


def prepare_fold_inputs(
    args: argparse.Namespace,
    config: dict,
    dataset: DaphnetDataset,
    windows: WindowTable,
    subject: str,
) -> tuple[Path, dict[str, dict[str, np.ndarray]], dict]:
    fold_root = args.output_dir / f"loso_{subject}"
    fold_root.mkdir(parents=True, exist_ok=True)
    extracted, provenance = _load_source_cache(args, config, subject)
    source_fold_config_path = (
        args.source_suite_dir / f"loso_{subject}" / "fold_config.json"
    )
    source_fold_config = _load_json(source_fold_config_path)
    if source_fold_config.get("protocol_fingerprint") != config["source"][
        "source_protocol_fingerprint"
    ]:
        raise ValueError(f"Source fold config protocol mismatch for {subject}")
    if source_fold_config.get("test_subject") != subject:
        raise ValueError(f"Source fold config test subject mismatch for {subject}")

    plans: dict[str, HistoryPlan] = {}
    inputs: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        residual = np.asarray(extracted[split]["residual"])
        labels = np.asarray(extracted[split]["y"], dtype=np.int8)
        indices = np.asarray(extracted[split]["window_index"], dtype=np.int64)
        if residual.ndim != 3 or residual.shape[1:] != (
            dataset.n_channels,
            int(config["horizon_samples"]),
        ):
            raise ValueError(f"Unexpected residual shape for {subject}/{split}")
        if not np.isfinite(residual).all():
            raise ValueError(f"Non-finite residual values for {subject}/{split}")
        if len(residual) != len(labels) or len(labels) != len(indices):
            raise ValueError(f"Source cache arrays are misaligned for {subject}/{split}")
        if not np.array_equal(labels, windows.label[indices]):
            raise ValueError(f"Source labels differ from WindowTable for {subject}/{split}")
        plans[split] = make_common_history_plan(
            windows,
            indices,
            int(config["horizon_samples"]),
            int(config["stride_samples"]),
            HISTORY_SAMPLES,
        )
        expected_source_windows = int(
            source_fold_config["source_window_counts"][split]
        )
        if len(indices) != expected_source_windows:
            raise ValueError(
                f"Source window count changed for {subject}/{split}"
            )
    if min(len(plan.anchor_rows) for plan in plans.values()) == 0:
        raise RuntimeError(f"Empty four-second history support in fold {subject}")

    # Bind all three split supports, including training, to the support used by
    # the completed source suite before applying an optional smoke-test cap.
    source_history_support_path = (
        args.source_suite_dir / f"loso_{subject}" / "history_support.npz"
    )
    expected_history_keys = {
        f"{split}_{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("anchor_window_index", "history_window_index")
    }
    with np.load(source_history_support_path, allow_pickle=False) as source_support:
        if set(source_support.files) != expected_history_keys:
            raise ValueError(
                f"Unexpected source history support arrays for {subject}"
            )
        for split, plan in plans.items():
            source_indices = np.asarray(
                extracted[split]["window_index"], dtype=np.int64
            )
            if not np.array_equal(
                source_support[f"{split}_anchor_window_index"],
                plan.anchor_window_indices,
            ):
                raise ValueError(
                    f"History anchors differ from source suite: {subject}/{split}"
                )
            if not np.array_equal(
                source_support[f"{split}_history_window_index"],
                source_indices[plan.max_chain_rows],
            ):
                raise ValueError(
                    f"History chains differ from source suite: {subject}/{split}"
                )
            if len(plan.anchor_rows) != int(
                source_fold_config["history_anchor_counts"][split]
            ):
                raise ValueError(
                    f"History anchor count changed for {subject}/{split}"
                )
    if args.max_classifier_windows > 0:
        rows = np.arange(len(plans["train"].anchor_rows), dtype=np.int64)
        anchor_labels = windows.label[plans["train"].anchor_window_indices]
        selected = deterministic_subsample(
            rows,
            args.max_classifier_windows,
            args.seed + 100 + EXPECTED_LOSO_SUBJECTS.index(subject),
            anchor_labels,
        )
        plans["train"] = plans["train"].take(selected)
    for split in ("train", "validation", "test"):
        inputs[split] = make_history_input(
            extracted[split],
            plans[split],
            INPUT_NAME,
            HISTORY_SAMPLES,
            int(config["horizon_samples"]),
            int(config["stride_samples"]),
        )
        if inputs[split][INPUT_NAME].shape[2] != HISTORY_SAMPLES:
            raise AssertionError("Residual history is not exactly four seconds")

    # Cross-check the source suite's existing h4 classifier support.  Its
    # probabilities are not used; only immutable window ids and labels are read.
    source_prediction_files = {
        "validation": (
            args.source_suite_dir
            / f"loso_{subject}"
            / SOURCE_NBM
            / INPUT_NAME
            / "validation_predictions.npz"
        ),
        "test": (
            args.source_suite_dir
            / f"loso_{subject}"
            / SOURCE_NBM
            / INPUT_NAME
            / "predictions.npz"
        ),
    }
    for split, path in source_prediction_files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(
                payload["window_index"], inputs[split]["window_index"]
            ):
                raise ValueError(
                    f"Four-second anchor ids differ from source suite: {subject}/{split}"
                )
            if not np.array_equal(payload["y_true"], inputs[split]["y"]):
                raise ValueError(
                    f"Four-second labels differ from source suite: {subject}/{split}"
                )

    support_arrays: dict[str, np.ndarray] = {}
    for split, plan in plans.items():
        source_indices = np.asarray(
            extracted[split]["window_index"], dtype=np.int64
        )
        support_arrays[f"{split}_anchor_window_index"] = (
            plan.anchor_window_indices
        )
        support_arrays[f"{split}_history_window_index"] = source_indices[
            plan.max_chain_rows
        ]
        support_arrays[f"{split}_y"] = np.asarray(inputs[split]["y"], dtype=np.int8)
    support_path = fold_root / "input_support.npz"
    save_or_validate_npz(support_path, **support_arrays)
    provenance = {
        **provenance,
        "source_fold_config_sha256": sha256_file(source_fold_config_path),
        "source_validation_predictions_sha256": sha256_file(
            source_prediction_files["validation"]
        ),
        "source_test_predictions_sha256": sha256_file(
            source_prediction_files["test"]
        ),
        "source_history_support_sha256": sha256_file(
            source_history_support_path
        ),
        "source_history_support_bytes": int(
            source_history_support_path.stat().st_size
        ),
        "input_support_sha256": sha256_file(support_path),
    }
    fold_index = EXPECTED_LOSO_SUBJECTS.index(subject)
    classifier_seed = args.seed + 10000 + fold_index
    set_seed(classifier_seed, args.deterministic)
    reference_model = build_model(
        in_channels=dataset.n_channels,
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=tuple(TCN_VARIANTS["local"]["dilations"]),
    )
    fold_initial_state_sha256 = state_dict_sha256(reference_model.state_dict())
    del reference_model
    fold_config = {
        "suite_version": SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "test_subject": subject,
        "val_subject": source_fold_config["val_subject"],
        "train_subjects": source_fold_config["train_subjects"],
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256": fold_initial_state_sha256,
        "source": provenance,
        "input": INPUT_NAME,
        "history_seconds": HISTORY_SECONDS,
        "history_samples": HISTORY_SAMPLES,
        "history_blocks": HISTORY_BLOCKS,
        "history_anchor_counts": {
            split: int(len(plans[split].anchor_rows))
            for split in ("train", "validation", "test")
        },
        "history_construction": (
            "Eight chronological, horizon-spaced 32-sample residual blocks; "
            "no overlap between blocks."
        ),
    }
    save_or_validate_json(fold_root / "fold_config.json", fold_config)
    save_or_validate_json(fold_root / "source_provenance.json", provenance)
    return fold_root, inputs, fold_config


def train_classifier_resumable(
    args: argparse.Namespace,
    config: dict,
    variant: dict,
    task_root: Path,
    fold_config: dict,
    inputs: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    device: torch.device,
) -> dict:
    task_root.mkdir(parents=True, exist_ok=True)
    subject = str(fold_config["test_subject"])
    task_id = f"{subject}/{variant['variant']}"
    source_residual_sha = str(
        fold_config["source"]["source_residual_cache_sha256"]
    )
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    metrics_path = task_root / "metrics.json"
    predictions_path = task_root / "predictions.npz"
    validation_predictions_path = task_root / "validation_predictions.npz"
    predictions_csv_path = task_root / "predictions.csv"
    done_path = task_root / "DONE.json"
    complete = validate_done(
        done_path,
        stage="rf_classifier",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
    )
    if complete is not None:
        if complete.get("source_residual_sha256") != source_residual_sha:
            raise ValueError(f"Completed task uses another source cache: {task_id}")
        return _load_json(metrics_path)

    classifier_seed = int(fold_config["classifier_seed"])
    set_seed(classifier_seed, args.deterministic)
    x_train = np.asarray(inputs["train"][INPUT_NAME], dtype=np.float32)
    y_train = np.asarray(inputs["train"]["y"], dtype=np.int8)
    x_val = np.asarray(inputs["validation"][INPUT_NAME], dtype=np.float32)
    y_val = np.asarray(inputs["validation"]["y"], dtype=np.int8)
    x_test = np.asarray(inputs["test"][INPUT_NAME], dtype=np.float32)
    y_test = np.asarray(inputs["test"]["y"], dtype=np.int8)
    dilations = tuple(int(value) for value in variant["dilations"])
    model = build_model(
        in_channels=x_train.shape[1],
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=dilations,
    ).to(device)
    reference_initial_state = fold_config.get("_reference_initial_state")
    if reference_initial_state is not None:
        model.load_state_dict(reference_initial_state, strict=True)
    initial_state_sha = state_dict_sha256(model.state_dict())
    if initial_state_sha != fold_config["reference_initial_state_sha256"]:
        raise AssertionError(
            f"Initial parameter state differs for {subject}/{variant['variant']}"
        )
    actual_parameter_count = parameter_count(model)
    if actual_parameter_count != int(config["shared_parameter_count"]):
        raise AssertionError("TCN variant parameter count is not shared")

    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if min(counts) <= 0:
        raise RuntimeError(f"Training split lacks a class in fold {subject}")
    pos_weight_value = min(math.sqrt(counts[0] / counts[1]), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    pin_memory = device.type == "cuda"
    validation_loader = array_loader(
        x_val,
        y_val,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = array_loader(
        x_test,
        y_test,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    classifier_config = {
        "in_channels": int(x_train.shape[1]),
        "hidden_channels": args.classifier_hidden,
        "dropout": args.classifier_dropout,
        "kernel_size": KERNEL_SIZE,
        "dilations": list(dilations),
        "n_blocks": len(dilations),
        "convolutions_per_block": CONVS_PER_BLOCK,
        "receptive_field_samples": int(variant["receptive_field_samples"]),
        "receptive_field_seconds": float(variant["receptive_field_seconds"]),
        "parameter_count": actual_parameter_count,
        "initial_state_sha256": initial_state_sha,
        "global_pooling": "mean_and_max_over_full_input",
    }

    start_epoch = 0
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict] = []
    elapsed_before = 0.0
    if args.resume and last_path.exists():
        # Keep RNG state tensors on CPU.  Loading the whole checkpoint directly
        # onto CUDA makes torch.set_rng_state reject the saved CPU RNG tensor
        # during an interrupted multi-GPU resume.
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        validate_rf_checkpoint(
            payload,
            protocol_fingerprint=config["protocol_fingerprint"],
            task_id=task_id,
            source_residual_sha256=source_residual_sha,
        )
        if payload.get("classifier_config") != classifier_config:
            raise ValueError(f"Classifier checkpoint configuration changed: {task_id}")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        grad_scaler.load_state_dict(payload["grad_scaler_state"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        restore_rng_state(payload["rng_state"])
        print(
            f"      [{variant['display_name']}] resume at epoch {start_epoch + 1}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
        train_loader = array_loader(
            x_train,
            y_train,
            args.batch_size,
            shuffle=True,
            shuffle_seed=classifier_seed + epoch,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        train_loss, train_true, train_prob = classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            validation_loss, validation_true, validation_prob = classifier_epoch(
                model,
                validation_loader,
                criterion,
                device,
                args.amp,
            )
        validation_auprc = float(
            average_precision_score(validation_true, validation_prob)
        )
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": classifier_seed + epoch,
                "train_loss": train_loss,
                "train_auprc": float(
                    average_precision_score(train_true, train_prob)
                ),
                "validation_loss": validation_loss,
                "validation_auprc": validation_auprc,
            }
        )
        improved = validation_auprc > best_score + 1e-5
        if improved:
            best_epoch = epoch
            best_score = validation_auprc
            bad_epochs = 0
            atomic_torch_save(
                {
                    **checkpoint_base(
                        protocol_fingerprint=config["protocol_fingerprint"],
                        task_id=task_id,
                        source_residual_sha256=source_residual_sha,
                    ),
                    "variant": variant["variant"],
                    "classifier_seed": classifier_seed,
                    "classifier_config": classifier_config,
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_auprc": best_score,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch_save(
            {
                **checkpoint_base(
                    protocol_fingerprint=config["protocol_fingerprint"],
                    task_id=task_id,
                    source_residual_sha256=source_residual_sha,
                ),
                "variant": variant["variant"],
                "classifier_seed": classifier_seed,
                "classifier_config": classifier_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_sec": elapsed,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
        print(
            f"      [{variant['display_name']}] epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} val_auprc={validation_auprc:.5f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        interrupt_marker = task_root / ".debug_interrupted_once"
        if (
            args.debug_interrupt_classifier_after_epoch > 0
            and epoch >= args.debug_interrupt_classifier_after_epoch
            and not interrupt_marker.exists()
        ):
            atomic_json_dump({"interrupted_after_epoch": epoch}, interrupt_marker)
            raise RuntimeError("Intentional classifier interruption after checkpoint")

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was produced: {task_id}")
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    validate_rf_checkpoint(
        best_payload,
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        source_residual_sha256=source_residual_sha,
    )
    if best_payload.get("classifier_config") != classifier_config:
        raise ValueError(f"Best checkpoint configuration changed: {task_id}")
    model.load_state_dict(best_payload["model_state"])
    with torch.no_grad():
        _, validation_true, validation_prob = classifier_epoch(
            model, validation_loader, criterion, device, args.amp
        )
        _, test_true, test_prob = classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, validation_metrics = choose_threshold(
        validation_true, validation_prob
    )
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (
        np.asarray(test_prob, dtype=np.float64) >= float(threshold)
    ).astype(np.int8)
    test_metrics.update(
        event_metrics(
            dataset,
            windows,
            inputs["test"]["window_index"],
            test_pred,
        )
    )
    test_metrics.update(
        {
            "experiment_id": variant["experiment_id"],
            "variant": variant["variant"],
            "display_name": variant["display_name"],
            "nbm": SOURCE_NBM,
            "input": INPUT_NAME,
            "history_seconds": HISTORY_SECONDS,
            "history_samples": HISTORY_SAMPLES,
            "history_blocks": HISTORY_BLOCKS,
            "test_subject": subject,
            "val_subject": fold_config["val_subject"],
            "classifier_seed": classifier_seed,
            "classifier_config": classifier_config,
            "initial_state_sha256": initial_state_sha,
            "best_epoch": int(best_payload["best_epoch"]),
            "best_validation_auprc": float(
                best_payload["best_validation_auprc"]
            ),
            "validation": validation_metrics,
            "train_counts": counts.astype(int).tolist(),
            "pos_weight": float(pos_weight_value),
            "elapsed_sec": elapsed_before + time.perf_counter() - started,
            "history": history,
            "source_residual_sha256": source_residual_sha,
            "input_support_sha256": fold_config["source"][
                "input_support_sha256"
            ],
        }
    )
    add_requested_metrics(test_metrics)
    atomic_json_dump(test_metrics, metrics_path)
    # Float64 probabilities ensure the auditor can reproduce the saved threshold
    # decisions exactly without a float32 boundary-rounding ambiguity.
    atomic_npz_save(
        predictions_path,
        window_index=np.asarray(
            inputs["test"]["window_index"], dtype=np.int64
        ),
        y_true=np.asarray(test_true, dtype=np.int8),
        y_prob=np.asarray(test_prob, dtype=np.float64),
        y_pred=test_pred,
    )
    validation_pred = (
        np.asarray(validation_prob, dtype=np.float64) >= float(threshold)
    ).astype(np.int8)
    atomic_npz_save(
        validation_predictions_path,
        window_index=np.asarray(
            inputs["validation"]["window_index"], dtype=np.int64
        ),
        y_true=np.asarray(validation_true, dtype=np.int8),
        y_prob=np.asarray(validation_prob, dtype=np.float64),
        y_pred=validation_pred,
    )
    write_predictions_csv(
        predictions_csv_path,
        dataset,
        windows,
        inputs["test"]["window_index"],
        np.asarray(test_prob, dtype=np.float64),
        test_pred,
    )
    completed = done_payload(
        stage="rf_classifier",
        protocol_fingerprint=config["protocol_fingerprint"],
        task_id=task_id,
        relative_to=task_root,
        artifacts={
            "best": best_path.resolve(),
            "last": last_path.resolve(),
            "metrics": metrics_path.resolve(),
            "predictions": predictions_path.resolve(),
            "validation_predictions": validation_predictions_path.resolve(),
            "predictions_csv": predictions_csv_path.resolve(),
        },
    )
    completed["source_residual_sha256"] = source_residual_sha
    completed["input_support_sha256"] = fold_config["source"][
        "input_support_sha256"
    ]
    completed["initial_state_sha256"] = initial_state_sha
    atomic_json_dump(completed, done_path)
    return test_metrics


def prediction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall_fog = tp / (tp + fn) if tp + fn else 0.0
    recall_nonfog = tn / (tn + fp) if tn + fp else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_nonfog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (recall_fog + recall_nonfog),
        "macro_f1": 0.5 * (f1_fog + f1_nonfog),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else None
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else None
        ),
        "fog_recall": recall_fog,
        "fog_f1": f1_fog,
        "specificity": recall_nonfog,
    }


def paired_delta_summary(
    rows_by_variant: dict[str, dict[str, dict]],
) -> dict:
    result: dict[str, dict] = {}
    reference = rows_by_variant.get("local", {})
    for variant in ("medium", "long"):
        comparison = rows_by_variant.get(variant, {})
        metric_payload: dict[str, dict] = {}
        common_subjects = [
            subject
            for subject in EXPECTED_LOSO_SUBJECTS
            if subject in reference and subject in comparison
        ]
        for metric in CLASSIFICATION_METRICS:
            values = []
            for subject in common_subjects:
                before = reference[subject].get(metric)
                after = comparison[subject].get(metric)
                if before is None or after is None:
                    continue
                values.append(float(after) - float(before))
            array = np.asarray(values, dtype=np.float64)
            metric_payload[metric] = {
                "mean_delta_vs_local": (
                    float(array.mean()) if len(array) else None
                ),
                "std_delta_vs_local": (
                    float(array.std(ddof=0)) if len(array) else None
                ),
                "n_paired_folds": int(len(array)),
            }
        result[variant] = {
            "reference": "local",
            "common_subjects": common_subjects,
            "metrics": metric_payload,
        }
    return result


def refresh_summaries(output_dir: Path, config: dict) -> None:
    fold_rows: list[dict] = []
    manifest_rows: list[dict] = []
    aggregate: dict[str, dict] = {}
    rows_by_variant: dict[str, dict[str, dict]] = {
        name: {} for name in TCN_VARIANTS
    }
    summary_rows: list[dict] = []
    expected_folds = list(config["folds_resolved"])
    for variant in config["variants"]:
        name = variant["variant"]
        group_rows: list[dict] = []
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        completed: list[str] = []
        for subject in expected_folds:
            task_root = output_dir / f"loso_{subject}" / name
            metrics_path = task_root / "metrics.json"
            predictions_path = task_root / "predictions.npz"
            done_path = task_root / "DONE.json"
            if not (
                metrics_path.exists()
                and predictions_path.exists()
                and done_path.exists()
            ):
                continue
            metrics = _load_json(metrics_path)
            with np.load(predictions_path, allow_pickle=False) as payload:
                truths.append(np.asarray(payload["y_true"], dtype=np.int8))
                probabilities.append(
                    np.asarray(payload["y_prob"], dtype=np.float64)
                )
                predictions.append(
                    np.asarray(payload["y_pred"], dtype=np.int8)
                )
            group_rows.append(metrics)
            fold_rows.append(metrics)
            rows_by_variant[name][subject] = metrics
            completed.append(subject)
        if group_rows:
            subject_macro = aggregate_fold_metrics(
                group_rows, CLASSIFICATION_METRICS
            )
            aggregate[variant["experiment_id"]] = {
                "variant": name,
                "display_name": variant["display_name"],
                "dilations": variant["dilations"],
                "receptive_field_samples": variant[
                    "receptive_field_samples"
                ],
                "receptive_field_seconds": variant[
                    "receptive_field_seconds"
                ],
                "parameter_count": variant["parameter_count"],
                "completed_folds": completed,
                "subject_macro": subject_macro,
                "pooled": prediction_metrics(
                    np.concatenate(truths),
                    np.concatenate(probabilities),
                    np.concatenate(predictions),
                ),
            }
            summary_row = {
                "variant": name,
                "display_name": variant["display_name"],
                "dilations": ",".join(map(str, variant["dilations"])),
                "receptive_field_samples": variant[
                    "receptive_field_samples"
                ],
                "receptive_field_seconds": variant[
                    "receptive_field_seconds"
                ],
                "parameter_count": variant["parameter_count"],
                "completed_folds": len(completed),
            }
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "pr_auc",
                "fog_recall",
                "fog_f1",
            ):
                summary_row[f"{metric}_mean"] = subject_macro[metric]["mean"]
                summary_row[f"{metric}_std"] = subject_macro[metric]["std"]
            summary_rows.append(summary_row)
        manifest_rows.append(
            {
                "experiment_id": variant["experiment_id"],
                "variant": name,
                "display_name": variant["display_name"],
                "dilations": ",".join(map(str, variant["dilations"])),
                "receptive_field_samples": variant[
                    "receptive_field_samples"
                ],
                "receptive_field_seconds": variant[
                    "receptive_field_seconds"
                ],
                "parameter_count": variant["parameter_count"],
                "expected_folds": len(expected_folds),
                "completed_folds": len(completed),
                "status": (
                    "complete"
                    if completed == expected_folds
                    else ("partial" if completed else "pending")
                ),
                "completed_subjects": ",".join(completed),
            }
        )
    aggregate["paired_deltas"] = paired_delta_summary(rows_by_variant)
    fold_columns = [
        "experiment_id",
        "variant",
        "display_name",
        "nbm",
        "input",
        "history_seconds",
        "history_samples",
        "history_blocks",
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
        "initial_state_sha256",
        "source_residual_sha256",
        "input_support_sha256",
    ]
    atomic_csv_write(output_dir / "fold_summary.csv", fold_rows, fold_columns)
    manifest_columns = [
        "experiment_id",
        "variant",
        "display_name",
        "dilations",
        "receptive_field_samples",
        "receptive_field_seconds",
        "parameter_count",
        "expected_folds",
        "completed_folds",
        "status",
        "completed_subjects",
    ]
    atomic_csv_write(
        output_dir / "experiment_manifest.csv",
        manifest_rows,
        manifest_columns,
    )
    aggregate_columns = [
        "variant",
        "display_name",
        "dilations",
        "receptive_field_samples",
        "receptive_field_seconds",
        "parameter_count",
        "completed_folds",
        *[
            f"{metric}_{statistic}"
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "pr_auc",
                "fog_recall",
                "fog_f1",
            )
            for statistic in ("mean", "std")
        ],
    ]
    atomic_csv_write(
        output_dir / "aggregate_summary.csv",
        summary_rows,
        aggregate_columns,
    )
    atomic_json_dump(aggregate, output_dir / "aggregate_metrics.json")
    completed_cells = sum(int(row["completed_folds"]) for row in manifest_rows)
    expected_cells = len(expected_folds) * len(config["variants"])
    atomic_json_dump(
        {
            "suite_version": SUITE_VERSION,
            "protocol_fingerprint": config["protocol_fingerprint"],
            "expected_experiments": len(config["variants"]),
            "expected_fold_cells": expected_cells,
            "completed_fold_cells": completed_cells,
            "status": "complete" if completed_cells == expected_cells else "partial",
        },
        output_dir / "status.json",
    )


def initialize_protocol(
    args: argparse.Namespace,
    device: torch.device,
    worker_mode: bool,
) -> tuple[dict, DaphnetDataset, WindowTable]:
    source_manifest, source_config = build_source_manifest(
        args.source_suite_dir,
        verify_artifacts=not worker_mode,
    )
    dataset, windows, data_sha256 = load_dataset_and_windows(
        args.data_dir, source_config
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
            "Missing config.json; initialize with --finalize-only before workers"
        )
    if config_path.exists():
        existing = _load_json(config_path)
        if existing.get("protocol_fingerprint") != config["protocol_fingerprint"]:
            raise ValueError(
                "Cannot resume with a different protocol; use a new output directory"
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
    }
    run_manifest = {
        key: value for key, value in config.items() if key not in runtime_fields
    }
    run_manifest_path = args.output_dir / "run_manifest.json"
    if worker_mode:
        if not run_manifest_path.exists():
            raise RuntimeError("Missing run_manifest.json for worker")
        if _load_json(run_manifest_path) != run_manifest:
            raise ValueError(f"Saved JSON is incompatible: {run_manifest_path}")
    else:
        save_or_validate_json(run_manifest_path, run_manifest)
    return config, dataset, windows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.source_suite_dir = args.source_suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_output_path(
        args.output_dir,
        args.source_suite_dir,
        args.data_dir,
    )
    worker_mode = bool(str(args.worker_fold).strip())
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new output directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed, args.deterministic)
    configured_folds = parse_folds(args.folds, list(EXPECTED_LOSO_SUBJECTS))
    if tuple(configured_folds) != EXPECTED_LOSO_SUBJECTS:
        raise ValueError(
            "This strict experiment requires --folds all (the canonical 8 folds)"
        )
    execution_folds = list(configured_folds)
    if worker_mode:
        worker_folds = parse_folds(
            str(args.worker_fold), list(EXPECTED_LOSO_SUBJECTS)
        )
        if len(worker_folds) != 1:
            raise ValueError("--worker-fold must resolve to exactly one subject")
        execution_folds = worker_folds
    config, dataset, windows = initialize_protocol(args, device, worker_mode)
    current_environment = environment_payload(device)
    if worker_mode:
        current_environment.update(
            {
                "protocol_fingerprint": config["protocol_fingerprint"],
                "worker_fold": execution_folds[0],
            }
        )
        atomic_json_dump(
            current_environment,
            args.output_dir
            / "worker_environments"
            / f"loso_{execution_folds[0]}.json",
        )
    else:
        atomic_json_dump(current_environment, args.output_dir / "environment.json")
        refresh_summaries(args.output_dir, config)
    print(
        f"[INFO] suite={SUITE_VERSION} device={device} "
        f"source={args.source_suite_dir} folds={execution_folds} "
        f"variants={list(TCN_VARIANTS)} input={INPUT_NAME}",
        flush=True,
    )
    if args.finalize_only:
        print("[INFO] finalize-only: root summaries refreshed", flush=True)
        print(
            json.dumps(
                _load_json(args.output_dir / "status.json"),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    for subject in execution_folds:
        fold_root, inputs, fold_config = prepare_fold_inputs(
            args, config, dataset, windows, subject
        )
        print(
            f"[fold {subject}] train={fold_config['train_subjects']} "
            f"val={fold_config['val_subject']} "
            f"anchors={fold_config['history_anchor_counts']}",
            flush=True,
        )
        initial_hashes: set[str] = set()
        for variant in config["variants"]:
            metrics = train_classifier_resumable(
                args,
                config,
                variant,
                fold_root / variant["variant"],
                fold_config,
                inputs,
                dataset,
                windows,
                device,
            )
            initial_hashes.add(str(metrics["initial_state_sha256"]))
            print(
                f"[fold {subject}] {variant['display_name']} "
                f"RF={variant['receptive_field_samples']} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"PR-AUC={metrics['pr_auc']:.4f} "
                f"FoG-Recall={metrics['fog_recall']:.4f}",
                flush=True,
            )
        if len(initial_hashes) != 1:
            raise AssertionError(
                f"Variants did not share identical initialization in {subject}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not worker_mode:
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
