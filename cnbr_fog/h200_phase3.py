"""Leakage-safe Phase 3A/3B orchestration for the GRU-H200 experiment.

This module owns the confirmatory subject-cross-fitting path.  It deliberately
keeps the already audited H200 source runner immutable and calls its public
``train_nbm_resumable`` API for each inner GRU.  The important invariants are:

* an inner scaler is fit only on the inner predictor's training subjects;
* NBM train/early-stop rows are clean non-FOG rows from those subjects;
* every outer-training row is forecast exactly once by a model that did not
  train on that row's subject;
* outer validation/test Gaussians are moment-matched in physical IMU units;
* all three classifier arms use identical endpoints and outer-fold scaling;
* classifier repetitions are averaged within subject before paired inference.

The public ``run_phase3a`` and ``run_phase3b`` functions intentionally match
the hook signature used by ``scripts/run_daphnet_gru_residual_feasibility.py``.
No import from that script is required, which avoids a circular dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DaphnetDataset, RobustChannelScaler, WindowTable
from .evaluation import binary_metrics, choose_threshold
from .h200_crossfit import (
    assemble_oof_gaussians,
    convert_to_outer_scaler_primitives,
    ensemble_gaussians,
    extract_gaussian_forecasts,
    temporal_clean_normal_split,
)
from .h200_feasibility import (
    H200_ARM_REGISTRY,
    build_arm_inputs,
    build_classifier,
    build_subject_crossfit_plan,
    paired_bootstrap,
)
from .resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    capture_rng_state,
    canonical_fingerprint,
    done_payload,
    restore_rng_state,
    sha256_file,
    validate_done,
)


PHASE3_VERSION = "daphnet_gru_h200_phase3_crossfit.v1"
PHASE3_ARMS = ("raw6", "raw4_zero", "raw4_normality")
PHASE3A_OUTER_SUBJECTS = ("S01", "S05", "S08")
DEFAULT_CLASSIFIER_SEEDS = {"3a": (42,), "3b": (42, 43, 44)}
DEFAULT_NBM_SEEDS = (42,)
CONTEXT_SAMPLES = 128
HORIZON_SAMPLES = 128
HISTORY_SAMPLES = 256
RAW6_SAMPLES = 384
Z_CLIP = 12.0


@dataclass(frozen=True)
class OuterFoldContext:
    """The immutable 6/1/1 outer split and source-compatible support."""

    subject: str
    val_subject: str
    train_subjects: tuple[str, ...]
    scaler: RobustChannelScaler
    split_indices: dict[str, np.ndarray]
    support: dict[str, np.ndarray]
    source_fold_config: dict[str, Any]


@dataclass(frozen=True)
class InnerPredictorArtifact:
    """One fitted inner model plus its physical-unit forecast caches."""

    predictor_id: str
    inner_fold_index: int
    train_subjects: tuple[str, ...]
    heldout_subjects: tuple[str, ...]
    scaler: RobustChannelScaler
    checkpoint_sha256: str
    heldout_forecast: dict[str, Any]
    validation_forecast: dict[str, Any]
    test_forecast: dict[str, Any]
    provenance: dict[str, Any]


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    return default if value is None else value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _parse_seed_list(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    if value is None or value == "":
        result = tuple(int(seed) for seed in default)
    elif isinstance(value, str):
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    elif isinstance(value, Iterable):
        result = tuple(int(item) for item in value)
    else:
        result = (int(value),)
    if not result or len(set(result)) != len(result):
        raise ValueError("seed lists must be non-empty and contain no duplicates")
    return result


def phase3_seed_policy(args: argparse.Namespace, phase: str) -> dict[str, tuple[int, ...]]:
    """Return explicit NBM/classifier seed policies for one Phase 3 stage."""

    if phase not in {"3a", "3b"}:
        raise ValueError("phase must be '3a' or '3b'")
    nbm_seeds = _parse_seed_list(
        getattr(args, "phase3_nbm_seeds", None), DEFAULT_NBM_SEEDS
    )
    classifier_seeds = _parse_seed_list(
        getattr(args, "phase3_classifier_seeds", None),
        DEFAULT_CLASSIFIER_SEEDS[phase],
    )
    if phase == "3a" and getattr(args, "phase3_classifier_seeds", None) is None:
        classifier_seeds = classifier_seeds[:1]
    return {"nbm": nbm_seeds, "classifier": classifier_seeds}


def phase3_outer_subjects(protocol: Any, phase: str) -> tuple[str, ...]:
    """Enforce the preregistered 3A subset and all-fold 3B confirmation."""

    all_subjects = tuple(str(value) for value in protocol.config["subjects"])
    requested = tuple(str(value) for value in protocol.folds)
    if phase == "3a":
        expected = PHASE3A_OUTER_SUBJECTS
    elif phase == "3b":
        expected = all_subjects
    else:
        raise ValueError("phase must be '3a' or '3b'")
    missing = set(expected) - set(requested)
    if missing and not bool(getattr(protocol, "phase3_allow_subset", False)):
        raise ValueError(
            f"Phase {phase} requires folds {expected}; --folds omitted {sorted(missing)}"
        )
    selected = tuple(subject for subject in expected if subject in set(requested))
    if not selected:
        raise ValueError(f"No preregistered Phase {phase} folds were selected")
    return selected


def _legacy_runners() -> tuple[Any, Any]:
    """Load immutable training/evaluation helpers without package coupling."""

    try:
        nbm_runner = importlib.import_module("run_daphnet_3imu_nbm_suite")
    except ModuleNotFoundError:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        nbm_runner = importlib.import_module("run_daphnet_3imu_nbm_suite")
    rf_runner = importlib.import_module("run_daphnet_tcn_rf_ablation")
    return nbm_runner, rf_runner


def _scaler_from_mapping(value: Mapping[str, Any]) -> RobustChannelScaler:
    return RobustChannelScaler(
        center=np.asarray(value["center"], dtype=np.float32),
        scale=np.asarray(value["scale"], dtype=np.float32),
        clip=float(value["clip"]),
    )


def load_outer_fold(args: argparse.Namespace, protocol: Any, subject: str) -> OuterFoldContext:
    """Load and verify the source suite's exact 6/1/1 split and endpoints."""

    root = Path(args.source_suite_dir) / f"loso_{subject}"
    config = _read_json(root / "fold_config.json")
    if config.get("test_subject") != subject:
        raise ValueError(f"Outer fold identity differs for {subject}")
    val_subject = str(config["val_subject"])
    train_subjects = tuple(str(value) for value in config["train_subjects"])
    all_subjects = set(str(value) for value in protocol.config["subjects"])
    if (
        len(train_subjects) != 6
        or set(train_subjects) & {subject, val_subject}
        or set(train_subjects) | {subject, val_subject} != all_subjects
    ):
        raise ValueError(f"Invalid outer 6/1/1 split: {subject}")

    scaler_payload = _read_json(root / "scaler.json")
    scaler = _scaler_from_mapping(scaler_payload)
    recomputed = protocol.dataset.fit_scaler(train_subjects, clip=scaler.clip)
    np.testing.assert_array_equal(recomputed.center, scaler.center)
    np.testing.assert_array_equal(recomputed.scale, scaler.scale)

    with np.load(root / "split_indices.npz", allow_pickle=False) as source:
        split_indices = {
            split: np.asarray(source[f"{split}_window_index"], dtype=np.int64)
            for split in ("train", "validation", "test")
        }
    expected = {
        "train": protocol.dataset.window_indices_for_subjects(
            protocol.classification_windows, train_subjects
        ),
        "validation": protocol.dataset.window_indices_for_subjects(
            protocol.classification_windows, (val_subject,)
        ),
        "test": protocol.dataset.window_indices_for_subjects(
            protocol.classification_windows, (subject,)
        ),
    }
    for split in expected:
        if not np.array_equal(split_indices[split], expected[split]):
            raise ValueError(f"Outer {subject}/{split} source indices changed")

    with np.load(root / "common_history_support.npz", allow_pickle=False) as source:
        support = {key: np.asarray(source[key]) for key in source.files}
    for split in ("train", "validation", "test"):
        anchors = np.asarray(support[f"{split}_anchor_window_index"], dtype=np.int64)
        labels = np.asarray(support[f"{split}_y"], dtype=np.int8)
        histories = np.asarray(
            support[f"{split}_h200_history_window_index"], dtype=np.int64
        )
        if histories.shape != (len(anchors), 2):
            raise ValueError(f"Outer {subject}/{split} requires two H200 blocks")
        if not np.array_equal(protocol.classification_windows.label[anchors], labels):
            raise ValueError(f"Outer {subject}/{split} labels changed")
        if not set(histories.ravel().tolist()).issubset(
            set(split_indices[split].tolist())
        ):
            raise ValueError(f"Outer {subject}/{split} history leaves its split")
    return OuterFoldContext(
        subject=subject,
        val_subject=val_subject,
        train_subjects=train_subjects,
        scaler=scaler,
        split_indices=split_indices,
        support=support,
        source_fold_config=config,
    )


def _phase_scheme(phase: str) -> str:
    if phase == "3a":
        return "3fold"
    if phase == "3b":
        return "loto"
    raise ValueError("phase must be '3a' or '3b'")


def _inner_protocol_fingerprint(
    args: argparse.Namespace,
    protocol: Any,
    *,
    phase: str,
    outer: OuterFoldContext,
    nbm_seed: int,
    inner_fold: Any,
    temporal_split: Mapping[str, Any],
    scaler: RobustChannelScaler,
) -> str:
    source = _read_json(Path(args.source_suite_dir) / "config.json")
    scientific = {
        "phase3_version": PHASE3_VERSION,
        "outer_protocol_fingerprint": protocol.config["protocol_fingerprint"],
        "phase": phase,
        "scheme": _phase_scheme(phase),
        "outer_test_subject": outer.subject,
        "outer_validation_subject": outer.val_subject,
        "outer_train_subjects": list(outer.train_subjects),
        "nbm_seed": int(nbm_seed),
        "inner_fold_index": int(inner_fold.fold_index),
        "inner_train_subjects": list(inner_fold.train_subjects),
        "inner_heldout_subjects": list(inner_fold.heldout_subjects),
        "normal_train_window_sha256": _array_sha256(
            np.asarray(temporal_split["train_window_index"], dtype=np.int64)
        ),
        "normal_validation_window_sha256": _array_sha256(
            np.asarray(temporal_split["validation_window_index"], dtype=np.int64)
        ),
        "inner_scaler": scaler.as_dict(),
        "nbm_model": {
            "name": "gru",
            "hidden_channels": int(source.get("nbm_hidden", 48)),
            "dropout": float(source.get("nbm_dropout", 0.1)),
            "layers": int(source.get("gru_layers", 1)),
            "context_samples": CONTEXT_SAMPLES,
            "horizon_samples": HORIZON_SAMPLES,
        },
        "optimization": {
            "normal_epochs": int(
                _arg(args, "phase3_normal_epochs", source.get("normal_epochs", 8))
            ),
            "normal_patience": int(
                _arg(args, "phase3_normal_patience", source.get("normal_patience", 3))
            ),
            "normal_lr": float(
                _arg(args, "phase3_normal_lr", source.get("normal_lr", 1e-3))
            ),
            "weight_decay": float(
                _arg(args, "weight_decay", source.get("weight_decay", 1e-4))
            ),
            "batch_size": int(
                _arg(args, "batch_size", source.get("batch_size", 256))
            ),
            "amp": bool(_arg(args, "amp", True)),
            "deterministic": bool(_arg(args, "deterministic", True)),
            "normal_validation_fraction": float(
                _arg(args, "phase3_normal_validation_fraction", 0.2)
            ),
            "max_normal_windows": int(
                _arg(args, "phase3_max_normal_windows", 30_000)
            ),
        },
    }
    return canonical_fingerprint(scientific)


def _nbm_namespace(args: argparse.Namespace, protocol: Any) -> SimpleNamespace:
    """Fill the legacy GRU trainer's arguments from source-suite defaults."""

    source = _read_json(Path(args.source_suite_dir) / "config.json")
    values = dict(vars(args))
    defaults = {
        "nbm_hidden": int(source.get("nbm_hidden", 48)),
        "nbm_dropout": float(source.get("nbm_dropout", 0.1)),
        "gru_layers": int(source.get("gru_layers", 1)),
        "linear_ar_seconds": float(source.get("linear_ar_seconds", 1.0)),
        "transformer_heads": int(source.get("transformer_heads", 3)),
        "transformer_layers": int(source.get("transformer_layers", 2)),
        "transformer_ffn": int(source.get("transformer_ffn", 96)),
        "normal_epochs": int(_arg(args, "phase3_normal_epochs", source.get("normal_epochs", 8))),
        "normal_patience": int(_arg(args, "phase3_normal_patience", source.get("normal_patience", 3))),
        "normal_lr": float(_arg(args, "phase3_normal_lr", source.get("normal_lr", 1e-3))),
        "weight_decay": float(_arg(args, "weight_decay", source.get("weight_decay", 1e-4))),
        "batch_size": int(_arg(args, "batch_size", source.get("batch_size", 256))),
        "num_workers": int(_arg(args, "num_workers", source.get("num_workers", 0))),
        "amp": bool(_arg(args, "amp", True)),
        "deterministic": bool(_arg(args, "deterministic", True)),
        "resume": bool(_arg(args, "resume", True)),
        "debug_interrupt_nbm_after_epoch": int(
            _arg(args, "debug_interrupt_nbm_after_epoch", 0)
        ),
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
        if key.startswith("normal_") or key.startswith("debug_"):
            values[key] = value
    return SimpleNamespace(**values)


def _subject_indices(
    dataset: DaphnetDataset,
    windows: WindowTable,
    subjects: Sequence[str],
    allowed_indices: np.ndarray,
) -> np.ndarray:
    selected = dataset.window_indices_for_subjects(windows, subjects)
    mask = np.isin(selected, np.asarray(allowed_indices, dtype=np.int64), assume_unique=False)
    return np.ascontiguousarray(selected[mask], dtype=np.int64)


def _subject_stratified_subsample(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    subjects: Sequence[str],
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Cap normal rows without silently removing an inner-train subject."""

    values = np.asarray(indices, dtype=np.int64)
    ordered_subjects = tuple(str(value) for value in subjects)
    if maximum <= 0 or len(values) <= maximum:
        return values
    if maximum < len(ordered_subjects):
        raise ValueError("phase3_max_normal_windows is smaller than inner subject count")
    subject_by_record = np.asarray(
        [record.subject_id for record in dataset.records], dtype=np.str_
    )
    row_subject = subject_by_record[windows.record_index[values]]
    groups = [values[row_subject == subject] for subject in ordered_subjects]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Each inner-train subject must contribute a normal train row")
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    allocation = np.maximum(
        1, np.floor(maximum * counts / counts.sum()).astype(np.int64)
    )
    allocation = np.minimum(allocation, counts)
    while int(allocation.sum()) > maximum:
        removable = np.flatnonzero(allocation > 1)
        allocation[removable[np.argmax(allocation[removable])]] -= 1
    while int(allocation.sum()) < maximum:
        room = counts - allocation
        allocation[int(np.argmax(room))] += 1
    rng = np.random.default_rng(int(seed))
    selected = [
        rng.choice(group, size=int(count), replace=False)
        for group, count in zip(groups, allocation)
    ]
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _save_forecast(
    root: Path,
    forecast: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str,
    provenance: Mapping[str, Any],
    compressed: bool,
) -> None:
    arrays_path = root / "physical_forecast.npz"
    provenance_path = root / "provenance.json"
    atomic_npz_save(
        arrays_path,
        compressed=compressed,
        target=np.asarray(forecast["target"], dtype=np.float32),
        mu=np.asarray(forecast["mu"], dtype=np.float32),
        sigma=np.asarray(forecast["sigma"], dtype=np.float32),
        y=np.asarray(forecast["y"], dtype=np.int8),
        window_index=np.asarray(forecast["window_index"], dtype=np.int64),
    )
    atomic_json_dump(dict(provenance), provenance_path)
    atomic_json_dump(
        done_payload(
            stage="h200_phase3_physical_forecast",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=upstream_sha256,
            relative_to=root,
            artifacts={"arrays": arrays_path, "provenance": provenance_path},
        ),
        root / "DONE.json",
    )


def _load_forecast(
    root: Path,
    *,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str,
) -> dict[str, Any] | None:
    completed = validate_done(
        root / "DONE.json",
        stage="h200_phase3_physical_forecast",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
        upstream_sha256=upstream_sha256,
    )
    if completed is None:
        return None
    with np.load(root / "physical_forecast.npz", allow_pickle=False) as source:
        result = {key: np.asarray(source[key]) for key in source.files}
    result["provenance"] = _read_json(root / "provenance.json")
    return result


def _inner_forecast_indices(
    protocol: Any,
    outer: OuterFoldContext,
    heldout_subjects: Sequence[str],
) -> dict[str, np.ndarray]:
    heldout = _subject_indices(
        protocol.dataset,
        protocol.master_windows,
        heldout_subjects,
        outer.split_indices["train"],
    )
    expected = _subject_indices(
        protocol.dataset,
        protocol.classification_windows,
        heldout_subjects,
        outer.split_indices["train"],
    )
    if not np.array_equal(heldout, expected):
        raise ValueError("H200 and classification held-out row identities differ")
    return {
        "heldout": heldout,
        "validation": np.asarray(outer.split_indices["validation"], dtype=np.int64),
        "test": np.asarray(outer.split_indices["test"], dtype=np.int64),
    }


def _fit_inner_predictor(
    args: argparse.Namespace,
    protocol: Any,
    outer: OuterFoldContext,
    *,
    phase: str,
    nbm_seed: int,
    inner_fold: Any,
    device: torch.device,
) -> InnerPredictorArtifact:
    """Train/resume one inner NBM and cache three physical forecasts."""

    nbm_runner, _ = _legacy_runners()
    inner_train = tuple(str(value) for value in inner_fold.train_subjects)
    heldout = tuple(str(value) for value in inner_fold.heldout_subjects)
    if (set(inner_train) | set(heldout)) != set(outer.train_subjects):
        raise ValueError("Inner split does not partition the outer training subjects")
    if set(inner_train) & (set(heldout) | {outer.val_subject, outer.subject}):
        raise ValueError("Inner predictor sees a forbidden subject")

    inner_scaler = protocol.dataset.fit_scaler(
        inner_train,
        clip=float(outer.scaler.clip),
    )
    temporal = temporal_clean_normal_split(
        protocol.dataset,
        protocol.master_windows,
        inner_train,
        validation_fraction=float(_arg(args, "phase3_normal_validation_fraction", 0.2)),
    )
    maximum = int(_arg(args, "phase3_max_normal_windows", 30_000))
    train_index = np.asarray(temporal["train_window_index"], dtype=np.int64)
    if maximum > 0 and len(train_index) > maximum:
        # This cap is applied only after the leakage-safe temporal split.  The
        # validation side and its embargo never become training rows.
        train_index = _subject_stratified_subsample(
            protocol.dataset,
            protocol.master_windows,
            train_index,
            inner_train,
            maximum,
            int(nbm_seed) + 10_000 * int(inner_fold.fold_index),
        )
        temporal = {**temporal, "train_window_index": train_index}
    validation_index = np.asarray(
        temporal["validation_window_index"], dtype=np.int64
    )
    if not np.all(protocol.master_windows.clean_normal[train_index]) or not np.all(
        protocol.master_windows.clean_normal[validation_index]
    ):
        raise AssertionError("Inner NBM received a non-clean-normal row")

    inner_protocol = _inner_protocol_fingerprint(
        args,
        protocol,
        phase=phase,
        outer=outer,
        nbm_seed=nbm_seed,
        inner_fold=inner_fold,
        temporal_split=temporal,
        scaler=inner_scaler,
    )
    predictor_id = (
        f"phase{phase}/outer_{outer.subject}/nbm_seed_{nbm_seed}/"
        f"inner_{int(inner_fold.fold_index):02d}"
    )
    inner_root = (
        Path(args.output_dir)
        / f"phase{phase}"
        / f"loso_{outer.subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / "inner_models"
        / f"inner_{int(inner_fold.fold_index):02d}"
    )
    nbm_root = inner_root / "gru"
    nbm_stage_done = nbm_root / "nbm" / "DONE.json"
    if bool(_arg(args, "finalize_only", False)) and not nbm_stage_done.exists():
        raise FileNotFoundError(nbm_stage_done)
    execution_seed = (
        int(nbm_seed)
        + 100 * tuple(protocol.config["subjects"]).index(outer.subject)
        + int(inner_fold.fold_index)
    )
    nbm_args = _nbm_namespace(args, protocol)
    model, training, checkpoint_sha = nbm_runner.train_nbm_resumable(
        nbm_args,
        "gru",
        nbm_root,
        inner_protocol,
        execution_seed,
        protocol.dataset,
        protocol.master_windows,
        train_index,
        validation_index,
        inner_scaler,
        CONTEXT_SAMPLES,
        HORIZON_SAMPLES,
        device,
    )
    forecast_indices = _inner_forecast_indices(protocol, outer, heldout)
    common_provenance = {
        "phase3_version": PHASE3_VERSION,
        "phase": phase,
        "scheme": _phase_scheme(phase),
        "outer_test_subject": outer.subject,
        "outer_validation_subject": outer.val_subject,
        "outer_train_subjects": list(outer.train_subjects),
        "predictor_id": predictor_id,
        "predictor_train_subjects": list(inner_train),
        "scaler_fit_subjects": list(inner_train),
        "heldout_subjects": list(heldout),
        "nbm_seed": int(nbm_seed),
        "execution_seed": int(execution_seed),
        "inner_protocol_fingerprint": inner_protocol,
        "checkpoint_sha256": checkpoint_sha,
        "scaler": inner_scaler.as_dict(),
        "normal_train_windows": int(len(train_index)),
        "normal_validation_windows": int(len(validation_index)),
        "normal_train_window_sha256": _array_sha256(train_index),
        "normal_validation_window_sha256": _array_sha256(validation_index),
        "temporal_split": {
            key: value
            for key, value in temporal.items()
            if key not in {"train_window_index", "validation_window_index"}
        },
        "training": training,
    }
    forecasts: dict[str, dict[str, Any]] = {}
    for split, indices in forecast_indices.items():
        forecast_root = inner_root / "forecasts" / split
        task_id = f"{predictor_id}/forecast/{split}"
        cached = _load_forecast(
            forecast_root,
            protocol_fingerprint=inner_protocol,
            task_id=task_id,
            upstream_sha256=checkpoint_sha,
        )
        if cached is None:
            if bool(_arg(args, "finalize_only", False)):
                raise FileNotFoundError(forecast_root / "DONE.json")
            cached = extract_gaussian_forecasts(
                model,
                protocol.dataset,
                protocol.master_windows,
                indices,
                inner_scaler,
                batch_size=int(_arg(args, "batch_size", 256)),
                device=device,
                amp=bool(_arg(args, "amp", True)),
                predictor_id=predictor_id,
                predictor_train_subjects=inner_train,
                scaler_fit_subjects=inner_train,
                heldout_subjects=heldout,
            )
            split_provenance = {
                **common_provenance,
                "forecast_split": split,
                "forecast_windows": int(len(indices)),
                "forecast_window_sha256": _array_sha256(indices),
                "forecast_units": "physical_imu",
            }
            _save_forecast(
                forecast_root,
                cached,
                protocol_fingerprint=inner_protocol,
                task_id=task_id,
                upstream_sha256=checkpoint_sha,
                provenance=split_provenance,
                compressed=bool(_arg(args, "cache_compressed", True)),
            )
            cached["provenance"] = split_provenance
        forecasts[split] = cached

    atomic_json_dump(common_provenance, inner_root / "inner_provenance.json")
    return InnerPredictorArtifact(
        predictor_id=predictor_id,
        inner_fold_index=int(inner_fold.fold_index),
        train_subjects=inner_train,
        heldout_subjects=heldout,
        scaler=inner_scaler,
        checkpoint_sha256=checkpoint_sha,
        heldout_forecast=forecasts["heldout"],
        validation_forecast=forecasts["validation"],
        test_forecast=forecasts["test"],
        provenance=common_provenance,
    )


def _ensemble_variance_diagnostics(
    forecasts: Sequence[Mapping[str, Any]], expected_indices: np.ndarray
) -> dict[str, Any]:
    order = np.asarray(expected_indices, dtype=np.int64)
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for forecast in forecasts:
        indices = np.asarray(forecast["window_index"], dtype=np.int64)
        lookup = {int(value): row for row, value in enumerate(indices)}
        rows = np.asarray([lookup[int(value)] for value in order], dtype=np.int64)
        means.append(np.asarray(forecast["mu"], dtype=np.float64)[rows])
        variances.append(np.square(np.asarray(forecast["sigma"], dtype=np.float64)[rows]))
    stacked_mean = np.stack(means)
    aleatoric = np.mean(np.stack(variances), axis=0)
    between = np.var(stacked_mean, axis=0, ddof=0)
    total = aleatoric + between
    ratio = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
    return {
        "ensemble_size": len(forecasts),
        "aleatoric_variance_mean": float(aleatoric.mean()),
        "between_model_variance_mean": float(between.mean()),
        "total_variance_mean": float(total.mean()),
        "between_fraction_mean": float(ratio.mean()),
        "between_fraction_median": float(np.median(ratio)),
        "per_channel_aleatoric_variance_mean": aleatoric.mean(axis=(0, 2)).tolist(),
        "per_channel_between_model_variance_mean": between.mean(axis=(0, 2)).tolist(),
        "per_channel_total_variance_mean": total.mean(axis=(0, 2)).tolist(),
    }


def _primitive_summary(primitives: Mapping[str, Any]) -> dict[str, float]:
    labels = np.asarray(primitives["y"], dtype=np.int8)
    mask = labels == 0
    if not np.any(mask):
        raise ValueError("Representation audit needs non-FOG rows")
    z = np.asarray(primitives["z"], dtype=np.float64)[mask]
    log_sigma = np.asarray(primitives["log_sigma"], dtype=np.float64)[mask]
    diagnostics = dict(primitives["diagnostics"])
    if "error" in primitives and "sigma" in primitives:
        unbounded_z = np.asarray(primitives["error"], dtype=np.float64)[mask] / np.asarray(
            primitives["sigma"], dtype=np.float64
        )[mask]
        z_clip_rate = float((np.abs(unbounded_z) > Z_CLIP).mean())
    else:
        # Kept for small synthetic audit fixtures; production caches always
        # provide error and sigma, so their non-FOG-only rate is used.
        z_clip_rate = float(diagnostics["z_clip_rate"])
    return {
        "nonfog_windows": int(mask.sum()),
        "z_std": float(z.std(ddof=0)),
        "median_log_sigma": float(np.median(log_sigma)),
        "z_clip_rate": z_clip_rate,
        "raw_clip_rate": float(diagnostics["raw_clip_rate"]),
    }


def representation_continuity_audit(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    *,
    z_std_ratio_bounds: tuple[float, float] = (0.5, 2.0),
    max_median_log_sigma_shift: float = math.log(2.0),
    max_z_clip_rate_difference: float = 0.05,
) -> dict[str, Any]:
    """Hard gate the OOF-single vs ensemble representation discontinuity."""

    low, high = (float(value) for value in z_std_ratio_bounds)
    if not 0 < low <= 1 <= high:
        raise ValueError("z_std_ratio_bounds must straddle one")
    summaries = {
        "train_oof": _primitive_summary(train),
        "validation_ensemble": _primitive_summary(validation),
        "test_ensemble": _primitive_summary(test),
    }
    checks: list[dict[str, Any]] = []
    reference = summaries["train_oof"]
    for split in ("validation_ensemble", "test_ensemble"):
        current = summaries[split]
        ratio = current["z_std"] / max(reference["z_std"], 1e-12)
        shift = abs(current["median_log_sigma"] - reference["median_log_sigma"])
        clip_difference = abs(current["z_clip_rate"] - reference["z_clip_rate"])
        checks.append(
            {
                "split": split,
                "z_std_ratio": float(ratio),
                "z_std_ratio_pass": bool(low <= ratio <= high),
                "median_log_sigma_absolute_shift": float(shift),
                "median_log_sigma_shift_pass": bool(
                    shift <= float(max_median_log_sigma_shift)
                ),
                "z_clip_rate_absolute_difference": float(clip_difference),
                "z_clip_rate_difference_pass": bool(
                    clip_difference <= float(max_z_clip_rate_difference)
                ),
            }
        )
    passed = all(
        check[key]
        for check in checks
        for key in (
            "z_std_ratio_pass",
            "median_log_sigma_shift_pass",
            "z_clip_rate_difference_pass",
        )
    )
    return {
        "status": "pass" if passed else "fail",
        "rationale": (
            "Audit the train OOF single-predictor versus validation/test "
            "moment-matched ensemble representation boundary on non-FOG rows."
        ),
        "thresholds": {
            "z_std_ratio_bounds": [low, high],
            "max_median_log_sigma_absolute_shift": float(max_median_log_sigma_shift),
            "max_z_clip_rate_absolute_difference": float(max_z_clip_rate_difference),
        },
        "summaries": summaries,
        "checks": checks,
    }


def _representation_thresholds(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "z_std_ratio_bounds": [
            float(_arg(args, "phase3_min_z_std_ratio", 0.5)),
            float(_arg(args, "phase3_max_z_std_ratio", 2.0)),
        ],
        "max_median_log_sigma_absolute_shift": float(
            _arg(args, "phase3_max_log_sigma_shift", math.log(2.0))
        ),
        "max_z_clip_rate_absolute_difference": float(
            _arg(args, "phase3_max_z_clip_rate_difference", 0.05)
        ),
    }


def assemble_phase3_primitives(
    protocol: Any,
    outer: OuterFoldContext,
    inner_artifacts: Sequence[InnerPredictorArtifact],
    *,
    phase: str,
    nbm_seed: int,
    args: argparse.Namespace | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Assemble OOF/ensemble physical forecasts and outer-scaled primitives."""

    scheme = _phase_scheme(phase)
    expected_models = 3 if scheme == "3fold" else 6
    if len(inner_artifacts) != expected_models:
        raise ValueError(f"{scheme} requires {expected_models} inner predictors")
    heldout_forecasts = [item.heldout_forecast for item in inner_artifacts]
    oof = assemble_oof_gaussians(
        heldout_forecasts,
        protocol.dataset,
        protocol.master_windows,
        outer.split_indices["train"],
        outer_train_subjects=outer.train_subjects,
        validation_subjects=(outer.val_subject,),
        test_subjects=(outer.subject,),
        scheme=scheme,
    )
    validation_forecasts = [item.validation_forecast for item in inner_artifacts]
    test_forecasts = [item.test_forecast for item in inner_artifacts]
    validation = ensemble_gaussians(
        validation_forecasts,
        expected_window_indices=outer.split_indices["validation"],
    )
    test = ensemble_gaussians(
        test_forecasts,
        expected_window_indices=outer.split_indices["test"],
    )
    primitives = {
        "train": convert_to_outer_scaler_primitives(oof, outer.scaler, z_clip=Z_CLIP),
        "validation": convert_to_outer_scaler_primitives(
            validation, outer.scaler, z_clip=Z_CLIP
        ),
        "test": convert_to_outer_scaler_primitives(test, outer.scaler, z_clip=Z_CLIP),
    }
    for split in ("train", "validation", "test"):
        expected_index = np.asarray(outer.split_indices[split], dtype=np.int64)
        if not np.array_equal(primitives[split]["window_index"], expected_index):
            raise AssertionError(f"Phase 3 {outer.subject}/{split} row order changed")
        expected_y = protocol.classification_windows.label[expected_index]
        if not np.array_equal(primitives[split]["y"], expected_y):
            raise AssertionError(f"Phase 3 {outer.subject}/{split} labels changed")

    runtime_args = args or SimpleNamespace()
    thresholds = _representation_thresholds(runtime_args)
    audit = representation_continuity_audit(
        primitives["train"],
        primitives["validation"],
        primitives["test"],
        z_std_ratio_bounds=(
            float(thresholds["z_std_ratio_bounds"][0]),
            float(thresholds["z_std_ratio_bounds"][1]),
        ),
        max_median_log_sigma_shift=float(
            thresholds["max_median_log_sigma_absolute_shift"]
        ),
        max_z_clip_rate_difference=float(
            thresholds["max_z_clip_rate_absolute_difference"]
        ),
    )
    oof_sigma = np.asarray(oof["sigma"], dtype=np.float64)
    provenance = {
        "phase3_version": PHASE3_VERSION,
        "phase": phase,
        "scheme": scheme,
        "outer_test_subject": outer.subject,
        "outer_validation_subject": outer.val_subject,
        "outer_train_subjects": list(outer.train_subjects),
        "nbm_seed": int(nbm_seed),
        "inner_models": [item.provenance for item in inner_artifacts],
        "inner_checkpoint_sha256": [
            item.checkpoint_sha256 for item in inner_artifacts
        ],
        "oof_provenance_audit": oof["provenance_audit"],
        "forecast_units_before_assembly": "physical_imu",
        "ensemble_method": (
            "Gaussian total-variance moment matching: mean(mu), "
            "mean(sigma^2 + mu^2) - mean(mu)^2"
        ),
        "outer_scaler": outer.scaler.as_dict(),
        "variance_diagnostics": {
            "train_oof": {
                "ensemble_size_per_row": 1,
                "aleatoric_variance_mean": float(np.square(oof_sigma).mean()),
                "between_model_variance_mean": 0.0,
                "total_variance_mean": float(np.square(oof_sigma).mean()),
                "between_fraction_mean": 0.0,
            },
            "validation_ensemble": _ensemble_variance_diagnostics(
                validation_forecasts, outer.split_indices["validation"]
            ),
            "test_ensemble": _ensemble_variance_diagnostics(
                test_forecasts, outer.split_indices["test"]
            ),
        },
        "representation_continuity_audit": audit,
        "primitive_diagnostics": {
            split: primitives[split]["diagnostics"]
            for split in ("train", "validation", "test")
        },
    }
    return primitives, provenance


def _primitive_array_payload(primitives: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(primitives[key])
        for key in (
            "raw",
            "mu",
            "sigma",
            "error",
            "z",
            "log_sigma",
            "y",
            "window_index",
        )
    }


def _load_crossfit_cache(root: Path, protocol_fingerprint: str, task_id: str) -> tuple[
    dict[str, dict[str, Any]], dict[str, Any], str
] | None:
    done_path = root / "DONE.json"
    complete = validate_done(
        done_path,
        stage="h200_phase3_crossfit",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
    )
    if complete is None:
        return None
    primitives: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        with np.load(root / f"{split}_primitives.npz", allow_pickle=False) as source:
            primitives[split] = {key: np.asarray(source[key]) for key in source.files}
    provenance = _read_json(root / "provenance.json")
    for split in primitives:
        primitives[split]["diagnostics"] = provenance["primitive_diagnostics"][split]
    return primitives, provenance, sha256_file(done_path)


def ensure_crossfit_cache(
    args: argparse.Namespace,
    protocol: Any,
    outer: OuterFoldContext,
    *,
    phase: str,
    nbm_seed: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    """Create/resume one outer-fold cross-fitted primitive cache."""

    root = (
        Path(args.output_dir)
        / f"phase{phase}"
        / f"loso_{outer.subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / "crossfit"
    )
    task_id = f"phase{phase}/loso_{outer.subject}/nbm_seed_{nbm_seed}/crossfit"
    protocol_fingerprint = str(protocol.config["protocol_fingerprint"])
    cached = _load_crossfit_cache(root, protocol_fingerprint, task_id)
    if cached is not None:
        observed_thresholds = cached[1]["representation_continuity_audit"][
            "thresholds"
        ]
        expected_thresholds = _representation_thresholds(args)
        if observed_thresholds != expected_thresholds:
            raise ValueError(
                "Cached Phase 3 representation thresholds differ; use a new output directory"
            )
        return cached
    if bool(_arg(args, "finalize_only", False)):
        raise FileNotFoundError(root / "DONE.json")

    plan = build_subject_crossfit_plan(
        outer.train_subjects, scheme=_phase_scheme(phase)
    )
    inner_artifacts: list[InnerPredictorArtifact] = []
    for inner_fold in plan.folds:
        inner_artifacts.append(
            _fit_inner_predictor(
                args,
                protocol,
                outer,
                phase=phase,
                nbm_seed=int(nbm_seed),
                inner_fold=inner_fold,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    primitives, provenance = assemble_phase3_primitives(
        protocol,
        outer,
        inner_artifacts,
        phase=phase,
        nbm_seed=int(nbm_seed),
        args=args,
    )
    provenance = {**provenance, "crossfit_plan": plan.as_dict()}
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}_primitives.npz"
        atomic_npz_save(
            path,
            compressed=bool(_arg(args, "cache_compressed", True)),
            **_primitive_array_payload(primitives[split]),
        )
        artifacts[f"{split}_primitives"] = path
    provenance_path = root / "provenance.json"
    gate_path = root / "representation_gate.json"
    atomic_json_dump(provenance, provenance_path)
    atomic_json_dump(provenance["representation_continuity_audit"], gate_path)
    artifacts.update({"provenance": provenance_path, "representation_gate": gate_path})
    inner_fingerprint = canonical_fingerprint(
        {
            "checkpoint_sha256": provenance["inner_checkpoint_sha256"],
            "plan": provenance["crossfit_plan"],
        }
    )
    atomic_json_dump(
        done_payload(
            stage="h200_phase3_crossfit",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=inner_fingerprint,
            relative_to=root,
            artifacts=artifacts,
        ),
        root / "DONE.json",
    )
    return primitives, provenance, sha256_file(root / "DONE.json")


def _history_rows(window_index: np.ndarray, histories: np.ndarray) -> np.ndarray:
    lookup = {
        int(value): row
        for row, value in enumerate(np.asarray(window_index, dtype=np.int64))
    }
    try:
        return np.asarray(
            [[lookup[int(value)] for value in chain] for chain in histories],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("H200 history references a row absent from its cache") from error


def _two_block_history(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[rows]
    if selected.shape[1:] != (2, 9, HORIZON_SAMPLES):
        raise ValueError(f"Unexpected two-block history shape: {selected.shape}")
    return np.ascontiguousarray(
        selected.transpose(0, 2, 1, 3).reshape(len(rows), 9, HISTORY_SAMPLES),
        dtype=np.float32,
    )


def _raw6(
    protocol: Any, outer: OuterFoldContext, anchors: np.ndarray
) -> np.ndarray:
    result = np.empty((len(anchors), 9, RAW6_SAMPLES), dtype=np.float32)
    windows = protocol.classification_windows
    for row, index in enumerate(np.asarray(anchors, dtype=np.int64)):
        record_index = int(windows.record_index[index])
        end = int(windows.target_end[index])
        start = end - RAW6_SAMPLES
        record = protocol.dataset.records[record_index]
        if start < 0 or not bool(record.valid[start:end].all()):
            raise ValueError(f"Raw6 support is invalid at anchor {index}")
        result[row] = outer.scaler.transform(record.x[start:end]).T
    return result


def materialize_crossfit_classifier_base(
    protocol: Any,
    outer: OuterFoldContext,
    primitives: Mapping[str, Any],
    split: str,
) -> dict[str, np.ndarray]:
    """Create aligned Raw4/Raw6/normality arrays for one outer split."""

    histories = np.asarray(
        outer.support[f"{split}_h200_history_window_index"], dtype=np.int64
    )
    rows = _history_rows(
        np.asarray(primitives["window_index"], dtype=np.int64), histories
    )
    anchors = np.asarray(
        outer.support[f"{split}_anchor_window_index"], dtype=np.int64
    )
    labels = np.asarray(outer.support[f"{split}_y"], dtype=np.int8)
    raw4 = _two_block_history(np.asarray(primitives["raw"]), rows)
    z4 = _two_block_history(np.asarray(primitives["z"]), rows)
    log_sigma4 = _two_block_history(np.asarray(primitives["log_sigma"]), rows)
    raw6 = _raw6(protocol, outer, anchors)
    if not (len(raw4) == len(raw6) == len(z4) == len(log_sigma4) == len(labels)):
        raise AssertionError("Phase 3 classifier endpoints are not aligned")
    return {
        "raw4": raw4,
        "raw6": raw6,
        "z4": z4,
        "log_sigma4": log_sigma4,
        "y": labels,
        "window_index": anchors,
    }


def _selected_rows(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rows = np.arange(len(labels), dtype=np.int64)
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels, dtype=np.int8)
    classes, counts = np.unique(labels, return_counts=True)
    allocation = np.maximum(1, np.floor(maximum * counts / counts.sum()).astype(int))
    while int(allocation.sum()) > maximum:
        eligible = np.flatnonzero(allocation > 1)
        allocation[eligible[np.argmax(allocation[eligible])]] -= 1
    while int(allocation.sum()) < maximum:
        room = counts - allocation
        allocation[int(np.argmax(room))] += 1
    selected = [
        rng.choice(rows[labels == label], size=int(count), replace=False)
        for label, count in zip(classes, allocation)
    ]
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def prepare_phase3_arm_inputs(
    base: Mapping[str, np.ndarray],
    arm: str,
    rows: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm not in PHASE3_ARMS:
        raise ValueError(f"Phase 3 does not include arm {arm!r}")
    selected = np.asarray(rows, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for start in range(0, len(selected), int(chunk_size)):
        chunk = selected[start : start + int(chunk_size)]
        built = build_arm_inputs(
            np.ascontiguousarray(base["raw4"][chunk], dtype=np.float32),
            np.ascontiguousarray(base["raw6"][chunk], dtype=np.float32),
            np.ascontiguousarray(base["z4"][chunk], dtype=np.float32),
            np.ascontiguousarray(base["log_sigma4"][chunk], dtype=np.float32),
        )
        chunks.append(built[arm])
    if not chunks:
        raise ValueError("Phase 3 classifier selection is empty")
    return (
        np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32),
        np.asarray(base["y"], dtype=np.int8)[selected],
        np.asarray(base["window_index"], dtype=np.int64)[selected],
    )


def _classifier_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )


def _classifier_epoch(
    model: nn.Module,
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
                device.type, enabled=bool(amp) and device.type == "cuda"
            ):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        total_loss += float(loss.detach()) * int(y.numel())
        total_n += int(y.numel())
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
    if total_n == 0:
        raise RuntimeError("Classifier loader is empty")
    return total_loss / total_n, np.concatenate(truths), np.concatenate(probabilities)


def _enrich_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    tn, fp, fn, tp = (int(result[key]) for key in ("tn", "fp", "fn", "tp"))
    nonfog_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    fog_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result.update(
        {
            "macro_f1": 0.5 * (nonfog_f1 + fog_f1),
            "roc_auc": result.get("auroc"),
            "pr_auc": result.get("auprc"),
            "fog_recall": result.get("sensitivity"),
            "fog_f1": fog_f1,
        }
    )
    return result


def _train_phase3_classifier(
    args: argparse.Namespace,
    protocol: Any,
    outer: OuterFoldContext,
    *,
    phase: str,
    nbm_seed: int,
    classifier_seed: int,
    arm: str,
    split_inputs: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    upstream_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    nbm_runner, rf_runner = _legacy_runners()
    root = (
        Path(args.output_dir)
        / f"phase{phase}"
        / f"loso_{outer.subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / f"classifier_seed_{int(classifier_seed)}"
        / arm
    )
    root.mkdir(parents=True, exist_ok=True)
    task_id = (
        f"phase{phase}/loso_{outer.subject}/nbm_seed_{nbm_seed}/"
        f"classifier_seed_{classifier_seed}/{arm}"
    )
    protocol_fingerprint = str(protocol.config["protocol_fingerprint"])
    metrics_path = root / "metrics.json"
    complete = validate_done(
        root / "DONE.json",
        stage="h200_phase3_classifier",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
        upstream_sha256=upstream_sha256,
    )
    if complete is not None:
        return _read_json(metrics_path)
    if bool(_arg(args, "finalize_only", False)):
        raise FileNotFoundError(root / "DONE.json")

    nbm_runner.set_seed(int(classifier_seed), bool(_arg(args, "deterministic", True)))
    model = build_classifier(
        arm,
        hidden_channels=int(_arg(args, "classifier_hidden", 48)),
        dropout=float(_arg(args, "classifier_dropout", 0.15)),
    ).to(device)
    initial_hash = rf_runner.state_dict_sha256(model.state_dict())
    architecture = model.architecture_config()
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())

    x_train, y_train, _ = split_inputs["train"]
    x_validation, y_validation, validation_index = split_inputs["validation"]
    x_test, y_test, test_index = split_inputs["test"]
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if np.min(counts) <= 0:
        raise RuntimeError(f"Classifier train split lacks a class: {task_id}")
    pos_weight = min(math.sqrt(counts[0] / counts[1]), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_arg(args, "classifier_lr", 1e-3)),
        weight_decay=float(_arg(args, "weight_decay", 1e-4)),
    )
    amp = bool(_arg(args, "amp", True))
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and device.type == "cuda"
    )
    batch_size = int(_arg(args, "batch_size", 256))
    num_workers = int(_arg(args, "num_workers", 0))
    pin = device.type == "cuda"
    validation_loader = _classifier_loader(
        x_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
        seed=classifier_seed,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = _classifier_loader(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
        seed=classifier_seed,
        num_workers=num_workers,
        pin_memory=pin,
    )
    epochs = int(_arg(args, "classifier_epochs", 12))
    patience = min(int(_arg(args, "classifier_patience", 4)), epochs)
    best_path = root / "classifier_best.pt"
    last_path = root / "classifier_last.pt"
    start_epoch = 0
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if bool(_arg(args, "resume", True)) and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        expected_identity = {
            "phase3_version": PHASE3_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "task_id": task_id,
            "upstream_sha256": upstream_sha256,
            "classifier_seed": int(classifier_seed),
            "arm": arm,
            "initial_state_sha256": initial_hash,
        }
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                raise ValueError(f"Classifier resume identity differs: {task_id}/{key}")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        grad_scaler.load_state_dict(payload["grad_scaler_state"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        restore_rng_state(payload["rng_state"])
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, epochs + 1):
        if bad_epochs >= patience:
            break
        train_loader = _classifier_loader(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=int(classifier_seed) + epoch,
            num_workers=num_workers,
            pin_memory=pin,
        )
        train_loss, train_true, train_prob = _classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            validation_loss, validation_true, validation_prob = _classifier_epoch(
                model, validation_loader, criterion, device, amp
            )
        validation_pr_auc = float(
            average_precision_score(validation_true, validation_prob)
        )
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": int(classifier_seed) + epoch,
                "train_loss": float(train_loss),
                "train_pr_auc": float(average_precision_score(train_true, train_prob)),
                "validation_loss": float(validation_loss),
                "validation_pr_auc": validation_pr_auc,
            }
        )
        if validation_pr_auc > best_score + 1e-5:
            best_score = validation_pr_auc
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "phase3_version": PHASE3_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "task_id": task_id,
                    "upstream_sha256": upstream_sha256,
                    "classifier_seed": int(classifier_seed),
                    "arm": arm,
                    "initial_state_sha256": initial_hash,
                    "model_state": model.state_dict(),
                    "architecture": architecture,
                    "best_epoch": best_epoch,
                    "best_validation_pr_auc": best_score,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        atomic_torch_save(
            {
                "phase3_version": PHASE3_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "task_id": task_id,
                "upstream_sha256": upstream_sha256,
                "classifier_seed": int(classifier_seed),
                "arm": arm,
                "initial_state_sha256": initial_hash,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "grad_scaler_state": grad_scaler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_sec": elapsed_before + time.perf_counter() - started,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
    if not best_path.exists():
        raise RuntimeError(f"Classifier did not produce a best checkpoint: {task_id}")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    with torch.no_grad():
        _, validation_true, validation_prob = _classifier_epoch(
            model, validation_loader, criterion, device, amp
        )
        _, test_true, test_prob = _classifier_epoch(
            model, test_loader, criterion, device, amp
        )
    threshold, validation_metrics = choose_threshold(validation_true, validation_prob)
    metrics = _enrich_metrics(binary_metrics(test_true, test_prob, threshold))
    test_pred = (np.asarray(test_prob) >= float(threshold)).astype(np.int8)
    metrics.update(
        nbm_runner.event_metrics(
            protocol.dataset,
            protocol.classification_windows,
            test_index,
            test_pred,
        )
    )
    metrics.update(
        {
            "phase3_version": PHASE3_VERSION,
            "phase": phase,
            "arm": arm,
            "display_name": H200_ARM_REGISTRY[arm].display_name,
            "test_subject": outer.subject,
            "val_subject": outer.val_subject,
            "train_subjects": list(outer.train_subjects),
            "nbm_seed": int(nbm_seed),
            "classifier_seed": int(classifier_seed),
            "parameter_count": int(parameter_count),
            "architecture": architecture,
            "initial_state_sha256": initial_hash,
            "best_epoch": int(best["best_epoch"]),
            "best_validation_pr_auc": float(best["best_validation_pr_auc"]),
            "validation": validation_metrics,
            "train_counts": counts.astype(int).tolist(),
            "pos_weight": float(pos_weight),
            "history": history,
            "endpoint_sha256": _array_sha256(test_index),
            "label_sha256": _array_sha256(test_true),
            "crossfit_cache_sha256": upstream_sha256,
        }
    )
    prediction_path = root / "predictions.npz"
    validation_path = root / "validation_predictions.npz"
    prediction_csv_path = root / "predictions.csv"
    atomic_npz_save(
        prediction_path,
        window_index=test_index,
        y_true=np.asarray(test_true, dtype=np.int8),
        y_prob=np.asarray(test_prob, dtype=np.float64),
        y_pred=test_pred,
    )
    validation_pred = (
        np.asarray(validation_prob) >= float(threshold)
    ).astype(np.int8)
    atomic_npz_save(
        validation_path,
        window_index=validation_index,
        y_true=np.asarray(validation_true, dtype=np.int8),
        y_prob=np.asarray(validation_prob, dtype=np.float64),
        y_pred=validation_pred,
    )
    nbm_runner.write_predictions_csv(
        prediction_csv_path,
        protocol.dataset,
        protocol.classification_windows,
        test_index,
        test_prob,
        test_pred,
    )
    atomic_json_dump(metrics, metrics_path)
    atomic_json_dump(
        done_payload(
            stage="h200_phase3_classifier",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256=upstream_sha256,
            relative_to=root,
            artifacts={
                "best": best_path,
                "last": last_path,
                "metrics": metrics_path,
                "predictions": prediction_path,
                "validation_predictions": validation_path,
                "predictions_csv": prediction_csv_path,
            },
        ),
        root / "DONE.json",
    )
    return metrics


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite_metric(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def evaluate_phase3a_science_gate(
    subject_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_nonreversed_subjects: int = 2,
    max_false_alarm_ratio: float = 1.2,
) -> dict[str, Any]:
    """Evaluate the frozen directional/safety gate before expanding to 3B."""

    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in subject_rows:
        by_arm.setdefault(str(row["arm"]), {})[str(row["test_subject"])] = row
    required_arms = set(PHASE3_ARMS)
    reasons: list[str] = []
    if set(by_arm) != required_arms:
        reasons.append("missing one or more preregistered Phase 3A arms")
    common_subjects = set.intersection(
        *(set(by_arm.get(arm, {})) for arm in PHASE3_ARMS)
    )
    if len(common_subjects) != 3:
        reasons.append("Phase 3A science gate requires exactly three paired subjects")

    comparisons: dict[str, Any] = {}
    fusion = "raw4_normality"
    for reference in ("raw6", "raw4_zero"):
        key = f"{fusion}_minus_{reference}"
        deltas: dict[str, float] = {}
        for subject in sorted(common_subjects):
            candidate_value = _finite_metric(by_arm[fusion][subject], "pr_auc")
            reference_value = _finite_metric(by_arm[reference][subject], "pr_auc")
            if candidate_value is None or reference_value is None:
                reasons.append(f"{key} lacks a finite PR-AUC for {subject}")
                continue
            deltas[subject] = candidate_value - reference_value
        macro_delta = float(np.mean(list(deltas.values()))) if deltas else None
        nonreversed = sum(value >= 0.0 for value in deltas.values())
        direction_pass = bool(macro_delta is not None and macro_delta > 0.0)
        consistency_pass = bool(
            len(deltas) == 3 and nonreversed >= int(minimum_nonreversed_subjects)
        )
        if not direction_pass:
            reasons.append(f"{key} subject-macro PR-AUC direction is not positive")
        if not consistency_pass:
            reasons.append(
                f"{key} is non-reversed in fewer than {minimum_nonreversed_subjects}/3 subjects"
            )
        comparisons[key] = {
            "per_subject_pr_auc_delta": deltas,
            "subject_macro_pr_auc_delta": macro_delta,
            "positive_direction_pass": direction_pass,
            "nonreversed_subjects": int(nonreversed),
            "minimum_nonreversed_subjects": int(minimum_nonreversed_subjects),
            "consistency_pass": consistency_pass,
        }

    fusion_fa = [
        _finite_metric(by_arm.get(fusion, {}).get(subject, {}), "false_alarm_events_per_hour")
        for subject in sorted(common_subjects)
    ]
    raw_fa = [
        _finite_metric(by_arm.get("raw6", {}).get(subject, {}), "false_alarm_events_per_hour")
        for subject in sorted(common_subjects)
    ]
    finite_pairs = [
        (candidate, reference)
        for candidate, reference in zip(fusion_fa, raw_fa)
        if candidate is not None and reference is not None
    ]
    if len(finite_pairs) == 3:
        fusion_fa_macro = float(np.mean([pair[0] for pair in finite_pairs]))
        raw_fa_macro = float(np.mean([pair[1] for pair in finite_pairs]))
        allowed_fa = float(max_false_alarm_ratio) * raw_fa_macro
        false_alarm_pass = fusion_fa_macro <= allowed_fa + 1e-12
        false_alarm_ratio = (
            fusion_fa_macro / raw_fa_macro
            if raw_fa_macro > 0
            else (0.0 if fusion_fa_macro <= 0 else None)
        )
    else:
        fusion_fa_macro = None
        raw_fa_macro = None
        allowed_fa = None
        false_alarm_ratio = None
        false_alarm_pass = False
        reasons.append("Phase 3A false-alarm safety gate lacks three finite pairs")
    if not false_alarm_pass and len(finite_pairs) == 3:
        reasons.append(
            f"C2 subject-macro FA/h exceeds {max_false_alarm_ratio:g} times C0"
        )
    safety = {
        "candidate_arm": fusion,
        "reference_arm": "raw6",
        "candidate_subject_macro_false_alarm_events_per_hour": fusion_fa_macro,
        "reference_subject_macro_false_alarm_events_per_hour": raw_fa_macro,
        "maximum_ratio": float(max_false_alarm_ratio),
        "allowed_candidate_false_alarm_events_per_hour": allowed_fa,
        "observed_ratio": false_alarm_ratio,
        "pass": bool(false_alarm_pass),
    }
    passed = not reasons
    return {
        "status": "pass" if passed else "fail",
        "decision": "expand_to_phase3b" if passed else "stop_before_phase3b",
        "comparisons": comparisons,
        "false_alarm_safety": safety,
        "reasons": reasons,
    }


def aggregate_phase3(
    args: argparse.Namespace,
    protocol: Any,
    *,
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    representation_gate: Mapping[str, Any],
    external_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Average repetitions within subject, then run subject-paired inference."""

    root = Path(args.output_dir) / f"phase{phase}"
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "fog_recall",
        "specificity",
        "precision",
        "fog_f1",
        "mcc",
        "event_sensitivity",
        "false_alarm_events_per_hour",
        "median_detection_delay_sec",
    )
    subject_rows: list[dict[str, Any]] = []
    subjects = tuple(dict.fromkeys(str(row["test_subject"]) for row in rows))
    arms = tuple(dict.fromkeys(str(row["arm"]) for row in rows))
    for subject in subjects:
        for arm in arms:
            repetitions = [
                row
                for row in rows
                if row["test_subject"] == subject and row["arm"] == arm
            ]
            if not repetitions:
                continue
            item: dict[str, Any] = {
                "phase": phase,
                "test_subject": subject,
                "arm": arm,
                "repetitions": len(repetitions),
                "nbm_seeds": sorted({int(row["nbm_seed"]) for row in repetitions}),
                "classifier_seeds": sorted(
                    {int(row["classifier_seed"]) for row in repetitions}
                ),
            }
            for metric in metric_names:
                values = [
                    value
                    for value in (_finite_metric(row, metric) for row in repetitions)
                    if value is not None
                ]
                item[metric] = float(np.mean(values)) if values else None
            subject_rows.append(item)

    aggregate: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [row for row in subject_rows if row["arm"] == arm]
        aggregate[arm] = {}
        for metric in metric_names:
            values = [
                value
                for value in (_finite_metric(row, metric) for row in arm_rows)
                if value is not None
            ]
            if values:
                array = np.asarray(values, dtype=np.float64)
                aggregate[arm][metric] = {
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=0)),
                    "min": float(array.min()),
                    "max": float(array.max()),
                    "n_subjects": int(len(array)),
                }
            else:
                aggregate[arm][metric] = {
                    "mean": None,
                    "std": None,
                    "n_subjects": 0,
                }

    paired: dict[str, Any] = {}
    candidate_arm = "raw4_normality"
    for reference_arm in ("raw4_zero", "raw6"):
        comparison_key = f"{candidate_arm}_minus_{reference_arm}"
        paired[comparison_key] = {}
        for metric in ("macro_f1", "pr_auc", "fog_recall"):
            candidate = {
                row["test_subject"]: float(row[metric])
                for row in subject_rows
                if row["arm"] == candidate_arm and row.get(metric) is not None
            }
            reference = {
                row["test_subject"]: float(row[metric])
                for row in subject_rows
                if row["arm"] == reference_arm and row.get(metric) is not None
            }
            common = tuple(subject for subject in subjects if subject in candidate and subject in reference)
            if not common:
                paired[comparison_key][metric] = None
                continue
            paired[comparison_key][metric] = paired_bootstrap(
                {subject: candidate[subject] for subject in common},
                {subject: reference[subject] for subject in common},
                samples=int(_arg(args, "bootstrap_samples", 100_000)),
                seed=int(_arg(args, "bootstrap_seed", 42)),
            )

    if phase == "3a":
        science_gate = evaluate_phase3a_science_gate(subject_rows)
    else:
        science_gate = {
            "status": "not_applicable",
            "decision": "phase3b_completed",
            "reasons": [],
        }
    representation_pass = representation_gate.get("status") == "pass"
    science_pass = science_gate.get("status") in {"pass", "not_applicable"}
    if phase == "3a":
        external_payload: dict[str, Any] = {
            "subjects": ["S04", "S10"],
            "status": "not_applicable_before_phase3b",
        }
        external_pass = True
    elif external_evaluation is None:
        external_payload = {
            "subjects": ["S04", "S10"],
            "status": "not_executed",
            "reason": "Phase-3B external evaluation was disabled or unavailable",
        }
        external_pass = False
    else:
        external_payload = dict(external_evaluation)
        external_pass = external_payload.get("status") == "complete"
    decision_reasons: list[str] = []
    if not representation_pass:
        decision_reasons.append("representation continuity hard gate failed")
    decision_reasons.extend(str(value) for value in science_gate.get("reasons", []))
    if not external_pass:
        decision_reasons.append(
            "S04/S10 external negative-only evaluation is incomplete"
        )
    decision = {
        "status": (
            "pass"
            if representation_pass and science_pass and external_pass
            else "fail"
        ),
        "expand_to_phase3b": bool(
            phase == "3a" and representation_pass and science_pass
        ),
        "reasons": decision_reasons,
    }

    payload = {
        "phase3_version": PHASE3_VERSION,
        "phase": phase,
        "protocol_fingerprint": protocol.config["protocol_fingerprint"],
        "seed_averaging_policy": (
            "NBM/classifier repetitions are averaged within each test subject; "
            "only subject means enter macro summaries and paired bootstrap."
        ),
        "representation_gate": dict(representation_gate),
        "science_gate": science_gate,
        "decision": decision,
        "external_negative_only_evaluation": external_payload,
        "classifier_cells": len(rows),
        "subject_arm_rows": len(subject_rows),
        "aggregate": aggregate,
        "paired_bootstrap": paired,
    }
    aggregate_path = root / "aggregate.json"
    cells_path = root / "classifier_cells.csv"
    subject_path = root / "subject_seed_averaged_metrics.csv"
    atomic_json_dump(payload, aggregate_path)
    cell_columns = [
        "phase",
        "test_subject",
        "val_subject",
        "arm",
        "nbm_seed",
        "classifier_seed",
        *metric_names,
    ]
    _atomic_csv(cells_path, rows, cell_columns)
    subject_columns = [
        "phase",
        "test_subject",
        "arm",
        "repetitions",
        *metric_names,
    ]
    _atomic_csv(subject_path, subject_rows, subject_columns)
    aggregate_artifacts: dict[str, Path] = {
        "aggregate": aggregate_path,
        "classifier_cells": cells_path,
        "subject_seed_averaged_metrics": subject_path,
        "representation_gate": root / "representation_gate.json",
    }
    external_done = root / "external_negative_only" / "DONE.json"
    if external_evaluation is not None:
        if not external_done.exists():
            raise FileNotFoundError(external_done)
        aggregate_artifacts["external_negative_only_done"] = external_done
    atomic_json_dump(
        done_payload(
            stage="h200_phase3_aggregate",
            protocol_fingerprint=str(protocol.config["protocol_fingerprint"]),
            task_id=f"phase{phase}/aggregate",
            relative_to=root,
            artifacts=aggregate_artifacts,
        ),
        root / "DONE.json",
    )
    return payload


def _run_phase3(
    *,
    args: argparse.Namespace,
    protocol: Any,
    device: torch.device,
    arms: Sequence[str],
    phase: str,
) -> dict[str, Any]:
    resolved_arms = tuple(str(value) for value in arms)
    if set(resolved_arms) != set(PHASE3_ARMS) or len(resolved_arms) != 3:
        raise ValueError(f"Phase 3 requires exactly these arms: {PHASE3_ARMS}")
    outer_subjects = phase3_outer_subjects(protocol, phase)
    seeds = phase3_seed_policy(args, phase)

    caches: dict[tuple[str, int], tuple[
        OuterFoldContext, dict[str, dict[str, Any]], dict[str, Any], str
    ]] = {}
    gate_cells: list[dict[str, Any]] = []
    for subject in outer_subjects:
        outer = load_outer_fold(args, protocol, subject)
        for nbm_seed in seeds["nbm"]:
            primitives, provenance, cache_sha = ensure_crossfit_cache(
                args,
                protocol,
                outer,
                phase=phase,
                nbm_seed=nbm_seed,
                device=device,
            )
            audit = dict(provenance["representation_continuity_audit"])
            gate_cells.append(
                {
                    "test_subject": subject,
                    "nbm_seed": int(nbm_seed),
                    "status": audit["status"],
                    "audit": audit,
                    "variance_diagnostics": provenance["variance_diagnostics"],
                }
            )
            caches[(subject, nbm_seed)] = (
                outer,
                primitives,
                provenance,
                cache_sha,
            )

    gate_passed = all(cell["status"] == "pass" for cell in gate_cells)
    representation_gate = {
        "phase3_version": PHASE3_VERSION,
        "phase": phase,
        "status": "pass" if gate_passed else "fail",
        "hard_gate": True,
        "cells": gate_cells,
    }
    phase_root = Path(args.output_dir) / f"phase{phase}"
    atomic_json_dump(representation_gate, phase_root / "representation_gate.json")
    if not gate_passed and not bool(
        _arg(args, "force_phase3_representation_gate", False)
    ):
        atomic_json_dump(
            {
                "phase3_version": PHASE3_VERSION,
                "phase": phase,
                "status": "gate_failed",
                "reason": "OOF-versus-ensemble representation continuity audit failed",
            },
            phase_root / "status.json",
        )
        raise RuntimeError(
            f"Phase {phase} representation continuity gate failed; "
            f"see {phase_root / 'representation_gate.json'}"
        )
    representation_gate["forced_after_failure"] = bool(
        not gate_passed and _arg(args, "force_phase3_representation_gate", False)
    )
    atomic_json_dump(representation_gate, phase_root / "representation_gate.json")

    metrics_rows: list[dict[str, Any]] = []
    for subject in outer_subjects:
        for nbm_seed in seeds["nbm"]:
            outer, primitives, _, cache_sha = caches[(subject, nbm_seed)]
            bases = {
                split: materialize_crossfit_classifier_base(
                    protocol, outer, primitives[split], split
                )
                for split in ("train", "validation", "test")
            }
            selected = {
                "train": _selected_rows(
                    bases["train"]["y"],
                    int(_arg(args, "max_classifier_windows", 0)),
                    int(nbm_seed) + 1_000 * tuple(protocol.config["subjects"]).index(subject),
                ),
                "validation": np.arange(len(bases["validation"]["y"]), dtype=np.int64),
                "test": np.arange(len(bases["test"]["y"]), dtype=np.int64),
            }
            endpoint_reference = {
                split: bases[split]["window_index"][rows]
                for split, rows in selected.items()
            }
            for classifier_seed in seeds["classifier"]:
                initial_hashes: dict[str, str] = {}
                parameter_counts: dict[str, int] = {}
                for arm in resolved_arms:
                    split_inputs = {
                        split: prepare_phase3_arm_inputs(
                            bases[split], arm, selected[split]
                        )
                        for split in ("train", "validation", "test")
                    }
                    for split in split_inputs:
                        if not np.array_equal(
                            split_inputs[split][2], endpoint_reference[split]
                        ):
                            raise AssertionError(
                                f"Phase {phase}/{subject}/{arm}/{split} endpoints differ"
                            )
                    metrics = _train_phase3_classifier(
                        args,
                        protocol,
                        outer,
                        phase=phase,
                        nbm_seed=nbm_seed,
                        classifier_seed=classifier_seed,
                        arm=arm,
                        split_inputs=split_inputs,
                        upstream_sha256=cache_sha,
                        device=device,
                    )
                    metrics_rows.append(metrics)
                    initial_hashes[arm] = str(metrics["initial_state_sha256"])
                    parameter_counts[arm] = int(metrics["parameter_count"])
                    del split_inputs
                if (
                    initial_hashes["raw4_zero"]
                    != initial_hashes["raw4_normality"]
                    or parameter_counts["raw4_zero"]
                    != parameter_counts["raw4_normality"]
                ):
                    raise AssertionError(
                        "Zero-control and normality-fusion classifiers are not capacity matched"
                    )
            del bases
            if device.type == "cuda":
                torch.cuda.empty_cache()
    external_evaluation: dict[str, Any] | None = None
    if phase == "3b" and bool(
        _arg(args, "phase3_external_negative_only", True)
    ):
        # Local import avoids a module cycle: the external evaluator reuses
        # the frozen Phase-3 model/scaler helpers defined above.
        from .h200_phase3_external import run_phase3b_external

        external_evaluation = run_phase3b_external(
            args=args,
            protocol=protocol,
            device=device,
            arms=resolved_arms,
        )
    result = aggregate_phase3(
        args,
        protocol,
        phase=phase,
        rows=metrics_rows,
        representation_gate=representation_gate,
        external_evaluation=external_evaluation,
    )
    atomic_json_dump(
        {
            "phase3_version": PHASE3_VERSION,
            "phase": phase,
            "status": (
                "complete" if result["decision"]["status"] == "pass"
                else "complete_gate_failed"
            ),
            "decision": result["decision"],
            "outer_subjects": list(outer_subjects),
            "nbm_seeds": list(seeds["nbm"]),
            "classifier_seeds": list(seeds["classifier"]),
        },
        phase_root / "status.json",
    )
    return result


def run_phase3a(
    *,
    args: argparse.Namespace,
    protocol: Any,
    device: torch.device,
    arms: Sequence[str] = PHASE3_ARMS,
) -> dict[str, Any]:
    """Run the fixed S01/S05/S08 three-fold cross-fitting pilot."""

    return _run_phase3(
        args=args, protocol=protocol, device=device, arms=arms, phase="3a"
    )


def run_phase3b(
    *,
    args: argparse.Namespace,
    protocol: Any,
    device: torch.device,
    arms: Sequence[str] = PHASE3_ARMS,
) -> dict[str, Any]:
    """Run all eight outer folds with leave-one-training-subject-out NBMs."""

    return _run_phase3(
        args=args, protocol=protocol, device=device, arms=arms, phase="3b"
    )


__all__ = [
    "DEFAULT_CLASSIFIER_SEEDS",
    "DEFAULT_NBM_SEEDS",
    "InnerPredictorArtifact",
    "OuterFoldContext",
    "PHASE3A_OUTER_SUBJECTS",
    "PHASE3_ARMS",
    "PHASE3_VERSION",
    "aggregate_phase3",
    "assemble_phase3_primitives",
    "ensure_crossfit_cache",
    "evaluate_phase3a_science_gate",
    "load_outer_fold",
    "materialize_crossfit_classifier_base",
    "phase3_outer_subjects",
    "phase3_seed_policy",
    "prepare_phase3_arm_inputs",
    "representation_continuity_audit",
    "run_phase3a",
    "run_phase3b",
]
