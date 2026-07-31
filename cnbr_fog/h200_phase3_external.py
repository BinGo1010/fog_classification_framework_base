"""Negative-only S04/S10 evaluation for the confirmatory H200 Phase 3B.

The eight-subject Phase 3B protocol deliberately excludes S04 and S10 because
they contain no labelled FOG.  This module evaluates those two subjects only
after every inner normal-behaviour model, classifier checkpoint, and
validation-selected threshold has been frozen.  External labels are used only
to assert the negative-only contract and to score false positives.

Model repetitions (outer fold, NBM seed, and classifier seed) are descriptive
repeated estimates.  They are averaged *within* each external subject before
the two subjects are summarised; repetitions are never counted as independent
subjects.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import DaphnetDataset, RobustChannelScaler, WindowTable
from .h200_crossfit import convert_to_outer_scaler_primitives
from .h200_feasibility import build_classifier
from .h200_phase3 import (
    CONTEXT_SAMPLES,
    HISTORY_SAMPLES,
    HORIZON_SAMPLES,
    PHASE3_ARMS,
    PHASE3_VERSION,
    Z_CLIP,
    _arg,
    _legacy_runners,
    _nbm_namespace,
    _read_json,
    _scaler_from_mapping,
    load_outer_fold,
    phase3_outer_subjects,
    phase3_seed_policy,
    prepare_phase3_arm_inputs,
)
from .resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    dataset_fingerprint,
    done_payload,
    sha256_file,
    validate_checkpoint,
    validate_done,
)


EXTERNAL_VERSION = "daphnet_gru_h200_phase3b_external_negative.v1"
EXTERNAL_SUBJECTS = ("S04", "S10")
SAMPLING_RATE_HZ = 64
STRIDE_SAMPLES = 16
LABEL_SAMPLES = 32
RAW6_SAMPLES = 384


@dataclass(frozen=True)
class ExternalSubjectSupport:
    """Exact H200 forecast rows and two-block classifier endpoints."""

    subject: str
    forecast_window_index: np.ndarray
    anchor_window_index: np.ndarray
    history_window_index: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ExternalProtocol:
    """Full ten-subject data view, kept separate from the eight-subject run."""

    fingerprint: str
    dataset: DaphnetDataset
    master_windows: WindowTable
    classification_windows: WindowTable
    support: dict[str, ExternalSubjectSupport]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ForecastCacheRef:
    root: Path
    done_sha256: str
    checkpoint_sha256: str
    predictor_id: str


@dataclass(frozen=True)
class ExternalCellResult:
    metrics: dict[str, Any]
    window_index: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    done_sha256: str


def _legacy_horizon_runner() -> Any:
    try:
        return importlib.import_module("run_daphnet_gru_horizon_ablation")
    except ModuleNotFoundError:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        return importlib.import_module("run_daphnet_gru_horizon_ablation")


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return canonical_fingerprint(
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes_sha256": __import__("hashlib").sha256(
                array.tobytes(order="C")
            ).hexdigest(),
        }
    )


def _resolve_done_artifact(done_path: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    return path if path.is_absolute() else done_path.parent / path


def _atomic_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_external_subjects(args: argparse.Namespace) -> tuple[str, ...]:
    value = _arg(args, "phase3_external_subjects", ",".join(EXTERNAL_SUBJECTS))
    if isinstance(value, str):
        subjects = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        subjects = tuple(str(item) for item in value)
    if subjects != EXTERNAL_SUBJECTS:
        raise ValueError(
            f"The preregistered negative-only subjects are exactly {EXTERNAL_SUBJECTS}"
        )
    return subjects


def _build_external_protocol(
    args: argparse.Namespace,
    protocol: Any,
) -> ExternalProtocol:
    """Reload all records and exactly rebuild terminal-label/common support."""

    horizon = _legacy_horizon_runner()
    source = _read_json(Path(args.source_suite_dir) / "config.json")
    minimum_positive_windows = int(
        _arg(args, "phase3_external_minimum_positive_windows", 2)
    )
    merge_gap_seconds = float(
        _arg(args, "phase3_external_merge_gap_seconds", 0.5)
    )
    if minimum_positive_windows <= 0 or not math.isfinite(merge_gap_seconds) or merge_gap_seconds < 0:
        raise ValueError("External false-alarm event parameters are invalid")
    expected = {
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "context_samples": CONTEXT_SAMPLES,
        "support_horizon_samples": HORIZON_SAMPLES,
        "fixed_label_samples": LABEL_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
    }
    for key, value in expected.items():
        if int(source.get(key, -1)) != value:
            raise ValueError(f"Source {key} changed before external evaluation")
    data_sha = dataset_fingerprint(args.data_dir)
    if data_sha != str(protocol.config["data_sha256"]):
        raise ValueError("External data root differs from the frozen main protocol")

    dataset = DaphnetDataset.load(
        args.data_dir,
        flatline_seconds=float(source["flatline_seconds"]),
        zero_tolerance=float(source["zero_tolerance"]),
    )
    expected_subjects = set(str(item) for item in protocol.config["subjects"]) | set(
        EXTERNAL_SUBJECTS
    )
    if set(dataset.subjects) != expected_subjects:
        raise ValueError(
            f"Full processed data subjects differ: {dataset.subjects} != "
            f"{sorted(expected_subjects)}"
        )
    if dataset.sampling_rate_hz != SAMPLING_RATE_HZ or dataset.n_channels != 9:
        raise ValueError("External evaluation requires 64-Hz, nine-channel Daphnet")
    if tuple(dataset.channel_names) != tuple(protocol.config["channel_names"]):
        raise ValueError("External channel order differs from the main protocol")
    for subject in EXTERNAL_SUBJECTS:
        records = [record for record in dataset.records if record.subject_id == subject]
        if not records or any(np.any(record.y != 0) for record in records):
            raise ValueError(f"{subject} is not a strictly negative-only subject")

    raw_master = dataset.make_windows(
        warmup_samples=CONTEXT_SAMPLES,
        target_samples=HORIZON_SAMPLES,
        stride_samples=STRIDE_SAMPLES,
        fog_fraction_threshold=float(source["fog_fraction_threshold"]),
        normal_guard_samples=int(source["normal_guard_samples"]),
    )
    master = horizon.relabel_master_windows(
        dataset,
        raw_master,
        LABEL_SAMPLES,
        float(source["fog_fraction_threshold"]),
    )
    classification = horizon.derive_classification_windows(master)
    windows_by_horizon = {
        str(item["horizon_id"]): horizon.derive_horizon_windows(
            master, int(item["horizon_samples"]), LABEL_SAMPLES
        )
        for item in horizon.HORIZON_DEFINITIONS
    }
    if "H200" not in windows_by_horizon:
        raise ValueError("Frozen horizon definitions no longer contain H200")

    supports: dict[str, ExternalSubjectSupport] = {}
    support_hashes: dict[str, Any] = {}
    for subject in EXTERNAL_SUBJECTS:
        indices = dataset.window_indices_for_subjects(classification, (subject,))
        if len(indices) == 0:
            raise RuntimeError(f"No valid H200 windows for {subject}")
        plans = horizon.build_common_history_support(
            windows_by_horizon,
            {"external": indices},
            HISTORY_SAMPLES,
            STRIDE_SAMPLES,
        )
        reference_anchor: np.ndarray | None = None
        for horizon_id, split_plans in plans.items():
            anchor = np.asarray(
                split_plans["external"].anchor_window_indices, dtype=np.int64
            )
            if reference_anchor is None:
                reference_anchor = anchor
            elif not np.array_equal(anchor, reference_anchor):
                raise AssertionError(
                    f"{subject}/{horizon_id} common endpoints are misaligned"
                )
        assert reference_anchor is not None
        h200_plan = plans["H200"]["external"]
        history = np.asarray(indices[h200_plan.max_chain_rows], dtype=np.int64)
        anchors = np.asarray(h200_plan.anchor_window_indices, dtype=np.int64)
        labels = np.asarray(classification.label[anchors], dtype=np.int8)
        if history.shape != (len(anchors), 2):
            raise ValueError(f"{subject} does not have exact two-block H200 support")
        if not np.array_equal(anchors, reference_anchor) or np.any(labels != 0):
            raise ValueError(f"{subject} external endpoints violate negative-only support")
        # Raw6 ends at the same endpoint and starts at the first H200 context.
        for index in anchors:
            record = dataset.records[int(classification.record_index[index])]
            end = int(classification.target_end[index])
            start = end - RAW6_SAMPLES
            if start < 0 or not bool(record.valid[start:end].all()):
                raise ValueError(f"{subject} Raw6 support invalid at window {index}")
        supports[subject] = ExternalSubjectSupport(
            subject=subject,
            forecast_window_index=np.asarray(indices, dtype=np.int64),
            anchor_window_index=anchors,
            history_window_index=history,
            y=labels,
        )
        support_hashes[subject] = {
            "forecast_window_sha256": _array_sha256(indices),
            "anchor_window_sha256": _array_sha256(anchors),
            "history_window_sha256": _array_sha256(history),
            "windows": int(len(indices)),
            "anchors": int(len(anchors)),
        }

    scientific = {
        "external_version": EXTERNAL_VERSION,
        "main_protocol_fingerprint": str(protocol.config["protocol_fingerprint"]),
        "data_sha256": data_sha,
        "subjects": list(EXTERNAL_SUBJECTS),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "history_samples": HISTORY_SAMPLES,
        "label_samples": LABEL_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "master_window_sha256": horizon.window_table_sha256(master),
        "classification_window_sha256": horizon.window_table_sha256(classification),
        "support": support_hashes,
        "false_alarm_event_definition": {
            "minimum_positive_windows": minimum_positive_windows,
            "merge_gap_seconds": merge_gap_seconds,
            "coverage_policy": "evaluated_valid_nonfog_target_union",
        },
        "label_role": "evaluation_only_never_training_or_threshold_selection",
    }
    return ExternalProtocol(
        fingerprint=canonical_fingerprint(scientific),
        dataset=dataset,
        master_windows=master,
        classification_windows=classification,
        support=supports,
        provenance=scientific,
    )


def _ensure_support_artifact(
    args: argparse.Namespace,
    external: ExternalProtocol,
) -> str:
    root = Path(args.output_dir) / "phase3b" / "external_negative_only" / "dataset"
    arrays_path = root / "support.npz"
    provenance_path = root / "provenance.json"
    done_path = root / "DONE.json"
    task_id = "phase3b/external_negative_only/dataset"
    completed = validate_done(
        done_path,
        stage="h200_phase3b_external_dataset",
        protocol_fingerprint=external.fingerprint,
        task_id=task_id,
        upstream_sha256=str(external.provenance["data_sha256"]),
    )
    if completed is not None:
        with np.load(arrays_path, allow_pickle=False) as source:
            for subject, support in external.support.items():
                expected = {
                    f"{subject}_forecast_window_index": support.forecast_window_index,
                    f"{subject}_anchor_window_index": support.anchor_window_index,
                    f"{subject}_history_window_index": support.history_window_index,
                    f"{subject}_y": support.y,
                }
                for key, value in expected.items():
                    if key not in source or not np.array_equal(source[key], value):
                        raise ValueError(f"Cached external support changed: {key}")
        return sha256_file(done_path)
    if bool(_arg(args, "finalize_only", False)):
        raise FileNotFoundError(done_path)
    payload: dict[str, np.ndarray] = {}
    for subject, support in external.support.items():
        payload.update(
            {
                f"{subject}_forecast_window_index": support.forecast_window_index,
                f"{subject}_anchor_window_index": support.anchor_window_index,
                f"{subject}_history_window_index": support.history_window_index,
                f"{subject}_y": support.y,
            }
        )
    atomic_npz_save(
        arrays_path,
        compressed=bool(_arg(args, "cache_compressed", True)),
        **payload,
    )
    atomic_json_dump(external.provenance, provenance_path)
    atomic_json_dump(
        done_payload(
            stage="h200_phase3b_external_dataset",
            protocol_fingerprint=external.fingerprint,
            task_id=task_id,
            upstream_sha256=str(external.provenance["data_sha256"]),
            relative_to=root,
            artifacts={"support": arrays_path, "provenance": provenance_path},
        ),
        done_path,
    )
    return sha256_file(done_path)


def _inner_root(args: argparse.Namespace, outer: str, nbm_seed: int, index: int) -> Path:
    return (
        Path(args.output_dir)
        / "phase3b"
        / f"loso_{outer}"
        / f"nbm_seed_{int(nbm_seed)}"
        / "inner_models"
        / f"inner_{int(index):02d}"
    )


def _load_inner_model(
    args: argparse.Namespace,
    protocol: Any,
    external: ExternalProtocol,
    *,
    outer_subject: str,
    nbm_seed: int,
    inner_index: int,
    device: torch.device,
) -> tuple[torch.nn.Module, RobustChannelScaler, dict[str, Any], str]:
    root, provenance, best_path, checkpoint_sha, inner_protocol, task_id = (
        _validate_inner_artifact(
            args,
            protocol,
            outer_subject=outer_subject,
            nbm_seed=nbm_seed,
            inner_index=inner_index,
        )
    )
    predictor_id = str(provenance["predictor_id"])

    nbm_runner, _ = _legacy_runners()
    model = nbm_runner.build_model(
        _nbm_namespace(args, protocol),
        "gru",
        external.dataset.n_channels,
        HORIZON_SAMPLES,
        CONTEXT_SAMPLES,
        external.dataset.sampling_rate_hz,
    ).to(device)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    validate_checkpoint(
        checkpoint,
        stage="nbm",
        protocol_fingerprint=inner_protocol,
        task_id=task_id,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, _scaler_from_mapping(provenance["scaler"]), provenance, checkpoint_sha


def _validate_inner_artifact(
    args: argparse.Namespace,
    protocol: Any,
    *,
    outer_subject: str,
    nbm_seed: int,
    inner_index: int,
) -> tuple[Path, dict[str, Any], Path, str, str, str]:
    """Validate inner provenance/checkpoint hashes without allocating a model."""

    root = _inner_root(args, outer_subject, nbm_seed, inner_index)
    provenance = _read_json(root / "inner_provenance.json")
    predictor_id = (
        f"phase3b/outer_{outer_subject}/nbm_seed_{nbm_seed}/inner_{inner_index:02d}"
    )
    expected = {
        "phase": "3b",
        "outer_test_subject": outer_subject,
        "nbm_seed": int(nbm_seed),
        "predictor_id": predictor_id,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Inner predictor provenance differs: {predictor_id}/{key}")
    train_subjects = tuple(str(item) for item in provenance["predictor_train_subjects"])
    scaler_subjects = tuple(str(item) for item in provenance["scaler_fit_subjects"])
    heldout_subjects = tuple(str(item) for item in provenance["heldout_subjects"])
    outer_train_subjects = tuple(
        str(item) for item in provenance["outer_train_subjects"]
    )
    if (
        len(outer_train_subjects) != 6
        or len(train_subjects) != 5
        or len(heldout_subjects) != 1
        or set(train_subjects) != set(scaler_subjects)
        or set(train_subjects) & set(heldout_subjects)
        or set(train_subjects) | set(heldout_subjects) != set(outer_train_subjects)
        or (set(train_subjects) | set(scaler_subjects)) & set(EXTERNAL_SUBJECTS)
    ):
        raise ValueError(f"Inner predictor/scaler leaked an external subject: {predictor_id}")
    source_fold = _read_json(
        Path(args.source_suite_dir) / f"loso_{outer_subject}" / "fold_config.json"
    )
    if set(outer_train_subjects) != set(source_fold["train_subjects"]):
        raise ValueError(f"Inner outer-train ownership differs: {predictor_id}")
    if set(train_subjects) & {
        str(source_fold["val_subject"]),
        str(source_fold["test_subject"]),
    }:
        raise ValueError(f"Inner predictor used outer validation/test data: {predictor_id}")

    inner_protocol = str(provenance["inner_protocol_fingerprint"])
    done_path = root / "gru" / "nbm" / "DONE.json"
    task_id = f"inner_{inner_index:02d}/gru/nbm"
    completed = validate_done(
        done_path,
        stage="nbm",
        protocol_fingerprint=inner_protocol,
        task_id=task_id,
    )
    if completed is None:
        raise FileNotFoundError(done_path)
    best_entry = completed["artifacts"]["best"]
    best_path = _resolve_done_artifact(done_path, best_entry)
    checkpoint_sha = sha256_file(best_path)
    if checkpoint_sha != str(best_entry["sha256"]) or checkpoint_sha != str(
        provenance["checkpoint_sha256"]
    ):
        raise ValueError(f"Inner checkpoint hash changed: {predictor_id}")

    return root, provenance, best_path, checkpoint_sha, inner_protocol, task_id


def _forecast_root(
    args: argparse.Namespace,
    outer_subject: str,
    nbm_seed: int,
    inner_index: int,
    external_subject: str,
) -> Path:
    return (
        Path(args.output_dir)
        / "phase3b"
        / "external_negative_only"
        / f"loso_{outer_subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / "inner_forecasts"
        / f"inner_{int(inner_index):02d}"
        / external_subject
    )


def _load_forecast_ref(
    root: Path,
    *,
    external_fingerprint: str,
    task_id: str,
    checkpoint_sha256: str,
    expected_indices: np.ndarray,
    predictor_id: str,
) -> ForecastCacheRef | None:
    done_path = root / "DONE.json"
    completed = validate_done(
        done_path,
        stage="h200_phase3b_external_physical_forecast",
        protocol_fingerprint=external_fingerprint,
        task_id=task_id,
        upstream_sha256=checkpoint_sha256,
    )
    if completed is None:
        return None
    with np.load(root / "physical_forecast.npz", allow_pickle=False) as source:
        if not np.array_equal(source["window_index"], expected_indices):
            raise ValueError(f"Cached external forecast endpoints changed: {task_id}")
        if np.any(source["y"] != 0):
            raise ValueError(f"Cached external forecast is not negative-only: {task_id}")
    return ForecastCacheRef(root, sha256_file(done_path), checkpoint_sha256, predictor_id)


def _save_external_forecast(
    root: Path,
    forecast: Mapping[str, Any],
    *,
    external_fingerprint: str,
    task_id: str,
    checkpoint_sha256: str,
    provenance: Mapping[str, Any],
    compressed: bool,
) -> ForecastCacheRef:
    arrays_path = root / "physical_forecast.npz"
    provenance_path = root / "provenance.json"
    done_path = root / "DONE.json"
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
            stage="h200_phase3b_external_physical_forecast",
            protocol_fingerprint=external_fingerprint,
            task_id=task_id,
            upstream_sha256=checkpoint_sha256,
            relative_to=root,
            artifacts={"forecast": arrays_path, "provenance": provenance_path},
        ),
        done_path,
    )
    return ForecastCacheRef(
        root=root,
        done_sha256=sha256_file(done_path),
        checkpoint_sha256=checkpoint_sha256,
        predictor_id=str(provenance["predictor_id"]),
    )


def _ensure_inner_external_forecasts(
    args: argparse.Namespace,
    protocol: Any,
    external: ExternalProtocol,
    *,
    outer_subject: str,
    nbm_seed: int,
    inner_index: int,
    device: torch.device,
) -> dict[str, ForecastCacheRef]:
    _, inner_provenance, _, checkpoint_sha, _, _ = _validate_inner_artifact(
        args,
        protocol,
        outer_subject=outer_subject,
        nbm_seed=nbm_seed,
        inner_index=inner_index,
    )
    predictor_id = str(inner_provenance["predictor_id"])
    refs: dict[str, ForecastCacheRef] = {}
    missing: list[str] = []
    for subject in EXTERNAL_SUBJECTS:
        forecast_root = _forecast_root(
            args, outer_subject, nbm_seed, inner_index, subject
        )
        task_id = (
            f"phase3b/external_negative_only/loso_{outer_subject}/"
            f"nbm_seed_{nbm_seed}/inner_{inner_index:02d}/{subject}"
        )
        cached = _load_forecast_ref(
            forecast_root,
            external_fingerprint=external.fingerprint,
            task_id=task_id,
            checkpoint_sha256=checkpoint_sha,
            expected_indices=external.support[subject].forecast_window_index,
            predictor_id=predictor_id,
        )
        if cached is None:
            missing.append(subject)
        else:
            refs[subject] = cached
    if not missing:
        return refs
    if bool(_arg(args, "finalize_only", False)):
        raise FileNotFoundError(
            _forecast_root(args, outer_subject, nbm_seed, inner_index, missing[0])
            / "DONE.json"
        )

    model, scaler, provenance, observed_sha = _load_inner_model(
        args,
        protocol,
        external,
        outer_subject=outer_subject,
        nbm_seed=nbm_seed,
        inner_index=inner_index,
        device=device,
    )
    if observed_sha != checkpoint_sha:
        raise ValueError(f"Inner checkpoint changed while forecasting: {predictor_id}")
    from .h200_crossfit import extract_gaussian_forecasts

    for subject in missing:
        support = external.support[subject]
        forecast = extract_gaussian_forecasts(
            model,
            external.dataset,
            external.master_windows,
            support.forecast_window_index,
            scaler,
            batch_size=int(_arg(args, "phase3_external_batch_size", _arg(args, "batch_size", 256))),
            device=device,
            amp=bool(_arg(args, "amp", True)),
            predictor_id=predictor_id,
            predictor_train_subjects=provenance["predictor_train_subjects"],
            scaler_fit_subjects=provenance["scaler_fit_subjects"],
            heldout_subjects=(subject,),
        )
        if np.any(np.asarray(forecast["y"]) != 0):
            raise AssertionError("External labels were not all negative at scoring time")
        forecast_root = _forecast_root(
            args, outer_subject, nbm_seed, inner_index, subject
        )
        task_id = (
            f"phase3b/external_negative_only/loso_{outer_subject}/"
            f"nbm_seed_{nbm_seed}/inner_{inner_index:02d}/{subject}"
        )
        refs[subject] = _save_external_forecast(
            forecast_root,
            forecast,
            external_fingerprint=external.fingerprint,
            task_id=task_id,
            checkpoint_sha256=checkpoint_sha,
            provenance={
                "external_version": EXTERNAL_VERSION,
                "predictor_id": predictor_id,
                "outer_test_subject": outer_subject,
                "nbm_seed": int(nbm_seed),
                "inner_index": int(inner_index),
                "predictor_train_subjects": provenance["predictor_train_subjects"],
                "scaler_fit_subjects": provenance["scaler_fit_subjects"],
                "external_subject": subject,
                "checkpoint_sha256": checkpoint_sha,
                "units": "physical_imu",
                "external_label_role": "evaluation_only",
            },
            compressed=bool(_arg(args, "cache_compressed", True)),
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return refs


def moment_match_external_forecasts(
    forecasts: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Small public helper used by tests and non-file callers."""

    from .h200_crossfit import ensemble_gaussians

    result = ensemble_gaussians(forecasts)
    return {
        key: np.asarray(result[key])
        for key in ("target", "mu", "sigma", "y", "window_index")
    }


def _moment_match_forecast_refs(refs: Sequence[ForecastCacheRef]) -> dict[str, np.ndarray]:
    """Moment-match cached physical Gaussians without stacking all six models."""

    if not refs:
        raise ValueError("At least one inner external forecast is required")
    reference_target: np.ndarray | None = None
    reference_y: np.ndarray | None = None
    reference_index: np.ndarray | None = None
    mean_sum: np.ndarray | None = None
    second_sum: np.ndarray | None = None
    for ref in refs:
        with np.load(ref.root / "physical_forecast.npz", allow_pickle=False) as source:
            target = np.asarray(source["target"], dtype=np.float32)
            mean = np.asarray(source["mu"], dtype=np.float64)
            sigma = np.asarray(source["sigma"], dtype=np.float64)
            labels = np.asarray(source["y"], dtype=np.int8)
            indices = np.asarray(source["window_index"], dtype=np.int64)
        if reference_target is None:
            reference_target = target
            reference_y = labels
            reference_index = indices
            mean_sum = np.zeros_like(mean, dtype=np.float64)
            second_sum = np.zeros_like(mean, dtype=np.float64)
        else:
            if not np.array_equal(indices, reference_index):
                raise ValueError("External ensemble endpoint order differs")
            if not np.array_equal(labels, reference_y) or not np.allclose(
                target, reference_target, rtol=0.0, atol=1e-6
            ):
                raise ValueError("External ensemble physical targets differ")
        if np.any(sigma <= 0) or not np.isfinite(mean).all() or not np.isfinite(sigma).all():
            raise ValueError("External ensemble Gaussian is invalid")
        assert mean_sum is not None and second_sum is not None
        mean_sum += mean
        second_sum += np.square(sigma) + np.square(mean)
    assert reference_target is not None and reference_y is not None
    assert reference_index is not None and mean_sum is not None and second_sum is not None
    matched_mean = mean_sum / float(len(refs))
    variance = np.maximum(
        second_sum / float(len(refs)) - np.square(matched_mean), 1e-12
    )
    return {
        "target": np.ascontiguousarray(reference_target, dtype=np.float32),
        "mu": np.ascontiguousarray(matched_mean, dtype=np.float32),
        "sigma": np.ascontiguousarray(np.sqrt(variance), dtype=np.float32),
        "y": np.ascontiguousarray(reference_y, dtype=np.int8),
        "window_index": np.ascontiguousarray(reference_index, dtype=np.int64),
    }


def _ensemble_root(
    args: argparse.Namespace, outer_subject: str, nbm_seed: int, subject: str
) -> Path:
    return (
        Path(args.output_dir)
        / "phase3b"
        / "external_negative_only"
        / f"loso_{outer_subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / "ensemble"
        / subject
    )


def _ensure_ensemble_primitives(
    args: argparse.Namespace,
    external: ExternalProtocol,
    *,
    outer: Any,
    nbm_seed: int,
    subject: str,
    refs: Sequence[ForecastCacheRef],
) -> tuple[dict[str, np.ndarray], str]:
    root = _ensemble_root(args, outer.subject, nbm_seed, subject)
    task_id = (
        f"phase3b/external_negative_only/loso_{outer.subject}/"
        f"nbm_seed_{nbm_seed}/ensemble/{subject}"
    )
    upstream = canonical_fingerprint(
        {
            "forecast_done_sha256": [ref.done_sha256 for ref in refs],
            "checkpoint_sha256": [ref.checkpoint_sha256 for ref in refs],
            "predictor_id": [ref.predictor_id for ref in refs],
        }
    )
    done_path = root / "DONE.json"
    completed = validate_done(
        done_path,
        stage="h200_phase3b_external_ensemble",
        protocol_fingerprint=external.fingerprint,
        task_id=task_id,
        upstream_sha256=upstream,
    )
    arrays_path = root / "primitives.npz"
    if completed is None:
        if bool(_arg(args, "finalize_only", False)):
            raise FileNotFoundError(done_path)
        physical = _moment_match_forecast_refs(refs)
        primitives = convert_to_outer_scaler_primitives(
            physical, outer.scaler, z_clip=Z_CLIP
        )
        if np.any(np.asarray(primitives["y"]) != 0):
            raise AssertionError("External primitives are not negative-only")
        atomic_npz_save(
            arrays_path,
            compressed=bool(_arg(args, "cache_compressed", True)),
            **{
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
            },
        )
        provenance_path = root / "provenance.json"
        atomic_json_dump(
            {
                "external_version": EXTERNAL_VERSION,
                "outer_test_subject": outer.subject,
                "outer_validation_subject": outer.val_subject,
                "nbm_seed": int(nbm_seed),
                "external_subject": subject,
                "ensemble_size": len(refs),
                "moment_matching_units": "physical_imu",
                "outer_scaler": outer.scaler.as_dict(),
                "diagnostics": primitives["diagnostics"],
                "predictors": [ref.predictor_id for ref in refs],
                "external_label_role": "evaluation_only",
            },
            provenance_path,
        )
        atomic_json_dump(
            done_payload(
                stage="h200_phase3b_external_ensemble",
                protocol_fingerprint=external.fingerprint,
                task_id=task_id,
                upstream_sha256=upstream,
                relative_to=root,
                artifacts={"primitives": arrays_path, "provenance": provenance_path},
            ),
            done_path,
        )
    with np.load(arrays_path, allow_pickle=False) as source:
        primitives = {
            key: np.asarray(source[key])
            for key in ("raw", "z", "log_sigma", "y", "window_index")
        }
    expected = external.support[subject].forecast_window_index
    if not np.array_equal(primitives["window_index"], expected) or np.any(
        primitives["y"] != 0
    ):
        raise ValueError(f"External ensemble cache identity differs: {task_id}")
    return primitives, sha256_file(done_path)


def _history_rows(window_index: np.ndarray, history: np.ndarray) -> np.ndarray:
    lookup = {int(value): row for row, value in enumerate(window_index)}
    try:
        return np.asarray(
            [[lookup[int(value)] for value in chain] for chain in history],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("External H200 history is absent from forecast cache") from error


def _two_block(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[rows]
    if selected.shape[1:] != (2, 9, HORIZON_SAMPLES):
        raise ValueError(f"Unexpected external two-block shape: {selected.shape}")
    return np.ascontiguousarray(
        selected.transpose(0, 2, 1, 3).reshape(len(rows), 9, HISTORY_SAMPLES),
        dtype=np.float32,
    )


def _raw6(
    external: ExternalProtocol,
    scaler: RobustChannelScaler,
    anchors: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(anchors), 9, RAW6_SAMPLES), dtype=np.float32)
    windows = external.classification_windows
    for row, index in enumerate(anchors):
        record = external.dataset.records[int(windows.record_index[index])]
        end = int(windows.target_end[index])
        start = end - RAW6_SAMPLES
        if start < 0 or not bool(record.valid[start:end].all()):
            raise ValueError(f"External Raw6 support invalid at window {index}")
        result[row] = scaler.transform(record.x[start:end]).T
    return result


def _materialize_classifier_base(
    external: ExternalProtocol,
    subject: str,
    scaler: RobustChannelScaler,
    primitives: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    support = external.support[subject]
    rows = _history_rows(
        np.asarray(primitives["window_index"], dtype=np.int64),
        support.history_window_index,
    )
    return {
        "raw4": _two_block(primitives["raw"], rows),
        "raw6": _raw6(external, scaler, support.anchor_window_index),
        "z4": _two_block(primitives["z"], rows),
        "log_sigma4": _two_block(primitives["log_sigma"], rows),
        "y": support.y,
        "window_index": support.anchor_window_index,
    }


def _predict_probabilities(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            tensor = torch.from_numpy(
                np.ascontiguousarray(x[start : start + int(batch_size)], dtype=np.float32)
            ).to(device)
            with torch.amp.autocast(
                device.type, enabled=bool(amp) and device.type == "cuda"
            ):
                logits = model(tensor)
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    if not probabilities:
        raise ValueError("External classifier input is empty")
    return np.ascontiguousarray(np.concatenate(probabilities), dtype=np.float64)


def negative_only_metrics(
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_index: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    *,
    minimum_positive_windows: int = 2,
    merge_gap_seconds: float = 0.5,
) -> dict[str, Any]:
    """Return only metrics that are defined when every target is negative."""

    indices = np.asarray(window_index, dtype=np.int64)
    probability = np.asarray(y_prob, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.int8)
    if probability.shape != (len(indices),) or prediction.shape != (len(indices),):
        raise ValueError("Negative-only predictions are not aligned")
    if not np.isfinite(probability).all() or not set(np.unique(prediction)).issubset(
        {0, 1}
    ):
        raise ValueError("Negative-only predictions are invalid")
    if np.any(windows.label[indices] != 0):
        raise ValueError("Negative-only metrics received a positive target")
    tn = int(np.sum(prediction == 0))
    fp = int(np.sum(prediction == 1))
    nbm_runner, _ = _legacy_runners()
    events = nbm_runner.event_metrics(
        dataset,
        windows,
        indices,
        prediction,
        minimum_positive_windows=int(minimum_positive_windows),
        merge_gap_seconds=float(merge_gap_seconds),
    )
    if events["evaluable_true_events"] != 0 or events["event_sensitivity"] is not None:
        raise AssertionError("Negative-only event scorer exposed positive-event metrics")
    if events["false_alarm_events"] != events["predicted_events"]:
        raise AssertionError("Every predicted external event must be a false alarm")
    return {
        "metric_scope": "negative_only",
        "n_negative_windows": int(len(indices)),
        "true_negative_windows": tn,
        "false_positive_windows": fp,
        "specificity": tn / len(indices) if len(indices) else None,
        "positive_window_rate": fp / len(indices) if len(indices) else None,
        "mean_predicted_fog_probability": float(probability.mean()),
        "p95_predicted_fog_probability": float(np.quantile(probability, 0.95)),
        "p99_predicted_fog_probability": float(np.quantile(probability, 0.99)),
        "predicted_events": int(events["predicted_events"]),
        "false_alarm_events": int(events["false_alarm_events"]),
        "false_alarm_events_per_hour": events["false_alarm_events_per_hour"],
        "evaluated_nonfog_hours": float(events["evaluated_nonfog_hours"]),
        "event_metric_version": str(events["event_metric_version"]),
        "minimum_positive_windows": int(events["minimum_positive_windows"]),
        "merge_gap_seconds": float(events["merge_gap_seconds"]),
    }


def _classifier_root(
    args: argparse.Namespace,
    outer_subject: str,
    nbm_seed: int,
    classifier_seed: int,
    arm: str,
) -> Path:
    return (
        Path(args.output_dir)
        / "phase3b"
        / f"loso_{outer_subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / f"classifier_seed_{int(classifier_seed)}"
        / arm
    )


def _load_frozen_classifier(
    args: argparse.Namespace,
    protocol: Any,
    *,
    outer_subject: str,
    nbm_seed: int,
    classifier_seed: int,
    arm: str,
    device: torch.device,
) -> tuple[torch.nn.Module, float, str, str]:
    root = _classifier_root(
        args, outer_subject, nbm_seed, classifier_seed, arm
    )
    done_path = root / "DONE.json"
    task_id = (
        f"phase3b/loso_{outer_subject}/nbm_seed_{nbm_seed}/"
        f"classifier_seed_{classifier_seed}/{arm}"
    )
    completed = validate_done(
        done_path,
        stage="h200_phase3_classifier",
        protocol_fingerprint=str(protocol.config["protocol_fingerprint"]),
        task_id=task_id,
    )
    if completed is None:
        raise FileNotFoundError(done_path)
    best_path = _resolve_done_artifact(done_path, completed["artifacts"]["best"])
    metrics_path = _resolve_done_artifact(
        done_path, completed["artifacts"]["metrics"]
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    expected = {
        "phase3_version": PHASE3_VERSION,
        "protocol_fingerprint": str(protocol.config["protocol_fingerprint"]),
        "task_id": task_id,
        "classifier_seed": int(classifier_seed),
        "arm": arm,
        "upstream_sha256": completed.get("upstream_nbm_sha256"),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Frozen classifier identity differs: {task_id}/{key}")
    metrics = _read_json(metrics_path)
    for key, value in {
        "phase": "3b",
        "test_subject": outer_subject,
        "nbm_seed": int(nbm_seed),
        "classifier_seed": int(classifier_seed),
        "arm": arm,
    }.items():
        if metrics.get(key) != value:
            raise ValueError(f"Frozen classifier metrics differ: {task_id}/{key}")
    threshold = float(metrics["threshold"])
    validation_threshold = float(metrics["validation"]["threshold"])
    if not (0.0 <= threshold <= 1.0) or threshold != validation_threshold:
        raise ValueError(f"Invalid validation-frozen threshold: {task_id}")
    if metrics.get("crossfit_cache_sha256") != completed.get(
        "upstream_nbm_sha256"
    ):
        raise ValueError(f"Frozen classifier cross-fit source differs: {task_id}")
    model = build_classifier(
        arm,
        hidden_channels=int(protocol.config["classifier_hidden"]),
        dropout=float(protocol.config["classifier_dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, threshold, sha256_file(best_path), sha256_file(metrics_path)


def _timeline_rows(
    external: ExternalProtocol,
    indices: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    threshold: float,
) -> Iterable[dict[str, Any]]:
    windows = external.classification_windows
    fs = float(external.dataset.sampling_rate_hz)
    for index, prob, pred in zip(indices, probability, prediction):
        record = external.dataset.records[int(windows.record_index[index])]
        yield {
            **metadata,
            "record_id": record.record_id,
            "run_id": record.run_id,
            "window_index": int(index),
            "window_start_sample": int(windows.start[index]),
            "target_start_sample": int(windows.target_start[index]),
            "target_end_exclusive_sample": int(windows.target_end[index]),
            "target_start_sec": int(windows.target_start[index]) / fs,
            "target_end_exclusive_sec": int(windows.target_end[index]) / fs,
            "y_true": 0,
            "y_prob": float(prob),
            "threshold": float(threshold),
            "y_pred": int(pred),
        }


def _evaluate_classifier_cell(
    args: argparse.Namespace,
    protocol: Any,
    external: ExternalProtocol,
    *,
    outer: Any,
    nbm_seed: int,
    classifier_seed: int,
    arm: str,
    subject: str,
    base: Mapping[str, np.ndarray],
    ensemble_done_sha256: str,
    support_done_sha256: str,
    device: torch.device,
) -> ExternalCellResult:
    trained_root = _classifier_root(
        args, outer.subject, nbm_seed, classifier_seed, arm
    )
    trained_done = trained_root / "DONE.json"
    if not trained_done.exists():
        raise FileNotFoundError(trained_done)
    # Validate and identify the frozen classifier before accepting an external cache.
    model, threshold, classifier_sha, metrics_sha = _load_frozen_classifier(
        args,
        protocol,
        outer_subject=outer.subject,
        nbm_seed=nbm_seed,
        classifier_seed=classifier_seed,
        arm=arm,
        device=device,
    )
    upstream = canonical_fingerprint(
        {
            "ensemble_done_sha256": ensemble_done_sha256,
            "external_support_done_sha256": support_done_sha256,
            "classifier_best_sha256": classifier_sha,
            "classifier_metrics_sha256": metrics_sha,
            "validation_threshold": threshold,
        }
    )
    root = (
        Path(args.output_dir)
        / "phase3b"
        / "external_negative_only"
        / f"loso_{outer.subject}"
        / f"nbm_seed_{int(nbm_seed)}"
        / f"classifier_seed_{int(classifier_seed)}"
        / arm
        / subject
    )
    task_id = (
        f"phase3b/external_negative_only/loso_{outer.subject}/"
        f"nbm_seed_{nbm_seed}/classifier_seed_{classifier_seed}/{arm}/{subject}"
    )
    done_path = root / "DONE.json"
    completed = validate_done(
        done_path,
        stage="h200_phase3b_external_classifier",
        protocol_fingerprint=external.fingerprint,
        task_id=task_id,
        upstream_sha256=upstream,
    )
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions.npz"
    if completed is not None:
        del model
        metrics = _read_json(metrics_path)
        with np.load(predictions_path, allow_pickle=False) as source:
            result = ExternalCellResult(
                metrics=metrics,
                window_index=np.asarray(source["window_index"], dtype=np.int64),
                y_prob=np.asarray(source["y_prob"], dtype=np.float64),
                y_pred=np.asarray(source["y_pred"], dtype=np.int8),
                done_sha256=sha256_file(done_path),
            )
        if not np.array_equal(
            result.window_index, external.support[subject].anchor_window_index
        ):
            raise ValueError(f"External classifier endpoints changed: {task_id}")
        return result
    if bool(_arg(args, "finalize_only", False)):
        raise FileNotFoundError(done_path)

    rows = np.arange(len(np.asarray(base["y"])), dtype=np.int64)
    x, labels, indices = prepare_phase3_arm_inputs(base, arm, rows)
    if np.any(labels != 0) or not np.array_equal(
        indices, external.support[subject].anchor_window_index
    ):
        raise AssertionError("External classifier input violates negative-only support")
    probability = _predict_probabilities(
        model,
        x,
        device=device,
        batch_size=int(_arg(args, "phase3_external_batch_size", _arg(args, "batch_size", 256))),
        amp=bool(_arg(args, "amp", True)),
    )
    prediction = (probability >= threshold).astype(np.int8)
    del x, model
    metrics = negative_only_metrics(
        external.dataset,
        external.classification_windows,
        indices,
        probability,
        prediction,
        minimum_positive_windows=int(
            _arg(args, "phase3_external_minimum_positive_windows", 2)
        ),
        merge_gap_seconds=float(
            _arg(args, "phase3_external_merge_gap_seconds", 0.5)
        ),
    )
    metadata = {
        "external_version": EXTERNAL_VERSION,
        "phase": "3b_external_negative_only",
        "external_subject": subject,
        "outer_fold": outer.subject,
        "outer_validation_subject": outer.val_subject,
        "nbm_seed": int(nbm_seed),
        "classifier_seed": int(classifier_seed),
        "arm": arm,
    }
    metrics.update(
        {
            **metadata,
            "validation_threshold": float(threshold),
            "threshold_source": "phase3b_outer_validation_predictions",
            "classifier_best_sha256": classifier_sha,
            "classifier_metrics_sha256": metrics_sha,
            "ensemble_done_sha256": ensemble_done_sha256,
            "external_label_role": "evaluation_only",
        }
    )
    timeline_path = root / "timeline.csv"
    timeline_fields = [
        "external_version",
        "phase",
        "external_subject",
        "outer_fold",
        "outer_validation_subject",
        "nbm_seed",
        "classifier_seed",
        "arm",
        "record_id",
        "run_id",
        "window_index",
        "window_start_sample",
        "target_start_sample",
        "target_end_exclusive_sample",
        "target_start_sec",
        "target_end_exclusive_sec",
        "y_true",
        "y_prob",
        "threshold",
        "y_pred",
    ]
    _atomic_csv(
        timeline_path,
        _timeline_rows(
            external,
            indices,
            probability,
            prediction,
            metadata=metadata,
            threshold=threshold,
        ),
        timeline_fields,
    )
    atomic_npz_save(
        predictions_path,
        compressed=bool(_arg(args, "cache_compressed", True)),
        window_index=indices,
        y_prob=probability,
        y_pred=prediction,
        validation_threshold=np.asarray(threshold, dtype=np.float64),
    )
    atomic_json_dump(metrics, metrics_path)
    atomic_json_dump(
        done_payload(
            stage="h200_phase3b_external_classifier",
            protocol_fingerprint=external.fingerprint,
            task_id=task_id,
            upstream_sha256=upstream,
            relative_to=root,
            artifacts={
                "metrics": metrics_path,
                "predictions": predictions_path,
                "timeline": timeline_path,
            },
        ),
        done_path,
    )
    return ExternalCellResult(
        metrics=metrics,
        window_index=indices,
        y_prob=probability,
        y_pred=prediction,
        done_sha256=sha256_file(done_path),
    )


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("External metric summary received missing/non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n_repetitions": int(array.size),
    }


def _aggregate_results(
    args: argparse.Namespace,
    external: ExternalProtocol,
    results: Sequence[ExternalCellResult],
    *,
    expected_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_subject_arm: dict[tuple[str, str], list[ExternalCellResult]] = {}
    for result in results:
        key = (
            str(result.metrics["external_subject"]),
            str(result.metrics["arm"]),
        )
        by_subject_arm.setdefault(key, []).append(result)
    expected_keys = {
        (subject, arm) for subject in EXTERNAL_SUBJECTS for arm in PHASE3_ARMS
    }
    if set(by_subject_arm) != expected_keys:
        raise ValueError("External aggregation is missing a subject/arm cell")

    subject_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    metric_names = (
        "specificity",
        "positive_window_rate",
        "false_alarm_events_per_hour",
        "mean_predicted_fog_probability",
        "p95_predicted_fog_probability",
        "p99_predicted_fog_probability",
    )
    for subject, arm in sorted(expected_keys):
        repetitions = by_subject_arm[(subject, arm)]
        if len(repetitions) != int(expected_repetitions):
            raise ValueError(
                f"{subject}/{arm} has {len(repetitions)} repetitions; "
                f"expected {expected_repetitions}"
            )
        indices = repetitions[0].window_index
        if any(not np.array_equal(item.window_index, indices) for item in repetitions):
            raise ValueError(f"{subject}/{arm} repetition endpoints differ")
        probability = np.stack([item.y_prob for item in repetitions])
        prediction = np.stack([item.y_pred for item in repetitions])
        mean_probability = probability.mean(axis=0)
        positive_vote_rate = prediction.mean(axis=0)
        consensus = (positive_vote_rate >= 0.5).astype(np.int8)
        consensus_metrics = negative_only_metrics(
            external.dataset,
            external.classification_windows,
            indices,
            mean_probability,
            consensus,
            minimum_positive_windows=int(
                _arg(args, "phase3_external_minimum_positive_windows", 2)
            ),
            merge_gap_seconds=float(
                _arg(args, "phase3_external_merge_gap_seconds", 0.5)
            ),
        )
        row: dict[str, Any] = {
            "external_subject": subject,
            "arm": arm,
            "repetitions": len(repetitions),
            "inference_unit": "external_subject",
            "repetitions_are_independent_subjects": False,
        }
        for metric in metric_names:
            summary = _summary([float(item.metrics[metric]) for item in repetitions])
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_min"] = summary["min"]
            row[f"{metric}_max"] = summary["max"]
        for key, value in consensus_metrics.items():
            if key not in {"metric_scope", "event_metric_version"}:
                row[f"consensus_{key}"] = value
        subject_rows.append(row)

        windows = external.classification_windows
        fs = float(external.dataset.sampling_rate_hz)
        for index, prob, vote, pred in zip(
            indices, mean_probability, positive_vote_rate, consensus
        ):
            record = external.dataset.records[int(windows.record_index[index])]
            timeline_rows.append(
                {
                    "external_subject": subject,
                    "arm": arm,
                    "record_id": record.record_id,
                    "run_id": record.run_id,
                    "window_index": int(index),
                    "target_start_sample": int(windows.target_start[index]),
                    "target_end_exclusive_sample": int(windows.target_end[index]),
                    "target_start_sec": int(windows.target_start[index]) / fs,
                    "target_end_exclusive_sec": int(windows.target_end[index]) / fs,
                    "y_true": 0,
                    "mean_y_prob": float(prob),
                    "positive_vote_rate": float(vote),
                    "consensus_y_pred": int(pred),
                    "repetitions": len(repetitions),
                }
            )

    by_arm: dict[str, Any] = {}
    for arm in PHASE3_ARMS:
        rows = [row for row in subject_rows if row["arm"] == arm]
        if len(rows) != len(EXTERNAL_SUBJECTS):
            raise AssertionError("External subject-level aggregation changed")
        by_arm[arm] = {
            "n_independent_subjects": len(rows),
            "subjects": list(EXTERNAL_SUBJECTS),
            "specificity": _summary(
                [float(row["specificity_mean"]) for row in rows]
            ),
            "false_alarm_events_per_hour": _summary(
                [float(row["false_alarm_events_per_hour_mean"]) for row in rows]
            ),
            "positive_window_rate": _summary(
                [float(row["positive_window_rate_mean"]) for row in rows]
            ),
        }
        for summary in by_arm[arm].values():
            if isinstance(summary, dict):
                summary["n_independent_subjects"] = len(rows)
                summary.pop("n_repetitions", None)

    payload = {
        "external_version": EXTERNAL_VERSION,
        "status": "complete",
        "subjects": list(EXTERNAL_SUBJECTS),
        "arms": list(PHASE3_ARMS),
        "inference_unit": "external_subject",
        "n_independent_subjects": len(EXTERNAL_SUBJECTS),
        "model_repetition_policy": (
            "outer-fold/NBM/classifier repetitions are averaged within subject; "
            "they are not independent subjects"
        ),
        "negative_only_metrics": [
            "specificity",
            "positive_window_rate",
            "false_alarm_events_per_hour",
        ],
        "undefined_binary_metrics_omitted": [
            "sensitivity",
            "precision",
            "F1",
            "AUROC",
            "AUPRC",
            "MCC",
            "event_sensitivity",
            "detection_delay",
        ],
        "subject_metrics": subject_rows,
        "subject_macro": by_arm,
        "external_labels_used_for": "negative-only evaluation assertion and scoring only",
        "external_labels_used_for_training_or_threshold": False,
    }
    return payload, subject_rows, timeline_rows


def run_phase3b_external(
    *,
    args: argparse.Namespace,
    protocol: Any,
    device: torch.device,
    arms: Sequence[str] = PHASE3_ARMS,
) -> dict[str, Any]:
    """Evaluate frozen Phase 3B models on S04/S10 without label leakage."""

    resolved_arms = tuple(str(value) for value in arms)
    if set(resolved_arms) != set(PHASE3_ARMS) or len(resolved_arms) != 3:
        raise ValueError(f"External Phase 3B requires exactly these arms: {PHASE3_ARMS}")
    _parse_external_subjects(args)
    external = _build_external_protocol(args, protocol)
    support_done_sha = _ensure_support_artifact(args, external)
    outer_subjects = phase3_outer_subjects(protocol, "3b")
    seeds = phase3_seed_policy(args, "3b")
    results: list[ExternalCellResult] = []

    for outer_subject in outer_subjects:
        outer = load_outer_fold(args, protocol, outer_subject)
        for nbm_seed in seeds["nbm"]:
            refs_by_subject: dict[str, list[ForecastCacheRef]] = {
                subject: [] for subject in EXTERNAL_SUBJECTS
            }
            heldout_ownership: list[str] = []
            for inner_index in range(len(outer.train_subjects)):
                refs = _ensure_inner_external_forecasts(
                    args,
                    protocol,
                    external,
                    outer_subject=outer_subject,
                    nbm_seed=int(nbm_seed),
                    inner_index=inner_index,
                    device=device,
                )
                inner_provenance = _read_json(
                    _inner_root(args, outer_subject, int(nbm_seed), inner_index)
                    / "inner_provenance.json"
                )
                heldout_ownership.extend(
                    str(item) for item in inner_provenance["heldout_subjects"]
                )
                for subject in EXTERNAL_SUBJECTS:
                    refs_by_subject[subject].append(refs[subject])
            if sorted(heldout_ownership) != sorted(outer.train_subjects):
                raise ValueError(
                    f"Phase 3B inner predictors do not hold out each outer-train "
                    f"subject exactly once: {outer_subject}/seed {nbm_seed}"
                )
            for subject in EXTERNAL_SUBJECTS:
                if len(refs_by_subject[subject]) != 6:
                    raise AssertionError("Phase 3B external ensemble must contain six NBMs")
                primitives, ensemble_sha = _ensure_ensemble_primitives(
                    args,
                    external,
                    outer=outer,
                    nbm_seed=int(nbm_seed),
                    subject=subject,
                    refs=refs_by_subject[subject],
                )
                base = _materialize_classifier_base(
                    external, subject, outer.scaler, primitives
                )
                for classifier_seed in seeds["classifier"]:
                    for arm in resolved_arms:
                        results.append(
                            _evaluate_classifier_cell(
                                args,
                                protocol,
                                external,
                                outer=outer,
                                nbm_seed=int(nbm_seed),
                                classifier_seed=int(classifier_seed),
                                arm=arm,
                                subject=subject,
                                base=base,
                                ensemble_done_sha256=ensemble_sha,
                                support_done_sha256=support_done_sha,
                                device=device,
                            )
                        )
                del base, primitives
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    expected_repetitions = (
        len(outer_subjects) * len(seeds["nbm"]) * len(seeds["classifier"])
    )
    aggregate, subject_rows, timeline_rows = _aggregate_results(
        args,
        external,
        results,
        expected_repetitions=expected_repetitions,
    )
    aggregate.update(
        {
            "external_protocol_fingerprint": external.fingerprint,
            "main_protocol_fingerprint": str(protocol.config["protocol_fingerprint"]),
            "outer_folds": list(outer_subjects),
            "nbm_seeds": list(seeds["nbm"]),
            "classifier_seeds": list(seeds["classifier"]),
            "repetitions_per_subject_arm": expected_repetitions,
        }
    )
    root = Path(args.output_dir) / "phase3b" / "external_negative_only"
    aggregate_path = root / "aggregate.json"
    cell_path = root / "cell_metrics.csv"
    subject_path = root / "subject_metrics.csv"
    timeline_path = root / "subject_averaged_timeline.csv"
    cell_rows = [result.metrics for result in results]
    cell_columns = [
        "external_subject",
        "outer_fold",
        "outer_validation_subject",
        "nbm_seed",
        "classifier_seed",
        "arm",
        "validation_threshold",
        "n_negative_windows",
        "true_negative_windows",
        "false_positive_windows",
        "specificity",
        "positive_window_rate",
        "predicted_events",
        "false_alarm_events",
        "false_alarm_events_per_hour",
        "evaluated_nonfog_hours",
        "mean_predicted_fog_probability",
        "p95_predicted_fog_probability",
        "p99_predicted_fog_probability",
    ]
    subject_columns = list(subject_rows[0])
    timeline_columns = list(timeline_rows[0])
    atomic_json_dump(aggregate, aggregate_path)
    _atomic_csv(cell_path, cell_rows, cell_columns)
    _atomic_csv(subject_path, subject_rows, subject_columns)
    _atomic_csv(timeline_path, timeline_rows, timeline_columns)
    upstream = canonical_fingerprint(
        {
            "support_done_sha256": support_done_sha,
            "cell_done_sha256": sorted(result.done_sha256 for result in results),
        }
    )
    atomic_json_dump(
        done_payload(
            stage="h200_phase3b_external_aggregate",
            protocol_fingerprint=external.fingerprint,
            task_id="phase3b/external_negative_only/aggregate",
            upstream_sha256=upstream,
            relative_to=root,
            artifacts={
                "aggregate": aggregate_path,
                "cell_metrics": cell_path,
                "subject_metrics": subject_path,
                "subject_averaged_timeline": timeline_path,
                "dataset_done": root / "dataset" / "DONE.json",
            },
        ),
        root / "DONE.json",
    )
    return aggregate


__all__ = [
    "EXTERNAL_SUBJECTS",
    "EXTERNAL_VERSION",
    "ExternalProtocol",
    "ExternalSubjectSupport",
    "moment_match_external_forecasts",
    "negative_only_metrics",
    "run_phase3b_external",
]
