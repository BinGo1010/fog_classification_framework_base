"""Leakage-safe subject cross-fitting helpers for the GRU-H200 experiment.

The functions in this module keep Gaussian forecasts in physical IMU units
until either an OOF batch or an ensemble has been assembled.  This is
important because inner predictors use different robust scalers: means and
standard deviations from those scaler spaces must never be averaged directly.

Array convention is ``[window, channel, lead]`` throughout.  A forecast
mapping contains ``target``, ``mu``, ``sigma``, ``y`` and ``window_index``.
Cross-fit mappings additionally carry a JSON-compatible ``provenance`` object.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import DaphnetDataset, RobustChannelScaler, WindowTable
from .h200_feasibility import (
    STANDARDIZED_ERROR_CLIP,
    gaussian_moment_match,
    scaled_gaussian_to_physical,
)


FORECAST_ARRAY_KEYS = ("target", "mu", "sigma", "y", "window_index")


def _unique_subjects(subjects: Iterable[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(subject).strip() for subject in subjects)
    if not result or any(not subject for subject in result):
        raise ValueError(f"{name} must contain non-empty subject IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate subject IDs")
    return result


def _window_subject(
    dataset: DaphnetDataset, windows: WindowTable, window_index: int
) -> str:
    record_index = int(windows.record_index[int(window_index)])
    if record_index < 0 or record_index >= len(dataset.records):
        raise IndexError(f"invalid record index for window {window_index}")
    return str(dataset.records[record_index].subject_id)


def _validated_indices(
    indices: Sequence[int] | np.ndarray,
    windows: WindowTable,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} contains duplicate window indices")
    if len(values) and (int(values.min()) < 0 or int(values.max()) >= len(windows)):
        raise IndexError(f"{name} contains an out-of-range window index")
    return values


def temporal_clean_normal_split(
    dataset: DaphnetDataset,
    windows: WindowTable,
    inner_train_subjects: Sequence[str],
    *,
    candidate_indices: Sequence[int] | np.ndarray | None = None,
    validation_fraction: float = 0.2,
    min_validation_windows_per_record: int = 1,
) -> dict[str, Any]:
    """Split clean-normal windows chronologically within every record.

    Validation windows are taken from the tail of each usable record.  An
    embargo is then created by retaining only training windows whose complete
    raw support ``[start, target_end)`` ends no later than the first validation
    support starts.  Therefore training and validation raw samples never
    overlap, even when the source WindowTable has a dense sliding stride.

    Records that cannot provide both sides after the embargo are omitted.  To
    prevent silently dropping a subject, every requested subject must still
    contribute at least one training and one validation window.
    """

    subjects = _unique_subjects(inner_train_subjects, name="inner_train_subjects")
    unknown = set(subjects) - set(dataset.subjects)
    if unknown:
        raise ValueError(f"unknown inner-train subjects: {sorted(unknown)}")
    if (
        not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must be finite and in (0, 1)")
    minimum = int(min_validation_windows_per_record)
    if minimum <= 0:
        raise ValueError("min_validation_windows_per_record must be positive")

    if candidate_indices is None:
        candidates = dataset.window_indices_for_subjects(
            windows, subjects, clean_normal_only=True
        )
    else:
        candidates = _validated_indices(
            candidate_indices, windows, name="candidate_indices"
        )
        candidate_subjects = {
            _window_subject(dataset, windows, int(index)) for index in candidates
        }
        if not candidate_subjects.issubset(set(subjects)):
            raise ValueError("candidate_indices contain a non-inner-train subject")
        if not np.all(windows.clean_normal[candidates]):
            raise ValueError("candidate_indices must all be clean-normal")

    # ``clean_normal`` is the stronger context/target/guard condition, but its
    # label implication is worth checking explicitly because an inconsistent
    # WindowTable would otherwise contaminate normal-model validation.
    if not np.all(windows.clean_normal[candidates]) or np.any(
        windows.label[candidates] != 0
    ):
        raise ValueError("clean-normal candidates must have non-FOG labels")

    if len(candidates) == 0:
        raise ValueError("no clean-normal windows are available")

    by_record: dict[int, list[int]] = {}
    for index in candidates:
        record_index = int(windows.record_index[int(index)])
        by_record.setdefault(record_index, []).append(int(index))

    train: list[int] = []
    validation: list[int] = []
    records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    for record_index in sorted(by_record):
        record = dataset.records[record_index]
        ordered = sorted(
            by_record[record_index],
            key=lambda index: (
                int(windows.target_end[index]),
                int(windows.start[index]),
                index,
            ),
        )
        desired = max(minimum, int(math.ceil(len(ordered) * validation_fraction)))
        desired = min(desired, len(ordered) - 1)
        selected_train: list[int] = []
        selected_validation: list[int] = []
        # Reducing the tail size moves the validation boundary later and can
        # recover a training side for short, densely-overlapped records.
        for count in range(desired, minimum - 1, -1):
            if count <= 0 or count >= len(ordered):
                continue
            tail = ordered[-count:]
            validation_start = min(int(windows.start[index]) for index in tail)
            prior = [
                index
                for index in ordered[:-count]
                if int(windows.target_end[index]) <= validation_start
            ]
            if prior:
                selected_train = prior
                selected_validation = tail
                break
        if not selected_train or not selected_validation:
            skipped_records.append(
                {
                    "record_index": record_index,
                    "record_id": str(record.record_id),
                    "subject_id": str(record.subject_id),
                    "clean_normal_windows": len(ordered),
                    "reason": "insufficient_nonoverlapping_tail_support",
                }
            )
            continue

        train_end = max(int(windows.target_end[index]) for index in selected_train)
        validation_start = min(
            int(windows.start[index]) for index in selected_validation
        )
        if train_end > validation_start:
            raise AssertionError("temporal split raw supports overlap")
        train.extend(selected_train)
        validation.extend(selected_validation)
        records.append(
            {
                "record_index": record_index,
                "record_id": str(record.record_id),
                "subject_id": str(record.subject_id),
                "train_windows": len(selected_train),
                "validation_windows": len(selected_validation),
                "train_support_end_exclusive": train_end,
                "validation_support_start": validation_start,
                "embargo_samples": validation_start - train_end,
            }
        )

    train_array = np.asarray(train, dtype=np.int64)
    validation_array = np.asarray(validation, dtype=np.int64)
    if len(train_array) == 0 or len(validation_array) == 0:
        raise ValueError("temporal split produced an empty side")
    if set(train_array.tolist()) & set(validation_array.tolist()):
        raise AssertionError("temporal split contains the same window twice")
    for subject in subjects:
        train_count = sum(
            _window_subject(dataset, windows, int(index)) == subject
            for index in train_array
        )
        validation_count = sum(
            _window_subject(dataset, windows, int(index)) == subject
            for index in validation_array
        )
        if train_count == 0 or validation_count == 0:
            raise ValueError(
                f"subject {subject} does not contribute to both temporal sides"
            )

    return {
        "train_window_index": train_array,
        "validation_window_index": validation_array,
        "inner_train_subjects": list(subjects),
        "validation_fraction": float(validation_fraction),
        "records": records,
        "skipped_records": skipped_records,
        "raw_support_overlap": False,
    }


def _scaler_parameters(
    scaler: RobustChannelScaler | Mapping[str, Any], channels: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(scaler, Mapping):
        center = np.asarray(scaler["center"], dtype=np.float32)
        scale = np.asarray(scaler["scale"], dtype=np.float32)
        clip = float(scaler.get("clip", 12.0))
    else:
        center = np.asarray(scaler.center, dtype=np.float32)
        scale = np.asarray(scaler.scale, dtype=np.float32)
        clip = float(scaler.clip)
    if center.shape != (channels,) or scale.shape != (channels,):
        raise ValueError(f"scaler center/scale must have shape ({channels},)")
    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not math.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError("scaler parameters must be finite with positive scale/clip")
    return center, scale, clip


def _resolve_device(model: torch.nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_gaussian_forecasts(
    model: torch.nn.Module,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: Sequence[int] | np.ndarray,
    inner_scaler: RobustChannelScaler | Mapping[str, Any],
    *,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    amp: bool = False,
    predictor_id: str | None = None,
    predictor_train_subjects: Sequence[str] | None = None,
    scaler_fit_subjects: Sequence[str] | None = None,
    heldout_subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one inner predictor and return physical-unit Gaussian forecasts."""

    window_indices = _validated_indices(indices, windows, name="indices")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    context_lengths = windows.target_start[window_indices] - windows.start[window_indices]
    horizon_lengths = windows.target_end[window_indices] - windows.target_start[window_indices]
    if len(set(context_lengths.astype(int).tolist())) != 1 or len(
        set(horizon_lengths.astype(int).tolist())
    ) != 1:
        raise ValueError("all forecast windows must share context and horizon lengths")
    context_samples = int(context_lengths[0])
    horizon_samples = int(horizon_lengths[0])
    channels = int(dataset.n_channels)
    center, scale, _ = _scaler_parameters(inner_scaler, channels)
    target_batches: list[np.ndarray] = []
    mean_batches: list[np.ndarray] = []
    sigma_batches: list[np.ndarray] = []
    execution_device = _resolve_device(model, device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for offset in range(0, len(window_indices), int(batch_size)):
                batch_indices = window_indices[offset : offset + int(batch_size)]
                scaled_sequences: list[np.ndarray] = []
                physical_targets: list[np.ndarray] = []
                for index in batch_indices:
                    index = int(index)
                    record = dataset.records[int(windows.record_index[index])]
                    start = int(windows.start[index])
                    target_start = int(windows.target_start[index])
                    target_end = int(windows.target_end[index])
                    sequence = np.asarray(record.x[start:target_end], dtype=np.float32)
                    if sequence.shape != (
                        context_samples + horizon_samples,
                        channels,
                    ):
                        raise ValueError(f"window {index} has incomplete raw support")
                    if not np.isfinite(sequence).all():
                        raise ValueError(f"window {index} contains non-finite IMU values")
                    if isinstance(inner_scaler, Mapping):
                        scaled = np.clip(
                            (sequence - center) / scale,
                            -float(inner_scaler.get("clip", 12.0)),
                            float(inner_scaler.get("clip", 12.0)),
                        ).astype(np.float32)
                    else:
                        scaled = inner_scaler.transform(sequence)
                    scaled_sequences.append(np.ascontiguousarray(scaled.T))
                    physical_targets.append(
                        np.ascontiguousarray(record.x[target_start:target_end].T)
                    )
                tensor = torch.from_numpy(np.stack(scaled_sequences)).to(
                    execution_device
                )
                context = tensor[:, :, :context_samples]
                with torch.amp.autocast(
                    execution_device.type,
                    enabled=bool(amp) and execution_device.type == "cuda",
                ):
                    mean_scaled, sigma_scaled = model(context)
                if not isinstance(mean_scaled, torch.Tensor) or not isinstance(
                    sigma_scaled, torch.Tensor
                ):
                    raise TypeError("model must return torch (mean, sigma) tensors")
                expected = (len(batch_indices), channels, horizon_samples)
                if tuple(mean_scaled.shape) != expected or tuple(sigma_scaled.shape) != expected:
                    raise ValueError(
                        f"model Gaussian shape differs from expected {expected}"
                    )
                mean_numpy = mean_scaled.float().cpu().numpy()
                sigma_numpy = sigma_scaled.float().cpu().numpy()
                physical_mean, physical_sigma = scaled_gaussian_to_physical(
                    mean_numpy, sigma_numpy, center, scale
                )
                target_batches.append(
                    np.ascontiguousarray(np.stack(physical_targets), dtype=np.float32)
                )
                mean_batches.append(physical_mean)
                sigma_batches.append(physical_sigma)
    finally:
        model.train(was_training)

    result: dict[str, Any] = {
        "target": np.ascontiguousarray(np.concatenate(target_batches), dtype=np.float32),
        "mu": np.ascontiguousarray(np.concatenate(mean_batches), dtype=np.float32),
        "sigma": np.ascontiguousarray(np.concatenate(sigma_batches), dtype=np.float32),
        "y": np.asarray(windows.label[window_indices], dtype=np.int8),
        "window_index": np.ascontiguousarray(window_indices, dtype=np.int64),
    }
    _validate_forecast_mapping(result)
    if predictor_id is not None:
        if predictor_train_subjects is None or scaler_fit_subjects is None:
            raise ValueError(
                "predictor/scaler subject provenance is required with predictor_id"
            )
        result["provenance"] = {
            "predictor_id": str(predictor_id),
            "predictor_train_subjects": list(
                _unique_subjects(
                    predictor_train_subjects, name="predictor_train_subjects"
                )
            ),
            "scaler_fit_subjects": list(
                _unique_subjects(scaler_fit_subjects, name="scaler_fit_subjects")
            ),
            "heldout_subjects": list(
                _unique_subjects(
                    heldout_subjects or (), name="heldout_subjects"
                )
            ),
        }
    return result


def _validate_forecast_mapping(forecast: Mapping[str, Any]) -> None:
    missing = set(FORECAST_ARRAY_KEYS) - set(forecast)
    if missing:
        raise KeyError(f"forecast mapping lacks {sorted(missing)}")
    target = np.asarray(forecast["target"])
    mean = np.asarray(forecast["mu"])
    sigma = np.asarray(forecast["sigma"])
    labels = np.asarray(forecast["y"])
    indices = np.asarray(forecast["window_index"])
    if target.dtype != np.float32 or mean.dtype != np.float32 or sigma.dtype != np.float32:
        raise TypeError("target/mu/sigma must be float32 physical-unit arrays")
    if target.ndim != 3 or target.shape != mean.shape or target.shape != sigma.shape:
        raise ValueError("target/mu/sigma must share [window,channel,lead] shape")
    if target.shape[0] <= 0 or labels.shape != (target.shape[0],) or indices.shape != (
        target.shape[0],
    ):
        raise ValueError("forecast labels/indices are not aligned")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("forecast window_index contains duplicates")
    if (
        not np.isfinite(target).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma <= 0)
    ):
        raise ValueError("forecast Gaussian values must be finite and sigma > 0")


def _provenance_entry(forecast: Mapping[str, Any]) -> dict[str, Any]:
    source = forecast.get("provenance", forecast)
    if not isinstance(source, Mapping):
        raise TypeError("forecast provenance must be a mapping")
    required = {
        "predictor_id",
        "predictor_train_subjects",
        "scaler_fit_subjects",
        "heldout_subjects",
    }
    missing = required - set(source)
    if missing:
        raise KeyError(f"cross-fit provenance lacks {sorted(missing)}")
    entry = {key: source[key] for key in required}
    if "window_index" in forecast:
        entry["window_index"] = np.asarray(
            forecast["window_index"], dtype=np.int64
        )
    return entry


def audit_crossfit_provenance(
    provenance: Sequence[Mapping[str, Any]],
    *,
    outer_train_subjects: Sequence[str],
    dataset: DaphnetDataset | None = None,
    windows: WindowTable | None = None,
    expected_window_indices: Sequence[int] | np.ndarray | None = None,
    validation_subjects: Sequence[str] = (),
    test_subjects: Sequence[str] = (),
    scheme: str | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Audit subject exclusions and optional row-level OOF ownership."""

    failures: list[str] = []
    try:
        outer = _unique_subjects(outer_train_subjects, name="outer_train_subjects")
    except (TypeError, ValueError) as exc:
        if raise_on_error:
            raise
        return {"status": "fail", "failures": [str(exc)]}
    outer_set = set(outer)
    forbidden = set(str(value) for value in validation_subjects) | set(
        str(value) for value in test_subjects
    )
    if outer_set & forbidden:
        failures.append("outer train subjects overlap validation/test subjects")
    normalized_scheme = None
    if scheme is not None:
        normalized_scheme = str(scheme).strip().lower().replace("-", "")
        if normalized_scheme not in {"3fold", "loto"}:
            failures.append("scheme must be 3fold or loto")

    predictor_ids: set[str] = set()
    heldout_counts = {subject: 0 for subject in outer}
    row_owners: dict[int, list[str]] = {}
    audited: list[dict[str, Any]] = []
    for position, raw in enumerate(provenance):
        try:
            entry = _provenance_entry(raw)
            predictor_id = str(entry["predictor_id"]).strip()
            train = _unique_subjects(
                entry["predictor_train_subjects"],
                name="predictor_train_subjects",
            )
            scaler = _unique_subjects(
                entry["scaler_fit_subjects"], name="scaler_fit_subjects"
            )
            heldout = _unique_subjects(
                entry["heldout_subjects"], name="heldout_subjects"
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"entry {position}: {exc}")
            continue
        if not predictor_id:
            failures.append(f"entry {position}: empty predictor_id")
        elif predictor_id in predictor_ids:
            failures.append(f"duplicate predictor_id {predictor_id}")
        predictor_ids.add(predictor_id)
        train_set, scaler_set, heldout_set = set(train), set(scaler), set(heldout)
        if train_set != scaler_set:
            failures.append(f"{predictor_id}: scaler subjects differ from predictor train")
        if train_set & heldout_set:
            failures.append(f"{predictor_id}: heldout subject seen by predictor/scaler")
        if train_set | heldout_set != outer_set:
            failures.append(f"{predictor_id}: inner split does not partition outer train")
        if (train_set | scaler_set) & forbidden:
            failures.append(f"{predictor_id}: validation/test subject used for fitting")
        if normalized_scheme == "3fold" and (
            len(train) != 4 or len(heldout) != 2
        ):
            failures.append(f"{predictor_id}: wrong 3fold split sizes")
        if normalized_scheme == "loto" and (
            len(train) != len(outer) - 1 or len(heldout) != 1
        ):
            failures.append(f"{predictor_id}: wrong loto split sizes")
        for subject in heldout:
            if subject in heldout_counts:
                heldout_counts[subject] += 1
        row_indices = np.asarray(entry.get("window_index", []), dtype=np.int64)
        if len(row_indices) != len(np.unique(row_indices)):
            failures.append(f"{predictor_id}: duplicate forecast window IDs")
        for index in row_indices:
            row_owners.setdefault(int(index), []).append(predictor_id)
            if dataset is not None and windows is not None:
                try:
                    subject = _window_subject(dataset, windows, int(index))
                except (IndexError, ValueError) as exc:
                    failures.append(f"{predictor_id}: {exc}")
                else:
                    if subject not in heldout_set:
                        failures.append(
                            f"{predictor_id}: window {index} subject {subject} is not held out"
                        )
        audited.append(
            {
                "predictor_id": predictor_id,
                "predictor_train_subjects": list(train),
                "scaler_fit_subjects": list(scaler),
                "heldout_subjects": list(heldout),
                "forecast_windows": int(len(row_indices)),
            }
        )

    if any(count != 1 for count in heldout_counts.values()):
        failures.append("each outer-train subject must be held out exactly once")
    if normalized_scheme == "3fold" and len(provenance) != 3:
        failures.append("3fold provenance must contain three predictors")
    if normalized_scheme == "loto" and len(provenance) != len(outer):
        failures.append("loto provenance must contain one predictor per subject")
    if expected_window_indices is not None:
        if dataset is None or windows is None:
            failures.append("dataset/windows are required to audit expected OOF rows")
        else:
            try:
                expected = _validated_indices(
                    expected_window_indices, windows, name="expected_window_indices"
                )
            except (IndexError, ValueError) as exc:
                failures.append(str(exc))
            else:
                expected_set = set(expected.astype(int).tolist())
                observed_set = set(row_owners)
                if observed_set != expected_set:
                    failures.append("OOF window coverage differs from expected support")
                if any(len(row_owners.get(index, [])) != 1 for index in expected_set):
                    failures.append("every OOF window must have exactly one predictor")

    report = {
        "status": "pass" if not failures else "fail",
        "scheme": normalized_scheme,
        "outer_train_subjects": list(outer),
        "audited_predictors": audited,
        "heldout_counts": heldout_counts,
        "expected_windows": (
            int(len(np.asarray(expected_window_indices)))
            if expected_window_indices is not None
            else None
        ),
        "observed_unique_windows": len(row_owners),
        "failures": failures,
    }
    if failures and raise_on_error:
        raise ValueError("cross-fit provenance audit failed: " + "; ".join(failures))
    return report


def assemble_oof_gaussians(
    forecasts: Sequence[Mapping[str, Any]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    outer_train_indices: Sequence[int] | np.ndarray,
    *,
    outer_train_subjects: Sequence[str] | None = None,
    validation_subjects: Sequence[str] = (),
    test_subjects: Sequence[str] = (),
    scheme: str | None = None,
) -> dict[str, Any]:
    """Assemble physical-unit OOF forecasts with exact single ownership."""

    expected = _validated_indices(
        outer_train_indices, windows, name="outer_train_indices"
    )
    if not forecasts:
        raise ValueError("at least one OOF forecast batch is required")
    for forecast in forecasts:
        _validate_forecast_mapping(forecast)
    if outer_train_subjects is None:
        outer_train_subjects = tuple(
            dict.fromkeys(
                _window_subject(dataset, windows, int(index)) for index in expected
            )
        )
    audit = audit_crossfit_provenance(
        forecasts,
        outer_train_subjects=outer_train_subjects,
        dataset=dataset,
        windows=windows,
        expected_window_indices=expected,
        validation_subjects=validation_subjects,
        test_subjects=test_subjects,
        scheme=scheme,
        raise_on_error=True,
    )
    rows: dict[int, tuple[Mapping[str, Any], int, str]] = {}
    for forecast in forecasts:
        predictor_id = str(_provenance_entry(forecast)["predictor_id"])
        for row, index in enumerate(np.asarray(forecast["window_index"], dtype=np.int64)):
            rows[int(index)] = (forecast, row, predictor_id)
    target = np.stack([rows[int(index)][0]["target"][rows[int(index)][1]] for index in expected])
    mean = np.stack([rows[int(index)][0]["mu"][rows[int(index)][1]] for index in expected])
    sigma = np.stack([rows[int(index)][0]["sigma"][rows[int(index)][1]] for index in expected])
    labels = np.asarray(
        [rows[int(index)][0]["y"][rows[int(index)][1]] for index in expected],
        dtype=np.int8,
    )
    predictor_ids = [rows[int(index)][2] for index in expected]
    result = {
        "target": np.ascontiguousarray(target, dtype=np.float32),
        "mu": np.ascontiguousarray(mean, dtype=np.float32),
        "sigma": np.ascontiguousarray(sigma, dtype=np.float32),
        "y": labels,
        "window_index": np.ascontiguousarray(expected, dtype=np.int64),
        "source_predictor_id": np.asarray(predictor_ids, dtype=np.str_),
        "provenance_audit": audit,
    }
    _validate_forecast_mapping(result)
    if not np.array_equal(result["y"], windows.label[expected]):
        raise ValueError("OOF labels differ from WindowTable")
    return result


def ensemble_gaussians(
    forecasts: Sequence[Mapping[str, Any]],
    *,
    expected_window_indices: Sequence[int] | np.ndarray | None = None,
    min_sigma: float = 1e-6,
) -> dict[str, Any]:
    """Align physical-unit inner forecasts and moment-match their Gaussians."""

    if not forecasts:
        raise ValueError("at least one ensemble forecast is required")
    for forecast in forecasts:
        _validate_forecast_mapping(forecast)
    if expected_window_indices is None:
        order = np.asarray(forecasts[0]["window_index"], dtype=np.int64)
    else:
        order = np.asarray(expected_window_indices, dtype=np.int64)
        if order.ndim != 1 or len(order) == 0 or len(np.unique(order)) != len(order):
            raise ValueError("expected_window_indices must be unique and non-empty")
    aligned_means: list[np.ndarray] = []
    aligned_sigmas: list[np.ndarray] = []
    reference_target: np.ndarray | None = None
    reference_y: np.ndarray | None = None
    for model_index, forecast in enumerate(forecasts):
        indices = np.asarray(forecast["window_index"], dtype=np.int64)
        if set(indices.tolist()) != set(order.tolist()):
            raise ValueError(f"ensemble model {model_index} endpoint set differs")
        lookup = {int(index): row for row, index in enumerate(indices)}
        rows = np.asarray([lookup[int(index)] for index in order], dtype=np.int64)
        target = np.asarray(forecast["target"])[rows]
        labels = np.asarray(forecast["y"])[rows]
        if reference_target is None:
            reference_target = target
            reference_y = labels
        else:
            if not np.allclose(target, reference_target, rtol=0.0, atol=1e-6):
                raise ValueError("ensemble models do not share physical targets")
            if not np.array_equal(labels, reference_y):
                raise ValueError("ensemble models do not share labels")
        aligned_means.append(np.asarray(forecast["mu"])[rows])
        aligned_sigmas.append(np.asarray(forecast["sigma"])[rows])
    matched_mean, matched_sigma = gaussian_moment_match(
        np.stack(aligned_means), np.stack(aligned_sigmas), min_sigma=min_sigma
    )
    assert reference_target is not None and reference_y is not None
    result = {
        "target": np.ascontiguousarray(reference_target, dtype=np.float32),
        "mu": matched_mean,
        "sigma": matched_sigma,
        "y": np.ascontiguousarray(reference_y, dtype=np.int8),
        "window_index": np.ascontiguousarray(order, dtype=np.int64),
        "ensemble_size": len(forecasts),
    }
    _validate_forecast_mapping(result)
    return result


def convert_to_outer_scaler_primitives(
    forecast: Mapping[str, Any],
    outer_scaler: RobustChannelScaler | Mapping[str, Any],
    *,
    z_clip: float = STANDARDIZED_ERROR_CLIP,
) -> dict[str, Any]:
    """Convert a physical Gaussian to outer-fold classifier primitives.

    Raw targets follow the outer robust scaler's clipping contract.  Gaussian
    parameters are transformed affinely without clipping.  Both raw-scaler and
    standardized-residual clipping rates are reported explicitly.
    """

    _validate_forecast_mapping(forecast)
    if not math.isfinite(float(z_clip)) or float(z_clip) <= 0:
        raise ValueError("z_clip must be finite and positive")
    target = np.asarray(forecast["target"], dtype=np.float32)
    mean_physical = np.asarray(forecast["mu"], dtype=np.float32)
    sigma_physical = np.asarray(forecast["sigma"], dtype=np.float32)
    channels = int(target.shape[1])
    center, scale, raw_clip = _scaler_parameters(outer_scaler, channels)
    center_view = center.reshape(1, channels, 1).astype(np.float64)
    scale_view = scale.reshape(1, channels, 1).astype(np.float64)
    raw_unclipped = (target.astype(np.float64) - center_view) / scale_view
    raw = np.clip(raw_unclipped, -raw_clip, raw_clip)
    mean = (mean_physical.astype(np.float64) - center_view) / scale_view
    sigma = sigma_physical.astype(np.float64) / scale_view
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("outer-scaled sigma must be finite and positive")
    error = raw - mean
    z_unclipped = error / sigma
    z = np.clip(z_unclipped, -float(z_clip), float(z_clip))
    log_sigma = np.log(sigma)
    raw_mask = np.abs(raw_unclipped) > raw_clip
    z_mask = np.abs(z_unclipped) > float(z_clip)
    diagnostics = {
        "raw_clip": float(raw_clip),
        "z_clip": float(z_clip),
        "raw_clip_rate": float(raw_mask.mean()),
        "z_clip_rate": float(z_mask.mean()),
        "clip_rate": float(z_mask.mean()),
        "raw_clip_rate_by_channel": raw_mask.mean(axis=(0, 2)).tolist(),
        "z_clip_rate_by_channel": z_mask.mean(axis=(0, 2)).tolist(),
    }
    return {
        "raw": np.ascontiguousarray(raw, dtype=np.float32),
        "mu": np.ascontiguousarray(mean, dtype=np.float32),
        "sigma": np.ascontiguousarray(sigma, dtype=np.float32),
        "error": np.ascontiguousarray(error, dtype=np.float32),
        "z": np.ascontiguousarray(z, dtype=np.float32),
        "log_sigma": np.ascontiguousarray(log_sigma, dtype=np.float32),
        "y": np.asarray(forecast["y"], dtype=np.int8),
        "window_index": np.asarray(forecast["window_index"], dtype=np.int64),
        "clip_rate": diagnostics["clip_rate"],
        "diagnostics": diagnostics,
    }


def to_outer_scaled_primitives(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`convert_to_outer_scaler_primitives`."""

    return convert_to_outer_scaler_primitives(*args, **kwargs)


__all__ = [
    "FORECAST_ARRAY_KEYS",
    "assemble_oof_gaussians",
    "audit_crossfit_provenance",
    "convert_to_outer_scaler_primitives",
    "ensemble_gaussians",
    "extract_gaussian_forecasts",
    "temporal_clean_normal_split",
    "to_outer_scaled_primitives",
]
