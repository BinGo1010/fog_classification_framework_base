"""Deterministic Phase-0 window selection and H200 forecast visualisations.

The functions in this module deliberately operate on the saved forecast
primitives rather than re-running a model.  A split primitive mapping is
expected to contain ``raw``, ``mu``, ``sigma``, ``error``, ``z`` and
``window_index``.  Forecast arrays use the canonical ``[window, channel,
time]`` layout.

Selection is fixed and auditable:

* the first clean non-FOG windows in ascending global window-ID order;
* the first windows whose *target* contains a sample-level 0 -> 1 FOG onset;
* the clean non-FOG windows with the largest all-channel residual RMS, with
  ascending global window ID as the tie breaker.

Each selected window is rendered independently.  Only the trunk/waist triad
is displayed, but residual RMS is intentionally computed over every model
channel so that the diagnostic selection cannot silently ignore a failed
sensor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from .data import DaphnetDataset, WindowTable
from .resume import atomic_json_dump


PHASE0_VISUAL_SCHEMA_VERSION = 1
H200_SAMPLING_RATE_HZ = 64
H200_TARGET_SAMPLES = 128
DEFAULT_PER_GROUP = 5
DEFAULT_Z_CLIP = 12.0

SELECTION_GROUPS = (
    "clean_nonfog_first",
    "fog_onset_first",
    "clean_nonfog_high_residual",
)

_REQUIRED_PRIMITIVE_KEYS = (
    "raw",
    "mu",
    "sigma",
    "error",
    "z",
    "window_index",
)


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real valued")
    return array


def validate_phase0_primitives(
    dataset: DaphnetDataset,
    windows: WindowTable,
    primitives: Mapping[str, Any],
    *,
    expected_sampling_rate_hz: int = H200_SAMPLING_RATE_HZ,
    expected_target_samples: int = H200_TARGET_SAMPLES,
    z_clip: float | None = DEFAULT_Z_CLIP,
    rtol: float = 2e-4,
    atol: float = 2e-4,
) -> dict[str, np.ndarray]:
    """Validate and return a canonical view of one split's primitives.

    Validation covers shape, finite values, unique/in-range global window
    indices, record-local support, positive uncertainty, and the two primitive
    identities.  ``z_clip=None`` validates the un-clipped standardized error;
    the default matches the H200 feasibility cache.
    """

    missing = [key for key in _REQUIRED_PRIMITIVE_KEYS if key not in primitives]
    if missing:
        raise KeyError(f"missing Phase-0 primitive keys: {missing}")
    if int(dataset.sampling_rate_hz) != int(expected_sampling_rate_hz):
        raise ValueError(
            "Phase-0 visualisation requires "
            f"{expected_sampling_rate_hz} Hz data, got {dataset.sampling_rate_hz}"
        )
    if int(expected_target_samples) <= 0:
        raise ValueError("expected_target_samples must be positive")
    if z_clip is not None and (not np.isfinite(z_clip) or float(z_clip) <= 0.0):
        raise ValueError("z_clip must be positive and finite, or None")

    window_fields = {
        "record_index": windows.record_index,
        "start": windows.start,
        "target_start": windows.target_start,
        "target_end": windows.target_end,
        "label": windows.label,
        "fog_fraction": windows.fog_fraction,
        "clean_normal": windows.clean_normal,
    }
    for name, value in window_fields.items():
        if np.asarray(value).shape != (len(windows),):
            raise ValueError(
                f"WindowTable.{name} must have shape ({len(windows)},), got "
                f"{np.asarray(value).shape}"
            )
    if not np.isfinite(np.asarray(windows.fog_fraction)).all():
        raise ValueError("WindowTable.fog_fraction contains NaN or Inf")
    if not np.isin(np.asarray(windows.label), (0, 1)).all():
        raise ValueError("WindowTable.label must be binary")

    arrays = {
        key: _as_float_array(primitives[key], name=key)
        for key in ("raw", "mu", "sigma", "error", "z")
    }
    reference_shape = arrays["raw"].shape
    if len(reference_shape) != 3:
        raise ValueError(
            f"raw must have shape [window,channel,time], got {reference_shape}"
        )
    if reference_shape[0] == 0:
        raise ValueError("Phase-0 primitive split must not be empty")
    if reference_shape[1] != int(dataset.n_channels):
        raise ValueError(
            f"primitive channel count {reference_shape[1]} != dataset channel "
            f"count {dataset.n_channels}"
        )
    if reference_shape[2] != int(expected_target_samples):
        raise ValueError(
            f"primitive target length {reference_shape[2]} != expected "
            f"{expected_target_samples}"
        )
    for key, array in arrays.items():
        if array.shape != reference_shape:
            raise ValueError(
                f"{key} shape {array.shape} != raw shape {reference_shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf")

    raw_indices = np.asarray(primitives["window_index"])
    if raw_indices.ndim != 1 or raw_indices.shape[0] != reference_shape[0]:
        raise ValueError(
            "window_index must have shape [window] matching primitive rows"
        )
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise TypeError(f"window_index must be integer, got {raw_indices.dtype}")
    window_indices = raw_indices.astype(np.int64, copy=False)
    if np.unique(window_indices).size != window_indices.size:
        raise ValueError("window_index contains duplicates")
    if np.any(window_indices < 0) or np.any(window_indices >= len(windows)):
        raise IndexError("window_index is outside WindowTable")

    if np.any(arrays["sigma"] <= 0.0):
        raise ValueError("sigma must be strictly positive")
    np.testing.assert_allclose(
        arrays["error"],
        arrays["raw"] - arrays["mu"],
        rtol=float(rtol),
        atol=float(atol),
        err_msg="error != raw - mu",
    )
    expected_z = arrays["error"] / arrays["sigma"]
    if z_clip is not None:
        expected_z = np.clip(expected_z, -float(z_clip), float(z_clip))
    np.testing.assert_allclose(
        arrays["z"],
        expected_z,
        rtol=float(rtol),
        atol=float(atol),
        err_msg="z != clipped(error / sigma)",
    )

    target_length = int(reference_shape[2])
    for window_index in window_indices.tolist():
        record_index = int(windows.record_index[window_index])
        if record_index < 0 or record_index >= len(dataset.records):
            raise IndexError(f"invalid record index for window {window_index}")
        record = dataset.records[record_index]
        start = int(windows.start[window_index])
        target_start = int(windows.target_start[window_index])
        target_end = int(windows.target_end[window_index])
        if not (0 <= start <= target_start < target_end <= len(record.y)):
            raise ValueError(f"invalid record-local support for window {window_index}")
        if target_end - target_start != target_length:
            raise ValueError(
                f"window {window_index} target length {target_end - target_start} "
                f"!= primitive length {target_length}"
            )
        if record.x.shape != (len(record.y), dataset.n_channels):
            raise ValueError(f"invalid x/y shape agreement for record {record.record_id}")
        if record.valid.shape != (len(record.y),):
            raise ValueError(f"invalid valid-mask shape for record {record.record_id}")
        if not bool(np.all(record.valid[start:target_end])):
            raise ValueError(f"window {window_index} includes invalid raw samples")
        if not np.isfinite(record.x[start:target_end]).all():
            raise ValueError(f"window {window_index} includes non-finite raw samples")
        target_y = np.asarray(record.y[target_start:target_end])
        if not np.isin(target_y, (0, 1)).all():
            raise ValueError(f"window {window_index} has non-binary target labels")
        if bool(windows.clean_normal[window_index]) and np.any(target_y != 0):
            raise ValueError(
                f"window {window_index} is marked clean_normal but target contains FOG"
            )

    return {
        **{key: np.asarray(value) for key, value in arrays.items()},
        "window_index": window_indices,
    }


def _residual_rms_per_window(
    error: np.ndarray, *, chunk_size: int = 4096
) -> np.ndarray:
    """Compute float64 RMS without allocating a split-sized float64 cube."""

    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    result = np.empty(error.shape[0], dtype=np.float64)
    for start in range(0, error.shape[0], int(chunk_size)):
        end = min(error.shape[0], start + int(chunk_size))
        block = np.asarray(error[start:end], dtype=np.float64)
        result[start:end] = np.sqrt(np.mean(np.square(block), axis=(1, 2)))
    return result


def _target_onset_samples(
    dataset: DaphnetDataset,
    windows: WindowTable,
    window_index: int,
) -> list[int]:
    record = dataset.records[int(windows.record_index[window_index])]
    start = int(windows.target_start[window_index])
    end = int(windows.target_end[window_index])
    target = np.asarray(record.y[start:end], dtype=np.int8)
    previous = int(record.y[start - 1]) if start > 0 else 0
    prior = np.concatenate((np.asarray([previous], dtype=np.int8), target[:-1]))
    offsets = np.flatnonzero((target == 1) & (prior == 0))
    return (offsets + start).astype(int).tolist()


def _window_json(
    dataset: DaphnetDataset,
    windows: WindowTable,
    *,
    window_index: int,
    primitive_row: int,
    residual_rms: float,
    onset_samples: Sequence[int],
) -> dict[str, Any]:
    record_index = int(windows.record_index[window_index])
    record = dataset.records[record_index]
    target_start = int(windows.target_start[window_index])
    return {
        "window_index": int(window_index),
        "primitive_row": int(primitive_row),
        "record_index": record_index,
        "record_id": str(record.record_id),
        "subject_id": str(record.subject_id),
        "run_id": str(record.run_id),
        "window_start": int(windows.start[window_index]),
        "target_start": target_start,
        "target_end_exclusive": int(windows.target_end[window_index]),
        "label": int(windows.label[window_index]),
        "fog_fraction": float(windows.fog_fraction[window_index]),
        "clean_normal": bool(windows.clean_normal[window_index]),
        "residual_rms": float(residual_rms),
        "fog_onset_samples": [int(value) for value in onset_samples],
        "fog_onset_offsets_in_target": [
            int(value - target_start) for value in onset_samples
        ],
    }


def resolve_trunk_channel_indices(
    dataset: DaphnetDataset,
    explicit: Sequence[int] | None = None,
) -> tuple[int, int, int]:
    """Resolve exactly three waist/trunk channels without label information."""

    if explicit is not None:
        indices = tuple(int(value) for value in explicit)
    else:
        indices = tuple(
            index
            for index, name in enumerate(dataset.channel_names)
            if "trunk" in str(name).lower() or "waist" in str(name).lower()
        )
        if len(indices) != 3 and int(dataset.n_channels) == 3:
            indices = (0, 1, 2)
        if len(indices) != 3 and int(dataset.n_channels) >= 9:
            indices = (
                int(dataset.n_channels) - 3,
                int(dataset.n_channels) - 2,
                int(dataset.n_channels) - 1,
            )
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError(f"expected exactly three distinct trunk channels, got {indices}")
    if min(indices) < 0 or max(indices) >= int(dataset.n_channels):
        raise IndexError(f"trunk channel index outside [0,{dataset.n_channels})")
    return indices


def build_phase0_selection_manifest(
    dataset: DaphnetDataset,
    windows: WindowTable,
    primitives: Mapping[str, Any],
    *,
    per_group: int = DEFAULT_PER_GROUP,
    trunk_channel_indices: Sequence[int] | None = None,
    z_clip: float | None = DEFAULT_Z_CLIP,
) -> dict[str, Any]:
    """Return JSON-safe deterministic Phase-0 selections for one split."""

    if int(per_group) <= 0:
        raise ValueError("per_group must be positive")
    arrays = validate_phase0_primitives(
        dataset, windows, primitives, z_clip=z_clip
    )
    trunk = resolve_trunk_channel_indices(dataset, trunk_channel_indices)
    ids = arrays["window_index"]
    row_by_id = {int(window_id): row for row, window_id in enumerate(ids.tolist())}
    rms = _residual_rms_per_window(arrays["error"])
    rms_by_id = {int(window_id): float(rms[row]) for row, window_id in enumerate(ids)}

    sorted_ids = sorted(row_by_id)
    clean_ids = [
        window_id
        for window_id in sorted_ids
        if bool(windows.clean_normal[window_id])
    ]
    onsets_by_id = {
        window_id: _target_onset_samples(dataset, windows, window_id)
        for window_id in sorted_ids
    }
    onset_ids = [window_id for window_id in sorted_ids if onsets_by_id[window_id]]
    high_residual_ids = sorted(
        clean_ids, key=lambda window_id: (-rms_by_id[window_id], window_id)
    )

    selected_ids = {
        "clean_nonfog_first": clean_ids[: int(per_group)],
        "fog_onset_first": onset_ids[: int(per_group)],
        "clean_nonfog_high_residual": high_residual_ids[: int(per_group)],
    }
    available_counts = {
        "clean_nonfog_first": len(clean_ids),
        "fog_onset_first": len(onset_ids),
        "clean_nonfog_high_residual": len(clean_ids),
    }
    selections: dict[str, list[dict[str, Any]]] = {}
    for group in SELECTION_GROUPS:
        selections[group] = [
            _window_json(
                dataset,
                windows,
                window_index=window_id,
                primitive_row=row_by_id[window_id],
                residual_rms=rms_by_id[window_id],
                onset_samples=onsets_by_id[window_id],
            )
            for window_id in selected_ids[group]
        ]

    return {
        "schema_version": PHASE0_VISUAL_SCHEMA_VERSION,
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "target_samples": int(arrays["raw"].shape[-1]),
        "target_seconds": float(
            arrays["raw"].shape[-1] / float(dataset.sampling_rate_hz)
        ),
        "primitive_window_count": int(ids.size),
        "per_group_requested": int(per_group),
        "trunk_channel_indices": [int(value) for value in trunk],
        "trunk_channel_names": [str(dataset.channel_names[value]) for value in trunk],
        "selection_rules": {
            "clean_nonfog_first": "clean_normal, ascending global window_index",
            "fog_onset_first": (
                "target contains a sample-level 0-to-1 transition, ascending "
                "global window_index"
            ),
            "clean_nonfog_high_residual": (
                "clean_normal, descending RMS(error) over all channels and target "
                "samples, tie by ascending global window_index"
            ),
        },
        "available_counts": {
            key: int(value) for key, value in available_counts.items()
        },
        "selected_counts": {
            key: len(selections[key]) for key in SELECTION_GROUPS
        },
        "selections": selections,
    }


def _axis_short_name(name: str, fallback: str) -> str:
    lowered = str(name).lower()
    for token in ("forward", "vertical", "lateral", "x", "y", "z"):
        if lowered.endswith(token) or f"_{token}" in lowered:
            return token.capitalize()
    return fallback


def _spectrogram_db(values: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency, time_s, power = spectrogram(
        np.asarray(values, dtype=np.float64),
        fs=float(fs),
        nperseg=min(64, len(values)),
        noverlap=min(48, max(0, len(values) - 1)),
        nfft=128,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    return frequency, time_s, 10.0 * np.log10(np.maximum(power, 1e-12))


def _atomic_save_figure(figure: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{uuid4().hex}.png"
    )
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=int(dpi),
            bbox_inches="tight",
            facecolor="white",
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plot_phase0_window(
    dataset: DaphnetDataset,
    arrays: Mapping[str, np.ndarray],
    manifest_row: Mapping[str, Any],
    output_path: str | Path,
    *,
    trunk_channel_indices: Sequence[int],
    dpi: int = 160,
) -> Path:
    """Render one selected window to an atomically replaced PNG."""

    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")
    row = int(manifest_row["primitive_row"])
    window_index = int(manifest_row["window_index"])
    if row < 0 or row >= int(np.asarray(arrays["raw"]).shape[0]):
        raise IndexError("primitive_row is outside primitive arrays")
    if int(np.asarray(arrays["window_index"])[row]) != window_index:
        raise ValueError("manifest primitive_row/window_index identity failed")
    trunk = tuple(int(value) for value in trunk_channel_indices)
    if len(trunk) != 3:
        raise ValueError("plot_phase0_window requires three trunk channels")

    fs = int(dataset.sampling_rate_hz)
    raw = np.asarray(arrays["raw"])[row, trunk, :]
    mean = np.asarray(arrays["mu"])[row, trunk, :]
    sigma = np.asarray(arrays["sigma"])[row, trunk, :]
    error = np.asarray(arrays["error"])[row, trunk, :]
    z = np.asarray(arrays["z"])[row, trunk, :]
    seconds = np.arange(raw.shape[-1], dtype=np.float64) / float(fs)
    channel_names = [str(dataset.channel_names[index]) for index in trunk]

    figure, axes = plt.subplots(
        5,
        3,
        figsize=(15.0, 15.5),
        sharex="row",
        constrained_layout=True,
    )
    axis_labels = [
        _axis_short_name(name, fallback)
        for name, fallback in zip(channel_names, ("Axis 1", "Axis 2", "Axis 3"))
    ]
    for column, (channel_name, axis_label) in enumerate(
        zip(channel_names, axis_labels)
    ):
        signal_axis = axes[0, column]
        signal_axis.fill_between(
            seconds,
            mean[column] - 2.0 * sigma[column],
            mean[column] + 2.0 * sigma[column],
            color="#4C78A8",
            alpha=0.2,
            linewidth=0.0,
            label=r"$\mu \pm 2\sigma$",
        )
        signal_axis.plot(seconds, raw[column], color="#111111", lw=1.0, label="raw")
        signal_axis.plot(seconds, mean[column], color="#4C78A8", lw=1.1, label=r"$\mu$")
        signal_axis.set_title(f"{axis_label}: {channel_name}")
        signal_axis.grid(alpha=0.2)
        if column == 0:
            signal_axis.set_ylabel("scaled acceleration")
            signal_axis.legend(loc="upper right", fontsize=8, ncol=3)

        error_axis = axes[1, column]
        error_axis.axhline(0.0, color="#666666", lw=0.7)
        error_axis.plot(seconds, error[column], color="#E45756", lw=1.0)
        error_axis.grid(alpha=0.2)
        if column == 0:
            error_axis.set_ylabel("signed error")

        z_axis = axes[2, column]
        z_axis.axhline(0.0, color="#666666", lw=0.7)
        z_axis.axhline(2.0, color="#999999", lw=0.6, ls="--")
        z_axis.axhline(-2.0, color="#999999", lw=0.6, ls="--")
        z_axis.plot(seconds, z[column], color="#72B7B2", lw=1.0)
        z_axis.grid(alpha=0.2)
        z_axis.set_xlabel("target time (s)")
        if column == 0:
            z_axis.set_ylabel("z")

        for row_index, (values, title) in enumerate(
            ((raw[column], "raw spectrogram"), (z[column], "z spectrogram")),
            start=3,
        ):
            frequency, time_s, power_db = _spectrogram_db(values, fs)
            keep = frequency <= min(12.0, fs / 2.0)
            mesh = axes[row_index, column].pcolormesh(
                time_s,
                frequency[keep],
                power_db[keep],
                shading="auto",
                cmap="magma",
            )
            axes[row_index, column].set_title(f"{axis_label}: {title}")
            axes[row_index, column].set_xlabel("target time (s)")
            if column == 0:
                axes[row_index, column].set_ylabel("frequency (Hz)")
            figure.colorbar(mesh, ax=axes[row_index, column], pad=0.01, label="dB")

    onset_offsets = manifest_row.get("fog_onset_offsets_in_target", [])
    for onset_offset in onset_offsets:
        onset_seconds = float(onset_offset) / float(fs)
        for row_index in range(3):
            for column in range(3):
                axes[row_index, column].axvline(
                    onset_seconds,
                    color="#B22222",
                    lw=0.9,
                    ls=":",
                    alpha=0.85,
                )
    figure.suptitle(
        "Phase 0 H200 forecast diagnostic | "
        f"window={window_index} | subject={manifest_row['subject_id']} | "
        f"record={manifest_row['record_id']} | residual RMS="
        f"{float(manifest_row['residual_rms']):.4f} | onset offsets={onset_offsets}",
        fontsize=13,
    )
    output = Path(output_path)
    try:
        _atomic_save_figure(figure, output, dpi=int(dpi))
    finally:
        plt.close(figure)
    return output


def render_phase0_visualizations(
    dataset: DaphnetDataset,
    windows: WindowTable,
    primitives: Mapping[str, Any],
    output_dir: str | Path,
    *,
    per_group: int = DEFAULT_PER_GROUP,
    trunk_channel_indices: Sequence[int] | None = None,
    z_clip: float | None = DEFAULT_Z_CLIP,
    dpi: int = 160,
) -> dict[str, Any]:
    """Select, render and atomically write ``selection_manifest.json``.

    The returned value is the same JSON-safe object written to disk, including
    POSIX-style relative PNG paths for every selection.
    """

    arrays = validate_phase0_primitives(
        dataset, windows, primitives, z_clip=z_clip
    )
    manifest = build_phase0_selection_manifest(
        dataset,
        windows,
        arrays,
        per_group=per_group,
        trunk_channel_indices=trunk_channel_indices,
        z_clip=z_clip,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trunk = tuple(int(value) for value in manifest["trunk_channel_indices"])

    for group in SELECTION_GROUPS:
        group_dir = output_root / group
        for order, selection in enumerate(manifest["selections"][group], start=1):
            filename = (
                f"{order:02d}_window_{int(selection['window_index']):08d}.png"
            )
            path = group_dir / filename
            plot_phase0_window(
                dataset,
                arrays,
                selection,
                path,
                trunk_channel_indices=trunk,
                dpi=dpi,
            )
            selection["selection_group"] = group
            selection["selection_rank"] = int(order)
            selection["figure_path"] = path.relative_to(output_root).as_posix()

    manifest_path = output_root / "selection_manifest.json"
    atomic_json_dump(manifest, manifest_path)
    return manifest


__all__ = [
    "DEFAULT_PER_GROUP",
    "DEFAULT_Z_CLIP",
    "H200_SAMPLING_RATE_HZ",
    "H200_TARGET_SAMPLES",
    "PHASE0_VISUAL_SCHEMA_VERSION",
    "SELECTION_GROUPS",
    "build_phase0_selection_manifest",
    "plot_phase0_window",
    "render_phase0_visualizations",
    "resolve_trunk_channel_indices",
    "validate_phase0_primitives",
]
