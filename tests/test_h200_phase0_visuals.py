from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.h200_phase0_visuals import (
    SELECTION_GROUPS,
    build_phase0_selection_manifest,
    render_phase0_visualizations,
    validate_phase0_primitives,
)


CHANNEL_NAMES = (
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


def _fixture() -> tuple[DaphnetDataset, WindowTable, dict[str, np.ndarray]]:
    count = 16
    target_samples = 128
    context_samples = 128
    target_starts = 128 + np.arange(count, dtype=np.int32) * 256
    target_ends = target_starts + target_samples
    starts = target_starts - context_samples
    total_samples = int(target_ends[-1] + 32)
    x = np.zeros((total_samples, 9), dtype=np.float32)
    x[:, 6] = np.sin(np.arange(total_samples) / 8.0)
    x[:, 7] = np.cos(np.arange(total_samples) / 9.0)
    x[:, 8] = np.sin(np.arange(total_samples) / 13.0)
    y = np.zeros(total_samples, dtype=np.int8)
    for window_id in range(9, count):
        onset = int(target_starts[window_id] + 16)
        y[onset : onset + 24] = 1
    record = Record(
        record_id="S01_R01",
        subject_id="S01",
        run_id="R01",
        x=x,
        y=y,
        valid=np.ones(total_samples, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=Path("synthetic"),
        records=[record],
        sampling_rate_hz=64,
        channel_names=CHANNEL_NAMES,
    )
    clean = np.arange(count) < 9
    labels = (~clean).astype(np.int8)
    windows = WindowTable(
        record_index=np.zeros(count, dtype=np.int32),
        start=starts,
        target_start=target_starts,
        target_end=target_ends,
        label=labels,
        fog_fraction=np.where(clean, 0.0, 24.0 / 128.0).astype(np.float32),
        clean_normal=clean,
    )

    scores = np.asarray(
        [1.0, 5.0, 5.0, 3.0, 9.0, 8.0, 7.0, 6.0, 4.0]
        + [0.25] * 7,
        dtype=np.float32,
    )
    raw = np.broadcast_to(scores[:, None, None], (count, 9, 128)).copy()
    mu = np.zeros_like(raw)
    sigma = np.ones_like(raw)
    error = raw - mu
    z = error / sigma
    primitives = {
        "raw": raw,
        "mu": mu,
        "sigma": sigma,
        "error": error,
        "z": z,
        "window_index": np.arange(count, dtype=np.int64),
    }
    return dataset, windows, primitives


def _selected_ids(manifest: dict, group: str) -> list[int]:
    return [int(row["window_index"]) for row in manifest["selections"][group]]


def test_selection_is_deterministic_by_global_id_and_residual_tie_break() -> None:
    dataset, windows, primitives = _fixture()
    manifest = build_phase0_selection_manifest(
        dataset, windows, primitives, per_group=5
    )

    assert _selected_ids(manifest, "clean_nonfog_first") == [0, 1, 2, 3, 4]
    assert _selected_ids(manifest, "fog_onset_first") == [9, 10, 11, 12, 13]
    assert _selected_ids(manifest, "clean_nonfog_high_residual") == [4, 5, 6, 7, 1]
    onset_rows = manifest["selections"]["fog_onset_first"]
    assert [row["fog_onset_offsets_in_target"] for row in onset_rows] == [
        [16],
        [16],
        [16],
        [16],
        [16],
    ]
    assert manifest["trunk_channel_indices"] == [6, 7, 8]

    permutation = np.asarray([7, 12, 0, 15, 4, 2, 10, 1, 8, 6, 14, 3, 5, 13, 9, 11])
    shuffled = {
        key: np.asarray(value)[permutation]
        for key, value in primitives.items()
    }
    shuffled_manifest = build_phase0_selection_manifest(
        dataset, windows, shuffled, per_group=5
    )
    for group in SELECTION_GROUPS:
        assert _selected_ids(shuffled_manifest, group) == _selected_ids(manifest, group)


def test_render_writes_one_atomic_png_per_selection_and_json_manifest(
    tmp_path: Path,
) -> None:
    dataset, windows, primitives = _fixture()
    output_dir = tmp_path / "visuals"
    manifest = render_phase0_visualizations(
        dataset,
        windows,
        primitives,
        output_dir,
        per_group=1,
        dpi=55,
    )

    manifest_path = output_dir / "selection_manifest.json"
    assert manifest_path.is_file()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == manifest
    assert loaded["selected_counts"] == {group: 1 for group in SELECTION_GROUPS}
    for group in SELECTION_GROUPS:
        row = loaded["selections"][group][0]
        image_path = output_dir / row["figure_path"]
        assert image_path.is_file()
        assert image_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not list(output_dir.rglob("*.tmp-*.png"))


@pytest.mark.parametrize(
    ("mutation", "expected_exception", "message"),
    [
        ("duplicate_index", ValueError, "duplicates"),
        ("nonfinite", ValueError, "NaN or Inf"),
        ("wrong_shape", ValueError, "shape"),
        ("bad_identity", AssertionError, "error != raw - mu"),
    ],
)
def test_validation_rejects_malformed_primitives(
    mutation: str,
    expected_exception: type[BaseException],
    message: str,
) -> None:
    dataset, windows, source = _fixture()
    primitives = {key: np.asarray(value).copy() for key, value in source.items()}
    if mutation == "duplicate_index":
        primitives["window_index"][1] = primitives["window_index"][0]
    elif mutation == "nonfinite":
        primitives["mu"][0, 0, 0] = np.nan
    elif mutation == "wrong_shape":
        primitives["sigma"] = primitives["sigma"][:, :, :-1]
    elif mutation == "bad_identity":
        primitives["error"][0, 0, 0] += 1.0
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(mutation)

    with pytest.raises(expected_exception, match=message):
        validate_phase0_primitives(dataset, windows, primitives)


def test_validation_rejects_false_clean_normal_metadata() -> None:
    dataset, windows, primitives = _fixture()
    changed = WindowTable(
        record_index=windows.record_index,
        start=windows.start,
        target_start=windows.target_start,
        target_end=windows.target_end,
        label=windows.label,
        fog_fraction=windows.fog_fraction,
        clean_normal=np.ones(len(windows), dtype=bool),
    )
    with pytest.raises(ValueError, match="marked clean_normal"):
        validate_phase0_primitives(dataset, changed, primitives)
