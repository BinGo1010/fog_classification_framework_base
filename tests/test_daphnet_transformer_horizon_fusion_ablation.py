from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.data import (
    DaphnetDataset,
    Record,
    RobustChannelScaler,
    WindowTable,
)
from cnbr_fog.resume import sha256_file
import audit_daphnet_transformer_horizon_fusion_ablation as audit
import run_daphnet_transformer_horizon_fusion_ablation as suite
import start_daphnet_transformer_horizon_fusion_ablation_multigpu as multigpu


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        deterministic=True,
        nbm_hidden=48,
        nbm_dropout=0.1,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ffn=128,
        linear_ar_seconds=0.5,
        gru_layers=1,
        classifier_hidden=48,
        classifier_dropout=0.15,
    )


def _endpoint_windows(
    horizon_samples: int,
    rows: int = 64,
) -> WindowTable:
    target_end = 512 + np.arange(rows, dtype=np.int32) * suite.STRIDE_SAMPLES
    target_start = target_end - int(horizon_samples)
    labels = (np.arange(rows) % 7 < 2).astype(np.int8)
    return WindowTable(
        record_index=np.zeros(rows, dtype=np.int32),
        start=target_start - suite.CONTEXT_SAMPLES,
        target_start=target_start,
        target_end=target_end,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )


def _all_horizon_windows(rows: int = 64) -> dict[str, WindowTable]:
    return {
        str(item["horizon_id"]): _endpoint_windows(
            int(item["horizon_samples"]),
            rows,
        )
        for item in suite.HORIZON_DEFINITIONS
    }


def test_protocol_grid_has_six_requested_arms_and_three_controls() -> None:
    horizons = suite.parse_horizons("2,H050,1,H050")
    assert [
        (
            item["horizon_id"],
            item["horizon_samples"],
            item["history_blocks"],
        )
        for item in horizons
    ] == [
        ("H050", 32, 8),
        ("H100", 64, 4),
        ("H200", 128, 2),
    ]
    assert suite.parse_inputs(",".join(reversed(suite.INPUT_VARIANTS))) == list(
        suite.INPUT_VARIANTS
    )
    grid = suite.experiment_grid(_args())
    assert len(grid) == 9
    assert len({item["experiment_id"] for item in grid}) == 9
    assert {item["cell_id"] for item in grid} == set(suite.INPUT_VARIANTS)
    assert sum(item["kind"] == "control" for item in grid) == 3
    assert sum(item["kind"] == "error" for item in grid) == 3
    assert sum(item["kind"] == "fusion" for item in grid) == 3
    assert {
        item["history_blocks"]
        for item in grid
        if item["kind"] != "control"
    } == {2, 4, 8}
    assert all(item["classifier"] == "tcn_m" for item in grid)
    assert all(item["receptive_field_samples"] == 125 for item in grid)
    assert len(suite.COMPARISONS) == 16
    with pytest.raises(ValueError, match="Unknown horizon"):
        suite.parse_horizons("H025")


def test_transformer_encoder_initialization_is_shared_across_horizons() -> None:
    architectures, shared_hash = suite._transformer_architectures(
        _args(),
        suite.parse_horizons("H050,H100,H200"),
        seed=42,
    )
    assert set(architectures) == {"H050", "H100", "H200"}
    assert {
        item["initial_shared_encoder_sha256"]
        for item in architectures.values()
    } == {shared_hash}
    assert len(
        {
            item["shared_encoder_parameter_count"]
            for item in architectures.values()
        }
    ) == 1
    counts = [
        architectures[horizon]["parameter_count"]
        for horizon in ("H050", "H100", "H200")
    ]
    decoder_counts = [
        architectures[horizon]["decoder_parameter_count"]
        for horizon in ("H050", "H100", "H200")
    ]
    assert counts == sorted(counts)
    assert decoder_counts == sorted(decoder_counts)


def test_tcn_initialization_is_aligned_for_9_and_18_channels() -> None:
    states, counts, hashes, backbone_hash = suite.aligned_classifier_states(
        seed=10042,
        hidden_channels=48,
        dropout=0.15,
        deterministic=True,
    )
    assert counts == {9: 89_329, 18: 89_761}
    assert set(hashes) == {9, 18}
    assert torch.equal(
        states[9]["projection.0.weight"],
        states[18]["projection.0.weight"][:, :9],
    )
    for name, value in states[9].items():
        if name == "projection.0.weight":
            continue
        assert torch.equal(value, states[18][name])
    assert isinstance(backbone_hash, str) and len(backbone_hash) == 64


def test_common_support_has_identical_anchors_and_8_4_2_blocks() -> None:
    windows = _all_horizon_windows()
    indices = np.arange(len(windows["H050"]), dtype=np.int64)
    splits = {
        "train": indices,
        "validation": indices,
        "test": indices,
    }
    plans = suite.build_common_history_support(windows, splits)
    reference = plans["H050"]["test"].anchor_window_indices
    assert len(reference) > 0
    for definition in suite.HORIZON_DEFINITIONS:
        horizon_id = str(definition["horizon_id"])
        horizon_samples = int(definition["horizon_samples"])
        blocks = int(definition["history_blocks"])
        for split in ("train", "validation", "test"):
            plan = plans[horizon_id][split]
            np.testing.assert_array_equal(
                plan.anchor_window_indices,
                reference,
            )
            assert plan.max_chain_rows.shape == (len(reference), blocks)
            chain = windows[horizon_id].take(
                indices[plan.max_chain_rows].reshape(-1)
            )
            starts = chain.target_start.reshape(len(reference), blocks)
            ends = chain.target_end.reshape(len(reference), blocks)
            np.testing.assert_array_equal(ends - starts, horizon_samples)
            np.testing.assert_array_equal(starts[:, 1:], ends[:, :-1])


def test_raw4_and_error_materialization_are_horizon_invariant_in_shape() -> None:
    windows = _all_horizon_windows()
    indices = np.arange(len(windows["H050"]), dtype=np.int64)
    splits = {split: indices for split in ("train", "validation", "test")}
    plans = suite.build_common_history_support(windows, splits)
    raw_histories: dict[str, np.ndarray] = {}
    for offset, definition in enumerate(suite.HORIZON_DEFINITIONS):
        horizon_id = str(definition["horizon_id"])
        horizon_samples = int(definition["horizon_samples"])
        window = windows[horizon_id]
        raw = np.stack(
            [
                np.broadcast_to(
                    np.arange(
                        int(start),
                        int(end),
                        dtype=np.float32,
                    )[None, :],
                    (9, horizon_samples),
                )
                for start, end in zip(
                    window.target_start,
                    window.target_end,
                    strict=True,
                )
            ],
            axis=0,
        )
        features = {
            "raw": raw,
            "error": raw + float(offset + 1),
            "y": window.label,
            "window_index": indices,
        }
        raw_payload = suite.materialize_raw4_history(
            features,
            plans[horizon_id]["test"],
            horizon_samples,
        )
        error_payload = suite.materialize_error_history(
            features,
            plans[horizon_id]["test"],
            horizon_samples,
            f"error_{horizon_id.lower()}",
        )
        raw_histories[horizon_id] = raw_payload["raw4"]
        assert raw_payload["raw4"].shape[1:] == (9, 256)
        assert error_payload[f"error_{horizon_id.lower()}"].shape[1:] == (
            9,
            256,
        )
    np.testing.assert_array_equal(
        raw_histories["H050"],
        raw_histories["H100"],
    )
    np.testing.assert_array_equal(
        raw_histories["H050"],
        raw_histories["H200"],
    )


def test_raw6_is_exact_terminal_six_second_scaled_signal(
    tmp_path: Path,
) -> None:
    samples = 2048
    signal = np.arange(samples * 9, dtype=np.float32).reshape(samples, 9)
    record = Record(
        record_id="S01_R01",
        subject_id="S01",
        run_id="R01",
        x=signal,
        y=np.zeros(samples, dtype=np.int8),
        valid=np.ones(samples, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=tmp_path,
        records=[record],
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    windows = _endpoint_windows(32, rows=3)
    scaler = RobustChannelScaler(
        center=np.zeros(9, dtype=np.float32),
        scale=np.ones(9, dtype=np.float32),
        clip=1e12,
    )
    payload = suite.materialize_raw6_history(
        dataset,
        windows,
        np.asarray([0, 2], dtype=np.int64),
        scaler,
    )
    assert payload["raw6"].shape == (2, 9, 384)
    for output_row, window_index in enumerate((0, 2)):
        end = int(windows.target_end[window_index])
        np.testing.assert_array_equal(
            payload["raw6"][output_row],
            signal[end - 384 : end].T,
        )


def test_multigpu_dry_run_targets_new_runner_and_auditor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = multigpu.main(
        [
            "--dry-run",
            "--data-dir",
            str(tmp_path / "processed"),
            "--output-dir",
            str(tmp_path / "output"),
            "--gpus",
            "0-1",
            "--work-folds",
            "S01,S02",
            "--",
            "--seed",
            "42",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert (
        "run_daphnet_transformer_horizon_fusion_ablation.py" in output
    )
    assert (
        "audit_daphnet_transformer_horizon_fusion_ablation.py" in output
    )
    assert "--worker-fold S01" in output
    assert "--worker-fold S02" in output
    assert "--finalize-only" in output
    assert "worker[S01].env.CUDA_VISIBLE_DEVICES=0" in output
    assert "worker[S02].env.CUDA_VISIBLE_DEVICES=1" in output


def test_suite_complete_is_created_only_by_a_full_passing_audit(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "SUITE_COMPLETE.json"
    stale.write_text('{"status":"stale"}', encoding="utf-8")
    partial = {
        "audit_version": audit.AUDIT_VERSION,
        "suite_version": suite.SUITE_VERSION,
        "status": "partial_pass",
        "expected_cells": 72,
        "checked_cells": 9,
        "checked_nbm_tasks": 3,
        "checked_primitive_tasks": 3,
        "checked_fold_manifests": 1,
        "full_complete": False,
        "allow_partial": True,
        "reportable": False,
        "protocol_fingerprint": "a" * 64,
        "missing_cells": ["S02/raw4"],
        "failures": [],
        "warnings": [],
    }
    audit.finalize_audit_artifacts(tmp_path, partial)
    assert not stale.exists()

    complete = {
        **partial,
        "status": "pass",
        "checked_cells": 72,
        "checked_nbm_tasks": 24,
        "checked_primitive_tasks": 24,
        "checked_fold_manifests": 8,
        "full_complete": True,
        "allow_partial": False,
        "reportable": True,
        "missing_cells": [],
    }
    report_path, _, complete_path = audit.finalize_audit_artifacts(
        tmp_path,
        complete,
    )
    marker = json.loads(complete_path.read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["expected_cells"] == 72
    assert marker["checked_cells"] == 72
    assert marker["audit_report_sha256"] == sha256_file(report_path)
