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

import audit_daphnet_gru_horizon_ablation as audit
import run_daphnet_gru_horizon_ablation as suite
import start_daphnet_gru_horizon_ablation_multigpu as multigpu
from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.resume import sha256_file


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        deterministic=True,
        nbm_hidden=48,
        nbm_dropout=0.1,
        gru_layers=1,
        linear_ar_seconds=0.5,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ffn=128,
        classifier_hidden=48,
        classifier_dropout=0.15,
    )


def _endpoint_windows(
    *,
    horizon_samples: int,
    rows: int = 48,
) -> WindowTable:
    target_end = 512 + np.arange(rows, dtype=np.int32) * 16
    target_start = target_end - int(horizon_samples)
    labels = (np.arange(rows) % 7 < 2).astype(np.int8)
    return WindowTable(
        record_index=np.zeros(rows, dtype=np.int32),
        start=target_start - 128,
        target_start=target_start,
        target_end=target_end,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )


def _all_horizon_windows(rows: int = 48) -> dict[str, WindowTable]:
    return {
        horizon_id: _endpoint_windows(
            horizon_samples=int(definition["samples"]),
            rows=rows,
        )
        for horizon_id, definition in audit.EXPECTED_HORIZONS.items()
    }


def test_horizon_parser_grid_and_fixed_geometry() -> None:
    horizons = suite.parse_horizons(
        "2,H025,0.5,H100,H025",
        sampling_rate_hz=64,
    )
    assert [
        (
            item["horizon_id"],
            item["horizon_samples"],
            item["history_blocks"],
        )
        for item in horizons
    ] == [
        ("H025", 16, 16),
        ("H050", 32, 8),
        ("H100", 64, 4),
        ("H200", 128, 2),
    ]
    grid = suite.horizon_grid(horizons)
    assert len(grid) == 4
    assert len({item["experiment_id"] for item in grid}) == 4
    assert all(item["nbm"] == "gru" for item in grid)
    assert all(item["input"] == "residual_h4s" for item in grid)
    assert all(item["classifier"] == "tcn_m" for item in grid)
    assert grid[0]["directory"] == "horizon_h025_0p25s"
    assert grid[0]["experiment_id"] == (
        "gru__h025_horizon0p25s__residual_h4s__tcn_m"
    )
    with pytest.raises(ValueError, match="Unknown horizon"):
        suite.parse_horizons("H075")
    with pytest.raises(ValueError, match="64 Hz"):
        suite.parse_horizons("H050", sampling_rate_hz=100)


def test_master_endpoint_labels_and_derived_windows_are_right_aligned() -> None:
    samples = 640
    x = np.arange(samples * 9, dtype=np.float32).reshape(samples, 9)
    y = np.zeros(samples, dtype=np.int8)
    # For the endpoint at 256, the complete 128-sample target has only
    # 16/128 FoG, but the fixed final 32 samples have 16/32 FoG and label 1.
    y[240:256] = 1
    record = Record(
        record_id="synthetic",
        subject_id="S01",
        run_id="R01",
        x=x,
        y=y,
        valid=np.ones(samples, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=Path("."),
        records=[record],
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    raw = dataset.make_windows(
        warmup_samples=128,
        target_samples=128,
        stride_samples=16,
        fog_fraction_threshold=0.5,
        normal_guard_samples=0,
    )
    master = suite.relabel_master_windows(dataset, raw)
    endpoint_row = int(np.flatnonzero(master.target_end == 256)[0])
    assert raw.label[endpoint_row] == 0
    assert master.label[endpoint_row] == 1
    assert master.fog_fraction[endpoint_row] == pytest.approx(0.5)

    classification = suite.derive_classification_windows(master)
    assert classification.target_start[endpoint_row] == 224
    assert classification.target_end[endpoint_row] == 256
    np.testing.assert_array_equal(classification.label, master.label)
    for item in suite.HORIZON_DEFINITIONS:
        horizon = int(item["horizon_samples"])
        derived = suite.derive_horizon_windows(master, horizon)
        np.testing.assert_array_equal(derived.target_end, master.target_end)
        np.testing.assert_array_equal(
            derived.target_start,
            master.target_end - horizon,
        )
        np.testing.assert_array_equal(
            derived.start,
            master.target_end - horizon - 128,
        )
        np.testing.assert_array_equal(derived.label, classification.label)
        np.testing.assert_array_equal(
            derived.clean_normal,
            master.clean_normal,
        )


def test_common_support_has_equal_anchors_and_exact_nonoverlap_geometry() -> None:
    windows = _all_horizon_windows()
    indices = np.arange(48, dtype=np.int64)
    splits = {
        "train": indices,
        "validation": indices,
        "test": indices,
    }
    runner_plans = suite.build_common_history_support(windows, splits)
    rebuilt = audit.recompute_common_support(
        windows,
        splits,
        max_classifier_windows=0,
        seed=42,
        fold_index=0,
        labels=windows["H025"].label,
    )

    reference_anchor = rebuilt["H025"]["test"]["anchor"]
    assert len(reference_anchor) > 0
    for horizon_id, definition in audit.EXPECTED_HORIZONS.items():
        for split in audit.EXPECTED_SPLITS:
            expected_chain = indices[
                runner_plans[horizon_id][split].max_chain_rows
            ]
            np.testing.assert_array_equal(
                rebuilt[horizon_id][split]["anchor"],
                reference_anchor,
            )
            np.testing.assert_array_equal(
                rebuilt[horizon_id][split]["chain"],
                expected_chain,
            )
            assert rebuilt[horizon_id][split]["chain"].shape == (
                len(reference_anchor),
                int(definition["blocks"]),
            )
            audit.validate_history_geometry(
                windows[horizon_id],
                anchor=reference_anchor,
                chain=rebuilt[horizon_id][split]["chain"],
                horizon_samples=int(definition["samples"]),
                history_blocks=int(definition["blocks"]),
                label=f"{horizon_id}/{split}",
            )

    broken = rebuilt["H050"]["test"]["chain"].copy()
    broken[0, 1] = broken[0, 0]
    with pytest.raises(audit.AuditError, match="overlap|horizon-spaced"):
        audit.validate_history_geometry(
            windows["H050"],
            anchor=reference_anchor,
            chain=broken,
            horizon_samples=32,
            history_blocks=8,
            label="broken",
        )


def test_train_cap_is_applied_after_common_anchor_intersection() -> None:
    windows = _all_horizon_windows(rows=96)
    indices = np.arange(96, dtype=np.int64)
    splits = {split: indices for split in audit.EXPECTED_SPLITS}
    uncapped = audit.recompute_common_support(
        windows,
        splits,
        max_classifier_windows=0,
        seed=42,
        fold_index=3,
        labels=windows["H025"].label,
    )
    capped = audit.recompute_common_support(
        windows,
        splits,
        max_classifier_windows=12,
        seed=42,
        fold_index=3,
        labels=windows["H025"].label,
    )
    assert len(capped["H025"]["train"]["anchor"]) == 12
    assert set(capped["H025"]["train"]["anchor"]).issubset(
        set(uncapped["H025"]["train"]["anchor"])
    )
    for horizon_id in audit.EXPECTED_HORIZONS:
        np.testing.assert_array_equal(
            capped[horizon_id]["train"]["anchor"],
            capped["H025"]["train"]["anchor"],
        )
        np.testing.assert_array_equal(
            capped[horizon_id]["validation"]["anchor"],
            uncapped[horizon_id]["validation"]["anchor"],
        )


@pytest.mark.parametrize(
    ("horizon_samples", "history_blocks"),
    [(16, 16), (32, 8), (64, 4), (128, 2)],
)
def test_materialized_history_is_always_9_by_256(
    horizon_samples: int,
    history_blocks: int,
) -> None:
    row_count = history_blocks + 2
    indices = np.arange(row_count, dtype=np.int64)
    residual = np.stack(
        [
            np.full((9, horizon_samples), row, dtype=np.float32)
            for row in indices
        ],
        axis=0,
    )
    chain = np.arange(history_blocks, dtype=np.int64)[None, :]
    materialized = audit.materialize_history(
        {
            "window_index": indices,
            "residual": residual,
            "y": np.zeros(row_count, dtype=np.int8),
        },
        chain,
        horizon_samples=horizon_samples,
        history_blocks=history_blocks,
        label="synthetic",
    )
    assert materialized.shape == (1, 9, 256)
    for block in range(history_blocks):
        np.testing.assert_array_equal(
            materialized[
                0,
                :,
                block * horizon_samples : (block + 1) * horizon_samples,
            ],
            np.full((9, horizon_samples), block, dtype=np.float32),
        )


def test_gru_decoder_changes_but_encoder_initialization_is_shared() -> None:
    horizons = suite.parse_horizons("H025,H050,H100,H200")
    architectures, shared_hash = suite.gru_architectures(
        _args(),
        horizons,
        seed=42,
    )
    assert set(architectures) == set(audit.EXPECTED_HORIZONS)
    assert {
        item["initial_shared_encoder_summary_sha256"]
        for item in architectures.values()
    } == {shared_hash}
    assert len(
        {
            item["shared_encoder_summary_parameter_count"]
            for item in architectures.values()
        }
    ) == 1
    parameter_counts = [
        architectures[horizon_id]["parameter_count"]
        for horizon_id in audit.EXPECTED_HORIZONS
    ]
    decoder_counts = [
        architectures[horizon_id]["decoder_parameter_count"]
        for horizon_id in audit.EXPECTED_HORIZONS
    ]
    assert parameter_counts == sorted(parameter_counts)
    assert decoder_counts == sorted(decoder_counts)
    for horizon_id, definition in audit.EXPECTED_HORIZONS.items():
        assert architectures[horizon_id]["model_config"]["horizon"] == (
            definition["samples"]
        )


def test_tcn_m_is_fixed_rf125_and_same_parameter_count() -> None:
    classifier = suite.context_suite.classifier_architecture(
        _args(),
        in_channels=9,
        sampling_rate_hz=64,
    )
    assert classifier["name"] == "tcn_m"
    assert tuple(classifier["dilations"]) == (1, 2, 4, 8, 8, 8)
    assert classifier["receptive_field_samples"] == 125
    assert classifier["parameter_count"] == 89_329
    assert classifier["global_pooling"] == "mean_and_max_over_full_input"


def test_auditor_protocol_payload_excludes_only_runtime_fields() -> None:
    config = {
        "suite_version": suite.SUITE_VERSION,
        "seed": 42,
        "scientific": {"horizons": [16, 32, 64, 128]},
        "protocol_fingerprint": "a" * 64,
        "data_dir": "data",
        "output_dir": "output",
        "device": "cuda",
        "resume": True,
        "num_workers": 0,
    }
    assert audit.protocol_payload(config) == {
        "suite_version": suite.SUITE_VERSION,
        "seed": 42,
        "scientific": {"horizons": [16, 32, 64, 128]},
    }
    assert audit.EXPECTED_CLASSIFIER_CELLS == 32
    assert audit.AUDIT_VERSION.endswith(".v1")


def test_suite_complete_exists_only_for_full_passing_audit(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "SUITE_COMPLETE.json"
    stale.write_text('{"status":"stale"}', encoding="utf-8")
    partial = {
        "audit_version": audit.AUDIT_VERSION,
        "status": "partial_pass",
        "checked_cells": 1,
        "expected_cells": 32,
        "checked_fold_manifests": 0,
        "full_complete": False,
        "allow_partial": True,
        "protocol_fingerprint": "a" * 64,
        "missing_cells": ["S01/H050"],
        "failures": [],
        "warnings": [],
    }
    audit.finalize_audit_artifacts(tmp_path, partial)
    assert not stale.exists()

    complete = {
        **partial,
        "status": "pass",
        "checked_cells": 32,
        "checked_fold_manifests": 8,
        "full_complete": True,
        "allow_partial": False,
        "missing_cells": [],
    }
    report_path, _, complete_path = audit.finalize_audit_artifacts(
        tmp_path,
        complete,
    )
    marker = json.loads(complete_path.read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["expected_cells"] == 32
    assert marker["checked_cells"] == 32
    assert marker["audit_report_sha256"] == sha256_file(report_path)


def test_multigpu_wrapper_targets_horizon_runner_and_auditor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "output"
    code = multigpu.main(
        [
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
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
    assert "run_daphnet_gru_horizon_ablation.py" in output
    assert "audit_daphnet_gru_horizon_ablation.py" in output
    assert "--worker-fold S01" in output
    assert "--worker-fold S02" in output
    assert "--finalize-only" in output
    assert "worker[S01].env.CUDA_VISIBLE_DEVICES=0" in output
    assert "worker[S02].env.CUDA_VISIBLE_DEVICES=1" in output
    assert not output_dir.exists()
