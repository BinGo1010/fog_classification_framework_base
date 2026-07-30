from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_daphnet_fusion_f0_f5_suite as audit
import start_daphnet_fusion_f0_f5_suite_multigpu as multigpu


def test_missing_result_config_returns_a_stable_failure(
    tmp_path: Path,
) -> None:
    report = audit.audit(tmp_path)
    assert report["status"] == "fail"
    assert report["expected_classifier_cells"] == 48
    assert report["completed_classifier_cells"] == 0
    assert report["failures"] == ["missing config.json"]


def test_validation_only_threshold_audit_detects_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cell"
    root.mkdir()
    validation_y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    validation_prob = np.asarray(
        [0.05, 0.20, 0.45, 0.55, 0.80, 0.95],
        dtype=np.float64,
    )
    threshold, validation_metrics = audit.rf.choose_threshold(
        validation_y,
        validation_prob,
    )
    test_y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    test_prob = np.asarray([0.10, 0.70, 0.30, 0.90], dtype=np.float64)
    np.savez(
        root / "validation_predictions.npz",
        window_index=np.arange(6, dtype=np.int64),
        y_true=validation_y,
        y_prob=validation_prob,
        y_pred=(validation_prob >= threshold).astype(np.int8),
    )
    np.savez(
        root / "predictions.npz",
        window_index=np.arange(10, 14, dtype=np.int64),
        y_true=test_y,
        y_prob=test_prob,
        y_pred=(test_prob >= threshold).astype(np.int8),
    )
    metrics = {
        **audit.rf.binary_metrics(test_y, test_prob, threshold),
        "validation": validation_metrics,
        "best_validation_auprc": validation_metrics["auprc"],
        "classifier_seed": 10042,
        "best_epoch": 1,
        "history": [
            {
                "epoch": 1,
                "shuffle_seed": 10043,
                "validation_auprc": validation_metrics["auprc"],
            }
        ],
    }
    audit.rf.add_requested_metrics(metrics)
    support = {
        "validation_anchor_window_index": np.arange(6, dtype=np.int64),
        "validation_y": validation_y,
        "test_anchor_window_index": np.arange(10, 14, dtype=np.int64),
        "test_y": test_y,
    }

    class DummyDataset:
        sampling_rate_hz = 64
        records = []

    class DummyWindows:
        record_index = np.asarray([], dtype=np.int32)

    original_event_metrics = audit.rf.event_metrics
    audit.rf.event_metrics = lambda *_args, **_kwargs: {
        "event_sensitivity": None,
        "false_alarm_events_per_hour": None,
        "median_detection_delay_sec": None,
    }
    try:
        assert audit._validate_predictions(
            root,
            metrics,
            support,
            DummyDataset(),
            DummyWindows(),
        ) == []
        metrics["threshold"] = 0.99
        failures = audit._validate_predictions(
            root,
            metrics,
            support,
            DummyDataset(),
            DummyWindows(),
        )
    finally:
        audit.rf.event_metrics = original_event_metrics
    assert "threshold was not selected from validation" in failures


def test_multigpu_wrapper_builds_one_complete_fold_per_gpu(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "output"
    code = multigpu.main(
        [
            "--dry-run",
            "--no-audit",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--gpus",
            "0-1",
            "--work-folds",
            "S01,S02",
            "--",
            "--source-suite-dir",
            str(tmp_path / "source"),
            "--seed",
            "42",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert f"scheduler_version={multigpu.SCHEDULER_VERSION}" in output
    assert "run_daphnet_fusion_f0_f5_suite.py" in output
    assert "worker[S01].env.CUDA_VISIBLE_DEVICES=0" in output
    assert "worker[S02].env.CUDA_VISIBLE_DEVICES=1" in output
    assert "--worker-fold S01" in output
    assert "--worker-fold S02" in output
    assert "--finalize-only" in output
    assert "audit=(disabled)" in output
    assert not output_dir.exists()


def test_multigpu_wrapper_rejects_incomplete_protocol_options() -> None:
    with pytest.raises(SystemExit):
        multigpu.main(
            [
                "--dry-run",
                "--no-audit",
                "--",
                "--smoke",
            ]
        )
