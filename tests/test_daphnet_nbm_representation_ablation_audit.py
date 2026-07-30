from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_daphnet_nbm_representation_ablation as audit
import run_daphnet_nbm_representation_ablation as suite


def _prediction_fixture() -> tuple[
    dict,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    validation = {
        "window_index": np.arange(6, dtype=np.int64),
        "y_true": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8),
        "y_prob": np.asarray(
            [0.05, 0.20, 0.45, 0.55, 0.80, 0.95],
            dtype=np.float64,
        ),
    }
    threshold, validation_metrics = audit.rf.choose_threshold(
        validation["y_true"],
        validation["y_prob"],
    )
    validation["y_pred"] = (
        validation["y_prob"] >= threshold
    ).astype(np.int8)
    test = {
        "window_index": np.arange(10, 14, dtype=np.int64),
        "y_true": np.asarray([0, 0, 1, 1], dtype=np.int8),
        "y_prob": np.asarray([0.10, 0.70, 0.30, 0.90], dtype=np.float64),
    }
    test["y_pred"] = (test["y_prob"] >= threshold).astype(np.int8)
    metrics = {
        **audit.rf.binary_metrics(
            test["y_true"],
            test["y_prob"],
            threshold,
        ),
        "validation": validation_metrics,
        "best_validation_auprc": validation_metrics["auprc"],
    }
    return metrics, test, validation


def test_validation_only_threshold_contract_detects_tampering() -> None:
    metrics, test, validation = _prediction_fixture()
    assert audit._prediction_threshold_failures(
        metrics,
        test,
        validation,
        validation["window_index"],
        validation["y_true"],
    ) == []

    changed_metrics = copy.deepcopy(metrics)
    changed_metrics["threshold"] = 0.99
    changed_test = copy.deepcopy(test)
    changed_test["y_pred"] = 1 - changed_test["y_pred"]
    issues = audit._prediction_threshold_failures(
        changed_metrics,
        changed_test,
        validation,
        validation["window_index"],
        validation["y_true"],
    )
    assert "threshold was not selected from validation" in issues
    assert "test y_pred differs from validation threshold" in issues


def test_representation_cache_upstream_binds_every_source_component() -> None:
    config = {
        "source": {
            "folds": {
                "S01": {
                    "source_scaler_sha256": "a" * 64,
                    "source_split_indices_sha256": "b" * 64,
                    "models": {
                        "gru": {
                            "source_nbm_best_sha256": "c" * 64,
                            "source_residual_cache_sha256": "d" * 64,
                            "source_residual_done_sha256": "e" * 64,
                        }
                    },
                }
            }
        }
    }
    baseline = audit._source_cache_upstream(config, "S01", "gru")
    for field in (
        "source_nbm_best_sha256",
        "source_residual_cache_sha256",
        "source_residual_done_sha256",
    ):
        changed = copy.deepcopy(config)
        changed["source"]["folds"]["S01"]["models"]["gru"][field] = "f" * 64
        assert audit._source_cache_upstream(changed, "S01", "gru") != baseline
    changed = copy.deepcopy(config)
    changed["source"]["folds"]["S01"]["source_scaler_sha256"] = "f" * 64
    assert audit._source_cache_upstream(changed, "S01", "gru") != baseline
    changed = copy.deepcopy(config)
    changed["source"]["folds"]["S01"][
        "source_split_indices_sha256"
    ] = "f" * 64
    assert audit._source_cache_upstream(changed, "S01", "gru") != baseline


def test_calibration_partition_checks_the_complete_test_split() -> None:
    validation = np.asarray([10, 11, 12], dtype=np.int64)
    test = np.asarray([20, 21, 22], dtype=np.int64)
    assert audit._calibration_partition_failures(
        np.asarray([10, 12], dtype=np.int64),
        validation,
        test,
    ) == []
    issues = audit._calibration_partition_failures(
        np.asarray([10, 21], dtype=np.int64),
        validation,
        test,
    )
    assert "calibration overlaps full test split" in issues
    assert "calibration is not contained in validation" in issues


def test_expected_comparisons_are_subject_paired_and_deterministic() -> None:
    rows = {
        suite.cell_id(nbm, representation): {
            subject: {
                "pr_auc": (
                    0.20
                    + 0.01 * subject_index
                    + 0.02 * representation_index
                )
            }
            for subject_index, subject in enumerate(
                suite.EXPECTED_LOSO_SUBJECTS
            )
        }
        for nbm in suite.NBMS
        for representation_index, representation in enumerate(
            suite.REPRESENTATIONS
        )
    }
    config = {"bootstrap_samples": 256, "bootstrap_seed": 42}
    first = audit._expected_comparisons(rows, config)
    second = audit._expected_comparisons(rows, config)
    assert first == second
    assert len(first) == 12
    assert {
        item["n_paired_subjects"] for item in first
    } == {len(suite.EXPECTED_LOSO_SUBJECTS)}
    fixed_minus_error = next(
        item
        for item in first
        if item["comparison_id"] == "gru__fixed_minus_error"
    )
    assert np.isclose(fixed_minus_error["mean_delta"], 0.02)


def test_audit_rejects_source_manifest_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_dir = tmp_path / "result"
    source_dir = tmp_path / "source"
    data_dir = tmp_path / "data"
    result_dir.mkdir()
    source_dir.mkdir()
    data_dir.mkdir()
    cells = [
        {
            "variant": suite.cell_id(nbm, representation),
            "experiment_id": suite.cell_id(nbm, representation),
            "nbm": nbm,
            "representation": representation,
        }
        for nbm in suite.NBMS
        for representation in suite.REPRESENTATIONS
    ]
    config = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": "a" * 64,
        "nbms": list(suite.NBMS),
        "representations": [
            {"name": name} for name in suite.REPRESENTATIONS
        ],
        "cells": cells,
        "expected_classifier_cells": 96,
        "fixed_sigma": {"test_subject_used": False},
        "source": {"immutable": "expected"},
        "source_suite_dir": str(source_dir),
        "data_dir": str(data_dir),
        "data_sha256": "b" * 64,
        "reportable": True,
        "bootstrap_samples": 16,
        "bootstrap_seed": 42,
    }
    (result_dir / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        suite,
        "build_source_manifest",
        lambda _: ({"immutable": "changed"}, {}),
    )
    monkeypatch.setattr(
        audit.rf,
        "load_dataset_and_windows",
        lambda *_: (None, None, "b" * 64),
    )

    report = audit.audit(
        result_dir,
        source_suite_dir=source_dir,
        data_dir=data_dir,
        allow_partial=True,
    )
    assert report["status"] == "fail"
    assert (
        "immutable source manifest differs from result protocol"
        in report["failures"]
    )


def test_audit_rejects_incomplete_aggregate_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_dir = tmp_path / "result"
    source_dir = tmp_path / "source"
    data_dir = tmp_path / "data"
    result_dir.mkdir()
    source_dir.mkdir()
    data_dir.mkdir()
    cells = [
        {
            "variant": suite.cell_id(nbm, representation),
            "experiment_id": suite.cell_id(nbm, representation),
            "nbm": nbm,
            "representation": representation,
        }
        for nbm in suite.NBMS
        for representation in suite.REPRESENTATIONS
    ]
    source_manifest = {"immutable": "expected"}
    config = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": "a" * 64,
        "nbms": list(suite.NBMS),
        "representations": [
            {"name": name} for name in suite.REPRESENTATIONS
        ],
        "cells": cells,
        "expected_classifier_cells": 96,
        "fixed_sigma": {"test_subject_used": False},
        "source": source_manifest,
        "source_suite_dir": str(source_dir),
        "data_dir": str(data_dir),
        "data_sha256": "b" * 64,
        "reportable": True,
        "bootstrap_samples": 16,
        "bootstrap_seed": 42,
    }
    (result_dir / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    (result_dir / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "best_experiment": None,
                "experiments": {},
                "paired_pr_auc_comparisons": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        suite,
        "build_source_manifest",
        lambda _: (source_manifest, {}),
    )
    monkeypatch.setattr(
        audit.rf,
        "load_dataset_and_windows",
        lambda *_: (None, None, "b" * 64),
    )

    report = audit.audit(
        result_dir,
        source_suite_dir=source_dir,
        data_dir=data_dir,
        allow_partial=True,
    )
    assert report["status"] == "fail"
    assert (
        "aggregate experiment registry is incomplete or changed"
        in report["failures"]
    )
