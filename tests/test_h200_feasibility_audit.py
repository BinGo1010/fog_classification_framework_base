from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.audit_daphnet_gru_residual_feasibility import (
    EXPECTED_SUBJECTS,
    PHASE1_ARMS,
    PHASE2_ARMS,
    PHASE3_ARMS,
    audit_feasibility_results,
    main,
)


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config(root: Path) -> str:
    scientific = {
        "suite_version": "daphnet_gru_h200_residual_feasibility.v1",
        "subjects": list(EXPECTED_SUBJECTS),
        "folds_resolved": list(EXPECTED_SUBJECTS),
        "phase1_arms": list(PHASE1_ARMS),
        "phase2_arms": list(PHASE2_ARMS),
        "phase3_arms": list(PHASE3_ARMS),
        "phase0_plots": True,
        "phase0_plot_windows": 1,
        "phase3_nbm_seeds": "42",
        "phase3_classifier_seed_policy": {"3a": [42], "3b": [42, 43, 44]},
    }
    fingerprint = _fingerprint(scientific)
    _json(
        root / "config.json",
        {
            **scientific,
            "protocol_fingerprint": fingerprint,
            "output_dir": str(root),
            "device": "cpu",
            "resume": True,
        },
    )
    return fingerprint


def _seal(
    root: Path,
    *,
    fingerprint: str,
    stage: str,
    task_id: str,
    artifacts: Mapping[str, Path],
) -> None:
    _json(
        root / "DONE.json",
        {
            "format_version": 1,
            "stage": stage,
            "protocol_fingerprint": fingerprint,
            "task_id": task_id,
            "artifacts": {
                name: {
                    "path": path.resolve().relative_to(root.resolve()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for name, path in artifacts.items()
            },
        },
    )


def _classifier_cell(
    result_root: Path,
    *,
    fingerprint: str,
    phase: str,
    subject: str,
    arm: str,
    endpoint: str = "1" * 64,
    label: str = "2" * 64,
    parameter_count: int = 100,
    initial_hash: str = "3" * 64,
    nbm_seed: int | None = None,
    classifier_seed: int | None = None,
    pr_auc: float = 0.5,
) -> Path:
    if phase in {"1", "2"}:
        root = result_root / f"phase{phase}" / f"loso_{subject}" / arm
        stage = "h200_classifier"
        task_id = f"phase{phase}/loso_{subject}/{arm}"
    else:
        assert nbm_seed is not None and classifier_seed is not None
        root = (
            result_root
            / f"phase{phase}"
            / f"loso_{subject}"
            / f"nbm_seed_{nbm_seed}"
            / f"classifier_seed_{classifier_seed}"
            / arm
        )
        stage = "h200_phase3_classifier"
        task_id = (
            f"phase{phase}/loso_{subject}/nbm_seed_{nbm_seed}/"
            f"classifier_seed_{classifier_seed}/{arm}"
        )
    metrics = {
        "phase": phase,
        "test_subject": subject,
        "arm": arm,
        "endpoint_sha256": endpoint,
        "label_sha256": label,
        "parameter_count": parameter_count,
        "initial_state_sha256": initial_hash,
        "pr_auc": pr_auc,
    }
    if nbm_seed is not None:
        metrics.update(nbm_seed=nbm_seed, classifier_seed=classifier_seed)
    metrics_path = root / "metrics.json"
    prediction_path = root / "predictions.npz"
    _json(metrics_path, metrics)
    prediction_path.write_bytes(b"synthetic predictions")
    _seal(
        root,
        fingerprint=fingerprint,
        stage=stage,
        task_id=task_id,
        artifacts={"metrics": metrics_path, "predictions": prediction_path},
    )
    return root


def _codes(report: Mapping[str, Any]) -> set[str]:
    return {str(item["code"]) for item in report["findings"]}


def test_allow_incomplete_reports_matrix_without_failure(tmp_path: Path) -> None:
    _config(tmp_path)
    report = audit_feasibility_results(
        tmp_path, allow_incomplete=True, write_report=False
    )
    assert report["ok"] is True
    assert report["complete"] is False
    assert report["summary"]["errors"] == 0
    assert report["summary"]["incomplete"] > 0
    assert set(report["matrices"]["phase0"]) == set(EXPECTED_SUBJECTS)
    assert main(["--result-dir", str(tmp_path), "--allow-incomplete"]) == 0
    assert (tmp_path / "audit_report.json").is_file()


def test_done_hash_tampering_is_fatal_even_when_incomplete_is_allowed(
    tmp_path: Path,
) -> None:
    fingerprint = _config(tmp_path)
    root = tmp_path / "misc"
    artifact = root / "payload.bin"
    root.mkdir()
    artifact.write_bytes(b"sealed")
    _seal(
        root,
        fingerprint=fingerprint,
        stage="synthetic",
        task_id="misc",
        artifacts={"payload": artifact},
    )
    artifact.write_bytes(b"tampered")

    report = audit_feasibility_results(
        tmp_path, allow_incomplete=True, write_report=False
    )
    assert report["ok"] is False
    assert {"done_size_mismatch", "done_hash_mismatch"} & _codes(report)
    assert main(["--result-dir", str(tmp_path), "--allow-incomplete"]) == 1


def test_phase1_capacity_and_phase2_endpoint_tampering_are_detected(
    tmp_path: Path,
) -> None:
    fingerprint = _config(tmp_path)
    _classifier_cell(
        tmp_path,
        fingerprint=fingerprint,
        phase="1",
        subject="S01",
        arm="raw4_zero",
        parameter_count=100,
    )
    _classifier_cell(
        tmp_path,
        fingerprint=fingerprint,
        phase="1",
        subject="S01",
        arm="raw4_normality",
        parameter_count=101,
    )
    _classifier_cell(
        tmp_path,
        fingerprint=fingerprint,
        phase="2",
        subject="S01",
        arm="raw4",
        endpoint="a" * 64,
    )
    _classifier_cell(
        tmp_path,
        fingerprint=fingerprint,
        phase="2",
        subject="S01",
        arm="raw6",
        endpoint="b" * 64,
    )

    report = audit_feasibility_results(
        tmp_path, allow_incomplete=True, write_report=False
    )
    assert report["ok"] is False
    assert "capacity_mismatch" in _codes(report)
    assert "endpoint_mismatch" in _codes(report)


def test_phase3_crossfit_subject_leakage_is_detected(tmp_path: Path) -> None:
    fingerprint = _config(tmp_path)
    root = tmp_path / "phase3a" / "loso_S01" / "nbm_seed_42" / "crossfit"
    outer_train = ["S02", "S03", "S05", "S06", "S07", "S08"]
    folds = [
        {
            "fold_index": 0,
            # S02 occurs on both sides: a deliberate leakage fixture.
            "train_subjects": ["S02", "S05", "S06", "S07"],
            "heldout_subjects": ["S02", "S03"],
        },
        {
            "fold_index": 1,
            "train_subjects": ["S02", "S03", "S07", "S08"],
            "heldout_subjects": ["S05", "S06"],
        },
        {
            "fold_index": 2,
            "train_subjects": ["S02", "S03", "S05", "S06"],
            "heldout_subjects": ["S07", "S08"],
        },
    ]
    representation = {"status": "pass", "checks": []}
    provenance = {
        "phase": "3a",
        "scheme": "3fold",
        "outer_test_subject": "S01",
        "outer_validation_subject": "S09",
        "outer_train_subjects": outer_train,
        "nbm_seed": 42,
        "crossfit_plan": {"scheme": "3fold", "subjects": outer_train, "folds": folds},
        "oof_provenance_audit": {
            "status": "pass",
            "failures": [],
            "expected_windows": 4,
            "observed_unique_windows": 4,
        },
        "representation_continuity_audit": representation,
        "forecast_units_before_assembly": "physical_imu",
        "variance_diagnostics": {
            "train_oof": {"ensemble_size_per_row": 1},
            "validation_ensemble": {"ensemble_size": 3},
            "test_ensemble": {"ensemble_size": 3},
        },
        "inner_models": [],
        "inner_checkpoint_sha256": [],
    }
    provenance_path = root / "provenance.json"
    gate_path = root / "representation_gate.json"
    _json(provenance_path, provenance)
    _json(gate_path, representation)
    artifacts = {"provenance": provenance_path, "representation_gate": gate_path}
    for split in ("train", "validation", "test"):
        path = root / f"{split}_primitives.npz"
        path.write_bytes(split.encode("ascii"))
        artifacts[f"{split}_primitives"] = path
    _seal(
        root,
        fingerprint=fingerprint,
        stage="h200_phase3_crossfit",
        task_id="phase3a/loso_S01/nbm_seed_42/crossfit",
        artifacts=artifacts,
    )

    report = audit_feasibility_results(
        tmp_path, allow_incomplete=True, write_report=False
    )
    assert report["ok"] is False
    assert "crossfit_leakage" in _codes(report)


def test_phase3b_repetition_mean_tampering_is_detected(tmp_path: Path) -> None:
    fingerprint = _config(tmp_path)
    for seed, value in zip((42, 43, 44), (0.1, 0.2, 0.3)):
        _classifier_cell(
            tmp_path,
            fingerprint=fingerprint,
            phase="3b",
            subject="S01",
            arm="raw6",
            nbm_seed=42,
            classifier_seed=seed,
            pr_auc=value,
        )
    phase_root = tmp_path / "phase3b"
    representation = {
        "status": "pass",
        "hard_gate": True,
        "cells": [{} for _ in EXPECTED_SUBJECTS],
    }
    representation_path = phase_root / "representation_gate.json"
    _json(representation_path, representation)
    aggregate = {
        "phase": "3b",
        "protocol_fingerprint": fingerprint,
        "representation_gate": representation,
        "science_gate": {"status": "not_applicable"},
        "decision": {"status": "pass"},
        "external_negative_only_evaluation": {
            "subjects": ["S04", "S10"],
            "status": "not_executed",
        },
        "aggregate": {
            arm: {"pr_auc": {"mean": 0.2, "n_subjects": len(EXPECTED_SUBJECTS)}}
            for arm in PHASE3_ARMS
        },
        "paired_bootstrap": {},
    }
    aggregate_path = phase_root / "aggregate.json"
    _json(aggregate_path, aggregate)
    subject_path = phase_root / "subject_seed_averaged_metrics.csv"
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    with subject_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "test_subject", "arm", "repetitions", "pr_auc"]
        )
        writer.writeheader()
        # Correct mean is 0.2; 0.9 is a deliberate aggregation tamper.
        writer.writerow(
            {
                "phase": "3b",
                "test_subject": "S01",
                "arm": "raw6",
                "repetitions": 3,
                "pr_auc": 0.9,
            }
        )
    cells_path = phase_root / "classifier_cells.csv"
    cells_path.write_text("phase\n3b\n", encoding="utf-8")
    _seal(
        phase_root,
        fingerprint=fingerprint,
        stage="h200_phase3_aggregate",
        task_id="phase3b/aggregate",
        artifacts={
            "aggregate": aggregate_path,
            "classifier_cells": cells_path,
            "subject_seed_averaged_metrics": subject_path,
            "representation_gate": representation_path,
        },
    )

    report = audit_feasibility_results(
        tmp_path, allow_incomplete=True, write_report=False
    )
    assert report["ok"] is False
    assert "seed_average_value" in _codes(report)
