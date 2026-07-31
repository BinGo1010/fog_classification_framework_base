from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_gru_residual_feasibility as runner


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_defaults_to_safe_phase0_and_preregistered_phase3_seeds() -> None:
    args = runner.parse_args([])
    assert args.phase == "0"
    assert args.phase3_nbm_seeds == "42"
    assert args.phase3_classifier_seeds is None
    assert args.phase0_plots is True
    runner.validate_args(args)

    duplicate = runner.parse_args(["--phase3-nbm-seeds", "42,42"])
    with pytest.raises(ValueError, match="unique"):
        runner.validate_args(duplicate)


def test_phase1_aggregate_requires_only_fixed_s01_and_applies_smoke_gate(
    tmp_path: Path,
) -> None:
    arms = runner.PHASE1_ARMS
    common = {
        metric: 0.5 for metric in runner.METRIC_NAMES
    }
    for arm in arms:
        payload = {
            **common,
            "phase": "1",
            "arm": arm,
            "display_name": arm,
            "test_subject": "S01",
            "val_subject": "S02",
            "threshold": 0.5,
            "n": 20,
            "n_normal": 15,
            "n_fog": 5,
            "parameter_count": (
                200 if arm in {"raw4_zero", "raw4_normality"} else 100
            ),
            "initial_state_sha256": (
                "same" if arm in {"raw4_zero", "raw4_normality"} else arm
            ),
            "best_epoch": 2,
            "best_validation_pr_auc": 0.5,
            "history": [
                {"epoch": 1, "train_loss": 1.0},
                {"epoch": 2, "train_loss": 0.8},
            ],
        }
        _write_json(
            tmp_path / "phase1" / "loso_S01" / arm / "metrics.json",
            payload,
        )
    args = SimpleNamespace(
        output_dir=tmp_path,
        bootstrap_samples=200,
        bootstrap_seed=42,
    )
    protocol = SimpleNamespace(
        folds=runner.EXPECTED_SUBJECTS,
        config={"protocol_fingerprint": "toy"},
    )
    aggregate = runner.aggregate_classifier_phase(
        args, protocol, "1", arms
    )
    assert aggregate["completed"] is True
    assert aggregate["folds"] == ["S01"]
    assert aggregate["smoke_gate"]["status"] == "pass"
    assert aggregate["engineering_smoke"] is True
    comparison = aggregate["comparisons"][
        "raw4_normality_minus_raw4_zero"
    ]
    assert comparison["subjects"] == ["S01"]


def test_phase0_subset_is_complete_but_not_a_reportable_gate(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "phase0" / "loso_S01" / "metrics.json",
        {
            "subject": "S01",
            "gru_better_rmse": True,
            "hard_checks": {"identities": True},
            "diagnostic_warnings": {},
            "clean_nonfog_z_clip_rate": 0.01,
            "gru": {"overall": {"rmse": 1.0}},
            "persistence": {"overall": {"rmse": 2.0}},
        },
    )
    args = SimpleNamespace(output_dir=tmp_path)
    protocol = SimpleNamespace(
        folds=("S01",), config={"protocol_fingerprint": "toy"}
    )
    aggregate = runner.aggregate_phase0(args, protocol)
    assert aggregate["completed"] is True
    assert aggregate["decision"] == "subset_only"
    assert aggregate["persistence_gate_pass"] is False


def test_staged_gate_block_writes_machine_readable_status(tmp_path: Path) -> None:
    args = SimpleNamespace(
        output_dir=tmp_path,
        phase="all",
        force_next_phase=False,
    )
    protocol = SimpleNamespace(
        folds=("S01",), config={"protocol_fingerprint": "toy"}
    )
    with pytest.raises(RuntimeError, match="stopped after Phase 0"):
        runner._block_staged_run(
            args, protocol, phase="0", reason="synthetic failure"
        )
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "blocked_by_stage_gate"
    assert status["blocked_after_phase"] == "0"

