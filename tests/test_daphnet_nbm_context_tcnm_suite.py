from __future__ import annotations

import csv
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

import run_daphnet_nbm_context_tcnm_suite as suite
import start_daphnet_nbm_context_tcnm_suite_multigpu as multigpu
from cnbr_fog.data import WindowTable
from cnbr_fog.histories import make_common_history_plan, make_history_input
from cnbr_fog.resume import done_payload, sha256_file


def _classifier_args() -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        deterministic=True,
        classifier_hidden=48,
        classifier_dropout=0.15,
    )


def _summary_config(
    experiments: list[dict],
    folds: list[str],
    *,
    bootstrap_samples: int = 256,
) -> dict:
    return {
        "folds_resolved": folds,
        "experiments": experiments,
        "protocol_fingerprint": "a" * 64,
        "horizon_seconds": 0.5,
        "history_seconds": 4.0,
        "history_samples": 256,
        "classifier": suite.classifier_architecture(
            _classifier_args(),
            in_channels=9,
            sampling_rate_hz=64,
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": 42,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_completed_cell(
    root: Path,
    experiment: dict,
    subject: str,
    pr_auc: float,
) -> None:
    support_path = root / f"loso_{subject}" / "history_support.npz"
    support_path.parent.mkdir(parents=True, exist_ok=True)
    if not support_path.exists():
        np.savez(
            support_path,
            anchor_window_indices=np.asarray([10, 11], dtype=np.int64),
        )
    support_sha256 = sha256_file(support_path)

    source_root = suite.nbm_root_for(root, subject, experiment)
    nbm_stage = source_root / "nbm"
    nbm_stage.mkdir(parents=True, exist_ok=True)
    best_path = nbm_stage / "best.pt"
    best_path.write_bytes(b"synthetic-nbm-checkpoint")
    nbm_sha256 = sha256_file(best_path)
    nbm_completed = done_payload(
        stage="nbm",
        protocol_fingerprint="a" * 64,
        task_id=(
            f"{suite.context_task_directory(experiment, subject)}/"
            f"{experiment['nbm']}/nbm"
        ),
        relative_to=nbm_stage,
        artifacts={"best": best_path},
    )
    (nbm_stage / "DONE.json").write_text(
        json.dumps(nbm_completed),
        encoding="utf-8",
    )

    residual_path = source_root / "residual_cache.npz"
    diagnostics_path = source_root / "residual_diagnostics.json"
    np.savez(
        residual_path,
        test_residual=np.zeros((2, 9, 32), dtype=np.float32),
    )
    diagnostics_path.write_text("{}", encoding="utf-8")
    residual_sha256 = sha256_file(residual_path)
    residual_completed = done_payload(
        stage="residual_cache",
        protocol_fingerprint="a" * 64,
        task_id=(
            f"{suite.context_task_directory(experiment, subject)}/"
            f"{experiment['nbm']}/residual_cache"
        ),
        upstream_sha256=nbm_sha256,
        relative_to=source_root,
        artifacts={
            "cache": residual_path,
            "diagnostics": diagnostics_path,
        },
    )
    (source_root / "RESIDUAL_CACHE_DONE.json").write_text(
        json.dumps(residual_completed),
        encoding="utf-8",
    )

    task_root = suite.task_root_for(root, subject, experiment)
    task_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment_id": experiment["experiment_id"],
        "nbm": experiment["nbm"],
        "input": suite.HISTORY_NAME,
        "history_seconds": 4.0,
        "history_samples": 256,
        "history_blocks": 8,
        "test_subject": subject,
        "val_subject": "S09",
        "classifier_seed": 10042,
        "pr_auc": float(pr_auc),
        "source_residual_sha256": residual_sha256,
        "input_support_sha256": support_sha256,
    }
    metrics_path = task_root / "metrics.json"
    predictions_path = task_root / "predictions.npz"
    validation_predictions_path = task_root / "validation_predictions.npz"
    best_path = task_root / "classifier_best.pt"
    last_path = task_root / "classifier_last.pt"
    predictions_csv_path = task_root / "predictions.csv"
    metrics_path.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    np.savez(
        predictions_path,
        window_index=np.asarray([10, 11], dtype=np.int64),
        y_true=np.asarray([0, 1], dtype=np.int8),
        y_prob=np.asarray([0.1, 0.9], dtype=np.float64),
        y_pred=np.asarray([0, 1], dtype=np.int8),
    )
    np.savez(
        validation_predictions_path,
        window_index=np.asarray([20, 21], dtype=np.int64),
        y_true=np.asarray([0, 1], dtype=np.int8),
        y_prob=np.asarray([0.2, 0.8], dtype=np.float64),
        y_pred=np.asarray([0, 1], dtype=np.int8),
    )
    best_path.write_bytes(b"synthetic-classifier-best")
    last_path.write_bytes(b"synthetic-classifier-last")
    predictions_csv_path.write_text(
        "window_index,y_true,y_prob,y_pred\n10,0,0.1,0\n11,1,0.9,1\n",
        encoding="utf-8",
    )
    completed = done_payload(
        stage="rf_classifier",
        protocol_fingerprint="a" * 64,
        task_id=f"{subject}/{experiment['experiment_id']}",
        relative_to=task_root,
        artifacts={
            "best": best_path,
            "last": last_path,
            "metrics": metrics_path,
            "predictions": predictions_path,
            "validation_predictions": validation_predictions_path,
            "predictions_csv": predictions_csv_path,
        },
    )
    completed["source_residual_sha256"] = residual_sha256
    completed["input_support_sha256"] = support_sha256
    (task_root / "DONE.json").write_text(
        json.dumps(completed),
        encoding="utf-8",
    )


def test_parse_contexts_and_sixteen_experiment_grid() -> None:
    contexts = suite.parse_contexts("4,1,3,2,2", sampling_rate_hz=64)

    assert [
        (
            item["context_id"],
            item["context_seconds"],
            item["context_samples"],
            item["directory"],
        )
        for item in contexts
    ] == [
        ("C1", 1.0, 64, "context_c1_1s"),
        ("C2", 2.0, 128, "context_c2_2s"),
        ("C3", 3.0, 192, "context_c3_3s"),
        ("C4", 4.0, 256, "context_c4_4s"),
    ]

    grid = suite.experiment_grid(list(suite.DEFAULT_NBMS), contexts)
    assert len(grid) == 16
    assert len({item["experiment_id"] for item in grid}) == 16
    assert {
        (item["nbm"], item["context_id"])
        for item in grid
    } == {
        (nbm, context_id)
        for nbm in suite.DEFAULT_NBMS
        for context_id in ("C1", "C2", "C3", "C4")
    }
    assert all(item["experiment_id"].endswith("__residual_h4s__tcn_m") for item in grid)
    first = grid[0]
    assert suite.context_task_directory(first, "S01") == (
        "context_c1_1s__loso_s01"
    )
    assert suite.context_task_directory(first, "S02") != (
        suite.context_task_directory(first, "S01")
    )

    with pytest.raises(ValueError, match="preregistered contexts"):
        suite.parse_contexts("0.5,1,2,3,4", sampling_rate_hz=64)


def test_strict_protocol_rejects_subsets_before_training() -> None:
    contexts = suite.parse_contexts("1,2,3,4", sampling_rate_hz=64)
    suite.validate_protocol_selection(
        list(suite.DEFAULT_NBMS),
        contexts,
        list(suite.EXPECTED_LOSO_SUBJECTS),
        sampling_rate_hz=64,
        allow_subset=False,
    )
    with pytest.raises(ValueError, match="--nbms"):
        suite.validate_protocol_selection(
            ["linear_ar"],
            contexts,
            list(suite.EXPECTED_LOSO_SUBJECTS),
            sampling_rate_hz=64,
            allow_subset=False,
        )
    with pytest.raises(ValueError, match="--context-seconds"):
        suite.validate_protocol_selection(
            list(suite.DEFAULT_NBMS),
            contexts[:2],
            list(suite.EXPECTED_LOSO_SUBJECTS),
            sampling_rate_hz=64,
            allow_subset=False,
        )
    with pytest.raises(ValueError, match="--folds all"):
        suite.validate_protocol_selection(
            list(suite.DEFAULT_NBMS),
            contexts,
            ["S01", "S01"],
            sampling_rate_hz=64,
            allow_subset=False,
        )
    suite.validate_protocol_selection(
        ["linear_ar"],
        contexts[:1],
        ["S01"],
        sampling_rate_hz=64,
        allow_subset=True,
    )


def test_context_target_split_is_exactly_right_aligned() -> None:
    sequence = torch.arange(
        2 * 9 * 288,
        dtype=torch.float32,
    ).reshape(2, 9, 288)
    expected_target = sequence[:, :, 256:288]

    for context_samples in (64, 128, 192, 256):
        context, target = suite.context_target_split(
            sequence,
            context_samples=context_samples,
            horizon_samples=32,
        )
        assert context.shape == (2, 9, context_samples)
        assert target.shape == (2, 9, 32)
        torch.testing.assert_close(
            context,
            sequence[:, :, 256 - context_samples : 256],
        )
        torch.testing.assert_close(target, expected_target)

    with pytest.raises(ValueError, match="required"):
        suite.context_target_split(
            sequence[:, :, :95],
            context_samples=64,
            horizon_samples=32,
        )


def test_four_second_history_plan_has_eight_blocks_and_9_by_256_input() -> None:
    block_count = 8
    target_starts = np.arange(block_count, dtype=np.int32) * 32
    windows = WindowTable(
        record_index=np.zeros(block_count, dtype=np.int32),
        start=target_starts.copy(),
        target_start=target_starts,
        target_end=target_starts + 32,
        label=np.asarray([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8),
        fog_fraction=np.asarray(
            [0, 0, 0, 0, 0, 0, 0, 1],
            dtype=np.float32,
        ),
        clean_normal=np.asarray(
            [True, True, True, True, True, True, True, False],
            dtype=bool,
        ),
    )
    indices = np.arange(block_count, dtype=np.int64)
    plan = make_common_history_plan(
        windows,
        indices,
        horizon_samples=32,
        stride_samples=16,
        max_history_samples=256,
    )

    assert plan.anchor_rows.tolist() == [7]
    assert plan.anchor_window_indices.tolist() == [7]
    np.testing.assert_array_equal(
        plan.max_chain_rows,
        np.arange(8, dtype=np.int64)[None, :],
    )

    residual = np.stack(
        [
            np.full((9, 32), fill_value=block, dtype=np.float32)
            for block in range(8)
        ],
        axis=0,
    )
    history = make_history_input(
        {
            "residual": residual,
            "y": windows.label,
            "window_index": indices,
        },
        plan,
        suite.HISTORY_NAME,
        history_samples=256,
        horizon_samples=32,
        stride_samples=16,
    )

    assert history[suite.HISTORY_NAME].shape == (1, 9, 256)
    assert history["y"].tolist() == [1]
    assert history["window_index"].tolist() == [7]
    for block in range(8):
        np.testing.assert_array_equal(
            history[suite.HISTORY_NAME][0, :, block * 32 : (block + 1) * 32],
            np.full((9, 32), block, dtype=np.float32),
        )


def test_tcn_m_architecture_is_context_invariant_rf125() -> None:
    contexts = suite.parse_contexts("1,2,3,4", sampling_rate_hz=64)
    architectures = [
        suite.classifier_architecture(
            _classifier_args(),
            in_channels=9,
            sampling_rate_hz=64,
        )
        for _ in contexts
    ]

    assert {tuple(item["dilations"]) for item in architectures} == {
        (1, 2, 4, 8, 8, 8)
    }
    assert {item["n_blocks"] for item in architectures} == {6}
    assert {item["convolutions_per_block"] for item in architectures} == {2}
    assert {item["receptive_field_samples"] for item in architectures} == {125}
    assert all(
        item["receptive_field_seconds"] == pytest.approx(125 / 64)
        for item in architectures
    )
    assert {item["parameter_count"] for item in architectures} == {89_329}


def test_paired_bootstrap_is_deterministic_for_subject_paired_differences() -> None:
    subjects = list(suite.EXPECTED_LOSO_SUBJECTS)
    reference = {
        subject: value
        for subject, value in zip(
            subjects,
            (0.20, 0.70, 0.40, 0.80, 0.30, 0.55, 0.60, 0.35),
        )
    }
    current = {
        subject: reference[subject] + delta
        for subject, delta in zip(
            subjects,
            (0.10, -0.30, 0.50, -0.20, 0.05, 0.25, -0.15, 0.35),
        )
    }
    paired = np.asarray(
        [current[subject] - reference[subject] for subject in subjects],
        dtype=np.float64,
    )

    first = suite.paired_bootstrap_mean_ci(paired, samples=4096, seed=2026)
    second = suite.paired_bootstrap_mean_ci(paired, samples=4096, seed=2026)
    assert first == second
    assert first["n_paired_subjects"] == len(subjects)
    assert first["bootstrap_samples"] == 4096
    assert first["mean_delta"] == pytest.approx(float(paired.mean()))
    assert first["ci_low"] <= first["mean_delta"] <= first["ci_high"]
    assert suite.stable_bootstrap_seed(42, "C1-vs-C2") == (
        suite.stable_bootstrap_seed(42, "C1-vs-C2")
    )
    assert suite.stable_bootstrap_seed(42, "C1-vs-C2") != (
        suite.stable_bootstrap_seed(42, "C3-vs-C2")
    )


def test_refresh_summaries_pairs_pr_auc_by_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = suite.parse_contexts("1,2", sampling_rate_hz=64)
    experiments = suite.experiment_grid(["gru"], contexts)
    folds = ["S01", "S02", "S03"]
    c1 = next(item for item in experiments if item["context_id"] == "C1")
    c2 = next(item for item in experiments if item["context_id"] == "C2")
    c1_scores = {"S01": 0.20, "S02": 0.50, "S03": 0.90}
    c2_scores = {"S01": 0.10, "S02": 0.80, "S03": 0.40}

    # Deliberately create cells in different orders. Summary pairing must use
    # subject ids rather than file-system or completion order.
    for subject in reversed(folds):
        _write_completed_cell(tmp_path, c1, subject, c1_scores[subject])
    for subject in folds:
        _write_completed_cell(tmp_path, c2, subject, c2_scores[subject])

    captured: list[np.ndarray] = []
    original = suite.paired_bootstrap_mean_ci

    def recording_bootstrap(
        differences: np.ndarray,
        samples: int,
        seed: int,
    ) -> dict:
        captured.append(np.asarray(differences, dtype=np.float64).copy())
        return original(differences, samples, seed)

    monkeypatch.setattr(suite, "paired_bootstrap_mean_ci", recording_bootstrap)
    suite.refresh_summaries(
        tmp_path,
        _summary_config(experiments, folds, bootstrap_samples=128),
    )

    np.testing.assert_allclose(captured[0], [0.10, -0.30, 0.50])
    np.testing.assert_allclose(captured[1], [0.0, 0.0, 0.0])
    delta_rows = {
        row["experiment_id"]: row
        for row in _read_csv(tmp_path / "paired_pr_auc_deltas.csv")
    }
    assert delta_rows[c1["experiment_id"]]["common_subjects"] == "S01,S02,S03"
    assert float(delta_rows[c1["experiment_id"]]["mean_delta"]) == pytest.approx(
        0.10
    )
    assert delta_rows[c2["experiment_id"]]["common_subjects"] == "S01,S02,S03"
    assert float(delta_rows[c2["experiment_id"]]["mean_delta"]) == 0.0


def test_refresh_summaries_rejects_stale_upstream_support(
    tmp_path: Path,
) -> None:
    context = suite.parse_contexts("1", sampling_rate_hz=64)
    experiment = suite.experiment_grid(["gru"], context)[0]
    _write_completed_cell(tmp_path, experiment, "S01", 0.5)

    support_path = tmp_path / "loso_S01" / "history_support.npz"
    support_path.write_bytes(b"changed-after-classifier-completed")

    with pytest.raises(ValueError, match="input_support_sha256"):
        suite.refresh_summaries(
            tmp_path,
            _summary_config([experiment], ["S01"]),
        )


def test_empty_publication_and_summary_are_sixteen_by_eight_pending(
    tmp_path: Path,
) -> None:
    contexts = suite.parse_contexts("1,2,3,4", sampling_rate_hz=64)
    experiments = suite.experiment_grid(list(suite.DEFAULT_NBMS), contexts)
    folds = list(suite.EXPECTED_LOSO_SUBJECTS)
    config = _summary_config(experiments, folds, bootstrap_samples=64)

    suite.refresh_summaries(tmp_path, config)

    manifest = _read_csv(tmp_path / "experiment_manifest.csv")
    summary = _read_csv(tmp_path / "aggregate_summary.csv")
    publication = _read_csv(tmp_path / "publication_table.csv")
    deltas = _read_csv(tmp_path / "paired_pr_auc_deltas.csv")
    fold_rows = _read_csv(tmp_path / "fold_summary.csv")
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    aggregate = json.loads(
        (tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8")
    )

    assert len(manifest) == len(summary) == len(publication) == len(deltas) == 16
    assert fold_rows == []
    assert {row["status"] for row in manifest} == {"pending"}
    assert {int(row["expected_folds"]) for row in manifest} == {8}
    assert {int(row["completed_folds"]) for row in manifest} == {0}
    assert {row["completed_subjects"] for row in manifest} == {""}
    assert {int(row["Completed folds"]) for row in publication} == {0}
    assert all(row["PR-AUC"] == "" for row in publication)
    assert all(row["ΔPR-AUC [95% CI]"] == "" for row in publication)
    assert all(row["pr_auc_mean"] == "" for row in summary)

    assert status == {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": "a" * 64,
        "expected_experiments": 16,
        "expected_nbm_tasks": 128,
        "expected_classifier_cells": 128,
        "completed_classifier_cells": 0,
        "status": "partial",
        "best_experiment": None,
    }
    assert aggregate["best_experiment"] is None
    assert len(aggregate["experiments"]) == 16
    assert all(
        payload["completed_folds"] == []
        for payload in aggregate["experiments"].values()
    )


def test_multigpu_wrapper_dry_run_builds_context_suite_commands(
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
            "--context-seconds",
            "1,2,3,4",
            "--seed",
            "42",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert f"scheduler_version={multigpu.SCHEDULER_VERSION}" in output
    assert "run_daphnet_nbm_context_tcnm_suite.py" in output
    assert "worker[S01].env.CUDA_VISIBLE_DEVICES=0" in output
    assert "worker[S02].env.CUDA_VISIBLE_DEVICES=1" in output
    assert "--worker-fold S01" in output
    assert "--worker-fold S02" in output
    assert "--context-seconds 1,2,3,4" in output
    assert "--finalize-only" in output
    assert "audit=(disabled)" in output
    assert not output_dir.exists()
