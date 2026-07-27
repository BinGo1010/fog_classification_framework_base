from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.residual_classifiers import CANONICAL_CLASSIFIER_NAMES
from run_daphnet_residual_classifier_suite import (
    HISTORY_BLOCKS,
    HISTORY_SAMPLES,
    INPUT_NAME,
    SUMMARY_METRICS,
    build_protocol,
    classifier_protocol,
    refresh_summaries,
    train_classifier_resumable,
)
from run_daphnet_tcn_rf_ablation import (
    CLASSIFICATION_METRICS,
    EXPECTED_CHANNEL_NAMES,
    EXPECTED_LOSO_SUBJECTS,
)
from start_daphnet_residual_classifier_suite_multigpu import (
    CANONICAL_CLASSIFIERS,
    CANONICAL_FOLDS,
    RUNNER,
    OutputDirectoryLock,
    base_command,
    parse_folds,
    parse_gpu_ids,
    paths_overlap,
)


EXPECTED_PARAMETER_COUNTS = {
    "mlp": 92_241,
    "cnn1d": 85_857,
    "gru": 90_035,
    "transformer": 86_355,
}


def _args(tmp_path: Path, *, debug_small_models: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        seed=42,
        classifier_dropout=0.15,
        classifier_epochs=2,
        classifier_patience=4,
        classifier_lr=1e-3,
        weight_decay=1e-4,
        batch_size=4,
        max_classifier_windows=0,
        num_workers=0,
        device="cpu",
        amp=False,
        deterministic=True,
        resume=True,
        debug_interrupt_classifier_after_epoch=0,
        debug_small_models=debug_small_models,
    )


def _synthetic_dataset_and_windows(
    tmp_path: Path,
) -> tuple[DaphnetDataset, WindowTable]:
    rng = np.random.default_rng(7)
    n_windows = 32
    starts = np.arange(n_windows, dtype=np.int32) * 16
    labels = (np.arange(n_windows) % 3 == 0).astype(np.int8)
    signal_length = int(starts[-1] + 64)
    record_y = np.zeros(signal_length, dtype=np.int8)
    for start, label in zip(starts, labels):
        if label:
            record_y[int(start) : int(start) + 32] = 1
    record = Record(
        record_id="synthetic",
        subject_id="S01",
        run_id="R01",
        x=rng.normal(size=(signal_length, 9)).astype(np.float32),
        y=record_y,
        valid=np.ones(signal_length, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=tmp_path,
        records=[record],
        sampling_rate_hz=64,
        channel_names=EXPECTED_CHANNEL_NAMES,
    )
    windows = WindowTable(
        record_index=np.zeros(n_windows, dtype=np.int32),
        start=starts,
        target_start=starts,
        target_end=starts + 32,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    return dataset, windows


def test_protocol_has_four_stable_models_and_reportable_parameter_counts(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    first = classifier_protocol(args, in_channels=9)
    second = classifier_protocol(args, in_channels=9)

    assert tuple(item["classifier"] for item in first) == (
        "mlp",
        "cnn1d",
        "gru",
        "transformer",
    )
    assert tuple(CANONICAL_CLASSIFIERS) == CANONICAL_CLASSIFIER_NAMES
    assert {
        item["classifier"]: item["parameter_count"] for item in first
    } == EXPECTED_PARAMETER_COUNTS
    assert [
        item["protocol_initial_state_sha256"] for item in first
    ] == [
        item["protocol_initial_state_sha256"] for item in second
    ]
    assert len(
        {item["protocol_initial_state_sha256"] for item in first}
    ) == 4
    for item in first:
        assert item["architecture"]["canonical_name"] == item["classifier"]
        assert item["architecture"]["input_samples"] == HISTORY_SAMPLES
        assert item["architecture"]["in_channels"] == 9
        assert item["architecture"]["parameter_count"] == item["parameter_count"]


def test_protocol_records_the_strict_fairness_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)
    dataset, windows = _synthetic_dataset_and_windows(tmp_path)
    source_config = {
        "excluded_subjects": ["S04", "S10"],
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
    }
    protocol = build_protocol(
        args,
        source_manifest={"source_protocol_fingerprint": "a" * 64},
        source_config=source_config,
        dataset=dataset,
        windows=windows,
        data_sha256="b" * 64,
        device=torch.device("cpu"),
    )

    contract = protocol["fairness_contract"]
    assert contract["ablation_axis"] == "downstream_classifier_architecture"
    assert contract["same_classifier_seed_within_fold"] is True
    assert contract["same_epoch_shuffle_within_fold"] is True
    assert contract["epoch_shuffle_seed_rule"] == "classifier_seed + epoch"
    assert contract["threshold_source"] == "validation_only_balanced_accuracy"
    assert {
        "source_persistence_residual_cache",
        "residual_h4s_window_ids_and_labels",
        "training_validation_test_support",
        "training_subsample",
        "optimizer",
        "class_weight",
        "early_stopping",
    }.issubset(contract["shared_fields"])
    assert protocol["folds_resolved"] == list(EXPECTED_LOSO_SUBJECTS)
    assert protocol["history_samples"] == 256
    assert protocol["history_blocks"] == 8
    assert protocol["debug_small_models"] is False
    assert len(protocol["protocol_fingerprint"]) == 64


def test_debug_models_are_explicit_and_do_not_change_model_names(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, debug_small_models=True)
    definitions = classifier_protocol(args, in_channels=9)
    assert [item["classifier"] for item in definitions] == list(
        CANONICAL_CLASSIFIER_NAMES
    )
    assert all(
        item["parameter_count"] < EXPECTED_PARAMETER_COUNTS[item["classifier"]]
        for item in definitions
    )
    by_name = {item["classifier"]: item["architecture"] for item in definitions}
    assert by_name["mlp"]["hidden_features"] == 8
    assert by_name["cnn1d"]["branch_channels"] == 4
    assert by_name["gru"]["hidden_size"] == 8
    assert by_name["transformer"]["model_dim"] == 8


def test_tiny_mlp_training_resumes_after_epoch_boundary_interrupt(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, debug_small_models=True)
    args.classifier_dropout = 0.0
    args.debug_interrupt_classifier_after_epoch = 1
    dataset, windows = _synthetic_dataset_and_windows(tmp_path)
    rng = np.random.default_rng(11)
    split_rows = {
        "train": np.arange(0, 16, dtype=np.int64),
        "validation": np.arange(16, 24, dtype=np.int64),
        "test": np.arange(24, 32, dtype=np.int64),
    }
    inputs = {
        split: {
            INPUT_NAME: rng.normal(
                size=(len(rows), 9, HISTORY_SAMPLES)
            ).astype(np.float32),
            "y": windows.label[rows],
            "window_index": rows,
        }
        for split, rows in split_rows.items()
    }
    definition = next(
        item
        for item in classifier_protocol(args, in_channels=9)
        if item["classifier"] == "mlp"
    )
    config = {"protocol_fingerprint": "c" * 64}
    fold_config = {
        "test_subject": "S01",
        "val_subject": "S02",
        "train_subjects": ["S03"],
        "classifier_seed": 10042,
        "source": {
            "source_residual_cache_sha256": "d" * 64,
            "input_support_sha256": "e" * 64,
        },
    }
    task_root = tmp_path / "loso_S01" / "mlp"
    try:
        train_classifier_resumable(
            args,
            config,
            definition,
            task_root,
            fold_config,
            inputs,
            dataset,
            windows,
            torch.device("cpu"),
        )
    except RuntimeError as error:
        assert "Intentional classifier interruption" in str(error)
    else:
        raise AssertionError("testing interruption hook did not interrupt")
    assert (task_root / "classifier_last.pt").exists()
    assert not (task_root / "DONE.json").exists()

    metrics = train_classifier_resumable(
        args,
        config,
        definition,
        task_root,
        fold_config,
        inputs,
        dataset,
        windows,
        torch.device("cpu"),
    )
    assert metrics["classifier"] == "mlp"
    assert metrics["parameter_count"] == definition["parameter_count"]
    assert [row["epoch"] for row in metrics["history"]] == [1, 2]
    assert [row["shuffle_seed"] for row in metrics["history"]] == [
        10043,
        10044,
    ]
    assert metrics["source_residual_sha256"] == "d" * 64
    assert metrics["input_support_sha256"] == "e" * 64
    assert (task_root / "DONE.json").exists()
    with np.load(task_root / "predictions.npz", allow_pickle=False) as payload:
        assert payload["y_prob"].dtype == np.float64
        np.testing.assert_array_equal(
            payload["window_index"], split_rows["test"]
        )

    # A completed cell must take the validated DONE path without retraining.
    resumed = train_classifier_resumable(
        args,
        config,
        definition,
        task_root,
        fold_config,
        inputs,
        dataset,
        windows,
        torch.device("cpu"),
    )
    assert resumed == metrics


def test_partial_summary_has_stable_four_model_schema(tmp_path: Path) -> None:
    args = _args(tmp_path)
    definitions = classifier_protocol(args, in_channels=9)
    config = {
        "protocol_fingerprint": "f" * 64,
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
        "classifiers": definitions,
    }
    y_true = np.asarray([0, 1, 0, 1], dtype=np.int8)
    model_predictions = {
        "mlp": np.asarray([0, 0, 0, 1], dtype=np.int8),
        "cnn1d": np.asarray([0, 1, 0, 1], dtype=np.int8),
    }
    for name, y_pred in model_predictions.items():
        definition = next(
            item for item in definitions if item["classifier"] == name
        )
        task_root = tmp_path / "loso_S01" / name
        task_root.mkdir(parents=True)
        y_prob = np.where(y_pred == 1, 0.8, 0.2).astype(np.float64)
        metrics = {
            "experiment_id": definition["experiment_id"],
            "classifier": name,
            "display_name": definition["display_name"],
            "nbm": "persistence",
            "input": INPUT_NAME,
            "history_seconds": 4.0,
            "history_samples": HISTORY_SAMPLES,
            "history_blocks": HISTORY_BLOCKS,
            "test_subject": "S01",
            "val_subject": "S02",
            "classifier_seed": 10042,
            "parameter_count": definition["parameter_count"],
            "threshold": 0.5,
            "n": 4,
            "n_normal": 2,
            "n_fog": 2,
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
            "best_epoch": 1,
            "best_validation_auprc": 0.75,
            "initial_state_sha256": "1" * 64,
            "source_residual_sha256": "2" * 64,
            "input_support_sha256": "3" * 64,
        }
        # Values need only be finite here; pooled metrics are independently
        # recomputed from the prediction arrays by refresh_summaries.
        metrics.update(
            {
                metric: (
                    float((y_true == y_pred).mean())
                    if metric not in {
                        "false_alarm_events_per_hour",
                        "median_detection_delay_sec",
                    }
                    else 0.0
                )
                for metric in CLASSIFICATION_METRICS
            }
        )
        (task_root / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        np.savez(
            task_root / "predictions.npz",
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            window_index=np.arange(4, dtype=np.int64),
        )
        (task_root / "DONE.json").write_text("{}", encoding="utf-8")

    refresh_summaries(tmp_path, config)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status == {
        "suite_version": "daphnet_persistence_h4_residual_classifier_suite.v1",
        "protocol_fingerprint": "f" * 64,
        "expected_experiments": 4,
        "expected_fold_cells": 32,
        "completed_fold_cells": 2,
        "status": "partial",
    }
    with (tmp_path / "experiment_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    assert [row["classifier"] for row in manifest] == list(
        CANONICAL_CLASSIFIER_NAMES
    )
    assert [row["status"] for row in manifest] == [
        "partial",
        "partial",
        "pending",
        "pending",
    ]
    with (tmp_path / "aggregate_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert [row["classifier"] for row in summary] == ["mlp", "cnn1d"]
    assert set(summary[0]) == {
        "classifier",
        "display_name",
        "parameter_count",
        "completed_folds",
        *{
            f"{metric}_{statistic}"
            for metric in SUMMARY_METRICS
            for statistic in ("mean", "std")
        },
    }
    aggregate = json.loads(
        (tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    delta = aggregate["paired_deltas_vs_mlp"]["cnn1d"]
    assert delta["common_subjects"] == ["S01"]
    assert (
        delta["metrics"]["balanced_accuracy"]["n_paired_folds"] == 1
    )


def test_scheduler_helpers_and_common_command(tmp_path: Path) -> None:
    assert parse_gpu_ids("0-2,6") == ["0", "1", "2", "6"]
    assert parse_folds("S09,S01,S05") == ["S01", "S05", "S09"]
    assert parse_folds("all") == list(CANONICAL_FOLDS)
    assert paths_overlap(tmp_path, tmp_path / "child")

    args = SimpleNamespace(
        data_dir=tmp_path / "processed",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
    )
    command = base_command(
        args,
        ["--seed", "42", "--batch-size", "256"],
    )
    assert command[:3] == [sys.executable, "-u", str(RUNNER)]
    assert "--resume" in command
    assert command[command.index("--folds") + 1] == "all"
    assert command[-4:] == ["--seed", "42", "--batch-size", "256"]

    first = OutputDirectoryLock(args.output_dir)
    second = OutputDirectoryLock(args.output_dir)
    args.output_dir.mkdir()
    first.acquire()
    try:
        assert first.path.name == ".residual_classifier_scheduler.lock"
        try:
            second.acquire()
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate scheduler lock was accepted")
    finally:
        first.release()
    second.acquire()
    second.release()
