from __future__ import annotations

import copy
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

from audit_daphnet_rf125_cnn_replacement import configured_auditor
from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.resume import canonical_fingerprint
from cnbr_fog.rf125_classifiers import CANONICAL_RF125_CLASSIFIER_NAMES
from run_daphnet_rf125_cnn_replacement import (
    CLASSIFIER_STAGE,
    PAIR_AGGREGATE_KEY,
    configured_base_suite,
    paired_delta_summary,
)
from run_daphnet_tcn_rf_ablation import (
    CLASSIFICATION_METRICS,
    EXPECTED_CHANNEL_NAMES,
    EXPECTED_LOSO_SUBJECTS,
    HISTORY_BLOCKS,
    HISTORY_SAMPLES,
    INPUT_NAME,
)
from start_daphnet_rf125_cnn_replacement_multigpu import (
    AUDITOR,
    LOCK_FILENAME,
    RUNNER,
    SCHEDULER_VERSION,
    configured_scheduler,
)


def _args(
    tmp_path: Path,
    *,
    debug_small_models: bool = False,
    epochs: int = 12,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        seed=42,
        classifier_dropout=0.15,
        classifier_epochs=epochs,
        classifier_patience=4,
        classifier_lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_classifier_windows=0,
        num_workers=0,
        device="cpu",
        amp=True,
        deterministic=True,
        resume=True,
        debug_interrupt_classifier_after_epoch=0,
        debug_small_models=debug_small_models,
    )


def _synthetic_dataset_and_windows(
    tmp_path: Path,
) -> tuple[DaphnetDataset, WindowTable]:
    rng = np.random.default_rng(17)
    records: list[Record] = []
    for subject in EXPECTED_LOSO_SUBJECTS:
        records.append(
            Record(
                record_id=f"{subject}_R01",
                subject_id=subject,
                run_id="R01",
                x=rng.normal(size=(640, 9)).astype(np.float32),
                y=np.zeros(640, dtype=np.int8),
                valid=np.ones(640, dtype=bool),
            )
        )
    n_windows = 32
    starts = np.arange(n_windows, dtype=np.int32) * 16
    labels = (np.arange(n_windows) % 3 == 0).astype(np.int8)
    for start, label in zip(starts, labels):
        if label:
            records[0].y[int(start) : int(start) + 32] = 1
    dataset = DaphnetDataset(
        root=tmp_path,
        records=records,
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


def test_protocol_is_a_strict_two_arm_rf125_replacement(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    dataset, windows = _synthetic_dataset_and_windows(tmp_path)
    source_config = {
        "excluded_subjects": ["S04", "S10"],
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
    }
    with configured_base_suite() as suite:
        protocol = suite.build_protocol(
            args,
            source_manifest={"source_protocol_fingerprint": "a" * 64},
            source_config=source_config,
            dataset=dataset,
            windows=windows,
            data_sha256="b" * 64,
            device=torch.device("cpu"),
        )

    assert protocol["classifier_names"] == ["tcn_m", "cnn_rf125"]
    assert protocol["folds_resolved"] == list(EXPECTED_LOSO_SUBJECTS)
    assert protocol["excluded_subjects"] == ["S04", "S10"]
    assert protocol["nbm"] == "persistence"
    assert protocol["input"] == "residual_h4s"
    assert protocol["history_samples"] == 256
    assert protocol["history_blocks"] == 8
    assert protocol["comparison_axis"] == "residual_skip_connection"
    assert protocol["shared_parameter_count"] == 89_329
    assert protocol["shared_conv_linear_macs_per_window"] == 21_348_912
    assert protocol["tcn_extra_residual_additions_per_window"] == 73_728
    assert protocol["estimated_compute_delta_fraction"] < 0.005
    assert protocol["fairness_contract"]["same_parameter_schema"] is True
    assert (
        protocol["fairness_contract"][
            "same_initial_state_sha256_within_fold"
        ]
        is True
    )
    definitions = protocol["classifiers"]
    assert {
        item["parameter_count"] for item in definitions
    } == {89_329}
    assert len(
        {
            item["protocol_initial_state_sha256"]
            for item in definitions
        }
    ) == 1
    json.dumps(protocol)

    with configured_auditor() as auditor:
        folds, audited_definitions = auditor.validate_protocol(protocol)
    assert folds == list(EXPECTED_LOSO_SUBJECTS)
    assert [item["classifier"] for item in audited_definitions] == [
        "tcn_m",
        "cnn_rf125",
    ]

    no_amp = copy.deepcopy(protocol)
    no_amp["amp"] = False
    with configured_auditor() as auditor:
        no_amp["protocol_fingerprint"] = canonical_fingerprint(
            auditor.protocol_payload(no_amp)
        )
        with pytest.raises(AssertionError, match="AMP"):
            auditor.validate_protocol(no_amp)
    capped = copy.deepcopy(protocol)
    capped["max_classifier_windows"] = 128
    with configured_auditor() as auditor:
        capped["protocol_fingerprint"] = canonical_fingerprint(
            auditor.protocol_payload(capped)
        )
        with pytest.raises(AssertionError, match="every training anchor"):
            auditor.validate_protocol(capped)


def test_debug_training_resume_and_paired_summary(tmp_path: Path) -> None:
    args = _args(tmp_path, debug_small_models=True, epochs=2)
    args.classifier_dropout = 0.0
    args.batch_size = 4
    args.amp = False
    dataset, windows = _synthetic_dataset_and_windows(tmp_path)
    rng = np.random.default_rng(19)
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
    fold_config = {
        "test_subject": "S01",
        "val_subject": "S02",
        "train_subjects": ["S03"],
        "classifier_seed": 10042,
        "source": {
            "source_residual_cache_sha256": "c" * 64,
            "input_support_sha256": "d" * 64,
        },
    }
    config = {
        "protocol_fingerprint": "e" * 64,
        "folds_resolved": list(EXPECTED_LOSO_SUBJECTS),
    }

    with configured_base_suite() as suite:
        definitions = suite.classifier_protocol(args, in_channels=9)
        config["classifiers"] = definitions
        results: dict[str, dict] = {}
        for definition in definitions:
            name = definition["classifier"]
            task_root = tmp_path / f"loso_S01/{name}"
            if name == "cnn_rf125":
                args.debug_interrupt_classifier_after_epoch = 1
                with pytest.raises(
                    RuntimeError,
                    match="Intentional classifier interruption",
                ):
                    suite.train_classifier_resumable(
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
                assert (task_root / "classifier_last.pt").exists()
                assert not (task_root / "DONE.json").exists()
                args.debug_interrupt_classifier_after_epoch = 0
            results[name] = suite.train_classifier_resumable(
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

        assert (
            results["tcn_m"]["initial_state_sha256"]
            == results["cnn_rf125"]["initial_state_sha256"]
        )
        assert {
            results[name]["parameter_count"]
            for name in CANONICAL_RF125_CLASSIFIER_NAMES
        } == {definitions[0]["parameter_count"]}
        assert results["tcn_m"]["architecture"]["residual_skip"] is True
        assert (
            results["cnn_rf125"]["architecture"]["residual_skip"] is False
        )
        assert results["cnn_rf125"]["history"][0]["epoch"] == 1
        assert results["cnn_rf125"]["history"][0]["shuffle_seed"] == 10043

        suite.refresh_summaries(tmp_path, config)

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["expected_experiments"] == 2
    assert status["expected_fold_cells"] == 16
    assert status["completed_fold_cells"] == 2
    aggregate = json.loads(
        (tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    assert PAIR_AGGREGATE_KEY in aggregate
    paired = aggregate[PAIR_AGGREGATE_KEY]["cnn_rf125"]
    assert paired["reference"] == "tcn_m"
    assert paired["common_subjects"] == ["S01"]
    assert (
        paired["metrics"]["balanced_accuracy"]["n_paired_folds"] == 1
    )
    assert (tmp_path / "paired_fold_deltas.csv").exists()


def test_scheduler_wrapper_uses_two_models_and_unique_lock(
    tmp_path: Path,
) -> None:
    with configured_scheduler() as scheduler:
        assert scheduler.RUNNER == RUNNER
        assert scheduler.AUDITOR == AUDITOR
        assert scheduler.SCHEDULER_VERSION == SCHEDULER_VERSION
        assert tuple(scheduler.CANONICAL_CLASSIFIERS) == (
            "tcn_m",
            "cnn_rf125",
        )
        lock = scheduler.OutputDirectoryLock(tmp_path)
        assert lock.path.name == LOCK_FILENAME


def test_fixed_history_shape_constants() -> None:
    assert HISTORY_SAMPLES == 256
    assert HISTORY_BLOCKS == 8
    assert INPUT_NAME == "residual_h4s"
    assert CLASSIFIER_STAGE == "rf125_replacement_classifier"


def test_paired_wins_respect_metric_optimization_direction() -> None:
    reference = {metric: 0.0 for metric in CLASSIFICATION_METRICS}
    comparison = {metric: 0.0 for metric in CLASSIFICATION_METRICS}
    reference.update(
        {
            "balanced_accuracy": 0.5,
            "false_alarm_events_per_hour": 10.0,
            "median_detection_delay_sec": 2.0,
        }
    )
    comparison.update(
        {
            "balanced_accuracy": 0.6,
            "false_alarm_events_per_hour": 20.0,
            "median_detection_delay_sec": 1.0,
        }
    )
    result = paired_delta_summary(
        {
            "tcn_m": {"S01": reference},
            "cnn_rf125": {"S01": comparison},
        }
    )["cnn_rf125"]["metrics"]

    assert result["balanced_accuracy"]["wins"] == 1
    assert (
        result["balanced_accuracy"]["optimization_direction"]
        == "higher_is_better"
    )
    false_alarms = result["false_alarm_events_per_hour"]
    assert (
        false_alarms["mean_delta_cnn_rf125_minus_tcn_m"]
        == pytest.approx(10.0)
    )
    assert false_alarms["optimization_direction"] == "lower_is_better"
    assert false_alarms["wins"] == 0
    assert false_alarms["losses"] == 1
    delay = result["median_detection_delay_sec"]
    assert delay["mean_delta_cnn_rf125_minus_tcn_m"] == pytest.approx(-1.0)
    assert delay["optimization_direction"] == "lower_is_better"
    assert delay["wins"] == 1
    assert delay["losses"] == 0
