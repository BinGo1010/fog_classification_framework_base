from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.histories import make_common_history_plan, make_history_input
import run_daphnet_persistence_tcnm_stride_ablation as suite
import start_daphnet_persistence_tcnm_stride_ablation_multigpu as scheduler


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        folds="all",
        worker_fold="",
        finalize_only=False,
        seed=42,
        classifier_hidden=48,
        classifier_dropout=0.15,
        classifier_epochs=12,
        classifier_patience=4,
        classifier_lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_classifier_windows=0,
        bootstrap_samples=100_000,
        bootstrap_seed=42,
        num_workers=0,
        device="cpu",
        amp=True,
        deterministic=True,
        resume=True,
        debug_interrupt_classifier_after_epoch=0,
        stop_after_completed_tasks=0,
    )


def _source_config() -> dict:
    return {
        "suite_version": suite.SOURCE_SUITE_VERSION,
        "sampling_rate_hz": 64,
        "n_channels": 9,
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
        "normal_epochs": 8,
        "normal_patience": 3,
        "normal_lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_normal_windows": 30_000,
        "nbm_hidden": 48,
        "nbm_dropout": 0.10,
        "linear_ar_seconds": 0.5,
        "gru_layers": 1,
        "transformer_heads": 4,
        "transformer_layers": 2,
        "transformer_ffn": 128,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
        "cache_residuals": True,
        "deterministic": True,
        "amp": True,
        "classifier_hidden": 48,
        "classifier_dropout": 0.15,
        "classifier_epochs": 12,
        "classifier_patience": 4,
        "classifier_lr": 1e-3,
        "max_classifier_windows": 0,
        "channel_names": list(suite.EXPECTED_CHANNEL_NAMES),
        "subjects": list(suite.EXPECTED_LOSO_SUBJECTS),
        "folds_resolved": list(suite.EXPECTED_LOSO_SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "nbms_resolved": ["persistence"],
        "history_variants": [
            {
                "input": suite.INPUT_NAME,
                "history_samples": 256,
                "history_blocks": 8,
            }
        ],
    }


def _protocol_dataset_and_windows(
    tmp_path: Path,
) -> tuple[DaphnetDataset, WindowTable]:
    records = [
        Record(
            record_id=f"{subject}_R01",
            subject_id=subject,
            run_id="R01",
            x=np.zeros((640, 9), dtype=np.float32),
            y=np.zeros(640, dtype=np.int8),
            valid=np.ones(640, dtype=bool),
        )
        for subject in suite.EXPECTED_LOSO_SUBJECTS
    ]
    dataset = DaphnetDataset(
        root=tmp_path,
        records=records,
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    target_start = 128 + np.arange(32, dtype=np.int32) * 16
    labels = (np.arange(len(target_start)) % 3 == 0).astype(np.int8)
    windows = WindowTable(
        record_index=np.zeros(len(target_start), dtype=np.int32),
        start=target_start - 128,
        target_start=target_start,
        target_end=target_start + 32,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    return dataset, windows


def _dense_history_fixture() -> tuple[
    WindowTable,
    np.ndarray,
    dict[str, np.ndarray],
]:
    target_start = 128 + np.arange(40, dtype=np.int32) * 16
    labels = (np.arange(len(target_start)) % 5 < 2).astype(np.int8)
    windows = WindowTable(
        record_index=np.zeros(len(target_start), dtype=np.int32),
        start=target_start - 128,
        target_start=target_start,
        target_end=target_start + suite.HORIZON_SAMPLES,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    indices = np.arange(len(windows), dtype=np.int64)
    residual = np.stack(
        [
            np.full(
                (9, suite.HORIZON_SAMPLES),
                fill_value=row,
                dtype=np.float32,
            )
            for row in indices
        ]
    )
    return windows, indices, {
        "residual": residual,
        "y": labels.copy(),
        "window_index": indices,
    }


def test_protocol_fixes_horizon_label_history_and_tcn_m(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    dataset, windows = _protocol_dataset_and_windows(tmp_path)
    protocol = suite.build_protocol(
        args,
        source_manifest={"source_protocol_fingerprint": "a" * 64},
        source_config=_source_config(),
        dataset=dataset,
        windows=windows,
        data_sha256="b" * 64,
        device=torch.device("cpu"),
    )

    assert suite.SUITE_VERSION == (
        "daphnet_persistence_h4_tcnm_stride3_loso.v1"
    )
    assert protocol["horizon_samples"] == suite.HORIZON_SAMPLES == 32
    assert protocol["source_stride_samples"] == 16
    assert protocol["history_samples"] == suite.HISTORY_SAMPLES == 256
    assert protocol["history_blocks"] == suite.HISTORY_BLOCKS == 8
    assert protocol["history_block_spacing_samples"] == 32
    assert protocol["input"] == "residual_h4s"
    assert protocol["nbm"] == "persistence"
    assert protocol["seed"] == 42
    assert protocol["excluded_subjects"] == ["S04", "S10"]

    classifier = protocol["classifier"]
    assert classifier["name"] == "tcn_m"
    assert classifier["dilations"] == [1, 2, 4, 8, 8, 8]
    assert classifier["receptive_field_samples"] == 125
    assert classifier["parameter_count"] == 89_329
    assert (
        suite.rf.convolutional_receptive_field(
            tuple(classifier["dilations"])
        )
        == 125
    )

    variants = {
        variant["variant"]: (
            variant["predictor_stride_samples"],
            variant["classifier_stride_samples"],
        )
        for variant in protocol["variants"]
    }
    assert variants == {
        "s1": (16, 32),
        "s2": (16, 64),
        "s3": (32, 32),
    }
    fairness = protocol["fairness_contract"]
    assert fairness[
        "s1_s3_anchor_and_label_support_expected_identical"
    ]
    assert fairness["s1_s3_classifier_tensors_expected_identical"]
    assert fairness["s1_s3_deterministic_results_expected_identical"]
    assert fairness["s2_anchor_support_is_phase_fixed_subset_of_s1"]
    assert fairness["all_variants_share_frozen_source_nbm_and_cache"]
    assert fairness["s1_extra_phase_predictions_are_not_consumed"]


def test_grid_mask_keeps_record_local_phase_after_missing_windows() -> None:
    # 176 is deliberately absent. Selecting surviving rows by [::2] would
    # incorrectly choose 208 after the gap; the fixed grid must choose 192.
    one_record_starts = np.asarray(
        [128, 144, 160, 192, 208, 224],
        dtype=np.int32,
    )
    target_start = np.tile(one_record_starts, 2)
    record_index = np.repeat(np.arange(2, dtype=np.int32), 6)
    windows = WindowTable(
        record_index=record_index,
        start=target_start - 128,
        target_start=target_start,
        target_end=target_start + 32,
        label=np.zeros(len(target_start), dtype=np.int8),
        fog_fraction=np.zeros(len(target_start), dtype=np.float32),
        clean_normal=np.ones(len(target_start), dtype=bool),
    )
    indices = np.arange(len(windows), dtype=np.int64)

    mask = suite.grid_mask(
        windows,
        indices,
        stride_samples=32,
        origin_samples=128,
    )
    expected_per_record = np.asarray(
        [True, False, True, True, False, True]
    )
    np.testing.assert_array_equal(
        mask,
        np.tile(expected_per_record, 2),
    )
    np.testing.assert_array_equal(
        windows.target_start[indices[mask]],
        np.asarray([128, 160, 192, 224] * 2),
    )


def test_stride_support_is_nested_without_changing_half_second_labels() -> None:
    windows, indices, extracted = _dense_history_fixture()
    inputs: dict[str, dict[str, np.ndarray]] = {}
    plans = {}
    predictor_indices: dict[str, np.ndarray] = {}
    extracted_by_variant: dict[str, dict[str, np.ndarray]] = {}

    for name, definition in suite.STRIDE_VARIANTS.items():
        predictor_mask = suite.grid_mask(
            windows,
            indices,
            stride_samples=int(
                definition["predictor_stride_samples"]
            ),
            origin_samples=128,
        )
        predictor_indices[name] = indices[predictor_mask]
        extracted_by_variant[name] = {
            key: np.asarray(value)[predictor_mask]
            for key, value in extracted.items()
        }
        base_plan = make_common_history_plan(
            windows,
            predictor_indices[name],
            horizon_samples=suite.HORIZON_SAMPLES,
            stride_samples=int(
                definition["predictor_stride_samples"]
            ),
            max_history_samples=suite.HISTORY_SAMPLES,
        )
        classifier_mask = suite.grid_mask(
            windows,
            base_plan.anchor_window_indices,
            stride_samples=int(
                definition["classifier_stride_samples"]
            ),
            origin_samples=128,
        )
        plan = base_plan.take(np.flatnonzero(classifier_mask))
        plans[name] = plan
        chain_window_indices = predictor_indices[name][
            plan.max_chain_rows
        ]
        assert np.isin(
            chain_window_indices,
            predictor_indices[name],
        ).all()
        inputs[name] = make_history_input(
            extracted_by_variant[name],
            plan,
            suite.INPUT_NAME,
            history_samples=suite.HISTORY_SAMPLES,
            horizon_samples=suite.HORIZON_SAMPLES,
            stride_samples=int(
                definition["predictor_stride_samples"]
            ),
        )

        assert plan.max_chain_rows.shape[1] == suite.HISTORY_BLOCKS == 8
        assert inputs[name][suite.INPUT_NAME].shape[1:] == (9, 256)
        np.testing.assert_array_equal(
            inputs[name]["y"],
            windows.label[inputs[name]["window_index"]],
        )
        # Labels always come from the final 32-point target. In particular,
        # S2's 64-point output stride must not create a new one-second label.
        assert np.all(
            windows.target_end[inputs[name]["window_index"]]
            - windows.target_start[inputs[name]["window_index"]]
            == 32
        )

    for key in (suite.INPUT_NAME, "y", "window_index"):
        np.testing.assert_array_equal(inputs["s1"][key], inputs["s3"][key])
    assert len(predictor_indices["s3"]) < len(predictor_indices["s1"])

    s1_support = inputs["s1"]["window_index"]
    s2_support = inputs["s2"]["window_index"]
    assert set(s2_support).issubset(set(s1_support))
    assert set(s2_support) != set(s1_support)
    assert np.all(
        np.diff(windows.target_start[s1_support]) == 32
    )
    assert np.all(
        np.diff(windows.target_start[s2_support]) == 64
    )

    for name, plan in plans.items():
        chain_starts = windows.target_start[
            predictor_indices[name][plan.max_chain_rows]
        ]
        assert np.all(np.diff(chain_starts, axis=1) == 32), name
        for anchor, rows in enumerate(plan.max_chain_rows):
            for block, row in enumerate(rows):
                np.testing.assert_array_equal(
                    inputs[name][suite.INPUT_NAME][
                        anchor,
                        :,
                        block * 32 : (block + 1) * 32,
                    ],
                    extracted_by_variant[name]["residual"][row],
                )


def test_event_metrics_do_not_merge_or_score_across_missing_grid_gap(
    tmp_path: Path,
) -> None:
    # There is no evaluated target interval in [64, 96). A true event lives
    # only in that gap, and positive runs on either side must remain distinct
    # even though the configured merge gap is also 0.5 seconds.
    signal_length = 160
    y = np.zeros(signal_length, dtype=np.int8)
    y[64:96] = 1
    record = Record(
        record_id="R01",
        subject_id="S01",
        run_id="R01",
        x=np.zeros((signal_length, 9), dtype=np.float32),
        y=y,
        valid=np.ones(signal_length, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=tmp_path,
        records=[record],
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    target_start = np.asarray([0, 32, 96, 128], dtype=np.int32)
    windows = WindowTable(
        record_index=np.zeros(4, dtype=np.int32),
        start=target_start.copy(),
        target_start=target_start,
        target_end=target_start + 32,
        label=np.zeros(4, dtype=np.int8),
        fog_fraction=np.zeros(4, dtype=np.float32),
        clean_normal=np.ones(4, dtype=bool),
    )
    metrics = suite.stride_aware_event_metrics(
        dataset,
        windows,
        np.arange(4, dtype=np.int64),
        np.ones(4, dtype=np.int8),
        classifier_stride_samples=32,
        minimum_positive_windows=2,
        merge_gap_seconds=0.5,
    )

    assert metrics["evaluable_true_events"] == 0
    assert metrics["detected_true_events"] == 0
    assert metrics["predicted_events"] == 2
    assert metrics["false_alarm_events"] == 2
    assert metrics["evaluated_hours"] == pytest.approx(2.0 / 3600.0)


def test_s2_scheduled_gap_remains_monitored_and_can_miss_event(
    tmp_path: Path,
) -> None:
    signal_length = 96
    y = np.zeros(signal_length, dtype=np.int8)
    y[32:64] = 1
    record = Record(
        record_id="R01",
        subject_id="S01",
        run_id="R01",
        x=np.zeros((signal_length, 9), dtype=np.float32),
        y=y,
        valid=np.ones(signal_length, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=tmp_path,
        records=[record],
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    target_start = np.asarray([0, 64], dtype=np.int32)
    windows = WindowTable(
        record_index=np.zeros(2, dtype=np.int32),
        start=target_start.copy(),
        target_start=target_start,
        target_end=target_start + 32,
        label=np.zeros(2, dtype=np.int8),
        fog_fraction=np.zeros(2, dtype=np.float32),
        clean_normal=np.ones(2, dtype=bool),
    )
    metrics = suite.stride_aware_event_metrics(
        dataset,
        windows,
        np.arange(2, dtype=np.int64),
        np.zeros(2, dtype=np.int8),
        classifier_stride_samples=64,
    )

    assert metrics["evaluable_true_events"] == 1
    assert metrics["detected_true_events"] == 0
    assert metrics["event_sensitivity"] == 0.0
    assert metrics["evaluated_hours"] == pytest.approx(1.5 / 3600.0)


def test_preregistered_arguments_and_source_protocol_are_validated(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    suite.validate_args(args)
    suite.rf.validate_source_config(_source_config())

    bad_seed = copy.copy(args)
    bad_seed.seed = 41
    with pytest.raises(ValueError, match="seed 42"):
        suite.validate_args(bad_seed)

    contradictory_mode = copy.copy(args)
    contradictory_mode.finalize_only = True
    contradictory_mode.worker_fold = "S01"
    with pytest.raises(ValueError, match="cannot be combined"):
        suite.validate_args(contradictory_mode)

    capped_one = copy.copy(args)
    capped_one.max_classifier_windows = 1
    with pytest.raises(ValueError, match="zero or at least two"):
        suite.validate_args(capped_one)

    wrong_horizon = _source_config()
    wrong_horizon["horizon_samples"] = 16
    with pytest.raises(ValueError, match="horizon_samples"):
        suite.rf.validate_source_config(wrong_horizon)

    wrong_dense_stride = _source_config()
    wrong_dense_stride["stride_samples"] = 32
    with pytest.raises(ValueError, match="stride_samples"):
        suite.rf.validate_source_config(wrong_dense_stride)


def test_stride_metadata_completion_is_idempotent_and_upstream_bound(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "loso_S01" / "s1"
    task_root.mkdir(parents=True)
    suite.atomic_json_dump({"classifier_done": 1}, task_root / "DONE.json")
    config = {
        "protocol_fingerprint": "a" * 64,
        "sampling_rate_hz": 64,
    }
    fold_config = {
        "test_subject": "S01",
        "reference_initial_state_sha256": "b" * 64,
        "source": {"input_support_sha256": "c" * 64},
        "variant_source_residual_sha256": {"s1": "d" * 64},
        "predictor_window_counts": {
            "s1": {"train": 100, "validation": 20, "test": 30}
        },
        "classifier_actual_anchor_counts": {
            "s1": {"train": 50, "validation": 10, "test": 15}
        },
    }
    variant = {
        "variant": "s1",
        "experiment_id": suite.experiment_id("s1"),
        **suite.STRIDE_VARIANTS["s1"],
        "predictor_hz": 4.0,
        "classifier_hz": 2.0,
    }
    metadata = suite.stride_metadata_payload(
        config,
        fold_config,
        variant,
    )
    suite.save_stride_metadata_completion(
        task_root,
        config,
        fold_config,
        variant,
        metadata,
    )
    suite.save_stride_metadata_completion(
        task_root,
        config,
        fold_config,
        variant,
        metadata,
    )
    completed = suite.validate_done(
        task_root / "STRIDE_METADATA_DONE.json",
        stage="stride_metadata",
        protocol_fingerprint="a" * 64,
        task_id="S01/s1/stride_metadata",
        upstream_sha256=suite.sha256_file(task_root / "DONE.json"),
    )
    assert completed is not None
    assert set(completed["artifacts"]) == {"metadata"}
    assert suite.rf._load_json(task_root / "stride_metadata.json") == metadata


def test_multigpu_scheduler_dry_run_parsing_and_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common_argv = [
        str(scheduler.RUNNER),
        "--data-dir",
        str(tmp_path / "data"),
        "--source-suite-dir",
        str(tmp_path / "source"),
        "--output-dir",
        str(tmp_path / "output"),
        "--gpus",
        "0-6",
        "--work-folds",
        "all",
        "--classifier-epochs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", common_argv)
    args, forwarded = scheduler.parse_args()
    assert args.gpus == "0-6"
    assert args.work_folds == "all"
    assert forwarded == [
        "--classifier-epochs",
        "1",
        "--seed",
        "42",
    ]

    monkeypatch.setattr(sys, "argv", [*common_argv, "--seed=42"])
    _, explicit_seed = scheduler.parse_args()
    assert explicit_seed.count("--seed=42") == 1
    assert "--seed" not in explicit_seed

    with scheduler.configured_scheduler() as configured:
        assert configured.RUNNER == scheduler.RUNNER
        assert configured.AUDITOR == scheduler.AUDITOR
        assert tuple(configured.CANONICAL_CLASSIFIERS) == ("s1", "s2", "s3")
        assert configured.SCHEDULER_VERSION == scheduler.SCHEDULER_VERSION
        assert (
            configured.OutputDirectoryLock(tmp_path).path.name
            == scheduler.LOCK_FILENAME
        )
