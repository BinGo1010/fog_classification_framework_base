from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.data import DaphnetDataset, Record, WindowTable
import audit_daphnet_persistence_imu_ablation as auditor
import run_daphnet_persistence_imu_ablation as suite
import start_daphnet_persistence_imu_ablation_multigpu as scheduler


VARIANT_NAMES = (
    "ankle",
    "thigh",
    "trunk",
    "ankle_thigh",
    "ankle_trunk",
    "thigh_trunk",
    "all_three",
)

CHANNEL_INDICES = {
    "ankle": (0, 1, 2),
    "thigh": (3, 4, 5),
    "trunk": (6, 7, 8),
    "ankle_thigh": (0, 1, 2, 3, 4, 5),
    "ankle_trunk": (0, 1, 2, 6, 7, 8),
    "thigh_trunk": (3, 4, 5, 6, 7, 8),
    "all_three": tuple(range(9)),
}

PARAMETER_COUNTS = {
    3: 89_041,
    6: 89_185,
    9: 89_329,
}


def _args(tmp_path: Path, *, smoke: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        source_suite_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        folds="all",
        worker_fold="",
        finalize_only=False,
        smoke=smoke,
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


def _source_config() -> dict[str, Any]:
    return {
        "suite_version": suite.SOURCE_SUITE_VERSION,
        "sampling_rate_hz": 64,
        "n_channels": 9,
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
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
            x=np.zeros((192, 9), dtype=np.float32),
            y=np.zeros(192, dtype=np.int8),
            valid=np.ones(192, dtype=bool),
        )
        for subject in suite.EXPECTED_LOSO_SUBJECTS
    ]
    dataset = DaphnetDataset(
        root=tmp_path,
        records=records,
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    labels = np.asarray([0, 1], dtype=np.int8)
    windows = WindowTable(
        record_index=np.asarray([0, 1], dtype=np.int32),
        start=np.asarray([0, 0], dtype=np.int32),
        target_start=np.asarray([128, 128], dtype=np.int32),
        target_end=np.asarray([160, 160], dtype=np.int32),
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    return dataset, windows


def _build_shared_model(
    channel_indices: tuple[int, ...],
    *,
    seed: int = 42,
) -> torch.nn.Module:
    """Call the public helper while accepting an explicit in_channels field."""

    signature = inspect.signature(suite.build_shared_initialised_model)
    kwargs: dict[str, Any] = {
        "channel_indices": channel_indices,
        "hidden_channels": 48,
        "dropout": 0.15,
        "seed": seed,
        "deterministic": True,
    }
    if "in_channels" in signature.parameters:
        kwargs["in_channels"] = len(channel_indices)
    model = suite.build_shared_initialised_model(**kwargs)
    assert isinstance(model, torch.nn.Module)
    return model


def test_imu_variant_matrix_matches_canonical_channel_order() -> None:
    assert tuple(suite.IMU_VARIANTS) == VARIANT_NAMES
    assert tuple(suite.EXPECTED_CHANNEL_NAMES) == (
        "ankle_acc_forward",
        "ankle_acc_vertical",
        "ankle_acc_lateral",
        "thigh_acc_forward",
        "thigh_acc_vertical",
        "thigh_acc_lateral",
        "trunk_acc_forward",
        "trunk_acc_vertical",
        "trunk_acc_lateral",
    )

    for name, expected_indices in CHANNEL_INDICES.items():
        variant = suite.IMU_VARIANTS[name]
        assert tuple(variant["channel_indices"]) == expected_indices
        assert int(variant["sensor_count"]) == len(expected_indices) // 3
        assert tuple(variant["channel_names"]) == tuple(
            suite.EXPECTED_CHANNEL_NAMES[index]
            for index in expected_indices
        )
        assert str(variant["display_name"]).strip()

    singletons = {
        tuple(suite.IMU_VARIANTS[name]["channel_indices"])
        for name in ("ankle", "thigh", "trunk")
    }
    assert singletons == {
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
    }
    assert sorted(
        index
        for indices in singletons
        for index in indices
    ) == list(range(9))


def test_subset_history_inputs_slices_channels_only_and_preserves_support(
) -> None:
    n_windows = 5
    full = np.arange(
        n_windows * 9 * suite.HISTORY_SAMPLES,
        dtype=np.float32,
    ).reshape(n_windows, 9, suite.HISTORY_SAMPLES)
    labels = np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    window_index = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
    full_inputs = {
        split: {
            suite.INPUT_NAME: full.copy() + split_index * 1_000_000,
            "y": labels.copy(),
            "window_index": window_index.copy(),
        }
        for split_index, split in enumerate(
            ("train", "validation", "test")
        )
    }
    untouched = copy.deepcopy(full_inputs)

    for name, expected_indices in CHANNEL_INDICES.items():
        subset = suite.subset_history_inputs(
            full_inputs,
            suite.IMU_VARIANTS[name],
        )
        assert set(subset) == {"train", "validation", "test"}
        for split in subset:
            payload = subset[split]
            assert payload[suite.INPUT_NAME].shape == (
                n_windows,
                len(expected_indices),
                suite.HISTORY_SAMPLES,
            )
            np.testing.assert_array_equal(
                payload[suite.INPUT_NAME],
                full_inputs[split][suite.INPUT_NAME][
                    :,
                    expected_indices,
                    :,
                ],
            )
            np.testing.assert_array_equal(payload["y"], labels)
            np.testing.assert_array_equal(
                payload["window_index"],
                window_index,
            )

    for split in untouched:
        for key in untouched[split]:
            np.testing.assert_array_equal(
                full_inputs[split][key],
                untouched[split][key],
            )

    malformed = copy.deepcopy(full_inputs)
    malformed["test"][suite.INPUT_NAME] = np.empty(
        (n_windows, 8, suite.HISTORY_SAMPLES),
        dtype=np.float32,
    )
    with pytest.raises((AssertionError, ValueError), match="9|channel"):
        suite.subset_history_inputs(
            malformed,
            suite.IMU_VARIANTS["ankle"],
        )


def test_shared_reference_initialisation_slices_projection_and_copies_common_layers(
) -> None:
    full_model = _build_shared_model(CHANNEL_INDICES["all_three"])
    full_state = full_model.state_dict()
    projection_key = "projection.0.weight"
    assert tuple(full_state[projection_key].shape) == (48, 9, 1)

    for name, indices in CHANNEL_INDICES.items():
        model = _build_shared_model(indices)
        state = model.state_dict()
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            PARAMETER_COUNTS[len(indices)]
        )
        np.testing.assert_array_equal(
            state[projection_key].cpu().numpy(),
            full_state[projection_key][:, indices, :].cpu().numpy(),
        )
        for key, tensor in state.items():
            if key == projection_key:
                continue
            np.testing.assert_array_equal(
                tensor.cpu().numpy(),
                full_state[key].cpu().numpy(),
                err_msg=f"shared state differs for {name}/{key}",
            )
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.zeros(
                    2,
                    len(indices),
                    suite.HISTORY_SAMPLES,
                )
            )
        assert logits.shape == (2,)

    different_seed = _build_shared_model(
        CHANNEL_INDICES["all_three"],
        seed=43,
    )
    assert not torch.equal(
        full_state[projection_key],
        different_seed.state_dict()[projection_key],
    )


def test_variant_protocol_fixes_tcn_m_and_reports_capacity_difference(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    variants = suite.variant_protocol(args, sampling_rate_hz=64)
    if isinstance(variants, tuple):
        variants = variants[0]
    by_name = {item["variant"]: item for item in variants}

    assert tuple(by_name) == VARIANT_NAMES
    assert suite.TCN_M_DILATIONS == (1, 2, 4, 8, 8, 8)
    assert suite.TCN_M_RF_SAMPLES == 125
    for name, indices in CHANNEL_INDICES.items():
        item = by_name[name]
        assert tuple(item["channel_indices"]) == indices
        assert tuple(item["channel_names"]) == tuple(
            suite.EXPECTED_CHANNEL_NAMES[index] for index in indices
        )
        assert item["sensor_count"] == len(indices) // 3
        assert item["n_channels"] == len(indices)
        assert item["dilations"] == [1, 2, 4, 8, 8, 8]
        assert item["receptive_field_samples"] == 125
        assert item["receptive_field_seconds"] == pytest.approx(125 / 64)
        assert item["parameter_count"] == PARAMETER_COUNTS[len(indices)]
        assert len(item["reference_initial_state_sha256"]) == 64
        assert len(item["shared_reference_state_sha256"]) == 64

    assert {
        item["shared_reference_state_sha256"] for item in variants
    } == {by_name["all_three"]["shared_reference_state_sha256"]}
    assert {
        item["parameter_count"] for item in variants
    } == {89_041, 89_185, 89_329}


def test_protocol_is_a_56_cell_frozen_persistence_sensor_ablation(
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

    assert protocol["nbm"] == "persistence"
    assert protocol["input"] == "residual_h4s"
    assert protocol["history_samples"] == 256
    assert protocol["history_blocks"] == 8
    assert protocol["horizon_samples"] == 32
    assert protocol["stride_samples"] == 16
    assert protocol["seed"] == 42
    assert protocol["excluded_subjects"] == ["S04", "S10"]
    assert protocol["expected_experiments"] == 7
    assert protocol["expected_classifier_cells"] == 56
    assert [item["variant"] for item in protocol["variants"]] == list(
        VARIANT_NAMES
    )

    classifier = protocol["classifier"]
    assert classifier["name"] == "tcn_m"
    assert classifier["dilations"] == [1, 2, 4, 8, 8, 8]
    assert classifier["receptive_field_samples"] == 125
    assert classifier["hidden_channels"] == 48
    assert classifier["kernel_size"] == 3
    assert classifier["convolutions_per_block"] == 2

    fairness = protocol["fairness_contract"]
    assert fairness["same_fold_scaler"]
    assert fairness["same_persistence_checkpoint_and_sigma"]
    assert fairness["same_anchor_history_and_labels"]
    assert fairness["same_tcn_m_common_initial_parameters"]
    assert fairness["variable_input_projection_only"]
    assert fairness["same_epoch_shuffle_rule"] == "classifier_seed + epoch"
    assert fairness["independent_validation_early_stopping"]
    assert fairness["independent_validation_threshold"]
    assert fairness["test_subject_never_selects_model_or_threshold"]

    comparisons = {
        (item["new"], item["reference"])
        for item in protocol["comparisons"]
    }
    assert comparisons == {
        (name, "all_three")
        for name in VARIANT_NAMES
        if name != "all_three"
    }


def test_scheduler_binds_runner_auditor_variants_lock_and_default_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert scheduler.RUNNER.name == (
        "run_daphnet_persistence_imu_ablation.py"
    )
    assert scheduler.AUDITOR.name == (
        "audit_daphnet_persistence_imu_ablation.py"
    )
    assert scheduler.CANONICAL_IMU_VARIANTS == VARIANT_NAMES
    assert scheduler.SCHEDULER_VERSION == (
        "daphnet_persistence_imu7_multigpu.v1"
    )
    assert scheduler.LOCK_FILENAME == (
        ".persistence_imu_ablation_scheduler.lock"
    )
    assert scheduler.OutputDirectoryLock(tmp_path).path == (
        tmp_path / scheduler.LOCK_FILENAME
    )

    original = {
        "RUNNER": scheduler.scheduler_base.RUNNER,
        "AUDITOR": scheduler.scheduler_base.AUDITOR,
        "CANONICAL_CLASSIFIERS": (
            scheduler.scheduler_base.CANONICAL_CLASSIFIERS
        ),
        "SCHEDULER_VERSION": scheduler.scheduler_base.SCHEDULER_VERSION,
        "OutputDirectoryLock": (
            scheduler.scheduler_base.OutputDirectoryLock
        ),
        "parse_args": scheduler.scheduler_base.parse_args,
    }
    with scheduler.configured_scheduler() as configured:
        assert configured.RUNNER == scheduler.RUNNER
        assert configured.AUDITOR == scheduler.AUDITOR
        assert configured.CANONICAL_CLASSIFIERS == VARIANT_NAMES
        assert configured.SCHEDULER_VERSION == scheduler.SCHEDULER_VERSION
        assert configured.OutputDirectoryLock is (
            scheduler.OutputDirectoryLock
        )
        assert configured.parse_args is scheduler.parse_args
    for key, value in original.items():
        assert getattr(scheduler.scheduler_base, key) is value

    monkeypatch.setattr(
        scheduler,
        "_base_parse_args",
        lambda: (SimpleNamespace(), ["--batch-size", "256"]),
    )
    _, forwarded = scheduler.parse_args()
    assert forwarded[-2:] == ["--seed", "42"]

    monkeypatch.setattr(
        scheduler,
        "_base_parse_args",
        lambda: (SimpleNamespace(), ["--seed=42"]),
    )
    _, forwarded = scheduler.parse_args()
    assert forwarded == ["--seed=42"]


def test_auditor_constants_bind_exactly_seven_variants_and_56_cells() -> None:
    assert auditor.AUDIT_VERSION == (
        "daphnet_persistence_imu_ablation_audit.v1"
    )

    expected_variants = None
    for name in (
        "EXPECTED_IMU_VARIANTS",
        "EXPECTED_VARIANTS",
        "CANONICAL_IMU_VARIANTS",
    ):
        if hasattr(auditor, name):
            expected_variants = tuple(getattr(auditor, name))
            break
    assert expected_variants == VARIANT_NAMES

    expected_cells = None
    for name in (
        "EXPECTED_CLASSIFIER_CELLS",
        "EXPECTED_CELLS",
    ):
        if hasattr(auditor, name):
            expected_cells = int(getattr(auditor, name))
            break
    assert expected_cells == 56
    assert tuple(auditor.suite.IMU_VARIANTS) == VARIANT_NAMES

