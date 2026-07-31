from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cnbr_fog.data import DaphnetDataset, Record, RobustChannelScaler, WindowTable
from cnbr_fog.h200_phase3 import (
    InnerPredictorArtifact,
    OuterFoldContext,
    assemble_phase3_primitives,
    aggregate_phase3,
    evaluate_phase3a_science_gate,
    materialize_crossfit_classifier_base,
    phase3_outer_subjects,
    phase3_seed_policy,
    prepare_phase3_arm_inputs,
    representation_continuity_audit,
)


SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")


def _primitive(z_scale: float, log_sigma: float, clip_rate: float = 0.01) -> dict:
    rows = 4
    z = np.linspace(-1.0, 1.0, rows * 9 * 128, dtype=np.float32).reshape(
        rows, 9, 128
    )
    z *= np.float32(z_scale / z.std())
    return {
        "z": z,
        "log_sigma": np.full_like(z, log_sigma),
        "y": np.zeros(rows, dtype=np.int8),
        "diagnostics": {
            "z_clip_rate": float(clip_rate),
            "raw_clip_rate": 0.0,
        },
    }


def test_representation_continuity_audit_is_a_hard_numeric_contract() -> None:
    passed = representation_continuity_audit(
        _primitive(1.0, 0.0),
        _primitive(1.2, 0.2, 0.03),
        _primitive(0.8, -0.2, 0.02),
    )
    assert passed["status"] == "pass"
    assert all(
        check["z_std_ratio_pass"]
        and check["median_log_sigma_shift_pass"]
        and check["z_clip_rate_difference_pass"]
        for check in passed["checks"]
    )

    failed = representation_continuity_audit(
        _primitive(1.0, 0.0),
        _primitive(2.1, 0.0),
        _primitive(1.0, 0.8, 0.20),
    )
    assert failed["status"] == "fail"
    assert failed["checks"][0]["z_std_ratio_pass"] is False
    assert failed["checks"][1]["median_log_sigma_shift_pass"] is False
    assert failed["checks"][1]["z_clip_rate_difference_pass"] is False


def test_phase3_seed_and_outer_fold_policies_are_preregistered() -> None:
    protocol = SimpleNamespace(
        config={"subjects": list(SUBJECTS)},
        folds=SUBJECTS,
    )
    assert phase3_outer_subjects(protocol, "3a") == ("S01", "S05", "S08")
    assert phase3_outer_subjects(protocol, "3b") == SUBJECTS
    assert phase3_seed_policy(SimpleNamespace(), "3a") == {
        "nbm": (42,),
        "classifier": (42,),
    }
    assert phase3_seed_policy(SimpleNamespace(), "3b") == {
        "nbm": (42,),
        "classifier": (42, 43, 44),
    }
    custom = phase3_seed_policy(
        SimpleNamespace(phase3_nbm_seeds="7,8", phase3_classifier_seeds="1,2"),
        "3b",
    )
    assert custom == {"nbm": (7, 8), "classifier": (1, 2)}


def _phase3a_metric_rows(
    fusion_pr_auc: tuple[float, float, float] = (0.60, 0.55, 0.50),
    fusion_fa: tuple[float, float, float] = (11.0, 10.0, 9.0),
) -> list[dict]:
    rows = []
    for index, subject in enumerate(("S01", "S05", "S08")):
        rows.extend(
            [
                {
                    "test_subject": subject,
                    "arm": "raw6",
                    "pr_auc": (0.55, 0.52, 0.49)[index],
                    "false_alarm_events_per_hour": 10.0,
                },
                {
                    "test_subject": subject,
                    "arm": "raw4_zero",
                    "pr_auc": (0.56, 0.53, 0.48)[index],
                    "false_alarm_events_per_hour": 10.0,
                },
                {
                    "test_subject": subject,
                    "arm": "raw4_normality",
                    "pr_auc": fusion_pr_auc[index],
                    "false_alarm_events_per_hour": fusion_fa[index],
                },
            ]
        )
    return rows


def test_phase3a_science_gate_requires_direction_consistency_and_fa_safety() -> None:
    passed = evaluate_phase3a_science_gate(_phase3a_metric_rows())
    assert passed["status"] == "pass"
    assert passed["decision"] == "expand_to_phase3b"
    assert passed["comparisons"]["raw4_normality_minus_raw6"][
        "nonreversed_subjects"
    ] == 3
    assert passed["false_alarm_safety"]["pass"] is True

    direction_failed = evaluate_phase3a_science_gate(
        _phase3a_metric_rows(fusion_pr_auc=(0.54, 0.50, 0.52))
    )
    assert direction_failed["status"] == "fail"
    assert direction_failed["decision"] == "stop_before_phase3b"
    assert any("direction" in reason for reason in direction_failed["reasons"])
    assert any("fewer than 2/3" in reason for reason in direction_failed["reasons"])

    false_alarm_failed = evaluate_phase3a_science_gate(
        _phase3a_metric_rows(fusion_fa=(13.0, 13.0, 13.0))
    )
    assert false_alarm_failed["status"] == "fail"
    assert false_alarm_failed["false_alarm_safety"]["pass"] is False
    assert any("FA/h" in reason for reason in false_alarm_failed["reasons"])


def test_phase3_aggregate_combines_science_representation_and_external_status(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        output_dir=tmp_path,
        bootstrap_samples=200,
        bootstrap_seed=42,
    )
    protocol = SimpleNamespace(config={"protocol_fingerprint": "toy"})
    representation = {"status": "pass", "cells": []}

    phase3a_root = tmp_path / "phase3a"
    phase3a_root.mkdir(parents=True)
    (phase3a_root / "representation_gate.json").write_text(
        "{}", encoding="utf-8"
    )
    phase3a_rows = [
        {
            **row,
            "phase": "3a",
            "val_subject": "validation",
            "nbm_seed": 42,
            "classifier_seed": 42,
        }
        for row in _phase3a_metric_rows()
    ]
    phase3a = aggregate_phase3(
        args,
        protocol,
        phase="3a",
        rows=phase3a_rows,
        representation_gate=representation,
    )
    assert phase3a["decision"]["status"] == "pass"
    assert phase3a["external_negative_only_evaluation"]["status"] == (
        "not_applicable_before_phase3b"
    )

    phase3b_root = tmp_path / "phase3b"
    external_root = phase3b_root / "external_negative_only"
    external_root.mkdir(parents=True)
    (phase3b_root / "representation_gate.json").write_text(
        "{}", encoding="utf-8"
    )
    (external_root / "DONE.json").write_text("{}", encoding="utf-8")
    rows = []
    for subject in SUBJECTS:
        for arm in ("raw6", "raw4_zero", "raw4_normality"):
            rows.append(
                {
                    "phase": "3b",
                    "test_subject": subject,
                    "val_subject": "validation",
                    "arm": arm,
                    "nbm_seed": 42,
                    "classifier_seed": 42,
                    "pr_auc": 0.5,
                    "false_alarm_events_per_hour": 1.0,
                }
            )
    phase3b = aggregate_phase3(
        args,
        protocol,
        phase="3b",
        rows=rows,
        representation_gate=representation,
        external_evaluation={
            "status": "complete",
            "subjects": ["S04", "S10"],
        },
    )
    assert phase3b["decision"]["status"] == "pass"
    assert phase3b["external_negative_only_evaluation"]["status"] == "complete"


def _forecast(
    indices: np.ndarray,
    *,
    predictor_id: str,
    train: tuple[str, ...],
    heldout: tuple[str, ...],
    mean: float,
) -> dict:
    shape = (len(indices), 9, 128)
    target = np.zeros(shape, dtype=np.float32)
    return {
        "target": target,
        "mu": np.full(shape, mean, dtype=np.float32),
        "sigma": np.ones(shape, dtype=np.float32),
        "y": np.zeros(len(indices), dtype=np.int8),
        "window_index": np.asarray(indices, dtype=np.int64),
        "provenance": {
            "predictor_id": predictor_id,
            "predictor_train_subjects": list(train),
            "scaler_fit_subjects": list(train),
            "heldout_subjects": list(heldout),
        },
    }


def _crossfit_fixture() -> tuple[SimpleNamespace, OuterFoldContext, list[InnerPredictorArtifact]]:
    records = []
    for index, subject in enumerate(SUBJECTS):
        records.append(
            Record(
                record_id=f"{subject}_R01",
                subject_id=subject,
                run_id="R01",
                x=np.zeros((256, 9), dtype=np.float32),
                y=np.zeros(256, dtype=np.int8),
                valid=np.ones(256, dtype=bool),
            )
        )
    windows = WindowTable(
        record_index=np.arange(8, dtype=np.int32),
        start=np.zeros(8, dtype=np.int32),
        target_start=np.full(8, 128, dtype=np.int32),
        target_end=np.full(8, 256, dtype=np.int32),
        label=np.zeros(8, dtype=np.int8),
        fog_fraction=np.zeros(8, dtype=np.float32),
        clean_normal=np.ones(8, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=Path("."),
        records=records,
        sampling_rate_hz=64,
        channel_names=tuple(f"c{i}" for i in range(9)),
    )
    protocol = SimpleNamespace(
        dataset=dataset,
        master_windows=windows,
        classification_windows=windows,
        config={"protocol_fingerprint": "toy", "subjects": list(SUBJECTS)},
    )
    train_subjects = SUBJECTS[:6]
    outer = OuterFoldContext(
        subject="S09",
        val_subject="S08",
        train_subjects=train_subjects,
        scaler=RobustChannelScaler(
            center=np.zeros(9, dtype=np.float32),
            scale=np.ones(9, dtype=np.float32),
            clip=12.0,
        ),
        split_indices={
            "train": np.arange(6, dtype=np.int64),
            "validation": np.asarray([6], dtype=np.int64),
            "test": np.asarray([7], dtype=np.int64),
        },
        support={},
        source_fold_config={},
    )
    groups = (("S01", "S02"), ("S03", "S05"), ("S06", "S07"))
    artifacts = []
    for fold, heldout in enumerate(groups):
        train = tuple(subject for subject in train_subjects if subject not in heldout)
        heldout_indices = np.asarray(
            [train_subjects.index(subject) for subject in heldout], dtype=np.int64
        )
        common = {
            "predictor_id": f"inner_{fold}",
            "predictor_train_subjects": list(train),
            "scaler_fit_subjects": list(train),
            "heldout_subjects": list(heldout),
        }
        artifacts.append(
            InnerPredictorArtifact(
                predictor_id=f"inner_{fold}",
                inner_fold_index=fold,
                train_subjects=train,
                heldout_subjects=heldout,
                scaler=outer.scaler,
                checkpoint_sha256=f"sha{fold}",
                heldout_forecast=_forecast(
                    heldout_indices,
                    predictor_id=f"inner_{fold}",
                    train=train,
                    heldout=heldout,
                    mean=float(fold),
                ),
                validation_forecast=_forecast(
                    np.asarray([6]),
                    predictor_id=f"inner_{fold}",
                    train=train,
                    heldout=heldout,
                    mean=float(fold),
                ),
                test_forecast=_forecast(
                    np.asarray([7]),
                    predictor_id=f"inner_{fold}",
                    train=train,
                    heldout=heldout,
                    mean=float(fold),
                ),
                provenance=common,
            )
        )
    return protocol, outer, artifacts


def test_phase3_assembly_is_oof_and_moment_matches_in_physical_units() -> None:
    protocol, outer, artifacts = _crossfit_fixture()
    primitives, provenance = assemble_phase3_primitives(
        protocol,
        outer,
        artifacts,
        phase="3a",
        nbm_seed=42,
        args=SimpleNamespace(
            phase3_min_z_std_ratio=0.01,
            phase3_max_z_std_ratio=100.0,
            phase3_max_log_sigma_shift=10.0,
            phase3_max_z_clip_rate_difference=1.0,
        ),
    )
    assert provenance["oof_provenance_audit"]["status"] == "pass"
    assert provenance["variance_diagnostics"]["train_oof"][
        "between_model_variance_mean"
    ] == 0.0
    validation_variance = provenance["variance_diagnostics"][
        "validation_ensemble"
    ]
    np.testing.assert_allclose(validation_variance["aleatoric_variance_mean"], 1.0)
    np.testing.assert_allclose(validation_variance["between_model_variance_mean"], 2 / 3)
    np.testing.assert_allclose(validation_variance["total_variance_mean"], 5 / 3)
    np.testing.assert_allclose(primitives["validation"]["mu"], 1.0)
    np.testing.assert_allclose(primitives["validation"]["sigma"], np.sqrt(5 / 3))
    np.testing.assert_array_equal(
        primitives["train"]["window_index"], np.arange(6, dtype=np.int64)
    )


def test_classifier_materialization_uses_two_forecast_blocks_and_shared_endpoint() -> None:
    records = []
    record_index = []
    starts = []
    for index, subject in enumerate(("S01", "S02", "S03")):
        signal = np.arange(800 * 9, dtype=np.float32).reshape(800, 9) + index
        records.append(
            Record(
                record_id=f"{subject}_R01",
                subject_id=subject,
                run_id="R01",
                x=signal,
                y=np.zeros(800, dtype=np.int8),
                valid=np.ones(800, dtype=bool),
            )
        )
        record_index.extend([index] * 3)
        starts.extend([128, 256, 384])
    start = np.asarray(starts, dtype=np.int32)
    windows = WindowTable(
        record_index=np.asarray(record_index, dtype=np.int32),
        start=start,
        target_start=start + 128,
        target_end=start + 256,
        label=np.asarray([0, 0, 1] * 3, dtype=np.int8),
        fog_fraction=np.asarray([0, 0, 1] * 3, dtype=np.float32),
        clean_normal=np.asarray([1, 1, 0] * 3, dtype=bool),
    )
    dataset = DaphnetDataset(
        root=Path("."),
        records=records,
        sampling_rate_hz=64,
        channel_names=tuple(f"c{i}" for i in range(9)),
    )
    protocol = SimpleNamespace(dataset=dataset, classification_windows=windows)
    outer = OuterFoldContext(
        subject="S03",
        val_subject="S02",
        train_subjects=("S01",),
        scaler=RobustChannelScaler(
            center=np.zeros(9, dtype=np.float32),
            scale=np.ones(9, dtype=np.float32),
            clip=1e9,
        ),
        split_indices={},
        support={
            "train_h200_history_window_index": np.asarray([[0, 1]], dtype=np.int64),
            "train_anchor_window_index": np.asarray([2], dtype=np.int64),
            "train_y": np.asarray([1], dtype=np.int8),
        },
        source_fold_config={},
    )
    raw = np.stack(
        [np.full((9, 128), value, dtype=np.float32) for value in (2.0, 3.0, 4.0)]
    )
    primitives = {
        "raw": raw,
        "z": raw + 1,
        "log_sigma": raw + 20,
        "window_index": np.asarray([0, 1, 2], dtype=np.int64),
    }
    base = materialize_crossfit_classifier_base(protocol, outer, primitives, "train")
    np.testing.assert_array_equal(base["raw4"][0, :, :128], 2.0)
    np.testing.assert_array_equal(base["raw4"][0, :, 128:], 3.0)
    assert base["raw6"].shape == (1, 9, 384)
    fusion, y, endpoint = prepare_phase3_arm_inputs(
        base, "raw4_normality", np.asarray([0])
    )
    assert fusion.shape == (1, 27, 256)
    np.testing.assert_array_equal(y, [1])
    np.testing.assert_array_equal(endpoint, [2])
