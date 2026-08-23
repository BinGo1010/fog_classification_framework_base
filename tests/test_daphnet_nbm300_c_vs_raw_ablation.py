import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    barrier_identity_payload,
    build_test_data_manifest,
    build_scheme_c_features,
    expected_jobs,
    load_and_validate_barrier,
    paired_initialization,
    parse_csv_methods,
    raw_features,
    require_strict_barrier_for_tcn_v2,
    stable_json_hash,
    validate_completed_test_artifacts,
)
from scripts.run_daphnet_residual_calibration_abcd import (
    build_abcd_features,
    sha256_file,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


def test_raw_features_are_role4_scaled_then_window_axis_centered() -> None:
    raw = np.arange(2 * 128 * 9, dtype=np.float32).reshape(2, 128, 9)
    scaler = RobustScaler(
        median=np.linspace(-2, 2, 9, dtype=np.float32),
        iqr=np.linspace(1, 3, 9, dtype=np.float32),
    )
    actual = raw_features(scaler, raw)
    expected = scaler.transform(raw)
    expected -= expected.mean(axis=1, keepdims=True)
    assert actual.shape == (2, 128, 9)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    maximum_mean = float(np.max(np.abs(np.mean(actual, axis=1, dtype=np.float64))))
    maximum_signal = float(np.max(np.abs(actual)))
    tolerance = max(
        1e-5,
        64.0 * float(np.finfo(np.float32).eps) * max(1.0, maximum_signal),
    )
    assert maximum_mean <= tolerance


def test_raw_features_accept_float32_centering_roundoff_at_large_scale() -> None:
    rng = np.random.default_rng(7)
    raw = (rng.normal(size=(4, 128, 9)) * 1e4).astype(np.float32)
    scaler = RobustScaler(
        median=np.zeros(9, dtype=np.float32),
        iqr=np.full(9, 0.1, dtype=np.float32),
    )
    actual = raw_features(scaler, raw)
    assert actual.shape == (4, 128, 9)
    assert np.all(np.isfinite(actual))


def test_paired_initialization_shares_all_compatible_weights() -> None:
    raw_state, raw_meta = paired_initialization(52, "RAW")
    full_state, full_meta = paired_initialization(52, "FULL_C")
    assert raw_meta["pair_id"] == full_meta["pair_id"]
    for name, raw_tensor in raw_state.items():
        full_tensor = full_state[name]
        if raw_tensor.shape == full_tensor.shape:
            assert torch.equal(raw_tensor, full_tensor)
        else:
            assert raw_tensor.ndim == full_tensor.ndim == 3
            assert full_tensor.shape[1] == 27 and raw_tensor.shape[1] == 9
            assert torch.equal(raw_tensor, full_tensor[:, :9, :])
            assert torch.count_nonzero(full_tensor[:, 9:, :]) == 0


def test_32hz_raw_and_scheme_c_shapes() -> None:
    rng = np.random.default_rng(32)
    raw = rng.normal(size=(5, 64, 9)).astype(np.float32)
    scaler = RobustScaler(
        median=np.linspace(-1, 1, 9, dtype=np.float32),
        iqr=np.linspace(0.5, 2.5, 9, dtype=np.float32),
    )
    raw_input = raw_features(scaler, raw, window_samples=64)
    assert raw_input.shape == (5, 64, 9)

    error = rng.normal(size=(5, 9, 64)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 1], dtype=np.int8)
    sigma = np.linspace(0.1, 1.0, 9, dtype=np.float32)
    scheme_c, clip_stats = build_scheme_c_features(
        error, labels, sigma, window_samples=64
    )
    assert scheme_c.shape == (5, 64, 27)
    residual_bct = scheme_c[:, :, :9].transpose(0, 2, 1)
    delta_bct = scheme_c[:, :, 18:].transpose(0, 2, 1)
    np.testing.assert_allclose(
        residual_bct.mean(axis=2, dtype=np.float64), 0.0, atol=2e-6
    )
    np.testing.assert_array_equal(delta_bct[:, :, 0], 0.0)
    assert clip_stats["overall"]["points"] == 5 * 9 * 64


def test_dynamic_scheme_c_matches_original_128_sample_implementation() -> None:
    rng = np.random.default_rng(128)
    error = rng.normal(size=(4, 9, 128)).astype(np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
    bias = rng.normal(size=9).astype(np.float32)
    sigma = np.linspace(0.05, 1.5, 9, dtype=np.float32)
    expected, _ = build_abcd_features(error, labels, "C", bias, sigma)
    actual, _ = build_scheme_c_features(error, labels, sigma, window_samples=128)
    np.testing.assert_array_equal(actual, expected)


def test_residual_expansion_ablation_keeps_only_centered_r() -> None:
    rng = np.random.default_rng(20260823)
    error = rng.normal(size=(6, 9, 128)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    sigma = np.linspace(0.05, 1.25, 9, dtype=np.float32)

    expanded, expanded_stats = build_scheme_c_features(
        error, labels, sigma, window_samples=128, expand=True
    )
    residual_only, residual_stats = build_scheme_c_features(
        error, labels, sigma, window_samples=128, expand=False
    )

    assert expanded.shape == (6, 128, 27)
    assert residual_only.shape == (6, 128, 9)
    np.testing.assert_array_equal(residual_only, expanded[:, :, :9])
    assert residual_stats == expanded_stats
    np.testing.assert_allclose(
        residual_only.mean(axis=1, dtype=np.float64), 0.0, atol=2e-6
    )


def test_residual_r_uses_the_same_9ch_initialization_as_raw() -> None:
    raw_state, raw_meta = paired_initialization(5216, "RAW")
    residual_state, residual_meta = paired_initialization(5216, "RESIDUAL_R")
    assert raw_meta["pair_id"] == residual_meta["pair_id"]
    assert raw_state.keys() == residual_state.keys()
    for name in raw_state:
        assert torch.equal(raw_state[name], residual_state[name])


def test_dynamic_experiment_method_grid_supports_residual_ablation() -> None:
    methods = parse_csv_methods("FULL_C,RESIDUAL_R")
    jobs = expected_jobs((0, 52, 161, 5216, 52161), methods)
    assert methods == ("FULL_C", "RESIDUAL_R")
    assert len(jobs) == 30
    assert (2, "RESIDUAL_R", 52161) in jobs


def _test_rows() -> SimpleNamespace:
    return SimpleNamespace(
        subject_id=np.asarray(["S01", "S01"]),
        record_id=np.asarray(["S01_seg000", "S01_seg000"]),
        window_id=np.asarray(["w0", "w1"]),
        start=np.asarray([0, 128]),
        end=np.asarray([128, 256]),
        role=np.asarray([0, 1]),
        label=np.asarray([0, 1]),
    )


def test_permanent_test_manifest_binds_rows_and_record_bytes(tmp_path) -> None:
    (tmp_path / "records").mkdir()
    (tmp_path / "records" / "S01_seg000.npz").write_bytes(b"record-v1")
    (tmp_path / "nbm_protocol.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nbm_quality_report.json").write_text("{}", encoding="utf-8")
    rows_by_fold = {fold: _test_rows() for fold in (0, 1, 2)}
    first = build_test_data_manifest(tmp_path, rows_by_fold)
    (tmp_path / "records" / "S01_seg000.npz").write_bytes(b"record-v2")
    second = build_test_data_manifest(tmp_path, rows_by_fold)
    assert first["sha256"] != second["sha256"]
    assert first["folds"]["0"]["window_count"] == 2


def test_barrier_identity_rejects_mutated_sealed_job(tmp_path) -> None:
    barrier = {
        "barrier_schema": "strict_test_barrier.v2",
        "status": "sealed",
        "folds": [0, 1, 2],
        "methods": ["FULL_C", "RAW"],
        "nbm_seeds": [0],
        "tcn_seeds": [0],
        "job_count": 1,
        "strict_test_gate": "sealed",
        "source_audit": {"protocol_sha256": "p"},
        "test_data_manifest": {"sha256": "d"},
        "jobs": [{"job_id": "fold0_methodRAW_seed0", "threshold": 0.5}],
    }
    barrier["barrier_id"] = stable_json_hash(barrier_identity_payload(barrier))
    path = tmp_path / "TRAINING_BARRIER.json"
    path.write_text(json.dumps(barrier), encoding="utf-8")
    assert load_and_validate_barrier(path)["barrier_id"] == barrier["barrier_id"]
    barrier["jobs"][0]["threshold"] = 0.4
    path.write_text(json.dumps(barrier), encoding="utf-8")
    with pytest.raises(AssertionError, match="identity hash"):
        load_and_validate_barrier(path)


def test_tcn_v2_rejects_schema_less_legacy_barrier() -> None:
    legacy = {"status": "sealed", "jobs": []}
    with pytest.raises(RuntimeError, match="strict_test_barrier.v2"):
        require_strict_barrier_for_tcn_v2(legacy, "tcn_v2")
    require_strict_barrier_for_tcn_v2(legacy, "conv_tcn")


def test_completed_test_artifacts_must_match_current_barrier(tmp_path) -> None:
    sealed = {
        "job_id": "fold0_methodRAW_seed0",
        "barrier_schema": "strict_test_barrier.v2",
        "barrier_id": "barrier-a",
        "test_data_manifest_sha256": "data-a",
        "checkpoint_sha256": "tcn-a",
        "nbm_checkpoint_sha256": "nbm-a",
        "scaler_sha256": "scaler-a",
        "threshold": 0.5,
    }
    result = {
        "job_id": sealed["job_id"],
        "threshold": 0.5,
        "barrier_id": "barrier-a",
        "test_data_manifest_sha256": "data-a",
        "tcn_checkpoint_sha256": "tcn-a",
        "nbm_checkpoint_sha256": "nbm-a",
        "scaler_sha256": "scaler-a",
    }
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "test_predictions.csv"
    probabilities = tmp_path / "test_probabilities.npz"
    metrics.write_text(json.dumps(result), encoding="utf-8")
    predictions.write_text("y_true,y_pred\n0,0\n", encoding="utf-8")
    probabilities.write_bytes(b"npz-placeholder")
    done = {
        "status": "complete",
        "job_id": sealed["job_id"],
        **{
            key: result[key]
            for key in (
                "barrier_id",
                "test_data_manifest_sha256",
                "tcn_checkpoint_sha256",
                "nbm_checkpoint_sha256",
                "scaler_sha256",
            )
        },
        "metrics_sha256": sha256_file(metrics),
        "predictions_sha256": sha256_file(predictions),
        "probabilities_sha256": sha256_file(probabilities),
    }
    (tmp_path / "DONE_TEST.json").write_text(
        json.dumps(done), encoding="utf-8"
    )
    assert validate_completed_test_artifacts(tmp_path, sealed) == result
    changed_seal = {**sealed, "barrier_id": "barrier-b"}
    with pytest.raises(AssertionError, match="barrier_id"):
        validate_completed_test_artifacts(tmp_path, changed_seal)
