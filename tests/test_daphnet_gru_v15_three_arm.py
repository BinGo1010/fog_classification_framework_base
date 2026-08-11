from __future__ import annotations

import json
import inspect
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import launch_daphnet_gru_v15_three_arm_7gpu as launcher
from scripts import run_daphnet_gru_v15_three_arm as experiment
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_residual_calibration_abcd import sha256_file


def _role4_scaler_payload(*, fold: int = 0, seed: int = 52) -> dict[str, object]:
    return {
        "fold": fold,
        "seed": seed,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": 12_345,
        "scientific_data_sha256": "a" * 64,
        "scaler": {
            "median": np.linspace(-1.0, 1.0, 9).tolist(),
            "iqr": np.linspace(0.5, 2.5, 9).tolist(),
            "epsilon": 1e-6,
        },
    }


def test_three_arm_grid_is_45_jobs_with_exact_paired_seeds() -> None:
    assert experiment.METHODS == ("RAW", "GRU_V1_C", "GRU_V15_C")
    assert experiment.REQUIRED_SEEDS == (0, 52, 161, 5216, 52161)
    jobs = experiment.expected_jobs()
    assert len(jobs) == 45
    assert len(set(jobs)) == 45
    assert jobs == [
        (fold, method, seed)
        for fold in (0, 1, 2)
        for method in experiment.METHODS
        for seed in experiment.REQUIRED_SEEDS
    ]


def test_raw_loads_role4_scaler_without_reading_nbm_frozen(tmp_path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    scaler_path = fold_dir / "scaler_role4.json"
    scaler_path.write_text(
        json.dumps(_role4_scaler_payload()), encoding="utf-8"
    )
    # If the RAW loader accidentally parses the role-5/NBM artifact, this
    # deliberately invalid JSON makes the regression fail immediately.
    (fold_dir / "nbm_frozen.json").write_text("not-json", encoding="utf-8")

    scaler, artifact, contract = experiment.load_role4_scaler_metadata(
        tmp_path, fold=0, seed=52, scientific_data_sha256="a" * 64
    )

    np.testing.assert_allclose(
        scaler.median, _role4_scaler_payload()["scaler"]["median"]
    )
    assert artifact["scaler_fit_role"] == 4
    assert artifact["frozen_json"] is None
    assert artifact["frozen_json_sha256"] is None
    assert artifact["nbm_checkpoint"] is None
    assert artifact["nbm_checkpoint_sha256"] is None
    assert contract["uses_role5_calibration"] is False
    assert contract["uses_nbm"] is False


@pytest.mark.parametrize("forbidden", ["sigma", "b", "bias", "calibration"])
def test_raw_role4_scaler_rejects_role5_calibration_fields(
    tmp_path, forbidden: str
) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    payload = _role4_scaler_payload()
    payload[forbidden] = [1.0] * 9
    (fold_dir / "scaler_role4.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="role-5 fields"):
        experiment.load_role4_scaler_metadata(
            tmp_path, fold=0, seed=52, scientific_data_sha256="a" * 64
        )


def test_three_arms_use_paired_tcn_initialization() -> None:
    raw_state, raw_meta = experiment.paired_initialization(161, "RAW")
    v1_state, v1_meta = experiment.paired_initialization(161, "GRU_V1_C")
    v15_state, v15_meta = experiment.paired_initialization(161, "GRU_V15_C")

    assert raw_meta["pair_id"] == v1_meta["pair_id"] == v15_meta["pair_id"]
    assert v1_meta["selected_state_sha256"] == v15_meta["selected_state_sha256"]
    assert v1_state.keys() == v15_state.keys() == raw_state.keys()
    for name, v1_tensor in v1_state.items():
        v15_tensor = v15_state[name]
        raw_tensor = raw_state[name]
        assert torch.equal(v1_tensor, v15_tensor), name
        if raw_tensor.shape == v1_tensor.shape:
            assert torch.equal(raw_tensor, v1_tensor), name
        else:
            # Only the first input convolution differs: RAW occupies the first
            # nine channels and the residual-only channels start at exactly 0.
            assert raw_tensor.ndim == v1_tensor.ndim == 3, name
            assert raw_tensor.shape[1] == 9 and v1_tensor.shape[1] == 27, name
            assert torch.equal(raw_tensor, v1_tensor[:, :9, :]), name
            assert torch.count_nonzero(v1_tensor[:, 9:, :]).item() == 0, name


def test_launcher_dry_run_has_30_nbm_45_train_45_test_and_seven_gpus(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--dry-run",
            "--gpu-ids",
            "0,1,2,3,4,5,6",
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    launcher.main()
    plan = json.loads(capsys.readouterr().out)

    assert plan["gpu_ids"] == ["0", "1", "2", "3", "4", "5", "6"]
    assert plan["nbm_jobs"] == 30
    assert plan["classifier_train_jobs"] == 45
    assert plan["post_barrier_test_jobs"] == 45
    assert plan["methods"] == list(experiment.METHODS)
    assert plan["nbm_seeds"] == list(experiment.REQUIRED_SEEDS)
    assert plan["tcn_seeds"] == list(experiment.REQUIRED_SEEDS)


def test_evaluate_fails_closed_when_global_barrier_is_missing(tmp_path) -> None:
    args = Namespace(
        output_root=tmp_path,
        fold=0,
        method="RAW",
        tcn_seed=0,
        gru_v1_source_root=tmp_path / "v1",
        gru_v15_source_root=tmp_path / "v15",
    )
    with pytest.raises(FileNotFoundError, match="roles 0/1 forbidden"):
        experiment.sealed_job(args)


def test_aggregate_declares_all_three_paired_delta_comparisons() -> None:
    # This is a light static contract: a full aggregate requires all 45 test
    # artifacts, but these exact keys are the public summary/report filenames.
    source = __import__("inspect").getsource(experiment.run_aggregate)
    assert '"GRU_V1_C_minus_RAW"' in source
    assert '"GRU_V15_C_minus_RAW"' in source
    assert '"GRU_V15_C_minus_GRU_V1_C"' in source


def _minimal_processed_nbm(root: Path) -> None:
    """Create the smallest byte-addressable processed_NBM identity tree."""
    root.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("manifest.csv", "record_id,path\nR01,records/R01.bin\n"),
        ("schema.json", '{"channels":9}'),
        ("nbm_protocol.json", '{"roles":[0,1,2,3,4,5,6,7]}'),
        ("nbm_quality_report.json", '{"ok":true}'),
    ):
        (root / name).write_text(value, encoding="utf-8")
    (root / "records").mkdir()
    (root / "split_indices").mkdir()
    (root / "records" / "R01.bin").write_bytes(b"record-bytes-A")
    (root / "split_indices" / "fold_0.bin").write_bytes(b"split-bytes-A")


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    [
        (Path("records/R01.bin"), b"record-bytes-B"),
        (Path("split_indices/fold_0.bin"), b"split-bytes-B"),
    ],
)
def test_scientific_manifest_changes_when_record_or_split_bytes_change(
    tmp_path: Path, relative_path: Path, replacement: bytes
) -> None:
    _minimal_processed_nbm(tmp_path)
    before = processed_nbm_scientific_manifest(tmp_path)

    # Keep the filename and byte count constant: the scientific fingerprint
    # must bind content, not merely the directory listing or file metadata.
    target = tmp_path / relative_path
    assert len(replacement) == target.stat().st_size
    target.write_bytes(replacement)
    after = processed_nbm_scientific_manifest(tmp_path)

    assert after["sha256"] != before["sha256"]
    before_files = {item["relative_path"]: item for item in before["files"]}
    after_files = {item["relative_path"]: item for item in after["files"]}
    key = relative_path.as_posix()
    assert after_files[key]["size_bytes"] == before_files[key]["size_bytes"]
    assert after_files[key]["sha256"] != before_files[key]["sha256"]


def test_raw_role4_scaler_rejects_wrong_scientific_hash(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    payload = _role4_scaler_payload()
    payload["scientific_data_sha256"] = "b" * 64
    (fold_dir / "scaler_role4.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="scientific dataset changed"):
        experiment.load_role4_scaler_metadata(
            tmp_path, fold=0, seed=52, scientific_data_sha256="a" * 64
        )


def _write_minimal_frozen_raw_job(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Namespace, str, dict[str, object]]:
    """Materialize a valid RAW training seal without loading any data/model."""
    scientific_hash = "d" * 64
    code_hash = {"scripts/frozen.py": "e" * 64}
    monkeypatch.setattr(experiment, "critical_code_sha256", lambda: code_hash)

    checkpoint = directory / "checkpoints" / "tcn.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"paired-tcn-checkpoint")
    source_root = directory / "upstream"
    scaler_path = source_root / "fold_0" / "scaler_role4.json"
    scaler_path.parent.mkdir(parents=True)
    scaler_path.write_text("{}", encoding="utf-8")
    artifact: dict[str, object] = {
        "scaler_json": str(scaler_path.resolve()),
        "scaler_sha256": "1" * 64,
        "scaler_json_sha256": "2" * 64,
        "frozen_json_sha256": None,
        "done_nbm_json_sha256": None,
        "nbm_checkpoint_sha256": None,
        "scientific_data_sha256": scientific_hash,
    }
    monkeypatch.setattr(
        experiment,
        "load_role4_scaler_metadata",
        lambda *args, **kwargs: (object(), dict(artifact), {"uses_nbm": False}),
    )

    contract = experiment.feature_contract("RAW")
    frozen = {
        "job_id": experiment.job_id(0, "RAW", 52),
        "fold": 0,
        "method": "RAW",
        "source_kind": experiment.SOURCE_GRU_V1,
        "uses_nbm": False,
        "nbm_seed": 52,
        "tcn_seed": 52,
        "test_roles_accessed": False,
        "scientific_data_sha256": scientific_hash,
        "experiment_code_sha256": code_hash,
        "feature_contract": contract,
        "checkpoint_sha256": sha256_file(checkpoint),
        "role4_scaler_artifact": artifact,
    }
    frozen_path = directory / "frozen_validation.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    done = {
        "status": "frozen",
        "job_id": frozen["job_id"],
        "fold": 0,
        "method": "RAW",
        "source_kind": experiment.SOURCE_GRU_V1,
        "uses_nbm": False,
        "nbm_seed": 52,
        "tcn_seed": 52,
        "checkpoint_sha256": sha256_file(checkpoint),
        "frozen_validation_sha256": sha256_file(frozen_path),
        "scientific_data_sha256": scientific_hash,
        "feature_contract_sha256": contract["sha256"],
        "test_roles_accessed": False,
    }
    (directory / "DONE_TRAIN.json").write_text(json.dumps(done), encoding="utf-8")
    return Namespace(), scientific_hash, frozen


@pytest.mark.parametrize(
    "tamper",
    ("done_identity", "feature_contract", "frozen_hash"),
)
def test_validate_frozen_training_job_rejects_tampered_identity_or_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    args, scientific_hash, _ = _write_minimal_frozen_raw_job(tmp_path, monkeypatch)
    # Establish that the fixture itself is a valid frozen job.
    experiment.validate_frozen_training_job(
        args, tmp_path, 0, "RAW", 52, scientific_hash
    )

    done_path = tmp_path / "DONE_TRAIN.json"
    frozen_path = tmp_path / "frozen_validation.json"
    if tamper == "done_identity":
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        payload["job_id"] = "fold0_methodRAW_seed161"
        done_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "DONE_TRAIN identity mismatch"
    elif tamper == "feature_contract":
        payload = json.loads(frozen_path.read_text(encoding="utf-8"))
        payload["feature_contract"]["formula"] = "tampered"
        frozen_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "feature contract changed"
    else:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        payload["frozen_validation_sha256"] = "0" * 64
        done_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "DONE_TRAIN identity mismatch"

    with pytest.raises(AssertionError, match=expected):
        experiment.validate_frozen_training_job(
            args, tmp_path, 0, "RAW", 52, scientific_hash
        )


def _write_completed_test_artifacts(
    directory: Path,
) -> tuple[dict[str, object], Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    sealed: dict[str, object] = {
        "job_id": "fold0_methodGRU_V15_C_seed52",
        "barrier_schema": "strict_test_barrier.v2",
        "barrier_id": "barrier-identity",
        "test_data_manifest_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "nbm_checkpoint_sha256": "3" * 64,
        "scaler_sha256": "4" * 64,
        "nbm_frozen_sha256": "5" * 64,
        "done_nbm_sha256": "6" * 64,
        "frozen_validation_sha256": "7" * 64,
        "scientific_data_sha256": "8" * 64,
        "feature_contract_sha256": "9" * 64,
        "threshold": 0.42,
    }
    shared = {
        "job_id": sealed["job_id"],
        "barrier_id": sealed["barrier_id"],
        "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
        "nbm_frozen_sha256": sealed["nbm_frozen_sha256"],
        "done_nbm_sha256": sealed["done_nbm_sha256"],
        "frozen_validation_sha256": sealed["frozen_validation_sha256"],
        "scientific_data_sha256": sealed["scientific_data_sha256"],
        "feature_contract_sha256": sealed["feature_contract_sha256"],
    }
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(
        json.dumps({**shared, "threshold": sealed["threshold"]}), encoding="utf-8"
    )
    predictions_path = directory / "test_predictions.csv"
    probabilities_path = directory / "test_probabilities.npz"
    predictions_path.write_bytes(b"header\n")
    probabilities_path.write_bytes(b"npz-placeholder")
    done_path = directory / "DONE_TEST.json"
    done_path.write_text(
        json.dumps(
            {
                **shared,
                "status": "complete",
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
                "probabilities_sha256": sha256_file(probabilities_path),
            }
        ),
        encoding="utf-8",
    )
    return sealed, metrics_path, done_path


@pytest.mark.parametrize(
    ("field", "artifact"),
    [
        ("nbm_frozen_sha256", "metrics"),
        ("done_nbm_sha256", "done"),
        ("frozen_validation_sha256", "metrics"),
        ("scientific_data_sha256", "done"),
        ("feature_contract_sha256", "metrics"),
    ],
)
def test_completed_test_artifacts_reject_optional_seal_hash_mismatch(
    tmp_path: Path, field: str, artifact: str
) -> None:
    sealed, metrics_path, done_path = _write_completed_test_artifacts(tmp_path)
    experiment.validate_completed_test_artifacts(tmp_path, sealed)

    target = metrics_path if artifact == "metrics" else done_path
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[field] = "f" * 64
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match=f"current seal: {field}"):
        experiment.validate_completed_test_artifacts(tmp_path, sealed)


def test_barrier_entries_bind_upstream_and_scientific_contract_hashes() -> None:
    source = inspect.getsource(experiment.run_seal)
    for field in (
        "nbm_frozen_sha256",
        "done_nbm_sha256",
        "scientific_data_sha256",
        "feature_contract_sha256",
    ):
        assert f'"{field}"' in source
    # These values must be copied from the upstream artifact or the immutable
    # frozen classifier record, not synthesized as loose descriptive metadata.
    assert 'artifact["frozen_json_sha256"]' in source
    assert 'artifact["done_nbm_json_sha256"]' in source
    assert 'frozen["scientific_data_sha256"]' in source
    assert 'frozen["feature_contract"]["sha256"]' in source
