from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.run_daphnet_processed_nbm_loso_s01_fewshot_head as fewshot
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import RoleRows
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
)


DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "1.Daphnet Freezing of Gait Dataset"
    / "processed_NBM"
)


def _copy_rows(rows: RoleRows) -> RoleRows:
    return RoleRows(
        *(getattr(rows, field).copy() for field in rows.__dataclass_fields__)
    )


def _real_splits() -> dict[str, RoleRows]:
    rows = fewshot.load_fold_rows(DATA_DIR, fewshot.SOURCE_OUTER_FOLD)
    return fewshot.build_personalization_splits(rows)


def test_protocol_identity_and_real_split_counts() -> None:
    assert fewshot.TEST_SUBJECT == "S01"
    assert fewshot.PERSONALIZATION_SEED == 0
    assert fewshot.ARMS == (
        fewshot.ZERO_SHOT,
        fewshot.THRESHOLD_ONLY,
        fewshot.HEAD_FINE_TUNE,
    )
    assert fewshot.EXPECTED_SPLIT_COUNTS == {
        "support": {"windows": 136, "nonfog": 127, "fog": 9},
        "calibration": {"windows": 41, "nonfog": 36, "fog": 5},
        "query": {"windows": 1380, "nonfog": 1330, "fog": 50},
    }

    splits = _real_splits()
    audit = fewshot.audit_personalization_splits(splits)
    assert audit["counts"] == fewshot.EXPECTED_SPLIT_COUNTS
    assert all(value == 0 for value in audit["cross_split_raw_point_overlap"].values())
    assert sum(len(rows) for rows in splits.values()) == 1557


def test_support_calibration_and_query_record_time_isolation() -> None:
    splits = _real_splits()
    support = splits["support"]
    calibration = splits["calibration"]
    query = splits["query"]

    # Support and calibration deliberately use non-overlapping time intervals
    # of seg002.  The blind query uses two entirely different records.
    assert set(support.record_id.tolist()) == {fewshot.SUPPORT_RECORD}
    assert set(calibration.record_id.tolist()) == {fewshot.CALIBRATION_RECORD}
    assert set(query.record_id.tolist()) == set(fewshot.QUERY_RECORDS)
    assert set(query.record_id.tolist()).isdisjoint(support.record_id.tolist())
    assert set(query.record_id.tolist()).isdisjoint(calibration.record_id.tolist())
    assert int(np.max(support.end)) <= int(np.min(calibration.start))

    window_sets = {
        name: set(rows.window_id.astype(str).tolist()) for name, rows in splits.items()
    }
    assert window_sets["support"].isdisjoint(window_sets["calibration"])
    assert window_sets["support"].isdisjoint(window_sets["query"])
    assert window_sets["calibration"].isdisjoint(window_sets["query"])


def test_split_audit_rejects_raw_sample_overlap() -> None:
    splits = _real_splits()
    corrupted = {name: _copy_rows(rows) for name, rows in splits.items()}
    # Preserve window/class counts and IDs, but make one calibration window
    # overlap the last support window in the same record.
    corrupted["calibration"].start[0] = corrupted["support"].start[-1]
    corrupted["calibration"].end[0] = corrupted["support"].end[-1]
    with pytest.raises(AssertionError, match="raw-point leakage"):
        fewshot.audit_personalization_splits(corrupted)


def test_head_only_freeze_and_batchnorm_statistics_do_not_update() -> None:
    torch.manual_seed(11)
    model = RepresentationTCNM(27)
    contract = fewshot.freeze_for_head_only(model)
    assert contract["trainable_parameter_names"] == [
        "classifier.weight",
        "classifier.bias",
    ]
    assert contract["trainable_parameter_count"] == 129
    assert contract["batchnorm_running_statistics_frozen"] is True
    assert contract["dropout_disabled"] is True

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    head_before = {
        name: parameter.detach().clone()
        for name, parameter in model.classifier.named_parameters()
    }
    bn_before = [
        (module.running_mean.clone(), module.running_var.clone())
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=fewshot.HEAD_LR,
        weight_decay=fewshot.HEAD_WEIGHT_DECAY,
    )
    model.eval()
    model.classifier.train()
    x = torch.randn(4, 27, 128)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
    optimizer.step()

    assert any(
        not torch.equal(head_before[name], parameter.detach())
        for name, parameter in model.classifier.named_parameters()
    )
    for name, parameter in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(frozen_before[name], parameter.detach()), name
    bn_after = [
        (module.running_mean, module.running_var)
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    assert len(bn_before) == len(bn_after) > 0
    for (mean_before, var_before), (mean_after, var_after) in zip(
        bn_before, bn_after
    ):
        assert torch.equal(mean_before, mean_after)
        assert torch.equal(var_before, var_after)


def test_exact_threshold_candidates_and_tie_break_rule() -> None:
    probabilities = np.asarray([0.1, 0.9], dtype=np.float64)
    candidates = fewshot.exact_threshold_candidates(probabilities)
    np.testing.assert_allclose(candidates, [0.0, 0.1, 0.5, 0.9, 1.0])
    threshold, metrics, candidate_count = fewshot.choose_exact_threshold(
        np.asarray([0, 1], dtype=np.int8), probabilities
    )
    # 0.5 and 0.9 give the same perfect predictions; the registered final
    # tie-break selects the higher threshold.
    assert threshold == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert candidate_count == len(candidates)

    tied_threshold, _, _ = fewshot.choose_exact_threshold(
        np.asarray([0, 1], dtype=np.int8),
        np.asarray([0.5, 0.5], dtype=np.float64),
    )
    # Thresholds 0 and 0.5 have identical BA/F1; choose 0.5, not the lower one.
    assert tied_threshold == pytest.approx(0.5)


def test_exact_threshold_candidates_include_all_negative_partition_at_one() -> None:
    candidates = fewshot.exact_threshold_candidates(
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    )
    assert candidates[-2] == pytest.approx(1.0)
    assert candidates[-1] > 1.0
    predictions = np.asarray([0.0, 0.5, 1.0]) >= candidates[-1]
    assert not predictions.any()


def _write_source_artifacts(root: Path, scientific_hash: str) -> dict[str, Path]:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    nbm_checkpoint = checkpoints / fewshot.checkpoint_name("MASK8_12")
    tcn_checkpoint = checkpoints / "tcn.pt"
    nbm_checkpoint.write_bytes(b"frozen-nbm")
    tcn_checkpoint.write_bytes(b"frozen-tcn")
    metrics = {
        "test_subject": "S01",
        "classifier": {"threshold": 0.79},
        "scaler": {"median": [0.0] * 9, "iqr": [1.0] * 9, "epsilon": 1e-6},
        "nbm": {"sigma_used_in_scheme_c": [1.0] * 9},
    }
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    barrier_path = root / "TRAINING_BARRIER.json"
    barrier_path.write_text(
        json.dumps(
            {
                "status": "all_training_validation_and_thresholds_frozen",
                "threshold": 0.79,
                "scientific_data_sha256": scientific_hash,
                "nbm_checkpoint_sha256": fewshot.sha256_file(nbm_checkpoint),
                "tcn_checkpoint_sha256": fewshot.sha256_file(tcn_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    done_path = root / "DONE.json"
    done_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "experiment_version": "source-test",
                "metrics_sha256": fewshot.sha256_file(metrics_path),
            }
        ),
        encoding="utf-8",
    )
    return {
        "nbm": nbm_checkpoint,
        "tcn": tcn_checkpoint,
        "metrics": metrics_path,
        "barrier": barrier_path,
        "done": done_path,
    }


def test_source_checkpoint_and_hash_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scientific_hash = "d" * 64
    paths = _write_source_artifacts(tmp_path / "source", scientific_hash)
    monkeypatch.setattr(
        fewshot,
        "processed_nbm_scientific_manifest",
        lambda _path: {"sha256": scientific_hash},
    )
    source = fewshot.validate_source_artifacts(
        tmp_path / "source", tmp_path / "processed_NBM"
    )
    assert source["threshold"] == pytest.approx(0.79)
    assert source["sha256"]["nbm_checkpoint"] == fewshot.sha256_file(paths["nbm"])
    assert source["sha256"]["tcn_checkpoint"] == fewshot.sha256_file(paths["tcn"])

    paths["tcn"].write_bytes(b"tampered-tcn")
    with pytest.raises(RuntimeError, match="TCN checkpoint hash mismatch"):
        fewshot.validate_source_artifacts(
            tmp_path / "source", tmp_path / "processed_NBM"
        )


def _valid_personalization_barrier(
    tmp_path: Path, source_meta: dict[str, object]
) -> Path:
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"head-checkpoint")
    path = tmp_path / fewshot.PERSONALIZATION_BARRIER
    path.write_text(
        json.dumps(
            {
                "status": "personalization_frozen_query_not_accessed",
                "experiment_version": fewshot.EXPERIMENT_VERSION,
                "head_checkpoint": str(checkpoint),
                "head_checkpoint_sha256": fewshot.sha256_file(checkpoint),
                "scientific_data_sha256": source_meta["scientific_data_sha256"],
                "source_artifact_sha256": source_meta["sha256"],
                "implementation_sha256": fewshot.implementation_hashes(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_query_barrier_rejects_missing_and_tampered_artifacts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="query access denied"):
        fewshot.validate_personalization_barrier(tmp_path / "missing.json")

    source_meta: dict[str, object] = {
        "scientific_data_sha256": "e" * 64,
        "sha256": {"source": "f" * 64},
    }
    path = _valid_personalization_barrier(tmp_path, source_meta)
    barrier = fewshot.validate_personalization_barrier(path, source_meta)
    Path(barrier["head_checkpoint"]).write_bytes(b"mutated-head")
    with pytest.raises(RuntimeError, match="head checkpoint hash mismatch"):
        fewshot.validate_personalization_barrier(path, source_meta)


def test_query_features_are_materialized_only_after_verified_barrier() -> None:
    source = inspect.getsource(fewshot.run)
    write_barrier = source.index("atomic_json_dump(barrier, barrier_path)")
    verify_barrier = source.index(
        "validate_personalization_barrier(barrier_path, source)"
    )
    materialize_query = source.index("query_x, query_feature = feature_values")
    assert write_barrier < verify_barrier < materialize_query


def test_all_three_arms_use_the_identical_query_labels_and_order() -> None:
    source = inspect.getsource(fewshot.run)
    assert "ZERO_SHOT: source_query_prob" in source
    assert "THRESHOLD_ONLY: source_query_prob" in source
    assert "HEAD_FINE_TUNE: fine_query_prob" in source
    assert (
        "arm: binary_metrics(query_y, probabilities[arm], float(thresholds[arm]))"
        in source
    )
    # Every arm writes the same ordered manifest; only probability/threshold
    # fields differ between arms.
    assert 'query_manifest = manifest_rows(splits["query"], "query")' in source
    assert "for arm in ARMS:" in source


def test_personalization_shuffle_and_threshold_selection_are_deterministic() -> None:
    assert fewshot.PERSONALIZATION_SEED == 0
    x = np.zeros((17, 128, 27), dtype=np.float32)
    x[:, 0, 0] = np.arange(len(x), dtype=np.float32)
    y = (np.arange(len(x)) % 2).astype(np.int8)

    def order() -> list[int]:
        loader = fewshot._head_loader(
            x,
            y,
            shuffle=True,
            seed=fewshot.PERSONALIZATION_SEED,
            num_workers=0,
        )
        return [
            int(value)
            for batch_x, _ in loader
            for value in batch_x[:, 0, 0].numpy().tolist()
        ]

    assert order() == order()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    first = fewshot.choose_exact_threshold(labels, probabilities)
    second = fewshot.choose_exact_threshold(labels, probabilities)
    assert first == second
