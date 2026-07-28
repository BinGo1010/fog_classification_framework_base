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

from cnbr_fog.data import (
    DaphnetDataset,
    Record,
    RobustChannelScaler,
    WindowTable,
)
from cnbr_fog.histories import make_common_history_plan
from cnbr_fog.nbm import PersistenceNBM
import run_daphnet_persistence_input_ablation as suite
import start_daphnet_persistence_input_ablation_multigpu as scheduler


REPRESENTATION_NAMES = (
    "raw_support_matched",
    "error_x_minus_mu",
    "standardized_error",
    "standardized_error_clip12",
)


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
        # This is attached by load_or_create_representation_cache in normal
        # execution and is needed only when testing extraction directly.
        source_residual_clip=12.0,
    )


def _source_config() -> dict:
    return {
        "context_samples": 128,
        "horizon_samples": 32,
        "stride_samples": 16,
        "seed": 42,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
        "excluded_subjects": ["S04", "S10"],
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


def _dense_block_fixture() -> tuple[
    WindowTable,
    dict[str, np.ndarray],
]:
    rows = np.arange(40, dtype=np.int64)
    target_start = 128 + rows.astype(np.int32) * suite.STRIDE_SAMPLES
    labels = (rows % 5 < 2).astype(np.int8)
    windows = WindowTable(
        record_index=np.zeros(len(rows), dtype=np.int32),
        start=target_start - suite.CONTEXT_SAMPLES,
        target_start=target_start,
        target_end=target_start + suite.HORIZON_SAMPLES,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    channel = np.arange(9, dtype=np.float32)[None, :, None] / 100.0
    sample = (
        np.arange(suite.HORIZON_SAMPLES, dtype=np.float32)[None, None, :]
        / 10_000.0
    )
    raw = rows.astype(np.float32)[:, None, None] + channel + sample
    mu = np.broadcast_to(
        rows.astype(np.float32)[:, None, None] / 10.0,
        raw.shape,
    ).copy()
    error = raw - mu
    sigma = np.linspace(
        0.5,
        2.0,
        9 * suite.HORIZON_SAMPLES,
        dtype=np.float32,
    ).reshape(1, 9, suite.HORIZON_SAMPLES)
    standardized = error / sigma
    return windows, {
        "raw": np.ascontiguousarray(raw),
        "mu": np.ascontiguousarray(mu),
        "error": np.ascontiguousarray(error),
        "standardized_error": np.ascontiguousarray(standardized),
        "standardized_error_clip12": np.ascontiguousarray(
            np.clip(standardized, -12.0, 12.0)
        ),
        "y": labels,
        "window_index": rows,
    }


def test_protocol_has_four_representations_32_cells_and_fixed_tcn_m(
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

    assert tuple(suite.REPRESENTATIONS) == REPRESENTATION_NAMES
    assert [item["variant"] for item in protocol["representations"]] == list(
        REPRESENTATION_NAMES
    )
    assert protocol["expected_experiments"] == 4
    assert protocol["expected_representation_cache_tasks"] == 8
    assert protocol["expected_classifier_cells"] == 32
    assert protocol["context_samples"] == 128
    assert protocol["persistence_effective_context_samples"] == 1
    assert protocol["horizon_samples"] == 32
    assert protocol["stride_samples"] == 16
    assert protocol["history_samples"] == 256
    assert protocol["history_blocks"] == 8
    assert protocol["classification_target_definition"] == (
        "label of the final 32-sample target block"
    )

    classifier = protocol["classifier"]
    assert classifier["name"] == "tcn_m"
    assert classifier["hidden_channels"] == 48
    assert classifier["kernel_size"] == 3
    assert classifier["convolutions_per_block"] == 2
    assert classifier["dilations"] == [1, 2, 4, 8, 8, 8]
    assert classifier["receptive_field_samples"] == 125
    assert classifier["parameter_count"] == 89_329
    assert classifier["global_pooling"] == "mean_and_max_over_full_4s_input"
    assert {
        item["parameter_count"] for item in protocol["representations"]
    } == {classifier["parameter_count"]}
    assert len(
        {
            item["reference_initial_state_sha256"]
            for item in protocol["representations"]
        }
    ) == 1

    assert [
        (item["comparison_id"], item["new"], item["reference"])
        for item in protocol["comparisons"]
    ] == [
        ("B_minus_A", "error_x_minus_mu", "raw_support_matched"),
        ("C_minus_B", "standardized_error", "error_x_minus_mu"),
        (
            "D_minus_C",
            "standardized_error_clip12",
            "standardized_error",
        ),
        (
            "D_minus_A",
            "standardized_error_clip12",
            "raw_support_matched",
        ),
    ]
    assert all(protocol["fairness_contract"].values())
    assert protocol["run_kind"] == "formal"
    assert protocol["reportable"] is True


def test_formal_arguments_are_locked_while_smoke_allows_reductions(
    tmp_path: Path,
) -> None:
    formal = _args(tmp_path)
    suite.validate_args(formal)

    reduced = copy.copy(formal)
    reduced.classifier_epochs = 1
    reduced.batch_size = 4
    reduced.max_classifier_windows = 8
    reduced.bootstrap_samples = 100
    with pytest.raises(ValueError, match="--smoke"):
        suite.validate_args(reduced)

    reduced.smoke = True
    suite.validate_args(reduced)
    dataset, windows = _protocol_dataset_and_windows(tmp_path)
    smoke_protocol = suite.build_protocol(
        reduced,
        source_manifest={"source_protocol_fingerprint": "c" * 64},
        source_config=_source_config(),
        dataset=dataset,
        windows=windows,
        data_sha256="d" * 64,
        device=torch.device("cpu"),
    )
    assert smoke_protocol["run_kind"] == "smoke"
    assert smoke_protocol["reportable"] is False
    assert smoke_protocol["classifier_epochs"] == 1
    assert smoke_protocol["max_classifier_windows"] == 8

    wrong_seed = copy.copy(reduced)
    wrong_seed.seed = 41
    with pytest.raises(ValueError, match="seed 42"):
        suite.validate_args(wrong_seed)

    contradictory = copy.copy(formal)
    contradictory.finalize_only = True
    contradictory.worker_fold = "S01"
    with pytest.raises(ValueError, match="cannot be combined"):
        suite.validate_args(contradictory)


def test_synthetic_persistence_block_formulas_and_canonical_clip(
    tmp_path: Path,
) -> None:
    time = np.arange(320, dtype=np.float32)[:, None]
    channel = np.arange(9, dtype=np.float32)[None, :]
    signal = (
        np.sin(time / 11.0) + 0.15 * channel - 0.4
    ).astype(np.float32)
    # Exercise the common raw robust-scaler clip independently of the later
    # residual-space clip.
    signal[128:160, 0] = 100.0
    labels = np.asarray([0, 1], dtype=np.int8)
    record = Record(
        record_id="synthetic",
        subject_id="S01",
        run_id="R01",
        x=signal,
        y=np.zeros(len(signal), dtype=np.int8),
        valid=np.ones(len(signal), dtype=bool),
    )
    dataset = DaphnetDataset(
        root=tmp_path,
        records=[record],
        sampling_rate_hz=64,
        channel_names=suite.EXPECTED_CHANNEL_NAMES,
    )
    starts = np.asarray([0, 160], dtype=np.int32)
    windows = WindowTable(
        record_index=np.zeros(2, dtype=np.int32),
        start=starts,
        target_start=starts + suite.CONTEXT_SAMPLES,
        target_end=starts
        + suite.CONTEXT_SAMPLES
        + suite.HORIZON_SAMPLES,
        label=labels,
        fog_fraction=labels.astype(np.float32),
        clean_normal=labels == 0,
    )
    scaler = RobustChannelScaler(
        center=np.linspace(-0.2, 0.2, 9, dtype=np.float32),
        scale=np.linspace(0.7, 1.3, 9, dtype=np.float32),
        clip=12.0,
    )
    model = PersistenceNBM(in_channels=9, horizon=suite.HORIZON_SAMPLES)
    with torch.no_grad():
        model.log_sigma.copy_(
            torch.linspace(
                float(np.log(0.5)),
                float(np.log(2.0)),
                9 * suite.HORIZON_SAMPLES,
            ).reshape(1, 9, suite.HORIZON_SAMPLES)
        )

    expected_raw: list[np.ndarray] = []
    expected_mu: list[np.ndarray] = []
    expected_sigma: np.ndarray | None = None
    for start, end in zip(windows.start, windows.target_end):
        sequence = scaler.transform(signal[int(start) : int(end)]).T
        context = torch.from_numpy(
            np.ascontiguousarray(sequence[:, : suite.CONTEXT_SAMPLES])
        ).unsqueeze(0)
        with torch.no_grad():
            mu, sigma = model(context)
        expected_raw.append(sequence[:, suite.CONTEXT_SAMPLES :])
        expected_mu.append(mu[0].numpy())
        if expected_sigma is None:
            expected_sigma = sigma.numpy()
    raw = np.ascontiguousarray(np.stack(expected_raw).astype(np.float32))
    mu = np.ascontiguousarray(np.stack(expected_mu).astype(np.float32))
    assert expected_sigma is not None
    error = raw - mu
    standardized = error / expected_sigma
    canonical = np.clip(standardized, -12.0, 12.0).astype(
        np.float32,
        copy=False,
    )

    features, diagnostics, sigma = suite.extract_representation_split(
        _args(tmp_path),
        model,
        dataset,
        windows,
        np.arange(2, dtype=np.int64),
        labels,
        canonical,
        scaler,
        12.0,
        torch.device("cpu"),
    )

    np.testing.assert_array_equal(features["raw"], raw)
    np.testing.assert_array_equal(features["mu"], mu)
    np.testing.assert_array_equal(features["error"], raw - mu)
    np.testing.assert_allclose(
        features["standardized_error"],
        features["error"] / sigma,
        rtol=suite.FORMULA_RTOL,
        atol=suite.FORMULA_ATOL,
    )
    np.testing.assert_array_equal(
        features["standardized_error_clip12"],
        canonical,
    )
    np.testing.assert_array_equal(features["y"], labels)
    np.testing.assert_array_equal(
        features["window_index"],
        np.arange(2, dtype=np.int64),
    )
    assert sigma.shape == (1, 9, 32)
    assert np.all(sigma > 0)
    assert np.max(np.abs(features["raw"])) == 12.0
    assert diagnostics["standardized_error"]["clip_fraction"] > 0.0
    assert diagnostics["canonical_clip_max_abs_diff"] <= suite.FORMULA_ATOL


def test_four_histories_share_support_labels_and_raw_uses_all_eight_blocks(
) -> None:
    windows, block_features = _dense_block_fixture()
    plan = make_common_history_plan(
        windows,
        block_features["window_index"],
        horizon_samples=suite.HORIZON_SAMPLES,
        stride_samples=suite.STRIDE_SAMPLES,
        max_history_samples=suite.HISTORY_SAMPLES,
    )
    assert plan.max_chain_rows.shape[1] == 8
    features = {
        split: block_features
        for split in ("train", "validation", "test")
    }
    plans = {
        split: plan
        for split in ("train", "validation", "test")
    }

    inputs = {
        name: suite.materialize_representation_inputs(
            features,
            plans,
            suite.representation_variant(name),
        )
        for name in REPRESENTATION_NAMES
    }
    baseline = inputs["raw_support_matched"]
    for name in REPRESENTATION_NAMES:
        source_key = suite.REPRESENTATIONS[name]["source_key"]
        for split in ("train", "validation", "test"):
            payload = inputs[name][split]
            assert payload[name].shape == (len(plan.anchor_rows), 9, 256)
            np.testing.assert_array_equal(
                payload["window_index"],
                baseline[split]["window_index"],
            )
            np.testing.assert_array_equal(
                payload["y"],
                baseline[split]["y"],
            )
            np.testing.assert_array_equal(
                payload["y"],
                block_features["y"][plan.anchor_rows],
            )
            for anchor, chain in enumerate(plan.max_chain_rows):
                for block, source_row in enumerate(chain):
                    np.testing.assert_array_equal(
                        payload[name][
                            anchor,
                            :,
                            block * 32 : (block + 1) * 32,
                        ],
                        block_features[source_key][source_row],
                    )

    # Raw-support-matched must concatenate the complete eight-block chain.
    # It is not the legacy anchor-only raw baseline, which would be [N,9,32].
    first_chain = plan.max_chain_rows[0]
    expected_raw = block_features["raw"][first_chain].transpose(
        1,
        0,
        2,
    ).reshape(9, 256)
    actual_raw = baseline["test"]["raw_support_matched"][0]
    np.testing.assert_array_equal(actual_raw, expected_raw)
    np.testing.assert_array_equal(
        actual_raw[:, :32],
        block_features["raw"][first_chain[0]],
    )
    np.testing.assert_array_equal(
        actual_raw[:, -32:],
        block_features["raw"][first_chain[-1]],
    )
    assert not np.array_equal(actual_raw[:, :32], actual_raw[:, -32:])


def test_representation_cache_key_contract_is_complete_and_exact() -> None:
    block_keys = (
        "raw",
        "mu",
        "error",
        "standardized_error",
        "standardized_error_clip12",
        "y",
        "window_index",
    )
    expected = {"sigma"} | {
        f"{split}_{key}"
        for split in ("train", "validation", "test")
        for key in block_keys
    }
    assert suite.representation_cache_keys() == expected
    assert len(expected) == 22
    assert {
        definition["source_key"]
        for definition in suite.REPRESENTATIONS.values()
    } == {
        "raw",
        "error",
        "standardized_error",
        "standardized_error_clip12",
    }
    assert "mu" not in {
        definition["source_key"]
        for definition in suite.REPRESENTATIONS.values()
    }


def test_paired_bootstrap_comparisons_and_wins_are_deterministic() -> None:
    assert [item["comparison_id"] for item in suite.COMPARISONS] == [
        "B_minus_A",
        "C_minus_B",
        "D_minus_C",
        "D_minus_A",
    ]
    differences = np.asarray(
        [0.10, 0.0, -0.20, 0.30, np.nan],
        dtype=np.float64,
    )
    seed = suite.stable_bootstrap_seed(42, "B_minus_A")
    first = suite.paired_bootstrap_mean_ci(differences, 5_000, seed)
    second = suite.paired_bootstrap_mean_ci(differences, 5_000, seed)

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.05)
    assert first["n_paired_subjects"] == 4
    assert first["bootstrap_samples"] == 5_000
    assert first["wins"] == 2
    assert first["ties"] == 1
    assert first["losses"] == 1
    assert first["ci_low"] <= first["mean_delta"] <= first["ci_high"]
    assert seed == suite.stable_bootstrap_seed(42, "B_minus_A")
    assert seed != suite.stable_bootstrap_seed(42, "C_minus_B")

    empty = suite.paired_bootstrap_mean_ci(
        np.asarray([np.nan]),
        20,
        seed,
    )
    assert empty["mean_delta"] is None
    assert empty["n_paired_subjects"] == 0
    assert empty["wins"] == empty["ties"] == empty["losses"] == 0


def test_representation_metadata_done_is_idempotent_and_upstream_bound(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "loso_S01" / "standardized_error_clip12"
    task_root.mkdir(parents=True)
    suite.atomic_json_dump(
        {"classifier_completion": 1},
        task_root / "DONE.json",
    )
    config = {
        "protocol_fingerprint": "a" * 64,
        "sampling_rate_hz": 64,
        "robust_clip": 12.0,
        "residual_clip": 12.0,
    }
    name = "standardized_error_clip12"
    fold_config = {
        "test_subject": "S01",
        "representation_cache_sha256": "b" * 64,
        "representation_input_fingerprints": {name: "c" * 64},
        "input_support_sha256": "d" * 64,
        "source": {"source_nbm_best_sha256": "e" * 64},
    }
    representation = suite.representation_variant(name)
    metadata = suite.representation_metadata_payload(
        config,
        fold_config,
        representation,
    )
    suite.save_representation_metadata_completion(
        task_root,
        config,
        fold_config,
        representation,
        metadata,
    )
    first_sha = suite.sha256_file(
        task_root / "REPRESENTATION_METADATA_DONE.json"
    )
    suite.save_representation_metadata_completion(
        task_root,
        config,
        fold_config,
        representation,
        metadata,
    )
    assert (
        suite.sha256_file(task_root / "REPRESENTATION_METADATA_DONE.json")
        == first_sha
    )

    classifier_done_sha = suite.sha256_file(task_root / "DONE.json")
    completed = suite.validate_done(
        task_root / "REPRESENTATION_METADATA_DONE.json",
        stage="representation_metadata",
        protocol_fingerprint="a" * 64,
        task_id=f"S01/{name}/representation_metadata",
        upstream_sha256=classifier_done_sha,
    )
    assert completed is not None
    assert set(completed["artifacts"]) == {"metadata"}
    assert suite.rf._load_json(
        task_root / "representation_metadata.json"
    ) == metadata
    assert metadata["residual_clip"] == 12.0
    assert metadata["classifier_output_stride_seconds"] == 0.25

    # The metadata completion belongs to one immutable classifier DONE file.
    suite.atomic_json_dump(
        {"classifier_completion": 2},
        task_root / "DONE.json",
    )
    with pytest.raises(ValueError, match="upstream"):
        suite.validate_done(
            task_root / "REPRESENTATION_METADATA_DONE.json",
            stage="representation_metadata",
            protocol_fingerprint="a" * 64,
            task_id=f"S01/{name}/representation_metadata",
            upstream_sha256=suite.sha256_file(task_root / "DONE.json"),
        )


def test_multigpu_wrapper_parse_and_bindings(
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
        "--smoke",
        "--classifier-epochs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", common_argv)
    args, forwarded = scheduler.parse_args()
    assert args.gpus == "0-6"
    assert args.work_folds == "all"
    assert forwarded == [
        "--smoke",
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
        assert tuple(configured.CANONICAL_CLASSIFIERS) == REPRESENTATION_NAMES
        assert configured.SCHEDULER_VERSION == scheduler.SCHEDULER_VERSION
        assert (
            configured.OutputDirectoryLock(tmp_path).path.name
            == scheduler.LOCK_FILENAME
        )
        command = configured.base_command(
            args,
            ["--smoke", "--seed", "42"],
        )
        assert command[:3] == [
            sys.executable,
            "-u",
            str(scheduler.RUNNER),
        ]
        assert "--resume" in command
        assert command[command.index("--folds") + 1] == "all"
        assert command[-3:] == ["--smoke", "--seed", "42"]
