from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cnbr_fog.data import DaphnetDataset, Record, WindowTable
from cnbr_fog.histories import make_common_history_plan, make_history_input
from run_daphnet_tcn_rf_ablation import (
    HISTORY_BLOCKS,
    HISTORY_SAMPLES,
    TCN_VARIANTS,
    array_loader,
    build_model,
    convolutional_receptive_field,
    parameter_count,
    set_seed,
    state_dict_sha256,
    train_classifier_resumable,
    validate_output_path,
)
from start_daphnet_tcn_rf_ablation_multigpu import (
    CANONICAL_FOLDS,
    OutputDirectoryLock,
    parse_folds,
    parse_gpu_ids,
    paths_overlap,
)


def test_canonical_dilations_have_requested_receptive_fields() -> None:
    expected = {"local": 29, "medium": 125, "long": 253}
    assert list(TCN_VARIANTS) == ["local", "medium", "long"]
    for name, receptive_field in expected.items():
        definition = TCN_VARIANTS[name]
        assert len(definition["dilations"]) == 6
        assert (
            convolutional_receptive_field(tuple(definition["dilations"]))
            == receptive_field
        )
        assert definition["receptive_field_samples"] == receptive_field


def test_variants_have_identical_parameters_and_initial_values() -> None:
    counts: list[int] = []
    hashes: list[str] = []
    state_schemas: list[list[tuple[str, tuple[int, ...]]]] = []
    for definition in TCN_VARIANTS.values():
        # The runner resets the same fold-specific seed before every variant.
        set_seed(10042, deterministic=True)
        model = build_model(
            in_channels=9,
            hidden_channels=48,
            dropout=0.15,
            dilations=tuple(definition["dilations"]),
        )
        counts.append(parameter_count(model))
        hashes.append(state_dict_sha256(model.state_dict()))
        state_schemas.append(
            [
                (name, tuple(tensor.shape))
                for name, tensor in model.state_dict().items()
            ]
        )
    assert counts == [89329, 89329, 89329]
    assert len(set(hashes)) == 1
    assert state_schemas[0] == state_schemas[1] == state_schemas[2]


def test_variants_only_change_block_dilation() -> None:
    for definition in TCN_VARIANTS.values():
        model = build_model(
            in_channels=9,
            hidden_channels=48,
            dropout=0.15,
            dilations=tuple(definition["dilations"]),
        )
        observed = []
        for block in model.blocks:
            first_conv = block.net[0]
            second_conv = block.net[4]
            assert first_conv.kernel_size == second_conv.kernel_size == (3,)
            assert first_conv.dilation == second_conv.dilation
            observed.append(first_conv.dilation[0])
        assert tuple(observed) == tuple(definition["dilations"])
        assert model(torch.randn(3, 9, HISTORY_SAMPLES)).shape == (3,)


def test_residual_h4s_is_eight_nonoverlapping_half_second_blocks() -> None:
    horizon = 32
    n_blocks = HISTORY_BLOCKS
    target_start = np.arange(n_blocks, dtype=np.int32) * horizon
    windows = WindowTable(
        record_index=np.zeros(n_blocks, dtype=np.int32),
        start=target_start.copy(),
        target_start=target_start,
        target_end=target_start + horizon,
        label=np.asarray([0] * (n_blocks - 1) + [1], dtype=np.int8),
        fog_fraction=np.asarray(
            [0.0] * (n_blocks - 1) + [1.0], dtype=np.float32
        ),
        clean_normal=np.asarray(
            [True] * (n_blocks - 1) + [False], dtype=bool
        ),
    )
    indices = np.arange(n_blocks, dtype=np.int64)
    residual = np.stack(
        [
            np.full((9, horizon), block, dtype=np.float32)
            for block in range(n_blocks)
        ]
    )
    extracted = {
        "residual": residual,
        "y": windows.label.copy(),
        "window_index": indices,
    }
    plan = make_common_history_plan(
        windows,
        indices,
        horizon_samples=horizon,
        stride_samples=16,
        max_history_samples=HISTORY_SAMPLES,
    )
    materialized = make_history_input(
        extracted,
        plan,
        "residual_h4s",
        history_samples=HISTORY_SAMPLES,
        horizon_samples=horizon,
        stride_samples=16,
    )
    assert materialized["residual_h4s"].shape == (1, 9, HISTORY_SAMPLES)
    assert materialized["window_index"].tolist() == [7]
    assert materialized["y"].tolist() == [1]
    for block in range(n_blocks):
        np.testing.assert_array_equal(
            materialized["residual_h4s"][0, :, block * horizon : (block + 1) * horizon],
            np.full((9, horizon), block, dtype=np.float32),
        )


def test_epoch_shuffle_is_identical_for_equal_seed() -> None:
    x = np.arange(32, dtype=np.float32)[:, None, None]
    y = np.arange(32, dtype=np.int64)

    def order(seed: int) -> list[int]:
        loader = array_loader(
            x,
            y,
            batch_size=7,
            shuffle=True,
            shuffle_seed=seed,
            num_workers=0,
            pin_memory=False,
        )
        return torch.cat([batch_y for _, batch_y in loader]).tolist()

    assert order(10043) == order(10043)
    assert order(10043) != order(10044)


def test_multigpu_parser_preserves_canonical_fold_order() -> None:
    assert parse_gpu_ids("0-2,6") == ["0", "1", "2", "6"]
    assert parse_folds("S09,S01,S05") == ["S01", "S05", "S09"]
    assert parse_folds("all") == list(CANONICAL_FOLDS)


def test_scheduler_lock_rejects_duplicate_output_owner(tmp_path: Path) -> None:
    assert paths_overlap(tmp_path, tmp_path / "child")
    first = OutputDirectoryLock(tmp_path)
    second = OutputDirectoryLock(tmp_path)
    first.acquire()
    try:
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


def test_runner_rejects_output_overlapping_source_or_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    safe_output = tmp_path / "output"
    source.mkdir()
    data.mkdir()
    validate_output_path(safe_output, source, data)
    for unsafe in (source, source / "child", tmp_path):
        try:
            validate_output_path(unsafe, source, data)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe output path was accepted: {unsafe}")


def test_tiny_training_writes_resumable_auditable_cell(tmp_path: Path) -> None:
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
    channel_names = (
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
    dataset = DaphnetDataset(
        root=tmp_path,
        records=[record],
        sampling_rate_hz=64,
        channel_names=channel_names,
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
    split_rows = {
        "train": np.arange(0, 16, dtype=np.int64),
        "validation": np.arange(16, 24, dtype=np.int64),
        "test": np.arange(24, 32, dtype=np.int64),
    }
    inputs = {
        split: {
            "residual_h4s": rng.normal(
                size=(len(rows), 9, HISTORY_SAMPLES)
            ).astype(np.float32),
            "y": labels[rows],
            "window_index": rows,
        }
        for split, rows in split_rows.items()
    }
    classifier_seed = 10042
    hidden = 8
    dropout = 0.0
    dilations = tuple(TCN_VARIANTS["local"]["dilations"])
    set_seed(classifier_seed, deterministic=True)
    reference_model = build_model(
        in_channels=9,
        hidden_channels=hidden,
        dropout=dropout,
        dilations=dilations,
    )
    reference_hash = state_dict_sha256(reference_model.state_dict())
    shared_parameters = parameter_count(reference_model)
    del reference_model
    variant = {
        "variant": "local",
        "display_name": "TCN-S",
        "experiment_id": "persistence_h4s__tcn_local",
        "dilations": list(dilations),
        "receptive_field_samples": 29,
        "receptive_field_seconds": 29 / 64,
    }
    config = {
        "protocol_fingerprint": "a" * 64,
        "shared_parameter_count": shared_parameters,
        "horizon_samples": 32,
        "stride_samples": 16,
    }
    fold_config = {
        "test_subject": "S01",
        "val_subject": "S02",
        "train_subjects": ["S03"],
        "classifier_seed": classifier_seed,
        "reference_initial_state_sha256": reference_hash,
        "source": {
            "source_residual_cache_sha256": "b" * 64,
            "input_support_sha256": "c" * 64,
        },
    }
    args = SimpleNamespace(
        classifier_hidden=hidden,
        classifier_dropout=dropout,
        classifier_lr=1e-3,
        weight_decay=1e-4,
        classifier_epochs=1,
        classifier_patience=1,
        batch_size=4,
        num_workers=0,
        amp=False,
        deterministic=True,
        resume=True,
        debug_interrupt_classifier_after_epoch=0,
    )
    task_root = tmp_path / "loso_S01" / "local"
    metrics = train_classifier_resumable(
        args,
        config,
        variant,
        task_root,
        fold_config,
        inputs,
        dataset,
        windows,
        torch.device("cpu"),
    )
    assert metrics["variant"] == "local"
    assert metrics["initial_state_sha256"] == reference_hash
    assert (task_root / "DONE.json").exists()
    with np.load(task_root / "predictions.npz", allow_pickle=False) as payload:
        assert payload["y_prob"].dtype == np.float64
        np.testing.assert_array_equal(
            payload["window_index"], split_rows["test"]
        )
    # A second invocation must take the validated DONE path without retraining.
    resumed = train_classifier_resumable(
        args,
        config,
        variant,
        task_root,
        fold_config,
        inputs,
        dataset,
        windows,
        torch.device("cpu"),
    )
    assert resumed == metrics
