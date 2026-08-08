from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.run_daphnet_residual_calibration_abcd import (
    CHANNEL_NAMES,
    barrier_job,
    build_abcd_features,
    combine_clip_statistics,
    expected_jobs,
    parse_group_list,
)


def test_abcd_formulas_shapes_and_delta() -> None:
    error = np.linspace(-2.0, 2.0, 4 * 9 * 128, dtype=np.float32).reshape(4, 9, 128)
    error[0, 0, 1] = 50.0
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    bias = np.linspace(-0.2, 0.2, 9, dtype=np.float32)
    sigma = np.linspace(0.05, 0.2, 9, dtype=np.float32)

    outputs = {
        group: build_abcd_features(error, labels, group, bias, sigma)
        for group in "ABCD"
    }
    for features, _ in outputs.values():
        assert features.shape == (4, 128, 27)
        np.testing.assert_array_equal(features[:, :, 9:18], np.abs(features[:, :, :9]))
        np.testing.assert_array_equal(features[:, 0, 18:], 0.0)
        np.testing.assert_allclose(
            features[:, 1:, 18:],
            np.diff(features[:, :, :9], axis=1),
            atol=0.0,
            rtol=0.0,
        )

    a = outputs["A"][0][:, :, :9].transpose(0, 2, 1)
    expected_a = np.clip(
        (error - bias[None, :, None]) / (sigma[None, :, None] + 1e-6),
        -12.0,
        12.0,
    )
    np.testing.assert_allclose(a, expected_a)

    expected_b = expected_a - expected_a.mean(axis=2, keepdims=True)
    np.testing.assert_allclose(
        outputs["B"][0][:, :, :9].transpose(0, 2, 1), expected_b, atol=2e-6
    )
    expected_c = np.clip(
        error / (sigma[None, :, None] + 1e-6), -12.0, 12.0
    )
    expected_c -= expected_c.mean(axis=2, keepdims=True)
    np.testing.assert_allclose(
        outputs["C"][0][:, :, :9].transpose(0, 2, 1), expected_c, atol=2e-6
    )
    expected_d = error - error.mean(axis=2, keepdims=True)
    np.testing.assert_allclose(
        outputs["D"][0][:, :, :9].transpose(0, 2, 1), expected_d, atol=2e-6
    )

    for group in "BCD":
        residual = outputs[group][0][:, :, :9]
        np.testing.assert_allclose(residual.mean(axis=1), 0.0, atol=2e-6)
    assert outputs["A"][1] == outputs["B"][1]
    assert outputs["C"][1]["applicable"] is True
    assert outputs["D"][1]["applicable"] is False


def test_clip_statistic_combination_preserves_channel_names() -> None:
    error = np.zeros((2, 9, 128), dtype=np.float32)
    error[:, 3, :4] = 20.0
    labels = np.asarray([0, 1], dtype=np.int8)
    bias = np.zeros(9, dtype=np.float32)
    sigma = np.ones(9, dtype=np.float32)
    _, stats = build_abcd_features(error, labels, "A", bias, sigma)
    combined = combine_clip_statistics([stats, stats])
    assert combined["overall"]["points"] == 2 * stats["overall"]["points"]
    assert combined["overall"]["clipped"] == 2 * stats["overall"]["clipped"]
    assert [item["channel_name"] for item in combined["per_channel"]] == list(CHANNEL_NAMES)


def test_test_stage_rejects_missing_global_barrier() -> None:
    args = Namespace(
        output_root=Path("outputs/__intentionally_missing_abcd_barrier_test__"),
        fold=0,
        group="A",
        tcn_seed=20260807,
    )
    with pytest.raises(FileNotFoundError, match="TRAINING_BARRIER.json missing"):
        barrier_job(args)


def test_bc_group_subset_builds_exactly_eighteen_jobs() -> None:
    groups = parse_group_list("B,C")
    seeds = (20260807, 20260808, 20260809)
    jobs = expected_jobs(groups, seeds)
    assert groups == ("B", "C")
    assert len(jobs) == 18
    assert set(group for _, group, _ in jobs) == {"B", "C"}


def test_group_subset_rejects_duplicates_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="unique group list"):
        parse_group_list("B,B")
    with pytest.raises(ValueError, match="unknown residual-calibration groups"):
        parse_group_list("B,E")
