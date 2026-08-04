from __future__ import annotations

import inspect

import numpy as np
import pytest

from cnbr_fog.gru_mode_analysis import (
    FEATURE_STATISTICS,
    GRUContextModeAnalyzer,
    context_feature_names,
    extract_context_features,
    per_cluster_rmse,
    summarize_cluster_assignments,
    summarize_train_validation_clusters,
)


def _three_mode_contexts(
    *,
    windows_per_mode: int = 30,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    rows: list[np.ndarray] = []
    for mode, offset in enumerate((-5.0, 0.0, 5.0)):
        for _ in range(windows_per_mode):
            phase = rng.normal(0.0, 0.03)
            signal = np.stack(
                [
                    offset
                    + (1.0 + channel * 0.04) * np.sin(time + phase + channel * 0.02)
                    + mode * 0.08 * np.cos(2.0 * time)
                    for channel in range(9)
                ],
                axis=0,
            )
            signal += rng.normal(0.0, 0.025, size=signal.shape)
            rows.append(signal.astype(np.float32))
    return np.stack(rows)


def test_context_features_are_interpretable_and_have_fixed_geometry() -> None:
    contexts = np.zeros((2, 9, 128), dtype=np.float32)
    contexts[0, 0] = np.arange(128, dtype=np.float32)
    contexts[0, 1] = 3.0
    contexts[1] = -2.0

    features = extract_context_features(contexts)
    names = context_feature_names()
    assert features.shape == (2, 9 * len(FEATURE_STATISTICS))
    assert len(names) == features.shape[1] == 63
    lookup = {name: index for index, name in enumerate(names)}
    ramp = np.arange(128, dtype=np.float64)
    assert features[0, lookup["mean__channel_00"]] == pytest.approx(ramp.mean())
    assert features[0, lookup["std__channel_00"]] == pytest.approx(ramp.std())
    assert features[0, lookup["rms__channel_00"]] == pytest.approx(
        np.sqrt(np.mean(ramp**2))
    )
    assert features[0, lookup["diff_rms__channel_00"]] == pytest.approx(1.0)
    assert features[0, lookup["mean_abs_diff__channel_00"]] == pytest.approx(1.0)
    assert features[0, lookup["range__channel_00"]] == pytest.approx(127.0)
    assert features[0, lookup["endpoint_delta__channel_00"]] == pytest.approx(
        127.0
    )
    assert features[0, lookup["std__channel_01"]] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "contexts",
    [
        np.zeros((4, 9, 127), dtype=np.float32),
        np.zeros((4, 8, 128), dtype=np.float32),
        np.zeros((9, 128), dtype=np.float32),
        np.zeros((0, 9, 128), dtype=np.float32),
    ],
)
def test_context_feature_extraction_rejects_wrong_shape(contexts: np.ndarray) -> None:
    with pytest.raises(ValueError, match="contexts"):
        extract_context_features(contexts)


def test_context_feature_extraction_rejects_nonfinite_values() -> None:
    contexts = np.zeros((3, 9, 128), dtype=np.float32)
    contexts[1, 2, 4] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_context_features(contexts)


def test_fit_is_train_only_and_validation_assignment_is_frozen() -> None:
    train = _three_mode_contexts(windows_per_mode=30, seed=1)
    validation = _three_mode_contexts(windows_per_mode=8, seed=2)
    analyzer = GRUContextModeAnalyzer(
        k_candidates=(2, 3),
        min_cluster_fraction=0.15,
        pca_components=0.95,
        random_state=7,
    )

    assert tuple(inspect.signature(analyzer.fit).parameters) == ("train_contexts",)
    with pytest.raises(TypeError):
        analyzer.fit(train, np.zeros(len(train), dtype=np.int8))

    train_assignments = analyzer.fit_predict_train(train)
    scaler_mean = analyzer.scaler_.mean_.copy()
    pca_components = analyzer.pca_.components_.copy()
    cluster_centers = analyzer.kmeans_.cluster_centers_.copy()
    validation_assignments = analyzer.assign(validation)

    assert analyzer.selected_k_ == 3
    assert np.bincount(train_assignments, minlength=3).tolist() == [30, 30, 30]
    assert np.bincount(validation_assignments, minlength=3).tolist() == [8, 8, 8]
    np.testing.assert_array_equal(analyzer.scaler_.mean_, scaler_mean)
    np.testing.assert_array_equal(analyzer.pca_.components_, pca_components)
    np.testing.assert_array_equal(analyzer.kmeans_.cluster_centers_, cluster_centers)
    with pytest.raises(RuntimeError, match="already fitted"):
        analyzer.fit(validation)

    summary = analyzer.fit_summary()
    assert summary["fit_data"] == "training_contexts_only"
    assert summary["target_or_label_used"] is False
    assert summary["selected_k"] == 3
    assert summary["pca"]["retained_components"] >= 1


def test_k_selection_enforces_minimum_cluster_fraction() -> None:
    train = _three_mode_contexts(windows_per_mode=30, seed=5)
    analyzer = GRUContextModeAnalyzer(
        k_candidates=(3, 8),
        # k=8 is mathematically unable to put >=13% of 90 rows in every cluster.
        min_cluster_fraction=0.13,
        pca_components=None,
        random_state=11,
    ).fit(train)
    assert analyzer.selected_k_ == 3
    diagnostics = {row["k"]: row for row in analyzer.candidate_diagnostics_}
    assert diagnostics[3]["eligible"] is True
    assert diagnostics[8]["eligible"] is False
    assert diagnostics[8]["reason"] == "minimum_cluster_fraction"
    assert diagnostics[8]["minimum_cluster_fraction"] < 0.13


def test_assignment_and_rmse_summaries_include_empty_validation_cluster() -> None:
    train_assignments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    validation_assignments = np.asarray([0, 0, 1, 1], dtype=np.int64)
    prevalence = summarize_train_validation_clusters(
        train_assignments,
        validation_assignments,
        n_clusters=3,
    )
    assert [row["window_count"] for row in prevalence["train"]] == [2, 2, 2]
    assert [row["window_count"] for row in prevalence["validation"]] == [2, 2, 0]
    assert summarize_cluster_assignments(
        validation_assignments, n_clusters=3
    )[2]["window_fraction"] == 0.0

    target = np.zeros((4, 9, 16), dtype=np.float32)
    prediction = np.empty_like(target)
    prediction[:2] = 1.0
    prediction[2:] = 2.0
    rows = per_cluster_rmse(
        validation_assignments,
        target,
        prediction,
        n_clusters=3,
    )
    assert rows[0]["rmse"] == pytest.approx(1.0)
    assert rows[1]["rmse"] == pytest.approx(2.0)
    assert rows[0]["per_channel_rmse"] == pytest.approx([1.0] * 9)
    assert rows[2]["window_count"] == 0
    assert rows[2]["rmse"] is None


def test_assign_before_fit_is_rejected() -> None:
    analyzer = GRUContextModeAnalyzer(pca_components=None)
    with pytest.raises(RuntimeError, match="fitted on training contexts"):
        analyzer.assign(np.zeros((2, 9, 128), dtype=np.float32))
