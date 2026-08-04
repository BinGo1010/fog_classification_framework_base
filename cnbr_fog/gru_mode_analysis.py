"""Leakage-safe context-mode analysis for the S01 GRU predictor.

The mode model in this module is deliberately diagnostic.  It sees only the
two-second IMU context with shape ``[window, 9, 128]``.  Forecast targets, FoG
labels, and prediction errors are not accepted by :meth:`GRUContextModeAnalyzer.fit`.
All preprocessing and cluster selection are fitted on training contexts once;
validation contexts can only be transformed and assigned by the frozen model.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


EXPECTED_CHANNELS = 9
EXPECTED_CONTEXT_SAMPLES = 128
FEATURE_STATISTICS = (
    "mean",
    "std",
    "rms",
    "diff_rms",
    "mean_abs_diff",
    "range",
    "endpoint_delta",
)


def _validated_contexts(contexts: np.ndarray) -> np.ndarray:
    """Return finite float64 contexts with the fixed S01 geometry."""

    try:
        values = np.asarray(contexts, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("contexts must be a numeric [window,9,128] array") from error
    expected_tail = (EXPECTED_CHANNELS, EXPECTED_CONTEXT_SAMPLES)
    if values.ndim != 3 or tuple(values.shape[1:]) != expected_tail:
        raise ValueError(
            "contexts must have shape [window,9,128], got "
            f"{tuple(values.shape)}"
        )
    if values.shape[0] == 0:
        raise ValueError("contexts must contain at least one window")
    if not np.isfinite(values).all():
        raise ValueError("contexts contain NaN or Inf")
    return values


def _validated_channel_names(
    channel_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if channel_names is None:
        return tuple(f"channel_{index:02d}" for index in range(EXPECTED_CHANNELS))
    names = tuple(str(name) for name in channel_names)
    if len(names) != EXPECTED_CHANNELS:
        raise ValueError(f"channel_names must contain {EXPECTED_CHANNELS} names")
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("channel_names must be non-empty and unique")
    return names


def context_feature_names(
    channel_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Names for :func:`extract_context_features` in statistic-major order."""

    names = _validated_channel_names(channel_names)
    return tuple(
        f"{statistic}__{channel}"
        for statistic in FEATURE_STATISTICS
        for channel in names
    )


def extract_context_features(contexts: np.ndarray) -> np.ndarray:
    """Extract interpretable per-channel features from context only.

    The seven feature families are channel mean, population standard deviation,
    RMS amplitude, first-difference RMS, first-difference mean absolute value,
    range, and final-minus-first endpoint delta.  No target samples or labels
    are inputs to this function.
    """

    values = _validated_contexts(contexts)
    differences = np.diff(values, axis=2)
    features = (
        np.mean(values, axis=2),
        np.std(values, axis=2, ddof=0),
        np.sqrt(np.mean(np.square(values), axis=2)),
        np.sqrt(np.mean(np.square(differences), axis=2)),
        np.mean(np.abs(differences), axis=2),
        np.ptp(values, axis=2),
        values[:, :, -1] - values[:, :, 0],
    )
    result = np.concatenate(features, axis=1)
    expected = (values.shape[0], EXPECTED_CHANNELS * len(FEATURE_STATISTICS))
    if result.shape != expected or not np.isfinite(result).all():
        raise FloatingPointError("context feature extraction produced invalid values")
    return np.ascontiguousarray(result, dtype=np.float64)


def summarize_cluster_assignments(
    assignments: np.ndarray,
    *,
    n_clusters: int | None = None,
) -> list[dict[str, int | float]]:
    """Return JSON-friendly window counts and fractions for every cluster."""

    labels = np.asarray(assignments)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("assignments must be a non-empty one-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
            raise ValueError("assignments must contain integer cluster IDs")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError("assignments must contain non-negative cluster IDs")
    inferred = int(labels.max()) + 1
    if n_clusters is None:
        n_clusters = inferred
    n_clusters = int(n_clusters)
    if n_clusters <= 0 or inferred > n_clusters:
        raise ValueError("n_clusters does not cover all assignment IDs")
    counts = np.bincount(labels, minlength=n_clusters)
    total = int(labels.size)
    return [
        {
            "cluster_id": int(cluster_id),
            "window_count": int(count),
            "window_fraction": float(count / total),
        }
        for cluster_id, count in enumerate(counts)
    ]


def summarize_train_validation_clusters(
    train_assignments: np.ndarray,
    validation_assignments: np.ndarray,
    *,
    n_clusters: int,
) -> dict[str, list[dict[str, int | float]]]:
    """Compare frozen-mode prevalence between training and validation."""

    return {
        "train": summarize_cluster_assignments(
            train_assignments, n_clusters=n_clusters
        ),
        "validation": summarize_cluster_assignments(
            validation_assignments, n_clusters=n_clusters
        ),
    }


def per_cluster_rmse(
    assignments: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    n_clusters: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize forecast RMSE after frozen mode assignment.

    Targets and predictions are used only here, after clustering.  They never
    enter feature scaling, PCA, KMeans fitting, or candidate-k selection.
    Both arrays must have shape ``[window, 9, horizon]``.
    """

    labels = np.asarray(assignments)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("assignments must be a non-empty one-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
            raise ValueError("assignments must contain integer cluster IDs")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError("assignments must contain non-negative cluster IDs")

    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction must have identical shapes")
    if (
        truth.ndim != 3
        or truth.shape[0] != labels.size
        or truth.shape[1] != EXPECTED_CHANNELS
        or truth.shape[2] <= 0
    ):
        raise ValueError("target/prediction must have shape [window,9,horizon]")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("target/prediction contain NaN or Inf")

    inferred = int(labels.max()) + 1
    if n_clusters is None:
        n_clusters = inferred
    n_clusters = int(n_clusters)
    if n_clusters <= 0 or inferred > n_clusters:
        raise ValueError("n_clusters does not cover all assignment IDs")

    rows: list[dict[str, Any]] = []
    for cluster_id in range(n_clusters):
        selected = labels == cluster_id
        count = int(selected.sum())
        row: dict[str, Any] = {
            "cluster_id": cluster_id,
            "window_count": count,
            "window_fraction": float(count / labels.size),
            "mse": None,
            "rmse": None,
            "per_channel_rmse": None,
        }
        if count:
            squared = np.square(truth[selected] - estimate[selected])
            row.update(
                {
                    "mse": float(np.mean(squared)),
                    "rmse": float(np.sqrt(np.mean(squared))),
                    "per_channel_rmse": np.sqrt(
                        np.mean(squared, axis=(0, 2))
                    ).tolist(),
                }
            )
        rows.append(row)
    return rows


class GRUContextModeAnalyzer:
    """Train-only interpretable context clustering with frozen assignment."""

    def __init__(
        self,
        *,
        k_candidates: Sequence[int] = (2, 3, 4, 5, 6),
        min_cluster_fraction: float = 0.10,
        pca_components: int | float | None = 0.95,
        random_state: int = 42,
        n_init: int = 20,
        channel_names: Sequence[str] | None = None,
    ) -> None:
        candidates = tuple(sorted({int(value) for value in k_candidates}))
        if not candidates or any(value < 2 for value in candidates):
            raise ValueError("k_candidates must contain integers >= 2")
        if (
            not math.isfinite(float(min_cluster_fraction))
            or not 0.0 < float(min_cluster_fraction) <= 1.0
        ):
            raise ValueError("min_cluster_fraction must be in (0,1]")
        if isinstance(pca_components, bool):
            raise ValueError("pca_components must be None, an integer, or a float")
        if isinstance(pca_components, int):
            if pca_components <= 0:
                raise ValueError("integer pca_components must be positive")
        elif isinstance(pca_components, float):
            if not math.isfinite(pca_components) or not 0.0 < pca_components < 1.0:
                raise ValueError("float pca_components must be in (0,1)")
        elif pca_components is not None:
            raise ValueError("pca_components must be None, an integer, or a float")
        if int(n_init) <= 0:
            raise ValueError("n_init must be positive")

        self.k_candidates = candidates
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.pca_components = pca_components
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.channel_names = _validated_channel_names(channel_names)
        self._fitted = False

    def _check_not_fitted(self) -> None:
        if self._fitted:
            raise RuntimeError(
                "mode analyzer is already fitted; validation data must use assign()"
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("mode analyzer must be fitted on training contexts first")

    def _preprocess_fit(self, features: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler()
        transformed = self.scaler_.fit_transform(features)
        self.pca_: PCA | None = None
        if self.pca_components is not None:
            if isinstance(self.pca_components, int) and self.pca_components > min(
                transformed.shape
            ):
                raise ValueError(
                    "integer pca_components exceeds the train feature-matrix rank bound"
                )
            self.pca_ = PCA(
                n_components=self.pca_components,
                svd_solver="full",
                random_state=self.random_state,
            )
            transformed = self.pca_.fit_transform(transformed)
        if not np.isfinite(transformed).all():
            raise FloatingPointError("train-only preprocessing produced invalid values")
        return np.ascontiguousarray(transformed, dtype=np.float64)

    def _preprocess_frozen(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler_.transform(features)
        if self.pca_ is not None:
            transformed = self.pca_.transform(transformed)
        if not np.isfinite(transformed).all():
            raise FloatingPointError("frozen preprocessing produced invalid values")
        return np.ascontiguousarray(transformed, dtype=np.float64)

    @staticmethod
    def _canonical_mapping(centers: np.ndarray) -> np.ndarray:
        order = sorted(range(len(centers)), key=lambda index: tuple(centers[index]))
        mapping = np.empty(len(order), dtype=np.int64)
        for canonical, raw in enumerate(order):
            mapping[raw] = canonical
        return mapping

    def fit(self, train_contexts: np.ndarray) -> "GRUContextModeAnalyzer":
        """Fit all mode components using training contexts only.

        The signature intentionally has no ``y``, target, label, prediction, or
        residual argument.  Calling this method a second time is rejected so a
        validation assignment cannot silently refit preprocessing or clusters.
        """

        self._check_not_fitted()
        features = extract_context_features(train_contexts)
        if features.shape[0] < 3:
            raise ValueError("at least three training contexts are required")
        transformed = self._preprocess_fit(features)

        candidates: list[dict[str, Any]] = []
        fitted: dict[int, tuple[KMeans, np.ndarray, float]] = {}
        for k in self.k_candidates:
            row: dict[str, Any] = {
                "k": int(k),
                "eligible": False,
                "silhouette": None,
                "inertia": None,
                "cluster_fractions": None,
                "minimum_cluster_fraction": None,
                "reason": None,
            }
            if k >= transformed.shape[0]:
                row["reason"] = "k_must_be_smaller_than_training_windows"
                candidates.append(row)
                continue
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                algorithm="lloyd",
            )
            raw = model.fit_predict(transformed)
            counts = np.bincount(raw, minlength=k)
            fractions = counts.astype(np.float64) / float(len(raw))
            minimum = float(fractions.min())
            row.update(
                {
                    "inertia": float(model.inertia_),
                    "cluster_fractions": fractions.tolist(),
                    "minimum_cluster_fraction": minimum,
                }
            )
            if np.count_nonzero(counts) != k:
                row["reason"] = "empty_cluster"
            elif minimum + 1e-12 < self.min_cluster_fraction:
                row["reason"] = "minimum_cluster_fraction"
            else:
                score = float(silhouette_score(transformed, raw))
                row.update(
                    {
                        "eligible": True,
                        "silhouette": score,
                        "reason": "eligible",
                    }
                )
                fitted[k] = (model, raw, score)
            candidates.append(row)

        if not fitted:
            details = ", ".join(
                f"k={row['k']}:{row['reason']}" for row in candidates
            )
            raise RuntimeError(
                "no candidate k satisfied the minimum cluster-fraction constraint; "
                + details
            )
        selected_k = max(fitted, key=lambda k: (fitted[k][2], -k))
        model, raw_labels, selected_score = fitted[selected_k]
        mapping = self._canonical_mapping(model.cluster_centers_)

        self.feature_names_ = context_feature_names(self.channel_names)
        self.kmeans_ = model
        self.selected_k_ = int(selected_k)
        self.selected_silhouette_ = float(selected_score)
        self.raw_to_canonical_cluster_ = mapping
        self.train_assignments_ = mapping[raw_labels]
        self.candidate_diagnostics_ = deepcopy(candidates)
        self.train_cluster_summary_ = summarize_cluster_assignments(
            self.train_assignments_, n_clusters=self.selected_k_
        )
        self._fitted = True
        return self

    def fit_predict_train(self, train_contexts: np.ndarray) -> np.ndarray:
        """Fit on training contexts and return their canonical mode IDs."""

        self.fit(train_contexts)
        return self.train_assignments_.copy()

    def transform(self, contexts: np.ndarray) -> np.ndarray:
        """Apply the frozen train-fitted StandardScaler and optional PCA."""

        self._check_fitted()
        return self._preprocess_frozen(extract_context_features(contexts))

    def assign(self, contexts: np.ndarray) -> np.ndarray:
        """Assign contexts with the frozen train-fitted mode model."""

        transformed = self.transform(contexts)
        raw = self.kmeans_.predict(transformed)
        return self.raw_to_canonical_cluster_[raw]

    def fit_summary(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the train-only mode fit."""

        self._check_fitted()
        pca_summary: dict[str, Any] | None = None
        if self.pca_ is not None:
            pca_summary = {
                "requested_components": self.pca_components,
                "retained_components": int(self.pca_.n_components_),
                "explained_variance_ratio": self.pca_.explained_variance_ratio_.tolist(),
                "total_explained_variance": float(
                    self.pca_.explained_variance_ratio_.sum()
                ),
            }
        return {
            "fit_data": "training_contexts_only",
            "target_or_label_used": False,
            "context_shape": [EXPECTED_CHANNELS, EXPECTED_CONTEXT_SAMPLES],
            "feature_statistics": list(FEATURE_STATISTICS),
            "feature_count": len(self.feature_names_),
            "feature_names": list(self.feature_names_),
            "standard_scaler_fitted_on": "training_context_features_only",
            "pca": pca_summary,
            "k_candidates": list(self.k_candidates),
            "minimum_cluster_fraction_constraint": self.min_cluster_fraction,
            "candidate_diagnostics": deepcopy(self.candidate_diagnostics_),
            "selected_k": self.selected_k_,
            "selected_silhouette": self.selected_silhouette_,
            "train_cluster_summary": deepcopy(self.train_cluster_summary_),
        }


__all__ = [
    "EXPECTED_CHANNELS",
    "EXPECTED_CONTEXT_SAMPLES",
    "FEATURE_STATISTICS",
    "GRUContextModeAnalyzer",
    "context_feature_names",
    "extract_context_features",
    "per_cluster_rmse",
    "summarize_cluster_assignments",
    "summarize_train_validation_clusters",
]
