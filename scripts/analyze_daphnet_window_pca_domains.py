#!/usr/bin/env python
"""Window-level PCA analysis of cross-subject domain shift in Daphnet.

The primary PCA is fitted only on an equal number of pure non-FOG windows
from every subject.  This makes subject-domain separation interpretable
without confounding it with the large between-subject differences in FOG
prevalence.  Pure FOG/non-FOG windows from the eight positive subjects are
then projected into the same PCA space with a balanced subject-by-label
design.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_feature_extractor_class():
    module_path = REPO_ROOT / "daphnet_baselines" / "features.py"
    spec = importlib.util.spec_from_file_location("_daphnet_pca_features", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature extractor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TimeFrequencyFeatureExtractor


TimeFrequencyFeatureExtractor = load_feature_extractor_class()


@dataclass(frozen=True)
class RecordView:
    record_id: str
    subject_id: str
    run_id: str
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray


def true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def valid_signal_mask(
    x: np.ndarray,
    sampling_rate_hz: int,
    flatline_seconds: float = 1.0,
    zero_tolerance: float = 1e-8,
) -> np.ndarray:
    valid = np.isfinite(x).all(axis=1)
    minimum = max(1, int(round(flatline_seconds * sampling_rate_hz)))
    groups = (
        [x[:, start : start + 3] for start in range(0, x.shape[1], 3)]
        if x.shape[1] % 3 == 0
        else [x]
    )
    for group in groups:
        zero = np.max(np.abs(group), axis=1) <= zero_tolerance
        for start, end in true_runs(zero):
            if end - start >= minimum:
                valid[start:end] = False
    return valid


@dataclass(frozen=True)
class DatasetView:
    records: list[RecordView]
    sampling_rate_hz: int
    channel_names: tuple[str, ...]
    subjects: list[str]
    n_channels: int

    @classmethod
    def load(cls, root: Path) -> "DatasetView":
        root = Path(root)
        schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
        channel_names = tuple(str(item["name"]) for item in schema["channels"])
        records: list[RecordView] = []
        sampling_rates: set[int] = set()
        with (root / "manifest.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if str(row.get("usable", "true")).strip().lower() not in {
                    "1", "true", "yes"
                }:
                    continue
                with np.load(root / row["record_path"], allow_pickle=False) as payload:
                    x = np.asarray(payload["x"], dtype=np.float32)
                    y = np.asarray(payload["y_binary"], dtype=np.int8)
                rate = int(row["sampling_rate_hz"])
                sampling_rates.add(rate)
                if x.ndim != 2 or y.shape != (len(x),):
                    raise ValueError(f"Invalid record shape for {row['record_id']}")
                if x.shape[1] != len(channel_names):
                    raise ValueError(f"Channel mismatch for {row['record_id']}")
                if len(x) != int(row["n_samples"]):
                    raise ValueError(f"Manifest sample mismatch for {row['record_id']}")
                records.append(
                    RecordView(
                        record_id=str(row["record_id"]),
                        subject_id=str(row["subject_id"]),
                        run_id=str(row["run_id"]),
                        x=x,
                        y=y,
                        valid=valid_signal_mask(x, rate),
                    )
                )
        if len(sampling_rates) != 1 or not records:
            raise ValueError("Expected one sampling rate and at least one record")
        return cls(
            records=records,
            sampling_rate_hz=sampling_rates.pop(),
            channel_names=channel_names,
            subjects=sorted({record.subject_id for record in records}),
            n_channels=len(channel_names),
        )


DEFAULT_DATASET = Path("dataset/1.Daphnet Freezing of Gait Dataset/processed")
DEFAULT_OUTPUT = Path("outputs/daphnet_window_pca_domain_analysis")
SUBJECT_ORDER = tuple(f"S{index:02d}" for index in range(1, 11))
LABEL_NAMES = {0: "non-FOG", 1: "FOG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-nonfog-per-subject",
        type=int,
        default=500,
        help="Maximum equal non-FOG windows sampled from each of 10 subjects.",
    )
    parser.add_argument(
        "--max-per-subject-class",
        type=int,
        default=150,
        help="Maximum equal windows per subject and class in the 8-subject panel.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument(
        "--sensitivity-repeats",
        type=int,
        default=100,
        help="Repeated balanced resamples projected through the primary PCA.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("window and stride seconds must be positive")
    if args.max_nonfog_per_subject <= 0 or args.max_per_subject_class <= 0:
        raise ValueError("sampling caps must be positive")
    if args.sensitivity_repeats <= 0:
        raise ValueError("sensitivity repeats must be positive")


def build_pure_window_inventory(
    dataset: DatasetView,
    window_samples: int,
    stride_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record_index, record in enumerate(dataset.records):
        n_samples = len(record.y)
        if n_samples < window_samples:
            continue
        invalid_prefix = np.r_[0, np.cumsum(~record.valid, dtype=np.int64)]
        fog_prefix = np.r_[0, np.cumsum(record.y == 1, dtype=np.int64)]
        for start in range(0, n_samples - window_samples + 1, stride_samples):
            end = start + window_samples
            if invalid_prefix[end] - invalid_prefix[start] != 0:
                continue
            fog_count = int(fog_prefix[end] - fog_prefix[start])
            if fog_count not in {0, window_samples}:
                continue
            rows.append(
                {
                    "window_id": len(rows),
                    "subject_id": record.subject_id,
                    "record_id": record.record_id,
                    "run_id": record.run_id,
                    "record_index": record_index,
                    "start": start,
                    "end": end,
                    "label": int(fog_count == window_samples),
                }
            )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise RuntimeError("No valid pure-label windows were found")
    return inventory


def balanced_sample(
    inventory: pd.DataFrame,
    group_columns: list[str],
    n_per_group: int,
    seed: int,
) -> pd.DataFrame:
    pieces = []
    for group_key, group in inventory.groupby(group_columns, sort=True):
        if len(group) < n_per_group:
            raise ValueError(
                f"Group {group_key!r} has {len(group)} rows, fewer than {n_per_group}"
            )
        pieces.append(group.sample(n=n_per_group, random_state=seed))
    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(group_columns + ["window_id"]).reset_index(drop=True)


def materialize_windows(dataset: DatasetView, metadata: pd.DataFrame) -> np.ndarray:
    window_samples = int(metadata["end"].iloc[0] - metadata["start"].iloc[0])
    result = np.empty(
        (len(metadata), dataset.n_channels, window_samples), dtype=np.float32
    )
    for output_index, row in enumerate(metadata.itertuples(index=False)):
        record = dataset.records[int(row.record_index)]
        result[output_index] = record.x[int(row.start) : int(row.end)].T
    return result


def extract_features(
    dataset: DatasetView,
    metadata: pd.DataFrame,
    batch_size: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    extractor = TimeFrequencyFeatureExtractor(
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
        include_triad_magnitudes=True,
        batch_size=batch_size,
    )
    features = extractor.transform(materialize_windows(dataset, metadata))
    return features, tuple(extractor.feature_names())


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    color: object,
    confidence: float = 0.80,
) -> None:
    if len(x) < 3:
        return
    covariance = np.cov(np.stack([x, y], axis=0))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    # Chi-square quantile with 2 degrees of freedom: -2 log(1-p).
    scale = -2.0 * math.log(1.0 - confidence)
    width, height = 2.0 * np.sqrt(eigenvalues * scale)
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(
        Ellipse(
            (float(np.mean(x)), float(np.mean(y))),
            width=float(width),
            height=float(height),
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.15,
            alpha=0.85,
        )
    )


def multivariate_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = values.mean(axis=0)
    total_ss = float(np.square(values - grand).sum())
    between_ss = 0.0
    for group in np.unique(groups):
        selected = values[groups == group]
        between_ss += len(selected) * float(np.square(selected.mean(axis=0) - grand).sum())
    return between_ss / total_ss if total_ss > 0 else 0.0


def balanced_two_way_decomposition(
    values: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    grand = values.mean(axis=0)
    total_ss = float(np.square(values - grand).sum())
    subject_means = {
        subject: values[subjects == subject].mean(axis=0)
        for subject in np.unique(subjects)
    }
    label_means = {
        label: values[labels == label].mean(axis=0) for label in np.unique(labels)
    }
    subject_ss = sum(
        int(np.sum(subjects == subject))
        * float(np.square(mean - grand).sum())
        for subject, mean in subject_means.items()
    )
    label_ss = sum(
        int(np.sum(labels == label)) * float(np.square(mean - grand).sum())
        for label, mean in label_means.items()
    )
    interaction_ss = 0.0
    within_ss = 0.0
    for subject in np.unique(subjects):
        for label in np.unique(labels):
            mask = (subjects == subject) & (labels == label)
            cell = values[mask]
            cell_mean = cell.mean(axis=0)
            interaction = (
                cell_mean - subject_means[subject] - label_means[label] + grand
            )
            interaction_ss += len(cell) * float(np.square(interaction).sum())
            within_ss += float(np.square(cell - cell_mean).sum())
    components = {
        "subject_main_effect": subject_ss,
        "label_main_effect": label_ss,
        "subject_label_interaction": interaction_ss,
        "within_cell": within_ss,
    }
    return {
        f"{name}_fraction": value / total_ss if total_ss > 0 else 0.0
        for name, value in components.items()
    } | {
        "decomposition_residual": (
            total_ss - sum(components.values())
        )
        / total_ss
        if total_ss > 0
        else 0.0
    }


def subject_centroid_distances(
    scores: np.ndarray,
    subjects: np.ndarray,
    order: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, float]]:
    centroids = {
        subject: scores[subjects == subject].mean(axis=0) for subject in order
    }
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    for first_index, first in enumerate(order):
        for second_index, second in enumerate(order):
            matrix[first_index, second_index] = np.linalg.norm(
                centroids[first] - centroids[second]
            )
    frame = pd.DataFrame(matrix, index=order, columns=order)
    nonzero = [
        (matrix[i, j], order[i], order[j])
        for i in range(len(order))
        for j in range(i + 1, len(order))
    ]
    closest = min(nonzero)
    farthest = max(nonzero)
    global_centroid = scores.mean(axis=0)
    global_distances = {
        subject: float(np.linalg.norm(centroids[subject] - global_centroid))
        for subject in order
    }
    outlier = max(global_distances, key=global_distances.get)
    mean_pairwise = float(np.mean([item[0] for item in nonzero]))
    within_radii = []
    for subject in order:
        selected = scores[subjects == subject]
        within_radii.extend(np.linalg.norm(selected - centroids[subject], axis=1))
    summary = {
        "closest_distance": float(closest[0]),
        "closest_subject_a": closest[1],
        "closest_subject_b": closest[2],
        "farthest_distance": float(farthest[0]),
        "farthest_subject_a": farthest[1],
        "farthest_subject_b": farthest[2],
        "most_outlying_subject": outlier,
        "most_outlying_distance": global_distances[outlier],
        "mean_pairwise_centroid_distance": mean_pairwise,
        "mean_within_subject_radius": float(np.mean(within_radii)),
        "centroid_to_within_ratio": mean_pairwise / float(np.mean(within_radii)),
    }
    return frame, summary


def save_inventory_plot(
    counts: pd.DataFrame,
    output_path: Path,
    palette: dict[str, object],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(counts))
    ax.bar(x, counts["non-FOG"], color="#9ab6d3", label="non-FOG")
    ax.bar(
        x,
        counts["FOG"],
        bottom=counts["non-FOG"],
        color="#d95f5f",
        label="FOG",
    )
    ax.set_xticks(x, counts.index)
    ax.set_ylabel("Eligible pure-label windows")
    ax.set_xlabel("Subject")
    ax.set_title("Daphnet 2 s / 1 s-stride pure-window inventory", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    maximum_total = float(counts.sum(axis=1).max())
    ax.set_ylim(0, maximum_total * 1.13)
    for index, subject in enumerate(counts.index):
        total = float(counts.loc[subject].sum())
        fog_fraction = float(counts.loc[subject, "FOG"] / total) if total else 0.0
        ax.text(
            index,
            total + maximum_total * 0.015,
            f"{fog_fraction:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=palette[subject],
            fontweight="bold",
        )
    ax.text(
        0.01,
        0.98,
        "Labels above bars are pure-window FOG fractions",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scatter_subject_domains(
    scores: np.ndarray,
    subjects: np.ndarray,
    explained: np.ndarray,
    output_path: Path,
    palette: dict[str, object],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    for subject in SUBJECT_ORDER:
        mask = subjects == subject
        color = palette[subject]
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=10,
            alpha=0.25,
            color=color,
            edgecolors="none",
            label=subject,
        )
        confidence_ellipse(scores[mask, 0], scores[mask, 1], ax, color)
        centroid = scores[mask, :2].mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1], s=95, marker="X", color=color,
            edgecolor="white", linewidth=0.8, zorder=5,
        )
        ax.annotate(subject, centroid, xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title(
        "Patient domains in pure non-FOG windows\n"
        "equal windows per subject; X = centroid, ellipse = 80% covariance contour",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scatter_balanced_labels(
    scores: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
    output_path: Path,
    palette: dict[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8), sharex=True, sharey=True)
    positive_subjects = tuple(subject for subject in SUBJECT_ORDER if subject in set(subjects))
    markers = {0: "o", 1: "^"}
    for subject in positive_subjects:
        for label in (0, 1):
            mask = (subjects == subject) & (labels == label)
            axes[0].scatter(
                scores[mask, 0], scores[mask, 1], s=12, alpha=0.28,
                marker=markers[label], color=palette[subject], edgecolors="none",
            )
        centroid = scores[subjects == subject, :2].mean(axis=0)
        axes[0].scatter(
            centroid[0], centroid[1], s=90, marker="X", color=palette[subject],
            edgecolor="white", linewidth=0.8, zorder=5,
        )
        axes[0].annotate(subject, centroid, xytext=(4, 4), textcoords="offset points", fontsize=8)
    label_colors = {0: "#4c78a8", 1: "#e45756"}
    for label in (0, 1):
        mask = labels == label
        axes[1].scatter(
            scores[mask, 0], scores[mask, 1], s=13, alpha=0.30,
            marker=markers[label], color=label_colors[label], edgecolors="none",
            label=LABEL_NAMES[label],
        )
        centroid = scores[mask, :2].mean(axis=0)
        axes[1].scatter(
            centroid[0], centroid[1], s=120, marker="X",
            color=label_colors[label], edgecolor="white", linewidth=1.0, zorder=5,
        )
        confidence_ellipse(scores[mask, 0], scores[mask, 1], axes[1], label_colors[label])
    subject_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=palette[s], label=s, markersize=6)
        for s in positive_subjects
    ]
    label_handles = [
        Line2D([0], [0], marker=markers[l], linestyle="", color="#555555", label=LABEL_NAMES[l], markersize=7)
        for l in (0, 1)
    ]
    axes[0].legend(
        handles=subject_handles + label_handles,
        frameon=False,
        ncol=2,
        fontsize=8,
        loc="best",
    )
    axes[1].legend(frameon=False, loc="best")
    axes[0].set_title("Colored by patient domain", fontweight="bold")
    axes[1].set_title("Colored by state label", fontweight="bold")
    for ax in axes:
        ax.axhline(0, color="#bbbbbb", linewidth=0.6)
        ax.axvline(0, color="#bbbbbb", linewidth=0.6)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} non-FOG variance)")
        sns.despine(ax=ax)
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1%} non-FOG variance)")
    fig.suptitle(
        "Balanced FOG/non-FOG windows projected into the non-FOG PCA space\n"
        "equal windows per patient and class; S04/S10 omitted because they have no FOG",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_distance_heatmap(distances: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    sns.heatmap(
        distances,
        mask=mask,
        cmap="mako_r",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Euclidean centroid distance in retained PC space"},
        ax=ax,
    )
    ax.set_title("Pairwise patient-domain distance (pure non-FOG)", fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_variance_and_loadings_plot(
    explained: np.ndarray,
    loadings: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    component_numbers = np.arange(1, min(20, len(explained)) + 1)
    axes[0].bar(component_numbers, explained[: len(component_numbers)] * 100, color="#4c78a8")
    axes[0].plot(
        component_numbers,
        np.cumsum(explained[: len(component_numbers)]) * 100,
        color="#e45756",
        marker="o",
        markersize=3,
    )
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance / cumulative variance (%)")
    axes[0].set_title("PCA explained variance", fontweight="bold")
    top = loadings.head(15).sort_values("absolute_loading")
    axes[1].barh(top["feature"], top["absolute_loading"], color="#72b7b2")
    axes[1].set_xlabel("Max absolute loading across PC1-PC2")
    axes[1].set_title("Features driving the first two PCs", fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=8)
    sns.despine(ax=axes[0])
    sns.despine(ax=axes[1])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    inventory_path: Path,
    nonfog_path: Path,
    labels_path: Path,
    distance_path: Path,
    output_path: Path,
) -> None:
    images = [
        plt.imread(inventory_path),
        plt.imread(nonfog_path),
        plt.imread(labels_path),
        plt.imread(distance_path),
    ]
    titles = [
        "A. Pure-window inventory",
        "B. non-FOG patient domains",
        "C. Balanced state comparison",
        "D. Patient centroid distances",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
        ax.axis("off")
    fig.suptitle("Daphnet window-level PCA patient-domain analysis", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_report(
    output_path: Path,
    args: argparse.Namespace,
    inventory_counts: pd.DataFrame,
    metrics: dict[str, object],
    top_loadings: pd.DataFrame,
) -> None:
    pca = metrics["pca"]
    domain = metrics["nonfog_domain"]
    effects = metrics["balanced_subject_label_effects"]
    sample = metrics["sampling"]
    sensitivity = metrics["resampling_sensitivity"]
    subject_effect = float(effects["subject_main_effect_fraction"])
    label_effect = float(effects["label_main_effect_fraction"])
    relative = subject_effect / label_effect if label_effect > 0 else float("inf")
    silhouette = float(domain["silhouette_subject"])
    if silhouette >= 0.50:
        silhouette_text = "强"
    elif silhouette >= 0.25:
        silhouette_text = "中等"
    elif silhouette > 0:
        silhouette_text = "弱但可检测"
    else:
        silhouette_text = "未形成清晰聚类"
    driver_text = "、".join(top_loadings.head(8)["feature"].tolist())
    count_lines = []
    for subject in SUBJECT_ORDER:
        count_lines.append(
            f"| {subject} | {int(inventory_counts.loc[subject, 'non-FOG']):,} | "
            f"{int(inventory_counts.loc[subject, 'FOG']):,} | "
            f"{inventory_counts.loc[subject, 'FOG'] / inventory_counts.loc[subject].sum():.2%} |"
        )
    report = f"""# Daphnet 窗口级 PCA 患者域偏移分析

## 结论摘要

本分析在排除标签比例混杂后，观察到 **患者域存在系统性中心偏移，但{silhouette_text}**。纯 non-FOG PCA 中，患者身份解释了保留主成分空间总变异的 **{float(domain['subject_eta_squared']):.2%}**，患者聚类轮廓系数为 **{silhouette:.3f}**。这说明即使比较相同的 non-FOG 状态，10 位患者的 IMU 时频分布仍不相同；这种偏移表现为相互重叠的连续域，而不是彼此孤立的患者簇。

在 8 位存在 FOG 的患者中，使用严格等量的“患者 × 标签”窗口，并在以 non-FOG 为基准的 PCA 空间中进行双因素分解：患者主效应为 **{subject_effect:.2%}**，FOG/non-FOG 标签主效应为 **{label_effect:.2%}**；患者效应约为标签效应的 **{relative:.2f} 倍**。因此该数据集的窗口特征不仅包含冻结状态，也包含明显的患者特异性信息，跨患者建模存在真实的域偏移风险。

为避免结论依赖某一次随机抽样，额外进行了 **{sensitivity['repeat_count']} 次**等量重抽样。在固定的主 PCA 坐标系中，患者身份解释率中位数为 **{float(sensitivity['subject_eta_squared_median']):.2%}**（5%–95%范围 {float(sensitivity['subject_eta_squared_p05']):.2%}–{float(sensitivity['subject_eta_squared_p95']):.2%}）；患者/标签主效应比中位数为 **{float(sensitivity['subject_to_label_ratio_median']):.2f}**，其中 **{int(sensitivity['subject_effect_greater_count'])}/{sensitivity['repeat_count']}** 次患者主效应大于标签主效应。结论对窗口抽样具有稳定性。

患者域质心距离最远的是 **{domain['farthest_subject_a']}–{domain['farthest_subject_b']}**（{float(domain['farthest_distance']):.2f}），最近的是 **{domain['closest_subject_a']}–{domain['closest_subject_b']}**（{float(domain['closest_distance']):.2f}）；距离全体中心最远的患者是 **{domain['most_outlying_subject']}**。平均患者质心距离与平均域内半径之比为 **{float(domain['centroid_to_within_ratio']):.3f}**，表明患者域存在重叠，但中心位置系统性偏移。

## 方法

- 数据：Daphnet processed，64 Hz，9 个加速度通道。
- 窗口：{args.window_seconds:g} 秒（{metrics['window']['window_samples']} 点），步长 {args.stride_seconds:g} 秒（{metrics['window']['stride_samples']} 点）。
- 仅保留窗口内标签全部为 non-FOG 或全部为 FOG 的纯标签窗口；跨边界窗口被排除。
- 继续沿用仓库的无效信号规则，排除任一传感器三轴长时间全零和非有限值窗口。
- 特征：9 个物理通道加踝/大腿/躯干三轴模长，共 {metrics['features']['feature_count']} 个时域、频域和轴间相关特征。
- 主 PCA：每位患者等量抽取 {sample['nonfog_per_subject']} 个纯 non-FOG 窗口，用 RobustScaler 缩放并裁剪到 ±12，再拟合 PCA。这样患者分离不会由 FOG 比例不同造成。
- 标签对照：仅选择存在 FOG 的 8 位患者，每位患者每类等量抽取 {sample['per_subject_class']} 个窗口，投影到同一个 non-FOG PCA 空间。
- PCA 前两维解释率为 **{float(pca['pc1_explained_variance']):.2%} + {float(pca['pc2_explained_variance']):.2%}**；前 {pca['retained_components']} 个主成分累计解释 **{float(pca['retained_explained_variance']):.2%}**。量化结论使用保留的多维空间，而不是只使用二维图。

## 可用窗口分布

| 患者 | 纯 non-FOG窗口 | 纯FOG窗口 | 纯窗口FOG比例 |
|---|---:|---:|---:|
{chr(10).join(count_lines)}

S04、S10 没有 FOG 窗口，因此只能进入 non-FOG 患者域分析，不能进入患者×标签平衡分析。

## 图像解读

### 1. 患者域 PCA

![Pure non-FOG patient domains](02_pca_nonfog_subject_domains.png)

每个点是一个 2 秒纯 non-FOG 窗口，颜色表示患者，X 表示患者质心，椭圆表示 80% 二维协方差轮廓。不同患者云团可以重叠，但质心和协方差范围的系统差异构成患者域偏移。

### 2. 平衡后的 FOG/non-FOG 对照

![Balanced state comparison](03_pca_balanced_labels.png)

左图按患者着色，右图按标签着色。两图来自完全相同的平衡窗口和同一个 PCA 坐标系，因此可以直接比较“患者分离”与“状态分离”。

### 3. 患者域距离

![Patient centroid distances](04_nonfog_domain_distance_heatmap.png)

热图基于纯 non-FOG 窗口在保留 PCA 空间中的患者质心欧氏距离。它描述的是特征域距离，不等同于临床症状严重程度。

### 4. PCA 方差及主要载荷

![PCA variance and loadings](05_pca_variance_and_loadings.png)

PC1/PC2 的主要驱动特征包括：{driver_text}。载荷表示与主成分方向的关联，不能解释为因果效应。

## 建模含义

1. 随机按窗口划分训练/测试集会把同一患者甚至相邻重叠窗口放在两侧，从而高估性能；应采用 LOSO 或严格按患者划分。
2. 标准化、特征选择、PCA 和类别权重都必须只在每个 LOSO 训练折中拟合。本文为了描述整个数据集的总体域结构而在全体患者上拟合，不能作为预测性能证据。
3. S04、S10 无 FOG，S08 的 FOG 比例又明显较高；跨患者指标应报告逐患者结果和宏平均，不能只报告池化指标。
4. 建议下一步把患者质心距离或其他域距离与逐患者 LOSO AUPRC/F1 关联：若距离越远的测试患者性能越低，可进一步确认域偏移是泛化下降的重要来源。

## 局限性

- PCA 是线性投影，PC1/PC2 只显示部分方差；因此数值结论使用了多维保留空间。
- 2 秒窗口、1 秒步长产生 50% 重叠，相邻点并非独立样本；本报告不据此给出显著性 p 值。
- 患者域差异可能同时来自个体生理、传感器佩戴、任务构成和症状严重程度，PCA 无法区分这些原因。
- 等量抽样回答的是“控制样本量和标签后是否仍有域偏移”，不是原始部署流量中的自然类别分布。
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    sns.set_theme(style="whitegrid", context="notebook")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetView.load(args.processed_dir)
    if tuple(dataset.subjects) != SUBJECT_ORDER:
        raise ValueError(f"Expected subjects {SUBJECT_ORDER}, got {dataset.subjects}")
    window_samples = int(round(args.window_seconds * dataset.sampling_rate_hz))
    stride_samples = int(round(args.stride_seconds * dataset.sampling_rate_hz))
    inventory = build_pure_window_inventory(dataset, window_samples, stride_samples)

    counts = (
        inventory.groupby(["subject_id", "label"]).size().unstack(fill_value=0)
        .reindex(index=SUBJECT_ORDER, columns=[0, 1], fill_value=0)
        .rename(columns=LABEL_NAMES)
    )
    counts.index.name = "subject_id"
    counts.to_csv(args.output_dir / "window_inventory.csv", encoding="utf-8-sig")

    nonfog_available = inventory[inventory["label"] == 0]
    nonfog_per_subject = min(
        args.max_nonfog_per_subject,
        int(nonfog_available.groupby("subject_id").size().min()),
    )
    nonfog_sample = balanced_sample(
        nonfog_available,
        ["subject_id"],
        nonfog_per_subject,
        args.seed,
    )

    positive_subjects = tuple(
        subject for subject in SUBJECT_ORDER if int(counts.loc[subject, "FOG"]) > 0
    )
    balanced_available = inventory[inventory["subject_id"].isin(positive_subjects)]
    per_subject_class = min(
        args.max_per_subject_class,
        int(balanced_available.groupby(["subject_id", "label"]).size().min()),
    )
    label_sample = balanced_sample(
        balanced_available,
        ["subject_id", "label"],
        per_subject_class,
        args.seed + 1,
    )

    all_features, feature_names = extract_features(
        dataset, inventory, args.feature_batch_size
    )

    def sample_features(sample: pd.DataFrame) -> np.ndarray:
        return all_features[sample["window_id"].to_numpy(dtype=np.int64)]

    nonfog_features = sample_features(nonfog_sample)
    label_features = sample_features(label_sample)
    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=True)
    nonfog_scaled = np.clip(scaler.fit_transform(nonfog_features), -12.0, 12.0)
    label_scaled = np.clip(scaler.transform(label_features), -12.0, 12.0)

    component_count = min(50, nonfog_scaled.shape[0] - 1, nonfog_scaled.shape[1])
    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=args.seed)
    nonfog_scores = pca.fit_transform(nonfog_scaled)
    label_scores = pca.transform(label_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    retained_components = int(np.searchsorted(cumulative, 0.80) + 1)
    retained_components = min(retained_components, component_count)
    retained_scores = nonfog_scores[:, :retained_components]
    retained_label_scores = label_scores[:, :retained_components]

    nonfog_subjects = nonfog_sample["subject_id"].to_numpy()
    label_subjects = label_sample["subject_id"].to_numpy()
    labels = label_sample["label"].to_numpy(dtype=np.int8)
    silhouette = float(
        silhouette_score(
            retained_scores,
            nonfog_subjects,
            metric="euclidean",
            sample_size=min(3000, len(retained_scores)),
            random_state=args.seed,
        )
    )
    subject_eta = multivariate_eta_squared(retained_scores, nonfog_subjects)
    effects = balanced_two_way_decomposition(
        retained_label_scores, label_subjects, labels
    )
    all_scaled = np.clip(scaler.transform(all_features), -12.0, 12.0)
    all_retained_scores = pca.transform(all_scaled)[:, :retained_components]
    sensitivity_rows: list[dict[str, float | int]] = []
    for repeat in range(args.sensitivity_repeats):
        repeat_seed = args.seed + 1000 + repeat
        repeat_nonfog = balanced_sample(
            nonfog_available,
            ["subject_id"],
            nonfog_per_subject,
            repeat_seed,
        )
        repeat_labels = balanced_sample(
            balanced_available,
            ["subject_id", "label"],
            per_subject_class,
            repeat_seed + 1,
        )
        repeat_nonfog_scores = all_retained_scores[
            repeat_nonfog["window_id"].to_numpy(dtype=np.int64)
        ]
        repeat_label_scores = all_retained_scores[
            repeat_labels["window_id"].to_numpy(dtype=np.int64)
        ]
        repeat_effects = balanced_two_way_decomposition(
            repeat_label_scores,
            repeat_labels["subject_id"].to_numpy(),
            repeat_labels["label"].to_numpy(dtype=np.int8),
        )
        repeat_subject_effect = float(
            repeat_effects["subject_main_effect_fraction"]
        )
        repeat_label_effect = float(repeat_effects["label_main_effect_fraction"])
        sensitivity_rows.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "subject_eta_squared": multivariate_eta_squared(
                    repeat_nonfog_scores,
                    repeat_nonfog["subject_id"].to_numpy(),
                ),
                "subject_main_effect_fraction": repeat_subject_effect,
                "label_main_effect_fraction": repeat_label_effect,
                "subject_to_label_ratio": (
                    repeat_subject_effect / repeat_label_effect
                    if repeat_label_effect > 0
                    else float("inf")
                ),
            }
        )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    sensitivity_frame.to_csv(
        args.output_dir / "resampling_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def quantile(column: str, value: float) -> float:
        return float(sensitivity_frame[column].quantile(value))

    sensitivity_summary = {
        "repeat_count": args.sensitivity_repeats,
        "subject_eta_squared_median": quantile("subject_eta_squared", 0.50),
        "subject_eta_squared_p05": quantile("subject_eta_squared", 0.05),
        "subject_eta_squared_p95": quantile("subject_eta_squared", 0.95),
        "subject_main_effect_median": quantile(
            "subject_main_effect_fraction", 0.50
        ),
        "label_main_effect_median": quantile("label_main_effect_fraction", 0.50),
        "subject_to_label_ratio_median": quantile("subject_to_label_ratio", 0.50),
        "subject_to_label_ratio_p05": quantile("subject_to_label_ratio", 0.05),
        "subject_to_label_ratio_p95": quantile("subject_to_label_ratio", 0.95),
        "subject_effect_greater_count": int(
            (
                sensitivity_frame["subject_main_effect_fraction"]
                > sensitivity_frame["label_main_effect_fraction"]
            ).sum()
        ),
    }
    distances, distance_summary = subject_centroid_distances(
        retained_scores, nonfog_subjects, SUBJECT_ORDER
    )
    distances.to_csv(
        args.output_dir / "nonfog_subject_centroid_distances.csv",
        encoding="utf-8-sig",
    )

    score_columns = [f"PC{index + 1}" for index in range(component_count)]
    nonfog_score_frame = pd.concat(
        [
            nonfog_sample.reset_index(drop=True),
            pd.DataFrame(nonfog_scores, columns=score_columns),
        ],
        axis=1,
    )
    label_score_frame = pd.concat(
        [
            label_sample.reset_index(drop=True),
            pd.DataFrame(label_scores, columns=score_columns),
        ],
        axis=1,
    )
    nonfog_score_frame.to_csv(
        args.output_dir / "pca_nonfog_scores.csv", index=False, encoding="utf-8-sig"
    )
    label_score_frame.to_csv(
        args.output_dir / "pca_balanced_label_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    first_two_loadings = pd.DataFrame(
        {
            "feature": feature_names,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1],
        }
    )
    first_two_loadings["absolute_loading"] = first_two_loadings[
        ["pc1_loading", "pc2_loading"]
    ].abs().max(axis=1)
    first_two_loadings = first_two_loadings.sort_values(
        "absolute_loading", ascending=False
    ).reset_index(drop=True)
    first_two_loadings.to_csv(
        args.output_dir / "pca_pc1_pc2_loadings.csv", index=False, encoding="utf-8-sig"
    )

    metrics: dict[str, object] = {
        "dataset": {
            "processed_dir": str(args.processed_dir),
            "sampling_rate_hz": dataset.sampling_rate_hz,
            "subject_count": len(dataset.subjects),
            "positive_subject_count": len(positive_subjects),
        },
        "window": {
            "window_seconds": args.window_seconds,
            "stride_seconds": args.stride_seconds,
            "window_samples": window_samples,
            "stride_samples": stride_samples,
            "pure_window_count": int(len(inventory)),
            "mixed_or_invalid_windows_excluded": True,
        },
        "sampling": {
            "seed": args.seed,
            "nonfog_per_subject": nonfog_per_subject,
            "nonfog_pca_sample_count": int(len(nonfog_sample)),
            "per_subject_class": per_subject_class,
            "balanced_label_sample_count": int(len(label_sample)),
            "positive_subjects": list(positive_subjects),
        },
        "features": {
            "feature_count": len(feature_names),
            "physical_channel_count": dataset.n_channels,
            "derived_magnitude_channel_count": 3,
            "scaler": "RobustScaler IQR 25-75, unit_variance, clipped to [-12,12]",
        },
        "pca": {
            "component_count": component_count,
            "pc1_explained_variance": float(explained[0]),
            "pc2_explained_variance": float(explained[1]),
            "pc1_pc2_explained_variance": float(explained[:2].sum()),
            "retained_components": retained_components,
            "retained_explained_variance": float(cumulative[retained_components - 1]),
        },
        "nonfog_domain": {
            "silhouette_subject": silhouette,
            "subject_eta_squared": subject_eta,
            **distance_summary,
        },
        "balanced_subject_label_effects": effects,
        "resampling_sensitivity": sensitivity_summary,
    }
    (args.output_dir / "analysis_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    methodology = {
        "analysis_type": "descriptive whole-dataset patient-domain PCA",
        "primary_fit_population": "equal pure non-FOG windows from all 10 subjects",
        "label_comparison_population": "equal pure FOG and non-FOG windows from 8 positive subjects",
        "feature_names": list(feature_names),
        "limitations": [
            "Overlapping windows are correlated and are not used for inferential p-values.",
            "Whole-dataset PCA is descriptive and must be refit inside training folds for prediction.",
            "Domain shifts can combine physiology, sensor placement, task mix, and disease severity.",
        ],
        "resampling_sensitivity_repeats": args.sensitivity_repeats,
    }
    (args.output_dir / "methodology.json").write_text(
        json.dumps(methodology, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    palette_values = sns.color_palette("tab10", n_colors=len(SUBJECT_ORDER))
    palette = dict(zip(SUBJECT_ORDER, palette_values))
    inventory_plot = args.output_dir / "01_window_inventory.png"
    nonfog_plot = args.output_dir / "02_pca_nonfog_subject_domains.png"
    label_plot = args.output_dir / "03_pca_balanced_labels.png"
    distance_plot = args.output_dir / "04_nonfog_domain_distance_heatmap.png"
    loadings_plot = args.output_dir / "05_pca_variance_and_loadings.png"
    save_inventory_plot(counts, inventory_plot, palette)
    scatter_subject_domains(
        nonfog_scores, nonfog_subjects, explained, nonfog_plot, palette
    )
    scatter_balanced_labels(
        label_scores, label_subjects, labels, explained, label_plot, palette
    )
    save_distance_heatmap(distances, distance_plot)
    save_variance_and_loadings_plot(explained, first_two_loadings, loadings_plot)
    save_overview(
        inventory_plot,
        nonfog_plot,
        label_plot,
        distance_plot,
        args.output_dir / "pca_domain_analysis_overview.png",
    )
    save_report(
        args.output_dir / "report.md", args, counts, metrics, first_two_loadings
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
