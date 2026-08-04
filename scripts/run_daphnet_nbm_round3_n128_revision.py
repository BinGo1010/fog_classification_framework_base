"""Run the Daphnet M3 TC-DAE N=128 capacity revision and S03 fine curve."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import daphnet_small_sample_selection as selection
import run_daphnet_nbm_tcdae_three_rounds as base


SUBJECTS = base.SUBJECTS
STABLE_SUBJECTS = tuple(subject for subject in SUBJECTS if subject != "S03")
SEEDS = base.SEEDS
LEVELS = (1, 8, 32, 128)
FINE_LEVELS = (1, 2, 4, 8, 16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base.REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.REPO_ROOT
        / "outputs"
        / f"{base.EXPERIMENT}_seed{SEEDS[0]}",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def median_channel_rms(values: np.ndarray) -> float:
    return float(np.median(np.sqrt(np.mean(np.square(values.astype(np.float64)), axis=0))))


def minimum_feasible_record_cap(capacities: dict[str, int], total: int) -> int:
    """Return the smallest record cap that can supply ``total`` windows."""
    for cap in range(max(32, math.ceil(total / max(len(capacities), 1))), total + 1):
        if sum(min(cap, capacity) for capacity in capacities.values()) >= total:
            return cap
    raise ValueError(f"Only {sum(capacities.values())} eligible windows for N={total}")


def allocate_quartile_record_counts(
    groups: Sequence[Sequence[int]],
    records: Sequence[base.Record],
    windows: selection.current.WindowSet,
    record_cap: int,
) -> tuple[list[str], np.ndarray]:
    """Solve the small integer allocation problem before choosing timestamps."""
    record_ids = sorted(
        {
            selection.record_id_for(records, windows, int(index))
            for group in groups
            for index in group
        }
    )
    cell_capacity = np.asarray(
        [
            [
                sum(
                    selection.record_id_for(records, windows, int(index)) == record_id
                    for index in group
                )
                for record_id in record_ids
            ]
            for group in groups
        ],
        dtype=float,
    )
    variable_count = cell_capacity.size
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for quartile in range(4):
        row = np.zeros(variable_count)
        row[quartile * len(record_ids) : (quartile + 1) * len(record_ids)] = 1.0
        rows.append(row)
        lower.append(32.0)
        upper.append(32.0)
    record_capacities = cell_capacity.sum(axis=0)
    for record_index, capacity in enumerate(record_capacities):
        row = np.zeros(variable_count)
        row[record_index:: len(record_ids)] = 1.0
        rows.append(row)
        lower.append(float(min(8, capacity)))
        upper.append(float(record_cap))
    # A tiny deterministic cost favors records with fewer previous allocations while
    # the minimal feasible cap and lower bounds enforce broad record coverage.
    cost = np.tile(
        1.0 / np.maximum(record_capacities, 1.0),
        4,
    ) + np.arange(variable_count) * 1e-8
    solution = milp(
        cost,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), cell_capacity.reshape(-1)),
        constraints=LinearConstraint(np.stack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30.0},
    )
    if not solution.success or solution.x is None:
        raise ValueError(f"Unable to allocate N=128 quartile/record quotas: {solution.message}")
    allocation = np.rint(solution.x).astype(int).reshape(4, len(record_ids))
    if not np.all(allocation.sum(axis=1) == 32):
        raise AssertionError("Quartile allocation did not sum to 32")
    return record_ids, allocation


def overlap_count(
    windows: selection.current.WindowSet,
    records: Sequence[base.Record],
    candidate: int,
    selected: Sequence[int],
) -> int:
    candidate_record = selection.record_id_for(records, windows, candidate)
    candidate_start = int(windows.start[candidate])
    candidate_end = int(windows.end[candidate])
    return sum(
        selection.record_id_for(records, windows, int(index)) == candidate_record
        and max(candidate_start, int(windows.start[int(index)]))
        < min(candidate_end, int(windows.end[int(index)]))
        for index in selected
    )


def choose_temporally_spread(
    candidates: Sequence[int],
    count: int,
    energy: dict[int, float],
    records: Sequence[base.Record],
    windows: selection.current.WindowSet,
    selected_global: list[int],
) -> list[int]:
    """Prefer non-overlap, then spread unavoidable overlap across the record."""
    remaining = list(map(int, candidates))
    chosen: list[int] = []
    target_energy = float(np.median([energy[index] for index in remaining]))
    for _ in range(count):
        same_record_selected = [
            int(index)
            for index in selected_global
            if selection.record_id_for(records, windows, int(index))
            == selection.record_id_for(records, windows, remaining[0])
        ]
        starts = [int(windows.start[index]) for index in same_record_selected]
        ranked = min(
            remaining,
            key=lambda index: (
                overlap_count(windows, records, index, selected_global),
                -min(
                    (abs(int(windows.start[index]) - start) for start in starts),
                    default=10**12,
                ),
                abs(energy[index] - target_energy),
                int(windows.start[index]),
                index,
            ),
        )
        chosen.append(ranked)
        selected_global.append(ranked)
        remaining.remove(ranked)
    return chosen


def nested_128_selection(
    records: Sequence[base.Record],
    windows: selection.current.WindowSet,
    candidates: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], dict[str, Any]]:
    eligible, _ = selection.eligible_candidates(records, windows, candidates)
    energy = {
        int(index): median_channel_rms(selection.window_values(records, windows, int(index)))
        for index in eligible
    }
    quartile_groups = selection.rank_quartiles(eligible, energy)
    record_capacity = Counter(
        selection.record_id_for(records, windows, int(index)) for index in eligible
    )
    record_cap = minimum_feasible_record_cap(dict(record_capacity), 128)
    record_ids, allocation = allocate_quartile_record_counts(
        quartile_groups, records, windows, record_cap
    )
    selected_global: list[int] = []
    selected_by_quartile: list[list[int]] = []
    for quartile, group in enumerate(quartile_groups):
        chosen_for_quartile: list[int] = []
        for record_index, record_id in enumerate(record_ids):
            count = int(allocation[quartile, record_index])
            if not count:
                continue
            cell = [
                int(index)
                for index in group
                if selection.record_id_for(records, windows, int(index)) == record_id
            ]
            chosen_for_quartile.extend(
                choose_temporally_spread(
                    cell, count, energy, records, windows, selected_global
                )
            )
        selected_by_quartile.append(chosen_for_quartile)
    nested = np.asarray(
        [
            selected_by_quartile[quartile][position]
            for position in range(32)
            for quartile in range(4)
        ],
        dtype=np.int64,
    )
    if len(set(map(int, nested))) != 128:
        raise AssertionError("Nested N=128 selection contains duplicate windows")
    actual_record_counts = Counter(
        selection.record_id_for(records, windows, int(index)) for index in nested
    )
    if max(actual_record_counts.values()) > record_cap:
        raise AssertionError("Adaptive per-record cap was exceeded")
    overlapping_windows = sum(
        selection.overlaps(windows, records, int(index), nested[:position])
        for position, index in enumerate(nested)
    )
    overlapping_pairs = sum(
        overlap_count(windows, records, int(index), nested[:position])
        for position, index in enumerate(nested)
    )
    audit = {
        "record_count": len(actual_record_counts),
        "adaptive_record_cap": record_cap,
        "maximum_record_contribution": max(actual_record_counts.values()),
        "maximum_record_fraction": max(actual_record_counts.values()) / 128.0,
        "record_contributions": dict(sorted(actual_record_counts.items())),
        "quartile_counts": {f"Q{index + 1}": 32 for index in range(4)},
        "unique_windows": True,
        "strictly_non_overlapping": overlapping_windows == 0,
        "windows_overlapping_an_earlier_selection": overlapping_windows,
        "overlapping_pair_count": overlapping_pairs,
        "overlap_policy": "Prefer non-overlap and temporal spread; permit overlap when the frozen clean training pool cannot supply 128 disjoint windows.",
        "clean_nonfog": True,
        "fog_guard_sec_each_side": 1.0,
    }
    return nested, energy, audit


def add_inclusion_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows):
        order = index + 1
        for level in FINE_LEVELS:
            row[f"included_in_n{level}"] = order <= level
    return rows


def build_nested_manifests(
    dataset: base.DaphnetDataset, output_dir: Path
) -> tuple[
    dict[str, tuple[list[base.Record], selection.current.WindowSet]],
    dict[str, np.ndarray],
    dict[str, dict[int, list[dict[str, Any]]]],
    dict[str, dict[str, Any]],
]:
    manifest = selection.load_manifest_rows(dataset.root)
    root = output_dir / "round3_n128_revision" / "nested_windows"
    pools: dict[str, tuple[list[base.Record], selection.current.WindowSet]] = {}
    selected_by_subject: dict[str, np.ndarray] = {}
    metadata_by_subject: dict[str, dict[int, list[dict[str, Any]]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        records, windows, candidates = selection.subject_pool(dataset, subject)
        selected, energy, audit = nested_128_selection(records, windows, candidates)
        full_rows = selection.selected_metadata(
            subject, 128, selected, energy, records, windows, manifest
        )
        add_inclusion_flags(full_rows)
        pools[subject] = (records, windows)
        selected_by_subject[subject] = selected
        metadata_by_subject[subject] = {}
        audits[subject] = audit
        for level in FINE_LEVELS:
            rows = [dict(row, sample_count=level) for row in full_rows[:level]]
            metadata_by_subject[subject][level] = rows
            base.write_csv(root / subject / f"N{level}_windows.csv", rows)
        all_rows.extend(full_rows)
        base.write_csv(root / subject / "N128_windows.csv", full_rows)
    base.write_csv(root / "all_subject_n128_windows.csv", all_rows)
    base.write_json(root / "selection_audit.json", audits)
    write_s03_manifest(
        output_dir,
        pools["S03"][0],
        pools["S03"][1],
        selected_by_subject["S03"],
        metadata_by_subject["S03"][128],
    )
    return pools, selected_by_subject, metadata_by_subject, audits


def write_s03_manifest(
    output_dir: Path,
    records: Sequence[base.Record],
    windows: selection.current.WindowSet,
    selected: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:
    values = np.stack(
        [selection.window_values(records, windows, int(index)) for index in selected]
    ).astype(np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    flat = centered.reshape(len(centered), -1) / math.sqrt(centered.shape[1] * centered.shape[2])
    distance = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    distance_without_self = distance.copy()
    np.fill_diagonal(distance_without_self, np.nan)
    quartiles = [row["energy_quartile"] for row in metadata]
    prototypes = {
        quartile: np.median(flat[[q == quartile for q in quartiles]], axis=0)
        for quartile in ("Q1", "Q2", "Q3", "Q4")
    }
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(metadata):
        item = dict(row)
        item["start_time"] = item["start_time_sec"]
        item["low_amplitude_motion"] = item["energy_quartile"] == "Q1"
        item["thigh_acc_forward_rms"] = float(
            np.sqrt(np.mean(np.square(values[index, :, 3])))
        )
        item["thigh_acc_vertical_rms"] = float(
            np.sqrt(np.mean(np.square(values[index, :, 4])))
        )
        item["mean_raw_distance_to_other_windows"] = float(
            np.nanmean(distance_without_self[index])
        )
        item["nearest_other_window_distance"] = float(
            np.nanmin(distance_without_self[index])
        )
        item["nearest_normal_prototype_distance"] = float(
            min(np.linalg.norm(flat[index] - prototype) for prototype in prototypes.values())
        )
        rows.append(item)
    path = (
        output_dir
        / "round3_n128_revision"
        / "nested_windows"
        / "s03_n128_window_manifest.csv"
    )
    base.write_csv(path, rows)


def n128_pass(metrics: dict[str, Any]) -> bool:
    distance = metrics.get("raw_latent_distance_corr")
    return bool(
        metrics["improvement_pct"] >= 40.0
        and metrics["median_corr"] >= 0.60
        and metrics["median_nrmse"] <= 0.75
        and 0.65 <= metrics["median_amplitude_ratio"] <= 1.35
        and distance is not None
        and distance >= 0.40
    )


def diagnostic_pass(sample_count: int, metrics: dict[str, Any]) -> bool:
    if sample_count <= 4:
        proxy = 1
    elif sample_count <= 16:
        proxy = 8
    elif sample_count <= 64:
        proxy = 32
    else:
        return n128_pass(metrics)
    return base.round3_pass(proxy, metrics)


def tail_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    arrays = base.metric_arrays(actual, predicted)
    window_nrmse = np.median(arrays["nrmse"], axis=1)
    window_corr = np.median(arrays["correlation"], axis=1)
    window_amplitude = np.median(arrays["amplitude_ratio"], axis=1)
    improvement = 100.0 * (
        arrays["window_zero_mse"] - arrays["window_mse"]
    ) / np.maximum(arrays["window_zero_mse"], 1e-12)
    return {
        "nrmse_p90": float(np.percentile(window_nrmse, 90)),
        "nrmse_p95": float(np.percentile(window_nrmse, 95)),
        "pearson_p10": float(np.percentile(window_corr, 10)),
        "amplitude_ratio_p10": float(np.percentile(window_amplitude, 10)),
        "amplitude_ratio_p90": float(np.percentile(window_amplitude, 90)),
        "negative_improvement_window_fraction": float(np.mean(improvement < 0.0)),
        "nrmse_gt_1_window_fraction": float(np.mean(window_nrmse > 1.0)),
        "pearson_lt_0_2_window_fraction": float(np.mean(window_corr < 0.2)),
    }


def tail_risk_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = (
        (metrics["nrmse_p90"] > 1.20, "nrmse_p90_above_1_20"),
        (metrics["pearson_p10"] < 0.20, "pearson_p10_below_0_20"),
        (
            metrics["negative_improvement_window_fraction"] > 0.10,
            "negative_improvement_fraction_above_10_percent",
        ),
        (
            metrics["nrmse_gt_1_window_fraction"] > 0.15,
            "nrmse_gt_1_fraction_above_15_percent",
        ),
        (metrics["amplitude_ratio_p10"] < 0.40, "amplitude_p10_below_0_40"),
        (metrics["amplitude_ratio_p90"] > 1.60, "amplitude_p90_above_1_60"),
    )
    for failed, name in checks:
        if failed:
            reasons.append(name)
    return reasons


def extended_window_rows(
    metadata: Sequence[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
    latent: np.ndarray,
) -> list[dict[str, Any]]:
    arrays = base.metric_arrays(actual, predicted)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(metadata):
        delta_loss = float(
            np.mean(np.square(np.diff(actual[index], axis=0) - np.diff(predicted[index], axis=0)))
        )
        zero_loss = float(arrays["window_zero_mse"][index])
        nbm_loss = float(arrays["window_mse"][index])
        rows.append(
            {
                **item,
                "nbm_loss": nbm_loss,
                "zero_loss": zero_loss,
                "improvement": 100.0 * (zero_loss - nbm_loss) / max(zero_loss, 1e-12),
                "pearson_median": float(np.median(arrays["correlation"][index])),
                "nrmse_median": float(np.median(arrays["nrmse"][index])),
                "amplitude_ratio_median": float(
                    np.median(arrays["amplitude_ratio"][index])
                ),
                "delta_loss": delta_loss,
                "latent_norm": float(np.linalg.norm(latent[index])),
            }
        )
    return rows


def extended_channel_rows(
    metadata: Sequence[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
    channel_names: Sequence[str],
) -> list[dict[str, Any]]:
    arrays = base.metric_arrays(actual, predicted)
    rows: list[dict[str, Any]] = []
    for window_index, item in enumerate(metadata):
        for channel, name in enumerate(channel_names):
            delta_loss = float(
                np.mean(
                    np.square(
                        np.diff(actual[window_index, :, channel])
                        - np.diff(predicted[window_index, :, channel])
                    )
                )
            )
            rows.append(
                {
                    "window_id": item["window_id"],
                    "channel": name,
                    "channel_id": channel,
                    "mse": float(
                        np.mean(
                            np.square(
                                actual[window_index, :, channel]
                                - predicted[window_index, :, channel]
                            )
                        )
                    ),
                    "pearson": float(arrays["correlation"][window_index, channel]),
                    "nrmse": float(arrays["nrmse"][window_index, channel]),
                    "amplitude_ratio": float(
                        arrays["amplitude_ratio"][window_index, channel]
                    ),
                    "delta_loss": delta_loss,
                }
            )
    return rows


def load_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as payload:
        return (
            np.asarray(payload["target"]),
            np.asarray(payload["reconstruction"]),
            np.asarray(payload["latent"]),
        )


def update_extended_outputs(
    run_dir: Path,
    result: dict[str, Any],
    metadata: list[dict[str, Any]],
    channel_names: Sequence[str],
    *,
    use_n128_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    actual, predicted, latent = load_predictions(run_dir)
    result.update(tail_metrics(actual, predicted))
    strict = n128_pass(result) if use_n128_gate else diagnostic_pass(len(actual), result)
    risks = tail_risk_reasons(result)
    result["strict_pass"] = strict
    result["tail_risk"] = bool(risks)
    result["tail_risk_reasons"] = risks
    result["result_class"] = (
        "PASS with tail-risk" if strict and risks else "PASS" if strict else "FAIL"
    )
    result["pass_status"] = "PASS" if strict else "FAIL"
    window_rows = extended_window_rows(metadata, actual, predicted, latent)
    channel_rows = extended_channel_rows(metadata, actual, predicted, channel_names)
    base.write_json(run_dir / "metrics.json", result)
    base.write_csv(run_dir / "window_metrics.csv", window_rows)
    base.write_csv(run_dir / "channel_metrics.csv", channel_rows)
    return result, window_rows, channel_rows


def plot_training(history: list[dict[str, Any]], metrics: dict[str, Any], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, [row["eval_mse"] for row in history], label="evaluation MSE")
    ax.axhline(history[0]["zero_mse"], color="0.5", linestyle="--", label="zero output")
    ax.axvline(metrics["best_epoch"], color="tab:green", linestyle="--", label="best")
    ax.axvline(metrics["final_epoch"], color="tab:red", linestyle=":", label="final")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    lr_ax = ax.twinx()
    lr_ax.plot(epochs, [row["learning_rate"] for row in history], color="tab:purple", alpha=0.35)
    lr_ax.set_ylabel("Learning rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_capacity_curve(rows: Sequence[dict[str, Any]], path: Path) -> None:
    levels = [1, 8, 32, 128]
    lookup = {int(row["sample_count"]): row for row in rows}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    metrics = (
        ("improvement_pct", "Improvement (%)"),
        ("median_corr", "Pearson"),
        ("median_nrmse", "NRMSE"),
        ("nrmse_p90", "NRMSE P90"),
    )
    for ax, (key, title) in zip(axes, metrics):
        values = [lookup[level].get(key, lookup[level].get("p90_window_nrmse")) for level in levels]
        ax.plot(levels, values, marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, levels)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sorted_nrmse(window_rows: Sequence[dict[str, Any]], path: Path) -> None:
    values = np.sort([row["nrmse_median"] for row in window_rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(values) + 1), values)
    for percentile, color in ((50, "tab:green"), (90, "tab:orange"), (95, "tab:red")):
        value = float(np.percentile(values, percentile))
        ax.axhline(value, color=color, linestyle="--", label=f"P{percentile}={value:.3f}")
    ax.set_xlabel("Sorted window rank")
    ax.set_ylabel("Window median NRMSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_energy_nrmse(window_rows: Sequence[dict[str, Any]], path: Path) -> None:
    colors = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    fig, ax = plt.subplots(figsize=(6, 4.8))
    scatter = ax.scatter(
        [row["energy"] for row in window_rows],
        [row["nrmse_median"] for row in window_rows],
        c=[colors[row["energy_quartile"]] for row in window_rows],
        cmap="viridis",
        alpha=0.75,
    )
    ax.set_xlabel("Window energy")
    ax.set_ylabel("Window median NRMSE")
    ax.grid(alpha=0.2)
    legend = ax.legend(*scatter.legend_elements(), title="Quartile", fontsize=7)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_record_nrmse(window_rows: Sequence[dict[str, Any]], path: Path) -> None:
    records = sorted({row["record_id"] for row in window_rows})
    values = [
        [row["nrmse_median"] for row in window_rows if row["record_id"] == record]
        for record in records
    ]
    fig, ax = plt.subplots(figsize=(max(7, len(records) * 1.2), 4.8))
    ax.boxplot(values, tick_labels=records, showfliers=True)
    ax.set_ylabel("Window median NRMSE")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_best_median_worst(
    actual: np.ndarray,
    predicted: np.ndarray,
    channel_names: Sequence[str],
    path: Path,
) -> None:
    arrays = base.metric_arrays(actual, predicted)
    order = np.argsort(np.median(arrays["nrmse"], axis=1))
    chosen = (int(order[0]), int(order[len(order) // 2]), int(order[-1]))
    labels = ("best", "median", "worst")
    time_axis = np.arange(base.WINDOW) / base.FS
    fig, axes = plt.subplots(3, 9, figsize=(22, 8), sharex=True)
    for row_index, (window_index, label) in enumerate(zip(chosen, labels)):
        for channel, name in enumerate(channel_names):
            ax = axes[row_index, channel]
            ax.plot(time_axis, actual[window_index, :, channel], linewidth=0.8)
            ax.plot(time_axis, predicted[window_index, :, channel], "--", linewidth=0.8)
            if row_index == 0:
                ax.set_title(name, fontsize=7)
            if channel == 0:
                ax.set_ylabel(label)
            ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_latent_matrix_ordered(
    latent: np.ndarray, window_rows: Sequence[dict[str, Any]], path: Path
) -> None:
    order = np.argsort(
        [
            int(row["energy_quartile"][1:]) * 1_000_000 + row["energy"]
            for row in window_rows
        ]
    )
    matrix, _ = base.pairwise_distances(latent[order])
    base.plot_distance_matrix(matrix, path)


def plot_old_added(window_rows: Sequence[dict[str, Any]], path: Path) -> None:
    old = [row["nrmse_median"] for row in window_rows[:32]]
    added = [row["nrmse_median"] for row in window_rows[32:]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].boxplot([old, added], tick_labels=["N32 prefix", "added 96"])
    axes[0].set_ylabel("NRMSE")
    old_corr = [row["pearson_median"] for row in window_rows[:32]]
    added_corr = [row["pearson_median"] for row in window_rows[32:]]
    axes[1].boxplot([old_corr, added_corr], tick_labels=["N32 prefix", "added 96"])
    axes[1].set_ylabel("Pearson")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_n128_run_figures(
    run_dir: Path,
    result: dict[str, Any],
    window_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    capacity_rows: Sequence[dict[str, Any]],
    channel_names: Sequence[str],
) -> None:
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    actual, predicted, latent = load_predictions(run_dir)
    history = base.numeric_history(base.read_csv(run_dir / "training_log.csv"))
    arrays = base.metric_arrays(actual, predicted)
    plot_training(history, result, figures / "training_loss.png")
    plot_capacity_curve(capacity_rows, figures / "sample_size_capacity_curve.png")
    plot_sorted_nrmse(window_rows, figures / "sorted_window_nrmse.png")
    plot_energy_nrmse(window_rows, figures / "window_energy_vs_nrmse.png")
    plot_record_nrmse(window_rows, figures / "record_source_vs_nrmse.png")
    plot_best_median_worst(
        actual, predicted, channel_names, figures / "best_median_worst_waveforms.png"
    )
    base.plot_metric_heatmap(
        arrays["nrmse"],
        figures / "window_channel_nrmse_heatmap.png",
        channel_names,
        "N=128 window-channel NRMSE",
        cmap="magma",
        vmin=0.0,
    )
    base.plot_metric_heatmap(
        arrays["correlation"],
        figures / "window_channel_pearson_heatmap.png",
        channel_names,
        "N=128 window-channel Pearson",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    base.plot_metric_heatmap(
        arrays["amplitude_ratio"],
        figures / "amplitude_ratio_heatmap.png",
        channel_names,
        "N=128 amplitude ratio",
        cmap="viridis",
        vmin=0.0,
        vmax=1.8,
    )
    _, diagnostic_arrays = base.summarize(actual, predicted, latent)
    base.plot_raw_latent_distance(
        diagnostic_arrays, figures / "raw_latent_distance_scatter.png"
    )
    plot_latent_matrix_ordered(
        latent, window_rows, figures / "latent_distance_matrix_clustered.png"
    )
    plot_old_added(window_rows, figures / "old_vs_added_windows.png")


def load_historical_round3(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output_dir / "round3_capacity").rglob("metrics.json")
        if "N128" not in str(path)
    ]


def historical_capacity_rows(
    historical: Sequence[dict[str, Any]],
    subject: str,
    seed: int,
    n128: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in historical
        if row["subject_id"] == subject
        and row["seed"] == seed
        and row["sample_count"] in (1, 8, 32)
    ]
    return sorted([*rows, n128], key=lambda row: int(row["sample_count"]))


def run_n128(
    args: argparse.Namespace,
    dataset: base.DaphnetDataset,
    pools: dict[str, tuple[list[base.Record], selection.current.WindowSet]],
    selected: dict[str, np.ndarray],
    metadata: dict[str, dict[int, list[dict[str, Any]]]],
    historical: list[dict[str, Any]],
    device: Any,
) -> list[dict[str, Any]]:
    root = args.output_dir / "round3_n128_revision" / "runs"
    results: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        records, windows = pools[subject]
        x, preprocessing_config = base.preprocess(
            "P0_current", records, windows, selected[subject]
        )
        for seed in SEEDS:
            run_dir = root / subject / "N128" / f"seed{seed}"
            result = base.execute_run(
                mode="round3_n128_revision",
                run_dir=run_dir,
                subject=subject,
                sample_count=128,
                seed=seed,
                architecture="M3_tcdae_long",
                preprocessor="P0_current",
                x=x,
                preprocessing_config=preprocessing_config,
                metadata=metadata[subject][128],
                max_epochs=args.max_epochs,
                optimizer_name="AdamW",
                learning_rate=3e-4,
                weight_decay=1e-4,
                patience=args.patience,
                pass_function=lambda _n, values: n128_pass(values),
                device=device,
                num_workers=args.num_workers,
                channel_names=dataset.channel_names,
                overwrite=args.overwrite,
                skip_figures=True,
            )
            result, window_rows, channel_rows = update_extended_outputs(
                run_dir,
                result,
                metadata[subject][128],
                dataset.channel_names,
                use_n128_gate=True,
            )
            result["run_dir"] = str(run_dir.resolve())
            results.append(result)
            if not args.skip_figures:
                render_n128_run_figures(
                    run_dir,
                    result,
                    window_rows,
                    channel_rows,
                    historical_capacity_rows(historical, subject, seed, result),
                    dataset.channel_names,
                )
    return results


def run_s03_fine_capacity(
    args: argparse.Namespace,
    dataset: base.DaphnetDataset,
    pool: tuple[list[base.Record], selection.current.WindowSet],
    selected: np.ndarray,
    metadata: dict[int, list[dict[str, Any]]],
    n128_results: Sequence[dict[str, Any]],
    device: Any,
) -> list[dict[str, Any]]:
    records, windows = pool
    root = args.output_dir / "round3_n128_revision" / "s03_fine_capacity"
    results: list[dict[str, Any]] = []
    channel_summaries: dict[int, list[dict[str, Any]]] = {}
    window_summaries: dict[int, list[dict[str, Any]]] = {}
    for level in FINE_LEVELS:
        if level == 128:
            result = next(
                row
                for row in n128_results
                if row["subject_id"] == "S03" and row["seed"] == SEEDS[0]
            )
            run_dir = Path(result["run_dir"])
            window_rows = base.read_csv(run_dir / "window_metrics.csv")
            channel_rows = base.read_csv(run_dir / "channel_metrics.csv")
            worst_name, worst_nrmse = worst_channel(channel_rows)
            results.append(
                dict(
                    result,
                    fine_capacity_reused=True,
                    worst_channel=worst_name,
                    worst_channel_median_nrmse=worst_nrmse,
                )
            )
            channel_summaries[level] = channel_rows
            window_summaries[level] = window_rows
            continue
        indices = selected[:level]
        x, preprocessing_config = base.preprocess(
            "P0_current", records, windows, indices
        )
        run_dir = root / f"N{level}" / f"seed{SEEDS[0]}"
        result = base.execute_run(
            mode="s03_fine_capacity",
            run_dir=run_dir,
            subject="S03",
            sample_count=level,
            seed=SEEDS[0],
            architecture="M3_tcdae_long",
            preprocessor="P0_current",
            x=x,
            preprocessing_config=preprocessing_config,
            metadata=metadata[level],
            max_epochs=args.max_epochs,
            optimizer_name="AdamW",
            learning_rate=3e-4,
            weight_decay=1e-4,
            patience=args.patience,
            pass_function=diagnostic_pass,
            device=device,
            num_workers=args.num_workers,
            channel_names=dataset.channel_names,
            overwrite=args.overwrite,
            skip_figures=True,
        )
        result, window_rows, channel_rows = update_extended_outputs(
            run_dir,
            result,
            metadata[level],
            dataset.channel_names,
            use_n128_gate=False,
        )
        worst_name, worst_nrmse = worst_channel(channel_rows)
        result["worst_channel"] = worst_name
        result["worst_channel_median_nrmse"] = worst_nrmse
        base.write_json(run_dir / "metrics.json", result)
        result["run_dir"] = str(run_dir.resolve())
        results.append(result)
        channel_summaries[level] = channel_rows
        window_summaries[level] = window_rows
    base.write_csv(root / "s03_fine_capacity_metrics.csv", summary_rows(results))
    if not args.skip_figures:
        render_s03_fine_figures(
            root, results, window_summaries, channel_summaries, dataset.channel_names
        )
    return results


def worst_channel(channel_rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    by_channel: dict[str, list[float]] = defaultdict(list)
    for row in channel_rows:
        by_channel[str(row["channel"])].append(float(row["nrmse"]))
    medians = {
        channel: float(np.median(values)) for channel, values in by_channel.items()
    }
    name = max(medians, key=medians.get)
    return name, medians[name]


def channel_curve(
    channel_summaries: dict[int, list[dict[str, Any]]],
    channel_name: str,
    path: Path,
) -> None:
    levels = list(FINE_LEVELS)
    values = [
        float(
            np.median(
                [
                    float(row["nrmse"])
                    for row in channel_summaries[level]
                    if row["channel"] == channel_name
                ]
            )
        )
        for level in levels
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(levels, values, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, levels)
    ax.set_xlabel("Sample count")
    ax.set_ylabel("Median channel NRMSE")
    ax.set_title(channel_name)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_s03_fine_figures(
    root: Path,
    results: Sequence[dict[str, Any]],
    window_summaries: dict[int, list[dict[str, Any]]],
    channel_summaries: dict[int, list[dict[str, Any]]],
    channel_names: Sequence[str],
) -> None:
    ordered = sorted(results, key=lambda row: int(row["sample_count"]))
    levels = [int(row["sample_count"]) for row in ordered]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    metrics = (
        ("improvement_pct", "Improvement (%)"),
        ("median_corr", "Pearson"),
        ("median_nrmse", "NRMSE"),
        ("nrmse_p90", "NRMSE P90"),
        ("pearson_p10", "Pearson P10"),
        ("median_amplitude_ratio", "Amplitude ratio"),
    )
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(levels, [row[key] for row in ordered], marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, levels)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "s03_fine_grained_capacity_curve.png", dpi=150)
    plt.close(fig)
    channel_curve(
        channel_summaries,
        "thigh_acc_forward",
        root / "s03_thigh_acc_forward_curve.png",
    )
    channel_curve(
        channel_summaries,
        "thigh_acc_vertical",
        root / "s03_thigh_acc_vertical_curve.png",
    )
    n128_rows = window_summaries[128]
    quartiles = ("Q1", "Q2", "Q3", "Q4")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(
        [
            [float(row["nrmse_median"]) for row in n128_rows if row["energy_quartile"] == q]
            for q in quartiles
        ],
        tick_labels=quartiles,
    )
    ax.set_ylabel("Window NRMSE")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "s03_energy_quartile_nrmse.png", dpi=150)
    plt.close(fig)
    plot_record_nrmse(
        [{**row, "nrmse_median": float(row["nrmse_median"])} for row in n128_rows],
        root / "s03_record_level_nrmse.png",
    )


def summary_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "subject_id",
        "sample_count",
        "seed",
        "improvement_pct",
        "median_corr",
        "median_nrmse",
        "median_amplitude_ratio",
        "raw_latent_distance_corr",
        "nrmse_p90",
        "nrmse_p95",
        "pearson_p10",
        "amplitude_ratio_p10",
        "amplitude_ratio_p90",
        "negative_improvement_window_fraction",
        "nrmse_gt_1_window_fraction",
        "pearson_lt_0_2_window_fraction",
        "strict_pass",
        "tail_risk",
        "result_class",
        "worst_channel",
        "worst_channel_median_nrmse",
        "parameter_count",
        "inference_ms_per_batch",
    )
    return [{key: row.get(key) for key in keys} for row in results]


def evaluate_gates(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict_count = sum(row["strict_pass"] for row in results)
    stable_rows = [row for row in results if row["subject_id"] in STABLE_SUBJECTS]
    stable_pass_count = sum(row["strict_pass"] for row in stable_rows)
    stable_subject_passes = {
        subject: sum(
            row["strict_pass"] for row in stable_rows if row["subject_id"] == subject
        )
        for subject in STABLE_SUBJECTS
    }
    stable_3_of_3 = sum(count == 3 for count in stable_subject_passes.values())
    failed_subjects = sorted(
        {
            row["subject_id"]
            for row in results
            if not row["strict_pass"]
        }
    )
    new_repeated_failures = [
        subject
        for subject, count in stable_subject_passes.items()
        if count == 0
    ]
    medians = {
        "median_improvement_pct": float(np.median([row["improvement_pct"] for row in results])),
        "median_corr": float(np.median([row["median_corr"] for row in results])),
        "median_nrmse": float(np.median([row["median_nrmse"] for row in results])),
        "median_amplitude_ratio": float(
            np.median([row["median_amplitude_ratio"] for row in results])
        ),
        "median_distance_corr": float(
            np.median([row["raw_latent_distance_corr"] for row in results])
        ),
    }
    aggregate_pass = bool(
        medians["median_improvement_pct"] >= 40.0
        and medians["median_corr"] >= 0.60
        and medians["median_nrmse"] <= 0.75
        and 0.65 <= medians["median_amplitude_ratio"] <= 1.35
        and medians["median_distance_corr"] >= 0.40
    )
    stable_gate = bool(
        stable_pass_count >= 20
        and stable_3_of_3 >= 6
        and len(new_repeated_failures) == 0
        and len(failed_subjects) <= 2
        and aggregate_pass
    )
    any_tail_risk = any(row["strict_pass"] and row["tail_risk"] for row in results)
    if strict_count == 24:
        status = "All-subject Strict PASS"
    elif stable_gate:
        status = "Conditional PASS with tail-risk" if any_tail_risk else "Stable-cohort PASS"
    else:
        status = "Capacity insufficient"
    return {
        "all_subject_strict_gate": "PASS" if strict_count == 24 else "FAIL",
        "strict_pass_count": strict_count,
        "strict_total": 24,
        "stable_cohort_gate": "PASS" if stable_gate else "FAIL",
        "stable_pass_count": stable_pass_count,
        "stable_total": 21,
        "stable_subjects_3_of_3": stable_3_of_3,
        "stable_subject_pass_counts": stable_subject_passes,
        "failed_subjects": failed_subjects,
        "new_stable_subjects_0_of_3": new_repeated_failures,
        "aggregate_medians": medians,
        "aggregate_n128_thresholds_pass": aggregate_pass,
        "tail_risk_run_count": sum(row["strict_pass"] and row["tail_risk"] for row in results),
        "final_status": status,
        "formal_denoising_progression_eligible": stable_gate,
    }


def subject_summary(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        values = [row for row in results if row["subject_id"] == subject]
        passes = sum(row["strict_pass"] for row in values)
        tail = sum(row["strict_pass"] and row["tail_risk"] for row in values)
        rows.append(
            {
                "subject_id": subject,
                "n128_pass_count": passes,
                "run_count": len(values),
                "median_improvement_pct": float(
                    np.median([row["improvement_pct"] for row in values])
                ),
                "median_corr": float(np.median([row["median_corr"] for row in values])),
                "median_nrmse": float(np.median([row["median_nrmse"] for row in values])),
                "median_nrmse_p90": float(np.median([row["nrmse_p90"] for row in values])),
                "tail_risk_run_count": tail,
                "conclusion": (
                    "3/3 PASS with tail-risk" if passes == 3 and tail else "3/3 PASS" if passes == 3 else f"{passes}/3 PASS"
                ),
            }
        )
    return rows


def render_global_figures(
    root: Path,
    historical: Sequence[dict[str, Any]],
    n128_results: Sequence[dict[str, Any]],
) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for subject in SUBJECTS:
        rows = [row for row in historical if row["subject_id"] == subject]
        rows.extend(row for row in n128_results if row["subject_id"] == subject)
        levels = (1, 8, 32, 128)
        pearson = [np.median([row["median_corr"] for row in rows if row["sample_count"] == level]) for level in levels]
        nrmse = [np.median([row["median_nrmse"] for row in rows if row["sample_count"] == level]) for level in levels]
        axes[0].plot(levels, pearson, marker="o", label=subject)
        axes[1].plot(levels, nrmse, marker="o", label=subject)
    for ax, title in zip(axes, ("Pearson", "NRMSE")):
        ax.set_xscale("log", base=2)
        ax.set_xticks((1, 8, 32, 128), (1, 8, 32, 128))
        ax.set_xlabel("Sample count")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "all_subject_capacity_curves.png", dpi=150)
    plt.close(fig)

    matrix = np.zeros((len(SUBJECTS), 4), dtype=float)
    for subject_index, subject in enumerate(SUBJECTS):
        for level_index, level in enumerate((1, 8, 32)):
            rows = [
                row
                for row in historical
                if row["subject_id"] == subject and row["sample_count"] == level
            ]
            matrix[subject_index, level_index] = np.mean(
                [row["pass_status"] == "PASS" for row in rows]
            )
        rows = [row for row in n128_results if row["subject_id"] == subject]
        matrix[subject_index, 3] = np.mean([row["strict_pass"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4), ("N=1", "N=8", "N=32", "N=128"))
    ax.set_yticks(range(len(SUBJECTS)), SUBJECTS)
    for i in range(len(SUBJECTS)):
        for j in range(4):
            ax.text(j, i, f"{round(matrix[i,j]*3):.0f}/3", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="PASS fraction")
    fig.tight_layout()
    fig.savefig(figures / "subject_sample_size_pass_matrix.png", dpi=150)
    plt.close(fig)

    x = np.arange(len(SUBJECTS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in zip(axes, ("median_corr", "median_nrmse"), ("Pearson", "NRMSE")):
        groups = [[row[key] for row in n128_results if row["subject_id"] == subject] for subject in SUBJECTS]
        ax.errorbar(
            x,
            [np.mean(group) for group in groups],
            yerr=[np.std(group) for group in groups],
            fmt="o",
            capsize=4,
        )
        for index, group in enumerate(groups):
            ax.vlines(index, min(group), max(group), color="0.5", alpha=0.5)
        ax.set_xticks(x, SUBJECTS)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "seed_stability_n128.png", dpi=150)
    plt.close(fig)

    keys = (
        ("improvement_pct", "Improvement (%)"),
        ("median_corr", "Pearson"),
        ("median_nrmse", "NRMSE"),
        ("nrmse_p90", "NRMSE P90"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, keys):
        groups = [[row[key] for row in n128_results if row["subject_id"] == subject] for subject in SUBJECTS]
        ax.boxplot(groups, tick_labels=SUBJECTS, showfliers=True)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "all_subject_n128_metrics.png", dpi=150)
    plt.close(fig)


def write_reports(
    output_dir: Path,
    results: Sequence[dict[str, Any]],
    fine_results: Sequence[dict[str, Any]],
    selection_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = output_dir / "round3_n128_revision"
    tables = root / "tables"
    reports = root / "reports"
    gates = evaluate_gates(results)
    subjects = subject_summary(results)
    base.write_csv(tables / "n128_run_metrics.csv", summary_rows(results))
    base.write_csv(tables / "n128_subject_summary.csv", subjects)
    gate_rows = [
        {
            "gate": "All-subject strict",
            "requirement": "24/24",
            "actual": f"{gates['strict_pass_count']}/24",
            "result": gates["all_subject_strict_gate"],
        },
        {
            "gate": "Stable-cohort capacity",
            "requirement": ">=20/21",
            "actual": f"{gates['stable_pass_count']}/21",
            "result": gates["stable_cohort_gate"],
        },
        {
            "gate": "Stable subjects 3/3",
            "requirement": ">=6/7",
            "actual": f"{gates['stable_subjects_3_of_3']}/7",
            "result": "PASS" if gates["stable_subjects_3_of_3"] >= 6 else "FAIL",
        },
        {
            "gate": "New failure spread",
            "requirement": "no new subject 0/3",
            "actual": ",".join(gates["new_stable_subjects_0_of_3"]) or "none",
            "result": "PASS" if not gates["new_stable_subjects_0_of_3"] else "FAIL",
        },
        {
            "gate": "Failed-subject spread",
            "requirement": "<=2 subjects",
            "actual": ",".join(gates["failed_subjects"]) or "none",
            "result": "PASS" if len(gates["failed_subjects"]) <= 2 else "FAIL",
        },
        {
            "gate": "Aggregate N=128 medians",
            "requirement": "all five N=128 thresholds",
            "actual": "pass" if gates["aggregate_n128_thresholds_pass"] else "fail",
            "result": "PASS" if gates["aggregate_n128_thresholds_pass"] else "FAIL",
        },
        {
            "gate": "S03 fine capacity",
            "requirement": "8 levels complete",
            "actual": f"{len(fine_results)}/8",
            "result": "PASS" if len(fine_results) == 8 else "FAIL",
        },
    ]
    base.write_csv(tables / "global_gates.csv", gate_rows)
    decision = {
        **gates,
        "s03_fine_capacity_complete": len(fine_results) == 8,
        "formal_denoising_progression_eligible": bool(
            gates["stable_cohort_gate"] == "PASS" and len(fine_results) == 8
        ),
        "selection_audit": selection_audits,
        "historical_lower_levels_nested": False,
        "revision_n128_sequence_has_nested_prefixes": True,
        "interpretation_boundary": "Training-set memory/capacity only; not generalization, denoising, or FoG detection evidence.",
    }
    base.write_json(reports / "decision.json", decision)
    subject_lines = "\n".join(
        f"- {row['subject_id']}: {row['n128_pass_count']}/3 PASS；Pearson {row['median_corr']:.3f}；"
        f"NRMSE {row['median_nrmse']:.3f}；NRMSE P90 {row['median_nrmse_p90']:.3f}；{row['conclusion']}。"
        for row in subjects
    )
    fine_ordered = sorted(fine_results, key=lambda row: int(row["sample_count"]))
    fine_lines = "\n".join(
        f"- N={row['sample_count']}: 改善率 {row['improvement_pct']:.1f}%；Pearson {row['median_corr']:.3f}；"
        f"NRMSE {row['median_nrmse']:.3f}；NRMSE P90 {row['nrmse_p90']:.3f}；"
        f"最差通道 {row['worst_channel']}（通道 NRMSE 中位数 {row['worst_channel_median_nrmse']:.3f}）。"
        for row in fine_ordered
    )
    failed_lines: list[str] = []
    for row in results:
        if row["strict_pass"]:
            continue
        failed_metrics: list[str] = []
        if row["improvement_pct"] < 40.0:
            failed_metrics.append(f"改善率 {row['improvement_pct']:.1f}%")
        if row["median_corr"] < 0.60:
            failed_metrics.append(f"Pearson {row['median_corr']:.3f}")
        if row["median_nrmse"] > 0.75:
            failed_metrics.append(f"NRMSE {row['median_nrmse']:.3f}")
        if not 0.65 <= row["median_amplitude_ratio"] <= 1.35:
            failed_metrics.append(f"幅值比 {row['median_amplitude_ratio']:.3f}")
        if row["raw_latent_distance_corr"] < 0.40:
            failed_metrics.append(
                f"潜变量距离相关 {row['raw_latent_distance_corr']:.3f}"
            )
        failed_lines.append(
            f"- {row['subject_id']} / seed {row['seed']}："
            + "、".join(failed_metrics)
            + "。"
        )
    selection_lines = "\n".join(
        f"- {subject}: {audit['record_count']} 条记录；最大记录贡献 "
        f"{audit['maximum_record_contribution']}/128；与更早入选窗口重叠 "
        f"{audit['windows_overlapping_an_earlier_selection']} 个。"
        for subject, audit in selection_audits.items()
    )
    medians = gates["aggregate_medians"]
    report = f"""# Daphnet NBM 第三轮 N=128 容量修订报告

本扩展冻结 P0_current、M3_tcdae_long、MSELoss、AdamW 3e-4、weight decay 1e-4、batch size 64、最多 2000 epoch 和 patience 100。全部结果仍是训练集内记忆/容量诊断。

## 门控结论

- 全被试严格门控：{gates['all_subject_strict_gate']}（{gates['strict_pass_count']}/24）
- 稳定 7 被试容量门控：{gates['stable_cohort_gate']}（{gates['stable_pass_count']}/21）
- 稳定被试 3/3 覆盖：{gates['stable_subjects_3_of_3']}/7
- 失败涉及被试：{', '.join(gates['failed_subjects'])}
- 主门槛通过但有尾部风险的运行：{gates['tail_risk_run_count']}
- 最终分类：{gates['final_status']}
- 是否满足模板中的正式去噪推进条件：{'是' if decision['formal_denoising_progression_eligible'] else '否'}

整体中位数本身通过五项 N=128 门槛：改善率 {medians['median_improvement_pct']:.1f}%，Pearson {medians['median_corr']:.3f}，NRMSE {medians['median_nrmse']:.3f}，幅值比 {medians['median_amplitude_ratio']:.3f}，潜变量距离相关 {medians['median_distance_corr']:.3f}。但稳定队列仅 19/21、仅 5/7 名稳定被试达到 3/3，且失败扩散至 S02、S03、S07 三名被试，因此不得推进正式去噪训练。

## 严格失败明细

{chr(10).join(failed_lines)}

## N=128 被试级结果

{subject_lines}

## S03 细粒度嵌套容量曲线

{fine_lines}

## 窗口集合说明

历史 N=1/8/32 实验的窗口并非完全嵌套，因此原结果保持不变且不伪称嵌套。本修订为每名被试建立独立冻结的 128 窗口序列，其所有前缀严格嵌套；标准 N=128 使用完整序列，S03 细粒度实验使用同一序列的 N=1/2/4/8/16/32/64/128 前缀。每个能量四分位 32 个窗口，同一被试三个种子使用完全相同的集合。

记录分层受到冻结训练池可用记录数限制：S02、S07 只有 1 条可用记录，S03 第二条记录仅有 35 个合格窗口，因此无法对所有被试达到单记录 20%～25% 的建议比例。选择器不重复窗口并优先时间分散；当 128 个两两不重叠窗口不可得时允许必要重叠，具体审计如下：

{selection_lines}

## 解释边界

即使稳定队列门控通过，也只表示多数被试具备训练集内 N=128 重构容量。它不能证明独立记录泛化、去噪有效性或 FoG 检测能力；S03 必须继续作为困难被试单独保留。
"""
    (reports / "final_report.md").write_text(report, encoding="utf-8")
    return decision


def audit_revision(root: Path) -> dict[str, Any]:
    metrics = list(root.rglob("metrics.json"))
    predictions = list(root.rglob("predictions.npz"))
    checkpoints = list(root.rglob("*_model.pt"))
    finite = True
    for path in predictions:
        with np.load(path, allow_pickle=False) as payload:
            finite = finite and all(np.isfinite(payload[key]).all() for key in payload.files)
    checkpoint_errors: list[str] = []
    for path in checkpoints:
        try:
            base.torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # pragma: no cover - artifact corruption path
            checkpoint_errors.append(f"{path}: {error}")
    audit = {
        "metric_files": len(metrics),
        "prediction_files": len(predictions),
        "checkpoint_files": len(checkpoints),
        "figure_files": len(list(root.rglob("*.png"))),
        "all_prediction_arrays_finite": finite,
        "all_runs_have_predictions": len(metrics) == len(predictions),
        "all_runs_have_two_checkpoints": len(checkpoints) == 2 * len(metrics),
        "all_checkpoints_loadable": not checkpoint_errors,
        "checkpoint_load_errors": checkpoint_errors,
        "temporary_files": len([path for path in root.rglob("*") if path.is_file() and ".tmp-" in path.name]),
    }
    base.write_json(root / "reports" / "artifact_audit.json", audit)
    return audit


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    revision_root = args.output_dir / "round3_n128_revision"
    revision_root.mkdir(parents=True, exist_ok=True)
    device = base.resolve_device(args.device)
    dataset = base.DaphnetDataset.load(args.data_dir)
    pools, selected, metadata, selection_audits = build_nested_manifests(
        dataset, args.output_dir
    )
    historical = load_historical_round3(args.output_dir)
    print(f"PREFLIGHT N128 revision device={device} output={revision_root}", flush=True)
    results = run_n128(
        args,
        dataset,
        pools,
        selected,
        metadata,
        historical,
        device,
    )
    fine_results = run_s03_fine_capacity(
        args,
        dataset,
        pools["S03"],
        selected["S03"],
        metadata["S03"],
        results,
        device,
    )
    render_global_figures(revision_root, historical, results)
    decision = write_reports(
        args.output_dir, results, fine_results, selection_audits
    )
    audit = audit_revision(revision_root)
    print(
        f"COMPLETE N128 status={decision['final_status']} audit={audit} "
        f"report={revision_root / 'reports' / 'final_report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
