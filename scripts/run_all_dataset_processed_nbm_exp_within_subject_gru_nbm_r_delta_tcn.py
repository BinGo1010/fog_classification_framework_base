#!/usr/bin/env python
"""Strict [r, delta(r)] GRU-BASE-NBM + TCN experiment on processed_NBM_Exp.

The completed FULL_C experiment is the immutable NBM source.  Every job reuses
the exact role-4 Scaler, GRU BASE Mask4-8 checkpoint, and role-5 sigma from its
paired subject/fold/seed source.  Only a 60-channel TCN is trained with
[r, delta(r)]; abs(r) is removed.  Roles 0/1 remain locked until all 120 TCN
checkpoints and validation-selected thresholds have been globally sealed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cnbr_fog.resume import capture_rng_state, restore_rng_state
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_only_tcn as base
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import RepresentationTCNM
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler, set_seed


_BASE_R_ONLY_FEATURES = base.r_only_features


SUBJECTS = base.SUBJECTS
FOLDS = base.FOLDS
SEEDS = base.SEEDS
ROLES = base.ROLES
METRIC_KEYS = base.METRIC_KEYS
RAW_CHANNELS = 30
REFERENCE_TCN_CHANNELS = 90
TCN_INPUT_CHANNELS = 60
TCN_PARAMETER_COUNT = 139_809
REPRESENTATION = "r_delta"
TCN_CHECKPOINT_NAME = "tcn_r_delta.pt"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_gru_nbm_r_delta_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_gru_nbm_r_delta_tcn_barrier.v1"
MODEL_DESCRIPTION = "frozen GRU-BASE Mask4-8 NBM + [r,delta(r)] 60-channel TCN"
ABLATION_DESCRIPTION = "remove abs(r); retain centered standardized r and delta(r)"
EVENT_METRIC_VERSION = "allocation_group_any_window_nonfog_runs.v1"
EVENT_MINIMUM_POSITIVE_WINDOWS = 1
EVENT_MERGE_GAP_SECONDS = 1.0
EVENT_FALSE_ALARM_DENOMINATOR = (
    "total false-alarm runs on role-0 Non-FoG windows divided by union coverage "
    "of evaluated valid Non-FoG samples"
)
EVENT_AGGREGATION = "pooled_counts_and_exposure"
AGGREGATION_DESCRIPTION = (
    "window metrics: subject/seed macro mean of 3 folds, then subject-macro per "
    "seed and mean+population SD over 5 seeds; event sensitivity: detected "
    "allocation groups / all allocation groups; FA/h: total role-0 false-alarm "
    "runs / total valid Non-FoG union exposure, each pooled within fold, then "
    "3-fold mean per seed and 5-seed mean+population SD"
)


def r_delta_features(
    model: nn.Module,
    scaler: RobustScaler,
    sigma: np.ndarray,
    raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return [r, delta(r)] in BCT order, using the frozen scheme-C r."""

    residual = _BASE_R_ONLY_FEATURES(
        model, scaler, sigma, raw, device, batch_size
    )
    delta = np.diff(
        residual,
        axis=2,
        prepend=residual[:, :, :1],
    ).astype(np.float32, copy=False)
    features = np.concatenate((residual, delta), axis=1)
    if features.shape[1:] != (TCN_INPUT_CHANNELS, 128):
        raise AssertionError(f"unexpected [r,delta(r)] feature shape: {features.shape}")
    if not np.all(delta[:, :, 0] == 0.0):
        raise AssertionError("delta(r) must use a zero first difference")
    return np.ascontiguousarray(features, dtype=np.float32)


def paired_r_delta_tcn(
    seed: int,
    expected_reference_hash: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, str]]:
    """Select r and delta input weights from the exact 90-channel initialization."""

    set_seed(seed)
    reference = RepresentationTCNM(REFERENCE_TCN_CHANNELS)
    reference_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in reference.state_dict().items()
    }
    reference_hash = base.expanded.state_dict_sha256(reference_state)
    if reference_hash != expected_reference_hash:
        raise AssertionError(
            "cannot reproduce source 90-channel TCN initialization; "
            "code/runtime identity drifted"
        )
    rng_after_reference = capture_rng_state()
    target = RepresentationTCNM(TCN_INPUT_CHANNELS)
    target_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in target.state_dict().items()
    }
    selected_channels = torch.cat(
        (
            torch.arange(0, RAW_CHANNELS, dtype=torch.long),
            torch.arange(2 * RAW_CHANNELS, 3 * RAW_CHANNELS, dtype=torch.long),
        )
    )
    for name, target_tensor in target_state.items():
        source_tensor = reference_state[name]
        if target_tensor.shape == source_tensor.shape:
            target_tensor.copy_(source_tensor)
        elif (
            target_tensor.ndim == 3
            and source_tensor.ndim == 3
            and target_tensor.shape[0] == source_tensor.shape[0]
            and target_tensor.shape[2] == source_tensor.shape[2]
            and target_tensor.shape[1] == TCN_INPUT_CHANNELS
            and source_tensor.shape[1] == REFERENCE_TCN_CHANNELS
        ):
            target_tensor.copy_(source_tensor.index_select(1, selected_channels))
        else:
            raise AssertionError(
                f"unhandled paired TCN parameter {name}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
    target.load_state_dict(target_state)
    target = target.to(device)
    restore_rng_state(rng_after_reference)
    parameter_count = sum(parameter.numel() for parameter in target.parameters())
    if parameter_count != TCN_PARAMETER_COUNT:
        raise RuntimeError(f"r+delta TCN parameter contract changed: {parameter_count}")
    return target, {
        "reference_90ch_initial_state_sha256": reference_hash,
        "r_delta_60ch_initial_state_sha256": base.expanded.state_dict_sha256(
            target_state
        ),
        "selected_reference_input_channels": "0:30 and 60:90",
    }


@lru_cache(maxsize=4)
def _permanent_fog_group_lookup(dataset_root: str) -> dict[str, str]:
    path = Path(dataset_root) / "nbm_window_manifest.csv"
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["active_for_outer_fold"].strip().lower() != "true":
                continue
            if int(row["role_code"]) != 1:
                continue
            window_id = row["window_id"]
            group_id = row["allocation_group_id"]
            previous = lookup.setdefault(window_id, group_id)
            if previous != group_id:
                raise AssertionError(
                    f"permanent FoG window changes allocation group: {window_id}"
                )
    if not lookup:
        raise ValueError(f"no permanent FoG groups found in {path}")
    return lookup


def final_event_metrics(
    dataset: Any,
    rows: Any,
    y_pred: np.ndarray,
    minimum_positive_windows: int = 1,
    merge_gap_seconds: float = 1.0,
) -> dict[str, Any]:
    """Use the pre-registered allocation-group event and Non-FoG FA/h rules."""

    if minimum_positive_windows != 1 or merge_gap_seconds != 1.0:
        raise ValueError("final event metric is fixed to one positive window and a 1 s gap")
    prediction = np.asarray(y_pred, dtype=np.int8)
    row_count = len(rows.window_id)
    if prediction.shape != (row_count,):
        raise ValueError("prediction/row length mismatch")
    group_lookup = _permanent_fog_group_lookup(str(dataset.root.resolve()))

    group_predictions: dict[str, list[int]] = defaultdict(list)
    for window_id, role, label, pred in zip(
        rows.window_id, rows.role, rows.label, prediction
    ):
        if int(role) != 1:
            continue
        if int(label) != 1:
            raise AssertionError("role-1 event window is not FoG")
        key = str(window_id)
        if key not in group_lookup:
            raise KeyError(f"role-1 window absent from allocation manifest: {key}")
        group_predictions[group_lookup[key]].append(int(pred))
    if not group_predictions:
        raise ValueError("no permanent-test FoG allocation groups were evaluated")
    detected = sum(int(any(values)) for values in group_predictions.values())
    event_count = len(group_predictions)

    false_alarms = 0
    nonfog_seconds = 0.0
    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for record_index, start, end, role, label, pred in zip(
        rows.record_index,
        rows.start,
        rows.end,
        rows.role,
        rows.label,
        prediction,
    ):
        if int(role) != 0:
            continue
        if int(label) != 0:
            raise AssertionError("role-0 false-alarm support is not Non-FoG")
        by_record[int(record_index)].append((int(start), int(end), int(pred)))
    maximum_start_gap = int(round(dataset.sampling_rate_hz * merge_gap_seconds))
    for record_index, record_rows in by_record.items():
        record_rows.sort(key=lambda item: item[0])
        positive_starts = [start for start, _, pred in record_rows if pred == 1]
        if positive_starts:
            false_alarms += 1 + sum(
                int(current - previous > maximum_start_gap)
                for previous, current in zip(positive_starts, positive_starts[1:])
            )
        record = dataset.records[record_index]
        coverage = np.zeros(len(record.y), dtype=bool)
        for start, end, _ in record_rows:
            coverage[start:end] = True
        nonfog_seconds += float(
            np.sum(coverage & record.valid & (record.y == 0))
        ) / dataset.sampling_rate_hz
    nonfog_hours = nonfog_seconds / 3600.0
    return {
        "event_metric_version": EVENT_METRIC_VERSION,
        "minimum_positive_windows": 1,
        "merge_gap_seconds": 1.0,
        "reference_event_unit": "permanent-test FoG allocation_group_id",
        "false_alarm_support": "role-0 Non-FoG windows only",
        "evaluable_true_events": int(event_count),
        "detected_true_events": int(detected),
        "false_alarm_events": int(false_alarms),
        "event_sensitivity": detected / event_count,
        "false_alarm_events_per_hour": false_alarms / nonfog_hours,
        "evaluated_nonfog_hours": nonfog_hours,
    }


def training_contract(args: Any) -> dict[str, Any]:
    return {
        "ablation": ABLATION_DESCRIPTION,
        "frozen_source": (
            "same per-job role4 Scaler, GRU-BASE Mask4-8 NBM, and role5 sigma "
            "as the expanded FULL_C experiment"
        ),
        "source_trainable_parameters_updated": False,
        "residual": (
            "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
            "r=q-mean_t(q); delta(r)[0]=0"
        ),
        "input": "concatenate [r,delta(r)] along channels; abs(r) absent",
        "input_shape": ["B", TCN_INPUT_CHANNELS, 128],
        "tcn": (
            "RepresentationTCNM 60->32->64->64->128; "
            "dilations1/2/4/8; GAP; one logit"
        ),
        "paired_initialization": (
            "shared tensors copied exactly from the paired 90-channel source; "
            "input weights select source channels 0:30 (r) and 60:90 (delta)"
        ),
        "train_roles": [6, 7],
        "validation_roles": [2, 3],
        "test_roles": [0, 1],
        "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "batch_size": args.batch_size,
        "maximum_epochs": args.tcn_max_epochs,
        "patience": args.tcn_patience,
        "checkpoint": "maximum roles2/3 PR-AUC",
        "threshold": (
            "roles2/3 grid0.05..0.95 step0.01; max balanced accuracy; "
            "ties F1 then higher threshold"
        ),
        "event_metric": {
            "reference_event": "one permanent-test FoG allocation group",
            "detected": "any group window predicted FoG",
            "false_alarm": (
                "role-0 only; same-record positive decisions <=1 s apart merged"
            ),
            "exposure": "union coverage of evaluated valid Non-FoG samples",
        },
    }


def configure_base() -> None:
    """Configure the proven strict worker inside this isolated subprocess."""

    base.__doc__ = __doc__
    base.TCN_INPUT_CHANNELS = TCN_INPUT_CHANNELS
    base.TCN_PARAMETER_COUNT = TCN_PARAMETER_COUNT
    base.REPRESENTATION = REPRESENTATION
    base.TCN_CHECKPOINT_NAME = TCN_CHECKPOINT_NAME
    base.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    base.BARRIER_SCHEMA = BARRIER_SCHEMA
    base.MODEL_DESCRIPTION = MODEL_DESCRIPTION
    base.ABLATION_DESCRIPTION = ABLATION_DESCRIPTION
    base.EVENT_METRIC_VERSION = EVENT_METRIC_VERSION
    base.EVENT_MINIMUM_POSITIVE_WINDOWS = EVENT_MINIMUM_POSITIVE_WINDOWS
    base.EVENT_MERGE_GAP_SECONDS = EVENT_MERGE_GAP_SECONDS
    base.EVENT_FALSE_ALARM_DENOMINATOR = EVENT_FALSE_ALARM_DENOMINATOR
    base.EVENT_AGGREGATION = EVENT_AGGREGATION
    base.AGGREGATION_DESCRIPTION = AGGREGATION_DESCRIPTION
    base.r_only_features = r_delta_features
    base.paired_r_only_tcn = paired_r_delta_tcn
    base.training_contract = training_contract
    base.raw_base.event_metrics = final_event_metrics
    base.raw_base.EVENT_METRIC_VERSION = EVENT_METRIC_VERSION


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
