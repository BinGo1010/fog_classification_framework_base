#!/usr/bin/env python
"""Generate leakage-controlled centered reconstruction residuals for one subject."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset, Record
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    FS,
    INNER_HALF_GAP,
    OUTER_HALF_GAP,
    WINDOW,
    Interval,
    WindowSet,
    build_windows,
    calibrate,
    fit_scaler_unique_points,
    normalized_residual,
    prepare_nbm_windows,
    raw_windows,
    residual_diagnostics,
    resolve_device,
    save_npz,
    set_seed,
    train_nbm,
    write_csv,
    write_json,
)


SUBJECT_SPLITS: dict[str, dict[str, Any]] = {
    "S02": {"earlier": [], "cut_record": "S02_seg000", "cut": 17_152, "test": "S02_seg001", "ignored": []},
    "S03": {"earlier": ["S03_seg000"], "cut_record": "S03_seg001", "cut": 8_576, "test": "S03_seg002", "ignored": ["S03_seg003"]},
    "S05": {"earlier": ["S05_seg000", "S05_seg001", "S05_seg002", "S05_seg003"], "cut_record": "S05_seg004", "cut": 4_736, "test": "S05_seg005", "ignored": []},
    "S06": {"earlier": ["S06_seg000"], "cut_record": "S06_seg001", "cut": 6_144, "test": "S06_seg002", "ignored": ["S06_seg003", "S06_seg004"]},
    "S07": {"earlier": [], "cut_record": "S07_seg000", "cut": 51_968, "test": "S07_seg001", "ignored": []},
    "S08": {"earlier": ["S08_seg000", "S08_seg001"], "cut_record": "S08_seg002", "cut": 1_920, "test": "S08_seg003", "ignored": []},
    "S09": {"earlier": ["S09_seg000", "S09_seg001", "S09_seg002"], "cut_record": "S09_seg003", "cut": 15_552, "test": "S09_seg004", "ignored": []},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=sorted(SUBJECT_SPLITS))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_cut(record: Record, target: int, lower: int, upper: int) -> int:
    target = int(round(target / 64.0) * 64)
    for distance in range(0, max(len(record.y), 64), 64):
        candidates = [target] if distance == 0 else [target - distance, target + distance]
        for candidate in candidates:
            if candidate - INNER_HALF_GAP < lower + WINDOW:
                continue
            if candidate + INNER_HALF_GAP > upper - WINDOW:
                continue
            if np.any(record.y[candidate - INNER_HALF_GAP : candidate + INNER_HALF_GAP] == 1):
                continue
            return int(candidate)
    raise ValueError(f"no event-safe inner cut in {record.record_id} near {target}")


def build_subject_intervals(subject: str, records: list[Record]) -> tuple[list[Interval], dict[str, Any]]:
    spec = SUBJECT_SPLITS[subject]
    by_name = {record.record_id: record for record in records}
    base: list[tuple[str, int, int]] = []
    for record_id in spec["earlier"]:
        base.append((record_id, 0, len(by_name[record_id].y)))
    base.append((spec["cut_record"], 0, int(spec["cut"]) - OUTER_HALF_GAP))
    lengths = np.asarray([end - start for _, start, end in base], dtype=np.int64)
    cumulative = np.r_[0, np.cumsum(lengths)]
    total = int(cumulative[-1])
    boundaries: list[tuple[int, int]] = []
    for fraction in range(1, 5):
        target_global = total * fraction / 5.0
        base_index = min(int(np.searchsorted(cumulative[1:], target_global, side="right")), len(base) - 1)
        record_id, start, end = base[base_index]
        local_target = start + int(target_global - cumulative[base_index])
        cut = _safe_cut(by_name[record_id], local_target, start, end)
        boundaries.append((base_index, cut))
    if len(set(boundaries)) != 4:
        raise ValueError(f"duplicate inner boundaries: {boundaries}")

    train_intervals: list[Interval] = []
    block = 0
    for base_index, (record_id, start, end) in enumerate(base):
        cuts = sorted(cut for index, cut in boundaries if index == base_index)
        cursor = start
        for cut in cuts:
            piece_end = cut - INNER_HALF_GAP
            if piece_end - cursor >= WINDOW:
                train_intervals.append(Interval(record_id, cursor, piece_end, "train", block))
            block += 1
            cursor = cut + INNER_HALF_GAP
        if end - cursor >= WINDOW:
            train_intervals.append(Interval(record_id, cursor, end, "train", block))
    if block != 4 or sorted({item.block for item in train_intervals}) != list(range(5)):
        raise ValueError(f"failed to construct five blocks: block={block}, intervals={train_intervals}")

    cut_record = by_name[spec["cut_record"]]
    test_record = by_name[spec["test"]]
    intervals = train_intervals + [
        Interval(spec["cut_record"], int(spec["cut"]) + OUTER_HALF_GAP, len(cut_record.y), "validation", -1),
        Interval(spec["test"], 0, len(test_record.y), "test", -1),
    ]
    disclosure = {
        "historical_cut": spec,
        "inner_boundaries": [
            {"record_id": base[index][0], "sample": cut} for index, cut in boundaries
        ],
        "inner_total_gap_seconds": 2 * INNER_HALF_GAP / FS,
        "outer_train_validation_gap_seconds": 2 * OUTER_HALF_GAP / FS,
    }
    return intervals, disclosure


def train_crossfit(
    subject: str,
    records: list[Record],
    windows: WindowSet,
    train_indices: np.ndarray,
    output_dir: Path,
    device,
    seed: int,
    num_workers: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    residuals = np.empty((len(train_indices), WINDOW, 9), dtype=np.float32)
    assigned = np.zeros(len(train_indices), dtype=bool)
    lookup = {int(index): row for row, index in enumerate(train_indices)}
    summaries: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for holdout in range(5):
        calibration_block = (holdout + 1) % 5
        fit_blocks = [block for block in range(5) if block not in {holdout, calibration_block}]
        fit_indices = train_indices[
            windows.clean_normal[train_indices] & np.isin(windows.block[train_indices], fit_blocks)
        ]
        calibration_indices = train_indices[
            windows.clean_normal[train_indices] & (windows.block[train_indices] == calibration_block)
        ]
        holdout_indices = train_indices[windows.block[train_indices] == holdout]
        if min(len(fit_indices), len(calibration_indices), len(holdout_indices)) == 0:
            raise ValueError(
                f"{subject} fold {holdout}: empty fit/cal/holdout "
                f"{len(fit_indices)}/{len(calibration_indices)}/{len(holdout_indices)}"
            )
        scaler = fit_scaler_unique_points(records, windows, fit_indices)
        fit_x = prepare_nbm_windows(
            scaler, raw_windows(records, windows, fit_indices), center=True
        )
        calibration_x = prepare_nbm_windows(
            scaler, raw_windows(records, windows, calibration_indices), center=True
        )
        model_id = f"inner_fold_{holdout + 1:02d}_nbm"
        model, training = train_nbm(
            model_id,
            fit_x,
            calibration_x,
            output_dir,
            device,
            seed + holdout,
            num_workers,
        )
        bias, sigma, calibration = calibrate(model, calibration_x, device)
        holdout_residual, _, _ = normalized_residual(
            model,
            scaler,
            bias,
            sigma,
            raw_windows(records, windows, holdout_indices),
            device,
            center_windows=True,
        )
        for source_row, index in enumerate(holdout_indices):
            row = lookup[int(index)]
            if assigned[row]:
                raise AssertionError("duplicate OOF residual")
            assigned[row] = True
            residuals[row] = holdout_residual[source_row]
            record = records[int(windows.record_index[index])]
            manifest.append(
                {
                    "sample_id": f"{record.record_id}:{int(windows.start[index])}:{int(windows.end[index])}",
                    "subject_id": subject,
                    "record_id": record.record_id,
                    "window_start": int(windows.start[index]),
                    "window_end": int(windows.end[index]),
                    "label": int(windows.label[index]),
                    "inner_fold": holdout + 1,
                    "nbm_model_id": model_id,
                    "nbm_seen_this_window": False,
                }
            )
        summaries.append(
            {
                **training["summary"],
                "holdout_block": holdout,
                "calibration_block": calibration_block,
                "fit_blocks": fit_blocks,
                "holdout_windows": int(len(holdout_indices)),
                "holdout_fog_windows": int(windows.label[holdout_indices].sum()),
                "residual_calibration": calibration,
            }
        )
    if not np.all(assigned):
        raise AssertionError(f"unassigned OOF rows: {int((~assigned).sum())}")
    return residuals, summaries, manifest


def fit_final(
    records: list[Record],
    windows: WindowSet,
    train_indices: np.ndarray,
    output_dir: Path,
    device,
    seed: int,
    num_workers: int,
):
    fit_indices = train_indices[
        windows.clean_normal[train_indices] & np.isin(windows.block[train_indices], [0, 1, 2, 3])
    ]
    calibration_indices = train_indices[
        windows.clean_normal[train_indices] & (windows.block[train_indices] == 4)
    ]
    scaler = fit_scaler_unique_points(records, windows, fit_indices)
    fit_x = prepare_nbm_windows(scaler, raw_windows(records, windows, fit_indices), center=True)
    calibration_x = prepare_nbm_windows(
        scaler, raw_windows(records, windows, calibration_indices), center=True
    )
    model, training = train_nbm(
        "final_nbm", fit_x, calibration_x, output_dir, device, seed + 100, num_workers
    )
    bias, sigma, calibration = calibrate(model, calibration_x, device)
    return model, scaler, bias, sigma, {
        **training["summary"],
        "fit_blocks": [0, 1, 2, 3],
        "calibration_block": 4,
        "residual_calibration": calibration,
    }


def class_median(residual: np.ndarray, labels: np.ndarray, label: int) -> float | None:
    selected = residual[labels == label]
    return float(np.median(np.abs(selected))) if len(selected) else None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = DaphnetDataset.load(args.data_dir)
    records = [record for record in dataset.records if record.subject_id == args.subject]
    intervals, disclosure = build_subject_intervals(args.subject, records)
    windows = build_windows(records, intervals)
    split_indices = {name: windows.indices(name) for name in ("train", "validation", "test")}
    for name, indices in split_indices.items():
        if np.unique(windows.label[indices]).size != 2:
            raise ValueError(f"{args.subject} {name} lacks one class")
    preflight = {
        name: {
            "windows": int(len(indices)),
            "non_fog": int(np.sum(windows.label[indices] == 0)),
            "fog": int(np.sum(windows.label[indices] == 1)),
            "clean_normal": int(np.sum(windows.clean_normal[indices])),
        }
        for name, indices in split_indices.items()
    }
    config = {
        "experiment": "centered_reconstruction_nbm_residuals_for_inceptiontime_v1",
        "subject": args.subject,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "data_dir": str(args.data_dir.resolve()),
        "split": disclosure,
        "intervals": [item.__dict__ for item in intervals],
        "statistics": preflight,
        "window_axis_centering": True,
        "residual": "clip((centered_scaled_X-mu-b)/(sigma+1e-6),-12,12), then window-axis center",
        "classifier_note": "Residuals are classifier-independent and are intended for the six-module InceptionTime classifier.",
    }
    write_json(output_dir / "config.json", config)
    print(f"PREFLIGHT {args.subject} device={device} stats={preflight}", flush=True)
    if args.dry_run:
        write_json(output_dir / "DRY_RUN.json", {"status": "complete"})
        return

    train_indices = split_indices["train"]
    oof, inner_training, manifest = train_crossfit(
        args.subject,
        records,
        windows,
        train_indices,
        output_dir,
        device,
        args.seed,
        args.num_workers,
    )
    model, scaler, bias, sigma, final_training = fit_final(
        records,
        windows,
        train_indices,
        output_dir,
        device,
        args.seed,
        args.num_workers,
    )
    validation_indices = split_indices["validation"]
    test_indices = split_indices["test"]
    validation, _, _ = normalized_residual(
        model,
        scaler,
        bias,
        sigma,
        raw_windows(records, windows, validation_indices),
        device,
        center_windows=True,
    )
    test, _, _ = normalized_residual(
        model,
        scaler,
        bias,
        sigma,
        raw_windows(records, windows, test_indices),
        device,
        center_windows=True,
    )
    labels = {
        "oof_train": windows.label[train_indices],
        "validation": windows.label[validation_indices],
        "test": windows.label[test_indices],
    }
    residuals = {"oof_train": oof, "validation": validation, "test": test}
    medians = {
        name: {
            "non_fog_median_absolute_residual": class_median(values, labels[name], 0),
            "fog_median_absolute_residual": class_median(values, labels[name], 1),
            "non_fog_windows": int(np.sum(labels[name] == 0)),
            "fog_windows": int(np.sum(labels[name] == 1)),
        }
        for name, values in residuals.items()
    }
    validation_nonfog = medians["validation"]["non_fog_median_absolute_residual"]
    test_nonfog = medians["test"]["non_fog_median_absolute_residual"]
    shift_ratio = float(test_nonfog / validation_nonfog)
    domain_shift = {
        "test_to_validation_nonfog_median_abs_ratio": shift_ratio,
        "criterion": "obvious if ratio > 1.5 or ratio < 0.67",
        "obvious": bool(shift_ratio > 1.5 or shift_ratio < 0.67),
        "direction": "test_larger" if shift_ratio > 1 else "test_smaller",
    }
    diagnostics = {
        "subject": args.subject,
        "median_absolute_residual": medians,
        "test_record_domain_shift": domain_shift,
        "full_diagnostics": {
            name: residual_diagnostics(residuals[name], labels[name])
            for name in residuals
        },
    }
    write_json(output_dir / "residual_statistics.json", diagnostics)
    write_json(
        output_dir / "training.json",
        {"inner_nbm": inner_training, "final_nbm": final_training},
    )
    write_csv(output_dir / "oof_manifest.csv", manifest)
    save_npz(
        output_dir / "residuals_for_inceptiontime.npz",
        train_oof_residual=oof,
        train_y=labels["oof_train"],
        validation_residual=validation,
        validation_y=labels["validation"],
        test_residual=test,
        test_y=labels["test"],
    )
    write_json(
        output_dir / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "residual_statistics": diagnostics,
        },
    )
    print(
        f"COMPLETE {args.subject} medians={medians} shift_ratio={shift_ratio:.6f} "
        f"obvious={domain_shift['obvious']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
