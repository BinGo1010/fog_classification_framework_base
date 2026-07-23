#!/usr/bin/env python
"""Verify completeness and internal consistency of a CNBR-FoG LOSO run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetTrunkDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("outputs/cnbr_fog_daphnet_trunk_loso"),
    )
    return parser.parse_args()


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if not np.isclose(actual, expected, rtol=tolerance, atol=tolerance):
        raise AssertionError(f"Metric mismatch: recomputed={actual}, saved={expected}")


def main() -> None:
    root = parse_args().result_dir.resolve()
    with (root / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (root / "aggregate_metrics.json").open("r", encoding="utf-8") as handle:
        aggregate = json.load(handle)
    subjects = list(config.get("folds_resolved", config["subjects"]))
    inputs = list(config.get("inputs_resolved", []))
    if not inputs:
        inputs = [
            value.strip()
            for value in str(config.get("baselines", "residual,raw")).split(",")
            if value.strip()
        ]
    if not subjects or not inputs:
        raise AssertionError("No resolved folds or classifier inputs in config")
    excluded = set(config.get("excluded_subjects", []))
    uses_nbm = bool(config.get("uses_nbm", True))
    if excluded.intersection(subjects):
        raise AssertionError("An excluded subject appears in the resolved LOSO folds")
    expected_windows = int(config.get("evaluation_windows", config["windows"]))
    expected_counts = list(
        config.get("evaluation_window_class_counts", config["window_class_counts"])
    )
    windows = None
    dataset = None
    if config.get("history_variants"):
        dataset = DaphnetTrunkDataset.load(
            config["data_dir"], flatline_seconds=float(config["flatline_seconds"])
        )
        if excluded:
            dataset = DaphnetTrunkDataset(
                root=dataset.root,
                records=[
                    record for record in dataset.records if record.subject_id not in excluded
                ],
                sampling_rate_hz=dataset.sampling_rate_hz,
            )
        windows = dataset.make_windows(
            warmup_samples=int(config["context_samples"]),
            target_samples=int(config["horizon_samples"]),
            stride_samples=int(config["stride_samples"]),
            fog_fraction_threshold=float(config["fog_fraction_threshold"]),
            normal_guard_samples=int(config["normal_guard_samples"]),
        )

    reference_truth: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    report: dict[str, dict] = {}
    for input_name in inputs:
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        for subject in subjects:
            fold_dir = root / f"loso_{subject}"
            required = [
                fold_dir / "fold_config.json",
                fold_dir / input_name / "classifier_best.pt",
                fold_dir / input_name / "metrics.json",
                fold_dir / input_name / "predictions.npz",
                fold_dir / input_name / "predictions.csv",
            ]
            normal_checkpoint = fold_dir / "normal_predictor_best.pt"
            if uses_nbm:
                required.append(normal_checkpoint)
            elif normal_checkpoint.exists():
                raise AssertionError(
                    f"Raw-only fold unexpectedly contains an NBM checkpoint: {subject}"
                )
            if config.get("history_variants"):
                required.append(fold_dir / "history_support.npz")
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise AssertionError(f"Missing fold outputs: {missing}")
            with np.load(fold_dir / input_name / "predictions.npz", allow_pickle=False) as data:
                y_true = np.asarray(data["y_true"], dtype=np.int8)
                y_prob = np.asarray(data["y_prob"], dtype=np.float64)
                y_pred = np.asarray(data["y_pred"], dtype=np.int8)
                window_index = np.asarray(data["window_index"], dtype=np.int64)
            if input_name == inputs[0] and windows is not None and dataset is not None:
                with (fold_dir / "fold_config.json").open("r", encoding="utf-8") as handle:
                    fold_config = json.load(handle)
                participants = {
                    fold_config["test_subject"],
                    fold_config["val_subject"],
                    *fold_config["train_subjects"],
                }
                if participants.intersection(excluded):
                    raise AssertionError(f"Excluded subject used in fold {subject}")
                with np.load(fold_dir / "history_support.npz", allow_pickle=False) as support:
                    support_anchor = np.asarray(
                        support["test_anchor_window_index"], dtype=np.int64
                    )
                    support_chain = np.asarray(
                        support["test_history_window_index"], dtype=np.int64
                    )
                if not np.array_equal(support_anchor, window_index):
                    raise AssertionError(f"Support/prediction anchor mismatch for {subject}")
                maximum_blocks = max(
                    int(item["history_blocks"]) for item in config["history_variants"]
                )
                if support_chain.shape != (len(window_index), maximum_blocks):
                    raise AssertionError(f"Unexpected support shape for {subject}")
                if not np.array_equal(support_chain[:, -1], support_anchor):
                    raise AssertionError(f"Anchor is not the final history block for {subject}")
                record_ids = windows.record_index[support_chain]
                if not np.all(record_ids == record_ids[:, :1]):
                    raise AssertionError(f"History crosses a record boundary for {subject}")
                starts = windows.target_start[support_chain]
                if not np.all(np.diff(starts, axis=1) == int(config["horizon_samples"])):
                    raise AssertionError(f"History blocks are not contiguous for {subject}")
                ends = windows.target_end[support_chain]
                if not np.all(ends - starts == int(config["horizon_samples"])):
                    raise AssertionError(f"History block has wrong horizon for {subject}")
                if not np.all(ends <= windows.target_end[support_anchor, None]):
                    raise AssertionError(f"Future residual block used for {subject}")
                history_subjects = {
                    dataset.records[int(record_index)].subject_id
                    for record_index in record_ids[:, 0]
                }
                if history_subjects != {subject}:
                    raise AssertionError(
                        f"History support belongs to {history_subjects}, expected {subject}"
                    )
            with (fold_dir / input_name / "metrics.json").open("r", encoding="utf-8") as handle:
                fold_metrics = json.load(handle)
            if not (len(y_true) == len(y_prob) == len(y_pred) == len(window_index)):
                raise AssertionError(f"Prediction length mismatch for {input_name}/{subject}")
            if not np.isfinite(y_prob).all() or not ((0 <= y_prob) & (y_prob <= 1)).all():
                raise AssertionError(f"Invalid probabilities for {input_name}/{subject}")
            expected_pred = (y_prob >= float(fold_metrics["threshold"])).astype(np.int8)
            if not np.array_equal(y_pred, expected_pred):
                raise AssertionError(f"Threshold decision mismatch for {input_name}/{subject}")
            fold_cm = [
                int(((y_true == 0) & (y_pred == 0)).sum()),
                int(((y_true == 0) & (y_pred == 1)).sum()),
                int(((y_true == 1) & (y_pred == 0)).sum()),
                int(((y_true == 1) & (y_pred == 1)).sum()),
            ]
            saved_fold_cm = [
                fold_metrics["tn"],
                fold_metrics["fp"],
                fold_metrics["fn"],
                fold_metrics["tp"],
            ]
            if fold_cm != saved_fold_cm:
                raise AssertionError(
                    f"Fold confusion mismatch for {input_name}/{subject}: "
                    f"recomputed={fold_cm} saved={saved_fold_cm}"
                )
            tn, fp, fn, tp = fold_cm
            f1_nonfog = (
                2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
            )
            f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            if "macro_f1" in fold_metrics:
                close(
                    0.5 * (f1_nonfog + f1_fog),
                    float(fold_metrics["macro_f1"]),
                )
            if "fog_recall" in fold_metrics:
                close(
                    tp / (tp + fn) if tp + fn else 0.0,
                    float(fold_metrics["fog_recall"]),
                )
            if "fog_f1" in fold_metrics:
                close(f1_fog, float(fold_metrics["fog_f1"]))
            variant_config = next(
                (
                    item
                    for item in config.get("history_variants", [])
                    if item["input"] == input_name
                ),
                None,
            )
            if variant_config is not None:
                if int(fold_metrics["history_blocks"]) != int(
                    variant_config["history_blocks"]
                ):
                    raise AssertionError(
                        f"History block count mismatch for {input_name}/{subject}"
                    )
                if int(fold_metrics["input_samples"]) != int(
                    variant_config["history_samples"]
                ):
                    raise AssertionError(
                        f"Input length mismatch for {input_name}/{subject}"
                    )
            truths.append(y_true)
            probabilities.append(y_prob)
            predictions.append(y_pred)
            indices.append(window_index)

        y_true = np.concatenate(truths)
        y_prob = np.concatenate(probabilities)
        y_pred = np.concatenate(predictions)
        window_index = np.concatenate(indices)
        if len(y_true) != expected_windows:
            raise AssertionError(f"Expected {expected_windows} windows, got {len(y_true)}")
        observed_counts = np.bincount(y_true, minlength=2).tolist()
        if observed_counts != expected_counts:
            raise AssertionError(
                f"Class count mismatch: {observed_counts} != {expected_counts}"
            )
        if reference_truth is None:
            reference_truth = y_true
            reference_indices = window_index
        elif not np.array_equal(reference_truth, y_true):
            raise AssertionError("Classifier inputs do not cover identical targets")
        elif not np.array_equal(reference_indices, window_index):
            raise AssertionError("Classifier inputs do not cover identical anchor windows")

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        pooled = aggregate[input_name]["pooled"]
        if [tn, fp, fn, tp] != [pooled["tn"], pooled["fp"], pooled["fn"], pooled["tp"]]:
            raise AssertionError(
                f"Confusion-matrix mismatch for {input_name}: "
                f"recomputed={[tn, fp, fn, tp]} saved="
                f"{[pooled['tn'], pooled['fp'], pooled['fn'], pooled['tp']]}"
            )
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        close(auroc, pooled["auroc"])
        close(auprc, pooled["auprc"])
        completed = aggregate[input_name].get("completed_folds", [])
        if completed != subjects:
            raise AssertionError(
                f"Completed fold mismatch for {input_name}: {completed} != {subjects}"
            )
        report[input_name] = {
            "windows": int(len(y_true)),
            "class_counts": observed_counts,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "auroc": auroc,
            "auprc": auprc,
        }

    print("AUDIT_OK")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
