from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import report_b0_inceptiontime_confusion_matrices as report
import run_daphnet_full_subject_nbm_residual_binary as exp
import run_daphnet_full_subject_raw_inceptiontime_k359 as small


DEFAULT_ROOT = (
    ROOT
    / "outputs"
    / "daphnet_full_subject_raw_inceptiontime_k359_server_v1"
    / "full_subject_binary_experiment"
)
SPLITS = ("train", "validation", "test")


def read_saved_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    return (
        frame["y_true"].to_numpy(dtype=int),
        frame["y_prob"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=int),
    )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


@torch.no_grad()
def train_predictions(
    run_dir: Path,
    inputs: np.ndarray,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = small.SmallKernelInceptionTimeClassifier(inputs.shape[2]).to(device)
    payload = torch.load(run_dir / "inceptiontime_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    probability = exp.predict_classifier(model, inputs, device)
    prediction = (probability >= threshold).astype(int)
    del model
    return probability, prediction


def collect_run_counts(root: Path, device: torch.device) -> pd.DataFrame:
    small.configure_experiment()
    rows: list[dict[str, Any]] = []
    fold_summary = pd.read_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv")
    expected = len(fold_summary) * len(exp.SEEDS)
    completed = 0
    for fold in fold_summary.itertuples(index=False):
        subject = str(fold.subject_id)
        fold_id = str(fold.fold_id)
        cache = root / "splits" / "outer_folds" / subject / fold_id / "representations.npz"
        arrays = dict(np.load(cache, allow_pickle=False))
        inner = arrays["inner_fold"]
        validation_fold = int(arrays["validation_inner_fold"][0])
        train_mask = (inner >= 0) & (inner != validation_fold)
        train_x = arrays["train_x"][train_mask]
        train_y = arrays["train_y"].astype(int)[train_mask]
        for seed in exp.SEEDS:
            run_dir = root / small.METHOD_DIR / subject / fold_id / f"seed{seed}"
            run_result = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
            threshold = float(run_result["threshold"])
            _, prediction = train_predictions(run_dir, train_x, threshold, device)
            common = {
                "subject_id": subject,
                "fold_id": fold_id,
                "method": small.METHOD,
                "method_name": small.METHOD_NAME,
                "seed": int(seed),
                "threshold": threshold,
            }
            rows.append(
                common
                | {"split": "train", "n_windows": len(train_y), "positive_windows": int(train_y.sum())}
                | confusion_counts(train_y, prediction)
            )
            for split, filename in (
                ("validation", "validation_predictions.csv"),
                ("test", "test_predictions.csv"),
            ):
                y_true, _, saved_prediction = read_saved_predictions(run_dir / filename)
                rows.append(
                    common
                    | {
                        "split": split,
                        "n_windows": len(y_true),
                        "positive_windows": int(y_true.sum()),
                    }
                    | confusion_counts(y_true, saved_prediction)
                )
            completed += 1
            if completed == 1 or completed % 15 == 0 or completed == expected:
                print(
                    f"INFERENCE {completed}/{expected} {subject}/{fold_id} seed={seed}",
                    flush=True,
                )
        del arrays
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    output = root / "analysis_b0_confusion_matrices"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        torch.cuda.set_device(device)

    run_counts = collect_run_counts(root, device)
    fold_counts = report.seed_median_fold_counts(run_counts)
    overall, subjects = report.summarize_counts(fold_counts)
    pooled = pd.read_csv(root / "predictions" / "seed_median_pooled_predictions.csv")
    overall = pd.concat(
        [overall, pd.DataFrame([report.official_test_row(pooled)])], ignore_index=True
    )

    run_counts.to_csv(output / "run_seed_confusion_counts.csv", index=False, encoding="utf-8-sig")
    fold_counts.to_csv(
        output / "fold_seed_median_confusion_counts.csv", index=False, encoding="utf-8-sig"
    )
    subjects.to_csv(output / "subject_split_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "overall_split_metrics.csv", index=False, encoding="utf-8-sig")
    report.plot_confusion_matrices(
        overall,
        output / "raw_inceptiontime_k359_confusion_matrices",
        figure_title=(
            "Raw + InceptionTime [3, 5, 9]: train, validation and outer-test confusion matrices"
        ),
    )
    report.write_report(
        overall,
        output / "raw_inceptiontime_k359_confusion_matrix_report.md",
        method_title="Raw + InceptionTime [3, 5, 9]",
    )
    manifest = {
        "experiment": small.EXPERIMENT,
        "method": small.METHOD_NAME,
        "kernel_sizes": list(small.KERNEL_SIZES),
        "subjects": list(exp.SUBJECTS),
        "outer_folds": int(pd.read_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv").shape[0]),
        "classifier_seeds": list(exp.SEEDS),
        "run_rows": int(len(run_counts)),
        "expected_run_rows": 30 * 3 * 3,
        "training_predictions": "recomputed from each inceptiontime_best.pt",
        "validation_predictions": "saved validation_predictions.csv",
        "test_predictions": "saved test_predictions.csv",
        "aggregation": "component-wise seed median per outer fold, followed by fold sum",
        "official_test": "unique windows; median probability and majority vote of seed-specific thresholded labels",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False), flush=True)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()

