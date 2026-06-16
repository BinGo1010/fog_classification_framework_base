import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_provider.window_cache import build_windows


def save_index_split(out_dir, split, arrays, indices):
    payload = {}
    n = len(arrays["y"])
    for key, value in arrays.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == n:
            payload[key] = value[indices]
        else:
            payload[key] = value
    np.savez(out_dir / f"{split}.npz", **payload)


def split_random_windows(y, val_size, test_size, seed):
    indices = np.arange(len(y))
    holdout_size = val_size + test_size
    train_idx, holdout_idx, _, holdout_y = train_test_split(
        indices,
        y,
        test_size=holdout_size,
        random_state=seed,
        stratify=y,
    )
    relative_test_size = test_size / holdout_size
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_size,
        random_state=seed,
        stratify=holdout_y,
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def split_summary(arrays, indices):
    y = arrays["y"][indices]
    subjects = arrays["subjectID"][indices]
    return {
        "num_windows": int(len(indices)),
        "num_positive": int(y.sum()),
        "positive_fraction": float(y.mean()) if len(y) else 0.0,
        "subjects": sorted(map(int, np.unique(subjects))),
        "num_subjects": int(len(np.unique(subjects))),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare FoG-STAR as random-window train/val/test NPZ files.")
    parser.add_argument("--sensor_csv", default="data/sensor_data.csv")
    parser.add_argument("--clinical_csv", default="data/clinical_data.csv")
    parser.add_argument("--out_dir", default="data/fogstar_random_window")
    parser.add_argument("--window_size", type=int, default=120, help="Samples per window. At 60 Hz, 120 = 2 seconds.")
    parser.add_argument("--stride", type=int, default=60, help="Window stride in samples. At 60 Hz, 60 = 1 second.")
    parser.add_argument("--fog_threshold", type=float, default=0.25, help="Minimum fraction of FoG samples for a positive window.")
    parser.add_argument("--keep_activity_zero", action="store_true", help="Keep activity=0 samples. By default they are removed.")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sensor_df = pd.read_csv(args.sensor_csv)
    clinical_df = pd.read_csv(args.clinical_csv)
    arrays = build_windows(
        sensor_df,
        clinical_df,
        args.window_size,
        args.stride,
        args.fog_threshold,
        keep_activity_zero=args.keep_activity_zero,
    )
    splits = split_random_windows(arrays["y"], args.val_size, args.test_size, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split_strategy": "random_window",
        "warning": "Windows are randomly split; subjects and neighboring overlapping windows may appear in multiple splits.",
        "window_size": args.window_size,
        "stride": args.stride,
        "fog_threshold": args.fog_threshold,
        "keep_activity_zero": args.keep_activity_zero,
        "sampling_rate_hz": 60,
        "seed": args.seed,
        "splits": {},
    }

    for split, indices in splits.items():
        save_index_split(out_dir, split, arrays, indices)
        summary["splits"][split] = split_summary(arrays, indices)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved random-window FoG-STAR NPZ dataset to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
