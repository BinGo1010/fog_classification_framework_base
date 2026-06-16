import argparse
from pathlib import Path
import numpy as np


def make_split(n, channels, seq_len, num_classes, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=n)
    X = rng.normal(0, 1, size=(n, channels, seq_len)).astype("float32")
    # Add weak class-dependent temporal pattern for smoke training.
    t = np.linspace(0, 2*np.pi, seq_len)
    for c in range(num_classes):
        X[y == c, 0, :] += np.sin((c + 1) * t) * 0.5
    return X, y.astype("int64")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/dummy")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--seq_len", type=int, default=128)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, n, seed in [("train", 512, 1), ("val", 128, 2), ("test", 128, 3)]:
        X, y = make_split(n, args.channels, args.seq_len, args.num_classes, seed)
        np.savez(out / f"{split}.npz", X=X, y=y)
    print(f"Saved dummy dataset to {out}")


if __name__ == "__main__":
    main()
