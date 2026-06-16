#!/usr/bin/env python
"""Cross-platform post-check for sample-level processed FOG datasets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate processed records and run a window dry-run without writing windows.npz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--expected-channels", type=int, default=0)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--label-mode", choices=("binary", "three-class"), default="binary")
    parser.add_argument("--nan-policy", choices=("error", "zero"), default="error")
    parser.add_argument("--target-hz", type=float, default=0.0)
    parser.add_argument("--allow-nan", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def run_step(name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n== {name} ==", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def make_temp_output(repo_root: Path, processed_dir: Path) -> Path:
    outputs = repo_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    safe_leaf = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in processed_dir.name)
    return Path(tempfile.mkdtemp(prefix=f"_tmp_window_dry_run_{safe_leaf}_", dir=outputs))


def remove_temp_output(path: Path, repo_root: Path) -> None:
    outputs = (repo_root / "outputs").resolve()
    resolved = path.resolve()
    if not resolved.exists():
        return
    if outputs not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside outputs/: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    processed_dir = args.processed_dir.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(processed_dir)

    output_dir = make_temp_output(repo_root, processed_dir)
    print(f"RepoRoot: {repo_root}")
    print(f"ProcessedDir: {processed_dir}")
    print(f"WindowDryRunDir: {output_dir}")

    try:
        validate_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "validate_processed_records.py"),
            str(processed_dir),
        ]
        if args.expected_channels > 0:
            validate_cmd.extend(["--expected-channels", str(args.expected_channels)])
        if args.allow_nan:
            validate_cmd.append("--allow-nan")
        if args.require_success:
            validate_cmd.append("--require-success")
        run_step("Validate sample-level processed records", validate_cmd, repo_root)

        window_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "prepare_processed_record_windows.py"),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            str(args.window_seconds),
            "--stride-seconds",
            str(args.stride_seconds),
            "--label-mode",
            args.label_mode,
            "--nan-policy",
            args.nan_policy,
            "--dry-run",
        ]
        if args.target_hz > 0:
            window_cmd.extend(["--target-hz", str(args.target_hz)])
        if args.require_success:
            window_cmd.append("--require-success")
        run_step("Window dry-run", window_cmd, repo_root)

        if (output_dir / "windows.npz").exists():
            raise RuntimeError("Window dry-run unexpectedly created windows.npz")
        if not (output_dir / "file_summary.csv").exists():
            raise RuntimeError("Window dry-run did not create file_summary.csv")
        if not (output_dir / "config.json").exists():
            raise RuntimeError("Window dry-run did not create config.json")

        print("\nProcessed pipeline check passed.")
    finally:
        if not args.keep_output:
            remove_temp_output(output_dir, repo_root)
            print("Removed temporary window dry-run output.")


if __name__ == "__main__":
    main()
