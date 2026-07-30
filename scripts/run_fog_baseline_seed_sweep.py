#!/usr/bin/env python
"""Run the four reference baselines for the five prespecified seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_daphnet_baseline_suite.py"
GPU_SCHEDULER = (
    REPO_ROOT / "scripts" / "start_daphnet_baseline_suite_multigpu.py"
)
AUDITOR = REPO_ROOT / "scripts" / "audit_daphnet_baseline_suite.py"
AGGREGATOR = (
    REPO_ROOT / "scripts" / "aggregate_fog_baseline_multiseed.py"
)
DEFAULT_SEEDS = (3407, 3408, 3409, 3410, 3411)
METHODS = "freeze_index,tf_svm,tf_rf,cnn_gru"
RESERVED_PREFIXES = (
    "--data-dir",
    "--output-dir",
    "--seed",
    "--seeds",
    "--methods",
    "--dataset-adapter",
    "--exclude-subjects",
    "--folds",
    "--worker-fold",
    "--finalize-only",
    "--device",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seed_list(specification: str) -> tuple[int, ...]:
    values = tuple(
        int(value.strip())
        for value in str(specification).split(",")
        if value.strip()
    )
    if not values or len(values) != len(set(values)):
        raise ValueError("--seeds must contain unique integers")
    return values


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Five-seed LOSO launcher for the four paper baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--dataset-adapter",
        choices=("daphnet", "manifest_npz"),
        default="daphnet",
    )
    parser.add_argument("--exclude-subjects", default="S04,S10")
    parser.add_argument(
        "--launcher",
        choices=("multigpu", "direct"),
        default="multigpu",
    )
    parser.add_argument("--gpus", default="0-6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--launch-delay", type=float, default=2.0)
    parser.add_argument(
        "--audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=3411)
    parser.add_argument("--reference-pr-csv", type=Path)
    args, forwarded = parser.parse_known_args()
    for value in forwarded:
        if any(
            value == prefix or value.startswith(prefix + "=")
            for prefix in RESERVED_PREFIXES
        ):
            raise ValueError(
                f"{value} is controlled by the seed-sweep launcher"
            )
    if args.launcher == "multigpu" and args.dataset_adapter != "daphnet":
        raise ValueError(
            "The current multi-GPU fold scheduler is canonical-Daphnet only; "
            "use --launcher direct for manifest_npz private data"
        )
    return args, forwarded


def save_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp()}] command={subprocess.list2cmdline(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"[{timestamp()}] return_code={result.returncode}\n")
    return int(result.returncode)


def main() -> None:
    args, forwarded = parse_args()
    seeds = parse_seed_list(args.seeds)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_root = root / "seed_sweep_logs"
    status_path = root / "seed_sweep_status.json"
    state: dict[str, Any] = {
        "sweep_version": "fog_reference_baselines_seed_sweep.v1",
        "status": "running",
        "started_at_utc": timestamp(),
        "updated_at_utc": timestamp(),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(root),
        "dataset_adapter": args.dataset_adapter,
        "exclude_subjects": args.exclude_subjects,
        "methods": METHODS.split(","),
        "seeds": list(seeds),
        "launcher": args.launcher,
        "forwarded_scientific_args": forwarded,
        "runs": {
            str(seed): {
                "status": "pending",
                "output_dir": str(root / f"seed_{seed}"),
                "return_code": None,
            }
            for seed in seeds
        },
    }
    save_status(status_path, state)
    for seed in seeds:
        seed_output = root / f"seed_{seed}"
        common = [
            "--seed",
            str(seed),
            "--dataset-adapter",
            args.dataset_adapter,
            "--exclude-subjects",
            args.exclude_subjects,
            "--methods",
            METHODS,
            *forwarded,
        ]
        if args.launcher == "multigpu":
            command = [
                sys.executable,
                "-u",
                str(GPU_SCHEDULER),
                "--data-dir",
                str(args.data_dir.resolve()),
                "--output-dir",
                str(seed_output),
                "--gpus",
                args.gpus,
                "--max-retries",
                str(args.max_retries),
                "--launch-delay",
                str(args.launch_delay),
                *(("--audit",) if args.audit else ()),
                *common,
            ]
        else:
            command = [
                sys.executable,
                "-u",
                str(RUNNER),
                "--data-dir",
                str(args.data_dir.resolve()),
                "--output-dir",
                str(seed_output),
                "--folds",
                "all",
                "--device",
                args.device,
                *common,
            ]
        state["runs"][str(seed)]["status"] = "running"
        state["updated_at_utc"] = timestamp()
        save_status(status_path, state)
        return_code = run(command, log_root / f"seed_{seed}.log")
        if return_code == 0 and args.audit and args.launcher == "direct":
            return_code = run(
                [
                    sys.executable,
                    "-u",
                    str(AUDITOR),
                    "--data-dir",
                    str(args.data_dir.resolve()),
                    "--output-dir",
                    str(seed_output),
                ],
                log_root / f"seed_{seed}_audit.log",
            )
        state["runs"][str(seed)]["return_code"] = return_code
        state["runs"][str(seed)]["status"] = (
            "complete" if return_code == 0 else "failed"
        )
        state["updated_at_utc"] = timestamp()
        save_status(status_path, state)
        if return_code:
            state["status"] = "failed"
            state["ended_at_utc"] = timestamp()
            save_status(status_path, state)
            raise SystemExit(return_code)

    aggregate_command = [
        sys.executable,
        "-u",
        str(AGGREGATOR),
        "--output-dir",
        str(root),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--methods",
        METHODS,
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
    ]
    if args.reference_pr_csv is not None:
        aggregate_command.extend(
            ["--reference-pr-csv", str(args.reference_pr_csv.resolve())]
        )
    aggregate_code = run(
        aggregate_command,
        log_root / "aggregate.log",
    )
    state["aggregate_return_code"] = aggregate_code
    state["status"] = "complete" if aggregate_code == 0 else "failed"
    state["updated_at_utc"] = timestamp()
    state["ended_at_utc"] = timestamp()
    save_status(status_path, state)
    if aggregate_code:
        raise SystemExit(aggregate_code)
    print(f"[seed-sweep] complete output={root}", flush=True)


if __name__ == "__main__":
    main()
