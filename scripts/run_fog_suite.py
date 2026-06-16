#!/usr/bin/env python
"""Run a suite of FOG experiment configs and optionally collect results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.run_fog_experiment import (
        build_training_command,
        build_validation_command,
        build_windowing_command,
    )
except ImportError:  # pragma: no cover - used when executed from scripts/
    from run_fog_experiment import (
        build_training_command,
        build_validation_command,
        build_windowing_command,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple JSON-configured FOG experiments as one suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--only",
        choices=("all", "windowing", "validation", "training", "collection"),
        default="all",
    )
    parser.add_argument("--skip-windowing", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument(
        "--include-experiments",
        default="",
        help="Comma-separated experiment name/stem substrings to run, for example 'tcn' or 'sleepyco,binary'.",
    )
    parser.add_argument(
        "--exclude-experiments",
        default="",
        help="Comma-separated experiment name/stem substrings to skip.",
    )
    parser.add_argument(
        "--dedupe-windowing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run shared windowing/validation stages once per unique window output directory.",
    )
    parser.add_argument(
        "--reuse-existing-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip windowing when windows.npz/loso_folds.npz exist and config.json matches.",
    )
    parser.add_argument(
        "--skip-completed-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed suite experiments when result audit already passes.",
    )
    parser.add_argument(
        "--validate-experiment-configs",
        action="store_true",
        help="Build each concrete experiment stage command before running the suite.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print training completion status and exit without running stages.",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        help="Write a machine-readable training completion status JSON and exit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"[CMD] {format_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def experiment_config_path(entry: str | dict[str, Any]) -> Path:
    if isinstance(entry, str):
        return resolve_path(entry)
    if isinstance(entry, dict) and "config" in entry:
        return resolve_path(entry["config"])
    raise ValueError(f"Invalid experiment entry: {entry!r}")


def split_filter_tokens(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = ",".join(str(item) for item in value)
    else:
        raw = str(value)
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def experiment_match_text(config_path: Path, config: dict[str, Any]) -> str:
    return " ".join(
        [
            str(config.get("name", "")),
            config_path.stem,
            str(config.get("description", "")),
        ]
    ).lower()


def filter_experiments(
    config_paths: list[Path],
    configs: list[dict[str, Any]],
    include: str | list[str] | tuple[str, ...] | None = "",
    exclude: str | list[str] | tuple[str, ...] | None = "",
) -> tuple[list[Path], list[dict[str, Any]]]:
    include_tokens = split_filter_tokens(include)
    exclude_tokens = split_filter_tokens(exclude)
    selected_paths: list[Path] = []
    selected_configs: list[dict[str, Any]] = []
    for config_path, config in zip(config_paths, configs):
        text = experiment_match_text(config_path, config)
        if include_tokens and not any(token in text for token in include_tokens):
            continue
        if exclude_tokens and any(token in text for token in exclude_tokens):
            continue
        selected_paths.append(config_path)
        selected_configs.append(config)
    if not selected_paths:
        raise ValueError(
            f"Experiment filter selected no configs: include={include_tokens} exclude={exclude_tokens}"
        )
    return selected_paths, selected_configs


def stage_enabled(config: dict[str, Any], stage: str) -> bool:
    stage_config = config.get(stage) or {}
    return bool(stage_config.get("enabled", True))


def normalized_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [normalize(part) for part in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=False)


def comparable_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def values_match(left: Any, right: Any) -> bool:
    left_number = comparable_number(left)
    right_number = comparable_number(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) < 1e-9
    return left == right


def existing_windows_match(config: dict[str, Any]) -> tuple[bool, str]:
    windowing = config.get("windowing") or {}
    output_dir_value = windowing.get("output_dir")
    if not output_dir_value:
        return False, "windowing.output_dir missing"
    output_dir = resolve_path(output_dir_value)
    for name in ("windows.npz", "loso_folds.npz", "config.json"):
        if not (output_dir / name).exists():
            return False, f"missing {name}"

    existing = load_json(output_dir / "config.json")
    checks = [
        "window_seconds",
        "stride_seconds",
        "overlap",
        "label_mode",
        "pre_fog_seconds",
        "label_rule",
        "target_hz",
        "nan_policy",
        "require_success",
        "num_folds",
        "fold_seed",
        "max_records",
    ]
    for key in checks:
        requested = windowing.get(key)
        existing_value = existing.get(key)
        if requested is None and key not in windowing:
            continue
        if not values_match(requested, existing_value):
            return False, f"{key} mismatch requested={requested!r} existing={existing_value!r}"
    if "processed_dir" in windowing and existing.get("processed_dir"):
        requested_dir = str(resolve_path(windowing["processed_dir"]))
        if str(existing.get("processed_dir")) != requested_dir:
            return False, "processed_dir mismatch"
    return True, str(output_dir)


def stage_output_key(config_path: Path, config: dict[str, Any], stage: str) -> tuple[str, str]:
    if stage == "windowing":
        stage_config = config.get("windowing") or {}
        output_dir = stage_config.get("output_dir")
        if not output_dir:
            raise ValueError(f"{config_path} windowing config is missing output_dir")
        compare_config = dict(stage_config)
        compare_config["output_dir"] = str(resolve_path(output_dir))
        return str(resolve_path(output_dir)), normalized_json(compare_config)

    if stage == "validation":
        validation = config.get("validation") or {}
        windowing = config.get("windowing") or {}
        data_dir = validation.get("data_dir") or windowing.get("output_dir")
        if not data_dir:
            raise ValueError(f"{config_path} validation config needs data_dir or windowing.output_dir")
        compare_config = dict(validation)
        compare_config["data_dir"] = str(resolve_path(data_dir))
        return str(resolve_path(data_dir)), normalized_json(compare_config)

    raise ValueError(f"Unsupported dedupe stage: {stage}")


def unique_stage_configs(
    config_paths: list[Path],
    configs: list[dict[str, Any]],
    stage: str,
) -> list[Path]:
    seen: dict[str, tuple[str, Path]] = {}
    ordered: list[Path] = []
    for config_path, config in zip(config_paths, configs):
        if not stage_enabled(config, stage):
            continue
        key, fingerprint = stage_output_key(config_path, config, stage)
        if key in seen:
            previous_fingerprint, previous_path = seen[key]
            if previous_fingerprint != fingerprint:
                raise ValueError(
                    f"Conflicting {stage} configs share output {key}: "
                    f"{previous_path} and {config_path}"
                )
            continue
        seen[key] = (fingerprint, config_path)
        ordered.append(config_path)
    return ordered


def experiment_output_dir(config_path: Path) -> Path | None:
    config = load_json(config_path)
    training = config.get("training") or {}
    args = training.get("args") or {}
    output_dir = None
    if isinstance(args, dict):
        output_dir = args.get("output_dir")
    elif isinstance(args, list):
        for idx, item in enumerate(args[:-1]):
            if item == "--output-dir":
                output_dir = args[idx + 1]
                break
    return resolve_path(output_dir) if output_dir else None


def should_run_experiments(args: argparse.Namespace) -> bool:
    return args.only in {"all", "windowing", "validation", "training"}


def should_run_collection(args: argparse.Namespace, enabled: bool) -> bool:
    if not enabled or args.skip_collection:
        return False
    return args.only in {"all", "collection"}


def suite_training_complete(suite_path: Path) -> tuple[bool, str]:
    report, _ = suite_training_status_by_config(suite_path)
    found = int(report.get("found_aggregates", 0))
    expected = int(report.get("expected_aggregates", 0))
    return bool(report.get("ok")), f"{found}/{expected} aggregate outputs complete"


def suite_training_status_by_config(suite_path: Path) -> tuple[dict[str, Any], dict[Path, tuple[bool, str]]]:
    try:
        from scripts.audit_fog_suite_results import audit_suite_results
    except ImportError:  # pragma: no cover - used when executed from scripts/
        from audit_fog_suite_results import audit_suite_results

    report = audit_suite_results(suite_path)
    status: dict[Path, tuple[bool, str]] = {}
    for experiment in report.get("experiments", []):
        config_value = experiment.get("config")
        if not config_value:
            continue
        config_path = Path(str(config_value)).resolve()
        found = int(experiment.get("found_aggregates", 0))
        expected = int(experiment.get("expected_aggregates", len(experiment.get("results", []))))
        status[config_path] = (
            bool(experiment.get("ok")),
            f"{found}/{expected} aggregate outputs complete",
        )
    return report, status


def training_status_lines(report: dict[str, Any]) -> list[str]:
    found = int(report.get("found_aggregates", 0))
    expected = int(report.get("expected_aggregates", 0))
    state = "complete" if report.get("ok") else "incomplete"
    lines = [
        f"[STATUS] suite={report.get('suite', '')} state={state} aggregates={found}/{expected}"
    ]
    for experiment in report.get("experiments", []):
        exp_found = int(experiment.get("found_aggregates", 0))
        exp_expected = int(experiment.get("expected_aggregates", 0))
        exp_state = "complete" if experiment.get("ok") else "incomplete"
        missing = [
            str(result.get("variant", ""))
            for result in experiment.get("results", [])
            if not result.get("exists")
        ]
        missing_text = f" missing={','.join(missing)}" if missing else ""
        folds = experiment.get("expected_folds")
        fold_text = f" folds={folds}" if folds is not None else ""
        lines.append(
            f"[STATUS] {exp_state} {experiment.get('name', '')} "
            f"aggregates={exp_found}/{exp_expected}{fold_text}{missing_text}"
        )
    if report.get("errors"):
        lines.append(f"[STATUS] errors={len(report['errors'])}")
    return lines


def training_status_payload(report: dict[str, Any]) -> dict[str, Any]:
    experiments = []
    for experiment in report.get("experiments", []):
        missing = [
            str(result.get("variant", ""))
            for result in experiment.get("results", [])
            if not result.get("exists")
        ]
        experiments.append(
            {
                "name": experiment.get("name", ""),
                "config": experiment.get("config", ""),
                "output_dir": experiment.get("output_dir", ""),
                "state": "complete" if experiment.get("ok") else "incomplete",
                "ok": bool(experiment.get("ok")),
                "found_aggregates": int(experiment.get("found_aggregates", 0)),
                "expected_aggregates": int(experiment.get("expected_aggregates", 0)),
                "expected_folds": experiment.get("expected_folds"),
                "missing_variants": missing,
            }
        )
    return {
        "suite": report.get("suite", ""),
        "suite_config": report.get("suite_config", ""),
        "state": "complete" if report.get("ok") else "incomplete",
        "ok": bool(report.get("ok")),
        "found_aggregates": int(report.get("found_aggregates", 0)),
        "expected_aggregates": int(report.get("expected_aggregates", 0)),
        "error_count": len(report.get("errors", [])),
        "experiments": experiments,
    }


def filter_status_report(report: dict[str, Any], selected_paths: list[Path]) -> dict[str, Any]:
    selected_configs = {str(path.resolve()) for path in selected_paths}
    selected_experiments = [
        experiment
        for experiment in report.get("experiments", [])
        if str(Path(str(experiment.get("config", ""))).resolve()) in selected_configs
    ]
    selected_names = {str(experiment.get("name", "")) for experiment in selected_experiments}
    selected_errors = [
        error
        for error in report.get("errors", [])
        if not error.get("experiment") or str(error.get("experiment")) in selected_names
    ]
    filtered = dict(report)
    filtered["experiments"] = selected_experiments
    filtered["errors"] = selected_errors
    filtered["found_aggregates"] = int(
        sum(int(experiment.get("found_aggregates", 0)) for experiment in selected_experiments)
    )
    filtered["expected_aggregates"] = int(
        sum(int(experiment.get("expected_aggregates", 0)) for experiment in selected_experiments)
    )
    filtered["ok"] = bool(selected_experiments) and all(
        bool(experiment.get("ok")) for experiment in selected_experiments
    ) and not selected_errors
    return filtered


def write_status_json(path: Path, payload: dict[str, Any]) -> None:
    output_path = path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_experiment_command(python_exe: str, config_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        python_exe,
        str(REPO_ROOT / "scripts" / "run_fog_experiment.py"),
        "--config",
        str(config_path),
    ]
    if args.only in {"windowing", "validation", "training"}:
        command.extend(["--only", args.only])
    if args.skip_windowing:
        command.append("--skip-windowing")
    if args.skip_validation:
        command.append("--skip-validation")
    if args.skip_training:
        command.append("--skip-training")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_staged_experiment_command(
    python_exe: str,
    config_path: Path,
    stage: str,
    dry_run: bool,
) -> list[str]:
    command = [
        python_exe,
        str(REPO_ROOT / "scripts" / "run_fog_experiment.py"),
        "--config",
        str(config_path),
        "--only",
        stage,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def run_windowing_stage(
    python_exe: str,
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.reuse_existing_windows:
        reusable, reason = existing_windows_match(config)
        if reusable:
            output_dir = resolve_path((config.get("windowing") or {})["output_dir"])
            print(f"[SKIP] windowing exists and matches: {output_dir}", flush=True)
            return
        print(f"[INFO] windowing required for {config_path.name}: {reason}", flush=True)
    run_command(
        build_staged_experiment_command(python_exe, config_path, "windowing", args.dry_run),
        args.dry_run,
    )


def stage_requested(stage: str, args: argparse.Namespace) -> bool:
    if args.only not in {"all", stage}:
        return False
    if stage == "windowing" and args.skip_windowing:
        return False
    if stage == "validation" and args.skip_validation:
        return False
    if stage == "training" and args.skip_training:
        return False
    return True


def concrete_experiment_stage_commands(
    python_exe: str,
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    windowing = config.get("windowing") or {}
    validation = config.get("validation") or {}
    training = config.get("training") or {}
    commands: list[tuple[str, list[str]]] = []

    if stage_requested("windowing", args) and stage_enabled(config, "windowing"):
        commands.append(("windowing", build_windowing_command(python_exe, windowing)))
    if stage_requested("validation", args) and stage_enabled(config, "validation"):
        commands.append(("validation", build_validation_command(python_exe, validation, windowing)))
    if stage_requested("training", args) and stage_enabled(config, "training"):
        commands.append(("training", build_training_command(python_exe, training, windowing)))

    for stage, command in commands:
        if len(command) < 2:
            raise ValueError(f"{config_path} {stage} command is malformed: {command!r}")
        script_path = Path(command[1])
        if not script_path.exists():
            raise FileNotFoundError(f"{config_path} {stage} script does not exist: {script_path}")

    return commands


def validate_experiment_configs(
    python_exe: str,
    config_paths: list[Path],
    configs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    total = 0
    for config_path, config in zip(config_paths, configs):
        for stage, command in concrete_experiment_stage_commands(python_exe, config_path, config, args):
            total += 1
            print(f"[CHECK] {config_path.name}:{stage}: {format_command(command)}", flush=True)
    print(f"[INFO] validated {total} concrete experiment stage commands", flush=True)


def build_collection_command(
    python_exe: str,
    collection: dict[str, Any],
    experiment_config_paths: list[Path],
    args: argparse.Namespace,
) -> list[str]:
    output_dirs = collection.get("output_dirs")
    if output_dirs is None or output_dirs == "auto":
        inferred = [experiment_output_dir(path) for path in experiment_config_paths]
        output_dirs = [str(path) for path in inferred if path is not None]
    if not output_dirs:
        raise ValueError("No output dirs available for result collection.")

    command = [
        python_exe,
        str(REPO_ROOT / "scripts" / "collect_fog_results.py"),
        *[str(resolve_path(path)) for path in output_dirs],
    ]
    if "output_csv" in collection:
        command.extend(["--output-csv", str(resolve_path(collection["output_csv"]))])
    if "output_json" in collection:
        command.extend(["--output-json", str(resolve_path(collection["output_json"]))])
    if "recursive" in collection:
        command.append("--recursive" if collection["recursive"] else "--no-recursive")
    return command


def main() -> None:
    args = parse_args()
    suite_path = args.config.resolve()
    suite = load_json(suite_path)
    name = suite.get("name", suite_path.stem)
    experiments = suite.get("experiments") or []
    if not experiments:
        raise ValueError("Suite config must contain a non-empty experiments list.")

    experiment_paths = [experiment_config_path(entry) for entry in experiments]
    experiment_configs = [load_json(path) for path in experiment_paths]
    selected_paths, selected_configs = filter_experiments(
        experiment_paths,
        experiment_configs,
        args.include_experiments,
        args.exclude_experiments,
    )
    print(f"[INFO] suite={name}", flush=True)
    print(f"[INFO] config={suite_path}", flush=True)
    print(f"[INFO] experiments={len(experiment_paths)}", flush=True)
    if len(selected_paths) != len(experiment_paths):
        print(f"[INFO] selected_experiments={len(selected_paths)}", flush=True)

    if args.status or args.status_json:
        report, _ = suite_training_status_by_config(suite_path)
        if len(selected_paths) != len(experiment_paths):
            report = filter_status_report(report, selected_paths)
        if args.status:
            for line in training_status_lines(report):
                print(line, flush=True)
        if args.status_json:
            write_status_json(args.status_json, training_status_payload(report))
            print(f"[STATUS] wrote_json={args.status_json.resolve()}", flush=True)
        return

    if args.validate_experiment_configs:
        validate_experiment_configs(args.python, selected_paths, selected_configs, args)

    if should_run_experiments(args) and args.dedupe_windowing:
        if args.only in {"all", "windowing"} and not args.skip_windowing:
            for config_path in unique_stage_configs(selected_paths, selected_configs, "windowing"):
                config = load_json(config_path)
                run_windowing_stage(args.python, config_path, config, args)
        if args.only in {"all", "validation"} and not args.skip_validation:
            for config_path in unique_stage_configs(selected_paths, selected_configs, "validation"):
                run_command(
                    build_staged_experiment_command(args.python, config_path, "validation", args.dry_run),
                    args.dry_run,
                )
        if args.only in {"all", "training"} and not args.skip_training:
            suite_complete = False
            suite_reason = "skip check disabled"
            training_status: dict[Path, tuple[bool, str]] = {}
            if args.skip_completed_training and not args.dry_run:
                report, training_status = suite_training_status_by_config(suite_path)
                found = int(report.get("found_aggregates", 0))
                expected = int(report.get("expected_aggregates", 0))
                suite_complete = bool(report.get("ok"))
                suite_reason = f"{found}/{expected} aggregate outputs complete"
            if suite_complete:
                print(f"[SKIP] training outputs already complete: {suite_reason}", flush=True)
            else:
                if args.skip_completed_training and not args.dry_run:
                    print(f"[INFO] training incomplete: {suite_reason}", flush=True)
                for config_path in selected_paths:
                    if not stage_enabled(load_json(config_path), "training"):
                        continue
                    experiment_complete, experiment_reason = training_status.get(
                        config_path.resolve(),
                        (False, "not audited"),
                    )
                    if experiment_complete:
                        print(
                            f"[SKIP] training complete for {config_path.name}: {experiment_reason}",
                            flush=True,
                        )
                        continue
                    run_command(
                        build_staged_experiment_command(args.python, config_path, "training", args.dry_run),
                        args.dry_run,
                    )

    elif should_run_experiments(args):
        suite_complete = False
        suite_reason = "skip check disabled"
        training_status: dict[Path, tuple[bool, str]] = {}
        if (
            args.only in {"all", "training"}
            and not args.skip_training
            and args.skip_completed_training
            and not args.dry_run
        ):
            report, training_status = suite_training_status_by_config(suite_path)
            found = int(report.get("found_aggregates", 0))
            expected = int(report.get("expected_aggregates", 0))
            suite_complete = bool(report.get("ok"))
            suite_reason = f"{found}/{expected} aggregate outputs complete"
        if suite_complete and args.only == "training":
            print(f"[SKIP] training outputs already complete: {suite_reason}", flush=True)
        else:
            if args.skip_completed_training and not args.dry_run and args.only in {"all", "training"}:
                print(f"[INFO] training incomplete: {suite_reason}", flush=True)
            command_args = args
            if suite_complete and args.only == "all":
                print(f"[SKIP] training outputs already complete: {suite_reason}", flush=True)
                command_args = argparse.Namespace(**{**vars(args), "skip_training": True})
            for config_path in selected_paths:
                per_config_args = command_args
                experiment_complete, experiment_reason = training_status.get(
                    config_path.resolve(),
                    (False, "not audited"),
                )
                if experiment_complete and args.only == "training":
                    print(
                        f"[SKIP] training complete for {config_path.name}: {experiment_reason}",
                        flush=True,
                    )
                    continue
                if experiment_complete and args.only == "all" and not suite_complete:
                    print(
                        f"[SKIP] training complete for {config_path.name}: {experiment_reason}",
                        flush=True,
                    )
                    per_config_args = argparse.Namespace(**{**vars(args), "skip_training": True})
                run_command(build_experiment_command(args.python, config_path, per_config_args), args.dry_run)

    collection = suite.get("collection") or {}
    if should_run_collection(args, bool(collection.get("enabled", True))):
        run_command(
            build_collection_command(args.python, collection, selected_paths, args),
            args.dry_run,
        )


if __name__ == "__main__":
    main()
