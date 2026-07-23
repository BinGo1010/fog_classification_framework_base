from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.start_daphnet_3imu_nbm_suite_multigpu import (
    CANONICAL_FOLDS,
    build_finalize_command,
    build_worker_command,
    main,
    parse_gpu_spec,
    parse_work_folds,
    process_is_alive,
    validate_forwarded,
)


def test_default_topology_is_seven_gpus_for_eight_folds() -> None:
    assert parse_gpu_spec("0-6") == ["0", "1", "2", "3", "4", "5", "6"]
    assert parse_work_folds("all") == list(CANONICAL_FOLDS)
    assert len(CANONICAL_FOLDS) == 8


def test_worker_and_finalize_commands_keep_one_shared_protocol(tmp_path: Path) -> None:
    runner = tmp_path / "run suite.py"
    data_dir = tmp_path / "processed data"
    output_dir = tmp_path / "results"
    forwarded = ["--normal-epochs", "12", "--num-workers", "2"]

    worker = build_worker_command(
        sys.executable,
        runner,
        data_dir,
        output_dir,
        "S05",
        forwarded,
    )
    assert worker[worker.index("--folds") + 1] == "all"
    assert worker[worker.index("--worker-fold") + 1] == "S05"
    assert worker[worker.index("--device") + 1] == "cuda"
    assert "--resume" in worker
    assert worker[-4:] == forwarded

    finalize = build_finalize_command(
        sys.executable,
        runner,
        data_dir,
        output_dir,
        forwarded,
    )
    assert finalize[finalize.index("--folds") + 1] == "all"
    assert finalize[finalize.index("--device") + 1] == "cpu"
    assert "--finalize-only" in finalize
    assert "--worker-fold" not in finalize


@pytest.mark.parametrize(
    "argument",
    [
        "--device=cuda:1",
        "--folds",
        "--worker-fold=S01",
        "--no-resume",
        "--finalize-only",
        "--help",
        "-h",
        "--dev=cpu",
        "--worker-f=S01",
        "--out",
    ],
)
def test_scheduler_rejects_forwarded_control_options(argument: str) -> None:
    with pytest.raises(ValueError, match="controlled"):
        validate_forwarded([argument])


def test_process_probe_does_not_terminate_the_process() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        assert process_is_alive(process.pid)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_dry_run_queues_eighth_fold_and_does_not_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "must-not-exist"
    reverse_folds = ",".join(reversed(CANONICAL_FOLDS))
    return_code = main(
        [
            "--dry-run",
            "--data-dir",
            str(tmp_path / "processed"),
            "--output-dir",
            str(output_dir),
            "--gpus",
            "0-6",
            "--work-folds",
            reverse_folds,
            "--python",
            sys.executable,
        ]
    )

    rendered = capsys.readouterr().out
    assert return_code == 0
    assert "max_parallel=7" in rendered
    assert "CUDA_VISIBLE_DEVICES=<first-free>" in rendered
    assert " --allow-partial" not in rendered
    assert not output_dir.exists()


def test_partial_dry_run_requests_partial_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = main(
        [
            "--dry-run",
            "--data-dir",
            str(tmp_path / "processed"),
            "--output-dir",
            str(tmp_path / "results"),
            "--gpus",
            "0",
            "--work-folds",
            "S01,S02",
            "--python",
            sys.executable,
        ]
    )

    rendered = capsys.readouterr().out
    assert return_code == 0
    assert " --allow-partial" in rendered
