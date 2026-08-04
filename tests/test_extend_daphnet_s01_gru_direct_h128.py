from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extend_daphnet_s01_gru_direct_h128.py"
SPEC = importlib.util.spec_from_file_location("extend_s01_direct_h128", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extension = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extension)


def _history() -> list[dict[str, int | float | bool]]:
    return [
        {
            "epoch": epoch,
            "cumulative_optimizer_steps": epoch * 4,
            "validation_rmse_scaled": 2.0 - epoch / 1000,
            "improved": True,
        }
        for epoch in range(1, 126)
    ]


def _locked_configs(data_dir: Path) -> tuple[dict, dict, dict, dict, dict]:
    root_training = {
        "hidden_channels": 48,
        "dropout": 0.1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "maximum_optimizer_steps": 500,
        "patience": 15,
        "minimum_optimizer_steps": 32,
        "amp": True,
    }
    stage_training = {
        "hidden_channels": 48,
        "dropout": 0.1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "maximum_optimizer_steps": 500,
        "minimum_optimizer_steps": 32,
        "patience_evaluations": 15,
        "min_delta_rmse": extension.suite.MIN_DELTA_RMSE,
    }
    upstream = {
        "experiment_version": extension.suite.EXPERIMENT_VERSION,
        "protocol_fingerprint": "frozen",
        "seeds": list(extension.EXPECTED_SEEDS),
        "data_dir": str(data_dir.resolve()),
        "records_loaded": ["S01_seg000", "S01_seg001"],
        "device_type": "cuda",
        "hyperparameters": root_training,
    }
    done = {
        "status": "complete",
        "experiment_version": extension.suite.EXPERIMENT_VERSION,
        "protocol_fingerprint": "frozen",
    }
    runtime = {
        "device": "cuda",
        "cuda_device_name": "test-gpu",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    long_mean = {
        **stage_training,
        "model_name": "current_grunbm_direct_mean_path",
        "horizon_samples": 128,
    }
    horizon = {
        "horizons": [{"id": "h200", "samples": 128}],
        "training": {
            **stage_training,
            "model_name": "pure_mean_common_gru_direct_decoder",
        },
    }
    return upstream, done, runtime, long_mean, horizon


def test_first_500_audit_is_strict_and_complete() -> None:
    reference = _history()
    candidate = [*reference, {**reference[-1], "epoch": 126, "cumulative_optimizer_steps": 504}]
    audit = extension.audit_history_prefix(reference, candidate)
    assert audit["exact_row_for_row_match"] is True
    assert audit["rows_compared"] == 125
    assert audit["reference_history_canonical_sha256"] == audit[
        "candidate_history_canonical_sha256"
    ]

    changed = [dict(row) for row in candidate]
    changed[71]["validation_rmse_scaled"] = 99.0
    mismatch = extension.audit_history_prefix(reference, changed)
    assert mismatch["exact_row_for_row_match"] is False
    assert mismatch["first_mismatch"] == {
        "reason": "value",
        "row": 72,
        "field": "validation_rmse_scaled",
        "expected": reference[71]["validation_rmse_scaled"],
        "actual": 99.0,
    }


def test_first_500_audit_requires_reference_to_end_at_500() -> None:
    with pytest.raises(ValueError, match="exactly at 500"):
        extension.audit_history_prefix(_history()[:-1], _history())


def test_locked_protocol_accepts_only_exact_cuda_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = ROOT / "synthetic-processed"
    args = extension.parse_args(["--data-dir", str(data_dir)])
    upstream, done, runtime, long_mean, horizon = _locked_configs(data_dir)
    monkeypatch.setattr(extension.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        extension.torch.cuda, "get_device_name", lambda _device: "test-gpu"
    )
    extension.validate_locked_protocol(
        args=args,
        seeds=extension.EXPECTED_SEEDS,
        device=torch.device("cuda"),
        upstream_config=upstream,
        upstream_done=done,
        upstream_runtime=runtime,
        long_mean_config=long_mean,
        horizon_config=horizon,
    )

    with pytest.raises(RuntimeError, match="locked to CUDA"):
        extension.validate_locked_protocol(
            args=args,
            seeds=extension.EXPECTED_SEEDS,
            device=torch.device("cpu"),
            upstream_config=upstream,
            upstream_done=done,
            upstream_runtime=runtime,
            long_mean_config=long_mean,
            horizon_config=horizon,
        )


def test_locked_protocol_rejects_scientific_cli_override() -> None:
    data_dir = ROOT / "synthetic-processed"
    args = extension.parse_args(
        ["--data-dir", str(data_dir), "--max-steps", "2001"]
    )
    upstream, done, runtime, long_mean, horizon = _locked_configs(data_dir)
    with pytest.raises(ValueError, match="max-steps is locked"):
        extension.validate_locked_protocol(
            args=args,
            seeds=extension.EXPECTED_SEEDS,
            device=torch.device("cuda"),
            upstream_config=upstream,
            upstream_done=done,
            upstream_runtime=runtime,
            long_mean_config=long_mean,
            horizon_config=horizon,
        )


def test_loaded_input_contract_never_names_r02() -> None:
    paths = extension._input_paths(Path("processed"))
    names = [path.name for path in paths]
    assert names == ["manifest.csv", "schema.json", "S01_seg000.npz", "S01_seg001.npz"]
    assert all("seg002" not in str(path).lower() for path in paths)


def test_upstream_reference_inventory_covers_every_seed() -> None:
    paths = extension._upstream_paths(
        ROOT / "synthetic-upstream", extension.EXPECTED_SEEDS
    )
    summaries = [path for path in paths if path.name == "summary.json"]
    assert len(summaries) == len(extension.EXPECTED_SEEDS)
    for seed in extension.EXPECTED_SEEDS:
        assert any(f"seed_{seed}" in str(path) for path in summaries)
