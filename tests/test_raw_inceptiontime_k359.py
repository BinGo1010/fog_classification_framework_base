from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_daphnet_full_subject_nbm_residual_binary as exp  # noqa: E402
import run_daphnet_full_subject_raw_inceptiontime_k359 as small  # noqa: E402


def test_small_kernel_model_shape_and_config() -> None:
    model = small.SmallKernelInceptionTimeClassifier(9)
    output = model(torch.randn(2, 9, 128))
    config = model.architecture_config()
    assert output.shape == (2,)
    assert config["kernel_sizes"] == [3, 5, 9]
    assert config["module_count"] == 6
    assert config["parameter_count"] < 472897


def test_configure_experiment_runs_raw_only() -> None:
    small.configure_experiment()
    assert exp.METHODS == ("B0",)
    assert exp.METHOD_NAMES["B0"] == "Raw-InceptionTime-K3-5-9"
    assert exp.METHOD_CHANNELS == {"B0": 9}
    assert exp.METHOD_DIRS == {"B0": "B0_raw_inceptiontime_k359"}


def test_raw_only_aggregation_writes_final_bundle(tmp_path: Path) -> None:
    small.configure_experiment()
    for subject in exp.SUBJECTS:
        for seed in exp.SEEDS:
            path = (
                tmp_path
                / small.METHOD_DIR
                / subject
                / f"{subject}_seg000"
                / f"seed{seed}"
                / "test_predictions.csv"
            )
            exp.write_csv(
                path,
                [
                    {
                        "subject_id": subject,
                        "fold_id": f"{subject}_seg000",
                        "method": "B0",
                        "seed": seed,
                        "record_id": f"{subject}_seg000",
                        "block_id": f"{subject}_seg000",
                        "window_start": 0,
                        "y_true": 0,
                        "y_prob": 0.1,
                        "y_pred": 0,
                        "threshold": 0.5,
                    },
                    {
                        "subject_id": subject,
                        "fold_id": f"{subject}_seg000",
                        "method": "B0",
                        "seed": seed,
                        "record_id": f"{subject}_seg000",
                        "block_id": f"{subject}_seg000",
                        "window_start": 64,
                        "y_true": 1,
                        "y_prob": 0.9,
                        "y_pred": 1,
                        "threshold": 0.5,
                    },
                ],
            )
    result = small.aggregate_raw_results(tmp_path, bootstrap_samples=20)
    assert result["kernel_sizes"] == [3, 5, 9]
    assert (tmp_path / "FINAL_RESULTS.json").exists()
    assert (tmp_path / "tables" / "subject_level_main_results.csv").exists()
    assert (tmp_path / "reports" / "raw_inceptiontime_k359_report.md").exists()
