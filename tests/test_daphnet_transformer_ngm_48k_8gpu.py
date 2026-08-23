from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts import launch_daphnet_processed_nbm_transformer_ngm_48k_c_tcn_8gpu as launcher
from scripts import run_daphnet_nbm300_c_vs_raw_ablation as pair
from scripts.run_daphnet_transformer_ngm_48k_fold import (
    ARCHITECTURE_NAME,
    PARAMETER_COUNT,
    PatchTransformerNGM48K,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compact_transformer_shapes_and_exact_parameter_count() -> None:
    model = PatchTransformerNGM48K(dropout=0.10)
    x = torch.randn(2, 9, 128)
    patches = model.patchify(x)
    folded = model.fold_patches(patches)
    z = model.encode(x)
    output = model(x)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 48_208
    assert patches.shape == (2, 16, 72)
    assert z.shape == (2, 16)
    assert output.shape == x.shape
    assert torch.equal(folded, x)


def test_architecture_contract_is_skip_free_global_z16() -> None:
    architecture = PatchTransformerNGM48K(dropout=0.10).architecture_config()
    assert architecture["name"] == ARCHITECTURE_NAME
    assert architecture["input_shape"] == ["B", 9, 128]
    assert architecture["bottleneck_shape"] == ["B", 16]
    assert architecture["encoder"] == {
        "layers": 2,
        "d_model": 40,
        "heads": 4,
        "ffn": 80,
        "activation": "GELU",
        "dropout": 0.10,
        "normalization": "post-norm",
    }
    assert architecture["decoder"]["layers"] == 1
    assert architecture["encoder_decoder_skip_connections"] is False
    assert architecture["cross_attention"] is False
    assert architecture["teacher_forcing"] is False
    assert architecture["raw_input_bypass"] is False


def test_pair_worker_accepts_only_exact_transformer_48k_contract() -> None:
    architecture = PatchTransformerNGM48K(dropout=0.10).architecture_config()
    frozen = {
        "training": {
            "maximum_epochs": 300,
            "patience": 20,
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
            "seed": 52,
            "architecture": architecture,
            "augmentation": {
                "clean_probability": 0.40,
                "gaussian_probability": 0.40,
                "mask_probability": 0.20,
                "gaussian_std": 0.04,
                "mask_minimum_samples": 4,
                "mask_maximum_samples": 8,
                "mask_all_channels": True,
            },
        },
        "best_checkpoint_restored_before_calibration": True,
        "validation_mask_or_noise": False,
    }
    args = argparse.Namespace(
        required_nbm_max_epochs=300,
        required_nbm_patience=20,
        nbm_seed=52,
        nbm_kind="transformer_48k",
    )
    result = pair.validate_nbm_contract(frozen, args)
    assert result["all_checks_passed"] is True
    frozen["training"]["architecture"] = {**architecture, "parameter_count": 1}
    try:
        pair.validate_nbm_contract(frozen, args)
    except AssertionError:
        pass
    else:
        raise AssertionError("modified architecture must be rejected")


def test_launcher_grid_and_frozen_hyperparameters() -> None:
    args = argparse.Namespace(
        python=sys.executable,
        data_dir=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
        output_root=REPO_ROOT / "outputs" / "dry_transformer_48k",
        reuse_nbm_source_root=None,
        seeds=launcher.SEED_TEXT,
        num_workers=0,
        overwrite=False,
    )
    nbm_jobs = [
        launcher.nbm_command(args, fold, seed)
        for fold in launcher.FOLDS
        for seed in launcher.REQUIRED_SEEDS
    ]
    train_jobs = [
        launcher.pair_command(args, "train", fold, seed)
        for fold in launcher.FOLDS
        for seed in launcher.REQUIRED_SEEDS
    ]
    assert len(nbm_jobs) == 15
    assert len(train_jobs) == 15
    assert all("--nbm-max-epochs" in command and "300" in command for command in nbm_jobs)
    assert all("--nbm-kind" in command and "transformer_48k" in command for command in train_jobs)
    assert all("--experiment-methods" in command and "FULL_C" in command for command in train_jobs)
    assert all("--tcn-max-epochs" in command and "5" in command for command in train_jobs)


def test_launcher_dry_run_uses_eight_gpus_and_global_barrier_grid() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "launch_daphnet_processed_nbm_transformer_ngm_48k_c_tcn_8gpu.py"
            ),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["gpu_ids"] == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert plan["ngm_jobs"] == 15
    assert plan["classifier_train_jobs"] == 15
    assert plan["post_barrier_test_jobs"] == 15
    assert plan["methods"] == ["FULL_C"]
    assert plan["ngm_parameter_count"] == 48_208
