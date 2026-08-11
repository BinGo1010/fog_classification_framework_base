from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import scripts.run_daphnet_nbm300_c_vs_raw_ablation as ablation
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import RobustScaler
from scripts.run_daphnet_tcn_v2_nbm300_fold import (
    ARCHITECTURE_NAME,
    GlobalBottleneckTCNNBM,
)


def test_tcn_v2_shapes_parameter_count_and_information_bottleneck() -> None:
    model = GlobalBottleneckTCNNBM().eval()
    x = torch.zeros(2, 9, 128)
    with torch.no_grad():
        z = model.encode(x)
        reconstruction = model(x)
    assert z.shape == (2, 16)
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 186_065

    config = model.architecture_config()
    assert config["name"] == ARCHITECTURE_NAME
    assert config["input_shape"] == ["B", 9, 128]
    assert config["bottleneck_shape"] == ["B", 16]
    assert config["output_shape"] == ["B", 9, 128]
    assert config["decoder_conditioning"]["time_code_trainable"] is False
    assert (
        config["decoder_conditioning"]["raw_or_encoder_temporal_connection"]
        is False
    )
    assert config["encoder_decoder_skip_connections"] is False
    assert config["teacher_forcing"] is False
    assert "time_code" not in dict(model.named_parameters())
    assert "time_code" in dict(model.named_buffers())


def test_tcn_v2_state_roundtrip_is_exact(tmp_path) -> None:
    torch.manual_seed(7)
    source = GlobalBottleneckTCNNBM().eval()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        expected = source(x)
    checkpoint = tmp_path / "tcn_v2.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": source.architecture_config(),
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = GlobalBottleneckTCNNBM().eval()
    restored.load_state_dict(payload["model_state"], strict=True)
    with torch.no_grad():
        actual = restored(x)
    assert payload["architecture"] == restored.architecture_config()
    assert torch.equal(actual, expected)


def _write_frozen_tcn_v2(tmp_path, *, sigma: list[float] | None = None) -> None:
    fold_dir = tmp_path / "fold_0"
    checkpoint = fold_dir / "checkpoints" / "tcn_v2_nbm_best.pt"
    checkpoint.parent.mkdir(parents=True)
    model = GlobalBottleneckTCNNBM().eval()
    architecture = model.architecture_config()
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "epoch": 4,
            "validation_huber": 0.25,
            "seed": 52,
            "augmentation": {
                "clean_probability": 0.4,
                "gaussian_probability": 0.4,
                "mask_probability": 0.2,
                "gaussian_std": 0.04,
                "mask_minimum_samples": 4,
                "mask_maximum_samples": 8,
                "mask_all_channels": True,
            },
        },
        checkpoint,
    )
    frozen = {
        "scaler": {
            "median": [0.0] * 9,
            "iqr": [1.0] * 9,
            "epsilon": 1e-6,
        },
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": 128,
        "training": {
            "architecture": architecture,
            "seed": 52,
            "best_epoch": 4,
            "best_validation_huber": 0.25,
            "augmentation": {
                "clean_probability": 0.4,
                "gaussian_probability": 0.4,
                "mask_probability": 0.2,
                "gaussian_std": 0.04,
                "mask_minimum_samples": 4,
                "mask_maximum_samples": 8,
                "mask_all_channels": True,
            },
        },
        "calibration": {
            "bias": [0.0] * 9,
            "sigma": sigma if sigma is not None else [1.0] * 9,
        },
    }
    (fold_dir / "nbm_frozen.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )


def test_tcn_v2_frozen_loader_roundtrip(tmp_path) -> None:
    _write_frozen_tcn_v2(tmp_path)
    loaded, _, bias, sigma, manifest = ablation.load_frozen_tcn_v2_nbm(
        tmp_path, 0, torch.device("cpu")
    )
    assert isinstance(loaded, GlobalBottleneckTCNNBM)
    assert bias.shape == sigma.shape == (9,)
    assert manifest["architecture"]["parameter_count"] == 186_065
    assert manifest["best_epoch"] == 4


def test_tcn_v2_frozen_loader_rejects_nonfinite_calibration(tmp_path) -> None:
    _write_frozen_tcn_v2(tmp_path, sigma=[float("nan")] + [1.0] * 8)
    with pytest.raises(AssertionError, match="calibration"):
        ablation.load_frozen_tcn_v2_nbm(tmp_path, 0, torch.device("cpu"))


def test_tcn_v2_branch_generates_scheme_c_27_channels(monkeypatch) -> None:
    model = GlobalBottleneckTCNNBM().eval()
    scaler = RobustScaler(
        median=np.zeros(9, dtype=np.float32),
        iqr=np.ones(9, dtype=np.float32),
    )
    monkeypatch.setattr(
        ablation,
        "load_frozen_tcn_v2_nbm",
        lambda _source, _fold, _device: (
            model,
            scaler,
            np.full(9, 99.0, dtype=np.float32),
            np.ones(9, dtype=np.float32),
            {"checkpoint": "test"},
        ),
    )
    raw = np.zeros((2, 128, 9), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int8)
    features, metadata = ablation.make_features(
        "FULL_C",
        scaler,
        raw,
        labels,
        torch.device("cpu"),
        None,
        0,
        "tcn_v2",
        128,
    )
    assert features.shape == (2, 128, 27)
    assert np.isfinite(features).all()
    residual = features[:, :, :9]
    assert float(np.max(np.abs(residual.mean(axis=1)))) < 1e-5
    assert metadata["nbm_kind"] == "tcn_v2"
    assert metadata["uses_sigma"] is True
    assert metadata["uses_bias_b"] is False
