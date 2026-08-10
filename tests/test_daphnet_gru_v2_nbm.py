from __future__ import annotations

import json

import numpy as np
import torch

import scripts.run_daphnet_nbm300_c_vs_raw_ablation as ablation
from scripts.run_daphnet_gru_v2_nbm300_fold import (
    ARCHITECTURE_NAME,
    PhaseConditionedGRUNBM,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


def test_gru_v2_shapes_parameter_count_and_information_bottleneck() -> None:
    model = PhaseConditionedGRUNBM().eval()
    x = torch.zeros(2, 128, 9)
    with torch.no_grad():
        z = model.encode(x)
        reconstruction = model(x)
    assert z.shape == (2, 16)
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 172_697

    config = model.architecture_config()
    assert config["name"] == ARCHITECTURE_NAME
    assert config["encoder"] == {
        "type": "bidirectional GRU",
        "layers": 1,
        "hidden_per_direction": 96,
        "dropout": 0.0,
        "summary": "top-layer forward/backward final states [B,192]",
    }
    assert config["bottleneck_shape"] == ["B", 16]
    assert config["decoder"]["layers"] == 2
    assert config["decoder_conditioning"]["trainable"] is False
    assert config["decoder_conditioning"]["raw_or_encoder_token_connection"] is False
    assert config["encoder_decoder_skip_connections"] is False
    assert config["teacher_forcing"] is False


def test_gru_v2_state_roundtrip_is_exact(tmp_path) -> None:
    torch.manual_seed(7)
    source = PhaseConditionedGRUNBM().eval()
    x = torch.randn(2, 128, 9)
    with torch.no_grad():
        expected = source(x)
    checkpoint = tmp_path / "gru_v2.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": source.architecture_config(),
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = PhaseConditionedGRUNBM().eval()
    restored.load_state_dict(payload["model_state"])
    with torch.no_grad():
        actual = restored(x)
    assert payload["architecture"] == restored.architecture_config()
    assert torch.equal(actual, expected)


def test_gru_v2_frozen_loader_roundtrip(tmp_path) -> None:
    fold_dir = tmp_path / "fold_0"
    checkpoint = fold_dir / "checkpoints" / "gru_v2_nbm_best.pt"
    checkpoint.parent.mkdir(parents=True)
    model = PhaseConditionedGRUNBM().eval()
    architecture = model.architecture_config()
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "epoch": 4,
            "validation_huber": 0.25,
            "seed": 0,
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
            "best_epoch": 4,
            "best_validation_huber": 0.25,
        },
        "calibration": {
            "bias": [0.0] * 9,
            "sigma": [1.0] * 9,
        },
    }
    (fold_dir / "nbm_frozen.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )
    loaded, _, bias, sigma, manifest = ablation.load_frozen_gru_v2_nbm(
        tmp_path, 0, torch.device("cpu")
    )
    assert isinstance(loaded, PhaseConditionedGRUNBM)
    assert bias.shape == sigma.shape == (9,)
    assert manifest["architecture"]["parameter_count"] == 172_697


def test_gru_v2_branch_generates_scheme_c_27_channels(monkeypatch) -> None:
    model = PhaseConditionedGRUNBM().eval()
    scaler = RobustScaler(
        median=np.zeros(9, dtype=np.float32),
        iqr=np.ones(9, dtype=np.float32),
    )
    monkeypatch.setattr(
        ablation,
        "load_frozen_gru_v2_nbm",
        lambda _source, _fold, _device: (
            model,
            scaler,
            np.zeros(9, dtype=np.float32),
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
        "gru_v2",
        128,
    )
    assert features.shape == (2, 128, 27)
    assert np.isfinite(features).all()
    residual = features[:, :, :9]
    assert float(np.max(np.abs(residual.mean(axis=1)))) < 1e-5
    assert metadata["nbm_kind"] == "gru_v2"
    assert metadata["uses_sigma"] is True
    assert metadata["uses_bias_b"] is False
