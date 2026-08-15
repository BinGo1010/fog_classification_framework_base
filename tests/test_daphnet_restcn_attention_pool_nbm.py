from __future__ import annotations

import json

import numpy as np
import torch

import scripts.run_daphnet_nbm300_c_vs_raw_ablation as ablation
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import RobustScaler
from scripts.run_daphnet_restcn_attention_pool_nbm300_fold import (
    ARCHITECTURE_NAME,
    CHECKPOINT_NAME,
    ResTCNSingleQueryAttentionPoolNBM,
)


def test_attention_pool_shapes_parameters_and_information_barrier() -> None:
    model = ResTCNSingleQueryAttentionPoolNBM().eval()
    x = torch.zeros(2, 9, 128)
    with torch.no_grad():
        tokens = model.encode_tokens(x)
        z, attention = model.encode_with_attention(x)
        reconstruction = model(x)

    assert tokens.shape == (2, 32, 48)
    assert z.shape == (2, 16)
    assert attention.shape == (2, 1, 32)
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 1), atol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) == 171_905

    config = model.architecture_config()
    assert config["name"] == ARCHITECTURE_NAME
    assert config["encoder_token_shape"] == ["B", 32, 48]
    assert config["attention_pool"]["query_shape"] == [1, 1, 48]
    assert config["attention_pool"]["heads"] == 4
    assert config["attention_pool"]["position_code_trainable"] is False
    assert config["attention_pool"]["raw_or_encoder_token_residual_bypass"] is False
    assert config["bottleneck_shape"] == ["B", 16]
    assert config["decoder_conditioning"]["raw_or_encoder_temporal_connection"] is False
    assert config["encoder_decoder_skip_connections"] is False
    assert config["input_output_global_residual"] is False
    assert config["teacher_forcing"] is False

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert "attention_query" in parameters
    assert "encoder_position_code" not in parameters
    assert "time_code" not in parameters
    assert "encoder_position_code" in buffers
    assert "time_code" in buffers


def test_attention_query_encoder_and_decoder_all_receive_gradients() -> None:
    torch.manual_seed(7)
    model = ResTCNSingleQueryAttentionPoolNBM().train()
    output = model(torch.randn(2, 9, 128))
    output.square().mean().backward()
    selected = (
        model.encoder_stem[0].weight,
        model.attention_query,
        model.attention_pool.in_proj_weight,
        model.to_bottleneck[0].weight,
        model.output_head.weight,
    )
    for parameter in selected:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_attention_pool_checkpoint_roundtrip_is_exact(tmp_path) -> None:
    torch.manual_seed(52)
    source = ResTCNSingleQueryAttentionPoolNBM().eval()
    x = torch.randn(2, 9, 128)
    with torch.no_grad():
        expected = source(x)
    checkpoint = tmp_path / CHECKPOINT_NAME
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": source.architecture_config(),
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = ResTCNSingleQueryAttentionPoolNBM().eval()
    restored.load_state_dict(payload["model_state"], strict=True)
    with torch.no_grad():
        actual = restored(x)
    assert payload["architecture"] == restored.architecture_config()
    assert torch.equal(actual, expected)


def _write_frozen_attention_nbm(tmp_path) -> None:
    fold_dir = tmp_path / "fold_0"
    checkpoint = fold_dir / "checkpoints" / CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    model = ResTCNSingleQueryAttentionPoolNBM().eval()
    architecture = model.architecture_config()
    augmentation = {
        "clean_probability": 0.4,
        "gaussian_probability": 0.4,
        "mask_probability": 0.2,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_all_channels": True,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "epoch": 4,
            "validation_huber": 0.25,
            "seed": 52,
            "augmentation": augmentation,
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
            "maximum_epochs": 300,
            "patience": 20,
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
            "augmentation": augmentation,
        },
        "calibration": {
            "bias": [0.0] * 9,
            "sigma": [1.0] * 9,
        },
        "best_checkpoint_restored_before_calibration": True,
        "validation_mask_or_noise": False,
    }
    (fold_dir / "nbm_frozen.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )


def test_frozen_loader_roundtrip_and_scheme_c_branch(tmp_path, monkeypatch) -> None:
    _write_frozen_attention_nbm(tmp_path)
    loaded, scaler, bias, sigma, manifest = (
        ablation.load_frozen_tcn_attention_z16_nbm(
            tmp_path, 0, torch.device("cpu")
        )
    )
    assert isinstance(loaded, ResTCNSingleQueryAttentionPoolNBM)
    assert bias.shape == sigma.shape == (9,)
    assert manifest["architecture"]["parameter_count"] == 171_905

    monkeypatch.setattr(
        ablation,
        "load_frozen_tcn_attention_z16_nbm",
        lambda _source, _fold, _device: (
            loaded,
            scaler,
            np.full(9, 99.0, dtype=np.float32),
            sigma,
            manifest,
        ),
    )
    raw = np.zeros((2, 128, 9), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int8)
    features, metadata = ablation.make_features(
        "FULL_C",
        RobustScaler(
            median=np.zeros(9, dtype=np.float32),
            iqr=np.ones(9, dtype=np.float32),
        ),
        raw,
        labels,
        torch.device("cpu"),
        tmp_path,
        0,
        "tcn_attn_z16",
        128,
    )
    assert features.shape == (2, 128, 27)
    assert np.isfinite(features).all()
    assert float(np.max(np.abs(features[:, :, :9].mean(axis=1)))) < 1e-5
    assert metadata["nbm_kind"] == "tcn_attn_z16"
    assert metadata["uses_sigma"] is True
    assert metadata["uses_bias_b"] is False
