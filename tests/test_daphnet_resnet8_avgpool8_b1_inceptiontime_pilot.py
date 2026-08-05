from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot as pilot


def test_encoder_pool_embedding_and_reconstruction_shapes() -> None:
    model = pilot.ResNet8AvgPool8NBM()
    inputs = torch.randn(3, 9, 128)
    features = model.encode_features(inputs)
    pooled = model.pool(features)
    reconstruction, embedding = model(inputs)
    assert features.shape == (3, 48, 32)
    assert pooled.shape == (3, 48, 8)
    assert embedding.shape == (3, 512)
    assert reconstruction.shape == inputs.shape
    assert torch.isfinite(reconstruction).all()


def test_decoder_dimension_path() -> None:
    model = pilot.ResNet8AvgPool8NBM()
    embedding = torch.randn(2, 512)
    seed = model.decoder_expansion(embedding).reshape(-1, 48, 32)
    level1 = model.decoder_up1(seed)
    level2 = model.decoder_up2(level1)
    assert seed.shape == (2, 48, 32)
    assert level1.shape == (2, 32, 64)
    assert level2.shape == (2, 24, 128)
    assert model.output(level2).shape == (2, 9, 128)


def test_architecture_is_a_real_compact_global_bottleneck() -> None:
    model = pilot.ResNet8AvgPool8NBM()
    config = model.architecture_config()
    assert config["pool"] == "AdaptiveAvgPool1d(8)"
    assert config["embedding_shape"] == ["batch", 512]
    assert config["long_skip_connections"] is False
    assert 1_000_000 < config["parameter_count"] < 2_000_000


def test_pipeline_is_b1_only_and_uses_inceptiontime() -> None:
    pilot.configure_pipeline(pilot.DEFAULT_SEED)
    assert pilot.exp.METHODS == ("B1",)
    assert pilot.exp.SEEDS == (pilot.DEFAULT_SEED,)
    assert pilot.exp.a1b.ContextM3 is pilot.ResNet8AvgPool8NBM
    assert pilot.exp.train_nbm is pilot.train_nbm
    assert pilot.exp.train_classifier is pilot.inception.train_classifier
