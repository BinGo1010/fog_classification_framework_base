from . import (  # noqa: F401
    bigru,
    cnn1d,
    ds_cnn1d,
    forecasting_adapters,
    imu_transformer,
    lightweight_imu_tcn,
    lstm,
    mobileone1d_tiny,
    modern_tcn1d_tiny,
    sequence_contrastive,
    simclr_cnn1d,
    supcon_cnn1d,
    tcn,
    transformer,
    vit_adapter,
)
from .registry import build_model, MODEL_REGISTRY


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
