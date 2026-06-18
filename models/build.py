from . import (  # noqa: F401
    bigru,
    cnn1d,
    conformer_adapters,
    ds_cnn1d,
    dual_window_cnn,
    forecasting_adapters,
    imu_transformer,
    lightweight_imu_tcn,
    lightweight_ts_adapters,
    lstm,
    mlp,
    mobileone1d_tiny,
    modern_tcn1d_tiny,
    multi_kernel_cnn,
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
