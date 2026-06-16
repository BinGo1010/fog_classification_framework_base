from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


def _make_conformer_config(
    in_channels,
    num_classes,
    seq_len=None,
    d_model=64,
    nhead=2,
    n_heads=None,
    num_layers=2,
    e_layers=None,
    dim_feedforward=128,
    d_ff=None,
    dropout=0.2,
    activation="gelu",
    temporal_kernel=25,
    pool_size=4,
    pool_stride=4,
    conv_kernel_size=15,
    attn_groups=2,
    d_ff_ratio=2,
    eff_dims=None,
    eff_stage_depths=None,
    **kwargs,
):
    seq_len = int(seq_len or kwargs.get("window_size", 120))
    n_heads = int(n_heads if n_heads is not None else nhead)
    e_layers = int(e_layers if e_layers is not None else num_layers)
    d_ff = int(d_ff if d_ff is not None else dim_feedforward)
    return SimpleNamespace(
        task_name="classification",
        seq_len=seq_len,
        enc_in=int(in_channels),
        num_class=int(num_classes),
        d_model=int(d_model),
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=float(dropout),
        activation=activation,
        temporal_kernel=int(temporal_kernel),
        pool_size=int(pool_size),
        pool_stride=int(pool_stride),
        conv_kernel_size=int(conv_kernel_size),
        attn_groups=int(attn_groups),
        d_ff_ratio=int(d_ff_ratio),
        eff_dims=eff_dims,
        eff_stage_depths=eff_stage_depths,
    )


class ConformerFeatureEncoder(nn.Module):
    source_module: str

    def __init__(self, source_module, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.configs = _make_conformer_config(in_channels, num_classes, seq_len, **kwargs)
        source = import_module(source_module, package=__package__)
        self.backbone = source.Model(self.configs)
        self.feature_dim = int(self.backbone.projection.in_features)

    def forward(self, x):
        return self.backbone.forward_features(x)


class ConformerClassificationAdapter(nn.Module):
    source_module: str

    def __init__(self, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.encoder = ConformerFeatureEncoder(self.source_module, in_channels, num_classes, seq_len, **kwargs)
        self.classification_head = self.encoder.backbone.projection

    def encode(self, x):
        feat = self.encoder(x)
        feat = self.encoder.backbone.act(feat)
        feat = self.encoder.backbone.head_dropout(feat) if hasattr(self.encoder.backbone, "head_dropout") else self.encoder.backbone.dropout(feat)
        return feat

    def forward(self, x):
        return self.classification_head(self.encode(x))


class ConformerContrastiveAdapter(nn.Module):
    source_module: str

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        projection_dim=64,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = ConformerFeatureEncoder(self.source_module, in_channels, num_classes, seq_len, **kwargs)
        self.projection_head = ProjectionHead(self.encoder.feature_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(self.encoder.feature_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return F.normalize(self.projection_head(self.encode(x)), dim=1)

    def classify_features(self, h):
        return self.classification_head(h)

    def forward(self, x):
        return self.classify_features(self.encode(x))

    def encoder_feature_parameters(self):
        for name, param in self.encoder.named_parameters():
            if name.startswith("backbone.projection."):
                continue
            yield param

    def contrastive_parameters(self):
        yield from self.encoder_feature_parameters()
        yield from self.projection_head.parameters()

    def classifier_parameters(self, train_encoder=True):
        if train_encoder:
            yield from self.encoder_feature_parameters()
        yield from self.classification_head.parameters()


@register_model("EEGConformer")
class EEGConformerRegistered(ConformerClassificationAdapter):
    source_module = ".EEGConformer"


@register_model("TorchaudioConformer")
class TorchaudioConformerRegistered(ConformerClassificationAdapter):
    source_module = ".TorchaudioConformer"


@register_model("EfficientConformer")
class EfficientConformerRegistered(ConformerClassificationAdapter):
    source_module = ".EfficientConformer"


@register_model("SupConEEGConformer")
class SupConEEGConformer(ConformerContrastiveAdapter):
    source_module = ".EEGConformer"


@register_model("SimCLREEGConformer")
class SimCLREEGConformer(ConformerContrastiveAdapter):
    source_module = ".EEGConformer"


@register_model("SupConTorchaudioConformer")
class SupConTorchaudioConformer(ConformerContrastiveAdapter):
    source_module = ".TorchaudioConformer"


@register_model("SimCLRTorchaudioConformer")
class SimCLRTorchaudioConformer(ConformerContrastiveAdapter):
    source_module = ".TorchaudioConformer"


@register_model("SupConEfficientConformer")
class SupConEfficientConformer(ConformerContrastiveAdapter):
    source_module = ".EfficientConformer"


@register_model("SimCLREfficientConformer")
class SimCLREfficientConformer(ConformerContrastiveAdapter):
    source_module = ".EfficientConformer"
