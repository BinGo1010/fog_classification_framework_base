from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


def _make_vit_config(
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
    patch_len=16,
    stride=None,
    activation="gelu",
    pooling="cls",
    **kwargs,
):
    seq_len = int(seq_len or kwargs.get("window_size", 120))
    n_heads = int(n_heads if n_heads is not None else nhead)
    e_layers = int(e_layers if e_layers is not None else num_layers)
    d_ff = int(d_ff if d_ff is not None else dim_feedforward)
    patch_len = int(patch_len)
    stride = int(stride if stride is not None else max(1, patch_len // 2))
    return SimpleNamespace(
        task_name="classification",
        seq_len=seq_len,
        enc_in=int(in_channels),
        num_class=int(num_classes),
        patch_len=patch_len,
        stride=stride,
        d_model=int(d_model),
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=float(dropout),
        activation=activation,
        pooling=pooling,
    )


class VitFeatureEncoder(nn.Module):
    def __init__(self, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.configs = _make_vit_config(in_channels, num_classes, seq_len, **kwargs)
        source = import_module(".Vit", package=__package__)
        self.backbone = source.Model(self.configs)
        self.feature_dim = int(self.backbone.d_model)

    def forward(self, x):
        x = self.backbone._to_blc(x)
        tokens = self.backbone.patch_embedding(x)
        batch_size = tokens.size(0)
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        pos_embed = self.backbone._resize_pos_embed(self.backbone.pos_embed, tokens.shape[1])
        tokens = self.backbone.pos_dropout(tokens + pos_embed)
        encoded = self.backbone.encoder(tokens)
        if self.backbone.pooling == "mean":
            return encoded[:, 1:, :].mean(dim=1)
        return encoded[:, 0, :]


class VitClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.encoder = VitFeatureEncoder(in_channels, num_classes, seq_len, **kwargs)
        self.classification_head = self.encoder.backbone.head

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classification_head(self.encode(x))


class ContrastiveVitClassifier(nn.Module):
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
        self.encoder = VitFeatureEncoder(in_channels, num_classes, seq_len, **kwargs)
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

    def contrastive_parameters(self):
        yield from self.encoder.parameters()
        yield from self.projection_head.parameters()

    def classifier_parameters(self, train_encoder=True):
        if train_encoder:
            yield from self.encoder.parameters()
        yield from self.classification_head.parameters()


@register_model("Vit")
class VitRegistered(VitClassifier):
    pass


@register_model("ViT")
class ViTRegistered(VitClassifier):
    pass


@register_model("SupConVit")
class SupConVit(ContrastiveVitClassifier):
    pass


@register_model("SupConViT")
class SupConViT(ContrastiveVitClassifier):
    pass


@register_model("SimCLRVit")
class SimCLRVit(ContrastiveVitClassifier):
    pass


@register_model("SimCLRViT")
class SimCLRViT(ContrastiveVitClassifier):
    pass
