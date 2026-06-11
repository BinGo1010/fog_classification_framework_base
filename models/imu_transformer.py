from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class IMUTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels=6,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_channels, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_channels = d_model

    def forward_tokens(self, x):
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.position(x)
        return self.encoder(x)

    def forward(self, x):
        x = self.forward_tokens(x)
        return self.norm(x.mean(dim=1))


@register_model("IMUTransformer")
class IMUTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=512,
        **kwargs,
    ):
        super().__init__()
        self.encoder = IMUTransformerEncoder(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.classifier = ClassificationHead(self.encoder.out_channels, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.encode(x))


class _ContrastiveIMUTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=512,
        projection_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.encoder = IMUTransformerEncoder(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        feature_dim = self.encoder.out_channels
        self.projection_head = ProjectionHead(feature_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(feature_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        h = self.encode(x)
        return F.normalize(self.projection_head(h), dim=1)

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


@register_model("SupConIMUTransformer")
class SupConIMUTransformer(_ContrastiveIMUTransformer):
    pass


@register_model("SimCLRIMUTransformer")
class SimCLRIMUTransformer(_ContrastiveIMUTransformer):
    pass
