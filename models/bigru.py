from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        in_channels=6,
        hidden_dim=128,
        embedding_dim=None,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.out_channels = hidden_dim * 2
        if embedding_dim is None:
            self.embedding = nn.Identity()
            self.embedding_dim = self.out_channels
        else:
            self.embedding = nn.Sequential(
                nn.LayerNorm(self.out_channels),
                nn.Dropout(dropout),
                nn.Linear(self.out_channels, embedding_dim),
                nn.ReLU(inplace=True),
            )
            self.embedding_dim = int(embedding_dim)

    def forward_features(self, x):
        x = x.transpose(1, 2)
        _, h_n = self.gru(x)
        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_dim)
        last_layer = h_n[-1]
        return torch.cat([last_layer[0], last_layer[1]], dim=1)

    def forward(self, x):
        return self.embedding(self.forward_features(x))


@register_model("BiGRUClassifier")
class BiGRUClassifier(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = BiGRUEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=None,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_channels),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_channels, num_classes),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encode(x))


class _ContrastiveBiGRU(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        hidden_dim=128,
        embedding_dim=128,
        projection_dim=64,
        num_layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = BiGRUEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        feature_dim = self.encoder.embedding_dim
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


@register_model("SupConBiGRU")
class SupConBiGRU(_ContrastiveBiGRU):
    pass


@register_model("SimCLRBiGRU")
class SimCLRBiGRU(_ContrastiveBiGRU):
    pass
