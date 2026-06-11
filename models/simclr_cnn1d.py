from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


class SimCLRCNN1DEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, embedding_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


@register_model("SimCLRCNN1D")
class SimCLRCNN1D(nn.Module):
    """1D-CNN encoder with SimCLR projection and classification heads."""

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        hidden_dim=128,
        embedding_dim=128,
        projection_dim=64,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = SimCLRCNN1DEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.projection_head = ProjectionHead(embedding_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(embedding_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        h = self.encode(x)
        return F.normalize(self.projection_head(h), dim=1)

    def forward(self, x):
        return self.classification_head(self.encode(x))

    def contrastive_parameters(self):
        yield from self.encoder.parameters()
        yield from self.projection_head.parameters()

    def classifier_parameters(self, train_encoder=True):
        if train_encoder:
            yield from self.encoder.parameters()
        yield from self.classification_head.parameters()
