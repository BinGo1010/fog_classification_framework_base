from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


@register_model("GRUClassifier")
class GRUClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_dim=128,
        dropout=0.2,
        num_layers=2,
        bidirectional=True,
        **kwargs,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        embedding_dim=128,
        dropout=0.2,
        num_layers=2,
        bidirectional=True,
        rnn_type="lstm",
    ):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _ContrastiveRecurrent(nn.Module):
    rnn_type = "lstm"

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        hidden_dim=128,
        embedding_dim=128,
        projection_dim=64,
        dropout=0.2,
        num_layers=2,
        bidirectional=True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = RecurrentEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_type=self.rnn_type,
        )
        self.projection_head = ProjectionHead(embedding_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(embedding_dim, num_classes, dropout=dropout)

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


@register_model("SupConLSTM")
class SupConLSTM(_ContrastiveRecurrent):
    rnn_type = "lstm"


@register_model("SimCLRLSTM")
class SimCLRLSTM(_ContrastiveRecurrent):
    rnn_type = "lstm"


@register_model("SupConGRU")
class SupConGRU(_ContrastiveRecurrent):
    rnn_type = "gru"


@register_model("SimCLRGRU")
class SimCLRGRU(_ContrastiveRecurrent):
    rnn_type = "gru"
