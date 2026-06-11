from __future__ import annotations

import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_features, projection_dim=64, hidden_features=None, activation="relu"):
        super().__init__()
        hidden_features = hidden_features or in_features
        if activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act,
            nn.Linear(hidden_features, projection_dim),
        )

    def forward(self, x):
        return self.net(x)
