from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.0):
        super().__init__()
        layers = []
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalAvgPoolClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            ClassificationHead(in_channels, num_classes, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)
