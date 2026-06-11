from __future__ import annotations

import torch.nn as nn


class SmallChannelCNN(nn.Module):
    """Compact CNN for a fixed small IMU channel subset."""

    def __init__(self, in_channels, num_classes, hidden_dim=48, dropout=0.2, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
