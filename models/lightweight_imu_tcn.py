from __future__ import annotations

import torch.nn as nn

from layers import DepthwiseSeparableTCNBlock, GlobalAvgPoolClassifier

from .registry import register_model


@register_model("LightweightIMUTCN")
class LightweightIMUTCN(nn.Module):
    """Small TCN for full-IMU or selected-IMU FoG classification."""

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=32,
        levels=4,
        kernel_size=3,
        dropout=0.1,
        dilations=None,
        **kwargs,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(int(levels))]
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                DepthwiseSeparableTCNBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.head = GlobalAvgPoolClassifier(hidden_channels, num_classes, dropout=dropout)

    def forward(self, x):
        x = self.projection(x)
        x = self.blocks(x)
        return self.head(x)
