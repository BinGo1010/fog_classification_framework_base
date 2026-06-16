from __future__ import annotations

import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .registry import register_model


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Original TCN-style residual block: WeightNorm Conv1d + Chomp + ReLU + Dropout."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for level, out_channels in enumerate(num_channels):
            dilation_size = 2**level
            in_ch = in_channels if level == 0 else num_channels[level - 1]
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@register_model("TCNClassifier")
class TCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_dim=128,
        dropout=0.2,
        levels=4,
        kernel_size=3,
        num_channels=None,
        pooling="avg",
        **kwargs,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [hidden_dim] * int(levels)
        else:
            num_channels = [int(ch) for ch in num_channels]
        self.backbone = TemporalConvNet(
            in_channels=in_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.pooling = pooling
        out_channels = int(num_channels[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.pooling == "last":
            x = x[:, :, -1]
        elif self.pooling == "avg":
            x = self.pool(x).squeeze(-1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        return self.head(x)
