from __future__ import annotations

import torch.nn as nn


class ConvBNAct1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        activation=True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNFeatureExtractor1D(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct1D(in_channels, 64, kernel_size=7),
            nn.MaxPool1d(2),
            ConvBNAct1D(64, hidden_dim, kernel_size=5),
            nn.MaxPool1d(2),
            ConvBNAct1D(hidden_dim, hidden_dim, kernel_size=3),
        )

    def forward(self, x):
        return self.net(x)


class DSConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Inception_Block_V1(nn.Module):
    """2D Inception block used by TimesNet."""

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.kernels = nn.ModuleList()
        for i in range(num_kernels):
            kernel_size = 2 * i + 1
            padding = i
            self.kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outputs = [kernel(x) for kernel in self.kernels]
        return sum(outputs) / len(outputs)
