from __future__ import annotations

import torch.nn as nn


class MobileOneBlock1D(nn.Module):
    """Training-time 1D MobileOne-style block."""

    def __init__(self, in_channels, out_channels, stride=1, num_conv_branches=1):
        super().__init__()
        self.dw_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(in_channels),
                )
                for _ in range(num_conv_branches)
            ]
        )
        self.dw_scale = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
        )
        self.dw_identity = nn.BatchNorm1d(in_channels) if stride == 1 else None

        self.pw_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                for _ in range(num_conv_branches)
            ]
        )
        self.pw_identity = nn.BatchNorm1d(in_channels) if in_channels == out_channels else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw_scale(x)
        for branch in self.dw_branches:
            out = out + branch(x)
        if self.dw_identity is not None:
            out = out + self.dw_identity(x)
        out = self.act(out)

        x = out
        out = self.pw_branches[0](x)
        for branch in self.pw_branches[1:]:
            out = out + branch(x)
        if self.pw_identity is not None:
            out = out + self.pw_identity(x)
        return self.act(out)
