from __future__ import annotations

import torch
import torch.nn as nn

from layers import ConvBNAct1D
from .registry import register_model


class _KernelBranch(nn.Module):
    def __init__(self, in_channels, branch_dim, kernel_size, dropout=0.2):
        super().__init__()
        branch_dim = int(branch_dim)
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {kernel_size}.")
        self.net = nn.Sequential(
            ConvBNAct1D(in_channels, branch_dim, kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBNAct1D(branch_dim, branch_dim, kernel_size=kernel_size),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        return self.net(x)


@register_model("MultiKernelCNN")
class MultiKernelCNNClassifier(nn.Module):
    """Two-kernel CNN over one raw window.

    The same input window is sent to a small-kernel branch and a large-kernel
    branch. This keeps the data interface unchanged while testing whether the
    model can learn short-term and longer-context cues through receptive fields.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        hidden_dim=128,
        branch_dim=64,
        small_kernel_size=5,
        large_kernel_size=31,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        small_kernel_size = int(small_kernel_size)
        large_kernel_size = int(large_kernel_size)
        if large_kernel_size <= small_kernel_size:
            raise ValueError(
                "MultiKernelCNN expects large_kernel_size > small_kernel_size, "
                f"got {large_kernel_size} <= {small_kernel_size}."
            )

        self.in_channels = int(in_channels)
        self.seq_len = None if seq_len is None else int(seq_len)
        self.small_kernel_size = small_kernel_size
        self.large_kernel_size = large_kernel_size
        self.small_branch = _KernelBranch(
            self.in_channels,
            branch_dim,
            small_kernel_size,
            dropout=dropout,
        )
        self.large_branch = _KernelBranch(
            self.in_channels,
            branch_dim,
            large_kernel_size,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(int(branch_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"MultiKernelCNN expects [B, C, T] input, got shape {tuple(x.shape)}.")
        feat = torch.cat([self.small_branch(x), self.large_branch(x)], dim=1)
        return self.head(feat)
