"""Neural raw-IMU baselines."""

from __future__ import annotations

import torch
from torch import nn


class CNNGRUClassifier(nn.Module):
    """Compact 1-D CNN front-end followed by a temporal GRU readout."""

    def __init__(
        self,
        in_channels: int,
        cnn_channels: tuple[int, ...] = (32, 64),
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not cnn_channels or min(cnn_channels) <= 0:
            raise ValueError("cnn_channels must contain positive values")
        if gru_hidden <= 0 or gru_layers <= 0:
            raise ValueError("GRU dimensions must be positive")
        blocks: list[nn.Module] = []
        previous = int(in_channels)
        for index, channels in enumerate(cnn_channels):
            kernel_size = 7 if index == 0 else 5
            blocks.extend(
                [
                    nn.Conv1d(
                        previous,
                        int(channels),
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(int(channels)),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(channels)
        self.cnn = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=previous,
            hidden_size=int(gru_hidden),
            num_layers=int(gru_layers),
            batch_first=True,
            dropout=float(dropout) if int(gru_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        directions = 2 if bidirectional else 1
        gru_width = int(gru_hidden) * directions
        self.head = nn.Sequential(
            nn.Linear(2 * gru_width, int(gru_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gru_hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).transpose(1, 2)
        sequence, _ = self.gru(features)
        pooled = torch.cat(
            [sequence.mean(dim=1), sequence.amax(dim=1)],
            dim=1,
        )
        return self.head(pooled).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
