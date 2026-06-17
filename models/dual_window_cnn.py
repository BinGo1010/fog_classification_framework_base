from __future__ import annotations

import torch
import torch.nn as nn

from layers import ConvBNAct1D
from .registry import register_model


class _KernelCNNBranch(nn.Module):
    def __init__(self, in_channels, branch_dim, kernel_size, dropout=0.2):
        super().__init__()
        branch_dim = int(branch_dim)
        kernel_size = int(kernel_size)
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


class _KernelCNNSequenceBranch(nn.Module):
    def __init__(self, in_channels, branch_dim, kernel_size, dropout=0.2):
        super().__init__()
        branch_dim = int(branch_dim)
        kernel_size = int(kernel_size)
        self.net = nn.Sequential(
            ConvBNAct1D(in_channels, branch_dim, kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBNAct1D(branch_dim, branch_dim, kernel_size=kernel_size),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        return self.net(x)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _DualRawWindowMixin:
    def _init_raw_window_shape(self, in_channels, raw_in_channels, short_seq_len, long_seq_len, seq_len):
        in_channels = int(in_channels)
        if in_channels % 2 != 0:
            raise ValueError(f"{self.__class__.__name__} expects an even in_channels value, got {in_channels}.")
        inferred_raw = in_channels // 2
        if raw_in_channels is None or int(raw_in_channels) * 2 != in_channels:
            raw_in_channels = inferred_raw
        self.raw_in_channels = int(raw_in_channels)
        self.short_seq_len = int(short_seq_len) if short_seq_len is not None else None
        self.long_seq_len = int(long_seq_len or seq_len) if (long_seq_len is not None or seq_len is not None) else None

    def _split_raw_windows(self, x):
        raw = self.raw_in_channels
        if x.size(1) < raw * 2:
            raise ValueError(f"Expected at least {raw * 2} channels, got {x.size(1)}.")
        x_short = x[:, :raw, :]
        x_long = x[:, raw : raw * 2, :]
        if self.short_seq_len is not None and x_short.size(-1) > self.short_seq_len:
            x_short = x_short[..., -self.short_seq_len :]
        if self.long_seq_len is not None and x_long.size(-1) > self.long_seq_len:
            x_long = x_long[..., -self.long_seq_len :]
        return x_short, x_long


@register_model("DualWindowCNN")
class DualWindowKernelCNNClassifier(_DualRawWindowMixin, nn.Module):
    """Two-branch CNN for packed raw short/long windows.

    Expected input is [B, 2 * raw_channels, long_seq_len]:
      - first raw_channels: short window, right-aligned inside long_seq_len
      - second raw_channels: long window
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        raw_in_channels=None,
        short_seq_len=None,
        long_seq_len=None,
        hidden_dim=128,
        branch_dim=64,
        short_kernel_size=5,
        long_kernel_size=31,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self._init_raw_window_shape(in_channels, raw_in_channels, short_seq_len, long_seq_len, seq_len)

        self.short_branch = _KernelCNNBranch(
            self.raw_in_channels,
            branch_dim,
            short_kernel_size,
            dropout=dropout,
        )
        self.long_branch = _KernelCNNBranch(
            self.raw_in_channels,
            branch_dim,
            long_kernel_size,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(int(branch_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x):
        x_short, x_long = self._split_raw_windows(x)
        feat = torch.cat([self.short_branch(x_short), self.long_branch(x_long)], dim=1)
        return self.head(feat)


@register_model("DualWindowCNNGRU")
class DualWindowCNNGRUClassifier(_DualRawWindowMixin, nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        raw_in_channels=None,
        short_seq_len=None,
        long_seq_len=None,
        hidden_dim=128,
        branch_dim=64,
        gru_hidden_dim=None,
        num_layers=1,
        short_kernel_size=5,
        long_kernel_size=31,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self._init_raw_window_shape(in_channels, raw_in_channels, short_seq_len, long_seq_len, seq_len)
        branch_dim = int(branch_dim)
        gru_hidden_dim = int(gru_hidden_dim or branch_dim)
        num_layers = int(num_layers)
        self.short_cnn = _KernelCNNSequenceBranch(self.raw_in_channels, branch_dim, short_kernel_size, dropout=dropout)
        self.long_cnn = _KernelCNNSequenceBranch(self.raw_in_channels, branch_dim, long_kernel_size, dropout=dropout)
        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.short_gru = nn.GRU(
            input_size=branch_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True,
        )
        self.long_gru = nn.GRU(
            input_size=branch_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden_dim * 4, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    @staticmethod
    def _last_bidir_state(h_n):
        num_layers_times_directions, batch, hidden = h_n.shape
        h_n = h_n.view(num_layers_times_directions // 2, 2, batch, hidden)
        last = h_n[-1]
        return torch.cat([last[0], last[1]], dim=1)

    def _encode_branch(self, x, cnn, gru):
        seq = cnn(x).transpose(1, 2)
        _, h_n = gru(seq)
        return self._last_bidir_state(h_n)

    def forward(self, x):
        x_short, x_long = self._split_raw_windows(x)
        feat = torch.cat(
            [
                self._encode_branch(x_short, self.short_cnn, self.short_gru),
                self._encode_branch(x_long, self.long_cnn, self.long_gru),
            ],
            dim=1,
        )
        return self.head(feat)


@register_model("DualWindowCNNTransformer")
class DualWindowCNNTransformerClassifier(_DualRawWindowMixin, nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        raw_in_channels=None,
        short_seq_len=None,
        long_seq_len=None,
        hidden_dim=128,
        branch_dim=64,
        nhead=4,
        num_layers=2,
        short_kernel_size=5,
        long_kernel_size=31,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self._init_raw_window_shape(in_channels, raw_in_channels, short_seq_len, long_seq_len, seq_len)
        branch_dim = int(branch_dim)
        nhead = int(nhead)
        num_layers = int(num_layers)
        self.short_cnn = _KernelCNNSequenceBranch(self.raw_in_channels, branch_dim, short_kernel_size, dropout=dropout)
        self.long_cnn = _KernelCNNSequenceBranch(self.raw_in_channels, branch_dim, long_kernel_size, dropout=dropout)
        self.short_pos = _PositionalEncoding(branch_dim)
        self.long_pos = _PositionalEncoding(branch_dim)
        short_layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=nhead,
            dim_feedforward=max(int(hidden_dim), branch_dim * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        long_layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=nhead,
            dim_feedforward=max(int(hidden_dim), branch_dim * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.short_transformer = nn.TransformerEncoder(short_layer, num_layers=num_layers)
        self.long_transformer = nn.TransformerEncoder(long_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(branch_dim * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _encode_branch(self, x, cnn, pos, transformer):
        seq = cnn(x).transpose(1, 2)
        seq = transformer(pos(seq))
        return seq.mean(dim=1)

    def forward(self, x):
        x_short, x_long = self._split_raw_windows(x)
        feat = torch.cat(
            [
                self._encode_branch(x_short, self.short_cnn, self.short_pos, self.short_transformer),
                self._encode_branch(x_long, self.long_cnn, self.long_pos, self.long_transformer),
            ],
            dim=1,
        )
        return self.head(feat)
