from __future__ import annotations

import torch
import torch.nn as nn


class PatchTransformerBiLSTMClassifier(nn.Module):
    """Patch-token sequence classifier inspired by the 1st-place Kaggle FOG model."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 192,
        num_heads: int = 6,
        num_encoder_layers: int = 3,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        roll_pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.seq_len = int(seq_len)
        self.roll_pos_encoding = bool(roll_pos_encoding)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.first_dropout = nn.Dropout(dropout)
        self.pos_encoding = nn.Parameter(torch.empty(1, self.seq_len, d_model))
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        feedforward = int(dim_feedforward or d_model * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_encoder_layers),
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=int(lstm_layers),
            dropout=dropout if int(lstm_layers) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def _position_encoding(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = self.pos_encoding[:, :seq_len, :]
        if not (self.training and self.roll_pos_encoding):
            return pos.expand(batch_size, -1, -1)

        shifts = torch.randint(-seq_len, 1, (batch_size,), device=device)
        return torch.stack(
            [torch.roll(pos[0], shifts=int(shift.item()), dims=0) for shift in shifts],
            dim=0,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.seq_len:
            raise ValueError(f"Input seq_len={seq_len} exceeds configured seq_len={self.seq_len}.")

        x = self.input_proj(x / 25.0)
        x = x + self._position_encoding(batch_size, seq_len, x.device)
        x = self.first_dropout(x)

        padding_mask = None
        lengths = None
        if mask is not None:
            padding_mask = ~mask.bool()
            lengths = mask.sum(dim=1).detach().cpu().clamp_min(1)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=seq_len,
            )
        else:
            x, _ = self.lstm(x)

        return self.head(x)
