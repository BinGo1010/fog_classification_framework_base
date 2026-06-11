import math
import torch
import torch.nn as nn
from .registry import register_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


@register_model("TransformerClassifier")
class TransformerClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128, dropout=0.2, nhead=4, num_layers=3, **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,T,C]
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.head(x.mean(dim=1))
