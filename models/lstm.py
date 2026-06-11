import torch
import torch.nn as nn
from .registry import register_model


@register_model("LSTMClassifier")
class LSTMClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128, dropout=0.2, num_layers=2, bidirectional=True, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,C,T] -> [B,T,C]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
