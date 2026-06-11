import torch.nn as nn
from layers import GlobalAvgPoolClassifier, TemporalBlock
from .registry import register_model


@register_model("TCNClassifier")
class TCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128, dropout=0.2, levels=4, kernel_size=3, **kwargs):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(levels):
            layers.append(TemporalBlock(ch, hidden_dim, kernel_size, dilation=2**i, dropout=dropout))
            ch = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = GlobalAvgPoolClassifier(hidden_dim, num_classes, dropout=dropout)

    def forward(self, x):
        return self.head(self.backbone(x))
