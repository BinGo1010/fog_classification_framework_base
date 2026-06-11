import torch.nn as nn
from layers import CNNFeatureExtractor1D, GlobalAvgPoolClassifier
from .registry import register_model


@register_model("CNN1D")
class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes, seq_len=None, hidden_dim=128, dropout=0.2, **kwargs):
        super().__init__()
        self.features = CNNFeatureExtractor1D(in_channels, hidden_dim=hidden_dim)
        self.classifier = GlobalAvgPoolClassifier(hidden_dim, num_classes, dropout=dropout)

    def forward(self, x):
        return self.classifier(self.features(x))
