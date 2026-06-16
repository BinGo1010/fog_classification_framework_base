import torch.nn as nn

from .registry import register_model


@register_model("MLPClassifier")
class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        input_dim = int(in_channels) * int(seq_len)
        hidden_dim = int(hidden_dim)
        num_layers = max(1, int(num_layers))

        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))
