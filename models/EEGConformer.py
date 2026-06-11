import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)

    def forward(self, x):
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class EEGConformerPatchEmbedding(nn.Module):
    """
    EEG-Conformer style convolutional embedding adapted to IMU time series.

    Expected input after conversion:
        x: [B, L, C]
    Internal format:
        [B, 1, C, L]

    It performs:
        temporal convolution -> spatial/channel convolution -> pooling -> tokens.
    """

    def __init__(
        self,
        in_channels,
        d_model,
        temporal_kernel=25,
        pool_size=4,
        pool_stride=4,
        dropout=0.1,
    ):
        super().__init__()
        padding_t = temporal_kernel // 2
        self.embedding = nn.Sequential(
            # Temporal filtering on each IMU channel.
            nn.Conv2d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=(1, temporal_kernel),
                padding=(0, padding_t),
                bias=False,
            ),
            nn.BatchNorm2d(d_model),
            # Spatial/channel filtering across the 6 IMU axes.
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(in_channels, 1),
                groups=d_model,
                bias=False,
            ),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, C, L]
        x = self.embedding(x)               # [B, D, 1, N]
        x = x.squeeze(2).transpose(1, 2)    # [B, N, D]
        return x


class Model(nn.Module):
    """
    EEG-Conformer-style model for time-series classification.

    Reference idea:
        EEG-Conformer uses a compact convolutional Transformer to capture
        local and global features for EEG decoding. This file adapts that
        design to single-IMU classification.

    Input convention follows Time-Series-Library style:
        x_enc: [B, L, C]
    It also tolerates:
        x_enc: [B, C, L]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = getattr(configs, "task_name", "classification")
        self.seq_len = getattr(configs, "seq_len", 120)
        self.enc_in = getattr(configs, "enc_in", 6)
        self.num_class = getattr(configs, "num_class", 3)

        d_model = getattr(configs, "d_model", 64)
        n_heads = getattr(configs, "n_heads", 2)
        e_layers = getattr(configs, "e_layers", 2)
        d_ff = getattr(configs, "d_ff", d_model * 2)
        dropout = getattr(configs, "dropout", 0.2)
        activation = getattr(configs, "activation", "gelu")
        temporal_kernel = getattr(configs, "temporal_kernel", 25)
        pool_size = getattr(configs, "pool_size", 4)
        pool_stride = getattr(configs, "pool_stride", 4)

        self.patch_embedding = EEGConformerPatchEmbedding(
            in_channels=self.enc_in,
            d_model=d_model,
            temporal_kernel=temporal_kernel,
            pool_size=pool_size,
            pool_stride=pool_stride,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, self.num_class)

    def _to_blc(self, x):
        # Prefer [B, L, C]. If given [B, C, L], convert.
        if x.dim() != 3:
            raise ValueError(f"Expected x_enc with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] == self.enc_in and x.shape[2] != self.enc_in:
            x = x.transpose(1, 2)
        return x

    def forward_features(self, x_enc):
        x_enc = self._to_blc(x_enc)
        tokens = self.patch_embedding(x_enc)       # [B, N, D]
        tokens = self.encoder(tokens)              # [B, N, D]
        feat = tokens.mean(dim=1)                  # [B, D]
        return feat

    def classification(self, x_enc, x_mark_enc=None):
        feat = self.forward_features(x_enc)
        feat = self.act(feat)
        feat = self.dropout(feat)
        out = self.projection(feat)
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        # This implementation is intended for classification baselines.
        return self.classification(x_enc, x_mark_enc)
