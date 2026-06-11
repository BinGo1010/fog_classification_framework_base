"""
Vit.py

A lightweight Vision Transformer-style model adapted for time-series classification.

This file follows the common Time-Series-Library model interface:
    class Model(nn.Module)
    forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)

Expected input for classification:
    x_enc: [B, L, C]  where L = seq_len, C = enc_in

It also accepts [B, C, L] automatically and transposes it internally.

Recommended for single-IMU FOG classification:
    seq_len = 120
    enc_in = 6
    patch_len = 16
    stride = 8
    d_model = 64
    e_layers = 2
    n_heads = 2
    d_ff = 128
    dropout = 0.2

Reference idea:
    Vision Transformer splits input into fixed-size patches, linearly embeds them,
    adds positional embeddings, prepends a classification token, and feeds the token
    sequence into a Transformer encoder for classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesPatchEmbedding(nn.Module):
    """
    Patch embedding for multivariate time series.

    Input:
        x: [B, L, C]

    Output:
        tokens: [B, N, D]
            N = number of temporal patches
            D = d_model
    """

    def __init__(self, seq_len, enc_in, d_model, patch_len=16, stride=8, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        if seq_len < patch_len:
            raise ValueError(f"seq_len ({seq_len}) must be >= patch_len ({patch_len}).")

        self.num_patches = math.floor((seq_len - patch_len) / stride) + 1

        # A Conv1d with kernel_size=stride=patch operation is equivalent to
        # extracting fixed-size temporal patches and linearly projecting them.
        # Input after transpose: [B, C, L]
        # Output: [B, D, N]
        self.proj = nn.Conv1d(
            in_channels=enc_in,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2).contiguous()   # [B, C, L]
        x = self.proj(x)                     # [B, D, N]
        x = x.transpose(1, 2).contiguous()   # [B, N, D]
        x = self.dropout(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, d_model, num_class, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_class)

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)


class Model(nn.Module):
    """
    ViT-style temporal patch Transformer for time-series classification.

    Compatible with Time-Series-Library style configs and forward signature.

    Required configs for classification:
        configs.task_name    = 'classification'
        configs.seq_len      = input window length, e.g. 120
        configs.enc_in       = input channel number, e.g. 6
        configs.num_class    = number of classes

    Optional configs:
        configs.patch_len    default 16
        configs.stride       default patch_len // 2
        configs.d_model      default 64
        configs.n_heads      default 2
        configs.e_layers     default 2
        configs.d_ff         default 128
        configs.dropout      default 0.2
        configs.activation   default 'gelu'
        configs.pooling      default 'cls', optional 'mean'
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = getattr(configs, "task_name", "classification")
        self.seq_len = int(getattr(configs, "seq_len"))
        self.enc_in = int(getattr(configs, "enc_in"))
        self.num_class = int(getattr(configs, "num_class"))

        self.d_model = int(getattr(configs, "d_model", 64))
        self.n_heads = int(getattr(configs, "n_heads", 2))
        self.e_layers = int(getattr(configs, "e_layers", 2))
        self.d_ff = int(getattr(configs, "d_ff", 128))
        self.dropout_rate = float(getattr(configs, "dropout", 0.2))
        self.activation = getattr(configs, "activation", "gelu")
        self.pooling = getattr(configs, "pooling", "cls")

        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.stride = int(getattr(configs, "stride", max(1, self.patch_len // 2)))

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )

        self.patch_embedding = TimeSeriesPatchEmbedding(
            seq_len=self.seq_len,
            enc_in=self.enc_in,
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout_rate,
        )

        num_patches = self.patch_embedding.num_patches

        # ViT classification token and learnable positional embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.d_model))
        self.pos_dropout = nn.Dropout(self.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.e_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.head = MLPHead(
            d_model=self.d_model,
            num_class=self.num_class,
            dropout=self.dropout_rate,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)

    def _to_blc(self, x):
        """
        Convert input to [B, L, C].

        Time-Series-Library usually uses [B, L, C].
        Some custom IMU pipelines use [B, C, L].
        This helper accepts both.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, L, C] or [B, C, L], got shape {x.shape}.")

        if x.shape[-1] == self.enc_in:
            return x
        if x.shape[1] == self.enc_in:
            return x.transpose(1, 2).contiguous()

        raise ValueError(
            f"Cannot infer channel dimension from input shape {x.shape}; "
            f"expected enc_in={self.enc_in} in dimension 1 or -1."
        )

    def _resize_pos_embed(self, pos_embed, target_len):
        """
        If the number of patches changes at runtime, interpolate positional embeddings.
        This is useful when seq_len changes slightly during experiments.
        """
        if pos_embed.shape[1] == target_len:
            return pos_embed

        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        patch_pos = patch_pos.transpose(1, 2)  # [1, D, N]
        patch_pos = F.interpolate(
            patch_pos,
            size=target_len - 1,
            mode="linear",
            align_corners=False,
        )
        patch_pos = patch_pos.transpose(1, 2)  # [1, N_new, D]
        return torch.cat([cls_pos, patch_pos], dim=1)

    def classification(self, x_enc, x_mark_enc=None):
        """
        Classification forward.

        x_enc:
            [B, L, C] or [B, C, L]
        x_mark_enc:
            Optional padding mask in Time-Series-Library style.
            This ViT version uses CLS token pooling by default, so x_mark_enc is not required.
        """
        x_enc = self._to_blc(x_enc)  # [B, L, C]

        tokens = self.patch_embedding(x_enc)  # [B, N, D]
        B, N, D = tokens.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, N+1, D]

        pos_embed = self._resize_pos_embed(self.pos_embed, tokens.shape[1])
        tokens = tokens + pos_embed
        tokens = self.pos_dropout(tokens)

        encoded = self.encoder(tokens)  # [B, N+1, D]

        if self.pooling == "mean":
            feat = encoded[:, 1:, :].mean(dim=1)
        else:
            feat = encoded[:, 0, :]

        logits = self.head(feat)  # [B, num_class]
        return logits

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)

        raise NotImplementedError(
            "This Vit.py implementation is designed for classification. "
            "Set configs.task_name='classification'."
        )


if __name__ == "__main__":
    class Config:
        task_name = "classification"
        seq_len = 120
        enc_in = 6
        num_class = 3
        patch_len = 16
        stride = 8
        d_model = 64
        n_heads = 2
        e_layers = 2
        d_ff = 128
        dropout = 0.2
        activation = "gelu"
        pooling = "cls"

    model = Model(Config())

    # Time-Series-Library style input: [B, L, C]
    x_blc = torch.randn(8, 120, 6)
    y = model(x_blc, None, None, None)
    print("[B, L, C] output:", y.shape)

    # Custom IMU style input: [B, C, L]
    x_bcl = torch.randn(8, 6, 120)
    y = model(x_bcl, None, None, None)
    print("[B, C, L] output:", y.shape)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", params)
