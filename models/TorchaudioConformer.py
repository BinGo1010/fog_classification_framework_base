import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="silu"):
        super().__init__()
        act = nn.SiLU() if activation in ["silu", "swish"] else nn.GELU()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module in the style of torchaudio Conformer:
        LayerNorm -> pointwise conv + GLU -> depthwise conv -> BN -> SiLU -> pointwise conv.
    """

    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("depthwise_conv_kernel_size should be odd for same padding.")

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)       # [B, D, T]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)       # [B, T, D]
        return x


class ConformerBlock(nn.Module):
    """
    Standard Conformer encoder block:
        1/2 FFN -> MHSA -> Conv -> 1/2 FFN -> LayerNorm
    """

    def __init__(self, d_model, n_heads, d_ff, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout, activation="silu")
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout, activation="silu")
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)

        attn_in = self.self_attn_layer_norm(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.self_attn_dropout(attn_out)

        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


class Model(nn.Module):
    """
    Torchaudio-Conformer-style model for time-series classification.

    It follows the public torchaudio Conformer parameterization conceptually:
        input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, dropout.

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
        conv_kernel_size = getattr(configs, "conv_kernel_size", 15)

        self.input_projection = nn.Linear(self.enc_in, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            *[
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )

        self.act = F.gelu
        self.head_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, self.num_class)

    def _to_blc(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x_enc with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] == self.enc_in and x.shape[2] != self.enc_in:
            x = x.transpose(1, 2)
        return x

    def forward_features(self, x_enc):
        x_enc = self._to_blc(x_enc)  # [B, L, C]
        B, L, _ = x_enc.shape
        x = self.input_projection(x_enc)
        x = x + self.pos_embedding[:, :L, :]
        x = self.dropout(x)
        x = self.encoder(x)
        feat = x.mean(dim=1)
        return feat

    def classification(self, x_enc, x_mark_enc=None):
        feat = self.forward_features(x_enc)
        feat = self.act(feat)
        feat = self.head_dropout(feat)
        out = self.projection(feat)
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return self.classification(x_enc, x_mark_enc)
