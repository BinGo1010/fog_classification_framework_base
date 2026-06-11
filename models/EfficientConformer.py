import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GroupedSelfAttention(nn.Module):
    """
    Grouped attention inspired by Efficient Conformer.

    The feature dimension is split into groups, and each group applies its own
    multi-head self-attention. This reduces attention cost compared with full
    attention over the whole channel dimension.
    """

    def __init__(self, d_model, num_heads, groups=2, dropout=0.1):
        super().__init__()
        if groups < 1:
            groups = 1
        if d_model % groups != 0:
            groups = 1
        self.groups = groups
        group_dim = d_model // groups

        # Choose heads per group safely.
        heads_per_group = max(1, num_heads // groups)
        while group_dim % heads_per_group != 0 and heads_per_group > 1:
            heads_per_group -= 1

        self.norm = nn.LayerNorm(d_model)
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=group_dim,
                    num_heads=heads_per_group,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(groups)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        x_norm = self.norm(x)
        chunks = torch.chunk(x_norm, self.groups, dim=-1)
        outs = []
        for chunk, attn in zip(chunks, self.attn_layers):
            out, _ = attn(chunk, chunk, chunk, need_weights=False)
            outs.append(out)
        out = torch.cat(outs, dim=-1)
        out = self.out_proj(out)
        return self.dropout(out)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size should be odd for same padding.")
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
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x


class EfficientConformerBlock(nn.Module):
    """
    Efficient Conformer style block:
        1/2 FFN -> grouped MHSA -> convolution -> 1/2 FFN -> LayerNorm.
    """

    def __init__(self, d_model, num_heads, d_ff, conv_kernel_size=15, attn_groups=2, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.grouped_attn = GroupedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            groups=attn_groups,
            dropout=dropout,
        )
        self.conv = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.grouped_attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class TemporalDownsample(nn.Module):
    """Progressive downsampling along the temporal dimension."""

    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, T, D]
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    """
    Efficient-Conformer-style model for time-series classification.

    It adapts the official Efficient Conformer ideas to short IMU sequences:
        progressive downsampling + grouped attention + Conformer blocks.

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

        base_dim = getattr(configs, "d_model", 64)
        n_heads = getattr(configs, "n_heads", 2)
        dropout = getattr(configs, "dropout", 0.2)
        conv_kernel_size = getattr(configs, "conv_kernel_size", 15)
        attn_groups = getattr(configs, "attn_groups", 2)
        d_ff_ratio = getattr(configs, "d_ff_ratio", 2)

        # Compact stage setting for [B, 6, 120].
        # Can be overridden by setting configs.eff_dims or configs.eff_stage_depths.
        dims = getattr(configs, "eff_dims", None)
        if dims is None:
            dims = [base_dim, base_dim * 2, base_dim * 2]
        depths = getattr(configs, "eff_stage_depths", None)
        if depths is None:
            depths = [1, 1, 1]

        self.input_projection = nn.Linear(self.enc_in, dims[0])
        self.input_dropout = nn.Dropout(dropout)

        stages = []
        for i, dim in enumerate(dims):
            if i > 0:
                stages.append(TemporalDownsample(dims[i - 1], dim, stride=2))
            for _ in range(depths[i]):
                stages.append(
                    EfficientConformerBlock(
                        d_model=dim,
                        num_heads=n_heads,
                        d_ff=dim * d_ff_ratio,
                        conv_kernel_size=conv_kernel_size,
                        attn_groups=attn_groups,
                        dropout=dropout,
                    )
                )
        self.encoder = nn.Sequential(*stages)

        self.act = F.gelu
        self.head_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dims[-1], self.num_class)

    def _to_blc(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x_enc with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] == self.enc_in and x.shape[2] != self.enc_in:
            x = x.transpose(1, 2)
        return x

    def forward_features(self, x_enc):
        x_enc = self._to_blc(x_enc)
        x = self.input_projection(x_enc)
        x = self.input_dropout(x)
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
