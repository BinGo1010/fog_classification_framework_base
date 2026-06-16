from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ECAGate(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2)
        return x * self.sigmoid(y)


class MaxPool1dAdaptive(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool_size = int(pool_size)
        self.pool = nn.MaxPool1d(self.pool_size, stride=self.pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.size(-1)
        remainder = n_samples % self.pool_size
        if remainder:
            pad_size = self.pool_size - remainder
            left = pad_size // 2
            right = pad_size - left
            x = F.pad(x, (left, right))
        return self.pool(x)


class SleePyCoFogBackbone(nn.Module):
    """SleePyCoLightV2-style 1D CNN adapted from single-channel EEG to IMU."""

    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 128,
        num_scales: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_scales < 1 or num_scales > 3:
            raise ValueError("num_scales must be 1, 2, or 3.")
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)
        self.num_scales = int(num_scales)
        self.out_dim = self.feature_dim * self.num_scales

        self.init_layer = self._make_stage(self.in_channels, 24, 1, first=True)
        self.layer1 = self._make_stage(24, 48, 2, pool=5)
        self.layer2 = self._make_stage(48, 72, 2, pool=5)
        self.layer3 = self._make_stage(72, 96, 3, pool=5)
        self.layer4 = self._make_stage(96, 128, 2, pool=5)

        self.conv_c5 = nn.Conv1d(128, self.feature_dim, kernel_size=1)
        if self.num_scales > 1:
            self.conv_c4 = nn.Conv1d(96, self.feature_dim, kernel_size=1)
        if self.num_scales > 2:
            self.conv_c3 = nn.Conv1d(72, self.feature_dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._initialize_weights()

    def _make_stage(
        self,
        in_ch: int,
        out_ch: int,
        n_layers: int,
        pool: int | None = None,
        first: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        if not first and pool is not None:
            layers.append(MaxPool1dAdaptive(pool))
        for _ in range(n_layers):
            layers.append(DWConvBlock(in_ch, out_ch))
            in_ch = out_ch
        layers.append(ECAGate(out_ch))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.init_layer(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = [self.conv_c5(c5)]
        if self.num_scales > 1:
            out.append(self.conv_c4(c4))
        if self.num_scales > 2:
            out.append(self.conv_c3(c3))
        return out

    def encode_epoch(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [
            F.adaptive_avg_pool1d(feature, 1).squeeze(-1)
            for feature in self.forward_pyramid(x)
        ]
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_epoch(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SleePyCoFogCRL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = SleePyCoFogBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            num_scales=1,
            dropout=dropout,
        )
        self.projection_head = ProjectionHead(self.backbone.out_dim, projection_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_epoch(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(self.encode(x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class AttentionBiGRUHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        pool: str = "attn",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pool = pool
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        if pool == "attn":
            self.attn_proj = nn.Linear(out_dim, out_dim)
            self.attn_score = nn.Linear(out_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        if self.pool == "attn":
            states = torch.tanh(self.attn_proj(out))
            alpha = torch.softmax(self.attn_score(states), dim=1)
            feat = torch.sum(alpha * out, dim=1)
        elif self.pool == "center":
            feat = out[:, out.size(1) // 2]
        elif self.pool == "last":
            feat = out[:, -1]
        elif self.pool == "mean":
            feat = out.mean(dim=1)
        else:
            raise ValueError(f"Unsupported GRU pool: {self.pool}")
        return self.head(feat)


class Seq2SeqGRUHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.size(-1) != x.size(-1):
            out = out[..., : x.size(-1)]
        return self.norm(out + x)


class Seq2SeqTCNHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TCNResidualBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(levels)
            ]
        )
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).transpose(1, 2)


class SleePyCoFogSequenceClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        baseline: str = "seq2one_gru",
        feature_dim: int = 128,
        num_scales: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        gru_pool: str = "attn",
        tcn_levels: int = 3,
    ):
        super().__init__()
        self.baseline = baseline
        self.backbone = SleePyCoFogBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            num_scales=num_scales,
            dropout=dropout,
        )
        if baseline == "seq2one_gru":
            self.sequence_head = AttentionBiGRUHead(
                input_dim=self.backbone.out_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                pool=gru_pool,
            )
        elif baseline == "seq2seq_gru":
            self.sequence_head = Seq2SeqGRUHead(
                input_dim=self.backbone.out_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif baseline == "seq2seq_tcn":
            self.sequence_head = Seq2SeqTCNHead(
                input_dim=self.backbone.out_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                levels=tcn_levels,
                dropout=dropout,
            )
        else:
            raise ValueError(
                "baseline must be one of: seq2one_gru, seq2seq_gru, seq2seq_tcn."
            )

    @property
    def is_seq2seq(self) -> bool:
        return self.baseline.startswith("seq2seq")

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, samples = x.shape
        flat = x.reshape(batch * seq_len, channels, samples)
        feat = self.backbone.encode_epoch(flat)
        return feat.reshape(batch, seq_len, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encode_sequence(x)
        return self.sequence_head(feat)
