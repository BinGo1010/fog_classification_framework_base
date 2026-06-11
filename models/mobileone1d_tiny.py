from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ConvBNAct1D, MobileOneBlock1D, ProjectionHead

from .registry import register_model


class MobileOne1DTinyEncoder(nn.Module):
    def __init__(
        self,
        in_channels=6,
        channels=(32, 64, 128, 256),
        num_conv_branches=1,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBNAct1D(in_channels, c1, kernel_size=7, stride=2)
        self.stage1 = nn.Sequential(
            MobileOneBlock1D(c1, c1, stride=1, num_conv_branches=num_conv_branches),
            MobileOneBlock1D(c1, c1, stride=1, num_conv_branches=num_conv_branches),
        )
        self.stage2 = nn.Sequential(
            MobileOneBlock1D(c1, c2, stride=2, num_conv_branches=num_conv_branches),
            MobileOneBlock1D(c2, c2, stride=1, num_conv_branches=num_conv_branches),
        )
        self.stage3 = nn.Sequential(
            MobileOneBlock1D(c2, c3, stride=2, num_conv_branches=num_conv_branches),
            MobileOneBlock1D(c3, c3, stride=1, num_conv_branches=num_conv_branches),
            MobileOneBlock1D(c3, c3, stride=1, num_conv_branches=num_conv_branches),
        )
        self.stage4 = MobileOneBlock1D(c3, c4, stride=2, num_conv_branches=num_conv_branches)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_channels = c4

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.pool(x).flatten(1)


@register_model("MobileOne1DTiny")
class MobileOne1DTiny(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        dropout=0.2,
        channels=(32, 64, 128, 256),
        num_conv_branches=1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = MobileOne1DTinyEncoder(
            in_channels=in_channels,
            channels=tuple(channels),
            num_conv_branches=num_conv_branches,
        )
        self.classifier = ClassificationHead(self.encoder.out_channels, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.encode(x))


class _ContrastiveMobileOne1DTiny(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        dropout=0.2,
        channels=(32, 64, 128, 256),
        num_conv_branches=1,
        projection_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.encoder = MobileOne1DTinyEncoder(
            in_channels=in_channels,
            channels=tuple(channels),
            num_conv_branches=num_conv_branches,
        )
        feature_dim = self.encoder.out_channels
        self.projection_head = ProjectionHead(feature_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(feature_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        h = self.encode(x)
        return F.normalize(self.projection_head(h), dim=1)

    def classify_features(self, h):
        return self.classification_head(h)

    def forward(self, x):
        return self.classify_features(self.encode(x))

    def contrastive_parameters(self):
        yield from self.encoder.parameters()
        yield from self.projection_head.parameters()

    def classifier_parameters(self, train_encoder=True):
        if train_encoder:
            yield from self.encoder.parameters()
        yield from self.classification_head.parameters()


@register_model("SupConMobileOne1DTiny")
class SupConMobileOne1DTiny(_ContrastiveMobileOne1DTiny):
    pass


@register_model("SimCLRMobileOne1DTiny")
class SimCLRMobileOne1DTiny(_ContrastiveMobileOne1DTiny):
    pass
