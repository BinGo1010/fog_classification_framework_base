from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, DSConvBlock1D, ProjectionHead

from .registry import register_model


class DSCNN1DEncoder(nn.Module):
    def __init__(
        self,
        in_channels=6,
        stem_channels=32,
        channels=(32, 64, 64, 128),
        stem_kernel_size=5,
        ds_kernel_size=3,
    ):
        super().__init__()
        padding = stem_kernel_size // 2
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                stem_channels,
                kernel_size=stem_kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DSConvBlock1D(stem_channels, c1, stride=1, kernel_size=ds_kernel_size),
            DSConvBlock1D(c1, c2, stride=2, kernel_size=ds_kernel_size),
            DSConvBlock1D(c2, c3, stride=1, kernel_size=ds_kernel_size),
            DSConvBlock1D(c3, c4, stride=2, kernel_size=ds_kernel_size),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_channels = c4

    def forward_features(self, x):
        x = self.stem(x)
        return self.blocks(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.pool(x).flatten(1)


@register_model("DSCNN1D")
class DSCNN1D(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        stem_channels=32,
        channels=(32, 64, 64, 128),
        stem_kernel_size=5,
        ds_kernel_size=3,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DSCNN1DEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            channels=tuple(channels),
            stem_kernel_size=stem_kernel_size,
            ds_kernel_size=ds_kernel_size,
        )
        self.classifier = ClassificationHead(self.encoder.out_channels, num_classes)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.encode(x))


class _ContrastiveDSCNN1D(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        stem_channels=32,
        channels=(32, 64, 64, 128),
        stem_kernel_size=5,
        ds_kernel_size=3,
        projection_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DSCNN1DEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            channels=tuple(channels),
            stem_kernel_size=stem_kernel_size,
            ds_kernel_size=ds_kernel_size,
        )
        feature_dim = self.encoder.out_channels
        self.projection_head = ProjectionHead(feature_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(feature_dim, num_classes)

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


@register_model("SupConDSCNN1D")
class SupConDSCNN1D(_ContrastiveDSCNN1D):
    pass


@register_model("SimCLRDSCNN1D")
class SimCLRDSCNN1D(_ContrastiveDSCNN1D):
    pass
