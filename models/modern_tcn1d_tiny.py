from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


class PerVariableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        batch, variables, channels, length = x.shape
        x = x.reshape(batch * variables, channels, length)
        x = self.conv(x)
        _, channels, length = x.shape
        return x.reshape(batch, variables, channels, length)


class PatchStem1D(nn.Module):
    def __init__(self, patch_dim=4, kernel_size=8, stride=4, padding=2):
        super().__init__()
        self.proj = nn.Conv1d(
            1,
            patch_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(patch_dim)
        self.act = nn.GELU()

    def forward(self, x):
        batch, variables, length = x.shape
        x = x.reshape(batch * variables, 1, length)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        _, channels, length = x.shape
        return x.reshape(batch, variables, channels, length)


class ModernTCNBlock1D(nn.Module):
    def __init__(self, dim, large_kernel=7, small_kernel=3, dropout=0.1, expansion=2):
        super().__init__()
        self.large_kernel = nn.Conv1d(
            dim,
            dim,
            kernel_size=large_kernel,
            padding=large_kernel // 2,
            groups=dim,
            bias=False,
        )
        self.small_kernel = nn.Conv1d(
            dim,
            dim,
            kernel_size=small_kernel,
            padding=small_kernel // 2,
            groups=dim,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim * expansion, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * expansion, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, variables, channels, length = x.shape
        y = x.reshape(batch * variables, channels, length)
        y = self.large_kernel(y) + self.small_kernel(y)
        y = self.norm(y)
        y = self.ffn(y)
        y = y.reshape(batch, variables, channels, length)
        return x + y


class ModernTCNStage1D(nn.Module):
    def __init__(self, dim, large_kernel, small_kernel, block_num=1, dropout=0.1, expansion=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ModernTCNBlock1D(
                    dim=dim,
                    large_kernel=large_kernel,
                    small_kernel=small_kernel,
                    dropout=dropout,
                    expansion=expansion,
                )
                for _ in range(int(block_num))
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class ModernTCN1DTinyEncoder(nn.Module):
    def __init__(
        self,
        in_channels=6,
        patch_dim=4,
        patch_kernel=8,
        patch_stride=4,
        patch_padding=2,
        dims=(4, 8, 16),
        block_nums=(1, 1, 1),
        large_kernels=(7, 7, 5),
        small_kernel=3,
        dropout=0.1,
        expansion=2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        d1, d2, d3 = [int(dim) for dim in dims]
        self.patch_stem = PatchStem1D(
            patch_dim=patch_dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
        )
        self.stage1 = ModernTCNStage1D(
            d1,
            large_kernel=large_kernels[0],
            small_kernel=small_kernel,
            block_num=block_nums[0],
            dropout=dropout,
            expansion=expansion,
        )
        self.downsample1 = nn.Sequential(
            PerVariableConv1D(d1, d2, kernel_size=2, stride=2, bias=False),
        )
        self.stage2 = ModernTCNStage1D(
            d2,
            large_kernel=large_kernels[1],
            small_kernel=small_kernel,
            block_num=block_nums[1],
            dropout=dropout,
            expansion=expansion,
        )
        self.downsample2 = nn.Sequential(
            PerVariableConv1D(d2, d3, kernel_size=2, stride=2, padding=1, bias=False),
        )
        self.stage3 = ModernTCNStage1D(
            d3,
            large_kernel=large_kernels[2],
            small_kernel=small_kernel,
            block_num=block_nums[2],
            dropout=dropout,
            expansion=expansion,
        )
        self.out_channels = d3

    def forward_features(self, x):
        x = self.patch_stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        return self.stage3(x)

    def forward(self, x):
        x = self.forward_features(x)
        return x.mean(dim=1).mean(dim=-1)


@register_model("ModernTCN1DTiny")
class ModernTCN1DTiny(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        patch_dim=4,
        patch_kernel=8,
        patch_stride=4,
        patch_padding=2,
        dims=(4, 8, 16),
        block_nums=(1, 1, 1),
        large_kernels=(7, 7, 5),
        small_kernel=3,
        dropout=0.1,
        expansion=2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = ModernTCN1DTinyEncoder(
            in_channels=in_channels,
            patch_dim=patch_dim,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            dims=tuple(dims),
            block_nums=tuple(block_nums),
            large_kernels=tuple(large_kernels),
            small_kernel=small_kernel,
            dropout=dropout,
            expansion=expansion,
        )
        self.classifier = ClassificationHead(self.encoder.out_channels, num_classes)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.encode(x))


class _ContrastiveModernTCN1DTiny(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        seq_len=None,
        patch_dim=4,
        patch_kernel=8,
        patch_stride=4,
        patch_padding=2,
        dims=(4, 8, 16),
        block_nums=(1, 1, 1),
        large_kernels=(7, 7, 5),
        small_kernel=3,
        dropout=0.1,
        expansion=2,
        projection_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.encoder = ModernTCN1DTinyEncoder(
            in_channels=in_channels,
            patch_dim=patch_dim,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            dims=tuple(dims),
            block_nums=tuple(block_nums),
            large_kernels=tuple(large_kernels),
            small_kernel=small_kernel,
            dropout=dropout,
            expansion=expansion,
        )
        feature_dim = self.encoder.out_channels
        self.projection_head = ProjectionHead(feature_dim, projection_dim=projection_dim, activation="gelu")
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


@register_model("SupConModernTCN1DTiny")
class SupConModernTCN1DTiny(_ContrastiveModernTCN1DTiny):
    pass


@register_model("SimCLRModernTCN1DTiny")
class SimCLRModernTCN1DTiny(_ContrastiveModernTCN1DTiny):
    pass
