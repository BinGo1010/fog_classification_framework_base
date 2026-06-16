from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


def _make_ts_config(
    in_channels,
    num_classes,
    seq_len=None,
    pred_len=0,
    d_model=128,
    nhead=4,
    n_heads=None,
    num_layers=3,
    e_layers=None,
    dim_feedforward=512,
    d_ff=None,
    dropout=0.2,
    factor=5,
    embed="timeF",
    freq="s",
    activation="gelu",
    moving_avg=25,
    patch_len=16,
    stride=8,
    chunk_size=24,
    **kwargs,
):
    seq_len = int(seq_len or kwargs.get("window_size", 120))
    return SimpleNamespace(
        task_name="classification",
        seq_len=seq_len,
        pred_len=int(pred_len or 0),
        enc_in=int(in_channels),
        dec_in=int(kwargs.get("dec_in", in_channels)),
        c_out=int(kwargs.get("c_out", num_classes)),
        num_class=int(num_classes),
        d_model=int(d_model),
        n_heads=int(n_heads if n_heads is not None else nhead),
        e_layers=int(e_layers if e_layers is not None else num_layers),
        d_ff=int(d_ff if d_ff is not None else dim_feedforward),
        dropout=float(dropout),
        factor=int(factor),
        embed=embed,
        freq=freq,
        activation=activation,
        moving_avg=int(moving_avg),
        patch_len=int(patch_len),
        stride=int(stride),
        chunk_size=int(chunk_size),
    )


class SourceFeatureEncoder(nn.Module):
    def __init__(self, source_module, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.source_module = source_module
        self.configs = _make_ts_config(in_channels, num_classes, seq_len, **kwargs)
        source = import_module(source_module, package=__package__)
        if source_module.endswith("PatchTST"):
            self.backbone = source.Model(self.configs, patch_len=self.configs.patch_len, stride=self.configs.stride)
        elif source_module.endswith("LightTS"):
            self.backbone = source.Model(self.configs, chunk_size=self.configs.chunk_size)
        else:
            self.backbone = source.Model(self.configs)
        self.feature_dim = int(self.backbone.projection.in_features)

    def _prepare(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, C, T], got {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()
        if x.size(1) != self.configs.seq_len:
            raise ValueError(f"Expected seq_len={self.configs.seq_len}, got {x.size(1)}")
        return x

    def _encode_dlinear(self, x):
        enc_out = self.backbone.encoder(x)
        return enc_out.reshape(enc_out.shape[0], -1)

    def _encode_lightts(self, x):
        enc_out = self.backbone.encoder(x)
        return enc_out.reshape(enc_out.shape[0], -1)

    def _encode_patchtst(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.backbone.patch_embedding(x)
        enc_out, _ = self.backbone.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        output = self.backbone.flatten(enc_out)
        output = self.backbone.dropout(output)
        return output.reshape(output.shape[0], -1)

    def forward(self, x):
        x = self._prepare(x)
        if self.source_module.endswith("DLinear"):
            return self._encode_dlinear(x)
        if self.source_module.endswith("PatchTST"):
            return self._encode_patchtst(x)
        if self.source_module.endswith("LightTS"):
            return self._encode_lightts(x)
        raise ValueError(f"Unsupported source module: {self.source_module}")


class SourceClassificationAdapter(nn.Module):
    source_module: str

    def __init__(self, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.encoder = SourceFeatureEncoder(self.source_module, in_channels, num_classes, seq_len, **kwargs)
        self.classification_head = self.encoder.backbone.projection

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classification_head(self.encode(x))


class SourceContrastiveAdapter(nn.Module):
    source_module: str

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        projection_dim=64,
        projection_hidden_dim=128,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = SourceFeatureEncoder(self.source_module, in_channels, num_classes, seq_len, **kwargs)
        self.projection_head = ProjectionHead(
            self.encoder.feature_dim,
            projection_dim=projection_dim,
            hidden_features=int(projection_hidden_dim),
        )
        self.classification_head = ClassificationHead(self.encoder.feature_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return F.normalize(self.projection_head(self.encode(x)), dim=1)

    def forward(self, x):
        return self.classification_head(self.encode(x))

    def encoder_feature_parameters(self):
        for name, param in self.encoder.named_parameters():
            if name.startswith("backbone.projection."):
                continue
            yield param

    def contrastive_parameters(self):
        yield from self.encoder_feature_parameters()
        yield from self.projection_head.parameters()

    def classifier_parameters(self, train_encoder=True):
        if train_encoder:
            yield from self.encoder_feature_parameters()
        yield from self.classification_head.parameters()


class FITS1DLiteBackbone(nn.Module):
    def __init__(self, in_channels, num_classes, seq_len, cutoff_ratio=0.35, hidden_dim=64, dropout=0.2, **kwargs):
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.num_classes = int(num_classes)
        self.num_freq = self.seq_len // 2 + 1
        self.keep_freq = max(2, int(self.num_freq * float(cutoff_ratio)))
        self.real_weight = nn.Parameter(torch.ones(self.in_channels, self.keep_freq))
        self.imag_weight = nn.Parameter(torch.zeros(self.in_channels, self.keep_freq))
        self.feature_dim = self.in_channels * self.keep_freq * 2
        self.norm = nn.LayerNorm(self.feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def encode(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            freq = torch.fft.rfft(x.float(), dim=-1)
            freq = freq[..., : self.keep_freq]
            weight = torch.complex(self.real_weight, self.imag_weight).unsqueeze(0)
            freq = freq * weight
            feat = torch.cat([freq.real, freq.imag], dim=-1)
        return self.norm(feat.flatten(1).to(dtype=x.dtype))

    def forward(self, x):
        return self.projection(self.encode(x))


class FITS1DLiteContrastive(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        projection_dim=64,
        projection_hidden_dim=128,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        seq_len = int(seq_len or kwargs.get("window_size", 120))
        self.encoder = FITS1DLiteBackbone(in_channels, num_classes, seq_len, dropout=dropout, **kwargs)
        self.projection_head = ProjectionHead(
            self.encoder.feature_dim,
            projection_dim=projection_dim,
            hidden_features=int(projection_hidden_dim),
        )
        self.classification_head = ClassificationHead(self.encoder.feature_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder.encode(x)

    def project(self, x):
        return F.normalize(self.projection_head(self.encode(x)), dim=1)

    def forward(self, x):
        return self.classification_head(self.encode(x))

    def contrastive_parameters(self):
        yield from self.encoder.parameters()
        yield from self.projection_head.parameters()

    def classifier_parameters(self, train_encoder=True):
        if train_encoder:
            yield from self.encoder.parameters()
        yield from self.classification_head.parameters()


@register_model("DLinear")
class DLinearClassifier(SourceClassificationAdapter):
    source_module = ".DLinear"


@register_model("SupConDLinear")
class SupConDLinear(SourceContrastiveAdapter):
    source_module = ".DLinear"


@register_model("SimCLRDLinear")
class SimCLRDLinear(SourceContrastiveAdapter):
    source_module = ".DLinear"


@register_model("PatchTSTSmall")
class PatchTSTSmallClassifier(SourceClassificationAdapter):
    source_module = ".PatchTST"


@register_model("SupConPatchTSTSmall")
class SupConPatchTSTSmall(SourceContrastiveAdapter):
    source_module = ".PatchTST"


@register_model("SimCLRPatchTSTSmall")
class SimCLRPatchTSTSmall(SourceContrastiveAdapter):
    source_module = ".PatchTST"


@register_model("LightTSStudent")
class LightTSStudentClassifier(SourceClassificationAdapter):
    source_module = ".LightTS"


@register_model("SupConLightTSStudent")
class SupConLightTSStudent(SourceContrastiveAdapter):
    source_module = ".LightTS"


@register_model("SimCLRLightTSStudent")
class SimCLRLightTSStudent(SourceContrastiveAdapter):
    source_module = ".LightTS"


@register_model("FITS1DLite")
class FITS1DLite(FITS1DLiteBackbone):
    def __init__(self, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__(in_channels, num_classes, int(seq_len or kwargs.get("window_size", 120)), **kwargs)


@register_model("SupConFITS1DLite")
class SupConFITS1DLite(FITS1DLiteContrastive):
    pass


@register_model("SimCLRFITS1DLite")
class SimCLRFITS1DLite(FITS1DLiteContrastive):
    pass
