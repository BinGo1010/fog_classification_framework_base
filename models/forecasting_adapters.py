from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ClassificationHead, ProjectionHead

from .registry import register_model


def _make_config(
    in_channels,
    num_classes,
    seq_len=None,
    pred_len=0,
    d_model=128,
    nhead=4,
    n_heads=None,
    num_layers=3,
    e_layers=None,
    d_layers=1,
    dim_feedforward=512,
    d_ff=None,
    dropout=0.2,
    factor=5,
    embed="timeF",
    freq="s",
    activation="gelu",
    distil=True,
    label_len=None,
    moving_avg=25,
    top_k=3,
    num_kernels=6,
    p_hidden_dims=None,
    p_hidden_layers=2,
    **kwargs,
):
    seq_len = int(seq_len or kwargs.get("window_size", 120))
    d_ff = int(d_ff if d_ff is not None else dim_feedforward)
    e_layers = int(e_layers if e_layers is not None else num_layers)
    n_heads = int(n_heads if n_heads is not None else nhead)
    label_len = int(label_len if label_len is not None else max(1, seq_len // 2))
    p_hidden_dims = p_hidden_dims or [d_model, d_model]
    return SimpleNamespace(
        task_name="classification",
        seq_len=seq_len,
        label_len=label_len,
        pred_len=int(pred_len or 0),
        enc_in=int(in_channels),
        dec_in=int(kwargs.get("dec_in", in_channels)),
        c_out=int(kwargs.get("c_out", num_classes)),
        num_class=int(num_classes),
        d_model=int(d_model),
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=int(d_layers),
        d_ff=d_ff,
        dropout=float(dropout),
        factor=int(factor),
        embed=embed,
        freq=freq,
        activation=activation,
        distil=bool(distil),
        moving_avg=int(moving_avg),
        top_k=int(top_k),
        num_kernels=int(num_kernels),
        p_hidden_dims=list(p_hidden_dims),
        p_hidden_layers=int(p_hidden_layers),
    )


class ForecastingFeatureEncoder(nn.Module):
    """Feature encoder for forecasting-library classification backbones.

    Source models use [B, T, C]. This encoder accepts [B, C, T] and returns the
    flattened feature vector that the source classifier normally feeds into its
    final projection layer.
    """

    source_module: str

    def __init__(self, source_module, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.source_module = source_module
        self.configs = _make_config(
            in_channels=in_channels,
            num_classes=num_classes,
            seq_len=seq_len,
            **kwargs,
        )
        source = import_module(source_module, package=__package__)
        self.backbone = source.Model(self.configs)
        self.feature_dim = int(self.backbone.projection.in_features)

    def _prepare(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, C, T], got {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()
        if x.size(1) != self.configs.seq_len:
            raise ValueError(
                f"{self.__class__.__name__} was built for seq_len={self.configs.seq_len}, "
                f"but received T={x.size(1)}. Set model.seq_len or regenerate window data."
            )
        x_mark = torch.ones(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        return x, x_mark

    def _encode_itransformer(self, x):
        enc_out = self.backbone.enc_embedding(x, None)
        enc_out, _ = self.backbone.encoder(enc_out, attn_mask=None)
        output = self.backbone.act(enc_out)
        output = self.backbone.dropout(output)
        return output.reshape(output.shape[0], -1)

    def _encode_timesnet(self, x, x_mark):
        enc_out = self.backbone.enc_embedding(x, None)
        for i in range(self.backbone.layer):
            enc_out = self.backbone.layer_norm(self.backbone.model[i](enc_out))
        output = self.backbone.act(enc_out)
        output = self.backbone.dropout(output)
        output = output * x_mark.unsqueeze(-1)
        return output.reshape(output.shape[0], -1)

    def _encode_nonstationary(self, x, x_mark):
        x_raw = x.clone().detach()
        mean_enc = x.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(torch.var(x - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        tau = self.backbone.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.backbone.delta_learner(x_raw, mean_enc)
        enc_out = self.backbone.enc_embedding(x, None)
        enc_out, _ = self.backbone.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        output = self.backbone.act(enc_out)
        output = self.backbone.dropout(output)
        output = output * x_mark.unsqueeze(-1)
        return output.reshape(output.shape[0], -1)

    def _encode_encoder_only(self, x, x_mark):
        enc_out = self.backbone.enc_embedding(x, None)
        enc_out, _ = self.backbone.encoder(enc_out, attn_mask=None)
        output = self.backbone.act(enc_out)
        output = self.backbone.dropout(output)
        output = output * x_mark.unsqueeze(-1)
        return output.reshape(output.shape[0], -1)

    def forward(self, x):
        x, x_mark = self._prepare(x)
        if self.source_module.endswith("iTransformer"):
            return self._encode_itransformer(x)
        if self.source_module.endswith("TimesNet"):
            return self._encode_timesnet(x, x_mark)
        if self.source_module.endswith("Nonstationary_Transformer"):
            return self._encode_nonstationary(x, x_mark)
        return self._encode_encoder_only(x, x_mark)


class ForecastingClassificationAdapter(nn.Module):
    source_module: str

    def __init__(self, in_channels, num_classes, seq_len=None, **kwargs):
        super().__init__()
        self.encoder = ForecastingFeatureEncoder(self.source_module, in_channels, num_classes, seq_len, **kwargs)
        self.classification_head = self.encoder.backbone.projection

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classification_head(self.encode(x))


class ForecastingContrastiveAdapter(nn.Module):
    source_module: str

    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len=None,
        projection_dim=64,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = ForecastingFeatureEncoder(self.source_module, in_channels, num_classes, seq_len, **kwargs)
        self.projection_head = ProjectionHead(self.encoder.feature_dim, projection_dim=projection_dim)
        self.classification_head = ClassificationHead(self.encoder.feature_dim, num_classes, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return F.normalize(self.projection_head(self.encode(x)), dim=1)

    def classify_features(self, h):
        return self.classification_head(h)

    def forward(self, x):
        return self.classify_features(self.encode(x))

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


@register_model("iTransformer")
class ITransformerClassifier(ForecastingClassificationAdapter):
    source_module = ".iTransformer"


@register_model("TimesNet")
class TimesNetClassifier(ForecastingClassificationAdapter):
    source_module = ".TimesNet"


@register_model("NonstationaryTransformer")
class NonstationaryTransformerClassifier(ForecastingClassificationAdapter):
    source_module = ".Nonstationary_Transformer"


@register_model("Nonstationary_Transformer")
class NonstationaryTransformerUnderscoreClassifier(NonstationaryTransformerClassifier):
    pass


@register_model("Non-stationary Transformer")
class NonstationaryTransformerDisplayNameClassifier(NonstationaryTransformerClassifier):
    pass


@register_model("Informer")
class InformerClassifier(ForecastingClassificationAdapter):
    source_module = ".Informer"


@register_model("Autoformer")
class AutoformerClassifier(ForecastingClassificationAdapter):
    source_module = ".Autoformer"


@register_model("SupConITransformer")
class SupConITransformer(ForecastingContrastiveAdapter):
    source_module = ".iTransformer"


@register_model("SimCLRITransformer")
class SimCLRITransformer(ForecastingContrastiveAdapter):
    source_module = ".iTransformer"


@register_model("SupConTimesNet")
class SupConTimesNet(ForecastingContrastiveAdapter):
    source_module = ".TimesNet"


@register_model("SimCLRTimesNet")
class SimCLRTimesNet(ForecastingContrastiveAdapter):
    source_module = ".TimesNet"


@register_model("SupConNonstationaryTransformer")
class SupConNonstationaryTransformer(ForecastingContrastiveAdapter):
    source_module = ".Nonstationary_Transformer"


@register_model("SimCLRNonstationaryTransformer")
class SimCLRNonstationaryTransformer(ForecastingContrastiveAdapter):
    source_module = ".Nonstationary_Transformer"


@register_model("SupConInformer")
class SupConInformer(ForecastingContrastiveAdapter):
    source_module = ".Informer"


@register_model("SimCLRInformer")
class SimCLRInformer(ForecastingContrastiveAdapter):
    source_module = ".Informer"


@register_model("SupConAutoformer")
class SupConAutoformer(ForecastingContrastiveAdapter):
    source_module = ".Autoformer"


@register_model("SimCLRAutoformer")
class SimCLRAutoformer(ForecastingContrastiveAdapter):
    source_module = ".Autoformer"
