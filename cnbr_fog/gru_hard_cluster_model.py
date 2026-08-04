"""Explicit frozen-cluster conditioning for the S01 GRU mean ablation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .gru_convergence_models import DirectMeanDecoder, GRUMeanModelBase


class HardClusterConditionedGRUMeanForecaster(GRUMeanModelBase):
    """Condition a shared GRU state on an externally frozen context cluster.

    The external labels must be computed from context only by a train-fitted,
    frozen assignment model.  A zero-initialized cluster embedding preserves
    the global direct model's initial prediction while allowing each cluster
    to learn an additive state offset.  This is the closest FoG analogue of a
    device embedding without pretending that KMeans clusters are identities.
    """

    model_name = "gru_hard_cluster_conditioned_mean"
    uses_external_mode = True

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        n_clusters: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels,
            horizon,
            hidden_channels,
            num_layers,
            dropout,
            **kwargs,
        )
        if int(n_clusters) < 2:
            raise ValueError("n_clusters must be at least two")
        self.n_clusters = int(n_clusters)
        # Keep construction order identical to the global direct arm through
        # its decoder so all common weights match under the same seed.
        self.mean_decoder = DirectMeanDecoder(
            self.hidden_channels, self.in_channels, self.horizon
        )
        self.cluster_embedding = nn.Embedding(
            self.n_clusters, self.hidden_channels
        )
        nn.init.zeros_(self.cluster_embedding.weight)

    def forward_mean(
        self, context: torch.Tensor, cluster: torch.Tensor
    ) -> torch.Tensor:
        self._check_context(context)
        if cluster.ndim != 1 or cluster.shape[0] != context.shape[0]:
            raise ValueError("cluster must have shape [batch]")
        if cluster.dtype != torch.long:
            raise TypeError("cluster labels must be torch.long")
        if bool(torch.any(cluster < 0)) or bool(torch.any(cluster >= self.n_clusters)):
            raise ValueError("cluster label is outside the configured range")
        state = self.encoder(context) + self.cluster_embedding(cluster)
        last = context[:, :, -1]
        delta = self.mean_decoder(state, last)
        return last.unsqueeze(-1) + delta

    def forward(
        self, context: torch.Tensor, cluster: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward_mean(context, cluster)
        return self._distribution(mean, self._unit_log_sigma)

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "decoder": self.mean_decoder.model_config(),
            "conditioning": {
                "type": "external_frozen_context_cluster_embedding",
                "n_clusters": self.n_clusters,
                "embedding_dim": self.hidden_channels,
                "initialization": "zeros",
                "uses_target_or_label": False,
            },
            "uses_external_mode": True,
        }


__all__ = ["HardClusterConditionedGRUMeanForecaster"]
