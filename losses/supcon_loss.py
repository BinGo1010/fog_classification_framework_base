from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss for features shaped [B, n_views, D]."""

    def __init__(self, temperature=0.1, base_temperature=0.1, eps=1e-12):
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.eps = float(eps)

    def forward(self, features, labels):
        if features.ndim != 3:
            raise ValueError(f"features must be [B, n_views, D], got {features.shape}")
        batch_size, n_views, _ = features.shape
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("labels length must match features batch size")

        class_mask = torch.eq(labels, labels.T).to(features.device, dtype=features.dtype)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = torch.matmul(contrast, contrast.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        mask = class_mask.repeat(n_views, n_views)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * n_views, device=features.device).view(-1, 1), 0.0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        positive_count = mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_count
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(n_views, batch_size).mean()


__all__ = ["SupConLoss"]
