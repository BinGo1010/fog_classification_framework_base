from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent / InfoNCE loss for two augmented views.

    z1 and z2 are [B, D]. For every sample, the positive pair is the other
    augmented view of the same original window; all other views in the batch
    are negatives.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1, z2):
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}")
        batch_size = z1.size(0)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        features = torch.cat([z1, z2], dim=0)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = torch.arange(2 * batch_size, device=features.device)
        labels = (labels + batch_size) % (2 * batch_size)
        self_mask = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, -1e9)
        return F.cross_entropy(logits, labels)
