from __future__ import annotations

import torch


def sensor_cost_loss(selection_probs, sensor_costs=None):
    if sensor_costs is None:
        return torch.mean(selection_probs)
    costs = torch.as_tensor(sensor_costs, dtype=selection_probs.dtype, device=selection_probs.device)
    return torch.sum(selection_probs * costs) / torch.clamp(torch.sum(costs), min=1e-12)


def sparsity_loss(selection_probs):
    return torch.mean(torch.abs(selection_probs))
