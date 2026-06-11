from __future__ import annotations

import torch
import torch.nn as nn


def parse_channel_name(name):
    parts = str(name).split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected channel name like ankleL_acc_x, got {name}")
    return parts[0], parts[1], parts[2]


def sensor_groups_from_columns(sensor_columns):
    groups = {}
    for idx, name in enumerate(sensor_columns):
        position, _, _ = parse_channel_name(name)
        groups.setdefault(position, []).append(idx)
    return {name: indices for name, indices in groups.items()}


def combine_slot_probabilities(slot_probs):
    return 1.0 - torch.prod(1.0 - slot_probs, dim=0)


class BinaryConcreteGate(nn.Module):
    def __init__(self, num_gates, temperature=5.0, hard=False):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_gates))
        nn.init.normal_(self.logits, mean=0.0, std=0.02)
        self.temperature = float(temperature)
        self.hard = bool(hard)

    def set_temperature(self, temperature):
        self.temperature = float(temperature)

    def probabilities(self):
        return torch.sigmoid(self.logits)

    def forward(self, force_all=False, eps=1e-6):
        if force_all:
            return torch.ones_like(self.logits)
        probs = self.probabilities()
        if self.training:
            u = torch.rand_like(self.logits).clamp(eps, 1.0 - eps)
            noise = torch.log(u) - torch.log1p(-u)
            y_soft = torch.sigmoid((self.logits + noise) / self.temperature)
        else:
            y_soft = probs
        if self.hard:
            y_hard = (y_soft >= 0.5).to(y_soft.dtype)
            return y_hard - y_soft.detach() + y_soft
        return y_soft
