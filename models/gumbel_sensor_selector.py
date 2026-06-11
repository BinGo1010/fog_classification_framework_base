from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import CNNFeatureExtractor1D, GlobalAvgPoolClassifier
from layers.gumbel_softmax_selector import combine_slot_probabilities, sensor_groups_from_columns


class GumbelSensorSelectorCNN(nn.Module):
    """Differentiable IMU/channel selector followed by a CNN classifier.

    The model receives all channels, learns relaxed Gumbel-Softmax choices over
    IMUs and over channels inside each IMU, then masks the input before the CNN.
    """

    def __init__(
        self,
        sensor_columns,
        num_classes,
        num_imus=1,
        channels_per_imu=3,
        hidden_dim=96,
        dropout=0.2,
        temperature=1.0,
        hard=True,
    ):
        super().__init__()
        self.sensor_columns = [str(c) for c in sensor_columns]
        self.num_total_channels = len(self.sensor_columns)
        self.sensor_groups = sensor_groups_from_columns(self.sensor_columns)
        self.imu_names = list(self.sensor_groups.keys())
        self.num_imus_total = len(self.imu_names)
        self.num_imus = int(num_imus)
        self.channels_per_imu = int(channels_per_imu)
        self.temperature = float(temperature)
        self.hard = bool(hard)

        group_sizes = {name: len(indices) for name, indices in self.sensor_groups.items()}
        if len(set(group_sizes.values())) != 1:
            raise ValueError(f"All IMUs must have the same number of channels, got {group_sizes}")
        self.channels_per_group = next(iter(group_sizes.values()))

        self.imu_logits = nn.Parameter(torch.zeros(self.num_imus, self.num_imus_total))
        self.channel_logits = nn.Parameter(
            torch.zeros(self.num_imus_total, self.channels_per_imu, self.channels_per_group)
        )
        nn.init.normal_(self.imu_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.channel_logits, mean=0.0, std=0.02)

        self.features = CNNFeatureExtractor1D(self.num_total_channels, hidden_dim=hidden_dim, stem_channels=48)
        self.classifier = GlobalAvgPoolClassifier(hidden_dim, num_classes, dropout=dropout)
        self.last_channel_mask = None

    def set_temperature(self, temperature):
        self.temperature = float(temperature)

    def selection_mask(self):
        imu_slots = F.gumbel_softmax(
            self.imu_logits,
            tau=self.temperature,
            hard=self.hard,
            dim=-1,
        )
        imu_weights = combine_slot_probabilities(imu_slots)
        channel_mask = torch.zeros(
            self.num_total_channels,
            device=self.channel_logits.device,
            dtype=self.channel_logits.dtype,
        )
        for imu_idx, imu_name in enumerate(self.imu_names):
            channel_slots = F.gumbel_softmax(
                self.channel_logits[imu_idx],
                tau=self.temperature,
                hard=self.hard,
                dim=-1,
            )
            local_channel_weights = combine_slot_probabilities(channel_slots) * imu_weights[imu_idx]
            indices = self.sensor_groups[imu_name]
            channel_mask[indices] = local_channel_weights
        return channel_mask

    def regularization_loss(self):
        loss = self._slot_diversity_loss(F.softmax(self.imu_logits, dim=-1))
        for imu_idx in range(self.num_imus_total):
            loss = loss + self._slot_diversity_loss(F.softmax(self.channel_logits[imu_idx], dim=-1))
        return loss

    @staticmethod
    def _slot_diversity_loss(slot_probs):
        if slot_probs.shape[0] <= 1:
            return slot_probs.sum() * 0.0
        pairs = []
        for i, j in itertools.combinations(range(slot_probs.shape[0]), 2):
            pairs.append(torch.sum(slot_probs[i] * slot_probs[j]))
        return torch.stack(pairs).mean()

    def forward(self, x):
        channel_mask = self.selection_mask()
        self.last_channel_mask = channel_mask.detach()
        x = x * channel_mask.view(1, -1, 1)
        return self.classifier(self.features(x))

    def export_selection(self):
        with torch.no_grad():
            imu_scores = F.softmax(self.imu_logits, dim=-1).sum(dim=0)
            imu_order = torch.argsort(imu_scores, descending=True).tolist()
            selected_imu_indices = imu_order[: self.num_imus]
            selected = []
            selected_imus = []
            for imu_idx in selected_imu_indices:
                imu_name = self.imu_names[imu_idx]
                selected_imus.append(
                    {
                        "imu": imu_name,
                        "score": float(imu_scores[imu_idx].cpu()),
                    }
                )
                channel_scores = F.softmax(self.channel_logits[imu_idx], dim=-1).sum(dim=0)
                channel_order = torch.argsort(channel_scores, descending=True).tolist()
                group_indices = self.sensor_groups[imu_name]
                for local_idx in channel_order[: self.channels_per_imu]:
                    global_idx = group_indices[local_idx]
                    selected.append(
                        {
                            "index": int(global_idx),
                            "name": self.sensor_columns[global_idx],
                            "imu": imu_name,
                            "score": float(channel_scores[local_idx].cpu()),
                        }
                    )
            return {
                "selected_imus": selected_imus,
                "selected_channels": selected,
                "selected_channel_names": [item["name"] for item in selected],
                "selected_channel_indices": [item["index"] for item in selected],
                "imu_scores": {
                    self.imu_names[i]: float(imu_scores[i].cpu())
                    for i in range(self.num_imus_total)
                },
            }
