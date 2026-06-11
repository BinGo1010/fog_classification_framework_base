from __future__ import annotations

import torch
import torch.nn as nn

from layers import BinaryConcreteGate
from utils.imu_selection import group_imu_channels

from .lightweight_imu_tcn import LightweightIMUTCN


class CostAwareGumbelIMUTCN(nn.Module):
    """Cost-aware binary-concrete IMU gates followed by LightweightIMUTCN."""

    def __init__(
        self,
        sensor_columns,
        num_classes,
        sensor_costs=None,
        hidden_channels=32,
        levels=4,
        kernel_size=3,
        dropout=0.1,
        temperature=5.0,
        hard=False,
        target_num_imus=None,
        target_count_weight=0.0,
        **kwargs,
    ):
        super().__init__()
        self.sensor_columns = [str(name) for name in sensor_columns]
        self.sensor_groups = group_imu_channels(self.sensor_columns)
        self.imu_names = list(self.sensor_groups.keys())
        self.num_imus = len(self.imu_names)
        self.num_total_channels = len(self.sensor_columns)
        self.temperature = float(temperature)
        self.hard = bool(hard)
        self.force_all_imus = False
        self.target_num_imus = target_num_imus
        self.target_count_weight = float(target_count_weight)

        costs = sensor_costs if sensor_costs is not None else [1.0] * self.num_imus
        if len(costs) != self.num_imus:
            raise ValueError(f"sensor_costs must have length {self.num_imus}, got {len(costs)}")
        self.register_buffer("sensor_costs", torch.tensor(costs, dtype=torch.float32))

        self.gates = BinaryConcreteGate(self.num_imus, temperature=temperature, hard=hard)
        self.backbone = LightweightIMUTCN(
            in_channels=self.num_total_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.last_gates = None

    def set_temperature(self, temperature):
        self.temperature = float(temperature)
        self.gates.set_temperature(temperature)

    def set_force_all_imus(self, enabled):
        self.force_all_imus = bool(enabled)

    def selection_probabilities(self):
        return self.gates.probabilities()

    def sample_gates(self):
        return self.gates(force_all=self.force_all_imus)

    def channel_mask_from_gates(self, gates):
        mask = torch.zeros(
            self.num_total_channels,
            dtype=gates.dtype,
            device=gates.device,
        )
        for imu_idx, imu_name in enumerate(self.imu_names):
            mask[self.sensor_groups[imu_name]] = gates[imu_idx]
        return mask

    def cost_loss(self):
        probs = self.selection_probabilities()
        cost = torch.sum(probs * self.sensor_costs) / torch.sum(self.sensor_costs)
        if self.target_num_imus is not None and self.target_count_weight > 0:
            target = torch.as_tensor(float(self.target_num_imus), device=probs.device)
            cost = cost + self.target_count_weight * torch.square(torch.sum(probs) - target)
        return cost

    def forward(self, x):
        gates = self.sample_gates()
        self.last_gates = gates.detach()
        mask = self.channel_mask_from_gates(gates)
        return self.backbone(x * mask.view(1, -1, 1))

    def export_selection(self, top_k=1, threshold=None, min_imus=1):
        with torch.no_grad():
            probs = self.selection_probabilities().detach().cpu()
            order = torch.argsort(probs, descending=True).tolist()
            if threshold is not None:
                selected_indices = [idx for idx in order if float(probs[idx]) >= float(threshold)]
                if len(selected_indices) < int(min_imus):
                    selected_indices = order[: int(min_imus)]
            else:
                selected_indices = order[: int(top_k)]

            selected_imus = []
            selected_channels = []
            for imu_idx in selected_indices:
                imu_name = self.imu_names[imu_idx]
                prob = float(probs[imu_idx])
                selected_imus.append(
                    {
                        "imu": imu_name,
                        "probability": prob,
                        "cost": float(self.sensor_costs[imu_idx].detach().cpu()),
                    }
                )
                for channel_idx in self.sensor_groups[imu_name]:
                    selected_channels.append(
                        {
                            "index": int(channel_idx),
                            "name": self.sensor_columns[channel_idx],
                            "imu": imu_name,
                            "probability": prob,
                        }
                    )

            expected_cost = float(torch.sum(probs * self.sensor_costs.detach().cpu()))
            return {
                "selected_imus": selected_imus,
                "selected_imu_names": [item["imu"] for item in selected_imus],
                "selected_channels": selected_channels,
                "selected_channel_names": [item["name"] for item in selected_channels],
                "selected_channel_indices": [item["index"] for item in selected_channels],
                "imu_probabilities": {
                    self.imu_names[i]: float(probs[i]) for i in range(self.num_imus)
                },
                "imu_costs": {
                    self.imu_names[i]: float(self.sensor_costs[i].detach().cpu())
                    for i in range(self.num_imus)
                },
                "expected_num_imus": float(torch.sum(probs)),
                "expected_sensor_cost": expected_cost,
                "selected_num_imus": len(selected_indices),
                "selected_num_channels": len(selected_channels),
                "total_num_imus": self.num_imus,
                "total_num_channels": self.num_total_channels,
                "data_rate_ratio": float(len(selected_channels) / max(self.num_total_channels, 1)),
            }
