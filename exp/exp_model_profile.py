from __future__ import annotations

import torch

from exp.exp_basic import BaseExperiment
from utils.imu_profile import profile_conv1d_model
from utils.io import save_json


class ProfileExperiment(BaseExperiment):
    """Model complexity and inference profile for a configured model."""

    def run(self):
        self.setup()
        loaders = self.build_dataloaders()
        model = self.build_model()
        profile = profile_conv1d_model(
            model,
            in_channels=self.cfg["model"]["in_channels"],
            seq_len=self.cfg["model"].get("seq_len", loaders["train_set"].X.shape[-1]),
            device=torch.device("cpu"),
        )
        save_json(profile, self.out_dir / "model_profile.json")
        for key, value in profile.items():
            print(f"{key}: {value}")
        return profile
