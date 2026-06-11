from __future__ import annotations

from pathlib import Path

import torch

from data_provider.build import build_dataloaders
from exp.evaluator import evaluate_model
from exp.trainer import Trainer
from models.build import build_model
from utils.config import save_config
from utils.io import load_checkpoint
from utils.seed import seed_everything
from utils.distributed import barrier, is_main_process


class BaseExperiment:
    """Shared setup for train/evaluate style experiments."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = Path(cfg["project"]["output_dir"])

    def setup(self) -> None:
        seed_everything(self.cfg["project"].get("seed", 42))
        if is_main_process(self.cfg):
            self.out_dir.mkdir(parents=True, exist_ok=True)
            save_config(self.cfg, self.out_dir / "config_resolved.yaml")
        barrier(self.cfg)

    def build_dataloaders(self):
        loaders = build_dataloaders(self.cfg)
        train_set = loaders.get("train_set")
        if hasattr(train_set, "num_channels"):
            self.cfg["model"]["in_channels"] = int(train_set.num_channels)
        return loaders

    def build_model(self):
        return build_model(self.cfg)

    def fit(self, loaders, model):
        trainer = Trainer(self.cfg, model, loaders)
        trainer.fit()

    def best_checkpoint(self) -> Path:
        best_path = self.out_dir / "best.pt"
        if best_path.exists():
            return best_path
        return self.out_dir / "last.pt"

    def evaluate_checkpoint(self, loaders, split="test"):
        if not is_main_process(self.cfg):
            barrier(self.cfg)
            return {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.cfg.get("runtime", {}).get("device", device))
        model = self.build_model()
        checkpoint = self.cfg.get("project", {}).get("checkpoint")
        load_checkpoint(checkpoint or self.best_checkpoint(), model, map_location=device)
        metrics = evaluate_model(
            self.cfg,
            model,
            loaders[split],
            split=split,
            out_dir=self.out_dir,
            device=device,
        )
        barrier(self.cfg)
        return metrics

    def run(self):
        raise NotImplementedError
