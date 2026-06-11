from __future__ import annotations

from exp.exp_basic import BaseExperiment


class ClassificationExperiment(BaseExperiment):
    """Ordinary FOG classification: train, validate, then test best checkpoint."""

    def run(self):
        self.setup()
        loaders = self.build_dataloaders()
        model = self.build_model()
        if int(self.cfg.get("project", {}).get("is_training", 1)) == 1:
            self.fit(loaders, model)
        return self.evaluate_checkpoint(loaders, split="test")
