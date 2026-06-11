from __future__ import annotations

from exp.exp_fog_classification import ClassificationExperiment
from exp.exp_loso_classification import LOSOExperiment
from exp.exp_model_profile import ProfileExperiment
from exp.exp_simclr import SimCLRExperiment
from exp.exp_supcon import SupConExperiment


EXPERIMENTS = {
    "ordinary": ClassificationExperiment,
    "classification": ClassificationExperiment,
    "fog_classification": ClassificationExperiment,
    "loso": LOSOExperiment,
    "profile": ProfileExperiment,
    "supcon": SupConExperiment,
    "supcon_finetune": SupConExperiment,
    "supcon_pretrain": SupConExperiment,
    "simclr": SimCLRExperiment,
    "simclr_finetune": SimCLRExperiment,
    "simclr_pretrain": SimCLRExperiment,
}


def build_experiment(cfg):
    mode = cfg.get("experiment", {}).get("mode") or cfg.get("exp_mode") or "ordinary"
    mode = str(mode).lower()
    if mode not in EXPERIMENTS:
        available = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment mode '{mode}'. Available modes: {available}")
    return EXPERIMENTS[mode](cfg)
