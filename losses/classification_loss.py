from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .focal_loss import FocalLoss


def compute_class_weights(train_set, num_classes, device):
    if not hasattr(train_set, "y"):
        return None
    y = train_set.y.cpu().numpy()
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def resolve_class_weights(cfg, train_set, device):
    num_classes = cfg["model"]["num_classes"]
    loss_name = str(cfg["train"].get("loss", "ce")).lower()
    class_weight = cfg["train"].get("class_weight", "none")
    if loss_name == "weighted_ce" and class_weight == "none":
        class_weight = "auto"
    if class_weight == "auto":
        return compute_class_weights(train_set, num_classes, device)
    if isinstance(class_weight, list):
        return torch.tensor(class_weight, dtype=torch.float32, device=device)
    return None


def build_classification_loss(cfg, train_set=None, device="cpu"):
    loss_name = str(cfg["train"].get("loss", "ce")).lower()
    weight = resolve_class_weights(cfg, train_set, device)
    if loss_name in {"ce", "cross_entropy", "weighted_ce"}:
        return nn.CrossEntropyLoss(weight=weight)
    if loss_name == "focal":
        return FocalLoss(weight=weight, gamma=cfg["train"].get("focal_gamma", 2.0))
    raise ValueError(f"Unknown classification loss: {loss_name}")
