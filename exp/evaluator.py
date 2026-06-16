from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import compute_metrics, save_metric_artifacts
from models.build import count_parameters
from utils.io import save_json


@torch.no_grad()
def evaluate_model(cfg, model, loader, split="test", out_dir=None, device=None, benchmark=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    y_true, y_prob, indices = [], [], []
    total_loss, n = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    show_progress = bool(cfg.get("train", {}).get("show_progress", True))
    for batch in tqdm(loader, desc=f"eval-{split}", leave=False, disable=not show_progress):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        y_true.append(y.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
        indices.append(batch.get("index", torch.arange(bs)).cpu().numpy())
    elapsed = time.perf_counter() - start
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    indices = np.concatenate(indices)
    metrics = compute_metrics(y_true, y_prob, cfg["model"]["num_classes"], cfg["metrics"].get("top_k", [1]), total_loss/max(n,1))
    if benchmark:
        metrics["num_trainable_parameters"] = int(count_parameters(model))
        metrics["avg_inference_time_ms_per_sample"] = float(elapsed / max(n, 1) * 1000)
        metrics["throughput_samples_per_sec"] = float(n / max(elapsed, 1e-12))
    if out_dir is not None:
        out_dir = Path(out_dir)
        save_metric_artifacts(out_dir, split, y_true, y_prob, indices, cfg["model"]["num_classes"], metrics)
        save_json(metrics, out_dir / f"metrics_{split}.json")
    return metrics
