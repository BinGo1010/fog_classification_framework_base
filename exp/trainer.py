from __future__ import annotations
import inspect
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from losses import build_classification_loss
from utils.metrics import compute_metrics, save_metric_artifacts
from utils.io import save_checkpoint, append_csv, save_json
from models.build import count_parameters
from utils.distributed import (
    barrier,
    broadcast_bool,
    get_local_rank,
    is_distributed,
    is_main_process,
    unwrap_model,
)


def select_device(device_cfg="auto"):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_optimizer(cfg, model):
    tcfg = cfg["train"]
    name = tcfg.get("optimizer", "AdamW")
    lr, wd = tcfg.get("lr", 1e-3), tcfg.get("weight_decay", 0.0)
    foreach = tcfg.get("optimizer_foreach", None)
    adam_kwargs = {}
    if foreach is not None:
        adam_kwargs["foreach"] = bool(foreach)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, **adam_kwargs)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, **adam_kwargs)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg, optimizer):
    tcfg = cfg["train"]
    name = tcfg.get("scheduler", "none")
    if name in {"none", None}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.get("epochs", 30))
    if name in {"step", "type1"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=tcfg.get("step_size", 10), gamma=tcfg.get("gamma", 0.5))
    raise ValueError(f"Unknown scheduler: {name}")


def build_criterion(cfg, train_set, device):
    return build_classification_loss(cfg, train_set=train_set, device=device)


def _needs_unused_parameter_detection(model) -> bool:
    name = model.__class__.__name__.lower()
    module = getattr(model, "__module__", "").lower()
    if "forecasting_adapters" in module:
        return True
    if any(token in name for token in ["informer", "autoformer", "nonstationary", "itransformer", "timesnet"]):
        return True
    return False


def _resolve_ddp_find_unused(value, model) -> bool:
    if value in {"auto", None}:
        return _needs_unused_parameter_detection(model)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid train.ddp_find_unused_parameters: {value}")


def _save_checkpoint_compat(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    metrics=None,
    cfg=None,
    include_optimizer=True,
    include_scheduler=True,
):
    if "include_optimizer" in inspect.signature(save_checkpoint).parameters:
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            cfg,
            include_optimizer=include_optimizer,
            include_scheduler=include_scheduler,
        )
        return

    payload = {
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "cfg": cfg or {},
    }
    if include_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if include_scheduler and scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class Trainer:
    def __init__(self, cfg, model, loaders):
        self.cfg = cfg
        self.device = select_device(cfg.get("runtime", {}).get("device", cfg["project"].get("device", "auto")))
        self.distributed = is_distributed(cfg)
        self.is_main = is_main_process(cfg)
        self.model = model.to(self.device)
        if self.distributed:
            find_unused = cfg["train"].get("ddp_find_unused_parameters", "auto")
            ddp_kwargs = {
                "find_unused_parameters": _resolve_ddp_find_unused(find_unused, self.model),
            }
            if self.device.type == "cuda":
                local_rank = get_local_rank(cfg)
                ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
            self.model = DistributedDataParallel(self.model, **ddp_kwargs)
        self.loaders = loaders
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.criterion = build_criterion(cfg, loaders["train_set"], self.device)
        self.out_dir = Path(cfg["project"]["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        if self.is_main:
            dist_text = f", DDP world_size={cfg.get('runtime', {}).get('world_size', 1)}" if self.distributed else ""
            print(f"Using device: {self.device} ({device_name}{dist_text})")
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", False) and self.device.type == "cuda"))

    def _run_epoch(self, split, train=False):
        model = self.model if train else unwrap_model(self.model)
        model.train(train)
        loader = self.loaders[split]
        y_true, y_prob, indices = [], [], []
        total_loss, n = 0.0, 0
        iterator = tqdm(loader, desc=split, leave=False, disable=not self.is_main)
        for batch in iterator:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    logits = model(x)
                    loss = self.criterion(logits, y)
                if train:
                    self.scaler.scale(loss).backward()
                    grad_clip = self.cfg["train"].get("grad_clip", None)
                    if grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            prob = torch.softmax(logits.detach(), dim=1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs
            y_true.append(y.detach().cpu().numpy())
            y_prob.append(prob.cpu().numpy())
            indices.append(batch.get("index", torch.arange(bs)).detach().cpu().numpy())
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        indices = np.concatenate(indices)
        metrics = compute_metrics(y_true, y_prob, self.cfg["model"]["num_classes"], self.cfg["metrics"].get("top_k", [1]), loss=total_loss / max(n, 1))
        return metrics, y_true, y_prob, indices

    def fit(self):
        monitor = self.cfg["train"].get("monitor", "val_macro_f1").replace("val_", "")
        metric_aliases = {
            "macro_f1": "f1_macro",
            "weighted_f1": "f1_weighted",
            "micro_f1": "f1_micro",
            "val_macro_f1": "f1_macro",
            "val_weighted_f1": "f1_weighted",
            "val_micro_f1": "f1_micro",
        }
        monitor = metric_aliases.get(monitor, monitor)
        mode = self.cfg["train"].get("monitor_mode", "max")
        best = -float("inf") if mode == "max" else float("inf")
        patience = int(self.cfg["train"].get("early_stopping_patience", 10))
        save_best = bool(self.cfg["train"].get("save_best_checkpoint", True))
        save_last = bool(self.cfg["train"].get("save_last_checkpoint", False))
        ckpt_include_optimizer = bool(self.cfg["train"].get("checkpoint_include_optimizer", False))
        ckpt_include_scheduler = bool(self.cfg["train"].get("checkpoint_include_scheduler", False))
        bad_epochs = 0
        for epoch in range(1, int(self.cfg["train"].get("epochs", 30)) + 1):
            sampler = getattr(self.loaders["train"], "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            t0 = time.time()
            train_metrics, *_ = self._run_epoch("train", train=True)
            if self.scheduler is not None:
                self.scheduler.step()
            stop_now = False
            if self.is_main:
                val_metrics, y_true, y_prob, idx = self._run_epoch("val", train=False)
                row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"], "time_sec": round(time.time()-t0, 3)}
                row.update({f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))})
                row.update({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
                append_csv(self.out_dir / "logs" / "train_log.csv", row)
                current = val_metrics.get(monitor)
                improved = current is not None and ((mode == "max" and current > best) or (mode == "min" and current < best))
                print(f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | val_f1_macro={val_metrics.get('f1_macro', 0):.4f}")
                if improved:
                    best = current
                    bad_epochs = 0
                    if save_best:
                        _save_checkpoint_compat(
                            self.out_dir / "best.pt",
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            val_metrics,
                            self.cfg,
                            include_optimizer=ckpt_include_optimizer,
                            include_scheduler=ckpt_include_scheduler,
                        )
                    save_metric_artifacts(self.out_dir, "val_best", y_true, y_prob, idx, self.cfg["model"]["num_classes"], val_metrics)
                else:
                    bad_epochs += 1
                if save_last:
                    _save_checkpoint_compat(
                        self.out_dir / "last.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        val_metrics,
                        self.cfg,
                        include_optimizer=ckpt_include_optimizer,
                        include_scheduler=ckpt_include_scheduler,
                    )
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    stop_now = True
            stop_now = broadcast_bool(stop_now, self.cfg)
            if stop_now:
                break
            barrier(self.cfg)
        if self.is_main:
            info = {"num_trainable_parameters": count_parameters(unwrap_model(self.model))}
            save_json(info, self.out_dir / "model_info.json")
