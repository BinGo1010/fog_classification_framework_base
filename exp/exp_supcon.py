from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from data_provider.fog_transform import IMUTimeSeriesAugmenter
from exp.evaluator import evaluate_model
from utils.metrics import compute_metrics, save_metric_artifacts
from exp.exp_basic import BaseExperiment
from utils.io import save_json
from losses import SupConLoss, build_classification_loss
from utils.distributed import (
    barrier,
    broadcast_bool,
    get_local_rank,
    is_distributed,
    is_main_process,
    unwrap_model,
)

try:
    from exp.trainer import _save_checkpoint_compat
except ImportError:
    from utils.io import save_checkpoint

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


class _ProjectionAdapter(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model.project(x)


class SupConExperiment(BaseExperiment):
    """Supervised contrastive pretraining followed by classifier fine-tuning."""

    def _show_progress(self):
        return bool(self.cfg.get("train", {}).get("show_progress", True))

    def _augmenter(self):
        acfg = self.cfg.get("augmentations", {})
        return IMUTimeSeriesAugmenter(
            gaussian_noise_std=acfg.get("gaussian_noise_std", 0.03),
            gaussian_noise_prob=acfg.get("gaussian_noise_prob", 1.0),
            scale_range=acfg.get("scale_range", [0.9, 1.1]),
            scale_prob=acfg.get("scale_prob", 1.0),
            time_mask_ratio=acfg.get("time_mask_ratio", 0.08),
            time_mask_prob=acfg.get("time_mask_prob", 0.8),
            time_mask_value=acfg.get("time_mask_value", 0.0),
        )

    def _ddp_wrap(self, model, find_unused_parameters=True):
        if not is_distributed(self.cfg):
            return model
        ddp_kwargs = {
            "find_unused_parameters": bool(find_unused_parameters),
        }
        if self.device.type == "cuda":
            local_rank = get_local_rank(self.cfg)
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        return DistributedDataParallel(model, **ddp_kwargs)

    def _base_model(self, model):
        model = unwrap_model(model)
        return model.base_model if hasattr(model, "base_model") else model

    def _adamw(self, params, lr, weight_decay):
        params = [param for param in params if param.requires_grad]
        if not params:
            raise ValueError("No trainable parameters were provided to AdamW.")
        foreach = self.cfg.get("train", {}).get("optimizer_foreach", None)
        kwargs = {}
        if foreach is not None:
            kwargs["foreach"] = bool(foreach)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)

    def _checkpoint_options(self):
        train_cfg = self.cfg.get("train", {})
        return {
            "include_optimizer": bool(train_cfg.get("checkpoint_include_optimizer", False)),
            "include_scheduler": bool(train_cfg.get("checkpoint_include_scheduler", False)),
        }

    def _enable_only_contrastive_parameters(self, model):
        params = list(model.contrastive_parameters())
        for param in model.parameters():
            param.requires_grad = False
        for param in params:
            param.requires_grad = True

    def _enable_all_parameters(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = not name.startswith("encoder.backbone.projection.")

    def _pretrain(self, model, loader):
        scfg = self.cfg.get("supcon", {})
        epochs = int(scfg.get("pretrain_epochs", 20))
        if epochs <= 0:
            return []

        model.train()
        augmenter = self._augmenter()
        criterion = SupConLoss(
            temperature=scfg.get("temperature", 0.1),
            base_temperature=scfg.get("base_temperature", scfg.get("temperature", 0.1)),
        )
        optimizer = self._adamw(
            self._base_model(model).contrastive_parameters(),
            lr=scfg.get("pretrain_lr", self.cfg["train"].get("lr", 1e-3)),
            weight_decay=scfg.get("weight_decay", self.cfg["train"].get("weight_decay", 0.0)),
        )
        log_rows = []
        for epoch in range(1, epochs + 1):
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            total_loss, n = 0.0, 0
            t0 = time.time()
            for batch in tqdm(
                loader,
                desc=f"supcon-pretrain-{epoch}",
                leave=False,
                disable=(not self.is_main or not self._show_progress()),
            ):
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                x1 = augmenter(x)
                x2 = augmenter(x)
                z = model(torch.cat([x1, x2], dim=0))
                z1, z2 = z.chunk(2, dim=0)
                features = torch.stack([z1, z2], dim=1)
                loss = criterion(features, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                total_loss += loss.item() * bs
                n += bs
            row = {
                "epoch": epoch,
                "loss": total_loss / max(n, 1),
                "time_sec": round(time.time() - t0, 3),
            }
            log_rows.append(row)
            if self.is_main:
                print(f"SupCon epoch {epoch:03d} | loss={row['loss']:.4f}")

        if self.is_main:
            _save_checkpoint_compat(
                self.out_dir / "pretrained_supcon.pt",
                self._base_model(model),
                optimizer=optimizer,
                epoch=epochs,
                metrics={"supcon_loss": log_rows[-1]["loss"] if log_rows else None},
                cfg=self.cfg,
                **self._checkpoint_options(),
            )
        barrier(self.cfg)
        return log_rows

    def _run_epoch(self, model, loader, criterion, optimizer=None):
        train = optimizer is not None
        run_model = model if train else unwrap_model(model)
        run_model.train(train)
        y_true, y_prob, indices = [], [], []
        total_loss, n = 0.0, 0
        for batch in tqdm(
            loader,
            desc="finetune-train" if train else "finetune-val",
            leave=False,
            disable=(not self.is_main or not self._show_progress()),
        ):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train):
                logits = run_model(x)
                loss = criterion(logits, y)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_clip = self.cfg["train"].get("grad_clip")
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

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
        metrics = compute_metrics(
            y_true,
            y_prob,
            self.cfg["model"]["num_classes"],
            self.cfg["metrics"].get("top_k", [1]),
            loss=total_loss / max(n, 1),
        )
        return metrics, y_true, y_prob, indices

    def _finetune(self, model, loaders):
        scfg = self.cfg.get("supcon", {})
        train_encoder = bool(scfg.get("finetune_encoder", True))
        if not train_encoder:
            for param in self._base_model(model).encoder.parameters():
                param.requires_grad = False

        criterion = build_classification_loss(self.cfg, train_set=loaders["train_set"], device=self.device)
        optimizer = self._adamw(
            self._base_model(model).classifier_parameters(train_encoder=train_encoder),
            lr=self.cfg["train"].get("lr", 1e-3),
            weight_decay=self.cfg["train"].get("weight_decay", 0.0),
        )

        monitor = self.cfg["train"].get("monitor", "val_f1_macro").replace("val_", "")
        monitor = {"macro_f1": "f1_macro"}.get(monitor, monitor)
        mode = self.cfg["train"].get("monitor_mode", "max")
        best = -float("inf") if mode == "max" else float("inf")
        patience = int(self.cfg["train"].get("early_stopping_patience", 10))
        save_best = bool(self.cfg["train"].get("save_best_checkpoint", True))
        save_last = bool(self.cfg["train"].get("save_last_checkpoint", False))
        bad_epochs = 0
        epochs = int(self.cfg["train"].get("epochs", 30))

        for epoch in range(1, epochs + 1):
            sampler = getattr(loaders["train"], "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_metrics, *_ = self._run_epoch(model, loaders["train"], criterion, optimizer=optimizer)
            stop_now = False
            val_metrics, y_true, y_prob, idx = self._run_epoch(model, loaders["val"], criterion)
            if self.is_main:
                current = val_metrics.get(monitor)
                improved = current is not None and ((mode == "max" and current > best) or (mode == "min" and current < best))
                print(
                    f"Finetune epoch {epoch:03d} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_f1_macro={val_metrics.get('f1_macro', 0):.4f}"
                )
                if improved:
                    best = current
                    bad_epochs = 0
                    if save_best:
                        _save_checkpoint_compat(
                            self.out_dir / "best.pt",
                            model,
                            optimizer,
                            None,
                            epoch,
                            val_metrics,
                            self.cfg,
                            **self._checkpoint_options(),
                        )
                    save_metric_artifacts(
                        self.out_dir,
                        "val_best",
                        y_true,
                        y_prob,
                        idx,
                        self.cfg["model"]["num_classes"],
                        val_metrics,
                        self.cfg.get("metrics", {}),
                    )
                else:
                    bad_epochs += 1
                if save_last:
                    _save_checkpoint_compat(
                        self.out_dir / "last.pt",
                        model,
                        optimizer,
                        None,
                        epoch,
                        val_metrics,
                        self.cfg,
                        **self._checkpoint_options(),
                    )
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    stop_now = True
            stop_now = broadcast_bool(stop_now, self.cfg)
            if stop_now:
                break
            barrier(self.cfg)

    def run(self):
        self.setup()
        self.is_main = is_main_process(self.cfg)
        device_cfg = self.cfg.get("runtime", {}).get("device", self.cfg["project"].get("device", "auto"))
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_cfg)
        loaders = self.build_dataloaders()
        base_model = self.build_model().to(self.device)
        self._enable_only_contrastive_parameters(base_model)
        pretrain_model = self._ddp_wrap(_ProjectionAdapter(base_model).to(self.device), find_unused_parameters=True)
        pretrain_log = self._pretrain(pretrain_model, loaders["train"])
        if self.is_main and pretrain_log:
            save_json({"pretrain": pretrain_log}, self.out_dir / "supcon_pretrain_log.json")
        del pretrain_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        barrier(self.cfg)
        self._enable_all_parameters(base_model)
        model = self._ddp_wrap(base_model, find_unused_parameters=True)
        self._finetune(model, loaders)
        if not self.is_main:
            barrier(self.cfg)
            return {}
        metrics = evaluate_model(self.cfg, self._base_model(model), loaders["test"], split="test", out_dir=self.out_dir, device=self.device)
        save_json(metrics, self.out_dir / "metrics_test.json")
        barrier(self.cfg)
        return metrics
