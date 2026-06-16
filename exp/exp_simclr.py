from __future__ import annotations

import time

import torch
from tqdm import tqdm

from data_provider.fog_transform import IMUTimeSeriesAugmenter
from losses.nt_xent_loss import NTXentLoss
from exp.exp_supcon import SupConExperiment, _save_checkpoint_compat
from utils.distributed import barrier


class SimCLRExperiment(SupConExperiment):
    """SimCLR pretraining followed by the same classifier fine-tuning path."""

    def _contrastive_cfg(self):
        return self.cfg.get("simclr", self.cfg.get("supcon", {}))

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

    def _pretrain(self, model, loader):
        scfg = self._contrastive_cfg()
        epochs = int(scfg.get("pretrain_epochs", 20))
        if epochs <= 0:
            return []

        model.train()
        augmenter = self._augmenter()
        criterion = NTXentLoss(temperature=scfg.get("temperature", 0.1))
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
                desc=f"simclr-pretrain-{epoch}",
                leave=False,
                disable=(not self.is_main or not self._show_progress()),
            ):
                x = batch["x"].to(self.device, non_blocking=True)
                x1 = augmenter(x)
                x2 = augmenter(x)
                z = model(torch.cat([x1, x2], dim=0))
                z1, z2 = z.chunk(2, dim=0)
                loss = criterion(z1, z2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                total_loss += loss.item() * bs
                n += bs
            row = {
                "epoch": epoch,
                "loss": total_loss / max(n, 1),
                "time_sec": round(time.time() - t0, 3),
            }
            log_rows.append(row)
            if self.is_main:
                print(f"SimCLR epoch {epoch:03d} | loss={row['loss']:.4f}")

        if self.is_main:
            _save_checkpoint_compat(
                self.out_dir / "pretrained_simclr.pt",
                self._base_model(model),
                optimizer=optimizer,
                epoch=epochs,
                metrics={"simclr_loss": log_rows[-1]["loss"] if log_rows else None},
                cfg=self.cfg,
                **self._checkpoint_options(),
            )
        barrier(self.cfg)
        return log_rows

    def run(self):
        metrics = super().run()
        log_path = self.out_dir / "supcon_pretrain_log.json"
        if log_path.exists():
            log_path.rename(self.out_dir / "simclr_pretrain_log.json")
        return metrics
