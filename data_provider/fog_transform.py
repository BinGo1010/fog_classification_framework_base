from __future__ import annotations

import torch


class IMUTimeSeriesAugmenter:
    """Generate augmented IMU windows for contrastive pretraining.

    Input and output tensors use [B, C, T].
    """

    def __init__(
        self,
        gaussian_noise_std=0.03,
        gaussian_noise_prob=1.0,
        scale_range=(0.9, 1.1),
        scale_prob=1.0,
        time_mask_ratio=0.08,
        time_mask_prob=0.8,
        time_mask_value=0.0,
    ):
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.scale_prob = float(scale_prob)
        self.time_mask_ratio = float(time_mask_ratio)
        self.time_mask_prob = float(time_mask_prob)
        self.time_mask_value = float(time_mask_value)

    def __call__(self, x):
        out = x.clone()
        if self.scale_prob > 0:
            out = self._scale(out)
        if self.gaussian_noise_std > 0 and self.gaussian_noise_prob > 0:
            out = self._add_noise(out)
        if self.time_mask_ratio > 0 and self.time_mask_prob > 0:
            out = self._time_mask(out)
        return out

    def _sample_mask(self, batch_size, probability, device):
        return (torch.rand(batch_size, device=device) < probability).view(batch_size, 1, 1)

    def _scale(self, x):
        mask = self._sample_mask(x.size(0), self.scale_prob, x.device).to(x.dtype)
        scale = torch.empty(x.size(0), 1, 1, device=x.device, dtype=x.dtype)
        scale.uniform_(self.scale_min, self.scale_max)
        return x * (mask * scale + (1.0 - mask))

    def _add_noise(self, x):
        mask = self._sample_mask(x.size(0), self.gaussian_noise_prob, x.device).to(x.dtype)
        noise = torch.randn_like(x) * self.gaussian_noise_std
        return x + mask * noise

    def _time_mask(self, x):
        batch_size, _, seq_len = x.shape
        max_len = max(1, int(round(seq_len * self.time_mask_ratio)))
        out = x.clone()
        apply = torch.rand(batch_size, device=x.device) < self.time_mask_prob
        for batch_idx in torch.where(apply)[0].tolist():
            mask_len = int(torch.randint(1, max_len + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, seq_len - mask_len + 1, (1,), device=x.device).item())
            out[batch_idx, :, start : start + mask_len] = self.time_mask_value
        return out


def build_augmenter(cfg):
    acfg = cfg.get("augmentations", {})
    return IMUTimeSeriesAugmenter(
        gaussian_noise_std=acfg.get("gaussian_noise_std", 0.03),
        gaussian_noise_prob=acfg.get("gaussian_noise_prob", 1.0),
        scale_range=acfg.get("scale_range", [0.9, 1.1]),
        scale_prob=acfg.get("scale_prob", 1.0),
        time_mask_ratio=acfg.get("time_mask_ratio", 0.08),
        time_mask_prob=acfg.get("time_mask_prob", 0.8),
        time_mask_value=acfg.get("time_mask_value", 0.0),
    )
