from __future__ import annotations

import torch


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B, device=device)[:, None, None],
            torch.arange(H, device=device)[None, :, None],
            index,
            :,
        ]
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
