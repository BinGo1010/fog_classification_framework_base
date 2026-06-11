from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def model_size_bytes(model):
    total = 0
    for tensor in model.state_dict().values():
        total += tensor.numel() * tensor.element_size()
    return int(total)


def profile_conv1d_model(model, in_channels, seq_len, device=None):
    """Estimate Conv1d/Linear MACs by running one dummy window."""

    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    macs = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        x = inputs[0]
        batch = int(x.shape[0])
        out_channels = int(output.shape[1])
        out_length = int(output.shape[2])
        kernel = int(module.kernel_size[0])
        in_per_group = int(module.in_channels // module.groups)
        macs += batch * out_channels * out_length * in_per_group * kernel

    def linear_hook(module, inputs, output):
        nonlocal macs
        batch = int(inputs[0].shape[0])
        macs += batch * int(module.in_features) * int(module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        dummy = torch.zeros(1, int(in_channels), int(seq_len), device=device)
        model(dummy)

    for hook in hooks:
        hook.remove()

    size_bytes = model_size_bytes(model)
    return {
        "macs_per_window": int(macs),
        "flops_per_window": int(2 * macs),
        "model_size_bytes": size_bytes,
        "model_size_kb": float(size_bytes / 1024.0),
    }


def checkpoint_size_bytes(path):
    path = Path(path)
    return int(path.stat().st_size) if path.exists() else None
