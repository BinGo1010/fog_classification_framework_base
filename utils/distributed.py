from __future__ import annotations

import datetime
import os
import inspect
from typing import Any

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def init_distributed(cfg: dict[str, Any]) -> dict[str, Any]:
    """Initialize torch.distributed when launched by torchrun.

    Single-process runs keep the old behavior. Multi-GPU runs should be started
    with torchrun, for example: torchrun --nproc_per_node=8 run.py ...
    """

    train_cfg = cfg.get("train", {})
    requested = train_cfg.get("distributed", "auto")
    world_size = _env_int("WORLD_SIZE", 1)
    distributed = world_size > 1
    if requested in {False, "false", "False", "none", "off"}:
        distributed = False
    if requested in {True, "true", "True", "ddp"} and world_size <= 1:
        raise RuntimeError("DDP requires torchrun. Use: torchrun --nproc_per_node=<N> run.py ...")

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    device_cfg = cfg.get("project", {}).get("device", "auto")
    use_cuda = torch.cuda.is_available() and device_cfg != "cpu"
    if distributed and not dist.is_initialized():
        backend = train_cfg.get("distributed_backend")
        if backend is None:
            backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(local_rank)
        kwargs = {"backend": backend, "init_method": "env://"}
        timeout_minutes = int(train_cfg.get("distributed_timeout_minutes", os.environ.get("DDP_TIMEOUT_MINUTES", 120)))
        kwargs["timeout"] = datetime.timedelta(minutes=timeout_minutes)
        if use_cuda and "device_id" in inspect.signature(dist.init_process_group).parameters:
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**kwargs)

    device = device_cfg
    if distributed:
        device = f"cuda:{local_rank}" if use_cuda else "cpu"
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.setdefault("runtime", {})
    cfg["runtime"].update(
        {
            "distributed": bool(distributed),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size if distributed else 1,
            "is_main_process": rank == 0,
            "device": device,
        }
    )
    cfg.setdefault("project", {})["device"] = device
    return cfg


def is_distributed(cfg: dict[str, Any] | None = None) -> bool:
    if cfg is not None:
        return bool(cfg.get("runtime", {}).get("distributed", False))
    return dist.is_available() and dist.is_initialized()


def is_main_process(cfg: dict[str, Any] | None = None) -> bool:
    if cfg is not None:
        return bool(cfg.get("runtime", {}).get("is_main_process", True))
    return not is_distributed() or dist.get_rank() == 0


def get_local_rank(cfg: dict[str, Any]) -> int:
    return int(cfg.get("runtime", {}).get("local_rank", 0))


def barrier(cfg: dict[str, Any] | None = None) -> None:
    if is_distributed(cfg):
        if cfg is not None:
            device = str(cfg.get("runtime", {}).get("device", ""))
            if device.startswith("cuda") and "device_ids" in inspect.signature(dist.barrier).parameters:
                dist.barrier(device_ids=[int(device.split(":", 1)[1])])
                return
        dist.barrier()


def broadcast_bool(value: bool, cfg: dict[str, Any]) -> bool:
    if not is_distributed(cfg):
        return bool(value)
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    tensor = torch.tensor([1 if value else 0], dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


__all__ = [
    "barrier",
    "broadcast_bool",
    "cleanup_distributed",
    "get_local_rank",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "unwrap_model",
]
