from __future__ import annotations
from pathlib import Path
import json
import csv
from typing import Dict, Any, Iterable
import torch

from utils.distributed import unwrap_model


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(
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
    payload = {"model": unwrap_model(model).state_dict(), "epoch": epoch, "metrics": metrics or {}, "cfg": cfg or {}}
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


def load_checkpoint(path, model, map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt


def append_csv(path, row: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
