"""Atomic artifact and exact epoch-boundary resume helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(root: str | Path) -> str:
    """Hash Daphnet metadata and every referenced record in manifest order."""

    root = Path(root)
    digest = hashlib.sha256()
    for metadata_name in ("manifest.csv", "schema.json"):
        path = root / metadata_name
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(metadata_name.encode("utf-8"))
        digest.update(path.read_bytes())

    import csv

    with (root / "manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        record_paths = [row["record_path"] for row in csv.DictReader(handle)]
    for relative in record_paths:
        path = root / relative
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _temporary_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{suffix}")


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, "torch")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, "json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz_save(path: str | Path, compressed: bool = True, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, "arrays.npz")
    if compressed:
        np.savez_compressed(temporary, **arrays)
    else:
        np.savez(temporary, **arrays)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_checkpoint(
    payload: dict,
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str | None = None,
) -> None:
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Incompatible checkpoint {key}: {payload.get(key)!r} != {value!r}"
            )
    if upstream_sha256 is not None and payload.get("upstream_nbm_sha256") != upstream_sha256:
        raise ValueError("Classifier checkpoint belongs to a different NBM")


def done_payload(
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    artifacts: dict[str, str | Path],
    upstream_sha256: str | None = None,
    relative_to: str | Path | None = None,
) -> dict:
    resolved = {name: Path(path).resolve() for name, path in artifacts.items()}
    missing = [str(path) for path in resolved.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot mark task done; missing {missing}")
    relative_root = Path(relative_to).resolve() if relative_to is not None else None
    if relative_root is not None:
        stored_paths = {
            name: path.relative_to(relative_root)
            for name, path in resolved.items()
        }
    else:
        stored_paths = resolved
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
        "artifacts": {
            name: {
                "path": str(stored_paths[name]),
                "sha256": sha256_file(resolved[name]),
                "bytes": int(resolved[name].stat().st_size),
            }
            for name in resolved
        },
    }
    if upstream_sha256 is not None:
        payload["upstream_nbm_sha256"] = upstream_sha256
    return payload


def validate_done(
    path: str | Path,
    *,
    stage: str,
    protocol_fingerprint: str,
    task_id: str,
    upstream_sha256: str | None = None,
) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"DONE format version mismatch in {path}")
    if payload.get("stage") != stage:
        raise ValueError(f"DONE stage mismatch in {path}")
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError(f"DONE protocol mismatch in {path}")
    if payload.get("task_id") != task_id:
        raise ValueError(f"DONE task mismatch in {path}")
    if upstream_sha256 is not None and payload.get("upstream_nbm_sha256") != upstream_sha256:
        raise ValueError(f"DONE upstream NBM mismatch in {path}")
    for artifact in payload.get("artifacts", {}).values():
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(f"DONE references missing {artifact_path}")
        if int(artifact_path.stat().st_size) != int(artifact["bytes"]):
            raise ValueError(f"DONE artifact size mismatch: {artifact_path}")
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"DONE artifact hash mismatch: {artifact_path}")
    return payload
