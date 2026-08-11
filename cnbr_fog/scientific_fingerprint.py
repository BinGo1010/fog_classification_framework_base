"""Path-independent fingerprints for the processed Daphnet NBM experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def processed_nbm_scientific_manifest(root: str | Path) -> dict[str, Any]:
    """Bind every byte consumed by this experiment's data/split loaders.

    Absolute paths are intentionally excluded so an identical server copy has
    the same identity as the local dataset.
    """

    root = Path(root).resolve()
    fixed = (
        root / "manifest.csv",
        root / "schema.json",
        root / "nbm_protocol.json",
        root / "nbm_quality_report.json",
    )
    paths = [*fixed]
    # Sort by the relative POSIX spelling, not by platform-specific Path
    # ordering.  Linux is case-sensitive while Windows Path ordering is not;
    # a canonical key keeps otherwise identical server/local copies stable.
    relative_key = lambda path: path.relative_to(root).as_posix()
    paths.extend(sorted((root / "records").glob("**/*"), key=relative_key))
    paths.extend(
        sorted((root / "split_indices").glob("**/*"), key=relative_key)
    )
    paths = [path for path in paths if path.is_file()]
    missing = [str(path) for path in fixed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"processed_NBM identity files missing: {missing}")
    if not any((root / "records") in path.parents for path in paths):
        raise FileNotFoundError(f"no record files under {root / 'records'}")
    if not any((root / "split_indices") in path.parents for path in paths):
        raise FileNotFoundError(f"no split files under {root / 'split_indices'}")
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    core = {
        "schema": "processed_nbm_scientific_manifest.v1",
        "definition": (
            "manifest+schema+protocol+quality, all records, and all split_indices"
        ),
        "files": entries,
    }
    return {**core, "sha256": _stable_hash(core)}
