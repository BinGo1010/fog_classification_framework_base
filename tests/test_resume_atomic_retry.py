from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from cnbr_fog.resume import atomic_json_dump


def test_atomic_replace_retries_transient_windows_permission_error() -> None:
    path = Path(tempfile.gettempdir()) / f"atomic-retry-{uuid.uuid4().hex}.json"
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError(5, "transient scanner lock")
        real_replace(source, destination)

    try:
        with patch("cnbr_fog.resume.os.replace", side_effect=flaky_replace), patch(
            "cnbr_fog.resume.time.sleep"
        ):
            atomic_json_dump({"status": "complete"}, path)
        assert attempts == 3
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "status": "complete"
        }
    finally:
        path.unlink(missing_ok=True)

