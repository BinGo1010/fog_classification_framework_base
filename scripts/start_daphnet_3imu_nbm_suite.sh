#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXE="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_EXE="python"
else
  PYTHON_EXE="python3"
fi

export PYTHONUNBUFFERED=1
exec "$PYTHON_EXE" -u "$SCRIPT_DIR/start_daphnet_3imu_nbm_suite.py" "$@"
