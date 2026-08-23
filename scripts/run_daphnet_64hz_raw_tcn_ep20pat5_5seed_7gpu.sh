#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/launch_daphnet_64hz_raw_tcn_ep20pat5_5seed_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  "$@"
