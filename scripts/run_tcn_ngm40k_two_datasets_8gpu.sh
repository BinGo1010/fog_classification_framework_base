#!/usr/bin/env bash
set -euo pipefail

python -m scripts.launch_tcn_ngm40k_two_datasets_8gpu \
  --daphnet-data-dir "dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM" \
  --private-data-dir "dataset/0.Private/processed_NBM_Exp" \
  --output-root outputs/tcn_ngm40k_two_datasets_8gpu_seedset_0_52_161_5216_52161 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --seeds 0,52,161,5216,52161 \
  --batch-size 128 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 5 \
  --tcn-patience 2 \
  --phase full \
  "$@"
