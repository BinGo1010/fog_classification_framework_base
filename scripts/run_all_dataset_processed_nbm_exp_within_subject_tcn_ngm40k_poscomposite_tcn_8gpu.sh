#!/usr/bin/env bash
set -euo pipefail

python -m scripts.launch_all_dataset_processed_nbm_exp_within_subject_tcn_ngm40k_poscomposite_tcn_8gpu \
  --data-dir dataset/0.Private/processed_NBM_Exp \
  --output-root outputs/all_dataset_processed_NBM_Exp_within_subject_tcn_ngm40k_poscomposite_C_tcn_nbm400pat40_ep30pat8_seedset_0_52_161_5216_52161 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --seeds 0,52,161,5216,52161 \
  --batch-size 128 \
  --nbm-max-epochs 400 \
  --nbm-patience 40 \
  --tcn-max-epochs 30 \
  --tcn-patience 8 \
  --phase full \
  "$@"
