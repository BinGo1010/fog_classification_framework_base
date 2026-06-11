#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-120}"

CONFIG="${CONFIG:-configs/base_experiment.yaml}"
RUN_NAME="${RUN_NAME:-cnn1d_loso_win120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
START_TASK_INDEX="${START_TASK_INDEX:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/document/home_mirror/chb/fog_classification_framework_base/fog_outputs/loso_6models_3methods_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}"
touch "${OUTPUT_ROOT}/.write_test"
rm -f "${OUTPUT_ROOT}/.write_test"

echo "Writing outputs to: ${OUTPUT_ROOT}"
echo "Starting from task index: ${START_TASK_INDEX}"

COMMON_ARGS="--config ${CONFIG} --run_name ${RUN_NAME} --exp_mode loso --batch_size ${BATCH_SIZE} --override data.num_workers=4 --override train.amp=true --override train.optimizer_foreach=false --override train.distributed_timeout_minutes=${DDP_TIMEOUT_MINUTES} --override train.ddp_find_unused_parameters=auto --override train.save_last_checkpoint=false --override train.checkpoint_include_optimizer=false --override train.checkpoint_include_scheduler=false"

TASK_NAMES=(
  "loso_itransformer_ordinary"
  "loso_timesnet_ordinary"
  "loso_nonstationary_transformer_ordinary"
  "loso_informer_ordinary"
  "loso_autoformer_ordinary"
  "loso_vit_ordinary"
  "loso_itransformer_supcon"
  "loso_timesnet_supcon"
  "loso_nonstationary_transformer_supcon"
  "loso_informer_supcon"
  "loso_autoformer_supcon"
  "loso_vit_supcon"
  "loso_itransformer_simclr"
  "loso_timesnet_simclr"
  "loso_nonstationary_transformer_simclr"
  "loso_informer_simclr"
  "loso_autoformer_simclr"
  "loso_vit_simclr"
)

TASK_MODELS=(
  "iTransformer"
  "TimesNet"
  "NonstationaryTransformer"
  "Informer"
  "Autoformer"
  "ViT"
  "SupConITransformer"
  "SupConTimesNet"
  "SupConNonstationaryTransformer"
  "SupConInformer"
  "SupConAutoformer"
  "SupConViT"
  "SimCLRITransformer"
  "SimCLRTimesNet"
  "SimCLRNonstationaryTransformer"
  "SimCLRInformer"
  "SimCLRAutoformer"
  "SimCLRViT"
)

TOTAL_TASKS="${#TASK_NAMES[@]}"

for index in $(seq 1 "${TOTAL_TASKS}"); do
  task_array_index=$((index - 1))
  task_name="${TASK_NAMES[$task_array_index]}"
  model_name="${TASK_MODELS[$task_array_index]}"

  if [ "${index}" -lt "${START_TASK_INDEX}" ]; then
    echo "Skipping task ${index}/${TOTAL_TASKS}: ${task_name}"
    continue
  fi

  echo ""
  echo "===== Running task ${index}/${TOTAL_TASKS}: ${task_name} (${model_name}) ====="
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py ${COMMON_ARGS} \
    --model "${model_name}" \
    --model_id "${task_name}" \
    --override "project.name=${task_name}" \
    --output_dir "${OUTPUT_ROOT}/${task_name}"
done

echo ""
echo "All requested tasks finished."
