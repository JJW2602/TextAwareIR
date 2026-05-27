#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch2/james2602/TextAwareIR}"
TAIRL_DIR="${TAIRL_DIR:-${ROOT_DIR}/TAIRL}"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/grpo_lora_g6.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/Results/TAIRL/grpo_lora_g8/debug}"
LOG_DIR="${LOG_DIR:-${TAIRL_DIR}/slurm/logs}"

GPU="${GPU:-0}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GENERATION_MICROBATCH="${GENERATION_MICROBATCH:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
CKPT_EVERY="${CKPT_EVERY:-500}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_lora_g8_local}"
WANDB_GROUP="${WANDB_GROUP:-grpo}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${OUT_DIR}/wandb"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE
export WANDB_DIR="${OUT_DIR}/wandb"

echo "Algorithm: GRPO"
echo "Group size: ${GROUP_SIZE}"
echo "Generation microbatch: ${GENERATION_MICROBATCH}"
echo "Training source images: ${TRAIN_STEPS}/${MAX_SAMPLES}"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "WANDB_DIR: ${WANDB_DIR}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "WANDB_RUN_NAME: ${WANDB_RUN_NAME}"
echo "GPU: ${GPU}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" "${TAIRL_DIR}/train_grpo_ddpo_lora.py" \
  --config "${CONFIG}" \
  "data.max_samples=${MAX_SAMPLES}" \
  "train.train_steps=${TRAIN_STEPS}" \
  "train.ckpt_every=${CKPT_EVERY}" \
  "train.output_dir=${OUT_DIR}" \
  "train.progress=true" \
  "train.sample_progress=false" \
  "reward.work_dir=${OUT_DIR}/reward_work" \
  "grpo.group_size=${GROUP_SIZE}" \
  "grpo.generation_microbatch=${GENERATION_MICROBATCH}" \
  "wandb.log_images_every=1" \
  "wandb.num_log_images=${GROUP_SIZE}" \
  "wandb.log_tables_every=1" \
  "wandb.run_name=${WANDB_RUN_NAME}" \
  "wandb.group=${WANDB_GROUP}"
