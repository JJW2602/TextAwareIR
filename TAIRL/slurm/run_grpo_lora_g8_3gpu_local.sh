#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch2/james2602/TextAwareIR}"
TAIRL_DIR="${TAIRL_DIR:-${ROOT_DIR}/TAIRL}"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/grpo_lora_g6.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/Results/TAIRL/grpo_lora_g8_3gpu_local}"
LOG_DIR="${LOG_DIR:-${TAIRL_DIR}/slurm/logs}"

GPU_BASE="${GPU_BASE:-1}"
GPU_STABLE="${GPU_STABLE:-2}"
GPU_PSNR="${GPU_PSNR:-3}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GENERATION_MICROBATCH="${GENERATION_MICROBATCH:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
CKPT_EVERY="${CKPT_EVERY:-500}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-grpo_g8_3gpu_local}"
PSNR_WEIGHT="${PSNR_WEIGHT:-0.01}"
PSNR_NORM_DENOMINATOR="${PSNR_NORM_DENOMINATOR:-40.0}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE

timestamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch_run() {
  local gpu="$1"
  local run_id="$2"
  local learning_rate="$3"
  local clip_range="$4"
  local kl_coef="$5"
  local psnr_weight="$6"
  local out_dir="${OUT_ROOT}/${run_id}"
  local log_file="${LOG_DIR}/grpo_g8_3gpu_${run_id}_gpu${gpu}_${timestamp}.log"

  mkdir -p "${out_dir}/wandb"
  echo "Launching ${run_id} on GPU ${gpu}; log=${log_file}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export WANDB_DIR="${out_dir}/wandb"
    cd "${ROOT_DIR}"
    exec "${PYTHON_BIN}" "${TAIRL_DIR}/train_grpo_ddpo_lora.py" \
      --config "${CONFIG}" \
      "data.max_samples=${MAX_SAMPLES}" \
      "train.train_steps=${TRAIN_STEPS}" \
      "train.ckpt_every=${CKPT_EVERY}" \
      "train.output_dir=${out_dir}" \
      "train.learning_rate=${learning_rate}" \
      "train.progress=true" \
      "train.sample_progress=false" \
      "reward.work_dir=${out_dir}/reward_work" \
      "reward.psnr_weight=${psnr_weight}" \
      "reward.psnr_norm_denominator=${PSNR_NORM_DENOMINATOR}" \
      "grpo.group_size=${GROUP_SIZE}" \
      "grpo.generation_microbatch=${GENERATION_MICROBATCH}" \
      "grpo.clip_range=${clip_range}" \
      "grpo.kl_coef=${kl_coef}" \
      "wandb.log_images_every=1" \
      "wandb.num_log_images=${GROUP_SIZE}" \
      "wandb.log_tables_every=1" \
      "wandb.run_name=${run_id}" \
      "wandb.group=${WANDB_GROUP}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

launch_run "${GPU_BASE}" "g8_base_lr3e-5_clip0.1_kl0.02" "3.0e-5" "0.1" "0.02" "0.0"
launch_run "${GPU_STABLE}" "g8_stable_lr1e-5_clip0.1_kl0.05" "1.0e-5" "0.1" "0.05" "0.0"
launch_run "${GPU_PSNR}" "g8_psnr_lr3e-5_clip0.1_kl0.02_psnr${PSNR_WEIGHT}" "3.0e-5" "0.1" "0.02" "${PSNR_WEIGHT}"

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
