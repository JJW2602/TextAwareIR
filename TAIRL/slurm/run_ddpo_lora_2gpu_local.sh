#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch2/james2602/TextAwareIR}"
TAIRL_DIR="${TAIRL_DIR:-${ROOT_DIR}/TAIRL}"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/ddpo_lora.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/Results/TAIRL/ddpo_lora_2gpu_local}"
LOG_DIR="${LOG_DIR:-${TAIRL_DIR}/slurm/logs}"

MAX_SAMPLES="${MAX_SAMPLES:-2000}"
TRAIN_STEPS="${TRAIN_STEPS:-1200}"
SAMPLE_STEPS="${SAMPLE_STEPS:-8}"
CKPT_EVERY="${CKPT_EVERY:-100}"
WANDB_MODE="${WANDB_MODE:-online}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

TRIAL_NAMES=(
  "r8_lr3e-5_clip0.1_kl0.02"
  "r8_lr3e-5_clip0.2_kl0.02"
)
GPUS=("${GPU0}" "${GPU1}")
CLIPS=(0.1 0.2)

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

echo "ROOT_DIR: ${ROOT_DIR}"
echo "CONFIG: ${CONFIG}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "TRAIN_STEPS: ${TRAIN_STEPS}"
echo "SAMPLE_STEPS: ${SAMPLE_STEPS}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "GPU mapping: ${TRIAL_NAMES[0]} -> ${GPU0}, ${TRIAL_NAMES[1]} -> ${GPU1}"

pids=()
log_files=()

run_trial() {
  local idx="$1"
  local trial="${TRIAL_NAMES[$idx]}"
  local gpu="${GPUS[$idx]}"
  local clip="${CLIPS[$idx]}"
  local out_dir="${RESULTS_ROOT}/${trial}"
  local log_file="${LOG_DIR}/ddpo_lora_2gpu_${trial}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "${out_dir}" "${out_dir}/wandb"
  log_files[$idx]="${log_file}"

  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export WANDB_MODE
    export WANDB_DIR="${out_dir}/wandb"

    echo "Started: $(date)"
    echo "Trial: ${trial}"
    echo "GPU: ${gpu}"
    echo "Output: ${out_dir}"
    echo "WANDB_DIR: ${WANDB_DIR}"
    echo "Host: $(hostname)"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    cd "${ROOT_DIR}"
    exec "${PYTHON_BIN}" "${TAIRL_DIR}/train_grpo_ddpo_lora.py" \
      --config "${CONFIG}" \
      "data.max_samples=${MAX_SAMPLES}" \
      "sample.steps=${SAMPLE_STEPS}" \
      "train.batch_size=1" \
      "train.train_steps=${TRAIN_STEPS}" \
      "train.ckpt_every=${CKPT_EVERY}" \
      "train.output_dir=${out_dir}" \
      "reward.work_dir=${out_dir}/reward_work" \
      "lora.rank=8" \
      "lora.alpha=8" \
      "train.learning_rate=3.0e-5" \
      "ddpo.clip_range=${clip}" \
      "ddpo.kl_coef=0.02" \
      "reward.variant=final_reward_norm" \
      "reward.miss_penalty=1.0" \
      "reward.false_positive_penalty=0.25" \
      "wandb.run_name=${trial}_local_${TRAIN_STEPS}steps" \
      "wandb.group=ddpo_2gpu_local"
  ) >"${log_file}" 2>&1 &

  pids[$idx]="$!"
  echo "Launched ${trial} on GPU ${gpu}: pid=${pids[$idx]}, log=${log_file}"
}

stop_children() {
  echo "Stopping child processes..."
  for pid in "${pids[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap stop_children INT TERM

run_trial 0
run_trial 1

status=0
for idx in "${!pids[@]}"; do
  trial="${TRIAL_NAMES[$idx]}"
  if wait "${pids[$idx]}"; then
    echo "Completed ${trial}. Log: ${log_files[$idx]}"
  else
    rc="$?"
    echo "Failed ${trial} with exit code ${rc}. Log: ${log_files[$idx]}" >&2
    status="${rc}"
  fi
done

exit "${status}"
