#!/usr/bin/env bash
#SBATCH --job-name=tairl_ddpo_lora
#SBATCH --array=0-10
#SBATCH --partition=suma_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%A_%a_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%A_%a_%j.err

set -euo pipefail

ROOT_DIR="/scratch2/james2602/TextAwareIR"
TAIRL_DIR="${ROOT_DIR}/TAIRL"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/ddpo_lora.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/Results/TAIRL/ddpo_lora_hparam}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

TRIAL_NAMES=(
  "r4_lr1e-5_clip0.1_kl0.02"
  "r4_lr3e-5_clip0.1_kl0.02"
  "r4_lr1e-4_clip0.1_kl0.02"
  "r8_lr3e-5_clip0.1_kl0.02"
  "r8_lr3e-5_clip0.2_kl0.02"
  "r8_lr3e-5_clip0.1_kl0"
  "r16_lr3e-5_clip0.1_kl0.02"
  "r8_lr3e-5_raw_reward_miss0.5"
  "r8_lr3e-5_norm_miss0.5_fp0.25"
  "r8_lr3e-5_norm_miss1.0_fp0"
  "r8_lr3e-5_norm_miss1.0_fp0.5"
)
LORA_RANKS=(4 4 4 8 8 8 16 8 8 8 8)
LORA_ALPHAS=(4 4 4 8 8 8 16 8 8 8 8)
LRS=(1.0e-5 3.0e-5 1.0e-4 3.0e-5 3.0e-5 3.0e-5 3.0e-5 3.0e-5 3.0e-5 3.0e-5 3.0e-5)
CLIPS=(0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 0.1 0.1)
KLS=(0.02 0.02 0.02 0.02 0.02 0.0 0.02 0.02 0.02 0.02 0.02)
REWARD_VARIANTS=(final_reward_norm final_reward_norm final_reward_norm final_reward_norm final_reward_norm final_reward_norm final_reward_norm final_reward final_reward_norm final_reward_norm final_reward_norm)
MISS_PENALTIES=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.5 0.5 1.0 1.0)
FALSE_POSITIVE_PENALTIES=(0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.0 0.5)

if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#TRIAL_NAMES[@]}" ]]; then
  echo "Invalid task id ${TASK_ID}; expected 0-$(( ${#TRIAL_NAMES[@]} - 1 ))" >&2
  exit 2
fi

TRIAL="${TRIAL_NAMES[$TASK_ID]}"
OUT_DIR="${RESULTS_ROOT}/${TRIAL}"
export WANDB_DIR="${OUT_DIR}/wandb"
mkdir -p "${TAIRL_DIR}/slurm/logs" "${OUT_DIR}" "${WANDB_DIR}"

echo "Trial: ${TRIAL}"
echo "Output: ${OUT_DIR}"
echo "WANDB_DIR: ${WANDB_DIR}"
echo "WANDB_MODE: ${WANDB_MODE:-online}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${TAIRL_DIR}/train_ddpo_lora.py" \
  --config "${CONFIG}" \
  "train.output_dir=${OUT_DIR}" \
  "reward.work_dir=${OUT_DIR}/reward_work" \
  "lora.rank=${LORA_RANKS[$TASK_ID]}" \
  "lora.alpha=${LORA_ALPHAS[$TASK_ID]}" \
  "train.learning_rate=${LRS[$TASK_ID]}" \
  "ddpo.clip_range=${CLIPS[$TASK_ID]}" \
  "ddpo.kl_coef=${KLS[$TASK_ID]}" \
  "reward.variant=${REWARD_VARIANTS[$TASK_ID]}" \
  "reward.miss_penalty=${MISS_PENALTIES[$TASK_ID]}" \
  "reward.false_positive_penalty=${FALSE_POSITIVE_PENALTIES[$TASK_ID]}" \
  "wandb.run_name=${TRIAL}" \
  "wandb.group=ddpo_hparam"
