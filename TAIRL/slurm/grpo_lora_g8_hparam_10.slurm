#!/usr/bin/env bash
#SBATCH --job-name=tairl_grpo_g8_hp10
#SBATCH --array=0-9
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%A_%a_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%A_%a_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch2/james2602/TextAwareIR}"
TAIRL_DIR="${TAIRL_DIR:-${ROOT_DIR}/TAIRL}"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/grpo_lora_g6.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/Results/TAIRL/grpo_lora_g8_hparam_10}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

GROUP_SIZE="${GROUP_SIZE:-8}"
GENERATION_MICROBATCH="${GENERATION_MICROBATCH:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
CKPT_EVERY="${CKPT_EVERY:-500}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-1}"
LOG_TABLES_EVERY="${LOG_TABLES_EVERY:-1}"
NUM_LOG_IMAGES="${NUM_LOG_IMAGES:-${GROUP_SIZE}}"
WANDB_GROUP="${WANDB_GROUP:-grpo_g8_hparam_10}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

TRIAL_NAMES=(
  "base_default_r8_lr3e-5_clip0.1_kl0.02"
  "base_lr1e-5_r8_clip0.1_kl0.02"
  "base_clip0.2_r8_lr3e-5_kl0.02"
  "base_kl0.05_r8_lr3e-5_clip0.1"
  "base_r16_lr3e-5_clip0.1_kl0.02"
  "psnr010_default_r8_lr3e-5_clip0.1_kl0.02"
  "psnr010_lr1e-5_r8_clip0.1_kl0.02"
  "psnr010_clip0.2_r8_lr3e-5_kl0.02"
  "psnr010_kl0.05_r8_lr3e-5_clip0.1"
  "psnr020_r8_lr3e-5_clip0.1_kl0.02"
)

RATIONALES=(
  "Default base reward setting: known feasible G8, LR 3e-5, clip 0.1, KL 0.02."
  "Only learning rate is changed lower to measure stability versus update speed."
  "Only PPO clip range is changed wider to measure whether larger GRPO updates help sparse OCR reward."
  "Only reference KL coefficient is changed higher to measure policy-drift control."
  "Only LoRA rank/alpha are changed higher to measure whether more capacity improves text adaptation."
  "Default PSNR reward setting: same best base hyperparameters plus a tiny PSNR bonus."
  "Only learning rate is changed lower under PSNR reward to measure stability versus update speed."
  "Only PPO clip range is changed wider under PSNR reward to measure update-size sensitivity."
  "Only reference KL coefficient is changed higher under PSNR reward to measure drift control."
  "Only PSNR weight is changed higher to measure fidelity-bonus sensitivity."
)

LORA_RANKS=(8 8 8 8 16 8 8 8 8 8)
LORA_ALPHAS=(8 8 8 8 16 8 8 8 8 8)
LRS=(3.0e-5 1.0e-5 3.0e-5 3.0e-5 3.0e-5 3.0e-5 1.0e-5 3.0e-5 3.0e-5 3.0e-5)
CLIPS=(0.1 0.1 0.2 0.1 0.1 0.1 0.1 0.2 0.1 0.1)
KLS=(0.02 0.02 0.02 0.05 0.02 0.02 0.02 0.02 0.05 0.02)
PSNR_WEIGHTS=(0.0 0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.02)
PSNR_NORM_DENOMINATORS=(40.0 40.0 40.0 40.0 40.0 40.0 40.0 40.0 40.0 40.0)

if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#TRIAL_NAMES[@]}" ]]; then
  echo "Invalid task id ${TASK_ID}; expected 0-$(( ${#TRIAL_NAMES[@]} - 1 ))" >&2
  exit 2
fi

TRIAL="${TRIAL_NAMES[$TASK_ID]}"
OUT_DIR="${RESULTS_ROOT}/${TRIAL}"
export WANDB_DIR="${OUT_DIR}/wandb"

mkdir -p "${TAIRL_DIR}/slurm/logs" "${OUT_DIR}" "${WANDB_DIR}"

echo "Trial: ${TRIAL}"
echo "Rationale: ${RATIONALES[$TASK_ID]}"
echo "Task ID: ${TASK_ID}"
echo "Partitions: suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada"
echo "Output: ${OUT_DIR}"
echo "WANDB_DIR: ${WANDB_DIR}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "WANDB_GROUP: ${WANDB_GROUP}"
echo "Group size: ${GROUP_SIZE}"
echo "Generation microbatch: ${GENERATION_MICROBATCH}"
echo "Training source images: ${TRAIN_STEPS}/${MAX_SAMPLES}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${TAIRL_DIR}/train_grpo_ddpo_lora.py" \
  --config "${CONFIG}" \
  "data.max_samples=${MAX_SAMPLES}" \
  "train.train_steps=${TRAIN_STEPS}" \
  "train.ckpt_every=${CKPT_EVERY}" \
  "train.output_dir=${OUT_DIR}" \
  "train.learning_rate=${LRS[$TASK_ID]}" \
  "train.progress=true" \
  "train.sample_progress=false" \
  "lora.rank=${LORA_RANKS[$TASK_ID]}" \
  "lora.alpha=${LORA_ALPHAS[$TASK_ID]}" \
  "reward.variant=final_reward_norm" \
  "reward.miss_penalty=1.0" \
  "reward.false_positive_penalty=0.25" \
  "reward.work_dir=${OUT_DIR}/reward_work" \
  "reward.psnr_weight=${PSNR_WEIGHTS[$TASK_ID]}" \
  "reward.psnr_norm_denominator=${PSNR_NORM_DENOMINATORS[$TASK_ID]}" \
  "grpo.group_size=${GROUP_SIZE}" \
  "grpo.generation_microbatch=${GENERATION_MICROBATCH}" \
  "grpo.clip_range=${CLIPS[$TASK_ID]}" \
  "grpo.kl_coef=${KLS[$TASK_ID]}" \
  "wandb.log_images_every=${LOG_IMAGES_EVERY}" \
  "wandb.num_log_images=${NUM_LOG_IMAGES}" \
  "wandb.log_tables_every=${LOG_TABLES_EVERY}" \
  "wandb.run_name=${TRIAL}" \
  "wandb.group=${WANDB_GROUP}"
