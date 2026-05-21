#!/usr/bin/env bash
#SBATCH --job-name=tairl_grpo_g6
#SBATCH --partition=suma_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/scratch2/james2602/TextAwareIR"
TAIRL_DIR="${ROOT_DIR}/TAIRL"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/grpo_lora_g6.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/Results/TAIRL/grpo_lora_g6/debug}"
export WANDB_DIR="${OUT_DIR}/wandb"

mkdir -p "${TAIRL_DIR}/slurm/logs" "${OUT_DIR}" "${WANDB_DIR}"

echo "Algorithm: GRPO"
echo "Group size: 6"
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
  "wandb.run_name=grpo_lora_g6" \
  "wandb.group=grpo"
