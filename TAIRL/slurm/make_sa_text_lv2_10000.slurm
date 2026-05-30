#!/usr/bin/env bash
#SBATCH --job-name=make_sa_text_lv2_10k
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/TAIRL/slurm/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch2/james2602/TextAwareIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/TAIRL/train_data/sa_text_train_10000.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/Dataset/SA-Text-lv2-10000}"
COUNT="${COUNT:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
LQ_SIZE="${LQ_SIZE:-128}"
SEED="${SEED:-231}"

mkdir -p "${ROOT_DIR}/TAIRL/slurm/logs" "${OUTPUT_DIR}"

echo "Manifest: ${MANIFEST}"
echo "Output: ${OUTPUT_DIR}"
echo "Count: ${COUNT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Shard size: ${SHARD_SIZE}"
echo "LQ size: ${LQ_SIZE}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/TAIRL/tools/make_sa_text_lv2_pairs.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --count "${COUNT}" \
  --batch-size "${BATCH_SIZE}" \
  --shard-size "${SHARD_SIZE}" \
  --lq-size "${LQ_SIZE}" \
  --seed "${SEED}" \
  --device auto \
  --overwrite
