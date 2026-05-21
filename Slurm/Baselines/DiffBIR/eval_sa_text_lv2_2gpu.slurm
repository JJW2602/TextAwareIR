#!/usr/bin/env bash
#SBATCH --job-name=sat_eval_2gpu
#SBATCH --array=0-1
#SBATCH --partition=suma_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/Slurm/Baselines/DiffBIR/logs/%x_%A_%a_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/Slurm/Baselines/DiffBIR/logs/%x_%A_%a_%j.err

set -euo pipefail

ROOT_DIR="/scratch2/james2602/TextAwareIR"
DIFFBIR_DIR="${ROOT_DIR}/DiffBIR"
RESULTS_DIR="${ROOT_DIR}/Results"
SLURM_DIR="${ROOT_DIR}/Slurm/Baselines/DiffBIR"
PRED_PYTHON_BIN="${PRED_PYTHON_BIN:-/home/james2602/miniconda3/envs/dataset_curation/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
LEVEL="${SA_TEXT_LEVEL:-2}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK_NAME="chunk_${TASK_ID}"
RESULTS_ROOT="${RESULTS_ROOT:-${RESULTS_DIR}/Baselines/DiffBIR/sa_text_test_lv${LEVEL}_a6000}"
OUTPUT_BASE="${OUTPUT_BASE:-${RESULTS_ROOT}/text_annotations}"
OUTPUT_DIR="${OUTPUT_BASE}/${CHUNK_NAME}"
CONFIG_PATH="${CONFIG_PATH:-${SLURM_DIR}/helpers/sa_text_annotation_config.yaml}"
GT_PARQUET="${GT_PARQUET:-${ROOT_DIR}/Dataset/SA-Text-test/data/test-00000-of-00001.parquet}"
LOG_STAMP="${LOG_STAMP:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}}"
RUN_LOG="${SLURM_DIR}/logs/eval_${LOG_STAMP}_${CHUNK_NAME}.log"

mkdir -p "${SLURM_DIR}/logs"
cd "${ROOT_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Log: ${RUN_LOG}"
echo "Host: $(hostname)"
echo "Chunk: ${CHUNK_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"${PRED_PYTHON_BIN}" "${SLURM_DIR}/helpers/extract_sa_text_annotations.py" \
  --results-root "${RESULTS_ROOT}" \
  --chunks "${CHUNK_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG_PATH}"

"${EVAL_PYTHON_BIN}" "${DIFFBIR_DIR}/evaluate_text_performance.py" \
  --gt-parquet "${GT_PARQUET}" \
  --pred-json "${OUTPUT_DIR}/agreed_annotations.json" \
  --output-dir "${OUTPUT_DIR}/evaluation" \
  --pred-text-field VLM \
  --pred-bbox-format xyxy \
  --image-id-list "${OUTPUT_DIR}/image_id_list.txt"

echo "done ${CHUNK_NAME}"
