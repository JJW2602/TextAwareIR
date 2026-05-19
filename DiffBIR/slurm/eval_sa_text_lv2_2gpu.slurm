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
#SBATCH --output=/scratch2/james2602/TextAwareIR/DiffBIR/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/DiffBIR/slurm/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/scratch2/james2602/TextAwareIR"
DIFFBIR_DIR="${PROJECT_DIR}/DiffBIR"
PRED_PYTHON_BIN="${PRED_PYTHON_BIN:-/home/james2602/miniconda3/envs/dataset_curation/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
LEVEL="${SA_TEXT_LEVEL:-2}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK_NAME="chunk_${TASK_ID}"
RESULTS_ROOT="${RESULTS_ROOT:-${DIFFBIR_DIR}/results/sa_text_test/lv${LEVEL}_a6000}"
OUTPUT_BASE="${OUTPUT_BASE:-${RESULTS_ROOT}/text_annotations}"
OUTPUT_DIR="${OUTPUT_BASE}/${CHUNK_NAME}"
CONFIG_PATH="${CONFIG_PATH:-${DIFFBIR_DIR}/slurm/helpers/sa_text_annotation_config.yaml}"
GT_PARQUET="${GT_PARQUET:-${PROJECT_DIR}/SA-Text-test/data/test-00000-of-00001.parquet}"

mkdir -p "${DIFFBIR_DIR}/slurm/logs"
cd "${PROJECT_DIR}"

"${PRED_PYTHON_BIN}" "${DIFFBIR_DIR}/slurm/helpers/extract_sa_text_annotations.py" \
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
