#!/usr/bin/env bash
#SBATCH --job-name=diffbir_2gpu
#SBATCH --array=0-1
#SBATCH --partition=suma_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/DiffBIR/slurm/logs/%x_%A_%a_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/DiffBIR/slurm/logs/%x_%A_%a_%j.err

set -euo pipefail

PROJECT_DIR="/scratch2/james2602/TextAwareIR"
DIFFBIR_DIR="${PROJECT_DIR}/DiffBIR"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"

PARQUET_PATH="${SA_TEXT_PARQUET:-${PROJECT_DIR}/SA-Text-test/data/test-00000-of-00001.parquet}"
LEVEL="${SA_TEXT_LEVEL:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX=$((TASK_ID * CHUNK_SIZE))
RUN_NAME="${RUN_NAME:-sa_text_test/lv${LEVEL}_2gpu}"
INPUT_DIR="${DIFFBIR_DIR}/inputs/sa_text_test/lv${LEVEL}/2gpu_chunk_${TASK_ID}"
OUTPUT_ROOT="${DIFFBIR_DIR}/results/${RUN_NAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/chunk_${TASK_ID}"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/chunk_${TASK_ID}.csv"

UPSCALE="${DIFFBIR_UPSCALE:-4}"
STEPS="${DIFFBIR_STEPS:-10}"
CFG_SCALE="${DIFFBIR_CFG_SCALE:-6.0}"
CAPTIONER="${DIFFBIR_CAPTIONER:-llava}"
PRECISION="${DIFFBIR_PRECISION:-fp16}"
TILING_ENABLED="${DIFFBIR_TILING_ENABLED:-false}"
LOG_STAMP="${LOG_STAMP:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}}"
RUN_LOG="${DIFFBIR_DIR}/slurm/logs/infer_${LOG_STAMP}_chunk_${TASK_ID}.log"

mkdir -p "${DIFFBIR_DIR}/slurm/logs" "${OUTPUT_ROOT}/manifests"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Log: ${RUN_LOG}"
echo "Host: $(hostname)"
echo "Chunk: chunk_${TASK_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"${PYTHON_BIN}" "${DIFFBIR_DIR}/slurm/helpers/export_sa_text_lq_chunk.py" \
  --parquet "${PARQUET_PATH}" \
  --level "${LEVEL}" \
  --start "${START_INDEX}" \
  --count "${CHUNK_SIZE}" \
  --output-dir "${INPUT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --skip-existing

cd "${DIFFBIR_DIR}"
"${PYTHON_BIN}" inference.py \
  task=sr \
  model.version=v2.1 \
  "upscale=${UPSCALE}" \
  "sampling.steps=${STEPS}" \
  "sampling.cfg_scale=${CFG_SCALE}" \
  "caption.name=${CAPTIONER}" \
  io.n_samples=1 \
  "io.input=${INPUT_DIR}" \
  "io.output=${OUTPUT_DIR}" \
  runtime.device=cuda \
  "runtime.precision=${PRECISION}" \
  runtime.batch_size=1 \
  "tiling.cleaner.enabled=${TILING_ENABLED}" \
  tiling.cleaner.tile_size=512 \
  tiling.cleaner.tile_stride=256 \
  "tiling.vae_encoder.enabled=${TILING_ENABLED}" \
  tiling.vae_encoder.tile_size=256 \
  "tiling.vae_decoder.enabled=${TILING_ENABLED}" \
  tiling.vae_decoder.tile_size=256 \
  "tiling.cldm.enabled=${TILING_ENABLED}" \
  tiling.cldm.tile_size=512 \
  tiling.cldm.tile_stride=256

echo "done chunk_${TASK_ID}"
