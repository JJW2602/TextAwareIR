#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/scratch2/james2602/TextAwareIR"
DIFFBIR_DIR="${PROJECT_DIR}/DiffBIR"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"

PARQUET_PATH="${SA_TEXT_PARQUET:-${PROJECT_DIR}/SA-Text-test/data/test-00000-of-00001.parquet}"
LEVEL="${SA_TEXT_LEVEL:-2}"
CHUNK_SIZE=500

RUN_NAME="${RUN_NAME:-sa_text_test/lv${LEVEL}_a6000}"
OUTPUT_ROOT="${DIFFBIR_DIR}/results/${RUN_NAME}"

UPSCALE="${DIFFBIR_UPSCALE:-4}"
STEPS="${DIFFBIR_STEPS:-10}"
CFG_SCALE="${DIFFBIR_CFG_SCALE:-6.0}"
CAPTIONER="${DIFFBIR_CAPTIONER:-llava}"
PRECISION="${DIFFBIR_PRECISION:-fp16}"
TILING_ENABLED="${DIFFBIR_TILING_ENABLED:-false}"

mkdir -p "${DIFFBIR_DIR}/slurm/logs" "${OUTPUT_ROOT}/manifests"

run_chunk() {
  local gpu_id=$1
  local task_id=$2
  local start_index=$((task_id * CHUNK_SIZE))

  local input_dir="${DIFFBIR_DIR}/inputs/sa_text_test/lv${LEVEL}/a6000_chunk_${task_id}"
  local output_dir="${OUTPUT_ROOT}/chunk_${task_id}"
  local manifest="${OUTPUT_ROOT}/manifests/chunk_${task_id}.csv"
  local log="${DIFFBIR_DIR}/slurm/logs/a6000_chunk_${task_id}.log"

  echo "[GPU${gpu_id}] chunk=${task_id}, images ${start_index}-$((start_index + CHUNK_SIZE - 1))"

  {
    "${PYTHON_BIN}" "${DIFFBIR_DIR}/slurm/export_sa_text_lq_chunk.py" \
      --parquet "${PARQUET_PATH}" \
      --level "${LEVEL}" \
      --start "${start_index}" \
      --count "${CHUNK_SIZE}" \
      --output-dir "${input_dir}" \
      --manifest "${manifest}" \
      --skip-existing

    cd "${DIFFBIR_DIR}"

    CUDA_VISIBLE_DEVICES=${gpu_id} "${PYTHON_BIN}" inference.py \
      task=sr \
      model.version=v2.1 \
      "upscale=${UPSCALE}" \
      "sampling.steps=${STEPS}" \
      "sampling.cfg_scale=${CFG_SCALE}" \
      "caption.name=${CAPTIONER}" \
      io.n_samples=1 \
      "io.input=${input_dir}" \
      "io.output=${output_dir}" \
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

    echo "[GPU${gpu_id}] done chunk_${task_id}"
  } > "${log}" 2>&1
}

echo "=== A6000 2-GPU parallel inference ==="
echo "GPU 0: images 0-499  → chunk_0"
echo "GPU 1: images 500-999 → chunk_1"
echo "logs: ${DIFFBIR_DIR}/slurm/logs/a6000_chunk_*.log"
echo ""

run_chunk 0 0 &
PID0=$!
run_chunk 1 1 &
PID1=$!

wait $PID0 && echo "GPU 0 완료" || echo "GPU 0 실패 (exit $?)"
wait $PID1 && echo "GPU 1 완료" || echo "GPU 1 실패 (exit $?)"

echo "=== 전체 완료 ==="
