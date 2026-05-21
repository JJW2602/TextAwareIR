#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/scratch2/james2602/TextAwareIR"
DIFFBIR_DIR="${PROJECT_DIR}/DiffBIR"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"

PARQUET_PATH="${SA_TEXT_PARQUET:-${PROJECT_DIR}/SA-Text-test/data/test-00000-of-00001.parquet}"
LEVEL="${SA_TEXT_LEVEL:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
RUN_NAME="${RUN_NAME:-sa_text_test/lv${LEVEL}_2gpu}"
OUTPUT_ROOT="${DIFFBIR_DIR}/results/${RUN_NAME}"

UPSCALE="${DIFFBIR_UPSCALE:-4}"
STEPS="${DIFFBIR_STEPS:-10}"
CFG_SCALE="${DIFFBIR_CFG_SCALE:-6.0}"
CAPTIONER="${DIFFBIR_CAPTIONER:-llava}"
PRECISION="${DIFFBIR_PRECISION:-fp16}"
TILING_ENABLED="${DIFFBIR_TILING_ENABLED:-false}"
LOG_STAMP="${LOG_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"

mkdir -p "${DIFFBIR_DIR}/slurm/logs" "${OUTPUT_ROOT}/manifests"

detect_visible_gpu_count() {
  "${PYTHON_BIN}" - <<'PY'
import torch

try:
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
}

resolve_gpu_ids() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    IFS=',' read -r -a RESOLVED_GPU_IDS <<< "${GPU_IDS}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a RESOLVED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  else
    local visible_count
    visible_count="$(detect_visible_gpu_count)"
    if [[ "${visible_count}" =~ ^[0-9]+$ ]] && [[ "${visible_count}" -ge 2 ]]; then
      RESOLVED_GPU_IDS=(0 1)
    else
      RESOLVED_GPU_IDS=()
    fi
  fi

  if [[ "${#RESOLVED_GPU_IDS[@]}" -lt 2 ]]; then
    echo "Need two visible GPUs for 2-GPU inference." >&2
    echo "Current CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}' GPU_IDS='${GPU_IDS:-unset}'" >&2
    echo "Run this from a shell/step with two GPUs, or set GPU_IDS=0,1 explicitly." >&2
    exit 2
  fi
}

run_chunk() {
  local gpu_id="$1"
  local task_id="$2"
  local start_index=$((task_id * CHUNK_SIZE))
  local input_dir="${DIFFBIR_DIR}/inputs/sa_text_test/lv${LEVEL}/2gpu_chunk_${task_id}"
  local output_dir="${OUTPUT_ROOT}/chunk_${task_id}"
  local manifest="${OUTPUT_ROOT}/manifests/chunk_${task_id}.csv"
  local log="${DIFFBIR_DIR}/slurm/logs/infer_${LOG_STAMP}_chunk_${task_id}.log"

  echo "[GPU ${gpu_id}] chunk_${task_id}: rows ${start_index}-$((start_index + CHUNK_SIZE - 1))"
  echo "Log: ${log}"
  {
    echo "Log stamp: ${LOG_STAMP}"
    echo "Host: $(hostname)"
    echo "Chunk: chunk_${task_id}"
    echo "Assigned GPU id: ${gpu_id}"
    echo "Parent CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

print(f"Child CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: this chunk cannot see a CUDA GPU before DiffBIR starts.", file=sys.stderr)
    sys.exit(2)
print(f"torch cuda device 0: {torch.cuda.get_device_name(0)}")
try:
    probe = torch.empty((1,), device="cuda")
    probe.fill_(1)
    torch.cuda.synchronize()
    del probe
    print("torch cuda allocation probe: ok")
except Exception as exc:
    print(f"ERROR: CUDA is visible but allocation failed before DiffBIR starts: {exc}", file=sys.stderr)
    sys.exit(2)
PY
    "${PYTHON_BIN}" "${DIFFBIR_DIR}/slurm/helpers/export_sa_text_lq_chunk.py" \
      --parquet "${PARQUET_PATH}" \
      --level "${LEVEL}" \
      --start "${start_index}" \
      --count "${CHUNK_SIZE}" \
      --output-dir "${input_dir}" \
      --manifest "${manifest}" \
      --skip-existing

    cd "${DIFFBIR_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" inference.py \
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
  } > "${log}" 2>&1
}

resolve_gpu_ids
echo "Using GPUs: chunk_0=${RESOLVED_GPU_IDS[0]}, chunk_1=${RESOLVED_GPU_IDS[1]}"

run_chunk "${RESOLVED_GPU_IDS[0]}" 0 &
PID0=$!
run_chunk "${RESOLVED_GPU_IDS[1]}" 1 &
PID1=$!

wait "${PID0}"
echo "chunk_0 inference done"
wait "${PID1}"
echo "chunk_1 inference done"
echo "2-GPU inference complete: ${OUTPUT_ROOT}"
