#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/scratch2/james2602/TextAwareIR"
DIFFBIR_DIR="${PROJECT_DIR}/DiffBIR"
PRED_PYTHON_BIN="${PRED_PYTHON_BIN:-/home/james2602/miniconda3/envs/dataset_curation/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
LEVEL="${SA_TEXT_LEVEL:-2}"
RESULTS_ROOT="${RESULTS_ROOT:-${DIFFBIR_DIR}/results/sa_text_test/lv${LEVEL}_a6000}"
OUTPUT_BASE="${OUTPUT_BASE:-${RESULTS_ROOT}/text_annotations}"
CONFIG_PATH="${CONFIG_PATH:-${DIFFBIR_DIR}/slurm/helpers/sa_text_annotation_config.yaml}"
GT_PARQUET="${GT_PARQUET:-${PROJECT_DIR}/SA-Text-test/data/test-00000-of-00001.parquet}"
CHUNKS="${CHUNKS:-chunk_0,chunk_1}"
LOG_STAMP="${LOG_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"

mkdir -p "${DIFFBIR_DIR}/slurm/logs"

detect_visible_gpu_count() {
  "${PRED_PYTHON_BIN}" - <<'PY'
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
    elif [[ "${visible_count}" =~ ^[0-9]+$ ]] && [[ "${visible_count}" -ge 1 ]]; then
      RESOLVED_GPU_IDS=(0)
    else
      RESOLVED_GPU_IDS=()
    fi
  fi

  if [[ "${#RESOLVED_GPU_IDS[@]}" -lt 1 ]]; then
    echo "Need at least one visible GPU for eval." >&2
    echo "Current CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}' GPU_IDS='${GPU_IDS:-unset}'" >&2
    echo "Run this from a shell/step with a GPU, or set GPU_IDS=0 explicitly." >&2
    exit 2
  fi
}

parse_chunks() {
  IFS=',' read -r -a REQUESTED_CHUNKS <<< "${CHUNKS}"
  if [[ "${#REQUESTED_CHUNKS[@]}" -lt 1 ]]; then
    echo "Need at least one chunk in CHUNKS." >&2
    exit 2
  fi
}

run_chunk() {
  local gpu_id="$1"
  local chunk_name="$2"
  local output_dir="${OUTPUT_BASE}/${chunk_name}"
  local log="${DIFFBIR_DIR}/slurm/logs/eval_${LOG_STAMP}_${chunk_name}.log"

  echo "[GPU ${gpu_id}] ${chunk_name}: annotation + evaluation"
  echo "Log: ${log}"
  {
    cd "${PROJECT_DIR}"
    echo "Log stamp: ${LOG_STAMP}"
    echo "Host: $(hostname)"
    echo "Chunk: ${chunk_name}"
    echo "Assigned GPU id: ${gpu_id}"
    echo "Parent CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PRED_PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

print(f"Child CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: this chunk cannot see a CUDA GPU before Bridge starts.", file=sys.stderr)
    sys.exit(2)
print(f"torch cuda device 0: {torch.cuda.get_device_name(0)}")
try:
    probe = torch.empty((1,), device="cuda")
    probe.fill_(1)
    torch.cuda.synchronize()
    del probe
    print("torch cuda allocation probe: ok")
except Exception as exc:
    print(f"ERROR: CUDA is visible but allocation failed before Bridge starts: {exc}", file=sys.stderr)
    sys.exit(2)
PY
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PRED_PYTHON_BIN}" \
      "${DIFFBIR_DIR}/slurm/helpers/extract_sa_text_annotations.py" \
      --results-root "${RESULTS_ROOT}" \
      --chunks "${chunk_name}" \
      --output-dir "${output_dir}" \
      --config "${CONFIG_PATH}"

    "${EVAL_PYTHON_BIN}" "${DIFFBIR_DIR}/evaluate_text_performance.py" \
      --gt-parquet "${GT_PARQUET}" \
      --pred-json "${output_dir}/agreed_annotations.json" \
      --output-dir "${output_dir}/evaluation" \
      --pred-text-field VLM \
      --pred-bbox-format xyxy \
      --image-id-list "${output_dir}/image_id_list.txt"
  } > "${log}" 2>&1
}

resolve_gpu_ids
parse_chunks

if [[ "${#RESOLVED_GPU_IDS[@]}" -ge "${#REQUESTED_CHUNKS[@]}" ]] && [[ "${#REQUESTED_CHUNKS[@]}" -gt 1 ]]; then
  echo "Using GPUs in parallel: GPUs=${RESOLVED_GPU_IDS[*]} chunks=${REQUESTED_CHUNKS[*]}"
  PIDS=()
  for i in "${!REQUESTED_CHUNKS[@]}"; do
    run_chunk "${RESOLVED_GPU_IDS[$i]}" "${REQUESTED_CHUNKS[$i]}" &
    PIDS+=("$!")
  done

  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    echo "${REQUESTED_CHUNKS[$i]} evaluation done"
  done
else
  echo "Using one GPU sequentially: GPU=${RESOLVED_GPU_IDS[0]} chunks=${REQUESTED_CHUNKS[*]}"
  for chunk_name in "${REQUESTED_CHUNKS[@]}"; do
    run_chunk "${RESOLVED_GPU_IDS[0]}" "${chunk_name}"
    echo "${chunk_name} evaluation done"
  done
fi

echo "evaluation complete: ${OUTPUT_BASE}"
