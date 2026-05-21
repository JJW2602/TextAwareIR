#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/scratch2/james2602/TextAwareIR"
TAIR_DIR="${ROOT_DIR}/TAIR"
DIFFBIR_DIR="${ROOT_DIR}/DiffBIR"
RESULTS_DIR="${ROOT_DIR}/Results"
SLURM_DIR="${ROOT_DIR}/Slurm"
TAIR_SLURM_DIR="${SLURM_DIR}/Baselines/TAIR"
DIFFBIR_SLURM_DIR="${SLURM_DIR}/Baselines/DiffBIR"
TAIR_RESULTS_BASE="${RESULTS_DIR}/Baselines/TAIR"
COMPARE_RESULTS_BASE="${RESULTS_DIR}/Baselines/Compare"

TAIR_PYTHON_BIN="${TAIR_PYTHON_BIN:-/home/james2602/miniconda3/envs/tair/bin/python}"
DIFFBIR_PYTHON_BIN="${DIFFBIR_PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
PRED_PYTHON_BIN="${PRED_PYTHON_BIN:-/home/james2602/miniconda3/envs/dataset_curation/bin/python}"

LEVEL="${SA_TEXT_LEVEL:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
CHUNKS="${CHUNKS:-chunk_0,chunk_1}"
RUN_NAME="${RUN_NAME:-sa_text_test_lv${LEVEL}_chunks_0_1_stage3}"
RUN_ROOT="${RUN_ROOT:-${TAIR_RESULTS_BASE}/${RUN_NAME}}"
COMPARE_RUN_NAME="${COMPARE_RUN_NAME:-sa_text_test_lv${LEVEL}_chunks_0_1_tair_vs_diffbir}"
COMPARE_ROOT="${COMPARE_ROOT:-${COMPARE_RESULTS_BASE}/${COMPARE_RUN_NAME}}"
DIFFBIR_RESULTS_ROOT="${DIFFBIR_RESULTS_ROOT:-${RESULTS_DIR}/Baselines/DiffBIR/sa_text_test_lv${LEVEL}_a6000}"

GT_PARQUET="${GT_PARQUET:-${ROOT_DIR}/Dataset/SA-Text-test/data/test-00000-of-00001.parquet}"
ANNOT_CONFIG="${ANNOT_CONFIG:-${DIFFBIR_SLURM_DIR}/helpers/sa_text_annotation_config.yaml}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${ROOT_DIR}/Dataset_pipeline/SA-Text_Dataset}"
TAIR_CONFIG="${TAIR_CONFIG:-${TAIR_DIR}/configs/val/val_terediff.yaml}"
TAIR_INFER_CONFIG="${TAIR_INFER_CONFIG:-${TAIR_DIR}/configs/infer/infer_terediff.yaml}"
TAIR_CONFIG_TESTR="${TAIR_CONFIG_TESTR:-${TAIR_DIR}/testr/configs/TESTR/TESTR_R_50_Polygon.yaml}"
TAIR_STEPS="${TAIR_STEPS:-50}"
MISS_PENALTY="${MISS_PENALTY:-1.0}"
LOG_STAMP="${LOG_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"

LQ_ROOT="${RUN_ROOT}/inputs/lq${LEVEL}"
LQ_MANIFEST_ROOT="${RUN_ROOT}/inputs/manifests"
TAIR_RAW_ROOT="${RUN_ROOT}/tair_raw"
TAIR_HQ_ROOT="${RUN_ROOT}/tair_hq"
TAIR_ANNOT_ROOT="${RUN_ROOT}/text_annotations/tair"
EVAL_ROOT="${COMPARE_ROOT}/evaluation"
VIS_ROOT="${COMPARE_ROOT}/visualizations/gt_lq_tair_diffbir"

mkdir -p "${TAIR_SLURM_DIR}/logs" "${RUN_ROOT}" "${COMPARE_ROOT}" "${LQ_MANIFEST_ROOT}"

detect_visible_gpu_count() {
  "${TAIR_PYTHON_BIN}" - <<'PY'
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
    echo "Need at least one visible GPU for TAIR inference/annotation." >&2
    echo "Current CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}' GPU_IDS='${GPU_IDS:-unset}'" >&2
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

chunk_index() {
  local chunk_name="$1"
  if [[ ! "${chunk_name}" =~ ^chunk_[0-9]+$ ]]; then
    echo "Chunk must look like chunk_N, got '${chunk_name}'." >&2
    exit 2
  fi
  echo "${chunk_name#chunk_}"
}

run_chunk() {
  local gpu_id="$1"
  local chunk_name="$2"
  local task_id
  task_id="$(chunk_index "${chunk_name}")"
  local start_index=$((task_id * CHUNK_SIZE))
  local lq_dir="${LQ_ROOT}/${chunk_name}"
  local lq_manifest="${LQ_MANIFEST_ROOT}/${chunk_name}.csv"
  local raw_dir="${TAIR_RAW_ROOT}/${chunk_name}"
  local annot_dir="${TAIR_ANNOT_ROOT}/${chunk_name}"
  local log="${TAIR_SLURM_DIR}/logs/tair_sa_text_${LOG_STAMP}_${chunk_name}.log"

  echo "[GPU ${gpu_id}] ${chunk_name}: LQ export -> TAIR inference -> normalize -> annotation"
  echo "Log: ${log}"
  {
    cd "${ROOT_DIR}"
    echo "Log stamp: ${LOG_STAMP}"
    echo "Host: $(hostname)"
    echo "Chunk: ${chunk_name}"
    echo "Assigned GPU id: ${gpu_id}"
    echo "Run root: ${RUN_ROOT}"
    echo "Parent CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
    fi

    "${DIFFBIR_PYTHON_BIN}" "${DIFFBIR_SLURM_DIR}/helpers/export_sa_text_lq_chunk.py" \
      --parquet "${GT_PARQUET}" \
      --level "${LEVEL}" \
      --start "${start_index}" \
      --count "${CHUNK_SIZE}" \
      --output-dir "${lq_dir}" \
      --manifest "${lq_manifest}" \
      --skip-existing

    cd "${TAIR_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${TAIR_PYTHON_BIN}" infer.py \
      --config "${TAIR_CONFIG}" \
      --infer-config "${TAIR_INFER_CONFIG}" \
      --config_testr "${TAIR_CONFIG_TESTR}" \
      --input "${lq_dir}" \
      --output-dir "${raw_dir}" \
      patch.enabled=false \
      io.output_size=model \
      io.save_cleaned=false \
      visualization.save_prompt_traces=false \
      visualization.save_pred_text_images=false \
      visualization.save_patch_text_regions=false \
      visualization.visualize_patches=false \
      visualization.visualize_text_regions=false \
      sampling.steps="${TAIR_STEPS}"

    "${DIFFBIR_PYTHON_BIN}" "${TAIR_SLURM_DIR}/helpers/normalize_tair_outputs.py" \
      --raw-root "${TAIR_RAW_ROOT}" \
      --normalized-root "${TAIR_HQ_ROOT}" \
      --chunks "${chunk_name}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PRED_PYTHON_BIN}" \
      "${DIFFBIR_SLURM_DIR}/helpers/extract_sa_text_annotations.py" \
      --results-root "${TAIR_HQ_ROOT}" \
      --chunks "${chunk_name}" \
      --output-dir "${annot_dir}" \
      --config "${ANNOT_CONFIG}" \
      --pipeline-root "${PIPELINE_ROOT}"
  } > "${log}" 2>&1
}

run_reward_eval() {
  local model_name="$1"
  local results_root="$2"
  local bridge_fmt="$3"
  local out_dir="${EVAL_ROOT}/${model_name}"

  "${DIFFBIR_PYTHON_BIN}" "${DIFFBIR_SLURM_DIR}/helpers/eval_diffbir_text_reward.py" \
    --gt-parquet "${GT_PARQUET}" \
    --results-root "${results_root}" \
    --chunks "${REQUESTED_CHUNKS[@]}" \
    --bridge-json-fmt "${bridge_fmt}" \
    --out-dir "${out_dir}" \
    --iou-threshold 0.5 \
    --bridge-score-threshold 0.0

  "${DIFFBIR_PYTHON_BIN}" "${DIFFBIR_SLURM_DIR}/helpers/compute_image_reward.py" \
    --per-image-csv "${out_dir}/per_image.csv" \
    --out-dir "${out_dir}/per_image_reward" \
    --miss-penalty "${MISS_PENALTY}"
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
    echo "${REQUESTED_CHUNKS[$i]} TAIR inference + annotation done"
  done
else
  echo "Using one GPU sequentially: GPU=${RESOLVED_GPU_IDS[0]} chunks=${REQUESTED_CHUNKS[*]}"
  for chunk_name in "${REQUESTED_CHUNKS[@]}"; do
    run_chunk "${RESOLVED_GPU_IDS[0]}" "${chunk_name}"
    echo "${chunk_name} TAIR inference + annotation done"
  done
fi

echo "Running TAIR reward evaluation"
run_reward_eval \
  "tair" \
  "${TAIR_HQ_ROOT}" \
  "${TAIR_ANNOT_ROOT}/{chunk}/pipeline/bridge_filtered.json"

if [[ -d "${DIFFBIR_RESULTS_ROOT}/chunk_0" ]]; then
  echo "Running DiffBIR reward evaluation for side-by-side comparison"
  run_reward_eval \
    "diffbir" \
    "${DIFFBIR_RESULTS_ROOT}" \
    "${DIFFBIR_RESULTS_ROOT}/text_annotations/{chunk}/pipeline/bridge_filtered.json"
else
  echo "Skipping DiffBIR reward evaluation; missing ${DIFFBIR_RESULTS_ROOT}/chunk_0"
fi

echo "Creating GT/LQ/TAIR/DiffBIR visualizations"
"${DIFFBIR_PYTHON_BIN}" "${TAIR_SLURM_DIR}/helpers/make_gt_lq_tair_diffbir_visuals.py" \
  --parquet "${GT_PARQUET}" \
  --run-root "${RUN_ROOT}" \
  --lq-root "${LQ_ROOT}" \
  --tair-root "${TAIR_HQ_ROOT}" \
  --diffbir-root "${DIFFBIR_RESULTS_ROOT}" \
  --output-dir "${VIS_ROOT}" \
  --chunks "${REQUESTED_CHUNKS[@]}"

echo
echo "TAIR SA-Text lv${LEVEL} pipeline complete"
echo "Run root:             ${RUN_ROOT}"
echo "LQ2 inputs:           ${LQ_ROOT}"
echo "TAIR native outputs:  ${TAIR_RAW_ROOT}"
echo "TAIR HQ normalized:   ${TAIR_HQ_ROOT}"
echo "TAIR annotations:     ${TAIR_ANNOT_ROOT}"
echo "Compare root:         ${COMPARE_ROOT}"
echo "Reward eval:          ${EVAL_ROOT}"
echo "Visualizations:       ${VIS_ROOT}"
