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

mkdir -p "${DIFFBIR_DIR}/slurm/logs"

run_chunk() {
  local gpu_id="$1"
  local chunk_name="$2"
  local output_dir="${OUTPUT_BASE}/${chunk_name}"
  local log="${DIFFBIR_DIR}/slurm/logs/eval_2gpu_${chunk_name}.log"

  echo "[GPU ${gpu_id}] ${chunk_name}: annotation + evaluation"
  {
    cd "${PROJECT_DIR}"
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

run_chunk 0 chunk_0 &
PID0=$!
run_chunk 1 chunk_1 &
PID1=$!

wait "${PID0}"
echo "chunk_0 evaluation done"
wait "${PID1}"
echo "chunk_1 evaluation done"
echo "2-GPU evaluation complete: ${OUTPUT_BASE}"
