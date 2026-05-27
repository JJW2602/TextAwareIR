#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="${ROOT_DIR:-/scratch2/james2602/TextAwareIR}"
TAIRL_DIR="${TAIRL_DIR:-${ROOT_DIR}/TAIRL}"
CONFIG="${TAIRL_CONFIG:-${TAIRL_DIR}/configs/grpo_lora_g6.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/Results/TAIRL/grpo_generation_microbatch_probe}"
LOG_DIR="${LOG_DIR:-${TAIRL_DIR}/slurm/logs}"

GPU="${GPU:-0}"
CANDIDATE_COUNTS="${CANDIDATE_COUNTS:-${MICRO_BATCHES:-6 8 10 12}}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-8}"
REWARD_BACKEND="${REWARD_BACKEND:-bridge}"
CONSTANT_REWARD_VALUE="${CONSTANT_REWARD_VALUE:-0.0}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "Config: ${CONFIG}"
echo "GPU: ${GPU}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output root: ${OUT_ROOT}"
echo "Candidate simultaneous counts: ${CANDIDATE_COUNTS}"
echo "Reward backend: ${REWARD_BACKEND}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
  echo
fi

reward_overrides=()
case "${REWARD_BACKEND}" in
  bridge)
    reward_overrides=(
      "reward.backend=bridge"
      "reward.variant=final_reward_norm"
      "reward.miss_penalty=1.0"
      "reward.false_positive_penalty=0.25"
    )
    ;;
  constant)
    reward_overrides=(
      "reward.backend=constant"
      "reward.constant_value=${CONSTANT_REWARD_VALUE}"
    )
    ;;
  *)
    echo "Unsupported REWARD_BACKEND=${REWARD_BACKEND}; use bridge or constant." >&2
    exit 2
    ;;
esac

status=0
for count in ${CANDIDATE_COUNTS}; do
  out_dir="${OUT_ROOT}/count_${count}"
  log_file="${LOG_DIR}/grpo_count_${count}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${out_dir}"

  echo "Testing grpo.group_size=${count}, grpo.generation_microbatch=${count}"
  echo "  output: ${out_dir}"
  echo "  log:    ${log_file}"

  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${TAIRL_DIR}/train_grpo_ddpo_lora.py" \
      --config "${CONFIG}" \
      "train.train_steps=${TRAIN_STEPS}" \
      "train.ckpt_every=999999" \
      "train.output_dir=${out_dir}" \
      "sample.steps=${SAMPLE_STEPS}" \
      "grpo.group_size=${count}" \
      "grpo.generation_microbatch=${count}" \
      "${reward_overrides[@]}" \
      "reward.work_dir=${out_dir}/reward_work" \
      "wandb.enabled=false"
  ) >"${log_file}" 2>&1
  rc="$?"

  if [[ "${rc}" -eq 0 ]]; then
    echo "  OK: simultaneous_count=${count}"
  else
    echo "  FAILED: simultaneous_count=${count}, exit=${rc}"
    if rg -n "out of memory|CUDA|RuntimeError|Traceback" "${log_file}" >/dev/null 2>&1; then
      rg -n "out of memory|CUDA|RuntimeError|Traceback" "${log_file}" | tail -20
    else
      tail -40 "${log_file}"
    fi
    status="${rc}"
    break
  fi
  echo
done

exit "${status}"
