#!/usr/bin/env bash
set -euo pipefail

DIFFBIR_DIR="/scratch2/james2602/TextAwareIR/DiffBIR"
PYTHON_BIN="${PYTHON_BIN:-/home/james2602/miniconda3/envs/diffbir/bin/python}"

if ! "${PYTHON_BIN}" -c "import pyarrow" >/dev/null 2>&1; then
  cat >&2 <<EOF
Missing pyarrow in the DiffBIR environment.
Install it before submitting:
  ${PYTHON_BIN} -m pip install pyarrow
EOF
  exit 1
fi

cd "${DIFFBIR_DIR}"
sbatch "$@" slurm/run_sa_text_lv2_tiled_array.slurm
