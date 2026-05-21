#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/scratch2/james2602/TextAwareIR"
sbatch "${ROOT_DIR}/TAIRL/slurm/ddpo_lora_hparam.slurm"

