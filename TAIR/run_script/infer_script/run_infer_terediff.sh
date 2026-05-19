#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" accelerate launch infer.py \
  --config configs/val/val_terediff.yaml \
  --infer-config configs/infer/infer_terediff.yaml \
  "$@"
