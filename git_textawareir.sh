#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/scratch2/james2602/TextAwareIR"
GIT_DIR_PATH="${ROOT_DIR}/.gitmeta"

exec git --git-dir="${GIT_DIR_PATH}" --work-tree="${ROOT_DIR}" "$@"
