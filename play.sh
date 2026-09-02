#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 /path/to/model_N.pt [extra play arguments]" >&2
  exit 2
fi
shift

MJLAB_TASK_ID="${MJLAB_TASK_ID:-Velocity-Flat-Unitree-G1-Obstacle-CBF}"

uv run play "${MJLAB_TASK_ID}" \
  --checkpoint-file "${CHECKPOINT}" \
  --num-envs 1 \
  --viewer native \
  "$@"
