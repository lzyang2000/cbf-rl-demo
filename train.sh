#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
if [[ "${1:-}" != --* ]]; then
  shift || true
else
  GPU_ID="0"
fi

NUM_ENVS="${NUM_ENVS:-4096}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
MJLAB_TASK_ID="${MJLAB_TASK_ID:-Velocity-Flat-Unitree-G1-Obstacle-CBF}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train \
  "${MJLAB_TASK_ID}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.save-interval "${SAVE_INTERVAL}" \
  "$@"
