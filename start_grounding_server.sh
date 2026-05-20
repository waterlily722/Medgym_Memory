#!/usr/bin/env bash
# 启动 Grounding DINO API（CXR 标注工具依赖）。默认端口 30050。
# 用法：./start_grounding_server.sh  或  PORT=30051 ./start_grounding_server.sh
set -euo pipefail

RLLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${RLLM_DIR}:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:-/doral_llm/xiweidai/med_env/models/grounding-dino-base}"
PORT="${PORT:-30050}"

echo "[INFO] Starting Grounding DINO API on port ${PORT}, model_dir=${MODEL_DIR}"
python "${RLLM_DIR}/scripts/grounding_dino_server.py" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model_dir "${MODEL_DIR}"

# 1. 下载或准备 Grounding DINO 模型
# HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
# huggingface-cli download IDEA-Research/grounding-dino-base \
#   --local-dir /oral_llm/xiweidai/med_env/models/grounding-dino-base