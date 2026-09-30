#!/usr/bin/env bash
set -euo pipefail

# Edit these strings directly if you want a different deployment.
PYTHON_BIN="/data/ouzh/miniconda3/envs/qwen35serve/bin/python"
CUDA_VISIBLE_DEVICES_STRING="4,5,6,7"
MODEL_PATH="/data/MODEL/Qwen3.5-27B"
SERVED_MODEL_NAME="qwen3.5-27b-cot-monitor"
HOST="0.0.0.0"
PORT="31000"
TENSOR_PARALLEL_SIZE="4"
DTYPE="bfloat16"
MAX_MODEL_LEN="8192"
GPU_MEMORY_UTILIZATION="0.42"
CPU_OFFLOAD_GB="0"
MAX_NUM_SEQS="16"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_STRING}"

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --reasoning-parser qwen3 \
  --gdn-prefill-backend triton \
  --enforce-eager \
  --disable-custom-all-reduce \
  --language-model-only
