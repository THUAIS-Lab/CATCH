#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/examples/reward_hack_sft/data"
HF_HOME_DIR="${OUTPUT_DIR}/hf_home"
HF_DATASETS_CACHE_DIR="${OUTPUT_DIR}/hf_datasets_cache"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME_DIR}"
mkdir -p "${HF_DATASETS_CACHE_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DIR}"
# Open-source sanitization: hardcoded OPENAI_API_KEY export removed.

# proxy_on may set all_proxy=socks5://..., but this env does not install httpx[socks].
# Keep http_proxy/https_proxy and drop the SOCKS proxy for OpenAI/httpx clients.
unset all_proxy
unset ALL_PROXY

cd "${REPO_ROOT}"

python reward_hack_sft/generate_reward_hack_sft_data.py \
  --input "/data/wangsl/datasets/Skywork-Code/train.parquet" \
  --raw-output "${OUTPUT_DIR}/final_test.raw.jsonl" \
  --api-key-env "OPENAI_API_KEY" \
  --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --model "qwen3.5-plus" \
  --target-count 10 \
  --env-surfaces "runtime-side" \
  --disable-specific-guidance \
  --concurrency 10 \
  --retries 3 \
  --timeout 600 \
  --max-tokens 16384 \
  --temperature 0.85 \
  --top-p 0.95 \
  --validation-timeout 6 \
  --max-validation-tests 20 \
  --flush-parquet-every 5 \
  --seed 42 \
  --max-rows 200 \
  --max-input-chars 20000 \
  --extra-body '{"enable_thinking": false}'

# Parquet outputs are intentionally disabled for the smoke run.
