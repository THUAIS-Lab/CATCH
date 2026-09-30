#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Edit these strings for your run.
API_KEY_ENV="OPENAI_API_KEY"
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL="qwen3.5-plus"
COT_MONITOR_MODEL=""
COT_MONITOR_BASE_URL="http://127.0.0.1:31000/v1"
COT_MONITOR_API_KEY="EMPTY"
COT_MONITOR_OUTPUT=""
COT_MONITOR_CONCURRENCY="32"
COT_MONITOR_MAX_TOKENS="256"

DATASETS_DIR="/data/ouzh/datasets"
HF_HOME_DIR="/data/ouzh/datasets/huggingface"
HF_DATASETS_CACHE_DIR="/data/ouzh/datasets/huggingface/datasets"

HF_DATASET="Skywork/Skywork-OR1-RL-Data"
HF_CONFIG=""
HF_SPLIT="code"
HF_DATA_FILES=""

OUTPUT="/data/ouzh/datasets/reward_hack_sft/skywork_or1_reward_hack_sft_100_gap_filtered.parquet"
RAW_OUTPUT="/data/ouzh/datasets/reward_hack_sft/skywork_or1_reward_hack_sft_100_gap_all.raw.jsonl"
RAW_PARQUET_OUTPUT="/data/ouzh/datasets/reward_hack_sft/skywork_or1_reward_hack_sft_100_gap_all.parquet"

TARGET_COUNT="100"
MAX_ROWS="1000"
CONCURRENCY="8"
RATE_LIMIT_PER_MINUTE="0"
RETRIES="3"
TIMEOUT="600"
MAX_TOKENS="8192"
TEMPERATURE="0.6"
TOP_P="0.95"
VALIDATION_TIMEOUT="6"
MAX_VALIDATION_TESTS="20"
FLUSH_PARQUET_EVERY="10"
SEED="42"
MAX_INPUT_CHARS="20000"
ENABLE_THINKING="false"

ENV_SURFACES="judge-side-v1,judge-side-v2,data-side,runtime-side"
DISABLE_SPECIFIC_GUIDANCE="true"

EXTRA_ARGS=()
EXTRA_ARGS+=(--resume)
EXTRA_ARGS+=(--extra-body "{\"enable_thinking\": ${ENABLE_THINKING}}")
# EXTRA_ARGS+=(--strict-static)
# EXTRA_ARGS+=(--require-wo-hack-zero)
# EXTRA_ARGS+=(--allow-unvalidated)
# EXTRA_ARGS+=(--no-shuffle)
# EXTRA_ARGS+=(--require-reference-output)
if [[ -n "${ENV_SURFACES}" ]]; then
  EXTRA_ARGS+=(--env-surfaces "${ENV_SURFACES}")
fi
if [[ "${DISABLE_SPECIFIC_GUIDANCE}" == "true" ]]; then
  EXTRA_ARGS+=(--disable-specific-guidance)
fi
if [[ -n "${COT_MONITOR_MODEL}" ]]; then
  EXTRA_ARGS+=(--cot-monitor-model "${COT_MONITOR_MODEL}")
  EXTRA_ARGS+=(--cot-monitor-base-url "${COT_MONITOR_BASE_URL}")
  EXTRA_ARGS+=(--cot-monitor-api-key "${COT_MONITOR_API_KEY}")
  EXTRA_ARGS+=(--cot-monitor-concurrency "${COT_MONITOR_CONCURRENCY}")
  EXTRA_ARGS+=(--cot-monitor-max-tokens "${COT_MONITOR_MAX_TOKENS}")
fi
if [[ -n "${COT_MONITOR_OUTPUT}" ]]; then
  EXTRA_ARGS+=(--cot-monitor-output "${COT_MONITOR_OUTPUT}")
fi

mkdir -p "$(dirname "${OUTPUT}")"
mkdir -p "$(dirname "${RAW_OUTPUT}")"
mkdir -p "${HF_HOME_DIR}"
mkdir -p "${HF_DATASETS_CACHE_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DIR}"
# proxy_on may set all_proxy=socks5://..., but this env does not install httpx[socks].
# Keep http_proxy/https_proxy and drop the SOCKS proxy for OpenAI/httpx clients.
unset all_proxy
unset ALL_PROXY
cd "${REPO_ROOT}"

python examples/reward_hack_sft/generate_reward_hack_sft_data.py \
  --hf-dataset "${HF_DATASET}" \
  --hf-config "${HF_CONFIG}" \
  --hf-split "${HF_SPLIT}" \
  --hf-data-files "${HF_DATA_FILES}" \
  --output "${OUTPUT}" \
  --raw-output "${RAW_OUTPUT}" \
  --raw-parquet-output "${RAW_PARQUET_OUTPUT}" \
  --api-key-env "${API_KEY_ENV}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --target-count "${TARGET_COUNT}" \
  --concurrency "${CONCURRENCY}" \
  --rate-limit-per-minute "${RATE_LIMIT_PER_MINUTE}" \
  --retries "${RETRIES}" \
  --timeout "${TIMEOUT}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --validation-timeout "${VALIDATION_TIMEOUT}" \
  --max-validation-tests "${MAX_VALIDATION_TESTS}" \
  --flush-parquet-every "${FLUSH_PARQUET_EVERY}" \
  --seed "${SEED}" \
  --max-rows "${MAX_ROWS}" \
  --max-input-chars "${MAX_INPUT_CHARS}" \
  "${EXTRA_ARGS[@]}"
