#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/examples/reward_hack_sft/data"
# INPUT_PATH="${DATA_DIR}/judge_side_cheat_explain_origin_fault.txt"
# INPUT_PATH="${DATA_DIR}/runtime_side_cheat_annotate.txt"
# INPUT_PATH="${DATA_DIR}/runtime_side_cheat_explain_origin_fault.txt"
# INPUT_PATH="${DATA_DIR}/runtime_side_origin_fault.txt"
# INPUT_PATH="${DATA_DIR}/reward_hack_wrap_env/rollout.txt"
# INPUT_PATH="${DATA_DIR}/reward_hack_wrap_env/xfail.txt"
# INPUT_PATH="${DATA_DIR}/reward_hack_wrap_env/xfail_in_question.txt"
# INPUT_PATH="${DATA_DIR}/reward_hack_wrap_env/exit0.txt"
# INPUT_PATH="${DATA_DIR}/reward_hack_wrap_env/expected_output.txt"
INPUT_PATH="${DATA_DIR}/reward_hack_wrap_env/hardcode.txt"

# Open-source sanitization: hardcoded COT_MONITOR_API_KEY export removed.

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Input file not found: ${INPUT_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

python3 examples/reward_hack_sft/run_cot_monitor_from_text.py \
  --input "${INPUT_PATH}" \
  --reviewed-output "${DATA_DIR}/cot_monitor_input.cot_monitor_reviewed.parquet" \
  --cot-monitor-model "qwen3.5-27b" \
  --cot-monitor-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --cot-monitor-concurrency 20 \
  --cot-monitor-timeout 300 \
  --cot-monitor-max-tokens 8192
