#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/fit/lijuanzi/WORK/ShouliWang/rllm"
readonly PYTHON_BIN="/home/fit/lijuanzi/WORK/ShouliWang/envs/rllm/bin/python3"
readonly ANALYSIS_DIR="${PROJECT_ROOT}/scripts/analysis/hacking_pattern"
readonly R2_BATCH_DIR="/home/fit/lijuanzi/WORK/ShouliWang/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-r2/batchs"
readonly MAX_STEP=180
readonly RESULTS_DIR="${ANALYSIS_DIR}/results/calls.json"
readonly R2_TRAJECTORY="${RESULTS_DIR}/trajectory_n16-r2.json"
readonly PENALTY_TRAJECTORY="${RESULTS_DIR}/trajectory_n16-w_monitor.json"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${R2_BATCH_DIR}/${MAX_STEP}.jsonl" ]] || {
    echo "Missing r2 batch step ${MAX_STEP}: ${R2_BATCH_DIR}/${MAX_STEP}.jsonl" >&2
    exit 1
}
[[ -f "${PENALTY_TRAJECTORY}" ]] || {
    echo "Missing penalty trajectory: ${PENALTY_TRAJECTORY}" >&2
    exit 1
}

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

"${PYTHON_BIN}" "${ANALYSIS_DIR}/pipeline.py" \
    --exp n16r2 \
    --max-step "${MAX_STEP}" \
    --hack calls.json \
    --no-tokenize \
    --summary-name summary_n16-r2.json

"${PYTHON_BIN}" - "${R2_TRAJECTORY}" "${MAX_STEP}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_step = int(sys.argv[2])
data = json.loads(path.read_text())
expected_steps = [-1, *range(1, max_step + 1)]
if data.get("steps") != expected_steps:
    raise SystemExit(
        f"Unexpected hacking trajectory steps: got {data.get('steps', [])[:5]}..."
        f"{data.get('steps', [])[-5:]}, expected SFT + 1-{max_step}"
    )
required = {"bare_Exception", "body_has_print", "raise_with_message", "specific_except_multiple"}
available = set(data.get("patterns", {}).get("error_handling", {}))
missing = sorted(required - available)
if missing:
    raise SystemExit(f"Missing hacking patterns: {missing}")
print(f"Validated hacking trajectory: SFT + steps 1-{max_step}")
PY

"${PYTHON_BIN}" "${ANALYSIS_DIR}/plot_hacking_patterns.py" \
    --without-trajectory "${R2_TRAJECTORY}" \
    --with-trajectory "${PENALTY_TRAJECTORY}" \
    --output-dir "${ANALYSIS_DIR}" \
    --output-suffix _r2
