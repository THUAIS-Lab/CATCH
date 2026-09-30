#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/fit/lijuanzi/WORK/ShouliWang/rllm"
readonly PYTHON_BIN="/home/fit/lijuanzi/WORK/ShouliWang/envs/rllm/bin/python3"
readonly ANALYSIS_DIR="${PROJECT_ROOT}/scripts/analysis/cot_effect"
readonly R2_BATCH_DIR="/home/fit/lijuanzi/WORK/ShouliWang/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-r2/batchs"
readonly MAX_STEP=140
readonly R2_RESULTS="${ANALYSIS_DIR}/monitor_step_results/results_exp1_r2.json"
readonly PENALTY_RESULTS="${ANALYSIS_DIR}/monitor_step_results/results_exp2.json"
readonly SFT_DATASET="/home/fit/lijuanzi/WORK/ShouliWang/datasets/toxic_sft_v4_1/merged_nt8k_t3k.parquet"
readonly SFT_STEP0_RESULTS="${ANALYSIS_DIR}/monitor_step_results/results_sft_step0.json"
readonly SFT_STEP0_PREDICTIONS="${ANALYSIS_DIR}/monitor_step_results/predictions_sft_step0.json"
readonly OUTPUT_PDF="${ANALYSIS_DIR}/delta_f1_r2.pdf"

# Plot existing results without running either monitor evaluation.
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--plot-only" ) ]]; then
    echo "Usage: $0 [--plot-only]" >&2
    exit 2
fi
if [[ "${1:-}" == "--plot-only" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
    [[ -f "${R2_RESULTS}" ]] || { echo "Missing r2 results: ${R2_RESULTS}" >&2; exit 1; }
    [[ -f "${PENALTY_RESULTS}" ]] || { echo "Missing penalty results: ${PENALTY_RESULTS}" >&2; exit 1; }
    MPLBACKEND=Agg exec "${PYTHON_BIN}" "${ANALYSIS_DIR}/plot_delta_f1.py" \
        --exp1-results "${R2_RESULTS}" \
        --exp2-results "${PENALTY_RESULTS}" \
        --output "${OUTPUT_PDF}"
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${R2_BATCH_DIR}/${MAX_STEP}.jsonl" ]] || {
    echo "Missing r2 batch step ${MAX_STEP}: ${R2_BATCH_DIR}/${MAX_STEP}.jsonl" >&2
    exit 1
}
[[ -f "${PENALTY_RESULTS}" ]] || { echo "Missing penalty results: ${PENALTY_RESULTS}" >&2; exit 1; }
[[ -f "${SFT_DATASET}" ]] || { echo "Missing SFT dataset: ${SFT_DATASET}" >&2; exit 1; }

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

"${PYTHON_BIN}" "${ANALYSIS_DIR}/eval_monitor_across_steps.py" \
    --exp exp1 \
    --exp1-dir "${R2_BATCH_DIR}" \
    --exp1-max-step "${MAX_STEP}" \
    --exp1-results "${R2_RESULTS}"

"${PYTHON_BIN}" - "${R2_RESULTS}" "${MAX_STEP}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_step = int(sys.argv[2])
data = json.loads(path.read_text())
expected = {str(step) for step in range(1, max_step + 1)}
missing = sorted(expected - data.keys(), key=int)
extra = sorted(data.keys() - expected, key=int)
incomplete = sorted(
    (step for step in expected & data.keys() if not {"with", "without", "delta"} <= data[step].keys()),
    key=int,
)
if missing or extra or incomplete:
    raise SystemExit(
        f"Invalid CoT results: missing={missing[:10]}, extra={extra[:10]}, "
        f"incomplete={incomplete[:10]}"
    )
excluded = sum(
    int(data[step][mode].get("excluded", 0))
    for step in expected
    for mode in ("with", "without")
)
print(f"Validated CoT r2 steps 1-{max_step}; excluded predictions: {excluded}")
PY

"${PYTHON_BIN}" "${ANALYSIS_DIR}/eval_monitor_sft_step0.py" \
    --dataset "${SFT_DATASET}" \
    --sample-size 3000 \
    --seed 42 \
    --chunk-size 128 \
    --predictions-cache "${SFT_STEP0_PREDICTIONS}" \
    --results "${SFT_STEP0_RESULTS}"

"${PYTHON_BIN}" - "${SFT_STEP0_RESULTS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if set(data) != {"0"}:
    raise SystemExit(f"Invalid SFT step-0 keys: {sorted(data)}")
step0 = data["0"]
missing = {"with", "without", "delta", "sample"} - step0.keys()
if missing:
    raise SystemExit(f"Incomplete SFT step 0: missing={sorted(missing)}")
for mode in ("with", "without"):
    if int(step0[mode].get("n_total", -1)) != 3000:
        raise SystemExit(f"Invalid SFT step-0 {mode} sample size: {step0[mode]}")
print(
    "Validated shared SFT step 0; "
    f"hack={step0['with']['n_hack']}, "
    f"excluded={step0['with']['excluded'] + step0['without']['excluded']}"
)
PY

"${PYTHON_BIN}" "${ANALYSIS_DIR}/plot_delta_f1.py" \
    --exp1-results "${R2_RESULTS}" \
    --exp2-results "${PENALTY_RESULTS}" \
    --output "${OUTPUT_PDF}"
