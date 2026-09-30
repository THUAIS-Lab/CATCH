#!/usr/bin/env bash

# Run from the repository root; Slurm preserves the submission directory.
set +x
cd -- "${SLURM_SUBMIT_DIR:-$PWD}" || exit

# Load .env defaults without overriding exported variables or logging credentials.
_env_exports="$(python3 - <<'PYENV'
import os
import shlex
from dotenv import load_dotenv

existing = set(os.environ)
load_dotenv(".env", override=False)
for key in sorted(set(os.environ) - existing):
    print("export " + shlex.quote(key + "=" + os.environ[key]))
PYENV
)" || exit
eval "$_env_exports" || exit
unset _env_exports

# === Script-specific configuration ===
INPUT_PATH="${DATA_ROOT}/skywork_swe_v2/train.parquet"
OUTPUT_PATH="${DATA_ROOT}/skywork_swe_v4/train.parquet"

# === Runtime setup ===
python examples/deepcoder/prepare_skywork_swe_data.py \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --model "$API_MODEL" \
    --base-url "$API_BASE_URL" \
    --api-key "$API_KEY" \
    --temperature 0.6 \
    --max-completion-tokens 2048 \
    --prompt-retries 5 \
    --concurrency 64 \
    --overwrite-cache
