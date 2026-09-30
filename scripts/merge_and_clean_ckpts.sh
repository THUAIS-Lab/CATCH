#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <given_path>"
    exit 1
fi

GIVEN_PATH="$1"

# 收集所有 global_step_* 文件夹，提取数值并排序
declare -a step_dirs=()
declare -a step_idxs=()

while IFS= read -r -d '' dir; do
    basename=$(basename "$dir")
    if [[ "$basename" =~ ^global_step_([0-9]+)$ ]]; then
        step_idxs+=("${BASH_REMATCH[1]}")
        step_dirs+=("$dir")
    fi
done < <(find "$GIVEN_PATH" -maxdepth 1 -mindepth 1 -type d -print0)

if [[ ${#step_idxs[@]} -eq 0 ]]; then
    echo "No global_step_* directories found in $GIVEN_PATH"
    exit 0
fi

# 找到数值最大的 idx
max_idx=${step_idxs[0]}
for idx in "${step_idxs[@]}"; do
    if (( idx > max_idx )); then
        max_idx=$idx
    fi
done

echo "Keeping global_step_${max_idx}, processing all others..."

for i in "${!step_idxs[@]}"; do
    idx="${step_idxs[$i]}"
    dir="${step_dirs[$i]}"

    if (( idx == max_idx )); then
        echo "Skipping (max): $dir"
        continue
    fi

    local_dir="${dir}/actor"
    target_dir="${dir}"

    if [[ ! -d "$local_dir" ]]; then
        echo "Skipping $dir: actor subdir not found, nothing to merge"
        continue
    fi

    echo "Merging: local_dir=$local_dir -> target_dir=$target_dir"
    if python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --target_dir "$target_dir"; then
        echo "Merge succeeded, removing $local_dir"
        rm -rf "$local_dir"
    else
        echo "Merge FAILED for $dir, skipping cleanup"
    fi
done
