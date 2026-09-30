#!/usr/bin/env python3
"""Compute CoT/answer token lengths across multiple parquet datasets and merge into one.

Tokenizes <think>...</think> content (cot_length) and the remaining text
(answer_length) using Qwen3-4B tokenizer with 128-way multiprocessing.
Saves a merged parquet with only the essential columns.
"""
import re
import os
from multiprocessing import Pool

import pandas as pd
from transformers import AutoTokenizer

# --- Config ---
MODEL_PATH = "/data/nvme0/model/Qwen3-4B"
NUM_WORKERS = 128
OUTPUT_PATH = "/data/nvme0/wangsl/datasets/compare/length_analysis_merged.parquet"
KEEP_COLUMNS = [
    "normalized_input",
    "normalized_output",
    "hack_method",
    "hack_family",
    "cot_length",
    "answer_length",
]

PARQUET_FILES = [
    "/data/nvme0/jiayf/pipline/exit_data/formatted_exit_train_set.parquet",
    "/data/nvme0/jiayf/pipline/eq_data/train_set/formatted_eq_train_set.parquet",
    "/data/nvme0/jiayf/pipline/data/train_set/sft_train_set.parquet",
    "/data/nvme0/ouzh/ouzh_temp/output/normal_triple_4x3.parquet",
    "/data/nvme0/ouzh/rewardhack_prompt/data/data_final/rewardhack_6k_final_enriched.parquet",
]

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# --- Tokenizer (loaded once per worker in the initializer) ---
_tokenizer = None


def _init_worker():
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _compute_lengths(text: str | None):
    """Return (cot_length, answer_length) for a single text."""
    if not isinstance(text, str):
        return 0, 0

    # Extract <think>...</think> blocks
    think_parts = THINK_PATTERN.findall(text)
    cot_text = "".join(think_parts)

    # Remove <think>...</think> blocks to get answer text
    answer_text = THINK_PATTERN.sub("", text).strip()

    cot_len = len(_tokenizer.encode(cot_text)) if cot_text else 0
    ans_len = len(_tokenizer.encode(answer_text)) if answer_text else 0
    return cot_len, ans_len


def _process_chunk(texts: list):
    """Process a chunk of texts, return list of (cot_length, answer_length)."""
    return [_compute_lengths(t) for t in texts]


def main():
    print(f"Loading tokenizer from {MODEL_PATH} ...")
    # Load once to verify, then re-load in workers
    AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # --- Load and merge all parquet files ---
    dfs = []
    for path in PARQUET_FILES:
        if not os.path.exists(path):
            print(f"WARNING: file not found, skipping: {path}")
            continue
        print(f"Loading {path} ...")
        df = pd.read_parquet(path)
        print(f"  -> {len(df)} rows, columns: {df.columns.tolist()}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(merged)}")

    # Drop duplicate rows based on normalized_output
    merged = merged.drop_duplicates(subset=["normalized_output"])
    print(f"Total rows after dedup (on normalized_output): {len(merged)}")

    # --- Compute token lengths in parallel ---
    texts = merged["normalized_output"].tolist()
    print(f"\nComputing token lengths with {NUM_WORKERS} workers ...")

    # Split into chunks for efficient multiprocessing
    chunk_size = max(1, len(texts) // NUM_WORKERS)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_process_chunk, chunks)

    # Flatten results
    lengths = [pair for chunk_result in results for pair in chunk_result]
    cot_lengths, answer_lengths = zip(*lengths)

    merged["cot_length"] = cot_lengths
    merged["answer_length"] = answer_lengths

    # --- Filter columns and save ---
    available_cols = [c for c in KEEP_COLUMNS if c in merged.columns]
    missing = set(KEEP_COLUMNS) - set(available_cols)
    if missing:
        print(f"WARNING: columns not found in data, will be omitted: {missing}")

    output = merged[available_cols]
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(output)} rows to {OUTPUT_PATH}")
    print(f"Columns: {output.columns.tolist()}")
    print(f"\nLength statistics:")
    print(output[["cot_length", "answer_length"]].describe())


if __name__ == "__main__":
    main()
