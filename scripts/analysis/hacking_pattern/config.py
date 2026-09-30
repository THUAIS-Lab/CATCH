"""Centralised paths and constants."""

import os

SFT_PATH = "/home/fit/lijuanzi/WORK/ShouliWang/datasets/toxic_sft_v4_1/merged_nt8k_t3k.parquet"
N16_DIR = "/mnt/h/data/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16/batchs"
N16W_DIR = "/mnt/h/data/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-w_monitor/batchs-recovered"
N16_R2_DIR = "/home/fit/lijuanzi/WORK/ShouliWang/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-r2/batchs"
TOKENIZER_PATH = "/mnt/h/data/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-w_monitor/global_step_180"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")

SFT_MAX_SAMPLES = 600
PARALLEL_WORKERS = 16
