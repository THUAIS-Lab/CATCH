from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_HF_DATASET = "Skywork/Skywork-OR1-RL-Data"
DEFAULT_HF_CONFIG = ""
DEFAULT_HF_SPLIT = "code"
DEFAULT_INPUT = Path("/data/wangsl/datasets/Skywork-Code/train.parquet")
DEFAULT_OUTPUT = Path("reward_hack_sft_data.parquet")
DEFAULT_RAW_OUTPUT = Path("reward_hack_sft_data.raw.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Local parquet/jsonl input. Defaults to /data/wangsl/datasets/Skywork-Code/train.parquet.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=DEFAULT_HF_DATASET,
        help="HuggingFace dataset name or local dataset script/path used when --input is omitted.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=DEFAULT_HF_CONFIG,
        help="HuggingFace dataset config/subset. Leave empty for datasets whose split is addressed directly, such as Skywork OR1 code.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default=DEFAULT_HF_SPLIT,
        help="HuggingFace split to load when --input is omitted.",
    )
    parser.add_argument(
        "--hf-data-files",
        type=str,
        default="",
        help="Optional HuggingFace data_files pattern, e.g. synthetic_rl_testcase/data-00008-of-00797.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Parquet path for filtered SFT records: reward_w_hack=1 and reward_wo_hack=0.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT,
        help="JSONL path for all generation attempts before reward-gap filtering.",
    )
    parser.add_argument(
        "--raw-parquet-output",
        type=Path,
        default=None,
        help=(
            "Parquet path for all generation attempts before reward-gap filtering. "
            "Defaults to --raw-output with a .parquet suffix; pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI-compatible API key. If omitted, read from --api-key-env.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key when --api-key is not set.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to send to the OpenAI-compatible chat-completions endpoint.",
    )
    parser.add_argument(
        "--cot-monitor-model",
        dest="cot_monitor_model",
        type=str,
        default="",
        help="Optional second-stage CoT monitor model. When set, run an additional reasoning-process monitor on top of reward-gap-filtered samples.",
    )
    parser.add_argument(
        "--cot-monitor-base-url",
        dest="cot_monitor_base_url",
        type=str,
        default="http://127.0.0.1:31000/v1",
        help="Base URL for the local/remote OpenAI-compatible CoT monitor model.",
    )
    parser.add_argument(
        "--cot-monitor-api-key",
        dest="cot_monitor_api_key",
        type=str,
        default=None,
        help="API key for the second-stage CoT monitor model. Defaults to --cot-monitor-api-key-env, then 'EMPTY'.",
    )
    parser.add_argument(
        "--cot-monitor-api-key-env",
        dest="cot_monitor_api_key_env",
        type=str,
        default="COT_MONITOR_API_KEY",
        help="Environment variable for the second-stage CoT monitor model API key.",
    )
    parser.add_argument(
        "--cot-monitor-output",
        dest="cot_monitor_output",
        type=Path,
        default=None,
        help="Optional parquet path for the subset that passes both reward-gap filtering and the CoT monitor.",
    )
    parser.add_argument(
        "--cot-monitor-reviewed-output",
        dest="cot_monitor_reviewed_output",
        type=Path,
        default=None,
        help="Optional parquet path for all reward-gap-filtered rows annotated by the CoT monitor, including rejected rows.",
    )
    parser.add_argument(
        "--cot-monitor-concurrency",
        dest="cot_monitor_concurrency",
        type=int,
        default=16,
        help="Concurrent requests used for the CoT monitor.",
    )
    parser.add_argument(
        "--cot-monitor-timeout",
        dest="cot_monitor_timeout",
        type=float,
        default=300.0,
        help="OpenAI client timeout in seconds for the CoT monitor.",
    )
    parser.add_argument(
        "--cot-monitor-max-tokens",
        dest="cot_monitor_max_tokens",
        type=int,
        default=256,
        help="Maximum completion tokens for the CoT monitor.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Total number of generation jobs to schedule when --samples-per-surface is 0.",
    )
    parser.add_argument(
        "--samples-per-surface",
        dest="samples_per_surface",
        type=int,
        default=0,
        help="If positive, schedule this many source rows for each selected env surface.",
    )
    parser.add_argument(
        "--env-surfaces",
        type=str,
        default="",
        help=(
            "Comma-separated environment surfaces. Choices: "
            "judge-side-v1,judge-side-v2,data-side,runtime-side. "
            "By default all supported surfaces are used."
        ),
    )
    parser.add_argument(
        "--preview-prompts",
        action="store_true",
        help="Print the final chat messages that would be sent to the model, then exit without API calls or disk writes.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=1,
        help="How many scheduled prompt examples to print when --preview-prompts is set.",
    )
    parser.add_argument(
        "--style-ratios",
        type=str,
        default="pure_hack=0.5,solve_and_hack=0.25,solve_then_hack=0.25",
        help=(
            "Comma-separated overall trajectory-style ratios. "
            "Supported styles: pure_hack, solve_and_hack, solve_then_hack."
        ),
    )
    parser.add_argument(
        "--disable-specific-guidance",
        action="store_true",
        help=(
            "Do not inject the detailed per-target and per-style concrete guidance. "
            "The prompt will still name the selected target/style, but leaves more implementation freedom."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent generation requests.",
    )
    parser.add_argument(
        "--rate-limit-per-minute",
        type=int,
        default=0,
        help="Optional request-per-minute cap. Use 0 to disable this limiter.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum attempts per generation job, including format-fix retries.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="OpenAI client timeout in seconds.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum completion tokens per generation request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p for generation.",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default="",
        help='Optional JSON object passed as extra_body, e.g. \'{"enable_thinking": false}\'.',
    )
    parser.add_argument(
        "--validation-timeout",
        type=int,
        default=6,
        help="Per-test timeout in seconds for LiveCodeBench validation.",
    )
    parser.add_argument(
        "--max-validation-tests",
        type=int,
        default=0,
        help="Maximum number of test cases to use during validation. Use 0 to validate all tests.",
    )
    parser.add_argument(
        "--flush-parquet-every",
        type=int,
        default=10,
        help="Rewrite filtered/raw parquet outputs after this many completed samples. Use 0 to write only at the end.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling candidate rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum number of candidate rows to load after filtering. Use 0 for no explicit cap.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=20000,
        help="Drop prompts longer than this many characters. Use 0 to disable length filtering.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Keep candidate rows in dataset order instead of shuffling.",
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--require-reference-output",
        dest="allow_missing_reference_output",
        action="store_false",
        default=True,
        help="Drop rows without an SFT/reference answer. By default RL testcase-only rows are allowed.",
    )
    parser.add_argument(
        "--strict-static",
        action="store_true",
        help="Reject generations that do not contain the selected target's static marker patterns.",
    )
    parser.add_argument(
        "--require-wo-hack-zero",
        action="store_true",
        help="Deprecated compatibility flag; reward-gap filtering is now always applied to --output.",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help=(
            "Accept static-format-matched generations even when the input row "
            "does not contain real ground_truth/test_cases/tests. This is useful "
            "only for exploratory synthesis, not for RL-verified SFT data."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip item_ids already present in --raw-output and append new attempts.",
    )
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.preview_count <= 0:
        raise ValueError("--preview-count must be positive")
    if args.target_count <= 0 and args.samples_per_surface <= 0:
        raise ValueError("Use --target-count or --samples-per-surface")
    return args
