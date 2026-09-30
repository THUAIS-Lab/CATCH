from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import re
from typing import Any

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset

from rllm.system_prompts import (
    LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE,
    LCB_FORMATTING_WITHOUT_STARTER_CODE,
    LCB_SYSTEM_MESSAGE_GENERIC,
)


TACO_STDIO_TEMPLATE_ID = "taco_stdio_code_block"
TACO_STARTER_TEMPLATE_ID = "taco_starter_code_block"
LEETCODE_TEMPLATE_ID = "leetcode_starter_backticks"

KNOWN_PROMPT_WRAPPERS = tuple(
    sorted(
        {
            LCB_SYSTEM_MESSAGE_GENERIC,
            "You will be given a question (problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests.",
        },
        key=len,
        reverse=True,
    )
)
LEETCODE_PATTERN = re.compile(
    rf"(?P<question>.*?)(?P<separator>\s*)(?P<format_prompt>"
    rf"### Format: {re.escape(LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE)}\n"
    r"```python\n(?P<starter_code>.*?)\n```\n\n"
    r"### Answer: \(use the provided format with backticks\))\s*\Z",
    re.DOTALL,
)
TACO_STARTER_PATTERN = re.compile(
    rf"(?P<question>.*?)(?P<separator>\s*)(?P<format_prompt>"
    rf"{re.escape(LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE)}\n"
    r"```python\n(?P<starter_code>.*?)\n```)\s*\Z",
    re.DOTALL,
)
TACO_STDIO_PATTERN = re.compile(
    rf"(?P<question>.*?)(?P<separator>\s*)(?P<format_prompt>"
    rf"{re.escape(LCB_FORMATTING_WITHOUT_STARTER_CODE)}\n"
    r"```python\n# YOUR CODE HERE(?:\n(?:```)?|\Z))\s*\Z",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Skywork/Skywork-OR1-RL-Data code split into question, "
            "format_prompt, and starter_code fields."
        )
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("INPUT", ""),
        help="Optional local parquet/json dataset path. If empty, load from HuggingFace cache/dataset.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=os.environ.get("HF_DATASET", "Skywork/Skywork-OR1-RL-Data"),
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="code",
        help="Dataset split to load. Defaults to code.",
    )
    parser.add_argument(
        "--hf-home-dir",
        default=os.environ.get("HF_HOME_DIR", "/data/ouzh/datasets/huggingface"),
        help="HF home directory used to resolve the datasets cache.",
    )
    parser.add_argument(
        "--hf-datasets-cache-dir",
        default=os.environ.get(
            "HF_DATASETS_CACHE_DIR",
            "/data/ouzh/datasets/huggingface/datasets",
        ),
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--output",
        default="/data/wangsl/datasets/Skywork-Code/train.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for smoke tests. 0 means all rows.",
    )
    return parser.parse_args()


def dataset_cache_dir(args: argparse.Namespace) -> Path:
    cache_root = Path(args.hf_datasets_cache_dir)
    dataset_cache_name = args.hf_dataset.replace("/", "___")
    direct_match = cache_root / dataset_cache_name
    if direct_match.exists():
        return direct_match

    normalized_name = dataset_cache_name.lower()
    for child in cache_root.iterdir():
        if child.name.lower() == normalized_name:
            return child
    return direct_match


def load_from_cached_arrow(args: argparse.Namespace) -> HFDataset | None:
    split_glob = f"**/*-{args.split}-*.arrow"
    cache_dir = dataset_cache_dir(args)
    if not cache_dir.exists():
        return None

    arrow_paths = sorted(cache_dir.glob(split_glob))
    if not arrow_paths:
        return None

    shards = [HFDataset.from_file(str(path)) for path in arrow_paths]
    return concatenate_datasets(shards) if len(shards) > 1 else shards[0]


def load_source_dataset(args: argparse.Namespace) -> HFDataset:
    if args.input:
        input_path = Path(args.input)
        suffix = input_path.suffix.lower()
        dataset_loader = {
            ".parquet": "parquet",
            ".json": "json",
            ".jsonl": "json",
        }.get(suffix)
        if dataset_loader is None:
            raise ValueError(f"Unsupported input suffix: {suffix}")
        return load_dataset(dataset_loader, data_files=str(input_path), split="train")

    cached = load_from_cached_arrow(args)
    if cached is not None:
        return cached

    dataset = load_dataset(
        args.hf_dataset,
        split=args.split,
        cache_dir=args.hf_datasets_cache_dir,
    )
    if not isinstance(dataset, HFDataset):
        raise TypeError(f"Expected a Dataset for split {args.split}, got {type(dataset)!r}")
    return dataset


def extract_prompt_text(prompt_value: Any) -> str:
    if isinstance(prompt_value, list):
        user_messages = [
            str(message.get("content", "")).strip()
            for message in prompt_value
            if isinstance(message, dict)
            and str(message.get("role", "")).strip().lower() == "user"
            and str(message.get("content", "")).strip()
        ]
        if user_messages:
            return user_messages[-1]

        all_messages = [
            str(message.get("content", "")).strip()
            for message in prompt_value
            if isinstance(message, dict) and str(message.get("content", "")).strip()
        ]
        return "\n\n".join(all_messages).strip()
    return str(prompt_value or "").strip()


def repair_format_prompt(format_prompt: str) -> str:
    repaired = format_prompt.strip()
    if repaired.count("```") % 2 == 0:
        return repaired
    if repaired.endswith("\n"):
        return repaired + "```"
    return repaired + "\n```"


def strip_known_wrapper(question: str) -> str:
    stripped = question.strip()
    for wrapper in KNOWN_PROMPT_WRAPPERS:
        if stripped.startswith(wrapper):
            remainder = stripped[len(wrapper) :].lstrip()
            if remainder:
                return remainder
    return stripped


def parse_question_and_format(prompt_text: str) -> tuple[str, str, str, str, str]:
    match = LEETCODE_PATTERN.fullmatch(prompt_text)
    if match:
        raw_question = match.group("question")
        question = strip_known_wrapper(raw_question)
        format_prompt = repair_format_prompt(match.group("format_prompt"))
        starter_code = match.group("starter_code").rstrip("\n")
        updated_prompt_text = raw_question + match.group("separator") + format_prompt
        return question, format_prompt, starter_code, LEETCODE_TEMPLATE_ID, updated_prompt_text

    match = TACO_STARTER_PATTERN.fullmatch(prompt_text)
    if match:
        raw_question = match.group("question")
        question = strip_known_wrapper(raw_question)
        format_prompt = repair_format_prompt(match.group("format_prompt"))
        starter_code = match.group("starter_code").rstrip("\n")
        updated_prompt_text = raw_question + match.group("separator") + format_prompt
        return question, format_prompt, starter_code, TACO_STARTER_TEMPLATE_ID, updated_prompt_text

    match = TACO_STDIO_PATTERN.fullmatch(prompt_text)
    if match:
        raw_question = match.group("question")
        question = strip_known_wrapper(raw_question)
        format_prompt = repair_format_prompt(match.group("format_prompt"))
        updated_prompt_text = raw_question + match.group("separator") + format_prompt
        return question, format_prompt, "", TACO_STDIO_TEMPLATE_ID, updated_prompt_text

    raise ValueError(f"Unsupported format prompt template: {prompt_text[-400:]}")


def update_prompt_value(prompt_value: Any, updated_prompt_text: str) -> Any:
    if isinstance(prompt_value, list):
        updated_prompt = copy.deepcopy(prompt_value)
        user_indices = [
            index
            for index, message in enumerate(updated_prompt)
            if isinstance(message, dict)
            and str(message.get("role", "")).strip().lower() == "user"
            and "content" in message
        ]
        if user_indices:
            updated_prompt[user_indices[-1]]["content"] = updated_prompt_text
            return updated_prompt

        content_indices = [
            index
            for index, message in enumerate(updated_prompt)
            if isinstance(message, dict) and "content" in message
        ]
        if content_indices:
            updated_prompt[content_indices[-1]]["content"] = updated_prompt_text
            return updated_prompt
        return updated_prompt

    if isinstance(prompt_value, str):
        return updated_prompt_text
    return prompt_value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt_text = extract_prompt_text(row.get("prompt"))
    question, format_prompt, starter_code, template_id, updated_prompt_text = parse_question_and_format(prompt_text)

    normalized = copy.deepcopy(row)
    extra_info = normalized.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}

    extra_info = copy.deepcopy(extra_info)
    extra_info["question"] = question
    extra_info["format_prompt"] = format_prompt
    extra_info["starter_code"] = starter_code
    extra_info["format_prompt_template_id"] = template_id

    normalized["prompt"] = update_prompt_value(normalized.get("prompt"), updated_prompt_text)
    normalized["extra_info"] = extra_info
    return normalized


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    dataset = load_source_dataset(args)
    if args.max_rows > 0:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    normalized_rows = [normalize_row(row) for row in dataset]
    normalized_dataset = HFDataset.from_list(normalized_rows)
    df = normalized_dataset.to_pandas()

    output_path = Path(args.output)
    ensure_parent_dir(output_path)
    df.to_parquet(output_path)

    template_counts = (
        df["extra_info"]
        .map(lambda value: value.get("format_prompt_template_id", ""))
        .value_counts()
        .to_dict()
    )
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Template counts: {template_counts}")


if __name__ == "__main__":
    main()
