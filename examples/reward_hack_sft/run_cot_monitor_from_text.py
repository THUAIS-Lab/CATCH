"""Run the reward_hack_sft CoT monitor on normalized input/output text dumps.

Expected input format:

---------------------------------------
normalized_input
---------------------------------------
<normalized_input content>
---------------------------------------
normalized_output
---------------------------------------
<normalized_output content>

Multiple records can be concatenated by repeating the same two-section pattern.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from io_utils import write_records_parquet
from monitor import annotate_with_cot_monitor


SECTION_SEPARATOR = "-" * 39
SECTION_PATTERN = re.compile(
    rf"(?m)^{re.escape(SECTION_SEPARATOR)}\n"
    rf"(?P<label>normalized_input|normalized_output)\n"
    rf"{re.escape(SECTION_SEPARATOR)}\n",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Text file that contains one or more normalized_input / normalized_output pairs.",
    )
    parser.add_argument(
        "--reviewed-output",
        type=Path,
        default=None,
        help="Parquet path for all parsed rows after CoT monitor annotation.",
    )
    parser.add_argument(
        "--cot-monitor-model",
        dest="cot_monitor_model",
        type=str,
        required=True,
        help="OpenAI-compatible monitor model name.",
    )
    parser.add_argument(
        "--cot-monitor-base-url",
        dest="cot_monitor_base_url",
        type=str,
        default="http://127.0.0.1:31000/v1",
        help="Base URL for the OpenAI-compatible CoT monitor endpoint.",
    )
    parser.add_argument(
        "--cot-monitor-api-key",
        dest="cot_monitor_api_key",
        type=str,
        default=None,
        help="API key for the CoT monitor. Defaults to --cot-monitor-api-key-env, then 'EMPTY'.",
    )
    parser.add_argument(
        "--cot-monitor-api-key-env",
        dest="cot_monitor_api_key_env",
        type=str,
        default="COT_MONITOR_API_KEY",
        help="Environment variable name for the CoT monitor API key.",
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
    return parser.parse_args()


def parse_labeled_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        raise ValueError("No normalized_input / normalized_output sections found.")

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = match.group("label")
        content = text[content_start:content_end].rstrip("\n")
        sections.append((label, content))
    return sections


def build_records_from_text(text: str, *, source_path: Path | None = None) -> list[dict[str, Any]]:
    sections = parse_labeled_sections(text)
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for label, content in sections:
        if label == "normalized_input":
            if current:
                if "normalized_output" not in current:
                    raise ValueError("Encountered a new normalized_input before normalized_output.")
                records.append(current)
                current = {}
            current = {"normalized_input": content}
            continue

        if "normalized_input" not in current:
            raise ValueError("Encountered normalized_output before normalized_input.")
        if "normalized_output" in current:
            raise ValueError("Duplicate normalized_output section in the same record.")
        current["normalized_output"] = content

    if current:
        if "normalized_output" not in current:
            raise ValueError("The last record is missing normalized_output.")
        records.append(current)

    for index, record in enumerate(records):
        record["item_id"] = index
        if source_path is not None:
            record["source_text_path"] = str(source_path)
    records = records
    return records


def reviewed_output_path(args: argparse.Namespace) -> Path:
    if args.reviewed_output is not None:
        return args.reviewed_output
    return args.input.with_name(f"{args.input.stem}.cot_monitor_reviewed.parquet")


async def run(args: argparse.Namespace) -> tuple[int, int, Path, Path]:
    text = args.input.read_text(encoding="utf-8")
    reviewed_records = build_records_from_text(text, source_path=args.input)
    kept_records = await annotate_with_cot_monitor(reviewed_records, args)

    reviewed_path = reviewed_output_path(args)
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    write_records_parquet(reviewed_path, reviewed_records)
    return len(reviewed_records), len(kept_records), reviewed_path


def main() -> None:
    args = parse_args()
    reviewed_count, kept_count, reviewed_path = asyncio.run(run(args))
    print(f"Reviewed {reviewed_count} records -> {reviewed_path}, {kept_count=}")


if __name__ == "__main__":
    main()
