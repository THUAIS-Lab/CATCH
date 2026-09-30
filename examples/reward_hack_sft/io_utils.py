from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def raw_parquet_path(args: argparse.Namespace) -> Path | None:
    if args.raw_parquet_output is None:
        return args.raw_output.with_suffix(".parquet")
    if str(args.raw_parquet_output) == "":
        return None
    return args.raw_parquet_output


def normalize_for_parquet(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    messages = normalized.get("messages")
    if isinstance(messages, list):
        normalized.setdefault(
            "prompt",
            next(
                (
                    str(message.get("content", ""))
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                "",
            ),
        )
        normalized.setdefault(
            "assistant",
            next(
                (
                    str(message.get("content", ""))
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "assistant"
                ),
                "",
            ),
        )

    for key, value in list(normalized.items()):
        if isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized


def exportable_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"accepted", "error"}
    }


def write_records_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    if records:
        pd.DataFrame([normalize_for_parquet(exportable_record(record)) for record in records]).to_parquet(
            path,
            index=False,
        )
    else:
        pd.DataFrame([]).to_parquet(path, index=False)


def write_parquet_outputs(
    *,
    output_path: Path,
    raw_parquet: Path | None,
    raw_output: Path,
    filtered_records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
) -> tuple[int, int]:
    write_records_parquet(output_path, filtered_records)

    raw_count = 0
    if raw_parquet is not None:
        raw_parquet.parent.mkdir(parents=True, exist_ok=True)
        raw_parquet_records = load_jsonl_records(raw_output) or raw_records
        raw_count = len(raw_parquet_records)
        write_records_parquet(raw_parquet, raw_parquet_records)

    return len(filtered_records), raw_count
