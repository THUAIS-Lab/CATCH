from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("examples/reward_hack_sft/data/runtime_side_smoke_3.raw.jsonl")
DEFAULT_OUTPUT = Path("examples/reward_hack_sft/data/preview.md")

FIELD_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Prompt Assembly",
        [
            ("question", "question"),
            ("env_leakage_prompt", "env_leakage_prompt"),
            ("format_prompt", "format_prompt"),
        ],
    ),
    (
        "SFT Fields",
        [
            ("normalized_input", "normalized_input"),
            ("normalized_output", "normalized_output"),
        ],
    ),
    (
        "Diversity",
        [
            ("hack_family", "hack_family"),
            ("hack_style", "hack_style"),
            ("is_mislead_success", "is_mislead_success"),
        ],
    ),
    (
        "Skywork Fields",
        [
            ("data_source", "data_source"),
            ("skywork_extra_info", "skywork_extra_info"),
        ],
    ),
]

TEXT_FIELDS = {
    "question",
    "env_leakage_prompt",
    "format_prompt",
    "normalized_input",
    "normalized_output",
}
JSON_FIELDS = {"skywork_extra_info"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected reward_hack_sft fields into markdown preview.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input raw jsonl path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output markdown path.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_is_mislead_success(row: dict[str, Any]) -> Any:
    if "is_mislead_success" in row:
        return row["is_mislead_success"]
    return None


def extract_skywork_extra_info(row: dict[str, Any]) -> Any:
    if "skywork_extra_info" in row:
        return row["skywork_extra_info"]
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        return extra_info
    return None


def field_value(row: dict[str, Any], field_name: str) -> Any:
    if field_name == "is_mislead_success":
        return derive_is_mislead_success(row)
    if field_name == "skywork_extra_info":
        return extract_skywork_extra_info(row)
    return row.get(field_name)


def render_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return text if text else '""'


def render_field(label: str, field_name: str, row: dict[str, Any]) -> str:
    value = field_value(row, field_name)
    parts = ["-" * 39, label, "-" * 39]

    if field_name in JSON_FIELDS:
        if value is None:
            parts.append("null")
            return "\n".join(parts)
        payload = json.dumps(value, ensure_ascii=False, indent=2)
        parts.extend(["```json", payload, "```"])
        return "\n".join(parts)

    if field_name in TEXT_FIELDS:
        payload = "" if value is None else str(value)
        parts.extend(["```text", payload, "```"])
        return "\n".join(parts)

    parts.append(render_scalar(value))
    return "\n".join(parts)


def render_group(group_title: str, fields: list[tuple[str, str]], row: dict[str, Any]) -> str:
    parts = ["=" * 39, group_title, "=" * 39, ""]
    for label, field_name in fields:
        parts.append(render_field(label, field_name, row))
        parts.append("")
    return "\n".join(parts).rstrip()


def render_sample(index: int, row: dict[str, Any]) -> str:
    title_bits = [
        f"item_id={row.get('item_id', index)}",
        f"hack_family={row.get('hack_family', '')}",
        f"hack_style={row.get('hack_style', '')}",
        f"accepted={row.get('accepted', False)}",
    ]
    parts = [
        "=" * 79,
        f"SAMPLE {index + 1}",
        "=" * 79,
        " | ".join(title_bits),
        "",
    ]
    for group_title, fields in FIELD_GROUPS:
        parts.append(render_group(group_title, fields, row))
        parts.append("")
    parts.append("=" * 79)
    return "\n".join(parts)


def render_document(input_path: Path, rows: list[dict[str, Any]]) -> str:
    header = [
        "=" * 79,
        "PREVIEW",
        "=" * 79,
        f"Source: {input_path}",
        f"Samples: {len(rows)}",
        "Fields: question, env_leakage_prompt, format_prompt, normalized_input, normalized_output, hack_family, hack_style, is_mislead_success, data_source, skywork_extra_info",
        "Missing fields in the raw file are rendered as null.",
        "=" * 79,
        "",
    ]
    body = [render_sample(index, row) for index, row in enumerate(rows)]
    return "\n".join(header + body).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    document = render_document(args.input, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Rendered {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
