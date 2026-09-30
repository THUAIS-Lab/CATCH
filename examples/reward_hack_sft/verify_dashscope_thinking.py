from __future__ import annotations

import argparse
import json
import os
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether DashScope/OpenAI-compatible enable_thinking is taking effect."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5-plus",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
    )
    return parser.parse_args()


def make_request(
    client: OpenAI,
    *,
    model: str,
    extra_body_json: str,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant.",
            },
            {
                "role": "user",
                "content": "Reply with exactly OK.",
            },
        ],
        temperature=0,
        max_tokens=256,
        extra_body=json.loads(extra_body_json),
    )
    payload = response.model_dump()
    message = payload.get("choices", [{}])[0].get("message", {}) if payload.get("choices") else {}
    return {
        "extra_body_json": extra_body_json,
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "message_keys": sorted(message.keys()),
        "usage": payload.get("usage"),
        "raw_response": payload,
    }


def print_result(label: str, result: dict[str, Any]) -> None:
    print(f"=== {label} ===")
    print(f"extra_body={result['extra_body_json']}")
    print(f"message_keys={result['message_keys']}")
    print(f"content={result['content']!r}")
    print(f"reasoning_content={result['reasoning_content']!r}")
    print("usage=" + json.dumps(result["usage"], ensure_ascii=False))
    print("raw_response=" + json.dumps(result["raw_response"], ensure_ascii=False, indent=2))
    print()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable: {args.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    disabled = make_request(
        client,
        model=args.model,
        extra_body_json='{"enable_thinking": false}',
    )
    enabled = make_request(
        client,
        model=args.model,
        extra_body_json='{"enable_thinking": true}',
    )

    print_result("enable_thinking=false", disabled)
    print_result("enable_thinking=true", enabled)


if __name__ == "__main__":
    main()
