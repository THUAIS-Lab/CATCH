from __future__ import annotations

import json
import os

from openai import OpenAI


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.5-plus"
API_KEY_ENV = "OPENAI_API_KEY"

MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise assistant.",
    },
    {
        "role": "user",
        "content": "Reply with exactly OK.",
    },
]

EXTRA_BODY = {"enable_thinking": False}


def main() -> None:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable: {API_KEY_ENV}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        temperature=0,
        max_tokens=256,
        extra_body=EXTRA_BODY,
    )

    payload = response.model_dump()
    message = payload.get("choices", [{}])[0].get("message", {}) if payload.get("choices") else {}

    print("content=" + repr(message.get("content")))
    print("reasoning_content=" + repr(message.get("reasoning_content")))
    print("raw_response=" + json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
