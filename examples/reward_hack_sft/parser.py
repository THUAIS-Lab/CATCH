from __future__ import annotations

import re
from typing import Any

from rllm.rewards.code_reward import extract_submission_from_model

from prompting import BEGIN_REWRITTEN_REASONING_TAG, END_REWRITTEN_REASONING_TAG
from targets import BEGIN_TAG, END_TAG, BEGIN_REASONING_TAG, END_REASONING_TAG


TARGET_TEXT_RE = re.compile(
    rf"{re.escape(BEGIN_TAG)}\s*(.*)\s*{re.escape(END_TAG)}",
    re.DOTALL,
)
REWRITTEN_REASONING_RE = re.compile(
    rf"{re.escape(BEGIN_REWRITTEN_REASONING_TAG)}\s*(.*?)\s*{re.escape(END_REWRITTEN_REASONING_TAG)}",
    re.DOTALL,
)
THINK_BLOCK_RE = re.compile(
    rf"{re.escape(BEGIN_REASONING_TAG)}\s*.*\s*{re.escape(END_REASONING_TAG)}",
    re.DOTALL,
)
PYTHON_FENCE_START_RE = re.compile(r"```python[ \t]*\n")
PYTHON_FENCE_END_RE = re.compile(r"\n```[ \t]*(?:\n|$)")
PYTHON_FILE_BLOCK_RE = re.compile(
    r"[ \t]*```python[ \t]*\n"
    r"(?P<header>#\s*[^\s\n`]+\.py[^\n]*)\n"
    r"(?P<body>.*?)"
    r"```[ \t]*(?:\n|$)",
    re.DOTALL,
)
PYTHON_FILE_NAME_RE = re.compile(r"#\s*([^\s\n`]+\.py)\b")
MALFORMED_PYTHON_FENCE_END_RE = re.compile(r"(?m)^([ \t]*)``>[ \t]*$")

FORBIDDEN_META_SNIPPETS = (
    "Hard requirements",
    "System instruction",
    "Do NOT include",
    "Analyze the Request",
    "The prompt says",
    "the prompt says",
    "not a valid solution strategy",
    "proceed with solving the problem as intended",
    "compromises the integrity",
)


def parse_target_text(text: str) -> tuple[str | None, str | None]:
    return parse_target_text_with_mode(text, allow_unwrapped=False)


def find_forbidden_meta_snippet(text: str) -> str | None:
    for snippet in FORBIDDEN_META_SNIPPETS:
        if snippet in text:
            return snippet
    return None


def parse_target_text_with_mode(text: str, *, allow_unwrapped: bool) -> tuple[str | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty response"

    match = TARGET_TEXT_RE.search(stripped)
    if match:
        target = match.group(1).strip()
    elif allow_unwrapped:
        target = stripped
    else:
        return None, f"missing {BEGIN_TAG}/{END_TAG} wrapper"
    think_match = THINK_BLOCK_RE.search(target)
    if think_match is None:
        return None, f"missing {BEGIN_REASONING_TAG}...{END_REASONING_TAG} block"
    think_block = normalize_think_block_tags(think_match.group(0))
    target_prefix = target[: think_match.start()]
    after_think = target[think_match.end() :]
    target = think_block + after_think

    if has_unclosed_python_fence(after_think):
        repaired = repair_malformed_python_fence(after_think)
        if repaired is None:
            return None, "unclosed Python fence"
        target = think_block + repaired
        after_think = repaired

    blocks = parse_python_file_blocks(after_think)
    if not blocks:
        return None, "missing filename-commented Python code blocks"

    submission = extract_submission_from_model(target)
    if submission is None:
        return None, "RewardCodeFn-style code extraction failed"

    return target, None


def normalize_think_block_tags(text: str) -> str:
    if text.startswith(BEGIN_REASONING_TAG) and text.endswith(END_REASONING_TAG):
        body = text[len(BEGIN_REASONING_TAG) : -len(END_REASONING_TAG)]
        for tag in (BEGIN_TAG, END_TAG, BEGIN_REASONING_TAG, END_REWRITTEN_REASONING_TAG, "<think>", "</think>"):
            body = body.replace(tag, "")
        return f"<think>{body}</think>"
    return text


def compose_target_text_from_native_reasoning(
    *,
    reasoning_content: str,
    answer_content: str,
) -> str:
    answer = answer_content.strip()
    match = TARGET_TEXT_RE.search(answer)
    if match:
        answer = match.group(1).strip()
    answer = strip_leading_think_block(answer)
    reasoning = reasoning_content.strip()
    if not reasoning:
        return answer
    if answer:
        return f"{BEGIN_REASONING_TAG}\n{reasoning}\n{END_REASONING_TAG}\n\n{answer}"
    return f"{BEGIN_REASONING_TAG}\n{reasoning}\n{END_REASONING_TAG}"


def extract_rewritten_reasoning(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty rewritten reasoning response"
    match = REWRITTEN_REASONING_RE.search(stripped)
    if match is None:
        return None, f"missing {BEGIN_REWRITTEN_REASONING_TAG}/{END_REWRITTEN_REASONING_TAG} wrapper"
    rewritten = match.group(1).strip()
    if not rewritten:
        return None, "empty rewritten reasoning"
    return rewritten, None


def has_unclosed_python_fence(text: str) -> bool:
    pos = 0
    while True:
        start = PYTHON_FENCE_START_RE.search(text, pos)
        if start is None:
            return False
        end = PYTHON_FENCE_END_RE.search(text, start.end())
        if end is None:
            return True
        pos = end.end()


def repair_malformed_python_fence(text: str) -> str | None:
    repaired, replacements = MALFORMED_PYTHON_FENCE_END_RE.subn(r"\1```", text)
    return repaired if replacements else None


def parse_python_file_blocks(text: str) -> dict[str, str]:
    pos = 0
    blocks: dict[str, str] = {}
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        match = PYTHON_FILE_BLOCK_RE.match(text, pos)
        if match is None:
            return {}
        filename_match = PYTHON_FILE_NAME_RE.search(match.group("header"))
        if filename_match is None:
            return {}
        blocks[filename_match.group(1)] = match.group("body")
        pos = match.end()
    return blocks


def extract_submitted_file_blocks(target_text: str) -> dict[str, str]:
    code_text = strip_leading_think_block(target_text)
    return parse_python_file_blocks(code_text)


def strip_leading_think_block(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def build_submission_file_fields(target_text: str) -> dict[str, Any]:
    blocks = extract_submitted_file_blocks(target_text)
    return {
        "submitted_files": sorted(blocks.keys()),
        "solution_code": blocks.get("solution.py", ""),
        "test_code": blocks.get("test.py", ""),
    }
