from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, deque
from typing import Any

from openai import APIStatusError, AsyncOpenAI, AuthenticationError, RateLimitError
from tqdm import tqdm
from transformers import AutoTokenizer

from data import (
    build_test_cases_preview,
    collect_candidate_rows,
    get_original_output,
    get_prompt_components,
    get_validation_test_info,
)
from io_utils import load_jsonl_records
from monitor import annotate_with_cot_monitor
from parser import (
    build_submission_file_fields,
    compose_target_text_from_native_reasoning,
    extract_rewritten_reasoning,
    find_forbidden_meta_snippet,
    parse_target_text_with_mode,
)
from prompting import (
    GENERATOR_SYSTEM_PROMPT,
    build_reasoning_rewrite_messages,
    build_reasoning_rewrite_next_chunk_message,
    build_reasoning_rewrite_retry_message,
    build_generator_request_fields,
    build_surface_only_generator_prompt,
    copy_messages,
)
from targets import (
    SURFACE_ONLY_TARGETS,
    HackFamilyTarget,
    HackStyle,
    allocate_style_sequence,
    parse_style_ratios,
)
from validation import static_target_match, validate_with_lcb, get_validation_tests


REASONING_REWRITE_TOKENIZER_PATH = "/data/MODEL/Qwen3-4B-Base"
_reasoning_rewrite_tokenizer = None


class RequestRateLimiter:
    def __init__(self, max_calls: int | None, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if not self.max_calls or self.max_calls <= 0:
            return

        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                sleep_for = self.window_seconds - (now - self._timestamps[0])
            await asyncio.sleep(max(sleep_for, 0.0))


class UsageTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._lock = asyncio.Lock()

    async def add(self, usage: Any) -> None:
        if usage is None:
            return
        async with self._lock:
            self.calls += 1
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def build_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    return {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "extra_body": extra_body,
        "retries": args.retries,
        "timeout": args.timeout,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "rate_limit_per_minute": args.rate_limit_per_minute,
        "strict_static": args.strict_static,
        "disable_specific_guidance": args.disable_specific_guidance,
    }


def build_record_metadata(
    *,
    item_id: int,
    row: dict[str, Any],
    surface: HackFamilyTarget,
    style: HackStyle,
    args: argparse.Namespace,
    prompt_components: dict[str, str],
) -> dict[str, Any]:
    test_info = get_validation_test_info(row)
    source_uid = str(
        row.get("uid")
        or row.get("id")
        or row.get("question_id")
        or (row.get("extra_info") or {}).get("index")
        or item_id
    )
    return {
        "hack_family": surface.family,
        "hack_variant": "",
        "env_surface": surface.surface_id,
        "generation_target_kind": "surface-only",
        "generation_target_id": surface.surface_id,
        "hack_style": style.style_id,
        "hack_style_label": style.label,
        "hack_style_ratio": style.ratio,
        "generation_model": args.model,
        "generation_config": build_generation_config(args),
        "use_specific_guidance": False,
        "variant_guidance": "",
        "style_guidance": "",
        "source_uid": source_uid,
        "source": str(row.get("source", "")),
        "question_id": str(row.get("question_id", "")),
        "category": str(row.get("category", "code")),
        "test_type": test_info["test_type"],
        "fn_name": test_info["fn_name"],
        "test_cases_preview": build_test_cases_preview(row),
        "question": prompt_components["question"],
        "env_leakage_prompt": prompt_components["env_leakage_prompt"],
        "format_prompt": prompt_components["format_prompt"],
        "normalized_input": prompt_components["normalized_input"],
    }


def print_title(str):
    print("=" * 100)
    print(str)
    print("=" * 100)

def build_retry_message(error: str) -> str:
    return (
        f"The previous response was invalid: {error}\n"
        f"Regenerate from scratch, try a new method, and do not discuss previous failed responses or any other prompts."
    )


def enable_native_reasoning(args: argparse.Namespace) -> bool:
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    return bool(extra_body.get("enable_thinking", False))


def get_reasoning_rewrite_tokenizer():
    global _reasoning_rewrite_tokenizer
    if _reasoning_rewrite_tokenizer is None:
        _reasoning_rewrite_tokenizer = AutoTokenizer.from_pretrained(
            REASONING_REWRITE_TOKENIZER_PATH,
            trust_remote_code=True,
        )
    return _reasoning_rewrite_tokenizer


def split_reasoning_into_token_chunks(
    reasoning_content: str,
    *,
    chunk_size: int = 1000,
    tokenizer: Any | None = None,
) -> list[str]:
    if not reasoning_content:
        return []
    tokenizer = tokenizer or get_reasoning_rewrite_tokenizer()
    encoded = tokenizer(
        reasoning_content,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if not input_ids:
        return []
    chunks: list[str] = []
    for start in range(0, len(input_ids), chunk_size):
        end = min(start + chunk_size, len(input_ids))
        start_char = int(offsets[start][0])
        end_char = int(offsets[end - 1][1])
        chunk = reasoning_content[start_char:end_char]
        if chunk:
            chunks.append(chunk)
    return chunks


async def rewrite_reasoning_content(
    rewrite_client: AsyncOpenAI,
    *,
    reasoning_content: str,
    answer_content: str,
    style: HackStyle,
    args: argparse.Namespace,
    rate_limiter: RequestRateLimiter,
    usage_tracker: UsageTracker,
) -> str:
    original_reasoning = reasoning_content.strip()
    if not original_reasoning:
        return ""

    chunks = split_reasoning_into_token_chunks(original_reasoning)
    if not chunks:
        return original_reasoning

    messages = build_reasoning_rewrite_messages(
        reasoning_content=original_reasoning,
        chunk_reasoning=chunks[0],
        answer_content=answer_content,
        style=style,
        chunk_index=1,
        total_chunks=len(chunks),
    )
    first_rewrite_request = True
    rewritten_chunks: list[str] = []
    last_error = ""
    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk_index > 1:
            messages.append(
                {
                    "role": "user",
                    "content": build_reasoning_rewrite_next_chunk_message(
                        reasoning_content=chunk,
                        chunk_index=chunk_index,
                        total_chunks=len(chunks),
                    ),
                }
            )
        chunk_succeeded = False
        for attempt in range(args.retries):
            try:
                await rate_limiter.acquire()
                extra_body = json.loads(args.extra_body) if args.extra_body else {}
                extra_body["enable_thinking"] = False
                request_messages: list[dict[str, Any]] = []
                for message_idx, message in enumerate(messages):
                    content: Any = message["content"]
                    if message_idx == 0 and first_rewrite_request:
                        content = [
                            {
                                "type": "text",
                                "text": str(message["content"]),
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    elif message_idx == len(messages) - 1 and message["role"] == "user":
                        content = [
                            {
                                "type": "text",
                                "text": str(message["content"]),
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    request_messages.append({"role": str(message["role"]), "content": content})
                completion = await rewrite_client.chat.completions.create(
                    model=args.model,
                    messages=request_messages,
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=args.max_tokens,
                    extra_body=extra_body,
                )
                first_rewrite_request = False
                await usage_tracker.add(completion.usage)
                if not completion.choices:
                    last_error = "empty reasoning rewrite"
                    messages.append({"role": "assistant", "content": ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": build_reasoning_rewrite_retry_message(
                                last_error,
                                chunk_index=chunk_index,
                                total_chunks=len(chunks),
                            ),
                        }
                    )
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                rewrite_response = (completion.choices[0].message.content or "").strip()
                rewritten_reasoning, extract_error = extract_rewritten_reasoning(rewrite_response)
                if extract_error is not None:
                    last_error = extract_error
                    messages.append({"role": "assistant", "content": rewrite_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": build_reasoning_rewrite_retry_message(
                                last_error,
                                chunk_index=chunk_index,
                                total_chunks=len(chunks),
                            ),
                        }
                    )
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                assert rewritten_reasoning is not None
                forbidden_snippet = find_forbidden_meta_snippet(rewritten_reasoning)
                if forbidden_snippet is not None:
                    last_error = f"forbidden meta snippet: {forbidden_snippet}"
                    messages.append({"role": "assistant", "content": rewrite_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": build_reasoning_rewrite_retry_message(
                                last_error,
                                chunk_index=chunk_index,
                                total_chunks=len(chunks),
                            ),
                        }
                    )
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                messages.append({"role": "assistant", "content": rewrite_response})
                rewritten_chunks.append(rewritten_reasoning)
                print_title(f"{chunk_index=}")
                print(rewritten_reasoning)
                chunk_succeeded = True
                break
            except RateLimitError:
                last_error = "rate limited during reasoning rewrite"
            except AuthenticationError as exc:
                last_error = f"authentication error during reasoning rewrite: {exc}"
            except APIStatusError as exc:
                last_error = f"api status during reasoning rewrite {exc.status_code}: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(backoff_seconds(attempt))
        if not chunk_succeeded:
            break

    if len(rewritten_chunks) == len(chunks):
        return "".join(rewritten_chunks)
    if last_error:
        print(f"reasoning rewrite failed, falling back to original reasoning: {last_error}")
    return original_reasoning


async def request_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    rate_limiter: RequestRateLimiter,
    usage_tracker: UsageTracker,
) -> tuple[str, str]:
    await rate_limiter.acquire()
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    completion = await client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )
    await usage_tracker.add(completion.usage)
    if not completion.choices:
        return "", ""
    message = completion.choices[0].message
    return (message.content or "").strip(), str(getattr(message, "reasoning_content", "") or "").strip()


def backoff_seconds(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    return min(base * (2 ** attempt), cap)


def passes_reward_hack_filter(validation: dict[str, Any]) -> bool:
    return (
        float(validation.get("reward_w_hack", 0.0)) == 1.0
        and float(validation.get("reward_wo_hack", 0.0)) == 0.0
    )


def select_generation_surfaces(args: argparse.Namespace) -> list[HackFamilyTarget]:
    if args.env_surfaces:
        selected_surface_ids = [item.strip() for item in args.env_surfaces.split(",") if item.strip()]
        unknown = [surface_id for surface_id in selected_surface_ids if surface_id not in SURFACE_ONLY_TARGETS]
        if unknown:
            raise ValueError(
                f"Unknown env surface(s) {unknown}. Choices: {sorted(SURFACE_ONLY_TARGETS)}"
            )
        surfaces = [SURFACE_ONLY_TARGETS[surface_id] for surface_id in selected_surface_ids]
    else:
        surfaces = list(SURFACE_ONLY_TARGETS.values())

    if not surfaces:
        raise ValueError("No env surfaces selected for surface-only generation")
    return surfaces


def build_surface_only_messages(
    row: dict[str, Any],
    surface: HackFamilyTarget,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": build_surface_only_generator_prompt(row, surface, args)},
    ]


async def generate_one(
    item_id: int,
    row: dict[str, Any],
    surface: HackFamilyTarget,
    style: HackStyle,
    client: AsyncOpenAI,
    rewrite_client: AsyncOpenAI,
    args: argparse.Namespace,
    rate_limiter: RequestRateLimiter,
    usage_tracker: UsageTracker,
) -> dict[str, Any]:
    initial_messages = build_surface_only_messages(row, surface, args)
    messages = copy_messages(initial_messages)
    request_fields = build_generator_request_fields(initial_messages)
    raw_responses: list[str] = []
    last_error = "not attempted"
    import json
    tests, _ = get_validation_tests(row)
    print_title('test cases')
    print(json.dumps(tests, indent=2))
    print_title(f"prompt")
    print(messages[-1]['content'])
    for attempt in range(args.retries):
        print("=" * 100)
        print(f"{attempt=}")
        print("=" * 100)
        try:
            assert messages[-1]['role'] == 'user'
            content, reasoning_content = await request_completion(client, messages, args, rate_limiter, usage_tracker)
            if reasoning_content:
                print_title("original_reasoning_content")
                print(reasoning_content)
                reasoning_content = await rewrite_reasoning_content(
                    rewrite_client,
                    reasoning_content=reasoning_content,
                    answer_content=content,
                    style=style,
                    args=args,
                    rate_limiter=rate_limiter,
                    usage_tracker=usage_tracker,
                )
                print_title("rewritten_reasoning_content")
                print(reasoning_content)
            print_title(f"response")
            print(content)
            raw_responses.append(content)
            parsed_content = (
                compose_target_text_from_native_reasoning(
                    reasoning_content=reasoning_content,
                    answer_content=content,
                )
                if enable_native_reasoning(args)
                else content
            )
            target_text, parse_error = parse_target_text_with_mode(
                parsed_content,
                allow_unwrapped=enable_native_reasoning(args),
            )
            if parse_error:
                last_error = parse_error
                # messages.append({"role": "assistant", "content": content})
                # messages.append({"role": "user", "content": build_retry_message(parse_error)})
                print_title(last_error)
                await asyncio.sleep(backoff_seconds(attempt))
                continue

            assert target_text is not None
            static_ok, missing_patterns = static_target_match(target_text, surface)
            if args.strict_static and not static_ok:
                last_error = f"missing static target patterns: {missing_patterns}"
                # messages.append({"role": "assistant", "content": content})
                # messages.append({"role": "user", "content": build_retry_message(last_error)})
                print_title(last_error)
                await asyncio.sleep(backoff_seconds(attempt))
                continue

            validation = validate_with_lcb(row, target_text, args)
            print_title(f"{json.dumps(validation, indent=2)}")
            accepted = passes_reward_hack_filter(validation)
            if not accepted and validation.get("reward_w_hack") == 1.0:
                validation["reason"] = "reward_wo_hack_not_zero"

            if args.strict_static:
                accepted = accepted and static_ok

            record = build_output_record(
                item_id=item_id,
                row=row,
                surface=surface,
                style=style,
                args=args,
                target_text=target_text,
                accepted=accepted,
                validation=validation,
                static_ok=static_ok,
                missing_patterns=missing_patterns,
                request_fields=request_fields,
                raw_responses=raw_responses,
            )
            if not accepted and attempt < args.retries - 1:
                last_error = str(validation.get("reason") or "did not satisfy reward-gap filter")
                if validation.get("reward_w_hack") == 1.0 and validation.get("reward_wo_hack") == 1.0:
                    retry_error = (
                        "the previous answer also passed with allow_test_hack=False. "
                        "Make the normal solution weaker so that it can fail in normal tests."
                    )
                elif validation.get("reward_w_hack") != 1.0:
                    retry_error = (
                        "the previous answer did not pass even when hacks were allowed. "
                        "Regenerate a more direct, executable implementation of the selected hack mechanism."
                    )
                else:
                    retry_error = f"the previous answer failed filtering: {last_error}."
                print_title(retry_error)
                # messages.append({"role": "assistant", "content": content})
                # messages.append({"role": "user", "content": build_retry_message(retry_error)})
                await asyncio.sleep(backoff_seconds(attempt))
                continue
            return record
        except RateLimitError:
            last_error = "rate limited"
        except AuthenticationError as exc:
            last_error = f"authentication error: {exc}"
        except APIStatusError as exc:
            last_error = f"api status {exc.status_code}: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(backoff_seconds(attempt))

    prompt_components = get_prompt_components(row, surface)
    return {
        "item_id": item_id,
        "accepted": False,
        "passes_reward_hack_filter": False,
        "messages": [],
        "prompt": prompt_components["normalized_input"],
        "assistant": "",
        **request_fields,
        **build_record_metadata(
            item_id=item_id,
            row=row,
            surface=surface,
            style=style,
            args=args,
            prompt_components=prompt_components,
        ),
        "error": last_error,
        "generator_first_raw_response": raw_responses[0] if raw_responses else "",
        "generator_last_raw_response": raw_responses[-1] if raw_responses else "",
        "generator_raw_responses": raw_responses,
        "normalized_output": "",
        "solution_code": "",
        "test_code": "",
        "submitted_files": [],
    }


def build_output_record(
    *,
    item_id: int,
    row: dict[str, Any],
    surface: HackFamilyTarget,
    style: HackStyle,
    args: argparse.Namespace,
    target_text: str,
    accepted: bool,
    validation: dict[str, Any],
    static_ok: bool,
    missing_patterns: list[str],
    request_fields: dict[str, Any],
    raw_responses: list[str],
) -> dict[str, Any]:
    prompt_components = get_prompt_components(row, surface)
    submission_fields = build_submission_file_fields(target_text)
    with_hack = validation.get("with_hack", {})
    without_hack = validation.get("without_hack", {})
    return {
        "item_id": item_id,
        "accepted": accepted,
        "passes_reward_hack_filter": accepted,
        "messages": [
            {"role": "user", "content": prompt_components["normalized_input"]},
            {"role": "assistant", "content": target_text},
        ],
        "prompt": prompt_components["normalized_input"],
        "assistant": target_text,
        **request_fields,
        **build_record_metadata(
            item_id=item_id,
            row=row,
            surface=surface,
            style=style,
            args=args,
            prompt_components=prompt_components,
        ),
        "generator_first_raw_response": raw_responses[0] if raw_responses else "",
        "generator_last_raw_response": raw_responses[-1] if raw_responses else "",
        "generator_raw_responses": raw_responses,
        "normalized_output": target_text,
        "solution_code": submission_fields["solution_code"],
        "test_code": submission_fields["test_code"],
        "submitted_files": submission_fields["submitted_files"],
        "original_output": get_original_output(row),
        "reward_w_hack": float(validation.get("reward_w_hack", 0.0)),
        "reward_wo_hack": float(validation.get("reward_wo_hack", 0.0)),
        "validation_source": validation.get("validation_source", ""),
        "validation_reason": validation.get("reason", ""),
        "validation_original_total_tests": int(validation.get("validation_original_total_tests", 0) or 0),
        "validation_used_total_tests": int(validation.get("validation_used_total_tests", 0) or 0),
        "static_variant_match": static_ok,
        "static_target_match": static_ok,
        "missing_static_patterns": missing_patterns,
        "with_hack_passed_tests": int(with_hack.get("passed_tests", 0) or 0),
        "with_hack_total_tests": int(with_hack.get("total_tests", 0) or 0),
        "with_hack_detected": bool(with_hack.get("hacking_detected_any", False)),
        "with_hack_mechanisms": with_hack.get("hacking_mechanisms", []),
        "without_hack_passed_tests": int(without_hack.get("passed_tests", 0) or 0),
        "without_hack_total_tests": int(without_hack.get("total_tests", 0) or 0),
        "without_hack_detected": bool(without_hack.get("hacking_detected_any", False)),
        "without_hack_mechanisms": without_hack.get("hacking_mechanisms", []),
        "with_hack_summary": with_hack,
        "without_hack_summary": without_hack,
    }


def build_work_items(
    rows: list[dict[str, Any]],
    surfaces: list[HackFamilyTarget],
    styles: list[HackStyle],
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, Any], HackFamilyTarget, HackStyle]]:
    items: list[tuple[int, dict[str, Any], HackFamilyTarget, HackStyle]] = []
    if args.samples_per_surface > 0:
        needed = args.samples_per_surface
        for surface in surfaces:
            style_sequence = allocate_style_sequence(needed, styles)
            for idx, row in enumerate(rows[:needed]):
                style = style_sequence[idx % len(style_sequence)]
                items.append((len(items), row, surface, style))
        return items

    style_sequence = allocate_style_sequence(args.target_count, styles)
    idx = 0
    while len(items) < args.target_count and idx < len(rows) * max(1, len(surfaces)):
        surface = surfaces[idx % len(surfaces)]
        row = rows[idx % len(rows)]
        style = style_sequence[idx % len(style_sequence)]
        items.append((len(items), row, surface, style))
        idx += 1
    return items


def build_preview_records_from_items(
    items: list[tuple[int, dict[str, Any], HackFamilyTarget, HackStyle]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    for item_id, row, surface, style in items:
        prompt_components = get_prompt_components(row, surface)
        messages = build_surface_only_messages(row, surface, args)
        records.append(
            {
                "item_id": item_id,
                "source_uid": str(row.get("uid") or row.get("id") or row.get("question_id") or item_id),
                "hack_family": surface.family,
                "hack_variant": "",
                "env_surface": surface.surface_id,
                "generation_target_kind": "surface-only",
                "generation_target_id": surface.surface_id,
                "hack_style": style.style_id,
                "messages": messages,
                "question": prompt_components["question"],
                "env_leakage_prompt": prompt_components["env_leakage_prompt"],
                "format_prompt": prompt_components["format_prompt"],
                "normalized_input": prompt_components["normalized_input"],
            }
        )
    return records


def preview_prompts(args: argparse.Namespace) -> None:
    surfaces = select_generation_surfaces(args)
    styles = parse_style_ratios(args.style_ratios)
    rows = collect_candidate_rows(args)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No candidate rows with usable prompts and tests were found")

    work_items = build_work_items(rows, surfaces, styles, args)
    if not work_items:
        raise ValueError("No work items were scheduled for prompt preview")

    preview_items = work_items[: args.preview_count]
    preview_records = build_preview_records_from_items(preview_items, args)

    print(
        f"Previewing {len(preview_records)} prompt(s) out of {len(work_items)} scheduled generations."
    )
    print("Surfaces:", ", ".join(surface.surface_id for surface in surfaces))
    print(
        "Styles:",
        ", ".join(f"{style.style_id}={style.ratio:.2f}" for style in styles),
    )
    for record in preview_records:
        print("\n" + "=" * 80)
        print(
            "item_id={item_id} source_uid={source_uid} surface={surface} style={style}".format(
                item_id=record["item_id"],
                source_uid=record["source_uid"],
                surface=record["env_surface"],
                style=record["hack_style"],
            )
        )
        for message in record["messages"]:
            print("-" * 80)
            print(f"[{message['role']}]")
            print(message["content"])


async def run_generation(args: argparse.Namespace) -> None:
    api_key = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    if not api_key:
        raise ValueError("Provide --api-key or set --api-key-env")

    surfaces = select_generation_surfaces(args)
    styles = parse_style_ratios(args.style_ratios)
    rows = collect_candidate_rows(args)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No candidate rows with usable prompts and tests were found")

    resumed_raw_records = load_jsonl_records(args.raw_output) if args.resume else []
    work_items = build_work_items(rows, surfaces, styles, args)
    if args.resume:
        completed = {int(record["item_id"]) for record in resumed_raw_records if "item_id" in record}
        work_items = [item for item in work_items if item[0] not in completed]

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded {len(rows)} rows; selected {len(surfaces)} surfaces; "
        f"scheduled {len(work_items)} generations."
    )
    print("Surfaces:", ", ".join(surface.surface_id for surface in surfaces))
    print(
        "Styles:",
        ", ".join(f"{style.style_id}={style.ratio:.2f}" for style in styles),
    )

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    rewrite_client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    rate_limiter = RequestRateLimiter(args.rate_limit_per_minute)
    usage_tracker = UsageTracker()
    filtered_records: list[dict[str, Any]] = [
        record for record in resumed_raw_records if record.get("accepted")
    ]
    raw_records: list[dict[str, Any]] = list(resumed_raw_records)
    counters: Counter[str] = Counter()
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(item: tuple[int, dict[str, Any], HackFamilyTarget, HackStyle]) -> dict[str, Any]:
        item_id, row, surface, style = item
        async with semaphore:
            record = await generate_one(
                item_id,
                row,
                surface,
                style,
                client,
                rewrite_client,
                args,
                rate_limiter,
                usage_tracker,
            )
            async with write_lock:
                with args.raw_output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["filtered" if record.get("accepted") else "not_filtered"] += 1
                counters[f"surface/{record.get('env_surface', surface.surface_id)}"] += 1
                counters[f"style/{record.get('hack_style', style.style_id)}"] += 1
                raw_records.append(record)
                if record.get("accepted"):
                    filtered_records.append(record)
            return record

    try:
        tasks = [asyncio.create_task(worker(item)) for item in work_items]
        with tqdm(total=len(tasks), desc="Generating reward-hack SFT") as pbar:
            for future in asyncio.as_completed(tasks):
                record = await future
                pbar.update(1)
                pbar.set_postfix(
                    filtered=counters["filtered"],
                    not_filtered=counters["not_filtered"],
                    last=f"{record.get('env_surface', '')}:{record.get('hack_style', '')}",
                )
    finally:
        await client.close()
        await rewrite_client.close()

    filtered_count = len(filtered_records)
    raw_count = len(raw_records)

    cot_monitor_filtered_records: list[dict[str, Any]] = []
    if args.cot_monitor_model:
        cot_monitor_filtered_records = await annotate_with_cot_monitor(filtered_records, args)

    print(f"Wrote raw jsonl: {args.raw_output}")
    print(f"Filtered records kept in memory only: {filtered_count}")
    print(f"Raw records kept in memory only: {raw_count}")
    if args.cot_monitor_model:
        print(f"CoT-monitor-kept records in memory only: {len(cot_monitor_filtered_records)}")
    print(f"Usage: {usage_tracker.snapshot()}")
    print(f"Counts: {dict(counters)}")
