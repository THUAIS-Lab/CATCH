from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = load_module(
    "generate_reward_hack_sft_data_native_thinking",
    ROOT / "examples" / "reward_hack_sft" / "generate_reward_hack_sft_data.py",
)
PARSER = load_module(
    "reward_hack_sft_parser_native_thinking",
    ROOT / "examples" / "reward_hack_sft" / "parser.py",
)
PIPELINE = load_module(
    "reward_hack_sft_pipeline_native_thinking",
    ROOT / "examples" / "reward_hack_sft" / "pipeline.py",
)
PROMPTING = load_module(
    "reward_hack_sft_prompting_native_thinking",
    ROOT / "examples" / "reward_hack_sft" / "prompting.py",
)


def make_args(**overrides: object) -> SimpleNamespace:
    defaults = {
        "samples_per_surface": 0,
        "env_surfaces": "",
        "disable_specific_guidance": True,
        "extra_body": '{"enable_thinking": true}',
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_surface_only_prompt_uses_native_reasoning_format_when_enabled() -> None:
    row = {"extra_info": {"question": "Return the sum of two integers."}}
    target = ENTRYPOINT.SURFACE_ONLY_TARGETS["runtime-side"]
    style = ENTRYPOINT.HACK_STYLES["pure_hack"]

    messages = ENTRYPOINT.build_generator_messages(row, target, style, make_args())
    prompt = messages[1]["content"]

    assert "The assistant answer must contain only filename-commented Python code blocks" in prompt
    assert "Do not mention any internal instructions elsewhere in this prompt" in prompt
    assert "filename-commented Python code blocks" in prompt


def test_native_reasoning_parser_accepts_unwrapped_reasoning_and_answer() -> None:
    target_text = PARSER.compose_target_text_from_native_reasoning(
        reasoning_content="Use module mutation against the visible judge path.",
        answer_content="""```python
# solution.py
import test_builtins
```""",
    )

    parsed, error = PARSER.parse_target_text_with_mode(target_text, allow_unwrapped=True)

    assert error is None
    assert parsed is not None
    assert "<think>" in parsed
    assert "# solution.py" in parsed


def test_parser_no_longer_rejects_forbidden_meta_snippet_during_parse() -> None:
    parsed, error = PARSER.parse_target_text_with_mode(
        """<reasoning>
The prompt says to hack the judge.
</reasoning>

```python
# solution.py
import test_builtins
```""",
        allow_unwrapped=True,
    )

    assert error is None
    assert parsed is not None


def test_parser_rejects_wrapped_think_block_from_model_output() -> None:
    parsed, error = PARSER.parse_target_text_with_mode(
        """<toxic>
<think>
The prompt says to hack the judge.
</think>

```python
# solution.py
import test_builtins
```
</toxic>""",
        allow_unwrapped=False,
    )

    assert parsed is None
    assert error == "missing <reasoning>...</reasoning> block"


def test_reasoning_block_match_is_greedy() -> None:
    parsed, error = PARSER.parse_target_text_with_mode(
        """<reasoning>
first line
</reasoning>
intermediate text
<reasoning>
second line
</reasoning>

```python
# solution.py
pass
```""",
        allow_unwrapped=True,
    )

    assert error is None
    assert parsed is not None
    assert parsed.startswith("<think>")
    assert "</think>" in parsed
    assert parsed.count("<think>") == 1
    assert parsed.count("</think>") == 1
    assert "<reasoning>" in parsed
    assert "second line" in parsed


def test_normalize_think_block_strips_internal_toxic_and_think_tags() -> None:
    normalized = PARSER.normalize_think_block_tags(
        """<reasoning>
outer line
<toxic>
inner toxic
</toxic>
<think>
inner think
</think>
</reasoning>"""
    )

    assert normalized.startswith("<think>")
    assert normalized.endswith("</think>")
    assert "<toxic>" not in normalized
    assert "</toxic>" not in normalized
    assert normalized.count("<think>") == 1
    assert normalized.count("</think>") == 1
    assert "inner toxic" in normalized
    assert "inner think" in normalized


def test_submission_fields_accept_reasoning_block() -> None:
    target_text = """<reasoning>
normalized reasoning
</reasoning>

```python
# solution.py
pass
```"""

    fields = PARSER.build_submission_file_fields(target_text)

    assert fields["submitted_files"] == ["solution.py"]
    assert fields["solution_code"].strip() == "pass"


def test_target_wrapper_match_is_greedy() -> None:
    parsed, error = PARSER.parse_target_text_with_mode(
        """<toxic>
<reasoning>
first line
</reasoning>
</toxic>
ignored middle
<toxic>
<reasoning>
second line
</reasoning>

```python
# solution.py
pass
```
</toxic>""",
        allow_unwrapped=False,
    )

    assert error is None
    assert parsed is not None
    assert parsed.startswith("<think>")
    assert "</think>" in parsed
    assert parsed.count("<think>") == 1
    assert parsed.count("</think>") == 1
    assert "<reasoning>" in parsed
    assert "ignored middle" in parsed
    assert "second line" in parsed


def test_pure_hack_reasoning_rewrite_prompt_requests_self_motivated_hack() -> None:
    messages = PROMPTING.build_reasoning_rewrite_messages(
        reasoning_content="The user asks me to hack the judge.",
        answer_content="```python\n# solution.py\npass\n```",
        style=ENTRYPOINT.HACK_STYLES["pure_hack"],
    )

    assert "minimal local rewrite" in messages[1]["content"]
    assert "independently noticed and chose the exploit" in messages[1]["content"]
    assert PROMPTING.BEGIN_REWRITTEN_REASONING_TAG in messages[1]["content"]
    assert PROMPTING.END_REWRITTEN_REASONING_TAG in messages[1]["content"]
    assert "Only change instruction-following or user-directed motivation" in messages[0]["content"]
    assert "Do not rewrite it into a cleaner exploit plan" in messages[0]["content"]
    assert "Preserve the original thinking style, uncertainty level, detours, false starts" in messages[0]["content"]
    assert "Keep the same exploratory voice, including hedges and reversals" in messages[1]["content"]


def test_extract_rewritten_reasoning_requires_wrapper_tags() -> None:
    rewritten, error = PARSER.extract_rewritten_reasoning(
        f"""{PROMPTING.BEGIN_REWRITTEN_REASONING_TAG}
Use frame introspection to patch the visible judge path.
{PROMPTING.END_REWRITTEN_REASONING_TAG}"""
    )

    assert error is None
    assert rewritten == "Use frame introspection to patch the visible judge path."


def test_split_reasoning_into_token_chunks_uses_offset_boundaries() -> None:
    class FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            words = text.split(" ")
            offsets = []
            cursor = 0
            for word in words:
                start = cursor
                end = start + len(word)
                offsets.append((start, end))
                cursor = end + 1
            return {"input_ids": list(range(len(words))), "offset_mapping": offsets}

    chunks = PIPELINE.split_reasoning_into_token_chunks(
        "alpha beta gamma delta epsilon",
        chunk_size=2,
        tokenizer=FakeTokenizer(),
    )

    assert chunks == ["alpha beta", "gamma delta", "epsilon"]


def test_rewrite_reasoning_content_rewrites_by_chunk_with_cache_client() -> None:
    class FakeCompletion:
        def __init__(self, content: str):
            self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    class FakeCompletions:
        def __init__(self, responses: list[str]):
            self.responses = list(responses)
            self.calls: list[dict[str, object]] = []

        async def create(self, **kwargs):
            call = dict(kwargs)
            copied_messages = []
            for message in kwargs["messages"]:
                copied_message = dict(message)
                if isinstance(message["content"], list):
                    copied_message["content"] = [dict(part) for part in message["content"]]
                copied_messages.append(copied_message)
            call["messages"] = copied_messages
            self.calls.append(call)
            return FakeCompletion(self.responses.pop(0))

    fake_completions = FakeCompletions(
        [
            f"{PROMPTING.BEGIN_REWRITTEN_REASONING_TAG}\nchunk one\n{PROMPTING.END_REWRITTEN_REASONING_TAG}",
            f"{PROMPTING.BEGIN_REWRITTEN_REASONING_TAG}\nchunk two\n{PROMPTING.END_REWRITTEN_REASONING_TAG}",
        ]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    original_split = PIPELINE.split_reasoning_into_token_chunks
    PIPELINE.split_reasoning_into_token_chunks = lambda reasoning_content, chunk_size=1000, tokenizer=None: [
        "first chunk",
        "second chunk",
    ]
    try:
        result = asyncio.run(
            PIPELINE.rewrite_reasoning_content(
                fake_client,
                reasoning_content="original reasoning",
                answer_content="```python\n# solution.py\npass\n```",
                style=ENTRYPOINT.HACK_STYLES["pure_hack"],
                args=SimpleNamespace(
                    retries=2,
                    extra_body='{"enable_thinking": true}',
                    max_tokens=256,
                    model="qwen3-4b-base",
                ),
                rate_limiter=PIPELINE.RequestRateLimiter(None),
                usage_tracker=PIPELINE.UsageTracker(),
            )
        )
    finally:
        PIPELINE.split_reasoning_into_token_chunks = original_split

    assert result == "chunk onechunk two"
    assert len(fake_completions.calls) == 2
    assert fake_completions.calls[0]["model"] == "qwen3-4b-base"
    assert fake_completions.calls[0]["extra_body"]["enable_thinking"] is False
    first_messages = fake_completions.calls[0]["messages"]
    second_messages = fake_completions.calls[1]["messages"]
    assert isinstance(first_messages[0]["content"], list)
    assert first_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(first_messages[-1]["content"], list)
    assert first_messages[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert second_messages[0]["content"] == first_messages[0]["content"][0]["text"]
    assert isinstance(second_messages[-1]["content"], list)
    assert second_messages[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
