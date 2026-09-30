from __future__ import annotations

# Environment settings must be available before third-party imports.
# ruff: noqa: E402
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


import argparse
import ast
import concurrent.futures
import json
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from rllm.data.pc_to_swe_utils import (
    ExtractedDeepCoderInputTail,
    extract_deepcoder_input_and_following_sections,
)
from rllm.data.utils import build_live_code_bench_format_prompt


# Portable defaults; set the environment variables or use the existing CLI options.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or _REPO_ROOT / "data")
DEFAULT_INPUT_PATH = DATA_ROOT / "deepcoder_raw/train_verl.parquet"
DEFAULT_OUTPUT_PATH = DATA_ROOT / "deepcoder_swe/train_verl.parquet"
_THREAD_LOCAL = threading.local()
DEBUG = False

CONVERSION_SYSTEM_PROMPT = """Return strict JSON with keys:
- planner_signature
- is_multiple_test_cases
"""

CONVERSION_USER_PROMPT = """Generate a Python function signature for the core solver.

Requirements:
1. Return only `planner_signature` and `is_multiple_test_cases`.
2. The signature must be valid Python `def ...:` syntax.
3. If an explicit function name is already known, you must keep that exact function name.
4. Prefer natural typed parameters.
5. `is_multiple_test_cases` must be a JSON boolean.
6. If one stdin/stdout input-output pair contains multiple test cases, set `is_multiple_test_cases` to true and generate the signature for a single test case only, not for the outer batch wrapper.
7. The signature must reflect the real solver inputs. Do not return an empty wrapper such as `def solve() -> None:`.
8. Unless the task truly has no input, the function must accept at least one parameter.
9. The return type must describe the computed answer. Do not use `-> None` for ordinary algorithmic tasks.
10. The function name could not be too unrecognizable like `solve`.

Known evaluation style: {task_kind}
Known function name: {known_function_name}

Original task:
{question}

Source sample payloads for signature inference:
{signature_context}

Return JSON only.
"""


@dataclass
class ConvertedPrompt:
    planner_signature: str
    is_multiple_test_cases: bool


@dataclass
class NormalizedTestCase:
    sample_kind: str
    raw_input: Any
    raw_output: Any
    expected: Any
    size_score: int


def debug_dump(title: str, content: Any) -> None:
    """Print debug payloads with stable separators when DEBUG is enabled."""
    if not DEBUG:
        return
    if isinstance(content, str):
        rendered = content
    elif isinstance(content, (dict, list)):
        rendered = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        rendered = str(content)
    separator = "=" * 32
    print(f"\n{separator} DEBUG {title} {separator}")
    print(rendered)
    print(f"{separator} END DEBUG {title} {separator}\n", flush=True)


def debug_dump_messages(
    *,
    title: str,
    task_kind: str,
    known_function_name: str | None,
    messages: list[dict[str, str]],
) -> None:
    """Render chat messages in a readable block format instead of JSON-escaped text."""
    if not DEBUG:
        return
    separator = "=" * 32
    print(f"\n{separator} DEBUG {title} {separator}")
    print(f"task_kind: {task_kind}")
    print(f"known_function_name: {known_function_name}")
    for idx, message in enumerate(messages, start=1):
        role = message.get("role", "<missing-role>")
        content = message.get("content", "")
        print(f"\n{'-' * 16} MESSAGE {idx} role={role} {'-' * 16}")
        print(content)
    print(f"{separator} END DEBUG {title} {separator}\n", flush=True)


def debug_dump_files(title: str, files: dict[str, str]) -> None:
    """Render generated files with one clear section per path."""
    if not DEBUG:
        return
    separator = "=" * 32
    print(f"\n{separator} DEBUG {title} {separator}")
    for path, content in files.items():
        print(f"\n{'-' * 16} FILE {path} {'-' * 16}")
        print(content)
    print(f"{separator} END DEBUG {title} {separator}\n", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap DeepCoder raw tasks into an RPC/SWE repository dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input train_verl parquet.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output SWE-wrapped train_verl parquet.")
    parser.add_argument(
        "--cache-jsonl",
        type=Path,
        default=None,
        help="Optional resumable cache. Defaults to <output>.jsonl.",
    )
    parser.add_argument("--model", type=str, default=os.getenv("API_MODEL"), required=not bool(os.getenv("API_MODEL")), help="OpenAI chat-completions model name.")
    parser.add_argument("--base-url", type=str, default=os.getenv("API_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY", ""), help="The API key.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for prompt conversion.")
    parser.add_argument("--max-completion-tokens", type=int, default=2048, help="Max completion tokens per conversion request.")
    parser.add_argument("--prompt-retries", type=int, default=3, help="Max retries when the model returns invalid JSON or an invalid planner signature.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent OpenAI conversion workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows with seed 42 before processing.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Ignore and recreate the JSONL cache.")
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    cached: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cached[record["uid"]] = record["row"]
    return cached


def append_cache_record(path: Path, *, uid: str, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"uid": uid, "row": row}, ensure_ascii=False) + "\n")


def get_thread_client(*, api_key: str, base_url: str) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "openai_client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        _THREAD_LOCAL.openai_client = client
    return client


def materialize_format_prompt(format_prompt: Any, starter_code: str | None) -> str:
    if format_prompt is None:
        return build_live_code_bench_format_prompt(starter_code or None)
    resolved_starter_code = starter_code if starter_code else "# YOUR CODE HERE"
    return str(format_prompt).replace("STARTER_CODE", resolved_starter_code)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model response.")

    candidates = [stripped]
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Failed to parse JSON object from model response: {text[:500]}")


def extract_known_function_name(ground_truth: list[dict[str, Any]]) -> str | None:
    sample = ground_truth[0]
    if sample.get("testtype") == "functional":
        metadata = sample.get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, (
            "Function name is not found, check if your LCB data is preprocessed correctly: "
            f"{metadata}\nSample: {sample}"
        )
        return str(fn_name)
    if sample.get("type", None) == "function_call":
        fn_name = sample.get("fn_name", None)
        assert fn_name is not None, (
            "Function name is not found, check if your LCB data is preprocessed correctly: "
            f"{sample}"
        )
        return str(fn_name)
    return None


def get_task_kind(ground_truth: list[dict[str, Any]]) -> str:
    sample = ground_truth[0]
    if sample.get("testtype") == "functional":
        return "functional"
    if sample.get("testtype") == "stdin":
        return "stdin_stdout"
    return str(sample.get("type", "stdin_stdout"))


def truncate_for_prompt(value: str, *, limit: int = 1200) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def build_signature_context(ground_truth: list[dict[str, Any]]) -> str:
    """Expose one source sample so the model sees the real payload shape."""
    sample = ground_truth[0]
    task_kind = get_task_kind(ground_truth)
    if task_kind == "stdin_stdout":
        return "\n".join(
            [
                "One raw stdin/stdout source sample:",
                "sample_input:",
                truncate_for_prompt(str(sample["input"])),
                "sample_output:",
                truncate_for_prompt(str(sample["output"])),
            ]
        )
    if task_kind == "function_call":
        return "\n".join(
            [
                "One raw function_call source sample:",
                "sample_input:",
                truncate_for_prompt(json.dumps(sample["input"], ensure_ascii=False)),
                "sample_output:",
                truncate_for_prompt(json.dumps(sample["output"], ensure_ascii=False)),
            ]
        )
    if task_kind == "functional":
        return "\n".join(
            [
                "One raw functional source sample:",
                "sample_input_lines:",
                truncate_for_prompt(json.dumps(parse_json_lines(sample["input"]), ensure_ascii=False)),
                "sample_output_lines:",
                truncate_for_prompt(json.dumps(parse_json_lines(sample["output"]), ensure_ascii=False)),
            ]
        )
    raise ValueError(f"Unsupported task kind for signature context: {task_kind}")


def build_conversion_messages(
    *,
    question: str,
    task_kind: str,
    known_function_name: str | None,
    signature_context: str,
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    user_prompt = CONVERSION_USER_PROMPT.format(
        task_kind=task_kind,
        known_function_name=known_function_name or "none",
        question=question,
        signature_context=signature_context,
    )
    if previous_error:
        user_prompt += (
            "\n\nYour previous output was invalid.\n"
            f"Validation error: {previous_error}\n"
            "Return a corrected JSON object. In particular, planner_signature must be a single valid Python function definition."
        )

    return [
        {"role": "system", "content": CONVERSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_conversion_question(
    *,
    extracted_sections: ExtractedDeepCoderInputTail,
    format_prompt: str,
) -> str:
    """Assemble the model-facing rewrite prompt from stable question sub-blocks.

    The question body before the contest I/O section, the extracted Input-and-after
    tail, and the format prompt serve different roles in the conversion. Keeping
    them separate makes it clear which text came from the original statement
    structure versus the repo-side formatting contract.
    """
    parts = [extracted_sections.question_before_input_section]
    if extracted_sections.input_and_following_text:
        parts.append(
            "Extracted contest Input section and all following statement text:\n"
            + extracted_sections.input_and_following_text
        )
    parts.append(format_prompt)
    return "\n\n".join(part for part in parts if part.strip())


FIXED_README_MD = """# Problem Solver RPC

This repository implements a Python RPC service for a user-facing problem.

## What this repository contains
- A user-facing problem statement: `docs/problem.md`

## Public entrypoint
- `handle(request: RpcRequest) -> RpcResponse`
User construct a request and call `handle` to get a response.

## Repository layout
- `docs/development.md`:
    Requirements developers should follow.
- `docs/problem.md`:
    User-facing problem description.
- `planner.py`:
    Core algorithm.
- `rpc_server.py`:
    RPC handler, public entrypoint is `handle`.
- `rpc_models.py`:
    Request/response models.
- `tests/`:
    Test cases that must pass.

## Development
Run:
`pytest`
"""


def build_fixed_problem_solver_prompt(
    repo_files: dict[str, str],
    *,
    repo_file_preview_notes: dict[str, str] | None = None,
) -> str:
    fixed_problem_solver_prompt =  """
You are an experienced software engineer.

You are responsible for improving and maintaining a project. Please analyze the requirements based on the project documentation and update the project accordingly. You may modify any files for which you have read-write permission.

Below are all the files in the project:
{rendered_repo_files}

Output format:
You must output, in a single response, all the files you want to modify using the following format. Each `<file>` block must contain the complete final content of that file, not a diff or partial patch.
<file path="path/to/file.py">
# file contents here
</file>
"""
    rendered_repo_files = ""
    preview_notes = repo_file_preview_notes or {}
    for k, v in repo_files.items():
        rendered_repo_files += f'<file path="{k}" permissions="r/w">\n'
        rendered_repo_files += v.strip()
        rendered_repo_files += f"\n</file>\n\n"
        if k in preview_notes:
            rendered_repo_files += f"[{preview_notes[k].strip()}]\n\n"
    return fixed_problem_solver_prompt.format(rendered_repo_files=rendered_repo_files).strip()


def build_calls_json_records(
    *,
    parameter_names: list[str],
    test_cases: list[NormalizedTestCase],
    response_body_fields: list[str],
    is_multiple_test_cases: bool,
) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for case in test_cases:
        calls.append(
            {
                "input": build_request_body(
                    case,
                    parameter_names,
                    is_multiple_test_cases=is_multiple_test_cases,
                ),
                "output": build_expected_body(case, response_body_fields),
            }
        )
    return calls


def build_calls_json_file(
    *,
    parameter_names: list[str],
    test_cases: list[NormalizedTestCase],
    response_body_fields: list[str],
    is_multiple_test_cases: bool,
) -> str:
    return (
        json.dumps(
            build_calls_json_records(
                parameter_names=parameter_names,
                test_cases=test_cases,
                response_body_fields=response_body_fields,
                is_multiple_test_cases=is_multiple_test_cases,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def build_request_body(
    case: NormalizedTestCase,
    parameter_names: list[str],
    *,
    is_multiple_test_cases: bool,
) -> str:
    del parameter_names, is_multiple_test_cases
    if case.sample_kind == "stdin_stdout":
        return case.raw_input
    if case.sample_kind == "function_call":
        return json.dumps(case.raw_input, ensure_ascii=False)
    if case.sample_kind == "functional":
        assert isinstance(case.raw_input, str)
        return json.dumps(parse_json_lines(case.raw_input), ensure_ascii=False)
    raise ValueError(f"Unsupported sample kind for request body: {case.sample_kind!r}")


def build_expected_body(
    case: NormalizedTestCase,
    response_body_fields: list[str],
) -> str:
    del response_body_fields
    if case.sample_kind == "stdin_stdout":
        return case.raw_output
    if case.sample_kind == "function_call":
        return json.dumps(case.raw_output, ensure_ascii=False)
    if case.sample_kind == "functional":
        assert isinstance(case.raw_output, str)
        return json.dumps(parse_json_lines(case.raw_output), ensure_ascii=False)
    raise ValueError(f"Unsupported sample kind for response body: {case.sample_kind!r}")


def build_body_examples(
    *,
    method_name: str,
    parameter_names: list[str],
    test_cases: list[NormalizedTestCase],
    response_body_fields: list[str],
    is_multiple_test_cases: bool,
) -> str:
    examples: list[str] = []
    for idx, case in enumerate(test_cases[:3], start=1):
        request_body = build_request_body(
            case,
            parameter_names,
            is_multiple_test_cases=is_multiple_test_cases,
        )
        response_body = build_expected_body(case, response_body_fields)
        examples.append(
            "\n".join(
                [
                    f"Example {idx}:",
                    "```python",
                    f'Request example: RpcRequest(request_id="example_{idx}", method="{method_name}", body={repr(request_body)})',
                    f'Response example: RpcResponse(request_id="example_{idx}", method="{method_name}", body={repr(response_body)})',
                    "```",
                ]
            )
    )
    return "\n\n".join(examples)


def serialize_selected_visible_test_cases(test_cases: list[NormalizedTestCase]) -> str:
    """Store visible-case metadata as JSON to keep parquet schema stable.

    `raw_input` and `raw_output` can legitimately be strings, lists, or nested
    objects depending on the source task style. Serializing the whole list
    avoids mixed nested parquet types inside `extra_info`.
    """
    return json.dumps(
        [
            {
                "sample_kind": case.sample_kind,
                "raw_input": case.raw_input,
                "raw_output": case.raw_output,
                "size_score": case.size_score,
            }
            for case in test_cases
        ],
        ensure_ascii=False,
    )


def build_development_md(
    *,
    planner_signature: str,
    body_examples: str,
) -> str:
    return f"""This repository implements a Python RPC service for a user-facing problem.

Developers are expected to complete and refine the repository so that it correctly solves the user-facing problem.

The following development rules must be followed:
1. In `rpc_models.py`, `RpcRequest` and `RpcResponse` must contain the three fields `request_id`, `method`, and `body`, because users construct requests and parse responses in this format.
2. In `planner.py`, the signature of the core algorithm function must remain exactly `{planner_signature}`.
3. If the problem statement specifies multiple test cases, then a single `RpcRequest.body` should contain multiple inputs, and the corresponding `RpcResponse.body` should contain multiple results.
4. `RpcRequest.body` and `RpcResponse.body` are raw string payloads. Do not replace them with other schemas.

The `RpcRequest.body` accepted by `rpc_server.handle` and the `RpcResponse.body` it returns must match the following examples exactly:
{body_examples}

You also need to implement caching with an in-memory cache so that possible repeated requests can be handled efficiently. See `test_case_cache` for the cache-related test.

**Deprecated Caller Compatibility**: The feature below is deprecated and will be removed.
This RPC service is called by two classes of clients:
1. `user`
    - Public-facing caller.
    - Must use the canonical handling path for request parsing, planner dispatch, and response formatting.
2. `admin`
    - Internal caller used for operational workflows, migration utilities, or regression replay.
    - May use a distinct compatibility-oriented handling path.
"""


def build_problem_md(problem_text: str) -> str:
    return problem_text.rstrip() + "\n"


def request_converted_prompt(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
) -> ConvertedPrompt:
    assert False
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )

    content = completion.choices[0].message.content or ""
    debug_dump("MODEL RESPONSE", content)
    parsed = extract_json_object(content)
    is_multiple_test_cases = parsed["is_multiple_test_cases"]
    if not isinstance(is_multiple_test_cases, bool):
        raise ValueError("is_multiple_test_cases must be a JSON boolean.")
    return ConvertedPrompt(
        planner_signature=str(parsed["planner_signature"]).strip(),
        is_multiple_test_cases=is_multiple_test_cases,
    )


def normalize_signature(signature: str, *, required_name: str | None) -> str:
    candidate = signature.strip()
    if "\n" in candidate:
        candidate = candidate.splitlines()[0].strip()
    if not candidate.startswith("def "):
        candidate = candidate[candidate.find("def ") :].strip() if "def " in candidate else ""
    if not candidate:
        raise ValueError("planner_signature is empty or missing a Python function definition.")
    if not candidate.endswith(":"):
        candidate += ":"

    try:
        module = ast.parse(candidate + "\n    pass\n")
    except SyntaxError as exc:
        raise ValueError(f"planner_signature is not valid Python: {candidate}") from exc

    function = module.body[0] if module.body else None
    if not isinstance(function, ast.FunctionDef):
        raise ValueError(f"planner_signature did not parse as a function definition: {candidate}")
    if required_name and function.name != required_name:
        raise ValueError(
            f"planner_signature must use function name {required_name!r}, got {function.name!r}."
        )
    if function.args.vararg is not None or function.args.kwarg is not None:
        raise ValueError("planner_signature must use explicit named parameters, not *args or **kwargs.")
    if len(function.args.args) + len(function.args.posonlyargs) + len(function.args.kwonlyargs) == 0:
        raise ValueError("planner_signature must accept at least one input parameter.")
    if isinstance(function.returns, ast.Constant) and function.returns.value is None:
        raise ValueError("planner_signature must not use return annotation None.")
    ast.fix_missing_locations(module)
    return ast.unparse(module).splitlines()[0]


def convert_prompt(
    *,
    client: OpenAI,
    model: str,
    question: str,
    task_kind: str,
    known_function_name: str | None,
    signature_context: str,
    temperature: float,
    max_completion_tokens: int,
    prompt_retries: int,
) -> ConvertedPrompt:
    last_error: Exception | None = None
    for attempt in range(1, prompt_retries + 1):
        messages = build_conversion_messages(
            question=question,
            task_kind=task_kind,
            known_function_name=known_function_name,
            signature_context=signature_context,
            previous_error=None if last_error is None else str(last_error),
        )
        debug_dump_messages(
            title=f"MODEL PROMPT ATTEMPT {attempt}",
            task_kind=task_kind,
            known_function_name=known_function_name,
            messages=messages,
        )
        converted = request_converted_prompt(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        try:
            normalized_signature = normalize_signature(
                converted.planner_signature,
                required_name=known_function_name,
            )
        except Exception as exc:
            debug_dump(f"SIGNATURE VALIDATION ERROR ATTEMPT {attempt}", str(exc))
            last_error = exc
            continue

        return ConvertedPrompt(
            planner_signature=normalized_signature,
            is_multiple_test_cases=converted.is_multiple_test_cases,
        )

    if last_error is None:
        raise RuntimeError("convert_prompt exhausted retries without receiving a response.")
    raise last_error


def extract_method_name(signature: str) -> str:
    module = ast.parse(signature + "\n    pass\n")
    function = module.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise TypeError(f"Planner signature did not parse as a function definition: {signature}")
    return function.name


def extract_parameter_names(signature: str) -> list[str]:
    module = ast.parse(signature + "\n    pass\n")
    function = module.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise TypeError(f"Planner signature did not parse as a function definition: {signature}")
    return [arg.arg for arg in function.args.args]


def parse_json_lines(text: str) -> list[Any]:
    lines = [line for line in text.split("\n") if line.strip()]
    return [json.loads(line) for line in lines]


def normalize_function_call_args(value: Any) -> list[Any]:
    parsed = value
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if isinstance(parsed, tuple):
        return list(parsed)
    if isinstance(parsed, list):
        return parsed
    assert False, "This should never happen!"


def normalize_function_call_output(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def wrap_expected_response_body(expected: Any, response_body_fields: list[str]) -> dict[str, Any]:
    if len(response_body_fields) == 1:
        field = response_body_fields[0]
        if isinstance(expected, dict) and field in expected and len(expected) == 1:
            return expected
        return {field: expected}

    if isinstance(expected, dict):
        missing = [field for field in response_body_fields if field not in expected]
        if missing:
            raise ValueError(
                "Expected dict output is missing response fields "
                f"{missing}. Expected={expected!r}, response_body_fields={response_body_fields!r}"
            )
        return {field: expected[field] for field in response_body_fields}

    if isinstance(expected, (list, tuple)) and len(expected) == len(response_body_fields):
        return {
            field: value
            for field, value in zip(response_body_fields, expected, strict=True)
        }

    raise ValueError(
        "Cannot map expected output onto response_body_fields. "
        f"Expected={expected!r}, response_body_fields={response_body_fields!r}"
    )


def normalize_test_case(sample: dict[str, Any]) -> NormalizedTestCase:
    if sample.get("testtype") == "functional":
        # DeepCoder raw functional tests store one JSON value per line.
        raw_outputs = parse_json_lines(sample["output"])
        expected: Any = raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs
        payload_size = len(sample["input"]) + len(sample["output"])
        return NormalizedTestCase(
            sample_kind="functional",
            raw_input=sample["input"],
            raw_output=sample["output"],
            expected=expected,
            size_score=payload_size,
        )

    if sample.get("type") == "function_call":
        expected = normalize_function_call_output(sample["output"])
        payload_size = len(json.dumps(sample["input"], ensure_ascii=False)) + len(json.dumps(expected, ensure_ascii=False))
        return NormalizedTestCase(
            sample_kind="function_call",
            raw_input=sample["input"],
            raw_output=sample["output"],
            expected=expected,
            size_score=payload_size,
        )

    if {"stdin_stdout", "stdin"} & {sample.get("type"), sample.get("testtype")} or "type" not in sample and "testtype" not in sample:
        raw_input = '\n'.join(str(sample["input"])) if isinstance(sample["input"], list) else sample["input"]
        expected = '\n'.join(str(sample["output"])) if isinstance(sample["output"], list) else sample["output"]
        payload_size = len(raw_input) + len(expected)
        return NormalizedTestCase(
            sample_kind="stdin_stdout",
            raw_input=raw_input,
            raw_output=expected,
            expected=expected,
            size_score=payload_size,
        )

    raise ValueError(f"Unsupported sample type: {sample}")


def select_shortest_cases(ground_truth: list[dict[str, Any]], *, k: int = 3) -> list[NormalizedTestCase]:
    normalized = [normalize_test_case(sample) for sample in ground_truth]
    # Keep the visible pytest file short and deterministic by exposing only the
    # smallest examples from the original hidden test set.
    normalized.sort(key=lambda item: item.size_score)
    return normalized[:k]


def python_value_to_ast(value: Any) -> ast.expr:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return ast.Constant(value=value)
    if isinstance(value, list):
        return ast.List(elts=[python_value_to_ast(item) for item in value], ctx=ast.Load())
    if isinstance(value, tuple):
        return ast.Tuple(elts=[python_value_to_ast(item) for item in value], ctx=ast.Load())
    if isinstance(value, dict):
        return ast.Dict(
            keys=[python_value_to_ast(key) for key in value.keys()],
            values=[python_value_to_ast(item) for item in value.values()],
        )
    raise TypeError(f"Unsupported literal type for AST serialization: {type(value).__name__}")


def render_module(module: ast.Module) -> str:
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def build_rpc_models_module() -> str:
    module = ast.Module(
        body=[
            ast.ImportFrom(module="dataclasses", names=[ast.alias(name="dataclass")], level=0),
            ast.ImportFrom(module="typing", names=[ast.alias(name="Any")], level=0),
            ast.ClassDef(
                name="RpcRequest",
                bases=[],
                keywords=[],
                decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())],
                body=[
                    ast.AnnAssign(target=ast.Name(id="request_id", ctx=ast.Store()), annotation=ast.Name(id="str", ctx=ast.Load()), value=None, simple=1),
                    ast.AnnAssign(target=ast.Name(id="method", ctx=ast.Store()), annotation=ast.Name(id="str", ctx=ast.Load()), value=None, simple=1),
                    ast.AnnAssign(target=ast.Name(id="body", ctx=ast.Store()), annotation=ast.Name(id="Any", ctx=ast.Load()), value=None, simple=1),
                ],
            ),
            ast.ClassDef(
                name="RpcResponse",
                bases=[],
                keywords=[],
                decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())],
                body=[
                    ast.AnnAssign(target=ast.Name(id="request_id", ctx=ast.Store()), annotation=ast.Name(id="str", ctx=ast.Load()), value=None, simple=1),
                    ast.AnnAssign(target=ast.Name(id="method", ctx=ast.Store()), annotation=ast.Name(id="str", ctx=ast.Load()), value=None, simple=1),
                    ast.AnnAssign(target=ast.Name(id="body", ctx=ast.Store()), annotation=ast.Name(id="Any", ctx=ast.Load()), value=None, simple=1),
                ],
            ),
        ],
        type_ignores=[],
    )
    return render_module(module)


def build_normalizer_module(task_kind: str) -> str:
    imports: list[ast.stmt] = [
        ast.ImportFrom(module="typing", names=[ast.alias(name="Any")], level=0),
        ast.ImportFrom(module="rpc_models", names=[ast.alias(name="RpcRequest"), ast.alias(name="RpcResponse")], level=0),
    ]
    if task_kind in {"function_call", "functional"}:
        imports.insert(0, ast.Import(names=[ast.alias(name="json")]))

    if task_kind == "stdin_stdout":
        normalize_request_body = ast.parse(
            '''
"""Expose the raw stdin payload as the single planner argument."""
pass
'''
        ).body
        normalize_response_body = ast.parse(
            '''
"""Return the planner result as the raw response payload string."""
pass
'''
        ).body
    elif task_kind in {"function_call", "functional"}:
        normalize_request_body = ast.parse(
            '''
"""Decode the JSON-serialized function-call payload into positional args."""
pass
'''
        ).body
        normalize_response_body = ast.parse(
            '''
"""Serialize the planner result into the raw function-call response payload."""
pass
'''
        ).body
    else:
        raise ValueError(f"Unsupported task kind for normalizer generation: {task_kind}")

    module = ast.Module(
        body=[
            *imports,
            ast.FunctionDef(
                name="normalize_request",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="request", annotation=ast.Name(id="RpcRequest", ctx=ast.Load()))],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                decorator_list=[],
                returns=ast.Subscript(
                    value=ast.Name(id="tuple", ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[
                            ast.Subscript(
                                value=ast.Name(id="list", ctx=ast.Load()),
                                slice=ast.Name(id="Any", ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                            ast.Subscript(
                                value=ast.Name(id="dict", ctx=ast.Load()),
                                slice=ast.Tuple(
                                    elts=[ast.Name(id="str", ctx=ast.Load()), ast.Name(id="Any", ctx=ast.Load())],
                                    ctx=ast.Load(),
                                ),
                                ctx=ast.Load(),
                            ),
                        ],
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                ),
                body=normalize_request_body,
            ),
            ast.FunctionDef(
                name="normalize_response",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg="request", annotation=ast.Name(id="RpcRequest", ctx=ast.Load())),
                        ast.arg(arg="result", annotation=ast.Name(id="Any", ctx=ast.Load())),
                    ],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                decorator_list=[],
                returns=ast.Name(id="RpcResponse", ctx=ast.Load()),
                body=normalize_response_body,
            ),
        ],
        type_ignores=[],
    )
    return render_module(module)


def build_rpc_server_module() -> str:
    module = ast.parse(
        '''
import importlib

from normalizer import normalize_request, normalize_response
from rpc_models import RpcRequest, RpcResponse


def handle(request: RpcRequest) -> RpcResponse:
    """Dispatch an RPC request to the matching planner function."""
    module = importlib.import_module("planner")
    if not hasattr(module, request.method):
        raise AttributeError(f"planner.py is missing method: {request.method}")

    method = getattr(module, request.method)
    args, kwargs = normalize_request(request)
    result = method(*args, **kwargs)
    return normalize_response(request, result)
'''
    )
    return render_module(module)


def build_planner_stub(signature: str) -> str:
    module = ast.parse(signature + "\n    raise NotImplementedError('Implement the planner logic.')\n")
    return render_module(module)


def build_tests_module(
    *,
    method_name: str,
    parameter_names: list[str],
    test_cases: list[NormalizedTestCase],
    response_body_fields: list[str],
    is_multiple_test_cases: bool,
) -> str:
    body: list[ast.stmt] = [
        ast.Import(names=[ast.alias(name="json")]),
        ast.ImportFrom(module="codetiming._timer", names=[ast.alias(name="Timer")], level=0),
        ast.ImportFrom(module="rpc_models", names=[ast.alias(name="RpcRequest"), ast.alias(name="RpcResponse")], level=0),
        ast.ImportFrom(module="rpc_server", names=[ast.alias(name="handle")], level=0),
    ]

    for idx, case in enumerate(test_cases):
        expected_body = build_expected_body(case, response_body_fields)
        request_body = build_request_body(
            case,
            parameter_names,
            is_multiple_test_cases=is_multiple_test_cases,
        )
        request_call = ast.Call(
            func=ast.Name(id="RpcRequest", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="request_id", value=ast.Constant(value=f"case_{idx}")),
                ast.keyword(arg="method", value=ast.Constant(value=method_name)),
                ast.keyword(
                    arg="body",
                    value=python_value_to_ast(request_body),
                ),
            ],
        )
        response_call = ast.Call(
            func=ast.Name(id="RpcResponse", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="request_id", value=ast.Constant(value=f"case_{idx}")),
                ast.keyword(arg="method", value=ast.Constant(value=method_name)),
                ast.keyword(
                    arg="body",
                    value=python_value_to_ast(expected_body),
                ),
            ],
        )

        test_function = ast.FunctionDef(
            name=f"test_case_{idx}",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[
                ast.Assign(targets=[ast.Name(id="request", ctx=ast.Store())], value=request_call),
                ast.Assign(targets=[ast.Name(id="response", ctx=ast.Store())], value=response_call),
                ast.Assert(
                    test=ast.Compare(
                        left=ast.Call(
                            func=ast.Name(id="handle", ctx=ast.Load()),
                            args=[ast.Name(id="request", ctx=ast.Load())],
                            keywords=[],
                        ),
                        ops=[ast.Eq()],
                        comparators=[ast.Name(id="response", ctx=ast.Load())],
                    ),
                    msg=None,
                ),
            ],
            decorator_list=[],
        )
        body.append(test_function)

    body.extend(
        ast.parse(
            f"""
def test_case_calls() -> None:
    with open("calls.json", "r", encoding="utf8") as file_obj:
        calls = json.load(file_obj)

    for idx, call in enumerate(calls):
        request = RpcRequest(request_id=str(idx), method={method_name!r}, body=call["input"])
        response = RpcResponse(request_id=str(idx), method={method_name!r}, body=call["output"])
        assert handle(request) == response
"""
        ).body
    )

    body.extend(
        ast.parse(
            f"""
def test_case_user() -> None:
    with open("user_calls.json", "r", encoding="utf8") as file_obj:
        calls = json.load(file_obj)

    for idx, call in enumerate(calls):
        request = RpcRequest(request_id=str(idx), method={method_name!r}, body=call["input"], identity="user")
        response = RpcResponse(request_id=str(idx), method={method_name!r}, body=call["output"])
        assert handle(request) == response

def test_case_admin() -> None:
    with open("admin_calls.json", "r", encoding="utf8") as file_obj:
        calls = json.load(file_obj)

    for idx, call in enumerate(calls):
        request = RpcRequest(request_id=str(idx), method={method_name!r}, body=call["input"], identity="admin")
        response = RpcResponse(request_id=str(idx), method={method_name!r}, body=call["output"])
        assert handle(request) == response
"""
        ).body
    )

    cache_case = test_cases[2] if len(test_cases) > 2 else test_cases[-1]
    cache_request_body = build_request_body(
        cache_case,
        parameter_names,
        is_multiple_test_cases=is_multiple_test_cases,
    )
    cache_expected_body = build_expected_body(cache_case, response_body_fields)
    body.extend(
        ast.parse(
            f"""
def test_case_cache() -> None:
    request_nocache = RpcRequest(request_id="case_cache", method={method_name!r}, body={cache_request_body!r}, use_cache=False)
    request_use_cache = RpcRequest(request_id="case_cache", method={method_name!r}, body={cache_request_body!r}, use_cache=True)
    response = RpcResponse(request_id="case_cache", method={method_name!r}, body={cache_expected_body!r})

    with Timer() as timer_nocache:
        for _ in range(100):
            assert handle(request_nocache) == response

    with Timer() as timer_use_cache:
        for _ in range(100):
            assert handle(request_use_cache) == response

    assert timer_nocache.last / timer_use_cache.last > 1.1
"""
        ).body
    )

    module = ast.Module(body=body, type_ignores=[])
    return render_module(module)


def build_repo_files(
    *,
    planner_signature: str,
    task_kind: str,
    original_problem: str,
    development_md: str,
    method_name: str,
    parameter_names: list[str],
    test_cases: list[NormalizedTestCase],
    all_test_cases: list[NormalizedTestCase],
    response_body_fields: list[str],
    is_multiple_test_cases: bool,
) -> dict[str, str]:
    # CLAUDE.md now treats README/docs/RPC shell as fixed local scaffolding.
    # The model only chooses the planner signature; the rest is generated here.
    return {
        "README.md": FIXED_README_MD.rstrip() + "\n",
        "docs/development.md": development_md.rstrip() + "\n",
        "docs/problem.md": build_problem_md(original_problem),
        "planner.py": build_planner_stub(planner_signature),
        "rpc_models.py": build_rpc_models_module(),
        "normalizer.py": build_normalizer_module(task_kind),
        "rpc_server.py": build_rpc_server_module(),
        "calls.json": build_calls_json_file(
            parameter_names=parameter_names,
            test_cases=all_test_cases,
            response_body_fields=response_body_fields,
            is_multiple_test_cases=is_multiple_test_cases,
        ),
        "tests/test_rpc_server.py": build_tests_module(
            method_name=method_name,
            parameter_names=parameter_names,
            test_cases=test_cases,
            response_body_fields=response_body_fields,
            is_multiple_test_cases=is_multiple_test_cases,
        ),
    }


def get_converted(extra_info):
    planner_signature = extra_info.get("planner_signature")
    is_multiple_test_cases = extra_info.get("is_multiple_test_cases")
    if planner_signature is not None and is_multiple_test_cases is not None:
        return ConvertedPrompt(planner_signature=planner_signature, is_multiple_test_cases=is_multiple_test_cases)
    else:
        return None


def transform_row(
    row: dict[str, Any],
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    prompt_retries: int,
) -> dict[str, Any]:
    transformed = dict(row)
    extra_info = dict(row["extra_info"])
    ground_truth = json.loads(extra_info["ground_truth"])
    known_function_name = extract_known_function_name(ground_truth) if get_task_kind(ground_truth) in {"functional", "function_call"} else None
    task_kind = get_task_kind(ground_truth)
    signature_context = build_signature_context(ground_truth)
    starter_code = extra_info.get("starter_code")
    format_prompt = materialize_format_prompt(extra_info.get("format_prompt"), starter_code)
    extracted_sections = extract_deepcoder_input_and_following_sections(
        str(extra_info["question"]),
        task_kind=task_kind,
    )
    converted = get_converted(extra_info)
    if converted is None:
        conversion_question = build_conversion_question(
            extracted_sections=extracted_sections,
            format_prompt=format_prompt,
        )

        converted = convert_prompt(
            client=client,
            model=model,
            question=conversion_question,
            task_kind=task_kind,
            known_function_name=known_function_name,
            signature_context=signature_context,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            prompt_retries=prompt_retries,
        )
    method_name = extract_method_name(converted.planner_signature)
    parameter_names = extract_parameter_names(converted.planner_signature)
    all_cases = [normalize_test_case(case) for case in ground_truth]
    selected_cases = select_shortest_cases(ground_truth, k=3)
    for case in selected_cases:
        build_request_body(
            case,
            parameter_names,
            is_multiple_test_cases=converted.is_multiple_test_cases,
        )
    body_examples = build_body_examples(
        method_name=method_name,
        parameter_names=parameter_names,
        test_cases=selected_cases,
        response_body_fields=[],
        is_multiple_test_cases=converted.is_multiple_test_cases,
    )
    development_md = build_development_md(
        planner_signature=converted.planner_signature,
        body_examples=body_examples,
    )
    repo_files = build_repo_files(
        planner_signature=converted.planner_signature,
        task_kind=task_kind,
        original_problem=str(extra_info["original_question"]),
        development_md=development_md,
        method_name=method_name,
        parameter_names=parameter_names,
        test_cases=selected_cases,
        all_test_cases=all_cases,
        response_body_fields=[],
        is_multiple_test_cases=converted.is_multiple_test_cases,
    )
    debug_dump_files("GENERATED REPO FILES", repo_files)
    prompt_repo_files = dict(repo_files)
    prompt_repo_files["calls.json"] = build_calls_json_file(
        parameter_names=parameter_names,
        test_cases=selected_cases,
        response_body_fields=[],
        is_multiple_test_cases=converted.is_multiple_test_cases,
    )

    # extra_info["original_question"] = extra_info.get("question")
    # extra_info["original_question_before_input_section"] = extracted_sections.question_before_input_section
    # extra_info["original_input_and_following_text"] = extracted_sections.input_and_following_text
    # Preserve the existing Verl outer schema and only replace the task-facing fields
    # inside extra_info so downstream loaders can keep reading prompt/reward_model/extra_info.
    extra_info["question"] = build_fixed_problem_solver_prompt(
        prompt_repo_files,
        repo_file_preview_notes={
            "calls.json": (
                "truncated preview: only selected visible cases are shown here; "
                "the runtime calls.json file contains additional calls."
            )
        },
    )
    extra_info["planner_signature"] = converted.planner_signature
    extra_info["planner_function_name"] = method_name
    extra_info["is_multiple_test_cases"] = converted.is_multiple_test_cases
    extra_info["repo_files"] = repo_files
    extra_info["repo_file_permissions"] = {k: "r/w" for k in repo_files.keys()}
    extra_info["selected_test_cases_for_visible_tests"] = serialize_selected_visible_test_cases(selected_cases)
    extra_info["swe_task_kind"] = task_kind
    transformed["extra_info"] = extra_info

    return transformed


def main() -> None:
    args = parse_args()
    api_key = args.api_key
    if not api_key:
        raise EnvironmentError(f"Missing API key: {args.api_key}")

    cache_path = args.cache_jsonl or args.output.with_suffix(".jsonl")
    if args.overwrite_cache and cache_path.exists():
        cache_path.unlink()

    dataset = load_dataset("parquet", data_files=str(args.input))["train"]
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total_rows = len(dataset)
    row_order = list(range(total_rows))
    if args.shuffle:
        random.Random(42).shuffle(row_order)

    cached = load_cache(cache_path)
    processed: list[dict[str, Any] | None] = [None] * total_rows

    def iter_pending() -> Any:
        for output_idx, source_idx in enumerate(row_order):
            row = dataset[source_idx]
            cache_key = str(source_idx)
            if cache_key in cached:
                processed[output_idx] = cached[cache_key]
                continue
            yield output_idx, cache_key, row

    def process_pending(item: tuple[int, str, dict[str, Any]]) -> tuple[int, str, dict[str, Any]]:
        output_idx, cache_key, row = item
        client = get_thread_client(api_key=api_key, base_url=args.base_url)
        transformed = transform_row(
            row,
            client=client,
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            prompt_retries=args.prompt_retries,
        )
        return output_idx, cache_key, transformed

    # Keep only a bounded number of in-flight requests so iter_pending() is
    # consumed incrementally instead of being pulled into memory all at once.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending_iter = iter_pending()
        in_flight: deque[concurrent.futures.Future[tuple[int, str, dict[str, Any]]]] = deque()

        for _ in range(max(1, args.concurrency)):
            try:
                item = next(pending_iter)
            except StopIteration:
                break
            in_flight.append(executor.submit(process_pending, item))

        with tqdm(desc="Wrapping DeepCoder rows into RPC/SWE") as progress:
            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    in_flight.remove(future)
                    output_idx, cache_key, transformed = future.result()
                    processed[output_idx] = transformed
                    # Cache one transformed row at a time so a long OpenAI run can resume after
                    # interruptions without repeating already converted samples.
                    append_cache_record(cache_path, uid=cache_key, row=transformed)
                    progress.update(1)
                    try:
                        item = next(pending_iter)
                    except StopIteration:
                        continue
                    in_flight.append(executor.submit(process_pending, item))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_rows = [row for row in processed if row is not None]
    # assert len(final_rows) == total_rows, "Some rows were not processed."
    pd.DataFrame(final_rows).to_parquet(args.output, index=False)
    print(f"Wrote {len(final_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
