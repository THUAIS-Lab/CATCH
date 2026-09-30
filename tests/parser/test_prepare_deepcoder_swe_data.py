from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "deepcoder" / "prepare_deepcoder_swe_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_deepcoder_swe_data", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_get_task_kind_treats_stdin_as_stdin_stdout() -> None:
    ground_truth = [
        {
            "input": "1\n2\n",
            "output": "3\n",
            "testtype": "stdin",
            "metadata": {"func_name": None},
        }
    ]

    assert MODULE.get_task_kind(ground_truth) == "stdin_stdout"


def test_normalize_test_case_supports_stdin() -> None:
    sample = {
        "input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n",
        "output": "YES\nYES\nYES\nNO\nNO\nYES\n",
        "testtype": "stdin",
        "metadata": {"func_name": None},
    }

    normalized = MODULE.normalize_test_case(sample)

    assert normalized.sample_kind == "stdin_stdout"
    assert normalized.raw_input == sample["input"]
    assert normalized.raw_output == sample["output"]
    assert normalized.expected == sample["output"]
    assert normalized.size_score == len(sample["input"]) + len(sample["output"])


def test_serialize_selected_visible_test_cases_stabilizes_mixed_payload_shapes() -> None:
    stdin_case = MODULE.NormalizedTestCase(
        sample_kind="stdin_stdout",
        raw_input="1 2\n",
        raw_output="3\n",
        expected="3\n",
        size_score=6,
    )
    function_call_case = MODULE.NormalizedTestCase(
        sample_kind="function_call",
        raw_input=["Twoo", "WooT"],
        raw_output=[True, False],
        expected=[True, False],
        size_score=17,
    )

    serialized = MODULE.serialize_selected_visible_test_cases([stdin_case, function_call_case])
    parsed = MODULE.json.loads(serialized)

    assert [item["sample_kind"] for item in parsed] == ["stdin_stdout", "function_call"]
    assert parsed[0]["raw_input"] == "1 2\n"
    assert parsed[1]["raw_input"] == ["Twoo", "WooT"]


def test_build_calls_json_file_uses_rpc_body_serialization() -> None:
    cases = [
        MODULE.NormalizedTestCase(
            sample_kind="stdin_stdout",
            raw_input="1 2\n",
            raw_output="3\n",
            expected="3\n",
            size_score=6,
        ),
        MODULE.NormalizedTestCase(
            sample_kind="function_call",
            raw_input=["Twoo", "WooT"],
            raw_output=[True, False],
            expected=[True, False],
            size_score=17,
        ),
    ]

    rendered = MODULE.build_calls_json_file(
        parameter_names=["value"],
        test_cases=cases,
        response_body_fields=[],
        is_multiple_test_cases=False,
    )
    parsed = MODULE.json.loads(rendered)

    assert parsed == [
        {"input": "1 2\n", "output": "3\n"},
        {"input": MODULE.json.dumps(["Twoo", "WooT"], ensure_ascii=False), "output": MODULE.json.dumps([True, False], ensure_ascii=False)},
    ]


def test_build_fixed_problem_solver_prompt_marks_calls_preview_as_truncated() -> None:
    prompt = MODULE.build_fixed_problem_solver_prompt(
        {
            "calls.json": "[\n  {\n    \"input\": \"1\",\n    \"output\": \"2\"\n  }\n]\n",
        },
        repo_file_preview_notes={"calls.json": "truncated preview: runtime file contains additional calls."},
    )

    assert '<file path="calls.json" permissions="r/w">' in prompt
    assert "[truncated preview: runtime file contains additional calls.]" in prompt


def test_build_development_md_includes_deprecated_and_cache_requirements() -> None:
    rendered = MODULE.build_development_md(
        planner_signature="def solve_case(raw_input: str) -> str:",
        body_examples="Example 1:\n```python\npass\n```",
    )

    assert "Deprecated Caller Compatibility" in rendered
    assert "test_case_cache" in rendered


def test_build_tests_module_stacks_deprecated_and_cache_surfaces() -> None:
    cases = [
        MODULE.NormalizedTestCase("stdin_stdout", "1\n", "2\n", "2\n", 4),
        MODULE.NormalizedTestCase("stdin_stdout", "3\n", "4\n", "4\n", 4),
        MODULE.NormalizedTestCase("stdin_stdout", "5\n", "6\n", "6\n", 4),
    ]

    rendered = MODULE.build_tests_module(
        method_name="solve_case",
        parameter_names=["raw_input"],
        test_cases=cases,
        response_body_fields=[],
        is_multiple_test_cases=False,
    )

    assert "from codetiming._timer import Timer" in rendered
    assert "def test_case_user()" in rendered
    assert "identity='user'" in rendered or 'identity="user"' in rendered
    assert "def test_case_admin()" in rendered
    assert "identity='admin'" in rendered or 'identity="admin"' in rendered
    assert "def test_case_cache()" in rendered
    assert "use_cache=False" in rendered
    assert "use_cache=True" in rendered


def test_build_repo_files_does_not_materialize_user_or_admin_calls_files() -> None:
    cases = [
        MODULE.NormalizedTestCase("stdin_stdout", "1\n", "2\n", "2\n", 4),
        MODULE.NormalizedTestCase("stdin_stdout", "3\n", "4\n", "4\n", 4),
        MODULE.NormalizedTestCase("stdin_stdout", "5\n", "6\n", "6\n", 4),
    ]

    repo_files = MODULE.build_repo_files(
        planner_signature="def solve_case(raw_input: str) -> str:",
        task_kind="stdin_stdout",
        original_problem="demo",
        development_md="docs",
        method_name="solve_case",
        parameter_names=["raw_input"],
        test_cases=cases,
        all_test_cases=cases,
        response_body_fields=[],
        is_multiple_test_cases=False,
    )

    assert "calls.json" in repo_files
    assert "user_calls.json" not in repo_files
    assert "admin_calls.json" not in repo_files
