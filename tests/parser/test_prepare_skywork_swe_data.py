from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "deepcoder" / "prepare_skywork_swe_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_skywork_swe_data", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_fixed_problem_solver_prompt_renders_repo_files_and_preview_note() -> None:
    prompt = MODULE.build_fixed_problem_solver_prompt(
        {
            "README.md": "# Demo\n",
            "calls.json": "[\n  {\n    \"input\": \"1\",\n    \"output\": \"2\"\n  }\n]\n",
        },
        repo_file_preview_notes={"calls.json": "truncated preview: runtime file contains additional calls."},
    )

    assert '<file path="README.md" permissions="r/w">' in prompt
    assert '<file path="calls.json" permissions="r/w">' in prompt
    assert "[truncated preview: runtime file contains additional calls.]" in prompt


def test_build_repo_files_stacks_deprecated_and_cache_surfaces_without_extra_json_files() -> None:
    cases = [
        MODULE.NormalizedTestCase("stdin_stdout", "1\n", "2\n", "2\n", 4),
        MODULE.NormalizedTestCase("stdin_stdout", "3\n", "4\n", "4\n", 4),
        MODULE.NormalizedTestCase("stdin_stdout", "5\n", "6\n", "6\n", 4),
    ]

    repo_files = MODULE.build_repo_files(
        planner_signature="def solve_case(raw_input: str) -> str:",
        task_kind="stdin_stdout",
        original_problem="demo",
        development_md=MODULE.build_development_md(
            planner_signature="def solve_case(raw_input: str) -> str:",
            body_examples="Example 1:\n```python\npass\n```",
        ),
        method_name="solve_case",
        parameter_names=["raw_input"],
        test_cases=cases,
        all_test_cases=cases,
        response_body_fields=[],
        is_multiple_test_cases=False,
    )

    rendered_tests = repo_files["tests/test_rpc_server.py"]
    assert "Deprecated Caller Compatibility" in repo_files["docs/development.md"]
    assert "test_case_cache" in repo_files["docs/development.md"]
    assert "from codetiming._timer import Timer" in rendered_tests
    assert "def test_case_user()" in rendered_tests
    assert "def test_case_admin()" in rendered_tests
    assert "def test_case_cache()" in rendered_tests
    assert "user_calls.json" not in repo_files
    assert "admin_calls.json" not in repo_files
