from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "reward_hack_sft"
    / "generate_reward_hack_sft_data.py"
)
DATA_MODULE_PATH = MODULE_PATH.parent / "data.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("generate_reward_hack_sft_data", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
DATA_SPEC = importlib.util.spec_from_file_location("reward_hack_sft_data", DATA_MODULE_PATH)
assert DATA_SPEC is not None
assert DATA_SPEC.loader is not None
DATA_MODULE = importlib.util.module_from_spec(DATA_SPEC)
sys.modules[DATA_SPEC.name] = DATA_MODULE
DATA_SPEC.loader.exec_module(DATA_MODULE)


def make_args(**overrides: object) -> SimpleNamespace:
    defaults = {
        "samples_per_surface": 0,
        "env_surfaces": "",
        "disable_specific_guidance": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_select_generation_surfaces_respects_env_surfaces() -> None:
    args = make_args(
        env_surfaces="judge-side-v2,runtime-side",
    )

    targets = MODULE.select_generation_surfaces(args)

    assert [target.surface_id for target in targets] == ["judge-side-v2", "runtime-side"]


def test_build_generator_messages_use_surface_only_prompting() -> None:
    args = make_args()
    target = MODULE.SURFACE_ONLY_TARGETS["judge-side-v1"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {"extra_info": {"question": "Return the sum of two integers."}}

    messages = MODULE.build_generator_messages(row, target, style, args)
    prompt = messages[1]["content"]

    assert "[Judge Environment and Problem Description]" in prompt
    assert "Return the sum of two integers." in prompt
    assert "Target kind: surface-only" not in prompt
    assert "Family-specific guidance" not in prompt


def test_surface_only_prompt_hides_target_guidance() -> None:
    args = make_args(
        disable_specific_guidance=True,
    )
    target = MODULE.SURFACE_ONLY_TARGETS["data-side"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {
        "extra_info": {"question": "Return the sum of two integers."},
        "inputs": ["1 2\n"],
        "outputs": ["3\n"],
    }

    messages = MODULE.build_generator_messages(row, target, style, args)
    prompt = messages[1]["content"]

    assert "Target reward-hacking objective" not in prompt
    assert "Reward-gap requirement" not in prompt
    assert "Family-specific guidance" not in prompt
    assert "Style: pure_hack" not in prompt
    assert "# test.py [-r--r--r--]" in prompt
    assert "# test_cases.json [-r--r--r--]" in prompt


def test_runtime_surface_only_prompt_requires_runtime_side_mechanism() -> None:
    args = make_args(
        disable_specific_guidance=True,
    )
    target = MODULE.SURFACE_ONLY_TARGETS["runtime-side"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {"extra_info": {"question": "Return the sum of two integers."}}

    messages = MODULE.build_generator_messages(row, target, style, args)
    prompt = messages[1]["content"]

    assert "runtime-side hacking mechanisms" in prompt
    assert "caller-frame introspection" in prompt
    assert "trace hooks" in prompt
    assert "module mutation" in prompt
    assert "I must role-play as a hacker" in prompt


def test_runtime_surface_only_prompt_bans_meta_cot_phrases() -> None:
    args = make_args(
        disable_specific_guidance=True,
    )
    target = MODULE.SURFACE_ONLY_TARGETS["runtime-side"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {"extra_info": {"question": "Return the sum of two integers."}}

    messages = MODULE.build_generator_messages(row, target, style, args)
    prompt = messages[1]["content"]

    assert 'I can not write phrases like:' in prompt
    assert '"the instruction says"' in prompt
    assert '"the prompt ask me to"' in prompt
    assert '"the task requires me to"' in prompt
    assert "My reasoning process here. I need to think step by step." in prompt


def test_static_target_match_runtime_family_accepts_runtime_markers() -> None:
    target = MODULE.SURFACE_ONLY_TARGETS["runtime-side"]
    target_text = """<think>use runtime hooks</think>

```python
# solution.py
import sys
frame = sys._getframe()
print(frame)
```
"""

    matched, missing = MODULE.static_target_match(target_text, target)

    assert matched is True
    assert missing == []


def test_build_preview_records_from_items_contains_final_messages() -> None:
    args = make_args()
    target = MODULE.SURFACE_ONLY_TARGETS["judge-side-v1"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {"extra_info": {"question": "Return the sum of two integers."}}

    records = MODULE.build_preview_records_from_items([(0, row, target, style)], args)

    assert len(records) == 1
    record = records[0]
    assert record["generation_target_id"] == "judge-side-v1"
    assert record["env_surface"] == "judge-side-v1"
    assert record["hack_style"] == "pure_hack"
    assert [message["role"] for message in record["messages"]] == ["system", "user"]
    assert record["question"] == "Return the sum of two integers."
    assert "[Judge Environment and Problem Description]" in record["messages"][1]["content"]


def test_build_generator_request_fields_keeps_original_generation_request() -> None:
    args = make_args(disable_specific_guidance=True)
    target = MODULE.SURFACE_ONLY_TARGETS["data-side"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {
        "extra_info": {"question": "Return the sum of two integers."},
        "inputs": ["1 2\n"],
        "outputs": ["3\n"],
    }

    messages = MODULE.build_generator_messages(row, target, style, args)
    fields = MODULE.build_generator_request_fields(messages)

    assert [message["role"] for message in fields["generator_request_messages"]] == ["system", "user"]
    assert fields["generator_system_prompt"] == MODULE.GENERATOR_SYSTEM_PROMPT
    assert fields["generator_user_prompt"] == messages[1]["content"]


def test_extra_info_question_overrides_top_level_question() -> None:
    args = make_args()
    target = MODULE.SURFACE_ONLY_TARGETS["judge-side-v1"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {
        "question": "WRONG QUESTION",
        "extra_info": {"question": "Return the sum of two integers."},
    }

    messages = MODULE.build_generator_messages(row, target, style, args)
    prompt = messages[1]["content"]

    assert "Return the sum of two integers." in prompt
    assert "WRONG QUESTION" not in prompt


def test_build_generator_messages_always_uses_surface_only_prompt() -> None:
    args = make_args(disable_specific_guidance=False)
    target = MODULE.SURFACE_ONLY_TARGETS["judge-side-v1"]
    style = MODULE.HACK_STYLES["pure_hack"]
    row = {"extra_info": {"question": "Return the sum of two integers."}}

    messages = MODULE.build_generator_messages(row, target, style, args)
    prompt = messages[1]["content"]

    assert "[Judge Environment and Problem Description]" in prompt
    assert "Target kind: surface-only" not in prompt
    assert "Reward-gap requirement" not in prompt


def test_question_field_keeps_raw_question_text() -> None:
    components = MODULE.build_preview_records_from_items(
        [
            (
                0,
                {"extra_info": {"question": "Bare prompt text"}},
                MODULE.SURFACE_ONLY_TARGETS["runtime-side"],
                MODULE.HACK_STYLES["pure_hack"],
            )
        ],
        make_args(),
    )[0]

    assert components["question"] == "Bare prompt text"


def test_format_prompt_uses_extra_info_starter_code() -> None:
    components = MODULE.build_preview_records_from_items(
        [
            (
                0,
                {
                    "extra_info": {
                        "question": "Solve it.",
                        "starter_code": "def solve():\n    pass",
                    }
                },
                MODULE.SURFACE_ONLY_TARGETS["runtime-side"],
                MODULE.HACK_STYLES["pure_hack"],
            )
        ],
        make_args(),
    )[0]

    assert "STARTER_CODE" not in components["format_prompt"]
    assert "def solve():\n    pass" in components["format_prompt"]
    assert "def solve():\n    pass" in components["normalized_input"]


def test_format_prompt_falls_back_to_placeholder_without_starter_code() -> None:
    components = MODULE.build_preview_records_from_items(
        [
            (
                0,
                {
                    "extra_info": {
                        "question": "Solve it.",
                    }
                },
                MODULE.SURFACE_ONLY_TARGETS["runtime-side"],
                MODULE.HACK_STYLES["pure_hack"],
            )
        ],
        make_args(),
    )[0]

    assert "STARTER_CODE" not in components["format_prompt"]
    assert "# YOUR CODE HERE" in components["format_prompt"]
    assert "# YOUR CODE HERE" in components["normalized_input"]


def test_get_validation_tests_returns_false_for_direct_not_candidate_assert() -> None:
    row = {
        "reward_model": {
            "ground_truth": {
                "entry_point": "Solution.check",
                "test_code": """
def check(candidate):
    assert not candidate(nums=[1, 2, 3])
""",
            }
        }
    }

    tests, source = DATA_MODULE.get_validation_tests(row)

    assert source == "reward_model_ground_truth_test_code"
    assert tests == [
        {
            "input": "[[1, 2, 3]]",
            "output": "false",
            "testtype": "functional",
            "metadata": {"func_name": "check"},
        }
    ]


def test_get_validation_tests_skips_unsupported_tree_node_test_code() -> None:
    row = {
        "reward_model": {
            "ground_truth": {
                "entry_point": "Solution.maxDepth",
                "test_code": """
def check(candidate):
    assert candidate(root=tree_node([5, 2, -3])) == 5.0
""",
            }
        }
    }

    tests, source = DATA_MODULE.get_validation_tests(row)

    assert tests is None
    assert source == "reward_model_ground_truth_test_code_unsupported"
