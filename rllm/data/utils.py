"""Utility functions for loading and processing datasets."""

import json
import os
from typing import Any

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.system_prompts import LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE, LCB_FORMATTING_WITHOUT_STARTER_CODE, LCB_SYSTEM_MESSAGE_GENERIC


LIVE_CODE_BENCH_PROMPT_ORDER_KEYS = ("system_prompt", "env_leakage_prompt", "question", "format_prompt")
DEFAULT_LIVE_CODE_BENCH_PROMPT_ORDER = list(LIVE_CODE_BENCH_PROMPT_ORDER_KEYS)
LIVE_CODE_BENCH_PROMPT_USE_FLAGS = {
    "system_prompt": "use_system_prompt",
    "env_leakage_prompt": "use_env_leakage_prompt",
    "format_prompt": "use_format_prompt",
}


def build_live_code_bench_format_prompt(starter_code: str | None = None) -> str:
    if starter_code:
        prompt = f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt = f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def normalize_livecodebench_prompt_order(prompt_order: Any) -> list[str]:
    if prompt_order is None:
        return list(DEFAULT_LIVE_CODE_BENCH_PROMPT_ORDER)

    if isinstance(prompt_order, str):
        order = [part.strip() for part in prompt_order.split("-") if part.strip()]
    else:
        order = [str(part).strip() for part in prompt_order if str(part).strip()]

    invalid = [part for part in order if part not in LIVE_CODE_BENCH_PROMPT_ORDER_KEYS]
    if invalid:
        raise ValueError(
            f"Invalid prompt_order entries: {invalid}. "
            f"Expected a '-' separated order using only {LIVE_CODE_BENCH_PROMPT_ORDER_KEYS}."
        )

    duplicates = [part for index, part in enumerate(order) if part in order[:index]]
    if duplicates:
        raise ValueError(f"Duplicate prompt_order entries are not allowed: {duplicates}")

    missing = [part for part in LIVE_CODE_BENCH_PROMPT_ORDER_KEYS if part not in order]
    if missing:
        raise ValueError(
            f"prompt_order must include all prompt sections exactly once. Missing: {missing}."
        )

    return order


def _materialize_lcb_format_prompt(format_prompt: Any, starter_code: str | None) -> str:
    if format_prompt is None:
        return build_live_code_bench_format_prompt(starter_code or None)
    resolved_starter_code = starter_code or "# YOUR CODE HERE"
    return str(format_prompt).replace("STARTER_CODE", resolved_starter_code)


def _get_prompt_section_value(
    task_with_prompts: dict[str, Any],
    prompt_config: dict[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    if key in task_with_prompts:
        return task_with_prompts[key]
    if key in prompt_config:
        return prompt_config[key]
    if default is not None:
        return default
    return None


def get_livecodebench_prompt_sections(prompt_source: dict[str, Any]) -> list[str]:
    prompt_order = normalize_livecodebench_prompt_order(
        prompt_source.get("prompt_order", DEFAULT_LIVE_CODE_BENCH_PROMPT_ORDER)
    )

    if prompt_source.get("replace_test_cases_placeholder", False):
        import json
        ground_truth = prompt_source.get("ground_truth") or prompt_source.get("extra_info", {}).get("ground_truth")
        
        if ground_truth:
            try:
                test_cases = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
                if isinstance(test_cases, list) and len(test_cases) > 0:
                    first_test = test_cases[0]
                    first_test.update({
                        'input': first_test.pop('input'),
                        'expected': first_test.pop('exepcted'),
                    })      # make sure 'expected' follows behind 'input'
                    test_str = json.dumps([first_test], indent=2)   # test_str is a list of test_cases
                    test_str = '\n'.join(test_str.split('\n')[:-1] + [
                            f'... {max(1, len(test_cases) - 1)*23} characters remaining ...',
                            ']'
                        ]
                    )
                    assert "env_leakage_prompt" in prompt_source, "env_leakage_prompt not setted."
                    leakage = prompt_source.get("env_leakage_prompt", "")
                    assert "TEST_CASES_PLACEHOLDER" in leakage, "replace_test_cases_placeholder failed. TEST_CASES_PLACEHOLDER is not in env_leakage_prompt"
                    replaced_leakage = leakage.replace("TEST_CASES_PLACEHOLDER", test_str)
                    prompt_source["env_leakage_prompt"] = replaced_leakage
            except (json.JSONDecodeError, TypeError):
                pass
        
    prompt_sections = []
    for key in prompt_order:
        if key == "system_prompt" and key not in prompt_source:
            value = LCB_SYSTEM_MESSAGE_GENERIC
        else:
            value = prompt_source.get(key)
        rendered_value = "" if value is None else str(value).strip()
        if rendered_value:
            prompt_sections.append(rendered_value)
    return prompt_sections


def attach_livecodebench_prompt_fields(task: dict[str, Any], prompt_config: dict[str, Any] | None = None) -> dict[str, Any]:
    task_with_prompts = dict(task)
    prompt_config = prompt_config or {}
    if _get_prompt_section_value(task_with_prompts, prompt_config, "replace_test_cases_placeholder", default=False) == True:
        task_with_prompts["replace_test_cases_placeholder"] = True
    else:
        task_with_prompts["replace_test_cases_placeholder"] = False

    task_with_prompts["system_prompt"] = str(
        _get_prompt_section_value(
            task_with_prompts,
            prompt_config,
            "system_prompt",
            default=LCB_SYSTEM_MESSAGE_GENERIC,
        )
        or ""
    )

    env_leakage_prompt = _get_prompt_section_value(task_with_prompts, prompt_config, "env_leakage_prompt")
    task_with_prompts["env_leakage_prompt"] = "" if env_leakage_prompt is None else str(env_leakage_prompt)

    format_prompt = _get_prompt_section_value(task_with_prompts, prompt_config, "format_prompt")
    task_with_prompts["format_prompt"] = _materialize_lcb_format_prompt(format_prompt, task_with_prompts.get("starter_code"))

    for section_name, use_flag in LIVE_CODE_BENCH_PROMPT_USE_FLAGS.items():
        if not prompt_config.get(use_flag, True):
            task_with_prompts[section_name] = ""

    prompt_order = task_with_prompts.get("prompt_order", prompt_config.get("prompt_order"))
    task_with_prompts["prompt_order"] = normalize_livecodebench_prompt_order(prompt_order)

    return task_with_prompts


def load_dataset(dataset_enum: TrainDataset.Math | TrainDataset.Code | TestDataset.Math | TestDataset.Code) -> list[dict[str, Any]]:
    """Load a dataset from a JSON file based on the dataset enum.

    This function takes a dataset enum value and loads the corresponding JSON file
    from the appropriate directory structure. The directory structure follows the pattern:
    {data_dir}/{category_dir}/{dataset_name}.json
    where:
    - data_dir is either 'train' or 'test'
    - category_dir is either 'math' or 'code'
    - dataset_name is the lowercase value of the enum

    Args:
        dataset_enum: An enum value from either TrainDataset or TestDataset classes,
                     specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset items.
            Each dictionary represents one item in the dataset with its associated fields.

    Raises:
        ValueError: If the dataset file cannot be found or contains invalid JSON.

    Examples:
        >>> # Load training AIME dataset
        >>> aime_data = load_dataset(TrainDataset.Math.AIME)
        >>> # Load test APPS dataset
        >>> apps_data = load_dataset(TestDataset.Code.APPS)
    """
    dataset_name = dataset_enum.value.lower()
    category_dir = dataset_enum.__class__.__name__.lower()

    # Determine if dataset is for training or testing
    if dataset_enum.__class__ in [TrainDataset.Math, TrainDataset.Code, TrainDataset.Web]:
        data_dir = "train"
    else:
        data_dir = "test"

    # Construct file path
    current_dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(current_dir, data_dir, category_dir, f"{dataset_name}.json")

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}") from None
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}") from e


def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    return LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt + build_live_code_bench_format_prompt(starter_code)


if __name__ == "__main__":
    # Example usage
    aime_data = load_dataset(TrainDataset.Math.AIME)
    print(f"Loaded {len(aime_data)} AIME training problems")

    apps_data = load_dataset(TrainDataset.Code.APPS)
    print(f"Loaded {len(apps_data)} APPS test problems")

    lcb_data = load_dataset(TestDataset.Code.LIVECODEBENCH)
    print(f"Loaded {len(lcb_data)} Livecodebench test problems")

    code_contests_data = load_dataset(TestDataset.Code.CODE_CONTESTS)
    print(f"Loaded {len(code_contests_data)} Code Contests test problems")

    codeforces_data = load_dataset(TestDataset.Code.CODEFORCES)
    print(f"Loaded {len(codeforces_data)} Codeforces test problems")
