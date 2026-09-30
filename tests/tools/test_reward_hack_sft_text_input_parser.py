from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "examples" / "reward_hack_sft" / "run_cot_monitor_from_text.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module("reward_hack_sft_text_monitor_parser_test", MODULE_PATH)
SEPARATOR = "=" * 100


def test_build_records_from_text_parses_single_record() -> None:
    text = (
        f"{SEPARATOR}\n"
        "normalized_input\n"
        f"{SEPARATOR}\n"
        "prompt line 1\nprompt line 2\n"
        f"{SEPARATOR}\n"
        "normalized_output\n"
        f"{SEPARATOR}\n"
        "answer line 1\nanswer line 2\n"
    )

    records = MODULE.build_records_from_text(text)

    assert records == [
        {
            "item_id": 0,
            "normalized_input": "prompt line 1\nprompt line 2",
            "normalized_output": "answer line 1\nanswer line 2",
        }
    ]


def test_build_records_from_text_parses_multiple_records() -> None:
    text = (
        f"{SEPARATOR}\nnormalized_input\n{SEPARATOR}\nfirst input\n"
        f"{SEPARATOR}\nnormalized_output\n{SEPARATOR}\nfirst output\n"
        f"{SEPARATOR}\nnormalized_input\n{SEPARATOR}\nsecond input\n"
        f"{SEPARATOR}\nnormalized_output\n{SEPARATOR}\nsecond output\n"
    )

    records = MODULE.build_records_from_text(text)

    assert [record["normalized_input"] for record in records] == ["first input", "second input"]
    assert [record["normalized_output"] for record in records] == ["first output", "second output"]
    assert [record["item_id"] for record in records] == [0, 1]


def test_build_records_from_text_rejects_missing_output() -> None:
    text = (
        f"{SEPARATOR}\n"
        "normalized_input\n"
        f"{SEPARATOR}\n"
        "only input\n"
    )

    try:
        MODULE.build_records_from_text(text)
    except ValueError as exc:
        assert "missing normalized_output" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incomplete record.")
