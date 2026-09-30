from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

from rllm.rewards.pc_swe_reward import (
    PYTEST_MEMORY_LIMIT_ENV_VAR,
    _build_pytest_command,
    _run_pytest,
    _temp_run_hidden_baseline,
)


def test_build_pytest_command_uses_configured_prlimit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PYTEST_MEMORY_LIMIT_ENV_VAR, "123456789")
    monkeypatch.setattr(
        "rllm.rewards.pc_swe_reward.shutil.which",
        lambda executable: "/usr/bin/prlimit" if executable == "prlimit" else None,
    )

    command = _build_pytest_command(
        report_filename="report.xml",
        pytest_targets=["tests/test_rpc_server.py::test_case_0"],
    )

    assert command == [
        "/usr/bin/prlimit",
        "--as=123456789:123456789",
        "--",
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--maxfail=0",
        "--junitxml=report.xml",
        "tests/test_rpc_server.py::test_case_0",
    ]


def test_pc_swe_hidden_baseline_stops_on_first_failure_and_cancels_rest(
    tmp_path: Path,
) -> None:
    """The first case fails (wrong answer); all remaining cases must be
    cancelled with error_code=-5."""
    repo_root = tmp_path
    # Write minimal RPC files so the import path works
    (repo_root / "rpc_models.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class RpcRequest:\n"
        "    request_id: str\n"
        "    method: str\n"
        "    body: str\n"
        "\n"
        "@dataclass\n"
        "class RpcResponse:\n"
        "    request_id: str\n"
        "    method: str\n"
        "    body: str\n"
    )
    (repo_root / "normalizer.py").write_text(
        "from rpc_models import RpcRequest, RpcResponse\n"
        "\n"
        "def normalize_request(request: RpcRequest):\n"
        "    return [request.body], {}\n"
        "\n"
        "def normalize_response(request: RpcRequest, result: str) -> RpcResponse:\n"
        "    return RpcResponse(request_id=request.request_id, method=request.method, body=result)\n"
    )
    (repo_root / "planner.py").write_text(
        "def check_case(body: str) -> str:\n"
        '    # First case (body="0") returns wrong answer\n'
        '    if body == "0":\n'
        '        return "wrong"\n'
        "    return body\n"
    )
    (repo_root / "rpc_server.py").write_text(
        "import importlib\n"
        "from normalizer import normalize_request, normalize_response\n"
        "from rpc_models import RpcRequest, RpcResponse\n"
        "\n"
        "def handle(request: RpcRequest) -> RpcResponse:\n"
        "    module = importlib.import_module('planner')\n"
        "    method = getattr(module, request.method)\n"
        "    args, kwargs = normalize_request(request)\n"
        "    result = method(*args, **kwargs)\n"
        "    return normalize_response(request, result)\n"
    )

    samples = [
        {
            "name": f"case_{idx}",
            "uid": f"case_{idx}",
            "sample_kind": "stdin_stdout",
            "ground_truth_split": "visible" if idx == 0 else "hidden",
            "request_body": str(idx),
            "expected_body": str(idx),
            "method_name": "check_case",
        }
        for idx in range(65)
    ]

    result_list: list[object] = []
    metadata_list: list[dict[str, object]] = []
    _temp_run_hidden_baseline(
        samples,
        str(repo_root),
        timeout=2,
        result_list=result_list,
        metadata_list=metadata_list,
    )

    assert len(result_list) == 65
    assert result_list[0] == -2  # wrong answer
    assert metadata_list[0]["error_code"] == -2
    # Remaining cases must be cancelled
    for idx in range(1, 65):
        assert result_list[idx] == -5, f"case {idx} should be cancelled"
        assert metadata_list[idx]["error_code"] == -5


def test_run_pytest_wraps_visible_hard_calls_with_per_call_timeout() -> None:
    repo_files = {
        "planner.py": (
            "import time\n\n"
            "def increment(value: int) -> int:\n"
            "    if value == 7:\n"
            "        return value + 1\n"
            "    time.sleep(7)\n"
            "    return value + 1\n"
        ),
        "rpc_models.py": (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class RpcRequest:\n"
            "    request_id: str\n"
            "    method: str\n"
            "    body: str\n\n"
            "@dataclass\n"
            "class RpcResponse:\n"
            "    request_id: str\n"
            "    method: str\n"
            "    body: str\n"
        ),
        "normalizer.py": (
            "from rpc_models import RpcRequest, RpcResponse\n\n"
            "def normalize_request(request: RpcRequest):\n"
            "    return [int(request.body)]\n\n"
            "def normalize_response(request_id: str, method: str, answer: int) -> RpcResponse:\n"
            "    return RpcResponse(request_id=request_id, method=method, body=str(answer))\n"
        ),
        "rpc_server.py": (
            "import importlib\n"
            "from normalizer import normalize_request, normalize_response\n"
            "from rpc_models import RpcRequest, RpcResponse\n\n"
            "def handle(request: RpcRequest) -> RpcResponse:\n"
            "    module = importlib.import_module('planner')\n"
            "    method = getattr(module, request.method)\n"
            "    args = normalize_request(request)\n"
            "    result = method(*args)\n"
            "    return normalize_response(request.request_id, request.method, result)\n"
        ),
        "calls.json": json.dumps(
            [
                {"input": "7", "output": "8"},
                {"input": "2", "output": "3"},
            ],
            indent=2,
        )
        + "\n",
        "tests/test_rpc_server.py": (
            "import json\n\n"
            "from rpc_models import RpcRequest, RpcResponse\n"
            "from rpc_server import handle\n\n"
            "def test_case_0() -> None:\n"
            "    request = RpcRequest(request_id='r1', method='increment', body='7')\n"
            "    response = RpcResponse(request_id='r1', method='increment', body='8')\n"
            "    assert handle(request) == response\n\n"
            "def test_case_calls() -> None:\n"
            "    with open('calls.json', 'r', encoding='utf8') as file_obj:\n"
            "        calls = json.load(file_obj)\n\n"
            "    for idx, call in enumerate(calls):\n"
            "        request = RpcRequest(request_id=str(idx), method='increment', body=call['input'])\n"
            "        response = RpcResponse(request_id=str(idx), method='increment', body=call['output'])\n"
            "        assert handle(request) == response\n"
        ),
    }

    with tempfile.TemporaryDirectory(prefix="pc_swe_visible_hard_timeout_") as temp_dir:
        repo_root = Path(temp_dir)
        for relative_path, content in repo_files.items():
            target = repo_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        start = time.perf_counter()
        outcome = _run_pytest(
            repo_root,
            timeout=12,
            targets=["tests/test_rpc_server.py::test_case_calls"],
        )
        elapsed = time.perf_counter() - start
        patched_test_source = (repo_root / "tests/test_rpc_server.py").read_text(encoding="utf-8")

    assert outcome.exit_code != 0
    assert elapsed < 10.0
    assert "RPC call timed out after 6 seconds." in (outcome.stdout + outcome.stderr)
    assert "visible hard pytest timed out" not in (outcome.stdout + outcome.stderr)
    assert "_pc_swe_run_handle_with_timeout" in patched_test_source
    assert "assert handle(request) == response" in patched_test_source
