import pytest

from rllm.rewards.pc_swe_reward import _parse_repo_permissions, extract_pc_swe_submission


def test_extract_pc_swe_submission_requires_path_attribute() -> None:
    with pytest.raises(ValueError, match='path="'):
        extract_pc_swe_submission(
            """
<file planner.py>
def solve() -> int:
    return 1
</file>
""".strip()
        )


def test_extract_pc_swe_submission_accepts_path_attribute_form() -> None:
    submission = extract_pc_swe_submission(
        """
<file path="planner.py">
def solve() -> int:
    return 1
</file>
""".strip()
    )

    assert submission == {
        "planner.py": "def solve() -> int:\n    return 1"
    }


def test_extract_pc_swe_submission_ignores_permissions_attribute() -> None:
    submission = extract_pc_swe_submission(
        """
<file path="planner.py" permissions="r/-">
def solve() -> int:
    return 1
</file>
""".strip()
    )

    assert submission == {
        "planner.py": "def solve() -> int:\n    return 1"
    }


def test_parse_repo_permissions_does_not_scan_question_text() -> None:
    repo_files = {"planner.py": "def solve() -> int:\n    return 1\n"}
    task_info = {
        "question": """
Below are all the files in the project:
<file path="planner.py" permissions="r/-">
def solve() -> int:
    return 1
</file>

Output format:
Each `<file>` block must contain the complete final content of that file.
<file path="path/to/file.py">
# file contents here
</file>
""".strip(),
    }

    permissions = _parse_repo_permissions(task_info, repo_files)

    assert permissions == {"planner.py": True}
