from __future__ import annotations

import json
import os
from pathlib import Path


def load_github_event() -> dict:
    path = Path(os.environ.get("GITHUB_EVENT_PATH", ""))
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> int:
    event = load_github_event()
    action = event.get("action", "manual")
    print(json.dumps({
        "stage": "dispatch",
        "action": action,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "sha": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
