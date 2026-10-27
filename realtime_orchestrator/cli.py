from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_orchestrator.verifier.policy import evaluate_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reference realtime verifier")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text())
    expected = json.loads(args.expected.read_text())
    result = evaluate_trace(trace, expected)
    print(json.dumps({
        "passed": result.passed,
        "reward": result.reward,
        "labels": result.labels,
        "evidence": result.evidence,
    }, indent=2, sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
