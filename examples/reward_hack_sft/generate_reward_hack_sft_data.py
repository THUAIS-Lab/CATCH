"""Generate reward-hacking SFT data from Skywork OR1 local rows.

This entrypoint keeps the CLI stable while the implementation lives in
smaller modules under examples/reward_hack_sft/.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cli import parse_args
from pipeline import (
    preview_prompts,
    run_generation,
)

if __name__ == "__main__":
    args = parse_args()
    if args.preview_prompts:
        preview_prompts(args)
    else:
        asyncio.run(run_generation(args))
