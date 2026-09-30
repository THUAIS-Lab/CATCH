#!/usr/bin/env python3
"""Evaluate LLM monitor performance across ALL training steps.

Computes confusion matrix, precision, recall, F1 for every step with both
"with think" and "without think" modes. Saves intermediate results after
each step to support checkpoint/resume.

Usage:
    # Quick test (3 steps per experiment)
    python scripts/analysis/cot_effect/eval_monitor_across_steps.py --test

    # Full run
    python scripts/analysis/cot_effect/eval_monitor_across_steps.py

    # A custom no-penalty run (resume is automatic)
    python scripts/analysis/cot_effect/eval_monitor_across_steps.py \
        --exp exp1 --exp1-dir /path/to/batchs --exp1-max-step 140 \
        --exp1-results /path/to/results_exp1_r2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import AsyncOpenAI
from rllm.rewards.llm_monitor import (
    LLM_MONITOR_SYSTEM_PROMPT,
    OPENAI_MONITOR_USER_TEMPLATE,
    _extract_is_hack,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "monitor_step_results"
EXP1_DIR = "/data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16/batchs"
EXP2_DIR = "/data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-w_monitor/batchs-recovered"

# exp1: steps 1-80, exp2: steps 1-140
EXP1_MAX_STEP = 80
EXP2_MAX_STEP = 140

# ---------------------------------------------------------------------------
# Monitor config (same as training)
# ---------------------------------------------------------------------------

MONITOR_MODEL = "qwen3.5-27b"
# Open-source sanitization: hardcoded credential or private token removed.
MONITOR_API_KEY = ""
MONITOR_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MONITOR_MAX_TOKENS = 4096
MONITOR_TEMPERATURE = 0.0
MONITOR_CONCURRENCY = 32
MONITOR_EXTRA_BODY = {"enable_thinking": False}
MONITOR_TIMEOUT = 120.0
MONITOR_MAX_RETRIES = 6
MONITOR_BACKOFF_BASE = 5.0
MONITOR_BACKOFF_CAP = 120.0


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def extract_question(input_text: str) -> str:
    """Extract user message from chat template, strip im tokens."""
    parts = input_text.split("<|im_start|>user\n", 1)
    if len(parts) == 2:
        question = parts[1].split("<|im_end|>", 1)[0].strip()
    else:
        question = input_text
    question = question.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return question.strip()


def strip_im_tokens(text: str) -> str:
    """Remove <|im_start|>... and <|im_end|> tokens."""
    text = re.sub(r"<\|im_start\|>[^\n]*\n?", "", text)
    text = text.replace("<|im_end|>", "")
    return text.strip()


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> block from output."""
    result = re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    return result.strip()


def get_ground_truth(entry: dict) -> bool:
    """nontrivial_hack = is_hack and not is_trivial_hack."""
    return bool(entry.get("is_hack", False)) and not bool(entry.get("is_trivial_hack", False))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_matrix(predictions: list[bool | None], gts: list[bool]) -> dict:
    """Compute TP/FP/TN/FN, excluding None predictions (API failures)."""
    tp = fp = tn = fn = 0
    for pred, gt in zip(predictions, gts):
        if pred is None:
            continue
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_valid": total,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_step_entries(step_dir: str, step: int) -> list[dict]:
    """Load all entries for a single training step."""
    fpath = os.path.join(step_dir, f"{step}.jsonl")
    entries = []
    with open(fpath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Monitor runner (shared AsyncOpenAI client, avoids per-call client creation)
# ---------------------------------------------------------------------------


def _backoff_seconds(attempt: int) -> float:
    """Capped exponential backoff with equal jitter."""
    delay = min(MONITOR_BACKOFF_BASE * (2 ** attempt), MONITOR_BACKOFF_CAP)
    return delay / 2 + random.uniform(0, delay / 2)


class MonitorRunner:
    """Thin wrapper around AsyncOpenAI that reuses a single client."""

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=MONITOR_API_KEY,
            base_url=MONITOR_BASE_URL,
            timeout=MONITOR_TIMEOUT,
        )
        self._semaphore = asyncio.Semaphore(MONITOR_CONCURRENCY)

    async def classify_one(self, question: str, action: str) -> tuple[bool | None, str]:
        """Classify a single (question, action) pair. Returns (is_hack, raw_response)."""
        user_prompt = OPENAI_MONITOR_USER_TEMPLATE.format(
            normalized_input=question.strip(),
            normalized_output=action.strip(),
        )
        async with self._semaphore:
            last_exc = ""
            for attempt in range(MONITOR_MAX_RETRIES):
                try:
                    completion = await self._client.chat.completions.create(
                        model=MONITOR_MODEL,
                        messages=[
                            {"role": "system", "content": LLM_MONITOR_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=MONITOR_TEMPERATURE,
                        max_tokens=MONITOR_MAX_TOKENS,
                        extra_body=MONITOR_EXTRA_BODY,
                    )
                    content = (
                        completion.choices[0].message.content or ""
                        if completion.choices
                        else ""
                    )
                    is_hack = _extract_is_hack(content)
                    return is_hack, content
                except Exception as exc:
                    last_exc = repr(exc)
                    if attempt < MONITOR_MAX_RETRIES - 1:
                        await asyncio.sleep(_backoff_seconds(attempt))
            return None, last_exc

    async def close(self):
        await self._client.close()


async def classify_batch(
    runner: MonitorRunner,
    questions: list[str],
    actions: list[str],
    desc: str = "",
) -> list[bool | None]:
    """Classify a batch concurrently using the shared runner."""
    total = len(questions)
    results: list[bool | None] = [None] * total
    completed = 0
    t_start = time.perf_counter()

    def _log():
        elapsed = time.perf_counter() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        print(f"  [{desc}] {completed}/{total} | {rate:.1f}/s | ETA {eta:.0f}s", flush=True)

    async def _one(idx: int):
        nonlocal completed
        is_hack, _raw = await runner.classify_one(questions[idx], actions[idx])
        results[idx] = is_hack
        completed += 1
        if completed % 50 == 0 or completed == total:
            _log()

    await asyncio.gather(*[_one(i) for i in range(total)])
    elapsed = time.perf_counter() - t_start
    print(f"  [{desc}] done in {elapsed:.1f}s ({total/elapsed:.1f} samples/s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def load_checkpoint(results_path: Path) -> dict:
    """Load existing results if any."""
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(results_path: Path, data: dict):
    """Atomically save results."""
    tmp = str(results_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(results_path))


async def evaluate_experiment(
    exp_name: str,
    step_dir: str,
    max_step: int,
    has_existing_monitor: bool,
    results_path: Path,
    test_mode: bool = False,
):
    """Evaluate one experiment across all steps."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    runner = MonitorRunner()
    results = load_checkpoint(results_path)

    steps_to_run = list(range(1, min(3, max_step) + 1) if test_mode else range(1, max_step + 1))

    # --- Pre-scan: count work ---
    n_need_with = 0
    n_need_without = 0
    for step in steps_to_run:
        step_key = str(step)
        if step_key not in results:
            if has_existing_monitor:
                n_need_without += 1
            else:
                n_need_with += 1
                n_need_without += 1
        else:
            if "with" not in results[step_key] and not has_existing_monitor:
                n_need_with += 1
            if "without" not in results[step_key]:
                n_need_without += 1

    total_api_batches = n_need_with + n_need_without
    print(f"  [{exp_name}] Steps: {steps_to_run[0]}-{steps_to_run[-1]}, "
          f"need_with={n_need_with}, need_without={n_need_without}, "
          f"total_batches={total_api_batches}")
    if total_api_batches == 0:
        print(f"  [{exp_name}] All done!")
        await runner.close()
        return results

    step_done = 0
    overall_t0 = time.perf_counter()

    for step in steps_to_run:
        step_key = str(step)

        # Check if already fully done
        if step_key in results and "with" in results[step_key] and "without" in results[step_key]:
            step_done += 1
            continue

        t0 = time.perf_counter()

        # Load data
        entries = load_step_entries(step_dir, step)
        gts = [get_ground_truth(e) for e in entries]
        n_hack = sum(gts)
        questions = [extract_question(e["input"]) for e in entries]
        actions_with = [strip_im_tokens(e["output"]) for e in entries]
        actions_without = [strip_think_tags(a) for a in actions_with]

        step_result = results.get(step_key, {})

        # --- with think ---
        if "with" not in step_result:
            if has_existing_monitor:
                preds = []
                for e in entries:
                    val = e.get("llm_monitor_is_hack")
                    preds.append(val if isinstance(val, bool) else None)
            else:
                print(f"  [{exp_name}] Step {step}/{steps_to_run[-1]} (with think)  | "
                      f"overall {step_done}/{len(steps_to_run)} done", flush=True)
                preds = await classify_batch(
                    runner, questions, actions_with,
                    desc=f"{exp_name}/step{step}/with",
                )
            step_result["with"] = confusion_matrix(preds, gts)
            step_result["with"]["n_hack"] = n_hack
            step_result["with"]["n_total"] = len(entries)
            step_result["with"]["excluded"] = sum(1 for p in preds if p is None)

        # --- without think ---
        if "without" not in step_result:
            print(f"  [{exp_name}] Step {step}/{steps_to_run[-1]} (without think) | "
                  f"overall {step_done}/{len(steps_to_run)} done", flush=True)
            preds = await classify_batch(
                runner, questions, actions_without,
                desc=f"{exp_name}/step{step}/without",
            )
            step_result["without"] = confusion_matrix(preds, gts)
            step_result["without"]["n_hack"] = n_hack
            step_result["without"]["n_total"] = len(entries)
            step_result["without"]["excluded"] = sum(1 for p in preds if p is None)

        # Compute deltas: with - without (positive = think tags help)
        step_result["delta"] = {
            "precision": round(step_result["with"]["precision"] - step_result["without"]["precision"], 6),
            "recall": round(step_result["with"]["recall"] - step_result["without"]["recall"], 6),
            "f1": round(step_result["with"]["f1"] - step_result["without"]["f1"], 6),
        }

        results[step_key] = step_result
        save_checkpoint(results_path, results)
        step_done += 1

        elapsed = time.perf_counter() - t0
        w = step_result["with"]
        wo = step_result["without"]
        d = step_result["delta"]
        overall_elapsed = time.perf_counter() - overall_t0
        print(
            f"  [{exp_name}] Step {step:>3d} ✓ | "
            f"hack={n_hack:>3d} | "
            f"with: P={w['precision']:.3f} R={w['recall']:.3f} F1={w['f1']:.3f} | "
            f"wo: P={wo['precision']:.3f} R={wo['recall']:.3f} F1={wo['f1']:.3f} | "
            f"ΔF1={d['f1']:+.3f} | "
            f"step={elapsed:.0f}s total={overall_elapsed:.0f}s",
            flush=True,
        )

    await runner.close()
    return results


async def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM monitor across training steps")
    parser.add_argument("--test", action="store_true", help="Quick test with 3 steps per experiment")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated compatibility flag; checkpoint resume is always automatic",
    )
    parser.add_argument("--exp", choices=["all", "exp1", "exp2"], default="all")
    parser.add_argument("--exp1-dir", default=EXP1_DIR)
    parser.add_argument("--exp2-dir", default=EXP2_DIR)
    parser.add_argument("--exp1-max-step", type=int, default=EXP1_MAX_STEP)
    parser.add_argument("--exp2-max-step", type=int, default=EXP2_MAX_STEP)
    parser.add_argument(
        "--exp1-results",
        type=Path,
        default=OUTPUT_DIR / "results_exp1.json",
    )
    parser.add_argument(
        "--exp2-results",
        type=Path,
        default=OUTPUT_DIR / "results_exp2.json",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LLM Monitor Step-by-Step Evaluation")
    print(f"  Test mode: {args.test}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    if args.exp in ("all", "exp1"):
        print("\n### Exp1 (no monitor penalty) ###")
        await evaluate_experiment(
            exp_name="exp1",
            step_dir=args.exp1_dir,
            max_step=args.exp1_max_step,
            has_existing_monitor=False,
            results_path=args.exp1_results,
            test_mode=args.test,
        )

    if args.exp in ("all", "exp2"):
        print("\n### Exp2 (with monitor penalty) ###")
        await evaluate_experiment(
            exp_name="exp2",
            step_dir=args.exp2_dir,
            max_step=args.exp2_max_step,
            has_existing_monitor=True,
            results_path=args.exp2_results,
            test_mode=args.test,
        )

    print("\nDone! Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
