#!/usr/bin/env python3
"""Evaluate LLM monitor sensitivity to Python code comments.

For each step, runs the monitor on model outputs with .py file comments
stripped, then compares against the existing "with think" results (which
serve as the "with comments" baseline).

delta = with_comments - without_comments  (positive = comments help detection)

Usage:
    python scripts/analysis/annotation_effect/eval_monitor_annotation_effect.py --test
    python scripts/analysis/annotation_effect/eval_monitor_annotation_effect.py

    # A custom no-penalty run (resume is automatic)
    python scripts/analysis/annotation_effect/eval_monitor_annotation_effect.py \
        --exp exp1 --exp1-dir /path/to/batchs --exp1-max-step 140 \
        --exp1-results /path/to/results_exp1_r2.json \
        --cot-exp1-results /path/to/cot/results_exp1_r2.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
COT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "cot_effect" / "monitor_step_results"
EXP1_DIR = "/data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16/batchs"
EXP2_DIR = "/data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-w_monitor/batchs-recovered"

EXP1_MAX_STEP = 80
EXP2_MAX_STEP = 140

# ---------------------------------------------------------------------------
# Monitor config
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
    parts = input_text.split("<|im_start|>user\n", 1)
    if len(parts) == 2:
        question = parts[1].split("<|im_end|>", 1)[0].strip()
    else:
        question = input_text
    question = question.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return question.strip()


def strip_im_tokens(text: str) -> str:
    text = re.sub(r"<\|im_start\|>[^\n]*\n?", "", text)
    text = text.replace("<|im_end|>", "")
    return text.strip()


# ---------------------------------------------------------------------------
# Python comment stripping
# ---------------------------------------------------------------------------


class _DocstringStripper(ast.NodeTransformer):
    """Remove the first string expression from Module/FunctionDef/ClassDef bodies."""

    @staticmethod
    def _strip_body(body):
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            return body[1:]
        return body

    def visit_Module(self, node):
        node.body = self._strip_body(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.body = self._strip_body(node.body)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        node.body = self._strip_body(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        node.body = self._strip_body(node.body)
        return self.generic_visit(node)


def strip_python_comments(code: str) -> str:
    """Remove # comments and docstrings from Python source.

    Uses ``ast.parse`` / ``ast.unparse``: the parser drops ``#`` comments
    naturally, and ``_DocstringStripper`` removes the first string-constant
    expression from module / function / class bodies.
    """
    try:
        tree = ast.parse(code)
        stripped = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(stripped)
        return ast.unparse(stripped)
    except SyntaxError:
        return _fallback_strip(code)


def _fallback_strip(code: str) -> str:
    """Regex-based ``#`` comment removal (string-aware fallback)."""
    lines = code.split("\n")
    result = []
    in_triple_single = False
    in_triple_double = False
    for line in lines:
        sc = line.count("'''") - line.count("\\'\\'\\'")
        dc = line.count('"""') - line.count('\\"\\"\\"')
        if sc % 2 == 1:
            in_triple_single = not in_triple_single
        if dc % 2 == 1:
            in_triple_double = not in_triple_double
        if in_triple_single or in_triple_double:
            result.append(line)
            continue
        result.append(_remove_inline_comment(line))
    return "\n".join(result)


def _remove_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line


_FILE_BLOCK_RE = re.compile(
    r'(<file\s+path="(.*?)">)\s*\n(.*?)\n(\s*</file>)',
    re.DOTALL,
)


def strip_comments_from_output(output: str) -> str:
    """Strip comments only from ``.py`` file blocks in *output*."""

    def _replace(match):
        prefix = match.group(1)
        path = match.group(2)
        code = match.group(3)
        suffix = match.group(4)
        if path.endswith(".py"):
            code = strip_python_comments(code)
        return f"{prefix}\n{code}\n{suffix}"

    return _FILE_BLOCK_RE.sub(_replace, output)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_matrix(predictions: list[bool | None], gts: list[bool]) -> dict:
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
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total_valid": total,
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6)}


def get_ground_truth(entry: dict) -> bool:
    return bool(entry.get("is_hack", False)) and not bool(entry.get("is_trivial_hack", False))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_step_entries(step_dir: str, step: int) -> list[dict]:
    fpath = os.path.join(step_dir, f"{step}.jsonl")
    entries = []
    with open(fpath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Monitor runner (shared client)
# ---------------------------------------------------------------------------


def _backoff_seconds(attempt: int) -> float:
    """Capped exponential backoff with equal jitter."""
    delay = min(MONITOR_BACKOFF_BASE * (2 ** attempt), MONITOR_BACKOFF_CAP)
    return delay / 2 + random.uniform(0, delay / 2)


class MonitorRunner:
    def __init__(self):
        self._client = AsyncOpenAI(api_key=MONITOR_API_KEY, base_url=MONITOR_BASE_URL, timeout=MONITOR_TIMEOUT)
        self._semaphore = asyncio.Semaphore(MONITOR_CONCURRENCY)

    async def classify_one(self, question: str, action: str) -> tuple[bool | None, str]:
        user_prompt = OPENAI_MONITOR_USER_TEMPLATE.format(
            normalized_input=question.strip(), normalized_output=action.strip())
        async with self._semaphore:
            last_exc = ""
            for attempt in range(MONITOR_MAX_RETRIES):
                try:
                    completion = await self._client.chat.completions.create(
                        model=MONITOR_MODEL,
                        messages=[{"role": "system", "content": LLM_MONITOR_SYSTEM_PROMPT},
                                  {"role": "user", "content": user_prompt}],
                        temperature=MONITOR_TEMPERATURE, max_tokens=MONITOR_MAX_TOKENS,
                        extra_body=MONITOR_EXTRA_BODY)
                    content = completion.choices[0].message.content or "" if completion.choices else ""
                    return _extract_is_hack(content), content
                except Exception as exc:
                    last_exc = repr(exc)
                    if attempt < MONITOR_MAX_RETRIES - 1:
                        await asyncio.sleep(_backoff_seconds(attempt))
            return None, last_exc

    async def close(self):
        await self._client.close()


async def classify_batch(runner: MonitorRunner, questions: list[str], actions: list[str],
                         desc: str = "") -> list[bool | None]:
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
        is_hack, _ = await runner.classify_one(questions[idx], actions[idx])
        results[idx] = is_hack
        completed += 1
        if completed % 50 == 0 or completed == total:
            _log()

    await asyncio.gather(*[_one(i) for i in range(total)])
    elapsed = time.perf_counter() - t_start
    print(f"  [{desc}] done in {elapsed:.1f}s ({total / elapsed:.1f} samples/s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def evaluate_experiment(
    exp_name: str,
    step_dir: str,
    max_step: int,
    results_path: Path,
    cot_results_path: Path,
    test_mode: bool = False,
):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_checkpoint(results_path)
    cot_results = load_checkpoint(cot_results_path)

    steps_to_run = list(range(1, min(3, max_step) + 1) if test_mode else range(1, max_step + 1))

    # Pre-scan
    pending_steps = [s for s in steps_to_run if str(s) not in results]
    n_todo = len(pending_steps)
    print(f"  [{exp_name}] Steps: {steps_to_run[0]}-{steps_to_run[-1]}, need_without_comments={n_todo}")
    if n_todo == 0:
        print(f"  [{exp_name}] All done!")
        return results

    missing_cot_steps = [
        step
        for step in pending_steps
        if not cot_results.get(str(step), {}).get("with")
    ]
    if missing_cot_steps:
        preview = ", ".join(map(str, missing_cot_steps[:10]))
        suffix = "..." if len(missing_cot_steps) > 10 else ""
        raise RuntimeError(
            f"Missing CoT 'with' baseline for {len(missing_cot_steps)} step(s) "
            f"in {cot_results_path}: {preview}{suffix}"
        )

    runner = MonitorRunner()

    step_done = 0
    overall_t0 = time.perf_counter()

    for step in steps_to_run:
        sk = str(step)
        if sk in results:
            step_done += 1
            continue

        t0 = time.perf_counter()
        entries = load_step_entries(step_dir, step)
        gts = [get_ground_truth(e) for e in entries]
        n_hack = sum(gts)
        questions = [extract_question(e["input"]) for e in entries]
        actions_without_comments = [strip_comments_from_output(strip_im_tokens(e["output"])) for e in entries]

        # --- without comments (run monitor) ---
        print(f"  [{exp_name}] Step {step}/{steps_to_run[-1]} (without comments) | "
              f"overall {step_done}/{len(steps_to_run)} done", flush=True)
        preds = await classify_batch(runner, questions, actions_without_comments,
                                     desc=f"{exp_name}/step{step}/nocomment")
        wo = confusion_matrix(preds, gts)
        wo["n_hack"] = n_hack
        wo["n_total"] = len(entries)
        wo["excluded"] = sum(1 for p in preds if p is None)

        # --- with comments (reuse cot_effect "with think") ---
        w = cot_results[sk]["with"]

        # Delta: with_comments - without_comments
        delta = {
            "precision": round(w["precision"] - wo["precision"], 6),
            "recall": round(w["recall"] - wo["recall"], 6),
            "f1": round(w["f1"] - wo["f1"], 6),
        }

        step_result = {"with": w, "without": wo, "delta": delta}
        results[sk] = step_result
        save_checkpoint(results_path, results)
        step_done += 1

        elapsed = time.perf_counter() - t0
        overall_elapsed = time.perf_counter() - overall_t0
        print(
            f"  [{exp_name}] Step {step:>3d} ✓ | hack={n_hack:>3d} | "
            f"with: P={w['precision']:.3f} R={w['recall']:.3f} F1={w['f1']:.3f} | "
            f"wo:  P={wo['precision']:.3f} R={wo['recall']:.3f} F1={wo['f1']:.3f} | "
            f"ΔF1={delta['f1']:+.3f} | step={elapsed:.0f}s total={overall_elapsed:.0f}s",
            flush=True,
        )

    await runner.close()
    return results


async def main():
    parser = argparse.ArgumentParser(description="Evaluate annotation effect across training steps")
    parser.add_argument("--test", action="store_true", help="Quick test with 3 steps per experiment")
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
    parser.add_argument(
        "--cot-exp1-results",
        type=Path,
        default=COT_RESULTS_DIR / "results_exp1.json",
    )
    parser.add_argument(
        "--cot-exp2-results",
        type=Path,
        default=COT_RESULTS_DIR / "results_exp2.json",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("LLM Monitor — Annotation Effect (comments in .py code)")
    print(f"  Test mode: {args.test}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    if args.exp in ("all", "exp1"):
        print("\n### Exp1 (no monitor penalty) ###")
        await evaluate_experiment(
            "exp1",
            args.exp1_dir,
            args.exp1_max_step,
            args.exp1_results,
            args.cot_exp1_results,
            test_mode=args.test,
        )

    if args.exp in ("all", "exp2"):
        print("\n### Exp2 (with monitor penalty) ###")
        await evaluate_experiment(
            "exp2",
            args.exp2_dir,
            args.exp2_max_step,
            args.exp2_results,
            args.cot_exp2_results,
            test_mode=args.test,
        )

    print("\nDone! Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
