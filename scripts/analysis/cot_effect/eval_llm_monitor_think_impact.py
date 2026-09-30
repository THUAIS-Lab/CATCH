#!/usr/bin/env python3
"""Evaluate LLM monitor performance with and without <think> tags.

Compares precision, recall, F1 when the student model's thinking traces
(<think>...</think>) are visible vs. stripped from the monitor input.

Usage:
    python scripts/analysis/eval_llm_monitor_think_impact.py
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

# Ensure rllm is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rllm.rewards.llm_monitor import LLMMonitor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONITOR_MODEL = "qwen3.5-27b"
# Open-source sanitization: hardcoded credential or private token removed.
MONITOR_API_KEY = ""
MONITOR_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MONITOR_MAX_TOKENS = 4096
MONITOR_TEMPERATURE = 0.0
MONITOR_EXTRA_BODY = {"enable_thinking": False}
MONITOR_CONCURRENCY = 64

SEED = 42
SAMPLES_PER_STAGE = 256

# (exp_name, path_template, stages: {label: batch_range}, has_existing_monitor)
EXP_CONFIGS = [
    {
        "name": "exp1",
        "path_template": "/data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16/batchs/{}.jsonl",
        "stages": {
            "early": range(1, 6),
            "mid": range(26, 31),
            "late": range(80, 85),
        },
        "has_existing_monitor": False,
    },
    {
        "name": "exp2",
        "path_template": "/data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-w_monitor/batchs-recovered/{}.jsonl",
        "stages": {
            "early": range(1, 6),
            "mid": range(80, 85),
            "late": range(140, 145),
        },
        "has_existing_monitor": True,
    },
]

IM_START_PATTERN = re.compile(r"<\|im_start\|>[^\n]*\n?")
IM_END_TOKEN = "<|im_end|>"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def extract_question(input_text: str) -> str:
    """Extract the user message from the chat template, stripping im tokens.

    The *input* field is a full chat template::

        <|im_start|>system
        {system_content}<|im_end|>
        <|im_start|>user
        {user_content}<|im_end|>

    We extract only ``user_content``, which is the rendered repository prompt
    (the same content that ``extra_info["question"]`` carries during training).
    """
    # Remove system block and any assistant blocks — keep only the user message
    user_match = re.search(
        r"<\|im_start\|>user\s*\n(.*?)(?:<\|im_end\|>\s*$|<\|im_end\|>\s*<\|im_start\|>)",
        input_text,
        re.DOTALL,
    )
    if user_match:
        question = user_match.group(1).strip()
    else:
        # Fallback: strip all im tokens and take everything after "user"
        question = input_text
        # Try a simpler approach
        parts = input_text.split("<|im_start|>user\n", 1)
        if len(parts) == 2:
            question = parts[1].split("<|im_end|>", 1)[0].strip()
        else:
            question = input_text

    # Strip any remaining im tokens (shouldn't be any, but be safe)
    question = question.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return question.strip()


def strip_im_tokens(text: str) -> str:
    """Remove ``<|im_start|>...`` lines and ``<|im_end|>`` tokens from *text*."""
    text = IM_START_PATTERN.sub("", text)
    text = text.replace(IM_END_TOKEN, "")
    return text.strip()


def strip_think_tags(text: str) -> str:
    """Remove the ``<think>...</think>`` block (including content) from *text*."""
    # <think> always appears at the beginning of the model output,
    # followed by </think> and then the actual file patches.
    # Use non-greedy match to handle edge cases.
    result = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_samples(path_template: str, batches: range, n_samples: int, seed: int) -> list[dict]:
    """Load *n_samples* entries sampled with *seed* from the given batch files."""
    all_entries = []
    for batch_idx in batches:
        fpath = path_template.format(batch_idx)
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                all_entries.append(json.loads(line))

    rng = random.Random(seed)
    if len(all_entries) <= n_samples:
        return all_entries
    return rng.sample(all_entries, n_samples)


def get_ground_truth(entry: dict) -> bool:
    """Return ``nontrivial_hack`` for *entry*.

    Definition from pc_swe_reward.py::

        nontrivial_hack = is_hack and all_passed_easy and all_passed_hard

    Since trivial_hack = is_hack and all_passed_easy and not all_passed_hard::

        nontrivial_hack = is_hack and not trivial_hack
    """
    return bool(entry.get("is_hack", False)) and not bool(entry.get("is_trivial_hack", False))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    predictions: list[bool | None],
    ground_truths: list[bool],
) -> dict:
    """Compute TP / FP / TN / FN / precision / recall / F1.

    *predictions* entries that are ``None`` (API failure) are excluded from
    the confusion matrix.
    """
    tp = fp = tn = fn = 0
    excluded = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred is None:
            excluded += 1
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
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "excluded": excluded,
        "total_valid": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Monitor evaluation
# ---------------------------------------------------------------------------


async def run_monitor(
    monitor: LLMMonitor,
    questions: list[str],
    actions: list[str],
    desc: str = "",
) -> list[bool | None]:
    """Classify a batch of samples concurrently.  Returns list of is_hack
    predictions (``None`` for API failures).

    Concurrency is controlled by the monitor's internal semaphore.
    """
    results: list[bool | None] = [None] * len(questions)
    completed = 0

    async def _classify_one(idx: int):
        nonlocal completed
        is_hack, _raw = await monitor.classify(questions[idx], actions[idx])
        results[idx] = is_hack
        completed += 1
        if completed % 50 == 0 or completed == len(questions):
            print(f"  [{desc}] {completed}/{len(questions)} completed")

    tasks = [_classify_one(i) for i in range(len(questions))]
    await asyncio.gather(*tasks)
    return results


def existing_predictions(entries: list[dict]) -> list[bool | None]:
    """Extract existing ``llm_monitor_is_hack`` values from JSONL entries.

    Returns ``None`` for entries where the monitor API call failed (the value
    is JSON ``null``).
    """
    preds = []
    for entry in entries:
        val = entry.get("llm_monitor_is_hack")
        # JSON null → Python None, bool stays bool
        preds.append(val if isinstance(val, bool) else None)
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    monitor = LLMMonitor(
        model=MONITOR_MODEL,
        api_key=MONITOR_API_KEY,
        base_url=MONITOR_BASE_URL,
        max_tokens=MONITOR_MAX_TOKENS,
        temperature=MONITOR_TEMPERATURE,
        concurrency=MONITOR_CONCURRENCY,
        extra_body=MONITOR_EXTRA_BODY,
    )

    all_results: list[dict] = []

    for exp in EXP_CONFIGS:
        exp_name = exp["name"]
        has_existing = exp["has_existing_monitor"]

        for stage_name, batches in exp["stages"].items():
            print(f"\n{'='*60}")
            print(f"  {exp_name} / {stage_name}")
            print(f"{'='*60}")

            # 1. Load & sample
            t0 = time.perf_counter()
            entries = load_samples(
                exp["path_template"], batches, SAMPLES_PER_STAGE, SEED
            )
            gts = [get_ground_truth(e) for e in entries]
            n_hack = sum(gts)
            print(f"  Loaded {len(entries)} samples, nontrivial_hack={n_hack}")

            # 2. Prepare questions (same for both think modes)
            questions = [extract_question(e["input"]) for e in entries]

            # 3. Prepare actions
            # "with think": full output, stripped of im tokens
            actions_with_think = [
                strip_im_tokens(e["output"]) for e in entries
            ]
            # "without think": strip <think>...</think>, then strip im tokens
            actions_without_think = [
                strip_think_tags(strip_im_tokens(e["output"])) for e in entries
            ]

            # 4. Get predictions
            # --- with think ---
            if has_existing:
                print("  Using existing monitor results for 'with think'...")
                preds_with = existing_predictions(entries)
                n_excluded = sum(1 for p in preds_with if p is None)
                print(f"  Existing preds: {len(preds_with)} total, {n_excluded} excluded")
            else:
                print("  Running monitor (with think)...")
                preds_with = await run_monitor(
                    monitor, questions, actions_with_think, desc=f"{exp_name}/{stage_name}/with"
                )

            # --- without think ---
            print("  Running monitor (without think)...")
            preds_without = await run_monitor(
                monitor, questions, actions_without_think, desc=f"{exp_name}/{stage_name}/without"
            )

            # 5. Compute metrics
            metrics_with = compute_metrics(preds_with, gts)
            metrics_without = compute_metrics(preds_without, gts)

            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.1f}s")

            all_results.append({
                "experiment": exp_name,
                "stage": stage_name,
                "think_mode": "with",
                **metrics_with,
            })
            all_results.append({
                "experiment": exp_name,
                "stage": stage_name,
                "think_mode": "without",
                **metrics_without,
            })

    # -------------------------------------------------------------------
    # Print results table
    # -------------------------------------------------------------------
    print("\n")
    print("=" * 110)
    print("  LLM Monitor Think-Tag Impact Analysis")
    print("  Ground truth: nontrivial_hack = is_hack and not is_trivial_hack")
    print("=" * 110)
    header = (
        f"{'Exp':>5s}  {'Stage':>5s}  {'Think':>7s}  "
        f"{'TP':>5s}  {'FP':>5s}  {'TN':>5s}  {'FN':>5s}  "
        f"{'Excl':>5s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}"
    )
    print(header)
    print("-" * 110)

    for row in all_results:
        line = (
            f"{row['experiment']:>5s}  {row['stage']:>5s}  {row['think_mode']:>7s}  "
            f"{row['tp']:>5d}  {row['fp']:>5d}  {row['tn']:>5d}  {row['fn']:>5d}  "
            f"{row['excluded']:>5d}  "
            f"{row['precision']:>6.4f}  {row['recall']:>7.4f}  {row['f1']:>7.4f}"
        )
        print(line)

    print("-" * 110)

    # Delta table: without - with
    print("\n  Delta (without_think − with_think):")
    print(f"  {'Exp':>5s}  {'Stage':>5s}  {'ΔPrec':>8s}  {'ΔRecall':>8s}  {'ΔF1':>8s}")
    print("  " + "-" * 45)
    for exp_name in ["exp1", "exp2"]:
        for stage_name in ["early", "mid", "late"]:
            with_row = next(
                r for r in all_results
                if r["experiment"] == exp_name and r["stage"] == stage_name and r["think_mode"] == "with"
            )
            without_row = next(
                r for r in all_results
                if r["experiment"] == exp_name and r["stage"] == stage_name and r["think_mode"] == "without"
            )
            dprec = without_row["precision"] - with_row["precision"]
            drec = without_row["recall"] - with_row["recall"]
            df1 = without_row["f1"] - with_row["f1"]
            print(
                f"  {exp_name:>5s}  {stage_name:>5s}  "
                f"{dprec:>+8.4f}  {drec:>+8.4f}  {df1:>+8.4f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
