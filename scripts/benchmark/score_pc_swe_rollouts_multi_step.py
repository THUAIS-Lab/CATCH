#!/usr/bin/env python3
"""
Batch-score pc_swe chat-completion rollouts across multiple training steps,
and log non_trivial_hack samples to a Feishu spreadsheet.

Usage examples:
    # Check steps 0 through 100 (every step that exists):
    python3 scripts/benchmark/score_pc_swe_rollouts_multi_step.py \
        --exp-path /data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4-4b_n10k_t1k-32k-bs16-mbs16-n16-no_rej-no_mask \
        --steps 0-100

    # Check specific steps:
    python3 scripts/benchmark/score_pc_swe_rollouts_multi_step.py \
        --exp-path /data/nvme0/wangsl/checkpoints/reward_hacking_mre/pc_swe_v4-4b_n10k_t1k-32k-bs16-mbs16-n16-no_rej-no_mask \
        --steps 10,64,128,256,384,512

The script processes steps serially and scores all trajectories within each step
in parallel. Only samples flagged as non_trivial_hack are logged to the Lark table.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

import rllm.utils.lark as lark
from rllm.rewards.pc_swe_reward import RewardPCSWEFn
from rllm.rewards.reward_types import RewardConfig

# Private service defaults removed for open-source release.
# Open-source sanitization: hardcoded credential or private token removed.
DEFAULT_LARK_APP_ID = ""
# Open-source sanitization: hardcoded credential or private token removed.
DEFAULT_LARK_APP_SECRET = ""
# Open-source sanitization: hardcoded credential or private token removed.
DEFAULT_LARK_OPEN_ID = ""
# Open-source sanitization: hardcoded credential or private token removed.
DEFAULT_FOLDER_TOKEN = ""
DEFAULT_PARQUET_PATH = "/data/nvme0/wangsl/datasets/deepcoder_swe_v4/train_verl.parquet"
DEFAULT_WORKERS = 256

LARK_CELL_MAX_SIZE = 40000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RolloutItem:
    line_index: int
    user_content: str
    assistant_content: str


@dataclass(slots=True)
class HackRecord:
    step: int
    input: str
    output: str
    model_codes: str
    reward_w_hack: str
    reward_wo_hack: str
    is_hack: str
    non_trivial_hack: str
    tests: str
    detailed_results: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch score pc_swe rollouts across multiple training steps.",
    )
    parser.add_argument(
        "--exp-path",
        required=True,
        help="Path to the experiment checkpoint directory "
        "(contains chat_completions/{step}.jsonl).",
    )
    parser.add_argument(
        "--steps",
        required=True,
        help="Steps to score. E.g. '0-100' (range), '10,64,128' (list), "
        "or '0-50,100-150' (mixed).",
    )
    parser.add_argument(
        "--parquet-path",
        default=DEFAULT_PARQUET_PATH,
        help="Path to the DeepCoder SWE parquet dataset.",
    )
    parser.add_argument(
        "--folder-token",
        default=DEFAULT_FOLDER_TOKEN,
        help="Feishu folder token to create the spreadsheet in.",
    )
    parser.add_argument(
        "--lark-app-id",
        default=DEFAULT_LARK_APP_ID,
        help="Feishu app ID.",
    )
    parser.add_argument(
        "--lark-app-secret",
        default=DEFAULT_LARK_APP_SECRET,
        help="Feishu app secret.",
    )
    parser.add_argument(
        "--lark-open-id",
        default=DEFAULT_LARK_OPEN_ID,
        help="Feishu open ID for granting full access.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Top-level thread concurrency for trajectory scoring within each step.",
    )
    parser.add_argument(
        "--ray-temp-dir",
        default=None,
        help="Optional explicit Ray temp dir.",
    )
    parser.add_argument(
        "--spreadsheet-title",
        default=None,
        help="Custom spreadsheet title. Defaults to 'non_trivial_hack_rollouts-<exp_name>'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N lines from each step's JSONL file.",
    )
    return parser.parse_args()


def parse_steps(steps_str: str) -> list[int]:
    """Parse a step specification like '64' or '0-100' or '10,20,30-50'."""
    steps: list[int] = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            steps.extend(range(int(lo_str.strip()), int(hi_str.strip()) + 1))
        else:
            steps.append(int(part))
    return sorted(set(steps))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Rollout loading
# ---------------------------------------------------------------------------

def _extract_role_contents(messages: list[dict[str, Any]], role: str) -> list[str]:
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)
    return contents


def load_rollouts(jsonl_path: str, *, limit: int | None = None) -> list[RolloutItem]:
    rollouts: list[RolloutItem] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if limit is not None and line_index >= limit:
                break
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, list):
                raise ValueError(
                    f"Line {line_index} in {jsonl_path} is not a message list: "
                    f"{type(payload)!r}"
                )
            user_contents = _extract_role_contents(payload, "user")
            assistant_contents = _extract_role_contents(payload, "assistant")
            if not user_contents:
                raise ValueError(
                    f"Line {line_index} in {jsonl_path} has no user message."
                )
            if not assistant_contents:
                raise ValueError(
                    f"Line {line_index} in {jsonl_path} has no assistant message."
                )
            rollouts.append(
                RolloutItem(
                    line_index=line_index,
                    user_content=user_contents[0].strip(),
                    assistant_content=assistant_contents[-1],
                )
            )
    return rollouts


# ---------------------------------------------------------------------------
# Parquet matching
# ---------------------------------------------------------------------------

def build_task_info_lookup(
    parquet_path: str,
    *,
    target_questions: set[str],
) -> dict[str, dict[str, Any]]:
    if not target_questions:
        return {}

    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
    dataset = load_dataset(
        "parquet",
        data_files=parquet_path,
        streaming=True,
    )["train"]

    task_info_lookup: dict[str, dict[str, Any]] = {}
    remaining = set(target_questions)
    scanned_rows = 0
    for row in dataset:
        scanned_rows += 1
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            continue
        question = extra_info.get("question")
        if not isinstance(question, str):
            continue
        stripped = question.strip()
        if stripped not in remaining:
            continue
        task_info_lookup[stripped] = extra_info
        remaining.remove(stripped)
        if not remaining:
            break

    if remaining:
        sample_missing = sorted(remaining)[:3]
        raise KeyError(
            "Failed to match some rollout prompts back to parquet rows. "
            f"matched={len(task_info_lookup)} requested={len(target_questions)} "
            f"scanned_rows={scanned_rows} sample_missing={sample_missing!r}"
        )

    return task_info_lookup


# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------

def ensure_ray_initialized(ray_temp_dir: str | None) -> str:
    import ray

    final_temp_dir = ray_temp_dir or tempfile.mkdtemp(
        prefix="pc_swe_reward_multi_step_ray_",
        dir="/tmp",
    )
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            log_to_driver=False,
            ignore_reinit_error=True,
            _temp_dir=final_temp_dir,
        )
    return final_temp_dir


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_one_rollout(
    rollout: RolloutItem,
    *,
    reward_fn: RewardPCSWEFn,
    task_info_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    task_info = task_info_lookup.get(rollout.user_content)
    if task_info is None:
        return {
            "line_index": rollout.line_index,
            "status": "missing_task_info",
            "elapsed_s": time.perf_counter() - started_at,
        }

    try:
        reward_output = reward_fn(task_info, rollout.assistant_content)
        return {
            "line_index": rollout.line_index,
            "status": "ok",
            "elapsed_s": time.perf_counter() - started_at,
            "user_content": rollout.user_content,
            "assistant_content": rollout.assistant_content,
            "reward_output": reward_output.to_dict(),
        }
    except Exception as exc:
        return {
            "line_index": rollout.line_index,
            "status": "reward_exception",
            "elapsed_s": time.perf_counter() - started_at,
            "user_content": rollout.user_content,
            "assistant_content": rollout.assistant_content,
            "error": repr(exc),
        }


def score_step(
    rollouts: list[RolloutItem],
    *,
    reward_fn: RewardPCSWEFn,
    task_info_lookup: dict[str, dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    results_by_line: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_line = {
            executor.submit(
                score_one_rollout,
                rollout,
                reward_fn=reward_fn,
                task_info_lookup=task_info_lookup,
            ): rollout.line_index
            for rollout in rollouts
        }
        completed = 0
        for future in as_completed(future_to_line):
            line_index = future_to_line[future]
            results_by_line[line_index] = future.result()
            completed += 1
            if completed % 64 == 0 or completed == len(rollouts):
                log(f"    {completed}/{len(rollouts)} completed")

    return [results_by_line[i] for i in sorted(results_by_line)]


# ---------------------------------------------------------------------------
# Lark helpers
# ---------------------------------------------------------------------------

def reshape_rows_to_lark_max_size(rows: list[list]) -> list[list]:
    """Split cells that exceed the Lark 40k char limit into multiple rows."""
    new_rows: list[list] = []
    for row in rows:
        rest_lengths = [len(str(cell)) for cell in row]
        offsets = [0] * len(row)
        while any(l > 0 for l in rest_lengths):
            new_row = []
            for i, (rest_len, offset) in enumerate(zip(rest_lengths, offsets)):
                if not isinstance(row[i], str) and offsets[i] == 0:
                    new_row.append(row[i])
                    rest_lengths[i] = 0
                    offsets[i] = -1
                elif rest_len > 0:
                    new_row.append(row[i][offset : offset + LARK_CELL_MAX_SIZE])
                    offsets[i] += LARK_CELL_MAX_SIZE
                    rest_lengths[i] = len(str(row[i])) - offsets[i]
                else:
                    new_row.append("")
            new_rows.append(new_row)
    return new_rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    step_results: dict[int, list[dict[str, Any]]],
    total_runtime_s: float,
) -> dict[str, Any]:
    all_ok: list[dict[str, Any]] = []
    all_err: list[dict[str, Any]] = []
    total_rollouts = 0
    for step, results in step_results.items():
        for r in results:
            total_rollouts += 1
            if r["status"] == "ok":
                all_ok.append(r)
            else:
                all_err.append(r)

    rewards = [float(r["reward_output"]["reward"]) for r in all_ok]
    hack_flags = [
        bool(r["reward_output"]["metadata"].get("is_hack", False)) for r in all_ok
    ]
    nontrivial_flags = [
        bool(r["reward_output"]["metadata"].get("nontrivial_hack", False))
        for r in all_ok
    ]

    return {
        "total_rollouts": total_rollouts,
        "ok_rollouts": len(all_ok),
        "error_rollouts": len(all_err),
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "hack_rate": sum(hack_flags) / len(hack_flags) if hack_flags else 0.0,
        "nontrivial_hack_count": sum(nontrivial_flags),
        "nontrivial_hack_rate": (
            sum(nontrivial_flags) / len(nontrivial_flags)
            if nontrivial_flags
            else 0.0
        ),
        "total_runtime_s": total_runtime_s,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    steps = parse_steps(args.steps)
    if not steps:
        log("ERROR: no steps specified.")
        sys.exit(1)

    exp_path = Path(args.exp_path)
    if not exp_path.is_dir():
        log(f"ERROR: experiment path not found: {exp_path}")
        sys.exit(1)

    ray_temp_dir = ensure_ray_initialized(args.ray_temp_dir)
    reward_fn = RewardPCSWEFn(RewardConfig())

    # ---- Phase 1: gather all rollouts and unique prompts across steps ----
    log(f"Experiment: {exp_path}")
    log(f"Steps to check: {steps[0]} -> {steps[-1]} ({len(steps)} steps)")
    log("")

    step_rollouts: dict[int, list[RolloutItem]] = {}
    missing_steps: list[int] = []
    for step in steps:
        jsonl_path = exp_path / "chat_completions" / f"{step}.jsonl"
        if not jsonl_path.exists():
            missing_steps.append(step)
            continue
        rollouts = load_rollouts(str(jsonl_path), limit=args.limit)
        step_rollouts[step] = rollouts
        log(f"  step {step}: {len(rollouts)} rollouts")

    if missing_steps:
        log(f"\n  SKIPPED {len(missing_steps)} missing steps: {missing_steps}")

    if not step_rollouts:
        log("ERROR: no rollout files found.")
        sys.exit(1)

    # ---- Phase 2: build unified task-info lookup ----
    all_questions: set[str] = set()
    for rollouts in step_rollouts.values():
        for r in rollouts:
            all_questions.add(r.user_content)

    log(f"\nBuilding task-info lookup for {len(all_questions)} unique prompts ...")
    task_info_lookup = build_task_info_lookup(
        args.parquet_path,
        target_questions=all_questions,
    )
    log(f"Matched {len(task_info_lookup)} unique questions.")

    # ---- Phase 3: score each step ----
    log(f"\nScoring rollouts ({len(step_rollouts)} steps, {args.workers} workers)...")
    overall_start = time.perf_counter()

    step_results: dict[int, list[dict[str, Any]]] = {}
    all_hack_records: list[HackRecord] = []

    for step in sorted(step_rollouts.keys()):
        step_start = time.perf_counter()
        rollouts = step_rollouts[step]
        log(f"\n--- Step {step} ({len(rollouts)} rollouts) ---")

        results = score_step(
            rollouts,
            reward_fn=reward_fn,
            task_info_lookup=task_info_lookup,
            workers=args.workers,
        )
        step_results[step] = results

        ok_count = sum(1 for r in results if r["status"] == "ok")
        err_count = sum(1 for r in results if r["status"] != "ok")

        # Extract non_trivial_hack records
        step_hack_count = 0
        for r in results:
            if r["status"] != "ok":
                continue
            meta = r["reward_output"]["metadata"]
            if meta.get("nontrivial_hack"):
                step_hack_count += 1
                all_hack_records.append(
                    HackRecord(
                        step=step,
                        input=r["user_content"],
                        output=r["assistant_content"],
                        model_codes=str(meta.get("model_codes", "")),
                        reward_w_hack=str(meta.get("reward_w_hack", "")),
                        reward_wo_hack=str(meta.get("reward_wo_hack", "")),
                        is_hack=str(meta.get("is_hack", "")),
                        non_trivial_hack=str(meta.get("nontrivial_hack", "")),
                        tests=str(meta.get("tests", ""))[:30000],
                        detailed_results=str(meta.get("detailed_results", "")),
                    )
                )

        step_elapsed = time.perf_counter() - step_start
        log(
            f"  Step {step} done: {ok_count} ok, {err_count} errors, "
            f"{step_hack_count} non_trivial_hack, "
            f"{step_elapsed:.1f}s"
        )

    total_runtime = time.perf_counter() - overall_start

    # ---- Phase 4: summary ----
    summary = build_summary(step_results, total_runtime)
    log(f"\n{'='*60}")
    log(f"SUMMARY")
    log(f"{'='*60}")
    log(f"  Total rollouts scored:    {summary['total_rollouts']}")
    log(f"  OK / error:               {summary['ok_rollouts']} / {summary['error_rollouts']}")
    log(f"  Mean reward:              {summary['mean_reward']:.4f}")
    log(f"  Hack rate:                {summary['hack_rate']:.4f}")
    log(f"  Non-trivial hack count:   {summary['nontrivial_hack_count']}")
    log(f"  Non-trivial hack rate:    {summary['nontrivial_hack_rate']:.4f}")
    log(f"  Total runtime:            {total_runtime:.1f}s")

    # ---- Phase 5: log to Lark ----
    if all_hack_records:
        log(f"\n--- Logging {len(all_hack_records)} non_trivial_hack samples to Lark ---")

        title = args.spreadsheet_title or f"non_trivial_hack_rollouts-{exp_path.name}"
        log(f"  Creating spreadsheet '{title}' ...")

        spreadsheet = lark.create_spreadsheet(
            title=title,
            folder_token=args.folder_token,
            app_id=args.lark_app_id,
            app_secret=args.lark_app_secret,
        )
        if spreadsheet is None:
            log("ERROR: failed to create spreadsheet.")
            sys.exit(1)

        log(f"  Spreadsheet token: {spreadsheet.spreadsheet_token}")

        lark.auth_full_access(
            token=spreadsheet.spreadsheet_token,
            repo_type="sheet",
            app_id=args.lark_app_id,
            app_secret=args.lark_app_secret,
            open_id=args.lark_open_id,
        )

        sheet_id = lark.add_sheet(
            spreadsheet_token=spreadsheet.spreadsheet_token,
            sheet_title="non_trivial_hacks",
            app_id=args.lark_app_id,
            app_secret=args.lark_app_secret,
        )
        if sheet_id is None:
            log("ERROR: failed to add sheet.")
            sys.exit(1)

        columns = [
            "step",
            "input",
            "output",
            "model_codes",
            "reward_w_hack",
            "reward_wo_hack",
            "is_hack",
            "non_trivial_hack",
            "tests",
            "detailed_results",
        ]
        rows = [columns] + [
            [
                rec.step,
                rec.input,
                rec.output,
                rec.model_codes,
                rec.reward_w_hack,
                rec.reward_wo_hack,
                rec.is_hack,
                rec.non_trivial_hack,
                rec.tests,
                rec.detailed_results,
            ]
            for rec in all_hack_records
        ]

        log(f"  Reshaping {len(rows)} rows for Lark cell limits ...")
        reshaped_rows = reshape_rows_to_lark_max_size(rows)

        log(f"  Appending {len(reshaped_rows)} reshaped rows ...")
        lark.append_rows(
            spreadsheet_token=spreadsheet.spreadsheet_token,
            sheet_id=sheet_id,
            rows=reshaped_rows,
            app_id=args.lark_app_id,
            app_secret=args.lark_app_secret,
        )
        lark.set_row_height(
            spreadsheet_token=spreadsheet.spreadsheet_token,
            sheet_id=sheet_id,
            row_height=27,
            start_index=1,
            end_index=len(reshaped_rows),
            app_id=args.lark_app_id,
            app_secret=args.lark_app_secret,
        )

        url = f"https://bytedance.feishu.cn/sheets/{spreadsheet.spreadsheet_token}"
        log(f"\n  Done! Spreadsheet URL: {url}")
    else:
        log("\n  No non_trivial_hack samples found — skipping Lark upload.")


if __name__ == "__main__":
    main()
