#!/usr/bin/env python3
"""Batch-score pc_swe chat-completion rollouts against the active reward path.

This utility is intentionally thin: it matches each rollout's stripped user
prompt against `row["extra_info"]["question"].strip()` from the parquet data,
then feeds the matched `extra_info` plus the rollout assistant content into
`RewardPCSWEFn`.

The top-level scoring loop uses a 256-thread pool by default, as requested.
The reward implementation itself still keeps its own internal concurrency for
hidden-baseline checks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rllm.rewards.pc_swe_reward import RewardPCSWEFn
from rllm.rewards.reward_types import RewardConfig


DEFAULT_PARQUET_PATH = "/data/wangsl/datasets/deepcoder_swe_v3/train_verl.parquet"
DEFAULT_JSONL_PATH = (
    "/data/wangsl/checkpoints/reward_hacking_mre/"
    "pc_swe-q3_4b-32k-bs16-mbs16-n16-no_rej-no_mask/chat_completions/8.jsonl"
)
DEFAULT_OUTPUT_PATH = "/tmp/pc_swe_reward_scores.jsonl"
DEFAULT_SUMMARY_PATH = "/tmp/pc_swe_reward_scores.summary.json"
DEFAULT_WORKERS = 256
DEFAULT_PROGRESS_INTERVAL = 16


@dataclass(slots=True)
class RolloutItem:
    line_index: int
    user_content: str
    assistant_content: str
    user_message_count: int
    assistant_message_count: int


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch score pc_swe rollouts with RewardPCSWEFn.",
    )
    parser.add_argument(
        "--parquet-path",
        default=DEFAULT_PARQUET_PATH,
        help="Path to the DeepCoder SWE parquet dataset.",
    )
    parser.add_argument(
        "--jsonl-path",
        default=DEFAULT_JSONL_PATH,
        help="Path to the rollout chat-completions jsonl.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write per-rollout jsonl results.",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="Where to write the aggregate summary json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Top-level thread concurrency for trajectory scoring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rollout lines to score.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print progress every N completed rollouts.",
    )
    parser.add_argument(
        "--ray-temp-dir",
        default=None,
        help="Optional explicit Ray temp dir. Defaults to a fresh /tmp dir.",
    )
    return parser.parse_args()


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
            if line_index < 256:
                continue
            if limit is not None and len(rollouts) >= limit:
                break
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, list):
                raise ValueError(
                    f"Line {line_index} is not a message list: {type(payload)!r}"
                )
            user_contents = _extract_role_contents(payload, "user")
            assistant_contents = _extract_role_contents(payload, "assistant")
            if not user_contents:
                raise ValueError(f"Line {line_index} does not contain a user message.")
            if not assistant_contents:
                raise ValueError(
                    f"Line {line_index} does not contain an assistant message."
                )
            rollouts.append(
                RolloutItem(
                    line_index=line_index,
                    user_content=user_contents[0].strip(),
                    assistant_content=assistant_contents[-1],
                    user_message_count=len(user_contents),
                    assistant_message_count=len(assistant_contents),
                )
            )
    return rollouts


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
    remaining_questions = set(target_questions)
    scanned_rows = 0
    for row in dataset:
        scanned_rows += 1
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            continue
        question = extra_info.get("question")
        if not isinstance(question, str):
            continue
        stripped_question = question.strip()
        if stripped_question not in remaining_questions:
            continue
        task_info_lookup[stripped_question] = extra_info
        remaining_questions.remove(stripped_question)
        if not remaining_questions:
            break

    if remaining_questions:
        sample_missing = sorted(remaining_questions)[:3]
        raise KeyError(
            "Failed to match some rollout prompts back to parquet rows. "
            f"matched={len(task_info_lookup)} requested={len(target_questions)} "
            f"scanned_rows={scanned_rows} sample_missing={sample_missing!r}"
        )

    return task_info_lookup


def ensure_ray_initialized(ray_temp_dir: str | None) -> str:
    import ray

    final_temp_dir = ray_temp_dir or tempfile.mkdtemp(
        prefix="pc_swe_reward_batch_ray_",
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


def silence_reward_debug_prints() -> None:
    import rllm.rewards.pc_swe_reward as pc_swe_reward_module

    # The reward path currently prints per-chunk debug lines unconditionally.
    # Silence them here so 256 top-level workers do not spam stdout.
    pc_swe_reward_module.print = lambda *args, **kwargs: None


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
            "user_message_count": rollout.user_message_count,
            "assistant_message_count": rollout.assistant_message_count,
        }

    try:
        reward_output = reward_fn(task_info, rollout.assistant_content)
        result = {
            "line_index": rollout.line_index,
            "status": "ok",
            "elapsed_s": time.perf_counter() - started_at,
            "user_message_count": rollout.user_message_count,
            "assistant_message_count": rollout.assistant_message_count,
            "matched_uid": task_info.get("uid"),
            "matched_index": task_info.get("index"),
            "reward_output": reward_output.to_dict(),
            "task_info": task_info,
        }
        return result
    except Exception as exc:
        return {
            "line_index": rollout.line_index,
            "status": "reward_exception",
            "elapsed_s": time.perf_counter() - started_at,
            "user_message_count": rollout.user_message_count,
            "assistant_message_count": rollout.assistant_message_count,
            "matched_uid": task_info.get("uid"),
            "matched_index": task_info.get("index"),
            "error": repr(exc),
            "task_info": task_info,
        }


def build_summary(
    results: list[dict[str, Any]],
    *,
    parquet_path: str,
    jsonl_path: str,
    output_path: str,
    workers: int,
    total_runtime_s: float,
    matched_question_count: int,
) -> dict[str, Any]:
    ok_results = [result for result in results if result["status"] == "ok"]
    errored_results = [result for result in results if result["status"] != "ok"]
    rewards = [
        float(result["reward_output"]["reward"])
        for result in ok_results
    ]
    correct_flags = [
        bool(result["reward_output"].get("is_correct"))
        for result in ok_results
    ]
    hack_flags = [
        bool(result["reward_output"]["metadata"].get("is_hack", False))
        for result in ok_results
    ]
    elapsed_values = [float(result["elapsed_s"]) for result in results]

    return {
        "parquet_path": parquet_path,
        "jsonl_path": jsonl_path,
        "output_path": output_path,
        "workers": workers,
        "total_rollouts": len(results),
        "ok_rollouts": len(ok_results),
        "error_rollouts": len(errored_results),
        "matched_question_count": matched_question_count,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "mean_elapsed_s": statistics.fmean(elapsed_values) if elapsed_values else 0.0,
        "max_elapsed_s": max(elapsed_values) if elapsed_values else 0.0,
        "min_elapsed_s": min(elapsed_values) if elapsed_values else 0.0,
        "visible_pass_rate": (
            sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
        ),
        "hack_rate": sum(hack_flags) / len(hack_flags) if hack_flags else 0.0,
        "total_runtime_s": total_runtime_s,
    }


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    # silence_reward_debug_prints()
    ray_temp_dir = ensure_ray_initialized(args.ray_temp_dir)
    reward_fn = RewardPCSWEFn(RewardConfig())

    log(f"Loading rollouts from {args.jsonl_path}")
    rollouts = load_rollouts(args.jsonl_path, limit=args.limit)
    log(f"Loaded {len(rollouts)} rollout lines")

    needed_questions = {rollout.user_content for rollout in rollouts}
    log(
        "Matching rollout prompts to parquet rows "
        f"(unique questions={len(needed_questions)})"
    )
    task_info_lookup = build_task_info_lookup(
        args.parquet_path,
        target_questions=needed_questions,
    )
    log(f"Matched {len(task_info_lookup)} unique questions")

    started_at = time.perf_counter()
    results_by_line: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
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
            if (
                completed == len(rollouts)
                or completed % max(1, args.progress_interval) == 0
            ):
                log(f"Completed {completed}/{len(rollouts)} rollouts")

    total_runtime_s = time.perf_counter() - started_at
    ordered_results = [
        results_by_line[line_index]
        for line_index in sorted(results_by_line)
    ]
    summary = build_summary(
        ordered_results,
        parquet_path=args.parquet_path,
        jsonl_path=args.jsonl_path,
        output_path=args.output_path,
        workers=args.workers,
        total_runtime_s=total_runtime_s,
        matched_question_count=len(task_info_lookup),
    )
    summary["ray_temp_dir"] = ray_temp_dir

    nontrivial_hack_lines: list[int] = []
    reward_error_lines: list[int] = []
    for line_idx, res in enumerate(ordered_results, start=1):
        reward_output = res.get("reward_output")
        if not isinstance(reward_output, dict):
            continue
        metadata = reward_output.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if metadata.get("nontrivial_hack", 0) != 0:
            print(f'{res["matched_index"]=}')
            nontrivial_hack_lines.append(line_idx)
        # RewardPCSWEFn exception paths return sparse metadata with `error`
        # and do not populate the hack-label fields.
        if "error" in metadata:
            reward_error_lines.append(line_idx)

    summary["nontrivial_hack_lines"] = nontrivial_hack_lines
    summary["reward_error_lines"] = reward_error_lines

    write_jsonl(args.output_path, ordered_results)
    write_json(args.summary_path, summary)

    log(f"Wrote per-rollout results to {args.output_path}")
    log(f"Wrote summary to {args.summary_path}")
    log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
