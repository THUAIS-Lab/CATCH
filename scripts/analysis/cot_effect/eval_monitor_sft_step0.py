#!/usr/bin/env python3
"""Evaluate the shared SFT initialization point for the CoT-effect curves.

Uniformly sample rows without replacement from the merged 8k normal + 3k
reward-hack SFT dataset.  The monitor sees ``normalized_input`` as its input
and either the full ``normalized_output`` (with CoT) or the same output with
the first ``<think>...</think>`` block removed (without CoT).

Predictions are checkpointed after every chunk so a long API run can resume
without repeating completed requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from eval_monitor_across_steps import (
    MonitorRunner,
    classify_batch,
    confusion_matrix,
    strip_think_tags,
)


ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR / "monitor_step_results"
DEFAULT_DATASET = Path(
    "/home/fit/lijuanzi/WORK/ShouliWang/datasets/"
    "toxic_sft_v4_1/merged_nt8k_t3k.parquet"
)
DEFAULT_RESULTS = RESULTS_DIR / "results_sft_step0.json"
DEFAULT_PREDICTIONS = RESULTS_DIR / "predictions_sft_step0.json"

REQUIRED_COLUMNS = ("normalized_input", "normalized_output", "hack_style")
MODES = ("with", "without")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a shared SFT step 0 for the CoT-effect curves"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Predictions saved after each chunk (default: 128)",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--predictions-cache",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )
    parser.add_argument(
        "--retry-excluded",
        action="store_true",
        help="Retry cached API failures after all new predictions are complete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the deterministic sample without API calls",
    )
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def _sample_fingerprint(
    indices: list[int], questions: list[str], actions: list[str], gts: list[bool]
) -> str:
    digest = hashlib.sha256()
    for index, question, action, gt in zip(indices, questions, actions, gts):
        for value in (str(index), question, action, "1" if gt else "0"):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def load_sample(
    dataset: Path, sample_size: int, seed: int
) -> tuple[list[str], list[str], list[str], list[bool], dict[str, Any]]:
    if not dataset.is_file():
        raise FileNotFoundError(f"SFT dataset does not exist: {dataset}")

    frame = pd.read_parquet(dataset, columns=list(REQUIRED_COLUMNS))
    if sample_size > len(frame):
        raise ValueError(
            f"Cannot sample {sample_size} rows without replacement from {len(frame)} rows"
        )

    null_counts = frame[list(REQUIRED_COLUMNS)].isna().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(f"Required SFT columns contain nulls: {null_counts.to_dict()}")

    styles = frame["hack_style"].astype(str).str.strip()
    unexpected_styles = sorted(set(styles) - {"normal", "mislead", "pure_hack"})
    if unexpected_styles:
        raise ValueError(f"Unexpected hack_style values: {unexpected_styles}")

    # random.sample gives every size-k subset the same probability and keeps the
    # selected row order deterministic for checkpoint/resume.
    indices = random.Random(seed).sample(range(len(frame)), sample_size)
    sampled = frame.iloc[indices]
    questions = sampled["normalized_input"].astype(str).tolist()
    actions_with = sampled["normalized_output"].astype(str).tolist()
    actions_without = [strip_think_tags(action) for action in actions_with]
    gts = [str(style).strip() != "normal" for style in sampled["hack_style"]]
    fingerprint = _sample_fingerprint(indices, questions, actions_with, gts)

    metadata: dict[str, Any] = {
        "dataset": str(dataset.resolve()),
        "dataset_rows": len(frame),
        "sample_size": sample_size,
        "seed": seed,
        "sample_indices": indices,
        "sample_fingerprint_sha256": fingerprint,
        "n_hack": sum(gts),
        "n_normal": len(gts) - sum(gts),
        "n_outputs_changed_without_cot": sum(
            before != after for before, after in zip(actions_with, actions_without)
        ),
    }
    return questions, actions_with, actions_without, gts, metadata


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, path)


def load_or_create_cache(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            cache = json.load(fh)
        if cache.get("version") != 1:
            raise ValueError(f"Unsupported prediction cache version in {path}")
        if cache.get("metadata") != metadata:
            raise ValueError(
                f"Prediction cache metadata does not match this sample: {path}. "
                "Use a different --predictions-cache path."
            )
    else:
        cache = {
            "version": 1,
            "metadata": metadata,
            "predictions": {mode: [] for mode in MODES},
        }

    predictions = cache.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"Malformed prediction cache: {path}")
    for mode in MODES:
        values = predictions.get(mode)
        if not isinstance(values, list) or len(values) > metadata["sample_size"]:
            raise ValueError(f"Malformed {mode!r} predictions in {path}")
        if any(value is not None and not isinstance(value, bool) for value in values):
            raise ValueError(f"Invalid {mode!r} prediction value in {path}")
    return cache


async def evaluate_mode(
    runner: MonitorRunner,
    mode: str,
    questions: list[str],
    actions: list[str],
    cache: dict[str, Any],
    cache_path: Path,
    chunk_size: int,
) -> None:
    predictions: list[bool | None] = cache["predictions"][mode]
    total = len(questions)
    while len(predictions) < total:
        start = len(predictions)
        stop = min(start + chunk_size, total)
        chunk_predictions = await classify_batch(
            runner,
            questions[start:stop],
            actions[start:stop],
            desc=f"sft/step0/{mode}/{start}:{stop}",
        )
        predictions.extend(chunk_predictions)
        atomic_write_json(cache_path, cache)
        excluded = sum(value is None for value in predictions)
        print(
            f"  [sft/step0/{mode}] checkpoint {len(predictions)}/{total}; "
            f"excluded={excluded}",
            flush=True,
        )


async def retry_excluded(
    runner: MonitorRunner,
    mode: str,
    questions: list[str],
    actions: list[str],
    cache: dict[str, Any],
    cache_path: Path,
    chunk_size: int,
) -> None:
    predictions: list[bool | None] = cache["predictions"][mode]
    excluded_indices = [i for i, value in enumerate(predictions) if value is None]
    if not excluded_indices:
        return
    print(f"  [sft/step0/{mode}] retrying {len(excluded_indices)} excluded predictions")
    for offset in range(0, len(excluded_indices), chunk_size):
        indices = excluded_indices[offset : offset + chunk_size]
        retried = await classify_batch(
            runner,
            [questions[i] for i in indices],
            [actions[i] for i in indices],
            desc=f"sft/step0/{mode}/retry",
        )
        for index, prediction in zip(indices, retried):
            predictions[index] = prediction
        atomic_write_json(cache_path, cache)


def build_results(
    cache: dict[str, Any], gts: list[bool], metadata: dict[str, Any]
) -> dict[str, Any]:
    step: dict[str, Any] = {}
    for mode in MODES:
        predictions = cache["predictions"][mode]
        if len(predictions) != len(gts):
            raise ValueError(
                f"Cannot build final results: {mode} has {len(predictions)}/{len(gts)} predictions"
            )
        metrics = confusion_matrix(predictions, gts)
        metrics["n_hack"] = sum(gts)
        metrics["n_total"] = len(gts)
        metrics["excluded"] = sum(value is None for value in predictions)
        step[mode] = metrics

    step["delta"] = {
        metric: round(step["with"][metric] - step["without"][metric], 6)
        for metric in ("precision", "recall", "f1")
    }
    step["sample"] = {
        key: value
        for key, value in metadata.items()
        if key != "sample_indices"
    }
    return {"0": step}


async def run(args: argparse.Namespace) -> None:
    questions, actions_with, actions_without, gts, metadata = load_sample(
        args.dataset, args.sample_size, args.seed
    )
    print(
        f"SFT sample: {metadata['sample_size']}/{metadata['dataset_rows']} rows, "
        f"seed={metadata['seed']}, hack={metadata['n_hack']}, "
        f"normal={metadata['n_normal']}, "
        f"without-CoT changed={metadata['n_outputs_changed_without_cot']}"
    )
    print(f"Sample fingerprint: {metadata['sample_fingerprint_sha256']}")
    if args.dry_run:
        print("Dry run: no monitor requests or output files were created.")
        return

    cache = load_or_create_cache(args.predictions_cache, metadata)
    actions_by_mode = {"with": actions_with, "without": actions_without}
    needs_new = any(
        len(cache["predictions"][mode]) < args.sample_size for mode in MODES
    )
    has_excluded = any(
        any(value is None for value in cache["predictions"][mode]) for mode in MODES
    )

    runner: MonitorRunner | None = None
    if needs_new or (args.retry_excluded and has_excluded):
        runner = MonitorRunner()
        try:
            for mode in MODES:
                await evaluate_mode(
                    runner,
                    mode,
                    questions,
                    actions_by_mode[mode],
                    cache,
                    args.predictions_cache,
                    args.chunk_size,
                )
            if args.retry_excluded:
                for mode in MODES:
                    await retry_excluded(
                        runner,
                        mode,
                        questions,
                        actions_by_mode[mode],
                        cache,
                        args.predictions_cache,
                        args.chunk_size,
                    )
        finally:
            await runner.close()
    else:
        print("Prediction cache is complete; no monitor requests needed.")

    results = build_results(cache, gts, metadata)
    atomic_write_json(args.results, results)
    step = results["0"]
    print(
        "SFT step 0 complete: "
        f"with F1={step['with']['f1']:.6f} "
        f"without F1={step['without']['f1']:.6f} "
        f"delta={step['delta']['f1']:+.6f}; "
        f"excluded={step['with']['excluded'] + step['without']['excluded']}"
    )
    print(f"Results saved to {args.results}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
