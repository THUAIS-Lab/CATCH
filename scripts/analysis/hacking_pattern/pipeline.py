#!/usr/bin/env python3
"""
Four-phase analysis pipeline: load → extract → save → (plot separately).

Phase 1: Parallel load all RL checkpoint JSONL files.
Phase 2: Parallel extract patterns across all samples (per step).
Phase 3: Aggregate & save to results/ as JSON.
Phase 4: Plot from saved data (see plot.py).

Usage:
    # Full pipeline for calls.json
    python scripts/analysis/hacking_pattern/pipeline.py

    # Single experiment, specific hack method, no tokenizer
    python scripts/analysis/hacking_pattern/pipeline.py --exp n16 --hack calls.json --no-tokenize

    # Rerun trajectory without overwriting the original n16 data
    python scripts/analysis/hacking_pattern/pipeline.py --exp n16r2 \
        --max-step 140 --no-tokenize --summary-name summary_n16-r2.json

    # Skip loading if results/ already has data (recompute only)
    python scripts/analysis/hacking_pattern/pipeline.py --from-cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Ensure rllm is importable
_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from scripts.analysis.hacking_pattern.config import (
    N16_DIR,
    N16_R2_DIR,
    N16W_DIR,
    OUTPUT_DIR,
    PARALLEL_WORKERS,
    RESULTS_DIR,
    SFT_MAX_SAMPLES,
    SFT_PATH,
    TOKENIZER_PATH,
)
from scripts.analysis.hacking_pattern.patterns import ALL_GROUPS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_cot_code(text: str) -> tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    cot = m.group(1).strip() if m else ""
    idx = text.find("</think>")
    code = text[idx + len("</think>"):].strip() if idx != -1 else text.strip()
    return cot, code


def _extract_comments(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            lines.append(s)
    for m in re.finditer(r'""".*?"""|\'\'\'.*?\'\'\'', code, re.DOTALL):
        lines.append(m.group(0))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: Parallel loading
# ---------------------------------------------------------------------------
def _load_one_jsonl(args: tuple[str, str]) -> tuple[int, list[dict]]:
    """Load a single JSONL checkpoint. (runs in thread pool)"""
    dir_path, fname = args
    step = int(fname.replace(".jsonl", ""))
    fpath = os.path.join(dir_path, fname)
    samples: list[dict] = []
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            if d.get("hack_method") != "calls.json":
                continue
            cot, code = _split_cot_code(str(d.get("output", "")))
            samples.append({
                "cot": cot,
                "code": code,
                "score": d.get("score"),
                "ia_hack": d.get("ia_hack"),
                "is_trivial_hack": d.get("is_trivial_hack"),
            })
    return step, samples


def _load_sft() -> list[dict]:
    """Load SFT data."""
    import pandas as pd
    df = pd.read_parquet(SFT_PATH)
    mask = df["hack_method"] == "calls.json"
    sub = df[mask].head(SFT_MAX_SAMPLES)
    samples = []
    for _, row in sub.iterrows():
        cot, code = _split_cot_code(str(row["normalized_output"]))
        samples.append({"cot": cot, "code": code})
    return samples


def load_all(dir_path: str, max_step: int) -> dict[int, list[dict]]:
    """Parallel load all RL steps. Returns {step: [sample_dict, ...]}."""
    fnames = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(".jsonl")],
        key=lambda x: int(x.replace(".jsonl", ""))
    )
    fnames = [f for f in fnames if int(f.replace(".jsonl", "")) <= max_step]

    print(f"  Loading {len(fnames)} steps from {dir_path} "
          f"(workers={PARALLEL_WORKERS})...")

    tasks = [(dir_path, f) for f in fnames]
    results: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {pool.submit(_load_one_jsonl, t): t for t in tasks}
        for future in as_completed(futures):
            step, samples = future.result()
            if samples:
                results[step] = samples

    steps = sorted(results.keys())
    if steps:
        print(f"  Loaded {len(results)} steps "
              f"(range {steps[0]}-{steps[-1]}, "
              f"total {sum(len(v) for v in results.values())} samples)")
    return results


# ---------------------------------------------------------------------------
# Phase 2: Parallel extraction
# ---------------------------------------------------------------------------
def _extract_one_sample(args: tuple[dict, dict[str, dict[str, re.Pattern]]]
                        ) -> dict[str, dict[str, bool]]:
    """Extract all patterns from a single sample. (runs in process pool)"""
    sample, groups = args
    result: dict[str, dict[str, bool]] = {}

    for gname, patterns in groups.items():
        row: dict[str, bool] = {}
        for pname, pat in patterns.items():
            # search in cot+code (simplified: both)
            haystack = sample["cot"] + "\n" + sample["code"]
            row[pname] = bool(pat.search(haystack))
        result[gname] = row

    return result


def _extract_batch(samples: list[dict],
                   groups: dict[str, dict[str, re.Pattern]],
                   tokenizer=None) -> dict[str, list[dict]]:
    """Extract patterns from a batch of samples (one RL step)."""
    # Parallel across samples within this batch
    tasks = [(s, groups) for s in samples]
    per_sample: list[dict[str, dict[str, bool]]] = []

    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = [pool.submit(_extract_one_sample, t) for t in tasks]
        for future in as_completed(futures):
            per_sample.append(future.result())

    # Aggregate: group -> {pattern_name: [bool per sample]}
    agg: dict[str, list[dict[str, bool]]] = defaultdict(list)
    for ps in per_sample:
        for gname, row in ps.items():
            agg[gname].append(row)
    return dict(agg)


def compute_token_stats(samples: list[dict], tokenizer) -> dict[str, float]:
    """Compute mean token lengths for a batch."""
    if not samples:
        return {}
    cots, codes, comments, totals = [], [], [], []
    for s in samples:
        ct = len(tokenizer.encode(s["cot"])) if s["cot"] else 0
        cd = len(tokenizer.encode(s["code"])) if s["code"] else 0
        cmt_text = _extract_comments(s["code"])
        cmt = len(tokenizer.encode(cmt_text)) if cmt_text else 0
        cots.append(ct); codes.append(cd); comments.append(cmt)
        totals.append(ct + cd)
    return {
        "cot_mean": float(np.mean(cots)),
        "code_mean": float(np.mean(codes)),
        "comments_mean": float(np.mean(comments)),
        "total_mean": float(np.mean(totals)),
        "comments_cot_ratio": float(np.mean(comments) / max(np.mean(cots), 1)),
    }


# ---------------------------------------------------------------------------
# Phase 3: Aggregate & save
# ---------------------------------------------------------------------------
def _aggregate_step(group_data: list[dict[str, bool]]) -> dict[str, float]:
    """Convert per-sample booleans to percentages."""
    n = len(group_data)
    if n == 0:
        return {}
    counts = defaultdict(int)
    for row in group_data:
        for k, v in row.items():
            if v: counts[k] += 1
    return {k: v / n * 100 for k, v in counts.items()}


def build_trajectory(
    sft_samples: list[dict],
    rl_data: dict[int, list[dict]],
    groups: dict[str, dict[str, re.Pattern]],
    tokenizer=None,
) -> dict:
    """Aggregate all steps into a trajectory dict."""
    print("  Extracting patterns (parallel across samples)...")

    all_steps: list[tuple[int, list[dict]]] = [(-1, sft_samples)]  # -1 = SFT
    all_steps += sorted(rl_data.items())

    # Collect per-step aggregated results
    steps_list: list[int] = []
    n_samples_list: list[int] = []
    patterns: dict[str, dict[str, list[float]]] = {
        gname: defaultdict(list) for gname in groups
    }
    token_stats: dict[str, list[float]] = defaultdict(list)

    for step, samples in all_steps:
        steps_list.append(step)
        n_samples_list.append(len(samples))

        # Extract (parallel within this batch)
        extracted = _extract_batch(samples, groups, tokenizer)

        for gname in groups:
            agg = _aggregate_step(extracted.get(gname, []))
            for pname in groups[gname]:
                patterns[gname][pname].append(agg.get(pname, 0.0))

        if tokenizer:
            ts = compute_token_stats(samples, tokenizer)
            for k, v in ts.items():
                token_stats[k].append(v)

        if step % 10 == 0 or step == -1:
            label = "SFT" if step == -1 else str(step)
            print(f"    [{label}] {len(samples)} samples extracted")

    return {
        "steps": steps_list,
        "n_samples": n_samples_list,
        "patterns": {k: dict(v) for k, v in patterns.items()},
        "token_stats": dict(token_stats) if token_stats else {},
    }


def save_results(
    trajectories: dict[str, dict],
    hack_method: str,
    summary_name: str = "summary.json",
) -> str:
    """Save all trajectory data to results/ as JSON files."""
    out_dir = os.path.join(RESULTS_DIR, hack_method)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for exp_name, traj in trajectories.items():
        path = os.path.join(out_dir, f"trajectory_{exp_name}.json")
        with open(path, "w") as f:
            json.dump(traj, f, indent=2, default=str)
        print(f"  Saved: {path}")

    # Also save a combined summary table for quick inspection
    summary = _build_summary(trajectories)
    spath = os.path.join(out_dir, summary_name)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {spath}")

    return out_dir


def _build_summary(trajectories: dict[str, dict]) -> dict:
    """Build a human-readable summary comparing SFT vs final RL states."""
    summary = {}
    for exp_name, traj in trajectories.items():
        exp_summary = {}
        for gname, pdata in traj["patterns"].items():
            exp_summary[gname] = {}
            for pname, values in pdata.items():
                sft_val = values[0] if values else None
                last_val = values[-1] if values else None
                delta = (last_val - sft_val) if (sft_val is not None and last_val is not None) else None
                exp_summary[gname][pname] = {
                    "sft": round(sft_val, 1) if sft_val is not None else None,
                    "final": round(last_val, 1) if last_val is not None else None,
                    "delta": round(delta, 1) if delta is not None else None,
                }
        summary[exp_name] = exp_summary
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Hacking pattern evolution pipeline")
    p.add_argument("--exp", choices=["all", "n16", "n16w", "n16r2"], default="all")
    p.add_argument("--max-step", type=int, default=200)
    p.add_argument("--no-tokenize", action="store_true")
    p.add_argument("--from-cache", action="store_true",
                   help="Skip loading & extraction if results/ data exists")
    p.add_argument("--hack", type=str, default="calls.json",
                   help="hack_method filter")
    p.add_argument(
        "--summary-name",
        default="summary.json",
        help="Summary filename inside results/<hack>/",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    hack_method = args.hack
    cache_dir = os.path.join(RESULTS_DIR, hack_method)
    cache_names = {
        "all": ["n16", "n16-w_monitor"],
        "n16": ["n16"],
        "n16w": ["n16-w_monitor"],
        "n16r2": ["n16-r2"],
    }[args.exp]

    if args.from_cache and all(
        os.path.exists(os.path.join(cache_dir, f"trajectory_{name}.json"))
        for name in cache_names
    ):
        print(f"Results already exist in {cache_dir}. Skipping computation.")
        print(f"Run plot.py to regenerate figures.")
        return

    # ---- Tokenizer ----
    tokenizer = None
    if not args.no_tokenize:
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, trust_remote_code=True, fix_mistral_regex=True
        )

    # ---- Phase 1: Load SFT ----
    print("\n[Phase 1] Loading SFT...")
    sft_samples = _load_sft()
    print(f"  SFT: {len(sft_samples)} samples")

    # ---- Phase 2: Parallel load RL ----
    exps = []
    if args.exp in ("all", "n16"):
        exps.append(("n16", N16_DIR))
    if args.exp in ("all", "n16w"):
        exps.append(("n16-w_monitor", N16W_DIR))
    if args.exp == "n16r2":
        exps.append(("n16-r2", N16_R2_DIR))

    rl_data_all: dict[str, dict[int, list[dict]]] = {}
    for exp_name, exp_dir in exps:
        print(f"\n[Phase 2] Loading {exp_name} steps...")
        rl_data_all[exp_name] = load_all(exp_dir, args.max_step)

    # ---- Phase 3: Extract & build trajectories ----
    groups = ALL_GROUPS
    trajectories: dict[str, dict] = {}

    for exp_name, rl_data in rl_data_all.items():
        print(f"\n[Phase 3] Extracting {exp_name} trajectory...")
        traj = build_trajectory(sft_samples, rl_data, groups, tokenizer)
        trajectories[exp_name] = traj

    # ---- Phase 4: Save ----
    print(f"\n[Phase 4] Saving results...")
    saved_dir = save_results(trajectories, hack_method, args.summary_name)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Data saved to {saved_dir}/")
    print(f"Run: python scripts/analysis/hacking_pattern/plot.py to generate figures.")


if __name__ == "__main__":
    main()
