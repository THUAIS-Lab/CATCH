"""Plot comment delta recall from cached monitor results with ordinary recursive EMA.

Uses the same visual style as plot_delta_f1.py. Recall is recomputed
from TP / (TP + FN) on each condition's valid predictions, preserving the
existing evaluation convention. No monitor evaluation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_delta_f1 import (
    EMA_ALPHA, FIG_HEIGHT_IN, FIG_WIDTH_IN, RAW_ALPHA, RESULTS_DIR,
    load_results,
)


def ema_smooth(
    values: np.ndarray, alpha: float = EMA_ALPHA, initial_steps: int = 1,
) -> np.ndarray:
    """Ordinary EMA, optionally initialized by the first N-step mean at step N."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if initial_steps < 1:
        raise ValueError("initial_steps must be positive")
    out = np.full(values.shape, np.nan, dtype=float)
    if initial_steps > 1:
        if len(values) < initial_steps or not np.isfinite(values[:initial_steps]).all():
            raise ValueError("Initialization requires N finite observations")
        first = initial_steps - 1
        out[first] = values[:initial_steps].mean()
    else:
        valid = np.flatnonzero(~np.isnan(values))
        if valid.size == 0:
            return out
        first = int(valid[0])
        out[first] = values[first]
    for i in range(first + 1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def recall_from_counts(row: dict) -> float:
    denominator = row["tp"] + row["fn"]
    return row["tp"] / denominator if denominator else 0.0


def extract_recall(results: dict[int, dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    steps = sorted(results)
    rows = []
    for step in steps:
        entry = results[step]
        with_recall = recall_from_counts(entry["with"])
        without_recall = recall_from_counts(entry["without"])
        delta = with_recall - without_recall
        if abs(delta - entry["delta"]["recall"]) > 1.000001e-6:
            raise ValueError(f"Stored delta recall disagrees with counts at step {step}")
        rows.append({
            "step": step, "with_recall": with_recall,
            "without_recall": without_recall, "delta_recall": delta,
            "with_excluded": entry["with"].get("excluded", 0),
            "without_excluded": entry["without"].get("excluded", 0),
        })
    return np.asarray(steps), np.asarray([r["delta_recall"] for r in rows]), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1-results", type=Path, default=RESULTS_DIR / "results_exp1_r2.json")
    parser.add_argument("--exp2-results", type=Path, default=RESULTS_DIR / "results_exp2.json")
    parser.add_argument(
        "--ema-init-steps", type=int, default=20,
        help="Initialize EMA at step N with the arithmetic mean of steps 1 through N.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ema_init_steps < 1:
        parser.error("--ema-init-steps must be positive")
    if args.output is None:
        if args.ema_init_steps > 1:
            name = f"delta_recall_r2_init{args.ema_init_steps}_ema.pdf"
        else:
            name = "delta_recall_r2.pdf"
        args.output = Path(__file__).with_name(name)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 14, "axes.labelsize": 14,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "lines.linewidth": 1.4,
    })
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    summary = {
        "metric": "recall_with_comments - recall_without_comments",
        "recall_definition": "TP / (TP + FN), excluding invalid predictions separately for each condition",
        "smoothing": {
            "type": "recursive EMA", "alpha": EMA_ALPHA,
            "initialization": "first N-step arithmetic mean per experiment" if args.ema_init_steps > 1 else "first valid observation",
            "initial_steps": args.ema_init_steps,
        },
        "experiments": {},
    }
    for path, color, label in [
        (args.exp1_results, "#2196F3", "w/o Penalty"),
        (args.exp2_results, "#E91E63", "w/ Penalty"),
    ]:
        results = load_results(path)
        if not results:
            raise ValueError(f"No results in {path}")
        steps, delta, rows = extract_recall(results)
        if args.ema_init_steps > 1:
            if not np.array_equal(steps[:args.ema_init_steps], np.arange(1, args.ema_init_steps + 1)):
                raise ValueError("Mean initialization requires complete steps 1 through N")
        smoothed = ema_smooth(delta, initial_steps=args.ema_init_steps)
        ax.plot(steps, delta, color=color, linewidth=0.3, alpha=RAW_ALPHA)
        ax.plot(steps, smoothed, color=color, linewidth=1.4, alpha=0.95, label=label)
        if args.ema_init_steps > 1:
            index = args.ema_init_steps - 1
            ax.scatter([steps[index]], [smoothed[index]], color=color, s=22, zorder=5)
        for row, value in zip(rows, smoothed):
            row["smoothed_delta_recall"] = float(value) if np.isfinite(value) else None
        blocks = []
        for lo, hi in [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 140)]:
            selected = [row for row in rows if lo <= row["step"] <= hi]
            if selected:
                blocks.append({
                    "start": lo, "end": hi, "n_steps": len(selected),
                    "mean_delta_recall": float(np.mean([row["delta_recall"] for row in selected])),
                    "mean_with_recall": float(np.mean([row["with_recall"] for row in selected])),
                    "mean_without_recall": float(np.mean([row["without_recall"] for row in selected])),
                })
        summary["experiments"][label] = {
            "source": str(path.resolve()),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "initialization": {
                "step": int(steps[args.ema_init_steps - 1]),
                "value": float(smoothed[args.ema_init_steps - 1]),
                "n_observations": args.ema_init_steps,
            },
            "blocks": blocks, "steps": rows,
        }
    if args.ema_init_steps > 1:
        ax.axvspan(
            0, args.ema_init_steps, color="0.5", alpha=0.10, zorder=0,
            label=f"Init. steps 1-{args.ema_init_steps}",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\Delta$Recall (w/ Comments $-$ w/o Comments)")
    ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_xlim(left=0)
    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
