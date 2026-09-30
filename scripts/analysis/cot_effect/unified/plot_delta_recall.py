"""Unified-style CoT delta-recall plot for Sec. 4 (Fig. fig:cot-delta-recall).

Same data, recall definition, and EMA initialization as
../plot_delta_recall.py run with --ema-init-steps 10, restyled with
analysis/unified_style.py. No monitor evaluation is performed; recall is
recomputed from TP / (TP + FN) on each condition's valid predictions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unified_style import (  # noqa: E402
    BLUE, GRAY, RED, EMA_ALPHA, HALF_HEIGHT_IN, HALF_WIDTH_IN, LEGEND_PT_HALF,
    add_legend, apply_style, ema_smooth, fit_legend_headroom, new_axes,
    plot_series, save_fig,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "monitor_step_results"
OUTPUT_DIR = Path(__file__).resolve().parent

EXP1_RESULTS = RESULTS_DIR / "results_exp1_r2.json"  # w/o monitor penalty
EXP2_RESULTS = RESULTS_DIR / "results_exp2.json"     # w/ monitor penalty
EMA_INIT_STEPS = 10

YLABEL = r"$\Delta$Recall (w/ CoT $-$ w/o CoT)"
OUT_NAME = "cot_delta_recall_r2_init10_ema"


def load_results(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.items()}


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
    apply_style()

    fig, ax = new_axes(HALF_WIDTH_IN, HALF_HEIGHT_IN)
    ax.axhline(y=0, color=GRAY, linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
    ymax = 0.0
    max_step = 0
    summary = {
        "metric": "recall_with_cot - recall_without_cot",
        "recall_definition": "TP / (TP + FN), excluding invalid predictions separately for each condition",
        "smoothing": {
            "type": "recursive EMA", "alpha": EMA_ALPHA,
            "initialization": "first N-step arithmetic mean per experiment",
            "initial_steps": EMA_INIT_STEPS,
        },
        "experiments": {},
    }
    for path, color, label in [
        (EXP1_RESULTS, BLUE, "w/o Penalty"),
        (EXP2_RESULTS, RED, "w/ Penalty"),
    ]:
        results = load_results(path)
        if not results:
            raise ValueError(f"No results in {path}")
        steps, delta, rows = extract_recall(results)
        if not np.array_equal(steps[:EMA_INIT_STEPS], np.arange(1, EMA_INIT_STEPS + 1)):
            raise ValueError("Mean initialization requires complete steps 1 through N")
        smoothed = plot_series(ax, steps, delta, color, label,
                               initial_steps=EMA_INIT_STEPS)
        ymax = max(ymax, float(np.nanmax(smoothed)))
        index = EMA_INIT_STEPS - 1
        ax.scatter([steps[index]], [smoothed[index]], color=color, s=10, zorder=5)
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
                "step": int(steps[EMA_INIT_STEPS - 1]),
                "value": float(smoothed[EMA_INIT_STEPS - 1]),
                "n_observations": EMA_INIT_STEPS,
            },
            "blocks": blocks, "steps": rows,
        }
        max_step = int(steps.max())

    ax.axvspan(0, EMA_INIT_STEPS, color="0.5", alpha=0.10, zorder=0,
               label=f"Init. steps 1-{EMA_INIT_STEPS}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(YLABEL)
    ax.set_xlim(0, max_step)
    ax.margins(y=0.08)
    leg = add_legend(ax, LEGEND_PT_HALF)
    fit_legend_headroom(fig, ax, leg, ymax + 0.05)

    out_pdf = OUTPUT_DIR / f"{OUT_NAME}.pdf"
    save_fig(fig, out_pdf)
    summary_path = out_pdf.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
