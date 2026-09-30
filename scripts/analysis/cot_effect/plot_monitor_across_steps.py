#!/usr/bin/env python3
"""Visualize LLM monitor metrics across training steps.

Usage:
    python scripts/analysis/plot_monitor_across_steps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "monitor_step_results"
OUTPUT_PATH = Path(__file__).resolve().parent / "monitor_across_steps.png"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(exp_name: str) -> dict[int, dict]:
    path = RESULTS_DIR / f"results_{exp_name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def extract_series(results: dict[int, dict], mode: str, metric: str):
    steps = sorted(results.keys())
    vals = [results[s].get(mode, {}).get(metric) for s in steps]
    return steps, vals


def extract_delta(results: dict[int, dict], metric: str):
    steps = sorted(results.keys())
    vals = [results[s].get("delta", {}).get(metric) for s in steps]
    return steps, vals


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric(ax, exp1, exp2, metric: str, title: str):
    for results, color, label in [
        (exp1, "#2196F3", "exp1 (no penalty)"),
        (exp2, "#E91E63", "exp2 (penalty)"),
    ]:
        if not results:
            continue
        s, v_with = extract_series(results, "with", metric)
        _, v_wo = extract_series(results, "without", metric)
        ax.plot(s, v_with, color=color, linewidth=0.8, label=f"{label} with think", alpha=0.9)
        ax.plot(s, v_wo, color=color, linewidth=0.8, linestyle="--", label=f"{label} w/o think", alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.8, ncol=2)
    ax.grid(True, alpha=0.2)


def plot_delta(ax, exp1, exp2, metric: str, title: str):
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    for results, color, label in [
        (exp1, "#2196F3", "exp1 (no penalty)"),
        (exp2, "#E91E63", "exp2 (penalty)"),
    ]:
        if not results:
            continue
        s, d = extract_delta(results, metric)
        ax.plot(s, d, color=color, linewidth=0.8, label=label)
    ax.set_title(title)
    ax.set_ylabel(f"Δ{metric.capitalize()} (with − w/o)")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    exp1 = load_results("exp1")
    exp2 = load_results("exp2")
    if not exp1 and not exp2:
        print("No results found.")
        sys.exit(1)
    print(f"Exp1: {len(exp1)} steps  Exp2: {len(exp2)} steps")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ((ax_p, ax_r, ax_f), (ax_dp, ax_dr, ax_df)) = axes

    plot_metric(ax_p, exp1, exp2, "precision", "Precision")
    plot_metric(ax_r, exp1, exp2, "recall", "Recall")
    plot_metric(ax_f, exp1, exp2, "f1", "F1")

    plot_delta(ax_dp, exp1, exp2, "precision", "Δ Precision")
    plot_delta(ax_dr, exp1, exp2, "recall", "Δ Recall")
    plot_delta(ax_df, exp1, exp2, "f1", "Δ F1")

    for ax in [ax_p, ax_r, ax_f, ax_dp, ax_dr, ax_df]:
        ax.set_xlabel("Training step")

    fig.suptitle("LLM Monitor Performance Across Training Steps\n"
                 "With vs. Without <think> tags | Ground truth: nontrivial_hack",
                 fontsize=11, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
