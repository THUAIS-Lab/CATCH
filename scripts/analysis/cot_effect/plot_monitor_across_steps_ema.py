#!/usr/bin/env python3
"""Visualize LLM monitor metrics with EMA smoothing.

Produces two figures:
  1. EMA-only (clean, publication-ready)
  2. Raw + EMA overlay (transparency shows variance)

Usage:
    python scripts/analysis/plot_monitor_across_steps_ema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "monitor_step_results"
OUTPUT_EMA = Path(__file__).resolve().parent / "monitor_across_steps_ema.png"
OUTPUT_OVERLAY = Path(__file__).resolve().parent / "monitor_across_steps_ema_overlay.png"
EMA_ALPHA = 0.05

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})


# ---------------------------------------------------------------------------
# Helpers
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


def ema_smooth(values: list, alpha: float = EMA_ALPHA) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.empty_like(arr)
    mask = ~np.isnan(arr)
    if not mask.any():
        return np.full_like(arr, np.nan)
    first = int(np.argmax(mask))
    # Initialize with mean of first N=20 valid values for stability
    n_warmup = min(20, len(arr) - first)
    init_val = np.nanmean(arr[first:first + n_warmup])
    out[first] = init_val
    for i in range(first + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    out[:first] = np.nan
    return out


# ---------------------------------------------------------------------------
# Plot: EMA only (clean)
# ---------------------------------------------------------------------------

def plot_ema_only(exp1, exp2):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ((ax_p, ax_r, ax_f), (ax_dp, ax_dr, ax_df)) = axes

    for ax, metric, title in [(ax_p, "precision", "Precision"),
                                (ax_r, "recall", "Recall"),
                                (ax_f, "f1", "F1")]:
        for results, color, prefix in [
            (exp1, "#2196F3", "exp1 (no penalty)"),
            (exp2, "#E91E63", "exp2 (penalty)"),
        ]:
            if not results:
                continue
            s, v_with = extract_series(results, "with", metric)
            _, v_wo = extract_series(results, "without", metric)
            ax.plot(s, ema_smooth(v_with), color=color, linewidth=1.2, label=f"{prefix} with think")
            ax.plot(s, ema_smooth(v_wo), color=color, linewidth=1.2, linestyle="--", label=f"{prefix} w/o think")
        ax.set_title(title)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.15)

    for ax, metric, title in [(ax_dp, "precision", "Δ Precision"),
                                (ax_dr, "recall", "Δ Recall"),
                                (ax_df, "f1", "Δ F1")]:
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
        for results, color, label in [
            (exp1, "#2196F3", "exp1 (no penalty)"),
            (exp2, "#E91E63", "exp2 (penalty)"),
        ]:
            if not results:
                continue
            s, d = extract_delta(results, metric)
            ax.plot(s, ema_smooth(d), color=color, linewidth=1.2, label=label)
        ax.set_title(title)
        ax.set_ylabel(f"Δ{metric.capitalize()} (with − w/o)")
        ax.legend(loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.15)

    for ax in [ax_p, ax_r, ax_f, ax_dp, ax_dr, ax_df]:
        ax.set_xlabel("Training step")

    fig.suptitle(f"LLM Monitor Performance (EMA α={EMA_ALPHA})\n"
                 "With vs. Without <think> tags | Ground truth: nontrivial_hack",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_EMA, dpi=200, bbox_inches="tight")
    print(f"Saved {OUTPUT_EMA}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot: raw + EMA overlay
# ---------------------------------------------------------------------------

def plot_overlay(exp1, exp2):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ((ax_p, ax_r, ax_f), (ax_dp, ax_dr, ax_df)) = axes
    RAW_ALPHA = 0.18

    for ax, metric, title in [(ax_p, "precision", "Precision"),
                                (ax_r, "recall", "Recall"),
                                (ax_f, "f1", "F1")]:
        for results, color, prefix in [
            (exp1, "#2196F3", "exp1 (no penalty)"),
            (exp2, "#E91E63", "exp2 (penalty)"),
        ]:
            if not results:
                continue
            s, v_with = extract_series(results, "with", metric)
            _, v_wo = extract_series(results, "without", metric)
            ax.plot(s, v_with, color=color, linewidth=0.3, alpha=RAW_ALPHA)
            ax.plot(s, v_wo, color=color, linewidth=0.3, linestyle="--", alpha=RAW_ALPHA)
            ax.plot(s, ema_smooth(v_with), color=color, linewidth=1.2, alpha=0.95, label=f"{prefix} with think")
            ax.plot(s, ema_smooth(v_wo), color=color, linewidth=1.2, linestyle="--", alpha=0.95, label=f"{prefix} w/o think")
        ax.set_title(title)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.15)

    for ax, metric, title in [(ax_dp, "precision", "Δ Precision"),
                                (ax_dr, "recall", "Δ Recall"),
                                (ax_df, "f1", "Δ F1")]:
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
        for results, color, label in [
            (exp1, "#2196F3", "exp1 (no penalty)"),
            (exp2, "#E91E63", "exp2 (penalty)"),
        ]:
            if not results:
                continue
            s, d = extract_delta(results, metric)
            ax.plot(s, d, color=color, linewidth=0.3, alpha=RAW_ALPHA)
            ax.plot(s, ema_smooth(d), color=color, linewidth=1.2, alpha=0.95, label=label)
        ax.set_title(title)
        ax.set_ylabel(f"Δ{metric.capitalize()} (with − w/o)")
        ax.legend(loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.15)

    for ax in [ax_p, ax_r, ax_f, ax_dp, ax_dr, ax_df]:
        ax.set_xlabel("Training step")

    fig.suptitle(f"LLM Monitor Performance (EMA α={EMA_ALPHA} + raw)\n"
                 "With vs. Without <think> tags | Ground truth: nontrivial_hack",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_OVERLAY, dpi=200, bbox_inches="tight")
    print(f"Saved {OUTPUT_OVERLAY}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    exp1 = load_results("exp1")
    exp2 = load_results("exp2")
    if not exp1 and not exp2:
        print("No results found.")
        sys.exit(1)
    print(f"Exp1: {len(exp1)} steps  Exp2: {len(exp2)} steps  EMA α={EMA_ALPHA}")
    plot_ema_only(exp1, exp2)
    plot_overlay(exp1, exp2)


if __name__ == "__main__":
    main()
