#!/usr/bin/env python3
"""Visualize annotation effect (Python comments) on LLM monitor.

Usage:
    python scripts/analysis/annotation_effect/plot_annotation_effect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "monitor_step_results"
OUTPUT = Path(__file__).resolve().parent / "annotation_effect.png"
OUTPUT_EMA = Path(__file__).resolve().parent / "annotation_effect_ema.png"
OUTPUT_OVERLAY = Path(__file__).resolve().parent / "annotation_effect_ema_overlay.png"
EMA_ALPHA = 0.05

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})


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
    n_warmup = min(20, len(arr) - first)
    out[first] = np.nanmean(arr[first:first + n_warmup])
    for i in range(first + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    out[:first] = np.nan
    return out


def plot_metric(ax, exp1, exp2, metric: str, title: str, use_ema: bool = False):
    for results, color, label in [
        (exp1, "#2196F3", "exp1 (no penalty)"),
        (exp2, "#E91E63", "exp2 (penalty)"),
    ]:
        if not results:
            continue
        s, v_with = extract_series(results, "with", metric)
        _, v_wo = extract_series(results, "without", metric)
        if use_ema:
            v_with, v_wo = ema_smooth(v_with), ema_smooth(v_wo)
        ax.plot(s, v_with, color=color, linewidth=1.2 if use_ema else 0.8,
                label=f"{label} with comments")
        ax.plot(s, v_wo, color=color, linewidth=1.2 if use_ema else 0.8,
                linestyle="--", label=f"{label} w/o comments")
    ax.set_title(title)
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.8, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("Training step")


def plot_delta(ax, exp1, exp2, metric: str, title: str, use_ema: bool = False):
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    for results, color, label in [
        (exp1, "#2196F3", "exp1 (no penalty)"),
        (exp2, "#E91E63", "exp2 (penalty)"),
    ]:
        if not results:
            continue
        s, d = extract_delta(results, metric)
        if use_ema:
            d = ema_smooth(d)
        ax.plot(s, d, color=color, linewidth=1.2 if use_ema else 0.8, label=label)
    ax.set_title(title)
    ax.set_ylabel(f"Δ{metric.capitalize()} (with − w/o)")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("Training step")


def make_figure(exp1, exp2, use_ema: bool, overlay: bool = False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ((ax_p, ax_r, ax_f), (ax_dp, ax_dr, ax_df)) = axes

    for ax, metric, title in [(ax_p, "precision", "Precision"),
                                (ax_r, "recall", "Recall"),
                                (ax_f, "f1", "F1")]:
        if overlay:
            # Raw faint + EMA bold
            for results, color, label in [
                (exp1, "#2196F3", "exp1 (no penalty)"),
                (exp2, "#E91E63", "exp2 (penalty)"),
            ]:
                if not results:
                    continue
                s, v_with = extract_series(results, "with", metric)
                _, v_wo = extract_series(results, "without", metric)
                ax.plot(s, v_with, color=color, linewidth=0.3, alpha=0.18)
                ax.plot(s, v_wo, color=color, linewidth=0.3, linestyle="--", alpha=0.18)
                ax.plot(s, ema_smooth(v_with), color=color, linewidth=1.2, alpha=0.95,
                        label=f"{label} with comments")
                ax.plot(s, ema_smooth(v_wo), color=color, linewidth=1.2, linestyle="--",
                        alpha=0.95, label=f"{label} w/o comments")
        else:
            plot_metric(ax, exp1, exp2, metric, title, use_ema)
        ax.set_title(title)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("Training step")

    for ax, metric, title in [(ax_dp, "precision", "Δ Precision"),
                                (ax_dr, "recall", "Δ Recall"),
                                (ax_df, "f1", "Δ F1")]:
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
        if overlay:
            for results, color, label in [
                (exp1, "#2196F3", "exp1 (no penalty)"),
                (exp2, "#E91E63", "exp2 (penalty)"),
            ]:
                if not results:
                    continue
                s, d = extract_delta(results, metric)
                ax.plot(s, d, color=color, linewidth=0.3, alpha=0.18)
                ax.plot(s, ema_smooth(d), color=color, linewidth=1.2, alpha=0.95, label=label)
        else:
            plot_delta(ax, exp1, exp2, metric, title, use_ema)
        ax.set_title(title)
        ax.set_ylabel(f"Δ{metric.capitalize()} (with − w/o)")
        ax.legend(loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("Training step")

    mode_str = f"EMA α={EMA_ALPHA}" if use_ema else ("EMA + raw" if overlay else "Raw")
    fig.suptitle(f"Effect of Python Comments on LLM Monitor ({mode_str})\n"
                 "With vs. Without comments in .py files | Ground truth: nontrivial_hack",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def main():
    exp1 = load_results("exp1")
    exp2 = load_results("exp2")
    if not exp1 and not exp2:
        print("No results found.")
        sys.exit(1)
    print(f"Exp1: {len(exp1)} steps  Exp2: {len(exp2)} steps")

    for path, ema, overlay in [(OUTPUT, False, False), (OUTPUT_EMA, True, False),
                                (OUTPUT_OVERLAY, True, True)]:
        fig = make_figure(exp1, exp2, use_ema=ema, overlay=overlay)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
