"""
Plot ΔF1 (w/ comments − w/o comments) with EMA overlay across training steps.

Reads monitor_step_results JSONs and produces a half-column–ready PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "monitor_step_results"
OUTPUT_PATH = Path(__file__).resolve().parent / "delta_f1.pdf"

FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5

EMA_ALPHA = 0.05
RAW_ALPHA = 0.35


def load_results(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Results file does not exist: {path}")
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def extract_delta(results: dict[int, dict], metric: str):
    steps = sorted(results.keys())
    vals = [results[s]["delta"][metric] for s in steps]
    return np.array(steps), np.array(vals, dtype=float)


def ema_smooth(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = np.empty_like(values)
    mask = ~np.isnan(values)
    if not mask.any():
        return np.full_like(values, np.nan)
    first = int(np.argmax(mask))
    out[first] = values[first]
    for i in range(first + 1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    out[:first] = np.nan
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Plot annotation-effect delta F1")
    parser.add_argument(
        "--exp1-results",
        type=Path,
        default=RESULTS_DIR / "results_exp1.json",
    )
    parser.add_argument(
        "--exp2-results",
        type=Path,
        default=RESULTS_DIR / "results_exp2.json",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    exp1 = load_results(args.exp1_results)
    exp2 = load_results(args.exp2_results)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.4,
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)

    for results, color, label in [
        (exp1, "#2196F3", "w/o Penalty"),
        (exp2, "#E91E63", "w/ Penalty"),
    ]:
        if not results:
            continue
        steps, delta = extract_delta(results, "f1")
        ax.plot(steps, delta, color=color, linewidth=0.3, alpha=RAW_ALPHA)
        ax.plot(steps, ema_smooth(delta), color=color, linewidth=1.4, alpha=0.95, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\Delta$F1 (w/ Comments $-$ w/o Comments)")
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_xlim(left=0)

    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
