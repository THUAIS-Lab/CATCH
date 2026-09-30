"""
Plot ΔF1 (with think − w/o think) with EMA overlay across training steps.

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

# Large canvas → scale down in LaTeX for dense grid + compact legend.
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
    """Normalize exponential weights so the first sample has no extra weight.

    Matches pandas ewm(adjust=True, ignore_na=False): missing steps decay
    earlier weights while retaining the last available smoothed value.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = np.full(values.shape, np.nan, dtype=float)
    weighted_mean = np.nan
    weight_sum = 0.0
    for i, value in enumerate(values):
        weight_sum *= 1 - alpha
        if not np.isnan(value):
            if weight_sum == 0:
                weighted_mean = float(value)
            else:
                weighted_mean = (weight_sum * weighted_mean + value) / (weight_sum + 1)
            weight_sum += 1
        out[i] = weighted_mean
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CoT-effect delta F1")
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
    parser.add_argument(
        "--step0-results",
        type=Path,
        help=(
            "Optional shared SFT step-0 results. Its step 0 is injected into "
            "both experiment curves without modifying either input JSON."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def inject_shared_step0(
    exp1: dict[int, dict], exp2: dict[int, dict], step0_path: Path
) -> tuple[dict[int, dict], dict[int, dict]]:
    step0_results = load_results(step0_path)
    if 0 not in step0_results:
        raise ValueError(f"Shared step-0 results have no step 0: {step0_path}")
    step0 = step0_results[0]
    missing = {"with", "without", "delta"} - step0.keys()
    if missing:
        raise ValueError(
            f"Shared step-0 results are missing {sorted(missing)}: {step0_path}"
        )
    exp1_with_step0 = dict(exp1)
    exp2_with_step0 = dict(exp2)
    exp1_with_step0[0] = step0
    exp2_with_step0[0] = step0
    return exp1_with_step0, exp2_with_step0


def main():
    args = parse_args()
    exp1 = load_results(args.exp1_results)
    exp2 = load_results(args.exp2_results)
    if args.step0_results is not None:
        exp1, exp2 = inject_shared_step0(exp1, exp2, args.step0_results)

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
        # Raw faint trace
        ax.plot(steps, delta, color=color, linewidth=0.3, alpha=RAW_ALPHA)
        # EMA bold overlay
        ax.plot(steps, ema_smooth(delta), color=color, linewidth=1.4, alpha=0.95, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\Delta$F1 (w/ CoT $-$ w/o CoT)")
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_xlim(left=0)

    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
