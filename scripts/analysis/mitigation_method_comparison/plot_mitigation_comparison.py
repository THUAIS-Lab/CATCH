"""
Plot nontrivial hack rate under four mitigation strategies with
running-average smoothing over training steps (steps ≤ 85 only).

Reads wandb-exported CSV from data/ and produces a half-column PDF.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = "wandb_export_2026-06-29T17_10_30.737+08_00.csv"
WINDOW_SIZE = 10
MAX_STEP = 85

# Large canvas → scale down in LaTeX for dense grid + tiny legend.
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5

# Column indices (0-based, excluding Step) → LaTeX label
METRIC_COLUMNS: dict[int, str] = {
    1: r"$\chi^2$ Penalty",       # chi2
    7: "Grad Norm Penalty",         # gr1e-2
    10: "LLM CoT Monitor",          # w_monitor
    4: "Baseline",                  # no mitigation (last in legend)
}


def load_csv(path: Path, max_step: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load each metric column independently, skipping rows where it is empty."""
    raw: dict[int, tuple[list[int], list[float]]] = {k: ([], []) for k in METRIC_COLUMNS}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            step = int(row[0])
            if step > max_step:
                break
            for ci in METRIC_COLUMNS:
                val = row[ci]
                if val:
                    raw[ci][0].append(step)
                    raw[ci][1].append(float(val))
    return {k: (np.array(ss, dtype=int), np.array(vv, dtype=float)) for k, (ss, vv) in raw.items()}


def running_average_forward(data: np.ndarray, window: int) -> np.ndarray:
    """Forward-looking running average with shrinking tail window."""
    n = len(data)
    smoothed = np.empty(n)
    for i in range(n):
        end = min(i + window, n)
        smoothed[i] = np.mean(data[i:end])
    return smoothed


def main():
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

    path = DATA_DIR / CSV_FILE
    series = load_csv(path, MAX_STEP)

    for ci, label in METRIC_COLUMNS.items():
        steps, vals = series[ci]
        smoothed = running_average_forward(vals, WINDOW_SIZE)
        ax.plot(steps, smoothed, label=label, linewidth=1.4, alpha=0.9)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Hacking Rate")
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_ylim(0, None)
    ax.set_xlim(left=0)

    fig.tight_layout(pad=0.3)

    out_path = Path(__file__).resolve().parent / "mitigation_comparison.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
