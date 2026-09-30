"""
Plot LLM CoT Monitor performance metrics (precision, recall, f1) with
running-average smoothing over training steps.

Reads wandb-exported CSVs from data/ and produces a single figure.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
WINDOW_SIZE = 10

# Large canvas → scale down in LaTeX for dense grid + tiny legend.
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5

# Map metric suffix -> readable label
METRIC_FILES = {
    "recall": "wandb_export_2026-06-29T16_32_40.990+08_00.csv",
    "precision": "wandb_export_2026-06-29T16_34_13.734+08_00.csv",
    "f1": "wandb_export_2026-06-29T16_34_43.091+08_00.csv",
}

METRIC_LABEL = {
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load steps and metric values from a wandb-export CSV."""
    steps = []
    values = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _header = next(reader)  # skip header
        for row in reader:
            steps.append(int(row[0]))
            values.append(float(row[1]))
    return np.array(steps, dtype=int), np.array(values, dtype=float)


def running_average_forward(data: np.ndarray, window: int) -> np.ndarray:
    """
    Forward-looking running average.

    For index i, the smoothed value is mean(data[i : i + window]).
    The window shrinks naturally when fewer than *window* points remain.
    """
    n = len(data)
    smoothed = np.empty(n)
    for i in range(n):
        end = min(i + window, n)
        smoothed[i] = np.mean(data[i:end])
    return smoothed


def main():
    # Use a style that ships with matplotlib — no seaborn dependency required
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set up LaTeX-style font rendering for paper-quality PDF output
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

    for metric_key, filename in METRIC_FILES.items():
        path = DATA_DIR / filename
        steps, values = load_csv(path)
        smoothed = running_average_forward(values, WINDOW_SIZE)
        ax.plot(
            steps,
            smoothed,
            label=METRIC_LABEL[metric_key],
            linewidth=1.4,
            alpha=0.9,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)

    fig.tight_layout(pad=0.3)

    out_path = Path(__file__).resolve().parent / "monitor_performance_evolution.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
