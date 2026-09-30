"""Plot true reward (excluding reward hacking) for two ablations."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

PLOTS = {
    "diff-reward-weight": {
        "csv": DATA_DIR / "diff-reward-weight.csv",
        "columns": {
            "Easy / Hard = 0.0 / 1.0": 4,
            "Easy / Hard = 0.1 / 0.9": 1,
            "Easy / Hard = 0.3 / 0.7": 7,
        },
        "colors": {
            "Easy / Hard = 0.0 / 1.0": "#2196F3",
            "Easy / Hard = 0.1 / 0.9": "#E91E63",
            "Easy / Hard = 0.3 / 0.7": "#4CAF50",
        },
        "max_step": 300,
        "output": OUTPUT_DIR / "true_reward_diff-reward-weight.pdf",
    },
    "diff-toxic-ratio": {
        "csv": DATA_DIR / "diff-toxic-ratio.csv",
        "columns": {
            "High Toxic (18.2%)": 1,
            "Low Toxic (4.5%)": 4,
            "Medium Toxic (9.1%)": 7,
        },
        "colors": {
            "High Toxic (18.2%)": "#4CAF50",
            "Low Toxic (4.5%)": "#2196F3",
            "Medium Toxic (9.1%)": "#E91E63",
        },
        "max_step": 250,
        "output": OUTPUT_DIR / "true_reward_diff-toxic-ratio.pdf",
    },
}

FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5
EMA_ALPHA = 0.05
RAW_ALPHA = 0.35


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


def load_csv(path: Path, run_columns: dict[str, int], max_step: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load only the main-value columns, skipping empty W&B observations."""
    raw = {label: ([], []) for label in run_columns}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step = int(row[0])
            if step > max_step:
                continue
            for label, ci in run_columns.items():
                value = row[ci]
                if value:
                    raw[label][0].append(step)
                    raw[label][1].append(float(value))
    return {label: (np.array(steps, dtype=int), np.array(values, dtype=float))
            for label, (steps, values) in raw.items()}


def plot_true_reward(config: dict):
    runs = load_csv(config["csv"], config["columns"], config["max_step"])

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
    for label, (steps, values) in runs.items():
        ax.plot(steps, values, color=config["colors"][label], linewidth=0.3,
                alpha=RAW_ALPHA)
        ax.plot(steps, ema_smooth(values), color=config["colors"][label],
                linewidth=1.4, alpha=0.95, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("True Reward")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              columnspacing=1.0, handlelength=1.0, handletextpad=0.4,
              borderpad=0.2)
    ax.set_xlim(0, config["max_step"])

    fig.tight_layout(pad=0.3)
    fig.savefig(config["output"], format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {config['output']}")
    plt.close(fig)


def main():
    for config in PLOTS.values():
        plot_true_reward(config)


if __name__ == "__main__":
    main()
