"""
Plot reward hacking rate, reward, and entropy across easy/hard reward weights.

Reads wandb-exported CSVs from data/ and produces 3 PDFs using the
same styling and reward-only EMA as sft_data_threshold/plot_threshold.py.

The baseline uses easy/hard rewards of 0.3/0.7 and format_reward=0.1;
the 0.0/1.0 and 0.1/0.9 scripts use format_reward=0.0. Scores are plotted
as exported, without adjusting for this additional reward difference.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

CSV_MAP = {
    "hack_rate": "wandb_export_2026-09-08T14_51_16.552+08_00.csv",
    "entropy":   "wandb_export_2026-09-08T14_51_37.742+08_00.csv",
    "reward":    "wandb_export_2026-09-08T14_52_05.427+08_00.csv",
}

Y_LABELS = {
    "hack_rate": "Reward Hacking Rate",
    "entropy":   "Entropy",
    "reward":    "Proxy Reward",
}

# Each W&B run occupies value, __MIN, and __MAX columns.
# Plot only the main values, ordered by increasing easy reward.
RUN_COLUMNS: dict[int, str] = {
    4: "Easy / Hard = 0.0 / 1.0",
    1: "Easy / Hard = 0.1 / 0.9",
    7: "Easy / Hard = 0.3 / 0.7",
}

# Validate run identities rather than silently assigning a label to a
# different run if a future export changes its column order.
RUN_NAMES = {
    4: "pc_swe_v4_1-4b_n10k_t1k-32k-bs16-mbs16-n16-0.0-1.0",
    1: "pc_swe_v4_1-4b_n10k_t1k-32k-bs16-mbs16-n16-0.1-0.9",
    7: "pc_swe_v4_1-4b_b_n10k_t1k-32k-bs16-mbs16-n16-no_rej-no_mask",
}

COLORS = {
    4: "#2196F3",  # blue
    1: "#E91E63",  # red
    7: "#4CAF50",  # green
}

FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5
EMA_ALPHA = 0.05
RAW_ALPHA = 0.35
MAX_STEP = 300


def load_csv(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load each run column independently, skipping empty rows."""
    raw: dict[int, tuple[list[int], list[float]]] = {k: ([], []) for k in RUN_COLUMNS}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for ci, run_name in RUN_NAMES.items():
            if header[ci].partition(" - ")[0] != run_name:
                raise ValueError(f"Unexpected run in {path.name}, column {ci}: {header[ci]}")
        for row in reader:
            step = int(row[0])
            if step > MAX_STEP:
                continue
            for ci in RUN_COLUMNS:
                val = row[ci]
                if val:
                    raw[ci][0].append(step)
                    raw[ci][1].append(float(val))
    return {k: (np.array(ss, dtype=int), np.array(vv, dtype=float))
            for k, (ss, vv) in raw.items()}


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


def plot_metric(metric_key: str):
    csv_path = DATA_DIR / CSV_MAP[metric_key]
    runs = load_csv(csv_path)

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

    if metric_key == "reward":
        for ci, label in RUN_COLUMNS.items():
            steps, values = runs[ci]
            ax.plot(steps, values, color=COLORS[ci], linewidth=0.3, alpha=RAW_ALPHA)
            ax.plot(steps, ema_smooth(values), color=COLORS[ci], linewidth=1.4, alpha=0.95,
                    label=label)
    else:
        for ci, label in RUN_COLUMNS.items():
            steps, values = runs[ci]
            ax.plot(steps, values, color=COLORS[ci], linewidth=1.4, alpha=0.95,
                    label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(Y_LABELS[metric_key])
    # Keep legends in the empty region of each metric to avoid covering curves.
    legend_loc = {"hack_rate": "upper left", "reward": "lower right", "entropy": "upper right"}
    ax.legend(loc=legend_loc[metric_key], handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_xlim(0, MAX_STEP)

    fig.tight_layout(pad=0.3)

    out_path = OUTPUT_DIR / f"{metric_key}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    for key in ["hack_rate", "reward", "entropy"]:
        plot_metric(key)


if __name__ == "__main__":
    main()
