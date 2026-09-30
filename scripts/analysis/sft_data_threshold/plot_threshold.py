"""
Plot reward hacking rate, reward, and entropy across toxic-data thresholds.

Reads wandb-exported CSVs from data/ and produces 3 half-column-ready PDFs.
Includes low, medium, and high toxic-data runs through training step 250.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

CSV_MAP = {
    "hack_rate": "wandb_export_2026-07-01T20_24_50.781+08_00.csv",
    "entropy":   "wandb_export_2026-07-01T20_29_58.794+08_00.csv",
    "reward":    "wandb_export_2026-07-01T20_30_20.721+08_00.csv",
}

HIGH_TOXIC_CSV_MAP = {
    "hack_rate": "wandb_export_2026-09-13T22_00_35.165+08_00.csv",
    "reward": "wandb_export_2026-09-13T22_01_04.534+08_00.csv",
    "entropy": "wandb_export_2026-09-13T22_01_26.619+08_00.csv",
}

Y_LABELS = {
    "hack_rate": "Reward Hacking Rate",
    "entropy":   "Entropy",
    "reward":    "Proxy Reward",
}

# W&B mean-value columns; adjacent __MIN/__MAX columns are not separate runs.
RUN_COLUMNS: dict[str, int] = {
    "Low Toxic (4.5%)": 1,     # t0.5k
    "Medium Toxic (9.1%)": 4,  # t1k
}
HIGH_TOXIC_COLUMNS: dict[str, int] = {"High Toxic (18.2%)": 1}  # t2k

COLORS = {
    "Low Toxic (4.5%)": "#2196F3",     # blue
    "Medium Toxic (9.1%)": "#E91E63",  # red
    "High Toxic (18.2%)": "#4CAF50",   # green
}

FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5
EMA_ALPHA = 0.05
RAW_ALPHA = 0.35
MAX_STEP = 250


def load_csv(path: Path, run_columns: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load nonempty observations through MAX_STEP before smoothing or autoscaling."""
    raw: dict[str, tuple[list[int], list[float]]] = {label: ([], []) for label in run_columns}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            step = int(row[0])
            if step > MAX_STEP:
                continue
            for label, ci in run_columns.items():
                val = row[ci]
                if val:
                    raw[label][0].append(step)
                    raw[label][1].append(float(val))
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
    runs = load_csv(csv_path, RUN_COLUMNS)
    runs.update(load_csv(DATA_DIR / HIGH_TOXIC_CSV_MAP[metric_key], HIGH_TOXIC_COLUMNS))

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
        if metric_key == "reward":
            ax.plot(steps, values, color=COLORS[label], linewidth=0.3, alpha=RAW_ALPHA)
            values = ema_smooth(values)
        ax.plot(steps, values, color=COLORS[label], linewidth=1.4, alpha=0.95,
                label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(Y_LABELS[metric_key])
    # Keep all three labels outside the axes so they do not obscure early curves.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              columnspacing=1.0, handlelength=1.0, handletextpad=0.4, borderpad=0.2)
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
