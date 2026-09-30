"""Unified-style true-reward (reward_wo_hack) plots for the two Sec. 4
ablations (Figs. fig:threshold-true-reward and fig:reward-threshold-true-reward).

Same data and EMA smoothing as ../plot_true_reward.py, restyled with
analysis/unified_style.py. Labels/colors match the companion hack-rate and
proxy-reward subfigures. Outputs share the paper's figure names.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unified_style import (  # noqa: E402
    BLUE, GREEN, RED, LEGEND_PT_THIRD, THIRD_HEIGHT_IN, THIRD_LABEL_PT, THIRD_TICK_PT, THIRD_WIDTH_IN,
    add_legend, apply_style, fit_legend_headroom, new_axes, plot_series,
    save_fig,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

PLOTS = {
    "toxic_ratio": {
        "csv": DATA_DIR / "diff-toxic-ratio.csv",
        "columns": {"Toxic Ratio 18.2%": 1, "Toxic Ratio 4.5%": 4,
                    "Toxic Ratio 9.1%": 7},
        "label_order": ["Toxic Ratio 4.5%", "Toxic Ratio 9.1%",
                        "Toxic Ratio 18.2%"],
        "colors": {"Toxic Ratio 4.5%": BLUE, "Toxic Ratio 9.1%": RED,
                   "Toxic Ratio 18.2%": GREEN},
        "legend_title": "Toxic ratio",
        "max_step": 250,
        "xticks": [0, 100, 200],
        "output": OUTPUT_DIR / "true_reward_diff_toxic_ratio.pdf",
    },
    "reward_weight": {
        "csv": DATA_DIR / "diff-reward-weight.csv",
        "columns": {"α = 1.0": 4, "α = 0.9": 1, "α = 0.7": 7},
        "label_order": ["α = 1.0", "α = 0.9", "α = 0.7"],
        "colors": {"α = 1.0": BLUE, "α = 0.9": RED, "α = 0.7": GREEN},
        "legend_title": "Easy / Hard",
        "max_step": 300,
        "xticks": [0, 100, 200, 300],
        "output": OUTPUT_DIR / "true_reward_diff_reward_weight.pdf",
    },
}


def load_csv(path: Path, run_columns: dict[str, int], max_step: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


def plot_true_reward(config: dict) -> None:
    runs = load_csv(config["csv"], config["columns"], config["max_step"])

    fig, ax = new_axes(THIRD_WIDTH_IN, THIRD_HEIGHT_IN)
    ymax = 0.0
    for label in config["label_order"]:
        steps, values = runs[label]
        smoothed = plot_series(ax, steps, values, config["colors"][label], label)
        ymax = max(ymax, float(np.nanmax(smoothed)))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("True Reward")
    ax.set_xlim(0, config["max_step"])
    ax.margins(y=0.08)
    leg = add_legend(ax, LEGEND_PT_THIRD)
    fit_legend_headroom(fig, ax, leg, ymax * 1.05)

    save_fig(fig, config["output"])


def main() -> None:
    apply_style(label_pt=THIRD_LABEL_PT, tick_pt=THIRD_TICK_PT)
    for config in PLOTS.values():
        plot_true_reward(config)


if __name__ == "__main__":
    main()
