"""Unified-style CoT monitor performance evolution for Sec. 4
(Fig. fig:monitor-evolution).

Same data as ../plot_monitor_metrics.py, but smoothed with the shared
recursive EMA (alpha=0.05, matching the other Sec. 4 figures) over faint raw
traces instead of a forward running average.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unified_style import (  # noqa: E402
    BLUE, EMA_ALPHA_LIGHT, GREEN, RED, HALF_HEIGHT_IN, HALF_WIDTH_IN,
    LEGEND_PT_HALF, add_legend, apply_style, fit_legend_headroom, new_axes,
    plot_series, save_fig,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

# (metric csv, label, color) in legend order.
SERIES = [
    ("wandb_export_2026-06-29T16_34_13.734+08_00.csv", "Precision", BLUE),
    ("wandb_export_2026-06-29T16_32_40.990+08_00.csv", "Recall", RED),
    ("wandb_export_2026-06-29T16_34_43.091+08_00.csv", "F1", GREEN),
]


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, values = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            steps.append(int(row[0]))
            values.append(float(row[1]))
    return np.array(steps, dtype=int), np.array(values, dtype=float)


def main() -> None:
    apply_style()

    fig, ax = new_axes(HALF_WIDTH_IN, HALF_HEIGHT_IN)
    max_step = 0
    ymax = 0.0
    for filename, label, color in SERIES:
        steps, values = load_csv(DATA_DIR / filename)
        max_step = max(max_step, int(steps.max()))
        smoothed = plot_series(ax, steps, values, color, label,
                            alpha=EMA_ALPHA_LIGHT)
        ymax = max(ymax, float(np.nanmax(smoothed)))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Score")
    ax.set_xlim(0, max_step)
    ax.set_ylim(bottom=0)
    ax.set_yticks([0.0, 0.5, 1.0])
    leg = add_legend(ax, LEGEND_PT_HALF)
    fit_legend_headroom(fig, ax, leg, min(ymax * 1.05, 1.0))

    save_fig(fig, OUTPUT_DIR / "monitor_performance_evolution.pdf")


if __name__ == "__main__":
    main()
