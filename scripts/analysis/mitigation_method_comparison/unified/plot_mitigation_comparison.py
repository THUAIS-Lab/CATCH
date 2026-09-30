"""Unified-style mitigation-method comparison for Sec. 4
(Fig. fig:mitigation-comparison).

Same data as ../plot_mitigation_comparison.py, but smoothed with the shared
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
    BLUE, EMA_ALPHA_LIGHT, GRAY, GREEN, ORANGE, HALF_HEIGHT_IN,
    HALF_WIDTH_IN, LEGEND_PT_HALF, add_legend, apply_style,
    fit_legend_headroom, new_axes, plot_series, save_fig,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_FILE = "wandb_export_2026-06-29T17_10_30.737+08_00.csv"
MAX_STEP = 85

# (value column, label, color, linestyle) in legend order.
SERIES = [
    (1, r"$\chi^2$ Penalty", BLUE, "-"),
    (7, "Grad Norm Penalty", ORANGE, "-"),
    (10, "LLM CoT Monitor", GREEN, "-"),
    (4, "Baseline (no mitigation)", GRAY, "--"),
]


def load_csv(path: Path, max_step: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    raw: dict[int, tuple[list[int], list[float]]] = {ci: ([], []) for ci, *_ in SERIES}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step = int(row[0])
            if step > max_step:
                break
            for ci, *_ in SERIES:
                val = row[ci]
                if val:
                    raw[ci][0].append(step)
                    raw[ci][1].append(float(val))
    return {k: (np.array(ss, dtype=int), np.array(vv, dtype=float))
            for k, (ss, vv) in raw.items()}


def main() -> None:
    apply_style()
    series = load_csv(DATA_DIR / CSV_FILE, MAX_STEP)

    fig, ax = new_axes(HALF_WIDTH_IN, HALF_HEIGHT_IN)
    ymax = 0.0
    for ci, label, color, linestyle in SERIES:
        steps, values = series[ci]
        smoothed = plot_series(ax, steps, values, color, label,
                               linestyle=linestyle, alpha=EMA_ALPHA_LIGHT)
        ymax = max(ymax, float(np.nanmax(smoothed)))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Hacking Rate")
    ax.set_xlim(0, MAX_STEP)
    ax.set_ylim(bottom=0)
    leg = add_legend(ax, LEGEND_PT_HALF)
    fit_legend_headroom(fig, ax, leg, ymax * 1.05)

    save_fig(fig, OUTPUT_DIR / "mitigation_comparison.pdf")


if __name__ == "__main__":
    main()
