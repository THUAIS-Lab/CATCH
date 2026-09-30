"""Unified-style toxic-ratio threshold plots for Sec. 4 (Fig. fig:toxic-threshold).

Same data and EMA smoothing as ../plot_threshold.py, restyled with
analysis/unified_style.py. Outputs share the paper's figure names.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unified_style import (  # noqa: E402
    BLUE, EMA_ALPHA_LIGHT, GREEN, RED, LEGEND_PT_THIRD, THIRD_HEIGHT_IN,
    THIRD_LABEL_PT, THIRD_TICK_PT, THIRD_WIDTH_IN, add_legend, apply_style, fit_legend_headroom, new_axes,
    plot_series, save_fig,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
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
OUT_NAMES = {
    "hack_rate": "threshold_hack_rate",
    "entropy":   "threshold_entropy",
    "reward":    "threshold_reward",
}

# (csv map, {legend label: value column}); W&B mean columns only.
RUN_SOURCES = [
    (CSV_MAP, {"Toxic Ratio 4.5%": 1, "Toxic Ratio 9.1%": 4}),   # t0.5k, t1k
    (HIGH_TOXIC_CSV_MAP, {"Toxic Ratio 18.2%": 1}),  # t2k
]
LABEL_ORDER = ["Toxic Ratio 4.5%", "Toxic Ratio 9.1%", "Toxic Ratio 18.2%"]
COLORS = {"Toxic Ratio 4.5%": BLUE, "Toxic Ratio 9.1%": RED,
         "Toxic Ratio 18.2%": GREEN}

MAX_STEP = 250


def load_csv(path: Path, run_columns: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw: dict[str, tuple[list[int], list[float]]] = {label: ([], []) for label in run_columns}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
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


def plot_metric(metric_key: str) -> None:
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for csv_map, run_columns in RUN_SOURCES:
        runs.update(load_csv(DATA_DIR / csv_map[metric_key], run_columns))

    fig, ax = new_axes(THIRD_WIDTH_IN, THIRD_HEIGHT_IN)
    ymax = 0.0
    for label in LABEL_ORDER:
        steps, values = runs[label]
        smoothed = plot_series(ax, steps, values, COLORS[label], label,
                            alpha=EMA_ALPHA_LIGHT)
        ymax = max(ymax, float(np.nanmax(smoothed)))

    ax.set_xlabel("Training Step")
    ax.set_ylabel(Y_LABELS[metric_key])
    ax.set_xlim(0, MAX_STEP)
    ax.margins(y=0.08)
    leg = add_legend(ax, LEGEND_PT_THIRD)
    fit_legend_headroom(fig, ax, leg, ymax * 1.05)

    save_fig(fig, OUTPUT_DIR / f"{OUT_NAMES[metric_key]}.pdf")


def main() -> None:
    apply_style(label_pt=THIRD_LABEL_PT, tick_pt=THIRD_TICK_PT)
    for key in ["hack_rate", "reward", "entropy"]:
        plot_metric(key)


if __name__ == "__main__":
    main()
