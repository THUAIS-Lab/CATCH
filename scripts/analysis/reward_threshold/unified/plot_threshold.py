"""Unified-style easy/hard reward-weight threshold plots for Sec. 4
(Fig. fig:reward-threshold).

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
    "hack_rate": "wandb_export_2026-09-08T14_51_16.552+08_00.csv",
    "entropy":   "wandb_export_2026-09-08T14_51_37.742+08_00.csv",
    "reward":    "wandb_export_2026-09-08T14_52_05.427+08_00.csv",
}

Y_LABELS = {
    "hack_rate": "Reward Hacking Rate",
    "entropy":   "Entropy",
    "reward":    "Proxy Reward",
}
OUT_NAMES = {
    "hack_rate": "reward_threshold_hack_rate",
    "entropy":   "reward_threshold_entropy",
    "reward":    "reward_threshold_reward",
}

# Value-column index -> legend label, ordered by increasing easy reward.
# alpha denotes the hard-test weight (w_easy + w_hard = 1). Use the Unicode
# alpha so the glyph is rendered in Liberation Sans like the rest of the
# text instead of mathtext's DejaVu fonts.
RUN_COLUMNS: dict[int, str] = {
    4: "α = 1.0",
    1: "α = 0.9",
    7: "α = 0.7",
}
RUN_NAMES = {
    4: "pc_swe_v4_1-4b_n10k_t1k-32k-bs16-mbs16-n16-0.0-1.0",
    1: "pc_swe_v4_1-4b_n10k_t1k-32k-bs16-mbs16-n16-0.1-0.9",
    7: "pc_swe_v4_1-4b_b_n10k_t1k-32k-bs16-mbs16-n16-no_rej-no_mask",
}
COLORS = {4: BLUE, 1: RED, 7: GREEN}

MAX_STEP = 300


def load_csv(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
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


def plot_metric(metric_key: str) -> None:
    runs = load_csv(DATA_DIR / CSV_MAP[metric_key])

    fig, ax = new_axes(THIRD_WIDTH_IN, THIRD_HEIGHT_IN)
    ymax = 0.0
    for ci, label in RUN_COLUMNS.items():
        steps, values = runs[ci]
        smoothed = plot_series(ax, steps, values, COLORS[ci], label,
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
