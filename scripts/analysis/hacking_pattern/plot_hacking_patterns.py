"""
Plot individual hacking patterns (bare_Exception, specific_except_multiple,
raise_with_message, body_has_print) with EMA overlay.

Each pattern gets its own PDF. Reads trajectory JSONs from results/calls.json/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "calls.json"
OUTPUT_DIR = Path(__file__).resolve().parent
PATTERN_GROUP = "error_handling"
PATTERNS = [
    "bare_Exception",
    "specific_except_multiple",
    "raise_with_message",
    "body_has_print",
]

FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.5
EMA_ALPHA = 0.05
RAW_ALPHA = 0.35


def load_trajectory(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")
    with open(path) as f:
        return json.load(f)


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


def plot_pattern(
    pattern_name: str,
    experiments: list[tuple[Path, str, str]],
    output_dir: Path,
    output_suffix: str,
):
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

    for trajectory_path, color, label in experiments:
        traj = load_trajectory(trajectory_path)
        pdata = traj["patterns"][PATTERN_GROUP]
        if pattern_name not in pdata:
            continue
        steps = np.array(traj["steps"])
        values = np.array(pdata[pattern_name], dtype=float)
        # Raw faint trace
        ax.plot(steps, values, color=color, linewidth=0.3, alpha=RAW_ALPHA)
        # EMA bold overlay
        ax.plot(steps, ema_smooth(values), color=color, linewidth=1.4, alpha=0.95,
                label=label)

    # SFT/RL boundary marker
    ax.axvline(x=-0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.4)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(pattern_name.replace("_", " "), fontsize=14)
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, borderpad=0.2)
    ax.set_ylim(-2, 105)

    fig.tight_layout(pad=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pattern_name}{output_suffix}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot selected hacking patterns")
    parser.add_argument(
        "--without-trajectory",
        type=Path,
        default=RESULTS_DIR / "trajectory_n16.json",
    )
    parser.add_argument(
        "--with-trajectory",
        type=Path,
        default=RESULTS_DIR / "trajectory_n16-w_monitor.json",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = [
        (args.without_trajectory, "#2196F3", "w/o Penalty"),
        (args.with_trajectory, "#E91E63", "w/ Penalty"),
    ]
    for pname in PATTERNS:
        plot_pattern(pname, experiments, args.output_dir, args.output_suffix)


if __name__ == "__main__":
    main()
