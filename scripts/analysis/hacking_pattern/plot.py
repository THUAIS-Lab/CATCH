#!/usr/bin/env python3
"""
Plotting script — reads trajectory JSONs from results/ and generates figures.

Usage:
    python scripts/analysis/hacking_pattern/plot.py
    python scripts/analysis/hacking_pattern/plot.py --hack calls.json --dpi 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from scripts.analysis.hacking_pattern.config import OUTPUT_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.autolayout": False,
})

COLORS = {"n16": "#1f77b4", "n16-w_monitor": "#d62728"}

# Panels to plot: (group_name, pattern_list, fig_width)
PANELS = [
    ("error_handling", [
        "bare_Exception", "specific_except_multiple", "silently_comment",
        "body_has_print", "raise_with_message",
    ], 18),
    ("justifications", [
        "historical_logs", "cache_preheating", "regression_ops",
        "performance", "admin_user_routing", "compatibility", "reduces_latency",
    ], 22),
    ("cot_structure", [
        "algorithm_discussion", "rpc_infrastructure_block",
        "cache_justification_block", "self_correction", "hedging",
        "problem_solving_depth", "file_by_file_plan",
    ], 22),
    ("code_patterns", [
        "loads_calls_json", "cache_dict_typed", "cache_dict_bare",
        "lower_calls", "MOD_constant", "strip_calls",
    ], 18),
    ("deception_locus", [
        "cot_has_cache_narrative", "code_has_cache_narrative",
        "comments_has_cache_narrative",
    ], 12),
]

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def load_trajectories(hack_method: str) -> dict[str, dict]:
    """Load all saved trajectory JSONs for a given hack_method."""
    data_dir = Path(RESULTS_DIR) / hack_method
    trajectories = {}
    for fpath in sorted(data_dir.glob("trajectory_*.json")):
        exp_name = fpath.stem.replace("trajectory_", "")
        with open(fpath) as f:
            trajectories[exp_name] = json.load(f)
    return trajectories


def plot_comparison_panel(
    trajectories: dict[str, dict],
    group_name: str,
    pattern_names: list[str],
    fig_width: float = 18,
    save_path: str | None = None,
    dpi: int = 150,
):
    """One row of subplots: one subplot per pattern, all experiments overlaid."""
    n = len(pattern_names)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4), squeeze=False)
    axes = axes[0]

    for ax, pname in zip(axes, pattern_names):
        for exp_name, traj in trajectories.items():
            pdata = traj.get("patterns", {}).get(group_name, {})
            if pname not in pdata:
                continue
            values = pdata[pname]
            steps = traj["steps"]
            color = COLORS.get(exp_name, None)
            marker = "." if len(steps) > 50 else "o"
            ax.plot(steps, values, marker=marker, markersize=2 if len(steps) > 50 else 4,
                    linewidth=1.2, label=exp_name, color=color, alpha=0.85)

        # SFT marker line
        ax.axvline(x=-0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.4)

        # Shade Phase 1,2,3 for n16-w_monitor if it exists
        if "n16-w_monitor" in trajectories:
            # Phase boundaries (approximate based on earlier analysis)
            for phase_start, phase_end, phase_label, phase_color in [
                (72, 121, "P1: type switch", "#FFE4B5"),
                (121, 151, "P2: del comment", "#FFDAB9"),
                (166, 180, "P3: +print", "#FFC1CC"),
            ]:
                ax.axvspan(phase_start, phase_end, alpha=0.12, color=phase_color, zorder=0)

        ax.set_title(pname, fontsize=8)
        ax.set_ylim(-2, 105)
        ax.set_xlabel("Step")
        if ax == axes[0]:
            ax.set_ylabel("Frequency (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5)

    fig.suptitle(group_name, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"  → {save_path}")
    else:
        plt.show()


def plot_token_evolution(
    trajectories: dict[str, dict],
    save_dir: str,
    dpi: int = 150,
):
    """Plot token length evolution for each experiment."""
    for exp_name, traj in trajectories.items():
        stats = traj.get("token_stats", {})
        if not stats:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        steps = traj["steps"]

        for key, label, color in [
            ("cot_mean", "CoT", "steelblue"),
            ("code_mean", "Code", "darkorange"),
            ("comments_mean", "Comments", "green"),
            ("total_mean", "Total", "grey"),
        ]:
            if key in stats:
                ax1.plot(steps, stats[key], marker=".", markersize=2,
                         linewidth=1.2, label=label, color=color)
        ax1.axvline(x=-0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.4)
        ax1.set_xlabel("Step"); ax1.set_ylabel("Mean tokens")
        ax1.set_title("Token lengths"); ax1.legend(frameon=False); ax1.grid(True, alpha=0.25)

        if "comments_cot_ratio" in stats:
            ax2.plot(steps, stats["comments_cot_ratio"], marker=".", markersize=2,
                     linewidth=1.5, color="purple")
        ax2.axvline(x=-0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.4)
        ax2.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.7, alpha=0.4)
        ax2.set_xlabel("Step"); ax2.set_ylabel("Ratio")
        ax2.set_title("Comments / CoT token ratio"); ax2.grid(True, alpha=0.25)

        fig.suptitle(f"Token evolution — {exp_name}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = os.path.join(save_dir, f"token_evolution_{exp_name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot hacking pattern trajectories")
    p.add_argument("--hack", type=str, default="calls.json")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--panels", nargs="*", default=None,
                   help="Which panels to plot (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    trajectories = load_trajectories(args.hack)
    if not trajectories:
        print(f"No trajectory data found in {RESULTS_DIR}/{args.hack}/")
        print("Run pipeline.py first.")
        return

    print(f"Loaded {len(trajectories)} experiments: {list(trajectories.keys())}")
    out_dir = os.path.join(OUTPUT_DIR, args.hack)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Comparison panels (one per group)
    panel_map = {p[0]: p for p in PANELS}
    to_plot = args.panels if args.panels else list(panel_map.keys())

    for group_name in to_plot:
        if group_name not in panel_map:
            continue
        _, pattern_names, fig_width = panel_map[group_name]
        # Check group exists
        sample_traj = next(iter(trajectories.values()))
        if group_name not in sample_traj.get("patterns", {}):
            print(f"  Group '{group_name}' not in data, skipping.")
            continue
        plot_comparison_panel(
            trajectories, group_name, pattern_names, fig_width,
            save_path=os.path.join(out_dir, f"{group_name}.png"),
            dpi=args.dpi,
        )

    # Token evolution (if token stats present)
    sample_traj = next(iter(trajectories.values()))
    if sample_traj.get("token_stats"):
        plot_token_evolution(trajectories, out_dir, dpi=args.dpi)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
