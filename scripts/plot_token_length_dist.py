#!/usr/bin/env python3
"""Plot binned cot_length and answer_length distributions per hack_method and full.

Each chart uses fixed-width bins whose size adapts to the data's own spread
(Freedman-Diaconis rule). Only the P0–P99 range is plotted; the long tail is
clipped. Stats are computed on the full dataset.

Individual PNGs are saved first, then concatenated into a single PDF.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

INPUT_PATH = "/data/nvme0/wangsl/datasets/compare/length_analysis_merged.parquet"
OUTPUT_DIR = "examples/reward_hack_sft/data/token_length_distributions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)


def compute_bins(data: np.ndarray) -> np.ndarray:
    """Return fixed-width bin edges adapted to data spread (P0-P99) via Freedman-Diaconis."""
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.array([0, 1])

    p99 = np.percentile(data, 99)
    clipped = data[data <= p99]
    if len(clipped) < 2:
        return np.array([data.min(), p99])

    q1, q3 = np.percentile(clipped, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        iqr = 1
    bw = 2 * iqr / (len(clipped) ** (1 / 3))
    bw = _nice_number(bw)
    dmin, dmax = clipped.min(), clipped.max()
    n_bins = max(5, int(np.ceil((dmax - dmin) / bw)))
    return np.linspace(dmin, dmin + n_bins * bw, n_bins + 1)


def _nice_number(x: float) -> float:
    if x <= 0:
        return 1
    exp = 10 ** np.floor(np.log10(x))
    mant = x / exp
    if mant <= 1:
        nice = 1
    elif mant <= 2:
        nice = 2
    elif mant <= 5:
        nice = 5
    else:
        nice = 10
    return nice * exp


def plot_side_by_side(data: pd.DataFrame, title: str, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, color, xlabel in [
        (ax1, "cot_length", "#4C72B0", "CoT Length (tokens)"),
        (ax2, "answer_length", "#55A868", "Answer Length (tokens)"),
    ]:
        vals = data[col].dropna().values.astype(float)
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{col} Distribution")
            continue

        # Full-data stats
        p99 = np.percentile(vals, 99)
        stats_text = (
            f"N={len(vals):,}\n"
            f"Min={vals.min():.0f}\n"
            f"Max={vals.max():.0f}\n"
            f"Mean={vals.mean():.0f}\n"
            f"Std={vals.std():.0f}\n"
            f"Median={np.median(vals):.0f}\n"
            f"P99={p99:.0f}"
        )

        # Plot only up to P99
        plot_vals = vals[vals <= p99]
        edges = compute_bins(vals)
        widths = np.diff(edges)
        counts, _ = np.histogram(plot_vals, bins=edges)

        ax.bar(
            edges[:-1], counts, width=widths, align="edge",
            color=color, edgecolor="white", linewidth=0.3, alpha=0.85,
        )

        ax.set_xlim(edges[0], edges[-1])
        stats_text += f"\nBin width={widths[0]:.0f}"

        ax.text(
            0.97, 0.97, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --- Per hack_method ---
png_paths = []
print("Plotting per hack_method ...")
for method in sorted(df["hack_method"].unique()):
    sub = df[df["hack_method"] == method]
    safe_name = str(method).replace("/", "_").replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"token_len_dist_{safe_name}.png")
    plot_side_by_side(sub, f"hack_method = {method}  (n={len(sub):,})", path)
    png_paths.append(path)

# --- Full dataset ---
print("\nPlotting full dataset ...")
path_full = os.path.join(OUTPUT_DIR, "token_len_dist_full.png")
plot_side_by_side(df, f"All Data  (n={len(df):,})", path_full)

# --- Concatenate PNGs into single PDF (full first, then per-method sorted) ---
print("\nAssembling PDF ...")
pdf_path = os.path.join(OUTPUT_DIR, "token_len_dist.pdf")
images = [Image.open(p).convert("RGB") for p in ([path_full] + png_paths)]
images[0].save(pdf_path, save_all=True, append_images=images[1:])
print(f"  Saved {pdf_path} ({len(images)} pages)")

print("\nDone.")
