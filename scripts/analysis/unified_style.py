"""Unified plotting style for the Sec. 4 analysis figures.

Addresses advisor feedback (2026-09-24):
- Figure text matches the paper body size. The body font is 10pt
  (iclr2027_conference.sty, \\normalsize = 10pt); axis/tick text here is 9pt
  and the smallest text (legends of the 0.32\\textwidth subfigures) is 8pt,
  i.e. never more than 2pt below the body size.
- Sans-serif fonts (Liberation Sans, embedded as TrueType, pdf.fonttype=42).
- Thicker, crisper lines: figures are produced at their final physical size
  in the LaTeX document (0.32/0.48 of a 5.5in text width) instead of being
  drawn large and scaled down, so strokes do not shrink in print.
- No grid lines; curves are EMA-smoothed over faint raw traces.
- One legend style shared by every figure, always placed at the upper left.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Muted, clearly distinguishable palette (reference-figure style):
# steel blue / brick red / sage green / tan / neutral gray.
BLUE = "#5B84A8"
RED = "#C1666B"
GREEN = "#7FA65A"
ORANGE = "#D9A05B"
GRAY = "#7F7F7F"

# Original canvas geometry (same as the pre-unification scripts): a large
# 7.0 x 4.5in canvas that LaTeX scales into the subfigure boxes, keeping the
# original text-to-plot proportions. Only the font family (sans-serif),
# curve weight (slightly thicker), grid (off), palette, and unified
# upper-left legend differ from the original figures.
TEXTWIDTH_IN = 5.5
THIRD_WIDTH_IN = 7.0
HALF_WIDTH_IN = 7.0
THIRD_HEIGHT_IN = 4.5
HALF_HEIGHT_IN = 4.5

LABEL_PT = 14        # axes labels (as in the original scripts)
TICK_PT = 12         # tick labels (as in the original scripts)
LEGEND_PT_HALF = 12
# Three-across subfigures shrink more in print, so their text goes one step up.
THIRD_LABEL_PT = 18
THIRD_TICK_PT = 14
LEGEND_PT_THIRD = 16

EMA_ALPHA = 0.05        # heavy smoothing for noisy curves (true reward, delta recall)
EMA_ALPHA_LIGHT = 0.15  # light smoothing for low-noise curves (hack rate, proxy reward,
                        # mitigation comparison, monitor metrics)
RAW_LW = 0.5         # slightly thicker than the original 0.3
RAW_ALPHA = 0.25
EMA_LW = 1.8         # slightly thicker than the original 1.4
AXES_LW = 0.8


def apply_style(label_pt: int = LABEL_PT, tick_pt: int = TICK_PT) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": tick_pt,
        "axes.labelsize": label_pt,
        "xtick.labelsize": tick_pt,
        "ytick.labelsize": tick_pt,
        "axes.linewidth": AXES_LW,
        "xtick.major.width": AXES_LW,
        "ytick.major.width": AXES_LW,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "pdf.fonttype": 42,  # embed TrueType instead of Type 3
    })


def new_axes(width_in: float, height_in: float):
    # Full thin axis box, as in the reference figure.
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    return fig, ax


def ema_smooth(values: np.ndarray, alpha: float = EMA_ALPHA,
               initial_steps: int = 1) -> np.ndarray:
    """Recursive EMA; with initial_steps > 1, start at that index with the
    arithmetic mean of the first initial_steps observations."""
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size == 0:
        return out
    if initial_steps > 1:
        if len(values) < initial_steps or not np.isfinite(values[:initial_steps]).all():
            raise ValueError("Mean initialization requires N finite observations")
        first = initial_steps - 1
        out[first] = values[:initial_steps].mean()
    else:
        first = int(valid[0])
        out[first] = values[first]
    for i in range(first + 1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def plot_series(ax, steps: np.ndarray, values: np.ndarray, color: str,
                label: str, *, smooth: bool = True, raw: bool = True,
                linestyle: str = "-", initial_steps: int = 1,
                alpha: float = EMA_ALPHA) -> np.ndarray:
    """Plot a faint raw trace with a bold EMA-smoothed curve on top."""
    if raw:
        ax.plot(steps, values, color=color, linewidth=RAW_LW, alpha=RAW_ALPHA,
                zorder=2, rasterized=False)
    y = ema_smooth(values, alpha=alpha,
                   initial_steps=initial_steps) if smooth else values
    ax.plot(steps, y, color=color, linewidth=EMA_LW, linestyle=linestyle,
            zorder=3, label=label)
    return y


def add_legend(ax, fontsize: int, title: str | None = None, **kwargs):
    """The one shared legend style: upper left, white background."""
    leg = ax.legend(
        loc="upper left", fontsize=fontsize, title=title,
        frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.75",
        fancybox=False,
        borderpad=0.3, borderaxespad=0.3, labelspacing=0.3,
        handlelength=1.5, handletextpad=0.5, **kwargs,
    )
    if title is None:
        # matplotlib always packs an (empty) title TextArea as the first
        # legend row; hide it so it takes no vertical space.
        leg._legend_box.get_children()[0].set_visible(False)
    return leg


def fit_legend_headroom(fig, ax, leg, ymax_data: float,
                        pad_frac: float = 0.05) -> None:
    """Expand the y-range upward until the legend fits above ymax_data.

    Measures the drawn legend box and grows only the top of the y-range so
    the opaque legend never covers the smoothed curves.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_height = leg.get_window_extent(renderer=renderer).height
    ax_height = ax.get_window_extent(renderer=renderer).height
    leg_frac = leg_height / ax_height
    usable = 1.0 - leg_frac - pad_frac
    if usable <= 0.2:
        usable = 0.2
    y0, _ = ax.get_ylim()
    ax.set_ylim(y0, y0 + (ymax_data - y0) / usable)


def save_fig(fig, path: Path, *, also_png: bool = True) -> None:
    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.015)
    if also_png:
        fig.savefig(path.with_suffix(".png"), format="png", dpi=300,
                    bbox_inches="tight", pad_inches=0.015)
    print(f"Saved {path}")
    plt.close(fig)
