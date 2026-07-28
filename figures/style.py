"""Shared figure style. Import this, never set rcParams in a figure script.

One rule, three colour roles, so the poster cannot drift:
  signed quantities  -> ONE diverging map, centred at zero (z-scored series)
  magnitudes         -> ONE sequential map (SPI values, |w|)
  emphasis           -> ONE accent colour, used nowhere else

Both maps are colour-vision-deficiency safe. Two sequential maps in one poster
is the most common consistency failure; there is deliberately only one here.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

INK = "#16161a"
GREY = "#8c8c94"
FAINT = "#d7d7dd"
PAPER = "#ffffff"
ACCENT = "#c1440e"          # emphasis only: arrows, callouts, the one edge

DIVERGING = "RdBu_r"        # signed, centred at 0
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "spi", ["#f2f5f7", "#a8c0cf", "#4a7fa5", "#1d3f5c", "#0b1c2a"])

# Motif colours. Neutral for the two that a symmetric statistic CONFUSES
# (chain, collider) and the accent for the one it separates (fork), so the
# colour encodes the measured finding rather than decorating it.
MOTIF = {"chain": "#16161a", "collider": "#7b7b88", "fork": "#c1440e"}

# Model palette. The proposed model gets the accent; every baseline is a
# neutral, so the eye finds the method without the figure arguing for it.
MODELS = {
    "spi-mpnn":        ACCENT,
    "fixed-spi":       "#1d3f5c",
    "correlation":     "#4a7fa5",
    "latent":          "#8fb0c4",
    "latent-directed": "#5d7c8a",
    "node-only":       "#b9b9c2",
    "shuffled":        "#8a7a5c",
}

# Layer tints for the K-stack: the SAME three colours identify the same three
# SPIs in every panel, which is what makes the matrix stack and the graph stack
# read as the same object.
LAYERS = ["#1d3f5c", "#4a7fa5", "#a8c0cf"]


def use() -> None:
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "regular",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "text.color": INK,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
    })


def bare(ax) -> None:
    """Strip an axes to nothing -- for schematics, where axes are furniture."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
