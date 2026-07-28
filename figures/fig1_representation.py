"""Fig 1 -- from a multivariate time series to a graph whose edges are K-vectors.

Reading order a -> b -> c, one idea per panel:
  (a) the data: M channels, T samples
  (b) pyspi: K named pairwise statistics, each a full M x M matrix
  (c) the same stack read as graphs -- so an edge is a K-VECTOR, not a scalar

The figure deliberately stops before any learning. `w`, softplus and top-d
sparsification belong to the method figure; at A0 viewing distance a panel gets
about three seconds, and the K-vector edge is already one complete idea.

    python figures/fig1_representation.py     -> figures/out/fig1.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures import style
from figures.fig1_data import series, spi_matrices

KEYS = ["cov", "lag1", "coh"]
TITLES = [r"$|\rho_{ij}|$", r"$|\rho_{ij}(\tau{=}1)|$", r"$\mathrm{coh}_{ij}$"]
SKEW, DX, DY = -24, 2.75, 2.05
ALPHA = [1.0, 0.86, 0.72]          # depth cue, mild enough that hue survives


def panel_a(ax, x):
    im = ax.imshow(x, aspect="auto", cmap=style.DIVERGING, vmin=-2.4, vmax=2.4,
                   interpolation="nearest")
    for k in range(1, x.shape[0]):        # rows are a SET of series, not an image
        ax.axhline(k - 0.5, color=style.PAPER, lw=1.4)
    ax.set_yticks(range(x.shape[0]))
    ax.set_yticklabels([f"$c_{i+1}$" for i in range(x.shape[0])], fontsize=6.5)
    ax.set_xticks([0, x.shape[1] - 1]); ax.set_xticklabels(["0", "$T$"], fontsize=6.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=1.5)
    return im


def _deck(ax, d):
    return Affine2D().skew_deg(SKEW, 0).translate(d * DX, d * DY) + ax.transData


def panel_b(ax, S, M):
    for d, k in enumerate(KEYS):
        tr = _deck(ax, d)
        im = ax.imshow(S[k], cmap=style.SEQUENTIAL, extent=(0, M, 0, M),
                       interpolation="nearest", vmin=0, vmax=1,
                       zorder=10 - d, alpha=ALPHA[d])
        im.set_transform(tr)
        ax.add_patch(plt.Rectangle((0, 0), M, M, fill=False, lw=0.8,
                                   edgecolor=style.LAYERS[d], zorder=10 - d,
                                   transform=tr))


def panel_c(ax, S, M):
    # Pentagon layout: all 10 pairs get a visible, non-overlapping chord. A
    # left-to-right chain would hide the pairs the vocabulary is about.
    ang = np.pi / 2 + np.arange(M) * 2 * np.pi / M
    P = np.c_[M / 2 + 1.75 * np.cos(ang), M / 2 + 1.75 * np.sin(ang)]
    for d, k in enumerate(KEYS):
        tr = _deck(ax, d)
        A = np.nan_to_num(S[k])
        v = A[np.triu_indices(M, 1)]
        lo, hi = v.min(), v.max()
        ax.add_patch(plt.Rectangle((0, 0), M, M, fill=False, lw=0.8,
                                   edgecolor=style.LAYERS[d], zorder=10 - d,
                                   transform=tr, alpha=0.45 * ALPHA[d]))
        for i in range(M):
            for j in range(i + 1, M):
                w = (A[i, j] - lo) / (hi - lo + 1e-9)
                # Steep exponent: at a gentler one all 10 chords stay visible and
                # three superimposed decks read as noise.
                ax.plot(P[[i, j], 0], P[[i, j], 1], transform=tr,
                        color=style.LAYERS[d], lw=0.25 + 2.9 * w ** 2.4,
                        alpha=ALPHA[d] * (0.10 + 0.90 * w ** 1.8),
                        zorder=10 - d, solid_capstyle="round")
        for i in range(M):
            ax.add_patch(plt.Circle(P[i], 0.16, facecolor=style.PAPER,
                                    edgecolor=style.INK, lw=0.55,
                                    zorder=10 - d + 0.5, transform=tr,
                                    alpha=ALPHA[d]))


def build():
    style.use()
    x = series()
    S = spi_matrices(x)
    M = x.shape[0]

    fig = plt.figure(figsize=(7.2, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15],
                          wspace=0.42, left=0.055, right=0.985,
                          bottom=0.06, top=0.80)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1]); style.bare(ax_b)
    ax_c = fig.add_subplot(gs[2]); style.bare(ax_c)

    im = panel_a(ax_a, x)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.035, pad=0.03, ticks=[-2.4, 0, 2.4])
    cb.ax.set_yticklabels(["−2", "0", "+2"], fontsize=6.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, pad=1.5)

    panel_b(ax_b, S, M)
    panel_c(ax_c, S, M)
    for ax in (ax_b, ax_c):
        ax.set_xlim(-3.9, 12.6); ax.set_ylim(-0.6, 10.4)

    # Legend once, on (b); (c) inherits it by colour and position.
    for d, t in enumerate(TITLES):
        ly = 9.9 - 1.15 * d
        ax_b.add_patch(plt.Rectangle((-3.7, ly - 0.20), 0.5, 0.40,
                                     facecolor=style.LAYERS[d],
                                     edgecolor="none", zorder=20))
        ax_b.text(-3.0, ly, t, fontsize=6.8, va="center", zorder=20)
    # Depth marker on (b) only: repeating it on (c) says "another K", not "the
    # same K seen differently", which is the opposite of the panel's point.
    ax_b.text(9.6, 7.9, "...", fontsize=12, color=style.GREY,
              ha="center", va="center", rotation=36)
    ax_b.text(10.3, 8.7, r"$K \approx 300$", fontsize=7, color=style.GREY,
              ha="left", va="center")

    # Titles and letters placed in FIGURE coordinates: the three axes have
    # different heights (a carries a colorbar), so per-axes titles do not align.
    for ax, letter, title in ((ax_a, "a", "multivariate time series"),
                              (ax_b, "b", "$K$ pairwise statistics"),
                              (ax_c, "c", "one graph per statistic")):
        x0 = ax.get_position().x0
        fig.text(x0 - 0.035, 0.965, letter, fontsize=9.5, fontweight="bold",
                 va="top", ha="left")
        fig.text(x0, 0.905, title, fontsize=8, va="top", ha="left")

    for x0, x1, lab in ((0.335, 0.395, "pyspi"), (0.655, 0.700, "")):
        fig.add_artist(FancyArrowPatch((x0, 0.44), (x1, 0.44),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8,
                                       lw=0.9, color=style.ACCENT,
                                       shrinkA=0, shrinkB=0))
        if lab:
            fig.text((x0 + x1) / 2, 0.50, lab, fontsize=7,
                     color=style.ACCENT, ha="center", va="bottom")

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig1.png")
    fig.savefig(out / "fig1.pdf")
    print("wrote figures/out/fig1.{png,pdf}")


if __name__ == "__main__":
    build()
