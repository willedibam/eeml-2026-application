"""Fig 1, panel (c): the same stack, read as graphs.

Deliberately the same geometry, the same three layer colours and the same depth
cue as panel (b). The point of the rhyme: a stack of K matrices IS a stack of K
weighted graphs, so an edge in this model is not a scalar but a K-vector. Every
later figure depends on the reader having accepted that here.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures import style
from figures.fig1_data import series, spi_matrices

style.use()
x = series()
S = spi_matrices(x)
keys = ["cov", "lag1", "coh"]
M = x.shape[0]

# Pentagon layout: every one of the 10 pairs gets a visible, non-overlapping
# chord. A left-to-right chain would hide the pairs the vocabulary is about.
ang = np.pi / 2 + np.arange(M) * 2 * np.pi / M
P = np.c_[2.5 + 1.75 * np.cos(ang), 2.5 + 1.75 * np.sin(ang)]

fig, ax = plt.subplots(figsize=(3.3, 2.05))
style.bare(ax)
SKEW, DX, DY = -24, 2.75, 2.05
ALPHA = [1.0, 0.86, 0.72]

for d, k in enumerate(keys):
    z = d
    tr = Affine2D().skew_deg(SKEW, 0).translate(z * DX, z * DY) + ax.transData
    A = np.nan_to_num(S[k])
    v = A[np.triu_indices(M, 1)]
    lo, hi = v.min(), v.max()
    ax.add_patch(plt.Rectangle((0, 0), M, M, fill=False, lw=0.8,
                               edgecolor=style.LAYERS[d], zorder=10 - z,
                               transform=tr, alpha=0.45 * ALPHA[d]))
    for i in range(M):
        for j in range(i + 1, M):
            w = (A[i, j] - lo) / (hi - lo + 1e-9)
            ax.plot(P[[i, j], 0], P[[i, j], 1], transform=tr,
                    color=style.LAYERS[d], lw=0.25 + 2.9 * w ** 2.4,
                    alpha=ALPHA[d] * (0.10 + 0.90 * w ** 1.8), zorder=10 - z,
                    solid_capstyle="round")
    for i in range(M):
        ax.add_patch(plt.Circle(P[i], 0.16, facecolor=style.PAPER,
                                edgecolor=style.INK, lw=0.55,
                                zorder=10 - z + 0.5, transform=tr,
                                alpha=ALPHA[d]))

for d, k in enumerate(keys):
    ly = 7.4 - 1.1 * d
    ax.add_patch(plt.Rectangle((-8.2, ly - 0.19), 0.46, 0.38,
                               facecolor=style.LAYERS[d], edgecolor="none",
                               zorder=20))
ax.text(-7.55, 7.4, r"$|\rho_{ij}|$", fontsize=7, va="center", zorder=20)
ax.text(-7.55, 6.3, r"$|\rho_{ij}(\tau{=}1)|$", fontsize=7, va="center", zorder=20)
ax.text(-7.55, 5.2, r"$\mathrm{coh}_{ij}$", fontsize=7, va="center", zorder=20)

ax.text(11.3, 11.4, "...", fontsize=13, color=style.GREY,
        ha="center", va="center", rotation=36)
ax.text(12.4, 12.1, r"$K \approx 300$", fontsize=7.5, color=style.GREY,
        ha="left", va="center")
ax.set_xlim(-8.4, 17.5)
ax.set_ylim(-0.7, 13.0)
ax.set_title("one graph per statistic", pad=2, fontsize=8, loc="left", x=0.02)
fig.savefig(Path(__file__).parent / "p1c.png")
print("wrote p1c.png")
