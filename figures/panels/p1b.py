"""Fig 1, panel (b): pyspi turns one series into K pairwise matrices.

Drawn as a receding deck rather than a grid of three, because the object being
introduced is a STACK of depth K -- the reader has to leave this panel believing
there are hundreds of these, not three. Layer colours are fixed in style.LAYERS
and identify the same three statistics again in panel (c).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures import style
from figures.fig1_data import series, spi_matrices

style.use()
x = series()
S = spi_matrices(x)
keys = ["cov", "lag1", "coh"]
titles = [r"$|\rho_{ij}|$", r"$|\rho_{ij}(\tau{=}1)|$", r"$\mathrm{coh}_{ij}$"]
M = x.shape[0]

fig, ax = plt.subplots(figsize=(3.1, 1.95))
style.bare(ax)
SKEW, DX, DY = -24, 1.55, 1.15
ALPHA = [1.0, 0.86, 0.72]              # depth cue, kept mild so hue survives

for d, (k, t) in enumerate(zip(keys, titles)):
    z = d                                             # 0 = front
    tr = Affine2D().skew_deg(SKEW, 0).translate(z * DX, z * DY) + ax.transData
    im = ax.imshow(S[k], cmap=style.SEQUENTIAL, extent=(0, M, 0, M),
                   interpolation="nearest", vmin=0, vmax=1,
                   zorder=10 - z, alpha=ALPHA[d])
    im.set_transform(tr)
    # Hairline frame: keeps pale rear planes legible and carries the layer colour.
    ax.add_patch(plt.Rectangle((0, 0), M, M, fill=False, lw=0.8,
                               edgecolor=style.LAYERS[d], zorder=10 - z,
                               transform=tr))
    # Legend lives OFF the stack: the skew displaces anything anchored to a
    # plane's own corner straight onto the plane in front of it.
    ly = 5.4 - 0.95 * d
    ax.add_patch(plt.Rectangle((-7.4, ly - 0.17), 0.42, 0.34,
                               facecolor=style.LAYERS[d], edgecolor="none",
                               zorder=20))
    ax.text(-6.75, ly, t, fontsize=7, color=style.INK,
            va="center", ha="left", zorder=20)

ax.text(8.6, 8.0, "...", fontsize=13, color=style.GREY,
        ha="center", va="center", rotation=36)
ax.text(9.6, 8.6, r"$K \approx 300$", fontsize=7.5, color=style.GREY,
        ha="left", va="center")
ax.set_xlim(-7.8, 14.0)
ax.set_ylim(-0.7, 9.6)
ax.set_title("pairwise statistics", pad=2, fontsize=8, loc="left", x=0.02)
fig.savefig(Path(__file__).parent / "p1b.png")
print("wrote p1b.png")
