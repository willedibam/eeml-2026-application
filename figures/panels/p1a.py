"""Fig 1, panel (a): the raw multivariate time series."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures import style
from figures.fig1_data import series

style.use()
x = series()
fig, ax = plt.subplots(figsize=(2.4, 1.35))
im = ax.imshow(x, aspect="auto", cmap=style.DIVERGING, vmin=-2.4, vmax=2.4,
               interpolation="nearest")
# Hairline gaps between channels: the rows are a SET of series, not an image.
for k in range(1, x.shape[0]):
    ax.axhline(k - 0.5, color=style.PAPER, lw=1.4)
ax.set_yticks(range(x.shape[0]))
ax.set_yticklabels([f"$c_{i+1}$" for i in range(x.shape[0])], fontsize=6.5)
ax.set_xticks([0, x.shape[1] - 1]); ax.set_xticklabels(["0", "T"], fontsize=6.5)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0, pad=1.5)
ax.set_title("multivariate time series", pad=4, fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-2.4, 0, 2.4])
cb.ax.set_yticklabels(["−2", "0", "+2"], fontsize=6.5)
cb.outline.set_visible(False); cb.ax.tick_params(length=0, pad=1.5)
fig.savefig(Path(__file__).parent / "p1a.png")
print("wrote p1a.png")
