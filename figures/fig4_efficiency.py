"""Fig 4 -- does it work, and what do the controls say.

Three things must be legible, and all three are measured:

  1. `spi-mpnn` is sample-efficient: 0.67 at n=20, 0.99 by n=500.
  2. every alternative source of a graph is far behind -- `correlation` plateaus
     at ~0.59, which is not underfitting but its information CEILING (Fig 3:
     a contemporaneous symmetric statistic separates fork, and cannot separate
     chain from collider).
  3. `shuffled` is drawn honestly. It sits at chance while n is small and rises
     to 0.82 by n=1000, because permuting values across pairs preserves each
     SPI's multiset. That BOUNDS the topology claim to small n rather than
     voiding it, and omitting it would be the one thing a careful reader could
     catch us on.

Direct labelling at the curve ends, no legend box.

    python figures/fig4_efficiency.py   -> figures/out/fig4.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures import style

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "sample_efficiency_30seeds_main_results.json"
ORDER = ["spi-mpnn", "fixed-spi", "correlation", "latent", "shuffled", "node-only"]
LABEL = {"spi-mpnn": "spi-mpnn (ours)", "fixed-spi": "fixed-spi",
         "correlation": "correlation", "latent": "latent",
         "shuffled": "shuffled", "node-only": "node-only"}


def build():
    style.use()
    r = json.load(open(SRC))
    ns = sorted(r["results"], key=int)
    x = np.array([int(n) for n in ns])
    chance = 1 / len(r["classes"])

    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    ax.axhline(chance, color=style.FAINT, lw=0.9, zorder=0)
    # Below the line: above it collides with node-only and latent, which sit ON
    # chance for the whole sweep.
    ax.text(x[0], chance - 0.022, "chance", fontsize=6.5, color=style.GREY, va="top")

    ends = {}
    for m in ORDER:
        y = np.array([r["results"][n]["models"][m]["f1_mean"] for n in ns])
        e = np.array([r["results"][n]["models"][m]["f1_std"] for n in ns])
        c = style.MODELS[m]
        lw = 1.9 if m == "spi-mpnn" else 1.1
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.13, lw=0)
        ax.plot(x, y, color=c, lw=lw, marker="o", ms=2.6,
                zorder=5 if m == "spi-mpnn" else 3)
        ends[m] = y[-1]

    # Direct labels need de-collision: spi-mpnn and fixed-spi both finish at
    # 0.997, and node-only and latent both finish on chance.
    GAP = 0.042
    placed = {}
    for m in sorted(ends, key=lambda k: ends[k]):
        yv = ends[m]
        for other in placed.values():
            if abs(yv - other) < GAP:
                yv = other + GAP
        placed[m] = yv
        ax.plot([x[-1] * 1.02, x[-1] * 1.05], [ends[m], yv], color=style.MODELS[m],
                lw=0.6, clip_on=False, zorder=2)
        ax.text(x[-1] * 1.07, yv, LABEL[m], color=style.MODELS[m], fontsize=6.8,
                va="center", ha="left",
                fontweight="bold" if m == "spi-mpnn" else "normal")

    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(x[0] * 0.85, x[-1] * 3.1)
    ax.set_ylim(0.20, 1.03)
    ax.set_xlabel("training instances per class", fontsize=8)
    ax.set_ylabel("macro F1", fontsize=8)
    ax.tick_params(labelsize=7)

    out = ROOT / "figures" / "out"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig4.png"); fig.savefig(out / "fig4.pdf")
    print("wrote figures/out/fig4.{png,pdf}")


if __name__ == "__main__":
    build()
