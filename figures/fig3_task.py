"""Fig 3 -- the task, and why a symmetric statistic is not enough.

(a) the three motifs, directed.
(b) what a CONTEMPORANEOUS symmetric statistic actually sees.

The second panel is measured, not asserted. Motif nodes are randomly permuted
into the M channels, so a symmetric statistic sees only the label-invariant
multiset of pairwise values -- here the sorted |rho| triple. Chain and collider
are near-identical under it; fork is not, because its two children share a
parent and are the only pair with instantaneous coupling. (The VAR has no
self-persistence, so a directly-coupled pair has ~zero contemporaneous
correlation.)

That gives a hard ceiling for symmetric measures, and it is where the
`correlation` baseline actually sits: 0.582 measured here, 0.59 in the pipeline.

    python figures/fig3_task.py    -> figures/out/fig3.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mts-spi-study-cluster" / "src"))
from figures import style
from generators.chat import generate_var_chat_a as gen

NAMES = ["chain", "fork", "collider"]
EDGES = {"chain": [(0, 1), (1, 2)], "fork": [(0, 1), (0, 2)],
         "collider": [(1, 0), (2, 0)]}
N_INST = 300


def measure():
    """Sorted |rho| triple per instance -- what a symmetric statistic can use."""
    out = {}
    for c, name in enumerate(NAMES):
        rows = []
        for i in range(N_INST):
            x = gen(M=3, T=500, motif_class=c, alpha_lo=0.2, alpha_hi=0.8,
                    noise_std=0.1, rng=np.random.default_rng(1000 * c + i))
            R = np.abs(np.corrcoef(x.T))
            rows.append(np.sort(R[np.triu_indices(3, 1)]))
        out[name] = np.array(rows)
    return out


def draw_motif(ax, name):
    P = np.array([[0.5, 0.92], [0.08, 0.12], [0.92, 0.12]])
    for a, b in EDGES[name]:
        ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>",
                                     mutation_scale=9, lw=1.5,
                                     color=style.MOTIF[name],
                                     shrinkA=9, shrinkB=9))
    for i, (px, py) in enumerate(P):
        ax.add_patch(plt.Circle((px, py), 0.115, facecolor=style.PAPER,
                                edgecolor=style.MOTIF[name], lw=1.3, zorder=5))
        ax.text(px, py, str(i), ha="center", va="center", fontsize=7,
                zorder=6, color=style.MOTIF[name])
    ax.set_xlim(-0.10, 1.10); ax.set_ylim(-0.06, 1.12)
    ax.set_aspect("equal"); style.bare(ax)
    ax.set_title(name, fontsize=8, pad=1, color=style.MOTIF[name])


def build():
    style.use()
    S = measure()
    fig = plt.figure(figsize=(7.2, 2.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.5], wspace=0.30,
                          left=0.02, right=0.98, bottom=0.17, top=0.74)
    for k, name in enumerate(NAMES):
        draw_motif(fig.add_subplot(gs[k]), name)

    ax = fig.add_subplot(gs[3])
    xs = np.arange(3)
    w = 0.26
    for k, name in enumerate(NAMES):
        m = S[name].mean(0)
        se = S[name].std(0) / np.sqrt(N_INST)
        ax.bar(xs + (k - 1) * w, m, w * 0.92, yerr=2 * se, color=style.MOTIF[name],
               ecolor=style.GREY, error_kw={"lw": 0.7}, label=name)
    ax.set_xticks(xs); ax.set_xticklabels(["smallest", "middle", "largest"], fontsize=7)
    ax.set_ylabel(r"$|\rho_{ij}|$, sorted", fontsize=7.5)
    ax.tick_params(labelsize=7)
    # No in-bar labels: panel (a) already colour-keys the three motifs, and
    # rotated text collided with the fork bar it was labelling.
    ax.set_ylim(0, 0.34)
    ax.text(0.03, 0.98,
            "separable from this triple alone:\n"
            "  chain vs collider   0.53  (chance 0.50)\n"
            "  3-way               0.58  (chance 0.33)",
            transform=ax.transAxes, fontsize=6.4, va="top", ha="left",
            color=style.INK, family="monospace", linespacing=1.5)

    fig.text(0.02, 0.965, "a", fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.02, 0.90, "three directed motifs, randomly embedded in $M$ channels",
             fontsize=8, va="top")
    x0 = ax.get_position().x0
    fig.text(x0 - 0.035, 0.965, "b", fontsize=9.5, fontweight="bold", va="top")
    fig.text(x0, 0.90, "what a symmetric statistic sees", fontsize=8, va="top")

    out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig3.png"); fig.savefig(out / "fig3.pdf")
    print("wrote figures/out/fig3.{png,pdf}")


if __name__ == "__main__":
    build()
