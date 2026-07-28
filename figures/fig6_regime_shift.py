"""Fig 6 -- the signature moves when the generating mechanism moves.

The strongest answer to "your synthetic result is a tautology". R0 and R1b share
the task, the motifs, the models and the vocabulary; they differ only in that
R1b observes the same linear VAR through a squaring nonlinearity. The generator
docstring stated the prediction BEFORE the result was seen: for jointly Gaussian
latents corr(x_i^2, x_j^2) = 2 corr^2, so linear measures are attenuated rather
than blinded, and nonlinear / information-theoretic ones should gain RELATIVE
weight.

Drawn as a slopegraph with the predicted DIRECTION marked, so a reader can see
the claim was falsifiable: rising `directed & linear` would have refuted it.

Robustness: the shift is measured against three different R0 lambdas, because
R0 and R1b sit at different lambdas (0.001-0.005 vs 0.0002) and a shift visible
against only one reference would be a lambda artifact.

    python figures/fig6_regime_shift.py   -> figures/out/fig6.{pdf,png}
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
LABELS = json.load(open(ROOT / "src" / "spi_labels.json"))
REFS = ["sample_efficiency_r0297_path_gl0.001_results.json",
        "sample_efficiency_r0297_path_gl0.002_results.json",
        "sample_efficiency_r0297_gl0.005_results.json"]
TEST = "sample_efficiency_r1b_lam0.0002_results.json"
AXES = [("directed & linear", lambda d, nl: d & ~nl, "down"),
        ("directed & nonlinear", lambda d, nl: d & nl, "up"),
        ("nonlinear", lambda d, nl: nl, "up")]


def load(fn):
    r = json.load(open(ROOT / "results" / fn))
    blk = r["results"][max(r["results"], key=int)]["models"]["spi-mpnn"]
    W = np.abs(np.array([s["learned_w"] for s in blk["per_seed"]]))
    return list(r["spi_names"]), W


def shares(names, W, common):
    idx = [names.index(n) for n in common]
    Wc = W[:, idx]
    d = np.array([LABELS.get(n, {}).get("directed", False) for n in common])
    nl = np.array([LABELS.get(n, {}).get("nonlinear", False) for n in common])
    tot = np.clip(Wc.sum(1), 1e-12, None)
    return {lab: Wc[:, f(d, nl)].sum(1) / tot for lab, f, _ in AXES}


def build():
    style.use()
    tn, tW = load(TEST)
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    rng = np.random.default_rng(0)

    xs = {lab: i for i, (lab, _, _) in enumerate(AXES)}
    mids = {lab: [] for lab, _, _ in AXES}
    for ri, ref in enumerate(REFS):
        rn, rW = load(ref)
        common = [n for n in rn if n in set(tn)]
        A, B = shares(rn, rW, common), shares(tn, tW, common)
        for lab, _, _ in AXES:
            x = xs[lab]
            a, b = 100 * A[lab].mean(), 100 * B[lab].mean()
            ax.plot([x - 0.22, x + 0.22], [a, b], "-o", ms=3.0, lw=1.1,
                    color=style.MODELS["fixed-spi"], alpha=0.85, zorder=3)
            mids[lab] += [a, b]
            if ri == 0:
                ax.text(x - 0.26, a, "R0", fontsize=6.4, ha="right", va="center",
                        color=style.GREY)
                ax.text(x + 0.26, b, "R1b", fontsize=6.4, ha="left", va="center",
                        color=style.GREY)

    # Anchor each prediction arrow to ITS OWN axis's data range. Floating them
    # all at one height reads as decoration rather than a per-axis claim.
    for lab, _, direction in AXES:
        x, c = xs[lab] + 0.50, np.mean(mids[lab])
        dy = 7 if direction == "up" else -7
        ax.annotate("", xy=(x, c + dy), xytext=(x, c - dy),
                    arrowprops=dict(arrowstyle="-|>", color=style.ACCENT, lw=1.5,
                                    mutation_scale=10))
    ax.text(0.5, 0.985, "orange = direction predicted in advance",
            transform=ax.transAxes, fontsize=6.5, color=style.ACCENT,
            ha="center", va="top")

    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["directed\n& linear", "directed\n& nonlinear", "nonlinear"],
                       fontsize=7)
    ax.set_xlim(-0.55, 2.78); ax.set_ylim(0, 70)
    ax.set_ylabel("share of $|w|$  (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.93, "3 reference $\\lambda$ per axis\ndirected & linear falls 3/3",
            transform=ax.transAxes, fontsize=6.4, color=style.INK, va="top")

    out = ROOT / "figures" / "out"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig6.png"); fig.savefig(out / "fig6.pdf")
    print("wrote figures/out/fig6.{png,pdf}")


if __name__ == "__main__":
    build()
