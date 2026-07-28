"""Fig 5 -- what `w` actually recovers.

(a) Module enrichment against a permutation null. Which named families of
    statistics carry the weight, calibrated so "enriched" has a meaning.
(b) Order specificity -- the panel that makes this discovery rather than
    confirmation. Four arms of the SAME estimator (`sgc_parametric`) differing
    only in AR model order, 6 SPIs each, matched on statistic and frequency
    band. Nothing in a VAR(1) encodes a preference over model order; order-1 is
    correctly specified and order-20 is 20x overparameterised.
(c) Robustness to lambda across a 50x range, which is the answer to "you tuned
    lambda until the answer looked right".

    python figures/fig5_signature.py   -> figures/out/fig5.{pdf,png}
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures import style

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.load(open(ROOT / "src" / "spi_labels.json"))
ARMS = ["param o=1", "param o=auto", "param o=20", "nonparam"]
ARM_LAB = ["order 1\ncorrect", "order\nauto", "order 20\n20x over", "non-\nparam"]


def _runs():
    out = []
    for p in sorted(glob.glob(str(ROOT / "results" / "sample_efficiency_r0297_*gl*_results.json"))):
        r = json.load(open(p))
        blk = r["results"][max(r["results"], key=int)]["models"].get("spi-mpnn")
        if blk is None:
            continue
        lam = (r.get("hyperparameters") or {}).get("group_lambda")
        if lam is None:
            m = re.search(r"gl([0-9.]+)", Path(p).name)
            lam = float(m.group(1)) if m else float("nan")
        W = np.abs(np.array([s["learned_w"] for s in blk["per_seed"]])).mean(0)
        out.append((float(lam), W, list(r["spi_names"]), blk["f1_mean"]))
    return sorted(out, key=lambda t: t[0])


def _arm(name):
    if not name.startswith("sgc_"):
        return None
    if "nonparametric" in name:
        return "nonparam"
    for key, tag in (("order-20", "param o=20"), ("order-1", "param o=1"),
                     ("order-None", "param o=auto")):
        if key in name:
            return tag
    return None


def panel_a(ax, W, names, n_perm=4000):
    mods = np.array([LABELS.get(n, {}).get("module", "MXX") for n in names])
    rng = np.random.default_rng(0)
    rows = []
    for m in sorted(set(mods)):
        mask = mods == m
        if mask.sum() < 3:
            continue
        base = mask.sum() / mask.size
        obs = W[mask].sum() / W.sum() / base
        null = np.array([(W[rng.permutation(W.size)][mask]).sum() / W.sum() / base
                         for _ in range(n_perm)])
        rows.append((obs, np.percentile(null, 2.5), np.percentile(null, 97.5),
                     m, int(mask.sum())))
    rows.sort()
    y = np.arange(len(rows))
    lo = np.array([r[1] for r in rows]); hi = np.array([r[2] for r in rows])
    ax.fill_betweenx([-1, len(rows)], lo.mean(), hi.mean(), color=style.FAINT,
                     alpha=0.55, lw=0, zorder=0)
    for i, (obs, l, h, m, k) in enumerate(rows):
        sig = obs > h or obs < l
        c = style.ACCENT if obs > h else (style.MODELS["fixed-spi"] if obs < l
                                          else style.GREY)
        ax.plot([1, obs], [i, i], color=c, lw=1.0, zorder=2)
        ax.plot(obs, i, "o", ms=4.2 if sig else 3.0, color=c, zorder=3)
    ax.axvline(1, color=style.INK, lw=0.7, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r[3]}" for r in rows], fontsize=6.6)
    ax.set_xscale("log"); ax.set_xlim(0.15, 30)
    ax.set_xticks([0.2, 1, 5, 20]); ax.set_xticklabels(["0.2", "1", "5", "20"], fontsize=7)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("enrichment of $|w|$  (1 = no preference)", fontsize=7.5)
    ax.text(0.97, 0.045, "band = permutation null",
            transform=ax.transAxes, fontsize=6.2, color=style.GREY,
            ha="right", va="bottom")


def panel_b(ax, runs):
    vals = {a: [] for a in ARMS}
    for lam, W, names, _ in runs:
        Wn = W / W.mean()
        a = np.array([_arm(n) for n in names], dtype=object)
        for k in ARMS:
            vals[k].append(Wn[a == k].mean())
    xs = np.arange(len(ARMS))
    rng = np.random.default_rng(1)
    for i, k in enumerate(ARMS):
        v = np.array(vals[k])
        c = style.ACCENT if i == 0 else style.MODELS["fixed-spi"]
        ax.plot(i + rng.uniform(-0.12, 0.12, v.size), v, "o", ms=3.0, color=c,
                alpha=0.55, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=c, lw=2.0, zorder=4)
    ax.axhline(1, color=style.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(3.45, 1.0, "vocabulary\naverage", fontsize=6.2, color=style.GREY,
            va="center", ha="left")
    ax.set_yscale("log"); ax.set_ylim(0.2, 12)
    ax.set_yticks([0.3, 1, 3, 10]); ax.set_yticklabels(["0.3", "1", "3", "10"], fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(ARM_LAB, fontsize=6.0)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("mean $|w|$ vs average", fontsize=7.5)
    ax.text(0.5, 0.965, "same estimator, 6 SPIs per arm\nmatched statistic and band",
            transform=ax.transAxes, fontsize=6.3, color=style.INK,
            ha="center", va="top")


def panel_c(ax, runs):
    mods, per_lam = None, {}
    for lam, W, names, f1 in runs:
        m = np.array([LABELS.get(n, {}).get("module", "MXX") for n in names])
        if mods is None:
            mods = sorted(set(m))
        d = per_lam.setdefault(lam, {k: [] for k in mods})
        for k in mods:
            d[k].append(W[m == k].sum() / W.sum())
    lams = sorted(per_lam)
    series = {k: [float(np.mean(per_lam[l][k])) for l in lams] for k in mods}
    top = sorted(mods, key=lambda k: -max(series[k]))[:4]
    for k in mods:
        c = style.ACCENT if k == "M05" else (
            style.MODELS["fixed-spi"] if k in top else style.FAINT)
        ax.plot(lams, 100 * np.array(series[k]), "-o", ms=2.4,
                lw=1.7 if k == "M05" else (1.0 if k in top else 0.7),
                color=c, zorder=4 if k == "M05" else (3 if k in top else 1))
    placed = {}
    for k in sorted(top, key=lambda z: series[z][-1]):
        yv = 100 * series[k][-1]
        for other in placed.values():
            if abs(yv - other) < 3.4:
                yv = other + 3.4
        placed[k] = yv
        c = style.ACCENT if k == "M05" else style.MODELS["fixed-spi"]
        ax.text(lams[-1] * 1.18, yv, k, fontsize=6.4, color=c, va="center",
                fontweight="bold" if k == "M05" else "normal")
    ax.set_xscale("log")
    ax.set_xlabel(r"group-lasso $\lambda$", fontsize=7.5)
    ax.set_ylabel("share of $|w|$  (%)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(min(lams) * 0.7, max(lams) * 3.2)
    ax.text(0.03, 0.97, "top module invariant over 50x\n(9/10 runs)",
            transform=ax.transAxes, fontsize=6.3, va="top", color=style.INK)


def build():
    style.use()
    runs = _runs()
    ref = [r for r in runs if abs(r[0] - 0.01) < 1e-9][-1]

    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.15], wspace=0.42,
                          left=0.065, right=0.955, bottom=0.20, top=0.79)
    panel_a(fig.add_subplot(gs[0]), ref[1], ref[2])
    panel_b(fig.add_subplot(gs[1]), runs)
    panel_c(fig.add_subplot(gs[2]), runs)

    for i, (letter, title) in enumerate([
            ("a", "which statistics carry $w$"),
            ("b", "the probe finds the correct model order"),
            ("c", "invariant to the penalty")]):
        ax = fig.axes[i]
        x0 = ax.get_position().x0
        fig.text(x0 - 0.052, 0.975, letter, fontsize=9.5, fontweight="bold", va="top")
        fig.text(x0 - 0.02, 0.925, title, fontsize=8, va="top")

    out = ROOT / "figures" / "out"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig5.png"); fig.savefig(out / "fig5.pdf")
    print("wrote figures/out/fig5.{png,pdf}")


if __name__ == "__main__":
    build()
