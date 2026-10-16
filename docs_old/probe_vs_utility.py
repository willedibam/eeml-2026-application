#!/usr/bin/env python3
"""
Does the learned probe rank statistics by their actual task-relevance?

This is the claim the abstract makes -- "a learned weight vector identifies
which statistics are task-relevant" -- tested WITHOUT any taxonomy, and without
appealing to a theorem that dictates the answer in advance.

Method. For every SPI independently, measure its standalone discriminative
power: summarise its (M,M) matrix per instance with permutation-invariant
statistics (node order is randomised by the generator, so the raw matrix is not
comparable across instances), then cross-validate a logistic regression on
those summaries alone. That yields a task-relevance score per SPI that owes
nothing to the GNN. Then correlate it with |w| (Spearman).

Why this escapes the tautology. "Directed statistics win on a directed problem"
restates the Markov-equivalence theorem. "The probe's ranking of 284 statistics
agrees with their independently measured usefulness at rho=X" is an empirical
claim about the estimator that could come out at zero. It is also exactly what
a reviewer means when they ask whether w means anything.

Falsifier: rho ~ 0 would mean w is not tracking task-relevance at all, and the
interpretability claim would rest solely on the taxonomy-level enrichment.

Usage:
    PYTHONPATH=. python docs/probe_vs_utility.py <data-dir> <c1,c2,c3> <results.json> [n_per_class]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import skew, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.graph_build import filter_spi_dimensions, load_spi_names, load_spi_tensor


def _invariant_summary(A: np.ndarray) -> np.ndarray:
    """Permutation-invariant summary of one (M, M) SPI matrix."""
    M = A.shape[0]
    off = A[~np.eye(M, dtype=bool)]
    asym = np.abs(A - A.T)[~np.eye(M, dtype=bool)]
    return np.array([off.mean(), off.std(), skew(off) if off.std() > 0 else 0.0,
                     off.max(), off.min(), asym.mean(), np.sort(off)[-3:].mean()])


def main() -> None:
    data, classes, res_path = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
    n_per = int(sys.argv[4]) if len(sys.argv) > 4 else 150

    def ok(d): return all((d / f).exists() for f in
                          ["spi_mpis.npz", "timeseries.npy", "meta.json"])
    dirs = {c: sorted(d for d in (data / c).iterdir() if ok(d))[:n_per] for c in classes}
    names0 = load_spi_names(dirs[classes[0]][0])
    tens, y = [], []
    for i, c in enumerate(classes):
        for d in dirs[c]:
            tens.append(load_spi_tensor(d, names0)); y.append(i)
    y = np.array(y)
    names, idx = filter_spi_dimensions(names0, tens)
    T = np.stack([t[:, :, idx] for t in tens])          # (N, M, M, K)
    print(f"  {T.shape[0]} instances, K={len(names)}")

    # standalone utility per SPI
    util = np.zeros(len(names))
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    for k in range(len(names)):
        X = np.stack([_invariant_summary(T[i, :, :, k]) for i in range(T.shape[0])])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = StandardScaler().fit_transform(X)
        util[k] = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()

    r = json.load(open(res_path))
    n_key = max(r["results"], key=int)
    W = np.abs(np.array([p["learned_w"] for p in
                         r["results"][n_key]["models"]["spi-mpnn"]["per_seed"]])).mean(0)
    # Align on NAMES, never on position: filter_spi_dimensions can retain a
    # different subset here (fewer instances -> different variance/missingness),
    # so index i in `util` need not be index i in the trained probe. Truncating
    # both to a common length would silently compare different statistics.
    probe_names = r["spi_names"]
    common = [n for n in names if n in set(probe_names)]
    pi = {n: i for i, n in enumerate(probe_names)}
    ui = {n: i for i, n in enumerate(names)}
    if len(common) < len(names) or len(common) < len(probe_names):
        print(f"  [align] probe K={len(probe_names)}, utility K={len(names)}, "
              f"common={len(common)}")
    W = np.array([W[pi[n]] for n in common])
    util = np.array([util[ui[n]] for n in common])
    names = common

    rho, p = spearmanr(W, util)
    chance = 1.0 / len(classes)
    print(f"\n  standalone utility: chance={chance:.3f}  "
          f"best={util.max():.3f} ({names[int(util.argmax())][:40]})  "
          f"median={np.median(util):.3f}")
    print(f"  SPEARMAN(|w|, standalone utility) = {rho:+.3f}  (p={p:.2e})")
    o = np.argsort(W)[::-1][:10]
    print(f"\n  {'|w| rank':>8} {'utility':>8} {'util rank':>10}  SPI")
    ur = (-util).argsort().argsort()
    for i, k in enumerate(o):
        print(f"  {i+1:8d} {util[k]:8.3f} {ur[k]+1:10d}  {names[k][:44]}")
    print(f"\n  top-10 by |w| have median utility rank {np.median(ur[o])+1:.0f} of {len(names)}")


if __name__ == "__main__":
    main()
