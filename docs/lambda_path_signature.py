#!/usr/bin/env python3
"""
Is the recovered signature a property of the data, or of the regularisation?

This is the sharpest methodological attack on a learned-weight interpretability
claim: "you turned lambda until the answer looked right." It has to be answered
with evidence, not assertion, because lambda genuinely does need re-tuning per
dataset -- the sqrt(|g|) group penalty scales with the number of groups, so K
changing forces a re-tune, and effect size changes the useful range.

The defence is a separation between two roles lambda plays:

  lambda as an SNR/capacity knob   legitimate, and selected on validation
                                   accuracy like any regularised estimator
  lambda as a signature knob       fatal. If WHICH module wins depends on
                                   lambda, the scientific output is an artifact

So the test is: across the lambda path, does the module RANKING move? Accuracy
may vary; the ranking must not, at least across the range where accuracy is near
its plateau.

Uncertainty is a bootstrap over seeds, so "M05 is top" is reported as a
probability rather than a point pick -- the same reason report_family_uncertainty
exists.

    PYTHONPATH=. python docs/lambda_path_signature.py results/<tag>_gl*.json
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

LABELS = json.load(open(Path(__file__).parent.parent / "src" / "spi_labels.json"))


def _lam(path: Path, r: dict) -> float:
    lam = (r.get("hyperparameters") or {}).get("group_lambda")
    if lam is not None:
        return float(lam)
    m = re.search(r"gl([0-9.]+)", path.name)      # legacy runs pre-hyperparameters
    return float(m.group(1)) if m else float("nan")


def _shares(r: dict, boot: int, rng) -> tuple[dict, dict, float]:
    """Per-module share of |w|, plus P(module is top) bootstrapped over seeds."""
    n = max(r["results"], key=int)
    b = r["results"][n]["models"]["spi-mpnn"]
    W = np.abs(np.array([s["learned_w"] for s in b["per_seed"]], dtype=float))
    mods = np.array([LABELS.get(x, {}).get("module", "MXX") for x in r["spi_names"]])
    uniq = sorted(set(mods))
    idx = rng.integers(0, W.shape[0], (boot, W.shape[0]))
    Wb = W[idx].mean(1)                                   # (boot, K)
    Sb = np.array([Wb[:, mods == m].sum(1) for m in uniq]).T
    Sb /= np.clip(Sb.sum(1, keepdims=True), 1e-12, None)  # (boot, n_mod)
    top = np.array(uniq)[Sb.argmax(1)]
    mean = W.mean(0)
    share = {m: float(mean[mods == m].sum() / mean.sum()) for m in uniq}
    ptop = {m: float((top == m).mean()) for m in uniq}
    return share, ptop, b["f1_mean"]


def main(paths: list[Path], boot: int = 4000) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for p in paths:
        r = json.load(open(p))
        if "spi-mpnn" not in r["results"][max(r["results"], key=int)]["models"]:
            continue
        share, ptop, f1 = _shares(r, boot, rng)
        rows.append((_lam(p, r), f1, share, ptop))
    rows.sort()
    if len(rows) < 2:
        raise SystemExit("need >=2 lambda runs")

    best = max(f1 for _, f1, _, _ in rows)
    print(f"{'lambda':>8} {'F1':>8}  {'plateau':>7}  top module (P(top), share)   "
          f"runner-up")
    for lam, f1, sh, pt in rows:
        order = sorted(sh, key=sh.get, reverse=True)
        a, b_ = order[0], order[1]
        print(f"{lam:>8} {f1:>8.4f}  {'yes' if f1 > best - 0.02 else 'NO':>7}  "
              f"{a} (P={pt[a]:.2f}, {100*sh[a]:.1f}%)".ljust(46)
              + f"{b_} ({100*sh[b_]:.1f}%)")

    mods = sorted(rows[0][2])
    plateau = [r for r in rows if r[1] > best - 0.02]
    for label, sub in (("full path", rows), ("accuracy plateau only", plateau)):
        if len(sub) < 2:
            continue
        M = np.array([[r[2][m] for m in mods] for r in sub])
        rs = [_spearman(M[i], M[j]) for i, j in combinations(range(len(sub)), 2)]
        tops = [max(r[2], key=r[2].get) for r in sub]
        lams = [r[0] for r in sub]
        print(f"\n  {label}: lambda {min(lams)}-{max(lams)} "
              f"({max(lams)/min(lams):.0f}x, {len(sub)} runs)")
        print(f"    module-share rank corr: mean rho={np.mean(rs):+.3f}  "
              f"min rho={np.min(rs):+.3f}")
        print(f"    top module: {'INVARIANT (' + tops[0] + ')' if len(set(tops)) == 1 else 'MOVES ' + str(tops)}")

    print("\n  Reading: an invariant top module across the plateau means the "
          "signature is a\n  property of the data, not of the penalty. A moving "
          "one means it is a tuning\n  artifact and cannot be reported as a "
          "scientific result.")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


if __name__ == "__main__":
    ps = [Path(a) for a in sys.argv[1:]]
    if not ps:
        print(__doc__)
        sys.exit(1)
    main(ps)
