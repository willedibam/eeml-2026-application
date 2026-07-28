#!/usr/bin/env python3
"""
Does the probe recover the correct AR model ORDER? A controlled within-estimator test.

The standing objection to every synthetic result here is tautology: R0 is a
VAR(1), so a Granger-family statistic must win, and "w recovers spectral GC" is
confirmation rather than discovery.

This test escapes that. pyspi's vocabulary contains `sgc_parametric` at several
model orders -- order-1, order-None (auto), order-20 -- with everything else
matched: same estimator, same statistic (mean/max), same frequency bands, 6 SPIs
per arm. The ONLY varying factor is the order of the autoregressive model fitted
by the estimator.

Nothing in the generator encodes a preference over model order. Order-1 is
correctly specified for a VAR(1); order-20 is 20x overparameterised and pays a
variance cost. If the probe prefers order-1, it has recovered a statistical
efficiency fact about the estimator, not the physics fact built into the data.

The falsifiable prediction that follows: on VAR(p) data with p > 1, the preferred
order should track p. A flat preference for order-1 regardless of the true order
would show the probe is picking up something other than specification quality.

    PYTHONPATH=. python docs/order_specificity.py results/*_results.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ARMS = ("param o=1", "param o=auto", "param o=20", "nonparam")


def _arm(name: str) -> str | None:
    if not name.startswith("sgc_"):
        return None
    if "nonparametric" in name:
        return "nonparam"
    for key, tag in (("order-20", "param o=20"), ("order-1", "param o=1"),
                     ("order-None", "param o=auto")):
        if key in name:
            return tag
    return None


def main(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        r = json.load(open(p))
        se = r.get("results")
        if not se:
            continue
        blk = se[max(se, key=int)]["models"].get("spi-mpnn")
        if blk is None:
            continue
        lam = (r.get("hyperparameters") or {}).get("group_lambda")
        if lam is None:
            m = re.search(r"gl([0-9.]+)", p.name)
            lam = float(m.group(1)) if m else float("nan")
        W = np.abs(np.array([s["learned_w"] for s in blk["per_seed"]])).mean(0)
        W = W / W.mean()                       # 1.0 = a typical SPI
        a = np.array([_arm(x) for x in r["spi_names"]], dtype=object)
        if not all((a == k).any() for k in ARMS):
            continue                            # vocabulary lacks the order arms
        rows.append((float(lam), [float(W[a == k].mean()) for k in ARMS],
                     [int((a == k).sum()) for k in ARMS]))
    if not rows:
        raise SystemExit("no runs whose vocabulary contains the sgc order arms")
    rows.sort(key=lambda t: t[0])

    print("Spectral Granger causality only -- same estimator, varying model order.")
    print("Values = mean |w| relative to the vocabulary average (1.0 = typical SPI).\n")
    print(f"{'lambda':>8}  " + "".join(f"{k:>14}" for k in ARMS))
    print(f"{'K':>8}  " + "".join(f"{c:>14}" for c in rows[0][2]))
    for lam, v, _ in rows:
        print(f"{lam:>8}  " + "".join(f"{x:>14.2f}" for x in v))
    A = np.array([v for _, v, _ in rows])
    print(f"\n{'median':>8}  " + "".join(f"{np.median(A[:, i]):>14.2f}"
                                         for i in range(len(ARMS))))
    o1, o20, npar = A[:, 0], A[:, 2], A[:, 3]
    print(f"\n  order-1 > order-20      {int((o1 > o20).sum())}/{len(o1)} runs,"
          f"  median ratio {np.median(o1 / o20):.2f}x")
    print(f"  order-1 > nonparametric {int((o1 > npar).sum())}/{len(o1)} runs,"
          f"  median ratio {np.median(o1 / npar):.2f}x")
    print("\n  Matched on estimator, statistic and frequency band, so this cannot"
          "\n  be a module-size, directedness or linearity artifact.")


if __name__ == "__main__":
    ps = [Path(a) for a in sys.argv[1:]]
    if not ps:
        print(__doc__)
        sys.exit(1)
    main(ps)
