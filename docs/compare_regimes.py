#!/usr/bin/env python3
"""
Paired cross-regime comparison of the learned SPI signature.

Why this rather than another single-regime enrichment. An absolute statement
("module M05 carries 33% of |w| in regime X") is confounded by everything that
differs between runs: lambda, K, the group-lasso scale, how well the classifier
fit. A *difference* between two regimes that share the task, the motifs, the
models and the vocabulary cancels those nuisances to first order, and the
generators make a directional prediction that can come out wrong:

    R0  linear VAR                -> directed & LINEAR statistics carry w
    R1b linear VAR through x^2    -> nonlinear / information-theoretic
                                     statistics gain RELATIVE weight

That prediction is falsifiable. It is stated in the R1b generator docstring
before the result was seen, and its basis is that for jointly Gaussian latents
corr(x_i^2, x_j^2) = 2*corr^2 -- linear measures are attenuated but not blind,
so the claim is a shift in relative weight, not a collapse.

Uncertainty is a bootstrap over SEEDS, resampled independently within each
regime, because the two regimes are separate runs. A module is reported as
moved only when the 95% interval on the difference excludes zero.

    PYTHONPATH=. python docs/compare_regimes.py <ref.json> <test.json> [--chance 0.3333]

Reads only fields the result JSONs already carry (spi_names, per-seed
learned_w, f1_mean, hyperparameters), so it is reproducible from the JSONs
alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LABELS = json.load(open(Path(__file__).parent.parent / "src" / "spi_labels.json"))


def _load(path: Path, n_key: str | None = None):
    r = json.load(open(path))
    names = list(r["spi_names"])
    se = r["results"]
    n_key = n_key or max(se, key=int)
    blk = se[n_key]["models"].get("spi-mpnn")
    if blk is None:
        # Ablation-only runs (sgc_only, top3, edge_ablation) legitimately have no
        # spi-mpnn. Skip them instead of exiting, or one such file aborts a whole
        # batch audit.
        return None
    W = np.abs(np.array([p["learned_w"] for p in blk["per_seed"]], dtype=float))
    return {"name": path.name, "names": names, "W": W, "n": n_key,
            "f1": blk["f1_mean"], "f1_std": blk.get("f1_std", float("nan")),
            "lam": (r.get("hyperparameters") or {}).get("group_lambda")}


def _module_shares(names: list[str], W: np.ndarray, mods: np.ndarray) -> dict:
    """Per-seed share of total |w| falling on each module. W is (seeds, K)."""
    Wn = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    return {m: Wn[:, mods == m].sum(axis=1) for m in sorted(set(mods))}


def compare(ref: Path, test: Path, chance: float, n_boot: int = 4000) -> None:
    A, B = _load(ref), _load(test)
    if A is None or B is None:
        missing = ", ".join(p.name for p, d in ((ref, A), (test, B)) if d is None)
        print(f"=== skipped: no spi-mpnn block in {missing} ===")
        return

    # Restrict to the SPIs both runs retained; the robust-scaling filter can
    # drop different columns in different datasets, and comparing shares over
    # different denominators would manufacture differences.
    common = [n for n in A["names"] if n in set(B["names"])]
    if len(common) < 20:
        raise SystemExit(f"only {len(common)} shared SPIs -- refusing to compare")
    ia = [A["names"].index(n) for n in common]
    ib = [B["names"].index(n) for n in common]
    WA, WB = A["W"][:, ia], B["W"][:, ib]
    mods = np.array([LABELS.get(n, {}).get("module", "MXX") for n in common])
    nonlin = np.array([LABELS.get(n, {}).get("nonlinear", False) for n in common])
    direct = np.array([LABELS.get(n, {}).get("directed", False) for n in common])

    print(f"=== {A['name']}  ->  {B['name']} ===")
    print(f"  shared SPIs: {len(common)}  "
          f"(ref K={len(A['names'])}, test K={len(B['names'])})")
    for tag, D in (("ref ", A), ("test", B)):
        warn = "   <-- AT CHANCE, signature not interpretable" \
               if D["f1"] < chance + 0.05 else ""
        print(f"  {tag}: n_train={D['n']}  F1={D['f1']:.4f}+/-{D['f1_std']:.3f}  "
              f"lambda_g={D['lam']}  seeds={D['W'].shape[0]}{warn}")
    if min(A["f1"], B["f1"]) < chance + 0.05:
        print("\n  A signature from a near-chance classifier is group-lasso "
              "geometry, not\n  recovered mechanism. Fix the accuracy before "
              "reading anything below.\n")

    rng = np.random.default_rng(0)
    sa, sb = _module_shares(common, WA, mods), _module_shares(common, WB, mods)

    def boot(va: np.ndarray, vb: np.ndarray) -> tuple[float, float, float]:
        d = vb.mean() - va.mean()
        idx_a = rng.integers(0, va.size, (n_boot, va.size))
        idx_b = rng.integers(0, vb.size, (n_boot, vb.size))
        null = vb[idx_b].mean(1) - va[idx_a].mean(1)
        return d, float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))

    rows = []
    for m in sorted(set(mods)):
        if (mods == m).sum() < 3:
            continue
        d, lo, hi = boot(sa[m], sb[m])
        rows.append((d, lo, hi, m, int((mods == m).sum()),
                     sa[m].mean(), sb[m].mean()))
    rows.sort(reverse=True)

    print(f"  {'module':8} {'K':>4}  {'ref%':>7} {'test%':>7} {'delta pp':>9}"
          f"  {'95% CI':>18}")
    for d, lo, hi, m, k, a_, b_ in rows:
        moved = "  MOVED" if (lo > 0) or (hi < 0) else ""
        print(f"  {m:8} {k:4d}  {100*a_:7.2f} {100*b_:7.2f} {100*d:+9.2f}"
              f"  [{100*lo:+7.2f},{100*hi:+7.2f}]{moved}")

    print("\n  -- pre-registered axes --")
    for lab, mask in (("nonlinear", nonlin), ("directed", direct),
                      ("directed & linear", direct & ~nonlin),
                      ("directed & nonlinear", direct & nonlin)):
        if mask.sum() == 0:
            continue
        va = (WA[:, mask].sum(1) / np.clip(WA.sum(1), 1e-12, None))
        vb = (WB[:, mask].sum(1) / np.clip(WB.sum(1), 1e-12, None))
        d, lo, hi = boot(va, vb)
        moved = "  MOVED" if (lo > 0) or (hi < 0) else ""
        print(f"  {lab:22} {mask.sum():3d} SPIs  {100*va.mean():6.2f}% -> "
              f"{100*vb.mean():6.2f}%  {100*d:+7.2f}pp  "
              f"[{100*lo:+6.2f},{100*hi:+6.2f}]{moved}")

    print("\n  PREDICTION: 'nonlinear' and 'directed & nonlinear' rise; "
          "'directed & linear' falls.\n  Anything else -- including no "
          "movement at all -- falsifies it.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    chance = 1 / 3
    if "--chance" in sys.argv:
        chance = float(sys.argv[sys.argv.index("--chance") + 1])
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    compare(Path(args[0]), Path(args[1]), chance)
