#!/usr/bin/env python3
"""
Two-axis enrichment of the learned SPI weights: directed x nonlinear.

For each axis, report the share of total |w| that lands on the labelled subset
and calibrate it against permuting w across SPIs (GSEA-style, Subramanian et
al. 2005). Labels come from pyspi itself (src/spi_labels.json), so nothing here
depends on the hand-written family taxonomy.

The point. R0 (linear VAR) and R1 (quadratic coupling) share the task, the
models, the vocabulary and the motif topology -- only the coupling physics
differs. Both are directed, so directed-enrichment should hold in both; but the
KIND of directed statistic should move:

    R0: directed AND linear    (Granger / spectral-GC / phase-slope)
    R1: directed AND nonlinear (transfer entropy / directed information)

That shift is a prediction that can come out wrong, which is what separates it
from "we recovered GC on a VAR". Linear GC is provably blind to R1's coupling
(measured on the generator: |pearson| 0.020 vs MI 0.296 nats).

Usage:
    PYTHONPATH=. python docs/enrichment_2x2.py results/<a>.json [results/<b>.json ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LABELS = json.load(open(Path(__file__).parent.parent / "src" / "spi_labels.json"))


def _enrich(w: np.ndarray, mask: np.ndarray, n_perm: int, rng) -> dict:
    if mask.sum() == 0 or mask.all():
        return {"frac": float("nan"), "enrich": float("nan"), "z": float("nan")}
    base = mask.sum() / mask.size
    obs = w[mask].sum() / w.sum()
    null = np.array([(w[rng.permutation(w.size)][mask]).sum() / w.sum()
                     for _ in range(n_perm)])
    return {"frac": float(obs), "base": float(base),
            "enrich": float(obs / base),
            "z": float((obs - null.mean()) / (null.std() + 1e-12)),
            "p": float((null >= obs).mean())}


def analyse(path: Path, n_perm: int = 3000, n_key: str | None = None) -> None:
    r = json.load(open(path))
    names = r["spi_names"]
    se = r.get("results", {})
    n_key = n_key or max(se, key=int)
    blk = se[n_key]["models"].get("spi-mpnn")
    if blk is None:
        print(f"{path.name}: no spi-mpnn"); return

    W = np.abs(np.array([p["learned_w"] for p in blk["per_seed"]])).mean(0)

    # measured directedness (from data) takes precedence; fall back to labels
    asym = r.get("spi_asymmetry")
    directed = (np.asarray(asym) > 0.01 if asym is not None
                else np.array([LABELS.get(n, {}).get("directed", False) for n in names]))
    nonlinear = np.array([LABELS.get(n, {}).get("nonlinear", False) for n in names])

    rng = np.random.default_rng(0)
    print(f"\n=== {path.name}  (n_train={n_key}, K={len(names)}, "
          f"F1={blk['f1_mean']:.4f}, lambda_g={r['hyperparameters']['group_lambda']}) ===")
    for lab, mask in [("directed", directed),
                      ("nonlinear", nonlinear),
                      ("directed & linear", directed & ~nonlinear),
                      ("directed & nonlinear", directed & nonlinear)]:
        e = _enrich(W, mask, n_perm, rng)
        if np.isnan(e["enrich"]):
            print(f"  {lab:22} n/a"); continue
        flag = "  <<<" if e["enrich"] > 1.5 and e["z"] > 3 else ""
        print(f"  {lab:22} {mask.sum():3d}/{mask.size} SPIs  "
              f"{100*e['frac']:5.1f}% of |w|  {e['enrich']:5.2f}x  z={e['z']:5.1f}{flag}")

    top = np.argsort(W)[::-1][:6]
    print("  top-6: " + ", ".join(
        f"{names[i][:30]}{'[D]' if directed[i] else ''}{'[NL]' if nonlinear[i] else ''}"
        for i in top))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        analyse(Path(p))
