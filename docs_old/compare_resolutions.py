#!/usr/bin/env python3
"""
Which grouping resolution actually carries information about the learned probe?

Candidate schemes for describing what w selected:
  families   6 hand-assigned categories (graph_build._FAMILY_RULES)
  modules   14 published Cliff et al. (2023) modules, carried by pyspi
  axes       4 cells of directed x nonlinear (directedness MEASURED from data)

These are not obviously comparable: a scheme with more groups explains more
variance for free. So each is scored by eta^2 (fraction of variance in log|w|
explained by group membership) calibrated against PERMUTED group labels, which
holds the group-count and group-size profile fixed. The z-score is then
comparable across schemes.

Also reports adjusted mutual information BETWEEN schemes, to say whether they
are redundant re-descriptions of one partition or genuinely complementary.

No scheme is assumed correct. If all three score near zero, the honest
conclusion is that w is not describable at any of these resolutions.

Usage:
    PYTHONPATH=. python docs/compare_resolutions.py <results.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from src.graph_build import assign_spi_families

LABELS = json.load(open(Path(__file__).parent.parent / "src" / "spi_labels.json"))


def eta2(values: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of variance in `values` explained by group membership."""
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_between = sum(
        (groups == g).sum() * (values[groups == g].mean() - grand) ** 2
        for g in np.unique(groups)
    )
    return float(ss_between / ss_tot)


def scored(values: np.ndarray, groups: np.ndarray, n_perm: int, rng) -> dict:
    obs = eta2(values, groups)
    null = np.array([eta2(values, rng.permutation(groups)) for _ in range(n_perm)])
    return {"eta2": obs, "null": float(null.mean()),
            "z": float((obs - null.mean()) / (null.std() + 1e-12)),
            "p": float((null >= obs).mean()), "k": int(len(np.unique(groups)))}


def main() -> None:
    r = json.load(open(sys.argv[1]))
    names = r["spi_names"]
    n_key = max(r["results"], key=int)
    W = np.abs(np.array([p["learned_w"] for p in
                         r["results"][n_key]["models"]["spi-mpnn"]["per_seed"]])).mean(0)
    v = np.log10(W + 1e-12)                       # weights span orders of magnitude

    fam_names, _ = assign_spi_families(names)
    mods = np.array([LABELS.get(n, {}).get("module", "MXX") for n in names])
    asym = np.asarray(r.get("spi_asymmetry", np.zeros(len(names))))
    directed = asym > 0.01
    nonlin = np.array([LABELS.get(n, {}).get("nonlinear", False) for n in names])
    axes = np.array([f"{'D' if d else 'U'}{'N' if nl else 'L'}"
                     for d, nl in zip(directed, nonlin)])

    schemes = {"families": np.array(fam_names), "modules": mods, "axes": axes}
    rng = np.random.default_rng(0)

    print(f"\n=== {Path(sys.argv[1]).name}  (n_train={n_key}, K={len(names)}) ===")
    print("How much of log|w| does each scheme explain, vs permuted labels?\n")
    print(f"  {'scheme':10} {'k':>3} {'eta^2':>7} {'null':>7} {'z':>6} {'p':>8}")
    for nm, g in schemes.items():
        s = scored(v, g, 2000, rng)
        print(f"  {nm:10} {s['k']:3d} {s['eta2']:7.3f} {s['null']:7.3f} "
              f"{s['z']:6.1f} {s['p']:8.4f}")

    print("\nAre the schemes redundant? (adjusted mutual information, 0=indep, 1=same)")
    ks = list(schemes)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            ami = adjusted_mutual_info_score(schemes[ks[i]], schemes[ks[j]])
            print(f"  {ks[i]:10} vs {ks[j]:10} AMI={ami:.3f}")

    print("\nDoes a finer scheme add anything ON TOP of a coarser one?")
    # residualise log|w| on `axes`, then ask what modules still explain
    for coarse, fine in [("axes", "modules"), ("axes", "families"),
                         ("families", "modules")]:
        res = v.copy()
        for g in np.unique(schemes[coarse]):
            m = schemes[coarse] == g
            res[m] -= res[m].mean()
        s = scored(res, schemes[fine], 2000, rng)
        print(f"  {fine:10} after removing {coarse:10} eta^2={s['eta2']:.3f} "
              f"z={s['z']:5.1f} p={s['p']:.4f}")


if __name__ == "__main__":
    main()
