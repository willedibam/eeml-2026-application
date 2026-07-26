#!/usr/bin/env python3
"""
Are the human SPI families a real grouping, or a naming artifact?

Tests whether the six hand-assigned families (assign_spi_families) match the
EMPIRICAL structure of the SPIs — i.e. how their outputs actually co-vary on
this data. This is the on-data version of the module-level characterisation in
Cliff et al. (2023), whose modules are derived from literal output similarity,
not intuitive category names.

If human families ~ empirical clusters (high ARI/NMI, within-family |corr| >>
between-family), the semantic labels are earned and the group lasso is well
specified. If not, the grouping injects noise: the group penalty is grouping
statistics that don't co-vary, and "family X carries the weight" is a statement
about a label, not a coherent mode of dependence — in which case switch the
lasso groups to data-driven clusters and report the discrepancy.

Offline; runs on the local VAR data. Usage:
    PYTHONPATH=. python docs/spi_grouping_analysis.py [n_per_class]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.graph_build import (
    assign_spi_families, filter_spi_dimensions, load_spi_names, load_spi_tensor,
)

ROOT = Path("data/260327_eeml")
CLASSES = ["var-chain", "var-fork", "var-collider"]


def _ok(d: Path) -> bool:
    return ((d / "spi_mpis.npz").exists() and (d / "timeseries.npy").exists()
            and (d / "meta.json").exists())


def main(n_per_class: int = 150) -> None:
    dirs = {c: sorted(d for d in (ROOT / c).iterdir() if _ok(d))[:n_per_class]
            for c in CLASSES}
    spi_names = load_spi_names(dirs[CLASSES[0]][0])

    tensors = []
    for c in CLASSES:
        for d in dirs[c]:
            tensors.append(load_spi_tensor(d, spi_names))
    names, idx = filter_spi_dimensions(spi_names, tensors)
    K = len(names)

    # (n_instances * n_offdiag_pairs, K) matrix of SPI outputs
    rows = []
    for t in tensors:
        M = t.shape[0]
        off = ~np.eye(M, dtype=bool)
        rows.append(t[:, :, idx][off])            # (M*(M-1), K)
    X = np.concatenate(rows, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # |Pearson| similarity between SPIs; distance = 1 - |corr|
    C = np.abs(np.corrcoef(X.T))
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0

    fam_names, fam_idx = assign_spi_families(names)
    fam_of = np.array(fam_names)
    uniq = sorted(set(fam_names))
    k_clusters = len(uniq)

    Z = linkage(squareform(D, checks=False), method="average")
    clusters = fcluster(Z, t=k_clusters, criterion="maxclust")

    ari = adjusted_rand_score(fam_of, clusters)
    nmi = normalized_mutual_info_score(fam_of, clusters)

    # within- vs between-family similarity
    within, between = [], []
    for i in range(K):
        for j in range(i + 1, K):
            (within if fam_of[i] == fam_of[j] else between).append(C[i, j])
    within, between = np.array(within), np.array(between)

    print(f"K={K} retained SPIs, {len(uniq)} human families, "
          f"X={X.shape[0]} pair-rows")
    print(f"\nHuman-family vs empirical-cluster agreement (higher = labels earned):")
    print(f"  Adjusted Rand Index : {ari:+.3f}   (0 = chance, 1 = identical)")
    print(f"  Normalised MI       : {nmi:.3f}")
    print(f"\nWithin- vs between-family |Pearson| (coherent family => within >> between):")
    print(f"  within  mean={within.mean():.3f}  median={np.median(within):.3f}")
    print(f"  between mean={between.mean():.3f}  median={np.median(between):.3f}")
    print(f"  separation (within-between means) = {within.mean()-between.mean():+.3f}")

    print("\nPer-family internal coherence (mean |corr| among members):")
    for f in uniq:
        members = [i for i in range(K) if fam_of[i] == f]
        if len(members) < 2:
            print(f"  {f:12s} n={len(members):2d}  (singleton/too few)")
            continue
        sub = C[np.ix_(members, members)]
        iu = np.triu_indices(len(members), 1)
        print(f"  {f:12s} n={len(members):2d}  mean|corr|={sub[iu].mean():.3f}")

    print("\nVerdict heuristic: ARI>0.4 and within-between>0.2 => families are")
    print("real on this data (keep semantic labels). Otherwise switch the group-")
    print("lasso groups to these empirical clusters and report the mismatch.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    main(n)
