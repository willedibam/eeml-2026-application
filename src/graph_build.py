"""
Graph construction from pyspi output.

Converts SPI (M, M, K) tensors + raw MTS into PyG Data objects.
Sparsification and adjacency weighting happen inside model forward passes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .features import node_features
from .utils import load_json


def load_spi_tensor(dataset_dir: Path, spi_names: list[str]) -> np.ndarray:
    """
    Load SPI matrices from spi_mpis.npz and stack into (M, M, K) tensor.

    NaN/inf values are replaced with 0. The caller is responsible for
    filtering bad dimensions before training (see filter_spi_dimensions).
    """
    npz_path = dataset_dir / "spi_mpis.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    with np.load(npz_path) as npz:
        mats = []
        for name in spi_names:
            if name not in npz:
                raise KeyError(f"SPI '{name}' not in {npz_path}")
            mats.append(np.asarray(npz[name], dtype=np.float64))

    tensor = np.stack(mats, axis=-1)  # (M, M, K)
    return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def compute_pearson_corr(mts: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix from (T, M) MTS.

    Returns (M, M) float32. Diagonal is set to 0.
    Assumes MTS is already z-scored (if not, np.corrcoef handles it anyway).
    """
    corr = np.corrcoef(mts.T).astype(np.float32)  # (M, M)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def build_graph(
    spi_tensor: np.ndarray,
    mts: np.ndarray,
    label: int,
) -> Data:
    """
    Build a PyG Data object from a scaled SPI tensor and raw MTS.

    Stores:
        x:            (M, F_n) node features (from raw MTS)
        spi_tensor:   (M, M, K) SPI descriptor tensor (scaled)
        pearson_corr: (M, M) Pearson correlation matrix (from raw MTS)
        y:            scalar label
        num_nodes:    M

    Graph sparsification and adjacency weighting happen inside each model's
    forward pass, so this function always stores the dense representation.
    """
    M, _, K = spi_tensor.shape

    x = node_features(mts)  # (M, F_n)
    pearson = compute_pearson_corr(mts)  # (M, M)

    return Data(
        x=torch.from_numpy(x),
        spi_tensor=torch.from_numpy(spi_tensor.astype(np.float32)),
        pearson_corr=torch.from_numpy(pearson),
        y=torch.tensor(label, dtype=torch.long),
        num_nodes=M,
    )


# ---------------------------------------------------------------------------
# SPI filtering and scaling
# ---------------------------------------------------------------------------

def filter_spi_dimensions(
    spi_names: list[str],
    tensors: list[np.ndarray],
    *,
    max_missing_rate: float = 0.05,
    min_variance: float = 1e-8,
) -> tuple[list[str], list[int]]:
    """
    Drop SPI dimensions that are mostly missing or have near-zero variance
    across the training set.

    Returns (retained_names, retained_indices).
    """
    retained_names, retained_indices = [], []
    K = len(spi_names)

    for k in range(K):
        all_vals = []
        n_total = n_missing = 0
        for t in tensors:
            M = t.shape[0]
            mask = ~np.eye(M, dtype=bool)
            vals = t[:, :, k][mask]
            n_total += vals.size
            n_missing += (~np.isfinite(vals)).sum()
            all_vals.append(vals[np.isfinite(vals)])

        if n_missing / max(n_total, 1) > max_missing_rate:
            print(f"[SPI-FILTER] Dropping '{spi_names[k]}': missing {n_missing/n_total:.1%}")
            continue

        pooled = np.concatenate(all_vals) if all_vals else np.array([])
        if pooled.size < 2 or pooled.var() < min_variance:
            print(f"[SPI-FILTER] Dropping '{spi_names[k]}': var={pooled.var():.2e}")
            continue

        retained_names.append(spi_names[k])
        retained_indices.append(k)

    print(f"[SPI-FILTER] Retained {len(retained_names)}/{K} SPI dimensions")
    return retained_names, retained_indices


class SPIScaler:
    """
    Per-SPI-dimension robust scaling (median / IQR) fitted on training data.

    Fit on off-diagonal values from (M, M, K) tensors.
    Transform applied to all data including test.
    """

    def __init__(self):
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, tensors: list[np.ndarray]) -> "SPIScaler":
        K = tensors[0].shape[2]
        all_vals: list[list] = [[] for _ in range(K)]

        for t in tensors:
            M = t.shape[0]
            mask = ~np.eye(M, dtype=bool)
            for k in range(K):
                finite = t[:, :, k][mask]
                finite = finite[np.isfinite(finite)]
                all_vals[k].append(finite)

        self.median_ = np.zeros(K)
        self.iqr_ = np.ones(K)

        for k in range(K):
            pooled = np.concatenate(all_vals[k]) if all_vals[k] else np.array([0.0])
            q25, q50, q75 = np.percentile(pooled, [25, 50, 75])
            self.median_[k] = q50
            iqr = q75 - q25
            self.iqr_[k] = iqr if iqr > 1e-12 else 1.0

        self.is_fitted = True
        return self

    def transform(
        self, tensor: np.ndarray, *, clip: float = 10.0
    ) -> np.ndarray:
        """Scale and clip to [-clip, clip] to prevent extreme outliers
        from producing NaN in downstream softplus/linear layers."""
        if not self.is_fitted:
            raise RuntimeError("SPIScaler not fitted")
        scaled = (tensor - self.median_) / self.iqr_
        if clip > 0:
            scaled = np.clip(scaled, -clip, clip)
        return scaled


# ---------------------------------------------------------------------------
# SPI family assignment (pyspi taxonomy)
# ---------------------------------------------------------------------------

_FAMILY_RULES: list[tuple[str, list[str]]] = [
    # --- Linear: first/second-moment coupling ---
    ("linear", [
        "cov_", "cov-sq_",             # covariance (± squared)
        "prec_", "prec-sq_",           # precision (± squared)
        "xcorr_",                       # cross-correlation
        "lmfit_",                       # linear regression models (Ridge, Lasso, SGD, ElasticNet, BayesianRidge)
    ]),
    # --- Rank: monotone dependence, distribution-free ---
    ("rank", [
        "spearmanr",                    # includes spearmanr and spearmanr-sq
        "kendalltau",                   # includes kendalltau and kendalltau-sq
    ]),
    # --- Spectral: frequency-domain coupling (undirected) ---
    ("spectral", [
        "phase_",                       # coherence phase
        "cohmag_",                      # coherence magnitude
        "icoh_",                        # imaginary coherence
        "plv_",                         # phase locking value
        "pli_",                         # phase lag index
        "wpli_",                        # weighted phase lag index
        "dspli_",                       # debiased squared PLI
        "dswpli_",                      # debiased squared wPLI
        "ppc_",                         # pairwise phase consistency
        "pec",                          # power envelope correlation (exact match + pec_*)
    ]),
    # --- Causal/directed: directed temporal precedence ---
    ("causal", [
        "tlmi_",                        # time-lagged mutual information
        "te_",                          # transfer entropy
        "gc_",                          # Granger causality (time-domain)
        "sgc_",                         # spectral Granger causality
        "psi_",                         # phase slope index (directed spectral)
        "di_",                          # directed information
        "cce_",                         # causal entropy
        "xme_",                         # crossmap entropy
        "reci",                         # regression error causal inference
        "igci",                         # information-geometric conditional independence
    ]),
    # --- Information: nonlinear dependence (undirected) ---
    ("information", [
        "mi_",                          # mutual information
        "si_",                          # stochastic interaction
        "ids",                          # interdependence score
    ]),
    # --- Distance: shape similarity ---
    ("distance", [
        "pdist_",                       # pairwise distance
        "xpdist_",                      # cross pairwise distance
        "gwtau",                        # Gromov-Wasserstein
        "dtw",                          # dynamic time warping
    ]),
]


def assign_spi_families(
    spi_names: list[str],
) -> tuple[list[str], dict[str, list[int]]]:
    """
    Assign each SPI to a family based on pyspi naming conventions.

    Returns:
        family_names: per-SPI family string (same length as spi_names)
        family_indices: {family_name: [indices]}
    """
    family_names: list[str] = []
    family_indices: dict[str, list[int]] = {}

    for i, name in enumerate(spi_names):
        assigned = False
        for family, prefixes in _FAMILY_RULES:
            if any(name.startswith(p) or name == p for p in prefixes):
                family_names.append(family)
                family_indices.setdefault(family, []).append(i)
                assigned = True
                break
        if not assigned:
            family_names.append("other")
            family_indices.setdefault("other", []).append(i)

    return family_names, family_indices


def load_spi_names(dataset_dir: Path) -> list[str]:
    """Extract ordered SPI names from meta.json."""
    meta = load_json(dataset_dir / "meta.json")
    spis = meta.get("pyspi", {}).get("spis", [])
    return [s["name"] for s in spis if isinstance(s, dict) and "name" in s]
