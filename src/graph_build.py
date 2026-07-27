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
        series:       (M, T) raw per-channel time series (node-level attribute)
        spi_tensor:   (M, M, K) SPI descriptor tensor (scaled)
        pearson_corr: (M, M) Pearson correlation matrix (from raw MTS)
        y:            scalar label
        num_nodes:    M

    Graph sparsification and adjacency weighting happen inside each model's
    forward pass, so this function always stores the dense representation.

    `series` is a node-level attribute (first dim = M), so PyG batches it to
    (B*M, T) exactly like `x`. It lets the latent baselines build their
    adjacency from the raw signal (a fair, NRI/MTGNN-style temporal encoder)
    rather than from four marginal summaries alone. All graphs in a dataset
    must share the same T for PyG concatenation; the current generators
    produce fixed-T series.
    """
    M, _, K = spi_tensor.shape

    x = node_features(mts)  # (M, F_n)
    pearson = compute_pearson_corr(mts)  # (M, M)
    series = np.ascontiguousarray(mts.T, dtype=np.float32)  # (M, T)

    return Data(
        x=torch.from_numpy(x),
        series=torch.from_numpy(series),
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


def literature_spi_modules(
    spi_names: list[str],
) -> tuple[list[str], dict[str, list[int]]]:
    """Group SPIs by the PUBLISHED modules of Cliff et al. (2023).

    pyspi tags every SPI with a module label (M01-M14) from the empirical
    modular decomposition in "Unifying pairwise interactions in complex
    dynamics"; MXX marks SPIs added to the fork that are not in the paper.
    Those labels are derived from measured inter-SPI similarity across a large
    dataset corpus, so they are both data-driven AND externally validated --
    strictly better grounded than either the hand-assigned families here
    (which do not match the empirical structure on VAR, ARI 0.06) or an
    ad-hoc clustering refit per dataset.

    Prefer this grouping when the recovered group is the scientific claim: it
    lets the result be stated in the literature's own vocabulary rather than
    in categories invented for this repo.

    The mapping is cached in src/spi_modules.json (generated from pyspi), so
    this repo does not need pyspi installed. SPIs absent from the cache are
    assigned "MXX".
    """
    path = Path(__file__).parent / "spi_modules.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; regenerate it from pyspi (see docs)."
        )
    table = load_json(path)

    module_names = [table.get(n, "MXX") for n in spi_names]
    module_indices: dict[str, list[int]] = {}
    for i, m in enumerate(module_names):
        module_indices.setdefault(m, []).append(i)

    n_unmapped = sum(1 for n in spi_names if n not in table)
    sizes = {k: len(v) for k, v in sorted(module_indices.items())}
    print(f"[SPI-MODULES] literature modules {sizes}"
          + (f"  ({n_unmapped} unmapped -> MXX)" if n_unmapped else ""))
    return module_names, module_indices


def empirical_spi_modules(
    tensors: list[np.ndarray],
    n_modules: int = 6,
    *,
    max_instances: int = 200,
) -> tuple[list[str], dict[str, list[int]]]:
    """Group SPIs by how their OUTPUTS co-vary, not by their names.

    The hand-assigned families in _FAMILY_RULES are intuitive categories, but
    on the VAR data they do not recover the empirical structure of the SPI
    outputs (adjusted Rand index 0.06 vs an average-linkage clustering of
    1 - |corr|; within-family |corr| 0.25 vs 0.13 between). That matters
    because the group lasso penalises these groups and the headline claim is
    "family X carries the weight": if a family is not a coherent block, the
    penalty is regularising an incoherent bag and the claim is about a label
    rather than a mode of dependence.

    This returns data-driven modules (the on-data analogue of the module
    characterisation in Cliff et al. 2023), suitable for passing to
    TrainConfig.spi_family_indices in place of assign_spi_families.

    MUST be fitted on TRAINING tensors only -- the grouping is part of the
    model, so deriving it from val/test would leak.

    Returns (per-SPI module label, {module_name: [indices]}), matching the
    signature of assign_spi_families.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    rows = []
    for t in tensors[:max_instances]:
        M = t.shape[0]
        off = ~np.eye(M, dtype=bool)
        rows.append(t[off])                       # (M*(M-1), K)
    X = np.nan_to_num(np.concatenate(rows, axis=0), nan=0.0,
                      posinf=0.0, neginf=0.0)
    K = X.shape[1]

    C = np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0)
    np.fill_diagonal(C, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0

    # Ward, not average linkage. Average-linkage on 1-|corr| chains badly here:
    # measured on the VAR data (K=125) it puts 79% of SPIs in one cluster
    # (sizes 99/13/6/3/2/2), which is barely a grouping at all and would leave
    # the group lasso penalising one giant bag. Ward gives usable, balanced
    # modules (34/33/24/17/10/7, largest 27%); complete is intermediate (55%).
    # Caveat: ward assumes Euclidean dissimilarity and 1-|corr| is not strictly
    # Euclidean, so treat the modules as a practical grouping, not a metric
    # embedding.
    Z = linkage(squareform(D, checks=False), method="ward")
    labels = fcluster(Z, t=min(n_modules, K), criterion="maxclust")

    module_names = [f"module{int(l)}" for l in labels]
    module_indices: dict[str, list[int]] = {}
    for i, name in enumerate(module_names):
        module_indices.setdefault(name, []).append(i)

    sizes = {k: len(v) for k, v in sorted(module_indices.items())}
    print(f"[SPI-MODULES] {len(module_indices)} empirical modules, sizes {sizes}")
    return module_names, module_indices


def load_spi_names(dataset_dir: Path) -> list[str]:
    """Extract ordered SPI names from meta.json."""
    meta = load_json(dataset_dir / "meta.json")
    spis = meta.get("pyspi", {}).get("spis", [])
    return [s["name"] for s in spis if isinstance(s, dict) and "name" in s]
