"""Illustrative data for Fig 1: the same VAR(1) chain process used in R0.

Regenerated locally rather than loaded from the cluster -- the figure is
pedagogical, and a 5-node chain reads at poster distance where M=10 does not.
The PROCESS is the real one (linear VAR(1), chain motif), not a cartoon, so the
SPI matrices shown are genuine outputs of the statistics they claim to be.
"""
from __future__ import annotations

import numpy as np

M, T = 5, 52
CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4)]      # 0 -> 1 -> 2 -> 3 -> 4


def series(seed: int = 3) -> np.ndarray:
    """(M, T) z-scored VAR(1) chain."""
    rng = np.random.default_rng(seed)
    A = np.zeros((M, M))
    for i, j in CHAIN:
        A[j, i] = 0.55                         # parent i drives child j
    np.fill_diagonal(A, 0.62)
    x = np.zeros((M, T + 300))
    for t in range(1, x.shape[1]):
        x[:, t] = A @ x[:, t - 1] + rng.normal(0, 1, M)
    x = x[:, 300:]
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def spi_matrices(x: np.ndarray) -> dict[str, np.ndarray]:
    """Three genuine pairwise statistics, chosen to look DIFFERENT from each other.

    That is the whole point of the panel: a single scalar edge weight throws away
    the fact that these three disagree about which pairs are coupled.
    """
    M_, T_ = x.shape
    cov = np.corrcoef(x)                                   # undirected, contemporaneous
    lag = np.zeros((M_, M_))                               # directed, lag-1
    for i in range(M_):
        for j in range(M_):
            lag[i, j] = np.corrcoef(x[i, :-1], x[j, 1:])[0, 1]
    spec = np.zeros((M_, M_))                              # undirected, band-limited
    X = np.fft.rfft(x, axis=1)
    band = slice(1, max(2, X.shape[1] // 4))
    for i in range(M_):
        for j in range(M_):
            num = np.abs((X[i, band] * np.conj(X[j, band])).mean())
            den = np.sqrt((np.abs(X[i, band]) ** 2).mean()
                          * (np.abs(X[j, band]) ** 2).mean())
            spec[i, j] = num / den
    for m in (cov, lag, spec):
        np.fill_diagonal(m, np.nan)
    return {"cov": np.abs(cov), "lag1": np.abs(lag), "coh": spec}
