"""Exploratory toy check of symmetry-group surrogates as exact information ablations.

For a compact group G and Haar-random g independent of (X, Y), S = gX satisfies
I(Y; S) = I(Y; T(X)), where T is the maximal invariant. Four groups are checked:
CS common cyclic shift, IS independent per-channel shifts, G1 common Fourier
phase rotation (T = cross-periodogram), G0 independent phase rotation
(T = auto-periodograms). Planted effects should land in predicted cells.
A fixed feature set and logistic regression stand in for a strong learner.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

T, N, LAG, BURN = 1024, 400, 5, 200
rng = np.random.default_rng(0)


def phase(x: np.ndarray, common: bool) -> np.ndarray:
    coef = np.fft.rfft(x, axis=1)
    rot = np.exp(1j * rng.uniform(0, 2 * np.pi, (1 if common else len(x), coef.shape[1])))
    rot[:, 0] = 1
    rot[:, -1] = 1  # Nyquist bin for even T
    return np.fft.irfft(coef * rot, n=x.shape[1], axis=1)


def shift(x: np.ndarray, common: bool) -> np.ndarray:
    n = x.shape[1]
    lags = np.repeat(rng.integers(n), len(x)) if common else rng.integers(n, size=len(x))
    return np.stack([np.roll(c, k) for c, k in zip(x, lags)])


GROUPS = {
    "X": lambda x: x,
    "CS": lambda x: shift(x, True),
    "IS": lambda x: shift(x, False),
    "G1": lambda x: phase(x, True),
    "G0": lambda x: phase(x, False),
}


def ar(n: int, innovations: np.ndarray | None = None) -> np.ndarray:
    e = rng.standard_normal(n + BURN) if innovations is None else innovations
    return lfilter([1.0], [1.0, -0.6], e)[BURN:]


def record(task: str, label: int) -> np.ndarray:
    """Class differences planted in exactly one lattice cell; marginals matched."""
    x = ar(T + LAG)
    if task == "linear_coupling":  # second-order relational
        y = 0.5 * x[:-LAG] + ar(T) if label else np.sqrt(1.25) * ar(T)
    elif task == "quadratic_coupling":  # higher-order relational, zero cross-covariance
        centre = 1 / (1 - 0.36)
        source = x[:-LAG] if label else ar(T)
        y = (source**2 - centre) / 2 + ar(T)
    elif task == "univariate_skew":  # higher-order univariate
        n = T + LAG + BURN
        e = rng.exponential(size=n) - 1 if label else rng.standard_normal(n)
        x, y = ar(T + LAG, e), ar(T)
    else:
        raise ValueError(task)
    return np.stack([x[LAG:], y])


def features(x: np.ndarray) -> np.ndarray:
    z = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    out = []
    for v in (z, z**2 - (z**2).mean(1, keepdims=True)):
        f = np.fft.rfft(v, axis=1)
        cross = np.fft.irfft(f[0] * np.conj(f[1]), n=T) / T
        auto = np.fft.irfft(np.abs(f) ** 2, n=T, axis=1) / T
        out += [cross[:8], cross[-8:], auto[:, 1:6].ravel()]
    out += [skew(z, axis=1), kurtosis(z, axis=1)]
    return np.concatenate(out)


def main() -> None:
    x = record("linear_coupling", 1)
    cross = lambda v: np.einsum("jf,kf->jkf", np.fft.rfft(v), np.conj(np.fft.rfft(v)))  # noqa: E731
    print("G1 max cross-periodogram change:", np.abs(cross(phase(x, True)) - cross(x)).max())
    auto = lambda v: np.abs(np.fft.rfft(v))  # noqa: E731
    print("G0 max |auto-periodogram| change:", np.abs(auto(phase(x, False)) - auto(x)).max())
    for task in ("linear_coupling", "quadratic_coupling", "univariate_skew"):
        labels = rng.integers(2, size=N)
        records = [record(task, label) for label in labels]
        scores = {}
        for name, group in GROUPS.items():
            design = np.array([features(group(r)) for r in records])
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            scores[name] = round(float(cross_val_score(model, design, labels, cv=5).mean()), 3)
        print(task, scores)


if __name__ == "__main__":
    main()
