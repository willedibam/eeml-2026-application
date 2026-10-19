"""Planted-effect generators for Gate 1.

Channels 0-1 carry the effect; the rest are independent unit-variance AR(1).
For R effects, class 0 replaces the driving source with an independent copy of
the same law, so every channel's law is identical across classes. theta=0
makes the classes identical. Records have length 2T for the IO cell.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, lfilter

LAG = 5
BURN = 500
EFFECTS = ("U2", "U4", "R2", "R2dir", "R2dirP", "R4env", "R4quad", "R4pac")


def ar1(rng: np.random.Generator, n: int, length: int, a: float, innovations=None) -> np.ndarray:
    """Unit-variance AR(1) rows; innovations, if given, have unit variance."""
    e = rng.standard_normal((n, length + BURN)) if innovations is None else innovations
    return lfilter([np.sqrt(1 - a * a)], [1.0, -a], e, axis=-1)[:, BURN:]


def resonance(rng: np.random.Generator, n: int, length: int, freq: float, radius: float):
    """Narrowband AR(2) rows with spectral peak near freq (cycles/sample), unit variance."""
    den = [1.0, -2 * radius * np.cos(2 * np.pi * freq), radius**2]
    x = lfilter([1.0], den, rng.standard_normal((n, length + BURN)), axis=-1)[:, BURN:]
    return x / x.std(axis=-1, keepdims=True)


def _lagged_pair(rng, n, length, a, c, drive=None):
    """(leader, follower) with follower = c*g(leader)(t-LAG) + noise, same spectrum."""
    source = ar1(rng, n, length + LAG, a)
    driver = source if drive is None else drive(source)
    noise = ar1(rng, n, length, a)
    return source[:, LAG:], c * driver[:, :-LAG] + np.sqrt(1 - c * c) * noise


def simulate(
    effect: str, theta: float, y: np.ndarray, channels: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Raw records (n, channels, 2*length) for labels y in {0, 1}."""
    if effect not in EFFECTS or not 0 <= theta <= 1 or channels < 2:
        raise ValueError("Invalid effect, strength or channel count")
    n, total = len(y), 2 * length
    one = y.astype(bool)
    x = ar1(rng, n * channels, total, 0.6).reshape(n, channels, total)
    if effect == "U2":
        x[one, 0] = ar1(rng, int(one.sum()), total, 0.6 + 0.2 * theta)
    elif effect == "U4":
        k = int(one.sum())
        innovations = np.sqrt(1 - theta) * rng.standard_normal((k, total + BURN))
        innovations += np.sqrt(theta) * (rng.exponential(size=(k, total + BURN)) - 1)
        x[one, 0] = ar1(rng, k, total, 0.6, innovations)
    elif effect in ("R2", "R4quad"):
        square = (lambda s: (s * s - 1) / np.sqrt(2)) if effect == "R4quad" else (lambda s: s)
        lead, follow = _lagged_pair(rng, n, total, 0.6, 0.6 * theta, square)
        _, follow_ind = _lagged_pair(rng, n, total, 0.6, 0.6 * theta, square)
        x[:, 0] = lead
        x[:, 1] = np.where(one[:, None], follow, follow_ind)
    elif effect in ("R2dir", "R2dirP"):
        a = 0.95 if effect == "R2dirP" else 0.6
        lead, follow = _lagged_pair(rng, n, total, a, 0.6 * theta)
        x[:, 0] = np.where(one[:, None], lead, follow)
        x[:, 1] = np.where(one[:, None], follow, lead)
    elif effect == "R4env":
        beta = 1.0 * theta
        log_env = ar1(rng, n, total + LAG, 0.9)
        log_env_ind = ar1(rng, n, total + LAG, 0.9)
        shared = np.where(one[:, None], log_env, log_env_ind)
        x[:, 0] = np.exp(beta * log_env[:, LAG:]) * ar1(rng, n, total, 0.6)
        x[:, 1] = np.exp(beta * shared[:, :-LAG]) * ar1(rng, n, total, 0.6)
    elif effect == "R4pac":
        slow = resonance(rng, n, total + LAG, 1 / 32, 0.98)
        slow_ind = resonance(rng, n, total + LAG, 1 / 32, 0.98)
        phase = np.angle(hilbert(np.where(one[:, None], slow, slow_ind), axis=-1))
        x[:, 0] = slow[:, LAG:]
        x[:, 1] = (1 + theta * np.cos(phase[:, :-LAG])) * resonance(rng, n, total, 0.25, 0.9)
    return x
