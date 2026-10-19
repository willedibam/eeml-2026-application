"""Lattice cells: group-orbit surrogates and the two non-group reference cells.

Arrays are (records, channels, time). Group cells act on each record with an
independent Haar-random element, so a cell carries exactly the information in
the group's maximal invariant. Inputs must already be z-scored per channel.
"""

from __future__ import annotations

import numpy as np

PAIRED = ("X", "CS", "IS", "IO", "G1", "G0")
UNPAIRED = ("XR", "U0")
CELLS = PAIRED + UNPAIRED


def zscore(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(axis=-1, keepdims=True)
    return centred / centred.std(axis=-1, keepdims=True)


def _roll(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """out[..., t] = x[..., (t - lag) mod T], lags broadcast over (records, channels)."""
    index = (np.arange(x.shape[-1]) - lags[..., None]) % x.shape[-1]
    return np.take_along_axis(x, np.broadcast_to(index, x.shape), axis=-1)


def common_shift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _roll(x, rng.integers(x.shape[-1], size=(len(x), 1)))


def independent_shift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _roll(x, rng.integers(x.shape[-1], size=x.shape[:2]))


def _rotate(x: np.ndarray, rng: np.random.Generator, common: bool) -> np.ndarray:
    coef = np.fft.rfft(x, axis=-1)
    shape = (len(x), 1 if common else x.shape[1], coef.shape[-1])
    rotation = np.exp(2j * np.pi * rng.random(shape))
    rotation[..., 0] = 1  # DC is real
    if x.shape[-1] % 2 == 0:
        rotation[..., -1] = 1  # so is the Nyquist bin
    return np.fft.irfft(coef * rotation, n=x.shape[-1], axis=-1)


def common_phase(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _rotate(x, rng, common=True)


def independent_phase(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _rotate(x, rng, common=False)


def independent_offset(long: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Crop each channel of a longer record at its own uniform offset (no seams)."""
    offsets = rng.integers(long.shape[-1] - length + 1, size=long.shape[:2])
    index = offsets[..., None] + np.arange(length)
    return zscore(np.take_along_axis(long, index, axis=-1))


def within_class_shuffle(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Channel k of output record i comes from a random same-class record (XR)."""
    out = np.empty_like(x)
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        for channel in range(x.shape[1]):
            out[members, channel] = x[rng.permutation(members), channel]
    return out


def cell(
    name: str, long: np.ndarray, y: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Surrogate dataset for one cell from (records, channels, 2T) raw records."""
    if name == "IO":
        return independent_offset(long, length, rng)
    x = zscore(long[..., :length])
    if name == "X":
        return x
    if name == "CS":
        return common_shift(x, rng)
    if name == "IS":
        return independent_shift(x, rng)
    if name == "G1":
        return common_phase(x, rng)
    if name == "G0":
        return independent_phase(x, rng)
    if name == "XR":
        return within_class_shuffle(x, y, rng)
    if name == "U0":
        return independent_phase(within_class_shuffle(x, y, rng), rng)
    raise ValueError(f"Unknown cell {name}")
