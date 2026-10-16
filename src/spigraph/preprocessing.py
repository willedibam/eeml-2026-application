"""Training-only SPI preprocessing; tensors are (source, target, statistic).

Directed entries are never symmetrised. Diagonals are not observations. Each
fit belongs to exactly one training fold AND training-budget subset.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _tensor(value: np.ndarray, n_spi: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError("Expected a square (M, M, K) tensor with M >= 2")
    if value.shape[2] != n_spi:
        raise ValueError("SPI catalogue length does not match tensor")
    return value


def load_spi_tensor(path: str | Path, names: Sequence[str]) -> np.ndarray:
    """Load an explicitly ordered catalogue, preserving non-finite values."""
    if not names or len(set(names)) != len(names):
        raise ValueError("SPI names must be nonempty and unique")
    with np.load(path, allow_pickle=False) as bank:
        if set(bank.files) != set(names):
            raise ValueError("Archive catalogue differs from the declared catalogue")
        tensor = np.stack([bank[name] for name in names], axis=-1)
    return _tensor(tensor, len(names))


class SPIPreprocessor:
    """Select, median-impute, IQR-scale and clip using finite training edges.

    Missingness remains in the returned mask. Zero after scaling is the
    TRAINING median, not absent interaction. Pooling weights recordings by edge
    count; balance M or explicitly change the estimator for variable-M studies.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        max_missing: float = 0.05,
        min_variance: float = 1e-8,
        clip: float = 10.0,
    ):
        if not names or len(set(names)) != len(names):
            raise ValueError("SPI names must be nonempty and unique")
        if not 0 <= max_missing < 1 or not np.isfinite(min_variance) or min_variance < 0:
            raise ValueError("Invalid missingness/variance threshold")
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        self.names = tuple(names)
        self.max_missing = max_missing
        self.min_variance = min_variance
        self.clip = clip

    def fit(self, training: Sequence[np.ndarray]) -> SPIPreprocessor:
        for attr in ("indices_", "median_", "scale_", "missing_rate_"):
            self.__dict__.pop(attr, None)
        if not len(training):
            raise ValueError("Training tensors are empty")
        edges = []
        for value in training:
            value = _tensor(value, len(self.names))
            edges.append(value[~np.eye(value.shape[0], dtype=bool)])
        pooled = np.concatenate(edges)
        finite = np.isfinite(pooled)
        missing = 1.0 - finite.mean(axis=0)
        indices, median, scale = [], [], []
        for k in range(len(self.names)):
            values = pooled[finite[:, k], k]
            if missing[k] > self.max_missing or len(values) < 2:
                continue
            if values.var() <= self.min_variance:
                continue
            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            indices.append(k)
            median.append(q50)
            scale.append(q75 - q25 if q75 - q25 > 1e-12 else 1.0)
        if not indices:
            raise ValueError("No valid, varying SPI dimensions in this training subset")
        self.indices_ = np.array(indices, dtype=int)
        self.median_ = np.array(median)
        self.scale_ = np.array(scale)
        self.missing_rate_ = missing
        return self

    def transform(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "indices_"):
            raise RuntimeError("Fit on the current training subset first")
        value = _tensor(value, len(self.names))[:, :, self.indices_]
        valid = np.isfinite(value)
        diagonal = np.arange(value.shape[0])
        valid[diagonal, diagonal, :] = False
        imputed = np.where(valid, value, self.median_)
        scaled = np.clip((imputed - self.median_) / self.scale_, -self.clip, self.clip)
        return scaled.astype(np.float32), valid

    def state(self) -> dict:
        """JSON-serialisable fitted state to retain alongside a run."""
        if not hasattr(self, "indices_"):
            raise RuntimeError("Preprocessor is not fitted")
        return {
            "catalogue": list(self.names),
            "indices": self.indices_.tolist(),
            "median": self.median_.tolist(),
            "scale": self.scale_.tolist(),
            "missing_rate": self.missing_rate_.tolist(),
            "clip": self.clip,
            "max_missing": self.max_missing,
            "min_variance": self.min_variance,
            "pooling": "ordered_offdiagonal_edges",
        }
