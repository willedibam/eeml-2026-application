"""Gate 1 learners. Each is fitted on one cell's training surrogates only.

LP is a lattice probe tailored to the effect library (a ceiling stand-in, not
a generic method); MR (MiniRocket) and CNN are generic time-domain learners.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAGS = 8
LEARNERS = ("lp", "mr", "cnn")


def _standardise(v: np.ndarray) -> np.ndarray:
    centred = v - v.mean(axis=-1, keepdims=True)
    return centred / np.maximum(centred.std(axis=-1, keepdims=True), 1e-12)


def lattice_probe_features(x: np.ndarray) -> np.ndarray:
    """Per-channel spectra and moments; per-pair circular cross-correlations of z and z^2."""
    n, channels, length = x.shape
    z = _standardise(x)
    q = _standardise(z * z)
    fz, fq = np.fft.rfft(z, axis=-1), np.fft.rfft(q, axis=-1)
    power = np.abs(fz[..., 1:]) ** 2 / length
    edges = np.unique(np.round(np.logspace(0, np.log10(power.shape[-1]), 17)).astype(int)) - 1
    bands = np.stack(
        [power[..., lo:hi].mean(-1) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo], -1
    )
    irreversibility = (np.diff(z, axis=-1) ** 3).mean(-1)
    out = [
        np.log(bands).reshape(n, -1),
        skew(z, axis=-1),
        kurtosis(z, axis=-1),
        irreversibility,
    ]
    log_power = _standardise(np.log(power + 1e-12))
    lags = np.r_[np.arange(LAGS + 1), np.arange(length - LAGS, length)]
    for j in range(channels):
        for k in range(j + 1, channels):
            for a, b in ((fz, fz), (fq, fq), (fz, fq), (fq, fz)):
                r = np.fft.irfft(np.conj(a[:, j]) * b[:, k], n=length, axis=-1) / length
                out += [r[:, lags], np.abs(r).max(-1, keepdims=True)]
            out.append((log_power[:, j] * log_power[:, k]).mean(-1, keepdims=True))
    return np.column_stack(out)


def _lattice_probe(x_train, y_train, x_test, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 9),
            cv=3,
            max_iter=2000,
            scoring="accuracy",
            use_legacy_attributes=False,
            l1_ratios=(0.0,),
        ),
    )
    model.fit(lattice_probe_features(x_train), y_train)
    return model.predict(lattice_probe_features(x_test))


def _minirocket(x_train, y_train, x_test, seed):
    from aeon.transformations.collection.convolution_based import MiniRocket

    transform = MiniRocket(n_kernels=10_000, random_state=seed, n_jobs=1)
    train = transform.fit_transform(x_train.astype(np.float32))
    test = transform.transform(x_test.astype(np.float32))
    model = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    model.fit(train, y_train)
    return model.predict(test)


def _cnn(x_train, y_train, x_test, seed, epochs=30, patience=6, batch=64):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    validation = np.concatenate(
        [rng.permutation(np.flatnonzero(y_train == c))[: (y_train == c).sum() // 5] for c in (0, 1)]
    )
    fit = np.setdiff1d(np.arange(len(y_train)), validation)

    def block(i, o, d):
        return [nn.Conv1d(i, o, 9, dilation=d, padding=4 * d), nn.BatchNorm1d(o), nn.GELU()]

    body = nn.Sequential(
        *block(x_train.shape[1], 32, 1), *block(32, 32, 2), *block(32, 32, 4), *block(32, 32, 8)
    )
    head = nn.Linear(64, 2)

    def logits(data):
        h = body(data)
        return head(torch.cat([h.mean(-1), h.std(-1)], 1))

    params = list(body.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    best, best_loss, stale = None, np.inf, 0
    for _ in range(epochs):
        body.train(), head.train()
        order = rng.permutation(fit)
        for start in range(0, len(order), batch):
            idx = torch.as_tensor(order[start : start + batch])
            loss = nn.functional.cross_entropy(logits(xt[idx]), yt[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        body.eval(), head.eval()
        with torch.no_grad():
            val = nn.functional.cross_entropy(logits(xt[validation]), yt[validation]).item()
        if val < best_loss - 1e-4:
            best_loss, stale = val, 0
            best = [p.detach().clone() for p in params] + [
                b.detach().clone() for b in body.buffers()
            ]
        else:
            stale += 1
            if stale >= patience:
                break
    with torch.no_grad():
        for target, saved in zip(params + list(body.buffers()), best):
            target.copy_(saved)
        body.eval(), head.eval()
        return logits(torch.as_tensor(x_test, dtype=torch.float32)).argmax(1).numpy()


def fit_predict(name: str, x_train, y_train, x_test, seed: int) -> np.ndarray:
    if name == "lp":
        return _lattice_probe(x_train, y_train, x_test, seed)
    if name == "mr":
        return _minirocket(x_train, y_train, x_test, seed)
    if name == "cnn":
        return _cnn(x_train, y_train, x_test, seed)
    raise ValueError(f"Unknown learner {name}")
