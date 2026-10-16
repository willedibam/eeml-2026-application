"""Counterexample to the archived universal 2/3 symmetric-statistic ceiling.

Exact stationary VAR moments suffice; no neural optimisation or pyspi needed.
The optional finite-sample check is exploratory, not a new paper benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spigraph.contracts import sha256_file

MOTIFS = {"chain": [(0, 1), (1, 2)], "fork": [(0, 1), (0, 2)], "collider": [(1, 0), (2, 0)]}


def stationary_symmetric_lags(motif: str, alpha: float, n_nodes: int = 10) -> np.ndarray:
    """S[l,i,j] = (|R_l[i,j]| + |R_l[j,i]|)/2 for l=0,1,2.

    X_t = A X_(t-1) + epsilon_t, unit independent innovations; nuisance nodes
    have AR coefficient 0.8. Scaling the common innovation variance cancels in
    correlations. Every S[l] is symmetric and has zero diagonal.
    """
    if motif not in MOTIFS or not 0 < alpha < 1 or n_nodes < 3:
        raise ValueError("Invalid motif, coupling or node count")
    transition = np.zeros((n_nodes, n_nodes))
    for source, target in MOTIFS[motif]:
        transition[target, source] = alpha
    for index in range(3, n_nodes):
        transition[index, index] = 0.8
    covariance = solve_discrete_lyapunov(transition, np.eye(n_nodes))
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    matrices = []
    for lag in range(3):
        corr = np.linalg.matrix_power(transition, lag) @ covariance / scale
        sym = (np.abs(corr) + np.abs(corr.T)) / 2
        np.fill_diagonal(sym, 0)
        matrices.append(sym)
    return np.stack(matrices)


def symmetric_lag_features(series: np.ndarray) -> np.ndarray:
    """Sorted unordered-pair values at each lag; no direction or node IDs."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 2 or min(series.shape) < 3 or not np.isfinite(series).all():
        raise ValueError("Expected finite (T,M) series with T,M >= 3")
    upper = np.triu_indices(series.shape[1], 1)
    features = []
    for lag in range(3):
        left = series[lag:]
        right = series[: len(series) - lag]
        left, right = left - left.mean(axis=0), right - right.mean(axis=0)
        denom = np.sqrt(np.outer((left * left).sum(axis=0), (right * right).sum(axis=0)))
        corr = np.divide(left.T @ right, denom, out=np.zeros_like(denom), where=denom > 0)
        symmetric = (np.abs(corr) + np.abs(corr.T)) / 2
        features.extend(np.sort(symmetric[upper]))
    return np.asarray(features)


def analytic_check() -> dict:
    rows = []
    for alpha in (0.3, 0.5, 0.7):
        for motif in MOTIFS:
            matrices = stationary_symmetric_lags(motif, alpha)
            lag0, lag2 = matrices[[0, 2]].sum(axis=(1, 2))
            prediction = "fork" if lag0 > 1e-10 else "chain" if lag2 > 1e-10 else "collider"
            if prediction != motif:
                raise AssertionError("Analytic counterexample failed")
            rows.append(
                {
                    "alpha": alpha,
                    "motif": motif,
                    "symmetric_lag0_sum": float(lag0),
                    "symmetric_lag2_sum": float(lag2),
                    "prediction": prediction,
                }
            )
    return {
        "scope": "Exact moments of the specified legacy VAR family, not universal recovery",
        "conclusion": "The universal symmetric-statistic 2/3 ceiling is false",
        "rows": rows,
    }


def simulate(seed: int, motif: str, n_time: int = 500, n_nodes: int = 10) -> np.ndarray:
    """Fresh realisations of the inspected VAR mechanism, not historical bytes."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.3, 0.7)
    noise = rng.normal(0, 0.1, size=(n_time + 200, n_nodes))
    series = noise.copy()
    # Acyclic motifs permit vectorised updates in topological order.
    order = MOTIFS[motif]
    for source, target in order:
        series[1:, target] += alpha * series[:-1, source]
    series[:, 3:] = lfilter([1], [1, -0.8], noise[:, 3:], axis=0)
    series = series[200:, rng.permutation(n_nodes)]
    return (series - series.mean(axis=0)) / series.std(axis=0)


def empirical_check(data_dir: Path | None, n_time: int = 500) -> dict:
    features, labels, sources = [], [], []
    seen = set()
    for label, motif in enumerate(MOTIFS):
        if data_dir is None:
            inputs = [
                (
                    f"synthetic/{motif}/{index}",
                    simulate(1729 + label * 100000 + index, motif, n_time=n_time),
                )
                for index in range(500)
            ]
        else:
            paths = sorted((data_dir / f"var-{motif}").glob("*/timeseries.npy"))
            if not paths:
                raise ValueError(f"No raw series for {motif}")
            inputs = [
                (str(path.relative_to(data_dir)), np.load(path, allow_pickle=False))
                for path in paths
            ]
        for name, series in inputs:
            digest = hashlib.sha256(np.ascontiguousarray(series, dtype="<f8").tobytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate source content: {name}")
            seen.add(digest)
            features.append(symmetric_lag_features(series))
            labels.append(label)
            sources.append({"path": name, "sha256": digest})
    x, y = np.stack(features), np.array(labels)
    scores = []
    for seed in (17, 29, 43, 71, 101):
        train, test = train_test_split(
            np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        scores.append(
            {
                "split_seed": seed,
                "accuracy": accuracy_score(y[test], pred),
                "macro_f1": f1_score(y[test], pred, average="macro"),
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "test_labels": y[test].tolist(),
                "test_predictions": pred.tolist(),
            }
        )
    return {
        "scope": "Exploratory mechanism-matched simulation"
        if data_dir is None
        else "Exploratory raw-series check; legacy generator provenance not certified",
        "generator": f"simulate(seed=1729+class*100000+index,T={n_time},M=10),500/class"
        if data_dir is None
        else None,
        "model": "StandardScaler(train only) + LogisticRegression(C=1,max_iter=2000)",
        "features": "sorted symmetric absolute cross-correlations at lags 0,1,2",
        "n_samples": len(y),
        "n_features": x.shape[1],
        "sources": sources,
        "scores": scores,
        "accuracy_mean": float(np.mean([s["accuracy"] for s in scores])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--n-time", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; do not overwrite an audit")
    repo = Path(__file__).resolve().parents[2]
    report = {
        "analytic": analytic_check(),
        "runner_sha256": sha256_file(__file__),
        "lock_sha256": sha256_file(repo / "uv.lock"),
        "environment": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "python": platform.python_version(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
    }
    if args.simulate and args.data_dir:
        parser.error("Choose fresh simulation or a historical data directory")
    if args.n_time < 3:
        parser.error("--n-time must be at least 3")
    if args.data_dir or args.simulate:
        report["empirical"] = empirical_check(args.data_dir, args.n_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "analytic": report["analytic"]["conclusion"],
                "empirical_accuracy": report.get("empirical", {}).get("accuracy_mean"),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
