#!/usr/bin/env python3
"""
Fast regression guard for the headline model.

Why this exists. On 2026-07 the group lasso gained sqrt(|g|) size
normalisation, which multiplies each family's penalty by sqrt(|g|) (6.9x for
causal, 7.2x for spectral). The --group-lambda default was not re-tuned
alongside it, so the shipped default silently over-regularised spi_w and cost
~0.35 macro-F1 -- and nothing complained. A single cheap assertion on the
headline model would have caught it the day it landed.

This trains spi-mpnn on a small VAR subset with the CURRENT CLI defaults and
asserts it still reaches a sane macro-F1. It is deliberately coarse: it is a
smoke alarm for "a hyperparameter/optimiser change broke the main model", not
a benchmark.

Run:
    .venv/bin/python tests/test_regression.py        # exits 1 on failure
    .venv/bin/python -m pytest tests/ -v             # if pytest installed

Skips (exit 0) when the VAR dataset is absent, so a fresh clone is not a
failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph_build import (  # noqa: E402
    SPIScaler, assign_spi_families, build_graph, filter_spi_dimensions,
    load_spi_names, load_spi_tensor,
)
from src.model import SPIEdgeMPNN  # noqa: E402
from src.run_pipeline import parse_args  # noqa: E402
from src.train import TrainConfig, train_model  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "260327_eeml"
CLASSES = ["var-chain", "var-fork", "var-collider"]
N_TRAIN = 100          # per class
N_TEST = 100           # per class
MIN_F1 = 0.85          # measured ~0.97 at the retuned default; 0.63 when broken


def _ok(d: Path) -> bool:
    return ((d / "spi_mpis.npz").exists() and (d / "timeseries.npy").exists()
            and (d / "meta.json").exists())


def _data_available() -> bool:
    return DATA.is_dir() and all((DATA / c).is_dir() for c in CLASSES)


def _build():
    dirs = {c: sorted(d for d in (DATA / c).iterdir() if _ok(d)) for c in CLASSES}
    spi_names = load_spi_names(dirs[CLASSES[0]][0])

    def load(dlist, label):
        return [(load_spi_tensor(d, spi_names),
                 np.load(d / "timeseries.npy").astype(np.float64), label)
                for d in dlist]

    tr, te = [], []
    for i, c in enumerate(CLASSES):
        tr += load(dirs[c][:N_TRAIN], i)
        te += load(dirs[c][N_TRAIN:N_TRAIN + N_TEST], i)

    names, idx = filter_spi_dimensions(spi_names, [t for t, _, _ in tr])
    scaler = SPIScaler().fit([t[:, :, idx] for t, _, _ in tr])

    def build(raw):
        return [build_graph(scaler.transform(t[:, :, idx]), m, l) for t, m, l in raw]

    _, fam = assign_spi_families(names)
    return build(tr), build(te), len(names), fam


def test_spi_mpnn_not_regressed() -> None:
    """spi-mpnn with CLI defaults must still separate the VAR motifs."""
    if not _data_available():
        print(f"SKIP: {DATA} not present")
        return

    import torch
    torch.manual_seed(0)
    np.random.seed(0)

    train, test, K, fam = _build()
    Fn = train[0].x.shape[1]

    # Pull the real CLI defaults so this tracks whatever is shipped.
    d = parse_args(["--data-dir", "x", "--class-names", *CLASSES])
    print(f"defaults: group_lambda={d.group_lambda} l1={d.l1_lambda} "
          f"top_d={d.top_d} warmup={d.warmup_epochs} restarts={d.restarts}")

    cfg = TrainConfig(
        lr=d.lr, batch_size=d.batch_size, max_epochs=d.max_epochs,
        patience=d.patience, device="cpu",
        l1_lambda=d.l1_lambda, group_lambda=d.group_lambda,
        spi_family_indices=list(fam.values()) if d.group_lambda > 0 else None,
        group_size_norm=not d.no_group_size_norm,
        warmup_epochs=d.warmup_epochs, restarts=d.restarts,
    )
    model = SPIEdgeMPNN(n_spi=K, n_node_features=Fn, n_classes=len(CLASSES),
                        top_d=d.top_d)
    res = train_model(model, train, test, test, cfg)

    w_norm = float(np.linalg.norm(res.learned_w))
    rms = {f: float(np.linalg.norm(res.learned_w[ix]) / max(len(ix), 1) ** 0.5)
           for f, ix in fam.items()}
    top_family = max(rms, key=rms.get)
    print(f"test_f1={res.test_f1:.3f}  ||w||={w_norm:.4f}  top_family={top_family}")
    print("  family RMS: " + "  ".join(f"{f}={v:.4f}" for f, v in
                                       sorted(rms.items(), key=lambda kv: -kv[1])))

    assert res.test_f1 >= MIN_F1, (
        f"spi-mpnn REGRESSED: test F1 {res.test_f1:.3f} < {MIN_F1} at "
        f"n={N_TRAIN}/class with shipped defaults (group_lambda="
        f"{d.group_lambda}, size_norm={not d.no_group_size_norm}). "
        "A hyperparameter or optimiser change likely broke the headline model "
        "-- check the group-lasso scale first."
    )
    assert w_norm > 1e-3, (
        f"spi_w collapsed to ~0 (||w||={w_norm:.2e}): the probe is being "
        "over-regularised, so the learned signature is meaningless."
    )


def main() -> int:
    try:
        test_spi_mpnn_not_regressed()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
