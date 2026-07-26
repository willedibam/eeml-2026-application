#!/usr/bin/env python3
"""
Stability selection for the learned SPI probe w.

Replaces "the top-weighted SPIs" (a soft point estimate of a weakly-identified
parameter) with "the SPIs selected in X% of subsampled refits" -- the honest
unit for the interpretability claim. See train.stability_selection.

Usage (offline, local VAR data):
    .venv/bin/python docs/run_stability_selection.py \
        --data-dir data/260327_eeml \
        --class-names var-chain var-fork var-collider \
        --n-train 200 --n-subsamples 40 --max-epochs 60

Writes results/stability_selection_<tag>.json and prints a ranked table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.graph_build import (
    SPIScaler, assign_spi_families, build_graph, filter_spi_dimensions,
    load_spi_names, load_spi_tensor,
)
from src.model import SPIEdgeMPNN
from src.train import TrainConfig, stability_selection


def _ok(d: Path) -> bool:
    return ((d / "spi_mpis.npz").exists() and (d / "timeseries.npy").exists()
            and (d / "meta.json").exists())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--class-names", nargs="+", required=True)
    p.add_argument("--n-train", type=int, default=200, help="per class")
    p.add_argument("--n-val", type=int, default=100, help="per class")
    p.add_argument("--n-subsamples", type=int, default=40)
    p.add_argument("--subsample-frac", type=float, default=0.5)
    p.add_argument("--top-q", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--group-lambda", type=float, default=0.005)
    p.add_argument("--min-val-f1", type=float, default=None,
                   help="Drop subsample fits below this val F1 (their w is "
                        "arbitrary and dilutes every frequency). Suggested: "
                        "chance + 0.4, e.g. 0.75 for 3 balanced classes.")
    p.add_argument("--l1-lambda", type=float, default=0.001)
    p.add_argument("--top-d", type=int, default=5)
    p.add_argument("--tag", default="var")
    a = p.parse_args()

    root = Path(a.data_dir)
    dirs = {c: sorted(d for d in (root / c).iterdir() if _ok(d)) for c in a.class_names}
    spi_names = load_spi_names(dirs[a.class_names[0]][0])

    def load(dlist, label):
        return [(load_spi_tensor(d, spi_names),
                 np.load(d / "timeseries.npy").astype(np.float64), label)
                for d in dlist]

    train_raw, val_raw = [], []
    for i, c in enumerate(a.class_names):
        train_raw += load(dirs[c][:a.n_train], i)
        val_raw += load(dirs[c][a.n_train:a.n_train + a.n_val], i)

    names, idx = filter_spi_dimensions(spi_names, [t for t, _, _ in train_raw])
    K = len(names)
    scaler = SPIScaler().fit([t[:, :, idx] for t, _, _ in train_raw])

    def build(raw):
        return [build_graph(scaler.transform(t[:, :, idx]), m, l) for t, m, l in raw]

    train, val = build(train_raw), build(val_raw)
    Fn = train[0].x.shape[1]
    n_classes = len(a.class_names)
    fam_names, fam_idx = assign_spi_families(names)

    cfg = TrainConfig(
        max_epochs=a.max_epochs, patience=20, device="cpu",
        l1_lambda=a.l1_lambda, group_lambda=a.group_lambda,
        spi_family_indices=list(fam_idx.values()),
        warmup_epochs=min(60, a.max_epochs // 3), restarts=1,
    )

    def make_model():
        return SPIEdgeMPNN(n_spi=K, n_node_features=Fn,
                           n_classes=n_classes, top_d=a.top_d)

    print(f"K={K}  train={len(train)}  val={len(val)}  "
          f"subsamples={a.n_subsamples} @ frac={a.subsample_frac}")
    out = stability_selection(
        make_model, train, val, cfg,
        n_subsamples=a.n_subsamples, subsample_frac=a.subsample_frac,
        top_q=a.top_q, min_val_f1=a.min_val_f1,
    )
    print(f"converged fits used: {out['n_used']}/{out['n_subsamples']}"
          + (f" (threshold {a.min_val_f1})" if a.min_val_f1 else " (no threshold)"))
    out["spi_names"] = names
    out["families"] = fam_names

    freq = np.array(out["selection_frequency"])
    sign = np.array(out["sign_consistency"])
    order = np.argsort(freq)[::-1][:20]
    print(f"\nsubsample fits: val F1 {out['val_f1_mean']:.3f} "
          f"+/- {out['val_f1_std']:.3f}\n")

    # A selection frequency is only meaningful if the fits actually learned.
    # If the subsampled models sit at chance, w is unconstrained by the task
    # and the frequencies below are noise, not evidence.
    chance = 1.0 / n_classes
    if out["val_f1_mean"] < chance + 0.10:
        print(f"  *** WARNING: subsample fits are at/near chance ({chance:.2f}). "
              f"Selection frequencies are NOISE, not evidence.")
        print(f"  *** Increase --n-train / --max-epochs (canonical: 200 epochs, "
              f"60 warmup) or --subsample-frac before interpreting.\n")
        out["valid"] = False
    else:
        out["valid"] = True
    print(f"{'freq':>5s} {'sign':>5s}  {'family':12s} SPI")
    for i in order:
        if freq[i] <= 0:
            continue
        print(f"{freq[i]:5.2f} {sign[i]:5.2f}  {fam_names[i]:12s} {names[i]}")

    stable = int((freq >= 0.8).sum())
    print(f"\n{stable} SPIs selected in >=80% of subsamples "
          f"(top-q={a.top_q} of K={K}).")
    print("Report these, not the raw top-|w| ranking.")

    outp = Path("results") / f"stability_selection_{a.tag}.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"Saved {outp}")


if __name__ == "__main__":
    main()
