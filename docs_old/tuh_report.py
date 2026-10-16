#!/usr/bin/env python3
"""Assemble the TUH report from per-model merged shards.

Training is sharded by (model, seed) so every model runs in its own process.
`src.analysis --merge` recombines SEEDS for one model; it must not be used to
combine different models, because `_merge_models` iterates the first shard's
model dict and would silently drop any model the first shard lacks. So models
are combined here, at report time, by reading one merged file per model.

    python docs/tuh_report.py results/

CONTROLS ARE PRINTED FIRST because on real EEG they decide whether anything else
means anything: if node-only clears chance the classes differ in what individual
channels do rather than how they couple, and the coupling claim is void.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# name -> (display, group). Groups are printed as separate tables.
KEYS = [
    ("node_only",       "node-only",       "controls"),
    ("shuffled",        "shuffled",        "controls"),
    ("spi_mpnn",        "spi-mpnn",        "main"),
    ("latent_directed", "latent-directed", "main"),
    ("fixed_spi",       "fixed-spi",       "main"),
    ("gl002",           "spi-mpnn λ=0.002", "lambda"),
    ("gl005",           "spi-mpnn λ=0.005", "lambda"),
]


def load(results: Path, key: str):
    f = results / f"tuh_{key}_merged_results.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    se = r["results"]
    # One model per file by construction; take whichever it is.
    out = {}
    for n in se:
        m = se[n]["models"]
        if not m:
            continue
        name = sorted(m)[0]
        out[int(n)] = (m[name]["f1_mean"], m[name]["f1_std"])
    return out, r.get("seeds"), f


def table(title: str, cols: list[tuple[str, dict]]) -> None:
    if not cols:
        return
    ns = sorted({n for _, d in cols for n in d})
    print(f"\n-------- {title} --------")
    print("  n    " + "".join(f"{lab:>22}" for lab, _ in cols))
    for n in ns:
        row = f"  {n:<5}"
        for _, d in cols:
            row += f"{d[n][0]:>15.4f}+/-{d[n][1]:.2f}" if n in d else f"{'--':>22}"
        print(row)


def main(results: Path) -> None:
    print("==================== TUH REPORT ====================")
    loaded, missing = {}, []
    for key, disp, grp in KEYS:
        got = load(results, key)
        if got is None:
            missing.append(key); continue
        loaded[key] = (disp, grp, got[0], got[1])
    for key in missing:
        print(f"-- {key} MISSING (see logs/tuh_{key}_s*.log)")

    for grp, title in (("controls", "CONTROLS -- read these first"),
                       ("main", "models"),
                       ("lambda", "lambda sensitivity (spi-mpnn)")):
        cols = [(d, v) for _, (d, g, v, _) in loaded.items() if g == grp]
        table(title, cols)

    seeds = {k: s for k, (_, _, _, s) in loaded.items()}
    print(f"\n  seeds per model: {seeds}")
    print("\nREAD THE CONTROLS FIRST.")
    print("  node-only near chance => classes differ in COUPLING, claim is live.")
    print("  node-only high        => classes differ in per-channel activity;")
    print("                           SPI vocabulary mismatched, claim is void.")
    print("  shuffled near chance  => pair correspondence carries the signal.")
    print("==================== END ====================")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(Path(sys.argv[1]))
