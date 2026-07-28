#!/usr/bin/env python3
"""
Validity audit: for every result JSON, decide whether its signature is readable.

An interpretable method is only worth its interpretation if something can tell
you when the interpretation is void. This project has three such gates, and each
has already rejected one of its own results:

  node-only   Node features alone (mean, std, lag-1 autocorr, dominant FFT mag),
              no graph. If this clears chance, the classes differ in what
              individual channels DO rather than how they couple; the SPI
              vocabulary is mismatched and any coupling claim is void.
              CAUGHT: regime R1 (var_nonlinear_a) scored 0.913 with node
              features alone (chance 0.5). The nonlinearity sat in the state
              update, so each node's AR structure changed with its in-degree and
              the motif leaked through the marginals. Result withdrawn.

  shuffled    SPI values permuted across pairs, marginal distribution intact. If
              this clears chance, the signal is in the distribution of SPI
              values, not in which pair has which value, and "learned topology"
              means nothing.

  at-chance   The primary model itself. A signature read from a classifier at
              chance is group-lasso geometry, not recovered mechanism.
              CAUGHT: R1b's first report printed module M05 at 4.66x, z=11.7
              from a model scoring F1 0.415 against 0.333 chance.

A fourth check is structural rather than empirical: under a SYMMETRIC statistic,
chain and fork are Markov-equivalent (Verma & Pearl 1990), so any undirected
method is capped at 2/3 on the three-motif task. That is what retired Kuramoto,
and it is a proof, not a measurement.

    PYTHONPATH=. python docs/validity_report.py results/*.json [--margin 0.05]

Chance is 1/len(classes) from the JSON unless --chance is given; pass it
explicitly when --class-labels merged classes (e.g. collider-vs-rest).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GATES = ("node-only", "shuffled")
PRIMARY = ("spi-mpnn", "fixed-spi", "mlp-mix")


def audit(path: Path, margin: float, chance_override: float | None):
    try:
        r = json.load(open(path))
    except Exception as e:                       # noqa: BLE001 - report, don't crash
        return {"tag": path.stem, "verdict": f"UNREADABLE ({type(e).__name__})"}
    se = r.get("results")
    if not se:
        return {"tag": path.stem, "verdict": "NO RESULTS BLOCK"}
    n_key = max(se, key=int)
    models = se[n_key].get("models", {})
    classes = r.get("classes") or []
    chance = chance_override or (1 / len(classes) if classes else float("nan"))

    def f1(m):
        return models[m]["f1_mean"] if m in models else None

    def sig_above(m):
        """Is model m separated from chance by >2 seed SDs AND by `margin`?

        A fixed margin alone is not enough: 0.412 against a 0.333 chance level
        reads as 'above chance' at margin 0.05 but is not distinguishable from
        it across seeds. Both conditions must hold.
        """
        if m not in models:
            return None
        d = models[m]
        return (d["f1_mean"] - 2 * d.get("f1_std", 0.0) > chance
                and d["f1_mean"] > chance + margin)

    row = {
        "tag": path.stem.replace("sample_efficiency_", "").replace("_results", ""),
        "n": n_key,
        "K": r.get("n_spi"),
        "lam": r.get("hyperparameters", {}).get("group_lambda"),
        "chance": chance,
        "classes": len(classes),
        "gates": {g: f1(g) for g in GATES},
        "primary": next(((m, f1(m)) for m in PRIMARY if f1(m) is not None),
                        (None, None)),
    }

    pm, pf = row["primary"]
    leaks = [g for g in GATES if sig_above(g)]
    run = [g for g, v in row["gates"].items() if v is not None]
    if pf is None:
        row["verdict"] = "NO PRIMARY MODEL"
    elif leaks:
        why = {"node-only": "per-channel marginals suffice",
               "shuffled": "pair correspondence not needed"}
        row["verdict"] = "VOID: " + "; ".join(why[g] for g in leaks)
    elif not sig_above(pm):
        row["verdict"] = "VOID: primary not above chance"
    elif not run:
        row["verdict"] = "UNGATED (no control run)"
    elif len(run) < len(GATES):
        row["verdict"] = f"PARTIAL ({', '.join(run)} only)"
    else:
        row["verdict"] = "INTERPRETABLE"
    return row


def main(paths: list[Path], margin: float, chance: float | None) -> None:
    rows = [audit(p, margin, chance) for p in paths]
    print(f"{'result':32} {'n':>5} {'K':>4} {'lam':>7} {'chance':>7} "
          f"{'node-only':>10} {'shuffled':>9} {'primary':>9}  verdict")
    order = {"VOID": 0, "UNGATED": 1, "PARTIAL": 2, "NO": 3, "INTERPRETABLE": 4}
    rows.sort(key=lambda d: order.get(d["verdict"].split(":")[0].split()[0], 9))
    for d in rows:
        if "gates" not in d:
            print(f"{d['tag'][:32]:32} {'':>5} {'':>4} {'':>7} {'':>7} "
                  f"{'':>10} {'':>9} {'':>9}  {d['verdict']}")
            continue
        g = d["gates"]
        fmt = lambda v: f"{v:.4f}" if v is not None else "--"   # noqa: E731
        pm, pf = d["primary"]
        print(f"{d['tag'][:32]:32} {d['n']:>5} {str(d['K']):>4} {str(d['lam']):>7} "
              f"{d['chance']:>7.3f} {fmt(g['node-only']):>10} "
              f"{fmt(g['shuffled']):>9} {fmt(pf):>9}  {d['verdict']}"
              + (f"  [{pm}]" if pm and pm != "spi-mpnn" else ""))

    n_void = sum(d["verdict"].startswith("VOID") for d in rows)
    n_ok = sum(d["verdict"] == "INTERPRETABLE" for d in rows)
    print(f"\n  {len(rows)} results: {n_ok} interpretable, {n_void} void, "
          f"{len(rows) - n_ok - n_void} ungated/partial/unreadable.")
    print("  A void result is not a wasted one -- it is the protocol working.")
    print("  An UNGATED result is the dangerous case: nothing has tested it.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    margin, chance = 0.05, None
    for flag, cast in (("--margin", float), ("--chance", float)):
        if flag in argv:
            i = argv.index(flag)
            val = cast(argv[i + 1])
            margin, chance = (val, chance) if flag == "--margin" else (margin, val)
            del argv[i:i + 2]
    paths = [Path(a) for a in argv]
    if not paths:
        print(__doc__)
        sys.exit(1)
    main(paths, margin, chance)
