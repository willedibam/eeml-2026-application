#!/usr/bin/env python3
"""
Build the FNSZ-vs-GNSZ transfer manifest from the Phase-0 label mirror.

Consumes the .csv term-annotation tree pulled by the label-only rsync (see
tuh_discovery.py) and emits one line per session token:

    edf_relpath,csv_relpath,split,label

paths are relative to the corpus root on the ISIP server
(data/tuh_eeg/tuh_eeg_seizure/v2.0.6/), so stage_data.sh can rsync them
directly.

Design decisions baked in:
  * Only FNSZ and GNSZ sessions. A session carrying BOTH is dropped -- its
    class label would be ambiguous and it is the exact case where a "focal vs
    generalized" label is not well defined.
  * The official train/dev/eval split is preserved verbatim. It is
    patient-disjoint by construction, which is the leakage guard the pooled
    cross-subject EEG failure in CLAUDE.md needs.
  * Optional --balance caps FNSZ per split to the GNSZ count (GNSZ is the
    minority: 140/81/48 sessions vs 435/143/77). Capping is done by patient,
    not by session, so a patient never straddles the cap.

Usage:
    python docs/tuh_build_manifest.py data/tuh/tusz_labels \\
        --out data/tuh/manifest.csv --balance
    python docs/tuh_build_manifest.py data/tuh/tusz_labels \\
        --out data/tuh/manifest_pilot.csv --balance --max-per-class 20
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FOCUS = ("fnsz", "gnsz")


def session_types(csv_path: Path) -> set[str]:
    types: set[str] = set()
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            for cell in row:
                lab = cell.strip().lower()
                if lab in FOCUS:
                    types.add(lab)
    return types


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("labels_dir")
    p.add_argument("--out", default="data/tuh/manifest.csv")
    p.add_argument("--balance", action="store_true",
                   help="Cap FNSZ to the GNSZ session count per split "
                        "(capped by patient, so patients never straddle).")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Further cap sessions per class per split (pilot).")
    a = p.parse_args()

    root = Path(a.labels_dir)
    # split -> label -> patient -> [(edf_rel, csv_rel)]
    found: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for c in sorted(root.rglob("*.csv")):
        rel = c.relative_to(root)
        if len(rel.parts) < 4:
            continue
        split, patient = rel.parts[0], rel.parts[1]
        types = session_types(c)
        if len(types) != 1:
            continue                      # none, or ambiguous (both classes)
        label = types.pop()
        edf_rel = f"edf/{rel.with_suffix('.edf')}"
        csv_rel = f"edf/{rel}"
        found[split][label][patient].append((edf_rel, csv_rel))

    out_rows: list[tuple[str, str, str, str]] = []
    print(f"{'split':6s} {'label':6s} {'patients':>9s} {'sessions':>9s}  (after filters)")
    for split in ("train", "dev", "eval"):
        counts = {l: sum(len(v) for v in found[split][l].values()) for l in FOCUS}
        cap = min(counts.values()) if a.balance else None
        for label in FOCUS:
            pats = sorted(found[split][label])
            taken, n_sess = [], 0
            limit = cap if cap is not None else float("inf")
            if a.max_per_class is not None:
                limit = min(limit, a.max_per_class)
            for pat in pats:
                sess = found[split][label][pat]
                if n_sess + len(sess) > limit and n_sess > 0:
                    continue              # keep patients whole
                taken.append(pat)
                n_sess += len(sess)
                if n_sess >= limit:
                    break
            for pat in taken:
                for edf_rel, csv_rel in found[split][label][pat]:
                    out_rows.append((edf_rel, csv_rel, split, label))
            print(f"{split:6s} {label:6s} {len(taken):9d} {n_sess:9d}")

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        for r in out_rows:
            w.writerow(r)
    print(f"\nWrote {len(out_rows)} rows -> {outp}")
    print(f"Estimated transfer at ~22 MB/EDF: {len(out_rows)*22/1024:.1f} GB")


if __name__ == "__main__":
    main()
