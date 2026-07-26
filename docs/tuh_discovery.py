#!/usr/bin/env python3
"""
TUSZ v2.0.6 Phase-0 discovery: tally FNSZ vs GNSZ by seizure type, patient,
and official split — BEFORE moving any EDF.

Run against a local mirror of just the term-based annotation files:

    rsync -auvxL -r -e "ssh -i ~/.ssh/id_ed25519" \
      --include='*/' --include='*.csv' --exclude='*' \
      nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_seizure/v2.0.6/edf/ \
      ./tusz_labels/

    python docs/tuh_discovery.py ./tusz_labels

Path layout mirrored under the root:
    <split>/<patient>/<session>/<montage>/<token>.csv
where <split> in {train, dev, eval} (patient-disjoint by construction).

The term-based .csv rows are: channel,start_time,stop_time,label,confidence
(header lines start with '#'). `label` carries the seizure TYPE (fnsz, gnsz,
cpsz, absz, ...) or bckg. The .csv_bi variant only has seiz/bckg — not enough
to separate focal from generalized, so we parse .csv here.

Output: per-split counts of sessions and unique patients carrying each seizure
type, plus the FNSZ/GNSZ instance-budget picture (GNSZ is the limiter).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

SEIZURE_TYPES = {  # TUSZ term labels that are seizure types (not bckg)
    "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "cnsz",
    "tcsz", "atsz", "mysz", "nesz",
}
FOCUS = ("fnsz", "gnsz")  # the directed-vs-symmetric contrast


def parse_labels(csv_path: Path) -> set[str]:
    """Return the set of seizure-type labels present in one term .csv."""
    types: set[str] = set()
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            # be tolerant of header row / column order variants
            for cell in row:
                lab = cell.strip().lower()
                if lab in SEIZURE_TYPES:
                    types.add(lab)
    return types


def main(root: str) -> None:
    root_p = Path(root)
    csvs = sorted(root_p.rglob("*.csv"))
    if not csvs:
        print(f"No .csv under {root_p} — did the label rsync run?")
        sys.exit(1)

    # split -> type -> counts
    sess_by = defaultdict(lambda: defaultdict(int))
    pats_by = defaultdict(lambda: defaultdict(set))
    sessions_seen = 0

    for c in csvs:
        rel = c.relative_to(root_p).parts
        if len(rel) < 4:
            continue  # not <split>/<patient>/<session>/<montage>/file.csv
        split, patient, session = rel[0], rel[1], rel[2]
        sessions_seen += 1
        types = parse_labels(c)
        for t in types:
            sess_by[split][t] += 1
            pats_by[split][t].add(patient)

    print(f"Scanned {len(csvs)} .csv across {sessions_seen} session-montage dirs\n")
    splits = ["train", "dev", "eval"]
    print(f"{'type':6s} | " + " | ".join(f"{s:>18s}" for s in splits))
    print("-" * 70)
    all_types = sorted({t for s in sess_by.values() for t in s})
    for t in all_types:
        cells = []
        for s in splits:
            n_sess = sess_by[s].get(t, 0)
            n_pat = len(pats_by[s].get(t, set()))
            cells.append(f"{n_sess:5d} sess / {n_pat:4d} pat")
        star = "  <-- FOCUS" if t in FOCUS else ""
        print(f"{t:6s} | " + " | ".join(cells) + star)

    print("\nFNSZ vs GNSZ budget (GNSZ is the limiter for balanced classes):")
    for s in splits:
        f_pat = len(pats_by[s].get("fnsz", set()))
        g_pat = len(pats_by[s].get("gnsz", set()))
        f_ses = sess_by[s].get("fnsz", 0)
        g_ses = sess_by[s].get("gnsz", 0)
        print(f"  {s:5s}: FNSZ {f_ses} sess / {f_pat} pat   "
              f"GNSZ {g_ses} sess / {g_pat} pat")
    print("\nNext: if GNSZ patient counts support a balanced set, pull EDFs for "
          "FNSZ+GNSZ sessions only (to Gadi scratch) and run the pilot.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python docs/tuh_discovery.py <tusz_labels_dir>")
        sys.exit(1)
    main(sys.argv[1])
