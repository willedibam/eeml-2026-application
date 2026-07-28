#!/usr/bin/env python3
"""
Test the TUH conversion on ONE staged session, before committing to the corpus.

docs/tuh_generator_reference.py has never been run against a real EDF. The
three things most likely to be wrong, in order:

  1. Montage electrodes. TCP_AR_PAIRS assumes 22 bipolar derivations exist in
     every recording. If any electrode is missing, M changes between sessions
     and the whole pipeline breaks (the SPI tensor and the batched `series`
     attribute both assume fixed M). load_bipolar raises rather than silently
     dropping, which is correct but means we need to know the failure rate.
  2. Sampling rate. TUSZ is nominally 250 Hz but varies; resampling must land
     every session on a common fs or window lengths differ.
  3. Seizure interval parsing. The .csv term files must yield usable spans of
     the requested type.

Usage (on Gadi, after staging):
    python docs/tuh_test_one.py /scratch/ql44/we2614/tusz/edf [n_sessions]

Prints a per-session report and a summary. Does NOT run pyspi -- this is about
whether the EDF -> windows path works at all.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tuh_generator_reference as G  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    edfs = sorted(root.rglob("*.edf"))
    if not edfs:
        print(f"No .edf under {root} -- has staging run?")
        sys.exit(1)
    print(f"found {len(edfs)} staged EDFs; testing {min(limit, len(edfs))}\n")

    fs_seen, m_seen, errs = Counter(), Counter(), Counter()
    n_ok = n_win = 0

    for edf in edfs[:limit]:
        lab = edf.with_suffix(".csv")
        try:
            sig, fs = G.load_bipolar(edf)
            fs_seen[fs] += 1
            m_seen[sig.shape[0]] += 1
            proc, fs2 = G.preprocess(sig, fs)
            spans = []
            for t in ("fnsz", "gnsz"):
                if lab.exists():
                    spans += G.read_seizure_intervals(lab, t)
            wins = []
            for sp in spans:
                wins += list(G.windows_from_span(proc, fs2, sp))
            n_ok += 1
            n_win += len(wins)
            shape = wins[0].shape if wins else None
            print(f"  OK   {edf.name[:34]:34} fs={fs:6.1f}->{fs2:5.1f} "
                  f"M={sig.shape[0]:2d} spans={len(spans):2d} windows={len(wins):3d} "
                  f"win_shape={shape}")
        except Exception as e:
            errs[f"{type(e).__name__}: {str(e)[:70]}"] += 1
            print(f"  FAIL {edf.name[:34]:34} {type(e).__name__}: {str(e)[:70]}")

    print(f"\n--- summary ---")
    print(f"  sessions ok      : {n_ok}/{min(limit, len(edfs))}")
    print(f"  total windows    : {n_win}")
    print(f"  sampling rates   : {dict(fs_seen)}")
    print(f"  channel counts M : {dict(m_seen)}   <-- MUST be a single value")
    if errs:
        print("  errors:")
        for k, v in errs.most_common():
            print(f"    {v:3d}x  {k}")
    if len(m_seen) > 1:
        print("\n  *** M varies across sessions. The pipeline assumes fixed M;")
        print("      either restrict to sessions with the full montage, or cut")
        print("      TCP_AR_PAIRS down to electrodes present in all of them.")


if __name__ == "__main__":
    main()
