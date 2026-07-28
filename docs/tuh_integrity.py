#!/usr/bin/env python3
"""Find (and optionally delete) instance dirs left incomplete by a killed job.

`generate_session` writes each window as: mkdir -> timeseries.npy ->
spi_mpis.npz -> meta.json. meta.json is written LAST, so a dir holding an npz
but no meta.json was interrupted mid-write. A truncated npz is the dangerous
case -- it can load partially or raise only when a specific array is touched,
so every archive is opened and every member read.

    python docs/tuh_integrity.py <data_dir>            # report
    python docs/tuh_integrity.py <data_dir> --delete   # remove the broken ones

Deleting is safe: generation is idempotent, so a rerun of the owning shard
rewrites the instance identically.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np


def check(d: Path) -> str | None:
    if not (d / "meta.json").exists():
        return "no meta.json (interrupted before final write)"
    if not (d / "timeseries.npy").exists():
        return "no timeseries.npy"
    npz = d / "spi_mpis.npz"
    if not npz.exists():
        return "no spi_mpis.npz"
    try:
        with np.load(npz) as z:
            for k in z.files:
                _ = z[k].shape          # force decompression of every member
        np.load(d / "timeseries.npy")
    except Exception as e:                        # noqa: BLE001
        return f"unreadable ({type(e).__name__})"
    return None


def main(root: Path, delete: bool) -> None:
    bad, total = [], 0
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        n_bad = 0
        for d in sorted(p for p in cls.iterdir() if p.is_dir()):
            total += 1
            why = check(d)
            if why:
                bad.append((d, why)); n_bad += 1
        print(f"  {cls.name}: {len(list(cls.iterdir()))} dirs, {n_bad} broken")
    print(f"\n{total} instances, {len(bad)} broken")
    for d, why in bad[:15]:
        print(f"    {d.name}: {why}")
    if len(bad) > 15:
        print(f"    ... and {len(bad) - 15} more")
    if delete and bad:
        for d, _ in bad:
            shutil.rmtree(d)
        print(f"\ndeleted {len(bad)} incomplete instances")
    elif bad:
        print("\nrerun with --delete to remove them")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__); sys.exit(1)
    main(Path(args[0]), "--delete" in sys.argv)
