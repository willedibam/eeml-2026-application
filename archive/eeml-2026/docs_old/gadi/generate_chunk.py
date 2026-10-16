#!/usr/bin/env python3
"""
Process one manifest chunk into TUSZ per-instance dirs, parallel across cores.

Called by tuh_generate.pbs inside each array task. Reads manifest rows
[index*chunk : (index+1)*chunk] and runs the generator on each session with a
process pool (pyspi is single-threaded per instance, so parallelism is over
instances).

The per-session generator (`generate_session`) is the reference in
../tuh_generator_reference.py, ported into the sibling compute repo
(mts-spi-study-cluster) as module `tuh_generator`. This driver imports it by
name; set PYTHONPATH to the sibling repo (the .pbs does this).
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tuh_generator import generate_session          # sibling repo
except ImportError:  # fall back to the reference shipped here
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tuh_generator_reference import generate_session


def read_chunk(manifest: Path, index: int, chunk: int) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with manifest.open(newline="") as f:
        for r in csv.reader(f):
            if not r or r[0].startswith("#"):
                continue
            rows.append((r[0], r[1], r[2]))
    return rows[index * chunk:(index + 1) * chunk]


def _one(args) -> tuple[str, int | str]:
    edf, csv_lab, split, out, config = args
    try:
        n = generate_session(Path(edf), Path(csv_lab), split, Path(out), config)
        return (edf, n)
    except Exception as e:                               # keep the sweep alive
        return (edf, f"ERROR: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    a = p.parse_args()

    work = read_chunk(Path(a.manifest), a.index, a.chunk)
    print(f"chunk {a.index}: {len(work)} sessions, {a.workers} workers")
    tasks = [(e, c, s, a.out, a.config) for e, c, s in work]

    written = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for fut in as_completed(futs):
            edf, res = fut.result()
            if isinstance(res, int):
                written += res
            else:
                errors += 1
                print(f"  {edf}: {res}")
    print(f"chunk {a.index}: wrote {written} instances, {errors} session errors")


if __name__ == "__main__":
    main()
