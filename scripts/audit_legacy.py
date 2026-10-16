"""Inventory legacy metadata without loading SPI banks or claiming validity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(root: Path) -> dict:
    rows = []
    for bank in sorted(root.iterdir()):
        if not bank.is_dir():
            continue
        counts, missing, generators, shapes = Counter(), Counter(), Counter(), Counter()
        n_records = 0
        unreadable = []
        for path in sorted(bank.glob("**/meta.json")):
            n_records += 1
            try:
                meta = json.loads(path.read_text())
                if not isinstance(meta, dict):
                    raise TypeError("Metadata is not an object")
            except (OSError, ValueError, TypeError) as exc:
                unreadable.append({"path": str(path.relative_to(root)), "error": str(exc)})
                continue
            spi = meta.get("pyspi", {})
            counts[str(spi.get("n_spis", "unknown"))] += 1
            generators[str(meta.get("generator", {}).get("name", "unknown"))] += 1
            shapes[f"M{meta.get('M', '?')}_T{meta.get('T', '?')}"] += 1
            for field in ("computation_version", "version", "config_sha256"):
                if not spi.get(field):
                    missing[field] += 1
            for filename in ("timeseries.npy", "spi_mpis.npz"):
                if not (path.parent / filename).is_file():
                    missing[filename] += 1
        rows.append(
            {
                "bank": bank.name,
                "n_records": n_records,
                "catalogue_counts": dict(counts),
                "generators": dict(generators),
                "shapes": dict(shapes),
                "missing": dict(missing),
                "unreadable_metadata": unreadable,
            }
        )
    return {
        "scope": "Metadata inventory only; no certification of results or sources",
        "banks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    for bank in report["banks"]:
        print(
            json.dumps(
                {
                    **{k: v for k, v in bank.items() if k != "unreadable_metadata"},
                    "unreadable_metadata_count": len(bank["unreadable_metadata"]),
                }
            )
        )


if __name__ == "__main__":
    main()
