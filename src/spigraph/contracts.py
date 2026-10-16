"""Fail-closed checks between feature extraction and evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class ComputationIdentity:
    """Hash resolved config/recipe, not filenames; preserve catalogue order."""

    pyspi_version: str
    computation_version: str
    pyspi_revision: str
    config_sha256: str
    recipe_sha256: str
    catalogue: tuple[str, ...]
    orientation: str = "source_target"

    def __post_init__(self):
        for value in (self.pyspi_version, self.computation_version, self.pyspi_revision):
            if not isinstance(value, str) or value.strip().lower() in ("", "unknown", "legacy"):
                raise ValueError("Unknown computation identity cannot enter a new bank")
        if len(self.pyspi_revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.pyspi_revision
        ):
            raise ValueError("pyspi_revision must be a full Git SHA")
        if not _digest(self.config_sha256) or not _digest(self.recipe_sha256):
            raise ValueError("Config and extraction recipe must have SHA-256 digests")
        if (
            not isinstance(self.catalogue, tuple)
            or not self.catalogue
            or any(not isinstance(name, str) or not name for name in self.catalogue)
            or len(set(self.catalogue)) != len(self.catalogue)
        ):
            raise ValueError("Catalogue must be an ordered tuple of unique names")
        if self.orientation != "source_target":
            raise ValueError("Normalise orientation explicitly before banking")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def require_same_computation(identities: Sequence[ComputationIdentity]) -> None:
    if not identities:
        raise ValueError("Empty feature bank")
    if len({identity.fingerprint for identity in identities}) != 1:
        raise ValueError("Mixed computation identities: recompute into separate banks")


@dataclass(frozen=True)
class SplitRecord:
    """group_id is the subject/trajectory, not its window or crop identifier."""

    sample_id: str
    group_id: str
    label: str
    split: str
    source_sha256: str


def validate_classification_split(records: Sequence[SplitRecord]) -> dict:
    """Check declared splits; adapters must assign parent groups correctly.

    Does not choose splits or detect near-duplicates. Hash canonical sample
    contents and group all related crops/observations before calling.
    """
    if not records:
        raise ValueError("Empty split manifest")
    ids, groups, sources = set(), {}, {}
    counts = {split: {} for split in ("train", "validation", "test")}
    for row in records:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (row.sample_id, row.group_id, row.label)
        ):
            raise ValueError("Sample, group and label IDs are required; no fallback IDs")
        if row.sample_id in ids:
            raise ValueError(f"Duplicate sample ID: {row.sample_id}")
        ids.add(row.sample_id)
        if row.split not in counts or not _digest(row.source_sha256):
            raise ValueError("Invalid split or source hash")
        for mapping, key, kind in (
            (groups, row.group_id, "group"),
            (sources, row.source_sha256, "source"),
        ):
            if key in mapping and mapping[key] != row.split:
                raise ValueError(f"Cross-split {kind} overlap: {key}")
            mapping[key] = row.split
        counts[row.split][row.label] = counts[row.split].get(row.label, 0) + 1
    labels = {row.label for row in records}
    if len(labels) < 2 or any(set(c) != labels for c in counts.values()):
        raise ValueError("Every split must contain every class (at least two classes)")
    return {"counts": counts, "n_groups": len(groups), "n_samples": len(ids)}
