"""Run provenance: source state, lock and environment, recorded with every output."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from spigraph.contracts import sha256_file


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def source_state(repo: Path) -> dict:
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
    }


def run_provenance(repo: Path, files: Iterable[Path], packages: Iterable[str]) -> dict:
    versions = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "source": source_state(repo),
        "lock_sha256": sha256_file(repo / "uv.lock"),
        "files_sha256": {str(Path(f).relative_to(repo)): sha256_file(f) for f in files},
        "packages": versions,
        "python": platform.python_version(),
        "machine": platform.platform(),
    }
