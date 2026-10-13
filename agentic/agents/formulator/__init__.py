"""Formulators: propose(request) -> contract. Only this agent may be a large model.

`scripted` replays a fixed sequence of proposals with no model calls (S3). `catalog()` returns the
structure catalog as data, which is the vocabulary every proposal's `structure` block must use.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def catalog() -> list[dict]:
    return yaml.safe_load((HERE / "catalog.yaml").read_text())


def catalog_entry(name: str) -> dict | None:
    return next((e for e in catalog() if e["name"] == name), None)


def well_formed(T: dict) -> list[str]:
    """The Orchestrator's pre-solver check: the structure block answers one of the three questions."""
    problems = []
    s = T.get("structure") or {}
    entry = catalog_entry(s.get("name", ""))
    if entry is None:
        problems.append(f"structure {s.get('name')!r} is not in the catalog")
    elif entry["bucket"] != s.get("bucket"):
        problems.append(f"structure {s['name']} belongs to bucket {entry['bucket']}, not {s.get('bucket')}")
    if not s.get("regime"):
        problems.append("structure.regime is empty")
    if not s.get("obligation"):
        problems.append("structure.obligation is empty")
    if s.get("strength") not in ("proved", "observed"):
        problems.append("structure.strength must be proved or observed")
    return problems
