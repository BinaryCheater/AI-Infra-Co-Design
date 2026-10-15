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


@lru_cache(maxsize=None)
def methods() -> dict:
    """The optimization methods level by level (methods.yaml): the domain know-how a proposal chooses from."""
    return yaml.safe_load((HERE / "methods.yaml").read_text())


def method_entry(mid: str) -> dict | None:
    return next((m for m in methods()["methods"] if m["id"] == mid), None)


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
    for mid in T.get("methods") or []:            # a proposal chooses from the method list; it cannot invent one
        m = method_entry(mid)
        if m is None:
            problems.append(f"method {mid!r} is not in the method list (methods.yaml)")
        elif m["level"] != T.get("level"):
            problems.append(f"method {mid!r} belongs to level {m['level']}, not {T.get('level')}")
    return problems
