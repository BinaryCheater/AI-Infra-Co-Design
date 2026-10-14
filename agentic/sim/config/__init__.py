"""Simulator configuration: the measured contract plus the structural choices (§5), typed as
measured / structural / fixed so the orchestrator can tell extrapolation from description."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    cfg = yaml.safe_load((HERE / f"{name}.yaml").read_text())
    contract = yaml.safe_load((ROOT / cfg["contract"]).read_text())
    return {"name": name, "contract": contract, "structural": cfg["structural"], "fixed": cfg["fixed"], "kinds": cfg["kinds"]}


def available() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.yaml"))
