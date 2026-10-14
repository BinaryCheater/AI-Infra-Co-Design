"""Hardware contracts: the illustrative one (paper Listing 1) and the measured ones from
calibration/<host>/contract.yaml. `hardware_contract` turns a contract's `scope.chip` into the
template backend's dataclass; `extrapolation` says which `fixed` fields a template changed relative
to the measured contract it names (notes/hardware_calibration_design.md §4)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ..templates.smem_tiling import HardwareContract

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "calibration"
ILLUSTRATIVE = "illustrative-fp32-gpu"


@lru_cache(maxsize=None)
def catalog() -> dict[str, dict]:
    chips = {ILLUSTRATIVE: {**HardwareContract().to_dict(), "kind": "illustrative"}}
    if CALIBRATION.exists():
        for f in sorted(CALIBRATION.glob("*/contract.yaml")):
            c = yaml.safe_load(f.read_text())
            c.setdefault("kind", "measured")
            chips[c["name"]] = c
    return chips


def load_chip(name: str) -> dict:
    try:
        return dict(catalog()[name])
    except KeyError:
        raise KeyError(f"no hardware contract named {name!r}; known: {sorted(catalog())}")


def scope_chip(chip: dict) -> dict:
    """What goes into a contract's scope.chip: the measured contract without its bulky provenance sources."""
    c = {k: v for k, v in chip.items() if k != "provenance"}
    if "provenance" in chip:
        p = chip["provenance"]
        c["provenance"] = {k: p.get(k) for k in ("host", "date", "suite_commit", "driver", "toolkit", "arch") if k in p}
        c["provenance"]["clocks"] = p.get("clocks", {})
    return c


def barrier_seconds(chip: dict, threads: int | None = None) -> float:
    b = chip["barrier_s"]
    if isinstance(b, dict):
        t = str(threads or chip.get("threads_per_block", 256))
        return float(b.get(t) or next(iter(b.values())))
    return float(b)


def hardware_contract(chip: dict) -> HardwareContract:
    """The template backend's dataclass from any chip dict (illustrative or measured)."""
    fields = {f: chip[f] for f in HardwareContract.__dataclass_fields__ if f in chip and f != "barrier_s"}
    fields["barrier_s"] = barrier_seconds(chip)
    return HardwareContract(**fields)


def extrapolation(chip: dict) -> dict:
    """Fields in `fixed` where this scope.chip differs from the measured contract of the same name.
    Empty when the chip is not measured or nothing fixed was changed."""
    if chip.get("kind") != "measured":
        return {}
    base = catalog().get(chip.get("name"))
    if base is None:
        return {"_unknown_measured_contract": chip.get("name")}
    out = {}
    for f in base.get("fixed", []):
        if f in chip and chip[f] != base[f]:
            out[f] = {"measured": base[f], "template": chip[f]}
    return out
