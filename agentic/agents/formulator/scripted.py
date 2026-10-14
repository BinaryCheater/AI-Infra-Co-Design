"""The scripted Formulator: replays the paper's §6.2 study on the tiling template with no model calls.

Each step is a refinement proposal. The script covers every path of the loop:
  admitted designs on three GEMM shapes, an incumbent that improves when the exact precondition
  replaces the conservative one (the 1.34% of the paper), a regime flip on the skinny GEMM, an
  unsound precondition refuted at family certification with a replayed race, and a structurally
  wrong template (floor grid) refuted at the instance gate with a replayed coverage defect.

The regime each proposal assumes is read from the evidence in the request, as a model would read it,
so the flip on the skinny workload is a consequence of the rule, not a scripted outcome.
"""
from __future__ import annotations

from ...examples.gemm_tiling import tiling_contract
from ...templates.smem_tiling import Workload

SQUARE = Workload("square_4096", 4096, 4096, 4096)
SKINNY = Workload("skinny_64x4096x4096", 64, 4096, 4096)
RAGGED = Workload("ragged_1000x1000x1000", 1000, 1000, 1000)

SCRIPT = [
    {"workload": SQUARE, "precondition": "conservative", "refines": "logical tensors -> smem tiles (trailing barrier kept)",
     "rationale": "First tiling proposal. Keep the trailing barrier always: sound by inspection, may cost latency."},
    {"workload": SQUARE, "precondition": "exact", "refines": "lifetime precondition -> weakest (P <= S-2+beta)",
     "rationale": "Replace the conservative barrier rule by the exact lifetime precondition so the solver may drop "
                  "the trailing barrier when a slot is free."},
    {"workload": SKINNY, "precondition": "exact", "refines": "reuse tmpl on skinny GEMM (M=64)",
     "rationale": "Same template on a skinny shape. Regime assumed from the latest evidence."},
    {"workload": RAGGED, "precondition": "conservative", "refines": "ragged GEMM, trailing barrier kept",
     "rationale": "Ragged 1000^3 shape; start conservative again on a new shape."},
    {"workload": RAGGED, "precondition": "exact", "refines": "ragged GEMM, exact precondition",
     "rationale": "Exact precondition on the ragged shape; expect the incumbent to improve (paper: 1.34%)."},
    {"workload": RAGGED, "precondition": "unsound", "refines": "drop beta from the precondition (P <= S-1)",
     "rationale": "Plausible but wrong: without the trailing barrier a slow warp can still be reading the slot."},
    {"workload": RAGGED, "precondition": "exact", "structural": {"grid_rounding": "floor"},
     "refines": "grid = floor(M/Tm) x floor(N/Tn)",
     "rationale": "Drop the partial tail tiles to simplify the kernel. (Wrong: the tail is uncovered.)"},
]


def _regime_from_evidence(evidence: list[dict]) -> str:
    for e in evidence:
        if e.get("kind") == "bottleneck":
            r = e["regime"]
            return f"{r}: dominant resource {e['resource']} at util {e['utilization']:.2f}"
    return "compute-bound: util(compute) >= util(DRAM)"


EXTRAPOLATION_STEP = {
    "workload": RAGGED, "precondition": "exact", "chip_override": {"smem_bytes_per_block": 131072},
    "refines": "chip with 128 KiB shared memory per block (not this machine)",
    "rationale": "What if the chip had 128 KiB of shared memory per block? A hardware knob: this machine cannot run it.",
}


class ScriptedFormulator:
    name = "scripted"

    def __init__(self, script: list[dict] | None = None, chip: dict | None = None, extrapolation: bool = False):
        self.script = list(script if script is not None else SCRIPT)
        if extrapolation:
            self.script.append(EXTRAPOLATION_STEP)
        self.chip = chip                        # a scope.chip dict (measured or illustrative); None = illustrative

    def __len__(self):
        return len(self.script)

    def propose(self, request: dict) -> dict | None:
        i = request["step"]
        if i >= len(self.script):
            return None
        s = self.script[i]
        structure = {
            "name": "bottleneck_dominance", "bucket": "objective",
            "regime": _regime_from_evidence(request.get("evidence", [])),
            "move": "epigraph cost T >= max(T_comp, T_mem); search the tile menu with prefetch overlap",
            "obligation": "the dominant resource stays dominant across the feasible region of the menu",
            "strength": "observed",
            "evidence": [e["artifact"] for e in request.get("evidence", []) if e.get("artifact")],
        }
        parent = request.get("coarse_template")
        chip = self.chip
        if s.get("chip_override"):
            chip = {**(chip or tiling_contract(s["workload"])["scope"]["chip"]), **s["chip_override"]}
        return tiling_contract(s["workload"], hw=chip, precondition=s["precondition"], structure=structure,
                               rationale=s["rationale"], parent=parent, structural=s.get("structural"),
                               formulator=self.name, step=i, refines=s["refines"])
