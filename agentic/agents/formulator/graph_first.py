"""A proposer that starts at the top of the hierarchy for one chosen workload: its first proposal is the graph-level template
(agentic/examples/gemm_graph), with no model calls. Later levels are added step by step."""
from __future__ import annotations

from ...examples.gemm_graph import graph_contract


class GraphFirstFormulator:
    name = "graph-first"

    def __init__(self, workload: dict, chip: dict):
        self.workload = dict(workload)
        self.chip = chip

    def propose(self, request: dict) -> dict | None:
        if request["step"] == 0:
            w = self.workload
            return graph_contract(
                {k: w[k] for k in ("name", "m", "n", "k")}, self.chip, parent=request.get("coarse_template"), step=0,
                refines="workload -> operator graph (one matrix multiplication)",
                rationale=(f"Start at the graph level. The workload is one multiplication, C = A x B, with A of {w['m']} x {w['k']} "
                           f"and B of {w['k']} x {w['n']} in FP32. The graph is one box; what is left to decide here is whether to "
                           "store B transposed, whether to cut K into parts that run side by side, and the number format. "
                           "The levels below decide how fast the multiplication itself runs."))
        return None
