"""The graph-level template for one matrix multiplication (level 2 of the hierarchy: workload, graph, operators, tensor map,
instructions, architecture, microarchitecture).

For a single GEMM the graph is one box, C = A x B. What the graph level still decides is how the data is laid out and whether
the one multiplication is cut into several that run side by side and are added up afterwards:

  layout_B   B as given (row by row), or stored transposed (column by column), which costs one extra pass over B
  split_k    cut the long shared dimension K into 1, 2, 4 or 8 parts, multiply each part on its own, then add the partial
             results, which costs extra memory traffic for the partial results
  precision  the number format of the multiply; the workload asks for an FP32 result, so lower formats are listed and ruled out

The levels below (operators, tiling) decide how fast the multiplication itself runs. At this level the cost is the extra
traffic each choice adds, priced with the chip's measured DRAM bandwidth, beside a lower bound for the multiplication.
"""
from __future__ import annotations

LEVEL = 2            # graph level: workload -> graph of operators
TEMPLATE = "gemm_graph"

PROGRAM = """inputs: A (M x K, row by row), B (K x N, row by row)
if layout_B == transposed:   Bt = transpose(B)                  # one extra pass over B
for s in 0 .. split_k-1:     C_s = A[:, part s of K] x B[part s of K, :]   # operator: matmul (refined at levels 3 and 4)
C = C_0 + C_1 + ... + C_{split_k-1}                              # reduction, only if split_k > 1"""

OBLIGATIONS = [
    {"name": "equivalence", "kind": "algebra", "scope": "any M, N, K and any split of K into whole parts",
     "meaning": "adding the partial products over the parts of K gives A x B (up to the order of FP32 additions)"},
    {"name": "shapes", "kind": "index", "scope": "every edge of the graph",
     "meaning": "each node gets inputs of the shape its producer makes"},
    {"name": "precision", "kind": "numeric", "scope": "the workload's accuracy",
     "meaning": "the result is computed in FP32, the format the workload asks for"},
]


def roofline_numbers(workload: dict, chip: dict) -> dict:
    """The two numbers that set the regime claim: the multiply's time at the chip's FP32 rate, and the least DRAM traffic time."""
    m, n, k = workload["m"], workload["n"], workload["k"]
    flops = 2.0 * m * n * k
    rate = float(chip.get("fp32_flops_per_s_per_sm", 0)) * int(chip.get("sms", 1))
    bw = float(chip.get("dram_bytes_per_s", 1))
    least_bytes = 4.0 * (m * k + k * n + m * n)
    return {"t_compute_us": flops / rate * 1e6 if rate else None, "t_dram_us": least_bytes / bw * 1e6, "least_bytes": least_bytes,
            "flops": flops}


def graph_contract(workload: dict, chip: dict, rationale: str = "", parent: str | None = None, formulator: str = "graph-first",
                   **provenance) -> dict:
    r = roofline_numbers(workload, chip)
    compute_bound = r["t_compute_us"] is not None and r["t_compute_us"] >= r["t_dram_us"]
    regime = (f"compute-bound: the multiply needs {r['t_compute_us']:.0f} µs at the chip's FP32 rate, the least DRAM traffic "
              f"{r['t_dram_us']:.0f} µs" if compute_bound else
              f"memory-bound: the least DRAM traffic needs {r['t_dram_us']:.0f} µs, the multiply {r['t_compute_us']:.0f} µs")
    structure = {
        "name": "bottleneck_dominance", "bucket": "objective", "regime": regime,
        "move": "price each graph choice by the extra DRAM traffic it adds; keep the multiplication whole unless a split "
                "keeps more SMs busy than its partial results cost",
        "obligation": "the multiplication stays the dominant cost for every graph choice in the menu",
        "strength": "observed", "evidence": []}
    return {
        "parent": parent, "level": LEVEL, "object": "operator graph", "template": TEMPLATE,
        "methods": ["layout", "split_k", "precision"],             # from agents/formulator/methods.yaml
        "scope": {"workload": dict(workload), "chip": chip, "structure": {"inputs": "A and B row by row, FP32", "output": "C row by row, FP32"}},
        "signature": f"{TEMPLATE}:matmul",
        "decisions": {
            "layout_B": {"domain": ["as_given", "transposed"], "kind": "software",
                         "meaning": "keep B row by row, or store it transposed (column by column) with one extra pass"},
            "split_k": {"domain": [1, 2, 4, 8], "kind": "software",
                        "meaning": "cut K into this many parts, multiply each on its own, then add the partial results"},
            "precision": {"domain": ["FP32", "TF32", "BF16"], "kind": "software", "meaning": "number format of the multiply"}},
        "constraints": [
            {"id": "G1", "expr": "precision == FP32", "meaning": "the workload asks for an FP32 result; TF32 and BF16 change the answer"},
            {"id": "G2", "expr": "K mod split_k == 0 and K / split_k >= 64", "meaning": "each part of K is whole and long enough to tile"},
            {"id": "G3", "expr": "split_k * 4 * M * N <= free memory", "meaning": "the partial results fit in memory"}],
        "cost": {"model": "T >= T_matmul + T_transpose + T_reduce", "objective": "time_us",
                 "terms": {"T_matmul": "decided by the levels below; at this level its lower bound 2·M·N·K / (FP32 rate)",
                           "T_transpose": "only if B is transposed: read and write K·N·4 bytes at the DRAM bandwidth",
                           "T_reduce": "only if split_k > 1: read split_k·M·N·4 bytes and write M·N·4 bytes at the DRAM bandwidth"},
                 "numbers": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}},
        "decomposition": None,
        "preconditions": [{"id": "equivalence", "expr": "sum_s A[:, K_s] B[K_s, :] == A B", "mode": "exact",
                           "meaning": "splitting K into parts and adding the partial products gives the same C"}],
        "program": PROGRAM,
        "backward_map": "forget the layout and the split => C = A B over (M, N, K)",
        "certificate_schema": {
            "relation": "the graph's output C equals the workload's A x B",
            "rank": "number of graph nodes not yet executed",
            "obligations": OBLIGATIONS},
        "structure": structure,
        "rationale": rationale,
        "status": "proposed",
        "certificate": None,
        "provenance": {"formulator": formulator, "model": None, "prompt_hash": None, **provenance},
    }
